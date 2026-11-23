"""
Grid ACPL Agent — grid_acpl_agent.py

Continuous-action ACPL agent tuned for the energy grid.
Integrates all v6 upgrades:
  - GRU encoder for temporal memory (price cycles, stress history)
  - Multi-horizon consequence net (immediate stress, 4h degradation, billing)
  - State-conditioned lambda (near-capacity states get higher penalty)
  - Delay estimator learning P(τ|h) for stress signal timing
  - Sparse consequence store for irregular billing feedback
  - Hit-frequency EMA for unconverged hit-rate fix
"""
import math
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from networks.grid_networks import (
    GRUPolicyNet, MultiHorizonConsequenceNet, LambdaNet,
    DelayEstimatorNet, GaussianActorGRU, CriticGRU,
    gru_batch_forward, sigmoid, Adam
)
from training.replay_buffer import GridReplayBuffer
from utils.normalizer import RunningNormalizer


class GridACPLAgent:
    """
    Continuous-action ACPL for energy grid load balancing.

    Observation: 18-dim grid state
    Action:      5-dim continuous in [-1, 1]
                 [gas_delta, coal_delta, market, load_shed, battery]

    Key design choices:
      - GRU hidden state captures 24h price/demand cycles
      - Lambda network uses [load_norm, stress, reserve_margin] as inputs
        (indices 0, 3, 4) — the three most consequence-relevant features
      - Delay estimator range: 0–20 steps (0–5 hours)
      - Hit-frequency EMA with alpha=0.03 (slow — grid runs for days)
    """
    name = "GridACPL"

    def __init__(
        self,
        state_dim            = 18,
        action_dim           = 5,
        gru_dim              = 64,
        hidden_dim           = 128,
        n_layers             = 2,
        consequence_dim      = 64,
        consequence_layers   = 2,
        lambda_hidden_dim    = 32,
        tau_max              = 20,     # 5 hours at 15-min steps
        gamma                = 0.995,  # long horizon — grid planning
        gae_lambda           = 0.97,
        tau_soft             = 0.005,
        lr_actor             = 8e-4,
        lr_critic            = 3e-3,
        lr_consequence       = 1e-3,
        lr_lambda            = 5e-4,
        lr_delay             = 1e-3,
        sigma_weight         = 0.3,
        lambda_max           = 2.5,
        lambda_warmup        = 60,
        lambda_weight_decay  = 5e-5,
        hit_freq_weight      = 0.10,
        clip_eps             = 0.2,
        ent_coef             = 0.005,
        gov_coef             = 0.03,
        use_consequence      = True,
        use_causal_attr      = True,
        fixed_lambda         = None,
        name                 = None,
        batch_size           = 128,
        buffer_capacity      = 100_000,
        n_steps              = 96,     # 1 day rollout
        n_epochs             = 4,
        seed                 = 42,
    ):
        if name is not None:
            self.name = name
        self.state_dim     = state_dim
        self.action_dim    = action_dim
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.sigma_weight  = sigma_weight
        self.batch_size    = batch_size
        self.n_steps       = n_steps
        self.n_epochs      = n_epochs
        self.lambda_max    = lambda_max
        self.lambda_warmup = lambda_warmup
        self.lambda_wd     = lambda_weight_decay
        self.hit_freq_weight = hit_freq_weight
        self.clip_eps      = clip_eps
        self.ent_coef      = ent_coef
        self.gov_coef      = gov_coef
        self.use_consequence = use_consequence
        self.use_causal_attr = use_causal_attr
        self.fixed_lambda  = fixed_lambda
        self.steps_done    = 0
        self.episodes_done = 0
        self.update_count  = 0
        self.rng           = np.random.default_rng(seed)

        # Actor-critic
        self.actor  = GaussianActorGRU(state_dim, action_dim, gru_dim,
                                        hidden_dim, n_layers, lr_actor, seed)
        self.critic = CriticGRU(state_dim, gru_dim, hidden_dim,
                                 n_layers, lr_critic, seed)

        # Consequence + lambda + delay
        self.consequence_net = MultiHorizonConsequenceNet(
            state_dim, action_dim, consequence_dim, consequence_layers,
            lr_consequence, seed)
        self.lambda_net = LambdaNet(
            state_dim, hidden_dim=lambda_hidden_dim,
            lambda_max=lambda_max, lr=lr_lambda,
            input_idx=[0, 3, 4],  # load, stress, reserve
            seed=seed)
        self.delay_net = DelayEstimatorNet(
            gru_dim=gru_dim, hidden_dim=32, tau_max=tau_max,
            lr=lr_delay, seed=seed)

        # Buffer + normalizer
        self.buffer     = GridReplayBuffer(action_dim, buffer_capacity, seed)
        self.normalizer = RunningNormalizer(state_dim)

        # Hidden state
        self._h_actor  = self.actor.zero_state(1)
        self._h_critic = self.critic.zero_state(1)

        # On-policy rollout
        self._rollout: list = []
        self._last_raw_action = np.zeros(action_dim, np.float32)
        self._last_log_prob   = 0.0

        # Hit frequency EMA (grid runs slowly — use alpha=0.03)
        self._hit_freq_ema   = 0.3
        self._hit_freq_alpha = 0.03

        # Diagnostics
        self.last_policy_loss      = 0.0
        self.last_value_loss       = 0.0
        self.last_consequence_loss = 0.0
        self.last_delay_loss       = 0.0
        self.last_mean_lambda      = 0.5
        self.last_mean_C           = 0.0
        self.last_expected_delay   = 5.0

        # Theory logs
        self._lambda_log:  list = []
        self._C_log:       list = []
        self._delay_log:   list = []

    # ── Warmup ────────────────────────────────────────────────────────────────

    @property
    def lambda_scale(self) -> float:
        return min(1.0, self.episodes_done / max(self.lambda_warmup, 1))

    def reset_hidden(self):
        self._h_actor  = self.actor.zero_state(1)
        self._h_critic = self.critic.zero_state(1)

    def episode_end(self, hit_occurred: bool = False):
        self.episodes_done += 1
        self._hit_freq_ema = (self._hit_freq_alpha * float(hit_occurred)
                              + (1 - self._hit_freq_alpha) * self._hit_freq_ema)

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, state, eval_mode=False):
        if not eval_mode:
            self.steps_done += 1

        s_norm = self.normalizer.normalize(state)[None]

        h_before = self._h_actor.copy()
        action, log_prob, h_new, raw = self.actor.sample(
            s_norm, self._h_actor, eval_mode=eval_mode)
        self._h_actor = h_new
        self._last_raw_action = raw
        self._last_log_prob   = log_prob

        # Log delay estimate
        exp_d = float(self.delay_net.expected_delay(h_before.squeeze()))
        self.last_expected_delay = exp_d
        self._delay_log.append(exp_d)
        if len(self._delay_log) > 5000: self._delay_log.pop(0)

        # Log lambda (use raw value for logging, scale applied in Q computation)
        if not self.use_consequence:
            lam = 0.0
        elif self.fixed_lambda is not None:
            lam = float(self.fixed_lambda) * self.lambda_scale
        else:
            lam_raw = float(self.lambda_net.forward(s_norm.squeeze()))
            lam = lam_raw * self.lambda_scale
        self.last_mean_lambda = lam
        self._lambda_log.append(lam)
        if len(self._lambda_log) > 5000: self._lambda_log.pop(0)

        return action

    # ── Store ─────────────────────────────────────────────────────────────────

    def store(self, state, action, reward, next_state, consequence, done,
              hidden=None, next_hidden=None, obs_delay=-1):
        self.normalizer.update(state)
        s_norm  = self.normalizer.normalize(state)
        ns_norm = self.normalizer.normalize(next_state)

        h  = (hidden       if hidden       is not None else self._h_actor).flatten()
        nh = (next_hidden  if next_hidden  is not None else self._h_actor).flatten()

        # Off-policy buffer
        self.buffer.push(s_norm, action, reward, ns_norm, consequence, done, h, nh,
                         obs_delay=obs_delay)

        # On-policy rollout
        v = self.critic.forward(s_norm[None], h[None]).item()
        self._rollout.append({
            "s": s_norm, "a": np.asarray(action, np.float32),
            "raw": self._last_raw_action.copy(), "lp": float(self._last_log_prob),
            "r": float(reward),
            "ns": ns_norm, "c": float(consequence), "d": float(done),
            "h": h, "nh": nh, "v": float(v),
        })

        self._C_log.append(float(consequence))
        if len(self._C_log) > 5000: self._C_log.pop(0)

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self):
        if len(self._rollout) < self.n_steps:
            return False

        traj   = self._rollout[-self.n_steps:]
        self._rollout.clear()
        S      = np.stack([t["s"]   for t in traj])
        NS     = np.stack([t["ns"]  for t in traj])
        A      = np.stack([t["a"]   for t in traj])
        RAW    = np.stack([t["raw"] for t in traj])
        OLD_LP = np.array([t["lp"]  for t in traj], np.float32)
        R      = np.array([t["r"]   for t in traj], np.float32)
        C      = np.array([t["c"]   for t in traj], np.float32)
        D      = np.array([t["d"]   for t in traj], np.float32)
        H      = np.stack([t["h"]   for t in traj])
        NH     = np.stack([t["nh"]  for t in traj])
        V      = np.array([t["v"]   for t in traj], np.float32)
        B      = len(traj)

        if self.use_consequence:
            # 0. Update delay net first so P(tau|h) reflects latest observed delays
            batch = self.buffer.sample(min(64, len(self.buffer)))
            if batch is not None:
                obs_tau = batch["obs_delays"]
                if np.any(obs_tau >= 0):
                    self.last_delay_loss = self.delay_net.update(
                        batch["hiddens"], obs_tau,
                        weights=np.ones(len(obs_tau), np.float32))

            # 1. Causal consequence back-attribution using P(tau|h)
            if self.use_causal_attr:
                P_tau = self.delay_net.forward(H)  # (B, tau_max+1)
                C_target = np.zeros(B, np.float32)
                tau_max = P_tau.shape[1] - 1
                for t in range(B):
                    max_k = min(tau_max, B - 1 - t)
                    w_t = P_tau[t, :max_k + 1]
                    mass = float(w_t.sum())
                    if mass > 1e-4:
                        C_target[t] = float(np.dot(w_t, C[t:t + max_k + 1]) / mass)
                    else:
                        C_target[t] = C[t]
            else:
                C_target = C

            # 2. Train multi-horizon consequence net on continuous actions (S, A) -> C_target
            c_losses = []
            for _ in range(self.n_epochs):
                perm_c = self.rng.permutation(B)
                for start in range(0, B, 32):
                    mb = perm_c[start:start+32]
                    c_losses.append(self.consequence_net.update_step(
                        S[mb], A[mb], C_target[mb], np.ones(len(mb), np.float32)))
            self.last_consequence_loss = float(np.mean(c_losses)) if c_losses else 0.0
            C_pred, sigma_pred = self.consequence_net.predict(S, A)

            # 3. State-conditioned Lambda
            if self.fixed_lambda is not None:
                lam_raw = np.full(B, float(self.fixed_lambda), np.float32)
            else:
                lam_raw = self.lambda_net.forward(S)
            lam_batch = lam_raw * self.lambda_scale

            # 4. Consequence-corrected rewards
            C_eff  = 0.5 * C_target + 0.5 * C_pred if self.use_causal_attr else C_pred
            pen    = np.clip(lam_batch * (C_eff + self.sigma_weight * sigma_pred),
                             0.0, 4.0)
            R_corr = R - pen
        else:
            C_target = C
            C_pred = np.zeros(B, np.float32)
            self.last_consequence_loss = 0.0
            R_corr = R

        # 5. GAE advantages
        NV      = self.critic.forward(NS, NH)
        deltas  = R_corr + self.gamma * NV * (1 - D) - V
        adv     = np.zeros(B, np.float32)
        gae     = 0.0
        for t in reversed(range(B)):
            gae    = deltas[t] + self.gamma * self.gae_lambda * (1 - D[t]) * gae
            adv[t] = gae
        returns  = adv + V
        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)

        # 6. Critic update (minibatch epochs for accurate value baseline)
        v_losses = []
        for _ in range(self.n_epochs):
            perm_v = self.rng.permutation(B)
            for start in range(0, B, 32):
                mb = perm_v[start:start+32]
                v_losses.append(self.critic.backward_update(
                    S[mb], H[mb], returns[mb], np.ones(len(mb), np.float32)))
        self.last_value_loss = float(np.mean(v_losses)) if v_losses else 0.0

        # 7. Policy update (PPO-clipped + differentiable consequence-action governance)
        p_losses = []
        for _ in range(self.n_epochs):
            perm = self.rng.permutation(B)
            for start in range(0, B, 32):
                mb     = perm[start:start+32]
                gov_grad = None
                if self.use_consequence and self.gov_coef > 0 and self.lambda_scale > 0:
                    mean_mb, _, _, _ = self.actor.forward_batch(S[mb], H[mb])
                    a_det = np.tanh(np.clip(mean_mb, -20.0, 20.0))
                    dC_da = self.consequence_net.action_gradient(S[mb], a_det)
                    if self.fixed_lambda is not None:
                        lam_mb = np.full(len(mb), float(self.fixed_lambda), np.float32)
                    else:
                        lam_mb = self.lambda_net.forward(S[mb])
                    lam_mb = lam_mb * self.lambda_scale
                    gov_grad = (self.gov_coef * lam_mb[:, None]
                                * dC_da * (1.0 - a_det**2))
                    gov_grad = np.clip(gov_grad, -1.0, 1.0)
                p_loss = self.actor.backward_update(
                    S[mb], H[mb], RAW[mb], OLD_LP[mb], adv_norm[mb],
                    clip_eps=self.clip_eps, ent_coef=self.ent_coef,
                    gov_grad_mean=gov_grad)
                p_losses.append(p_loss)
        self.last_policy_loss = float(np.mean(p_losses)) if p_losses else 0.0

        # 8. Lambda update (unscaled lam_raw supervised by causal consequence + hit-freq)
        if self.use_consequence and self.fixed_lambda is None:
            lam_raw_now = self.lambda_net.forward(S)
            mag_target  = sigmoid(2.0 * (C_target - 1.0))
            freq_target = np.full(B, self._hit_freq_ema, np.float32)
            w           = self.hit_freq_weight
            lam_target  = (1 - w) * mag_target + w * freq_target
            lam_norm    = lam_raw_now / self.lambda_max
            lam_errors  = lam_norm - lam_target
            self.lambda_net.backward_update(S, lam_errors, weight_decay=self.lambda_wd)

        self.last_mean_C = float(C_pred.mean())
        self.update_count += 1
        return True

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def diagnostics(self) -> dict:
        return {
            "steps":            self.steps_done,
            "episodes":         self.episodes_done,
            "updates":          self.update_count,
            "lambda_scale":     round(self.lambda_scale, 4),
            "hit_freq_ema":     round(self._hit_freq_ema, 4),
            "mean_lambda":      round(self.last_mean_lambda, 4),
            "mean_C":           round(self.last_mean_C, 4),
            "expected_delay":   round(self.last_expected_delay, 2),
            "policy_loss":      round(self.last_policy_loss, 4),
            "value_loss":       round(self.last_value_loss, 4),
            "consequence_loss": round(self.last_consequence_loss, 5),
            "delay_loss":       round(self.last_delay_loss, 5),
            "buffer_size":      len(self.buffer),
        }

    def get_theory_logs(self) -> dict:
        return {
            "lambda_log":      list(self._lambda_log),
            "C_log":           list(self._C_log),
            "expected_delays": list(self._delay_log),
            "observed_delays": [],
        }
