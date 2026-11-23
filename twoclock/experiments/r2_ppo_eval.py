"""R2: properly-implemented PPO baselines for the wear-price problem.

Baselines (all trained on seeds >= 101, evaluated deterministically on test
seeds 1-5, 10-day episodes):

  B1  PPO-price            action = log10(mu_t) in [LOG_MU_LO, LOG_MU_HI]
                           (tanh-squashed Gaussian); dispatch = fast_dispatch
                           OPF with that wear price (identical to OPFAware).
                           Reward = -step_cost (the env reward).
  B2  Lagrangian PPO-price same action space; reward = -(step_cost + lam*c_t),
                           c_t = mean unit wear state, CMDP constraint
                           E_t[c_t] <= d, dual ascent on lam after every rollout.
  B3  End-to-end PPO       action = 5-D dispatch (gas, coal, nuclear, battery,
                           import); env enforces balance (unserved at VOLL).

PPO: clipped surrogate, GAE(lambda), separate value net, K epochs of minibatch
Adam updates per rollout, running obs normalisation, return-based reward
scaling, entropy bonus, grad-norm clipping, linear LR anneal.  Implemented in
torch (CPU, 1 thread).  Several environments are stepped sequentially in one
process so the network forward is batched.

Hyperparameters were chosen on *training/validation* seeds (201-203) only.

Run:  python -m twoclock.experiments.r2_ppo_eval            (full)
      python -m twoclock.experiments.r2_ppo_eval --quick    (smoke test)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import dataclass, asdict, field

import numpy as np
import torch
import torch.nn as nn

from twoclock.agents import OPFAware, OPFBlind, ValueGradientAgent
from twoclock.agents.baselines import _battery_bounds, action_from_opf
from twoclock.envs import make_env
from twoclock.experiments.runner import run_episode
from twoclock.paths import RESULTS_DIR
from twoclock.physics import (
    AgingParams, GEN_DISPATCHABLE, GEN_PMAX, GEN_RATING, build_5bus,
    fast_dispatch, wear_free_mw, wear_price_from_mu,
)

torch.set_num_threads(1)

TEST_SEEDS = (1, 2, 3, 4, 5)
VAL_SEEDS = (201, 202, 203)
TRAIN_SEED0 = 1001          # training episodes use seeds 1001, 1002, ... (>=101)
DAYS = 10
LOG_MU_LO, LOG_MU_HI = 0.0, 6.0     # mu in [1, 1e6] $/wear-unit (VG uses mu_max=1e6)
REF = {"normal": {"blind": 7.649, "const_mu1e4": 6.730, "vg": 6.730}}

NET = build_5bus()
AGING = AgingParams()
WEAR_FREE = wear_free_mw(AGING)


# ---------------------------------------------------------------------------
# Environment wrapper
# ---------------------------------------------------------------------------

def aug_obs(obs, env) -> np.ndarray:
    """Env features + dispatch context a policy needs (prev_p, pmax, time)."""
    ctx = obs.ctx
    extra = np.concatenate([
        ctx["prev_p"] / GEN_PMAX,
        ctx["pmax_eff"] / GEN_PMAX,
        [ctx["load"] / 1000.0, ctx["reserve_req"] / 100.0,
         env._step / env.cfg.horizon],
    ])
    return np.concatenate([obs.x.astype(float), extra])


OBS_DIM = 17 + 11


def mu_from_z(z: float) -> float:
    lm = LOG_MU_LO + (LOG_MU_HI - LOG_MU_LO) * 0.5 * (np.tanh(z) + 1.0)
    return float(10.0 ** lm)


def price_dispatch(obs, mu: float) -> np.ndarray:
    ctx = obs.ctx
    lo, hi = _battery_bounds(ctx["soc"], 0.25)
    wp = wear_price_from_mu(mu, AGING, GEN_RATING, GEN_DISPATCHABLE)
    res = fast_dispatch(
        NET, ctx["load"], ctx["prev_p"], ctx["renewable"], ctx["pmax_eff"],
        reserve_req_mw=ctx["reserve_req"], battery_min_mw=lo, battery_max_mw=hi,
        spot_price=ctx["spot"], wear_price_mwh=wp, wear_free_mw=WEAR_FREE)
    return action_from_opf(res, ctx["load"])


def direct_action(u: np.ndarray) -> np.ndarray:
    """5-D policy output in R -> 6-D env action (shed is never volunteered)."""
    u = np.clip(u, -1.0, 1.0)
    a = np.zeros(6)
    a[0:3] = 0.5 * (u[0:3] + 1.0)
    a[3] = u[3]
    a[4] = 0.5 * (u[4] + 1.0)
    return a


class Wrapped:
    def __init__(self, mode: str, regime: str, seed: int, days: int = DAYS):
        self.mode = mode
        self.env = make_env(regime, days=days, seed=seed)
        self.obs = self.env.reset()

    def x(self) -> np.ndarray:
        return aug_obs(self.obs, self.env)

    def step(self, u: np.ndarray):
        if self.mode == "price":
            mu = mu_from_z(float(u[0]))
            a = price_dispatch(self.obs, mu)
        else:
            mu = float("nan")
            a = direct_action(u)
        self.obs, r, done, info = self.env.step(a)
        info["mu"] = mu
        return r, done, info


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    mode: str = "price"            # "price" or "direct"
    regime: str = "normal"
    total_steps: int = 200_000
    n_envs: int = 4
    n_steps: int = 240             # per env per rollout -> 960 steps/update
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 10
    minibatch: int = 240
    lr: float = 3e-4
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 64
    init_log_std: float = -0.5
    lagrangian: bool = False
    wear_target: float = 0.06
    dual_lr: float = 5.0e4
    lam_init: float = 0.0
    seed: int = 0
    eval_every: int = 20_000
    eval_seeds: tuple = TEST_SEEDS
    train_seed0: int = TRAIN_SEED0
    log_mu_lo: float = LOG_MU_LO
    log_mu_hi: float = LOG_MU_HI


class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        d = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + d * bc / tot
        m2 = self.var * self.count + bv * bc + d ** 2 * self.count * bc / tot
        self.var = m2 / tot
        self.count = tot


def mlp(i, h, o, out_gain):
    net = nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(),
                        nn.Linear(h, o))
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.zeros_(m.bias)
    nn.init.orthogonal_(net[-1].weight, gain=out_gain)
    return net


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden, init_log_std):
        super().__init__()
        self.pi = mlp(obs_dim, hidden, act_dim, 0.01)
        self.v = mlp(obs_dim, hidden, 1, 1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

    def dist(self, x):
        return torch.distributions.Normal(self.pi(x), self.log_std.exp())


class PPOPolicy:
    """Frozen-normaliser policy used for evaluation (deterministic mean)."""

    def __init__(self, ac: ActorCritic, rms: RunningMeanStd, mode: str):
        self.ac = copy.deepcopy(ac)
        self.mean = rms.mean.copy()
        self.std = np.sqrt(rms.var + 1e-8)
        self.mode = mode

    def __call__(self, x: np.ndarray) -> np.ndarray:
        xn = np.clip((x - self.mean) / self.std, -10, 10)
        with torch.no_grad():
            return self.ac.pi(torch.as_tensor(xn, dtype=torch.float32)).numpy()


def evaluate_policy(policy, mode: str, regime: str, seeds=TEST_SEEDS) -> dict:
    costs, uns, wear, mus, wavg = [], [], [], [], []
    for s in seeds:
        w = Wrapped(mode, regime, s)
        ep_mu = []
        while not w.env.done:
            u = policy(w.x())
            _, _, info = w.step(u)
            ep_mu.append(info["mu"])
        summ = w.env.episode_summary()
        costs.append(summ["episode_cost"] / 1e6)
        uns.append(summ["unserved_mwh"])
        wear.append(summ["wear_end_mean"])
        wavg.append(summ["wear_integral"] / max(1, summ["steps"]) / 4.0)
        mus.append(float(np.mean(ep_mu)) if mode == "price" else float("nan"))
    return stats(costs, uns, wear, mus, wavg)


def stats(costs, uns, wear, mus, wavg=None) -> dict:
    c = np.asarray(costs)
    n = len(c)
    from scipy import stats as st
    ci = float(st.t.ppf(0.975, n - 1) * c.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    out = {"cost_mean_m": float(c.mean()), "cost_ci95_m": ci,
           "cost_per_seed_m": [float(v) for v in c],
           "unserved_mwh": float(np.mean(uns)), "wear_end_mean": float(np.mean(wear))}
    if wavg is not None:
        out["wear_episode_mean"] = float(np.mean(wavg))
    m = np.asarray(mus, dtype=float)
    if np.all(np.isfinite(m)):
        out["mean_mu"] = float(m.mean())
        out["mean_log10_mu"] = float(np.mean(np.log10(m)))
    return out


def train_ppo(cfg: PPOConfig, label: str, log=print) -> dict:
    global LOG_MU_LO, LOG_MU_HI
    LOG_MU_LO, LOG_MU_HI = cfg.log_mu_lo, cfg.log_mu_hi
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    act_dim = 1 if cfg.mode == "price" else 5
    ac = ActorCritic(OBS_DIM, act_dim, cfg.hidden, cfg.init_log_std)
    opt = torch.optim.Adam(ac.parameters(), lr=cfg.lr, eps=1e-5)
    rms = RunningMeanStd((OBS_DIM,))
    ret_rms = RunningMeanStd(())
    next_seed = [cfg.train_seed0 + 1000 * cfg.seed]

    def new_env():
        s = next_seed[0]
        next_seed[0] += 1
        return Wrapped(cfg.mode, cfg.regime, s)

    envs = [new_env() for _ in range(cfg.n_envs)]
    disc_ret = np.zeros(cfg.n_envs)
    lam_mult = cfg.lam_init
    T, N = cfg.n_steps, cfg.n_envs
    n_updates = int(np.ceil(cfg.total_steps / (T * N)))
    steps = 0
    curve = []
    t_start = time.perf_counter()
    t_eval = 0.0
    ep_costs = []

    def do_eval():
        nonlocal t_eval
        t0 = time.perf_counter()
        ev = evaluate_policy(PPOPolicy(ac, rms, cfg.mode), cfg.mode, cfg.regime,
                             cfg.eval_seeds)
        t_eval += time.perf_counter() - t0
        ev["env_steps"] = steps
        ev["train_wall_s"] = time.perf_counter() - t_start - t_eval
        ev["lambda"] = lam_mult if cfg.lagrangian else None
        ev["log_std"] = ac.log_std.detach().numpy().tolist()
        curve.append(ev)
        log(f"  [{label}] steps={steps:7d} cost={ev['cost_mean_m']:.3f}"
            f"±{ev['cost_ci95_m']:.3f} uns={ev['unserved_mwh']:.1f} "
            f"wear={ev['wear_end_mean']:.3f} mu={ev.get('mean_mu', float('nan')):.3g} "
            f"lam={lam_mult:.3g} std={np.exp(ac.log_std.detach().numpy()).round(3)} "
            f"t={ev['train_wall_s']:.0f}s")

    do_eval()
    next_eval = cfg.eval_every
    for upd in range(n_updates):
        frac = 1.0 - upd / n_updates
        for g in opt.param_groups:
            g["lr"] = cfg.lr * frac
        obs_b = np.zeros((T, N, OBS_DIM))
        act_b = np.zeros((T, N, act_dim))
        logp_b = np.zeros((T, N))
        rew_b = np.zeros((T, N))
        done_b = np.zeros((T, N))
        val_b = np.zeros((T + 1, N))
        c_b = np.zeros((T, N))
        cost_b = np.zeros((T, N))
        for t in range(T):
            raw = np.stack([e.x() for e in envs])
            rms.update(raw)
            xn = np.clip((raw - rms.mean) / np.sqrt(rms.var + 1e-8), -10, 10)
            with torch.no_grad():
                xt = torch.as_tensor(xn, dtype=torch.float32)
                d = ac.dist(xt)
                a = d.sample()
                lp = d.log_prob(a).sum(-1)
                v = ac.v(xt).squeeze(-1)
            obs_b[t], act_b[t], logp_b[t], val_b[t] = xn, a.numpy(), lp.numpy(), v.numpy()
            for i, e in enumerate(envs):
                _, done, info = e.step(a[i].numpy())
                c_t = float(np.mean(info["wear"]))
                cost = info["step_cost"]
                pen = cost + (lam_mult * c_t if cfg.lagrangian else 0.0)
                rew_b[t, i] = -pen / 1e4
                c_b[t, i] = c_t
                cost_b[t, i] = cost
                done_b[t, i] = float(done)
                if done:
                    ep_costs.append(e.env.episode_summary()["episode_cost"] / 1e6)
                    envs[i] = new_env()
            # return-based reward scaling (as in standard PPO implementations)
            disc_ret = disc_ret * cfg.gamma + rew_b[t]
            ret_rms.update(disc_ret.reshape(-1))
            disc_ret = disc_ret * (1.0 - done_b[t])
        steps += T * N
        raw = np.stack([e.x() for e in envs])
        xn = np.clip((raw - rms.mean) / np.sqrt(rms.var + 1e-8), -10, 10)
        with torch.no_grad():
            val_b[T] = ac.v(torch.as_tensor(xn, dtype=torch.float32)).squeeze(-1).numpy()
        rew_n = rew_b / np.sqrt(ret_rms.var + 1e-8)
        # GAE (value net predicts normalised-reward returns)
        adv = np.zeros((T, N))
        last = np.zeros(N)
        for t in reversed(range(T)):
            nonterm = 1.0 - done_b[t]
            delta = rew_n[t] + cfg.gamma * val_b[t + 1] * nonterm - val_b[t]
            last = delta + cfg.gamma * cfg.lam * nonterm * last
            adv[t] = last
        ret = adv + val_b[:T]

        B = T * N
        ob = torch.as_tensor(obs_b.reshape(B, -1), dtype=torch.float32)
        ac_t = torch.as_tensor(act_b.reshape(B, -1), dtype=torch.float32)
        lp_old = torch.as_tensor(logp_b.reshape(B), dtype=torch.float32)
        adv_t = torch.as_tensor(adv.reshape(B), dtype=torch.float32)
        ret_t = torch.as_tensor(ret.reshape(B), dtype=torch.float32)
        for _ in range(cfg.epochs):
            perm = torch.as_tensor(rng.permutation(B))
            for s in range(0, B, cfg.minibatch):
                idx = perm[s:s + cfg.minibatch]
                d = ac.dist(ob[idx])
                lp = d.log_prob(ac_t[idx]).sum(-1)
                ratio = torch.exp(lp - lp_old[idx])
                A = adv_t[idx]
                A = (A - A.mean()) / (A.std() + 1e-8)
                pl = -torch.min(ratio * A,
                                torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * A).mean()
                vl = 0.5 * ((ac.v(ob[idx]).squeeze(-1) - ret_t[idx]) ** 2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pl + cfg.vf_coef * vl - cfg.ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
                opt.step()
        if cfg.lagrangian:
            lam_mult = max(0.0, lam_mult + cfg.dual_lr * (float(c_b.mean()) - cfg.wear_target))
        if steps >= next_eval or upd == n_updates - 1:
            do_eval()
            next_eval += cfg.eval_every
    wall = time.perf_counter() - t_start - t_eval
    return {"label": label, "config": asdict(cfg), "curve": curve,
            "final": curve[-1], "train_wall_s": wall,
            "train_episode_costs_m": ep_costs, "final_policy": PPOPolicy(ac, rms, cfg.mode)}


# ---------------------------------------------------------------------------
# References (recomputed so every number in the JSON comes from this code)
# ---------------------------------------------------------------------------

def reference_rows(regime: str) -> dict:
    out = {}
    for name, mk in [("blind_opf", lambda: OPFBlind(NET, AGING)),
                     ("const_mu1e3", lambda: OPFAware(NET, AGING, 1e3)),
                     ("const_mu1e4", lambda: OPFAware(NET, AGING, 1e4))]:
        rows = [run_episode(make_env(regime, days=DAYS, seed=s), mk()) for s in TEST_SEEDS]
        out[name] = stats([r["episode_cost"] / 1e6 for r in rows],
                          [r["unserved_mwh"] for r in rows],
                          [r["wear_end_mean"] for r in rows], [np.nan] * 5,
                          [r["wear_integral"] / r["steps"] / 4.0 for r in rows])
    # value-gradient, exact recipe of rigorous_suite (4,320 steps)
    t0 = time.perf_counter()
    vg = ValueGradientAgent(NET, AGING, seed=7, mu_max=1e6)
    vg.enabled = False
    st_ = 0
    for r in range(3):
        for s in (101, 102, 103):
            ep = run_episode(make_env(regime, days=5, seed=s + 10 * r), vg, learn=True)
            st_ += int(ep["steps"])
        vg.update()
        vg.enabled = True
    tw = time.perf_counter() - t0
    rows = [run_episode(make_env(regime, days=DAYS, seed=s), vg) for s in TEST_SEEDS]
    out["value_gradient"] = stats([r["episode_cost"] / 1e6 for r in rows],
                                  [r["unserved_mwh"] for r in rows],
                                  [r["wear_end_mean"] for r in rows], [np.nan] * 5,
                          [r["wear_integral"] / r["steps"] / 4.0 for r in rows])
    out["value_gradient"]["train_steps"] = st_
    out["value_gradient"]["train_wall_s"] = tw
    return out


def steps_to_within(curve: list, target: float, tol: float = 0.01):
    for ev in curve:
        if ev["cost_mean_m"] <= target * (1 + tol):
            return ev["env_steps"]
    return None


def save(payload: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=lambda o: (
            o.item() if hasattr(o, "item") else str(o)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="", help="comma list of run labels")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "r2_ppo.json"))
    ap.add_argument("--b1-steps", type=int, default=150_000)
    ap.add_argument("--b2-steps", type=int, default=150_000)
    ap.add_argument("--b3-steps", type=int, default=300_000)
    ap.add_argument("--peak-steps", type=int, default=60_000)
    ap.add_argument("--train-seeds", type=int, default=3)
    args = ap.parse_args()
    q = args.quick
    k = 0.02 if q else 1.0

    payload = {"description": __doc__, "test_seeds": list(TEST_SEEDS), "days": DAYS,
               "log10_mu_range": [LOG_MU_LO, LOG_MU_HI], "runs": {}, "references": {}}
    if os.path.exists(args.out) and not q:
        with open(args.out, encoding="utf-8") as fh:
            old = json.load(fh)
        payload["runs"].update(old.get("runs", {}))
        payload["references"].update(old.get("references", {}))

    regimes = ["normal", "peak"]
    for reg in regimes:
        if reg not in payload["references"]:
            print(f"== references ({reg})")
            payload["references"][reg] = reference_rows(reg)
            print(json.dumps({kk: round(v["cost_mean_m"], 3)
                              for kk, v in payload["references"][reg].items()}))
            save(payload, args.out)

    runs = []
    ev = lambda n: max(960, int(n * k))
    for s in range(args.train_seeds):
        runs.append((f"B1_ppo_price_s{s}", PPOConfig(
            mode="price", total_steps=int(args.b1_steps * k), seed=s, eval_every=ev(25_000))))
    for d in (0.06, 0.01):
        runs.append((f"B2_lagppo_price_d{d}_s0", PPOConfig(
            mode="price", total_steps=int(args.b2_steps * k), seed=0, lagrangian=True,
            wear_target=d, eval_every=ev(25_000))))
    runs.append(("B3_ppo_direct_s0", PPOConfig(
        mode="direct", total_steps=int(args.b3_steps * k), seed=0, eval_every=ev(50_000))))
    runs.append(("B1_ppo_price_peak_s0", PPOConfig(
        mode="price", regime="peak", total_steps=int(args.peak_steps * k), seed=0,
        eval_every=ev(20_000))))
    only = [o for o in args.only.split(",") if o]

    for label, cfg in runs:
        if only and label not in only:
            continue
        if label in payload["runs"] and not q:
            print(f"skip {label} (done)")
            continue
        print(f"== {label}")
        res = train_ppo(cfg, label)
        res.pop("final_policy")
        ref = payload["references"][cfg.regime]["const_mu1e4"]["cost_mean_m"]
        res["steps_to_within_1pct_of_const_mu1e4"] = steps_to_within(res["curve"], ref)
        payload["runs"][label] = res
        save(payload, args.out)

    print("\n| run | steps | cost M$ ±CI | unserved MWh | wear | mean mu | wall s | steps to 1% |")
    for label, r in payload["runs"].items():
        f = r["final"]
        print(f"| {label} | {f['env_steps']} | {f['cost_mean_m']:.3f}±{f['cost_ci95_m']:.3f} "
              f"| {f['unserved_mwh']:.1f} | {f['wear_end_mean']:.3f} "
              f"| {f.get('mean_mu', float('nan')):.3g} | {r['train_wall_s']:.0f} "
              f"| {r['steps_to_within_1pct_of_const_mu1e4']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
