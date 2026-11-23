"""Value-gradient dispatch: the price *is* the policy.

No policy network.  We learn ``V(s)`` -- the expected *remaining dollar cost*
from state ``s`` -- and read the price of the slow constraint off its gradient:

    mu_g(s) = dV / d wear_g          [$/wear-unit]

Because thermal aging is piecewise linear in MW, that price becomes a $/MWh
coefficient and the fast-clock dispatch stays an LP:

    dispatch = argmin  fuel(u) + sum_g  mu_g * d(wear_g)/du * u_g

Training is approximate policy iteration:

  1. evaluate  -- fit V by Monte-Carlo regression on the running policy
  2. improve   -- set the price to the value gradient and re-dispatch

Because V is in **dollars** and wear is dimensionless, ``mu`` is already in the
units the OPF needs: there is no reward scale, no Lagrange multiplier and no
dimensionless penalty weight to tune.  The price is anchored to a physical
marginal cost, which a CMDP dual-ascent penalty weight does not have.

The default value model is ridge regression on interpretable features.  It is
closed-form, deterministic and -- because wear enters linearly -- its gradient
is exactly a price.  A tanh MLP variant is kept as an ablation: if a
state-conditioned price from a nonlinear V is no better than a single scalar,
that is itself the reduction result.
"""
from __future__ import annotations

import numpy as np

from .. import ENV_VERSION
from ..physics import (
    BATTERY_POWER_MW, GEN_DISPATCHABLE, GEN_RATING,
    dc_opf, fast_dispatch, wear_price_from_mu, wear_free_mw,
)
from .baselines import BaseAgent, action_from_opf, _battery_bounds
from .networks import MLP, Adam

WEAR_IDX = (3, 4, 5, 6)
N_WEAR = len(WEAR_IDX)


# ---------------------------------------------------------------------------
# Value models
# ---------------------------------------------------------------------------

def wear_index(x: np.ndarray) -> float:
    """Damage index: total wear of the three thermal units.

    One number is enough because the price converts back to per-unit $/MWh
    through each unit's marginal aging rate.  Using a single index is what
    makes the learned object a *scalar price* rather than a price vector.

    The index is a **sum**, not a mean, so ``dV/dwear_g = dV/dW`` exactly and
    the price needs no rescaling before it enters the OPF.
    """
    return float(np.sum(np.asarray(x, dtype=float).ravel()[3:6]))


def value_features(x: np.ndarray, step_frac: float = 0.0,
                   horizon_frac: float = 0.0, per_unit: bool = False) -> np.ndarray:
    """Interpretable features for the cost-to-go.

    Scalar mode (default)::

        [1, W, W**2, eff_mean, load, renew, spot, soc, sin_h, cos_h,
         t, t**2, t * W]         W = total thermal wear

    ``W**2`` is what makes the price *state dependent*: dV/dW then falls as
    wear grows, so a nearly-new fleet is charged little and a worn one is
    charged a lot.  ``t * W`` says wear incurred late in the horizon is
    cheaper.  ``per_unit`` mode keeps four wear channels separate, as an
    ablation.
    """
    x = np.asarray(x, dtype=float).ravel()
    w = wear_index(x)
    eff = float(np.mean(x[7:11]))
    base = [1.0, w, w ** 2, eff,
            float(x[0]), float(x[1]), float(x[2]), float(x[11]),
            float(x[12]), float(x[13]),
            float(step_frac), float(step_frac ** 2)]
    if per_unit:
        return np.concatenate([base, x[list(WEAR_IDX)], step_frac * x[list(WEAR_IDX)]])
    return np.concatenate([base, [step_frac * w]])


FEATURE_DIM = len(value_features(np.zeros(17)))
FEATURE_DIM_PER_UNIT = len(value_features(np.zeros(17), per_unit=True))


class RidgeValue:
    """Closed-form ridge regression with a wear-gradient readout."""

    def __init__(self, dim: int = FEATURE_DIM, l2: float = 10.0,
                 per_unit: bool = False):
        self.dim = dim
        self.l2 = l2
        self.per_unit = per_unit
        self.w = np.zeros(dim)
        self.fitted = False
        self.mu = np.zeros(N_WEAR)      # $/wear-unit, per unit
        self._loss = float("nan")

    def fit(self, Phi: np.ndarray, y: np.ndarray) -> None:
        if len(Phi) < self.dim * 2:
            return
        mean = Phi.mean(0)
        sd = Phi.std(0)
        sd[sd < 1e-9] = 1.0
        Z = (Phi - mean) / sd
        A = Z.T @ Z + self.l2 * np.eye(self.dim)
        b = Z.T @ y
        wz = np.linalg.solve(A, b)
        w = wz / sd
        self.w = w
        self.w[0] = float(y.mean() - wz @ (mean / sd))
        pred = Phi @ self.w
        self._loss = float(np.mean((pred - y) ** 2))
        self.mu = self.mu_mean.copy()
        self.fitted = True

    def mu_at(self, x: np.ndarray, step_frac: float = 0.0) -> np.ndarray:
        """Shadow price vector at a state, $/wear-unit."""
        w = wear_index(x)
        if self.per_unit:
            base = 12
            m = np.array([self.w[base + k] + step_frac * self.w[base + N_WEAR + k]
                          for k in range(N_WEAR)])
        else:
            m = np.full(N_WEAR, self.w[1] + 2.0 * self.w[2] * w
                        + step_frac * self.w[-1])
        return np.maximum(m, 0.0)

    @property
    def mu_mean(self) -> np.ndarray:
        """Price at the mean observed state (for reporting only)."""
        return np.maximum(self.w[1:1 + N_WEAR] if self.per_unit
                          else np.full(N_WEAR, self.w[1]), 0.0)

    def predict(self, Phi: np.ndarray) -> np.ndarray:
        return Phi @ self.w

    @property
    def loss(self) -> float:
        return self._loss


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ValueGradientAgent(BaseAgent):
    """Dispatch at the value gradient of the slow constraint."""

    name = "vg"

    def __init__(self, net, aging_params, state_dim: int = 17,
                 hidden: int = 64, lr: float = 3e-3, delta: float = 0.02,
                 price_clip: float = 400.0, mu_scale: float = 1.0,
                 mu_max: float = 2.0e5, seed: int = 0, enabled: bool = True,
                 fast: bool = True, model: str = "ridge", l2: float = 10.0,
                 per_unit: bool = False, max_samples: int = 6000):
        self.net = net
        self.aging = aging_params
        self.state_dim = state_dim
        self.model = model
        self.per_unit = per_unit
        self.mu_scale = mu_scale
        self.mu_max = mu_max
        self.price_clip = price_clip
        self.delta = delta
        self.enabled = enabled
        self.fast = fast
        self.rng = np.random.default_rng(seed)
        dim = FEATURE_DIM_PER_UNIT if per_unit else FEATURE_DIM

        if model == "mlp":
            self.value = MLP([dim, hidden, hidden, 1], seed=seed, out_scale=0.1)
            self.optim = Adam(self.value.params, lr=lr, clip=2.0)
        else:
            self.value = RidgeValue(dim, l2=l2, per_unit=per_unit)

        self.mu = np.zeros(N_WEAR)
        self.mu_log: list[np.ndarray] = []
        self.price_log: list[np.ndarray] = []
        self.loss_log: list[float] = []
        self.dataset: list[tuple[np.ndarray, np.ndarray, float]] = []
        self.max_samples = max_samples
        self._ep: list[tuple[np.ndarray, float, float]] = []
        self.outer = 0
        self.horizon = 1
        self._trained = False

    # -------------------------------------------------------------- lifecycle
    def reset(self, env=None) -> None:
        if env is not None:
            self.horizon = max(1, getattr(env.cfg, "horizon", 1))

    def begin_episode(self) -> None:
        self._ep = []

    def record(self, x: np.ndarray, step_cost: float, reward: float | None = None) -> None:
        """Store one visited state and the realised cost of that step."""
        step = 0.0
        if self._ep:
            step = self._ep[-1][2] + 1.0
        else:
            step = 0.0
        self._ep.append((np.asarray(x, dtype=float).copy(), float(step_cost), step))

    def end_episode(self) -> None:
        """Turn the episode into (features, cost-to-go) pairs."""
        if not self._ep:
            return
        costs = np.array([c for _, c, _ in self._ep])
        # cost-to-go, computed backwards
        g = np.cumsum(costs[::-1])[::-1]
        H = max(1.0, float(self.horizon))
        for (x, _, k), y in zip(self._ep, g):
            phi = value_features(x, step_frac=k / H, per_unit=self.per_unit)
            self.dataset.append((phi, x, float(y)))
        self._ep = []
        if len(self.dataset) > self.max_samples:
            self.dataset = self.dataset[-self.max_samples:]

    # ------------------------------------------------------------------ price
    def update(self) -> dict:
        """Refit V and read off the new price."""
        if len(self.dataset) < 200:
            return self.diagnostics()
        Phi = np.stack([d[0] for d in self.dataset])
        y = np.array([d[2] for d in self.dataset])
        if self.model == "mlp":
            for _ in range(3):
                perm = self.rng.permutation(len(Phi))
                for s in range(0, len(Phi), 256):
                    idx = perm[s:s + 256]
                    pred = self.value.forward(Phi[idx]).ravel()
                    err = pred - y[idx]
                    _, grads = self.value.backward((2.0 * err / len(err))[:, None])
                    self.optim.step(grads)
            xs = np.stack([d[1] for d in self.dataset[-4000:]])
            self.mu = self._finite_difference_mu(xs)
            self.loss_log.append(float("nan"))
        else:
            self.value.fit(Phi, y)
            self.mu = self.value.mu.copy()
            pred = self.value.predict(Phi)
            self.loss_log.append(float(np.mean((pred - y) ** 2)))
        self.mu = np.clip(self.mu * self.mu_scale, 0.0, self.mu_max)
        self._trained = True
        self.outer += 1
        return self.diagnostics()

    def _finite_difference_mu(self, xs: np.ndarray) -> np.ndarray:
        out = np.zeros(N_WEAR)
        idx = self.rng.choice(len(xs), size=min(400, len(xs)), replace=False)
        for k, wi in enumerate(WEAR_IDX):
            xp = xs[idx].copy()
            xm = xs[idx].copy()
            xp[:, wi] += self.delta
            xm[:, wi] -= self.delta
            Ph = np.stack([value_features(x, per_unit=self.per_unit) for x in xp])
            Phm = np.stack([value_features(x, per_unit=self.per_unit) for x in xm])
            gp = self.value.forward(Ph).ravel()
            gm = self.value.forward(Phm).ravel()
            out[k] = float(np.mean((gp - gm) / (2.0 * self.delta)))
        out = np.maximum(out, 0.0)
        return out if self.per_unit else np.full(N_WEAR, float(np.mean(out[:3])))

    def mu_at(self, x: np.ndarray) -> np.ndarray:
        """Shadow price at one state, $/wear-unit."""
        if not getattr(self, "_trained", False):
            return np.zeros(N_WEAR)
        if self.model == "ridge":
            return self.value.mu_at(x)
        return self._finite_difference_mu(np.asarray(x, dtype=float)[None, :])

    def wear_price(self, x: np.ndarray) -> np.ndarray:
        mu = self.mu_at(x) if self.enabled else np.zeros(N_WEAR)
        wp = wear_price_from_mu(mu, self.aging, GEN_RATING, GEN_DISPATCHABLE)
        return np.clip(wp, 0.0, self.price_clip)

    @property
    def mu_scalar(self) -> float:
        return float(np.mean(self.mu[:3])) if np.size(self.mu) else 0.0

    # -------------------------------------------------------------------- act
    def act(self, obs) -> np.ndarray:
        ctx = obs.ctx
        mu = self.mu_at(obs.x) if self.enabled else np.zeros(N_WEAR)
        wp = None
        if self.enabled:
            wp = np.clip(wear_price_from_mu(mu, self.aging, GEN_RATING,
                                            GEN_DISPATCHABLE),
                         0.0, self.price_clip)
        lo, hi = _battery_bounds(ctx["soc"], 0.25)
        solve = fast_dispatch if self.fast else dc_opf
        res = solve(
            self.net, ctx["load"], ctx["prev_p"], ctx["renewable"], ctx["pmax_eff"],
            reserve_req_mw=ctx["reserve_req"],
            battery_min_mw=lo, battery_max_mw=hi,
            spot_price=ctx["spot"], wear_price_mwh=wp,
            wear_free_mw=wear_free_mw(self.aging),
        )
        self.mu = np.asarray(mu, dtype=float).copy()
        self.mu_log.append(self.mu.copy())
        self.price_log.append(np.zeros(4) if wp is None else wp.copy())
        if len(self.mu_log) > 100_000:
            self.mu_log.pop(0)
            self.price_log.pop(0)
        return action_from_opf(res, ctx["load"])

    # ------------------------------------------------------------ diagnostics
    def diagnostics(self) -> dict:
        mu = np.array(self.mu_log[-200:]) if self.mu_log else np.zeros((1, N_WEAR))
        pr = np.array(self.price_log[-200:]) if self.price_log else np.zeros((1, 4))
        return {
            "mu": np.asarray(self.mu, dtype=float).tolist(),
            "mu_scalar": self.mu_scalar,
            "mu_max_seen": float(np.max(mu)),
            "price_mean_mwh": pr.mean(0).tolist(),
            "value_loss": self.loss_log[-1] if self.loss_log else float("nan"),
            "n_fit": len(self.dataset),
            "outer": self.outer,
            "enabled": self.enabled,
            "model": self.model,
            "env_version": ENV_VERSION,
        }
