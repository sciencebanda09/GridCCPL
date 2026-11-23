"""Pathwise (counterfactual) value-gradient price.

Why a new learner
-----------------
``ValueGradientAgent`` reads ``mu = dV/dW`` off a ridge regression of
Monte-Carlo cost-to-go on state features.  That slope is *correlational*:

* on blind-policy data, wear is high exactly when load is high, so the wear
  coefficient absorbs the cost of high-load hours;
* once the price is on, the on-policy wear is identically ~0, so the wear
  coefficient is not identified from on-policy data; the replay buffer then
  mixes blind (high W, high cost) and priced (W = 0, low cost) episodes and
  the slope measures the *policy* difference divided by a wear difference --
  a number in the 1e5-1e6 range that hits ``mu_max``;
* the saturated price is then clipped per unit at ``price_clip`` $/MWh, which
  makes all three excess blocks cost ~the same and *inverts* their merit order
  (the most wear-intensive unit, nuclear, becomes the cheapest excess block).
  That inversion, not the size of mu itself, is what costs money on hard/peak.

This learner estimates the derivative causally instead of by regression:

    mu_g(pi)  =  E_s [ (J^pi_H(s with w_g + delta) - J^pi_H(s)) / delta ]

at states ``s`` visited by the current policy ``pi``, by re-rolling a cloned
environment for ``H`` steps under *common random numbers* (the exogenous
stream is the same; the dispatch is deterministic), so the difference is
purely the effect of the extra wear -- including the policy's own reaction to
the derated capacity.  ``delta`` is a *secant*, of the order of the wear one
hour of excess loading adds, because the price has to value non-infinitesimal
wear increments (heat-rate and derate are convex in wear).

Policy iteration with damping:  mu_{k+1} = (1 - a) mu_k + a * g(mu_k).

No caps are active by construction: the price vector is passed unclipped to
the LP (``mu_max`` is a sanity bound only and its activation is reported).
"""
from __future__ import annotations

import copy

import numpy as np

from ..envs import make_env
from ..physics import (
    GEN_DISPATCHABLE, GEN_RATING, fast_dispatch, wear_free_mw, wear_price_from_mu,
)
from .baselines import BaseAgent, _battery_bounds, action_from_opf

N_THERMAL = 3


def clone_env(env):
    """Cheap exact copy of a TwoClockEnv (same exogenous RNG stream = CRN)."""
    e = copy.copy(env)
    e.rng = copy.deepcopy(env.rng)
    a = copy.copy(env.aging)
    for k in ("wear", "eff", "outage_until", "outage_count", "outage_steps"):
        setattr(a, k, getattr(env.aging, k).copy())
    a.rng = copy.deepcopy(env.aging.rng)
    e.aging = a
    e._prev_p = env._prev_p.copy()
    e._exo = dict(env._exo)
    e._history = []
    return e


class PathwisePriceAgent(BaseAgent):
    """Wear-aware OPF whose per-unit price is a pathwise value gradient."""

    name = "pvg"

    def __init__(self, net, aging_params, mu0: float = 0.0,
                 mu_max: float = 1.0e6, seed: int = 0, trim: float = 0.2):
        self.net = net
        self.aging = aging_params
        self.mu = np.zeros(4)
        self.mu[:N_THERMAL] = mu0
        self.mu_max = mu_max
        self.trim = trim
        self.rng = np.random.default_rng(seed)
        self.free = wear_free_mw(aging_params)
        self.history: list[dict] = []
        self.train_steps = 0          # env steps simulated during training
        self.cap_hit = False

    # ------------------------------------------------------------- policy
    def wear_price(self) -> np.ndarray:
        return wear_price_from_mu(self.mu, self.aging, GEN_RATING, GEN_DISPATCHABLE)

    def act(self, obs) -> np.ndarray:
        ctx = obs.ctx
        lo, hi = _battery_bounds(ctx["soc"], 0.25)
        wp = self.wear_price() if np.any(self.mu > 0) else None
        res = fast_dispatch(
            self.net, ctx["load"], ctx["prev_p"], ctx["renewable"], ctx["pmax_eff"],
            reserve_req_mw=ctx["reserve_req"], battery_min_mw=lo, battery_max_mw=hi,
            spot_price=ctx["spot"], wear_price_mwh=wp,
            wear_free_mw=self.free if wp is not None else None)
        return action_from_opf(res, ctx["load"])

    @property
    def mu_scalar(self) -> float:
        return float(np.mean(self.mu[:N_THERMAL]))

    # ----------------------------------------------------------- learning
    def _roll(self, env, n: int) -> float:
        c = 0.0
        obs = env._observe()
        for _ in range(n):
            if env.done:
                break
            obs, _, _, info = env.step(self.act(obs))
            c += info["step_cost"]
            self.train_steps += 1
        return c

    def pathwise_gradient(self, env_name: str, seeds, days: int = 10,
                          n_snap: int = 16, horizon: int = 96,
                          delta: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        """dJ/dw_g at on-policy states, by CRN finite difference.

        Returns (mean gradient per thermal unit, standard error).
        """
        G = []
        for s in seeds:
            env = make_env(env_name, days=days, seed=s)
            obs = env.reset()
            T = env.cfg.horizon
            snaps = set(self.rng.choice(np.arange(0, T - 1), size=n_snap,
                                        replace=False).tolist())
            for t in range(T):
                if t in snaps:
                    base = self._roll(clone_env(env), horizon)
                    row = []
                    for g in range(N_THERMAL):
                        e1 = clone_env(env)
                        w0 = e1.aging.wear[g]
                        e1.aging.wear[g] = min(1.0, w0 + delta)
                        d = e1.aging.wear[g] - w0
                        row.append((self._roll(e1, horizon) - base) / max(d, 1e-9))
                    G.append(row)
                obs, _, _, _ = env.step(self.act(obs))
                self.train_steps += 1
        G = np.asarray(G, dtype=float)
        if self.trim <= 0:
            return G.mean(0), G.std(0, ddof=1) / np.sqrt(len(G))
        # Robust location: CRN differences of a piecewise-constant policy have
        # rare, large, *two-sided* excursions (a perturbation flips one
        # dispatch decision, which cascades into a different ramp/unserved
        # path).  A symmetric trimmed mean removes them.
        k = int(np.floor(self.trim * len(G)))
        Gs = np.sort(G, axis=0)[k:len(G) - k]
        return Gs.mean(0), Gs.std(0, ddof=1) / np.sqrt(len(Gs))

    def train(self, env_name: str, seeds, days: int = 10, iters: int = 6,
              damping: float = 0.5, n_snap: int = 16, horizon: int = 96,
              delta: float = 0.05, tol: float = 0.05, verbose: bool = True) -> dict:
        """Damped policy iteration on the pathwise gradient."""
        for k in range(iters):
            g, se = self.pathwise_gradient(env_name, seeds, days, n_snap, horizon, delta)
            g = np.maximum(g, 0.0)
            old = self.mu[:N_THERMAL].copy()
            new = (1.0 - damping) * old + damping * g if k > 0 else g
            self.cap_hit = self.cap_hit or bool(np.any(new > self.mu_max))
            self.mu[:N_THERMAL] = np.clip(new, 0.0, self.mu_max)
            rel = float(np.max(np.abs(self.mu[:N_THERMAL] - old)
                               / np.maximum(np.abs(old), 1.0)))
            rec = {"iter": k + 1, "grad": g.tolist(), "grad_se": se.tolist(),
                   "mu": self.mu[:N_THERMAL].tolist(), "rel_change": rel,
                   "train_steps": self.train_steps}
            self.history.append(rec)
            if verbose:
                print(f"    iter {k + 1}: g = {np.round(g)}  ± {np.round(se)}  "
                      f"-> mu = {np.round(self.mu[:N_THERMAL])}  (rel {rel:.3f})",
                      flush=True)
            if k > 0 and rel < tol:
                break
        return {"mu": self.mu.tolist(), "history": self.history,
                "train_steps": self.train_steps, "cap_hit": self.cap_hit}

    def train_bisect(self, env_name: str, seeds, days: int = 10, iters: int = 7,
                     lo: float = 300.0, hi: float = 1.0e6, n_snap: int = 16,
                     horizon: int = 96, delta: float = 0.05,
                     shared: bool = False, verbose: bool = True) -> dict:
        """Smallest self-consistent price, per unit, by log-bisection.

        ``shared=True`` learns one scalar price for all thermal units (the
        paper's scalar ``mu``): a price is accepted only if it is consistent
        for *every* unit, ``max_g g_g(mu) <= mu``.  This avoids the coupling
        of the per-unit search (a unit's marginal wear cost depends on the
        other units' prices).

        ``g(mu)`` -- the pathwise marginal cost of wear under the policy that
        ``mu`` induces -- is (empirically) *decreasing* in ``mu``: a higher
        price keeps the fleet out of the convex part of the damage curve, where
        wear is expensive.  Plain or damped iteration ``mu <- g(mu)`` on a
        decreasing step map oscillates (price high -> no wear -> wear looks
        cheap -> price low -> heavy wear -> wear looks dear).  The fixed point
        is where ``g`` jumps through the diagonal: the smallest ``mu`` with
        ``g(mu) <= mu``, i.e. the cheapest price not undercut by the marginal
        cost of the wear it lets through.  We bracket it per unit in log space
        and return the upper end of the bracket (a tested price that satisfied
        ``g <= mu``).
        """
        lo_v = np.full(N_THERMAL, float(lo))
        hi_v = np.full(N_THERMAL, float(hi))
        hi_ok = np.zeros(N_THERMAL, dtype=bool)
        for k in range(iters):
            mid = np.sqrt(lo_v * hi_v)
            self.mu[:N_THERMAL] = mid
            g, se = self.pathwise_gradient(env_name, seeds, days, n_snap, horizon, delta)
            ok = g <= mid                     # mid is self-consistent
            if shared:                        # one scalar price for the fleet
                ok = np.full(N_THERMAL, bool(np.all(ok)))
            hi_v = np.where(ok, mid, hi_v)
            hi_ok |= ok
            lo_v = np.where(ok, lo_v, mid)
            rec = {"iter": k + 1, "mu_tested": mid.tolist(), "grad": g.tolist(),
                   "grad_se": se.tolist(), "consistent": ok.tolist(),
                   "bracket_lo": lo_v.tolist(), "bracket_hi": hi_v.tolist(),
                   "train_steps": self.train_steps}
            self.history.append(rec)
            if verbose:
                print(f"    bisect {k + 1}: mu = {np.round(mid)}  g = {np.round(g)} "
                      f"+- {np.round(se)}  ok = {ok.astype(int)}", flush=True)
        # a unit never found consistent sits at the top of the search range
        self.cap_hit = bool(np.any(~hi_ok))
        self.mu[:N_THERMAL] = hi_v
        return {"mu": self.mu.tolist(), "history": self.history,
                "train_steps": self.train_steps, "cap_hit": self.cap_hit}
