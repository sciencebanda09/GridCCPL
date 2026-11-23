"""Baselines for the two-clock grid.

All agents share one interface::

    act(obs) -> action vector (6,)

``obs`` is ``Obs(x, ctx)`` from the environment, so optimisation-based agents
can see the dispatch context without the environment leaking it to learners.
"""
from __future__ import annotations

import numpy as np

from ..physics import (
    BATTERY_CAPACITY_MWH, BATTERY_POWER_MW, GEN_COST, GEN_DISPATCHABLE, GEN_PMAX,
    GEN_RAMP, GEN_RATING, IMPORT_CAP_MW, dc_opf, fast_dispatch,
    wear_price_from_mu, wear_free_mw,
)
from ..physics.opf import OPFResult

SHED_CAP_FRAC = 0.30


def action_from_opf(res: OPFResult, load: float) -> np.ndarray:
    a = np.zeros(6)
    for g in (0, 1, 2):
        a[g] = res.p_gen[g] / GEN_PMAX[g]
    a[3] = res.battery / BATTERY_POWER_MW
    a[4] = res.import_mw / IMPORT_CAP_MW
    a[5] = res.shed_total / max(1e-9, SHED_CAP_FRAC * load)
    return np.clip(a, -1.0, 1.0)


def _battery_bounds(soc: float, dt: float) -> tuple[float, float]:
    max_dis = min(BATTERY_POWER_MW, soc * BATTERY_CAPACITY_MWH / dt)
    max_chg = min(BATTERY_POWER_MW, (1.0 - soc) * BATTERY_CAPACITY_MWH / dt)
    return -max_chg, max_dis


class BaseAgent:
    name = "base"

    def reset(self, env=None) -> None:
        pass

    def act(self, obs) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    # some agents want to observe the realised step (for learning)
    def observe(self, step_info: dict) -> None:
        pass


# ---------------------------------------------------------------------------


class RandomAgent(BaseAgent):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs) -> np.ndarray:
        return self.rng.uniform(-1, 1, 6)


class RuleBase(BaseAgent):
    """Greedy merit-order rule heuristic with fixed time-of-day battery cycling.

    Allocates available ramp-bounded capacity in fuel-cost order without solving
    the OPF network or reserve constraints, plus fixed peak/off-peak battery
    scheduling and small Gaussian jitter.  This is deliberately *not* an economic
    optimiser; it serves as a hand-coded heuristic reference below the blind OPF.
    """

    name = "rule"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs) -> np.ndarray:
        ctx = obs.ctx
        load = ctx["load"]
        pmax = ctx["pmax_eff"]
        prev = ctx["prev_p"]
        cap = float(pmax.sum()) + BATTERY_POWER_MW + IMPORT_CAP_MW

        a = np.zeros(6)
        # greedy merit order (cheapest fuel first, bounded by ramp and pmax_eff)
        order = [3, 2, 1, 0]
        alloc = np.zeros(4)
        remaining = load
        for g in order:
            room = max(0.0, min(pmax[g], prev[g] + GEN_RAMP[g]))
            add = min(room, remaining)
            alloc[g] = add
            remaining -= add
        for g in (0, 1, 2):
            a[g] = alloc[g] / GEN_PMAX[g]

        # battery: discharge at peak, charge when cheap and low load
        h = np.arctan2(obs.x[12], obs.x[13]) / (2 * np.pi) * 24 % 24
        peak = (7 < h < 10) or (18 < h < 22)
        if peak and ctx["soc"] > 0.25:
            a[3] = -0.6
        elif not peak and ctx["soc"] < 0.8 and ctx["spot"] < 60:
            a[3] = 0.5

        if remaining > 0:
            a[4] = min(1.0, remaining / IMPORT_CAP_MW)
        if load > 0.97 * cap:
            a[5] = 0.3

        a[:3] += self.rng.normal(0, 0.01, 3)
        return np.clip(a, -1, 1)


class OPFAgent(BaseAgent):
    """Dispatch from the fast-clock OPF, optionally wear-aware.

    * ``mu = None``  -> degradation-blind OPF (the industry default)
    * ``mu = value`` -> wear-aware OPF (oracle if the value is the true price)
    """

    name = "opf"

    def __init__(self, net, aging_params, mu: float | None = None,
                 use_true_price: bool = False, fast: bool = True):
        self.net = net
        self.aging = aging_params
        self.mu = mu
        self.use_true_price = use_true_price
        self.fast = fast
        self.last_result = None
        self.name = "opf_blind" if mu is None else f"opf_mu{mu:g}"

    def act(self, obs) -> np.ndarray:
        ctx = obs.ctx
        lo, hi = _battery_bounds(ctx["soc"], 0.25)
        wp = None
        free = None
        if self.mu is not None:
            wp = wear_price_from_mu(self.mu, self.aging, GEN_RATING, GEN_DISPATCHABLE)
            free = wear_free_mw(self.aging)
        solve = fast_dispatch if self.fast else dc_opf
        res = solve(
            self.net, ctx["load"], ctx["prev_p"], ctx["renewable"], ctx["pmax_eff"],
            reserve_req_mw=ctx["reserve_req"],
            battery_min_mw=lo, battery_max_mw=hi,
            spot_price=ctx["spot"], wear_price_mwh=wp, wear_free_mw=free,
        )
        self.last_result = res
        return action_from_opf(res, ctx["load"])


class OPFBlind(OPFAgent):
    name = "opf_blind"

    def __init__(self, net, aging_params):
        super().__init__(net, aging_params, mu=None)


class OPFAware(OPFAgent):
    name = "opf_aware"

    def __init__(self, net, aging_params, mu: float, fast: bool = True):
        super().__init__(net, aging_params, mu=mu, fast=fast)
        self.name = f"opf_mu{mu:g}"
