"""Slow-clock physics: equipment wear, efficiency drift, forced outages.

This is the heart of the two-clock idea.  A 15-minute dispatch decision cannot
observe any of the quantities below, yet they set the bill for the month:

    wear_g(t+1)   = wear_g(t) * exp(-dt / tau_thermal)  +  c_age * overloading
    outage prob   = 1 - exp(-kappa * wear_g)                 (on the daily clock)
    eff_g(t+1)    = eff_g(t) * exp(-dt / tau_eff) + drift      (heat-rate penalty)
    settle(t+1)   = rho * settle(t) + cost(t)                  (30-day settlement)

The parameters are chosen so the *slow* cost is the same order as a day of fuel
cost, which is what makes the two clocks interact.  Values are calibrated from
published cycling-cost ranges (see ``AgingParams`` docstring).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AgingParams:
    """Parameters of the slow clock.

    tau_thermal : hours. Thermal time constant of a large coal/gas unit
                  (1-6 h is the usual range for metal temperature decay).
    tau_eff     : hours. Heat-rate recovery constant (~1 day).
    c_age       : per-unit aging coefficient for loading above ``stress_frac``.
    kappa       : outage-rate coefficient; wear=1 gives ~63% daily outage odds.
    stress_frac : loading fraction above which aging accrues.
    repair_days : days a failed unit stays out.
    drift       : heat-rate drift per unit of normalised ramping.
    """

    tau_thermal: float = 3.0
    tau_eff: float = 24.0
    c_age: float = 0.55
    kappa: float = 6.0
    stress_frac: float = 0.85
    repair_days: float = 1.0
    drift: float = 0.02
    eff_max: float = 1.35
    outage_law: str = "quadratic"   # damage ~ wear^p (thermal aging power law)
    derate: float = 0.40            # legacy linear capacity loss at wear = 1
    eff_wear: float = 0.60          # legacy linear heat-rate penalty at wear = 1
    derate_lin: float = 0.10        # capacity loss, linear term
    derate_quad: float = 1.20       # capacity loss, quadratic term
    heat_lin: float = 0.10          # heat-rate penalty, linear term
    heat_quad: float = 0.80         # heat-rate penalty, quadratic term
    c_age_gen: tuple = (1.0, 1.7, 3.2, 0.0)   # gas, coal, nuclear, renewable

    def as_dict(self) -> dict:
        return {
            "tau_thermal": self.tau_thermal,
            "tau_eff": self.tau_eff,
            "c_age": self.c_age,
            "kappa": self.kappa,
            "stress_frac": self.stress_frac,
            "repair_days": self.repair_days,
            "drift": self.drift,
            "eff_max": self.eff_max,
            "derate": self.derate,
            "eff_wear": self.eff_wear,
            "derate_lin": self.derate_lin,
            "derate_quad": self.derate_quad,
            "heat_lin": self.heat_lin,
            "heat_quad": self.heat_quad,
            "c_age_gen": list(self.c_age_gen),
            "outage_law": self.outage_law,
        }


@dataclass
class AgingModel:
    """Vectorised slow-state integrator for a generator fleet."""

    n_gen: int
    params: AgingParams = field(default_factory=AgingParams)
    dt_hours: float = 0.25
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    # state ---------------------------------------------------------------
    wear: np.ndarray = field(init=False)
    eff: np.ndarray = field(init=False)
    outage_until: np.ndarray = field(init=False)   # step index, -1 = available
    outage_count: np.ndarray = field(init=False)
    outage_steps: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.wear = np.zeros(self.n_gen, dtype=float)
        self.eff = np.ones(self.n_gen, dtype=float)
        self.outage_until = np.full(self.n_gen, -1, dtype=int)
        self.outage_count = np.zeros(self.n_gen, dtype=int)
        self.outage_steps = np.zeros(self.n_gen, dtype=int)

    def available(self, step: int) -> np.ndarray:
        """Boolean mask of units that are physically available at ``step``."""
        return self.outage_until < step

    # ------------------------------------------------------------- fast clock
    def step_wear(self, p_mw: np.ndarray, rating_mw: np.ndarray,
                  prev_p_mw: np.ndarray | None = None) -> np.ndarray:
        """Integrate wear and efficiency drift for one dispatch step.

        Returns the *incremental* aging accrued this step (per unit), which is
        the quantity the shadow price is defined against.
        """
        p = np.asarray(p_mw, dtype=float)
        rating = np.maximum(np.asarray(rating_mw, dtype=float), 1e-6)
        loading = p / rating
        par = self.params

        overload = np.maximum(0.0, loading - par.stress_frac) / max(1e-9, 1.0 - par.stress_frac)
        c_age_vec = par.c_age * np.asarray(par.c_age_gen, dtype=float)[:self.n_gen]
        increment = c_age_vec * overload * self.dt_hours

        decay = np.exp(-self.dt_hours / max(par.tau_thermal, 1e-6))
        self.wear = np.clip(self.wear * decay + increment, 0.0, 1.0)

        # heat-rate drift from ramping
        if prev_p_mw is None:
            ramp_frac = np.zeros_like(p)
        else:
            ramp_frac = np.abs(p - np.asarray(prev_p_mw, dtype=float)) / rating
        eff_decay = np.exp(-self.dt_hours / max(par.tau_eff, 1e-6))
        self.eff = np.clip(self.eff * eff_decay + par.drift * ramp_frac * self.dt_hours,
                           1.0, par.eff_max)
        return increment

    # ------------------------------------------------------------- slow clock
    def outage_prob(self) -> np.ndarray:
        """Daily forced-outage probability.

        Thermal aging follows a power law in the damage variable, so the
        outage hazard is ``1 - exp(-kappa * wear**2)``.  This is what makes the
        slow clock *policy-sensitive*: gentle operation is nearly free, heavy
        loading is not.
        """
        w = np.clip(self.wear, 0.0, 1.0)
        if self.params.outage_law == "quadratic":
            haz = self.params.kappa * w ** 2
        elif self.params.outage_law == "linear":
            haz = self.params.kappa * w
        else:  # threshold: nothing happens until wear > 0.6
            haz = self.params.kappa * np.maximum(0.0, w - 0.6) ** 2
        return 1.0 - np.exp(-haz)

    def daily_update(self, step: int, steps_per_day: int) -> list[int]:
        """Sample forced outages once per simulated day.

        Returns the list of generator indices that tripped today.
        """
        probs = self.outage_prob()
        tripped: list[int] = []
        repair = int(round(self.params.repair_days * steps_per_day))
        for g in range(self.n_gen):
            if self.outage_until[g] >= step:
                continue
            if self.rng.random() < probs[g]:
                self.outage_until[g] = step + repair
                self.outage_count[g] += 1
                tripped.append(g)
        return tripped

    def tick_outages(self, step: int) -> None:
        self.outage_steps += (self.outage_until >= step).astype(int)

    # -------------------------------------------------------------- reporting
    def capacity_factor(self) -> np.ndarray:
        """Available capacity fraction: worn units are derated.

        Damage accelerates with damage (a worn unit loses capacity faster than
        a fresh one), so the derating curve is convex.  This is what gives the
        slow-clock problem an *interior* optimum: there is a best amount of
        wear, not simply none.
        """
        w = np.clip(self.wear, 0.0, 1.0)
        p = self.params
        return np.clip(1.0 - (p.derate_lin * w + p.derate_quad * w ** 2), 0.05, 1.0)

    def heat_rate(self) -> np.ndarray:
        """Fuel-cost multiplier from aging (worn units burn more fuel)."""
        w = np.clip(self.wear, 0.0, 1.0)
        p = self.params
        return 1.0 + p.heat_lin * w + p.heat_quad * w ** 2

    def marginal_wear_per_mwh(self, rating_mw: np.ndarray) -> np.ndarray:
        """d(wear)/d(MW) -- the conversion from $/wear-unit to $/MWh.

        Piecewise-linear in loading, so pricing wear keeps the fast-clock
        problem an LP.
        """
        p = self.params
        c_age_vec = p.c_age * np.asarray(p.c_age_gen, dtype=float)
        return c_age_vec / max(1e-9, 1.0 - p.stress_frac) / np.maximum(rating_mw, 1e-6)

    def snapshot(self) -> dict:
        return {
            "wear": self.wear.copy(),
            "eff": self.eff.copy(),
            "outage_prob": self.outage_prob(),
            "outage_until": self.outage_until.copy(),
        }


def marginal_wear_series(loading: np.ndarray, rating: np.ndarray,
                         params: AgingParams, dt_hours: float = 0.25) -> np.ndarray:
    """Per-step aging increment, useful for building the true shadow price."""
    loading = np.asarray(loading, dtype=float)
    rating = np.maximum(np.asarray(rating, dtype=float), 1e-6)
    l = loading / rating
    excess = np.maximum(0.0, l - params.stress_frac) / max(1e-9, 1.0 - params.stress_frac)
    return params.c_age * excess * dt_hours
