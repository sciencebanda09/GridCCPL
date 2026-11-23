"""Two-clock grid environment.

Fast clock (15 min):  balance, DC flows, ramps, reserve  -> measured, not hidden
Slow clock (3 h / 24 h / 30 d):  wear -> forced outage, heat-rate drift, settlement

The environment deliberately does **not** clip an infeasible dispatch into
feasibility.  If the controller cannot serve load, the load is unserved and
costs VOLL; if it overloads a line, the overload is metered and penalised.
Silently repairing the agent's mistakes is what makes safe-RL benchmarks
uninformative.

Action (6-dim, absolute, normalised):
    [0] gas      fraction of nominal rating   [0, 1]
    [1] coal     fraction of nominal rating   [0, 1]
    [2] nuclear  fraction of nominal rating   [0, 1]
    [3] battery  + discharge / - charge       [-1, 1]
    [4] import   fraction of tie capacity     [0, 1]
    [5] shed     fraction of load             [0, 1]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from .. import ENV_VERSION
from ..physics import (
    BATTERY_CAPACITY_MWH, BATTERY_POWER_MW, GEN_BUS, GEN_COST, GEN_PMAX,
    GEN_PMIN, GEN_RAMP, GEN_RATING, IMPORT_CAP_MW, LOAD_SHARE, VOLL,
    AgingModel, AgingParams, build_5bus, dc_opf,
)
from ..physics.opf import wear_price_from_mu

STATE_DIM = 17
ACTION_DIM = 6
OVERLOAD_COST = 2000.0     # $/MWh for flow above a line limit
RESERVE_COST = 400.0       # $/MWh for reserve shortfall
REWARD_SCALE = 10_000.0


class Obs(NamedTuple):
    x: np.ndarray
    ctx: dict


@dataclass
class EnvConfig:
    days: int = 10
    steps_per_day: int = 96
    dt_hours: float = 0.25
    difficulty: str = "normal"
    seed: int = 0
    aging: AgingParams = field(default_factory=AgingParams)
    reserve_frac: float = 0.06
    demand_scale: float = 1.0
    wear_cost_per_unit: float = 2.0e4
    noise: float = 0.03
    voll: float = VOLL
    outage_enabled: bool = False
    wear_enabled: bool = True

    @property
    def horizon(self) -> int:
        return self.days * self.steps_per_day

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "aging"}
        d["aging"] = self.aging.as_dict()
        d["env_version"] = ENV_VERSION
        return d


class TwoClockEnv:
    name = "two_clock"

    def __init__(self, config: EnvConfig | None = None, **kw):
        self.cfg = config or EnvConfig(**kw)
        self.net = build_5bus()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.aging = AgingModel(
            n_gen=len(GEN_PMAX),
            params=self.cfg.aging,
            dt_hours=self.cfg.dt_hours,
            rng=np.random.default_rng(self.cfg.seed + 1),
        )
        self._step = 0
        self._prev_p = GEN_PMIN.astype(float).copy()
        self._soc = 0.5
        self._settle = 0.0
        self._done = False
        self._episode_cost = 0.0
        self._episode_unserved = 0.0
        self._episode_overload_mwh = 0.0
        self._episode_outage_steps = 0
        self._history: list[dict] = []
        self._demand_profile = self._build_demand_profile()

    # ------------------------------------------------------------------ shape
    @property
    def state_dim(self) -> int:
        return STATE_DIM

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    @property
    def done(self) -> bool:
        return self._done

    # -------------------------------------------------------------- exogenous
    def _build_demand_profile(self) -> np.ndarray:
        h = np.arange(self.cfg.steps_per_day) * self.cfg.dt_hours
        base = 660.0
        morn = 90.0 * np.exp(-0.5 * ((h - 8.0) / 1.8) ** 2)
        eve = 120.0 * np.exp(-0.5 * ((h - 19.0) / 1.8) ** 2)
        night = -110.0 * np.exp(-0.5 * ((h - 3.0) / 2.2) ** 2)
        prof = base + morn + eve + night
        d = self.cfg.difficulty
        scale = {"easy": 0.90, "normal": 1.00, "hard": 1.06, "peak": 1.12}.get(d, 1.0)
        return prof * scale * float(self.cfg.demand_scale)

    def _hour(self) -> float:
        return (self._step % self.cfg.steps_per_day) * self.cfg.dt_hours

    def _demand(self) -> float:
        h = self._step % self.cfg.steps_per_day
        base = self._demand_profile[h]
        temp = 15 + 8 * np.sin(np.pi * (self._hour() - 6) / 12) + self.rng.normal(0, 2)
        temp_effect = 9.0 * max(0.0, temp - 22.0)
        noise = self.rng.normal(0.0, self.cfg.noise * base)
        return float(max(120.0, base + temp_effect + noise))

    def _renewable(self) -> float:
        h = self._hour()
        solar = 150.0 * 0.65 * np.sin(np.pi * (h - 6) / 12) if 6 < h < 18 else 0.0
        wind = 150.0 * 0.35 * (0.5 + 0.5 * np.sin(self._step * 0.1))
        noise = self.rng.normal(0, 8.0)
        out = float(np.clip(solar + wind + noise, 0.0, GEN_PMAX[3]))
        if self.cfg.difficulty == "storm" and self.rng.random() < 0.05:
            out = float(self.rng.uniform(0, 20))
        return out

    def _spot(self) -> float:
        h = self._hour()
        peak = 40 * (np.exp(-0.5 * ((h - 8) / 2) ** 2) + np.exp(-0.5 * ((h - 19) / 2) ** 2))
        return float(np.clip(80 + peak + self.rng.normal(0, 12), 20, 400))

    # ------------------------------------------------------------------ state
    def _sample_exogenous(self) -> dict:
        """Draw the exogenous quantities for the *current* step exactly once."""
        load = self._demand()
        renew = self._renewable()
        return {
            "load": load,
            "renewable": renew,
            "spot": self._spot(),
            "step": self._step,
        }

    def pmax_eff(self) -> np.ndarray:
        avail = self.aging.available(self._step).astype(float)
        if not self.cfg.outage_enabled:
            avail[:] = 1.0
        cf = self.aging.capacity_factor() if self.cfg.wear_enabled else np.ones(4)
        pm = GEN_PMAX * avail * cf
        pm[3] = min(GEN_PMAX[3], self._exo.get("renewable", 0.0)) if avail[3] else 0.0
        return pm

    def context(self) -> dict:
        load = self._exo["load"]
        pmax = self.pmax_eff()
        reserve_req = self.cfg.reserve_frac * load
        return {
            "step": self._step,
            "load": load,
            "renewable": float(pmax[3]),
            "pmax_eff": pmax,
            "prev_p": self._prev_p.copy(),
            "soc": self._soc,
            "spot": self._exo["spot"],
            "reserve_req": reserve_req,
            "available": self.aging.available(self._step).copy(),
            "wear": self.aging.wear.copy(),
            "eff": self.aging.eff.copy(),
        }

    def _observe(self) -> Obs:
        ctx = self.context()
        load = ctx["load"]
        nominal = float(GEN_PMAX.sum())
        h = self._hour()
        dow = (self._step // self.cfg.steps_per_day) % 7
        outflow = (~ctx["available"]).astype(float)
        x = np.concatenate([
            [load / nominal,
             ctx["renewable"] / GEN_PMAX[3],
             ctx["spot"] / 400.0],
            self.aging.wear,                       # 4
            self.aging.eff,                        # 4
            [self._soc,
             np.sin(2 * np.pi * h / 24),
             np.cos(2 * np.pi * h / 24),
             np.sin(2 * np.pi * dow / 7),
             self._settle / 1e6,
             outflow.mean()],
        ]).astype(np.float32)
        return Obs(x=x, ctx=ctx)

    # ------------------------------------------------------------------- loop
    def reset(self, seed: int | None = None) -> Obs:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.aging.rng = np.random.default_rng(seed + 1)
        self.aging.reset()
        self._step = 0
        self._soc = 0.5
        self._settle = 0.0
        self._done = False
        self._episode_cost = 0.0
        self._episode_unserved = 0.0
        self._episode_overload_mwh = 0.0
        self._episode_outage_steps = 0
        self._history = []
        self._exo = self._sample_exogenous()
        # start near a realistic overnight operating point (not at pmin),
        # so the first step is not an artificial ramp-limited emergency
        self._prev_p = np.array([250.0, 200.0, 190.0,
                                 min(GEN_PMAX[3], self._exo["renewable"])])
        return self._observe()

    def step(self, action) -> tuple[Obs, float, bool, dict]:
        cfg = self.cfg
        ctx = self.context()
        load = ctx["load"]
        pmax = ctx["pmax_eff"]
        prev = self._prev_p
        dt = cfg.dt_hours

        a = np.clip(np.asarray(action, dtype=float).ravel(), -1.0, 1.0)

        # ---- fast-clock dispatch --------------------------------------
        p = np.zeros(4)
        # renewable is uncontrollable: must-take, no ramp limit
        p[3] = pmax[3]
        for g in (0, 1, 2):
            p[g] = np.clip(a[g], 0.0, 1.0) * GEN_PMAX[g]
            p[g] = np.clip(p[g],
                           max(GEN_PMIN[g], prev[g] - GEN_RAMP[g]),
                           max(0.0, min(pmax[g], prev[g] + GEN_RAMP[g])))

        batt_req = a[3] * BATTERY_POWER_MW
        max_dis = min(BATTERY_POWER_MW, self._soc * BATTERY_CAPACITY_MWH / dt)
        max_chg = min(BATTERY_POWER_MW, (1.0 - self._soc) * BATTERY_CAPACITY_MWH / dt)
        batt = float(np.clip(batt_req, -max_chg, max_dis))
        self._soc = float(np.clip(
            self._soc - batt * dt / BATTERY_CAPACITY_MWH, 0.0, 1.0))

        imp = float(np.clip(a[4], 0.0, 1.0) * IMPORT_CAP_MW)
        vol_shed = float(np.clip(a[5], 0.0, 1.0) * 0.30 * load)

        # ---- physical balance (never silently repaired) ----------------
        target = max(0.0, load - vol_shed)
        gen = float(p.sum())
        supply = gen + imp + batt
        curtail = 0.0
        if supply > target:
            surplus = supply - target
            curtail = min(surplus, p[3])
            p[3] -= curtail
            gen -= curtail
            supply = gen + imp + batt
            if supply > target:                # spill firm energy
                p[0] -= min(p[0], supply - target)
                gen = float(p.sum())
                supply = gen + imp + batt
        unserved = max(0.0, target - supply)
        total_unserved = unserved + vol_shed

        served = target - unserved
        inj = np.zeros(self.net.n_bus)
        for g, bus in enumerate(GEN_BUS):
            inj[bus] += p[g]
        inj[4] += imp + batt
        inj -= LOAD_SHARE * served
        theta = self.net.theta(inj)
        flows = self.net.flows(theta)
        overload = float(self.net.overload(flows).sum())

        # ---- reserve ---------------------------------------------------
        reserve = float(np.sum(pmax[:3] - p[:3]) + (BATTERY_POWER_MW - batt))
        reserve_req = ctx["reserve_req"]
        reserve_short = max(0.0, reserve_req - reserve)

        # ---- costs -----------------------------------------------------
        heat = self.aging.heat_rate() * self.aging.eff
        fuel = float(np.dot(p * heat, GEN_COST) * dt)
        imp_cost = imp * ctx["spot"] * dt
        unserved_cost = total_unserved * cfg.voll * dt
        overload_cost = overload * OVERLOAD_COST * dt
        reserve_cost = reserve_short * RESERVE_COST * dt
        step_cost = fuel + imp_cost + unserved_cost + overload_cost + reserve_cost

        # ---- slow clock ------------------------------------------------
        aging_inc = self.aging.step_wear(p, GEN_RATING, prev)
        tripped = []
        if cfg.outage_enabled and self._step > 0 and self._step % cfg.steps_per_day == 0:
            tripped = self.aging.daily_update(self._step, cfg.steps_per_day)
        self.aging.tick_outages(self._step + 1)
        wear_cost = cfg.wear_cost_per_unit * float(np.sum(aging_inc)) / max(dt, 1e-9) * dt

        rho = float(np.exp(-dt / (30 * 24)))
        self._settle = rho * self._settle + step_cost

        # ---- bookkeeping ----------------------------------------------
        self._prev_p = p.copy()
        self._episode_cost += step_cost
        self._episode_unserved += total_unserved * dt
        self._episode_overload_mwh += overload * dt
        self._episode_outage_steps += int((~self.aging.available(self._step + 1)).sum())
        self._step += 1
        self._done = self._step >= cfg.horizon
        self._exo = self._sample_exogenous()

        reward = -step_cost / REWARD_SCALE

        info = {
            "step": self._step,
            "p_gen": p.copy(),
            "battery": batt,
            "import_mw": imp,
            "unserved_mw": total_unserved,
            "voluntary_shed_mw": vol_shed,
            "curtail_mw": curtail,
            "flows": flows.copy(),
            "overload_mw": overload,
            "reserve_mw": reserve,
            "reserve_short_mw": reserve_short,
            "fuel_cost": fuel,
            "unserved_cost": unserved_cost,
            "overload_cost": overload_cost,
            "reserve_cost": reserve_cost,
            "step_cost": step_cost,
            "wear": self.aging.wear.copy(),
            "wear_inc": aging_inc.copy(),
            "eff": self.aging.eff.copy(),
            "outage_prob": self.aging.outage_prob(),
            "tripped": list(tripped),
            "outage_frac": float((~self.aging.available(self._step)).mean()),
            "soc": self._soc,
            "settle": self._settle,
            "lmp_hint": float(ctx["spot"]),
        }
        self._history.append({k: info[k] for k in
                              ("step", "step_cost", "unserved_mw", "overload_mw",
                               "wear", "outage_frac")})
        return self._observe(), reward, self._done, info

    # --------------------------------------------------------------- reporting
    def episode_summary(self) -> dict:
        """The domain-native scoreboard (all in physical units)."""
        h = self._history
        wear_end = self.aging.wear.copy()
        p = self.aging.params
        hr_series = [
            float(np.mean(1.0 + p.heat_lin * r["wear"][:3]
                          + p.heat_quad * (r["wear"][:3] ** 2)))
            for r in h
        ]
        heat_rate_mean = float(np.mean(hr_series)) if hr_series else 1.0
        return {
            "episode_cost": float(self._episode_cost),
            "unserved_mwh": float(self._episode_unserved),
            "overload_mwh": float(self._episode_overload_mwh),
            "outage_unit_steps": int(self._episode_outage_steps),
            "wear_end_mean": float(wear_end.mean()),
            "wear_end_max": float(wear_end.max()),
            "wear_integral": float(np.sum([np.sum(r["wear"]) for r in h])),
            "eff_end_mean": float(self.aging.eff.mean()),
            "heat_rate_mean": heat_rate_mean,
            "trips": int(self.aging.outage_count.sum()),
            "steps": int(self._step),
            "settle": float(self._settle),
            "cost_per_mwh_served": float(
                self._episode_cost / max(1e-9, 660.0 * self.cfg.days)),
        }

    def sub_episodes(self, days: int) -> list[dict]:
        """Split the run into ``days``-long chunks (for 'cheapest day' analysis)."""
        spd = self.cfg.steps_per_day
        out = []
        for d0 in range(0, self.cfg.days, days):
            lo, hi = d0 * spd, min((d0 + days) * spd, len(self._history))
            chunk = self._history[lo:hi]
            if not chunk:
                continue
            out.append({
                "day_start": d0,
                "cost": float(sum(r["step_cost"] for r in chunk)),
                "unserved_mwh": float(sum(r["unserved_mw"] for r in chunk) * self.cfg.dt_hours),
                "wear": float(np.mean([np.mean(r["wear"]) for r in chunk])),
                "outage_frac": float(np.mean([r["outage_frac"] for r in chunk])),
            })
        return out


# ---------------------------------------------------------------------------
# Environment variants (same physics, different exogenous stress)
# ---------------------------------------------------------------------------

def make_env(name: str = "normal", days: int = 10, seed: int = 0, **kw) -> TwoClockEnv:
    diff = {"normal": "normal", "hard": "hard", "peak": "peak", "storm": "storm"}.get(name, "normal")
    cfg = EnvConfig(days=days, difficulty=diff, seed=seed, **kw)
    env = TwoClockEnv(cfg)
    env.name = name
    return env


def env_registry() -> dict:
    return {
        "normal": lambda days=10, seed=0, **kw: make_env("normal", days, seed, **kw),
        "hard": lambda days=10, seed=0, **kw: make_env("hard", days, seed, **kw),
        "peak": lambda days=10, seed=0, **kw: make_env("peak", days, seed, **kw),
        "storm": lambda days=10, seed=0, **kw: make_env("storm", days, seed, **kw),
    }
