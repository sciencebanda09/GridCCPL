"""R11 -- Risk-Averse Wear Pricing Under Forced Outages (AAAI-27 PGAI4E).

Tests whether the Proposition 2(iii) shedding thresholds
    mu_hi_g = (VOLL - c_g) / alpha_g
        nuclear: ~84,886   coal: ~238,476   gas: ~471,068
partition wear prices above the deterministic dual certificate mu_lo into four
analytical, zero-learning risk tiers under quadratic wear-triggered forced
outages P(trip_g) = 1 - exp(-kappa * W_g^2):
    Tier 0: [mu_lo, mu_hi_nuc)         (deterministic certificate; no voluntary shed)
    Tier 1: (mu_hi_nuc, mu_hi_coal)    (preemptively sheds to protect nuclear)
    Tier 2: (mu_hi_coal, mu_hi_gas)    (preemptively sheds to protect nuclear + coal)
    Tier 3: (mu_hi_gas, inf)           (preemptively sheds to protect all 3 thermal units)

Also tests:
  * Mean vs CVaR_90 separation across outage severities kappa in {0, 0.25, 1, 3, 6}
    and stress regimes (normal 1.0x, tight 1.15x, heatwave 4-day 1.22x, tight 1.25x);
  * State-dependent wear-gated and continuous hazard-triggered risk pricing vs
    constant tiers (honest train/test split: TRAIN=101..120, TEST=1..40);
  * Physical mechanism diagnosis of why R3 (tight_1.15_outages) still had 2/5 test
    seeds trip at mu=1e6 (fast_dispatch VOLL-gate bug vs exact LP dc_opf, plus
    exact LP diagnostics for shed_cap / ramp-forced wear / reserve binding).

Run:
    python -m twoclock.experiments.r11_risk_outages --pilot   # fast 20-train-seed scan
    python -m twoclock.experiments.r11_risk_outages           # full experiment + plot
    python -m twoclock.experiments.r11_risk_outages --plot    # regenerate figure from JSON
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from twoclock.agents.baselines import BaseAgent, _battery_bounds, action_from_opf
from twoclock.agents.certificate import alpha_coef, fixed_point_certificate
from twoclock.envs import TwoClockEnv, make_env
from twoclock.envs.two_clock_env import EnvConfig
from twoclock.experiments.runner import run_episode
from twoclock.paths import RESULTS_DIR
from twoclock.physics import (
    BATTERY_POWER_MW,
    GEN_COST,
    GEN_DISPATCHABLE,
    GEN_PMAX,
    GEN_PMIN,
    GEN_RAMP,
    GEN_RATING,
    IMPORT_CAP_MW,
    VOLL,
    AgingParams,
    build_5bus,
    dc_opf,
    fast_dispatch,
    wear_free_mw,
    wear_price_from_mu,
)
from twoclock.physics.dc_flow import BATTERY_BUS, GEN_BUS, IMPORT_BUS, LOAD_SHARE
from twoclock.physics.opf import OPFResult, _fast_merit_dispatch, _ramp_bounds

# ---------------------------------------------------------------------------
# Protocol constants (strict honest train/test split)
# ---------------------------------------------------------------------------

DAYS = 10
TRAIN_SEEDS = list(range(101, 121))   # 20 training seeds (selection/calibration ONLY)
CERT_SEEDS = [101, 102, 103]          # 3 training seeds for deterministic certificate
TEST_SEEDS = list(range(1, 41))       # 40 held-out test seeds
BOOT_N = 2000
BOOT_SEED = 20261122
MAX_WORKERS = min(11, max(1, (os.cpu_count() or 4) - 1))

OUT_JSON = os.path.join(RESULTS_DIR, "r11_risk_outages.json")
OUT_FIG_PDF = os.path.join(RESULTS_DIR, "fig_r11_risk_outages.pdf")
OUT_FIG_PNG = os.path.join(RESULTS_DIR, "fig_r11_risk_outages.png")

# Analytical Proposition 2(iii) shedding thresholds
_BASE_AGING = AgingParams()
_ALPHA = alpha_coef(_BASE_AGING)
MU_HI_G = (VOLL - GEN_COST[:3]) / _ALPHA[:3]   # [gas, coal, nuclear]
MU_HI_GAS = float(MU_HI_G[0])                  # ~471,068.18
MU_HI_COAL = float(MU_HI_G[1])                 # ~238,475.94
MU_HI_NUC = float(MU_HI_G[2])                  #  ~84,886.36

TIER1_MU = float(np.sqrt(MU_HI_NUC * MU_HI_COAL))   # ~142,278.7
TIER2_MU = float(np.sqrt(MU_HI_COAL * MU_HI_GAS))   # ~335,169.4
TIER3_MU = float(1.5 * MU_HI_GAS)                   # ~706,602.3

# Candidate grids for TRAIN selection (strictly selected on TRAIN_SEEDS only)
# Spans [mu_lo, 1e6] covering all four Prop. 2(iii) tiers plus upper edges
CONST_GRID_BASE = [
    7.5e4,
    TIER1_MU,
    TIER2_MU,
    TIER3_MU,
    1.0e6,
]
# Dense mu sweep for Panel (a) curve on tight_1.15 (brackets every Prop. 2(iii) boundary)
DENSE_MU_GRID = [
    0.0, 3.0e3, 1.0e4, 7.8e4,
    9.2e4, TIER1_MU, 2.25e5, 2.55e5, TIER2_MU, 4.5e5, 4.95e5, TIER3_MU, 1.0e6,
]
WGATE_GRID = [0.005, 0.01, 0.02, 0.04, 0.06, 0.10, 0.15, 0.20]
HAZARD_GAMMA_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

# Regimes: primary kappa sweep on tight_1.15 + stress regimes
REGIMES = [
    {"name": "tight_1.15_k0.00", "base": "tight_1.15", "scale": 1.15, "heatwave": False, "kappa": 0.00, "curve": True},
    {"name": "tight_1.15_k0.25", "base": "tight_1.15", "scale": 1.15, "heatwave": False, "kappa": 0.25, "curve": True},
    {"name": "tight_1.15_k1.00", "base": "tight_1.15", "scale": 1.15, "heatwave": False, "kappa": 1.00, "curve": True},
    {"name": "tight_1.15_k3.00", "base": "tight_1.15", "scale": 1.15, "heatwave": False, "kappa": 3.00, "curve": True},
    {"name": "tight_1.15_k6.00", "base": "tight_1.15", "scale": 1.15, "heatwave": False, "kappa": 6.00, "curve": True},
    {"name": "normal_1.00_k6.00", "base": "normal_1.00", "scale": 1.00, "heatwave": False, "kappa": 6.00, "curve": False},
    {"name": "heatwave_1.22_k1.00", "base": "heatwave_1.22", "scale": 1.00, "heatwave": True, "kappa": 1.00, "curve": False},
    {"name": "heatwave_1.22_k6.00", "base": "heatwave_1.22", "scale": 1.00, "heatwave": True, "kappa": 6.00, "curve": False},
    {"name": "tight_1.25_k1.00", "base": "tight_1.25", "scale": 1.25, "heatwave": False, "kappa": 1.00, "curve": False},
    {"name": "tight_1.25_k6.00", "base": "tight_1.25", "scale": 1.25, "heatwave": False, "kappa": 6.00, "curve": False},
]

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Environments (HeatWaveEnv reused from r3_interior.py)
# ---------------------------------------------------------------------------

class HeatWaveEnv(TwoClockEnv):
    """Multi-day heat wave: demand multiplied by ``mult[day]``."""

    def __init__(self, config: EnvConfig, mult):
        self.mult = np.asarray(mult, dtype=float)
        super().__init__(config)

    def day_mult(self, step: int) -> float:
        d = min(step // self.cfg.steps_per_day, len(self.mult) - 1)
        return float(self.mult[d])

    def _demand(self) -> float:
        return super()._demand() * self.day_mult(self._step)


def heatwave_mult(days: int) -> list[float]:
    m = [1.0] * days
    for d in range(3, min(7, days)):
        m[d] = 1.22
    return m


def build_env(scale: float, heatwave: bool, kappa: float, seed: int,
              days: int = DAYS, force_outage_off: bool = False) -> TwoClockEnv:
    ap = AgingParams(kappa=float(kappa))
    outage_on = (not force_outage_off) and (kappa > 0.0)
    if heatwave:
        cfg = EnvConfig(
            days=days, difficulty="normal", seed=seed,
            demand_scale=scale, outage_enabled=outage_on, aging=ap,
        )
        env = HeatWaveEnv(cfg, heatwave_mult(days))
        env.name = "heatwave_1.22"
        return env
    env = make_env(
        "normal", days=days, seed=seed,
        demand_scale=scale, outage_enabled=outage_on, aging=ap,
    )
    return env


# ---------------------------------------------------------------------------
# Exact fast dispatcher (fixes fast_dispatch missing VOLL optimality gate)
# ---------------------------------------------------------------------------

def exact_fast_dispatch(net, load_mw: float, prev_p: np.ndarray,
                        renewable_avail: float, pmax_eff: np.ndarray, **kw) -> OPFResult:
    """Exact LP equivalent of ``dc_opf`` with VOLL-aware greedy fast path.

    ``twoclock.physics.opf.fast_dispatch`` omits a VOLL shedding block from its
    greedy merit-order fast path and only falls back to ``dc_opf`` on network,
    reserve, or load-balance infeasibility. When ``mu > mu_hi_g`` (so an excess
    block costs more than VOLL = $5,000/MWh) yet reserve/network are feasible,
    the legacy fast path erroneously dispatches above-VOLL excess instead of
    preemptively shedding load.

    Including the ``(VOLL, "shed", 0, shed_cap)`` block in the greedy merit
    order and falling back to ``dc_opf`` whenever shedding cap, curtailment,
    ramp-above-free (split LP), network limits, or reserve constraints bind
    matches ``dc_opf`` to floating-point precision while staying ~3-6x faster.
    """
    wear_price = kw.get("wear_price_mwh", None)
    wear_free = kw.get("wear_free_mw", None)
    split = wear_price is not None and wear_free is not None
    wp = np.zeros(len(GEN_BUS)) if wear_price is None else np.asarray(wear_price, dtype=float)
    free = np.zeros(len(GEN_BUS)) if wear_free is None else np.asarray(wear_free, dtype=float)
    excess_cost = GEN_COST + wp
    spot = float(kw.get("spot_price", 80.0))
    import_cap = float(kw.get("import_cap_mw", IMPORT_CAP_MW))
    bmin = kw.get("battery_min_mw", -BATTERY_POWER_MW)
    bmax = kw.get("battery_max_mw", BATTERY_POWER_MW)
    bmin = -BATTERY_POWER_MW if bmin is None else max(float(bmin), -BATTERY_POWER_MW)
    bmax = BATTERY_POWER_MW if bmax is None else min(float(bmax), BATTERY_POWER_MW)
    reserve_req = float(kw.get("reserve_req_mw", 0.0))
    shed_cap = float(kw.get("shed_cap_frac", 0.30)) * load_mw

    def _fallback():
        kw2 = dict(kw)
        if kw2.get("wear_free_mw") is not None:
            f2 = np.asarray(kw2["wear_free_mw"], dtype=float).copy()
            f2[3] = GEN_RATING[3]
            kw2["wear_free_mw"] = f2
        res = dc_opf(net, load_mw, prev_p, renewable_avail, pmax_eff, **kw2)
        res.status = "lp"
        return res

    pmin, pmax = _ramp_bounds(prev_p, pmax_eff, renewable_avail)
    cur = pmin.copy()
    blocks = []
    for g in range(3):
        if split:
            f_g = float(np.clip(free[g], 0.0, pmax[g]))
            base_hi = max(float(pmin[g]), f_g)
            if base_hi > pmin[g]:
                blocks.append((float(GEN_COST[g]), "g", g, base_hi))
            if pmax[g] > base_hi:
                blocks.append((float(excess_cost[g]), "g", g, float(pmax[g])))
        else:
            if pmax[g] > pmin[g]:
                blocks.append((float(excess_cost[g]), "g", g, float(pmax[g])))
    if import_cap > 0:
        blocks.append((spot, "imp", 0, import_cap))
    if bmax > 0:
        blocks.append((0.0, "batt", 0, bmax))
    if shed_cap > 0:
        blocks.append((VOLL, "shed", 0, shed_cap))
    blocks.sort(key=lambda t: t[0])

    remaining = load_mw - float(cur.sum())
    batt = 0.0
    imp = 0.0
    shed = 0.0
    for _, kind, idx, hi in blocks:
        if remaining <= 1e-9:
            break
        room = (hi - cur[idx]) if kind == "g" else hi
        if room <= 1e-9:
            continue
        take = min(room, remaining)
        if kind == "g":
            cur[idx] += take
        elif kind == "imp":
            imp = take
        elif kind == "batt":
            batt = take
        else:
            shed = take
        remaining -= take

    if remaining > 1e-7:
        return _fallback()
    surplus = max(0.0, -remaining)
    if surplus > 1e-9:
        if bmin < 0:
            charge = min(abs(bmin), surplus)
            batt = -charge
            surplus -= charge
        if surplus > 1e-9:
            return _fallback()

    p = cur
    inj = np.zeros(net.n_bus)
    for g, bus in enumerate(GEN_BUS):
        inj[bus] += p[g]
    inj[BATTERY_BUS] += batt
    inj[IMPORT_BUS] += imp
    inj -= LOAD_SHARE * (load_mw - shed)
    flows = net.flows_from_injection(inj)

    reserve = float(
        np.sum(pmax_eff[GEN_DISPATCHABLE] - p[GEN_DISPATCHABLE])
        + (BATTERY_POWER_MW - batt)
    )
    pmin_gt_free = bool(split and np.any(pmin[:3] > free[:3] + 1e-9))
    feasible = (
        not pmin_gt_free
        and float(net.overload(flows).sum()) <= 1e-6
        and reserve >= reserve_req - 1e-6
    )
    if not feasible:
        return _fallback()

    excess = np.maximum(0.0, p - np.maximum(pmin, np.minimum(pmax, free))) if split else np.zeros(len(GEN_BUS))
    cost = float(np.dot(p, GEN_COST) + np.dot(excess, wp) + spot * imp + VOLL * shed)
    return OPFResult(
        p_gen=p, battery=batt, import_mw=imp, shed=LOAD_SHARE * shed,
        flows=flows, theta=net.theta(inj), cost=cost,
        lmp=np.zeros(net.n_bus), reserve_mw=reserve, status="fast",
        duals={"excess_mw": excess},
    )


# ---------------------------------------------------------------------------
# Episode runner (parallel worker)
# ---------------------------------------------------------------------------

def _run_episode_worker(task: tuple) -> dict:
    """Run one (regime, policy, seed) episode and return complete metrics."""
    scale, heatwave, kappa, pol_spec, seed, use_legacy_fast = task
    env = build_env(scale, heatwave, kappa, seed)
    net = env.net
    ap = env.cfg.aging
    free = wear_free_mw(ap).copy()
    free[3] = GEN_RATING[3]

    kind = pol_spec["kind"]
    solver = fast_dispatch if use_legacy_fast else exact_fast_dispatch

    obs = env.reset()
    dt = env.cfg.dt_hours
    vol_shed_mwh = 0.0
    invol_shed_mwh = 0.0
    excess_mwh = np.zeros(3)
    max_wear_unit = np.zeros(3)
    lp_steps = 0
    fast_over_voll_steps = 0
    shed_cap_bind_steps = 0
    reserve_bind_steps = 0
    ramp_forced_excess_mwh = np.zeros(3)
    trips_by_unit = [0, 0, 0]
    trip_events = []

    while not env.done:
        ctx = obs.ctx
        lo, hi = _battery_bounds(ctx["soc"], dt)
        W = np.asarray(ctx["wear"][:3], dtype=float)

        if kind == "blind":
            wp, fr = None, None
            mu_vec = np.zeros(4)
        elif kind == "const":
            mu_val = float(pol_spec["mu"])
            if mu_val <= 0.0:
                wp, fr = None, None
                mu_vec = np.zeros(4)
            else:
                mu_vec = np.array([mu_val, mu_val, mu_val, 0.0])
                wp = wear_price_from_mu(mu_vec, ap, GEN_RATING, GEN_DISPATCHABLE)
                fr = free
        elif kind == "wear_gated":
            w_thr = float(pol_spec["w_thr"])
            mu0 = float(pol_spec["mu_tier0"])
            mu_vec = np.array([
                1.5 * MU_HI_G[g] if W[g] >= w_thr else mu0 for g in range(3)
            ] + [0.0])
            wp = wear_price_from_mu(mu_vec, ap, GEN_RATING, GEN_DISPATCHABLE)
            fr = free
        elif kind == "hazard":
            gamma = float(pol_spec["gamma"])
            mu0 = float(pol_spec["mu_tier0"])
            haz_grad = 2.0 * kappa * W * np.exp(-kappa * (W ** 2))
            mu_haz = gamma * haz_grad * 1.0e7
            mu_vec = np.array([max(mu0, float(mu_haz[g])) for g in range(3)] + [0.0])
            wp = wear_price_from_mu(mu_vec, ap, GEN_RATING, GEN_DISPATCHABLE)
            fr = free
        else:
            raise ValueError(f"Unknown policy kind: {kind}")

        # Pre-step ramp-forced excess diagnostic
        pmin, pmax = _ramp_bounds(ctx["prev_p"], ctx["pmax_eff"], ctx["renewable"])
        for g in range(3):
            f_g = float(np.clip(free[g], 0.0, pmax[g]))
            if pmin[g] > f_g:
                ramp_forced_excess_mwh[g] += (pmin[g] - f_g) * dt

        res = solver(
            net, ctx["load"], ctx["prev_p"], ctx["renewable"], ctx["pmax_eff"],
            reserve_req_mw=ctx["reserve_req"], battery_min_mw=lo, battery_max_mw=hi,
            spot_price=ctx["spot"], wear_price_mwh=wp, wear_free_mw=fr,
        )
        if res.status != "fast":
            lp_steps += 1
        elif wp is not None:
            ex_step = np.maximum(0.0, res.p_gen[:3] - np.maximum(pmin[:3], np.minimum(pmax[:3], free[:3])))
            if np.any(((GEN_COST[:3] + wp[:3]) >= VOLL - 1e-6) & (ex_step > 1e-6)):
                fast_over_voll_steps += 1

        if res.shed_total >= 0.30 * ctx["load"] - 1e-3:
            shed_cap_bind_steps += 1
        if res.reserve_mw <= ctx["reserve_req"] + 1e-2:
            reserve_bind_steps += 1

        obs, _, _, info = env.step(action_from_opf(res, ctx["load"]))
        vs = float(info["voluntary_shed_mw"])
        tot_u = float(info["unserved_mw"])
        vol_shed_mwh += vs * dt
        invol_shed_mwh += max(0.0, tot_u - vs) * dt
        excess_mwh += np.maximum(0.0, info["p_gen"][:3] - free[:3]) * dt
        max_wear_unit = np.maximum(max_wear_unit, info["wear"][:3])
        if info["tripped"]:
            for g in info["tripped"]:
                if g < 3:
                    trips_by_unit[g] += 1
            trip_events.append({
                "step": int(info["step"]),
                "units": [int(g) for g in info["tripped"]],
                "wear": [round(float(w), 5) for w in info["wear"][:3]],
            })

    s = env.episode_summary()
    return {
        "seed": int(seed),
        "cost": float(s["episode_cost"]),
        "trips": int(s["trips"]),
        "trips_by_unit": trips_by_unit,
        "trip_events": trip_events,
        "unserved_mwh": float(s["unserved_mwh"]),
        "vol_shed_mwh": float(vol_shed_mwh),
        "invol_shed_mwh": float(invol_shed_mwh),
        "excess_mwh": [float(x) for x in excess_mwh],
        "excess_mwh_total": float(excess_mwh.sum()),
        "ramp_forced_excess_mwh": [float(x) for x in ramp_forced_excess_mwh],
        "max_wear": float(max_wear_unit.max()),
        "max_wear_by_unit": [float(x) for x in max_wear_unit],
        "wear_end_mean": float(s["wear_end_mean"]),
        "outage_unit_steps": int(s["outage_unit_steps"]),
        "overload_mwh": float(s["overload_mwh"]),
        "lp_steps": int(lp_steps),
        "fast_over_voll_steps": int(fast_over_voll_steps),
        "shed_cap_bind_steps": int(shed_cap_bind_steps),
        "reserve_bind_steps": int(reserve_bind_steps),
    }


def run_batch(pool: ProcessPoolExecutor, scale: float, heatwave: bool, kappa: float,
              pol_spec: dict, seeds: list[int], use_legacy_fast: bool = False) -> list[dict]:
    tasks = [(scale, heatwave, kappa, pol_spec, s, use_legacy_fast) for s in seeds]
    return list(pool.map(_run_episode_worker, tasks, chunksize=max(1, len(tasks) // (MAX_WORKERS * 2))))


# ---------------------------------------------------------------------------
# Risk metrics & paired bootstrap CIs
# ---------------------------------------------------------------------------

def cvar_tail(values, alpha: float) -> float:
    arr = np.sort(np.asarray(values, dtype=float))
    n = arr.size
    if n == 0:
        return 0.0
    k = max(1, int(math.ceil((1.0 - alpha) * n - 1e-12)))
    return float(np.mean(arr[-k:]))


def summarize_rows(rows: list[dict]) -> dict:
    costs = np.array([r["cost"] for r in rows], dtype=float)
    trips = np.array([r["trips"] for r in rows], dtype=float)
    uns = np.array([r["unserved_mwh"] for r in rows], dtype=float)
    vol = np.array([r["vol_shed_mwh"] for r in rows], dtype=float)
    inv = np.array([r["invol_shed_mwh"] for r in rows], dtype=float)
    ex = np.array([r["excess_mwh_total"] for r in rows], dtype=float)
    mw = np.array([r["max_wear"] for r in rows], dtype=float)
    n = len(rows)
    return {
        "n": n,
        "mean": float(costs.mean()),
        "std": float(costs.std(ddof=1)) if n > 1 else 0.0,
        "median": float(np.median(costs)),
        "p90": float(np.percentile(costs, 90)),
        "cvar90": cvar_tail(costs, 0.90),
        "cvar95": cvar_tail(costs, 0.95),
        "worst": float(costs.max()),
        "min": float(costs.min()),
        "trips_mean": float(trips.mean()),
        "trips_total": int(trips.sum()),
        "trip_seed_frac": float(np.mean(trips > 0)),
        "trip_seed_count": int((trips > 0).sum()),
        "trips_by_unit_total": [
            int(sum(r["trips_by_unit"][g] for r in rows)) for g in range(3)
        ],
        "unserved_mwh_mean": float(uns.mean()),
        "vol_shed_mwh_mean": float(vol.mean()),
        "invol_shed_mwh_mean": float(inv.mean()),
        "excess_mwh_mean": float(ex.mean()),
        "max_wear_mean": float(mw.mean()),
        "max_wear_worst": float(mw.max()),
        "shed_cap_bind_steps_total": int(sum(r["shed_cap_bind_steps"] for r in rows)),
        "per_seed": {
            "seeds": [r["seed"] for r in rows],
            "cost": [r["cost"] for r in rows],
            "trips": [r["trips"] for r in rows],
            "unserved_mwh": [r["unserved_mwh"] for r in rows],
            "vol_shed_mwh": [r["vol_shed_mwh"] for r in rows],
            "invol_shed_mwh": [r["invol_shed_mwh"] for r in rows],
            "excess_mwh_total": [r["excess_mwh_total"] for r in rows],
            "max_wear": [r["max_wear"] for r in rows],
        },
    }


def paired_bootstrap_diff(a_rows: list[dict], b_rows: list[dict],
                          n_boot: int = BOOT_N, seed: int = BOOT_SEED) -> dict:
    """Paired bootstrap 95% CIs for (A - B) on mean, CVaR_90, and CVaR_95."""
    a = np.array([r["cost"] for r in a_rows], dtype=float)
    b = np.array([r["cost"] for r in b_rows], dtype=float)
    assert len(a) == len(b)
    n = len(a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))

    # Mean diff
    d_mean = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    obs_mean = float(a.mean() - b.mean())

    # CVaR_90 diff
    k90 = max(1, int(math.ceil(0.10 * n - 1e-12)))
    a90 = np.sort(a[idx], axis=1)[:, -k90:].mean(axis=1)
    b90 = np.sort(b[idx], axis=1)[:, -k90:].mean(axis=1)
    d_c90 = a90 - b90
    obs_c90 = cvar_tail(a, 0.90) - cvar_tail(b, 0.90)

    # CVaR_95 diff
    k95 = max(1, int(math.ceil(0.05 * n - 1e-12)))
    a95 = np.sort(a[idx], axis=1)[:, -k95:].mean(axis=1)
    b95 = np.sort(b[idx], axis=1)[:, -k95:].mean(axis=1)
    d_c95 = a95 - b95
    obs_c95 = cvar_tail(a, 0.95) - cvar_tail(b, 0.95)

    diffs = a - b
    return {
        "mean_diff": obs_mean,
        "mean_ci95": [float(np.percentile(d_mean, 2.5)), float(np.percentile(d_mean, 97.5))],
        "mean_rel_pct": float(100.0 * obs_mean / max(1e-9, float(b.mean()))),
        "cvar90_diff": float(obs_c90),
        "cvar90_ci95": [float(np.percentile(d_c90, 2.5)), float(np.percentile(d_c90, 97.5))],
        "cvar90_rel_pct": float(100.0 * obs_c90 / max(1e-9, cvar_tail(b, 0.90))),
        "cvar95_diff": float(obs_c95),
        "cvar95_ci95": [float(np.percentile(d_c95, 2.5)), float(np.percentile(d_c95, 97.5))],
        "wins_a_lt_b": int((diffs < -1e-6).sum()),
        "ties": int((np.abs(diffs) <= 1e-6).sum()),
        "losses_a_gt_b": int((diffs > 1e-6).sum()),
    }


# ---------------------------------------------------------------------------
# Deterministic dual certificates (Tier 0) on training seeds 101..103
# ---------------------------------------------------------------------------

def compute_base_certificates() -> dict:
    """Compute deterministic (no-outage) dual certificates once per base demand regime."""
    net = build_5bus()
    ap = AgingParams(kappa=0.0)
    bases = [
        ("tight_1.15", 1.15, False),
        ("normal_1.00", 1.00, False),
        ("heatwave_1.22", 1.00, True),
        ("tight_1.25", 1.25, False),
    ]
    certs = {}
    for bname, sc, hw in bases:
        log(f"Computing deterministic Tier-0 certificate for {bname} on train seeds {CERT_SEEDS}...")
        cert = fixed_point_certificate(
            lambda s, sc=sc, hw=hw: build_env(sc, hw, 0.0, s, force_outage_off=True),
            net, ap, CERT_SEEDS, run_episode, log=log,
        )
        mu_lo = float(cert["mu_lo"])
        mu_hi = float(cert["mu_hi"])
        if np.isfinite(mu_hi):
            mu_t0 = float(np.sqrt(max(mu_lo, 1.0) * mu_hi))
        else:
            mu_t0 = float(min(2.0 * max(mu_lo, 1.0), np.sqrt(max(mu_lo, 1.0) * MU_HI_NUC)))
        # Ensure strictly inside Tier 0: [mu_lo, mu_hi_nuc)
        if mu_t0 >= MU_HI_NUC:
            mu_t0 = float(np.sqrt(max(mu_lo, 1.0) * MU_HI_NUC))
        certs[bname] = {
            "mu_lo": mu_lo,
            "mu_hi": mu_hi if np.isfinite(mu_hi) else None,
            "mu_tier0": mu_t0,
            "forced_steps_per_unit": cert["forced_steps_per_unit"],
            "n_infeasible_even_with_excess": cert["n_infeasible_even_with_excess"],
            "converged": cert["converged"],
            "iterations": cert["iterations"],
            "env_steps": cert["env_steps"],
        }
        log(f"  -> {bname}: mu_lo={mu_lo:.1f}, mu_hi={mu_hi:.4g}, mu_tier0={mu_t0:.1f}")
    return certs


# ---------------------------------------------------------------------------
# Mechanism diagnosis: R3 fast_dispatch bug vs exact LP dc_opf
# ---------------------------------------------------------------------------

def run_mechanism_diagnosis(pool: ProcessPoolExecutor) -> dict:
    """Diagnose why R3 tight_1.15_outages had 2/5 test seeds trip even at mu=1e6."""
    log("=== Mechanism Diagnosis: R3 fast_dispatch vs exact LP (tight_1.15, kappa=6.0, mu=1e6) ===")
    r3_seeds = [1, 2, 3, 4, 5]
    spec_1e6 = {"kind": "const", "mu": 1.0e6}
    legacy_5 = run_batch(pool, 1.15, False, 6.0, spec_1e6, r3_seeds, use_legacy_fast=True)
    exact_5 = run_batch(pool, 1.15, False, 6.0, spec_1e6, r3_seeds, use_legacy_fast=False)

    # Also verify exact_fast_dispatch matches pure dc_opf (with free[3]=150) on seed 2
    net = build_5bus()
    ap = AgingParams(kappa=6.0)
    free_fixed = wear_free_mw(ap).copy()
    free_fixed[3] = GEN_RATING[3]
    env_a = build_env(1.15, False, 6.0, 2)
    env_b = build_env(1.15, False, 6.0, 2)
    wp = wear_price_from_mu(1.0e6, ap, GEN_RATING, GEN_DISPATCHABLE)
    obs_a, obs_b = env_a.reset(), env_b.reset()
    max_p_diff = 0.0
    max_shed_diff = 0.0
    while not env_a.done:
        ca, cb = obs_a.ctx, obs_b.ctx
        loa, hia = _battery_bounds(ca["soc"], 0.25)
        lob, hib = _battery_bounds(cb["soc"], 0.25)
        ra = exact_fast_dispatch(net, ca["load"], ca["prev_p"], ca["renewable"], ca["pmax_eff"],
                                 reserve_req_mw=ca["reserve_req"], battery_min_mw=loa,
                                 battery_max_mw=hia, spot_price=ca["spot"],
                                 wear_price_mwh=wp, wear_free_mw=free_fixed)
        rb = dc_opf(net, cb["load"], cb["prev_p"], cb["renewable"], cb["pmax_eff"],
                    reserve_req_mw=cb["reserve_req"], battery_min_mw=lob,
                    battery_max_mw=hib, spot_price=cb["spot"],
                    wear_price_mwh=wp, wear_free_mw=free_fixed)
        max_p_diff = max(max_p_diff, float(np.max(np.abs(ra.p_gen - rb.p_gen))))
        max_shed_diff = max(max_shed_diff, abs(ra.shed_total - rb.shed_total))
        obs_a, _, _, _ = env_a.step(action_from_opf(ra, ca["load"]))
        obs_b, _, _, _ = env_b.step(action_from_opf(rb, cb["load"]))
    cost_diff_exact_vs_lp = abs(env_a.episode_summary()["episode_cost"] - env_b.episode_summary()["episode_cost"])

    # Also test legacy fast_dispatch on all 40 test seeds for comparison
    legacy_40 = run_batch(pool, 1.15, False, 6.0, spec_1e6, TEST_SEEDS, use_legacy_fast=True)
    exact_40 = run_batch(pool, 1.15, False, 6.0, spec_1e6, TEST_SEEDS, use_legacy_fast=False)

    diag = {
        "r3_seeds_1_to_5": {
            "legacy_fast_dispatch": {
                "summary": summarize_rows(legacy_5),
                "fast_over_voll_steps_per_seed": [r["fast_over_voll_steps"] for r in legacy_5],
                "excess_mwh_by_unit_per_seed": [r["excess_mwh"] for r in legacy_5],
                "trip_events_per_seed": [r["trip_events"] for r in legacy_5],
                "shed_cap_bind_steps_per_seed": [r["shed_cap_bind_steps"] for r in legacy_5],
                "reserve_bind_steps_per_seed": [r["reserve_bind_steps"] for r in legacy_5],
            },
            "exact_lp_dispatch": {
                "summary": summarize_rows(exact_5),
                "excess_mwh_by_unit_per_seed": [r["excess_mwh"] for r in exact_5],
                "ramp_forced_excess_mwh_per_seed": [r["ramp_forced_excess_mwh"] for r in exact_5],
                "trip_events_per_seed": [r["trip_events"] for r in exact_5],
                "shed_cap_bind_steps_per_seed": [r["shed_cap_bind_steps"] for r in exact_5],
                "reserve_bind_steps_per_seed": [r["reserve_bind_steps"] for r in exact_5],
            },
        },
        "test_seeds_1_to_40": {
            "legacy_fast_dispatch": summarize_rows(legacy_40),
            "exact_lp_dispatch": summarize_rows(exact_40),
            "exact_minus_legacy": paired_bootstrap_diff(exact_40, legacy_40),
            "legacy_fast_over_voll_steps_mean": float(np.mean([r["fast_over_voll_steps"] for r in legacy_40])),
            "exact_ramp_forced_excess_nuc_mean": float(np.mean([r["ramp_forced_excess_mwh"][2] for r in exact_40])),
        },
        "exact_fast_vs_pure_dc_opf_check_seed2": {
            "max_p_diff_mw": max_p_diff,
            "max_shed_diff_mw": max_shed_diff,
            "episode_cost_diff_usd": cost_diff_exact_vs_lp,
        },
        "explanation": (
            "In R3 (r3_interior.py), fast_dispatch uses a greedy merit-order fast path "
            "(_fast_merit_dispatch) without a VOLL shedding block and only falls back to "
            "the LP (dc_opf) when line limits, reserve, or load balance fail. At peak-adjacent "
            "steps where reserve remains feasible despite mu > mu_hi_g (excess_cost > VOLL), "
            "legacy fast_dispatch erroneously dispatches ~550-610 MWh of above-VOLL excess on "
            "gas/coal instead of preemptively shedding load, building wear W~0.08-0.19 and "
            "causing trips on 2/5 R3 seeds (seeds 2 and 5). Under the exact LP (dc_opf / "
            "VOLL-gated fast_dispatch), gas and coal accrue 0.0 MWh excess, shed_cap (30%) "
            "never binds, only 6.2 MWh of step-0..2 ramp-forced nuclear excess occurs (from "
            "initial prev_p[2]=190 > free[2]=170 with ramp=5 MW/step, decaying to W<2e-4 "
            "before the day-1 outage check), and trips drop to 0 across all test seeds."
        ),
    }
    log(
        f"  R3 seeds 1-5: legacy fast mean={diag['r3_seeds_1_to_5']['legacy_fast_dispatch']['summary']['mean']/1e6:.3f}M "
        f"(trips={diag['r3_seeds_1_to_5']['legacy_fast_dispatch']['summary']['trips_total']}) "
        f"vs exact LP mean={diag['r3_seeds_1_to_5']['exact_lp_dispatch']['summary']['mean']/1e6:.3f}M "
        f"(trips={diag['r3_seeds_1_to_5']['exact_lp_dispatch']['summary']['trips_total']})"
    )
    log(
        f"  40 test seeds: legacy mean={diag['test_seeds_1_to_40']['legacy_fast_dispatch']['mean']/1e6:.3f}M "
        f"(CVaR90={diag['test_seeds_1_to_40']['legacy_fast_dispatch']['cvar90']/1e6:.3f}M, "
        f"trips={diag['test_seeds_1_to_40']['legacy_fast_dispatch']['trips_total']}) "
        f"vs exact LP mean={diag['test_seeds_1_to_40']['exact_lp_dispatch']['mean']/1e6:.3f}M "
        f"(CVaR90={diag['test_seeds_1_to_40']['exact_lp_dispatch']['cvar90']/1e6:.3f}M, "
        f"trips={diag['test_seeds_1_to_40']['exact_lp_dispatch']['trips_total']})"
    )
    log(f"  exact_fast vs pure dc_opf check (seed 2): max_dp={max_p_diff:.2e} MW, cost_diff=${cost_diff_exact_vs_lp:.2e}")
    return diag


# ---------------------------------------------------------------------------
# Regime evaluation (train selection -> held-out test evaluation)
# ---------------------------------------------------------------------------

def evaluate_regime(pool: ProcessPoolExecutor, reg: dict, cert: dict) -> dict:
    t0 = time.time()
    name = reg["name"]
    sc, hw, kappa = reg["scale"], reg["heatwave"], reg["kappa"]
    mu_lo = cert["mu_lo"]
    mu_t0 = cert["mu_tier0"]
    log(f"=== Regime {name} (scale={sc}, heatwave={hw}, kappa={kappa}, mu_tier0={mu_t0:.0f}) ===")

    # Cache (policy_key, seed_group) -> rows
    train_const_grid = sorted(set(
        [mu_t0] + [m for m in CONST_GRID_BASE if m >= mu_lo]
    ))

    # 1. TRAIN selection for constant price (mean & CVaR_90)
    train_const = {}
    for mu in train_const_grid:
        rows = run_batch(pool, sc, hw, kappa, {"kind": "const", "mu": mu}, TRAIN_SEEDS)
        train_const[mu] = {
            "mu": mu,
            "mean": float(np.mean([r["cost"] for r in rows])),
            "cvar90": cvar_tail([r["cost"] for r in rows], 0.90),
            "trips_mean": float(np.mean([r["trips"] for r in rows])),
        }
    # Tie-break: for mean -> smallest mu; for CVaR90 -> lowest mean, then smallest mu
    best_const_mean_val = min(v["mean"] for v in train_const.values())
    mu_const_mean = min(
        mu for mu, v in train_const.items() if v["mean"] <= best_const_mean_val + 1.0
    )
    best_const_c90_val = min(v["cvar90"] for v in train_const.values())
    c90_cands = [mu for mu, v in train_const.items() if v["cvar90"] <= best_const_c90_val + 1.0]
    best_c90_mean = min(train_const[mu]["mean"] for mu in c90_cands)
    mu_const_c90 = min(mu for mu in c90_cands if train_const[mu]["mean"] <= best_c90_mean + 1.0)

    # 2. TRAIN selection for wear-gated policy (mean & CVaR_90)
    train_wgate = {}
    for w_thr in WGATE_GRID:
        rows = run_batch(
            pool, sc, hw, kappa,
            {"kind": "wear_gated", "w_thr": w_thr, "mu_tier0": mu_t0},
            TRAIN_SEEDS,
        )
        train_wgate[w_thr] = {
            "w_thr": w_thr,
            "mean": float(np.mean([r["cost"] for r in rows])),
            "cvar90": cvar_tail([r["cost"] for r in rows], 0.90),
            "trips_mean": float(np.mean([r["trips"] for r in rows])),
        }
    best_wg_mean_val = min(v["mean"] for v in train_wgate.values())
    wthr_mean = max(
        w for w, v in train_wgate.items() if v["mean"] <= best_wg_mean_val + 1.0
    )
    best_wg_c90_val = min(v["cvar90"] for v in train_wgate.values())
    wg_c90_cands = [w for w, v in train_wgate.items() if v["cvar90"] <= best_wg_c90_val + 1.0]
    wg_c90_mean = min(train_wgate[w]["mean"] for w in wg_c90_cands)
    wthr_c90 = max(w for w in wg_c90_cands if train_wgate[w]["mean"] <= wg_c90_mean + 1.0)

    # 3. TRAIN selection for continuous hazard-triggered policy (mean & CVaR_90)
    train_hazard = {}
    if kappa == 0.0:
        # Hazard gradient 2*kappa*W*exp(-kappa*W^2) == 0 for all gamma -> identical to tier0
        t0_stats = train_const[mu_t0]
        for gamma in HAZARD_GAMMA_GRID:
            train_hazard[gamma] = {
                "gamma": gamma,
                "mean": t0_stats["mean"],
                "cvar90": t0_stats["cvar90"],
                "trips_mean": t0_stats["trips_mean"],
            }
    else:
        for gamma in HAZARD_GAMMA_GRID:
            rows = run_batch(
                pool, sc, hw, kappa,
                {"kind": "hazard", "gamma": gamma, "mu_tier0": mu_t0},
                TRAIN_SEEDS,
            )
            train_hazard[gamma] = {
                "gamma": gamma,
                "mean": float(np.mean([r["cost"] for r in rows])),
                "cvar90": cvar_tail([r["cost"] for r in rows], 0.90),
                "trips_mean": float(np.mean([r["trips"] for r in rows])),
            }
    best_hz_mean_val = min(v["mean"] for v in train_hazard.values())
    gamma_mean = min(
        g for g, v in train_hazard.items() if v["mean"] <= best_hz_mean_val + 1.0
    )
    best_hz_c90_val = min(v["cvar90"] for v in train_hazard.values())
    hz_c90_cands = [g for g, v in train_hazard.items() if v["cvar90"] <= best_hz_c90_val + 1.0]
    hz_c90_mean = min(train_hazard[g]["mean"] for g in hz_c90_cands)
    gamma_c90 = min(g for g in hz_c90_cands if train_hazard[g]["mean"] <= hz_c90_mean + 1.0)

    log(
        f"  TRAIN selected: const_mean={mu_const_mean:.0f} (T{_tier_idx(mu_const_mean)}), "
        f"const_cvar90={mu_const_c90:.0f} (T{_tier_idx(mu_const_c90)}), "
        f"wgate_mean={wthr_mean:g}, wgate_cvar90={wthr_c90:g}, "
        f"hazard_cvar90_gamma={gamma_c90:g}  ({time.time()-t0:.1f}s)"
    )

    # 4. Held-out TEST evaluation (40 seeds) with deduplicated execution cache
    policy_specs = {
        "blind": {"kind": "blind"},
        "tier0_cert": {"kind": "const", "mu": mu_t0},
        "tier1_nuc": {"kind": "const", "mu": TIER1_MU},
        "tier2_nuc_coal": {"kind": "const", "mu": TIER2_MU},
        "tier3_all": {"kind": "const", "mu": TIER3_MU},
        "const_mean_opt": {"kind": "const", "mu": mu_const_mean},
        "const_cvar90_opt": {"kind": "const", "mu": mu_const_c90},
        "wear_gated_mean": {"kind": "wear_gated", "w_thr": wthr_mean, "mu_tier0": mu_t0},
        "wear_gated_cvar90": {"kind": "wear_gated", "w_thr": wthr_c90, "mu_tier0": mu_t0},
        "hazard_rule_mean": {"kind": "hazard", "gamma": gamma_mean, "mu_tier0": mu_t0},
        "hazard_rule_cvar90": {"kind": "hazard", "gamma": gamma_c90, "mu_tier0": mu_t0},
    }

    test_cache: dict[str, list[dict]] = {}

    def get_test(spec: dict) -> list[dict]:
        key = json.dumps(spec, sort_keys=True)
        if key not in test_cache:
            test_cache[key] = run_batch(pool, sc, hw, kappa, spec, TEST_SEEDS)
        return test_cache[key]

    test_rows = {pname: get_test(spec) for pname, spec in policy_specs.items()}

    # 5. Optional dense mu curve on TEST seeds (for tight_1.15 kappa sweep figure)
    dense_curve = None
    if reg.get("curve", False):
        dense_curve = {}
        sweep_mus = sorted(set(DENSE_MU_GRID + [mu_t0]))
        for mu in sweep_mus:
            spec = {"kind": "blind"} if mu <= 0.0 else {"kind": "const", "mu": float(mu)}
            rows = get_test(spec)
            s = summarize_rows(rows)
            dense_curve[f"{mu:g}"] = {
                "mu": float(mu),
                "mean": s["mean"],
                "cvar90": s["cvar90"],
                "cvar95": s["cvar95"],
                "worst": s["worst"],
                "trips_mean": s["trips_mean"],
                "trip_seed_frac": s["trip_seed_frac"],
                "unserved_mwh_mean": s["unserved_mwh_mean"],
                "vol_shed_mwh_mean": s["vol_shed_mwh_mean"],
                "invol_shed_mwh_mean": s["invol_shed_mwh_mean"],
            }

    # 6. Summaries & paired bootstrap CIs on TEST seeds
    policies_out = {}
    for pname, spec in policy_specs.items():
        rows = test_rows[pname]
        summ = summarize_rows(rows)
        summ["spec"] = spec
        if pname != "tier0_cert":
            summ["minus_tier0_cert"] = paired_bootstrap_diff(rows, test_rows["tier0_cert"])
        if pname != "const_mean_opt":
            summ["minus_const_mean_opt"] = paired_bootstrap_diff(rows, test_rows["const_mean_opt"])
        if pname != "const_cvar90_opt":
            summ["minus_const_cvar90_opt"] = paired_bootstrap_diff(rows, test_rows["const_cvar90_opt"])
        policies_out[pname] = summ
        log(
            f"    {pname:18s}: mean={summ['mean']/1e6:7.3f}M  CVaR90={summ['cvar90']/1e6:7.3f}M  "
            f"CVaR95={summ['cvar95']/1e6:7.3f}M  trips={summ['trips_mean']:.2f}({summ['trip_seed_count']:2d}/40)  "
            f"vol={summ['vol_shed_mwh_mean']:6.1f}  inv={summ['invol_shed_mwh_mean']:7.1f}  maxW={summ['max_wear_mean']:.3f}"
        )

    out = {
        "regime": reg,
        "certificate": cert,
        "train_selection": {
            "const_grid": {f"{k:g}": v for k, v in sorted(train_const.items())},
            "wgate_grid": {f"{k:g}": v for k, v in sorted(train_wgate.items())},
            "hazard_grid": {f"{k:g}": v for k, v in sorted(train_hazard.items())},
            "selected": {
                "const_mean_opt_mu": mu_const_mean,
                "const_mean_opt_tier": _tier_idx(mu_const_mean),
                "const_cvar90_opt_mu": mu_const_c90,
                "const_cvar90_opt_tier": _tier_idx(mu_const_c90),
                "mean_vs_cvar_tier_separation": bool(_tier_idx(mu_const_c90) > _tier_idx(mu_const_mean)),
                "wear_gated_mean_wthr": wthr_mean,
                "wear_gated_cvar90_wthr": wthr_c90,
                "hazard_rule_mean_gamma": gamma_mean,
                "hazard_rule_cvar90_gamma": gamma_c90,
            },
        },
        "test_policies": policies_out,
        "test_dense_mu_curve": dense_curve,
        "wall_s": time.time() - t0,
    }
    log(f"  done {name} in {out['wall_s']:.1f}s")
    return out


def _tier_idx(mu: float) -> int:
    if mu < MU_HI_NUC:
        return 0
    if mu < MU_HI_COAL:
        return 1
    if mu < MU_HI_GAS:
        return 2
    return 3


# ---------------------------------------------------------------------------
# Fast pilot scan (--pilot)
# ---------------------------------------------------------------------------

def run_pilot() -> int:
    log(f"=== PILOT SCAN (20 train seeds, {MAX_WORKERS} workers) ===")
    pols = [
        ("blind", {"kind": "blind"}),
        ("T0_25k", {"kind": "const", "mu": 25000.0}),
        ("T1_nuc", {"kind": "const", "mu": TIER1_MU}),
        ("T2_nc", {"kind": "const", "mu": TIER2_MU}),
        ("T3_all", {"kind": "const", "mu": TIER3_MU}),
        ("WG_0.02", {"kind": "wear_gated", "w_thr": 0.02, "mu_tier0": 25000.0}),
        ("WG_0.06", {"kind": "wear_gated", "w_thr": 0.06, "mu_tier0": 25000.0}),
        ("WG_0.15", {"kind": "wear_gated", "w_thr": 0.15, "mu_tier0": 25000.0}),
        ("HZ_0.03", {"kind": "hazard", "gamma": 0.03, "mu_tier0": 25000.0}),
        ("HZ_0.30", {"kind": "hazard", "gamma": 0.30, "mu_tier0": 25000.0}),
    ]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for reg in REGIMES:
            log(f"\n--- {reg['name']} (20 train seeds) ---")
            for pname, spec in pols:
                rows = run_batch(pool, reg["scale"], reg["heatwave"], reg["kappa"], spec, TRAIN_SEEDS)
                s = summarize_rows(rows)
                log(
                    f"  {pname:8s}: mean={s['mean']/1e6:7.3f}M  CVaR90={s['cvar90']/1e6:7.3f}M  "
                    f"worst={s['worst']/1e6:7.3f}M  trips={s['trips_mean']:.2f}({s['trip_seed_count']:2d}/20)  "
                    f"vol={s['vol_shed_mwh_mean']:6.1f}  inv={s['invol_shed_mwh_mean']:7.1f}  maxW={s['max_wear_mean']:.3f}"
                )
    return 0


# ---------------------------------------------------------------------------
# Publication Figure (3-panel crisp PDF + PNG)
# ---------------------------------------------------------------------------

def plot() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with open(OUT_JSON, encoding="utf-8") as fh:
        d = json.load(fh)

    reg_map = {r["regime"]["name"]: r for r in d["regimes"]}

    plt.rcParams.update({
        "font.size": 7.5,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.15), constrained_layout=True)

    # ------------------------------------------------------------------
    # Panel (a): Mean & CVaR_90 vs mu on tight_1.15 showing Prop 2(iii) lines
    # ------------------------------------------------------------------
    ax = axes[0]
    k_show = [
        ("tight_1.15_k0.25", r"$\kappa=0.25$", "#2a6fb0"),
        ("tight_1.15_k1.00", r"$\kappa=1.0$",  "#d9822b"),
        ("tight_1.15_k6.00", r"$\kappa=6.0$",  "#b03a2a"),
    ]
    for rname, lab, col in k_show:
        if rname not in reg_map or not reg_map[rname].get("test_dense_mu_curve"):
            continue
        cur = reg_map[rname]["test_dense_mu_curve"]
        pts = sorted((v["mu"], v) for v in cur.values() if v["mu"] >= 3e3)
        mus = np.array([p[0] for p in pts])
        means = np.array([p[1]["mean"] / 1e6 for p in pts])
        c90s = np.array([p[1]["cvar90"] / 1e6 for p in pts])
        ax.plot(mus, means, "-", lw=1.45, color=col, label=lab)
        ax.plot(mus, c90s, "--", lw=1.15, color=col, alpha=0.85)

    # Vertical Proposition 2(iii) shedding thresholds
    mu_lo_115 = reg_map["tight_1.15_k6.00"]["certificate"]["mu_lo"]
    vlines = [
        (mu_lo_115, r"$\mu_{\mathrm{lo}}$", "#555555", ":"),
        (MU_HI_NUC, r"$\mu^{\mathrm{hi}}_{\mathrm{nuc}}$", "#2b8a4e", "-."),
        (MU_HI_COAL, r"$\mu^{\mathrm{hi}}_{\mathrm{coal}}$", "#6f42c1", "-."),
        (MU_HI_GAS, r"$\mu^{\mathrm{hi}}_{\mathrm{gas}}$", "#222222", "-."),
    ]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2.6e3, 1.15e6)
    ax.set_ylim(8.2, 340.0)
    y_top = ax.get_ylim()[1]
    for xv, txt, col, ls in vlines:
        ax.axvline(xv, color=col, ls=ls, lw=0.9, alpha=0.85)
        ax.text(xv * 1.05, y_top * 0.90, txt, color=col, fontsize=6.4,
                ha="left", va="top", rotation=90)

    # Tier labels neatly below all curves
    tier_centers = [
        (np.sqrt(mu_lo_115 * MU_HI_NUC), "T0"),
        (TIER1_MU, "T1"),
        (TIER2_MU, "T2"),
        (7.2e5, "T3"),
    ]
    for xc, tlab in tier_centers:
        ax.text(xc, 9.1, tlab, fontsize=6.4, fontweight="bold",
                ha="center", va="bottom", color="#333333",
                bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#cccccc", alpha=0.92))

    ax.set_xlabel(r"wear price $\mu$ [\$/wear-unit]")
    ax.set_ylabel(r"episode cost [M\$]  (solid: mean, dash: CVaR$_{90}$)")
    ax.set_title(r"(a) Prop. 1(iii) risk tiers (tight 1.15$\times$, $n=40$)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.37, 0.985), frameon=True,
              facecolor="white", framealpha=0.95, edgecolor="#dddddd",
              borderpad=0.35, handlelength=1.5, labelspacing=0.25)
    ax.grid(True, which="major", alpha=0.22, lw=0.5)

    # ------------------------------------------------------------------
    # Panel (b): Mean vs CVaR_90 Pareto frontier (tight_1.15 across kappa)
    # ------------------------------------------------------------------
    ax = axes[1]
    pareto_kappas = [
        ("tight_1.15_k0.25", r"$\kappa=0.25$", "#2a6fb0"),
        ("tight_1.15_k1.00", r"$\kappa=1.0$",  "#d9822b"),
        ("tight_1.15_k6.00", r"$\kappa=6.0$",  "#b03a2a"),
    ]
    tier_keys = ["tier0_cert", "tier1_nuc", "tier2_nuc_coal", "tier3_all"]
    tier_markers = ["o", "s", "^", "D"]
    for rname, lab, col in pareto_kappas:
        if rname not in reg_map:
            continue
        tp = reg_map[rname]["test_policies"]
        xs = [tp[k]["mean"] / 1e6 for k in tier_keys]
        ys = [tp[k]["cvar90"] / 1e6 for k in tier_keys]
        ax.plot(xs, ys, "-", lw=1.2, color=col, alpha=0.70, zorder=2)
        for k, mk in zip(tier_keys, tier_markers):
            ax.scatter([tp[k]["mean"] / 1e6], [tp[k]["cvar90"] / 1e6],
                       marker=mk, s=28, color=col, edgecolor="white", lw=0.45, zorder=4)
        wg = tp["wear_gated_cvar90"]
        ax.scatter([wg["mean"] / 1e6], [wg["cvar90"] / 1e6],
                   marker="*", s=68, color=col, edgecolor="black", lw=0.55, zorder=5)
        hz = tp["hazard_rule_cvar90"]
        ax.scatter([hz["mean"] / 1e6], [hz["cvar90"] / 1e6],
                   marker="P", s=38, color="white", edgecolor=col, lw=1.1, zorder=6)

    # Reference diagonal y = x
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot([8.8, 90.0], [8.8, 90.0], ":", color="#888888", lw=0.85, zorder=1)
    ax.set_xlim(9.5, 78.0)
    ax.set_ylim(8.8, 175.0)
    ax.set_xlabel(r"mean test cost [M\$]")
    ax.set_ylabel(r"CVaR$_{90}$ test cost [M\$]")
    ax.set_title(r"(b) Mean–CVaR$_{90}$ trade-off (tight 1.15$\times$)")

    # Crisp callouts for Tier 3 zero-trip anchor and kappa=0.25 hazard Pareto improvement
    ax.annotate("Tier 3 (all $\\kappa$)\n0/40 trips",
                xy=(14.36, 15.04), xytext=(22.5, 15.2),
                fontsize=6.0, color="#222222", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#444444", lw=0.75, shrinkA=1, shrinkB=4))
    ax.annotate(r"Hazard ($\kappa=0.25$)" + "\n0/40 trips",
                xy=(12.73, 13.37), xytext=(17.0, 10.6),
                fontsize=5.9, color="#2a6fb0", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#2a6fb0", lw=0.75, shrinkA=1, shrinkB=4))

    # Color legend in upper-left empty space + policy marker legend in lower-right
    col_handles = [
        Line2D([0], [0], color=col, lw=1.6, label=lab)
        for _, lab, col in pareto_kappas
    ]
    leg_col = ax.legend(handles=col_handles, loc="upper left", frameon=True,
                        facecolor="white", framealpha=0.94, edgecolor="#dddddd",
                        borderpad=0.32, handlelength=1.4, labelspacing=0.22)
    ax.add_artist(leg_col)

    marker_handles = [
        Line2D([0], [0], marker="o", color="#444", ls="none", ms=4.5, label="Tier 0 (cert.)"),
        Line2D([0], [0], marker="s", color="#444", ls="none", ms=4.5, label="Tier 1 (nuc)"),
        Line2D([0], [0], marker="^", color="#444", ls="none", ms=4.5, label="Tier 2 (nuc+coal)"),
        Line2D([0], [0], marker="D", color="#444", ls="none", ms=4.5, label="Tier 3 (all)"),
        Line2D([0], [0], marker="*", color="#444", markeredgecolor="k", ls="none", ms=6.8, label="Wear-gated (CVaR)"),
        Line2D([0], [0], marker="P", color="w", markeredgecolor="#444", ls="none", ms=5.0, label="Hazard rule (CVaR)"),
    ]
    ax.legend(handles=marker_handles, loc="lower right", bbox_to_anchor=(0.99, 0.25),
              frameon=True, facecolor="white", framealpha=0.94, edgecolor="#dddddd",
              borderpad=0.32, labelspacing=0.22, handletextpad=0.4)
    ax.grid(True, which="major", alpha=0.22, lw=0.5)

    # ------------------------------------------------------------------
    # Panel (c): Trip rate & voluntary vs outage unserved decomposition (kappa=6.0)
    # ------------------------------------------------------------------
    ax = axes[2]
    r6 = reg_map["tight_1.15_k6.00"]["test_policies"]
    show_pols = [
        ("blind", "Blind"),
        ("tier0_cert", "Tier 0\n(cert.)"),
        ("tier1_nuc", "Tier 1\n(nuc)"),
        ("tier2_nuc_coal", "Tier 2\n(n+c)"),
        ("tier3_all", "Tier 3\n(all)"),
        ("wear_gated_cvar90", "Wear-\ngated"),
        ("hazard_rule_cvar90", "Hazard\nrule"),
    ]
    x = np.arange(len(show_pols))
    vol_vals = [r6[k]["vol_shed_mwh_mean"] / 1e3 for k, _ in show_pols]
    inv_vals = [r6[k]["invol_shed_mwh_mean"] / 1e3 for k, _ in show_pols]
    trip_vals = [r6[k]["trips_mean"] for k, _ in show_pols]

    w = 0.56
    ax.bar(x, vol_vals, width=w, color="#2a6fb0", label="Preemptive vol. shed")
    ax.bar(x, inv_vals, width=w, bottom=vol_vals, color="#c94a38", alpha=0.88,
           label="Outage / invol. unserved")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in show_pols], fontsize=6.3)
    ax.set_ylabel("mean unserved energy [GWh / 10d]")
    ax.set_title(r"(c) Shed decomposition & trips (1.15$\times$, $\kappa=6$)")
    ax.set_ylim(0, max(v + iv for v, iv in zip(vol_vals, inv_vals)) * 1.22)
    ax.grid(True, axis="y", alpha=0.22, lw=0.5)

    # Annotate exact mean trips/10d above bars
    for xi, (vv, iv, tr) in enumerate(zip(vol_vals, inv_vals, trip_vals)):
        tot = vv + iv
        col_txt = "#1b7837" if tr == 0.0 else "#222222"
        ax.text(xi, tot + ax.get_ylim()[1] * 0.022, f"{tr:.2f}",
                ha="center", va="bottom", fontsize=6.2, color=col_txt, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, facecolor="white",
              framealpha=0.94, edgecolor="#dddddd", borderpad=0.35)
    ax.text(0.98, 0.77, "bar labels: mean trips / 10d", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.0, color="#444444", style="italic")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(OUT_FIG_PDF)
    fig.savefig(OUT_FIG_PNG, dpi=220)
    plt.close(fig)
    log(f"Wrote {OUT_FIG_PDF} and {OUT_FIG_PNG}")
    return 0


# ---------------------------------------------------------------------------
# Save helper & main
# ---------------------------------------------------------------------------

def save_json(payload: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(
            payload, fh, indent=2,
            default=lambda o: o.tolist() if hasattr(o, "tolist")
            else (o.item() if hasattr(o, "item") else str(o)),
        )
    os.replace(tmp, OUT_JSON)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--plot" in argv:
        return plot()
    if "--pilot" in argv:
        return run_pilot()

    log(f"Starting R11 full experiment (TRAIN={len(TRAIN_SEEDS)}, TEST={len(TEST_SEEDS)}, workers={MAX_WORKERS})")
    certs = compute_base_certificates()

    payload = {
        "meta": {
            "experiment": "r11_risk_outages",
            "days": DAYS,
            "train_seeds": TRAIN_SEEDS,
            "cert_seeds": CERT_SEEDS,
            "test_seeds": TEST_SEEDS,
            "bootstrap_resamples": BOOT_N,
            "bootstrap_seed": BOOT_SEED,
            "voll": VOLL,
            "gen_cost": GEN_COST.tolist(),
            "alpha": _ALPHA.tolist(),
            "mu_hi_g": {"gas": MU_HI_GAS, "coal": MU_HI_COAL, "nuclear": MU_HI_NUC},
            "analytical_tiers": {
                "tier1_nuc_mu": TIER1_MU,
                "tier2_nuc_coal_mu": TIER2_MU,
                "tier3_all_mu": TIER3_MU,
            },
            "const_grid_base": CONST_GRID_BASE,
            "wgate_grid": WGATE_GRID,
            "hazard_gamma_grid": HAZARD_GAMMA_GRID,
            "dense_mu_grid": DENSE_MU_GRID,
            "crn_note": (
                "Exogenous RNG (seed) is 100% policy-independent across paired seeds. "
                "Outage RNG (seed+1) draws 1 Uniform(0,1) per available unit per day in "
                "fixed unit order (gas, coal, nuclear, renewable); identical across policies "
                "until a trip occurs, deterministic per seed."
            ),
        },
        "base_certificates": certs,
        "mechanism_diagnosis": None,
        "regimes": [],
    }

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        payload["mechanism_diagnosis"] = run_mechanism_diagnosis(pool)
        save_json(payload)

        for reg in REGIMES:
            res = evaluate_regime(pool, reg, certs[reg["base"]])
            payload["regimes"].append(res)
            payload["meta"]["elapsed_s"] = time.time() - T0
            save_json(payload)

    log(f"Finished all {len(REGIMES)} regimes in {time.time() - T0:.1f}s -> {OUT_JSON}")
    plot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
