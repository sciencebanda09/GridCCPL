"""R3 -- is there a regime where a STATE-DEPENDENT wear price beats the best CONSTANT one?

Protocol (honest split):
  * every selection (constant-mu sweep, rule-shape search, value-gradient fit)
    uses TRAINING seeds >= 101 only;
  * the selected constant and the selected state-dependent rule are then
    evaluated once on held-out TEST seeds 1-5 with paired differences
    (common random numbers: exogenous draws do not depend on the policy).

Objective per episode = operational cost (fuel incl. heat-rate, imports, VOLL,
overload, reserve) + optional terminal salvage  S * sum_g W_g(T)  for damage
that outlives the episode.

Policy families (all dispatch with the exact piecewise LP; only the price varies):
  blind     : no wear price (industry default)
  const     : mu_g = mu0
  wear      : mu_g = mu0 * (1 + b * W_g)                 (convex damage -> rising price)
  peak      : mu_g = mu0 * clip(r, .5, 2)**k,  r = max F[t+1..t+H] / F[t]
              (F = day-ahead deterministic load forecast; k>0 -> price high
               while the peak is still ahead, low once load is falling)
  ttg       : mu_g = mu0, but mu_end in the last k steps (time-to-go)
  vg        : fitted value-gradient price  mu_g(s) = lam * max(0, beta_g . phi(s)),
              beta fitted to finite-difference estimates of dV/dW_g obtained by
              CRN rollouts of the best-constant policy from training states.

Run:  python -m twoclock.experiments.r3_interior
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time

import numpy as np
from scipy import stats

from ..agents.baselines import BaseAgent, OPFBlind, _battery_bounds, action_from_opf
from ..envs import TwoClockEnv, make_env
from ..envs.two_clock_env import EnvConfig
from ..paths import RESULTS_DIR
from ..physics import (
    GEN_DISPATCHABLE, GEN_RATING, AgingParams, build_5bus, fast_dispatch,
    wear_free_mw, wear_price_from_mu,
)

TRAIN = (101, 102, 103)
TEST = (1, 2, 3, 4, 5)
CONST_GRID = (1e3, 3e3, 1e4, 2e4, 5e4, 1e5, 3e5, 1e6)
OUT = os.path.join(RESULTS_DIR, "r3_interior.json")
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------

class HeatWaveEnv(TwoClockEnv):
    """Multi-day heat wave: demand multiplied by ``mult[day]``."""

    def __init__(self, config, mult):
        self.mult = np.asarray(mult, dtype=float)
        super().__init__(config)

    def day_mult(self, step: int) -> float:
        d = min(step // self.cfg.steps_per_day, len(self.mult) - 1)
        return float(self.mult[d])

    def _demand(self) -> float:
        return super()._demand() * self.day_mult(self._step)


def heatwave_mult(days: int) -> list:
    m = [1.0] * days
    for d in range(3, min(7, days)):
        m[d] = 1.22
    return m


PERSIST = AgingParams(tau_thermal=720.0, c_age=0.55 * 0.05)

REGIMES = {
    "nominal_1.00": dict(days=10, scale=1.00),
    "tight_1.15": dict(days=10, scale=1.15),
    "tight_1.25": dict(days=10, scale=1.25),
    "tight_1.35": dict(days=10, scale=1.35),
    "peak_difficulty": dict(days=10, scale=1.00, name="peak"),
    "tight_1.15_outages": dict(days=10, scale=1.15, outage=True),
    "heatwave_4day_1.22x": dict(days=10, scale=1.00, heatwave=True),
    "tight_1.15_20d_salvage": dict(days=20, scale=1.15, salvage=2.0e4),
    "persistent_damage_1.15_30d_salvage": dict(days=30, scale=1.15, salvage=2.0e6,
                                               aging=PERSIST),
}


def build_env(reg: dict, seed: int):
    aging = reg.get("aging", AgingParams())
    kw = dict(demand_scale=reg["scale"], outage_enabled=reg.get("outage", False),
              aging=copy.deepcopy(aging))
    if reg.get("heatwave"):
        cfg = EnvConfig(days=reg["days"], difficulty="normal", seed=seed, **kw)
        env = HeatWaveEnv(cfg, heatwave_mult(reg["days"]))
        env.name = "heatwave"
        return env
    return make_env(reg.get("name", "normal"), days=reg["days"], seed=seed, **kw)


def forecast(env) -> np.ndarray:
    """Deterministic day-ahead load forecast over the whole horizon (+1 day pad)."""
    spd = env.cfg.steps_per_day
    n = env.cfg.horizon + spd
    f = np.array([env._demand_profile[t % spd] for t in range(n)])
    if isinstance(env, HeatWaveEnv):
        f = f * np.array([env.day_mult(min(t, env.cfg.horizon - 1)) for t in range(n)])
    return f


# ---------------------------------------------------------------------------
# Price-rule agent
# ---------------------------------------------------------------------------

class RuleAgent(BaseAgent):
    def __init__(self, net, aging, family: str, **p):
        self.net, self.aging, self.family, self.p = net, aging, family, p
        self.free = wear_free_mw(aging)
        self.coef = wear_price_from_mu(1.0, aging, GEN_RATING, GEN_DISPATCHABLE)
        self.name = family + str(p)

    def reset(self, env=None) -> None:
        self.env = env
        self.T = env.cfg.horizon
        self.F = forecast(env)

    def mu(self, ctx) -> np.ndarray:
        p, t = self.p, int(ctx["step"])
        W = np.asarray(ctx["wear"], dtype=float)
        mu0 = p["mu0"]
        f = self.family
        if f == "const":
            m = np.full(4, mu0)
        elif f == "wear":
            m = mu0 * (1.0 + p["b"] * W)
        elif f == "peak":
            H = p["H"]
            r = self.F[t + 1:t + 1 + H].max() / self.F[t]
            m = np.full(4, mu0 * float(np.clip(r, 0.5, 2.0)) ** p["k"])
        elif f == "ttg":
            m = np.full(4, mu0 if self.T - t > p["k"] else p["mu_end"])
        elif f == "vg":
            phi = vg_features(W, t, self.T, self.F)
            m = np.zeros(4)
            for g in range(3):
                m[g] = p["lam"] * max(p.get("floor", 0.0), float(phi[g] @ p["beta"][g]))
        else:
            raise ValueError(f)
        m = np.maximum(m, 0.0)
        m[3] = 0.0
        return m

    def act(self, obs):
        ctx = obs.ctx
        lo, hi = _battery_bounds(ctx["soc"], 0.25)
        wp = self.mu(ctx) * self.coef
        res = fast_dispatch(self.net, ctx["load"], ctx["prev_p"], ctx["renewable"],
                            ctx["pmax_eff"], reserve_req_mw=ctx["reserve_req"],
                            battery_min_mw=lo, battery_max_mw=hi, spot_price=ctx["spot"],
                            wear_price_mwh=wp, wear_free_mw=self.free)
        return action_from_opf(res, ctx["load"])


def vg_features(W, t, T, F) -> list:
    r = F[t + 1:t + 9].max() / F[t]          # 2-h peak-ahead ratio
    lf = F[t] / F[:T].max()
    ttg = min(1.0, (T - t) / 48.0)            # time-to-go, saturating at 12 h
    return [np.array([1.0, W[g], W[g] ** 2, r - 1.0, lf, ttg]) for g in range(3)]


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run(reg: dict, agent, seed: int) -> dict:
    env = build_env(reg, seed)
    obs = env.reset()
    agent.reset(env)
    free = wear_free_mw(env.cfg.aging)
    excess = 0.0
    while not env.done:
        obs, _, _, info = env.step(agent.act(obs))
        excess += float(np.maximum(0.0, info["p_gen"][:3] - free[:3]).sum()) * env.cfg.dt_hours
    s = env.episode_summary()
    salv = reg.get("salvage", 0.0) * float(env.aging.wear.sum())
    return {"seed": seed, "total": s["episode_cost"] + salv, "op_cost": s["episode_cost"],
            "salvage": salv, "unserved_mwh": s["unserved_mwh"], "excess_mwh": excess,
            "wear_end_sum": float(env.aging.wear.sum()), "trips": s["trips"]}


def evaluate(reg, agent, seeds) -> dict:
    rows = [run(reg, agent, s) for s in seeds]
    return {"mean": float(np.mean([r["total"] for r in rows])), "rows": rows}


def paired(a: dict, b: dict) -> dict:
    """Statistics of a - b (per seed, paired)."""
    d = np.array([ra["total"] - rb["total"] for ra, rb in zip(a["rows"], b["rows"])])
    n = d.size
    m, sd = float(d.mean()), float(d.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if sd > 0 else 0.0
    p = float(stats.ttest_rel([r["total"] for r in a["rows"]],
                              [r["total"] for r in b["rows"]]).pvalue) if sd > 0 else 1.0
    base = float(np.mean([r["total"] for r in b["rows"]]))
    return {"mean_diff": m, "ci95": [m - half, m + half], "rel_pct": 100 * m / base,
            "p_value": p, "n": n, "diffs": d.tolist(),
            "wins": int((d < 0).sum())}


# ---------------------------------------------------------------------------
# Value-gradient fit
# ---------------------------------------------------------------------------

def rollout_cost(env, agent, steps: int, salvage: float) -> float:
    c = 0.0
    obs = env._observe()
    for _ in range(steps):
        if env.done:
            break
        obs, _, _, info = env.step(agent.act(obs))
        c += info["step_cost"]
    if env.done:
        c += salvage * float(env.aging.wear.sum())
    return c


def fit_value_gradient(reg, net, aging, mu_c, seeds=(101, 102), every=24,
                       horizon=48, eps=0.03) -> dict:
    base_agent = RuleAgent(net, aging, "const", mu0=mu_c)
    X = [[], [], []]
    Y = [[], [], []]
    salv = reg.get("salvage", 0.0)
    for s in seeds:
        env = build_env(reg, s)
        obs = env.reset()
        base_agent.reset(env)
        F, T = base_agent.F, env.cfg.horizon
        while not env.done:
            t = env._step
            # only sample states where stress is plausible (load above median forecast)
            if t % every == 0 and t > 0:
                snap = copy.deepcopy(env)
                ag = copy.deepcopy(base_agent)
                ag.env = snap
                c0 = rollout_cost(copy.deepcopy(snap), ag, horizon, salv)
                phi = vg_features(env.aging.wear, t, T, F)
                for g in range(3):
                    e2 = copy.deepcopy(snap)
                    e2.aging.wear[g] = min(1.0, e2.aging.wear[g] + eps)
                    c1 = rollout_cost(e2, ag, horizon, salv)
                    X[g].append(phi[g])
                    Y[g].append((c1 - c0) / eps)
            obs, _, _, _ = env.step(base_agent.act(obs))
    beta, r2, ymean = [], [], []
    for g in range(3):
        A, y = np.array(X[g]), np.array(Y[g])
        b, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ b
        ss = float(((y - y.mean()) ** 2).sum())
        beta.append(b)
        r2.append(1 - float(((y - pred) ** 2).sum()) / ss if ss > 0 else 0.0)
        ymean.append(float(y.mean()))
    return {"beta": beta, "r2": r2, "dVdW_mean": ymean,
            "dVdW_q": [np.percentile(Y[g], [10, 50, 90]).tolist() for g in range(3)],
            "n_states": len(Y[0])}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def tag(fam, p):
    q = {k: v for k, v in p.items() if k not in ("beta",)}
    return f"{fam}:" + ",".join(f"{k}={v:g}" for k, v in q.items())


def do_regime(name: str, reg: dict, net) -> dict:
    aging = reg.get("aging", AgingParams())
    out = {"config": {k: (v.as_dict() if hasattr(v, "as_dict") else v)
                      for k, v in reg.items()}}
    train = TRAIN if reg["days"] <= 10 else TRAIN[:2]
    out["train_seeds"] = list(train)

    def ev(fam, seeds=train, **p):
        return evaluate(reg, RuleAgent(net, aging, fam, **p), seeds)

    # (1) constant sweep ---------------------------------------------------
    blind_tr = evaluate(reg, OPFBlind(net, aging), train)
    sweep = {}
    for mu in CONST_GRID:
        sweep[mu] = ev("const", mu0=mu)
    order = sorted(sweep)
    best = min(order, key=lambda m: sweep[m]["mean"])
    i = order.index(best)
    for m in ([np.sqrt(best * order[i - 1])] if i > 0 else []) + \
             ([np.sqrt(best * order[i + 1])] if i < len(order) - 1 else []):
        sweep[float(m)] = ev("const", mu0=float(m))
    order = sorted(sweep)
    mu_c = min(order, key=lambda m: sweep[m]["mean"])
    j = order.index(mu_c)
    interior_mu = (0 < j < len(order) - 1 and
                   sweep[order[j - 1]]["mean"] > sweep[mu_c]["mean"] * 1.001 and
                   sweep[order[j + 1]]["mean"] > sweep[mu_c]["mean"] * 1.001)
    out["train_blind"] = blind_tr["mean"]
    out["train_const_sweep"] = {f"{m:g}": {
        "total": sweep[m]["mean"],
        "unserved": float(np.mean([r["unserved_mwh"] for r in sweep[m]["rows"]])),
        "excess_mwh": float(np.mean([r["excess_mwh"] for r in sweep[m]["rows"]])),
        "wear_end_sum": float(np.mean([r["wear_end_sum"] for r in sweep[m]["rows"]]))}
        for m in order}
    out["best_const_mu"] = mu_c
    z = sweep[1e6]["mean"]
    out["prop1"] = {
        "zero_excess_policy_mu": 1e6,
        "zero_excess_excess_mwh": out["train_const_sweep"]["1e+06"]["excess_mwh"],
        "zero_excess_total": z,
        "best_const_total": sweep[mu_c]["mean"],
        "gain_of_allowing_excess_pct": 100 * (z - sweep[mu_c]["mean"]) / z,
        "interior_in_mu": bool(interior_mu),
    }
    log(f"{name}: blind={blind_tr['mean']/1e6:.3f}  best const mu={mu_c:g} "
        f"J={sweep[mu_c]['mean']/1e6:.4f}  J(1e6)={z/1e6:.4f}  interior={interior_mu}")

    # (2) state-dependent search -------------------------------------------
    cands = {}

    def tryit(fam, **p):
        key = tag(fam, p)
        if key not in cands:
            cands[key] = (fam, p, ev(fam, **p)["mean"])
        return cands[key][2]

    fam_best = {}
    for b in (3.0, 30.0):
        tryit("wear", mu0=mu_c, b=b)
    for k in (-10.0, 30.0):
        tryit("peak", mu0=mu_c, H=8, k=k)
    tryit("ttg", mu0=mu_c, k=4, mu_end=0.0)
    vg = fit_value_gradient(reg, net, aging, mu_c, seeds=(101,) if reg["days"] > 10 else (101, 102),
                            every=48)
    out["value_gradient_fit"] = {k: (np.asarray(v).tolist() if k == "beta" else v)
                                 for k, v in vg.items()}
    log(f"  vg fit: dV/dW mean={['%.3g' % v for v in vg['dVdW_mean']]} r2={['%.2f' % v for v in vg['r2']]}")
    for lam in (1.0, 2.0):
        tryit("vg", lam=lam, beta=vg["beta"], mu0=0.0)
    # rescale mu0 for the best shape of each non-vg family
    for fam in ("wear", "peak"):
        kb = min((k for k in cands if cands[k][0] == fam), key=lambda k: cands[k][2])
        p = dict(cands[kb][1])
        for s in (0.5, 2.0):
            q = dict(p, mu0=p["mu0"] * s)
            tryit(fam, **q)
    for fam in ("wear", "peak", "ttg", "vg"):
        kb = min((k for k in cands if cands[k][0] == fam), key=lambda k: cands[k][2])
        fam_best[fam] = kb
    out["train_state_dep"] = {k: v[2] for k, v in cands.items()}
    kbest = min(cands, key=lambda k: cands[k][2])
    out["best_state_dep"] = kbest
    log(f"  best state-dep (train) {kbest}  J={cands[kbest][2]/1e6:.4f} "
        f"vs const {sweep[mu_c]['mean']/1e6:.4f}")

    # (3) held-out test -------------------------------------------------------
    blind = evaluate(reg, OPFBlind(net, aging), TEST)
    const = evaluate(reg, RuleAgent(net, aging, "const", mu0=mu_c), TEST)
    fam, p, _ = cands[kbest]
    sd = evaluate(reg, RuleAgent(net, aging, fam, **p), TEST)
    out["test"] = {
        "blind": blind, "best_const": const, "best_state_dep": sd,
        "state_dep_minus_const": paired(sd, const),
        "const_minus_blind": paired(const, blind),
    }
    # secondary (not pre-selected; multiple comparisons): each family's best on test
    sec = {}
    for f2, k2 in fam_best.items():
        if k2 == kbest or reg["days"] > 10:
            continue
        ff, pp, _ = cands[k2]
        e = evaluate(reg, RuleAgent(net, aging, ff, **pp), TEST)
        sec[k2] = paired(e, const)
    out["test_secondary_family_bests_minus_const"] = sec
    pc = out["test"]["state_dep_minus_const"]
    log(f"  TEST blind={blind['mean']/1e6:.4f} const={const['mean']/1e6:.4f} "
        f"sd={sd['mean']/1e6:.4f}  diff={pc['mean_diff']:.0f} "
        f"CI=[{pc['ci95'][0]:.0f},{pc['ci95'][1]:.0f}] p={pc['p_value']:.3f}")
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    net = build_5bus()
    payload = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as fh:
            payload = json.load(fh)
    payload["protocol"] = ({
        "train_seeds": list(TRAIN), "train_seeds_long": list(TRAIN[:2]),
        "test_seeds": list(TEST), "const_grid": list(CONST_GRID),
        "objective": "operational episode cost + salvage*sum(W_end)",
        "stat": "paired t-test, df=n-1, 95% t-CI of per-seed differences"})
    payload.setdefault("regimes", {})
    names = argv or list(REGIMES)
    for name in names:
        log(f"=== {name}")
        payload["regimes"][name] = do_regime(name, REGIMES[name], net)
        with open(OUT, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, default=lambda o: (
                o.tolist() if hasattr(o, "tolist") else
                o.item() if hasattr(o, "item") else str(o)))
    log(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
