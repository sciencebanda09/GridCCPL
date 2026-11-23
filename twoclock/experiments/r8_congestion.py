"""R8 -- does the wear price need to be LOCATIONAL when transmission congests?

Prior runs (R4-R6) never congested: LMPs were identical at all buses, so a
scalar wear price could not be distinguished from a per-unit (locational) one.
Here we build congested variants of the 5/14/30-bus networks by scaling line
limits down (all lines, or chosen corridors) and/or moving load, confirm that
congestion binds (binding-line frequency, LMP spread across buses), and compare

  * blind OPF (mu = 0)
  * scalar certificate          mu   = rule(mu_lo, mu_hi)      (fixed point, train seeds)
  * per-unit certificate        mu_g = rule(mu_lo_g, mu_hi_g)  (per-unit fixed point,
                                       mu_lo_g = max_t (LMP0_{t,b(g)} - c_g)/alpha_g)
  * best scalar constant        dense log sweep on training seeds
  * best per-unit constant      coordinate search over (mu_gas, mu_coal, mu_nuc)
                                on training seeds, started at the best scalar

on test seeds 1-5 (10-day episodes).  rule(lo, hi) = sqrt(lo*hi) if hi finite
else 2*lo (same rule as R6).

Env fix (R8): ``TwoClockEnv.step`` used to inject import and battery power at
bus 4 regardless of network; it now injects at the module-level
``BATTERY_BUS`` / ``IMPORT_BUS`` that ``set_network`` swaps together with
``GEN_BUS`` / ``LOAD_SHARE``.  ``bus4_bug_check`` quantifies the difference.

Run:  python -m twoclock.experiments.r8_congestion            (full)
      python -m twoclock.experiments.r8_congestion --explore  (congestion scan)
      python -m twoclock.experiments.r8_congestion --plot     (figure only)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy import stats

import twoclock.envs.two_clock_env as env_mod
import twoclock.physics.dc_flow as dcf
import twoclock.physics.opf as opf_mod
from twoclock.agents.baselines import OPFAgent
from twoclock.agents.certificate import CertificateProbe, alpha_coef, certificate_from_rows
from twoclock.envs import make_env
from twoclock.paths import RESULTS_DIR
from twoclock.physics import AgingParams, DCNetwork, Line, build_5bus

DAYS = 10
TEST_SEEDS = (1, 2, 3, 4, 5)
CERT_SEEDS = (101, 102, 103, 104, 105)
TRAIN_SEEDS = (101, 102)
GRID = [float(x) for x in np.logspace(2, 6, 24)]           # dense scalar sweep
PU_GRID = [0.0] + [float(x) for x in np.logspace(2.5, 5.5, 7)]   # per-unit coords
PU_PASSES = 2
OUT_JSON = os.path.join(RESULTS_DIR, "r8_congestion.json")
OUT_FIG = os.path.join(RESULTS_DIR, "fig_r8_congestion.pdf")
UNITS = ("gas", "coal", "nuclear")


def log(*a):
    print(*a, flush=True)


def ci95(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def summ(x):
    x = np.asarray(x, dtype=float)
    return {"mean": float(np.mean(x)), "ci95": ci95(x), "per_seed": [float(v) for v in x]}


def rule(lo, hi):
    lo = max(float(lo), 1.0)
    return float(np.sqrt(lo * hi)) if np.isfinite(hi) else 2.0 * lo


# ---------------------------------------------------------------------------
# networks / variants
# ---------------------------------------------------------------------------

ORIG = (dcf.GEN_BUS.copy(), dcf.LOAD_SHARE.copy(), dcf.BATTERY_BUS, dcf.IMPORT_BUS)


def base_network(tag):
    if tag == "5bus":
        gb, ls, bb, ib = ORIG
        return build_5bus(), gb.copy(), ls.copy(), bb, ib
    if tag == "14bus":
        from twoclock.experiments.stage2_transfer import build_14bus
        return build_14bus()
    from twoclock.experiments.rigorous_suite import build_30bus
    return build_30bus()


def set_network(v):
    """Build the variant network and swap the topology globals (dcf, opf, env)."""
    net, gb, ls, bb, ib = base_network(v["net"])
    protect = set()
    if v.get("protect"):          # keep radial must-run outlets (nuclear, renewable)
        protect = {int(gb[2]), int(gb[3])}
    lines = []
    for k, ln in enumerate(net.lines):
        s = v.get("scale", 1.0)
        if k in v.get("line_scale", {}):
            s = v["line_scale"][k]
        if ln.i in protect or ln.j in protect:
            s = 1.0
        lines.append(Line(ln.i, ln.j, ln.x, ln.limit_mw * s))
    net = DCNetwork(n_bus=net.n_bus, lines=lines, slack=net.slack, bus_names=net.bus_names)
    if v.get("load_share") is not None:
        ls = np.asarray(v["load_share"], dtype=float)
        ls = ls / ls.sum()
    for mod in (dcf, opf_mod, env_mod):
        mod.GEN_BUS = gb
        mod.LOAD_SHARE = ls
        mod.BATTERY_BUS = bb
        mod.IMPORT_BUS = ib
    return net


# 5-bus lines: L0 0-1, L1 1-2, L2 2-3, L3 3-4, L4 4-0, L5 1-3
# gens: gas@0 coal@1 nuclear@2 renewable@3; battery+import @4
VARIANTS = [   # ordered by priority (run in this order)
    {"name": "5bus nominal", "net": "5bus"},
    {"name": "5bus all-lines x0.3", "net": "5bus", "scale": 0.3},
    # nuclear in a load pocket: load moved to bus 2, its two lines at 0.35x
    {"name": "5bus nuclear pocket", "net": "5bus", "line_scale": {1: 0.35, 2: 0.35},
     "load_share": [0.10, 0.20, 0.40, 0.10, 0.20]},
    {"name": "14bus x0.7 (protected outlets)", "net": "14bus", "scale": 0.7, "protect": True},
    {"name": "30bus x0.8 (protected outlets)", "net": "30bus", "scale": 0.8, "protect": True},
    # gas in a load pocket: load moved to bus 0, its two lines at 0.4x
    {"name": "5bus gas pocket", "net": "5bus", "line_scale": {0: 0.4, 4: 0.4},
     "load_share": [0.40, 0.15, 0.15, 0.10, 0.20]},
    {"name": "5bus all-lines x0.4", "net": "5bus", "scale": 0.4},
    {"name": "14bus nominal", "net": "14bus"},
]


# ---------------------------------------------------------------------------
# agent + episode with congestion / balance instrumentation
# ---------------------------------------------------------------------------

class PricedOPF(OPFAgent):
    """OPF dispatch with scalar or per-unit wear price (None/0 -> blind)."""

    def __init__(self, net, aging, mu):
        mu_arr = None if mu is None else np.asarray(mu, dtype=float)
        if mu_arr is not None and mu_arr.ndim == 0 and float(mu_arr) <= 0:
            mu_arr = None
        if mu_arr is not None and mu_arr.ndim == 1 and mu_arr.size == 3:
            mu_arr = np.concatenate([mu_arr, [0.0]])
        super().__init__(net, aging, mu=None)
        self.mu = mu_arr
        self.name = "priced_opf"

    def act(self, obs):
        a = super().act(obs)
        r = self.last_result
        # pass the OPF's per-bus shed location to the env (R8 env extension)
        return np.concatenate([a, np.asarray(r.shed, dtype=float)])


class VecProbe(CertificateProbe):
    """CertificateProbe that dispatches with a per-unit price vector."""

    def __init__(self, net, aging, mu_vec):
        super().__init__(net, aging, 0.0)
        self.inner = PricedOPF(net, aging, mu_vec)


def run(env, agent, net):
    obs = env.reset()
    agent.reset(env)
    lb = net.limits
    gb = np.asarray(dcf.GEN_BUS[:3])
    st = {"steps": 0, "lp": 0, "fallback": 0, "bind": 0, "lmp_spread": [], "gen_spread": [],
          "lmp_gen": [], "bind_line": np.zeros(net.n_line), "bal_max": 0.0,
          "flow_dev_max": 0.0, "flow_dev_sum": 0.0, "flow_dev_gt1": 0}
    while not env.done:
        a = agent.act(obs)
        r = getattr(agent, "last_result", None)
        if r is None and hasattr(agent, "inner"):
            r = agent.inner.last_result
        load = obs.ctx["load"]
        obs, _, _, info = env.step(a)
        st["steps"] += 1
        if r is not None:
            fb = r.status not in ("fast", "optimal") and np.all(r.lmp == 0)
            st["fallback"] += int(fb)
            st["lp"] += int(r.status != "fast")
            b = np.abs(r.flows) >= lb - 1e-3
            st["bind_line"] += b
            st["bind"] += int(b.any())
            if not fb:
                st["lmp_spread"].append(float(np.ptp(r.lmp)))
                st["gen_spread"].append(float(np.ptp(r.lmp[gb])))
                st["lmp_gen"].append(r.lmp[gb].tolist())
            dev = float(np.max(np.abs(info["flows"] - r.flows)))
            st["flow_dev_max"] = max(st["flow_dev_max"], dev)
            st["flow_dev_sum"] += dev
            st["flow_dev_gt1"] += int(dev > 1.0)
        # power balance of what the env actually metered
        served = load - info["unserved_mw"]
        bal = float(np.sum(info["p_gen"]) + info["import_mw"] + info["battery"] - served)
        st["bal_max"] = max(st["bal_max"], abs(bal))
    out = env.episode_summary()
    n = max(1, st["steps"])
    sp = np.array(st["lmp_spread"]) if st["lmp_spread"] else np.zeros(1)
    gs = np.array(st["gen_spread"]) if st["gen_spread"] else np.zeros(1)
    # fast-path steps have a uniform LMP (spread 0): pad to all steps
    pad = n - st["fallback"] - len(st["lmp_spread"]) + (0 if st["lmp_spread"] else 1)
    sp_all = np.concatenate([sp, np.zeros(max(0, pad))])
    gs_all = np.concatenate([gs, np.zeros(max(0, pad))])
    lg = np.array(st["lmp_gen"]) if st["lmp_gen"] else np.zeros((1, 3))
    out["congestion"] = {
        "frac_steps_binding": st["bind"] / n,
        "frac_binding_per_line": (st["bind_line"] / n).round(4).tolist(),
        "frac_lp_steps": st["lp"] / n,
        "frac_fallback_steps": st["fallback"] / n,
        "lmp_spread_all_mean": float(sp_all.mean()),
        "lmp_spread_all_median": float(np.median(sp_all)),
        "gen_lmp_spread_mean": float(gs_all.mean()),
        "gen_lmp_spread_median": float(np.median(gs_all)),
        "gen_lmp_spread_p90": float(np.percentile(gs_all, 90)),
        "frac_steps_gen_spread_gt1": float(np.mean(gs_all > 1.0)),
        "frac_steps_gen_spread_gt20": float(np.mean(gs_all > 20.0)),
        "lp_gen_lmp_median": np.median(lg, axis=0).tolist(),
        "lp_gen_lmp_mean": lg.mean(axis=0).tolist(),
    }
    out["balance"] = {"max_abs_imbalance_mw": st["bal_max"],
                      "flow_env_minus_opf_max_mw": st["flow_dev_max"],
                      "flow_env_minus_opf_mean_mw": st["flow_dev_sum"] / n,
                      "frac_steps_flow_dev_gt1mw": st["flow_dev_gt1"] / n}
    return out


class Ctx:
    def __init__(self, v, net, aging):
        self.v, self.net, self.aging = v, net, aging
        self.cache = {}
        self.n_env_steps = 0

    def env(self, seed, days=DAYS):
        e = make_env(self.v.get("difficulty", "normal"), days=days, seed=seed,
                     demand_scale=self.v.get("demand_scale", 1.0))
        e.net = self.net
        return e

    @staticmethod
    def key(mu):
        if mu is None:
            return ("blind",)
        m = np.atleast_1d(np.asarray(mu, dtype=float))
        if m.size == 1:
            m = np.repeat(m, 3)
        m = m[:3]
        if np.all(m <= 0):
            return ("blind",)
        return tuple(float(f"{x:.6g}") for x in m)

    def ep(self, mu, seed):
        k = (self.key(mu), seed)
        if k not in self.cache:
            mu_vec = None if k[0] == ("blind",) else np.array(k[0])
            r = run(self.env(seed), PricedOPF(self.net, self.aging, mu_vec), self.net)
            self.n_env_steps += r["steps"]
            self.cache[k] = r
        return self.cache[k]

    def costs(self, mu, seeds):
        return np.array([self.ep(mu, s)["episode_cost"] for s in seeds])

    def mean_cost(self, mu, seeds):
        return float(np.mean(self.costs(mu, seeds)))


# ---------------------------------------------------------------------------
# certificates (scalar and per-unit fixed points on training seeds)
# ---------------------------------------------------------------------------

def run_probe(c, probe, seed):
    env = c.env(seed)
    obs = env.reset()
    probe.reset(env)
    while not env.done:
        obs, _, _, _ = env.step(probe.act(obs))
    c.n_env_steps += env._step
    return env.episode_summary()


def fixed_point(c, per_unit, max_iter=5, rtol=0.10):
    mu = np.zeros(3)
    hist, rows_hist = [], []
    steps = 0
    converged = False
    for k in range(max_iter):
        seed = CERT_SEEDS[k % len(CERT_SEEDS)]
        probe = VecProbe(c.net, c.aging, mu if mu.max() > 0 else None)
        ep = run_probe(c, probe, seed)
        steps += DAYS * 96
        cert = certificate_from_rows(probe.rows, c.aging)
        rows_hist.append(probe.rows)
        new = (np.maximum(np.array(cert["per_unit_lo"]), 1.0) if per_unit
               else np.full(3, max(cert["mu_lo"], 1.0)))
        hist.append({"iter": k, "seed": seed, "mu_run": mu.tolist(), "mu_lo": cert["mu_lo"],
                     "mu_hi": cert["mu_hi"], "per_unit_lo": cert["per_unit_lo"],
                     "per_unit_hi": cert["per_unit_hi"], "cost": ep["episode_cost"],
                     "forced": cert["forced_steps_per_unit"]})
        log(f"      {'PU' if per_unit else 'SC'} iter {k} seed {seed} mu_run={np.round(mu, 0)} "
            f"-> lo={np.round(cert['per_unit_lo'], 0)} (scalar {cert['mu_lo']:.0f}) "
            f"hi={cert['mu_hi']:.4g} forced={cert['forced_steps_per_unit']}")
        if k >= 1 and mu.max() > 0:
            rel = np.abs(np.log(np.maximum(new, 1.0) / np.maximum(mu, 1.0)))
            if np.all(rel < np.log(1 + rtol)):
                converged = True
                break
        mu = new
    priced = [rows_hist[i] for i, h in enumerate(hist) if max(h["mu_run"]) > 0][-2:]
    pooled = [r for rs in (priced or rows_hist[-1:]) for r in rs]
    final = certificate_from_rows(pooled, c.aging)
    lmp0 = np.array([r["lmp0"] for r in pooled])
    ok = np.array([r["ok0"] for r in pooled])
    l0 = lmp0[ok] if ok.any() else lmp0
    final["lmp0_gen_spread"] = {
        "mean": float(np.mean(np.ptp(l0, axis=1))),
        "median": float(np.median(np.ptp(l0, axis=1))),
        "p90": float(np.percentile(np.ptp(l0, axis=1), 90)),
        "frac_gt1": float(np.mean(np.ptp(l0, axis=1) > 1.0)),
        "lmp0_mean_per_unit": l0.mean(axis=0).tolist(),
        "lmp0_max_per_unit": l0.max(axis=0).tolist(),
    }
    final["history"] = hist
    final["env_steps"] = steps
    final["iterations"] = len(hist)
    final["converged"] = converged
    if per_unit:
        lo = np.array(final["per_unit_lo"]); hi = np.array(final["per_unit_hi"])
        final["price"] = [rule(l, h) for l, h in zip(lo, hi)]
    else:
        final["price"] = [rule(final["mu_lo"], final["mu_hi"])] * 3
    final["band_nonempty"] = bool(final["mu_lo"] < final["mu_hi"])
    return final


# ---------------------------------------------------------------------------
# constant-price searches on training seeds
# ---------------------------------------------------------------------------

def scalar_sweep(c):
    tr = {0.0: c.mean_cost(None, TRAIN_SEEDS)}
    for mu in GRID:
        tr[mu] = c.mean_cost(mu, TRAIN_SEEDS)
    best = min(tr, key=tr.get)
    return best, tr


def per_unit_search(c, mu_star):
    cur = np.full(3, float(mu_star))
    cur_cost = c.mean_cost(cur if mu_star > 0 else None, TRAIN_SEEDS)
    trace = [{"mu": cur.tolist(), "train_cost": cur_cost}]
    for p in range(PU_PASSES):
        moved = False
        for g in range(3):
            cand = sorted(set(PU_GRID + [float(mu_star)]))
            best_v, best_c = cur[g], cur_cost
            for val in cand:
                m = cur.copy(); m[g] = val
                cc = c.mean_cost(m, TRAIN_SEEDS)
                if cc < best_c - 1e-6:
                    best_v, best_c = val, cc
            if best_v != cur[g]:
                moved = True
            cur[g], cur_cost = best_v, best_c
            trace.append({"pass": p, "unit": UNITS[g], "mu": cur.tolist(),
                          "train_cost": cur_cost})
        if not moved:
            break
    return cur, cur_cost, trace


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def bus4_bug_check(aging):
    """Compare metered overload with import/battery injected at bus 4 (old
    behaviour) vs at the network's own buses (fixed), 14-bus nominal, mu=1e4."""
    out = {}
    v = {"name": "14bus nominal", "net": "14bus"}
    for label, force4 in (("fixed", False), ("old_bus4", True)):
        net = set_network(v)
        if force4:
            env_mod.BATTERY_BUS = 4
            env_mod.IMPORT_BUS = 4
        c = Ctx(v, net, aging)
        r = run(c.env(1, days=3), PricedOPF(net, aging, 1e4), net)
        out[label] = {"overload_mwh": r["overload_mwh"], "cost": r["episode_cost"],
                      **r["balance"]}
    set_network(v)
    return out


# ---------------------------------------------------------------------------
# one variant
# ---------------------------------------------------------------------------

def evaluate_variant(v, aging):
    t0 = time.time()
    net = set_network(v)
    c = Ctx(v, net, aging)
    log(f"== {v['name']}  ({net.n_bus} buses, {net.n_line} lines)")

    # congestion diagnostics from blind dispatch on test seeds
    cong_blind = [c.ep(None, s)["congestion"] for s in TEST_SEEDS]

    # 1. certificates (training seeds 101-105)
    cert_sc = fixed_point(c, per_unit=False)
    cert_pu = fixed_point(c, per_unit=True)
    mu_sc = cert_sc["price"][0]
    mu_pu = np.array(cert_pu["price"])
    log(f"   scalar cert [{cert_sc['mu_lo']:.0f}, {cert_sc['mu_hi']:.4g}] -> {mu_sc:.0f};  "
        f"per-unit lo={np.round(cert_pu['per_unit_lo'], 0)} hi={cert_pu['per_unit_hi']} "
        f"-> {np.round(mu_pu, 0)}  ({time.time() - t0:.0f}s)")

    # 2. best constants on training seeds
    mu_star, tr = scalar_sweep(c)
    log(f"   best scalar (train) mu*={mu_star:.0f}  ({time.time() - t0:.0f}s)")
    mu_pu_star, pu_cost, trace = per_unit_search(c, mu_star)
    log(f"   best per-unit (train) {np.round(mu_pu_star, 0)}  train gain vs scalar "
        f"{(tr[mu_star] - pu_cost) / 1e3:+.1f}k  ({time.time() - t0:.0f}s)")

    # 3. test evaluation
    methods = {
        "blind": None,
        "scalar_certificate": mu_sc,
        "per_unit_certificate": mu_pu,
        "best_scalar_constant": mu_star,
        "best_per_unit_constant": mu_pu_star,
    }
    rows = {k: [c.ep(m, s) for s in TEST_SEEDS] for k, m in methods.items()}
    cost = {k: np.array([r["episode_cost"] for r in rs]) for k, rs in rows.items()}
    ref_s = cost["best_scalar_constant"]
    ref_b = cost["blind"]
    ctl = {}
    for k, m in methods.items():
        cs = cost[k]
        ctl[k] = {
            "mu": None if m is None else np.round(np.atleast_1d(m), 3).tolist(),
            "cost": summ(cs),
            "minus_best_scalar": summ(cs - ref_s),
            "minus_blind": summ(cs - ref_b),
            "minus_scalar_certificate": summ(cs - cost["scalar_certificate"]),
            "saving_vs_blind_pct": float(100 * np.mean((ref_b - cs) / ref_b)),
            "unserved_mwh": summ([r["unserved_mwh"] for r in rows[k]]),
            "overload_mwh": summ([r["overload_mwh"] for r in rows[k]]),
            "wear_end_mean": summ([r["wear_end_mean"] for r in rows[k]]),
            "frac_steps_binding": float(np.mean([r["congestion"]["frac_steps_binding"]
                                                 for r in rows[k]])),
            "gen_lmp_spread_mean": float(np.mean([r["congestion"]["gen_lmp_spread_mean"]
                                                  for r in rows[k]])),
            "max_abs_imbalance_mw": float(max(r["balance"]["max_abs_imbalance_mw"]
                                              for r in rows[k])),
        }
        log(f"     {k:24s} mu={np.round(np.atleast_1d(m), 0) if m is not None else '-'!s:24s} "
            f"cost={cs.mean() / 1e6:8.4f}+-{ci95(cs) / 1e6:.4f}M "
            f"d_scalar*={np.mean(cs - ref_s) / 1e3:+8.1f}k+-{ci95(cs - ref_s) / 1e3:.1f}k "
            f"unserved={np.mean([r['unserved_mwh'] for r in rows[k]]):7.1f} "
            f"ovl={np.mean([r['overload_mwh'] for r in rows[k]]):6.2f} "
            f"bind={ctl[k]['frac_steps_binding']:.2f} genspread={ctl[k]['gen_lmp_spread_mean']:.1f}")

    def avg_cong(lst):
        keys = [k for k in lst[0] if isinstance(lst[0][k], float)]
        out = {k: float(np.mean([d[k] for d in lst])) for k in keys}
        out["frac_binding_per_line"] = np.mean([d["frac_binding_per_line"] for d in lst],
                                               axis=0).round(4).tolist()
        out["lp_gen_lmp_mean"] = np.mean([d["lp_gen_lmp_mean"] for d in lst], axis=0).tolist()
        return out

    out = {
        "variant": v, "n_bus": net.n_bus, "n_line": net.n_line,
        "line_limits": net.limits.tolist(),
        "congestion_blind_test": avg_cong(cong_blind),
        "congestion_best_scalar_test": avg_cong([r["congestion"] for r in
                                                 rows["best_scalar_constant"]]),
        "congestion_best_per_unit_test": avg_cong([r["congestion"] for r in
                                                   rows["best_per_unit_constant"]]),
        "balance_best_scalar_test": rows["best_scalar_constant"][0]["balance"],
        "certificate_scalar": cert_sc,
        "certificate_per_unit": cert_pu,
        "best_scalar_train": mu_star,
        "train_curve_scalar": {f"{k:g}": v_ for k, v_ in sorted(tr.items())},
        "best_per_unit_train": mu_pu_star.tolist(),
        "per_unit_search_trace": trace,
        "per_unit_train_gain_vs_scalar": float(tr[mu_star] - pu_cost),
        "control": ctl,
        "episodes_run": len(c.cache),
        "env_steps_total": c.n_env_steps,
        "wall_s": time.time() - t0,
    }
    log(f"   done in {out['wall_s']:.0f}s ({len(c.cache)} cached episodes)")
    return out


def save(payload):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1,
                  default=lambda o: o.tolist() if hasattr(o, "tolist")
                  else (o.item() if hasattr(o, "item") else str(o)))
    os.replace(tmp, OUT_JSON)


def explore(aging, days=3):
    for v in VARIANTS:
        net = set_network(v)
        c = Ctx(v, net, aging)
        for mu in (None, 1e4):
            r = run(c.env(101, days=days), PricedOPF(net, aging, mu), net)
            cg = r["congestion"]
            log(f"{v['name']:32s} mu={mu}: bind={cg['frac_steps_binding']:.2f} "
                f"fallback={cg['frac_fallback_steps']:.2f} genspread mean={cg['gen_lmp_spread_mean']:.1f} "
                f"p90={cg['gen_lmp_spread_p90']:.1f} allspread={cg['lmp_spread_all_mean']:.0f} "
                f"unserved={r['unserved_mwh']:.0f} ovl={r['overload_mwh']:.1f} "
                f"bal={r['balance']['max_abs_imbalance_mw']:.2g} "
                f"fdev={r['balance']['flow_env_minus_opf_max_mw']:.1f}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--plot" in argv:
        return plot()
    aging = AgingParams()
    if "--explore" in argv:
        return explore(aging)
    a = alpha_coef(aging)
    only = os.environ.get("R8_ONLY")
    variants = [v for v in VARIANTS if only is None or v["name"] in only.split(";")]
    payload = {"meta": {
        "days": DAYS, "test_seeds": TEST_SEEDS, "cert_seeds": CERT_SEEDS,
        "train_seeds": TRAIN_SEEDS, "grid": GRID, "per_unit_grid": PU_GRID,
        "per_unit_passes": PU_PASSES, "alpha": a.tolist(),
        "gen_cost": dcf.GEN_COST.tolist(), "voll": opf_mod.VOLL,
        "voll_caps": ((opf_mod.VOLL - dcf.GEN_COST[:3]) / a[:3]).tolist(),
        "rule": "price = sqrt(lo*hi) if hi finite else 2*lo (scalar and per unit)",
        "notes": "certificates from training seeds 101-105; constant-price searches on "
                 "training seeds 101-102; all reported costs on test seeds 1-5. "
                 "Env fixed to inject import/battery at the network's own buses."},
        "bus4_bug_check": None, "variants": []}
    t0 = time.time()
    payload["bus4_bug_check"] = bus4_bug_check(aging)
    log("bus-4 check:", payload["bus4_bug_check"])
    save(payload)
    for v in variants:
        payload["variants"].append(evaluate_variant(v, aging))
        payload["meta"]["elapsed_s"] = time.time() - t0
        save(payload)
    set_network(VARIANTS[0])
    log(f"total {time.time() - t0:.0f}s")
    plot()
    return 0


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with open(OUT_JSON, encoding="utf-8") as fh:
        d = json.load(fh)
    V = d["variants"]
    if not V:
        return 0
    short = [r["variant"]["name"].replace(" (protected outlets)", "").replace("all-lines ", "")
             for r in V]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), constrained_layout=True)
    y = np.arange(len(V))[::-1]
    # (a) congestion
    ax = axes[0]
    bind = [r["congestion_best_scalar_test"]["frac_steps_binding"] for r in V]
    spr = [r["certificate_scalar"]["lmp0_gen_spread"]["mean"] for r in V]
    ax.barh(y, bind, color="#2a6fb0", height=0.6)
    ax.set_xlim(0, 1.05)
    ax.set_yticks(y); ax.set_yticklabels(short, fontsize=6.5)
    ax.set_xlabel("frac. steps with binding line", fontsize=7)
    for yi, s in zip(y, spr):
        ax.text(1.02, yi, f"{s:.0f}", fontsize=6, va="center", ha="right", color="#444")
    ax.set_title("(a) congestion  (num.: mean gen-bus\nLMP$^0$ spread, \\$/MWh)", fontsize=7)
    ax.tick_params(labelsize=6)
    # (b) per-unit certificate prices
    ax = axes[1]
    cols = ["#d9822b", "#6b6b6b", "#3a9a5b"]
    for g, (u, col) in enumerate(zip(UNITS, cols)):
        lo = [max(r["certificate_per_unit"]["per_unit_lo"][g], 1.0) for r in V]
        ax.plot(lo, y, "o", ms=3.5, color=col, label=rf"$\mu^{{lo}}_{{{u}}}$")
    sc = [max(r["certificate_scalar"]["mu_lo"], 1.0) for r in V]
    ax.plot(sc, y, "|", ms=9, mew=1.5, color="k", label=r"scalar $\mu_{lo}$")
    hi = [r["certificate_scalar"]["mu_hi"] for r in V]
    for yi, h in zip(y, hi):
        if h is not None and np.isfinite(h):
            ax.plot([h], [yi], ">", ms=3.5, color="#b03a2a")
    ax.plot([], [], ">", ms=3.5, color="#b03a2a", label=r"scalar $\mu_{hi}$")
    ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_xlabel(r"certificate price [\$/wear-unit]", fontsize=7)
    ax.set_title("(b) per-unit vs scalar certificate", fontsize=7)
    ax.legend(fontsize=5.5, frameon=False, loc="lower left", ncol=1)
    ax.tick_params(labelsize=6)
    # (c) test cost vs best scalar constant
    ax = axes[2]
    keys = [("per_unit_certificate", "per-unit cert.", "#3a9a5b", -0.22),
            ("scalar_certificate", "scalar cert.", "#2a6fb0", 0.0),
            ("best_per_unit_constant", "best per-unit const.", "#d9822b", 0.22)]
    for k, lab, col, off in keys:
        m = [100 * r["control"][k]["minus_best_scalar"]["mean"]
             / r["control"]["best_scalar_constant"]["cost"]["mean"] for r in V]
        e = [100 * r["control"][k]["minus_best_scalar"]["ci95"]
             / r["control"]["best_scalar_constant"]["cost"]["mean"] for r in V]
        ax.errorbar(m, y + off, xerr=e, fmt="o", ms=3, lw=0.9, color=col, label=lab)
    ax.axvline(0, color="#444", lw=0.7)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_xlabel("test cost minus best scalar [%]", fontsize=7)
    ax.set_title("(c) paired test diff (seeds 1-5, 95% CI)", fontsize=7)
    ax.legend(fontsize=5.5, frameon=False, loc="lower right")
    ax.tick_params(labelsize=6)
    fig.savefig(OUT_FIG)
    fig.savefig(OUT_FIG.replace(".pdf", ".png"), dpi=200)
    log(f"wrote {OUT_FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
