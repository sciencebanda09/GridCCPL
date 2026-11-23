"""R7 -- a dual (LP-sensitivity) certificate for the wear-price threshold.

Question
--------
R4 found that the closed form mu_cf = (pi_bar - c_gas)/alpha_gas with a FIXED
peak price pi_bar has the right median but essentially no correlation with the
empirical threshold mu_emp across 42 parameter configurations.  Does a
certificate built from *realised* LP duals on training seeds do better?

Certificate
-----------
Run the wear-aware DC-OPF controller (exact LP, HiGHS duals) for one episode on
each training seed at a price mu_k.  For every step t and thermal unit g that is
at (or above) its wear-free level sigma*rating_g with excess headroom available,
the excess block e_g stays out of the optimal basis iff its reduced cost is
non-negative:

    c_g + alpha_g * mu - LMP_{t,b(g)} - y_res,t >= 0
    <=>  mu >= theta_{t,g} = (LMP_{t,b(g)} + y_res,t - c_g) / alpha_g

(y_res <= 0 is the dual of the reserve-headroom row, which also prices e_g;
the "raw" variant drops it, as in the task statement.)  Excluded: steps where
shedding is marginal (LP sheds and LMP ~ VOLL); excess forced by ramp-down limits; dispatched
excess whose removal forces shedding (checked with a counterfactual LP in which
ALL thermal units are capped at sigma*rating, so the replacement cannot be
another unit's excess block).  For dispatched, non-forced excess the
counterfactual LMP (replacement price) is used.  mu_lo = max_t,g theta (also
p95/p99); mu_{k+1} = mu_lo(mu_k) iterated from a high start.  Upper end:
mu_hi = min_g (VOLL - c_g)/alpha_g (above it, shedding beats excess).

Comparison with R4: mu_emp and cost curves are read from R4's JSON (copied
verbatim into this package's results dir); nothing of R4 is rerun except a few
explicit spot checks on test seeds 1-3.

Run:  python -m twoclock.experiments.r7_cert_robust          (full)
      python -m twoclock.experiments.r7_cert_robust --plot   (figure from JSON)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from ..agents import OPFAware
from ..agents.baselines import _battery_bounds, action_from_opf
from ..physics import build_5bus
from ..physics import dc_flow as dcf
from ..physics import opf as opf_mod
from ..physics.opf import _ramp_bounds, wear_free_mw, wear_price_from_mu
from ..paths import RESULTS_DIR
from .r4_robust import Applied, aging_from, make
from .runner import run_episode

R4_JSON = os.path.join(RESULTS_DIR, "r4_robust.json")
OUT_JSON = os.path.join(RESULTS_DIR, "r7_cert_robust.json")
OUT_FIG = os.path.join(RESULTS_DIR, "fig_r7_cert.pdf")

TRAIN_SEEDS = (101, 102)
N_ITER = 4
MU_START = 64000.0         # high start (top of R4's OAT grid); capped below mu_hi
FLAT_TOL = 0.005
TOL_MW = 1e-3
CTRL_MULT = 2.0
SPOT_CHECK = ["oat:nominal", "oat:stress_frac=0.75", "oat:voll=0", "oat:demand_scale=1.2",
              "lhs:draw04", "lhs:draw13", "lhs:draw15", "lhs:draw18"]
THERMAL = (0, 1, 2)


def alphas(aging) -> np.ndarray:
    return wear_price_from_mu(1.0, aging, dcf.GEN_RATING, dcf.GEN_DISPATCHABLE)


def mu_hi_of(cfg, aging) -> float:
    a = alphas(aging)
    c = np.array([cfg["c_gas"], cfg["c_coal"], cfg["c_nuc"]])
    return float(np.min((cfg["voll"] - c) / a[:3]))


# ---------------------------------------------------------------------------
# LMP verification by finite differences (nodal load perturbation)
# ---------------------------------------------------------------------------

def fd_check(net, ctx, kw, h=0.25) -> dict:
    base = opf_mod.dc_opf(net, ctx["load"], ctx["prev_p"], ctx["renewable"], ctx["pmax_eff"], **kw)
    share0 = opf_mod.LOAD_SHARE.copy()
    fd = np.zeros(net.n_bus)
    try:
        for b in range(net.n_bus):
            cs = []
            for s in (+1, -1):
                opf_mod.LOAD_SHARE[:] = share0
                opf_mod.LOAD_SHARE[b] += s * h / ctx["load"]
                r = opf_mod.dc_opf(net, ctx["load"], ctx["prev_p"], ctx["renewable"],
                                   ctx["pmax_eff"], **kw)
                cs.append(r.cost)
            fd[b] = (cs[0] - cs[1]) / (2 * h)
    finally:
        opf_mod.LOAD_SHARE[:] = share0
    return {"lmp": base.lmp.tolist(), "fd": fd.tolist(),
            "max_abs_err": float(np.max(np.abs(fd - base.lmp))),
            "congested": bool(np.ptp(base.lmp) > 1e-6)}


# ---------------------------------------------------------------------------
# One certificate episode
# ---------------------------------------------------------------------------

def cert_episode(cfg, aging, net, mu, seed, fd_budget=0) -> dict:
    env = make(cfg, aging, seed)
    obs = env.reset()
    a = alphas(aging)
    free = wear_free_mw(aging)
    wp = mu * a
    voll = float(cfg["voll"])
    c = dcf.GEN_COST
    th_res, th_raw, th_heat = [], [], []
    who = []
    info = []
    n = {"cand": 0, "excl_shed_step": 0, "excl_voll_lmp": 0, "excl_ramp_forced": 0,
         "excl_cf_forced_shed": 0, "excess_dispatched": 0, "lp_fail": 0, "steps": 0}
    fds = []
    while not env.done:
        ctx = obs.ctx
        lo, hi = _battery_bounds(ctx["soc"], 0.25)
        kw = dict(reserve_req_mw=ctx["reserve_req"], battery_min_mw=lo, battery_max_mw=hi,
                  spot_price=ctx["spot"], wear_price_mwh=wp, wear_free_mw=free)
        res = opf_mod.dc_opf(net, ctx["load"], ctx["prev_p"], ctx["renewable"],
                             ctx["pmax_eff"], **kw)
        n["steps"] += 1
        if res.status != "optimal":
            n["lp_fail"] += 1
        else:
            if fd_budget and len(fds) < 2 * fd_budget and (
                    (env._step % 197 == 13 and sum(not f["congested"] for f in fds) < fd_budget)
                    or (np.ptp(res.lmp) > 1e-6 and sum(f["congested"] for f in fds) < fd_budget)):
                fds.append(fd_check(net, ctx, kw))
            pmin, pmax = _ramp_bounds(ctx["prev_p"], ctx["pmax_eff"], ctx["renewable"])
            heat = env.aging.heat_rate() * env.aging.eff
            ex = res.duals["excess_mw"]
            e_lo = np.maximum(0.0, pmin - free)
            src = res
            skip = None
            shed_marg = res.shed_total > 1e-6 and np.any(np.abs(res.lmp - voll) <= 1e-6 * max(1.0, voll))
            if shed_marg:
                skip = "excl_shed_step"          # shedding is the marginal resource
            elif np.any(ex[:3] - e_lo[:3] > TOL_MW):
                # non-forced excess dispatched: price the step in the no-excess world
                n["excess_dispatched"] += 1
                pm = ctx["pmax_eff"].copy()
                pm[:3] = np.minimum(pm[:3], free[:3])
                cfr = opf_mod.dc_opf(net, ctx["load"], ctx["prev_p"], ctx["renewable"], pm, **kw)
                if cfr.status != "optimal" or cfr.shed_total > 1e-6:
                    skip = "excl_cf_forced_shed"
                else:
                    src = cfr
            yr = float(src.duals.get("reserve", 0.0))
            for g in THERMAL:
                if pmax[g] <= free[g] + TOL_MW:      # no excess headroom
                    continue
                if res.p_gen[g] < free[g] - TOL_MW:  # interior / below kink
                    continue
                if e_lo[g] > TOL_MW:
                    n["excl_ramp_forced"] += 1
                    continue
                if skip:
                    n[skip] += 1
                    continue
                lmp = float(src.lmp[dcf.GEN_BUS[g]])
                if abs(lmp - voll) <= 1e-6 * max(1.0, voll):
                    n["excl_voll_lmp"] += 1
                    continue
                n["cand"] += 1
                th_res.append((lmp + yr - c[g]) / a[g])
                th_raw.append((lmp - c[g]) / a[g])
                th_heat.append((lmp + yr - c[g] * heat[g]) / a[g])
                who.append(g)
                info.append((int(env._step), float(lmp), float(yr), float(ctx["spot"]),
                             bool(src is not res)))
        obs, _, _, _ = env.step(action_from_opf(res, ctx["load"]))
    s = env.episode_summary()
    return {"seed": seed, "mu": mu, "th_res": th_res, "th_raw": th_raw, "th_heat": th_heat,
            "who": who, "info": info, "counts": n, "episode_cost": s["episode_cost"],
            "unserved_mwh": s["unserved_mwh"], "fd": fds}


def stats_of(th) -> dict:
    t = np.asarray(th, dtype=float)
    pos = t[t > 0]
    if pos.size == 0:
        return {"max": None, "p99": None, "p95": None, "n_pos": 0}
    return {"max": float(pos.max()), "p99": float(np.percentile(pos, 99)),
            "p95": float(np.percentile(pos, 95)), "n_pos": int(pos.size)}


def certificate(cfg, net, fd_budget=0) -> dict:
    aging = aging_from(cfg)
    mh = mu_hi_of(cfg, aging)
    mu = MU_START if mh <= 0 else min(MU_START, 0.5 * mh)
    iters = []
    steps = 0
    with Applied(cfg):
        for k in range(N_ITER):
            eps = [cert_episode(cfg, aging, net, mu, s, fd_budget if k == 0 else 0)
                   for s in TRAIN_SEEDS]
            steps += sum(e["counts"]["steps"] for e in eps)
            cat = lambda key: sum((e[key] for e in eps), [])
            who = np.array(cat("who"), dtype=int)
            st = {v: stats_of(cat(v)) for v in ("th_res", "th_raw", "th_heat")}
            thr = np.asarray(cat("th_res"))
            arg = int(np.argmax(thr)) if thr.size else -1
            inf = cat("info")
            it = {"mu": mu, "stats": st,
                  "argmax_unit": int(who[arg]) if arg >= 0 else None,
                  "argmax_info": (dict(zip(("step", "lmp", "y_res", "spot", "counterfactual"),
                                           inf[arg])) if arg >= 0 else None),
                  "counts": {k2: int(sum(e["counts"][k2] for e in eps)) for k2 in eps[0]["counts"]},
                  "train_cost": float(np.mean([e["episode_cost"] for e in eps])),
                  "train_unserved": float(np.mean([e["unserved_mwh"] for e in eps])),
                  "fd": sum((e["fd"] for e in eps), [])}
            iters.append(it)
            new = st["th_res"]["max"]
            if new is None:
                break
            conv = abs(np.log(new / mu)) < 0.02
            mu = new
            if conv:
                break
    last = iters[-1]
    fixed = last["stats"]["th_res"]["max"]
    converged = fixed is not None and abs(np.log(fixed / last["mu"])) < 0.02
    return {"mu_hi": mh, "iters": iters, "mu_lo": fixed,
            "mu_lo_p99": last["stats"]["th_res"]["p99"],
            "mu_lo_p95": last["stats"]["th_res"]["p95"],
            "mu_lo_raw": last["stats"]["th_raw"]["max"],
            "mu_lo_heat": last["stats"]["th_heat"]["max"],
            "mu_lo_oneshot": iters[0]["stats"]["th_res"]["max"],
            "converged": bool(converged), "env_steps": steps, "lp_iters": len(iters)}


# ---------------------------------------------------------------------------
# Stored-curve evaluation
# ---------------------------------------------------------------------------

def interp_curve(curve: dict, mu: float):
    """Piecewise-linear in log(mu) on positive grid; flat beyond the top."""
    pts = sorted((float(k), v) for k, v in curve.items() if float(k) > 0)
    xs = np.log([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    if mu is None or not np.isfinite(mu) or mu <= 0:
        return None, "invalid"
    lm = np.log(mu)
    if lm < xs[0] - 1e-9:
        return float(ys[0]), "below_grid"
    if lm >= xs[-1]:
        return float(ys[-1]), "above_grid"
    return float(np.interp(lm, xs, ys)), "interp"


def curve_from_detail(row, seeds) -> dict:
    out = {}
    for key, v in row["test_detail"].items():
        m, s = key.split("|")
        out.setdefault(float(m), {})[int(s)] = v["episode_cost"]
    return {m: float(np.mean([d[s] for s in seeds])) for m, d in out.items()
            if all(s in d for s in seeds)}


def ctrl_eval(row, mu_ctrl) -> dict:
    curve = {float(k): v for k, v in row["test_curve"].items()}
    best = min(curve.values())
    c, where = interp_curve(curve, mu_ctrl)
    if c is None:
        return {"mu": mu_ctrl, "gap_pct": None, "where": where, "within": False}
    g = 100.0 * (c / best - 1.0)
    return {"mu": mu_ctrl, "gap_pct": g, "where": where, "within": bool(g <= 100 * FLAT_TOL + 1e-9)}


def key_of(row, block):
    if block == "oat":
        return f"oat:{row['name']}" + ("" if row["level"] is None else f"={row['level']:g}")
    return f"lhs:{row['name']}"


def spot_check(row, mu_ctrl, net) -> dict:
    cfg = row["params"]
    aging = aging_from(cfg)
    seeds = (1, 2, 3)
    costs = []
    with Applied(cfg):
        for s in seeds:
            r = run_episode(make(cfg, aging, s), OPFAware(net, aging, float(mu_ctrl)))
            costs.append(r["episode_cost"])
    curve = curve_from_detail(row, seeds)
    best = min(curve.values())
    pos = {m: v for m, v in curve.items() if m > 0}
    emp3 = min(m for m, v in pos.items() if v <= best * (1 + FLAT_TOL))
    mean = float(np.mean(costs))
    return {"mu_ctrl": mu_ctrl, "costs": costs, "mean": mean, "best_stored_seeds123": best,
            "best_mu_seeds123": min(curve, key=curve.get), "mu_emp_seeds123": emp3,
            "gap_pct": 100.0 * (mean / best - 1.0),
            "interp_gap_pct_seeds123": 100.0 * (interp_curve(curve, mu_ctrl)[0] / best - 1.0)}


# ---------------------------------------------------------------------------

def corr_block(rows, key) -> dict:
    from scipy import stats
    ok = [r for r in rows if r.get(key) and r[key] > 0 and r["mu_emp"]]
    if len(ok) < 3:
        return {"n": len(ok)}
    e = np.log([r["mu_emp"] for r in ok]); c = np.log([r[key] for r in ok])
    ratio = np.exp(c - e)
    return {"n": len(ok), "log_pearson": float(np.corrcoef(e, c)[0, 1]),
            "spearman": float(stats.spearmanr(e, c).statistic),
            "ratio_median": float(np.median(ratio)),
            "ratio_p5": float(np.percentile(ratio, 5)), "ratio_p95": float(np.percentile(ratio, 95)),
            "ratio_min": float(ratio.min()), "ratio_max": float(ratio.max()),
            "frac_within_2x": float(np.mean((ratio >= 0.5) & (ratio <= 2.0))),
            "frac_cert_ge_emp_over2": float(np.mean(ratio >= 0.5)),
            "rmse_log2": float(np.sqrt(np.mean(np.log2(ratio) ** 2)))}


def summarise(rows) -> dict:
    out = {}
    keys = ["mu_cf_gas_paper", "mu_cf_max_paper", "mu_lo", "mu_lo_p99", "mu_lo_p95",
            "mu_lo_raw", "mu_lo_heat", "mu_lo_oneshot"]
    r4_rows = [r for r in rows if r["r4_threshold_row"]]
    for tag, sub in (("oat", [r for r in r4_rows if r["block"] == "oat"]),
                     ("lhs", [r for r in r4_rows if r["block"] == "lhs"]),
                     ("pooled", r4_rows), ("pooled_all42", rows)):
        out[tag] = {k: corr_block(sub, k) for k in keys}
    ctrl = {}
    for name in ("cert_2x", "cert_1x", "cert_4x", "cert_p95_2x", "cf_paper_1x", "cf_paper_2x"):
        ev = [r["ctrl"][name] for r in rows]
        g = [e["gap_pct"] for e in ev if e["gap_pct"] is not None]
        ctrl[name] = {"n": len(ev), "frac_within_0.5pct": float(np.mean([e["within"] for e in ev])),
                      "median_gap_pct": float(np.median(g)) if g else None,
                      "max_gap_pct": float(np.max(g)) if g else None,
                      "n_below_grid": int(sum(e["where"] == "below_grid" for e in ev)),
                      "failures": [r["key"] for r, e in zip(rows, ev) if not e["within"]]}
    out["control_on_stored_curves"] = ctrl
    # non-discriminative baselines: one fixed price for every configuration
    out["control_fixed_price_baseline"] = {
        f"{m:g}": {"frac_within_0.5pct": float(np.mean([ctrl_eval(r, m)["within"] for r in rows])),
                   "failures": [r["key"] for r in rows if not ctrl_eval(r, m)["within"]]}
        for m in (4000.0, 8000.0, 16000.0, 32000.0, 64000.0)}
    # validity of the certificate's claim: test curve flat for grid mu >= mu_lo
    dev = {}
    for r in rows:
        c = {float(k): v for k, v in r["test_curve"].items()}
        top = c[max(c)]
        ab = [100 * abs(v / top - 1) for m, v in c.items() if r["mu_lo"] and m >= 0.999 * r["mu_lo"]]
        dev[r["key"]] = max(ab) if ab else None
    v = [x for x in dev.values() if x is not None]
    out["flat_above_mu_lo_test"] = {
        "max_dev_pct_vs_top_of_grid": dev,
        "frac_dev_le_0.05pct": float(np.mean(np.array(v) <= 0.05)),
        "frac_dev_le_0.5pct": float(np.mean(np.array(v) <= 0.5)),
        "frac_dev_le_1pct": float(np.mean(np.array(v) <= 1.0))}
    lo_emp = np.array([r["mu_lo"] / r["mu_emp"] for r in rows if r["mu_lo"] and r["mu_emp"]])
    out["mu_lo_over_mu_emp_min_max"] = [float(lo_emp.min()), float(lo_emp.max())]
    st = [r["cert"]["env_steps"] for r in rows]
    out["env_steps_per_certificate"] = {"median": float(np.median(st)), "min": int(min(st)),
                                        "max": int(max(st)),
                                        "episodes_per_cert_median": float(np.median(st)) / 960}
    out["n_converged"] = int(sum(r["cert"]["converged"] for r in rows))
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--plot" in argv:
        return plot()
    if "--resummarise" in argv:
        with open(OUT_JSON, encoding="utf-8") as fh:
            payload = json.load(fh)
        payload["summary"] = summarise(payload["rows"])
        save(payload)
        print(json.dumps({k: v for k, v in payload["summary"].items()
                          if k.startswith(("control_fixed", "flat_above", "mu_lo_over"))}, indent=1)[:3000])
        return plot()
    with open(R4_JSON, encoding="utf-8") as fh:
        r4 = json.load(fh)
    net = build_5bus()
    rows = []
    t0 = time.time()
    jobs = [("oat", r) for r in r4["oat"]] + [("lhs", r) for r in r4["lhs"]]
    for i, (block, r) in enumerate(jobs):
        t = time.time()
        cert = certificate(r["params"], net, fd_budget=4 if i < 3 else 0)
        cf = r["closed_form"]
        mh, lo = cert["mu_hi"], cert["mu_lo"]

        def cap(m):
            if m is None:
                return None
            return min(m, 0.9 * mh) if mh > 0 else m

        ctrl = {
            "cert_1x": ctrl_eval(r, cap(lo)),
            "cert_2x": ctrl_eval(r, cap(None if lo is None else CTRL_MULT * lo)),
            "cert_4x": ctrl_eval(r, cap(None if lo is None else 4 * lo)),
            "cert_p95_2x": ctrl_eval(r, cap(None if cert["mu_lo_p95"] is None else 2 * cert["mu_lo_p95"])),
            "cf_paper_1x": ctrl_eval(r, cf["mu_cf_gas_paper"]),
            "cf_paper_2x": ctrl_eval(r, 2 * cf["mu_cf_gas_paper"]),
        }
        row = {"block": block, "name": r["name"], "level": r["level"], "key": key_of(r, block),
               "params": r["params"], "mu_emp": r["mu_emp"], "best_test_mu": r["best_test_mu"],
               "best_saving_pct": r["best_saving_pct"],
               "r4_threshold_row": bool(r["mu_emp"] and cf["mu_cf_gas_paper"] > 0
                                        and r["best_saving_pct"] > 0.5),
               "test_curve": r["test_curve"],
               "mu_cf_gas_paper": cf["mu_cf_gas_paper"], "mu_cf_max_paper": cf["mu_cf_max_paper"],
               "mu_lo": lo, "mu_lo_p99": cert["mu_lo_p99"], "mu_lo_p95": cert["mu_lo_p95"],
               "mu_lo_raw": cert["mu_lo_raw"], "mu_lo_heat": cert["mu_lo_heat"],
               "mu_lo_oneshot": cert["mu_lo_oneshot"], "mu_hi": mh,
               "cert": cert, "ctrl": ctrl}
        rows.append(row)
        print(f"[{i+1}/{len(jobs)}] {row['key']:24s} emp={r['mu_emp']:>7g} cf={cf['mu_cf_gas_paper']:8.0f} "
              f"lo={lo if lo is None else round(lo)} p95={cert['mu_lo_p95'] and round(cert['mu_lo_p95'])} "
              f"hi={mh:.0f} it={cert['lp_iters']} conv={cert['converged']} "
              f"argmax_g={cert['iters'][-1]['argmax_unit']} "
              f"gap2x={ctrl['cert_2x']['gap_pct']} ({time.time()-t:.0f}s)", flush=True)
        save({"rows": rows, "meta": meta(time.time() - t0)})
    summ = summarise(rows)
    payload = {"meta": meta(time.time() - t0), "summary": summ, "rows": rows}
    save(payload)
    # spot checks on real test seeds 1-3
    spots = []
    for row in rows:
        if row["key"] in SPOT_CHECK and row["ctrl"]["cert_2x"]["mu"] is not None:
            r = next(x for x in r4[row["block"]] if key_of(x, row["block"]) == row["key"])
            t = time.time()
            sc = spot_check(r, row["ctrl"]["cert_2x"]["mu"], net)
            sc["key"] = row["key"]
            spots.append(sc)
            print(f"spot {row['key']}: mu={sc['mu_ctrl']:.0f} gap={sc['gap_pct']:+.3f}% "
                  f"(interp {sc['interp_gap_pct_seeds123']:+.3f}%) ({time.time()-t:.0f}s)", flush=True)
            payload["spot_checks"] = spots
            save(payload)
    payload["meta"] = meta(time.time() - t0)
    save(payload)
    print(json.dumps(summ, indent=1))
    plot()
    return 0


def meta(el):
    return {"train_seeds": list(TRAIN_SEEDS), "n_iter_max": N_ITER, "mu_start": MU_START,
            "mu_start_rule": "min(MU_START, 0.5*mu_hi)", "flat_tol": FLAT_TOL,
            "ctrl_mult": CTRL_MULT, "ctrl_cap": "0.9*mu_hi",
            "r4_json": R4_JSON, "solver": "scipy.linprog HiGHS (dc_opf, exact LP)",
            "elapsed_s": el,
            "notes": "mu_lo = max over candidate (t,g) of (LMP + y_res - c_g)/alpha_g; "
                     "raw = without reserve dual; heat = c_g * heat_rate * eff; "
                     "r4_threshold_row = same row filter as R4 summary (mu_emp, cf>0, best saving>0.5%)."}


def save(payload):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, OUT_JSON)


# ---------------------------------------------------------------------------

def plot() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(OUT_JSON, encoding="utf-8") as fh:
        d = json.load(fh)
    rows = d["rows"]
    s = d.get("summary", {}).get("pooled", {})
    blue, orange, grey = "#2a6fb0", "#d9822b", "#6b6b6b"
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1), constrained_layout=True)
    lim = [1.2e2, 2e5]
    for ax, key, title in ((axes[0], "mu_cf_gas_paper", r"(a) closed form $(\bar\pi-c_{gas})/\alpha_{gas}$"),
                           (axes[1], "mu_lo", r"(b) dual certificate $\mu_{lo}$ (train seeds)")):
        for block, mk, col in (("oat", "s", orange), ("lhs", "o", blue)):
            pts = [(r[key], r["mu_emp"]) for r in rows if r["block"] == block
                   and r.get(key) and r[key] > 0 and r["mu_emp"]]
            if pts:
                x, y = np.array(pts).T
                ax.scatter(np.clip(x, *lim), y, s=22, marker=mk, facecolors="none",
                           edgecolors=col, lw=1.0, label=f"{block.upper()} (n={len(pts)})")
        ax.plot(lim, lim, color=grey, lw=0.9, label="y = x")
        ax.fill_between(lim, [v / 2 for v in lim], [v * 2 for v in lim], color=grey, alpha=0.12,
                        lw=0, label="factor 2")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lim); ax.set_ylim(lim)
        st = s.get(key, {})
        if st.get("n", 0) >= 3:
            ax.text(0.97, 0.04, f"log-r = {st['log_pearson']:.2f}\n"
                    f"Spearman = {st['spearman']:.2f}\nwithin 2x: {100*st['frac_within_2x']:.0f}%",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
        ax.set_xlabel(r"predicted threshold ($/wear-unit)", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel(r"empirical $\mu_{emp}$ (R4, factor-2 grid)", fontsize=9)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.savefig(OUT_FIG)
    fig.savefig(OUT_FIG.replace(".pdf", ".png"), dpi=200)
    print(f"wrote {OUT_FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
