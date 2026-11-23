"""R6 -- dual (LMP) certificate for the wear shadow price.

For each regime:
  1. certificate [mu_lo, mu_hi] by fixed-point iteration on TRAINING seeds
     (>= 101), see ``twoclock.agents.certificate``;
  2. empirical J(mu) on TEST seeds 1-5 over a dense log grid (+ blind);
  3. control: certificate price vs best constant price (train-seed sweep),
     ridge value-gradient learner, blind OPF -- test seeds 1-5;
  4. per-unit certificate prices (5-bus).
Networks: 5-bus (7 regimes), 14-bus and 30-bus (3 regimes, coarser grid).

Run:  python -m twoclock.experiments.r6_certificate          (full)
      python -m twoclock.experiments.r6_certificate --plot   (figure only)
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
from twoclock.agents import OPFAware, OPFBlind, ValueGradientAgent
from twoclock.agents.baselines import _battery_bounds
from twoclock.agents.certificate import (
    alpha_coef, fixed_point_certificate, priced_lp,
)
from twoclock.envs import make_env
from twoclock.experiments.runner import run_episode
from twoclock.paths import RESULTS_DIR
from twoclock.physics import AgingParams, build_5bus
from twoclock.physics.opf import wear_free_mw, wear_price_from_mu

DAYS = 10
TEST_SEEDS = (1, 2, 3, 4, 5)
CERT_SEEDS = (101, 102, 103, 104, 105)
CONST_TRAIN_SEEDS = (101, 102)
GRID = [float(x) for x in np.logspace(2, 6, 24)]
GRID_COARSE = GRID[::2]                       # 12 points, factor ~2.2
TOLS = (0.001, 0.0025, 0.01)
TOL_MAIN = 0.0025
OUT_JSON = os.path.join(RESULTS_DIR, "r6_certificate.json")
OUT_FIG = os.path.join(RESULTS_DIR, "fig_r6_certificate.pdf")

REGIMES_5 = [("normal", "normal", 1.0), ("hard", "hard", 1.0), ("peak", "peak", 1.0),
             ("storm", "storm", 1.0), ("tight1.15", "normal", 1.15),
             ("tight1.25", "normal", 1.25), ("tight1.35", "normal", 1.35)]
REGIMES_NET = [("normal", "normal", 1.0), ("peak", "peak", 1.0),
               ("tight1.25", "normal", 1.25)]


def log(*a):
    print(*a, flush=True)


def ci95(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def summ(x):
    return {"mean": float(np.mean(x)), "ci95": ci95(x), "per_seed": [float(v) for v in x]}


# ---------------------------------------------------------------------------
# network switching (same mechanism as stage2_transfer)
# ---------------------------------------------------------------------------

ORIG = (dcf.GEN_BUS.copy(), dcf.LOAD_SHARE.copy(), dcf.BATTERY_BUS, dcf.IMPORT_BUS)


def set_network(tag):
    if tag == "5bus":
        gb, ls, bb, ib = ORIG
        net = build_5bus()
    elif tag == "14bus":
        from twoclock.experiments.stage2_transfer import build_14bus
        net, gb, ls, bb, ib = build_14bus()
    else:
        from twoclock.experiments.rigorous_suite import build_30bus
        net, gb, ls, bb, ib = build_30bus()
    for mod in (dcf, opf_mod, env_mod):
        mod.GEN_BUS = gb
        mod.LOAD_SHARE = ls
        mod.BATTERY_BUS = bb
        mod.IMPORT_BUS = ib
    return net


class Ctx:
    def __init__(self, net, diff, ds, aging):
        self.net, self.diff, self.ds, self.aging = net, diff, ds, aging
        self.cache = {}
        self.steps = {}

    def env(self, seed, days=DAYS):
        e = make_env(self.diff, days=days, seed=seed, demand_scale=self.ds)
        e.net = self.net
        return e

    def run_mu(self, mu, seed):
        key = (round(float(mu), 6) if np.ndim(mu) == 0 else tuple(np.round(mu, 3)), seed)
        if key not in self.cache:
            if np.ndim(mu) == 0 and mu <= 0:
                ag = OPFBlind(self.net, self.aging)
            elif np.ndim(mu) == 0:
                ag = OPFAware(self.net, self.aging, float(mu))
            else:
                ag = OPFAware(self.net, self.aging, 1.0)
                ag.mu = np.asarray(mu, dtype=float)
            r = run_episode(self.env(seed), ag)
            self.cache[key] = {k: float(r[k]) for k in
                               ("episode_cost", "unserved_mwh", "overload_mwh",
                                "wear_end_mean", "heat_rate_mean")}
        return self.cache[key]

    def costs(self, mu, seeds):
        return [self.run_mu(mu, s)["episode_cost"] for s in seeds]


def empirical_set(curve: dict, tol: float):
    """curve: {mu: mean cost} incl. 0.  Returns onset/upper of the near-optimal
    contiguous region around the best positive-grid point."""
    mus = sorted(m for m in curve if m > 0)
    best = min(curve[m] for m in mus)
    ok = [curve[m] <= best * (1 + tol) for m in mus]
    ib = int(np.argmin([curve[m] for m in mus]))
    lo = hi = ib
    while lo - 1 >= 0 and ok[lo - 1]:
        lo -= 1
    while hi + 1 < len(mus) and ok[hi + 1]:
        hi += 1
    return {"onset": mus[lo], "upper": mus[hi], "upper_is_grid_top": hi == len(mus) - 1,
            "onset_prev": mus[lo - 1] if lo > 0 else None,
            "upper_next": mus[hi + 1] if hi + 1 < len(mus) else None,
            "first_any_within": next((m for m, o in zip(mus, ok) if o), None),
            "best": best, "best_mu": mus[ib]}


def gap_pct(c, best):
    return 100.0 * (c / best - 1.0)


# ---------------------------------------------------------------------------
# LMP sign/unit check
# ---------------------------------------------------------------------------

def lmp_check(aging, mu=2000.0, steps=96, every=4):
    from twoclock.physics import dc_opf
    net = set_network("5bus")
    env = make_env("normal", days=1, seed=3, demand_scale=1.2)
    obs = env.reset()
    agent = OPFAware(net, aging, mu)
    free = wear_free_mw(aging)
    wp = wear_price_from_mu(mu, aging, dcf.GEN_RATING, dcf.GEN_DISPATCHABLE)
    max_diff = max_fd = 0.0
    n = 0
    lmps = []
    for k in range(steps):
        ctx = obs.ctx
        lo, hi = _battery_bounds(ctx["soc"], 0.25)
        kw = dict(reserve_req_mw=ctx["reserve_req"], battery_min_mw=lo, battery_max_mw=hi,
                  spot_price=ctx["spot"])
        args = (net, ctx["load"], ctx["prev_p"], ctx["renewable"], ctx["pmax_eff"])
        r1 = dc_opf(*args, wear_price_mwh=wp, wear_free_mw=free, **kw)
        r2 = priced_lp(*args, wear_price_mwh=wp, wear_free=free, **kw)
        max_diff = max(max_diff, abs(r1.cost - r2["cost"]), float(np.abs(r1.lmp - r2["lmp"]).max()))
        if k % every == 0:
            lmps.append(float(r2["lmp"][0]))
            for b in range(net.n_bus):
                bump = np.zeros(net.n_bus); bump[b] = 0.01
                rp = priced_lp(*args, wear_price_mwh=wp, wear_free=free, load_bump=bump, **kw)
                rm = priced_lp(*args, wear_price_mwh=wp, wear_free=free, load_bump=-bump, **kw)
                fd = (rp["cost"] - rm["cost"]) / 0.02
                max_fd = max(max_fd, abs(fd - r2["lmp"][b])); n += 1
        obs, _, _, _ = env.step(agent.act(obs))
    return {"mu": mu, "max_abs_diff_dc_opf_vs_certificate_lp": max_diff,
            "max_abs_err_lmp_vs_central_fd": max_fd, "n_fd_checks": n,
            "lmp_range_bus0": [min(lmps), max(lmps)],
            "conclusion": "res.eqlin.marginals[:n_bus] = +d(cost)/d(load_b) in $/MWh "
                          "(per-step LP objective is $/h; LMP in $/MWh)"}


# ---------------------------------------------------------------------------
# one regime
# ---------------------------------------------------------------------------

def train_ridge(c: Ctx):
    vg = ValueGradientAgent(c.net, c.aging, seed=7, mu_max=1e6)
    vg.enabled = False
    steps = 0
    for r in range(3):
        for s in (101, 102, 103):
            ep = run_episode(c.env(s + 10 * r, days=5), vg, learn=True)
            steps += int(ep["steps"])
        vg.update()
        vg.enabled = True
    return vg, steps


def evaluate_regime(net_tag, name, diff, ds, aging, grid, do_control=True, do_per_unit=True):
    t0 = time.time()
    net = set_network(net_tag)
    c = Ctx(net, diff, ds, aging)
    log(f"== {net_tag} {name} (difficulty={diff}, demand x{ds})")

    # 1. certificate (training seeds)
    cert = fixed_point_certificate(lambda s: c.env(s), net, aging, CERT_SEEDS, run_episode,
                                   log=log)
    mu_lo, mu_hi = cert["mu_lo"], cert["mu_hi"]
    mu_cert = float(np.sqrt(mu_lo * mu_hi)) if np.isfinite(mu_hi) else 2.0 * mu_lo
    log(f"   certificate [{mu_lo:.0f}, {mu_hi:.4g}]  price={mu_cert:.0f} "
        f"({cert['env_steps']} steps, {time.time() - t0:.0f}s)")

    # 2. empirical sweep (test seeds)
    curve = {0.0: float(np.mean(c.costs(0.0, TEST_SEEDS)))}
    for mu in grid:
        curve[mu] = float(np.mean(c.costs(mu, TEST_SEEDS)))
    emp = {str(t): empirical_set(curve, t) for t in TOLS}
    em = emp[str(TOL_MAIN)]
    best = em["best"]
    J0 = curve[0.0]
    log(f"   empirical (tol {TOL_MAIN}): onset={em['onset']:.0f} upper={em['upper']:.0f}"
        f"{' (grid top)' if em['upper_is_grid_top'] else ''} best={best:.4g} blind={J0:.4g}"
        f"  ({time.time() - t0:.0f}s)")

    # certificate evaluated on test seeds (off-grid points)
    def pt(mu):
        cs = c.costs(mu, TEST_SEEDS)
        m = float(np.mean(cs))
        return {"mu": mu, "cost": m, "gap_pct_of_best": gap_pct(m, best),
                "frac_of_saving_lost": (m - best) / (J0 - best) if J0 > best else None}

    probes = {"mu_lo": pt(mu_lo), "mu_cert": pt(mu_cert), "mu_lo/2": pt(mu_lo / 2),
              "2*mu_lo": pt(2 * mu_lo)}
    if np.isfinite(mu_hi):
        probes["mu_hi/1.5"] = pt(mu_hi / 1.5)
        probes["1.5*mu_hi"] = pt(mu_hi * 1.5)
    for k, v in probes.items():
        log(f"     J({k}={v['mu']:.0f}) gap {v['gap_pct_of_best']:+.3f}%")

    out = {"network": net_tag, "regime": name, "difficulty": diff, "demand_scale": ds,
           "certificate": cert, "mu_cert_price": mu_cert,
           "rule": "mu_cert = sqrt(mu_lo*mu_hi) if mu_hi finite else 2*mu_lo",
           "test_curve": {f"{k:g}": v for k, v in sorted(curve.items())},
           "test_curve_per_seed": {f"{k:g}": c.costs(k, TEST_SEEDS) for k in sorted(curve)},
           "empirical": emp, "probes": probes,
           "ratio_mu_lo_over_onset": mu_lo / em["onset"],
           "ratio_mu_hi_over_upper": (mu_hi / em["upper"]) if np.isfinite(mu_hi) else None}

    # 3. control comparison
    if do_control:
        tr = {0.0: float(np.mean(c.costs(0.0, CONST_TRAIN_SEEDS)))}
        for mu in GRID_COARSE:
            tr[mu] = float(np.mean(c.costs(mu, CONST_TRAIN_SEEDS)))
        mu_star = min(tr, key=tr.get)
        const_steps = len(tr) * len(CONST_TRAIN_SEEDS) * DAYS * 96
        vg, vg_steps = train_ridge(c)
        vg_rows = [run_episode(c.env(s), vg) for s in TEST_SEEDS]
        vg_costs = [float(r["episode_cost"]) for r in vg_rows]
        methods = {
            "blind": (c.costs(0.0, TEST_SEEDS), 0, 0.0),
            "best_constant_train_sweep": (c.costs(mu_star, TEST_SEEDS), const_steps, mu_star),
            "certificate_mid": (c.costs(mu_cert, TEST_SEEDS), cert["env_steps"], mu_cert),
            "certificate_lo": (c.costs(mu_lo, TEST_SEEDS), cert["env_steps"], mu_lo),
            "ridge_value_gradient": (vg_costs, vg_steps, float(vg.mu_scalar)),
            "oracle_best_test_grid": (c.costs(em["best_mu"], TEST_SEEDS), None, em["best_mu"]),
        }
        ctl = {}
        ref_c = np.array(methods["best_constant_train_sweep"][0])
        ref_b = np.array(methods["blind"][0])
        for k, (cs, st, mu) in methods.items():
            cs = np.array(cs)
            ctl[k] = {"mu": mu, "env_steps": st, "cost": summ(cs),
                      "minus_best_constant": summ(cs - ref_c),
                      "minus_blind": summ(cs - ref_b),
                      "saving_vs_blind_pct": float(100 * np.mean((ref_b - cs) / ref_b))}
            log(f"     {k:28s} mu={mu:9.0f} cost={cs.mean()/1e6:7.4f}+-{ci95(cs)/1e6:.4f}M "
                f"d_const={np.mean(cs - ref_c)/1e3:+8.1f}k+-{ci95(cs - ref_c)/1e3:.1f}k "
                f"steps={st}")
        out["control"] = ctl
        out["train_curve_const"] = {f"{k:g}": v for k, v in tr.items()}

    # 4. per-unit certificate
    if do_per_unit:
        lo_g = np.array(cert["per_unit_lo"]); hi_g = np.array(cert["per_unit_hi"])
        mu_g = np.where(np.isfinite(hi_g), np.sqrt(np.maximum(lo_g, 1.0) * hi_g),
                        2.0 * np.maximum(lo_g, 1.0))
        mu_vec = np.concatenate([mu_g, [0.0]])
        cs = np.array(c.costs(mu_vec, TEST_SEEDS))
        ref = np.array(c.costs(mu_cert, TEST_SEEDS))
        out["per_unit"] = {"mu_g": mu_g.tolist(), "cost": summ(cs),
                           "minus_scalar_certificate": summ(cs - ref),
                           "gap_pct_of_best": gap_pct(float(cs.mean()), best)}
        log(f"     per-unit mu_g={np.round(mu_g, 0)} gap {gap_pct(float(cs.mean()), best):+.3f}% "
            f"vs scalar cert {np.mean(cs - ref)/1e3:+.1f}k")
    out["episodes_run"] = len(c.cache)
    out["wall_s"] = time.time() - t0
    log(f"   done in {out['wall_s']:.0f}s")
    return out


def save(payload):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, OUT_JSON)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--plot" in argv:
        return plot()
    aging = AgingParams()
    a = alpha_coef(aging)
    payload = {"meta": {
        "days": DAYS, "test_seeds": TEST_SEEDS, "cert_seeds": CERT_SEEDS,
        "const_train_seeds": CONST_TRAIN_SEEDS, "grid": GRID, "grid_coarse": GRID_COARSE,
        "tols": TOLS, "tol_main": TOL_MAIN, "alpha": a.tolist(),
        "gen_cost": dcf.GEN_COST.tolist(), "voll": opf_mod.VOLL,
        "voll_caps": ((opf_mod.VOLL - dcf.GEN_COST[:3]) / a[:3]).tolist(),
        "notes": "certificate from training seeds only; sweeps on test seeds 1-5. "
                 "Networks 14/30-bus use stage2_transfer global-swap mechanism "
                 "(env still meters import/battery at bus 4)."}, "lmp_check": None,
        "five_bus": [], "networks": []}
    t0 = time.time()
    payload["lmp_check"] = lmp_check(aging)
    log("LMP check:", payload["lmp_check"])
    save(payload)
    only = os.environ.get("R6_ONLY")
    if only != "net":
        for name, diff, ds in REGIMES_5:
            payload["five_bus"].append(evaluate_regime("5bus", name, diff, ds, aging, GRID))
            payload["meta"]["elapsed_s"] = time.time() - t0
            save(payload)
    if only != "5bus":
        for tag in ("14bus", "30bus"):
            for name, diff, ds in REGIMES_NET:
                payload["networks"].append(evaluate_regime(tag, name, diff, ds, aging,
                                                           GRID_COARSE, do_control=False,
                                                           do_per_unit=False))
                payload["meta"]["elapsed_s"] = time.time() - t0
                save(payload)
        set_network("5bus")
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
    rows = {(r["network"], r["regime"]): r for r in d["five_bus"] + d["networks"]}
    panels = [[("5bus", "normal"), ("5bus", "peak")],
              [("5bus", "tight1.15"), ("5bus", "tight1.25"), ("5bus", "tight1.35")],
              [("14bus", "tight1.25"), ("30bus", "tight1.25")]]
    titles = ["(a) 5-bus, nominal / peak", "(b) 5-bus, tight demand",
              "(c) 14/30-bus, demand x1.25"]
    cols = ["#2a6fb0", "#d9822b", "#3a9a5b"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), constrained_layout=True, sharey=True)
    for ax, keys, title in zip(axes, panels, titles):
        for i, key in enumerate(keys):
            r = rows.get(key)
            if r is None:
                continue
            cur = {float(k): v for k, v in r["test_curve"].items() if float(k) > 0}
            mus = np.array(sorted(cur))
            best = r["empirical"][str(d["meta"]["tol_main"])]["best"]
            gap = np.array([100 * (cur[m] / best - 1) for m in mus])
            lab = key[1].replace("tight", "x") if key[0] == "5bus" else key[0]
            ax.plot(mus, np.maximum(gap, 1e-3), "-o", ms=2.2, lw=1.1, color=cols[i], label=lab)
            lo, hi = r["certificate"]["mu_lo"], r["certificate"]["mu_hi"]
            y = 10 ** (1.55 - 0.32 * i)
            hi_plot = hi if hi is not None and np.isfinite(hi) else 1e6
            fin = hi is not None and np.isfinite(hi)
            ax.plot([lo, hi_plot], [y, y], color=cols[i], lw=1.4, solid_capstyle="butt")
            ax.plot([lo], [y], marker="|", ms=6, mew=1.4, color=cols[i])
            ax.plot([hi_plot], [y], marker="|" if fin else ">", ms=6 if fin else 4,
                    mew=1.4, color=cols[i])
        ax.axhline(100 * d["meta"]["tol_main"], color="#6b6b6b", lw=0.7, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1e2, 1e6); ax.set_ylim(1e-3, 80)
        ax.set_xlabel(r"wear price $\mu$ [\$/wear-unit]", fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=6, frameon=False, loc="lower left")
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("test cost above best [%]", fontsize=7)
    axes[0].text(1.2e2, 1.4e-3, "bars: certificate $[\\mu_{lo},\\mu_{hi}]$\n"
                 "(training seeds); dotted: 0.25% tol.",
                 fontsize=5.5, color="#444", va="bottom")
    axes[0].legend(fontsize=6, frameon=False, loc="center left")
    for ax in axes:
        ax.set_yticks([1e-3, 1e-2, 1e-1, 1, 10])
        ax.set_yticklabels([r"$\leq$0.001", "0.01", "0.1", "1", "10"])
    fig.savefig(OUT_FIG)
    fig.savefig(OUT_FIG.replace(".pdf", ".png"), dpi=200)
    log(f"wrote {OUT_FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
