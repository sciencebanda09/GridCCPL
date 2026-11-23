"""R4 -- robustness of the two headline claims to the physical parameters.

Claims under test
-----------------
(A) A constant wear price mu*, chosen on *training* seeds (101, 102) only,
    beats degradation-blind OPF on held-out test seeds.
(B) Cost is a step function of mu: above an empirical threshold mu_emp the
    curve is flat, so 2x and 4x over-pricing cost (almost) nothing, and the
    threshold sits near the closed-form switching hyperplane
        mu_cf = (pi_peak - c_gas) / alpha_gas,
        alpha_gas = c_age * c_age_gen[gas] / ((1 - sigma) * rating_gas).

Design
------
* Nominal configuration + one-at-a-time (OAT) low/high perturbations,
  evaluated on test seeds 1-5.
* Latin-hypercube (LHS) joint perturbation of 15 parameters, test seeds 1-3.
* 10-day episodes, ``normal`` regime.  mu* is picked from a coarse grid on
  training seeds only; the test curve uses a factor-2 log grid, so mu_emp has
  factor-2 resolution and 2x / 4x mu_emp are grid points (evaluated extra if
  they fall off the grid).
* Perturbations are applied in memory only (AgingParams instance, in-place
  GEN_COST array, opf.VOLL global + env voll, spot multiplier, demand scale)
  and restored after each configuration.  No package file is modified.

Run:  python -m twoclock.experiments.r4_robust            (full study)
      python -m twoclock.experiments.r4_robust --plot     (figure from JSON)
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace

import numpy as np

from ..agents import OPFAware, OPFBlind
from ..envs.two_clock_env import EnvConfig, TwoClockEnv
from ..physics import AgingParams, build_5bus
from ..physics import dc_flow as dcf
from ..physics import opf as opf_mod
from ..paths import RESULTS_DIR
from .runner import run_episode

DAYS = 10
TRAIN_SEEDS = (101, 102)
OAT_SEEDS = (1, 2, 3, 4, 5)
LHS_SEEDS = (1, 2, 3)
TRAIN_GRID = (0.0, 250.0, 1000.0, 4000.0, 16000.0, 64000.0, 256000.0)
OAT_GRID = (500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 32000.0, 64000.0)
LHS_GRID = (250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 32000.0,
            64000.0, 128000.0)
FLAT_TOL = 0.005           # "within 0.5% of best"
N_LHS = int(os.environ.get("R4_NLHS", "30"))
LHS_SEED = 20261122

BASE_COST = dcf.GEN_COST.copy()
BASE_VOLL = float(opf_mod.VOLL)
RATING_GAS = float(dcf.GEN_RATING[0])
RATING_COAL = float(dcf.GEN_RATING[1])
PI_PEAK_PAPER = 117.0      # $/MWh, convention used in the closed-form predictor
PI_PEAK_PROFILE = 120.0    # max of the deterministic spot profile 80 + 40*bump

OUT_JSON = os.path.join(RESULTS_DIR, "r4_robust.json")
OUT_FIG = os.path.join(RESULTS_DIR, "fig_r4_robust.pdf")


# ---------------------------------------------------------------------------
# Configuration handling
# ---------------------------------------------------------------------------

NOMINAL = {
    "c_age": 0.55, "c_age_gas": 1.0, "c_age_coal": 1.7, "c_age_nuc": 3.2,
    "stress_frac": 0.85, "tau_thermal": 3.0,
    "derate_lin": 0.10, "derate_quad": 1.20, "heat_lin": 0.10, "heat_quad": 0.80,
    "c_gas": 65.0, "c_coal": 45.0, "c_nuc": 20.0,
    "spot_scale": 1.0, "voll": 5000.0, "demand_scale": 1.0,
}

# (name, kind, lo, hi) for the LHS; kind "log" = log-uniform, "lin" = uniform
LHS_SPACE = [
    ("c_age", "log", 0.55 / 2, 0.55 * 2),
    ("c_age_gas", "log", 0.5, 2.0),
    ("c_age_coal", "log", 1.7 / 2, 1.7 * 2),
    ("c_age_nuc", "log", 3.2 / 2, 3.2 * 2),
    ("stress_frac", "lin", 0.75, 0.92),
    ("tau_thermal", "log", 1.0, 12.0),
    ("derate_lin", "lin", 0.0, 0.3),
    ("derate_quad", "log", 0.6, 2.4),
    ("heat_lin", "lin", 0.0, 0.3),
    ("heat_quad", "log", 0.4, 1.6),
    ("c_gas", "lin", 65.0 * 0.8, 65.0 * 1.2),
    ("c_coal", "lin", 45.0 * 0.8, 45.0 * 1.2),
    ("c_nuc", "lin", 20.0 * 0.8, 20.0 * 1.2),
    ("spot_scale", "lin", 0.8, 1.25),
    ("voll", "lin", 0.0, 10000.0),
    ("demand_scale", "lin", 0.9, 1.2),
]

OAT_LEVELS = [
    ("c_age", [0.275, 1.10]),
    ("c_age_gas", [0.5, 2.0]),
    ("stress_frac", [0.75, 0.92]),
    ("tau_thermal", [1.0, 12.0]),
    ("derate", [0.5, 2.0]),          # multiplies derate_lin and derate_quad
    ("heat", [0.5, 2.0]),            # multiplies heat_lin and heat_quad
    ("c_gas", [52.0, 78.0]),
    ("spot_scale", [0.8, 1.25]),
    ("voll", [0.0, 1000.0, 10000.0]),
    ("demand_scale", [0.9, 1.2]),
]


def oat_config(name: str, level: float) -> dict:
    cfg = dict(NOMINAL)
    if name == "derate":
        cfg["derate_lin"] *= level
        cfg["derate_quad"] *= level
    elif name == "heat":
        cfg["heat_lin"] *= level
        cfg["heat_quad"] *= level
    else:
        cfg[name] = float(level)
    return cfg


def lhs_configs(n: int) -> list[dict]:
    from scipy.stats import qmc
    u = qmc.LatinHypercube(d=len(LHS_SPACE), seed=LHS_SEED).random(n)
    out = []
    for row in u:
        cfg = dict(NOMINAL)
        for (name, kind, lo, hi), x in zip(LHS_SPACE, row):
            if kind == "log":
                cfg[name] = float(np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo))))
            else:
                cfg[name] = float(lo + x * (hi - lo))
        out.append(cfg)
    return out


def aging_from(cfg: dict) -> AgingParams:
    return replace(
        AgingParams(),
        c_age=cfg["c_age"],
        c_age_gen=(cfg["c_age_gas"], cfg["c_age_coal"], cfg["c_age_nuc"], 0.0),
        stress_frac=cfg["stress_frac"], tau_thermal=cfg["tau_thermal"],
        derate_lin=cfg["derate_lin"], derate_quad=cfg["derate_quad"],
        heat_lin=cfg["heat_lin"], heat_quad=cfg["heat_quad"],
    )


def closed_form(cfg: dict) -> dict:
    """Switching hyperplanes (pi_peak - c_g)/alpha_g, alpha_g = dWear$/dMWh per mu."""
    one_m = 1.0 - cfg["stress_frac"]
    a_gas = cfg["c_age"] * cfg["c_age_gas"] / (one_m * RATING_GAS)
    a_coal = cfg["c_age"] * cfg["c_age_coal"] / (one_m * RATING_COAL)
    out = {"alpha_gas": a_gas, "alpha_coal": a_coal}
    for tag, pi in (("paper", PI_PEAK_PAPER), ("profile", PI_PEAK_PROFILE)):
        pk = pi * cfg["spot_scale"]
        gas = (pk - cfg["c_gas"]) / a_gas
        coal = (pk - cfg["c_coal"]) / a_coal
        out[f"pi_peak_{tag}"] = pk
        out[f"mu_cf_gas_{tag}"] = gas
        out[f"mu_cf_max_{tag}"] = max(gas, coal)
    return out


class SpotScaledEnv(TwoClockEnv):
    """Same env, spot price multiplied by ``spot_scale`` (identical RNG use)."""

    spot_scale = 1.0

    def _spot(self) -> float:
        h = self._hour()
        peak = 40 * (np.exp(-0.5 * ((h - 8) / 2) ** 2) + np.exp(-0.5 * ((h - 19) / 2) ** 2))
        raw = 80 + peak + self.rng.normal(0, 12)
        return float(np.clip(self.spot_scale * raw, 20, 400))


class Applied:
    """Context manager: apply fuel cost / VOLL globals, restore on exit."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def __enter__(self):
        dcf.GEN_COST[:] = [self.cfg["c_gas"], self.cfg["c_coal"], self.cfg["c_nuc"],
                           BASE_COST[3]]
        opf_mod.VOLL = float(self.cfg["voll"])
        return self

    def __exit__(self, *exc):
        dcf.GEN_COST[:] = BASE_COST
        opf_mod.VOLL = BASE_VOLL
        return False


def make(cfg: dict, aging: AgingParams, seed: int) -> TwoClockEnv:
    ec = EnvConfig(days=DAYS, difficulty="normal", seed=seed, aging=aging,
                   demand_scale=cfg["demand_scale"], voll=cfg["voll"])
    env = SpotScaledEnv(ec)
    env.spot_scale = cfg["spot_scale"]
    env.name = "normal"
    return env


# ---------------------------------------------------------------------------
# Evaluation of one configuration
# ---------------------------------------------------------------------------

def evaluate_config(cfg: dict, seeds, grid, net) -> dict:
    aging = aging_from(cfg)
    cache: dict[tuple, dict] = {}

    def run(mu: float, seed: int) -> dict:
        key = (float(mu), int(seed))
        if key not in cache:
            agent = OPFBlind(net, aging) if mu == 0 else OPFAware(net, aging, float(mu))
            r = run_episode(make(cfg, aging, seed), agent)
            cache[key] = {k: r[k] for k in ("episode_cost", "unserved_mwh",
                                            "wear_end_mean", "heat_rate_mean")}
        return cache[key]

    def mean_cost(mu, ss):
        return float(np.mean([run(mu, s)["episode_cost"] for s in ss]))

    with Applied(cfg):
        # (A) mu* on training seeds only
        train_curve = {mu: mean_cost(mu, TRAIN_SEEDS) for mu in TRAIN_GRID}
        mu_star = min(train_curve, key=train_curve.get)
        # test curve
        test_mus = [0.0] + sorted(set(grid) | {mu_star} - {0.0})
        curve = {mu: mean_cost(mu, seeds) for mu in test_mus}
        pos = {m: c for m, c in curve.items() if m > 0}
        best = min(curve.values())
        mu_emp = min(m for m, c in pos.items() if c <= best * (1 + FLAT_TOL)) \
            if any(c <= best * (1 + FLAT_TOL) for c in pos.values()) else None
        extra = {}
        if mu_emp is not None:
            for f in (2.0, 4.0):
                m = mu_emp * f
                extra[f] = mean_cost(m, seeds)
                curve.setdefault(m, extra[f])
        best = min(curve.values())   # may improve with extra points
        # high-price tail: largest grid price (half-line upper end)
        mu_top = max(grid)

    blind = [run(0.0, s)["episode_cost"] for s in seeds]
    priced = [run(mu_star, s)["episode_cost"] for s in seeds]
    sav = [100.0 * (b - p) / b for b, p in zip(blind, priced)]
    J0 = curve[0.0]
    achievable = max(1e-9, J0 - best)

    def gap(m):
        c = curve[m]
        return {"mu": m, "cost": c, "gap_pct_of_best": 100.0 * (c / best - 1.0),
                "gap_pct_of_saving": 100.0 * (c - best) / achievable if J0 > best else None,
                "within_tol": bool(c <= best * (1 + FLAT_TOL))}

    flat = {}
    if mu_emp is not None:
        flat = {"2x": gap(mu_emp * 2.0), "4x": gap(mu_emp * 4.0)}
    cf = closed_form(cfg)
    best_mu = min(curve, key=curve.get)
    return {
        "params": cfg,
        "seeds": list(seeds),
        "train_curve": {str(k): v for k, v in train_curve.items()},
        "mu_star": mu_star,
        "test_curve": {str(k): v for k, v in sorted(curve.items())},
        "test_detail": {f"{k[0]:g}|{k[1]}": v for k, v in cache.items() if k[1] < 100},
        "blind_costs": blind, "priced_costs": priced,
        "saving_pct_per_seed": sav,
        "saving_pct": float(np.mean(sav)),
        "saving_pct_min_seed": float(np.min(sav)),
        "best_test_mu": best_mu,
        "best_saving_pct": 100.0 * (J0 - best) / J0,
        "mu_emp": mu_emp,
        "mu_emp_at_grid_bottom": bool(mu_emp is not None and mu_emp == min(grid)),
        "flat": flat,
        "top_of_grid": gap(mu_top),
        "blind_unserved_mwh": float(np.mean([run(0.0, s)["unserved_mwh"] for s in seeds])),
        "priced_unserved_mwh": float(np.mean([run(mu_star, s)["unserved_mwh"] for s in seeds])),
        "blind_wear": float(np.mean([run(0.0, s)["wear_end_mean"] for s in seeds])),
        "priced_wear": float(np.mean([run(mu_star, s)["wear_end_mean"] for s in seeds])),
        "closed_form": cf,
        "n_episodes": len(cache),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def save(payload):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, OUT_JSON)


def summarise(payload) -> dict:
    out = {}
    for block in ("oat", "lhs"):
        rows = payload.get(block, [])
        if not rows:
            continue
        s = np.array([r["saving_pct"] for r in rows])
        smin = np.array([r["saving_pct_min_seed"] for r in rows])
        d = {
            "n": len(rows),
            "frac_saving_pos": float(np.mean(s > 0)),
            "frac_saving_pos_every_seed": float(np.mean(smin > 0)),
            "saving_median": float(np.median(s)),
            "saving_p5": float(np.percentile(s, 5)),
            "saving_p95": float(np.percentile(s, 95)),
            "saving_min": float(s.min()), "saving_max": float(s.max()),
        }
        fl = [r for r in rows if r["flat"]]
        for f in ("2x", "4x"):
            d[f"frac_within_tol_{f}"] = float(np.mean([r["flat"][f]["within_tol"] for r in fl])) if fl else None
            d[f"max_gap_pct_{f}"] = float(np.max([r["flat"][f]["gap_pct_of_best"] for r in fl])) if fl else None
        ok = [r for r in rows if r["mu_emp"] and r["closed_form"]["mu_cf_gas_paper"] > 0
              and r["best_saving_pct"] > 0.5]
        if len(ok) >= 3:
            from scipy import stats
            e = np.log([r["mu_emp"] for r in ok])
            for tag in ("mu_cf_gas_paper", "mu_cf_max_paper", "mu_cf_gas_profile"):
                c = np.log([max(1.0, r["closed_form"][tag]) for r in ok])
                d[f"log_pearson_{tag}"] = float(np.corrcoef(e, c)[0, 1])
                d[f"spearman_{tag}"] = float(stats.spearmanr(e, c).statistic)
            ratio = np.exp(e) / np.array([r["closed_form"]["mu_cf_gas_paper"] for r in ok])
            d["ratio_emp_over_cf_gas_median"] = float(np.median(ratio))
            d["ratio_emp_over_cf_gas_p5_p95"] = [float(np.percentile(ratio, 5)),
                                                 float(np.percentile(ratio, 95))]
            d["frac_ratio_in_[0.5,2]"] = float(np.mean((ratio >= 0.5) & (ratio <= 2.0)))
            d["n_threshold_rows"] = len(ok)
        out[block] = d
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--plot" in argv:
        return plot()
    net = build_5bus()
    payload = {"meta": {
        "days": DAYS, "train_seeds": list(TRAIN_SEEDS), "oat_seeds": list(OAT_SEEDS),
        "lhs_seeds": list(LHS_SEEDS), "train_grid": list(TRAIN_GRID),
        "oat_grid": list(OAT_GRID), "lhs_grid": list(LHS_GRID), "flat_tol": FLAT_TOL,
        "lhs_space": LHS_SPACE, "lhs_seed": LHS_SEED, "nominal": NOMINAL,
        "pi_peak_paper": PI_PEAK_PAPER, "pi_peak_profile": PI_PEAK_PROFILE,
        "regime": "normal",
        "notes": "saving = (blind - priced(mu*))/blind, mu* from TRAIN_GRID on train seeds; "
                 "mu_emp = smallest test-grid mu with mean cost <= (1+tol)*best."},
        "oat": [], "lhs": []}
    t0 = time.time()
    jobs = [("oat", "nominal", None, NOMINAL)]
    for name, levels in OAT_LEVELS:
        for lv in levels:
            jobs.append(("oat", name, lv, oat_config(name, lv)))
    lhs = lhs_configs(N_LHS)
    only = os.environ.get("R4_ONLY")
    for i, c in enumerate(lhs):
        jobs.append(("lhs", f"draw{i:02d}", i, c))
    if only:
        jobs = [j for j in jobs if j[0] == only]
    for k, (block, name, lv, cfg) in enumerate(jobs):
        seeds, grid = (OAT_SEEDS, OAT_GRID) if block == "oat" else (LHS_SEEDS, LHS_GRID)
        t = time.time()
        r = evaluate_config(cfg, seeds, grid, net)
        r["name"], r["level"] = name, lv
        payload[block].append(r)
        payload["summary"] = summarise(payload)
        payload["meta"]["elapsed_s"] = time.time() - t0
        save(payload)
        cf = r["closed_form"]["mu_cf_gas_paper"]
        print(f"[{k + 1}/{len(jobs)}] {block} {name}={lv}  mu*={r['mu_star']:g} "
              f"saving={r['saving_pct']:+.2f}% (min seed {r['saving_pct_min_seed']:+.2f}) "
              f"best={r['best_saving_pct']:.2f}%  mu_emp={r['mu_emp']}  cf={cf:.0f}  "
              f"flat2x={r['flat'].get('2x', {}).get('gap_pct_of_best', float('nan')):.3f}% "
              f"flat4x={r['flat'].get('4x', {}).get('gap_pct_of_best', float('nan')):.3f}%  "
              f"({time.time() - t:.0f}s, eps={r['n_episodes']}, total {time.time() - t0:.0f}s)",
              flush=True)
    print(json.dumps(payload["summary"], indent=1))
    plot()
    return 0


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(OUT_JSON, encoding="utf-8") as fh:
        d = json.load(fh)
    oat, lhs = d.get("oat", []), d.get("lhs", [])
    blue, orange, grey = "#2a6fb0", "#d9822b", "#6b6b6b"
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), constrained_layout=True)

    # (a) distribution of savings across LHS draws (+ OAT points)
    ax = axes[0]
    s = np.array([r["saving_pct"] for r in lhs])
    if len(s):
        ax.hist(s, bins=np.linspace(min(-1.0, s.min() - 1), s.max() + 1, 16),
                color=blue, alpha=0.85, edgecolor="white", lw=0.5, label=f"LHS (n={len(s)})")
    nom = [r for r in oat if r["name"] == "nominal"]
    if nom:
        ax.axvline(nom[0]["saving_pct"], color=orange, lw=1.2, label="nominal")
    ax.axvline(0.0, color=grey, lw=0.8, ls=":")
    ax.set_xlabel("saving of priced vs blind OPF (%)")
    ax.set_ylabel("draws")
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    ax.set_title("(a) claim A across draws", fontsize=8)

    # (b) OAT tornado: saving at each perturbed level
    ax = axes[1]
    rows = [r for r in oat if r["name"] != "nominal"]
    names = []
    for r in rows:
        if r["name"] not in names:
            names.append(r["name"])
    for i, n in enumerate(names):
        rs = [r for r in rows if r["name"] == n]
        xs = [r["saving_pct"] for r in rs]
        ax.plot([min(xs), max(xs)], [i, i], color=grey, lw=0.8)
        for r in rs:
            ax.plot(r["saving_pct"], i, "o", ms=3, color=blue)
            ax.annotate(f"{r['level']:g}", (r["saving_pct"], i), fontsize=5,
                        xytext=(0, 3), textcoords="offset points", ha="center")
    if nom:
        ax.axvline(nom[0]["saving_pct"], color=orange, lw=1.0)
    ax.axvline(0.0, color=grey, lw=0.8, ls=":")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("saving (%), 5 test seeds")
    ax.set_title("(b) one-at-a-time", fontsize=8)

    # (c) empirical vs closed-form threshold
    ax = axes[2]
    for rows_, mk, lab in ((lhs, "o", "LHS"), (oat, "s", "OAT")):
        pts = [(r["closed_form"]["mu_cf_gas_paper"], r["mu_emp"],
                bool(r["flat"] and r["flat"]["4x"]["within_tol"]))
               for r in rows_ if r["mu_emp"] and r["closed_form"]["mu_cf_gas_paper"] > 0
               and r["best_saving_pct"] > 0.5]
        if not pts:
            continue
        x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
        ok = np.array([p[2] for p in pts])
        ax.scatter(x[ok], y[ok], s=10, marker=mk, color=blue, alpha=0.8,
                   label=f"{lab}, 4x flat")
        if (~ok).any():
            ax.scatter(x[~ok], y[~ok], s=12, marker=mk, facecolors="none",
                       edgecolors=orange, label=f"{lab}, 4x not flat")
    lim = [1e2, 3e5]
    ax.plot(lim, lim, color=grey, lw=0.8)
    ax.plot(lim, [2 * v for v in lim], color=grey, lw=0.6, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"closed-form $(\pi_{peak}-c_{gas})/\alpha_{gas}$")
    ax.set_ylabel(r"empirical $\mu_{emp}$ (factor-2 grid)")
    ax.legend(fontsize=5.5, frameon=False, loc="upper left")
    ax.set_title("(c) threshold location", fontsize=8)
    for a in axes:
        a.tick_params(labelsize=6)
    fig.savefig(OUT_FIG)
    print(f"wrote {OUT_FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
