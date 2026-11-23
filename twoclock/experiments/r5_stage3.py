"""R5 -- Stage 3 identification study and outcome-independent real-data audit.

Two weaknesses of the original Stage 3 evaluation are addressed:

1. **Monte Carlo identification study (synthetic).**  Hidden prices mu drawn
   log-uniformly on [100, 1e5], noise in {0, 1, 2, 5} MW, window lengths
   {24, 72, 168} h, several environment seeds.  We report coverage of the
   profile interval, identification rate (interval bounded above), relative
   error when identified, interval width, and test the prediction that mu is
   identified iff it lies below the closed-form threshold
   mu* = (pi_max - c_gas) r_gas (1 - sigma) / (c_age c_gas) ~= 4,964.

2. **Outcome-independent real-data protocol (ENTSO-E Spain).**  Instead of the
   window chosen to maximise hours with coal>0.70 & gas>0.45 (selection on the
   outcome), the estimator is run on *every* non-overlapping 72-h window of
   2015-2018, plus sensitivity to sigma in {0.80, 0.85, 0.90} and to the
   capacity normalisation, plus placebo / falsification tests.

Exact profile computation
-------------------------
``estimate_mu`` (stage3_market.py) evaluates SSE(mu) with ~100 scipy ``linprog``
calls per row; that is ~8 ms/LP and too slow for thousands of windows.  Because
mu enters the LP only through the cost vector, every row's optimal dispatch is
a piecewise-constant function of mu.  We compute it *exactly* (up to a 0.5%
breakpoint resolution) by a coarse log-grid plus recursive bisection, solving
the identical LP with a persistent HiGHS model (same solver, only cost/bounds
changed between solves; verified to reproduce ``dc_opf`` to 1e-10 MW).  The
window SSE(mu) is then a step function on the union of row breakpoints, and the
profile interval / point estimate follow the same rule as ``estimate_mu``
(contiguous set with SSE <= 1.05 * SSE_min containing the minimiser, search
range [1, 3e5]).  A validation block compares against ``estimate_mu`` directly.

Run (from the folder that contains ``twoclock``):
    python -m twoclock.experiments.r5_stage3
"""
from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import scipy.optimize._highspy._core as _h
from scipy.sparse import csc_matrix

from ..agents.baselines import _battery_bounds
from ..physics import AgingParams, build_5bus, dc_opf, wear_free_mw, wear_price_from_mu
from ..physics.dc_flow import (BATTERY_BUS, BATTERY_POWER_MW, GEN_BUS, GEN_COST,
                               GEN_DISPATCHABLE, GEN_RATING, IMPORT_BUS,
                               IMPORT_CAP_MW, LOAD_SHARE)
from ..physics.opf import VOLL, _ramp_bounds
from ..paths import RAW_CSV, RESULTS_DIR
from .stage3_market import estimate_mu, synthetic_market

LO, HI = 1.0, 3.0e5          # search range of estimate_mu
TOL = 1.05                   # profile tolerance of estimate_mu
RES = 0.005                  # breakpoint resolution (relative)
INF = _h.kHighsInf
CACHE = os.path.join(RESULTS_DIR, "r5_cache")


# ---------------------------------------------------------------------------
# Fast exact re-implementation of dc_opf (split / piecewise wear cost)
# ---------------------------------------------------------------------------

class FastOPF:
    """Same LP as ``dc_opf(..., wear_price_mwh, wear_free_mw)``, persistent HiGHS."""

    def __init__(self, net):
        self.net = net
        nb, ng, nl = net.n_bus, 4, net.n_line
        self.idx_p = np.arange(ng)
        self.ib, self.ii = ng, ng + 1
        self.tb = [b for b in range(nb) if b != net.slack]
        self.idx_theta = {b: ng + 2 + k for k, b in enumerate(self.tb)}
        off = ng + 2 + len(self.tb)
        self.idx_f = off + np.arange(nl); off += nl
        self.idx_shed = off + np.arange(nb); off += nb
        self.idx_e = off + np.arange(ng)
        self.nv = nv = off + ng
        A = np.zeros((nb + nl + 1, nv))
        for g, bus in enumerate(GEN_BUS):
            A[bus, self.idx_p[g]] += 1.0
            A[bus, self.idx_e[g]] += 1.0
        A[BATTERY_BUS, self.ib] += 1.0
        A[IMPORT_BUS, self.ii] += 1.0
        for b in range(nb):
            A[b, self.idx_f] -= net.incidence[b]
            A[b, self.idx_shed[b]] += 1.0
        for k, ln in enumerate(net.lines):
            r = nb + k
            A[r, self.idx_f[k]] = 1.0
            if ln.i != net.slack:
                A[r, self.idx_theta[ln.i]] -= ln.b
            if ln.j != net.slack:
                A[r, self.idx_theta[ln.j]] += ln.b
        r = nb + nl
        A[r, self.idx_p[GEN_DISPATCHABLE]] = 1.0
        A[r, self.idx_e[GEN_DISPATCHABLE]] = 1.0
        A[r, self.ib] = 1.0
        self.nb, self.nl, self.nr = nb, nl, A.shape[0]
        M = csc_matrix(A)
        lp = _h.HighsLp()
        lp.num_col_, lp.num_row_ = nv, self.nr
        lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = nv, self.nr
        lp.a_matrix_.format_ = _h.MatrixFormat.kColwise
        lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = M.indptr, M.indices, M.data
        lp.col_cost_, lp.col_lower_, lp.col_upper_ = np.zeros(nv), np.zeros(nv), np.ones(nv)
        lp.row_lower_, lp.row_upper_ = np.zeros(self.nr), np.zeros(self.nr)
        self.h = _h._Highs()
        self.h.setOptionValue("output_flag", False)
        self.h.passModel(lp)
        self.allidx = np.arange(nv, dtype=np.int32)
        self.row = None

    def set_row(self, r, free):
        self.row = r
        self.free = free
        lo_b, hi_b = _battery_bounds(r.get("soc", 0.5), 0.25)
        pmax_eff = np.asarray(r["pmax_eff"], float)
        prev = np.asarray(r["prev_p"], float)
        loads = LOAD_SHARE * r["load"]
        cap_res = float(np.sum(pmax_eff[GEN_DISPATCHABLE]) + BATTERY_POWER_MW - r.get("reserve_req", 0.0))
        pmin, pmax = _ramp_bounds(prev, pmax_eff, r["renewable"])
        L, U = np.zeros(self.nv), np.zeros(self.nv)
        for g in range(4):
            f = float(np.clip(free[g], 0.0, pmax[g]))
            s_lo = float(min(pmin[g], f))
            L[g], U[g] = s_lo, max(s_lo, f)
            e_lo = max(0.0, float(pmin[g]) - f)
            e_hi = max(e_lo, float(pmax[g]) - max(f, float(pmin[g])))
            if g == 3:
                e_lo = e_hi = 0.0
            L[self.idx_e[g]], U[self.idx_e[g]] = e_lo, e_hi
        lo, hi = max(lo_b, -BATTERY_POWER_MW), min(hi_b, BATTERY_POWER_MW)
        if hi < lo:
            lo = hi = 0.0
        L[self.ib], U[self.ib] = lo, hi
        L[self.ii], U[self.ii] = 0.0, IMPORT_CAP_MW
        for b in self.tb:
            L[self.idx_theta[b]], U[self.idx_theta[b]] = -INF, INF
        L[self.idx_f], U[self.idx_f] = -self.net.limits, self.net.limits
        L[self.idx_shed], U[self.idx_shed] = 0.0, 0.30 * loads
        self.h.changeColsBounds(self.nv, self.allidx, L, U)
        for i in range(self.nb):
            self.h.changeRowBounds(i, float(loads[i]), float(loads[i]))
        for i in range(self.nb, self.nb + self.nl):
            self.h.changeRowBounds(i, 0.0, 0.0)
        self.h.changeRowBounds(self.nr - 1, -INF, cap_res)

    def solve(self, wp):
        c = np.zeros(self.nv)
        c[self.idx_p] = GEN_COST
        c[self.idx_e] = GEN_COST + wp
        c[self.ii] = self.row["spot"]
        c[self.idx_shed] = VOLL
        self.h.changeColsCost(self.nv, self.allidx, c)
        self.h.run()
        if self.h.getModelStatus() != _h.HighsModelStatus.kOptimal:
            return None
        x = np.asarray(self.h.getSolution().col_value)
        return x[self.idx_p] + x[self.idx_e]


def row_profile(F, net, row, coef, free, n_coarse=25):
    """Exact piecewise-constant dispatch p(mu) on [LO, HI]: (breakpoints, P)."""
    F.set_row(row, free)
    sol = lambda m: F.solve(m * coef)
    grid = np.geomspace(LO, HI, n_coarse)
    P = [sol(m) for m in grid]
    if any(p is None for p in P):          # infeasible LP -> dc_opf merit fallback (mu-independent)
        lo, hi = _battery_bounds(row.get("soc", 0.5), 0.25)
        res = dc_opf(net, row["load"], np.asarray(row["prev_p"]), row["renewable"],
                     np.asarray(row["pmax_eff"]), spot_price=row["spot"],
                     wear_price_mwh=coef * 1000.0, wear_free_mw=free,
                     reserve_req_mw=row.get("reserve_req", 0.0),
                     battery_min_mw=lo, battery_max_mw=hi)
        return np.zeros(0), res.p_gen[None, :]
    same = lambda a, b: np.max(np.abs(a - b)) < 1e-6

    def refine(a, pa, b, pb, out):
        if same(pa, pb):
            return
        if b / a < 1.0 + RES:
            out.append((np.sqrt(a * b), pb))
            return
        m = np.sqrt(a * b)
        pm = sol(m)
        refine(a, pa, m, pm, out)
        refine(m, pm, b, pb, out)

    bps, vals = [], [P[0]]
    for k in range(n_coarse - 1):
        out = []
        refine(grid[k], P[k], grid[k + 1], P[k + 1], out)
        for bp, pv in out:
            if not same(pv, vals[-1]):
                bps.append(bp)
                vals.append(pv)
    return np.asarray(bps), np.asarray(vals)


def profiles_for_rows(net, rows, aging):
    F = FastOPF(net)
    coef = wear_price_from_mu(1.0, aging, GEN_RATING, GEN_DISPATCHABLE)
    free = wear_free_mw(aging)
    return [row_profile(F, net, r, coef, free) for r in rows]


def predict_at(profiles, mu):
    return np.array([P[np.searchsorted(b, mu)] for b, P in profiles])


def window_sse(profiles, obs):
    """SSE(mu) as a step function: returns (edges, sse) with len(sse)=len(edges)-1."""
    allb = np.concatenate([b for b, _ in profiles]) if profiles else np.zeros(0)
    U = np.unique(allb[(allb > LO) & (allb < HI)])
    edges = np.concatenate([[LO], U, [HI]])
    mids = np.sqrt(edges[:-1] * edges[1:])
    sse = np.zeros(len(mids))
    for (b, P), o in zip(profiles, obs):
        d = o[None, :] - P[np.searchsorted(b, mids)]
        sse += np.sum(d * d, axis=1)
    return edges, sse


def estimate_exact(profiles, obs, tol=TOL, lr_thr=None):
    """Same decision rule as estimate_mu, evaluated on the exact step profile."""
    edges, sse = window_sse(profiles, obs)
    m = float(sse.min())
    j = int(np.argmin(sse))
    thr = m * tol + 1e-6 if lr_thr is None else m + lr_thr + 1e-6
    jl, jr = j, j
    while jl > 0 and sse[jl - 1] <= thr:
        jl -= 1
    while jr < len(sse) - 1 and sse[jr + 1] <= thr:
        jr += 1
    return {"mu_hat": float(np.sqrt(edges[j] * edges[j + 1])),
            "interval": [float(edges[jl]), float(edges[jr + 1])],
            "sse_min": m, "sse_floor": float(sse[0]),
            "n_breaks": int(len(edges) - 2),
            "max_break": float(edges[-2]) if len(edges) > 2 else LO}


# ---------------------------------------------------------------------------
# Part 1: synthetic Monte Carlo
# ---------------------------------------------------------------------------

def gas_threshold(sigma, pi_max, aging=None):
    a = aging or AgingParams(stress_frac=sigma)
    return (pi_max - GEN_COST[0]) * GEN_RATING[0] * (1 - sigma) / (a.c_age * a.c_age_gen[0])


def synthetic_mc(net, n_mu=40, seeds=(0, 1, 2), noises=(0.0, 1.0, 2.0, 5.0),
                 windows=(24, 72, 168)):
    path = os.path.join(CACHE, "synthetic.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    aging = AgingParams()
    recs = []
    t0 = time.time()
    for seed in seeds:
        rng_mu = np.random.default_rng(1000 + seed)
        mus = np.exp(rng_mu.uniform(np.log(100.0), np.log(1e5), n_mu))
        for k, mu in enumerate(mus):
            rows15 = synthetic_market(net, aging, float(mu), days=max(windows) // 24, seed=seed)
            rows = rows15[::4]                       # hourly observations
            prof = profiles_for_rows(net, rows, aging)
            truth = np.array([r["p_gen"] for r in rows])
            spot = np.array([r["spot"] for r in rows])
            for nz in noises:
                rng = np.random.default_rng(10_000 * seed + 100 * k + int(10 * nz))
                obs = truth.copy()
                if nz > 0:
                    obs[:, :3] = np.maximum(obs[:, :3] + rng.normal(0, nz, obs[:, :3].shape), 0.0)
                for L in windows:
                    est = estimate_exact(prof[:L], obs[:L])
                    lr = (estimate_exact(prof[:L], obs[:L], lr_thr=3.84 * nz ** 2)
                          if nz > 0 else est)
                    recs.append({"seed": seed, "mu_true": float(mu), "noise": nz, "L": L,
                                 "max_spot": float(spot[:L].max()),
                                 "thr_window": float(gas_threshold(0.85, spot[:L].max())),
                                 **est, "interval_lr": lr["interval"], "mu_hat_lr": lr["mu_hat"]})
            print(f"  seed {seed} mu {k+1}/{n_mu} ({mu:8.0f})  {time.time()-t0:6.0f}s", flush=True)
    pickle.dump(recs, open(path, "wb"))
    return recs


def summarise_synthetic(recs, thr_nom):
    df = pd.DataFrame(recs)
    s = 1.0 + RES
    df["covered"] = (df.interval.str[0] / s <= df.mu_true) & (df.mu_true <= df.interval.str[1] * s)
    df["covered_lr"] = (df.interval_lr.str[0] / s <= df.mu_true) & (df.mu_true <= df.interval_lr.str[1] * s)
    df["identified"] = df.interval.str[1] < HI / 1.01
    df["two_sided"] = df.identified & (df.interval.str[0] > LO * 1.01)
    df["relerr"] = (df.mu_hat - df.mu_true).abs() / df.mu_true
    df["log10_width"] = np.log10(df.interval.str[1] / df.interval.str[0])
    df["below_nom"] = df.mu_true < thr_nom
    df["below_win"] = df.mu_true < df.thr_window
    out = {"cells": [], "by_ratio_bins": []}
    for (nz, L), g in df.groupby(["noise", "L"]):
        ident = g[g.identified]
        out["cells"].append({
            "noise_mw": nz, "L_h": int(L), "n": int(len(g)),
            "coverage_5pct_rule": float(g.covered.mean()),
            "coverage_LR_chi2": float(g.covered_lr.mean()),
            "frac_identified": float(g.identified.mean()),
            "frac_identified_below_nominal_thr": float(g[g.below_nom].identified.mean()),
            "frac_identified_above_nominal_thr": float(g[~g.below_nom].identified.mean()),
            "frac_identified_below_window_thr": float(g[g.below_win].identified.mean()),
            "frac_identified_above_window_thr": float(g[~g.below_win].identified.mean()),
            "rule_accuracy_nominal": float((g.identified == g.below_nom).mean()),
            "rule_accuracy_window": float((g.identified == g.below_win).mean()),
            "coverage_when_identified": float(ident.covered.mean()) if len(ident) else None,
            "median_relerr_identified": float(ident.relerr.median()) if len(ident) else None,
            "p90_relerr_identified": float(ident.relerr.quantile(0.9)) if len(ident) else None,
            "median_log10_width_identified": float(ident.log10_width.median()) if len(ident) else None,
            "frac_identified_cell_touches_floor": float((ident.interval.str[0] <= LO * 1.01).mean()) if len(ident) else None,
            "median_relerr_two_sided": float(g[g.two_sided].relerr.median()) if g.two_sided.any() else None,
            "p90_relerr_two_sided": float(g[g.two_sided].relerr.quantile(0.9)) if g.two_sided.any() else None,
            "median_log10_width_two_sided": float(g[g.two_sided].log10_width.median()) if g.two_sided.any() else None,
            "median_n_breaks": float(g.n_breaks.median()),
            "median_window_thr": float(g.thr_window.median()),
        })
    bins = np.array([0.02, 0.1, 0.3, 0.6, 0.9, 1.1, 1.5, 3, 10, 30])
    df["rbin"] = pd.cut(df.mu_true / thr_nom, bins)
    for (nz, L, rb), g in df.groupby(["noise", "L", "rbin"], observed=True):
        out["by_ratio_bins"].append({"noise_mw": nz, "L_h": int(L),
                                     "ratio_lo": float(rb.left), "ratio_hi": float(rb.right),
                                     "n": int(len(g)), "frac_identified": float(g.identified.mean()),
                                     "coverage": float(g.covered.mean())})
    return out, df


# ---------------------------------------------------------------------------
# Part 2: real ENTSO-E data, all windows
# ---------------------------------------------------------------------------

def load_real(cap_mode="rolling30_q99"):
    df = pd.read_csv(RAW_CSV)
    req = ["time", "generation fossil gas", "generation fossil hard coal",
           "generation nuclear", "total load actual", "price actual"]
    df = df.dropna(subset=req).copy().reset_index(drop=True)
    coal = (df["generation fossil hard coal"].to_numpy(float)
            + df["generation fossil brown coal/lignite"].fillna(0.0).to_numpy(float))
    gas = df["generation fossil gas"].to_numpy(float)
    nuc = df["generation nuclear"].to_numpy(float)
    ren = (df["generation wind onshore"].fillna(0.0).to_numpy(float)
           + df["generation solar"].fillna(0.0).to_numpy(float))
    spot = df["price actual"].to_numpy(float)

    def cap(x):
        if cap_mode == "rolling30_q99":      # original: rolling 30-day 99th percentile
            return pd.Series(x).rolling(720, min_periods=168).quantile(0.99).bfill().to_numpy()
        if cap_mode == "static_q99":         # one fleet capacity for the whole sample
            return np.full(len(x), np.quantile(x, 0.99))
        raise ValueError(cap_mode)

    l = {k: np.clip(v / np.maximum(cap(v), 1.0), 0, 1) for k, v in
         (("gas", gas), ("coal", coal), ("nuc", nuc), ("ren", ren))}
    p = np.stack([350 * l["gas"], 300 * l["coal"], np.clip(200 * l["nuc"], 170, 200),
                  150 * l["ren"]], axis=1)
    t = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/Madrid")
    return {"p": p, "load": p.sum(1), "spot": np.clip(spot * 1.15, 20, 300),
            "time": df["time"].astype(str).to_numpy(), "year": t.dt.year.to_numpy(),
            "month": t.dt.month.to_numpy(), "l_gas": l["gas"], "l_coal": l["coal"]}


def real_rows(R, spot=None):
    p, sp = R["p"], (R["spot"] if spot is None else spot)
    rows = []
    for i in range(len(p)):
        rows.append({"load": float(R["load"][i]), "renewable": float(p[i, 3]), "spot": float(sp[i]),
                     "prev_p": p[max(i - 1, 0)].tolist(),
                     "pmax_eff": [350.0, 300.0, 200.0, float(p[i, 3])], "p_gen": p[i].tolist()})
    return rows


def window_starts(n, L=72):
    return list(range(1, n - L + 1, L))


def _profile_chunk(args):
    rows, sigma = args
    return profiles_for_rows(build_5bus(), rows, AgingParams(stress_frac=sigma))


def real_profiles(net, R, sigma, tag, spot=None):
    path = os.path.join(CACHE, f"real_{tag}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    t0 = time.time()
    rows = real_rows(R, spot)
    from concurrent.futures import ProcessPoolExecutor
    chunks = [rows[i:i + 2500] for i in range(0, len(rows), 2500)]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("R5_WORKERS", "5"))) as ex:
        parts = ex.map(_profile_chunk, [(c, sigma) for c in chunks])
        prof = [x for part in parts for x in part]
    pickle.dump(prof, open(path, "wb"))
    print(f"  profiles {tag}: {len(rows)} rows in {time.time()-t0:.0f}s", flush=True)
    return prof


def season(m):
    return {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
            6: "JJA", 7: "JJA", 8: "JJA"}.get(int(m), "SON")


def run_windows(R, prof, obs=None, L=72):
    obs = R["p"] if obs is None else obs
    out = []
    for s in window_starts(len(R["p"]), L):
        est = estimate_exact(prof[s:s + L], obs[s:s + L])
        est.update({"start": str(R["time"][s]), "year": int(R["year"][s]),
                    "season": season(R["month"][s])})
        out.append(est)
    return out


def summarise_windows(W, label):
    mu = np.array([w["mu_hat"] for w in W])
    lo = np.array([w["interval"][0] for w in W])
    hi = np.array([w["interval"][1] for w in W])
    imp = np.array([1 - w["sse_min"] / max(w["sse_floor"], 1e-12) for w in W])
    ex0 = lo > LO * 1.01
    q = lambda x: [float(np.quantile(x, v)) for v in (0.1, 0.25, 0.5, 0.75, 0.9)]
    d = {"label": label, "n_windows": int(len(W)),
         "frac_interval_excludes_floor": float(ex0.mean()),
         "frac_mu_hat_above_100": float((mu > 100).mean()),
         "frac_bounded_above": float((hi < HI / 1.01).mean()),
         "frac_no_mu_dependence": float(np.mean([w["n_breaks"] == 0 for w in W])),
         "mu_hat_q10_25_50_75_90": q(mu),
         "lower_bound_q10_25_50_75_90": q(lo),
         "lower_bound_median_given_excl": float(np.median(lo[ex0])) if ex0.any() else None,
         "sse_reduction_vs_mu0_q10_50_90": [float(np.quantile(imp, v)) for v in (0.1, 0.5, 0.9)],
         "by_year": {}, "by_season": {}}
    for key, field in (("by_year", "year"), ("by_season", "season")):
        for v in sorted({w[field] for w in W}):
            m = np.array([w[field] == v for w in W])
            d[key][str(v)] = {"n": int(m.sum()), "frac_excl_floor": float(ex0[m].mean()),
                              "median_mu_hat": float(np.median(mu[m])),
                              "median_lower": float(np.median(lo[m])),
                              "median_sse_reduction": float(np.median(imp[m]))}
    return d


def placebo_obs(R, prof, W_real, mu_gen, rng, L=72, gauss=None):
    """Synthetic observations on real covariates: OPF(mu_gen) + noise.

    Noise is either Gaussian (``gauss`` MW on gas/coal/nuclear) or the real
    window's own residuals at its fitted mu_hat, permuted across hours (keeps
    the size and marginal distribution of the real misfit, breaks its timing).
    """
    obs = predict_at(prof, mu_gen)
    for s, w in zip(window_starts(len(R["p"]), L), W_real):
        sl = slice(s, s + L)
        if gauss is not None:
            obs[sl, :3] += rng.normal(0, gauss, (L, 3))
        else:
            resid = R["p"][sl] - predict_at(prof[sl], w["mu_hat"])
            obs[sl, :3] += resid[rng.permutation(L), :3]
    return obs


# ---------------------------------------------------------------------------
# Validation against the original estimator
# ---------------------------------------------------------------------------

def validate(net):
    path = os.path.join(CACHE, "validate.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    aging = AgingParams()
    out = []
    cases = []
    for mu in (1000.0, 5000.0, 20000.0):
        rows = synthetic_market(net, aging, mu, days=3, seed=3)[::4]
        cases.append((f"synthetic mu={mu:g}, 72h", rows))
    R = load_real()
    s = window_starts(len(R["p"]))[100]
    cases.append((f"ENTSO-E window {R['time'][s][:10]}", real_rows(R)[s:s + 72]))
    for name, rows in cases:
        t0 = time.time()
        a = estimate_mu(net, aging, rows)
        t1 = time.time()
        prof = profiles_for_rows(net, rows, aging)
        b = estimate_exact(prof, np.array([r["p_gen"] for r in rows]))
        t2 = time.time()
        # SSE agreement at the reference grid
        g = np.asarray(a["grid"])
        sse_fast = [float(np.sum((np.array([r["p_gen"] for r in rows]) - predict_at(prof, m)) ** 2))
                    for m in g]
        out.append({"case": name, "orig_mu_hat": a["mu_hat"], "orig_interval": a["interval"],
                    "exact_mu_hat": b["mu_hat"], "exact_interval": b["interval"],
                    "max_abs_sse_diff_on_grid": float(np.max(np.abs(np.array(sse_fast) - np.array(a["sse"])))),
                    "orig_seconds": t1 - t0, "exact_seconds": t2 - t1})
        print("  validate:", out[-1], flush=True)
    pickle.dump(out, open(path, "wb"))
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(df_syn, thr_nom, real_main, placebos, sens, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 6.8,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.35))

    a = ax[0]
    bins = np.geomspace(100, 1e5, 13)
    cen = np.sqrt(bins[:-1] * bins[1:])
    for c, nz in zip(C, (0.0, 1.0, 2.0, 5.0)):
        g = df_syn[(df_syn.noise == nz) & (df_syn.L == 72)]
        idx = np.digitize(g.mu_true, bins) - 1
        fr = [g.identified[idx == k].mean() if (idx == k).any() else np.nan for k in range(len(cen))]
        a.semilogx(cen, fr, "-o", color=c, lw=1.5, ms=3, label=f"{nz:g} MW")
    a.axvline(thr_nom, color="#52514e", ls="--", lw=1)
    a.text(thr_nom * 1.1, 0.5, r"$\mu^*\approx$" + f"{thr_nom:,.0f}", fontsize=6.8, color="#52514e")
    a.set_xlabel(r"hidden $\mu$ (\$/wear-unit)")
    a.set_ylabel("fraction identified\n(interval bounded above)")
    a.set_title("(a) synthetic, 72-h windows")
    a.set_ylim(-0.03, 1.03)
    a.legend(title="noise", loc="lower left", frameon=False, title_fontsize=6.8)

    b = ax[1]
    for c, (lab, W) in zip(C, [("ENTSO-E real", real_main)] + placebos):
        lo = np.sort([max(w["interval"][0], LO) for w in W])
        b.step(lo, np.arange(1, len(lo) + 1) / len(lo), where="post", color=c, lw=1.5, label=lab)
    b.set_xscale("log")
    b.set_xlabel(r"profile-interval lower bound on $\mu$")
    b.set_ylabel("ECDF over 72-h windows")
    b.set_title("(b) real vs placebo, all windows")
    b.legend(loc="lower right", frameon=False)

    c_ = ax[2]
    labels = [s["label"].replace("sigma=", r"$\sigma$=").replace(", ", "\n") for s in sens]
    vals = [s["frac_interval_excludes_floor"] for s in sens]
    y = np.arange(len(sens))
    c_.barh(y, vals, color=C[0], height=0.6)
    for yi, v in zip(y, vals):
        c_.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=6.8)
    c_.set_yticks(y)
    c_.set_yticklabels(labels, fontsize=6.3)
    c_.invert_yaxis()
    c_.set_xlim(0, 1.15)
    c_.set_xlabel("fraction of windows with lower bound > floor")
    c_.set_title("(c) sensitivity (real data)")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------

def main() -> int:
    os.makedirs(CACHE, exist_ok=True)
    net = build_5bus()
    thr_nom = float(gas_threshold(0.85, 117.0))
    print(f"nominal threshold mu* = {thr_nom:.0f}")

    print("== validation vs estimate_mu ==")
    val = validate(net)

    print("== synthetic Monte Carlo ==")
    recs = synthetic_mc(net)
    syn, df_syn = summarise_synthetic(recs, thr_nom)
    for c in syn["cells"]:
        print(f"  noise {c['noise_mw']:3.0f} L {c['L_h']:3d}: cov {c['coverage_5pct_rule']:.2f} "
              f"covLR {c['coverage_LR_chi2']:.2f} ident {c['frac_identified']:.2f} "
              f"(below thr {c['frac_identified_below_nominal_thr']:.2f}, above {c['frac_identified_above_nominal_thr']:.2f}) "
              f"rule acc nom {c['rule_accuracy_nominal']:.2f} win {c['rule_accuracy_window']:.2f} "
              f"relerr {c['median_relerr_identified']} floor-cell {c['frac_identified_cell_touches_floor']} "
              f"relerr2s {c['median_relerr_two_sided']}")

    print("== real data: all non-overlapping 72-h windows ==")
    configs = [(0.85, "rolling30_q99"), (0.80, "rolling30_q99"), (0.90, "rolling30_q99"),
               (0.85, "static_q99"), (0.80, "static_q99"), (0.90, "static_q99")]
    Rs = {}
    sens, results_by_cfg = [], {}
    for sigma, cap in configs:
        if cap not in Rs:
            Rs[cap] = load_real(cap)
        R = Rs[cap]
        prof = real_profiles(net, R, sigma, f"s{sigma:.2f}_{cap}")
        W = run_windows(R, prof)
        lab = f"sigma={sigma:.2f}, cap={cap}"
        sm = summarise_windows(W, lab)
        sm["threshold_mu_star_at_max_spot"] = float(gas_threshold(sigma, float(R["spot"].max())))
        sens.append(sm)
        results_by_cfg[(sigma, cap)] = (R, prof, W)
        print(f"  {lab}: excl-floor {sm['frac_interval_excludes_floor']:.2f}, "
              f"median mu_hat {sm['mu_hat_q10_25_50_75_90'][2]:.0f}, "
              f"median SSE reduction {sm['sse_reduction_vs_mu0_q10_50_90'][1]:.3f}", flush=True)

    R, prof, W = results_by_cfg[(0.85, "rolling30_q99")]
    base = sens[0]

    # descriptive confound diagnostic: where does coal sit when gas is running?
    gas_on = R["l_gas"] > 0.45
    lc = R["l_coal"][gas_on]
    diag = {"n_hours_gas_gt_0.45": int(gas_on.sum()),
            "coal_loading_quantiles_q10_25_50_75_90": [float(np.quantile(lc, v)) for v in (0.1, .25, .5, .75, .9)],
            "frac_coal_in_0.80_0.90_band": float(np.mean((lc >= 0.80) & (lc <= 0.90))),
            "frac_coal_below_0.80": float(np.mean(lc < 0.80)),
            "frac_coal_above_0.90": float(np.mean(lc > 0.90))}

    print("== placebo / falsification ==")
    rng = np.random.default_rng(2024)
    plc = {}
    obs0 = placebo_obs(R, prof, W, LO, rng)
    plc["null_mu0_permuted_real_residuals"] = summarise_windows(run_windows(R, prof, obs0), "null mu=0 + perm. real residuals")
    obsg = placebo_obs(R, prof, W, LO, rng, gauss=5.0)
    plc["null_mu0_gauss5"] = summarise_windows(run_windows(R, prof, obsg), "null mu=0 + 5 MW Gaussian")
    W_null = run_windows(R, prof, obs0)
    obs2k = placebo_obs(R, prof, W, 2000.0, rng)
    W_pos = run_windows(R, prof, obs2k)
    pos = summarise_windows(W_pos, "positive control mu=2000 + perm. residuals")
    pos["coverage_of_2000"] = float(np.mean([w["interval"][0] / 1.005 <= 2000 <= w["interval"][1] * 1.005 for w in W_pos]))
    plc["positive_mu2000_permuted_real_residuals"] = pos
    # spot-shuffle falsification: permute prices across hours within each window
    sp = R["spot"].copy()
    for s in window_starts(len(sp)):
        sp[s:s + 72] = sp[s:s + 72][rng.permutation(72)]
    prof_sh = real_profiles(net, R, 0.85, "s0.85_rolling30_q99_spotshuffle", spot=sp)
    W_sh = run_windows(R, prof_sh)
    plc["spot_shuffled_within_window"] = summarise_windows(W_sh, "spot shuffled within window")
    # paired comparison real vs spot-shuffled
    d_mu = np.log10([a["mu_hat"] for a in W]) - np.log10([b["mu_hat"] for b in W_sh])
    plc["spot_shuffled_within_window"]["paired_median_log10_mu_hat_diff_real_minus_shuffled"] = float(np.median(d_mu))
    plc["spot_shuffled_within_window"]["paired_frac_real_sse_min_lower"] = float(np.mean(
        [a["sse_min"] < b["sse_min"] for a, b in zip(W, W_sh)]))
    for k, v in plc.items():
        print(f"  {k}: excl-floor {v['frac_interval_excludes_floor']:.2f}, median mu_hat "
              f"{v['mu_hat_q10_25_50_75_90'][2]:.0f}", flush=True)

    # the original outcome-selected window, for reference
    payload = {
        "description": "R5 Stage 3: Monte Carlo identification + outcome-independent ENTSO-E audit",
        "estimator": {"rule": "profile interval SSE<=1.05*SSE_min (estimate_mu rule), exact step profile",
                      "search_range": [LO, HI], "breakpoint_resolution_rel": RES,
                      "synthetic_observation_cadence": "hourly (every 4th 15-min step)"},
        "nominal_threshold_mu_star": thr_nom,
        "validation_vs_estimate_mu": val,
        "synthetic": syn,
        "synthetic_raw": [{k: (v if not isinstance(v, np.generic) else v.item()) for k, v in r.items()}
                          for r in recs],
        "real": {"window_h": 72, "n_hours": int(len(R["p"])), "base": base,
                 "sensitivity": sens, "coal_when_gas_running": diag, "placebo": plc,
                 "base_windows": W},
    }
    out = os.path.join(RESULTS_DIR, "r5_stage3.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=float)
    make_figure(df_syn, thr_nom, W,
                [("null: OPF($\\mu$=0)+resid.", W_null), ("positive: OPF($\\mu$=2000)+resid.", W_pos),
                 ("spot shuffled", W_sh)], sens, os.path.join(RESULTS_DIR, "fig_r5_stage3.pdf"))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
