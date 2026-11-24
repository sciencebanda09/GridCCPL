"""R28 -- hazard calibration on all US EPA CEMS units, 2015-2024 (large sample version of R25).

Per unit-year (rated output = P99.5 of loaded hours; >= 2000 operating hours; rated >= 100 MW) we build
the model wear path W_t (default aging rate, sigma = 0.85, dt = 1 h, clipped to [0,1]) for several tau
and tabulate, over all running hours, the next-hour trip-like event (load >= 0.6 rated in hour t-1,
zero operating time in hour t) against W_{t-1}^2.  E2 is the stricter event that also requires >= 4
consecutive off hours.  Hazard model: lambda = h0 + beta * W^2 (Poisson), kappa_hat = 24 * beta.
RISK SET: hours with load >= 0.6 rated in hour t-1 (so the event definition cannot mechanically
correlate with W); baseline hazard stratified by the previous-hour load fraction (4 strata).
Binned sufficient statistics (400 bins of W^2) keep memory small.  Placebo: W path circularly shifted
by a random offset within the unit-year.

Run: python -m twoclock.experiments.r28_hazard_all_us [dir with cems_all_<year>.parquet]
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

SIGMA, K_AGE, NB = 0.85, 0.55, 400
TAUS = (1.5, 3.0, 6.0, 12.0, 24.0)


def wear_matrix(lf, tau):
    """lf: T x U load fraction. Vectorised over units, clipped IIR."""
    a = np.exp(-1.0 / tau)
    inc = (K_AGE / (1.0 - SIGMA)) * np.maximum(0.0, lf - SIGMA)
    W = np.empty_like(lf)
    w = np.zeros(lf.shape[1], dtype=lf.dtype)
    for t in range(lf.shape[0]):
        w = np.minimum(1.0, a * w + inc[t])
        W[t] = w
    return W


def load_year(path):
    d = pd.read_parquet(path, columns=["state", "plant_id_eia", "emissions_unit_id_epa",
                                       "operating_datetime_utc", "operating_time_hours", "gross_load_mw"])
    d["gross_load_mw"] = d["gross_load_mw"].fillna(0.0)
    d["operating_time_hours"] = d["operating_time_hours"].fillna(0.0)
    t = d["operating_datetime_utc"].astype("datetime64[ns]")
    h = ((t - t.min()).dt.total_seconds() // 3600).astype(np.int64).to_numpy()
    ucode, _ = pd.factorize(d["emissions_unit_id_epa"])
    key = d["plant_id_eia"].to_numpy(np.int64) * 100000 + ucode.astype(np.int64)
    uid, inv = np.unique(key, return_inverse=True)
    T = int(h.max()) + 1
    P = np.zeros((T, len(uid)), np.float32)
    O = np.zeros((T, len(uid)), np.float32)
    P[h, inv] = d["gross_load_mw"].to_numpy(np.float32)
    O[h, inv] = d["operating_time_hours"].to_numpy(np.float32)
    return P, O


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/kumar/cems_big"
    from twoclock.paths import RESULTS_DIR
    rng = np.random.default_rng(28)
    NS = 4
    stats = {tau: {k: np.zeros((2, NS, NB)) for k in ("n", "x", "d")} for tau in TAUS}   # [real, placebo]
    ev2 = {tau: np.zeros((2, NS, NB)) for tau in TAUS}
    tot_units = tot_hours = tot_e1 = tot_e2 = 0
    for path in sorted(glob.glob(os.path.join(base, "cems_all_*.parquet"))):
        P, O = load_year(path)
        opn = O > 0
        keep = opn.sum(0) >= 2000
        P, O, opn = P[:, keep], O[:, keep], opn[:, keep]
        Ppos = np.where(opn, P, np.nan)
        rated = np.nanpercentile(Ppos, 99.5, axis=0)
        big = rated >= 100
        P, O, opn, rated = P[:, big], O[:, big], opn[:, big], rated[big]
        U = P.shape[1]
        tot_units += U
        y_r1 = y_r2 = y_h = 0
        for c0 in range(0, U, 250):
            sl = slice(c0, min(U, c0 + 250))
            Pc, Oc, opc, rc = P[:, sl], O[:, sl], opn[:, sl], rated[sl]
            lf = np.clip(Pc / rc[None, :], 0, 1.5).astype(np.float32)
            hi = Pc[:-1] >= 0.6 * rc[None, :]
            off_next = ~opc[1:]
            e1 = hi & off_next
            off4 = np.zeros_like(off_next)
            off4[:-3] = off_next[:-3] & off_next[1:-2] & off_next[2:-1] & off_next[3:]
            e2 = hi & off4
            risk = hi
            strat = np.digitize(Pc[:-1] / rc[None, :], [0.75, 0.85, 0.95]).astype(np.int32)
            y_r1 += int(e1.sum()); y_r2 += int(e2.sum()); y_h += int(risk.sum())
            Uc = lf.shape[1]
            sh = rng.integers(500, lf.shape[0] - 500, size=Uc)
            for tau in TAUS:
                W = wear_matrix(lf, tau)
                x_real = W[:-1] ** 2
                idx = ((np.arange(lf.shape[0] - 1)[:, None] + sh[None, :]) % (lf.shape[0] - 1)).astype(np.int32)
                x_pl = np.take_along_axis(x_real, idx, axis=0)
                for r, x in enumerate((x_real, x_pl)):
                    flat = strat * NB + np.minimum((x * NB).astype(np.int32), NB - 1)
                    stats[tau]["n"][r] += np.bincount(flat[risk], minlength=NS * NB).reshape(NS, NB)
                    stats[tau]["x"][r] += np.bincount(flat[risk], weights=x[risk], minlength=NS * NB).reshape(NS, NB)
                    stats[tau]["d"][r] += np.bincount(flat[e1], minlength=NS * NB).reshape(NS, NB)
                    ev2[tau][r] += np.bincount(flat[e2], minlength=NS * NB).reshape(NS, NB)
                del W, x_real, idx, x_pl
            del lf, hi, e1, e2, off4, off_next, risk, strat
        tot_hours += y_h; tot_e1 += y_r1; tot_e2 += y_r2
        print(os.path.basename(path), "units", U, "risk hours", y_h, "E1", y_r1, "E2", y_r2, flush=True)
        del P, O, opn

    def fit(n, xs, d):
        S = n.shape[0]
        mean_x = np.where(n > 0, xs / np.maximum(n, 1), 0.0)
        def nll(t, beta=None):
            b = np.exp(t[S]) if beta is None else beta
            tot = 0.0
            for k in range(S):
                lam = np.exp(t[k]) + b * mean_x[k]
                lam = np.maximum(lam, 1e-300)
                tot += np.sum(n[k] * lam) - np.sum(d[k] * np.log(lam))
            return float(tot)
        h0s = [max(d[k].sum() / max(n[k].sum(), 1), 1e-12) for k in range(S)]
        t0 = [np.log(h) for h in h0s]
        best = None
        for b0 in (1e-6, 1e-5, 1e-4, 1e-3):
            r = minimize(nll, t0 + [np.log(b0)], method="Nelder-Mead",
                         options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 20000, "maxfev": 20000})
            if best is None or r.fun < best.fun:
                best = r
        l0 = sum(float(np.sum(n[k]) * h0s[k] - d[k].sum() * np.log(h0s[k])) for k in range(S))
        beta = float(np.exp(best.x[S]))
        def prof(bb):
            r = minimize(lambda t: nll(list(t) + [0.0], beta=bb), best.x[:S], method="Nelder-Mead",
                         options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-9})
            return 2 * (r.fun - best.fun)
        up = lo = None
        for f in np.geomspace(1.05, 1e3, 60):
            if prof(beta * f) > chi2.ppf(0.95, 1):
                up = beta * f; break
        for f in np.geomspace(1.05, 1e3, 60):
            if prof(beta / f) > chi2.ppf(0.95, 1):
                lo = beta / f; break
        return {"beta_per_h": beta, "kappa_hat": 24 * beta, "kappa_ci95": [None if lo is None else 24 * lo, None if up is None else 24 * up],
                "LR_vs_null": max(0.0, 2 * (l0 - best.fun)), "h0_per_h": [float(np.exp(v)) for v in best.x[:S]]}

    out = {"units": tot_units, "risk_hours": tot_hours, "events_E1": tot_e1, "events_E2": tot_e2, "taus": {}}
    for tau in TAUS:
        s = stats[tau]
        out["taus"][str(tau)] = {
            "E1": fit(s["n"][0], s["x"][0], s["d"][0]), "E1_placebo": fit(s["n"][1], s["x"][1], s["d"][1]),
            "E2": fit(s["n"][0], s["x"][0], ev2[tau][0]), "E2_placebo": fit(s["n"][1], s["x"][1], ev2[tau][1])}
        print(tau, {k: (round(v["kappa_hat"], 5), round(v["LR_vs_null"], 1)) for k, v in out["taus"][str(tau)].items()}, flush=True)
    json.dump(out, open(os.path.join(RESULTS_DIR, "r28_hazard_all_us.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
