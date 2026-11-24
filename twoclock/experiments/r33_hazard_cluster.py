"""R33 -- cluster-robust uncertainty for the all-US hazard fit (R28) and data for the calibration figure.

Same data, wear path (tau = 12 h), risk set (load >= 0.6 rated in hour t-1), event (trip-like shutdown in
hour t) and 4 load strata as R28.  Binned sufficient statistics are kept PER STATE (49 clusters), so we can
(a) refit on the pooled data, (b) delete-one-state jackknife, (c) cluster bootstrap over states (400 draws).
The R28 CI treats hours as independent; clustering by state absorbs plant-, unit- and day-level dependence.
Also saves pooled (stratum x W^2 bin) counts for a figure of observed vs. baseline-expected trip rate.

Run: python -m twoclock.experiments.r33_hazard_cluster [dir with cems_all_<year>.parquet]
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

SIGMA, K_AGE, NB, NS, TAU = 0.85, 0.55, 400, 4, 12.0


def wear_matrix(lf, tau):
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
    st = np.empty(len(uid), dtype=object)
    st[inv] = d["state"].to_numpy()
    return P, O, st


def fit(n, xs, d):
    S = n.shape[0]
    mean_x = np.where(n > 0, xs / np.maximum(n, 1), 0.0)

    def nll(t):
        b = np.exp(t[S]); tot = 0.0
        for k in range(S):
            lam = np.maximum(np.exp(t[k]) + b * mean_x[k], 1e-300)
            tot += np.sum(n[k] * lam) - np.sum(d[k] * np.log(lam))
        return float(tot)
    h0s = [max(d[k].sum() / max(n[k].sum(), 1), 1e-12) for k in range(S)]
    best = None
    for b0 in (1e-5, 1e-4, 1e-3):
        r = minimize(nll, [np.log(h) for h in h0s] + [np.log(b0)], method="Nelder-Mead",
                     options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 6000, "maxfev": 6000})
        if best is None or r.fun < best.fun:
            best = r
    l0 = sum(float(np.sum(n[k]) * h0s[k] - d[k].sum() * np.log(h0s[k])) for k in range(S))
    return 24 * float(np.exp(best.x[S])), max(0.0, 2 * (l0 - best.fun))


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/kumar/cems_big"
    from twoclock.paths import RESULTS_DIR
    stats = {}
    for path in sorted(glob.glob(os.path.join(base, "cems_all_*.parquet"))):
        P, O, st = load_year(path)
        opn = O > 0
        keep = opn.sum(0) >= 2000
        P, O, opn, st = P[:, keep], O[:, keep], opn[:, keep], st[keep]
        rated = np.nanpercentile(np.where(opn, P, np.nan), 99.5, axis=0)
        big = rated >= 100
        P, O, opn, rated, st = P[:, big], O[:, big], opn[:, big], rated[big], st[big]
        U = P.shape[1]
        for c0 in range(0, U, 250):
            sl = slice(c0, min(U, c0 + 250))
            Pc, opc, rc, sc = P[:, sl], opn[:, sl], rated[sl], st[sl]
            lf = np.clip(Pc / rc[None, :], 0, 1.5).astype(np.float32)
            risk = Pc[:-1] >= 0.6 * rc[None, :]
            ev = risk & (~opc[1:])
            strat = np.digitize(Pc[:-1] / rc[None, :], [0.75, 0.85, 0.95]).astype(np.int32)
            W = wear_matrix(lf, TAU)
            x = W[:-1] ** 2
            flat = strat * NB + np.minimum((x * NB).astype(np.int32), NB - 1)
            for s in np.unique(sc):
                cols = np.where(sc == s)[0]
                f = flat[:, cols]; r = risk[:, cols]; e = ev[:, cols]; xx = x[:, cols]
                a = stats.setdefault(str(s), {k: np.zeros(NS * NB) for k in ("n", "x", "d")})
                a["n"] += np.bincount(f[r], minlength=NS * NB)
                a["x"] += np.bincount(f[r], weights=xx[r], minlength=NS * NB)
                a["d"] += np.bincount(f[e], minlength=NS * NB)
            del W, x, flat, lf, risk, ev, strat
        print(os.path.basename(path), "states so far", len(stats), flush=True)
        del P, O, opn

    states = sorted(stats)
    def pool(sel):
        return [sum(stats[s][k] for s in sel).reshape(NS, NB) for k in ("n", "x", "d")]
    n, xs, d = pool(states)
    k_hat, lr = fit(n, xs, d)
    print("pooled kappa", k_hat, "LR", lr, flush=True)
    jack = []
    for s in states:
        jack.append(fit(*pool([t for t in states if t != s]))[0])
    G = len(states); jm = float(np.mean(jack))
    se = float(np.sqrt((G - 1) / G * np.sum((np.array(jack) - jm) ** 2)))
    print("jackknife se", se, flush=True)
    rng = np.random.default_rng(33)
    boot = []
    for b in range(400):
        sel = list(rng.choice(states, G, replace=True))
        boot.append(fit(*pool(sel)))
    bk = np.array([b[0] for b in boot]); bl = np.array([b[1] for b in boot])
    out = {"states": G, "kappa_hat": k_hat, "LR_iid": lr, "jackknife_se": se,
           "jackknife_ci95": [k_hat - 1.96 * se, k_hat + 1.96 * se],
           "boot_ci95": [float(np.percentile(bk, 2.5)), float(np.percentile(bk, 97.5))],
           "boot_frac_kappa_positive": float((bk > 0).mean()), "boot_LR_p05": float(np.percentile(bl, 5)),
           "per_state_jackknife_min_max": [float(min(jack)), float(max(jack))],
           "pooled_n": n.tolist(), "pooled_x": xs.tolist(), "pooled_d": d.tolist()}
    json.dump(out, open(os.path.join(RESULTS_DIR, "r33_hazard_cluster.json"), "w"))
    print({k: v for k, v in out.items() if not k.startswith("pooled")})


if __name__ == "__main__":
    main()
