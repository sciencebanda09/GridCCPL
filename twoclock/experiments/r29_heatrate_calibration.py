"""R29 -- does heat rate degrade with the model's wear state?  (calibrating h1, h2 and tau)

EPA CEMS hourly heat input and gross load, all US units, 2022-2024.  For each unit-year (rated = P99.5,
>= 2000 operating hours, rated >= 100 MW) take hours with load >= 0.5 rated; hr_t = heat / load.
Within cells of (unit, load bin [0.5-0.65-0.8-0.9-1.0], month) we express hr relative to the cell mean,
y_t = hr_t / mean_cell - 1, so the static heat-rate curve, unit effects and seasonality drop out.
The model predicts y = h1 (W - Wbar) + h2 (W^2 - W2bar) with W from the wear ODE (default aging rate,
sigma = 0.85, dt = 1 h, clipped).  OLS of y on (W, W^2) demeaned within cell gives (h1, h2); standard
errors are clustered by unit-year.  Placebo: W shifted circularly within each unit-year.
Confounders left: ambient temperature (hot days raise both loading and heat rate); we report with and without
month-by-hour-of-day cells (partial control), so the estimate is an upper bound on a true wear effect.

Run: python -m twoclock.experiments.r29_heatrate_calibration [dir with hr_<year>.parquet]
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
import pandas as pd

SIGMA, K_AGE = 0.85, 0.55
TAUS = (3.0, 12.0, 24.0)
BINS = [0.65, 0.8, 0.9]


def wear_matrix(lf, tau):
    a = np.exp(-1.0 / tau)
    inc = (K_AGE / (1.0 - SIGMA)) * np.maximum(0.0, lf - SIGMA)
    W = np.empty_like(lf)
    w = np.zeros(lf.shape[1], dtype=lf.dtype)
    for t in range(lf.shape[0]):
        w = np.minimum(1.0, a * w + inc[t])
        W[t] = w
    return W


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/kumar/cems_big"
    from twoclock.paths import RESULTS_DIR
    rng = np.random.default_rng(29)
    acc = {(tau, ctrl, pl): [np.zeros((2, 2)), np.zeros(2), 0.0, []] for tau in TAUS for ctrl in ("month", "month_hod") for pl in (0, 1)}
    n_rows = n_units = 0
    for path in sorted(glob.glob(os.path.join(base, "hr_*.parquet"))):
        d = pd.read_parquet(path, columns=["plant_id_eia", "emissions_unit_id_epa", "operating_datetime_utc",
                                           "gross_load_mw", "heat_content_mmbtu"])
        t = d["operating_datetime_utc"].astype("datetime64[ns]")
        h = ((t - t.min()).dt.total_seconds() // 3600).astype(np.int64).to_numpy()
        ucode, _ = pd.factorize(d["emissions_unit_id_epa"])
        key = d["plant_id_eia"].to_numpy(np.int64) * 100000 + ucode.astype(np.int64)
        uid, inv = np.unique(key, return_inverse=True)
        T = int(h.max()) + 1
        P = np.zeros((T, len(uid)), np.float32); H = np.zeros((T, len(uid)), np.float32)
        P[h, inv] = d["gross_load_mw"].to_numpy(np.float32)
        H[h, inv] = d["heat_content_mmbtu"].to_numpy(np.float32)
        month = (t.dt.month.to_numpy() - 1)
        mo_of_h = np.zeros(T, np.int64); mo_of_h[h] = month
        hod = (np.arange(T) % 24)
        del d
        opn = P > 0
        keep = opn.sum(0) >= 2000
        P, H, opn = P[:, keep], H[:, keep], opn[:, keep]
        rated = np.nanpercentile(np.where(opn, P, np.nan), 99.5, axis=0)
        big = rated >= 100
        P, H, opn, rated = P[:, big], H[:, big], opn[:, big], rated[big]
        U = P.shape[1]
        lf = np.clip(P / rated[None, :], 0, 1.5).astype(np.float32)
        sel = (lf >= 0.5) & (H > 0)
        hr = np.where(sel, H / np.maximum(P, 1e-6), np.nan)
        lb = np.digitize(lf, BINS)                                    # 0..3
        sh = rng.integers(500, T - 500, size=U)
        n_units += U
        for tau in TAUS:
            W = wear_matrix(lf, tau)
            idxs = (np.arange(T)[:, None] + sh[None, :]) % T
            Wp = np.take_along_axis(W, idxs, axis=0)
            for pl, Wx in ((0, W), (1, Wp)):
                for ctrl in ("month", "month_hod"):
                    cell_t = mo_of_h if ctrl == "month" else mo_of_h * 24 + hod
                    nct = 12 if ctrl == "month" else 12 * 24
                    # cell id = (unit, load bin, time cell)
                    cid = ((np.arange(U)[None, :] * 4 + lb) * nct + cell_t[:, None])
                    m = sel
                    c = cid[m]; y = hr[m]; w1 = Wx[m].astype(np.float64); w2 = w1 ** 2
                    uc, cinv = np.unique(c, return_inverse=True)
                    cnt = np.bincount(cinv)
                    ok = cnt[cinv] >= 5
                    cm = np.bincount(cinv, weights=y) / cnt
                    yy = y / cm[cinv] - 1.0
                    m1 = np.bincount(cinv, weights=w1) / cnt; m2 = np.bincount(cinv, weights=w2) / cnt
                    x1 = (w1 - m1[cinv])[ok]; x2 = (w2 - m2[cinv])[ok]; yo = yy[ok]
                    X = np.stack([x1, x2], 1)
                    # cluster by unit-year (unit index of each obs)
                    ucol = np.where(m)[1][ok]
                    a = acc[(tau, ctrl, pl)]
                    a[0] += X.T @ X; a[1] += X.T @ yo
                    cols = [x1 * x1, x1 * x2, x2 * x2, x1 * yo, x2 * yo]
                    a[3].append(np.stack([np.bincount(ucol, weights=c_, minlength=U) for c_ in cols], 1))
            n_rows += int(sel.sum())
        print(os.path.basename(path), "units", U, "rows", int(sel.sum()), flush=True)
        # keep memory: reduce stored obs to cluster score sums at year end
        for key_, a in acc.items():
            if a[3]:
                XtX_year = None
        del P, H, lf, hr, opn
    out = {"units": n_units, "rows": n_rows, "fits": {}}
    for (tau, ctrl, pl), a in acc.items():
        XtX, Xty = a[0], a[1]
        beta = np.linalg.solve(XtX, Xty)
        # clustered sandwich: score per cluster
        meat = np.zeros((2, 2))
        for st in a[3]:
            for row in st:
                xx = np.array([[row[0], row[1]], [row[1], row[2]]]); xy = np.array([row[3], row[4]])
                sc = xy - xx @ beta
                meat += np.outer(sc, sc)
        Vinv = np.linalg.inv(XtX)
        cov = Vinv @ meat @ Vinv
        out["fits"][f"tau{tau}_{ctrl}_{'placebo' if pl else 'real'}"] = {
            "h1": float(beta[0]), "h2": float(beta[1]), "se_h1": float(np.sqrt(cov[0, 0])), "se_h2": float(np.sqrt(cov[1, 1])),
            "hr_gain_at_W0.5_pct": float(100 * (beta[0] * 0.5 + beta[1] * 0.25)),
            "hr_gain_at_W1_pct": float(100 * (beta[0] + beta[1]))}
    json.dump(out, open(os.path.join(RESULTS_DIR, "r29_heatrate_calibration.json"), "w"), indent=1)
    for k, v in out["fits"].items():
        print(k, {kk: round(vv, 4) for kk, vv in v.items()})


if __name__ == "__main__":
    main()
