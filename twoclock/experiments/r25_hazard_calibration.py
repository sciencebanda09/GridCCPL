"""R25 -- calibrating the outage-hazard link kappa and the thermal time constant tau from real data.

Data: EPA CEMS hourly gross load, all large NY units (P99.5 >= 100 MW, >= 2000 operating hours, 2024).
Event: a trip-like shutdown = load >= 0.6 * P99.5 in hour t-1 and zero operating time in hour t
(planned shutdowns ramp down first; this is a proxy, not a forced-outage log).
Model: per running hour hazard  lambda_t = h0 + beta * W_t^2,  W_t from the model's wear ODE with the
default aging rate (k = 0.55, sigma = 0.85, dt = 1 h, rated output = P99.5), for tau on a grid.
Model hazard per day is 1 - exp(-kappa W^2), so kappa_hat = 24 * beta (small-hazard regime).
Reports MLE, profile-likelihood 95% upper bound for kappa, LR test vs beta = 0, and a placebo in which
each unit's W path is circularly shifted by a random offset (breaks the link between loading history
and trips, keeps autocorrelation).

Run: python -m twoclock.experiments.r25_hazard_calibration [path/to/cems_ny_2024.parquet]
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

SIGMA, K_AGE = 0.85, 0.55
TAUS = (1.5, 3.0, 6.0, 12.0, 24.0, 48.0)


def wear_path(load_frac, tau, dt=1.0):
    a = np.exp(-dt / tau)
    inc = K_AGE * dt / (1.0 - SIGMA) * np.maximum(0.0, load_frac - SIGMA)
    W = np.empty_like(load_frac)
    w = 0.0
    for i in range(len(load_frac)):
        w = min(1.0, a * w + inc[i])
        W[i] = w
    return W


def nll(theta, y, x):
    h0, beta = np.exp(theta[0]), np.exp(theta[1]) if theta[1] > -50 else 0.0
    lam = h0 + beta * x
    return float(np.sum(lam) - np.sum(y * np.log(lam)))


def fit(y, x, beta_fixed=None):
    h0 = max(y.sum() / len(y), 1e-9)
    if beta_fixed is not None:
        return nll([np.log(h0), -60.0 if beta_fixed == 0 else np.log(beta_fixed)], y, x) if False else \
            float(len(y) * h0 - y.sum() * np.log(h0))
    best = None
    for b0 in (1e-5, 1e-4, 1e-3, 1e-2):
        r = minimize(lambda t: nll(t, y, x), [np.log(h0), np.log(b0)], method="Nelder-Mead",
                     options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    return best


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "research", "r12_unit_real_data", "data", "cems_ny_2024.parquet")
    from twoclock.paths import RESULTS_DIR
    c = pd.read_parquet(path).sort_values(["plant_id_eia", "emissions_unit_id_epa", "operating_datetime_utc"])
    units = []
    for k, g in c.groupby(["plant_id_eia", "emissions_unit_id_epa"]):
        p = g["gross_load_mw"].fillna(0).to_numpy(float)
        op = g["operating_time_hours"].fillna(0).to_numpy(float)
        if (op > 0).sum() < 2000:
            continue
        r = np.percentile(p[op > 0], 99.5)
        if r < 100:
            continue
        units.append((str(k), p / r, op, p >= 0.6 * r))
    rng = np.random.default_rng(25)
    out = {"n_units": len(units), "taus": {}}
    for tau in TAUS:
        Y, X, Xp = [], [], []
        for _, lf, op, hi in units:
            W = wear_path(lf, tau)
            y = ((hi[:-1]) & (op[1:] == 0)).astype(float)       # trip in next hour
            risk = op[:-1] > 0
            Y.append(y[risk]); X.append(W[:-1][risk] ** 2)
            Xp.append(np.roll(W, int(rng.integers(500, len(W) - 500)))[:-1][risk] ** 2)
        y, x, xp = map(np.concatenate, (Y, X, Xp))
        r1 = fit(y, x)
        l0 = fit(y, x, beta_fixed=0)
        beta = float(np.exp(r1.x[1]))
        lr = max(0.0, 2 * (l0 - r1.fun))
        # profile upper bound for beta (LR = 3.84 one-sided -> chi2 1df at 0.90 gives 2.71)
        up = beta
        for f in np.geomspace(1.0, 1e4, 60):
            b = beta * f if beta > 0 else 1e-9 * f
            rr = minimize(lambda t: nll([t[0], np.log(b)], y, x), [np.log(max(y.mean(), 1e-9))], method="Nelder-Mead")
            if 2 * (rr.fun - r1.fun) > chi2.ppf(0.90, 1):
                up = b
                break
        rp = fit(y, xp)
        out["taus"][str(tau)] = {"events": int(y.sum()), "risk_hours": int(len(y)), "h0_per_h": float(np.exp(r1.x[0])),
                                 "beta_per_h": beta, "kappa_hat": 24 * beta, "kappa_upper90": 24 * up,
                                 "LR_vs_null": lr, "loglik_gain_placebo": float(max(0.0, 2 * (l0 - rp.fun))),
                                 "mean_W2": float(x.mean()), "hi_load_share": float(np.mean(x > 0))}
        print(tau, out["taus"][str(tau)])
    json.dump(out, open(os.path.join(RESULTS_DIR, "r25_hazard_calibration.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
