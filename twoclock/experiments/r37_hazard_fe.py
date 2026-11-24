"""R37 -- is the hazard link confounded?  Unit-year x load-stratum x quarter fixed effects, and a
state-clustered comparison of tau = 12 h against tau = 1.5 h.

Model (multiplicative, fixed effects profiled out exactly): for hour i in cell c (unit-year x 4 load strata x
4 quarters), trip rate = h_c * (1 + theta * W_i^2).  Conditional (profile) log-likelihood
    l(theta) = sum_events log(1 + theta x_e) - sum_c D_c log(N_c + theta X_c),   x = W^2,
where N_c = risk hours, X_c = sum of x over them, D_c = events.  This absorbs everything constant within a
unit-year, load stratum and season (unit identity, hot-day/summer effects at quarter resolution, load level).
'pool' = same with cells = 4 load strata only (R28-style baseline), for comparison on the same scale.
theta is the relative excess trip rate at W = 1.  Clustering by state: jackknife-by-state and
per-state log-likelihood differences between tau = 12 h and tau = 1.5 h.

Run: python -m twoclock.experiments.r37_hazard_fe [dir with cems_all_<year>.parquet]
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

SIGMA, K_AGE = 0.85, 0.55
TAUS = (1.5, 12.0)


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
    P = np.zeros((T, len(uid)), np.float32); O = np.zeros((T, len(uid)), np.float32)
    P[h, inv] = d["gross_load_mw"].to_numpy(np.float32); O[h, inv] = d["operating_time_hours"].to_numpy(np.float32)
    q = np.zeros(T, np.int32); q[h] = ((t.dt.month.to_numpy() - 1) // 3).astype(np.int32)
    st = np.empty(len(uid), dtype=object); st[inv] = d["state"].to_numpy()
    return P, O, q, st


def theta_hat(cells, ex):
    """cells: array (n,3) of N,X,D ; ex: event x values."""
    N, X, D = cells[:, 0], cells[:, 1], cells[:, 2]
    def nll(th):
        return -(np.sum(np.log1p(th * ex)) - np.sum(D * np.log(N + th * X)))
    r = minimize_scalar(nll, bounds=(0.0, 1e4), method="bounded", options={"xatol": 1e-6})
    l1 = -r.fun; l0 = -nll(0.0)
    return float(r.x), float(l1), float(2 * (l1 - l0))


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "cems_data"
    from twoclock.paths import RESULTS_DIR
    store = {(tau, mode): {} for tau in TAUS for mode in ("fe", "pool")}      # -> state -> [cells list, ex list]
    for path in sorted(glob.glob(os.path.join(base, "cems_all_*.parquet"))):
        P, O, q, st = load_year(path)
        opn = O > 0
        keep = opn.sum(0) >= 2000
        P, O, opn, st = P[:, keep], O[:, keep], opn[:, keep], st[keep]
        rated = np.nanpercentile(np.where(opn, P, np.nan), 99.5, axis=0)
        big = rated >= 100
        P, opn, rated, st = P[:, big], opn[:, big], rated[big], st[big]
        U = P.shape[1]
        for c0 in range(0, U, 60):
            sl = slice(c0, min(U, c0 + 60))
            Pc, opc, rc, sc = P[:, sl], opn[:, sl], rated[sl], st[sl]
            Uc = Pc.shape[1]
            lf = np.clip(Pc / rc[None, :], 0, 1.5).astype(np.float32)
            risk = Pc[:-1] >= 0.6 * rc[None, :]
            ev = risk & (~opc[1:])
            strat = np.digitize(Pc[:-1] / rc[None, :], [0.75, 0.85, 0.95]).astype(np.int32)
            qq = np.broadcast_to(q[:-1, None], risk.shape)
            cid_fe = (np.arange(Uc, dtype=np.int32)[None, :] * 16 + strat * 4 + qq)
            for tau in TAUS:
                x = wear_matrix(lf, tau)[:-1] ** 2
                for mode, cid, ncell in (("fe", cid_fe, Uc * 16), ("pool", strat, 4)):
                    Nn = np.bincount(cid[risk], minlength=ncell).astype(np.float64)
                    Xx = np.bincount(cid[risk], weights=x[risk], minlength=ncell)
                    Dd = np.bincount(cid[ev], minlength=ncell).astype(np.float64)
                    for s in np.unique(sc):
                        cols = np.where(sc == s)[0]
                        if mode == "fe":
                            idx = (cols[:, None] * 16 + np.arange(16)[None, :]).ravel()
                            cells = np.stack([Nn[idx], Xx[idx], Dd[idx]], 1)
                            cells = cells[cells[:, 2] > 0]
                        else:
                            # pool mode: strata cells are shared across states -> need per-state strata counts
                            cells = None
                        exv = x[:, cols][ev[:, cols]]
                        a = store[(tau, mode)].setdefault(str(s), [[], []])
                        if mode == "fe":
                            a[0].append(cells)
                        a[1].append(exv.astype(np.float64))
                # pool mode per-state strata counts
                for s in np.unique(sc):
                    cols = np.where(sc == s)[0]
                    r = risk[:, cols]; e = ev[:, cols]; sg = strat[:, cols]; xs = x[:, cols]
                    cells = np.stack([np.bincount(sg[r], minlength=4).astype(np.float64),
                                      np.bincount(sg[r], weights=xs[r], minlength=4),
                                      np.bincount(sg[e], minlength=4).astype(np.float64)], 1)
                    store[(tau, "pool")][str(s)][0].append(cells)
                del x
            del lf, risk, ev, strat, qq, cid_fe
        print(os.path.basename(path), "states", len(store[(TAUS[0], "fe")]), flush=True)
        del P, O, opn

    out = {}
    states = sorted(store[(TAUS[0], "fe")])

    def pooled(tau, mode, sel):
        if mode == "fe":
            cells = np.concatenate([np.concatenate(store[(tau, mode)][s][0]) for s in sel])
        else:
            cells = sum(sum(store[(tau, mode)][s][0]) for s in sel)
            cells = cells[cells[:, 2] > 0]
        ex = np.concatenate([np.concatenate(store[(tau, mode)][s][1]) for s in sel])
        return cells, ex

    for tau in TAUS:
        for mode in ("fe", "pool"):
            th, l1, lr = theta_hat(*pooled(tau, mode, states))
            jack = [theta_hat(*pooled(tau, mode, [t for t in states if t != s]))[0] for s in states]
            G = len(states); jm = np.mean(jack); se = float(np.sqrt((G - 1) / G * np.sum((np.array(jack) - jm) ** 2)))
            out[f"tau{tau}_{mode}"] = {"theta": th, "loglik": l1, "LR_iid": lr, "jackknife_se": se,
                                      "ci95": [th - 1.96 * se, th + 1.96 * se], "n_events": int(sum(len(np.concatenate(store[(tau, mode)][s][1])) for s in states))}
            print(tau, mode, out[f"tau{tau}_{mode}"], flush=True)
    # state-clustered comparison tau=12 vs 1.5 (fe model): per-state profile log-lik at pooled theta-hat
    per = []
    for s in states:
        l = {}
        for tau in TAUS:
            th = out[f"tau{tau}_fe"]["theta"]
            cells, ex = pooled(tau, "fe", [s])
            l[tau] = float(np.sum(np.log1p(th * ex)) - np.sum(cells[:, 2] * np.log(cells[:, 0] + th * cells[:, 1])))
        per.append(l[12.0] - l[1.5])
    per = np.array(per)
    out["tau_comparison_fe"] = {"states": len(states), "states_favouring_tau12": int((per > 0).sum()),
                                "sum_dloglik": float(per.sum()), "mean_dloglik_per_state": float(per.mean()),
                                "se_dloglik_per_state": float(per.std(ddof=1) / np.sqrt(len(per))),
                                "t_stat": float(per.mean() / (per.std(ddof=1) / np.sqrt(len(per))))}
    print(out["tau_comparison_fe"], flush=True)
    json.dump(out, open(os.path.join(RESULTS_DIR, "r37_hazard_fe.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
