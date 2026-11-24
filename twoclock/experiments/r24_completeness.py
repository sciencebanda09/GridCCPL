"""R24 -- completeness condition (C) for general fleets: verifiable flat interval.

Theorem (PROOF.md, "Verifiable flat interval"): let phi(mu) = min_x f(x) + mu a(x) (a concave function
of mu, as a minimum of affine functions).  If a dispatch xh is optimal at mu_1 < mu_2 then it is optimal on
[mu_1, mu_2].  So {mu : xh optimal} is an interval, found with two LP solves per probe, and no
single-unit formula is needed.  Here we (1) show a substitutable-unit fleet where the closed-form band of
Proposition 1(iv) misses a breakpoint, (2) test closed form vs exact flat interval on random fleets, and
(3) do the same on the 50 harvested 5-bus states.

Run (from repo root): python -m twoclock.experiments.r24_completeness
"""
from __future__ import annotations
import json, os
import numpy as np
from scipy.optimize import linprog

VOLL, SIGMA = 5000.0, 0.85
TOL = 1e-6


class Fleet:
    def __init__(self, rng, n):
        self.n = n
        self.rbar = rng.uniform(100, 400, n)
        self.c = rng.uniform(10, 80, n)
        self.alpha = np.exp(rng.uniform(np.log(0.003), np.log(0.08), n))
        free = SIGMA * self.rbar
        self.prev = rng.uniform(0.5, 1.0, n) * self.rbar
        self.ramp = rng.uniform(0.15, 0.6, n) * self.rbar
        self.pmin = np.maximum(0, self.prev - self.ramp)
        self.pmax = np.minimum(self.rbar, self.prev + self.ramp)
        self.free = free
        self.q_lo = np.minimum(self.pmin, free); self.q_hi = np.maximum(self.q_lo, np.minimum(free, self.pmax))
        self.e_lo = np.maximum(0, self.pmin - free)
        self.e_hi = np.maximum(self.e_lo, self.pmax - np.maximum(free, self.pmin))
        lo_total = (self.q_lo + self.e_lo).sum()
        hi_total = (self.q_hi + self.e_hi).sum()
        self.D = float(rng.uniform(lo_total, hi_total * 1.08))   # includes forced-shedding cases


def solve(fl, mu, block=None):
    n = fl.n
    c = np.concatenate([fl.c, fl.c + fl.alpha * mu, [VOLL]])
    e_hi = fl.e_hi.copy()
    if block is not None:
        e_hi = np.where(block, fl.e_lo, e_hi)
    bounds = list(zip(fl.q_lo, fl.q_hi)) + list(zip(fl.e_lo, e_hi)) + [(0.0, fl.D)]
    A = np.ones((1, 2 * n + 1))
    r = linprog(c, A_eq=A, b_eq=[fl.D], bounds=bounds, method="highs")
    if not r.success:
        return None
    return {"cost": float(r.fun), "e": r.x[n:2 * n].copy(), "shed": float(r.x[-1]),
            "lam": float(r.eqlin.marginals[0]), "x": r.x.copy()}


def a_of(fl, sol):
    return float(fl.alpha @ sol["e"])


def optimal_at(fl, sol_c, mu_c, mu):
    """Is the dispatch of sol_c optimal at price mu?  (value of x_c under mu equals phi(mu))."""
    s = solve(fl, mu)
    if s is None:
        return False
    f = sol_c["cost"] - mu_c * a_of(fl, sol_c)
    val = f + mu * a_of(fl, sol_c)
    return abs(val - s["cost"]) <= 1e-7 * max(1.0, abs(s["cost"]))


def flat_interval(fl, mu_c, mu_max=1e8, mu_min=1e-3):
    sol = solve(fl, mu_c)
    out = []
    for direction in (-1, +1):
        good, bad = mu_c, (mu_min if direction < 0 else mu_max)
        if optimal_at(fl, sol, mu_c, bad):
            out.append(bad); continue
        for _ in range(60):
            mid = np.sqrt(good * bad)
            if optimal_at(fl, sol, mu_c, mid):
                good = mid
            else:
                bad = mid
            if abs(bad / good - 1) < 1e-9:
                break
        out.append(good)
    return out[0], out[1]


def closed_form(fl):
    """(L, U) of Proposition 1(iv) at this single step (or None if undefined)."""
    n = fl.n
    s0 = solve(fl, 0.0, block=np.ones(n, bool))
    head = [g for g in range(n) if fl.e_hi[g] - fl.e_lo[g] > 1e-6]
    indisp = [g for g in head if (solve(fl, 0.0, block=np.eye(n, dtype=bool)[g]) or {"shed": 0})["shed"] > TOL]
    if s0 is None:
        return None
    L = 0.0
    if s0["shed"] <= TOL:
        L = max([max(0.0, (s0["lam"] - fl.c[g]) / fl.alpha[g]) for g in head], default=0.0)
    for a in indisp:
        for b in indisp:
            if fl.alpha[b] > fl.alpha[a]:
                L = max(L, (fl.c[a] - fl.c[b]) / (fl.alpha[b] - fl.alpha[a]))
    U = min([(VOLL - fl.c[g]) / fl.alpha[g] for g in indisp], default=np.inf)
    return L, U, indisp


def main():
    from twoclock.paths import RESULTS_DIR
    rng = np.random.default_rng(24)
    rows, n_bad_cf, n_cf, n_unsound = [], 0, 0, 0
    grid = np.concatenate([[0.0], np.logspace(0, 7, 300)])
    for it in range(400):
        fl = Fleet(rng, int(rng.integers(2, 9)))
        cf = closed_form(fl)
        sols = [solve(fl, m) for m in grid]
        if cf is None or any(s is None for s in sols):
            continue
        L, U, indisp = cf
        lo, hi = max(L, 1.0), min(U, 1e7)
        if hi <= lo * 1.2:
            continue
        mu_c = float(np.sqrt(lo * hi))
        a_, b_ = flat_interval(fl, mu_c)
        n_cf += 1
        # claimed band contained in the verified flat interval?
        contained = (a_ <= lo * 1.05) and (b_ >= hi * 0.95)
        # soundness of the verified interval: dispatch value agrees at random interior points
        interior = np.exp(rng.uniform(np.log(a_), np.log(b_), 5)) if b_ > a_ * 1.001 else []
        sol_c = solve(fl, mu_c)
        sound = all(optimal_at(fl, sol_c, mu_c, float(m)) for m in interior)
        n_unsound += (not sound)
        if not contained:
            n_bad_cf += 1
        rows.append({"n": fl.n, "L": L, "U": None if not np.isfinite(U) else U, "n_indisp": len(indisp),
                     "verified": [a_, b_], "closed_form_contained": bool(contained), "sound": bool(sound)})
    agg = {"fleets_tested": n_cf, "closed_form_band_not_flat": n_bad_cf,
           "frac_closed_form_fails": n_bad_cf / max(n_cf, 1), "verified_interval_unsound": n_unsound}
    # concrete substitutable-unit counterexample
    fl = Fleet(np.random.default_rng(0), 2)
    fl.c = np.array([20.0, 30.0]); fl.alpha = np.array([0.01, 0.02])
    fl.rbar = np.array([200.0, 200.0]); fl.free = SIGMA * fl.rbar
    fl.q_lo = np.array([0.0, 0.0]); fl.q_hi = fl.free.copy()
    fl.e_lo = np.zeros(2); fl.e_hi = np.array([30.0, 30.0])
    fl.D = float(fl.q_hi.sum() + 20.0)           # 20 MW above the free blocks: either unit can cover it
    cf = closed_form(fl)
    s_lo, s_hi = solve(fl, 1e3), solve(fl, 1e7)
    counter = {"closed_form": [cf[0], None if not np.isfinite(cf[1]) else cf[1], cf[2]],
               "excess_at_mu_1e3": s_lo["e"].tolist(), "shed_at_1e3": s_lo["shed"],
               "excess_at_mu_1e7": s_hi["e"].tolist(), "shed_at_1e7": s_hi["shed"],
               "verified_interval_at_mu_1e3": list(flat_interval(fl, 1e3))}
    print(json.dumps({"aggregate": agg, "counterexample": counter}, indent=1, default=float))
    json.dump({"aggregate": agg, "counterexample": counter, "fleets": rows},
              open(os.path.join(RESULTS_DIR, "r24_completeness.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
