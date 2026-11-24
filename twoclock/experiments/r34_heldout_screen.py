"""R34 -- HELD-OUT validation of the (A) deviation screen.

R32 scored the screen on the same 24 random-parameter cases (R23) that were used to look at the failures.
Here we draw 12 NEW parameter sets (seeds 24000+i, same priors, same two regimes, 24 new cases), run the
R23 comparison (certificate vs. best constant on a 24-point grid, test seeds 1-3) and the R32 screen with
the thresholds FROZEN from R32 (flag if cost falls at > 10% of sampled steps; deviation at every 32nd
step; fail = gap to best constant > 1% or worse than blind).  Nothing is tuned.

Run: python -m twoclock.experiments.r34_heldout_screen
"""
from __future__ import annotations
import json, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np

SEED0 = 24000


def _draw(i):
    r = np.random.default_rng(SEED0 + i)
    return dict(c_age=float(np.exp(r.uniform(np.log(0.25), np.log(4.0)))),
                derate_lin=float(r.uniform(0.02, 0.30)), derate_quad=float(r.uniform(0.4, 2.4)),
                heat_lin=float(r.uniform(0.03, 0.30)), heat_quad=float(r.uniform(0.3, 1.6)),
                tau_thermal=float(r.uniform(1.5, 12.0)))


def run23(args):
    from twoclock.experiments import r23_param_prior as R23
    R23.draw = _draw
    return R23.one(args)


def run32(row):
    from twoclock.experiments import r32_A_screen as R32
    return R32.one(row)


def main():
    from twoclock.paths import RESULTS_DIR
    from twoclock.experiments import r23_param_prior as R23
    jobs = [(n, d, s, i) for i in range(R23.NS) for (n, d, s) in R23.REG]
    with ProcessPoolExecutor(max_workers=4) as ex:
        rows = list(ex.map(run23, jobs))
    json.dump(rows, open(os.path.join(RESULTS_DIR, "r34_heldout_draws.json"), "w"), indent=1, default=float)
    print("draws done", flush=True)
    with ProcessPoolExecutor(max_workers=3) as ex:
        res = list(ex.map(run32, rows))
    tab = {"flag_fail": 0, "flag_ok": 0, "noflag_fail": 0, "noflag_ok": 0}
    for r in res:
        fail = (r["gap_pct"] > 1.0) or (r["saving_pct"] < 0)
        flag = r["share_A_violated"] > 0.10
        r["fail"], r["flag"] = bool(fail), bool(flag)
        tab[("flag" if flag else "noflag") + "_" + ("fail" if fail else "ok")] += 1
    g = np.array([r["gap_pct"] for r in rows])
    summ = {"n": len(rows), "within_0.5pct": int((g <= 0.5).sum()), "within_1pct": int((g <= 1.0).sum()),
            "worse_than_blind": int(sum(r["saving_pct"] < 0 for r in rows)),
            "gap_range_of_failures": [float(g[g > 1.0].min()) if (g > 1.0).any() else None, float(g.max())]}
    json.dump({"table": tab, "summary": summ, "cases": res}, open(os.path.join(RESULTS_DIR, "r34_heldout_screen.json"), "w"), indent=1)
    print(tab, summ)


if __name__ == "__main__":
    main()
