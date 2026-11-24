"""R27 -- trajectory-level verification of (C) for the seven paper regimes.

Along each certificate-price trajectory, at every step we solve the priced LP at mu_c and at the band
endpoints mu_lo*1.02 and mu_hi*0.98 (1e6 if mu_hi is infinite) and test that the dispatch chosen at mu_c
is optimal at both endpoints (value comparison, Theorem 2 in PROOF.md).  If it passes at every step, the
band is verified flat on this trajectory *without* the single-unit formulas.

Run: python -m twoclock.experiments.r27_verify_band
"""
from __future__ import annotations
import json, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np

REG = [("normal", "normal", 1.0), ("hard", "hard", 1.0), ("peak", "peak", 1.0), ("storm", "storm", 1.0),
       ("tight1.15", "normal", 1.15), ("tight1.25", "normal", 1.25), ("tight1.35", "normal", 1.35)]
SEEDS = (1, 2)


def one(args):
    name, diff, ds = args
    from twoclock.agents import OPFAware
    from twoclock.envs import make_env
    from twoclock.experiments.r10_verify import make_solver
    from twoclock.paths import RESULTS_DIR
    from twoclock.physics import AgingParams, build_5bus
    aging, net = AgingParams(), build_5bus()
    row = [r for r in json.load(open(os.path.join(RESULTS_DIR, "r13_baselines.json")))["regimes"]
           if r["regime"] == name][0]["sel_certificate"]
    mu_c, lo, hi = row["mu"], row["mu_lo"], row["mu_hi"]
    ends = (lo * 1.02, (hi * 0.98) if hi and np.isfinite(hi) else 1e6)
    n_steps = n_pass = 0
    worst = 0.0
    regret = 0.0
    fails = []
    for s in SEEDS:
        env = make_env(diff, days=10, seed=s, demand_scale=ds, aging=aging)
        env.net = net
        ag = OPFAware(net, aging, float(mu_c))
        o = env.reset(); ag.reset(env)
        while not env.done:
            solve, alpha = make_solver(net, aging, o.ctx)
            rc = solve(mu_c)
            if rc["status"] == "ok":
                a = float(alpha[:3] @ rc["excess"][:3])
                f = rc["cost"] - mu_c * a
                ok = True
                t_step = n_steps
                for m in ends:
                    ro = solve(m)
                    err = abs(f + m * a - ro["cost"]) / max(1.0, abs(ro["cost"]))
                    worst = max(worst, err)
                    ok &= err < 1e-7
                    if err >= 1e-7:
                        regret += (f + m * a - ro["cost"]) * 0.25   # $ over a 15-min step
                        fails.append({"seed": s, "k": t_step, "mu": float(m), "gap_per_h": float(f + m * a - ro["cost"])})
                n_steps += 1; n_pass += int(ok)
            o, _, _, _ = env.step(ag.act(o))
    return {"regime": name, "mu_c": mu_c, "mu_lo": lo, "mu_hi": hi, "endpoints": list(ends),
            "steps": n_steps, "pass": n_pass, "worst_rel_err": worst,
            "regret_usd_if_dispatch_kept": regret, "failures": fails}


def main():
    from twoclock.paths import RESULTS_DIR
    with ProcessPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(one, REG))
    json.dump(res, open(os.path.join(RESULTS_DIR, "r27_verify_band.json"), "w"), indent=1, default=float)
    for r in res:
        print("%-9s steps %d pass %d (%.2f%%) worst rel err %.1e  regret $%.0f over 2 episodes" %
              (r["regime"], r["steps"], r["pass"], 100 * r["pass"] / r["steps"], r["worst_rel_err"], r["regret_usd_if_dispatch_kept"]))


if __name__ == "__main__":
    main()
