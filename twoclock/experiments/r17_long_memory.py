"""R17 -- long-memory wear: does the certificate still match the best price when the
thermal time constant tau is 12-48 h (vs 3 h in the main experiments)?

For each true tau and regime: certificate (correct model, training seeds >=101), blind OPF,
and a dense 24-point log grid of constant prices on test seeds 1-5 (oracle best constant).
Reports the certificate's cost gap to the oracle grid-best and its saving vs blind.

Run: python -m twoclock.experiments.r17_long_memory
"""
from __future__ import annotations
import json, os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import numpy as np

TAUS = (12.0, 24.0, 48.0)
REG = [("normal", "normal", 1.0), ("peak", "peak", 1.0), ("tight1.15", "normal", 1.15)]
TEST = (1, 2, 3, 4, 5)


def one(args):
    name, diff, ds, tau = args
    from twoclock.agents import OPFAware, OPFBlind
    from twoclock.agents.certificate import fixed_point_certificate
    from twoclock.envs import make_env
    from twoclock.experiments import r6_certificate as R
    from twoclock.experiments.runner import run_episode
    from twoclock.physics import AgingParams
    aging = replace(AgingParams(), tau_thermal=tau)
    net = R.set_network("5bus")

    def env(seed):
        e = make_env(diff, days=10, seed=seed, demand_scale=ds, aging=aging)
        e.net = net
        return e

    def cost(mu, seed):
        ag = OPFBlind(net, aging) if mu <= 0 else OPFAware(net, aging, float(mu))
        return float(run_episode(env(seed), ag)["episode_cost"])

    cert = fixed_point_certificate(env, net, aging, R.CERT_SEEDS, run_episode, log=lambda *x: None)
    lo, hi = cert["mu_lo"], cert["mu_hi"]
    mu = float(np.sqrt(lo * hi)) if np.isfinite(hi) else 2.0 * lo
    blind = float(np.mean([cost(0.0, s) for s in TEST]))
    curve = {float(g): float(np.mean([cost(g, s) for s in TEST])) for g in R.GRID}
    best_mu = min(curve, key=curve.get)
    best = curve[best_mu]
    c_cert = float(np.mean([cost(mu, s) for s in TEST]))
    near = [g for g, v in curve.items() if v <= best * 1.0025]
    return {"regime": name, "tau": tau, "mu_lo": lo, "mu_hi": hi, "mu_cert": mu,
            "iterations": cert["iterations"], "converged": cert["converged"],
            "env_steps": cert["env_steps"], "blind": blind, "cert": c_cert,
            "oracle_best": best, "oracle_best_mu": best_mu,
            "gap_pct": 100 * (c_cert / best - 1), "saving_pct": 100 * (1 - c_cert / blind),
            "band_grid_lo": min(near), "band_grid_hi": max(near)}


def main():
    from twoclock.paths import RESULTS_DIR
    jobs = [(n, d, s, t) for t in TAUS for (n, d, s) in REG]
    with ProcessPoolExecutor(max_workers=3) as ex:
        rows = list(ex.map(one, jobs))
    with open(os.path.join(RESULTS_DIR, "r17_long_memory.json"), "w") as fh:
        json.dump(rows, fh, indent=1, default=float)
    for r in rows:
        print("tau=%4.0f %-9s cert=[%.0f,%.3g] mu=%.0f  gap %.3f%%  saving %.1f%%  oracle mu=%.0f band[%.0f,%.0f] conv=%s"
              % (r["tau"], r["regime"], r["mu_lo"], r["mu_hi"], r["mu_cert"], r["gap_pct"],
                 r["saving_pct"], r["oracle_best_mu"], r["band_grid_lo"], r["band_grid_hi"], r["converged"]))


if __name__ == "__main__":
    main()
