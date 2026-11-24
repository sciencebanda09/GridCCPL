"""R23 -- random reduced-form parameters (a calibration-free robustness check): wear parameters
drawn from wide priors; does the certificate stay within a small gap of the best constant price?

Original R17 text: long-memory wear

For each true tau and regime: certificate (correct model, training seeds >=101), blind OPF,
and a dense 24-point log grid of constant prices on test seeds 1-5 (oracle best constant).
Reports the certificate's cost gap to the oracle grid-best and its saving vs blind.

Run: python -m twoclock.experiments.r23_param_prior
"""
from __future__ import annotations
import json, os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import numpy as np

REG = [("peak", "peak", 1.0), ("tight1.15", "normal", 1.15)]
TEST = (1, 2, 3)
NS = 12


def draw(i):
    r = np.random.default_rng(23000 + i)
    return dict(c_age=float(np.exp(r.uniform(np.log(0.25), np.log(4.0)))) ,
                derate_lin=float(r.uniform(0.02, 0.30)), derate_quad=float(r.uniform(0.4, 2.4)),
                heat_lin=float(r.uniform(0.03, 0.30)), heat_quad=float(r.uniform(0.3, 1.6)),
                tau_thermal=float(r.uniform(1.5, 12.0)))


def one(args):
    name, diff, ds, idx = args
    pr = draw(idx)
    from twoclock.agents import OPFAware, OPFBlind
    from twoclock.agents.certificate import fixed_point_certificate
    from twoclock.envs import make_env
    from twoclock.experiments import r6_certificate as R
    from twoclock.experiments.runner import run_episode
    from twoclock.physics import AgingParams
    aging = replace(AgingParams(), **pr)
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
    curve = {float(g): float(np.mean([cost(g, s) for s in TEST])) for g in R.GRID_COARSE}
    best_mu = min(curve, key=curve.get)
    best = curve[best_mu]
    c_cert = float(np.mean([cost(mu, s) for s in TEST]))
    near = [g for g, v in curve.items() if v <= best * 1.0025]
    return {"regime": name, "sample": idx, "params": pr, "mu_lo": lo, "mu_hi": hi, "mu_cert": mu,
            "iterations": cert["iterations"], "converged": cert["converged"],
            "env_steps": cert["env_steps"], "blind": blind, "cert": c_cert,
            "oracle_best": best, "oracle_best_mu": best_mu,
            "gap_pct": 100 * (c_cert / best - 1), "saving_pct": 100 * (1 - c_cert / blind),
            "band_grid_lo": min(near), "band_grid_hi": max(near)}


def main():
    from twoclock.paths import RESULTS_DIR
    jobs = [(n, d, s, i) for i in range(NS) for (n, d, s) in REG]
    with ProcessPoolExecutor(max_workers=4) as ex:
        rows = list(ex.map(one, jobs))
    with open(os.path.join(RESULTS_DIR, "r23_param_prior.json"), "w") as fh:
        json.dump(rows, fh, indent=1, default=float)
    g = np.array([r["gap_pct"] for r in rows])
    print("n=%d  gap%%: median %.3f  mean %.3f  max %.3f  frac<=0.5%%: %.2f  min saving %.1f%%"
          % (len(g), np.median(g), g.mean(), g.max(), (g <= 0.5).mean(), min(r["saving_pct"] for r in rows)))
    for r in rows:
        print(r["regime"], r["sample"], "tau=%.1f c_age=%.2f gap %.3f%% saving %.1f%% conv=%s"
              % (r["params"]["tau_thermal"], r["params"]["c_age"], r["gap_pct"], r["saving_pct"], r["converged"]))

if __name__ == "__main__":
    main()
