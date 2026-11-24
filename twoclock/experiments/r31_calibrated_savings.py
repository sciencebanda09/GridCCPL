"""R31 -- savings under CALIBRATED wear parameters (heat-rate channel off as estimated in R29, tau = 12 h as in R28;
derating a1,a2 unidentified, so varied x1, x0.5, x0).  Original R23 text: random reduced-form parameters (a calibration-free robustness check): wear parameters
drawn from wide priors; does the certificate stay within a small gap of the best constant price?

Original R17 text: long-memory wear

For each true tau and regime: certificate (correct model, training seeds >=101), blind OPF,
and a dense 24-point log grid of constant prices on test seeds 1-5 (oracle best constant).
Reports the certificate's cost gap to the oracle grid-best and its saving vs blind.

Run: python -m twoclock.experiments.r31_calibrated_savings
"""
from __future__ import annotations
import json, os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import numpy as np

REG = [("normal", "normal", 1.0), ("peak", "peak", 1.0), ("tight1.15", "normal", 1.15)]
TEST = (1, 2, 3)
NS = 5


VARIANTS = [("cal_tau12_derate1.0", dict(heat_lin=0.0, heat_quad=0.0, tau_thermal=12.0)),
            ("cal_tau12_derate0.5", dict(heat_lin=0.0, heat_quad=0.0, tau_thermal=12.0, derate_lin=0.05, derate_quad=0.6)),
            ("cal_tau12_derate0.25", dict(heat_lin=0.0, heat_quad=0.0, tau_thermal=12.0, derate_lin=0.025, derate_quad=0.3)),
            ("cal_tau12_derate0", dict(heat_lin=0.0, heat_quad=0.0, tau_thermal=12.0, derate_lin=0.0, derate_quad=0.0)),
            ("cal_tau3_derate1.0", dict(heat_lin=0.0, heat_quad=0.0, tau_thermal=3.0))]


def draw(i):
    return VARIANTS[i][1]


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
    return {"regime": name, "sample": idx, "variant": VARIANTS[idx][0], "params": pr, "mu_lo": lo, "mu_hi": hi, "mu_cert": mu,
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
    with open(os.path.join(RESULTS_DIR, "r31_calibrated_savings.json"), "w") as fh:
        json.dump(rows, fh, indent=1, default=float)
    for r in rows:
        print(r["variant"], r["regime"], "gap %.3f%% saving vs blind %.1f%% (cert %.2fM blind %.2fM best %.2fM, cert mu %.0f best mu %.0f)"
              % (r["gap_pct"], r["saving_pct"], r["cert"] / 1e6, r["blind"] / 1e6, r["oracle_best"] / 1e6, r["mu_cert"], r["oracle_best_mu"]))

if __name__ == "__main__":
    main()
