"""R15 -- certificate under a misspecified aging model.

The operator's *belief* model has wrong slow-clock parameters.  The certificate is
computed by rolling out the belief simulator (training seeds >=101) and priced with the
belief's alpha_g / sigma; the resulting price is then deployed in the TRUE environment
(test seeds 1-20).  Reported: cost vs the correctly-specified certificate and vs blind.

Run: python -m twoclock.experiments.r15_misspec
"""
from __future__ import annotations
import json, os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import numpy as np
from scipy import stats

TEST = tuple(range(1, 21))
REG = [("normal", "normal", 1.0), ("hard", "hard", 1.0), ("peak", "peak", 1.0),
       ("storm", "storm", 1.0), ("tight1.15", "normal", 1.15),
       ("tight1.25", "normal", 1.25), ("tight1.35", "normal", 1.35)]
VARIANTS = {
    "exact": {},
    "age_rate_x0.5": {"c_age": 0.275},
    "age_rate_x2": {"c_age": 1.10},
    "sigma_0.80": {"stress_frac": 0.80},
    "sigma_0.90": {"stress_frac": 0.90},
    "damage_curv_x0.5": {"derate_quad": 0.60, "heat_quad": 0.40},
    "damage_curv_x2": {"derate_quad": 2.40, "heat_quad": 1.60},
    "tau_1.5h": {"tau_thermal": 1.5},
    "tau_6h": {"tau_thermal": 6.0},
}


def ci95(x):
    x = np.asarray(x, float)
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def one(args):
    name, diff, ds = args
    from twoclock.agents import OPFAware, OPFBlind
    from twoclock.agents.certificate import fixed_point_certificate
    from twoclock.envs import make_env
    from twoclock.experiments import r6_certificate as R
    from twoclock.experiments.runner import run_episode
    from twoclock.physics import AgingParams
    true = AgingParams()
    net = R.set_network("5bus")

    def env_true(seed):
        e = make_env(diff, days=10, seed=seed, demand_scale=ds)
        e.net = net
        return e

    def cost(agent, seed):
        return float(run_episode(env_true(seed), agent)["episode_cost"])

    blind = np.array([cost(OPFBlind(net, true), s) for s in TEST])
    out = {"regime": name, "blind": float(blind.mean()), "variants": {}}
    ref = None
    for vn, over in VARIANTS.items():
        belief = replace(true, **over)

        def env_belief(seed, _b=belief):
            e = make_env(diff, days=10, seed=seed, demand_scale=ds, aging=_b)
            e.net = net
            return e

        cert = fixed_point_certificate(env_belief, net, belief, R.CERT_SEEDS, run_episode,
                                       log=lambda *x: None)
        lo, hi = cert["mu_lo"], cert["mu_hi"]
        mu = float(np.sqrt(lo * hi)) if np.isfinite(hi) else 2.0 * lo
        c = np.array([cost(OPFAware(net, belief, mu), s) for s in TEST])
        if vn == "exact":
            ref = c
        out["variants"][vn] = {
            "mu": mu, "mu_lo": lo, "mu_hi": hi, "cost_mean": float(c.mean()),
            "cost_ci95": ci95(c), "per_seed": c.tolist(),
            "minus_exact_k": float(np.mean(c - ref) / 1e3),
            "minus_exact_ci_k": ci95(c - ref) / 1e3,
            "saving_vs_blind_pct": float(100 * np.mean((blind - c) / blind)),
            "frac_saving_lost": float(np.mean(c - ref) / np.mean(blind - ref))
            if np.mean(blind - ref) > 0 else None}
    return out


def main():
    from twoclock.paths import RESULTS_DIR
    with ProcessPoolExecutor(max_workers=len(REG)) as ex:
        rows = list(ex.map(one, REG))
    with open(os.path.join(RESULTS_DIR, "r15_misspec.json"), "w") as fh:
        json.dump(rows, fh, indent=1, default=float)
    for r in rows:
        print(r["regime"], {k: (round(v["mu"]), round(v["minus_exact_k"], 1))
                            for k, v in r["variants"].items()})


if __name__ == "__main__":
    main()
