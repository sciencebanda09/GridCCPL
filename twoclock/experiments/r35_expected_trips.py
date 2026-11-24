"""R35 -- expected cost at the CALIBRATED hazard, computed analytically instead of by sampling rare trips.

At kappa_hat = 0.0086/day a 40-seed sample sees 0-1 trips, so the Tier-0-vs-Tier-3 verdict of R30 is
sampling noise.  The trip probability is a closed function of the wear path, p = 1 - exp(-kappa W_g^2)
per unit-day.  We run each policy with kappa = 0 (trips off, so the wear path is unaffected), record the
per-day hazard sum  E_trips(kappa) = sum_days sum_g (1 - exp(-kappa W_g^2))  for kappa_hat and for a
reference kappa where R11 sampled trips, and combine with a per-trip cost c_trip measured at that
reference kappa:   c_trip = (mean cost at kappa_ref - mean cost at 0) / mean trips at kappa_ref.
   E[cost(kappa_hat)] ~= cost0 + E_trips(kappa_hat) * c_trip
   CVaR90 (population) ~= CVaR90 of kappa=0 costs + E_trips(kappa_hat) * c_trip / 0.10   (valid for p << 0.1)
Validation: E_trips(kappa_ref) from the formula vs. trips sampled in R11.

Run: python -m twoclock.experiments.r35_expected_trips
"""
from __future__ import annotations
import json, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np

KAPPA_HAT = 0.0086
TIER3_MU = 706602.2727272727
TEST_SEEDS = list(range(1, 41))
REF_KAPPA = {"tight_1.15": 0.25, "heatwave_1.22": 1.0, "tight_1.25": 1.0}
_HAZ = {"hat": 0.0, "ref": 0.0, "kref": 1.0}

from twoclock.physics import aging as _ag
_orig = _ag.AgingModel.daily_update


def _patched(self, step, steps_per_day):
    w = np.clip(self.wear, 0.0, 1.0)
    _HAZ["hat"] += float(np.sum(1.0 - np.exp(-KAPPA_HAT * w ** 2)))
    _HAZ["ref"] += float(np.sum(1.0 - np.exp(-_HAZ["kref"] * w ** 2)))
    return _orig(self, step, steps_per_day)


_ag.AgingModel.daily_update = _patched


def work(task):
    scale, hw, mu, seed, kref = task
    from twoclock.experiments import r11_risk_outages as R
    _HAZ["hat"] = _HAZ["ref"] = 0.0; _HAZ["kref"] = kref
    row = R._run_episode_worker((scale, hw, 1e-12, {"kind": "const", "mu": mu}, seed, False))
    return {"seed": seed, "cost": row["cost"], "E_hat": _HAZ["hat"], "E_ref": _HAZ["ref"]}


def cvar(v, a=0.9):
    v = np.sort(np.asarray(v)); k = max(1, int(np.ceil((1 - a) * len(v))))
    return float(v[-k:].mean())


def main():
    from twoclock.paths import RESULTS_DIR
    r30 = json.load(open(os.path.join(RESULTS_DIR, "r30_outages_calibrated.json")))
    r11 = json.load(open(os.path.join(RESULTS_DIR, "r11_risk_outages.json")))
    out = {}
    rng = np.random.default_rng(35)
    with ProcessPoolExecutor(max_workers=4) as pool:
        for reg in r30:
            base = reg["regime"]["base"]; kref = REF_KAPPA[base]
            mu0 = reg["certificate"]["mu_tier0"]
            ref = [x for x in r11["regimes"] if x["regime"]["base"] == base and abs(x["regime"]["kappa"] - kref) < 1e-9][0]
            zero = [x for x in r11["regimes"] if x["regime"]["base"] == base and x["regime"]["kappa"] == 0.0]
            res = {}
            for pol, mu in (("tier0", mu0), ("tier3", TIER3_MU)):
                rows = list(pool.map(work, [(reg["regime"]["scale"], reg["regime"]["heatwave"], mu, s, kref) for s in TEST_SEEDS]))
                res[pol] = rows
                print(base, pol, "done", flush=True)
            # per-trip cost from R11 at kappa_ref (tier0): (cost_ref - cost0)/trips_ref
            t0_ref = ref["test_policies"]["tier0_cert"]
            c0 = float(np.mean([r["cost"] for r in res["tier0"]]))
            trips_ref = t0_ref["trips_mean"]
            c_trip = (t0_ref["mean"] - c0) / max(trips_ref, 1e-9)
            o = {"kappa_ref": kref, "c_trip": c_trip, "trips_ref_sampled_R11": trips_ref}
            for pol in ("tier0", "tier3"):
                cost = np.array([r["cost"] for r in res[pol]]); Eh = np.array([r["E_hat"] for r in res[pol]])
                Er = np.array([r["E_ref"] for r in res[pol]])
                p = float(Eh.mean())
                boots = [rng.choice(len(cost), len(cost)) for _ in range(2000)]
                ecost = [float(cost[b].mean() + Eh[b].mean() * c_trip) for b in boots]
                ecv = [float(cvar(cost[b]) + Eh[b].mean() * c_trip / 0.10) for b in boots]
                o[pol] = {"cost_no_trips": float(cost.mean()), "cvar90_no_trips": cvar(cost),
                          "E_trips_hat": p, "E_trips_ref_formula": float(Er.mean()),
                          "E_cost_hat": float(cost.mean() + p * c_trip), "E_cost_ci95": [float(np.percentile(ecost, 2.5)), float(np.percentile(ecost, 97.5))],
                          "CVaR90_hat": float(cvar(cost) + p * c_trip / 0.10), "CVaR90_ci95": [float(np.percentile(ecv, 2.5)), float(np.percentile(ecv, 97.5))]}
            out[base] = o
            print(base, json.dumps({k: (v if not isinstance(v, dict) else {a: b for a, b in v.items() if a in ("E_trips_hat", "E_trips_ref_formula", "E_cost_hat", "CVaR90_hat")}) for k, v in o.items()}), flush=True)
    json.dump(out, open(os.path.join(RESULTS_DIR, "r35_expected_trips.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
