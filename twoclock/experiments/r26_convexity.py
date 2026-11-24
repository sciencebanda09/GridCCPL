"""R26 -- is the cost-to-go convex in the wear state, and does heat-rate drift break it?

(A') needs V_{t+1} convex along the wear direction.  V* is not computable, so we use the best-of-grid
upper bound  Vhat(s) = min_{mu in GRID} remaining cost over H = 192 steps (2 days, true simulator, true
future) starting from the state with the wear vector overwritten.  Along lines W(theta) = W0 + theta d
(theta = 0, .08, ..., .32; d = all thermal units, or unit 0 only) we compute second differences; convex
iff all >= -TOL.  We compare the default model (heat-rate drift: heat_lin=.1, heat_quad=.8, drift=.02)
with a variant with the heat-rate channel switched off (heat_lin = heat_quad = drift = 0).
States: every 96th step of a certificate-price trajectory, test seeds 1-3, regimes peak, tight1.15.

Run: python -m twoclock.experiments.r26_convexity
"""
from __future__ import annotations
import copy, json, os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import numpy as np

GRID = [0.0, 1e4, 3e4, 1e5, 3e5]
H = 192
THETAS = [0.0, 0.08, 0.16, 0.24, 0.32]
DIRS = {"all": np.array([1.0, 1.0, 1.0, 0.0]), "gas": np.array([1.0, 0.0, 0.0, 0.0])}
TOL = 1.0
REG = [("peak", "peak", 1.0), ("tight1.15", "normal", 1.15)]
SEEDS = (1, 2, 3)


def one(args):
    name, diff, ds, drift_on, seed = args
    from twoclock.agents import OPFAware, OPFBlind
    from twoclock.envs import make_env
    from twoclock.paths import RESULTS_DIR
    from twoclock.physics import AgingParams, build_5bus
    aging = AgingParams() if drift_on else replace(AgingParams(), heat_lin=0.0, heat_quad=0.0, drift=0.0)
    net = build_5bus()
    base = json.load(open(os.path.join(RESULTS_DIR, "r13_baselines.json")))
    mu_c = [r for r in base["regimes"] if r["regime"] == name][0]["sel_certificate"]["mu"]

    def ag(m):
        return OPFBlind(net, aging) if m <= 0 else OPFAware(net, aging, float(m))

    def rem_cost(env0, W):
        best = np.inf
        for m in GRID:
            e = copy.deepcopy(env0)
            e.aging.wear = np.clip(np.asarray(W, float), 0.0, 1.0)
            a = ag(m)
            o = e._observe()
            c = 0.0
            for _ in range(H):
                if e.done:
                    break
                o, _, _, info = e.step(a.act(o))
                c += info["step_cost"]
            best = min(best, c)
        return best

    env = make_env(diff, days=10, seed=seed, demand_scale=ds, aging=aging)
    env.net = net
    a = ag(mu_c)
    o = env.reset(); a.reset(env)
    t, lines = 0, []
    while not env.done and t < 700:
        if t >= 96 and t % 96 == 0:
            W0 = env.aging.wear.copy()
            for dn, d in DIRS.items():
                V = [rem_cost(env, W0 + th * d) for th in THETAS]
                sd = [V[k - 1] - 2 * V[k] + V[k + 1] for k in range(1, len(V) - 1)]
                lines.append({"t": t, "dir": dn, "W0": W0.tolist(), "V": V, "second_diff": sd,
                              "convex": bool(min(sd) >= -TOL)})
        o, _, _, info = env.step(a.act(o))
        t += 1
    return {"regime": name, "drift_on": drift_on, "seed": seed, "lines": lines}


def main():
    from twoclock.paths import RESULTS_DIR
    jobs = [(n, d, s, on, sd) for on in (True, False) for (n, d, s) in REG for sd in SEEDS]
    with ProcessPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(one, jobs))
    json.dump(res, open(os.path.join(RESULTS_DIR, "r26_convexity.json"), "w"), indent=1)
    for on in (True, False):
        for reg in ("peak", "tight1.15"):
            L = [ln for r in res if r["drift_on"] == on and r["regime"] == reg for ln in r["lines"]]
            sd = np.array([min(l["second_diff"]) for l in L])
            rel = np.array([min(l["second_diff"]) / max(abs(l["V"][0]), 1.0) for l in L])
            print("heat-rate drift %-5s %-9s lines=%d  convex=%d (%.0f%%)  worst 2nd diff=%.0f $ (%.4f%% of V)" %
                  (on, reg, len(L), sum(l["convex"] for l in L), 100 * np.mean([l["convex"] for l in L]),
                   sd.min(), 100 * rel.min()))


if __name__ == "__main__":
    main()
