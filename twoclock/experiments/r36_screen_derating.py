"""R36 -- (derived from R32; calibrated-derating variants of R31) does the one-step-deviation screen for (A) predict which random-parameter draws (R23) fail?

For each R23 draw x regime (24 cases) we follow the certificate price on test seed 1 and, at every 32nd
step, let the operator deviate ONCE (a) to wear-blind dispatch (mu = 0) and (b) to a much higher price
(10 x mu_cert), then return to mu_cert, replaying the true future (as R21).  Delta = remaining cost with
deviation minus without.  (A) holds at a step if neither deviation lowers cost.  Screen statistic:
share of sampled steps at which some deviation lowers cost by more than $1.  A case is *flagged* if that
share exceeds 10%.  Outcome (from R23): fail = certificate gap to best constant > 1% (or worse than blind).

Run: python -m twoclock.experiments.r32_A_screen
"""
from __future__ import annotations
import copy, json, os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import numpy as np

STRIDE, SEED = 32, 1


def one(args):
    row = args
    from twoclock.agents import OPFAware, OPFBlind
    from twoclock.envs import make_env
    from twoclock.physics import AgingParams, build_5bus
    name = row["regime"]
    diff, ds = {"peak": ("peak", 1.0), "normal": ("normal", 1.0), "tight1.15": ("normal", 1.15)}[name]
    aging = replace(AgingParams(), **row["params"])
    net = build_5bus()
    mu = row["mu_cert"]

    def ag(m):
        return OPFBlind(net, aging) if m <= 0 else OPFAware(net, aging, float(m))

    def rollout(env, first, rest):
        a1, a2 = ag(first), ag(rest)
        o = env._observe(); c = 0.0; k = 0
        while not env.done:
            o, _, _, info = env.step((a1 if k == 0 else a2).act(o))
            c += info["step_cost"]; k += 1
        return c

    env = make_env(diff, days=10, seed=SEED, demand_scale=ds, aging=aging)
    env.net = net
    a = ag(mu); o = env.reset(); a.reset(env)
    t, rec = 0, []
    while not env.done:
        if t >= STRIDE and t % STRIDE == 0 and t < 800:
            ref = rollout(copy.deepcopy(env), mu, mu)
            dn = rollout(copy.deepcopy(env), 0.0, mu) - ref
            up = rollout(copy.deepcopy(env), 10.0 * mu, mu) - ref
            rec.append((dn, up))
        o, _, _, _ = env.step(a.act(o)); t += 1
    d = np.array(rec)
    share = float(np.mean(np.min(d, axis=1) < -1.0))
    return {"regime": name, "sample": row["sample"], "params": row["params"], "n_steps": len(rec),
            "share_A_violated": share, "share_down_better": float(np.mean(d[:, 0] < -1.0)),
            "share_up_better": float(np.mean(d[:, 1] < -1.0)),
            "gap_pct": row["gap_pct"], "saving_pct": row["saving_pct"]}


def main():
    from twoclock.paths import RESULTS_DIR
    rows = [r for r in json.load(open(os.path.join(RESULTS_DIR, "r31_calibrated_savings.json")))]
    with ProcessPoolExecutor(max_workers=2) as ex:
        res = list(ex.map(one, rows))
    for r, src in zip(res, rows):
        r["variant"] = src["variant"]
        r["flag"] = bool(r["share_A_violated"] > 0.10)
        r["fail"] = bool((r["gap_pct"] > 1.0) or (r["saving_pct"] < 0))
    json.dump(res, open(os.path.join(RESULTS_DIR, "r36_screen_derating.json"), "w"), indent=1)
    for r in res:
        print(r["regime"], r["variant"], "A-viol share %.2f flag %s | gap %.2f%% saving %.1f%%" % (r["share_A_violated"], r["flag"], r["gap_pct"], r["saving_pct"]))


if __name__ == "__main__":
    main()
