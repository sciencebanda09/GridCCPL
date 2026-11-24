"""R21 -- direct test of assumption (A) by one-step deviations.

(A): each discretionary excess MWh costs more in future damage than it saves now.
Test: follow the certificate price mu_cert along a trajectory; at sampled steps t, copy the
true simulator and let the operator deviate ONCE to the wear-blind dispatch (mu=0) at step t,
then return to mu_cert.  Delta_t = (remaining cost with deviation) - (remaining cost without).
(A) predicts Delta_t >= 0 whenever the deviation changes dispatch.  We also report the immediate
saving of the deviation (first-step cost difference), and test tau=3 h (the main setting) vs 48 h.

Run: python -m twoclock.experiments.r21_assumption_A
"""
from __future__ import annotations
import copy, json, os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import numpy as np

SEEDS = (1, 2, 3)
STRIDE = 24
JOBS = [("peak", "peak", 1.0, 3.0), ("tight1.15", "normal", 1.15, 3.0),
        ("peak", "peak", 1.0, 48.0), ("tight1.15", "normal", 1.15, 48.0)]


def one(args):
    name, diff, ds, tau = args
    from twoclock.agents import OPFAware, OPFBlind
    from twoclock.envs import make_env
    from twoclock.paths import RESULTS_DIR
    from twoclock.physics import AgingParams, build_5bus
    aging = replace(AgingParams(), tau_thermal=tau)
    net = build_5bus()
    if tau == 3.0:
        base = json.load(open(os.path.join(RESULTS_DIR, "r13_baselines.json")))
        mu = [r for r in base["regimes"] if r["regime"] == name][0]["sel_certificate"]["mu"]
    else:
        mu = [r for r in json.load(open(os.path.join(RESULTS_DIR, "r17_long_memory.json")))
              if r["regime"] == name and r["tau"] == tau][0]["mu_cert"]

    def ag(m):
        return OPFBlind(net, aging) if m <= 0 else OPFAware(net, aging, float(m))

    def rollout(env, first_mu, rest_mu):
        a1, a2 = ag(first_mu), ag(rest_mu)
        o = env._observe()
        c0 = c = 0.0
        k = 0
        while not env.done:
            o, _, _, info = env.step((a1 if k == 0 else a2).act(o))
            c += info["step_cost"]
            if k == 0:
                c0 = info["step_cost"]
            k += 1
        return c, c0

    rows = []
    for s in SEEDS:
        env = make_env(diff, days=10, seed=s, demand_scale=ds, aging=aging)
        env.net = net
        a = ag(mu)
        o = env.reset()
        a.reset(env)
        t = 0
        while not env.done:
            if t >= STRIDE and t % STRIDE == 0 and t < 800:
                dev, dev0 = rollout(copy.deepcopy(env), 0.0, mu)
                ref, ref0 = rollout(copy.deepcopy(env), mu, mu)
                rows.append({"seed": s, "t": t, "delta": dev - ref, "immediate": dev0 - ref0})
            o, _, _, info = env.step(a.act(o))
            t += 1
    d = np.array([r["delta"] for r in rows])
    imm = np.array([r["immediate"] for r in rows])
    act = np.abs(imm) > 1e-6          # deviation changed dispatch at that step
    return {"regime": name, "tau": tau, "mu_cert": mu, "n": len(rows), "n_active": int(act.sum()),
            "n_active_delta_negative": int(((d < -1.0) & act).sum()),
            "min_delta_active": float(d[act].min()) if act.any() else None,
            "mean_delta_active": float(d[act].mean()) if act.any() else None,
            "mean_immediate_active": float(imm[act].mean()) if act.any() else None,
            "rows": rows}


def main():
    from twoclock.paths import RESULTS_DIR
    with ProcessPoolExecutor(max_workers=2) as ex:
        res = list(ex.map(one, JOBS))
    json.dump(res, open(os.path.join(RESULTS_DIR, "r21_assumption_A.json"), "w"), indent=1)
    for r in res:
        print("tau=%4.0f %-9s n=%d active=%d  Delta<0 on %d  mean Delta(active)=%s immediate=%s min=%s" %
              (r["tau"], r["regime"], r["n"], r["n_active"], r["n_active_delta_negative"],
               r["mean_delta_active"], r["mean_immediate_active"], r["min_delta_active"]))


if __name__ == "__main__":
    main()
