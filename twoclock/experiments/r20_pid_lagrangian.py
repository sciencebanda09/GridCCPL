"""R20 -- PID-Lagrangian (Stooke et al. 2020) vs plain dual ascent vs the certificate.

The constrained-MDP view tunes mu so that mean end-of-episode wear <= eps.  We run
(i) integral-only dual ascent, (ii) a PID multiplier update, each for 30 rounds of
5 training episodes (5-day, seeds >= 101), with two targets:
  eps_fixed = 0.06  (the target used in the main CMDP baseline)
  eps_cert  = mean wear reached at the certificate price (an oracle-tuned target)
and evaluate the final mu on test seeds 1-5 (10-day).  Reports mu trajectory, final mu,
cost relative to the certificate price.

Run: python -m twoclock.experiments.r20_pid_lagrangian
"""
from __future__ import annotations
import json, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np

TRAIN = (101, 102, 103, 104, 105)
TEST = (1, 2, 3, 4, 5)
ROUNDS = 30
REG = [("normal", "normal", 1.0), ("tight1.15", "normal", 1.15)]


def one(args):
    name, diff, ds = args
    from twoclock.agents import OPFAware, OPFBlind
    from twoclock.envs import make_env
    from twoclock.experiments.runner import run_episode
    from twoclock.paths import RESULTS_DIR
    from twoclock.physics import AgingParams, build_5bus
    aging, net = AgingParams(), build_5bus()
    base = json.load(open(os.path.join(RESULTS_DIR, "r13_baselines.json")))
    row = [r for r in base["regimes"] if r["regime"] == name][0]
    mu_cert, mu_lo = row["sel_certificate"]["mu"], row["sel_certificate"]["mu_lo"]

    def agent(mu):
        return OPFBlind(net, aging) if mu <= 0 else OPFAware(net, aging, float(mu))

    def run(mu, seeds, days):
        out = []
        for s in seeds:
            e = make_env(diff, days=days, seed=s, demand_scale=ds)
            e.net = net
            out.append(run_episode(e, agent(mu)))
        return out

    wear_cert = float(np.mean([e["wear_end_mean"] for e in run(mu_cert, TRAIN, 5)]))
    wear_blind = float(np.mean([e["wear_end_mean"] for e in run(0.0, TRAIN, 5)]))
    cost_cert = float(np.mean([e["episode_cost"] for e in run(mu_cert, TEST, 10)]))

    def train(kind, eps, kp, ki, kd):
        mu, integ, prev, traj = 0.0, 0.0, 0.0, []
        for r in range(ROUNDS):
            w = float(np.mean([e["wear_end_mean"] for e in run(mu, TRAIN, 5)]))
            err = w - eps
            integ = max(0.0, integ + ki * err)          # anti-windup at 0
            if kind == "dual":
                mu = integ
            else:
                mu = max(0.0, kp * err + integ + kd * max(0.0, err - prev))
            prev = err
            traj.append(float(mu))
        return mu, traj

    res = {"regime": name, "mu_cert": mu_cert, "mu_lo": mu_lo, "wear_cert": wear_cert,
           "wear_blind": wear_blind, "cost_cert": cost_cert, "variants": {}}
    for tgt, eps in (("eps0.06", 0.06), ("eps_cert", wear_cert)):
        for kind, (kp, ki, kd) in (("dual", (0, 25000.0, 0)), ("pid", (1.5e5, 25000.0, 1.5e5))):
            mu, traj = train(kind, eps, kp, ki, kd)
            c = float(np.mean([e["episode_cost"] for e in run(mu, TEST, 10)]))
            res["variants"][f"{kind}_{tgt}"] = {
                "eps": eps, "mu_final": float(mu), "mu_traj": traj,
                "cost": c, "cost_vs_cert_pct": 100 * (c / cost_cert - 1),
                "in_band_or_above": bool(mu >= mu_lo * 0.9)}
    return res


def main():
    from twoclock.paths import RESULTS_DIR
    with ProcessPoolExecutor(max_workers=2) as ex:
        rows = list(ex.map(one, REG))
    json.dump(rows, open(os.path.join(RESULTS_DIR, "r20_pid_lagrangian.json"), "w"), indent=1)
    for r in rows:
        print(r["regime"], "cert mu %.0f lo %.0f wear_cert %.4f wear_blind %.4f" %
              (r["mu_cert"], r["mu_lo"], r["wear_cert"], r["wear_blind"]))
        for k, v in r["variants"].items():
            print("  %-16s mu=%9.0f  cost vs cert %+.2f%%  (traj tail %s)" %
                  (k, v["mu_final"], v["cost_vs_cert_pct"],
                   [round(x) for x in v["mu_traj"][-4:]]))


if __name__ == "__main__":
    main()
