"""R19 -- the R16 receding-horizon oracle repeated at long memory (tau 24/48 h) on the regimes
where R17 found a gap (peak, tight1.15), test seeds 1-5, 10-day episodes.

Original R16 description: receding-horizon MPC oracle for the wear price.

Every K=24 steps (6 h) the controller deep-copies the TRUE simulator state, rolls it
forward H=192 steps (2 days) under perfect foresight (the env's exogenous draws are
policy-independent, so the copy replays the true future) for each mu on a price grid,
and applies the cheapest mu for the next K steps.  This is the strongest
state-dependent price controller available in the price family (exact model, perfect
foresight, re-optimised every 6 h); it is NOT implementable and is used as an upper
bound on what state dependence could add over one scalar certificate price.

Run: python -m twoclock.experiments.r19_mpc_longmemory
"""
from __future__ import annotations
import copy, json, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy import stats

SEEDS = (1, 2, 3, 4, 5)
TAUS = (24.0, 48.0)
K, H = 24, 192
GRID = [0.0, 3e3, 6e3, 1e4, 2e4, 5e4, 1e5, 3e5]
REG = [("peak", "peak", 1.0), ("tight1.15", "normal", 1.15)]


def ci95(x):
    x = np.asarray(x, float)
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def one(args):
    name, diff, ds, tau = args
    from dataclasses import replace
    from twoclock.envs import make_env
    from twoclock.agents import OPFAware, OPFBlind
    from twoclock.agents.baselines import BaseAgent
    from twoclock.experiments import r6_certificate as R
    from twoclock.experiments.runner import run_episode
    from twoclock.paths import RESULTS_DIR
    from twoclock.physics import AgingParams
    aging = replace(AgingParams(), tau_thermal=tau)
    net = R.set_network("5bus")
    rr = [r for r in json.load(open(os.path.join(RESULTS_DIR, "r17_long_memory.json")))
          if r["regime"] == name and r["tau"] == tau][0]
    mu_cert, mu_lo, best_const = rr["mu_cert"], rr["mu_lo"], rr["oracle_best"]

    class _C:
        def env(self, seed):
            e = make_env(diff, days=10, seed=seed, demand_scale=ds, aging=aging)
            e.net = net
            return e
    ctx = _C()

    def make_agent(mu):
        return OPFBlind(net, aging) if mu <= 0 else OPFAware(net, aging, float(mu))

    class MPC(BaseAgent):
        name = "mpc_oracle"

        def __init__(self):
            self.env = None
            self.t = 0
            self.cur = make_agent(0.0)
            self.chosen = []

        def reset(self, env=None):
            self.env = env
            self.t = 0
            self.chosen = []

        def act(self, obs):
            if self.t % K == 0:
                best, best_mu = None, None
                for mu in GRID:
                    e2 = copy.deepcopy(self.env)
                    ag = make_agent(mu)
                    o2 = e2._observe()
                    c = 0.0
                    for _ in range(H):
                        if e2.done:
                            break
                        o2, _, _, info = e2.step(ag.act(o2))
                        c += info["step_cost"]
                    if best is None or c < best - 1e-9:
                        best, best_mu = c, mu
                self.cur = make_agent(best_mu)
                self.chosen.append(best_mu)
            self.t += 1
            return self.cur.act(obs)

    cert_c, blind_c, mpc_c, chosen = [], [], [], []
    for s in SEEDS:
        cert_c.append(float(run_episode(ctx.env(s), make_agent(mu_cert))["episode_cost"]))
        blind_c.append(float(run_episode(ctx.env(s), make_agent(0.0))["episode_cost"]))
        m = MPC()
        mpc_c.append(float(run_episode(ctx.env(s), m)["episode_cost"]))
        chosen.append(m.chosen)
    cert_c, blind_c, mpc_c = map(np.array, (cert_c, blind_c, mpc_c))
    flat = [x for c in chosen for x in c]
    d = mpc_c - cert_c
    return {"tau": tau, "best_const_cost_r17": best_const, "regime": name, "seeds": list(SEEDS), "K": K, "H": H, "grid": GRID,
            "mu_cert": mu_cert, "mu_lo": mu_lo,
            "blind": float(blind_c.mean()), "cert": float(cert_c.mean()),
            "mpc": float(mpc_c.mean()),
            "mpc_minus_cert_k": float(d.mean() / 1e3), "mpc_minus_cert_ci_k": ci95(d) / 1e3,
            "mpc_wins_seeds": int((d < 0).sum()),
            "frac_saving_extra_pct": float(100 * (cert_c.mean() - mpc_c.mean())
                                           / (blind_c.mean() - cert_c.mean())),
            "chosen_mu_counts": {str(g): int(sum(1 for x in flat if x == g)) for g in GRID},
            "per_seed": {"cert": cert_c.tolist(), "mpc": mpc_c.tolist(),
                         "blind": blind_c.tolist()}}


def main():
    from twoclock.paths import RESULTS_DIR
    with ProcessPoolExecutor(max_workers=4) as ex:
        rows = list(ex.map(one, [(n, d, x, t) for t in TAUS for (n, d, x) in REG]))
    with open(os.path.join(RESULTS_DIR, "r19_mpc_longmemory.json"), "w") as fh:
        json.dump(rows, fh, indent=1, default=float)
    for r in rows:
        print(r["tau"], r["regime"], "cert %.4fM mpc %.4fM  d=%+.1fk+-%.1fk  wins %d/5  extra %.2f%%"
              % (r["cert"] / 1e6, r["mpc"] / 1e6, r["mpc_minus_cert_k"],
                 r["mpc_minus_cert_ci_k"], r["mpc_wins_seeds"], r["frac_saving_extra_pct"]),
              r["chosen_mu_counts"])


if __name__ == "__main__":
    main()
