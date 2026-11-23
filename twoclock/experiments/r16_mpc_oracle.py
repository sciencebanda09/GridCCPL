"""R16 -- receding-horizon MPC oracle for the wear price.

Every K=24 steps (6 h) the controller deep-copies the TRUE simulator state, rolls it
forward H=192 steps (2 days) under perfect foresight (the env's exogenous draws are
policy-independent, so the copy replays the true future) for each mu on a price grid,
and applies the cheapest mu for the next K steps.  This is the strongest
state-dependent price controller available in the price family (exact model, perfect
foresight, re-optimised every 6 h); it is NOT implementable and is used as an upper
bound on what state dependence could add over one scalar certificate price.

Run: python -m twoclock.experiments.r16_mpc_oracle
"""
from __future__ import annotations
import copy, json, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy import stats

SEEDS = tuple(range(1, 7))
K, H = 24, 192
GRID = [0.0, 3e3, 6e3, 1e4, 2e4, 5e4, 1e5, 3e5]
REG = [("normal", "normal", 1.0), ("hard", "hard", 1.0), ("peak", "peak", 1.0),
       ("storm", "storm", 1.0), ("tight1.15", "normal", 1.15),
       ("tight1.25", "normal", 1.25), ("tight1.35", "normal", 1.35)]


def ci95(x):
    x = np.asarray(x, float)
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def one(args):
    name, diff, ds = args
    from twoclock.agents import OPFAware, OPFBlind
    from twoclock.agents.baselines import BaseAgent
    from twoclock.experiments import r6_certificate as R
    from twoclock.experiments.runner import run_episode
    from twoclock.paths import RESULTS_DIR
    from twoclock.physics import AgingParams
    aging = AgingParams()
    net = R.set_network("5bus")
    ctx = R.Ctx(net, diff, ds, aging)
    base = json.load(open(os.path.join(RESULTS_DIR, "r13_baselines.json")))
    row = [r for r in base["regimes"] if r["regime"] == name][0]
    mu_cert = row["sel_certificate"]["mu"]
    mu_lo = row["sel_certificate"]["mu_lo"]

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
    return {"regime": name, "seeds": list(SEEDS), "K": K, "H": H, "grid": GRID,
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
        rows = list(ex.map(one, REG))
    with open(os.path.join(RESULTS_DIR, "r16_mpc_oracle.json"), "w") as fh:
        json.dump(rows, fh, indent=1, default=float)
    for r in rows:
        print(r["regime"], "cert %.4fM mpc %.4fM  d=%+.1fk+-%.1fk  wins %d/6  extra %.2f%%"
              % (r["cert"] / 1e6, r["mpc"] / 1e6, r["mpc_minus_cert_k"],
                 r["mpc_minus_cert_ci_k"], r["mpc_wins_seeds"], r["frac_saving_extra_pct"]),
              r["chosen_mu_counts"])


if __name__ == "__main__":
    main()
