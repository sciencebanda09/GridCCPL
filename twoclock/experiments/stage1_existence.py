"""Stage 1 -- existence.

Claim: a slow-clock shadow price, read off a learned value function, beats the
degradation-blind OPF that industry actually runs, on the quantity the industry
actually pays for.

Protocol
--------
* Train the price on *training* seeds (100+), never on the evaluation seeds.
* One scalar price is calibrated by sweep; the value-gradient agent calibrates
  itself.  Both are evaluated on 4 exogenous regimes x 5 held-out seeds.
* Metrics are domain-native: total cost, unserved energy, end-of-episode wear
  and heat-rate multiplier.  Cost includes fuel, imports, VOLL, line overload
  and reserve shortfall -- but *not* any penalty we invented for wear, so a
  cheaper episode is genuinely cheaper.

Run:  python -m twoclock.experiments.stage1_existence
"""
from __future__ import annotations

import os

import sys

import numpy as np

from ..agents import OPFBlind, OPFAware, RandomAgent, RuleBase, ValueGradientAgent
from ..envs import make_env
from ..physics import AgingParams, build_5bus
from ..paths import RESULTS_DIR
from .runner import Results, evaluate, fmt_row, header, paired_delta, run_episode

ENVS = ("normal", "hard", "peak", "storm")
TEST_SEEDS = (1, 2, 3, 4, 5)
TRAIN_SEEDS = (101, 102, 103)
DAYS = 10


def calibrate_mu(net, aging, env_name="normal", days=DAYS, seeds=(101, 102),
                 grid=(0.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0)) -> tuple[float, dict]:
    """Cheapest constant price on training seeds.  Returns (mu*, curve)."""
    curve = {}
    for mu in grid:
        agent = OPFBlind(net, aging) if mu == 0 else OPFAware(net, aging, mu)
        r = evaluate(agent, env_name, seeds, days=days)
        curve[mu] = r["episode_cost"]["mean"]
    best = min(curve, key=curve.get)
    return float(best), curve


def train_value_agent(net, aging, env_name="normal", days=DAYS,
                      seeds=TRAIN_SEEDS, rounds=3) -> ValueGradientAgent:
    """Approximate policy iteration: evaluate, price, re-dispatch."""
    agent = ValueGradientAgent(net, aging, seed=7)
    agent.enabled = False            # round 0: degradation-blind exploration
    for r in range(rounds):
        for s in seeds:
            run_episode(make_env(env_name, days=days, seed=s + 100 * r), agent, learn=True)
        agent.update()
        agent.enabled = True
        print(f"    round {r + 1}: mu = {np.round(agent.mu, 1)} $/wear-unit")
    return agent


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    net, aging = build_5bus(), AgingParams()
    res = Results()

    print("== calibrating a constant price on training seeds ==")
    mu_star, curve = calibrate_mu(net, aging)
    print("   J(mu) curve:", {int(k): round(v / 1e6, 3) for k, v in curve.items()},
          "-> mu* =", mu_star)

    print("== training the value-gradient price (held-out seeds only below) ==")
    vg = train_value_agent(net, aging)
    print("   learned mu:", np.round(vg.mu, 1), "$/wear-unit  =>  mu_scalar",
          round(vg.mu_scalar, 1))

    payload = {"env_version": "twoclock-env-1", "days": DAYS,
               "test_seeds": list(TEST_SEEDS),
               "mu_star": mu_star, "mu_curve": curve,
               "vg_mu": vg.mu.tolist(), "envs": {}}

    blind = OPFBlind(net, aging)
    agents = [
        ("random", lambda: RandomAgent(0)),
        ("rule", lambda: RuleBase(0)),
        ("opf_blind", lambda: OPFBlind(net, aging)),
        (f"opf_mu*={mu_star:g}", lambda: OPFAware(net, aging, mu_star)),
        ("vg_learned", lambda: vg),
    ]

    for env_name in ENVS:
        print(f"\n## {env_name}  ({DAYS} days, seeds {TEST_SEEDS})")
        print(header())
        base = evaluate(blind, env_name, TEST_SEEDS, days=DAYS)
        payload["envs"][env_name] = {"opf_blind": base}
        for label, ctor in agents:
            r = evaluate(ctor(), env_name, TEST_SEEDS, days=DAYS)
            payload["envs"][env_name][label] = r
            print(fmt_row(label, r, base if label != "opf_blind" else None))
        d = paired_delta(payload["envs"][env_name]["vg_learned"], base)
        payload["envs"][env_name]["vg_vs_blind"] = d
        print(f"   paired saving vs blind: "
              f"{100 * -d['episode_cost']['rel']:.1f}% ± "
              f"{100 * d['episode_cost']['ci95'] / max(1e-9, base['episode_cost']['mean']):.1f}%")

    path = res.save("stage1_existence.json", payload)
    with open(os.path.join(RESULTS_DIR, "stage1_existence.md"), "w", encoding="utf-8") as fh:
        fh.write("# Stage 1 -- existence\n\n")
        fh.write(f"* constant price mu* = {mu_star:g} $/wear-unit\n")
        fh.write(f"* learned price mu = {np.round(vg.mu, 1).tolist()} $/wear-unit\n")
        fh.write(f"* {DAYS} days, seeds {list(TEST_SEEDS)}\n\n")
        for env_name, d in payload["envs"].items():
            fh.write(f"## {env_name}\n\n")
            fh.write(header() + "\n")
            base = d["opf_blind"]
            for label, r in d.items():
                if label in ("opf_blind", "vg_vs_blind"):
                    continue
                fh.write(fmt_row(label, r, base) + "\n")
            rel = d["vg_vs_blind"]["episode_cost"]["rel"]
            fh.write(f"\nvg vs blind: {100 * -rel:+.2f}% paired\n\n")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
