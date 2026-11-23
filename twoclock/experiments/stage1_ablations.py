"""Stage 1 ablations -- which part of the method is doing the work?

Table 2 of the paper.  Everything here is evaluated on the same five held-out
seeds (1..5) and the same ten-day episodes, so the rows are directly comparable.

Rows:
  * degradation-blind OPF   -- no value model, no price (the industry default)
  * analytical 4h price     -- single-step LP with hardcoded myopic mu = 360
  * negative price          -- the right magnitude with the wrong sign
  * tanh MLP value          -- nonlinear V, gradient read out by finite difference
  * CMDP dual ascent        -- primal-dual multiplier update (6 rounds, 8,640 steps)
  * per-unit ridge (4)      -- four separate prices instead of one scalar (4,320 steps)
  * scalar ridge (ours)     -- one price, total thermal damage index (4,320 steps)

Run:  python -m twoclock.experiments.stage1_ablations
"""
from __future__ import annotations

import os

import numpy as np

from ..agents import OPFAware, OPFBlind, ValueGradientAgent
from ..envs import make_env
from ..physics import AgingParams, build_5bus
from ..paths import RESULTS_DIR
from .runner import Results, evaluate, run_episode

TRAIN_SEEDS = (101, 102, 103)
TEST_SEEDS = (1, 2, 3, 4, 5)
DAYS = 10
TRAIN_DAYS = 5
ROUNDS = 3


def train(net, aging, **kw) -> ValueGradientAgent:
    agent = ValueGradientAgent(net, aging, seed=7, mu_max=1e6, **kw)
    agent.enabled = False
    for r in range(ROUNDS):
        for s in TRAIN_SEEDS:
            run_episode(make_env("normal", days=TRAIN_DAYS, seed=s + 10 * r),
                        agent, learn=True)
        agent.update()
        agent.enabled = True
    return agent


def train_dual_ascent(net, aging, target_wear: float = 0.06,
                      eta: float = 25000.0, rounds: int = 6) -> OPFAware:
    """Standard CMDP primal-dual ascent on the slow constraint E[W] <= target_wear."""
    mu = 0.0
    for r in range(rounds):
        w_list = []
        for s in TRAIN_SEEDS:
            ag = OPFBlind(net, aging) if mu <= 0 else OPFAware(net, aging, mu)
            ep = run_episode(make_env("normal", days=TRAIN_DAYS, seed=s + 10 * r), ag)
            w_list.append(ep["wear_end_mean"])
        mu = max(0.0, mu + eta * (float(np.mean(w_list)) - target_wear))
    agent = OPFAware(net, aging, mu)
    agent.learned_mu = float(mu)
    return agent


def main() -> int:
    net, aging = build_5bus(), AgingParams()
    res = Results()
    payload = {"env_version": "twoclock-env-1", "days": DAYS,
               "test_seeds": list(TEST_SEEDS),
               "train_seeds": list(TRAIN_SEEDS), "rows": {}}

    def record(label, agent):
        r = evaluate(agent, "normal", TEST_SEEDS, days=DAYS)
        payload["rows"][label] = r
        print(f"| {label:28s} | {r['episode_cost']['mean']/1e6:6.3f} "
              f"| {r['wear_end_mean']['mean']:5.3f} "
              f"| {r['unserved_mwh']['mean']:7.1f} |")
        return r

    print("| variant                      | cost  | wear  | unserved |")
    record("blind", OPFBlind(net, aging))
    record("myopic_4h_mu360", OPFAware(net, aging, 360.0))
    record("negative_price", OPFAware(net, aging, -1.0e4))
    record("mlp_value", train(net, aging, model="mlp"))
    da_agent = train_dual_ascent(net, aging, target_wear=0.06)
    record("dual_ascent_eps006", da_agent)
    record("per_unit_ridge", train(net, aging, per_unit=True))
    record("scalar_ridge", train(net, aging))

    # Stochastic forced-outage tail-risk evaluation (5 held-out seeds)
    out_seeds = (1, 2, 3, 4, 5)
    b_out, p_out, b_trips, p_trips, b_uns, p_uns = [], [], [], [], [], []
    for s in out_seeds:
        eb = run_episode(make_env("normal", days=DAYS, seed=s, outage_enabled=True),
                         OPFBlind(net, aging))
        ep = run_episode(make_env("normal", days=DAYS, seed=s, outage_enabled=True),
                         OPFAware(net, aging, 1.0e4))
        b_out.append(eb["episode_cost"]); p_out.append(ep["episode_cost"])
        b_trips.append(eb["trips"]); p_trips.append(ep["trips"])
        b_uns.append(eb["unserved_mwh"]); p_uns.append(ep["unserved_mwh"])
    payload["outage_tail_risk"] = {
        "seeds": list(out_seeds),
        "blind_cost_mean": float(np.mean(b_out)),
        "priced_cost_mean": float(np.mean(p_out)),
        "saving_pct": float(100.0 * (1.0 - np.mean(p_out) / np.mean(b_out))),
        "blind_trips_mean": float(np.mean(b_trips)),
        "priced_trips_mean": float(np.mean(p_trips)),
        "blind_unserved_mwh": float(np.mean(b_uns)),
        "priced_unserved_mwh": float(np.mean(p_uns)),
    }
    print(f"\nOutage tail-risk (5 seeds): blind={np.mean(b_out)/1e6:.3f} M$ "
          f"(trips={np.mean(b_trips):.1f}) -> priced={np.mean(p_out)/1e6:.3f} M$ "
          f"(trips={np.mean(p_trips):.1f}), saving={payload['outage_tail_risk']['saving_pct']:.1f}%")

    print("\n| VOLL ($/MWh) | blind (M$) | priced (M$) | saving |")
    payload["voll_sensitivity"] = []
    for voll in (0.0, 2500.0, 5000.0, 10000.0):
        blind_costs, priced_costs = [], []
        for s in TEST_SEEDS:
            env_b = make_env("normal", days=DAYS, seed=s, voll=voll)
            blind_costs.append(run_episode(env_b, OPFBlind(net, aging))["episode_cost"])
            env_p = make_env("normal", days=DAYS, seed=s, voll=voll)
            priced_costs.append(run_episode(env_p, OPFAware(net, aging, 1.0e4))["episode_cost"])
        b_mean = float(np.mean(blind_costs))
        p_mean = float(np.mean(priced_costs))
        sav_pct = 100.0 * (1.0 - p_mean / b_mean)
        payload["voll_sensitivity"].append({
            "voll": voll,
            "blind_cost": b_mean,
            "priced_cost": p_mean,
            "saving_pct": sav_pct,
        })
        print(f"| {voll:12.0f} | {b_mean/1e6:10.3f} | {p_mean/1e6:11.3f} | {sav_pct:5.1f}% |")

    path = res.save("stage1_ablations.json", payload)

    base = payload["rows"]["blind"]["episode_cost"]["mean"]
    n_seeds = {3: "three", 4: "four", 5: "five"}.get(len(TEST_SEEDS), str(len(TEST_SEEDS)))
    with open(os.path.join(RESULTS_DIR, "stage1_ablations.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# Stage 1 -- ablations (normal regime, {n_seeds} held-out seeds)\n\n")
        fh.write("| variant | cost (M$) | vs blind | wear | unserved MWh |\n")
        fh.write("|---|---|---|---|---|\n")
        for label, r in payload["rows"].items():
            c = r["episode_cost"]["mean"]
            fh.write(f"| {label} | {c/1e6:.3f} | {100*(c/base-1):+.1f}% "
                     f"| {r['wear_end_mean']['mean']:.3f} "
                     f"| {r['unserved_mwh']['mean']:.1f} |\n")
        fh.write(f"\n## VOLL sensitivity (normal regime, {n_seeds} held-out seeds)\n\n")
        fh.write("| VOLL ($/MWh) | blind (M$) | priced (M$) | saving |\n")
        fh.write("|---|---|---|---|\n")
        for row in payload["voll_sensitivity"]:
            fh.write(f"| {row['voll']:.0f} | {row['blind_cost']/1e6:.3f} "
                     f"| {row['priced_cost']/1e6:.3f} | {row['saving_pct']:.1f}% |\n")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
