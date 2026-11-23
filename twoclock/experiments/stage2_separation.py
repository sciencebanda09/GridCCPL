"""Stage 2 -- reduction.

Claim: the slow-clock problem collapses to one scalar.  Specifically:

  (a) the achievable cost is a *step function* of the price -- there is a
      threshold ``mu_suff`` above which nothing improves;
  (b) a single scalar captures almost all of the benefit of a price that is
      re-optimised every day;
  (c) the price can be wrong by a factor of 4 and still capture >99% of the
      achievable saving.

Together these say the control object is a *price with the right sign and
scale*, not a finely tuned penalty weight.  That is the reduction the paper
needs: it explains why a scalar is enough, and it explains why dual ascent on a
penalty weight diverges -- the set of optimal multipliers is a half-line, so
the multiplier itself is not identifiable even though the policy is.

Run:  python -m twoclock.experiments.stage2_separation
"""
from __future__ import annotations

import os

import numpy as np

from ..agents import OPFAware, OPFBlind, ValueGradientAgent
from ..envs import make_env
from ..physics import AgingParams, build_5bus
from ..paths import RESULTS_DIR
from .runner import Results, run_episode, summarise

MU_GRID = (0.0, 100.0, 300.0, 1000.0, 2000.0, 3000.0, 5000.0,
           10000.0, 20000.0, 40000.0, 100000.0, 300000.0, 1000000.0)
SEEDS = (1, 2, 3, 4, 5)
DAYS = 10


def curve(net, aging, env_name="normal", days=DAYS, seeds=SEEDS):
    """J(mu) and per-day J_d(mu) for the price grid."""
    out = {"mu": [], "cost": [], "ci": [], "unserved": [], "wear": [], "per_day": []}
    for mu in MU_GRID:
        agent = OPFBlind(net, aging) if mu == 0 else OPFAware(net, aging, mu)
        costs, unserved, wear, per_day = [], [], [], []
        for s in seeds:
            env = make_env(env_name, days=days, seed=s)
            r = run_episode(env, agent)
            costs.append(r["episode_cost"])
            unserved.append(r["unserved_mwh"])
            wear.append(r["wear_end_mean"])
            daysplit = env.sub_episodes(1)
            per_day.append([d["cost"] for d in daysplit])
        out["mu"].append(mu)
        out["cost"].append(float(np.mean(costs)))
        out["ci"].append(float(1.96 * np.std(costs, ddof=1) / np.sqrt(len(costs))))
        out["unserved"].append(float(np.mean(unserved)))
        out["wear"].append(float(np.mean(wear)))
        out["per_day"].append(np.mean(np.array(per_day), axis=0).tolist())
    return out


def main() -> int:
    net, aging = build_5bus(), AgingParams()
    res = Results()
    c = curve(net, aging)

    cost = np.array(c["cost"])
    best = cost.min()
    suff_idx = int(np.argmax(cost <= best * 1.005))
    mu_suff = c["mu"][suff_idx]

    # (b) value of a day-by-day optimal price
    per_day = np.array(c["per_day"])            # (n_mu, n_days)
    single = np.array(cost)
    daily = per_day.min(axis=0).sum()
    best_single = single.min()
    daily_gain = (best_single - daily) / max(1e-9, c["cost"][0] - best)

    # (c) price-error tolerance relative to mu_suff
    tol = {}
    for factor in (0.25, 0.5, 1.0, 2.0, 4.0):
        target = mu_suff * factor
        i = int(np.argmin([abs(m - target) for m in c["mu"]]))
        tol[factor] = {
            "mu": c["mu"][i],
            "cost": c["cost"][i],
            "gap_pct": 100.0 * (c["cost"][i] - best) / max(1e-9, c["cost"][0] - best),
        }

    print(f"mu_suff            = {mu_suff:g} $/wear-unit")
    print(f"J(0)               = {c['cost'][0]/1e6:.3f} M$")
    print(f"J_min              = {best/1e6:.3f} M$   ({100*(best/c['cost'][0]-1):+.1f}%)")
    print(f"daily re-pricing   = {daily/1e6:.3f} M$   "
          f"(captures {100*(1-daily_gain):.1f}% of single-price benefit beyond daily optimum)")
    print("price-error tolerance (gap = share of the achievable saving lost):")
    for f, d in tol.items():
        print(f"   mu = {d['mu']:>10.0f} ({f:>4.2f}x)  cost {d['cost']/1e6:.3f} M$  "
              f"gap {d['gap_pct']:6.2f}%")

    payload = {"env_version": "twoclock-env-1", "days": DAYS,
               "seeds": list(SEEDS), "mu_grid": c["mu"], "J": c["cost"],
               "J_ci": c["ci"], "unserved": c["unserved"], "wear": c["wear"],
               "per_day": per_day.tolist(), "mu_suff": mu_suff,
               "best_single": best_single, "daily_reprice": float(daily),
               "tolerance": tol}
    path = res.save("stage2_separation.json", payload)

    with open(os.path.join(RESULTS_DIR, "stage2_separation.md"), "w", encoding="utf-8") as fh:
        fh.write("# Stage 2 -- reduction\n\n")
        fh.write(f"* threshold price mu_suff = {mu_suff:g} $/wear-unit\n")
        fh.write(f"* J(0) = {c['cost'][0]/1e6:.3f} M$, J_min = {best/1e6:.3f} M$ "
                 f"({100*(best/c['cost'][0]-1):+.1f}%)\n\n")
        fh.write("| mu ($/wear-unit) | cost (M$) | ±95% CI | unserved MWh | wear |\n")
        fh.write("|---|---|---|---|---|\n")
        for i, mu in enumerate(c["mu"]):
            fh.write(f"| {mu:,.0f} | {c['cost'][i]/1e6:.3f} | {c['ci'][i]/1e6:.3f} "
                     f"| {c['unserved'][i]:.1f} | {c['wear'][i]:.3f} |\n")
        fh.write("\n## Price-error tolerance\n\n")
        fh.write("| multiple of mu_suff | mu | cost (M$) | share of saving lost |\n")
        fh.write("|---|---|---|---|\n")
        for f, d in tol.items():
            fh.write(f"| {f:g}x | {d['mu']:,.0f} | {d['cost']/1e6:.3f} "
                     f"| {d['gap_pct']:.2f}% |\n")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
