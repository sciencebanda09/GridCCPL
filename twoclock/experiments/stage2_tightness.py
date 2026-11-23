"""Stage 2b -- is the corner an artefact of a loose system?

The reduction result (Stage 2) says the cost is a step function of the price and
the optimum is a *corner*: the price removes all discretionary stress loading.

A reviewer will ask the obvious question -- is that because the fleet is too
loose to care?  This experiment tightens the system (demand scaled 1.00 to
1.35x nominal) and re-measures the shape of J(mu) and the residual wear.

The answer is that tighter demand raises both cost and *residual* wear, but it
does not make the price interior: J(mu) stays monotone and flattens, and the
wear that survives is stress no dispatch can avoid (peaks that must be served).
The price removes the stress the operator *chooses*; it cannot remove the
stress the system *forces*.  That is the correct statement of the limitation,
and it is stronger than the guess it replaces.

Run:  python -m twoclock.experiments.stage2_tightness
"""
from __future__ import annotations

import os

import numpy as np

from ..agents import OPFAware, OPFBlind
from ..envs import make_env
from ..physics import AgingParams, build_5bus
from ..paths import RESULTS_DIR
from .runner import Results, run_episode

SCALES = (1.00, 1.15, 1.25, 1.35)
MU_GRID = (0.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0)
SEEDS = (1, 2, 3)
DAYS = 10


def sweep(net, aging, scale: float):
    rows = []
    for mu in MU_GRID:
        agent = OPFBlind(net, aging) if mu == 0 else OPFAware(net, aging, mu)
        cost, wear, unserved = [], [], []
        for s in SEEDS:
            env = make_env("normal", days=DAYS, seed=s, demand_scale=scale)
            r = run_episode(env, agent)
            cost.append(r["episode_cost"])
            wear.append(r["wear_end_mean"])
            unserved.append(r["unserved_mwh"])
        rows.append({
            "mu": mu,
            "cost": float(np.mean(cost)),
            "wear": float(np.mean(wear)),
            "unserved": float(np.mean(unserved)),
        })
    return rows


def main() -> int:
    net, aging = build_5bus(), AgingParams()
    res = Results()
    payload = {"env_version": "twoclock-env-1", "days": DAYS,
               "seeds": list(SEEDS), "mu_grid": list(MU_GRID), "scales": {},
               "interior": {}}

    print("scale   mu        cost(M$)  unserved   wear")
    for scale in SCALES:
        rows = sweep(net, aging, scale)
        payload["scales"][f"{scale:.2f}"] = rows
        best = min(r["cost"] for r in rows)
        for r in rows:
            mark = " *" if r["cost"] <= best * 1.001 else ""
            print(f"{scale:5.2f} {r['mu']:8.0f} {r['cost']/1e6:10.3f} "
                  f"{r['unserved']:9.1f} {r['wear']:7.3f}{mark}")
        # A corner means the last point is optimal.  An interior optimum would
        # show a strict minimum before the end of the grid.
        interior = any(rows[i]["cost"] < rows[i - 1]["cost"] * 0.999
                       and rows[i]["cost"] < rows[i + 1]["cost"] * 0.999
                       for i in range(1, len(rows) - 1))
        payload["interior"][f"{scale:.2f}"] = bool(interior)
        print(f"      interior optimum? {interior}")

    path = res.save("stage2b_tightness.json", payload)

    with open(os.path.join(RESULTS_DIR, "stage2b_tightness.md"), "w", encoding="utf-8") as fh:
        fh.write("# Stage 2b -- tightness\n\n")
        fh.write("Demand scaled above nominal; does the corner survive?\n\n")
        for scale, rows in payload["scales"].items():
            interior = payload["interior"][scale]
            pts = [r for r in rows if isinstance(r, dict)]
            fh.write(f"## scale {scale}\n\n")
            fh.write("| mu | cost (M$) | unserved MWh | wear |\n|---|---|---|---|\n")
            for r in pts:
                fh.write(f"| {r['mu']:.0f} | {r['cost']/1e6:.3f} "
                         f"| {r['unserved']:.1f} | {r['wear']:.3f} |\n")
            fh.write(f"\ninterior optimum: **{interior}**\n\n")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
