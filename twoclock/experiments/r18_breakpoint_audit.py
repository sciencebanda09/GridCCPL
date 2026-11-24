"""R18 -- complete breakpoint audit of the price band (addresses substitutable units).

Proposition 1(iv) builds the flat band from single-unit quantities (LP0 lower edge,
crossings and VOLL ceilings of indispensable units).  A reviewer's objection is that
substitutable units (each can replace the other's shedding) are not "indispensable", so
dispatch might change inside the claimed band or shedding might start below the claimed
ceiling.  Here we sweep mu densely on harvested fast-clock states, locate EVERY dispatch
breakpoint (any change in excess e or shedding), and test
  (a) no breakpoint lies strictly inside the claimed per-step band (L_s, U_s);
  (b) the first shedding onset equals the claimed ceiling min_g mu_g^hi (indispensable g),
      or no shedding occurs at any mu <= MU_MAX when no unit is indispensable;
  (c) how many states have a 'joint' shedding threshold the single-unit formula misses.

Run from the repository root:  python -m twoclock.experiments.r18_breakpoint_audit
"""
from __future__ import annotations

import json
import os

import numpy as np

import twoclock.physics.opf as opf_mod
from twoclock.experiments.r10_verify import (MU_MAX, TOL_E, harvest, make_solver)
from twoclock.paths import RESULTS_DIR
from twoclock.physics import AgingParams, build_5bus

OUT = os.path.join(RESULTS_DIR, "r18_breakpoint_audit.json")
REL = 0.05   # grid spacing is ~3.5% in mu; tolerance for edge comparisons


def main():
    net, aging = build_5bus(), AgingParams()
    c = np.asarray(opf_mod.GEN_COST, float)
    VOLL = float(opf_mod.VOLL)
    states = harvest(net, aging)
    grid = np.concatenate([[0.0], np.logspace(0, np.log10(MU_MAX), 400)])
    agg = dict(n_states=0, n_with_breakpoint_in_band=0, n_indispensable_states=0,
               n_shed_onset_checked=0, n_shed_onset_mismatch=0, n_joint_threshold=0,
               n_states_no_shedding_any_mu=0, max_rel_ceiling_err=0.0)
    rows = []
    for st in states:
        solve, alpha = make_solver(net, aging, st["ctx"])
        rs = [solve(m) for m in grid]
        if any(r["status"] != "ok" for r in rs):
            continue
        agg["n_states"] += 1
        E = np.array([r["excess"][:3] for r in rs])
        S = np.array([r["shed"] for r in rs])
        bp = np.where((np.abs(np.diff(E, axis=0)).max(1) > 1e-6) | (np.abs(np.diff(S)) > 1e-6))[0]
        bp_mu = [float(np.sqrt(max(grid[i], 1e-12) * grid[i + 1])) for i in bp]
        # claimed band
        elo, ehi = rs[0]["e_lo"][:3], rs[0]["e_hi"][:3]
        r0 = solve(0.0, block=np.array([1, 1, 1, 0], bool))
        head = [g for g in range(3) if ehi[g] - elo[g] > 1e-6 and alpha[g] > 0]
        indisp = []
        for g in head:
            b = np.zeros(4, bool); b[g] = True
            if solve(0.0, block=b)["shed"] > TOL_E:
                indisp.append(g)
        if r0["status"] == "ok" and r0["shed"] <= TOL_E:
            lmp0 = r0["lmp"][opf_mod.GEN_BUS[:3]] + r0["y_res"]
            L = max([max(0.0, (lmp0[g] - c[g]) / alpha[g]) for g in head], default=0.0)
        else:
            L = 0.0
        for a in indisp:
            for b_ in indisp:
                if alpha[b_] > alpha[a]:
                    L = max(L, (c[a] - c[b_]) / (alpha[b_] - alpha[a]))
        hi_g = [(VOLL - c[g]) / alpha[g] for g in indisp]
        U = min(hi_g) if hi_g else np.inf
        inside = [m for m in bp_mu if L * (1 + REL) < m < U * (1 - REL)]
        if inside:
            agg["n_with_breakpoint_in_band"] += 1
        if indisp:
            agg["n_indispensable_states"] += 1
        shed_idx = np.where(S > TOL_E)[0]
        onset = float(grid[shed_idx[0]]) if len(shed_idx) else np.inf
        if not np.isfinite(onset):
            agg["n_states_no_shedding_any_mu"] += 1
        # shedding onset only meaningful if LP0 itself does not shed
        if r0["status"] == "ok" and r0["shed"] <= TOL_E:
            agg["n_shed_onset_checked"] += 1
            if np.isfinite(U):
                err = abs(onset - U) / U
                agg["max_rel_ceiling_err"] = max(agg["max_rel_ceiling_err"], float(err))
                if err > REL + 0.04:
                    agg["n_shed_onset_mismatch"] += 1
            elif np.isfinite(onset):
                agg["n_joint_threshold"] += 1   # shedding starts although no unit indispensable
        rows.append(dict(regime=st["regime"], step=st["step"], L=L, U=None if not np.isfinite(U) else U,
                         indispensable=indisp, shed_onset=None if not np.isfinite(onset) else onset,
                         n_breakpoints=len(bp_mu), breakpoints_in_band=inside))
    json.dump({"experiment": "r18_breakpoint_audit", "aggregate": agg, "states": rows},
              open(OUT, "w"), indent=1)
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
