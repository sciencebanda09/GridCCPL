"""R3 refinement: dense log-spaced constant-mu grid (train seeds only), then
evaluate the refined best constant on the TEST seeds (rows kept so paired
tests against r3_interior.json test rows can be computed; CRN => seeds align).

Run:  python -m twoclock.experiments.r3_refine REGIME [REGIME ...]
Writes results/r3_refine_<REGIME>.json per regime.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from ..physics import AgingParams, build_5bus
from ..paths import RESULTS_DIR
from .r3_interior import REGIMES, TEST, TRAIN, RuleAgent, evaluate

_g = [float(x) for x in os.environ.get("R3_GRID", "3e3,1e5,13").split(",")]
GRID = [float(f"{m:.4g}") for m in np.geomspace(_g[0], _g[1], int(_g[2]))]
SUFFIX = os.environ.get("R3_SUFFIX", "")
T0 = time.time()


def main(argv=None) -> int:
    names = sys.argv[1:] if argv is None else argv
    net = build_5bus()
    for name in names:
        reg = REGIMES[name]
        aging = reg.get("aging", AgingParams())
        train = TRAIN if reg["days"] <= 10 else TRAIN[:2]
        sweep = {}
        for mu in GRID:
            e = evaluate(reg, RuleAgent(net, aging, "const", mu0=mu), train)
            sweep[mu] = e
            print(f"[{time.time()-T0:7.1f}s] {name} mu={mu:g} J={e['mean']/1e6:.4f}", flush=True)
        best = min(GRID, key=lambda m: sweep[m]["mean"])
        jb = sweep[best]["mean"]
        plateau = [m for m in GRID if sweep[m]["mean"] <= jb * (1 + 1e-6)]
        test = evaluate(reg, RuleAgent(net, aging, "const", mu0=best), TEST)
        out = {"regime": name, "train_seeds": list(train), "grid": GRID,
               "train_sweep": {f"{m:g}": {
                   "total": sweep[m]["mean"],
                   "unserved": float(np.mean([r["unserved_mwh"] for r in sweep[m]["rows"]])),
                   "excess_mwh": float(np.mean([r["excess_mwh"] for r in sweep[m]["rows"]])),
                   "wear_end_sum": float(np.mean([r["wear_end_sum"] for r in sweep[m]["rows"]]))}
                   for m in GRID},
               "best_mu": best, "best_train_total": jb,
               "plateau_mu": [min(plateau), max(plateau)],
               "test_best_const": test}
        with open(os.path.join(RESULTS_DIR, f"r3_refine_{name}{SUFFIX}.json"), "w") as fh:
            json.dump(out, fh, indent=1, default=float)
        print(f"[{time.time()-T0:7.1f}s] {name} DONE best={best:g} plateau={plateau[0]:g}..{plateau[-1]:g} "
              f"test={test['mean']/1e6:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
