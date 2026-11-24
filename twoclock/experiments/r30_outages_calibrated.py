"""R30 -- outage tiers at the CALIBRATED hazard coefficient (kappa = 0.0086/day from R28), instead of the
synthetic kappa in {0.25..6}.  Same protocol as R11 (20 training seeds, 40 held-out seeds, exact LP).

Run: python -m twoclock.experiments.r30_outages_calibrated
"""
from __future__ import annotations
import json, os
from concurrent.futures import ProcessPoolExecutor
from twoclock.experiments import r11_risk_outages as R
from twoclock.paths import RESULTS_DIR

KAPPA = 0.0086
REG = [
    {"name": "tight_1.15_k%.4f" % KAPPA, "base": "tight_1.15", "scale": 1.15, "heatwave": False, "kappa": KAPPA, "curve": False},
    {"name": "heatwave_1.22_k%.4f" % KAPPA, "base": "heatwave_1.22", "scale": 1.00, "heatwave": True, "kappa": KAPPA, "curve": False},
    {"name": "tight_1.25_k%.4f" % KAPPA, "base": "tight_1.25", "scale": 1.25, "heatwave": False, "kappa": KAPPA, "curve": False},
]


def main():
    certs = R.compute_base_certificates()
    out = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for reg in REG:
            out.append(R.evaluate_regime(pool, reg, certs[reg["base"]]))
    json.dump(out, open(os.path.join(RESULTS_DIR, "r30_outages_calibrated.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
