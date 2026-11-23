"""Package-relative locations for data, results and figures.

* ``DATA_DIR``    -- external input data (ENTSO-E CSV) and derived CSVs.
                     Override with the ``TWOCLOCK_DATA`` environment variable;
                     defaults to ``twoclock/data``.
* ``RESULTS_DIR`` -- JSON / Markdown results (``twoclock/results``).
* ``FIG_DIR``     -- generated figures. Override with ``TWOCLOCK_FIGS``;
                     defaults to ``twoclock/results/figures``.
"""
from __future__ import annotations

import os

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("TWOCLOCK_DATA", os.path.join(PKG_DIR, "data"))
RESULTS_DIR = os.path.join(PKG_DIR, "results")
FIG_DIR = os.environ.get("TWOCLOCK_FIGS", os.path.join(RESULTS_DIR, "figures"))
RAW_CSV = os.path.join(DATA_DIR, "energy_dataset.csv")


def fig_path(name: str) -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    return os.path.join(FIG_DIR, name)
