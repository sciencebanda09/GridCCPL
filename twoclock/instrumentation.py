"""Instrumentation: everything a long-lived research programme needs logged now.

Design rule: if a quantity cannot be reconstructed from a log, the run is
forfeited.  In particular we log the *true* shadow price (from the LP dual),
the slow state, and the physical parameters of the run.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field, asdict

import numpy as np


def _clean(v):
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return [_clean(x) for x in v.tolist()]
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return v


@dataclass
class RunLogger:
    """JSONL logger with a run header containing all physical parameters."""

    out_dir: str
    run_name: str
    header: dict = field(default_factory=dict)
    enabled: bool = True
    _path: str = field(init=False, default="")

    def __post_init__(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        self._path = os.path.join(self.out_dir, f"{self.run_name}.jsonl")
        if self.enabled:
            full = {
                "type": "header",
                "ts": time.time(),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                **self.header,
            }
            with open(self._path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(_clean(full)) + "\n")

    @property
    def path(self) -> str:
        return self._path

    def log(self, **row) -> None:
        if not self.enabled:
            return
        row["type"] = "row"
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(_clean(row)) + "\n")

    def log_episode(self, episode: int, **summary) -> None:
        self.log(episode=episode, kind="episode", **summary)


def summarise(values, ci: bool = True) -> dict:
    """Mean and 95% CI (normal approximation) for a list of episode values."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return {"mean": 0.0, "std": 0.0, "n": 0, "ci95": 0.0}
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if v.size > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(v.size)) if ci else 0.0
    return {"mean": mean, "std": std, "n": int(v.size), "ci95": ci95,
            "min": float(v.min()), "max": float(v.max())}
