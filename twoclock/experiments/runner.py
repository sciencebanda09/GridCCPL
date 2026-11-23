"""Shared experiment plumbing: episodes, seeds, paired comparison, logging."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from .. import ENV_VERSION
from ..envs import make_env
from ..instrumentation import RunLogger, summarise
from ..paths import RESULTS_DIR


def run_episode(env, agent, learn: bool = False, log: RunLogger | None = None) -> dict:
    """Run one episode to completion and return the domain scoreboard."""
    obs = env.reset()
    agent.reset(env)
    if learn and hasattr(agent, "begin_episode"):
        agent.begin_episode()
    while not env.done:
        x = np.asarray(obs.x, dtype=float).copy()
        a = agent.act(obs)
        obs, reward, done, info = env.step(a)
        if learn and hasattr(agent, "record"):
            agent.record(x, info["step_cost"], reward)
        if log is not None:
            log.log(step=info["step"], action=list(map(float, np.ravel(a))),
                    p_gen=list(map(float, info["p_gen"])),
                    battery=info["battery"], import_mw=info["import_mw"],
                    unserved_mw=info["unserved_mw"], overload_mw=info["overload_mw"],
                    step_cost=info["step_cost"], wear=list(map(float, info["wear"])),
                    wear_inc=list(map(float, info["wear_inc"])),
                    soc=info["soc"], settle=info["settle"],
                    reserve_mw=info["reserve_mw"])
    if learn and hasattr(agent, "end_episode"):
        agent.end_episode()
    out = env.episode_summary()
    out["env"] = env.name
    out["seed"] = int(env.cfg.seed)
    out["days"] = int(env.cfg.days)
    out["agent"] = getattr(agent, "name", type(agent).__name__)
    return out


def evaluate(agent, env_name: str, seeds, days: int = 10, **env_kw) -> dict:
    """Mean scoreboard over seeds for one agent and one environment."""
    rows = [run_episode(make_env(env_name, days=days, seed=s, **env_kw), agent)
            for s in seeds]
    keys = ["episode_cost", "unserved_mwh", "overload_mwh", "wear_end_mean",
            "wear_end_max", "eff_end_mean", "heat_rate_mean", "trips", "cost_per_mwh_served"]
    out = {k: summarise([r[k] for r in rows]) for k in keys}
    out["agent"] = getattr(agent, "name", "?")
    out["env"] = env_name
    out["rows"] = rows
    return out


def paired_delta(a: dict, b: dict) -> dict:
    """Per-seed difference ``a - b`` (same seeds, so this is paired)."""
    keys = ["episode_cost", "unserved_mwh", "wear_end_mean"]
    ra = {r["seed"]: r for r in a["rows"]}
    rb = {r["seed"]: r for r in b["rows"]}
    seeds = sorted(set(ra) & set(rb))
    out = {}
    for k in keys:
        d = [ra[s][k] - rb[s][k] for s in seeds]
        out[k] = summarise(d)
        out[k]["rel"] = float(np.mean([d[i] / max(1e-9, rb[s][k])
                                       for i, s in enumerate(seeds)]))
    out["n_pairs"] = len(seeds)
    return out


def fmt_row(label: str, r: dict, base: dict | None = None) -> str:
    """One markdown table row."""
    c = r["episode_cost"]
    rel = ""
    if base is not None:
        rel = f"{100.0 * (c['mean'] / max(1e-9, base['episode_cost']['mean']) - 1):+6.1f}%"
    return (f"| {label:22s} | {c['mean']/1e6:7.3f} ± {c['ci95']/1e6:5.3f} | {rel:>7s} "
            f"| {r['unserved_mwh']['mean']:8.2f} "
            f"| {r['wear_end_mean']['mean']:6.3f} "
            f"| {r['heat_rate_mean']['mean']:5.3f} |")


def header() -> str:
    return ("| agent                  | cost (M$) ± 95% CI  | vs blind "
            "| unserved MWh | wear  | hr    |\n"
            "|------------------------|---------------------|----------"
            "|--------------|-------|-------|")


@dataclass
class Results:
    out_dir: str = RESULTS_DIR

    def save(self, name: str, payload: dict) -> str:
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=lambda o: (
                o.item() if hasattr(o, "item") else str(o)))
        return path
