"""Figures (matplotlib only, deterministic)."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..agents import OPFAware, OPFBlind
from ..envs import make_env
from ..paths import RESULTS_DIR, fig_path
from ..physics import AgingParams, build_5bus


def fig_price_curve(path=os.path.join(RESULTS_DIR, "stage2_separation.json"),
                    out=None) -> str:
    out = out or fig_path("fig_price_curve.pdf")
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    mu = np.array(d["mu_grid"], dtype=float)
    J = np.array(d["J"], dtype=float) / 1e6
    J0 = J[0]
    ci = np.array(d["J_ci"], dtype=float) / 1e6
    m = mu > 0
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.errorbar(mu[m], J[m], yerr=ci[m], marker="o", ms=3, lw=1.2, capsize=2,
                color="C0", label="priced OPF")
    ax.axhline(J0, color="C3", ls="--", lw=1.1, label="degradation-blind OPF")
    suff = d["mu_suff"]
    mu_1 = 426.1  # analytical merit-order onset (c_coal - c_nuc) / alpha_nuc
    ax.axvspan(suff, mu.max() * 1.1, color="C0", alpha=0.08)
    ax.axvline(mu_1, color="0.45", ls="--", lw=0.9)
    ax.text(mu_1 * 0.85, J.min() + 0.18, r"$\mu_1$", color="0.35",
            fontsize=8, ha="right", va="center")
    ax.axvline(suff, color="C0", ls=":", lw=1.1)
    ax.text(suff * 1.25, J.min() + 0.45, r"$\mu_{\rm suff}$", color="C0",
            fontsize=9, ha="left", va="center")
    ax.set_xscale("log")
    ax.set_xlim(mu[m].min() * 0.8, mu.max() * 1.1)
    ax.set_xlabel(r"wear price $\mu$ (\$/wear-unit)")
    ax.set_ylabel("total cost (M\\$)")
    ax.set_title(r"cost is flat above $\mu_{\rm suff}$", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper right", frameon=True, facecolor="white",
              edgecolor="none", framealpha=1.0)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_trajectories(out=None, seed=1, days=10) -> str:
    out = out or fig_path("fig_trajectories.pdf")
    net, aging = build_5bus(), AgingParams()
    env1 = make_env("normal", days=days, seed=seed)
    env2 = make_env("normal", days=days, seed=seed)
    blind = OPFBlind(net, aging)
    priced = OPFAware(net, aging, 1.0e4)  # constant price mu = 1e4 (not the learned agent)

    def rollout(env, agent):
        obs = env.reset()
        wear, uns, heat = [], [], []
        while not env.done:
            obs, _, _, info = env.step(agent.act(obs))
            wear.append(float(np.mean(info["wear"][:3])))
            uns.append(float(info["unserved_mw"]) * 0.25)
            heat.append(float(np.mean(1.0 + 0.10 * info["wear"][:3]
                                      + 0.80 * info["wear"][:3] ** 2)))
        return np.array(wear), np.array(uns), np.array(heat)

    w1, u1, h1 = rollout(env1, blind)
    w2, u2, h2 = rollout(env2, priced)
    t = np.arange(len(w1)) * 0.25 / 24.0

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.2))
    ax[0].plot(t, w1, color="C3", lw=1.2, label="degradation-blind")
    ax[0].plot(t, w2, color="C0", lw=1.2, label=r"constant price ($\mu=10^4$)")
    ax[0].set_xlabel("day")
    ax[0].set_ylabel("mean thermal wear $W$")
    ax[0].set_title("(a) slow state", fontsize=9)
    ax[0].legend(fontsize=6.5, frameon=False)
    ax[1].plot(t, h1, color="C3", lw=1.2)
    ax[1].plot(t, h2, color="C0", lw=1.2)
    ax[1].set_xlabel("day")
    ax[1].set_ylabel("heat-rate multiplier")
    ax[1].set_title("(b) heat rate", fontsize=9)
    for a in ax:
        a.tick_params(labelsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(fig_price_curve())
    print(fig_trajectories())
