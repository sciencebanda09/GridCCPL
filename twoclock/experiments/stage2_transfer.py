"""Stage 2c -- Zero-shot topology transfer (5-bus -> 14-bus 20-line network)
and the CMDP Dual-Ascent Dilemma trajectory.

Tests a core prediction of 'The Price Is the Policy':
Because the shadow price mu = dV/dW [$/wear-unit] prices equipment damage
rather than bus-specific actions, a scalar price learned on the 5-bus network
transfers *zero-shot* (with zero retraining) to a 14-bus, 20-line meshed network
where a neural dispatch policy cannot even execute due to changed bus/line topology.

Run:  python -m twoclock.experiments.stage2_transfer
"""
from __future__ import annotations

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from twoclock.physics import AgingParams, DCNetwork, Line, build_5bus
from twoclock.envs import EnvConfig, TwoClockEnv, make_env
from twoclock.agents import OPFAware, OPFBlind, ValueGradientAgent
from twoclock.experiments.runner import Results, run_episode
from twoclock.paths import fig_path


def build_14bus() -> tuple[DCNetwork, np.ndarray, np.ndarray, int, int]:
    """IEEE 14-bus inspired 20-line meshed transmission topology.

    Returns (net, gen_bus, load_share, battery_bus, import_bus).
    """
    lines = [
        Line(0, 1, x=0.059, limit_mw=260.0),
        Line(0, 4, x=0.223, limit_mw=220.0),
        Line(1, 2, x=0.198, limit_mw=220.0),
        Line(1, 3, x=0.176, limit_mw=220.0),
        Line(1, 4, x=0.174, limit_mw=220.0),
        Line(2, 3, x=0.171, limit_mw=200.0),
        Line(3, 4, x=0.042, limit_mw=280.0),
        Line(3, 6, x=0.209, limit_mw=180.0),
        Line(3, 8, x=0.556, limit_mw=160.0),
        Line(4, 5, x=0.252, limit_mw=200.0),
        Line(5, 10, x=0.199, limit_mw=150.0),
        Line(5, 11, x=0.256, limit_mw=150.0),
        Line(5, 12, x=0.130, limit_mw=180.0),
        Line(6, 7, x=0.176, limit_mw=180.0),
        Line(6, 8, x=0.110, limit_mw=200.0),
        Line(8, 9, x=0.085, limit_mw=180.0),
        Line(8, 13, x=0.270, limit_mw=150.0),
        Line(9, 10, x=0.192, limit_mw=150.0),
        Line(11, 12, x=0.200, limit_mw=150.0),
        Line(12, 13, x=0.348, limit_mw=150.0),
    ]
    net = DCNetwork(
        n_bus=14,
        lines=lines,
        slack=0,
        bus_names=tuple(f"bus{i}" for i in range(14)),
    )
    gen_bus = np.array([0, 1, 2, 7])
    load_share = np.array([
        0.00, 0.08, 0.18, 0.14, 0.04, 0.06, 0.00,
        0.00, 0.12, 0.05, 0.04, 0.05, 0.10, 0.14
    ])
    load_share = load_share / load_share.sum()
    return net, gen_bus, load_share, 3, 5


def run_14bus_transfer(aging: AgingParams) -> dict:
    import twoclock.physics.dc_flow as dcf
    import twoclock.physics.opf as opf_mod
    import twoclock.envs.two_clock_env as env_mod

    # 1. Train ValueGradientAgent on canonical 5-bus normal regime (seeds 101-103)
    net5 = build_5bus()
    vg_5bus = ValueGradientAgent(net5, aging, seed=7, mu_max=1e6)
    vg_5bus.enabled = False
    for r in range(3):
        for s in (101, 102, 103):
            run_episode(make_env("normal", days=5, seed=s + 10 * r), vg_5bus, learn=True)
        vg_5bus.update()
        vg_5bus.enabled = True
    mu_learned_5bus = float(vg_5bus.mu_scalar)

    # 2. Switch topology globals cleanly to 14-bus (restoring in finally block)
    net14, gbus14, lshare14, bbus14, ibus14 = build_14bus()
    orig = (dcf.GEN_BUS.copy(), dcf.LOAD_SHARE.copy(), dcf.BATTERY_BUS, dcf.IMPORT_BUS)
    results_14 = {"mu_learned_5bus": mu_learned_5bus, "regimes": {}}
    try:
        for mod in (dcf, opf_mod, env_mod):
            mod.GEN_BUS = gbus14
            mod.LOAD_SHARE = lshare14
            mod.BATTERY_BUS = bbus14
            mod.IMPORT_BUS = ibus14

        for regime in ("normal", "hard", "peak", "storm"):
            blind_c, zero_c, native_c = [], [], []
            blind_u, zero_u, native_u = [], [], []
            blind_w, zero_w, native_w = [], [], []

            # Train native 14-bus ValueGradientAgent for comparison
            vg_14 = ValueGradientAgent(net14, aging, seed=7, mu_max=1e6)
            vg_14.enabled = False
            for r in range(3):
                for s in (101, 102, 103):
                    env_tr = make_env(regime, days=5, seed=s + 10 * r)
                    env_tr.net = net14
                    run_episode(env_tr, vg_14, learn=True)
                vg_14.update()
                vg_14.enabled = True

            for s in (1, 2, 3, 4, 5):
                # Blind 14-bus
                e_b = make_env(regime, days=10, seed=s)
                e_b.net = net14
                rb = run_episode(e_b, OPFBlind(net14, aging))
                blind_c.append(rb["episode_cost"])
                blind_u.append(rb["unserved_mwh"])
                blind_w.append(rb["wear_end_mean"])

                # Zero-shot transfer of 5-bus price onto 14-bus
                e_z = make_env(regime, days=10, seed=s)
                e_z.net = net14
                rz = run_episode(e_z, OPFAware(net14, aging, mu_learned_5bus))
                zero_c.append(rz["episode_cost"])
                zero_u.append(rz["unserved_mwh"])
                zero_w.append(rz["wear_end_mean"])

                # Native 14-bus learner
                e_n = make_env(regime, days=10, seed=s)
                e_n.net = net14
                rn = run_episode(e_n, vg_14)
                native_c.append(rn["episode_cost"])
                native_u.append(rn["unserved_mwh"])
                native_w.append(rn["wear_end_mean"])

            bc, zc, nc = float(np.mean(blind_c)), float(np.mean(zero_c)), float(np.mean(native_c))
            results_14["regimes"][regime] = {
                "blind_cost_m": bc / 1e6,
                "zeroshot_cost_m": zc / 1e6,
                "native_cost_m": nc / 1e6,
                "zeroshot_saving_pct": 100.0 * (1.0 - zc / bc),
                "native_saving_pct": 100.0 * (1.0 - nc / bc),
                "blind_unserved_mwh": float(np.mean(blind_u)),
                "zeroshot_unserved_mwh": float(np.mean(zero_u)),
                "blind_wear": float(np.mean(blind_w)),
                "zeroshot_wear": float(np.mean(zero_w)),
            }
            print(f"14-bus [{regime:6s}]: blind={bc/1e6:.3f} M$ -> zero-shot={zc/1e6:.3f} M$ "
                  f"({100*(1-zc/bc):+.1f}%), native={nc/1e6:.3f} M$ ({100*(1-nc/bc):+.1f}%)")
    finally:
        for mod in (dcf, opf_mod, env_mod):
            mod.GEN_BUS = orig[0]
            mod.LOAD_SHARE = orig[1]
            mod.BATTERY_BUS = orig[2]
            mod.IMPORT_BUS = orig[3]
    return results_14


def run_dual_dilemma_trace(net, aging) -> dict:
    """Trace mu across 8 rounds for Dual Ascent (eps=0.06, eps=0.00 on peak) vs Value-Gradient."""
    rounds = 8
    eta = 25000.0
    trace_da_loose = [0.0]
    trace_da_zero_peak = [0.0]
    trace_vg = [0.0]

    mu_l = 0.0
    mu_z = 0.0
    vg = ValueGradientAgent(net, aging, seed=7, mu_max=1e6)
    vg.enabled = False

    for r in range(rounds):
        # 1. Dual ascent loose (eps=0.06, normal)
        wl = []
        for s in (101, 102, 103):
            ag = OPFBlind(net, aging) if mu_l <= 0 else OPFAware(net, aging, mu_l)
            wl.append(run_episode(make_env("normal", days=5, seed=s + 10 * r), ag)["wear_end_mean"])
        mu_l = max(0.0, mu_l + eta * (float(np.mean(wl)) - 0.06))
        trace_da_loose.append(float(mu_l))

        # 2. Dual ascent zero target on peak (eps=0.00, peak: forced wear > 0 -> diverges!)
        wz = []
        for s in (101, 102, 103):
            ag = OPFBlind(net, aging) if mu_z <= 0 else OPFAware(net, aging, mu_z)
            wz.append(run_episode(make_env("peak", days=5, seed=s + 10 * r), ag)["wear_end_mean"])
        mu_z = max(0.0, mu_z + eta * (float(np.mean(wz)) - 0.00))
        trace_da_zero_peak.append(float(mu_z))

        # 3. Value-gradient
        for s in (101, 102, 103):
            run_episode(make_env("normal", days=5, seed=s + 10 * r), vg, learn=True)
        vg.update()
        vg.enabled = True
        trace_vg.append(float(vg.mu_scalar))

    return {
        "rounds": list(range(rounds + 1)),
        "dual_ascent_eps006_normal": trace_da_loose,
        "dual_ascent_eps000_peak": trace_da_zero_peak,
        "value_gradient": trace_vg,
    }


def plot_dilemma_and_transfer(dilemma: dict, transfer: dict) -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.5))

    r = np.asarray(dilemma["rounds"])
    ax1.plot(r, np.asarray(dilemma["dual_ascent_eps006_normal"]) / 1e3, "-s", color="#ff7f0e", lw=1.6, ms=4,
             label="CMDP dual ascent ($\\epsilon=0.06$, oscillates $<\\mu_{\\mathrm{suff}}$)")
    ax1.plot(r, np.asarray(dilemma["dual_ascent_eps000_peak"]) / 1e3, "-^", color="#d62728", lw=1.6, ms=4,
             label="CMDP dual ascent ($\\epsilon=0$, peak: diverges)")
    ax1.axhline(5.0, color="#1f77b4", ls="--", lw=1.2, label="Threshold $\\mu_{\\mathrm{suff}} = 5$ k\\$/wear")
    ax1.axhspan(5.0, 18.0, color="#1f77b4", alpha=0.12, label="Optimal half-line $[\\mu_{\\mathrm{suff}},\\infty)$")
    ax1.set_xlabel("Training round $k$")
    ax1.set_ylabel("Multiplier $\\mu_k$ (k\\$/wear-unit)")
    ax1.set_title("(a) The CMDP dual-ascent dilemma")
    ax1.set_ylim(0, 17.5)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9)

    regimes = ["normal", "hard", "peak", "storm"]
    x = np.arange(len(regimes))
    w = 0.26
    b_vals = [transfer["regimes"][rg]["blind_cost_m"] for rg in regimes]
    z_vals = [transfer["regimes"][rg]["zeroshot_cost_m"] for rg in regimes]
    n_vals = [transfer["regimes"][rg]["native_cost_m"] for rg in regimes]

    ax2.bar(x - w, b_vals, width=w, color="#d62728", alpha=0.85, label="Blind OPF (14-bus)")
    ax2.bar(x, z_vals, width=w, color="#1f77b4", alpha=0.90, label="Zero-shot 5$\\to$14 bus (ours)")
    ax2.bar(x + w, n_vals, width=w, color="#2ca02c", alpha=0.85, label="Retrained on 14-bus")
    ax2.set_xticks(x)
    ax2.set_xticklabels(regimes)
    ax2.set_ylim(0, 20.0)
    ax2.set_ylabel("10-day system cost (M\\$)")
    ax2.set_title("(b) Zero-shot price transfer to 14-bus grid")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=6.8)

    fig.tight_layout(w_pad=1.5)
    for out_path in (
        fig_path("fig_dilemma_transfer.png"),
        fig_path("fig_dilemma_transfer.pdf"),
    ):
        fig.savefig(out_path, dpi=300 if out_path.endswith(".png") else None, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    net, aging = build_5bus(), AgingParams()
    res = Results()
    dilemma = run_dual_dilemma_trace(net, aging)
    transfer = run_14bus_transfer(aging)
    payload = {
        "env_version": "twoclock-env-1",
        "dual_ascent_dilemma": dilemma,
        "transfer_5_to_14bus": transfer,
    }
    path = res.save("stage2c_transfer.json", payload)
    plot_dilemma_and_transfer(dilemma, transfer)
    print(f"wrote {path} and fig_dilemma_transfer.png/.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
