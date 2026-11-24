"""Calibration figure: (a) observed vs. baseline-expected trip-like shutdowns by wear-state group (R33),
(b) heat-rate effect at W = 1, real vs. placebo (R29).  Writes fig_calibration.pdf next to the paper.

Run: python -m twoclock.experiments.plot_calibration [out_dir]
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    from twoclock.paths import RESULTS_DIR
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/kumar/OneDrive/Desktop/GridCCPL-main/paper_v3"
    r33 = json.load(open(os.path.join(RESULTS_DIR, "r33_hazard_cluster.json")))
    r29 = json.load(open(os.path.join(RESULTS_DIR, "r29_heatrate_calibration.json")))
    n = np.array(r33["pooled_n"]); x = np.array(r33["pooled_x"]); d = np.array(r33["pooled_d"])
    r28 = json.load(open(os.path.join(RESULTS_DIR, 'r28_hazard_all_us.json')))
    h0 = np.array(r28['taus']['12.0']['E1']['h0_per_h'])   # fitted no-wear baseline per load stratum
    mx = np.where(n > 0, x / np.maximum(n, 1), 0.0)
    # groups of W^2 bins with equal numbers of risk hours (over strata), only where wear is nonzero
    edges = [0, 1, 2, 4, 8, 16, 40, 100, n.shape[1]]
    xs_, ratio, lo, hi, fit = [], [], [], [], []
    kap = r33["kappa_hat"] / 24  # beta per hour
    for a, b in zip(edges[:-1], edges[1:]):
        nn = n[:, a:b]; dd = d[:, a:b].sum(); ee = (h0[:, None] * nn).sum()
        if ee <= 0:
            continue
        xbar = x[:, a:b].sum() / max(nn.sum(), 1)
        r = dd / ee; se = np.sqrt(max(dd, 1)) / ee
        mod = ((h0[:, None] + kap * mx[:, a:b]) * nn).sum() / ee   # kap per hour
        xs_.append(np.sqrt(xbar)); ratio.append(r); lo.append(r - 1.96 * se); hi.append(r + 1.96 * se); fit.append(mod)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.2))
    a0 = ax[0]
    a0.errorbar(xs_, ratio, yerr=[np.array(ratio) - np.array(lo), np.array(hi) - np.array(ratio)], fmt="o", ms=4, color="#2a6db0", capsize=2, label="observed")
    a0.plot(xs_, fit, "-", color="#c0392b", label=r"fit, $\hat\kappa$=%.4f/day" % r33["kappa_hat"])
    a0.axhline(1, color="gray", lw=0.7, ls=":")
    a0.set_xlabel(r"model wear state $W$ (group mean, $\tau$=12 h)"); a0.set_ylabel("trip-like events / expected\nunder no-wear baseline")
    a0.set_title("(a) Trip rate rises with the wear path", fontsize=8.5); a0.legend(fontsize=7, frameon=False)
    a1 = ax[1]
    taus = [3.0, 12.0, 24.0]; w = 0.26
    for j, (key, lab, col) in enumerate((("month", "real (month cells)", "#2a6db0"), ("month_hod", "real (+hour-of-day)", "#7fb3e6"), ("month", "placebo", "#999999"))):
        vals = []
        for t in taus:
            k = f"tau{t}_{key}_{'placebo' if lab == 'placebo' else 'real'}"
            vals.append(r29["fits"][k]["hr_gain_at_W1_pct"])
        a1.bar(np.arange(3) + (j - 1) * w, vals, w, color=col, label=lab)
    a1.axhline(0, color="k", lw=0.7)
    a1.set_xticks(range(3)); a1.set_xticklabels([r"$\tau$=%g h" % t for t in taus])
    a1.set_ylabel("heat-rate change at $W$=1 [%]"); a1.set_ylim(-2, 2)
    a1.text(0.98, 0.07, "model assumed +90%", transform=a1.transAxes, ha="right", va="bottom", fontsize=7, style="italic")
    a1.set_title("(b) No short-horizon heat-rate degradation", fontsize=8.5); a1.legend(fontsize=6.5, frameon=False, loc="upper left")
    for a in ax:
        a.tick_params(labelsize=7); a.xaxis.label.set_size(8); a.yaxis.label.set_size(8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_calibration.pdf"))
    print("saved", os.path.join(out_dir, "fig_calibration.pdf"), len(xs_), "groups")


if __name__ == "__main__":
    main()
