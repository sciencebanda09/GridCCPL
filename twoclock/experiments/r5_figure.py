"""Paper figure for R5 Stage 3, drawn from results/r5_stage3.json (no re-run).

Run (from the folder that contains ``twoclock``):
    python -m twoclock.experiments.r5_figure
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..paths import RESULTS_DIR

HI, LO = 3.0e5, 1.0
INK, MUTED, GRID = "#1f1e1c", "#52514e", "#e4e3df"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MK = ["o", "s", "^", "D"]


def main() -> int:
    d = json.load(open(os.path.join(RESULTS_DIR, "r5_stage3.json"), encoding="utf-8"))
    thr = d["nominal_threshold_mu_star"]
    raw = d["synthetic_raw"]
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.labelcolor": INK, "text.color": INK, "pdf.fonttype": 42})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.1), gridspec_kw={"width_ratios": [1, 1.15]})

    # (a) synthetic identification, 72-h windows
    bins = np.geomspace(100, 1e5, 13)
    cen = np.sqrt(bins[:-1] * bins[1:])
    for c, m, nz in zip(C, MK, (0.0, 1.0, 2.0, 5.0)):
        g = [r for r in raw if r["noise"] == nz and r["L"] == 72]
        mu = np.array([r["mu_true"] for r in g])
        ident = np.array([r["interval"][1] < HI / 1.01 for r in g])
        idx = np.digitize(mu, bins) - 1
        fr = [ident[idx == k].mean() if (idx == k).any() else np.nan for k in range(len(cen))]
        a.semilogx(cen, fr, "-", marker=m, color=c, lw=1.5, ms=3.5, label=f"{nz:g} MW",
                   markeredgecolor="white", markeredgewidth=0.5)
    a.axvline(thr, color=MUTED, ls="--", lw=1)
    a.annotate(r"$\mu_{\rm suff}\approx$" + f"{thr:,.0f}", xy=(thr, 0.62), xytext=(thr * 2.6, 0.62),
               fontsize=7, color=MUTED, va="center",
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    a.set_xlabel(r"hidden wear price $\mu$")
    a.set_ylabel("fraction identified")
    a.set_title("(a) Synthetic market, 72-h windows", loc="left")
    a.set_ylim(-0.03, 1.05)
    a.grid(axis="y", color=GRID, lw=0.6)
    a.set_axisbelow(True)
    a.legend(title="obs. noise", loc="lower left", frameon=False, title_fontsize=7,
             handlelength=1.6, borderaxespad=0.2)

    # (b) real vs placebos: fraction of windows with lower bound > floor
    real = d["real"]
    P = real["placebo"]
    items = [("ENTSO-E real", real["base"]["frac_interval_excludes_floor"], C[0]),
             (r"null $\mu$=0 + perm. resid.", P["null_mu0_permuted_real_residuals"]["frac_interval_excludes_floor"], "#9a9893"),
             (r"null $\mu$=0 + 5 MW Gauss.", P["null_mu0_gauss5"]["frac_interval_excludes_floor"], "#9a9893"),
             (r"positive $\mu$=2000", P["positive_mu2000_permuted_real_residuals"]["frac_interval_excludes_floor"], "#6b6a66"),
             ("spot shuffled", P["spot_shuffled_within_window"]["frac_interval_excludes_floor"], "#6b6a66")]
    y = np.arange(len(items))
    b.barh(y, [v for _, v, _ in items], color=[c for *_, c in items], height=0.62)
    for yi, (_, v, _) in zip(y, items):
        b.text(v + 0.008, yi, f"{v:.2f}", va="center", fontsize=7, color=INK)
    b.set_yticks(y)
    b.set_yticklabels([l for l, *_ in items])
    b.tick_params(axis="y", length=0)
    b.invert_yaxis()
    b.set_xlim(0, 0.36)
    b.set_xlabel("fraction of 72-h windows with lower bound > floor")
    b.set_title(f"(b) ENTSO-E Spain 2015-18 vs. placebos (n={real['base']['n_windows']})", loc="left")
    b.grid(axis="x", color=GRID, lw=0.6)
    b.set_axisbelow(True)

    fig.tight_layout(w_pad=1.5)
    out = os.path.join(RESULTS_DIR, "fig_r5_paper")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(out + ".pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
