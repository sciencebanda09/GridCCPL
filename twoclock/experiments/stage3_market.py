"""Stage 3 -- discovery.

Can the price of an unpriced externality be read off market data?

Setup: an operator runs the wear-aware OPF with an unknown price ``mu_true``.
We observe only the *outcome* -- MW dispatched, load, spot, availability -- and
invert the optimisation to recover the price:

    mu_hat = argmin_mu  || p_observed - p_opf(mu) ||^2

Because ``mu`` enters the OPF as a linear cost coefficient, the predicted
dispatch is monotone in ``mu``, and the inverse problem is a 1-D profile
likelihood.  We report the *profile interval* -- the connected set of prices
whose fit is within 5% of the best fit -- rather than only a point estimate,
because the flat region identified in stage 2 makes the price weakly identified
by construction.  It is a set of near-optimal prices, not a confidence interval:
the dispatch is a step function of ``mu``, so no coverage claim is available.

Validation on a synthetic market with a known price, then a loader for real ISO
data (``load_iso_csv``) so the same estimator runs unchanged on PJM / CAISO /
ERCOT exports.

Run:  python -m twoclock.experiments.stage3_market
"""
from __future__ import annotations

import csv
import os

import numpy as np

from ..agents import OPFAware
from ..agents.baselines import _battery_bounds
from ..envs import make_env
from ..physics import (AgingParams, GEN_DISPATCHABLE, GEN_RATING, build_5bus,
                       dc_opf, wear_free_mw, wear_price_from_mu)
from .runner import Results

FREE = None


# ---------------------------------------------------------------------------
# Synthetic market: an operator with a hidden price
# ---------------------------------------------------------------------------

def synthetic_market(net, aging, mu_true: float, days: int = 3, seed: int = 0,
                     noise_mw: float = 0.0, rng=None) -> list[dict]:
    rng = rng or np.random.default_rng(1234)
    env = make_env("normal", days=days, seed=seed)
    agent = OPFAware(net, aging, mu_true, fast=False)
    obs = env.reset()
    rows = []
    while not env.done:
        ctx = obs.ctx
        a = agent.act(obs)
        # The observable is the operator's *scheduled* dispatch (the OPF
        # solution), not the environment's realised output after physical
        # cleanup -- that is what market data contains.
        p = np.asarray(agent.last_result.p_gen, dtype=float).copy()
        obs, _, _, info = env.step(a)
        if noise_mw > 0:
            p[:3] += rng.normal(0.0, noise_mw, 3)
            p = np.maximum(p, 0.0)
        rows.append({
            "load": float(ctx["load"]),
            "renewable": float(ctx["renewable"]),
            "spot": float(ctx["spot"]),
            "prev_p": np.asarray(ctx["prev_p"], dtype=float).tolist(),
            "pmax_eff": np.asarray(ctx["pmax_eff"], dtype=float).tolist(),
            "soc": float(ctx["soc"]),
            "reserve_req": float(ctx["reserve_req"]),
            "p_gen": p.tolist(),
        })
    return rows


def _predict(net, aging, rows, mu):
    free = wear_free_mw(aging)
    wp = None if mu <= 0 else wear_price_from_mu(mu, aging, GEN_RATING, GEN_DISPATCHABLE)
    sse = 0.0
    for r in rows:
        lo, hi = _battery_bounds(r.get("soc", 0.5), 0.25)
        res = dc_opf(net, r["load"], np.asarray(r["prev_p"]),
                     r["renewable"], np.asarray(r["pmax_eff"]),
                     spot_price=r["spot"], wear_price_mwh=wp, wear_free_mw=free,
                     reserve_req_mw=r.get("reserve_req", 0.0),
                     battery_min_mw=lo, battery_max_mw=hi)
        d = np.asarray(r["p_gen"]) - res.p_gen
        sse += float(np.sum(d * d))
    return sse


def profile_interval(net, aging, rows, mu_hat: float, sse_min: float,
                     lo: float = 1.0, hi: float = 3.0e5, tol: float = 1.05,
                     steps: int = 14) -> list[float]:
    """Contiguous set of prices whose fit is within ``tol`` of the best fit.

    The set is found by geometric expansion from the optimum until the fit
    leaves the tolerance band, then bisection to the boundary.  This is a
    continuous profile interval, not a grid artefact, which matters because a
    coarse grid would report a single point for every half-decent fit and
    overstate how well the price is identified.
    """
    thr = sse_min * tol

    def _side(direction: int) -> float:
        inside = float(mu_hat)
        out = inside * (2.0 if direction > 0 else 0.5)
        for _ in range(48):
            out = min(max(out, lo), hi)
            if _predict(net, aging, rows, out) > thr:
                break
            if out == lo or out == hi:
                return out
            inside = out
            out = inside * (2.0 if direction > 0 else 0.5)
        for _ in range(steps):
            mid = 0.5 * (inside + out)
            if _predict(net, aging, rows, mid) <= thr:
                inside = mid
            else:
                out = mid
        return 0.5 * (inside + out)

    return [_side(-1), _side(+1)]


def estimate_mu(net, aging, rows, lo: float = 1.0, hi: float = 3.0e5,
                n_grid: int = 40) -> dict:
    """Grid + golden-section refinement in log space, then a profile interval."""
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), n_grid))
    sse = np.array([_predict(net, aging, rows, m) for m in grid])
    i = int(np.argmin(sse))

    # golden-section refinement around the discrete minimiser
    a = grid[max(0, i - 1)]
    b = grid[min(len(grid) - 1, i + 1)]
    gr = (np.sqrt(5) - 1) / 2
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = _predict(net, aging, rows, c), _predict(net, aging, rows, d)
    for _ in range(18):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = _predict(net, aging, rows, c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = _predict(net, aging, rows, d)
    mu_hat = float(0.5 * (a + b))
    sse_min = float(min(float(fc), float(fd), float(sse.min())))
    interval = profile_interval(net, aging, rows, mu_hat, sse_min, lo=lo, hi=hi)
    return {"mu_hat": mu_hat, "grid": grid.tolist(), "sse": sse.tolist(),
            "sse_min": sse_min, "interval": interval}


# ---------------------------------------------------------------------------
# Real market data
# ---------------------------------------------------------------------------

def load_iso_csv(path: str) -> list[dict]:
    """Load a dispatch export.

    Expected columns (extra columns ignored)::

        timestamp, load_mw, renewable_mw, spot, p_gas, p_coal, p_nuclear

    ``prev_p`` is reconstructed from the previous row; ``pmax_eff`` assumes the
    fleet is available (override by supplying ``pmax_gas`` etc. if present).
    """
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rd = csv.DictReader(fh)
        prev = None
        for rec in rd:
            p = np.array([float(rec["p_gas"]), float(rec["p_coal"]),
                          float(rec["p_nuclear"]), float(rec.get("renewable_mw", 0.0))])
            if prev is None:
                prev = p.copy()
            rows.append({
                "load": float(rec["load_mw"]),
                "renewable": float(rec["renewable_mw"]),
                "spot": float(rec["spot"]),
                "prev_p": prev.tolist(),
                "pmax_eff": [350.0, 300.0, 200.0, float(rec["renewable_mw"])],
                "p_gen": p.tolist(),
            })
            prev = p
    return rows


def main() -> int:
    net, aging = build_5bus(), AgingParams()
    res = Results()
    payload = {"env_version": "twoclock-env-1", "truths": []}

    print("== recovery of a known price (noiseless) ==")
    print(" mu_true      mu_hat     rel.err    profile interval")
    for mu_true in (1000.0, 5000.0, 20000.0, 100000.0):
        rows = synthetic_market(net, aging, mu_true, days=3, seed=3)
        est = estimate_mu(net, aging, rows)
        rel = (est["mu_hat"] - mu_true) / mu_true
        print(f" {mu_true:9.0f}  {est['mu_hat']:9.0f}  {100*rel:+7.1f}%   "
              f"[{est['interval'][0]:.0f}, {est['interval'][1]:.0f}]")
        payload["truths"].append({"mu_true": mu_true, "noise_mw": 0.0,
                                  **{k: est[k] for k in ("mu_hat", "interval", "sse_min")}})

    print("\n== recovery under measurement noise (2 MW per unit) ==")
    for mu_true in (5000.0, 20000.0):
        rows = synthetic_market(net, aging, mu_true, days=3, seed=3,
                                noise_mw=2.0, rng=np.random.default_rng(7))
        est = estimate_mu(net, aging, rows)
        rel = (est["mu_hat"] - mu_true) / mu_true
        print(f" mu_true {mu_true:9.0f}  mu_hat {est['mu_hat']:9.0f}  {100*rel:+6.1f}%")
        payload["truths"].append({"mu_true": mu_true, "noise_mw": 2.0,
                                  **{k: est[k] for k in ("mu_hat", "interval", "sse_min")}})

    # If a real export is present, run the identical estimator on it.
    iso = os.environ.get("TWOCLOCK_ISO_CSV", "")
    if iso and os.path.exists(iso):
        print(f"\n== real market data: {iso} ==")
        rows = load_iso_csv(iso)
        est = estimate_mu(net, aging, rows)
        print(f"   mu_hat = {est['mu_hat']:.0f} $/wear-unit  "
              f"(profile interval [{est['interval'][0]:.0f}, {est['interval'][1]:.0f}])")
        payload["iso"] = {"path": iso, "n_rows": len(rows),
                          "mu_hat": est["mu_hat"], "interval": est["interval"]}
    else:
        print("\n(no TWOCLOCK_ISO_CSV set; synthetic validation only)")

    path = res.save("stage3_market.json", payload)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
