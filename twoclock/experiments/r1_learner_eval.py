"""R1 -- a calibrated learned wear price (pathwise value gradient).

Compares, on 4 regimes x held-out seeds 1-5 (10-day episodes):

* ``opf_blind``          degradation-blind OPF
* ``opf_mu*``            best constant price, calibrated on *normal* training
                         seeds 101-102 exactly as in ``stage1_existence``
* ``vg_old``             the ridge value-gradient learner of stage 1
* ``pvg``                new: pathwise (CRN finite-difference) value gradient
                         with a 20% trimmed mean; one fleet-wide scalar price,
                         the smallest self-consistent one (max_g g_g(mu) <= mu)
                         found by log-bisection, trained per regime on that
                         regime's training seeds 101-102.  ``shared=True`` was
                         chosen on validation seeds 103-104, never on test.

Reference only (not a competitor that the claims rest on):

* ``opf_mu_regime``      constant price calibrated per regime on seeds 101-102

Also a training-seed diagnostic of the old learner's failure mode: the same
saturated mu with and without the per-unit ``price_clip``.

Hyper-parameters of the new learner are fixed across regimes (no per-regime
tuning, nothing tuned on test seeds).

Run:  python -m twoclock.experiments.r1_learner_eval
"""
from __future__ import annotations

import sys
import time

import numpy as np

from ..agents import OPFAware, OPFBlind, PathwisePriceAgent
from ..agents.baselines import OPFAgent, _battery_bounds, action_from_opf
from ..physics import (GEN_DISPATCHABLE, GEN_RATING, AgingParams, build_5bus,
                       fast_dispatch, wear_free_mw, wear_price_from_mu)
from .runner import Results, evaluate, paired_delta
from .stage1_existence import calibrate_mu, train_value_agent

ENVS = ("normal", "hard", "peak", "storm")
TEST_SEEDS = (1, 2, 3, 4, 5)
TRAIN_SEEDS = (101, 102)
DAYS = 10
STEPS_PER_EP = 96 * DAYS
MU_SUFF = 4964.0                     # closed-form threshold (stage 2)
PVG_KW = dict(iters=7, lo=300.0, hi=1.0e6, n_snap=16, horizon=96, delta=0.05,
              shared=True)
CAL_GRID = (0.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0)


class ClippedPriceOPF(OPFAgent):
    """Wear-aware OPF with the per-unit $/MWh clip of ValueGradientAgent."""

    def __init__(self, net, aging, mu, clip):
        super().__init__(net, aging, mu)
        self.clip = clip
        self.name = f"opf_mu{mu:g}_clip{clip:g}"

    def act(self, obs):
        ctx = obs.ctx
        lo, hi = _battery_bounds(ctx["soc"], 0.25)
        wp = np.clip(wear_price_from_mu(self.mu, self.aging, GEN_RATING,
                                        GEN_DISPATCHABLE), 0.0, self.clip)
        res = fast_dispatch(self.net, ctx["load"], ctx["prev_p"], ctx["renewable"],
                            ctx["pmax_eff"], reserve_req_mw=ctx["reserve_req"],
                            battery_min_mw=lo, battery_max_mw=hi,
                            spot_price=ctx["spot"], wear_price_mwh=wp,
                            wear_free_mw=wear_free_mw(self.aging))
        return action_from_opf(res, ctx["load"])


def _brief(r: dict) -> dict:
    return {k: r[k] for k in ("episode_cost", "unserved_mwh", "wear_end_mean",
                              "wear_end_max", "heat_rate_mean")} | {
        "per_seed_cost": [row["episode_cost"] for row in r["rows"]]}


def main(argv=None) -> int:
    t0 = time.time()
    net, aging = build_5bus(), AgingParams()
    coef = wear_price_from_mu(1.0, aging, GEN_RATING, GEN_DISPATCHABLE)[:3]
    payload = {"env_version": "twoclock-env-1", "days": DAYS,
               "test_seeds": list(TEST_SEEDS), "train_seeds": list(TRAIN_SEEDS),
               "mu_suff": MU_SUFF, "identified_set_guess": [MU_SUFF, 4 * MU_SUFF],
               "pvg_hyperparams": PVG_KW,
               "wear_coef_mwh_per_mu": coef.tolist(), "envs": {}}

    print("== constant price, calibrated on normal training seeds (stage-1 protocol) ==")
    mu_star, curve = calibrate_mu(net, aging, seeds=TRAIN_SEEDS, grid=CAL_GRID)
    print("   mu* =", mu_star)
    payload["mu_star"] = mu_star
    payload["mu_star_curve"] = curve
    payload["mu_star_train_steps"] = len(CAL_GRID) * len(TRAIN_SEEDS) * STEPS_PER_EP

    print("== old ridge value-gradient learner (stage-1 protocol) ==")
    vg = train_value_agent(net, aging)
    payload["vg_old"] = {
        "mu": vg.mu.tolist(),
        "mu_max": vg.mu_max, "price_clip": vg.price_clip,
        "mu_cap_hit": bool(np.any(vg.mu >= vg.mu_max - 1e-6)),
        "wear_price_mwh_unclipped": wear_price_from_mu(
            vg.mu, aging, GEN_RATING, GEN_DISPATCHABLE).tolist(),
        "train_steps": 3 * 3 * STEPS_PER_EP,
    }

    print("== diagnostic (training seeds): old saturated price with/without clip ==")
    diag = {}
    for env_name in ("hard", "peak"):
        diag[env_name] = {}
        for label, ag in (("mu2e5_clip400", ClippedPriceOPF(net, aging, 2e5, 400.0)),
                          ("mu2e5_noclip", OPFAware(net, aging, 2e5)),
                          ("mu1e4_clip400", ClippedPriceOPF(net, aging, 1e4, 400.0)),
                          ("mu1e4", OPFAware(net, aging, 1e4))):
            r = evaluate(ag, env_name, TRAIN_SEEDS, days=DAYS)
            diag[env_name][label] = r["episode_cost"]["mean"]
            print(f"   {env_name:6s} {label:14s} {r['episode_cost']['mean'] / 1e6:.4f} M$")
    payload["clip_diagnostic_train"] = diag

    for env_name in ENVS:
        print(f"\n## {env_name}")
        e = {}
        # reference: per-regime constant calibration on training seeds
        mu_reg, curve_reg = calibrate_mu(net, aging, env_name=env_name,
                                         seeds=TRAIN_SEEDS, grid=CAL_GRID)
        e["mu_regime"] = mu_reg
        e["mu_regime_curve"] = curve_reg

        tt = time.time()
        pvg = PathwisePriceAgent(net, aging, seed=0)
        info = pvg.train_bisect(env_name, TRAIN_SEEDS, days=DAYS, **PVG_KW)
        mu_l = pvg.mu[:3].copy()
        e["pvg"] = {
            "mu": mu_l.tolist(),
            "mu_mean": float(mu_l.mean()),
            "wear_price_mwh": (mu_l * coef).tolist(),
            "wear_price_mwh_at_mu_star": (mu_star * coef).tolist(),
            "cap_hit": info["cap_hit"],
            "in_identified_set_per_unit": [bool(MU_SUFF <= m <= 4 * MU_SUFF) for m in mu_l],
            "train_steps": info["train_steps"],
            "train_seconds": time.time() - tt,
            "history": info["history"],
        }
        print(f"   pvg mu = {np.round(mu_l)}  ({info['train_steps']} env steps)")

        agents = [("opf_blind", OPFBlind(net, aging)),
                  ("opf_mu_star", OPFAware(net, aging, mu_star)),
                  ("vg_old", vg),
                  ("pvg", pvg),
                  ("opf_mu_regime", OPFBlind(net, aging) if mu_reg == 0
                   else OPFAware(net, aging, mu_reg))]
        res = {}
        for label, ag in agents:
            res[label] = evaluate(ag, env_name, TEST_SEEDS, days=DAYS)
            c = res[label]["episode_cost"]
            print(f"   {label:14s} {c['mean'] / 1e6:7.4f} +- {c['ci95'] / 1e6:.4f} M$  "
                  f"unserved {res[label]['unserved_mwh']['mean']:7.2f} MWh  "
                  f"wear {res[label]['wear_end_mean']['mean']:.4f}")
        e["results"] = {k: _brief(v) for k, v in res.items()}
        e["paired"] = {
            "pvg_minus_mu_star": paired_delta(res["pvg"], res["opf_mu_star"]),
            "pvg_minus_blind": paired_delta(res["pvg"], res["opf_blind"]),
            "pvg_minus_vg_old": paired_delta(res["pvg"], res["vg_old"]),
            "vg_old_minus_mu_star": paired_delta(res["vg_old"], res["opf_mu_star"]),
        }
        d = e["paired"]["pvg_minus_mu_star"]["episode_cost"]
        print(f"   pvg - mu*: {d['mean'] / 1e3:+.1f} k$ +- {d['ci95'] / 1e3:.1f} (paired)")
        payload["envs"][env_name] = e

    payload["wall_seconds"] = time.time() - t0
    path = Results().save("r1_learner.json", payload)
    print(f"\nwrote {path}  ({payload['wall_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
