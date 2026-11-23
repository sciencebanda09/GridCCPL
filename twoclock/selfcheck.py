"""Self-check: the invariants the experiments depend on.

Run:  python -m twoclock.selfcheck
Exits non-zero on the first failure.  Deliberately dependency-free (no pytest).
"""
from __future__ import annotations

import sys

import numpy as np

from .agents import OPFAware, OPFBlind, RandomAgent, RuleBase, ValueGradientAgent
from .envs import make_env
from .physics import (GEN_DISPATCHABLE, GEN_RATING, AgingParams, build_5bus,
                      dc_opf, fast_dispatch, wear_free_mw,
                      wear_price_from_mu)
from .experiments.runner import evaluate, run_episode


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        raise AssertionError(f"{name}: {detail}")
    print(f"  ok  {name} {detail}")


def main() -> int:
    net, aging = build_5bus(), AgingParams()
    free = wear_free_mw(aging)
    rng = np.random.default_rng(0)

    print("fast-clock physics")
    prev = np.array([250.0, 200.0, 190.0, 40.0])
    pm = np.array([350.0, 300.0, 200.0, 40.0])
    r = dc_opf(net, 700.0, prev, 40.0, pm, reserve_req_mw=40.0)
    gen = float(r.p_gen.sum()) + r.import_mw + r.battery - r.shed.sum()
    check("balance is an equality", abs(gen - 700.0) < 1e-6, f"{gen:.6f}")
    check("flows respect limits",
          bool(np.all(np.abs(r.flows) <= net.limits + 1e-6)),
          str(np.round(r.flows, 1)))
    check("LMP equals marginal unit cost",
          abs(r.lmp.mean() - 45.0) < 25.0, f"{r.lmp.mean():.2f} $/MWh")

    print("fast path")
    mism = 0
    devs = []
    for _ in range(120):
        load = float(rng.uniform(450, 760))
        renew = float(rng.uniform(0, 150))
        pme = np.array([350.0, 300.0, 200.0, renew])
        ko = dict(reserve_req_mw=0.04 * load, battery_min_mw=-100.0,
                  battery_max_mw=100.0, spot_price=float(rng.uniform(30, 250)))
        a = dc_opf(net, load, prev, renew, pme, **ko)
        b = fast_dispatch(net, load, prev, renew, pme, **ko)
        ok_balance = abs((b.p_gen.sum() + b.import_mw + b.battery
                          - b.shed.sum()) - load) < 1e-6
        ok_flows = bool(np.all(np.abs(b.flows) <= net.limits + 1e-6))
        if not (ok_balance and ok_flows):
            mism += 1
        devs.append(float(np.abs(a.p_gen - b.p_gen).max()))
    check("fast dispatch is always feasible", mism == 0, f"{mism} violations")
    check("fast dispatch equals the LP on most states",
          float(np.median(devs)) == 0.0,
          f"median |dp| = {np.median(devs):.3f} MW")

    # Verify LP and fast path agree under non-zero piecewise wear prices
    priced_devs = []
    for mu_chk in (1000.0, 5000.0):
        wp_chk = wear_price_from_mu(mu_chk, aging, GEN_RATING, GEN_DISPATCHABLE)
        for _ in range(30):
            load = float(rng.uniform(480, 740))
            renew = float(rng.uniform(0, 140))
            pme = np.array([350.0, 300.0, 200.0, renew])
            ko = dict(reserve_req_mw=0.04 * load, battery_min_mw=-100.0,
                      battery_max_mw=100.0, spot_price=float(rng.uniform(40, 180)),
                      wear_price_mwh=wp_chk, wear_free_mw=free)
            a = dc_opf(net, load, prev, renew, pme, **ko)
            b = fast_dispatch(net, load, prev, renew, pme, **ko)
            priced_devs.append(float(np.abs(a.p_gen - b.p_gen).max()))
    check("priced LP equals fast dispatch",
          float(np.median(priced_devs)) < 1e-4,
          f"median |dp| = {np.median(priced_devs):.4f} MW")

    print("price response")
    blind = evaluate(OPFBlind(net, aging), "normal", (1, 2), days=4)
    priced = evaluate(OPFAware(net, aging, 1e4), "normal", (1, 2), days=4)
    check("priced dispatch is cheaper than blind",
          priced["episode_cost"]["mean"] < blind["episode_cost"]["mean"],
          f"{blind['episode_cost']['mean']/1e6:.3f} -> "
          f"{priced['episode_cost']['mean']/1e6:.3f} M$")
    check("priced dispatch accrues less wear",
          priced["wear_end_mean"]["mean"] < blind["wear_end_mean"]["mean"],
          f"{blind['wear_end_mean']['mean']:.3f} -> "
          f"{priced['wear_end_mean']['mean']:.3f}")
    neg = evaluate(OPFAware(net, aging, -1e4), "normal", (1, 2), days=4)
    check("wrong sign is not helpful",
          neg["episode_cost"]["mean"] >= blind["episode_cost"]["mean"] * 0.999)

    print("learning")
    vg = ValueGradientAgent(net, aging, seed=3, mu_max=1e6)
    vg.enabled = False
    for r_i in range(2):
        for s in (101, 102):
            run_episode(make_env("normal", days=3, seed=s + 10 * r_i), vg, learn=True)
        vg.update()
        vg.enabled = True
    check("value model fits", bool(vg.mu_scalar > 0.0), f"mu = {vg.mu_scalar:.0f}")
    ev = evaluate(vg, "normal", (1, 2), days=4)
    check("learner beats blind", ev["episode_cost"]["mean"] < blind["episode_cost"]["mean"],
          f"{ev['episode_cost']['mean']/1e6:.3f} M$")

    print("baselines are not straw men")
    rule = evaluate(RuleBase(0), "normal", (1, 2), days=4)
    check("rule base is competent but not optimal",
          rule["episode_cost"]["mean"] < 5 * blind["episode_cost"]["mean"],
          f"{rule['episode_cost']['mean']/1e6:.2f} M$")
    rnd = evaluate(RandomAgent(0), "normal", (1,), days=2)
    check("random dispatch is catastrophic",
          rnd["episode_cost"]["mean"] > 10 * blind["episode_cost"]["mean"])

    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
