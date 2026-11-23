"""R10 -- numerical check of Proposition 2 (per-step price band).

On random fast-clock states harvested from env rollouts (normal / peak /
tight demand, several prices) we solve the certificate's priced LP on a dense
mu sweep and check
  (a) the wear rate  w(mu) = sum_g alpha_g e_g*(mu)  is non-increasing
      (and count component-wise violations of e_g*(mu));
  (b) the last breakpoint  m_t = inf{mu : e*(mu) = e_lo}  (bisection) equals
      the dual formula  max_g (pi0_{b(g)} - c_g)/alpha_g,  pi0 = LMP0 + reserve
      dual of the excess-blocked LP (also reported with LMP0 alone, as in R6);
  (c) per unit g: the mu above which e_g = e_lo,g equals
      (pi^{(-g)}_{b(g)} - c_g)/alpha_g  (LP with only g blocked, others priced
      at that mu), and equals (VOLL - c_g)/alpha_g whenever g is indispensable
      (blocking g sheds load) with shedding at the margin.

Run from research/r10_theory:  python -m twoclock.experiments.r10_verify
"""
from __future__ import annotations

import json
import os

import numpy as np

import twoclock.agents.certificate as cert
import twoclock.physics.dc_flow as dcf
import twoclock.physics.opf as opf_mod
from twoclock.agents import OPFAware, OPFBlind
from twoclock.agents.baselines import _battery_bounds
from twoclock.envs import make_env
from twoclock.paths import RESULTS_DIR
from twoclock.physics import AgingParams, build_5bus
from twoclock.physics.opf import wear_free_mw

OUT = os.path.join(RESULTS_DIR, "r10_verify.json")
RNG = np.random.default_rng(10)
TOL_E = 1e-6          # MW
MU_MAX = 3e6

# capture the raw HiGHS result of priced_lp (to read the reserve dual)
_last = {}
_orig_linprog = cert.linprog


def _cap_linprog(*a, **k):
    r = _orig_linprog(*a, **k)
    _last["res"] = r
    return r


cert.linprog = _cap_linprog


class Recorder:
    def __init__(self, inner):
        self.inner, self.ctxs = inner, []
        self.name = inner.name

    def reset(self, env=None):
        self.inner.reset(env)

    def act(self, obs):
        self.ctxs.append({k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
                          for k, v in obs.ctx.items()})
        return self.inner.act(obs)


def harvest(net, aging):
    from twoclock.experiments.runner import run_episode
    states = []
    for diff, ds, mu, seed, n in [("normal", 1.0, 0.0, 201, 8), ("normal", 1.0, 1e4, 202, 8),
                                  ("peak", 1.0, 1e4, 203, 10), ("tight", 1.25, 1e4, 204, 12),
                                  ("tight", 1.35, 3e3, 205, 12)]:
        env = make_env("normal" if diff == "tight" else diff, days=3, seed=seed, demand_scale=ds)
        env.net = net
        ag = OPFBlind(net, aging) if mu <= 0 else OPFAware(net, aging, mu)
        rec = Recorder(ag)
        run_episode(env, rec)
        idx = RNG.choice(len(rec.ctxs), size=n, replace=False)
        # bias half the tight picks toward the evening peak (highest load)
        if diff in ("tight", "peak"):
            loads = np.array([c["load"] for c in rec.ctxs])
            top = np.argsort(loads)[-3 * n:]
            idx = np.concatenate([idx[: n // 2], RNG.choice(top, size=n - n // 2, replace=False)])
        for i in idx:
            states.append({"regime": f"{diff}{ds:g}", "mu_run": mu, "step": int(i),
                           "ctx": rec.ctxs[int(i)]})
    return states


def make_solver(net, aging, ctx):
    free = wear_free_mw(aging)
    lo, hi = _battery_bounds(ctx["soc"], 0.25)
    kw = dict(reserve_req_mw=ctx["reserve_req"], battery_min_mw=lo, battery_max_mw=hi,
              spot_price=ctx["spot"], wear_free=free)
    args = (net, ctx["load"], ctx["prev_p"], ctx["renewable"], ctx["pmax_eff"])
    alpha = cert.alpha_coef(aging)

    def solve(mu, block=None, mu_vec=None):
        wp = alpha * (mu if mu_vec is None else mu_vec)
        r = cert.priced_lp(*args, wear_price_mwh=wp, block=block, **kw)
        res = _last["res"]
        r["y_res"] = float(res.ineqlin.marginals[0]) if r["status"] == "ok" else np.nan
        return r
    return solve, alpha


def last_switch(f, lo, hi, iters=60):
    """f(mu) True  <=> 'still using excess'.  f(lo) True, f(hi) False."""
    for _ in range(iters):
        mid = np.sqrt(lo * hi) if lo > 0 else 0.5 * (lo + hi)
        if f(mid):
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-9 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def main():
    net = build_5bus()
    aging = AgingParams()
    c = np.asarray(opf_mod.GEN_COST, float)
    VOLL = float(opf_mod.VOLL)
    gbus = opf_mod.GEN_BUS
    states = harvest(net, aging)
    grid = np.concatenate([[0.0], np.logspace(0, np.log10(MU_MAX), 400)])
    out_states = []
    agg = {"n_states": 0, "n_sweep_points": len(grid), "wear_rate_monotone_violations": 0,
           "max_wear_rate_increase": 0.0, "componentwise_increase_events": 0,
           "max_componentwise_increase_mw": 0.0,
           "max_rel_err_m_t_formula_with_reserve_dual": 0.0,
           "max_rel_err_m_t_formula_LMP0_only": 0.0,
           "n_reserve_binding_in_LP0": 0, "n_LP0_sheds": 0, "n_LP0_infeasible": 0,
           "n_states_with_discretionary_breakpoint": 0,
           "unit_switch_checks": 0, "max_rel_err_unit_switch_formula": 0.0,
           "indispensable_checks": 0, "max_rel_err_indispensable_vs_VOLL_formula": 0.0,
           "indispensable_mismatches": []}
    for k, st in enumerate(states):
        solve, alpha = make_solver(net, aging, st["ctx"])
        rs = [solve(m) for m in grid]
        if any(r["status"] != "ok" for r in rs):
            continue
        agg["n_states"] += 1
        E = np.array([r["excess"][:3] for r in rs])
        elo = rs[0]["e_lo"][:3]
        ehi = rs[0]["e_hi"][:3]
        w = E @ alpha[:3]
        dw = np.diff(w)
        viol = dw > 1e-7 * max(1.0, np.abs(w).max())
        agg["wear_rate_monotone_violations"] += int(viol.sum())
        agg["max_wear_rate_increase"] = max(agg["max_wear_rate_increase"], float(max(dw.max(), 0)))
        dE = np.diff(E, axis=0)
        agg["componentwise_increase_events"] += int((dE > 1e-6).sum())
        agg["max_componentwise_increase_mw"] = max(agg["max_componentwise_increase_mw"],
                                                   float(max(dE.max(), 0)))
        disc = E - elo[None, :]
        head = (ehi - elo) > 1e-6
        rec = {"regime": st["regime"], "step": st["step"], "load": float(st["ctx"]["load"]),
               "excess_at_mu0": disc[0].tolist(), "headroom": (ehi - elo).tolist()}
        # (b) last breakpoint vs dual formula on LP0
        r0 = solve(0.0, block=np.array([1, 1, 1, 0], bool))
        if r0["status"] != "ok":
            agg["n_LP0_infeasible"] += 1
            rec["LP0"] = "infeasible"
        else:
            lmp0 = r0["lmp"][gbus[:3]]
            pi0 = lmp0 + r0["y_res"]
            if abs(r0["y_res"]) > 1e-9:
                agg["n_reserve_binding_in_LP0"] += 1
            if r0["shed"] > TOL_E:
                agg["n_LP0_sheds"] += 1
            hg = [g for g in range(3) if head[g]]
            f_res = max([max(0.0, (pi0[g] - c[g]) / alpha[g]) for g in hg], default=0.0)
            f_lmp = max([max(0.0, (lmp0[g] - c[g]) / alpha[g]) for g in hg], default=0.0)
            using = lambda m: float(np.sum(solve(m)["excess"][:3] - elo)) > TOL_E
            if using(0.0):
                emp = last_switch(using, 0.0, MU_MAX) if not using(MU_MAX) else np.inf
                agg["n_states_with_discretionary_breakpoint"] += 1
            else:
                emp = 0.0
            e_res = abs(emp - f_res) / max(1.0, f_res) if np.isfinite(emp) else np.inf
            e_lmp = abs(emp - f_lmp) / max(1.0, f_lmp) if np.isfinite(emp) else np.inf
            agg["max_rel_err_m_t_formula_with_reserve_dual"] = max(
                agg["max_rel_err_m_t_formula_with_reserve_dual"], e_res)
            agg["max_rel_err_m_t_formula_LMP0_only"] = max(
                agg["max_rel_err_m_t_formula_LMP0_only"], e_lmp)
            rec.update({"LP0_shed": r0["shed"], "y_res": r0["y_res"], "m_t_empirical": emp,
                        "m_t_formula": f_res, "m_t_formula_LMP0_only": f_lmp,
                        "rel_err": e_res})
        # (c) per-unit switch points
        units = []
        for g in range(3):
            if not head[g]:
                continue
            use_g = lambda m, g=g: float(solve(m)["excess"][g] - elo[g]) > TOL_E
            on = disc[:, g] > TOL_E
            if not on.any():
                continue
            i_last = int(np.where(on)[0].max())
            if i_last == len(grid) - 1:
                continue
            sw = last_switch(use_g, grid[i_last], grid[i_last + 1])
            blk = np.zeros(4, bool); blk[g] = True
            rg = solve(sw * (1 + 1e-6), block=blk)
            pig = rg["lmp"][gbus[g]] + rg["y_res"]
            fg = (pig - c[g]) / alpha[g]
            err = abs(sw - fg) / max(1.0, abs(fg))
            agg["unit_switch_checks"] += 1
            agg["max_rel_err_unit_switch_formula"] = max(agg["max_rel_err_unit_switch_formula"], err)
            # indispensable: g blocked (others unpriced) sheds load
            rgf = solve(0.0, block=blk)
            indisp = rgf["status"] != "ok" or rgf["shed"] > TOL_E
            u = {"unit": g, "switch_emp": sw, "switch_formula_dual": fg, "rel_err": err,
                 "indispensable": bool(indisp)}
            if indisp:
                fv = (VOLL - c[g]) / alpha[g]
                ev = abs(sw - fv) / fv
                agg["indispensable_checks"] += 1
                u.update({"VOLL_formula": fv, "rel_err_VOLL": ev,
                          "pi_minus_g_at_switch": float(pig)})
                if ev > 1e-6:
                    agg["indispensable_mismatches"].append(
                        {"state": k, "regime": st["regime"], "unit": g, "switch_emp": sw,
                         "VOLL_formula": fv, "pi_minus_g": float(pig),
                         "lmp_minus_g": float(rg["lmp"][gbus[g]]), "y_res": rg["y_res"],
                         "shed_minus_g": rg["shed"]})
                else:
                    agg["max_rel_err_indispensable_vs_VOLL_formula"] = max(
                        agg["max_rel_err_indispensable_vs_VOLL_formula"], ev)
            units.append(u)
        rec["units"] = units
        out_states.append(rec)
        print(k, st["regime"], {kk: rec.get(kk) for kk in ("m_t_empirical", "m_t_formula",
                                                          "m_t_formula_LMP0_only")},
              [(u["unit"], round(u["switch_emp"], 1), u["indispensable"]) for u in units],
              flush=True)
    res = {"meta": {"alpha": cert.alpha_coef(aging)[:3].tolist(), "gen_cost": c[:3].tolist(),
                    "voll": VOLL, "VOLL_switch": ((VOLL - c[:3]) / cert.alpha_coef(aging)[:3]).tolist(),
                    "grid": "0 + logspace(0, log10(3e6), 400)", "tol_mw": TOL_E,
                    "bisection_rel_tol": 1e-9},
           "summary": agg, "states": out_states}
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps(agg, indent=1, default=float))


if __name__ == "__main__":
    main()
