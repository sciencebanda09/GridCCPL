"""Dual certificate for the wear shadow price.

Idea
----
In the priced fast-clock LP the excess block of unit g (output above
``stress_frac * rating_g``) is offered at ``c_g + alpha_g * mu`` $/MWh.  LP
duality says the block is dispatched only if that offer is below the nodal
price (LMP) at the unit's bus that would prevail *without* the block.  So
from a rollout we can read off, per step t and unit g,

    mu_req(t, g) = (LMP0_{t, b(g)} - c_g) / alpha_g,

where LMP0 is the LMP of the same LP with every excess block removed
(fixed at its ramp-forced lower bound).  Any price above mu_req(t, g)
removes unit g's excess at step t.

* A step is **discretionary** if the excess-free LP still serves all load
  (no shedding); there excess is a pure economic choice and pricing it out is
  free of unserved energy.
* A step is **forced** for unit g if the LP with only g's excess removed (all
  other excess blocks available) sheds load: g's excess is needed for
  adequacy.  If the price makes g's excess dearer than VOLL,
  ``mu > (VOLL - c_g) / alpha_g``, the LP sheds instead -> upper end.

Certificate (scalar):

    mu_lo = max( max_{discretionary t, g with headroom} mu_req(t, g),
                 merit-order crossings among units that are forced )
    mu_hi = min_{g forced somewhere} (VOLL - c_g) / alpha_g     (inf if none)

The crossings term makes the forced excess come from the unit with the
lowest wear per MWh: above mu_x(g,h) = (c_g - c_h)/(alpha_h - alpha_g) the
order of forced excess blocks no longer changes.

Effective cost.  The dispatch LP uses the nameplate fuel cost ``GEN_COST``
(heat-rate drift and derating are *not* in the LP objective; derating enters
through ``pmax_eff``, i.e. through whether a unit has excess headroom at
all).  The certificate is a statement about the LP, so c_eff = GEN_COST and
derating is handled automatically by solving the LP at the realised
``pmax_eff``.

Because LMPs depend on the trajectory (state of charge, ramps, derating), the
certificate is iterated to a fixed point: run an episode at mu_k, compute
mu_lo, set mu_{k+1} = mu_lo, stop when it no longer moves.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from ..physics import dc_flow as dcf
from ..physics import opf as opf_mod
from ..physics.opf import wear_free_mw, wear_price_from_mu
from .baselines import BaseAgent, OPFAware, OPFBlind, _battery_bounds


def alpha_coef(aging) -> np.ndarray:
    """$/MWh per unit of mu, per generator (0 for renewable)."""
    return wear_price_from_mu(1.0, aging, dcf.GEN_RATING, dcf.GEN_DISPATCHABLE)


# ---------------------------------------------------------------------------
# LP mirroring physics.opf.dc_opf with optional blocked excess blocks
# ---------------------------------------------------------------------------

def priced_lp(net, load_mw, prev_p, renewable_avail, pmax_eff, *, reserve_req_mw,
              battery_min_mw, battery_max_mw, spot_price, wear_price_mwh,
              wear_free, block=None, load_bump=None, shed_cap_frac=0.30):
    """Same LP as :func:`dc_opf` (split form).  ``block[g]=True`` fixes unit g's
    excess block at its ramp-forced lower bound.  ``load_bump`` (per bus MW)
    is used only for the finite-difference LMP check.

    Returns dict(cost, lmp, shed, excess, status).
    """
    GEN_BUS = opf_mod.GEN_BUS
    LOAD_SHARE = opf_mod.LOAD_SHARE
    BATTERY_BUS = opf_mod.BATTERY_BUS
    IMPORT_BUS = opf_mod.IMPORT_BUS
    GEN_COST = opf_mod.GEN_COST
    VOLL = opf_mod.VOLL
    bp = opf_mod.BATTERY_POWER_MW
    n_bus, n_g, n_l = net.n_bus, len(GEN_BUS), net.n_line
    wp = np.asarray(wear_price_mwh, dtype=float)
    block = np.zeros(n_g, bool) if block is None else np.asarray(block, bool)

    idx_p = np.arange(n_g)
    idx_batt, idx_imp = n_g, n_g + 1
    theta_buses = [b for b in range(n_bus) if b != net.slack]
    idx_theta = {b: n_g + 2 + k for k, b in enumerate(theta_buses)}
    off = n_g + 2 + len(theta_buses)
    idx_f = off + np.arange(n_l); off += n_l
    idx_shed = off + np.arange(n_bus); off += n_bus
    idx_e = off + np.arange(n_g); n_var = off + n_g

    c = np.zeros(n_var)
    c[idx_p] = GEN_COST
    c[idx_e] = GEN_COST + wp
    c[idx_imp] = spot_price
    c[idx_shed] = VOLL

    A_eq = np.zeros((n_bus + n_l, n_var)); b_eq = np.zeros(n_bus + n_l)
    loads = LOAD_SHARE * load_mw
    for g, bus in enumerate(GEN_BUS):
        A_eq[bus, idx_p[g]] += 1.0
        A_eq[bus, idx_e[g]] += 1.0
    A_eq[BATTERY_BUS, idx_batt] += 1.0
    A_eq[IMPORT_BUS, idx_imp] += 1.0
    for b in range(n_bus):
        A_eq[b, idx_f] -= net.incidence[b]
        A_eq[b, idx_shed[b]] += 1.0
        b_eq[b] = loads[b]
    if load_bump is not None:
        b_eq[:n_bus] += np.asarray(load_bump, dtype=float)
    for k, ln in enumerate(net.lines):
        row = n_bus + k
        A_eq[row, idx_f[k]] = 1.0
        if ln.i != net.slack:
            A_eq[row, idx_theta[ln.i]] -= ln.b
        if ln.j != net.slack:
            A_eq[row, idx_theta[ln.j]] += ln.b

    disp = opf_mod.GEN_DISPATCHABLE
    A_ub = np.zeros((1, n_var))
    A_ub[0, idx_p[disp]] = 1.0
    A_ub[0, idx_e[disp]] = 1.0
    A_ub[0, idx_batt] = 1.0
    b_ub = np.array([float(np.sum(pmax_eff[disp]) + bp - reserve_req_mw)])

    pmin, pmax = opf_mod._ramp_bounds(prev_p, pmax_eff, renewable_avail)
    bounds = []
    for g in range(n_g):
        free = float(np.clip(wear_free[g], 0.0, pmax[g]))
        s_lo = float(min(pmin[g], free))
        bounds.append((s_lo, max(s_lo, free)))
    lo = max(float(battery_min_mw), -bp); hi = min(float(battery_max_mw), bp)
    if hi < lo:
        lo = hi = 0.0
    bounds.append((lo, hi))
    bounds.append((0.0, opf_mod.IMPORT_CAP_MW))
    bounds += [(None, None)] * len(theta_buses)
    bounds += [(-ln.limit_mw, ln.limit_mw) for ln in net.lines]
    shed_cap = shed_cap_frac * loads
    bounds += [(0.0, float(shed_cap[b])) for b in range(n_bus)]
    e_lo_v = np.zeros(n_g); e_hi_v = np.zeros(n_g)
    for g in range(n_g):
        free = float(np.clip(wear_free[g], 0.0, pmax[g]))
        e_lo = max(0.0, float(pmin[g]) - free)
        e_hi = max(e_lo, float(pmax[g]) - max(free, float(pmin[g])))
        if g == 3:
            e_lo = e_hi = 0.0
        if block[g]:
            e_hi = e_lo
        e_lo_v[g], e_hi_v[g] = e_lo, e_hi
        bounds.append((e_lo, e_hi))

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                  method="highs")
    if not res.success:
        return {"status": "fail", "cost": np.nan, "lmp": np.full(n_bus, np.nan),
                "shed": np.nan, "excess": np.zeros(n_g), "e_lo": e_lo_v, "e_hi": e_hi_v}
    return {"status": "ok", "cost": float(res.fun),
            "lmp": np.asarray(res.eqlin.marginals[:n_bus], dtype=float),
            "shed": float(np.sum(res.x[idx_shed])),
            "excess": res.x[idx_e].copy(), "e_lo": e_lo_v, "e_hi": e_hi_v}


def headroom(prev_p, renewable, pmax_eff, free) -> np.ndarray:
    """Discretionary excess headroom per unit (MW above free and ramp floor)."""
    pmin, pmax = opf_mod._ramp_bounds(prev_p, pmax_eff, renewable)
    out = np.zeros(len(prev_p))
    for g in range(3):
        f = float(np.clip(free[g], 0.0, pmax[g]))
        out[g] = max(0.0, float(pmax[g]) - max(f, float(pmin[g])))
    return out


# ---------------------------------------------------------------------------
# Probe agent: dispatches exactly like OPFAware(mu) but records the certificate
# ---------------------------------------------------------------------------

class CertificateProbe(BaseAgent):
    """Acts as ``OPFAware(mu)`` (or blind if mu<=0) and, at every step, solves
    the excess-free LP and the per-unit "g blocked" LPs."""

    def __init__(self, net, aging, mu: float, tol_mw: float = 1e-6, lazy: bool = True):
        self.net, self.aging, self.mu = net, aging, float(mu)
        self.lazy = lazy
        self.inner = OPFBlind(net, aging) if mu <= 0 else OPFAware(net, aging, mu)
        self.name = f"cert_probe_mu{mu:g}"
        self.alpha = alpha_coef(aging)
        self.free = wear_free_mw(aging, dcf.GEN_RATING)
        self.tol = tol_mw
        self.rows: list[dict] = []

    def reset(self, env=None):
        self.rows = []

    def act(self, obs):
        a = self.inner.act(obs)
        ctx = obs.ctx
        lo, hi = _battery_bounds(ctx["soc"], 0.25)
        kw = dict(reserve_req_mw=ctx["reserve_req"], battery_min_mw=lo, battery_max_mw=hi,
                  spot_price=ctx["spot"], wear_free=self.free)
        args = (self.net, ctx["load"], ctx["prev_p"], ctx["renewable"], ctx["pmax_eff"])
        n_g = len(opf_mod.GEN_BUS)
        zero = np.zeros(n_g)
        # excess-free LP (all excess blocked): LMP0
        r0 = priced_lp(*args, wear_price_mwh=zero, block=np.array([1, 1, 1, 0], bool), **kw)
        hr = headroom(ctx["prev_p"], ctx["renewable"], ctx["pmax_eff"], self.free)
        forced = np.zeros(n_g, bool)
        # Lazy mode (exact): unblocking units only relaxes the LP, so if the
        # excess-free LP sheds nothing, no single-unit-blocked LP can shed and
        # the all-excess LP sheds nothing; skip them.  Units with no headroom
        # are masked out of the certificate anyway.
        need_extra = (not self.lazy) or (r0["status"] != "ok") or (r0["shed"] > self.tol)
        if need_extra:
            for g in range(3):
                if self.lazy and hr[g] <= 1e-3:
                    continue
                blk = np.zeros(n_g, bool); blk[g] = True
                rg = priced_lp(*args, wear_price_mwh=zero, block=blk, **kw)
                forced[g] = (rg["status"] == "ok" and rg["shed"] > self.tol) or rg["status"] != "ok"
            ra = priced_lp(*args, wear_price_mwh=zero, block=None, **kw)
        else:
            ra = {"shed": 0.0}
        res = self.inner.last_result
        self.rows.append({
            "lmp0": r0["lmp"][opf_mod.GEN_BUS[:3]].copy(),
            "shed0": r0["shed"], "shed_all": ra["shed"], "ok0": r0["status"] == "ok",
            "forced": forced[:3].copy(), "head": hr[:3].copy(),
            "excess_mu": np.asarray(res.duals.get("excess_mw", np.zeros(4)))[:3].copy()
            if res is not None else np.zeros(3),
            "spot": ctx["spot"],
        })
        return a


def certificate_from_rows(rows: list[dict], aging, head_tol: float = 1e-3,
                          shed_tol: float = 1e-6) -> dict:
    """Scalar and per-unit certificates from probe rows."""
    alpha = alpha_coef(aging)[:3]
    c = np.asarray(opf_mod.GEN_COST, dtype=float)[:3]
    VOLL = float(opf_mod.VOLL)
    lmp0 = np.array([r["lmp0"] for r in rows])
    head = np.array([r["head"] for r in rows])
    disc = np.array([r["ok0"] and r["shed0"] <= shed_tol for r in rows])
    forced = np.array([r["forced"] for r in rows]) & (head > head_tol)
    req = (lmp0 - c[None, :]) / alpha[None, :]
    mask = disc[:, None] & (head > head_tol)
    vals = req[mask]
    vals = np.maximum(vals, 0.0)

    per_unit_lo = np.array([float(np.max(np.maximum(req[mask[:, g], g], 0.0)))
                            if mask[:, g].any() else 0.0 for g in range(3)])
    any_forced = forced.any(axis=0)
    cross = 0.0
    fg = [g for g in range(3) if any_forced[g]]
    for i in fg:
        for j in fg:
            if alpha[j] > alpha[i] and c[i] > c[j]:   # i: cheaper wear, dearer fuel
                cross = max(cross, (c[i] - c[j]) / (alpha[j] - alpha[i]))
    cap_g = (VOLL - c) / alpha
    mu_hi = float(np.min(cap_g[any_forced])) if any_forced.any() else float("inf")
    disc_max = float(np.max(vals)) if vals.size else 0.0
    q = {f"q{int(p)}": float(np.percentile(vals, p)) if vals.size else 0.0
         for p in (50, 90, 95, 99)}
    q["q99.9"] = float(np.percentile(vals, 99.9)) if vals.size else 0.0
    return {
        "mu_lo": max(disc_max, cross),
        "mu_lo_disc_max": disc_max,
        "mu_lo_cross": cross,
        "mu_lo_quantiles": q,
        "mu_hi": mu_hi,
        "per_unit_lo": per_unit_lo.tolist(),
        "per_unit_hi": [float(x) if any_forced[g] else float("inf")
                        for g, x in enumerate(cap_g)],
        "forced_steps_per_unit": forced.sum(axis=0).astype(int).tolist(),
        "n_steps": len(rows),
        "n_discretionary": int(disc.sum()),
        "n_infeasible_even_with_excess": int(sum(r["shed_all"] > shed_tol for r in rows)),
        "argmax_unit": int(np.argmax(np.where(mask, req, -np.inf).max(axis=0)))
        if mask.any() else -1,
        "lmp0_max_disc": float(np.max(lmp0[mask])) if mask.any() else None,
        "excess_mwh_at_mu": float(np.sum([np.sum(r["excess_mu"]) for r in rows]) * 0.25),
    }


def merge_rows(*row_lists):
    out = []
    for r in row_lists:
        out.extend(r)
    return out


def fixed_point_certificate(make_env_fn, net, aging, train_seeds, run_episode,
                            mu0: float = 0.0, max_iter: int = 5, rtol: float = 0.10,
                            log=print) -> dict:
    """Iterate mu_{k+1} = mu_lo(rollout at mu_k), one fresh training seed per
    iteration.  Stop when |log(mu_lo_k / mu_lo_{k-1})| < log(1 + rtol) (and
    mu_hi unchanged).  Final certificate pools the last two rollouts."""
    mu = float(mu0)
    hist = []
    rows_hist = []
    steps = 0
    for k in range(max_iter):
        seed = train_seeds[k % len(train_seeds)]
        probe = CertificateProbe(net, aging, mu)
        ep = run_episode(make_env_fn(seed), probe)
        steps += int(ep["steps"])
        cert = certificate_from_rows(probe.rows, aging)
        rows_hist.append(probe.rows)
        hist.append({"iter": k, "seed": seed, "mu_run": mu, "mu_lo": cert["mu_lo"],
                     "mu_hi": cert["mu_hi"], "cost": ep["episode_cost"],
                     "per_unit_lo": cert["per_unit_lo"],
                     "excess_mwh": cert["excess_mwh_at_mu"]})
        log(f"    iter {k} seed {seed} mu_run={mu:9.1f} -> mu_lo={cert['mu_lo']:9.1f} "
            f"mu_hi={cert['mu_hi']:.4g} excess={cert['excess_mwh_at_mu']:.1f}MWh "
            f"forced={cert['forced_steps_per_unit']}")
        converged = (k >= 1 and mu > 0 and cert["mu_lo"] > 0
                     and abs(np.log(cert["mu_lo"] / mu)) < np.log(1 + rtol))
        if converged:
            break
        mu = max(cert["mu_lo"], 1.0)
    # pool the self-consistent (priced, mu>0) rollouts of the last two iterations;
    # the blind start rollout (mu=0) is excluded because its trajectory (wear,
    # derating) is not the one the certificate price induces.
    priced = [rows_hist[i] for i, h in enumerate(hist) if h["mu_run"] > 0][-2:]
    pooled = [r for rs in (priced or rows_hist[-1:]) for r in rs]
    final = certificate_from_rows(pooled, aging)
    final["history"] = hist
    final["env_steps"] = steps
    final["iterations"] = len(hist)
    final["converged"] = bool(len(hist) < max_iter or converged)
    return final
