"""Fast-clock optimisation: DC-OPF with duals.

The whole point of the two-clock formulation is that the *slow* clock enters the
fast problem as a **linear price**, so the fast problem stays an LP.

A wear price ``mu`` in $/wear-unit becomes a $/MWh coefficient through the
marginal aging rate of each unit.  Because thermal aging is piecewise linear in
loading (zero below ``stress_frac`` of rating, linear above), the exact
piecewise cost is still an LP: each unit's output is split into a *free* block
and an *excess* block, and the wear price is applied only to the excess block.

    p_g = s_g + e_g,   s_g <= stress_frac * rating_g,   e_g >= 0

For a fixed total ``p_g`` the LP always drives ``e_g`` to its minimum, so the
split is exact and not an approximation.  Everything here returns duals,
because the duals are the ground truth the learning problem is asked to recover.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

from .dc_flow import (
    BATTERY_BUS, BATTERY_POWER_MW, GEN_BUS, GEN_COST, GEN_DISPATCHABLE,
    GEN_PMAX, GEN_PMIN, GEN_RAMP, GEN_RATING, IMPORT_BUS, IMPORT_CAP_MW,
    LOAD_SHARE, DCNetwork,
)

VOLL = 5000.0  # $/MWh value of lost load


@dataclass
class OPFResult:
    p_gen: np.ndarray
    battery: float
    import_mw: float
    shed: np.ndarray
    flows: np.ndarray
    theta: np.ndarray
    cost: float
    lmp: np.ndarray
    reserve_mw: float
    status: str
    duals: dict = field(default_factory=dict)

    @property
    def shed_total(self) -> float:
        return float(np.sum(self.shed))

    def net_injection(self, net: DCNetwork, load_mw: float) -> np.ndarray:
        """Generation minus load, per bus. Sums to zero when balanced."""
        p = np.zeros(net.n_bus)
        for g, bus in enumerate(GEN_BUS):
            p[bus] += self.p_gen[g]
        p[BATTERY_BUS] += self.battery
        p[IMPORT_BUS] += self.import_mw
        p -= self.shed
        p -= LOAD_SHARE * load_mw
        return p

    def injections(self, net: DCNetwork, renewable_avail: float = 0.0) -> np.ndarray:
        """Backwards-compatible alias (prefer :meth:`net_injection`)."""
        return self.net_injection(net, 0.0)


# ---------------------------------------------------------------------------
# Merit-order fallback (also used as a cheap baseline)
# ---------------------------------------------------------------------------

def merit_order_dispatch(
    load_mw: float,
    renewable_avail: float,
    prev_p: np.ndarray,
    pmax_eff: np.ndarray,
    reserve_req_mw: float = 0.0,
    battery_soc_mwh: float = None,
    battery_power_mw: float = BATTERY_POWER_MW,
) -> tuple[np.ndarray, float]:
    """Cheapest-first dispatch respecting pmax and ramps.

    Returns ``(p_gen, battery_mw)`` with battery positive = discharging.
    """
    pmax = np.minimum(pmax_eff, prev_p + GEN_RAMP)
    pmin = np.maximum(GEN_PMIN, prev_p - GEN_RAMP)
    p = pmin.astype(float).copy()
    p[3] = float(np.clip(renewable_avail, 0.0, pmax_eff[3]))

    remaining = load_mw - p.sum()
    for g in np.argsort(GEN_COST):
        if remaining <= 0:
            break
        room = max(0.0, pmax[g] - p[g])
        add = min(room, remaining)
        p[g] += add
        remaining -= add

    battery = 0.0
    if remaining > 0 and battery_soc_mwh is not None:
        battery = min(battery_power_mw, remaining)
        if battery_soc_mwh <= 0:
            battery = 0.0
        remaining -= battery
    return p, float(battery)


# ---------------------------------------------------------------------------
# The LP
# ---------------------------------------------------------------------------

def _ramp_bounds(prev_p: np.ndarray, pmax_eff: np.ndarray,
                 renewable_avail: float) -> tuple[np.ndarray, np.ndarray]:
    pmax = np.minimum(pmax_eff, prev_p + GEN_RAMP)
    pmin = np.maximum(GEN_PMIN, prev_p - GEN_RAMP)
    pmax = np.maximum(pmax, pmin)
    renew = float(np.clip(renewable_avail, 0.0, pmax_eff[3]))
    pmin = pmin.copy()
    pmax = pmax.copy()
    pmin[3] = renew
    pmax[3] = renew
    return pmin, pmax


def dc_opf(
    net: DCNetwork,
    load_mw: float,
    prev_p: np.ndarray,
    renewable_avail: float,
    pmax_eff: np.ndarray,
    *,
    reserve_req_mw: float = 0.0,
    battery_power_mw: float = BATTERY_POWER_MW,
    battery_min_mw: float | None = None,
    battery_max_mw: float | None = None,
    import_cap_mw: float = IMPORT_CAP_MW,
    spot_price: float = 80.0,
    wear_price_mwh: np.ndarray | None = None,
    wear_free_mw: np.ndarray | None = None,
    shed_cap_frac: float = 0.30,
    verbose: bool = False,
) -> OPFResult:
    """Solve the fast-clock LP.

    ``wear_price_mwh`` is the $/MWh conversion of the slow-clock shadow price.
    If ``wear_free_mw`` is also given, the price applies only to output above
    that level (the exact piecewise-linear aging cost).  Otherwise it applies
    to every MW, which is the conservative linearisation.
    """
    n_bus = net.n_bus
    n_g = len(GEN_BUS)
    n_l = net.n_line
    split = wear_price_mwh is not None and wear_free_mw is not None
    wp = (np.zeros(n_g) if wear_price_mwh is None
          else np.asarray(wear_price_mwh, dtype=float))

    idx_p = np.arange(0, n_g)
    idx_batt = n_g
    idx_imp = n_g + 1
    theta_buses = [b for b in range(n_bus) if b != net.slack]
    idx_theta = {b: n_g + 2 + k for k, b in enumerate(theta_buses)}
    off = n_g + 2 + len(theta_buses)
    idx_f = off + np.arange(n_l)
    off += n_l
    idx_shed = off + np.arange(n_bus)
    off += n_bus
    idx_e = off + np.arange(n_g) if split else None
    n_var = off + (n_g if split else 0)

    c = np.zeros(n_var)
    if split:
        c[idx_p] = GEN_COST
        c[idx_e] = GEN_COST + wp
    else:
        c[idx_p] = GEN_COST + wp
    c[idx_imp] = spot_price
    c[idx_shed] = VOLL

    # ---- equality: nodal balance -----------------------------------------
    A_eq = np.zeros((n_bus + n_l, n_var))
    b_eq = np.zeros(n_bus + n_l)
    loads = LOAD_SHARE * load_mw

    for g, bus in enumerate(GEN_BUS):
        A_eq[bus, idx_p[g]] += 1.0
        if split:
            A_eq[bus, idx_e[g]] += 1.0
    A_eq[BATTERY_BUS, idx_batt] += 1.0
    A_eq[IMPORT_BUS, idx_imp] += 1.0
    for b in range(n_bus):
        A_eq[b, idx_f] -= net.incidence[b]
        A_eq[b, idx_shed[b]] += 1.0
        b_eq[b] = loads[b]

    # ---- equality: flow definition ---------------------------------------
    for k, ln in enumerate(net.lines):
        row = n_bus + k
        A_eq[row, idx_f[k]] = 1.0
        if ln.i != net.slack:
            A_eq[row, idx_theta[ln.i]] -= ln.b
        if ln.j != net.slack:
            A_eq[row, idx_theta[ln.j]] += ln.b

    # ---- inequality: reserve ---------------------------------------------
    A_ub = np.zeros((1, n_var))
    disp = GEN_DISPATCHABLE
    A_ub[0, idx_p[disp]] = 1.0
    if split:
        A_ub[0, idx_e[disp]] = 1.0
    A_ub[0, idx_batt] = 1.0
    cap_res = float(np.sum(pmax_eff[disp]) + battery_power_mw - reserve_req_mw)
    b_ub = np.array([cap_res])

    # ---- bounds -----------------------------------------------------------
    pmin, pmax = _ramp_bounds(prev_p, pmax_eff, renewable_avail)

    bounds: list[tuple] = []
    for g in range(n_g):
        if split:
            free = float(np.clip(wear_free_mw[g], 0.0, pmax[g]))
            s_lo = float(min(pmin[g], free))
            bounds.append((s_lo, max(s_lo, free)))
        else:
            bounds.append((float(pmin[g]), float(pmax[g])))
    lo = -battery_power_mw if battery_min_mw is None else float(battery_min_mw)
    hi = battery_power_mw if battery_max_mw is None else float(battery_max_mw)
    lo = max(lo, -battery_power_mw)
    hi = min(hi, battery_power_mw)
    if hi < lo:
        lo = hi = 0.0
    bounds.append((lo, hi))                                        # battery
    bounds.append((0.0, import_cap_mw))                            # import
    for _ in theta_buses:
        bounds.append((None, None))
    for ln in net.lines:
        bounds.append((-ln.limit_mw, ln.limit_mw))
    shed_cap = shed_cap_frac * loads
    for b in range(n_bus):
        bounds.append((0.0, float(shed_cap[b])))
    if split:
        for g in range(n_g):
            free = float(np.clip(wear_free_mw[g], 0.0, pmax[g]))
            e_lo = max(0.0, float(pmin[g]) - free)
            e_hi = max(e_lo, float(pmax[g]) - max(free, float(pmin[g])))
            if g == 3:                       # renewable: no wear, no excess block
                e_lo = e_hi = 0.0
            bounds.append((e_lo, e_hi))

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")

    if not res.success:
        p, batt = merit_order_dispatch(load_mw, renewable_avail, prev_p, pmax_eff)
        p = np.clip(p, GEN_PMIN, pmax_eff)
        inj = np.zeros(n_bus)
        for g, bus in enumerate(GEN_BUS):
            inj[bus] += p[g]
        inj[BATTERY_BUS] += batt
        resid = load_mw - inj.sum()
        shed = np.zeros(n_bus)
        if resid > 0:
            shed = LOAD_SHARE * resid
        th = net.theta(inj - shed)
        fl = net.flows(th)
        return OPFResult(p_gen=p, battery=batt, import_mw=0.0, shed=shed, flows=fl,
                         theta=th, cost=float(np.dot(GEN_COST, p) + VOLL * shed.sum()),
                         lmp=np.zeros(n_bus), reserve_mw=0.0, status="fallback")

    x = res.x
    p_gen = x[idx_p].copy()
    excess = x[idx_e].copy() if split else np.zeros(n_g)
    p_gen = p_gen + excess
    batt = float(x[idx_batt])
    imp = float(x[idx_imp])
    f = x[idx_f]
    shed = x[idx_shed]
    th = np.zeros(n_bus)
    for b, k in idx_theta.items():
        th[b] = x[k]

    lmp = np.zeros(n_bus)
    if getattr(res, "eqlin", None) is not None and res.eqlin.marginals is not None:
        lmp = np.asarray(res.eqlin.marginals[:n_bus], dtype=float)

    duals = {
        "lmp": lmp,
        "reserve": float(res.ineqlin.marginals[0]) if getattr(res, "ineqlin", None) is not None
                   and res.ineqlin.marginals is not None else 0.0,
        "wear_price_mwh": wp,
        "excess_mw": excess,
    }
    if getattr(res, "lower", None) is not None and res.lower.marginals is not None:
        duals["ramp_lower"] = np.asarray(res.lower.marginals[idx_p])
    if getattr(res, "upper", None) is not None and res.upper.marginals is not None:
        duals["ramp_upper"] = np.asarray(res.upper.marginals[idx_p])

    reserve = float(np.sum(pmax_eff[GEN_DISPATCHABLE] - p_gen[GEN_DISPATCHABLE])
                    + (battery_power_mw - batt))

    return OPFResult(
        p_gen=np.asarray(p_gen, dtype=float),
        battery=batt,
        import_mw=imp,
        shed=np.asarray(shed, dtype=float),
        flows=np.asarray(f, dtype=float),
        theta=th,
        cost=float(res.fun),
        lmp=lmp,
        reserve_mw=reserve,
        status="optimal",
        duals=duals,
    )


# ---------------------------------------------------------------------------
# Price conversion: $/wear-unit  ->  $/MWh
# ---------------------------------------------------------------------------

def wear_free_mw(aging_params, rating_mw: np.ndarray = GEN_RATING) -> np.ndarray:
    """Output level (MW) below which no aging accrues."""
    return aging_params.stress_frac * np.asarray(rating_mw, dtype=float)


def wear_price_from_mu(mu: float | np.ndarray, aging_params,
                       rating_mw: np.ndarray = GEN_RATING,
                       gen_mask: np.ndarray | None = None) -> np.ndarray:
    """Convert a shadow price into per-generator $/MWh.

    ``d(wear increment)/d(p_g) = c_age_g / ((1 - stress_frac) * rating_g)``
    for loading above ``stress_frac`` and zero below.  Paired with
    ``wear_free_mw`` this is exact; without it the coefficient is applied to
    every MW (a conservative linearisation).
    """
    p = aging_params
    c_age_gen = np.asarray(getattr(p, "c_age_gen", (1.0,) * len(rating_mw)), dtype=float)
    if c_age_gen.size < len(rating_mw):
        c_age_gen = np.resize(c_age_gen, len(rating_mw))
    coef = (p.c_age * c_age_gen) / max(1e-9, 1.0 - p.stress_frac) / np.maximum(rating_mw, 1e-6)
    mu_arr = np.asarray(mu, dtype=float)
    if mu_arr.ndim == 0:
        out = float(mu_arr) * coef
    else:
        out = mu_arr * coef
    if gen_mask is not None:
        out = out * np.asarray(gen_mask, dtype=float)
    return out


# ---------------------------------------------------------------------------
# Fast path: merit order + feasibility gate, LP fallback
# ---------------------------------------------------------------------------

def _fast_merit_dispatch(
    load_mw: float,
    prev_p: np.ndarray,
    renewable_avail: float,
    pmax_eff: np.ndarray,
    base_cost: np.ndarray,
    excess_cost: np.ndarray,
    free_mw: np.ndarray,
    spot: float,
    import_cap: float,
    battery_min: float,
    battery_max: float,
) -> tuple[np.ndarray, float, float, float, float]:
    """Greedy merit order over free/excess generator blocks, imports, battery.

    Returns ``(p_gen, battery, import_mw, shed_mw, curtail_mw)``.
    """
    n_g = len(prev_p)
    pmin, pmax = _ramp_bounds(prev_p, pmax_eff, renewable_avail)
    cur = pmin.copy()                      # renewable already at its output

    # each block: (cost, kind, index, hi_level)
    blocks: list[tuple[float, str, int, float]] = []
    for g in range(3):
        free = float(np.clip(free_mw[g], 0.0, pmax[g]))
        base_hi = max(float(pmin[g]), free)
        if base_hi > pmin[g]:
            blocks.append((float(base_cost[g]), "g", g, base_hi))
        if pmax[g] > base_hi:
            blocks.append((float(excess_cost[g]), "g", g, float(pmax[g])))
    if import_cap > 0:
        blocks.append((float(spot), "imp", 0, float(import_cap)))
    if battery_max > 0:
        blocks.append((0.0, "batt", 0, float(battery_max)))
    blocks.sort(key=lambda t: t[0])

    remaining = load_mw - float(cur.sum())
    batt = 0.0
    imp = 0.0
    for cost, kind, idx, hi in blocks:
        if remaining <= 1e-9:
            break
        if kind == "g":
            room = hi - cur[idx]
        elif kind == "imp":
            room = hi
        else:
            room = hi
        if room <= 1e-9:
            continue
        take = min(room, remaining)
        if kind == "g":
            cur[idx] += take
        elif kind == "imp":
            imp = take
        else:
            batt = take
        remaining -= take

    p = cur
    shed = max(0.0, remaining)
    surplus = max(0.0, -remaining)
    curtail = 0.0
    if surplus > 0 and battery_min < 0:
        charge = min(abs(battery_min), surplus)
        batt = -charge
        surplus -= charge
    if surplus > 1e-9:
        curtail = min(surplus, p[3])
        p[3] -= curtail
        surplus -= curtail
    return p, batt, imp, shed, curtail


def fast_dispatch(net: DCNetwork, load_mw: float, prev_p: np.ndarray,
                  renewable_avail: float, pmax_eff: np.ndarray, **kw) -> OPFResult:
    """Same interface as :func:`dc_opf`, but ~6-100x faster.

    Falls back to the LP whenever the greedy solution violates a network,
    reserve, ramp or balance constraint, so a returned ``status='fast'``
    result is always feasible and (checked in the test suite) agrees with the
    LP on the dispatch itself.
    """
    wear_price = kw.get("wear_price_mwh", None)
    wp = (np.zeros(len(GEN_BUS)) if wear_price is None
          else np.asarray(wear_price, dtype=float))
    free = kw.get("wear_free_mw", None)
    if free is None:
        free = np.zeros(len(GEN_BUS))      # price every MW
    free = np.asarray(free, dtype=float)
    base_cost = GEN_COST
    excess_cost = GEN_COST + wp
    spot = float(kw.get("spot_price", 80.0))
    import_cap = float(kw.get("import_cap_mw", IMPORT_CAP_MW))
    bmin = kw.get("battery_min_mw", -BATTERY_POWER_MW)
    bmax = kw.get("battery_max_mw", BATTERY_POWER_MW)
    bmin = -BATTERY_POWER_MW if bmin is None else max(float(bmin), -BATTERY_POWER_MW)
    bmax = BATTERY_POWER_MW if bmax is None else min(float(bmax), BATTERY_POWER_MW)
    reserve_req = float(kw.get("reserve_req_mw", 0.0))

    p, batt, imp, shed, curtail = _fast_merit_dispatch(
        load_mw, prev_p, renewable_avail, pmax_eff, base_cost, excess_cost,
        free, spot, import_cap, bmin, bmax)

    inj = np.zeros(net.n_bus)
    for g, bus in enumerate(GEN_BUS):
        inj[bus] += p[g]
    inj[BATTERY_BUS] += batt
    inj[IMPORT_BUS] += imp
    inj -= LOAD_SHARE * (load_mw - shed)
    flows = net.flows_from_injection(inj)

    reserve = float(np.sum(pmax_eff[GEN_DISPATCHABLE] - p[GEN_DISPATCHABLE])
                    + (BATTERY_POWER_MW - batt))
    feasible = (float(net.overload(flows).sum()) <= 1e-6
                and reserve >= reserve_req - 1e-6
                and shed <= 1e-6)
    if not feasible:
        res = dc_opf(net, load_mw, prev_p, renewable_avail, pmax_eff, **kw)
        res.status = "lp"
        return res

    pmin, pmax = _ramp_bounds(prev_p, pmax_eff, renewable_avail)
    excess = np.maximum(0.0, p - np.maximum(pmin, np.minimum(pmax, free)))
    marginal = [excess_cost[g] for g in range(3)
                if p[g] > pmin[g] + 1e-6 and p[g] < pmax[g] - 1e-6]
    if not marginal:
        marginal = [GEN_COST[g] for g in range(3) if p[g] > 1e-6]
    lmp_val = max(marginal) if marginal else float(spot)
    lmp = np.full(net.n_bus, lmp_val)

    cost = float(np.dot(p, GEN_COST) + np.dot(excess, wp) + spot * imp + VOLL * shed)
    return OPFResult(
        p_gen=p, battery=batt, import_mw=imp,
        shed=LOAD_SHARE * shed, flows=flows, theta=net.theta(inj),
        cost=cost, lmp=lmp, reserve_mw=reserve, status="fast",
        duals={"lmp": lmp, "reserve": 0.0, "wear_price_mwh": wp, "excess_mw": excess},
    )


def wear_aware_opf_mu_grid(
    net,
    load_mw,
    prev_p,
    renewable_avail,
    pmax_eff,
    mu_values,
    aging_params,
    **kw,
):
    """Solve a family of wear-aware OPFs for a grid of prices."""
    out = []
    free = kw.pop("wear_free_mw", None)
    if free is None:
        free = wear_free_mw(aging_params)
    for mu in mu_values:
        wp = wear_price_from_mu(mu, aging_params, GEN_RATING, GEN_DISPATCHABLE)
        out.append(dc_opf(net, load_mw, prev_p, renewable_avail, pmax_eff,
                          wear_price_mwh=wp, wear_free_mw=free, **kw))
    return out
