"""R9 -- a two-clock certificate: fast-clock VALUE of excess vs slow-clock DAMAGE.

Question
--------
R7 showed the fast-clock dual certificate mu_lo (the smallest price that prices
out *all* discretionary excess) tracks R4's empirical threshold mu_emp poorly
(log-Pearson 0.24): it cannot see the wear dynamics (tau, derating, heat-rate)
and mu_emp is a 0.5%-tolerance point, set where the remaining excess is
economically small.  Here the fast clock supplies the value of excess and a
pathwise (bumped-wear, common-random-number) estimator supplies its damage.

Two-clock predicted cost curve (first order around a trajectory)
----------------------------------------------------------------
Run the R4 policy (OPFAware(mu_k), fast dispatch) on TRAINING seeds 101, 102.
On every visited state s_t re-solve the same dispatch at a fine price grid
mu' (quarter-octave, 125 .. 256000) and record

    F_t(mu')   = fuel + import + VOLL*shed of the dispatch at price mu' ($/h)
    e_tg(mu')  = max(0, p_g - sigma*rating_g)     (wear-producing MW of unit g)

The fast clock says how much the excess used at mu' saves:  F_t(mu_k) - F_t(mu').
The slow clock says what it costs later:  G_g(mu_k) per wear-unit, from bumped-
wear rollouts (w_g += delta at K snapshot states, CRN, policy mu_k, horizon
max(24 h, 4 tau)).  Each MWh of excess adds alpha_g wear units, so

    J_pred(mu') = J(mu_k) + sum_t dt [F_t(mu') - F_t(mu_k)]
                          + sum_g G_g(mu_k) alpha_g sum_t dt [e_tg(mu') - e_tg(mu_k)]

i.e. excess at (t,g) is worth removing iff alpha_g G_g > v_t (its value per
MWh) -- the hypothesis -- and the tolerance prediction is cost-weighted:
gap_pred(mu') = J_pred(mu') / min J_pred - 1.

Rules (declared before looking at test-curve agreement)
-------------------------------------------------------
* mu_2c (PRIMARY, self-consistent tolerance threshold): the predicate
  "gap_pred(mu; G(mu)) <= 0.5%" (damage evaluated at the current price) is
  bisected over the configuration's own R4 test grid; mu_2c_grid = smallest
  passing grid point (exact analogue of R4's mu_emp definition); mu_2c refines
  continuously inside (failing grid point, passing grid point] on the curve of
  the passing evaluation.
* mu_2c_oneshot(_grid): one evaluation only, around the excess-free reference
  trajectory mu_ref = min(64000, 0.5 mu_hi) (R7 start rule).
* mu_2c_opt: argmin of the predicted curve (value = damage price; flat above
  the price that removes all net-damaging excess, so it is <= mu_lo-like).
* mu_min_lo_g: min(mu_lo, g_bar), g_bar = wear-at-stake-weighted G (the simple
  "min(mu_lo, fixed point)" rule; mu_lo from R7's JSON, training seeds).

Run:  python -m twoclock.experiments.r9_twoclock          (full)
      python -m twoclock.experiments.r9_twoclock --plot   (figure from JSON)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from ..agents import OPFAware
from ..agents.baselines import _battery_bounds, action_from_opf
from ..agents.pathwise_agent import clone_env
from ..physics import build_5bus
from ..physics import dc_flow as dcf
from ..physics.opf import fast_dispatch, wear_free_mw, wear_price_from_mu
from ..paths import RESULTS_DIR
from .r4_robust import LHS_GRID, OAT_GRID, Applied, aging_from, make
from .r7_cert_robust import (corr_block, ctrl_eval, curve_from_detail, interp_curve,
                             key_of, mu_hi_of)
from .runner import run_episode

R4_JSON = os.path.join(RESULTS_DIR, "r4_robust.json")
R7_JSON = os.path.join(RESULTS_DIR, "r7_cert_robust.json")
OUT_JSON = os.path.join(RESULTS_DIR, "r9_twoclock.json")
OUT_FIG = os.path.join(RESULTS_DIR, "fig_r9_twoclock.pdf")

TRAIN_SEEDS = (101, 102)
FLAT_TOL = 0.005
K_SNAP = 6                 # pathwise snapshots per training seed
DELTA = 0.05               # wear bump (secant)
TRIM = 0.1                 # symmetric trimmed mean over snapshots
DT = 0.25
FINE = tuple(float(125.0 * 2 ** (k / 2)) for k in range(23))   # 125 .. 256000
MU_BIG = FINE[-1]
MU_START = 64000.0
THERMAL = (0, 1, 2)
SPOT = ["oat:nominal", "oat:tau_thermal=12", "oat:heat=0.5", "oat:c_age=0.275",
        "lhs:draw13", "lhs:draw18"]


def alphas(aging):
    return wear_price_from_mu(1.0, aging, dcf.GEN_RATING, dcf.GEN_DISPATCHABLE)[:3]


def dispatch(net, ctx, aging, free, mu):
    """Exactly OPFAware(mu).act's solve; returns (F $/h without wear term, e (3,))."""
    lo, hi = _battery_bounds(ctx["soc"], 0.25)
    wp = wear_price_from_mu(mu, aging, dcf.GEN_RATING, dcf.GEN_DISPATCHABLE)
    res = fast_dispatch(net, ctx["load"], ctx["prev_p"], ctx["renewable"], ctx["pmax_eff"],
                        reserve_req_mw=ctx["reserve_req"], battery_min_mw=lo, battery_max_mw=hi,
                        spot_price=ctx["spot"], wear_price_mwh=wp, wear_free_mw=free)
    if res.status == "fast":
        ex = np.asarray(res.duals["excess_mw"])
    else:
        ex = np.asarray(res.duals.get("excess_mw", np.zeros(4))) if res.duals else np.zeros(4)
    F = float(res.cost - np.dot(wp, ex))
    e = np.maximum(0.0, np.asarray(res.p_gen)[:3] - free[:3])
    return F, e


class Counter:
    steps = 0


def roll(env, agent, n, counter) -> float:
    c = 0.0
    obs = env._observe()
    for _ in range(n):
        if env.done:
            break
        obs, _, _, info = env.step(agent.act(obs))
        c += info["step_cost"]
        counter.steps += 1
    return c


def evaluate_at(cfg, aging, net, mu_k, rng, counter, horizon) -> dict:
    """Trajectory at mu_k on training seeds + value grid + pathwise G(mu_k)."""
    a = alphas(aging)
    free = wear_free_mw(aging)
    mus = sorted(set(FINE) | {float(mu_k)})
    imk = mus.index(float(mu_k))
    Fsum = np.zeros(len(mus)); Wsum = np.zeros((len(mus), 3))   # per-seed means
    Jk = []
    Gs, Ws = [], []
    stake_tot = np.zeros(3)
    n_cand = 0
    for s in TRAIN_SEEDS:
        env = make(cfg, aging, s)
        obs = env.reset()
        agent = OPFAware(net, aging, float(mu_k))
        clones, stakes = [], []
        while not env.done:
            ctx = obs.ctx
            F0, e0 = dispatch(net, ctx, aging, free, 0.0)
            Fb, eb = dispatch(net, ctx, aging, free, MU_BIG)
            st = float(np.dot(a, e0 - eb))
            if st > 1e-9:
                n_cand += 1
                rows = [dispatch(net, ctx, aging, free, m) for m in mus]
                Fsum += np.array([r[0] for r in rows]) * DT / len(TRAIN_SEEDS)
                Wsum += np.array([r[1] for r in rows]) * a[None, :] * DT / len(TRAIN_SEEDS)
                # wear at stake per unit: excess still used at mu_k (fallback: at 0)
                sk = a * np.maximum(0.0, rows[imk][1] - eb)
                s0 = a * np.maximum(0.0, e0 - eb)
                stake_tot += sk * DT
                clones.append(clone_env(env)); stakes.append((sk, s0))
            obs, _, _, _ = env.step(agent.act(obs))
            counter.steps += 1
        Jk.append(env.episode_summary()["episode_cost"])
        if clones:
            sk = np.array([x[0] for x in stakes]); s0 = np.array([x[1] for x in stakes])
            use_k = sk.sum() > 1e-6 * max(1e-12, s0.sum())
            SU = sk if use_k else s0
            tot = SU.sum(1)
            if tot.sum() <= 0:
                SU = s0; tot = SU.sum(1)
            p = tot / tot.sum()
            k = min(K_SNAP, int(np.count_nonzero(p)))
            pick = rng.choice(len(clones), size=k, replace=False, p=p)
            for i in pick:
                base_env = clones[i]
                ag = OPFAware(net, aging, float(mu_k))
                base = roll(clone_env(base_env), ag, horizon, counter)
                row = []
                for g in THERMAL:
                    e1 = clone_env(base_env)
                    w0 = e1.aging.wear[g]
                    d = DELTA if w0 + DELTA <= 1.0 else -DELTA   # backward secant near w=1
                    e1.aging.wear[g] = w0 + d
                    row.append((roll(e1, ag, horizon, counter) - base) / d)
                Gs.append(row)
                Ws.append(SU[i] / max(tot[i], 1e-300))   # per-unit share of the stake
    Gs = np.asarray(Gs, dtype=float).reshape(-1, 3)
    Ws = np.asarray(Ws, dtype=float).reshape(-1, 3)
    G = np.zeros(3); G_se = np.zeros(3)
    for g in THERMAL:
        # self-normalised importance weights: snapshots drawn ∝ total stake,
        # unit g's damage weighted by its own share of the wear at stake
        w = Ws[:, g] if len(Ws) and Ws[:, g].sum() > 0 else np.ones(len(Gs))
        if len(Gs):
            w = w / w.sum()
            G[g] = max(0.0, float(np.dot(w, Gs[:, g])))
            G_se[g] = float(np.sqrt(np.dot(w ** 2, (Gs[:, g] - np.dot(w, Gs[:, g])) ** 2)))
    J = float(np.mean(Jk))
    Jpred = J + (Fsum - Fsum[imk]) + (Wsum - Wsum[imk]) @ G
    best = float(Jpred.min())
    gap = Jpred / best - 1.0
    w = stake_tot / stake_tot.sum() if stake_tot.sum() > 0 else np.ones(3) / 3
    return {"mu": float(mu_k), "mus": mus, "J_train": J, "G": G.tolist(), "G_se": G_se.tolist(),
            "G_snap": Gs.tolist(), "g_bar": float(np.dot(w, G)), "stake_weights": w.tolist(),
            "n_cand_steps": n_cand, "gap_pred": gap.tolist(),
            "gap_at_mu": float(gap[imk]), "pass": bool(gap[imk] <= FLAT_TOL + 1e-12),
            "mu_opt": float(mus[int(np.argmin(Jpred))]),
            "wear_at_mu": float(Wsum[imk].sum()), "wear_at_0": float(Wsum[mus.index(FINE[0])].sum()),
            "value_all_excess_vs_big": float(Fsum[mus.index(MU_BIG)] - Fsum[0])}


def smallest_pass(ev, lo_excl=None, hi_incl=None, grid=None):
    cand = [(m, g) for m, g in zip(ev["mus"], ev["gap_pred"])
            if (grid is None or m in grid) and (lo_excl is None or m > lo_excl * 1.0001)
            and (hi_incl is None or m <= hi_incl * 1.0001)]
    ok = [m for m, g in cand if g <= FLAT_TOL + 1e-12]
    return min(ok) if ok else None


def two_clock(cfg, block, net, idx) -> dict:
    aging = aging_from(cfg)
    grid = list(OAT_GRID if block == "oat" else LHS_GRID)
    mh = mu_hi_of(cfg, aging)
    mu_ref = MU_START if mh <= 0 else min(MU_START, 0.5 * mh)
    horizon = int(max(96, np.ceil(4 * cfg["tau_thermal"] / DT)))
    rng = np.random.default_rng(9000 + idx)
    counter = Counter()
    evals = {}
    t0 = time.time()
    with Applied(cfg):
        ref = evaluate_at(cfg, aging, net, mu_ref, rng, counter, horizon)
        steps_oneshot = counter.steps
        lo, hi = -1, len(grid) - 1           # top of grid assumed to pass
        while hi - lo > 1:
            mid = (lo + hi) // 2
            ev = evaluate_at(cfg, aging, net, grid[mid], rng, counter, horizon)
            evals[grid[mid]] = ev
            if ev["pass"]:
                hi = mid
            else:
                lo = mid
        if grid[hi] not in evals:             # top never evaluated: evaluate it now
            evals[grid[hi]] = evaluate_at(cfg, aging, net, grid[hi], rng, counter, horizon)
    ev_hi = evals[grid[hi]]
    mu_grid = grid[hi]
    lo_mu = grid[lo] if lo >= 0 else None
    mu_cont = smallest_pass(ev_hi, lo_excl=lo_mu, hi_incl=mu_grid) or mu_grid
    one_cont = smallest_pass(ref)
    one_grid = smallest_pass(ref, grid=set(grid))
    return {"mu_ref": mu_ref, "mu_hi": mh, "horizon": horizon, "grid": grid,
            "ref_eval": ref, "evals": {f"{k:g}": v for k, v in evals.items()},
            "top_failed": bool(not ev_hi["pass"] and hi == len(grid) - 1),
            "mu_2c": mu_cont, "mu_2c_grid": mu_grid,
            "mu_2c_oneshot": one_cont, "mu_2c_oneshot_grid": one_grid,
            "mu_2c_opt": ev_hi["mu_opt"], "mu_2c_opt_oneshot": ref["mu_opt"],
            "g_bar": ev_hi["g_bar"], "g_bar_ref": ref["g_bar"],
            "env_steps": counter.steps, "env_steps_oneshot": steps_oneshot,
            "n_evals": 1 + len(evals), "elapsed_s": time.time() - t0}


# ---------------------------------------------------------------------------

def snap_up(m, grid):
    if m is None:
        return None
    up = [g for g in grid if g >= m * 0.9999]
    return min(up) if up else max(grid)


def gap_agreement(rows) -> dict:
    """Predicted vs empirical gap at every evaluated grid point (incl. reference curve)."""
    pe, pp = [], []
    conf = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}   # positive = "within 0.5%"
    for r in rows:
        curve = {float(k): v for k, v in r["test_curve"].items()}
        best = min(curve.values())
        for k, ev in r["tc"]["evals"].items():
            m = float(k)
            if m not in curve:
                continue
            emp = curve[m] / best - 1.0
            pe.append(emp); pp.append(ev["gap_at_mu"])
            a, b = ev["pass"], emp <= FLAT_TOL
            conf["tp" if a and b else "fp" if a else "fn" if b else "tn"] += 1
    from scipy import stats
    pe, pp = np.array(pe), np.array(pp)
    return {"n": int(len(pe)), "confusion_pred_pass_vs_emp_pass": conf,
            "accuracy": float((conf["tp"] + conf["tn"]) / max(1, len(pe))),
            "spearman_gap": float(stats.spearmanr(pe, pp).statistic) if len(pe) > 2 else None}


def summarise(rows) -> dict:
    keys = ["mu_cf_gas_paper", "mu_lo", "mu_lo_p95", "mu_2c", "mu_2c_grid", "mu_2c_oneshot",
            "mu_2c_oneshot_grid", "mu_2c_opt", "mu_2c_opt_oneshot", "mu_min_lo_g",
            "mu_cf_grid", "mu_lo_grid", "mu_lo_p95_grid", "mu_train_bruteforce"]
    out = {}
    for tag, sub in (("pooled_all42", rows), ("oat", [r for r in rows if r["block"] == "oat"]),
                     ("lhs", [r for r in rows if r["block"] == "lhs"])):
        out[tag] = {k: corr_block(sub, k) for k in keys}
    # exact grid agreement (grid predictions only)
    out["exact_grid_match"] = {k: float(np.mean([r[k] == r["mu_emp"] for r in rows]))
                               for k in ("mu_2c_grid", "mu_2c_oneshot_grid", "mu_cf_grid",
                                         "mu_lo_grid", "mu_lo_p95_grid")}
    ctrl = {}
    names = ["2c", "2c_grid", "2c_oneshot", "2c_opt", "min_lo_g", "cert_2x", "cert_1x",
             "cf_paper_2x", "fixed_64000", "train_bruteforce"]
    for name in names:
        ev = [r["ctrl"][name] for r in rows]
        g = [e["gap_pct"] for e in ev if e["gap_pct"] is not None]
        ctrl[name] = {"frac_within_0.5pct": float(np.mean([e["within"] for e in ev])),
                      "median_gap_pct": float(np.median(g)), "max_gap_pct": float(np.max(g)),
                      "median_price": float(np.median([e["mu"] for e in ev if e["mu"]])),
                      "failures": [r["key"] for r, e in zip(rows, ev) if not e["within"]]}
    out["control_on_stored_curves"] = ctrl
    cmp = {}
    for other in ("cert_2x", "fixed_64000", "cf_paper_2x"):
        lower_ok = [r["key"] for r in rows if r["ctrl"]["2c"]["within"]
                    and r["ctrl"]["2c"]["mu"] < 0.999 * r["ctrl"][other]["mu"]]
        lower_loss = [r["key"] for r in rows if not r["ctrl"]["2c"]["within"]
                      and r["ctrl"][other]["within"]]
        better = [r["key"] for r in rows if r["ctrl"]["2c"]["within"]
                  and not r["ctrl"][other]["within"]]
        ratio = [r["ctrl"]["2c"]["mu"] / r["ctrl"][other]["mu"] for r in rows]
        cmp[other] = {"n_2c_lower_price_and_within": len(lower_ok), "keys_lower_ok": lower_ok,
                      "n_2c_fails_other_ok": len(lower_loss), "keys_2c_fails_other_ok": lower_loss,
                      "n_2c_ok_other_fails": len(better), "keys_2c_ok_other_fails": better,
                      "median_price_ratio_2c_over_other": float(np.median(ratio))}
    out["two_clock_vs_baselines"] = cmp
    out["gap_agreement_bisection_points"] = gap_agreement(rows)
    st = [r["tc"]["env_steps"] for r in rows]
    so = [r["tc"]["env_steps_oneshot"] for r in rows]
    out["env_steps"] = {"selfconsistent_median": float(np.median(st)), "min": int(min(st)),
                        "max": int(max(st)), "total": int(sum(st)),
                        "oneshot_median": float(np.median(so)),
                        "episodes_equiv_median": float(np.median(st)) / 960,
                        "r7_fast_cert_per_config": 3840,
                        "r4_train_curve_bruteforce_per_config": 7 * 2 * 960,
                        "n_evals_median": float(np.median([r["tc"]["n_evals"] for r in rows]))}
    out["n_top_failed"] = int(sum(r["tc"]["top_failed"] for r in rows))
    return out


def build_row(r, block, r7row, tc) -> dict:
    grid = tc["grid"]
    lo, hi = r7row["mu_lo"], r7row["mu_hi"]

    def cap(m):
        return None if m is None else (min(m, 0.9 * hi) if hi > 0 else m)

    row = {"block": block, "name": r["name"], "level": r["level"], "key": key_of(r, block),
           "params": r["params"], "mu_emp": r["mu_emp"], "best_test_mu": r["best_test_mu"],
           "best_saving_pct": r["best_saving_pct"], "test_curve": r["test_curve"],
           "mu_cf_gas_paper": r["closed_form"]["mu_cf_gas_paper"],
           "mu_lo": lo, "mu_lo_p95": r7row["mu_lo_p95"], "mu_hi": hi,
           "mu_2c": tc["mu_2c"], "mu_2c_grid": tc["mu_2c_grid"],
           "mu_2c_oneshot": tc["mu_2c_oneshot"], "mu_2c_oneshot_grid": tc["mu_2c_oneshot_grid"],
           "mu_2c_opt": tc["mu_2c_opt"], "mu_2c_opt_oneshot": tc["mu_2c_opt_oneshot"],
           "mu_min_lo_g": min(lo, tc["g_bar"]) if tc["g_bar"] > 0 else lo,
           "mu_cf_grid": snap_up(r["closed_form"]["mu_cf_gas_paper"], grid)
           if r["closed_form"]["mu_cf_gas_paper"] > 0 else None,
           "mu_lo_grid": snap_up(lo, grid), "mu_lo_p95_grid": snap_up(r7row["mu_lo_p95"], grid),
           "tc": tc}
    trc = {float(k): v for k, v in r["train_curve"].items()}
    tb = min(trc.values())
    row["mu_train_bruteforce"] = min(m for m, v in trc.items() if m > 0 and v <= tb * (1 + FLAT_TOL))
    row["ctrl"] = {
        "2c": ctrl_eval(r, row["mu_2c"]), "2c_grid": ctrl_eval(r, row["mu_2c_grid"]),
        "2c_oneshot": ctrl_eval(r, row["mu_2c_oneshot"]),
        "2c_opt": ctrl_eval(r, row["mu_2c_opt"]),
        "min_lo_g": ctrl_eval(r, row["mu_min_lo_g"]),
        "cert_2x": r7row["ctrl"]["cert_2x"], "cert_1x": r7row["ctrl"]["cert_1x"],
        "cf_paper_2x": r7row["ctrl"]["cf_paper_2x"],
        "fixed_64000": ctrl_eval(r, 64000.0),
        "train_bruteforce": ctrl_eval(r, row["mu_train_bruteforce"])}
    return row


def spot_check(r, mu, net) -> dict:
    cfg = r["params"]
    aging = aging_from(cfg)
    seeds = (1, 2, 3)
    with Applied(cfg):
        costs = [run_episode(make(cfg, aging, s), OPFAware(net, aging, float(mu)))["episode_cost"]
                 for s in seeds]
    curve = curve_from_detail(r, seeds)
    best = min(curve.values())
    mean = float(np.mean(costs))
    return {"mu": mu, "costs": costs, "mean": mean, "best_stored_seeds123": best,
            "gap_pct": 100.0 * (mean / best - 1.0),
            "interp_gap_pct_seeds123": 100.0 * (interp_curve(curve, mu)[0] / best - 1.0),
            "env_steps": 960 * len(seeds)}


def save(payload):
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, OUT_JSON)


def meta(el):
    return {"train_seeds": list(TRAIN_SEEDS), "k_snap_per_seed": K_SNAP, "delta": DELTA,
            "G_weighting": "per-unit self-normalised stake weights (no trimming)", "fine_grid": "125*2^(k/2), k=0..22", "flat_tol": FLAT_TOL,
            "horizon": "max(96, 4*tau/dt) steps", "mu_ref": "min(64000, 0.5*mu_hi) (R7 rule)",
            "policy": "OPFAware(mu) fast dispatch, identical to R4",
            "snapshot_sampling": "prob ∝ wear at stake sum_g alpha_g (e_g(mu_k)-e_g(256000))+ (fallback e_g(0) if none)",
            "primary": "mu_2c (self-consistent tolerance threshold, continuous refinement)",
            "r7_inputs": "mu_lo, mu_lo_p95, mu_hi, cert/cf control entries read from r7 JSON",
            "elapsed_s": el}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--plot" in argv:
        return plot()
    only = None
    for a in argv:
        if a.startswith("--only="):
            only = a.split("=", 1)[1].split(",")
    with open(R4_JSON, encoding="utf-8") as fh:
        r4 = json.load(fh)
    with open(R7_JSON, encoding="utf-8") as fh:
        r7 = {row["key"]: row for row in json.load(fh)["rows"]}
    net = build_5bus()
    jobs = [("oat", r) for r in r4["oat"]] + [("lhs", r) for r in r4["lhs"]]
    rows = []
    done = set()
    if "--resume" in argv and os.path.exists(OUT_JSON):
        with open(OUT_JSON, encoding="utf-8") as fh:
            rows = json.load(fh)["rows"]
        done = {x["key"] for x in rows}
    t0 = time.time()
    for i, (block, r) in enumerate(jobs):
        key = key_of(r, block)
        if (only and key not in only) or key in done:
            continue
        tc = two_clock(r["params"], block, net, i)
        row = build_row(r, block, r7[key], tc)
        rows.append(row)
        ev = tc["evals"]
        print(f"[{i+1}/42] {key:22s} emp={r['mu_emp']:>6g} lo={row['mu_lo']:7.0f} "
              f"2c={row['mu_2c']:7.0f} grid={row['mu_2c_grid']:>6g} one={row['mu_2c_oneshot']} "
              f"opt={row['mu_2c_opt']:.0f} Gref={np.round(tc['ref_eval']['G']).tolist()} "
              f"evals={[(k, round(v['gap_at_mu']*100, 2), np.round(v['G']).tolist()) for k, v in ev.items()]} "
              f"gap2c={row['ctrl']['2c']['gap_pct']:.3f}% steps={tc['env_steps']} "
              f"({tc['elapsed_s']:.0f}s)", flush=True)
        save({"meta": meta(time.time() - t0), "rows": rows})
    payload = {"meta": meta(time.time() - t0), "summary": summarise(rows), "rows": rows}
    save(payload)
    if not only:
        spots = []
        r4map = {key_of(r, b): r for b, r in jobs}
        for row in rows:
            if row["key"] in SPOT:
                sc = spot_check(r4map[row["key"]], row["mu_2c"], net)
                sc["key"] = row["key"]
                spots.append(sc)
                print(f"spot {row['key']}: mu={sc['mu']:.0f} gap={sc['gap_pct']:+.3f}% "
                      f"(interp {sc['interp_gap_pct_seeds123']:+.3f}%)", flush=True)
        payload["spot_checks"] = spots
        payload["meta"] = meta(time.time() - t0)
        save(payload)
        print(json.dumps({k: v for k, v in payload["summary"].items()
                          if k in ("pooled_all42", "control_on_stored_curves",
                                   "two_clock_vs_baselines", "gap_agreement_bisection_points",
                                   "env_steps", "exact_grid_match")}, indent=1))
        plot()
    return 0


# ---------------------------------------------------------------------------

def plot() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(OUT_JSON, encoding="utf-8") as fh:
        d = json.load(fh)
    rows = d["rows"]
    s = d["summary"]["pooled_all42"]
    blue, orange, grey = "#2a6fb0", "#d9822b", "#6b6b6b"
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), constrained_layout=True, sharey=True)
    lim = [1.2e2, 2e5]
    panels = (("mu_cf_gas_paper", "(a) closed form"),
              ("mu_lo", r"(b) fast certificate $\mu_{lo}$"),
              ("mu_2c", r"(c) two-clock $\mu_{2c}$"))
    for ax, (key, title) in zip(axes, panels):
        for block, mk, col in (("oat", "s", orange), ("lhs", "o", blue)):
            pts = [(r[key], r["mu_emp"]) for r in rows if r["block"] == block
                   and r.get(key) and r[key] > 0 and r["mu_emp"]]
            if pts:
                x, y = np.array(pts).T
                jit = np.exp(np.random.default_rng(1).uniform(-0.06, 0.06, len(y)))
                ax.scatter(np.clip(x, *lim), y * jit, s=20, marker=mk, facecolors="none",
                           edgecolors=col, lw=1.0, label=f"{block.upper()} (n={len(pts)})")
        ax.plot(lim, lim, color=grey, lw=0.9, label="y = x")
        ax.fill_between(lim, [v / 2 for v in lim], [v * 2 for v in lim], color=grey,
                        alpha=0.12, lw=0, label="factor 2")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lim); ax.set_ylim(lim)
        st = s.get(key, {})
        ax.text(0.97, 0.04, f"log-r = {st['log_pearson']:.2f}\nSpearman = {st['spearman']:.2f}\n"
                f"within 2x: {100*st['frac_within_2x']:.0f}%", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.5)
        ax.set_xlabel(r"predicted threshold ($/wear-unit)", fontsize=8.5)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel(r"empirical $\mu_{emp}$ (R4 test seeds)", fontsize=8.5)
    axes[0].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.savefig(OUT_FIG)
    fig.savefig(OUT_FIG.replace(".pdf", ".png"), dpi=200)
    print(f"wrote {OUT_FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
