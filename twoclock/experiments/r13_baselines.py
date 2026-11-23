"""R13 -- fairness baselines for the dual certificate.

Per 5-bus regime (7), 20 held-out seeds (1-20), paired:
  blind OPF | best constant (26-episode train sweep) | 10-episode bisection on log mu |
  hard cap (mu=1e9: never exceed sigma*rmax except ramp-forced) |
  cap-unless-shedding (per-unit mu_g = 0.99*mu_g^hi) | dual certificate.
Also reports environment steps, LP solves and wall-clock for every *selection* method.

Run: python -m twoclock.experiments.r13_baselines [regime ...]
"""
from __future__ import annotations
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy import stats

TEST = tuple(range(1, 21))
OUT = None


def ci95(x):
    x = np.asarray(x, float)
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def summ(x):
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "ci95": ci95(x), "per_seed": x.tolist()}


def run_regime(args):
    name, diff, ds = args
    import twoclock.agents.certificate as C
    import twoclock.physics.opf as O
    cnt = {"n": 0}
    for m in (C, O):
        f = m.linprog
        def w(*a, _f=f, **k):
            cnt["n"] += 1
            return _f(*a, **k)
        m.linprog = w
    from twoclock.experiments import r6_certificate as R
    from twoclock.agents.certificate import alpha_coef, fixed_point_certificate
    from twoclock.experiments.runner import run_episode
    import twoclock.physics.dc_flow as dcf
    import twoclock.physics.opf as opf_mod
    from twoclock.physics import AgingParams
    aging = AgingParams()
    net = R.set_network("5bus")
    c = R.Ctx(net, diff, ds, aging)
    a = alpha_coef(aging)
    hi_g = (opf_mod.VOLL - dcf.GEN_COST[:3]) / a[:3]
    out = {"regime": name, "test_seeds": list(TEST)}

    def timed(fn):
        n0, t0 = cnt["n"], time.time()
        r = fn()
        return r, {"wall_s": time.time() - t0, "lp_solves": cnt["n"] - n0}

    # --- selection methods -------------------------------------------------
    cert, cost = timed(lambda: fixed_point_certificate(
        lambda s: c.env(s), net, aging, R.CERT_SEEDS, run_episode, log=lambda *x: None))
    mu_lo, mu_hi = cert["mu_lo"], cert["mu_hi"]
    mu_cert = float(np.sqrt(mu_lo * mu_hi)) if np.isfinite(mu_hi) else 2.0 * mu_lo
    cost["env_steps"] = cert["env_steps"]
    out["sel_certificate"] = {**cost, "mu": mu_cert, "mu_lo": mu_lo, "mu_hi": mu_hi}

    def sweep():
        tr = {0.0: float(np.mean(c.costs(0.0, R.CONST_TRAIN_SEEDS)))}
        for m in R.GRID_COARSE:
            tr[m] = float(np.mean(c.costs(m, R.CONST_TRAIN_SEEDS)))
        return min(tr, key=tr.get)
    c.cache.clear()
    mu_sweep, cost = timed(sweep)
    cost["env_steps"] = 13 * 2 * 960
    out["sel_sweep"] = {**cost, "mu": mu_sweep}

    def bisect():
        a_, b_ = 1e2, 1e6
        for _ in range(5):
            m = float(np.sqrt(a_ * b_))
            j1 = c.costs(m, (101,))[0]
            j2 = c.costs(2 * m, (101,))[0]
            if j2 < j1 * (1 - 1e-3):
                a_ = m
            else:
                b_ = m
        return b_
    c.cache.clear()
    mu_bis, cost = timed(bisect)
    cost["env_steps"] = 10 * 960
    out["sel_bisection"] = {**cost, "mu": mu_bis}

    # --- evaluation on 20 held-out seeds -----------------------------------
    c.cache.clear()
    blind = np.array(c.costs(0.0, TEST))
    meth = {
        "blind": blind,
        "best_constant_sweep": np.array(c.costs(mu_sweep, TEST)),
        "bisection": np.array(c.costs(mu_bis, TEST)),
        "hard_cap_mu1e9": np.array(c.costs(1e9, TEST)),
        "certificate": np.array(c.costs(mu_cert, TEST)),
    }
    mu_cap = np.concatenate([0.99 * hi_g, [0.0]])
    meth["cap_unless_shedding"] = np.array(c.costs(mu_cap, TEST))
    ref = meth["certificate"]
    res = {}
    for k, v in meth.items():
        res[k] = {"cost": summ(v), "minus_certificate": summ(v - ref),
                  "saving_vs_blind_pct": float(100 * np.mean((blind - v) / blind)),
                  "unserved_mwh": float(np.mean([c.run_mu(
                      (mu_cap if k == "cap_unless_shedding" else
                       {"blind": 0.0, "best_constant_sweep": mu_sweep, "bisection": mu_bis,
                        "hard_cap_mu1e9": 1e9, "certificate": mu_cert}[k]), s)["unserved_mwh"]
                      for s in TEST]))}
    out["eval"] = res
    out["cap_mu_g"] = mu_cap.tolist()
    return out


def main():
    from twoclock.paths import RESULTS_DIR
    regs = [("normal", "normal", 1.0), ("hard", "hard", 1.0), ("peak", "peak", 1.0),
            ("storm", "storm", 1.0), ("tight1.15", "normal", 1.15),
            ("tight1.25", "normal", 1.25), ("tight1.35", "normal", 1.35)]
    sel = sys.argv[1:]
    if sel:
        regs = [r for r in regs if r[0] in sel]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(len(regs), os.cpu_count() or 1)) as ex:
        rows = list(ex.map(run_regime, regs))
    payload = {"meta": {"test_seeds": list(TEST), "elapsed_s": time.time() - t0}, "regimes": rows}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "r13_baselines.json"), "w") as fh:
        json.dump(payload, fh, indent=1)
    for r in rows:
        print(r["regime"], {k: round(v["cost"]["mean"] / 1e6, 4) for k, v in r["eval"].items()})
        print("   sel:", {k: (round(r[k]["wall_s"], 1), r[k]["lp_solves"], r[k]["env_steps"])
                         for k in ("sel_certificate", "sel_sweep", "sel_bisection")})


if __name__ == "__main__":
    main()
