"""R14 -- exactness and speed of the lazy dual-certificate probe.

Compares certificate outputs (mu_lo, mu_hi, per-unit bands, forced-step counts) and
wall-clock/LP-solve counts of the full probe (5 LPs/step) vs the lazy probe that
skips LPs which cannot change the result.   Run: python -m twoclock.experiments.r14_lazy_check
"""
from __future__ import annotations
import json, os, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

REG = [("normal", "normal", 1.0), ("hard", "hard", 1.0), ("peak", "peak", 1.0),
       ("storm", "storm", 1.0), ("tight1.15", "normal", 1.15),
       ("tight1.25", "normal", 1.25), ("tight1.35", "normal", 1.35)]


def one(args):
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
    from twoclock.experiments.runner import run_episode
    from twoclock.physics import AgingParams
    aging = AgingParams()
    net = R.set_network("5bus")
    c = R.Ctx(net, diff, ds, aging)
    out = {"regime": name}
    for lazy in (False, True):
        orig = C.CertificateProbe.__init__
        def init(self, net, aging, mu, tol_mw=1e-6, lazy_=lazy, _o=orig):
            _o(self, net, aging, mu, tol_mw, lazy_)
        C.CertificateProbe.__init__ = init
        n0, t0 = cnt["n"], time.time()
        cert = C.fixed_point_certificate(lambda s: c.env(s), net, aging, R.CERT_SEEDS,
                                         run_episode, log=lambda *x: None)
        out["lazy" if lazy else "full"] = {
            "wall_s": time.time() - t0, "lp_solves": cnt["n"] - n0,
            "env_steps": cert["env_steps"], "mu_lo": cert["mu_lo"], "mu_hi": cert["mu_hi"],
            "per_unit_lo": cert["per_unit_lo"], "per_unit_hi": cert["per_unit_hi"],
            "forced": cert["forced_steps_per_unit"]}
        C.CertificateProbe.__init__ = orig
    a, b = out["full"], out["lazy"]
    out["identical"] = bool(a["mu_lo"] == b["mu_lo"] and a["mu_hi"] == b["mu_hi"]
                            and a["per_unit_lo"] == b["per_unit_lo"]
                            and a["forced"] == b["forced"])
    return out


def main():
    from twoclock.paths import RESULTS_DIR
    with ProcessPoolExecutor(max_workers=len(REG)) as ex:
        rows = list(ex.map(one, REG))
    with open(os.path.join(RESULTS_DIR, "r14_lazy_check.json"), "w") as fh:
        json.dump(rows, fh, indent=1, default=float)
    for r in rows:
        print(r["regime"], "identical" if r["identical"] else "DIFFERENT",
              "full %.0fs/%d LP" % (r["full"]["wall_s"], r["full"]["lp_solves"]),
              "lazy %.0fs/%d LP" % (r["lazy"]["wall_s"], r["lazy"]["lp_solves"]))


if __name__ == "__main__":
    main()
