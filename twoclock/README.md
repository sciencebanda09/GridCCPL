# Two-Clock Dispatch (`twoclock/`)

A shadow-price controller for power-system dispatch: the marginal cost of
equipment wear is a *price with units* ($/wear-unit), learned from a value
function in dollars, and added to the DC-OPF as a linear cost coefficient so the
fast problem remains an LP.

See the repository `README.md` for how to run each experiment, data locations and seeds.

## Model decisions (deliberate)

* **Power balance is an equality**, not a swing equation. At 15-minute
  resolution, supply equals demand at each bus; frequency lives at seconds.
* **Renewable is must-take** and is not ramp-limited.
* **Slow clock = wear -> capacity derating + heat-rate penalty**, both convex.
  Forced outages are an optional tail-risk ablation (`outage_enabled=False` by
  default).
* **The environment never repairs an infeasible action.** Unserved load costs
  VOLL, overload costs money, reserve shortfall costs money.
* **The slow state is not in the reward as a penalty.** It only bites through
  physics.

## Layout

```
twoclock/
  paths.py              package-relative data / results / figure locations
  physics/dc_flow.py    5-bus DC network, fleet tables
  physics/aging.py      wear ODE, derate/heat-rate, optional outages
  physics/opf.py        DC-OPF LP with duals, exact piecewise wear price,
                        fast merit-order path (feasibility-gated LP fallback)
  envs/two_clock_env.py 15-min environment, slow clock, domain scoreboard
  agents/baselines.py   random, rule, blind OPF, priced OPF
  agents/price_agent.py value-gradient dispatch (ridge / MLP)
  agents/networks.py    numpy MLP + Adam
  instrumentation.py    JSONL runs with physical parameters and env version
  experiments/          stage1, stage1 ablations, stage2, stage2b, stage2c,
                        stage3, real-ISO discovery, rigorous suite, figures
  selfcheck.py          invariants
  results/              JSON / Markdown outputs of the experiments
```
