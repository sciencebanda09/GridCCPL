# GridCCPL — Energy Grid Load Balancing Benchmark

GridCCPL is an energy grid benchmark built on Causal Consequence-Penalized Learning applied to power grid management.
Designed to study consequence-aware reinforcement learning under delayed system stress, fluctuating demand, and operational constraints.

The repository has two parts:

| Part | Folder | What it is |
|---|---|---|
| Benchmark + RL agent | `agents/ environments/ networks/ training/ evaluation/ run.py` | 18-state / 5-action grid environment, the GridCCPL agent, PPO ablations, rule-based baselines |
| Wear-pricing research code | `twoclock/` | Two-timescale dispatch (DC-OPF + equipment wear): wear shadow price, LP-dual certificate, baselines, outage tiers, real-market audits |

---

## Part 1 — GridCCPL benchmark

### Quick Start

```bash
pip install -r requirements.txt
python run.py --quick          # 100 episodes, fast test
python run.py --episodes 1000  # full training
python run.py --help           # all options
```

### Project Structure

```
run.py                         ← ENTRY POINT
run_seeds.py                   ← multi-seed runs
environments/
└── grid_env.py                ← Grid simulation (18-state, 5-action)
agents/
├── grid_acpl_agent.py         ← GridCCPL continuous agent (+ PPO ablations)
└── baselines.py               ← RuleBase, RandomAgent
networks/
└── grid_networks.py           ← GRU, Actor, Critic, Lambda, Delay nets
training/
├── train_grid.py              ← Training loop, evaluation, logging
└── replay_buffer.py           ← Experience replay
evaluation/
└── plots.py                   ← Training curves, lambda heatmap, etc.
utils/
└── normalizer.py              ← Online state normalizer
dashboard/                     ← HTML dashboard generator
logs/  results/                ← auto-created at run time (git-ignored)
```

### What CCPL Does Here

| Component        | Grid Role                                              |
|------------------|--------------------------------------------------------|
| GRU encoder      | Remembers 24h price/demand cycles                      |
| Consequence net  | Predicts stress (4h), efficiency (24h), billing (30d) |
| Lambda net       | Raises penalty near capacity limits & high stress      |
| Delay estimator  | Learns that stress signals arrive ~4h after overload   |
| Hit-freq EMA     | Tracks how often consequence thresholds are breached   |

### State Space (18-dim)
load_norm, renewable_fraction, spot_price, equipment_stress, reserve_margin,
frequency_deviation, gas/coal/nuclear/renewable outputs, battery_soc,
time encoding (sin/cos), temperature, cumulative_stress_24h, market_trend

### Action Space (5-dim continuous, each ∈ [-1, 1])
gas_delta, coal_delta, market_action, load_shed, battery_action

### Environments
- `energy_grid`  — standard (training)
- `grid_easy`    — reduced demand (training)
- `grid_hard`    — high demand (training + eval)
- `grid_storm`   — extreme renewable variability (unseen)
- `grid_peak`    — heatwave +25% demand (unseen)

### Sample Results

#### Training Curves
![Training Curves](results_sample/01_training_curves.png)

#### State Conditioned Lambda
![Lambda Heatmap](results_sample/03_lambda_heatmap.png)

#### Benchmark Comparison
![Benchmark Comparison](results_sample/05_benchmark_comparison.png)

---

## Part 2 — Wear pricing for two-timescale dispatch (`twoclock/`)

Power-system dispatch runs on two clocks: minutes for dispatch, hours to months for
equipment wear that later appears as derated capacity, worse heat rates and forced
outages. `twoclock/` prices wear as a shadow price `mu` (dollars per model wear unit)
that enters the DC-OPF linearly above a stress threshold, so the fast problem stays a
linear program. With the model known, episode cost is flat on a price band whose edges
are LP duals; the **dual certificate** computes that band from a few rollouts.

Main components (`twoclock/`):

```
physics/        5-bus DC network, wear ODE, derating / heat-rate, DC-OPF LP with duals
envs/           15-minute environment with a slow (wear) clock
agents/         blind / priced OPF, dual certificate (lazy exact probe), value-gradient learners
experiments/    one script per experiment (table below)
results/        JSON outputs of every experiment (small, committed)
selfcheck.py    physics / LP-equivalence invariants
```

### Setup

```bash
pip install -r twoclock/requirements.txt
python -m twoclock.selfcheck      # run from the repository root
```

### Experiments

Run every command from the repository root. Training seeds are always `>= 101` and are
disjoint from held-out test seeds (`1-5`, `1-20` or `1-40` depending on the script).
Scripts that use several worker processes need roughly 1 GB RAM each.

| Command | What it does | Output |
|---|---|---|
| `python -m twoclock.experiments.r10_verify` | Numerical verification of the price-band proposition | `r10_verify.json` |
| `python -m twoclock.experiments.r6_certificate` | Dual certificate vs. constant-price sweep, 7 regimes + 14/30-bus | `r6_certificate.json` |
| `python -m twoclock.experiments.r13_baselines` | Bisection, hard-cap and cap-unless-shedding baselines; 20 seeds; simulator steps / LP solves / wall-clock | `r13_baselines.json` |
| `python -m twoclock.experiments.r14_lazy_check` | Lazy vs. full certificate probe: exactness and speed | `r14_lazy_check.json` |
| `python -m twoclock.experiments.r15_misspec` | Certificate computed under a misspecified aging model, deployed in the true one | `r15_misspec.json` |
| `python -m twoclock.experiments.r16_mpc_oracle` | Receding-horizon perfect-foresight price oracle vs. the certificate | `r16_mpc_oracle.json` |
| `python -m twoclock.experiments.r17_long_memory` | Long-memory wear (thermal time constant 12-48 h) | `r17_long_memory.json` |
| `python -m twoclock.experiments.r18_breakpoint_audit` | Complete breakpoint sweep: no dispatch change inside the claimed band; no substitutable-unit shedding threshold on the 5-bus fleet | `r18_breakpoint_audit.json` |
| `python -m twoclock.experiments.r19_mpc_longmemory` | MPC price oracle at thermal time constant 24/48 h (peak, tight 1.15) | `r19_mpc_longmemory.json` |
| `python -m twoclock.experiments.r20_pid_lagrangian` | PID-Lagrangian vs dual ascent (target mis-set vs oracle-tuned) | `r20_pid_lagrangian.json` |
| `python -m twoclock.experiments.r21_assumption_A` | One-step-deviation test of the dynamic assumption (A), tau = 3 h vs 48 h | `r21_assumption_A.json` |
| `python -m twoclock.experiments.r22_independent_dgp [DATA_DIR]` | Independent-generator controls for the NYISO audit (no inverse crime) | `r22_independent_dgp.json` |
| `python -m twoclock.experiments.r23_param_prior` | Random wear parameters: certificate vs best constant | `r23_param_prior.json` |
| `python -m twoclock.experiments.r24_completeness` | Verifiable flat interval (completeness condition C) on random fleets + substitutable-unit counterexample | `r24_completeness.json` |
| `python -m twoclock.experiments.r25_hazard_calibration [cems.parquet]` | Outage-hazard / time-constant calibration from EPA CEMS 2024 (needs the cached R12 data) | `r25_hazard_calibration.json` |
| `python -m twoclock.experiments.r26_convexity` | Convexity of the cost-to-go in wear, with and without heat-rate drift | `r26_convexity.json` |
| `python -m twoclock.experiments.r27_verify_band` | Trajectory-level check that the certified band is flat (7 regimes, 13,440 steps) | `r27_verify_band.json` |
| `python -m twoclock.experiments.r28_hazard_all_us [dir]` | Hazard calibration on all US EPA CEMS 2015-2024 (needs the CEMS parquet files, see script header) | `r28_hazard_all_us.json` |
| `python -m twoclock.experiments.r29_heatrate_calibration [dir]` | Heat-rate vs model wear state on CEMS 2022-2024 | `r29_heatrate_calibration.json` |
| `python -m twoclock.experiments.r30_outages_calibrated` | Outage tiers at the calibrated hazard coefficient | `r30_outages_calibrated.json` |
| `python -m twoclock.experiments.r31_calibrated_savings` | Certificate savings with calibrated heat rate / tau and varied derating | `r31_calibrated_savings.json` |
| `python -m twoclock.experiments.r32_A_screen` | Does the (A) deviation screen predict failing random-parameter draws? | `r32_A_screen.json` |
| `python -m twoclock.experiments.r33_hazard_cluster` | State-clustered bootstrap/jackknife CI for the all-US hazard fit: (hatkappa=0.0086)/day, 95% CI 0.0073–0.0097 (48 states, 400/400 draws positive) | `r33_hazard_cluster.json` |
| `python -m twoclock.experiments.plot_calibration` | Calibration figure (trip rate vs wear state; heat-rate effect) | `fig_calibration.pdf` |
| `python -m twoclock.experiments.r34_heldout_screen` | Held-out validation of the (A) screen on 24 new random-parameter draws (frozen threshold): flags 12/14 failures, 3/10 sound; certificate within 0.5% of best constant in 10/24, worse than blind in 6 | `r34_heldout_screen.json` |
| `python -m twoclock.experiments.r35_expected_trips` | Expected cost of Tier 0 vs Tier 3 at the calibrated hazard, computed from the trip-free wear path (no rare-event sampling) | `r35_expected_trips.json` |
| `python -m twoclock.experiments.r36_screen_derating` | (A) screen on the calibrated derating variants: flags all 4 losing cases but 8 of 11 sound ones | `r36_screen_derating.json` |
| `python -m twoclock.experiments.r37_hazard_fe` | Relative-hazard fit with unit-year x load x quarter fixed effects; tau 12 h vs 1.5 h by state: hazard x8.3 at W=1, tau=1.5 h preferred in 33/48 states | `r37_hazard_fe.json` |
| `python -m twoclock.experiments.r1_learner_eval` | Pathwise vs. ridge value-gradient learners | `r1_learner.json` |
| `python -m twoclock.experiments.r2_ppo_eval` | Tuned PPO baselines | `r2_ppo.json` |
| `python -m twoclock.experiments.r3_interior` | State-dependent price rules vs. best constant | `r3_interior.json` |
| `python -m twoclock.experiments.r4_robust` | 42-configuration robustness suite | `r4_robust.json` |
| `python -m twoclock.experiments.r7_cert_robust` | Certificate across the 42 configurations | `r7_cert_robust.json` |
| `python -m twoclock.experiments.r8_congestion` | Congested 5/14/30-bus grids, per-unit vs. scalar price | `r8_congestion.json` |
| `python -m twoclock.experiments.r9_twoclock` | Two-clock certificate analysis | `r9_twoclock.json` |
| `python -m twoclock.experiments.r11_risk_outages` | Outage risk tiers (40 test seeds, exact LP) | `r11_risk_outages.json` |
| `python -m twoclock.experiments.r5_stage3` | Synthetic price-identification study + ENTSO-E aggregate audit | `r5_stage3.json` |
| `python -m twoclock.experiments.r12_unit_real_data` | NYISO 2024 + EPA CEMS unit-level audit | `r12_unit_real_data.json` |

### External data (only for the two audit experiments)

* **ENTSO-E Spain** (`r5_stage3`, `real_iso_discovery`): place `energy_dataset.csv` from the
  public Kaggle dataset *energy-consumption-generation-prices-and-weather* in
  `twoclock/data/` (or set `TWOCLOCK_DATA`). Not included in this repository.
* **NYISO 2024 + EPA CEMS** (`r12_unit_real_data`): downloaded and cached automatically from
  public sources (Catalyst Cooperative PUDL open data and NYISO OASIS day-ahead LBMP
  archives). Needs `duckdb`, `pyarrow` and `requests`.

### Known limits of the method (measured, see the result JSONs)

* The certificate needs the stress threshold `sigma` to be right: a wrong belief
  (0.80 or 0.90 instead of 0.85) loses a large share of the saving (`r15_misspec.json`).
  It is insensitive to the aging rate, damage curvature and thermal time constant
  within 1.5-6 h.
* It is a method for wear that decays within about a day; with a thermal time constant
  of 24-48 h the best constant price lies above the certificate band (`r17_long_memory.json`).
* It needs fewer simulator steps than a sweep or bisection, but a bisection is faster in
  wall-clock (`r13_baselines.json`, `r14_lazy_check.json`).
* At long memory the perfect-foresight oracle beats the certificate by 1.2-7.2% but the best constant
  price by only <=0.3%: the loss is a price level above the band, not state dependence (`r19_mpc_longmemory.json`).
* The certificate's price in $/MWh depends on the fuel merit order, not on how damaging wear is. With random wear
  parameters it matched the best constant in 11/24 draws; it fails when (A) fails, upward (best price above the band)
  or downward (damage too small, certificate costs more than blind). Screen with `r21_assumption_A`.
* A PID-Lagrangian does not fix a mis-set constraint target (`r20_pid_lagrangian.json`).
* On NYISO data, an independent no-wear smooth-supply generator makes the one-sided 'excludes zero' test fire on 100% of
  windows (`r22_independent_dgp.json`), so that test is uninformative.
* The proof of the price-band proposition, with its completeness condition (C), is in `twoclock/PROOF.md`.
* Calibration (EPA CEMS 2015-2024, 101,848 trip-like shutdowns): the model's wear path predicts trips, kappa_hat = 0.0086/day
  (95% CI 0.0082-0.0091), best tau about 12 h; heat rate does not degrade with wear (-0.6 to -1.2% at W=1, model assumed +90%).
  Synthetic kappa in the outage experiments is 29-700x larger; at the calibrated kappa Tier 0 (the certificate) is optimal.
  Capacity derating, which the savings depend on, is not identifiable from public data (`r31_calibrated_savings.json`).
* The closed-form band of the proposition can miss breakpoints for substitutable units (15% of 391 random fleets);
  the verified flat interval (two LP solves per step, `PROOF.md`, `r24_completeness`) is exact. Cost-to-go convexity held on 95-100% of
  tested lines, with or without heat-rate drift (`r26_convexity`).
* Wear parameters are reduced-form and uncalibrated; dollar savings are illustrative.
