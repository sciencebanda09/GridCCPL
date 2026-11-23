# Stage 1 -- ablations (normal regime, five held-out seeds)

| variant | cost (M$) | vs blind | wear | unserved MWh |
|---|---|---|---|---|
| blind | 7.649 | +0.0% | 0.173 | 119.3 |
| myopic_4h_mu360 | 7.649 | -0.0% | 0.173 | 119.3 |
| negative_price | 7.661 | +0.2% | 0.172 | 115.6 |
| mlp_value | 7.649 | +0.0% | 0.173 | 119.3 |
| dual_ascent_eps006 | 6.974 | -8.8% | 0.072 | 40.7 |
| per_unit_ridge | 6.760 | -11.6% | 0.023 | 3.1 |
| scalar_ridge | 6.730 | -12.0% | 0.000 | 3.1 |

## VOLL sensitivity (normal regime, five held-out seeds)

| VOLL ($/MWh) | blind (M$) | priced (M$) | saving |
|---|---|---|---|
| 0 | 7.053 | 6.715 | 4.8% |
| 2500 | 7.351 | 6.723 | 8.5% |
| 5000 | 7.649 | 6.730 | 12.0% |
| 10000 | 8.246 | 6.746 | 18.2% |
