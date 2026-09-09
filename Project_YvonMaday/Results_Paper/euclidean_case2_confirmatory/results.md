# Local Euclidean Case-2 experiments

All errors are percentages. Validation coefficient errors and reporting HDM state errors are different metrics.

## Full-trajectory validation

Two existing ANN-validation parameters; 500 free-running time steps, no reference predecessors.

| Method | Validation 1 | Validation 2 | Mean coefficient error | Mean local time (s) |
|---|---:|---:|---:|---:|
| Case 2 + ANN derivative directions | 3.9144 | 5.2316 | 4.5730 | 43.6 |
| Case 2, original initialization | 3.7123 | 3.9458 | 3.8290 | 32.4 |
| Case 2, known initial state | 3.8076 | 3.9374 | 3.8725 | 36.1 |
| Case 1, original initialization | 0.7503 | 1.2453 | 0.9978 | 60.7 |
| Case 1, known initial state | 0.7503 | 1.2480 | 0.9992 | 58.1 |
| Case 3, original initialization | 0.6774 | 0.9339 | 0.8057 | 60.4 |
| Case 3, known initial state | 0.6791 | 0.9347 | 0.8069 | 59.7 |
| Affine feedback, unregularized | 4.1046 | 4.2348 | 4.1697 | 34.0 |
| Affine feedback, ridge=1 | 4.1826 | 4.3032 | 4.2429 | 34.0 |
| Case 2 + fixed SVD, r=10 | 3.5498 | 3.6590 | 3.6044 | 57.5 |
| Case 2 + fixed SVD, r=3 | 3.6347 | 3.8683 | 3.7515 | 44.3 |
| Case 2 + residual-adaptive r=1 | 1.8873 | 2.0896 | 1.9884 | 57.7 |
| Case 2 + residual-adaptive r=3 | 0.9612 | 1.2244 | 1.0928 | 66.1 |
| Case 2 + residual-adaptive r=3 + known initial state | 0.2068 | 0.2549 | 0.2308 | 69.3 |
| Residual-adaptive r=3, no ANN after t=0 | 3.5803 | 3.9185 | 3.7494 | 90.4 |
| Case 2 + residual-adaptive r=5 | 0.9908 | 1.2554 | 1.1231 | 77.5 |

Times include online direction construction, where applicable. They are local CPU observations, not publication-grade repeated benchmarks.

## Reporting state errors against HDM

Original reference rows come from unchanged Euclidean campaign summaries. New rows require all four points to be complete.

| Method | Verification | Off-grid 1 | Off-grid 2 | In-domain mean | Extrapolation |
|---|---:|---:|---:|---:|---:|
| Linear PROM (151) | 0.4104 | 0.4498 | 0.4632 | 0.4411 | 0.8499 |
| PROM-ANN Case 1 | 0.4177 | 0.5432 | 0.7357 | 0.5655 | 1.6732 |
| PROM-ANN Case 2 (10) | 1.1629 | 1.9003 | 1.7580 | 1.6071 | 2.6962 |
| PROM-ANN Case 2 (20) | 1.1575 | 1.5565 | 1.6649 | 1.4597 | 2.2008 |
| PROM-ANN Case 3 | 0.4075 | 0.5115 | 0.5627 | 0.4939 | 1.5210 |
| Case 2, original initialization | 1.1629 | 1.9003 | 1.7580 | 1.6071 | 2.6962 |
| Case 2, known initial state | 1.1618 | 1.9466 | 1.7433 | 1.6172 | 2.6428 |
| Case 1, known initial state | 0.4177 | 0.5425 | 0.7365 | 0.5656 | 1.6729 |
| Case 3, known initial state | 0.4076 | 0.5118 | 0.5628 | 0.4940 | 1.5208 |
| Case 2 + residual-adaptive r=3 | 0.6864 | 0.6355 | 0.6739 | 0.6653 | 1.0233 |
| Case 2 + residual-adaptive r=3 + known initial state | 0.4115 | 0.4457 | 0.4831 | 0.4468 | 0.8503 |

Baseline reproduction: maximum relative coefficient-trajectory difference from the downloaded Case-2 run is 1.469e-07 (4 completed points).

## Matched local timing check

Reporting point (4.56, 0.019), two CPU threads. Repeated rows report the mean and population standard deviation; single observations are labeled n=1.

| Method | Repetitions | Mean seconds | Standard deviation |
|---|---:|---:|---:|
| Case 2, original initialization | 1 | 25.19 | not estimated |
| Case 2, known initial state | 1 | 25.75 | not estimated |
| Case 1, original initialization | 2 | 56.53 | 0.72 |
| Case 1, known initial state | 1 | 57.46 | not estimated |
| Case 2, n=20 | 2 | 51.17 | 0.04 |
| Case 3, original initialization | 2 | 58.42 | 0.98 |
| Case 3, known initial state | 1 | 55.86 | not estimated |
| Case 2 + residual-adaptive r=3 | 1 | 59.80 | not estimated |
| Case 2 + residual-adaptive r=3 + known initial state | 1 | 60.53 | not estimated |

## Scope and limitations

- The ANN, its architecture and its nine-parameter training data are unchanged.
- The selected rank is determined only from validation; see selection.json.
- Three adaptive amplitudes add to the ten primary unknowns, but building their directions accesses all 141 tail modes and the full residual/Jacobian.
- The adaptive basis is frozen throughout each time-step nonlinear solve. The method changes the online trial space, not just ANN training.
- The known-initial variant uses the full linear PROM projection of the prescribed initial state. Subsequent states are not teacher forced.
- The zero-tail ablation removes ANN predictions after t=0 but retains the original ANN initial state.
- These are single-checkpoint, single-discretization results, not evidence of universal superiority or novelty.
- The 151-mode linear PROM is a reference approximation, not a mathematical lower bound on HDM state error.
- Neither the manuscript nor MLSPG/HPROM results were modified.
