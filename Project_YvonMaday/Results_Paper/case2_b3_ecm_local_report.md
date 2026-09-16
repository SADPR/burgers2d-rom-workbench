# Local ECM experiment for Case 2+B3

The baseline HPROM-trained master ANN and the online B3 solver are unchanged.
Only the cubature rule is replaced. The same known initial representation
and predictor reuse are used in both runs. The nine baseline training
trajectories fit the rule; two external parameters select it. Four reporting
points are evaluated only after the selection is frozen.

## Cubature and validation

| Compression tolerance | Rank | Cells | Operator checks | Trajectory checks |
|---:|---:|---:|:---:|:---:|
| 0.1 | 387 | 388 | fail | not run |
| 0.05 | 505 | 506 | fail | not run |
| 0.02 | 676 | 677 | pass | fail |
| 0.01 | 838 | 839 | pass | pass |

The first three tolerances were rejected before extending the sweep.
The original rejection is archived; all acceptance thresholds were retained.
Compression tolerance is a relative Frobenius projection error, not the
relative singular-value cutoff used elsewhere in the manuscript.

Selected support: 839 residual cells, 2318 stencil cells.
Previous rule: 1506 residual cells, 4385 stencil cells.
Generalized Gram eigenvalues: [0.971811, 1.028214].
Maximum gradient relative error: 0.006256 (limit 0.05).
Maximum linearized residual ratio: 1.017650 (limit 1.05).
Validation coefficient-error ratios ECM/previous: 0.893114, 0.920199.

## Reporting accuracy

State errors are percentages against the HDM. Coefficient errors are
percentages against the matching linear HPROM trajectory.

| Parameter | Previous state | ECM state | Previous coefficients | ECM coefficients |
|---|---:|---:|---:|---:|
| (4.875, 0.0225) | 0.417693 | 0.415791 | 0.137095 | 0.112514 |
| (4.56, 0.019) | 0.456755 | 0.453765 | 0.275277 | 0.260366 |
| (5.19, 0.026) | 0.494481 | 0.493201 | 0.359779 | 0.345387 |
| (4.0, 0.033) | 0.850651 | 0.844079 | 0.421671 | 0.354172 |

In-domain mean state error: 0.456310% -> 0.454252%.

The selected ECM rule reduces support and preserves accuracy on the tested
baseline campaign. This is not a guarantee for other parameters or enrichment
budgets; enriched maps require their own fitted and validated rules.
Local wall times are diagnostic and excluded from this accuracy report.
Official timing comparison remains to be run on Sherlock.

## Reproducibility

34 focused numerical tests passed. The raw moment matrix reproduces the
previous fit's cached moments within 1e-4 relative tolerance, with full-mesh
ECM candidates and an independently checked compression residual.
The deployed rule contains positive weights and integrates the compressed
basis with relative error 6.736e-13.
The protocol, original block errors, and detailed audits are retained in
`euclidean_b3_ecm_local/`; the compact deployment is in
`euclidean_b3_ecm_deployment/`. The production manuscript is unchanged.
