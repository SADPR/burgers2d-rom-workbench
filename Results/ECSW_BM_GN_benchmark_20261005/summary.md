# Classical ECSW versus BM-ECSW: repeated online timings

3 fresh-process runs per method and parameter. Sequential execution; 20 CPU threads by default (actual pools recorded per run).
Matched training: 225 consecutive pairs, nine parameters, 96 POD modes, relative training tolerance 1e-5.
Classical: 2492 cells, weighted Gauss-Newton with lstsq. BM: 300 cells, gauss_newton matrix, maximum 20 iterations. Both enable plateau stopping at 1e-2, on their respective norm.
Times cover the production HPROM function: mesh/operator setup, initial projection and 500 time steps. They exclude training, Python startup, model loading, full-field decoding, HDM comparison, plotting and output files.

| Parameters | Method | Run 1 (s) | Run 2 (s) | Run 3 (s) | Mean (s) | Sample SD (s) |
|---|---|---:|---:|---:|---:|---:|
| (4.56, 0.019) | ECSW | 18.889134 | 18.945267 | 17.795273 | 18.543225 | 0.648353 |
| (4.56, 0.019) | BM-ECSW | 1.237558 | 1.205376 | 1.212362 | 1.218432 | 0.016928 |
| (4.75, 0.020) | ECSW | 19.681983 | 18.408255 | 18.389555 | 18.826598 | 0.740844 |
| (4.75, 0.020) | BM-ECSW | 1.208621 | 1.203551 | 1.199531 | 1.203901 | 0.004555 |
| (5.19, 0.026) | ECSW | 17.304445 | 19.879754 | 18.525874 | 18.570024 | 1.288222 |
| (5.19, 0.026) | BM-ECSW | 1.206142 | 1.210906 | 1.207941 | 1.208330 | 0.002405 |

| Parameters | Mean ECSW / mean BM-ECSW |
|---|---:|
| (4.56, 0.019) | 15.219x |
| (4.75, 0.020) | 15.638x |
| (5.19, 0.026) | 15.368x |

BM projected-equation tolerance met: 4500/4500 steps.
These measurements compare the implemented solvers and rules. The classical and BM stopping criteria differ, as documented in their production summaries.
Three runs characterize observed repeatability on this machine; no statistical significance claim is made.
