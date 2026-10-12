# Independent PSD patch hyperreduction

The candidate set consists of all disjoint 2x2 cell blocks of the original
250x250 mesh (15 625 groups). Training starts with an empty selected set.
No support or weights from classical ECSW or BM-ECSW enter this training.
Both saved methods are read only for the subsequent online comparison.

For each group P, the method fits a fixed PSD residual-space metric
A_P = L_P L_P^T. The online objective is one half the sum of
||L_P^T R_P(q)||^2. Gauss-Newton applies exactly the same L_P^T to the
sampled residual and its test-basis block. Its correction matrix is
symmetric positive semidefinite by construction. This is a residual
objective, without a claim about physical mechanical energy.

At each selection iteration the full candidate scan forms
S_P = sum_c sym((W_P h_c) R_P^T). Among unit-trace PSD additions, the
largest eigenvalue of S_P gives the greatest initial decrease of the
training objective. The selected group is then refitted jointly with
all previously selected groups. Factor directions and magnitudes remain
free. Factor ranks grow along negative eigenvectors of the gradient of
the original convex active-group fit. An exact least-squares line
minimizer initializes each rank expansion.

L-BFGS termination does not certify the convex fit. PSD dual feasibility
and complementarity are measured separately (scaled tolerance 1e-8),
and further selection requires these numerical KKT checks, unless the
training tolerance has already been attained. Checks concern the active
group fit, not minimum mesh cardinality. Residual error is always checked
on the original 21 600 training rows, with relative tolerance 1e-5.

Normal products are incrementally updated and reused in factor iterations;
this is the same quadratic objective, without an SVD truncation, objective
ridge or frozen matrix directions. Candidate scans use cached local
residuals/test bases (about 22 GB), avoiding a roughly 97 GB dictionary of
all cross-cell features. At most ten native threads leave CPU capacity for
concurrent AERO-F work.

The first experiment gives selection/refitting a 30-minute runtime budget,
after assembly. This is not a prescribed cell budget. An unfinished fit
is saved with stage training_incomplete. It cannot trigger an automatic
online comparison. Online eligibility requires both tolerance attainment
and full reduced rank at the sampled matrix checks.

The training plan is the existing 225 configurations, nine parameters,
96-mode affine POD basis, and seed42. Online runs use the three unsampled
parameters, 500 steps, relative nonlinear tolerance1e-5, max20 iterations,
and the same 0.01 plateau criterion as classical ECSW. Each measurement
uses a fresh process. Existing weights are not overwritten.

Run from the repository root:

```bash
.venv/bin/python -u run_patch_metric_experiment.py --threads 10 --tolerance 1e-5 --selection-seconds 1800
```

The new dated Results directory contains status.json, training_history.jsonl,
patch_metric.npz, training.json, source copies, provenance.json, and the
online comparisons. Its status file is updated during assembly, candidate
scans, factor iterations, rank checks and online runs.

Meaningful analytical checks cover the factor gradient, an exact PSD
projection optimum requiring increased rank, full candidate correlations,
the common objective gradient, preservation of the classical online solve
with identity factors, and selection starting from an empty set.

References: Burer and Monteiro (2003), DOI10.1007/s10107-002-0352-8;
Journee, Bach, Absil and Sepulchre, arXiv0807.4423. The current algorithm
adapts factor optimization and numerical KKT checks to the fixed-active-group
quadratic fit; reference algorithms' full convergence claims are not asserted
for a time-limited L-BFGS implementation.

The square-factor case retains PSD descent additions even at full column
count: a square factor may still have deficient numerical rank. A thin SVD
reexpresses the updated metric with at most d columns and retains every
singular value; it does not truncate the metric. A regression test and the
actual four-group refit that first exposed the issue verify this behavior.
Small active refits (at most 1 024 coefficient columns) use one BLAS thread,
based on a matched comparison (7.70 s versus 9.91 s) with identical original
error and KKT requirements; candidate scans retain the process limit.

When every factor has full column count and the Gram-coordinate fit requires
further refinement, an economy QR stabilizes the same selected least-squares
objective. No rows or singular values are truncated. Factor steps use the
triangular projected operator, while tolerance and KKT checks still use the
original training rows. This addresses the precision floor observed at 22
groups; the saved case passed the original KKT requirements after about
0.56 s for QR and 1.25 s for factor refinement. The QR overhead is recorded
separately and does not establish an end-to-end offline speedup.
