# B3 predictor reuse

This experiment retains the 3532-cell positive rule, rank 3, and a newly
constructed correction space at every time step. It changes only reuse of
the weighted residual and weighted Jacobian evaluated at the predictor.

The predictor is offset + Vp q_previous. The initial affine coordinates are
(q_previous, 0), so it is also the initial state of the 13-coordinate solve.
The new optional initial_evaluation argument supplies that state, its
residual, and its Jacobian to the first Gauss-Newton iteration. All later
Jacobians, the least-squares solver, and stopping criteria are unchanged.
The weighted operators retain sqrt(xi) on both residual and Jacobian.

Reconstruction via a 10-column versus a 13-column product can differ in
floating-point summation order. The cached state and its cached operators
are kept together, so they are always consistent. Equivalence is tested
numerically, including the stopping iteration.

Existing production entry points retain reuse_predictor=False. The paired
benchmark explicitly enables reuse for the candidate and disables it for
the reference. It leaves the production campaign outputs intact.

## Local evidence

Results are in Results_Paper/euclidean_b3_reuse_local:

- Four reporting parameters, 500 steps, one thread.
- Original and reuse variants each run once at each parameter: eight trajectories.
- Alternating variant order, no full-trajectory warmups.
- All four coefficient trajectories are elementwise identical between variants.
- Iteration totals are unchanged: 1500, 1500, 1502, 1500.
- In-domain mean time: 6.464641 s original, 5.746448 s reuse.
- Observed time reduction: 11.110%; speed ratio: 1.124980.
- Unit tests: 29 passed, including weighted sampled solves, full-mesh recovery,
  nonlinear Jacobians, and exactly one saved residual/Jacobian evaluation.

Phase timers separate predictor evaluation, Krylov construction, tangent
construction, and the affine solve. They are enabled in both benchmark
variants. These are local measurements, not Sherlock timings, and one
measurement per variant/point does not estimate timing variability.

## Sherlock

Upload the files listed in b3_reuse_upload.txt from the repository root.
Run run_case2_b3_reuse_benchmark.sh with bash inside the existing allocation
and active myenv. It uses one numerical thread for both variants and checks
the basis, model, data metadata, and rule hashes against the frozen runs.
There is no training, rule fitting, or HDM solve.

The comparison is written to Results_Paper/euclidean_b3_reuse_sherlock.
An existing comparison directory is rejected to avoid accidental replacement.
The output contains summary.txt, comparison.csv, comparison.json, the
environment/input manifest, phase timings, and both coefficient trajectories.
Accuracy scoring and file I/O occur outside the online timers.

Download the complete comparison directory after completion. Only after
reviewing this experiment should production timings or manuscript tables
be updated.
