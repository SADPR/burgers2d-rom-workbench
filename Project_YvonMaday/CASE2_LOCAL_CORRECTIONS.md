# Euclidean Case-2 correction experiments

These experiments reuse the baseline nine-parameter Euclidean master ANN.
They do not change the manuscript, the original checkpoints, or any MLSPG or
HPROM campaign. The full mesh has 125000 state entries, the basis has 151
Euclidean-orthonormal columns, and all reporting rollouts have 500 time steps.

## Protocol

- Learn any additional offline map using only the original nine training
  parameters. No enrichment and no ANN retraining in this experiment.
- Screen candidates at the existing two validation parameters, (4.5625,
  0.02625) and (5.1875, 0.01875). These are not new independent holdouts: the
  original ANN already used them for early stopping.
- Judge validation accuracy against the saved full linear PROM trajectories.
  Do not load HDM data or use reference predecessors inside a solve.
- Freeze the selected method before evaluating the four manuscript points.
  The verification point is itself a training parameter; the two off-grid
  and the extrapolatory points are the out-of-training comparisons.
- Every rollout uses its own previous computed state. All low-rank variants
  start from the same ANN-offset initial state as the original Case 2.
- Keep the original Gauss-Newton least-squares solver and stopping rules:
  20 iterations maximum, relative residual cutoff 1e-5, stagnation 1e-2.
- Record local CPU wall times, including direction construction. Do not
  compare a local timing with a Sherlock timing as if hardware were matched.
- Store new arrays and JSON summaries separately. Completed runs are skipped
  only if their input and implementation fingerprints match.

## 1. Fixed and parameter-dependent correction spaces

Write the master ANN as (M_p, M_s), with 10 primary and 141 secondary
coordinates. The low-rank trial state at time step m+1 is

    u(q,a) = u_ref + Vp q + Vs (M_s(mu,t) + B a).

Its tangent is [Vp, Vs B], and the online problem is the usual unweighted
full-residual LSPG minimization over (q,a).

Two controls are tested:

- A fixed B from an SVD of the nine training trajectories' tail prediction
  errors. This is an in-sample SVD, not the older out-of-fold MLSPG study.
- A three-column B spanning the ANN tail derivatives with respect to
  (mu1,mu2,t), evaluated at the requested parameter and time. It permits
  infinitesimal parameter/time-shift patterns, with independent amplitudes.

Both spaces are fixed during the nonlinear solve at a given time step.

## 2. Residual-adaptive correction space

At the beginning of a time step, form a candidate from the previous primary
coordinates and the current ANN tail. Evaluate its full residual r and
Jacobian J using the previous *computed* state. Define

    A = J Vp,  Q = orth(A),  P = I - Q Q^T,  C = P J Vs.

Eliminating the primary increment from the linearized LSPG objective gives

    min_b ||P r + C b||_2^2.

Instead of solving for all 141 secondary coordinates, build

    B_r = orth K_r(C^T C, C^T P r).

Here K_r(H,g) = span{g,Hg,...,H^(r-1)g}. The code uses Jacobian actions and
twice-reorthogonalized Arnoldi; it does not form C or the 141-column J Vs.
Freeze B_r and solve the nonlinear affine LSPG problem above with 10+r
unknowns. Construct a fresh B_r at the next time step.

For r=1 the secondary linearized update is the exact line minimizer along
the Schur-complement negative gradient. Higher ranks enlarge this Krylov
space. Unit tests compare the full-rank linearized update with a direct
151-coordinate-style least-squares solve on a small problem, and check the
rank-zero rollout against the existing production Case-2 solver.

This is a residual-dependent *search-space construction*, not a new neural
decoder whose state derivative has been omitted. Since B_r is frozen before
the nonlinear solve, [Vp,Vs B_r] is its exact tangent. A single direction
can affect all 141 secondary coordinates.

Important limits:

- Rank zero is the original Case 2.
- With the entire secondary space, the affine trial space is the linear
  PROM space. Its trajectory also matches the linear PROM only when the
  initialization and nonlinear convergence choices match.
- Reducing the residual for a fixed predecessor does not prove that a
  free-running trajectory is more accurate against the HDM.
- Direction construction touches all tail modes and the full residual.
  Ten plus r unknowns does not mean the cost is identical to a fixed
  (10+r)-mode PROM. Timings must include this extra work.
- Euclidean coefficient norms are justified by V^T V = I, checked on load.
  The implementation intentionally rejects a nonorthonormal MLSPG basis.

## 3. Affine primary-innovation feedback

An alternative with no additional online unknowns is

    tail(q,mu,t) = M_s(mu,t) + K (q - M_p(mu,t)).

The primary coordinates remain residual-solved. K is a constant 141-by-10
matrix fitted offline, so the tangent is Vp + Vs K and no ANN derivative is
needed online. With training errors Ep = q_ref,p - M_p and
Es = q_ref,s - M_s, solve the precise ridge problem

    min_K ||Es - K Ep||_F^2 + lambda ||K||_F^2,
    lambda = rho trace(Ep Ep^T)/10.

The unregularized solution uses an SVD least-squares solve. This is a
linear predictor of correlated primary/tail errors, not an arbitrary
penalty on the online PDE residual. It can fail if those errors are weakly
correlated or if the relationship does not generalize.

The feedback manifold changes the initial least-squares representation of
the initial condition. Its runner explicitly records this distinction.

## Commands

Run from the repository root; the local CPU dependencies are in `.venv`.

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q tests/test_case2_residual_correction.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -u Project_YvonMaday/run_case2_local_corrections.py --stage validation
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -u Project_YvonMaday/run_case2_affine_feedback.py --stage validation --ridges 0 1
```

Outputs go to `Results_Paper/euclidean_case2_local_corrections/` and
`Results_Paper/euclidean_case2_affine_feedback/`. Numerical conclusions must
be taken from completed full-trajectory summaries, not short smoke tests.

## Additional controls and reporting

The first validation screen selects rank 3 from ranks 1, 3 and 5 by mean
trajectory coefficient error at the two validation parameters. The
input fingerprints and frozen selection are stored in
`Results_Paper/euclidean_case2_confirmatory/selection.json` before running
the reporting points.

The existing production Case-1 and Case-3 solvers are also run locally for
comparison. `run_case2_local_reference_benchmarks.py` wraps their saved
networks in the same vector adapters used by the production runners. Its
`krylov3_zero` ablation removes ANN predictions after t=0, while retaining
the common ANN-based initialization. This tests the contribution of the
ANN forecast rather than attributing every gain to residual correction.

A second ablation sets the initial reduced state to the least-squares
projection of the *known initial condition* onto all 151 Euclidean modes:

    qN(0) = Vtot^T (u0 - u_ref).

The orthonormal basis makes this the unique Euclidean best approximation.
The solver uses this as the predecessor of step 1, and uses the ANN-offset
trial manifold for every subsequent time step. There is no requirement for
the initial state to lie on the t>0 ANN manifold. It is not an HDM-trained
online target or teacher forcing. The projected initial condition is not
exactly representable here, so this is called `linear` or `known initial`,
not `exact initial`.

This initial projection is not a 151-coordinate time-dependent residual
solve. It can be precomputed once for the prescribed initial condition;
the current timing includes its setup. Every subsequent adaptive solve has
only ten primary and three correction amplitudes.

Both baseline Case 2 and adaptive rank 3 are tested with this initialization.
They are kept separate from the original-initialization runs, so a benefit
from initialization is not incorrectly credited to adaptive directions.

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -u Project_YvonMaday/run_case2_local_corrections.py --stage validation --methods baseline krylov3 --initialization linear --output Project_YvonMaday/Results_Paper/euclidean_case2_known_initial
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python Project_YvonMaday/summarize_case2_local_corrections.py
```

The summary script regenerates `euclidean_case2_confirmatory/results.md`,
CSV tables and comparison figures. It includes only completed runs and
keeps the pre-existing manuscript reference errors separate from new runs.

`run_case2_local_reference_benchmarks.py` additionally provides
`case1_known_initial` and `case3_known_initial` controls. They use the same
projected initial state as the corrected Case 2 and then the unchanged
nonlinear decoders, their full state-dependent tangents, and the native
Gauss-Newton update. Separate small-grid tests check these controls against
an equivalent affine closure with a known tangent. Their outputs are saved
under `euclidean_case2_matched_initial_references`.

The auxiliary `primary_error_percent` in the summary files consistently
uses coordinates 1--10, including for the n=20 timing control. Full
coefficient and HDM state errors use all 151 coordinates.

For provenance, the report's `source_snapshot` directory contains the
initial validation runner and the runner extended with known-initial-state
controls. The earlier runner's SHA-256 matches the initial validation
summaries. If the current runner is used to repeat that first screen, use
a fresh output directory: the strict fingerprint check deliberately
rejects resuming a run generated by a different implementation version.
