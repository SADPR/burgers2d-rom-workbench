# Local ECM study for Case 2+B3

This experiment compares a dedicated ECM rule with the accepted baseline
2048-draw positive moment fit. It uses the baseline HPROM-trained master ANN,
the common 151-column basis, ten primary coordinates, and three online Krylov
directions. No ANN is retrained. Production results and the manuscript are
not overwritten.

## Offline construction

The nine existing linear-HPROM training trajectories supply the same 90
parameter-time anchors used by the accepted fit. The cell-moment matrix
contains the complete-coordinate residual gradient, the adaptive 13-column
Jacobian Gram, residual energy, and the complete trial-basis Gram. Blocks
have the same normalization as the accepted fit. Assembly is checked against
the existing cached moments and their full-mesh sums.

Every mesh cell is eligible for ECM. There is no candidate draw count and no
importance-sampling restriction. The large matrix is stored on disk.

An adaptive randomized range finder with reorthogonalization and one power
iteration compresses this matrix. The retained rank is set by the relative
Frobenius projection error of the **complete** matrix, including energy
outside the randomized range. A small eigendecomposition selects its dominant
compressed singular directions. This tolerance is not the relative
singular-value cutoff of the conventional campaign. A rank cap is only a
resource limit: hitting it without satisfying the tolerance aborts instead
of accepting an inaccurate compression.

The constant is included explicitly in the orthonormal integration basis.
The existing EmpiricalCubatureMethod selector computes positive cell weights,
with relative compressed integration tolerance 1e-8. Positivity, the actual
compressed integration residual, and errors in the original moment blocks
are checked independently of the selector's own stopping flag.

## Acceptance

Only the two external validation parameters are used to select a rule.
The same fourteen operator anchors as the accepted study must satisfy:

- Complete-coordinate Jacobian Gram generalized eigenvalues in [0.8, 1.2].
- Maximum relative full-coordinate gradient error at most 0.05.
- Maximum sampled/full linearized residual ratio at most 1.05, with both
  updates evaluated in the full residual and each using its own B3 directions.

A rule passing those checks advances both complete 500-step validation
trajectories. Each coefficient error must be at most 1.05 times the accepted
2048-draw rule's error, rerun with the same local model and solver.

Tolerances 0.1, 0.05, and 0.02 are tried in that order. The first passing rule
is frozen before accessing the four reporting trajectories. This finds an
acceptable tested rule, not a proof of minimum support. Reporting results
do not change the selection. All local times are diagnostic; official online
cost comparisons require a separate Sherlock deployment.

After the first three rules were rejected using validation only, the local
study was extended to tolerances 0.01, 0.005, and 0.002, with unchanged
acceptance criteria. The original protocol and rejection are archived in
`rejected_protocol_3.json`. This is sequential validation-based development;
the reporting points remain untouched until a rule is accepted.

## Execution

From the repository root:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -u Project_YvonMaday/study_case2_b3_ecm.py --threads 4
```

The default result directory is
`Project_YvonMaday/Results_Paper/euclidean_b3_ecm_local/`.
Use `--stage moments` for moment assembly only. Repeating the same command
reuses completed assembly, compression, rule fits, audits and trajectories.
The full moment matrix occupies approximately 16.7 GB; it is an offline
artifact and is not needed to deploy the selected weights.

To resume the extended study:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -u Project_YvonMaday/study_case2_b3_ecm.py --threads 4 \
  --tolerances .1 .05 .02 .01 .005 .002 --extend-rejected
```

Outputs include `moments.json`, per-rule `fit.json` and
`validation_operator_audit.json`, `validation_comparison.json`,
`selection.json`, and, only on acceptance, `reporting_comparison.json`.
The accepted manuscript rule remains in its original directory.
