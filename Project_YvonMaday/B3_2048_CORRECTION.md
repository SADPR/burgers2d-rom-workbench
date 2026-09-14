# B3 enrichment correction (2026-09-14)

Scope: rerun only Case 2+B3 for the Euclidean HPROM enrichment checkpoints
lhs8, lhs12 and lhs18, with 2048 importance draws instead of the hardcoded 4096.
This is a candidate-draw budget, not a prescribed number of positive cells.
The number of retained cells is determined by the nonnegative fit.

## Changes

- `Results_Paper/scripts/run_euclidean_hprom_case2_b3.sh`: configurable draw
  budget (default 2048), explicit seed 418, rule-specific logs and selection,
  predictor reuse, positive thread-count checks, and validation fingerprints
  checked before reporting. Offline fitting/auditing defaults to 24 threads;
  validation/reporting defaults to one thread.
- `Results_Paper/scripts/run_euclidean_hprom_nested_enrichment_b3.sh`:
  explicit 2048 and predictor-reuse defaults for each enriched checkpoint.
- `run_case2_hyperreduction.py`: expose the already tested predictor-reuse
  implementation via `--reuse-predictor`, and record it in evaluation metadata.
  The weighted residual, tangent and Krylov construction are unchanged.
- `validate_case2_hyper_rule.py`: load validation trajectories for exactly the
  requested rule, check weight/input fingerprints and held-out audit coverage,
  and optionally write a separate selection manifest atomically.
- `run_euclidean_hprom_b3_2048_correction.sh`: allocated-node launcher for all
  three enrichments, or a named subset. It pins the existing baseline teacher
  and reference datasets, forbids overwriting completed runs, and continues
  with the next enrichment if one fails. Reinvoke through an interactive `if`
  block to keep shell error handling from closing the allocated session.
- `../tests/test_case2_b3_2048_campaign.py`: regression tests for rule
  isolation, validation failure/staleness, predictor reuse, resumption and
  launcher arguments/thread counts.

## What Is Rerun

For each enriched ANN, construct the seed-418 2048-draw candidate support,
assemble the same moments from the nine existing baseline linear-HPROM
trajectories, and fit new nonnegative weights. The 4096 matrices are not
reused: the new support/prior and new fit are constructed explicitly.
No new training trajectories or neural-network training are required.
The baseline 2048 B3 benchmark is not repeated.

The frozen rule must pass the unchanged 14-anchor held-out operator audit:
Gram eigenvalues in [0.8, 1.2], relative coordinate-gradient error <= 0.05,
and sampled/full B3 linearized-residual ratio <= 1.05. The full comparator
in this ratio uses the same B3 update dimension, not all 151 coordinates.
Two held-out validation trajectories are then evaluated. Four reporting
trajectories run only if that rule passes. Thus the complete correction has
at most 18 trajectories (three enrichments times six), without full-trajectory
warmups or timing repetitions. These checks are validation criteria, not a
proof of an a priori global accuracy bound.

## Outputs And Resumption

Within each `Results_Paper/euclidean_hprom_enrichment_lhs{8,12,18}` root:

- `Stage4/case2_b3/leverage2048/`
- `Stage4/case2_b3/positive_fit2048/`, including its own `selection.json`
- `Stage4/case2_b3/validation_positive_fit2048_hprom3_p{0,1}_steps500/`
- `Stage4/case2_b3/reporting_positive_fit2048_hprom3_p{0,1,2,3}_steps500/`
- `logs/case2_b3/positive_fit2048/`

Existing 4096 weights, results, logs and root-level selection remain intact.
An identical completed fit or trajectory is skipped on resumption; changed
configurations are rejected rather than silently overwritten. The operator
audit and selection gate are checked again. Interrupted fits reuse their own
completed 2048 moment matrix/target if present, but restart the optimizer.

The manuscript and its current 4096-enrichment asset generator are not
changed by this correction. Switch their input paths and regenerate figures
and tables after the new 2048 results have been downloaded and checked.
