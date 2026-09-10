# Final model hyperparameters

Harvested on 2026-09-10 from the artifacts each table row was reported with,
not from notes: the stage summaries (`stage*_summary.txt`), the offline
metadata (`*_metadata.npz`) and the per-point online summaries. Every local
candidate was matched to its table row by `N_e` **and** by the reported
avg/max error, because `N_e` alone is ambiguous — in the POD-GPR sweep it is
set by `n_primary`, so `nc5/nprimary6` and `nc8/nprimary6` share `N_e = 541`.

Each knob was tuned to hit the same accuracy target, so the families are
compared at matched accuracy rather than at matched cost: **max relative state
error of 1% ± 0.2%** over the three test points.

## Common to every model

| Setting | Value |
|---|---|
| Mesh | 250 × 250 cells, `N` = 125 000 (two fields) |
| Time integration | implicit, `dt` = 0.05, 500 steps |
| Training set | 9 parameters, μ₁ ∈ {4.25, 4.875, 5.5} × μ₂ ∈ {0.015, 0.0225, 0.03} → 4 509 snapshots |
| Test points | (4.56, 0.019), (4.75, 0.020), (5.19, 0.026) |
| ECSW snapshots | 2% of candidates, seed 42, SVD, rel. tol 1e-8 |
| ECSW policy | `global_param_time_stratified` (global) / `local_cluster_param_time_stratified` (local) |
| Gauss–Newton | `relnorm_cutoff` = 1e-5, `min_delta` = 1e-2, `max_its` = 20 |
| Local clustering | k-means, `random_state` = 1, overlap φ = 0.1 |

## Global models

| Model | Offline knobs | Resulting dimension | `N_e` |
|---|---|---|---|
| HPROM | `pod_tol` = 5e-4 | n = 96 of 4 509 available (energy 0.999503) | 4 824 |
| HQPROM | `pod_tol` = 1e-4, `zeta_qua` = 1.5, `ridge_alpha` = 1.0, `q_norm` = std, centred | n = 39 (from `n_trad` = 151, `n_max_ls` = 94) | 3 505 |
| HPROM-GPR | Matérn ν=3/2, `alpha` = 1e-10, `length_scale₀` = 1.0 in [0.01, 5], `constant_value` in [1e-3, 1e3], 3 restarts, `normalize_y` = False, duplicate tol 1e-3, seed 42 | n_p = 20, n_s = 131 (basis 151) | 1 801 |

## Local models

| Model | `N_c` | Offline knobs | Modes per cluster | `N_e` |
|---|---|---|---|---|
| Local HPROM | 3 | `pod_tol` = 1.0e-3 | [48, 35, 44] | 3 405 |
| Local HPROM | 5 | `pod_tol` = 1.5e-3 | [23, 31, 29, 25, 27] | 2 384 |
| Local HPROM | 8 | `pod_tol` = 1.5e-3 | [26, 17, 20, 24, 18, 16, 17, 21] | 1 781 |
| Local HQPROM | 3 | `zeta_qua` = 1.2, `ridge_alpha` = 1e6 | [27, 21, 24] | 2 152 |
| Local HQPROM | 5 | `zeta_qua` = 1.0, `ridge_alpha` = 1e4 | [15, 20, 20, 16, 18] | 1 593 |
| Local HQPROM | 8 | `zeta_qua` = 0.7, `ridge_alpha` = 1e4 | [15, 11, 12, 14, 11, 10, 10, 13] | 1 075 |
| Local HPROM-GPR | 3 | `n_primary` = 9, `eps2_pod` = 1e-4, Matérn ν=3/2, α per cluster [1e-10, 1e-8, 1e-10] | r = [97, 60, 79] | 811 |
| Local HPROM-GPR | 5 | `n_primary` = 7, `eps2_pod` = 1e-4, Matérn ν=3/2, α per cluster [1e-8, 1e-10, 1e-10, 1e-10, 1e-10] | r = [41, 68, 65, 47, 54] | 631 |
| Local HPROM-GPR | 8 | `n_primary` = 6, `eps2_pod` = 1e-4, Matérn ν=3/2, α = 1e-10 in every cluster | r = [54, 31, 40, 52, 34, 28, 30, 46] | 541 |

The POD-GPR kernel is selected per cluster from a one-element candidate list
(`matern15`), with `alpha` chosen per cluster from {1e-10, 1e-8} by up to
5-fold cross-validation; the α columns above are what that search returned.

## Where each model lives

| Row | Path |
|---|---|
| HPROM (global) | `POD/` |
| HQPROM (global) | `Quadratic/` |
| HPROM-GPR (global) | `POD-GPR/pod_gpr_model/` |
| Local HPROM `N_c`=3/5/8 | `LocalPODSweep/nc{3,5,8}_postfix/tol{1.0,1.5,1.5}e-03/` |
| Local HQPROM `N_c`=3 | `LocalQuadraticSweep/nc3_alphahigh/z1.2_a1e06/` |
| Local HQPROM `N_c`=5 | `LocalQuadraticSweep/local_hqprom_sweep_nc5_postfix/z1_a1e04/` |
| Local HQPROM `N_c`=8 | `LocalQuadraticSweep/local_hqprom_sweep_nc8_postfix/z0.7_a1e04/` |
| Local HPROM-GPR `N_c`=3 | `LocalPOD-GPRSweep/nc3_check/nprimary9/` |
| Local HPROM-GPR `N_c`=5 | `LocalPOD-GPRSweep/nc5/nprimary7/` |
| Local HPROM-GPR `N_c`=8 | `LocalPOD-GPRSweep/nc8/nprimary6/` |

## Caveat on the two global rows

The nine local rows were verified on 2026-09-09 to reproduce their published
avg/max errors exactly from the artifacts above. Two global rows do not:

- **HQPROM (global).** The published row is from a run on 2026-04-30. The
  current `Quadratic/` model plus `Results/hqprom_ecsw_weights.npy` give
  0.784 / 0.807 / 0.861 % instead of avg 0.871 / max 0.933 %. Model, weights
  and code all match between the local machine and Sherlock, and the online
  solver is byte-identical to its 2026-08-06 version, so the model was
  regenerated after April and the knobs listed above are the *current* ones.
- **HPROM-GPR (global).** The published row is from 2026-05-01, but
  `stage3_train_gpr_summary.txt` is dated 2026-05-14 — the GPR was retrained
  afterwards. The published summary records
  `learned_kernel: 9.21**2 * Matern(length_scale=5, nu=1.5)`; the current
  local model yields `9.95**2`. Sherlock holds a third copy with a different
  checksum. The knobs above describe the retrained model.

**HPROM (global)** does reproduce its row exactly (avg 1.030 / max 1.096 %),
so its entry is trustworthy.

Note also that `compute_ECSW_training_matrix_2D` (global linear) was corrected
on 2026-09-09 to evaluate the residual at the *projected* snapshot, matching
what the quadratic, POD-GPR and local assemblies already did. Its `N_e` = 4 824
above therefore predates the fix and will change when the weights are rebuilt.
