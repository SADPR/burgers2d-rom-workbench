# Project Memory: Maday Metric-Aware HPROM/ANN Campaign

Last update: 2026-06-04

Purpose: keep a reproducible record of the main decisions, commands, outputs, and open checks for the CMAME-level manuscript campaign. This file is meant to be appended as we advance.

## 0. Current Working Thesis

We are moving the final paper campaign away from the old Euclidean/PROM-only pipeline and toward:

- Basis: LSPG-sensitive metric basis for the main production campaign.
- Backend: HPROM, not PROM, for the main data used to train and evaluate online ANN models.
- Learning models: ANN only for the final campaign. Dense GPR, sparse GPR, and RBF tests were useful diagnostics but are not currently part of the final production path.
- Main correction/data-driven tests: compare data-driven ANN and later Case 1/2/3 ANN strategies on the same LSPG-sensitive HPROM dataset.
- Important caveat: LSPG-sensitive metric is not claimed as universally superior. It is selected because it reduced low-mode contamination in oracle high-mode perturbation tests for Case 2, and it must still be checked against Linear HPROM, Case 1, Case 3, and POD-AE/POD-DL style variants.

Notation decision for manuscript:

- Use `W` for the metric/weighting operator.
- Use `C` for the sensitivity/cross-coupling object.
- Avoid heavy names like `W_MLSPG` unless needed internally in code/scripts.

## 1. Euclidean vs LSPG-Sensitive Metric Study

### 1.1 Goal

We first needed a clean justification for why changing the POD metric is relevant before running the full nonlinear closure campaign. The diagnostic question was:

> If high-mode coefficients are perturbed, does the metric choice change how much the low resolved coordinates are polluted through the LSPG solve?

This was tested through oracle Case 2 runs using the same target linear coordinates, with controlled perturbations of the high block.

### 1.2 Basis Artifacts

Euclidean basis summary:

```text
Project_YvonMaday/Results_Paper/MetricStudy/euclidean/Stage1/stage1_euclidean_summary.txt
n_keep = 151
n_available = 4509
energy_captured = 9.99900284e-01
energy_lost = 9.97164041e-05
pod_reconstruction_relative_error = 3.87857543e-03
```

LSPG-sensitive basis summary:

```text
Project_YvonMaday/Results_Paper/MetricStudy/lspg_sensitive/Stage1/stage1_lspg_sensitive_summary.txt
n_keep = 151
n_available = 1508
energy_captured = 9.99882797e-01
energy_lost = 1.17203232e-04
reconstruction_rel_err_mu_test = 3.87182481e-03
metric_samples_used = 2250
metric_samples_candidates = 4500
metric_time_weighting = trapezoid
eps_mode = trace_ratio
eps_ratio = 1e-10
```

Important note:

- The first direct rebuild of the LSPG-sensitive metric basis was too expensive locally. One run showed roughly 8 hours estimated wall time for the full metric assembly. The usable basis was therefore copied/reused from the previous completed metric study rather than rebuilt from scratch locally.

Command attempted for LSPG-sensitive basis construction:

```bash
cd /home/kratos/burgers2d-rom-workbench/Project_YvonMaday
python3 -u stage1_lspg_proposal1_weff_pod.py \
  --maday-results-root Results_Paper/MetricStudy \
  --maday-tag lspg_sensitive \
  --snapshot-select-mode strided \
  --snapshot-time-stride 1 \
  --metric-select-mode ecsw_param_time_stratified \
  --metric-percent 50 \
  --metric-time-offset 1 \
  --metric-time-weighting trapezoid \
  --eps-mode trace_ratio \
  --eps-ratio 1e-10 \
  --pod-tol 1e-4 \
  --basis-name basis.npy \
  --sigma-name sigma.npy \
  --uref-name u_ref.npy \
  --summary-name stage1_lspg_sensitive_summary.txt \
  --decay-plot-name stage1_lspg_sensitive_decay.png \
  --progress-every 25 \
  2>&1 | tee Results_Paper/MetricStudy/lspg_sensitive/stage1_lspg_sensitive_build.log
```

### 1.3 Oracle Perturbation Protocol

We used three evaluation points:

```text
mu = (4.560, 0.0190)  off-grid test
mu = (4.875, 0.0225)  in-grid verification
mu = (5.190, 0.0260)  off-grid test
```

Perturbation levels:

```text
0%, 1%, 2%
```

For 1% and 2%, five seeds were used for each basis and each point.

Sanity check used for every perturbation level:

```bash
cd /home/kratos/burgers2d-rom-workbench/Project_YvonMaday
OUTROOT="Results_Paper/MetricStudy/oracle_euclidean_vs_lspg"

grep -R "linear_qn_reconstruction_rel_error" "$OUTROOT"/*/pert*pct*/*_summary.txt
grep -R "linear_qn_source" "$OUTROOT"/*/pert*pct*/*_summary.txt
```

Interpretation:

- The strict/oracle construction must not silently use inconsistent linear coordinates.
- If a summary reports a large reconstruction inconsistency, that run must not be used as final evidence without understanding the source.

### 1.4 Aggregated Oracle Results

Aggregated over three points and five seeds where applicable.

| Basis | Perturbation | baseline err vs HDM (%) | err vs linear after perturbation (%) | low-q err vs linear after perturbation (%) |
|---|---:|---:|---:|---:|
| Euclidean | 0% | 0.441 | n/a | n/a |
| Euclidean | 1% | 0.441 | 0.0910 | 0.0140 |
| Euclidean | 2% | 0.441 | 0.1827 | 0.0450 |
| LSPG-sensitive | 0% | 0.448 | n/a | n/a |
| LSPG-sensitive | 1% | 0.448 | 0.0882 | 0.0117 |
| LSPG-sensitive | 2% | 0.448 | 0.1767 | 0.0347 |

Conclusion from this first diagnostic:

- LSPG-sensitive basis has nearly the same unperturbed oracle accuracy as Euclidean POD.
- Under high-block perturbation, LSPG-sensitive basis reduces the induced low-coordinate error.
- The effect is modest in full-state error but clearer in the low-coordinate error.
- This justifies using LSPG-sensitive basis for the main Case 2 closure campaign, but not as a universal conclusion.

## 2. Transition to Main HPROM Campaign

### 2.1 Reason for HPROM Dataset

The final campaign should not train ANN closures using old PROM/Euclidean data if the paper is centered on the metric-aware HPROM setting. Therefore we moved to:

```text
Results_Paper/mlspg_hprom_main
```

This directory is the current production root for the final LSPG-sensitive HPROM campaign.

### 2.2 Stage 2 HPROM Coefficient Dataset

Script:

```text
Project_YvonMaday/Results_Paper/scripts/run_mlspg_hprom_main_stage2_only.sh
```

Command on Sherlock:

```bash
cd /scratch/users/sadpr/Code12May/burgers2d-rom-workbench/Project_YvonMaday
bash Results_Paper/scripts/run_mlspg_hprom_main_stage2_only.sh
```

The script runs:

```bash
python3 -u stage2_build_prom_qn_dataset.py \
  --backend hprom \
  --total-modes 151 \
  --basis-path "$BASIS" \
  --u-ref-path "$UREF" \
  --output-dir "$DATASET_DIR" \
  --ecsw-weights-dir "$ECSW_DIR" \
  --rebuild-ecsw \
  --ecsw-snapshot-percent 2.0 \
  --ecsw-num-training-mu 9 \
  --ecsw-snap-time-offset 3 \
  --ecsw-random-seed 42 \
  --ecsw-ensure-mu-coverage \
  --linear-solver lstsq \
  --normal-eq-reg 1e-12 \
  --max-its 20 \
  --relnorm-cutoff 1e-5 \
  --min-delta 1e-2 \
  --no-save-rom-snaps \
  --no-plots
```

Stage 2 check file:

```text
Project_YvonMaday/Results_Paper/mlspg_hprom_main/logs/stage2_hprom_qn_ntot151_check.txt
```

Recorded values:

```text
solve_backend: hprom
dataset_dir: Results_Paper/mlspg_hprom_main/Stage2/prom_coeff_dataset_ntot151
basis_path: Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy
u_ref_path: Results_Paper/MetricStudy/lspg_sensitive/Stage1/u_ref.npy
total_modes: 151
ecsw_snapshot_percent: 2.0
ecsw_num_selected_total: 90
ecsw_residual: 2.015664567911159e-09
n_ecsw_elements: 5975
```

Operational conclusion:

- Stage 2 is correctly using the LSPG-sensitive basis and HPROM backend.
- The dataset directory to use for ANN training is:

```text
Results_Paper/mlspg_hprom_main/Stage2/prom_coeff_dataset_ntot151
```

## 3. Linear HPROM at Three Points

Script:

```text
Project_YvonMaday/Results_Paper/scripts/run_mlspg_hprom_main_linear_3pts.sh
```

Command on Sherlock:

```bash
cd /scratch/users/sadpr/Code12May/burgers2d-rom-workbench/Project_YvonMaday
bash Results_Paper/scripts/run_mlspg_hprom_main_linear_3pts.sh
```

The script runs Linear HPROM at:

```text
mu = (4.560, 0.0190)
mu = (4.875, 0.0225)
mu = (5.190, 0.0260)
```

with explicit artifacts:

```bash
BASIS="$PWD/Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy"
UREF="$PWD/Results_Paper/MetricStudy/lspg_sensitive/Stage1/u_ref.npy"
ECSW_WEIGHTS="$PWD/Results_Paper/mlspg_hprom_main/Stage2/ecsw/ecsw_weights_lspg_ntot151.npy"
```

Quick summary output:

```text
Results_Paper/mlspg_hprom_main/logs/linear_hprom_3pts_quick_summary.txt
```

These results are needed as the linear reference for final online ANN comparisons.

## 4. Data-Driven ANN Architecture Sweep

### 4.1 Goal

Before running all Case 1/2/3 ANN variants, we first tested the data-driven map:

```text
(mu1, mu2, t) -> q_N in R^151
```

using the HPROM coefficient dataset.

### 4.2 Sweep Script

Script:

```text
Project_YvonMaday/Results_Paper/scripts/run_mlspg_hprom_data_driven_arch_sweep.sh
```

Command on Sherlock:

```bash
cd /scratch/users/sadpr/Code12May/burgers2d-rom-workbench/Project_YvonMaday
bash Results_Paper/scripts/run_mlspg_hprom_data_driven_arch_sweep.sh
```

Summary CSV:

```text
Results_Paper/mlspg_hprom_main/logs/data_driven_arch_sweep/data_driven_arch_sweep_summary.csv
```

Top five recorded models:

| Label | Architecture | Activation | Batch | LR | val rel Frobenius (%) | best val MSE |
|---|---|---|---:|---:|---:|---:|
| A10_silu_wide_b128_lr5e4 | 256,512,512,256 | SiLU | 128 | 5e-4 | 0.9075 | 0.09029 |
| A05_wide_b128_lr5e4 | 256,512,512,256 | ELU | 128 | 5e-4 | 1.4362 | 0.22614 |
| A00_current_b128_lr1e3 | 32,64,128,256,256 | ELU | 128 | 1e-3 | 1.4697 | 0.23681 |
| A06_wide_b256_lr5e4 | 256,512,512,256 | ELU | 256 | 5e-4 | 1.4906 | 0.24360 |
| A07_deep_wide_b128_lr5e4 | 128,256,512,512,256,128 | ELU | 128 | 5e-4 | 1.5654 | 0.26865 |

Decision:

- Use A10 as current best data-driven ANN candidate for the online three-point prediction test.
- A10 is about 3.82x larger than the baseline `32,64,128,256,256` model in trainable parameters.
- Evaluation cost increases in the neural forward pass, but full online cost is still dominated by reconstruction and plotting, not by the MLP itself.

## 5. A10 Data-Driven Prediction at Three Points

Script:

```text
Project_YvonMaday/Results_Paper/scripts/run_mlspg_hprom_data_driven_A10_3pts.sh
```

Command on Sherlock:

```bash
cd /scratch/users/sadpr/Code12May/burgers2d-rom-workbench/Project_YvonMaday
bash Results_Paper/scripts/run_mlspg_hprom_data_driven_A10_3pts.sh
```

The script uses:

```text
MODEL = Results_Paper/mlspg_hprom_main/Stage3/models/rom_data_driven_ann_mu_t_ntot151_A10_silu_wide_b128_lr5e4.pt
BASIS = Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy
UREF  = Results_Paper/MetricStudy/lspg_sensitive/Stage1/u_ref.npy
```

The script runs:

```text
mu = (4.560, 0.0190)
mu = (4.875, 0.0225)
mu = (5.190, 0.0260)
```

Outputs:

```text
Results_Paper/mlspg_hprom_main/Runs/DataDriven_A10
Results_Paper/mlspg_hprom_main/logs/data_driven_A10_online/data_driven_A10_3pts_quick_summary.txt
```

Disk policy:

- The script uses `--no-save-rom-snaps` to avoid storing full `rom_snaps.npy` files.
- It still stores `qN.npy`, time vectors, summaries, and plots.
- If state overlays later require full reconstructed snapshots, either reconstruct from `qN.npy + basis.npy + u_ref.npy`, or rerun without `--no-save-rom-snaps`.

## 6. Commands and Checks to Keep Reusing

### 6.1 Check Stage 2 Dataset Backend and Basis

```bash
cd /scratch/users/sadpr/Code12May/burgers2d-rom-workbench/Project_YvonMaday
cat Results_Paper/mlspg_hprom_main/logs/stage2_hprom_qn_ntot151_check.txt
```

Expected critical lines:

```text
solve_backend: hprom
basis_path: .../Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy
u_ref_path: .../Results_Paper/MetricStudy/lspg_sensitive/Stage1/u_ref.npy
```

### 6.2 Check A10 Prediction Summary

```bash
cd /scratch/users/sadpr/Code12May/burgers2d-rom-workbench/Project_YvonMaday
cat Results_Paper/mlspg_hprom_main/logs/data_driven_A10_online/data_driven_A10_3pts_quick_summary.txt
```

### 6.3 Check Architecture Sweep Winner

```bash
cd /scratch/users/sadpr/Code12May/burgers2d-rom-workbench/Project_YvonMaday
sort -t, -k3,3n Results_Paper/mlspg_hprom_main/logs/data_driven_arch_sweep/data_driven_arch_sweep_summary.csv | head
```

If CSV columns change, inspect manually:

```bash
python3 - <<'PY'
import pandas as pd
p = 'Results_Paper/mlspg_hprom_main/logs/data_driven_arch_sweep/data_driven_arch_sweep_summary.csv'
df = pd.read_csv(p)
print(df.sort_values('val_rel_frob_percent').head(10).to_string(index=False))
PY
```

## 7. Manuscript Status

Main file:

```text
Project_YvonMaday/Results_Paper/manuscript.tex
```

Current title direction:

```text
Robust nonlinear projection-based reduced-order models: comparative assessment of closure and manifold strategies
```

Important manuscript edits already made:

- Added a body section on metric-aware basis construction for closure robustness.
- Added the Euclidean vs LSPG-sensitive oracle table.
- Reframed the metric as a diagnostic/design choice, not as the main contribution alone.
- Need to merge the clearer writing style and experimental explanations from:

```text
Project_YvonMaday/250x250/report.tex
```

while replacing old PROM/Euclidean assumptions by the new HPROM/LSPG-sensitive campaign where appropriate.

Known compile caveat:

- `manuscript.tex` may fail or warn until all final figures from the new campaign are generated and paths are updated.

## 8. Things We Decided Not to Carry Forward for Now

Dense GPR, sparse GPR, and RBF experiments:

- Useful as diagnostics.
- Not reliable enough for the current final pipeline.
- Do not include as primary final results unless we later decide to discuss them briefly as negative evidence.

Small cropped tests with `ntot=41`:

- Useful debugging campaign.
- Not part of the final paper pipeline.
- Main campaign returns to `ntot=151`.

Row split vs mu-group split confusion:

- For final ANN sweep, coefficient validation is still useful, but final acceptance must be based on the three online points and state-space behavior.
- We should avoid overclaiming based only on coefficient validation MSE.

## 9. Next Planned Steps

Immediate next steps after A10 three-point prediction:

1. Pull or inspect A10 online summaries for the three points.
2. Compare A10 data-driven states against HDM and Linear HPROM.
3. Decide whether A10 is sufficient as data-driven baseline.
4. Train Case 1/2/3 ANN variants on the same HPROM dataset using comparable architecture/hyperparameters, likely starting from A10-style width and SiLU if compatible.
5. Generate final figures with consistent colors:
   - HDM: black
   - Linear PROM/HPROM: red
   - ANN 131: blue
   - ANN 141: green
   - ANN 151/data-driven: darkgoldenrod or other non-magenta high-contrast color
6. Update `manuscript.tex` with final tables and plots.

## 10. Append Log

### 2026-06-04

- Created portable A10 prediction script:

```text
Project_YvonMaday/Results_Paper/scripts/run_mlspg_hprom_data_driven_A10_3pts.sh
```

- Created this project memory file.

- Compared A10 against the baseline data-driven ANN A00 on the three online state-reconstruction points.

| Model | Architecture | Activation | `mu=(4.560,0.0190)` err vs HDM (%) | `mu=(4.875,0.0225)` err vs HDM (%) | `mu=(5.190,0.0260)` err vs HDM (%) | Mean (%) |
|---|---|---|---:|---:|---:|---:|
| A00 | 32,64,128,256,256 | ELU | 2.21 | 0.62 | 2.39 | 1.74 |
| A10 | 256,512,512,256 | SiLU | 1.93 | 0.48 | 2.40 | 1.60 |

Interpretation:

- A10 is better on two points: `(4.560,0.0190)` and `(4.875,0.0225)`.
- At `(5.190,0.0260)`, A10 and A00 are essentially tied; A10 is marginally worse by about `0.01%` absolute.
- The coefficient-validation gain of A10 is clear, but the final state-space gain is moderate rather than dramatic.
- Current practical decision: keep A10 as the stronger data-driven ANN candidate, but do not overclaim from these three state errors alone.

- Added Case 2 architecture sweep infrastructure:

```text
Project_YvonMaday/Results_Paper/scripts/run_mlspg_hprom_case2_arch_sweep.sh
```

The sweep trains Case 2 maps:

```text
(mu1, mu2, t) -> q_{n+1:151}
```

for:

```text
n = 10  (secondary output dimension 141)
n = 20  (secondary output dimension 131)
```

using five architecture families:

| Label | Hidden dimensions | Activation | Batch | LR |
|---|---|---|---:|---:|
| B00_current_b128_lr1e3 | 32,64,128,256,256 | ELU | 128 | 1e-3 |
| B01_A10_like_b128_lr5e4 | 256,512,512,256 | SiLU | 128 | 5e-4 |
| B02_A10_elu_b128_lr5e4 | 256,512,512,256 | ELU | 128 | 5e-4 |
| B03_medium_silu_b128_lr1e3 | 128,256,256,128 | SiLU | 128 | 1e-3 |
| B04_deep_silu_b128_lr5e4 | 128,256,512,512,256,128 | SiLU | 128 | 5e-4 |

Total trainings:

```text
5 architectures x 2 primary-mode choices = 10 trainings
```

The trainer now records:

```text
train_rel_frob_percent
val_rel_frob_percent
trainable_parameters
```

so Case 2 architectures can be ranked similarly to the data-driven ANN sweep.

- Case 2 sweep result received from Sherlock:

```text
best primary=10:
  np10_B01_A10_like_b128_lr5e4  val_rel_frob_percent = 2.3365
  np10_B00_current_b128_lr1e3   val_rel_frob_percent = 3.3968
  np10_B02_A10_elu_b128_lr5e4   val_rel_frob_percent = 3.4086

best primary=20:
  np20_B01_A10_like_b128_lr5e4  val_rel_frob_percent = 4.5504
  np20_B02_A10_elu_b128_lr5e4   val_rel_frob_percent = 5.4198
  np20_B00_current_b128_lr1e3   val_rel_frob_percent = 6.6980
```

Decision:

- B01 A10-like architecture is the online candidate for both `n=10` and `n=20`.
- Do not run all Case 2 architectures online yet. First evaluate the two B01 winners on the three points.

Added online evaluation script:

```text
Project_YvonMaday/Results_Paper/scripts/run_mlspg_hprom_case2_B01_3pts.sh
```

Also updated `run_prom_ann_case_2.py` so it can:

- load configurable Case 2 ANN architectures from checkpoint;
- use explicit `basis.npy` and `u_ref.npy`;
- write outputs to the isolated `Results_Paper/mlspg_hprom_main` campaign;
- store Case 2 ANN ECSW weights in an isolated directory with model-specific names;
- skip saving full `rom_snaps.npy` when requested.

- Added a 1% ECSW version of the same B01 online evaluation:

```text
Project_YvonMaday/Results_Paper/scripts/run_mlspg_hprom_case2_B01_3pts_ecsw1pct.sh
```

This variant uses:

```text
--ecsw-snapshot-percent 1.0
--rebuild-ecsw
```

and writes to isolated directories:

```text
Results_Paper/mlspg_hprom_main/Runs/Case2_B01_ecsw1pct
Results_Paper/mlspg_hprom_main/Runs/ECSW_Case2_B01_ecsw1pct
Results_Paper/mlspg_hprom_main/logs/case2_B01_online_ecsw1pct
```

Reason for a separate script:

- Case 2 ECSW weights depend on the ANN map and the sampling percentage.
- Reusing the same ECSW weights directory could silently load already-computed 2% weights.
- The 1% campaign therefore keeps outputs and weights isolated.
- Correction: the 1% script must not pass `--rebuild-ecsw` for every parameter point. It now rebuilds only on the first point for each `primary` value (`n=10` and `n=20`) and loads the same model-specific ECSW weights for the remaining two points.

## 7. Petrov-Galerkin Case 2 Check

### 7.1 Motivation

After the standard Case 2 B01 online runs, we decided to test whether the enriched Petrov-Galerkin residual testing variant changes the behavior for the same learned Case 2 closures. This is a diagnostic comparison, not yet the main production result.

The standard Case 2 ECSW weights are not reused because the residual testing space is different. The standard Case 2 hyper-reduction matrix is built from the primary testing contribution, while the Petrov-Galerkin variant tests the residual with the enriched space `V_tot = [V, Vbar]`. Therefore each primary split and each model checkpoint receives its own PG ECSW weights.

### 7.2 Code Changes

Updated runner:

```text
Project_YvonMaday/run_prom_ann_case_2_petrov_galerkin.py
```

The runner now supports:

```text
--device auto
--model-path
--basis-path
--u-ref-path
--output-root
--ecsw-weights-dir
--no-save-rom-snaps
--no-plot
```

It also loads the architecture metadata stored in the checkpoint:

```text
hidden_dims
activation
dropout
```

This is required for the B01 model family (`256,512,512,256`, SiLU).

### 7.3 Execution Script

New script:

```text
Project_YvonMaday/Results_Paper/scripts/run_mlspg_hprom_case2_B01_pg_3pts.sh
```

Sherlock command:

```bash
cd /scratch/users/sadpr/Code12May/burgers2d-rom-workbench/Project_YvonMaday
bash Results_Paper/scripts/run_mlspg_hprom_case2_B01_pg_3pts.sh
```

This runs both primary splits:

```text
n_p = 10
n_p = 20
```

at the three evaluation points:

```text
(4.560, 0.0190)
(4.875, 0.0225)
(5.190, 0.0260)
```

using `2%` ECSW snapshots. The first point for each primary split builds the PG ECSW weights if missing; the remaining points load the same PG weights.

PG outputs:

```text
Results_Paper/mlspg_hprom_main/Runs/Case2_B01_PG
```

PG ECSW weights:

```text
Results_Paper/mlspg_hprom_main/Runs/ECSW_Case2_B01_PG
```

PG logs:

```text
Results_Paper/mlspg_hprom_main/logs/case2_B01_pg_online
```

## 8. Current MLSPG-HPROM Manuscript Assets

Generated manuscript assets from the downloaded current MLSPG-sensitive HPROM campaign:

```bash
cd /home/kratos/burgers2d-rom-workbench
python3 -u Project_YvonMaday/Results_Paper/generate_mlspg_hprom_current_assets.py
cd Project_YvonMaday/Results_Paper
pdflatex -interaction=nonstopmode manuscript.tex
pdflatex -interaction=nonstopmode manuscript.tex
```

Source campaign:

```text
Project_YvonMaday/Results_Paper/mlspg_hprom_main
```

Reference for coefficient diagnostics:

```text
Results_Paper/mlspg_hprom_main/Runs/Linear/linear_hprom_mu1_*_mu2_*_ntot151/qN.npy
```

This is the MLSPG-sensitive linear HPROM coefficient trajectory. For Case 2 state snapshots, comparable coefficients are recovered by least-squares projection onto the same MLSPG-sensitive basis, because the basis is not Euclidean-orthonormal.

Generated tables:

```text
Results_Paper/tables/mlspg_hprom_current_errors.tex
Results_Paper/tables/mlspg_hprom_current_hyperreduction.tex
```

Generated figures:

```text
Results_Paper/Figures/parameter_domain_sampling_points.png
Results_Paper/Figures/mlspg_hprom_current/mlspg_hprom_solution_overlays.png
Results_Paper/Figures/mlspg_hprom_current/coeff_errors/mlspg_hprom_coeff_abs_rel_all_points.png
Results_Paper/Figures/mlspg_hprom_current/coeff_errors/mlspg_hprom_coeff_abs_heatmaps.png
Results_Paper/Figures/mlspg_hprom_current/coeff_errors/mlspg_hprom_coeff_rel_heatmaps.png
```

Current populated model families:

```text
Linear HPROM
PROM-ANN Case 2, n = 10
PROM-ANN Case 2, n = 20
POD-NN-ROM A10
```

Current blank/TBD model families in the manuscript table:

```text
PROM-ANN Case 1
PROM-ANN Case 3
PROM-POD-AE
POD-DL-ROM
```

Compiled manuscript:

```text
Project_YvonMaday/Results_Paper/manuscript.pdf
```

## 9. MLSPG-Sensitive Coordinate-Recovery Notes

Important code correction kept for future PROM-only diagnostics:

```text
Project_YvonMaday/stage2_build_prom_qn_dataset.py
```

For `--backend prom`, the script previously recovered coordinates using

```text
qN = Vtot.T @ (rom_snaps - u_ref)
```

This is only valid for an Euclidean-orthonormal basis. For the MLSPG-sensitive basis, the correct recovery is least-squares:

```text
qN = solve((Vtot.T @ Vtot), Vtot.T @ (rom_snaps - u_ref))
```

This least-squares recovery was an intermediate diagnostic correction, but it
is not the final paper workflow. It was superseded on 2026-06-07 by exposing
the reduced-coordinate history carried by the PROM solver itself.

```text
coordinate_recovery: solver_coordinates
coordinate_source: solver_coordinates
```

Both PROM and HPROM Stage 2 datasets now store solver-side coordinates
directly. The linear PROM runner also stores the solver-side coordinates rather
than projecting its reconstructed states afterward.

The temporary PROM-probe scripts were deleted to avoid confusing them with the active manuscript workflow:

```text
run_mlspg_prom_probe_stage2_only.sh
run_mlspg_prom_probe_train_case2_B01_n10.sh
run_mlspg_prom_probe_linear_3pts.sh
run_mlspg_prom_probe_case2_B01_n10_3pts.sh
```

The replacement is split by thread profile to avoid BLAS oversubscription in
the ANN online solve.

First, run PROM Stage 2, train the ANN, and compute the linear PROM references
with 24 threads:

```bash
cd /scratch/users/sadpr/Code12May/burgers2d-rom-workbench/Project_YvonMaday
bash Results_Paper/scripts/run_mlspg_prom_case2_prepare_24threads.sh
```

Then run the three PROM-ANN online evaluations with one thread:

```bash
cd /scratch/users/sadpr/Code12May/burgers2d-rom-workbench/Project_YvonMaday
bash Results_Paper/scripts/run_mlspg_prom_case2_online_1thread.sh
```

The preparation script exports all BLAS/OpenMP thread variables as 24. The
online script exports them as 1. The preparation phase overwrites the isolated
`Results_Paper/mlspg_prom_probe` campaign. The online phase replaces only
`Runs/Case2_B01_PROM` and its online logs.

## 10. Case 2 Solver-Side Coordinate Output

Important diagnostic correction: Case 2 coefficient plots should use the coordinates carried by the online ROM solve, not coordinates recovered afterward from the reconstructed state.

Patched files:

```text
burgers/pod_ann_manifold.py
Project_YvonMaday/run_prom_ann_case_2.py
Project_YvonMaday/Results_Paper/generate_mlspg_hprom_current_assets.py
```

The PROM Case 2 solver can now optionally return the reduced primary coordinates:

```text
return_red_coords=True
```

The main Case 2 online runner now always saves a full coefficient matrix:

```text
case2_{prom|hprom}_ann_mu1_..._mu2_..._n{n_p}_ntot151_qN.npy
```

Its source is:

```text
qN = [q_primary_from_solver ; q_secondary_from_ANN(mu,t)]
```

The summary file records:

```text
qN_source: solver_primary_plus_ann_secondary
qN_output: ...
```

The manuscript asset generator now prefers this saved solver-side `qN.npy` for Case 2. If it is missing, it falls back to least-squares projection from `rom_snaps.npy` only with an explicit warning. Existing coefficient figures generated before rerunning Case 2 should therefore be regenerated after the online cases are rerun with the patched code.

Authoritative Case 2 rerun script for the active manuscript workflow:

```bash
cd /scratch/users/sadpr/Code12May/burgers2d-rom-workbench/Project_YvonMaday
bash Results_Paper/scripts/run_mlspg_hprom_case2_B01_3pts_save_snaps.sh
```

Deleted redundant/confusing Case 2 variants:

```text
run_mlspg_hprom_case2_B01_3pts.sh
run_mlspg_hprom_case2_B01_3pts_ecsw1pct.sh
run_mlspg_hprom_case2_B01_pg_3pts.sh
```

For POD-NN-ROM A10 online plots, keep the save-snaps script:

```bash
cd /scratch/users/sadpr/Code12May/burgers2d-rom-workbench/Project_YvonMaday
bash Results_Paper/scripts/run_mlspg_hprom_data_driven_A10_3pts_save_snaps.sh
```

Deleted redundant no-save DataDriven variant:

```text
run_mlspg_hprom_data_driven_A10_3pts.sh
```

## 11. Regenerated Manuscript Assets From Solver-Side Case 2 qN

Date: 2026-06-06.

The downloaded `mlspg_hprom_main` data now contains solver-side Case 2 coefficient histories:

```text
Results_Paper/mlspg_hprom_main/Runs/Case2_B01/*_qN.npy
```

Each Case 2 summary records:

```text
qN_source: solver_primary_plus_ann_secondary
```

The manuscript assets were regenerated with:

```bash
cd /home/kratos/burgers2d-rom-workbench/Project_YvonMaday
MPLCONFIGDIR=Results_Paper/.mplcache python3 Results_Paper/generate_mlspg_hprom_current_assets.py

cd Results_Paper
pdflatex -interaction=nonstopmode manuscript.tex
pdflatex -interaction=nonstopmode manuscript.tex
```

The generator used the saved Case 2 `qN.npy` files; no least-squares projection fallback warnings were emitted.

Current error table values:

```text
mu^(1): Linear HPROM 0.460%, Case2 n=10 1.630%, Case2 n=20 4.510%, POD-NN-ROM A10 1.927%
mu^(v): Linear HPROM 0.413%, Case2 n=10 0.646%, Case2 n=20 0.664%, POD-NN-ROM A10 0.479%
mu^(2): Linear HPROM 0.472%, Case2 n=10 1.426%, Case2 n=20 4.336%, POD-NN-ROM A10 2.396%
```

Current hyper-reduction table values:

```text
Linear HPROM: n_e=5975, mean online time 86.5897 s, speedup 1.0
Case2 n=10: n_e=901, mean online time 6.2257 s, speedup 13.9
Case2 n=20: n_e=1801, mean online time 8.7364 s, speedup 9.9
POD-NN-ROM A10: mean online time 0.0395 s, speedup 2190.1
```

Updated outputs:

```text
Results_Paper/manuscript.pdf
Results_Paper/tables/mlspg_hprom_current_errors.tex
Results_Paper/tables/mlspg_hprom_current_hyperreduction.tex
Results_Paper/Figures/mlspg_hprom_current/mlspg_hprom_solution_overlays.png
Results_Paper/Figures/mlspg_hprom_current/coeff_errors/mlspg_hprom_coeff_abs_rel_all_points.png
Results_Paper/Figures/mlspg_hprom_current/coeff_errors/mlspg_hprom_coeff_abs_heatmaps.png
Results_Paper/Figures/mlspg_hprom_current/coeff_errors/mlspg_hprom_coeff_rel_heatmaps.png
```

Figure style update: the current coefficient figures follow the clearer `250x250` report convention:

```text
PROM-ANN Case 2 n=10: tab:blue
PROM-ANN Case 2 n=20: tab:green
Data-driven POD-NN A10: tab:orange
Heatmaps: linear viridis scale, time on x-axis, coefficient index on y-axis
Coefficient curves: absolute errors on top row, relative errors on bottom row
```

## 12. Case 1 Architecture Sweep

Date: 2026-06-06.

Case 1 learns the coefficient closure

```text
q_primary in R^10 -> q_secondary in R^141.
```

The trainer now supports configurable hidden layers, activation, batch size,
learning rate, weight decay, dropout, early stopping, gradient clipping, and
`ReduceLROnPlateau`. The online Case 1 loader remains compatible with legacy
checkpoints and with the new configurable checkpoints.

The controlled four-candidate sweep is:

```bash
cd /scratch/users/sadpr/Code12May/burgers2d-rom-workbench/Project_YvonMaday
bash Results_Paper/scripts/run_mlspg_hprom_case1_arch_sweep.sh
```

The sweep deletes stale Case 1 candidate models before starting. After a
successful sweep, it retains only:

```text
Results_Paper/mlspg_hprom_main/Stage3/models/case1_ann_ntot151_best.pt
Results_Paper/mlspg_hprom_main/Stage3/case1_ann_ntot151_best_summary.txt
Results_Paper/mlspg_hprom_main/logs/case1_arch_sweep/case1_ann_ntot151_best.log
Results_Paper/mlspg_hprom_main/logs/case1_arch_sweep/case1_arch_sweep_summary.csv
```

If the sweep fails, incomplete candidate checkpoints and summaries are removed.

The Stage 3 metadata loader was also made compatible with `meta.npy` files
written by NumPy 2 and read under NumPy 1.x. Future Stage 2 builds additionally
write portable `meta.json` metadata to avoid pickle-version coupling.

After the sweep selected `C02_wide_silu` with
`val_rel_frob_percent=1.3708197512`, the canonical Case 1 checkpoint is evaluated
at the two test points and the verification point with:

```bash
cd /scratch/users/sadpr/Code12May/burgers2d-rom-workbench/Project_YvonMaday
bash Results_Paper/scripts/run_mlspg_hprom_case1_best_3pts_save_snaps.sh
```

This online script uses the MLSPG-sensitive basis, constructs a model-specific
2% ECSW rule only once, and reuses it for the remaining points. It stores the
reconstructed states and the full manifold coordinates
`[q_primary; ANN(q_primary)]` in:

```text
Results_Paper/mlspg_hprom_main/Runs/Case1_Best
```

## 13. Case 3 Architecture Sweep

Date: 2026-06-07.

Case 3 learns

```text
[q_primary in R^10, mu1, mu2, t] -> q_secondary in R^141.
```

The Case 3 trainer and online loader support the same configurable MLP family
used for Case 1, while retaining compatibility with legacy checkpoints. The
controlled four-candidate sweep is:

```bash
cd /scratch/users/sadpr/Code12May/burgers2d-rom-workbench/Project_YvonMaday
bash Results_Paper/scripts/run_mlspg_hprom_case3_arch_sweep.sh
```

It uses the HPROM Stage 2 dataset, a fixed row-wise 90/10 split with seed 42,
train-only z-score scaling, AdamW, `ReduceLROnPlateau`, and early stopping.
After completion it retains only:

```text
Results_Paper/mlspg_hprom_main/Stage3/models/case3_ann_ntot151_best.pt
Results_Paper/mlspg_hprom_main/Stage3/case3_ann_ntot151_best_summary.txt
Results_Paper/mlspg_hprom_main/logs/case3_arch_sweep/case3_ann_ntot151_best.log
Results_Paper/mlspg_hprom_main/logs/case3_arch_sweep/case3_arch_sweep_summary.csv
```

The sweep selected `C02_wide_silu`:

```text
val_rel_frob_percent: 0.5368914921
best_val_mse: 0.0015586764
```

The canonical checkpoint is evaluated at the two test points and the
verification point with a model-specific 2% ECSW rule:

```bash
cd /scratch/users/sadpr/Code12May/burgers2d-rom-workbench/Project_YvonMaday
bash Results_Paper/scripts/run_mlspg_hprom_case3_best_3pts_save_snaps.sh
```

The ECSW rule is constructed once in a dedicated `--ecsw-only` phase using
24 BLAS threads. The three online solves then reuse it with one BLAS thread,
so their reported online times are not polluted by ECSW construction or
thread oversubscription. The script stores both reconstructed states and the complete coordinates
`[q_primary; H(q_primary, mu, t)]` under:

```text
Results_Paper/mlspg_hprom_main/Runs/Case3_Best
```

## 14. Direct low/high LSPG transfer diagnostic

Date: 2026-06-07.

The oracle perturbation study measures empirical contamination, but it does
not store the local transfer operator itself. The direct diagnostic now
evaluates, for the split `n_primary=10`, `n_secondary=141`,

```text
T_LH = -(V^T J^T P J V)^dagger V^T J^T P J Vbar
```

at the same HDM states for the Euclidean and LSPG-sensitive bases. The default
residual metric is `P=I`, matching the metric construction recorded in
`stage1_lspg_sensitive_summary.txt`.

Two gains are reported:

```text
coordinate gain: sigma_max(T_LH)
physical gain:   sup_z ||V T_LH z||_M / ||Vbar z||_M
```

The physical mass-norm gain is the primary cross-basis comparison because it
is invariant to coordinate rescaling. The coordinate gain is retained as a
diagnostic, not as the sole basis-selection criterion.

Sherlock command:

```bash
cd /scratch/users/sadpr/Code12May/burgers2d-rom-workbench/Project_YvonMaday
bash Results_Paper/scripts/run_mlspg_low_high_transfer_diagnostic.sh
```

Outputs are written to:

```text
Results_Paper/MetricStudy/low_high_transfer/
```

including per-sample and aggregate CSV files, a direct Euclidean/LSPG
comparison CSV, a LaTeX table, time-history/distribution figures, and the
worst-case `10 x 141` transfer matrices.

## 15. Canonical production layout and ECSW 1% campaign

Date: 2026-06-07.

The architecture-sweep identifiers (`A10`, `B01`, and `C02`) are useful for
provenance, but they are not sufficiently descriptive for production paths.
The campaign now uses semantic `Best` names:

```text
Stage3/models/case1_ann_ntot151_best.pt
Stage3/models/case2_ann_ntot151_np10_best.pt
Stage3/models/case2_ann_ntot151_np20_best.pt
Stage3/models/case3_ann_ntot151_best.pt
Stage3/models/data_driven_ann_ntot151_best.pt
```

The original sweep labels and complete candidate artifacts are retained under
`Stage3/sweeps/` and recorded in `Stage3/production_models.txt`. Architecture,
optimizer, validation, and parameter-count details remain in the corresponding
`*_best_summary.txt` files.

Existing 2% results were reorganized without deleting data:

```text
Runs/ECSW2pct/Case1_Best
Runs/ECSW2pct/Case2_Best/np10
Runs/ECSW2pct/Case2_Best/np20
Runs/ECSW2pct/Case3_Best
Runs/DataDriven_Best

ECSW/2pct/Case1_Best
ECSW/2pct/Case2_Best/np10
ECSW/2pct/Case2_Best/np20
ECSW/2pct/Case3_Best
```

The layout migration is idempotent:

```bash
python3 Results_Paper/scripts/normalize_mlspg_hprom_main_layout.py
```

Case 1 and Case 2 now support `--ecsw-only`, matching Case 3. This separates
ECSW construction from online execution. Each model-specific ECSW rule is
built once with 24 BLAS threads and then reused for all three evaluation points,
whose online solves use one BLAS thread.

The single production launcher accepts either `1.0` or `2.0`:

```bash
cd /scratch/users/sadpr/Code12May/burgers2d-rom-workbench/Project_YvonMaday
bash Results_Paper/scripts/run_mlspg_hprom_all_best_3pts.sh 1.0
```

The 1% campaign writes:

```text
Runs/ECSW1pct/Case1_Best
Runs/ECSW1pct/Case2_Best/np10
Runs/ECSW1pct/Case2_Best/np20
Runs/ECSW1pct/Case3_Best

ECSW/1pct/Case1_Best
ECSW/1pct/Case2_Best/np10
ECSW/1pct/Case2_Best/np20
ECSW/1pct/Case3_Best
```

The launcher saves solver-produced full `qN`, reconstructed snapshots, plots,
per-point summaries, model-specific ECSW weights, and an aggregate campaign
summary. It skips complete points and reuses existing rules, so it is safe to
restart.

After the corrected download, the campaign contains the Case 3 winner as well.
The normalized production layout contains the five selected checkpoints, four
2% ECSW rules, and the complete 2% online outputs for Case 1, Case 2
(`n_p=10` and `n_p=20`), Case 3, and the data-driven model.

## 16. Manuscript placement of basis and ECSW diagnostics

Date: 2026-06-07.

The main MLSPG-sensitive HPROM results section now reports the 1% ECSW campaign
as the production intrusive campaign. The main hyperreduction table intentionally
omits the confusing `ECSW target` column and reports only the quantities that
matter for performance: retained online elements `N_e`, mean online time, and
mean speedup.

The text explains that the 1% ECSW rule is a nominal parameter-time stratified
sampling fraction used to build the residual/tangent training set for ECSW. It
is not a physical parameter and it is not the final online mesh size. The final
online mesh size is the number of nonzero elements after the ECSW nonnegative
least-squares solve.

The direct 1% vs 2% ECSW comparison was moved to an appendix. This comparison is
treated as a sampling-fraction sensitivity check, not as a monotone mesh-refinement
study: changing the sampled residual/tangent snapshots changes the ECSW NNLS
problem and therefore the selected elements and nonlinear residual.

The Burgers basis-selection numerical evidence was also moved to an appendix,
because those numerical diagnostics should appear after the Burgers benchmark is
introduced. The appendix now contains both:

```text
MetricStudy/low_high_transfer/low_high_transfer_summary.tex
```

and the oracle high-mode perturbation table. The low/high transfer diagnostic is
the primary basis-selection evidence: the LSPG-sensitive basis reduces the mean
`sigmamax(T_LH)` by about 50% relative to the Euclidean POD basis over the three
evaluation trajectories.

## 2026-06-07: HDM-Based Speedup Reference and POD-AE Sweep Preparation

The speed-up reference in the manuscript was corrected to use the HDM runtime, not the linear HPROM runtime.  The HDM timing source is `Results/fom_training_summary.txt` at the workbench root, with

```text
mean_time_per_parameter_seconds: 7.37437560e+02
```

Thus the manuscript speed-up definition is now

```text
S^(m) = t_HDM / t_m,  t_HDM = 737.44 s.
```

Regenerated assets:

```text
Results_Paper/tables/mlspg_hprom_current_hyperreduction.csv
Results_Paper/tables/mlspg_hprom_current_hyperreduction.tex
Results_Paper/tables/mlspg_hprom_ecsw_1pct_vs_2pct.csv
Results_Paper/tables/mlspg_hprom_ecsw_1pct_vs_2pct.tex
Results_Paper/manuscript.pdf
```

The current main hyperreduction table now reports speed-up versus HDM.  Example values from the regenerated table:

```text
Linear HPROM: speedup 8.5x
PROM-ANN Case 1: speedup 38.9x
PROM-ANN Case 2 (n=10): speedup 142.4x
PROM-ANN Case 2 (n=20): speedup 109.2x
PROM-ANN Case 3: speedup 42.6x
POD-NN-ROM: speedup 18651.6x
```

A paper-layout POD-AE architecture sweep was prepared:

```text
Results_Paper/scripts/run_mlspg_hprom_pod_ae_arch_sweep.sh
```

The POD-AE trainer now accepts explicit dataset/output paths, writes candidate summaries with `val_rel_frob_percent`, uses ReduceLROnPlateau, stores trainable parameter counts, and records the MLSPG-sensitive basis paths from the Stage-2 dataset metadata.  The sweep keeps only the selected checkpoint:

```text
Results_Paper/mlspg_hprom_main/Stage3/models/prom_pod_ae_ntot151_best.pt
Results_Paper/mlspg_hprom_main/Stage3/prom_pod_ae_ntot151_best_summary.txt
```

The intended Sherlock command is:

```bash
cd /scratch/users/sadpr/Code12May/burgers2d-rom-workbench/Project_YvonMaday
bash Results_Paper/scripts/run_mlspg_hprom_pod_ae_arch_sweep.sh
```

## 2026-06-07: POD-DL-ROM Sweep Preparation

A paper-layout POD-DL-ROM architecture sweep was prepared while the PROM-POD-AE sweep was running.  The implementation used here is the non-intrusive latent coefficient-space variant:

```text
(mu1, mu2, t) -> z -> q_N
```

with an encoder `E(q_N)` and decoder `D(z)` used during training.  This follows the POD-DL-ROM principle of Fresca--Manzoni after the prior POD reduction, but the current implementation uses MLP encoder/decoder/dynamics networks in coefficient space rather than a convolutional autoencoder on full fields.  This is intentional for the current campaign because the object learned by Stage 3 is `q_N in R^151`; imposing convolutions on the one-dimensional coefficient vector would introduce an artificial topology.

Modified files:

```text
stage3_perform_training_pod_dl_data_driven.py
run_pod_dl_data_driven.py
```

The trainer now accepts explicit paper-layout paths, writes `val_rel_frob_percent`, uses ReduceLROnPlateau, records trainable parameter counts, and stores MLSPG-sensitive basis/u_ref paths from the Stage-2 dataset metadata.  The runner now prefers the checkpoint basis/u_ref paths instead of blindly using `Results/Stage1`.

New sweep script:

```text
Results_Paper/scripts/run_mlspg_hprom_pod_dl_arch_sweep.sh
```

The sweep explores latent dimensions, encoder/decoder/dynamics widths, activation, and loss weights.  The winner is selected by validation relative Frobenius error in coefficient space and is stored as:

```text
Results_Paper/mlspg_hprom_main/Stage3/models/pod_dl_data_driven_ntot151_best.pt
Results_Paper/mlspg_hprom_main/Stage3/pod_dl_data_driven_ntot151_best_summary.txt
```

Sherlock command:

```bash
cd /scratch/users/sadpr/Code12May/burgers2d-rom-workbench/Project_YvonMaday
bash Results_Paper/scripts/run_mlspg_hprom_pod_dl_arch_sweep.sh
```

## 2026-06-07: POD-AE/POD-DL Sweep Bugfix

The first Sherlock launch of both latent sweeps failed with:

```text
NameError: name 'SEED' is not defined
```

Cause: the trainers were changed to use runtime `--seed`, but two `train_test_split(... random_state=SEED)` calls remained.

Fixed files:

```text
stage3_perform_training_prom_pod_ae.py
stage3_perform_training_pod_dl_data_driven.py
```

Both now use `random_state=seed`.  They also localize metadata paths copied from another machine: if a Stage-2 `meta.npy` contains `/home/kratos/.../Project_YvonMaday/...` but the current checkout is on Sherlock, the trainer rewrites the stored checkpoint basis/u_ref paths to the local `Project_YvonMaday` path when possible.

Validation run locally:

```bash
python3 -m py_compile stage3_perform_training_prom_pod_ae.py stage3_perform_training_pod_dl_data_driven.py
bash -n Results_Paper/scripts/run_mlspg_hprom_pod_ae_arch_sweep.sh
bash -n Results_Paper/scripts/run_mlspg_hprom_pod_dl_arch_sweep.sh
```

## 2026-06-08: POD-AE/POD-DL Best Online Scripts

The latent sweeps finished on Sherlock.

POD-AE winner:

```text
PAE04_l8_silu_zscore_wide
val_rel_frob_percent = 0.126632081810385
model   = Results_Paper/mlspg_hprom_main/Stage3/models/prom_pod_ae_ntot151_best.pt
summary = Results_Paper/mlspg_hprom_main/Stage3/prom_pod_ae_ntot151_best_summary.txt
```

POD-DL-ROM winner:

```text
PDL05_l20_silu_wide
val_rel_frob_percent = 0.6360284052789211
model   = Results_Paper/mlspg_hprom_main/Stage3/models/pod_dl_data_driven_ntot151_best.pt
summary = Results_Paper/mlspg_hprom_main/Stage3/pod_dl_data_driven_ntot151_best_summary.txt
```

Online scripts were added for the three evaluation parameters:

```text
Results_Paper/scripts/run_mlspg_hprom_pod_ae_best_3pts.sh
Results_Paper/scripts/run_mlspg_hprom_pod_dl_best_3pts.sh
Results_Paper/scripts/run_mlspg_hprom_latent_best_3pts.sh
```

Threading policy:

```text
POD-AE ECSW build: 24 threads, done once.
POD-AE online solves: 1 thread.
POD-DL inference: 1 thread.
```

The one-thread online environment is:

```bash
export BLIS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

POD-AE runner changes:

```text
run_prom_pod_ae.py
```

now supports explicit paper-layout output paths and ECSW-weight paths:

```text
--output-root
--ecsw-weights-dir
--ecsw-only
```

and now saves the online latent trajectory, reconstructed state snapshots, and decoded coefficient trajectory directly from the online solution:

```text
*_latent.npy
*_snaps.npy
*_qN.npy
```

This avoids using least-squares projection to recover coordinates after the solve.

POD-DL runner changes:

```text
run_pod_dl_data_driven.py
```

now supports:

```text
--output-root
```

and writes the usual direct non-intrusive outputs under the paper run directory:

```text
qN.npy
rom_snaps.npy
pod_dl_data_driven_summary.txt
```

Validation performed locally:

```bash
python3 -m py_compile run_prom_pod_ae.py run_pod_dl_data_driven.py
bash -n Results_Paper/scripts/run_mlspg_hprom_pod_ae_best_3pts.sh
bash -n Results_Paper/scripts/run_mlspg_hprom_pod_dl_best_3pts.sh
bash -n Results_Paper/scripts/run_mlspg_hprom_latent_best_3pts.sh
```

Sherlock commands:

```bash
cd /scratch/users/sadpr/Code12May/burgers2d-rom-workbench/Project_YvonMaday
bash Results_Paper/scripts/run_mlspg_hprom_pod_ae_best_3pts.sh
bash Results_Paper/scripts/run_mlspg_hprom_pod_dl_best_3pts.sh
```

or combined:

```bash
cd /scratch/users/sadpr/Code12May/burgers2d-rom-workbench/Project_YvonMaday
bash Results_Paper/scripts/run_mlspg_hprom_latent_best_3pts.sh
```

## 2026-06-08: POD-AE Online Bugfix After Large Errors

The first PROM-POD-AE HPROM online run produced very large errors:

```text
mu=(4.560,0.0190): 22.71%
mu=(4.875,0.0225): 29.78%
mu=(5.190,0.0260): 34.68%
```

This should not be treated as a final POD-AE result yet.  A real implementation issue was found in the latent intrusive manifold:

```text
burgers/pod_dl_manifold.py
```

The POD-AE online/ECSW routines were converting states to coefficient coordinates with

```text
q = V.T @ (u - u_ref)
```

This is invalid for the MLSPG-sensitive basis, which is not assumed Euclidean-orthonormal.  The coordinate recovery was changed to least-squares projection:

```text
q = argmin_y || V y - (u - u_ref) ||_2
```

For ECSW training, the projection is vectorized across all selected snapshots to avoid solving one large least-squares problem per snapshot.

The POD-AE online script now supports:

```bash
FORCE=1 bash Results_Paper/scripts/run_mlspg_hprom_pod_ae_best_3pts.sh
```

`FORCE=1` removes previous POD-AE ECSW1pct outputs and POD-AE ECSW weights before rebuilding, which is necessary because the old ECSW weights were built with the incorrect coordinate map.

Validation performed locally:

```bash
python3 -m py_compile burgers/pod_dl_manifold.py Project_YvonMaday/run_prom_pod_ae.py Project_YvonMaday/run_pod_dl_data_driven.py
bash -n Project_YvonMaday/Results_Paper/scripts/run_mlspg_hprom_pod_ae_best_3pts.sh
bash -n Project_YvonMaday/Results_Paper/scripts/run_mlspg_hprom_latent_best_3pts.sh
```


## 2026-06-08: Latent Sweeps Fixed to Dimension 10

The separate `L10` training/online scripts were removed to avoid duplicating workflows and output conventions.  The original latent sweep scripts are now the only entry points again, but all candidates use fixed latent dimension 10.

Modified sweep scripts:

```text
Results_Paper/scripts/run_mlspg_hprom_pod_ae_arch_sweep.sh
Results_Paper/scripts/run_mlspg_hprom_pod_dl_arch_sweep.sh
```

POD-AE sweep convention:

```text
All candidates: latent_dim=10
Architecture/scaling/activation still varied.
Winner retained as: Stage3/models/prom_pod_ae_ntot151_best.pt
```

POD-DL sweep convention:

```text
All candidates: latent_dim=10
Architecture/loss weights/activation still varied.
Winner retained as: Stage3/models/pod_dl_data_driven_ntot151_best.pt
```

The previous standalone L10 scripts were deleted:

```text
Results_Paper/scripts/run_mlspg_hprom_pod_ae_l10_train.sh
Results_Paper/scripts/run_mlspg_hprom_pod_dl_l10_train.sh
Results_Paper/scripts/run_mlspg_hprom_latent_l10_train.sh
Results_Paper/scripts/run_mlspg_hprom_pod_ae_l10_3pts.sh
Results_Paper/scripts/run_mlspg_hprom_pod_dl_l10_3pts.sh
Results_Paper/scripts/run_mlspg_hprom_latent_l10_3pts.sh
```

The online best scripts remain unchanged as the deployment entry points, except that POD-DL Best now supports `FORCE=1` to remove old output folders before rerunning.  This matters because old POD-DL outputs may have `nz20` directory names, while the new fixed-latent sweep will produce `nz10`.

Validation performed locally:

```bash
bash -n Results_Paper/scripts/run_mlspg_hprom_pod_ae_arch_sweep.sh
bash -n Results_Paper/scripts/run_mlspg_hprom_pod_dl_arch_sweep.sh
bash -n Results_Paper/scripts/run_mlspg_hprom_pod_ae_best_3pts.sh
bash -n Results_Paper/scripts/run_mlspg_hprom_pod_dl_best_3pts.sh
bash -n Results_Paper/scripts/run_mlspg_hprom_latent_best_3pts.sh
python3 -m py_compile stage3_perform_training_prom_pod_ae.py stage3_perform_training_pod_dl_data_driven.py run_prom_pod_ae.py run_pod_dl_data_driven.py ../burgers/pod_dl_manifold.py
```

## 2026-06-09: Complete Non-Enriched Campaign Added to the Manuscript

The downloaded `mlspg_hprom_main` directory now contains direct saved
coefficient trajectories and state snapshots for all methods used in the
non-enriched comparison:

```text
Linear HPROM
PROM-ANN Case 1
PROM-ANN Case 2 (n=10)
PROM-ANN Case 2 (n=20)
PROM-ANN Case 3
PROM-POD-AE (latent dimension 10)
POD-NN-ROM
POD-DL-ROM (latent dimension 10)
```

All coefficient arrays have shape `(151, 501)` and all state arrays have
shape `(125000, 501)`. The paper assets use these direct saved trajectories;
no least-squares reconstruction or synthetic replacement is applied.

The selected PROM-POD-AE has latent dimension 10, hidden dimensions
`(512,256,128)`, GELU activation, z-score scaling, and 486817 trainable
parameters. The selected POD-DL-ROM has latent dimension 10, encoder
`(512,256)`, decoder `(256,512)`, dynamics network
`(256,512,512,256)`, SiLU activation, z-score scaling, and 952747 trainable
parameters.

The final relative trajectory errors with respect to the HDM are shown with
the verification point first, followed by the two off-grid test points:

| Method | verification mu^(v)=(4.875,0.0225) | off-grid mu^(1)=(4.560,0.0190) | off-grid mu^(2)=(5.190,0.0260) |
|---|---:|---:|---:|
| Linear HPROM | 0.413% | 0.460% | 0.472% |
| PROM-ANN Case 1 | 0.688% | 0.922% | 0.699% |
| PROM-ANN Case 2 (n=10) | 0.636% | 1.616% | 1.422% |
| PROM-ANN Case 2 (n=20) | 0.958% | 4.225% | 5.655% |
| PROM-ANN Case 3 | 0.450% | 0.684% | 0.699% |
| PROM-POD-AE | 0.442% | 0.567% | 0.750% |
| POD-NN-ROM | 0.479% | 1.927% | 2.396% |
| POD-DL-ROM | 0.439% | 1.504% | 1.540% |

The generated paper assets are:

```text
Results_Paper/Figures/mlspg_hprom_current/mlspg_hprom_solution_overlays.png
Results_Paper/Figures/mlspg_hprom_current/coeff_errors/mlspg_hprom_coeff_abs_rel_all_points.png
Results_Paper/Figures/mlspg_hprom_current/coeff_errors/mlspg_hprom_coeff_abs_heatmaps.png
Results_Paper/Figures/mlspg_hprom_current/coeff_errors/mlspg_hprom_coeff_rel_heatmaps.png
Results_Paper/tables/mlspg_hprom_current_errors.tex
Results_Paper/tables/mlspg_hprom_current_hyperreduction.tex
Results_Paper/tables/mlspg_hprom_ecsw_1pct_vs_2pct.tex
```

Regenerate and compile with:

```bash
cd /home/kratos/burgers2d-rom-workbench/Project_YvonMaday/Results_Paper
MPLCONFIGDIR=/tmp/mplconfig python3 -u generate_mlspg_hprom_current_assets.py
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex
```

The resulting manuscript is:

```text
Results_Paper/manuscript.pdf
```

## 2026-06-09: MLSPG-HPROM Enrichment Campaign

The enrichment campaign is isolated from the completed non-enriched results:

```text
Baseline root:   Results_Paper/mlspg_hprom_main
Enrichment root: Results_Paper/mlspg_hprom_enrichment
```

The controlled comparison changes only the amount of Stage-2 training data.
The following baseline artifacts remain fixed:

```text
Results_Paper/MetricStudy/lspg_sensitive/Stage1/basis.npy
Results_Paper/MetricStudy/lspg_sensitive/Stage1/u_ref.npy
Results_Paper/mlspg_hprom_main/Stage2/ecsw/ecsw_weights_lspg_ntot151.npy
```

The 20 new Latin-hypercube trajectories reuse the exact existing linear HPROM
ECSW file. No new linear ECSW rule is computed, and the weights are not copied
into the enrichment directory. The enrichment metadata stores the original
path and its SHA-256 checksum. The local checksum of the current baseline file
is:

```text
769c56aeb4f37de1c20d869d49eecd923564ebbf5de8710978daabdd9a4b94f1
```

The enriched Stage-2 dataset contains:

```text
9 copied baseline direct solver-side qN trajectories
20 new direct solver-side linear-HPROM qN trajectories
29 trajectories total
14529 coefficient/time samples total (29 x 501)
```

The deterministic LHS uses seed 42 and excludes the evaluation coordinates:

```text
verification: (4.875, 0.0225)
off-grid 1:   (4.560, 0.0190)
off-grid 2:   (5.190, 0.0260)
```

The Stage-2 entry point is:

```bash
cd /scratch/users/sadpr/Code12May/burgers2d-rom-workbench/Project_YvonMaday
bash Results_Paper/scripts/run_mlspg_hprom_enrichment_stage2_only.sh
```

The script is resumable. Completed LHS trajectories are validated and reused.
After all solves, it requires exactly 29 finite arrays of shape `(151, 501)`
and verifies that the ECSW checksum still matches the baseline linear rule.

Because the additional 20 trajectories can change the preferred capacity and
regularization, the enrichment campaign repeats the controlled architecture
sweeps used for the baseline campaign. The master entry point is:

```text
Results_Paper/scripts/run_mlspg_hprom_enrichment_arch_sweeps.sh
```

It validates the complete Stage-2 dataset before starting any training:

```text
29 trajectories total
9 baseline trajectories
20 LHS trajectories
all qN arrays finite and of shape (151, 501)
direct solver-side qN only
fixed baseline linear ECSW path and SHA-256 unchanged
no ECSW copy and no ECSW rebuild
```

The production sweep contains 46 candidates:

```text
Case 1:                  4
Case 2, n=10:            5
Case 2, n=20:            5
Case 3:                  4
POD-NN-ROM:             12
PROM-POD-AE, n_z=10:     8
POD-DL-ROM, n_z=10:      8
Total:                   46
```

The candidate definitions reproduce the corresponding non-enriched sweep
templates. The latent dimension is fixed to 10 for both PROM-POD-AE and
POD-DL-ROM. Selection uses `val_rel_frob_percent`; the best validation loss is
used only as a fallback if that metric is absent.

Train all families sequentially with one command:

```bash
cd /scratch/users/sadpr/Code12May/burgers2d-rom-workbench/Project_YvonMaday
SWEEP_EXECUTION=sequential TRAIN_NUM_THREADS=1 \
  bash Results_Paper/scripts/run_mlspg_hprom_enrichment_arch_sweeps.sh all
```

On a CPU allocation with enough memory, launch the seven family sweeps
concurrently while keeping each training process single-threaded:

```bash
cd /scratch/users/sadpr/Code12May/burgers2d-rom-workbench/Project_YvonMaday
SWEEP_EXECUTION=parallel TRAIN_NUM_THREADS=1 \
  bash Results_Paper/scripts/run_mlspg_hprom_enrichment_arch_sweeps.sh all
```

Do not use the parallel mode when all families would compete for one GPU.
Individual selectors are:

```text
case1
case2
case2_np10
case2_np20
case3
data_driven
pod_ae
pod_dl
```

After each family completes, the script writes a ranking CSV, promotes the
winner to a canonical `*_best.pt` checkpoint, writes a canonical summary and
winner log, and removes all non-winning checkpoints, candidate summaries, and
candidate logs. The only final checkpoints are:

```text
Results_Paper/mlspg_hprom_enrichment/Stage3/models

case1_ann_ntot151_best.pt
case2_ann_ntot151_np10_best.pt
case2_ann_ntot151_np20_best.pt
case3_ann_ntot151_best.pt
data_driven_ann_ntot151_best.pt
prom_pod_ae_ntot151_best.pt
pod_dl_data_driven_ntot151_best.pt
```

The strict linear-ECSW reuse applies to Stage-2 target generation. Later online
intrusive models still require their own case-specific 1% ECSW rules because
Case 1, Case 2, Case 3, and PROM-POD-AE have different nonlinear trial
manifolds and tangents. Those are not replacements for, or rebuilds of, the
fixed linear HPROM ECSW rule.

Local validation was performed with all 29 downloaded trajectories and
`SWEEP_SMOKE_TEST=1`. Every family loaded 14529 samples, initialized, completed
one CPU epoch, saved and selected a checkpoint, and finished with exactly the
seven canonical checkpoints above and no candidate checkpoint leftovers.

## 2026-06-11: Serial Online Deployment for the Enriched Winners

After the seven enrichment architecture sweeps finish, deploy all selected
models with:

```text
Results_Paper/scripts/run_mlspg_hprom_enrichment_all_best_3pts_serial.sh
```

The script is deliberately serial. It completes one family before starting the
next, and within each family it evaluates one parameter point at a time. The
fixed order is:

```text
1. verification mu^(v) = (4.875, 0.0225)
2. off-grid    mu^(1) = (4.560, 0.0190)
3. off-grid    mu^(2) = (5.190, 0.0260)
```

The family order is:

```text
1. PROM-ANN Case 1
2. PROM-ANN Case 2, n=10
3. PROM-ANN Case 2, n=20
4. PROM-ANN Case 3
5. PROM-POD-AE
6. POD-NN-ROM
7. POD-DL-ROM
```

Run on Sherlock with:

```bash
cd /scratch/users/sadpr/Code12May/burgers2d-rom-workbench/Project_YvonMaday

export BLIS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

bash Results_Paper/scripts/run_mlspg_hprom_enrichment_all_best_3pts_serial.sh
```

The script itself sets 24 BLAS threads only while constructing a missing
case-specific ECSW rule, then restores one thread for every online trajectory.
Override this with `ECSW_BUILD_THREADS` or `ONLINE_THREADS` if required.

Saving follows the non-enriched template:

```text
Results_Paper/mlspg_hprom_enrichment/Runs/ECSW1pct/Case1_Best
Results_Paper/mlspg_hprom_enrichment/Runs/ECSW1pct/Case2_Best/np10
Results_Paper/mlspg_hprom_enrichment/Runs/ECSW1pct/Case2_Best/np20
Results_Paper/mlspg_hprom_enrichment/Runs/ECSW1pct/Case3_Best
Results_Paper/mlspg_hprom_enrichment/Runs/ECSW1pct/PODAE_Best
Results_Paper/mlspg_hprom_enrichment/Runs/DataDriven_Best
Results_Paper/mlspg_hprom_enrichment/Runs/PODDL_Best
```

Intrusive outputs contain solver-side `qN`, state snapshots, plots, and
summaries. The non-intrusive outputs contain predicted `qN`, reconstructed state
snapshots, plots, and summaries. Logs are stored under:

```text
Results_Paper/mlspg_hprom_enrichment/logs/online
```

Each nonlinear intrusive winner receives its own case-specific 1% ECSW rule,
built once from the same nine structured ECSW-training trajectories used in the
non-enriched campaign and reused at all three evaluation points. This preserves
the controlled comparison: only the neural-network training dataset changes.
The original linear Stage-2 ECSW rule remains fixed, referenced from
`mlspg_hprom_main`, and is neither copied nor rebuilt.

The script is resumable: a point is skipped only when its summary, `qN`, and
state-snapshot outputs all exist. `FORCE=1` explicitly removes the previous
enriched online outputs and nonlinear case-specific ECSW rules before rerunning.
`PLAN_ONLY=1` validates paths and prints the execution plan without running any
solver.
