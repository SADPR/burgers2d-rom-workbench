# Maday Experiment Runbook (Isolated from Baseline)

This runbook keeps your existing `Results/Stage*` baseline untouched by writing experiment artifacts under:

`Project_YvonMaday/Results_Maday/<tag>/...`

## 0) Choose a tag

Use one short tag per experiment batch:

```bash
export MADAY_TAG=maday_p2_try01
cd /scratch/users/sadpr/Code12May/burgers2d-rom-workbench/Project_YvonMaday
```

## 1) Build weighted Stage-1 basis (Maday branch)

Cell-area diagonal metric (safe starting point):

```bash
python3 -u stage1_lspg_weighted_pod.py \
  --maday-tag "$MADAY_TAG" \
  --weighting cell_area \
  --pod-tol 1e-6 \
  2>&1 | tee "Results_Maday/${MADAY_TAG}/stage1_weighted.log"
```

Outputs:

- `Results_Maday/<tag>/Stage1/basis_weighted.npy`
- `Results_Maday/<tag>/Stage1/weights_diag.npy`
- `Results_Maday/<tag>/Stage1/stage1_lspg_weighted_pod_summary.txt`

## 2) Apply Maday basis correction (Proposal 2 or 3)

Proposal 2 (modify high modes only):

```bash
python3 -u stage1_lspg_basis_correction.py \
  --maday-tag "$MADAY_TAG" \
  --basis-file basis_weighted.npy \
  --primary-modes 10 \
  --proposal high \
  --metric-source diag_file \
  --metric-file "Results_Maday/${MADAY_TAG}/Stage1/weights_diag.npy" \
  2>&1 | tee "Results_Maday/${MADAY_TAG}/stage1_correction_p2.log"
```

Proposal 3 (modify low modes only):

```bash
python3 -u stage1_lspg_basis_correction.py \
  --maday-tag "$MADAY_TAG" \
  --basis-file basis_weighted.npy \
  --primary-modes 10 \
  --proposal low \
  --metric-source diag_file \
  --metric-file "Results_Maday/${MADAY_TAG}/Stage1/weights_diag.npy" \
  2>&1 | tee "Results_Maday/${MADAY_TAG}/stage1_correction_p3.log"
```

## 3) Clone Stage-2 dataset into Maday branch

Clone existing PROM dataset (no rebuild, no baseline edits):

```bash
python3 -u stage2_clone_dataset_to_maday.py \
  --maday-tag "$MADAY_TAG" \
  --dataset-dir "$(pwd)/Results/Stage2/prom_coeff_dataset_ntot151" \
  2>&1 | tee "Results_Maday/${MADAY_TAG}/stage2_clone.log"
```

Cloned dataset path:

- `Results_Maday/<tag>/Stage2/prom_coeff_dataset_ntot151`

## 4) Train isolated Stage-3 models (writes to Results_Maday)

ANN n=20 example:

```bash
python3 -u stage3_perform_training_case_2_ann_test_n20_maday.py \
  --maday-tag "$MADAY_TAG" \
  --dataset-backend prom \
  --dataset-ntot 151 \
  --dataset-dir "$(pwd)/Results_Maday/${MADAY_TAG}/Stage2/prom_coeff_dataset_ntot151" \
  --primary-modes 20 \
  --model-name case2_model_n20_maday_c1_s23.pt \
  --hidden-dims 32,64,128,256,256 \
  --activation elu \
  --lr 0.001 \
  --weight-decay 1e-6 \
  --dropout 0.0 \
  --batch-size 128 \
  --seed 23 \
  2>&1 | tee "Results_Maday/${MADAY_TAG}/train_case2_ann_n20.log"
```

GPR example:

```bash
python3 -u stage3_perform_training_case_2_gpr_maday.py \
  --maday-tag "$MADAY_TAG" \
  --dataset-backend prom \
  --dataset-ntot 151 \
  --dataset-dir "$(pwd)/Results_Maday/${MADAY_TAG}/Stage2/prom_coeff_dataset_ntot151" \
  --primary-modes 10 \
  --model-name case2_model_n10_maday_gpr.pt \
  --kernel-name matern15 \
  --val-split mu_group \
  --ard \
  --max-train-samples 4509 \
  2>&1 | tee "Results_Maday/${MADAY_TAG}/train_case2_gpr_n10.log"
```

## 5) Run isolated ANN sweep (optional)

```bash
python3 -u run_case2_n20_sweep_maday.py \
  --maday-tag "$MADAY_TAG" \
  --dataset-backend prom \
  --dataset-ntot 151 \
  --dataset-dir "$(pwd)/Results_Maday/${MADAY_TAG}/Stage2/prom_coeff_dataset_ntot151" \
  --unbuffered \
  2>&1 | tee "Results_Maday/${MADAY_TAG}/run_case2_n20_sweep_maday.log"
```

Ranking output:

- `Results_Maday/<tag>/Figures/offline_case2/case2_n20_sweep_ranking.txt`

## Notes

- These scripts are designed to avoid writing into baseline `Results/Stage*` for Stage-1/Stage-2 clones and Stage-3 wrappers.
- `check_case2_offline_errors.py` still reads linear references from baseline `Results/Runs/Linear` when `--reference-source linear_runs` is used.
- Start with Proposal 2 (`--proposal high`) before Proposal 1+3 combinations for a cleaner ablation.

