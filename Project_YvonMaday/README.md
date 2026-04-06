# Project_YvonMaday Workflow

This project is organized in two completely separated pipelines:
- Baseline pipeline (non-enriched) writes to `Results/`.
- Enrichment pipeline writes to `Results_Enrichment/`.

The recommended order is:
1. Run the baseline workflow end-to-end.
2. Freeze baseline artifacts.
3. Run enrichment workflow end-to-end.
4. Compare baseline vs enriched from their separate output folders.

## 1) Baseline Workflow (`Results/`)

### Stage 1: Build POD basis
Script:
- `stage1_pod.py`

Run:
```bash
cd Project_YvonMaday
python3 stage1_pod.py
```

Main outputs:
- `Results/Stage1/basis.npy`
- `Results/Stage1/u_ref.npy`
- `Results/Stage1/sigma.npy`
- `Results/Stage1/stage1_pod_summary.txt`

Key settings are inside `main()` in `stage1_pod.py`:
- `num_modes` or `pod_tol`
- `pod_method`
- `test_mu`

### Stage 2: Build PROM/HPROM qN dataset
Script:
- `stage2_build_prom_qn_dataset.py`

Run:
```bash
python3 stage2_build_prom_qn_dataset.py --backend prom
# or
python3 stage2_build_prom_qn_dataset.py --backend hprom
```

Main outputs:
- `Results/Stage2/prom_coeff_dataset_ntot*/per_mu/*`
- `Results/Stage2/prom_coeff_dataset_ntot*/meta.npy`
- `Results/Stage2/prom_coeff_dataset_ntot*/stage2_summary.txt`
- Optional ECSW weights inside `Results/Stage2/` when `--backend hprom`.

Useful flags:
```bash
--backend {prom,hprom}
--total-modes <int>
--rebuild-ecsw / --no-rebuild-ecsw
--ecsw-num-training-mu <int>
--ecsw-snap-sample-factor <int>
--ecsw-snap-time-offset <int>
--no-save-rom-snaps
--no-plots
```

Per-parameter Stage-2 files are now canonical:
- `mu.npy`
- `t.npy`
- `qN.npy` (full reduced coordinates)
- `rom_stats.npy` (+ backend alias stats file)

### Stage 3: Train nonlinear maps
Scripts:
- `stage3_perform_training_case_1_ann.py`
- `stage3_perform_training_case_2_ann.py`
- `stage3_perform_training_case_3_ann.py`
- `stage3_perform_training_rom_data_driven.py`

Run:
```bash
python3 stage3_perform_training_case_1_ann.py --dataset-backend prom
python3 stage3_perform_training_case_2_ann.py --dataset-backend prom
python3 stage3_perform_training_case_3_ann.py --dataset-backend prom
python3 stage3_perform_training_rom_data_driven.py --dataset-backend prom
```

If your Stage-2 dataset was built with HPROM, use `--dataset-backend hprom` instead.
For Case 1/2/3 trainers, you can choose the split online without re-running Stage 2:
```bash
python3 stage3_perform_training_case_2_ann.py --dataset-backend prom --primary-modes 20
```

Optional checkpoint naming (keeps backward compatibility):
```bash
python3 stage3_perform_training_case_2_ann.py --dataset-backend prom --model-name case2_model_n20.pt
python3 stage3_perform_training_rom_data_driven.py --dataset-backend prom --model-name data_model_n20
```
- If `--model-name` has no `.pt`, it is appended automatically.
- Defaults stay unchanged:
  - `case1_model.pt`, `case2_model.pt`, `case3_model.pt`, `rom_data_driven_model.pt`.

Main outputs:
- Models in `Results/Stage3/models/`
- Summaries in `Results/Stage3/`

### Baseline online runs

Linear non-ANN ROM/HPROM:
- `run_prom.py` (CLI)

Examples:
```bash
python3 run_prom.py --backend hprom --mu1 4.56 --mu2 0.019
python3 run_prom.py --backend prom  --mu1 4.56 --mu2 0.019
```

Note:
- `run_prom.py` now uses the linear `qN` space directly and saves `qN.npy` (no `qN_p/qN_s` split logic in this runner).

ANN closures (CLI):
- `run_prom_ann_case_1.py`
- `run_prom_ann_case_2.py`
- `run_prom_ann_case_3.py`
- `run_prom_ann_case_2_petrov_galerkin.py` (experimental Case 2 variant; PROM/HPROM)

Quick runs (defaults):
```bash
python3 run_prom_ann_case_1.py
python3 run_prom_ann_case_2.py
python3 run_prom_ann_case_3.py
```

Experimental Case 2 with enriched residual testing (PROM or HPROM):
```bash
python3 run_prom_ann_case_2_petrov_galerkin.py --backend prom  --mu1 4.56 --mu2 0.019
python3 run_prom_ann_case_2_petrov_galerkin.py --backend hprom --mu1 4.56 --mu2 0.019
python3 run_prom_ann_case_2_petrov_galerkin.py --backend hprom --mu1 5.19 --mu2 0.026
python3 run_prom_ann_case_2_petrov_galerkin.py --backend hprom --mu1 4.875 --mu2 0.0225
```

Notes for the experimental Case 2 variant:
- It keeps the standard Case-2 trial manifold, but uses an enriched residual-testing update based on `V_tot`.
- It supports `--backend {prom,hprom}`.
- In HPROM mode, ECSW is used. Use `--rebuild-ecsw` when you want to regenerate weights.
- It supports the same checkpoint-selection pattern:
```bash
python3 run_prom_ann_case_2_petrov_galerkin.py --model-name case2_model_n20.pt --mu1 4.56 --mu2 0.019
python3 run_prom_ann_case_2_petrov_galerkin.py --model-path /abs/path/to/case2_model_enriched.pt --mu1 4.56 --mu2 0.019
```

PROM vs HPROM examples:
```bash
# Case 1
python3 run_prom_ann_case_1.py --backend hprom --mu1 4.56 --mu2 0.019
python3 run_prom_ann_case_1.py --backend prom  --mu1 4.56 --mu2 0.019

# Case 2
python3 run_prom_ann_case_2.py --backend hprom --mu1 5.19 --mu2 0.026
python3 run_prom_ann_case_2.py --backend prom  --mu1 5.19 --mu2 0.026

# Case 3
python3 run_prom_ann_case_3.py --backend hprom --mu1 4.56 --mu2 0.019
python3 run_prom_ann_case_3.py --backend prom  --mu1 4.56 --mu2 0.019
```

Optional checkpoint selection for baseline ANN/data-driven runs:
```bash
python3 run_prom_ann_case_2.py --backend prom --model-name case2_model_n20.pt --mu1 4.56 --mu2 0.019
python3 run_prom_ann_case_2.py --backend prom --model-path /abs/path/to/case2_model_custom.pt --mu1 4.56 --mu2 0.019
python3 run_rom_data_driven.py --model-name data_model_n20.pt --mu1 4.56 --mu2 0.019
```
- `--model-path` overrides `--model-name`.
- Defaults stay unchanged (`case*_model.pt`, `rom_data_driven_model.pt`).

Useful flags for all `run_prom_ann_case_*` scripts:
```bash
--device {cpu,cuda}
--model-name <checkpoint_filename.pt>
--model-path <absolute_or_relative_checkpoint_path>
--no-ecsw
--rebuild-ecsw
--ecsw-num-training-mu <int>
--ecsw-snap-sample-factor <int>
--ecsw-snap-time-offset <int>
--max-its <int>
--relnorm-cutoff <float>
--min-delta <float>
--linear-solver {lstsq,normal_eq}
--normal-eq-reg <float>
```

Backend parsing logic (same for `run_prom.py`, `run_prom_ann_case_*`, and Petrov-Galerkin variants):
- Requested `--backend prom`:
  - effective backend is PROM
  - ECSW flags are ignored
- Requested `--backend hprom` (default ECSW ON):
  - effective backend is HPROM
  - ECSW setup + online hyperreduced solve are used
- Requested `--backend hprom --no-ecsw`:
  - effective backend falls back to PROM
  - this is printed in the run log

You can verify this in the console and in each run summary:
- `solve_backend_requested`
- `solve_backend_effective`
- `use_ecsw`

Fallback example:
```bash
python3 run_prom_ann_case_2_petrov_galerkin.py --backend hprom --no-ecsw --mu1 4.56 --mu2 0.019
python3 run_prom_ann_case_2_petrov_galerkin_enriched.py --backend hprom --no-ecsw --mu1 4.56 --mu2 0.019
```

Data-driven non-intrusive model:
- `run_rom_data_driven.py` (CLI)

Example:
```bash
python3 run_rom_data_driven.py --mu1 4.56 --mu2 0.019
```

Run outputs are under:
- `Results/Runs/Linear/`
- `Results/Runs/Case1/`
- `Results/Runs/Case2/`
- `Results/Runs/Case3/`
- `Results/Runs/DataDriven/`

Manual 3-point comparison (baseline, hard-coded):
```bash
# Point 1: (4.56, 0.019)
python3 run_prom.py --backend prom --mu1 4.56 --mu2 0.019
python3 run_prom_ann_case_1.py --backend prom --mu1 4.56 --mu2 0.019
python3 run_prom_ann_case_2.py --backend prom --mu1 4.56 --mu2 0.019
python3 run_prom_ann_case_3.py --backend prom --mu1 4.56 --mu2 0.019
python3 run_rom_data_driven.py --mu1 4.56 --mu2 0.019

# Point 2: (5.19, 0.026)
python3 run_prom.py --backend prom --mu1 5.19 --mu2 0.026
python3 run_prom_ann_case_1.py --backend prom --mu1 5.19 --mu2 0.026
python3 run_prom_ann_case_2.py --backend prom --mu1 5.19 --mu2 0.026
python3 run_prom_ann_case_3.py --backend prom --mu1 5.19 --mu2 0.026
python3 run_rom_data_driven.py --mu1 5.19 --mu2 0.026

# Verification point (center of 3x3 box): (4.875, 0.0225)
python3 run_prom.py --backend prom --mu1 4.875 --mu2 0.0225
python3 run_prom_ann_case_1.py --backend prom --mu1 4.875 --mu2 0.0225
python3 run_prom_ann_case_2.py --backend prom --mu1 4.875 --mu2 0.0225
python3 run_prom_ann_case_3.py --backend prom --mu1 4.875 --mu2 0.0225
python3 run_rom_data_driven.py --mu1 4.875 --mu2 0.0225
```

Notes:
- `run_prom.py --backend prom` is the linear `V_tot` upper-bound run.
- For fully HPROM runs, switch all ROM/ANN runners to `--backend hprom`.

Manual 3-point comparison (baseline, ANN with HPROM backend):
```bash
# Point 1: (4.56, 0.019)
python3 run_prom.py --backend hprom --mu1 4.56 --mu2 0.019
python3 run_prom_ann_case_1.py --backend hprom --mu1 4.56 --mu2 0.019
python3 run_prom_ann_case_2.py --backend hprom --mu1 4.56 --mu2 0.019
python3 run_prom_ann_case_3.py --backend hprom --mu1 4.56 --mu2 0.019
python3 run_rom_data_driven.py --mu1 4.56 --mu2 0.019

# Point 2: (5.19, 0.026)
python3 run_prom.py --backend hprom --mu1 5.19 --mu2 0.026
python3 run_prom_ann_case_1.py --backend hprom --mu1 5.19 --mu2 0.026
python3 run_prom_ann_case_2.py --backend hprom --mu1 5.19 --mu2 0.026
python3 run_prom_ann_case_3.py --backend hprom --mu1 5.19 --mu2 0.026
python3 run_rom_data_driven.py --mu1 5.19 --mu2 0.026

# Verification point (center): (4.875, 0.0225)
python3 run_prom.py --backend hprom --mu1 4.875 --mu2 0.0225
python3 run_prom_ann_case_1.py --backend hprom --mu1 4.875 --mu2 0.0225
python3 run_prom_ann_case_2.py --backend hprom --mu1 4.875 --mu2 0.0225
python3 run_prom_ann_case_3.py --backend hprom --mu1 4.875 --mu2 0.0225
python3 run_rom_data_driven.py --mu1 4.875 --mu2 0.0225
```

## 2) Enrichment Workflow (`Results_Enrichment/`)

This workflow does not modify baseline `Results/`.

### Stage 2 Enrichment: LHS with PROM/HPROM backend
Script:
- `stage2_build_enrichment_lhs_qn_dataset.py`

Run:
```bash
python3 stage2_build_enrichment_lhs_qn_dataset.py --backend prom
# or
python3 stage2_build_enrichment_lhs_qn_dataset.py --backend hprom
```

Main outputs:
- `Results_Enrichment/Stage2/prom_coeff_dataset_ntot*_enriched_lhs*/per_mu/*`
- `Results_Enrichment/Stage2/.../meta.npy`
- `Results_Enrichment/Stage2/.../stage2_enrichment_summary.txt`
- For `--backend hprom`, ECSW weights are copied from the baseline Stage-2 dataset.

Design behavior:
- Optionally copies baseline per-parameter qN samples.
- Adds new LHS points.
- Stores reduced data only (`qN`, stats), no full snapshots.

### Stage 3 Enrichment: train enriched nonlinear maps
Option A (recommended orchestrator):
- `stage3_train_enriched_nonintrusive_maps.py`

Run:
```bash
python3 stage3_train_enriched_nonintrusive_maps.py --dataset-backend prom
# or
python3 stage3_train_enriched_nonintrusive_maps.py --dataset-backend hprom
```

Optional split override for Case 1/2/3 enriched trainers:
```bash
python3 stage3_train_enriched_nonintrusive_maps.py --dataset-backend prom --primary-modes 20
```

Optional enriched checkpoint names via orchestrator:
```bash
python3 stage3_train_enriched_nonintrusive_maps.py --dataset-backend prom \
  --case1-model-name case1_model_enriched_n20.pt \
  --case2-model-name case2_model_enriched_n20.pt \
  --case3-model-name case3_model_enriched_n20.pt \
  --data-model-name data_model_enriched_n20.pt
```

Option B (individual scripts):
- `stage3_perform_training_case_1_ann_enriched.py`
- `stage3_perform_training_case_2_ann_enriched.py`
- `stage3_perform_training_case_3_ann_enriched.py`
- `stage3_perform_training_rom_data_driven_enriched.py`

Each individual enriched trainer also accepts `--model-name`.

Main outputs:
- `Results_Enrichment/Stage3/<dataset_name>/models/*_enriched.pt`
- `Results_Enrichment/Stage3/<dataset_name>/*_summary_enriched.txt`

### Enriched online runs

ANN closures (CLI):
- `run_prom_ann_case_1_enriched.py`
- `run_prom_ann_case_2_enriched.py`
- `run_prom_ann_case_3_enriched.py`
- `run_prom_ann_case_2_petrov_galerkin_enriched.py` (experimental Case 2 variant; PROM/HPROM)

Quick runs (defaults):
```bash
python3 run_prom_ann_case_1_enriched.py
python3 run_prom_ann_case_2_enriched.py
python3 run_prom_ann_case_3_enriched.py
```

Experimental enriched Case 2 with enriched residual testing (PROM or HPROM):
```bash
python3 run_prom_ann_case_2_petrov_galerkin_enriched.py --backend prom  --mu1 4.56 --mu2 0.019
python3 run_prom_ann_case_2_petrov_galerkin_enriched.py --backend hprom --mu1 4.56 --mu2 0.019
python3 run_prom_ann_case_2_petrov_galerkin_enriched.py --backend hprom --mu1 5.19 --mu2 0.026
python3 run_prom_ann_case_2_petrov_galerkin_enriched.py --backend hprom --mu1 4.875 --mu2 0.0225
```

Notes for the experimental enriched Case 2 variant:
- It supports `--backend {prom,hprom}`.
- In HPROM mode, ECSW is used. Use `--rebuild-ecsw` when you want to regenerate weights.

Optional checkpoint selection:
```bash
python3 run_prom_ann_case_2_petrov_galerkin_enriched.py --model-name case2_model_enriched_n20.pt --mu1 4.56 --mu2 0.019
python3 run_prom_ann_case_2_petrov_galerkin_enriched.py --model-path /abs/path/to/case2_model_enriched.pt --mu1 4.56 --mu2 0.019
```

PROM vs HPROM examples:
```bash
# Case 1 enriched
python3 run_prom_ann_case_1_enriched.py --backend hprom --mu1 4.56 --mu2 0.019
python3 run_prom_ann_case_1_enriched.py --backend prom  --mu1 4.56 --mu2 0.019

# Case 2 enriched
python3 run_prom_ann_case_2_enriched.py --backend hprom --mu1 5.19 --mu2 0.026
python3 run_prom_ann_case_2_enriched.py --backend prom  --mu1 5.19 --mu2 0.026

# Case 3 enriched
python3 run_prom_ann_case_3_enriched.py --backend hprom --mu1 4.56 --mu2 0.019
python3 run_prom_ann_case_3_enriched.py --backend prom  --mu1 4.56 --mu2 0.019
```

Optional checkpoint selection for enriched ANN/data-driven runs:
```bash
python3 run_prom_ann_case_2_enriched.py --backend prom --model-name case2_model_enriched_n20.pt --mu1 4.56 --mu2 0.019
python3 run_prom_ann_case_2_enriched.py --backend prom --model-path /abs/path/to/case2_model_enriched_custom.pt --mu1 4.56 --mu2 0.019
python3 run_rom_data_driven_enriched.py --model-name data_model_enriched_n20.pt --mu1 4.56 --mu2 0.019
```
- `--model-path` overrides `--model-name`.
- Defaults stay unchanged (`case*_model_enriched.pt`, `rom_data_driven_model_enriched.pt`).

Extra enriched-only flags:
```bash
--model-name <enriched_checkpoint_filename.pt>
--model-path <path_to_enriched_checkpoint.pt>
--ecsw-tag <string>
```

Data-driven enriched:
- `run_rom_data_driven_enriched.py` (CLI)

Example:
```bash
python3 run_rom_data_driven_enriched.py --mu1 4.56 --mu2 0.019
```

Run outputs are under:
- `Results_Enrichment/Runs/Case1/`
- `Results_Enrichment/Runs/Case2/`
- `Results_Enrichment/Runs/Case3/`
- `Results_Enrichment/Runs/DataDriven/`
- `Results_Enrichment/Runs/ECSW/`

Manual 3-point comparison (enriched):
```bash
# Point 1: (4.56, 0.019)
python3 run_prom.py --backend prom --mu1 4.56 --mu2 0.019
python3 run_prom_ann_case_1_enriched.py --backend prom --mu1 4.56 --mu2 0.019
python3 run_prom_ann_case_2_enriched.py --backend prom --mu1 4.56 --mu2 0.019
python3 run_prom_ann_case_3_enriched.py --backend prom --mu1 4.56 --mu2 0.019
python3 run_rom_data_driven_enriched.py --mu1 4.56 --mu2 0.019

# Point 2: (5.19, 0.026)
python3 run_prom.py --backend prom --mu1 5.19 --mu2 0.026
python3 run_prom_ann_case_1_enriched.py --backend prom --mu1 5.19 --mu2 0.026
python3 run_prom_ann_case_2_enriched.py --backend prom --mu1 5.19 --mu2 0.026
python3 run_prom_ann_case_3_enriched.py --backend prom --mu1 5.19 --mu2 0.026
python3 run_rom_data_driven_enriched.py --mu1 5.19 --mu2 0.026

# Verification point: (4.875, 0.0225)
python3 run_prom.py --backend prom --mu1 4.875 --mu2 0.0225
python3 run_prom_ann_case_1_enriched.py --backend prom --mu1 4.875 --mu2 0.0225
python3 run_prom_ann_case_2_enriched.py --backend prom --mu1 4.875 --mu2 0.0225
python3 run_prom_ann_case_3_enriched.py --backend prom --mu1 4.875 --mu2 0.0225
python3 run_rom_data_driven_enriched.py --mu1 4.875 --mu2 0.0225
```

Manual 3-point comparison (enriched, ANN with HPROM backend):
```bash
# Point 1: (4.56, 0.019)
python3 run_prom.py --backend hprom --mu1 4.56 --mu2 0.019
python3 run_prom_ann_case_1_enriched.py --backend hprom --mu1 4.56 --mu2 0.019
python3 run_prom_ann_case_2_enriched.py --backend hprom --mu1 4.56 --mu2 0.019
python3 run_prom_ann_case_3_enriched.py --backend hprom --mu1 4.56 --mu2 0.019
python3 run_rom_data_driven_enriched.py --mu1 4.56 --mu2 0.019

# Point 2: (5.19, 0.026)
python3 run_prom.py --backend hprom --mu1 5.19 --mu2 0.026
python3 run_prom_ann_case_1_enriched.py --backend hprom --mu1 5.19 --mu2 0.026
python3 run_prom_ann_case_2_enriched.py --backend hprom --mu1 5.19 --mu2 0.026
python3 run_prom_ann_case_3_enriched.py --backend hprom --mu1 5.19 --mu2 0.026
python3 run_rom_data_driven_enriched.py --mu1 5.19 --mu2 0.026

# Verification point: (4.875, 0.0225)
python3 run_prom.py --backend hprom --mu1 4.875 --mu2 0.0225
python3 run_prom_ann_case_1_enriched.py --backend hprom --mu1 4.875 --mu2 0.0225
python3 run_prom_ann_case_2_enriched.py --backend hprom --mu1 4.875 --mu2 0.0225
python3 run_prom_ann_case_3_enriched.py --backend hprom --mu1 4.875 --mu2 0.0225
python3 run_rom_data_driven_enriched.py --mu1 4.875 --mu2 0.0225
```

## 3) Practical Notes

- Baseline and enrichment are intentionally decoupled by file names and output roots.
- Baseline scripts (`*.py`) read/write `Results/*`.
- Enriched scripts (`*_enriched.py`) read/write `Results_Enrichment/*`.
- `stage1_pod.py` is shared; enrichment reuses the same POD basis by design.
- Stage-2 and Stage-3 training scripts are CLI-driven for backend selection.
- Keep Stage-2 and Stage-3 backend consistent (`prom` with `prom`, `hprom` with `hprom`).
- Stage-2 dataset folder names are backend-agnostic (`prom_coeff_dataset_ntot*`), so rebuilding the same `ntot` with another backend overwrites that dataset.
- Stage-2 stores full reduced coordinates only (`qN.npy`); primary/truncated split is chosen in Stage-3 (`--primary-modes`).
- Run scripts are CLI-driven: `run_prom.py`, `run_prom_ann_case_*`, `run_rom_data_driven.py`, and enriched counterparts.
- In HPROM runs (`run_prom.py` and ANN runners), summaries separate timing as `ecsw_setup_elapsed_s` (weights load/build) and `online_solve_elapsed_s` (ROM solve only). `elapsed_s` is kept for compatibility and equals `online_solve_elapsed_s`.

## 4) Minimal End-to-End Command Sequence

Baseline:
```bash
python3 stage1_pod.py
python3 stage2_build_prom_qn_dataset.py --backend prom
python3 stage3_perform_training_case_1_ann.py --dataset-backend prom
python3 stage3_perform_training_case_2_ann.py --dataset-backend prom
python3 stage3_perform_training_case_3_ann.py --dataset-backend prom
python3 stage3_perform_training_rom_data_driven.py --dataset-backend prom
python3 run_prom_ann_case_1.py
python3 run_prom_ann_case_2.py
python3 run_prom_ann_case_3.py
python3 run_rom_data_driven.py --mu1 4.56 --mu2 0.019
```

Enriched:
```bash
python3 stage2_build_enrichment_lhs_qn_dataset.py --backend prom
python3 stage3_train_enriched_nonintrusive_maps.py --dataset-backend prom
python3 run_prom_ann_case_1_enriched.py
python3 run_prom_ann_case_2_enriched.py
python3 run_prom_ann_case_3_enriched.py
python3 run_rom_data_driven_enriched.py --mu1 4.56 --mu2 0.019
```

## 5) Backend Recipes

Fully PROM training data + training:
```bash
python3 stage2_build_prom_qn_dataset.py --backend prom
python3 stage3_perform_training_case_1_ann.py --dataset-backend prom
python3 stage3_perform_training_case_2_ann.py --dataset-backend prom
python3 stage3_perform_training_case_3_ann.py --dataset-backend prom
python3 stage3_perform_training_rom_data_driven.py --dataset-backend prom
```

Fully HPROM training data + training:
```bash
python3 stage2_build_prom_qn_dataset.py --backend hprom
python3 stage3_perform_training_case_1_ann.py --dataset-backend hprom
python3 stage3_perform_training_case_2_ann.py --dataset-backend hprom
python3 stage3_perform_training_case_3_ann.py --dataset-backend hprom
python3 stage3_perform_training_rom_data_driven.py --dataset-backend hprom
```

## 6) Coefficient-Error Analysis (No New Runs)

To analyze coefficient errors against the linear `n_tot` PROM reference, use:

```bash
python3 250x250/analyze_coefficient_errors.py --stages baseline enriched
```

Important:
- This script **does not run** PROM/HPROM/ANN/FOM solvers.
- It only reads existing files under:
  - `250x250/Results/`
  - `250x250/Results_Enrichment/`

Default points (verification first):
- `(4.875, 0.0225)`
- `(4.56, 0.019)`
- `(5.19, 0.026)`

Main outputs:
- Plots in `250x250/Figures/coeff_errors/`
  - `*_coeff_abs_rel_vs_index.png` (absolute + relative per-coefficient errors)
  - `*_coeff_abs_heatmap_grid.png` (time-propagation absolute error heatmaps)
  - `*_coeff_rel_heatmap_grid.png` (time-propagation relative error heatmaps)
  - `*_case{1,2,3}_secondary_decomposition.png` (source-of-error decomposition)
- CSV summaries:
  - `250x250/Figures/coeff_errors/model_error_summary.csv`
  - `250x250/Figures/coeff_errors/decomposition_summary.csv`
