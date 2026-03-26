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
python3 stage2_build_prom_qn_dataset.py
```

Main outputs:
- `Results/Stage2/prom_coeff_dataset_ntot*/per_mu/*`
- `Results/Stage2/prom_coeff_dataset_ntot*/meta.npy`
- `Results/Stage2/prom_coeff_dataset_ntot*/stage2_summary.txt`
- Optional ECSW weights inside `Results/Stage2/` when `solve_backend="hprom"`.

Important:
- Stage 3 scripts default to `dataset_backend = "hprom"`.
- If you want to train Stage 3 directly from Stage 2 defaults, set `solve_backend = "hprom"` in `stage2_build_prom_qn_dataset.py`.

Key settings in `main()`:
- `solve_backend` (`"prom"` or `"hprom"`)
- `total_modes`
- `primary_modes`
- `rebuild_ecsw_weights`

### Stage 3: Train nonlinear maps
Scripts:
- `stage3_perform_training_case_1_ann.py`
- `stage3_perform_training_case_2_ann.py`
- `stage3_perform_training_case_3_ann.py`
- `stage3_perform_training_rom_data_driven.py`

Run:
```bash
python3 stage3_perform_training_case_1_ann.py
python3 stage3_perform_training_case_2_ann.py
python3 stage3_perform_training_case_3_ann.py
python3 stage3_perform_training_rom_data_driven.py
```

Main outputs:
- Models in `Results/Stage3/models/`
- Summaries in `Results/Stage3/`

### Baseline online runs

Linear non-ANN ROM/HPROM:
- `run_prom.py` (CLI)

Examples:
```bash
python3 run_prom.py --backend hprom --mu1 4.56 --mu2 0.019 --primary-modes 10
python3 run_prom.py --backend prom  --mu1 4.56 --mu2 0.019 --primary-modes 10
```

ANN closures (settings inside each script `main()`):
- `run_prom_ann_case_1.py`
- `run_prom_ann_case_2.py`
- `run_prom_ann_case_3.py`

Run:
```bash
python3 run_prom_ann_case_1.py
python3 run_prom_ann_case_2.py
python3 run_prom_ann_case_3.py
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

## 2) Enrichment Workflow (`Results_Enrichment/`)

This workflow does not modify baseline `Results/`.

### Stage 2 Enrichment: LHS + new HPROM ECSW
Script:
- `stage2_build_enrichment_lhs_qn_dataset.py`

Run:
```bash
python3 stage2_build_enrichment_lhs_qn_dataset.py
```

Main outputs:
- `Results_Enrichment/Stage2/prom_coeff_dataset_ntot*_enriched_lhs*/per_mu/*`
- `Results_Enrichment/Stage2/.../meta.npy`
- `Results_Enrichment/Stage2/.../stage2_enrichment_summary.txt`
- New enrichment ECSW weights in the same dataset folder.

Design behavior:
- Optionally copies baseline per-parameter qN samples.
- Adds new LHS points.
- Stores reduced data only (`qN`, `qN_p`, `qN_s`, stats), no full snapshots.

### Stage 3 Enrichment: train enriched nonlinear maps
Option A (recommended orchestrator):
- `stage3_train_enriched_nonintrusive_maps.py`

Run:
```bash
python3 stage3_train_enriched_nonintrusive_maps.py
```

Option B (individual scripts):
- `stage3_perform_training_case_1_ann_enriched.py`
- `stage3_perform_training_case_2_ann_enriched.py`
- `stage3_perform_training_case_3_ann_enriched.py`
- `stage3_perform_training_rom_data_driven_enriched.py`

Main outputs:
- `Results_Enrichment/Stage3/<dataset_name>/models/*_enriched.pt`
- `Results_Enrichment/Stage3/<dataset_name>/*_summary_enriched.txt`

### Enriched online runs

ANN closures:
- `run_prom_ann_case_1_enriched.py`
- `run_prom_ann_case_2_enriched.py`
- `run_prom_ann_case_3_enriched.py`

Run:
```bash
python3 run_prom_ann_case_1_enriched.py
python3 run_prom_ann_case_2_enriched.py
python3 run_prom_ann_case_3_enriched.py
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

## 3) Practical Notes

- Baseline and enrichment are intentionally decoupled by file names and output roots.
- Baseline scripts (`*.py`) read/write `Results/*`.
- Enriched scripts (`*_enriched.py`) read/write `Results_Enrichment/*`.
- `stage1_pod.py` is shared; enrichment reuses the same POD basis by design.
- Most Stage 2/3/case-run scripts are configured by editing values in `main()`.
- `run_prom.py` and `run_rom_data_driven.py` (and enriched data-driven run) provide CLI options.

## 4) Minimal End-to-End Command Sequence

Baseline:
```bash
python3 stage1_pod.py
python3 stage2_build_prom_qn_dataset.py
python3 stage3_perform_training_case_1_ann.py
python3 stage3_perform_training_case_2_ann.py
python3 stage3_perform_training_case_3_ann.py
python3 stage3_perform_training_rom_data_driven.py
python3 run_prom_ann_case_1.py
python3 run_prom_ann_case_2.py
python3 run_prom_ann_case_3.py
python3 run_rom_data_driven.py --mu1 4.56 --mu2 0.019
```

Enriched:
```bash
python3 stage2_build_enrichment_lhs_qn_dataset.py
python3 stage3_train_enriched_nonintrusive_maps.py
python3 run_prom_ann_case_1_enriched.py
python3 run_prom_ann_case_2_enriched.py
python3 run_prom_ann_case_3_enriched.py
python3 run_rom_data_driven_enriched.py --mu1 4.56 --mu2 0.019
```
