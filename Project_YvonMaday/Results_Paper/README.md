# Results_Paper (HPROM-Only Campaign)

This folder is the runbook for the **new paper campaign** in `Project_YvonMaday`.

Scope for first deployment:
- `Case 1` ANN closure
- `Case 2` ANN closure
- `Case 3` ANN closure
- non-intrusive coefficient-space POD-NN-ROM surrogate (`(mu,t) -> q_N`)
- enriched counterparts
- non-intrusive latent POD-DL-ROM surrogate (`(mu,t) -> z -> q_N`) (**baseline integrated**)
- PROM-POD-AE intrusive latent-manifold baseline (**baseline integrated**)

Naming convention used in this campaign:
- `POD-DL-ROM` means the non-intrusive latent data-driven family (Fresca/Manzoni style).
- `POD-NN-ROM` means the non-intrusive coefficient-space map `(mu,t) -> q_N`.
- `PROM-POD-AE` means the intrusive latent-manifold projection track (conceptually similar to intrusive nonlinear-manifold HPROM literature, but with our project naming).
- Paper/table labels (recommended):
  - `POD-NN-ROM`: direct coefficient-space map `(\mu,t) -> q_N`
  - `POD-DL-ROM`: latent map `(\mu,t) -> z -> q_N`
  - `PROM-POD-AE`: intrusive latent-manifold residual-minimization track

Important policy for this campaign:
- Stage-2 dataset generation uses `--backend hprom`.
- Stage-3 training uses `--dataset-backend hprom`.
- Online intrusive runs use `--backend hprom`.
- Quantitative tables/figures for this campaign should be read from HPROM-tagged runs only.

---

## 0) Start Here

```bash
cd /home/kratos/burgers2d-rom-workbench/Project_YvonMaday
```

Optional (recommended on clusters):
```bash
export MPLCONFIGDIR=/tmp/mplconfig_${USER}
```

### Safety note (important)
`stage2_build_prom_qn_dataset.py` writes to:
- `Results/Stage2/prom_coeff_dataset_ntot<...>`

So if you currently have a PROM Stage-2 dataset you want to keep untouched, back it up before rebuilding HPROM Stage-2:

```bash
mkdir -p Results_Paper/backups
cp -r Results/Stage2/prom_coeff_dataset_ntot151 Results_Paper/backups/prom_coeff_dataset_ntot151_before_hprom_$(date +%Y%m%d_%H%M%S)
```

---

## 1) Stage 1 POD Basis

```bash
python3 stage1_pod.py
```

Check:
- `Results/Stage1/stage1_pod_summary.txt`

---

## 2) Stage 2 Baseline Dataset (HPROM in V_tot)

Build the ROM-consistent coefficient dataset from **linear HPROM** in full truncated space (`n_tot`, default here 151):

```bash
python3 stage2_build_prom_qn_dataset.py \
  --backend hprom \
  --total-modes 151 \
  --rebuild-ecsw \
  --ecsw-num-training-mu 9 \
  --ecsw-snapshot-percent 2.0 \
  --ecsw-random-seed 42 \
  --ecsw-ensure-mu-coverage
```

Check:
- `Results/Stage2/prom_coeff_dataset_ntot151/meta.npy` (`solve_backend` should be `hprom`)
- `Results/Stage2/prom_coeff_dataset_ntot151/stage2_summary.txt`

---

## 3) Stage 3 Baseline Training (HPROM dataset)

Use dedicated HPROM model names to keep this campaign isolated from older checkpoints.

```bash
python3 stage3_perform_training_case_1_ann.py \
  --dataset-backend hprom \
  --dataset-ntot 151 \
  --primary-modes 10 \
  --model-name case1_model_hprom.pt

python3 stage3_perform_training_case_2_ann.py \
  --dataset-backend hprom \
  --dataset-ntot 151 \
  --primary-modes 10 \
  --model-name case2_model_hprom.pt

python3 stage3_perform_training_case_3_ann.py \
  --dataset-backend hprom \
  --dataset-ntot 151 \
  --primary-modes 10 \
  --model-name case3_model_hprom.pt

python3 stage3_perform_training_rom_data_driven.py \
  --dataset-backend hprom \
  --dataset-ntot 151 \
  --model-name rom_data_driven_model_hprom.pt

python3 stage3_perform_training_prom_pod_ae.py \
  --dataset-backend hprom \
  --dataset-ntot 151 \
  --model-name prom_pod_ae_model_hprom.pt \
  --latent-dim 5 \
  --hidden-dims 192,96,48 \
  --activation tanh \
  --scaling minmax_-1_1

python3 stage3_perform_training_pod_dl_data_driven.py \
  --dataset-backend hprom \
  --dataset-ntot 151 \
  --model-name pod_dl_data_driven_model_hprom.pt \
  --latent-dim 5 \
  --encoder-hidden-dims 192,96,48 \
  --decoder-hidden-dims 48,96,192 \
  --dynamics-hidden-dims 64,128,128 \
  --activation tanh \
  --x-scaling minmax_-1_1 \
  --q-scaling minmax_-1_1 \
  --omega-data 1.0 \
  --omega-latent 0.1

# Optional POD-DL-ROM variant (dynamics branch closer to POD-NN-ROM MLP):
python3 stage3_perform_training_pod_dl_data_driven.py \
  --dataset-backend hprom \
  --dataset-ntot 151 \
  --model-name <your_pod_dl_model_name>.pt \
  --latent-dim 20 \
  --encoder-hidden-dims 192,96,48 \
  --decoder-hidden-dims 48,96,192 \
  --dynamics-hidden-dims 32,64,128,256,256 \
  --activation elu \
  --x-scaling zscore \
  --q-scaling zscore \
  --omega-data 1.0 \
  --omega-latent 0.02 \
  --pretrain-epochs 150
```

---

## 4) Baseline Online Evaluations (HPROM)

Parameter points used in the report:
- Verification: `mu=(4.875, 0.0225)`
- Off-grid test 1: `mu=(4.56, 0.019)`
- Off-grid test 2: `mu=(5.19, 0.026)`

For all HPROM intrusive runners in this campaign, keep ECSW sampling aligned with Stage 2:

```bash
ECSW_ARGS="--ecsw-num-training-mu 9 --ecsw-snap-time-offset 3 --ecsw-snapshot-percent 2.0 --ecsw-random-seed 42 --ecsw-ensure-mu-coverage"
```

You can append `$ECSW_ARGS` to every `run_prom.py` / `run_prom_ann_case_*` HPROM command below.

### 4.1 Linear HPROM reference

```bash
python3 run_prom.py --backend hprom --mu1 4.875 --mu2 0.0225 --total-modes 151
python3 run_prom.py --backend hprom --mu1 4.56  --mu2 0.019  --total-modes 151
python3 run_prom.py --backend hprom --mu1 5.19  --mu2 0.026  --total-modes 151
```

### 4.2 ANN closures (Case 1/2/3)

```bash
python3 run_prom_ann_case_1.py --backend hprom --mu1 4.875 --mu2 0.0225 --model-name case1_model_hprom.pt
python3 run_prom_ann_case_1.py --backend hprom --mu1 4.56  --mu2 0.019  --model-name case1_model_hprom.pt
python3 run_prom_ann_case_1.py --backend hprom --mu1 5.19  --mu2 0.026  --model-name case1_model_hprom.pt

python3 run_prom_ann_case_2.py --backend hprom --mu1 4.875 --mu2 0.0225 --model-name case2_model_hprom.pt
python3 run_prom_ann_case_2.py --backend hprom --mu1 4.56  --mu2 0.019  --model-name case2_model_hprom.pt
python3 run_prom_ann_case_2.py --backend hprom --mu1 5.19  --mu2 0.026  --model-name case2_model_hprom.pt

python3 run_prom_ann_case_3.py --backend hprom --mu1 4.875 --mu2 0.0225 --model-name case3_model_hprom.pt
python3 run_prom_ann_case_3.py --backend hprom --mu1 4.56  --mu2 0.019  --model-name case3_model_hprom.pt
python3 run_prom_ann_case_3.py --backend hprom --mu1 5.19  --mu2 0.026  --model-name case3_model_hprom.pt

```

### 4.3 Data-driven

```bash
python3 run_rom_data_driven.py --mu1 4.875 --mu2 0.0225 --model-name rom_data_driven_model_hprom.pt
python3 run_rom_data_driven.py --mu1 4.56  --mu2 0.019  --model-name rom_data_driven_model_hprom.pt
python3 run_rom_data_driven.py --mu1 5.19  --mu2 0.026  --model-name rom_data_driven_model_hprom.pt

```

### 4.4 Latent tracks (baseline)

PROM-POD-AE (intrusive, online HPROM):
```bash
python3 run_prom_pod_ae.py --backend hprom --mu1 4.875 --mu2 0.0225 --model-name prom_pod_ae_model_hprom.pt $ECSW_ARGS
python3 run_prom_pod_ae.py --backend hprom --mu1 4.56  --mu2 0.019  --model-name prom_pod_ae_model_hprom.pt $ECSW_ARGS
python3 run_prom_pod_ae.py --backend hprom --mu1 5.19  --mu2 0.026  --model-name prom_pod_ae_model_hprom.pt $ECSW_ARGS
```

POD-DL-ROM (non-intrusive latent data-driven):
```bash
python3 run_pod_dl_data_driven.py --mu1 4.875 --mu2 0.0225 --model-name <your_pod_dl_model_name>.pt
python3 run_pod_dl_data_driven.py --mu1 4.56  --mu2 0.019  --model-name <your_pod_dl_model_name>.pt
python3 run_pod_dl_data_driven.py --mu1 5.19  --mu2 0.026  --model-name <your_pod_dl_model_name>.pt
```

---

## 5) Stage 2 Enrichment Dataset (HPROM)

```bash
python3 stage2_build_enrichment_lhs_qn_dataset.py \
  --backend hprom \
  --lhs-samples 20 \
  --lhs-seed 42 \
  --total-modes 151 \
  --copy-base-dataset \
  --reuse-base-ecsw-weights
```

Check:
- `Results_Enrichment/Stage2/prom_coeff_dataset_ntot151_enriched_lhs20/meta.npy`
- `Results_Enrichment/Stage2/prom_coeff_dataset_ntot151_enriched_lhs20/stage2_enrichment_summary.txt`

---

## 6) Stage 3 Enriched Training (HPROM dataset)

```bash
python3 stage3_perform_training_case_1_ann_enriched.py \
  --dataset-backend hprom \
  --dataset-ntot 151 \
  --primary-modes 10 \
  --model-name case1_model_enriched_hprom.pt

python3 stage3_perform_training_case_2_ann_enriched.py \
  --dataset-backend hprom \
  --dataset-ntot 151 \
  --primary-modes 10 \
  --model-name case2_model_enriched_hprom.pt

python3 stage3_perform_training_case_3_ann_enriched.py \
  --dataset-backend hprom \
  --dataset-ntot 151 \
  --primary-modes 10 \
  --model-name case3_model_enriched_hprom.pt

python3 stage3_perform_training_rom_data_driven_enriched.py \
  --dataset-backend hprom \
  --dataset-ntot 151 \
  --model-name rom_data_driven_model_enriched_hprom.pt

python3 stage3_perform_training_prom_pod_ae_enriched.py \
  --dataset-backend hprom \
  --dataset-ntot 151 \
  --latent-dim 5 \
  --hidden-dims 256,128,64 \
  --model-name prom_pod_ae_model_enriched_hprom.pt

python3 stage3_perform_training_pod_dl_data_driven_enriched.py \
  --dataset-backend hprom \
  --dataset-ntot 151 \
  --latent-dim 5 \
  --encoder-hidden-dims 256,128 \
  --decoder-hidden-dims 128,256 \
  --dynamics-hidden-dims 64,128,128 \
  --model-name pod_dl_data_driven_model_enriched_hprom.pt
```

---

## 7) Enriched Online Evaluations (HPROM)

```bash
python3 run_prom_ann_case_1_enriched.py --backend hprom --mu1 4.875 --mu2 0.0225 --model-name case1_model_enriched_hprom.pt
python3 run_prom_ann_case_1_enriched.py --backend hprom --mu1 4.56  --mu2 0.019  --model-name case1_model_enriched_hprom.pt
python3 run_prom_ann_case_1_enriched.py --backend hprom --mu1 5.19  --mu2 0.026  --model-name case1_model_enriched_hprom.pt

python3 run_prom_ann_case_2_enriched.py --backend hprom --mu1 4.875 --mu2 0.0225 --model-name case2_model_enriched_hprom.pt
python3 run_prom_ann_case_2_enriched.py --backend hprom --mu1 4.56  --mu2 0.019  --model-name case2_model_enriched_hprom.pt
python3 run_prom_ann_case_2_enriched.py --backend hprom --mu1 5.19  --mu2 0.026  --model-name case2_model_enriched_hprom.pt

python3 run_prom_ann_case_3_enriched.py --backend hprom --mu1 4.875 --mu2 0.0225 --model-name case3_model_enriched_hprom.pt
python3 run_prom_ann_case_3_enriched.py --backend hprom --mu1 4.56  --mu2 0.019  --model-name case3_model_enriched_hprom.pt
python3 run_prom_ann_case_3_enriched.py --backend hprom --mu1 5.19  --mu2 0.026  --model-name case3_model_enriched_hprom.pt

python3 run_rom_data_driven_enriched.py --mu1 4.875 --mu2 0.0225 --model-name rom_data_driven_model_enriched_hprom.pt
python3 run_rom_data_driven_enriched.py --mu1 4.56  --mu2 0.019  --model-name rom_data_driven_model_enriched_hprom.pt
python3 run_rom_data_driven_enriched.py --mu1 5.19  --mu2 0.026  --model-name rom_data_driven_model_enriched_hprom.pt

python3 run_prom_pod_ae_enriched.py --backend hprom --mu1 4.875 --mu2 0.0225 --model-name prom_pod_ae_model_enriched_hprom.pt
python3 run_prom_pod_ae_enriched.py --backend hprom --mu1 4.56  --mu2 0.019  --model-name prom_pod_ae_model_enriched_hprom.pt
python3 run_prom_pod_ae_enriched.py --backend hprom --mu1 5.19  --mu2 0.026  --model-name prom_pod_ae_model_enriched_hprom.pt

python3 run_pod_dl_data_driven_enriched.py --mu1 4.875 --mu2 0.0225 --model-name pod_dl_data_driven_model_enriched_hprom.pt
python3 run_pod_dl_data_driven_enriched.py --mu1 4.56  --mu2 0.019  --model-name pod_dl_data_driven_model_enriched_hprom.pt
python3 run_pod_dl_data_driven_enriched.py --mu1 5.19  --mu2 0.026  --model-name pod_dl_data_driven_model_enriched_hprom.pt

```

---

## 8) Latent Track Status (Current)

Now integrated directly in `Project_YvonMaday`:
- `stage3_perform_training_prom_pod_ae.py`
- `run_prom_pod_ae.py`
- `stage3_perform_training_pod_dl_data_driven.py`
- `run_pod_dl_data_driven.py`
- `stage3_perform_training_prom_pod_ae_enriched.py`
- `run_prom_pod_ae_enriched.py`
- `stage3_perform_training_pod_dl_data_driven_enriched.py`
- `run_pod_dl_data_driven_enriched.py`

Naming used in this manuscript:
- `POD-NN-ROM`: non-intrusive coefficient-space map `(mu,t) -> q_N`.
- `POD-DL-ROM`: non-intrusive latent data-driven map `(mu,t) -> z -> q_N`.
- `PROM-POD-AE`: intrusive latent-manifold method solved online by PROM/HPROM residual minimization.

Current status:
- baseline latent comparisons: integrated.
- enriched latent comparisons: integrated.

---

## 9) Where Results Are Written

Canonical run outputs are still written by each runner under:
- baseline: `Results/Runs/...`
- enriched: `Results_Enrichment/Runs/...`

Latent baseline output folders:
- `Results/Runs/PODAE/`
- `Results/Runs/PODDL/`

Latent enriched output folders:
- `Results_Enrichment/Runs/PODAE/`
- `Results_Enrichment/Runs/PODDL/`

For paper assembly, keep this folder (`Results_Paper/`) as:
- runbook + manuscript draft,
- optional curated exports copied from canonical run folders.

---

## 10) Manuscript Update Checklist

For baseline tables/figures in `manuscript.tex`, collect summaries from:
- linear HPROM reference: `Results/Runs/Linear/*_summary.txt`
- ANN closures: `Results/Runs/Case1/*_summary.txt`, `Case2/*_summary.txt`, `Case3/*_summary.txt`
- POD-NN-ROM: `Results/Runs/DataDriven/*/rom_data_driven_summary.txt`
- PROM-POD-AE: `Results/Runs/PODAE/*_summary.txt`
- POD-DL-ROM: `Results/Runs/PODDL/*/pod_dl_data_driven_summary.txt`

For enriched tables/figures, use:
- ANN closures: `Results_Enrichment/Runs/Case1/*_summary.txt`, `Case2/*_summary.txt`, `Case3/*_summary.txt`
- POD-NN-ROM: `Results_Enrichment/Runs/DataDriven/*/rom_data_driven_enriched_summary.txt`
- PROM-POD-AE: `Results_Enrichment/Runs/PODAE/*_summary.txt`
- POD-DL-ROM: `Results_Enrichment/Runs/PODDL/*/pod_dl_data_driven_enriched_summary.txt`

For enriched tables/figures:
- use `Results_Enrichment/Runs/Case1`, `Case2`, `Case3`, `DataDriven`.
- latent enriched rows should be marked pending until enriched latent wrappers are added.
