#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export PAPER_RESULTS_ROOT="${PAPER_RESULTS_ROOT:-$PWD/Results_Paper}"
export PAPER_TAG="${PAPER_TAG:-mlspg_hprom_enrichment}"
export PAPER_ROOT="$PAPER_RESULTS_ROOT/$PAPER_TAG"
export DATASET_DIR="${DATASET_DIR:-$PWD/Results_Paper/mlspg_hprom_enrichment/Stage2/prom_coeff_dataset_ntot151_enriched_lhs20}"
export STAGE3_DIR="$PAPER_ROOT/Stage3"
export MODELS_DIR="$STAGE3_DIR/models"
export LOG_ROOT="$PAPER_ROOT/logs/arch_sweeps"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PAPER_ROOT/.mplcache}"

TRAIN_NUM_THREADS="${TRAIN_NUM_THREADS:-1}"
export BLIS_NUM_THREADS="$TRAIN_NUM_THREADS"
export GOTO_NUM_THREADS="$TRAIN_NUM_THREADS"
export MKL_NUM_THREADS="$TRAIN_NUM_THREADS"
export OMP_NUM_THREADS="$TRAIN_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$TRAIN_NUM_THREADS"

SWEEP_SMOKE_TEST="${SWEEP_SMOKE_TEST:-0}"
SWEEP_EXECUTION="${SWEEP_EXECUTION:-sequential}"
family="${1:-all}"

case "$family" in
  all|case1|case2|case2_np10|case2_np20|case3|data_driven|pod_ae|pod_dl) ;;
  *)
    echo "Usage: $0 [all|case1|case2|case2_np10|case2_np20|case3|data_driven|pod_ae|pod_dl]" >&2
    exit 2
    ;;
esac
case "$SWEEP_EXECUTION" in
  sequential|parallel) ;;
  *)
    echo "SWEEP_EXECUTION must be sequential or parallel." >&2
    exit 2
    ;;
esac

mkdir -p "$MODELS_DIR" "$LOG_ROOT" "$MPLCONFIGDIR"

python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

import numpy as np

project = Path.cwd().resolve()
dataset = Path(os.environ["DATASET_DIR"]).expanduser().resolve()
meta_path = dataset / "meta.json"
if not meta_path.is_file():
    raise SystemExit(f"Missing enrichment metadata: {meta_path}")
meta = json.loads(meta_path.read_text())

def localize(path_like):
    path = Path(path_like).expanduser()
    if path.exists():
        return path.resolve()
    marker = "/Project_YvonMaday/"
    text = str(path)
    if marker in text:
        candidate = project / text.split(marker, 1)[1]
        if candidate.exists():
            return candidate.resolve()
    return path

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

ecsw = localize(meta["ecsw_weights_path"])
if not ecsw.is_file():
    raise SystemExit(f"Cannot resolve the fixed baseline linear ECSW file: {ecsw}")
if meta.get("solve_backend") != "hprom":
    raise SystemExit("Expected solve_backend=hprom.")
if meta.get("num_traj") != 29:
    raise SystemExit(f"Expected 29 trajectories, got {meta.get('num_traj')}.")
if meta.get("num_base_traj_copied") != 9 or meta.get("num_lhs_traj") != 20:
    raise SystemExit("Expected the complete 9+20 enrichment dataset.")
if meta.get("coefficient_storage") != "direct_solver_qN_only":
    raise SystemExit("Expected direct solver-side qN targets.")
if meta.get("ecsw_weights_copied") is not False or meta.get("ecsw_weights_rebuilt") is not False:
    raise SystemExit("The dataset does not have strict linear-ECSW reuse provenance.")
if sha256(ecsw) != meta.get("ecsw_weights_sha256"):
    raise SystemExit("The fixed baseline linear ECSW checksum does not match metadata.")

mu_dirs = sorted(path for path in (dataset / "per_mu").iterdir() if path.is_dir())
if len(mu_dirs) != 29:
    raise SystemExit(f"Expected 29 per_mu directories, found {len(mu_dirs)}.")
for mu_dir in mu_dirs:
    qn = np.load(mu_dir / "qN.npy", allow_pickle=False)
    if qn.shape != (151, 501) or not np.all(np.isfinite(qn)):
        raise SystemExit(f"Invalid qN in {mu_dir}: shape={qn.shape}.")

print("[enrichment-sweep-check] dataset:", dataset)
print("[enrichment-sweep-check] trajectories:", len(mu_dirs), "(29 x 501 = 14529 rows)")
print("[enrichment-sweep-check] direct qN:", meta["coefficient_storage"])
print("[enrichment-sweep-check] fixed linear ECSW:", ecsw)
print("[enrichment-sweep-check] ECSW SHA-256:", meta["ecsw_weights_sha256"])
PY

if [[ "$SWEEP_SMOKE_TEST" == "1" ]]; then
  echo "[enrichment-sweep] SMOKE TEST: one representative candidate and one epoch per family."
fi

effective_epochs() {
  if [[ "$SWEEP_SMOKE_TEST" == "1" ]]; then echo 1; else echo "$1"; fi
}

effective_patience() {
  if [[ "$SWEEP_SMOKE_TEST" == "1" ]]; then echo 1; else echo "$1"; fi
}

effective_pretrain() {
  if [[ "$SWEEP_SMOKE_TEST" == "1" ]]; then echo 0; else echo "$1"; fi
}

finalize_sweep() {
  local log_dir="$1"
  local summary_glob="$2"
  local label_prefix="$3"
  local canonical_model="$4"
  local canonical_summary="$5"
  local canonical_log="$6"
  local ranking_csv="$7"
  local expected_count="$8"

  export FINAL_LOG_DIR="$log_dir"
  export FINAL_SUMMARY_GLOB="$summary_glob"
  export FINAL_LABEL_PREFIX="$label_prefix"
  export FINAL_CANONICAL_MODEL="$canonical_model"
  export FINAL_CANONICAL_SUMMARY="$canonical_summary"
  export FINAL_CANONICAL_LOG="$canonical_log"
  export FINAL_RANKING_CSV="$ranking_csv"
  export FINAL_EXPECTED_COUNT="$expected_count"

  python3 - <<'PY'
from pathlib import Path
import csv
import os
import shutil

stage3 = Path(os.environ["STAGE3_DIR"])
models = Path(os.environ["MODELS_DIR"])
log_dir = Path(os.environ["FINAL_LOG_DIR"])
summary_glob = os.environ["FINAL_SUMMARY_GLOB"]
label_prefix = os.environ["FINAL_LABEL_PREFIX"]
canonical_model = Path(os.environ["FINAL_CANONICAL_MODEL"])
canonical_summary = Path(os.environ["FINAL_CANONICAL_SUMMARY"])
canonical_log = Path(os.environ["FINAL_CANONICAL_LOG"])
ranking_csv = Path(os.environ["FINAL_RANKING_CSV"])
expected_count = int(os.environ["FINAL_EXPECTED_COUNT"])

rows = []
for summary in sorted(stage3.glob(summary_glob)):
    data = {}
    for line in summary.read_text(errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    label = summary.name
    if not label.startswith(label_prefix) or not label.endswith("_summary.txt"):
        continue
    label = label[len(label_prefix):-len("_summary.txt")]
    model_name = data.get("model_name", "")
    model_path = models / model_name
    rows.append(
        {
            "label": label,
            "val_rel_frob_percent": data.get("val_rel_frob_percent", ""),
            "train_rel_frob_percent": data.get("train_rel_frob_percent", ""),
            "best_val_mse": data.get("best_val_mse", ""),
            "best_val_total": data.get("best_val_total", ""),
            "epochs_ran": data.get("epochs_ran", ""),
            "primary_modes": data.get("primary_modes", ""),
            "secondary_modes": data.get("secondary_modes", ""),
            "latent_dim": data.get("latent_dim", ""),
            "hidden_dims": data.get("hidden_dims", ""),
            "encoder_hidden_dims": data.get("encoder_hidden_dims", ""),
            "decoder_hidden_dims": data.get("decoder_hidden_dims", ""),
            "dynamics_hidden_dims": data.get("dynamics_hidden_dims", ""),
            "activation": data.get("activation", ""),
            "scaling": data.get("scaling", ""),
            "x_scaling": data.get("x_scaling", ""),
            "q_scaling": data.get("q_scaling", ""),
            "batch_size": data.get("batch_size", ""),
            "lr": data.get("lr", ""),
            "weight_decay": data.get("weight_decay", ""),
            "dropout": data.get("dropout", ""),
            "trainable_parameters": data.get("trainable_parameters", ""),
            "_summary": summary,
            "_model": model_path,
        }
    )

if len(rows) != expected_count:
    raise SystemExit(
        f"Expected {expected_count} candidate summaries matching {summary_glob}, found {len(rows)}."
    )

def score(row):
    for key in ("val_rel_frob_percent", "best_val_mse", "best_val_total"):
        try:
            return float(row[key])
        except (TypeError, ValueError):
            pass
    return float("inf")

rows.sort(key=score)
winner = rows[0]
if not winner["_model"].is_file():
    raise SystemExit(f"Missing winner checkpoint: {winner['_model']}")

ranking_csv.parent.mkdir(parents=True, exist_ok=True)
fields = [key for key in rows[0] if not key.startswith("_")]
with ranking_csv.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows([{key: row[key] for key in fields} for row in rows])

canonical_model.parent.mkdir(parents=True, exist_ok=True)
canonical_summary.parent.mkdir(parents=True, exist_ok=True)
canonical_model.unlink(missing_ok=True)
canonical_summary.unlink(missing_ok=True)
canonical_log.unlink(missing_ok=True)
shutil.move(winner["_model"], canonical_model)
shutil.move(winner["_summary"], canonical_summary)

winner_log = log_dir / f"{winner['label']}.log"
if winner_log.is_file():
    shutil.move(winner_log, canonical_log)

text = canonical_summary.read_text(errors="replace")
lines = []
for line in text.splitlines():
    if line.startswith("model_name:"):
        line = f"model_name: {canonical_model.name}"
    elif line.startswith("model_path:"):
        line = f"model_path: {canonical_model}"
    lines.append(line)
lines.extend(
    [
        "",
        f"sweep_winner_label: {winner['label']}",
        f"sweep_candidates: {len(rows)}",
        "sweep_selection_metric: val_rel_frob_percent",
        "enrichment_protocol: baseline_9_plus_lhs_20_fixed_linear_hprom",
        "architecture_policy: enriched_dataset_sweep",
    ]
)
canonical_summary.write_text("\n".join(lines) + "\n")

for row in rows[1:]:
    row["_model"].unlink(missing_ok=True)
    row["_summary"].unlink(missing_ok=True)
for path in log_dir.glob("*.log"):
    if path != canonical_log:
        path.unlink()

print("[sweep-summary]", ranking_csv)
for row in rows:
    print(
        " ",
        row["label"],
        "val_rel_frob_percent=",
        row["val_rel_frob_percent"],
        "best_val_mse=",
        row["best_val_mse"],
        "best_val_total=",
        row["best_val_total"],
    )
print("[sweep-winner]", winner["label"])
print("[sweep-model]", canonical_model)
print("[sweep-summary-file]", canonical_summary)
PY
}

sweep_case1() (
  local log_dir="$LOG_ROOT/case1"
  mkdir -p "$log_dir"
  cleanup() {
    rm -f "$MODELS_DIR"/case1_ann_ntot151_C*.pt \
      "$STAGE3_DIR"/case1_ann_ntot151_C*_summary.txt \
      "$log_dir"/C*.log
  }
  trap 'status=$?; if [[ $status -ne 0 ]]; then cleanup; fi; exit $status' EXIT
  cleanup
  rm -f "$MODELS_DIR/case1_ann_ntot151_best.pt" \
    "$STAGE3_DIR/case1_ann_ntot151_best_summary.txt" \
    "$log_dir/case1_ann_ntot151_best.log" \
    "$log_dir/case1_arch_sweep_summary.csv"

  run_cfg() {
    local label="$1" hidden="$2" activation="$3" batch="$4" lr="$5" epochs="$6" patience="$7"
    epochs="$(effective_epochs "$epochs")"
    patience="$(effective_patience "$patience")"
    echo "==== Enriched Case 1 sweep: $label"
    python3 -u stage3_perform_training_case_1_ann_maday.py \
      --maday-results-root "$PAPER_RESULTS_ROOT" --maday-tag "$PAPER_TAG" \
      --dataset-backend hprom --dataset-ntot 151 --dataset-dir "$DATASET_DIR" \
      --primary-modes 10 \
      --model-name "case1_ann_ntot151_${label}.pt" \
      --summary-name "case1_ann_ntot151_${label}_summary.txt" \
      --seed 42 --val-frac 0.1 \
      --hidden-dims "$hidden" --activation "$activation" \
      --batch-size "$batch" --lr "$lr" --weight-decay 1e-6 --dropout 0.0 \
      --epochs "$epochs" --patience "$patience" \
      --lr-scheduler-factor 0.5 --lr-scheduler-patience 50 --lr-scheduler-min-lr 1e-6 \
      2>&1 | tee "$log_dir/${label}.log"
  }

  if [[ "$SWEEP_SMOKE_TEST" == "1" ]]; then
    run_cfg C02_wide_silu "256,512,512,256" silu 128 5e-4 6000 220
    expected=1
  else
    run_cfg C00_legacy_elu  "32,64,128,256,256" elu 64 1e-3 5000 180
    run_cfg C01_medium_silu "128,256,256,128" silu 128 5e-4 6000 220
    run_cfg C02_wide_silu   "256,512,512,256" silu 128 5e-4 6000 220
    run_cfg C03_wide_elu    "256,512,512,256" elu 128 5e-4 6000 220
    expected=4
  fi

  finalize_sweep "$log_dir" "case1_ann_ntot151_C*_summary.txt" \
    "case1_ann_ntot151_" \
    "$MODELS_DIR/case1_ann_ntot151_best.pt" \
    "$STAGE3_DIR/case1_ann_ntot151_best_summary.txt" \
    "$log_dir/case1_ann_ntot151_best.log" \
    "$log_dir/case1_arch_sweep_summary.csv" "$expected"
  trap - EXIT
)

sweep_case2_primary() (
  local primary="$1"
  local log_dir="$LOG_ROOT/case2_np${primary}"
  mkdir -p "$log_dir"
  cleanup() {
    rm -f "$MODELS_DIR"/case2_ann_ntot151_np${primary}_B*.pt \
      "$STAGE3_DIR"/case2_ann_ntot151_np${primary}_B*_summary.txt \
      "$log_dir"/np${primary}_B*.log
  }
  trap 'status=$?; if [[ $status -ne 0 ]]; then cleanup; fi; exit $status' EXIT
  cleanup
  rm -f "$MODELS_DIR/case2_ann_ntot151_np${primary}_best.pt" \
    "$STAGE3_DIR/case2_ann_ntot151_np${primary}_best_summary.txt" \
    "$log_dir/case2_ann_ntot151_np${primary}_best.log" \
    "$log_dir/case2_np${primary}_arch_sweep_summary.csv"

  run_cfg() {
    local label="$1" hidden="$2" activation="$3" batch="$4" lr="$5" wd="$6"
    local dropout="$7" epochs="$8" patience="$9"
    local full_label="np${primary}_${label}"
    epochs="$(effective_epochs "$epochs")"
    patience="$(effective_patience "$patience")"
    echo "==== Enriched Case 2 sweep: $full_label"
    python3 -u stage3_perform_training_case_2_ann_test_n20_maday.py \
      --maday-results-root "$PAPER_RESULTS_ROOT" --maday-tag "$PAPER_TAG" \
      --dataset-backend hprom --dataset-ntot 151 --dataset-dir "$DATASET_DIR" \
      --primary-modes "$primary" \
      --model-name "case2_ann_ntot151_${full_label}.pt" \
      --summary-name "case2_ann_ntot151_${full_label}_summary.txt" \
      --val-split-mode row --val-frac 0.1 \
      --hidden-dims "$hidden" --activation "$activation" \
      --batch-size "$batch" --lr "$lr" --weight-decay "$wd" --dropout "$dropout" \
      --epochs "$epochs" --patience "$patience" \
      --lr-scheduler-factor 0.5 --lr-scheduler-patience 50 --lr-scheduler-min-lr 1e-6 \
      --seed 42 2>&1 | tee "$log_dir/${full_label}.log"
  }

  if [[ "$SWEEP_SMOKE_TEST" == "1" ]]; then
    run_cfg B01_A10_like_b128_lr5e4 "256,512,512,256" silu 128 5e-4 1e-6 0.00 6000 220
    expected=1
  else
    run_cfg B00_current_b128_lr1e3      "32,64,128,256,256" elu 128 1e-3 1e-6 0.00 5000 160
    run_cfg B01_A10_like_b128_lr5e4    "256,512,512,256" silu 128 5e-4 1e-6 0.00 6000 220
    run_cfg B02_A10_elu_b128_lr5e4     "256,512,512,256" elu 128 5e-4 1e-6 0.00 6000 220
    run_cfg B03_medium_silu_b128_lr1e3 "128,256,256,128" silu 128 1e-3 1e-6 0.00 5000 180
    run_cfg B04_deep_silu_b128_lr5e4   "128,256,512,512,256,128" silu 128 5e-4 1e-6 0.00 6000 240
    expected=5
  fi

  finalize_sweep "$log_dir" "case2_ann_ntot151_np${primary}_B*_summary.txt" \
    "case2_ann_ntot151_np${primary}_" \
    "$MODELS_DIR/case2_ann_ntot151_np${primary}_best.pt" \
    "$STAGE3_DIR/case2_ann_ntot151_np${primary}_best_summary.txt" \
    "$log_dir/case2_ann_ntot151_np${primary}_best.log" \
    "$log_dir/case2_np${primary}_arch_sweep_summary.csv" "$expected"
  trap - EXIT
)

sweep_case3() (
  local log_dir="$LOG_ROOT/case3"
  mkdir -p "$log_dir"
  cleanup() {
    rm -f "$MODELS_DIR"/case3_ann_ntot151_C*.pt \
      "$STAGE3_DIR"/case3_ann_ntot151_C*_summary.txt \
      "$log_dir"/C*.log
  }
  trap 'status=$?; if [[ $status -ne 0 ]]; then cleanup; fi; exit $status' EXIT
  cleanup
  rm -f "$MODELS_DIR/case3_ann_ntot151_best.pt" \
    "$STAGE3_DIR/case3_ann_ntot151_best_summary.txt" \
    "$log_dir/case3_ann_ntot151_best.log" \
    "$log_dir/case3_arch_sweep_summary.csv"

  run_cfg() {
    local label="$1" hidden="$2" activation="$3" batch="$4" lr="$5" epochs="$6" patience="$7"
    epochs="$(effective_epochs "$epochs")"
    patience="$(effective_patience "$patience")"
    echo "==== Enriched Case 3 sweep: $label"
    python3 -u stage3_perform_training_case_3_ann_maday.py \
      --maday-results-root "$PAPER_RESULTS_ROOT" --maday-tag "$PAPER_TAG" \
      --dataset-backend hprom --dataset-ntot 151 --dataset-dir "$DATASET_DIR" \
      --primary-modes 10 \
      --model-name "case3_ann_ntot151_${label}.pt" \
      --summary-name "case3_ann_ntot151_${label}_summary.txt" \
      --seed 42 --val-frac 0.1 \
      --hidden-dims "$hidden" --activation "$activation" \
      --batch-size "$batch" --lr "$lr" --weight-decay 1e-6 --dropout 0.0 \
      --epochs "$epochs" --patience "$patience" \
      --lr-scheduler-factor 0.5 --lr-scheduler-patience 50 --lr-scheduler-min-lr 1e-6 \
      2>&1 | tee "$log_dir/${label}.log"
  }

  if [[ "$SWEEP_SMOKE_TEST" == "1" ]]; then
    run_cfg C02_wide_silu "256,512,512,256" silu 128 5e-4 6000 220
    expected=1
  else
    run_cfg C00_legacy_elu  "32,64,128,256,256" elu 128 1e-3 5000 180
    run_cfg C01_medium_silu "128,256,256,128" silu 128 5e-4 6000 220
    run_cfg C02_wide_silu   "256,512,512,256" silu 128 5e-4 6000 220
    run_cfg C03_wide_elu    "256,512,512,256" elu 128 5e-4 6000 220
    expected=4
  fi

  finalize_sweep "$log_dir" "case3_ann_ntot151_C*_summary.txt" \
    "case3_ann_ntot151_" \
    "$MODELS_DIR/case3_ann_ntot151_best.pt" \
    "$STAGE3_DIR/case3_ann_ntot151_best_summary.txt" \
    "$log_dir/case3_ann_ntot151_best.log" \
    "$log_dir/case3_arch_sweep_summary.csv" "$expected"
  trap - EXIT
)

sweep_data_driven() (
  local log_dir="$LOG_ROOT/data_driven"
  mkdir -p "$log_dir"
  cleanup() {
    rm -f "$MODELS_DIR"/rom_data_driven_ann_mu_t_ntot151_A*.pt \
      "$STAGE3_DIR"/rom_data_driven_ann_mu_t_ntot151_A*_summary.txt \
      "$log_dir"/A*.log
  }
  trap 'status=$?; if [[ $status -ne 0 ]]; then cleanup; fi; exit $status' EXIT
  cleanup
  rm -f "$MODELS_DIR/data_driven_ann_ntot151_best.pt" \
    "$STAGE3_DIR/data_driven_ann_ntot151_best_summary.txt" \
    "$log_dir/data_driven_ann_ntot151_best.log" \
    "$log_dir/data_driven_arch_sweep_summary.csv"

  run_cfg() {
    local label="$1" hidden="$2" activation="$3" batch="$4" lr="$5" wd="$6"
    local dropout="$7" epochs="$8" patience="$9"
    epochs="$(effective_epochs "$epochs")"
    patience="$(effective_patience "$patience")"
    echo "==== Enriched POD-NN sweep: $label"
    python3 -u stage3_perform_training_rom_data_driven_maday.py \
      --maday-results-root "$PAPER_RESULTS_ROOT" --maday-tag "$PAPER_TAG" \
      --dataset-backend hprom --dataset-ntot 151 --dataset-dir "$DATASET_DIR" \
      --model-name "rom_data_driven_ann_mu_t_ntot151_${label}.pt" \
      --summary-name "rom_data_driven_ann_mu_t_ntot151_${label}_summary.txt" \
      --hidden-dims "$hidden" --activation "$activation" \
      --batch-size "$batch" --lr "$lr" --weight-decay "$wd" --dropout "$dropout" \
      --epochs "$epochs" --patience "$patience" \
      --lr-scheduler-factor 0.5 --lr-scheduler-patience 50 --lr-scheduler-min-lr 1e-6 \
      --seed 42 2>&1 | tee "$log_dir/${label}.log"
  }

  if [[ "$SWEEP_SMOKE_TEST" == "1" ]]; then
    run_cfg A10_silu_wide_b128_lr5e4 "256,512,512,256" silu 128 5e-4 1e-6 0.00 6000 220
    expected=1
  else
    run_cfg A00_current_b128_lr1e3       "32,64,128,256,256" elu 128 1e-3 1e-6 0.00 5000 160
    run_cfg A01_compact64_b64_lr1e3      "64,64,64,64" elu 64 1e-3 1e-6 0.00 5000 160
    run_cfg A02_compact128_b64_lr1e3     "128,128,128" elu 64 1e-3 1e-6 0.00 5000 160
    run_cfg A03_bottleneck_b64_lr1e3     "64,128,256,128,64" elu 64 1e-3 1e-6 0.00 5000 180
    run_cfg A04_medium_b128_lr1e3        "128,256,256,128" elu 128 1e-3 1e-6 0.00 5000 180
    run_cfg A05_wide_b128_lr5e4          "256,512,512,256" elu 128 5e-4 1e-6 0.00 6000 220
    run_cfg A06_wide_b256_lr5e4          "256,512,512,256" elu 256 5e-4 1e-6 0.00 6000 220
    run_cfg A07_deep_wide_b128_lr5e4     "128,256,512,512,256,128" elu 128 5e-4 1e-6 0.00 6000 240
    run_cfg A08_deep_wide_reg_b128_lr5e4 "128,256,512,512,256,128" elu 128 5e-4 1e-5 0.02 6000 240
    run_cfg A09_silu_medium_b128_lr1e3   "128,256,256,128" silu 128 1e-3 1e-6 0.00 5000 180
    run_cfg A10_silu_wide_b128_lr5e4     "256,512,512,256" silu 128 5e-4 1e-6 0.00 6000 220
    run_cfg A11_small_b64_lr1e3          "32,64,128" elu 64 1e-3 1e-6 0.00 5000 160
    expected=12
  fi

  finalize_sweep "$log_dir" "rom_data_driven_ann_mu_t_ntot151_A*_summary.txt" \
    "rom_data_driven_ann_mu_t_ntot151_" \
    "$MODELS_DIR/data_driven_ann_ntot151_best.pt" \
    "$STAGE3_DIR/data_driven_ann_ntot151_best_summary.txt" \
    "$log_dir/data_driven_ann_ntot151_best.log" \
    "$log_dir/data_driven_arch_sweep_summary.csv" "$expected"
  trap - EXIT
)

sweep_pod_ae() (
  local log_dir="$LOG_ROOT/pod_ae"
  mkdir -p "$log_dir"
  cleanup() {
    rm -f "$MODELS_DIR"/prom_pod_ae_ntot151_PAE*.pt \
      "$STAGE3_DIR"/prom_pod_ae_ntot151_PAE*_summary.txt \
      "$log_dir"/PAE*.log
  }
  trap 'status=$?; if [[ $status -ne 0 ]]; then cleanup; fi; exit $status' EXIT
  cleanup
  rm -f "$MODELS_DIR/prom_pod_ae_ntot151_best.pt" \
    "$STAGE3_DIR/prom_pod_ae_ntot151_best_summary.txt" \
    "$log_dir/prom_pod_ae_ntot151_best.log" \
    "$log_dir/pod_ae_arch_sweep_summary.csv"

  run_cfg() {
    local label="$1" latent="$2" hidden="$3" activation="$4" scaling="$5"
    local batch="$6" lr="$7" epochs="$8" patience="$9"
    epochs="$(effective_epochs "$epochs")"
    patience="$(effective_patience "$patience")"
    echo "==== Enriched PROM-POD-AE sweep: $label"
    python3 -u stage3_perform_training_prom_pod_ae.py \
      --dataset-backend hprom --dataset-ntot 151 --dataset-dir "$DATASET_DIR" \
      --stage3-dir "$STAGE3_DIR" --models-dir "$MODELS_DIR" \
      --model-name "prom_pod_ae_ntot151_${label}.pt" \
      --summary-name "prom_pod_ae_ntot151_${label}_summary.txt" \
      --seed 42 --val-frac 0.1 \
      --latent-dim "$latent" --hidden-dims "$hidden" \
      --activation "$activation" --scaling "$scaling" \
      --batch-size "$batch" --lr "$lr" --weight-decay 1e-6 \
      --epochs "$epochs" --patience "$patience" \
      --lr-scheduler-factor 0.5 --lr-scheduler-patience 50 --lr-scheduler-min-lr 1e-6 \
      2>&1 | tee "$log_dir/${label}.log"
  }

  if [[ "$SWEEP_SMOKE_TEST" == "1" ]]; then
    run_cfg PAE06_l10_gelu_zscore_wide 10 "512,256,128" gelu zscore 128 5e-4 6500 300
    expected=1
  else
    run_cfg PAE00_l10_tanh_minmax_small  10 "192,96,48" tanh minmax_-1_1 256 1e-3 4500 220
    run_cfg PAE01_l10_silu_zscore_medium 10 "256,128,64" silu zscore 128 5e-4 5500 260
    run_cfg PAE02_l10_elu_zscore_medium  10 "256,128,64" elu zscore 128 5e-4 5500 260
    run_cfg PAE03_l10_gelu_zscore_medium 10 "256,128,64" gelu zscore 128 5e-4 5500 260
    run_cfg PAE04_l10_silu_zscore_wide   10 "512,256,128" silu zscore 128 5e-4 6500 300
    run_cfg PAE05_l10_elu_zscore_wide    10 "512,256,128" elu zscore 128 5e-4 6500 300
    run_cfg PAE06_l10_gelu_zscore_wide   10 "512,256,128" gelu zscore 128 5e-4 6500 300
    run_cfg PAE07_l10_silu_minmax_wide   10 "512,256,128" silu minmax_-1_1 128 5e-4 6500 300
    expected=8
  fi

  finalize_sweep "$log_dir" "prom_pod_ae_ntot151_PAE*_summary.txt" \
    "prom_pod_ae_ntot151_" \
    "$MODELS_DIR/prom_pod_ae_ntot151_best.pt" \
    "$STAGE3_DIR/prom_pod_ae_ntot151_best_summary.txt" \
    "$log_dir/prom_pod_ae_ntot151_best.log" \
    "$log_dir/pod_ae_arch_sweep_summary.csv" "$expected"
  trap - EXIT
)

sweep_pod_dl() (
  local log_dir="$LOG_ROOT/pod_dl"
  mkdir -p "$log_dir"
  cleanup() {
    rm -f "$MODELS_DIR"/pod_dl_data_driven_ntot151_PDL*.pt \
      "$STAGE3_DIR"/pod_dl_data_driven_ntot151_PDL*_summary.txt \
      "$log_dir"/PDL*.log
  }
  trap 'status=$?; if [[ $status -ne 0 ]]; then cleanup; fi; exit $status' EXIT
  cleanup
  rm -f "$MODELS_DIR/pod_dl_data_driven_ntot151_best.pt" \
    "$STAGE3_DIR/pod_dl_data_driven_ntot151_best_summary.txt" \
    "$log_dir/pod_dl_data_driven_ntot151_best.log" \
    "$log_dir/pod_dl_arch_sweep_summary.csv"

  run_cfg() {
    local label="$1" latent="$2" enc="$3" dec="$4" dyn="$5" activation="$6"
    local x_scaling="$7" q_scaling="$8" omega_latent="$9" omega_recon="${10}"
    local pretrain="${11}" batch="${12}" lr="${13}" epochs="${14}" patience="${15}"
    epochs="$(effective_epochs "$epochs")"
    patience="$(effective_patience "$patience")"
    pretrain="$(effective_pretrain "$pretrain")"
    echo "==== Enriched POD-DL sweep: $label"
    python3 -u stage3_perform_training_pod_dl_data_driven.py \
      --dataset-backend hprom --dataset-ntot 151 --dataset-dir "$DATASET_DIR" \
      --stage3-dir "$STAGE3_DIR" --models-dir "$MODELS_DIR" \
      --model-name "pod_dl_data_driven_ntot151_${label}.pt" \
      --summary-name "pod_dl_data_driven_ntot151_${label}_summary.txt" \
      --seed 42 --val-frac 0.1 \
      --latent-dim "$latent" \
      --encoder-hidden-dims "$enc" --decoder-hidden-dims "$dec" \
      --dynamics-hidden-dims "$dyn" --activation "$activation" \
      --x-scaling "$x_scaling" --q-scaling "$q_scaling" \
      --omega-data 1.0 --omega-latent "$omega_latent" --omega-recon "$omega_recon" \
      --pretrain-epochs "$pretrain" \
      --batch-size "$batch" --lr "$lr" --weight-decay 1e-6 \
      --epochs "$epochs" --patience "$patience" \
      --lr-scheduler-factor 0.5 --lr-scheduler-patience 50 --lr-scheduler-min-lr 1e-6 \
      2>&1 | tee "$log_dir/${label}.log"
  }

  if [[ "$SWEEP_SMOKE_TEST" == "1" ]]; then
    run_cfg PDL07_l10_silu_lowlatreg 10 "512,256" "256,512" "256,512,512,256" \
      silu zscore zscore 0.03 0.01 300 128 5e-4 7000 320
    expected=1
  else
    run_cfg PDL00_l10_legacy_elu      10 "256,128" "128,256" "64,128,128" \
      elu zscore zscore 0.10 0.00 0 256 1e-3 5000 220
    run_cfg PDL01_l10_silu_balanced   10 "256,128" "128,256" "256,512,256" \
      silu zscore zscore 0.05 0.01 250 128 5e-4 6000 260
    run_cfg PDL02_l10_elu_balanced    10 "256,128" "128,256" "256,512,256" \
      elu zscore zscore 0.05 0.01 250 128 5e-4 6000 260
    run_cfg PDL03_l10_silu_wide       10 "512,256" "256,512" "256,512,512,256" \
      silu zscore zscore 0.05 0.01 300 128 5e-4 7000 320
    run_cfg PDL04_l10_elu_wide        10 "512,256" "256,512" "256,512,512,256" \
      elu zscore zscore 0.05 0.01 300 128 5e-4 7000 320
    run_cfg PDL05_l10_gelu_wide       10 "512,256" "256,512" "256,512,512,256" \
      gelu zscore zscore 0.05 0.01 300 128 5e-4 7000 320
    run_cfg PDL06_l10_silu_paper_loss 10 "512,256" "256,512" "256,512,512,256" \
      silu zscore zscore 0.10 0.00 0 128 5e-4 7000 320
    run_cfg PDL07_l10_silu_lowlatreg  10 "512,256" "256,512" "256,512,512,256" \
      silu zscore zscore 0.03 0.01 300 128 5e-4 7000 320
    expected=8
  fi

  finalize_sweep "$log_dir" "pod_dl_data_driven_ntot151_PDL*_summary.txt" \
    "pod_dl_data_driven_ntot151_" \
    "$MODELS_DIR/pod_dl_data_driven_ntot151_best.pt" \
    "$STAGE3_DIR/pod_dl_data_driven_ntot151_best_summary.txt" \
    "$log_dir/pod_dl_data_driven_ntot151_best.log" \
    "$log_dir/pod_dl_arch_sweep_summary.csv" "$expected"
  trap - EXIT
)

run_requested_family() {
  case "$1" in
    case1) sweep_case1 ;;
    case2) sweep_case2_primary 10; sweep_case2_primary 20 ;;
    case2_np10) sweep_case2_primary 10 ;;
    case2_np20) sweep_case2_primary 20 ;;
    case3) sweep_case3 ;;
    data_driven) sweep_data_driven ;;
    pod_ae) sweep_pod_ae ;;
    pod_dl) sweep_pod_dl ;;
  esac
}

if [[ "$family" == "all" ]]; then
  families=(case1 case2_np10 case2_np20 case3 data_driven pod_ae pod_dl)
  if [[ "$SWEEP_EXECUTION" == "parallel" ]]; then
    echo "[enrichment-sweep] Running seven sweeps concurrently."
    echo "[enrichment-sweep] Use this only for CPU training with sufficient memory."
    pids=()
    for requested in "${families[@]}"; do
      run_requested_family "$requested" &
      pids+=("$!")
    done
    status=0
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        status=1
      fi
    done
    if [[ "$status" -ne 0 ]]; then
      echo "[error] At least one enrichment sweep failed." >&2
      exit "$status"
    fi
  else
    echo "[enrichment-sweep] Running all seven sweeps sequentially."
    for requested in "${families[@]}"; do
      run_requested_family "$requested"
    done
  fi
else
  run_requested_family "$family"
fi

echo "[done] Enrichment architecture sweep request: $family"
echo "[done] Winner checkpoints: $MODELS_DIR"
find "$MODELS_DIR" -maxdepth 1 -type f -name '*_best.pt' -printf '  %f\n' | sort
