#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[run] PROM-POD-AE Best online"
bash "$SCRIPT_DIR/run_mlspg_hprom_pod_ae_best_3pts.sh"

echo "[run] POD-DL-ROM Best online"
bash "$SCRIPT_DIR/run_mlspg_hprom_pod_dl_best_3pts.sh"

echo "[done] Latent model online runs completed."
