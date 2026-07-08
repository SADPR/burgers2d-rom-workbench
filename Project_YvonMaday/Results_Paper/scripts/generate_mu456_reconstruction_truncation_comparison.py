#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pure reconstruction truncation diagnostic for mu=(4.560, 0.0190).

Compares HDM against reconstructions

    u_ref + V[:, :m] q[:m, :]

for the linear PROM coefficients and for data-driven q_tot predictions.
The output directory is recreated on each run to avoid stale figures/tables.
"""

from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg_mu456_recon_trunc")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent

MU_LABEL = "mu1_4.560_mu2_0.0190"
MU_TITLE = r"$\mu=(4.560,0.0190)$"

BASIS = ROOT / "MetricStudy" / "lspg_sensitive" / "Stage1" / "basis.npy"
U_REF = ROOT / "MetricStudy" / "lspg_sensitive" / "Stage1" / "u_ref.npy"
HDM = PROJECT / "Results" / "param_snaps" / "mu1_4.56+mu2_0.019.npy"

TRUE_QN = ROOT / "mlspg_prom_main" / "Runs" / "Linear" / f"linear_prom_{MU_LABEL}_ntot151" / "qN.npy"
NON_ENRICHED_QN = (
    ROOT
    / "mlspg_prom_main"
    / "Runs"
    / "DataDriven_MasterANN"
    / f"rom_data_driven_{MU_LABEL}_ntot151"
    / "qN.npy"
)

ENRICHED_CANDIDATES = [
    ROOT
    / "mlspg_prom_enriched_main"
    / "Runs"
    / "DataDriven_MasterANN"
    / f"rom_data_driven_{MU_LABEL}_ntot151"
    / "qN.npy",
    ROOT
    / "mlspg_prom_enrichment"
    / "Runs"
    / "DataDriven_MasterANN"
    / f"rom_data_driven_{MU_LABEL}_ntot151"
    / "qN.npy",
    ROOT
    / "mlspg_hprom_enrichment_ext25_lhs36"
    / "Runs"
    / "DataDriven_MasterANN"
    / f"rom_data_driven_{MU_LABEL}_ntot151"
    / "qN.npy",
]

MODE_COUNTS = [
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    10,
    12,
    15,
    20,
    25,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    120,
    140,
    151,
]
OUT_DIR = ROOT / "Prom_MasterANN_Diagnostic" / "mu456_reconstruction_truncation"


def _load_q(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    q = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if q.ndim != 2:
        raise ValueError(f"Expected 2D q array, got {q.shape} from {path}")
    if q.shape == (501, 151):
        q = q.T
    if q.shape[0] != 151:
        raise ValueError(f"Expected q shape (151, nt), got {q.shape} from {path}")
    return q


def _load_matrix(path: Path, name: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    return np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)


def _rel_state_error(reference: np.ndarray, approx: np.ndarray) -> float:
    return 100.0 * float(np.linalg.norm(reference - approx) / np.linalg.norm(reference))


def _reconstruct(v: np.ndarray, u_ref: np.ndarray, q: np.ndarray, m: int) -> np.ndarray:
    return u_ref[:, None] + v[:, :m] @ q[:m, :]


def _find_enriched_qn() -> Path | None:
    env_path = os.environ.get("ENRICHED_QN", "").strip()
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"ENRICHED_QN was set but does not exist: {p}")
        return p
    for p in ENRICHED_CANDIDATES:
        if p.exists():
            return p
    return None


def _make_rows(v: np.ndarray, u_ref: np.ndarray, hdm: np.ndarray, true_q: np.ndarray, models: list[tuple[str, Path, np.ndarray]]):
    rows = []
    true_full = _reconstruct(v, u_ref, true_q, 151)
    linear_full_vs_hdm = _rel_state_error(hdm, true_full)

    for m in MODE_COUNTS:
        true_recon = _reconstruct(v, u_ref, true_q, m)
        rows.append(
            {
                "family": "linear_PROM_coefficients",
                "q_source": str(TRUE_QN),
                "modes": m,
                "state_error_vs_hdm_percent": _rel_state_error(hdm, true_recon),
                "state_error_vs_linear_full_percent": _rel_state_error(true_full, true_recon),
                "extra_error_vs_same_m_linear_percent": 0.0,
                "coefficient_error_first_m_percent": 0.0,
                "linear_full_state_error_vs_hdm_percent": linear_full_vs_hdm,
            }
        )

        for label, source, pred_q in models:
            pred_recon = _reconstruct(v, u_ref, pred_q, m)
            pred_coeff = pred_q[:m, :]
            true_coeff = true_q[:m, :]
            rows.append(
                {
                    "family": label,
                    "q_source": str(source),
                    "modes": m,
                    "state_error_vs_hdm_percent": _rel_state_error(hdm, pred_recon),
                    "state_error_vs_linear_full_percent": _rel_state_error(true_full, pred_recon),
                    "extra_error_vs_same_m_linear_percent": _rel_state_error(true_recon, pred_recon),
                    "coefficient_error_first_m_percent": 100.0
                    * float(np.linalg.norm(pred_coeff - true_coeff) / np.linalg.norm(true_coeff)),
                    "linear_full_state_error_vs_hdm_percent": linear_full_vs_hdm,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, object]], out_dir: Path) -> None:
    families = []
    for r in rows:
        fam = str(r["family"])
        if fam not in families:
            families.append(fam)

    colors = {
        "linear_PROM_coefficients": "black",
        "non_enriched_master_ANN": "#1f77b4",
        "enriched_master_ANN": "#2ca02c",
    }
    markers = {
        "linear_PROM_coefficients": "o",
        "non_enriched_master_ANN": "s",
        "enriched_master_ANN": "^",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))

    ykeys = [
        ("state_error_vs_hdm_percent", r"$\|u_{\rm HDM}-u_m\|/\|u_{\rm HDM}\|$ (%)"),
        ("state_error_vs_linear_full_percent", r"$\|u_{151}^{\rm lin}-u_m\|/\|u_{151}^{\rm lin}\|$ (%)"),
        ("coefficient_error_first_m_percent", r"$\|q_{1:m}^{\rm pred}-q_{1:m}^{\rm lin}\|/\|q_{1:m}^{\rm lin}\|$ (%)"),
    ]

    for ax, (key, ylabel) in zip(axes, ykeys):
        for fam in families:
            rr = [r for r in rows if str(r["family"]) == fam]
            x = np.array([int(r["modes"]) for r in rr], dtype=int)
            y = np.array([float(r[key]) for r in rr], dtype=float)
            ax.plot(
                x,
                y,
                marker=markers.get(fam, "o"),
                color=colors.get(fam, None),
                linewidth=1.8,
                label=fam.replace("_", " "),
            )
        ax.set_xlabel("retained reconstruction modes")
        ax.set_ylabel(ylabel)
        ax.set_xticks(MODE_COUNTS)
        ax.grid(True, alpha=0.28)
        if np.nanmax([float(r[key]) for r in rows]) / max(np.nanmin([float(r[key]) for r in rows if float(r[key]) > 0.0]), 1e-12) > 50:
            ax.set_yscale("log")

    axes[0].set_title(MU_TITLE + ": pure reconstruction")
    axes[0].legend(loc="best", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_dir / "mu456_pure_reconstruction_truncation_comparison.png", dpi=230)
    plt.close(fig)


def _write_summary(path: Path, rows: list[dict[str, object]], enriched_path: Path | None) -> None:
    with path.open("w") as f:
        f.write(f"mu: {MU_TITLE}\n")
        f.write(f"basis: {BASIS}\n")
        f.write(f"u_ref: {U_REF}\n")
        f.write(f"hdm: {HDM}\n")
        f.write(f"linear_qn: {TRUE_QN}\n")
        f.write(f"non_enriched_qn: {NON_ENRICHED_QN}\n")
        f.write(f"enriched_qn: {enriched_path if enriched_path is not None else 'not_found'}\n")
        f.write(f"mode_counts: {MODE_COUNTS}\n\n")
        f.write("rows:\n")
        for r in rows:
            f.write(
                "  "
                f"{r['family']} m={r['modes']}: "
                f"u_vs_hdm={float(r['state_error_vs_hdm_percent']):.6g}% "
                f"u_vs_linear151={float(r['state_error_vs_linear_full_percent']):.6g}% "
                f"q_first_m={float(r['coefficient_error_first_m_percent']):.6g}%\n"
            )


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    v = _load_matrix(BASIS, "basis")
    u_ref = _load_matrix(U_REF, "u_ref").reshape(-1)
    hdm = _load_matrix(HDM, "HDM snapshots")
    true_q = _load_q(TRUE_QN)
    non_enriched_q = _load_q(NON_ENRICHED_QN)

    if hdm.shape[0] != v.shape[0] and hdm.shape[1] == v.shape[0]:
        hdm = hdm.T
    if hdm.shape != (v.shape[0], true_q.shape[1]):
        raise ValueError(f"HDM shape {hdm.shape} incompatible with basis {v.shape} and q {true_q.shape}")

    models = [("non_enriched_master_ANN", NON_ENRICHED_QN, non_enriched_q)]
    enriched_path = _find_enriched_qn()
    if enriched_path is not None:
        models.append(("enriched_master_ANN", enriched_path, _load_q(enriched_path)))

    rows = _make_rows(v, u_ref, hdm, true_q, models)
    _write_csv(OUT_DIR / "mu456_pure_reconstruction_truncation_comparison.csv", rows)
    _plot(rows, OUT_DIR)
    _write_summary(OUT_DIR / "summary.txt", rows, enriched_path)

    print(f"[mu456-recon-trunc] output_dir = {OUT_DIR}")
    print(f"[mu456-recon-trunc] enriched_qn = {enriched_path if enriched_path else 'not_found'}")
    print(f"[mu456-recon-trunc] csv = {OUT_DIR / 'mu456_pure_reconstruction_truncation_comparison.csv'}")
    print(f"[mu456-recon-trunc] figure = {OUT_DIR / 'mu456_pure_reconstruction_truncation_comparison.png'}")


if __name__ == "__main__":
    main()
