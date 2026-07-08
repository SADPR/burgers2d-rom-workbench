#!/usr/bin/env python3
"""Temporary reconstruction-only diagnostic for the baseline MLSPG campaign.

This does not solve PROM/HPROM systems.  It takes the saved baseline linear
HPROM coefficient trajectory q_ref(t), applies each learned nonlinear
representation, reconstructs the state, and reports the relative state error
against HDM.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT = Path(__file__).resolve()
PROJECT = SCRIPT.parents[2]
REPO = PROJECT.parent
for path in (str(PROJECT), str(REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)

from burgers.config import DT, GRID_X, GRID_Y, NUM_STEPS, W0  # noqa: E402
from burgers.core import load_or_compute_snaps  # noqa: E402
from project_layout import write_kv_txt  # noqa: E402
from run_prom_ann_case_1 import _build_case1_full_coordinates, _load_case1_model  # noqa: E402
from run_prom_ann_case_2 import _load_case2_model, _predict_case2_secondary_coords  # noqa: E402
from run_prom_ann_case_3 import _build_case3_full_coordinates, _load_case3_model  # noqa: E402
from run_prom_pod_ae import _load_pod_ae_checkpoint  # noqa: E402


POINTS = (
    ("verification", 4.875, 0.0225),
    ("offgrid1", 4.560, 0.0190),
    ("offgrid2", 5.190, 0.0260),
    ("extrapolation20pct", 4.000, 0.0330),
)


def compact_float(x: float) -> str:
    return str(float(x))


def mu_tag(mu1: float, mu2: float) -> str:
    return f"mu1_{mu1:.3f}_mu2_{mu2:.4f}"


def linear_q_path(hprom_root: Path, mu1: float, mu2: float) -> Path:
    tag = mu_tag(mu1, mu2)
    if abs(mu1 - 4.0) < 1.0e-12 and abs(mu2 - 0.0330) < 1.0e-12:
        return hprom_root / "Runs" / "Extrapolation20pct" / "Linear" / f"linear_hprom_{tag}_ntot151" / "qN.npy"
    return hprom_root / "Runs" / "Linear" / f"linear_hprom_{tag}_ntot151" / "qN.npy"


def find_hdm_path(mu1: float, mu2: float) -> Path | None:
    fname = f"mu1_{compact_float(mu1)}+mu2_{compact_float(mu2)}.npy"
    for root in (
        REPO / "Results" / "param_snaps",
        PROJECT / "Results" / "param_snaps",
        PROJECT / "250x250" / "param_snaps",
    ):
        path = root / fname
        if path.exists():
            return path
    return None


def load_hdm(mu1: float, mu2: float, allow_compute: bool) -> tuple[np.ndarray, Path | str]:
    path = find_hdm_path(mu1, mu2)
    if path is not None:
        return np.load(path, mmap_mode="r", allow_pickle=False), path
    if not allow_compute:
        raise FileNotFoundError(
            f"Missing HDM snapshots for mu=({mu1}, {mu2}). "
            "Set --allow-compute-hdm only if you intentionally want to run HDM."
        )
    snap_folder = PROJECT / "Results" / "param_snaps"
    snap_folder.mkdir(parents=True, exist_ok=True)
    hdm = load_or_compute_snaps(
        mu=[mu1, mu2],
        grid_x=GRID_X,
        grid_y=GRID_Y,
        w0=np.asarray(W0, dtype=np.float64).reshape(-1),
        dt=DT,
        num_steps=NUM_STEPS,
        snap_folder=str(snap_folder),
    )
    return hdm, "computed"


def reconstruct_error(
    V: np.ndarray,
    u_ref: np.ndarray,
    q: np.ndarray,
    hdm: np.ndarray,
    save_path: Path | None,
    chunk_cols: int,
) -> float:
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2:
        raise ValueError(f"q must be 2D, got {q.shape}")
    if q.shape[0] > V.shape[1]:
        raise ValueError(f"q has {q.shape[0]} rows but basis has only {V.shape[1]} columns")
    if hdm.shape[1] != q.shape[1]:
        raise ValueError(f"HDM/q time mismatch: hdm={hdm.shape}, q={q.shape}")

    Vq = np.asarray(V[:, : q.shape[0]], dtype=np.float64)
    u_ref = np.asarray(u_ref, dtype=np.float64).reshape(-1)
    if hdm.shape[0] != u_ref.size:
        raise ValueError(f"HDM/u_ref state mismatch: hdm={hdm.shape}, u_ref={u_ref.shape}")

    writer = None
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        writer = np.lib.format.open_memmap(
            save_path,
            mode="w+",
            dtype=np.float64,
            shape=hdm.shape,
        )

    num = 0.0
    den = 0.0
    for j0 in range(0, q.shape[1], int(chunk_cols)):
        j1 = min(q.shape[1], j0 + int(chunk_cols))
        rec = u_ref[:, None] + Vq @ q[:, j0:j1]
        h = np.asarray(hdm[:, j0:j1], dtype=np.float64)
        d = h - rec
        num += float(np.sum(d * d))
        den += float(np.sum(h * h))
        if writer is not None:
            writer[:, j0:j1] = rec

    if writer is not None:
        writer.flush()
    return 100.0 * float(np.sqrt(num / den))


def q_case1(q_ref: np.ndarray, model_path: Path, device: str) -> tuple[np.ndarray, dict]:
    model, n_p, n_s = _load_case1_model(str(model_path), torch.device(device))
    q = _build_case1_full_coordinates(q_ref[:n_p, :], model, device=device)
    return q, {"primary_modes": n_p, "secondary_modes": n_s}


def q_case2(q_ref: np.ndarray, model_path: Path, device: str) -> tuple[np.ndarray, dict]:
    model, n_s, ckpt = _load_case2_model(str(model_path), torch.device(device))
    n_p = int(ckpt["primary_modes"])
    q_secondary = _predict_case2_secondary_coords(q_ref.shape[1], model, (0.0, 0.0), DT, device=device)
    raise RuntimeError("Internal error: q_case2 requires mu; call q_case2_mu instead.")


def q_case2_mu(q_ref: np.ndarray, model_path: Path, device: str, mu: tuple[float, float]) -> tuple[np.ndarray, dict]:
    model, n_s, ckpt = _load_case2_model(str(model_path), torch.device(device))
    n_p = int(ckpt["primary_modes"])
    q_secondary = _predict_case2_secondary_coords(q_ref.shape[1], model, mu, DT, device=device)
    q = np.vstack((q_ref[:n_p, :], q_secondary))
    return q, {"primary_modes": n_p, "secondary_modes": n_s}


def q_case3(q_ref: np.ndarray, model_path: Path, device: str, mu: tuple[float, float]) -> tuple[np.ndarray, dict]:
    model, _in_dim, n_p, n_s, _ckpt = _load_case3_model(str(model_path), torch.device(device))
    q = _build_case3_full_coordinates(q_ref[:n_p, :], model, mu=mu, dt=DT, device=device)
    return q, {"primary_modes": n_p, "secondary_modes": n_s}


def q_podae(q_ref: np.ndarray, model_path: Path, device: str) -> tuple[np.ndarray, dict]:
    model, q_dim, latent_dim, hidden_dims, scaling, activation, _ckpt = _load_pod_ae_checkpoint(
        str(model_path), torch.device(device)
    )
    with torch.no_grad():
        q_t = torch.tensor(q_ref[:q_dim, :].T, dtype=torch.float32, device=torch.device(device))
        q_hat = model(q_t).detach().cpu().numpy().astype(np.float64, copy=False).T
    return q_hat, {
        "q_dim": q_dim,
        "latent_dim": latent_dim,
        "hidden_dims": hidden_dims,
        "scaling": scaling,
        "activation": activation,
    }


def model_specs(models_dir: Path, family: str):
    specs = [
        ("case1", "PROM-ANN Case 1 reconstruction", "Case1_Best", models_dir / "case1_ann_ntot151_best.pt"),
        ("case2_np10", "PROM-ANN Case 2 n=10 reconstruction", "Case2_Best_np10", models_dir / "case2_ann_ntot151_np10_best.pt"),
        ("case2_np20", "PROM-ANN Case 2 n=20 reconstruction", "Case2_Best_np20", models_dir / "case2_ann_ntot151_np20_best.pt"),
        ("case3", "PROM-ANN Case 3 reconstruction", "Case3_Best", models_dir / "case3_ann_ntot151_best.pt"),
        ("podae", "PROM-POD-AE reconstruction", "PODAE_Best", models_dir / "prom_pod_ae_ntot151_best.pt"),
    ]
    if family == "all":
        return specs
    if family == "ann":
        return [s for s in specs if s[0] != "podae"]
    return [s for s in specs if s[0] == family]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["all", "ann", "case1", "case2_np10", "case2_np20", "case3", "podae"], default="all")
    parser.add_argument("--hprom-root", type=Path, default=PROJECT / "Results_Paper" / "mlspg_hprom_main")
    parser.add_argument("--output-root", type=Path, default=PROJECT / "Results_Paper" / "tmp_reconstruction_only_mlspg_main")
    parser.add_argument("--basis-path", type=Path, default=PROJECT / "Results_Paper" / "MetricStudy" / "lspg_sensitive" / "Stage1" / "basis.npy")
    parser.add_argument("--u-ref-path", type=Path, default=PROJECT / "Results_Paper" / "MetricStudy" / "lspg_sensitive" / "Stage1" / "u_ref.npy")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--save-snaps", action="store_true")
    parser.add_argument("--allow-compute-hdm", action="store_true")
    parser.add_argument("--chunk-cols", type=int, default=32)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available; using CPU.")
        args.device = "cpu"

    V = np.load(args.basis_path, mmap_mode="r", allow_pickle=False)
    u_ref = np.load(args.u_ref_path, allow_pickle=False).reshape(-1)
    models_dir = args.hprom_root / "Stage3" / "models"
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for key, label, family_dir, model_path in model_specs(models_dir, args.family):
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
        for point_label, mu1, mu2 in POINTS:
            q_ref_path = linear_q_path(args.hprom_root, mu1, mu2)
            if not q_ref_path.exists():
                raise FileNotFoundError(f"Missing linear-HPROM q reference: {q_ref_path}")
            q_ref = np.load(q_ref_path, allow_pickle=False)

            if key == "case1":
                q_rec, meta = q_case1(q_ref, model_path, args.device)
            elif key in ("case2_np10", "case2_np20"):
                q_rec, meta = q_case2_mu(q_ref, model_path, args.device, (mu1, mu2))
            elif key == "case3":
                q_rec, meta = q_case3(q_ref, model_path, args.device, (mu1, mu2))
            elif key == "podae":
                q_rec, meta = q_podae(q_ref, model_path, args.device)
            else:
                raise AssertionError(key)

            hdm, hdm_source = load_hdm(mu1, mu2, allow_compute=args.allow_compute_hdm)
            out_dir = args.output_root / "Runs" / family_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            tag = mu_tag(mu1, mu2)
            q_out = out_dir / f"{key}_reconstruction_only_{tag}_qN.npy"
            np.save(q_out, q_rec)
            snaps_out = out_dir / f"{key}_reconstruction_only_{tag}_snaps.npy" if args.save_snaps else None
            rel_err = reconstruct_error(
                np.asarray(V[:, : q_rec.shape[0]], dtype=np.float64),
                u_ref,
                q_rec,
                hdm,
                save_path=snaps_out,
                chunk_cols=args.chunk_cols,
            )

            summary = out_dir / f"{key}_reconstruction_only_{tag}_summary.txt"
            write_kv_txt(
                str(summary),
                [
                    ("diagnostic", "reconstruction_only_from_linear_hprom_qref"),
                    ("model_label", label),
                    ("point_label", point_label),
                    ("mu_test", (mu1, mu2)),
                    ("model_path", model_path),
                    ("basis_path", args.basis_path),
                    ("u_ref_path", args.u_ref_path),
                    ("linear_hprom_q_ref", q_ref_path),
                    ("hdm_source", hdm_source),
                    ("device", args.device),
                    ("q_shape", q_rec.shape),
                    ("relative_error_percent", rel_err),
                    ("qN_output", q_out),
                    ("snaps_output", snaps_out if snaps_out is not None else "not_saved"),
                    *[(k, v) for k, v in meta.items()],
                ],
            )
            rows.append(
                {
                    "family": key,
                    "model_label": label,
                    "point_label": point_label,
                    "mu1": mu1,
                    "mu2": mu2,
                    "relative_error_percent": rel_err,
                    "q_shape": str(tuple(q_rec.shape)),
                    "summary": str(summary),
                    "qN_output": str(q_out),
                    "snaps_output": str(snaps_out) if snaps_out is not None else "not_saved",
                }
            )
            print(f"[recon] {label} | {point_label} | mu=({mu1:.3f},{mu2:.4f}) | error={rel_err:.4f}%")

    csv_path = args.output_root / "reconstruction_only_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] summary csv: {csv_path}")


if __name__ == "__main__":
    main()
