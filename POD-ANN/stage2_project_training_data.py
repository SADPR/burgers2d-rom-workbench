#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 2: PROJECT SNAPSHOTS ONTO POD BASIS (POD-ANN)

This stage projects HDM snapshots onto the POD basis from stage1 and stores:
  - q_p.npy, q_s.npy           (training projections)
  - q_p_test.npy, q_s_test.npy (test projections)
"""

import os
import sys
import time
from datetime import datetime

import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from burgers.core import load_or_compute_snaps, get_snapshot_params
from burgers.config import GRID_X, GRID_Y, W0, DT, NUM_STEPS


def _format_report_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if np.isfinite(value):
            return f"{value:.8e}"
        return str(value)
    return str(value)


def write_txt_report(report_path, sections):
    lines = []
    for section_name, items in sections:
        lines.append(f"[{section_name}]")
        for key, value in items:
            lines.append(f"{key}: {_format_report_value(value)}")
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines).rstrip() + "\n")


def resolve_u_ref(uref_mode, uref_file, n_dofs):
    mode = str(uref_mode).strip().lower()
    if mode not in ("auto", "on", "off"):
        raise ValueError("uref_mode must be one of: 'auto', 'on', 'off'.")

    u_ref_path_used = None
    use_u_ref = False

    if mode == "off":
        u_ref = np.zeros(n_dofs, dtype=np.float64)
    else:
        if not os.path.exists(uref_file):
            if mode == "on":
                raise FileNotFoundError(
                    f"uref_mode='on' requires u_ref file, but not found: {uref_file}"
                )
            u_ref = np.zeros(n_dofs, dtype=np.float64)
        else:
            u_ref = np.asarray(np.load(uref_file, allow_pickle=False), dtype=np.float64).reshape(-1)
            if u_ref.size != n_dofs:
                raise ValueError(f"u_ref size mismatch: got {u_ref.size}, expected {n_dofs}.")
            u_ref_path_used = uref_file
            use_u_ref = True

    return u_ref, use_u_ref, u_ref_path_used


def aggregate_snapshots(mu_list, grid_x, grid_y, w0, dt, num_steps, snap_folder):
    snaps_list = []
    loaded = []

    for mu in mu_list:
        s_mu = np.asarray(
            load_or_compute_snaps(mu, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder),
            dtype=np.float64,
        )
        snaps_list.append(s_mu)
        loaded.append(list(mu))

    if len(snaps_list) == 0:
        return np.empty((0, 0), dtype=np.float64), loaded

    n_dofs, n_time = snaps_list[0].shape
    for i, s in enumerate(snaps_list):
        if s.shape != (n_dofs, n_time):
            raise RuntimeError(
                "Snapshot shape mismatch while aggregating snapshots: "
                f"first={snaps_list[0].shape}, current={s.shape}, mu={loaded[i]}."
            )

    snaps = np.hstack(snaps_list)
    return snaps, loaded


def main(
    basis_file=os.path.join(script_dir, "basis.npy"),
    uref_file=os.path.join(script_dir, "u_ref.npy"),
    q_p_file=os.path.join(script_dir, "q_p.npy"),
    q_s_file=os.path.join(script_dir, "q_s.npy"),
    q_p_test_file=os.path.join(script_dir, "q_p_test.npy"),
    q_s_test_file=os.path.join(script_dir, "q_s_test.npy"),
    metadata_file=os.path.join(script_dir, "stage2_projection_metadata.npz"),
    report_file=os.path.join(script_dir, "stage2_projection_summary.txt"),
    snap_folder=os.path.join(parent_dir, "Results", "param_snaps"),
    dt=DT,
    num_steps=NUM_STEPS,
    primary_modes=10,
    total_modes=None,
    uref_mode="auto",
    test_params=None,
):
    if not os.path.exists(basis_file):
        raise FileNotFoundError(f"Missing POD basis file: {basis_file}. Run stage1 first.")

    if test_params is None:
        test_params = [[4.75, 0.020], [4.56, 0.019], [5.19, 0.026]]

    os.makedirs(os.path.dirname(q_p_file), exist_ok=True)
    os.makedirs(snap_folder, exist_ok=True)

    basis = np.asarray(np.load(basis_file, allow_pickle=False), dtype=np.float64)
    n_dofs, n_basis = basis.shape

    if total_modes is None:
        total_modes = n_basis

    if total_modes < 2:
        raise ValueError("total_modes must be >= 2.")
    if total_modes > n_basis:
        raise ValueError(f"Requested total_modes={total_modes}, but basis has {n_basis} columns.")
    if primary_modes < 1:
        raise ValueError("primary_modes must be >= 1.")
    if primary_modes >= total_modes:
        raise ValueError(f"primary_modes={primary_modes} must be < total_modes={total_modes}.")

    print("\n====================================================")
    print("      STAGE 2: PROJECT SNAPSHOTS TO (q_p, q_s)")
    print("====================================================")
    print(f"[STAGE2] basis shape: {basis.shape}")
    print(f"[STAGE2] primary_modes={primary_modes}, total_modes={total_modes}")

    w0 = np.asarray(W0, dtype=np.float64).copy()
    basis_modes = basis[:, :total_modes]
    u_ref, use_u_ref, u_ref_path_used = resolve_u_ref(uref_mode, uref_file, n_dofs)
    print(
        f"[STAGE2] u_ref mode={uref_mode}, use_u_ref={use_u_ref}, "
        f"||u_ref||_2={np.linalg.norm(u_ref):.3e}"
    )

    train_params = get_snapshot_params()

    t0 = time.time()
    snaps_train, train_loaded = aggregate_snapshots(
        train_params,
        GRID_X,
        GRID_Y,
        w0,
        dt,
        num_steps,
        snap_folder,
    )
    elapsed_train_load = time.time() - t0

    if snaps_train.shape[0] != n_dofs:
        raise RuntimeError(
            f"State size mismatch between basis ({n_dofs}) and training snapshots ({snaps_train.shape[0]})."
        )

    t0 = time.time()
    q_train = basis_modes.T @ (snaps_train - u_ref[:, None])
    q_p = q_train[:primary_modes, :]
    q_s = q_train[primary_modes:total_modes, :]
    elapsed_train_project = time.time() - t0

    np.save(q_p_file, q_p)
    np.save(q_s_file, q_s)

    print(f"[STAGE2] Training snapshot matrix shape: {snaps_train.shape}")
    print(f"[STAGE2] q_p shape: {q_p.shape}, q_s shape: {q_s.shape}")

    t0 = time.time()
    snaps_test, test_loaded = aggregate_snapshots(
        test_params,
        GRID_X,
        GRID_Y,
        w0,
        dt,
        num_steps,
        snap_folder,
    )
    elapsed_test_load = time.time() - t0

    if snaps_test.shape[0] != n_dofs:
        raise RuntimeError(
            f"State size mismatch between basis ({n_dofs}) and test snapshots ({snaps_test.shape[0]})."
        )

    t0 = time.time()
    q_test = basis_modes.T @ (snaps_test - u_ref[:, None])
    q_p_test = q_test[:primary_modes, :]
    q_s_test = q_test[primary_modes:total_modes, :]
    elapsed_test_project = time.time() - t0

    np.save(q_p_test_file, q_p_test)
    np.save(q_s_test_file, q_s_test)
    np.savez(
        metadata_file,
        u_ref_used=u_ref,
        use_u_ref=np.asarray(use_u_ref, dtype=np.int64),
        uref_mode=np.asarray(uref_mode),
        u_ref_file=np.asarray(u_ref_path_used if u_ref_path_used is not None else ""),
        primary_modes=np.asarray(primary_modes, dtype=np.int64),
        total_modes=np.asarray(total_modes, dtype=np.int64),
    )

    print(f"[STAGE2] Test snapshot matrix shape: {snaps_test.shape}")
    print(f"[STAGE2] q_p_test shape: {q_p_test.shape}, q_s_test shape: {q_s_test.shape}")

    write_txt_report(
        report_file,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("script", "stage2_project_training_data.py"),
                ],
            ),
            (
                "configuration",
                [
                    ("basis_file", basis_file),
                    ("uref_mode", uref_mode),
                    ("use_u_ref", use_u_ref),
                    ("u_ref_file", u_ref_path_used),
                    ("u_ref_l2_norm", float(np.linalg.norm(u_ref))),
                    ("snap_folder", snap_folder),
                    ("dt", dt),
                    ("num_steps", num_steps),
                    ("primary_modes", primary_modes),
                    ("total_modes", total_modes),
                ],
            ),
            (
                "train_projection",
                [
                    ("num_train_parameters", len(train_loaded)),
                    ("snaps_train_shape", snaps_train.shape),
                    ("q_p_shape", q_p.shape),
                    ("q_s_shape", q_s.shape),
                ],
            ),
            (
                "test_projection",
                [
                    ("test_parameters", test_loaded),
                    ("snaps_test_shape", snaps_test.shape),
                    ("q_p_test_shape", q_p_test.shape),
                    ("q_s_test_shape", q_s_test.shape),
                ],
            ),
            (
                "timings_seconds",
                [
                    ("load_training_snapshots", elapsed_train_load),
                    ("project_training_snapshots", elapsed_train_project),
                    ("load_test_snapshots", elapsed_test_load),
                    ("project_test_snapshots", elapsed_test_project),
                    (
                        "total",
                        elapsed_train_load
                        + elapsed_train_project
                        + elapsed_test_load
                        + elapsed_test_project,
                    ),
                ],
            ),
            (
                "outputs",
                [
                    ("q_p_npy", q_p_file),
                    ("q_s_npy", q_s_file),
                    ("q_p_test_npy", q_p_test_file),
                    ("q_s_test_npy", q_s_test_file),
                    ("projection_metadata_npz", metadata_file),
                    ("summary_txt", report_file),
                ],
            ),
        ],
    )
    print(f"[STAGE2] Summary saved: {report_file}")


if __name__ == "__main__":
    main()
