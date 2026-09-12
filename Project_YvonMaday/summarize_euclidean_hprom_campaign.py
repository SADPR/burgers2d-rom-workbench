#!/usr/bin/env python3
"""Build the publication HPROM accuracy/timing table from frozen campaign outputs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


POINTS = (
    ("verification", 4.875, 0.0225),
    ("offgrid1", 4.560, 0.0190),
    ("offgrid2", 5.190, 0.0260),
    ("extrapolation20pct", 4.000, 0.0330),
)


def kv(path):
    result = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result


def unique(paths, description):
    matches = sorted(paths)
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {description}; found {len(matches)}: {matches}")
    return matches[0]


def flat_record(root, prefix, mu1, mu2):
    tag = f"mu1_{mu1:.3f}_mu2_{mu2:.4f}"
    summary = unique(root.glob(f"{prefix}*{tag}*_summary.txt"), f"summary in {root} for {tag}")
    qpath = unique(root.glob(f"{prefix}*{tag}*_qN.npy"), f"qN in {root} for {tag}")
    data = kv(summary)
    return qpath, float(data["relative_error_percent"]), float(data["online_solve_elapsed_s"]), int(data["n_ecsw_elements"])


def directory_record(root, prefix, summary_name, mu1, mu2):
    tag = f"mu1_{mu1:.3f}_mu2_{mu2:.4f}"
    folder = unique(root.glob(f"{prefix}_{tag}_ntot151*"), f"run directory in {root} for {tag}")
    data = kv(folder / summary_name)
    return folder / "qN.npy", float(data["relative_error_percent"])


def direct_times(path):
    data = kv(path)
    return {
        "POD-NN-ROM": {name: float(data[f"podnn_{name}_mean_inference_time_s"]) for name, *_ in POINTS},
        "POD-DL-ROM": {name: float(data[f"poddl_{name}_mean_inference_time_s"]) for name, *_ in POINTS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    output = (args.output or root / "reporting").resolve()
    output.mkdir(parents=True, exist_ok=True)

    direct_summary = unique((root / "timing").glob("direct_inference_repeat*_summary.txt"), "direct timing summary")
    direct = direct_times(direct_summary)
    hdm = json.loads((root / "timing/hdm/hdm_timing.json").read_text())
    hdm_time = float(hdm["mean_in_domain_seconds"])

    rows = []
    method_specs = (
        ("Linear HPROM", "linear", root / "Runs/Linear", "linear_hprom"),
        ("HPROM-ANN Case 1", "flat", root / "Runs/HPROM_ANN_C1", "case1_hprom_ann"),
        ("HPROM-ANN Case 2", "flat", root / "Runs/HPROM_ANN_C2", "case2_hprom_ann"),
        ("HPROM-ANN Case 2+B3", "b3", root / "Stage4/case2_b3", ""),
        ("HPROM-ANN Case 3", "flat", root / "Runs/HPROM_ANN_C3", "case3_hprom_ann"),
        ("HPROM-POD-AE", "flat", root / "Runs/HPROM_POD_AE", "podae_hprom"),
        ("POD-NN-ROM", "direct", root / "Runs/PODNN_ROM", "rom_data_driven"),
        ("POD-DL-ROM", "direct_dl", root / "Runs/PODDL_ROM", "pod_dl_data_driven"),
    )

    for method, kind, method_root, prefix in method_specs:
        errors, coefficient_errors, times, cells = [], [], [], []
        for point, (label, mu1, mu2) in enumerate(POINTS):
            linear_folder = unique((root / "Runs/Linear").glob(
                f"linear_hprom_mu1_{mu1:.3f}_mu2_{mu2:.4f}_ntot151"
            ), f"linear HPROM reference for {label}")
            teacher = np.load(linear_folder / "qN.npy", allow_pickle=False)
            if kind in ("linear", "flat"):
                qpath, error, elapsed, ncell = flat_record(method_root, prefix, mu1, mu2) if kind == "flat" else (
                    linear_folder / "qN.npy", float(kv(linear_folder / "summary.txt")["relative_error_percent"]),
                    float(kv(linear_folder / "summary.txt")["online_solve_elapsed_s"]),
                    int(kv(linear_folder / "summary.txt")["n_ecsw_elements"]),
                )
            elif kind == "b3":
                summary = method_root / f"reporting_positive_fit4096_hprom3_p{point}_steps500/summary.json"
                data = json.loads(summary.read_text())
                qpath, error, elapsed, ncell = summary.parent / "qN.npy", float(data["state_error_percent"]), float(data["online_seconds"]), int(data["sampled_cells"])
            elif kind == "direct":
                qpath, error = directory_record(method_root, prefix, "rom_data_driven_summary.txt", mu1, mu2)
                elapsed, ncell = direct[method][label], None
            else:
                qpath, error = directory_record(method_root, prefix, "pod_dl_data_driven_summary.txt", mu1, mu2)
                elapsed, ncell = direct[method][label], None
            q = np.load(qpath, allow_pickle=False)
            if q.shape != teacher.shape:
                raise ValueError(f"Coefficient shape mismatch for {method}, {label}: {q.shape} vs {teacher.shape}")
            coefficient = 100.0 * np.linalg.norm(q - teacher) / np.linalg.norm(teacher)
            errors.append(error); coefficient_errors.append(float(coefficient)); times.append(elapsed)
            if ncell is not None: cells.append(ncell)
        row = {
            "model": method,
            "state_verification_percent": errors[0],
            "state_offgrid1_percent": errors[1],
            "state_offgrid2_percent": errors[2],
            "state_in_domain_mean_percent": float(np.mean(errors[:3])),
            "state_extrapolation_percent": errors[3],
            "coefficient_in_domain_mean_percent": float(np.mean(coefficient_errors[:3])),
            "coefficient_extrapolation_percent": coefficient_errors[3],
            "online_in_domain_mean_seconds": float(np.mean(times[:3])),
            "speedup_vs_hdm": hdm_time / float(np.mean(times[:3])),
            "sampled_cells": "" if not cells else int(round(np.mean(cells))),
        }
        rows.append(row)

    fields = list(rows[0])
    with (output / "hprom_accuracy_timing.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    lines = [f"Matched fresh HDM in-domain mean: {hdm_time:.6f} s", ""]
    for row in rows:
        lines.append(
            f"{row['model']}: state mean={row['state_in_domain_mean_percent']:.4f}%, "
            f"extrap={row['state_extrapolation_percent']:.4f}%, "
            f"time={row['online_in_domain_mean_seconds']:.6f}s, speedup={row['speedup_vs_hdm']:.2f}x"
        )
    (output / "hprom_accuracy_timing.txt").write_text("\n".join(lines) + "\n")

    latex = [
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Model & In-domain error (\%) & $\mu^{(3)}$ error (\%) & Mean time (s) & Speed-up \\",
        r"\midrule",
    ]
    for row in rows:
        name = row["model"].replace("+", r"$+$")
        latex.append(f"{name} & {row['state_in_domain_mean_percent']:.3f} & {row['state_extrapolation_percent']:.3f} & {row['online_in_domain_mean_seconds']:.4g} & {row['speedup_vs_hdm']:.1f} \\")
    latex += [r"\bottomrule", r"\end{tabular}"]
    (output / "hprom_accuracy_timing_table.tex").write_text("\n".join(latex) + "\n")
    print("\n".join(lines))
    print(f"Outputs: {output}")


if __name__ == "__main__":
    main()
