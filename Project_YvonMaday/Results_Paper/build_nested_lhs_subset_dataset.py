#!/usr/bin/env python3
"""Create a reproducible nested subset of an existing 9+36 ROM dataset.

The source trajectories are copied rather than re-solved. This keeps every
intermediate-enrichment study on exactly the same LHS design as the completed
36-point campaign while making the new dataset self-contained for training.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--n-interior", type=int, required=True)
    parser.add_argument("--n-exterior", type=int, required=True)
    parser.add_argument("--subset-label", default="nested_lhs_subset")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _normalize(points: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return (points - lower) / (upper - lower)


def _greedy_maximin_order(points: np.ndarray, initial: list[int]) -> list[int]:
    """Return a deterministic space-filling order, preserving ``initial``."""
    if points.ndim != 2 or len(points) == 0:
        raise ValueError("Expected a non-empty two-dimensional point array.")

    chosen = list(dict.fromkeys(initial))
    remaining = [idx for idx in range(len(points)) if idx not in chosen]
    if not chosen:
        centroid = points.mean(axis=0)
        first = min(remaining, key=lambda idx: (np.linalg.norm(points[idx] - centroid), idx))
        chosen.append(first)
        remaining.remove(first)

    while remaining:
        selected = points[np.asarray(chosen)]
        candidate = max(
            remaining,
            key=lambda idx: (float(np.min(np.linalg.norm(selected - points[idx], axis=1))), -idx),
        )
        chosen.append(candidate)
        remaining.remove(candidate)
    return chosen


def _exterior_order(
    points: np.ndarray, labels: list[str], lower: np.ndarray, upper: np.ndarray
) -> list[int]:
    normalized = _normalize(points, lower, upper)
    expected_labels = ("left_margin", "right_margin", "top_margin", "bottom_margin")
    if set(labels) != set(expected_labels):
        raise ValueError(f"Expected four margin strata {expected_labels}, got {sorted(set(labels))}.")

    # Seed every margin once. Any prefix with at least four samples therefore
    # covers all four directions before greedy maximin fills the rest.
    seeds: list[int] = []
    for label in expected_labels:
        members = [idx for idx, value in enumerate(labels) if value == label]
        centroid = normalized[members].mean(axis=0)
        seeds.append(min(members, key=lambda idx: (np.linalg.norm(normalized[idx] - centroid), idx)))
    return _greedy_maximin_order(normalized, seeds)


def _selected_rows(
    rows: list[dict[str, str]], selected_interior: list[int], selected_exterior: list[int]
) -> list[dict[str, str]]:
    baseline = [row.copy() for row in rows if row["role"] == "baseline_training"]
    lhs_rows = [row.copy() for row in rows if row["role"] == "lhs_enrichment"]
    interior_rows = [row for row in lhs_rows if row["region"] == "interior_original_box"]
    exterior_rows = [row for row in lhs_rows if row["region"] != "interior_original_box"]
    if len(baseline) != 9 or len(interior_rows) != 18 or len(exterior_rows) != 18:
        raise ValueError("Source manifest must contain 9 baseline, 18 interior, and 18 exterior rows.")

    output = baseline
    for index in selected_interior:
        row = interior_rows[index]
        row["source_lhs_group"] = "interior"
        row["source_lhs_index"] = str(index)
        output.append(row)
    for index in selected_exterior:
        row = exterior_rows[index]
        row["source_lhs_group"] = "exterior"
        row["source_lhs_index"] = str(index)
        output.append(row)
    return output


def _copy_trajectory(source: Path, destination: Path) -> None:
    qn = np.load(source / "qN.npy", allow_pickle=False)
    if qn.shape != (151, 501) or not np.all(np.isfinite(qn)):
        raise ValueError(f"Invalid source solver coordinates in {source}: shape={qn.shape}.")
    shutil.copytree(source, destination)


def main() -> None:
    args = _parse_args()
    source = args.source_dataset.expanduser().resolve()
    output = args.output_dataset.expanduser().resolve()
    if args.n_interior < 1 or args.n_interior > 18:
        raise SystemExit("--n-interior must be between 1 and 18.")
    if args.n_exterior < 4 or args.n_exterior > 18:
        raise SystemExit("--n-exterior must be between 4 and 18 so every margin is represented.")
    if not (source / "meta.json").is_file():
        raise SystemExit(f"Missing source metadata: {source / 'meta.json'}")

    meta: dict[str, Any] = json.loads((source / "meta.json").read_text())
    manifest = _load_csv(source / "parameter_manifest.csv")
    interior = np.load(source / "interior_lhs_mu.npy", allow_pickle=False)
    exterior = np.load(source / "exterior_lhs_mu.npy", allow_pickle=False)
    labels = json.loads((source / "exterior_region_labels.json").read_text())
    if interior.shape != (18, 2) or exterior.shape != (18, 2) or len(labels) != 18:
        raise SystemExit("Expected the completed 18+18 LHS source design.")

    lower = np.asarray([meta["expanded_mu1_range"][0], meta["expanded_mu2_range"][0]], dtype=float)
    upper = np.asarray([meta["expanded_mu1_range"][1], meta["expanded_mu2_range"][1]], dtype=float)
    interior_order = _greedy_maximin_order(_normalize(interior, lower, upper), [])
    exterior_order = _exterior_order(exterior, labels, lower, upper)
    selected_interior = interior_order[: args.n_interior]
    selected_exterior = exterior_order[: args.n_exterior]
    selected_rows = _selected_rows(manifest, selected_interior, selected_exterior)

    print("[nested-lhs-subset] source:", source)
    print("[nested-lhs-subset] output:", output)
    print("[nested-lhs-subset] selected interior source indices:", selected_interior)
    print("[nested-lhs-subset] selected exterior source indices:", selected_exterior)
    print("[nested-lhs-subset] exterior labels:", [labels[idx] for idx in selected_exterior])
    print("[nested-lhs-subset] trajectories:", len(selected_rows), f"(9 + {args.n_interior} + {args.n_exterior})")

    if args.plan_only:
        print("[nested-lhs-subset] PLAN_ONLY=1; no files were written.")
        return
    if output.exists():
        if not args.force:
            raise SystemExit(f"Output dataset already exists: {output}\nPass --force to replace it.")
        shutil.rmtree(output)
    (output / "per_mu").mkdir(parents=True)

    for row in selected_rows:
        source_dir = source / "per_mu" / row["directory_tag"]
        if not source_dir.is_dir():
            raise SystemExit(f"Missing source trajectory directory: {source_dir}")
        _copy_trajectory(source_dir, output / "per_mu" / row["directory_tag"])

    selected_interior_array = interior[np.asarray(selected_interior)]
    selected_exterior_array = exterior[np.asarray(selected_exterior)]
    baseline_array = np.asarray([[float(row["mu1"]), float(row["mu2"])] for row in selected_rows[:9]])
    np.save(output / "baseline_mu.npy", baseline_array)
    np.save(output / "interior_lhs_mu.npy", selected_interior_array)
    np.save(output / "exterior_lhs_mu.npy", selected_exterior_array)
    np.save(output / "lhs_mu.npy", np.vstack((selected_interior_array, selected_exterior_array)))
    candidate = source / "evaluation_mu.npy"
    if candidate.is_file():
        shutil.copy2(candidate, output / "evaluation_mu.npy")

    fieldnames = ["role", "region", "mu1", "mu2", "directory_tag", "source_lhs_group", "source_lhs_index"]
    with (output / "parameter_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    selected_labels = [labels[idx] for idx in selected_exterior]
    (output / "exterior_region_labels.json").write_text(json.dumps(selected_labels, indent=2) + "\n")
    subset_manifest = {
        "subset_label": args.subset_label,
        "source_dataset": str(source),
        "source_lhs_seed": meta.get("lhs_seed"),
        "selection_method": "deterministic nested centre-seeded greedy maximin in expanded normalized parameter space; exterior prefixes are seeded with one representative from each margin",
        "interior_order_source_indices": interior_order,
        "exterior_order_source_indices": exterior_order,
        "selected_interior_source_indices": selected_interior,
        "selected_exterior_source_indices": selected_exterior,
        "selected_exterior_labels": selected_labels,
    }
    (output / "nested_subset_manifest.json").write_text(json.dumps(subset_manifest, indent=2) + "\n")

    new_meta = dict(meta)
    new_meta.update(
        {
            "is_nested_subset_dataset": True,
            "enrichment_protocol": f"baseline_9_plus_nested_subset_{args.n_interior}interior_{args.n_exterior}margin_from_lhs36",
            "source_enrichment_dataset": str(source),
            "subset_label": args.subset_label,
            "num_traj": len(selected_rows),
            "num_base_traj_copied": 9,
            "num_lhs_traj": args.n_interior + args.n_exterior,
            "num_interior_lhs_traj": args.n_interior,
            "num_exterior_lhs_traj": args.n_exterior,
            "exterior_region_labels": selected_labels,
            "selected_interior_source_indices": selected_interior,
            "selected_exterior_source_indices": selected_exterior,
        }
    )
    (output / "meta.json").write_text(json.dumps(new_meta, indent=2) + "\n")
    (output / "stage2_nested_subset_summary.txt").write_text(
        "\n".join(
            [
                f"source_dataset: {source}",
                f"output_dataset: {output}",
                f"subset_label: {args.subset_label}",
                f"n_interior: {args.n_interior}",
                f"n_exterior: {args.n_exterior}",
                f"num_traj: {len(selected_rows)}",
                f"selected_interior_source_indices: {selected_interior}",
                f"selected_exterior_source_indices: {selected_exterior}",
                f"selection_method: {subset_manifest['selection_method']}",
            ]
        )
        + "\n"
    )
    print("[nested-lhs-subset] created:", output)


if __name__ == "__main__":
    main()
