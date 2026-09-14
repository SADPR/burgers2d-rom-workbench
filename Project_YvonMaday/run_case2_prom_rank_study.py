#!/usr/bin/env python3
"""Complete the full-residual PROM rank study for the frozen B3 choice."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "Project_YvonMaday"
sys.path.insert(0, str(ROOT))
RANKS = (0, 1, 2, 3, 5, 10)
FROZEN_RANK = 3
CONTROL_DIMENSIONS = tuple(10 + rank for rank in RANKS)


def read(path):
    return json.loads(path.read_text())


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_runner(output, stage, methods, threads, nprimary=10):
    command = [
        sys.executable, "-u", str(PROJECT / "run_case2_local_corrections.py"),
        "--stage", stage, "--methods", *methods,
        "--threads", str(threads), "--n-primary", str(nprimary),
        "--initialization", "linear", "--output", str(output),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def rank_summary(root):
    rows = []
    for point in range(2):
        for rank in RANKS:
            method = "baseline" if rank == 0 else f"krylov{rank}"
            data = read(root / "validation_ranks" /
                        f"validation_{point}_{method}_steps500/summary.json")
            rows.append({
                "point": point,
                "rank": rank,
                "online_unknowns": 10 + rank,
                "coefficient_error_percent": data["coefficient_error_percent"],
                "primary_error_percent": data["primary_error_percent"],
                "secondary_error_percent": data["secondary_error_percent"],
                "iterations": data["iterations"],
                "attained_rank_min": data["min_rank"],
                "attained_rank_max": data["max_rank"],
                "mean_residual_norm": data["mean_residual_norm"],
            })
    return rows


def equal_dimension_summary(root):
    """Collect plain Case-2 controls with the same number of online unknowns."""
    rows = []
    for point in range(2):
        for rank, dimension in zip(RANKS, CONTROL_DIMENSIONS):
            if dimension == 10:
                parent = root / "validation_ranks"
                tag = "baseline"
            elif dimension == 13:
                parent = root / "matched_n13"
                tag = "baseline_n13"
            else:
                parent = root / "equal_dimension_controls" / f"n{dimension}"
                tag = f"baseline_n{dimension}"
            path = parent / f"validation_{point}_{tag}_steps500/summary.json"
            data = read(path)
            rows.append({
                "point": point,
                "rank": rank,
                "primary_dimension": dimension,
                "online_unknowns": dimension,
                "coefficient_error_percent": data["coefficient_error_percent"],
                "primary_error_percent": data["primary_error_percent"],
                "secondary_error_percent": data["secondary_error_percent"],
                "iterations": data["iterations"],
                "mean_residual_norm": data["mean_residual_norm"],
            })
    return rows


def matched_reporting(root):
    previous = (PROJECT / "Results_Paper" /
                "euclidean_case2_sherlock_42621312_1Dptdf/rep01")
    specifications = (
        ("Case 2, n=10", previous / "case2_known_initial", "baseline", 10),
        ("Case 2 + B3, n=10", previous / "adaptive3_known_initial", "krylov3", 13),
        ("Case 2, n=13", root / "matched_n13", "baseline_n13", 13),
    )
    rows = []
    for point in range(4):
        for label, parent, tag, unknowns in specifications:
            matches = list(parent.glob(f"reporting_{point}_{tag}_steps500/summary.json"))
            if len(matches) != 1:
                raise FileNotFoundError(f"Expected one result for {label}, point {point}: {matches}")
            data = read(matches[0])
            rows.append({
                "point": point,
                "model": label,
                "online_unknowns": unknowns,
                "state_error_percent": data["state_error_percent"],
                "coefficient_error_percent": data["coefficient_error_percent"],
                "iterations": data["iterations"],
            })
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    from Project_YvonMaday.run_case2_local_corrections import BASIS, MODEL, sha256

    protocol = {
        "purpose": "Full-residual PROM characterization of the previously frozen B3 rank",
        "ranks": list(RANKS),
        "primary_dimension": 10,
        "frozen_rank": FROZEN_RANK,
        "frozen_rank_basis": "Earlier validation screening over ranks 1, 3, and 5",
        "rank_sweep_role": "Post-selection characterization; it does not alter the frozen rank",
        "matched_control": "Case 2 n=13 versus Case 2 n=10 plus B3",
        "initialization": "common linear-PROM initial representation",
        "selection_uses_reporting_points": False,
        "timings_reported": False,
        "threads": args.threads,
        "inputs": {
            "model": sha256(MODEL),
            "basis": sha256(BASIS / "basis.npy"),
            "reference_state": sha256(BASIS / "u_ref.npy"),
            "runner": sha256(PROJECT / "run_case2_local_corrections.py"),
            "solver": sha256(ROOT / "burgers/case2_residual_correction.py"),
        },
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and read(protocol_path) != protocol:
        raise RuntimeError("Protocol changed; use a fresh output directory")
    write_json(protocol_path, protocol)

    control_protocol = {
        "purpose": "Equal-online-dimension controls for the validation-only correction-rank sweep",
        "correction_ranks": list(RANKS),
        "plain_case2_primary_dimensions": list(CONTROL_DIMENSIONS),
        "pairing": "Case 2 plus B_r at n=10 versus plain Case 2 at n=10+r",
        "initialization": "common linear-PROM initial representation",
        "parameters": "two external ANN-validation parameters only",
        "selection_uses_reporting_points": False,
        "timings_reported": False,
        "threads": args.threads,
        "inputs": protocol["inputs"],
    }
    control_protocol_path = output / "equal_dimension_protocol.json"
    if control_protocol_path.exists() and read(control_protocol_path) != control_protocol:
        raise RuntimeError("Equal-dimension protocol changed; use a fresh output directory")
    write_json(control_protocol_path, control_protocol)

    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS", "GOTO_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = str(args.threads)
    os.environ.update({key: environment[key] for key in environment if key.endswith("NUM_THREADS")})

    methods = ["baseline", *[f"krylov{rank}" for rank in RANKS if rank]]
    run_runner(output / "validation_ranks", "validation", methods, args.threads)
    for dimension in CONTROL_DIMENSIONS[1:]:
        parent = (output / "matched_n13" if dimension == 13 else
                  output / "equal_dimension_controls" / f"n{dimension}")
        run_runner(parent, "validation", ["baseline"], args.threads, dimension)
    run_runner(output / "matched_n13", "reporting", ["baseline"], args.threads, 13)

    ranks = rank_summary(output)
    controls = equal_dimension_summary(output)
    reporting = matched_reporting(output)
    write_json(output / "rank_validation.json", ranks)
    write_csv(output / "rank_validation.csv", ranks)
    write_json(output / "equal_dimension_validation.json", controls)
    write_csv(output / "equal_dimension_validation.csv", controls)
    write_json(output / "matched_reporting.json", reporting)
    write_csv(output / "matched_reporting.csv", reporting)
    print(f"Complete: {output}")


if __name__ == "__main__":
    main()
