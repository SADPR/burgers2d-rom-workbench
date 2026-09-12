#!/usr/bin/env python3
"""Accept a frozen Case-2+B3 cubature rule using validation data only."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rule-file", type=Path, required=True)
    parser.add_argument("--min-gram", type=float, default=0.8)
    parser.add_argument("--max-gram", type=float, default=1.2)
    parser.add_argument("--max-gradient-error", type=float, default=0.05)
    parser.add_argument("--max-linearized-ratio", type=float, default=1.05)
    args = parser.parse_args()

    audit_path = args.rule_file.parent / "validation_operator_audit.json"
    if not audit_path.is_file():
        raise SystemExit(f"Missing validation audit: {audit_path}")
    audit = json.loads(audit_path.read_text())
    records = audit.get("records", [])
    if len(records) != 14:
        raise SystemExit(f"Expected 14 held-out operator records, found {len(records)}")

    mins = np.asarray([r["Gram_eigenvalue_min"] for r in records])
    maxs = np.asarray([r["Gram_eigenvalue_max"] for r in records])
    gradients = np.asarray([r["gradient_relative_error"] for r in records])
    ratios = np.asarray([
        r["sampled_true_linearized_residual"] / max(r["full_true_linearized_residual"], 1e-30)
        for r in records
    ])
    checks = {
        "minimum_Gram_eigenvalue": float(mins.min()),
        "maximum_Gram_eigenvalue": float(maxs.max()),
        "maximum_gradient_relative_error": float(gradients.max()),
        "maximum_sampled_to_full_linearized_residual_ratio": float(ratios.max()),
    }
    passed = (
        np.isfinite(list(checks.values())).all()
        and checks["minimum_Gram_eigenvalue"] >= args.min_gram
        and checks["maximum_Gram_eigenvalue"] <= args.max_gram
        and checks["maximum_gradient_relative_error"] <= args.max_gradient_error
        and checks["maximum_sampled_to_full_linearized_residual_ratio"] <= args.max_linearized_ratio
    )

    validation = []
    for point in range(2):
        matches = sorted(args.output.glob(f"validation_*_hprom3_p{point}_steps500/summary.json"))
        if len(matches) != 1:
            raise SystemExit(f"Expected one B3 validation run for point {point}, found {len(matches)}")
        record = json.loads(matches[0].read_text())
        for key in ("coefficient_error_percent", "online_seconds"):
            if key not in record or not np.isfinite(record[key]):
                raise SystemExit(f"Invalid {key} in {matches[0]}")
        validation.append({"point": point, "summary": str(matches[0]),
                           "state_error_percent": record.get("state_error_percent"),
                           "coefficient_error_percent": record["coefficient_error_percent"]})

    selection = {
        "accepted": bool(passed),
        "selection_uses_reporting_points": False,
        "rule_file": str(args.rule_file.resolve()),
        "operator_audit": str(audit_path.resolve()),
        "thresholds": {
            "min_gram": args.min_gram,
            "max_gram": args.max_gram,
            "max_gradient_error": args.max_gradient_error,
            "max_linearized_ratio": args.max_linearized_ratio,
        },
        "checks": checks,
        "held_out_validation": validation,
    }
    target = args.output / "selection.json"
    target.write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    print(json.dumps(selection, indent=2))
    if not passed:
        raise SystemExit("Case-2+B3 cubature rule failed the held-out operator audit")


if __name__ == "__main__":
    main()
