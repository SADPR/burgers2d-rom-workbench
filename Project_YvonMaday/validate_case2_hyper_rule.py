#!/usr/bin/env python3
"""Accept a frozen Case-2+B3 cubature rule using validation data only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rule-file", type=Path, required=True)
    parser.add_argument("--selection-file", type=Path,
                        help="Separate selection manifest for this frozen rule")
    parser.add_argument("--min-gram", type=float, default=0.8)
    parser.add_argument("--max-gram", type=float, default=1.2)
    parser.add_argument("--max-gradient-error", type=float, default=0.05)
    parser.add_argument("--max-linearized-ratio", type=float, default=1.05)
    args = parser.parse_args()
    thresholds = (args.min_gram, args.max_gram, args.max_gradient_error, args.max_linearized_ratio)
    if not np.isfinite(thresholds).all() or not (
        0 < args.min_gram <= args.max_gram and args.max_gradient_error >= 0
        and args.max_linearized_ratio >= 1
    ):
        parser.error("Invalid acceptance thresholds")
    rule_hash = hashlib.sha256(args.rule_file.read_bytes()).hexdigest()

    audit_path = args.rule_file.parent / "validation_operator_audit.json"
    if not audit_path.is_file():
        raise SystemExit(f"Missing validation audit: {audit_path}")
    audit = json.loads(audit_path.read_text())
    if audit.get("weights_sha256") != rule_hash:
        raise SystemExit("Validation audit does not match the current rule weights")
    if not isinstance(audit.get("inputs"), dict) or not audit["inputs"]:
        raise SystemExit("Missing input fingerprints in the validation audit")
    records = audit.get("records", [])
    if len(records) != 14:
        raise SystemExit(f"Expected 14 held-out operator records, found {len(records)}")
    expected = {(p, k) for p in range(2) for k in (3, 15, 75, 175, 275, 425, 500)}
    if {(r.get("point"), r.get("step")) for r in records} != expected:
        raise SystemExit("Invalid held-out parameter/time audit coverage")

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
        summary_path = args.output / f"validation_{args.rule_file.parent.name}_hprom3_p{point}_steps500/summary.json"
        if not summary_path.is_file():
            raise SystemExit(f"Missing B3 validation run: {summary_path}")
        record = json.loads(summary_path.read_text())
        config = record.get("config", {})
        if config.get("weights_sha256") != rule_hash or config.get("inputs") != audit.get("inputs"):
            raise SystemExit(f"Validation run uses different rule weights or inputs: {summary_path}")
        for key in ("coefficient_error_percent", "online_seconds"):
            if key not in record or not np.isfinite(record[key]):
                raise SystemExit(f"Invalid {key} in {summary_path}")
        validation.append({"point": point, "summary": str(summary_path),
                           "state_error_percent": record.get("state_error_percent"),
                           "coefficient_error_percent": record["coefficient_error_percent"]})

    selection = {
        "accepted": bool(passed),
        "selection_uses_reporting_points": False,
        "rule_file": str(args.rule_file.resolve()),
        "weights_sha256": rule_hash,
        "inputs": audit.get("inputs"),
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
    target = args.selection_file or args.output / "selection.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)
    print(json.dumps(selection, indent=2))
    if not passed:
        raise SystemExit("Case-2+B3 cubature rule failed the held-out operator audit")


if __name__ == "__main__":
    main()
