#!/usr/bin/env python3
"""Print a terminal summary table for HPROM-family runs.

The table is modeled after the paper-style summary:
  Method | n | n_bar | N_e | max RE(%) | mean time (s) | speedup

It reads method summary files from Results/ and does not retrain anything.
"""

from __future__ import annotations

import argparse
import ast
import math
import re
from pathlib import Path


DEFAULT_POINTS = "4.56,0.019;4.75,0.020;5.19,0.026"


def parse_points(text: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for chunk in str(text).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 2:
            raise ValueError(
                f"Invalid point '{chunk}'. Use format: mu1,mu2;mu1,mu2;..."
            )
        points.append((float(parts[0]), float(parts[1])))
    if not points:
        raise ValueError("No test points provided.")
    return points


def parse_key_value_summary(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("["):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def to_float(val: str | None) -> float:
    if val is None:
        return math.nan
    v = str(val).strip()
    if not v or v.upper() == "N/A":
        return math.nan
    try:
        return float(v)
    except ValueError:
        return math.nan


def to_int(val: str | None) -> int | None:
    f = to_float(val)
    if not math.isfinite(f):
        return None
    return int(round(f))


def parse_list_ints(val: str | None) -> list[int]:
    if val is None:
        return []
    txt = str(val).strip()
    if not txt or txt.upper() == "N/A":
        return []
    try:
        obj = ast.literal_eval(txt)
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(obj, (list, tuple)):
        return []
    out: list[int] = []
    for x in obj:
        try:
            out.append(int(x))
        except Exception:  # noqa: BLE001
            pass
    return out


def parse_shape_second_dim(val: str | None) -> int | None:
    if val is None:
        return None
    m = re.search(r"\(\s*\d+\s*,\s*(\d+)\s*\)", str(val))
    if not m:
        return None
    return int(m.group(1))


def format_range(vals: list[int]) -> str:
    if not vals:
        return "--"
    lo = min(vals)
    hi = max(vals)
    return str(lo) if lo == hi else f"{lo}--{hi}"


def format_float(x: float, digits: int = 3) -> str:
    if x is None or not math.isfinite(x):
        return "--"
    return f"{x:.{digits}f}"


def summary_filename(method_key: str, mu1: float, mu2: float) -> str:
    return f"{method_key}_summary_mu1_{mu1:.2f}_mu2_{mu2:.3f}.txt"


def extract_dims(method_key: str, meta: dict[str, str]) -> tuple[str, str]:
    if method_key == "hprom":
        n = to_int(meta.get("basis_size"))
        return (str(n) if n is not None else "--", "--")
    if method_key == "hqprom":
        n = to_int(meta.get("n"))
        return (str(n) if n is not None else "--", "--")
    if method_key == "hprom_gpr":
        n = parse_shape_second_dim(meta.get("U_p_shape"))
        nbar = parse_shape_second_dim(meta.get("U_s_shape"))
        return (
            str(n) if n is not None else "--",
            str(nbar) if nbar is not None else "--",
        )
    if method_key == "hprom_dl":
        n = to_int(meta.get("latent_dim"))
        nbar = to_int(meta.get("q_dim"))
        return (
            str(n) if n is not None else "--",
            str(nbar) if nbar is not None else "--",
        )
    if method_key == "local_hprom":
        nlist = parse_list_ints(meta.get("retained_modes_per_cluster"))
        return (format_range(nlist), "--")
    if method_key == "local_hqprom":
        nlist = parse_list_ints(meta.get("retained_modes_per_cluster"))
        return (format_range(nlist), "--")
    if method_key == "local_hprom_gpr":
        n = to_int(meta.get("n_primary"))
        nbar_list = parse_list_ints(meta.get("retained_modes_per_cluster"))
        return (
            str(n) if n is not None else "--",
            format_range(nbar_list),
        )
    return ("--", "--")


def pick_time(meta: dict[str, str]) -> float:
    for key in (
        "total_hprom_time_seconds",
        "total_hqprom_time_seconds",
        "total_local_hprom_time_seconds",
        "total_local_hqprom_time_seconds",
    ):
        val = to_float(meta.get(key))
        if math.isfinite(val):
            return val
    return math.nan


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Print paper-style terminal summary for HPROM-family results."
    )
    p.add_argument("--results-dir", default="Results")
    p.add_argument("--points", default=DEFAULT_POINTS)
    p.add_argument(
        "--show-pointwise",
        action="store_true",
        help="Also print per-point tables.",
    )
    return p


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _fmt_row(cells: list[str]) -> str:
        return " | ".join(cells[i].ljust(widths[i]) for i in range(len(cells)))

    sep = "-+-".join("-" * w for w in widths)
    print(_fmt_row(headers))
    print(sep)
    for r in rows:
        print(_fmt_row(r))


def main() -> int:
    args = build_parser().parse_args()
    results_dir = Path(args.results_dir)
    points = parse_points(args.points)

    methods = [
        ("hprom", "HPROM"),
        ("hqprom", "HQPROM"),
        ("hprom_gpr", "HPROM-GPR"),
        ("hprom_dl", "HPROM-DL"),
        ("local_hprom", "Local HPROM"),
        ("local_hqprom", "Local HQPROM"),
        ("local_hprom_gpr", "Local HPROM-GPR"),
    ]

    hdm_times: list[float] = []
    for mu1, mu2 in points:
        fom_path = results_dir / f"fom_summary_mu1_{mu1:.2f}_mu2_{mu2:.3f}.txt"
        fom_meta = parse_key_value_summary(fom_path)
        t_hdm = to_float(fom_meta.get("total_hdm_time_seconds"))
        if math.isfinite(t_hdm):
            hdm_times.append(t_hdm)
    mean_hdm = sum(hdm_times) / len(hdm_times) if hdm_times else math.nan

    aggregate_rows: list[list[str]] = []
    pointwise_rows: list[list[str]] = []

    for key, label in methods:
        times: list[float] = []
        errs: list[float] = []
        statuses: list[str] = []
        first_meta: dict[str, str] | None = None

        for mu1, mu2 in points:
            s_path = results_dir / summary_filename(key, mu1, mu2)
            meta = parse_key_value_summary(s_path)
            if meta and first_meta is None:
                first_meta = meta

            t = pick_time(meta)
            e = to_float(meta.get("relative_error_percent"))

            ok = math.isfinite(t) and math.isfinite(e)
            statuses.append("ok" if ok else "missing")
            if ok:
                times.append(t)
                errs.append(e)

            if args.show_pointwise:
                pointwise_rows.append(
                    [
                        label,
                        f"({mu1:.2f}, {mu2:.3f})",
                        format_float(t, 4),
                        format_float(e, 4),
                        "ok" if ok else "missing",
                        str(s_path),
                    ]
                )

        if first_meta is None:
            n_txt, nbar_txt = ("--", "--")
            ne_txt = "--"
        else:
            n_txt, nbar_txt = extract_dims(key, first_meta)
            ne = to_int(first_meta.get("num_nonzero_weights"))
            ne_txt = f"{ne}" if ne is not None else "--"

        max_err = max(errs) if errs else math.nan
        mean_time = sum(times) / len(times) if times else math.nan
        speedup = (mean_hdm / mean_time) if (math.isfinite(mean_hdm) and math.isfinite(mean_time) and mean_time > 0.0) else math.nan
        status = "ok" if all(s == "ok" for s in statuses) else "partial"

        aggregate_rows.append(
            [
                label,
                n_txt,
                nbar_txt,
                ne_txt,
                format_float(max_err, 4),
                format_float(mean_time, 4),
                format_float(speedup, 3),
                status,
            ]
        )

    print("\nHPROM Family Summary (3 test points)")
    print(f"Results dir: {results_dir}")
    print(f"Points: {points}")
    print(f"Mean HDM time [s]: {format_float(mean_hdm, 6)}")
    print_table(
        headers=[
            "Method",
            "n",
            "n_bar",
            "N_e",
            "max RE [%]",
            "mean time [s]",
            "speedup",
            "status",
        ],
        rows=aggregate_rows,
    )

    if args.show_pointwise:
        print("\nPointwise details")
        print_table(
            headers=[
                "Method",
                "mu",
                "time [s]",
                "RE [%]",
                "status",
                "summary file",
            ],
            rows=pointwise_rows,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

