#!/usr/bin/env python3
"""Measure loaded direct-map inference without modifying any online ROM output.

The benchmark evaluates the complete 501-time-step coefficient trajectory at
each of the four reporting parameters.  It deliberately excludes checkpoint
loading, HDM evaluation, state reconstruction, plotting, and file output.  The
only write is the requested timing summary.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT = Path(__file__).resolve()
PROJECT = SCRIPT.parent.parent
REPOSITORY = PROJECT.parent
for path in (REPOSITORY, PROJECT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from burgers.config import DT, NUM_STEPS
from run_pod_dl_data_driven import _load_pod_dl_model
from run_rom_data_driven import _load_rom_data_driven_model


POINTS = (
    ("verification", 4.875, 0.0225),
    ("offgrid1", 4.560, 0.0190),
    ("offgrid2", 5.190, 0.0260),
    ("extrapolation20pct", 4.000, 0.0330),
)
MODEL_NAMES = {
    "podnn": "data_driven_ann_ntot151_best.pt",
    "poddl": "pod_dl_data_driven_ntot151_best.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--summary-path", type=Path, required=True)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    num_times = NUM_STEPS + 1
    time_values = DT * np.arange(num_times, dtype=np.float32)
    inputs: dict[str, torch.Tensor] = {}
    for key, mu1, mu2 in POINTS:
        x_raw = np.column_stack((
            np.full(num_times, mu1, dtype=np.float32),
            np.full(num_times, mu2, dtype=np.float32),
            time_values,
        ))
        inputs[key] = torch.from_numpy(x_raw).to(device)
    return inputs


def parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_method(
    kind: str,
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    device: torch.device,
    repeats: int,
    warmup: int,
) -> tuple[dict[str, np.ndarray], tuple[int, int]]:
    forward = model if kind == "podnn" else model.predict_q_from_x
    with torch.no_grad():
        for _ in range(warmup):
            for x_raw in inputs.values():
                _ = forward(x_raw)
        synchronize(device)

        output = forward(next(iter(inputs.values())))
        if output.ndim != 2 or not bool(torch.isfinite(output).all()):
            raise RuntimeError(f"{kind} produced an invalid coefficient trajectory: shape={tuple(output.shape)}")
        output_shape = tuple(int(v) for v in output.shape)

        samples: dict[str, np.ndarray] = {}
        for point_key, x_raw in inputs.items():
            elapsed = np.empty(repeats, dtype=np.float64)
            for rep in range(repeats):
                synchronize(device)
                start = time.perf_counter()
                _ = forward(x_raw)
                synchronize(device)
                elapsed[rep] = time.perf_counter() - start
            samples[point_key] = elapsed
    return samples, output_shape


def write_summary(
    path: Path,
    campaign_root: Path,
    device: torch.device,
    repeats: int,
    warmup: int,
    records: dict[str, dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "benchmark: repeated loaded direct-map forward inference",
        f"campaign_root: {campaign_root}",
        f"device: {device}",
        f"repeats_per_point: {repeats}",
        f"warmup_per_point: {warmup}",
        f"time_steps_per_trajectory: {NUM_STEPS + 1}",
        "timed_operation: coefficient prediction only; excludes checkpoint loading, state reconstruction, HDM, and file I/O",
    ]
    for kind in ("podnn", "poddl"):
        record = records[kind]
        samples = record["samples"]
        assert isinstance(samples, dict)
        all_samples = np.concatenate(list(samples.values()))
        lines += [
            "",
            f"{kind}_model_path: {record['model_path']}",
            f"{kind}_trainable_parameters: {record['parameters']}",
            f"{kind}_output_shape: {record['output_shape']}",
            f"{kind}_all_points_mean_inference_time_s: {float(np.mean(all_samples)):.12e}",
            f"{kind}_all_points_std_inference_time_s: {float(np.std(all_samples, ddof=1)):.12e}",
            f"{kind}_all_points_min_inference_time_s: {float(np.min(all_samples)):.12e}",
            f"{kind}_all_points_max_inference_time_s: {float(np.max(all_samples)):.12e}",
        ]
        for point_key, _, _ in POINTS:
            point_samples = samples[point_key]
            lines += [
                f"{kind}_{point_key}_mean_inference_time_s: {float(np.mean(point_samples)):.12e}",
                f"{kind}_{point_key}_std_inference_time_s: {float(np.std(point_samples, ddof=1)):.12e}",
            ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2 to estimate a standard deviation")
    if args.warmup < 0:
        raise ValueError("--warmup must be nonnegative")
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    campaign_root = args.campaign_root.resolve()
    models_dir = campaign_root / "Stage3" / "models"
    inputs = make_inputs(device)

    records: dict[str, dict[str, object]] = {}
    for kind, filename in MODEL_NAMES.items():
        model_path = models_dir / filename
        if kind == "podnn":
            model, _, _ = _load_rom_data_driven_model(str(model_path), device)
        else:
            model, _, _ = _load_pod_dl_model(str(model_path), device)
        samples, output_shape = benchmark_method(kind, model, inputs, device, args.repeats, args.warmup)
        records[kind] = {
            "model_path": model_path,
            "parameters": parameter_count(model),
            "output_shape": output_shape,
            "samples": samples,
        }
        all_samples = np.concatenate(list(samples.values()))
        print(
            f"[direct-timing] {kind}: mean={float(np.mean(all_samples)):.6e} s "
            f"std={float(np.std(all_samples, ddof=1)):.6e} s "
            f"params={parameter_count(model)}"
        )

    write_summary(args.summary_path.resolve(), campaign_root, device, args.repeats, args.warmup, records)
    print(f"[direct-timing] summary: {args.summary_path.resolve()}")


if __name__ == "__main__":
    main()
