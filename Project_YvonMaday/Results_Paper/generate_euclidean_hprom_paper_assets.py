#!/usr/bin/env python3
"""Audit frozen HPROM outputs and regenerate the HPROM-centred paper assets.

No solver, training, or result overwrite occurs here.  State errors are
recomputed from stored q_N and the common Euclidean basis; all reported
coefficient errors use the frozen linear HPROM, never the linear PROM.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
from matplotlib.patches import Rectangle
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from manuscript_plot_style import HDM_COLOR, METHOD_COLORS


PAPER = Path(__file__).resolve().parent
REPO = PAPER.parents[1]
FIGURES = PAPER / "Figures/euclidean_hprom"
TABLES = PAPER / "tables/euclidean_hprom"
BASE = PAPER / "euclidean_hprom_main"
B3_ECM_LOCAL = PAPER / "euclidean_b3_ecm_local"
B3_ECM_DEPLOYMENT = PAPER / "euclidean_b3_ecm_deployment"
B3_ECM_RUNS = sorted(PAPER.glob("euclidean_b3_ecm_sherlock_*"))
if len(B3_ECM_RUNS) != 1:
    raise ValueError(f"Expected one official B3 ECM benchmark, found {B3_ECM_RUNS}")
B3_ECM_SHERLOCK = B3_ECM_RUNS[0]
DESIGN_PATH = REPO / "Project_YvonMaday/euclidean_nested_enrichment_design.json"
BASIS_PATH = PAPER / "MetricStudy/euclidean/Stage1/basis.npy"
UREF_PATH = BASIS_PATH.with_name("u_ref.npy")
POINTS = ((4.875, .0225), (4.560, .0190), (5.190, .0260), (4., .0330))
POINT_KEYS = ("verification", "offgrid1", "offgrid2", "extrapolation20pct")
POINT_LABELS = (r"$\mu^{(v)}$", r"$\mu^{(1)}$", r"$\mu^{(2)}$", r"$\mu^{(3)}$")
VALIDATION_POINTS = ((4.5625, .02625), (5.1875, .01875))
HDM_NAMES = ("mu1_4.875+mu2_0.0225.npy", "mu1_4.56+mu2_0.019.npy",
             "mu1_5.19+mu2_0.026.npy", "mu1_4.0+mu2_0.033.npy")
LEVELS = ("baseline", "lhs8", "lhs12", "lhs18")
LEVEL_LABELS = ("9", "9+8", "9+12", "9+18")
CAMPAIGNS = {"baseline": BASE, **{
    level: PAPER / f"euclidean_hprom_enrichment_{level}" for level in LEVELS[1:]}}
METHODS = ("linear", "case1", "case2", "b3", "case3", "podae", "podnn", "poddl")
ENRICHMENT_METHODS = tuple(method for method in METHODS if method != "b3")
NAMES = {
    "linear": "Linear HPROM", "case1": "HPROM--ANN Case 1",
    "case2": "HPROM--ANN Case 2",
    "b3": r"HPROM--ANN Case 2+$\mathbf B_3$",
    "case3": "HPROM--ANN Case 3", "podae": "HPROM--POD--AE",
    "podnn": "POD--NN--ROM", "poddl": "POD--DL--ROM",
}
BASELINE_NAMES = {**NAMES, "case2": "HPROM--ANN Case 2"}
PLOT_NAMES = {
    "linear": "Linear HPROM", "case1": "HPROM-ANN C1",
    "case2": "HPROM-ANN C2", "b3": r"HPROM-ANN C2+$\mathbf B_3$",
    "case3": "HPROM-ANN C3", "podae": "HPROM-POD-AE",
    "podnn": "POD-NN-ROM", "poddl": "POD-DL-ROM",
}
COLORS = {key: METHOD_COLORS[key if key != "case2" else "case2_n10"]
          for key in METHODS if key != "b3"}
COLORS["b3"] = "#8C564B"
NETWORKS = {
    "case1": "case1_ann_ntot151_best", "case2": "master_ann_mu_t_to_qtot_ntot151_best",
    "case3": "case3_ann_ntot151_best", "podae": "prom_pod_ae_ntot151_best",
    "poddl": "pod_dl_data_driven_ntot151_best",
}
FLAT = {
    "case1": ("HPROM_ANN_C1", "case1_hprom_ann"),
    "case2": ("HPROM_ANN_C2", "case2_hprom_ann"),
    "case3": ("HPROM_ANN_C3", "case3_hprom_ann"),
    "podae": ("HPROM_POD_AE", "podae_hprom"),
}
plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "text.latex.preamble": r"\usepackage{amsmath}",
                     "axes.titlesize": 11, "axes.labelsize": 10,
                     "legend.fontsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8})


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def kv(path):
    return {key.strip(): value.strip() for key, value in
            (line.split(":", 1) for line in path.read_text().splitlines() if ":" in line)}


def unique(paths):
    paths = sorted(paths)
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one source, found {paths}")
    return paths[0]


def b3_provenance(level):
    """Resolve each validated B3 rule without altering raw results."""
    if level != "baseline":
        raise ValueError("Only the baseline B3 deployment has a validated ECM rule")
    selection_path = B3_ECM_DEPLOYMENT / "selection.json"
    selection = json.loads(selection_path.read_text())
    candidates = [candidate for candidate in selection["candidates"] if candidate["accepted"]]
    if not selection["accepted"] or len(candidates) != 1:
        raise ValueError("No unique validation-selected baseline B3 ECM rule")
    chosen = candidates[0]
    rule_dir = B3_ECM_LOCAL / f"ecm_tol{chosen['tolerance']:g}"
    rule = B3_ECM_DEPLOYMENT / "ecm_weights.npy"
    checks = dict(zip(("minimum_Gram_eigenvalue", "maximum_Gram_eigenvalue",
                       "maximum_gradient_relative_error",
                       "maximum_sampled_to_full_linearized_residual_ratio"),
                      (chosen[k] for k in ("min_gram", "max_gram", "max_gradient_error", "max_linearized_ratio"))))
    thresholds = dict(min_gram=.8, max_gram=1.2, max_gradient_error=.05,
                      max_linearized_ratio=1.05,
                      max_validation_error_ratio=selection["protocol"]["maximum_validation_coefficient_ratio"])
    manifest = dict(accepted=True,
                    selection_uses_reporting_points=selection["selection_uses_reporting_points"],
                    checks=checks, thresholds=thresholds,
                    inputs=selection["protocol"]["inputs"],
                    weights_sha256=chosen["weights_sha256"],
                    held_out_validation=[rule_dir / f"validation_p{p}.json" for p in range(2)])
    protocol_path = B3_ECM_SHERLOCK / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if (protocol["threads"] != 1 or not protocol["reuse_predictor"]
            or protocol["sampled_cells"]["ecm"] != chosen["sampled_cells"]
            or digest(rule) != chosen["weights_sha256"]):
        raise ValueError("Baseline B3 ECM deployment/protocol mismatch")
    if any(r > thresholds["max_validation_error_ratio"]
           for r in chosen["validation_coefficient_ratios"]):
        raise ValueError("Baseline B3 ECM failed its validation-trajectory criterion")
    provenance = [protocol_path, selection_path, B3_ECM_DEPLOYMENT / "fit.json"]
    audit_path = B3_ECM_DEPLOYMENT / "validation_operator_audit.json"
    return manifest, rule, audit_path, provenance


def verify_b3_provenance(level):
    selection, rule, audit_path, paths = b3_provenance(level)
    audit = json.loads(audit_path.read_text())
    if not selection["accepted"] or selection["selection_uses_reporting_points"]:
        raise ValueError(f"B3 rule was not accepted without reporting leakage: {level}")
    if len(selection["held_out_validation"]) != 2:
        raise ValueError(f"Incomplete B3 validation: {level}")
    rule_hash = digest(rule)
    expected_inputs = {str(p.relative_to(REPO)): digest(p) for p in (
        CAMPAIGNS[level] / "Stage3/models/master_ann_mu_t_to_qtot_ntot151_best.pt",
        BASIS_PATH, UREF_PATH, BASE / "Stage2/prom_coeff_dataset_ntot151/meta.json")}
    if (selection["weights_sha256"] != rule_hash or audit["weights_sha256"] != rule_hash
            or audit["inputs"] != expected_inputs or selection["inputs"] != expected_inputs):
        raise ValueError(f"Changed B3 rule/model/basis provenance: {level}")
    records = audit["records"]
    expected = {(p, k) for p in range(2) for k in (3, 15, 75, 175, 275, 425, 500)}
    if len(records) != 14 or {(r["point"], r["step"]) for r in records} != expected:
        raise ValueError(f"Invalid B3 held-out audit coverage: {level}")
    checks = dict(minimum_Gram_eigenvalue=min(r["Gram_eigenvalue_min"] for r in records),
                  maximum_Gram_eigenvalue=max(r["Gram_eigenvalue_max"] for r in records),
                  maximum_gradient_relative_error=max(r["gradient_relative_error"] for r in records),
                  maximum_sampled_to_full_linearized_residual_ratio=max(
                      r["sampled_true_linearized_residual"] / max(r["full_true_linearized_residual"], 1e-30)
                      for r in records))
    for key, value in checks.items():
        np.testing.assert_allclose(value, selection["checks"][key], rtol=0, atol=1e-12)
    bounds = selection["thresholds"]
    for key, value in dict(min_gram=.8, max_gram=1.2, max_gradient_error=.05, max_linearized_ratio=1.05).items():
        if bounds[key] != value:
            raise ValueError(f"Changed B3 acceptance threshold {key}: {level}")
    if (not np.isfinite(list(checks.values())).all()
            or checks["minimum_Gram_eigenvalue"] < bounds["min_gram"]
            or checks["maximum_Gram_eigenvalue"] > bounds["max_gram"]
            or checks["maximum_gradient_relative_error"] > bounds["max_gradient_error"]
            or checks["maximum_sampled_to_full_linearized_residual_ratio"] > bounds["max_linearized_ratio"]):
        raise ValueError(f"B3 operator audit fails its declared thresholds: {level}")
    config = json.loads((B3_ECM_LOCAL / "moments.json").read_text())["protocol"]
    if config["inputs"] != expected_inputs or len(config["training_sources"]) != 9:
        raise ValueError(f"B3 moment fit does not use its baseline nine teachers: {level}")
    teacher_root = BASE / "Stage2/prom_coeff_dataset_ntot151/per_mu"
    expected_sources = {str(p.relative_to(REPO)): digest(p / "qN.npy") for p in teacher_root.iterdir()}
    if config["training_sources"] != expected_sources:
        raise ValueError(f"Changed B3 moment-training trajectories: {level}")
    for point in range(2):
        path = selection["held_out_validation"][point]
        record = json.loads(path.read_text())
        meta = record
        if meta["weights_sha256"] != rule_hash or meta["threads"] != 1:
            raise ValueError(f"Validation trajectory uses a different B3 rule/protocol: {path}")
        if not np.isfinite(record["coefficient_error_percent"]):
            raise ValueError(f"Invalid B3 validation error: {path}")
        qpath = path.with_name(f"validation_p{point}_qN.npy")
        q = np.load(qpath, allow_pickle=False)
        if q.shape != (151,501) or not np.isfinite(q).all():
            raise ValueError(f"Incomplete B3 validation coordinates: {qpath}")
        reference_dir = BASE / "Stage2/prom_coeff_dataset_ntot151_validation2/per_mu"
        reference = unique([p / "qN.npy" for p in reference_dir.iterdir()
                            if np.allclose(np.load(p / "mu.npy").ravel(), VALIDATION_POINTS[point], rtol=0, atol=1e-10)])
        teacher = np.load(reference, allow_pickle=False)
        error = 100 * np.linalg.norm(q-teacher) / np.linalg.norm(teacher)
        np.testing.assert_allclose(error, record["coefficient_error_percent"], rtol=1e-9)
    return selection


@dataclass
class Record:
    q: np.ndarray
    summary: dict
    summary_path: Path
    qpath: Path
    weights_path: Path | None = None
    state: float = 0.
    coefficient: float = 0.
    history: np.ndarray | None = None


def load_record(level, method, point):
    root = CAMPAIGNS[level]
    a, b = POINTS[point]
    tag = f"mu1_{a:.3f}_mu2_{b:.4f}"
    if method == "linear":
        folder = BASE / "Runs/Linear" / f"linear_hprom_{tag}_ntot151"
        path, qpath = folder / "summary.txt", folder / "qN.npy"
        summary = kv(path)
        weights_path = BASE / "Stage2/ecsw/Linear/ecsw_weights_lspg_ntot151.npy"
    elif method in FLAT:
        folder, prefix = FLAT[method]
        path = unique((root / "Runs" / folder).glob(f"{prefix}*{tag}*_summary.txt"))
        qpath = unique((root / "Runs" / folder).glob(f"{prefix}*{tag}*_qN.npy"))
        summary = kv(path)
        weights_path = unique((root / "Stage4/ecm" / folder).glob("*.npy"))
        if summary["solve_backend_effective"] != "hprom":
            raise ValueError(f"Not an HPROM solve: {path}")
    elif method == "b3":
        if level != "baseline":
            raise ValueError("Enriched B3 results are omitted until their ECM rules are built")
        folder = B3_ECM_SHERLOCK / "ecm"
        path, qpath = folder / f"reporting_p{point}.json", folder / f"reporting_p{point}_qN.npy"
        summary = json.loads(path.read_text())
        _, weights_path, _, _ = b3_provenance(level)
    else:
        folder = root / "Runs" / ("PODNN_ROM" if method == "podnn" else "PODDL_ROM")
        folder = unique(folder.glob(f"*{tag}*"))
        path = folder / ("rom_data_driven_summary.txt" if method == "podnn"
                         else "pod_dl_data_driven_summary.txt")
        qpath, summary, weights_path = folder / "qN.npy", kv(path), None
    q = np.load(qpath, allow_pickle=False)
    if q.shape != (151, 501) or not np.isfinite(q).all():
        raise ValueError(f"Invalid coordinate array: {qpath}, {q.shape}")
    if weights_path:
        weights = np.load(weights_path, allow_pickle=False)
        if weights.size != 62500 or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError(f"Invalid positive cubature rule: {weights_path}")
        if not np.any(weights > 0):
            raise ValueError(f"Empty cubature rule: {weights_path}")
    return Record(q, summary, path, qpath, weights_path)


def check_design_and_models():
    design = json.loads(DESIGN_PATH.read_text())
    basis_hash, uref_hash = digest(BASIS_PATH), digest(UREF_PATH)
    weights_hash = digest(BASE / "Stage2/ecsw/Linear/ecsw_weights_lspg_ntot151.npy")
    datasets = {}
    model_details = {}
    for level in LEVELS:
        root = CAMPAIGNS[level]
        name = "prom_coeff_dataset_ntot151" if level == "baseline" else f"prom_coeff_dataset_ntot151_enriched_{level}"
        dataset = root / "Stage2" / name
        datasets[level] = dataset
        meta = json.loads((dataset / "meta.json").read_text())
        count = 9 if level == "baseline" else 9 + int(level[3:])
        folders = sorted((dataset / "per_mu").iterdir())
        if meta["solve_backend"] != "hprom" or meta["num_traj"] != count or len(folders) != count:
            raise ValueError(f"Incomplete or wrong-backend training dataset: {dataset}")
        for folder in folders:
            q = np.load(folder / "qN.npy", mmap_mode="r", allow_pickle=False)
            if q.shape != (151, 501) or not np.isfinite(q).all():
                raise ValueError(f"Invalid training coordinates: {folder}")
        if level != "baseline":
            config = json.loads((dataset / "construction_config.json").read_text())
            for key, expected in (("basis_sha256", basis_hash), ("u_ref_sha256", uref_hash),
                                  ("ecsw_weights_sha256", weights_hash), ("design_sha256", digest(DESIGN_PATH))):
                if config[key] != expected:
                    raise ValueError(f"Wrong enrichment provenance {key}: {dataset}")
            counts = design["levels"][level]
            np.testing.assert_array_equal(np.load(dataset / "interior_lhs_mu.npy"),
                                          design["ordered_interior_mu"][:counts["interior_count"]])
            np.testing.assert_array_equal(np.load(dataset / "exterior_lhs_mu.npy"),
                                          design["ordered_margin_mu"][:counts["margin_count"]])
        for method, stem in NETWORKS.items():
            checkpoint_path = root / "Stage3/models" / f"{stem}.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if checkpoint.get("dataset_backend") != "hprom" or checkpoint.get("validation_dataset_backend") != "hprom":
                raise ValueError(f"Non-HPROM learning target: {checkpoint_path}")
            summary = kv(root / "Stage3" / f"{stem}_summary.txt")
            if int(summary["train_samples"]) != count * 501 or int(summary["val_samples"]) != 1002:
                raise ValueError(f"Wrong learning sample counts: {checkpoint_path}")
            expected_activation = "gelu" if method == "podae" else "silu"
            if summary["activation"].lower() != expected_activation:
                raise ValueError(f"Unexpected activation: {checkpoint_path}")
            if level != "baseline":
                baseline_summary = model_details["baseline", method]
                if summary["trainable_parameters"] != baseline_summary["trainable_parameters"]:
                    raise ValueError(f"Enrichment changed network size: {checkpoint_path}")
            model_details[level, method] = summary
        if level == "baseline":
            verify_b3_provenance(level)
    for level in LEVELS:
        source = datasets["baseline"] if level == "baseline" else datasets["lhs18"]
        for folder in (datasets[level] / "per_mu").iterdir():
            if level == "baseline":
                continue
            if (datasets["baseline"] / "per_mu" / folder.name).is_dir():
                source_folder = datasets["baseline"] / "per_mu" / folder.name
            else:
                source_folder = source / "per_mu" / folder.name
            for name in ("mu.npy", "t.npy", "qN.npy"):
                if digest(folder / name) != digest(source_folder / name):
                    raise ValueError(f"Nested trajectory is not an exact copy: {folder / name}")
    print("Verified 17/21/27 nested HPROM targets, 20 HPROM-trained checkpoints, "
          "and the baseline ECM B3 rule.", flush=True)
    return model_details


def state_errors(q, projected, orthogonal_sq, truth_sq):
    """Use Pythagoras in the common Euclidean basis, without full reconstruction."""
    error_sq = orthogonal_sq + np.sum((q - projected) ** 2, axis=0)
    trajectory = float(100 * np.sqrt(error_sq.sum() / truth_sq.sum()))
    history = 100 * np.sqrt(error_sq / truth_sq)
    return trajectory, history


def recompute(records):
    basis = np.load(BASIS_PATH, allow_pickle=False)
    uref = np.load(UREF_PATH, allow_pickle=False).reshape(-1)
    np.testing.assert_allclose(basis.T @ basis, np.eye(151), atol=1e-10, rtol=0)
    truth_cuts = {}
    for point, name in enumerate(HDM_NAMES):
        path = REPO / "Project_YvonMaday/Results/param_snaps" / name
        truth = np.load(path, mmap_mode="r", allow_pickle=False)
        if truth.shape != (125000, 501):
            raise ValueError(f"Invalid HDM trajectory: {path}")
        projected = np.empty((151, 501))
        orthogonal_sq, truth_sq = np.empty(501), np.empty(501)
        for first in range(0, 501, 32):
            sl = slice(first, min(first + 32, 501))
            block = np.asarray(truth[:, sl], dtype=np.float64)
            centered = block - uref[:, None]
            projected[:, sl] = basis.T @ centered
            remainder = centered - basis @ projected[:, sl]
            orthogonal_sq[sl] = np.sum(remainder ** 2, axis=0)
            truth_sq[sl] = np.sum(block ** 2, axis=0)
        linear = records["baseline", "linear", point].q
        for level in LEVELS:
            for method in (METHODS if level == "baseline" else ENRICHMENT_METHODS):
                record = records[level, method, point]
                record.state, record.history = state_errors(record.q, projected, orthogonal_sq, truth_sq)
                record.coefficient = float(100 * np.linalg.norm(record.q - linear) / np.linalg.norm(linear))
                reported = float(record.summary["state_error_percent"] if method == "b3"
                                 else record.summary["relative_error_percent"])
                np.testing.assert_allclose(record.state, reported, rtol=2e-7, atol=1e-8,
                                           err_msg=f"State mismatch: {record.summary_path}")
                if method == "b3":
                    config = record.summary if level == "baseline" else record.summary["config"]
                    if config["weights_sha256"] != digest(record.weights_path) or config["threads"] != 1:
                        raise ValueError(f"B3 rule/protocol mismatch: {record.summary_path}")
                    if record.summary["min_rank"] != 3 or record.summary["max_rank"] != 3:
                        raise ValueError(f"B3 did not attain rank three: {record.summary_path}")
                    if level != "baseline" and (config["initialization"] != "known_linear"
                                                 or not config.get("reuse_predictor")):
                        raise ValueError(f"Unexpected B3 initial representation: {record.summary_path}")
                    np.testing.assert_allclose(record.q[:,0], projected[:,0], rtol=0, atol=1e-9,
                                               err_msg=f"B3 did not use the known full-space initial state: {record.qpath}")
                    selection, _, _, _ = b3_provenance(level)
                    if level != "baseline" and config["inputs"] != selection["inputs"]:
                        raise ValueError(f"B3 reporting inputs differ from validation: {record.summary_path}")
                    for key, value in selection["inputs"].items():
                        if digest(REPO / key) != value:
                            raise ValueError(f"Changed B3 input {key}: {record.summary_path}")
                    np.testing.assert_allclose(record.coefficient, record.summary["coefficient_error_percent"], rtol=1e-9)
        ix = 125 * 250 + np.arange(250)
        iy = np.arange(250) * 250 + 125
        truth_cuts[point] = (np.asarray(truth[ix][:, [100, 300, 500]]),
                             np.asarray(truth[iy][:, [100, 300, 500]]))
        print(f"Recomputed all state/coordinate errors: {POINT_LABELS[point]}", flush=True)
    return basis, uref, truth_cuts


def table(filename, columns, header, rows):
    lines = [r"\begin{tabular}{" + columns + "}", r"\toprule", header + r" \\", r"\midrule"]
    for row in rows:
        lines.append(r"\midrule" if row is None else " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / filename).write_text("\n".join(lines) + "\n")


def grouped(values):
    return [f"{x:.3f}" for x in (*values[:3], np.mean(values[:3]), values[3])]


def baseline_online_dimension(method):
    if method == "linear":
        return "151"
    if method == "b3":
        return "13"  # Ten primary coordinates plus three correction amplitudes.
    if method in ("case1", "case2", "case3", "podae"):
        return "10"
    return "--"  # Not applicable: direct maps have no residual solve.


def training_baseline_rows(details):
    rows = []
    for method in NETWORKS:
        if method == "poddl":
            rows.append(None)
            master = details["baseline", "case2"]
            rows.append([NAMES["podnn"], "--", "(256,512,512,256), SiLU",
                         master["trainable_parameters"],
                         f"{float(master['train_rel_frob_percent']):.3f}",
                         f"{float(master['val_rel_frob_percent']):.3f}"])
        summary = details["baseline", method]
        values = [summary["trainable_parameters"],
                  f"{float(summary['train_rel_frob_percent']):.3f}",
                  f"{float(summary['val_rel_frob_percent']):.3f}"]
        if method == "podae":
            encoder = summary["hidden_dims"].replace(" ", "")
            widths = [width.strip() for width in encoder.strip("()").split(",")]
            decoder = "(" + ",".join(reversed(widths)) + ")"
            rows.append([NAMES[method], baseline_online_dimension(method), r"$\mathcal E$: " + encoder, *values])
            rows.append(["", "", r"$\mathcal D$: " + decoder + "; GELU", "", "", ""])
        elif method == "poddl":
            encoder = summary["encoder_hidden_dims"].replace(" ", "")
            decoder = summary["decoder_hidden_dims"].replace(" ", "")
            dynamics = summary["dynamics_hidden_dims"].replace(" ", "")
            rows.append([NAMES[method], baseline_online_dimension(method), r"$\mathcal E$: " + encoder, *values])
            rows.append(["", "", r"$\mathcal D$: " + decoder, "", "", ""])
            rows.append(["", "", r"$\mathcal G_z$: " + dynamics + "; SiLU", "", "", ""])
        else:
            rows.append([NAMES[method], baseline_online_dimension(method),
                         "(256,512,512,256), SiLU", *values])
    return rows


def summaries(records, details):
    header = r"$\mu^{(v)}$ & $\mu^{(1)}$ & $\mu^{(2)}$ & Mean & $\mu^{(3)}$"
    for level in LEVELS:
        for metric in ("state", "coefficient"):
            rows = []
            for method in (METHODS if level == "baseline" else ENRICHMENT_METHODS):
                if method == "podnn": rows.append(None)
                values = grouped([getattr(records[level, method, p], metric) for p in range(4)])
                name = BASELINE_NAMES[method] if level == "baseline" else NAMES[method]
                rows.append([name, baseline_online_dimension(method), *values])
                if level != "baseline" and metric == "state" and method == "case2":
                    rows.append([BASELINE_NAMES["b3"], baseline_online_dimension("b3"),
                                 *(["--"] * 5)])
            columns = "lr|rrrr|r"
            prefix = r"Model & Online size & "
            table(f"{level}_{metric}.tex", columns, prefix + header, rows)
    for metric in ("state", "coefficient"):
        rows = []
        for method in ENRICHMENT_METHODS:
            if method == "podnn": rows.append(None)
            values = []
            for level in LEVELS:
                v = [getattr(records[level, method, p], metric) for p in range(4)]
                values += [f"{np.mean(v[:3]):.3f}", f"{v[3]:.3f}"]
            if metric == "state" and method == "linear":
                rows.append([NAMES[method], baseline_online_dimension(method), *values[:2],
                             r"\multicolumn{2}{c|}{unchanged}",
                             r"\multicolumn{2}{c|}{unchanged}",
                             r"\multicolumn{2}{c}{unchanged}"])
            else:
                rows.append([NAMES[method], baseline_online_dimension(method), *values])
            if metric == "state" and method == "case2":
                v = [records["baseline", "b3", p].state for p in range(4)]
                rows.append([BASELINE_NAMES["b3"], baseline_online_dimension("b3"),
                             f"{np.mean(v[:3]):.3f}", f"{v[3]:.3f}", *(["--"] * 6)])
        table(f"enrichment_{metric}.tex", "lr|rr|rr|rr|rr",
              r"\multicolumn{2}{c|}{} & \multicolumn{2}{c|}{Baseline 9} & \multicolumn{2}{c|}{$9+8$} & "
              r"\multicolumn{2}{c|}{$9+12$} & \multicolumn{2}{c}{$9+18$} \\ "
              r"Model & Online size & Mean & $\mu^{(3)}$ & Mean & $\mu^{(3)}$ & Mean & $\mu^{(3)}$ & Mean & $\mu^{(3)}$",
              rows)
    table("datasets.tex", "lrrrr|r", "Data & Baseline & Interior & Margin & Total & Validation",
          [[label, "9", str(p//2), str(p//2), str(9+p), "2"]
           for label,p in zip(LEVEL_LABELS,(0,8,12,18))])
    table("training_baseline.tex", "lrlr|rr", r"Model & Online size & Architecture & Parameters & Train (\%) & Val (\%)",
          training_baseline_rows(details))
    rows = []
    for method in NETWORKS:
        if method == "poddl":
            rows.append(None)
            master_values = []
            for level in LEVELS:
                master = details[level, "case2"]
                master_values += [f"{float(master[k]):.3f}" for k in
                                  ("train_rel_frob_percent", "val_rel_frob_percent")]
            rows.append([NAMES["podnn"], "--", *master_values])
        values = []
        for level in LEVELS:
            d = details[level, method]
            values += [f"{float(d[k]):.3f}" for k in ("train_rel_frob_percent", "val_rel_frob_percent")]
        rows.append([NAMES[method], baseline_online_dimension(method), *values])
    table("training_enrichment.tex", "lr|rr|rr|rr|rr",
          r"\multicolumn{2}{c|}{} & \multicolumn{2}{c|}{Baseline 9} & \multicolumn{2}{c|}{$9+8$} & "
          r"\multicolumn{2}{c|}{$9+12$} & \multicolumn{2}{c}{$9+18$} \\ "
          r"Model & Online size & Train (\%) & Val (\%) & Train (\%) & Val (\%) & "
          r"Train (\%) & Val (\%) & Train (\%) & Val (\%)", rows)
    rows = []
    selection, _, _, _ = b3_provenance("baseline")
    weights = np.load(records["baseline","b3",0].weights_path)
    checks = selection["checks"]
    rows.append(["9", str(np.count_nonzero(weights)),
                 f"{checks['minimum_Gram_eigenvalue']:.4f}", f"{checks['maximum_Gram_eigenvalue']:.4f}",
                 f"{checks['maximum_gradient_relative_error']:.4f}",
                 f"{checks['maximum_sampled_to_full_linearized_residual_ratio']:.4f}", "Yes"])
    table("b3_validation.tex", "lr|rrrr|l", r"Data & Cells & $\lambda_{\min}$ & $\lambda_{\max}$ & Gradient error & LS ratio & Accepted", rows)
    rows = []
    for method in METHODS[:6]:
        values = [str(np.count_nonzero(np.load(records["baseline",method,0].weights_path)))]
        values += ([r"--"] * 3 if method == "b3" else
                   [str(np.count_nonzero(np.load(records[level,method,0].weights_path)))
                    for level in LEVELS[1:]])
        rows.append([NAMES[method], baseline_online_dimension(method), *values])
    table("cubature.tex", "lr|r|r|r|r", r"Model & Online size & Baseline 9 & $9+8$ & $9+12$ & $9+18$", rows)
    metric_rows=[]
    for level in LEVELS:
        for method in (METHODS if level == "baseline" else ENRICHMENT_METHODS):
            for p in range(4):
                r=records[level,method,p]
                metric_rows.append(dict(level=level,method=method,point=POINT_KEYS[p],
                                        state_error_percent=r.state,coefficient_error_percent=r.coefficient,
                                        q_source=str(r.qpath.relative_to(PAPER))))
    with (TABLES / "metrics.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(metric_rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(metric_rows)


def save(fig, name):
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def title(point):
    a,b=POINTS[point]
    return POINT_LABELS[point] + rf": $(\mu_1,\mu_2)=({a:.3f},{b:.4f})$"


def limits(values):
    positive = np.concatenate([np.asarray(v).ravel() for v in values])
    positive = positive[np.isfinite(positive) & (positive>0)]
    return 10**np.floor(np.log10(positive.min())-.05), 10**np.ceil(np.log10(positive.max())+.05)


def coefficient_figures(records):
    all_absolute,all_relative=[],[]
    for level in LEVELS:
        for point in range(4):
            ref=records[level,"linear",point].q
            norm=np.linalg.norm(ref,axis=1)
            for method in (METHODS[1:] if level == "baseline" else ENRICHMENT_METHODS[1:]):
                absolute=np.linalg.norm(records[level,method,point].q-ref,axis=1)
                all_absolute.append(absolute);all_relative.append(100*absolute/norm)
    absolute_lim,relative_lim=limits(all_absolute),limits(all_relative)
    heat_max={kind:max(float(np.max(np.abs(records[level,method,p].q-records[level,"linear",p].q)
                                      * (1 if kind=="absolute" else 100/np.linalg.norm(records[level,"linear",p].q,axis=1)[:,None])))
                       for level in LEVELS
                       for method in (METHODS[1:] if level == "baseline" else ENRICHMENT_METHODS[1:])
                       for p in range(4))
              for kind in ("absolute","relative")}
    for level in LEVELS:
        fig,axes=plt.subplots(2,4,figsize=(15,6.6),sharex=True,sharey="row")
        for p in range(4):
            ref=records[level,"linear",p].q
            for method in (METHODS[1:] if level == "baseline" else ENRICHMENT_METHODS[1:]):
                absolute=np.linalg.norm(records[level,method,p].q-ref,axis=1)
                for row,v in enumerate((absolute,100*absolute/np.linalg.norm(ref,axis=1))):
                    axes[row,p].plot(np.arange(1,152),np.ma.masked_less_equal(v,0),color=COLORS[method],lw=1,label=PLOT_NAMES[method])
            axes[0,p].set_title(title(p))
            for row in range(2):
                axes[row,p].set_yscale("log");axes[row,p].set_ylim(*(absolute_lim if row==0 else relative_lim))
                axes[row,p].axvline(10,color=".4",ls=":",lw=.8);axes[row,p].grid(True,which="both",alpha=.16)
                axes[row,p].set_xlim(1,151)
            axes[1,p].set_xlabel("coefficient index $i$")
        axes[0,0].set_ylabel(r"$\|q_i-q_i^{\rm lin}\|_2$")
        axes[1,0].set_ylabel(r"relative trajectory error (\%)")
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc="upper center",ncol=4)
        fig.subplots_adjust(top=.83,hspace=.09,wspace=.13)
        save(fig,f"{level}_coefficients")
        for kind in ("absolute","relative"):
            plotted_methods = METHODS[1:] if level == "baseline" else ENRICHMENT_METHODS[1:]
            fig,axes=plt.subplots(len(plotted_methods),4,figsize=(14,1.8*len(plotted_methods)),sharex=True,sharey=True)
            shown_rows = []
            # One shared scale, logarithmic away from zero, keeps small
            # errors visible without clipping zeros or the extrapolation peak.
            threshold = heat_max[kind] / 10000
            intensity = SymLogNorm(linthresh=threshold, linscale=.25,
                                   vmin=0, vmax=heat_max[kind])
            for row,method in enumerate(plotted_methods):
                cmap=LinearSegmentedColormap.from_list(method,["#ffffff",COLORS[method]])
                for p in range(4):
                    ref=records[level,"linear",p].q
                    v=np.abs(records[level,method,p].q-ref)
                    if kind=="relative":v=100*v/np.linalg.norm(ref,axis=1)[:,None]
                    shown = axes[row,p].imshow(v,aspect="auto",origin="lower",extent=(0,25,.5,151.5),norm=intensity,cmap=cmap)
                    axes[row,p].axhline(10.5,color=".35",ls=":",lw=.7)
                    if row==0:axes[row,p].set_title(title(p))
                    if p==0:axes[row,p].set_ylabel(PLOT_NAMES[method]+"\ncoefficient $i$")
                    if row==len(plotted_methods)-1:axes[row,p].set_xlabel("time $t$")
                shown_rows.append(shown)
            fig.subplots_adjust(left=.16,right=.9,hspace=.12,wspace=.05)
            for row, shown in enumerate(shown_rows):
                position = axes[row,-1].get_position()
                colour_axis = fig.add_axes([.914, position.y0, .009, position.height])
                bar = fig.colorbar(shown, cax=colour_axis,
                                   ticks=[0, threshold, heat_max[kind]/100, heat_max[kind]],
                                   format="%.2g")
                bar.ax.tick_params(labelsize=7)
            fig.suptitle(f"{kind.capitalize()} coefficient errors: common logarithmic intensity, linear near zero; $[0,{heat_max[kind]:.3g}]$",y=.96)
            save(fig,f"{level}_heat_{kind}")
    return dict(absolute_log_limits=list(absolute_lim),relative_log_limits=list(relative_lim),
                heatmap_maxima=heat_max,heatmap_scale="symlog, nonnegative, linear below maximum/10000")


def solution_figures(records,basis,uref,truth_cuts):
    indices=(125*250+np.arange(250),np.arange(250)*250+125)
    cut_values=[]
    for level in LEVELS:
        for p in range(4):
            cut_values.extend(truth_cuts[p])
            for method in (METHODS if level == "baseline" else ENRICHMENT_METHODS):
                for idx in indices:
                    cut_values.append(uref[idx,None]+basis[idx]@records[level,method,p].q[:,[100,300,500]])
    ymin=min(float(v.min()) for v in cut_values);ymax=max(float(v.max()) for v in cut_values)
    margin=.04*(ymax-ymin);ylim=(min(0,ymin-margin),ymax+margin)
    for level in LEVELS:
        plotted_methods = METHODS if level == "baseline" else ENRICHMENT_METHODS
        fig,axes=plt.subplots(4,2,figsize=(14,12),sharex=True,sharey=True)
        for p in range(4):
            for col,idx in enumerate(indices):
                ax=axes[p,col]
                for k,alpha in enumerate((.23,.43,1.)):
                    ax.plot(np.linspace(.2,99.8,250),truth_cuts[p][col][:,k],color=HDM_COLOR,lw=1.8,alpha=alpha,label="HDM" if k==2 else None)
                    for method in plotted_methods:
                        cut=uref[idx]+basis[idx]@records[level,method,p].q[:,[100,300,500][k]]
                        ax.plot(np.linspace(.2,99.8,250),cut,color=COLORS[method],lw=1.1,alpha=alpha,label=PLOT_NAMES[method] if k==2 else None)
                ax.set_title(title(p)+("; $u_x(x,y_{\rm mid})$" if col==0 else "; $u_x(x_{\rm mid},y)$"))
                ax.set_ylim(*ylim);ax.set_xlim(0,100);ax.grid(alpha=.2);ax.set_ylabel("$u_x$")
                if p==3:ax.set_xlabel("$x$" if col==0 else "$y$")
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc="upper center",ncol=4)
        fig.subplots_adjust(top=.9,hspace=.34,wspace=.06)
        save(fig,f"{level}_solutions")


def enrichment_comparison_figure(records, metric):
    fills=("#9ecae9","#fdd0a2","#9dd9d2","#a1d99b")
    edges=("#376795","#e6550d","#258f83","#2b7a2b")
    fig,axes=plt.subplots(1,2,figsize=(15,5.8),sharey=True)
    methods=ENRICHMENT_METHODS[1:]
    positions=np.arange(len(methods),dtype=float)
    if metric == "state":
        positions[2:] += 1  # Reserve a full category between Cases 2 and 3 for baseline B3.
    for level,label,fill,edge,offset in zip(LEVELS,LEVEL_LABELS,fills,edges,(-.3,-.1,.1,.3)):
        for col in range(2):
            values=[]
            for method in methods:
                v=[getattr(records[level,method,p],metric) for p in range(4)]
                values.append(np.mean(v[:3]) if col==0 else v[3])
            axes[col].bar(positions+offset,values,.18,color=fill,edgecolor=edge,label=label)
    maxvalue=0
    for col in range(2):
        baseline=[getattr(records["baseline","linear",p],metric) for p in range(4)]
        reference=np.mean(baseline[:3]) if col==0 else baseline[3]
        axes[col].axhline(reference,color="black",lw=1.1,zorder=4,label="Linear HPROM")
        if metric == "state":
            b3=[records["baseline","b3",p].state for p in range(4)]
            b3_value=np.mean(b3[:3]) if col==0 else b3[3]
            axes[col].bar(2.0,b3_value,.18,color=fills[0],edgecolor=edges[0],lw=1.0)
            tick_positions=np.insert(positions,2,2.0)
            tick_labels=[PLOT_NAMES[m] for m in ("case1","case2","b3","case3","podae","podnn","poddl")]
        else:
            tick_positions=positions
            tick_labels=[PLOT_NAMES[m] for m in methods]
        axes[col].axvline(4.5 if metric == "state" else 3.5,color=".4",ls="--",lw=.9)
        axes[col].set_xticks(tick_positions,tick_labels,rotation=32 if metric == "state" else 28,ha="right")
        axes[col].grid(axis="y",alpha=.2)
        maxvalue=max(maxvalue,max(p.get_height() for p in axes[col].patches))
    for ax in axes:
        if metric == "state":
            ax.set_yscale("symlog",linthresh=1.0,linscale=1.0,base=10)
            ticks=(0,.5,1,2,5,10)
            ax.set_yticks(ticks,[f"{tick:g}" for tick in ticks])
        ax.set_ylim(0,1.1*maxvalue)
    axes[0].set_title(r"In-domain mean: $\mu^{(v)},\mu^{(1)},\mu^{(2)}$")
    axes[1].set_title(r"Extrapolation: $\mu^{(3)}$")
    axes[0].set_ylabel(("state error vs HDM" if metric=="state" else "coordinate error vs linear HPROM")+r" (\%)")
    axes[0].legend(ncol=2);fig.subplots_adjust(bottom=.31 if metric == "state" else .27,wspace=.06)
    save(fig,f"enrichment_{metric}")


def comparison_figures(records):
    for metric in ("state","coefficient"):
        enrichment_comparison_figure(records,metric)
    baseline_history_figure(records)


def baseline_history_figure(records):
    histories = [np.asarray(records["baseline", method, point].history)
                 for point in range(4)
                 for method in ("linear", "case1", "case2", "b3", "case3")]
    if not all(np.isfinite(values).all() and (values > 0).all() for values in histories):
        raise ValueError("Logarithmic state-error history requires finite positive values")
    history_min = min(float(values.min()) for values in histories)
    history_max = max(float(values.max()) for values in histories)
    fig,axes=plt.subplots(1,4,figsize=(15,3.7),sharex=True,sharey=True)
    for p in range(4):
        for method in ("linear","case1","case2","b3","case3"):
            axes[p].plot(np.linspace(0,25,501),records["baseline",method,p].history,
                         color=COLORS[method],label=PLOT_NAMES[method],lw=1)
        axes[p].set_title(title(p));axes[p].set_xlabel("time $t$");axes[p].grid(alpha=.2)
    for ax in axes.ravel():
        ax.set_yscale("log")
        ax.set_ylim(.8 * history_min, 1.2 * history_max)
    axes[0].set_ylabel(r"state error (\%)")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=5)
    fig.subplots_adjust(top=.78,wspace=.08)
    save(fig,"b3_histories")


def sampling_figures():
    design=json.loads(DESIGN_PATH.read_text())
    baseline=np.array([(a,b) for a in (4.25,4.875,5.5) for b in (.015,.0225,.03)])
    fig,axes=plt.subplots(2,2,figsize=(12,10),sharex=True,sharey=True)
    for ax,level,label in zip(axes.ravel(),LEVELS,LEVEL_LABELS):
        ax.add_patch(Rectangle((4.25,.015),1.25,.015,fill=False,edgecolor=".3",ls="--"))
        ax.scatter(*baseline.T,color="black",s=35,label="baseline linear HPROM (9)")
        if level!="baseline":
            counts=design["levels"][level]
            for role,key,n,color in (("interior","ordered_interior_mu",counts["interior_count"],"#258f83"),
                                      ("margin","ordered_margin_mu",counts["margin_count"],"#e6550d")):
                v=np.array(design[key][:n]);ax.scatter(*v.T,color=color,s=35,label=f"{role} linear HPROM ({n})")
        validation=np.array(design["parameter_validation_mu"])
        ax.scatter(*validation.T,color="#d18b20",marker="D",s=45,label="regression validation (2)")
        ax.scatter(*np.array(POINTS).T,color="#D62728",marker="*",s=90,label="online reporting (4)")
        ax.set_title(f"Training trajectories: ${label}$");ax.set_xlim(3.9,5.85);ax.set_ylim(.0108,.0342)
        ax.set_xlabel(r"$\mu_1$");ax.set_ylabel(r"$\mu_2$");ax.grid(alpha=.15)
        offsets=((6,-14),(6,-14),(6,6),(7,-2))
        for point,point_label,offset in zip(POINTS,POINT_LABELS,offsets):
            ax.annotate(point_label,point,textcoords="offset points",xytext=offset,fontsize=9)
    handles,labels=axes[-1,-1].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",ncol=3);fig.subplots_adjust(top=.88,hspace=.22)
    save(fig,"sampling")


def timing_table(records):
    protocol=json.loads((BASE/"timing/online_thread_protocol.json").read_text())
    hdm=json.loads((BASE/"timing/hdm/hdm_timing.json").read_text())
    direct=kv(BASE/"timing/direct_inference_repeat10_summary.txt")
    if (protocol["hdm_threads"],protocol["linear_hprom_threads"],protocol["learned_intrusive_online_threads"],
        protocol["case2_b3_online_threads"],protocol["direct_inference_threads"])!=(24,24,1,1,1):
        raise ValueError("Unexpected baseline timing protocol")
    if hdm["hostname"] != protocol["hostname"] or hdm["threads"] != 24:
        raise ValueError("HDM timing hardware/protocol mismatch")
    if int(direct["numerical_threads"]) != 1 or int(direct["repeats_per_point"]) != 10:
        raise ValueError("Unexpected direct inference timing protocol")
    ecm_protocol=json.loads((B3_ECM_SHERLOCK/"protocol.json").read_text())
    ecm_environment=json.loads((B3_ECM_SHERLOCK/"environment.json").read_text())
    if (ecm_protocol["hostname"]!=protocol["hostname"] or ecm_protocol["threads"]!=1
            or any(pool["num_threads"] != 1 for pool in ecm_environment["threadpools"])):
        raise ValueError("B3 ECM timing hardware/protocol mismatch")
    full_rule = np.load(records["baseline", "linear", 0].weights_path, allow_pickle=False)
    rows = [["HDM", "125{,}000", f"{full_rule.size:,}".replace(",", "{,}"),
             "--", f"{hdm['mean_in_domain_seconds']:.3f}", "--"], None]
    for method in METHODS:
        if method in ("podnn","poddl"):
            times=[float(direct[f"{method}_{key}_mean_inference_time_s"]) for key in POINT_KEYS[:3]]
        else:
            key="online_seconds" if method=="b3" else "online_solve_elapsed_s"
            times=[float(records["baseline",method,p].summary[key]) for p in range(3)]
        mean=float(np.mean(times));error=np.mean([records["baseline",method,p].state for p in range(3)])
        if method=="podnn":rows.append(None)
        if method in ("podnn", "poddl"):
            elements = "--"
            time_text = f"{mean:.5f}" + r"$^\dagger$"
            speedup = f"{hdm['mean_in_domain_seconds']/mean:,.2f}".replace(",", "{,}") + r"$^\dagger$"
        else:
            weights = np.load(records["baseline", method, 0].weights_path, allow_pickle=False)
            elements = f"{np.count_nonzero(weights):,}".replace(",", "{,}")
            time_text, speedup = f"{mean:.3f}", f"{hdm['mean_in_domain_seconds']/mean:.2f}"
        rows.append([BASELINE_NAMES[method], baseline_online_dimension(method), elements,
                     f"{error:.3f}", time_text, speedup])
    table("timing_baseline.tex", "lrr|rrr",
          r"Model & Online size & No. elements & Mean $e_u$ (\%) & Wall-clock time (s) & Speedup factor",
          rows)


def main():
    FIGURES.mkdir(parents=True,exist_ok=True);TABLES.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    with threadpool_limits(2):
        details=check_design_and_models()
        records={(level,method,p):load_record(level,method,p)
                 for level in LEVELS
                 for method in (METHODS if level == "baseline" else ENRICHMENT_METHODS)
                 for p in range(4)}
        basis,uref,truth_cuts=recompute(records)
        summaries(records,details)
        timing_table(records)
        scale=coefficient_figures(records)
        solution_figures(records,basis,uref,truth_cuts)
        comparison_figures(records)
        sampling_figures()
    fingerprints={str(p.relative_to(REPO)):digest(p) for p in (
        BASIS_PATH,UREF_PATH,DESIGN_PATH,Path(__file__),PAPER/"manuscript_plot_style.py")}
    for level, root in CAMPAIGNS.items():
        for stem in NETWORKS.values():
            for p in (root/"Stage3/models"/f"{stem}.pt", root/"Stage3"/f"{stem}_summary.txt"):
                fingerprints[str(p.relative_to(REPO))] = digest(p)
        if level == "baseline":
            _, _, audit_path, paths = b3_provenance(level)
            for p in (*paths, audit_path):
                fingerprints[str(p.relative_to(REPO))] = digest(p)
        if level == "baseline":
            selection, _, _, _ = b3_provenance(level)
            for path in selection["held_out_validation"]:
                for p in (path, path.with_name(path.stem + "_qN.npy")):
                    fingerprints[str(p.relative_to(REPO))] = digest(p)
    for name in HDM_NAMES:
        p = REPO/"Project_YvonMaday/Results/param_snaps"/name
        fingerprints[str(p.relative_to(REPO))] = digest(p)
    for record in records.values():
        for p in (record.qpath,record.summary_path,record.weights_path):
            if p:fingerprints[str(p.relative_to(REPO))]=digest(p)
    timing_sources = [BASE/"timing/online_thread_protocol.json", BASE/"timing/hdm/hdm_timing.json",
                      BASE/"timing/direct_inference_repeat10_summary.txt",
                      B3_ECM_DEPLOYMENT/"ecm_weights.npy"]
    timing_sources += [B3_ECM_SHERLOCK/"ecm"/f"reporting_p{p}{suffix}"
                       for p in range(4)
                       for suffix in (".json","_qN.npy")]
    for p in timing_sources:
        fingerprints[str(p.relative_to(REPO))] = digest(p)
    audit=dict(campaigns={k:str(v.relative_to(PAPER)) for k,v in CAMPAIGNS.items()},
               coefficient_reference="frozen 151-coordinate linear HPROM",state_reference="HDM",
               timing="matched baseline protocol only; direct timings are coefficient inference only",
               b3_rule="full-candidate, validation-selected ECM for the baseline B3 deployment",
               state_recomputation="Euclidean orthogonal decomposition, verified against saved state errors",
               scales=scale,sha256=fingerprints)
    (TABLES/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print("HPROM paper assets complete; raw campaign results were not modified.",flush=True)


if __name__=="__main__":
    main()
