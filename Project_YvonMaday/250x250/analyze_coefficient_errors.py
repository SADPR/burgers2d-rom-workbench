#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Coefficient-error analysis against the linear n_tot-PROM reference.

What this script does
---------------------
For each requested stage and parameter point:
1) Builds q_N trajectories for all models, with linear PROM q_N as reference.
2) Plots per-coefficient absolute and relative errors vs coefficient index.
3) Computes source-of-error decomposition terms for Case 1/2/3 secondary modes.
4) Writes CSV summaries for table-ready reporting.

Reference ("exact") used here
-----------------------------
q_exact := q_N from the linear PROM of size n_tot.

Model error conventions
-----------------------
Let q_ref be linear PROM coefficients and q_mod model coefficients.
Primary (source 1):
    e_q = q_ref,p - q_mod,p
Secondary total:
    e_s,total = q_ref,s - q_mod,s

Case 1 decomposition:
    e_s,total = (q_ref,s - N(q_ref,p)) + (N(q_ref,p) - N(q_mod,p)) + (N(q_mod,p) - q_mod,s)
               = source_2_map + source_3_continuity + consistency

Case 2 decomposition:
    e_s,total = (q_ref,s - M(mu,t)) + (M(mu,t) - q_mod,s)
               = source_2_map + consistency

Case 3 decomposition:
    e_s,total = (q_ref,s - H(q_ref,p,mu,t))
              + (H(q_ref,p,mu,t) - H(q_mod,p,mu,t))
              + (H(q_mod,p,mu,t) - q_mod,s)
              = source_2_map + source_3_continuity + consistency
"""

import argparse
import csv
import os
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent

DEFAULT_POINTS = [
    (4.875, 0.0225),  # verification
    (4.560, 0.0190),  # test 1
    (5.190, 0.0260),  # test 2
]


def set_plot_style():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.0,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.35,
        }
    )


def mu_tag(mu):
    return f"mu1_{mu[0]:.3f}_mu2_{mu[1]:.4f}"


def parse_mu_list(mu_args):
    if not mu_args:
        return list(DEFAULT_POINTS)
    out = []
    for item in mu_args:
        raw = item.strip()
        if "," not in raw:
            raise ValueError(f"Invalid --mu '{item}'. Use format: --mu 4.875,0.0225")
        a, b = raw.split(",", 1)
        out.append((float(a), float(b)))
    return out


class Scaler(nn.Module):
    def __init__(self, mean, std, eps=1e-12):
        super().__init__()
        std = np.maximum(std, eps)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) / self.std


class Unscaler(nn.Module):
    def __init__(self, mean, std, eps=1e-12):
        super().__init__()
        std = np.maximum(std, eps)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, y):
        return y * self.std + self.mean


class CoreMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, out_dim)
        self.act = nn.ELU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        x = self.act(self.fc5(x))
        return self.fc6(x)


class Case1Model(nn.Module):
    def __init__(self, n_p, n_s):
        super().__init__()
        self.scaler = Scaler(np.zeros((1, n_p)), np.ones((1, n_p)))
        self.core = CoreMLP(n_p, n_s)
        self.unscaler = Unscaler(np.zeros((1, n_s)), np.ones((1, n_s)))

    def forward(self, x_raw):
        return self.unscaler(self.core(self.scaler(x_raw)))


class Case2Model(nn.Module):
    def __init__(self, in_dim, n_s):
        super().__init__()
        self.scaler = Scaler(np.zeros((1, in_dim)), np.ones((1, in_dim)))
        self.core = CoreMLP(in_dim, n_s)
        self.unscaler = Unscaler(np.zeros((1, n_s)), np.ones((1, n_s)))

    def forward(self, x_raw):
        return self.unscaler(self.core(self.scaler(x_raw)))


class Case3Model(nn.Module):
    def __init__(self, in_dim, n_s):
        super().__init__()
        self.scaler = Scaler(np.zeros((1, in_dim)), np.ones((1, in_dim)))
        self.core = CoreMLP(in_dim, n_s)
        self.unscaler = Unscaler(np.zeros((1, n_s)), np.ones((1, n_s)))

    def forward(self, x_raw):
        return self.unscaler(self.core(self.scaler(x_raw)))


def load_stage_models(stage, n_p, n_tot):
    if stage == "baseline":
        model_dir = THIS_DIR / "Results" / "Stage3" / "models"
        names = {
            "case1": model_dir / "case1_model.pt",
            "case2": model_dir / "case2_model.pt",
            "case3": model_dir / "case3_model.pt",
        }
    elif stage == "enriched":
        model_dir = (
            THIS_DIR
            / "Results_Enrichment"
            / "Stage3"
            / f"prom_coeff_dataset_ntot{n_tot}_enriched_lhs20"
            / "models"
        )
        names = {
            "case1": model_dir / "case1_model_enriched.pt",
            "case2": model_dir / "case2_model_enriched.pt",
            "case3": model_dir / "case3_model_enriched.pt",
        }
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    out = {}
    for key, path in names.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing model checkpoint for {stage}/{key}: {path}")
        ckpt = torch.load(path, map_location="cpu")
        if key == "case1":
            n_s = int(ckpt["n_s"])
            model = Case1Model(n_p=int(ckpt["n_p"]), n_s=n_s)
        elif key == "case2":
            n_s = int(ckpt["n_s"])
            model = Case2Model(in_dim=int(ckpt["in_dim"]), n_s=n_s)
        else:
            n_s = int(ckpt["n_s"])
            model = Case3Model(in_dim=int(ckpt["in_dim"]), n_s=n_s)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.eval()
        out[key] = model
    return out


def load_basis_and_uref(n_tot):
    basis_path = THIS_DIR / "Results" / "Stage1" / "basis.npy"
    uref_path = THIS_DIR / "Results" / "Stage1" / "u_ref.npy"
    if not basis_path.exists():
        raise FileNotFoundError(f"Missing basis: {basis_path}")
    basis = np.asarray(np.load(basis_path, allow_pickle=False), dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError(f"Basis must be 2D, got shape {basis.shape}")
    if n_tot > basis.shape[1]:
        raise ValueError(f"Requested n_tot={n_tot}, basis has only {basis.shape[1]} modes.")
    basis = basis[:, :n_tot]

    if not uref_path.exists():
        raise FileNotFoundError(f"Missing u_ref: {uref_path}")
    u_ref = np.asarray(np.load(uref_path, allow_pickle=False), dtype=np.float64).reshape(-1)
    if u_ref.size != basis.shape[0]:
        raise ValueError(
            f"u_ref size mismatch: got {u_ref.size}, expected {basis.shape[0]} from basis."
        )
    return basis, u_ref


def linear_run_dir(mu, n_tot):
    return THIS_DIR / "Results" / "Runs" / "Linear" / f"linear_prom_{mu_tag(mu)}_ntot{n_tot}"


def case_snaps_path(stage, case_id, mu, n_p, n_tot):
    if stage == "baseline":
        runs_root = THIS_DIR / "Results" / "Runs"
        run_name = f"case{case_id}_prom_ann_{mu_tag(mu)}_n{n_p}_ntot{n_tot}"
    elif stage == "enriched":
        runs_root = THIS_DIR / "Results_Enrichment" / "Runs"
        run_name = f"case{case_id}_prom_ann_enriched_{mu_tag(mu)}_n{n_p}_ntot{n_tot}"
    else:
        raise ValueError(f"Unsupported stage: {stage}")
    return runs_root / f"Case{case_id}" / f"{run_name}_snaps.npy"


def dd_qn_path(stage, mu, n_tot):
    if stage == "baseline":
        root = THIS_DIR / "Results" / "Runs" / "DataDriven"
        run_name = f"rom_data_driven_{mu_tag(mu)}_ntot{n_tot}"
    elif stage == "enriched":
        root = THIS_DIR / "Results_Enrichment" / "Runs" / "DataDriven"
        run_name = f"rom_data_driven_enriched_{mu_tag(mu)}_ntot{n_tot}"
    else:
        raise ValueError(f"Unsupported stage: {stage}")
    return root / run_name / "qN.npy"


def project_snaps_to_qn(snaps_path, basis, u_ref, cache_projected=True, block_cols=64):
    snaps_path = Path(snaps_path)
    if not snaps_path.exists():
        raise FileNotFoundError(f"Missing ROM snapshots: {snaps_path}")

    cache_path = snaps_path.with_name(snaps_path.stem.replace("_snaps", "_qNproj") + ".npy")
    if cache_projected and cache_path.exists():
        qn = np.asarray(np.load(cache_path, allow_pickle=False), dtype=np.float64)
        return qn

    snaps = np.load(str(snaps_path), mmap_mode="r")
    if snaps.ndim != 2:
        raise ValueError(f"Expected 2D snapshots at {snaps_path}, got shape {snaps.shape}")

    n_tot = basis.shape[1]
    n_t = int(snaps.shape[1])
    qn = np.zeros((n_tot, n_t), dtype=np.float64)
    for j0 in range(0, n_t, block_cols):
        j1 = min(j0 + block_cols, n_t)
        blk = np.asarray(snaps[:, j0:j1], dtype=np.float64)
        qn[:, j0:j1] = basis.T @ (blk - u_ref[:, None])

    if cache_projected:
        np.save(cache_path, qn)
    return qn


def evaluate_model_numpy(model, x_raw_np, batch_size=2048):
    x_raw_np = np.asarray(x_raw_np, dtype=np.float32)
    out = []
    with torch.no_grad():
        for i in range(0, x_raw_np.shape[0], batch_size):
            xb = torch.from_numpy(x_raw_np[i : i + batch_size])
            yb = model(xb).detach().cpu().numpy()
            out.append(yb)
    return np.vstack(out)


def per_mode_abs_rel(q_ref, q_mod, eps=1e-14):
    diff = q_ref - q_mod
    abs_mode = np.linalg.norm(diff, axis=1)
    ref_mode = np.linalg.norm(q_ref, axis=1)
    rel_mode = abs_mode / np.maximum(ref_mode, eps)
    return abs_mode, rel_mode


def frob_abs_rel(a, ref, eps=1e-14):
    a_norm = float(np.linalg.norm(a))
    ref_norm = float(np.linalg.norm(ref))
    rel = a_norm / max(ref_norm, eps)
    return a_norm, rel


def plot_coeff_error_curves(
    out_path,
    n_p,
    n_tot,
    abs_curves,
    rel_curves,
    title,
):
    x = np.arange(1, n_tot + 1)
    model_order = ["Case 1", "Case 2", "Case 3", "Data-driven"]
    colors = {
        "Case 1": "tab:red",
        "Case 2": "tab:blue",
        "Case 3": "tab:green",
        "Data-driven": "tab:orange",
    }

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.0), sharex=True)

    for name in model_order:
        if name not in abs_curves:
            continue
        axes[0].semilogy(x, abs_curves[name], color=colors[name], label=name, linewidth=1.8)
        axes[1].semilogy(x, rel_curves[name], color=colors[name], label=name, linewidth=1.8)

    for ax in axes:
        ax.axvline(n_p + 0.5, color="0.3", linestyle="--", linewidth=1.0)
        ax.grid(True)

    axes[0].set_ylabel(r"$\|e_i\|_2$")
    axes[1].set_ylabel(r"$\|e_i\|_2 / \|q_i^{\mathrm{ref}}\|_2$")
    axes[1].set_xlabel("Coefficient index $i$")
    axes[0].set_title(title)
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_secondary_decomposition(
    out_path,
    n_p,
    n_tot,
    abs_terms,
    rel_terms,
    title,
):
    # Secondary indices in global numbering: n_p+1 .. n_tot
    x = np.arange(n_p + 1, n_tot + 1)
    order = ["Total secondary error", "Source 2 (map)", "Source 3 (continuity)", "Consistency"]
    colors = {
        "Total secondary error": "black",
        "Source 2 (map)": "tab:red",
        "Source 3 (continuity)": "tab:blue",
        "Consistency": "tab:green",
    }

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.6), sharex=True)
    for name in order:
        if name not in abs_terms:
            continue
        axes[0].semilogy(x, abs_terms[name], color=colors[name], label=name, linewidth=1.8)
        axes[1].semilogy(x, rel_terms[name], color=colors[name], label=name, linewidth=1.8)

    axes[0].set_ylabel(r"$\|e_i\|_2$")
    axes[1].set_ylabel(r"$\|e_i\|_2 / \|q_{s,i}^{\mathrm{ref}}\|_2$")
    axes[1].set_xlabel("Secondary coefficient index $i$")
    for ax in axes:
        ax.grid(True)
    axes[0].set_title(title)
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_error_heatmap_grid(
    out_path,
    t_vec,
    err_mats,
    title,
    cbar_label,
):
    model_order = ["Case 1", "Case 2", "Case 3", "Data-driven"]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 8.4),
        sharex=True,
        sharey=True,
    )
    axes = axes.ravel()

    vals = []
    for name in model_order:
        if name in err_mats:
            vals.append(err_mats[name].ravel())
    if not vals:
        raise ValueError("No heatmap data to plot.")
    vals = np.concatenate(vals)
    vmax = float(np.percentile(vals, 99.0))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0

    t0 = float(t_vec[0])
    t1 = float(t_vec[-1])
    n_tot = int(next(iter(err_mats.values())).shape[0])
    extent = [t0, t1, 1, n_tot]

    im = None
    for ax, name in zip(axes, model_order):
        mat = err_mats[name]
        im = ax.imshow(
            mat,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(name)
        ax.grid(False)

    axes[0].set_ylabel("Coefficient index $i$")
    axes[2].set_ylabel("Coefficient index $i$")
    axes[2].set_xlabel("Time $t$")
    axes[3].set_xlabel("Time $t$")
    fig.suptitle(title)
    # Keep a dedicated colorbar axis outside the subplot grid to avoid overlap.
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.08, top=0.90, wspace=0.08, hspace=0.12)
    cax = fig.add_axes([0.90, 0.18, 0.022, 0.65])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(cbar_label)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_stage_qn_dict(stage, mu, n_p, n_tot, basis, u_ref, cache_projected=True):
    q_linear = np.asarray(
        np.load(linear_run_dir(mu, n_tot) / "qN.npy", allow_pickle=False), dtype=np.float64
    )
    q_case1 = project_snaps_to_qn(
        case_snaps_path(stage, 1, mu, n_p, n_tot), basis, u_ref, cache_projected=cache_projected
    )
    q_case2 = project_snaps_to_qn(
        case_snaps_path(stage, 2, mu, n_p, n_tot), basis, u_ref, cache_projected=cache_projected
    )
    q_case3 = project_snaps_to_qn(
        case_snaps_path(stage, 3, mu, n_p, n_tot), basis, u_ref, cache_projected=cache_projected
    )
    q_dd = np.asarray(np.load(dd_qn_path(stage, mu, n_tot), allow_pickle=False), dtype=np.float64)

    for name, q in {
        "Linear": q_linear,
        "Case 1": q_case1,
        "Case 2": q_case2,
        "Case 3": q_case3,
        "Data-driven": q_dd,
    }.items():
        if q.shape != q_linear.shape:
            raise ValueError(
                f"Shape mismatch at stage={stage}, mu={mu}, model={name}: "
                f"got {q.shape}, expected {q_linear.shape}"
            )

    return {
        "Linear": q_linear,
        "Case 1": q_case1,
        "Case 2": q_case2,
        "Case 3": q_case3,
        "Data-driven": q_dd,
    }


def decomposition_case_terms(case_name, mu, t_vec, q_ref, q_mod, models, n_p, n_tot):
    q_ref_p = q_ref[:n_p, :]
    q_ref_s = q_ref[n_p:, :]
    q_mod_p = q_mod[:n_p, :]
    q_mod_s = q_mod[n_p:, :]
    n_s = n_tot - n_p

    mu1 = float(mu[0])
    mu2 = float(mu[1])
    t = np.asarray(t_vec, dtype=np.float64).reshape(-1)
    T = t.size

    if case_name == "Case 1":
        model = models["case1"]
        n_ref = evaluate_model_numpy(model, q_ref_p.T).T
        n_mod = evaluate_model_numpy(model, q_mod_p.T).T
        source2 = q_ref_s - n_ref
        source3 = n_ref - n_mod
        consistency = n_mod - q_mod_s
    elif case_name == "Case 2":
        model = models["case2"]
        x = np.column_stack(
            [
                np.full((T,), mu1, dtype=np.float64),
                np.full((T,), mu2, dtype=np.float64),
                t,
            ]
        )
        m_ref = evaluate_model_numpy(model, x).T
        source2 = q_ref_s - m_ref
        source3 = np.zeros((n_s, T), dtype=np.float64)
        consistency = m_ref - q_mod_s
    elif case_name == "Case 3":
        model = models["case3"]
        x_ref = np.column_stack(
            [
                q_ref_p.T,
                np.full((T,), mu1, dtype=np.float64),
                np.full((T,), mu2, dtype=np.float64),
                t,
            ]
        )
        x_mod = np.column_stack(
            [
                q_mod_p.T,
                np.full((T,), mu1, dtype=np.float64),
                np.full((T,), mu2, dtype=np.float64),
                t,
            ]
        )
        h_ref = evaluate_model_numpy(model, x_ref).T
        h_mod = evaluate_model_numpy(model, x_mod).T
        source2 = q_ref_s - h_ref
        source3 = h_ref - h_mod
        consistency = h_mod - q_mod_s
    else:
        raise ValueError(f"Unsupported case in decomposition: {case_name}")

    total_secondary = q_ref_s - q_mod_s
    source1_primary = q_ref_p - q_mod_p

    return {
        "source1_primary": source1_primary,
        "source2_map": source2,
        "source3_continuity": source3,
        "consistency": consistency,
        "total_secondary": total_secondary,
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser(description="Coefficient error analysis vs linear PROM reference.")
    parser.add_argument("--n-primary", type=int, default=10, help="Primary modes n.")
    parser.add_argument("--n-tot", type=int, default=151, help="Total modes n_tot.")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["baseline", "enriched"],
        help="Stages to process: baseline enriched",
    )
    parser.add_argument(
        "--mu",
        action="append",
        default=None,
        help="Point 'mu1,mu2'. Repeat for multiple points. Defaults to verification+2 tests.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(THIS_DIR / "Figures" / "coeff_errors"),
        help="Output directory for plots and CSVs.",
    )
    parser.add_argument(
        "--no-cache-projected-qn",
        action="store_true",
        help="Disable qN projection caching for ANN runs.",
    )
    args = parser.parse_args()

    n_p = int(args.n_primary)
    n_tot = int(args.n_tot)
    stages = [s.strip().lower() for s in args.stages]
    for s in stages:
        if s not in ("baseline", "enriched"):
            raise ValueError(f"Unsupported stage '{s}'. Use baseline/enriched.")
    mus = parse_mu_list(args.mu)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_projected = not args.no_cache_projected_qn

    set_plot_style()
    basis, u_ref = load_basis_and_uref(n_tot=n_tot)

    model_metrics_rows = []
    decomp_rows = []

    for stage in stages:
        models = load_stage_models(stage=stage, n_p=n_p, n_tot=n_tot)
        for mu in mus:
            q_dict = build_stage_qn_dict(
                stage=stage,
                mu=mu,
                n_p=n_p,
                n_tot=n_tot,
                basis=basis,
                u_ref=u_ref,
                cache_projected=cache_projected,
            )
            q_ref = q_dict["Linear"]

            t_path = linear_run_dir(mu, n_tot) / "t.npy"
            if not t_path.exists():
                raise FileNotFoundError(f"Missing linear t.npy: {t_path}")
            t_vec = np.asarray(np.load(t_path, allow_pickle=False), dtype=np.float64).reshape(-1)
            if t_vec.size != q_ref.shape[1]:
                raise ValueError(
                    f"Time size mismatch at mu={mu}: t has {t_vec.size}, q has {q_ref.shape[1]}."
                )

            abs_curves = {}
            rel_curves = {}
            for model_name in ("Case 1", "Case 2", "Case 3", "Data-driven"):
                q_mod = q_dict[model_name]
                abs_mode, rel_mode = per_mode_abs_rel(q_ref, q_mod)
                abs_curves[model_name] = abs_mode
                rel_curves[model_name] = rel_mode

                diff = q_ref - q_mod
                diff_p = diff[:n_p, :]
                diff_s = diff[n_p:, :]
                ref_p = q_ref[:n_p, :]
                ref_s = q_ref[n_p:, :]
                abs_all, rel_all = frob_abs_rel(diff, q_ref)
                abs_p, rel_p = frob_abs_rel(diff_p, ref_p)
                abs_s, rel_s = frob_abs_rel(diff_s, ref_s)

                model_metrics_rows.append(
                    {
                        "stage": stage,
                        "mu1": f"{mu[0]:.6f}",
                        "mu2": f"{mu[1]:.6f}",
                        "model": model_name,
                        "abs_total": f"{abs_all:.12e}",
                        "rel_total": f"{rel_all:.12e}",
                        "abs_primary": f"{abs_p:.12e}",
                        "rel_primary": f"{rel_p:.12e}",
                        "abs_secondary": f"{abs_s:.12e}",
                        "rel_secondary": f"{rel_s:.12e}",
                    }
                )

            fig_main = out_dir / f"{stage}_{mu_tag(mu)}_coeff_abs_rel_vs_index.png"
            title = (
                f"{stage.capitalize()} vs linear PROM reference at "
                f"$\\mu=({mu[0]:.3f},\\,{mu[1]:.4f})$"
            )
            plot_coeff_error_curves(
                out_path=fig_main,
                n_p=n_p,
                n_tot=n_tot,
                abs_curves=abs_curves,
                rel_curves=rel_curves,
                title=title,
            )

            # Time-resolved error propagation heatmaps (coefficient index vs time).
            abs_heat = {}
            rel_heat = {}
            denom_mode = np.maximum(np.linalg.norm(q_ref, axis=1, keepdims=True), 1e-14)
            for model_name in ("Case 1", "Case 2", "Case 3", "Data-driven"):
                q_mod = q_dict[model_name]
                eabs = np.abs(q_ref - q_mod)
                abs_heat[model_name] = eabs
                rel_heat[model_name] = eabs / denom_mode

            fig_abs_hm = out_dir / f"{stage}_{mu_tag(mu)}_coeff_abs_heatmap_grid.png"
            fig_rel_hm = out_dir / f"{stage}_{mu_tag(mu)}_coeff_rel_heatmap_grid.png"
            plot_error_heatmap_grid(
                out_path=fig_abs_hm,
                t_vec=t_vec,
                err_mats=abs_heat,
                title=(
                    f"{stage.capitalize()} absolute coefficient error heatmaps at "
                    f"$\\mu=({mu[0]:.3f},\\,{mu[1]:.4f})$"
                ),
                cbar_label=r"$|q_i^{\mathrm{ref}}(t)-q_i^{(m)}(t)|$",
            )
            plot_error_heatmap_grid(
                out_path=fig_rel_hm,
                t_vec=t_vec,
                err_mats=rel_heat,
                title=(
                    f"{stage.capitalize()} relative coefficient error heatmaps at "
                    f"$\\mu=({mu[0]:.3f},\\,{mu[1]:.4f})$"
                ),
                cbar_label=r"$|q_i^{\mathrm{ref}}(t)-q_i^{(m)}(t)|/\|q_i^{\mathrm{ref}}\|_2$",
            )

            for case_name in ("Case 1", "Case 2", "Case 3"):
                terms = decomposition_case_terms(
                    case_name=case_name,
                    mu=mu,
                    t_vec=t_vec,
                    q_ref=q_ref,
                    q_mod=q_dict[case_name],
                    models=models,
                    n_p=n_p,
                    n_tot=n_tot,
                )
                q_ref_p = q_ref[:n_p, :]
                q_ref_s = q_ref[n_p:, :]

                # Frobenius summary entries.
                source1_abs, source1_rel = frob_abs_rel(terms["source1_primary"], q_ref_p)
                source2_abs, source2_rel = frob_abs_rel(terms["source2_map"], q_ref_s)
                source3_abs, source3_rel = frob_abs_rel(terms["source3_continuity"], q_ref_s)
                cons_abs, cons_rel = frob_abs_rel(terms["consistency"], q_ref_s)
                total_abs, total_rel = frob_abs_rel(terms["total_secondary"], q_ref_s)

                if case_name == "Case 2":
                    decomp_residual = np.linalg.norm(
                        terms["total_secondary"] - (terms["source2_map"] + terms["consistency"])
                    )
                else:
                    decomp_residual = np.linalg.norm(
                        terms["total_secondary"]
                        - (
                            terms["source2_map"]
                            + terms["source3_continuity"]
                            + terms["consistency"]
                        )
                    )
                decomp_rows.append(
                    {
                        "stage": stage,
                        "mu1": f"{mu[0]:.6f}",
                        "mu2": f"{mu[1]:.6f}",
                        "case": case_name,
                        "source1_primary_abs": f"{source1_abs:.12e}",
                        "source1_primary_rel": f"{source1_rel:.12e}",
                        "source2_map_abs": f"{source2_abs:.12e}",
                        "source2_map_rel": f"{source2_rel:.12e}",
                        "source3_continuity_abs": f"{source3_abs:.12e}",
                        "source3_continuity_rel": f"{source3_rel:.12e}",
                        "consistency_abs": f"{cons_abs:.12e}",
                        "consistency_rel": f"{cons_rel:.12e}",
                        "total_secondary_abs": f"{total_abs:.12e}",
                        "total_secondary_rel": f"{total_rel:.12e}",
                        "decomposition_residual_abs": f"{decomp_residual:.12e}",
                    }
                )

                # Per-secondary-mode plots.
                denom_sec_mode = np.maximum(np.linalg.norm(q_ref_s, axis=1), 1e-14)
                abs_terms = {
                    "Total secondary error": np.linalg.norm(terms["total_secondary"], axis=1),
                    "Source 2 (map)": np.linalg.norm(terms["source2_map"], axis=1),
                    "Consistency": np.linalg.norm(terms["consistency"], axis=1),
                }
                if case_name != "Case 2":
                    abs_terms["Source 3 (continuity)"] = np.linalg.norm(
                        terms["source3_continuity"], axis=1
                    )

                rel_terms = {
                    key: val / denom_sec_mode
                    for key, val in abs_terms.items()
                }

                fig_decomp = out_dir / (
                    f"{stage}_{mu_tag(mu)}_{case_name.lower().replace(' ', '')}_secondary_decomposition.png"
                )
                decomp_title = (
                    f"{case_name} secondary decomposition at "
                    f"$\\mu=({mu[0]:.3f},\\,{mu[1]:.4f})$ ({stage})"
                )
                plot_secondary_decomposition(
                    out_path=fig_decomp,
                    n_p=n_p,
                    n_tot=n_tot,
                    abs_terms=abs_terms,
                    rel_terms=rel_terms,
                    title=decomp_title,
                )

            print(f"[done] stage={stage} mu={mu_tag(mu)}")

    model_csv = out_dir / "model_error_summary.csv"
    decomp_csv = out_dir / "decomposition_summary.csv"
    write_csv(
        model_csv,
        model_metrics_rows,
        fieldnames=[
            "stage",
            "mu1",
            "mu2",
            "model",
            "abs_total",
            "rel_total",
            "abs_primary",
            "rel_primary",
            "abs_secondary",
            "rel_secondary",
        ],
    )
    write_csv(
        decomp_csv,
        decomp_rows,
        fieldnames=[
            "stage",
            "mu1",
            "mu2",
            "case",
            "source1_primary_abs",
            "source1_primary_rel",
            "source2_map_abs",
            "source2_map_rel",
            "source3_continuity_abs",
            "source3_continuity_rel",
            "consistency_abs",
            "consistency_rel",
            "total_secondary_abs",
            "total_secondary_rel",
            "decomposition_residual_abs",
        ],
    )

    print(f"Saved CSV: {model_csv}")
    print(f"Saved CSV: {decomp_csv}")
    print(f"Saved plots in: {out_dir}")


if __name__ == "__main__":
    main()
