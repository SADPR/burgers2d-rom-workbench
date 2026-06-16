#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared sparse-GPR utilities for Stage-3 map training/inference."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def choose_inducing_points(
    x_train: np.ndarray,
    *,
    num_inducing: int,
    method: str,
    seed: int,
    kmeans_max_iters: int,
    kmeans_batch_size: int,
    kmeans_fit_samples: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    x_train = np.asarray(x_train, dtype=np.float64)
    n = int(x_train.shape[0])
    m = int(num_inducing)
    if m <= 0:
        raise ValueError("num_inducing must be > 0.")
    if m > n:
        raise ValueError(f"num_inducing={m} exceeds available train samples={n}.")

    mode = str(method).strip().lower()
    if mode == "random":
        idx = rng.choice(n, size=m, replace=False)
        return np.asarray(x_train[idx, :], dtype=np.float64)
    if mode != "kmeans":
        raise ValueError(f"Unsupported inducing-selection='{method}'.")

    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception as ex:
        raise RuntimeError(
            "kmeans inducing selection requires scikit-learn. "
            "Install scikit-learn or use --inducing-selection random."
        ) from ex

    n_fit = int(kmeans_fit_samples)
    if n_fit <= 0 or n_fit > n:
        n_fit = n
    if n_fit < m:
        n_fit = m

    if n_fit < n:
        idx_fit = rng.choice(n, size=n_fit, replace=False)
        x_fit = x_train[idx_fit, :]
    else:
        x_fit = x_train

    print(
        f"[Sparse-GPR] inducing selection: kmeans_minibatch | "
        f"fit_samples={x_fit.shape[0]} | centers={m} | "
        f"batch_size={int(kmeans_batch_size)} | iters={int(kmeans_max_iters)}"
    )
    km = MiniBatchKMeans(
        n_clusters=m,
        random_state=int(seed),
        batch_size=int(max(32, kmeans_batch_size)),
        max_iter=int(max(1, kmeans_max_iters)),
        n_init="auto",
        verbose=0,
    )
    km.fit(x_fit)
    return np.asarray(km.cluster_centers_, dtype=np.float64)


def _rbf_ard_kernel(
    x: np.ndarray,
    z: np.ndarray,
    lengthscales: np.ndarray,
    outputscale: float,
) -> np.ndarray:
    ls = np.maximum(np.asarray(lengthscales, dtype=np.float64), 1e-14)
    inv_l2 = 1.0 / (ls * ls)
    x2 = np.sum((x * np.sqrt(inv_l2)[None, :]) ** 2, axis=1, keepdims=True)
    z2 = np.sum((z * np.sqrt(inv_l2)[None, :]) ** 2, axis=1, keepdims=True).T
    cross = (x * inv_l2[None, :]) @ z.T
    sq = np.maximum(x2 + z2 - 2.0 * cross, 0.0)
    return float(outputscale) * np.exp(-0.5 * sq)


def _matern15_ard_kernel(
    x: np.ndarray,
    z: np.ndarray,
    lengthscales: np.ndarray,
    outputscale: float,
) -> np.ndarray:
    ls = np.maximum(np.asarray(lengthscales, dtype=np.float64), 1e-14)
    inv_l2 = 1.0 / (ls * ls)
    x2 = np.sum((x * np.sqrt(inv_l2)[None, :]) ** 2, axis=1, keepdims=True)
    z2 = np.sum((z * np.sqrt(inv_l2)[None, :]) ** 2, axis=1, keepdims=True).T
    cross = (x * inv_l2[None, :]) @ z.T
    sq = np.maximum(x2 + z2 - 2.0 * cross, 0.0)
    r = np.sqrt(3.0 * sq)
    return float(outputscale) * (1.0 + r) * np.exp(-r)


def kernel_ard_matrix(
    x: np.ndarray,
    z: np.ndarray,
    *,
    lengthscales: np.ndarray,
    outputscale: float,
    kernel_name: str,
) -> np.ndarray:
    key = str(kernel_name).strip().lower()
    if key == "rbf":
        return _rbf_ard_kernel(x, z, lengthscales, outputscale)
    if key == "matern15":
        return _matern15_ard_kernel(x, z, lengthscales, outputscale)
    raise ValueError(f"Unsupported kernel_name='{kernel_name}'. Use 'rbf' or 'matern15'.")


def predict_sparse_batch(x_scaled: np.ndarray, payload: Dict[str, np.ndarray]) -> np.ndarray:
    x_scaled = np.asarray(x_scaled, dtype=np.float64)
    inducing = np.asarray(payload["inducing_points"], dtype=np.float64)  # (out_dim,m,in_dim)
    alpha = np.asarray(payload["alpha"], dtype=np.float64)  # (out_dim,m)
    lengthscales = np.asarray(payload["lengthscales"], dtype=np.float64)  # (out_dim,in_dim)
    outputscales = np.asarray(payload["outputscales"], dtype=np.float64)  # (out_dim,)
    kernel_name = str(payload.get("kernel_name", "rbf")).strip().lower()

    out_dim = int(alpha.shape[0])
    n_eval = int(x_scaled.shape[0])
    y_pred = np.zeros((n_eval, out_dim), dtype=np.float64)

    for j in range(out_dim):
        kxz = kernel_ard_matrix(
            x_scaled,
            inducing[j, :, :],
            lengthscales=lengthscales[j, :],
            outputscale=float(outputscales[j]),
            kernel_name=kernel_name,
        )
        y_pred[:, j] = kxz @ alpha[j, :]

    return y_pred


def resolve_device(device_arg: str) -> str:
    d = str(device_arg).strip().lower()
    if d in ("cpu", "cuda"):
        return d
    if d != "auto":
        raise ValueError("--device must be one of: auto, cpu, cuda")
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def fit_sparse_gp_output(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    x_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    inducing_init: np.ndarray,
    kernel_name: str,
    ard: bool,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    min_noise: float,
    max_noise: float | None,
    elbo_beta: float,
    learn_inducing: bool,
    device: str,
    seed: int,
    log_every: int = 10,
) -> Dict[str, object]:
    try:
        import torch
        import gpytorch
    except Exception as ex:
        raise RuntimeError("Sparse-GPR training requires torch + gpytorch.") from ex

    torch.manual_seed(int(seed))
    dtype = torch.float64
    dev = torch.device(device)

    xtr = torch.as_tensor(np.asarray(x_train, dtype=np.float64), dtype=dtype, device=dev)
    ytr = torch.as_tensor(np.asarray(y_train, dtype=np.float64), dtype=dtype, device=dev)
    xva = (
        torch.as_tensor(np.asarray(x_val, dtype=np.float64), dtype=dtype, device=dev)
        if x_val is not None
        else None
    )
    yva = (
        torch.as_tensor(np.asarray(y_val, dtype=np.float64), dtype=dtype, device=dev)
        if y_val is not None
        else None
    )

    inducing = torch.as_tensor(np.asarray(inducing_init, dtype=np.float64), dtype=dtype, device=dev).clone()
    in_dim = int(inducing.size(-1))
    use_ard = bool(ard)

    class _SVGP(gpytorch.models.ApproximateGP):
        def __init__(self, inducing_points):
            vd = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
            vs = gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                vd,
                learn_inducing_locations=bool(learn_inducing),
            )
            super().__init__(vs)
            self.mean_module = gpytorch.means.ZeroMean()
            key = str(kernel_name).strip().lower()
            if key == "rbf":
                base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=(in_dim if use_ard else None))
            elif key == "matern15":
                base_kernel = gpytorch.kernels.MaternKernel(
                    nu=1.5,
                    ard_num_dims=(in_dim if use_ard else None),
                )
            else:
                raise ValueError(f"Unsupported kernel_name='{kernel_name}'. Use 'rbf' or 'matern15'.")
            self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x),
                self.covar_module(x),
            )

    model = _SVGP(inducing).to(dev).double()
    if max_noise is not None and float(max_noise) > float(min_noise):
        noise_constraint = gpytorch.constraints.Interval(float(min_noise), float(max_noise))
    else:
        noise_constraint = gpytorch.constraints.GreaterThan(float(min_noise))
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=noise_constraint
    ).to(dev).double()

    model.train()
    likelihood.train()
    params = list(model.parameters()) + list(likelihood.parameters())
    optimizer = torch.optim.Adam(params, lr=float(lr), weight_decay=float(weight_decay))
    mll = gpytorch.mlls.VariationalELBO(
        likelihood,
        model,
        num_data=xtr.size(0),
        beta=float(elbo_beta),
    )

    n_train = int(xtr.size(0))
    bs = int(max(8, batch_size))
    n_batches = max(1, int(np.ceil(n_train / bs)))
    best_state = None
    best_score = np.inf
    train_hist = []
    val_hist = []

    for ep in range(1, int(max(1, epochs)) + 1):
        perm = torch.randperm(n_train, device=dev)
        ep_loss = 0.0

        for b in range(n_batches):
            i0 = b * bs
            i1 = min((b + 1) * bs, n_train)
            idx = perm[i0:i1]
            xb = xtr[idx]
            yb = ytr[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = -mll(out, yb)
            loss.backward()
            optimizer.step()
            ep_loss += float(loss.item()) * float(i1 - i0)

        ep_loss /= float(n_train)
        train_hist.append(ep_loss)

        model.eval()
        with torch.no_grad():
            if xva is not None and yva is not None:
                pred = model(xva).mean
                val_mse = float(torch.mean((pred - yva) ** 2).item())
            else:
                pred = model(xtr).mean
                val_mse = float(torch.mean((pred - ytr) ** 2).item())
        val_hist.append(val_mse)
        if val_mse < best_score:
            best_score = val_mse
            best_state = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "likelihood": {k: v.detach().cpu().clone() for k, v in likelihood.state_dict().items()},
            }

        if ep == 1 or ep % int(max(1, log_every)) == 0 or ep == int(epochs):
            print(
                f"      epoch {ep:04d}/{int(epochs):04d} | "
                f"train_loss={ep_loss:.6e} | val_mse={val_mse:.6e}"
            )
        model.train()

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        likelihood.load_state_dict(best_state["likelihood"])

    model.eval()
    likelihood.eval()
    with torch.no_grad():
        z_dev = model.variational_strategy.inducing_points.detach()
        m_white = model.variational_strategy.variational_distribution.mean.detach()

        ls_raw = model.covar_module.base_kernel.lengthscale.detach().reshape(-1).cpu().numpy()
        if ls_raw.size == 1:
            ls = np.full((in_dim,), float(ls_raw[0]), dtype=np.float64)
        elif ls_raw.size == in_dim:
            ls = np.asarray(ls_raw, dtype=np.float64)
        else:
            ls = np.asarray(ls_raw[:in_dim], dtype=np.float64)

        oscale = float(model.covar_module.outputscale.detach().cpu().item())
        noise = float(likelihood.noise.detach().cpu().item())

        kzz_np = kernel_ard_matrix(
            np.asarray(z_dev.cpu().numpy(), dtype=np.float64),
            np.asarray(z_dev.cpu().numpy(), dtype=np.float64),
            lengthscales=ls,
            outputscale=oscale,
            kernel_name=kernel_name,
        )
        jitter = max(float(min_noise), 1.0e-12)
        kzz_np = kzz_np + jitter * np.eye(kzz_np.shape[0], dtype=np.float64)
        rhs = np.asarray(m_white.cpu().numpy(), dtype=np.float64)

        # VariationalStrategy in GPyTorch is whitened by default:
        # predictive mean uses K_xz K_zz^{-1/2} m_white.
        # With K_zz = L L^T, this is K_xz (L^{-T} m_white), so the
        # exported coefficients must be alpha = L^{-T} m_white.
        # Using K_zz^{-1} m_white would apply an extra L^{-1} and leads
        # to severe mismatch at inference time.
        L = np.linalg.cholesky(kzz_np)
        alpha = np.linalg.solve(L.T, rhs)

    return {
        "inducing_points": np.asarray(z_dev.cpu().numpy(), dtype=np.float64),
        "alpha": np.asarray(alpha, dtype=np.float64),
        "lengthscales": np.asarray(ls, dtype=np.float64),
        "outputscale": float(oscale),
        "noise": float(noise),
        "train_history": np.asarray(train_hist, dtype=np.float64),
        "val_history": np.asarray(val_hist, dtype=np.float64),
        "best_val_mse": float(best_score),
    }
