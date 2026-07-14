"""Train and animate a coefficient-space POD-AE manifold."""

import numpy as np
import torch
from torch import nn

from animation import AnimationSpec, render_manifold_animation
from benchmark import build_benchmark, relative_error


SEED = 11
LATENT_DIMENSION = 2


class CoefficientAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 24),
            nn.Tanh(),
            nn.Linear(24, 16),
            nn.Tanh(),
            nn.Linear(16, LATENT_DIMENSION),
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIMENSION, 16),
            nn.Tanh(),
            nn.Linear(16, 24),
            nn.Tanh(),
            nn.Linear(24, 3),
        )

    def forward(self, coefficients):
        latent = self.encoder(coefficients)
        return self.decoder(latent)


def train_autoencoder(coefficients):
    torch.manual_seed(SEED)
    torch.set_num_threads(1)

    mean = coefficients.mean(axis=0)
    scale = coefficients.std(axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    normalized = (coefficients - mean) / scale
    samples = torch.tensor(normalized, dtype=torch.float32)

    model = CoefficientAutoencoder()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3.0e-3,
        weight_decay=1.0e-8,
    )

    model.train()
    final_loss = None
    for epoch in range(8000):
        optimizer.zero_grad()
        reconstruction = model(samples)
        loss = torch.mean((reconstruction - samples) ** 2)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
        if epoch >= 1500 and final_loss < 2.0e-8:
            break

    model.eval()
    print(
        "POD-AE training:"
        f" epochs={epoch + 1}, normalized MSE={final_loss:.3e}"
    )
    return model, mean, scale, samples


def decode(model, latent, mean, scale):
    latent_tensor = torch.tensor(latent, dtype=torch.float32)
    with torch.no_grad():
        normalized = model.decoder(latent_tensor).cpu().numpy()
    return normalized * scale + mean


data = build_benchmark()
V_tot = np.column_stack((data.V, data.V_bar))
q_N = np.column_stack((data.q, data.closure_coordinate))

model, coefficient_mean, coefficient_scale, normalized_q_N = (
    train_autoencoder(q_N)
)

with torch.no_grad():
    latent_trajectory = model.encoder(normalized_q_N).cpu().numpy()

decoded_q_N = decode(
    model,
    latent_trajectory,
    coefficient_mean,
    coefficient_scale,
)
u_pod_ae = data.u_ref + decoded_q_N @ V_tot.T

padding = 0.08
z_1 = np.linspace(
    latent_trajectory[:, 0].min() - padding,
    latent_trajectory[:, 0].max() + padding,
    55,
)
z_2 = np.linspace(
    latent_trajectory[:, 1].min() - padding,
    latent_trajectory[:, 1].max() + padding,
    55,
)
Z_1, Z_2 = np.meshgrid(z_1, z_2)
latent_grid = np.column_stack((Z_1.ravel(), Z_2.ravel()))
decoded_grid = decode(
    model,
    latent_grid,
    coefficient_mean,
    coefficient_scale,
)
surface_states = (data.u_ref + decoded_grid @ V_tot.T).reshape(
    Z_1.shape + (3,)
)
surface = (
    surface_states[:, :, 0],
    surface_states[:, :, 1],
    surface_states[:, :, 2],
)

spec = AnimationSpec(
    title=r"\textbf{POD--AE latent manifold}",
    output_name="pod_ae_manifold.gif",
    surface_color="paleturquoise",
    curve_color="teal",
    approximation_label=(
        r"POD--AE reconstruction"
        "\n"
        r"\hspace{1em}"
        r"$(\mathbf{u}_{\mathrm{ref}}+\mathbf{V}_{\mathrm{tot}}"
        r"\mathcal{D}_{\mathrm{AE}}(\mathbf{z}))$"
    ),
    manifold_label=r"decoded latent manifold",
    error=relative_error(data.u, u_pod_ae, data.u_ref),
    show_linear_mapping=False,
)

render_manifold_animation(data, surface, u_pod_ae, spec)
