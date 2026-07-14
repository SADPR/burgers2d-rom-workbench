#!/usr/bin/env python3
"""Animate a stored full-order J2-plastic RVE displacement history."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize


SOURCE_ROOT = Path("/home/kratos/ML_assisted_CLs/RVE_homogenization_NeoHookean_using_Kratos")
MESH_PATH = SOURCE_ROOT / "rve_geometry.mdpa"
DISPLACEMENT_HISTORY_PATH = (
    SOURCE_ROOT
    / "stage_1_training_set_fom"
    / "trajectory_5"
    / "trajectory_5_U.npy"
)

ASSET_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ASSET_DIR / "outputs"
PREVIEW_DIR = ASSET_DIR / "previews"
GIF_PATH = OUTPUT_DIR / "plastic_rve_training_paths.gif"
PREVIEW_PATH = PREVIEW_DIR / "plastic_rve_training_paths_final.png"


def read_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read node coordinates and corner triangles from the Kratos mdpa mesh."""
    nodes: dict[int, tuple[float, float]] = {}
    triangles: list[tuple[int, int, int]] = []
    section = ""
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("Begin Nodes"):
            section = "nodes"
            continue
        if line.startswith("Begin Geometries Triangle2D6"):
            section = "triangles"
            continue
        if line.startswith("End "):
            section = ""
            continue
        if not line:
            continue
        fields = line.split()
        if section == "nodes":
            node_id = int(fields[0])
            nodes[node_id] = (float(fields[1]), float(fields[2]))
        elif section == "triangles":
            # Triangle2D6 stores the three corner-node IDs first.
            triangles.append((int(fields[1]) - 1, int(fields[2]) - 1, int(fields[3]) - 1))

    coordinates = np.array([nodes[index] for index in range(1, len(nodes) + 1)])
    return coordinates, np.asarray(triangles, dtype=np.int32)


def displacement_at(history: np.ndarray, index: int, node_count: int) -> np.ndarray:
    """Map the stored equation-vector displacement to node-wise components."""
    vector = np.asarray(history[index], dtype=float)
    expected_size = 2 * node_count
    if vector.size != expected_size:
        raise ValueError(
            "Displacement vector has {} entries; expected {}.".format(
                vector.size, expected_size
            )
        )
    return vector.reshape(node_count, 2)


def selected_frames(history_length: int) -> np.ndarray:
    """Sample the full FOM trajectory densely enough for a smooth loop."""
    return np.unique(np.rint(np.linspace(0, history_length - 1, 52)).astype(int))


def padded_bounds(coordinates: np.ndarray) -> tuple[float, float, float, float]:
    """Create square plotting bounds that preserve the RVE aspect ratio."""
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0)
    centre = 0.5 * (lower + upper)
    span = float(max(upper - lower))
    half_width = 0.57 * span
    return (
        centre[0] - half_width,
        centre[0] + half_width,
        centre[1] - half_width,
        centre[1] + half_width,
    )


def build_animation() -> tuple[plt.Figure, FuncAnimation]:
    """Build the field-only RVE deformation animation."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "savefig.facecolor": "white",
        }
    )
    reference_coordinates, triangles = read_mesh(MESH_PATH)
    history = np.load(DISPLACEMENT_HISTORY_PATH, mmap_mode="r")
    if history.ndim != 2:
        raise ValueError("Expected a two-dimensional displacement history.")
    frame_indices = selected_frames(history.shape[0])

    sampled_displacements = [
        displacement_at(history, int(index), reference_coordinates.shape[0])
        for index in frame_indices
    ]
    sampled_coordinates = [
        reference_coordinates + displacement for displacement in sampled_displacements
    ]
    max_displacement = max(
        float(np.linalg.norm(displacement, axis=1).max())
        for displacement in sampled_displacements
    )
    bounds = padded_bounds(np.vstack(sampled_coordinates))

    initial_coordinates = sampled_coordinates[0]
    initial_magnitude = np.linalg.norm(sampled_displacements[0], axis=1)
    initial_face_values = initial_magnitude[triangles].mean(axis=1)

    fig, axis = plt.subplots(figsize=(5.25, 5.80))
    fig.subplots_adjust(left=0.06, right=0.94, top=0.90, bottom=0.12)
    fig.suptitle("J2 plastic RVE", fontsize=17, fontweight="bold", y=0.975)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(bounds[0], bounds[1])
    axis.set_ylim(bounds[2], bounds[3])
    axis.set_axis_off()

    collection = PolyCollection(
        initial_coordinates[triangles],
        array=initial_face_values,
        cmap="turbo",
        norm=Normalize(vmin=0.0, vmax=max_displacement),
        edgecolors=(0.05, 0.05, 0.05, 0.58),
        linewidths=0.20,
    )
    axis.add_collection(collection)
    colorbar = fig.colorbar(
        collection,
        ax=axis,
        orientation="horizontal",
        fraction=0.050,
        pad=0.045,
        aspect=34,
    )
    colorbar.set_label(r"$\|\mathbf{u}\|$", fontsize=11)
    colorbar.ax.tick_params(labelsize=8)

    def update(frame: int) -> list[object]:
        displacement = sampled_displacements[frame]
        deformed_coordinates = sampled_coordinates[frame]
        magnitude = np.linalg.norm(displacement, axis=1)
        collection.set_verts(deformed_coordinates[triangles])
        collection.set_array(magnitude[triangles].mean(axis=1))
        return [collection]

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=115,
        repeat=True,
        blit=False,
    )
    return fig, animation


def main() -> None:
    """Write the field-only GIF and a static final-frame preview."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    fig, animation = build_animation()
    animation.save(GIF_PATH, writer=PillowWriter(fps=9), dpi=125)
    animation._func(animation.save_count - 1)
    fig.savefig(PREVIEW_PATH, dpi=185, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("Wrote {}".format(GIF_PATH))
    print("Wrote {}".format(PREVIEW_PATH))


if __name__ == "__main__":
    main()
