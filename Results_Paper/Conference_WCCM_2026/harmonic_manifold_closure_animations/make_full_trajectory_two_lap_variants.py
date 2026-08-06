"""Create non-overwriting full-trajectory, two-lap variants of all 3D GIFs.

The original presentation GIFs deliberately draw the trajectory first.  These
variants start at the last trajectory-only frame, hold that complete curve for
half a second, retain every following staged reveal, and replay the final
camera-turn sequence once so that the animation makes two laps in total.
"""

from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
FRAME_TIME_MS = 40
TRAJECTORY_LEAD_IN_MS = 500


# The original generators all use a trajectory-first timeline.  These indices
# identify its final frame and the first frame of the original final rotation.
# They are read from the rendered GIFs so the original animation code and GIFs
# remain unchanged.
VARIANTS = {
    "case1_ann_rbf_gpr_state.gif": (119, 356),
    "case2_ann_rbf_gpr_parameter_time.gif": (119, 355),
    "case3_ann_rbf_gpr_hybrid.gif": (119, 355),
    "general_ann_rbf_gpr_closure.gif": (119, 376),
    "generic_decoder_tangent.gif": (109, 327),
    "linear_manifold.gif": (119, 257),
    "local_prom_ann_two_bases_global_ann_rbf_gpr.gif": (119, 237),
    "local_prom_ann_two_bases_local_linear.gif": (119, 250),
    "local_prom_ann_two_bases_local_nonlinear.gif": (119, 257),
    "piecewise_linear_manifold.gif": (119, 572),
    "pod_ae_manifold.gif": (119, 257),
    "quadratic_manifold.gif": (119, 376),
}


def load_frames(path):
    frames = []
    durations = []
    with Image.open(path) as image:
        for frame_index in range(image.n_frames):
            image.seek(frame_index)
            frames.append(image.convert("RGB"))
            durations.append(image.info.get("duration", FRAME_TIME_MS))
    return frames, durations


def output_path(source_path):
    return source_path.with_name(
        f"{source_path.stem}_full_trajectory_two_laps.gif"
    )


def make_variant(source_path, trajectory_end, rotation_start):
    source_frames, source_durations = load_frames(source_path)
    if not (0 <= trajectory_end < rotation_start < len(source_frames)):
        raise ValueError(
            f"invalid timeline for {source_path.name}: "
            f"trajectory_end={trajectory_end}, rotation_start={rotation_start}, "
            f"n_frames={len(source_frames)}"
        )

    # Begin with the complete black trajectory and no manifold.  The source
    # continues immediately with its original progressive manifold reveal.
    frames = [source_frames[trajectory_end]]
    durations = [TRAJECTORY_LEAD_IN_MS]
    frames.extend(source_frames[trajectory_end + 1 :])
    durations.extend(source_durations[trajectory_end + 1 :])

    # The retained source already contains one complete final camera lap.
    # Replaying only that rotation is seamless because its end equals its start
    # after a full 360-degree turn.
    frames.extend(source_frames[rotation_start:])
    durations.extend(source_durations[rotation_start:])

    destination = output_path(source_path)
    frames[0].save(
        destination,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )
    print(destination)


def main():
    for name, (trajectory_end, rotation_start) in VARIANTS.items():
        make_variant(OUTPUTS / name, trajectory_end, rotation_start)

    print(
        "Skipped pod_ae_decoder_logic.gif: it is a 2D workflow diagram, "
        "without a trajectory, manifold, or camera turn."
    )


if __name__ == "__main__":
    main()
