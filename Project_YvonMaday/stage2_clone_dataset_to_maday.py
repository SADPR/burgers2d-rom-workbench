#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Clone an existing Stage-2 dataset into Results_Maday/<tag>/Stage2/.

This keeps baseline `Results/Stage2/*` immutable while running Maday experiments.
"""

from __future__ import annotations

import argparse
import os
import shutil

try:
    from maday_layout import get_paths
except ModuleNotFoundError:
    from .maday_layout import get_paths

try:
    from project_layout import stage2_dataset_dir
except ModuleNotFoundError:
    from .project_layout import stage2_dataset_dir


def _resolve_src(dataset_dir: str | None, dataset_ntot: int | None) -> str:
    if dataset_dir is not None:
        src = os.path.abspath(os.path.expanduser(str(dataset_dir)))
    else:
        if dataset_ntot is None:
            raise ValueError("Provide either --dataset-dir or --dataset-ntot.")
        src = stage2_dataset_dir(int(dataset_ntot))
    if not os.path.isdir(src):
        raise FileNotFoundError(f"Source dataset directory does not exist: {src}")
    if not os.path.isdir(os.path.join(src, "per_mu")):
        raise FileNotFoundError(f"Source dataset is missing per_mu/: {src}")
    if not os.path.exists(os.path.join(src, "meta.npy")):
        raise FileNotFoundError(f"Source dataset is missing meta.npy: {src}")
    return src


def main(argv=None):
    parser = argparse.ArgumentParser(description="Clone Stage2 dataset into Results_Maday.")
    parser.add_argument("--maday-tag", type=str, default="exp_maday_p2")
    parser.add_argument("--maday-results-root", type=str, default=None)
    parser.add_argument("--dataset-dir", type=str, default=None, help="Explicit source dataset directory.")
    parser.add_argument("--dataset-ntot", type=int, default=None, help="Fallback source selector from Results/Stage2.")
    parser.add_argument("--dest-name", type=str, default=None, help="Destination folder name under Results_Maday/<tag>/Stage2.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination if it already exists.")
    args = parser.parse_args(argv)

    src = _resolve_src(args.dataset_dir, args.dataset_ntot)
    paths = get_paths(tag=args.maday_tag, results_root=args.maday_results_root)

    dest_name = str(args.dest_name).strip() if args.dest_name else os.path.basename(src)
    if not dest_name:
        raise ValueError("Destination name is empty.")
    dest = os.path.join(paths.stage2, dest_name)

    if os.path.exists(dest):
        if not args.overwrite:
            raise FileExistsError(
                f"Destination already exists: {dest}\n"
                "Use --overwrite to replace it."
            )
        shutil.rmtree(dest)

    shutil.copytree(src, dest)
    print(f"[MADAY-STAGE2] cloned dataset")
    print(f"  src : {src}")
    print(f"  dest: {dest}")


if __name__ == "__main__":
    main()

