#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Maday-isolated wrapper for full data-driven GPR trainer (qN = G(mu,t))."""

from __future__ import annotations

import argparse
import os
import sys

try:
    from maday_layout import get_paths
except ModuleNotFoundError:
    from .maday_layout import get_paths


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--maday-tag", type=str, default="exp_maday_p2")
    parser.add_argument("--maday-results-root", type=str, default=None)
    args, passthrough = parser.parse_known_args(argv)

    paths = get_paths(tag=args.maday_tag, results_root=args.maday_results_root)

    import stage3_perform_training_rom_data_driven_gpr as base

    base.STAGE3_DIR = paths.stage3
    base.stage3_model_path = lambda name: os.path.join(paths.stage3_models, name)
    base.ensure_layout_dirs = lambda: None

    print(f"[MADAY-STAGE3-DD-GPR] tag={paths.tag}")
    print(f"[MADAY-STAGE3-DD-GPR] stage3_dir={paths.stage3}")
    print(f"[MADAY-STAGE3-DD-GPR] models_dir={paths.stage3_models}")
    base.main(passthrough)


if __name__ == "__main__":
    main(sys.argv[1:])
