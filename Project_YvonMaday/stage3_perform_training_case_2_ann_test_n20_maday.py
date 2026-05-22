#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Maday-isolated wrapper for Case-2 ANN n=20 trainer.

It reuses the production trainer logic but redirects Stage-3 outputs to
`Results_Maday/<tag>/Stage3/`.
"""

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

    import stage3_perform_training_case_2_ann_test_n20 as base

    # Redirect output paths inside imported module.
    base.STAGE3_DIR = paths.stage3
    base.stage3_model_path = lambda name: os.path.join(paths.stage3_models, name)
    base.ensure_layout_dirs = lambda: None

    if "--summary-name" not in passthrough:
        passthrough.extend(["--summary-name", f"case2_training_summary_test_n20__{paths.tag}.txt"])

    print(f"[MADAY-STAGE3-ANN] tag={paths.tag}")
    print(f"[MADAY-STAGE3-ANN] stage3_dir={paths.stage3}")
    print(f"[MADAY-STAGE3-ANN] models_dir={paths.stage3_models}")
    base.main(passthrough)


if __name__ == "__main__":
    main(sys.argv[1:])

