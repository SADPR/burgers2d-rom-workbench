import json
import sys

import numpy as np

from Project_YvonMaday import summarize_euclidean_hprom_campaign as summary


def write_kv(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{key}: {value}" for key, value in values.items()) + "\n")


def test_complete_eight_method_summary(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    direct_lines = []
    for point, (label, mu1, mu2) in enumerate(summary.POINTS):
        tag = f"mu1_{mu1:.3f}_mu2_{mu2:.4f}"
        teacher = np.full((3, 4), point + 1.0)
        linear = root / "Runs/Linear" / f"linear_hprom_{tag}_ntot151"
        linear.mkdir(parents=True)
        np.save(linear / "qN.npy", teacher)
        write_kv(linear / "summary.txt", {
            "relative_error_percent": 0.4 + point,
            "online_solve_elapsed_s": 4.0,
            "n_ecsw_elements": 100,
        })

        for folder, prefix in (
            ("HPROM_ANN_C1", "case1_hprom_ann"),
            ("HPROM_ANN_C2", "case2_hprom_ann"),
            ("HPROM_ANN_C3", "case3_hprom_ann"),
            ("HPROM_POD_AE", "podae_hprom"),
        ):
            path = root / "Runs" / folder
            path.mkdir(parents=True, exist_ok=True)
            np.save(path / f"{prefix}_{tag}_n10_ntot151_qN.npy", teacher * 1.01)
            write_kv(path / f"{prefix}_{tag}_n10_ntot151_summary.txt", {
                "relative_error_percent": 1.0 + point,
                "online_solve_elapsed_s": 2.0,
                "n_ecsw_elements": 80,
            })

        b3 = root / "Stage4/case2_b3" / f"reporting_positive_fit4096_hprom3_p{point}_steps500"
        b3.mkdir(parents=True)
        np.save(b3 / "qN.npy", teacher * 1.005)
        (b3 / "summary.json").write_text(json.dumps({
            "state_error_percent": 0.5 + point,
            "online_seconds": 3.0,
            "sampled_cells": 120,
        }))

        podnn = root / "Runs/PODNN_ROM" / f"rom_data_driven_{tag}_ntot151"
        podnn.mkdir(parents=True)
        np.save(podnn / "qN.npy", teacher * 1.02)
        write_kv(podnn / "rom_data_driven_summary.txt", {"relative_error_percent": 2.0 + point})
        poddl = root / "Runs/PODDL_ROM" / f"pod_dl_data_driven_{tag}_ntot151_nz10"
        poddl.mkdir(parents=True)
        np.save(poddl / "qN.npy", teacher * 1.03)
        write_kv(poddl / "pod_dl_data_driven_summary.txt", {"relative_error_percent": 3.0 + point})
        direct_lines += [
            f"podnn_{label}_mean_inference_time_s: 0.01",
            f"poddl_{label}_mean_inference_time_s: 0.02",
        ]

    timing = root / "timing"
    timing.mkdir(parents=True)
    (timing / "direct_inference_repeat10_summary.txt").write_text("\n".join(direct_lines) + "\n")
    hdm = timing / "hdm"
    hdm.mkdir()
    (hdm / "hdm_timing.json").write_text(json.dumps({"mean_in_domain_seconds": 100.0}))

    monkeypatch.setattr(sys, "argv", ["summary", "--campaign-root", str(root)])
    summary.main()
    rows = list(__import__("csv").DictReader((root / "reporting/hprom_accuracy_timing.csv").open()))
    assert [row["model"] for row in rows] == [
        "Linear HPROM", "HPROM-ANN Case 1", "HPROM-ANN Case 2",
        "HPROM-ANN Case 2+B3", "HPROM-ANN Case 3", "HPROM-POD-AE",
        "POD-NN-ROM", "POD-DL-ROM",
    ]
    assert float(rows[0]["speedup_vs_hdm"]) == 25.0
    assert float(rows[6]["speedup_vs_hdm"]) == 10000.0
