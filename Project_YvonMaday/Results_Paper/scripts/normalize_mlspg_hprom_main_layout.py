#!/usr/bin/env python3
"""Normalize production names in the MLSPG-sensitive HPROM campaign.

Sweep labels remain available in Stage3/sweeps and in the provenance manifest,
but production checkpoints and online outputs use semantic names.
"""

from __future__ import annotations

import shutil
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
PAPER_ROOT = PROJECT / "Results_Paper" / "mlspg_hprom_main"
STAGE3 = PAPER_ROOT / "Stage3"
MODELS = STAGE3 / "models"
RUNS = PAPER_ROOT / "Runs"
ECSW = PAPER_ROOT / "ECSW"
LOGS = PAPER_ROOT / "logs"


WINNERS = {
    "case2_np10": {
        "source_model": "case2_ann_ntot151_np10_B01_A10_like_b128_lr5e4.pt",
        "source_summary": "case2_ann_ntot151_np10_B01_A10_like_b128_lr5e4_summary.txt",
        "target_model": "case2_ann_ntot151_np10_best.pt",
        "target_summary": "case2_ann_ntot151_np10_best_summary.txt",
        "sweep_label": "B01_A10_like_b128_lr5e4",
    },
    "case2_np20": {
        "source_model": "case2_ann_ntot151_np20_B01_A10_like_b128_lr5e4.pt",
        "source_summary": "case2_ann_ntot151_np20_B01_A10_like_b128_lr5e4_summary.txt",
        "target_model": "case2_ann_ntot151_np20_best.pt",
        "target_summary": "case2_ann_ntot151_np20_best_summary.txt",
        "sweep_label": "B01_A10_like_b128_lr5e4",
    },
    "data_driven": {
        "source_model": "rom_data_driven_ann_mu_t_ntot151_A10_silu_wide_b128_lr5e4.pt",
        "source_summary": "rom_data_driven_ann_mu_t_ntot151_A10_silu_wide_b128_lr5e4_summary.txt",
        "target_model": "data_driven_ann_ntot151_best.pt",
        "target_summary": "data_driven_ann_ntot151_best_summary.txt",
        "sweep_label": "A10_silu_wide_b128_lr5e4",
    },
}


def copy_winner(spec: dict[str, str]) -> None:
    source_model = MODELS / spec["source_model"]
    target_model = MODELS / spec["target_model"]
    if source_model.exists():
        shutil.copy2(source_model, target_model)
        print(f"[normalize] checkpoint alias: {target_model.relative_to(PAPER_ROOT)}")
    elif not target_model.exists():
        print(f"[normalize] checkpoint unavailable: {source_model.relative_to(PAPER_ROOT)}")

    source_summary = STAGE3 / spec["source_summary"]
    target_summary = STAGE3 / spec["target_summary"]
    if source_summary.exists():
        text = source_summary.read_text(errors="replace")
        text = text.replace(spec["source_model"], spec["target_model"])
        if "sweep_winner_label:" not in text:
            text = text.rstrip() + f"\n\nsweep_winner_label: {spec['sweep_label']}\n"
        target_summary.write_text(text)
        print(f"[normalize] summary alias: {target_summary.relative_to(PAPER_ROOT)}")
    elif not target_summary.exists():
        print(f"[normalize] summary unavailable: {source_summary.relative_to(PAPER_ROOT)}")


def move_file(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if source.resolve() != target.resolve():
            source.unlink()
        return
    shutil.move(str(source), str(target))


def move_tree_contents(source: Path, target: Path) -> None:
    if not source.is_dir():
        return
    target.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.iterdir()):
        move_file(item, target / item.name)
    source.rmdir()


def replace_in_text_files(root: Path, replacements: dict[str, str]) -> None:
    if not root.is_dir():
        return
    for path in root.rglob("*.txt"):
        text = path.read_text(errors="replace")
        updated = text
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        if updated != text:
            path.write_text(updated)


def split_case2_outputs() -> None:
    source = RUNS / "Case2_B01"
    if not source.is_dir():
        return
    for item in sorted(source.iterdir()):
        if "_n10_" in item.name:
            target = RUNS / "ECSW2pct" / "Case2_Best" / "np10" / item.name
        elif "_n20_" in item.name:
            target = RUNS / "ECSW2pct" / "Case2_Best" / "np20" / item.name
        else:
            target = RUNS / "_legacy_unclassified" / "Case2_B01" / item.name
        move_file(item, target)
    source.rmdir()


def split_case2_weights() -> None:
    source = RUNS / "ECSW_Case2_B01"
    if not source.is_dir():
        return
    for item in sorted(source.iterdir()):
        if "_n10_" in item.name:
            target_name = item.name.replace(
                "case2_ann_ntot151_np10_B01_A10_like_b128_lr5e4",
                "case2_ann_ntot151_np10_best",
            )
            target = ECSW / "2pct" / "Case2_Best" / "np10" / target_name
        elif "_n20_" in item.name:
            target_name = item.name.replace(
                "case2_ann_ntot151_np20_B01_A10_like_b128_lr5e4",
                "case2_ann_ntot151_np20_best",
            )
            target = ECSW / "2pct" / "Case2_Best" / "np20" / target_name
        else:
            target = ECSW / "_legacy_unclassified" / "Case2_B01" / item.name
        move_file(item, target)
    source.rmdir()


def split_case2_logs() -> None:
    source = LOGS / "case2_B01_online_save_snaps"
    if not source.is_dir():
        return
    for item in sorted(source.iterdir()):
        if "np10_" in item.name or "_n10_" in item.name:
            target = LOGS / "online" / "ECSW2pct" / "Case2_Best" / "np10" / item.name
        elif "np20_" in item.name or "_n20_" in item.name:
            target = LOGS / "online" / "ECSW2pct" / "Case2_Best" / "np20" / item.name
        else:
            target = LOGS / "online" / "ECSW2pct" / "Case2" / item.name
        move_file(item, target)
    source.rmdir()


def archive_sweep_candidates() -> None:
    groups = {
        "case2": (
            "case2_ann_ntot151_np*_B*.pt",
            "case2_ann_ntot151_np*_B*_summary.txt",
        ),
        "data_driven": (
            "rom_data_driven_ann_mu_t_ntot151_A*.pt",
            "rom_data_driven_ann_mu_t_ntot151_A*_summary.txt",
        ),
    }
    for group, (model_glob, summary_glob) in groups.items():
        model_archive = STAGE3 / "sweeps" / group / "models"
        summary_archive = STAGE3 / "sweeps" / group / "summaries"
        for source in sorted(MODELS.glob(model_glob)):
            move_file(source, model_archive / source.name)
        for source in sorted(STAGE3.glob(summary_glob)):
            move_file(source, summary_archive / source.name)


def write_manifest() -> None:
    manifest = STAGE3 / "production_models.txt"
    lines = [
        "MLSPG-sensitive HPROM production checkpoints",
        "",
        "Case1_Best: models/case1_ann_ntot151_best.pt",
        "  sweep_winner: C02_wide_silu",
        "Case2_Best/np10: models/case2_ann_ntot151_np10_best.pt",
        "  sweep_winner: B01_A10_like_b128_lr5e4",
        "Case2_Best/np20: models/case2_ann_ntot151_np20_best.pt",
        "  sweep_winner: B01_A10_like_b128_lr5e4",
        "Case3_Best: models/case3_ann_ntot151_best.pt",
        "  sweep_winner: C02_wide_silu",
        "DataDriven_Best: models/data_driven_ann_ntot151_best.pt",
        "  sweep_winner: A10_silu_wide_b128_lr5e4",
        "",
        "Architecture and optimization details are stored in the corresponding",
        "*_best_summary.txt files. Full sweep artifacts are under Stage3/sweeps/.",
    ]
    manifest.write_text("\n".join(lines) + "\n")


def main() -> None:
    if not PAPER_ROOT.is_dir():
        raise SystemExit(f"Missing campaign directory: {PAPER_ROOT}")
    MODELS.mkdir(parents=True, exist_ok=True)

    for spec in WINNERS.values():
        copy_winner(spec)

    move_tree_contents(RUNS / "Case1_Best", RUNS / "ECSW2pct" / "Case1_Best")
    split_case2_outputs()
    move_tree_contents(RUNS / "Case3_Best", RUNS / "ECSW2pct" / "Case3_Best")
    move_tree_contents(RUNS / "DataDriven_A10", RUNS / "DataDriven_Best")

    move_tree_contents(RUNS / "ECSW_Case1_Best", ECSW / "2pct" / "Case1_Best")
    split_case2_weights()
    move_tree_contents(RUNS / "ECSW_Case3_Best", ECSW / "2pct" / "Case3_Best")

    move_tree_contents(
        LOGS / "case1_best_online_save_snaps",
        LOGS / "online" / "ECSW2pct" / "Case1_Best",
    )
    split_case2_logs()
    move_tree_contents(
        LOGS / "case3_best_online_save_snaps",
        LOGS / "online" / "ECSW2pct" / "Case3_Best",
    )
    move_tree_contents(
        LOGS / "data_driven_A10_online_save_snaps",
        LOGS / "online" / "DataDriven_Best",
    )

    archive_sweep_candidates()
    replace_in_text_files(
        RUNS / "ECSW2pct" / "Case1_Best",
        {
            "/Runs/Case1_Best": "/Runs/ECSW2pct/Case1_Best",
            "/Runs/Case1_Best/": "/Runs/ECSW2pct/Case1_Best/",
            "/Runs/ECSW_Case1_Best": "/ECSW/2pct/Case1_Best",
            "/Runs/ECSW_Case1_Best/": "/ECSW/2pct/Case1_Best/",
        },
    )
    replace_in_text_files(
        RUNS / "ECSW2pct" / "Case2_Best" / "np10",
        {
            "/Runs/Case2_B01": "/Runs/ECSW2pct/Case2_Best/np10",
            "/Runs/Case2_B01/": "/Runs/ECSW2pct/Case2_Best/np10/",
            "/Runs/ECSW_Case2_B01": "/ECSW/2pct/Case2_Best/np10",
            "/Runs/ECSW_Case2_B01/": "/ECSW/2pct/Case2_Best/np10/",
            "case2_ann_ntot151_np10_B01_A10_like_b128_lr5e4":
                "case2_ann_ntot151_np10_best",
        },
    )
    replace_in_text_files(
        RUNS / "ECSW2pct" / "Case2_Best" / "np20",
        {
            "/Runs/Case2_B01": "/Runs/ECSW2pct/Case2_Best/np20",
            "/Runs/Case2_B01/": "/Runs/ECSW2pct/Case2_Best/np20/",
            "/Runs/ECSW_Case2_B01": "/ECSW/2pct/Case2_Best/np20",
            "/Runs/ECSW_Case2_B01/": "/ECSW/2pct/Case2_Best/np20/",
            "case2_ann_ntot151_np20_B01_A10_like_b128_lr5e4":
                "case2_ann_ntot151_np20_best",
        },
    )
    replace_in_text_files(
        RUNS / "ECSW2pct" / "Case3_Best",
        {
            "/Runs/Case3_Best": "/Runs/ECSW2pct/Case3_Best",
            "/Runs/Case3_Best/": "/Runs/ECSW2pct/Case3_Best/",
            "/Runs/ECSW_Case3_Best": "/ECSW/2pct/Case3_Best",
            "/Runs/ECSW_Case3_Best/": "/ECSW/2pct/Case3_Best/",
        },
    )
    replace_in_text_files(
        RUNS / "DataDriven_Best",
        {
            "/Runs/DataDriven_A10": "/Runs/DataDriven_Best",
            "/Runs/DataDriven_A10/": "/Runs/DataDriven_Best/",
            "rom_data_driven_ann_mu_t_ntot151_A10_silu_wide_b128_lr5e4":
                "data_driven_ann_ntot151_best",
        },
    )
    write_manifest()
    print(f"[normalize] production layout ready: {PAPER_ROOT}")


if __name__ == "__main__":
    main()
