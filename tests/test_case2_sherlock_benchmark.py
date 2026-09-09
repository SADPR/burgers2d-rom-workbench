import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


FILE = Path(__file__).resolve().parents[1] / "Project_YvonMaday/run_case2_sherlock_benchmark.py"
spec = importlib.util.spec_from_file_location("sherlock_benchmark", FILE)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_commands_keep_initialization_controls_separate(tmp_path):
    for method in benchmark.METHODS:
        cmd = benchmark.command(method, tmp_path, 24)
        assert cmd[cmd.index("--threads")+1] == "24"
        assert cmd[cmd.index("--point-indices")+1:cmd.index("--threads")] == ["0", "1", "2", "3"]
        if benchmark.METHODS[method] is not None:
            assert cmd[cmd.index("--initialization")+1] == benchmark.METHODS[method][1]
        else:
            assert cmd[cmd.index("--methods")+1] == method


def test_summary_excludes_warmup_and_reports_missing_repetitions(tmp_path):
    for rep, seconds in [("warmup", 9999), ("rep01", 10), ("rep02", 14)]:
        folder = tmp_path / rep / "adaptive3_known_initial" / "reporting_0_krylov3_steps500"
        folder.mkdir(parents=True)
        (folder / "summary.json").write_text(json.dumps({"online_seconds": seconds,
                                                        "state_error_percent": .4}))
    benchmark.summarize(tmp_path, 10, warmups=1)
    import csv
    with (tmp_path / "timing_summary.csv").open() as stream:
        row, = list(csv.DictReader(stream))
    assert int(row["repetitions"]) == 2
    assert float(row["mean_seconds"]) == 12
    assert float(row["std_seconds"]) == pytest.approx(8**.5)
    assert "Status: PARTIAL" in (tmp_path / "timing_summary.txt").read_text()


def test_summary_complete_in_domain_mean(tmp_path):
    for method in benchmark.DEFAULT_METHODS:
        for point in range(4):
            folder = tmp_path / "rep01" / method / f"reporting_{point}_result"
            folder.mkdir(parents=True)
            (folder / "summary.json").write_text(json.dumps({
                "online_seconds": 10*(point+1), "state_error_percent": .4}))
    benchmark.summarize(tmp_path, 1)
    text = (tmp_path / "timing_summary.txt").read_text()
    assert "Status: COMPLETE" in text
    assert "adaptive3_known_initial: 20.000000 s" in text
    assert "timing variability is not estimated" in text
    assert "warmups per method/point: 0" in text
    import csv
    with (tmp_path / "timing_summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 16
    assert all(row["std_seconds"] == "" for row in rows)


def test_default_launch_is_16_trajectories_with_24_threads(tmp_path):
    result = subprocess.run([sys.executable, str(FILE), "--output", str(tmp_path), "--dry-run"],
                            check=True, capture_output=True, text=True)
    commands = result.stdout.splitlines()[:-1]
    assert len(commands) == 4
    assert all("--threads 24" in line for line in commands)
    assert "16 full trajectories; threads=24; warmups=0" in result.stdout
    assert not list(tmp_path.iterdir())
