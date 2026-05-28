# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Smoke test for hybrid benchmark report generation."""

from __future__ import annotations

import json
from pathlib import Path

from .hybrid_benchmark_lib import ROTATION_BENCHMARK_CONFIG, generate_report

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_hybrid_benchmark_report_generates(tmp_path: Path) -> None:
    """Benchmark runner produces JSON and Markdown with expected sections."""
    report = generate_report(tmp_path)
    json_path = tmp_path / "hybrid_tdvp_benchmark.json"
    md_path = tmp_path / "hybrid_tdvp_benchmark.md"
    assert json_path.is_file()
    assert md_path.is_file()

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert "baseline_hybrid_vs_tebd" in loaded
    assert "sweep_count_study" in loaded
    assert "initial_padding_study" in loaded
    assert len(loaded["baseline_hybrid_vs_tebd"]) >= 8

    md_text = md_path.read_text(encoding="utf-8")
    assert "# Hybrid TDVP Benchmark Report" in md_text
    assert "Effect of `tdvp_sweeps`" in md_text
    assert "initial bond padding" in md_text

    assert report["meta"]["output_files"]["json"].endswith("hybrid_tdvp_benchmark.json")


def test_hybrid_rotation_benchmark_report_generates(tmp_path: Path) -> None:
    """Rotation-only benchmark runner produces JSON and Markdown with expected sections."""
    report = generate_report(tmp_path, config=ROTATION_BENCHMARK_CONFIG)
    json_path = tmp_path / "hybrid_rotation_tdvp_benchmark.json"
    md_path = tmp_path / "hybrid_rotation_tdvp_benchmark.md"
    assert json_path.is_file()
    assert md_path.is_file()

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert "baseline_hybrid_vs_tebd" in loaded
    assert len(loaded["baseline_hybrid_vs_tebd"]) >= 2 * 10
    assert "rotation_angles" in loaded["meta"]["settings"]
    assert loaded["meta"]["settings"]["study_sweep_counts"][0] == 1
    assert loaded["meta"]["settings"]["study_pad_values"][0] == "None"
    assert "single_qubit_plus_lr_ryy_4q" in loaded["meta"]["probe_ids"]
    assert "lr_stack_mixed_12q" in loaded["meta"]["probe_ids"]
    assert "lr_rzz_dist_9_10q" in loaded["meta"]["probe_ids"]

    md_text = md_path.read_text(encoding="utf-8")
    assert "Rotation-Gate Benchmark" in md_text
    assert report["meta"]["output_files"]["json"].endswith("hybrid_rotation_tdvp_benchmark.json")
