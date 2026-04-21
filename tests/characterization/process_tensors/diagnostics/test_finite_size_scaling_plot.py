"""Smoke test: 3-panel finite-size figure builds from synthetic CSV rows."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[4]
    script_path = repo_root / "experiments" / "benchmark_finite_size_scaling.py"
    spec = importlib.util.spec_from_file_location("benchmark_finite_size_scaling", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plot_finite_size_scaling_writes_outputs(tmp_path):
    mod = _load_module()
    # Two L, two J, two cuts — consistent peak metadata on every row.
    rows: list[dict[str, str]] = []
    for L in (2, 4):
        for J in (0.5, 1.0):
            peak_e = 0.2 * L + 0.1 * J
            for c in (1, 2):
                rows.append(
                    {
                        "L": str(L),
                        "J": str(J),
                        "c": str(c),
                        "entropy": str(0.05 * c),
                        "peak_entropy": str(peak_e),
                        "peak_cut": "2",
                        "n_0_99": str(3 + L),
                    }
                )
    stem = tmp_path / "fig_finite_size_scaling"
    mod.plot_finite_size_figure(rows, out_stem=stem, profile_j=1.0, profile_ls=(2, 4))
    assert (tmp_path / "fig_finite_size_scaling.pdf").is_file()
    assert (tmp_path / "fig_finite_size_scaling.png").is_file()
