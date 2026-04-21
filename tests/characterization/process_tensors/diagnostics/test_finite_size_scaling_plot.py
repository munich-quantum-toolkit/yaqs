"""Smoke test: finite-size summary 2x2 figure and peak-cut plot from synthetic CSV rows."""

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


def test_plot_finite_size_summary_and_peak_cut(tmp_path):
    mod = _load_module()
    # Summary: two L, two J — metrics self-consistent for peak_cut=2 on synthetic profile.
    summary: list[dict[str, str]] = []
    detail: list[dict[str, str]] = []
    for L in (2, 4):
        for J in (0.5, 1.0):
            ent_by_c = {1: 0.1 * L, 2: 0.5 * L + 0.2 * J, 3: 0.15 * L}
            peak_c = 2
            peak_e = ent_by_c[peak_c]
            int_e = sum(ent_by_c.values())
            mean_e = int_e / 3.0
            summary.append(
                {
                    "L": str(L),
                    "J": str(J),
                    "peak_entropy": str(peak_e),
                    "integrated_entropy": str(int_e),
                    "mean_entropy": str(mean_e),
                    "peak_cut": str(peak_c),
                }
            )
            for c, e in ent_by_c.items():
                detail.append(
                    {
                        "L": str(L),
                        "J": str(J),
                        "c": str(c),
                        "entropy": str(e),
                    }
                )

    stem = tmp_path / "fig_finite_size_scaling_summary"
    mod.plot_finite_size_summary_figure(
        summary,
        detail,
        out_stem=stem,
        profile_j=1.0,
        profile_ls=(2, 4),
    )
    assert (tmp_path / "fig_finite_size_scaling_summary.pdf").is_file()
    assert (tmp_path / "fig_finite_size_scaling_summary.png").is_file()

    stem2 = tmp_path / "fig_peak_cut_vs_L"
    mod.plot_peak_cut_vs_l(summary, out_stem=stem2, profile_j=1.0)
    assert (tmp_path / "fig_peak_cut_vs_L.pdf").is_file()
    assert (tmp_path / "fig_peak_cut_vs_L.png").is_file()
