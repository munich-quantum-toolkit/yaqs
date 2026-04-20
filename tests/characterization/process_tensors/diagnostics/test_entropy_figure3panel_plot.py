from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_benchmark_module():
    repo_root = Path(__file__).resolve().parents[4]
    script_path = repo_root / "experiments" / "benchmark_entropy_vs_j_by_cut.py"
    spec = importlib.util.spec_from_file_location("benchmark_entropy_vs_j_by_cut", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plot_entropy_heatmap_cut_vs_j_writes_expected_outputs(tmp_path):
    mod = _load_benchmark_module()

    rows: list[dict[str, float | int]] = []
    for cut in range(1, 21):
        for ji in range(11):
            jv = 0.2 * ji
            # Simple synthetic smooth surface; stays positive for log plotting.
            entropy = (1e-6 if jv == 0.0 else 2.5e-5 * cut * jv) + 5e-6
            rows.append({"cut": cut, "J": jv, "entropy": entropy})

    out_stem = tmp_path / "fig_entropy_heatmap_cut_vs_J"
    mod.plot_entropy_heatmap_cut_vs_j(rows, out_stem)

    assert (tmp_path / "fig_entropy_heatmap_cut_vs_J.pdf").is_file()
    assert (tmp_path / "fig_entropy_heatmap_cut_vs_J.png").is_file()
    assert (tmp_path / "fig_entropy_heatmap_cut_vs_J_prl.pdf").is_file()
    assert (tmp_path / "fig_entropy_heatmap_cut_vs_J_prl.png").is_file()

