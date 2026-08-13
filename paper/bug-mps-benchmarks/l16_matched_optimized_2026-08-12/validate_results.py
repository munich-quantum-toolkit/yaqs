#!/usr/bin/env python3
"""Independent structural and numerical checks for the saved benchmark."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def load_runner():
    spec = importlib.util.spec_from_file_location("l16_benchmark", HERE / "run_benchmark.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load benchmark runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    benchmark = load_runner()
    payload = json.loads((HERE / "raw_results.json").read_text(encoding="utf-8"))
    checks: list[tuple[str, bool, str]] = []

    for model, builder in (
        ("tfim", benchmark.direct_ising_mpo),
        ("hs", benchmark.direct_haldane_shastry_mpo),
    ):
        mpo_matrix = builder().to_sparse_matrix()
        exact_matrix = benchmark.exact_sparse_hamiltonian(model)
        difference = mpo_matrix - exact_matrix
        difference_norm = float(np.sqrt(np.sum(np.abs(difference.data) ** 2))) if difference.nnz else 0.0
        exact_norm = float(np.sqrt(np.sum(np.abs(exact_matrix.data) ** 2)))
        relative = difference_norm / exact_norm
        checks.append((f"{model} MPO matches analytic Hamiltonian", relative < 1e-14, f"relative Frobenius error {relative:.3e}"))

    for model, model_result in payload["models"].items():
        checks.append((f"{model} initial state normalized", abs(model_result["initial_norm"] - 1) < 1e-12, f"norm {model_result['initial_norm']:.16g}"))
        checks.append((f"{model} reference normalized", abs(model_result["reference_norm"] - 1) < 1e-12, f"norm {model_result['reference_norm']:.16g}"))
        for dt, run in model_result["runs"].items():
            steps = run["steps"]
            for method, result in run["methods"].items():
                checks.append((f"{model} dt={dt} {method}: three timing samples", len(result["runtime_samples_seconds"]) == 3, str(result["runtime_samples_seconds"])))
                checks.append((f"{model} dt={dt} {method}: chi cap", result["max_chi"] <= 512, f"max chi {result['max_chi']}"))
                checks.append((f"{model} dt={dt} {method}: norm", abs(result["norm"] - 1) < 2e-8, f"norm {result['norm']:.16g}"))
                expected_calls = (32 if method == "bug" else 57) * steps
                checks.append((f"{model} dt={dt} {method}: Krylov call count", result["krylov_calls"] == expected_calls, f"{result['krylov_calls']} (expected {expected_calls})"))
            bug = run["methods"]["bug"]
            checkpoint_profiles = bug["first_step_bug_checkpoints"]
            expected_stages = {
                "first_half_sweep",
                "first_compression",
                "second_half_sweep",
                "second_compression",
            }
            checks.append((f"{model} dt={dt}: four BUG checkpoints", set(checkpoint_profiles) == expected_stages, ", ".join(checkpoint_profiles)))
            compressed = checkpoint_profiles["first_compression"] + checkpoint_profiles["second_compression"]
            checks.append((f"{model} dt={dt}: min_keep=2 after compression", min(compressed) >= 2, f"minimum retained rank {min(compressed)}"))

    failed = [check for check in checks if not check[1]]
    lines = ["# Benchmark validation", "", f"Passed {len(checks) - len(failed)} of {len(checks)} checks.", ""]
    for name, passed, detail in checks:
        lines.append(f"- {'PASS' if passed else 'FAIL'}: {name} ({detail})")
    (HERE / "VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failed:
        raise SystemExit(f"{len(failed)} validation checks failed")


if __name__ == "__main__":
    main()
