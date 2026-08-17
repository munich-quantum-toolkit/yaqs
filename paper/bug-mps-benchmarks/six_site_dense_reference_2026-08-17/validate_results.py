#!/usr/bin/env python3
"""Validate the saved six-site dense-reference benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent

EXPECTED_ERRORS = {
    "0.1": (0.1353, 0.06717, 0.06717),
    "0.05": (0.06729, 0.03345, 0.03345),
    "0.025": (0.03350, 0.01668, 0.01668),
    "0.0125": (0.01670, 0.008333, 0.008333),
    "0.00625": (0.008335, 0.004165, 0.004165),
}
VARIANTS = ("one_sweep_center", "two_sweeps_center", "two_sweeps_previous_basis")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "raw_results.json")
    parser.add_argument("--report", type=Path, default=HERE / "VALIDATION.md")
    return parser.parse_args()


def add_check(checks: list[tuple[str, bool, str]], name: str, passed: bool, detail: str) -> None:
    """Append one named validation result."""
    checks.append((name, bool(passed), detail))


def validate(payload: dict[str, Any]) -> list[tuple[str, bool, str]]:
    """Return all structural and numerical validation checks."""
    checks: list[tuple[str, bool, str]] = []
    structural = payload["structural_checks"]

    for key in (
        "mpo_dense_relative_frobenius_error",
        "reflected_mpo_dense_relative_frobenius_error",
    ):
        value = float(structural[key])
        add_check(checks, key, value < 2e-15, f"{value:.6e}")
    for key in ("site_ordering_gap", "reflection_asymmetry_residual"):
        value = float(structural[key])
        add_check(checks, key, abs(value - 0.4338667962246875) < 1e-12, f"{value:.15f}")
    add_check(
        checks,
        "dense_reference_norm_error",
        float(structural["dense_reference_norm_error"]) < 2e-14,
        f"{float(structural['dense_reference_norm_error']):.6e}",
    )
    add_check(
        checks,
        "site-0-LSB initial-state ordering",
        structural["initial_state_nonzero_index"] == structural["expected_initial_state_nonzero_index"] == 50,
        str(structural["initial_state_nonzero_index"]),
    )
    for key in (
        "preparation_preserves_input_tensors",
        "initial_input_tensors_preserved",
        "all_results_finite",
        "all_endpoints_restored",
    ):
        add_check(checks, key, structural[key] is True, str(structural[key]))

    runs = payload["runs"]
    add_check(checks, "complete published timestep grid", set(runs) == set(EXPECTED_ERRORS), ", ".join(runs))
    for dt, expected_row in EXPECTED_ERRORS.items():
        if dt not in runs:
            continue
        run = runs[dt]
        for variant, expected in zip(VARIANTS, expected_row, strict=True):
            result = run["variants"][variant]
            error = float(result["phase_aligned_state_error"])
            add_check(
                checks,
                f"h={dt} {variant} reproduces manuscript table",
                abs(error - expected) <= max(5e-7, 5e-4 * expected),
                f"{error:.9e} (table {expected:.9e})",
            )
            add_check(
                checks,
                f"h={dt} {variant} endpoint",
                result["endpoint_restored"] is True,
                f"center {result['orthogonality_center']}",
            )
        difference = float(run["two_sweep_variant_phase_aligned_difference"])
        add_check(checks, f"h={dt} two-sweep variants agree", difference < 5e-7, f"{difference:.6e}")

    ordered_dts = sorted(EXPECTED_ERRORS, key=float, reverse=True)
    for variant in VARIANTS:
        errors = [float(runs[dt]["variants"][variant]["phase_aligned_state_error"]) for dt in ordered_dts if dt in runs]
        for coarse, fine in zip(errors[:-1], errors[1:], strict=True):
            ratio = coarse / fine
            add_check(checks, f"{variant} refinement ratio near two", 1.9 < ratio < 2.1, f"{ratio:.6f}")
    return checks


def main() -> None:
    """Validate the result JSON and write a human-readable report."""
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    checks = validate(payload)
    failures = [check for check in checks if not check[1]]
    lines = [
        "# Six-site dense-reference validation",
        "",
        f"Passed {len(checks) - len(failures)} of {len(checks)} checks.",
        "",
    ]
    lines.extend(f"- {'PASS' if passed else 'FAIL'}: {name} ({detail})" for name, passed, detail in checks)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(f"{len(failures)} validation checks failed; see {args.report}")
    print(f"Passed all {len(checks)} checks; wrote {args.report}")


if __name__ == "__main__":
    main()
