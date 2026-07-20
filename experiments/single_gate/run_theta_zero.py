# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""CLI for θ=0 and identity-limit diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import OUTPUT_DIR
from theta_zero_diagnostics import run_and_save


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run θ=0 identity-limit diagnostics.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    report = run_and_save(args.output_dir.resolve())
    print(f"Gate construction: {'PASS' if report.summary['gate_construction_pass'] else 'FAIL'}")
    print(f"Initial state: {'PASS' if report.summary['initial_state_pass'] else 'FAIL'}")
    print(f"θ=0 TDVP/MPO: {'PASS' if report.summary['theta_zero_tdvp_mpo_pass'] else 'FAIL'}")
    print(f"θ=0 TEBD χ=16: {'PASS' if report.summary['theta_zero_tebd_chi16_pass'] else 'FAIL'}")
    print(f"Wrote {args.output_dir / 'theta_zero_diagnostics.csv'}")
    print(f"Wrote {args.output_dir / 'theta_zero_diagnostics.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
