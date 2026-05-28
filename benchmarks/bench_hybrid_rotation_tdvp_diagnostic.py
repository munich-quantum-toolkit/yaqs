#!/usr/bin/env python
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Generate the hybrid TDVP rotation diagnostic suite (JSON + Markdown).

Run:

    uv run python -m benchmarks.bench_hybrid_rotation_tdvp_diagnostic

Outputs:

    benchmarks/hybrid_rotation_tdvp_diagnostic.json
    benchmarks/hybrid_rotation_tdvp_diagnostic.md
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.hybrid_benchmark_lib import generate_rotation_diagnostic_report


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    report = generate_rotation_diagnostic_report(repo_root / "benchmarks")
    paths = report["meta"]["output_files"]
    print(f"Wrote {paths['json']}")
    print(f"Wrote {paths['markdown']}")


if __name__ == "__main__":
    main()

