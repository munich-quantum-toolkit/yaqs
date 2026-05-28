#!/usr/bin/env python
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Generate hybrid TDVP benchmark reports (JSON + Markdown).

Run:

    uv run python -m tests.digital.bench_hybrid_report

Outputs:

    benchmarks/hybrid_tdvp_benchmark.json
    benchmarks/hybrid_tdvp_benchmark.md
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.hybrid_benchmark_lib import DEFAULT_BENCHMARK_CONFIG, generate_report  # type: ignore[unused-ignore]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Run benchmark suites and print output paths."""
    output_dir = _REPO_ROOT / "benchmarks"
    report = generate_report(output_dir, config=DEFAULT_BENCHMARK_CONFIG)
    paths = report["meta"]["output_files"]
    print(f"Wrote {paths['json']}")
    print(f"Wrote {paths['markdown']}")


if __name__ == "__main__":
    main()
