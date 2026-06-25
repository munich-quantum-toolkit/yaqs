#!/usr/bin/env python3
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Run all operational-memory paper figure benchmarks in quick mode.

Recreates a small version of each published plot:
1. S_V heatmap vs cut and J
2. Middle-cut S_V(L, k) heatmap
3. Probe convergence S_V(m) and spectrum / rank vs J
5. S_V vs memory delay ell (shared past/future pools)
6. Gap / operational-memory horizon heatmap (τ vs J)

Figures are written under ``experiments/*_quick_results/``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = (
    ("cut × J heatmap", "benchmark_entropy_vs_j_by_cut.py"),
    ("middle-cut L × k", "benchmark_entropy_middlecut_vs_L_k.py"),
    ("convergence + spectrum", "benchmark_entropy_convergence_and_spectrum.py"),
    ("delay ℓ curves (shared pools)", "benchmark_entropy_vs_j_by_ell.py"),
    ("gap / τ heatmap (paper)", "benchmark_entropy_vs_j_by_gap.py"),
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parallel", action="store_true", default=False, help="Enable MCWF rollout parallelism.")
    p.add_argument(
        "--only", type=str, default="", help="Comma-separated script stems to run (e.g. benchmark_entropy_vs_j_by_cut)."
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    here = Path(__file__).resolve().parent
    only = {s.strip() for s in str(args.only).split(",") if s.strip()}
    parallel_flag = ["--parallel"] if args.parallel else ["--no-parallel"]

    for _label, script in SCRIPTS:
        stem = script.removesuffix(".py")
        if only and stem not in only:
            continue
        cmd = [sys.executable, str(here / script), "--quick", *parallel_flag]
        subprocess.run(cmd, cwd=here, check=True)

    for _, script in SCRIPTS:
        stem = script.removesuffix(".py")
        out = here / f"{stem}_quick_results"
        pngs = sorted(out.glob("fig*.png")) if out.is_dir() else []
        if pngs:
            for _p in pngs:
                pass


if __name__ == "__main__":
    main()
