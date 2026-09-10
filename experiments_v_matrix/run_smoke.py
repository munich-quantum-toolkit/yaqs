#!/usr/bin/env python3
"""Quick smoke run of all paper benchmarks (small grids, serial execution)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAVE = ROOT / "save" / "smoke"
PYTHON = sys.executable


def run(cmd: list[str], *, label: str) -> None:
    """Run one benchmark subprocess."""
    print(f"\n=== {label} ===", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    SAVE.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, list[str]]] = [
        (
            "cut vs J",
            [
                PYTHON,
                "cut_vs_j.py",
                "--no-parallel",
                "--n-pasts",
                "8",
                "--n-futures",
                "8",
                "--cuts",
                "1,2,3,4",
                "--j-values",
                "0.0,0.5,1.0,1.5,2.0",
                "--out-dir",
                str(SAVE / "cut_vs_j"),
            ],
        ),
        (
            "finite size",
            [
                PYTHON,
                "finite_size.py",
                "--no-parallel",
                "--n-pasts",
                "6",
                "--n-futures",
                "6",
                "--l-values",
                "2,3",
                "--j-values",
                "0.5,1.0",
                "--k",
                "4",
                "--profile-l",
                "2,3",
                "--out-dir",
                str(SAVE / "finite_size"),
            ],
        ),
        (
            "ell delay",
            [
                PYTHON,
                "ell_delay.py",
                "--no-parallel",
                "--n-pasts",
                "6",
                "--n-futures",
                "6",
                "--ells",
                "0,1,2",
                "--past-len",
                "3",
                "--future-len",
                "2",
                "--out-dir",
                str(SAVE / "ell_delay"),
            ],
        ),
        (
            "convergence",
            [
                PYTHON,
                "convergence.py",
                "--no-parallel",
                "--cut",
                "2",
                "--probe-draws",
                "2",
                "--m-values",
                "4,8,16",
                "--convergence-js",
                "0.4,1.0,2.0",
                "--out-dir",
                str(SAVE / "convergence"),
            ],
        ),
        (
            "modes",
            [
                PYTHON,
                "modes.py",
                "--no-parallel",
                "--cuts",
                "1,2",
                "--m-spectrum",
                "8",
                "--spectrum-draws",
                "1",
                "--spectrum-js",
                "0.0,0.5,1.0,1.5,2.0",
                "--plot-cuts",
                "1,2",
                "--out-dir",
                str(SAVE / "modes"),
            ],
        (
            "pt cut reference",
            [
                PYTHON,
                "pt_cut_reference.py",
                "--plot-prx-only",
                "--out-dir",
                str(SAVE / "pt_cut_reference"),
            ],
        ),
    ]

    for label, cmd in jobs:
        run(cmd, label=label)

    print(f"\nSmoke benchmarks finished. Outputs: {SAVE}", flush=True)


if __name__ == "__main__":
    main()
