#!/usr/bin/env python3
"""Medium-scale paper benchmarks (between smoke and full reproduction)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAVE = ROOT / "save" / "medium"
PYTHON = sys.executable

J_MEDIUM = ",".join(f"{0.1 * i:.1f}" for i in range(21))


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
                "--seed",
                "0",
                "--n-pasts",
                "32",
                "--n-futures",
                "32",
                "--cuts",
                "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20",
                "--j-values",
                J_MEDIUM,
                "--out-dir",
                str(SAVE / "cut_vs_j"),
            ],
        ),
        (
            "convergence",
            [
                PYTHON,
                "convergence.py",
                "--seed",
                "0",
                "--cut",
                "10",
                "--probe-draws",
                "5",
                "--m-values",
                "4,8,16,32,64",
                "--convergence-js",
                "0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0",
                "--out-dir",
                str(SAVE / "convergence"),
            ],
        ),
        (
            "modes",
            [
                PYTHON,
                "modes.py",
                "--seed",
                "0",
                "--cuts",
                "1,5,10,15,20",
                "--m-spectrum",
                "32",
                "--spectrum-draws",
                "2",
                "--spectrum-js",
                J_MEDIUM,
                "--plot-cuts",
                "1,5,10,15,20",
                "--out-dir",
                str(SAVE / "modes"),
            ],
        ),
    ]

    for label, cmd in jobs:
        run(cmd, label=label)

    print(f"\nMedium benchmarks finished. Outputs: {SAVE}", flush=True)


if __name__ == "__main__":
    main()
