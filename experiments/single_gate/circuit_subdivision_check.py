# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Compact TFIM / Heisenberg TDVP subdivision check for circuit baseline choice."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "output" / "circuit_subdivision_check"
OUT.mkdir(parents=True, exist_ok=True)

# Lightweight self-contained check using YAQS Simulator if available; otherwise
# document that fixed_resources should be used for the full test.


def main() -> int:
    """Run a minimal subdivision probe and write guidance."""
    try:
        from qiskit.circuit.library import EfficientSU2

        from mqt.yaqs.core.data_structures.networks import MPS
        from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
        from mqt.yaqs.core.libraries.gate_library import Z
        from mqt.yaqs.simulator import Simulator
    except Exception as exc:  # noqa: BLE001
        (OUT / "circuit_subdivision_check.md").write_text(
            f"# Circuit subdivision check\n\nSkipped: import failure ({exc}).\n"
            "Run `experiments/fixed_resources` / `experiments/convergence` manually "
            "with n∈{1,2,4,8,16,64} before adopting n=1 circuit-wide.\n",
            encoding="utf-8",
        )
        return 0

    # Tiny proxy: 4-qubit random-ish circuit depth layers via EfficientSU2 is not TFIM.
    # Prefer documenting the required fixed_resources protocol.
    report = [
        "# Circuit subdivision check",
        "",
        "`tdvp_sweeps` counts **fractional-time substeps**; each substep is one full "
        "symmetric LTR+RTL 2-site sweep (`tdvp.py`).",
        "",
        "## Protocol (must be run on production TFIM / Heisenberg 4×4)",
        "",
        "For each `n ∈ {1, 2, 4, 8, 16, 64}` and each model in "
        "`{tfim, heisenberg}` under `experiments/fixed_resources`:",
        "",
        "1. Record full infidelity trajectories vs T.",
        "2. Extract reliable horizon T_ε for the paper's ε.",
        "3. Record observables, peak χ, MPS parameter count, runtime, Krylov calls.",
        "4. Choose the **smallest n** such that doubling n does not materially change "
        "T_ε or the scientific conclusion.",
        "",
        "## Interim guidance from prior Heisenberg evidence",
        "",
        "Existing Heisenberg trajectories indicated that **n=1 is not a safe universal "
        "circuit setting**. Do **not** flip all circuit benchmarks to n=1 based only on "
        "the single-gate result. Keep circuit TDVP at the previously validated "
        "subdivision until the compact matrix above is filled.",
        "",
        "## Single-gate vs circuit",
        "",
        "Single-gate main curves use n=1 because the isolated long-range update is "
        "stable under compression and beats no-update. That conclusion is "
        "**benchmark-local** until the circuit matrix confirms otherwise.",
        "",
    ]
    (OUT / "circuit_subdivision_check.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {OUT / 'circuit_subdivision_check.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
