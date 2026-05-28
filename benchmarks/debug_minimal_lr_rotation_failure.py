#!/usr/bin/env python
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Debug the minimal hybrid LR TDVP rotation failure gate-by-gate.

Run:

    uv run python -m benchmarks.debug_minimal_lr_rotation_failure
"""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit

from benchmarks.hybrid_benchmark_lib import debug_gate_by_gate


def build_minimal_failing() -> QuantumCircuit:
    qc = QuantumCircuit(10)
    qc.rxx(0.21, 2, 9)
    qc.ryy(0.25, 3, 8)
    return qc


def main() -> None:
    qc = build_minimal_failing()

    print("=== HYBRID pad=None (gate-by-gate, direct MPS path) ===")
    debug_gate_by_gate(qc, mode="hybrid", pad=None)

    print("")
    print("=== HYBRID pad=2 (gate-by-gate, direct MPS path) ===")
    debug_gate_by_gate(qc, mode="hybrid", pad=2)

    print("")
    print("=== TEBD (gate-by-gate, direct MPS path) ===")
    debug_gate_by_gate(qc, mode="tebd", pad=None)


if __name__ == "__main__":
    main()

