# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Validation checks for R_PP(theta) = exp(-i theta P⊗P / 2)."""

from __future__ import annotations

import copy

import numpy as np
from gate_runtime import (
    L_DEFAULT,
    TARGET_BOND_PROFILE,
    _params,
    apply_gate_to_dense_state,
    apply_two_qubit_dense,
    fidelity,
    gate_matrix,
    make_dag_node,
    make_gate,
    phase_align,
    random_mps,
)
from mqt.yaqs.core.libraries.gate_library import X, Y, Z
from mqt.yaqs.digital.digital_tjm import apply_two_qubit_gate
from scipy.linalg import expm

PAULI_MATRICES = {"x": np.asarray(X().matrix), "y": np.asarray(Y().matrix), "z": np.asarray(Z().matrix)}


def independent_r_pp_matrix(gate_type: str, theta: float) -> np.ndarray:
    """Dense gate matrix from exp(-i theta P⊗P / 2)."""
    pauli = gate_type[-1]
    p = PAULI_MATRICES[pauli]
    p2 = np.kron(p, p)
    return expm(-0.5j * theta * p2)


def apply_pauli_tensor(vec: np.ndarray, gate_type: str, q0: int, q1: int) -> np.ndarray:
    """Apply P⊗P on the requested qubit pair."""
    pauli = gate_type[-1]
    p2 = np.kron(PAULI_MATRICES[pauli], PAULI_MATRICES[pauli])
    return apply_gate_to_dense_state(vec, p2, q0, q1, L_DEFAULT)


def check_gate_convention(*, seed: int = 11) -> list[str]:
    """Run gate-convention checks; return list of failure messages."""
    failures: list[str] = []
    rng = np.random.default_rng(seed)
    mps = random_mps(L_DEFAULT, list(TARGET_BOND_PROFILE), rng)
    psi = mps.to_vec()
    q0, q1 = 2, 9

    for gate_type in ("rxx", "ryy", "rzz"):
        for theta in (0.0, np.pi, 2.0 * np.pi):
            gate = make_gate(gate_type, float(theta), q0, q1)
            result = apply_two_qubit_dense(psi, L_DEFAULT, q0, q1, gate)
            indep = independent_r_pp_matrix(gate_type, float(theta))
            yaqs_u = gate_matrix(gate_type, float(theta))
            if float(np.max(np.abs(yaqs_u - indep))) > 1e-10:
                failures.append(f"{gate_type} theta={theta}: YAQS matrix != independent expm")

            norm_after = float(np.linalg.norm(result))
            if abs(norm_after - float(np.linalg.norm(psi))) > 1e-10:
                failures.append(f"{gate_type} theta={theta}: norm not preserved ({norm_after})")

            if theta == 0.0:
                if 1.0 - fidelity(result, psi) > 1e-12:
                    failures.append(f"{gate_type} theta=0: not identity")
            elif abs(theta - np.pi) < 1e-12:
                pauli_state = apply_pauli_tensor(psi, gate_type, q0, q1)
                expected = -1j * pauli_state
                phase = np.vdot(result, expected)
                if abs(phase) < 1e-14:
                    failures.append(f"{gate_type} theta=pi: zero overlap with -i P⊗P state")
                aligned = expected * (phase / abs(phase))
                if float(np.linalg.norm(aligned - result)) > 1e-10:
                    failures.append(f"{gate_type} theta=pi: not -i P⊗P up to phase")
            elif abs(theta - 2.0 * np.pi) < 1e-12:
                aligned = phase_align(psi, result)
                if float(np.linalg.norm(aligned - psi)) > 1e-10:
                    failures.append(f"{gate_type} theta=2pi: not initial state up to phase")

        tdvp = copy.deepcopy(mps)
        apply_two_qubit_gate(tdvp, make_dag_node(gate_type, 0.0, q0, q1, L_DEFAULT), _params(8, gate_mode="tdvp"))
        if 1.0 - fidelity(psi, tdvp.to_vec()) > 1e-12:
            failures.append(f"{gate_type} theta=0: TDVP not identity")

    return failures
