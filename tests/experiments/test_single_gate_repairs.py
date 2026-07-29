# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Regression tests for repaired MPS.compress / MPO zip-up / variational MPO."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

SINGLE_GATE = Path(__file__).resolve().parents[2] / "experiments" / "single_gate"
sys.path.insert(0, str(SINGLE_GATE))

from config import GATE_TYPE, Q0, Q1, SEED  # noqa: E402
from gate_runtime import (  # noqa: E402
    L_DEFAULT,
    TARGET_BOND_PROFILE,
    apply_method,
    apply_two_qubit_dense,
    make_dag_node,
    make_gate,
    normalized_state_fidelity,
    prepare_initial_state,
)
from variational import apply_variational_mpo_gate, tt_svd_from_vec  # noqa: E402

from mqt.yaqs.core.data_structures.mpo import MPO  # noqa: E402
from mqt.yaqs.core.libraries.gate_library import Z  # noqa: E402


def _zz_expectation(vec: np.ndarray) -> float:
    z = np.asarray(Z().matrix, dtype=np.complex128)
    z2 = np.kron(z, z)
    from gate_runtime import apply_gate_to_dense_state

    return float(np.real(np.vdot(vec, apply_gate_to_dense_state(vec, z2, Q0, Q1, L_DEFAULT))))


@pytest.fixture
def initial() -> dict:
    return prepare_initial_state(SEED)


def test_identity_mpo_application(initial: dict) -> None:
    """θ=0 RZZ MPO application must leave the state unchanged at χ=8."""
    node = make_dag_node(GATE_TYPE, 0.0, Q0, Q1, L_DEFAULT)
    out, _, _ = apply_method(initial["mps"], node, method="mpo_zipup", chi=8, substeps=1)
    inf = normalized_state_fidelity(initial["vec"], out.to_vec())["infidelity_normalized"]
    assert inf < 1e-12


@pytest.mark.parametrize("x", [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
def test_infinitesimal_rzz_zipup_continuous(initial: dict, x: float) -> None:
    """Zip-up infidelity must vanish as θ→0 (no finite plateau)."""
    theta = 2.0 * np.pi * x
    exact = apply_two_qubit_dense(initial["vec"], L_DEFAULT, Q0, Q1, make_gate(GATE_TYPE, theta, Q0, Q1))
    node = make_dag_node(GATE_TYPE, theta, Q0, Q1, L_DEFAULT)
    out, _, _ = apply_method(initial["mps"], node, method="mpo_zipup", chi=8, substeps=1)
    inf = normalized_state_fidelity(exact, out.to_vec())["infidelity_normalized"]
    # Must beat a finite 1e-2 plateau and stay near the no-update O(θ²) scale.
    no_update = normalized_state_fidelity(exact, initial["vec"])["infidelity_normalized"]
    assert inf < 1e-2
    assert inf <= max(2.0 * no_update, 1e-14)


def test_chi16_exact_zipup_and_variational(initial: dict) -> None:
    """At χ=16 the gate is exact for zip-up and variational MPO."""
    theta = 2.0 * np.pi * 0.1
    exact = apply_two_qubit_dense(initial["vec"], L_DEFAULT, Q0, Q1, make_gate(GATE_TYPE, theta, Q0, Q1))
    node = make_dag_node(GATE_TYPE, theta, Q0, Q1, L_DEFAULT)
    zip_out, _, _ = apply_method(initial["mps"], node, method="mpo_zipup", chi=16, substeps=1)
    var = apply_variational_mpo_gate(copy.deepcopy(initial["mps"]), node, chi=16)
    assert normalized_state_fidelity(exact, zip_out.to_vec())["infidelity_normalized"] < 1e-12
    assert normalized_state_fidelity(exact, var.state.to_vec())["infidelity_normalized"] < 1e-12


def test_no_update_analytic_identity(initial: dict) -> None:
    """1-F0 = sin²(θ/2)[1-⟨Z2 Z9⟩²]."""
    theta = 2.0 * np.pi * 1e-4
    exact = apply_two_qubit_dense(initial["vec"], L_DEFAULT, Q0, Q1, make_gate(GATE_TYPE, theta, Q0, Q1))
    measured = normalized_state_fidelity(exact, initial["vec"])["infidelity_normalized"]
    zz = _zz_expectation(initial["vec"])
    analytic = float(np.sin(theta / 2.0) ** 2 * (1.0 - zz**2))
    assert abs(measured - analytic) < 1e-14


def test_compress_matches_ttsvd_after_mpo(initial: dict) -> None:
    """Fixed MPS.compress after uncapped MPO must match independent TT-SVD."""
    theta = 2.0 * np.pi * 1e-4
    exact = apply_two_qubit_dense(initial["vec"], L_DEFAULT, Q0, Q1, make_gate(GATE_TYPE, theta, Q0, Q1))
    gate = make_gate(GATE_TYPE, theta, Q0, Q1)
    unc = copy.deepcopy(initial["mps"])
    MPO.from_gate(gate, L_DEFAULT).multiply(unc, compress=False)
    unc.compress(1e-13, max_bond_dim=8, trunc_mode="discarded_weight")
    tt = tt_svd_from_vec(exact, L_DEFAULT, 8)
    inf_c = normalized_state_fidelity(exact, unc.to_vec())["infidelity_normalized"]
    inf_t = normalized_state_fidelity(exact, tt.to_vec())["infidelity_normalized"]
    assert abs(inf_c - inf_t) < 1e-10
    assert max(TARGET_BOND_PROFILE) == 8
