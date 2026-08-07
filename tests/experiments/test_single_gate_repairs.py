# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Regression tests for repaired MPS compression and variational MPO application."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
INDIVIDUAL_GATES = REPO_ROOT / "experiments" / "individual_gates"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(INDIVIDUAL_GATES))

from common import (  # ruff: ignore[module-import-not-at-top-of-file]
    apply_gate_dense_yaqs,
    apply_method,
    digital_params,
    make_pauli_dag_node,
    make_pauli_gate,
    normalized_state_fidelity,
    prepare_initial_state,
)
from config import BOND_PROFILE, Q0, Q1, SEEDS, N  # ruff: ignore[module-import-not-at-top-of-file]

from experiments.variational_mpo import (  # ruff: ignore[module-import-not-at-top-of-file]
    VariationalMPOResult,
    _projected_target_tensor,  # ruff: ignore[import-private-name]
    _variational_fit_reference,  # ruff: ignore[import-private-name]
    apply_variational_mpo_node,
    variational_fit,
)
from mqt.yaqs.core.data_structures.mpo import MPO  # ruff: ignore[module-import-not-at-top-of-file]
from mqt.yaqs.core.data_structures.mps import MPS  # ruff: ignore[module-import-not-at-top-of-file]

GATE_TYPE = "rzz"
SEED = SEEDS[0]


def _tt_svd_from_vec(vec: np.ndarray, length: int, chi_max: int) -> MPS:
    """Construct an independent dense TT-SVD in YAQS site ordering.

    Returns:
        The normalized capped MPS used as a regression reference.
    """
    psi = np.asarray(vec, dtype=np.complex128).reshape([2] * length)
    psi = np.transpose(psi, list(reversed(range(length))))
    tensors = []
    chi_left = 1
    rest = psi
    for site in range(length - 1):
        rest = rest.reshape(chi_left * 2, -1)
        u_mat, singular, v_mat = np.linalg.svd(rest, full_matrices=False)
        keep = min(chi_max, singular.size)
        tensors.append(np.ascontiguousarray(u_mat[:, :keep].reshape(chi_left, 2, keep).transpose(1, 0, 2)))
        chi_left = keep
        rest = (singular[:keep, None] * v_mat[:keep]).reshape([chi_left] + [2] * (length - site - 1))
    tensors.append(np.ascontiguousarray(rest.reshape(chi_left, 2, 1).transpose(1, 0, 2)))
    state = MPS(length, tensors=tensors)
    state.normalize(form="B", decomposition="SVD")
    return state


def _zz_expectation(vec: np.ndarray) -> float:
    probabilities = np.abs(np.asarray(vec, dtype=np.complex128).reshape(-1)) ** 2
    indices = np.arange(probabilities.size, dtype=np.uint64)
    parity = ((indices >> np.uint64(Q0)) ^ (indices >> np.uint64(Q1))) & np.uint64(1)
    signs = 1.0 - 2.0 * parity.astype(np.float64)
    return float(np.dot(probabilities, signs))


@pytest.fixture
def initial() -> dict:
    """Return the deterministic initial state shared by the gate regressions."""
    return prepare_initial_state(SEED)


def test_identity_mpo_application(initial: dict) -> None:
    """θ=0 RZZ MPO application must leave the state unchanged at χ=8."""
    node = make_pauli_dag_node(GATE_TYPE, 0.0, Q0, Q1, N)
    out, _ = apply_method(initial["mps"], node, method="mpo_zipup", chi=8, n_sub=1)
    inf = normalized_state_fidelity(initial["vec"], out.to_vec())["infidelity_normalized"]
    assert inf < 1e-12


@pytest.mark.parametrize("x", [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
def test_infinitesimal_rzz_zipup_continuous(initial: dict, x: float) -> None:
    """Zip-up infidelity must vanish as θ→0 (no finite plateau)."""
    theta = 2.0 * np.pi * x
    gate = make_pauli_gate(GATE_TYPE, theta, Q0, Q1)
    exact = apply_gate_dense_yaqs(initial["vec"], N, Q0, Q1, gate)
    node = make_pauli_dag_node(GATE_TYPE, theta, Q0, Q1, N)
    out, _ = apply_method(initial["mps"], node, method="mpo_zipup", chi=8, n_sub=1)
    inf = normalized_state_fidelity(exact, out.to_vec())["infidelity_normalized"]
    # Must beat a finite 1e-2 plateau and stay near the no-update O(θ²) scale.
    no_update = normalized_state_fidelity(exact, initial["vec"])["infidelity_normalized"]
    assert inf < 1e-2
    assert inf <= max(2.0 * no_update, 1e-14)


def test_chi16_exact_zipup_and_variational(initial: dict) -> None:
    """At χ=16 the gate is exact for zip-up and variational MPO."""
    theta = 2.0 * np.pi * 0.1
    gate = make_pauli_gate(GATE_TYPE, theta, Q0, Q1)
    exact = apply_gate_dense_yaqs(initial["vec"], N, Q0, Q1, gate)
    node = make_pauli_dag_node(GATE_TYPE, theta, Q0, Q1, N)
    zip_out, _ = apply_method(initial["mps"], node, method="mpo_zipup", chi=16, n_sub=1)
    var = apply_variational_mpo_node(
        copy.deepcopy(initial["mps"]),
        node,
        compression_params=digital_params(16, method="mpo_zipup", n_sub=1),
    )
    assert normalized_state_fidelity(exact, zip_out.to_vec())["infidelity_normalized"] < 1e-12
    assert normalized_state_fidelity(exact, var.state.to_vec())["infidelity_normalized"] < 1e-12
    assert var.converged
    assert np.all(np.diff(var.objective_trace) <= 2e-12)
    assert np.all(np.diff(var.update_trace) <= 2e-12)


def test_no_update_analytic_identity(initial: dict) -> None:
    """1-F0 = sin²(θ/2)[1-⟨Z2 Z9⟩²]."""
    theta = 2.0 * np.pi * 1e-4
    exact = apply_gate_dense_yaqs(initial["vec"], N, Q0, Q1, make_pauli_gate(GATE_TYPE, theta, Q0, Q1))
    measured = normalized_state_fidelity(exact, initial["vec"])["infidelity_normalized"]
    zz = _zz_expectation(initial["vec"])
    analytic = float(np.sin(theta / 2.0) ** 2 * (1.0 - zz**2))
    assert abs(measured - analytic) < 1e-14


def test_compress_matches_ttsvd_after_mpo(initial: dict) -> None:
    """Fixed MPS.compress after uncapped MPO must match independent TT-SVD."""
    theta = 2.0 * np.pi * 1e-4
    gate = make_pauli_gate(GATE_TYPE, theta, Q0, Q1)
    exact = apply_gate_dense_yaqs(initial["vec"], N, Q0, Q1, gate)
    unc = copy.deepcopy(initial["mps"])
    MPO.from_gate(gate, N).multiply(unc, compress=False)
    unc.compress(1e-13, max_bond_dim=8, trunc_mode="discarded_weight")
    tt = _tt_svd_from_vec(exact, N, 8)
    inf_c = normalized_state_fidelity(exact, unc.to_vec())["infidelity_normalized"]
    inf_t = normalized_state_fidelity(exact, tt.to_vec())["infidelity_normalized"]
    assert abs(inf_c - inf_t) < 1e-10
    assert max(BOND_PROFILE) == 8


def test_variational_cap_binding_is_monotone(initial: dict) -> None:
    """A cap-binding fit must remain normalized, capped, and no worse than MPO initialization."""
    theta = 2.0 * np.pi * 0.25
    node = make_pauli_dag_node(GATE_TYPE, theta, Q0, Q1, N)
    result = apply_variational_mpo_node(
        initial["mps"],
        node,
        compression_params=digital_params(8, method="mpo_zipup", n_sub=1),
    )
    profile = [result.state.tensors[0].shape[1], *(tensor.shape[2] for tensor in result.state.tensors)]
    assert result.converged
    assert result.initializer_converged[result.best_initializer]
    assert set(result.initializer_converged) == {"mpo_contract_compress", "input"}
    assert max(profile) <= 8
    assert abs(np.linalg.norm(result.state.to_vec()) - 1.0) < 1e-12
    assert result.objective_final <= result.initializer_objectives["mpo_contract_compress"] + 1e-13
    assert np.all(np.diff(result.objective_trace) <= 2e-12)
    assert np.all(np.diff(result.update_trace) <= 2e-12)
    center = result.state.orthogonality_center
    assert center is not None
    assert center in result.state.check_canonical_form()


@pytest.mark.parametrize(
    ("converged", "expected_initializer"),
    [
        ((True, False), "mpo_contract_compress"),
        ((True, True), "input"),
        ((False, False), "input"),
    ],
)
def test_variational_initializer_selection_prioritizes_convergence(
    initial: dict,
    monkeypatch: pytest.MonkeyPatch,
    converged: tuple[bool, bool],
    expected_initializer: str,
) -> None:
    """A lower-residual unfinished start must not displace a converged fit."""
    outcomes = iter(((0.20, converged[0]), (0.10, converged[1])))

    def fake_variational_fit(_target: MPS, initializer: MPS, **_kwargs: Any) -> VariationalMPOResult:
        objective, did_converge = next(outcomes)
        return VariationalMPOResult(
            state=copy.deepcopy(initializer),
            objective_initial=0.30,
            objective_final=objective,
            sweeps=2,
            converged=did_converge,
        )

    monkeypatch.setattr("experiments.variational_mpo.variational_fit", fake_variational_fit)
    node = make_pauli_dag_node(GATE_TYPE, 2.0 * np.pi * 0.01, Q0, Q1, N)
    result = apply_variational_mpo_node(
        initial["mps"],
        node,
        compression_params=digital_params(8, method="mpo_zipup", n_sub=1),
    )

    assert result.best_initializer == expected_initializer
    assert result.converged is any(converged)
    assert result.initializer_converged == {
        "mpo_contract_compress": converged[0],
        "input": converged[1],
    }


def test_variational_fit_accepts_a_strict_improvement(initial: dict) -> None:
    """A stable cap-binding case must exercise and validate an accepted sweep update."""
    theta = 2.0 * np.pi * 0.01
    node = make_pauli_dag_node("ryy", theta, Q0, Q1, N)
    result = apply_variational_mpo_node(
        initial["mps"],
        node,
        compression_params=digital_params(8, method="mpo_zipup", n_sub=1),
    )
    assert result.converged
    assert result.objective_final < result.objective_initial - 1e-6
    assert np.all(np.diff(result.objective_trace) <= 2e-12)
    assert np.all(np.diff(result.update_trace) <= 2e-12)
    expected_objective = 2.0 - 2.0 * np.sqrt(result.fidelity_to_target)
    assert result.objective_final == pytest.approx(expected_objective, abs=2e-12)
    center = result.state.orthogonality_center
    assert center is not None
    assert center in result.state.check_canonical_form()


def test_projected_target_tensor_reproduces_full_complex_overlap(initial: dict) -> None:
    """The projected ket must reproduce the full overlap, including its phase."""
    theta = 2.0 * np.pi * 0.01
    gate = make_pauli_gate("ryy", theta, Q0, Q1)
    target = copy.deepcopy(initial["mps"])
    MPO.from_gate(gate, N).multiply(target, compress=False)
    approximate = copy.deepcopy(initial["mps"])
    MPO.from_gate(gate, N).multiply(
        approximate,
        sim_params=digital_params(8, method="mpo_zipup", n_sub=1),
        compress=True,
    )
    bond = 3
    target.set_canonical_form(bond, decomposition="QR")
    approximate.set_canonical_form(bond, decomposition="QR")
    approximate.tensors[bond] *= np.exp(0.37j)
    projected = _projected_target_tensor(target, approximate, bond)
    approximate_block = np.einsum(
        "pag,qgb->pqab",
        approximate.tensors[bond],
        approximate.tensors[bond + 1],
        optimize=True,
    )
    projected_overlap = np.vdot(projected, approximate_block)
    full_overlap = target.scalar_product(approximate)
    assert abs(full_overlap.imag) > 1e-2
    assert projected_overlap == pytest.approx(full_overlap, abs=2e-12)


def test_cached_sweep_matches_corrected_reference(initial: dict) -> None:
    """Cached environments must reproduce the corrected materialized sweep."""
    theta = 2.0 * np.pi * 0.01
    gate = make_pauli_gate("ryy", theta, Q0, Q1)
    target = copy.deepcopy(initial["mps"])
    MPO.from_gate(gate, N).multiply(target, compress=False)
    params = digital_params(8, method="mpo_zipup", n_sub=1)
    initializer = copy.deepcopy(initial["mps"])
    MPO.from_gate(gate, N).multiply(initializer, sim_params=params, compress=True)
    reference = _variational_fit_reference(
        target,
        initializer,
        compression_params=params,
        max_sweeps=8,
    )
    cached = variational_fit(
        target,
        initializer,
        compression_params=params,
        max_sweeps=8,
    )
    assert reference.converged
    assert cached.converged
    assert cached.objective_final == pytest.approx(reference.objective_final, abs=3e-12)
    comparison = normalized_state_fidelity(reference.state.to_vec(), cached.state.to_vec())
    assert comparison["infidelity_normalized"] < 2e-12
