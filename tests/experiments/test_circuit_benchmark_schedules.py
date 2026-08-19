# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Focused validation of the publication circuit protocol."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from experiments.circuit_benchmarks.circuits import (
    build_qiskit_circuit,
    build_schedule,
    build_trotter_step,
    circuit_fingerprint,
    edge_matchings,
    physical_edges,
    snake_index,
    step_qiskit_circuit,
)
from experiments.circuit_benchmarks.common import (
    apply_dense_step,
    initial_basis_string,
    initial_mps,
    initial_vector,
    normalized_state_fidelity,
    parameter_count,
    parameter_count_from_profile,
)
from experiments.circuit_benchmarks.config import CASE_KEYS, CASES, DT, METHOD_TO_GATE_MODE, N_STEPS, BenchmarkCase, N


def test_projection_benchmark_default_uses_full_tdvp() -> None:
    """The baseline Projection campaign sends every supported gate through TDVP."""
    assert METHOD_TO_GATE_MODE["gate_local_2tdvp"] == "full-tdvp"


def test_four_frozen_cases_have_sixteen_sites() -> None:
    assert CASE_KEYS == ("ising_1d", "heisenberg_1d", "ising_2d", "heisenberg_2d")
    assert all(case.n_qubits == N for case in CASES.values())
    assert len(build_schedule(CASES["ising_1d"])) == N_STEPS


def test_snake_order_and_open_square_edges() -> None:
    case = CASES["ising_2d"]
    assert [snake_index(0, col, case.cols) for col in range(case.cols)] == [0, 1, 2, 3]
    assert [snake_index(1, col, case.cols) for col in range(case.cols)] == [7, 6, 5, 4]
    edges = physical_edges(case)
    assert len(edges) == 24
    assert len(set(edges)) == 24
    assert any(abs(right - left) > 1 for left, right in edges)


@pytest.mark.parametrize("case_key", CASE_KEYS)
def test_edge_coloring_is_disjoint_and_complete(case_key: str) -> None:
    case = CASES[case_key]
    matchings = edge_matchings(case)
    expected_edges = 15 if case.geometry == "1d" else 24
    assert sum(len(matching.edges) for matching in matchings) == expected_edges
    for matching in matchings:
        sites = [site for edge in matching.edges for site in edge]
        assert len(sites) == len(set(sites))


@pytest.mark.parametrize("case_key", ["heisenberg_1d", "heisenberg_2d"])
def test_heisenberg_step_is_groupwise_palindromic(case_key: str) -> None:
    case = CASES[case_key]
    step = build_trotter_step(case)
    layers = step.layers
    assert len(layers) == 2 * (3 * len(edge_matchings(case))) - 1
    assert layers[len(layers) // 2].time_fraction == 1.0
    for forward, backward in zip(
        layers[: len(layers) // 2],
        reversed(layers[len(layers) // 2 + 1 :]),
        strict=True,
    ):
        assert forward.label == backward.label
        assert forward.time_fraction == backward.time_fraction == 0.5
        assert forward.gates == tuple(reversed(backward.gates))


@pytest.mark.parametrize("case_key", ["heisenberg_1d", "heisenberg_2d"])
def test_heisenberg_integrated_angle_matches_hamiltonian(case_key: str) -> None:
    case = CASES[case_key]
    totals: dict[tuple[str, tuple[int, ...]], float] = {}
    for gate in build_trotter_step(case).gates:
        key = (gate.name, gate.qubits)
        totals[key] = totals.get(key, 0.0) + gate.theta
    assert {name for name, _ in totals} == {"rxx", "ryy", "rzz"}
    assert len(totals) == 3 * len(physical_edges(case))
    assert all(theta == pytest.approx(-2.0 * DT) for theta in totals.values())


def test_ising_step_has_strang_bond_field_bond_order() -> None:
    case = CASES["ising_2d"]
    layers = build_trotter_step(case).layers
    assert [layer.label for layer in layers] == ["ising_bonds", "transverse_field", "ising_bonds"]
    assert [layer.time_fraction for layer in layers] == [0.5, 1.0, 0.5]
    assert layers[0].gates == tuple(reversed(layers[2].gates))


def test_initial_states_match_mps_vectors_and_physical_conventions() -> None:
    assert initial_basis_string(CASES["ising_1d"]) == "0" * N
    assert initial_basis_string(CASES["heisenberg_1d"]) == "01" * (N // 2)
    assert initial_basis_string(CASES["heisenberg_2d"]) == "01" * (N // 2)
    for case in CASES.values():
        assert np.array_equal(initial_vector(case), initial_mps(case).to_vec())


def test_dense_step_uses_identical_ordered_qiskit_circuit() -> None:
    small_case = BenchmarkCase("small", "Small Ising", "ising", "1d", 1, 4, "zeros")
    step = build_trotter_step(small_case)
    initial = initial_vector(small_case)
    sequential = apply_dense_step(initial, step, small_case.n_qubits)
    materialized = np.asarray(
        Statevector(initial).evolve(step_qiskit_circuit(step, small_case.n_qubits)).data,
        dtype=np.complex128,
    )
    assert np.linalg.norm(sequential - materialized) < 1e-14


def test_full_circuit_matches_repeated_step_circuits() -> None:
    small_case = BenchmarkCase("small", "Small Heisenberg", "heisenberg", "1d", 1, 4, "neel")
    schedule = build_schedule(small_case, steps=2)
    initial = initial_vector(small_case)
    repeated = Statevector(initial)
    for step in schedule:
        repeated = repeated.evolve(step_qiskit_circuit(step, small_case.n_qubits))
    full = Statevector(initial).evolve(build_qiskit_circuit(small_case, schedule=schedule))
    assert np.linalg.norm(np.asarray(repeated.data) - np.asarray(full.data)) < 1e-14


def test_fidelity_and_parameter_metrics() -> None:
    state = initial_mps(CASES["ising_1d"])
    vector = state.to_vec()
    metrics = normalized_state_fidelity(vector, vector * np.exp(0.37j))
    assert metrics["infidelity_normalized"] == pytest.approx(0.0, abs=1e-14)
    profile = [1] * (state.length + 1)
    assert parameter_count(state) == 2 * state.length
    assert parameter_count_from_profile(profile) == parameter_count(state)


def test_circuit_fingerprint_tracks_gate_order() -> None:
    case = CASES["ising_1d"]
    one_step = build_schedule(case, steps=1)
    two_steps = build_schedule(case, steps=2)
    assert circuit_fingerprint(case, one_step) == circuit_fingerprint(case, one_step)
    assert circuit_fingerprint(case, one_step) != circuit_fingerprint(case, two_steps)
