# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the Dissipation module."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.core import linalg
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.core.libraries.noise_library import PauliX, PauliZ
from mqt.yaqs.core.methods.dissipation import apply_dissipation, is_adjacent, is_longrange, is_pauli

rng = np.random.default_rng()


def test_apply_dissipation_one_site_canonical_0() -> None:
    """Test that apply_dissipation correctly shifts the MPS to be site-canonical at site 0.

    This test constructs a simple product-state MPS of length 3, where each tensor is of shape (pdim, 1, 1),
    representing an unentangled state. A minimal NoiseModel with one jump operator is created with a small strength,
    and apply_dissipation is applied with a small time step dt. Finally, the test checks that the orthogonality
    center of the MPS is shifted to site 0, as expected.
    """
    # 1) Create a simple product-state MPS of length 3.
    length = 3
    pdim = 2
    tensors = []
    for _ in range(length):
        # Create a random 2-element vector, normalize it, and reshape to (pdim, 1, 1)
        vec = rng.random(pdim).astype(complex)
        vec /= np.linalg.norm(vec)
        tensor = np.asarray(vec.reshape(pdim, 1, 1), dtype=np.complex128)
        tensors.append(tensor)

    state = MPS(length=length, tensors=tensors, physical_dimensions=[pdim] * length)

    noise_model = NoiseModel([
        {"name": name, "sites": [i], "strength": 0.1} for i in range(length) for name in ["lowering", "pauli_z"]
    ])
    dt = 0.1
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)

    apply_dissipation(state, noise_model, dt, sim_params)

    canonical_site = state.orthogonality_center
    assert canonical_site == 0, (
        f"MPS should be site-canonical at site 0 after apply_dissipation, but got canonical site: {canonical_site}"
    )


def test_apply_dissipation_two_site_canonical_0() -> None:
    """Test that apply_dissipation correctly shifts the MPS to be site-canonical at site 0.

    This test constructs a simple product-state MPS of length 3, where each tensor is of shape (pdim, 1, 1),
    representing an unentangled state. A minimal NoiseModel with two 2-site jump operators is created
    with a small strength, and apply_dissipation is applied with a small time step dt.
    Finally, the test checks that the orthogonality center of the MPS is shifted to site 0, as expected.
    """
    # 1) Create a simple product-state MPS of length 3.
    length = 3
    pdim = 2
    tensors = []
    for _ in range(length):
        # Create a random 2-element vector, normalize it, and reshape to (pdim, 1, 1)
        vec = rng.random(pdim).astype(complex)
        vec /= np.linalg.norm(vec)
        tensor = np.asarray(vec.reshape(pdim, 1, 1), dtype=np.complex128)
        tensors.append(tensor)

    state = MPS(length=length, tensors=tensors, physical_dimensions=[pdim] * length)

    noise_model = NoiseModel([
        {"name": name, "sites": [i, i + 1], "strength": 0.1}
        for i in range(length - 1)
        for name in ["crosstalk_xx", "crosstalk_yy"]
    ])
    dt = 0.1
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)

    apply_dissipation(state, noise_model, dt, sim_params)

    canonical_site = state.orthogonality_center
    assert canonical_site == 0, (
        f"MPS should be site-canonical at site 0 after apply_dissipation, but got canonical site: {canonical_site}"
    )


def test_is_adjacent_and_is_longrange() -> None:
    """Test adjacency helpers for two-site processes.

    Verifies that `is_adjacent` returns True for nearest neighbors and False otherwise,
    and that `is_longrange` returns True only for non-neighbor pairs.
    """
    proc_adj = {"sites": [0, 1]}
    proc_adj_unsorted = {"sites": [2, 1]}
    proc_long = {"sites": [0, 2]}
    proc_far = {"sites": [1, 3]}

    assert is_adjacent(proc_adj) is True
    assert is_adjacent(proc_adj_unsorted) is True
    assert is_adjacent(proc_long) is False
    assert is_adjacent(proc_far) is False

    assert is_longrange(proc_adj) is False
    assert is_longrange(proc_adj_unsorted) is False
    assert is_longrange(proc_long) is True
    assert is_longrange(proc_far) is True


def test_apply_dissipation_zero_noise_recenters_tracked_gauge() -> None:
    """Zero-strength dissipation recenters a tracked MPS at site 0 without applying noise."""
    state = MPS(3, state="zeros")
    state.set_center(2)
    noise_model = NoiseModel([{"name": "pauli_z", "sites": [0], "strength": 0.0}])
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)

    apply_dissipation(state, noise_model, dt=0.1, sim_params=sim_params)

    assert state.orthogonality_center == 0


def test_apply_dissipation_zero_noise_unknown_gauge() -> None:
    """Zero-strength dissipation canonicalizes at site 0 when the gauge is unknown."""
    state = MPS(3, state="haar-random", pad=2)
    state.set_center(None)
    noise_model = NoiseModel([{"name": "pauli_z", "sites": [0], "strength": 0.0}])
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)

    apply_dissipation(state, noise_model, dt=0.1, sim_params=sim_params)

    assert state.orthogonality_center == 0


def test_apply_dissipation_unknown_gauge_with_noise() -> None:
    """Noisy dissipation restores a tracked center from an unknown starting gauge."""
    state = MPS(3, state="haar-random", pad=2)
    state.set_center(None)
    noise_model = NoiseModel([{"name": "pauli_z", "sites": [0], "strength": 0.1}])
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)

    apply_dissipation(state, noise_model, dt=0.1, sim_params=sim_params)

    assert state.orthogonality_center == 0


def test_apply_dissipation_longrange_non_pauli_raises() -> None:
    """Non-Pauli long-range two-site dissipation is not implemented."""
    state = MPS(3, state="zeros")
    lowering = np.array([[0, 0], [1, 0]], dtype=np.complex128)
    noise_model = NoiseModel([
        {"name": "custom_lr", "sites": [0, 2], "strength": 0.1, "factors": (lowering, lowering)},
    ])
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)

    with pytest.raises(NotImplementedError, match="Long-range processes"):
        apply_dissipation(state, noise_model, dt=0.1, sim_params=sim_params)


def test_is_pauli_structure_phased_vs_scaled() -> None:
    """Unit-modulus phased Paulis count; scaled Paulis do not."""
    phased = NoiseModel([
        {"name": "custom", "sites": [0], "strength": 0.1, "matrix": 1j * PauliX.matrix},
    ]).processes[0]
    scaled = NoiseModel([
        {"name": "custom", "sites": [0], "strength": 0.1, "matrix": 2.0 * PauliX.matrix},
    ]).processes[0]
    longrange = NoiseModel([
        {"name": "longrange_crosstalk_xy", "sites": [0, 2], "strength": 0.1},
    ]).processes[0]
    adjacent = NoiseModel([
        {"name": "crosstalk_xz", "sites": [0, 1], "strength": 0.1},
    ]).processes[0]
    scaled_kron = NoiseModel([
        {
            "name": "custom",
            "sites": [0, 1],
            "strength": 0.1,
            "matrix": 2.0 * np.kron(PauliX.matrix, PauliZ.matrix),
        },
    ]).processes[0]

    assert is_pauli(phased) is True
    assert is_pauli(scaled) is False
    assert is_pauli(longrange) is True
    assert is_pauli(adjacent) is True
    assert is_pauli(scaled_kron) is False


def test_apply_dissipation_longrange_crosstalk_xy() -> None:
    """Documented longrange_crosstalk_xy uses the Pauli dissipator shortcut."""
    state = MPS(3, state="zeros")
    before = [t.copy() for t in state.tensors]
    noise_model = NoiseModel([
        {"name": "longrange_crosstalk_xy", "sites": [0, 2], "strength": 0.2},
    ])
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)
    apply_dissipation(state, noise_model, dt=0.1, sim_params=sim_params)
    assert state.orthogonality_center == 0
    # Scalar Pauli dissipator scales tensors; state remains product-like.
    assert any(not np.allclose(a, b) for a, b in zip(before, state.tensors, strict=True))


def test_apply_dissipation_independent_of_noncommuting_order() -> None:
    """Same-site channels with noncommuting L†L use one expm of the summed generator."""
    lowering = np.array([[0, 1], [0, 0]], dtype=np.complex128)  # L†L = |1><1|
    mixed = np.array([[0, 1], [1, 1]], dtype=np.complex128)  # L†L = [[1,1],[1,2]]
    a = np.conj(lowering).T @ lowering
    b = np.conj(mixed).T @ mixed
    assert not np.allclose(a @ b, b @ a)

    gamma_a, gamma_b, dt = 0.4, 0.3, 0.2
    procs_fwd = [
        {"name": "a", "sites": [0], "strength": gamma_a, "matrix": lowering},
        {"name": "b", "sites": [0], "strength": gamma_b, "matrix": mixed},
    ]
    procs_rev = [procs_fwd[1], procs_fwd[0]]
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)

    amp = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.complex128)

    def _state() -> MPS:
        s = MPS(2, state="zeros")
        s.tensors[0] = s.tensors[0].astype(np.complex128)
        s.tensors[0][0, 0, 0] = amp[0]
        s.tensors[0][1, 0, 0] = amp[1]
        return s

    state_fwd = _state()
    state_rev = _state()
    apply_dissipation(state_fwd, NoiseModel(procs_fwd), dt=dt, sim_params=sim_params)
    apply_dissipation(state_rev, NoiseModel(procs_rev), dt=dt, sim_params=sim_params)

    diff = sum(np.linalg.norm(x - y) for x, y in zip(state_fwd.tensors, state_rev.tensors, strict=True))
    assert diff == pytest.approx(0.0, abs=1e-12)

    # Match a single local expm of the summed generator (not sequential expm products).
    # Site 1 remains |0>; contract both tensors for a gauge-independent comparison.
    generator = gamma_a * a + gamma_b * b
    expected = np.kron(linalg.expm(-0.5 * dt * generator) @ amp, np.array([1.0, 0.0], dtype=np.complex128))
    got = np.einsum("alr,brc->ab", state_fwd.tensors[0], state_fwd.tensors[1]).reshape(-1)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_apply_dissipation_adjacent_independent_of_noncommuting_order() -> None:
    """Adjacent channels with noncommuting L†L use one expm of the summed generator."""
    lowering = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    mixed = np.array([[0, 1], [1, 1]], dtype=np.complex128)
    la = np.kron(lowering, np.eye(2, dtype=np.complex128))
    lb = np.kron(mixed, np.eye(2, dtype=np.complex128))
    a = np.conj(la).T @ la
    b = np.conj(lb).T @ lb
    assert not np.allclose(a @ b, b @ a)

    gamma_a, gamma_b, dt = 0.4, 0.3, 0.2
    procs_fwd = [
        {"name": "a", "sites": [0, 1], "strength": gamma_a, "matrix": la},
        {"name": "b", "sites": [0, 1], "strength": gamma_b, "matrix": lb},
    ]
    procs_rev = [procs_fwd[1], procs_fwd[0]]
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0, max_bond_dim=16, svd_threshold=1e-14)

    amp = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.complex128)
    vec = np.kron(amp, amp)

    def _state() -> MPS:
        s = MPS(2, state="zeros")
        s.tensors[0] = s.tensors[0].astype(np.complex128)
        s.tensors[1] = s.tensors[1].astype(np.complex128)
        s.tensors[0][0, 0, 0] = amp[0]
        s.tensors[0][1, 0, 0] = amp[1]
        s.tensors[1][0, 0, 0] = amp[0]
        s.tensors[1][1, 0, 0] = amp[1]
        return s

    state_fwd = _state()
    state_rev = _state()
    apply_dissipation(state_fwd, NoiseModel(procs_fwd), dt=dt, sim_params=sim_params)
    apply_dissipation(state_rev, NoiseModel(procs_rev), dt=dt, sim_params=sim_params)

    diff = sum(np.linalg.norm(x - y) for x, y in zip(state_fwd.tensors, state_rev.tensors, strict=True))
    assert diff == pytest.approx(0.0, abs=1e-10)

    generator = gamma_a * a + gamma_b * b
    expected = linalg.expm(-0.5 * dt * generator) @ vec
    got = np.einsum("alr,brc->ab", state_fwd.tensors[0], state_fwd.tensors[1]).reshape(-1)
    np.testing.assert_allclose(got, expected, atol=1e-10)
