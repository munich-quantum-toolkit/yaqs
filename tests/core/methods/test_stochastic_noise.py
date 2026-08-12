# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for sampling and applying stochastic gate-local noise."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from mqt.yaqs import XBasisDissipativeNoiseModel, XYZPauliNoiseModel
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.methods.stochastic_noise import apply_stochastic_noise, sample_xyz_pauli_event


class _ScriptedRng:
    """Minimal RNG with deterministic uniform and integer results."""

    def __init__(self, *, randoms: list[float], integers: list[int] | None = None) -> None:
        self.randoms = iter(randoms)
        self.integer_values = iter(integers or [])

    def random(self) -> float:
        """Return the next scripted uniform value."""
        return next(self.randoms)

    def integers(self, high: int) -> int:
        """Return the next scripted integer below ``high``."""
        value = next(self.integer_values)
        assert 0 <= value < high
        return value


def _rng(*, randoms: list[float], integers: list[int] | None = None) -> np.random.Generator:
    return cast("np.random.Generator", _ScriptedRng(randoms=randoms, integers=integers))


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        (0, "X"),
        (1, "Y"),
        (2, "Z"),
    ],
)
def test_xyz_one_qubit_rng_mapping(choice: int, expected: str) -> None:
    """The categorical RNG index maps exactly to X, Y, then Z."""
    event = sample_xyz_pauli_event(XYZPauliNoiseModel(1.0), _rng(randoms=[0.0], integers=[choice]))
    assert event == expected


def test_xyz_identity_boundary_uses_greater_equal_rule() -> None:
    """A draw on the jump-probability boundary selects identity."""
    p = 0.2
    event = sample_xyz_pauli_event(XYZPauliNoiseModel(p), _rng(randoms=[p]))
    assert event is None


def test_xyz_two_qubit_support_is_sampled_independently() -> None:
    """Each qubit touched by a two-qubit gate receives its own local channel draw."""
    state = MPS(2, state="zeros")
    apply_stochastic_noise(state, XYZPauliNoiseModel(1.0), [0, 1], _rng(randoms=[0.0, 0.0], integers=[0, 0]))
    expected = np.zeros(4, dtype=np.complex128)
    expected[3] = 1.0
    np.testing.assert_allclose(state.to_vec(), expected, atol=1e-14)


def test_xyz_two_qubit_bernoulli_samples_are_independent() -> None:
    """Intermediate p allows identity on one touched qubit and an error on the other."""
    state = MPS(2, state="zeros")
    rng = _rng(randoms=[0.75, 0.25], integers=[0])

    apply_stochastic_noise(state, XYZPauliNoiseModel(0.5), [0, 1], rng)

    expected = np.zeros(4, dtype=np.complex128)
    expected[2] = 1.0
    np.testing.assert_allclose(state.to_vec(), expected, atol=1e-14)


@pytest.mark.parametrize("model", [XYZPauliNoiseModel(0.0), XBasisDissipativeNoiseModel(0.0)])
def test_stochastic_p_zero_is_exact_identity_without_rng_draw(
    model: XYZPauliNoiseModel | XBasisDissipativeNoiseModel,
) -> None:
    """Both p=0 models leave every tensor byte-for-byte unchanged."""
    state = MPS(2, state="x+")
    before = [tensor.copy() for tensor in state.tensors]
    apply_stochastic_noise(state, model, [0, 1], _rng(randoms=[]))
    for actual, expected in zip(state.tensors, before, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_seeded_xyz_event_sampling_is_reproducible() -> None:
    """Seeded sampling follows the Bernoulli-then-integer RNG sequence."""
    model = XYZPauliNoiseModel(0.2)
    generators = [np.random.default_rng(31), np.random.default_rng(31)]
    streams = [[sample_xyz_pauli_event(model, rng) for _ in range(25)] for rng in generators]
    assert streams[0] == streams[1]

    reference_rng = np.random.default_rng(31)
    axes = ("X", "Y", "Z")
    expected = [axes[int(reference_rng.integers(3))] if reference_rng.random() < model.p else None for _ in range(25)]
    assert streams[0] == expected


@pytest.mark.parametrize(("state_name", "uniform"), [("x+", 0.9), ("x-", 0.1)])
def test_dissipative_p_one_maps_x_basis_states_to_plus(state_name: str, uniform: float) -> None:
    """Both endpoint branches of the p=1 channel normalize to the plus state."""
    state = MPS(1, state=state_name)
    apply_stochastic_noise(state, XBasisDissipativeNoiseModel(1.0), [0], _rng(randoms=[uniform]))
    expected = MPS(1, state="x+").to_vec()
    np.testing.assert_allclose(state.to_vec(), expected, atol=1e-14)
    assert float(state.norm()) == pytest.approx(1.0, abs=1e-14)


def test_dissipative_branch_probabilities_are_state_dependent() -> None:
    """For |->, p=1/4 selects K0 below 3/4 and K1 above 3/4."""
    no_jump = MPS(1, state="x-")
    apply_stochastic_noise(no_jump, XBasisDissipativeNoiseModel(0.25), [0], _rng(randoms=[0.5]))
    np.testing.assert_allclose(no_jump.to_vec(), MPS(1, state="x-").to_vec(), atol=1e-14)

    jump = MPS(1, state="x-")
    apply_stochastic_noise(jump, XBasisDissipativeNoiseModel(0.25), [0], _rng(randoms=[0.8]))
    np.testing.assert_allclose(jump.to_vec(), MPS(1, state="x+").to_vec(), atol=1e-14)


def test_dissipative_trajectory_stays_normalized_on_entangled_state() -> None:
    """A sampled local Kraus branch preserves trajectory normalization."""
    left = np.zeros((2, 1, 2), dtype=np.complex128)
    left[0, 0, 0] = 1.0 / np.sqrt(2.0)
    left[1, 0, 1] = 1.0 / np.sqrt(2.0)
    right = np.zeros((2, 2, 1), dtype=np.complex128)
    right[0, 0, 0] = 1.0
    right[1, 1, 0] = 1.0
    state = MPS(2, tensors=[left, right])
    apply_stochastic_noise(state, XBasisDissipativeNoiseModel(0.4), [0, 1], np.random.default_rng(17))
    assert float(state.norm()) == pytest.approx(1.0, abs=1e-12)


def test_seeded_stochastic_noise_application_is_reproducible() -> None:
    """Equal generator seeds produce identical multi-gate trajectories."""
    model = XBasisDissipativeNoiseModel(0.37)
    states = [MPS(2, state="x-") for _ in range(2)]
    for state, rng in zip(states, [np.random.default_rng(19), np.random.default_rng(19)], strict=True):
        for sites in ([0], [0, 1], [1]):
            apply_stochastic_noise(state, model, sites, rng)
    np.testing.assert_array_equal(states[0].to_vec(), states[1].to_vec())


def test_stochastic_noise_rejects_gates_outside_paper_arity() -> None:
    """The paper models do not silently invent semantics for three-qubit gates."""
    with pytest.raises(ValueError, match="one- and two-qubit"):
        apply_stochastic_noise(MPS(3), XYZPauliNoiseModel(0.1), [0, 1, 2], np.random.default_rng(0))
