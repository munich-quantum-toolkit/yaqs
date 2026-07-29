# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Dense numerical references for the state-preparation benchmark noise models."""

from __future__ import annotations

import math
from itertools import product
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from benchmarks.state_preparation import (
    TWO_SITE_DEPOLARIZING_OPERATORS,
    ballarin_local_pauli_rate,
    create_ballarin_noise_provider,
    create_standard_noise_provider,
    sample_product_pauli_channel,
)
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import (
    GateNoiseContext,
    KrotovTJMOptions,
    KrotovTruncation,
    ParameterizedCircuit,
    ParameterizedGate,
    RandomUnitaryInstruction,
    forward_tjm_trajectory,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from benchmarks.state_preparation.noise import PauliDistribution
    from mqt.yaqs.core.data_structures.noise_model import NoiseModel
    from mqt.yaqs.optimization import LocalOperator


_PAULIS: Mapping[str, NDArray[np.complex128]] = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}
_LABELS = tuple(_PAULIS)


class _SequenceRNG:
    """Minimal generator double returning a fixed sequence of uniform draws."""

    def __init__(self, values: tuple[float, ...]) -> None:
        """Store the draw sequence."""
        self._values = values
        self.calls = 0

    def random(self) -> float:
        """Return the next configured draw.

        Returns:
            The next uniform draw.

        Raises:
            AssertionError: If every configured draw has already been consumed.
        """
        if self.calls >= len(self._values):
            msg = "Test RNG exhausted."
            raise AssertionError(msg)
        value = self._values[self.calls]
        self.calls += 1
        return value


class _TJMBranchRNG:
    """Generator double selecting no jump or one fixed TJM process."""

    def __init__(self, process_index: int | None) -> None:
        """Store the requested process, with ``None`` denoting no jump."""
        self.process_index = process_index

    def random(self) -> float:
        """Select the requested jump/no-jump branch.

        Returns:
            A draw that deterministically selects the configured branch.
        """
        return np.nextafter(1.0, 0.0) if self.process_index is None else 0.0

    def choice(self, size: int, *, p: NDArray[np.float64]) -> int:
        """Return the configured process index after validating the weights."""
        assert self.process_index is not None
        assert 0 <= self.process_index < size
        assert p.shape == (size,)
        assert np.sum(p) == pytest.approx(1.0)
        return self.process_index


def _as_generator(rng: object) -> np.random.Generator:
    """Cast a generator test double to the production protocol.

    Returns:
        The typed test generator.
    """
    return cast("np.random.Generator", rng)


def _density(vector: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Return the pure-state density matrix."""
    return np.outer(vector, vector.conj())


def _normalized_vector(num_qubits: int, seed: int) -> NDArray[np.complex128]:
    """Return a reproducible dense statevector."""
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=2**num_qubits) + 1.0j * rng.normal(size=2**num_qubits)
    return np.asarray(vector / np.linalg.norm(vector), dtype=np.complex128)


def _dense_product_operator(
    num_qubits: int,
    operators: Mapping[int, NDArray[np.complex128]],
) -> NDArray[np.complex128]:
    """Embed local operators in YAQS little-endian statevector order.

    Returns:
        The full-system operator.
    """
    result = np.ones((1, 1), dtype=np.complex128)
    for site in reversed(range(num_qubits)):
        result = np.kron(result, operators.get(site, _PAULIS["I"]))
    return result


def _branch_draw(distribution: Mapping[str, float], label: str) -> float:
    """Return an interior uniform draw selecting one positive branch."""
    lower = math.fsum(distribution[item] for item in _LABELS[: _LABELS.index(label)])
    probability = distribution[label]
    assert probability > 0.0
    return lower + probability / 2.0


def _apply_local_operators(
    vector: NDArray[np.complex128],
    operators: tuple[LocalOperator, ...],
) -> NDArray[np.complex128]:
    """Apply a realized random-unitary branch through the production MPS path.

    Returns:
        The evolved dense statevector.
    """
    state = MPS.from_statevector(vector)
    for operator in operators:
        state.apply_local_operator(operator.matrix, operator.sites)
    return np.asarray(state.to_vec(), dtype=np.complex128)


def _exact_product_density(
    vector: NDArray[np.complex128],
    num_qubits: int,
    sites: tuple[int, int],
    distribution: Mapping[str, float],
) -> NDArray[np.complex128]:
    """Return the exhaustive two-site product-Pauli channel."""
    initial_density = _density(vector)
    result = np.zeros_like(initial_density)
    for first_label, second_label in product(_LABELS, repeat=2):
        probability = distribution[first_label] * distribution[second_label]
        operator = _dense_product_operator(
            num_qubits,
            {
                sites[0]: _PAULIS[first_label],
                sites[1]: _PAULIS[second_label],
            },
        )
        result += probability * operator @ initial_density @ operator.conj().T
    return result


@pytest.mark.parametrize(
    ("num_qubits", "sites", "distribution", "seed"),
    [
        (2, (0, 1), {"I": 0.83, "X": 0.17, "Y": 0.0, "Z": 0.0}, 13),
        (3, (0, 2), {"I": 0.79, "X": 0.07, "Y": 0.07, "Z": 0.07}, 29),
    ],
)
def test_product_pauli_sampler_matches_dense_branch_sum(
    num_qubits: int,
    sites: tuple[int, int],
    distribution: Mapping[str, float],
    seed: int,
) -> None:
    """Bit-flip and depolarizing products equal exhaustive dense evolution."""
    vector = _normalized_vector(num_qubits, seed)
    actual = np.zeros((vector.size, vector.size), dtype=np.complex128)
    typed_distribution = cast("PauliDistribution", distribution)

    for first_label, second_label in product(_LABELS, repeat=2):
        probability = distribution[first_label] * distribution[second_label]
        if math.isclose(probability, 0.0, abs_tol=0.0):
            continue
        rng = _SequenceRNG((
            _branch_draw(distribution, first_label),
            _branch_draw(distribution, second_label),
        ))
        operators = sample_product_pauli_channel(
            *sites,
            typed_distribution,
            typed_distribution,
            _as_generator(rng),
        )
        actual += probability * _density(_apply_local_operators(vector, operators))
        assert rng.calls == 2

    expected = _exact_product_density(vector, num_qubits, sites, distribution)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize(("angle", "sites", "seed"), [(0.37, (0, 1), 31), (-1.21, (0, 2), 47)])
def test_ballarin_provider_matches_exhaustive_dense_branch_sum(
    angle: float,
    sites: tuple[int, int],
    seed: int,
) -> None:
    """The real Ballarin provider implements its exact independent product law."""
    num_qubits = max(sites) + 1
    vector = _normalized_vector(num_qubits, seed)
    rate = ballarin_local_pauli_rate(abs(angle))
    distribution = {"I": 1.0 - 3.0 * rate, "X": rate, "Y": rate, "Z": rate}
    provider = create_ballarin_noise_provider()
    context = GateNoiseContext(0, "rzz", sites, 2, angle, "logical", "native", None)
    actual = np.zeros((vector.size, vector.size), dtype=np.complex128)

    for first_label, second_label in product(_LABELS, repeat=2):
        probability = distribution[first_label] * distribution[second_label]
        rng = _SequenceRNG((
            _branch_draw(distribution, first_label),
            _branch_draw(distribution, second_label),
        ))
        instruction = provider(context, _as_generator(rng))
        assert isinstance(instruction, RandomUnitaryInstruction)
        assert instruction.outcome_labels == (first_label, second_label)
        actual += probability * _density(_apply_local_operators(vector, instruction.operators))

    expected = _exact_product_density(vector, num_qubits, sites, distribution)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def _process_operator(
    num_qubits: int,
    process: Mapping[str, Any],
) -> NDArray[np.complex128]:
    """Embed one standard-noise process as a dense operator.

    Returns:
        The full-system process matrix.
    """
    sites = tuple(cast("list[int]", process["sites"]))
    if len(sites) == 1:
        return _dense_product_operator(
            num_qubits,
            {sites[0]: np.asarray(process["matrix"], dtype=np.complex128)},
        )
    factors = cast("tuple[NDArray[np.complex128], NDArray[np.complex128]]", process["factors"])
    return _dense_product_operator(num_qubits, dict(zip(sites, factors, strict=True)))


def _one_gate_trajectory(
    noise_id: str,
    process_index: int | None,
    vector: NDArray[np.complex128],
) -> tuple[NDArray[np.complex128], NoiseModel]:
    """Run one standard-noise TJM branch on an identity-valued gate.

    Returns:
        The final dense statevector and gate-local noise model.
    """
    two_site = noise_id != "dephasing_1s_1q"
    gate_name, sites = ("rzz", (0, 2)) if two_site else ("rz", (1,))
    gate = ParameterizedGate(
        gate_name,
        sites,
        angle_offset=0.0,
        logical_gate_id="logical",
        native_gate_id="native",
    )
    circuit = ParameterizedCircuit(3, [gate])
    provider = create_standard_noise_provider(noise_id)
    context = GateNoiseContext(0, gate.name, gate.sites, len(gate.sites), 0.0, "logical", "native", None)
    instruction = provider(context, _as_generator(_SequenceRNG(())))
    assert instruction is not None
    trajectory = forward_tjm_trajectory(
        circuit,
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        MPS.from_statevector(vector),
        KrotovTruncation(),
        None,
        KrotovTJMOptions(dt=1.0),
        _as_generator(_TJMBranchRNG(process_index)),
        noise_provider=provider,
    )
    return np.asarray(trajectory.states[-1].to_vec(), dtype=np.complex128), instruction.noise_model


@pytest.mark.parametrize(
    ("noise_id", "expected_processes"),
    [
        ("dephasing_1s_1q", 1),
        ("dephasing_2s_2q", 1),
        ("depolarizing_2s_2q", 9),
    ],
)
def test_standard_tjm_channels_match_dense_density_reference(
    noise_id: str,
    expected_processes: int,
) -> None:
    """Single Z, correlated ZZ, and all nine two-site Paulis match dense maps."""
    vector = _normalized_vector(3, 71)
    no_jump_vector, noise_model = _one_gate_trajectory(noise_id, None, vector)
    assert len(noise_model.processes) == expected_processes
    if noise_id == "depolarizing_2s_2q":
        assert tuple(str(process["name"]).removeprefix("crosstalk_").upper() for process in noise_model.processes) == (
            TWO_SITE_DEPOLARIZING_OPERATORS
        )

    strengths = np.array([float(process["strength"]) for process in noise_model.processes])
    total_rate = float(np.sum(strengths))
    no_jump_probability = math.exp(-total_rate)
    actual = no_jump_probability * _density(no_jump_vector)
    expected = no_jump_probability * _density(vector)

    for process_index, (process, strength) in enumerate(zip(noise_model.processes, strengths, strict=True)):
        jump_vector, _ = _one_gate_trajectory(noise_id, process_index, vector)
        probability = (1.0 - no_jump_probability) * float(strength) / total_rate
        actual += probability * _density(jump_vector)
        operator = _process_operator(3, process)
        expected += probability * operator @ _density(vector) @ operator.conj().T

    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert np.trace(actual) == pytest.approx(1.0)


def test_ballarin_trajectory_average_converges_to_exact_channel() -> None:
    """A fixed-seed trajectory ensemble approaches the enumerated Ballarin map."""
    angle = math.pi
    sites = (0, 1)
    vector = _normalized_vector(2, 101)
    rate = ballarin_local_pauli_rate(angle)
    distribution = {"I": 1.0 - 3.0 * rate, "X": rate, "Y": rate, "Z": rate}
    expected = _exact_product_density(vector, 2, sites, distribution)
    provider = create_ballarin_noise_provider()
    context = GateNoiseContext(0, "rzz", sites, 2, angle, "logical", "native", None)
    rng = np.random.default_rng(20260729)
    checkpoints = {2_000, 50_000}
    errors: dict[int, float] = {}
    average = np.zeros_like(expected)

    for trajectory_index in range(1, max(checkpoints) + 1):
        instruction = provider(context, rng)
        assert isinstance(instruction, RandomUnitaryInstruction)
        branch = _density(_apply_local_operators(vector, instruction.operators))
        average += (branch - average) / trajectory_index
        if trajectory_index in checkpoints:
            errors[trajectory_index] = float(np.linalg.norm(average - expected))

    assert errors[50_000] < errors[2_000]
    assert errors[50_000] < 1.5e-3
