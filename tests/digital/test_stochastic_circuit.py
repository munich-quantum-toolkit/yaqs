# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for stochastic circuit construction."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit

from mqt.yaqs import NoiseModel
from mqt.yaqs.core.random_utils import make_trajectory_rng
from mqt.yaqs.digital import sample_stochastic_circuit

if TYPE_CHECKING:
    from collections.abc import Sequence


class _SequenceRNG:
    """Return a fixed sequence of random draws."""

    def __init__(self, draws: Sequence[float]) -> None:
        self.draws = iter(draws)

    def random(self) -> float:
        """Return the next configured draw."""
        return next(self.draws)


def _rng(*draws: float) -> np.random.Generator:
    return cast("np.random.Generator", _SequenceRNG(draws))


def _operation_names(circuit: QuantumCircuit) -> list[str]:
    return [instruction.operation.name for instruction in circuit.data]


def _operation_sites(circuit: QuantumCircuit) -> list[tuple[str, tuple[int, ...]]]:
    return [
        (
            instruction.operation.name,
            tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits),
        )
        for instruction in circuit.data
    ]


def test_non_gate_instructions_are_not_noise_locations() -> None:
    """Barriers, resets, delays, and measurements are copied without noise."""
    circuit = QuantumCircuit(3, 1)
    circuit.barrier(0, 1)
    circuit.reset(0)
    circuit.delay(5, 1)
    circuit.measure(2, 0)
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 1e6}])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng())

    assert _operation_names(sampled) == _operation_names(circuit)


def test_rx_is_one_qubit_noise_location() -> None:
    """A Pauli process supported on an RX target is sampled after RX."""
    circuit = QuantumCircuit(1)
    circuit.rx(0.1, 0)
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 1.0}])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng(0.0, 0.0))

    assert _operation_sites(sampled) == [("rx", (0,)), ("x", (0,))]


def test_h_is_one_qubit_noise_location() -> None:
    """The placement rule applies to one-qubit gates other than rotations."""
    circuit = QuantumCircuit(1)
    circuit.h(0)
    noise_model = NoiseModel([{"name": "pauli_z", "sites": [0], "strength": 1.0}])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng(0.0, 0.0))

    assert _operation_sites(sampled) == [("h", (0,)), ("z", (0,))]


def test_two_site_process_is_excluded_from_one_qubit_gate_support() -> None:
    """A one-qubit gate sees only processes fully supported on its target."""
    circuit = QuantumCircuit(2)
    circuit.rx(0.1, 0)
    noise_model = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 1.0},
        {"name": "crosstalk_xx", "sites": [0, 1], "strength": 1e6},
    ])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng(0.0, 0.9))

    assert _operation_sites(sampled) == [("rx", (0,)), ("x", (0,))]


@pytest.mark.parametrize(
    ("process", "expected"),
    [
        ({"name": "pauli_x", "sites": [0], "strength": 1.0}, ("x", (0,))),
        ({"name": "pauli_z", "sites": [1], "strength": 1.0}, ("z", (1,))),
    ],
)
def test_one_site_process_can_target_either_gate_qubit(
    process: dict[str, object], expected: tuple[str, tuple[int, ...]]
) -> None:
    """A relevant one-site process retains its declared gate-qubit support."""
    circuit = QuantumCircuit(2)
    circuit.rzz(0.2, 0, 1)

    sampled = sample_stochastic_circuit(circuit, NoiseModel([process]), _rng(0.0, 0.0))

    assert _operation_sites(sampled) == [("rzz", (0, 1)), expected]


@pytest.mark.parametrize(
    ("name", "pauli_name"),
    [("crosstalk_xx", "x"), ("crosstalk_yy", "y"), ("crosstalk_zz", "z")],
)
def test_two_site_pauli_process_acts_on_both_gate_qubits(name: str, pauli_name: str) -> None:
    """A selected two-site process is one event represented by one Pauli gate per support site."""
    circuit = QuantumCircuit(2)
    circuit.rzz(0.2, 0, 1)
    noise_model = NoiseModel([{"name": name, "sites": [0, 1], "strength": 1.0}])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng(0.0, 0.0))

    assert _operation_sites(sampled) == [("rzz", (0, 1)), (pauli_name, (0,)), (pauli_name, (1,))]


def test_spectator_process_is_excluded_from_gate_support() -> None:
    """A process on an unrelated qubit cannot enter the gate-level categorical draw."""
    circuit = QuantumCircuit(3)
    circuit.rzz(0.2, 0, 1)
    noise_model = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 1.0},
        {"name": "pauli_z", "sites": [2], "strength": 1e6},
    ])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng(0.0, 0.9))

    assert _operation_sites(sampled) == [("rzz", (0, 1)), ("x", (0,))]


def test_at_most_one_categorical_process_is_selected_per_gate() -> None:
    """Shared gate support produces one categorical event rather than independent site events."""
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    noise_model = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 10.0},
        {"name": "pauli_y", "sites": [0], "strength": 10.0},
        {"name": "pauli_z", "sites": [1], "strength": 10.0},
    ])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng(0.0, 0.5))

    assert _operation_names(sampled) == ["cx", "y"]


def test_zero_total_gamma_inserts_no_noise_without_drawing() -> None:
    """A zero total rate skips event and categorical RNG draws."""
    circuit = QuantumCircuit(2)
    circuit.rzz(0.2, 0, 1)
    noise_model = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 0.0},
        {"name": "pauli_z", "sites": [1], "strength": 0.0},
    ])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng())

    assert _operation_names(sampled) == ["rzz"]


def test_event_probability_is_one_minus_exp_of_total_gamma() -> None:
    """The event boundary uses ``1 - exp(-sum(gamma_i))`` without treating gamma as probability."""
    circuit = QuantumCircuit(2)
    circuit.rzz(0.2, 0, 1)
    noise_model = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 0.2},
        {"name": "pauli_z", "sites": [1], "strength": 0.3},
    ])
    event_probability = -math.expm1(-0.5)

    event = sample_stochastic_circuit(circuit, noise_model, _rng(float(np.nextafter(event_probability, 0.0)), 0.0))
    no_event = sample_stochastic_circuit(circuit, noise_model, _rng(event_probability))

    assert _operation_names(event) == ["rzz", "x"]
    assert _operation_names(no_event) == ["rzz"]


@pytest.mark.parametrize(
    ("category_draw", "expected"),
    [(0.05, "x"), (0.2, "y"), (0.8, "z")],
)
def test_categorical_process_weighting_uses_gamma_ratios(category_draw: float, expected: str) -> None:
    """Categorical intervals are proportional to rates one, three, and six."""
    circuit = QuantumCircuit(2)
    circuit.rzz(0.2, 0, 1)
    noise_model = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 1.0},
        {"name": "pauli_y", "sites": [0], "strength": 3.0},
        {"name": "pauli_z", "sites": [0], "strength": 6.0},
    ])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng(0.0, category_draw))

    assert _operation_names(sampled) == ["rzz", expected]


def test_seeded_trajectory_realizations_are_reproducible_and_independent() -> None:
    """Equal trajectory streams reproduce a circuit while distinct trajectory indices can differ."""
    circuit = QuantumCircuit(2)
    for _ in range(20):
        circuit.rzz(0.2, 0, 1)
    noise_model = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 0.1},
        {"name": "pauli_z", "sites": [1], "strength": 0.2},
    ])

    first = sample_stochastic_circuit(circuit, noise_model, make_trajectory_rng(0, base_seed=17))
    repeated = sample_stochastic_circuit(circuit, noise_model, make_trajectory_rng(0, base_seed=17))
    second_trajectory = sample_stochastic_circuit(circuit, noise_model, make_trajectory_rng(1, base_seed=17))

    assert first == repeated
    assert first != second_trajectory


def test_multiple_one_and_two_qubit_gates_are_independent_opportunities() -> None:
    """One RNG stream supplies separate event and category draws for every eligible gate."""
    circuit = QuantumCircuit(2)
    circuit.rx(0.1, 0)
    circuit.rzz(0.2, 0, 1)
    circuit.rx(0.3, 1)
    noise_model = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 100.0},
        {"name": "pauli_z", "sites": [1], "strength": 100.0},
    ])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng(0.0, 0.0, 0.0, 0.75, 0.0, 0.0))

    assert _operation_sites(sampled) == [
        ("rx", (0,)),
        ("x", (0,)),
        ("rzz", (0, 1)),
        ("z", (1,)),
        ("rx", (1,)),
        ("z", (1,)),
    ]


def test_gate_wider_than_two_qubits_is_not_a_noise_location() -> None:
    """A three-qubit gate is copied without consuming stochastic draws."""
    circuit = QuantumCircuit(3)
    circuit.ccx(0, 1, 2)
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 1e6}])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng())

    assert _operation_sites(sampled) == [("ccx", (0, 1, 2))]


def test_input_is_not_mutated_and_native_rzz_is_preserved() -> None:
    """Construction copies circuit metadata and instructions without decomposing or mutating RZZ."""
    circuit = QuantumCircuit(2, name="ideal", metadata={"source": "test"})
    circuit.global_phase = 0.25
    circuit.rzz(0.2, 0, 1)
    original = circuit.copy()
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 1.0}])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng(0.0, 0.0))

    assert circuit == original
    assert sampled is not circuit
    assert sampled.name == "ideal"
    assert sampled.metadata == {"source": "test"}
    assert sampled.data[0].operation.name == "rzz"
    assert sampled.data[0].operation.params == [0.2]


def test_pauli_structure_is_resolved_from_custom_operator_matrix() -> None:
    """A custom process name is accepted when its resolved matrix has Pauli structure."""
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    matrix = -1j * NoiseModel.get_operator("pauli_y")
    noise_model = NoiseModel([{"name": "custom_jump", "sites": [1], "strength": 1.0, "matrix": matrix}])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng(0.0, 0.0))

    assert _operation_sites(sampled) == [("cx", (0, 1)), ("y", (1,))]
    assert np.isclose(np.exp(1j * float(sampled.global_phase)), -1j)


def test_distribution_strengths_are_sampled_once_per_circuit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Static strength disorder is resolved once before all gate opportunities."""
    circuit = QuantumCircuit(2)
    circuit.rzz(0.2, 0, 1)
    circuit.rzz(0.3, 0, 1)
    noise_model = NoiseModel([
        {
            "name": "pauli_x",
            "sites": [0],
            "strength": {"distribution": "normal", "mean": 1.0, "std": 0.0},
        }
    ])
    original_sample = NoiseModel.sample
    sample_calls = 0

    def recording_sample(self: NoiseModel, rng: np.random.Generator | int | None = None) -> NoiseModel:
        nonlocal sample_calls
        sample_calls += 1
        return original_sample(self, rng=rng)

    monkeypatch.setattr(NoiseModel, "sample", recording_sample)

    sampled = sample_stochastic_circuit(circuit, noise_model, np.random.default_rng(2))

    assert sample_calls == 1
    assert _operation_names(sampled).count("rzz") == 2


def test_positive_rate_unsupported_process_is_rejected_before_sampling() -> None:
    """A relevant lowering jump fails deterministically before any random draw."""
    circuit = QuantumCircuit(2)
    circuit.rzz(0.2, 0, 1)
    noise_model = NoiseModel([{"name": "lowering", "sites": [0], "strength": 1.0}])

    with pytest.raises(ValueError, match="supports only Pauli jump processes"):
        sample_stochastic_circuit(circuit, noise_model, _rng())


def test_zero_rate_unsupported_process_is_ignored() -> None:
    """A zero-rate unsupported process cannot affect the circuit or consume RNG draws."""
    circuit = QuantumCircuit(1)
    circuit.rx(0.2, 0)
    noise_model = NoiseModel([{"name": "lowering", "sites": [0], "strength": 0.0}])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng())

    assert _operation_names(sampled) == ["rx"]
