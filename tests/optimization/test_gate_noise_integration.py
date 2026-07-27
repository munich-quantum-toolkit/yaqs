# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Integration tests for gate-local noise providers in Krotov trajectories."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import pytest

import mqt.yaqs.optimization.krotov as krotov_module
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import Observable
from mqt.yaqs.optimization import (
    CompositeGateNoiseInstruction,
    GateNoiseContext,
    GateNoiseProvider,
    KrotovNoiseMap,
    KrotovOptions,
    KrotovReadout,
    KrotovResult,
    KrotovTJMOptions,
    KrotovTruncation,
    LocalOperator,
    ParameterizedCircuit,
    ParameterizedGate,
    RandomUnitaryInstruction,
    TJMNoiseInstruction,
    noisy_sample_contribution,
    noisy_sample_loss,
    noisy_state_preparation_contribution,
    noisy_state_preparation_cross_contribution,
    noisy_state_preparation_loss,
    noisy_state_preparation_metrics,
    train_krotov_noisy_state_preparation_batch,
    train_krotov_noisy_state_preparation_hybrid,
    train_krotov_noisy_state_preparation_online,
)
from mqt.yaqs.optimization.krotov import forward_tjm_trajectory

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


class _NoisyStatePreparationTrainer(Protocol):
    """Common public signature of the three noisy state-preparation trainers."""

    def __call__(
        self,
        circuit: ParameterizedCircuit,
        target_state: MPS | NDArray[np.complex128],
        noise_model: NoiseModel | None,
        tjm_options: KrotovTJMOptions | None = None,
        *,
        initial_theta: NDArray[np.float64],
        options: KrotovOptions | None = None,
        initial_state: MPS | None = None,
        noise_provider: GateNoiseProvider | None = None,
    ) -> KrotovResult:
        """Train one noisy state-preparation objective."""
        ...


_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
_H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)
_S = np.diag([1.0, 1.0j]).astype(np.complex128)
_CX = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=np.complex128,
)


def _forward(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    *,
    noise_provider: GateNoiseProvider | None = None,
    noise_model: NoiseModel | None = None,
    tjm_options: KrotovTJMOptions | None = None,
    noise_maps: list[KrotovNoiseMap] | None = None,
    x: NDArray[np.float64] | None = None,
    seed: int = 7,
    truncation: KrotovTruncation | None = None,
) -> krotov_module.KrotovTrajectory:
    """Run one small trajectory with exact local-operator application.

    Returns:
        The realized Krotov trajectory.
    """
    return forward_tjm_trajectory(
        circuit,
        theta,
        np.array([], dtype=np.float64) if x is None else x,
        MPS(circuit.num_qubits),
        KrotovTruncation() if truncation is None else truncation,
        noise_model,
        KrotovTJMOptions() if tjm_options is None else tjm_options,
        np.random.default_rng(seed),
        noise_maps=noise_maps,
        noise_provider=noise_provider,
    )


def _random_instruction(
    operators: tuple[LocalOperator, ...],
    *,
    channel_id: str = "test_channel",
    outcome_labels: tuple[str, ...] = (),
) -> RandomUnitaryInstruction:
    """Build a random-unitary instruction used by several tests.

    Returns:
        The configured random-unitary instruction.
    """
    return RandomUnitaryInstruction(
        operators=operators,
        channel_id=channel_id,
        outcome_labels=outcome_labels,
    )


def test_provider_receives_complete_context_with_resolved_angles() -> None:
    """Providers receive physical support, provenance, and fully resolved angles."""
    contexts: list[GateNoiseContext] = []

    def provider(context: GateNoiseContext, rng: np.random.Generator) -> None:
        del rng
        contexts.append(context)

    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("h", (0,)),
            ParameterizedGate(
                "ry",
                (1,),
                param_index=0,
                angle_scale=2.0,
                angle_offset=0.1,
                data_map=lambda sample: float(sample[0]),
                logical_gate_id=17,
                native_gate_id=21,
            ),
            ParameterizedGate("rzz", (1, 0), param_index=1, angle_scale=-0.5, angle_offset=0.2),
        ],
        num_params=2,
    )

    _forward(
        circuit,
        np.array([0.3, -0.6]),
        x=np.array([0.4]),
        noise_provider=provider,
    )

    assert all(isinstance(context, GateNoiseContext) for context in contexts)
    assert [context.gate_index for context in contexts] == [0, 1, 2]
    assert [context.gate_name for context in contexts] == ["h", "ry", "rzz"]
    assert [context.sites for context in contexts] == [(0,), (1,), (0, 1)]
    assert [context.arity for context in contexts] == [1, 1, 2]
    assert [context.parameter_index for context in contexts] == [None, 0, 1]
    assert [context.logical_gate_id for context in contexts] == [0, 17, 2]
    assert [context.native_gate_id for context in contexts] == [0, 21, 2]
    assert contexts[0].resolved_angle is None
    assert contexts[1].resolved_angle == pytest.approx(1.1)
    assert contexts[2].resolved_angle == pytest.approx(0.5)


def test_provider_is_not_invoked_for_excluded_gates() -> None:
    """Application filters run before provider invocation."""
    requested_indices: list[int] = []

    def provider(context: GateNoiseContext, rng: np.random.Generator) -> None:
        del rng
        requested_indices.append(context.gate_index)

    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("rx", (0,), param_index=0),
            ParameterizedGate("rzz", (0, 1), param_index=1),
            ParameterizedGate("ry", (1,), param_index=2),
        ],
    )
    _forward(
        circuit,
        np.array([0.1, 0.2, 0.3]),
        noise_provider=provider,
        tjm_options=KrotovTJMOptions(apply_noise_to="two-qubit", noisy_gate_indices=(0, 1)),
    )

    assert requested_indices == [1]


def test_noise_disabled_gate_does_not_request_provider() -> None:
    """Compiler-marked noiseless gates are excluded before provider invocation."""
    requested_indices: list[int] = []

    def provider(context: GateNoiseContext, _rng: np.random.Generator) -> None:
        requested_indices.append(context.gate_index)

    circuit = ParameterizedCircuit(
        1,
        [
            ParameterizedGate("h", (0,), noise_enabled=False),
            ParameterizedGate("x", (0,)),
        ],
    )

    _forward(circuit, np.array([]), noise_provider=provider)

    assert requested_indices == [1]


def test_noise_enabled_metadata_does_not_change_the_legacy_global_noise_path() -> None:
    """Provider-only gate metadata leaves existing global-model calls unchanged."""

    class LegacyGateMatrixCircuit(ParameterizedCircuit):
        """Circuit double exposing only the pre-provider matrix hook."""

        legacy_matrix_calls = 0

        def gate_matrix(
            self,
            gate: ParameterizedGate,
            theta: NDArray[np.float64],
            x: NDArray[np.float64] | None = None,
        ) -> tuple[NDArray[np.complex128], tuple[int, ...]]:
            """Return the fixed test gate through the legacy hook."""
            del gate, theta, x
            self.legacy_matrix_calls += 1
            return _I, (0,)

        def gate_matrix_and_angle(
            self,
            gate: ParameterizedGate,
            theta: NDArray[np.float64],
            x: NDArray[np.float64] | None = None,
        ) -> tuple[NDArray[np.complex128], tuple[int, ...], float | None]:
            """Fail if a legacy global-noise call uses the provider-only hook."""
            del gate, theta, x
            pytest.fail(f"{type(self).__name__} unexpectedly called gate_matrix_and_angle")

    circuit = LegacyGateMatrixCircuit(
        1,
        [ParameterizedGate("ry", (0,), angle_offset=0.0, noise_enabled=False)],
    )
    model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 2.0}])

    trajectory = _forward(circuit, np.array([]), noise_model=model, seed=1)

    np.testing.assert_allclose(trajectory.states[-1].to_vec(), np.array([0.0, 1.0]), atol=1e-12)
    assert circuit.legacy_matrix_calls == 1
    realized = trajectory.noise_maps[0]
    assert realized.jump_process_index == 0
    assert realized.channel_id is None
    assert realized.outcome_labels == ()
    assert realized.source_gate_index is None
    assert realized.resolved_native_angle is None
    assert realized.is_identity is None


@pytest.mark.parametrize("representation", ["raw", "wrapped"])
def test_provider_accepts_raw_and_wrapped_local_noise_models(representation: str) -> None:
    """Both supported TJM provider representations use existing sampling machinery."""
    model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 2.0}])

    def provider(
        context: GateNoiseContext,
        rng: np.random.Generator,
    ) -> NoiseModel | TJMNoiseInstruction:
        del context, rng
        if representation == "wrapped":
            return TJMNoiseInstruction(noise_model=model, channel_id="wrapped_tjm")
        return model

    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), angle_offset=0.0)])
    trajectory = _forward(circuit, np.array([]), noise_provider=provider, seed=1)

    np.testing.assert_allclose(trajectory.states[-1].to_vec(), np.array([0.0, 1.0]), atol=1e-12)
    realized = trajectory.noise_maps[0]
    assert realized.source_gate_index == 0
    assert realized.resolved_native_angle == pytest.approx(0.0)
    assert realized.is_identity is False
    assert realized.channel_id == ("wrapped_tjm" if representation == "wrapped" else None)
    assert realized.outcome_labels


@pytest.mark.parametrize(
    ("operators", "labels", "expected", "identity_kind"),
    [
        ((), ("I",), np.array([1.0, 0.0]), "identity"),
        (
            (LocalOperator(matrix=_X, sites=(0,)),),
            ("X",),
            np.array([0.0, 1.0]),
            "nonidentity",
        ),
        (
            (
                LocalOperator(matrix=_X, sites=(0,)),
                LocalOperator(matrix=_H, sites=(0,)),
            ),
            ("X", "H"),
            np.array([1.0, -1.0]) / np.sqrt(2.0),
            "nonidentity",
        ),
    ],
)
def test_random_unitary_provider_supports_zero_one_or_two_operators(
    operators: tuple[LocalOperator, ...],
    labels: tuple[str, ...],
    expected: NDArray[np.float64],
    identity_kind: str,
) -> None:
    """Random-unitary instructions preserve operator count and forward order."""

    def provider(
        context: GateNoiseContext,
        rng: np.random.Generator,
    ) -> RandomUnitaryInstruction:
        del context, rng
        return _random_instruction(operators, outcome_labels=labels)

    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), angle_offset=0.0)])
    trajectory = _forward(circuit, np.array([]), noise_provider=provider)
    realized = trajectory.noise_maps[0]

    np.testing.assert_allclose(trajectory.states[-1].to_vec(), expected, atol=1e-12)
    assert len(realized.operators) == len(operators)
    assert realized.channel_id == "test_channel"
    assert realized.outcome_labels == labels
    assert realized.source_gate_index == 0
    assert realized.resolved_native_angle == pytest.approx(0.0)
    assert realized.is_identity is (identity_kind == "identity")


def test_random_unitary_noise_is_applied_after_the_ideal_gate() -> None:
    """A noncommuting example fixes post-gate rather than pre-gate placement."""

    def provider(
        context: GateNoiseContext,
        rng: np.random.Generator,
    ) -> RandomUnitaryInstruction:
        del context, rng
        return _random_instruction(
            (LocalOperator(matrix=_Z, sites=(0,)),),
            outcome_labels=("Z",),
        )

    circuit = ParameterizedCircuit(1, [ParameterizedGate("h", (0,))])
    trajectory = _forward(circuit, np.array([]), noise_provider=provider)

    np.testing.assert_allclose(
        trajectory.states[-1].to_vec(),
        np.array([1.0, -1.0]) / np.sqrt(2.0),
        atol=1e-12,
    )


def test_random_unitary_map_uses_local_operator_labels_as_diagnostic_fallback() -> None:
    """Operator labels populate replay diagnostics unless branch labels are explicit."""

    def provider(
        context: GateNoiseContext,
        rng: np.random.Generator,
    ) -> RandomUnitaryInstruction:
        del context, rng
        return _random_instruction((LocalOperator(matrix=_X, sites=(0,), label="X"),))

    circuit = ParameterizedCircuit(1, [ParameterizedGate("h", (0,))])
    trajectory = _forward(circuit, np.array([]), noise_provider=provider)

    assert trajectory.noise_maps[0].outcome_labels == ("X",)


@pytest.mark.parametrize(
    ("operators", "identity_kind"),
    [
        ((LocalOperator(_I, (0,), "I"),), "identity"),
        ((LocalOperator(_X, (0,), "X"), LocalOperator(_X, (0,), "X")), "identity"),
        ((LocalOperator(_X, (0,), "X"),), "nonidentity"),
    ],
)
def test_random_unitary_identity_diagnostic_uses_the_realized_product(
    operators: tuple[LocalOperator, ...],
    identity_kind: str,
) -> None:
    """Identity diagnostics account for explicit and cancelling operators."""

    def provider(
        context: GateNoiseContext,
        rng: np.random.Generator,
    ) -> RandomUnitaryInstruction:
        del context, rng
        return _random_instruction(operators)

    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), angle_offset=0.0)])
    trajectory = _forward(circuit, np.array([]), noise_provider=provider)

    assert trajectory.noise_maps[0].is_identity is (identity_kind == "identity")


def test_explicit_composite_provider_sequences_tjm_and_random_unitary_noise() -> None:
    """Mixed mechanisms compose only through the explicit provider instruction."""
    model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 2.0}])

    def provider(
        context: GateNoiseContext,
        rng: np.random.Generator,
    ) -> CompositeGateNoiseInstruction:
        del context, rng
        return CompositeGateNoiseInstruction(
            (
                TJMNoiseInstruction(model, channel_id="tjm"),
                RandomUnitaryInstruction(
                    (LocalOperator(_Z, (0,), "Z"),),
                    channel_id="unitary",
                ),
            ),
            channel_id="explicit-composite",
        )

    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), angle_offset=0.0)])
    sampled = _forward(circuit, np.array([]), noise_provider=provider, seed=1)
    replayed = _forward(circuit, np.array([]), noise_maps=sampled.noise_maps, seed=999)

    np.testing.assert_allclose(sampled.states[-1].to_vec(), np.array([0.0, -1.0]), atol=1e-12)
    np.testing.assert_allclose(replayed.states[-1].to_vec(), sampled.states[-1].to_vec(), atol=1e-12)
    realized = sampled.noise_maps[0]
    assert len(realized.operators) == 2
    assert realized.channel_id == "explicit-composite"
    assert realized.outcome_labels == ("pauli_x", "Z")
    assert realized.is_identity is None
    assert realized.normalization_checkpoints == (1,)


def test_composite_replay_preserves_intermediate_tjm_normalization() -> None:
    """Replay normalizes before a later two-site operator and its truncation."""
    projector = np.diag([1.0, 0.0]).astype(np.complex128)
    model = NoiseModel(
        [{"name": "projector", "sites": [0], "strength": 0.1, "matrix": projector}],
    )

    def provider(
        context: GateNoiseContext,
        rng: np.random.Generator,
    ) -> CompositeGateNoiseInstruction:
        del context, rng
        return CompositeGateNoiseInstruction(
            (
                TJMNoiseInstruction(model),
                RandomUnitaryInstruction((
                    LocalOperator(_H, (0,), "H"),
                    LocalOperator(_CX, (0, 1), "CX"),
                )),
            ),
            channel_id="normalized-composite",
        )

    circuit = ParameterizedCircuit(2, [ParameterizedGate("rzz", (0, 1), angle_offset=0.0)])
    truncation = KrotovTruncation(max_bond_dim=1)
    sampled = _forward(
        circuit,
        np.array([]),
        noise_provider=provider,
        seed=0,
        truncation=truncation,
    )
    replayed = _forward(
        circuit,
        np.array([]),
        noise_maps=sampled.noise_maps,
        seed=999,
        truncation=truncation,
    )

    assert sampled.noise_maps[0].normalized is False
    assert sampled.noise_maps[0].normalization_checkpoints == (1,)
    assert sampled.noise_maps[0].is_identity is None
    np.testing.assert_allclose(replayed.states[-1].to_vec(), sampled.states[-1].to_vec(), atol=1e-12)
    assert replayed.states[-1].norm() == pytest.approx(sampled.states[-1].norm(), abs=1e-12)


def test_composite_replay_preserves_pauli_tjm_normalization_after_truncation() -> None:
    """A compact Pauli TJM outcome retains its composite normalization boundary."""
    model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 0.1}])

    def provider(
        context: GateNoiseContext,
        rng: np.random.Generator,
    ) -> CompositeGateNoiseInstruction:
        del context, rng
        return CompositeGateNoiseInstruction(
            (
                RandomUnitaryInstruction((
                    LocalOperator(_H, (0,), "H"),
                    LocalOperator(_CX, (0, 1), "CX"),
                )),
                TJMNoiseInstruction(model),
                RandomUnitaryInstruction((
                    LocalOperator(_CX, (0, 1), "CX"),
                    LocalOperator(_H, (0,), "H"),
                )),
            ),
            channel_id="pauli-normalized-composite",
        )

    circuit = ParameterizedCircuit(2, [ParameterizedGate("rzz", (0, 1), angle_offset=0.0)])
    truncation = KrotovTruncation(max_bond_dim=1)
    sampled = _forward(
        circuit,
        np.array([]),
        noise_provider=provider,
        seed=0,
        truncation=truncation,
    )
    replayed = _forward(
        circuit,
        np.array([]),
        noise_maps=sampled.noise_maps,
        seed=999,
        truncation=truncation,
    )

    assert sampled.noise_maps[0].normalized is False
    assert sampled.noise_maps[0].normalization_checkpoints == (2,)
    assert sampled.noise_maps[0].is_identity is None
    np.testing.assert_allclose(replayed.states[-1].to_vec(), sampled.states[-1].to_vec(), atol=1e-12)
    assert replayed.states[-1].norm() == pytest.approx(sampled.states[-1].norm(), abs=1e-12)


def test_provider_operator_support_must_be_within_current_gate() -> None:
    """A provider cannot attach a random operator to an unrelated site."""

    def provider(
        context: GateNoiseContext,
        rng: np.random.Generator,
    ) -> RandomUnitaryInstruction:
        del context, rng
        return _random_instruction(
            (LocalOperator(matrix=_X, sites=(1,)),),
            outcome_labels=("X",),
        )

    circuit = ParameterizedCircuit(2, [ParameterizedGate("rx", (0,), param_index=0)])
    with pytest.raises(ValueError, match=r"support|site"):
        _forward(circuit, np.array([0.2]), noise_provider=provider)


@pytest.mark.parametrize("representation", ["raw", "wrapped"])
def test_provider_noise_model_support_must_be_within_current_gate(representation: str) -> None:
    """Provider-local TJM models are validated instead of silently restricted."""
    model = NoiseModel([{"name": "pauli_x", "sites": [1], "strength": 0.1}])

    def provider(
        context: GateNoiseContext,
        rng: np.random.Generator,
    ) -> NoiseModel | TJMNoiseInstruction:
        del context, rng
        if representation == "wrapped":
            return TJMNoiseInstruction(noise_model=model, channel_id="invalid")
        return model

    circuit = ParameterizedCircuit(2, [ParameterizedGate("rx", (0,), param_index=0)])
    with pytest.raises(ValueError, match=r"support|site"):
        _forward(circuit, np.array([0.2]), noise_provider=provider)


def test_global_and_provider_noise_conflict_is_rejected() -> None:
    """Even an empty global model cannot be combined implicitly with a provider."""

    def provider(context: GateNoiseContext, rng: np.random.Generator) -> None:
        del context, rng

    circuit = ParameterizedCircuit(1, [ParameterizedGate("h", (0,))])
    with pytest.raises(ValueError, match=r"global|simultaneous|provider"):
        _forward(
            circuit,
            np.array([]),
            noise_model=NoiseModel([]),
            noise_provider=provider,
        )


def test_unsupported_provider_output_is_rejected() -> None:
    """Arbitrary categorical objects cannot bypass the supported instruction union."""

    def provider(context: GateNoiseContext, rng: np.random.Generator) -> object:
        del context, rng
        return object()

    circuit = ParameterizedCircuit(1, [ParameterizedGate("h", (0,))])
    with pytest.raises(TypeError, match=r"provider|instruction|NoiseModel"):
        _forward(circuit, np.array([]), noise_provider=cast("GateNoiseProvider", provider))


def test_fixed_map_replay_bypasses_provider() -> None:
    """Fixed-map replay is physical replay and does not request a fresh outcome."""

    def provider(context: GateNoiseContext, rng: np.random.Generator) -> None:
        del context, rng
        pytest.fail("fixed-map replay unexpectedly invoked the provider")

    fixed_map = KrotovNoiseMap(
        operators=((_X, (0,)),),
        channel_id="fixed",
        outcome_labels=("X",),
        source_gate_index=0,
        resolved_native_angle=0.0,
        is_identity=False,
    )
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), angle_offset=0.0)])
    trajectory = _forward(
        circuit,
        np.array([]),
        noise_provider=provider,
        noise_maps=[fixed_map],
    )

    np.testing.assert_allclose(trajectory.states[-1].to_vec(), np.array([0.0, 1.0]), atol=1e-12)
    assert trajectory.noise_maps[0].channel_id == "fixed"
    assert trajectory.noise_maps[0].outcome_labels == ("X",)


def test_noise_map_pullback_uses_reverse_adjoint_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """The adjoint of ``S @ H`` is replayed as ``H† @ S†`` on a costate."""
    applied: list[tuple[NDArray[np.complex128], tuple[int, ...]]] = []

    def record_application(
        _state: MPS,
        matrix: NDArray[np.complex128],
        sites: tuple[int, ...],
        _truncation: KrotovTruncation,
    ) -> None:
        applied.append((matrix.copy(), sites))

    monkeypatch.setattr(krotov_module, "_apply_operator", record_application)
    noise_map = KrotovNoiseMap(operators=((_H, (0,)), (_S, (0,))))
    pullback_noise_map = cast(
        "Callable[[MPS, KrotovNoiseMap, KrotovTruncation], None]",
        vars(krotov_module)["_pullback_noise_map"],
    )

    pullback_noise_map(MPS(1), noise_map, KrotovTruncation())

    assert [sites for _matrix, sites in applied] == [(0,), (0,)]
    np.testing.assert_allclose(applied[0][0], _S.conj().T)
    np.testing.assert_allclose(applied[1][0], _H.conj().T)


def test_provider_sampling_is_reproducible_from_trajectory_seed() -> None:
    """The provider consumes the trajectory RNG and reproduces realized maps."""

    def provider(
        context: GateNoiseContext,
        rng: np.random.Generator,
    ) -> RandomUnitaryInstruction:
        del context
        if rng.integers(2) == 0:
            return _random_instruction(
                (LocalOperator(matrix=_X, sites=(0,)),),
                channel_id="seeded",
                outcome_labels=("X",),
            )
        return _random_instruction(
            (LocalOperator(matrix=_Z, sites=(0,)),),
            channel_id="seeded",
            outcome_labels=("Z",),
        )

    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), param_index=0)])
    options = KrotovTJMOptions(num_trajectories=6, random_seed=19)
    target = np.array([1.0, 0.0], dtype=np.complex128)

    first = noisy_state_preparation_contribution(
        circuit,
        np.array([0.31]),
        target,
        None,
        options,
        MPS(1),
        KrotovTruncation(),
        iteration=4,
        noise_provider=provider,
    )
    second = noisy_state_preparation_contribution(
        circuit,
        np.array([0.31]),
        target,
        None,
        options,
        MPS(1),
        KrotovTruncation(),
        iteration=4,
        noise_provider=provider,
    )

    first_labels = [trajectory.noise_maps[0].outcome_labels for trajectory in first[3]]
    second_labels = [trajectory.noise_maps[0].outcome_labels for trajectory in second[3]]
    assert first_labels == second_labels
    for first_trajectory, second_trajectory in zip(first[3], second[3], strict=True):
        np.testing.assert_allclose(first_trajectory.states[-1].to_vec(), second_trajectory.states[-1].to_vec())


@pytest.mark.parametrize("identity_kind", ["none", "empty"])
def test_zero_noise_provider_matches_noiseless_evolution(identity_kind: str) -> None:
    """No instruction and an explicit identity channel are physically noiseless."""
    circuit = ParameterizedCircuit(
        1,
        [
            ParameterizedGate("rx", (0,), param_index=0),
            ParameterizedGate("ry", (0,), param_index=1),
        ],
    )
    theta = np.array([0.23, -0.41])
    baseline = _forward(circuit, theta)

    def provider(
        context: GateNoiseContext,
        rng: np.random.Generator,
    ) -> RandomUnitaryInstruction | None:
        del context, rng
        if identity_kind == "none":
            return None
        return _random_instruction((), channel_id="identity", outcome_labels=("I",))

    with_provider = _forward(circuit, theta, noise_provider=provider)

    np.testing.assert_allclose(with_provider.states[-1].to_vec(), baseline.states[-1].to_vec(), atol=1e-12)
    if identity_kind == "empty":
        assert all(noise_map.is_identity is True for noise_map in with_provider.noise_maps)


def test_noise_map_diagnostic_defaults_remain_backward_compatible() -> None:
    """Legacy maps omit diagnostics without changing their physical representation."""
    noise_map = KrotovNoiseMap()

    assert noise_map.channel_id is None
    assert noise_map.outcome_labels == ()
    assert noise_map.source_gate_index is None
    assert noise_map.resolved_native_angle is None
    assert noise_map.is_identity is None
    assert noise_map.normalization_checkpoints == ()


def test_provider_propagates_through_public_noisy_evaluation_apis() -> None:
    """Every public noisy evaluation API reaches the gate-local provider."""
    call_count = 0

    def provider(_context: GateNoiseContext, _rng: np.random.Generator) -> None:
        nonlocal call_count
        call_count += 1

    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), param_index=0)])
    theta = np.array([0.2])
    target = np.array([1.0, 0.0], dtype=np.complex128)
    options = KrotovTJMOptions(num_trajectories=1)
    truncation = KrotovTruncation()

    contribution = noisy_state_preparation_contribution(
        circuit,
        theta,
        target,
        None,
        options,
        MPS(1),
        truncation,
        noise_provider=provider,
    )
    cross = noisy_state_preparation_cross_contribution(
        circuit,
        theta,
        target,
        None,
        KrotovTJMOptions(num_trajectories=1, trajectory_update="cross"),
        MPS(1),
        truncation,
        noise_provider=provider,
    )
    metrics = noisy_state_preparation_metrics(
        circuit,
        theta,
        target,
        None,
        options,
        noise_provider=provider,
    )
    loss = noisy_state_preparation_loss(
        circuit,
        theta,
        target,
        None,
        options,
        noise_provider=provider,
    )
    readout = KrotovReadout(observable=Observable("z", 0), loss="mse")
    sample_contribution = noisy_sample_contribution(
        circuit,
        theta,
        np.array([], dtype=np.float64),
        1.0,
        readout,
        0.0,
        None,
        options,
        MPS(1),
        truncation,
        noise_provider=provider,
    )
    sample_loss = noisy_sample_loss(
        circuit,
        theta,
        np.array([], dtype=np.float64),
        1.0,
        readout,
        0.0,
        None,
        options,
        noise_provider=provider,
    )

    assert call_count == 6
    assert np.all(np.isfinite(contribution[0]))
    assert np.all(np.isfinite(cross[0]))
    assert np.all(np.isfinite(sample_contribution[0]))
    assert all(np.isfinite(value) for value in (*metrics[:2], loss, *sample_contribution[1:3], *sample_loss[:2]))


@pytest.mark.parametrize(
    "trainer",
    [
        train_krotov_noisy_state_preparation_online,
        train_krotov_noisy_state_preparation_batch,
        train_krotov_noisy_state_preparation_hybrid,
    ],
)
def test_provider_propagates_through_noisy_trainers_with_crn(
    trainer: _NoisyStatePreparationTrainer,
) -> None:
    """All noisy trainers sample provider maps once and replay them under CRN."""
    call_count = 0

    def provider(_context: GateNoiseContext, _rng: np.random.Generator) -> None:
        nonlocal call_count
        call_count += 1

    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), param_index=0)])
    target = np.array([1.0, 0.0], dtype=np.complex128)
    result = trainer(
        circuit,
        target,
        None,
        KrotovTJMOptions(num_trajectories=1, use_crn=True),
        initial_theta=np.array([0.2]),
        options=KrotovOptions(max_iterations=1, switch_iteration=0),
        noise_provider=provider,
    )

    assert call_count == 1
    assert np.all(np.isfinite(result.theta))
