# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for deterministic, checksum-sealed Krotov fixed-map ensembles."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.optimization import (
    CompositeGateNoiseInstruction,
    GateNoiseContext,
    KrotovFixedMapEnsemble,
    KrotovMapSchedule,
    KrotovNoiseMap,
    KrotovTJMOptions,
    KrotovTruncation,
    LocalOperator,
    ParameterizedCircuit,
    ParameterizedGate,
    RandomUnitaryInstruction,
    TJMNoiseInstruction,
    derive_krotov_trajectory_seed,
    sample_krotov_fixed_map_ensemble,
)

if TYPE_CHECKING:
    from mqt.yaqs.optimization import GateNoiseProvider, KrotovMapRole

_CHECKSUM_A = f"sha256:{'a' * 64}"
_CHECKSUM_B = f"sha256:{'b' * 64}"
_CHECKSUM_C = f"sha256:{'c' * 64}"
_CHECKSUM_D = f"sha256:{'d' * 64}"
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def _map(matrix: np.ndarray = _X, *, is_identity: bool | None = False) -> KrotovNoiseMap:
    """Build one diagnostic-rich fixed map.

    Returns:
        A map suitable for serialization and replay tests.
    """
    return KrotovNoiseMap(
        operators=((matrix, (0,)),),
        normalized=False,
        jump_process_index=1,
        channel_id="test_channel",
        outcome_labels=("X",),
        source_gate_index=0,
        resolved_native_angle=0.25,
        is_identity=is_identity,
        normalization_checkpoints=(1,),
    )


def _ensemble(
    trajectory_maps: list[list[KrotovNoiseMap]] | None = None,
    *,
    circuit_checksum: str = _CHECKSUM_B,
    provider_checksum: str = _CHECKSUM_C,
    ensemble_index: int = 3,
) -> KrotovFixedMapEnsemble:
    """Build one small fixed-map ensemble.

    Returns:
        A fully bound ensemble with stable test metadata.
    """
    return KrotovFixedMapEnsemble(
        role="training_trajectory",
        resolved_seed=17,
        stage_index=2,
        stage_id="noisy_finetune",
        stage_configuration_checksum=_CHECKSUM_A,
        circuit_checksum=circuit_checksum,
        provider_checksum=provider_checksum,
        ensemble_index=ensemble_index,
        refresh_index=1,
        global_iteration_start=20,
        trajectory_maps=[[_map()]] if trajectory_maps is None else trajectory_maps,
    )


def test_trajectory_seed_derivation_is_stable_and_domain_separated() -> None:
    """SHA-256 trajectory seeds are stable and every explicit coordinate separates streams."""
    seed = derive_krotov_trajectory_seed(
        role="training_trajectory",
        resolved_seed=17,
        stage_index=2,
        ensemble_index=3,
        trajectory_index=4,
        refresh_index=5,
    )
    assert seed == 11819421565009480905
    assert 0 <= seed < 2**64

    changed_seeds = (
        *(
            derive_krotov_trajectory_seed(
                role=cast("KrotovMapRole", role),
                resolved_seed=17,
                stage_index=2,
                ensemble_index=3,
                trajectory_index=4,
                refresh_index=5,
            )
            for role in (
                "checkpoint_validation",
                "pilot_evaluation",
                "screening_selection",
                "confirmatory_test",
            )
        ),
        derive_krotov_trajectory_seed(
            role="training_trajectory",
            resolved_seed=18,
            stage_index=2,
            ensemble_index=3,
            trajectory_index=4,
            refresh_index=5,
        ),
        derive_krotov_trajectory_seed(
            role="training_trajectory",
            resolved_seed=17,
            stage_index=3,
            ensemble_index=3,
            trajectory_index=4,
            refresh_index=5,
        ),
        derive_krotov_trajectory_seed(
            role="training_trajectory",
            resolved_seed=17,
            stage_index=2,
            ensemble_index=4,
            trajectory_index=4,
            refresh_index=5,
        ),
        derive_krotov_trajectory_seed(
            role="training_trajectory",
            resolved_seed=17,
            stage_index=2,
            ensemble_index=3,
            trajectory_index=5,
            refresh_index=5,
        ),
        derive_krotov_trajectory_seed(
            role="training_trajectory",
            resolved_seed=17,
            stage_index=2,
            ensemble_index=3,
            trajectory_index=4,
            refresh_index=6,
        ),
    )
    assert all(changed_seed != seed for changed_seed in changed_seeds)

    with pytest.raises(TypeError, match="resolved_seed must be an integer"):
        derive_krotov_trajectory_seed(
            role="training_trajectory",
            resolved_seed=True,
            stage_index=2,
            ensemble_index=3,
            trajectory_index=4,
            refresh_index=5,
        )


def test_ensemble_defensively_freezes_canonical_little_endian_matrices() -> None:
    """Source and replay-array mutations cannot alter sealed matrix content."""
    source = np.asfortranarray(_X.astype(">c16"))
    ensemble = _ensemble([[_map(source)]])
    expected_checksum = ensemble.content_checksum

    source[:] = 0.0
    first_replay = ensemble.replay_maps()
    assert np.array_equal(first_replay[0][0].operators[0][0], _X)
    assert first_replay[0][0].operators[0][0].dtype == np.dtype("<c16")
    first_replay[0][0].operators[0][0][:] = 0.0

    second_replay = ensemble.replay_maps()
    assert np.array_equal(second_replay[0][0].operators[0][0], _X)
    assert ensemble.content_checksum == expected_checksum
    serialized = cast("dict[str, Any]", json.loads(ensemble.to_json()))
    assert serialized["trajectory_maps"][0][0]["operators"][0]["dtype"] == "<c16"


def test_logical_identity_is_separate_from_bound_replay_content() -> None:
    """Circuit, provider, and map content change the checksum but not logical coordinates."""
    first = _ensemble([[_map(_X)]])
    changed_map = _ensemble([[_map(_Z)]])
    changed_bindings = _ensemble([[_map(_X)]], circuit_checksum=_CHECKSUM_D, provider_checksum=_CHECKSUM_B)

    assert first.ensemble_id == changed_map.ensemble_id == changed_bindings.ensemble_id
    assert len({first.content_checksum, changed_map.content_checksum, changed_bindings.content_checksum}) == 3
    assert _ensemble(ensemble_index=4).ensemble_id != first.ensemble_id
    assert _ensemble([[_map()], [_map()]]).ensemble_id != first.ensemble_id


def test_ensemble_round_trip_checksum_and_tamper_verification() -> None:
    """Canonical serialization round-trips and rejects altered sealed fields."""
    ensemble = _ensemble([[_map()], [KrotovNoiseMap(is_identity=True)]])
    payload = ensemble.to_json()
    restored = KrotovFixedMapEnsemble.from_json(payload + "\n")

    assert restored.to_json() == payload
    assert restored.ensemble_id == ensemble.ensemble_id
    assert restored.content_checksum == ensemble.content_checksum
    assert restored.nonidentity_event_count == 1

    tampered = copy.deepcopy(ensemble.to_dict())
    tampered["circuit_checksum"] = _CHECKSUM_D
    with pytest.raises(ValueError, match="content checksum mismatch"):
        KrotovFixedMapEnsemble.from_dict(tampered)

    tampered_matrix = cast("dict[str, Any]", json.loads(payload))
    encoded = cast("str", tampered_matrix["trajectory_maps"][0][0]["operators"][0]["data_base64"])
    replacement = "A" if encoded[0] != "A" else "B"
    tampered_matrix["trajectory_maps"][0][0]["operators"][0]["data_base64"] = replacement + encoded[1:]
    with pytest.raises(ValueError, match="content checksum mismatch"):
        KrotovFixedMapEnsemble.from_dict(tampered_matrix)

    wrong_identity = copy.deepcopy(ensemble.to_dict())
    wrong_identity["ensemble_id"] = f"krotov_map_ensemble_{'0' * 64}"
    with pytest.raises(ValueError, match="ensemble_id"):
        KrotovFixedMapEnsemble.from_dict(wrong_identity)


def test_ensemble_json_requires_canonical_unique_members() -> None:
    """Noncanonical whitespace and duplicate JSON member names are rejected."""
    ensemble = _ensemble()
    noncanonical = json.dumps(ensemble.to_dict())
    with pytest.raises(ValueError, match="canonical form"):
        KrotovFixedMapEnsemble.from_json(noncanonical)
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        KrotovFixedMapEnsemble.from_json('{"a":1,"a":2}')


def test_ensemble_rejects_invalid_map_shapes_and_nonfinite_entries() -> None:
    """Malformed local operators cannot enter a fixed replay ensemble."""
    with pytest.raises(ValueError, match="must have shape"):
        _ensemble([[KrotovNoiseMap(operators=((np.eye(4, dtype=np.complex128), (0,)),))]])
    invalid = _X.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite entries"):
        _ensemble([[_map(invalid)]])


def test_ensemble_rejects_a_map_bound_to_the_wrong_gate_position() -> None:
    """Serialized source-gate metadata cannot disagree with replay ordering."""
    wrong_source = KrotovNoiseMap(source_gate_index=1, is_identity=True)
    with pytest.raises(ValueError, match="source_gate_index"):
        _ensemble([[wrong_source]])


def test_replay_binding_validation_is_explicit() -> None:
    """Callers can verify exact stage, circuit, and provider bindings before replay."""
    ensemble = _ensemble()
    ensemble.verify_bindings(
        stage_configuration_checksum=_CHECKSUM_A,
        circuit_checksum=_CHECKSUM_B,
        provider_checksum=_CHECKSUM_C,
    )
    with pytest.raises(ValueError, match="provider_checksum"):
        ensemble.verify_bindings(provider_checksum=_CHECKSUM_D)


@pytest.mark.parametrize(
    ("schedule", "expected"),
    [
        (
            KrotovMapSchedule("resampled", global_iteration_offset=7),
            [(7, 7, 7, True), (8, 8, 8, True), (9, 9, 9, True)],
        ),
        (
            KrotovMapSchedule("crn_fixed", global_iteration_offset=0),
            [(0, 0, 0, True), (1, 0, 0, False), (2, 0, 0, False)],
        ),
        (
            KrotovMapSchedule("crn_refresh", 3, global_iteration_offset=2),
            [(2, 0, 0, False), (3, 1, 1, True), (4, 1, 1, False)],
        ),
    ],
)
def test_map_schedule_resolves_policy_and_global_offsets(
    schedule: KrotovMapSchedule,
    expected: list[tuple[int, int, int, bool]],
) -> None:
    """All sampling policies resolve schedule-continuous coordinates."""
    actual = []
    for local_iteration in range(3):
        point = schedule.point(local_iteration)
        actual.append((
            point.global_iteration,
            point.ensemble_index,
            point.refresh_index,
            point.is_refresh_boundary,
        ))
        assert schedule.indices_for_iteration(local_iteration) == (point.ensemble_index, point.refresh_index)
    assert actual == expected


def test_map_schedule_rejects_inconsistent_refresh_settings() -> None:
    """Only periodic CRN schedules accept a positive refresh interval."""
    with pytest.raises(ValueError, match="requires a positive"):
        KrotovMapSchedule("crn_refresh")
    with pytest.raises(ValueError, match="positive"):
        KrotovMapSchedule("crn_refresh", 0)
    with pytest.raises(ValueError, match="valid only"):
        KrotovMapSchedule("crn_fixed", 2)


def test_sampling_uses_forward_path_once_per_trajectory_and_replays_without_provider() -> None:
    """Sampling calls the provider once per noisy gate, while map replay is provider-free."""
    calls: list[GateNoiseContext] = []

    def provider(context: GateNoiseContext, rng: np.random.Generator) -> RandomUnitaryInstruction:
        del rng
        calls.append(context)
        return RandomUnitaryInstruction(
            operators=(LocalOperator(_X, context.sites, label="X"),),
            channel_id="fixed_x",
            outcome_labels=("X",),
        )

    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), param_index=0)])
    ensemble = sample_krotov_fixed_map_ensemble(
        circuit,
        np.array([0.2]),
        MPS(1),
        KrotovTruncation(),
        provider,
        KrotovTJMOptions(num_trajectories=3, random_seed=999),
        role="training_trajectory",
        resolved_seed=23,
        stage_index=1,
        stage_id="noisy_finetune",
        stage_configuration_checksum=_CHECKSUM_A,
        circuit_checksum=_CHECKSUM_B,
        provider_checksum=_CHECKSUM_C,
        ensemble_index=0,
        refresh_index=0,
        global_iteration_start=0,
    )

    assert len(calls) == 3
    assert ensemble.trajectory_count == 3
    assert ensemble.gate_count == 1
    assert ensemble.nonidentity_event_count == 3
    replayed = ensemble.replay_maps()
    assert len(calls) == 3
    assert all(np.array_equal(maps[0].operators[0][0], _X) for maps in replayed)


def test_legacy_compact_sampling_rejects_intermediate_normalization_checkpoints() -> None:
    """The archived compact-map convention cannot flatten composite checkpoints."""
    model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 2.0}])

    def provider(
        context: GateNoiseContext,
        rng: np.random.Generator,
    ) -> CompositeGateNoiseInstruction:
        del rng
        return CompositeGateNoiseInstruction((
            TJMNoiseInstruction(model, channel_id="tjm"),
            RandomUnitaryInstruction(
                operators=(LocalOperator(_Z, context.sites, label="Z"),),
                channel_id="unitary",
            ),
        ))

    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), param_index=0)])
    with pytest.raises(ValueError, match="requires no normalization checkpoints"):
        sample_krotov_fixed_map_ensemble(
            circuit,
            np.array([0.2]),
            MPS(1),
            KrotovTruncation(),
            provider,
            KrotovTJMOptions(),
            role="training_trajectory",
            resolved_seed=23,
            stage_index=1,
            stage_id="legacy_noisy_finetune",
            stage_configuration_checksum=_CHECKSUM_A,
            circuit_checksum=_CHECKSUM_B,
            provider_checksum=_CHECKSUM_C,
            ensemble_index=0,
            refresh_index=0,
            global_iteration_start=0,
            legacy_linear_seed=True,
            legacy_compact_replay=True,
        )


def test_sampling_is_execution_order_reproducible_and_ignores_tjm_seed() -> None:
    """Explicit coordinate seeds make maps independent of call order and legacy TJM seeds."""
    provider = _draw_recording_provider()
    first = _sample_draw_ensemble(provider, random_seed=1)
    _sample_draw_ensemble(provider, random_seed=77, ensemble_index=12)
    second = _sample_draw_ensemble(provider, random_seed=999)

    assert first.to_json() == second.to_json()


def _draw_recording_provider() -> GateNoiseProvider:
    """Create a provider whose labels expose its trajectory-local random draw.

    Returns:
        A state-independent provider suitable for seed/reproducibility tests.
    """

    def provider(context: GateNoiseContext, rng: np.random.Generator) -> RandomUnitaryInstruction:
        del context
        return RandomUnitaryInstruction(
            channel_id="draw_recorder",
            outcome_labels=(f"draw_{float(rng.random()).hex()}",),
        )

    return provider


def _sample_draw_ensemble(
    provider: GateNoiseProvider,
    *,
    random_seed: int,
    ensemble_index: int = 0,
) -> KrotovFixedMapEnsemble:
    """Sample a two-trajectory ensemble with a legacy TJM seed variation.

    Returns:
        The sampled fixed-map ensemble.
    """
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), param_index=0)])
    return sample_krotov_fixed_map_ensemble(
        circuit,
        np.array([0.2]),
        None,
        KrotovTruncation(),
        provider,
        KrotovTJMOptions(num_trajectories=2, random_seed=random_seed),
        role="training_trajectory",
        resolved_seed=31,
        stage_index=0,
        stage_id="stage_zero",
        stage_configuration_checksum=_CHECKSUM_A,
        circuit_checksum=_CHECKSUM_B,
        provider_checksum=_CHECKSUM_C,
        ensemble_index=ensemble_index,
        refresh_index=0,
        global_iteration_start=0,
    )
