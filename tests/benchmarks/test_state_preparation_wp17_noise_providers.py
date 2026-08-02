# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for WP17 scaled and historical fixed-rate noise providers."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from benchmarks.state_preparation.constants import BALLARIN_NOISE_ID, NOISELESS_NOISE_ID, STANDARD_NOISE_IDS
from benchmarks.state_preparation.noise import (
    FIXED_RATE_NOISE_DEFINITION_VERSION,
    HISTORICAL_FIXED_RATE_NOISE_ID,
    STANDARD_NOISE_REGISTRY,
    HistoricalFixedRateNoiseProvider,
    ScaledStandardNoiseProvider,
    create_historical_fixed_rate_noise_provider,
    create_scaled_standard_noise_provider,
    create_standard_noise_provider,
)
from mqt.yaqs.optimization import GateNoiseContext, TJMNoiseInstruction

if TYPE_CHECKING:
    from mqt.yaqs.core.data_structures.noise_model import NoiseModel


_ONE_QUBIT_CONTEXT = GateNoiseContext(0, "ry", (4,), 1, 0.3, "logical-0", "native-0", 0)
_TWO_QUBIT_CONTEXT = GateNoiseContext(1, "rzz", (1, 3), 2, -0.2, "logical-1", "native-1", 1)
_ADJACENT_TWO_QUBIT_CONTEXT = GateNoiseContext(1, "rzz", (1, 2), 2, -0.2, "logical-1", "native-1", 1)
_EXPECTED_PROCESS_COUNTS = {
    "dephasing_1s_1q": (1, 0),
    "dephasing_1s_2q": (0, 2),
    "dephasing_1s_all": (1, 2),
    "dephasing_2s_2q": (0, 1),
    "dephasing_1s2s_all": (1, 3),
    "depolarizing_1s_1q": (3, 0),
    "depolarizing_1s_2q": (0, 6),
    "depolarizing_1s_all": (3, 6),
    "depolarizing_2s_2q": (0, 9),
    "depolarizing_1s2s_all": (3, 15),
}


class _ForbiddenRNG:
    """Test double rejecting provider-side random sampling."""

    @staticmethod
    def random() -> float:
        """Reject an unexpected uniform draw.

        Raises:
            AssertionError: Always.
        """
        msg = "Fixed-rate providers must not consume randomness while constructing a model."
        raise AssertionError(msg)

    @staticmethod
    def choice(*args: object, **kwargs: object) -> int:
        """Reject an unexpected categorical draw.

        Raises:
            AssertionError: Always.
        """
        del args, kwargs
        msg = "Fixed-rate providers must not consume randomness while constructing a model."
        raise AssertionError(msg)


def _forbidden_rng() -> np.random.Generator:
    """Return the strict RNG double under the provider protocol type."""
    return cast("np.random.Generator", _ForbiddenRNG())


def _process_snapshot(noise_model: NoiseModel) -> list[tuple[str, tuple[int, ...], float]]:
    """Return ordered process names, supports, and strengths."""
    return [
        (str(process["name"]), tuple(cast("list[int]", process["sites"])), float(process["strength"]))
        for process in noise_model.processes
    ]


@pytest.mark.parametrize("noise_id", STANDARD_NOISE_IDS)
@pytest.mark.parametrize(
    ("context", "count_index"),
    [(_ONE_QUBIT_CONTEXT, 0), (_TWO_QUBIT_CONTEXT, 1)],
)
def test_scaled_provider_covers_all_standard_processes_exactly(
    noise_id: str,
    context: GateNoiseContext,
    count_index: int,
) -> None:
    """Every standard profile retains exact ordering/sites and scales only strengths."""
    scale = 2.5
    canonical = create_standard_noise_provider(noise_id)(context, _forbidden_rng())
    scaled = create_scaled_standard_noise_provider(noise_id, scale)(context, _forbidden_rng())
    expected_count = _EXPECTED_PROCESS_COUNTS[noise_id][count_index]

    assert (canonical is None) == (expected_count == 0)
    assert (scaled is None) == (expected_count == 0)
    if canonical is None or scaled is None:
        return
    assert isinstance(canonical, TJMNoiseInstruction)
    assert isinstance(scaled, TJMNoiseInstruction)
    canonical_snapshot = _process_snapshot(canonical.noise_model)
    scaled_snapshot = _process_snapshot(scaled.noise_model)
    assert len(scaled_snapshot) == expected_count
    assert [(name, sites) for name, sites, _strength in scaled_snapshot] == [
        (name, sites) for name, sites, _strength in canonical_snapshot
    ]
    assert [strength for _name, _sites, strength in scaled_snapshot] == pytest.approx(
        [strength * scale for _name, _sites, strength in canonical_snapshot],
    )


def test_scaled_provider_identity_is_complete_deterministic_and_immutable() -> None:
    """Base ID, version, scale, and logical placement fully identify scaling."""
    registry_snapshot = json.dumps(
        {noise_id: definition.to_dict() for noise_id, definition in STANDARD_NOISE_REGISTRY.items()},
        sort_keys=True,
        separators=(",", ":"),
    )
    provider = create_scaled_standard_noise_provider("depolarizing_1s_all", np.float64(2.0))
    repeated = create_scaled_standard_noise_provider("depolarizing_1s_all", 2)

    assert isinstance(provider, ScaledStandardNoiseProvider)
    assert provider.definition is STANDARD_NOISE_REGISTRY["depolarizing_1s_all"]
    assert type(provider.strength_scale) is float
    assert provider.identity == (
        "depolarizing_1s_all",
        FIXED_RATE_NOISE_DEFINITION_VERSION,
        2.0,
        "logical_parameterized_gates",
    )
    assert provider.to_dict() == {
        "base_noise_id": "depolarizing_1s_all",
        "noise_definition_version": "yaqs.state_preparation.noise.v1",
        "strength_scale": 2.0,
        "gate_placement": "logical_parameterized_gates",
    }
    assert repeated == provider
    assert repeated.content_checksum == provider.content_checksum
    assert provider.content_checksum == "sha256:3eb6c717b0894d08d8653123858685e818f0406c845d79d4631a85277d0d6d66"
    assert create_scaled_standard_noise_provider("depolarizing_1s_all", 1.0).content_checksum != (
        provider.content_checksum
    )
    with pytest.raises(FrozenInstanceError):
        provider.strength_scale = 3.0  # ty: ignore[invalid-assignment]

    detached = provider.to_dict()
    detached["strength_scale"] = 7.0
    assert provider.strength_scale == pytest.approx(2.0)
    assert registry_snapshot == json.dumps(
        {noise_id: definition.to_dict() for noise_id, definition in STANDARD_NOISE_REGISTRY.items()},
        sort_keys=True,
        separators=(",", ":"),
    )


@pytest.mark.parametrize(
    ("scale", "error"),
    [
        (True, TypeError),
        (np.bool_(0), TypeError),
        ("1", TypeError),
        (None, TypeError),
        (0.0, ValueError),
        (-1.0, ValueError),
        (float("inf"), ValueError),
        (float("-inf"), ValueError),
        (float("nan"), ValueError),
    ],
)
def test_scaled_provider_rejects_non_positive_finite_real_scales(
    scale: object,
    error: type[Exception],
) -> None:
    """Only strictly positive finite real scales can enter provider identity."""
    with pytest.raises(error, match="strength_scale"):
        create_scaled_standard_noise_provider("dephasing_1s_all", cast("Any", scale))


@pytest.mark.parametrize(
    "noise_id",
    [NOISELESS_NOISE_ID, BALLARIN_NOISE_ID, HISTORICAL_FIXED_RATE_NOISE_ID, "unknown"],
)
def test_scaled_factory_accepts_only_frozen_standard_base_ids(noise_id: str) -> None:
    """Noiseless, Ballarin, historical, and unknown profiles cannot masquerade as standard."""
    with pytest.raises(ValueError, match="standard noise"):
        create_scaled_standard_noise_provider(noise_id, 1.0)


def test_scaled_provider_constructs_fresh_detached_models() -> None:
    """Each call returns mutable working data detached from peers and the registry."""
    provider = create_scaled_standard_noise_provider("depolarizing_1s2s_all", 1.5)
    first = provider(_TWO_QUBIT_CONTEXT, _forbidden_rng())
    second = provider(_TWO_QUBIT_CONTEXT, _forbidden_rng())
    assert first is not None
    assert second is not None
    assert first.noise_model is not second.noise_model
    assert first.noise_model.processes is not second.noise_model.processes
    for first_process, second_process in zip(
        first.noise_model.processes,
        second.noise_model.processes,
        strict=True,
    ):
        assert first_process is not second_process
        if "matrix" in first_process:
            assert first_process["matrix"] is not second_process["matrix"]
            np.testing.assert_allclose(first_process["matrix"], second_process["matrix"])
        if "factors" in first_process:
            assert all(
                first_factor is not second_factor
                for first_factor, second_factor in zip(
                    first_process["factors"],
                    second_process["factors"],
                    strict=True,
                )
            )

    expected_strength = float(second.noise_model.processes[0]["strength"])
    first.noise_model.processes[0]["strength"] = 99.0
    assert float(second.noise_model.processes[0]["strength"]) == pytest.approx(expected_strength)
    assert provider.strength_scale == pytest.approx(1.5)


def test_historical_profile_metadata_is_exact_and_disclaims_hardware() -> None:
    """The archived label is frozen as a logical TJM simulation, not Ballarin or QPU data."""
    provider = create_historical_fixed_rate_noise_provider()
    repeated = HistoricalFixedRateNoiseProvider()

    assert provider == repeated
    assert provider.noise_id == "ibm_inspired_pauli_legacy_v1"
    assert provider.noise_id == HISTORICAL_FIXED_RATE_NOISE_ID
    assert provider.noise_definition_version == FIXED_RATE_NOISE_DEFINITION_VERSION
    assert provider.gate_placement == "logical_parameterized_gates"
    assert provider.tjm_dt == pytest.approx(1.0)
    assert provider.identity == (
        HISTORICAL_FIXED_RATE_NOISE_ID,
        FIXED_RATE_NOISE_DEFINITION_VERSION,
        "logical_parameterized_gates",
        1.0,
    )
    with pytest.raises(FrozenInstanceError):
        provider.noise_id = "changed"  # ty: ignore[invalid-assignment]
    metadata = provider.to_dict()
    assert metadata["channel_semantics"] == "fixed_rate_logical_tjm_simulation"
    assert metadata["is_ballarin"] is False
    assert metadata["is_hardware_execution"] is False
    assert provider.noise_id != BALLARIN_NOISE_ID
    assert repeated.content_checksum == provider.content_checksum
    assert provider.content_checksum == "sha256:6c10d1fdd11f57a546529b70343fe5aaf8441446e9b2e0e8c637f7d38aecf3de"
    assert metadata["two_qubit_crosstalk_connectivity"] == "adjacent_linear_chain_only"
    assert json.loads(json.dumps(metadata, sort_keys=True)) == metadata


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        (
            _ONE_QUBIT_CONTEXT,
            [
                ("pauli_x", (4,), 1.0e-4),
                ("pauli_y", (4,), 1.0e-4),
                ("pauli_z", (4,), 1.0e-4),
            ],
        ),
        (
            _ADJACENT_TWO_QUBIT_CONTEXT,
            [
                ("pauli_x", (1,), 1.0e-4),
                ("pauli_y", (1,), 1.0e-4),
                ("pauli_z", (1,), 1.0e-4),
                ("pauli_x", (2,), 1.0e-4),
                ("pauli_y", (2,), 1.0e-4),
                ("pauli_z", (2,), 1.0e-4),
                ("crosstalk_xx", (1, 2), 1.5e-3),
                ("crosstalk_zz", (1, 2), 1.5e-3),
            ],
        ),
    ],
)
def test_historical_profile_builds_exact_gate_local_processes(
    context: GateNoiseContext,
    expected: list[tuple[str, tuple[int, ...], float]],
) -> None:
    """Historical single-site and XX/ZZ strengths follow archived logical filtering."""
    provider = create_historical_fixed_rate_noise_provider()
    first = provider(context, _forbidden_rng())
    second = provider(context, _forbidden_rng())

    assert first.channel_id == HISTORICAL_FIXED_RATE_NOISE_ID
    assert _process_snapshot(first.noise_model) == expected
    assert _process_snapshot(second.noise_model) == expected
    assert first.noise_model is not second.noise_model
    for first_process, second_process in zip(
        first.noise_model.processes,
        second.noise_model.processes,
        strict=True,
    ):
        if "matrix" in first_process:
            assert first_process["matrix"] is not second_process["matrix"]
        if "factors" in first_process:
            assert all(
                first_factor is not second_factor
                for first_factor, second_factor in zip(
                    first_process["factors"],
                    second_process["factors"],
                    strict=True,
                )
            )


def test_historical_profile_omits_crosstalk_on_nonadjacent_logical_support() -> None:
    """Archived global-model filtering had XX/ZZ processes only on chain edges."""
    instruction = create_historical_fixed_rate_noise_provider()(_TWO_QUBIT_CONTEXT, _forbidden_rng())
    snapshot = _process_snapshot(instruction.noise_model)
    assert len(snapshot) == 6
    assert all(len(sites) == 1 for _name, sites, _strength in snapshot)


@pytest.mark.parametrize(
    "provider",
    [
        create_scaled_standard_noise_provider("dephasing_1s_all", 1.0),
        create_historical_fixed_rate_noise_provider(),
    ],
)
def test_wp17_fixed_rate_providers_reject_invalid_contexts(provider: object) -> None:
    """Both provider APIs enforce immutable gate-context inputs."""
    with pytest.raises(TypeError, match="GateNoiseContext"):
        cast("Any", provider)(None, _forbidden_rng())
