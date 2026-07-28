# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the standard state-preparation noise registry."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from benchmarks.state_preparation import (
    BALLARIN_NOISE_ID,
    NOISELESS_NOISE_ID,
    STANDARD_NOISE_IDS,
    STANDARD_NOISE_REGISTRY,
    STANDARD_NOISE_STRENGTH_INTERPRETATION,
    STANDARD_ONE_QUBIT_GATE_STRENGTH,
    STANDARD_TWO_QUBIT_GATE_STRENGTH,
    TWO_SITE_DEPOLARIZING_OPERATORS,
    NoiseConfig,
    StandardNoiseDefinition,
    StandardNoiseProvider,
    create_standard_noise_provider,
    get_standard_noise_definition,
)
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.optimization import (
    GateNoiseContext,
    KrotovTJMOptions,
    KrotovTruncation,
    ParameterizedCircuit,
    ParameterizedGate,
    RandomUnitaryInstruction,
    TJMNoiseInstruction,
    forward_tjm_trajectory,
)
from mqt.yaqs.optimization.gate_noise import validate_gate_noise_instruction

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import GateNoiseInstruction, KrotovNoiseMap


_PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
_PAULI_MATRICES: Mapping[str, NDArray[np.complex128]] = {
    "x": _PAULI_X,
    "y": _PAULI_Y,
    "z": _PAULI_Z,
}
_PAULI_PROCESS_NAMES = ("pauli_x", "pauli_y", "pauli_z")
_TWO_SITE_PROCESS_NAMES = tuple(f"crosstalk_{label.lower()}" for label in TWO_SITE_DEPOLARIZING_OPERATORS)

_EXPECTED_DEFINITIONS = {
    "dephasing_1s_1q": ("dephasing", "single_site", "single_qubit_gates"),
    "dephasing_1s_2q": ("dephasing", "single_site", "multi_qubit_gates"),
    "dephasing_1s_all": ("dephasing", "single_site", "all_gates"),
    "dephasing_2s_2q": ("dephasing", "two_site", "multi_qubit_gates"),
    "dephasing_1s2s_all": ("dephasing", "single_site_and_two_site", "all_gates"),
    "depolarizing_1s_1q": ("depolarizing", "single_site", "single_qubit_gates"),
    "depolarizing_1s_2q": ("depolarizing", "single_site", "multi_qubit_gates"),
    "depolarizing_1s_all": ("depolarizing", "single_site", "all_gates"),
    "depolarizing_2s_2q": ("depolarizing", "two_site", "multi_qubit_gates"),
    "depolarizing_1s2s_all": ("depolarizing", "single_site_and_two_site", "all_gates"),
}

_EXPECTED_COUNTS_AND_INDICES = {
    "dephasing_1s_1q": ((1, 0, 1, 0), (0, 2), 2),
    "dephasing_1s_2q": ((0, 2, 0, 2), (1, 3), 4),
    "dephasing_1s_all": ((1, 2, 1, 2), (0, 1, 2, 3), 6),
    "dephasing_2s_2q": ((0, 1, 0, 1), (1, 3), 2),
    "dephasing_1s2s_all": ((1, 3, 1, 3), (0, 1, 2, 3), 8),
    "depolarizing_1s_1q": ((3, 0, 3, 0), (0, 2), 6),
    "depolarizing_1s_2q": ((0, 6, 0, 6), (1, 3), 12),
    "depolarizing_1s_all": ((3, 6, 3, 6), (0, 1, 2, 3), 18),
    "depolarizing_2s_2q": ((0, 9, 0, 9), (1, 3), 18),
    "depolarizing_1s2s_all": ((3, 15, 3, 15), (0, 1, 2, 3), 36),
}

_CONTEXTS = (
    GateNoiseContext(0, "custom_1q", (0,), 1, None, "logical-0", "native-0", None),
    GateNoiseContext(1, "rzz", (0, 2), 2, 0.41, "logical-1", "native-1", 0),
    GateNoiseContext(2, "ry", (2,), 1, -0.27, "logical-2", "native-2", 1),
    GateNoiseContext(3, "custom_2q", (1, 2), 2, None, "logical-3", "native-3", None),
)


class _ForbiddenRNG:
    """RNG test double that fails if a provider tries to sample."""

    @staticmethod
    def random() -> float:
        """Reject an unexpected provider-side random draw.

        Raises:
            AssertionError: Always.
        """
        msg = "Standard providers must not consume randomness while building a model."
        raise AssertionError(msg)

    @staticmethod
    def choice(*args: object, **kwargs: object) -> int:
        """Reject an unexpected provider-side categorical draw.

        Raises:
            AssertionError: Always.
        """
        del args, kwargs
        msg = "Standard providers must not consume randomness while building a model."
        raise AssertionError(msg)


class _NoJumpRNG:
    """RNG test double that deterministically selects TJM no-jump outcomes."""

    def __init__(self) -> None:
        """Initialize call counters."""
        self.random_calls = 0
        self.choice_calls = 0

    def random(self) -> float:
        """Return a draw safely above every standard jump probability."""
        self.random_calls += 1
        return np.nextafter(1.0, 0.0)

    def choice(self, *args: object, **kwargs: object) -> int:
        """Reject an unexpected jump-process choice.

        Raises:
            AssertionError: Always, because the configured draw selects no jump.
        """
        del args, kwargs
        self.choice_calls += 1
        msg = "A deterministic no-jump trajectory unexpectedly selected a process."
        raise AssertionError(msg)


class _LastJumpRNG:
    """RNG test double that forces the final process of one TJM model."""

    def __init__(self) -> None:
        """Initialize call counters."""
        self.random_calls = 0
        self.choice_calls = 0

    def random(self) -> float:
        """Return zero to force a positive-rate jump."""
        self.random_calls += 1
        return 0.0

    def choice(
        self,
        size: int,
        *,
        p: NDArray[np.float64],
    ) -> int:
        """Choose the last available jump process.

        Returns:
            The last process index.
        """
        assert p.shape == (size,)
        assert np.sum(p) == pytest.approx(1.0)
        self.choice_calls += 1
        return size - 1


def _as_generator(rng: object) -> np.random.Generator:
    """Return an RNG test double cast to the production protocol."""
    return cast("np.random.Generator", rng)


def _single_site_processes(
    family: str,
    sites: tuple[int, ...],
    strength: float,
) -> tuple[tuple[str, tuple[int, ...], float], ...]:
    """Return the expected single-site process snapshot."""
    names = ("pauli_z",) if family == "dephasing" else _PAULI_PROCESS_NAMES
    return tuple((name, (site,), strength) for site in sites for name in names)


def _two_site_processes(
    family: str,
    sites: tuple[int, int],
    strength: float,
) -> tuple[tuple[str, tuple[int, ...], float], ...]:
    """Return the expected strictly two-site process snapshot."""
    names = ("crosstalk_zz",) if family == "dephasing" else _TWO_SITE_PROCESS_NAMES
    return tuple((name, sites, strength) for name in names)


def _expected_process_snapshot(
    noise_id: str,
    sites: tuple[int, ...],
) -> tuple[tuple[str, tuple[int, ...], float], ...]:
    """Return the exact expected model for one identifier and gate support."""
    family, site_support, gate_placement = _EXPECTED_DEFINITIONS[noise_id]
    if len(sites) == 1:
        if gate_placement == "multi_qubit_gates":
            return ()
        return _single_site_processes(family, sites, STANDARD_ONE_QUBIT_GATE_STRENGTH)

    if gate_placement == "single_qubit_gates":
        return ()
    pair = cast("tuple[int, int]", sites)
    processes: tuple[tuple[str, tuple[int, ...], float], ...] = ()
    if site_support in {"single_site", "single_site_and_two_site"}:
        processes += _single_site_processes(family, sites, STANDARD_TWO_QUBIT_GATE_STRENGTH)
    if site_support in {"two_site", "single_site_and_two_site"}:
        processes += _two_site_processes(family, pair, STANDARD_TWO_QUBIT_GATE_STRENGTH)
    return processes


def _instruction_snapshot(
    instruction: GateNoiseInstruction | None,
) -> tuple[tuple[str, tuple[int, ...], float], ...]:
    """Return names, physical sites, and strengths from one provider result."""
    if instruction is None:
        return ()
    assert isinstance(instruction, TJMNoiseInstruction)
    return tuple(
        (
            str(process["name"]),
            tuple(int(site) for site in process["sites"]),
            float(process["strength"]),
        )
        for process in instruction.noise_model.processes
    )


def _relative_processes(
    processes: tuple[tuple[str, tuple[int, ...], float], ...],
    gate_sites: tuple[int, ...],
) -> list[dict[str, object]]:
    """Return JSON-native relative-site templates for serialization checks."""
    site_to_relative = {site: index for index, site in enumerate(gate_sites)}
    return [
        {
            "name": name,
            "relative_sites": [site_to_relative[site] for site in sites],
            "strength": strength,
        }
        for name, sites, strength in processes
    ]


def _expected_definition_dict(noise_id: str) -> dict[str, object]:
    """Return the exact serialized canonical definition."""
    family, site_support, gate_placement = _EXPECTED_DEFINITIONS[noise_id]
    return {
        "noise_id": noise_id,
        "family": family,
        "site_support": site_support,
        "gate_placement": gate_placement,
        "strength_interpretation": STANDARD_NOISE_STRENGTH_INTERPRETATION,
        "one_qubit_gate_processes": _relative_processes(
            _expected_process_snapshot(noise_id, (7,)),
            (7,),
        ),
        "two_qubit_gate_processes": _relative_processes(
            _expected_process_snapshot(noise_id, (7, 9)),
            (7, 9),
        ),
    }


def _alternating_circuit() -> ParameterizedCircuit:
    """Return a circuit alternating one- and two-qubit gates."""
    return ParameterizedCircuit(
        3,
        [
            ParameterizedGate("h", (0,), logical_gate_id="logical-0", native_gate_id="native-0"),
            ParameterizedGate(
                "rzz",
                (0, 2),
                angle_offset=0.41,
                logical_gate_id="logical-1",
                native_gate_id="native-1",
            ),
            ParameterizedGate(
                "ry",
                (2,),
                angle_offset=-0.27,
                logical_gate_id="logical-2",
                native_gate_id="native-2",
            ),
            ParameterizedGate("cz", (1, 2), logical_gate_id="logical-3", native_gate_id="native-3"),
        ],
    )


def _forward_with_provider(
    circuit: ParameterizedCircuit,
    provider: StandardNoiseProvider,
    rng: object,
    *,
    global_noise: NoiseModel | None = None,
) -> tuple[list[KrotovNoiseMap], object]:
    """Run one exact small trajectory.

    Returns:
        The realized noise maps and final state.
    """
    trajectory = forward_tjm_trajectory(
        circuit,
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        MPS(circuit.num_qubits),
        KrotovTruncation(),
        global_noise,
        KrotovTJMOptions(dt=1.0),
        _as_generator(rng),
        noise_provider=provider,
    )
    return trajectory.noise_maps, trajectory.states[-1]


def test_standard_noise_constants_freeze_the_benchmark_convention() -> None:
    """Public constants should expose the fixed strengths and correlated set."""
    assert pytest.approx(6.4e-4) == STANDARD_ONE_QUBIT_GATE_STRENGTH
    assert pytest.approx(5.1e-3) == STANDARD_TWO_QUBIT_GATE_STRENGTH
    assert STANDARD_NOISE_STRENGTH_INTERPRETATION == "per_jump_operator"
    assert TWO_SITE_DEPOLARIZING_OPERATORS == (
        "XX",
        "XY",
        "XZ",
        "YX",
        "YY",
        "YZ",
        "ZX",
        "ZY",
        "ZZ",
    )


def test_registry_contains_exactly_the_ten_frozen_definitions_and_is_immutable() -> None:
    """The standard registry excludes noiseless and Ballarin configurations."""
    assert tuple(STANDARD_NOISE_REGISTRY) == STANDARD_NOISE_IDS
    assert tuple(STANDARD_NOISE_REGISTRY) == tuple(_EXPECTED_DEFINITIONS)
    assert len({id(definition) for definition in STANDARD_NOISE_REGISTRY.values()}) == 10
    mutable_view = cast("dict[str, StandardNoiseDefinition]", STANDARD_NOISE_REGISTRY)
    with pytest.raises(TypeError):
        mutable_view["other"] = next(iter(STANDARD_NOISE_REGISTRY.values()))


@pytest.mark.parametrize("noise_id", STANDARD_NOISE_IDS)
def test_definition_lookup_and_axes_are_exact(noise_id: str) -> None:
    """Every identifier resolves to its immutable canonical axis combination."""
    definition = get_standard_noise_definition(noise_id)
    family, site_support, gate_placement = _EXPECTED_DEFINITIONS[noise_id]

    assert definition is STANDARD_NOISE_REGISTRY[noise_id]
    assert isinstance(definition, StandardNoiseDefinition)
    assert definition.noise_id == noise_id
    assert definition.family == family
    assert definition.site_support == site_support
    assert definition.gate_placement == gate_placement
    with pytest.raises(FrozenInstanceError):
        definition.noise_id = "changed"  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize("noise_id", STANDARD_NOISE_IDS)
@pytest.mark.parametrize("arity", [1, 2])
def test_public_process_templates_match_exact_relative_models(noise_id: str, arity: int) -> None:
    """Public templates should expose canonical relative supports and ordering."""
    definition = get_standard_noise_definition(noise_id)
    relative_sites = tuple(range(arity))

    assert definition.process_templates(arity) == _expected_process_snapshot(noise_id, relative_sites)
    assert definition.process_templates(cast("Any", np.int64(arity))) == definition.process_templates(arity)


@pytest.mark.parametrize(
    ("arity", "error"),
    [
        (True, TypeError),
        (np.bool_(0), TypeError),
        (1.0, TypeError),
        ("1", TypeError),
        (0, ValueError),
        (3, ValueError),
        (-1, ValueError),
    ],
)
def test_public_process_templates_reject_invalid_arities(
    arity: object,
    error: type[Exception],
) -> None:
    """Template lookup should reject ambiguous types and unsupported arities."""
    definition = get_standard_noise_definition("depolarizing_1s2s_all")

    with pytest.raises(error, match="arit"):
        definition.process_templates(cast("Any", arity))


@pytest.mark.parametrize("noise_id", STANDARD_NOISE_IDS)
def test_definition_serialization_is_exact_json_native_and_detached(noise_id: str) -> None:
    """Canonical definitions survive strict JSON round trips without aliases."""
    definition = get_standard_noise_definition(noise_id)
    serialized = definition.to_dict()

    assert serialized == _expected_definition_dict(noise_id)
    assert json.loads(json.dumps(serialized, sort_keys=True)) == serialized
    assert StandardNoiseDefinition.from_dict(json.loads(json.dumps(serialized))) == definition

    one_qubit_processes = cast("list[dict[str, object]]", serialized["one_qubit_gate_processes"])
    one_qubit_processes.append({"name": "bad", "relative_sites": [0], "strength": 1.0})
    assert definition.to_dict() == _expected_definition_dict(noise_id)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"extra": True}, "keys"),
        ({"family": "other"}, "axes"),
        ({"site_support": "two_site"}, "axes"),
        ({"gate_placement": "two_qubit"}, "axes"),
        ({"strength_interpretation": "per-channel"}, "does not match"),
        ({"one_qubit_gate_processes": []}, "does not match"),
        ({"two_qubit_gate_processes": []}, "does not match"),
    ],
)
def test_definition_from_dict_rejects_noncanonical_payloads(
    mutation: dict[str, object],
    match: str,
) -> None:
    """Serialized definitions cannot redefine a frozen registry entry."""
    serialized = get_standard_noise_definition("depolarizing_1s2s_all").to_dict()
    serialized.update(mutation)

    with pytest.raises((TypeError, ValueError), match=match):
        StandardNoiseDefinition.from_dict(serialized)


def test_definition_from_dict_rejects_json_scalar_type_coercion() -> None:
    """Boolean relative sites must not compare equal to integer registry sites."""
    serialized = get_standard_noise_definition("depolarizing_1s2s_all").to_dict()
    processes = cast("list[dict[str, object]]", serialized["one_qubit_gate_processes"])
    processes[0]["relative_sites"] = [False]

    with pytest.raises(ValueError, match="does not match"):
        StandardNoiseDefinition.from_dict(serialized)

    serialized = get_standard_noise_definition("depolarizing_1s2s_all").to_dict()
    processes = cast("list[dict[str, object]]", serialized["one_qubit_gate_processes"])
    processes[0]["extra"] = None

    with pytest.raises(ValueError, match="does not match"):
        StandardNoiseDefinition.from_dict(serialized)


def test_definition_public_api_rejects_invalid_types() -> None:
    """Registry construction and lookup should reject malformed public inputs."""
    with pytest.raises(TypeError, match="noise_id"):
        get_standard_noise_definition(cast("Any", None))
    with pytest.raises(TypeError, match="noise_id"):
        create_standard_noise_provider(cast("Any", 1))
    with pytest.raises(TypeError, match="mapping"):
        StandardNoiseDefinition.from_dict(cast("Any", []))
    with pytest.raises(TypeError, match="family"):
        StandardNoiseDefinition(
            "dephasing_1s_1q",
            cast("Any", None),
            "single_site",
            "single_qubit_gates",
        )


@pytest.mark.parametrize(
    ("noise_id", "family", "site_support", "gate_placement"),
    [
        ("unknown", "dephasing", "single_site", "single_qubit_gates"),
        ("dephasing_1s_1q", "depolarizing", "single_site", "single_qubit_gates"),
        ("dephasing_1s_1q", "dephasing", "two_site", "single_qubit_gates"),
        ("dephasing_1s_1q", "dephasing", "single_site", "all_gates"),
    ],
)
def test_definition_constructor_rejects_unknown_or_inconsistent_axes(
    noise_id: str,
    family: str,
    site_support: str,
    gate_placement: str,
) -> None:
    """A definition cannot assign new semantics to an identifier."""
    with pytest.raises((TypeError, ValueError)):
        StandardNoiseDefinition(
            noise_id,
            cast("Any", family),
            cast("Any", site_support),
            cast("Any", gate_placement),
        )


@pytest.mark.parametrize(
    "noise_id",
    [
        NOISELESS_NOISE_ID,
        BALLARIN_NOISE_ID,
        "",
        "Dephasing_1s_1q",
        "dephasing_1s_1q ",
        "dephasing_1s",
        "dephasing_1s_1q,ballarin_coupled",
    ],
)
def test_standard_lookup_and_factory_reject_nonstandard_identifiers(noise_id: str) -> None:
    """The standard API cannot create noiseless, Ballarin, or composite noise."""
    with pytest.raises(ValueError, match="standard noise"):
        get_standard_noise_definition(noise_id)
    with pytest.raises(ValueError, match="standard noise"):
        create_standard_noise_provider(noise_id)


@pytest.mark.parametrize("noise_id", STANDARD_NOISE_IDS)
def test_noise_config_json_roundtrip_preserves_registry_lookup(noise_id: str) -> None:
    """Serialized benchmark configuration resolves back to the same definition."""
    config = NoiseConfig(noise_id, tjm_dt=1.0)
    restored = NoiseConfig.from_dict(json.loads(json.dumps(config.to_dict())))

    assert restored == config
    assert get_standard_noise_definition(restored.noise_id) is STANDARD_NOISE_REGISTRY[noise_id]
    assert create_standard_noise_provider(restored.noise_id).definition is STANDARD_NOISE_REGISTRY[noise_id]


def test_provider_canonicalizes_equivalent_definitions_and_validates_inputs() -> None:
    """Direct providers should retain one canonical immutable definition."""
    copied = StandardNoiseDefinition(
        "dephasing_1s_1q",
        "dephasing",
        "single_site",
        "single_qubit_gates",
    )
    canonical = get_standard_noise_definition(copied.noise_id)
    assert copied == canonical
    assert copied is not canonical

    provider = StandardNoiseProvider(copied)
    assert provider.definition is canonical
    assert provider.noise_id == copied.noise_id
    assert provider.to_dict() == canonical.to_dict()

    with pytest.raises(TypeError, match="definition"):
        StandardNoiseProvider(cast("Any", copied.noise_id))
    with pytest.raises(TypeError, match="context"):
        provider(cast("Any", None), np.random.default_rng(0))


@pytest.mark.parametrize("noise_id", STANDARD_NOISE_IDS)
@pytest.mark.parametrize("context", _CONTEXTS)
def test_provider_builds_exact_fresh_gate_local_processes(
    noise_id: str,
    context: GateNoiseContext,
) -> None:
    """Every registry provider should match its exact gate-local snapshot."""
    provider = create_standard_noise_provider(noise_id)
    instruction = provider(context, _as_generator(_ForbiddenRNG()))
    expected = _expected_process_snapshot(noise_id, context.sites)

    assert isinstance(provider, StandardNoiseProvider)
    assert provider.definition is get_standard_noise_definition(noise_id)
    assert _instruction_snapshot(instruction) == expected
    if not expected:
        assert instruction is None
        return

    assert isinstance(instruction, TJMNoiseInstruction)
    assert not isinstance(instruction, RandomUnitaryInstruction)
    assert instruction.channel_id == noise_id
    assert instruction.noise_model.scheduled_jumps == []
    assert validate_gate_noise_instruction(instruction, context) is instruction
    assert all(set(process["sites"]).issubset(context.sites) for process in instruction.noise_model.processes)
    assert all(type(process["strength"]) is float for process in instruction.noise_model.processes)

    repeated = provider(context, _as_generator(_ForbiddenRNG()))
    assert isinstance(repeated, TJMNoiseInstruction)
    assert repeated is not instruction
    assert repeated.noise_model is not instruction.noise_model
    instruction.noise_model.processes[0]["strength"] = 123.0
    assert _instruction_snapshot(repeated) == expected


def test_fresh_models_do_not_share_mutable_operator_payloads() -> None:
    """Mutating one generated model must not corrupt another or later calls."""
    one_site_provider = create_standard_noise_provider("depolarizing_1s_1q")
    one_site_context = _CONTEXTS[0]
    first_one_site = one_site_provider(one_site_context, np.random.default_rng(1))
    second_one_site = one_site_provider(one_site_context, np.random.default_rng(2))
    assert isinstance(first_one_site, TJMNoiseInstruction)
    assert isinstance(second_one_site, TJMNoiseInstruction)
    first_matrix = cast("NDArray[np.complex128]", first_one_site.noise_model.processes[0]["matrix"])
    second_matrix = cast("NDArray[np.complex128]", second_one_site.noise_model.processes[0]["matrix"])
    assert not np.shares_memory(first_matrix, second_matrix)

    first_matrix[0, 0] = 123.0
    np.testing.assert_array_equal(second_matrix, _PAULI_X)
    later_one_site = one_site_provider(one_site_context, np.random.default_rng(3))
    assert isinstance(later_one_site, TJMNoiseInstruction)
    np.testing.assert_array_equal(later_one_site.noise_model.processes[0]["matrix"], _PAULI_X)

    two_site_provider = create_standard_noise_provider("depolarizing_2s_2q")
    two_site_context = _CONTEXTS[1]
    first_two_site = two_site_provider(two_site_context, np.random.default_rng(4))
    second_two_site = two_site_provider(two_site_context, np.random.default_rng(5))
    assert isinstance(first_two_site, TJMNoiseInstruction)
    assert isinstance(second_two_site, TJMNoiseInstruction)
    first_factors = cast("tuple[NDArray[np.complex128], ...]", first_two_site.noise_model.processes[0]["factors"])
    second_factors = cast(
        "tuple[NDArray[np.complex128], ...]",
        second_two_site.noise_model.processes[0]["factors"],
    )
    assert all(not np.shares_memory(first, second) for first, second in zip(first_factors, second_factors, strict=True))

    first_factors[0][0, 0] = 456.0
    np.testing.assert_array_equal(second_factors[0], _PAULI_X)
    later_two_site = two_site_provider(two_site_context, np.random.default_rng(6))
    assert isinstance(later_two_site, TJMNoiseInstruction)
    later_factors = cast(
        "tuple[NDArray[np.complex128], ...]",
        later_two_site.noise_model.processes[0]["factors"],
    )
    np.testing.assert_array_equal(later_factors[0], _PAULI_X)


@pytest.mark.parametrize("noise_id", STANDARD_NOISE_IDS)
def test_direct_context_counts_indices_and_total_are_exact(noise_id: str) -> None:
    """Alternating contexts should produce the documented placement totals."""
    provider = create_standard_noise_provider(noise_id)
    snapshots = tuple(_instruction_snapshot(provider(context, _as_generator(_ForbiddenRNG()))) for context in _CONTEXTS)
    expected_counts, expected_indices, expected_total = _EXPECTED_COUNTS_AND_INDICES[noise_id]

    assert tuple(len(snapshot) for snapshot in snapshots) == expected_counts
    assert tuple(index for index, snapshot in enumerate(snapshots) if snapshot) == expected_indices
    assert sum(map(len, snapshots)) == expected_total


@pytest.mark.parametrize("noise_id", STANDARD_NOISE_IDS)
def test_provider_depends_on_arity_and_sites_not_name_angle_or_provenance(noise_id: str) -> None:
    """Provider definitions are state-independent and gate-name agnostic."""
    provider = create_standard_noise_provider(noise_id)
    first = GateNoiseContext(0, "h", (1, 4), 2, None, "logical-a", "native-a", None)
    second = GateNoiseContext(91, "unknown_custom_gate", (1, 4), 2, -3.2, 17, 23, 8)

    assert _instruction_snapshot(provider(first, np.random.default_rng(1))) == _instruction_snapshot(
        provider(second, np.random.default_rng(999))
    )


@pytest.mark.parametrize("noise_id", STANDARD_NOISE_IDS)
def test_standard_strengths_do_not_depend_on_tjm_step(noise_id: str) -> None:
    """The registry strength is per jump operator and is never rescaled by dt."""
    short_step = NoiseConfig(noise_id, tjm_dt=0.25)
    long_step = NoiseConfig(noise_id, tjm_dt=2.0)
    provider = create_standard_noise_provider(noise_id)

    short_snapshot = _instruction_snapshot(provider(_CONTEXTS[1], np.random.default_rng(2)))
    long_snapshot = _instruction_snapshot(provider(_CONTEXTS[1], np.random.default_rng(3)))

    assert short_step.tjm_dt != long_step.tjm_dt
    assert short_snapshot == long_snapshot == _expected_process_snapshot(noise_id, _CONTEXTS[1].sites)


@pytest.mark.parametrize("noise_id", ["dephasing_2s_2q", "depolarizing_2s_2q"])
def test_two_site_payloads_use_factors_nonadjacently_and_matrices_adjacently(noise_id: str) -> None:
    """Two-site process payloads follow YAQS local ordering on both supports."""
    provider = create_standard_noise_provider(noise_id)
    nonadjacent = provider(_CONTEXTS[1], np.random.default_rng(1))
    adjacent = provider(_CONTEXTS[3], np.random.default_rng(1))
    assert isinstance(nonadjacent, TJMNoiseInstruction)
    assert isinstance(adjacent, TJMNoiseInstruction)

    for process in nonadjacent.noise_model.processes:
        suffix = str(process["name"]).removeprefix("crosstalk_")
        assert "factors" in process
        assert "matrix" not in process
        factors = process["factors"]
        np.testing.assert_array_equal(factors[0], _PAULI_MATRICES[suffix[0]])
        np.testing.assert_array_equal(factors[1], _PAULI_MATRICES[suffix[1]])

    for process in adjacent.noise_model.processes:
        suffix = str(process["name"]).removeprefix("crosstalk_")
        assert "matrix" in process
        np.testing.assert_array_equal(
            process["matrix"],
            np.kron(_PAULI_MATRICES[suffix[0]], _PAULI_MATRICES[suffix[1]]),
        )


def test_support_regressions_distinguish_local_correlated_and_combined_models() -> None:
    """The support axes must not collapse into one another."""
    context = _CONTEXTS[1]
    local_dephasing = _instruction_snapshot(
        create_standard_noise_provider("dephasing_1s_2q")(context, np.random.default_rng(0))
    )
    correlated_dephasing = _instruction_snapshot(
        create_standard_noise_provider("dephasing_2s_2q")(context, np.random.default_rng(0))
    )
    combined_depolarizing = _instruction_snapshot(
        create_standard_noise_provider("depolarizing_1s2s_all")(context, np.random.default_rng(0))
    )

    assert [name for name, _, _ in local_dephasing] == ["pauli_z", "pauli_z"]
    assert [name for name, _, _ in correlated_dephasing] == ["crosstalk_zz"]
    assert [name for name, _, _ in combined_depolarizing[:6]] == [
        "pauli_x",
        "pauli_y",
        "pauli_z",
        "pauli_x",
        "pauli_y",
        "pauli_z",
    ]
    assert [name for name, _, _ in combined_depolarizing[6:]] == list(_TWO_SITE_PROCESS_NAMES)
    assert all(
        len(name.removeprefix("crosstalk_")) == 2 and set(name.removeprefix("crosstalk_")).issubset({"x", "y", "z"})
        for name in _TWO_SITE_PROCESS_NAMES
    )


@pytest.mark.parametrize("noise_id", STANDARD_NOISE_IDS)
def test_alternating_circuit_integration_has_exact_diagnostic_indices(noise_id: str) -> None:
    """WP3 integration should tag exactly the gates selected by each definition."""
    rng = _NoJumpRNG()
    maps, _final_state = _forward_with_provider(
        _alternating_circuit(),
        create_standard_noise_provider(noise_id),
        rng,
    )
    _counts, expected_indices, _total = _EXPECTED_COUNTS_AND_INDICES[noise_id]
    active_indices = tuple(index for index, noise_map in enumerate(maps) if noise_map.channel_id is not None)
    expected_angles = (None, 0.41, -0.27, None)

    assert active_indices == expected_indices
    assert rng.random_calls == len(expected_indices)
    assert rng.choice_calls == 0
    for index, noise_map in enumerate(maps):
        assert noise_map.source_gate_index == index
        assert noise_map.resolved_native_angle == expected_angles[index]
        assert noise_map.is_identity is True
        if index in expected_indices:
            assert noise_map.channel_id == noise_id
            assert noise_map.outcome_labels == ("no_jump",)
            assert noise_map.normalized is True
        else:
            assert noise_map.channel_id is None
            assert noise_map.outcome_labels == ()


@pytest.mark.parametrize(
    "noise_id",
    ["dephasing_2s_2q", "depolarizing_1s2s_all"],
)
def test_representative_nonadjacent_forced_jump_has_expected_diagnostics(noise_id: str) -> None:
    """A forced jump should realize the last correlated process on gate support."""
    circuit = ParameterizedCircuit(
        3,
        [ParameterizedGate("rzz", (0, 2), angle_offset=0.2)],
    )
    rng = _LastJumpRNG()
    maps, _final_state = _forward_with_provider(circuit, create_standard_noise_provider(noise_id), rng)

    assert rng.random_calls == 1
    assert rng.choice_calls == 1
    assert len(maps) == 1
    assert maps[0].channel_id == noise_id
    assert maps[0].outcome_labels == ("crosstalk_zz",)
    assert len(maps[0].operators) == 1
    matrix, sites = maps[0].operators[0]
    assert sites == (0, 2)
    np.testing.assert_array_equal(matrix, np.kron(_PAULI_Z, _PAULI_Z))


def test_standard_provider_conflicts_with_a_global_noise_model() -> None:
    """A standard provider cannot accidentally compose with global common noise."""
    global_noise = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 0.1}])

    with pytest.raises(ValueError, match=r"global noise_model.*noise_provider"):
        _forward_with_provider(
            ParameterizedCircuit(1, [ParameterizedGate("h", (0,))]),
            create_standard_noise_provider("dephasing_1s_1q"),
            np.random.default_rng(1),
            global_noise=global_noise,
        )
