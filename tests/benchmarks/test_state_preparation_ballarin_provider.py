# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Ballarin formulas and the gate-local product-Pauli provider."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

import benchmarks.state_preparation.ballarin as ballarin_module
from benchmarks.state_preparation import (
    BALLARIN_EPSILON_INTERCEPT,
    BALLARIN_EPSILON_SLOPE,
    BALLARIN_MAX_EPSILON,
    BALLARIN_NOISE_ID,
    BALLARIN_PRUNING_THRESHOLD,
    BallarinNoiseProvider,
    ballarin_epsilon,
    ballarin_local_pauli_probability,
    ballarin_local_pauli_rate,
    canonicalize_native_rzz_angle,
    canonicalize_rzz_angle,
    compile_quantinuum_native,
    create_ballarin_noise_provider,
    materialize_ballarin_circuit,
)
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import (
    GateNoiseContext,
    KrotovTJMOptions,
    KrotovTruncation,
    LocalOperator,
    ParameterizedCircuit,
    ParameterizedGate,
    RandomUnitaryInstruction,
)
from mqt.yaqs.optimization.krotov import forward_tjm_trajectory

if TYPE_CHECKING:
    from benchmarks.state_preparation.noise import PauliDistribution


class _SequenceRNG:
    """Minimal scalar RNG double with an observable draw count."""

    def __init__(self, values: tuple[float, ...]) -> None:
        self._values = values
        self.calls = 0

    def random(self) -> float:
        """Return the next fixed uniform draw.

        Returns:
            The next configured scalar.

        Raises:
            AssertionError: If the configured sequence has been exhausted.
        """
        if self.calls >= len(self._values):
            msg = "Test RNG exhausted."
            raise AssertionError(msg)
        value = self._values[self.calls]
        self.calls += 1
        return value


def _as_generator(rng: _SequenceRNG) -> np.random.Generator:
    """Treat a scalar test double as the provider's generator protocol.

    Returns:
        The typed test generator.
    """
    return cast("np.random.Generator", rng)


def _context(**overrides: object) -> GateNoiseContext:
    """Build one valid native-RZZ context with selected overrides.

    Returns:
        The immutable provider context.
    """
    values: dict[str, object] = {
        "gate_index": 4,
        "gate_name": "rzz",
        "sites": (0, 2),
        "arity": 2,
        "resolved_angle": 0.4,
        "logical_gate_id": "logical-rzz",
        "native_gate_id": 9,
        "parameter_index": None,
    }
    values.update(overrides)
    return GateNoiseContext(**cast("Any", values))


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (0.0, 0.0),
        (-0.0, 0.0),
        (math.pi, -math.pi),
        (-math.pi, -math.pi),
        (3.0 * math.pi, -math.pi),
        (-3.0 * math.pi, -math.pi),
        (2.0 * math.pi, 0.0),
        (-2.0 * math.pi, 0.0),
    ],
)
def test_canonical_angles_use_the_frozen_half_open_interval(angle: float, expected: float) -> None:
    """Exact turns and half turns should have deterministic representatives."""
    canonical = canonicalize_rzz_angle(angle)

    assert canonical == pytest.approx(expected, abs=1e-15)
    assert -math.pi <= canonical < math.pi
    assert canonicalize_native_rzz_angle(angle) == canonical


def test_canonicalization_preserves_values_immediately_inside_each_boundary() -> None:
    """Angles on either side of the branch cut should map to the correct side."""
    below_positive_pi = math.nextafter(math.pi, -math.inf)
    below_negative_pi = math.nextafter(-math.pi, -math.inf)

    assert canonicalize_rzz_angle(below_positive_pi) == below_positive_pi
    wrapped = canonicalize_rzz_angle(below_negative_pi)
    assert 0.0 < wrapped < math.pi
    assert wrapped == pytest.approx(math.pi, abs=1e-15)


@pytest.mark.parametrize("value", [True, np.bool_(0), "0.2", 0.2 + 0.0j])
def test_canonicalization_rejects_non_real_scalar_inputs(value: object) -> None:
    """Boolean, textual, and complex angles should not be coerced."""
    with pytest.raises(TypeError, match="finite real"):
        canonicalize_rzz_angle(cast("Any", value))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_canonicalization_rejects_nonfinite_angles(value: float) -> None:
    """Nonfinite angles cannot define a physical native rotation."""
    with pytest.raises(ValueError, match="finite"):
        canonicalize_rzz_angle(value)


@pytest.mark.parametrize(
    "magnitude",
    [
        0.0,
        BALLARIN_PRUNING_THRESHOLD,
        0.5,
        1.0,
        math.pi,
    ],
)
def test_ballarin_formula_values_and_consistency_identity(magnitude: float) -> None:
    """The public helpers should implement the frozen epsilon and rate equations."""
    expected_epsilon = BALLARIN_EPSILON_INTERCEPT + BALLARIN_EPSILON_SLOPE * magnitude
    expected_rate = (1.0 - math.sqrt(1.0 - 1.25 * expected_epsilon)) / 3.0

    epsilon = ballarin_epsilon(magnitude)
    rate = ballarin_local_pauli_rate(magnitude)

    assert epsilon == pytest.approx(expected_epsilon, rel=1e-15, abs=0.0)
    assert rate == pytest.approx(expected_rate, rel=1e-14, abs=1e-16)
    assert ballarin_local_pauli_probability(magnitude) == rate
    assert 0.0 <= rate <= 1.0 / 3.0
    assert epsilon == pytest.approx(
        (4.0 / 5.0) * (1.0 - (1.0 - 3.0 * rate) ** 2),
        rel=1e-12,
        abs=2e-16,
    )


def test_ballarin_rate_validates_the_square_root_domain() -> None:
    """The closed domain endpoint should work while a larger fit should fail."""
    maximum_magnitude = (BALLARIN_MAX_EPSILON - BALLARIN_EPSILON_INTERCEPT) / BALLARIN_EPSILON_SLOPE

    assert ballarin_epsilon(maximum_magnitude) == pytest.approx(BALLARIN_MAX_EPSILON)
    assert ballarin_local_pauli_rate(maximum_magnitude) == pytest.approx(1.0 / 3.0)
    with pytest.raises(ValueError, match="probability domain"):
        ballarin_local_pauli_rate(maximum_magnitude + 1e-6)


@pytest.mark.parametrize("magnitude", [-1e-12, -1.0])
def test_ballarin_formulas_reject_negative_magnitudes(magnitude: float) -> None:
    """A signed angle must be canonicalized and made absolute before formula use."""
    with pytest.raises(ValueError, match="nonnegative"):
        ballarin_epsilon(magnitude)


def test_provider_factory_and_serialization_are_fixed_and_strict() -> None:
    """The factory and codec should expose one immutable benchmark definition."""
    provider = create_ballarin_noise_provider()
    expected = {
        "noise_id": BALLARIN_NOISE_ID,
        "gate_name": "rzz",
        "gate_placement": "post_gate",
        "angle_convention": "canonical_magnitude",
        "canonical_interval": "[-pi, pi)",
        "pruning_threshold": BALLARIN_PRUNING_THRESHOLD,
        "epsilon_intercept": BALLARIN_EPSILON_INTERCEPT,
        "epsilon_slope": BALLARIN_EPSILON_SLOPE,
        "channel": "independent_product_pauli",
        "single_qubit_gates": "noiseless",
    }

    assert isinstance(provider, BallarinNoiseProvider)
    assert provider.noise_id == BALLARIN_NOISE_ID
    assert provider.to_dict() == expected
    restored = BallarinNoiseProvider.from_dict(provider.to_dict())
    assert isinstance(restored, BallarinNoiseProvider)
    assert restored.to_dict() == expected

    wrong_type = provider.to_dict()
    wrong_type["pruning_threshold"] = np.float64(BALLARIN_PRUNING_THRESHOLD)
    with pytest.raises(ValueError, match="pruning_threshold"):
        BallarinNoiseProvider.from_dict(wrong_type)

    missing = provider.to_dict()
    del missing["channel"]
    with pytest.raises(ValueError, match="keys mismatch"):
        BallarinNoiseProvider.from_dict(missing)

    mixed_keys = cast("dict[str, object]", {**provider.to_dict(), 7: "unexpected"})
    with pytest.raises(ValueError, match="keys mismatch"):
        BallarinNoiseProvider.from_dict(mixed_keys)

    with pytest.raises(TypeError, match="mapping"):
        BallarinNoiseProvider.from_dict(cast("Any", []))


@pytest.mark.parametrize(
    "angle",
    [
        0.0,
        math.nextafter(BALLARIN_PRUNING_THRESHOLD, 0.0),
        BALLARIN_PRUNING_THRESHOLD,
        -BALLARIN_PRUNING_THRESHOLD,
        2.0 * math.pi,
    ],
)
def test_provider_skips_angles_at_or_below_the_inclusive_threshold(angle: float) -> None:
    """A rotation assigned to pruning must not consume trajectory randomness."""
    rng = _SequenceRNG(())

    instruction = create_ballarin_noise_provider()(_context(resolved_angle=angle), _as_generator(rng))

    assert instruction is None
    assert rng.calls == 0


def test_provider_calls_product_sampler_once_and_records_two_identity_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One retained rotation should use two local draws and retain both identity outcomes."""
    original_sampler = ballarin_module.sample_product_pauli_channel
    calls: list[tuple[int, int, PauliDistribution, PauliDistribution, np.random.Generator]] = []

    def recording_sampler(
        first_site: int,
        second_site: int,
        first_distribution: PauliDistribution,
        second_distribution: PauliDistribution,
        rng: np.random.Generator,
    ) -> tuple[LocalOperator, ...]:
        calls.append((first_site, second_site, first_distribution, second_distribution, rng))
        return original_sampler(
            first_site,
            second_site,
            first_distribution,
            second_distribution,
            rng,
        )

    monkeypatch.setattr(ballarin_module, "sample_product_pauli_channel", recording_sampler)
    rng = _SequenceRNG((0.0, 0.0))
    typed_rng = _as_generator(rng)
    angle = math.nextafter(BALLARIN_PRUNING_THRESHOLD, math.inf)

    instruction = create_ballarin_noise_provider()(_context(resolved_angle=angle), typed_rng)

    assert isinstance(instruction, RandomUnitaryInstruction)
    assert instruction.channel_id == BALLARIN_NOISE_ID
    assert instruction.operators == ()
    assert instruction.outcome_labels == ("I", "I")
    assert len(calls) == 1
    first_site, second_site, first_distribution, second_distribution, supplied_rng = calls[0]
    assert (first_site, second_site) == (0, 2)
    assert first_distribution is second_distribution
    assert supplied_rng is typed_rng
    rate = ballarin_local_pauli_rate(abs(canonicalize_rzz_angle(angle)))
    assert dict(first_distribution) == pytest.approx({
        "I": 1.0 - 3.0 * rate,
        "X": rate,
        "Y": rate,
        "Z": rate,
    })
    assert rng.calls == 2


def test_provider_records_nonidentity_outcomes_in_site_order() -> None:
    """Sampled local labels and operators should retain independent site order."""
    angle = 1.0
    rate = ballarin_local_pauli_rate(angle)
    identity_probability = 1.0 - 3.0 * rate
    rng = _SequenceRNG((
        identity_probability + 0.5 * rate,
        identity_probability + 2.5 * rate,
    ))

    instruction = create_ballarin_noise_provider()(_context(resolved_angle=angle), _as_generator(rng))

    assert isinstance(instruction, RandomUnitaryInstruction)
    assert instruction.outcome_labels == ("X", "Z")
    assert tuple(operator.label for operator in instruction.operators) == ("X", "Z")
    assert tuple(operator.sites for operator in instruction.operators) == ((0,), (2,))
    assert rng.calls == 2


def test_provider_uses_equal_strength_for_opposite_signed_angles() -> None:
    """The canonical-magnitude convention should erase only the noise sign."""
    provider = create_ballarin_noise_provider()

    positive = provider(_context(resolved_angle=0.7), np.random.default_rng(81))
    negative = provider(_context(resolved_angle=-0.7), np.random.default_rng(81))

    assert positive == negative


def test_non_entangling_context_is_ignored_without_drawing() -> None:
    """A defensive one-qubit request should be a zero-cost no-op."""
    rng = _SequenceRNG(())
    context = _context(
        gate_name="h",
        sites=(1,),
        arity=1,
        resolved_angle=None,
        native_gate_id=3,
    )

    assert create_ballarin_noise_provider()(context, _as_generator(rng)) is None
    assert rng.calls == 0


@pytest.mark.parametrize(
    "context",
    [
        _context(gate_name="cx"),
        _context(gate_name="rxx"),
        _context(gate_name="rzz", sites=(1,), arity=1),
        _context(resolved_angle=None),
    ],
    ids=("non-native-two-qubit", "logical-entangler", "one-site-rzz", "unresolved-rzz"),
)
def test_provider_rejects_invalid_native_contexts_before_drawing(context: GateNoiseContext) -> None:
    """Malformed native requests must fail without changing the random stream."""
    rng = _SequenceRNG(())

    with pytest.raises(ValueError, match=r"native RZZ|resolved"):
        create_ballarin_noise_provider()(context, _as_generator(rng))

    assert rng.calls == 0


def test_provider_rejects_non_context_objects_before_drawing() -> None:
    """The callable boundary should reject arbitrary provider payloads."""
    rng = _SequenceRNG(())

    with pytest.raises(TypeError, match="GateNoiseContext"):
        create_ballarin_noise_provider()(cast("Any", object()), _as_generator(rng))

    assert rng.calls == 0


def test_materialized_circuit_invokes_provider_once_per_retained_native_rotation() -> None:
    """Only retained central RZZ gates should sample noise in final evaluation."""
    logical = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("ry", (0,), angle_offset=0.2, logical_gate_id="one-qubit"),
            ParameterizedGate("rxx", (1, 0), param_index=0, logical_gate_id="logical-x"),
            ParameterizedGate("rzz", (0, 1), param_index=1, logical_gate_id="logical-pruned"),
            ParameterizedGate("ryy", (0, 1), param_index=2, logical_gate_id="logical-y"),
        ],
        num_params=3,
    )
    compilation = compile_quantinuum_native(logical)
    theta = np.array(
        [
            2.0 * math.pi + 0.35,
            BALLARIN_PRUNING_THRESHOLD,
            -2.0 * math.pi - 0.4,
        ],
        dtype=np.float64,
    )
    materialization = materialize_ballarin_circuit(compilation, theta)
    provider = create_ballarin_noise_provider()
    contexts: list[GateNoiseContext] = []

    def tracking_provider(
        context: GateNoiseContext,
        rng: np.random.Generator,
    ) -> RandomUnitaryInstruction | None:
        contexts.append(context)
        return provider(context, rng)

    rng = _SequenceRNG((0.0, 0.0, 0.0, 0.0))
    trajectory = forward_tjm_trajectory(
        materialization.circuit,
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        MPS(2),
        KrotovTruncation(),
        None,
        KrotovTJMOptions(apply_noise_to="all"),
        _as_generator(rng),
        noise_provider=tracking_provider,
    )

    retained_records = (materialization.mapping[1], materialization.mapping[3])
    assert materialization.pruned_native_rzz_count == 1
    assert materialization.mapping[2].rotation_pruned is True
    assert materialization.mapping[2].final_native_rotation_gate_index is None
    assert materialization.retained_native_rzz_count == len(retained_records)
    assert [context.gate_index for context in contexts] == [
        cast("int", record.final_native_rotation_gate_index) for record in retained_records
    ]
    assert [context.gate_name for context in contexts] == ["rzz", "rzz"]
    assert [context.sites for context in contexts] == [(0, 1), (0, 1)]
    assert [context.logical_gate_id for context in contexts] == ["logical-x", "logical-y"]
    assert [context.native_gate_id for context in contexts] == [
        record.native_rotation_gate_id for record in retained_records
    ]
    assert [context.resolved_angle for context in contexts] == pytest.approx([0.35, -0.4])
    assert rng.calls == 2 * len(retained_records)

    noisy_indices = {cast("int", record.final_native_rotation_gate_index) for record in retained_records}
    assert {
        index for index, noise_map in enumerate(trajectory.noise_maps) if noise_map.channel_id == BALLARIN_NOISE_ID
    } == noisy_indices
    for index in noisy_indices:
        noise_map = trajectory.noise_maps[index]
        assert noise_map.outcome_labels == ("I", "I")
        assert noise_map.operators == ()
        assert noise_map.is_identity is True
        assert noise_map.resolved_native_angle == pytest.approx(materialization.circuit.gates[index].angle_offset)
