# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for gate-local noise-provider records and validation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.optimization.gate_noise import (
    CompositeGateNoiseInstruction,
    GateNoiseContext,
    GateNoiseInstruction,
    GateNoiseProvider,
    LocalOperator,
    RandomUnitaryInstruction,
    TJMNoiseInstruction,
    validate_gate_noise_instruction,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_ZZ = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.complex128)


def _context(**overrides: object) -> GateNoiseContext:
    values: dict[str, object] = {
        "gate_index": 3,
        "gate_name": "rzz",
        "sites": (1, 4),
        "arity": 2,
        "resolved_angle": 0.25,
        "logical_gate_id": 2,
        "native_gate_id": "native-7",
        "parameter_index": 5,
    }
    values.update(overrides)
    return GateNoiseContext(**cast("Any", values))


def _model(
    *,
    sites: list[int] | tuple[int, ...] = (1,),
    strength: object = 0.1,
) -> NoiseModel:
    return NoiseModel([{"name": "pauli_x", "sites": sites, "strength": strength}])


def test_gate_noise_context_preserves_complete_metadata_and_is_frozen() -> None:
    """Gate context should expose normalized immutable metadata."""
    context = _context(
        gate_index=np.int64(3),
        sites=[np.int64(1), np.int64(4)],
        arity=np.int64(2),
        resolved_angle=np.float64(0.25),
        logical_gate_id=np.int64(2),
        parameter_index=np.int64(5),
    )

    assert context == GateNoiseContext(3, "rzz", (1, 4), 2, 0.25, 2, "native-7", 5)
    assert isinstance(context.gate_index, int)
    assert context.sites == (1, 4)
    with pytest.raises(FrozenInstanceError):
        context.gate_index = 4  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    ("field", "value", "error", "match"),
    [
        ("gate_index", True, TypeError, "gate_index"),
        ("gate_index", -1, ValueError, "nonnegative"),
        ("gate_name", 1, TypeError, "gate_name"),
        ("gate_name", " rzz", ValueError, "gate_name"),
        ("sites", (), ValueError, "one or two"),
        ("sites", (1, 1), ValueError, "duplicate"),
        ("sites", (4, 1), ValueError, "ascending"),
        ("sites", (True,), TypeError, "Boolean"),
        ("sites", (-1,), ValueError, "nonnegative"),
        ("arity", True, TypeError, "arity"),
        ("arity", 1, ValueError, "does not match"),
        ("resolved_angle", True, TypeError, "resolved_angle"),
        ("resolved_angle", np.inf, ValueError, "finite"),
        ("logical_gate_id", True, TypeError, "logical_gate_id"),
        ("logical_gate_id", -1, ValueError, "nonnegative"),
        ("logical_gate_id", "", ValueError, "nonempty"),
        ("native_gate_id", object(), TypeError, "native_gate_id"),
        ("native_gate_id", " native", ValueError, "surrounding"),
        ("parameter_index", True, TypeError, "parameter_index"),
        ("parameter_index", -1, ValueError, "nonnegative"),
    ],
)
def test_gate_noise_context_rejects_invalid_metadata(
    field: str,
    value: object,
    error: type[Exception],
    match: str,
) -> None:
    """Every context field should reject ambiguous or inconsistent values."""
    with pytest.raises(error, match=match):
        _context(**{field: value})


def test_gate_noise_context_supports_fixed_gates_and_string_identifiers() -> None:
    """Non-parametric gates should use ``None`` for angle and parameter index."""
    context = _context(
        gate_name="x",
        sites=(0,),
        arity=1,
        resolved_angle=None,
        logical_gate_id="logical-x",
        native_gate_id=0,
        parameter_index=None,
    )

    assert context.resolved_angle is None
    assert context.parameter_index is None


def test_local_operator_defensively_copies_and_irreversibly_freezes_matrix() -> None:
    """A provider cannot mutate a realized operator through its input array."""
    source = _X.copy()
    operator = LocalOperator(source, cast("tuple[int, ...]", [np.int64(1)]), label="X")
    source[0, 0] = 10.0

    np.testing.assert_array_equal(operator.matrix, _X)
    assert operator.matrix.dtype == np.complex128
    assert not operator.matrix.flags["W"]
    with pytest.raises(ValueError, match=r"cannot set .* flag"):
        operator.matrix.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        operator.label = "changed"  # ty: ignore[invalid-assignment]


def test_local_operator_matrix_views_cannot_mutate_stored_metadata() -> None:
    """Shape and dtype changes on one view must not alter the stored operator."""
    operator = LocalOperator(_X, (0,), "X")
    first = operator.matrix
    first.shape = (4,)
    first.dtype = np.float64  # ty: ignore[invalid-assignment]

    second = operator.matrix
    assert second is not first
    assert second.shape == (2, 2)
    assert second.dtype == np.complex128
    np.testing.assert_array_equal(second, _X)


def test_local_operator_has_safe_value_equality() -> None:
    """Local-operator equality should not expose ambiguous NumPy truth values."""
    assert LocalOperator(_X, (0,), "X") == LocalOperator(_X.copy(), (0,), "X")
    assert LocalOperator(_X, (0,), "X") != LocalOperator(_Y, (0,), "Y")
    operator = LocalOperator(_X, (0,), "X")
    equality = operator.__eq__
    assert equality(object()) is NotImplemented


@pytest.mark.parametrize(
    ("matrix", "sites", "label", "error", "match"),
    [
        (np.ones(2), (0,), None, ValueError, "two-dimensional"),
        (np.eye(4), (0,), None, ValueError, "shape"),
        (np.array([[1.0, np.nan], [0.0, 1.0]]), (0,), None, ValueError, "finite"),
        (np.diag([1.0, 0.5]), (0,), None, ValueError, "unitary"),
        (np.eye(2), (), None, ValueError, "one or two"),
        (np.eye(4), (1, 1), None, ValueError, "duplicate"),
        (np.eye(4), (2, 1), None, ValueError, "ascending"),
        (np.eye(2), (True,), None, TypeError, "Boolean"),
        (np.eye(2), (0,), "", ValueError, "nonempty"),
        (np.eye(2), (0,), 1, TypeError, "string or None"),
    ],
)
def test_local_operator_rejects_invalid_unitaries_and_metadata(
    matrix: object,
    sites: object,
    label: object,
    error: type[Exception],
    match: str,
) -> None:
    """Local operators should be finite, unitary, and canonically supported."""
    with pytest.raises(error, match=match):
        LocalOperator(cast("Any", matrix), cast("Any", sites), cast("Any", label))


def test_local_operator_rejects_non_numeric_matrix() -> None:
    """Matrix conversion failures should produce a clear type error."""
    with pytest.raises(TypeError, match="complex matrix"):
        LocalOperator(cast("Any", [["not-a-number"]]), (0,))


def test_instruction_records_validate_and_freeze_diagnostics() -> None:
    """Tagged instructions should validate types and diagnostic labels."""
    model = _model()
    tjm = TJMNoiseInstruction(model, channel_id="dephasing")
    random = RandomUnitaryInstruction(
        (LocalOperator(_X, (1,), "X"), LocalOperator(_Y, (4,), "Y")),
        channel_id="product-pauli",
        outcome_labels=("X", "Y"),
    )
    composite = CompositeGateNoiseInstruction((tjm, random), channel_id="mixed")

    assert tjm.noise_model is model
    assert random.outcome_labels == ("X", "Y")
    assert [operator.label for operator in random.operators] == ["X", "Y"]
    assert composite.instructions == (tjm, random)
    with pytest.raises(FrozenInstanceError):
        random.channel_id = "changed"  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    ("factory", "error", "match"),
    [
        (lambda: TJMNoiseInstruction(cast("Any", object())), TypeError, "NoiseModel"),
        (lambda: TJMNoiseInstruction(_model(), channel_id=""), ValueError, "nonempty"),
        (lambda: RandomUnitaryInstruction(cast("Any", [])), TypeError, "tuple"),
        (lambda: RandomUnitaryInstruction(cast("Any", (object(),))), TypeError, "LocalOperator"),
        (lambda: RandomUnitaryInstruction(channel_id=" bad"), ValueError, "surrounding"),
        (lambda: RandomUnitaryInstruction(outcome_labels=cast("Any", ["I"])), TypeError, "tuple"),
        (lambda: RandomUnitaryInstruction(outcome_labels=("",)), ValueError, "nonempty"),
        (lambda: RandomUnitaryInstruction(outcome_labels=cast("Any", (1,))), TypeError, "string"),
        (lambda: CompositeGateNoiseInstruction(cast("Any", [])), TypeError, "tuple"),
        (lambda: CompositeGateNoiseInstruction(cast("Any", (_model(),))), TypeError, "only TJM"),
        (
            lambda: CompositeGateNoiseInstruction((), channel_id=" mixed"),
            ValueError,
            "surrounding",
        ),
    ],
    ids=(
        "wrong-noise-model",
        "empty-tjm-channel",
        "operator-list",
        "wrong-operator",
        "spaced-random-channel",
        "outcome-label-list",
        "empty-outcome-label",
        "non-string-outcome-label",
        "composite-list",
        "raw-model-composite-child",
        "spaced-composite-channel",
    ),
)
def test_instruction_records_reject_invalid_fields(
    factory: Callable[[], object],
    error: type[Exception],
    match: str,
) -> None:
    """Tagged instruction constructors should reject malformed payloads."""
    with pytest.raises(error, match=match):
        factory()


def test_validate_instruction_accepts_none_and_normalizes_raw_noise_model() -> None:
    """Raw ``NoiseModel`` output should be shorthand for a tagged TJM instruction."""
    context = _context()
    model = _model()

    assert validate_gate_noise_instruction(None, context) is None
    result = validate_gate_noise_instruction(model, context)

    assert isinstance(result, TJMNoiseInstruction)
    assert result.noise_model is model
    assert result.channel_id is None


def test_validate_instruction_preserves_tagged_instructions() -> None:
    """Already-tagged valid provider results should be returned unchanged."""
    context = _context()
    tjm = TJMNoiseInstruction(_model(), "dephasing")
    random = RandomUnitaryInstruction(
        (
            LocalOperator(_X, (1,), "X"),
            LocalOperator(_Y, (4,), "Y"),
        ),
        "product-pauli",
        ("X", "Y"),
    )
    composite = CompositeGateNoiseInstruction((tjm, random), "mixed")

    assert validate_gate_noise_instruction(tjm, context) is tjm
    assert validate_gate_noise_instruction(random, context) is random
    assert validate_gate_noise_instruction(composite, context) is composite


def test_random_unitary_support_may_be_a_strict_subset_of_gate_support() -> None:
    """One-site outcomes are valid after a two-site gate."""
    instruction = RandomUnitaryInstruction((LocalOperator(_X, (4,), "X"),))

    assert validate_gate_noise_instruction(instruction, _context()) is instruction


def test_random_unitary_allows_identity_and_two_site_outcomes() -> None:
    """Zero-operator identity and two-site unitary outcomes should both validate."""
    identity = RandomUnitaryInstruction((), "product-pauli", ("I", "I"))
    correlated = RandomUnitaryInstruction((LocalOperator(_ZZ, (1, 4), "ZZ"),))

    assert validate_gate_noise_instruction(identity, _context()) is identity
    assert validate_gate_noise_instruction(correlated, _context()) is correlated


def test_validate_random_unitary_rejects_off_gate_support() -> None:
    """Every realized local operator must be supported by the current gate."""
    instruction = RandomUnitaryInstruction((LocalOperator(_X, (0,), "X"),))

    with pytest.raises(ValueError, match="outside gate support"):
        validate_gate_noise_instruction(instruction, _context())


def test_validate_composite_checks_every_child_against_gate_support() -> None:
    """Explicit composition must not bypass child support validation."""
    instruction = CompositeGateNoiseInstruction(
        (
            TJMNoiseInstruction(_model()),
            RandomUnitaryInstruction((LocalOperator(_X, (0,), "X"),)),
        ),
        "mixed",
    )

    with pytest.raises(ValueError, match="outside gate support"):
        validate_gate_noise_instruction(instruction, _context())


def test_validate_noise_model_accepts_zero_strength_and_subset_support() -> None:
    """Concrete zero-rate local processes remain valid TJM instructions."""
    model = _model(sites=(4,), strength=np.float64(0.0))

    result = validate_gate_noise_instruction(TJMNoiseInstruction(model, "zero"), _context())

    assert isinstance(result, TJMNoiseInstruction)


@pytest.mark.parametrize(
    ("strength", "error", "match"),
    [
        ({"distribution": "normal", "mean": 0.1, "std": 0.01}, TypeError, "concrete real"),
        (True, TypeError, "concrete real"),
        (1.0j, TypeError, "concrete real"),
        (np.nan, ValueError, "finite"),
        (np.inf, ValueError, "finite"),
        (-0.1, ValueError, "nonnegative"),
    ],
)
def test_validate_noise_model_rejects_invalid_strengths(
    strength: object,
    error: type[Exception],
    match: str,
) -> None:
    """Gate-local TJM strengths must already be concrete, finite, and nonnegative."""
    model = _model(strength=strength)

    with pytest.raises(error, match=match):
        validate_gate_noise_instruction(model, _context())


@pytest.mark.parametrize(
    ("sites", "error", "match"),
    [
        ([], ValueError, "one or two"),
        ([1, 1], ValueError, "duplicate"),
        ([True], TypeError, "Boolean"),
        ([-1], ValueError, "nonnegative"),
        ([0], ValueError, "outside gate support"),
        ((4, 1), ValueError, "ascending"),
    ],
)
def test_validate_noise_model_rejects_invalid_process_support(
    sites: object,
    error: type[Exception],
    match: str,
) -> None:
    """Provider-local TJM processes must have valid support inside the gate."""
    model = _model()
    model.processes[0]["sites"] = sites

    with pytest.raises(error, match=match):
        validate_gate_noise_instruction(model, _context())


def test_validate_noise_model_rejects_scheduled_jumps() -> None:
    """Scheduled jumps have time semantics and are unsupported after one gate."""
    model = NoiseModel(scheduled_jumps=[{"name": "pauli_x", "sites": [1], "time": 0.0}])

    with pytest.raises(ValueError, match="scheduled jumps"):
        validate_gate_noise_instruction(model, _context())


@pytest.mark.parametrize(
    ("mutation", "error", "match"),
    [
        (lambda process: process.pop("name"), ValueError, "missing 'name'"),
        (lambda process: process.update(name=1), TypeError, "name.*string"),
        (lambda process: process.update(name=""), ValueError, "name.*nonempty"),
        (lambda process: process.pop("matrix"), ValueError, "provide a matrix"),
        (lambda process: process.update(matrix=np.eye(4)), ValueError, "matrix.*shape"),
        (
            lambda process: process.update(matrix=np.array([[1.0, np.nan], [0.0, 1.0]])),
            ValueError,
            "matrix.*finite",
        ),
        (lambda process: process.update(matrix=[["bad"]]), TypeError, "complex matrix"),
    ],
    ids=(
        "missing-name",
        "non-string-name",
        "empty-name",
        "missing-one-site-matrix",
        "wrong-matrix-shape",
        "nonfinite-matrix",
        "nonnumeric-matrix",
    ),
)
def test_validate_noise_model_rejects_malformed_matrix_payloads(
    mutation: Callable[[dict[str, object]], object],
    error: type[Exception],
    match: str,
) -> None:
    """Malformed provider matrices should fail before Krotov applies them."""
    model = _model()
    mutation(model.processes[0])

    with pytest.raises(error, match=match):
        validate_gate_noise_instruction(model, _context())


@pytest.mark.parametrize(
    ("factors", "error", "match"),
    [
        (None, ValueError, "matrix or two local factors"),
        (np.eye(2), TypeError, "sequence of two matrices"),
        ((np.eye(2),), ValueError, "exactly two"),
        ((np.eye(4), np.eye(2)), ValueError, "factors.*shape"),
        ((np.array([[1.0, np.inf], [0.0, 1.0]]), np.eye(2)), ValueError, "factors.*finite"),
        (([["bad"]], np.eye(2)), TypeError, "complex matrix"),
    ],
    ids=(
        "missing-factors",
        "factor-array",
        "one-factor",
        "wrong-factor-shape",
        "nonfinite-factor",
        "nonnumeric-factor",
    ),
)
def test_validate_noise_model_rejects_malformed_factor_payloads(
    factors: object,
    error: type[Exception],
    match: str,
) -> None:
    """Malformed long-range factors should fail during provider validation."""
    model = NoiseModel([{"name": "crosstalk_xx", "sites": [1, 4], "strength": 0.1}])
    if factors is None:
        model.processes[0].pop("factors")
    else:
        model.processes[0]["factors"] = factors

    with pytest.raises(error, match=match):
        validate_gate_noise_instruction(model, _context())


@pytest.mark.parametrize(
    ("mutation", "error", "match"),
    [
        (lambda model: model.processes.append(cast("Any", object())), TypeError, "mapping"),
        (lambda model: model.processes[0].pop("strength"), ValueError, "missing 'strength'"),
        (lambda model: model.processes[0].pop("sites"), ValueError, "missing 'sites'"),
    ],
)
def test_validate_noise_model_handles_mutated_malformed_models(
    mutation: Callable[[NoiseModel], None],
    error: type[Exception],
    match: str,
) -> None:
    """Validation should fail clearly even if a mutable model was corrupted."""
    model = _model()
    mutation(model)

    with pytest.raises(error, match=match):
        validate_gate_noise_instruction(model, _context())


def test_validate_instruction_rejects_wrong_context_and_output_types() -> None:
    """Arbitrary categorical or state-dependent provider outputs are unsupported."""
    with pytest.raises(TypeError, match="GateNoiseContext"):
        validate_gate_noise_instruction(None, cast("Any", object()))
    with pytest.raises(TypeError, match="providers must return"):
        validate_gate_noise_instruction(cast("Any", {"probabilities": [1.0]}), _context())


def test_gate_noise_provider_protocol_uses_context_and_supplied_rng() -> None:
    """A provider can sample solely from immutable context and trajectory RNG."""

    def provider(context: GateNoiseContext, rng: np.random.Generator) -> GateNoiseInstruction | None:
        if context.gate_name != "rzz":
            return None
        label = "X" if rng.random() < 0.5 else "I"
        operators = (LocalOperator(_X, (context.sites[0],), label),) if label == "X" else ()
        return RandomUnitaryInstruction(operators, "seeded", (label,))

    typed_provider: GateNoiseProvider = provider
    first = typed_provider(_context(), np.random.default_rng(12))
    second = typed_provider(_context(), np.random.default_rng(12))

    assert first == second
