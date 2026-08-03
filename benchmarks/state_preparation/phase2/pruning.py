# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Pure, mechanically verifiable top-down pruning primitives for WP21.

This module owns circuit-local pruning mathematics.  Target authorization,
fixed-rate noise sampling, pipeline execution, and artifact persistence live in
``topdown_pruning.py`` so the routines below remain usable in analytic tests.
"""

# The strict records share the validators in ``validation.py``. Repeating the
# same propagated exception lists in every small decoder would obscure the
# scientific contracts.
# ruff: file-ignore[docstring-missing-returns, docstring-missing-exception]

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol, cast

import numpy as np

from benchmarks.state_preparation.circuits import compile_quantinuum_native
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

from .canonical import canonical_checksum, verify_sealed_mapping
from .noisy_krotov import (
    NoisyKrotovCircuitBinding,
    decode_noisy_krotov_circuit_binding_document,
)
from .validation import (
    require_bool,
    require_checksum,
    require_float,
    require_int,
    require_mapping,
    require_slug,
)
from .wp20_resources import CircuitResourceMetrics, WP20WorkLedger, measure_circuit_resources

if TYPE_CHECKING:
    from numpy.typing import NDArray


PRUNING_STAGE_POLICY_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp21_pruning_policy.v1"
PARAMETER_SHIFT_REQUEST_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp21_parameter_shift_request.v1"
PARAMETER_SHIFT_EVALUATION_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp21_parameter_shift_evaluation.v1"
PRUNING_UNIT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp21_pruning_unit.v1"
PRUNING_UNIT_SCORE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp21_pruning_unit_score.v1"
PARAMETER_REMAP_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp21_parameter_remap.v1"
PRUNING_ROUND_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp21_pruning_round.v1"

TOPDOWN_RANDOM_METHOD_ID = "topdown_random"
TOPDOWN_MAGNITUDE_METHOD_ID = "topdown_magnitude"
TOPDOWN_IMPACT_ONE_SHOT_METHOD_ID = "topdown_impact_one_shot"
TOPDOWN_IMPACT_ITERATIVE_METHOD_ID = "topdown_impact_iterative"
TOPDOWN_METHOD_IDS = frozenset({
    TOPDOWN_RANDOM_METHOD_ID,
    TOPDOWN_MAGNITUDE_METHOD_ID,
    TOPDOWN_IMPACT_ONE_SHOT_METHOD_ID,
    TOPDOWN_IMPACT_ITERATIVE_METHOD_ID,
})

PruningUnitKind = Literal["parameter", "gate", "shared_parameter_group", "compiled_entangler_group"]
ScoringObjectiveKind = Literal["none", "noiseless_fidelity", "fixed_map_sample_average_fidelity"]
RemovalSchedule = Literal["fixed_count", "fraction_floor"]

PRUNING_UNIT_KINDS = frozenset({"parameter", "gate", "shared_parameter_group", "compiled_entangler_group"})
SCORING_OBJECTIVE_KINDS = frozenset({"none", "noiseless_fidelity", "fixed_map_sample_average_fidelity"})
SCORE_DATA_ROLE = "training_target"
SCORE_AGGREGATION = "sum_absolute_member_scores_v1"
TIE_BREAK_RULE = "score_ascending_then_unit_id_v1"
PER_ROUND_MAP_SCOPE = "per_round_input_circuit_v1"
PARAMETER_SHIFT_RULE = "gate_occurrence_pi_over_2_v1"
MAGNITUDE_RULE = "wrapped_effective_angle_l1_v1"
FRACTION_ROUNDING_RULE = "floor_v1"
RANDOM_RANKING_RULE = "numpy_pcg64_uniform_v1"

_POLICY_KEYS = frozenset({
    "schema_version",
    "pruning_unit",
    "scoring_objective_kind",
    "scoring_data_role",
    "score_aggregation",
    "removal_schedule",
    "removal_count",
    "removal_fraction",
    "tie_break_rule",
    "per_round_map_scope",
    "relax_after_round",
    "policy_checksum",
})


def _vector(value: object, count: int, name: str) -> NDArray[np.float64]:
    """Return a detached finite float64 parameter vector."""
    if not isinstance(value, np.ndarray):
        msg = f"{name} must be a NumPy array."
        raise TypeError(msg)
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (count,) or not np.all(np.isfinite(result)):
        msg = f"{name} must be finite with shape ({count},)."
        raise ValueError(msg)
    return result.copy()


def _vector_checksum(value: NDArray[np.float64]) -> str:
    """Return a stable checksum of canonical little-endian float64 bytes."""
    payload = np.ascontiguousarray(value, dtype=np.dtype("<f8")).tobytes(order="C")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _sequence(value: object, name: str) -> tuple[object, ...]:
    """Return one non-string sequence as a tuple."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = f"{name} must be a sequence."
        raise TypeError(msg)
    return tuple(value)


def _copy_gate(
    gate: ParameterizedGate,
    *,
    param_index: int | None,
    angle_offset: float | None = None,
    logical_gate_id: int | str | None = None,
) -> ParameterizedGate:
    """Return a detached gate preserving every execution-relevant field."""
    return ParameterizedGate(
        name=gate.name,
        sites=tuple(gate.sites),
        param_index=param_index,
        angle_scale=float(gate.angle_scale),
        angle_offset=float(gate.angle_offset if angle_offset is None else angle_offset),
        data_map=copy.deepcopy(gate.data_map),
        fixed_params=tuple(float(value) for value in gate.fixed_params),
        logical_gate_id=gate.logical_gate_id if logical_gate_id is None else logical_gate_id,
        native_gate_id=None,
        noise_enabled=gate.noise_enabled,
    )


@dataclass(frozen=True, slots=True)
class PruningStagePolicy:
    """Target-independent pruning choices embedded in a stage template."""

    pruning_unit: PruningUnitKind
    scoring_objective_kind: ScoringObjectiveKind
    removal_schedule: RemovalSchedule
    removal_count: int | None
    removal_fraction: float | None
    relax_after_round: bool
    scoring_data_role: str = SCORE_DATA_ROLE
    score_aggregation: str = SCORE_AGGREGATION
    tie_break_rule: str = TIE_BREAK_RULE
    per_round_map_scope: str = PER_ROUND_MAP_SCOPE
    schema_version: str = field(default=PRUNING_STAGE_POLICY_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate all frozen algorithmic choices."""
        if self.pruning_unit not in PRUNING_UNIT_KINDS:
            msg = f"pruning_unit must be one of {sorted(PRUNING_UNIT_KINDS)!r}."
            raise ValueError(msg)
        if self.scoring_objective_kind not in SCORING_OBJECTIVE_KINDS:
            msg = f"scoring_objective_kind must be one of {sorted(SCORING_OBJECTIVE_KINDS)!r}."
            raise ValueError(msg)
        if self.scoring_data_role != SCORE_DATA_ROLE:
            msg = f"scoring_data_role must be {SCORE_DATA_ROLE!r}."
            raise ValueError(msg)
        if self.score_aggregation != SCORE_AGGREGATION:
            msg = f"score_aggregation must be {SCORE_AGGREGATION!r}."
            raise ValueError(msg)
        if self.tie_break_rule != TIE_BREAK_RULE:
            msg = f"tie_break_rule must be {TIE_BREAK_RULE!r}."
            raise ValueError(msg)
        if self.per_round_map_scope != PER_ROUND_MAP_SCOPE:
            msg = f"per_round_map_scope must be {PER_ROUND_MAP_SCOPE!r}."
            raise ValueError(msg)
        if self.removal_schedule not in {"fixed_count", "fraction_floor"}:
            msg = "removal_schedule must be 'fixed_count' or 'fraction_floor'."
            raise ValueError(msg)
        count = self.removal_count
        fraction = self.removal_fraction
        if self.removal_schedule == "fixed_count":
            count = require_int(count, "removal_count", minimum=1)
            if fraction is not None:
                msg = "fixed_count removal forbids removal_fraction."
                raise ValueError(msg)
        else:
            if count is not None:
                msg = "fraction_floor removal forbids removal_count."
                raise ValueError(msg)
            fraction = require_float(fraction, "removal_fraction", minimum=0.0, maximum=1.0)
            if fraction <= 0.0 or fraction >= 1.0:
                msg = "removal_fraction must lie strictly between zero and one."
                raise ValueError(msg)
        object.__setattr__(self, "removal_count", count)
        object.__setattr__(self, "removal_fraction", fraction)
        object.__setattr__(self, "relax_after_round", require_bool(self.relax_after_round, "relax_after_round"))

    def _content_dict(self) -> dict[str, object]:
        """Return the exact stage-hyperparameter payload without its checksum."""
        return {
            "schema_version": self.schema_version,
            "pruning_unit": self.pruning_unit,
            "scoring_objective_kind": self.scoring_objective_kind,
            "scoring_data_role": self.scoring_data_role,
            "score_aggregation": self.score_aggregation,
            "removal_schedule": self.removal_schedule,
            "removal_count": self.removal_count,
            "removal_fraction": self.removal_fraction,
            "tie_break_rule": self.tie_break_rule,
            "per_round_map_scope": self.per_round_map_scope,
            "relax_after_round": self.relax_after_round,
        }

    @property
    def policy_checksum(self) -> str:
        """Checksum of the target-independent policy."""
        return canonical_checksum(self._content_dict())

    def to_mapping(self) -> dict[str, object]:
        """Return the exact mapping stored in ``optimizer_hyperparameters``."""
        return {**self._content_dict(), "policy_checksum": self.policy_checksum}

    @classmethod
    def from_mapping(cls, data: object) -> PruningStagePolicy:
        """Decode and checksum-verify one embedded policy mapping."""
        mapping = require_mapping(data, "WP21 pruning policy")
        if set(mapping) != _POLICY_KEYS:
            msg = "WP21 pruning policy fields do not match the exact schema."
            raise ValueError(msg)
        if mapping["schema_version"] != PRUNING_STAGE_POLICY_SCHEMA_VERSION:
            msg = "WP21 pruning policy uses an unsupported schema version."
            raise ValueError(msg)
        policy = cls(
            pruning_unit=cast("PruningUnitKind", mapping["pruning_unit"]),
            scoring_objective_kind=cast("ScoringObjectiveKind", mapping["scoring_objective_kind"]),
            removal_schedule=cast("RemovalSchedule", mapping["removal_schedule"]),
            removal_count=cast("int | None", mapping["removal_count"]),
            removal_fraction=cast("float | None", mapping["removal_fraction"]),
            relax_after_round=cast("bool", mapping["relax_after_round"]),
            scoring_data_role=cast("str", mapping["scoring_data_role"]),
            score_aggregation=cast("str", mapping["score_aggregation"]),
            tie_break_rule=cast("str", mapping["tie_break_rule"]),
            per_round_map_scope=cast("str", mapping["per_round_map_scope"]),
        )
        if mapping["policy_checksum"] != policy.policy_checksum:
            msg = "WP21 pruning policy checksum changed during normalization."
            raise ValueError(msg)
        return policy


@dataclass(frozen=True, slots=True)
class PruningStageSpec:
    """Resolved policy including method, score rule, and random stream."""

    method_id: str
    score_rule: str
    policy: PruningStagePolicy
    random_seed: int | None

    def __post_init__(self) -> None:
        """Require a method/rule/objective-consistent resolved policy."""
        method = require_slug(self.method_id, "method_id")
        if method not in TOPDOWN_METHOD_IDS:
            msg = f"method_id must be one of {sorted(TOPDOWN_METHOD_IDS)!r}."
            raise ValueError(msg)
        expected_rule = {
            TOPDOWN_RANDOM_METHOD_ID: "random",
            TOPDOWN_MAGNITUDE_METHOD_ID: "magnitude",
            TOPDOWN_IMPACT_ONE_SHOT_METHOD_ID: "impact_one_shot",
            TOPDOWN_IMPACT_ITERATIVE_METHOD_ID: "impact_iterative",
        }[method]
        if self.score_rule != expected_rule:
            msg = "score_rule does not agree with the resolved top-down method."
            raise ValueError(msg)
        if not isinstance(self.policy, PruningStagePolicy):
            msg = "policy must be a PruningStagePolicy."
            raise TypeError(msg)
        random = method == TOPDOWN_RANDOM_METHOD_ID
        if random != (self.random_seed is not None):
            msg = "A random seed is present exactly for topdown_random."
            raise ValueError(msg)
        seed = None if self.random_seed is None else require_int(self.random_seed, "random_seed")
        if method in {TOPDOWN_RANDOM_METHOD_ID, TOPDOWN_MAGNITUDE_METHOD_ID}:
            if self.policy.scoring_objective_kind != "none":
                msg = "Random and magnitude pruning do not use a target scoring objective."
                raise ValueError(msg)
        elif self.policy.scoring_objective_kind not in {
            "noiseless_fidelity",
            "fixed_map_sample_average_fidelity",
        }:
            msg = "Impact pruning requires an explicit noiseless or fixed-map fidelity objective."
            raise ValueError(msg)
        object.__setattr__(self, "method_id", method)
        object.__setattr__(self, "random_seed", seed)

    @property
    def pruning_unit(self) -> PruningUnitKind:
        """The frozen pruning unit."""
        return self.policy.pruning_unit

    @property
    def scoring_objective_kind(self) -> ScoringObjectiveKind:
        """The frozen score-objective kind."""
        return self.policy.scoring_objective_kind

    @property
    def removal_schedule(self) -> RemovalSchedule:
        """The frozen count/fraction schedule."""
        return self.policy.removal_schedule

    @property
    def removal_count(self) -> int | None:
        """The fixed removal count when configured."""
        return self.policy.removal_count

    @property
    def removal_fraction(self) -> float | None:
        """The removal fraction when configured."""
        return self.policy.removal_fraction

    @property
    def relax_after_round(self) -> bool:
        """Whether the pipeline must relax immediately after this round."""
        return self.policy.relax_after_round

    @property
    def content_checksum(self) -> str:
        """Checksum binding the policy to method, rule, and resolved seed."""
        return canonical_checksum({
            "method_id": self.method_id,
            "score_rule": self.score_rule,
            "policy": self.policy.to_mapping(),
            "random_seed": self.random_seed,
        })

    def to_dict(self) -> dict[str, object]:
        """Return complete resolved specification data."""
        return {
            "method_id": self.method_id,
            "score_rule": self.score_rule,
            "policy": self.policy.to_mapping(),
            "random_seed": self.random_seed,
            "content_checksum": self.content_checksum,
        }

    @classmethod
    def from_mapping(
        cls,
        data: object,
        *,
        method_id: str,
        score_rule: str,
        random_seed: int | None,
    ) -> PruningStageSpec:
        """Resolve one embedded policy against its stage context."""
        return cls(
            method_id=method_id,
            score_rule=score_rule,
            policy=PruningStagePolicy.from_mapping(data),
            random_seed=random_seed,
        )

    @classmethod
    def from_dict(cls, data: object) -> PruningStageSpec:
        """Decode one complete resolved specification."""
        mapping = require_mapping(data, "WP21 resolved pruning spec")
        expected = {"method_id", "score_rule", "policy", "random_seed", "content_checksum"}
        if set(mapping) != expected:
            msg = "WP21 resolved pruning spec fields do not match the exact schema."
            raise ValueError(msg)
        result = cls(
            method_id=cast("str", mapping["method_id"]),
            score_rule=cast("str", mapping["score_rule"]),
            policy=PruningStagePolicy.from_mapping(mapping["policy"]),
            random_seed=cast("int | None", mapping["random_seed"]),
        )
        if mapping["content_checksum"] != result.content_checksum:
            msg = "Resolved pruning specification checksum changed during normalization."
            raise ValueError(msg)
        return result


@dataclass(frozen=True, slots=True)
class ParameterShiftRequest:
    """One gate-occurrence shift in a generalized shared-parameter gradient."""

    gate_occurrence_index: int
    parameter_index: int
    sign: Literal[-1, 1]
    shift: float
    base_parameter_checksum: str
    shifted_circuit_checksum: str
    schema_version: str = field(default=PARAMETER_SHIFT_REQUEST_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the exact occurrence-level shift rule."""
        object.__setattr__(
            self,
            "gate_occurrence_index",
            require_int(self.gate_occurrence_index, "gate_occurrence_index"),
        )
        object.__setattr__(self, "parameter_index", require_int(self.parameter_index, "parameter_index"))
        if self.sign not in {-1, 1}:
            msg = "sign must equal -1 or +1."
            raise ValueError(msg)
        shift = float(require_float(self.shift, "shift", minimum=0.0))
        if shift.hex() != float(math.pi / 2).hex():
            msg = "The WP21 occurrence shift must equal pi/2 exactly."
            raise ValueError(msg)
        object.__setattr__(self, "shift", shift)
        object.__setattr__(
            self,
            "base_parameter_checksum",
            require_checksum(self.base_parameter_checksum, "base_parameter_checksum"),
        )
        object.__setattr__(
            self,
            "shifted_circuit_checksum",
            require_checksum(self.shifted_circuit_checksum, "shifted_circuit_checksum"),
        )

    def _content_dict(self) -> dict[str, object]:
        """Return all request fields."""
        return {
            "schema_version": self.schema_version,
            "gate_occurrence_index": self.gate_occurrence_index,
            "parameter_index": self.parameter_index,
            "sign": self.sign,
            "shift": self.shift,
            "base_parameter_checksum": self.base_parameter_checksum,
            "shifted_circuit_checksum": self.shifted_circuit_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of this objective request."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed request data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> ParameterShiftRequest:
        """Decode one sealed occurrence request."""
        expected = frozenset({
            "schema_version",
            "gate_occurrence_index",
            "parameter_index",
            "sign",
            "shift",
            "base_parameter_checksum",
            "shifted_circuit_checksum",
            "content_checksum",
        })
        mapping = verify_sealed_mapping(data, expected_keys=expected, name="WP21 parameter-shift request")
        if mapping["schema_version"] != PARAMETER_SHIFT_REQUEST_SCHEMA_VERSION:
            msg = "Parameter-shift request uses an unsupported schema version."
            raise ValueError(msg)
        request = cls(
            gate_occurrence_index=cast("int", mapping["gate_occurrence_index"]),
            parameter_index=cast("int", mapping["parameter_index"]),
            sign=cast("Literal[-1, 1]", mapping["sign"]),
            shift=cast("float", mapping["shift"]),
            base_parameter_checksum=cast("str", mapping["base_parameter_checksum"]),
            shifted_circuit_checksum=cast("str", mapping["shifted_circuit_checksum"]),
        )
        if mapping["content_checksum"] != request.content_checksum:
            msg = "Parameter-shift request checksum changed during normalization."
            raise ValueError(msg)
        return request


@dataclass(frozen=True, slots=True)
class ParameterShiftEvaluation:
    """One fidelity value returned for a sealed occurrence shift."""

    request: ParameterShiftRequest
    fidelity: float
    schema_version: str = field(default=PARAMETER_SHIFT_EVALUATION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate request type and bounded fidelity."""
        if not isinstance(self.request, ParameterShiftRequest):
            msg = "request must be a ParameterShiftRequest."
            raise TypeError(msg)
        object.__setattr__(self, "fidelity", require_float(self.fidelity, "fidelity", minimum=0.0, maximum=1.0))

    @property
    def content_checksum(self) -> str:
        """Checksum of the request/value pair."""
        return canonical_checksum({
            "schema_version": self.schema_version,
            "request": self.request.to_dict(),
            "fidelity": self.fidelity,
        })

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed evaluation data."""
        return {
            "schema_version": self.schema_version,
            "request": self.request.to_dict(),
            "fidelity": self.fidelity,
            "content_checksum": self.content_checksum,
        }

    @classmethod
    def from_dict(cls, data: object) -> ParameterShiftEvaluation:
        """Decode one sealed evaluation."""
        expected = frozenset({"schema_version", "request", "fidelity", "content_checksum"})
        mapping = verify_sealed_mapping(data, expected_keys=expected, name="WP21 parameter-shift evaluation")
        if mapping["schema_version"] != PARAMETER_SHIFT_EVALUATION_SCHEMA_VERSION:
            msg = "Parameter-shift evaluation uses an unsupported schema version."
            raise ValueError(msg)
        evaluation = cls(
            request=ParameterShiftRequest.from_dict(mapping["request"]),
            fidelity=cast("float", mapping["fidelity"]),
        )
        if mapping["content_checksum"] != evaluation.content_checksum:
            msg = "Parameter-shift evaluation checksum changed during normalization."
            raise ValueError(msg)
        return evaluation


class FidelityObjective(Protocol):
    """Circuit-aware scalar fidelity callback used by occurrence shifts."""

    def __call__(
        self,
        circuit: ParameterizedCircuit,
        parameters: NDArray[np.float64],
        request: ParameterShiftRequest,
    ) -> float:
        """Return fidelity for one shifted circuit and detached parameters."""


@dataclass(frozen=True, slots=True)
class PruningUnit:
    """One atomic pruning unit and its complete logical/native membership."""

    unit_id: str
    unit_kind: PruningUnitKind
    gate_indices: tuple[int, ...]
    parameter_indices: tuple[int, ...]
    native_gate_ids: tuple[int | str, ...]
    logical_gate_ids: tuple[int | str, ...]
    schema_version: str = field(default=PRUNING_UNIT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate stable identifiers and nonempty duplicate-free memberships."""
        object.__setattr__(self, "unit_id", require_slug(self.unit_id, "unit_id"))
        if self.unit_kind not in PRUNING_UNIT_KINDS:
            msg = "unit_kind is not a supported WP21 pruning unit."
            raise ValueError(msg)
        gates = tuple(require_int(value, "gate_indices item") for value in self.gate_indices)
        parameters = tuple(require_int(value, "parameter_indices item") for value in self.parameter_indices)
        if not gates or len(gates) != len(set(gates)) or tuple(sorted(gates)) != gates:
            msg = "gate_indices must be a nonempty sorted duplicate-free tuple."
            raise ValueError(msg)
        if len(parameters) != len(set(parameters)) or tuple(sorted(parameters)) != parameters:
            msg = "parameter_indices must be sorted and duplicate-free."
            raise ValueError(msg)
        native = tuple(self.native_gate_ids)
        logical = tuple(self.logical_gate_ids)
        if len(native) != len(set(native)) or not logical or len(logical) != len(set(logical)):
            msg = "Native and logical gate identities must be duplicate-free; logical membership is nonempty."
            raise ValueError(msg)
        object.__setattr__(self, "gate_indices", gates)
        object.__setattr__(self, "parameter_indices", parameters)
        object.__setattr__(self, "native_gate_ids", native)
        object.__setattr__(self, "logical_gate_ids", logical)

    @property
    def content_checksum(self) -> str:
        """Checksum of exact unit membership."""
        return canonical_checksum(self._content_dict())

    def _content_dict(self) -> dict[str, object]:
        """Return all unit fields."""
        return {
            "schema_version": self.schema_version,
            "unit_id": self.unit_id,
            "unit_kind": self.unit_kind,
            "gate_indices": list(self.gate_indices),
            "parameter_indices": list(self.parameter_indices),
            "native_gate_ids": list(self.native_gate_ids),
            "logical_gate_ids": list(self.logical_gate_ids),
        }

    def to_dict(self) -> dict[str, object]:
        """Return sealed unit data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> PruningUnit:
        """Decode one sealed pruning unit."""
        expected = frozenset({
            "schema_version",
            "unit_id",
            "unit_kind",
            "gate_indices",
            "parameter_indices",
            "native_gate_ids",
            "logical_gate_ids",
            "content_checksum",
        })
        mapping = verify_sealed_mapping(data, expected_keys=expected, name="WP21 pruning unit")
        if mapping["schema_version"] != PRUNING_UNIT_SCHEMA_VERSION:
            msg = "Pruning unit uses an unsupported schema version."
            raise ValueError(msg)
        unit = cls(
            unit_id=cast("str", mapping["unit_id"]),
            unit_kind=cast("PruningUnitKind", mapping["unit_kind"]),
            gate_indices=cast("tuple[int, ...]", mapping["gate_indices"]),
            parameter_indices=cast("tuple[int, ...]", mapping["parameter_indices"]),
            native_gate_ids=cast("tuple[int | str, ...]", mapping["native_gate_ids"]),
            logical_gate_ids=cast("tuple[int | str, ...]", mapping["logical_gate_ids"]),
        )
        if mapping["content_checksum"] != unit.content_checksum:
            msg = "Pruning-unit checksum changed during normalization."
            raise ValueError(msg)
        return unit


@dataclass(frozen=True, slots=True)
class PruningUnitScore:
    """One deterministic unit score used for ascending removal ranking."""

    unit_id: str
    score: float
    member_scores: tuple[float, ...]
    schema_version: str = field(default=PRUNING_UNIT_SCORE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate stable identity and nonnegative finite scores."""
        object.__setattr__(self, "unit_id", require_slug(self.unit_id, "unit_id"))
        score = require_float(self.score, "score", minimum=0.0)
        members = tuple(require_float(value, "member score", minimum=0.0) for value in self.member_scores)
        if not members or float(sum(members)).hex() != float(score).hex():
            msg = "score must equal the exact sum of nonempty member_scores."
            raise ValueError(msg)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "member_scores", members)

    @property
    def content_checksum(self) -> str:
        """Checksum of the unit score."""
        return canonical_checksum(self._content_dict())

    def _content_dict(self) -> dict[str, object]:
        """Return all score fields."""
        return {
            "schema_version": self.schema_version,
            "unit_id": self.unit_id,
            "score": self.score,
            "member_scores": list(self.member_scores),
        }

    def to_dict(self) -> dict[str, object]:
        """Return sealed score data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> PruningUnitScore:
        """Decode one sealed unit score."""
        expected = frozenset({"schema_version", "unit_id", "score", "member_scores", "content_checksum"})
        mapping = verify_sealed_mapping(data, expected_keys=expected, name="WP21 pruning score")
        if mapping["schema_version"] != PRUNING_UNIT_SCORE_SCHEMA_VERSION:
            msg = "Pruning score uses an unsupported schema version."
            raise ValueError(msg)
        result = cls(
            unit_id=cast("str", mapping["unit_id"]),
            score=cast("float", mapping["score"]),
            member_scores=cast("tuple[float, ...]", mapping["member_scores"]),
        )
        if mapping["content_checksum"] != result.content_checksum:
            msg = "Pruning-score checksum changed during normalization."
            raise ValueError(msg)
        return result


@dataclass(frozen=True, slots=True)
class ParameterRemap:
    """Exact old-to-new parameter and gate mapping after unit removal."""

    input_parameter_count: int
    output_parameter_count: int
    old_to_new_parameter_indices: tuple[tuple[int, int], ...]
    removed_parameter_indices: tuple[int, ...]
    retained_input_gate_indices: tuple[int, ...]
    removed_input_gate_indices: tuple[int, ...]
    schema_version: str = field(default=PARAMETER_REMAP_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate a complete injective remap and gate partition."""
        input_count = require_int(self.input_parameter_count, "input_parameter_count", minimum=1)
        output_count = require_int(self.output_parameter_count, "output_parameter_count", minimum=1)
        pairs = tuple(
            (require_int(old, "old index"), require_int(new, "new index"))
            for old, new in self.old_to_new_parameter_indices
        )
        if tuple(old for old, _new in pairs) != tuple(sorted(old for old, _new in pairs)):
            msg = "Parameter remap must be ordered by old index."
            raise ValueError(msg)
        if tuple(new for _old, new in pairs) != tuple(range(len(pairs))) or len(pairs) != output_count:
            msg = "New parameter indices must be contiguous and complete."
            raise ValueError(msg)
        removed = tuple(require_int(value, "removed parameter") for value in self.removed_parameter_indices)
        if {old for old, _new in pairs} | set(removed) != set(range(input_count)) or {old for old, _new in pairs} & set(
            removed
        ):
            msg = "Retained and removed parameter indices must partition the input vector."
            raise ValueError(msg)
        retained_gates = tuple(require_int(value, "retained gate") for value in self.retained_input_gate_indices)
        removed_gates = tuple(require_int(value, "removed gate") for value in self.removed_input_gate_indices)
        if set(retained_gates) & set(removed_gates):
            msg = "Retained and removed gate indices cannot overlap."
            raise ValueError(msg)
        object.__setattr__(self, "input_parameter_count", input_count)
        object.__setattr__(self, "output_parameter_count", output_count)
        object.__setattr__(self, "old_to_new_parameter_indices", pairs)
        object.__setattr__(self, "removed_parameter_indices", tuple(sorted(removed)))
        object.__setattr__(self, "retained_input_gate_indices", tuple(sorted(retained_gates)))
        object.__setattr__(self, "removed_input_gate_indices", tuple(sorted(removed_gates)))

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete remap."""
        return canonical_checksum(self._content_dict())

    def _content_dict(self) -> dict[str, object]:
        """Return all remap fields."""
        return {
            "schema_version": self.schema_version,
            "input_parameter_count": self.input_parameter_count,
            "output_parameter_count": self.output_parameter_count,
            "old_to_new_parameter_indices": [list(pair) for pair in self.old_to_new_parameter_indices],
            "removed_parameter_indices": list(self.removed_parameter_indices),
            "retained_input_gate_indices": list(self.retained_input_gate_indices),
            "removed_input_gate_indices": list(self.removed_input_gate_indices),
        }

    def to_dict(self) -> dict[str, object]:
        """Return sealed remap data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> ParameterRemap:
        """Decode one sealed parameter remap."""
        expected = frozenset({
            "schema_version",
            "input_parameter_count",
            "output_parameter_count",
            "old_to_new_parameter_indices",
            "removed_parameter_indices",
            "retained_input_gate_indices",
            "removed_input_gate_indices",
            "content_checksum",
        })
        mapping = verify_sealed_mapping(data, expected_keys=expected, name="WP21 parameter remap")
        if mapping["schema_version"] != PARAMETER_REMAP_SCHEMA_VERSION:
            msg = "Parameter remap uses an unsupported schema version."
            raise ValueError(msg)
        raw_pairs = _sequence(mapping["old_to_new_parameter_indices"], "old_to_new_parameter_indices")
        pairs: list[tuple[int, int]] = []
        for raw in raw_pairs:
            pair = _sequence(raw, "parameter remap pair")
            if len(pair) != 2:
                msg = "Every parameter remap entry must contain old and new indices."
                raise ValueError(msg)
            pairs.append((cast("int", pair[0]), cast("int", pair[1])))
        result = cls(
            input_parameter_count=cast("int", mapping["input_parameter_count"]),
            output_parameter_count=cast("int", mapping["output_parameter_count"]),
            old_to_new_parameter_indices=tuple(pairs),
            removed_parameter_indices=cast("tuple[int, ...]", mapping["removed_parameter_indices"]),
            retained_input_gate_indices=cast("tuple[int, ...]", mapping["retained_input_gate_indices"]),
            removed_input_gate_indices=cast("tuple[int, ...]", mapping["removed_input_gate_indices"]),
        )
        if mapping["content_checksum"] != result.content_checksum:
            msg = "Parameter-remap checksum changed during normalization."
            raise ValueError(msg)
        return result


def build_pruning_units(circuit: ParameterizedCircuit, unit_kind: PruningUnitKind) -> tuple[PruningUnit, ...]:
    """Build deterministic pruning units from logical and native provenance."""
    if not isinstance(circuit, ParameterizedCircuit):
        msg = "circuit must be a ParameterizedCircuit."
        raise TypeError(msg)
    if unit_kind not in PRUNING_UNIT_KINDS:
        msg = "unit_kind is not supported."
        raise ValueError(msg)
    compilation = None
    if unit_kind == "compiled_entangler_group":
        compilation = compile_quantinuum_native(circuit)
    occurrences: dict[int, list[int]] = {}
    for gate_index, gate in enumerate(circuit.gates):
        if gate.param_index is not None:
            occurrences.setdefault(gate.param_index, []).append(gate_index)
    units: list[PruningUnit] = []
    if unit_kind in {"parameter", "shared_parameter_group"}:
        for parameter_index in sorted(occurrences):
            gates = tuple(occurrences[parameter_index])
            if unit_kind == "parameter" and len(gates) != 1:
                msg = "parameter pruning requires every trainable parameter to occur in exactly one gate."
                raise ValueError(msg)
            logical_members: list[int | str] = []
            for index in gates:
                logical_gate_id = circuit.gates[index].logical_gate_id
                logical_members.append(index if logical_gate_id is None else logical_gate_id)
            logical = tuple(logical_members)
            prefix = "parameter" if unit_kind == "parameter" else "shared_parameter"
            units.append(
                PruningUnit(
                    unit_id=f"{prefix}_{parameter_index:06d}",
                    unit_kind=unit_kind,
                    gate_indices=gates,
                    parameter_indices=(parameter_index,),
                    native_gate_ids=(),
                    logical_gate_ids=logical,
                )
            )
    elif unit_kind == "gate":
        for gate_index, gate in enumerate(circuit.gates):
            if gate.param_index is None:
                continue
            logical_id = gate.logical_gate_id if gate.logical_gate_id is not None else gate_index
            units.append(
                PruningUnit(
                    unit_id=f"gate_{gate_index:06d}",
                    unit_kind=unit_kind,
                    gate_indices=(gate_index,),
                    parameter_indices=(gate.param_index,),
                    native_gate_ids=(),
                    logical_gate_ids=(logical_id,),
                )
            )
    else:
        assert compilation is not None
        for source in compilation.mapping:
            gate = circuit.gates[source.source_logical_gate_index]
            if len(gate.sites) != 2 or gate.param_index is None:
                continue
            native_ids = tuple(
                cast("int | str", compilation.circuit.gates[index].native_gate_id)
                for index in source.native_gate_indices
            )
            units.append(
                PruningUnit(
                    unit_id=f"compiled_entangler_{source.source_logical_gate_index:06d}",
                    unit_kind=unit_kind,
                    gate_indices=(source.source_logical_gate_index,),
                    parameter_indices=(gate.param_index,),
                    native_gate_ids=native_ids,
                    logical_gate_ids=(source.logical_gate_id,),
                )
            )
    if not units:
        msg = "The circuit does not contain any units of the requested pruning kind."
        raise ValueError(msg)
    return tuple(units)


def _shifted_occurrence_binding(
    binding: NoisyKrotovCircuitBinding,
    gate_index: int,
    sign: int,
) -> NoisyKrotovCircuitBinding:
    """Return a binding with exactly one gate angle offset shifted by pi/2."""
    circuit = binding.circuit
    gates: list[ParameterizedGate] = []
    for index, gate in enumerate(circuit.gates):
        gates.append(
            _copy_gate(
                gate,
                param_index=gate.param_index,
                angle_offset=(gate.angle_offset + sign * math.pi / 2 if index == gate_index else gate.angle_offset),
            )
        )
    shifted = ParameterizedCircuit(circuit.num_qubits, gates, num_params=circuit.num_params)
    return NoisyKrotovCircuitBinding(shifted, binding.topology_id)


def generalized_parameter_shift_derivative(
    circuit_binding: NoisyKrotovCircuitBinding,
    theta: NDArray[np.float64],
    objective: FidelityObjective,
    *,
    trajectory_count: int = 0,
    sampling_work: WP20WorkLedger | None = None,
) -> tuple[NDArray[np.float64], Mapping[int, float], tuple[ParameterShiftEvaluation, ...], WP20WorkLedger]:
    """Compute an exact generalized derivative by shifting each gate occurrence.

    A shared parameter is never shifted globally.  Each occurrence is shifted
    independently, its derivative contribution is multiplied by that gate's
    ``angle_scale``, and the contributions are summed into the shared parameter.
    """
    if not isinstance(circuit_binding, NoisyKrotovCircuitBinding):
        msg = "circuit_binding must be a NoisyKrotovCircuitBinding."
        raise TypeError(msg)
    if not callable(objective):
        msg = "objective must be callable."
        raise TypeError(msg)
    circuit = circuit_binding.circuit
    parameters = _vector(theta, circuit.num_params, "theta")
    trajectories = require_int(trajectory_count, "trajectory_count")
    base_checksum = _vector_checksum(parameters)
    gradient = np.zeros(circuit.num_params, dtype=np.float64)
    occurrence_derivatives: dict[int, float] = {}
    evaluations: list[ParameterShiftEvaluation] = []
    for gate_index, gate in enumerate(circuit.gates):
        if gate.param_index is None:
            continue
        values: dict[int, float] = {}
        for sign in (1, -1):
            shifted = _shifted_occurrence_binding(circuit_binding, gate_index, sign)
            request = ParameterShiftRequest(
                gate_occurrence_index=gate_index,
                parameter_index=gate.param_index,
                sign=sign,
                shift=math.pi / 2,
                base_parameter_checksum=base_checksum,
                shifted_circuit_checksum=shifted.content_checksum,
            )
            fidelity = require_float(
                objective(shifted.circuit, parameters.copy(), request),
                "objective fidelity",
                minimum=0.0,
                maximum=1.0,
            )
            values[sign] = fidelity
            evaluations.append(ParameterShiftEvaluation(request=request, fidelity=fidelity))
        derivative = float(gate.angle_scale) * 0.5 * (values[1] - values[-1])
        occurrence_derivatives[gate_index] = derivative
        gradient[gate.param_index] += derivative
    evaluation_count = len(evaluations)
    gate_count = len(circuit.gates)
    replay_work = WP20WorkLedger(
        forward_circuit_evaluations=evaluation_count * max(1, trajectories),
        trajectory_gate_applications=evaluation_count * trajectories * gate_count,
        training_trajectories=evaluation_count * trajectories,
        objective_calls=evaluation_count,
        gradient_calls=1,
    )
    work = replay_work
    if sampling_work is not None:
        if not isinstance(sampling_work, WP20WorkLedger):
            msg = "sampling_work must be a WP20WorkLedger or None."
            raise TypeError(msg)
        increments = sampling_work.to_dict()
        work = work.plus(**{
            name: cast("int | float", increments[name])
            for name in (
                "forward_circuit_evaluations",
                "backward_circuit_evaluations",
                "trajectory_gate_applications",
                "training_trajectories",
                "checkpoint_validation_trajectories",
                "test_trajectories",
                "objective_calls",
                "gradient_calls",
                "cross_trajectory_pairings",
                "wall_time_seconds",
                "peak_memory_bytes",
            )
        })
    gradient.setflags(write=False)
    return gradient, MappingProxyType(dict(occurrence_derivatives)), tuple(evaluations), work


def _wrapped_angle_magnitude(circuit: ParameterizedCircuit, gate_index: int, theta: NDArray[np.float64]) -> float:
    """Return absolute effective angle in the canonical [-pi, pi) interval."""
    angle = circuit.angle(circuit.gates[gate_index], theta, None)
    wrapped = (angle + math.pi) % (2 * math.pi) - math.pi
    return abs(float(wrapped))


def rank_pruning_units(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    units: Sequence[PruningUnit],
    spec: PruningStageSpec,
    *,
    gradient: NDArray[np.float64] | None = None,
    occurrence_derivatives: Mapping[int, float] | None = None,
) -> tuple[PruningUnitScore, ...]:
    """Return scores in deterministic ascending removal order."""
    parameters = _vector(theta, circuit.num_params, "theta")
    resolved_units = tuple(units)
    if not resolved_units or not all(isinstance(unit, PruningUnit) for unit in resolved_units):
        msg = "units must contain PruningUnit values."
        raise TypeError(msg)
    scores: list[PruningUnitScore] = []
    if spec.method_id == TOPDOWN_RANDOM_METHOD_ID:
        assert spec.random_seed is not None
        rng = np.random.Generator(np.random.PCG64(spec.random_seed))
        for unit in sorted(resolved_units, key=lambda item: item.unit_id):
            value = float(rng.random())
            scores.append(PruningUnitScore(unit_id=unit.unit_id, score=value, member_scores=(value,)))
    elif spec.method_id == TOPDOWN_MAGNITUDE_METHOD_ID:
        for unit in resolved_units:
            members = tuple(_wrapped_angle_magnitude(circuit, index, parameters) for index in unit.gate_indices)
            scores.append(PruningUnitScore(unit_id=unit.unit_id, score=float(sum(members)), member_scores=members))
    else:
        if gradient is None or occurrence_derivatives is None:
            msg = "Impact ranking requires full parameter and gate-occurrence derivatives."
            raise ValueError(msg)
        resolved_gradient = _vector(gradient, circuit.num_params, "gradient")
        for unit in resolved_units:
            if unit.unit_kind in {"parameter", "shared_parameter_group"}:
                members = tuple(
                    abs(float(parameters[index] * resolved_gradient[index])) for index in unit.parameter_indices
                )
            else:
                members = tuple(
                    abs(float(parameters[circuit.gates[index].param_index] * occurrence_derivatives[index]))
                    for index in unit.gate_indices
                    if circuit.gates[index].param_index is not None
                )
            scores.append(PruningUnitScore(unit_id=unit.unit_id, score=float(sum(members)), member_scores=members))
    return tuple(sorted(scores, key=lambda item: (item.score, item.unit_id)))


def _removal_count(spec: PruningStageSpec, unit_count: int) -> int:
    """Return the exact fixed or floor-rounded removal count."""
    if spec.removal_schedule == "fixed_count":
        count = cast("int", spec.removal_count)
    else:
        count = math.floor(unit_count * cast("float", spec.removal_fraction))
    if count < 1 or count >= unit_count:
        msg = "The resolved removal schedule must remove at least one but not every pruning unit."
        raise ValueError(msg)
    return count


def prune_circuit(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    units: Sequence[PruningUnit],
    removed_unit_ids: Sequence[str],
    *,
    output_topology_id: str,
) -> tuple[NoisyKrotovCircuitBinding, NDArray[np.float64], ParameterRemap]:
    """Remove complete units and rebuild parameter indices without semantic drift."""
    parameters = _vector(theta, circuit.num_params, "theta")
    topology = require_slug(output_topology_id, "output_topology_id")
    resolved_units = tuple(units)
    by_id = {unit.unit_id: unit for unit in resolved_units}
    removed_ids = tuple(require_slug(value, "removed_unit_id") for value in removed_unit_ids)
    if len(removed_ids) != len(set(removed_ids)) or not set(removed_ids) <= set(by_id):
        msg = "removed_unit_ids must be duplicate-free known pruning units."
        raise ValueError(msg)
    removed_gates = {index for unit_id in removed_ids for index in by_id[unit_id].gate_indices}
    retained_gate_indices = tuple(index for index in range(len(circuit.gates)) if index not in removed_gates)
    retained_parameters = tuple(
        sorted({
            cast("int", circuit.gates[index].param_index)
            for index in retained_gate_indices
            if circuit.gates[index].param_index is not None
        })
    )
    if not retained_parameters:
        msg = "A pruned Phase II circuit must retain at least one trainable parameter."
        raise ValueError(msg)
    old_to_new = {old: new for new, old in enumerate(retained_parameters)}
    output_gates: list[ParameterizedGate] = []
    for input_index in retained_gate_indices:
        gate = circuit.gates[input_index]
        output_gates.append(
            _copy_gate(
                gate,
                param_index=None if gate.param_index is None else old_to_new[gate.param_index],
                logical_gate_id=gate.logical_gate_id,
            )
        )
    output_theta = parameters[np.asarray(retained_parameters, dtype=np.int64)].copy()
    output_circuit = ParameterizedCircuit(
        circuit.num_qubits,
        output_gates,
        num_params=len(retained_parameters),
    )
    binding = NoisyKrotovCircuitBinding(output_circuit, topology)
    remap = ParameterRemap(
        input_parameter_count=circuit.num_params,
        output_parameter_count=output_circuit.num_params,
        old_to_new_parameter_indices=tuple(old_to_new.items()),
        removed_parameter_indices=tuple(sorted(set(range(circuit.num_params)) - set(retained_parameters))),
        retained_input_gate_indices=retained_gate_indices,
        removed_input_gate_indices=tuple(sorted(removed_gates)),
    )
    return binding, output_theta, remap


def _reconstruct_impact_derivatives(
    input_circuit_binding: NoisyKrotovCircuitBinding,
    input_theta: NDArray[np.float64],
    evaluations: Sequence[ParameterShiftEvaluation],
) -> tuple[NDArray[np.float64], Mapping[int, float]]:
    """Reconstruct and verify occurrence requests, then derive their gradient."""
    circuit = input_circuit_binding.circuit
    base_checksum = _vector_checksum(input_theta)
    expected_count = 2 * sum(gate.param_index is not None for gate in circuit.gates)
    if len(evaluations) != expected_count:
        msg = "Impact-pruning evaluations do not cover every trainable gate occurrence."
        raise ValueError(msg)
    gradient = np.zeros(circuit.num_params, dtype=np.float64)
    occurrence_derivatives: dict[int, float] = {}
    evaluation_index = 0
    for gate_index, gate in enumerate(circuit.gates):
        if gate.param_index is None:
            continue
        fidelities: dict[int, float] = {}
        for sign in (1, -1):
            evaluation = evaluations[evaluation_index]
            evaluation_index += 1
            shifted = _shifted_occurrence_binding(input_circuit_binding, gate_index, sign)
            request = evaluation.request
            expected = (
                request.gate_occurrence_index == gate_index
                and request.parameter_index == gate.param_index
                and request.sign == sign
                and float(request.shift).hex() == float(math.pi / 2).hex()
                and request.base_parameter_checksum == base_checksum
                and request.shifted_circuit_checksum == shifted.content_checksum
            )
            if not expected:
                msg = "Impact-pruning parameter-shift requests are not the exact occurrence schedule."
                raise ValueError(msg)
            fidelities[sign] = evaluation.fidelity
        derivative = float(gate.angle_scale) * 0.5 * (fidelities[1] - fidelities[-1])
        occurrence_derivatives[gate_index] = derivative
        gradient[gate.param_index] += derivative
    gradient.setflags(write=False)
    return gradient, MappingProxyType(occurrence_derivatives)


@dataclass(frozen=True, slots=True, init=False)
class PruningRoundResult:
    """Checksum-sealed result of one fully reconstructed pruning round."""

    method_id: str
    spec: PruningStageSpec
    round_index: int
    scoring_trajectory_count: int
    input_circuit_binding: NoisyKrotovCircuitBinding
    output_circuit_binding: NoisyKrotovCircuitBinding
    input_parameter_checksum: str
    output_parameter_checksum: str
    units: tuple[PruningUnit, ...]
    scores: tuple[PruningUnitScore, ...]
    parameter_shift_evaluations: tuple[ParameterShiftEvaluation, ...]
    removed_unit_ids: tuple[str, ...]
    retained_unit_ids: tuple[str, ...]
    parameter_remap: ParameterRemap
    input_resources: CircuitResourceMetrics
    output_resources: CircuitResourceMetrics
    work: WP20WorkLedger
    _input_theta: NDArray[np.float64] = field(repr=False)
    _output_theta: NDArray[np.float64] = field(repr=False)
    schema_version: str = field(default=PRUNING_ROUND_SCHEMA_VERSION, init=False)

    def __init__(
        self,
        *,
        spec: PruningStageSpec,
        round_index: int,
        scoring_trajectory_count: int,
        input_circuit_binding: NoisyKrotovCircuitBinding,
        output_circuit_binding: NoisyKrotovCircuitBinding,
        input_theta: NDArray[np.float64],
        output_theta: NDArray[np.float64],
        units: Sequence[PruningUnit],
        scores: Sequence[PruningUnitScore],
        parameter_shift_evaluations: Sequence[ParameterShiftEvaluation],
        removed_unit_ids: Sequence[str],
        retained_unit_ids: Sequence[str],
        parameter_remap: ParameterRemap,
        input_resources: CircuitResourceMetrics,
        output_resources: CircuitResourceMetrics,
        work: WP20WorkLedger,
    ) -> None:
        """Validate and defensively snapshot a complete round."""
        if not isinstance(spec, PruningStageSpec):
            msg = "spec must be a PruningStageSpec."
            raise TypeError(msg)
        if not isinstance(input_circuit_binding, NoisyKrotovCircuitBinding) or not isinstance(
            output_circuit_binding, NoisyKrotovCircuitBinding
        ):
            msg = "Round circuit bindings must be NoisyKrotovCircuitBinding values."
            raise TypeError(msg)
        input_parameters = _vector(input_theta, input_circuit_binding.circuit.num_params, "input_theta")
        output_parameters = _vector(output_theta, output_circuit_binding.circuit.num_params, "output_theta")
        resolved_units = tuple(units)
        resolved_scores = tuple(scores)
        evaluations = tuple(parameter_shift_evaluations)
        if not all(isinstance(value, PruningUnit) for value in resolved_units):
            msg = "units must contain PruningUnit values."
            raise TypeError(msg)
        if not all(isinstance(value, PruningUnitScore) for value in resolved_scores):
            msg = "scores must contain PruningUnitScore values."
            raise TypeError(msg)
        if not all(isinstance(value, ParameterShiftEvaluation) for value in evaluations):
            msg = "parameter_shift_evaluations contains an invalid value."
            raise TypeError(msg)
        removed = tuple(require_slug(value, "removed_unit_id") for value in removed_unit_ids)
        retained = tuple(require_slug(value, "retained_unit_id") for value in retained_unit_ids)
        unit_ids = tuple(unit.unit_id for unit in resolved_units)
        if set(removed) | set(retained) != set(unit_ids) or set(removed) & set(retained):
            msg = "Removed and retained unit identities must partition the input units."
            raise ValueError(msg)
        if tuple(score.unit_id for score in resolved_scores) != tuple(
            score.unit_id for score in sorted(resolved_scores, key=lambda item: (item.score, item.unit_id))
        ):
            msg = "scores must be stored in deterministic removal order."
            raise ValueError(msg)
        if removed != tuple(score.unit_id for score in resolved_scores[: len(removed)]):
            msg = "removed_unit_ids do not select the least-scoring deterministic prefix."
            raise ValueError(msg)
        expected_units = build_pruning_units(input_circuit_binding.circuit, spec.pruning_unit)
        if tuple(unit.to_dict() for unit in expected_units) != tuple(unit.to_dict() for unit in resolved_units):
            msg = "Pruning units are not mechanically derived from the input circuit."
            raise ValueError(msg)
        rebuilt_binding, rebuilt_theta, rebuilt_remap = prune_circuit(
            input_circuit_binding.circuit,
            input_parameters,
            resolved_units,
            removed,
            output_topology_id=output_circuit_binding.topology_id,
        )
        if (
            rebuilt_binding.to_dict() != output_circuit_binding.to_dict()
            or not np.array_equal(rebuilt_theta, output_parameters)
            or rebuilt_remap.to_dict() != parameter_remap.to_dict()
        ):
            msg = "Output circuit, parameters, or remap are not the exact result of unit removal."
            raise ValueError(msg)
        if not isinstance(input_resources, CircuitResourceMetrics) or not isinstance(
            output_resources, CircuitResourceMetrics
        ):
            msg = "Round resources must be CircuitResourceMetrics values."
            raise TypeError(msg)
        if input_resources.to_dict() != measure_circuit_resources(input_circuit_binding.circuit).to_dict():
            msg = "input_resources are not compiler-derived from the input circuit."
            raise ValueError(msg)
        if output_resources.to_dict() != measure_circuit_resources(output_circuit_binding.circuit).to_dict():
            msg = "output_resources are not compiler-derived from the output circuit."
            raise ValueError(msg)
        if not isinstance(parameter_remap, ParameterRemap) or not isinstance(work, WP20WorkLedger):
            msg = "parameter_remap and work have invalid types."
            raise TypeError(msg)
        trajectories = require_int(scoring_trajectory_count, "scoring_trajectory_count")
        impact = spec.method_id in {TOPDOWN_IMPACT_ONE_SHOT_METHOD_ID, TOPDOWN_IMPACT_ITERATIVE_METHOD_ID}
        if impact != bool(evaluations):
            msg = "Parameter-shift evaluations are present exactly for impact pruning."
            raise ValueError(msg)
        if impact:
            fixed_map_scoring = spec.scoring_objective_kind == "fixed_map_sample_average_fidelity"
            if fixed_map_scoring != (trajectories > 0):
                msg = "Fixed-map impact scoring requires positive trajectories; noiseless impact requires zero."
                raise ValueError(msg)
            gradient, occurrence_derivatives = _reconstruct_impact_derivatives(
                input_circuit_binding,
                input_parameters,
                evaluations,
            )
            evaluation_count = len(evaluations)
            gate_count = len(input_circuit_binding.circuit.gates)
            expected_work = WP20WorkLedger(
                forward_circuit_evaluations=evaluation_count * max(1, trajectories) + trajectories,
                trajectory_gate_applications=(evaluation_count + 1) * trajectories * gate_count,
                training_trajectories=(evaluation_count + 1) * trajectories,
                objective_calls=evaluation_count,
                gradient_calls=1,
            )
            if work.to_dict() != expected_work.to_dict():
                msg = "Impact-pruning work is not exactly implied by its occurrence shifts and scoring trajectories."
                raise ValueError(msg)
            expected_scores = rank_pruning_units(
                input_circuit_binding.circuit,
                input_parameters,
                resolved_units,
                spec,
                gradient=gradient,
                occurrence_derivatives=occurrence_derivatives,
            )
        else:
            if trajectories:
                msg = "Random and magnitude pruning require zero scoring trajectories."
                raise ValueError(msg)
            if work.to_dict() != WP20WorkLedger().to_dict():
                msg = "Random and magnitude pruning cannot claim objective, trajectory, or runtime work."
                raise ValueError(msg)
            expected_scores = rank_pruning_units(
                input_circuit_binding.circuit,
                input_parameters,
                resolved_units,
                spec,
            )
        if tuple(score.to_dict() for score in resolved_scores) != tuple(score.to_dict() for score in expected_scores):
            msg = "Pruning scores are not mechanically derived from the sealed method inputs."
            raise ValueError(msg)
        expected_remove_count = _removal_count(spec, len(resolved_units))
        if len(removed) != expected_remove_count:
            msg = "Removed-unit count does not match the sealed removal schedule."
            raise ValueError(msg)
        object.__setattr__(self, "method_id", spec.method_id)
        object.__setattr__(self, "spec", spec)
        object.__setattr__(self, "round_index", require_int(round_index, "round_index"))
        object.__setattr__(self, "scoring_trajectory_count", trajectories)
        object.__setattr__(self, "input_circuit_binding", input_circuit_binding)
        object.__setattr__(self, "output_circuit_binding", output_circuit_binding)
        object.__setattr__(self, "input_parameter_checksum", _vector_checksum(input_parameters))
        object.__setattr__(self, "output_parameter_checksum", _vector_checksum(output_parameters))
        object.__setattr__(self, "units", resolved_units)
        object.__setattr__(self, "scores", resolved_scores)
        object.__setattr__(self, "parameter_shift_evaluations", evaluations)
        object.__setattr__(self, "removed_unit_ids", removed)
        object.__setattr__(self, "retained_unit_ids", retained)
        object.__setattr__(self, "parameter_remap", parameter_remap)
        object.__setattr__(self, "input_resources", input_resources)
        object.__setattr__(self, "output_resources", output_resources)
        object.__setattr__(self, "work", work)
        input_parameters.setflags(write=False)
        output_parameters.setflags(write=False)
        object.__setattr__(self, "_input_theta", input_parameters)
        object.__setattr__(self, "_output_theta", output_parameters)
        object.__setattr__(self, "schema_version", PRUNING_ROUND_SCHEMA_VERSION)

    @property
    def input_theta(self) -> NDArray[np.float64]:
        """Detached pre-pruning parameters."""
        return self._input_theta.copy()

    @property
    def output_theta(self) -> NDArray[np.float64]:
        """Detached post-pruning parameters."""
        return self._output_theta.copy()

    def _content_dict(self) -> dict[str, object]:
        """Return all checksum-covered round fields."""
        return {
            "schema_version": self.schema_version,
            "method_id": self.method_id,
            "spec": self.spec.to_dict(),
            "round_index": self.round_index,
            "scoring_trajectory_count": self.scoring_trajectory_count,
            "input_circuit_binding": self.input_circuit_binding.to_dict(),
            "output_circuit_binding": self.output_circuit_binding.to_dict(),
            "input_theta": self.input_theta.tolist(),
            "output_theta": self.output_theta.tolist(),
            "input_parameter_checksum": self.input_parameter_checksum,
            "output_parameter_checksum": self.output_parameter_checksum,
            "units": [unit.to_dict() for unit in self.units],
            "scores": [score.to_dict() for score in self.scores],
            "parameter_shift_evaluations": [value.to_dict() for value in self.parameter_shift_evaluations],
            "removed_unit_ids": list(self.removed_unit_ids),
            "retained_unit_ids": list(self.retained_unit_ids),
            "parameter_remap": self.parameter_remap.to_dict(),
            "input_resources": self.input_resources.to_dict(),
            "output_resources": self.output_resources.to_dict(),
            "work": self.work.to_dict(),
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of complete round evidence."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return sealed round evidence."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> PruningRoundResult:
        """Decode and mechanically verify a complete pruning round."""
        expected = frozenset({
            "schema_version",
            "method_id",
            "spec",
            "round_index",
            "scoring_trajectory_count",
            "input_circuit_binding",
            "output_circuit_binding",
            "input_theta",
            "output_theta",
            "input_parameter_checksum",
            "output_parameter_checksum",
            "units",
            "scores",
            "parameter_shift_evaluations",
            "removed_unit_ids",
            "retained_unit_ids",
            "parameter_remap",
            "input_resources",
            "output_resources",
            "work",
            "content_checksum",
        })
        mapping = verify_sealed_mapping(data, expected_keys=expected, name="WP21 pruning round")
        if mapping["schema_version"] != PRUNING_ROUND_SCHEMA_VERSION:
            msg = "Pruning round uses an unsupported schema version."
            raise ValueError(msg)
        input_binding = decode_noisy_krotov_circuit_binding_document(mapping["input_circuit_binding"])
        output_binding = decode_noisy_krotov_circuit_binding_document(mapping["output_circuit_binding"])
        result = cls(
            spec=PruningStageSpec.from_dict(mapping["spec"]),
            round_index=cast("int", mapping["round_index"]),
            scoring_trajectory_count=cast("int", mapping["scoring_trajectory_count"]),
            input_circuit_binding=input_binding,
            output_circuit_binding=output_binding,
            input_theta=np.asarray(mapping["input_theta"], dtype=np.float64),
            output_theta=np.asarray(mapping["output_theta"], dtype=np.float64),
            units=tuple(PruningUnit.from_dict(value) for value in _sequence(mapping["units"], "units")),
            scores=tuple(PruningUnitScore.from_dict(value) for value in _sequence(mapping["scores"], "scores")),
            parameter_shift_evaluations=tuple(
                ParameterShiftEvaluation.from_dict(value)
                for value in _sequence(mapping["parameter_shift_evaluations"], "parameter_shift_evaluations")
            ),
            removed_unit_ids=cast("tuple[str, ...]", mapping["removed_unit_ids"]),
            retained_unit_ids=cast("tuple[str, ...]", mapping["retained_unit_ids"]),
            parameter_remap=ParameterRemap.from_dict(mapping["parameter_remap"]),
            input_resources=CircuitResourceMetrics.from_dict(mapping["input_resources"]),
            output_resources=CircuitResourceMetrics.from_dict(mapping["output_resources"]),
            work=WP20WorkLedger.from_dict(mapping["work"]),
        )
        aliases = {
            "method_id": result.method_id,
            "input_parameter_checksum": result.input_parameter_checksum,
            "output_parameter_checksum": result.output_parameter_checksum,
            "content_checksum": result.content_checksum,
        }
        if any(mapping[name] != value for name, value in aliases.items()):
            msg = "Pruning-round aliases or checksum changed during normalization."
            raise ValueError(msg)
        return result


@dataclass(frozen=True, slots=True)
class PruningRoundValidation:
    """Small typed projection returned after round reconstruction."""

    result: PruningRoundResult

    def __post_init__(self) -> None:
        """Require a reconstructed pruning result."""
        if not isinstance(self.result, PruningRoundResult):
            msg = "result must be a PruningRoundResult."
            raise TypeError(msg)

    @property
    def input_parameter_count(self) -> int:
        """The pre-pruning parameter count."""
        return self.result.input_circuit_binding.circuit.num_params

    @property
    def output_parameter_count(self) -> int:
        """The post-pruning parameter count."""
        return self.result.output_circuit_binding.circuit.num_params


def run_pruning_round(
    input_circuit_binding: NoisyKrotovCircuitBinding,
    theta: NDArray[np.float64],
    spec: PruningStageSpec,
    *,
    round_index: int,
    output_topology_id: str,
    objective: FidelityObjective | None = None,
    scoring_trajectory_count: int = 0,
    sampling_work: WP20WorkLedger | None = None,
) -> PruningRoundResult:
    """Score, rank, remove, remap, compile, and seal one pruning round."""
    if not isinstance(input_circuit_binding, NoisyKrotovCircuitBinding):
        msg = "input_circuit_binding must be a NoisyKrotovCircuitBinding."
        raise TypeError(msg)
    if not isinstance(spec, PruningStageSpec):
        msg = "spec must be a PruningStageSpec."
        raise TypeError(msg)
    circuit = input_circuit_binding.circuit
    parameters = _vector(theta, circuit.num_params, "theta")
    units = build_pruning_units(circuit, spec.pruning_unit)
    gradient: NDArray[np.float64] | None = None
    occurrences: Mapping[int, float] | None = None
    evaluations: tuple[ParameterShiftEvaluation, ...] = ()
    work = WP20WorkLedger()
    impact = spec.method_id in {TOPDOWN_IMPACT_ONE_SHOT_METHOD_ID, TOPDOWN_IMPACT_ITERATIVE_METHOD_ID}
    if impact:
        if objective is None:
            msg = "Impact pruning requires a fidelity objective."
            raise ValueError(msg)
        gradient, occurrences, evaluations, work = generalized_parameter_shift_derivative(
            input_circuit_binding,
            parameters,
            objective,
            trajectory_count=scoring_trajectory_count,
            sampling_work=sampling_work,
        )
    elif objective is not None or scoring_trajectory_count or sampling_work is not None:
        msg = "Random and magnitude pruning forbid objective and trajectory-scoring inputs."
        raise ValueError(msg)
    scores = rank_pruning_units(
        circuit,
        parameters,
        units,
        spec,
        gradient=gradient,
        occurrence_derivatives=occurrences,
    )
    remove_count = _removal_count(spec, len(units))
    removed = tuple(score.unit_id for score in scores[:remove_count])
    retained = tuple(unit.unit_id for unit in units if unit.unit_id not in set(removed))
    output_binding, output_theta, remap = prune_circuit(
        circuit,
        parameters,
        units,
        removed,
        output_topology_id=output_topology_id,
    )
    return PruningRoundResult(
        spec=spec,
        round_index=round_index,
        scoring_trajectory_count=scoring_trajectory_count,
        input_circuit_binding=input_circuit_binding,
        output_circuit_binding=output_binding,
        input_theta=parameters,
        output_theta=output_theta,
        units=units,
        scores=scores,
        parameter_shift_evaluations=evaluations,
        removed_unit_ids=removed,
        retained_unit_ids=retained,
        parameter_remap=remap,
        input_resources=measure_circuit_resources(circuit),
        output_resources=measure_circuit_resources(output_binding.circuit),
        work=work,
    )


def validate_pruning_round_result(data: object) -> PruningRoundValidation:
    """Decode a round document and return its verified typed projection."""
    result = data if isinstance(data, PruningRoundResult) else PruningRoundResult.from_dict(data)
    return PruningRoundValidation(result=result)


__all__ = [
    "FRACTION_ROUNDING_RULE",
    "MAGNITUDE_RULE",
    "PARAMETER_REMAP_SCHEMA_VERSION",
    "PARAMETER_SHIFT_EVALUATION_SCHEMA_VERSION",
    "PARAMETER_SHIFT_REQUEST_SCHEMA_VERSION",
    "PARAMETER_SHIFT_RULE",
    "PER_ROUND_MAP_SCOPE",
    "PRUNING_ROUND_SCHEMA_VERSION",
    "PRUNING_STAGE_POLICY_SCHEMA_VERSION",
    "PRUNING_UNIT_KINDS",
    "PRUNING_UNIT_SCHEMA_VERSION",
    "PRUNING_UNIT_SCORE_SCHEMA_VERSION",
    "RANDOM_RANKING_RULE",
    "SCORE_AGGREGATION",
    "SCORE_DATA_ROLE",
    "SCORING_OBJECTIVE_KINDS",
    "TIE_BREAK_RULE",
    "TOPDOWN_IMPACT_ITERATIVE_METHOD_ID",
    "TOPDOWN_IMPACT_ONE_SHOT_METHOD_ID",
    "TOPDOWN_MAGNITUDE_METHOD_ID",
    "TOPDOWN_METHOD_IDS",
    "TOPDOWN_RANDOM_METHOD_ID",
    "FidelityObjective",
    "ParameterRemap",
    "ParameterShiftEvaluation",
    "ParameterShiftRequest",
    "PruningRoundResult",
    "PruningRoundValidation",
    "PruningStagePolicy",
    "PruningStageSpec",
    "PruningUnit",
    "PruningUnitScore",
    "build_pruning_units",
    "generalized_parameter_shift_derivative",
    "prune_circuit",
    "rank_pruning_units",
    "run_pruning_round",
    "validate_pruning_round_result",
]
