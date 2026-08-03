# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Sealed WP22 training schedules and deterministic trajectory membership.

The records in this module describe scientific choices only.  They are strict,
canonical-JSON-roundtrippable, and checksum sealed so a resumed training run
cannot silently reinterpret a continuation, sampling, checkpoint, or screening
policy.
"""

# Strict public schemas deliberately repeat validation at every decode boundary;
# documenting every delegated validator exception would obscure their contracts.

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cache
from typing import TYPE_CHECKING, Literal, cast

from benchmarks.state_preparation.constants import STANDARD_NOISE_IDS

from .canonical import (
    canonical_checksum,
    canonical_json,
    freeze_json_mapping,
    load_canonical_json_object,
    thaw_json_mapping,
    verify_sealed_mapping,
)
from .validation import (
    require_bool,
    require_checksum,
    require_exact_keys,
    require_float,
    require_int,
    require_mapping,
    require_slug,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


NOISE_CONTINUATION_SCHEMA_VERSION = "yaqs.state_preparation.phase2.noise_strength_continuation.v1"
TRAJECTORY_COUNT_STEP_SCHEMA_VERSION = "yaqs.state_preparation.phase2.trajectory_count_step.v1"
TRAJECTORY_CURRICULUM_SCHEMA_VERSION = "yaqs.state_preparation.phase2.trajectory_count_curriculum.v1"
TRAJECTORY_SAMPLING_SCHEMA_VERSION = "yaqs.state_preparation.phase2.trajectory_sampling_policy.v1"
TRAJECTORY_MEMBERSHIP_SCHEMA_VERSION = "yaqs.state_preparation.phase2.trajectory_ensemble_membership.v1"
VALIDATION_CHECKPOINT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.validation_checkpoint.v1"
CHECKPOINT_VALIDATION_SCHEMA_VERSION = "yaqs.state_preparation.phase2.checkpoint_validation_policy.v1"
CHECKPOINT_SELECTION_SCHEMA_VERSION = "yaqs.state_preparation.phase2.checkpoint_validation_selection.v1"
CHECKPOINT_TRACKER_SCHEMA_VERSION = "yaqs.state_preparation.phase2.checkpoint_validation_tracker.v1"
TRAINING_PHASE_BOUNDARY_SCHEMA_VERSION = "yaqs.state_preparation.phase2.noiseless_pretrain_noisy_finetune.v1"
MULTISTART_SCHEMA_VERSION = "yaqs.state_preparation.phase2.limited_multistart.v1"
MULTISTART_SEEDS_SCHEMA_VERSION = "yaqs.state_preparation.phase2.multistart_seed_bundle.v1"
STANDARD_NOISE_MIXTURE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.standard_noise_mixture.v1"
NOISE_MIXTURE_COMPONENT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.noise_mixture_component.v1"
NOISE_MIXTURE_ALLOCATION_SCHEMA_VERSION = "yaqs.state_preparation.phase2.noise_mixture_allocation.v1"
TRAINING_STRATEGY_SCHEDULE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.training_strategy_schedule.v1"
FROZEN_TRAINING_POLICY_UNIVERSE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.frozen_training_policy_universe.v1"
SEED_DERIVATION_POLICY_SCHEMA_VERSION = "yaqs.state_preparation.phase2.seed_derivation_policy.v1"
EXECUTION_SEED_POLICY_SUITE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.execution_seed_policy_suite.v1"
SEED_DERIVATION_VERSION = "yaqs.state_preparation.phase2.schedule_seed_derivation.v1"

MAX_MULTISTART_COUNT = 8
FROZEN_TRAINING_POLICY_IDS = (
    "direct_matched_fixed_crn",
    "continuation_fixed_crn",
    "curriculum_fixed_crn",
    "periodic_refresh_20",
    "rolling_half_refresh_20",
    "resampled_each_update",
    "frozen_half_depolarizing_half_dephasing",
    "limited_multistart_3",
    "direct_noiseless_control",
)

SeedRole = Literal[
    "initialization",
    "optimizer_ordering",
    "training_trajectory",
    "checkpoint_validation",
    "pilot_evaluation",
    "screening_selection",
    "confirmatory_test",
]
SeedInputKind = Literal["checksum", "seed_role", "slug", "uint64"]
SeedPreimageKind = Literal["domain_identity_list", "named_mapping"]
SeedDomainBinding = SeedRole | Literal["coordinate_selected"] | None
TrajectoryRole = Literal["training_trajectory", "checkpoint_validation"]
SamplingKind = Literal["fixed_crn", "periodic_full_refresh", "rolling_ensemble", "resampled"]
TrainingNoiseMode = Literal["noiseless", "matched", "frozen_mixture"]
NoiseInterpolation = Literal["constant", "linear_clamped"]

_SEED_ROLES = frozenset({
    "initialization",
    "optimizer_ordering",
    "training_trajectory",
    "checkpoint_validation",
    "pilot_evaluation",
    "screening_selection",
    "confirmatory_test",
})
_TRAJECTORY_ROLES = frozenset({"training_trajectory", "checkpoint_validation"})
_SEED_INPUT_KINDS = frozenset({"checksum", "seed_role", "slug", "uint64"})
_SEED_PREIMAGE_KINDS = frozenset({"domain_identity_list", "named_mapping"})
_EXECUTION_PRESETS = frozenset({"training-smoke", "paper-pilot", "paper-screen", "paper-confirm"})

PILOT_OPTIMIZATION_SEED_POLICY_ID = "wp22_pilot_optimization_seed_v1"
SCREEN_OPTIMIZATION_SEED_POLICY_ID = "wp22_screen_optimization_seed_v1"
SMOKE_OPTIMIZATION_SEED_POLICY_ID = "wp22_smoke_optimization_seed_v1"
CONFIRMATORY_OPTIMIZATION_SEED_POLICY_ID = "wp22_confirmatory_optimization_seed_v1"
SMOKE_FRESH_EVALUATION_SEED_POLICY_ID = "wp22_smoke_fresh_evaluation_seed_v1"
PILOT_FRESH_EVALUATION_SEED_POLICY_ID = "wp22_pilot_fresh_evaluation_seed_v1"
SCREENING_ROOT_SEED_POLICY_ID = "wp22_screening_root_seed_v1"
SCREENING_CELL_SEED_POLICY_ID = "wp22_screening_cell_seed_v1"
CONFIRMATORY_FRESH_EVALUATION_SEED_POLICY_ID = "wp22_confirmatory_fresh_evaluation_seed_v1"
PILOT_DIAGNOSTIC_SEED_POLICY_ID = "wp22_pilot_diagnostic_seed_v1"
STAGE_SEED_DERIVATION_POLICY_ID = "phase2_stage_seed_derivation_v1"
SCHEDULE_SEED_DERIVATION_POLICY_ID = "phase2_schedule_seed_derivation_v1"

EXECUTION_SEED_POLICY_IDS = (
    PILOT_OPTIMIZATION_SEED_POLICY_ID,
    SCREEN_OPTIMIZATION_SEED_POLICY_ID,
    SMOKE_OPTIMIZATION_SEED_POLICY_ID,
    CONFIRMATORY_OPTIMIZATION_SEED_POLICY_ID,
    SMOKE_FRESH_EVALUATION_SEED_POLICY_ID,
    PILOT_FRESH_EVALUATION_SEED_POLICY_ID,
    SCREENING_ROOT_SEED_POLICY_ID,
    SCREENING_CELL_SEED_POLICY_ID,
    CONFIRMATORY_FRESH_EVALUATION_SEED_POLICY_ID,
    PILOT_DIAGNOSTIC_SEED_POLICY_ID,
    STAGE_SEED_DERIVATION_POLICY_ID,
    SCHEDULE_SEED_DERIVATION_POLICY_ID,
)

_SEED_POLICY_KEYS = frozenset({
    "schema_version",
    "policy_id",
    "output_semantic",
    "applicable_presets",
    "random_stream_domain",
    "sharing_scope",
    "preimage_kind",
    "constant_fields",
    "input_fields",
    "hash_algorithm",
    "encoding",
    "output_offset_bytes",
    "output_width_bits",
    "byte_order",
    "signed",
    "index_origin",
    "collision_action",
    "content_checksum",
})
_SEED_POLICY_SUITE_KEYS = frozenset({
    "schema_version",
    "suite_id",
    "policy_ids",
    "policies",
    "content_checksum",
})
_SAMPLING_KINDS = frozenset({"fixed_crn", "periodic_full_refresh", "rolling_ensemble", "resampled"})


def _sealed_dict(payload: dict[str, object]) -> dict[str, object]:
    """Return ``payload`` with its canonical content checksum attached."""
    return {**payload, "content_checksum": canonical_checksum(payload)}


def _verify_record(
    value: object,
    *,
    expected_keys: frozenset[str],
    schema_version: str,
    name: str,
) -> Mapping[str, object]:
    """Verify one strict sealed record and its schema version.

    Returns:
        The verified frozen record mapping.

    Raises:
        ValueError: If an input violates the sealed policy constraints.
    """
    mapping = verify_sealed_mapping(value, expected_keys=expected_keys, name=name)
    if mapping["schema_version"] != schema_version:
        msg = f"{name} uses an unsupported schema version."
        raise ValueError(msg)
    return mapping


def _require_uint64(value: object, name: str) -> int:
    """Return a strict unsigned 64-bit integer.

    Raises:
        ValueError: If an input violates the sealed policy constraints.
    """
    result = require_int(value, name)
    if result >= 2**64:
        msg = f"{name} must fit an unsigned 64-bit integer."
        raise ValueError(msg)
    return result


def _require_seed_role(value: object, name: str = "role") -> SeedRole:
    """Return one frozen, domain-separated random-stream role.

    Raises:
        ValueError: If an input violates the sealed policy constraints.
    """
    role = require_slug(value, name)
    if role not in _SEED_ROLES:
        msg = f"{name} is not a supported WP22 seed role."
        raise ValueError(msg)
    return cast("SeedRole", role)


def _require_trajectory_role(value: object, name: str = "role") -> TrajectoryRole:
    """Return one training or checkpoint-validation trajectory role.

    Raises:
        ValueError: If an input violates the sealed policy constraints.
    """
    role = _require_seed_role(value, name)
    if role not in _TRAJECTORY_ROLES:
        msg = f"{name} must be training_trajectory or checkpoint_validation."
        raise ValueError(msg)
    return cast("TrajectoryRole", role)


def _require_seed_input_fields(value: object) -> tuple[tuple[str, SeedInputKind], ...]:
    """Return an ordered, duplicate-free seed-input schema.

    Raises:
        TypeError: If an input has an invalid type or container shape.
        ValueError: If an input violates the sealed policy constraints.
    """
    if type(value) is not tuple:
        msg = "input_fields must be a tuple."
        raise TypeError(msg)
    result: list[tuple[str, SeedInputKind]] = []
    for index, item in enumerate(value):
        if type(item) is not tuple or len(item) != 2:
            msg = f"input_fields[{index}] must be a two-item tuple."
            raise TypeError(msg)
        name = require_slug(item[0], f"input_fields[{index}].name")
        kind = require_slug(item[1], f"input_fields[{index}].kind")
        if kind not in _SEED_INPUT_KINDS:
            msg = f"input_fields[{index}].kind is not supported."
            raise ValueError(msg)
        result.append((name, cast("SeedInputKind", kind)))
    names = tuple(name for name, _kind in result)
    if len(names) != len(set(names)):
        msg = "input_fields must not contain duplicate names."
        raise ValueError(msg)
    return tuple(result)


def _require_seed_coordinate(value: object, kind: SeedInputKind, name: str) -> str | int:
    """Validate one seed coordinate against its sealed scalar kind.

    Returns:
        The validated coordinate value.
    """
    if kind == "checksum":
        return require_checksum(value, name)
    if kind == "seed_role":
        return _require_seed_role(value, name)
    if kind == "slug":
        return require_slug(value, name)
    return _require_uint64(value, name)


@dataclass(frozen=True, slots=True)
class SeedDerivationPolicy:
    """Checksum-sealed algorithm and preimage schema for one seed purpose."""

    policy_id: str
    output_semantic: str
    applicable_presets: tuple[str, ...]
    random_stream_domain: SeedDomainBinding
    sharing_scope: str
    preimage_kind: SeedPreimageKind
    constant_fields: Mapping[str, object]
    input_fields: tuple[tuple[str, SeedInputKind], ...]
    hash_algorithm: str = field(default="sha256", init=False)
    encoding: str = field(default="canonical_json_utf8", init=False)
    output_offset_bytes: int = field(default=0, init=False)
    output_width_bits: int = field(default=64, init=False)
    byte_order: str = field(default="big", init=False)
    signed: bool = field(default=False, init=False)
    index_origin: int = field(default=0, init=False)
    collision_action: str = field(default="abort_before_output", init=False)
    schema_version: str = field(default=SEED_DERIVATION_POLICY_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate a complete, deterministic SHA-256 seed contract.

        Raises:
            TypeError: If an input has an invalid type or container shape.
            ValueError: If an input violates the sealed policy constraints.
        """
        object.__setattr__(self, "policy_id", require_slug(self.policy_id, "policy_id"))
        object.__setattr__(self, "output_semantic", require_slug(self.output_semantic, "output_semantic"))
        presets = self.applicable_presets
        if type(presets) is not tuple:
            msg = "applicable_presets must be a tuple."
            raise TypeError(msg)
        checked_presets = tuple(
            require_slug(preset, f"applicable_presets[{index}]") for index, preset in enumerate(presets)
        )
        if not checked_presets or len(checked_presets) != len(set(checked_presets)):
            msg = "applicable_presets must be nonempty and duplicate-free."
            raise ValueError(msg)
        if not set(checked_presets) <= _EXECUTION_PRESETS:
            msg = "applicable_presets contains an unsupported WP22 execution preset."
            raise ValueError(msg)
        object.__setattr__(self, "applicable_presets", checked_presets)

        domain = self.random_stream_domain
        if domain != "coordinate_selected" and domain is not None:
            domain = _require_seed_role(domain, "random_stream_domain")
        object.__setattr__(self, "random_stream_domain", domain)
        object.__setattr__(self, "sharing_scope", require_slug(self.sharing_scope, "sharing_scope"))
        kind = require_slug(self.preimage_kind, "preimage_kind")
        if kind not in _SEED_PREIMAGE_KINDS:
            msg = "preimage_kind is not supported."
            raise ValueError(msg)
        object.__setattr__(self, "preimage_kind", cast("SeedPreimageKind", kind))

        constants = freeze_json_mapping(self.constant_fields, "constant_fields")
        fields = _require_seed_input_fields(self.input_fields)
        if set(constants) & {name for name, _field_kind in fields}:
            msg = "constant_fields and input_fields must not overlap."
            raise ValueError(msg)
        if kind == "domain_identity_list":
            require_exact_keys(constants, frozenset({"domain"}), "domain-list constant_fields")
            require_slug(constants["domain"], "constant_fields.domain")
        elif not constants:
            msg = "named-mapping policies must seal at least one constant field."
            raise ValueError(msg)
        object.__setattr__(self, "constant_fields", constants)
        object.__setattr__(self, "input_fields", fields)

    def _payload(self) -> dict[str, object]:
        """Return the complete algorithm and ordered preimage contract."""
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "output_semantic": self.output_semantic,
            "applicable_presets": list(self.applicable_presets),
            "random_stream_domain": self.random_stream_domain,
            "sharing_scope": self.sharing_scope,
            "preimage_kind": self.preimage_kind,
            "constant_fields": thaw_json_mapping(self.constant_fields),
            "input_fields": [{"name": name, "kind": kind} for name, kind in self.input_fields],
            "hash_algorithm": self.hash_algorithm,
            "encoding": self.encoding,
            "output_offset_bytes": self.output_offset_bytes,
            "output_width_bits": self.output_width_bits,
            "byte_order": self.byte_order,
            "signed": self.signed,
            "index_origin": self.index_origin,
            "collision_action": self.collision_action,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete seed-derivation contract."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed_dict(self._payload())

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    def preimage(self, coordinates: Mapping[str, object]) -> dict[str, object]:
        """Build the exact validated canonical-JSON preimage.

        Returns:
            The exact canonical-JSON preimage.
        """
        mapping = require_mapping(coordinates, "coordinates")
        require_exact_keys(mapping, frozenset(name for name, _kind in self.input_fields), "coordinates")
        values = {
            name: _require_seed_coordinate(mapping[name], kind, f"coordinates.{name}")
            for name, kind in self.input_fields
        }
        constants = thaw_json_mapping(self.constant_fields)
        if self.preimage_kind == "domain_identity_list":
            return {
                "domain": constants["domain"],
                "identity": [values[name] for name, _kind in self.input_fields],
            }
        return {**constants, **values}

    def derive(self, coordinates: Mapping[str, object]) -> int:
        """Derive one unsigned 64-bit seed from exact named coordinates.

        Returns:
            The derived unsigned 64-bit seed.
        """
        digest = hashlib.sha256(canonical_json(self.preimage(coordinates)).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    @classmethod
    def from_dict(cls, value: object) -> SeedDerivationPolicy:
        """Decode and verify one complete seed-derivation policy.

        Returns:
            The verified SeedDerivationPolicy record.

        Raises:
            TypeError: If an input has an invalid type or container shape.
            ValueError: If an input violates the sealed policy constraints.
        """
        mapping = _verify_record(
            value,
            expected_keys=_SEED_POLICY_KEYS,
            schema_version=SEED_DERIVATION_POLICY_SCHEMA_VERSION,
            name="seed derivation policy",
        )
        fixed = {
            "hash_algorithm": "sha256",
            "encoding": "canonical_json_utf8",
            "output_offset_bytes": 0,
            "output_width_bits": 64,
            "byte_order": "big",
            "signed": False,
            "index_origin": 0,
            "collision_action": "abort_before_output",
        }
        if any(mapping[name] != expected for name, expected in fixed.items()):
            msg = "Seed derivation algorithm or extraction rule changed."
            raise ValueError(msg)
        raw_fields = mapping["input_fields"]
        if type(raw_fields) is not tuple:
            msg = "input_fields must be a JSON array."
            raise TypeError(msg)
        fields: list[tuple[str, SeedInputKind]] = []
        for index, item in enumerate(raw_fields):
            field_mapping = require_mapping(item, f"input_fields[{index}]")
            require_exact_keys(field_mapping, frozenset({"name", "kind"}), f"input_fields[{index}]")
            fields.append((cast("str", field_mapping["name"]), cast("SeedInputKind", field_mapping["kind"])))
        policy = cls(
            policy_id=cast("str", mapping["policy_id"]),
            output_semantic=cast("str", mapping["output_semantic"]),
            applicable_presets=cast("tuple[str, ...]", mapping["applicable_presets"]),
            random_stream_domain=cast("SeedDomainBinding", mapping["random_stream_domain"]),
            sharing_scope=cast("str", mapping["sharing_scope"]),
            preimage_kind=cast("SeedPreimageKind", mapping["preimage_kind"]),
            constant_fields=cast("Mapping[str, object]", mapping["constant_fields"]),
            input_fields=tuple(fields),
        )
        if mapping["content_checksum"] != policy.content_checksum:
            msg = "Seed derivation policy checksum changed during normalization."
            raise ValueError(msg)
        return policy

    @classmethod
    def from_json(cls, payload: str) -> SeedDerivationPolicy:
        """Decode canonical JSON into one verified seed policy.

        Returns:
            The verified SeedDerivationPolicy record.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def _seed_policy(
    policy_id: str,
    *,
    output_semantic: str,
    presets: tuple[str, ...],
    random_stream_domain: SeedDomainBinding,
    sharing_scope: str,
    domain: str,
    fields: tuple[tuple[str, SeedInputKind], ...],
) -> SeedDerivationPolicy:
    """Build one reviewed domain-plus-ordered-identity policy.

    Returns:
        The reviewed seed-derivation policy.
    """
    return SeedDerivationPolicy(
        policy_id=policy_id,
        output_semantic=output_semantic,
        applicable_presets=presets,
        random_stream_domain=random_stream_domain,
        sharing_scope=sharing_scope,
        preimage_kind="domain_identity_list",
        constant_fields={"domain": domain},
        input_fields=fields,
    )


def _reviewed_seed_policies() -> tuple[SeedDerivationPolicy, ...]:
    """Build the exact ordered WP22 seed-policy universe.

    Returns:
        The ordered reviewed seed-derivation policies.
    """
    checksum = cast("SeedInputKind", "checksum")
    uint64 = cast("SeedInputKind", "uint64")
    slug = cast("SeedInputKind", "slug")
    seed_role = cast("SeedInputKind", "seed_role")
    all_presets = ("training-smoke", "paper-pilot", "paper-screen", "paper-confirm")
    return (
        _seed_policy(
            PILOT_OPTIMIZATION_SEED_POLICY_ID,
            output_semantic="optimization_seed",
            presets=("paper-pilot",),
            random_stream_domain=None,
            sharing_scope="study_ordered_tuple",
            domain="wp22_paper_pilot_optimization",
            fields=(("preregistration_checksum", checksum), ("seed_index", uint64)),
        ),
        _seed_policy(
            SCREEN_OPTIMIZATION_SEED_POLICY_ID,
            output_semantic="optimization_seed",
            presets=("paper-screen",),
            random_stream_domain=None,
            sharing_scope="study_ordered_tuple",
            domain="wp22_paper_screen_optimization",
            fields=(("preregistration_checksum", checksum), ("seed_index", uint64)),
        ),
        _seed_policy(
            SMOKE_OPTIMIZATION_SEED_POLICY_ID,
            output_semantic="optimization_seed",
            presets=("training-smoke",),
            random_stream_domain=None,
            sharing_scope="publication_candidate",
            domain="smoke_optimization",
            fields=(("publication_candidate_checksum", checksum),),
        ),
        _seed_policy(
            CONFIRMATORY_OPTIMIZATION_SEED_POLICY_ID,
            output_semantic="optimization_seed",
            presets=("paper-confirm",),
            random_stream_domain=None,
            sharing_scope="target_seed_index",
            domain="confirmatory_optimization",
            fields=(
                ("final_seal_checksum", checksum),
                ("target_instance_spec_checksum", checksum),
                ("seed_index", uint64),
            ),
        ),
        _seed_policy(
            SMOKE_FRESH_EVALUATION_SEED_POLICY_ID,
            output_semantic="fresh_evaluation_seed",
            presets=("training-smoke",),
            random_stream_domain="pilot_evaluation",
            sharing_scope="publication_candidate",
            domain="smoke_fresh_evaluation",
            fields=(("publication_candidate_checksum", checksum),),
        ),
        _seed_policy(
            PILOT_FRESH_EVALUATION_SEED_POLICY_ID,
            output_semantic="fresh_evaluation_seed",
            presets=("paper-pilot",),
            random_stream_domain="pilot_evaluation",
            sharing_scope="target_optimization_candidate",
            domain="pilot_fresh_evaluation",
            fields=(
                ("target_manifest_checksum", checksum),
                ("target_instance_spec_checksum", checksum),
                ("optimization_seed", uint64),
                ("publication_candidate_checksum", checksum),
            ),
        ),
        _seed_policy(
            SCREENING_ROOT_SEED_POLICY_ID,
            output_semantic="screening_root_seed",
            presets=("paper-screen",),
            random_stream_domain=None,
            sharing_scope="screen_profile_target_manifest",
            domain="wp22_paper_screen_outer_root",
            fields=(
                ("preregistration_checksum", checksum),
                ("screen_execution_profile_checksum", checksum),
                ("screening_target_manifest_checksum", checksum),
            ),
        ),
        SeedDerivationPolicy(
            policy_id=SCREENING_CELL_SEED_POLICY_ID,
            output_semantic="fresh_evaluation_seed",
            applicable_presets=("paper-screen",),
            random_stream_domain="screening_selection",
            sharing_scope="screen_target_optimization_block",
            preimage_kind="named_mapping",
            constant_fields={"domain": "screening_selection"},
            input_fields=(("root_seed", uint64), ("target_instance_id", slug), ("optimization_seed", uint64)),
        ),
        _seed_policy(
            CONFIRMATORY_FRESH_EVALUATION_SEED_POLICY_ID,
            output_semantic="fresh_evaluation_seed",
            presets=("paper-confirm",),
            random_stream_domain="confirmatory_test",
            sharing_scope="target_seed_configuration",
            domain="confirmatory_fresh_evaluation",
            fields=(
                ("final_seal_checksum", checksum),
                ("target_instance_spec_checksum", checksum),
                ("seed_index", uint64),
                ("configuration_checksum", checksum),
            ),
        ),
        _seed_policy(
            PILOT_DIAGNOSTIC_SEED_POLICY_ID,
            output_semantic="diagnostic_trajectory_seed",
            presets=("paper-pilot",),
            random_stream_domain="pilot_evaluation",
            sharing_scope="job_diagnostic_repetition",
            domain="pilot_pathwise_update_diagnostic",
            fields=(
                ("target_manifest_checksum", checksum),
                ("target_instance_spec_checksum", checksum),
                ("optimization_seed", uint64),
                ("publication_candidate_checksum", checksum),
                ("repetition", uint64),
            ),
        ),
        SeedDerivationPolicy(
            policy_id=STAGE_SEED_DERIVATION_POLICY_ID,
            output_semantic="pipeline_stage_seed",
            applicable_presets=all_presets,
            random_stream_domain="coordinate_selected",
            sharing_scope="pipeline_stage_stream",
            preimage_kind="named_mapping",
            constant_fields={"derivation_version": "yaqs.state_preparation.phase2.stage_seed_derivation.v1"},
            input_fields=(
                ("optimization_seed", uint64),
                ("domain_id", seed_role),
                ("binding", slug),
                ("resolution_context_checksum", checksum),
            ),
        ),
        SeedDerivationPolicy(
            policy_id=SCHEDULE_SEED_DERIVATION_POLICY_ID,
            output_semantic="schedule_coordinate_seed",
            applicable_presets=all_presets,
            random_stream_domain="coordinate_selected",
            sharing_scope="schedule_coordinate",
            preimage_kind="named_mapping",
            constant_fields={"derivation_version": SEED_DERIVATION_VERSION},
            input_fields=(
                ("master_seed", uint64),
                ("role", seed_role),
                ("purpose", slug),
                ("stream_index", uint64),
                ("epoch", uint64),
                ("member_index", uint64),
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class ExecutionSeedPolicySuite:
    """Exact ordered universe of every WP22 execution seed derivation."""

    policies: tuple[SeedDerivationPolicy, ...]
    suite_id: str = field(default="wp22_execution_seed_policy_suite", init=False)
    schema_version: str = field(default=EXECUTION_SEED_POLICY_SUITE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require the exact complete reviewed policy identities and contracts.

        Raises:
            TypeError: If an input has an invalid type or container shape.
            ValueError: If an input violates the sealed policy constraints.
        """
        if type(self.policies) is not tuple or any(
            type(policy) is not SeedDerivationPolicy for policy in self.policies
        ):
            msg = "policies must be a tuple of SeedDerivationPolicy records."
            raise TypeError(msg)
        if tuple(policy.policy_id for policy in self.policies) != EXECUTION_SEED_POLICY_IDS:
            msg = "Execution seed policy identities or order changed."
            raise ValueError(msg)
        if self.policies != _reviewed_seed_policies():
            msg = "Execution seed policies differ from the reviewed WP22 universe."
            raise ValueError(msg)

    @classmethod
    @cache
    def frozen(cls) -> ExecutionSeedPolicySuite:
        """Return the exact reviewed seed-policy universe."""
        return cls(policies=_reviewed_seed_policies())

    def policy(self, policy_id: str) -> SeedDerivationPolicy:
        """Return one exact policy by its stable identity.

        Raises:
            KeyError: If the requested policy identifier is unknown.
        """
        checked_id = require_slug(policy_id, "policy_id")
        for policy in self.policies:
            if policy.policy_id == checked_id:
                return policy
        msg = f"Unknown execution seed policy {checked_id!r}."
        raise KeyError(msg)

    def derive(self, policy_id: str, coordinates: Mapping[str, object]) -> int:
        """Derive one seed through a reviewed policy.

        Returns:
            The derived unsigned 64-bit seed.
        """
        return self.policy(policy_id).derive(coordinates)

    def _payload(self) -> dict[str, object]:
        """Return every reviewed policy in frozen order."""
        return {
            "schema_version": self.schema_version,
            "suite_id": self.suite_id,
            "policy_ids": list(EXECUTION_SEED_POLICY_IDS),
            "policies": [policy.to_dict() for policy in self.policies],
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete execution seed-policy suite."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed_dict(self._payload())

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> ExecutionSeedPolicySuite:
        """Decode and verify the exact complete execution seed-policy suite.

        Returns:
            The verified ExecutionSeedPolicySuite record.

        Raises:
            TypeError: If an input has an invalid type or container shape.
            ValueError: If an input violates the sealed policy constraints.
        """
        mapping = _verify_record(
            value,
            expected_keys=_SEED_POLICY_SUITE_KEYS,
            schema_version=EXECUTION_SEED_POLICY_SUITE_SCHEMA_VERSION,
            name="execution seed policy suite",
        )
        if mapping["suite_id"] != "wp22_execution_seed_policy_suite":
            msg = "Execution seed-policy suite identity changed."
            raise ValueError(msg)
        if mapping["policy_ids"] != EXECUTION_SEED_POLICY_IDS:
            msg = "Execution seed-policy identities or order changed."
            raise ValueError(msg)
        raw_policies = mapping["policies"]
        if type(raw_policies) is not tuple:
            msg = "policies must be a JSON array."
            raise TypeError(msg)
        suite = cls(policies=tuple(SeedDerivationPolicy.from_dict(item) for item in raw_policies))
        if mapping["content_checksum"] != suite.content_checksum:
            msg = "Execution seed-policy suite checksum changed during normalization."
            raise ValueError(msg)
        return suite

    @classmethod
    def from_json(cls, payload: str) -> ExecutionSeedPolicySuite:
        """Decode canonical JSON into the reviewed execution seed suite.

        Returns:
            The verified ExecutionSeedPolicySuite record.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def derive_role_seed(
    master_seed: int,
    role: SeedRole,
    *,
    purpose: str,
    stream_index: int = 0,
    epoch: int = 0,
    member_index: int = 0,
) -> int:
    """Derive one stable unsigned seed in a role- and purpose-separated domain.

    Args:
        master_seed: Root unsigned 64-bit seed.
        role: Frozen scientific random-stream role.
        purpose: Stable purpose identifier within the role.
        stream_index: Independent optimizer-start or stream coordinate.
        epoch: Sampling epoch coordinate.
        member_index: Member coordinate within an ensemble.

    Returns:
        A deterministic unsigned 64-bit seed.
    """
    root = _require_uint64(master_seed, "master_seed")
    checked_role = _require_seed_role(role)
    checked_purpose = require_slug(purpose, "purpose")
    stream = require_int(stream_index, "stream_index")
    checked_epoch = require_int(epoch, "epoch")
    member = require_int(member_index, "member_index")
    coordinates = {
        "master_seed": root,
        "role": checked_role,
        "purpose": checked_purpose,
        "stream_index": stream,
        "epoch": checked_epoch,
        "member_index": member,
    }
    return ExecutionSeedPolicySuite.frozen().derive(SCHEDULE_SEED_DERIVATION_POLICY_ID, coordinates)


def derive_map_seed(
    master_seed: int,
    role: TrajectoryRole,
    *,
    stream_index: int = 0,
    epoch: int = 0,
) -> int:
    """Derive a trajectory-map seed independently from every member seed.

    Returns:
        The derived trajectory-map seed.
    """
    return derive_role_seed(
        master_seed,
        _require_trajectory_role(role),
        purpose="trajectory_map",
        stream_index=stream_index,
        epoch=epoch,
    )


def derive_member_seed(
    master_seed: int,
    role: TrajectoryRole,
    *,
    stream_index: int = 0,
    epoch: int = 0,
    member_index: int,
) -> int:
    """Derive one ensemble-member seed in its own role and purpose domain.

    Returns:
        The derived ensemble-member seed.
    """
    return derive_role_seed(
        master_seed,
        _require_trajectory_role(role),
        purpose="trajectory_member",
        stream_index=stream_index,
        epoch=epoch,
        member_index=member_index,
    )


@dataclass(frozen=True, slots=True)
class NoiseStrengthContinuation:
    """Constant or linear noise-strength schedule over update indices."""

    start_update: int
    end_update: int
    target_strength_scale: float
    start_strength_scale: float = 0.0
    interpolation: NoiseInterpolation = "linear_clamped"
    schema_version: str = field(default=NOISE_CONTINUATION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate one exact constant or noiseless-to-target schedule.

        Raises:
            ValueError: If an input violates the sealed policy constraints.
        """
        start = require_int(self.start_update, "start_update")
        end = require_int(self.end_update, "end_update")
        if end < start:
            msg = "end_update must not precede start_update."
            raise ValueError(msg)
        start_strength = require_float(self.start_strength_scale, "start_strength_scale", minimum=0.0)
        target = require_float(self.target_strength_scale, "target_strength_scale", minimum=0.0)
        if self.interpolation not in {"constant", "linear_clamped"}:
            msg = "interpolation must be constant or linear_clamped."
            raise ValueError(msg)
        if self.interpolation == "linear_clamped":
            if end <= start or not math.isclose(start_strength, 0.0, rel_tol=0.0, abs_tol=0.0) or target <= 0.0:
                msg = "Linear continuation requires a nonempty noiseless-to-positive interval."
                raise ValueError(msg)
        elif start_strength != target:
            msg = "A constant noise schedule requires identical start and target strengths."
            raise ValueError(msg)
        object.__setattr__(self, "start_update", start)
        object.__setattr__(self, "end_update", end)
        object.__setattr__(self, "start_strength_scale", start_strength)
        object.__setattr__(self, "target_strength_scale", target)

    def strength_at(self, update: int) -> float:
        """Return the clamped linear strength, with exact declared endpoints."""
        index = require_int(update, "update")
        if self.interpolation == "constant":
            return self.target_strength_scale
        if index <= self.start_update:
            return self.start_strength_scale
        if index >= self.end_update:
            return self.target_strength_scale
        fraction = (index - self.start_update) / (self.end_update - self.start_update)
        return self.target_strength_scale * fraction

    def _payload(self) -> dict[str, object]:
        """Return the unsealed JSON payload."""
        return {
            "schema_version": self.schema_version,
            "start_update": self.start_update,
            "end_update": self.end_update,
            "start_strength_scale": self.start_strength_scale,
            "target_strength_scale": self.target_strength_scale,
            "interpolation": self.interpolation,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the continuation."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return a detached, checksum-sealed JSON mapping."""
        return _sealed_dict(self._payload())

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> NoiseStrengthContinuation:
        """Decode and verify a strict sealed continuation.

        Returns:
            The verified NoiseStrengthContinuation record.
        """
        keys = frozenset({
            "schema_version",
            "start_update",
            "end_update",
            "start_strength_scale",
            "target_strength_scale",
            "interpolation",
            "content_checksum",
        })
        mapping = _verify_record(
            value,
            expected_keys=keys,
            schema_version=NOISE_CONTINUATION_SCHEMA_VERSION,
            name="noise strength continuation",
        )
        return cls(
            start_update=cast("int", mapping["start_update"]),
            end_update=cast("int", mapping["end_update"]),
            target_strength_scale=cast("float", mapping["target_strength_scale"]),
            start_strength_scale=cast("float", mapping["start_strength_scale"]),
            interpolation=cast("NoiseInterpolation", mapping["interpolation"]),
        )

    @classmethod
    def from_json(cls, payload: str) -> NoiseStrengthContinuation:
        """Decode canonical JSON into a verified continuation.

        Returns:
            The verified NoiseStrengthContinuation record.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class TrajectoryCountStep:
    """One inclusive update boundary in a trajectory-count curriculum."""

    start_update: int
    trajectory_count: int
    schema_version: str = field(default=TRAJECTORY_COUNT_STEP_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate nonnegative exact integer coordinates."""
        object.__setattr__(self, "start_update", require_int(self.start_update, "start_update"))
        object.__setattr__(self, "trajectory_count", require_int(self.trajectory_count, "trajectory_count"))

    def _payload(self) -> dict[str, object]:
        """Return the unsealed JSON payload."""
        return {
            "schema_version": self.schema_version,
            "start_update": self.start_update,
            "trajectory_count": self.trajectory_count,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the step."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return a detached, checksum-sealed JSON mapping."""
        return _sealed_dict(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> TrajectoryCountStep:
        """Decode and verify a strict sealed step.

        Returns:
            The verified TrajectoryCountStep record.
        """
        keys = frozenset({"schema_version", "start_update", "trajectory_count", "content_checksum"})
        mapping = _verify_record(
            value,
            expected_keys=keys,
            schema_version=TRAJECTORY_COUNT_STEP_SCHEMA_VERSION,
            name="trajectory count step",
        )
        return cls(
            start_update=cast("int", mapping["start_update"]),
            trajectory_count=cast("int", mapping["trajectory_count"]),
        )


@dataclass(frozen=True, slots=True)
class TrajectoryCountCurriculum:
    """Monotone, inclusive-boundary trajectory-count step schedule."""

    steps: tuple[TrajectoryCountStep, ...]
    schema_version: str = field(default=TRAJECTORY_CURRICULUM_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require an update-zero origin and monotone counts.

        Raises:
            TypeError: If an input has an invalid type or container shape.
            ValueError: If an input violates the sealed policy constraints.
        """
        if type(self.steps) is not tuple or not self.steps:
            msg = "steps must be a nonempty tuple."
            raise TypeError(msg)
        if any(not isinstance(step, TrajectoryCountStep) for step in self.steps):
            msg = "steps must contain only TrajectoryCountStep records."
            raise TypeError(msg)
        if self.steps[0].start_update != 0:
            msg = "The first trajectory-count step must start at update zero."
            raise ValueError(msg)
        starts = tuple(step.start_update for step in self.steps)
        counts = tuple(step.trajectory_count for step in self.steps)
        if any(right <= left for left, right in itertools.pairwise(starts)):
            msg = "Trajectory-count step boundaries must be strictly increasing."
            raise ValueError(msg)
        if any(right < left for left, right in itertools.pairwise(counts)):
            msg = "Trajectory counts must be monotone nondecreasing."
            raise ValueError(msg)
        if len(set(zip(starts, counts, strict=True))) != len(self.steps):
            msg = "Trajectory-count steps must be unique."
            raise ValueError(msg)

    def count_at(self, update: int) -> int:
        """Return the count active at an inclusive step boundary."""
        index = require_int(update, "update")
        result = self.steps[0].trajectory_count
        for step in self.steps[1:]:
            if index < step.start_update:
                break
            result = step.trajectory_count
        return result

    def _payload(self) -> dict[str, object]:
        """Return the unsealed JSON payload."""
        return {
            "schema_version": self.schema_version,
            "boundary_semantics": "inclusive_start_update",
            "steps": [step.to_dict() for step in self.steps],
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the curriculum."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return a detached, checksum-sealed JSON mapping."""
        return _sealed_dict(self._payload())

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> TrajectoryCountCurriculum:
        """Decode and verify a strict sealed curriculum.

        Returns:
            The verified TrajectoryCountCurriculum record.

        Raises:
            TypeError: If an input has an invalid type or container shape.
            ValueError: If an input violates the sealed policy constraints.
        """
        keys = frozenset({"schema_version", "boundary_semantics", "steps", "content_checksum"})
        mapping = _verify_record(
            value,
            expected_keys=keys,
            schema_version=TRAJECTORY_CURRICULUM_SCHEMA_VERSION,
            name="trajectory count curriculum",
        )
        if mapping["boundary_semantics"] != "inclusive_start_update":
            msg = "Trajectory curriculum uses unsupported boundary semantics."
            raise ValueError(msg)
        raw_steps = mapping["steps"]
        if type(raw_steps) is not tuple:
            msg = "trajectory count curriculum steps must be a JSON array."
            raise TypeError(msg)
        return cls(steps=tuple(TrajectoryCountStep.from_dict(step) for step in raw_steps))

    @classmethod
    def from_json(cls, payload: str) -> TrajectoryCountCurriculum:
        """Decode canonical JSON into a verified curriculum.

        Returns:
            The verified TrajectoryCountCurriculum record.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class TrajectorySamplingPolicy:
    """Frozen CRN, refresh, rolling, or independently resampled policy."""

    kind: SamplingKind
    refresh_interval: int | None = None
    retain_count: int | None = None
    retain_fraction: float | None = None
    schema_version: str = field(default=TRAJECTORY_SAMPLING_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate mutually exclusive policy-specific options.

        Raises:
            ValueError: If an input violates the sealed policy constraints.
        """
        if self.kind not in _SAMPLING_KINDS:
            msg = "kind is not a supported trajectory sampling policy."
            raise ValueError(msg)
        refresh = self.refresh_interval
        count = self.retain_count
        fraction = self.retain_fraction
        if self.kind in {"fixed_crn", "resampled"}:
            if refresh is not None or count is not None or fraction is not None:
                msg = f"{self.kind} does not accept refresh or retention options."
                raise ValueError(msg)
            return
        if refresh is None:
            msg = f"{self.kind} requires refresh_interval."
            raise ValueError(msg)
        object.__setattr__(self, "refresh_interval", require_int(refresh, "refresh_interval", minimum=1))
        if self.kind == "periodic_full_refresh":
            if count is not None or fraction is not None:
                msg = "periodic_full_refresh cannot retain trajectories."
                raise ValueError(msg)
            return
        if (count is None) == (fraction is None):
            msg = "rolling_ensemble requires exactly one of retain_count and retain_fraction."
            raise ValueError(msg)
        if count is not None:
            checked_count = require_int(count, "retain_count", minimum=1)
            object.__setattr__(self, "retain_count", checked_count)
        if fraction is not None:
            checked_fraction = require_float(fraction, "retain_fraction", minimum=0.0, maximum=1.0)
            if checked_fraction <= 0.0 or checked_fraction >= 1.0:
                msg = "retain_fraction must be strictly between zero and one."
                raise ValueError(msg)
            object.__setattr__(self, "retain_fraction", checked_fraction)

    def epoch_at(self, update: int) -> int:
        """Return the sampling epoch active at one update."""
        index = require_int(update, "update")
        if self.kind == "fixed_crn":
            return 0
        if self.kind == "resampled":
            return index
        assert self.refresh_interval is not None
        return index // self.refresh_interval

    def retained_count(self, previous_count: int, current_count: int) -> int:
        """Return the exact rolling retention count at a refresh boundary.

        Raises:
            ValueError: If an input violates the sealed policy constraints.
        """
        previous = require_int(previous_count, "previous_count")
        current = require_int(current_count, "current_count")
        if self.kind != "rolling_ensemble":
            msg = "retained_count is defined only for rolling_ensemble."
            raise ValueError(msg)
        if self.retain_count is not None:
            retained = self.retain_count
        else:
            assert self.retain_fraction is not None
            exact = Fraction(str(self.retain_fraction)) * previous
            if exact.denominator != 1:
                msg = "retain_fraction must yield an exact integer for the previous ensemble count."
                raise ValueError(msg)
            retained = exact.numerator
        if retained > min(previous, current):
            msg = "Declared rolling retention exceeds the previous or current ensemble count."
            raise ValueError(msg)
        return retained

    def _payload(self) -> dict[str, object]:
        """Return the unsealed JSON payload."""
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "refresh_interval": self.refresh_interval,
            "retain_count": self.retain_count,
            "retain_fraction": self.retain_fraction,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the sampling policy."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return a detached, checksum-sealed JSON mapping."""
        return _sealed_dict(self._payload())

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> TrajectorySamplingPolicy:
        """Decode and verify a strict sealed sampling policy.

        Returns:
            The verified TrajectorySamplingPolicy record.
        """
        keys = frozenset({
            "schema_version",
            "kind",
            "refresh_interval",
            "retain_count",
            "retain_fraction",
            "content_checksum",
        })
        mapping = _verify_record(
            value,
            expected_keys=keys,
            schema_version=TRAJECTORY_SAMPLING_SCHEMA_VERSION,
            name="trajectory sampling policy",
        )
        return cls(
            kind=cast("SamplingKind", mapping["kind"]),
            refresh_interval=cast("int | None", mapping["refresh_interval"]),
            retain_count=cast("int | None", mapping["retain_count"]),
            retain_fraction=cast("float | None", mapping["retain_fraction"]),
        )

    @classmethod
    def from_json(cls, payload: str) -> TrajectorySamplingPolicy:
        """Decode canonical JSON into a verified sampling policy.

        Returns:
            The verified TrajectorySamplingPolicy record.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class TrajectoryEnsembleMembership:
    """Persistable exact member identity for one update and trajectory role."""

    role: TrajectoryRole
    policy_checksum: str
    stream_index: int
    update: int
    epoch: int
    map_seed: int
    member_seeds: tuple[int, ...]
    retained_member_count: int
    predecessor_checksum: str | None
    schema_version: str = field(default=TRAJECTORY_MEMBERSHIP_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate exact membership cardinality and seed identities.

        Raises:
            TypeError: If an input has an invalid type or container shape.
            ValueError: If an input violates the sealed policy constraints.
        """
        object.__setattr__(self, "role", _require_trajectory_role(self.role))
        object.__setattr__(self, "policy_checksum", require_checksum(self.policy_checksum, "policy_checksum"))
        object.__setattr__(self, "stream_index", require_int(self.stream_index, "stream_index"))
        object.__setattr__(self, "update", require_int(self.update, "update"))
        object.__setattr__(self, "epoch", require_int(self.epoch, "epoch"))
        object.__setattr__(self, "map_seed", _require_uint64(self.map_seed, "map_seed"))
        if type(self.member_seeds) is not tuple:
            msg = "member_seeds must be a tuple."
            raise TypeError(msg)
        members = tuple(_require_uint64(seed, f"member_seeds[{index}]") for index, seed in enumerate(self.member_seeds))
        if len(set(members)) != len(members):
            msg = "member_seeds must be unique within an ensemble."
            raise ValueError(msg)
        if self.map_seed in members:
            msg = "The map seed must be domain-separated from every member seed."
            raise ValueError(msg)
        object.__setattr__(self, "member_seeds", members)
        retained = require_int(self.retained_member_count, "retained_member_count")
        if retained > len(members):
            msg = "retained_member_count cannot exceed ensemble cardinality."
            raise ValueError(msg)
        object.__setattr__(self, "retained_member_count", retained)
        if self.predecessor_checksum is not None:
            object.__setattr__(
                self,
                "predecessor_checksum",
                require_checksum(self.predecessor_checksum, "predecessor_checksum"),
            )
        if self.predecessor_checksum is None and retained != 0:
            msg = "A stream's first membership cannot retain predecessor members."
            raise ValueError(msg)
        if self.update == 0 and self.predecessor_checksum is not None:
            msg = "Update zero cannot reference a predecessor ensemble."
            raise ValueError(msg)

    @property
    def trajectory_count(self) -> int:
        """Exact number of ensemble members."""
        return len(self.member_seeds)

    def _payload(self) -> dict[str, object]:
        """Return the unsealed JSON payload."""
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "policy_checksum": self.policy_checksum,
            "stream_index": self.stream_index,
            "update": self.update,
            "epoch": self.epoch,
            "map_seed": self.map_seed,
            "trajectory_count": self.trajectory_count,
            "member_seeds": list(self.member_seeds),
            "retained_member_count": self.retained_member_count,
            "predecessor_checksum": self.predecessor_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering exact ensemble membership."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return a detached, checksum-sealed JSON mapping."""
        return _sealed_dict(self._payload())

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> TrajectoryEnsembleMembership:
        """Decode and verify exact persisted ensemble membership.

        Returns:
            The verified TrajectoryEnsembleMembership record.

        Raises:
            TypeError: If an input has an invalid type or container shape.
            ValueError: If an input violates the sealed policy constraints.
        """
        keys = frozenset({
            "schema_version",
            "role",
            "policy_checksum",
            "stream_index",
            "update",
            "epoch",
            "map_seed",
            "trajectory_count",
            "member_seeds",
            "retained_member_count",
            "predecessor_checksum",
            "content_checksum",
        })
        mapping = _verify_record(
            value,
            expected_keys=keys,
            schema_version=TRAJECTORY_MEMBERSHIP_SCHEMA_VERSION,
            name="trajectory ensemble membership",
        )
        members = mapping["member_seeds"]
        if type(members) is not tuple:
            msg = "member_seeds must be a JSON array."
            raise TypeError(msg)
        result = cls(
            role=cast("TrajectoryRole", mapping["role"]),
            policy_checksum=cast("str", mapping["policy_checksum"]),
            stream_index=cast("int", mapping["stream_index"]),
            update=cast("int", mapping["update"]),
            epoch=cast("int", mapping["epoch"]),
            map_seed=cast("int", mapping["map_seed"]),
            member_seeds=cast("tuple[int, ...]", members),
            retained_member_count=cast("int", mapping["retained_member_count"]),
            predecessor_checksum=cast("str | None", mapping["predecessor_checksum"]),
        )
        if mapping["trajectory_count"] != result.trajectory_count:
            msg = "trajectory_count does not match member_seeds."
            raise ValueError(msg)
        return result

    @classmethod
    def from_json(cls, payload: str) -> TrajectoryEnsembleMembership:
        """Decode canonical JSON into verified ensemble membership.

        Returns:
            The verified TrajectoryEnsembleMembership record.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def _rank_retained_members(
    member_seeds: tuple[int, ...],
    *,
    master_seed: int,
    role: TrajectoryRole,
    stream_index: int,
    epoch: int,
    retained_count: int,
) -> tuple[int, ...]:
    """Select an exact deterministic subset while preserving predecessor order.

    Returns:
        The deterministically ranked retained member seeds.
    """
    ranked = sorted(
        member_seeds,
        key=lambda seed: derive_role_seed(
            master_seed,
            role,
            purpose="rolling_retention_rank",
            stream_index=stream_index,
            epoch=epoch,
            member_index=seed,
        ),
    )
    selected = frozenset(ranked[:retained_count])
    return tuple(seed for seed in member_seeds if seed in selected)


def build_trajectory_membership(
    policy: TrajectorySamplingPolicy,
    *,
    master_seed: int,
    role: TrajectoryRole,
    update: int,
    trajectory_count: int,
    stream_index: int = 0,
    previous: TrajectoryEnsembleMembership | None = None,
    allow_stream_start: bool = False,
) -> TrajectoryEnsembleMembership:
    """Build exact deterministic membership for one consecutive update.

    A post-zero call must receive the prior persisted membership unless
    ``allow_stream_start`` explicitly marks the first sampled update after a
    zero-count noiseless prefix.  This checksum-binds every later transition.

    Returns:
        The exact map and member seed identity for the requested update.

    Raises:
        TypeError: If an input has an invalid type or container shape.
        ValueError: If an input violates the sealed policy constraints.
    """
    if not isinstance(policy, TrajectorySamplingPolicy):
        msg = "policy must be a TrajectorySamplingPolicy."
        raise TypeError(msg)
    root = _require_uint64(master_seed, "master_seed")
    checked_role = _require_trajectory_role(role)
    index = require_int(update, "update")
    count = require_int(trajectory_count, "trajectory_count")
    stream = require_int(stream_index, "stream_index")
    allow_start = require_bool(allow_stream_start, "allow_stream_start")
    if count == 0:
        msg = "Trajectory sampling requires a positive trajectory_count."
        raise ValueError(msg)
    if index == 0:
        if previous is not None:
            msg = "Update zero cannot receive a previous membership."
            raise ValueError(msg)
    else:
        if previous is None and not allow_start:
            msg = "Every post-zero update requires the immediately preceding membership."
            raise TypeError(msg)
        if previous is not None:
            if not isinstance(previous, TrajectoryEnsembleMembership):
                msg = "previous must be a TrajectoryEnsembleMembership."
                raise TypeError(msg)
            if (
                previous.update != index - 1
                or previous.role != checked_role
                or previous.stream_index != stream
                or previous.policy_checksum != policy.content_checksum
            ):
                msg = "Previous membership does not match the requested consecutive policy stream."
                raise ValueError(msg)

    epoch = policy.epoch_at(index)
    predecessor_checksum = None if previous is None else previous.content_checksum
    retained: tuple[int, ...] = ()
    if previous is not None:
        if epoch == previous.epoch:
            retained = (
                previous.member_seeds
                if count >= previous.trajectory_count
                else _rank_retained_members(
                    previous.member_seeds,
                    master_seed=root,
                    role=checked_role,
                    stream_index=stream,
                    epoch=epoch,
                    retained_count=count,
                )
            )
        elif policy.kind == "rolling_ensemble":
            retain_count = policy.retained_count(previous.trajectory_count, count)
            retained = _rank_retained_members(
                previous.member_seeds,
                master_seed=root,
                role=checked_role,
                stream_index=stream,
                epoch=epoch,
                retained_count=retain_count,
            )
        elif policy.kind == "fixed_crn":
            retained = previous.member_seeds[:count]

    new_count = count - len(retained)
    generated: list[int] = []
    generated_index = 0
    while len(generated) < new_count:
        seed = derive_member_seed(
            root,
            checked_role,
            stream_index=stream,
            epoch=epoch,
            member_index=generated_index,
        )
        generated_index += 1
        if seed not in retained and seed not in generated:
            generated.append(seed)
    members = (*retained, *generated)
    map_seed = derive_map_seed(root, checked_role, stream_index=stream, epoch=epoch)
    return TrajectoryEnsembleMembership(
        role=checked_role,
        policy_checksum=policy.content_checksum,
        stream_index=stream,
        update=index,
        epoch=epoch,
        map_seed=map_seed,
        member_seeds=members,
        retained_member_count=len(retained),
        predecessor_checksum=predecessor_checksum,
    )


@dataclass(frozen=True, slots=True)
class ValidationCheckpoint:
    """One aggregate checkpoint-validation score, never training or test data."""

    update: int
    score: float
    data_role: Literal["checkpoint_validation"] = field(default="checkpoint_validation", init=False)
    schema_version: str = field(default=VALIDATION_CHECKPOINT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate one finite aggregate validation observation."""
        object.__setattr__(self, "update", require_int(self.update, "update"))
        object.__setattr__(self, "score", require_float(self.score, "score"))

    def _payload(self) -> dict[str, object]:
        """Return the unsealed JSON payload."""
        return {
            "schema_version": self.schema_version,
            "data_role": self.data_role,
            "update": self.update,
            "score": self.score,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the validation observation."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return a detached, checksum-sealed JSON mapping."""
        return _sealed_dict(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> ValidationCheckpoint:
        """Decode and verify a validation-only checkpoint score.

        Returns:
            The verified ValidationCheckpoint record.

        Raises:
            ValueError: If an input violates the sealed policy constraints.
        """
        keys = frozenset({"schema_version", "data_role", "update", "score", "content_checksum"})
        mapping = _verify_record(
            value,
            expected_keys=keys,
            schema_version=VALIDATION_CHECKPOINT_SCHEMA_VERSION,
            name="validation checkpoint",
        )
        if mapping["data_role"] != "checkpoint_validation":
            msg = "Checkpoint selection accepts only checkpoint_validation data."
            raise ValueError(msg)
        return cls(update=cast("int", mapping["update"]), score=cast("float", mapping["score"]))


@dataclass(frozen=True, slots=True)
class CheckpointValidationSelection:
    """Best validation checkpoint and validation-only early-stop decision."""

    best_update: int
    best_score: float
    observed_checkpoint_count: int
    stopped_early: bool
    stop_update: int | None
    policy_checksum: str
    schema_version: str = field(default=CHECKPOINT_SELECTION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate selection and stop-coordinate consistency.

        Raises:
            ValueError: If an input violates the sealed policy constraints.
        """
        object.__setattr__(self, "best_update", require_int(self.best_update, "best_update"))
        object.__setattr__(self, "best_score", require_float(self.best_score, "best_score"))
        object.__setattr__(
            self,
            "observed_checkpoint_count",
            require_int(self.observed_checkpoint_count, "observed_checkpoint_count", minimum=1),
        )
        object.__setattr__(self, "stopped_early", require_bool(self.stopped_early, "stopped_early"))
        if self.stopped_early != (self.stop_update is not None):
            msg = "stop_update is present exactly when stopped_early is true."
            raise ValueError(msg)
        if self.stop_update is not None:
            stop = require_int(self.stop_update, "stop_update")
            if stop < self.best_update:
                msg = "stop_update cannot precede best_update."
                raise ValueError(msg)
            object.__setattr__(self, "stop_update", stop)
        object.__setattr__(self, "policy_checksum", require_checksum(self.policy_checksum, "policy_checksum"))

    def _payload(self) -> dict[str, object]:
        """Return the unsealed JSON payload."""
        return {
            "schema_version": self.schema_version,
            "best_update": self.best_update,
            "best_score": self.best_score,
            "observed_checkpoint_count": self.observed_checkpoint_count,
            "stopped_early": self.stopped_early,
            "stop_update": self.stop_update,
            "selection_data_role": "checkpoint_validation",
            "policy_checksum": self.policy_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the selection decision."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return a detached, checksum-sealed JSON mapping."""
        return _sealed_dict(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> CheckpointValidationSelection:
        """Decode and verify a validation-only selection result.

        Returns:
            The verified CheckpointValidationSelection record.

        Raises:
            ValueError: If an input violates the sealed policy constraints.
        """
        keys = frozenset({
            "schema_version",
            "best_update",
            "best_score",
            "observed_checkpoint_count",
            "stopped_early",
            "stop_update",
            "selection_data_role",
            "policy_checksum",
            "content_checksum",
        })
        mapping = _verify_record(
            value,
            expected_keys=keys,
            schema_version=CHECKPOINT_SELECTION_SCHEMA_VERSION,
            name="checkpoint validation selection",
        )
        if mapping["selection_data_role"] != "checkpoint_validation":
            msg = "Checkpoint selection must be validation-only."
            raise ValueError(msg)
        return cls(
            best_update=cast("int", mapping["best_update"]),
            best_score=cast("float", mapping["best_score"]),
            observed_checkpoint_count=cast("int", mapping["observed_checkpoint_count"]),
            stopped_early=cast("bool", mapping["stopped_early"]),
            stop_update=cast("int | None", mapping["stop_update"]),
            policy_checksum=cast("str", mapping["policy_checksum"]),
        )


@dataclass(frozen=True, slots=True)
class CheckpointValidationPolicy:
    """Validation-only early stopping and earliest-tie checkpoint selection."""

    patience: int | None
    min_delta: float = 0.0
    maximize: bool = True
    schema_version: str = field(default=CHECKPOINT_VALIDATION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate optional patience and maximize-only fidelity semantics.

        Raises:
            ValueError: If an input violates the sealed policy constraints.
        """
        if self.patience is not None:
            object.__setattr__(self, "patience", require_int(self.patience, "patience", minimum=1))
        object.__setattr__(self, "min_delta", require_float(self.min_delta, "min_delta", minimum=0.0))
        if not require_bool(self.maximize, "maximize"):
            msg = "WP22 fidelity checkpoint selection must maximize validation score."
            raise ValueError(msg)

    def select(self, checkpoints: Sequence[ValidationCheckpoint]) -> CheckpointValidationSelection:
        """Select the best observed checkpoint and apply validation-only patience.

        Exact score ties retain the earliest checkpoint.  ``min_delta`` affects
        only the patience reset; the best checkpoint remains the exact observed
        maximum before stopping.

        Returns:
            A checksum-bound checkpoint and optional early-stop decision.

        Raises:
            TypeError: If an input has an invalid type or container shape.
            ValueError: If an input violates the sealed policy constraints.
        """
        if not checkpoints:
            msg = "At least one validation checkpoint is required."
            raise ValueError(msg)
        if any(not isinstance(item, ValidationCheckpoint) for item in checkpoints):
            msg = "Checkpoint selection accepts only ValidationCheckpoint records."
            raise TypeError(msg)
        updates = tuple(item.update for item in checkpoints)
        if any(right <= left for left, right in itertools.pairwise(updates)):
            msg = "Validation checkpoint updates must be strictly increasing."
            raise ValueError(msg)
        best = checkpoints[0]
        patience_anchor = checkpoints[0].score
        stale = 0
        observed = 1
        stop_update: int | None = None
        for item in checkpoints[1:]:
            observed += 1
            if item.score > best.score:
                best = item
            if item.score > patience_anchor + self.min_delta:
                patience_anchor = item.score
                stale = 0
            else:
                stale += 1
            if self.patience is not None and stale >= self.patience:
                stop_update = item.update
                break
        return CheckpointValidationSelection(
            best_update=best.update,
            best_score=best.score,
            observed_checkpoint_count=observed,
            stopped_early=stop_update is not None,
            stop_update=stop_update,
            policy_checksum=self.content_checksum,
        )

    def _payload(self) -> dict[str, object]:
        """Return the unsealed JSON payload."""
        return {
            "schema_version": self.schema_version,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "maximize": self.maximize,
            "selection_data_role": "checkpoint_validation",
            "tie_rule": "earliest_update",
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering checkpoint semantics."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return a detached, checksum-sealed JSON mapping."""
        return _sealed_dict(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> CheckpointValidationPolicy:
        """Decode and verify a strict validation-only checkpoint policy.

        Returns:
            The verified CheckpointValidationPolicy record.

        Raises:
            ValueError: If an input violates the sealed policy constraints.
        """
        keys = frozenset({
            "schema_version",
            "patience",
            "min_delta",
            "maximize",
            "selection_data_role",
            "tie_rule",
            "content_checksum",
        })
        mapping = _verify_record(
            value,
            expected_keys=keys,
            schema_version=CHECKPOINT_VALIDATION_SCHEMA_VERSION,
            name="checkpoint validation policy",
        )
        if mapping["selection_data_role"] != "checkpoint_validation" or mapping["tie_rule"] != "earliest_update":
            msg = "Checkpoint policy must select validation-only scores with the earliest-tie rule."
            raise ValueError(msg)
        return cls(
            patience=cast("int | None", mapping["patience"]),
            min_delta=cast("float", mapping["min_delta"]),
            maximize=cast("bool", mapping["maximize"]),
        )


@dataclass(frozen=True, slots=True)
class CheckpointValidationTracker:
    """Immutable resumable state for validation-only checkpoint decisions."""

    policy: CheckpointValidationPolicy
    checkpoints: tuple[ValidationCheckpoint, ...] = ()
    schema_version: str = field(default=CHECKPOINT_TRACKER_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate exact record types, update order, and terminal state.

        Raises:
            TypeError: If an input has an invalid type or container shape.
            ValueError: If an input violates the sealed policy constraints.
        """
        if not isinstance(self.policy, CheckpointValidationPolicy):
            msg = "policy must be a CheckpointValidationPolicy."
            raise TypeError(msg)
        if type(self.checkpoints) is not tuple or any(
            not isinstance(checkpoint, ValidationCheckpoint) for checkpoint in self.checkpoints
        ):
            msg = "checkpoints must be a tuple of ValidationCheckpoint records."
            raise TypeError(msg)
        updates = tuple(checkpoint.update for checkpoint in self.checkpoints)
        if any(right <= left for left, right in itertools.pairwise(updates)):
            msg = "Validation checkpoint updates must be strictly increasing."
            raise ValueError(msg)
        if self.checkpoints:
            selection = self.policy.select(self.checkpoints)
            if selection.stopped_early and selection.observed_checkpoint_count != len(self.checkpoints):
                msg = "Tracker checkpoints cannot continue after validation-only early stopping."
                raise ValueError(msg)

    @property
    def selection(self) -> CheckpointValidationSelection | None:
        """Current validation-only selection, or ``None`` before observation."""
        if not self.checkpoints:
            return None
        return self.policy.select(self.checkpoints)

    @property
    def should_stop(self) -> bool:
        """Whether validation-only patience has reached its stopping boundary."""
        return self.selection is not None and self.selection.stopped_early

    def observe(self, checkpoint: ValidationCheckpoint) -> CheckpointValidationTracker:
        """Return a new tracker after one aggregate validation observation.

        Raises:
            TypeError: If an input has an invalid type or container shape.
            ValueError: If an input violates the sealed policy constraints.
        """
        if not isinstance(checkpoint, ValidationCheckpoint):
            msg = "checkpoint must be a ValidationCheckpoint."
            raise TypeError(msg)
        if self.should_stop:
            msg = "Cannot add checkpoints after validation-only early stopping."
            raise ValueError(msg)
        return CheckpointValidationTracker(self.policy, (*self.checkpoints, checkpoint))

    def _payload(self) -> dict[str, object]:
        """Return the unsealed JSON payload."""
        return {
            "schema_version": self.schema_version,
            "policy": self.policy.to_dict(),
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints],
            "selection": None if self.selection is None else self.selection.to_dict(),
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering policy, observations, and derived selection."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return a detached, checksum-sealed JSON mapping."""
        return _sealed_dict(self._payload())

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> CheckpointValidationTracker:
        """Decode and verify strict resumable validation tracker state.

        Returns:
            The verified CheckpointValidationTracker record.

        Raises:
            TypeError: If an input has an invalid type or container shape.
            ValueError: If an input violates the sealed policy constraints.
        """
        keys = frozenset({"schema_version", "policy", "checkpoints", "selection", "content_checksum"})
        mapping = _verify_record(
            value,
            expected_keys=keys,
            schema_version=CHECKPOINT_TRACKER_SCHEMA_VERSION,
            name="checkpoint validation tracker",
        )
        checkpoints = mapping["checkpoints"]
        if type(checkpoints) is not tuple:
            msg = "checkpoints must be a JSON array."
            raise TypeError(msg)
        tracker = cls(
            policy=CheckpointValidationPolicy.from_dict(mapping["policy"]),
            checkpoints=tuple(ValidationCheckpoint.from_dict(checkpoint) for checkpoint in checkpoints),
        )
        expected_selection = None if tracker.selection is None else tracker.selection.to_dict()
        if mapping["selection"] != expected_selection:
            msg = "Serialized checkpoint selection is inconsistent with validation observations."
            raise ValueError(msg)
        return tracker

    @classmethod
    def from_json(cls, payload: str) -> CheckpointValidationTracker:
        """Decode canonical JSON into verified resumable tracker state.

        Returns:
            The verified CheckpointValidationTracker record.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class NoiselessPretrainNoisyFinetune:
    """Exact optional boundary between noiseless and noisy training."""

    noiseless_pretrain_updates: int
    noisy_finetune_updates: int
    schema_version: str = field(default=TRAINING_PHASE_BOUNDARY_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require at least one update across the two optional phases.

        Raises:
            ValueError: If an input violates the sealed policy constraints.
        """
        object.__setattr__(
            self,
            "noiseless_pretrain_updates",
            require_int(self.noiseless_pretrain_updates, "noiseless_pretrain_updates"),
        )
        object.__setattr__(
            self,
            "noisy_finetune_updates",
            require_int(self.noisy_finetune_updates, "noisy_finetune_updates"),
        )
        if self.total_updates == 0:
            msg = "At least one noiseless or noisy training update is required."
            raise ValueError(msg)

    @property
    def mode(self) -> Literal["noiseless_only", "noisy_only", "pretrain_then_finetune"]:
        """Derived composition mode for direct controls and two-phase training."""
        if self.noisy_finetune_updates == 0:
            return "noiseless_only"
        if self.noiseless_pretrain_updates == 0:
            return "noisy_only"
        return "pretrain_then_finetune"

    @property
    def noisy_start_update(self) -> int:
        """First update assigned to noisy fine-tuning."""
        return self.noiseless_pretrain_updates

    @property
    def total_updates(self) -> int:
        """Complete pretraining and fine-tuning budget."""
        return self.noiseless_pretrain_updates + self.noisy_finetune_updates

    def phase_at(self, update: int) -> Literal["noiseless_pretrain", "noisy_finetune"]:
        """Return the phase for one in-budget update index.

        Raises:
            ValueError: If an input violates the sealed policy constraints.
        """
        index = require_int(update, "update")
        if index >= self.total_updates:
            msg = "update lies outside the declared training budget."
            raise ValueError(msg)
        return "noiseless_pretrain" if index < self.noisy_start_update else "noisy_finetune"

    def _payload(self) -> dict[str, object]:
        """Return the unsealed JSON payload."""
        return {
            "schema_version": self.schema_version,
            "noiseless_pretrain_updates": self.noiseless_pretrain_updates,
            "noisy_finetune_updates": self.noisy_finetune_updates,
            "noisy_start_update": self.noisy_start_update,
            "total_updates": self.total_updates,
            "mode": self.mode,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the phase boundary."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return a detached, checksum-sealed JSON mapping."""
        return _sealed_dict(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> NoiselessPretrainNoisyFinetune:
        """Decode and verify a strict phase boundary.

        Returns:
            The verified NoiselessPretrainNoisyFinetune record.

        Raises:
            ValueError: If an input violates the sealed policy constraints.
        """
        keys = frozenset({
            "schema_version",
            "noiseless_pretrain_updates",
            "noisy_finetune_updates",
            "noisy_start_update",
            "total_updates",
            "mode",
            "content_checksum",
        })
        mapping = _verify_record(
            value,
            expected_keys=keys,
            schema_version=TRAINING_PHASE_BOUNDARY_SCHEMA_VERSION,
            name="noiseless pretrain noisy finetune boundary",
        )
        result = cls(
            noiseless_pretrain_updates=cast("int", mapping["noiseless_pretrain_updates"]),
            noisy_finetune_updates=cast("int", mapping["noisy_finetune_updates"]),
        )
        if (
            mapping["noisy_start_update"] != result.noisy_start_update
            or mapping["total_updates"] != result.total_updates
            or mapping["mode"] != result.mode
        ):
            msg = "Serialized phase-boundary derived fields are inconsistent."
            raise ValueError(msg)
        return result


@dataclass(frozen=True, slots=True)
class MultistartSeedBundle:
    """Three independent seed domains for one bounded optimizer start."""

    start_index: int
    initialization_seed: int
    optimizer_ordering_seed: int
    training_trajectory_seed: int
    schema_version: str = field(default=MULTISTART_SEEDS_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate coordinates, uint64 values, and within-start isolation.

        Raises:
            ValueError: If an input violates the sealed policy constraints.
        """
        object.__setattr__(self, "start_index", require_int(self.start_index, "start_index"))
        names = ("initialization_seed", "optimizer_ordering_seed", "training_trajectory_seed")
        values = tuple(_require_uint64(getattr(self, name), name) for name in names)
        if len(set(values)) != len(values):
            msg = "Initialization, optimizer, and training seeds must be isolated."
            raise ValueError(msg)
        for name, value in zip(names, values, strict=True):
            object.__setattr__(self, name, value)

    def _payload(self) -> dict[str, object]:
        """Return the unsealed JSON payload."""
        return {
            "schema_version": self.schema_version,
            "start_index": self.start_index,
            "initialization_seed": self.initialization_seed,
            "optimizer_ordering_seed": self.optimizer_ordering_seed,
            "training_trajectory_seed": self.training_trajectory_seed,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the seed bundle."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return a detached, checksum-sealed JSON mapping."""
        return _sealed_dict(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> MultistartSeedBundle:
        """Decode and verify one strict multistart seed bundle.

        Returns:
            The verified MultistartSeedBundle record.
        """
        keys = frozenset({
            "schema_version",
            "start_index",
            "initialization_seed",
            "optimizer_ordering_seed",
            "training_trajectory_seed",
            "content_checksum",
        })
        mapping = _verify_record(
            value,
            expected_keys=keys,
            schema_version=MULTISTART_SEEDS_SCHEMA_VERSION,
            name="multistart seed bundle",
        )
        return cls(
            start_index=cast("int", mapping["start_index"]),
            initialization_seed=cast("int", mapping["initialization_seed"]),
            optimizer_ordering_seed=cast("int", mapping["optimizer_ordering_seed"]),
            training_trajectory_seed=cast("int", mapping["training_trajectory_seed"]),
        )


@dataclass(frozen=True, slots=True)
class LimitedMultistartPlan:
    """Predeclared limited multistart count with isolated random domains."""

    start_count: int
    declared_cap: int
    schema_version: str = field(default=MULTISTART_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Enforce both the declared and implementation-level caps.

        Raises:
            ValueError: If an input violates the sealed policy constraints.
        """
        count = require_int(self.start_count, "start_count", minimum=1)
        cap = require_int(self.declared_cap, "declared_cap", minimum=1)
        if cap > MAX_MULTISTART_COUNT:
            msg = f"declared_cap cannot exceed the WP22 maximum of {MAX_MULTISTART_COUNT}."
            raise ValueError(msg)
        if count > cap:
            msg = "start_count cannot exceed declared_cap."
            raise ValueError(msg)
        object.__setattr__(self, "start_count", count)
        object.__setattr__(self, "declared_cap", cap)

    def seed_bundles(self, master_seed: int) -> tuple[MultistartSeedBundle, ...]:
        """Return globally isolated initialization, optimizer, and training seeds.

        Raises:
            RuntimeError: If the derived seed bundles are not globally isolated.
        """
        bundles = tuple(
            MultistartSeedBundle(
                start_index=index,
                initialization_seed=derive_role_seed(
                    master_seed,
                    "initialization",
                    purpose="multistart",
                    stream_index=index,
                ),
                optimizer_ordering_seed=derive_role_seed(
                    master_seed,
                    "optimizer_ordering",
                    purpose="multistart",
                    stream_index=index,
                ),
                training_trajectory_seed=derive_role_seed(
                    master_seed,
                    "training_trajectory",
                    purpose="multistart",
                    stream_index=index,
                ),
            )
            for index in range(self.start_count)
        )
        all_seeds = [
            seed
            for bundle in bundles
            for seed in (bundle.initialization_seed, bundle.optimizer_ordering_seed, bundle.training_trajectory_seed)
        ]
        if len(set(all_seeds)) != len(all_seeds):
            msg = "Derived multistart random streams unexpectedly collided."
            raise RuntimeError(msg)
        return bundles

    def _payload(self) -> dict[str, object]:
        """Return the unsealed JSON payload."""
        return {
            "schema_version": self.schema_version,
            "start_count": self.start_count,
            "declared_cap": self.declared_cap,
            "implementation_cap": MAX_MULTISTART_COUNT,
            "seed_roles": ["initialization", "optimizer_ordering", "training_trajectory"],
            "selection_data_role": "checkpoint_validation",
            "selection_rule": "greatest_validation_fidelity",
            "tie_rules": ["earliest_checkpoint", "lowest_start_index"],
            "work_accounting": "all_starts",
            "promotion_work_rule": "complete_work_within_separately_sealed_cap",
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the multistart plan."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return a detached, checksum-sealed JSON mapping."""
        return _sealed_dict(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> LimitedMultistartPlan:
        """Decode and verify a strict limited multistart plan.

        Returns:
            The verified LimitedMultistartPlan record.

        Raises:
            ValueError: If an input violates the sealed policy constraints.
        """
        keys = frozenset({
            "schema_version",
            "start_count",
            "declared_cap",
            "implementation_cap",
            "seed_roles",
            "selection_data_role",
            "selection_rule",
            "tie_rules",
            "work_accounting",
            "promotion_work_rule",
            "content_checksum",
        })
        mapping = _verify_record(
            value,
            expected_keys=keys,
            schema_version=MULTISTART_SCHEMA_VERSION,
            name="limited multistart plan",
        )
        expected = {
            "implementation_cap": MAX_MULTISTART_COUNT,
            "seed_roles": ("initialization", "optimizer_ordering", "training_trajectory"),
            "selection_data_role": "checkpoint_validation",
            "selection_rule": "greatest_validation_fidelity",
            "tie_rules": ("earliest_checkpoint", "lowest_start_index"),
            "work_accounting": "all_starts",
            "promotion_work_rule": "complete_work_within_separately_sealed_cap",
        }
        if any(mapping[name] != value for name, value in expected.items()):
            msg = "Multistart implementation cap or random-stream roles are not frozen WP22 values."
            raise ValueError(msg)
        return cls(
            start_count=cast("int", mapping["start_count"]),
            declared_cap=cast("int", mapping["declared_cap"]),
        )


@dataclass(frozen=True, slots=True)
class NoiseMixtureComponent:
    """One ordered standard-noise profile and its exact declared weight."""

    noise_id: str
    weight: float
    schema_version: str = field(default=NOISE_MIXTURE_COMPONENT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require a registered standard profile and strictly positive weight.

        Raises:
            ValueError: If an input violates the sealed policy constraints.
        """
        noise_id = require_slug(self.noise_id, "noise_id")
        if noise_id not in STANDARD_NOISE_IDS:
            msg = f"noise_id {noise_id!r} is not a standard benchmark noise profile."
            raise ValueError(msg)
        weight = require_float(self.weight, "weight", minimum=0.0, maximum=1.0)
        if weight <= 0.0:
            msg = "weight must be strictly positive."
            raise ValueError(msg)
        object.__setattr__(self, "noise_id", noise_id)
        object.__setattr__(self, "weight", weight)

    def _payload(self) -> dict[str, object]:
        """Return the unsealed JSON payload."""
        return {"schema_version": self.schema_version, "noise_id": self.noise_id, "weight": self.weight}

    @property
    def content_checksum(self) -> str:
        """Checksum covering this ordered component."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return a detached, checksum-sealed JSON mapping."""
        return _sealed_dict(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> NoiseMixtureComponent:
        """Decode and verify one strict mixture component.

        Returns:
            The verified NoiseMixtureComponent record.
        """
        keys = frozenset({"schema_version", "noise_id", "weight", "content_checksum"})
        mapping = _verify_record(
            value,
            expected_keys=keys,
            schema_version=NOISE_MIXTURE_COMPONENT_SCHEMA_VERSION,
            name="noise mixture component",
        )
        return cls(noise_id=cast("str", mapping["noise_id"]), weight=cast("float", mapping["weight"]))


@dataclass(frozen=True, slots=True)
class NoiseMixtureAllocation:
    """Exact largest-remainder trajectory allocation across noise components."""

    mixture_checksum: str
    trajectory_count: int
    component_ids: tuple[str, ...]
    component_counts: tuple[int, ...]
    schema_version: str = field(default=NOISE_MIXTURE_ALLOCATION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate component order, counts, and complete allocation.

        Raises:
            TypeError: If an input has an invalid type or container shape.
            ValueError: If an input violates the sealed policy constraints.
        """
        object.__setattr__(
            self,
            "mixture_checksum",
            require_checksum(self.mixture_checksum, "mixture_checksum"),
        )
        total = require_int(self.trajectory_count, "trajectory_count")
        if type(self.component_ids) is not tuple or type(self.component_counts) is not tuple:
            msg = "component_ids and component_counts must be tuples."
            raise TypeError(msg)
        identities = tuple(require_slug(value, "component_id") for value in self.component_ids)
        counts = tuple(require_int(value, "component_count") for value in self.component_counts)
        if len(identities) != len(counts) or len(set(identities)) != len(identities):
            msg = "Mixture allocation components must be aligned and unique."
            raise ValueError(msg)
        if math.fsum(counts) != total:
            msg = "Mixture component counts must exhaust trajectory_count."
            raise ValueError(msg)
        object.__setattr__(self, "trajectory_count", total)
        object.__setattr__(self, "component_ids", identities)
        object.__setattr__(self, "component_counts", counts)

    @property
    def component_seed_domains(self) -> tuple[str, ...]:
        """Stable component-local training-trajectory seed-domain labels."""
        return tuple(f"training_trajectory.mixture.{component_id}" for component_id in self.component_ids)

    def _payload(self) -> dict[str, object]:
        """Return the unsealed JSON payload."""
        return {
            "schema_version": self.schema_version,
            "allocation_rule": "largest_remainder_declared_order_ties",
            "mixture_checksum": self.mixture_checksum,
            "trajectory_count": self.trajectory_count,
            "component_ids": list(self.component_ids),
            "component_counts": list(self.component_counts),
            "component_seed_domains": list(self.component_seed_domains),
            "membership_persistence": "component_local_with_predecessor_checksum",
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the exact component allocation."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return a detached, checksum-sealed JSON mapping."""
        return _sealed_dict(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> NoiseMixtureAllocation:
        """Decode and verify one exact component allocation.

        Returns:
            The verified NoiseMixtureAllocation record.

        Raises:
            TypeError: If an input has an invalid type or container shape.
            ValueError: If an input violates the sealed policy constraints.
        """
        keys = frozenset({
            "schema_version",
            "allocation_rule",
            "mixture_checksum",
            "trajectory_count",
            "component_ids",
            "component_counts",
            "component_seed_domains",
            "membership_persistence",
            "content_checksum",
        })
        mapping = _verify_record(
            value,
            expected_keys=keys,
            schema_version=NOISE_MIXTURE_ALLOCATION_SCHEMA_VERSION,
            name="noise mixture allocation",
        )
        if mapping["allocation_rule"] != "largest_remainder_declared_order_ties":
            msg = "Noise mixture allocation uses an unsupported rule."
            raise ValueError(msg)
        raw_ids = mapping["component_ids"]
        raw_counts = mapping["component_counts"]
        if type(raw_ids) is not tuple or type(raw_counts) is not tuple:
            msg = "Noise mixture allocation components must be JSON arrays."
            raise TypeError(msg)
        allocation = cls(
            mixture_checksum=cast("str", mapping["mixture_checksum"]),
            trajectory_count=cast("int", mapping["trajectory_count"]),
            component_ids=cast("tuple[str, ...]", raw_ids),
            component_counts=cast("tuple[int, ...]", raw_counts),
        )
        if (
            mapping["component_seed_domains"] != allocation.component_seed_domains
            or mapping["membership_persistence"] != "component_local_with_predecessor_checksum"
        ):
            msg = "Noise mixture component seed domains or persistence rule changed."
            raise ValueError(msg)
        return allocation


@dataclass(frozen=True, slots=True)
class StandardNoiseMixture:
    """Noiseless, matched-noise, or frozen-mixture training condition."""

    mode: TrainingNoiseMode
    components: tuple[NoiseMixtureComponent, ...]
    schema_version: str = field(default=STANDARD_NOISE_MIXTURE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require the exact component cardinality and normalized weights.

        Raises:
            TypeError: If an input has an invalid type or container shape.
            ValueError: If an input violates the sealed policy constraints.
        """
        if self.mode not in {"noiseless", "matched", "frozen_mixture"}:
            msg = "mode must be noiseless, matched, or frozen_mixture."
            raise ValueError(msg)
        if type(self.components) is not tuple:
            msg = "components must be a tuple."
            raise TypeError(msg)
        if any(not isinstance(component, NoiseMixtureComponent) for component in self.components):
            msg = "components must contain only NoiseMixtureComponent records."
            raise TypeError(msg)
        if self.mode == "noiseless":
            if self.components:
                msg = "noiseless training cannot declare standard-noise components."
                raise ValueError(msg)
            return
        if not self.components:
            msg = "Noisy training requires at least one standard-noise component."
            raise ValueError(msg)
        identities = tuple(component.noise_id for component in self.components)
        if len(set(identities)) != len(identities):
            msg = "Standard-noise mixture components must have unique identifiers."
            raise ValueError(msg)
        if not math.isclose(
            math.fsum(component.weight for component in self.components),
            1.0,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            msg = "Standard-noise mixture weights must sum exactly to 1.0."
            raise ValueError(msg)
        if self.mode == "matched" and (
            len(self.components) != 1 or not math.isclose(self.components[0].weight, 1.0, rel_tol=0.0, abs_tol=0.0)
        ):
            msg = "matched training requires exactly one unit-weight standard-noise component."
            raise ValueError(msg)
        if self.mode == "frozen_mixture" and len(self.components) < 2:
            msg = "frozen_mixture requires at least two ordered standard-noise components."
            raise ValueError(msg)

    def _payload(self) -> dict[str, object]:
        """Return the unsealed JSON payload without reordering components."""
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "components": [component.to_dict() for component in self.components],
            "weight_sum": 0.0 if self.mode == "noiseless" else 1.0,
            "ordering_rule": "declared_order",
            "allocation_rule": "largest_remainder_declared_order_ties",
            "component_evaluation_rule": "separate_component_means",
            "combination_rule": "exact_declared_weights",
        }

    def allocate(self, trajectory_count: int) -> NoiseMixtureAllocation:
        """Allocate a fixed trajectory count by deterministic largest remainder.

        Returns:
            The checksum-sealed component allocation.

        Raises:
            ValueError: If a noisy allocation is empty or cannot represent every
                positive-weight frozen-mixture component.
        """
        total = require_int(trajectory_count, "trajectory_count")
        if self.mode == "noiseless":
            if total != 0:
                msg = "Noiseless training cannot allocate noisy trajectories."
                raise ValueError(msg)
            counts: tuple[int, ...] = ()
        else:
            if total == 0:
                msg = "Noisy training requires a positive trajectory allocation."
                raise ValueError(msg)
            exact = tuple(Fraction(str(component.weight)) * total for component in self.components)
            floors = [value.numerator // value.denominator for value in exact]
            remainder = total - sum(floors)
            ranked = sorted(
                range(len(exact)),
                key=lambda index: (-(exact[index] - floors[index]), index),
            )
            for index in ranked[:remainder]:
                floors[index] += 1
            counts = tuple(floors)
            if self.mode == "frozen_mixture" and any(count == 0 for count in counts):
                msg = "Production frozen mixtures require at least one trajectory per component."
                raise ValueError(msg)
        return NoiseMixtureAllocation(
            mixture_checksum=self.content_checksum,
            trajectory_count=total,
            component_ids=tuple(component.noise_id for component in self.components),
            component_counts=counts,
        )

    @property
    def content_checksum(self) -> str:
        """Checksum covering order, identities, and exact weights."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return a detached, checksum-sealed JSON mapping."""
        return _sealed_dict(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> StandardNoiseMixture:
        """Decode and verify a strict ordered standard-noise mixture.

        Returns:
            The verified StandardNoiseMixture record.

        Raises:
            TypeError: If an input has an invalid type or container shape.
            ValueError: If an input violates the sealed policy constraints.
        """
        keys = frozenset({
            "schema_version",
            "mode",
            "components",
            "weight_sum",
            "ordering_rule",
            "allocation_rule",
            "component_evaluation_rule",
            "combination_rule",
            "content_checksum",
        })
        mapping = _verify_record(
            value,
            expected_keys=keys,
            schema_version=STANDARD_NOISE_MIXTURE_SCHEMA_VERSION,
            name="standard noise mixture",
        )
        weight_sum = require_float(mapping["weight_sum"], "weight_sum")
        mode = cast("TrainingNoiseMode", mapping["mode"])
        expected_sum = 0.0 if mode == "noiseless" else 1.0
        if (
            not math.isclose(weight_sum, expected_sum, rel_tol=0.0, abs_tol=0.0)
            or mapping["ordering_rule"] != "declared_order"
            or mapping["allocation_rule"] != "largest_remainder_declared_order_ties"
            or mapping["component_evaluation_rule"] != "separate_component_means"
            or mapping["combination_rule"] != "exact_declared_weights"
        ):
            msg = "Standard-noise mixture derived fields are not frozen WP22 values."
            raise ValueError(msg)
        components = mapping["components"]
        if type(components) is not tuple:
            msg = "components must be a JSON array."
            raise TypeError(msg)
        return cls(
            mode=mode,
            components=tuple(NoiseMixtureComponent.from_dict(component) for component in components),
        )


@dataclass(frozen=True, slots=True)
class TrainingStrategySchedule:
    """Checksum-bearing composition of every WP22 training strategy choice."""

    schedule_id: str
    noise_continuation: NoiseStrengthContinuation
    trajectory_curriculum: TrajectoryCountCurriculum
    sampling_policy: TrajectorySamplingPolicy
    checkpoint_validation: CheckpointValidationPolicy
    phase_boundary: NoiselessPretrainNoisyFinetune
    multistart: LimitedMultistartPlan
    training_noise: StandardNoiseMixture
    schema_version: str = field(default=TRAINING_STRATEGY_SCHEDULE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate component types and their cross-schedule boundaries.

        Raises:
            TypeError: If an input has an invalid type or container shape.
            ValueError: If an input violates the sealed policy constraints.
        """
        object.__setattr__(self, "schedule_id", require_slug(self.schedule_id, "schedule_id"))
        component_types: tuple[tuple[str, object, type[object]], ...] = (
            ("noise_continuation", self.noise_continuation, NoiseStrengthContinuation),
            ("trajectory_curriculum", self.trajectory_curriculum, TrajectoryCountCurriculum),
            ("sampling_policy", self.sampling_policy, TrajectorySamplingPolicy),
            ("checkpoint_validation", self.checkpoint_validation, CheckpointValidationPolicy),
            ("phase_boundary", self.phase_boundary, NoiselessPretrainNoisyFinetune),
            ("multistart", self.multistart, LimitedMultistartPlan),
            ("training_noise", self.training_noise, StandardNoiseMixture),
        )
        for name, value, expected_type in component_types:
            if not isinstance(value, expected_type):
                msg = f"{name} must be a {expected_type.__name__}."
                raise TypeError(msg)
        if self.noise_continuation.end_update >= self.phase_boundary.total_updates:
            msg = "Noise continuation must reach its target within the training budget."
            raise ValueError(msg)
        mode = self.phase_boundary.mode
        if mode == "pretrain_then_finetune":
            if (
                self.noise_continuation.interpolation != "linear_clamped"
                or self.noise_continuation.start_update != self.phase_boundary.noisy_start_update
                or self.noise_continuation.end_update < self.noise_continuation.start_update
                or self.training_noise.mode == "noiseless"
            ):
                msg = "Two-phase training requires continuation from its exact noisy boundary."
                raise ValueError(msg)
        elif mode == "noisy_only":
            if (
                self.noise_continuation.start_update != 0
                or self.noise_continuation.target_strength_scale <= 0.0
                or self.training_noise.mode == "noiseless"
            ):
                msg = "Direct noisy training requires a positive standard-noise schedule from update zero."
                raise ValueError(msg)
        elif (
            self.noise_continuation.interpolation != "constant"
            or self.noise_continuation.start_update != 0
            or not math.isclose(
                self.noise_continuation.target_strength_scale,
                0.0,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or self.training_noise.mode != "noiseless"
        ):
            msg = "A noiseless-only control requires a zero constant schedule and no noise components."
            raise ValueError(msg)
        if any(step.start_update >= self.phase_boundary.total_updates for step in self.trajectory_curriculum.steps):
            msg = "Trajectory curriculum contains a boundary outside the training budget."
            raise ValueError(msg)
        if mode == "noiseless_only" and any(step.trajectory_count != 0 for step in self.trajectory_curriculum.steps):
            msg = "A noiseless-only control cannot allocate training trajectories."
            raise ValueError(msg)
        if (
            mode != "noiseless_only"
            and self.trajectory_curriculum.count_at(self.phase_boundary.noisy_start_update) <= 0
        ):
            msg = "Noisy fine-tuning must begin with a positive trajectory count."
            raise ValueError(msg)

    def trajectory_membership_at(
        self,
        *,
        master_seed: int,
        role: TrajectoryRole,
        update: int,
        stream_index: int = 0,
    ) -> TrajectoryEnsembleMembership:
        """Reconstruct exact ensemble membership through an arbitrary resume point.

        Returns:
            Membership identical to uninterrupted consecutive execution.

        Raises:
            ValueError: If an input violates the sealed policy constraints.
        """
        final_update = require_int(update, "update")
        if final_update >= self.phase_boundary.total_updates:
            msg = "update lies outside the declared training budget."
            raise ValueError(msg)
        previous: TrajectoryEnsembleMembership | None = None
        for index in range(final_update + 1):
            count = self.trajectory_curriculum.count_at(index)
            if count <= 0:
                continue
            previous = build_trajectory_membership(
                self.sampling_policy,
                master_seed=master_seed,
                role=role,
                update=index,
                trajectory_count=count,
                stream_index=stream_index,
                previous=previous,
                allow_stream_start=previous is None,
            )
        if previous is None or previous.update != final_update:
            msg = "Trajectory membership is unavailable during a zero-count noiseless stage."
            raise ValueError(msg)
        return previous

    def _payload(self) -> dict[str, object]:
        """Return the unsealed JSON payload."""
        return {
            "schema_version": self.schema_version,
            "schedule_id": self.schedule_id,
            "noise_continuation": self.noise_continuation.to_dict(),
            "trajectory_curriculum": self.trajectory_curriculum.to_dict(),
            "sampling_policy": self.sampling_policy.to_dict(),
            "checkpoint_validation": self.checkpoint_validation.to_dict(),
            "phase_boundary": self.phase_boundary.to_dict(),
            "multistart": self.multistart.to_dict(),
            "training_noise": self.training_noise.to_dict(),
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering every composed strategy choice."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return a detached, checksum-sealed JSON mapping."""
        return _sealed_dict(self._payload())

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> TrainingStrategySchedule:
        """Decode and verify a complete strict WP22 training schedule.

        Returns:
            The verified TrainingStrategySchedule record.
        """
        keys = frozenset({
            "schema_version",
            "schedule_id",
            "noise_continuation",
            "trajectory_curriculum",
            "sampling_policy",
            "checkpoint_validation",
            "phase_boundary",
            "multistart",
            "training_noise",
            "content_checksum",
        })
        mapping = _verify_record(
            value,
            expected_keys=keys,
            schema_version=TRAINING_STRATEGY_SCHEDULE_SCHEMA_VERSION,
            name="training strategy schedule",
        )
        return cls(
            schedule_id=cast("str", mapping["schedule_id"]),
            noise_continuation=NoiseStrengthContinuation.from_dict(mapping["noise_continuation"]),
            trajectory_curriculum=TrajectoryCountCurriculum.from_dict(mapping["trajectory_curriculum"]),
            sampling_policy=TrajectorySamplingPolicy.from_dict(mapping["sampling_policy"]),
            checkpoint_validation=CheckpointValidationPolicy.from_dict(mapping["checkpoint_validation"]),
            phase_boundary=NoiselessPretrainNoisyFinetune.from_dict(mapping["phase_boundary"]),
            multistart=LimitedMultistartPlan.from_dict(mapping["multistart"]),
            training_noise=StandardNoiseMixture.from_dict(mapping["training_noise"]),
        )

    @classmethod
    def from_json(cls, payload: str) -> TrainingStrategySchedule:
        """Decode canonical JSON into a verified complete schedule.

        Returns:
            The verified TrainingStrategySchedule record.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def _frozen_training_policy_schedules() -> tuple[TrainingStrategySchedule, ...]:
    """Build the exact prospective WP22 development-policy universe.

    Returns:
        The ordered prospective training schedules.
    """
    matched_noise = StandardNoiseMixture(
        "matched",
        (NoiseMixtureComponent("depolarizing_1s_all", 1.0),),
    )
    fixed_mixture = StandardNoiseMixture(
        "frozen_mixture",
        (
            NoiseMixtureComponent("depolarizing_1s_all", 0.5),
            NoiseMixtureComponent("dephasing_1s_all", 0.5),
        ),
    )
    direct = NoiseStrengthContinuation(
        0,
        199,
        1.0,
        start_strength_scale=1.0,
        interpolation="constant",
    )
    constant_eight = TrajectoryCountCurriculum((TrajectoryCountStep(0, 8),))
    checkpoint = CheckpointValidationPolicy(patience=None)
    noisy_phase = NoiselessPretrainNoisyFinetune(0, 200)
    fixed_sampling = TrajectorySamplingPolicy("fixed_crn")
    single_start = LimitedMultistartPlan(1, 3)

    def schedule(
        schedule_id: str,
        *,
        continuation: NoiseStrengthContinuation = direct,
        curriculum: TrajectoryCountCurriculum = constant_eight,
        sampling: TrajectorySamplingPolicy = fixed_sampling,
        multistart: LimitedMultistartPlan = single_start,
        noise: StandardNoiseMixture = matched_noise,
        phase: NoiselessPretrainNoisyFinetune = noisy_phase,
    ) -> TrainingStrategySchedule:
        """Build one complete frozen schedule variant.

        Returns:
            The complete frozen schedule variant.
        """
        return TrainingStrategySchedule(
            schedule_id=schedule_id,
            noise_continuation=continuation,
            trajectory_curriculum=curriculum,
            sampling_policy=sampling,
            checkpoint_validation=checkpoint,
            phase_boundary=phase,
            multistart=multistart,
            training_noise=noise,
        )

    return (
        schedule("direct_matched_fixed_crn"),
        schedule(
            "continuation_fixed_crn",
            continuation=NoiseStrengthContinuation(0, 49, 1.0),
        ),
        schedule(
            "curriculum_fixed_crn",
            curriculum=TrajectoryCountCurriculum((
                TrajectoryCountStep(0, 2),
                TrajectoryCountStep(50, 4),
                TrajectoryCountStep(100, 8),
            )),
        ),
        schedule(
            "periodic_refresh_20",
            sampling=TrajectorySamplingPolicy("periodic_full_refresh", refresh_interval=20),
        ),
        schedule(
            "rolling_half_refresh_20",
            sampling=TrajectorySamplingPolicy(
                "rolling_ensemble",
                refresh_interval=20,
                retain_fraction=0.5,
            ),
        ),
        schedule(
            "resampled_each_update",
            sampling=TrajectorySamplingPolicy("resampled"),
        ),
        schedule(
            "frozen_half_depolarizing_half_dephasing",
            noise=fixed_mixture,
        ),
        schedule(
            "limited_multistart_3",
            multistart=LimitedMultistartPlan(3, 3),
        ),
        schedule(
            "direct_noiseless_control",
            continuation=NoiseStrengthContinuation(
                0,
                199,
                0.0,
                start_strength_scale=0.0,
                interpolation="constant",
            ),
            curriculum=TrajectoryCountCurriculum((TrajectoryCountStep(0, 0),)),
            noise=StandardNoiseMixture("noiseless", ()),
            phase=NoiselessPretrainNoisyFinetune(200, 0),
        ),
    )


@dataclass(frozen=True, slots=True)
class FrozenTrainingPolicyUniverse:
    """Rooted set of every exact WP22 schedule policy authorized for later profiles."""

    schedules: tuple[TrainingStrategySchedule, ...]
    production_update_count: int = field(default=200, init=False)
    production_terminal_update: int = field(default=199, init=False)
    checkpoint_validation_trajectory_count: int = field(default=256, init=False)
    checkpoint_validation_updates: tuple[int, ...] = field(
        default=(*range(0, 199, 10), 199),
        init=False,
    )
    optimizer_state_rule: str = field(default="preserve_across_schedule_boundaries", init=False)
    unsupported_composition_action: str = field(default="reject_without_approximation", init=False)
    schedule_seed_policy_id: str = field(default=SCHEDULE_SEED_DERIVATION_POLICY_ID, init=False)
    persisted_membership_seed_usage: str = field(default="direct_sampler_seed", init=False)
    sampler_seed_rederivation: bool = field(default=False, init=False)
    schema_version: str = field(default=FROZEN_TRAINING_POLICY_UNIVERSE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require the exact ordered, complete reviewed schedule universe.

        Raises:
            TypeError: If an input has an invalid type or container shape.
            ValueError: If an input violates the sealed policy constraints.
        """
        if type(self.schedules) is not tuple or any(
            not isinstance(schedule, TrainingStrategySchedule) for schedule in self.schedules
        ):
            msg = "schedules must be a tuple of TrainingStrategySchedule records."
            raise TypeError(msg)
        expected = _frozen_training_policy_schedules()
        if self.schedules != expected:
            msg = "Frozen training schedules differ from the reviewed WP22 policy universe."
            raise ValueError(msg)
        if tuple(schedule.schedule_id for schedule in self.schedules) != FROZEN_TRAINING_POLICY_IDS:
            msg = "Frozen training schedule identities or order changed."
            raise ValueError(msg)

    @property
    def schedule_seed_policy_checksum(self) -> str:
        """Checksum of the policy that derives persisted membership seeds."""
        return ExecutionSeedPolicySuite.frozen().policy(self.schedule_seed_policy_id).content_checksum

    @classmethod
    def frozen(cls) -> FrozenTrainingPolicyUniverse:
        """Build the exact reviewed training-policy universe.

        Returns:
            The exact reviewed training-policy universe.
        """
        return cls(schedules=_frozen_training_policy_schedules())

    def _payload(self) -> dict[str, object]:
        """Return every reviewed schedule and global execution invariant."""
        return {
            "schema_version": self.schema_version,
            "production_update_count": self.production_update_count,
            "production_terminal_update": self.production_terminal_update,
            "checkpoint_validation_trajectory_count": self.checkpoint_validation_trajectory_count,
            "checkpoint_validation_updates": list(self.checkpoint_validation_updates),
            "optimizer_state_rule": self.optimizer_state_rule,
            "unsupported_composition_action": self.unsupported_composition_action,
            "schedule_seed_policy_id": self.schedule_seed_policy_id,
            "schedule_seed_policy_checksum": self.schedule_seed_policy_checksum,
            "persisted_membership_seed_usage": self.persisted_membership_seed_usage,
            "sampler_seed_rederivation": self.sampler_seed_rederivation,
            "schedule_ids": list(FROZEN_TRAINING_POLICY_IDS),
            "schedules": [schedule.to_dict() for schedule in self.schedules],
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete prospective training-policy universe."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed_dict(self._payload())

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> FrozenTrainingPolicyUniverse:
        """Decode and verify the exact reviewed training-policy universe.

        Returns:
            The verified FrozenTrainingPolicyUniverse record.

        Raises:
            TypeError: If an input has an invalid type or container shape.
            ValueError: If an input violates the sealed policy constraints.
        """
        keys = frozenset({
            "schema_version",
            "production_update_count",
            "production_terminal_update",
            "checkpoint_validation_trajectory_count",
            "checkpoint_validation_updates",
            "optimizer_state_rule",
            "unsupported_composition_action",
            "schedule_seed_policy_id",
            "schedule_seed_policy_checksum",
            "persisted_membership_seed_usage",
            "sampler_seed_rederivation",
            "schedule_ids",
            "schedules",
            "content_checksum",
        })
        mapping = _verify_record(
            value,
            expected_keys=keys,
            schema_version=FROZEN_TRAINING_POLICY_UNIVERSE_SCHEMA_VERSION,
            name="frozen training policy universe",
        )
        fixed = {
            "production_update_count": 200,
            "production_terminal_update": 199,
            "checkpoint_validation_trajectory_count": 256,
            "checkpoint_validation_updates": (*range(0, 199, 10), 199),
            "optimizer_state_rule": "preserve_across_schedule_boundaries",
            "unsupported_composition_action": "reject_without_approximation",
            "schedule_seed_policy_id": SCHEDULE_SEED_DERIVATION_POLICY_ID,
            "schedule_seed_policy_checksum": (
                ExecutionSeedPolicySuite.frozen().policy(SCHEDULE_SEED_DERIVATION_POLICY_ID).content_checksum
            ),
            "persisted_membership_seed_usage": "direct_sampler_seed",
            "sampler_seed_rederivation": False,
            "schedule_ids": FROZEN_TRAINING_POLICY_IDS,
        }
        if any(mapping[name] != expected for name, expected in fixed.items()):
            msg = "Frozen training-policy global invariants changed."
            raise ValueError(msg)
        raw_schedules = mapping["schedules"]
        if type(raw_schedules) is not tuple:
            msg = "schedules must be a JSON array."
            raise TypeError(msg)
        return cls(schedules=tuple(TrainingStrategySchedule.from_dict(item) for item in raw_schedules))

    @classmethod
    def from_json(cls, payload: str) -> FrozenTrainingPolicyUniverse:
        """Decode canonical JSON into the reviewed policy universe.

        Returns:
            The verified FrozenTrainingPolicyUniverse record.
        """
        return cls.from_dict(load_canonical_json_object(payload))


__all__ = [
    "CHECKPOINT_SELECTION_SCHEMA_VERSION",
    "CHECKPOINT_TRACKER_SCHEMA_VERSION",
    "CHECKPOINT_VALIDATION_SCHEMA_VERSION",
    "CONFIRMATORY_FRESH_EVALUATION_SEED_POLICY_ID",
    "CONFIRMATORY_OPTIMIZATION_SEED_POLICY_ID",
    "EXECUTION_SEED_POLICY_IDS",
    "EXECUTION_SEED_POLICY_SUITE_SCHEMA_VERSION",
    "FROZEN_TRAINING_POLICY_IDS",
    "FROZEN_TRAINING_POLICY_UNIVERSE_SCHEMA_VERSION",
    "MAX_MULTISTART_COUNT",
    "MULTISTART_SCHEMA_VERSION",
    "MULTISTART_SEEDS_SCHEMA_VERSION",
    "NOISE_CONTINUATION_SCHEMA_VERSION",
    "NOISE_MIXTURE_ALLOCATION_SCHEMA_VERSION",
    "NOISE_MIXTURE_COMPONENT_SCHEMA_VERSION",
    "PILOT_DIAGNOSTIC_SEED_POLICY_ID",
    "PILOT_FRESH_EVALUATION_SEED_POLICY_ID",
    "PILOT_OPTIMIZATION_SEED_POLICY_ID",
    "SCHEDULE_SEED_DERIVATION_POLICY_ID",
    "SCREENING_CELL_SEED_POLICY_ID",
    "SCREENING_ROOT_SEED_POLICY_ID",
    "SCREEN_OPTIMIZATION_SEED_POLICY_ID",
    "SEED_DERIVATION_POLICY_SCHEMA_VERSION",
    "SEED_DERIVATION_VERSION",
    "SMOKE_FRESH_EVALUATION_SEED_POLICY_ID",
    "SMOKE_OPTIMIZATION_SEED_POLICY_ID",
    "STAGE_SEED_DERIVATION_POLICY_ID",
    "STANDARD_NOISE_MIXTURE_SCHEMA_VERSION",
    "TRAINING_PHASE_BOUNDARY_SCHEMA_VERSION",
    "TRAINING_STRATEGY_SCHEDULE_SCHEMA_VERSION",
    "TRAJECTORY_COUNT_STEP_SCHEMA_VERSION",
    "TRAJECTORY_CURRICULUM_SCHEMA_VERSION",
    "TRAJECTORY_MEMBERSHIP_SCHEMA_VERSION",
    "TRAJECTORY_SAMPLING_SCHEMA_VERSION",
    "VALIDATION_CHECKPOINT_SCHEMA_VERSION",
    "CheckpointValidationPolicy",
    "CheckpointValidationSelection",
    "CheckpointValidationTracker",
    "ExecutionSeedPolicySuite",
    "FrozenTrainingPolicyUniverse",
    "LimitedMultistartPlan",
    "MultistartSeedBundle",
    "NoiseInterpolation",
    "NoiseMixtureAllocation",
    "NoiseMixtureComponent",
    "NoiseStrengthContinuation",
    "NoiselessPretrainNoisyFinetune",
    "SamplingKind",
    "SeedDerivationPolicy",
    "SeedRole",
    "StandardNoiseMixture",
    "TrainingNoiseMode",
    "TrainingStrategySchedule",
    "TrajectoryCountCurriculum",
    "TrajectoryCountStep",
    "TrajectoryEnsembleMembership",
    "TrajectoryRole",
    "TrajectorySamplingPolicy",
    "ValidationCheckpoint",
    "build_trajectory_membership",
    "derive_map_seed",
    "derive_member_seed",
    "derive_role_seed",
]
