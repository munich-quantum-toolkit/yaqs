# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Frozen WP22 screening-design records shared by planning and analysis.

This module owns the target-independent candidate configurations and the
deterministic screening Cartesian design.  It deliberately contains no
screening outcomes, promotion logic, numerical evidence, or final-seal code,
so WP22D planning does not depend on the later WP22F analysis package.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

from benchmarks.state_preparation.constants import NOISELESS_NOISE_ID

from .canonical import (
    canonical_checksum,
    canonical_json,
    freeze_json_mapping,
    load_canonical_json_object,
    thaw_json_mapping,
    verify_sealed_mapping,
)
from .pipeline import TrainingPipelineTemplate
from .protocol import (
    InitialPreregistration,
    ScreeningCandidateRef,
    ScreeningCell,
    ScreeningManifest,
)
from .pruning import TOPDOWN_IMPACT_ITERATIVE_METHOD_ID
from .targets import TargetPopulationManifest, verify_screening_target_population
from .validation import require_bool, require_checksum, require_float, require_int, require_slug

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

WP22_CANDIDATE_CONFIGURATION_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp22_candidate_configuration.v1"
WP22_OPERATOR_GROWTH_TEMPLATE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp22_operator_growth_template.v1"
WP22_PUBLICATION_PRUNING_MAPPING_VERSION = "yaqs.state_preparation.phase2.wp22_pruning_mapping.v1"

IMPACT_PRUNING_PUBLICATION_METHOD_ID = "impact_pruning_crn"
ADAPT_STYLE_PUBLICATION_METHOD_ID = "adapt_style_state_preparation"

_CANDIDATE_KEYS = frozenset({
    "schema_version",
    "method_id",
    "implementation_kind",
    "implementation_method_id",
    "implementation_schema_version",
    "implementation_checksum",
    "strategy_schedule_checksum",
    "resource_stratum_id",
    "noisy_training",
    "matching_projection_checksum",
    "publication_mapping",
    "content_checksum",
})
_OPERATOR_TEMPLATE_KEYS = frozenset({
    "schema_version",
    "method_id",
    "pool_policy_id",
    "growth_policy_id",
    "max_operators",
    "reoptimization_steps",
    "gradient_threshold",
    "training_trajectory_count",
    "native_two_qubit_cap_per_edge",
    "content_checksum",
})


@dataclass(frozen=True, slots=True)
class OperatorGrowthScreeningTemplate:
    """Target-independent publication configuration for noisy operator growth."""

    pool_policy_id: str
    growth_policy_id: str
    max_operators: int
    reoptimization_steps: int
    gradient_threshold: float
    training_trajectory_count: int
    native_two_qubit_cap_per_edge: float
    method_id: str = field(default=ADAPT_STYLE_PUBLICATION_METHOD_ID, init=False)
    schema_version: str = field(default=WP22_OPERATOR_GROWTH_TEMPLATE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the bounded target-independent operator-growth policy."""
        object.__setattr__(self, "pool_policy_id", require_slug(self.pool_policy_id, "pool_policy_id"))
        object.__setattr__(self, "growth_policy_id", require_slug(self.growth_policy_id, "growth_policy_id"))
        object.__setattr__(self, "max_operators", require_int(self.max_operators, "max_operators", minimum=1))
        object.__setattr__(
            self,
            "reoptimization_steps",
            require_int(self.reoptimization_steps, "reoptimization_steps", minimum=0),
        )
        object.__setattr__(
            self,
            "gradient_threshold",
            require_float(self.gradient_threshold, "gradient_threshold", minimum=0.0),
        )
        object.__setattr__(
            self,
            "training_trajectory_count",
            require_int(self.training_trajectory_count, "training_trajectory_count", minimum=1),
        )
        object.__setattr__(
            self,
            "native_two_qubit_cap_per_edge",
            require_float(
                self.native_two_qubit_cap_per_edge,
                "native_two_qubit_cap_per_edge",
                minimum=0.0,
            ),
        )

    def _content_dict(self) -> dict[str, object]:
        """Return all checksum-covered fields."""
        return {
            "schema_version": self.schema_version,
            "method_id": self.method_id,
            "pool_policy_id": self.pool_policy_id,
            "growth_policy_id": self.growth_policy_id,
            "max_operators": self.max_operators,
            "reoptimization_steps": self.reoptimization_steps,
            "gradient_threshold": self.gradient_threshold,
            "training_trajectory_count": self.training_trajectory_count,
            "native_two_qubit_cap_per_edge": self.native_two_qubit_cap_per_edge,
        }

    @property
    def content_checksum(self) -> str:
        """Exact operator-growth configuration checksum."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed JSON-native data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> OperatorGrowthScreeningTemplate:
        """Decode and verify one operator-growth template.

        Returns:
            The verified operator-growth template.

        Raises:
            ValueError: If the schema, method, or checksum is invalid.
        """
        mapping = verify_sealed_mapping(
            data,
            expected_keys=_OPERATOR_TEMPLATE_KEYS,
            name="WP22 operator-growth template",
        )
        if mapping["schema_version"] != WP22_OPERATOR_GROWTH_TEMPLATE_SCHEMA_VERSION:
            msg = "Operator-growth screening template uses an unsupported schema version."
            raise ValueError(msg)
        if mapping["method_id"] != ADAPT_STYLE_PUBLICATION_METHOD_ID:
            msg = "Operator-growth screening templates require adapt_style_state_preparation."
            raise ValueError(msg)
        template = cls(
            pool_policy_id=cast("str", mapping["pool_policy_id"]),
            growth_policy_id=cast("str", mapping["growth_policy_id"]),
            max_operators=cast("int", mapping["max_operators"]),
            reoptimization_steps=cast("int", mapping["reoptimization_steps"]),
            gradient_threshold=cast("float", mapping["gradient_threshold"]),
            training_trajectory_count=cast("int", mapping["training_trajectory_count"]),
            native_two_qubit_cap_per_edge=cast("float", mapping["native_two_qubit_cap_per_edge"]),
        )
        if mapping["content_checksum"] != template.content_checksum:
            msg = "Operator-growth screening template checksum changed during normalization."
            raise ValueError(msg)
        return template

    @classmethod
    def from_json(cls, payload: str) -> OperatorGrowthScreeningTemplate:
        """Decode canonical checksum-sealed JSON.

        Returns:
            The verified operator-growth template.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class WP22CandidateConfiguration:
    """One predeclared screening candidate and its executable implementation."""

    method_id: str
    implementation_kind: Literal["phase2_pipeline", "operator_growth"]
    implementation_method_id: str
    implementation_schema_version: str
    implementation_checksum: str
    strategy_schedule_checksum: str
    resource_stratum_id: str
    noisy_training: bool
    matching_projection_checksum: str | None
    publication_mapping: Mapping[str, object]
    schema_version: str = field(default=WP22_CANDIDATE_CONFIGURATION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate method identity, implementation binding, and publication alias.

        Raises:
            ValueError: If the implementation does not match its publication identity.
        """
        method = require_slug(self.method_id, "method_id")
        implementation = require_slug(self.implementation_method_id, "implementation_method_id")
        if self.implementation_kind not in {"phase2_pipeline", "operator_growth"}:
            msg = "implementation_kind must be 'phase2_pipeline' or 'operator_growth'."
            raise ValueError(msg)
        object.__setattr__(self, "method_id", method)
        object.__setattr__(self, "implementation_method_id", implementation)
        object.__setattr__(
            self,
            "implementation_schema_version",
            require_slug(self.implementation_schema_version, "implementation_schema_version"),
        )
        object.__setattr__(
            self,
            "implementation_checksum",
            require_checksum(self.implementation_checksum, "implementation_checksum"),
        )
        object.__setattr__(
            self,
            "strategy_schedule_checksum",
            require_checksum(self.strategy_schedule_checksum, "strategy_schedule_checksum"),
        )
        object.__setattr__(
            self,
            "resource_stratum_id",
            require_slug(self.resource_stratum_id, "resource_stratum_id"),
        )
        object.__setattr__(self, "noisy_training", require_bool(self.noisy_training, "noisy_training"))
        if self.matching_projection_checksum is not None:
            object.__setattr__(
                self,
                "matching_projection_checksum",
                require_checksum(self.matching_projection_checksum, "matching_projection_checksum"),
            )
        mapping = freeze_json_mapping(self.publication_mapping, "publication_mapping")
        if method == IMPACT_PRUNING_PUBLICATION_METHOD_ID:
            expected = {
                "mapping_version": WP22_PUBLICATION_PRUNING_MAPPING_VERSION,
                "publication_method_id": IMPACT_PRUNING_PUBLICATION_METHOD_ID,
                "implementation_method_id": TOPDOWN_IMPACT_ITERATIVE_METHOD_ID,
                "pruning_rule": "impact_iterative",
                "minimum_pruning_rounds": 2,
                "required_final_finetune_sampling": "crn_fixed",
            }
            if thaw_json_mapping(mapping) != expected or implementation != TOPDOWN_IMPACT_ITERATIVE_METHOD_ID:
                msg = "impact_pruning_crn requires the sealed iterative-impact/noisy-CRN publication mapping."
                raise ValueError(msg)
        elif mapping:
            msg = "publication_mapping is nonempty only for an explicit publication alias."
            raise ValueError(msg)
        elif method != implementation:
            msg = "Non-aliased publication methods must retain their executable method identity."
            raise ValueError(msg)
        if self.implementation_kind == "operator_growth" and (
            method != ADAPT_STYLE_PUBLICATION_METHOD_ID or implementation != ADAPT_STYLE_PUBLICATION_METHOD_ID
        ):
            msg = "The operator-growth wrapper is reserved for adapt_style_state_preparation."
            raise ValueError(msg)
        object.__setattr__(self, "publication_mapping", mapping)

    @classmethod
    def from_pipeline(
        cls,
        template: TrainingPipelineTemplate,
        *,
        strategy_schedule_checksum: str,
        publication_method_id: str | None = None,
    ) -> WP22CandidateConfiguration:
        """Bind a typed pipeline template to one publication candidate.

        The preregistered pruning name is an explicit outer alias.  It is
        accepted only for iterative impact pruning with at least two rounds and
        a terminal noisy fixed-CRN fine-tuning stage.

        Returns:
            The sealed publication candidate configuration.

        Raises:
            TypeError: If ``template`` is not a pipeline template.
            ValueError: If the publication identity or pruning policy is invalid.
        """
        if not isinstance(template, TrainingPipelineTemplate):
            msg = "template must be a TrainingPipelineTemplate."
            raise TypeError(msg)
        method = (
            template.method_id
            if publication_method_id is None
            else require_slug(
                publication_method_id,
                "publication_method_id",
            )
        )
        mapping: dict[str, object] = {}
        if method == IMPACT_PRUNING_PUBLICATION_METHOD_ID:
            pruning_rounds = tuple(stage for stage in template.stages if stage.stage_policy["stage_kind"] == "prune")
            final = template.stages[-1].stage_policy
            if (
                template.method_id != TOPDOWN_IMPACT_ITERATIVE_METHOD_ID
                or len(pruning_rounds) < 2
                or final["stage_id"] != "final_finetune"
                or final["training_noise_id"] == NOISELESS_NOISE_ID
                or final["sampling_policy"] != "crn_fixed"
                or final["trajectory_count"] == 0
            ):
                msg = (
                    "impact_pruning_crn must bind iterative impact pruning with at least two rounds "
                    "and terminal noisy fixed-CRN fine-tuning."
                )
                raise ValueError(msg)
            mapping = {
                "mapping_version": WP22_PUBLICATION_PRUNING_MAPPING_VERSION,
                "publication_method_id": IMPACT_PRUNING_PUBLICATION_METHOD_ID,
                "implementation_method_id": TOPDOWN_IMPACT_ITERATIVE_METHOD_ID,
                "pruning_rule": "impact_iterative",
                "minimum_pruning_rounds": 2,
                "required_final_finetune_sampling": "crn_fixed",
            }
        elif method != template.method_id:
            msg = "Only impact_pruning_crn has a preregistered publication alias."
            raise ValueError(msg)
        noisy = any(stage.stage_policy["training_noise_id"] != NOISELESS_NOISE_ID for stage in template.stages)
        matching = (
            template.matching_projection_checksum
            if method in {"layerwise_bmpd_crn_v2", "layerwise_bmpd_noiseless"}
            else None
        )
        return cls(
            method_id=method,
            implementation_kind="phase2_pipeline",
            implementation_method_id=template.method_id,
            implementation_schema_version=template.schema_version,
            implementation_checksum=template.configuration_checksum,
            strategy_schedule_checksum=strategy_schedule_checksum,
            resource_stratum_id=template.resource_stratum_id,
            noisy_training=noisy,
            matching_projection_checksum=matching,
            publication_mapping=mapping,
        )

    @classmethod
    def from_operator_growth(
        cls,
        template: OperatorGrowthScreeningTemplate,
        *,
        strategy_schedule_checksum: str,
        resource_stratum_id: str,
    ) -> WP22CandidateConfiguration:
        """Bind one standalone noisy operator-growth template to its WP22 wrapper.

        Returns:
            The sealed operator-growth publication candidate.

        Raises:
            TypeError: If ``template`` is not an operator-growth template.
        """
        if not isinstance(template, OperatorGrowthScreeningTemplate):
            msg = "template must be an OperatorGrowthScreeningTemplate."
            raise TypeError(msg)
        return cls(
            method_id=template.method_id,
            implementation_kind="operator_growth",
            implementation_method_id=template.method_id,
            implementation_schema_version=template.schema_version,
            implementation_checksum=template.content_checksum,
            strategy_schedule_checksum=strategy_schedule_checksum,
            resource_stratum_id=resource_stratum_id,
            noisy_training=True,
            matching_projection_checksum=None,
            publication_mapping={},
        )

    def _content_dict(self) -> dict[str, object]:
        """Return all checksum-covered candidate fields."""
        return {
            "schema_version": self.schema_version,
            "method_id": self.method_id,
            "implementation_kind": self.implementation_kind,
            "implementation_method_id": self.implementation_method_id,
            "implementation_schema_version": self.implementation_schema_version,
            "implementation_checksum": self.implementation_checksum,
            "strategy_schedule_checksum": self.strategy_schedule_checksum,
            "resource_stratum_id": self.resource_stratum_id,
            "noisy_training": self.noisy_training,
            "matching_projection_checksum": self.matching_projection_checksum,
            "publication_mapping": thaw_json_mapping(self.publication_mapping),
        }

    @property
    def content_checksum(self) -> str:
        """Complete publication-candidate checksum."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed JSON-native data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    def screening_ref(self) -> ScreeningCandidateRef:
        """Project onto the locked WP15 screening-candidate schema.

        Returns:
            The candidate reference consumed by the frozen screening protocol.
        """
        return ScreeningCandidateRef(
            configuration_schema_version=self.schema_version,
            configuration_checksum=self.content_checksum,
            method_id=self.method_id,
            noisy_training=self.noisy_training,
            resource_stratum_id=self.resource_stratum_id,
            matching_projection_checksum=self.matching_projection_checksum,
        )

    @classmethod
    def from_dict(cls, data: object) -> WP22CandidateConfiguration:
        """Decode and checksum-verify one candidate configuration.

        Returns:
            The verified candidate configuration.

        Raises:
            ValueError: If the schema or checksum is invalid.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_CANDIDATE_KEYS, name="WP22 candidate configuration")
        if mapping["schema_version"] != WP22_CANDIDATE_CONFIGURATION_SCHEMA_VERSION:
            msg = "WP22 candidate configuration uses an unsupported schema version."
            raise ValueError(msg)
        candidate = cls(
            method_id=cast("str", mapping["method_id"]),
            implementation_kind=cast("Literal['phase2_pipeline', 'operator_growth']", mapping["implementation_kind"]),
            implementation_method_id=cast("str", mapping["implementation_method_id"]),
            implementation_schema_version=cast("str", mapping["implementation_schema_version"]),
            implementation_checksum=cast("str", mapping["implementation_checksum"]),
            strategy_schedule_checksum=cast("str", mapping["strategy_schedule_checksum"]),
            resource_stratum_id=cast("str", mapping["resource_stratum_id"]),
            noisy_training=cast("bool", mapping["noisy_training"]),
            matching_projection_checksum=cast("str | None", mapping["matching_projection_checksum"]),
            publication_mapping=cast("Mapping[str, object]", mapping["publication_mapping"]),
        )
        if mapping["content_checksum"] != candidate.content_checksum:
            msg = "WP22 candidate checksum changed during normalization."
            raise ValueError(msg)
        return candidate

    @classmethod
    def from_json(cls, payload: str) -> WP22CandidateConfiguration:
        """Decode canonical checksum-sealed JSON.

        Returns:
            The verified candidate configuration.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def _derived_screening_seed(root_seed: int, target_instance_id: str, optimization_seed: int) -> int:
    """Derive a stable outer-screening seed from its complete cell identity.

    Returns:
        A deterministic unsigned 64-bit seed.
    """
    root = require_int(root_seed, "screening_seed_root", minimum=0)
    payload = canonical_json({
        "domain": "screening_selection",
        "root_seed": root,
        "target_instance_id": require_slug(target_instance_id, "target_instance_id"),
        "optimization_seed": require_int(optimization_seed, "optimization_seed", minimum=0),
    }).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big", signed=False)


def build_screening_manifest(
    preregistration: InitialPreregistration,
    target_manifest: TargetPopulationManifest,
    candidates: Sequence[WP22CandidateConfiguration],
    *,
    optimization_seeds: Sequence[int],
    screening_seed_root: int,
    manifest_id: str = "wp22_paper_screen_v1",
) -> ScreeningManifest:
    """Build the deterministic complete q6 candidate-by-cell screening universe.

    Returns:
        The checksum-sealed Cartesian screening manifest.

    Raises:
        TypeError: If inputs use the wrong artifact types.
        ValueError: If candidates, targets, or seeds violate the preregistration.
        RuntimeError: If distinct cells unexpectedly derive the same outer seed.
    """
    if not isinstance(preregistration, InitialPreregistration):
        msg = "preregistration must be an InitialPreregistration."
        raise TypeError(msg)
    if not isinstance(target_manifest, TargetPopulationManifest):
        msg = "target_manifest must be a TargetPopulationManifest."
        raise TypeError(msg)
    candidate_values = tuple(candidates)
    if not candidate_values or not all(isinstance(item, WP22CandidateConfiguration) for item in candidate_values):
        msg = "candidates must contain WP22CandidateConfiguration values."
        raise TypeError(msg)
    candidate_by_method = {item.method_id: item for item in candidate_values}
    if len(candidate_by_method) != len(candidate_values):
        msg = "Screening candidates must contain exactly one configuration per method."
        raise ValueError(msg)
    required_methods = tuple(
        cast("str", item["method_id"]) for item in preregistration.candidate_methods if item["scope"] == "all_families"
    )
    if set(candidate_by_method) != set(required_methods):
        missing = sorted(set(required_methods) - set(candidate_by_method))
        extra = sorted(set(candidate_by_method) - set(required_methods))
        msg = f"Screening candidate methods differ from the preregistration: missing={missing!r}, extra={extra!r}."
        raise ValueError(msg)
    ordered_candidates = tuple(candidate_by_method[method_id] for method_id in required_methods)
    for candidate in ordered_candidates:
        policy = preregistration.method_policy(candidate.method_id)
        if candidate.noisy_training is not policy["noisy_training"]:
            msg = f"Candidate {candidate.method_id!r} contradicts its preregistered noisy-training identity."
            raise ValueError(msg)
    seeds = tuple(require_int(seed, "optimization seed", minimum=0) for seed in optimization_seeds)
    expected_seed_count = cast(
        "int",
        cast("Mapping[str, object]", preregistration.target_population_policy["role_allocation_policy"])[
            "screening_optimizer_seed_count"
        ],
    )
    if len(seeds) != expected_seed_count or len(seeds) != len(set(seeds)):
        msg = f"paper-screen requires exactly {expected_seed_count} distinct optimization seeds."
        raise ValueError(msg)
    cells: list[ScreeningCell] = []
    used_outer_seeds: set[int] = set()
    for target in target_manifest.instances:
        for optimization_seed in seeds:
            screening_seed = _derived_screening_seed(
                screening_seed_root,
                target.target_instance_id,
                optimization_seed,
            )
            if screening_seed in used_outer_seeds:
                msg = "Derived duplicate outer screening seed."
                raise RuntimeError(msg)
            used_outer_seeds.add(screening_seed)
            identity = {
                "target_instance_id": target.target_instance_id,
                "optimization_seed": optimization_seed,
                "screening_seed": screening_seed,
            }
            cells.append(
                ScreeningCell(
                    cell_id=f"screening_cell_{canonical_checksum(identity).removeprefix('sha256:')}",
                    family_id=target.family_id,
                    stratum_id=target.stratum_id,
                    qubit_count=target.qubit_count,
                    target_instance_id=target.target_instance_id,
                    optimization_seed=optimization_seed,
                    screening_seed=screening_seed,
                )
            )
    baseline = candidate_by_method["layerwise_bmpd_crn_v2"]
    manifest = ScreeningManifest(
        manifest_id=require_slug(manifest_id, "manifest_id"),
        preregistration_checksum=preregistration.content_checksum,
        screening_target_manifest_checksum=target_manifest.content_checksum,
        evaluation_policy_checksum=canonical_checksum({
            "endpoint": preregistration.primary_endpoint,
            "failure_policy": preregistration.failure_policy,
            "noise": preregistration.primary_noise_condition,
        }),
        resource_policy_checksum=canonical_checksum(preregistration.primary_resource_constraint),
        baseline_configuration_checksum=baseline.content_checksum,
        candidates=tuple(item.screening_ref() for item in ordered_candidates),
        cells=tuple(cells),
    )
    verify_screening_target_population(manifest, target_manifest)
    return manifest


__all__ = [
    "ADAPT_STYLE_PUBLICATION_METHOD_ID",
    "IMPACT_PRUNING_PUBLICATION_METHOD_ID",
    "WP22_CANDIDATE_CONFIGURATION_SCHEMA_VERSION",
    "WP22_OPERATOR_GROWTH_TEMPLATE_SCHEMA_VERSION",
    "WP22_PUBLICATION_PRUNING_MAPPING_VERSION",
    "OperatorGrowthScreeningTemplate",
    "WP22CandidateConfiguration",
    "build_screening_manifest",
]
