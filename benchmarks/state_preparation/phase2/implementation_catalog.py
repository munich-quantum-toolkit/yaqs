# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Executable, width-complete repository implementation catalog for WP22B.

The catalog closes the gap between WP22A's checksum-sealed implementation
artifacts and concrete repository runner factories.  It contains no targets,
role entropy, outcomes, or promotion decisions.  A dormant confirmation lookup
aliases an eligible q6 screening entry by object identity and therefore cannot
mint a new confirmatory configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, cast

from benchmarks.state_preparation.noise import FIXED_RATE_NOISE_DEFINITION_VERSION

from .canonical import (
    canonical_checksum,
    canonical_json,
    freeze_json_mapping,
    load_canonical_json_object,
    verify_sealed_mapping,
)
from .competitor_optimizers import (
    BMPDCompetitorStageRunner,
    build_parameter_shift_adam_layerwise_template,
    build_spsa_layerwise_template,
)
from .execution_bindings import (
    PILOT_METHOD_IDS,
    SCREEN_METHOD_IDS,
    SMOKE_METHOD_IDS,
    EnergyAdaptSmokeSpec,
    ExecutionImplementationArtifact,
    ImplementationKind,
    OperatorGrowthSmokeSpec,
    PipelineSmokeSpec,
    Preset,
    TargetScope,
)
from .execution_protocol import OperatorGrowthExecutionSpec
from .fair_controls import (
    FixedDepthBMPDStageRunner,
    build_fixed_depth_bmpd_crn_template,
    build_layerwise_bmpd_cross_crn_template,
    build_layerwise_bmpd_noiseless_template,
    build_layerwise_bmpd_resampled_template,
)
from .layerwise_bmpd import (
    LayerwiseBMPDStageRunner,
    build_layerwise_bmpd_crn_v2_template,
)
from .operator_growth import (
    OperatorGrowthResult,
    OperatorGrowthSpec,
    OperatorGrowthWork,
    run_standard_fixed_rate_noisy_operator_growth,
    target_bound_energy_adapt_vqe,
)
from .pipeline import TrainingPipelineConfig, TrainingPipelineTemplate, TrainingStageConfig, TrainingStageTemplate
from .targets import MaterializedTarget, TargetInstanceSpec, TargetPopulationManifest
from .topdown_pruning import TopDownPruningStageRunner, build_topdown_impact_iterative_template
from .training_schedules import (
    CheckpointValidationPolicy,
    FrozenTrainingPolicyUniverse,
    LimitedMultistartPlan,
    NoiselessPretrainNoisyFinetune,
    NoiseMixtureComponent,
    NoiseStrengthContinuation,
    StandardNoiseMixture,
    TrainingStrategySchedule,
    TrajectoryCountCurriculum,
    TrajectoryCountStep,
    TrajectorySamplingPolicy,
)
from .validation import require_checksum, require_exact_keys, require_int, require_mapping, require_slug

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

IMPLEMENTATION_CATALOG_SCHEMA_VERSION = "yaqs.state_preparation.phase2.implementation_catalog.v1"
EXECUTABLE_IMPLEMENTATION_ENTRY_SCHEMA_VERSION = "yaqs.state_preparation.phase2.executable_implementation_entry.v1"
REPOSITORY_RUNNER_ADAPTER_SCHEMA_VERSION = "yaqs.state_preparation.phase2.repository_runner_adapter.v1"
PIPELINE_SMOKE_RUNTIME_PROGRAM_SCHEMA_VERSION = "yaqs.state_preparation.phase2.pipeline_smoke_runtime_program.v1"
OPERATOR_GROWTH_SMOKE_RUNTIME_PROGRAM_SCHEMA_VERSION = (
    "yaqs.state_preparation.phase2.operator_growth_smoke_runtime_program.v1"
)

SMOKE_ENTRY_COUNT = 10
PILOT_ENTRY_COUNT = 6
SCREEN_ENTRY_COUNT = 9
IMPLEMENTATION_CATALOG_ENTRY_COUNT = SMOKE_ENTRY_COUNT + PILOT_ENTRY_COUNT + SCREEN_ENTRY_COUNT

CatalogPreset = Literal["training-smoke", "paper-pilot", "paper-screen"]
RunnerFamily = Literal[
    "layerwise_bmpd_stage",
    "fixed_depth_bmpd_stage",
    "bmpd_competitor_stage",
    "topdown_pruning_stage",
    "projector_operator_growth",
    "tfim_energy_adapt",
]

_CATALOG_PRESETS = frozenset({"training-smoke", "paper-pilot", "paper-screen"})
# Confirmation may promote only an eligible noisy screen method, but every
# valid final seal also carries the exact screened matched-noiseless control.
# The dormant alias therefore has to cover the complete final-configuration
# method universe, not just the promotion-eligible subset.
_CONFIRMATION_METHOD_IDS = frozenset(SCREEN_METHOD_IDS)
_RUNNER_FAMILY_BY_METHOD: dict[str, RunnerFamily] = {
    "layerwise_bmpd_crn_v2": "layerwise_bmpd_stage",
    "layerwise_bmpd_noiseless": "layerwise_bmpd_stage",
    "fixed_depth_bmpd_crn": "fixed_depth_bmpd_stage",
    "layerwise_bmpd_resampled": "layerwise_bmpd_stage",
    "layerwise_bmpd_cross_crn": "layerwise_bmpd_stage",
    "parameter_shift_adam_layerwise": "bmpd_competitor_stage",
    "spsa_layerwise": "bmpd_competitor_stage",
    "adapt_style_state_preparation": "projector_operator_growth",
    "impact_pruning_crn": "topdown_pruning_stage",
    "energy_adapt_vqe": "tfim_energy_adapt",
}
_RUNNER_SYMBOL_BY_FAMILY: dict[RunnerFamily, tuple[str, str]] = {
    "layerwise_bmpd_stage": (
        "benchmarks.state_preparation.phase2.layerwise_bmpd",
        "LayerwiseBMPDStageRunner",
    ),
    "fixed_depth_bmpd_stage": (
        "benchmarks.state_preparation.phase2.fair_controls",
        "FixedDepthBMPDStageRunner",
    ),
    "bmpd_competitor_stage": (
        "benchmarks.state_preparation.phase2.competitor_optimizers",
        "BMPDCompetitorStageRunner",
    ),
    "topdown_pruning_stage": (
        "benchmarks.state_preparation.phase2.topdown_pruning",
        "TopDownPruningStageRunner",
    ),
    "projector_operator_growth": (
        "benchmarks.state_preparation.phase2.operator_growth",
        "run_standard_fixed_rate_noisy_operator_growth",
    ),
    "tfim_energy_adapt": (
        "benchmarks.state_preparation.phase2.operator_growth",
        "target_bound_energy_adapt_vqe",
    ),
}
_PIPELINE_METHOD_IDS = frozenset(_RUNNER_FAMILY_BY_METHOD) - {
    "adapt_style_state_preparation",
    "energy_adapt_vqe",
}
_IMPLEMENTATION_KINDS_BY_METHOD = {
    **{method: frozenset({"phase2_pipeline", "phase2_pipeline_smoke"}) for method in _PIPELINE_METHOD_IDS},
    "adapt_style_state_preparation": frozenset({"operator_growth", "operator_growth_smoke"}),
    "energy_adapt_vqe": frozenset({"tfim_operator_growth"}),
}
_ADAPTER_KEYS = frozenset({
    "schema_version",
    "adapter_id",
    "publication_method_id",
    "target_scope_id",
    "implementation_kind",
    "implementation_payload_checksum",
    "runner_family",
    "repository_module",
    "repository_symbol",
    "runtime_status",
    "content_checksum",
})
_ENTRY_KEYS = frozenset({
    "schema_version",
    "preset",
    "publication_method_id",
    "target_scope_id",
    "strategy_schedule",
    "strategy_schedule_checksum",
    "implementation_artifact",
    "implementation_artifact_checksum",
    "runner_adapter",
    "runner_adapter_checksum",
    "smoke_runtime_program",
    "smoke_runtime_program_checksum",
    "content_checksum",
})
_CATALOG_KEYS = frozenset({
    "schema_version",
    "catalog_id",
    "screening_outer_trajectory_count",
    "smoke_evaluation_trajectory_count",
    "confirmation_alias_policy",
    "entries",
    "content_checksum",
})


def _sealed(payload: dict[str, object]) -> dict[str, object]:
    """Return a detached mapping carrying its canonical checksum."""
    return {**payload, "content_checksum": canonical_checksum(payload)}


def _runner_registry() -> dict[RunnerFamily, Callable[..., object]]:
    """Return concrete, imported repository runner factories."""
    return {
        "layerwise_bmpd_stage": LayerwiseBMPDStageRunner,
        "fixed_depth_bmpd_stage": FixedDepthBMPDStageRunner,
        "bmpd_competitor_stage": BMPDCompetitorStageRunner,
        "topdown_pruning_stage": TopDownPruningStageRunner,
        "projector_operator_growth": run_standard_fixed_rate_noisy_operator_growth,
        "tfim_energy_adapt": target_bound_energy_adapt_vqe,
    }


def _callable_identity(callback: Callable[..., object]) -> tuple[str, str]:
    """Return the import module and qualified symbol of a repository callable.

    Returns:
        The concrete module and qualified symbol names.

    Raises:
        TypeError: If a callable does not expose stable import metadata.
    """
    module = getattr(callback, "__module__", None)
    symbol = getattr(callback, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(symbol, str):
        msg = "Repository runners must expose stable module and qualified symbol metadata."
        raise TypeError(msg)
    return module, symbol


class SmokeRuntimeTrainingPipelineTemplate(TrainingPipelineTemplate):
    """One-stage runtime-only template with isolated matching semantics."""

    def __post_init__(self) -> None:
        """Require the explicit runtime identity after normal template checks.

        Raises:
            ValueError: If this is not a one-stage WP22B runtime template.
        """
        # The production top-down schema requires the complete pruning path.
        # This runtime-only type deliberately exercises only its root-stage
        # runner, so generic template validation is applied under a temporary
        # non-pruning method identity and the exact method is restored before
        # the object can escape construction.
        method = self.method_id
        if method == "topdown_impact_iterative":
            object.__setattr__(self, "method_id", "fixed_depth_bmpd_crn")
        TrainingPipelineTemplate.__post_init__(self)
        object.__setattr__(self, "method_id", method)
        if not self.template_id.startswith("wp22b_smoke_runtime_") or len(self.stages) != 1:
            msg = "Smoke runtime template requires its explicit identity and exactly one stage."
            raise ValueError(msg)

    def matching_projection(self) -> Mapping[str, object]:
        """Return the runtime-only identity without changing paper matching."""
        return freeze_json_mapping(self.identity_payload(), "smoke runtime matching projection")


def _derive_pipeline_smoke_runtime_template(
    spec: PipelineSmokeSpec,
) -> SmokeRuntimeTrainingPipelineTemplate:
    """Derive the sole exact one-update template from its typed smoke source.

    Returns:
        The canonical runtime-only template consumed by a production runner.

    Raises:
        TypeError: If ``spec`` is not a pipeline smoke specification.
    """
    if not isinstance(spec, PipelineSmokeSpec):
        msg = "spec must be a PipelineSmokeSpec."
        raise TypeError(msg)
    template = spec.structural_template_reference
    structural = template.stages[0]
    terminal = template.stages[-1]
    policy = dict(structural.stage_policy)
    bindings = dict(structural.seed_bindings)
    optimizer_hyperparameters = dict(cast("Mapping[str, object]", terminal.stage_policy["optimizer_hyperparameters"]))
    structural_hyperparameters = cast(
        "Mapping[str, object]",
        structural.stage_policy["optimizer_hyperparameters"],
    )
    for initialization_key in ("initialization_rng", "initialization_scale"):
        if initialization_key in structural_hyperparameters:
            optimizer_hyperparameters[initialization_key] = structural_hyperparameters[initialization_key]
    policy["optimizer_id"] = terminal.stage_policy["optimizer_id"]
    policy["optimizer_hyperparameters"] = optimizer_hyperparameters
    policy["iteration_budget"] = spec.effective_limits.training_update_count
    policy["checkpoint_validation_policy"] = {
        "schema_version": "yaqs.state_preparation.phase2.checkpoint_validation_config.v1",
        "noise_id": "noiseless",
        "noise_definition_version": "yaqs.state_preparation.noise.v1",
        "noise_strength_scale": None,
        "tjm_dt": None,
        "trajectory_count": 0,
        "sampling_policy": "none",
        "ensemble_refresh_interval": None,
        "cadence": None,
        "selection_rule": "last_iteration",
        "tie_breaker": "earliest_iteration",
    }
    bindings["checkpoint_validation"] = None
    training_count = spec.effective_limits.training_trajectory_count
    if training_count == 0:
        policy.update({
            "training_noise_id": "noiseless",
            "noise_definition_version": "yaqs.state_preparation.noise.v1",
            "noise_strength_scale": None,
            "tjm_dt": None,
            "trajectory_count": 0,
            "trajectory_update": None,
            "sampling_policy": "none",
            "crn_refresh_interval": None,
        })
        bindings["training"] = None
    else:
        for key in (
            "training_noise_id",
            "noise_definition_version",
            "noise_strength_scale",
            "tjm_dt",
            "trajectory_update",
            "sampling_policy",
            "crn_refresh_interval",
        ):
            policy[key] = terminal.stage_policy[key]
        policy["trajectory_count"] = 1
        bindings["training"] = terminal.seed_bindings["training"]
    runtime_stage = TrainingStageTemplate(stage_policy=policy, seed_bindings=bindings)
    return SmokeRuntimeTrainingPipelineTemplate(
        template_id=f"wp22b_smoke_runtime_{template.method_id}",
        preregistration_checksum=template.preregistration_checksum,
        target_scope_id=template.target_scope_id,
        ansatz_family=template.ansatz_family,
        method_id=template.method_id,
        method_version=template.method_version,
        resource_stratum_id=template.resource_stratum_id,
        stages=(runtime_stage,),
        seed_domains=template.seed_domains,
        final_materialization_policy=template.final_materialization_policy,
    )


@dataclass(frozen=True, slots=True)
class PipelineSmokeRuntimeProgram:
    """One isolated, executable one-update pipeline-family smoke program."""

    source_spec: PipelineSmokeSpec
    runtime_template: SmokeRuntimeTrainingPipelineTemplate
    publication_method_id: str
    runner_family: RunnerFamily
    training_trajectory_count: int
    schema_version: str = field(default=PIPELINE_SMOKE_RUNTIME_PROGRAM_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate exact tiny work and its consumable runtime template.

        Raises:
            TypeError: If the runtime template is not typed.
            ValueError: If the template exceeds or contradicts smoke limits.
        """
        if not isinstance(self.source_spec, PipelineSmokeSpec):
            msg = "source_spec must be a PipelineSmokeSpec."
            raise TypeError(msg)
        if not isinstance(self.runtime_template, SmokeRuntimeTrainingPipelineTemplate):
            msg = "runtime_template must be a SmokeRuntimeTrainingPipelineTemplate."
            raise TypeError(msg)
        method = require_slug(self.publication_method_id, "publication_method_id")
        object.__setattr__(self, "publication_method_id", method)
        expected_family = _RUNNER_FAMILY_BY_METHOD.get(method)
        implementation_method = "topdown_impact_iterative" if method == "impact_pruning_crn" else method
        if expected_family != self.runner_family or self.runner_family in {
            "projector_operator_growth",
            "tfim_energy_adapt",
        }:
            msg = "Pipeline smoke runtime uses the wrong concrete runner family."
            raise ValueError(msg)
        count = require_int(self.training_trajectory_count, "training_trajectory_count")
        expected_count = 0 if method == "layerwise_bmpd_noiseless" else 1
        if (
            count != expected_count
            or self.source_spec.method_id != implementation_method
            or self.source_spec.effective_limits.training_trajectory_count != count
            or self.runtime_template != _derive_pipeline_smoke_runtime_template(self.source_spec)
            or self.runtime_template.method_id != implementation_method
            or self.runtime_template.target_scope_id != "primary_q6"
            or self.runtime_template.template_id != f"wp22b_smoke_runtime_{implementation_method}"
        ):
            msg = "Pipeline smoke runtime method, scope, identity, or trajectory limit differs."
            raise ValueError(msg)
        stage = self.runtime_template.stages[0]
        policy = stage.stage_policy
        checkpoint = require_mapping(policy["checkpoint_validation_policy"], "checkpoint_validation_policy")
        if (
            stage.stage_index != 0
            or policy["stage_kind"] != "optimize"
            or policy["input_topology_id"] is not None
            or policy["input_parameter_count"] != 0
            or policy["iteration_budget"] != 1
            or policy["trajectory_count"] != count
            or checkpoint["trajectory_count"] != 0
            or stage.seed_bindings["checkpoint_validation"] is not None
        ):
            msg = "Pipeline smoke runtime must isolate exactly one update with no checkpoint trajectories."
            raise ValueError(msg)

    @property
    def source_spec_checksum(self) -> str:
        """Checksum of the exact typed smoke source used for derivation."""
        return self.source_spec.content_checksum

    @property
    def runner_factory(self) -> type[object]:
        """Return the concrete production-family runner consumed at execution."""
        return cast("type[object]", _runner_registry()[self.runner_family])

    def bind(
        self,
        target_manifest: TargetPopulationManifest,
        target: MaterializedTarget,
        *,
        optimization_seed: int,
    ) -> BoundPipelineSmokeRunner:
        """Resolve a full target-bound config and instantiate its real runner.

        Returns:
            The exact runtime pipeline, stage, and production-family runner.

        Raises:
            TypeError: If the target or manifest is untyped.
            ValueError: If their identities do not match the q6 smoke scope.
        """
        if not isinstance(target_manifest, TargetPopulationManifest) or not isinstance(target, MaterializedTarget):
            msg = "Smoke binding requires typed target manifest and materialization records."
            raise TypeError(msg)
        identity = target.identity_dict()
        target_id = cast("str", identity["target_instance_id"])
        spec = next(
            (candidate for candidate in target_manifest.instances if candidate.target_instance_id == target_id),
            None,
        )
        if spec is None or target_manifest.content_checksum != identity["target_manifest_checksum"]:
            msg = "Materialized target is absent from the supplied exact manifest."
            raise ValueError(msg)
        pipeline = self.runtime_template.resolve(
            target_namespace="phase2",
            target_manifest=target_manifest,
            target_instance_id=target_id,
            target_population_manifest_checksum=target_manifest.content_checksum,
            target_instance_spec_checksum=spec.content_checksum,
            target_family_id=spec.family_id,
            target_stratum_id=spec.stratum_id,
            qubit_count=spec.qubit_count,
            optimization_block_id=f"smoke_{self.publication_method_id}",
            optimization_seed=optimization_seed,
            data_role="development",
        )
        runner = self.runner_factory(pipeline, target)
        return BoundPipelineSmokeRunner(self, pipeline, runner)

    @property
    def content_checksum(self) -> str:
        """Checksum covering the runtime template and concrete dispatch."""
        return canonical_checksum(self._payload())

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered runtime-program field."""
        runner_module, runner_symbol = _callable_identity(self.runner_factory)
        return {
            "schema_version": self.schema_version,
            "source_spec": self.source_spec.to_dict(),
            "source_spec_checksum": self.source_spec_checksum,
            "publication_method_id": self.publication_method_id,
            "runner_family": self.runner_family,
            "runner_module": runner_module,
            "runner_symbol": runner_symbol,
            "training_update_count": 1,
            "training_trajectory_count": self.training_trajectory_count,
            "checkpoint_validation_trajectory_count": 0,
            "runtime_template": self.runtime_template.to_dict(),
        }

    def to_dict(self) -> dict[str, object]:
        """Return the checksum-sealed derived runtime program."""
        return _sealed(self._payload())


@dataclass(frozen=True, slots=True)
class BoundPipelineSmokeRunner:
    """A resolved runtime pipeline and its genuine registered stage runner."""

    program: PipelineSmokeRuntimeProgram
    pipeline: TrainingPipelineConfig
    runner: object

    def __post_init__(self) -> None:
        """Verify the full config and concrete runner before execution.

        Raises:
            TypeError: If the program or pipeline is untyped.
            ValueError: If the runner is not the registered family instance.
        """
        if not isinstance(self.program, PipelineSmokeRuntimeProgram):
            msg = "program must be a PipelineSmokeRuntimeProgram."
            raise TypeError(msg)
        if not isinstance(self.pipeline, TrainingPipelineConfig):
            msg = "pipeline must be a TrainingPipelineConfig."
            raise TypeError(msg)
        if self.pipeline.template != self.program.runtime_template or len(self.pipeline.stages) != 1:
            msg = "Bound smoke pipeline differs from its exact one-stage runtime template."
            raise ValueError(msg)
        if not isinstance(self.runner, self.program.runner_factory):
            msg = "Bound smoke runner is not the registered repository family instance."
            raise TypeError(msg)

    @property
    def stage(self) -> TrainingStageConfig:
        """Return the sole resolved one-update stage."""
        return self.pipeline.stages[0]

    def execute(self) -> object:
        """Execute one update through the genuine registered family runner.

        Returns:
            The real runner's one-stage numerical outcome.
        """
        runner = cast("Callable[[TrainingStageConfig, object | None], object]", self.runner)
        return runner(self.stage, None)


@dataclass(frozen=True, slots=True)
class OperatorGrowthSmokeRuntimeProgram:
    """One bounded operator-growth smoke program and concrete callback."""

    source_spec: OperatorGrowthSmokeSpec | EnergyAdaptSmokeSpec
    publication_method_id: str
    runtime_growth_spec: OperatorGrowthSpec
    runner_family: RunnerFamily
    training_trajectory_count: int
    schema_version: str = field(default=OPERATOR_GROWTH_SMOKE_RUNTIME_PROGRAM_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require one selection, one reoptimization update, and truthful noise work.

        Raises:
            TypeError: If the runtime growth specification is untyped.
            ValueError: If method, runner, or effective limits differ.
        """
        method = require_slug(self.publication_method_id, "publication_method_id")
        object.__setattr__(self, "publication_method_id", method)
        if not isinstance(self.runtime_growth_spec, OperatorGrowthSpec):
            msg = "runtime_growth_spec must be an OperatorGrowthSpec."
            raise TypeError(msg)
        expected_family = _RUNNER_FAMILY_BY_METHOD.get(method)
        expected_count = 0 if method == "energy_adapt_vqe" else 1
        expected_source_type = EnergyAdaptSmokeSpec if method == "energy_adapt_vqe" else OperatorGrowthSmokeSpec
        if not isinstance(self.source_spec, expected_source_type):
            msg = "Operator-growth smoke source has the wrong typed method contract."
            raise TypeError(msg)
        source_growth_spec = (
            self.source_spec.growth_spec
            if isinstance(self.source_spec, EnergyAdaptSmokeSpec)
            else self.source_spec.production_growth_spec
        )
        expected_runtime_spec = replace(source_growth_spec, max_operators=1, reoptimization_steps=1)
        if (
            expected_family != self.runner_family
            or self.runner_family not in {"projector_operator_growth", "tfim_energy_adapt"}
            or self.runtime_growth_spec.method_id
            != ("energy_adapt_vqe" if method == "energy_adapt_vqe" else "adapt_style_state_preparation")
            or self.runtime_growth_spec.max_operators != 1
            or self.runtime_growth_spec.reoptimization_steps != 1
            or self.training_trajectory_count != expected_count
            or self.source_spec.effective_limits.training_trajectory_count != expected_count
            or self.runtime_growth_spec != expected_runtime_spec
        ):
            msg = "Operator-growth smoke runtime differs from its exact bounded method contract."
            raise ValueError(msg)

    @property
    def source_spec_checksum(self) -> str:
        """Checksum of the exact typed operator-growth smoke source."""
        return self.source_spec.content_checksum

    @property
    def runner_callback(self) -> Callable[..., object]:
        """Return the concrete bounded operator-growth callback."""
        return _runner_registry()[self.runner_family]

    def execute_projector(
        self,
        target: MaterializedTarget,
        *,
        optimization_block_id: str,
        optimization_seed: int,
        resource_stratum_id: str,
        trajectory_seed: int,
    ) -> OperatorGrowthSmokeExecution:
        """Execute the exact one-growth noisy projector preflight.

        The noise family, definition, strength, time step, trajectory count,
        and growth limits are deliberately not caller parameters.  This keeps
        the executable smoke seam bound to the same standard-noise treatment
        named by its source artifact.

        Returns:
            The genuine noisy operator-growth result with measured work.

        Raises:
            ValueError: If this is not the projector-growth runtime program.
        """
        if self.runner_family != "projector_operator_growth":
            msg = "Only the projector smoke runtime can execute noisy projector growth."
            raise ValueError(msg)
        callback = cast("Callable[..., OperatorGrowthResult]", self.runner_callback)
        result = callback(
            target,
            optimization_block_id=optimization_block_id,
            optimization_seed=optimization_seed,
            resource_stratum_id=resource_stratum_id,
            noise_id="depolarizing_1s_all",
            noise_definition_version=FIXED_RATE_NOISE_DEFINITION_VERSION,
            noise_strength_scale=1.0,
            tjm_dt=1.0,
            trajectory_count=self.training_trajectory_count,
            trajectory_seed=trajectory_seed,
            growth_spec=self.runtime_growth_spec,
        )
        return OperatorGrowthSmokeExecution.from_result(self, result)

    def execute_energy(
        self,
        target: MaterializedTarget,
        target_instance_spec: TargetInstanceSpec,
    ) -> OperatorGrowthSmokeExecution:
        """Execute the exact one-growth analytic Energy-ADAPT API preflight.

        Returns:
            The genuine target-bound analytic Energy-ADAPT result.

        Raises:
            ValueError: If this is not the analytic Energy-ADAPT runtime.
        """
        if self.runner_family != "tfim_energy_adapt":
            msg = "Only the Energy-ADAPT smoke runtime can execute the analytic API preflight."
            raise ValueError(msg)
        callback = cast("Callable[..., OperatorGrowthResult]", self.runner_callback)
        result = callback(
            target,
            target_instance_spec,
            growth_spec=self.runtime_growth_spec,
        )
        return OperatorGrowthSmokeExecution.from_result(self, result)

    @property
    def content_checksum(self) -> str:
        """Checksum covering the bounded growth spec and concrete callback."""
        runner_module, runner_symbol = _callable_identity(self.runner_callback)
        return canonical_checksum({
            "schema_version": self.schema_version,
            "source_spec": self.source_spec.to_dict(),
            "source_spec_checksum": self.source_spec_checksum,
            "publication_method_id": self.publication_method_id,
            "runner_family": self.runner_family,
            "runner_module": runner_module,
            "runner_symbol": runner_symbol,
            "maximum_growth_steps": 1,
            "reoptimization_steps_per_growth": 1,
            "training_trajectory_count": self.training_trajectory_count,
            "runtime_growth_spec": self.runtime_growth_spec.to_dict(),
        })

    def to_dict(self) -> dict[str, object]:
        """Return the checksum-sealed bounded operator-growth program."""
        runner_module, runner_symbol = _callable_identity(self.runner_callback)
        payload = {
            "schema_version": self.schema_version,
            "source_spec": self.source_spec.to_dict(),
            "source_spec_checksum": self.source_spec_checksum,
            "publication_method_id": self.publication_method_id,
            "runner_family": self.runner_family,
            "runner_module": runner_module,
            "runner_symbol": runner_symbol,
            "maximum_growth_steps": 1,
            "reoptimization_steps_per_growth": 1,
            "training_trajectory_count": self.training_trajectory_count,
            "runtime_growth_spec": self.runtime_growth_spec.to_dict(),
        }
        return {**payload, "content_checksum": canonical_checksum(payload)}


@dataclass(frozen=True, slots=True)
class OperatorGrowthSmokeExecution:
    """Non-promotable summary of one genuine bounded growth execution."""

    program: OperatorGrowthSmokeRuntimeProgram
    numerical_result_checksum: str
    status: str
    execution_mode: Literal["analytic_reference", "noisy_training"]
    selected_operator_ids: tuple[str, ...]
    trace_count: int
    work: OperatorGrowthWork
    objective_request_trajectory_counts: tuple[int, ...]
    evidence_role: str = field(default="structural_smoke_preflight", init=False)
    promotion_eligible: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Validate a non-promotable summary without embedding promotable evidence.

        Raises:
            TypeError: If the program or work is not the exact typed record.
            ValueError: If mode, limits, actual work, or summary fields disagree.
        """
        if not isinstance(self.program, OperatorGrowthSmokeRuntimeProgram):
            msg = "program must be an OperatorGrowthSmokeRuntimeProgram."
            raise TypeError(msg)
        if not isinstance(self.work, OperatorGrowthWork):
            msg = "work must be an OperatorGrowthWork."
            raise TypeError(msg)
        object.__setattr__(
            self,
            "numerical_result_checksum",
            require_checksum(self.numerical_result_checksum, "numerical_result_checksum"),
        )
        expected_mode = "analytic_reference" if self.program.runner_family == "tfim_energy_adapt" else "noisy_training"
        selected = tuple(self.selected_operator_ids)
        counts = tuple(self.objective_request_trajectory_counts)
        trace_count = require_int(self.trace_count, "trace_count")
        if (
            self.status != "completed"
            or self.execution_mode != expected_mode
            or len(selected) > 1
            or trace_count > 1
            or any(type(count) is not int or count < 1 for count in counts)
            or (expected_mode == "analytic_reference" and (self.work.total_sampled_trajectories != 0 or counts))
            or (
                expected_mode == "noisy_training"
                and (
                    not counts
                    or any(count != self.program.training_trajectory_count for count in counts)
                    or self.work.total_sampled_trajectories <= 0
                )
            )
        ):
            msg = "Operator-growth summary differs from the exact non-promotable smoke execution."
            raise ValueError(msg)
        object.__setattr__(self, "selected_operator_ids", selected)
        object.__setattr__(self, "trace_count", trace_count)
        object.__setattr__(self, "objective_request_trajectory_counts", counts)

    @classmethod
    def from_result(
        cls,
        program: OperatorGrowthSmokeRuntimeProgram,
        result: OperatorGrowthResult,
    ) -> OperatorGrowthSmokeExecution:
        """Project a genuine numerical result into non-promotable smoke evidence.

        The underlying production result is validated in memory and only its
        checksum plus bounded work summary are retained.  Consequently the
        returned or serialized smoke object cannot be supplied where a
        promotion-eligible ``OperatorGrowthResult`` is required.

        Returns:
            A non-promotable, checksum-bound bounded-work summary.

        Raises:
            TypeError: If the program or numerical result is untyped.
            ValueError: If method, growth specification, or work exceeds the smoke contract.
        """
        if not isinstance(program, OperatorGrowthSmokeRuntimeProgram):
            msg = "program must be an OperatorGrowthSmokeRuntimeProgram."
            raise TypeError(msg)
        if not isinstance(result, OperatorGrowthResult):
            msg = "result must be an OperatorGrowthResult."
            raise TypeError(msg)
        if (
            result.method_id != program.runtime_growth_spec.method_id
            or result.growth_spec != program.runtime_growth_spec
            or len(result.selected_operator_ids) > 1
            or len(result.trace) > 1
        ):
            msg = "Numerical operator-growth result differs from its exact bounded smoke program."
            raise ValueError(msg)
        return cls(
            program=program,
            numerical_result_checksum=result.content_checksum,
            status=result.status,
            execution_mode=result.execution_mode,
            selected_operator_ids=result.selected_operator_ids,
            trace_count=len(result.trace),
            work=result.work,
            objective_request_trajectory_counts=tuple(
                request.trajectory_count for request in result.objective_requests
            ),
        )

    @property
    def content_checksum(self) -> str:
        """Checksum covering the program, bounded work, and evidence exclusion."""
        return canonical_checksum(self._payload())

    def _payload(self) -> dict[str, object]:
        """Return all checksum-covered smoke-execution fields."""
        return {
            "program_checksum": self.program.content_checksum,
            "source_spec_checksum": self.program.source_spec_checksum,
            "numerical_result_checksum": self.numerical_result_checksum,
            "evidence_role": self.evidence_role,
            "promotion_eligible": self.promotion_eligible,
            "status": self.status,
            "execution_mode": self.execution_mode,
            "selected_operator_ids": list(self.selected_operator_ids),
            "trace_count": self.trace_count,
            "work": self.work.to_dict(),
            "objective_request_trajectory_counts": list(self.objective_request_trajectory_counts),
        }

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed smoke execution evidence."""
        return _sealed(self._payload())


SmokeRuntimeProgram = PipelineSmokeRuntimeProgram | OperatorGrowthSmokeRuntimeProgram


def materialize_pipeline_smoke_runtime(spec: PipelineSmokeSpec) -> PipelineSmokeRuntimeProgram:
    """Derive the isolated one-update runtime stage from a typed structural spec.

    Returns:
        A bounded stage carrying its concrete production-family runner.

    Raises:
        TypeError: If ``spec`` is not a pipeline smoke specification.
    """
    if not isinstance(spec, PipelineSmokeSpec):
        msg = "spec must be a PipelineSmokeSpec."
        raise TypeError(msg)
    template = spec.structural_template_reference
    training_count = spec.effective_limits.training_trajectory_count
    return PipelineSmokeRuntimeProgram(
        source_spec=spec,
        runtime_template=_derive_pipeline_smoke_runtime_template(spec),
        publication_method_id=(
            "impact_pruning_crn" if template.method_id == "topdown_impact_iterative" else template.method_id
        ),
        runner_family=_RUNNER_FAMILY_BY_METHOD[
            "impact_pruning_crn" if template.method_id == "topdown_impact_iterative" else template.method_id
        ],
        training_trajectory_count=training_count,
    )


def materialize_operator_growth_smoke_runtime(
    spec: OperatorGrowthSmokeSpec,
) -> OperatorGrowthSmokeRuntimeProgram:
    """Derive the one-selection, one-update projector-growth runtime.

    Returns:
        The bounded projector-growth runtime program.

    Raises:
        TypeError: If ``spec`` is not the projector-growth smoke type.
    """
    if not isinstance(spec, OperatorGrowthSmokeSpec):
        msg = "spec must be an OperatorGrowthSmokeSpec."
        raise TypeError(msg)
    runtime = replace(spec.production_growth_spec, max_operators=1, reoptimization_steps=1)
    return OperatorGrowthSmokeRuntimeProgram(
        source_spec=spec,
        publication_method_id="adapt_style_state_preparation",
        runtime_growth_spec=runtime,
        runner_family="projector_operator_growth",
        training_trajectory_count=spec.effective_limits.training_trajectory_count,
    )


def materialize_energy_adapt_smoke_runtime(spec: EnergyAdaptSmokeSpec) -> OperatorGrowthSmokeRuntimeProgram:
    """Derive the one-selection, one-update analytic energy-ADAPT runtime.

    Returns:
        The bounded analytic energy-growth runtime program.

    Raises:
        TypeError: If ``spec`` is not the energy-ADAPT smoke type.
    """
    if not isinstance(spec, EnergyAdaptSmokeSpec):
        msg = "spec must be an EnergyAdaptSmokeSpec."
        raise TypeError(msg)
    runtime = replace(spec.growth_spec, max_operators=1, reoptimization_steps=1)
    return OperatorGrowthSmokeRuntimeProgram(
        source_spec=spec,
        publication_method_id="energy_adapt_vqe",
        runtime_growth_spec=runtime,
        runner_family="tfim_energy_adapt",
        training_trajectory_count=spec.effective_limits.training_trajectory_count,
    )


@dataclass(frozen=True, slots=True)
class RepositoryRunnerAdapter:
    """Checksum-bound resolver for one concrete repository runner factory."""

    publication_method_id: str
    target_scope_id: TargetScope
    implementation_kind: ImplementationKind
    implementation_payload_checksum: str
    schema_version: str = field(default=REPOSITORY_RUNNER_ADAPTER_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Reject unsupported methods, widths, kinds, or unresolvable callables.

        Raises:
            ValueError: If the method, width, or artifact kind is unsupported.
        """
        method = require_slug(self.publication_method_id, "publication_method_id")
        if method not in _RUNNER_FAMILY_BY_METHOD:
            msg = "Publication method has no repository-owned WP22B runner."
            raise ValueError(msg)
        object.__setattr__(self, "publication_method_id", method)
        if self.target_scope_id not in {"primary_q6", "secondary_q12"}:
            msg = "target_scope_id must be primary_q6 or secondary_q12."
            raise ValueError(msg)
        if self.target_scope_id == "secondary_q12" and method not in PILOT_METHOD_IDS:
            msg = "Only the three pilot methods have secondary-q12 repository runners."
            raise ValueError(msg)
        allowed_kinds = _IMPLEMENTATION_KINDS_BY_METHOD[method]
        if self.implementation_kind not in allowed_kinds or (
            self.target_scope_id == "secondary_q12" and self.implementation_kind != "phase2_pipeline"
        ):
            msg = "Implementation kind is incompatible with the exact method and target scope."
            raise ValueError(msg)
        object.__setattr__(
            self,
            "implementation_payload_checksum",
            require_checksum(self.implementation_payload_checksum, "implementation_payload_checksum"),
        )
        self.resolve_callable()

    @property
    def runner_family(self) -> RunnerFamily:
        """Exact runner family derived from publication identity."""
        return _RUNNER_FAMILY_BY_METHOD[self.publication_method_id]

    @property
    def adapter_id(self) -> str:
        """Stable method-, width-, and kind-specific adapter identity."""
        return f"{self.publication_method_id}_{self.target_scope_id}_{self.implementation_kind}_adapter"

    @property
    def repository_module(self) -> str:
        """Module owning the resolved callable."""
        if self.implementation_kind in {
            "phase2_pipeline_smoke",
            "operator_growth_smoke",
            "tfim_operator_growth",
        }:
            return __name__
        return _RUNNER_SYMBOL_BY_FAMILY[self.runner_family][0]

    @property
    def repository_symbol(self) -> str:
        """Symbol naming the resolved callable."""
        smoke_symbols = {
            "phase2_pipeline_smoke": "materialize_pipeline_smoke_runtime",
            "operator_growth_smoke": "materialize_operator_growth_smoke_runtime",
            "tfim_operator_growth": "materialize_energy_adapt_smoke_runtime",
        }
        if self.implementation_kind in smoke_symbols:
            return smoke_symbols[self.implementation_kind]
        return _RUNNER_SYMBOL_BY_FAMILY[self.runner_family][1]

    def resolve_callable(self) -> Callable[..., object]:
        """Resolve and verify the concrete repository callable before execution.

        Returns:
            The imported runner class or function bound by this adapter.

        Raises:
            RuntimeError: If repository code no longer matches the sealed route.
        """
        smoke_routes: dict[ImplementationKind, Callable[..., object]] = {
            "phase2_pipeline_smoke": materialize_pipeline_smoke_runtime,
            "operator_growth_smoke": materialize_operator_growth_smoke_runtime,
            "tfim_operator_growth": materialize_energy_adapt_smoke_runtime,
        }
        runner = smoke_routes.get(self.implementation_kind, _runner_registry()[self.runner_family])
        runner_module, runner_symbol = _callable_identity(runner)
        if runner_module != self.repository_module or runner_symbol != self.repository_symbol:
            msg = "Repository runner route no longer resolves to its checksum-bound callable."
            raise RuntimeError(msg)
        return runner

    @classmethod
    def for_artifact(cls, artifact: ExecutionImplementationArtifact) -> RepositoryRunnerAdapter:
        """Build the exact operational adapter for one typed artifact.

        Returns:
            A checksum-bound adapter whose route is validated immediately.

        Raises:
            TypeError: If ``artifact`` is untyped.
        """
        if not isinstance(artifact, ExecutionImplementationArtifact):
            msg = "artifact must be an ExecutionImplementationArtifact."
            raise TypeError(msg)
        adapter = cls(
            publication_method_id=artifact.publication_method_id,
            target_scope_id=artifact.target_scope_id,
            implementation_kind=artifact.implementation_kind,
            implementation_payload_checksum=artifact.implementation_payload_checksum,
        )
        if artifact.implementation_kind in {
            "phase2_pipeline_smoke",
            "operator_growth_smoke",
            "tfim_operator_growth",
        }:
            adapter.materialize_smoke_runtime(artifact)
        return adapter

    def materialize_smoke_runtime(
        self,
        artifact: ExecutionImplementationArtifact,
    ) -> SmokeRuntimeProgram:
        """Consume a typed smoke artifact and derive its bounded runtime.

        Returns:
            The exact one-update pipeline or operator-growth runtime program.

        Raises:
            TypeError: If the artifact is not a typed smoke payload.
            ValueError: If the artifact and adapter identities differ.
        """
        if not isinstance(artifact, ExecutionImplementationArtifact):
            msg = "artifact must be an ExecutionImplementationArtifact."
            raise TypeError(msg)
        if (
            artifact.publication_method_id != self.publication_method_id
            or artifact.target_scope_id != self.target_scope_id
            or artifact.implementation_kind != self.implementation_kind
            or artifact.implementation_payload_checksum != self.implementation_payload_checksum
        ):
            msg = "Smoke artifact does not match its executable repository adapter."
            raise ValueError(msg)
        if artifact.implementation_kind not in {
            "phase2_pipeline_smoke",
            "operator_growth_smoke",
            "tfim_operator_growth",
        }:
            msg = "Only typed smoke artifacts can be materialized through the smoke seam."
            raise ValueError(msg)
        runtime = self.resolve_callable()(artifact.implementation_payload)
        if not isinstance(runtime, (PipelineSmokeRuntimeProgram, OperatorGrowthSmokeRuntimeProgram)):
            msg = "Repository smoke materializer returned an unsupported runtime program."
            raise TypeError(msg)
        if runtime.source_spec_checksum != artifact.implementation_payload_checksum:
            msg = "Smoke runtime is not bound to the exact implementation payload."
            raise ValueError(msg)
        return runtime

    def _payload(self) -> dict[str, object]:
        """Return all checksum-covered adapter fields."""
        return {
            "schema_version": self.schema_version,
            "adapter_id": self.adapter_id,
            "publication_method_id": self.publication_method_id,
            "target_scope_id": self.target_scope_id,
            "implementation_kind": self.implementation_kind,
            "implementation_payload_checksum": self.implementation_payload_checksum,
            "runner_family": self.runner_family,
            "repository_module": self.repository_module,
            "repository_symbol": self.repository_symbol,
            "runtime_status": (
                "bounded_smoke_runtime_verified"
                if self.implementation_kind
                in {"phase2_pipeline_smoke", "operator_growth_smoke", "tfim_operator_growth"}
                else "repository_callable_verified"
            ),
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the exact executable route."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> RepositoryRunnerAdapter:
        """Decode and verify one repository runner adapter.

        Returns:
            The normalized, operational adapter.

        Raises:
            ValueError: If aliases, checksums, or route metadata differ.
        """
        mapping = verify_sealed_mapping(value, expected_keys=_ADAPTER_KEYS, name="repository runner adapter")
        if mapping["schema_version"] != REPOSITORY_RUNNER_ADAPTER_SCHEMA_VERSION:
            msg = "Repository runner adapter uses an unsupported schema version."
            raise ValueError(msg)
        adapter = cls(
            publication_method_id=cast("str", mapping["publication_method_id"]),
            target_scope_id=cast("TargetScope", mapping["target_scope_id"]),
            implementation_kind=cast("ImplementationKind", mapping["implementation_kind"]),
            implementation_payload_checksum=cast("str", mapping["implementation_payload_checksum"]),
        )
        if mapping != adapter.to_dict():
            msg = "Repository runner adapter aliases or callable route changed during normalization."
            raise ValueError(msg)
        return adapter


@dataclass(frozen=True, slots=True)
class ExecutableImplementationEntry:
    """One preset/method/width implementation with its executable adapter."""

    preset: CatalogPreset
    publication_method_id: str
    target_scope_id: TargetScope
    strategy_schedule: TrainingStrategySchedule
    implementation_artifact: ExecutionImplementationArtifact
    runner_adapter: RepositoryRunnerAdapter
    schema_version: str = field(default=EXECUTABLE_IMPLEMENTATION_ENTRY_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Close every schedule, artifact, scope, preset, and callable identity link.

        Raises:
            TypeError: If a schedule, artifact, or adapter is not typed.
            ValueError: If any catalog, artifact, schedule, or adapter identity differs.
        """
        if self.preset not in _CATALOG_PRESETS:
            msg = "Executable catalog entries support only smoke, pilot, and screen presets."
            raise ValueError(msg)
        method = require_slug(self.publication_method_id, "publication_method_id")
        object.__setattr__(self, "publication_method_id", method)
        if self.target_scope_id not in {"primary_q6", "secondary_q12"}:
            msg = "target_scope_id must be primary_q6 or secondary_q12."
            raise ValueError(msg)
        if not isinstance(self.strategy_schedule, TrainingStrategySchedule):
            msg = "strategy_schedule must be a TrainingStrategySchedule."
            raise TypeError(msg)
        if not isinstance(self.implementation_artifact, ExecutionImplementationArtifact):
            msg = "implementation_artifact must be an ExecutionImplementationArtifact."
            raise TypeError(msg)
        artifact = self.implementation_artifact
        if (
            artifact.preset != self.preset
            or artifact.publication_method_id != method
            or artifact.target_scope_id != self.target_scope_id
            or artifact.strategy_schedule_checksum != self.strategy_schedule.content_checksum
        ):
            msg = "Catalog key, strategy schedule, and implementation artifact disagree."
            raise ValueError(msg)
        if not isinstance(self.runner_adapter, RepositoryRunnerAdapter):
            msg = "runner_adapter must be a RepositoryRunnerAdapter."
            raise TypeError(msg)
        expected_adapter = RepositoryRunnerAdapter.for_artifact(artifact)
        if self.runner_adapter != expected_adapter:
            msg = "Runner adapter does not bind the exact typed implementation artifact."
            raise ValueError(msg)

    @property
    def key(self) -> tuple[CatalogPreset, str, TargetScope]:
        """Unique preset, publication method, and width key."""
        return (self.preset, self.publication_method_id, self.target_scope_id)

    def resolve_callable(self) -> Callable[..., object]:
        """Resolve the operational repository runner after rechecking its payload.

        Returns:
            The concrete repository runner class or function.

        Raises:
            ValueError: If the adapter and artifact payload identities differ.
        """
        if self.runner_adapter.implementation_payload_checksum != (
            self.implementation_artifact.implementation_payload_checksum
        ):
            msg = "Runner adapter payload identity changed before resolution."
            raise ValueError(msg)
        return self.runner_adapter.resolve_callable()

    def smoke_runtime_program(self) -> SmokeRuntimeProgram:
        """Derive the checksum-bound bounded runtime for a smoke entry.

        Returns:
            A one-update pipeline or one-growth operator program.

        Raises:
            ValueError: If this entry is not a smoke implementation.
        """
        if self.preset != "training-smoke":
            msg = "Only training-smoke entries have bounded smoke runtime programs."
            raise ValueError(msg)
        return self.runner_adapter.materialize_smoke_runtime(self.implementation_artifact)

    def _payload(self) -> dict[str, object]:
        """Return all checksum-covered entry fields."""
        runtime = self.smoke_runtime_program() if self.preset == "training-smoke" else None
        return {
            "schema_version": self.schema_version,
            "preset": self.preset,
            "publication_method_id": self.publication_method_id,
            "target_scope_id": self.target_scope_id,
            "strategy_schedule": self.strategy_schedule.to_dict(),
            "strategy_schedule_checksum": self.strategy_schedule.content_checksum,
            "implementation_artifact": self.implementation_artifact.to_dict(),
            "implementation_artifact_checksum": self.implementation_artifact.content_checksum,
            "runner_adapter": self.runner_adapter.to_dict(),
            "runner_adapter_checksum": self.runner_adapter.content_checksum,
            "smoke_runtime_program": None if runtime is None else runtime.to_dict(),
            "smoke_runtime_program_checksum": None if runtime is None else runtime.content_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the implementation and executable route."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> ExecutableImplementationEntry:
        """Decode and verify one executable catalog entry.

        Returns:
            The normalized executable entry.

        Raises:
            ValueError: If nested aliases or derived runtime evidence differ.
        """
        mapping = verify_sealed_mapping(value, expected_keys=_ENTRY_KEYS, name="executable implementation entry")
        if mapping["schema_version"] != EXECUTABLE_IMPLEMENTATION_ENTRY_SCHEMA_VERSION:
            msg = "Executable implementation entry uses an unsupported schema version."
            raise ValueError(msg)
        entry = cls(
            preset=cast("CatalogPreset", mapping["preset"]),
            publication_method_id=cast("str", mapping["publication_method_id"]),
            target_scope_id=cast("TargetScope", mapping["target_scope_id"]),
            strategy_schedule=TrainingStrategySchedule.from_dict(mapping["strategy_schedule"]),
            implementation_artifact=ExecutionImplementationArtifact.from_dict(mapping["implementation_artifact"]),
            runner_adapter=RepositoryRunnerAdapter.from_dict(mapping["runner_adapter"]),
        )
        runtime = entry.smoke_runtime_program() if entry.preset == "training-smoke" else None
        aliases = {
            "strategy_schedule_checksum": entry.strategy_schedule.content_checksum,
            "implementation_artifact_checksum": entry.implementation_artifact.content_checksum,
            "runner_adapter_checksum": entry.runner_adapter.content_checksum,
            "smoke_runtime_program": (
                None if runtime is None else freeze_json_mapping(runtime.to_dict(), "expected smoke runtime program")
            ),
            "smoke_runtime_program_checksum": None if runtime is None else runtime.content_checksum,
            "content_checksum": entry.content_checksum,
        }
        if any(mapping[name] != expected for name, expected in aliases.items()):
            msg = "Executable implementation entry checksum aliases changed during normalization."
            raise ValueError(msg)
        return entry


def _paper_schedule(method_id: str) -> TrainingStrategySchedule:
    """Return the exact frozen schedule assigned to one paper method."""
    schedule_id = (
        "direct_noiseless_control"
        if method_id == "layerwise_bmpd_noiseless"
        else "resampled_each_update"
        if method_id in {"layerwise_bmpd_resampled", "spsa_layerwise"}
        else "direct_matched_fixed_crn"
    )
    return next(
        schedule for schedule in FrozenTrainingPolicyUniverse.frozen().schedules if schedule.schedule_id == schedule_id
    )


def _smoke_schedule(method_id: str) -> TrainingStrategySchedule:
    """Return one exact one-update schedule for a smoke implementation."""
    noisy = method_id not in {"layerwise_bmpd_noiseless", "energy_adapt_vqe"}
    trajectory_count = 1 if noisy else 0
    return TrainingStrategySchedule(
        schedule_id=f"training_smoke_{method_id}_primary_q6",
        noise_continuation=NoiseStrengthContinuation(
            start_update=0,
            end_update=0,
            start_strength_scale=1.0 if noisy else 0.0,
            target_strength_scale=1.0 if noisy else 0.0,
            interpolation="constant",
        ),
        trajectory_curriculum=TrajectoryCountCurriculum((TrajectoryCountStep(0, trajectory_count),)),
        sampling_policy=TrajectorySamplingPolicy(
            "resampled" if method_id in {"layerwise_bmpd_resampled", "spsa_layerwise"} else "fixed_crn"
        ),
        checkpoint_validation=CheckpointValidationPolicy(patience=None),
        phase_boundary=NoiselessPretrainNoisyFinetune(
            noiseless_pretrain_updates=0 if noisy else 1,
            noisy_finetune_updates=1 if noisy else 0,
        ),
        multistart=LimitedMultistartPlan(start_count=1, declared_cap=1),
        training_noise=(
            StandardNoiseMixture("matched", (NoiseMixtureComponent("depolarizing_1s_all", 1.0),))
            if noisy
            else StandardNoiseMixture("noiseless", ())
        ),
    )


def _pipeline_template(
    method_id: str,
    scope: TargetScope,
    *,
    smoke: bool,
) -> TrainingPipelineTemplate:
    """Build one exact repository template for a catalog entry.

    Returns:
        The width- and method-specific repository pipeline template.

    Raises:
        ValueError: If the method or secondary width has no pipeline template.
    """
    qubit_count = 6 if scope == "primary_q6" else 12
    training_count = 1 if smoke else 8
    validation_count = 1 if smoke else 256
    update_count = 1 if smoke else 200
    if method_id == "layerwise_bmpd_crn_v2":
        return build_layerwise_bmpd_crn_v2_template(
            training_trajectory_count=training_count,
            checkpoint_validation_trajectory_count=validation_count,
            qubit_count=qubit_count,
        )
    if method_id == "layerwise_bmpd_noiseless":
        return build_layerwise_bmpd_noiseless_template(
            checkpoint_validation_trajectory_count=validation_count,
            qubit_count=qubit_count,
        )
    if method_id == "fixed_depth_bmpd_crn":
        return build_fixed_depth_bmpd_crn_template(
            iteration_budget=update_count,
            training_trajectory_count=training_count,
            checkpoint_validation_trajectory_count=validation_count,
            qubit_count=qubit_count,
        )
    if scope != "primary_q6":
        msg = "Only the three frozen pilot pipeline methods support secondary_q12."
        raise ValueError(msg)
    if method_id == "layerwise_bmpd_resampled":
        return build_layerwise_bmpd_resampled_template(
            training_trajectory_count=training_count,
            checkpoint_validation_trajectory_count=validation_count,
        )
    if method_id == "layerwise_bmpd_cross_crn":
        return build_layerwise_bmpd_cross_crn_template(
            training_trajectory_count=training_count,
            checkpoint_validation_trajectory_count=validation_count,
        )
    if method_id == "parameter_shift_adam_layerwise":
        return build_parameter_shift_adam_layerwise_template(
            training_trajectory_count=training_count,
            checkpoint_validation_trajectory_count=validation_count,
            qubit_count=qubit_count,
        )
    if method_id == "spsa_layerwise":
        return build_spsa_layerwise_template(
            training_trajectory_count=training_count,
            checkpoint_validation_trajectory_count=validation_count,
            qubit_count=qubit_count,
        )
    if method_id == "impact_pruning_crn":
        return build_topdown_impact_iterative_template(
            qubit_count=qubit_count,
            fine_tune_mode="fixed_crn",
            fine_tune_iterations=update_count,
            fine_tune_trajectory_count=training_count,
            checkpoint_validation_trajectory_count=validation_count,
        )
    msg = "Method has no repository pipeline template."
    raise ValueError(msg)


def _artifact(
    preset: CatalogPreset,
    method_id: str,
    scope: TargetScope,
    schedule: TrainingStrategySchedule,
    *,
    screening_outer_trajectory_count: int,
    smoke_evaluation_trajectory_count: int,
) -> ExecutionImplementationArtifact:
    """Build one exact typed WP22B implementation artifact.

    Returns:
        The preset-, method-, and width-specific implementation artifact.
    """
    if preset == "training-smoke":
        if method_id == "adapt_style_state_preparation":
            kind: ImplementationKind = "operator_growth_smoke"
            payload = OperatorGrowthSmokeSpec.frozen(smoke_evaluation_trajectory_count)
        elif method_id == "energy_adapt_vqe":
            kind = "tfim_operator_growth"
            payload = EnergyAdaptSmokeSpec.frozen(smoke_evaluation_trajectory_count)
        else:
            kind = "phase2_pipeline_smoke"
            payload = PipelineSmokeSpec.frozen(
                _pipeline_template(method_id, scope, smoke=True),
                smoke_evaluation_trajectory_count,
            )
    elif method_id == "adapt_style_state_preparation":
        kind = "operator_growth"
        payload = OperatorGrowthExecutionSpec.for_screening(screening_outer_trajectory_count)
    else:
        kind = "phase2_pipeline"
        payload = _pipeline_template(method_id, scope, smoke=False)
    implementation_method_id = "topdown_impact_iterative" if method_id == "impact_pruning_crn" else method_id
    return ExecutionImplementationArtifact(
        artifact_id=f"wp22b_{preset}_{method_id}_{scope}",
        preset=preset,
        publication_method_id=method_id,
        implementation_kind=kind,
        implementation_method_id=implementation_method_id,
        target_scope_id=scope,
        strategy_schedule_checksum=schedule.content_checksum,
        implementation_payload=payload,
    )


def _entry(
    preset: CatalogPreset,
    method_id: str,
    scope: TargetScope,
    *,
    screening_outer_trajectory_count: int,
    smoke_evaluation_trajectory_count: int,
) -> ExecutableImplementationEntry:
    """Build one exact operational catalog entry.

    Returns:
        The implementation artifact closed to its concrete runner adapter.
    """
    schedule = _smoke_schedule(method_id) if preset == "training-smoke" else _paper_schedule(method_id)
    artifact = _artifact(
        preset,
        method_id,
        scope,
        schedule,
        screening_outer_trajectory_count=screening_outer_trajectory_count,
        smoke_evaluation_trajectory_count=smoke_evaluation_trajectory_count,
    )
    return ExecutableImplementationEntry(
        preset=preset,
        publication_method_id=method_id,
        target_scope_id=scope,
        strategy_schedule=schedule,
        implementation_artifact=artifact,
        runner_adapter=RepositoryRunnerAdapter.for_artifact(artifact),
    )


def _canonical_entries(
    screening_outer_trajectory_count: int,
    smoke_evaluation_trajectory_count: int,
) -> tuple[ExecutableImplementationEntry, ...]:
    """Return the one exact ordered 25-entry repository universe.

    Returns:
        Ten q6 smoke, six paired pilot, and nine q6 screen entries.
    """
    return (
        *(
            _entry(
                "training-smoke",
                method,
                "primary_q6",
                screening_outer_trajectory_count=screening_outer_trajectory_count,
                smoke_evaluation_trajectory_count=smoke_evaluation_trajectory_count,
            )
            for method in SMOKE_METHOD_IDS
        ),
        *(
            _entry(
                "paper-pilot",
                method,
                scope,
                screening_outer_trajectory_count=screening_outer_trajectory_count,
                smoke_evaluation_trajectory_count=smoke_evaluation_trajectory_count,
            )
            for method in PILOT_METHOD_IDS
            for scope in cast("tuple[TargetScope, ...]", ("primary_q6", "secondary_q12"))
        ),
        *(
            _entry(
                "paper-screen",
                method,
                "primary_q6",
                screening_outer_trajectory_count=screening_outer_trajectory_count,
                smoke_evaluation_trajectory_count=smoke_evaluation_trajectory_count,
            )
            for method in SCREEN_METHOD_IDS
        ),
    )


@dataclass(frozen=True, slots=True)
class RepositoryImplementationCatalog:
    """Complete WP22B smoke, pilot, screen, and dormant-confirm registry."""

    entries: tuple[ExecutableImplementationEntry, ...]
    screening_outer_trajectory_count: int
    smoke_evaluation_trajectory_count: int = 1
    catalog_id: str = "wp22b_repository_implementation_catalog"
    schema_version: str = field(default=IMPLEMENTATION_CATALOG_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Enforce literal cardinality, membership, uniqueness, and runner closure.

        Raises:
            TypeError: If entries or nested smoke payloads are not typed.
            ValueError: If counts, membership, evaluation work, or routes differ.
        """
        object.__setattr__(self, "catalog_id", require_slug(self.catalog_id, "catalog_id"))
        object.__setattr__(
            self,
            "screening_outer_trajectory_count",
            require_int(
                self.screening_outer_trajectory_count,
                "screening_outer_trajectory_count",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "smoke_evaluation_trajectory_count",
            require_int(
                self.smoke_evaluation_trajectory_count,
                "smoke_evaluation_trajectory_count",
                minimum=1,
            ),
        )
        entries = tuple(self.entries)
        if any(not isinstance(entry, ExecutableImplementationEntry) for entry in entries):
            msg = "entries must contain ExecutableImplementationEntry records."
            raise TypeError(msg)
        keys = tuple(entry.key for entry in entries)
        if len(keys) != len(set(keys)):
            msg = "Implementation catalog keys must be unique before any runner is called."
            raise ValueError(msg)
        expected = {
            *(("training-smoke", method, "primary_q6") for method in SMOKE_METHOD_IDS),
            *(
                ("paper-pilot", method, scope)
                for method in PILOT_METHOD_IDS
                for scope in ("primary_q6", "secondary_q12")
            ),
            *(("paper-screen", method, "primary_q6") for method in SCREEN_METHOD_IDS),
        }
        if len(entries) != IMPLEMENTATION_CATALOG_ENTRY_COUNT or set(keys) != expected:
            msg = "Implementation catalog must contain exactly 10 smoke, 6 pilot, and 9 q6 screen entries."
            raise ValueError(msg)
        canonical_entries = _canonical_entries(
            self.screening_outer_trajectory_count,
            self.smoke_evaluation_trajectory_count,
        )
        if entries != canonical_entries:
            msg = "Implementation catalog entries must equal the exact ordered canonical repository universe."
            raise ValueError(msg)
        for entry in entries:
            entry.resolve_callable()
            payload = entry.implementation_artifact.implementation_payload
            if entry.preset == "training-smoke":
                if not isinstance(payload, (PipelineSmokeSpec, OperatorGrowthSmokeSpec, EnergyAdaptSmokeSpec)):
                    msg = "Smoke catalog entries require executable tiny-limit wrappers."
                    raise TypeError(msg)
                if payload.outer_evaluation_policy.trajectory_count != self.smoke_evaluation_trajectory_count:
                    msg = "Smoke entry evaluation count differs from the catalog boundary."
                    raise ValueError(msg)
            elif entry.publication_method_id == "adapt_style_state_preparation":
                if (
                    not isinstance(payload, OperatorGrowthExecutionSpec)
                    or payload.outer_evaluation_policy.trajectory_count != self.screening_outer_trajectory_count
                ):
                    msg = "Screen operator-growth entry differs from the common fixed outer count."
                    raise ValueError(msg)
        object.__setattr__(self, "entries", entries)

    @classmethod
    def frozen(
        cls,
        *,
        screening_outer_trajectory_count: int,
        smoke_evaluation_trajectory_count: int = 1,
    ) -> RepositoryImplementationCatalog:
        """Build the complete repository-owned catalog.

        Returns:
            All 25 executable implementation entries under fixed counts.
        """
        outer = require_int(
            screening_outer_trajectory_count,
            "screening_outer_trajectory_count",
            minimum=1,
        )
        smoke = require_int(
            smoke_evaluation_trajectory_count,
            "smoke_evaluation_trajectory_count",
            minimum=1,
        )
        return cls(
            entries=_canonical_entries(outer, smoke),
            screening_outer_trajectory_count=outer,
            smoke_evaluation_trajectory_count=smoke,
        )

    def resolve(
        self,
        preset: Preset,
        publication_method_id: str,
        target_scope_id: TargetScope,
    ) -> ExecutableImplementationEntry:
        """Resolve an exact catalog entry or dormant confirmation alias.

        Confirmation returns the existing eligible q6 screen object.  It does
        not construct a ``paper-confirm`` artifact or authorize execution.

        Returns:
            The unique executable entry, with confirmation aliased by identity.

        Raises:
            KeyError: If the key is missing or ineligible for confirmation.
            ValueError: If the method or target scope has invalid syntax.
        """
        method = require_slug(publication_method_id, "publication_method_id")
        if target_scope_id not in {"primary_q6", "secondary_q12"}:
            msg = "target_scope_id must be primary_q6 or secondary_q12."
            raise ValueError(msg)
        resolved_preset = "paper-screen" if preset == "paper-confirm" else preset
        if preset == "paper-confirm" and (target_scope_id != "primary_q6" or method not in _CONFIRMATION_METHOD_IDS):
            msg = "Dormant confirmation aliases only final-eligible q6 screen configurations."
            raise KeyError(msg)
        if resolved_preset not in _CATALOG_PRESETS:
            msg = "Preset has no WP22B repository implementation catalog."
            raise KeyError(msg)
        key = (resolved_preset, method, target_scope_id)
        matches = tuple(entry for entry in self.entries if entry.key == key)
        if len(matches) != 1:
            msg = f"No unique executable implementation exists for key {key!r}."
            raise KeyError(msg)
        entry = matches[0]
        entry.resolve_callable()
        return entry

    @staticmethod
    def _confirmation_alias_policy() -> dict[str, object]:
        """Return the fixed dormant confirmation alias contract."""
        return {
            "source_preset": "paper-screen",
            "alias_preset": "paper-confirm",
            "target_scope_id": "primary_q6",
            "eligible_method_ids": sorted(_CONFIRMATION_METHOD_IDS),
            "reuse_rule": "same_entry_object_and_implementation_checksum",
            "new_configuration_allowed": False,
            "execution_authorized": False,
        }

    def _payload(self) -> dict[str, object]:
        """Return all checksum-covered catalog fields."""
        return {
            "schema_version": self.schema_version,
            "catalog_id": self.catalog_id,
            "screening_outer_trajectory_count": self.screening_outer_trajectory_count,
            "smoke_evaluation_trajectory_count": self.smoke_evaluation_trajectory_count,
            "confirmation_alias_policy": self._confirmation_alias_policy(),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the complete executable catalog."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> RepositoryImplementationCatalog:
        """Decode and verify the complete repository implementation catalog.

        Returns:
            The normalized executable catalog.

        Raises:
            TypeError: If the serialized entry collection is not a JSON array.
            ValueError: If schema, nested records, policy, or checksums differ.
        """
        mapping = verify_sealed_mapping(value, expected_keys=_CATALOG_KEYS, name="implementation catalog")
        if mapping["schema_version"] != IMPLEMENTATION_CATALOG_SCHEMA_VERSION:
            msg = "Implementation catalog uses an unsupported schema version."
            raise ValueError(msg)
        raw_entries = mapping["entries"]
        if type(raw_entries) is not tuple:
            msg = "entries must be a JSON array."
            raise TypeError(msg)
        catalog = cls(
            catalog_id=cast("str", mapping["catalog_id"]),
            screening_outer_trajectory_count=cast("int", mapping["screening_outer_trajectory_count"]),
            smoke_evaluation_trajectory_count=cast("int", mapping["smoke_evaluation_trajectory_count"]),
            entries=tuple(ExecutableImplementationEntry.from_dict(entry) for entry in raw_entries),
        )
        policy = require_mapping(mapping["confirmation_alias_policy"], "confirmation_alias_policy")
        require_exact_keys(policy, frozenset(catalog._confirmation_alias_policy()), "confirmation_alias_policy")
        if (
            policy
            != freeze_json_mapping(
                catalog._confirmation_alias_policy(),
                "expected confirmation alias policy",
            )
            or mapping["content_checksum"] != catalog.content_checksum
        ):
            msg = "Implementation catalog policy or checksum changed during normalization."
            raise ValueError(msg)
        return catalog

    @classmethod
    def from_json(cls, payload: str) -> RepositoryImplementationCatalog:
        """Decode canonical JSON into a verified implementation catalog.

        Returns:
            The normalized executable catalog.
        """
        return cls.from_dict(load_canonical_json_object(payload))


__all__ = [
    "EXECUTABLE_IMPLEMENTATION_ENTRY_SCHEMA_VERSION",
    "IMPLEMENTATION_CATALOG_ENTRY_COUNT",
    "IMPLEMENTATION_CATALOG_SCHEMA_VERSION",
    "OPERATOR_GROWTH_SMOKE_RUNTIME_PROGRAM_SCHEMA_VERSION",
    "PILOT_ENTRY_COUNT",
    "PIPELINE_SMOKE_RUNTIME_PROGRAM_SCHEMA_VERSION",
    "REPOSITORY_RUNNER_ADAPTER_SCHEMA_VERSION",
    "SCREEN_ENTRY_COUNT",
    "SMOKE_ENTRY_COUNT",
    "BoundPipelineSmokeRunner",
    "CatalogPreset",
    "ExecutableImplementationEntry",
    "OperatorGrowthSmokeExecution",
    "OperatorGrowthSmokeRuntimeProgram",
    "PipelineSmokeRuntimeProgram",
    "RepositoryImplementationCatalog",
    "RepositoryRunnerAdapter",
    "RunnerFamily",
    "SmokeRuntimeProgram",
]
