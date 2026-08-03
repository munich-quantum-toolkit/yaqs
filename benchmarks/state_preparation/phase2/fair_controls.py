# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""WP20 fair-control pipeline templates for noisy state preparation.

The sealed screening controls in this module isolate one treatment at a time
from the corrected WP19 layerwise profile.  Two additional controls retain
explicit secondary-only identities: Phase-I-style noiseless optimization with
fresh noisy testing, and unpruned circuits whose native resource relationship
to the primary cap is reported without claiming a false exact match.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

from benchmarks.state_preparation.circuits import compile_quantinuum_native
from benchmarks.state_preparation.constants import NOISELESS_NOISE_ID
from benchmarks.state_preparation.noise import FIXED_RATE_NOISE_DEFINITION_VERSION

from .canonical import canonical_checksum
from .layerwise_bmpd import (
    LAYERWISE_BMPD_INITIAL_SCALE,
    LAYERWISE_BMPD_PROFILE_VERSION,
    bmpd_parameter_count,
    bmpd_topology_id,
    build_layerwise_bmpd_crn_v2_template,
    create_bmpd_circuit_binding,
    initialize_layerwise_stage_parameters,
)
from .noisy_krotov import (
    NoisyKrotovCircuitBinding,
    NoisyKrotovStageExecution,
    NoisyKrotovStageFailure,
    execute_fixed_rate_krotov_stage,
)
from .pipeline import (
    CheckpointValidationConfig,
    TrainingPipelineConfig,
    TrainingPipelineTemplate,
    TrainingStageConfig,
    TrainingStageTemplate,
)
from .protocol import TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM
from .targets import MaterializedTarget
from .wp20_resources import measure_circuit_resources

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np
    from numpy.typing import NDArray

WP20_CONTROL_PROFILE_VERSION = "1"
WP20_PRIMARY_QUBIT_COUNT = 6
WP20_PRIMARY_BMPD_DEPTH = 4
WP20_PRIMARY_NATIVE_TWO_QUBIT_CAP_PER_EDGE = 12.0

LAYERWISE_BMPD_NOISELESS_METHOD_ID = "layerwise_bmpd_noiseless"
FIXED_DEPTH_BMPD_CRN_METHOD_ID = "fixed_depth_bmpd_crn"
LAYERWISE_BMPD_RESAMPLED_METHOD_ID = "layerwise_bmpd_resampled"
LAYERWISE_BMPD_CROSS_CRN_METHOD_ID = "layerwise_bmpd_cross_crn"
PHASE1_NOISELESS_CHECKPOINT_CONTROL_METHOD_ID = "phase1_noiseless_checkpoint_control"
UNPRUNED_DEEP_BMPD_METHOD_ID = "unpruned_deep_bmpd"

_STANDARD_NOISE_ID = "depolarizing_1s_all"
_CORRECTED_INITIALIZATION_RNG = "numpy_pcg64_standard_normal_v1"
_FRESH_TEST_POLICY_ID = "fresh_independent_standard_noise_v1"
_RESOURCE_METRIC = "native_two_qubit_gates_per_chain_edge"
_BMPD_TOPOLOGY_PATTERN = re.compile(r"bmpd_q(?P<qubits>[1-9][0-9]*)_d(?P<depth>[1-9][0-9]*)")
_SECONDARY_METHOD_IDS = frozenset({
    PHASE1_NOISELESS_CHECKPOINT_CONTROL_METHOD_ID,
    UNPRUNED_DEEP_BMPD_METHOD_ID,
})


def _require_positive_int(value: object, name: str, *, minimum: int = 1) -> int:
    """Return a positive built-in integer at or above ``minimum``.

    Returns:
        The validated integer.

    Raises:
        TypeError: If ``value`` is not a built-in integer.
        ValueError: If ``value`` is below the required minimum.
    """
    if type(value) is not int:
        msg = f"{name} must be a built-in integer."
        raise TypeError(msg)
    if value < minimum:
        msg = f"{name} must be at least {minimum}."
        raise ValueError(msg)
    return value


def _is_exact_one(value: object) -> bool:
    """Return whether a value is the frozen floating-point scalar one."""
    return type(value) is float and math.isclose(value, 1.0, rel_tol=0.0, abs_tol=0.0)


def _seed_domains() -> dict[str, object]:
    """Return the seven role-separated WP16 seed domains."""
    return {
        "initialization": "initialization",
        "optimizer_ordering": "optimizer_ordering",
        "training_trajectory": "training_trajectory",
        "checkpoint_validation": "checkpoint_validation",
        "pilot_evaluation": "pilot_evaluation",
        "screening_selection": "screening_selection",
        "confirmatory_test": "confirmatory_test",
    }


def _materialization_policy() -> dict[str, object]:
    """Return the primary compiler, routing, and noise-placement policy."""
    return {
        "policy_id": "native_chain_v1",
        "compiler_policy_id": "quantinuum_rzz_chain_v1",
        "connectivity_id": "linear_chain",
        "routing_policy_id": "identity_no_swap",
        "optimization_level": 0,
        "noise_placement": "logical_parameterized_gates",
        "parameter_source": "selected_checkpoint",
    }


def _disabled_checkpoint_policy() -> dict[str, object]:
    """Return the seed-free disabled checkpoint-selection policy."""
    policy = CheckpointValidationConfig.disabled().to_dict()
    del policy["seed"]
    return policy


def _clone_stage(
    stage: TrainingStageTemplate,
    *,
    policy_updates: dict[str, object] | None = None,
    binding_updates: dict[str, object] | None = None,
) -> TrainingStageTemplate:
    """Copy one immutable stage template with explicit field replacements.

    Returns:
        The independently validated replacement stage.
    """
    policy = dict(stage.stage_policy)
    policy.update({} if policy_updates is None else policy_updates)
    bindings = dict(stage.seed_bindings)
    bindings.update({} if binding_updates is None else binding_updates)
    return TrainingStageTemplate(stage_policy=policy, seed_bindings=bindings)


def _rebind_stages(
    stages: tuple[TrainingStageTemplate, ...],
    *,
    prefix: str,
) -> tuple[TrainingStageTemplate, ...]:
    """Give a nonmatched method its own symbolic random-stream bindings.

    Returns:
        Stages with method-specific initialization, optimization, training, and
        checkpoint-validation bindings.
    """
    rebound: list[TrainingStageTemplate] = []
    for stage in stages:
        bindings: dict[str, object] = {
            role: (None if value is None else f"{prefix}_{stage.stage_id}_{role}")
            for role, value in stage.seed_bindings.items()
        }
        rebound.append(_clone_stage(stage, binding_updates=bindings))
    return tuple(rebound)


def _layerwise_variant(
    *,
    method_id: str,
    template_id: str,
    training_trajectory_count: int,
    checkpoint_validation_trajectory_count: int,
    trajectory_update: Literal["independent", "cross"],
    sampling_policy: Literal["resampled", "crn_fixed"],
    binding_prefix: str,
) -> TrainingPipelineTemplate:
    """Build a modern q6 layerwise treatment variant from the WP19 reference.

    Returns:
        A separately identified five-stage pipeline template.
    """
    training_count = _require_positive_int(training_trajectory_count, "training_trajectory_count")
    validation_count = _require_positive_int(
        checkpoint_validation_trajectory_count,
        "checkpoint_validation_trajectory_count",
    )
    reference = build_layerwise_bmpd_crn_v2_template(
        training_trajectory_count=training_count,
        checkpoint_validation_trajectory_count=validation_count,
    )
    final = _clone_stage(
        reference.stages[-1],
        policy_updates={
            "trajectory_update": trajectory_update,
            "sampling_policy": sampling_policy,
        },
    )
    stages = _rebind_stages((*reference.stages[:-1], final), prefix=binding_prefix)
    return TrainingPipelineTemplate(
        template_id=template_id,
        preregistration_checksum=TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM,
        target_scope_id="primary_q6",
        ansatz_family="bmpd_brickwall",
        method_id=method_id,
        method_version=WP20_CONTROL_PROFILE_VERSION,
        resource_stratum_id="primary_cap_12",
        stages=stages,
        seed_domains=_seed_domains(),
        final_materialization_policy=_materialization_policy(),
    )


def build_layerwise_bmpd_noiseless_template(
    *,
    checkpoint_validation_trajectory_count: int,
    qubit_count: int = WP20_PRIMARY_QUBIT_COUNT,
) -> TrainingPipelineTemplate:
    """Build the exact q6 or q12 noiseless match for layerwise v2.

    The final stage retains 200 Krotov updates and the same separately fixed
    noisy checkpoint-validation ensemble.  Only the final training treatment
    is removed, so the mechanically derived matching projection is identical
    to that of :func:`build_layerwise_bmpd_crn_v2_template`.

    Returns:
        The preregistered matched noiseless comparator.
    """
    validation_count = _require_positive_int(
        checkpoint_validation_trajectory_count,
        "checkpoint_validation_trajectory_count",
    )
    reference = build_layerwise_bmpd_crn_v2_template(
        training_trajectory_count=1,
        checkpoint_validation_trajectory_count=validation_count,
        qubit_count=qubit_count,
    )
    final = _clone_stage(
        reference.stages[-1],
        policy_updates={
            "training_noise_id": NOISELESS_NOISE_ID,
            "noise_strength_scale": None,
            "tjm_dt": None,
            "trajectory_count": 0,
            "trajectory_update": None,
            "sampling_policy": "none",
            "crn_refresh_interval": None,
        },
        binding_updates={"training": None},
    )
    q12 = qubit_count == 12
    return TrainingPipelineTemplate(
        template_id=("layerwise_bmpd_noiseless_default_q12_projection" if q12 else "layerwise_bmpd_noiseless_default"),
        preregistration_checksum=TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM,
        target_scope_id="secondary_q12" if q12 else "primary_q6",
        ansatz_family="bmpd_brickwall",
        method_id=LAYERWISE_BMPD_NOISELESS_METHOD_ID,
        method_version=LAYERWISE_BMPD_PROFILE_VERSION,
        resource_stratum_id="primary_cap_12",
        stages=(*reference.stages[:-1], final),
        seed_domains=reference.seed_domains,
        final_materialization_policy=reference.final_materialization_policy,
    )


def build_independent_fixed_crn_control_template(
    *,
    training_trajectory_count: int,
    checkpoint_validation_trajectory_count: int,
) -> TrainingPipelineTemplate:
    """Return the WP19 v2 profile as WP20's independent fixed-CRN control.

    Returns:
        The unchanged corrected WP19 template.
    """
    return build_layerwise_bmpd_crn_v2_template(
        training_trajectory_count=training_trajectory_count,
        checkpoint_validation_trajectory_count=checkpoint_validation_trajectory_count,
    )


def build_layerwise_bmpd_resampled_template(
    *,
    training_trajectory_count: int,
    checkpoint_validation_trajectory_count: int,
) -> TrainingPipelineTemplate:
    """Build modern independent layerwise training with per-update resampling.

    Returns:
        The preregistered resampled control template.
    """
    return _layerwise_variant(
        method_id=LAYERWISE_BMPD_RESAMPLED_METHOD_ID,
        template_id="layerwise_bmpd_resampled_default",
        training_trajectory_count=training_trajectory_count,
        checkpoint_validation_trajectory_count=checkpoint_validation_trajectory_count,
        trajectory_update="independent",
        sampling_policy="resampled",
        binding_prefix="layerwise_resampled",
    )


def build_layerwise_bmpd_cross_crn_template(
    *,
    training_trajectory_count: int,
    checkpoint_validation_trajectory_count: int,
) -> TrainingPipelineTemplate:
    """Build modern cross-update layerwise training with fixed normalized CRN.

    This control changes the update rule only.  It deliberately retains the
    standard Phase-II noise condition, modern hash-derived seed domains, and
    normalized fixed-map replay; the WP19 legacy compatibility identity is
    never requested.

    Returns:
        The preregistered cross-CRN control template.
    """
    return _layerwise_variant(
        method_id=LAYERWISE_BMPD_CROSS_CRN_METHOD_ID,
        template_id="layerwise_bmpd_cross_crn_default",
        training_trajectory_count=training_trajectory_count,
        checkpoint_validation_trajectory_count=checkpoint_validation_trajectory_count,
        trajectory_update="cross",
        sampling_policy="crn_fixed",
        binding_prefix="layerwise_cross_crn",
    )


def _direct_stage(
    *,
    stage_id: str,
    depth: int,
    qubit_count: int = WP20_PRIMARY_QUBIT_COUNT,
    iteration_budget: int,
    initialization_scale: float,
    learning_rate: float,
    noisy: bool,
    training_trajectory_count: int = 0,
    checkpoint_validation_policy: dict[str, object] | None = None,
    binding_prefix: str,
) -> TrainingStageTemplate:
    """Build one full-depth direct-training stage at the requested width.

    Returns:
        A validated stage with explicit initialization and optimizer work.

    Raises:
        ValueError: If counts or noisy-trajectory semantics are invalid.
    """
    depth_value = _require_positive_int(depth, "depth")
    budget = _require_positive_int(iteration_budget, "iteration_budget")
    training_count = _require_positive_int(
        training_trajectory_count,
        "training_trajectory_count",
        minimum=0,
    )
    if noisy != (training_count > 0):
        msg = "Noisy direct stages require positive trajectories; noiseless stages require zero trajectories."
        raise ValueError(msg)
    validation = _disabled_checkpoint_policy() if checkpoint_validation_policy is None else checkpoint_validation_policy
    validation_active = cast("int", validation["trajectory_count"]) > 0
    return TrainingStageTemplate(
        stage_policy={
            "stage_index": 0,
            "stage_id": stage_id,
            "stage_kind": "optimize",
            "input_topology_id": None,
            "output_topology_id": bmpd_topology_id(qubit_count, depth_value),
            "input_parameter_count": 0,
            "output_parameter_count": bmpd_parameter_count(qubit_count, depth_value),
            "parameter_transfer_rule": "initialize_random_normal",
            "optimizer_id": "krotov",
            "optimizer_hyperparameters": {
                "learning_rate": learning_rate,
                "schedule": "constant" if not noisy else "exp",
                "decay": 0.0 if not noisy else 0.01,
                "initialization_rng": _CORRECTED_INITIALIZATION_RNG,
                "initialization_scale": initialization_scale,
            },
            "iteration_budget": budget,
            "training_noise_id": _STANDARD_NOISE_ID if noisy else NOISELESS_NOISE_ID,
            "noise_definition_version": FIXED_RATE_NOISE_DEFINITION_VERSION,
            "noise_strength_scale": 1.0 if noisy else None,
            "tjm_dt": 1.0 if noisy else None,
            "trajectory_count": training_count,
            "trajectory_update": "independent" if noisy else None,
            "sampling_policy": "crn_fixed" if noisy else "none",
            "crn_refresh_interval": None,
            "checkpoint_validation_policy": validation,
            "pruning_rule": "none",
            "pruning_threshold": None,
            "max_bond_dimension": None,
            "svd_threshold": 0.0,
            "truncation_mode": "discarded_weight",
            "min_bond_dimension": 1,
        },
        seed_bindings={
            "initialization": f"{binding_prefix}_initialization",
            "optimizer": f"{binding_prefix}_optimizer",
            "training": f"{binding_prefix}_training" if noisy else None,
            "checkpoint_validation": f"{binding_prefix}_checkpoint_validation" if validation_active else None,
        },
    )


def build_fixed_depth_bmpd_crn_template(
    *,
    iteration_budget: int,
    training_trajectory_count: int,
    checkpoint_validation_trajectory_count: int,
    qubit_count: int = WP20_PRIMARY_QUBIT_COUNT,
) -> TrainingPipelineTemplate:
    """Build q6 or q12 direct depth-four Krotov training with fixed CRN.

    The optimizer budget is mandatory because WP20 does not silently decide
    whether a direct method receives 200 updates or the layerwise method's 600
    total updates.  WP22 must freeze that comparison choice from pilot evidence.

    Returns:
        The preregistered fixed-depth noisy control template.
    """
    budget = _require_positive_int(iteration_budget, "iteration_budget")
    training_count = _require_positive_int(training_trajectory_count, "training_trajectory_count")
    validation_count = _require_positive_int(
        checkpoint_validation_trajectory_count,
        "checkpoint_validation_trajectory_count",
    )
    reference = build_layerwise_bmpd_crn_v2_template(
        training_trajectory_count=training_count,
        checkpoint_validation_trajectory_count=validation_count,
        qubit_count=qubit_count,
    )
    validation = dict(cast("dict[str, object]", reference.stages[-1].stage_policy["checkpoint_validation_policy"]))
    validation["cadence"] = min(budget, cast("int", validation["cadence"]))
    stage = _direct_stage(
        stage_id="direct_depth4_noisy_training",
        depth=WP20_PRIMARY_BMPD_DEPTH,
        qubit_count=qubit_count,
        iteration_budget=budget,
        initialization_scale=LAYERWISE_BMPD_INITIAL_SCALE,
        learning_rate=0.2,
        noisy=True,
        training_trajectory_count=training_count,
        checkpoint_validation_policy=validation,
        binding_prefix="fixed_depth_bmpd_crn",
    )
    q12 = qubit_count == 12
    return TrainingPipelineTemplate(
        template_id=(f"fixed_depth_bmpd_crn_b{budget}_q12_projection" if q12 else f"fixed_depth_bmpd_crn_b{budget}"),
        preregistration_checksum=TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM,
        target_scope_id="secondary_q12" if q12 else "primary_q6",
        ansatz_family="bmpd_brickwall",
        method_id=FIXED_DEPTH_BMPD_CRN_METHOD_ID,
        method_version=WP20_CONTROL_PROFILE_VERSION,
        resource_stratum_id="primary_cap_12",
        stages=(stage,),
        seed_domains=_seed_domains(),
        final_materialization_policy=_materialization_policy(),
    )


def _fixed_depth_target_identity(target: object) -> Mapping[str, object]:
    """Return the identity of one authorized q6 or secondary-q12 target.

    Returns:
        The immutable target agreement-ledger fields.

    Raises:
        TypeError: If the target is not an authorized Phase-II materialization.
    """
    if not isinstance(target, MaterializedTarget):
        msg = "Fixed-depth pipeline execution requires a typed Phase-II materialized target."
        raise TypeError(msg)
    return target.identity_dict()


@dataclass(frozen=True, slots=True)
class FixedDepthBMPDStageRunner:
    """Bind a q6 or secondary-q12 direct BMPD control to the WP17 adapter."""

    pipeline: TrainingPipelineConfig
    target: MaterializedTarget

    def __post_init__(self) -> None:
        """Validate method, target, topology, and treatment before optimizer work.

        Raises:
            TypeError: If the pipeline or target has the wrong record type.
            ValueError: If the resolved control differs from the registered
                q6/q12 depth-four independent fixed-CRN treatment.
        """
        if not isinstance(self.pipeline, TrainingPipelineConfig):
            msg = "pipeline must be a TrainingPipelineConfig."
            raise TypeError(msg)
        if self.pipeline.method_id != FIXED_DEPTH_BMPD_CRN_METHOD_ID:
            msg = "FixedDepthBMPDStageRunner accepts only the registered fixed-depth BMPD method."
            raise ValueError(msg)
        expected_scope = {6: "primary_q6", 12: "secondary_q12"}.get(self.pipeline.qubit_count)
        if (
            self.pipeline.method_version != WP20_CONTROL_PROFILE_VERSION
            or self.pipeline.template.target_scope_id != expected_scope
            or self.pipeline.ansatz_family != "bmpd_brickwall"
            or self.pipeline.template.resource_stratum_id != "primary_cap_12"
            or expected_scope is None
            or len(self.pipeline.stages) != 1
        ):
            msg = "The registered fixed-depth BMPD control requires one q6 or secondary-q12 training stage."
            raise ValueError(msg)
        identity = _fixed_depth_target_identity(self.target)
        expected = {
            "target_instance_id": self.pipeline.target_instance_id,
            "target_instance_spec_checksum": self.pipeline.target_instance_spec_checksum,
            "target_manifest_checksum": self.pipeline.target_population_manifest_checksum,
            "family_id": self.pipeline.target_family_id,
            "stratum_id": self.pipeline.target_stratum_id,
            "qubit_count": self.pipeline.qubit_count,
        }
        if any(identity[name] != value for name, value in expected.items()):
            msg = "Materialized target identity does not match the resolved fixed-depth pipeline."
            raise ValueError(msg)
        stage = self.pipeline.stages[0]
        self._binding(stage)
        validation = stage.checkpoint_validation
        exact_smoke_validation = (
            self.pipeline.template.template_id == "wp22b_smoke_runtime_fixed_depth_bmpd_crn"
            and self.pipeline.qubit_count == 6
            and self.pipeline.data_role == "development"
            and stage.iteration_budget == 1
            and stage.trajectory_count == 1
            and not validation.enabled
            and validation == CheckpointValidationConfig.disabled()
        )
        exact_production_validation = (
            validation.enabled
            and validation.noise_id == _STANDARD_NOISE_ID
            and validation.noise_definition_version == FIXED_RATE_NOISE_DEFINITION_VERSION
            and _is_exact_one(validation.noise_strength_scale)
            and _is_exact_one(validation.tjm_dt)
            and validation.sampling_policy == "crn_fixed"
            and validation.ensemble_refresh_interval is None
            and validation.selection_rule == "best_validation_fidelity"
            and validation.tie_breaker == "earliest_iteration"
        )
        expected_treatment = (
            stage.stage_index == 0
            and stage.stage_kind == "optimize"
            and stage.input_topology_id is None
            and stage.input_parameter_count == 0
            and stage.parameter_transfer_rule == "initialize_random_normal"
            and stage.optimizer_id == "krotov"
            and dict(stage.optimizer_hyperparameters)
            == {
                "learning_rate": 0.2,
                "schedule": "exp",
                "decay": 0.01,
                "initialization_rng": _CORRECTED_INITIALIZATION_RNG,
                "initialization_scale": LAYERWISE_BMPD_INITIAL_SCALE,
            }
            and stage.training_noise_id == _STANDARD_NOISE_ID
            and stage.noise_definition_version == FIXED_RATE_NOISE_DEFINITION_VERSION
            and _is_exact_one(stage.noise_strength_scale)
            and _is_exact_one(stage.tjm_dt)
            and stage.trajectory_update == "independent"
            and stage.sampling_policy == "crn_fixed"
            and stage.crn_refresh_interval is None
            and (exact_production_validation or exact_smoke_validation)
        )
        if not expected_treatment:
            msg = "Resolved fixed-depth stage differs from the registered direct independent fixed-CRN treatment."
            raise ValueError(msg)

    def _binding(self, stage: TrainingStageConfig) -> NoisyKrotovCircuitBinding:
        """Build the circuit named by one exact fixed-depth stage.

        Returns:
            The width-matched depth-four circuit and frozen WP17 policy binding.

        Raises:
            ValueError: If topology width, depth, or parameter count differs
                from the registered fixed-depth control.
        """
        match = _BMPD_TOPOLOGY_PATTERN.fullmatch(stage.output_topology_id)
        if match is None:
            msg = "Fixed-depth stage output_topology_id is not a BMPD topology."
            raise ValueError(msg)
        qubits = int(match.group("qubits"))
        depth = int(match.group("depth"))
        if qubits != self.pipeline.qubit_count or depth != WP20_PRIMARY_BMPD_DEPTH:
            width_label = f"q{self.pipeline.qubit_count}"
            msg = f"The registered fixed-depth control requires the {width_label} depth-four BMPD topology."
            raise ValueError(msg)
        binding = create_bmpd_circuit_binding(qubits, depth)
        if (
            binding.topology_id != stage.output_topology_id
            or binding.circuit.num_params != stage.output_parameter_count
        ):
            msg = "Resolved fixed-depth topology differs from its BMPD circuit binding."
            raise ValueError(msg)
        return binding

    def __call__(
        self,
        stage: TrainingStageConfig,
        predecessor_parameters: NDArray[np.float64] | None,
    ) -> NoisyKrotovStageExecution | NoisyKrotovStageFailure:
        """Initialize and execute the exact direct fixed-depth stage.

        Returns:
            The complete WP17 execution or structured stage failure.

        Raises:
            ValueError: If the supplied stage is not the pipeline's sole stage
                or it is given predecessor parameters.
        """
        if stage != self.pipeline.stages[0]:
            msg = "Stage does not belong to this resolved fixed-depth pipeline."
            raise ValueError(msg)
        binding = self._binding(stage)
        initial_theta = initialize_layerwise_stage_parameters(stage, predecessor_parameters)
        return execute_fixed_rate_krotov_stage(
            stage,
            binding,
            self.target,
            initial_theta,
            compatibility_method_id=None,
        )

    def circuit_statistics(self, stage: TrainingStageConfig) -> Mapping[str, object]:
        """Return complete logical/native resource evidence for this stage.

        Returns:
            Circuit topology, depth, and compiler-derived WP20 resources.

        Raises:
            ValueError: If the supplied stage is not the pipeline's sole stage.
        """
        if stage != self.pipeline.stages[0]:
            msg = "Stage does not belong to this resolved fixed-depth pipeline."
            raise ValueError(msg)
        binding = self._binding(stage)
        resources = measure_circuit_resources(binding.circuit)
        return {
            "topology_id": stage.output_topology_id,
            "parameter_count": stage.output_parameter_count,
            "qubit_count": self.pipeline.qubit_count,
            "bmpd_depth": WP20_PRIMARY_BMPD_DEPTH,
            "logical_gate_count": len(binding.circuit.gates),
            "logical_two_qubit_gate_count": resources.logical_two_qubit_gates,
            "native_two_qubit_gate_count": resources.native_two_qubit_gates,
            "native_two_qubit_gates_per_chain_edge": list(resources.native_two_qubit_gates_per_chain_edge),
            "circuit_resource_metrics": resources.to_dict(),
        }


def _native_two_qubit_count_per_edge(depth: int) -> float:
    """Compile q6 BMPD and return its uniform native RZZ count per edge.

    Returns:
        The actual compiler-derived native two-qubit count on each chain edge.

    Raises:
        ValueError: If the compiled circuit is not a uniform nearest-neighbor
            chain, which would invalidate this control's resource descriptor.
    """
    depth_value = _require_positive_int(depth, "depth")
    native = compile_quantinuum_native(
        create_bmpd_circuit_binding(WP20_PRIMARY_QUBIT_COUNT, depth_value).circuit,
    ).circuit
    counts = {(site, site + 1): 0 for site in range(WP20_PRIMARY_QUBIT_COUNT - 1)}
    for gate in native.gates:
        if len(gate.sites) != 2:
            continue
        edge = tuple(sorted(gate.sites))
        if edge not in counts:
            msg = "Unpruned BMPD compilation emitted a non-nearest-neighbor two-qubit gate."
            raise ValueError(msg)
        counts[cast("tuple[int, int]", edge)] += 1
    unique_counts = set(counts.values())
    if len(unique_counts) != 1:
        msg = "Unpruned BMPD compilation is not uniform across chain edges."
        raise ValueError(msg)
    return float(next(iter(unique_counts)))


@dataclass(frozen=True, slots=True)
class NativeBudgetDescriptor:
    """Truthful relationship between one unpruned circuit and the primary cap."""

    attained_per_chain_edge: float
    match_status: Literal["exact_match", "below_cap_unmatched", "above_cap_unmatched"] = field(init=False)
    residual_gap: float = field(init=False)
    resource_excess: float = field(init=False)
    metric: str = field(default=_RESOURCE_METRIC, init=False)
    cap_per_chain_edge: float = field(default=WP20_PRIMARY_NATIVE_TWO_QUBIT_CAP_PER_EDGE, init=False)

    def __post_init__(self) -> None:
        """Validate the compiled resource and derive nonoverlapping status fields.

        Raises:
            TypeError: If the attained resource is not a finite float.
            ValueError: If the attained resource is negative.
        """
        if type(self.attained_per_chain_edge) is not float or not math.isfinite(self.attained_per_chain_edge):
            msg = "attained_per_chain_edge must be a finite float."
            raise TypeError(msg)
        if self.attained_per_chain_edge < 0.0:
            msg = "attained_per_chain_edge must be nonnegative."
            raise ValueError(msg)
        residual = max(0.0, self.cap_per_chain_edge - self.attained_per_chain_edge)
        excess = max(0.0, self.attained_per_chain_edge - self.cap_per_chain_edge)
        status: Literal["exact_match", "below_cap_unmatched", "above_cap_unmatched"]
        if math.isclose(
            self.attained_per_chain_edge,
            self.cap_per_chain_edge,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            status = "exact_match"
        elif residual > 0.0:
            status = "below_cap_unmatched"
        else:
            status = "above_cap_unmatched"
        object.__setattr__(self, "match_status", status)
        object.__setattr__(self, "residual_gap", residual)
        object.__setattr__(self, "resource_excess", excess)

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-native resource descriptor."""
        return {
            "metric": self.metric,
            "cap_per_chain_edge": self.cap_per_chain_edge,
            "attained_per_chain_edge": self.attained_per_chain_edge,
            "match_status": self.match_status,
            "residual_gap": self.residual_gap,
            "resource_excess": self.resource_excess,
        }


@dataclass(frozen=True, slots=True)
class SecondaryControlDescriptor:
    """Identity and resource evidence for a control excluded from screening."""

    template: TrainingPipelineTemplate
    resource: NativeBudgetDescriptor
    depth: int
    iteration_budget: int
    evaluation_policy_id: str = field(default=_FRESH_TEST_POLICY_ID, init=False)
    evaluation_noise_id: str = field(default=_STANDARD_NOISE_ID, init=False)
    evaluation_noise_definition_version: str = field(default=FIXED_RATE_NOISE_DEFINITION_VERSION, init=False)
    evaluation_noise_strength_scale: float = field(default=1.0, init=False)
    evaluation_tjm_dt: float = field(default=1.0, init=False)
    qubit_count: int = field(default=WP20_PRIMARY_QUBIT_COUNT, init=False)
    screening_eligible: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Validate the distinct secondary-only method namespace.

        Raises:
            TypeError: If the template or resource has the wrong record type.
            ValueError: If the method identity is not reserved for secondary use.
        """
        if not isinstance(self.template, TrainingPipelineTemplate):
            msg = "template must be a TrainingPipelineTemplate."
            raise TypeError(msg)
        if self.template.method_id not in _SECONDARY_METHOD_IDS:
            msg = "Secondary controls require a dedicated non-screening method identity."
            raise ValueError(msg)
        expected_scope = {
            PHASE1_NOISELESS_CHECKPOINT_CONTROL_METHOD_ID: "phase1_fixture",
            UNPRUNED_DEEP_BMPD_METHOD_ID: "primary_q6",
        }[self.template.method_id]
        if self.template.target_scope_id != expected_scope:
            msg = "Secondary control method identity does not match its isolated target scope."
            raise ValueError(msg)
        if not isinstance(self.resource, NativeBudgetDescriptor):
            msg = "resource must be a NativeBudgetDescriptor."
            raise TypeError(msg)
        if type(self.depth) is not int or self.depth < 1:
            msg = "depth must be a positive built-in integer."
            raise ValueError(msg)
        if type(self.iteration_budget) is not int or self.iteration_budget < 1:
            msg = "iteration_budget must be a positive built-in integer."
            raise ValueError(msg)
        if len(self.template.stages) != 1:
            msg = "Secondary direct controls require exactly one training stage."
            raise ValueError(msg)
        stage = self.template.stages[0].stage_policy
        if (
            stage["output_topology_id"] != bmpd_topology_id(self.qubit_count, self.depth)
            or stage["iteration_budget"] != self.iteration_budget
        ):
            msg = "Secondary control depth and work must match its q6 template."
            raise ValueError(msg)
        expected_resource = NativeBudgetDescriptor(_native_two_qubit_count_per_edge(self.depth))
        if self.resource != expected_resource:
            msg = "Secondary control resource evidence must be compiler-derived from its declared depth."
            raise ValueError(msg)

    @property
    def content_checksum(self) -> str:
        """Checksum-bind the template, evaluation semantics, and resource claim."""
        return canonical_checksum(self.to_dict())

    def require_data_role(self, data_role: str) -> None:
        """Reject use of this descriptor in sealed screening or confirmation.

        Raises:
            ValueError: If the requested role is screening or confirmatory.
        """
        if data_role in {"screening_selection", "confirmatory"}:
            msg = f"Secondary-only control {self.template.method_id!r} cannot enter sealed {data_role}."
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        """Return detached checksum content without duplicating the template."""
        return {
            "template_configuration_checksum": self.template.configuration_checksum,
            "method_id": self.template.method_id,
            "qubit_count": self.qubit_count,
            "depth": self.depth,
            "iteration_budget": self.iteration_budget,
            "evaluation_policy_id": self.evaluation_policy_id,
            "evaluation_noise_id": self.evaluation_noise_id,
            "evaluation_noise_definition_version": self.evaluation_noise_definition_version,
            "evaluation_noise_strength_scale": self.evaluation_noise_strength_scale,
            "evaluation_tjm_dt": self.evaluation_tjm_dt,
            "screening_eligible": self.screening_eligible,
            "resource": self.resource.to_dict(),
        }


def _secondary_noiseless_control(
    *,
    method_id: str,
    template_id: str,
    target_scope_id: Literal["primary_q6", "phase1_fixture"],
    depth: int,
    iteration_budget: int,
    initialization_scale: float,
    binding_prefix: str,
) -> SecondaryControlDescriptor:
    """Build one explicitly secondary direct-noiseless control.

    Returns:
        The template and compiler-derived resource descriptor.
    """
    depth_value = _require_positive_int(depth, "depth")
    budget = _require_positive_int(iteration_budget, "iteration_budget")
    stage = _direct_stage(
        stage_id="direct_noiseless_training",
        depth=depth_value,
        iteration_budget=budget,
        initialization_scale=initialization_scale,
        learning_rate=0.2,
        noisy=False,
        binding_prefix=binding_prefix,
    )
    resource = NativeBudgetDescriptor(_native_two_qubit_count_per_edge(depth_value))
    template = TrainingPipelineTemplate(
        template_id=template_id,
        preregistration_checksum=TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM,
        target_scope_id=target_scope_id,
        ansatz_family="bmpd_brickwall_unpruned",
        method_id=method_id,
        method_version=WP20_CONTROL_PROFILE_VERSION,
        resource_stratum_id=f"unpruned_d{depth_value}_{resource.match_status}",
        stages=(stage,),
        seed_domains=_seed_domains(),
        final_materialization_policy=_materialization_policy(),
    )
    return SecondaryControlDescriptor(
        template=template,
        resource=resource,
        depth=depth_value,
        iteration_budget=budget,
    )


def build_phase1_noiseless_test_control(
    *,
    depth: int,
    iteration_budget: int,
) -> SecondaryControlDescriptor:
    """Build Phase-I-style noiseless training followed by fresh noisy testing.

    Both depth and optimizer budget are mandatory and identity-bearing.  The
    descriptor fixes fresh independent standard-noise evaluation, while the
    evaluation trajectory count remains an explicit downstream pilot choice.

    Returns:
        A q6 Phase-I-fixture secondary control and its native resource status.
    """
    depth_value = _require_positive_int(depth, "depth")
    budget = _require_positive_int(iteration_budget, "iteration_budget")
    return _secondary_noiseless_control(
        method_id=PHASE1_NOISELESS_CHECKPOINT_CONTROL_METHOD_ID,
        template_id=f"phase1_noiseless_checkpoint_q6_d{depth_value}_b{budget}",
        target_scope_id="phase1_fixture",
        depth=depth_value,
        iteration_budget=budget,
        initialization_scale=0.1,
        binding_prefix="phase1_noiseless_checkpoint",
    )


def build_unpruned_deep_control(
    *,
    depth: int,
    iteration_budget: int,
) -> SecondaryControlDescriptor:
    """Build an unpruned q6 control with explicit depth, work, and match status.

    Depth four is the exact unpruned primary-cap circuit.  Deeper circuits are
    retained as explicitly unmatched controls with positive resource excess;
    they cannot be described as fixed-resource matches.

    Returns:
        A secondary-only unpruned control and compiler-derived resource status.
    """
    depth_value = _require_positive_int(depth, "depth", minimum=WP20_PRIMARY_BMPD_DEPTH)
    budget = _require_positive_int(iteration_budget, "iteration_budget")
    return _secondary_noiseless_control(
        method_id=UNPRUNED_DEEP_BMPD_METHOD_ID,
        template_id=f"unpruned_deep_bmpd_q6_d{depth_value}_b{budget}",
        target_scope_id="primary_q6",
        depth=depth_value,
        iteration_budget=budget,
        initialization_scale=0.1,
        binding_prefix=f"unpruned_deep_d{depth_value}",
    )


__all__ = [
    "FIXED_DEPTH_BMPD_CRN_METHOD_ID",
    "LAYERWISE_BMPD_CROSS_CRN_METHOD_ID",
    "LAYERWISE_BMPD_NOISELESS_METHOD_ID",
    "LAYERWISE_BMPD_RESAMPLED_METHOD_ID",
    "PHASE1_NOISELESS_CHECKPOINT_CONTROL_METHOD_ID",
    "UNPRUNED_DEEP_BMPD_METHOD_ID",
    "WP20_CONTROL_PROFILE_VERSION",
    "WP20_PRIMARY_BMPD_DEPTH",
    "WP20_PRIMARY_NATIVE_TWO_QUBIT_CAP_PER_EDGE",
    "WP20_PRIMARY_QUBIT_COUNT",
    "FixedDepthBMPDStageRunner",
    "NativeBudgetDescriptor",
    "SecondaryControlDescriptor",
    "build_fixed_depth_bmpd_crn_template",
    "build_independent_fixed_crn_control_template",
    "build_layerwise_bmpd_cross_crn_template",
    "build_layerwise_bmpd_noiseless_template",
    "build_layerwise_bmpd_resampled_template",
    "build_phase1_noiseless_test_control",
    "build_unpruned_deep_control",
]
