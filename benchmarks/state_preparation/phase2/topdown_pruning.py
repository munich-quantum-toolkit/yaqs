# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""WP21 top-down pruning pipeline templates and stage execution.

The pruning mathematics is implemented in :mod:`.pruning`.  This module binds
those pure transforms to authorized Phase II targets, resolved pipeline stages,
fixed-map CRN sampling, and WP18 stage evidence.  In particular, an impact
round samples maps on its *input* circuit and stores the pruned output circuit
as the stage checkpoint binding.
"""

# The strict records below delegate their detailed scalar validation to the
# shared Phase II records. Repeating every propagated exception would obscure
# the scientific contracts.
# ruff: file-ignore[docstring-missing-returns, docstring-missing-exception]

from __future__ import annotations

import hashlib
import itertools
import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from typing_extensions import TypedDict, Unpack

from benchmarks.state_preparation.constants import NOISELESS_NOISE_ID, STANDARD_NOISE_IDS
from benchmarks.state_preparation.noise import (
    FIXED_RATE_NOISE_DEFINITION_VERSION,
    ScaledStandardNoiseProvider,
    create_scaled_standard_noise_provider,
)
from mqt.yaqs.optimization import (
    GateNoiseProvider,
    KrotovFixedMapEnsemble,
    KrotovTJMOptions,
    KrotovTruncation,
    ParameterizedCircuit,
    noisy_state_preparation_metrics,
    sample_krotov_fixed_map_ensemble,
    state_preparation_metrics,
)

from .artifacts import Phase2ArtifactStore, StageExecutionEvidence
from .canonical import canonical_checksum, freeze_json_mapping, thaw_json_mapping, verify_sealed_mapping
from .layerwise_bmpd import (
    LAYERWISE_BMPD_INITIAL_SCALE,
    bmpd_parameter_count,
    bmpd_topology_id,
    create_bmpd_circuit_binding,
    initialize_layerwise_stage_parameters,
)
from .noisy_krotov import (
    NoisyKrotovCircuitBinding,
    NoisyKrotovObjectiveBinding,
    NoisyKrotovStageExecution,
    NoisyKrotovStageFailure,
    execute_fixed_rate_krotov_stage,
)
from .pipeline import (
    CheckpointValidationConfig,
    TrainingPipelineConfig,
    TrainingPipelineTemplate,
    TrainingStageConfig,
    TrainingStageResult,
    TrainingStageTemplate,
)
from .protocol import TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM
from .pruning import (
    TOPDOWN_IMPACT_ITERATIVE_METHOD_ID,
    TOPDOWN_IMPACT_ONE_SHOT_METHOD_ID,
    TOPDOWN_MAGNITUDE_METHOD_ID,
    TOPDOWN_METHOD_IDS,
    TOPDOWN_RANDOM_METHOD_ID,
    FidelityObjective,
    ParameterShiftRequest,
    PruningRoundResult,
    PruningStagePolicy,
    PruningStageSpec,
    PruningUnitKind,
    build_pruning_units,
    run_pruning_round,
)
from .targets import MaterializedTarget
from .validation import (
    require_checksum,
    require_int,
    require_slug,
)
from .wp20_resources import (
    ReachableResourceStratum,
    ResourceBudget,
    ResourceSelectionOutcome,
    WP20WorkLedger,
    measure_circuit_resources,
    select_reachable_resource_stratum,
    wp20_work_from_noisy_krotov,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray


TOPDOWN_PRUNING_METHOD_VERSION = "1"
TOPDOWN_PRUNING_EXECUTION_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp21_pruning_execution.v1"
TOPDOWN_PRUNING_TRACE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp21_pruning_trace.v1"

TOPDOWN_DEFAULT_DEEP_DEPTH = 4
TOPDOWN_DEFAULT_PRETRAIN_ITERATIONS = 100
TOPDOWN_DEFAULT_RELAXATION_ITERATIONS = 50
TOPDOWN_DEFAULT_FINETUNE_ITERATIONS = 200
TOPDOWN_DEFAULT_INITIAL_SCALE = LAYERWISE_BMPD_INITIAL_SCALE
TOPDOWN_DEFAULT_LEARNING_RATE = 0.2
TOPDOWN_DEFAULT_NOISE_ID = "depolarizing_1s_all"

FineTuneMode = Literal["none", "noiseless", "fixed_crn"]
ScoringKind = Literal["none", "noiseless_fidelity", "fixed_map_sample_average_fidelity"]


class TopDownTemplateOptions(TypedDict, total=False):
    """Keyword options shared by the four named WP21 template builders."""

    qubit_count: int
    deep_depth: int
    round_count: int | None
    pruning_unit: PruningUnitKind
    removal_count: int | None
    removal_fraction: float | None
    scoring_objective_kind: ScoringKind | None
    scoring_trajectory_count: int
    pretrain_iterations: int
    relaxation_iterations: int
    fine_tune_mode: FineTuneMode
    fine_tune_iterations: int
    fine_tune_trajectory_count: int
    checkpoint_validation_trajectory_count: int
    resource_stratum_id: str


_METHOD_RULES: Mapping[str, str] = {
    TOPDOWN_RANDOM_METHOD_ID: "random",
    TOPDOWN_MAGNITUDE_METHOD_ID: "magnitude",
    TOPDOWN_IMPACT_ONE_SHOT_METHOD_ID: "impact_one_shot",
    TOPDOWN_IMPACT_ITERATIVE_METHOD_ID: "impact_iterative",
}
_ROOT_TOPOLOGY_PATTERN = re.compile(r"^bmpd_q(?P<qubits>[0-9]+)_d(?P<depth>[0-9]+)$")
_EXECUTION_KEYS = frozenset({
    "schema_version",
    "stage_configuration_checksum",
    "stage_index",
    "stage_id",
    "method_id",
    "round",
    "objective_binding",
    "provider_checksum",
    "training_ensemble_checksums",
    "trace",
    "normalized_work",
    "content_checksum",
})
_NORMALIZED_WORK_KEYS = frozenset({
    "objective_evaluations",
    "gradient_evaluations",
    "training_trajectories",
    "checkpoint_validation_trajectories",
    "test_trajectories",
    "trajectory_gate_applications",
})


def _vector_checksum(vector: NDArray[np.float64]) -> str:
    """Return the canonical Phase II parameter checksum."""
    payload = np.ascontiguousarray(vector, dtype=np.dtype("<f8")).tobytes(order="C")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _seed_domains() -> dict[str, object]:
    """Return the frozen Phase II seed-domain mapping."""
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
    """Return the frozen primary logical-chain materialization policy."""
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
    """Return the seed-free disabled validation template."""
    policy = CheckpointValidationConfig.disabled().to_dict()
    del policy["seed"]
    return policy


def _enabled_checkpoint_policy(trajectory_count: int, cadence: int) -> dict[str, object]:
    """Return a seed-free fixed-CRN checkpoint policy."""
    policy = CheckpointValidationConfig(
        noise_id=TOPDOWN_DEFAULT_NOISE_ID,
        noise_definition_version=FIXED_RATE_NOISE_DEFINITION_VERSION,
        noise_strength_scale=1.0,
        tjm_dt=1.0,
        trajectory_count=trajectory_count,
        seed=1,
        sampling_policy="crn_fixed",
        ensemble_refresh_interval=None,
        cadence=cadence,
        selection_rule="best_validation_fidelity",
        tie_breaker="earliest_iteration",
    ).to_dict()
    del policy["seed"]
    return policy


def _stage_template(
    *,
    index: int,
    stage_id: str,
    stage_kind: Literal["optimize", "prune"],
    input_topology_id: str | None,
    output_topology_id: str,
    input_parameter_count: int,
    output_parameter_count: int,
    transfer_rule: str,
    optimizer_id: str,
    optimizer_hyperparameters: Mapping[str, object],
    iteration_budget: int,
    noisy: bool,
    trajectory_count: int,
    pruning_rule: str,
    pruning_threshold: float | None,
    initialization_binding: str | None,
    optimizer_binding: str | None,
    training_binding: str | None,
    validation_trajectory_count: int = 0,
) -> TrainingStageTemplate:
    """Build one exact WP21 stage template."""
    cadence = min(10, iteration_budget) if validation_trajectory_count else 0
    return TrainingStageTemplate(
        stage_policy={
            "stage_index": index,
            "stage_id": stage_id,
            "stage_kind": stage_kind,
            "input_topology_id": input_topology_id,
            "output_topology_id": output_topology_id,
            "input_parameter_count": input_parameter_count,
            "output_parameter_count": output_parameter_count,
            "parameter_transfer_rule": transfer_rule,
            "optimizer_id": optimizer_id,
            "optimizer_hyperparameters": dict(optimizer_hyperparameters),
            "iteration_budget": iteration_budget,
            "training_noise_id": TOPDOWN_DEFAULT_NOISE_ID if noisy else NOISELESS_NOISE_ID,
            "noise_definition_version": FIXED_RATE_NOISE_DEFINITION_VERSION,
            "noise_strength_scale": 1.0 if noisy else None,
            "tjm_dt": 1.0 if noisy else None,
            "trajectory_count": trajectory_count if noisy else 0,
            "trajectory_update": "independent" if noisy else None,
            "sampling_policy": "crn_fixed" if noisy else "none",
            "crn_refresh_interval": None,
            "checkpoint_validation_policy": (
                _enabled_checkpoint_policy(validation_trajectory_count, cadence)
                if validation_trajectory_count
                else _disabled_checkpoint_policy()
            ),
            "pruning_rule": pruning_rule,
            "pruning_threshold": pruning_threshold,
            "max_bond_dimension": None,
            "svd_threshold": 0.0,
            "truncation_mode": "discarded_weight",
            "min_bond_dimension": 1,
        },
        seed_bindings={
            "initialization": initialization_binding,
            "optimizer": optimizer_binding,
            "training": training_binding,
            "checkpoint_validation": (
                f"topdown_{stage_id}_checkpoint_validation" if validation_trajectory_count else None
            ),
        },
    )


def _validate_positive_int(value: object, name: str) -> int:
    """Return one positive built-in integer."""
    return require_int(value, name, minimum=1)


def _removed_count(policy: PruningStagePolicy, unit_count: int) -> int:
    """Resolve and validate one template's removal cardinality."""
    if policy.removal_schedule == "fixed_count":
        count = cast("int", policy.removal_count)
    else:
        count = math.floor(unit_count * cast("float", policy.removal_fraction))
    if count < 1 or count >= unit_count:
        msg = "Each configured pruning round must remove at least one but not every reachable unit."
        raise ValueError(msg)
    return count


def build_topdown_pruning_template(
    method_id: str,
    *,
    qubit_count: int = 6,
    deep_depth: int = TOPDOWN_DEFAULT_DEEP_DEPTH,
    round_count: int | None = None,
    pruning_unit: PruningUnitKind = "compiled_entangler_group",
    removal_count: int | None = 1,
    removal_fraction: float | None = None,
    scoring_objective_kind: ScoringKind | None = None,
    scoring_trajectory_count: int = 0,
    pretrain_iterations: int = TOPDOWN_DEFAULT_PRETRAIN_ITERATIONS,
    relaxation_iterations: int = TOPDOWN_DEFAULT_RELAXATION_ITERATIONS,
    fine_tune_mode: FineTuneMode = "none",
    fine_tune_iterations: int = TOPDOWN_DEFAULT_FINETUNE_ITERATIONS,
    fine_tune_trajectory_count: int = 0,
    checkpoint_validation_trajectory_count: int = 0,
    resource_stratum_id: str = "topdown_pilot_reachable",
) -> TrainingPipelineTemplate:
    """Build one target-independent WP21 top-down pipeline.

    Randomness and noisy training are stage treatments, not pruning-method
    identities.  The generic builder accepts all four registered method IDs;
    the named wrappers below provide the clearer public entry points.
    """
    method = require_slug(method_id, "method_id")
    if method not in TOPDOWN_METHOD_IDS:
        msg = f"method_id must be one of {sorted(TOPDOWN_METHOD_IDS)!r}."
        raise ValueError(msg)
    qubits = require_int(qubit_count, "qubit_count", minimum=2)
    if qubits not in {6, 12}:
        msg = "WP21 publication templates support the preregistered q6 and q12 scopes."
        raise ValueError(msg)
    depth = _validate_positive_int(deep_depth, "deep_depth")
    rounds = (
        (2 if method == TOPDOWN_IMPACT_ITERATIVE_METHOD_ID else 1)
        if round_count is None
        else _validate_positive_int(round_count, "round_count")
    )
    pretrain = _validate_positive_int(pretrain_iterations, "pretrain_iterations")
    if method == TOPDOWN_IMPACT_ITERATIVE_METHOD_ID and rounds < 2:
        msg = "topdown_impact_iterative requires at least two pruning rounds."
        raise ValueError(msg)
    if method != TOPDOWN_IMPACT_ITERATIVE_METHOD_ID and rounds != 1:
        msg = "Only topdown_impact_iterative may configure more than one pruning round."
        raise ValueError(msg)
    if method == TOPDOWN_IMPACT_ITERATIVE_METHOD_ID:
        _validate_positive_int(relaxation_iterations, "relaxation_iterations")
    if (removal_count is None) == (removal_fraction is None):
        msg = "Specify exactly one of removal_count and removal_fraction."
        raise ValueError(msg)
    schedule: Literal["fixed_count", "fraction_floor"] = (
        "fixed_count" if removal_count is not None else "fraction_floor"
    )
    impact = method in {TOPDOWN_IMPACT_ONE_SHOT_METHOD_ID, TOPDOWN_IMPACT_ITERATIVE_METHOD_ID}
    if scoring_objective_kind is None:
        scoring: ScoringKind = "noiseless_fidelity" if impact else "none"
    else:
        scoring = scoring_objective_kind
    if impact != (scoring != "none"):
        msg = "Impact methods require a fidelity objective; random and magnitude methods require 'none'."
        raise ValueError(msg)
    scoring_trajectories = require_int(scoring_trajectory_count, "scoring_trajectory_count")
    noisy_scoring = scoring == "fixed_map_sample_average_fidelity"
    if noisy_scoring != (scoring_trajectories > 0):
        msg = "Fixed-map scoring requires a positive trajectory count, and other scoring kinds require zero."
        raise ValueError(msg)
    if fine_tune_mode not in {"none", "noiseless", "fixed_crn"}:
        msg = "fine_tune_mode must be 'none', 'noiseless', or 'fixed_crn'."
        raise ValueError(msg)
    if fine_tune_mode == "none":
        if fine_tune_trajectory_count or checkpoint_validation_trajectory_count:
            msg = "An omitted fine-tune cannot configure training or checkpoint trajectories."
            raise ValueError(msg)
    else:
        _validate_positive_int(fine_tune_iterations, "fine_tune_iterations")
        if fine_tune_mode == "fixed_crn":
            _validate_positive_int(fine_tune_trajectory_count, "fine_tune_trajectory_count")
        elif fine_tune_trajectory_count:
            msg = "Noiseless fine-tuning requires zero training trajectories."
            raise ValueError(msg)
        if checkpoint_validation_trajectory_count:
            _validate_positive_int(
                checkpoint_validation_trajectory_count,
                "checkpoint_validation_trajectory_count",
            )

    root_binding = create_bmpd_circuit_binding(qubits, depth)
    root_circuit = root_binding.circuit
    units = build_pruning_units(root_circuit, pruning_unit)
    if any(len(unit.parameter_indices) != 1 for unit in units):
        msg = "The standard BMPD template requires one retained parameter per pruning unit."
        raise ValueError(msg)

    stages: list[TrainingStageTemplate] = []
    root_count = bmpd_parameter_count(qubits, depth)
    stages.append(
        _stage_template(
            index=0,
            stage_id="deep_pretrain",
            stage_kind="optimize",
            input_topology_id=None,
            output_topology_id=bmpd_topology_id(qubits, depth),
            input_parameter_count=0,
            output_parameter_count=root_count,
            transfer_rule="initialize_random_normal",
            optimizer_id="krotov",
            optimizer_hyperparameters={
                "learning_rate": 1.0,
                "schedule": "constant",
                "decay": 0.0,
                "initialization_rng": "numpy_pcg64_standard_normal_v1",
                "initialization_scale": float(TOPDOWN_DEFAULT_INITIAL_SCALE),
            },
            iteration_budget=pretrain,
            noisy=False,
            trajectory_count=0,
            pruning_rule="none",
            pruning_threshold=None,
            initialization_binding="topdown_deep_initialization",
            optimizer_binding="topdown_deep_pretrain_optimizer",
            training_binding=None,
        )
    )

    current_topology = root_binding.topology_id
    current_parameter_count = root_count
    current_unit_count = len(units)
    for round_index in range(rounds):
        relax_after = method == TOPDOWN_IMPACT_ITERATIVE_METHOD_ID and round_index < rounds - 1
        policy = PruningStagePolicy(
            pruning_unit=pruning_unit,
            scoring_objective_kind=scoring,
            removal_schedule=schedule,
            removal_count=removal_count,
            removal_fraction=removal_fraction,
            relax_after_round=relax_after,
        )
        removed = _removed_count(policy, current_unit_count)
        next_count = current_parameter_count - removed
        next_unit_count = current_unit_count - removed
        output_topology = f"topdown_q{qubits}_d{depth}_r{round_index + 1}_p{next_count}"
        stage_index = len(stages)
        threshold = float(cast("int", removal_count)) if schedule == "fixed_count" else removal_fraction
        stages.append(
            _stage_template(
                index=stage_index,
                stage_id=f"prune_round_{round_index + 1}",
                stage_kind="prune",
                input_topology_id=current_topology,
                output_topology_id=output_topology,
                input_parameter_count=current_parameter_count,
                output_parameter_count=next_count,
                transfer_rule="apply_pruning_mask",
                optimizer_id="none",
                optimizer_hyperparameters=policy.to_mapping(),
                iteration_budget=0,
                noisy=noisy_scoring,
                trajectory_count=scoring_trajectories,
                pruning_rule=_METHOD_RULES[method],
                pruning_threshold=cast("float", threshold),
                initialization_binding=None,
                optimizer_binding=(
                    f"topdown_random_round_{round_index + 1}" if method == TOPDOWN_RANDOM_METHOD_ID else None
                ),
                training_binding=(f"topdown_score_round_{round_index + 1}" if noisy_scoring else None),
            )
        )
        current_topology = output_topology
        current_parameter_count = next_count
        current_unit_count = next_unit_count
        if relax_after:
            stage_index = len(stages)
            stages.append(
                _stage_template(
                    index=stage_index,
                    stage_id=f"relax_round_{round_index + 1}",
                    stage_kind="optimize",
                    input_topology_id=current_topology,
                    output_topology_id=current_topology,
                    input_parameter_count=current_parameter_count,
                    output_parameter_count=current_parameter_count,
                    transfer_rule="copy",
                    optimizer_id="krotov",
                    optimizer_hyperparameters={
                        "learning_rate": TOPDOWN_DEFAULT_LEARNING_RATE,
                        "schedule": "constant",
                        "decay": 0.0,
                    },
                    iteration_budget=relaxation_iterations,
                    noisy=False,
                    trajectory_count=0,
                    pruning_rule="none",
                    pruning_threshold=None,
                    initialization_binding=None,
                    optimizer_binding=f"topdown_relax_round_{round_index + 1}_optimizer",
                    training_binding=None,
                )
            )

    if fine_tune_mode != "none":
        noisy_finetune = fine_tune_mode == "fixed_crn"
        stages.append(
            _stage_template(
                index=len(stages),
                stage_id="final_finetune",
                stage_kind="optimize",
                input_topology_id=current_topology,
                output_topology_id=current_topology,
                input_parameter_count=current_parameter_count,
                output_parameter_count=current_parameter_count,
                transfer_rule="copy",
                optimizer_id="krotov",
                optimizer_hyperparameters={
                    "learning_rate": TOPDOWN_DEFAULT_LEARNING_RATE,
                    "schedule": "exp",
                    "decay": 0.01,
                },
                iteration_budget=fine_tune_iterations,
                noisy=noisy_finetune,
                trajectory_count=fine_tune_trajectory_count,
                pruning_rule="none",
                pruning_threshold=None,
                initialization_binding=None,
                optimizer_binding="topdown_final_finetune_optimizer",
                training_binding=("topdown_final_finetune_training" if noisy_finetune else None),
                validation_trajectory_count=checkpoint_validation_trajectory_count,
            )
        )

    return TrainingPipelineTemplate(
        template_id=method,
        preregistration_checksum=TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM,
        target_scope_id="primary_q6" if qubits == 6 else "secondary_q12",
        ansatz_family="bmpd_brickwall",
        method_id=method,
        method_version=TOPDOWN_PRUNING_METHOD_VERSION,
        resource_stratum_id=require_slug(resource_stratum_id, "resource_stratum_id"),
        stages=tuple(stages),
        seed_domains=_seed_domains(),
        final_materialization_policy=_materialization_policy(),
    )


def build_topdown_random_template(**kwargs: Unpack[TopDownTemplateOptions]) -> TrainingPipelineTemplate:
    """Build the deterministic-seed random-pruning competitor."""
    return build_topdown_pruning_template(TOPDOWN_RANDOM_METHOD_ID, **kwargs)


def build_topdown_magnitude_template(**kwargs: Unpack[TopDownTemplateOptions]) -> TrainingPipelineTemplate:
    """Build the one-shot magnitude-pruning competitor."""
    return build_topdown_pruning_template(TOPDOWN_MAGNITUDE_METHOD_ID, **kwargs)


def build_topdown_impact_one_shot_template(**kwargs: Unpack[TopDownTemplateOptions]) -> TrainingPipelineTemplate:
    """Build the one-shot gradient-impact pruning competitor."""
    return build_topdown_pruning_template(TOPDOWN_IMPACT_ONE_SHOT_METHOD_ID, **kwargs)


def build_topdown_impact_iterative_template(**kwargs: Unpack[TopDownTemplateOptions]) -> TrainingPipelineTemplate:
    """Build the alternating impact-prune/relax competitor."""
    return build_topdown_pruning_template(TOPDOWN_IMPACT_ITERATIVE_METHOD_ID, **kwargs)


def _normalized_work(work: WP20WorkLedger) -> dict[str, int]:
    """Project detailed WP20 work onto the persisted Phase II ledger."""
    return {
        "objective_evaluations": work.objective_calls,
        "gradient_evaluations": work.gradient_calls,
        "training_trajectories": work.training_trajectories,
        "checkpoint_validation_trajectories": work.checkpoint_validation_trajectories,
        "test_trajectories": work.test_trajectories,
        "trajectory_gate_applications": work.trajectory_gate_applications,
    }


def _circuit_statistics(binding: NoisyKrotovCircuitBinding, *, round_index: int | None) -> dict[str, object]:
    """Return compiler-derived logical and native circuit statistics."""
    circuit = binding.circuit
    resources = measure_circuit_resources(circuit)
    return {
        "topology_id": binding.topology_id,
        "parameter_count": circuit.num_params,
        "qubit_count": circuit.num_qubits,
        "pruning_round_index": round_index,
        "logical_gate_count": len(circuit.gates),
        "logical_two_qubit_gate_count": resources.logical_two_qubit_gates,
        "native_two_qubit_gate_count": resources.native_two_qubit_gates,
        "native_two_qubit_gates_per_chain_edge": list(resources.native_two_qubit_gates_per_chain_edge),
        "circuit_resource_metrics": resources.to_dict(),
    }


def _pruning_trace(stage: TrainingStageConfig, result: PruningRoundResult) -> tuple[dict[str, object], ...]:
    """Return the one exact parameter-transform event for a pruning stage."""
    return (
        {
            "schema_version": TOPDOWN_PRUNING_TRACE_SCHEMA_VERSION,
            "event": "topdown_pruning_round",
            "stage_index": stage.stage_index,
            "stage_id": stage.stage_id,
            "stage_configuration_checksum": stage.configuration_checksum,
            "round_index": result.round_index,
            "round_checksum": result.content_checksum,
            "input_circuit_binding_checksum": result.input_circuit_binding.content_checksum,
            "output_circuit_binding_checksum": result.output_circuit_binding.content_checksum,
            "input_parameter_checksum": result.input_parameter_checksum,
            "output_parameter_checksum": result.output_parameter_checksum,
            "score_order": [score.unit_id for score in result.scores],
            "removed_unit_ids": list(result.removed_unit_ids),
            "retained_unit_ids": list(result.retained_unit_ids),
            "input_native_two_qubit_gate_count": result.input_resources.native_two_qubit_gates,
            "output_native_two_qubit_gate_count": result.output_resources.native_two_qubit_gates,
            "normalized_work": _normalized_work(result.work),
        },
    )


@dataclass(frozen=True, slots=True)
class TopDownPruningStageExecution:
    """Checksum-sealed stage context around one mechanically verified round."""

    stage_configuration_checksum: str
    stage_index: int
    stage_id: str
    method_id: str
    round: PruningRoundResult
    objective_binding: NoisyKrotovObjectiveBinding | None
    provider_checksum: str | None
    training_ensemble_checksums: tuple[str, ...]
    trace: tuple[Mapping[str, object], ...]
    normalized_work: Mapping[str, object]
    schema_version: str = field(default=TOPDOWN_PRUNING_EXECUTION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate all context aliases and immutable evidence."""
        object.__setattr__(
            self,
            "stage_configuration_checksum",
            require_checksum(self.stage_configuration_checksum, "stage_configuration_checksum"),
        )
        stage_index = require_int(self.stage_index, "stage_index")
        stage_id = require_slug(self.stage_id, "stage_id")
        if not isinstance(self.round, PruningRoundResult):
            msg = "round must be a PruningRoundResult."
            raise TypeError(msg)
        method = require_slug(self.method_id, "method_id")
        if method not in TOPDOWN_METHOD_IDS or method != self.round.method_id:
            msg = "Execution method_id must match its verified pruning round."
            raise ValueError(msg)
        object.__setattr__(self, "method_id", method)
        impact = method in {TOPDOWN_IMPACT_ONE_SHOT_METHOD_ID, TOPDOWN_IMPACT_ITERATIVE_METHOD_ID}
        if impact != isinstance(self.objective_binding, NoisyKrotovObjectiveBinding):
            msg = "A sealed objective binding is present exactly for impact pruning."
            raise ValueError(msg)
        provider = (
            None if self.provider_checksum is None else require_checksum(self.provider_checksum, "provider_checksum")
        )
        checksums = tuple(
            require_checksum(value, f"training_ensemble_checksums[{index}]")
            for index, value in enumerate(self.training_ensemble_checksums)
        )
        noisy = self.round.spec.scoring_objective_kind == "fixed_map_sample_average_fidelity"
        if noisy and (provider is None or len(checksums) != 1):
            msg = "Fixed-map impact scoring requires exactly one provider-bound ensemble."
            raise ValueError(msg)
        if not noisy and (provider is not None or checksums):
            msg = "Noiseless, random, and magnitude pruning cannot claim fixed-map evidence."
            raise ValueError(msg)
        expected_trace = _pruning_trace_from_execution(
            self.stage_configuration_checksum,
            stage_index,
            stage_id,
            self.round,
        )
        frozen_trace = tuple(freeze_json_mapping(row, f"trace[{index}]") for index, row in enumerate(self.trace))
        if tuple(thaw_json_mapping(row) for row in frozen_trace) != expected_trace:
            msg = "Pruning trace is not exactly implied by the sealed round."
            raise ValueError(msg)
        work = freeze_json_mapping(self.normalized_work, "normalized_work")
        if set(work) != _NORMALIZED_WORK_KEYS or thaw_json_mapping(work) != _normalized_work(self.round.work):
            msg = "normalized_work is not the exact WP20 projection of the round."
            raise ValueError(msg)
        object.__setattr__(self, "provider_checksum", provider)
        object.__setattr__(self, "stage_index", stage_index)
        object.__setattr__(self, "stage_id", stage_id)
        object.__setattr__(self, "training_ensemble_checksums", checksums)
        object.__setattr__(self, "trace", frozen_trace)
        object.__setattr__(self, "normalized_work", work)

    def _content_dict(self) -> dict[str, object]:
        """Return every execution field except its outer checksum."""
        return {
            "schema_version": self.schema_version,
            "stage_configuration_checksum": self.stage_configuration_checksum,
            "stage_index": self.stage_index,
            "stage_id": self.stage_id,
            "method_id": self.method_id,
            "round": self.round.to_dict(),
            "objective_binding": None if self.objective_binding is None else self.objective_binding.to_dict(),
            "provider_checksum": self.provider_checksum,
            "training_ensemble_checksums": list(self.training_ensemble_checksums),
            "trace": [thaw_json_mapping(row) for row in self.trace],
            "normalized_work": thaw_json_mapping(self.normalized_work),
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of complete pruning-stage execution evidence."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed execution data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> TopDownPruningStageExecution:
        """Decode and mechanically verify a pruning execution document."""
        mapping = verify_sealed_mapping(data, expected_keys=_EXECUTION_KEYS, name="WP21 pruning execution")
        if mapping["schema_version"] != TOPDOWN_PRUNING_EXECUTION_SCHEMA_VERSION:
            msg = "WP21 pruning execution uses an unsupported schema version."
            raise ValueError(msg)
        raw_objective = mapping["objective_binding"]
        execution = cls(
            stage_configuration_checksum=cast("str", mapping["stage_configuration_checksum"]),
            stage_index=cast("int", mapping["stage_index"]),
            stage_id=cast("str", mapping["stage_id"]),
            method_id=cast("str", mapping["method_id"]),
            round=PruningRoundResult.from_dict(mapping["round"]),
            objective_binding=(None if raw_objective is None else NoisyKrotovObjectiveBinding.from_dict(raw_objective)),
            provider_checksum=cast("str | None", mapping["provider_checksum"]),
            training_ensemble_checksums=cast("tuple[str, ...]", mapping["training_ensemble_checksums"]),
            trace=cast("tuple[Mapping[str, object], ...]", mapping["trace"]),
            normalized_work=cast("Mapping[str, object]", mapping["normalized_work"]),
        )
        if mapping["content_checksum"] != execution.content_checksum:
            msg = "WP21 pruning execution checksum changed during normalization."
            raise ValueError(msg)
        return execution


def _validated_pruning_path_rounds(
    values: Sequence[PruningRoundResult],
) -> tuple[PruningRoundResult, ...]:
    """Return one contiguous, single-method attempted pruning path."""
    rounds = tuple(values)
    if not rounds or not all(isinstance(item, PruningRoundResult) for item in rounds):
        msg = "rounds must contain at least one PruningRoundResult."
        raise TypeError(msg)
    if tuple(item.round_index for item in rounds) != tuple(range(len(rounds))):
        msg = "Pruning path round indices must be contiguous from zero."
        raise ValueError(msg)
    if len({item.method_id for item in rounds}) != 1:
        msg = "Every pruning path round must use the same registered method."
        raise ValueError(msg)
    method_id = rounds[0].method_id
    if len(rounds) > 1 and method_id != TOPDOWN_IMPACT_ITERATIVE_METHOD_ID:
        msg = "Only iterative impact pruning may contain more than one pruning round."
        raise ValueError(msg)
    relaxation_flags = tuple(item.spec.relax_after_round for item in rounds)
    invalid_relaxation = (
        any(relaxation_flags)
        if method_id != TOPDOWN_IMPACT_ITERATIVE_METHOD_ID
        else any(not value for value in relaxation_flags[:-1])
    )
    if invalid_relaxation:
        msg = "Pruning-path relaxation declarations do not match iterative method semantics."
        raise ValueError(msg)
    if any(
        next_.input_circuit_binding.to_dict() != previous.output_circuit_binding.to_dict()
        for previous, next_ in itertools.pairwise(rounds)
    ):
        msg = "Pruning path circuit bindings are not contiguous."
        raise ValueError(msg)
    return rounds


@dataclass(frozen=True, slots=True)
class TopDownPruningPathExecution:
    """Ordered observed pruning rounds and their reachable resource path."""

    rounds: tuple[PruningRoundResult, ...]
    root_prefix_work: WP20WorkLedger
    post_round_work: tuple[WP20WorkLedger, ...]

    def __post_init__(self) -> None:
        """Require one contiguous, single-method attempted pruning path."""
        rounds = _validated_pruning_path_rounds(self.rounds)
        if not isinstance(self.root_prefix_work, WP20WorkLedger):
            msg = "root_prefix_work must be an exact WP20WorkLedger."
            raise TypeError(msg)
        post = tuple(self.post_round_work)
        if len(post) != len(rounds) or not all(isinstance(item, WP20WorkLedger) for item in post):
            msg = "post_round_work must provide one exact WP20 ledger per attempted round."
            raise TypeError(msg)
        object.__setattr__(self, "rounds", rounds)
        object.__setattr__(self, "post_round_work", post)

    @property
    def reachable_strata(self) -> tuple[ReachableResourceStratum, ...]:
        """The deep root and every actually retained compiled round."""
        return topdown_reachable_resource_strata(
            self.rounds,
            root_prefix_work=self.root_prefix_work,
            post_round_work=self.post_round_work,
        )

    def select(self, budget: ResourceBudget) -> ResourceSelectionOutcome:
        """Select across every attempted round without assuming monotonic routing."""
        return select_reachable_resource_stratum(self.reachable_strata, budget)


def _pruning_trace_from_execution(
    stage_configuration_checksum: str,
    stage_index: int,
    stage_id: str,
    result: PruningRoundResult,
) -> tuple[dict[str, object], ...]:
    """Rebuild trace data without needing the original stage object."""
    return (
        {
            "schema_version": TOPDOWN_PRUNING_TRACE_SCHEMA_VERSION,
            "event": "topdown_pruning_round",
            "stage_index": stage_index,
            "stage_id": stage_id,
            "stage_configuration_checksum": stage_configuration_checksum,
            "round_index": result.round_index,
            "round_checksum": result.content_checksum,
            "input_circuit_binding_checksum": result.input_circuit_binding.content_checksum,
            "output_circuit_binding_checksum": result.output_circuit_binding.content_checksum,
            "input_parameter_checksum": result.input_parameter_checksum,
            "output_parameter_checksum": result.output_parameter_checksum,
            "score_order": [score.unit_id for score in result.scores],
            "removed_unit_ids": list(result.removed_unit_ids),
            "retained_unit_ids": list(result.retained_unit_ids),
            "input_native_two_qubit_gate_count": result.input_resources.native_two_qubit_gates,
            "output_native_two_qubit_gate_count": result.output_resources.native_two_qubit_gates,
            "normalized_work": _normalized_work(result.work),
        },
    )


def _execution_for_round(
    stage: TrainingStageConfig,
    result: PruningRoundResult,
    objective_binding: NoisyKrotovObjectiveBinding | None,
    provider_checksum: str | None,
    ensembles: Sequence[KrotovFixedMapEnsemble],
) -> TopDownPruningStageExecution:
    """Construct execution evidence with the exact resolved stage trace."""
    return TopDownPruningStageExecution(
        stage_configuration_checksum=stage.configuration_checksum,
        stage_index=stage.stage_index,
        stage_id=stage.stage_id,
        method_id=result.method_id,
        round=result,
        objective_binding=objective_binding,
        provider_checksum=provider_checksum,
        training_ensemble_checksums=tuple(item.content_checksum for item in ensembles),
        trace=_pruning_trace(stage, result),
        normalized_work=_normalized_work(result.work),
    )


def _training_summary(execution: TopDownPruningStageExecution) -> dict[str, object]:
    """Return the exact artifact summary implied by an execution."""
    result = execution.round
    impact = result.method_id in {TOPDOWN_IMPACT_ONE_SHOT_METHOD_ID, TOPDOWN_IMPACT_ITERATIVE_METHOD_ID}
    return {
        "pruning_execution_checksum": execution.content_checksum,
        "pruning_execution_document": execution.to_dict(),
        "method_id": result.method_id,
        "pruning_rule": result.spec.score_rule,
        "removed_unit_count": len(result.removed_unit_ids),
        "retained_unit_count": len(result.retained_unit_ids),
        "output_circuit_binding_checksum": result.output_circuit_binding.content_checksum,
        "map_circuit_binding_checksum": (
            result.input_circuit_binding.content_checksum if impact else result.output_circuit_binding.content_checksum
        ),
    }


def topdown_pruning_stage_evidence(
    stage: TrainingStageConfig,
    execution: TopDownPruningStageExecution,
    ensembles: Sequence[KrotovFixedMapEnsemble] = (),
) -> StageExecutionEvidence:
    """Adapt one WP21 pruning execution to the optimizer-neutral WP18 record."""
    result = execution.round
    maps = tuple(ensembles)
    if tuple(item.content_checksum for item in maps) != execution.training_ensemble_checksums:
        msg = "Supplied fixed maps do not match the pruning execution document."
        raise ValueError(msg)
    impact = result.method_id in {TOPDOWN_IMPACT_ONE_SHOT_METHOD_ID, TOPDOWN_IMPACT_ITERATIVE_METHOD_ID}
    map_binding = (
        result.input_circuit_binding.content_checksum if impact else result.output_circuit_binding.content_checksum
    )
    return StageExecutionEvidence(
        stage=stage,
        source_parameters=result.input_theta,
        initial_parameters=result.output_theta,
        final_parameters=result.output_theta,
        selected_parameters=result.output_theta,
        selected_global_iteration=0,
        completed_global_iteration=0,
        selected_checkpoint_validation_fidelity=None,
        circuit_binding_checksum=result.output_circuit_binding.content_checksum,
        map_circuit_binding_checksum=map_binding,
        provider_checksum=execution.provider_checksum,
        objective_checksum=(
            None if execution.objective_binding is None else execution.objective_binding.objective_checksum
        ),
        objective_binding=execution.objective_binding,
        trace=execution.trace,
        training_ensembles=maps,
        checkpoint_validation_ensembles=(),
        normalized_work=execution.normalized_work,
        training_summary=_training_summary(execution),
        checkpoint_validation_summary=None,
        circuit_topology=result.output_circuit_binding.to_dict(),
        circuit_statistics=_circuit_statistics(result.output_circuit_binding, round_index=result.round_index),
        optimizer_state=None,
        cumulative_cross_trajectory_pairings=0,
    )


def _provider_for_stage(stage: TrainingStageConfig) -> ScaledStandardNoiseProvider:
    """Construct the exact standard fixed-rate provider for a noisy score."""
    if stage.training_noise_id not in STANDARD_NOISE_IDS:
        msg = "WP21 fixed-map scoring currently supports the standard fixed-rate noise registry."
        raise ValueError(msg)
    if stage.noise_strength_scale is None:
        msg = "A noisy pruning score requires a resolved strength scale."
        raise ValueError(msg)
    return create_scaled_standard_noise_provider(stage.training_noise_id, stage.noise_strength_scale)


def _score_inputs(
    stage: TrainingStageConfig,
    binding: NoisyKrotovCircuitBinding,
    theta: NDArray[np.float64],
    target: MaterializedTarget,
) -> tuple[
    FidelityObjective | None,
    NoisyKrotovObjectiveBinding | None,
    str | None,
    tuple[KrotovFixedMapEnsemble, ...],
    int,
    WP20WorkLedger | None,
]:
    """Create the round objective and, when requested, one frozen CRN map."""
    spec = PruningStageSpec.from_mapping(
        stage.optimizer_hyperparameters,
        method_id={rule: method for method, rule in _METHOD_RULES.items()}[stage.pruning_rule],
        score_rule=stage.pruning_rule,
        random_seed=stage.optimizer_seed,
    )
    if spec.scoring_objective_kind == "none":
        return None, None, None, (), 0, None
    target_vector = target.state_vector_copy()
    objective_binding = NoisyKrotovObjectiveBinding.from_inputs(
        target,
        None,
        num_qubits=binding.circuit.num_qubits,
    )
    truncation = KrotovTruncation(
        max_bond_dim=stage.max_bond_dimension,
        svd_threshold=stage.svd_threshold,
        trunc_mode=stage.truncation_mode,
        min_bond_dim=stage.min_bond_dimension,
    )
    if spec.scoring_objective_kind == "noiseless_fidelity":

        def objective(
            circuit: ParameterizedCircuit,
            parameters: NDArray[np.float64],
            request: ParameterShiftRequest,
        ) -> float:
            del request
            _loss, fidelity = state_preparation_metrics(
                circuit,
                parameters,
                target_vector,
                truncation=truncation,
            )
            return float(np.clip(fidelity, 0.0, 1.0))

        return objective, objective_binding, None, (), 0, None

    provider = _provider_for_stage(stage)
    provider_checksum = provider.content_checksum
    if stage.training_seed is None or stage.tjm_dt is None:
        msg = "Fixed-map pruning requires a resolved training seed and TJM dt."
        raise ValueError(msg)
    options = KrotovTJMOptions(
        num_trajectories=stage.trajectory_count,
        random_seed=stage.training_seed,
        dt=stage.tjm_dt,
        apply_noise_to="all",
        noisy_gate_indices=binding.noisy_gate_indices,
        trajectory_update="independent",
        differentiate_jump_normalization=False,
        use_crn=False,
    )
    circuit = binding.circuit
    ensemble = sample_krotov_fixed_map_ensemble(
        circuit,
        theta,
        None,
        truncation,
        cast("GateNoiseProvider", provider),
        options,
        role="training_trajectory",
        resolved_seed=stage.training_seed,
        stage_index=stage.stage_index,
        stage_id=stage.stage_id,
        stage_configuration_checksum=stage.configuration_checksum,
        circuit_checksum=binding.content_checksum,
        provider_checksum=provider_checksum,
        ensemble_index=0,
        refresh_index=0,
        global_iteration_start=0,
    )

    def noisy_objective(
        circuit: ParameterizedCircuit,
        parameters: NDArray[np.float64],
        request: ParameterShiftRequest,
    ) -> float:
        del request
        _loss, fidelity, _values = noisy_state_preparation_metrics(
            circuit,
            parameters,
            target_vector,
            None,
            options,
            truncation=truncation,
            fixed_noise_maps=ensemble.replay_maps(),
            noise_provider=cast("GateNoiseProvider", provider),
        )
        return float(np.clip(fidelity, 0.0, 1.0))

    sampling_work = WP20WorkLedger(
        forward_circuit_evaluations=stage.trajectory_count,
        trajectory_gate_applications=stage.trajectory_count * len(circuit.gates),
        training_trajectories=stage.trajectory_count,
    )
    return (
        noisy_objective,
        objective_binding,
        provider_checksum,
        (ensemble,),
        stage.trajectory_count,
        sampling_work,
    )


@dataclass(slots=True)
class TopDownPruningStageRunner:
    """Execute a resolved WP21 pipeline, including artifact-prefix resume."""

    pipeline: TrainingPipelineConfig
    target: MaterializedTarget
    artifact_store: Phase2ArtifactStore | None = None
    _bindings: dict[str, NoisyKrotovCircuitBinding] = field(default_factory=dict, init=False, repr=False)
    _rounds: dict[int, PruningRoundResult] = field(default_factory=dict, init=False, repr=False)
    _root_prefix_work: WP20WorkLedger = field(default_factory=WP20WorkLedger, init=False, repr=False)
    _post_round_work: dict[int, WP20WorkLedger] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Verify pipeline/target agreement and reconstruct completed bindings."""
        if not isinstance(self.pipeline, TrainingPipelineConfig):
            msg = "pipeline must be a TrainingPipelineConfig."
            raise TypeError(msg)
        if self.pipeline.method_id not in TOPDOWN_METHOD_IDS:
            msg = "TopDownPruningStageRunner accepts only the four WP21 method identities."
            raise ValueError(msg)
        if not isinstance(self.target, MaterializedTarget):
            msg = "WP21 execution requires an authorized MaterializedTarget."
            raise TypeError(msg)
        identity = self.target.identity_dict()
        expected = {
            "target_instance_id": self.pipeline.target_instance_id,
            "target_instance_spec_checksum": self.pipeline.target_instance_spec_checksum,
            "target_manifest_checksum": self.pipeline.target_population_manifest_checksum,
            "family_id": self.pipeline.target_family_id,
            "stratum_id": self.pipeline.target_stratum_id,
            "qubit_count": self.pipeline.qubit_count,
        }
        if any(identity[name] != value for name, value in expected.items()):
            msg = "Materialized target identity does not match the resolved WP21 pipeline."
            raise ValueError(msg)
        first_topology = self.pipeline.stages[0].output_topology_id
        match = _ROOT_TOPOLOGY_PATTERN.fullmatch(first_topology)
        if match is None or int(match.group("qubits")) != self.pipeline.qubit_count:
            msg = "WP21 first stage must identify its deep BMPD root topology."
            raise ValueError(msg)
        root = create_bmpd_circuit_binding(self.pipeline.qubit_count, int(match.group("depth")))
        if root.content_checksum != canonical_checksum({
            key: value for key, value in root.to_dict().items() if key != "content_checksum"
        }):
            msg = "Deep BMPD root binding failed its own checksum closure."
            raise ValueError(msg)
        self._bindings[root.topology_id] = root
        if self.artifact_store is not None:
            if not isinstance(self.artifact_store, Phase2ArtifactStore):
                msg = "artifact_store must be a Phase2ArtifactStore or None."
                raise TypeError(msg)
            if self.artifact_store.pipeline.configuration_checksum != self.pipeline.configuration_checksum:
                msg = "artifact_store belongs to a different resolved pipeline."
                raise ValueError(msg)
            for artifact in self.artifact_store.stage_artifacts:
                stage_result = artifact.stage_result
                completed_stage = self.pipeline.stages[stage_result.stage_index]
                document = stage_result.training_summary.get("pruning_execution_document")
                if document is not None:
                    restored = TopDownPruningStageExecution.from_dict(document)
                    expected_round_index = sum(
                        candidate.stage_kind == "prune"
                        for candidate in self.pipeline.stages[: completed_stage.stage_index]
                    )
                    _validate_execution_stage(
                        completed_stage,
                        restored,
                        expected_round_index=expected_round_index,
                    )
                    self._bindings[restored.round.output_circuit_binding.topology_id] = (
                        restored.round.output_circuit_binding
                    )
                    self._rounds[restored.round.round_index] = restored.round
                    self._post_round_work[restored.round.round_index] = _plus_work(
                        self._post_round_work.get(restored.round.round_index, WP20WorkLedger()),
                        WP20WorkLedger(
                            wall_time_seconds=stage_result.wall_time_seconds,
                            peak_memory_bytes=stage_result.peak_memory_bytes,
                        ),
                    )
                elif completed_stage.optimizer_id == "krotov":
                    work = _wp20_work_from_stage_result(completed_stage, stage_result)
                    if not self._rounds:
                        self._root_prefix_work = _plus_work(self._root_prefix_work, work)
                    else:
                        round_index = max(self._rounds)
                        self._post_round_work[round_index] = _plus_work(
                            self._post_round_work.get(round_index, WP20WorkLedger()),
                            work,
                        )

    @property
    def pruning_path(self) -> TopDownPruningPathExecution | None:
        """The completed pruning prefix with authoritative measured work.

        When a store is attached, only published stages form the durable path
        and their persisted executor timing is authoritative.  A store-free
        runner exposes its completed in-memory path, whose runtime fields are
        necessarily zero because no executor boundary has measured them.
        """
        if self.artifact_store is not None:
            return self._persisted_pruning_path()
        if not self._rounds:
            return None
        indices = tuple(sorted(self._rounds))
        return TopDownPruningPathExecution(
            rounds=tuple(self._rounds[index] for index in indices),
            root_prefix_work=self._root_prefix_work,
            post_round_work=tuple(self._post_round_work.get(index, WP20WorkLedger()) for index in indices),
        )

    def _persisted_pruning_path(self) -> TopDownPruningPathExecution | None:
        """Reconstruct one exact path from the current committed artifact prefix."""
        if self.artifact_store is None:
            msg = "A persisted pruning path requires an attached artifact store."
            raise RuntimeError(msg)
        rounds: dict[int, PruningRoundResult] = {}
        root_prefix_work = WP20WorkLedger()
        post_round_work: dict[int, WP20WorkLedger] = {}
        for artifact in self.artifact_store.stage_artifacts:
            stage_result = artifact.stage_result
            completed_stage = self.pipeline.stages[stage_result.stage_index]
            document = stage_result.training_summary.get("pruning_execution_document")
            if document is not None:
                execution = TopDownPruningStageExecution.from_dict(document)
                expected_round_index = sum(
                    candidate.stage_kind == "prune" for candidate in self.pipeline.stages[: completed_stage.stage_index]
                )
                _validate_execution_stage(
                    completed_stage,
                    execution,
                    expected_round_index=expected_round_index,
                )
                rounds[execution.round.round_index] = execution.round
                post_round_work[execution.round.round_index] = _plus_work(
                    post_round_work.get(execution.round.round_index, WP20WorkLedger()),
                    WP20WorkLedger(
                        wall_time_seconds=stage_result.wall_time_seconds,
                        peak_memory_bytes=stage_result.peak_memory_bytes,
                    ),
                )
            elif completed_stage.optimizer_id == "krotov":
                work = _wp20_work_from_stage_result(completed_stage, stage_result)
                if not rounds:
                    root_prefix_work = _plus_work(root_prefix_work, work)
                else:
                    round_index = max(rounds)
                    post_round_work[round_index] = _plus_work(
                        post_round_work.get(round_index, WP20WorkLedger()),
                        work,
                    )
        if not rounds:
            return None
        indices = tuple(sorted(rounds))
        return TopDownPruningPathExecution(
            rounds=tuple(rounds[index] for index in indices),
            root_prefix_work=root_prefix_work,
            post_round_work=tuple(post_round_work.get(index, WP20WorkLedger()) for index in indices),
        )

    def _binding_for_stage(self, stage: TrainingStageConfig) -> NoisyKrotovCircuitBinding:
        """Return the exact current circuit for an optimize or prune stage."""
        topology = stage.input_topology_id if stage.stage_kind == "prune" else stage.output_topology_id
        if topology is None:
            topology = stage.output_topology_id
        binding = self._bindings.get(topology)
        if binding is None:
            msg = f"No verified circuit binding is available for topology {topology!r}."
            raise ValueError(msg)
        expected_count = stage.input_parameter_count if stage.stage_kind == "prune" else stage.output_parameter_count
        if binding.circuit.num_params != expected_count:
            msg = "Reconstructed WP21 circuit parameter count differs from the configured stage."
            raise ValueError(msg)
        return binding

    def __call__(
        self,
        stage: TrainingStageConfig,
        predecessor_parameters: NDArray[np.float64] | None,
    ) -> StageExecutionEvidence | NoisyKrotovStageExecution | NoisyKrotovStageFailure:
        """Execute exactly one configured top-down stage without filesystem mutation."""
        if stage.stage_index >= len(self.pipeline.stages) or stage != self.pipeline.stages[stage.stage_index]:
            msg = "Stage does not belong to this resolved WP21 pipeline."
            raise ValueError(msg)
        if stage.stage_kind != "prune":
            binding = self._binding_for_stage(stage)
            initial = initialize_layerwise_stage_parameters(stage, predecessor_parameters)
            outcome = execute_fixed_rate_krotov_stage(stage, binding, self.target, initial)
            if isinstance(outcome, NoisyKrotovStageExecution):
                work = wp20_work_from_noisy_krotov(stage, outcome)
                if not self._rounds:
                    self._root_prefix_work = _plus_work(self._root_prefix_work, work)
                else:
                    round_index = max(self._rounds)
                    self._post_round_work[round_index] = _plus_work(
                        self._post_round_work.get(round_index, WP20WorkLedger()),
                        work,
                    )
            return outcome

        if predecessor_parameters is None:
            msg = "A pruning stage requires its verified predecessor parameters."
            raise ValueError(msg)
        input_binding = self._binding_for_stage(stage)
        theta = np.asarray(predecessor_parameters, dtype=np.float64)
        spec = PruningStageSpec.from_mapping(
            stage.optimizer_hyperparameters,
            method_id=self.pipeline.method_id,
            score_rule=stage.pruning_rule,
            random_seed=stage.optimizer_seed,
        )
        objective, objective_binding, provider_checksum, ensembles, trajectories, sampling_work = _score_inputs(
            stage,
            input_binding,
            theta,
            self.target,
        )
        round_index = sum(candidate.stage_kind == "prune" for candidate in self.pipeline.stages[: stage.stage_index])
        result = run_pruning_round(
            input_binding,
            theta,
            spec,
            round_index=round_index,
            output_topology_id=stage.output_topology_id,
            objective=objective,
            scoring_trajectory_count=trajectories,
            sampling_work=sampling_work,
        )
        if result.output_circuit_binding.circuit.num_params != stage.output_parameter_count:
            msg = "Observed pruning output does not match the template's exact parameter schedule."
            raise ValueError(msg)
        execution = _execution_for_round(
            stage,
            result,
            objective_binding,
            provider_checksum,
            ensembles,
        )
        self._bindings[result.output_circuit_binding.topology_id] = result.output_circuit_binding
        self._rounds[result.round_index] = result
        self._post_round_work.setdefault(result.round_index, WP20WorkLedger())
        return topdown_pruning_stage_evidence(stage, execution, ensembles)

    def circuit_statistics(self, stage: TrainingStageConfig) -> Mapping[str, object]:
        """Return compiler-derived statistics for the stage output circuit."""
        if stage.stage_index >= len(self.pipeline.stages) or stage != self.pipeline.stages[stage.stage_index]:
            msg = "Stage does not belong to this resolved WP21 pipeline."
            raise ValueError(msg)
        binding = self._bindings.get(stage.output_topology_id)
        if binding is None:
            binding = self._binding_for_stage(stage)
        round_index = (
            sum(candidate.stage_kind == "prune" for candidate in self.pipeline.stages[: stage.stage_index])
            if stage.stage_kind == "prune"
            else None
        )
        return _circuit_statistics(binding, round_index=round_index)


def _validate_execution_stage(
    stage: TrainingStageConfig,
    execution: TopDownPruningStageExecution,
    *,
    expected_round_index: int | None = None,
) -> None:
    """Require exact agreement between a resolved stage and execution."""
    result = execution.round
    expected_method = {rule: method for method, rule in _METHOD_RULES.items()}[stage.pruning_rule]
    spec = PruningStageSpec.from_mapping(
        stage.optimizer_hyperparameters,
        method_id=expected_method,
        score_rule=stage.pruning_rule,
        random_seed=stage.optimizer_seed,
    )
    resolved_round_index = (
        None if expected_round_index is None else require_int(expected_round_index, "expected_round_index")
    )
    expected_provider_checksum = (
        _provider_for_stage(stage).content_checksum
        if spec.scoring_objective_kind == "fixed_map_sample_average_fidelity"
        else None
    )
    if (
        stage.stage_kind != "prune"
        or stage.optimizer_id != "none"
        or execution.stage_configuration_checksum != stage.configuration_checksum
        or execution.stage_index != stage.stage_index
        or execution.stage_id != stage.stage_id
        or execution.method_id != expected_method
        or execution.provider_checksum != expected_provider_checksum
        or result.spec.to_dict() != spec.to_dict()
        or (resolved_round_index is not None and result.round_index != resolved_round_index)
        or result.scoring_trajectory_count != stage.trajectory_count
        or result.input_circuit_binding.topology_id != stage.input_topology_id
        or result.output_circuit_binding.topology_id != stage.output_topology_id
        or result.input_circuit_binding.circuit.num_params != stage.input_parameter_count
        or result.output_circuit_binding.circuit.num_params != stage.output_parameter_count
    ):
        msg = "WP21 pruning execution does not match its resolved pipeline stage."
        raise ValueError(msg)


def validate_topdown_pruning_stage_evidence(
    stage: TrainingStageConfig,
    *,
    execution_document: Mapping[str, object],
    source_parameter_checksum: str | None,
    initial_parameter_checksum: str,
    final_parameter_checksum: str,
    selected_parameter_checksum: str,
    circuit_binding_checksum: str,
    map_circuit_binding_checksum: str,
    provider_checksum: str | None,
    objective_checksum: str | None,
    objective_binding_checksum: str | None,
    trace: Sequence[Mapping[str, object]],
    training_ensembles: Sequence[KrotovFixedMapEnsemble],
    normalized_work: Mapping[str, object],
    circuit_topology: Mapping[str, object],
    circuit_statistics: Mapping[str, object],
    expected_round_index: int | None = None,
) -> Mapping[str, object]:
    """Reconstruct WP21 evidence and return its sole valid stage summary."""
    execution = TopDownPruningStageExecution.from_dict(execution_document)
    _validate_execution_stage(stage, execution, expected_round_index=expected_round_index)
    result = execution.round
    impact = execution.method_id in {
        TOPDOWN_IMPACT_ONE_SHOT_METHOD_ID,
        TOPDOWN_IMPACT_ITERATIVE_METHOD_ID,
    }
    expected_map_binding = (
        result.input_circuit_binding.content_checksum if impact else result.output_circuit_binding.content_checksum
    )
    aliases = {
        "source_parameter_checksum": result.input_parameter_checksum,
        "initial_parameter_checksum": result.output_parameter_checksum,
        "final_parameter_checksum": result.output_parameter_checksum,
        "selected_parameter_checksum": result.output_parameter_checksum,
        "circuit_binding_checksum": result.output_circuit_binding.content_checksum,
        "map_circuit_binding_checksum": expected_map_binding,
    }
    supplied = {
        "source_parameter_checksum": source_parameter_checksum,
        "initial_parameter_checksum": initial_parameter_checksum,
        "final_parameter_checksum": final_parameter_checksum,
        "selected_parameter_checksum": selected_parameter_checksum,
        "circuit_binding_checksum": circuit_binding_checksum,
        "map_circuit_binding_checksum": map_circuit_binding_checksum,
    }
    if aliases != supplied:
        msg = "WP21 pruning vectors or circuit bindings do not match the sealed round."
        raise ValueError(msg)
    maps = tuple(training_ensembles)
    if tuple(item.content_checksum for item in maps) != execution.training_ensemble_checksums:
        msg = "WP21 pruning fixed maps do not match the sealed execution."
        raise ValueError(msg)
    if any(item.gate_count != len(result.input_circuit_binding.circuit.gates) for item in maps):
        msg = "WP21 pruning fixed maps must cover every input-circuit gate."
        raise ValueError(msg)
    if provider_checksum != execution.provider_checksum:
        msg = "WP21 pruning provider checksum does not match the sealed execution."
        raise ValueError(msg)
    expected_objective = None if execution.objective_binding is None else execution.objective_binding.objective_checksum
    expected_objective_binding = (
        None if execution.objective_binding is None else execution.objective_binding.content_checksum
    )
    if objective_checksum != expected_objective or objective_binding_checksum != expected_objective_binding:
        msg = "WP21 pruning target-objective provenance does not match the sealed execution."
        raise ValueError(msg)
    if tuple(thaw_json_mapping(freeze_json_mapping(row, "trace row")) for row in trace) != tuple(
        thaw_json_mapping(row) for row in execution.trace
    ):
        msg = "WP21 pruning trace differs from the sealed execution."
        raise ValueError(msg)
    if thaw_json_mapping(freeze_json_mapping(normalized_work, "normalized_work")) != thaw_json_mapping(
        execution.normalized_work
    ):
        msg = "WP21 pruning normalized work differs from the sealed execution."
        raise ValueError(msg)
    if (
        thaw_json_mapping(freeze_json_mapping(circuit_topology, "circuit_topology"))
        != result.output_circuit_binding.to_dict()
    ):
        msg = "WP21 pruning output topology differs from the sealed circuit binding."
        raise ValueError(msg)
    expected_statistics = _circuit_statistics(result.output_circuit_binding, round_index=result.round_index)
    if thaw_json_mapping(freeze_json_mapping(circuit_statistics, "circuit_statistics")) != expected_statistics:
        msg = "WP21 pruning circuit statistics are not compiler-derived from the sealed output."
        raise ValueError(msg)
    return _training_summary(execution)


def _plus_work(left: WP20WorkLedger, right: WP20WorkLedger) -> WP20WorkLedger:
    """Add two detailed ledgers without importing a private WP20 helper."""
    data = right.to_dict()
    return left.plus(**{
        name: cast("int | float", data[name])
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


def _wp20_work_from_stage_result(
    stage: TrainingStageConfig,
    result: TrainingStageResult,
) -> WP20WorkLedger:
    """Reconstruct verified Krotov work from a persisted WP18 stage row."""
    if (
        result.stage_index != stage.stage_index
        or result.stage_id != stage.stage_id
        or result.stage_configuration_checksum != stage.configuration_checksum
        or stage.optimizer_id != "krotov"
    ):
        msg = "Persisted Krotov work does not identify the supplied WP21 stage."
        raise ValueError(msg)
    raw = {name: require_int(result.normalized_work[name], f"normalized_work.{name}") for name in _NORMALIZED_WORK_KEYS}
    validation_evaluations = 0
    if stage.checkpoint_validation.enabled:
        trajectory_count = stage.checkpoint_validation.trajectory_count
        sampled = raw["checkpoint_validation_trajectories"]
        if sampled % trajectory_count or result.checkpoint_validation_ensemble_checksum is None:
            msg = "Persisted WP21 checkpoint-validation work is not fixed-CRN complete."
            raise ValueError(msg)
        validation_evaluations = sampled // trajectory_count - 1
        if validation_evaluations < 0:
            msg = "Persisted validation-map sampling exceeds recorded trajectory work."
            raise ValueError(msg)
    training_evaluations = raw["objective_evaluations"] - validation_evaluations
    if training_evaluations < 0:
        msg = "Persisted objective work is smaller than checkpoint-validation work."
        raise ValueError(msg)
    completed_updates = require_int(
        result.training_summary["completed_iterations"],
        "training_summary.completed_iterations",
    )
    cross_trajectory_pairings = require_int(
        result.training_summary["cumulative_cross_trajectory_pairings"],
        "training_summary.cumulative_cross_trajectory_pairings",
    )
    forward = raw["training_trajectories"] + raw["checkpoint_validation_trajectories"] + raw["test_trajectories"]
    if stage.trajectory_count == 0:
        forward += training_evaluations
    return WP20WorkLedger(
        forward_circuit_evaluations=forward,
        backward_circuit_evaluations=completed_updates * max(1, stage.trajectory_count),
        trajectory_gate_applications=raw["trajectory_gate_applications"],
        training_trajectories=raw["training_trajectories"],
        checkpoint_validation_trajectories=raw["checkpoint_validation_trajectories"],
        test_trajectories=raw["test_trajectories"],
        objective_calls=raw["objective_evaluations"],
        gradient_calls=raw["gradient_evaluations"],
        cross_trajectory_pairings=cross_trajectory_pairings,
        wall_time_seconds=result.wall_time_seconds,
        peak_memory_bytes=result.peak_memory_bytes,
    )


def topdown_reachable_resource_strata(
    rounds: Sequence[PruningRoundResult],
    *,
    root_prefix_work: WP20WorkLedger,
    post_round_work: Sequence[WP20WorkLedger],
) -> tuple[ReachableResourceStratum, ...]:
    """Project every attempted circuit with exact cumulative path work."""
    values = _validated_pruning_path_rounds(rounds)
    if not isinstance(root_prefix_work, WP20WorkLedger):
        msg = "root_prefix_work must be an exact WP20WorkLedger."
        raise TypeError(msg)
    post = tuple(post_round_work)
    if len(post) != len(values) or not all(isinstance(item, WP20WorkLedger) for item in post):
        msg = "post_round_work must provide one exact WP20 ledger per attempted round."
        raise TypeError(msg)
    cumulative = root_prefix_work
    strata = [
        ReachableResourceStratum(
            stratum_id="topdown_round_0",
            circuit_resources=values[0].input_resources,
            work=cumulative,
        )
    ]
    for index, (result, after_work) in enumerate(zip(values, post, strict=True), start=1):
        cumulative = _plus_work(cumulative, result.work)
        cumulative = _plus_work(cumulative, after_work)
        strata.append(
            ReachableResourceStratum(
                stratum_id=f"topdown_round_{index}",
                circuit_resources=result.output_resources,
                work=cumulative,
            )
        )
    return tuple(strata)


def select_topdown_reachable_resource_stratum(
    rounds: Sequence[PruningRoundResult],
    budget: ResourceBudget,
    *,
    root_prefix_work: WP20WorkLedger,
    post_round_work: Sequence[WP20WorkLedger],
) -> ResourceSelectionOutcome:
    """Select the richest observed pruning round within a sealed WP20 budget."""
    return select_reachable_resource_stratum(
        topdown_reachable_resource_strata(
            rounds,
            root_prefix_work=root_prefix_work,
            post_round_work=post_round_work,
        ),
        budget,
    )


__all__ = [
    "TOPDOWN_DEFAULT_DEEP_DEPTH",
    "TOPDOWN_DEFAULT_FINETUNE_ITERATIONS",
    "TOPDOWN_DEFAULT_INITIAL_SCALE",
    "TOPDOWN_DEFAULT_LEARNING_RATE",
    "TOPDOWN_DEFAULT_NOISE_ID",
    "TOPDOWN_DEFAULT_PRETRAIN_ITERATIONS",
    "TOPDOWN_DEFAULT_RELAXATION_ITERATIONS",
    "TOPDOWN_PRUNING_EXECUTION_SCHEMA_VERSION",
    "TOPDOWN_PRUNING_METHOD_VERSION",
    "TOPDOWN_PRUNING_TRACE_SCHEMA_VERSION",
    "TopDownPruningPathExecution",
    "TopDownPruningStageExecution",
    "TopDownPruningStageRunner",
    "TopDownTemplateOptions",
    "build_topdown_impact_iterative_template",
    "build_topdown_impact_one_shot_template",
    "build_topdown_magnitude_template",
    "build_topdown_pruning_template",
    "build_topdown_random_template",
    "select_topdown_reachable_resource_stratum",
    "topdown_pruning_stage_evidence",
    "topdown_reachable_resource_strata",
    "validate_topdown_pruning_stage_evidence",
]
