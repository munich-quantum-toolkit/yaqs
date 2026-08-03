# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""WP19 bottom-up layerwise BMPD profiles and stage execution.

The historical profile is intentionally isolated from the corrected Phase II
profile. It preserves the archived ``RandomState`` initialization arithmetic,
repeated optimizer seeds, cross-trajectory update, and linear fixed-CRN seed
formula. The corrected profile uses disjoint WP16 seed domains, independent
updates, separately fixed checkpoint validation, and an explicit pilot-frozen
trajectory budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from benchmarks.state_preparation.constants import NOISELESS_NOISE_ID
from benchmarks.state_preparation.noise import (
    FIXED_RATE_NOISE_DEFINITION_VERSION,
    HISTORICAL_FIXED_RATE_NOISE_ID,
)
from mqt.yaqs.optimization import (
    brickwall_matrix_product_disentangler_num_parameters,
    create_brickwall_matrix_product_disentangler_parameterized_circuit,
)

from .legacy_targets import LegacyMaterializedTarget
from .noisy_krotov import (
    NoisyKrotovCircuitBinding,
    NoisyKrotovStageExecution,
    NoisyKrotovStageFailure,
    execute_fixed_rate_krotov_stage,
)
from .pipeline import (
    LEGACY_LAYERWISE_SEED_BINDINGS,
    LEGACY_REPRODUCTION_MANIFEST_CHECKSUM,
    LEGACY_REPRODUCTION_TARGET_IDS,
    CheckpointValidationConfig,
    TrainingPipelineConfig,
    TrainingPipelineTemplate,
    TrainingStageConfig,
    TrainingStageTemplate,
    fixture_target_spec_checksum,
)
from .protocol import TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM
from .targets import MaterializedTarget
from .wp20_resources import measure_circuit_resources

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

LAYERWISE_BMPD_CRN_LEGACY_METHOD_ID = "layerwise_bmpd_crn_legacy_v1"
LAYERWISE_BMPD_CRN_V2_METHOD_ID = "layerwise_bmpd_crn_v2"
LAYERWISE_BMPD_PROFILE_VERSION = "1"
LAYERWISE_BMPD_DEPTHS = (1, 2, 3, 4)
LAYERWISE_BMPD_GROWTH_ITERATIONS = 100
LAYERWISE_BMPD_FINETUNE_ITERATIONS = 200
LAYERWISE_BMPD_INITIAL_SCALE = 0.05
LAYERWISE_BMPD_APPEND_SCALE = 0.001
LEGACY_TRAINING_TRAJECTORY_COUNT = 3
LEGACY_EVALUATION_TRAJECTORY_COUNT = 500
LEGACY_EVALUATION_SEED = 0

_NOISE_VERSION = FIXED_RATE_NOISE_DEFINITION_VERSION
_LEGACY_RNG = "numpy_randomstate_standard_normal_v1"
_CORRECTED_RNG = "numpy_pcg64_standard_normal_v1"
_MODERN_LAYERWISE_METHOD_IDS = frozenset({
    LAYERWISE_BMPD_CRN_V2_METHOD_ID,
    "layerwise_bmpd_cross_crn",
    "layerwise_bmpd_noiseless",
    "layerwise_bmpd_resampled",
})


def bmpd_parameter_count(qubit_count: int, depth: int) -> int:
    """Return the exact scalar parameter count for one layerwise topology.

    Args:
        qubit_count: Number of circuit qubits.
        depth: Positive BMPD depth.

    Returns:
        The number of initial-U3 and nine-scalar BMPD parameters.

    Raises:
        ValueError: If the qubit count or depth is outside the supported range.
    """
    if type(qubit_count) is not int or qubit_count < 2:
        msg = "qubit_count must be an integer of at least two."
        raise ValueError(msg)
    if type(depth) is not int or depth < 1:
        msg = "depth must be a positive integer."
        raise ValueError(msg)
    return brickwall_matrix_product_disentangler_num_parameters(
        qubit_count,
        depth,
        initial_single_qubit_layer=True,
    )


def bmpd_topology_id(qubit_count: int, depth: int) -> str:
    """Return the stable logical topology identifier for one BMPD depth."""
    bmpd_parameter_count(qubit_count, depth)
    return f"bmpd_q{qubit_count}_d{depth}"


def create_bmpd_circuit_binding(qubit_count: int, depth: int) -> NoisyKrotovCircuitBinding:
    """Create the exact logical circuit and frozen WP17 policy binding.

    Returns:
        The checksum-bearing logical circuit policy binding.
    """
    circuit = create_brickwall_matrix_product_disentangler_parameterized_circuit(
        qubit_count,
        depth,
        initial_single_qubit_layer=True,
    )
    return NoisyKrotovCircuitBinding(circuit, bmpd_topology_id(qubit_count, depth))


def _seed_domains() -> dict[str, object]:
    """Return the seven preregistered role-to-domain bindings."""
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
    """Return the frozen Phase II logical-chain materialization policy."""
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
    """Return the seed-free template form of disabled validation."""
    policy = CheckpointValidationConfig.disabled().to_dict()
    del policy["seed"]
    return policy


def _enabled_checkpoint_policy(trajectory_count: int) -> dict[str, object]:
    """Return the v2 fixed noisy validation policy without its resolved seed."""
    config = CheckpointValidationConfig(
        noise_id="depolarizing_1s_all",
        noise_definition_version=_NOISE_VERSION,
        noise_strength_scale=1.0,
        tjm_dt=1.0,
        trajectory_count=trajectory_count,
        seed=1,
        sampling_policy="crn_fixed",
        ensemble_refresh_interval=None,
        cadence=10,
        selection_rule="best_validation_fidelity",
        tie_breaker="earliest_iteration",
    ).to_dict()
    del config["seed"]
    return config


def _stage_template(
    *,
    index: int,
    qubit_count: int,
    depth: int,
    input_depth: int | None,
    initialization_rng: str | None,
    initialization_scale: float | None,
    initialization_binding: str | None,
    optimizer_binding: str,
    noisy: bool,
    training_trajectory_count: int,
    training_binding: str | None,
    checkpoint_validation_trajectory_count: int,
    validation_binding: str | None,
    legacy: bool,
) -> TrainingStageTemplate:
    """Build one exact identity-bearing layerwise stage template.

    Returns:
        A target-independent stage template.
    """
    initial = input_depth is None
    final = depth == LAYERWISE_BMPD_DEPTHS[-1] and input_depth == depth
    transfer = "initialize_random_normal" if initial else ("copy" if final else "append_random_normal")
    hyperparameters: dict[str, object] = {
        "learning_rate": 0.2 if final else 1.0,
        "schedule": "exp" if final else "constant",
        "decay": 0.01 if final else 0.0,
    }
    if initialization_rng is not None and initialization_scale is not None:
        hyperparameters.update({
            "initialization_rng": initialization_rng,
            "initialization_scale": initialization_scale,
        })
    validation_policy = (
        _enabled_checkpoint_policy(checkpoint_validation_trajectory_count)
        if checkpoint_validation_trajectory_count
        else _disabled_checkpoint_policy()
    )
    output_count = bmpd_parameter_count(qubit_count, depth)
    input_count = 0 if input_depth is None else bmpd_parameter_count(qubit_count, input_depth)
    return TrainingStageTemplate(
        stage_policy={
            "stage_index": index,
            "stage_id": "final_finetune" if final else f"grow_d{depth}",
            "stage_kind": "optimize" if initial or final else "grow",
            "input_topology_id": None if input_depth is None else bmpd_topology_id(qubit_count, input_depth),
            "output_topology_id": bmpd_topology_id(qubit_count, depth),
            "input_parameter_count": input_count,
            "output_parameter_count": output_count,
            "parameter_transfer_rule": transfer,
            "optimizer_id": "krotov",
            "optimizer_hyperparameters": hyperparameters,
            "iteration_budget": LAYERWISE_BMPD_FINETUNE_ITERATIONS if final else LAYERWISE_BMPD_GROWTH_ITERATIONS,
            "training_noise_id": (
                HISTORICAL_FIXED_RATE_NOISE_ID
                if legacy and noisy
                else "depolarizing_1s_all"
                if noisy
                else NOISELESS_NOISE_ID
            ),
            "noise_definition_version": _NOISE_VERSION,
            "noise_strength_scale": 1.0 if noisy else None,
            "tjm_dt": 1.0 if noisy else None,
            "trajectory_count": training_trajectory_count if noisy else 0,
            "trajectory_update": "cross" if legacy and noisy else "independent" if noisy else None,
            "sampling_policy": "crn_fixed" if noisy else "none",
            "crn_refresh_interval": None,
            "checkpoint_validation_policy": validation_policy,
            "pruning_rule": "none",
            "pruning_threshold": None,
            "max_bond_dimension": None,
            "svd_threshold": 0.0,
            "truncation_mode": "discarded_weight",
            "min_bond_dimension": 1,
        },
        seed_bindings={
            "initialization": initialization_binding,
            "optimizer": optimizer_binding,
            "training": training_binding,
            "checkpoint_validation": validation_binding,
        },
    )


def build_layerwise_bmpd_crn_legacy_v1_template() -> TrainingPipelineTemplate:
    """Build the faithful five-stage q8 historical reproduction profile.

    Returns:
        The isolated historical layerwise pipeline template.
    """
    stages = tuple(
        _stage_template(
            index=index,
            qubit_count=8,
            depth=depth,
            input_depth=(None if index == 0 else depth - 1),
            initialization_rng=_LEGACY_RNG,
            initialization_scale=(LAYERWISE_BMPD_INITIAL_SCALE if index == 0 else LAYERWISE_BMPD_APPEND_SCALE),
            initialization_binding=LEGACY_LAYERWISE_SEED_BINDINGS[index],
            optimizer_binding=LEGACY_LAYERWISE_SEED_BINDINGS[4],
            noisy=False,
            training_trajectory_count=0,
            training_binding=None,
            checkpoint_validation_trajectory_count=0,
            validation_binding=None,
            legacy=True,
        )
        for index, depth in enumerate(LAYERWISE_BMPD_DEPTHS)
    )
    final = _stage_template(
        index=4,
        qubit_count=8,
        depth=4,
        input_depth=4,
        initialization_rng=None,
        initialization_scale=None,
        initialization_binding=None,
        optimizer_binding=LEGACY_LAYERWISE_SEED_BINDINGS[5],
        noisy=True,
        training_trajectory_count=LEGACY_TRAINING_TRAJECTORY_COUNT,
        training_binding=LEGACY_LAYERWISE_SEED_BINDINGS[6],
        checkpoint_validation_trajectory_count=0,
        validation_binding=None,
        legacy=True,
    )
    return TrainingPipelineTemplate(
        template_id=LAYERWISE_BMPD_CRN_LEGACY_METHOD_ID,
        preregistration_checksum=TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM,
        target_scope_id="legacy_reproduction",
        ansatz_family="bmpd_brickwall",
        method_id=LAYERWISE_BMPD_CRN_LEGACY_METHOD_ID,
        method_version=LAYERWISE_BMPD_PROFILE_VERSION,
        resource_stratum_id="legacy_historical_unmatched",
        stages=(*stages, final),
        seed_domains=_seed_domains(),
        final_materialization_policy=_materialization_policy(),
    )


def build_layerwise_bmpd_crn_v2_template(
    *,
    training_trajectory_count: int,
    checkpoint_validation_trajectory_count: int,
) -> TrainingPipelineTemplate:
    """Build the corrected q6 profile with explicit pilot-frozen counts.

    No default counts are provided because WP22 must freeze them from pilot
    evidence before screening.

    Returns:
        The corrected publication-profile pipeline template.

    Raises:
        ValueError: If either pilot-frozen trajectory count is invalid.
    """
    for value, name in (
        (training_trajectory_count, "training_trajectory_count"),
        (checkpoint_validation_trajectory_count, "checkpoint_validation_trajectory_count"),
    ):
        if type(value) is not int or value < 1:
            msg = f"{name} must be a positive integer frozen by the pilot."
            raise ValueError(msg)
    stages = tuple(
        _stage_template(
            index=index,
            qubit_count=6,
            depth=depth,
            input_depth=(None if index == 0 else depth - 1),
            initialization_rng=_CORRECTED_RNG,
            initialization_scale=(LAYERWISE_BMPD_INITIAL_SCALE if index == 0 else LAYERWISE_BMPD_APPEND_SCALE),
            initialization_binding=f"layerwise_v2_depth{depth}_initialization",
            optimizer_binding=f"layerwise_v2_depth{depth}_optimizer",
            noisy=False,
            training_trajectory_count=0,
            training_binding=None,
            checkpoint_validation_trajectory_count=0,
            validation_binding=None,
            legacy=False,
        )
        for index, depth in enumerate(LAYERWISE_BMPD_DEPTHS)
    )
    final = _stage_template(
        index=4,
        qubit_count=6,
        depth=4,
        input_depth=4,
        initialization_rng=None,
        initialization_scale=None,
        initialization_binding=None,
        optimizer_binding="layerwise_v2_final_optimizer",
        noisy=True,
        training_trajectory_count=training_trajectory_count,
        training_binding="layerwise_v2_fixed_crn_training",
        checkpoint_validation_trajectory_count=checkpoint_validation_trajectory_count,
        validation_binding="layerwise_v2_fixed_crn_validation",
        legacy=False,
    )
    return TrainingPipelineTemplate(
        template_id="layerwise_bmpd_crn_v2_default",
        preregistration_checksum=TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM,
        target_scope_id="primary_q6",
        ansatz_family="bmpd_brickwall",
        method_id=LAYERWISE_BMPD_CRN_V2_METHOD_ID,
        method_version=LAYERWISE_BMPD_PROFILE_VERSION,
        resource_stratum_id="primary_cap_12",
        stages=(*stages, final),
        seed_domains=_seed_domains(),
        final_materialization_policy=_materialization_policy(),
    )


def resolve_layerwise_bmpd_crn_legacy_v1_pipeline(target_seed: int) -> TrainingPipelineConfig:
    """Resolve the faithful profile for one of the five archived target seeds.

    Returns:
        The exact target-bound historical pipeline.

    Raises:
        ValueError: If the target seed is outside the audited five-row universe.
    """
    if type(target_seed) is not int or target_seed not in {100, 200, 300, 400, 500}:
        msg = "target_seed must be one of 100, 200, 300, 400, or 500."
        raise ValueError(msg)
    target_id = f"legacy_tfim_seed_{target_seed}"
    assert target_id in LEGACY_REPRODUCTION_TARGET_IDS
    return build_layerwise_bmpd_crn_legacy_v1_template().resolve(
        target_namespace="legacy_reproduction",
        target_manifest=None,
        target_instance_id=target_id,
        target_population_manifest_checksum=LEGACY_REPRODUCTION_MANIFEST_CHECKSUM,
        target_instance_spec_checksum=fixture_target_spec_checksum("legacy_reproduction", target_id, 8),
        target_family_id="tfim_ground_state",
        target_stratum_id="legacy_disordered",
        qubit_count=8,
        optimization_block_id=f"legacy_reproduction_seed_{target_seed}",
        optimization_seed=target_seed,
        data_role="secondary_benchmark",
    )


def _normal_draws(rng_id: object, seed: int, count: int) -> NDArray[np.float64]:
    """Draw one detached standard-normal vector under a frozen RNG API.

    Returns:
        The generated standard-normal values.

    Raises:
        ValueError: If the configuration names an unsupported RNG API.
    """
    if rng_id == _LEGACY_RNG:
        return np.random.RandomState(seed).standard_normal(count)
    if rng_id == _CORRECTED_RNG:
        generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
        return generator.standard_normal(count)
    msg = f"Unsupported layerwise initialization RNG {rng_id!r}."
    raise ValueError(msg)


def initialize_layerwise_stage_parameters(
    stage: TrainingStageConfig,
    predecessor_parameters: NDArray[np.float64] | None,
    *,
    appended_values: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Apply exact random initialization, prefix growth, or final copying.

    ``appended_values`` is an explicit structural-test hook. Production callers
    leave it unset so the identity-bearing RNG and scale are always used.

    Returns:
        A detached parameter vector for the configured output topology.

    Raises:
        TypeError: If the stage has the wrong record type.
        ValueError: If predecessor, transfer, initialization, or appended values
            differ from the resolved stage identity.
    """
    if not isinstance(stage, TrainingStageConfig):
        msg = "stage must be a TrainingStageConfig."
        raise TypeError(msg)
    predecessor = None
    if predecessor_parameters is not None:
        predecessor = np.asarray(predecessor_parameters, dtype=np.float64)
        if predecessor.shape != (stage.input_parameter_count,) or not np.all(np.isfinite(predecessor)):
            msg = "predecessor_parameters do not match the configured stage input."
            raise ValueError(msg)
    if stage.input_parameter_count == 0 and predecessor is not None:
        msg = "The first layerwise stage cannot consume predecessor parameters."
        raise ValueError(msg)
    if stage.input_parameter_count and predecessor is None:
        msg = "A later layerwise stage requires its exact predecessor parameters."
        raise ValueError(msg)
    if stage.parameter_transfer_rule == "copy":
        if appended_values is not None:
            msg = "A copy stage cannot accept appended_values."
            raise ValueError(msg)
        assert predecessor is not None
        return predecessor.copy()
    if stage.parameter_transfer_rule not in {"initialize_random_normal", "append_random_normal"}:
        msg = "WP19 layerwise execution supports only normal initialization, prefix append, or copy transfer."
        raise ValueError(msg)
    hyperparameters = stage.optimizer_hyperparameters
    rng_id = hyperparameters.get("initialization_rng")
    scale_value = hyperparameters.get("initialization_scale")
    if type(scale_value) is not float or not np.isfinite(scale_value) or scale_value <= 0.0:
        msg = "Layerwise random transfer requires a positive initialization_scale."
        raise ValueError(msg)
    if stage.initialization_seed is None:
        msg = "Layerwise random transfer requires its resolved initialization seed."
        raise ValueError(msg)
    tail_count = stage.output_parameter_count - stage.input_parameter_count
    if appended_values is None:
        tail = _normal_draws(rng_id, stage.initialization_seed, tail_count) * scale_value
    else:
        tail = np.asarray(appended_values, dtype=np.float64)
        if tail.shape != (tail_count,) or not np.all(np.isfinite(tail)):
            msg = "appended_values do not match the newly added parameter slice."
            raise ValueError(msg)
        tail = tail.copy()
    if predecessor is None:
        return np.ascontiguousarray(tail, dtype=np.float64)
    return np.concatenate((predecessor, tail)).astype(np.float64, copy=False)


def _target_identity(target: object) -> Mapping[str, object]:
    """Return one supported typed target identity without accepting raw arrays.

    Returns:
        The immutable target agreement-ledger fields.

    Raises:
        TypeError: If the target is not a typed Phase II or legacy target.
    """
    if not isinstance(target, (MaterializedTarget, LegacyMaterializedTarget)):
        msg = "Layerwise pipeline execution requires a typed materialized target."
        raise TypeError(msg)
    return target.identity_dict()


@dataclass(frozen=True, slots=True)
class LayerwiseBMPDStageRunner:
    """Bind one typed target and resolved WP19 pipeline to the WP17 adapter."""

    pipeline: TrainingPipelineConfig
    target: MaterializedTarget | LegacyMaterializedTarget

    def __post_init__(self) -> None:
        """Verify target and method identity before any optimizer work.

        Raises:
            TypeError: If the pipeline has the wrong record type.
            ValueError: If method and target identities do not agree.
        """
        if not isinstance(self.pipeline, TrainingPipelineConfig):
            msg = "pipeline must be a TrainingPipelineConfig."
            raise TypeError(msg)
        if self.pipeline.method_id not in _MODERN_LAYERWISE_METHOD_IDS | {LAYERWISE_BMPD_CRN_LEGACY_METHOD_ID}:
            msg = "LayerwiseBMPDStageRunner accepts only the registered layerwise BMPD methods."
            raise ValueError(msg)
        identity = _target_identity(self.target)
        expected = {
            "target_instance_id": self.pipeline.target_instance_id,
            "target_instance_spec_checksum": self.pipeline.target_instance_spec_checksum,
            "target_manifest_checksum": self.pipeline.target_population_manifest_checksum,
            "family_id": self.pipeline.target_family_id,
            "stratum_id": self.pipeline.target_stratum_id,
            "qubit_count": self.pipeline.qubit_count,
        }
        if any(identity[name] != value for name, value in expected.items()):
            msg = "Materialized target identity does not match the resolved layerwise pipeline."
            raise ValueError(msg)

    def __call__(
        self,
        stage: TrainingStageConfig,
        predecessor_parameters: NDArray[np.float64] | None,
    ) -> NoisyKrotovStageExecution | NoisyKrotovStageFailure:
        """Initialize and execute exactly one configured layerwise stage.

        Returns:
            The complete WP17 execution or structured stage failure.

        Raises:
            ValueError: If the supplied stage is not part of this pipeline.
        """
        if stage.stage_index >= len(self.pipeline.stages) or stage != self.pipeline.stages[stage.stage_index]:
            msg = "Stage does not belong to this resolved layerwise pipeline."
            raise ValueError(msg)
        depth = LAYERWISE_BMPD_DEPTHS[min(stage.stage_index, len(LAYERWISE_BMPD_DEPTHS) - 1)]
        binding = create_bmpd_circuit_binding(self.pipeline.qubit_count, depth)
        if binding.topology_id != stage.output_topology_id:
            msg = "Resolved layerwise stage topology differs from its BMPD circuit binding."
            raise ValueError(msg)
        initial_theta = initialize_layerwise_stage_parameters(stage, predecessor_parameters)
        compatibility_method_id = (
            self.pipeline.method_id if self.pipeline.method_id == LAYERWISE_BMPD_CRN_LEGACY_METHOD_ID else None
        )
        return execute_fixed_rate_krotov_stage(
            stage,
            binding,
            self.target,
            initial_theta,
            compatibility_method_id=compatibility_method_id,
        )

    def circuit_statistics(self, stage: TrainingStageConfig) -> Mapping[str, object]:
        """Return deterministic logical statistics for persisted stage evidence.

        Returns:
            Logical BMPD topology and gate-count statistics.

        Raises:
            ValueError: If the supplied stage is not part of this pipeline.
        """
        if stage.stage_index >= len(self.pipeline.stages) or stage != self.pipeline.stages[stage.stage_index]:
            msg = "Stage does not belong to this resolved layerwise pipeline."
            raise ValueError(msg)
        depth = LAYERWISE_BMPD_DEPTHS[min(stage.stage_index, len(LAYERWISE_BMPD_DEPTHS) - 1)]
        binding = create_bmpd_circuit_binding(self.pipeline.qubit_count, depth)
        circuit = binding.circuit
        resources = measure_circuit_resources(circuit)
        return {
            "topology_id": stage.output_topology_id,
            "parameter_count": stage.output_parameter_count,
            "qubit_count": self.pipeline.qubit_count,
            "bmpd_depth": depth,
            "logical_gate_count": len(circuit.gates),
            "logical_two_qubit_gate_count": resources.logical_two_qubit_gates,
            "native_two_qubit_gate_count": resources.native_two_qubit_gates,
            "native_two_qubit_gates_per_chain_edge": list(resources.native_two_qubit_gates_per_chain_edge),
            "circuit_resource_metrics": resources.to_dict(),
        }


__all__ = [
    "LAYERWISE_BMPD_APPEND_SCALE",
    "LAYERWISE_BMPD_CRN_LEGACY_METHOD_ID",
    "LAYERWISE_BMPD_CRN_V2_METHOD_ID",
    "LAYERWISE_BMPD_DEPTHS",
    "LAYERWISE_BMPD_FINETUNE_ITERATIONS",
    "LAYERWISE_BMPD_GROWTH_ITERATIONS",
    "LAYERWISE_BMPD_INITIAL_SCALE",
    "LAYERWISE_BMPD_PROFILE_VERSION",
    "LEGACY_EVALUATION_SEED",
    "LEGACY_EVALUATION_TRAJECTORY_COUNT",
    "LEGACY_TRAINING_TRAJECTORY_COUNT",
    "LayerwiseBMPDStageRunner",
    "bmpd_parameter_count",
    "bmpd_topology_id",
    "build_layerwise_bmpd_crn_legacy_v1_template",
    "build_layerwise_bmpd_crn_v2_template",
    "create_bmpd_circuit_binding",
    "initialize_layerwise_stage_parameters",
    "resolve_layerwise_bmpd_crn_legacy_v1_pipeline",
]
