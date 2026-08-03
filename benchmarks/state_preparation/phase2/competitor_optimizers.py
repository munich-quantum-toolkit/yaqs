# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Honest parameter-shift Adam and SPSA competitors for WP20.

Both adapters consume the same resolved :class:`TrainingStageConfig` boundary
as noisy Krotov. Objective callbacks receive an immutable request that fixes
the trajectory stream independently of optimizer randomness. Parameter-shift
pairs and SPSA plus/minus pairs share one objective stream, while resampled
SPSA draws a fresh stream for every update.
"""

# The adapter boundary validates a complete scientific configuration before
# executing it. Small private validators inherit their exception contracts
# from ``validation.py``.
# ruff: noqa: DOC201, DOC501

from __future__ import annotations

import hashlib
import math
import operator
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.typing import NDArray

from benchmarks.state_preparation.constants import NOISELESS_NOISE_ID, STANDARD_NOISE_IDS
from benchmarks.state_preparation.noise import (
    FIXED_RATE_NOISE_DEFINITION_VERSION,
    ScaledStandardNoiseProvider,
    create_scaled_standard_noise_provider,
)
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import (
    KrotovFixedMapEnsemble,
    KrotovTJMOptions,
    KrotovTruncation,
    ParameterizedCircuit,
    noisy_state_preparation_metrics,
    sample_krotov_fixed_map_ensemble,
    state_preparation_metrics,
)

from .canonical import canonical_checksum, canonical_json, freeze_json_mapping, thaw_json_mapping
from .legacy_targets import LegacyMaterializedTarget
from .noisy_krotov import (
    NoisyKrotovCircuitBinding,
    NoisyKrotovObjectiveBinding,
    decode_noisy_krotov_circuit_binding_document,
)
from .pipeline import TrainingPipelineConfig, TrainingPipelineTemplate, TrainingStageConfig, TrainingStageTemplate
from .targets import MaterializedTarget
from .validation import require_checksum, require_exact_keys, require_float, require_int, require_mapping, require_slug
from .wp20_resources import CircuitResourceMetrics, WP20WorkLedger, measure_circuit_resources

if TYPE_CHECKING:
    from .artifacts import StageExecutionEvidence


PARAMETER_SHIFT_ADAM_CONFIG_SCHEMA_VERSION = "yaqs.state_preparation.phase2.parameter_shift_adam_config.v1"
SPSA_CONFIG_SCHEMA_VERSION = "yaqs.state_preparation.phase2.spsa_config.v1"
COMPETITOR_OBJECTIVE_REQUEST_SCHEMA_VERSION = "yaqs.state_preparation.phase2.competitor_objective_request.v1"
COMPETITOR_ITERATION_SCHEMA_VERSION = "yaqs.state_preparation.phase2.competitor_iteration.v1"
COMPETITOR_EXECUTION_SCHEMA_VERSION = "yaqs.state_preparation.phase2.competitor_execution.v1"
COMPETITOR_WORK_BUDGET_SCHEMA_VERSION = "yaqs.state_preparation.phase2.competitor_work_budget.v1"
FIXED_RATE_COMPETITOR_OBJECTIVE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.fixed_rate_competitor_objective.v1"

PARAMETER_SHIFT_POLICY_ID = "exact_pauli_pi_over_2_v1"
SPSA_PERTURBATION_DISTRIBUTION_ID = "rademacher_pcg64_v1"
PARAMETER_SHIFT_ADAM_LAYERWISE_METHOD_ID = "parameter_shift_adam_layerwise"
PARAMETER_SHIFT_ADAM_FIXED_METHOD_ID = "parameter_shift_adam_fixed"
SPSA_LAYERWISE_METHOD_ID = "spsa_layerwise"
SPSA_FIXED_METHOD_ID = "spsa_fixed"

_PAULI_ROTATIONS = frozenset({"rx", "ry", "rz", "rxx", "ryy", "rzz"})
_BMPD_TOPOLOGY_PATTERN = re.compile(r"^bmpd_q(?P<qubits>[1-9][0-9]*)_d(?P<depth>[1-9][0-9]*)$")
_INITIALIZATION_HYPERPARAMETERS = frozenset({"initialization_rng", "initialization_scale"})
_ADAM_HYPERPARAMETERS = frozenset({
    "learning_rate",
    "beta1",
    "beta2",
    "epsilon",
    "parameter_shift",
    "gradient_trajectory_count",
    "sampling_policy",
})
_SPSA_HYPERPARAMETERS = frozenset({
    "a",
    "A",
    "alpha",
    "c",
    "gamma",
    "perturbation_distribution",
    "gradient_trajectory_count",
    "sampling_policy",
})
_BUDGET_FIELDS = (
    "forward_circuit_evaluations",
    "trajectory_gate_applications",
    "training_trajectories",
    "checkpoint_validation_trajectories",
    "objective_calls",
    "gradient_calls",
)

ObjectiveRole = Literal["training", "checkpoint_validation"]
ObjectiveEvaluationKind = Literal["monitoring", "gradient_plus", "gradient_minus", "checkpoint_validation"]
StopReason = Literal["iteration_budget_reached", "work_budget_exhausted"]
CompetitorObjective = Callable[[NDArray[np.float64], "CompetitorObjectiveRequest"], float]


def _vector_bytes(value: NDArray[np.float64]) -> bytes:
    """Return canonical little-endian vector bytes."""
    return np.ascontiguousarray(value, dtype=np.dtype("<f8")).tobytes(order="C")


def _vector_checksum(value: NDArray[np.float64]) -> str:
    """Return a stable parameter-vector checksum."""
    return f"sha256:{hashlib.sha256(_vector_bytes(value)).hexdigest()}"


def _validated_theta(value: object, expected_count: int, name: str) -> NDArray[np.float64]:
    """Return a detached finite parameter vector of the required size."""
    if not isinstance(value, np.ndarray):
        msg = f"{name} must be a NumPy array."
        raise TypeError(msg)
    theta = np.asarray(value, dtype=np.float64)
    if theta.shape != (expected_count,) or not np.all(np.isfinite(theta)):
        msg = f"{name} must be finite with shape ({expected_count},)."
        raise ValueError(msg)
    return theta.copy()


def _validated_loss(value: object, name: str) -> float:
    """Return a finite physical infidelity, normalizing numerical roundoff."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        msg = f"{name} must be a real scalar."
        raise TypeError(msg)
    loss = float(value)
    if not math.isfinite(loss) or not -1e-10 <= loss <= 1.0 + 1e-10:
        msg = f"{name} must lie in [0, 1]."
        raise ValueError(msg)
    return min(1.0, max(0.0, loss))


def _derive_seed(payload: Mapping[str, object]) -> int:
    """Derive an unsigned 64-bit seed without consulting global RNG state."""
    return int.from_bytes(hashlib.sha256(canonical_json(dict(payload)).encode()).digest()[:8], "big")


def _copy_theta(payload: bytes) -> NDArray[np.float64]:
    """Return a detached writable vector from canonical bytes."""
    return np.frombuffer(payload, dtype=np.dtype("<f8")).astype(np.float64, copy=True)


def _validate_optimizer_stage(stage: TrainingStageConfig, optimizer_id: str) -> None:
    """Validate shared WP20 competitor-stage semantics."""
    if not isinstance(stage, TrainingStageConfig):
        msg = "stage must be a TrainingStageConfig."
        raise TypeError(msg)
    if stage.optimizer_id != optimizer_id:
        msg = f"stage.optimizer_id must be {optimizer_id!r}."
        raise ValueError(msg)
    if stage.trajectory_count:
        if stage.trajectory_update != "independent":
            msg = "Noisy Adam and SPSA use independent paired objective evaluations, not cross updates."
            raise ValueError(msg)
    elif stage.training_noise_id != NOISELESS_NOISE_ID or stage.trajectory_update is not None:
        msg = "A zero-trajectory competitor stage must be an explicitly noiseless stage."
        raise ValueError(msg)
    if stage.optimizer_seed is None:
        msg = "A competitor optimizer stage requires optimizer_seed."
        raise ValueError(msg)


def _validate_hyperparameter_keys(stage: TrainingStageConfig, required: frozenset[str]) -> None:
    """Require the named optimizer contract without hidden tuning fields."""
    actual = frozenset(stage.optimizer_hyperparameters)
    if actual - _INITIALIZATION_HYPERPARAMETERS != required:
        msg = (
            "optimizer_hyperparameters must contain exactly the optimizer contract "
            "plus optional initialization_rng/initialization_scale."
        )
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ParameterShiftAdamConfig:
    """Exact full-gradient Adam hyperparameters."""

    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    schema_version: str = field(default=PARAMETER_SHIFT_ADAM_CONFIG_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the conventional Adam parameter domain."""
        object.__setattr__(self, "learning_rate", require_float(self.learning_rate, "learning_rate", minimum=0.0))
        if self.learning_rate <= 0.0:
            msg = "learning_rate must be positive."
            raise ValueError(msg)
        for name in ("beta1", "beta2"):
            value = require_float(getattr(self, name), name, minimum=0.0, maximum=1.0)
            if value >= 1.0:
                msg = f"{name} must be smaller than one."
                raise ValueError(msg)
            object.__setattr__(self, name, value)
        epsilon = require_float(self.epsilon, "epsilon", minimum=0.0)
        if epsilon <= 0.0:
            msg = "epsilon must be positive."
            raise ValueError(msg)
        object.__setattr__(self, "epsilon", epsilon)

    @classmethod
    def from_stage(cls, stage: TrainingStageConfig) -> ParameterShiftAdamConfig:
        """Decode and verify the exact Adam stage policy."""
        _validate_optimizer_stage(stage, "parameter_shift_adam")
        _validate_hyperparameter_keys(stage, _ADAM_HYPERPARAMETERS)
        policy = stage.optimizer_hyperparameters
        if policy["parameter_shift"] != PARAMETER_SHIFT_POLICY_ID:
            msg = f"parameter_shift must be {PARAMETER_SHIFT_POLICY_ID!r}."
            raise ValueError(msg)
        if policy["sampling_policy"] != stage.sampling_policy or stage.sampling_policy not in {"none", "crn_fixed"}:
            msg = "Adam sampling_policy must match the resolved stage."
            raise ValueError(msg)
        if require_int(policy["gradient_trajectory_count"], "gradient_trajectory_count") != stage.trajectory_count:
            msg = "gradient_trajectory_count must equal the stage trajectory count."
            raise ValueError(msg)
        return cls(
            learning_rate=cast("float", policy["learning_rate"]),
            beta1=cast("float", policy["beta1"]),
            beta2=cast("float", policy["beta2"]),
            epsilon=cast("float", policy["epsilon"]),
        )

    def to_dict(self) -> dict[str, object]:
        """Return checksum-ready configuration data."""
        return {
            "schema_version": self.schema_version,
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "parameter_shift": PARAMETER_SHIFT_POLICY_ID,
        }


@dataclass(frozen=True, slots=True)
class SPSAConfig:
    """One-based Spall-style SPSA schedules and perturbation policy."""

    a: float
    stability_constant: float
    alpha: float
    c: float
    gamma: float
    schema_version: str = field(default=SPSA_CONFIG_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate positive gains and a nonnegative stability constant."""
        for name in ("a", "alpha", "c", "gamma"):
            value = require_float(getattr(self, name), name, minimum=0.0)
            if value <= 0.0:
                msg = f"{name} must be positive."
                raise ValueError(msg)
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "stability_constant",
            require_float(self.stability_constant, "stability_constant", minimum=0.0),
        )

    @classmethod
    def from_stage(cls, stage: TrainingStageConfig) -> SPSAConfig:
        """Decode and verify the exact SPSA stage policy."""
        _validate_optimizer_stage(stage, "spsa")
        _validate_hyperparameter_keys(stage, _SPSA_HYPERPARAMETERS)
        policy = stage.optimizer_hyperparameters
        if policy["perturbation_distribution"] != SPSA_PERTURBATION_DISTRIBUTION_ID:
            msg = f"perturbation_distribution must be {SPSA_PERTURBATION_DISTRIBUTION_ID!r}."
            raise ValueError(msg)
        expected_sampling = "none" if stage.trajectory_count == 0 else "resampled"
        if stage.sampling_policy != expected_sampling or policy["sampling_policy"] != expected_sampling:
            msg = "WP20 SPSA requires noiseless growth or a fresh resampled noisy objective on every update."
            raise ValueError(msg)
        if require_int(policy["gradient_trajectory_count"], "gradient_trajectory_count") != stage.trajectory_count:
            msg = "gradient_trajectory_count must equal the stage trajectory count."
            raise ValueError(msg)
        return cls(
            a=cast("float", policy["a"]),
            stability_constant=cast("float", policy["A"]),
            alpha=cast("float", policy["alpha"]),
            c=cast("float", policy["c"]),
            gamma=cast("float", policy["gamma"]),
        )

    def gains(self, iteration: int) -> tuple[float, float]:
        """Return one-based update and perturbation gains."""
        k = require_int(iteration, "iteration", minimum=1)
        return (
            self.a / (k + self.stability_constant) ** self.alpha,
            self.c / k**self.gamma,
        )

    def to_dict(self) -> dict[str, object]:
        """Return checksum-ready configuration data."""
        return {
            "schema_version": self.schema_version,
            "a": self.a,
            "A": self.stability_constant,
            "alpha": self.alpha,
            "c": self.c,
            "gamma": self.gamma,
            "perturbation_distribution": SPSA_PERTURBATION_DISTRIBUTION_ID,
        }


@dataclass(frozen=True, slots=True)
class CompetitorWorkBudget:
    """Optional hard caps checked before each complete optimizer update."""

    forward_circuit_evaluations: int | None = None
    trajectory_gate_applications: int | None = None
    training_trajectories: int | None = None
    checkpoint_validation_trajectories: int | None = None
    objective_calls: int | None = None
    gradient_calls: int | None = None
    schema_version: str = field(default=COMPETITOR_WORK_BUDGET_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate nonnegative optional caps and require at least one."""
        present = 0
        for name in _BUDGET_FIELDS:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, require_int(value, name))
                present += 1
        if not present:
            msg = "CompetitorWorkBudget requires at least one finite cap."
            raise ValueError(msg)

    def admits(self, work: WP20WorkLedger) -> bool:
        """Return whether the complete prospective ledger stays within every cap."""
        if not isinstance(work, WP20WorkLedger):
            msg = "work must be a WP20WorkLedger."
            raise TypeError(msg)
        return all(getattr(self, name) is None or getattr(work, name) <= getattr(self, name) for name in _BUDGET_FIELDS)


@dataclass(frozen=True, slots=True)
class CompetitorObjectiveRequest:
    """Immutable objective-evaluation and random-stream identity."""

    stage_configuration_checksum: str
    role: ObjectiveRole
    evaluation_kind: ObjectiveEvaluationKind
    global_iteration: int
    pair_index: int
    sampling_epoch: int
    trajectory_count: int
    trajectory_seed: int | None
    random_stream_checksum: str
    schema_version: str = field(default=COMPETITOR_OBJECTIVE_REQUEST_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate request coordinates and noiseless/noisy seed consistency."""
        object.__setattr__(
            self,
            "stage_configuration_checksum",
            require_checksum(self.stage_configuration_checksum, "stage_configuration_checksum"),
        )
        if self.role not in {"training", "checkpoint_validation"}:
            msg = "role must be training or checkpoint_validation."
            raise ValueError(msg)
        if self.evaluation_kind not in {"monitoring", "gradient_plus", "gradient_minus", "checkpoint_validation"}:
            msg = "evaluation_kind is not a WP20 objective-evaluation kind."
            raise ValueError(msg)
        object.__setattr__(self, "global_iteration", require_int(self.global_iteration, "global_iteration"))
        object.__setattr__(self, "pair_index", require_int(self.pair_index, "pair_index"))
        object.__setattr__(self, "sampling_epoch", require_int(self.sampling_epoch, "sampling_epoch"))
        count = require_int(self.trajectory_count, "trajectory_count")
        object.__setattr__(self, "trajectory_count", count)
        if (count == 0) != (self.trajectory_seed is None):
            msg = "trajectory_seed is present exactly for sampled objectives."
            raise ValueError(msg)
        if self.trajectory_seed is not None:
            seed = require_int(self.trajectory_seed, "trajectory_seed")
            if seed >= 2**64:
                msg = "trajectory_seed must fit an unsigned 64-bit integer."
                raise ValueError(msg)
            object.__setattr__(self, "trajectory_seed", seed)
        object.__setattr__(
            self,
            "random_stream_checksum",
            require_checksum(self.random_stream_checksum, "random_stream_checksum"),
        )

    @property
    def content_checksum(self) -> str:
        """Checksum of the evaluation request and stream binding."""
        return canonical_checksum(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        """Return detached JSON-native request data."""
        return {
            "schema_version": self.schema_version,
            "stage_configuration_checksum": self.stage_configuration_checksum,
            "role": self.role,
            "evaluation_kind": self.evaluation_kind,
            "global_iteration": self.global_iteration,
            "pair_index": self.pair_index,
            "sampling_epoch": self.sampling_epoch,
            "trajectory_count": self.trajectory_count,
            "trajectory_seed": self.trajectory_seed,
            "random_stream_checksum": self.random_stream_checksum,
        }

    @classmethod
    def from_dict(cls, value: object) -> CompetitorObjectiveRequest:
        """Decode one strict objective request from persisted trace data."""
        mapping = require_mapping(value, "competitor objective request")
        expected = frozenset({
            "schema_version",
            "stage_configuration_checksum",
            "role",
            "evaluation_kind",
            "global_iteration",
            "pair_index",
            "sampling_epoch",
            "trajectory_count",
            "trajectory_seed",
            "random_stream_checksum",
        })
        require_exact_keys(mapping, expected, "competitor objective request")
        if mapping["schema_version"] != COMPETITOR_OBJECTIVE_REQUEST_SCHEMA_VERSION:
            msg = "Competitor objective request uses an unsupported schema version."
            raise ValueError(msg)
        return cls(
            stage_configuration_checksum=cast("str", mapping["stage_configuration_checksum"]),
            role=cast("ObjectiveRole", mapping["role"]),
            evaluation_kind=cast("ObjectiveEvaluationKind", mapping["evaluation_kind"]),
            global_iteration=cast("int", mapping["global_iteration"]),
            pair_index=cast("int", mapping["pair_index"]),
            sampling_epoch=cast("int", mapping["sampling_epoch"]),
            trajectory_count=cast("int", mapping["trajectory_count"]),
            trajectory_seed=cast("int | None", mapping["trajectory_seed"]),
            random_stream_checksum=cast("str", mapping["random_stream_checksum"]),
        )


def _sampling_epoch(policy: str, call_index: int, refresh_interval: int | None) -> int:
    """Return the exact sampling window for one objective call or pair."""
    if policy in {"none", "crn_fixed"}:
        return 0
    if policy == "resampled":
        return call_index
    if policy == "crn_refresh":
        assert refresh_interval is not None
        return call_index // refresh_interval
    msg = f"Unsupported sampling policy {policy!r}."
    raise ValueError(msg)


def _objective_request(
    stage: TrainingStageConfig,
    *,
    role: ObjectiveRole,
    evaluation_kind: ObjectiveEvaluationKind,
    global_iteration: int,
    pair_index: int,
    call_index: int,
) -> CompetitorObjectiveRequest:
    """Create one request while keeping optimizer and trajectory streams disjoint."""
    if role == "training":
        policy = stage.sampling_policy
        refresh = stage.crn_refresh_interval
        count = stage.trajectory_count
        base_seed = stage.training_seed
    else:
        policy = stage.checkpoint_validation.sampling_policy
        refresh = stage.checkpoint_validation.ensemble_refresh_interval
        count = stage.checkpoint_validation.trajectory_count
        base_seed = stage.checkpoint_validation.seed
    epoch = _sampling_epoch(policy, call_index, refresh)
    stream_identity = {
        "derivation_version": "yaqs.state_preparation.phase2.competitor_objective_stream.v1",
        "stage_configuration_checksum": stage.configuration_checksum,
        "role": role,
        "sampling_policy": policy,
        "sampling_epoch": epoch,
        "base_seed": base_seed,
    }
    stream_checksum = canonical_checksum(stream_identity)
    trajectory_seed = None if count == 0 else cast("int", base_seed)
    return CompetitorObjectiveRequest(
        stage_configuration_checksum=stage.configuration_checksum,
        role=role,
        evaluation_kind=evaluation_kind,
        global_iteration=global_iteration,
        pair_index=pair_index,
        sampling_epoch=epoch,
        trajectory_count=count,
        trajectory_seed=trajectory_seed,
        random_stream_checksum=stream_checksum,
    )


class FixedRateNoisyCompetitorObjective:
    """Target-bound fixed-map objective shared by WP20 Adam and SPSA.

    The first request in each configured sampling window samples one exact
    provider-backed trajectory-map ensemble. Every other request in that
    window replays those maps, so paired perturbations use common randomness
    and the resulting ensembles can be persisted by the Phase II store.
    """

    def __init__(
        self,
        stage: TrainingStageConfig,
        circuit_binding: NoisyKrotovCircuitBinding,
        target: MaterializedTarget | LegacyMaterializedTarget,
    ) -> None:
        """Bind the resolved stage, logical circuit, and authorized target."""
        if not isinstance(stage, TrainingStageConfig):
            msg = "stage must be a TrainingStageConfig."
            raise TypeError(msg)
        if stage.optimizer_id not in {"parameter_shift_adam", "spsa"}:
            msg = "Fixed-rate competitor objectives require an Adam or SPSA stage."
            raise ValueError(msg)
        if not isinstance(circuit_binding, NoisyKrotovCircuitBinding):
            msg = "circuit_binding must be a NoisyKrotovCircuitBinding."
            raise TypeError(msg)
        if not isinstance(target, (MaterializedTarget, LegacyMaterializedTarget)):
            msg = "A publishable competitor objective requires an authorized materialized target."
            raise TypeError(msg)
        if (
            circuit_binding.topology_id != stage.output_topology_id
            or circuit_binding.circuit.num_params != stage.output_parameter_count
            or target.qubit_count != circuit_binding.circuit.num_qubits
        ):
            msg = "Competitor objective stage, circuit, and target dimensions do not agree."
            raise ValueError(msg)
        self.stage = stage
        self.circuit_binding = circuit_binding
        self.target = target
        self.objective_binding = NoisyKrotovObjectiveBinding.from_inputs(
            target,
            None,
            num_qubits=circuit_binding.circuit.num_qubits,
        )
        self.truncation = KrotovTruncation(
            max_bond_dim=stage.max_bond_dimension,
            svd_threshold=stage.svd_threshold,
            trunc_mode=stage.truncation_mode,
            min_bond_dim=stage.min_bond_dimension,
        )
        self._training_provider, self._training_options = self._noise_runtime("training")
        self._validation_provider, self._validation_options = self._noise_runtime("checkpoint_validation")
        self._ensembles: dict[tuple[ObjectiveRole, int], KrotovFixedMapEnsemble] = {}
        self._sampling_parameter_checksums: dict[tuple[ObjectiveRole, int], str] = {}
        self._ordered_training: list[KrotovFixedMapEnsemble] = []
        self._ordered_validation: list[KrotovFixedMapEnsemble] = []
        self._ordered_training_sampling_parameters: list[str] = []
        self._ordered_validation_sampling_parameters: list[str] = []

    @property
    def objective_checksum(self) -> str:
        """Return the exact target/initial-state objective checksum."""
        return self.objective_binding.objective_checksum

    @property
    def provider_checksum(self) -> str | None:
        """Return training-provider provenance, absent for noiseless growth."""
        provider = self._training_provider
        return None if provider is None else provider.content_checksum

    @property
    def checkpoint_validation_provider_checksum(self) -> str | None:
        """Return checkpoint-validation provider provenance when enabled."""
        provider = self._validation_provider
        return None if provider is None else provider.content_checksum

    @property
    def training_ensembles(self) -> tuple[KrotovFixedMapEnsemble, ...]:
        """Return sampled training ensembles in schedule order."""
        return tuple(self._ordered_training)

    @property
    def checkpoint_validation_ensembles(self) -> tuple[KrotovFixedMapEnsemble, ...]:
        """Return sampled validation ensembles in schedule order."""
        return tuple(self._ordered_validation)

    @property
    def training_ensemble_sampling_parameter_checksums(self) -> tuple[str, ...]:
        """Return the parameter center used to generate every training map."""
        return tuple(self._ordered_training_sampling_parameters)

    @property
    def checkpoint_validation_ensemble_sampling_parameter_checksums(self) -> tuple[str, ...]:
        """Return the candidate parameters used to generate validation maps."""
        return tuple(self._ordered_validation_sampling_parameters)

    @property
    def content_checksum(self) -> str:
        """Seal the stage, circuit, objective, and both provider identities."""
        return canonical_checksum({
            "schema_version": FIXED_RATE_COMPETITOR_OBJECTIVE_SCHEMA_VERSION,
            "stage_configuration_checksum": self.stage.configuration_checksum,
            "circuit_binding_checksum": self.circuit_binding.content_checksum,
            "objective_binding_checksum": self.objective_binding.content_checksum,
            "provider_checksum": self.provider_checksum,
            "checkpoint_validation_provider_checksum": self.checkpoint_validation_provider_checksum,
        })

    def _noise_runtime(
        self,
        role: ObjectiveRole,
    ) -> tuple[ScaledStandardNoiseProvider | None, KrotovTJMOptions | None]:
        """Construct one frozen standard-noise provider/options pair."""
        if role == "training":
            noise_id = self.stage.training_noise_id
            definition_version = self.stage.noise_definition_version
            strength = self.stage.noise_strength_scale
            dt = self.stage.tjm_dt
            count = self.stage.trajectory_count
            seed = self.stage.training_seed
        else:
            config = self.stage.checkpoint_validation
            if not config.enabled:
                return None, None
            noise_id = config.noise_id
            definition_version = config.noise_definition_version
            strength = config.noise_strength_scale
            dt = config.tjm_dt
            count = config.trajectory_count
            seed = config.seed
        if count == 0:
            if noise_id != NOISELESS_NOISE_ID:
                msg = "A zero-trajectory competitor objective must be explicitly noiseless."
                raise ValueError(msg)
            return None, None
        if (
            noise_id not in STANDARD_NOISE_IDS
            or definition_version != FIXED_RATE_NOISE_DEFINITION_VERSION
            or strength is None
            or dt is None
            or seed is None
        ):
            msg = "WP20 competitor objectives require a complete standard fixed-rate noise profile."
            raise ValueError(msg)
        provider = create_scaled_standard_noise_provider(noise_id, strength)
        options = KrotovTJMOptions(
            num_trajectories=count,
            random_seed=seed,
            dt=dt,
            apply_noise_to="all",
            noisy_gate_indices=self.circuit_binding.noisy_gate_indices,
            trajectory_update="independent",
            differentiate_jump_normalization=False,
            use_crn=False,
        )
        return provider, options

    def _expected_stream_checksum(self, request: CompetitorObjectiveRequest) -> str:
        """Recompute the exact role/window stream identity."""
        if request.role == "training":
            policy = self.stage.sampling_policy
            base_seed = self.stage.training_seed
        else:
            policy = self.stage.checkpoint_validation.sampling_policy
            base_seed = self.stage.checkpoint_validation.seed
        return canonical_checksum({
            "derivation_version": "yaqs.state_preparation.phase2.competitor_objective_stream.v1",
            "stage_configuration_checksum": self.stage.configuration_checksum,
            "role": request.role,
            "sampling_policy": policy,
            "sampling_epoch": request.sampling_epoch,
            "base_seed": base_seed,
        })

    def _validate_request(self, request: CompetitorObjectiveRequest) -> None:
        """Reject a request outside this exact stage and role schedule."""
        if not isinstance(request, CompetitorObjectiveRequest):
            msg = "request must be a CompetitorObjectiveRequest."
            raise TypeError(msg)
        if request.role == "training":
            count = self.stage.trajectory_count
            seed = self.stage.training_seed
        else:
            count = self.stage.checkpoint_validation.trajectory_count
            seed = self.stage.checkpoint_validation.seed
        if (
            request.stage_configuration_checksum != self.stage.configuration_checksum
            or request.trajectory_count != count
            or request.trajectory_seed != (None if count == 0 else seed)
            or request.random_stream_checksum != self._expected_stream_checksum(request)
        ):
            msg = "Competitor objective request does not match the target-bound stage schedule."
            raise ValueError(msg)

    def _schedule_coordinates(self, request: CompetitorObjectiveRequest) -> tuple[int, int, int]:
        """Return artifact-compatible coordinates for one request window."""
        if request.role == "training":
            policy = self.stage.sampling_policy
            refresh = self.stage.crn_refresh_interval
        else:
            policy = self.stage.checkpoint_validation.sampling_policy
            refresh = self.stage.checkpoint_validation.ensemble_refresh_interval
        if policy == "crn_fixed":
            return 0, 0, 0
        if policy == "resampled":
            return request.sampling_epoch, request.sampling_epoch, request.sampling_epoch
        if policy == "crn_refresh":
            assert refresh is not None
            return request.sampling_epoch, request.sampling_epoch, request.sampling_epoch * refresh
        msg = "A sampled competitor request requires a CRN or resampled policy."
        raise ValueError(msg)

    def extra_sampling_work_for_requests(
        self,
        requests: Sequence[CompetitorObjectiveRequest],
        gate_count: int,
    ) -> WP20WorkLedger:
        """Return map-generation work not already represented by objective calls."""
        unseen: set[tuple[ObjectiveRole, int]] = set()
        work = WP20WorkLedger()
        for request in requests:
            self._validate_request(request)
            if request.trajectory_count == 0:
                continue
            key = (request.role, request.sampling_epoch)
            if key in self._ensembles or key in unseen:
                continue
            unseen.add(key)
            work = _combine_work(
                work,
                WP20WorkLedger(
                    forward_circuit_evaluations=request.trajectory_count,
                    trajectory_gate_applications=request.trajectory_count * gate_count,
                    training_trajectories=request.trajectory_count if request.role == "training" else 0,
                    checkpoint_validation_trajectories=(
                        request.trajectory_count if request.role == "checkpoint_validation" else 0
                    ),
                ),
            )
        return work

    def _ensure_ensemble(
        self,
        parameters: NDArray[np.float64],
        request: CompetitorObjectiveRequest,
    ) -> KrotovFixedMapEnsemble:
        """Sample one unseen window at an explicitly supplied parameter center."""
        if request.role == "training":
            provider = self._training_provider
            options = self._training_options
            resolved_seed = self.stage.training_seed
            role = "training_trajectory"
            ordered = self._ordered_training
            ordered_parameters = self._ordered_training_sampling_parameters
        else:
            provider = self._validation_provider
            options = self._validation_options
            resolved_seed = self.stage.checkpoint_validation.seed
            role = "checkpoint_validation"
            ordered = self._ordered_validation
            ordered_parameters = self._ordered_validation_sampling_parameters
        assert provider is not None
        assert options is not None
        assert resolved_seed is not None
        key = (request.role, request.sampling_epoch)
        parameter_checksum = _vector_checksum(parameters)
        ensemble = self._ensembles.get(key)
        if ensemble is not None:
            return ensemble
        ensemble_index, refresh_index, global_start = self._schedule_coordinates(request)
        ensemble = sample_krotov_fixed_map_ensemble(
            self.circuit_binding.circuit,
            parameters,
            MPS(self.circuit_binding.circuit.num_qubits),
            self.truncation,
            provider,
            options,
            role=role,
            resolved_seed=resolved_seed,
            stage_index=self.stage.stage_index,
            stage_id=self.stage.stage_id,
            stage_configuration_checksum=self.stage.configuration_checksum,
            circuit_checksum=self.circuit_binding.content_checksum,
            provider_checksum=provider.content_checksum,
            ensemble_index=ensemble_index,
            refresh_index=refresh_index,
            global_iteration_start=global_start,
        )
        self._ensembles[key] = ensemble
        self._sampling_parameter_checksums[key] = parameter_checksum
        ordered.append(ensemble)
        ordered_parameters.append(parameter_checksum)
        return ensemble

    def prime_sampling_window(
        self,
        theta: NDArray[np.float64],
        request: CompetitorObjectiveRequest,
    ) -> None:
        """Generate an unseen noisy window at an unperturbed optimizer center."""
        self._validate_request(request)
        parameters = _validated_theta(theta, self.circuit_binding.circuit.num_params, "theta")
        if request.trajectory_count:
            self._ensure_ensemble(parameters, request)

    def __call__(self, theta: NDArray[np.float64], request: CompetitorObjectiveRequest) -> float:
        """Evaluate the exact noiseless or replayed fixed-map infidelity."""
        self._validate_request(request)
        parameters = _validated_theta(theta, self.circuit_binding.circuit.num_params, "theta")
        circuit = self.circuit_binding.circuit
        target = self.target.state_vector_copy()
        if request.trajectory_count == 0:
            loss, _fidelity = state_preparation_metrics(
                circuit,
                parameters,
                target,
                initial_state=MPS(circuit.num_qubits),
                truncation=self.truncation,
            )
            return loss
        if request.role == "training":
            provider = self._training_provider
            options = self._training_options
        else:
            provider = self._validation_provider
            options = self._validation_options
        assert provider is not None
        assert options is not None
        ensemble = self._ensure_ensemble(parameters, request)
        loss, _fidelity, _trajectory_fidelities = noisy_state_preparation_metrics(
            circuit,
            parameters,
            target,
            None,
            options,
            initial_state=MPS(circuit.num_qubits),
            truncation=self.truncation,
            fixed_noise_maps=ensemble.replay_maps(),
            noise_provider=provider,
        )
        return loss


def _evaluation_work(trajectory_count: int, gate_count: int, *, validation: bool = False) -> WP20WorkLedger:
    """Return exact work for one callback evaluation."""
    forward = max(1, trajectory_count)
    return WP20WorkLedger(
        forward_circuit_evaluations=forward,
        trajectory_gate_applications=trajectory_count * gate_count,
        training_trajectories=0 if validation else trajectory_count,
        checkpoint_validation_trajectories=trajectory_count if validation else 0,
        objective_calls=1,
    )


def _combine_work(*items: WP20WorkLedger) -> WP20WorkLedger:
    """Return the additive sum of detailed work ledgers."""
    total = WP20WorkLedger()
    for item in items:
        increments = item.to_dict()
        total = total.plus(**{
            name: cast("int | float", increments[name])
            for name in (
                *_BUDGET_FIELDS,
                "backward_circuit_evaluations",
                "test_trajectories",
                "cross_trajectory_pairings",
                "wall_time_seconds",
                "peak_memory_bytes",
            )
        })
    return total


def _prospective_work(
    work: WP20WorkLedger,
    *,
    training_evaluations: int,
    validation_evaluations: int,
    stage: TrainingStageConfig,
    gate_count: int,
    gradient_calls: int = 0,
) -> WP20WorkLedger:
    """Return work after one atomic group of evaluations."""
    training = _evaluation_work(stage.trajectory_count, gate_count)
    validation = _evaluation_work(
        stage.checkpoint_validation.trajectory_count,
        gate_count,
        validation=True,
    )
    increments = WP20WorkLedger(gradient_calls=gradient_calls)
    increments = _combine_work(
        increments,
        *([training] * training_evaluations),
        *([validation] * validation_evaluations),
    )
    return _combine_work(work, increments)


def _extra_sampling_work(
    callback: CompetitorObjective | None,
    requests: Sequence[CompetitorObjectiveRequest],
    gate_count: int,
) -> WP20WorkLedger:
    """Return concrete fixed-map generation work for a target-bound evaluator."""
    if isinstance(callback, FixedRateNoisyCompetitorObjective):
        return callback.extra_sampling_work_for_requests(requests, gate_count)
    return WP20WorkLedger()


def _call_objective(
    callback: CompetitorObjective,
    theta: NDArray[np.float64],
    request: CompetitorObjectiveRequest,
) -> float:
    """Evaluate a callback on detached parameters and validate its loss."""
    if not callable(callback):
        msg = "objective callback must be callable."
        raise TypeError(msg)
    return _validated_loss(callback(theta.copy(), request), "objective loss")


def _should_validate(stage: TrainingStageConfig, iteration: int) -> bool:
    """Return whether the frozen checkpoint cadence includes this state."""
    if not stage.checkpoint_validation.enabled:
        return False
    cadence = cast("int", stage.checkpoint_validation.cadence)
    return iteration in {0, stage.iteration_budget} or iteration % cadence == 0


def _bound_objective_for_execution(
    *,
    stage: TrainingStageConfig,
    circuit_binding: NoisyKrotovCircuitBinding,
    objective_checksum: str,
    provider_checksum: str | None,
    objective: CompetitorObjective,
    checkpoint_objective: CompetitorObjective | None,
) -> FixedRateNoisyCompetitorObjective | None:
    """Validate a publishable evaluator or retain a test-only generic callback."""
    if isinstance(objective, FixedRateNoisyCompetitorObjective):
        if (
            objective.stage != stage
            or objective.circuit_binding.content_checksum != circuit_binding.content_checksum
            or objective.objective_checksum != objective_checksum
            or objective.provider_checksum != provider_checksum
        ):
            msg = "Target-bound objective provenance does not match the competitor adapter."
            raise ValueError(msg)
        if stage.checkpoint_validation.enabled and checkpoint_objective is not objective:
            msg = "Checkpoint validation must use the same target-bound evaluator and evidence cache."
            raise ValueError(msg)
        if not stage.checkpoint_validation.enabled and checkpoint_objective is not None:
            msg = "A stage without checkpoint validation cannot accept a checkpoint objective."
            raise ValueError(msg)
        return objective
    if isinstance(checkpoint_objective, FixedRateNoisyCompetitorObjective):
        msg = "Generic and target-bound competitor callbacks cannot be mixed."
        raise TypeError(msg)
    return None


@dataclass(frozen=True, slots=True)
class CompetitorIterationRecord:
    """One fully accounted Adam or SPSA state transition."""

    global_iteration: int
    parameters: tuple[float, ...]
    parameter_checksum: str
    monitoring_loss: float
    checkpoint_validation_fidelity: float | None
    gradient: tuple[float, ...]
    gradient_norm: float
    update_norm: float
    learning_rate: float
    perturbation_scale: float | None
    plus_losses: tuple[float, ...]
    minus_losses: tuple[float, ...]
    objective_requests: tuple[CompetitorObjectiveRequest, ...]
    cumulative_work: WP20WorkLedger
    schema_version: str = field(default=COMPETITOR_ITERATION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate trace norms, losses, checksums, and initial-row semantics."""
        iteration = require_int(self.global_iteration, "global_iteration")
        object.__setattr__(self, "global_iteration", iteration)
        parameters = tuple(require_float(value, "parameters") for value in self.parameters)
        if not parameters:
            msg = "parameters must contain the complete nonempty optimizer state."
            raise ValueError(msg)
        parameter_vector = np.asarray(parameters, dtype=np.float64)
        checksum = require_checksum(self.parameter_checksum, "parameter_checksum")
        if checksum != _vector_checksum(parameter_vector):
            msg = "parameter_checksum is not derived from the persisted parameter vector."
            raise ValueError(msg)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "parameter_checksum", checksum)
        object.__setattr__(self, "monitoring_loss", _validated_loss(self.monitoring_loss, "monitoring_loss"))
        if self.checkpoint_validation_fidelity is not None:
            object.__setattr__(
                self,
                "checkpoint_validation_fidelity",
                1.0 - _validated_loss(1.0 - self.checkpoint_validation_fidelity, "checkpoint fidelity"),
            )
        gradient = tuple(require_float(value, "gradient") for value in self.gradient)
        object.__setattr__(self, "gradient", gradient)
        gradient_norm = require_float(self.gradient_norm, "gradient_norm", minimum=0.0)
        if not math.isclose(gradient_norm, float(np.linalg.norm(gradient)), rel_tol=1e-12, abs_tol=1e-12):
            msg = "gradient_norm does not match gradient."
            raise ValueError(msg)
        object.__setattr__(self, "gradient_norm", gradient_norm)
        object.__setattr__(self, "update_norm", require_float(self.update_norm, "update_norm", minimum=0.0))
        object.__setattr__(self, "learning_rate", require_float(self.learning_rate, "learning_rate", minimum=0.0))
        if self.perturbation_scale is not None:
            object.__setattr__(
                self,
                "perturbation_scale",
                require_float(self.perturbation_scale, "perturbation_scale", minimum=0.0),
            )
        plus = tuple(_validated_loss(value, "plus_loss") for value in self.plus_losses)
        minus = tuple(_validated_loss(value, "minus_loss") for value in self.minus_losses)
        if len(plus) != len(minus):
            msg = "plus_losses and minus_losses must have equal length."
            raise ValueError(msg)
        object.__setattr__(self, "plus_losses", plus)
        object.__setattr__(self, "minus_losses", minus)
        requests = tuple(self.objective_requests)
        if not all(isinstance(request, CompetitorObjectiveRequest) for request in requests):
            msg = "objective_requests must contain CompetitorObjectiveRequest values."
            raise TypeError(msg)
        if len(requests) != 1 + 2 * len(plus) + int(self.checkpoint_validation_fidelity is not None):
            msg = "Trace request count is not implied by monitoring, gradient pairs, and validation."
            raise ValueError(msg)
        object.__setattr__(self, "objective_requests", requests)
        if not isinstance(self.cumulative_work, WP20WorkLedger):
            msg = "cumulative_work must be a WP20WorkLedger."
            raise TypeError(msg)
        if iteration == 0 and (gradient or plus or minus or self.update_norm or self.learning_rate):
            msg = "The initial trace row cannot claim an optimizer update."
            raise ValueError(msg)

    @property
    def objective_request_checksums(self) -> tuple[str, ...]:
        """Return checksums of the complete persisted objective requests."""
        return tuple(request.content_checksum for request in self.objective_requests)

    @property
    def objective_stream_checksums(self) -> tuple[str, ...]:
        """Return objective random-stream checksums in evaluation order."""
        return tuple(request.random_stream_checksum for request in self.objective_requests)

    def to_dict(self) -> dict[str, object]:
        """Return detached JSON-native trace evidence."""
        return {
            "schema_version": self.schema_version,
            "global_iteration": self.global_iteration,
            "parameters": list(self.parameters),
            "parameter_checksum": self.parameter_checksum,
            "monitoring_loss": self.monitoring_loss,
            "checkpoint_validation_fidelity": self.checkpoint_validation_fidelity,
            "gradient": list(self.gradient),
            "gradient_norm": self.gradient_norm,
            "update_norm": self.update_norm,
            "learning_rate": self.learning_rate,
            "perturbation_scale": self.perturbation_scale,
            "plus_losses": list(self.plus_losses),
            "minus_losses": list(self.minus_losses),
            "objective_requests": [request.to_dict() for request in self.objective_requests],
            "objective_request_checksums": list(self.objective_request_checksums),
            "objective_stream_checksums": list(self.objective_stream_checksums),
            "cumulative_work": self.cumulative_work.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> CompetitorIterationRecord:
        """Decode one strict optimizer trace row and recompute all aliases."""
        mapping = require_mapping(value, "competitor iteration record")
        expected = frozenset({
            "schema_version",
            "global_iteration",
            "parameters",
            "parameter_checksum",
            "monitoring_loss",
            "checkpoint_validation_fidelity",
            "gradient",
            "gradient_norm",
            "update_norm",
            "learning_rate",
            "perturbation_scale",
            "plus_losses",
            "minus_losses",
            "objective_requests",
            "objective_request_checksums",
            "objective_stream_checksums",
            "cumulative_work",
        })
        require_exact_keys(mapping, expected, "competitor iteration record")
        if mapping["schema_version"] != COMPETITOR_ITERATION_SCHEMA_VERSION:
            msg = "Competitor iteration record uses an unsupported schema version."
            raise ValueError(msg)
        requests_value = mapping["objective_requests"]
        if isinstance(requests_value, (str, bytes)) or not isinstance(requests_value, Sequence):
            msg = "objective_requests must be a sequence."
            raise TypeError(msg)
        row = cls(
            global_iteration=cast("int", mapping["global_iteration"]),
            parameters=tuple(cast("Sequence[float]", mapping["parameters"])),
            parameter_checksum=cast("str", mapping["parameter_checksum"]),
            monitoring_loss=cast("float", mapping["monitoring_loss"]),
            checkpoint_validation_fidelity=cast("float | None", mapping["checkpoint_validation_fidelity"]),
            gradient=tuple(cast("Sequence[float]", mapping["gradient"])),
            gradient_norm=cast("float", mapping["gradient_norm"]),
            update_norm=cast("float", mapping["update_norm"]),
            learning_rate=cast("float", mapping["learning_rate"]),
            perturbation_scale=cast("float | None", mapping["perturbation_scale"]),
            plus_losses=tuple(cast("Sequence[float]", mapping["plus_losses"])),
            minus_losses=tuple(cast("Sequence[float]", mapping["minus_losses"])),
            objective_requests=tuple(CompetitorObjectiveRequest.from_dict(item) for item in requests_value),
            cumulative_work=WP20WorkLedger.from_dict(mapping["cumulative_work"]),
        )
        if row.objective_request_checksums != tuple(cast("Sequence[str]", mapping["objective_request_checksums"])):
            msg = "Objective-request checksum aliases do not match persisted requests."
            raise ValueError(msg)
        if row.objective_stream_checksums != tuple(cast("Sequence[str]", mapping["objective_stream_checksums"])):
            msg = "Objective-stream checksum aliases do not match persisted requests."
            raise ValueError(msg)
        return row


@dataclass(frozen=True, slots=True, init=False)
class CompetitorStageExecution:
    """Immutable Adam/SPSA result with artifact-ready evidence."""

    stage: TrainingStageConfig
    optimizer_id: str
    optimizer_config: Mapping[str, object]
    circuit_binding_checksum: str
    circuit_binding_document: Mapping[str, object]
    provider_checksum: str | None
    checkpoint_validation_provider_checksum: str | None
    objective_checksum: str
    objective_binding: NoisyKrotovObjectiveBinding | None
    circuit_gate_count: int
    completed_iterations: int
    stop_reason: StopReason
    selected_global_iteration: int
    selected_checkpoint_validation_fidelity: float | None
    trace: tuple[CompetitorIterationRecord, ...]
    training_ensembles: tuple[KrotovFixedMapEnsemble, ...]
    checkpoint_validation_ensembles: tuple[KrotovFixedMapEnsemble, ...]
    training_ensemble_sampling_parameter_checksums: tuple[str, ...]
    checkpoint_validation_ensemble_sampling_parameter_checksums: tuple[str, ...]
    work: WP20WorkLedger
    optimizer_state: Mapping[str, object]
    initial_parameter_checksum: str
    final_parameter_checksum: str
    selected_parameter_checksum: str
    _initial_theta_bytes: bytes = field(repr=False)
    _final_theta_bytes: bytes = field(repr=False)
    _selected_theta_bytes: bytes = field(repr=False)
    schema_version: str = field(default=COMPETITOR_EXECUTION_SCHEMA_VERSION, init=False)

    def __init__(
        self,
        *,
        stage: TrainingStageConfig,
        optimizer_config: Mapping[str, object],
        circuit_binding: NoisyKrotovCircuitBinding,
        provider_checksum: str | None,
        checkpoint_validation_provider_checksum: str | None,
        objective_checksum: str,
        objective_binding: NoisyKrotovObjectiveBinding | None,
        initial_theta: NDArray[np.float64],
        final_theta: NDArray[np.float64],
        selected_theta: NDArray[np.float64],
        completed_iterations: int,
        stop_reason: StopReason,
        selected_global_iteration: int,
        selected_checkpoint_validation_fidelity: float | None,
        trace: Sequence[CompetitorIterationRecord],
        training_ensembles: Sequence[KrotovFixedMapEnsemble],
        checkpoint_validation_ensembles: Sequence[KrotovFixedMapEnsemble],
        training_ensemble_sampling_parameter_checksums: Sequence[str],
        checkpoint_validation_ensemble_sampling_parameter_checksums: Sequence[str],
        work: WP20WorkLedger,
        optimizer_state: Mapping[str, object],
    ) -> None:
        """Defensively snapshot complete optimizer evidence."""
        if not isinstance(stage, TrainingStageConfig):
            msg = "stage must be a TrainingStageConfig."
            raise TypeError(msg)
        optimizer_id = require_slug(stage.optimizer_id, "optimizer_id")
        if optimizer_id not in {"parameter_shift_adam", "spsa"}:
            msg = "CompetitorStageExecution supports only WP20 Adam and SPSA."
            raise ValueError(msg)
        initial = _validated_theta(initial_theta, stage.output_parameter_count, "initial_theta")
        final = _validated_theta(final_theta, stage.output_parameter_count, "final_theta")
        selected = _validated_theta(selected_theta, stage.output_parameter_count, "selected_theta")
        completed = require_int(completed_iterations, "completed_iterations")
        if completed > stage.iteration_budget:
            msg = "completed_iterations cannot exceed the stage budget."
            raise ValueError(msg)
        if stop_reason not in {"iteration_budget_reached", "work_budget_exhausted"}:
            msg = "stop_reason is not a WP20 optimizer terminal state."
            raise ValueError(msg)
        if (completed == stage.iteration_budget) != (stop_reason == "iteration_budget_reached"):
            msg = "stop_reason must distinguish completed and budget-exhausted executions."
            raise ValueError(msg)
        rows = tuple(trace)
        if len(rows) != completed + 1 or tuple(row.global_iteration for row in rows) != tuple(range(completed + 1)):
            msg = "trace must contain the initial state and every completed update contiguously."
            raise ValueError(msg)
        if rows[0].parameter_checksum != _vector_checksum(initial) or rows[-1].parameter_checksum != _vector_checksum(
            final
        ):
            msg = "Trace endpoints do not match the supplied parameters."
            raise ValueError(msg)
        selected_iteration = require_int(selected_global_iteration, "selected_global_iteration")
        matches = tuple(row for row in rows if row.global_iteration == selected_iteration)
        if len(matches) != 1 or matches[0].parameter_checksum != _vector_checksum(selected):
            msg = "Selected parameters are not bound to one trace row."
            raise ValueError(msg)
        fidelity = selected_checkpoint_validation_fidelity
        if stage.checkpoint_validation.enabled:
            if fidelity is None or matches[0].checkpoint_validation_fidelity != fidelity:
                msg = "Checkpoint-selected execution requires trace-backed validation fidelity."
                raise ValueError(msg)
        elif fidelity is not None or selected_iteration != completed or not np.array_equal(selected, final):
            msg = "Without checkpoint validation, the completed final state must be selected."
            raise ValueError(msg)
        if not isinstance(work, WP20WorkLedger) or rows[-1].cumulative_work != work:
            msg = "Final trace work must equal the execution work ledger."
            raise ValueError(msg)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "optimizer_id", optimizer_id)
        object.__setattr__(self, "optimizer_config", freeze_json_mapping(optimizer_config, "optimizer_config"))
        if not isinstance(circuit_binding, NoisyKrotovCircuitBinding):
            msg = "circuit_binding must be a NoisyKrotovCircuitBinding."
            raise TypeError(msg)
        if (
            circuit_binding.topology_id != stage.output_topology_id
            or circuit_binding.circuit.num_params != stage.output_parameter_count
        ):
            msg = "Circuit binding does not match the completed stage output."
            raise ValueError(msg)
        binding_document = freeze_json_mapping(circuit_binding.to_dict(), "circuit_binding_document")
        object.__setattr__(self, "circuit_binding_checksum", circuit_binding.content_checksum)
        object.__setattr__(self, "circuit_binding_document", binding_document)
        object.__setattr__(
            self,
            "provider_checksum",
            None if provider_checksum is None else require_checksum(provider_checksum, "provider_checksum"),
        )
        object.__setattr__(
            self,
            "checkpoint_validation_provider_checksum",
            (
                None
                if checkpoint_validation_provider_checksum is None
                else require_checksum(
                    checkpoint_validation_provider_checksum,
                    "checkpoint_validation_provider_checksum",
                )
            ),
        )
        objective = require_checksum(objective_checksum, "objective_checksum")
        if objective_binding is not None:
            if not isinstance(objective_binding, NoisyKrotovObjectiveBinding):
                msg = "objective_binding must be a NoisyKrotovObjectiveBinding or None."
                raise TypeError(msg)
            if objective_binding.objective_checksum != objective:
                msg = "objective_binding does not reproduce objective_checksum."
                raise ValueError(msg)
        object.__setattr__(self, "objective_checksum", objective)
        object.__setattr__(self, "objective_binding", objective_binding)
        gate_count = len(circuit_binding.circuit.gates)
        if gate_count < 1:
            msg = "Competitor circuits must contain at least one gate."
            raise ValueError(msg)
        object.__setattr__(self, "circuit_gate_count", gate_count)
        object.__setattr__(self, "completed_iterations", completed)
        object.__setattr__(self, "stop_reason", stop_reason)
        object.__setattr__(self, "selected_global_iteration", selected_iteration)
        object.__setattr__(self, "selected_checkpoint_validation_fidelity", fidelity)
        object.__setattr__(self, "trace", rows)
        training_maps = tuple(training_ensembles)
        validation_maps = tuple(checkpoint_validation_ensembles)
        if not all(isinstance(item, KrotovFixedMapEnsemble) for item in (*training_maps, *validation_maps)):
            msg = "Competitor ensemble evidence must contain KrotovFixedMapEnsemble values."
            raise TypeError(msg)
        if any(item.role != "training_trajectory" for item in training_maps) or any(
            item.role != "checkpoint_validation" for item in validation_maps
        ):
            msg = "Competitor ensembles are assigned to the wrong evidence role."
            raise ValueError(msg)
        if any(
            item.circuit_checksum != circuit_binding.content_checksum for item in (*training_maps, *validation_maps)
        ):
            msg = "Competitor ensembles do not bind the executed circuit."
            raise ValueError(msg)
        if any(item.gate_count != gate_count for item in (*training_maps, *validation_maps)):
            msg = "Competitor ensemble maps do not cover every gate in the executed circuit."
            raise ValueError(msg)
        if any(item.provider_checksum != provider_checksum for item in training_maps) or any(
            item.provider_checksum != checkpoint_validation_provider_checksum for item in validation_maps
        ):
            msg = "Competitor ensembles do not bind the declared providers."
            raise ValueError(msg)
        object.__setattr__(self, "training_ensembles", training_maps)
        object.__setattr__(self, "checkpoint_validation_ensembles", validation_maps)
        training_sampling_parameters = tuple(
            require_checksum(value, "training_ensemble_sampling_parameter_checksums")
            for value in training_ensemble_sampling_parameter_checksums
        )
        validation_sampling_parameters = tuple(
            require_checksum(value, "checkpoint_validation_ensemble_sampling_parameter_checksums")
            for value in checkpoint_validation_ensemble_sampling_parameter_checksums
        )
        if len(training_sampling_parameters) != len(training_maps) or len(validation_sampling_parameters) != len(
            validation_maps
        ):
            msg = "Every competitor map requires its exact sampling-parameter checksum."
            raise ValueError(msg)
        object.__setattr__(
            self,
            "training_ensemble_sampling_parameter_checksums",
            training_sampling_parameters,
        )
        object.__setattr__(
            self,
            "checkpoint_validation_ensemble_sampling_parameter_checksums",
            validation_sampling_parameters,
        )
        object.__setattr__(self, "work", work)
        object.__setattr__(self, "optimizer_state", freeze_json_mapping(optimizer_state, "optimizer_state"))
        object.__setattr__(self, "initial_parameter_checksum", _vector_checksum(initial))
        object.__setattr__(self, "final_parameter_checksum", _vector_checksum(final))
        object.__setattr__(self, "selected_parameter_checksum", _vector_checksum(selected))
        object.__setattr__(self, "_initial_theta_bytes", _vector_bytes(initial))
        object.__setattr__(self, "_final_theta_bytes", _vector_bytes(final))
        object.__setattr__(self, "_selected_theta_bytes", _vector_bytes(selected))
        object.__setattr__(self, "schema_version", COMPETITOR_EXECUTION_SCHEMA_VERSION)

    @property
    def initial_theta(self) -> NDArray[np.float64]:
        """Detached initial parameters."""
        return _copy_theta(self._initial_theta_bytes)

    @property
    def final_theta(self) -> NDArray[np.float64]:
        """Detached final parameters."""
        return _copy_theta(self._final_theta_bytes)

    @property
    def selected_theta(self) -> NDArray[np.float64]:
        """Detached checkpoint-selected parameters."""
        return _copy_theta(self._selected_theta_bytes)

    def _content_dict(self) -> dict[str, object]:
        """Return all checksum-covered execution evidence."""
        return {
            "schema_version": self.schema_version,
            "stage_configuration_checksum": self.stage.configuration_checksum,
            "optimizer_id": self.optimizer_id,
            "optimizer_config": thaw_json_mapping(self.optimizer_config),
            "circuit_binding_checksum": self.circuit_binding_checksum,
            "circuit_binding_document": thaw_json_mapping(self.circuit_binding_document),
            "provider_checksum": self.provider_checksum,
            "checkpoint_validation_provider_checksum": self.checkpoint_validation_provider_checksum,
            "objective_checksum": self.objective_checksum,
            "objective_binding_checksum": (
                None if self.objective_binding is None else self.objective_binding.content_checksum
            ),
            "circuit_gate_count": self.circuit_gate_count,
            "completed_iterations": self.completed_iterations,
            "stop_reason": self.stop_reason,
            "selected_global_iteration": self.selected_global_iteration,
            "selected_checkpoint_validation_fidelity": self.selected_checkpoint_validation_fidelity,
            "initial_parameter_checksum": self.initial_parameter_checksum,
            "final_parameter_checksum": self.final_parameter_checksum,
            "selected_parameter_checksum": self.selected_parameter_checksum,
            "trace": [row.to_dict() for row in self.trace],
            "training_ensemble_checksums": [item.content_checksum for item in self.training_ensembles],
            "checkpoint_validation_ensemble_checksums": [
                item.content_checksum for item in self.checkpoint_validation_ensembles
            ],
            "training_ensemble_sampling_parameter_checksums": list(self.training_ensemble_sampling_parameter_checksums),
            "checkpoint_validation_ensemble_sampling_parameter_checksums": list(
                self.checkpoint_validation_ensemble_sampling_parameter_checksums
            ),
            "work": self.work.to_dict(),
            "optimizer_state": thaw_json_mapping(self.optimizer_state),
        }

    @property
    def content_checksum(self) -> str:
        """Sealed complete optimizer execution checksum."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return the complete checksum-sealed execution document."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_stage_evidence(
        self,
        *,
        source_parameters: NDArray[np.float64] | None,
        circuit_statistics: Mapping[str, object],
    ) -> StageExecutionEvidence:
        """Translate a complete competitor execution to the WP18 artifact boundary."""
        if self.stop_reason != "iteration_budget_reached":
            msg = "A resource-truncated competitor execution is not a completed stage artifact."
            raise ValueError(msg)
        if self.objective_binding is None:
            msg = "Only a target-bound competitor objective can cross the artifact boundary."
            raise ValueError(msg)
        from .artifacts import StageExecutionEvidence  # noqa: PLC0415

        validation_rows = tuple(row for row in self.trace if row.checkpoint_validation_fidelity is not None)
        validation_summary = None
        if self.stage.checkpoint_validation.enabled:
            validation_summary = {
                "evaluation_count": len(validation_rows),
                "selected_iteration": self.selected_global_iteration,
                "selected_fidelity": self.selected_checkpoint_validation_fidelity,
                "request_checksums": [row.objective_request_checksums[-1] for row in validation_rows],
                "ensemble_checksums": [item.content_checksum for item in self.checkpoint_validation_ensembles],
            }
        training_summary = {
            "competitor_execution_checksum": self.content_checksum,
            "competitor_execution_document": self.to_dict(),
            "optimizer_id": self.optimizer_id,
            "completed_iterations": self.completed_iterations,
            "stop_reason": self.stop_reason,
            "final_monitoring_loss": self.trace[-1].monitoring_loss,
            "selected_iteration": self.selected_global_iteration,
            "selected_parameter_checksum": self.selected_parameter_checksum,
            "final_parameter_checksum": self.final_parameter_checksum,
            "training_ensemble_checksums": [item.content_checksum for item in self.training_ensembles],
        }
        return StageExecutionEvidence(
            stage=self.stage,
            source_parameters=source_parameters,
            initial_parameters=self.initial_theta,
            final_parameters=self.final_theta,
            selected_parameters=self.selected_theta,
            selected_global_iteration=self.selected_global_iteration,
            completed_global_iteration=self.completed_iterations,
            selected_checkpoint_validation_fidelity=self.selected_checkpoint_validation_fidelity,
            circuit_binding_checksum=self.circuit_binding_checksum,
            provider_checksum=self.provider_checksum,
            objective_checksum=self.objective_checksum,
            objective_binding=self.objective_binding,
            trace=tuple(row.to_dict() for row in self.trace),
            training_ensembles=self.training_ensembles,
            checkpoint_validation_ensembles=self.checkpoint_validation_ensembles,
            normalized_work=self.work.phase2_projection(),
            training_summary=training_summary,
            checkpoint_validation_summary=validation_summary,
            circuit_topology=self.circuit_binding_document,
            circuit_statistics=circuit_statistics,
            optimizer_state=self.optimizer_state,
        )


_COMPETITOR_EXECUTION_DOCUMENT_KEYS = frozenset({
    "schema_version",
    "stage_configuration_checksum",
    "optimizer_id",
    "optimizer_config",
    "circuit_binding_checksum",
    "circuit_binding_document",
    "provider_checksum",
    "checkpoint_validation_provider_checksum",
    "objective_checksum",
    "objective_binding_checksum",
    "circuit_gate_count",
    "completed_iterations",
    "stop_reason",
    "selected_global_iteration",
    "selected_checkpoint_validation_fidelity",
    "initial_parameter_checksum",
    "final_parameter_checksum",
    "selected_parameter_checksum",
    "trace",
    "training_ensemble_checksums",
    "checkpoint_validation_ensemble_checksums",
    "training_ensemble_sampling_parameter_checksums",
    "checkpoint_validation_ensemble_sampling_parameter_checksums",
    "work",
    "optimizer_state",
    "content_checksum",
})


def _expected_trace_requests(
    stage: TrainingStageConfig,
    optimizer_id: str,
) -> tuple[tuple[CompetitorObjectiveRequest, ...], ...]:
    """Reconstruct the only valid request ordering for a competitor trace."""
    rows: list[tuple[CompetitorObjectiveRequest, ...]] = []
    validation_call_index = 0
    initial = [
        _objective_request(
            stage,
            role="training",
            evaluation_kind="monitoring",
            global_iteration=0,
            pair_index=0,
            call_index=0,
        )
    ]
    if _should_validate(stage, 0):
        initial.append(
            _objective_request(
                stage,
                role="checkpoint_validation",
                evaluation_kind="checkpoint_validation",
                global_iteration=0,
                pair_index=0,
                call_index=validation_call_index,
            )
        )
        validation_call_index += 1
    rows.append(tuple(initial))
    training_call_index = 1
    for iteration in range(1, stage.iteration_budget + 1):
        requests: list[CompetitorObjectiveRequest] = []
        if optimizer_id == "parameter_shift_adam":
            for parameter_index in range(stage.output_parameter_count):
                requests.extend(
                    _objective_request(
                        stage,
                        role="training",
                        evaluation_kind=cast("ObjectiveEvaluationKind", kind),
                        global_iteration=iteration,
                        pair_index=parameter_index,
                        call_index=training_call_index,
                    )
                    for kind in ("gradient_plus", "gradient_minus")
                )
                training_call_index += 1
            requests.append(
                _objective_request(
                    stage,
                    role="training",
                    evaluation_kind="monitoring",
                    global_iteration=iteration,
                    pair_index=0,
                    call_index=training_call_index,
                )
            )
            training_call_index += 1
        else:
            requests.extend(
                _objective_request(
                    stage,
                    role="training",
                    evaluation_kind=cast("ObjectiveEvaluationKind", kind),
                    global_iteration=iteration,
                    pair_index=0,
                    call_index=iteration - 1,
                )
                for kind in ("gradient_plus", "gradient_minus", "monitoring")
            )
        if _should_validate(stage, iteration):
            requests.append(
                _objective_request(
                    stage,
                    role="checkpoint_validation",
                    evaluation_kind="checkpoint_validation",
                    global_iteration=iteration,
                    pair_index=0,
                    call_index=validation_call_index,
                )
            )
            validation_call_index += 1
        rows.append(tuple(requests))
    return tuple(rows)


def _expected_map_sampling_parameter_checksums(
    rows: Sequence[CompetitorIterationRecord],
) -> dict[ObjectiveRole, tuple[str, ...]]:
    """Derive each map-generation center from the mechanically replayed trace."""
    seen: set[tuple[ObjectiveRole, int]] = set()
    expected: dict[ObjectiveRole, list[str]] = {"training": [], "checkpoint_validation": []}
    for row_index, row in enumerate(rows):
        for request in row.objective_requests:
            key = (request.role, request.sampling_epoch)
            if request.trajectory_count == 0 or key in seen:
                continue
            seen.add(key)
            center = row
            if request.role == "training" and request.evaluation_kind in {"gradient_plus", "gradient_minus"}:
                if row_index == 0:
                    msg = "A gradient request cannot precede the initial optimizer state."
                    raise ValueError(msg)
                center = rows[row_index - 1]
            expected[request.role].append(center.parameter_checksum)
    return {role: tuple(values) for role, values in expected.items()}


def _validate_competitor_optimizer_math(
    *,
    stage: TrainingStageConfig,
    binding: NoisyKrotovCircuitBinding,
    rows: Sequence[CompetitorIterationRecord],
    optimizer_state: Mapping[str, object],
) -> None:
    """Recompute gradients, gains, update norms, and terminal optimizer state."""
    previous_parameters = np.asarray(rows[0].parameters, dtype=np.float64)
    if previous_parameters.shape != (stage.output_parameter_count,):
        msg = "Competitor trace parameter vectors do not match the resolved stage."
        raise ValueError(msg)
    if stage.optimizer_id == "parameter_shift_adam":
        config = ParameterShiftAdamConfig.from_stage(stage)
        scales = _parameter_shift_scales(binding.circuit)
        first = np.zeros(stage.output_parameter_count, dtype=np.float64)
        second = np.zeros(stage.output_parameter_count, dtype=np.float64)
        for iteration, row in enumerate(rows[1:], start=1):
            if (
                len(row.parameters) != stage.output_parameter_count
                or len(row.gradient) != stage.output_parameter_count
                or len(row.plus_losses) != stage.output_parameter_count
                or row.perturbation_scale is not None
            ):
                msg = "Adam trace dimensions or perturbation metadata do not match exact parameter shift."
                raise ValueError(msg)
            expected_gradient = np.asarray(
                [
                    0.5 * scale * (plus - minus)
                    for scale, plus, minus in zip(
                        scales,
                        row.plus_losses,
                        row.minus_losses,
                        strict=False,
                    )
                ],
                dtype=np.float64,
            )
            if not np.allclose(expected_gradient, row.gradient, rtol=1e-12, atol=1e-12):
                msg = "Adam trace gradient is not implied by its parameter-shift losses."
                raise ValueError(msg)
            first = config.beta1 * first + (1.0 - config.beta1) * expected_gradient
            second = config.beta2 * second + (1.0 - config.beta2) * np.square(expected_gradient)
            corrected_first = first / (1.0 - config.beta1**iteration)
            corrected_second = second / (1.0 - config.beta2**iteration)
            update = config.learning_rate * corrected_first / (np.sqrt(corrected_second) + config.epsilon)
            expected_parameters = previous_parameters - update
            if (
                row.learning_rate != config.learning_rate
                or not math.isclose(
                    row.update_norm,
                    float(np.linalg.norm(update)),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or not np.allclose(expected_parameters, row.parameters, rtol=1e-12, atol=1e-12)
            ):
                msg = "Adam trace update or resulting parameter vector is not mechanically reproduced."
                raise ValueError(msg)
            previous_parameters = np.asarray(row.parameters, dtype=np.float64)
        expected_state: Mapping[str, object] = {
            "first_moment": tuple(float(value) for value in first),
            "second_moment": tuple(float(value) for value in second),
            "completed_iterations": stage.iteration_budget,
        }
    else:
        config = SPSAConfig.from_stage(stage)
        for iteration, row in enumerate(rows[1:], start=1):
            if (
                len(row.parameters) != stage.output_parameter_count
                or len(row.gradient) != stage.output_parameter_count
                or len(row.plus_losses) != 1
            ):
                msg = "SPSA trace dimensions do not match one paired objective update."
                raise ValueError(msg)
            assert stage.optimizer_seed is not None
            perturbation_seed = _derive_seed({
                "derivation_version": "yaqs.state_preparation.phase2.spsa_perturbation.v1",
                "stage_configuration_checksum": stage.configuration_checksum,
                "optimizer_seed": stage.optimizer_seed,
                "iteration": iteration,
            })
            rng = np.random.Generator(np.random.PCG64(perturbation_seed))
            perturbation = 2.0 * rng.integers(0, 2, size=stage.output_parameter_count).astype(np.float64) - 1.0
            learning_rate, scale = config.gains(iteration)
            expected_gradient = ((row.plus_losses[0] - row.minus_losses[0]) / (2.0 * scale)) * perturbation
            if not np.allclose(expected_gradient, row.gradient, rtol=1e-12, atol=1e-12):
                msg = "SPSA trace gradient is not implied by its seeded perturbation and paired losses."
                raise ValueError(msg)
            expected_update = learning_rate * expected_gradient
            expected_parameters = previous_parameters - expected_update
            if (
                row.learning_rate != learning_rate
                or row.perturbation_scale != scale
                or not math.isclose(
                    row.update_norm,
                    float(np.linalg.norm(expected_update)),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or not np.allclose(expected_parameters, row.parameters, rtol=1e-12, atol=1e-12)
            ):
                msg = "SPSA trace gains, update norm, or resulting parameters are not mechanically reproduced."
                raise ValueError(msg)
            previous_parameters = np.asarray(row.parameters, dtype=np.float64)
        expected_state = {
            "perturbation_distribution": SPSA_PERTURBATION_DISTRIBUTION_ID,
            "completed_iterations": stage.iteration_budget,
        }
    if freeze_json_mapping(expected_state, "expected_optimizer_state") != freeze_json_mapping(
        optimizer_state,
        "optimizer_state",
    ):
        msg = "Competitor optimizer_state is not implied by the sealed trace."
        raise ValueError(msg)


def validate_competitor_stage_evidence(
    *,
    stage: TrainingStageConfig,
    execution_document: Mapping[str, object],
    circuit_binding_checksum: str,
    provider_checksum: str | None,
    checkpoint_validation_provider_checksum: str | None,
    objective_checksum: str,
    objective_binding_checksum: str,
    initial_parameter_checksum: str,
    final_parameter_checksum: str,
    selected_parameter_checksum: str,
    selected_global_iteration: int,
    completed_global_iteration: int,
    selected_checkpoint_validation_fidelity: float | None,
    trace: Sequence[Mapping[str, object]],
    training_ensembles: Sequence[KrotovFixedMapEnsemble],
    checkpoint_validation_ensembles: Sequence[KrotovFixedMapEnsemble],
    normalized_work: Mapping[str, object],
    optimizer_state: Mapping[str, object] | None,
    circuit_topology: Mapping[str, object],
    circuit_statistics: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, object] | None]:
    """Strictly reconstruct all aliases of one publishable competitor stage."""
    document = freeze_json_mapping(execution_document, "competitor_execution_document")
    require_exact_keys(document, _COMPETITOR_EXECUTION_DOCUMENT_KEYS, "competitor_execution_document")
    if document["schema_version"] != COMPETITOR_EXECUTION_SCHEMA_VERSION:
        msg = "Competitor execution document uses an unsupported schema version."
        raise ValueError(msg)
    content = {key: value for key, value in document.items() if key != "content_checksum"}
    supplied_checksum = require_checksum(document["content_checksum"], "competitor_execution_document.content_checksum")
    if canonical_checksum(content) != supplied_checksum:
        msg = "Competitor execution document checksum does not cover its complete content."
        raise ValueError(msg)
    if (
        document["stage_configuration_checksum"] != stage.configuration_checksum
        or document["optimizer_id"] != stage.optimizer_id
        or document["completed_iterations"] != completed_global_iteration
        or completed_global_iteration != stage.iteration_budget
        or document["stop_reason"] != "iteration_budget_reached"
    ):
        msg = "Competitor execution document does not identify a completed resolved stage."
        raise ValueError(msg)
    expected_config = (
        ParameterShiftAdamConfig.from_stage(stage).to_dict()
        if stage.optimizer_id == "parameter_shift_adam"
        else SPSAConfig.from_stage(stage).to_dict()
    )
    if document["optimizer_config"] != freeze_json_mapping(expected_config, "expected_optimizer_config"):
        msg = "Competitor optimizer configuration is not implied by the resolved stage."
        raise ValueError(msg)
    binding = decode_noisy_krotov_circuit_binding_document(document["circuit_binding_document"])
    if (
        document["circuit_binding_checksum"] != circuit_binding_checksum
        or binding.content_checksum != circuit_binding_checksum
        or freeze_json_mapping(binding.to_dict(), "expected_circuit_topology")
        != freeze_json_mapping(circuit_topology, "circuit_topology")
        or document["circuit_gate_count"] != len(binding.circuit.gates)
    ):
        msg = "Competitor execution does not reproduce its exact circuit binding."
        raise ValueError(msg)
    aliases = {
        "provider_checksum": provider_checksum,
        "checkpoint_validation_provider_checksum": checkpoint_validation_provider_checksum,
        "objective_checksum": objective_checksum,
        "objective_binding_checksum": objective_binding_checksum,
        "initial_parameter_checksum": initial_parameter_checksum,
        "final_parameter_checksum": final_parameter_checksum,
        "selected_parameter_checksum": selected_parameter_checksum,
        "selected_global_iteration": selected_global_iteration,
        "selected_checkpoint_validation_fidelity": selected_checkpoint_validation_fidelity,
    }
    if any(document[name] != value for name, value in aliases.items()):
        msg = "Competitor execution document aliases differ from stage evidence."
        raise ValueError(msg)
    rows_value = document["trace"]
    if isinstance(rows_value, (str, bytes)) or not isinstance(rows_value, Sequence):
        msg = "Competitor execution trace must be a sequence."
        raise TypeError(msg)
    rows = tuple(CompetitorIterationRecord.from_dict(item) for item in rows_value)
    supplied_trace = tuple(freeze_json_mapping(item, "trace row") for item in trace)
    if tuple(freeze_json_mapping(row.to_dict(), "decoded trace row") for row in rows) != supplied_trace:
        msg = "Competitor execution document trace differs from persisted trace evidence."
        raise ValueError(msg)
    if (
        len(rows) != stage.iteration_budget + 1
        or tuple(row.global_iteration for row in rows) != tuple(range(stage.iteration_budget + 1))
        or rows[0].parameter_checksum != initial_parameter_checksum
        or rows[-1].parameter_checksum != final_parameter_checksum
    ):
        msg = "Competitor trace endpoints or iteration sequence are invalid."
        raise ValueError(msg)
    selected_rows = tuple(row for row in rows if row.global_iteration == selected_global_iteration)
    if len(selected_rows) != 1 or selected_rows[0].parameter_checksum != selected_parameter_checksum:
        msg = "Competitor selected checkpoint is not bound to its trace row."
        raise ValueError(msg)
    expected_requests = _expected_trace_requests(stage, stage.optimizer_id)
    if tuple(row.objective_requests for row in rows) != expected_requests:
        msg = "Competitor objective requests do not reproduce the resolved random-stream schedule."
        raise ValueError(msg)
    validation_candidates = tuple(
        (row.checkpoint_validation_fidelity, row.global_iteration, row.parameter_checksum)
        for row in rows
        if row.checkpoint_validation_fidelity is not None
    )
    if stage.checkpoint_validation.enabled:
        if stage.checkpoint_validation.selection_rule == "last_iteration":
            winner = max(validation_candidates, key=operator.itemgetter(1))
        else:
            best = max(item[0] for item in validation_candidates)
            tied = tuple(item for item in validation_candidates if item[0] == best)
            winner = (
                min(tied, key=operator.itemgetter(1))
                if stage.checkpoint_validation.tie_breaker == "earliest_iteration"
                else max(tied, key=operator.itemgetter(1))
            )
        if winner != (
            selected_checkpoint_validation_fidelity,
            selected_global_iteration,
            selected_parameter_checksum,
        ):
            msg = "Competitor selected checkpoint does not implement the configured winner rule."
            raise ValueError(msg)
    elif validation_candidates:
        msg = "A stage without checkpoint validation cannot contain validation candidates."
        raise ValueError(msg)
    gate_count = len(binding.circuit.gates)
    expected_work = WP20WorkLedger()
    seen_windows: set[tuple[ObjectiveRole, int]] = set()
    for index, (row, requests) in enumerate(zip(rows, expected_requests, strict=False)):
        validate = _should_validate(stage, index)
        expected_work = _prospective_work(
            expected_work,
            training_evaluations=(
                1
                if index == 0
                else (2 * stage.output_parameter_count + 1 if stage.optimizer_id == "parameter_shift_adam" else 3)
            ),
            validation_evaluations=int(validate),
            stage=stage,
            gate_count=gate_count,
            gradient_calls=int(index > 0),
        )
        sampling_work = WP20WorkLedger()
        for request in requests:
            key = (request.role, request.sampling_epoch)
            if request.trajectory_count == 0 or key in seen_windows:
                continue
            seen_windows.add(key)
            sampling_work = _combine_work(
                sampling_work,
                WP20WorkLedger(
                    forward_circuit_evaluations=request.trajectory_count,
                    trajectory_gate_applications=request.trajectory_count * gate_count,
                    training_trajectories=request.trajectory_count if request.role == "training" else 0,
                    checkpoint_validation_trajectories=(
                        request.trajectory_count if request.role == "checkpoint_validation" else 0
                    ),
                ),
            )
        expected_work = _combine_work(expected_work, sampling_work)
        if row.cumulative_work != expected_work:
            msg = "Competitor cumulative work is not implied by objective calls and map sampling."
            raise ValueError(msg)
    document_work = WP20WorkLedger.from_dict(document["work"])
    if document_work != expected_work or dict(normalized_work) != expected_work.phase2_projection():
        msg = "Competitor detailed and normalized work ledgers do not agree."
        raise ValueError(msg)
    training_maps = tuple(training_ensembles)
    validation_maps = tuple(checkpoint_validation_ensembles)
    if document["training_ensemble_checksums"] != tuple(item.content_checksum for item in training_maps) or document[
        "checkpoint_validation_ensemble_checksums"
    ] != tuple(item.content_checksum for item in validation_maps):
        msg = "Competitor execution ensemble aliases differ from fixed-map evidence."
        raise ValueError(msg)
    expected_sampling_parameters = _expected_map_sampling_parameter_checksums(rows)
    if (
        document["training_ensemble_sampling_parameter_checksums"] != expected_sampling_parameters["training"]
        or document["checkpoint_validation_ensemble_sampling_parameter_checksums"]
        != expected_sampling_parameters["checkpoint_validation"]
    ):
        msg = "Competitor map-generation parameters are not the unperturbed optimizer/checkpoint states."
        raise ValueError(msg)
    expected_windows = {
        role: tuple(
            epoch
            for candidate_role, epoch in sorted(seen_windows, key=operator.itemgetter(0, 1))
            if candidate_role == role
        )
        for role in ("training", "checkpoint_validation")
    }
    if len(training_maps) != len(expected_windows["training"]) or len(validation_maps) != len(
        expected_windows["checkpoint_validation"]
    ):
        msg = "Competitor fixed-map counts do not match sampled objective windows."
        raise ValueError(msg)
    if any(item.gate_count != gate_count for item in (*training_maps, *validation_maps)):
        msg = "Competitor ensemble maps do not cover every gate in the decoded circuit."
        raise ValueError(msg)
    state = None if optimizer_state is None else freeze_json_mapping(optimizer_state, "optimizer_state")
    if state is None or document["optimizer_state"] != state:
        msg = "Competitor optimizer_state is missing or differs from the execution document."
        raise ValueError(msg)
    _validate_competitor_optimizer_math(stage=stage, binding=binding, rows=rows, optimizer_state=state)
    statistics = freeze_json_mapping(circuit_statistics, "circuit_statistics")
    resource_document = statistics.get("circuit_resource_metrics")
    if not isinstance(resource_document, Mapping):
        msg = "Publishable competitor evidence requires complete circuit_resource_metrics."
        raise TypeError(msg)
    resources = CircuitResourceMetrics.from_dict(resource_document)
    expected_resources = measure_circuit_resources(binding.circuit)
    if resources.content_checksum != expected_resources.content_checksum:
        msg = "Competitor circuit resources are not mechanically derived from its binding."
        raise ValueError(msg)
    training_summary: Mapping[str, object] = {
        "competitor_execution_checksum": supplied_checksum,
        "competitor_execution_document": document,
        "optimizer_id": stage.optimizer_id,
        "completed_iterations": stage.iteration_budget,
        "stop_reason": "iteration_budget_reached",
        "final_monitoring_loss": rows[-1].monitoring_loss,
        "selected_iteration": selected_global_iteration,
        "selected_parameter_checksum": selected_parameter_checksum,
        "final_parameter_checksum": final_parameter_checksum,
        "training_ensemble_checksums": tuple(item.content_checksum for item in training_maps),
    }
    validation_summary: Mapping[str, object] | None = None
    if stage.checkpoint_validation.enabled:
        validation_rows = tuple(row for row in rows if row.checkpoint_validation_fidelity is not None)
        validation_summary = {
            "evaluation_count": len(validation_rows),
            "selected_iteration": selected_global_iteration,
            "selected_fidelity": selected_checkpoint_validation_fidelity,
            "request_checksums": tuple(row.objective_request_checksums[-1] for row in validation_rows),
            "ensemble_checksums": tuple(item.content_checksum for item in validation_maps),
        }
    return training_summary, validation_summary


def _parameter_shift_scales(circuit: ParameterizedCircuit) -> tuple[float, ...]:
    """Return one exact Pauli-rotation angle scale for every parameter."""
    if not isinstance(circuit, ParameterizedCircuit):
        msg = "circuit must be a ParameterizedCircuit."
        raise TypeError(msg)
    occurrences: list[list[float]] = [[] for _ in range(circuit.num_params)]
    for gate in circuit.gates:
        if gate.param_index is None:
            continue
        if gate.name not in _PAULI_ROTATIONS:
            msg = f"Exact parameter shift does not support trainable gate {gate.name!r}."
            raise ValueError(msg)
        scale = require_float(gate.angle_scale, "angle_scale")
        if not scale:
            msg = "Trainable Pauli rotations require nonzero angle_scale."
            raise ValueError(msg)
        occurrences[gate.param_index].append(scale)
    if any(len(values) != 1 for values in occurrences):
        msg = "Exact two-evaluation parameter shift requires each parameter in exactly one Pauli rotation."
        raise ValueError(msg)
    return tuple(values[0] for values in occurrences)


def _selection(
    stage: TrainingStageConfig,
    candidates: Sequence[tuple[float, int, NDArray[np.float64]]],
    final_theta: NDArray[np.float64],
    completed_iterations: int,
) -> tuple[int, float | None, NDArray[np.float64]]:
    """Apply the frozen checkpoint-selection and tie-breaking policy."""
    if not stage.checkpoint_validation.enabled:
        return completed_iterations, None, final_theta.copy()
    if not candidates:
        msg = "Checkpoint validation enabled without any evaluated candidate."
        raise ValueError(msg)
    if stage.checkpoint_validation.selection_rule == "last_iteration":
        winner = max(candidates, key=operator.itemgetter(1))
    else:
        best = max(item[0] for item in candidates)
        tied = tuple(item for item in candidates if item[0] == best)
        winner = (
            min(tied, key=operator.itemgetter(1))
            if stage.checkpoint_validation.tie_breaker == "earliest_iteration"
            else max(tied, key=operator.itemgetter(1))
        )
    return winner[1], winner[0], winner[2].copy()


def _initial_trace(
    *,
    stage: TrainingStageConfig,
    theta: NDArray[np.float64],
    objective: CompetitorObjective,
    checkpoint_objective: CompetitorObjective | None,
    gate_count: int,
    budget: CompetitorWorkBudget | None,
) -> tuple[CompetitorIterationRecord, WP20WorkLedger, list[tuple[float, int, NDArray[np.float64]]]]:
    """Evaluate and account for the common iteration-zero state."""
    validate = _should_validate(stage, 0)
    monitoring_request = _objective_request(
        stage,
        role="training",
        evaluation_kind="monitoring",
        global_iteration=0,
        pair_index=0,
        call_index=0,
    )
    requests = [monitoring_request]
    validation_request = None
    if validate:
        if checkpoint_objective is None:
            msg = "checkpoint_objective is required by the checkpoint-validation policy."
            raise ValueError(msg)
        validation_request = _objective_request(
            stage,
            role="checkpoint_validation",
            evaluation_kind="checkpoint_validation",
            global_iteration=0,
            pair_index=0,
            call_index=0,
        )
        requests.append(validation_request)
    prospective = _prospective_work(
        WP20WorkLedger(),
        training_evaluations=1,
        validation_evaluations=int(validate),
        stage=stage,
        gate_count=gate_count,
    )
    prospective = _combine_work(
        prospective,
        _extra_sampling_work(objective, (monitoring_request,), gate_count),
        _extra_sampling_work(
            checkpoint_objective,
            () if validation_request is None else (validation_request,),
            gate_count,
        ),
    )
    if budget is not None and not budget.admits(prospective):
        msg = "Work budget cannot cover the mandatory initial monitoring and validation state."
        raise ValueError(msg)
    loss = _call_objective(objective, theta, monitoring_request)
    fidelity = None
    candidates: list[tuple[float, int, NDArray[np.float64]]] = []
    if validation_request is not None:
        assert checkpoint_objective is not None
        fidelity = 1.0 - _call_objective(checkpoint_objective, theta, validation_request)
        candidates.append((fidelity, 0, theta.copy()))
    row = CompetitorIterationRecord(
        global_iteration=0,
        parameters=tuple(float(value) for value in theta),
        parameter_checksum=_vector_checksum(theta),
        monitoring_loss=loss,
        checkpoint_validation_fidelity=fidelity,
        gradient=(),
        gradient_norm=0.0,
        update_norm=0.0,
        learning_rate=0.0,
        perturbation_scale=None,
        plus_losses=(),
        minus_losses=(),
        objective_requests=tuple(requests),
        cumulative_work=prospective,
    )
    return row, prospective, candidates


class ParameterShiftAdamStageAdapter:
    """Execute exact full-gradient parameter-shift Adam on one resolved stage."""

    def __init__(
        self,
        stage: TrainingStageConfig,
        circuit_binding: NoisyKrotovCircuitBinding,
        *,
        objective_checksum: str,
        provider_checksum: str | None,
        work_budget: CompetitorWorkBudget | None = None,
    ) -> None:
        """Validate and freeze the exact adapter inputs."""
        self.stage = stage
        self.config = ParameterShiftAdamConfig.from_stage(stage)
        if not isinstance(circuit_binding, NoisyKrotovCircuitBinding):
            msg = "circuit_binding must be a NoisyKrotovCircuitBinding."
            raise TypeError(msg)
        circuit = circuit_binding.circuit
        self.circuit = circuit
        self.circuit_binding = circuit_binding
        self.scales = _parameter_shift_scales(circuit)
        if (
            circuit.num_params != stage.output_parameter_count
            or circuit_binding.topology_id != stage.output_topology_id
        ):
            msg = "Circuit parameter count does not match the stage output."
            raise ValueError(msg)
        self.circuit_binding_checksum = circuit_binding.content_checksum
        self.objective_checksum = require_checksum(objective_checksum, "objective_checksum")
        if stage.trajectory_count and provider_checksum is None:
            msg = "A noisy Adam stage requires exact noise-provider provenance."
            raise ValueError(msg)
        if not stage.trajectory_count and provider_checksum is not None:
            msg = "A noiseless Adam stage cannot claim a noise provider."
            raise ValueError(msg)
        self.provider_checksum = (
            None if provider_checksum is None else require_checksum(provider_checksum, "provider_checksum")
        )
        self.circuit_gate_count = len(circuit.gates)
        if self.circuit_gate_count < 1:
            msg = "Competitor circuits must contain at least one gate."
            raise ValueError(msg)
        if work_budget is not None and not isinstance(work_budget, CompetitorWorkBudget):
            msg = "work_budget must be a CompetitorWorkBudget or None."
            raise TypeError(msg)
        self.work_budget = work_budget

    def execute(
        self,
        initial_theta: NDArray[np.float64],
        objective: CompetitorObjective,
        *,
        checkpoint_objective: CompetitorObjective | None = None,
    ) -> CompetitorStageExecution:
        """Run Adam until the iteration or prospective fixed-work cap is reached."""
        bound_objective = _bound_objective_for_execution(
            stage=self.stage,
            circuit_binding=self.circuit_binding,
            objective_checksum=self.objective_checksum,
            provider_checksum=self.provider_checksum,
            objective=objective,
            checkpoint_objective=checkpoint_objective,
        )
        theta = _validated_theta(initial_theta, self.circuit.num_params, "initial_theta")
        initial = theta.copy()
        initial_row, work, candidates = _initial_trace(
            stage=self.stage,
            theta=theta,
            objective=objective,
            checkpoint_objective=checkpoint_objective,
            gate_count=self.circuit_gate_count,
            budget=self.work_budget,
        )
        trace = [initial_row]
        first_moment = np.zeros_like(theta)
        second_moment = np.zeros_like(theta)
        completed = 0
        stop_reason: StopReason = "iteration_budget_reached"
        training_call_index = 1
        validation_call_index = int(initial_row.checkpoint_validation_fidelity is not None)
        for iteration in range(1, self.stage.iteration_budget + 1):
            validate = _should_validate(self.stage, iteration)
            prospective = _prospective_work(
                work,
                training_evaluations=2 * self.circuit.num_params + 1,
                validation_evaluations=int(validate),
                stage=self.stage,
                gate_count=self.circuit_gate_count,
                gradient_calls=1,
            )
            if self.work_budget is not None and not self.work_budget.admits(prospective):
                stop_reason = "work_budget_exhausted"
                break
            gradient = np.empty_like(theta)
            plus_losses: list[float] = []
            minus_losses: list[float] = []
            requests: list[CompetitorObjectiveRequest] = []
            for parameter_index, scale in enumerate(self.scales):
                request_plus = _objective_request(
                    self.stage,
                    role="training",
                    evaluation_kind="gradient_plus",
                    global_iteration=iteration,
                    pair_index=parameter_index,
                    call_index=training_call_index,
                )
                request_minus = _objective_request(
                    self.stage,
                    role="training",
                    evaluation_kind="gradient_minus",
                    global_iteration=iteration,
                    pair_index=parameter_index,
                    call_index=training_call_index,
                )
                shift = math.pi / (2.0 * scale)
                plus_theta = theta.copy()
                minus_theta = theta.copy()
                plus_theta[parameter_index] += shift
                minus_theta[parameter_index] -= shift
                plus = _call_objective(objective, plus_theta, request_plus)
                minus = _call_objective(objective, minus_theta, request_minus)
                gradient[parameter_index] = 0.5 * scale * (plus - minus)
                plus_losses.append(plus)
                minus_losses.append(minus)
                requests.extend((request_plus, request_minus))
                training_call_index += 1
            first_moment = self.config.beta1 * first_moment + (1.0 - self.config.beta1) * gradient
            second_moment = self.config.beta2 * second_moment + (1.0 - self.config.beta2) * np.square(gradient)
            corrected_first = first_moment / (1.0 - self.config.beta1**iteration)
            corrected_second = second_moment / (1.0 - self.config.beta2**iteration)
            update = self.config.learning_rate * corrected_first / (np.sqrt(corrected_second) + self.config.epsilon)
            theta -= update
            monitoring_request = _objective_request(
                self.stage,
                role="training",
                evaluation_kind="monitoring",
                global_iteration=iteration,
                pair_index=0,
                call_index=training_call_index,
            )
            monitoring_loss = _call_objective(objective, theta, monitoring_request)
            training_call_index += 1
            requests.append(monitoring_request)
            fidelity = None
            if validate:
                if checkpoint_objective is None:
                    msg = "checkpoint_objective is required by the checkpoint-validation policy."
                    raise ValueError(msg)
                validation_request = _objective_request(
                    self.stage,
                    role="checkpoint_validation",
                    evaluation_kind="checkpoint_validation",
                    global_iteration=iteration,
                    pair_index=0,
                    call_index=validation_call_index,
                )
                fidelity = 1.0 - _call_objective(checkpoint_objective, theta, validation_request)
                candidates.append((fidelity, iteration, theta.copy()))
                requests.append(validation_request)
                validation_call_index += 1
            work = prospective
            completed = iteration
            trace.append(
                CompetitorIterationRecord(
                    global_iteration=iteration,
                    parameters=tuple(float(value) for value in theta),
                    parameter_checksum=_vector_checksum(theta),
                    monitoring_loss=monitoring_loss,
                    checkpoint_validation_fidelity=fidelity,
                    gradient=tuple(float(value) for value in gradient),
                    gradient_norm=float(np.linalg.norm(gradient)),
                    update_norm=float(np.linalg.norm(update)),
                    learning_rate=self.config.learning_rate,
                    perturbation_scale=None,
                    plus_losses=tuple(plus_losses),
                    minus_losses=tuple(minus_losses),
                    objective_requests=tuple(requests),
                    cumulative_work=work,
                )
            )
        selected_iteration, selected_fidelity, selected = _selection(
            self.stage,
            candidates,
            theta,
            completed,
        )
        return CompetitorStageExecution(
            stage=self.stage,
            optimizer_config=self.config.to_dict(),
            circuit_binding=self.circuit_binding,
            provider_checksum=self.provider_checksum,
            checkpoint_validation_provider_checksum=(
                None if bound_objective is None else bound_objective.checkpoint_validation_provider_checksum
            ),
            objective_checksum=self.objective_checksum,
            objective_binding=None if bound_objective is None else bound_objective.objective_binding,
            initial_theta=initial,
            final_theta=theta,
            selected_theta=selected,
            completed_iterations=completed,
            stop_reason=stop_reason,
            selected_global_iteration=selected_iteration,
            selected_checkpoint_validation_fidelity=selected_fidelity,
            trace=trace,
            training_ensembles=() if bound_objective is None else bound_objective.training_ensembles,
            checkpoint_validation_ensembles=(
                () if bound_objective is None else bound_objective.checkpoint_validation_ensembles
            ),
            training_ensemble_sampling_parameter_checksums=(
                () if bound_objective is None else bound_objective.training_ensemble_sampling_parameter_checksums
            ),
            checkpoint_validation_ensemble_sampling_parameter_checksums=(
                ()
                if bound_objective is None
                else bound_objective.checkpoint_validation_ensemble_sampling_parameter_checksums
            ),
            work=work,
            optimizer_state={
                "first_moment": first_moment.tolist(),
                "second_moment": second_moment.tolist(),
                "completed_iterations": completed,
            },
        )


class SPSAStageAdapter:
    """Execute fresh-objective two-evaluation SPSA on one resolved stage."""

    def __init__(
        self,
        stage: TrainingStageConfig,
        circuit_binding: NoisyKrotovCircuitBinding,
        *,
        objective_checksum: str,
        provider_checksum: str | None,
        work_budget: CompetitorWorkBudget | None = None,
    ) -> None:
        """Validate and freeze the exact adapter inputs."""
        self.stage = stage
        self.config = SPSAConfig.from_stage(stage)
        if not isinstance(circuit_binding, NoisyKrotovCircuitBinding):
            msg = "circuit_binding must be a NoisyKrotovCircuitBinding."
            raise TypeError(msg)
        circuit = circuit_binding.circuit
        if (
            circuit.num_params != stage.output_parameter_count
            or circuit_binding.topology_id != stage.output_topology_id
        ):
            msg = "Circuit parameter count does not match the stage output."
            raise ValueError(msg)
        self.circuit = circuit
        self.circuit_binding = circuit_binding
        self.circuit_binding_checksum = circuit_binding.content_checksum
        self.objective_checksum = require_checksum(objective_checksum, "objective_checksum")
        if stage.trajectory_count and provider_checksum is None:
            msg = "A noisy SPSA stage requires exact noise-provider provenance."
            raise ValueError(msg)
        if not stage.trajectory_count and provider_checksum is not None:
            msg = "A noiseless SPSA stage cannot claim a noise provider."
            raise ValueError(msg)
        self.provider_checksum = (
            None if provider_checksum is None else require_checksum(provider_checksum, "provider_checksum")
        )
        self.circuit_gate_count = len(circuit.gates)
        if self.circuit_gate_count < 1:
            msg = "Competitor circuits must contain at least one gate."
            raise ValueError(msg)
        if work_budget is not None and not isinstance(work_budget, CompetitorWorkBudget):
            msg = "work_budget must be a CompetitorWorkBudget or None."
            raise TypeError(msg)
        self.work_budget = work_budget

    def execute(
        self,
        initial_theta: NDArray[np.float64],
        objective: CompetitorObjective,
        *,
        checkpoint_objective: CompetitorObjective | None = None,
    ) -> CompetitorStageExecution:
        """Run SPSA with one fresh paired perturbation stream per update."""
        bound_objective = _bound_objective_for_execution(
            stage=self.stage,
            circuit_binding=self.circuit_binding,
            objective_checksum=self.objective_checksum,
            provider_checksum=self.provider_checksum,
            objective=objective,
            checkpoint_objective=checkpoint_objective,
        )
        theta = _validated_theta(initial_theta, self.circuit.num_params, "initial_theta")
        initial = theta.copy()
        initial_row, work, candidates = _initial_trace(
            stage=self.stage,
            theta=theta,
            objective=objective,
            checkpoint_objective=checkpoint_objective,
            gate_count=self.circuit_gate_count,
            budget=self.work_budget,
        )
        trace = [initial_row]
        completed = 0
        stop_reason: StopReason = "iteration_budget_reached"
        validation_call_index = int(initial_row.checkpoint_validation_fidelity is not None)
        for iteration in range(1, self.stage.iteration_budget + 1):
            validate = _should_validate(self.stage, iteration)
            request_plus = _objective_request(
                self.stage,
                role="training",
                evaluation_kind="gradient_plus",
                global_iteration=iteration,
                pair_index=0,
                call_index=iteration - 1,
            )
            request_minus = _objective_request(
                self.stage,
                role="training",
                evaluation_kind="gradient_minus",
                global_iteration=iteration,
                pair_index=0,
                call_index=iteration - 1,
            )
            monitoring_request = _objective_request(
                self.stage,
                role="training",
                evaluation_kind="monitoring",
                global_iteration=iteration,
                pair_index=0,
                call_index=iteration - 1,
            )
            requests = [request_plus, request_minus, monitoring_request]
            validation_request = None
            if validate:
                if checkpoint_objective is None:
                    msg = "checkpoint_objective is required by the checkpoint-validation policy."
                    raise ValueError(msg)
                validation_request = _objective_request(
                    self.stage,
                    role="checkpoint_validation",
                    evaluation_kind="checkpoint_validation",
                    global_iteration=iteration,
                    pair_index=0,
                    call_index=validation_call_index,
                )
                requests.append(validation_request)
            prospective = _prospective_work(
                work,
                training_evaluations=3,
                validation_evaluations=int(validate),
                stage=self.stage,
                gate_count=self.circuit_gate_count,
                gradient_calls=1,
            )
            prospective = _combine_work(
                prospective,
                _extra_sampling_work(
                    objective, (request_plus, request_minus, monitoring_request), self.circuit_gate_count
                ),
                _extra_sampling_work(
                    checkpoint_objective,
                    () if validation_request is None else (validation_request,),
                    self.circuit_gate_count,
                ),
            )
            if self.work_budget is not None and not self.work_budget.admits(prospective):
                stop_reason = "work_budget_exhausted"
                break
            assert self.stage.optimizer_seed is not None
            perturbation_seed = _derive_seed({
                "derivation_version": "yaqs.state_preparation.phase2.spsa_perturbation.v1",
                "stage_configuration_checksum": self.stage.configuration_checksum,
                "optimizer_seed": self.stage.optimizer_seed,
                "iteration": iteration,
            })
            rng = np.random.Generator(np.random.PCG64(perturbation_seed))
            perturbation = 2.0 * rng.integers(0, 2, size=theta.size).astype(np.float64) - 1.0
            learning_rate, perturbation_scale = self.config.gains(iteration)
            if bound_objective is not None:
                bound_objective.prime_sampling_window(theta, request_plus)
            plus = _call_objective(objective, theta + perturbation_scale * perturbation, request_plus)
            minus = _call_objective(objective, theta - perturbation_scale * perturbation, request_minus)
            gradient = ((plus - minus) / (2.0 * perturbation_scale)) * perturbation
            update = learning_rate * gradient
            theta -= update
            monitoring_loss = _call_objective(objective, theta, monitoring_request)
            fidelity = None
            if validation_request is not None:
                assert checkpoint_objective is not None
                fidelity = 1.0 - _call_objective(checkpoint_objective, theta, validation_request)
                candidates.append((fidelity, iteration, theta.copy()))
                validation_call_index += 1
            work = prospective
            completed = iteration
            trace.append(
                CompetitorIterationRecord(
                    global_iteration=iteration,
                    parameters=tuple(float(value) for value in theta),
                    parameter_checksum=_vector_checksum(theta),
                    monitoring_loss=monitoring_loss,
                    checkpoint_validation_fidelity=fidelity,
                    gradient=tuple(float(value) for value in gradient),
                    gradient_norm=float(np.linalg.norm(gradient)),
                    update_norm=float(np.linalg.norm(update)),
                    learning_rate=learning_rate,
                    perturbation_scale=perturbation_scale,
                    plus_losses=(plus,),
                    minus_losses=(minus,),
                    objective_requests=tuple(requests),
                    cumulative_work=work,
                )
            )
        selected_iteration, selected_fidelity, selected = _selection(
            self.stage,
            candidates,
            theta,
            completed,
        )
        return CompetitorStageExecution(
            stage=self.stage,
            optimizer_config=self.config.to_dict(),
            circuit_binding=self.circuit_binding,
            provider_checksum=self.provider_checksum,
            checkpoint_validation_provider_checksum=(
                None if bound_objective is None else bound_objective.checkpoint_validation_provider_checksum
            ),
            objective_checksum=self.objective_checksum,
            objective_binding=None if bound_objective is None else bound_objective.objective_binding,
            initial_theta=initial,
            final_theta=theta,
            selected_theta=selected,
            completed_iterations=completed,
            stop_reason=stop_reason,
            selected_global_iteration=selected_iteration,
            selected_checkpoint_validation_fidelity=selected_fidelity,
            trace=trace,
            training_ensembles=() if bound_objective is None else bound_objective.training_ensembles,
            checkpoint_validation_ensembles=(
                () if bound_objective is None else bound_objective.checkpoint_validation_ensembles
            ),
            training_ensemble_sampling_parameter_checksums=(
                () if bound_objective is None else bound_objective.training_ensemble_sampling_parameter_checksums
            ),
            checkpoint_validation_ensemble_sampling_parameter_checksums=(
                ()
                if bound_objective is None
                else bound_objective.checkpoint_validation_ensemble_sampling_parameter_checksums
            ),
            work=work,
            optimizer_state={
                "perturbation_distribution": SPSA_PERTURBATION_DISTRIBUTION_ID,
                "completed_iterations": completed,
            },
        )


@dataclass(frozen=True, slots=True)
class BMPDCompetitorStageRunner:
    """Execute a complete target-bound Adam or SPSA BMPD pipeline."""

    pipeline: TrainingPipelineConfig
    target: MaterializedTarget | LegacyMaterializedTarget
    work_budget: CompetitorWorkBudget | None = None

    def __post_init__(self) -> None:
        """Validate the method, target ledger identity, and optional budget."""
        if not isinstance(self.pipeline, TrainingPipelineConfig):
            msg = "pipeline must be a TrainingPipelineConfig."
            raise TypeError(msg)
        allowed = {
            PARAMETER_SHIFT_ADAM_LAYERWISE_METHOD_ID,
            PARAMETER_SHIFT_ADAM_FIXED_METHOD_ID,
            SPSA_LAYERWISE_METHOD_ID,
            SPSA_FIXED_METHOD_ID,
        }
        if self.pipeline.method_id not in allowed:
            msg = "BMPDCompetitorStageRunner accepts only registered WP20 Adam/SPSA methods."
            raise ValueError(msg)
        if not isinstance(self.target, (MaterializedTarget, LegacyMaterializedTarget)):
            msg = "A competitor runner requires an authorized materialized target."
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
            msg = "Materialized target identity does not match the resolved competitor pipeline."
            raise ValueError(msg)
        if self.work_budget is not None and not isinstance(self.work_budget, CompetitorWorkBudget):
            msg = "work_budget must be a CompetitorWorkBudget or None."
            raise TypeError(msg)

    def _binding(self, stage: TrainingStageConfig) -> NoisyKrotovCircuitBinding:
        """Materialize and verify the BMPD binding named by one resolved stage."""
        match = _BMPD_TOPOLOGY_PATTERN.fullmatch(stage.output_topology_id)
        if match is None:
            msg = "Competitor stage output_topology_id is not a BMPD topology."
            raise ValueError(msg)
        qubits = int(match.group("qubits"))
        depth = int(match.group("depth"))
        if qubits != self.pipeline.qubit_count:
            msg = "Competitor stage topology width differs from its pipeline."
            raise ValueError(msg)
        from .layerwise_bmpd import create_bmpd_circuit_binding  # noqa: PLC0415

        binding = create_bmpd_circuit_binding(qubits, depth)
        if binding.circuit.num_params != stage.output_parameter_count:
            msg = "Competitor BMPD binding parameter count differs from the resolved stage."
            raise ValueError(msg)
        return binding

    def __call__(
        self,
        stage: TrainingStageConfig,
        predecessor_parameters: NDArray[np.float64] | None,
    ) -> StageExecutionEvidence:
        """Initialize, optimize, and translate one exact pipeline stage."""
        if stage.stage_index >= len(self.pipeline.stages) or stage != self.pipeline.stages[stage.stage_index]:
            msg = "Stage does not belong to this resolved competitor pipeline."
            raise ValueError(msg)
        from .layerwise_bmpd import initialize_layerwise_stage_parameters  # noqa: PLC0415

        binding = self._binding(stage)
        initial_theta = initialize_layerwise_stage_parameters(stage, predecessor_parameters)
        objective = FixedRateNoisyCompetitorObjective(stage, binding, self.target)
        adapter_type = (
            ParameterShiftAdamStageAdapter if stage.optimizer_id == "parameter_shift_adam" else SPSAStageAdapter
        )
        adapter = adapter_type(
            stage,
            binding,
            objective_checksum=objective.objective_checksum,
            provider_checksum=objective.provider_checksum,
            work_budget=self.work_budget,
        )
        execution = adapter.execute(
            initial_theta,
            objective,
            checkpoint_objective=objective if stage.checkpoint_validation.enabled else None,
        )
        return execution.to_stage_evidence(
            source_parameters=predecessor_parameters,
            circuit_statistics=self.circuit_statistics(stage),
        )

    def circuit_statistics(self, stage: TrainingStageConfig) -> Mapping[str, object]:
        """Return complete logical/native resource evidence for one stage."""
        if stage.stage_index >= len(self.pipeline.stages) or stage != self.pipeline.stages[stage.stage_index]:
            msg = "Stage does not belong to this resolved competitor pipeline."
            raise ValueError(msg)
        binding = self._binding(stage)
        resources = measure_circuit_resources(binding.circuit)
        match = _BMPD_TOPOLOGY_PATTERN.fullmatch(stage.output_topology_id)
        assert match is not None
        return {
            "topology_id": stage.output_topology_id,
            "parameter_count": stage.output_parameter_count,
            "qubit_count": self.pipeline.qubit_count,
            "bmpd_depth": int(match.group("depth")),
            "logical_gate_count": len(binding.circuit.gates),
            "logical_two_qubit_gate_count": resources.logical_two_qubit_gates,
            "native_two_qubit_gate_count": resources.native_two_qubit_gates,
            "native_two_qubit_gates_per_chain_edge": list(resources.native_two_qubit_gates_per_chain_edge),
            "circuit_resource_metrics": resources.to_dict(),
        }


def _rebind_competitor_stages(
    stages: Sequence[TrainingStageTemplate],
    *,
    binding_prefix: str,
) -> tuple[TrainingStageTemplate, ...]:
    """Clone stage templates onto method-specific random-stream bindings."""
    result: list[TrainingStageTemplate] = []
    for raw_stage in stages:
        bindings = {
            role: None if value is None else f"{binding_prefix}_{raw_stage.stage_id}_{role}"
            for role, value in raw_stage.seed_bindings.items()
        }
        result.append(TrainingStageTemplate(stage_policy=dict(raw_stage.stage_policy), seed_bindings=bindings))
    return tuple(result)


def _competitor_pipeline_template(
    reference: TrainingPipelineTemplate,
    *,
    optimizer_id: Literal["parameter_shift_adam", "spsa"],
    method_id: str,
    template_id: str,
    binding_prefix: str,
    optimizer_hyperparameters: Mapping[str, object],
) -> TrainingPipelineTemplate:
    """Replace every optimizer stage while retaining the exact ansatz schedule."""
    if not isinstance(reference, TrainingPipelineTemplate):
        msg = "reference must be a TrainingPipelineTemplate."
        raise TypeError(msg)
    stages: list[TrainingStageTemplate] = []
    for rebound in _rebind_competitor_stages(reference.stages, binding_prefix=binding_prefix):
        policy = dict(rebound.stage_policy)
        previous_hyperparameters = cast("Mapping[str, object]", policy["optimizer_hyperparameters"])
        initialization = {
            name: previous_hyperparameters[name]
            for name in _INITIALIZATION_HYPERPARAMETERS
            if name in previous_hyperparameters
        }
        trajectory_count = cast("int", policy["trajectory_count"])
        sampling_policy = (
            "none"
            if trajectory_count == 0
            else ("crn_fixed" if optimizer_id == "parameter_shift_adam" else "resampled")
        )
        policy["optimizer_id"] = optimizer_id
        policy["optimizer_hyperparameters"] = {
            **optimizer_hyperparameters,
            "gradient_trajectory_count": trajectory_count,
            "sampling_policy": sampling_policy,
            **initialization,
        }
        policy["trajectory_update"] = None if trajectory_count == 0 else "independent"
        policy["sampling_policy"] = sampling_policy
        policy["crn_refresh_interval"] = None
        stages.append(TrainingStageTemplate(stage_policy=policy, seed_bindings=dict(rebound.seed_bindings)))
    return TrainingPipelineTemplate(
        template_id=template_id,
        preregistration_checksum=reference.preregistration_checksum,
        target_scope_id=reference.target_scope_id,
        ansatz_family=reference.ansatz_family,
        method_id=method_id,
        method_version="1",
        resource_stratum_id=reference.resource_stratum_id,
        stages=tuple(stages),
        seed_domains=reference.seed_domains,
        final_materialization_policy=reference.final_materialization_policy,
    )


def _adam_hyperparameters(
    trajectory_count: int,
    *,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
) -> dict[str, object]:
    """Return the exact template-level Adam policy after scalar validation."""
    config = ParameterShiftAdamConfig(
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
    )
    return {
        "learning_rate": config.learning_rate,
        "beta1": config.beta1,
        "beta2": config.beta2,
        "epsilon": config.epsilon,
        "parameter_shift": PARAMETER_SHIFT_POLICY_ID,
        "gradient_trajectory_count": require_int(trajectory_count, "trajectory_count"),
        "sampling_policy": "crn_fixed",
    }


def _spsa_hyperparameters(
    trajectory_count: int,
    *,
    a: float,
    stability_constant: float,
    alpha: float,
    c: float,
    gamma: float,
) -> dict[str, object]:
    """Return the exact template-level SPSA policy after scalar validation."""
    config = SPSAConfig(
        a=a,
        stability_constant=stability_constant,
        alpha=alpha,
        c=c,
        gamma=gamma,
    )
    return {
        "a": config.a,
        "A": config.stability_constant,
        "alpha": config.alpha,
        "c": config.c,
        "gamma": config.gamma,
        "perturbation_distribution": SPSA_PERTURBATION_DISTRIBUTION_ID,
        "gradient_trajectory_count": require_int(trajectory_count, "trajectory_count"),
        "sampling_policy": "resampled",
    }


def build_parameter_shift_adam_layerwise_template(
    *,
    training_trajectory_count: int,
    checkpoint_validation_trajectory_count: int,
    qubit_count: int = 6,
    learning_rate: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> TrainingPipelineTemplate:
    """Build the q6 or q12 layerwise parameter-shift Adam pipeline."""
    from .layerwise_bmpd import build_layerwise_bmpd_crn_v2_template  # noqa: PLC0415

    reference = build_layerwise_bmpd_crn_v2_template(
        training_trajectory_count=training_trajectory_count,
        checkpoint_validation_trajectory_count=checkpoint_validation_trajectory_count,
        qubit_count=qubit_count,
    )
    q12 = qubit_count == 12
    return _competitor_pipeline_template(
        reference,
        optimizer_id="parameter_shift_adam",
        method_id=PARAMETER_SHIFT_ADAM_LAYERWISE_METHOD_ID,
        template_id=(
            "parameter_shift_adam_layerwise_default_q12_projection" if q12 else "parameter_shift_adam_layerwise_default"
        ),
        binding_prefix="parameter_shift_adam_layerwise",
        optimizer_hyperparameters=_adam_hyperparameters(
            training_trajectory_count,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
        ),
    )


def build_spsa_layerwise_template(
    *,
    training_trajectory_count: int,
    checkpoint_validation_trajectory_count: int,
    qubit_count: int = 6,
    a: float = 0.1,
    stability_constant: float = 10.0,
    alpha: float = 0.602,
    c: float = 0.1,
    gamma: float = 0.101,
) -> TrainingPipelineTemplate:
    """Build the q6 or q12 layerwise fresh-objective SPSA pipeline."""
    from .layerwise_bmpd import build_layerwise_bmpd_crn_v2_template  # noqa: PLC0415

    reference = build_layerwise_bmpd_crn_v2_template(
        training_trajectory_count=training_trajectory_count,
        checkpoint_validation_trajectory_count=checkpoint_validation_trajectory_count,
        qubit_count=qubit_count,
    )
    q12 = qubit_count == 12
    return _competitor_pipeline_template(
        reference,
        optimizer_id="spsa",
        method_id=SPSA_LAYERWISE_METHOD_ID,
        template_id="spsa_layerwise_default_q12_projection" if q12 else "spsa_layerwise_default",
        binding_prefix="spsa_layerwise",
        optimizer_hyperparameters=_spsa_hyperparameters(
            training_trajectory_count,
            a=a,
            stability_constant=stability_constant,
            alpha=alpha,
            c=c,
            gamma=gamma,
        ),
    )


def build_parameter_shift_adam_fixed_template(
    *,
    iteration_budget: int,
    training_trajectory_count: int,
    checkpoint_validation_trajectory_count: int,
    qubit_count: int = 6,
    learning_rate: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> TrainingPipelineTemplate:
    """Build the q6 or q12 exploratory fixed-depth parameter-shift Adam pipeline."""
    from .fair_controls import build_fixed_depth_bmpd_crn_template  # noqa: PLC0415

    reference = build_fixed_depth_bmpd_crn_template(
        iteration_budget=iteration_budget,
        training_trajectory_count=training_trajectory_count,
        checkpoint_validation_trajectory_count=checkpoint_validation_trajectory_count,
        qubit_count=qubit_count,
    )
    q12 = qubit_count == 12
    return _competitor_pipeline_template(
        reference,
        optimizer_id="parameter_shift_adam",
        method_id=PARAMETER_SHIFT_ADAM_FIXED_METHOD_ID,
        template_id=(
            f"parameter_shift_adam_fixed_b{iteration_budget}_q12_projection"
            if q12
            else f"parameter_shift_adam_fixed_b{iteration_budget}"
        ),
        binding_prefix="parameter_shift_adam_fixed",
        optimizer_hyperparameters=_adam_hyperparameters(
            training_trajectory_count,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
        ),
    )


def build_spsa_fixed_template(
    *,
    iteration_budget: int,
    training_trajectory_count: int,
    checkpoint_validation_trajectory_count: int,
    qubit_count: int = 6,
    a: float = 0.1,
    stability_constant: float = 10.0,
    alpha: float = 0.602,
    c: float = 0.1,
    gamma: float = 0.101,
) -> TrainingPipelineTemplate:
    """Build the q6 or q12 exploratory fixed-depth fresh-objective SPSA pipeline."""
    from .fair_controls import build_fixed_depth_bmpd_crn_template  # noqa: PLC0415

    reference = build_fixed_depth_bmpd_crn_template(
        iteration_budget=iteration_budget,
        training_trajectory_count=training_trajectory_count,
        checkpoint_validation_trajectory_count=checkpoint_validation_trajectory_count,
        qubit_count=qubit_count,
    )
    q12 = qubit_count == 12
    return _competitor_pipeline_template(
        reference,
        optimizer_id="spsa",
        method_id=SPSA_FIXED_METHOD_ID,
        template_id=(f"spsa_fixed_b{iteration_budget}_q12_projection" if q12 else f"spsa_fixed_b{iteration_budget}"),
        binding_prefix="spsa_fixed",
        optimizer_hyperparameters=_spsa_hyperparameters(
            training_trajectory_count,
            a=a,
            stability_constant=stability_constant,
            alpha=alpha,
            c=c,
            gamma=gamma,
        ),
    )


__all__ = [
    "COMPETITOR_EXECUTION_SCHEMA_VERSION",
    "COMPETITOR_ITERATION_SCHEMA_VERSION",
    "COMPETITOR_OBJECTIVE_REQUEST_SCHEMA_VERSION",
    "COMPETITOR_WORK_BUDGET_SCHEMA_VERSION",
    "FIXED_RATE_COMPETITOR_OBJECTIVE_SCHEMA_VERSION",
    "PARAMETER_SHIFT_ADAM_CONFIG_SCHEMA_VERSION",
    "PARAMETER_SHIFT_ADAM_FIXED_METHOD_ID",
    "PARAMETER_SHIFT_ADAM_LAYERWISE_METHOD_ID",
    "PARAMETER_SHIFT_POLICY_ID",
    "SPSA_CONFIG_SCHEMA_VERSION",
    "SPSA_FIXED_METHOD_ID",
    "SPSA_LAYERWISE_METHOD_ID",
    "SPSA_PERTURBATION_DISTRIBUTION_ID",
    "BMPDCompetitorStageRunner",
    "CompetitorIterationRecord",
    "CompetitorObjective",
    "CompetitorObjectiveRequest",
    "CompetitorStageExecution",
    "CompetitorWorkBudget",
    "FixedRateNoisyCompetitorObjective",
    "ParameterShiftAdamConfig",
    "ParameterShiftAdamStageAdapter",
    "SPSAConfig",
    "SPSAStageAdapter",
    "build_parameter_shift_adam_fixed_template",
    "build_parameter_shift_adam_layerwise_template",
    "build_spsa_fixed_template",
    "build_spsa_layerwise_template",
    "validate_competitor_stage_evidence",
]
