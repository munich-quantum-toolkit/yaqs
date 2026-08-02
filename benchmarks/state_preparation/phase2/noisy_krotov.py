# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Benchmark-grade Phase II adapter for fixed-rate noisy Krotov stages.

This module deliberately owns the scientific scheduling and provenance rules
that do not belong in the Phase I Krotov implementation.  It accepts a fully
resolved :class:`TrainingStageConfig`, never an evaluation configuration, and
keeps optimizer-ordering, training-trajectory, and checkpoint-validation
randomness separate.
"""

# The adapter has many small private validation helpers whose return and error
# contracts are explicit in their names and annotations. Repeating them in every
# private docstring would obscure the public scientific contract below.
# The single stage-execution boundary deliberately normalizes every failure into
# a structured record and therefore owns one large try block.
# ruff: noqa: BLE001, DOC201, DOC501, PLW0717, TRY301

from __future__ import annotations

import copy
import hashlib
import math
import operator
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from benchmarks.state_preparation.constants import (
    BALLARIN_NOISE_ID,
    NOISELESS_NOISE_ID,
    STANDARD_NOISE_IDS,
)
from benchmarks.state_preparation.noise import (
    FIXED_RATE_NOISE_DEFINITION_VERSION,
    HISTORICAL_FIXED_RATE_NOISE_ID,
    create_historical_fixed_rate_noise_provider,
    create_scaled_standard_noise_provider,
)
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import (
    KrotovFixedMapEnsemble,
    KrotovOptions,
    KrotovTJMOptions,
    KrotovTruncation,
    noisy_state_preparation_contribution,
    noisy_state_preparation_metrics,
    sample_krotov_fixed_map_ensemble,
    state_preparation_contribution,
    state_preparation_metrics,
)
from mqt.yaqs.optimization.parameterized_circuit import ParameterizedCircuit, ParameterizedGate

from .canonical import canonical_checksum, freeze_json_mapping, thaw_json_mapping
from .pipeline import TrainingStageConfig
from .targets import MaterializedTarget
from .validation import (
    require_checksum,
    require_exact_keys,
    require_float,
    require_int,
    require_mapping,
    require_slug,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import (
        GateNoiseProvider,
        KrotovMapRole,
    )


NOISY_KROTOV_CIRCUIT_BINDING_SCHEMA_VERSION = "yaqs.state_preparation.phase2.noisy_krotov_circuit.v1"
NOISY_KROTOV_CHECKPOINT_SELECTION_SCHEMA_VERSION = "yaqs.state_preparation.phase2.noisy_krotov_checkpoint_selection.v1"
NOISY_KROTOV_RESUME_STATE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.noisy_krotov_resume_state.v1"
NOISY_KROTOV_TRACE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.noisy_krotov_trace.v3"
NOISY_KROTOV_OBJECTIVE_BINDING_SCHEMA_VERSION = "yaqs.state_preparation.phase2.noisy_krotov_objective.v1"
NOISY_KROTOV_EXECUTION_SCHEMA_VERSION = "yaqs.state_preparation.phase2.noisy_krotov_execution.v4"
NOISY_KROTOV_FAILURE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.noisy_krotov_failure.v1"
NOISY_KROTOV_ADAPTER_VERSION = "yaqs.state_preparation.phase2.fixed_rate_noisy_krotov.v1"

LOGICAL_PARAMETERIZED_GATE_PLACEMENT = "logical_parameterized_gates"
PRIMARY_COMPILER_POLICY_ID = "quantinuum_rzz_chain_v1"
PRIMARY_CONNECTIVITY = "linear_chain"
PRIMARY_ROUTING_POLICY_ID = "identity_no_swap"
PRIMARY_COUNTING_POLICY_ID = "native_two_qubit_gates_per_chain_edge"

_SUPPORTED_OPTIMIZER_HYPERPARAMETERS = frozenset({"learning_rate", "schedule", "decay"})
_SCHEDULES = frozenset({"constant", "inverse", "exp"})
_FAILURE_PHASES = frozenset({"validation", "sampling", "optimization", "checkpoint_validation"})
_OBJECTIVE_ID = "pure_state_infidelity_v1"
_COMPUTATIONAL_ZERO_INITIAL_STATE_POLICY = "computational_zero_v1"
_CUSTOM_INITIAL_STATE_POLICY = "custom_state_v1"
_OBJECTIVE_INITIAL_STATE_POLICIES = frozenset({
    _COMPUTATIONAL_ZERO_INITIAL_STATE_POLICY,
    _CUSTOM_INITIAL_STATE_POLICY,
})
_MATERIALIZED_TARGET_IDENTITY_KEYS = frozenset({
    "target_instance_id",
    "target_instance_spec_checksum",
    "population_config_checksum",
    "target_manifest_checksum",
    "parameter_checksum",
    "family_id",
    "stratum_id",
    "qubit_count",
    "norm",
    "vector_checksum",
})
_OBJECTIVE_BINDING_KEYS = frozenset({
    "schema_version",
    "objective_id",
    "target_state_checksum",
    "initial_state_policy",
    "initial_state_checksum",
    "materialized_target_identity",
    "objective_checksum",
    "content_checksum",
})

UpdateSignalKind = Literal[
    "none",
    "independent_pathwise_gradient",
    "independent_pathwise_update",
    "cross_dense_sum_update",
]
FailurePhase = Literal["validation", "sampling", "optimization", "checkpoint_validation"]


def _require_builtin_int(value: object, name: str, *, minimum: int = 0) -> int:
    """Return a validated built-in integer.

    Raises:
        TypeError: If ``value`` is not an integer or is a Boolean.
        ValueError: If ``value`` is smaller than ``minimum``.
    """
    if type(value) is not int:
        msg = f"{name} must be an integer, got {type(value).__name__}."
        raise TypeError(msg)
    normalized = value
    if normalized < minimum:
        msg = f"{name} must be at least {minimum}, got {normalized}."
        raise ValueError(msg)
    return normalized


def _require_finite_float(value: object, name: str, *, positive: bool = False) -> float:
    """Return a validated finite built-in float.

    Raises:
        TypeError: If ``value`` is a Boolean or not a real scalar.
        ValueError: If ``value`` is nonfinite or violates ``positive``.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        msg = f"{name} must be a finite real number."
        raise TypeError(msg)
    normalized = float(value)
    if not math.isfinite(normalized) or (positive and normalized <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        msg = f"{name} must be {qualifier}, got {normalized!r}."
        raise ValueError(msg)
    return normalized


def _vector_bytes(vector: NDArray[np.float64]) -> bytes:
    """Return canonical little-endian float64 bytes for a parameter vector."""
    return np.ascontiguousarray(vector, dtype=np.dtype("<f8")).tobytes(order="C")


def _vector_checksum(vector: NDArray[np.float64]) -> str:
    """Return a content checksum for a parameter vector."""
    return f"sha256:{hashlib.sha256(_vector_bytes(vector)).hexdigest()}"


def _statevector_checksum(value: MPS | NDArray[np.complex128], name: str) -> str:
    """Return a stable checksum for one normalized dense or MPS state."""
    vector = value.to_vec() if isinstance(value, MPS) else value
    resolved = np.asarray(vector, dtype=np.complex128)
    if resolved.ndim != 1 or not np.all(np.isfinite(resolved)):
        msg = f"{name} must resolve to a finite one-dimensional statevector."
        raise ValueError(msg)
    norm = float(np.linalg.norm(resolved))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-12):
        msg = f"{name} statevector norm must equal one, got {norm!r}."
        raise ValueError(msg)
    canonical = np.ascontiguousarray(resolved, dtype=np.dtype("<c16"))
    return canonical_checksum({
        "amplitude_count": int(canonical.size),
        "data_sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
        "dtype": "<c16",
    })


def _objective_checksum_from_state_checksums(target_state_checksum: str, initial_state_checksum: str) -> str:
    """Seal the pure-state infidelity objective and its state checksums."""
    return canonical_checksum({
        "initial_state_checksum": initial_state_checksum,
        "objective_id": _OBJECTIVE_ID,
        "target_state_checksum": target_state_checksum,
    })


def _copy_vector(payload: bytes) -> NDArray[np.float64]:
    """Restore a detached writable parameter vector from canonical bytes."""
    return np.frombuffer(payload, dtype=np.dtype("<f8")).astype(np.float64, copy=True)


def _validated_theta(
    initial_theta: NDArray[np.float64],
    *,
    expected_count: int,
) -> NDArray[np.float64]:
    """Validate and detach a resolved stage parameter vector.

    Raises:
        TypeError: If the vector is not a NumPy array.
        ValueError: If its shape or values are invalid.
    """
    if not isinstance(initial_theta, np.ndarray):
        msg = "initial_theta must be a NumPy array."
        raise TypeError(msg)
    theta = np.asarray(initial_theta, dtype=np.float64)
    if theta.shape != (expected_count,) or not np.all(np.isfinite(theta)):
        msg = f"initial_theta must be finite with shape ({expected_count},), got {theta.shape}."
        raise ValueError(msg)
    return theta.copy()


def _resolved_target(
    target: MaterializedTarget | MPS | NDArray[np.complex128],
    *,
    num_qubits: int,
) -> MPS | NDArray[np.complex128]:
    """Validate and detach the stage target.

    Raises:
        TypeError: If the target has an unsupported type.
        ValueError: If target size, shape, finiteness, or norm is invalid.
    """
    if isinstance(target, MaterializedTarget):
        if target.qubit_count != num_qubits:
            msg = f"Materialized target has {target.qubit_count} qubits, expected {num_qubits}."
            raise ValueError(msg)
        return target.state_vector_copy()
    if isinstance(target, MPS):
        if target.length != num_qubits:
            msg = f"Target MPS has {target.length} qubits, expected {num_qubits}."
            raise ValueError(msg)
        return copy.deepcopy(target)
    if not isinstance(target, np.ndarray):
        msg = "target must be a MaterializedTarget, MPS, or complex NumPy statevector."
        raise TypeError(msg)
    vector = np.asarray(target, dtype=np.complex128)
    if vector.shape != (2**num_qubits,) or not np.all(np.isfinite(vector)):
        msg = f"Target statevector must be finite with shape ({2**num_qubits},)."
        raise ValueError(msg)
    norm = float(np.linalg.norm(vector))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-12):
        msg = f"Target statevector norm must equal one, got {norm!r}."
        raise ValueError(msg)
    return vector.copy()


def _initial_state_template(initial_state: MPS | None, num_qubits: int) -> MPS:
    """Return a detached validated fixed input state."""
    if initial_state is None:
        return MPS(num_qubits)
    if not isinstance(initial_state, MPS):
        msg = "initial_state must be an MPS or None."
        raise TypeError(msg)
    if initial_state.length != num_qubits:
        msg = f"Initial MPS has {initial_state.length} qubits, expected {num_qubits}."
        raise ValueError(msg)
    return copy.deepcopy(initial_state)


def noisy_krotov_computational_zero_state_checksum(num_qubits: int) -> str:
    """Return the WP17 state checksum of the canonical all-zero input state.

    Args:
        num_qubits: Positive state-preparation system size.

    Returns:
        The exact statevector checksum used by the noisy-Krotov objective.
    """
    qubits = _require_builtin_int(num_qubits, "num_qubits", minimum=1)
    return _statevector_checksum(MPS(qubits), "computational_zero_state")


def _validated_materialized_target_identity(
    value: object,
    *,
    target_state_checksum: str,
) -> Mapping[str, object]:
    """Validate and freeze an authorized Phase II target identity."""
    identity = freeze_json_mapping(value, "materialized_target_identity")
    require_exact_keys(identity, _MATERIALIZED_TARGET_IDENTITY_KEYS, "materialized_target_identity")
    require_slug(identity["target_instance_id"], "materialized_target_identity.target_instance_id")
    for name in (
        "target_instance_spec_checksum",
        "population_config_checksum",
        "target_manifest_checksum",
        "parameter_checksum",
        "vector_checksum",
    ):
        require_checksum(identity[name], f"materialized_target_identity.{name}")
    require_slug(identity["family_id"], "materialized_target_identity.family_id")
    require_slug(identity["stratum_id"], "materialized_target_identity.stratum_id")
    qubit_count = require_int(identity["qubit_count"], "materialized_target_identity.qubit_count", minimum=1)
    norm = require_float(identity["norm"], "materialized_target_identity.norm", minimum=0.0)
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-12):
        msg = "materialized_target_identity.norm must equal one."
        raise ValueError(msg)
    vector_checksum = cast("str", identity["vector_checksum"])
    expected_state_checksum = canonical_checksum({
        "amplitude_count": 2**qubit_count,
        "data_sha256": vector_checksum.removeprefix("sha256:"),
        "dtype": "<c16",
    })
    if expected_state_checksum != target_state_checksum:
        msg = "Materialized target vector identity does not reproduce target_state_checksum."
        raise ValueError(msg)
    return identity


@dataclass(frozen=True, slots=True, init=False)
class NoisyKrotovObjectiveBinding:
    """Sealed pure-state objective operands and optional authorized target identity."""

    target_state_checksum: str
    initial_state_policy: str
    initial_state_checksum: str
    materialized_target_identity: Mapping[str, object] | None
    objective_id: str = field(default=_OBJECTIVE_ID, init=False)
    schema_version: str = field(default=NOISY_KROTOV_OBJECTIVE_BINDING_SCHEMA_VERSION, init=False)

    def __init__(
        self,
        *,
        target_state_checksum: str,
        initial_state_policy: str,
        initial_state_checksum: str,
        materialized_target_identity: Mapping[str, object] | None,
    ) -> None:
        """Validate checksums, initial-state semantics, and target provenance."""
        target_checksum = require_checksum(target_state_checksum, "target_state_checksum")
        initial_checksum = require_checksum(initial_state_checksum, "initial_state_checksum")
        if initial_state_policy not in _OBJECTIVE_INITIAL_STATE_POLICIES:
            msg = f"initial_state_policy must be one of {sorted(_OBJECTIVE_INITIAL_STATE_POLICIES)!r}."
            raise ValueError(msg)
        identity = (
            None
            if materialized_target_identity is None
            else _validated_materialized_target_identity(
                materialized_target_identity,
                target_state_checksum=target_checksum,
            )
        )
        object.__setattr__(self, "target_state_checksum", target_checksum)
        object.__setattr__(self, "initial_state_policy", initial_state_policy)
        object.__setattr__(self, "initial_state_checksum", initial_checksum)
        object.__setattr__(self, "materialized_target_identity", identity)
        object.__setattr__(self, "objective_id", _OBJECTIVE_ID)
        object.__setattr__(self, "schema_version", NOISY_KROTOV_OBJECTIVE_BINDING_SCHEMA_VERSION)

    @classmethod
    def from_inputs(
        cls,
        target: MaterializedTarget | MPS | NDArray[np.complex128],
        initial_state: MPS | None,
        *,
        num_qubits: int,
    ) -> NoisyKrotovObjectiveBinding:
        """Construct the exact binding used by one WP17 execution.

        Args:
            target: Authorized Phase II target or standalone target state.
            initial_state: Optional explicit input state; ``None`` means all-zero.
            num_qubits: Exact circuit width.

        Returns:
            A checksum-sealed objective binding.
        """
        qubits = _require_builtin_int(num_qubits, "num_qubits", minimum=1)
        resolved_target = _resolved_target(target, num_qubits=qubits)
        resolved_initial = _initial_state_template(initial_state, qubits)
        initial_checksum = _statevector_checksum(resolved_initial, "initial_state")
        zero_checksum = noisy_krotov_computational_zero_state_checksum(qubits)
        return cls(
            target_state_checksum=_statevector_checksum(resolved_target, "target"),
            initial_state_policy=(
                _COMPUTATIONAL_ZERO_INITIAL_STATE_POLICY
                if initial_checksum == zero_checksum
                else _CUSTOM_INITIAL_STATE_POLICY
            ),
            initial_state_checksum=initial_checksum,
            materialized_target_identity=(target.identity_dict() if isinstance(target, MaterializedTarget) else None),
        )

    @property
    def objective_checksum(self) -> str:
        """Checksum of the exact objective and both state operands."""
        return _objective_checksum_from_state_checksums(
            self.target_state_checksum,
            self.initial_state_checksum,
        )

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered objective-binding field."""
        return {
            "schema_version": self.schema_version,
            "objective_id": self.objective_id,
            "target_state_checksum": self.target_state_checksum,
            "initial_state_policy": self.initial_state_policy,
            "initial_state_checksum": self.initial_state_checksum,
            "materialized_target_identity": (
                None
                if self.materialized_target_identity is None
                else thaw_json_mapping(self.materialized_target_identity)
            ),
            "objective_checksum": self.objective_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum sealing objective semantics and authorized target identity."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return the complete checksum-sealed objective-binding document."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, value: object) -> NoisyKrotovObjectiveBinding:
        """Decode and exactly verify a serialized objective-binding document."""
        document = freeze_json_mapping(require_mapping(value, "objective_binding"), "objective_binding")
        require_exact_keys(document, _OBJECTIVE_BINDING_KEYS, "objective_binding")
        if document["schema_version"] != NOISY_KROTOV_OBJECTIVE_BINDING_SCHEMA_VERSION:
            msg = "objective_binding uses an unsupported schema version."
            raise ValueError(msg)
        if document["objective_id"] != _OBJECTIVE_ID:
            msg = f"objective_binding.objective_id must be {_OBJECTIVE_ID!r}."
            raise ValueError(msg)
        raw_identity = document["materialized_target_identity"]
        if raw_identity is not None and not isinstance(raw_identity, Mapping):
            msg = "objective_binding.materialized_target_identity must be a mapping or None."
            raise TypeError(msg)
        binding = cls(
            target_state_checksum=cast("str", document["target_state_checksum"]),
            initial_state_policy=cast("str", document["initial_state_policy"]),
            initial_state_checksum=cast("str", document["initial_state_checksum"]),
            materialized_target_identity=cast("Mapping[str, object] | None", raw_identity),
        )
        if (
            document["objective_checksum"] != binding.objective_checksum
            or document["content_checksum"] != binding.content_checksum
            or canonical_checksum(thaw_json_mapping(document)) != canonical_checksum(binding.to_dict())
        ):
            msg = "objective_binding does not reconstruct its exact sealed objective semantics."
            raise ValueError(msg)
        return binding


@dataclass(frozen=True, slots=True, init=False)
class NoisyKrotovCircuitBinding:
    """Frozen scientific binding for a logical Phase II training circuit.

    The circuit remains an in-memory execution object.  ``content_checksum``
    seals its complete serializable gate structure together with the compiler,
    connectivity, routing, placement, and resource-counting policies fixed by
    the preregistration.
    """

    _circuit: ParameterizedCircuit = field(repr=False, compare=False)
    topology_id: str
    placement: str = LOGICAL_PARAMETERIZED_GATE_PLACEMENT
    compiler_policy_id: str = PRIMARY_COMPILER_POLICY_ID
    connectivity: str = PRIMARY_CONNECTIVITY
    routing_policy_id: str = PRIMARY_ROUTING_POLICY_ID
    counting_policy_id: str = PRIMARY_COUNTING_POLICY_ID
    schema_version: str = field(default=NOISY_KROTOV_CIRCUIT_BINDING_SCHEMA_VERSION, init=False)

    def __init__(
        self,
        circuit: ParameterizedCircuit,
        topology_id: str,
        placement: str = LOGICAL_PARAMETERIZED_GATE_PLACEMENT,
        compiler_policy_id: str = PRIMARY_COMPILER_POLICY_ID,
        connectivity: str = PRIMARY_CONNECTIVITY,
        routing_policy_id: str = PRIMARY_ROUTING_POLICY_ID,
        counting_policy_id: str = PRIMARY_COUNTING_POLICY_ID,
    ) -> None:
        """Validate and defensively snapshot a logical circuit binding.

        Args:
            circuit: Logical parameterized circuit to snapshot.
            topology_id: Stable logical-topology identifier.
            placement: Frozen logical noise-placement policy.
            compiler_policy_id: Frozen compiler policy.
            connectivity: Frozen device connectivity.
            routing_policy_id: Frozen routing policy.
            counting_policy_id: Frozen native-resource counting policy.
        """
        object.__setattr__(self, "_circuit", circuit)
        object.__setattr__(self, "topology_id", topology_id)
        object.__setattr__(self, "placement", placement)
        object.__setattr__(self, "compiler_policy_id", compiler_policy_id)
        object.__setattr__(self, "connectivity", connectivity)
        object.__setattr__(self, "routing_policy_id", routing_policy_id)
        object.__setattr__(self, "counting_policy_id", counting_policy_id)
        object.__setattr__(self, "schema_version", NOISY_KROTOV_CIRCUIT_BINDING_SCHEMA_VERSION)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate the logical circuit and all preregistered policy bindings.

        Raises:
            TypeError: If ``circuit`` or a string field has the wrong type.
            ValueError: If the circuit is compiled/native, data-dependent, or
                bound to a different Phase II policy.
        """
        if not isinstance(self._circuit, ParameterizedCircuit):
            msg = "circuit must be a ParameterizedCircuit."
            raise TypeError(msg)
        for name in (
            "topology_id",
            "placement",
            "compiler_policy_id",
            "connectivity",
            "routing_policy_id",
            "counting_policy_id",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                msg = f"{name} must be a nonempty string without surrounding whitespace."
                raise TypeError(msg)
        expected = {
            "placement": LOGICAL_PARAMETERIZED_GATE_PLACEMENT,
            "compiler_policy_id": PRIMARY_COMPILER_POLICY_ID,
            "connectivity": PRIMARY_CONNECTIVITY,
            "routing_policy_id": PRIMARY_ROUTING_POLICY_ID,
            "counting_policy_id": PRIMARY_COUNTING_POLICY_ID,
        }
        for name, required in expected.items():
            if getattr(self, name) != required:
                msg = f"{name} must be the frozen Phase II value {required!r}."
                raise ValueError(msg)
        if any(gate.data_map is not None for gate in self._circuit.gates):
            msg = "Phase II state-preparation training circuits cannot contain data maps."
            raise ValueError(msg)
        if any(gate.native_gate_id is not None for gate in self._circuit.gates):
            msg = "Noisy Krotov training acts on logical gates; compiled native gates are not accepted."
            raise ValueError(msg)
        if any(len(gate.sites) == 2 and abs(gate.sites[0] - gate.sites[1]) != 1 for gate in self._circuit.gates):
            msg = "identity_no_swap on linear_chain requires every logical two-qubit gate to be nearest-neighbor."
            raise ValueError(msg)
        if not self.noisy_gate_indices:
            msg = "The logical circuit must contain at least one noise-enabled parameterized gate."
            raise ValueError(msg)
        object.__setattr__(self, "_circuit", copy.deepcopy(self._circuit))

    @property
    def circuit(self) -> ParameterizedCircuit:
        """A detached execution copy of the sealed logical circuit."""
        return copy.deepcopy(self._circuit)

    @property
    def noisy_gate_indices(self) -> tuple[int, ...]:
        """Indices of exactly the logical parameterized gates exposed to noise."""
        return tuple(
            index for index, gate in enumerate(self._circuit.gates) if gate.is_trainable and gate.noise_enabled
        )

    def identity_payload(self) -> dict[str, object]:
        """Return the complete serializable circuit-policy binding."""
        gates: list[dict[str, object]] = [
            {
                "name": gate.name,
                "sites": list(gate.sites),
                "param_index": gate.param_index,
                "angle_scale": float(gate.angle_scale),
                "angle_offset": float(gate.angle_offset),
                "fixed_params": [float(value) for value in gate.fixed_params],
                "logical_gate_id": gate.logical_gate_id,
                "native_gate_id": gate.native_gate_id,
                "noise_enabled": gate.noise_enabled,
            }
            for gate in self._circuit.gates
        ]
        return {
            "schema_version": self.schema_version,
            "topology_id": self.topology_id,
            "placement": self.placement,
            "compiler_policy_id": self.compiler_policy_id,
            "connectivity": self.connectivity,
            "routing_policy_id": self.routing_policy_id,
            "counting_policy_id": self.counting_policy_id,
            "num_qubits": self._circuit.num_qubits,
            "num_params": self._circuit.num_params,
            "noisy_gate_indices": list(self.noisy_gate_indices),
            "gates": gates,
        }

    @property
    def content_checksum(self) -> str:
        """Stable checksum of the circuit and scientific placement contract."""
        return canonical_checksum(self.identity_payload())

    def to_dict(self) -> dict[str, object]:
        """Return the checksum-sealed serializable binding metadata."""
        payload = self.identity_payload()
        return {**payload, "content_checksum": self.content_checksum}


def decode_noisy_krotov_circuit_binding_document(value: object) -> NoisyKrotovCircuitBinding:
    """Reconstruct and exactly verify a serialized noisy-Krotov circuit binding."""
    document = freeze_json_mapping(value, "circuit_binding_document")
    expected_keys = {
        "schema_version",
        "topology_id",
        "placement",
        "compiler_policy_id",
        "connectivity",
        "routing_policy_id",
        "counting_policy_id",
        "num_qubits",
        "num_params",
        "noisy_gate_indices",
        "gates",
        "content_checksum",
    }
    if set(document) != expected_keys:
        msg = "circuit_binding_document fields do not match the exact WP17 schema."
        raise ValueError(msg)
    if document["schema_version"] != NOISY_KROTOV_CIRCUIT_BINDING_SCHEMA_VERSION:
        msg = "circuit_binding_document uses an unsupported schema version."
        raise ValueError(msg)
    gates_value = document["gates"]
    noisy_indices = document["noisy_gate_indices"]
    if type(gates_value) is not tuple or type(noisy_indices) is not tuple:
        msg = "circuit binding gates and noisy_gate_indices must be serialized sequences."
        raise TypeError(msg)
    num_qubits = _require_builtin_int(document["num_qubits"], "circuit_binding_document.num_qubits", minimum=1)
    num_params = _require_builtin_int(document["num_params"], "circuit_binding_document.num_params")
    gate_keys = {
        "name",
        "sites",
        "param_index",
        "angle_scale",
        "angle_offset",
        "fixed_params",
        "logical_gate_id",
        "native_gate_id",
        "noise_enabled",
    }
    gates: list[ParameterizedGate] = []
    for index, raw_gate in enumerate(gates_value):
        if not isinstance(raw_gate, Mapping) or set(raw_gate) != gate_keys:
            msg = f"circuit_binding_document.gates[{index}] fields do not match the exact WP17 schema."
            raise ValueError(msg)
        gate_document = cast("Mapping[str, object]", raw_gate)
        sites = gate_document["sites"]
        fixed_params = gate_document["fixed_params"]
        if type(sites) is not tuple or type(fixed_params) is not tuple:
            msg = f"circuit_binding_document.gates[{index}] sites and fixed_params must be sequences."
            raise TypeError(msg)
        if not all(type(site) is int for site in sites):
            msg = f"circuit_binding_document.gates[{index}].sites must contain built-in integers."
            raise TypeError(msg)
        normalized_sites = cast("tuple[int, ...]", sites)
        if len(set(normalized_sites)) != len(normalized_sites) or any(
            site < 0 or site >= num_qubits for site in normalized_sites
        ):
            msg = f"circuit_binding_document.gates[{index}].sites are invalid for the declared qubit count."
            raise ValueError(msg)
        param_index = gate_document["param_index"]
        if param_index is not None and type(param_index) is not int:
            msg = f"circuit_binding_document.gates[{index}].param_index must be an integer or None."
            raise TypeError(msg)
        if param_index is not None and (param_index < 0 or param_index >= num_params):
            msg = f"circuit_binding_document.gates[{index}].param_index is outside the parameter vector."
            raise ValueError(msg)
        name = gate_document["name"]
        if type(name) is not str or not name:
            msg = f"circuit_binding_document.gates[{index}].name must be nonempty text."
            raise TypeError(msg)
        logical_gate_id = gate_document["logical_gate_id"]
        native_gate_id = gate_document["native_gate_id"]
        for field_name, field_value in (
            ("logical_gate_id", logical_gate_id),
            ("native_gate_id", native_gate_id),
        ):
            if field_value is not None and type(field_value) not in {int, str}:
                msg = f"circuit_binding_document.gates[{index}].{field_name} must be an integer, string, or None."
                raise TypeError(msg)
        noise_enabled = gate_document["noise_enabled"]
        if type(noise_enabled) is not bool:
            msg = f"circuit_binding_document.gates[{index}].noise_enabled must be a bool."
            raise TypeError(msg)
        angle_scale = _require_finite_float(
            gate_document["angle_scale"],
            f"circuit_binding_document.gates[{index}].angle_scale",
        )
        angle_offset = _require_finite_float(
            gate_document["angle_offset"],
            f"circuit_binding_document.gates[{index}].angle_offset",
        )
        normalized_fixed_params = tuple(
            _require_finite_float(item, f"circuit_binding_document.gates[{index}].fixed_params")
            for item in fixed_params
        )
        gates.append(
            ParameterizedGate(
                name=name,
                sites=normalized_sites,
                param_index=param_index,
                angle_scale=angle_scale,
                angle_offset=angle_offset,
                fixed_params=normalized_fixed_params,
                logical_gate_id=cast("int | str | None", logical_gate_id),
                native_gate_id=cast("int | str | None", native_gate_id),
                noise_enabled=noise_enabled,
            )
        )
    if not all(type(index) is int for index in noisy_indices):
        msg = "circuit_binding_document.noisy_gate_indices must contain built-in integers."
        raise TypeError(msg)
    circuit = ParameterizedCircuit(
        num_qubits,
        gates,
        num_params=num_params,
    )
    zero_theta = np.zeros(circuit.num_params, dtype=np.float64)
    for index, gate in enumerate(circuit.gates):
        try:
            matrix, _sites, _angle = circuit.gate_matrix_and_angle(gate, zero_theta)
        except (AttributeError, TypeError, ValueError) as error:
            msg = f"circuit_binding_document.gates[{index}] cannot reconstruct its gate matrix: {error}."
            raise ValueError(msg) from error
        expected_dimension = 2 ** len(gate.sites)
        if matrix.shape != (expected_dimension, expected_dimension):
            msg = f"circuit_binding_document.gates[{index}] matrix arity does not match its sites."
            raise ValueError(msg)
    binding = NoisyKrotovCircuitBinding(
        circuit,
        cast("str", document["topology_id"]),
        placement=cast("str", document["placement"]),
        compiler_policy_id=cast("str", document["compiler_policy_id"]),
        connectivity=cast("str", document["connectivity"]),
        routing_policy_id=cast("str", document["routing_policy_id"]),
        counting_policy_id=cast("str", document["counting_policy_id"]),
    )
    supplied_checksum = require_checksum(document["content_checksum"], "content_checksum")
    if (
        tuple(noisy_indices) != binding.noisy_gate_indices
        or supplied_checksum != binding.content_checksum
        or canonical_checksum(thaw_json_mapping(document)) != canonical_checksum(binding.to_dict())
    ):
        msg = "circuit_binding_document does not reconstruct the exact sealed WP17 circuit binding."
        raise ValueError(msg)
    return binding


@dataclass(frozen=True, slots=True)
class KrotovStageTranslation:
    """Exact translation of one WP16 stage into Krotov execution objects."""

    options: KrotovOptions
    tjm_options: KrotovTJMOptions | None
    noise_provider: GateNoiseProvider | None
    provider_checksum: str | None
    learning_rate: float
    schedule: Literal["constant", "inverse", "exp"]
    decay: float


@dataclass(frozen=True, slots=True)
class KrotovWorkLedger:
    """Cumulative normalized work performed by a WP17 stage execution.

    ``trajectory_gate_applications`` counts circuit-gate applications in
    forward trajectory simulations, including map generation and monitoring
    replay. The fixed WP16 ledger has no cross-pair-contraction counter, so a
    cross dense-sum update is deliberately not counted as a gradient evaluation.
    """

    objective_evaluations: int = 0
    gradient_evaluations: int = 0
    training_trajectories: int = 0
    checkpoint_validation_trajectories: int = 0
    test_trajectories: int = 0
    trajectory_gate_applications: int = 0

    def __post_init__(self) -> None:
        """Validate all counters as nonnegative built-in integers."""
        for name in (
            "objective_evaluations",
            "gradient_evaluations",
            "training_trajectories",
            "checkpoint_validation_trajectories",
            "test_trajectories",
            "trajectory_gate_applications",
        ):
            _require_builtin_int(getattr(self, name), name)

    def plus(self, **increments: int) -> KrotovWorkLedger:
        """Return a ledger with named nonnegative increments applied.

        Raises:
            ValueError: If an unknown counter is supplied.
        """
        fields = self.to_dict()
        unknown = set(increments) - set(fields)
        if unknown:
            msg = f"Unknown work counters: {sorted(unknown)!r}."
            raise ValueError(msg)
        for name, increment in increments.items():
            fields[name] += _require_builtin_int(increment, f"increments.{name}")
        return KrotovWorkLedger(**fields)

    def to_dict(self) -> dict[str, int]:
        """Return the exact WP16 normalized-work mapping."""
        return {
            "objective_evaluations": self.objective_evaluations,
            "gradient_evaluations": self.gradient_evaluations,
            "training_trajectories": self.training_trajectories,
            "checkpoint_validation_trajectories": self.checkpoint_validation_trajectories,
            "test_trajectories": self.test_trajectories,
            "trajectory_gate_applications": self.trajectory_gate_applications,
        }


@dataclass(frozen=True, slots=True)
class NoisyKrotovIterationRecord:
    """Semantically explicit monitoring and work record for one stage state."""

    local_iteration: int
    global_iteration: int
    parameter_checksum: str
    learning_rate: float
    monitoring_loss: float
    monitoring_fidelity: float
    checkpoint_validation_fidelity: float | None
    update_signal: tuple[float, ...]
    update_signal_kind: UpdateSignalKind
    update_signal_norm: float
    gradient_norm: float | None
    cross_dense_sum_norm: float | None
    update_norm: float
    trajectory_count: int
    nonidentity_events: int
    training_ensemble_id: str | None
    training_ensemble_checksum: str | None
    checkpoint_validation_ensemble_checksum: str | None
    cumulative_work: KrotovWorkLedger
    training_ensemble_sampled: bool = False
    checkpoint_validation_ensemble_sampled: bool = False
    cross_trajectory_pairings: int = 0
    cumulative_cross_trajectory_pairings: int = 0
    schema_version: str = field(default=NOISY_KROTOV_TRACE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate trace semantics, including cross/gradient separation."""
        _require_builtin_int(self.local_iteration, "local_iteration")
        _require_builtin_int(self.global_iteration, "global_iteration")
        object.__setattr__(self, "parameter_checksum", require_checksum(self.parameter_checksum, "parameter_checksum"))
        _require_finite_float(self.learning_rate, "learning_rate")
        loss = _require_finite_float(self.monitoring_loss, "monitoring_loss")
        fidelity = _require_finite_float(self.monitoring_fidelity, "monitoring_fidelity")
        if not 0.0 <= loss <= 1.0 + 1e-10 or not -1e-10 <= fidelity <= 1.0 + 1e-10:
            msg = "Monitoring loss and fidelity must be physical probabilities."
            raise ValueError(msg)
        object.__setattr__(self, "monitoring_loss", min(1.0, max(0.0, loss)))
        object.__setattr__(self, "monitoring_fidelity", min(1.0, max(0.0, fidelity)))
        if self.checkpoint_validation_fidelity is not None:
            validation = _require_finite_float(
                self.checkpoint_validation_fidelity,
                "checkpoint_validation_fidelity",
            )
            if not -1e-10 <= validation <= 1.0 + 1e-10:
                msg = "checkpoint_validation_fidelity must lie in [0, 1]."
                raise ValueError(msg)
            object.__setattr__(
                self,
                "checkpoint_validation_fidelity",
                min(1.0, max(0.0, validation)),
            )
        for name in ("training_ensemble_sampled", "checkpoint_validation_ensemble_sampled"):
            if type(getattr(self, name)) is not bool:
                msg = f"{name} must be a bool."
                raise TypeError(msg)
        if self.training_ensemble_sampled and self.training_ensemble_checksum is None:
            msg = "A sampled training ensemble requires its checksum."
            raise ValueError(msg)
        if self.checkpoint_validation_ensemble_sampled and self.checkpoint_validation_ensemble_checksum is None:
            msg = "A sampled checkpoint-validation ensemble requires its checksum."
            raise ValueError(msg)
        if self.update_signal_kind not in {
            "none",
            "independent_pathwise_gradient",
            "independent_pathwise_update",
            "cross_dense_sum_update",
        }:
            msg = f"Unknown update_signal_kind {self.update_signal_kind!r}."
            raise ValueError(msg)
        signal = tuple(_require_finite_float(value, "update_signal") for value in self.update_signal)
        object.__setattr__(self, "update_signal", signal)
        signal_norm = _require_finite_float(self.update_signal_norm, "update_signal_norm")
        if not math.isclose(signal_norm, float(np.linalg.norm(signal)), rel_tol=1e-12, abs_tol=1e-12):
            msg = "update_signal_norm does not match update_signal."
            raise ValueError(msg)
        _require_finite_float(self.update_norm, "update_norm")
        _require_builtin_int(self.trajectory_count, "trajectory_count")
        _require_builtin_int(self.nonidentity_events, "nonidentity_events")
        pairings = _require_builtin_int(self.cross_trajectory_pairings, "cross_trajectory_pairings")
        cumulative_pairings = _require_builtin_int(
            self.cumulative_cross_trajectory_pairings,
            "cumulative_cross_trajectory_pairings",
        )
        if pairings > cumulative_pairings:
            msg = "cross_trajectory_pairings cannot exceed its cumulative counter."
            raise ValueError(msg)
        if self.update_signal_kind == "cross_dense_sum_update" and pairings == 0:
            msg = "Cross updates must record their dense trajectory pairings."
            raise ValueError(msg)
        if self.update_signal_kind != "cross_dense_sum_update" and pairings != 0:
            msg = "Only cross updates may record dense trajectory pairings."
            raise ValueError(msg)
        if self.update_signal_kind == "independent_pathwise_gradient":
            if self.gradient_norm is None or self.cross_dense_sum_norm is not None:
                msg = "Independent updates require gradient_norm and forbid cross_dense_sum_norm."
                raise ValueError(msg)
            gradient_norm = _require_finite_float(self.gradient_norm, "gradient_norm")
            if not math.isclose(gradient_norm, signal_norm, rel_tol=1e-12, abs_tol=1e-12):
                msg = "gradient_norm does not match update_signal_norm."
                raise ValueError(msg)
        elif self.update_signal_kind == "independent_pathwise_update":
            if self.gradient_norm is not None or self.cross_dense_sum_norm is not None:
                msg = "Unvalidated pathwise updates must not populate a gradient or cross-sum norm."
                raise ValueError(msg)
        elif self.update_signal_kind == "cross_dense_sum_update":
            if self.gradient_norm is not None or self.cross_dense_sum_norm is None:
                msg = "Cross updates require cross_dense_sum_norm and must not claim gradient_norm."
                raise ValueError(msg)
            cross_norm = _require_finite_float(self.cross_dense_sum_norm, "cross_dense_sum_norm")
            if not math.isclose(cross_norm, signal_norm, rel_tol=1e-12, abs_tol=1e-12):
                msg = "cross_dense_sum_norm does not match update_signal_norm."
                raise ValueError(msg)
        elif self.gradient_norm is not None or self.cross_dense_sum_norm is not None or signal:
            msg = "Initial trace records cannot contain an update signal or norm."
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-native trace record."""
        return {
            "schema_version": self.schema_version,
            "local_iteration": self.local_iteration,
            "global_iteration": self.global_iteration,
            "parameter_checksum": self.parameter_checksum,
            "learning_rate": self.learning_rate,
            "monitoring_loss": self.monitoring_loss,
            "monitoring_fidelity": self.monitoring_fidelity,
            "checkpoint_validation_fidelity": self.checkpoint_validation_fidelity,
            "update_signal": list(self.update_signal),
            "update_signal_kind": self.update_signal_kind,
            "update_signal_norm": self.update_signal_norm,
            "gradient_norm": self.gradient_norm,
            "cross_dense_sum_norm": self.cross_dense_sum_norm,
            "update_norm": self.update_norm,
            "trajectory_count": self.trajectory_count,
            "nonidentity_events": self.nonidentity_events,
            "training_ensemble_id": self.training_ensemble_id,
            "training_ensemble_checksum": self.training_ensemble_checksum,
            "checkpoint_validation_ensemble_checksum": self.checkpoint_validation_ensemble_checksum,
            "cumulative_work": self.cumulative_work.to_dict(),
            "training_ensemble_sampled": self.training_ensemble_sampled,
            "checkpoint_validation_ensemble_sampled": self.checkpoint_validation_ensemble_sampled,
            "cross_trajectory_pairings": self.cross_trajectory_pairings,
            "cumulative_cross_trajectory_pairings": self.cumulative_cross_trajectory_pairings,
        }


@dataclass(frozen=True, slots=True, init=False)
class NoisyKrotovCheckpointSelection:
    """Provenance-bound best-validation state carried across resume chunks."""

    stage_configuration_checksum: str
    circuit_binding_checksum: str
    provider_checksum: str | None
    objective_checksum: str
    global_iteration: int
    validation_fidelity: float
    parameter_checksum: str
    _theta_bytes: bytes = field(repr=False)
    schema_version: str = field(
        default=NOISY_KROTOV_CHECKPOINT_SELECTION_SCHEMA_VERSION,
        init=False,
    )

    def __init__(
        self,
        *,
        stage_configuration_checksum: str,
        circuit_binding_checksum: str,
        provider_checksum: str | None,
        objective_checksum: str,
        global_iteration: int,
        validation_fidelity: float,
        theta: NDArray[np.float64],
    ) -> None:
        """Validate and defensively freeze a checkpoint-selection state."""
        stage_checksum = require_checksum(stage_configuration_checksum, "stage_configuration_checksum")
        circuit_checksum = require_checksum(circuit_binding_checksum, "circuit_binding_checksum")
        if provider_checksum is not None:
            provider_checksum = require_checksum(provider_checksum, "provider_checksum")
        objective = require_checksum(objective_checksum, "objective_checksum")
        iteration = _require_builtin_int(global_iteration, "global_iteration")
        fidelity = _require_finite_float(validation_fidelity, "validation_fidelity")
        if not -1e-10 <= fidelity <= 1.0 + 1e-10:
            msg = "validation_fidelity must lie in [0, 1]."
            raise ValueError(msg)
        fidelity = min(1.0, max(0.0, fidelity))
        if not isinstance(theta, np.ndarray):
            msg = "theta must be a NumPy array."
            raise TypeError(msg)
        vector = np.asarray(theta, dtype=np.float64)
        if vector.ndim != 1 or not np.all(np.isfinite(vector)):
            msg = "theta must be a finite one-dimensional parameter vector."
            raise ValueError(msg)
        object.__setattr__(self, "stage_configuration_checksum", stage_checksum)
        object.__setattr__(self, "circuit_binding_checksum", circuit_checksum)
        object.__setattr__(self, "provider_checksum", provider_checksum)
        object.__setattr__(self, "objective_checksum", objective)
        object.__setattr__(self, "global_iteration", iteration)
        object.__setattr__(self, "validation_fidelity", fidelity)
        object.__setattr__(self, "parameter_checksum", _vector_checksum(vector))
        object.__setattr__(self, "_theta_bytes", _vector_bytes(vector))
        object.__setattr__(self, "schema_version", NOISY_KROTOV_CHECKPOINT_SELECTION_SCHEMA_VERSION)

    @property
    def theta(self) -> NDArray[np.float64]:
        """A detached copy of the selected parameter vector."""
        return _copy_vector(self._theta_bytes)

    def identity_payload(self) -> dict[str, object]:
        """Return all checksum-covered selection metadata."""
        return {
            "schema_version": self.schema_version,
            "stage_configuration_checksum": self.stage_configuration_checksum,
            "circuit_binding_checksum": self.circuit_binding_checksum,
            "provider_checksum": self.provider_checksum,
            "objective_checksum": self.objective_checksum,
            "global_iteration": self.global_iteration,
            "validation_fidelity": self.validation_fidelity,
            "parameter_checksum": self.parameter_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Deterministic checksum of the resume-selection state."""
        return canonical_checksum(self.identity_payload())

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed selection metadata."""
        payload = self.identity_payload()
        return {**payload, "content_checksum": self.content_checksum}


@dataclass(frozen=True, slots=True)
class NoisyKrotovResumeState:
    """Provenance-bound state required to continue a completed stage chunk."""

    stage_configuration_checksum: str
    circuit_binding_checksum: str
    provider_checksum: str | None
    objective_checksum: str
    completed_global_iteration: int
    final_parameter_checksum: str
    checkpoint_selection: NoisyKrotovCheckpointSelection | None
    cumulative_work: KrotovWorkLedger
    cumulative_cross_trajectory_pairings: int
    schema_version: str = field(default=NOISY_KROTOV_RESUME_STATE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate provenance, progress, selection, and cumulative work."""
        object.__setattr__(
            self,
            "stage_configuration_checksum",
            require_checksum(self.stage_configuration_checksum, "stage_configuration_checksum"),
        )
        object.__setattr__(
            self,
            "circuit_binding_checksum",
            require_checksum(self.circuit_binding_checksum, "circuit_binding_checksum"),
        )
        if self.provider_checksum is not None:
            object.__setattr__(
                self,
                "provider_checksum",
                require_checksum(self.provider_checksum, "provider_checksum"),
            )
        object.__setattr__(
            self,
            "objective_checksum",
            require_checksum(self.objective_checksum, "objective_checksum"),
        )
        object.__setattr__(
            self,
            "completed_global_iteration",
            _require_builtin_int(self.completed_global_iteration, "completed_global_iteration", minimum=1),
        )
        object.__setattr__(
            self,
            "final_parameter_checksum",
            require_checksum(self.final_parameter_checksum, "final_parameter_checksum"),
        )
        if self.checkpoint_selection is not None:
            if not isinstance(self.checkpoint_selection, NoisyKrotovCheckpointSelection):
                msg = "checkpoint_selection must be a NoisyKrotovCheckpointSelection or None."
                raise TypeError(msg)
            selection_provenance = (
                self.checkpoint_selection.stage_configuration_checksum,
                self.checkpoint_selection.circuit_binding_checksum,
                self.checkpoint_selection.provider_checksum,
                self.checkpoint_selection.objective_checksum,
            )
            own_provenance = (
                self.stage_configuration_checksum,
                self.circuit_binding_checksum,
                self.provider_checksum,
                self.objective_checksum,
            )
            if selection_provenance != own_provenance:
                msg = "checkpoint_selection provenance does not match the resume state."
                raise ValueError(msg)
            if self.checkpoint_selection.global_iteration > self.completed_global_iteration:
                msg = "checkpoint_selection cannot come from a future iteration."
                raise ValueError(msg)
        if not isinstance(self.cumulative_work, KrotovWorkLedger):
            msg = "cumulative_work must be a KrotovWorkLedger."
            raise TypeError(msg)
        object.__setattr__(
            self,
            "cumulative_cross_trajectory_pairings",
            _require_builtin_int(
                self.cumulative_cross_trajectory_pairings,
                "cumulative_cross_trajectory_pairings",
            ),
        )

    def identity_payload(self) -> dict[str, object]:
        """Return all checksum-covered resume metadata."""
        return {
            "schema_version": self.schema_version,
            "stage_configuration_checksum": self.stage_configuration_checksum,
            "circuit_binding_checksum": self.circuit_binding_checksum,
            "provider_checksum": self.provider_checksum,
            "objective_checksum": self.objective_checksum,
            "completed_global_iteration": self.completed_global_iteration,
            "final_parameter_checksum": self.final_parameter_checksum,
            "checkpoint_selection_checksum": (
                None if self.checkpoint_selection is None else self.checkpoint_selection.content_checksum
            ),
            "cumulative_work": self.cumulative_work.to_dict(),
            "cumulative_cross_trajectory_pairings": self.cumulative_cross_trajectory_pairings,
        }

    @property
    def content_checksum(self) -> str:
        """Deterministic checksum of the complete resume state."""
        return canonical_checksum(self.identity_payload())

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed resume metadata."""
        payload = self.identity_payload()
        return {**payload, "content_checksum": self.content_checksum}


def validate_noisy_krotov_execution_trace(
    *,
    stage: TrainingStageConfig,
    circuit_binding_document: Mapping[str, object],
    provider_checksum: str | None,
    trace: Sequence[NoisyKrotovIterationRecord],
    training_ensembles: Sequence[KrotovFixedMapEnsemble],
    validation_ensembles: Sequence[KrotovFixedMapEnsemble],
    normalized_work: KrotovWorkLedger,
    input_resume_state: NoisyKrotovResumeState | None,
) -> None:
    """Recompute exact adapter work, map use, and validation cadence."""
    reconstructed_binding = decode_noisy_krotov_circuit_binding_document(circuit_binding_document)
    binding_document = freeze_json_mapping(reconstructed_binding.to_dict(), "circuit_binding_document")
    training_translation = translate_fixed_rate_krotov_stage(stage, reconstructed_binding)
    if provider_checksum != training_translation.provider_checksum:
        msg = "Execution provider checksum does not match the stage-derived training-noise provider."
        raise ValueError(msg)
    validation_provider_checksum = None
    if stage.checkpoint_validation.enabled:
        validation_provider_checksum = _validation_translation(
            stage,
            reconstructed_binding,
            training_translation,
        ).provider_checksum
    offset = trace[0].global_iteration
    if tuple(row.local_iteration for row in trace) != tuple(range(len(trace))) or tuple(
        row.global_iteration for row in trace
    ) != tuple(range(offset, offset + len(trace))):
        msg = "Trace iterations must form one contiguous local and stage-global execution chunk."
        raise ValueError(msg)
    if trace[-1].global_iteration > stage.iteration_budget:
        msg = "Trace progress exceeds the configured stage budget."
        raise ValueError(msg)
    gates = binding_document["gates"]
    if type(gates) is not tuple:
        msg = "circuit_binding_document.gates must be a serialized sequence."
        raise TypeError(msg)
    gate_count = len(gates)
    trainable_gate_count = sum(cast("Mapping[str, object]", gate).get("param_index") is not None for gate in gates)
    expected_work = KrotovWorkLedger() if input_resume_state is None else input_resume_state.cumulative_work
    expected_pairings = 0 if input_resume_state is None else input_resume_state.cumulative_cross_trajectory_pairings
    training_by_checksum = {ensemble.content_checksum: ensemble for ensemble in training_ensembles}
    validation_by_checksum = {ensemble.content_checksum: ensemble for ensemble in validation_ensembles}
    if len(training_by_checksum) != len(training_ensembles) or len(validation_by_checksum) != len(validation_ensembles):
        msg = "Execution fixed-map collections must not repeat ensemble content."
        raise ValueError(msg)
    seen_training: set[str] = set()
    seen_validation: set[str] = set()
    noisy_training = stage.training_noise_id != NOISELESS_NOISE_ID
    validation_count = 0 if not stage.checkpoint_validation.enabled else stage.checkpoint_validation.trajectory_count
    validation_call_index = (
        0
        if input_resume_state is None or not stage.checkpoint_validation.enabled
        else offset // cast("int", stage.checkpoint_validation.cadence) + 1
    )
    for index, row in enumerate(trace):
        if row.trajectory_count != stage.trajectory_count:
            msg = f"Trace row {index} does not use the configured training trajectory count."
            raise ValueError(msg)
        training_checksum = row.training_ensemble_checksum
        if (row.training_ensemble_id is None) != (training_checksum is None):
            msg = "Training ensemble identity and checksum must be present together."
            raise ValueError(msg)
        if noisy_training:
            ensemble = None if training_checksum is None else training_by_checksum.get(training_checksum)
            if ensemble is None or ensemble.ensemble_id != row.training_ensemble_id:
                msg = f"Trace row {index} does not reference an exact training ensemble."
                raise ValueError(msg)
            if row.nonidentity_events != ensemble.nonidentity_event_count:
                msg = f"Trace row {index} does not match its training-ensemble event count."
                raise ValueError(msg)
            training_coordinate = offset if index == 0 else row.global_iteration - 1
            expected_ensemble_index, expected_refresh_index, expected_window_start = _schedule_point(
                stage.sampling_policy,
                stage.crn_refresh_interval,
                training_coordinate,
            )
            if (
                ensemble.role != "training_trajectory"
                or ensemble.stage_index != stage.stage_index
                or ensemble.stage_id != stage.stage_id
                or ensemble.stage_configuration_checksum != stage.configuration_checksum
                or ensemble.circuit_checksum != binding_document["content_checksum"]
                or ensemble.provider_checksum != provider_checksum
                or ensemble.trajectory_count != stage.trajectory_count
                or ensemble.gate_count != gate_count
                or ensemble.ensemble_index != expected_ensemble_index
                or ensemble.refresh_index != expected_refresh_index
                or ensemble.global_iteration_start != expected_window_start
            ):
                msg = f"Trace row {index} training ensemble does not match its exact schedule coordinate."
                raise ValueError(msg)
            if row.training_ensemble_sampled:
                if training_checksum in seen_training or training_coordinate != expected_window_start:
                    msg = "A training ensemble cannot be sampled after its first use."
                    raise ValueError(msg)
                expected_work = expected_work.plus(
                    training_trajectories=stage.trajectory_count,
                    trajectory_gate_applications=stage.trajectory_count * gate_count,
                )
            seen_training.add(cast("str", training_checksum))
        elif training_checksum is not None or row.training_ensemble_sampled or row.nonidentity_events != 0:
            msg = "Noiseless training cannot claim a training map, sampling, or noise events."
            raise ValueError(msg)

        training_calls = 1 if index == 0 else 2
        if noisy_training:
            expected_work = expected_work.plus(
                objective_evaluations=training_calls,
                gradient_evaluations=(0 if index == 0 or stage.trajectory_update == "cross" else 1),
                training_trajectories=training_calls * stage.trajectory_count,
                trajectory_gate_applications=training_calls * stage.trajectory_count * gate_count,
            )
        else:
            expected_work = expected_work.plus(
                objective_evaluations=training_calls,
                gradient_evaluations=(0 if index == 0 else 1),
            )

        row_pairings = (
            stage.trajectory_count**2 * trainable_gate_count if index > 0 and stage.trajectory_update == "cross" else 0
        )
        expected_pairings += row_pairings
        if (
            row.cross_trajectory_pairings != row_pairings
            or row.cumulative_cross_trajectory_pairings != expected_pairings
        ):
            msg = f"Trace row {index} has incorrect dense cross-trajectory work."
            raise ValueError(msg)
        expected_kind: UpdateSignalKind = (
            "none"
            if index == 0
            else (
                "cross_dense_sum_update"
                if stage.trajectory_update == "cross"
                else (
                    "independent_pathwise_gradient"
                    if stage.max_bond_dimension is None and not stage.svd_threshold
                    else "independent_pathwise_update"
                )
            )
        )
        if row.update_signal_kind != expected_kind:
            msg = f"Trace row {index} update kind does not match the configured optimizer path."
            raise ValueError(msg)
        if index == 0:
            if row.update_signal or row.learning_rate or row.update_norm:
                msg = "Initial trace rows require an empty zero-rate, zero-norm update."
                raise ValueError(msg)
        else:
            if len(row.update_signal) != stage.output_parameter_count:
                msg = f"Trace row {index} update signal does not match the output parameter count."
                raise ValueError(msg)
            base_rate = _require_finite_float(
                stage.optimizer_hyperparameters["learning_rate"],
                "optimizer_hyperparameters.learning_rate",
                positive=True,
            )
            schedule = stage.optimizer_hyperparameters.get("schedule", "constant")
            decay = _require_finite_float(
                stage.optimizer_hyperparameters.get("decay", 0.0),
                "optimizer_hyperparameters.decay",
            )
            global_update_index = row.global_iteration - 1
            if schedule == "constant":
                expected_rate = base_rate
            elif schedule == "inverse":
                expected_rate = base_rate / (1.0 + decay * global_update_index)
            else:
                expected_rate = base_rate * float(np.exp(-decay * global_update_index))
            if not math.isclose(row.learning_rate, expected_rate, rel_tol=1e-12, abs_tol=1e-12):
                msg = f"Trace row {index} learning rate does not match the configured schedule."
                raise ValueError(msg)
            if not math.isclose(
                row.update_norm,
                abs(expected_rate) * row.update_signal_norm,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                msg = f"Trace row {index} update norm does not match its rate and update signal."
                raise ValueError(msg)

        validation_checksum = row.checkpoint_validation_ensemble_checksum
        cadence = stage.checkpoint_validation.cadence
        should_validate = stage.checkpoint_validation.enabled and (
            (index == 0 and offset == 0)
            or (
                index > 0
                and (row.global_iteration == stage.iteration_budget or row.global_iteration % cast("int", cadence) == 0)
            )
        )
        if should_validate != (row.checkpoint_validation_fidelity is not None):
            msg = f"Trace row {index} does not implement the checkpoint-validation cadence."
            raise ValueError(msg)
        if should_validate:
            ensemble = None if validation_checksum is None else validation_by_checksum.get(validation_checksum)
            if ensemble is None:
                msg = f"Trace row {index} does not reference an exact validation ensemble."
                raise ValueError(msg)
            expected_ensemble_index, expected_refresh_index, expected_window_start = _schedule_point(
                stage.checkpoint_validation.sampling_policy,
                stage.checkpoint_validation.ensemble_refresh_interval,
                validation_call_index,
            )
            if (
                ensemble.role != "checkpoint_validation"
                or ensemble.stage_index != stage.stage_index
                or ensemble.stage_id != stage.stage_id
                or ensemble.stage_configuration_checksum != stage.configuration_checksum
                or ensemble.circuit_checksum != binding_document["content_checksum"]
                or ensemble.provider_checksum != validation_provider_checksum
                or ensemble.trajectory_count != validation_count
                or ensemble.gate_count != gate_count
                or ensemble.ensemble_index != expected_ensemble_index
                or ensemble.refresh_index != expected_refresh_index
                or ensemble.global_iteration_start != expected_window_start
            ):
                msg = f"Trace row {index} validation ensemble does not match its exact call schedule."
                raise ValueError(msg)
            if row.checkpoint_validation_ensemble_sampled:
                if validation_checksum in seen_validation or validation_call_index != expected_window_start:
                    msg = "A validation ensemble cannot be sampled after its first use."
                    raise ValueError(msg)
                expected_work = expected_work.plus(
                    checkpoint_validation_trajectories=validation_count,
                    trajectory_gate_applications=validation_count * gate_count,
                )
            seen_validation.add(cast("str", validation_checksum))
            expected_work = expected_work.plus(
                objective_evaluations=1,
                checkpoint_validation_trajectories=validation_count,
                trajectory_gate_applications=validation_count * gate_count,
            )
            validation_call_index += 1
        elif validation_checksum is not None or row.checkpoint_validation_ensemble_sampled:
            msg = "A non-validation trace row cannot claim a validation map or sampling."
            raise ValueError(msg)
        if row.cumulative_work != expected_work:
            msg = f"Trace row {index} cumulative work is not implied by its recorded operations."
            raise ValueError(msg)
    if expected_work != normalized_work:
        msg = "normalized_work is not the exact trace-derived final work ledger."
        raise ValueError(msg)
    if seen_training != set(training_by_checksum) or seen_validation != set(validation_by_checksum):
        msg = "Trace does not account for every execution fixed-map ensemble."
        raise ValueError(msg)


@dataclass(frozen=True, slots=True, init=False)
class NoisyKrotovStageExecution:
    """Immutable successful in-memory result of a WP17 stage.

    WP18 is responsible for persisting this result as a checkpoint and
    converting it to :class:`TrainingStageResult`.
    """

    stage_index: int
    stage_id: str
    stage_configuration_checksum: str
    circuit_binding_checksum: str
    provider_checksum: str | None
    objective_checksum: str
    objective_binding: NoisyKrotovObjectiveBinding
    circuit_binding_document: Mapping[str, object]
    initial_parameter_checksum: str
    final_parameter_checksum: str
    selected_parameter_checksum: str
    selected_global_iteration: int
    selected_checkpoint_validation_fidelity: float | None
    trace: tuple[NoisyKrotovIterationRecord, ...]
    training_ensembles: tuple[KrotovFixedMapEnsemble, ...]
    checkpoint_validation_ensembles: tuple[KrotovFixedMapEnsemble, ...]
    normalized_work: Mapping[str, object]
    input_resume_state_checksum: str | None
    _initial_theta_bytes: bytes = field(repr=False)
    _final_theta_bytes: bytes = field(repr=False)
    _selected_theta_bytes: bytes = field(repr=False)
    _input_resume_state: NoisyKrotovResumeState | None = field(repr=False)
    schema_version: str = field(default=NOISY_KROTOV_EXECUTION_SCHEMA_VERSION, init=False)

    def __init__(
        self,
        *,
        stage: TrainingStageConfig,
        circuit_binding_checksum: str,
        circuit_binding_document: Mapping[str, object],
        provider_checksum: str | None,
        objective_binding: NoisyKrotovObjectiveBinding,
        initial_theta: NDArray[np.float64],
        final_theta: NDArray[np.float64],
        selected_theta: NDArray[np.float64],
        selected_global_iteration: int,
        selected_checkpoint_validation_fidelity: float | None,
        trace: Sequence[NoisyKrotovIterationRecord],
        training_ensembles: Sequence[KrotovFixedMapEnsemble],
        checkpoint_validation_ensembles: Sequence[KrotovFixedMapEnsemble],
        normalized_work: KrotovWorkLedger,
        input_resume_state: NoisyKrotovResumeState | None = None,
    ) -> None:
        """Freeze detached arrays, trace, ensembles, and work."""
        initial = np.asarray(initial_theta, dtype=np.float64)
        final = np.asarray(final_theta, dtype=np.float64)
        selected = np.asarray(selected_theta, dtype=np.float64)
        if initial.shape != final.shape or final.shape != selected.shape:
            msg = "Initial, final, and selected parameter vectors must have identical shapes."
            raise ValueError(msg)
        object.__setattr__(self, "stage_index", stage.stage_index)
        object.__setattr__(self, "stage_id", stage.stage_id)
        object.__setattr__(self, "stage_configuration_checksum", stage.configuration_checksum)
        binding_checksum = require_checksum(circuit_binding_checksum, "circuit_binding_checksum")
        reconstructed_binding = decode_noisy_krotov_circuit_binding_document(circuit_binding_document)
        binding_document = freeze_json_mapping(reconstructed_binding.to_dict(), "circuit_binding_document")
        if reconstructed_binding.content_checksum != binding_checksum:
            msg = "circuit_binding_document does not reproduce circuit_binding_checksum."
            raise ValueError(msg)
        if (
            binding_document.get("topology_id") != stage.output_topology_id
            or binding_document.get("num_params") != stage.output_parameter_count
        ):
            msg = "circuit_binding_document does not match the stage output topology."
            raise ValueError(msg)
        object.__setattr__(self, "circuit_binding_checksum", binding_checksum)
        object.__setattr__(self, "circuit_binding_document", binding_document)
        object.__setattr__(self, "provider_checksum", provider_checksum)
        if not isinstance(objective_binding, NoisyKrotovObjectiveBinding):
            msg = "objective_binding must be a NoisyKrotovObjectiveBinding."
            raise TypeError(msg)
        objective_checksum = objective_binding.objective_checksum
        object.__setattr__(self, "objective_checksum", objective_checksum)
        object.__setattr__(self, "objective_binding", objective_binding)
        object.__setattr__(self, "initial_parameter_checksum", _vector_checksum(initial))
        object.__setattr__(self, "final_parameter_checksum", _vector_checksum(final))
        object.__setattr__(self, "selected_parameter_checksum", _vector_checksum(selected))
        object.__setattr__(self, "selected_global_iteration", selected_global_iteration)
        if selected_checkpoint_validation_fidelity is not None:
            selected_checkpoint_validation_fidelity = _require_finite_float(
                selected_checkpoint_validation_fidelity,
                "selected_checkpoint_validation_fidelity",
            )
            if not -1e-10 <= selected_checkpoint_validation_fidelity <= 1.0 + 1e-10:
                msg = "selected_checkpoint_validation_fidelity must lie in [0, 1]."
                raise ValueError(msg)
            selected_checkpoint_validation_fidelity = min(
                1.0,
                max(0.0, selected_checkpoint_validation_fidelity),
            )
        object.__setattr__(
            self,
            "selected_checkpoint_validation_fidelity",
            selected_checkpoint_validation_fidelity,
        )
        trace_rows = tuple(trace)
        if not trace_rows or not all(isinstance(row, NoisyKrotovIterationRecord) for row in trace_rows):
            msg = "trace must contain only NoisyKrotovIterationRecord values."
            raise TypeError(msg)
        if trace_rows[0].parameter_checksum != _vector_checksum(initial) or trace_rows[
            -1
        ].parameter_checksum != _vector_checksum(final):
            msg = "Trace endpoint parameter checksums do not match initial_theta and final_theta."
            raise ValueError(msg)
        offset = trace_rows[0].global_iteration
        if offset == 0:
            if input_resume_state is not None:
                msg = "An initial execution chunk cannot carry an input resume state."
                raise ValueError(msg)
            inherited = None
        else:
            if not isinstance(input_resume_state, NoisyKrotovResumeState):
                msg = "A resumed execution chunk requires its verified input resume state."
                raise ValueError(msg)
            expected_resume_provenance = (
                stage.configuration_checksum,
                binding_checksum,
                provider_checksum,
                objective_checksum,
            )
            actual_resume_provenance = (
                input_resume_state.stage_configuration_checksum,
                input_resume_state.circuit_binding_checksum,
                input_resume_state.provider_checksum,
                input_resume_state.objective_checksum,
            )
            if (
                actual_resume_provenance != expected_resume_provenance
                or input_resume_state.completed_global_iteration != offset
                or input_resume_state.final_parameter_checksum != _vector_checksum(initial)
            ):
                msg = "Input resume state does not bind the resumed execution boundary."
                raise ValueError(msg)
            inherited = input_resume_state.checkpoint_selection
        selected_rows = tuple(row for row in trace_rows if row.global_iteration == selected_global_iteration)
        if selected_rows:
            if len(selected_rows) != 1 or selected_rows[0].parameter_checksum != _vector_checksum(selected):
                msg = "Trace does not uniquely bind the current selected checkpoint."
                raise ValueError(msg)
        else:
            if not isinstance(inherited, NoisyKrotovCheckpointSelection):
                msg = "A selected checkpoint predating this trace requires its verified inherited selection."
                raise ValueError(msg)
            expected_provenance = (
                stage.configuration_checksum,
                binding_checksum,
                provider_checksum,
                objective_checksum,
            )
            actual_provenance = (
                inherited.stage_configuration_checksum,
                inherited.circuit_binding_checksum,
                inherited.provider_checksum,
                inherited.objective_checksum,
            )
            if (
                inherited.global_iteration >= trace_rows[0].global_iteration
                or actual_provenance != expected_provenance
                or inherited.global_iteration != selected_global_iteration
                or inherited.validation_fidelity != selected_checkpoint_validation_fidelity
                or inherited.parameter_checksum != _vector_checksum(selected)
            ):
                msg = "Inherited checkpoint selection does not bind the selected stage state."
                raise ValueError(msg)
        validation_candidates = tuple(
            (
                row.checkpoint_validation_fidelity,
                row.global_iteration,
                row.parameter_checksum,
            )
            for row in trace_rows
            if row.checkpoint_validation_fidelity is not None
        ) + (
            ()
            if inherited is None
            else (
                (
                    inherited.validation_fidelity,
                    inherited.global_iteration,
                    inherited.parameter_checksum,
                ),
            )
        )
        if stage.checkpoint_validation.enabled:
            if selected_checkpoint_validation_fidelity is None or not validation_candidates:
                msg = "Checkpoint validation requires trace-backed or inherited selection evidence."
                raise ValueError(msg)
            best_fidelity = max(item[0] for item in validation_candidates)
            best_candidates = tuple(item for item in validation_candidates if item[0] == best_fidelity)
            winner = (
                min(best_candidates, key=operator.itemgetter(1))
                if stage.checkpoint_validation.tie_breaker == "earliest_iteration"
                else max(best_candidates, key=operator.itemgetter(1))
            )
            if winner != (
                selected_checkpoint_validation_fidelity,
                selected_global_iteration,
                _vector_checksum(selected),
            ):
                msg = "Selected checkpoint does not implement validation winner and tie-break semantics."
                raise ValueError(msg)
        elif inherited is not None or selected_checkpoint_validation_fidelity is not None:
            msg = "A stage without checkpoint validation cannot carry checkpoint-selection evidence."
            raise ValueError(msg)
        training_maps = tuple(training_ensembles)
        validation_maps = tuple(checkpoint_validation_ensembles)
        if not all(isinstance(item, KrotovFixedMapEnsemble) for item in (*training_maps, *validation_maps)):
            msg = "Execution map collections must contain only KrotovFixedMapEnsemble values."
            raise TypeError(msg)
        validate_noisy_krotov_execution_trace(
            stage=stage,
            circuit_binding_document=binding_document,
            provider_checksum=provider_checksum,
            trace=trace_rows,
            training_ensembles=training_maps,
            validation_ensembles=validation_maps,
            normalized_work=normalized_work,
            input_resume_state=input_resume_state,
        )
        object.__setattr__(self, "trace", trace_rows)
        object.__setattr__(self, "training_ensembles", training_maps)
        object.__setattr__(
            self,
            "checkpoint_validation_ensembles",
            validation_maps,
        )
        object.__setattr__(
            self,
            "normalized_work",
            freeze_json_mapping(normalized_work.to_dict(), "normalized_work"),
        )
        object.__setattr__(
            self,
            "input_resume_state_checksum",
            None if input_resume_state is None else input_resume_state.content_checksum,
        )
        object.__setattr__(self, "_input_resume_state", input_resume_state)
        object.__setattr__(self, "_initial_theta_bytes", _vector_bytes(initial))
        object.__setattr__(self, "_final_theta_bytes", _vector_bytes(final))
        object.__setattr__(self, "_selected_theta_bytes", _vector_bytes(selected))
        object.__setattr__(self, "schema_version", NOISY_KROTOV_EXECUTION_SCHEMA_VERSION)

    @property
    def initial_theta(self) -> NDArray[np.float64]:
        """A detached copy of the resolved input parameters."""
        return _copy_vector(self._initial_theta_bytes)

    @property
    def final_theta(self) -> NDArray[np.float64]:
        """A detached copy of the last-iteration parameters."""
        return _copy_vector(self._final_theta_bytes)

    @property
    def selected_theta(self) -> NDArray[np.float64]:
        """A detached copy of the checkpoint-selected parameters."""
        return _copy_vector(self._selected_theta_bytes)

    @property
    def checkpoint_selection(self) -> NoisyKrotovCheckpointSelection | None:
        """The provenance-bound selection state needed for a resumed chunk."""
        if self.selected_checkpoint_validation_fidelity is None:
            return None
        return NoisyKrotovCheckpointSelection(
            stage_configuration_checksum=self.stage_configuration_checksum,
            circuit_binding_checksum=self.circuit_binding_checksum,
            provider_checksum=self.provider_checksum,
            objective_checksum=self.objective_checksum,
            global_iteration=self.selected_global_iteration,
            validation_fidelity=self.selected_checkpoint_validation_fidelity,
            theta=self.selected_theta,
        )

    @property
    def resume_state(self) -> NoisyKrotovResumeState:
        """The provenance-bound state needed to continue after this chunk."""
        final_record = self.trace[-1]
        return NoisyKrotovResumeState(
            stage_configuration_checksum=self.stage_configuration_checksum,
            circuit_binding_checksum=self.circuit_binding_checksum,
            provider_checksum=self.provider_checksum,
            objective_checksum=self.objective_checksum,
            completed_global_iteration=final_record.global_iteration,
            final_parameter_checksum=self.final_parameter_checksum,
            checkpoint_selection=self.checkpoint_selection,
            cumulative_work=final_record.cumulative_work,
            cumulative_cross_trajectory_pairings=final_record.cumulative_cross_trajectory_pairings,
        )

    @property
    def training_ensemble_checksums(self) -> tuple[str, ...]:
        """Ordered unique training ensemble checksums."""
        return tuple(ensemble.content_checksum for ensemble in self.training_ensembles)

    @property
    def checkpoint_validation_ensemble_checksums(self) -> tuple[str, ...]:
        """Ordered unique validation ensemble checksums."""
        return tuple(ensemble.content_checksum for ensemble in self.checkpoint_validation_ensembles)

    @property
    def cross_trajectory_pairings(self) -> int:
        """Stage-global dense trajectory pairs evaluated through this chunk."""
        return 0 if not self.trace else self.trace[-1].cumulative_cross_trajectory_pairings

    def identity_payload(self) -> dict[str, object]:
        """Return successful stage content excluding full map payloads."""
        return {
            "schema_version": self.schema_version,
            "adapter_version": NOISY_KROTOV_ADAPTER_VERSION,
            "stage_index": self.stage_index,
            "stage_id": self.stage_id,
            "stage_configuration_checksum": self.stage_configuration_checksum,
            "circuit_binding_checksum": self.circuit_binding_checksum,
            "provider_checksum": self.provider_checksum,
            "objective_checksum": self.objective_checksum,
            "objective_binding_checksum": self.objective_binding.content_checksum,
            "initial_parameter_checksum": self.initial_parameter_checksum,
            "final_parameter_checksum": self.final_parameter_checksum,
            "selected_parameter_checksum": self.selected_parameter_checksum,
            "selected_global_iteration": self.selected_global_iteration,
            "selected_checkpoint_validation_fidelity": self.selected_checkpoint_validation_fidelity,
            "trace": [row.to_dict() for row in self.trace],
            "training_ensemble_checksums": list(self.training_ensemble_checksums),
            "checkpoint_validation_ensemble_checksums": list(self.checkpoint_validation_ensemble_checksums),
            "normalized_work": thaw_json_mapping(self.normalized_work),
            "input_resume_state_checksum": self.input_resume_state_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Deterministic successful-execution content checksum."""
        return canonical_checksum(self.identity_payload())

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed execution metadata."""
        payload = self.identity_payload()
        return {**payload, "content_checksum": self.content_checksum}


@dataclass(frozen=True, slots=True)
class NoisyKrotovStageFailure:
    """Structured, non-artifact failure returned by the WP17 adapter."""

    stage_index: int
    stage_id: str
    stage_configuration_checksum: str
    phase: FailurePhase
    exception_type: str
    message: str
    traceback_text: str
    retryable: bool
    partial_work: Mapping[str, object]
    schema_version: str = field(default=NOISY_KROTOV_FAILURE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate and freeze failure diagnostics."""
        _require_builtin_int(self.stage_index, "stage_index")
        if self.phase not in _FAILURE_PHASES:
            msg = f"phase must be one of {sorted(_FAILURE_PHASES)!r}."
            raise ValueError(msg)
        for name in ("stage_id", "stage_configuration_checksum", "exception_type", "message", "traceback_text"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                msg = f"{name} must be a nonempty string."
                raise TypeError(msg)
        if type(self.retryable) is not bool:
            msg = "retryable must be a bool."
            raise TypeError(msg)
        object.__setattr__(
            self,
            "partial_work",
            freeze_json_mapping(self.partial_work, "partial_work"),
        )

    def _content_dict(self) -> dict[str, object]:
        """Return all checksum-covered failure diagnostics."""
        return {
            "schema_version": self.schema_version,
            "adapter_version": NOISY_KROTOV_ADAPTER_VERSION,
            "stage_index": self.stage_index,
            "stage_id": self.stage_id,
            "stage_configuration_checksum": self.stage_configuration_checksum,
            "phase": self.phase,
            "exception_type": self.exception_type,
            "message": self.message,
            "traceback_text": self.traceback_text,
            "retryable": self.retryable,
            "partial_work": thaw_json_mapping(self.partial_work),
        }

    @property
    def content_checksum(self) -> str:
        """Deterministic checksum sealing the structured failure."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return detached checksum-sealed JSON-native failure diagnostics."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}


def _provider_checksum(provider: GateNoiseProvider) -> str:
    """Return the required stable provider checksum.

    Raises:
        TypeError: If the provider has no string ``content_checksum``.
    """
    checksum = getattr(provider, "content_checksum", None)
    if type(checksum) is not str:
        msg = "Fixed-rate noise providers must expose a stable content_checksum."
        raise TypeError(msg)
    return cast("str", checksum)


def translate_fixed_rate_krotov_stage(
    stage: TrainingStageConfig,
    circuit_binding: NoisyKrotovCircuitBinding,
) -> KrotovStageTranslation:
    """Translate a validated WP16 stage into Krotov options and provider.

    Args:
        stage: Fully resolved Phase II stage configuration.
        circuit_binding: Exact logical circuit and placement-policy binding.

    Returns:
        Frozen translation used by :class:`FixedRateNoisyKrotovStageAdapter`.

    Raises:
        TypeError: If inputs have the wrong type or hyperparameters have invalid
            scalar types.
        ValueError: If the stage requests an unsupported optimizer, provider,
            profile version, topology, or hyperparameter.
    """
    if not isinstance(stage, TrainingStageConfig):
        msg = "stage must be a TrainingStageConfig."
        raise TypeError(msg)
    if not isinstance(circuit_binding, NoisyKrotovCircuitBinding):
        msg = "circuit_binding must be a NoisyKrotovCircuitBinding."
        raise TypeError(msg)
    if stage.training_noise_id == BALLARIN_NOISE_ID:
        msg = "ballarin_coupled is evaluation-only and cannot enter noisy Krotov training."
        raise ValueError(msg)
    if stage.optimizer_id != "krotov":
        msg = "WP17 supports only optimizer_id='krotov'."
        raise ValueError(msg)
    circuit = circuit_binding.circuit
    if stage.output_topology_id != circuit_binding.topology_id:
        msg = "Stage output topology does not match the bound circuit topology."
        raise ValueError(msg)
    if stage.output_parameter_count != circuit.num_params:
        msg = "Stage output parameter count does not match the bound circuit."
        raise ValueError(msg)

    hyperparameters = dict(stage.optimizer_hyperparameters)
    unknown = set(hyperparameters) - _SUPPORTED_OPTIMIZER_HYPERPARAMETERS
    if unknown:
        msg = f"Unsupported Krotov optimizer hyperparameters: {sorted(unknown)!r}."
        raise ValueError(msg)
    if "learning_rate" not in hyperparameters:
        msg = "Krotov stages require an explicit learning_rate hyperparameter."
        raise ValueError(msg)
    learning_rate = _require_finite_float(hyperparameters["learning_rate"], "learning_rate", positive=True)
    schedule_value = hyperparameters.get("schedule", "constant")
    if type(schedule_value) is not str or schedule_value not in _SCHEDULES:
        msg = f"schedule must be one of {sorted(_SCHEDULES)!r}."
        raise ValueError(msg)
    schedule = cast("Literal['constant', 'inverse', 'exp']", schedule_value)
    decay = _require_finite_float(hyperparameters.get("decay", 0.0), "decay")
    if decay < 0.0:
        msg = "decay must be nonnegative."
        raise ValueError(msg)
    if schedule == "constant" and not math.isclose(decay, 0.0, rel_tol=0.0, abs_tol=0.0):
        msg = "A constant learning-rate schedule requires decay=0."
        raise ValueError(msg)

    truncation = KrotovTruncation(
        max_bond_dim=stage.max_bond_dimension,
        svd_threshold=stage.svd_threshold,
        trunc_mode=stage.truncation_mode,
        min_bond_dim=stage.min_bond_dimension,
    )
    options = KrotovOptions(
        max_iterations=stage.iteration_budget,
        batch_step_size=learning_rate,
        batch_schedule=schedule,
        batch_decay=decay,
        seed=cast("int", stage.optimizer_seed),
        truncation=truncation,
    )
    if stage.training_noise_id == NOISELESS_NOISE_ID:
        return KrotovStageTranslation(
            options=options,
            tjm_options=None,
            noise_provider=None,
            provider_checksum=None,
            learning_rate=learning_rate,
            schedule=schedule,
            decay=decay,
        )

    if stage.noise_strength_scale is None or stage.tjm_dt is None or stage.training_seed is None:
        msg = "Noisy Krotov stages require resolved positive scale, dt, and training seed."
        raise ValueError(msg)
    if stage.training_noise_id in STANDARD_NOISE_IDS:
        if stage.noise_definition_version != FIXED_RATE_NOISE_DEFINITION_VERSION:
            msg = (
                "Standard noisy Krotov stages require noise definition version "
                f"{FIXED_RATE_NOISE_DEFINITION_VERSION!r}."
            )
            raise ValueError(msg)
        provider = create_scaled_standard_noise_provider(
            stage.training_noise_id,
            stage.noise_strength_scale,
        )
    elif stage.training_noise_id == HISTORICAL_FIXED_RATE_NOISE_ID:
        if not math.isclose(stage.noise_strength_scale, 1.0, rel_tol=0.0, abs_tol=0.0):
            msg = "The frozen historical training profile requires noise_strength_scale=1.0."
            raise ValueError(msg)
        provider = create_historical_fixed_rate_noise_provider()
        if stage.noise_definition_version != provider.noise_definition_version:
            msg = f"Historical profile requires noise definition version {provider.noise_definition_version!r}."
            raise ValueError(msg)
        if not math.isclose(stage.tjm_dt, provider.tjm_dt, rel_tol=0.0, abs_tol=0.0):
            msg = f"Historical profile requires tjm_dt={provider.tjm_dt!r}."
            raise ValueError(msg)
    else:
        msg = f"Unsupported fixed-rate training noise identifier {stage.training_noise_id!r}."
        raise ValueError(msg)

    tjm_options = KrotovTJMOptions(
        num_trajectories=stage.trajectory_count,
        random_seed=stage.training_seed,
        dt=stage.tjm_dt,
        apply_noise_to="all",
        noisy_gate_indices=circuit_binding.noisy_gate_indices,
        trajectory_update=cast("Literal['independent', 'cross']", stage.trajectory_update),
        differentiate_jump_normalization=False,
        use_crn=False,
    )
    return KrotovStageTranslation(
        options=options,
        tjm_options=tjm_options,
        noise_provider=provider,
        provider_checksum=_provider_checksum(provider),
        learning_rate=learning_rate,
        schedule=schedule,
        decay=decay,
    )


def _learning_rate(translation: KrotovStageTranslation, global_update_index: int) -> float:
    """Evaluate the schedule at one zero-based global update index."""
    if translation.schedule == "constant":
        return translation.learning_rate
    if translation.schedule == "inverse":
        return translation.learning_rate / (1.0 + translation.decay * global_update_index)
    return translation.learning_rate * float(np.exp(-translation.decay * global_update_index))


def _gradient_is_established(translation: KrotovStageTranslation) -> bool:
    """Whether the configured fixed-map path is within the validated exact regime."""
    truncation = translation.options.truncation
    return truncation.max_bond_dim is None and math.isclose(
        truncation.svd_threshold,
        0.0,
        rel_tol=0.0,
        abs_tol=0.0,
    )


def _schedule_point(
    policy: str,
    refresh_interval: int | None,
    global_update_index: int,
) -> tuple[int, int, int]:
    """Return ensemble index, refresh index, and refresh-window start."""
    if policy == "crn_fixed":
        return 0, 0, 0
    if policy == "resampled":
        return global_update_index, global_update_index, global_update_index
    if policy == "crn_refresh":
        if refresh_interval is None:
            msg = "crn_refresh requires a refresh interval."
            raise ValueError(msg)
        refresh_index = global_update_index // refresh_interval
        return refresh_index, refresh_index, refresh_index * refresh_interval
    msg = f"Unsupported noisy sampling policy {policy!r}."
    raise ValueError(msg)


def _ensemble_cache(
    ensembles: Sequence[KrotovFixedMapEnsemble],
    *,
    role: KrotovMapRole,
) -> dict[tuple[int, int], KrotovFixedMapEnsemble]:
    """Build a strict replay cache from caller-supplied ensembles."""
    cache: dict[tuple[int, int], KrotovFixedMapEnsemble] = {}
    for index, ensemble in enumerate(ensembles):
        if not isinstance(ensemble, KrotovFixedMapEnsemble):
            msg = f"Replay ensemble {index} is not a KrotovFixedMapEnsemble."
            raise TypeError(msg)
        if ensemble.role != role:
            msg = f"Replay ensemble role {ensemble.role!r} does not match {role!r}."
            raise ValueError(msg)
        key = (ensemble.ensemble_index, ensemble.refresh_index)
        if key in cache:
            msg = f"Duplicate replay ensemble schedule point {key!r}."
            raise ValueError(msg)
        cache[key] = ensemble
    return cache


def _verify_resume_selection(
    selection: NoisyKrotovCheckpointSelection,
    *,
    stage: TrainingStageConfig,
    binding: NoisyKrotovCircuitBinding,
    provider_checksum: str | None,
    objective_checksum: str,
    global_iteration_offset: int,
) -> NDArray[np.float64]:
    """Verify and detach the best checkpoint carried from an earlier chunk."""
    if not isinstance(selection, NoisyKrotovCheckpointSelection):
        msg = "checkpoint_selection must be a NoisyKrotovCheckpointSelection."
        raise TypeError(msg)
    if selection.stage_configuration_checksum != stage.configuration_checksum:
        msg = "Resume checkpoint selection belongs to a different stage configuration."
        raise ValueError(msg)
    if selection.circuit_binding_checksum != binding.content_checksum:
        msg = "Resume checkpoint selection belongs to a different circuit binding."
        raise ValueError(msg)
    if selection.provider_checksum != provider_checksum:
        msg = "Resume checkpoint selection belongs to a different training-noise provider."
        raise ValueError(msg)
    if selection.objective_checksum != objective_checksum:
        msg = "Resume checkpoint selection belongs to a different target or initial state."
        raise ValueError(msg)
    if selection.global_iteration > global_iteration_offset:
        msg = "Resume checkpoint selection cannot come from a future stage iteration."
        raise ValueError(msg)
    cadence = cast("int", stage.checkpoint_validation.cadence)
    if selection.global_iteration != 0 and selection.global_iteration % cadence != 0:
        msg = "Resume checkpoint selection was not evaluated on the configured validation cadence."
        raise ValueError(msg)
    theta = selection.theta
    if theta.shape != (stage.output_parameter_count,):
        msg = "Resume checkpoint selection parameter count does not match the stage."
        raise ValueError(msg)
    return theta


def _verify_resume_state(
    resume_state: NoisyKrotovResumeState,
    *,
    stage: TrainingStageConfig,
    binding: NoisyKrotovCircuitBinding,
    circuit: ParameterizedCircuit,
    provider_checksum: str | None,
    objective_checksum: str,
    global_iteration_offset: int,
    theta: NDArray[np.float64],
) -> tuple[NDArray[np.float64] | None, float, int]:
    """Verify all provenance and progress carried by a resumed stage chunk."""
    if not isinstance(resume_state, NoisyKrotovResumeState):
        msg = "resume_state must be a NoisyKrotovResumeState."
        raise TypeError(msg)
    expected_provenance = (
        stage.configuration_checksum,
        binding.content_checksum,
        provider_checksum,
        objective_checksum,
    )
    actual_provenance = (
        resume_state.stage_configuration_checksum,
        resume_state.circuit_binding_checksum,
        resume_state.provider_checksum,
        resume_state.objective_checksum,
    )
    if actual_provenance != expected_provenance:
        msg = "Resume state belongs to a different stage, circuit, provider, target, or initial state."
        raise ValueError(msg)
    if resume_state.completed_global_iteration != global_iteration_offset:
        msg = "Resume state progress does not match global_iteration_offset."
        raise ValueError(msg)
    if resume_state.final_parameter_checksum != _vector_checksum(theta):
        msg = "Resume input parameters do not match the preceding chunk."
        raise ValueError(msg)
    trainable_gate_count = sum(gate.is_trainable for gate in circuit.gates)
    expected_cross_pairings = (
        global_iteration_offset * stage.trajectory_count**2 * trainable_gate_count
        if stage.trajectory_update == "cross"
        else 0
    )
    if resume_state.cumulative_cross_trajectory_pairings != expected_cross_pairings:
        msg = "Resume state cross-trajectory work does not match completed stage updates."
        raise ValueError(msg)
    selection = resume_state.checkpoint_selection
    if not stage.checkpoint_validation.enabled:
        if selection is not None:
            msg = "A stage without checkpoint validation cannot carry a checkpoint selection."
            raise ValueError(msg)
        return None, -math.inf, global_iteration_offset
    if selection is None:
        msg = "A resumed validation-enabled stage requires its prior best checkpoint selection."
        raise ValueError(msg)
    selected_theta = _verify_resume_selection(
        selection,
        stage=stage,
        binding=binding,
        provider_checksum=provider_checksum,
        objective_checksum=objective_checksum,
        global_iteration_offset=global_iteration_offset,
    )
    return selected_theta, selection.validation_fidelity, selection.global_iteration


def _verify_replay_binding(
    ensemble: KrotovFixedMapEnsemble,
    *,
    stage: TrainingStageConfig,
    binding: NoisyKrotovCircuitBinding,
    circuit: ParameterizedCircuit,
    provider_checksum: str,
    role: KrotovMapRole,
    resolved_seed: int,
    ensemble_index: int,
    refresh_index: int,
    global_iteration_start: int,
    trajectory_count: int,
) -> None:
    """Reject a caller-supplied ensemble from a different scientific context."""
    expected = {
        "role": role,
        "resolved_seed": resolved_seed,
        "stage_index": stage.stage_index,
        "stage_id": stage.stage_id,
        "stage_configuration_checksum": stage.configuration_checksum,
        "circuit_checksum": binding.content_checksum,
        "provider_checksum": provider_checksum,
        "ensemble_index": ensemble_index,
        "refresh_index": refresh_index,
        "global_iteration_start": global_iteration_start,
        "trajectory_count": trajectory_count,
        "gate_count": len(circuit.gates),
    }
    for name, value in expected.items():
        if getattr(ensemble, name) != value:
            msg = f"Replay ensemble {name} does not match the requested stage context."
            raise ValueError(msg)


def _sample_or_replay(
    *,
    cache: dict[tuple[int, int], KrotovFixedMapEnsemble],
    ordered: list[KrotovFixedMapEnsemble],
    stage: TrainingStageConfig,
    binding: NoisyKrotovCircuitBinding,
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    initial_state: MPS,
    translation: KrotovStageTranslation,
    role: KrotovMapRole,
    resolved_seed: int,
    ensemble_index: int,
    refresh_index: int,
    global_iteration_start: int,
) -> tuple[KrotovFixedMapEnsemble, bool]:
    """Return a verified cached ensemble or sample it exactly once."""
    key = (ensemble_index, refresh_index)
    if key in cache:
        ensemble = cache[key]
        _verify_replay_binding(
            ensemble,
            stage=stage,
            binding=binding,
            circuit=circuit,
            provider_checksum=cast("str", translation.provider_checksum),
            role=role,
            resolved_seed=resolved_seed,
            ensemble_index=ensemble_index,
            refresh_index=refresh_index,
            global_iteration_start=global_iteration_start,
            trajectory_count=cast("KrotovTJMOptions", translation.tjm_options).num_trajectories,
        )
        if ensemble not in ordered:
            ordered.append(ensemble)
        return ensemble, False

    ensemble = sample_krotov_fixed_map_ensemble(
        circuit,
        theta,
        copy.deepcopy(initial_state),
        translation.options.truncation,
        cast("GateNoiseProvider", translation.noise_provider),
        cast("KrotovTJMOptions", translation.tjm_options),
        role=role,
        resolved_seed=resolved_seed,
        stage_index=stage.stage_index,
        stage_id=stage.stage_id,
        stage_configuration_checksum=stage.configuration_checksum,
        circuit_checksum=binding.content_checksum,
        provider_checksum=cast("str", translation.provider_checksum),
        ensemble_index=ensemble_index,
        refresh_index=refresh_index,
        global_iteration_start=global_iteration_start,
    )
    cache[key] = ensemble
    ordered.append(ensemble)
    return ensemble, True


def _noisy_work(
    work: KrotovWorkLedger,
    *,
    trajectories: int,
    gate_count: int,
    role: Literal["training", "checkpoint_validation"],
    objective: bool,
) -> KrotovWorkLedger:
    """Count one ensemble forward execution under the frozen WP17 convention."""
    increments = {
        "objective_evaluations": int(objective),
        "trajectory_gate_applications": trajectories * gate_count,
    }
    if role == "training":
        increments["training_trajectories"] = trajectories
    else:
        increments["checkpoint_validation_trajectories"] = trajectories
    return work.plus(**increments)


def _replace_validation(
    row: NoisyKrotovIterationRecord,
    *,
    fidelity: float,
    ensemble_checksum: str,
    ensemble_sampled: bool,
    work: KrotovWorkLedger,
) -> NoisyKrotovIterationRecord:
    """Attach one checkpoint-validation observation to a trace row."""
    return replace(
        row,
        checkpoint_validation_fidelity=fidelity,
        checkpoint_validation_ensemble_checksum=ensemble_checksum,
        checkpoint_validation_ensemble_sampled=ensemble_sampled,
        cumulative_work=work,
    )


@dataclass(frozen=True, slots=True)
class FixedRateNoisyKrotovStageAdapter:
    """Execute one resolved fixed-rate Krotov stage without filesystem output."""

    adapter_version: str = field(default=NOISY_KROTOV_ADAPTER_VERSION, init=False)

    @staticmethod
    def execute(
        stage: TrainingStageConfig,
        circuit_binding: NoisyKrotovCircuitBinding,
        target: MaterializedTarget | MPS | NDArray[np.complex128],
        initial_theta: NDArray[np.float64],
        *,
        initial_state: MPS | None = None,
        global_iteration_offset: int = 0,
        iteration_count: int | None = None,
        replay_training_ensembles: Sequence[KrotovFixedMapEnsemble] = (),
        replay_validation_ensembles: Sequence[KrotovFixedMapEnsemble] = (),
        resume_state: NoisyKrotovResumeState | None = None,
    ) -> NoisyKrotovStageExecution | NoisyKrotovStageFailure:
        """Execute a stage, converting any failure into a structured record.

        The signature intentionally contains no final-test or evaluation
        configuration.  Consequently test settings cannot enter initialization,
        map construction, validation selection, or final parameters.

        ``global_iteration_offset`` is stage-global across resume chunks, not a
        cumulative index across different pipeline stages. Every resumed run
        must carry its provenance-bound prior state and cumulative work.
        """
        work = KrotovWorkLedger()
        phase: FailurePhase = "validation"
        # One catch boundary is intentional: structured failures are the public
        # adapter contract, and this function performs no external mutation.
        try:
            translation = translate_fixed_rate_krotov_stage(stage, circuit_binding)
            circuit = circuit_binding.circuit
            theta = _validated_theta(initial_theta, expected_count=stage.output_parameter_count)
            initial_theta_copy = theta.copy()
            resolved_target = _resolved_target(target, num_qubits=circuit.num_qubits)
            state = _initial_state_template(initial_state, circuit.num_qubits)
            objective_binding = NoisyKrotovObjectiveBinding.from_inputs(
                target,
                initial_state,
                num_qubits=circuit.num_qubits,
            )
            objective_checksum = objective_binding.objective_checksum
            offset = _require_builtin_int(global_iteration_offset, "global_iteration_offset")
            if offset > stage.iteration_budget:
                msg = "global_iteration_offset cannot exceed the stage iteration budget."
                raise ValueError(msg)
            remaining = stage.iteration_budget - offset
            count = remaining if iteration_count is None else _require_builtin_int(iteration_count, "iteration_count")
            if count < 1 or count > remaining:
                msg = "iteration_count must be positive and cannot exceed the remaining stage budget."
                raise ValueError(msg)

            validation_config = stage.checkpoint_validation
            selected_theta = theta.copy()
            selected_fidelity = -math.inf
            selected_global_iteration = offset
            cumulative_cross_pairings = 0
            if offset == 0:
                if resume_state is not None:
                    msg = "An initial stage chunk cannot accept a resume state."
                    raise ValueError(msg)
            else:
                if resume_state is None:
                    msg = "A resumed stage requires its provenance-bound prior state."
                    raise ValueError(msg)
                resumed_selection, selected_fidelity, selected_global_iteration = _verify_resume_state(
                    resume_state,
                    stage=stage,
                    binding=circuit_binding,
                    circuit=circuit,
                    provider_checksum=translation.provider_checksum,
                    objective_checksum=objective_checksum,
                    global_iteration_offset=offset,
                    theta=theta,
                )
                if resumed_selection is not None:
                    selected_theta = resumed_selection
                work = resume_state.cumulative_work
                cumulative_cross_pairings = resume_state.cumulative_cross_trajectory_pairings

            validation_call_index = (
                0
                if not validation_config.enabled or offset == 0
                else offset // cast("int", validation_config.cadence) + 1
            )

            training_cache = _ensemble_cache(replay_training_ensembles, role="training_trajectory")
            validation_cache = _ensemble_cache(
                replay_validation_ensembles,
                role="checkpoint_validation",
            )
            used_training: list[KrotovFixedMapEnsemble] = []
            used_validation: list[KrotovFixedMapEnsemble] = []
            trace_rows: list[NoisyKrotovIterationRecord] = []
            gate_count = len(circuit.gates)
            trainable_gate_count = sum(gate.is_trainable for gate in circuit.gates)

            current_ensemble: KrotovFixedMapEnsemble | None = None
            current_ensemble_sampled = False
            if translation.tjm_options is None:
                initial_loss, initial_fidelity = state_preparation_metrics(
                    circuit,
                    theta,
                    resolved_target,
                    initial_state=state,
                    truncation=translation.options.truncation,
                )
                work = work.plus(objective_evaluations=1)
            else:
                phase = "sampling"
                first_global_update = offset
                ensemble_index, refresh_index, window_start = _schedule_point(
                    stage.sampling_policy,
                    stage.crn_refresh_interval,
                    first_global_update,
                )
                if (
                    offset > 0
                    and stage.sampling_policy == "crn_fixed"
                    and (ensemble_index, refresh_index) not in training_cache
                ) or (
                    offset > 0
                    and stage.sampling_policy == "crn_refresh"
                    and offset != window_start
                    and (ensemble_index, refresh_index) not in training_cache
                ):
                    msg = "A mid-window resumed CRN stage requires its caller-supplied active replay ensemble."
                    raise ValueError(msg)
                current_ensemble, sampled = _sample_or_replay(
                    cache=training_cache,
                    ordered=used_training,
                    stage=stage,
                    binding=circuit_binding,
                    circuit=circuit,
                    theta=theta,
                    initial_state=state,
                    translation=translation,
                    role="training_trajectory",
                    resolved_seed=cast("int", stage.training_seed),
                    ensemble_index=ensemble_index,
                    refresh_index=refresh_index,
                    global_iteration_start=window_start,
                )
                current_ensemble_sampled = sampled
                if sampled:
                    work = _noisy_work(
                        work,
                        trajectories=stage.trajectory_count,
                        gate_count=gate_count,
                        role="training",
                        objective=False,
                    )
                initial_loss, initial_fidelity, _ = noisy_state_preparation_metrics(
                    circuit,
                    theta,
                    resolved_target,
                    None,
                    translation.tjm_options,
                    initial_state=state,
                    truncation=translation.options.truncation,
                    fixed_noise_maps=current_ensemble.replay_maps(),
                    noise_provider=translation.noise_provider,
                )
                work = _noisy_work(
                    work,
                    trajectories=stage.trajectory_count,
                    gate_count=gate_count,
                    role="training",
                    objective=True,
                )
            initial_row = NoisyKrotovIterationRecord(
                local_iteration=0,
                global_iteration=offset,
                parameter_checksum=_vector_checksum(theta),
                learning_rate=0.0,
                monitoring_loss=initial_loss,
                monitoring_fidelity=initial_fidelity,
                checkpoint_validation_fidelity=None,
                update_signal=(),
                update_signal_kind="none",
                update_signal_norm=0.0,
                gradient_norm=None,
                cross_dense_sum_norm=None,
                update_norm=0.0,
                trajectory_count=stage.trajectory_count,
                nonidentity_events=(0 if current_ensemble is None else current_ensemble.nonidentity_event_count),
                training_ensemble_id=(None if current_ensemble is None else current_ensemble.ensemble_id),
                training_ensemble_checksum=(None if current_ensemble is None else current_ensemble.content_checksum),
                checkpoint_validation_ensemble_checksum=None,
                cumulative_work=work,
                training_ensemble_sampled=current_ensemble_sampled,
                cumulative_cross_trajectory_pairings=cumulative_cross_pairings,
            )
            if validation_config.enabled and offset == 0:
                phase = "checkpoint_validation"
                validation_translation = _validation_translation(stage, circuit_binding, translation)
                validation_ensemble_index, validation_refresh_index, validation_window_start = _schedule_point(
                    validation_config.sampling_policy,
                    validation_config.ensemble_refresh_interval,
                    validation_call_index,
                )
                validation_ensemble, sampled = _sample_or_replay(
                    cache=validation_cache,
                    ordered=used_validation,
                    stage=stage,
                    binding=circuit_binding,
                    circuit=circuit,
                    theta=theta,
                    initial_state=state,
                    translation=validation_translation,
                    role="checkpoint_validation",
                    resolved_seed=cast("int", validation_config.seed),
                    ensemble_index=validation_ensemble_index,
                    refresh_index=validation_refresh_index,
                    global_iteration_start=validation_window_start,
                )
                if sampled:
                    work = _noisy_work(
                        work,
                        trajectories=validation_config.trajectory_count,
                        gate_count=gate_count,
                        role="checkpoint_validation",
                        objective=False,
                    )
                _validation_loss, validation_fidelity, _ = noisy_state_preparation_metrics(
                    circuit,
                    theta,
                    resolved_target,
                    None,
                    cast("KrotovTJMOptions", validation_translation.tjm_options),
                    initial_state=state,
                    truncation=translation.options.truncation,
                    fixed_noise_maps=validation_ensemble.replay_maps(),
                    noise_provider=validation_translation.noise_provider,
                )
                work = _noisy_work(
                    work,
                    trajectories=validation_config.trajectory_count,
                    gate_count=gate_count,
                    role="checkpoint_validation",
                    objective=True,
                )
                initial_row = _replace_validation(
                    initial_row,
                    fidelity=validation_fidelity,
                    ensemble_checksum=validation_ensemble.content_checksum,
                    ensemble_sampled=sampled,
                    work=work,
                )
                selected_theta = theta.copy()
                selected_fidelity = cast("float", initial_row.checkpoint_validation_fidelity)
                selected_global_iteration = 0
                validation_call_index += 1
            trace_rows.append(initial_row)

            for local_update_index in range(count):
                phase = "sampling"
                global_update_index = offset + local_update_index
                step = _learning_rate(translation, global_update_index)
                ensemble = None
                ensemble_sampled = False
                if translation.tjm_options is not None:
                    ensemble_index, refresh_index, window_start = _schedule_point(
                        stage.sampling_policy,
                        stage.crn_refresh_interval,
                        global_update_index,
                    )
                    ensemble, sampled = _sample_or_replay(
                        cache=training_cache,
                        ordered=used_training,
                        stage=stage,
                        binding=circuit_binding,
                        circuit=circuit,
                        theta=theta,
                        initial_state=state,
                        translation=translation,
                        role="training_trajectory",
                        resolved_seed=cast("int", stage.training_seed),
                        ensemble_index=ensemble_index,
                        refresh_index=refresh_index,
                        global_iteration_start=window_start,
                    )
                    ensemble_sampled = sampled
                    if sampled:
                        work = _noisy_work(
                            work,
                            trajectories=stage.trajectory_count,
                            gate_count=gate_count,
                            role="training",
                            objective=False,
                        )

                phase = "optimization"
                theta_before = theta.copy()
                if translation.tjm_options is None:
                    signal, _pre_loss, _pre_fidelity = state_preparation_contribution(
                        circuit,
                        theta,
                        resolved_target,
                        initial_state=state,
                        truncation=translation.options.truncation,
                    )
                    work = work.plus(objective_evaluations=1, gradient_evaluations=1)
                else:
                    signal, _pre_loss, _pre_fidelity, _ = noisy_state_preparation_contribution(
                        circuit,
                        theta,
                        resolved_target,
                        None,
                        translation.tjm_options,
                        copy.deepcopy(state),
                        translation.options.truncation,
                        fixed_noise_maps=cast("KrotovFixedMapEnsemble", ensemble).replay_maps(),
                        noise_provider=translation.noise_provider,
                    )
                    work = _noisy_work(
                        work,
                        trajectories=stage.trajectory_count,
                        gate_count=gate_count,
                        role="training",
                        objective=True,
                    ).plus(gradient_evaluations=int(stage.trajectory_update == "independent"))
                theta -= step * signal
                update_norm = float(np.linalg.norm(theta - theta_before))

                if translation.tjm_options is None:
                    monitoring_loss, monitoring_fidelity = state_preparation_metrics(
                        circuit,
                        theta,
                        resolved_target,
                        initial_state=state,
                        truncation=translation.options.truncation,
                    )
                    work = work.plus(objective_evaluations=1)
                else:
                    monitoring_loss, monitoring_fidelity, _ = noisy_state_preparation_metrics(
                        circuit,
                        theta,
                        resolved_target,
                        None,
                        translation.tjm_options,
                        initial_state=state,
                        truncation=translation.options.truncation,
                        fixed_noise_maps=cast("KrotovFixedMapEnsemble", ensemble).replay_maps(),
                        noise_provider=translation.noise_provider,
                    )
                    work = _noisy_work(
                        work,
                        trajectories=stage.trajectory_count,
                        gate_count=gate_count,
                        role="training",
                        objective=True,
                    )

                update_kind: UpdateSignalKind
                gradient_norm: float | None
                cross_norm: float | None
                if stage.trajectory_update == "cross":
                    update_kind = "cross_dense_sum_update"
                    gradient_norm = None
                    cross_norm = float(np.linalg.norm(signal))
                    cross_pairings = stage.trajectory_count**2 * trainable_gate_count
                    cumulative_cross_pairings += cross_pairings
                elif _gradient_is_established(translation):
                    update_kind = "independent_pathwise_gradient"
                    gradient_norm = float(np.linalg.norm(signal))
                    cross_norm = None
                    cross_pairings = 0
                else:
                    update_kind = "independent_pathwise_update"
                    gradient_norm = None
                    cross_norm = None
                    cross_pairings = 0
                completed_global_iteration = global_update_index + 1
                row = NoisyKrotovIterationRecord(
                    local_iteration=local_update_index + 1,
                    global_iteration=completed_global_iteration,
                    parameter_checksum=_vector_checksum(theta),
                    learning_rate=step,
                    monitoring_loss=monitoring_loss,
                    monitoring_fidelity=monitoring_fidelity,
                    checkpoint_validation_fidelity=None,
                    update_signal=tuple(float(value) for value in signal),
                    update_signal_kind=update_kind,
                    update_signal_norm=float(np.linalg.norm(signal)),
                    gradient_norm=gradient_norm,
                    cross_dense_sum_norm=cross_norm,
                    update_norm=update_norm,
                    trajectory_count=stage.trajectory_count,
                    nonidentity_events=(0 if ensemble is None else ensemble.nonidentity_event_count),
                    training_ensemble_id=(None if ensemble is None else ensemble.ensemble_id),
                    training_ensemble_checksum=(None if ensemble is None else ensemble.content_checksum),
                    checkpoint_validation_ensemble_checksum=None,
                    cumulative_work=work,
                    training_ensemble_sampled=ensemble_sampled,
                    cross_trajectory_pairings=cross_pairings,
                    cumulative_cross_trajectory_pairings=cumulative_cross_pairings,
                )

                should_validate = validation_config.enabled and (
                    completed_global_iteration % cast("int", validation_config.cadence) == 0
                    or completed_global_iteration == stage.iteration_budget
                )
                if should_validate:
                    phase = "checkpoint_validation"
                    validation_translation = _validation_translation(stage, circuit_binding, translation)
                    validation_ensemble_index, validation_refresh_index, validation_window_start = _schedule_point(
                        validation_config.sampling_policy,
                        validation_config.ensemble_refresh_interval,
                        validation_call_index,
                    )
                    if (
                        validation_call_index > 0
                        and validation_config.sampling_policy == "crn_fixed"
                        and (validation_ensemble_index, validation_refresh_index) not in validation_cache
                    ) or (
                        validation_call_index > 0
                        and validation_config.sampling_policy == "crn_refresh"
                        and validation_call_index != validation_window_start
                        and (validation_ensemble_index, validation_refresh_index) not in validation_cache
                    ):
                        msg = (
                            "A resumed checkpoint-validation CRN window requires its caller-supplied "
                            "active replay ensemble."
                        )
                        raise ValueError(msg)
                    validation_ensemble, sampled = _sample_or_replay(
                        cache=validation_cache,
                        ordered=used_validation,
                        stage=stage,
                        binding=circuit_binding,
                        circuit=circuit,
                        theta=theta,
                        initial_state=state,
                        translation=validation_translation,
                        role="checkpoint_validation",
                        resolved_seed=cast("int", validation_config.seed),
                        ensemble_index=validation_ensemble_index,
                        refresh_index=validation_refresh_index,
                        global_iteration_start=validation_window_start,
                    )
                    if sampled:
                        work = _noisy_work(
                            work,
                            trajectories=validation_config.trajectory_count,
                            gate_count=gate_count,
                            role="checkpoint_validation",
                            objective=False,
                        )
                    _validation_loss, validation_fidelity, _ = noisy_state_preparation_metrics(
                        circuit,
                        theta,
                        resolved_target,
                        None,
                        cast("KrotovTJMOptions", validation_translation.tjm_options),
                        initial_state=state,
                        truncation=translation.options.truncation,
                        fixed_noise_maps=validation_ensemble.replay_maps(),
                        noise_provider=validation_translation.noise_provider,
                    )
                    work = _noisy_work(
                        work,
                        trajectories=validation_config.trajectory_count,
                        gate_count=gate_count,
                        role="checkpoint_validation",
                        objective=True,
                    )
                    row = _replace_validation(
                        row,
                        fidelity=validation_fidelity,
                        ensemble_checksum=validation_ensemble.content_checksum,
                        ensemble_sampled=sampled,
                        work=work,
                    )
                    normalized_validation_fidelity = cast("float", row.checkpoint_validation_fidelity)
                    if normalized_validation_fidelity > selected_fidelity or (
                        normalized_validation_fidelity == selected_fidelity
                        and validation_config.tie_breaker == "latest_iteration"
                    ):
                        selected_theta = theta.copy()
                        selected_fidelity = normalized_validation_fidelity
                        selected_global_iteration = completed_global_iteration
                    validation_call_index += 1
                trace_rows.append(row)

            if not validation_config.enabled:
                selected_theta = theta.copy()
                selected_global_iteration = offset + count

            return NoisyKrotovStageExecution(
                stage=stage,
                circuit_binding_checksum=circuit_binding.content_checksum,
                circuit_binding_document=circuit_binding.to_dict(),
                provider_checksum=translation.provider_checksum,
                objective_binding=objective_binding,
                initial_theta=initial_theta_copy,
                final_theta=theta,
                selected_theta=selected_theta,
                selected_global_iteration=selected_global_iteration,
                selected_checkpoint_validation_fidelity=(
                    None if not math.isfinite(selected_fidelity) else selected_fidelity
                ),
                trace=trace_rows,
                training_ensembles=used_training,
                checkpoint_validation_ensembles=used_validation,
                normalized_work=work,
                input_resume_state=resume_state,
            )
        except Exception as error:
            return NoisyKrotovStageFailure(
                stage_index=getattr(stage, "stage_index", 0),
                stage_id=getattr(stage, "stage_id", "invalid_stage"),
                stage_configuration_checksum=getattr(
                    stage,
                    "configuration_checksum",
                    "sha256:" + "0" * 64,
                ),
                phase=phase,
                exception_type=type(error).__name__,
                message=str(error) or type(error).__name__,
                traceback_text="".join(traceback.format_exception(error)),
                retryable=False,
                partial_work=work.to_dict(),
            )


def _validation_translation(
    stage: TrainingStageConfig,
    binding: NoisyKrotovCircuitBinding,
    training_translation: KrotovStageTranslation,
) -> KrotovStageTranslation:
    """Build a separately seeded fixed-rate checkpoint-validation translation."""
    config = stage.checkpoint_validation
    if not config.enabled:
        msg = "Checkpoint-validation translation requires an enabled policy."
        raise ValueError(msg)
    if config.noise_id in STANDARD_NOISE_IDS:
        if config.noise_definition_version != FIXED_RATE_NOISE_DEFINITION_VERSION:
            msg = "Checkpoint validation uses an unsupported standard-noise definition version."
            raise ValueError(msg)
        provider = create_scaled_standard_noise_provider(
            config.noise_id,
            cast("float", config.noise_strength_scale),
        )
    elif config.noise_id == HISTORICAL_FIXED_RATE_NOISE_ID:
        if not math.isclose(cast("float", config.noise_strength_scale), 1.0, rel_tol=0.0, abs_tol=0.0):
            msg = "The frozen historical checkpoint profile requires noise_strength_scale=1.0."
            raise ValueError(msg)
        provider = create_historical_fixed_rate_noise_provider()
        if config.noise_definition_version != provider.noise_definition_version:
            msg = "Checkpoint validation uses an unsupported historical-noise definition version."
            raise ValueError(msg)
        if not math.isclose(cast("float", config.tjm_dt), provider.tjm_dt, rel_tol=0.0, abs_tol=0.0):
            msg = f"Historical checkpoint profile requires tjm_dt={provider.tjm_dt!r}."
            raise ValueError(msg)
    else:
        msg = f"Unsupported checkpoint-validation noise identifier {config.noise_id!r}."
        raise ValueError(msg)
    tjm_options = KrotovTJMOptions(
        num_trajectories=config.trajectory_count,
        random_seed=cast("int", config.seed),
        dt=cast("float", config.tjm_dt),
        apply_noise_to="all",
        noisy_gate_indices=binding.noisy_gate_indices,
        trajectory_update="independent",
        differentiate_jump_normalization=False,
        use_crn=False,
    )
    return KrotovStageTranslation(
        options=training_translation.options,
        tjm_options=tjm_options,
        noise_provider=provider,
        provider_checksum=_provider_checksum(provider),
        learning_rate=training_translation.learning_rate,
        schedule=training_translation.schedule,
        decay=training_translation.decay,
    )


def execute_fixed_rate_krotov_stage(
    stage: TrainingStageConfig,
    circuit_binding: NoisyKrotovCircuitBinding,
    target: MaterializedTarget | MPS | NDArray[np.complex128],
    initial_theta: NDArray[np.float64],
    *,
    initial_state: MPS | None = None,
    global_iteration_offset: int = 0,
    iteration_count: int | None = None,
    replay_training_ensembles: Sequence[KrotovFixedMapEnsemble] = (),
    replay_validation_ensembles: Sequence[KrotovFixedMapEnsemble] = (),
    resume_state: NoisyKrotovResumeState | None = None,
) -> NoisyKrotovStageExecution | NoisyKrotovStageFailure:
    """Execute one WP17 stage with the default stateless adapter.

    Returns:
        A successful immutable execution or a structured failure.
    """
    return FixedRateNoisyKrotovStageAdapter().execute(
        stage,
        circuit_binding,
        target,
        initial_theta,
        initial_state=initial_state,
        global_iteration_offset=global_iteration_offset,
        iteration_count=iteration_count,
        replay_training_ensembles=replay_training_ensembles,
        replay_validation_ensembles=replay_validation_ensembles,
        resume_state=resume_state,
    )


__all__ = [
    "FIXED_RATE_NOISE_DEFINITION_VERSION",
    "LOGICAL_PARAMETERIZED_GATE_PLACEMENT",
    "NOISY_KROTOV_ADAPTER_VERSION",
    "NOISY_KROTOV_CHECKPOINT_SELECTION_SCHEMA_VERSION",
    "NOISY_KROTOV_CIRCUIT_BINDING_SCHEMA_VERSION",
    "NOISY_KROTOV_EXECUTION_SCHEMA_VERSION",
    "NOISY_KROTOV_FAILURE_SCHEMA_VERSION",
    "NOISY_KROTOV_OBJECTIVE_BINDING_SCHEMA_VERSION",
    "NOISY_KROTOV_RESUME_STATE_SCHEMA_VERSION",
    "NOISY_KROTOV_TRACE_SCHEMA_VERSION",
    "PRIMARY_COMPILER_POLICY_ID",
    "PRIMARY_CONNECTIVITY",
    "PRIMARY_COUNTING_POLICY_ID",
    "PRIMARY_ROUTING_POLICY_ID",
    "FixedRateNoisyKrotovStageAdapter",
    "KrotovStageTranslation",
    "KrotovWorkLedger",
    "NoisyKrotovCheckpointSelection",
    "NoisyKrotovCircuitBinding",
    "NoisyKrotovIterationRecord",
    "NoisyKrotovObjectiveBinding",
    "NoisyKrotovResumeState",
    "NoisyKrotovStageExecution",
    "NoisyKrotovStageFailure",
    "decode_noisy_krotov_circuit_binding_document",
    "execute_fixed_rate_krotov_stage",
    "noisy_krotov_computational_zero_state_checksum",
    "translate_fixed_rate_krotov_stage",
    "validate_noisy_krotov_execution_trace",
]
