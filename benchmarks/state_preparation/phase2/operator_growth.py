# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Deterministic, provenance-sealed operator-growth competitors for Phase II.

The family-wide comparator in this module minimizes pure-state projector
infidelity through :func:`run_standard_fixed_rate_noisy_operator_growth` and is
never represented as ADAPT-VQE.  :func:`adapt_style_state_preparation` is its
analytic reference-only counterpart.  The separate energy-growth
implementations minimize the dense open-chain TFIM Hamiltonian and are
structurally applicable only to TFIM targets.

The growth implementations select the largest absolute parameter-shift gradient at
an appended zero parameter, break exact ties by frozen pool order, and fully
reoptimize all retained parameters after every append.  The execution records
every objective evaluation and scalar gradient request, including all noisy
trajectory work in the target-bound path.
"""

# The module contains many small strict record validators and codecs whose
# return/exception behavior is already explicit in their annotations and names.
# Repeating those mechanical details would obscure the scientific docstrings.
# ruff: noqa: DOC201, DOC501

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, cast

import numpy as np

from benchmarks.state_preparation.constants import STANDARD_NOISE_IDS
from benchmarks.state_preparation.noise import (
    FIXED_RATE_NOISE_DEFINITION_VERSION,
    ScaledStandardNoiseProvider,
    create_scaled_standard_noise_provider,
)
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import (
    KrotovTJMOptions,
    KrotovTruncation,
    ParameterizedCircuit,
    ParameterizedGate,
    noisy_state_preparation_metrics,
)

from .canonical import canonical_checksum, freeze_json_mapping
from .legacy_targets import LegacyMaterializedTarget
from .targets import MaterializedTarget, TargetInstanceSpec
from .validation import (
    require_bool,
    require_checksum,
    require_exact_keys,
    require_float,
    require_int,
    require_mapping,
    require_slug,
    require_string,
)
from .wp20_resources import CircuitResourceMetrics, WP20WorkLedger, measure_circuit_resources

if TYPE_CHECKING:
    from numpy.typing import NDArray


OPERATOR_POOL_SCHEMA_VERSION = "yaqs.state_preparation.phase2.operator_pool.v1"
OPERATOR_GROWTH_SPEC_SCHEMA_VERSION = "yaqs.state_preparation.phase2.operator_growth_spec.v1"
PROJECTOR_OBJECTIVE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.projector_objective.v1"
TFIM_HAMILTONIAN_SCHEMA_VERSION = "yaqs.state_preparation.phase2.tfim_hamiltonian.v1"
OPERATOR_GROWTH_APPLICABILITY_SCHEMA_VERSION = "yaqs.state_preparation.phase2.operator_growth_applicability.v1"
OPERATOR_GROWTH_RESULT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.operator_growth_result.v2"
OPERATOR_GROWTH_TRAINING_PROVENANCE_SCHEMA_VERSION = (
    "yaqs.state_preparation.phase2.operator_growth_training_provenance.v2"
)
OPERATOR_GROWTH_OBJECTIVE_REQUEST_SCHEMA_VERSION = "yaqs.state_preparation.phase2.operator_growth_objective_request.v1"
STANDARD_FIXED_RATE_OPERATOR_GROWTH_EVALUATOR_SCHEMA_VERSION = (
    "yaqs.state_preparation.phase2.standard_fixed_rate_operator_growth_evaluator.v1"
)
STANDARD_FIXED_RATE_OPERATOR_GROWTH_BINDING_SCHEMA_VERSION = (
    "yaqs.state_preparation.phase2.standard_fixed_rate_operator_growth_binding.v2"
)
TARGET_BOUND_TFIM_OBJECTIVE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.target_bound_tfim_objective.v1"

ADAPT_STYLE_METHOD_ID = "adapt_style_state_preparation"
ENERGY_ADAPT_METHOD_ID = "energy_adapt_vqe"
TFIM_FAMILY_ID = "tfim_ground_state"

PROJECTOR_COST_ID = "pure_state_projector_infidelity_v1"
TFIM_ENERGY_COST_ID = "open_chain_tfim_energy_v1"
QUANTINUUM_NATIVE_POLICY_ID = "quantinuum_rzz_chain_v1"
CONNECTIVITY_ID = "linear_chain"
ROUTING_POLICY_ID = "identity_no_swap"

_PROJECTOR_POOL_ID = "projector_pauli_rotation_pool_v1"
_TFIM_POOL_ID = "tfim_real_odd_y_pool_v1"
_DUPLICATE_POLICY = "forbidden"
_PROJECTOR_SYMMETRY = "none"
_TFIM_SYMMETRY = "real_state_odd_y_pauli_strings_only"
_PROJECTOR_SITE_ORDER = "site_major_rx_ry_rz_then_edge_major_rxx_ryy_rzz"
_TFIM_SITE_ORDER = "site_major_ry_then_edge_major_ryz_rzy"
_SELECTION_RULE_ID = "largest_absolute_parameter_shift_gradient_at_appended_zero_v1"
_TIE_BREAK_RULE_ID = "lowest_frozen_pool_index_v1"
_PARAMETER_SHIFT_RULE_ID = "pauli_rotation_plus_minus_pi_over_2_v1"
_INTERNAL_ADAM_ID = "deterministic_full_parameter_adam_v1"
_INITIAL_STATE_POLICY = "computational_zero_or_checksum_bound_explicit_state_v1"
_NOISY_SAMPLING_POLICY_ID = "independent_trajectory_mean_fixed_crn_v1"
_TRAJECTORY_GATE_COUNTING_POLICY_ID = "logical_gate_applications_per_objective_trajectory_v1"

_METHOD_IDS = frozenset({ADAPT_STYLE_METHOD_ID, ENERGY_ADAPT_METHOD_ID})
_GENERATORS = frozenset({"x", "y", "z", "xx", "yy", "zz", "yz", "zy"})
_TERMINATION_REASONS = frozenset({
    "gradient_tolerance",
    "max_operators",
    "native_cap",
    "pool_exhausted",
    "not_applicable",
})

_OPERATOR_KEYS = frozenset({
    "operator_id",
    "generator",
    "sites",
    "native_two_qubit_gates",
    "native_decomposition_id",
})
_POOL_KEYS = frozenset({
    "schema_version",
    "pool_id",
    "method_id",
    "num_qubits",
    "one_qubit_generators",
    "two_qubit_generators",
    "site_ordering",
    "duplicate_policy",
    "symmetry_restrictions",
    "cost_function_id",
    "native_compilation_policy_id",
    "connectivity",
    "routing_policy_id",
    "selection_reuse_policy",
    "operators",
    "content_checksum",
})
_GROWTH_SPEC_KEYS = frozenset({
    "schema_version",
    "method_id",
    "pool_checksum",
    "selection_rule_id",
    "tie_break_rule_id",
    "parameter_shift_rule_id",
    "reoptimization_rule_id",
    "initial_state_policy",
    "gradient_tolerance",
    "max_operators",
    "native_two_qubit_cap_per_edge",
    "reoptimization_steps",
    "learning_rate",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "content_checksum",
})
_PROJECTOR_OBJECTIVE_KEYS = frozenset({
    "schema_version",
    "objective_id",
    "num_qubits",
    "target_state_checksum",
    "initial_state_checksum",
    "content_checksum",
})
_TFIM_HAMILTONIAN_KEYS = frozenset({
    "schema_version",
    "objective_id",
    "num_qubits",
    "couplings",
    "fields",
    "boundary_condition",
    "basis_bit_order",
    "target_state_binding",
    "initial_state_checksum",
    "hamiltonian_checksum",
    "content_checksum",
})
_APPLICABILITY_KEYS = frozenset({
    "schema_version",
    "method_id",
    "family_id",
    "status",
    "reason",
    "promotion_eligible",
    "structural_not_applicable_is_failure",
    "content_checksum",
})
_WORK_KEYS = frozenset({
    "forward_circuit_evaluations",
    "backward_circuit_evaluations",
    "objective_calls",
    "gradient_calls",
    "parameter_shift_evaluations",
    "trajectory_gate_applications",
    "total_sampled_trajectories",
    "cross_trajectory_pairings",
    "reoptimization_iterations",
})
_CANDIDATE_KEYS = frozenset({
    "operator_id",
    "pool_index",
    "gradient",
    "absolute_gradient",
    "native_two_qubit_increment",
    "native_cap_feasible",
})
_STEP_KEYS = frozenset({
    "iteration",
    "event",
    "candidate_gradients",
    "selected_operator_id",
    "selected_gradient",
    "objective_before_reoptimization",
    "objective_after_reoptimization",
    "parameters",
    "native_two_qubit_counts_by_edge",
    "work_after_step",
})
_RESULT_KEYS = frozenset({
    "schema_version",
    "method_id",
    "algorithm_label",
    "status",
    "applicability",
    "pool_checksum",
    "growth_spec_checksum",
    "objective_binding_checksum",
    "selected_operator_ids",
    "parameters",
    "initial_objective",
    "final_objective",
    "native_two_qubit_counts_by_edge",
    "termination_reason",
    "trace",
    "work",
    "execution_mode",
    "training_provenance",
    "objective_requests",
    "circuit_resources",
    "pool",
    "growth_spec",
    "objective_binding",
    "evaluator_binding",
    "content_checksum",
})
_TRAINING_PROVENANCE_KEYS = frozenset({
    "schema_version",
    "target_instance_id",
    "target_family_id",
    "qubit_count",
    "target_vector_checksum",
    "initial_state_checksum",
    "optimization_block_id",
    "optimization_seed",
    "resource_stratum_id",
    "provider_id",
    "provider_checksum",
    "objective_id",
    "objective_checksum",
    "noise_id",
    "noise_definition_version",
    "noise_strength_scale",
    "tjm_dt",
    "noise_condition_checksum",
    "trajectory_count",
    "trajectory_seed",
    "trajectory_ensemble_checksum",
    "sampling_policy_id",
    "trajectory_gate_counting_policy_id",
    "content_checksum",
})
_OBJECTIVE_REQUEST_KEYS = frozenset({
    "schema_version",
    "training_provenance_checksum",
    "evaluation_index",
    "selected_operator_ids",
    "parameters",
    "circuit_resources_checksum",
    "logical_gate_count",
    "trajectory_count",
    "trajectory_ensemble_checksum",
    "content_checksum",
})
_EVALUATOR_BINDING_KEYS = frozenset({
    "schema_version",
    "target_instance_id",
    "target_instance_spec_checksum",
    "population_config_checksum",
    "target_manifest_checksum",
    "parameter_checksum",
    "target_family_id",
    "target_stratum_id",
    "qubit_count",
    "norm",
    "target_vector_checksum",
    "training_provenance",
    "content_checksum",
})
_TARGET_BOUND_TFIM_OBJECTIVE_KEYS = frozenset({
    "schema_version",
    "target_instance_spec",
    "target_manifest_checksum",
    "target_vector_checksum",
    "hamiltonian_binding",
    "content_checksum",
})


def _strict_int(value: object, name: str, *, minimum: int = 0) -> int:
    """Return a validated built-in integer."""
    return require_int(value, name, minimum=minimum)


def _strict_optional_float(value: object, name: str) -> float | None:
    """Return a finite float or ``None`` without numeric coercion."""
    return None if value is None else require_float(value, name)


def _float_tuple(value: object, name: str, *, length: int | None = None) -> tuple[float, ...]:
    """Validate a strict sequence of finite built-in floats."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = f"{name} must be a sequence of floats."
        raise TypeError(msg)
    result = tuple(require_float(item, f"{name}[{index}]") for index, item in enumerate(value))
    if length is not None and len(result) != length:
        msg = f"{name} must contain exactly {length} values."
        raise ValueError(msg)
    return result


def _int_tuple(value: object, name: str, *, length: int | None = None) -> tuple[int, ...]:
    """Validate a strict sequence of built-in nonnegative integers."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = f"{name} must be a sequence of integers."
        raise TypeError(msg)
    result = tuple(require_int(item, f"{name}[{index}]") for index, item in enumerate(value))
    if length is not None and len(result) != length:
        msg = f"{name} must contain exactly {length} values."
        raise ValueError(msg)
    return result


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    """Validate an ordered sequence of unique slug strings."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = f"{name} must be a sequence of identifiers."
        raise TypeError(msg)
    result = tuple(require_slug(item, f"{name}[{index}]") for index, item in enumerate(value))
    if len(result) != len(set(result)):
        msg = f"{name} must not contain duplicates."
        raise ValueError(msg)
    return result


def _array_checksum(vector: NDArray[np.complex128]) -> str:
    """Return a stable checksum of little-endian complex128 state bytes."""
    canonical = np.ascontiguousarray(vector, dtype=np.dtype("<c16"))
    return f"sha256:{hashlib.sha256(canonical.tobytes(order='C')).hexdigest()}"


def _normalized_state(
    value: object,
    name: str,
    *,
    num_qubits: int | None = None,
) -> tuple[NDArray[np.complex128], int]:
    """Return a finite normalized dense state and its inferred width."""
    try:
        vector = np.asarray(value, dtype=np.complex128)
    except (TypeError, ValueError) as error:
        msg = f"{name} must be convertible to a complex128 vector."
        raise TypeError(msg) from error
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        msg = f"{name} must be a nonempty finite one-dimensional vector."
        raise ValueError(msg)
    inferred = vector.size.bit_length() - 1
    if 2**inferred != vector.size:
        msg = f"{name} length must be a power of two."
        raise ValueError(msg)
    if num_qubits is not None and inferred != num_qubits:
        msg = f"{name} has {inferred} qubits, expected {num_qubits}."
        raise ValueError(msg)
    norm = float(np.linalg.norm(vector))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-12):
        msg = f"{name} must have unit norm."
        raise ValueError(msg)
    return np.ascontiguousarray(vector, dtype=np.complex128), inferred


def computational_zero_state(num_qubits: int) -> NDArray[np.complex128]:
    """Return the dense little-endian computational all-zero state.

    Args:
        num_qubits: Number of qubits.

    Returns:
        A normalized complex128 statevector.
    """
    qubits = _strict_int(num_qubits, "num_qubits", minimum=1)
    state = np.zeros(2**qubits, dtype=np.complex128)
    state[0] = 1.0
    return state


@dataclass(frozen=True, slots=True)
class PoolOperator:
    """One unique Pauli-product rotation in a frozen growth pool."""

    operator_id: str
    generator: str
    sites: tuple[int, ...]
    native_two_qubit_gates: int
    native_decomposition_id: str

    def __post_init__(self) -> None:
        """Validate generator identity, sites, and native cost."""
        operator_id = require_slug(self.operator_id, "operator_id")
        generator = require_slug(self.generator, "generator")
        if generator not in _GENERATORS:
            msg = f"Unsupported Pauli generator {generator!r}."
            raise ValueError(msg)
        sites = _int_tuple(self.sites, "sites", length=len(generator))
        if len(sites) not in {1, 2} or len(set(sites)) != len(sites):
            msg = "Pool operators must act on one or two distinct sites."
            raise ValueError(msg)
        if len(sites) == 2 and sites[1] != sites[0] + 1:
            msg = "Two-qubit pool operators must act on one ascending nearest-neighbor edge."
            raise ValueError(msg)
        expected_id = f"r{generator}_" + "_".join(f"q{site}" for site in sites)
        if operator_id != expected_id:
            msg = f"operator_id must be the canonical identifier {expected_id!r}."
            raise ValueError(msg)
        native_cost = _strict_int(self.native_two_qubit_gates, "native_two_qubit_gates")
        if native_cost != (len(sites) - 1):
            msg = "Every two-qubit pool rotation must compile to exactly one native RZZ."
            raise ValueError(msg)
        decomposition = require_slug(self.native_decomposition_id, "native_decomposition_id")
        expected_decomposition = "direct_1q_rotation" if len(sites) == 1 else "local_basis_rzz_local_basis"
        if decomposition != expected_decomposition:
            msg = f"native_decomposition_id must be {expected_decomposition!r}."
            raise ValueError(msg)
        object.__setattr__(self, "operator_id", operator_id)
        object.__setattr__(self, "generator", generator)
        object.__setattr__(self, "sites", sites)
        object.__setattr__(self, "native_two_qubit_gates", native_cost)
        object.__setattr__(self, "native_decomposition_id", decomposition)

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON-native operator record."""
        return {
            "operator_id": self.operator_id,
            "generator": self.generator,
            "sites": list(self.sites),
            "native_two_qubit_gates": self.native_two_qubit_gates,
            "native_decomposition_id": self.native_decomposition_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> PoolOperator:
        """Decode one exact operator record."""
        mapping = require_mapping(value, "pool operator")
        require_exact_keys(mapping, _OPERATOR_KEYS, "pool operator")
        return cls(
            operator_id=cast("str", mapping["operator_id"]),
            generator=cast("str", mapping["generator"]),
            sites=cast("tuple[int, ...]", mapping["sites"]),
            native_two_qubit_gates=cast("int", mapping["native_two_qubit_gates"]),
            native_decomposition_id=cast("str", mapping["native_decomposition_id"]),
        )


def _operator(generator: str, sites: tuple[int, ...]) -> PoolOperator:
    """Construct one canonical frozen pool operator."""
    return PoolOperator(
        operator_id=f"r{generator}_" + "_".join(f"q{site}" for site in sites),
        generator=generator,
        sites=sites,
        native_two_qubit_gates=len(sites) - 1,
        native_decomposition_id=("direct_1q_rotation" if len(sites) == 1 else "local_basis_rzz_local_basis"),
    )


def _expected_projector_operators(num_qubits: int) -> tuple[PoolOperator, ...]:
    """Return the exact site-major then edge-major projector pool."""
    singles = tuple(_operator(generator, (site,)) for site in range(num_qubits) for generator in ("x", "y", "z"))
    pairs = tuple(
        _operator(generator, (site, site + 1)) for site in range(num_qubits - 1) for generator in ("xx", "yy", "zz")
    )
    return singles + pairs


def _expected_tfim_operators(num_qubits: int) -> tuple[PoolOperator, ...]:
    """Return the exact real-state TFIM pool.

    The two-qubit choices are ordered ``Y_i Z_(i+1)`` then
    ``Z_i Y_(i+1)`` on every open-chain edge.  Each contains one Y and
    therefore has a real matrix exponential, while compiling to one RZZ plus
    local basis changes.
    """
    singles = tuple(_operator("y", (site,)) for site in range(num_qubits))
    pairs = tuple(
        _operator(generator, (site, site + 1)) for site in range(num_qubits - 1) for generator in ("yz", "zy")
    )
    return singles + pairs


@dataclass(frozen=True, slots=True)
class OperatorPoolSpec:
    """Complete checksum-sealed definition of an operator-growth pool."""

    pool_id: str
    method_id: str
    num_qubits: int
    one_qubit_generators: tuple[str, ...]
    two_qubit_generators: tuple[str, ...]
    site_ordering: str
    duplicate_policy: str
    symmetry_restrictions: str
    cost_function_id: str
    native_compilation_policy_id: str
    connectivity: str
    routing_policy_id: str
    selection_reuse_policy: str
    operators: tuple[PoolOperator, ...]
    schema_version: str = field(default=OPERATOR_POOL_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require an exact, immutable WP20 pool rather than a mutable variant."""
        method_id = require_slug(self.method_id, "method_id")
        if method_id not in _METHOD_IDS:
            msg = f"Unsupported operator-growth method {method_id!r}."
            raise ValueError(msg)
        num_qubits = _strict_int(self.num_qubits, "num_qubits", minimum=1)
        operators = tuple(self.operators)
        if any(not isinstance(operator, PoolOperator) for operator in operators):
            msg = "operators must contain only PoolOperator records."
            raise TypeError(msg)
        operator_ids = tuple(operator.operator_id for operator in operators)
        if len(operator_ids) != len(set(operator_ids)):
            msg = "Duplicate operators are forbidden."
            raise ValueError(msg)
        if any(max(operator.sites) >= num_qubits for operator in operators):
            msg = "Pool operator sites must lie inside the declared register."
            raise ValueError(msg)

        if method_id == ADAPT_STYLE_METHOD_ID:
            expected = {
                "pool_id": _PROJECTOR_POOL_ID,
                "one_qubit_generators": ("x", "y", "z"),
                "two_qubit_generators": ("xx", "yy", "zz"),
                "site_ordering": _PROJECTOR_SITE_ORDER,
                "symmetry_restrictions": _PROJECTOR_SYMMETRY,
                "cost_function_id": PROJECTOR_COST_ID,
            }
            expected_operators = _expected_projector_operators(num_qubits)
        else:
            expected = {
                "pool_id": _TFIM_POOL_ID,
                "one_qubit_generators": ("y",),
                "two_qubit_generators": ("yz", "zy"),
                "site_ordering": _TFIM_SITE_ORDER,
                "symmetry_restrictions": _TFIM_SYMMETRY,
                "cost_function_id": TFIM_ENERGY_COST_ID,
            }
            expected_operators = _expected_tfim_operators(num_qubits)

        normalized_generators = {
            "one_qubit_generators": _string_tuple(self.one_qubit_generators, "one_qubit_generators"),
            "two_qubit_generators": _string_tuple(self.two_qubit_generators, "two_qubit_generators"),
        }
        actual: dict[str, object] = {
            "pool_id": require_slug(self.pool_id, "pool_id"),
            **normalized_generators,
            "site_ordering": require_slug(self.site_ordering, "site_ordering"),
            "symmetry_restrictions": require_slug(self.symmetry_restrictions, "symmetry_restrictions"),
            "cost_function_id": require_slug(self.cost_function_id, "cost_function_id"),
        }
        if actual != expected:
            msg = f"Pool metadata differs from the frozen {method_id!r} definition."
            raise ValueError(msg)
        if operators != expected_operators:
            msg = f"Pool operators or ordering differ from the frozen {method_id!r} pool."
            raise ValueError(msg)
        fixed_policy = {
            "duplicate_policy": _DUPLICATE_POLICY,
            "native_compilation_policy_id": QUANTINUUM_NATIVE_POLICY_ID,
            "connectivity": CONNECTIVITY_ID,
            "routing_policy_id": ROUTING_POLICY_ID,
            "selection_reuse_policy": "without_replacement",
        }
        for name, expected_value in fixed_policy.items():
            if require_slug(getattr(self, name), name) != expected_value:
                msg = f"{name} must be the frozen value {expected_value!r}."
                raise ValueError(msg)

        object.__setattr__(self, "method_id", method_id)
        object.__setattr__(self, "num_qubits", num_qubits)
        object.__setattr__(self, "one_qubit_generators", normalized_generators["one_qubit_generators"])
        object.__setattr__(self, "two_qubit_generators", normalized_generators["two_qubit_generators"])
        object.__setattr__(self, "operators", operators)

    def _content_dict(self) -> dict[str, object]:
        """Return all checksum-covered pool fields."""
        return {
            "schema_version": self.schema_version,
            "pool_id": self.pool_id,
            "method_id": self.method_id,
            "num_qubits": self.num_qubits,
            "one_qubit_generators": list(self.one_qubit_generators),
            "two_qubit_generators": list(self.two_qubit_generators),
            "site_ordering": self.site_ordering,
            "duplicate_policy": self.duplicate_policy,
            "symmetry_restrictions": self.symmetry_restrictions,
            "cost_function_id": self.cost_function_id,
            "native_compilation_policy_id": self.native_compilation_policy_id,
            "connectivity": self.connectivity,
            "routing_policy_id": self.routing_policy_id,
            "selection_reuse_policy": self.selection_reuse_policy,
            "operators": [operator.to_dict() for operator in self.operators],
        }

    @property
    def content_checksum(self) -> str:
        """Checksum sealing every pool choice and its exact ordering."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return the complete sealed pool document."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, value: object) -> OperatorPoolSpec:
        """Decode and verify one exact sealed pool document."""
        mapping = freeze_json_mapping(require_mapping(value, "operator pool"), "operator pool")
        require_exact_keys(mapping, _POOL_KEYS, "operator pool")
        if mapping["schema_version"] != OPERATOR_POOL_SCHEMA_VERSION:
            msg = "operator pool uses an unsupported schema version."
            raise ValueError(msg)
        raw_operators = mapping["operators"]
        if not isinstance(raw_operators, Sequence):
            msg = "operator pool operators must be a sequence."
            raise TypeError(msg)
        pool = cls(
            pool_id=cast("str", mapping["pool_id"]),
            method_id=cast("str", mapping["method_id"]),
            num_qubits=cast("int", mapping["num_qubits"]),
            one_qubit_generators=cast("tuple[str, ...]", mapping["one_qubit_generators"]),
            two_qubit_generators=cast("tuple[str, ...]", mapping["two_qubit_generators"]),
            site_ordering=cast("str", mapping["site_ordering"]),
            duplicate_policy=cast("str", mapping["duplicate_policy"]),
            symmetry_restrictions=cast("str", mapping["symmetry_restrictions"]),
            cost_function_id=cast("str", mapping["cost_function_id"]),
            native_compilation_policy_id=cast("str", mapping["native_compilation_policy_id"]),
            connectivity=cast("str", mapping["connectivity"]),
            routing_policy_id=cast("str", mapping["routing_policy_id"]),
            selection_reuse_policy=cast("str", mapping["selection_reuse_policy"]),
            operators=tuple(PoolOperator.from_dict(item) for item in raw_operators),
        )
        if require_checksum(mapping["content_checksum"], "operator pool.content_checksum") != pool.content_checksum:
            msg = "operator pool content checksum mismatch."
            raise ValueError(msg)
        return pool


def build_projector_operator_pool(num_qubits: int) -> OperatorPoolSpec:
    """Build the exact family-wide projector/fidelity operator pool.

    Args:
        num_qubits: Register width.

    Returns:
        The frozen site-major ``RX/RY/RZ`` and edge-major ``RXX/RYY/RZZ``
        pool with no symmetry restriction.
    """
    qubits = _strict_int(num_qubits, "num_qubits", minimum=1)
    return OperatorPoolSpec(
        pool_id=_PROJECTOR_POOL_ID,
        method_id=ADAPT_STYLE_METHOD_ID,
        num_qubits=qubits,
        one_qubit_generators=("x", "y", "z"),
        two_qubit_generators=("xx", "yy", "zz"),
        site_ordering=_PROJECTOR_SITE_ORDER,
        duplicate_policy=_DUPLICATE_POLICY,
        symmetry_restrictions=_PROJECTOR_SYMMETRY,
        cost_function_id=PROJECTOR_COST_ID,
        native_compilation_policy_id=QUANTINUUM_NATIVE_POLICY_ID,
        connectivity=CONNECTIVITY_ID,
        routing_policy_id=ROUTING_POLICY_ID,
        selection_reuse_policy="without_replacement",
        operators=_expected_projector_operators(qubits),
    )


def build_tfim_real_operator_pool(num_qubits: int) -> OperatorPoolSpec:
    """Build the real-state TFIM pool used by genuine energy ADAPT-VQE.

    The ordered one-qubit choices are ``RY_i``.  The ordered two-qubit choices
    on each edge are rotations generated by ``Y_i Z_(i+1)`` and
    ``Z_i Y_(i+1)``.  Odd-Y Pauli strings preserve real states and each
    two-qubit rotation compiles to one native RZZ with local basis changes.

    Args:
        num_qubits: Register width.

    Returns:
        The checksum-sealed TFIM pool.
    """
    qubits = _strict_int(num_qubits, "num_qubits", minimum=1)
    return OperatorPoolSpec(
        pool_id=_TFIM_POOL_ID,
        method_id=ENERGY_ADAPT_METHOD_ID,
        num_qubits=qubits,
        one_qubit_generators=("y",),
        two_qubit_generators=("yz", "zy"),
        site_ordering=_TFIM_SITE_ORDER,
        duplicate_policy=_DUPLICATE_POLICY,
        symmetry_restrictions=_TFIM_SYMMETRY,
        cost_function_id=TFIM_ENERGY_COST_ID,
        native_compilation_policy_id=QUANTINUUM_NATIVE_POLICY_ID,
        connectivity=CONNECTIVITY_ID,
        routing_policy_id=ROUTING_POLICY_ID,
        selection_reuse_policy="without_replacement",
        operators=_expected_tfim_operators(qubits),
    )


# Readable aliases for callers that use the pool noun first.
projector_operator_pool = build_projector_operator_pool
tfim_real_operator_pool = build_tfim_real_operator_pool


@dataclass(frozen=True, slots=True)
class OperatorGrowthSpec:
    """Checksum-sealed selection, reoptimization, and stopping specification."""

    method_id: str
    pool_checksum: str
    gradient_tolerance: float = 1e-10
    max_operators: int = 16
    native_two_qubit_cap_per_edge: int | None = None
    reoptimization_steps: int = 100
    learning_rate: float = 0.08
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    reoptimization_rule_id: str = _INTERNAL_ADAM_ID
    selection_rule_id: str = field(default=_SELECTION_RULE_ID, init=False)
    tie_break_rule_id: str = field(default=_TIE_BREAK_RULE_ID, init=False)
    parameter_shift_rule_id: str = field(default=_PARAMETER_SHIFT_RULE_ID, init=False)
    initial_state_policy: str = field(default=_INITIAL_STATE_POLICY, init=False)
    schema_version: str = field(default=OPERATOR_GROWTH_SPEC_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate all optimizer choices without coercing numeric types."""
        method_id = require_slug(self.method_id, "method_id")
        if method_id not in _METHOD_IDS:
            msg = f"Unsupported operator-growth method {method_id!r}."
            raise ValueError(msg)
        pool_checksum = require_checksum(self.pool_checksum, "pool_checksum")
        tolerance = require_float(self.gradient_tolerance, "gradient_tolerance", minimum=0.0)
        max_operators = _strict_int(self.max_operators, "max_operators")
        cap = self.native_two_qubit_cap_per_edge
        if cap is not None:
            cap = _strict_int(cap, "native_two_qubit_cap_per_edge")
        steps = _strict_int(self.reoptimization_steps, "reoptimization_steps")
        learning_rate = require_float(self.learning_rate, "learning_rate", minimum=0.0)
        beta1 = require_float(self.adam_beta1, "adam_beta1", minimum=0.0, maximum=1.0)
        beta2 = require_float(self.adam_beta2, "adam_beta2", minimum=0.0, maximum=1.0)
        epsilon = require_float(self.adam_epsilon, "adam_epsilon", minimum=0.0)
        if beta1 >= 1.0 or beta2 >= 1.0 or epsilon <= 0.0:
            msg = "Adam beta values must be below one and epsilon must be positive."
            raise ValueError(msg)
        rule = require_slug(self.reoptimization_rule_id, "reoptimization_rule_id")
        fixed_rules = {
            "selection_rule_id": _SELECTION_RULE_ID,
            "tie_break_rule_id": _TIE_BREAK_RULE_ID,
            "parameter_shift_rule_id": _PARAMETER_SHIFT_RULE_ID,
            "initial_state_policy": _INITIAL_STATE_POLICY,
        }
        for name, expected in fixed_rules.items():
            if getattr(self, name) != expected:
                msg = f"{name} differs from the frozen WP20 rule."
                raise ValueError(msg)
        object.__setattr__(self, "method_id", method_id)
        object.__setattr__(self, "pool_checksum", pool_checksum)
        object.__setattr__(self, "gradient_tolerance", tolerance)
        object.__setattr__(self, "max_operators", max_operators)
        object.__setattr__(self, "native_two_qubit_cap_per_edge", cap)
        object.__setattr__(self, "reoptimization_steps", steps)
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "adam_beta1", beta1)
        object.__setattr__(self, "adam_beta2", beta2)
        object.__setattr__(self, "adam_epsilon", epsilon)
        object.__setattr__(self, "reoptimization_rule_id", rule)

    @classmethod
    def for_pool(
        cls,
        pool: OperatorPoolSpec,
        *,
        gradient_tolerance: float = 1e-10,
        max_operators: int = 16,
        native_two_qubit_cap_per_edge: int | None = None,
        reoptimization_steps: int = 100,
        learning_rate: float = 0.08,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        reoptimization_rule_id: str = _INTERNAL_ADAM_ID,
    ) -> OperatorGrowthSpec:
        """Construct a spec bound to one exact pool.

        Args:
            pool: Frozen pool to bind.
            gradient_tolerance: Absolute selection-gradient stopping value.
            max_operators: Maximum number of retained pool operators.
            native_two_qubit_cap_per_edge: Optional compiled RZZ cap per edge.
            reoptimization_steps: Full-parameter reoptimization steps per append.
            learning_rate: Adam learning rate.
            adam_beta1: Adam first-moment decay.
            adam_beta2: Adam second-moment decay.
            adam_epsilon: Adam denominator regularizer.
            reoptimization_rule_id: Sealed optimizer or callback identifier.

        Returns:
            A checksum-sealed growth specification.
        """
        if not isinstance(pool, OperatorPoolSpec):
            msg = "pool must be an OperatorPoolSpec."
            raise TypeError(msg)
        return cls(
            method_id=pool.method_id,
            pool_checksum=pool.content_checksum,
            gradient_tolerance=gradient_tolerance,
            max_operators=max_operators,
            native_two_qubit_cap_per_edge=native_two_qubit_cap_per_edge,
            reoptimization_steps=reoptimization_steps,
            learning_rate=learning_rate,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            reoptimization_rule_id=reoptimization_rule_id,
        )

    def _content_dict(self) -> dict[str, object]:
        """Return all checksum-covered growth choices."""
        return {
            "schema_version": self.schema_version,
            "method_id": self.method_id,
            "pool_checksum": self.pool_checksum,
            "selection_rule_id": self.selection_rule_id,
            "tie_break_rule_id": self.tie_break_rule_id,
            "parameter_shift_rule_id": self.parameter_shift_rule_id,
            "reoptimization_rule_id": self.reoptimization_rule_id,
            "initial_state_policy": self.initial_state_policy,
            "gradient_tolerance": self.gradient_tolerance,
            "max_operators": self.max_operators,
            "native_two_qubit_cap_per_edge": self.native_two_qubit_cap_per_edge,
            "reoptimization_steps": self.reoptimization_steps,
            "learning_rate": self.learning_rate,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_epsilon": self.adam_epsilon,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum sealing the complete algorithm specification."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return the complete sealed specification document."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, value: object) -> OperatorGrowthSpec:
        """Decode and verify one strict specification document."""
        mapping = freeze_json_mapping(require_mapping(value, "operator growth spec"), "operator growth spec")
        require_exact_keys(mapping, _GROWTH_SPEC_KEYS, "operator growth spec")
        if mapping["schema_version"] != OPERATOR_GROWTH_SPEC_SCHEMA_VERSION:
            msg = "operator growth spec uses an unsupported schema version."
            raise ValueError(msg)
        spec = cls(
            method_id=cast("str", mapping["method_id"]),
            pool_checksum=cast("str", mapping["pool_checksum"]),
            gradient_tolerance=cast("float", mapping["gradient_tolerance"]),
            max_operators=cast("int", mapping["max_operators"]),
            native_two_qubit_cap_per_edge=cast("int | None", mapping["native_two_qubit_cap_per_edge"]),
            reoptimization_steps=cast("int", mapping["reoptimization_steps"]),
            learning_rate=cast("float", mapping["learning_rate"]),
            adam_beta1=cast("float", mapping["adam_beta1"]),
            adam_beta2=cast("float", mapping["adam_beta2"]),
            adam_epsilon=cast("float", mapping["adam_epsilon"]),
            reoptimization_rule_id=cast("str", mapping["reoptimization_rule_id"]),
        )
        frozen_values = {
            "selection_rule_id": spec.selection_rule_id,
            "tie_break_rule_id": spec.tie_break_rule_id,
            "parameter_shift_rule_id": spec.parameter_shift_rule_id,
            "initial_state_policy": spec.initial_state_policy,
        }
        if any(mapping[name] != expected for name, expected in frozen_values.items()):
            msg = "operator growth spec changes a frozen selection rule."
            raise ValueError(msg)
        if (
            require_checksum(mapping["content_checksum"], "operator growth spec.content_checksum")
            != spec.content_checksum
        ):
            msg = "operator growth spec content checksum mismatch."
            raise ValueError(msg)
        return spec


@dataclass(frozen=True, slots=True)
class ProjectorObjectiveSpec:
    """Checksum binding for a target projector and fixed initial state."""

    num_qubits: int
    target_state_checksum: str
    initial_state_checksum: str
    objective_id: str = field(default=PROJECTOR_COST_ID, init=False)
    schema_version: str = field(default=PROJECTOR_OBJECTIVE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the objective identity record."""
        object.__setattr__(self, "num_qubits", _strict_int(self.num_qubits, "num_qubits", minimum=1))
        object.__setattr__(
            self,
            "target_state_checksum",
            require_checksum(self.target_state_checksum, "target_state_checksum"),
        )
        object.__setattr__(
            self,
            "initial_state_checksum",
            require_checksum(self.initial_state_checksum, "initial_state_checksum"),
        )

    @classmethod
    def from_states(
        cls,
        target_state: object,
        initial_state: object | None = None,
    ) -> ProjectorObjectiveSpec:
        """Construct the binding for exact normalized state operands."""
        target, qubits = _normalized_state(target_state, "target_state")
        initial = (
            computational_zero_state(qubits)
            if initial_state is None
            else _normalized_state(
                initial_state,
                "initial_state",
                num_qubits=qubits,
            )[0]
        )
        return cls(qubits, _array_checksum(target), _array_checksum(initial))

    def _content_dict(self) -> dict[str, object]:
        """Return all checksum-covered projector fields."""
        return {
            "schema_version": self.schema_version,
            "objective_id": self.objective_id,
            "num_qubits": self.num_qubits,
            "target_state_checksum": self.target_state_checksum,
            "initial_state_checksum": self.initial_state_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum binding the objective to both state operands."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return the sealed objective binding."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, value: object) -> ProjectorObjectiveSpec:
        """Decode and verify a projector objective binding."""
        mapping = require_mapping(value, "projector objective")
        require_exact_keys(mapping, _PROJECTOR_OBJECTIVE_KEYS, "projector objective")
        if (
            mapping["schema_version"] != PROJECTOR_OBJECTIVE_SCHEMA_VERSION
            or mapping["objective_id"] != PROJECTOR_COST_ID
        ):
            msg = "projector objective identity differs from the frozen definition."
            raise ValueError(msg)
        spec = cls(
            num_qubits=cast("int", mapping["num_qubits"]),
            target_state_checksum=cast("str", mapping["target_state_checksum"]),
            initial_state_checksum=cast("str", mapping["initial_state_checksum"]),
        )
        if (
            require_checksum(mapping["content_checksum"], "projector objective.content_checksum")
            != spec.content_checksum
        ):
            msg = "projector objective content checksum mismatch."
            raise ValueError(msg)
        return spec


def dense_open_chain_tfim_hamiltonian(
    couplings: Sequence[float] | NDArray[np.float64],
    fields: Sequence[float] | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Construct ``-sum J_i Z_i Z_(i+1) - sum h_i X_i`` exactly.

    Site ``i`` is bit ``i`` of the dense little-endian basis, matching the
    sealed Phase II target generator.

    Args:
        couplings: Open-chain nearest-neighbor couplings, of length ``n - 1``.
        fields: On-site transverse fields, of length ``n``.

    Returns:
        A real symmetric dense Hamiltonian.
    """
    try:
        coupling_array = np.asarray(couplings, dtype=np.float64)
        field_array = np.asarray(fields, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = "couplings and fields must be real numeric sequences."
        raise TypeError(msg) from error
    if field_array.ndim != 1 or field_array.size == 0 or coupling_array.shape != (field_array.size - 1,):
        msg = "An open-chain TFIM requires n fields and exactly n - 1 couplings."
        raise ValueError(msg)
    if not np.all(np.isfinite(coupling_array)) or not np.all(np.isfinite(field_array)):
        msg = "TFIM couplings and fields must be finite."
        raise ValueError(msg)
    num_qubits = int(field_array.size)
    dimension = 2**num_qubits
    basis = np.arange(dimension, dtype=np.int64)
    hamiltonian = np.zeros((dimension, dimension), dtype=np.float64)
    diagonal = np.zeros(dimension, dtype=np.float64)
    for site, coupling in enumerate(coupling_array):
        left = (basis >> site) & 1
        right = (basis >> (site + 1)) & 1
        diagonal -= float(coupling) * np.where(left == right, 1.0, -1.0)
    hamiltonian[basis, basis] = diagonal
    for site, field_value in enumerate(field_array):
        hamiltonian[basis, basis ^ (1 << site)] -= float(field_value)
    return hamiltonian


# Short public alias matching the physics noun order.
open_chain_tfim_hamiltonian = dense_open_chain_tfim_hamiltonian


@dataclass(frozen=True, slots=True)
class TFIMHamiltonianSpec:
    """Checksum-sealed physical binding for genuine energy ADAPT-VQE."""

    couplings: tuple[float, ...]
    fields: tuple[float, ...]
    initial_state_checksum: str
    objective_id: str = field(default=TFIM_ENERGY_COST_ID, init=False)
    boundary_condition: str = field(default="open", init=False)
    basis_bit_order: str = field(default="little_endian_site_i_is_bit_i", init=False)
    target_state_binding: str = field(default="forbidden_hamiltonian_parameters_only", init=False)
    schema_version: str = field(default=TFIM_HAMILTONIAN_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate physical arrays and their fixed-state binding."""
        fields = _float_tuple(self.fields, "fields")
        if not fields:
            msg = "fields must contain at least one value."
            raise ValueError(msg)
        couplings = _float_tuple(self.couplings, "couplings", length=len(fields) - 1)
        # Construct once so shape and finiteness are checked by the public oracle.
        dense_open_chain_tfim_hamiltonian(couplings, fields)
        object.__setattr__(self, "couplings", couplings)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(
            self,
            "initial_state_checksum",
            require_checksum(self.initial_state_checksum, "initial_state_checksum"),
        )

    @property
    def num_qubits(self) -> int:
        """Number of sites in the bound Hamiltonian."""
        return len(self.fields)

    def dense_matrix(self) -> NDArray[np.float64]:
        """Return a detached dense Hamiltonian matrix."""
        return dense_open_chain_tfim_hamiltonian(self.couplings, self.fields)

    @property
    def hamiltonian_checksum(self) -> str:
        """Checksum of exact little-endian float64 matrix bytes."""
        matrix = np.ascontiguousarray(self.dense_matrix(), dtype=np.dtype("<f8"))
        return f"sha256:{hashlib.sha256(matrix.tobytes(order='C')).hexdigest()}"

    def _content_dict(self) -> dict[str, object]:
        """Return every Hamiltonian-bound objective field."""
        return {
            "schema_version": self.schema_version,
            "objective_id": self.objective_id,
            "num_qubits": self.num_qubits,
            "couplings": list(self.couplings),
            "fields": list(self.fields),
            "boundary_condition": self.boundary_condition,
            "basis_bit_order": self.basis_bit_order,
            "target_state_binding": self.target_state_binding,
            "initial_state_checksum": self.initial_state_checksum,
            "hamiltonian_checksum": self.hamiltonian_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum binding selection exclusively to Hamiltonian parameters."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return the strict sealed Hamiltonian binding."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, value: object) -> TFIMHamiltonianSpec:
        """Decode and verify a TFIM Hamiltonian binding."""
        mapping = require_mapping(value, "TFIM Hamiltonian")
        require_exact_keys(mapping, _TFIM_HAMILTONIAN_KEYS, "TFIM Hamiltonian")
        expected_constants = {
            "schema_version": TFIM_HAMILTONIAN_SCHEMA_VERSION,
            "objective_id": TFIM_ENERGY_COST_ID,
            "boundary_condition": "open",
            "basis_bit_order": "little_endian_site_i_is_bit_i",
            "target_state_binding": "forbidden_hamiltonian_parameters_only",
        }
        if any(mapping[name] != expected for name, expected in expected_constants.items()):
            msg = "TFIM Hamiltonian binding changes a frozen physical convention."
            raise ValueError(msg)
        spec = cls(
            couplings=cast("tuple[float, ...]", mapping["couplings"]),
            fields=cast("tuple[float, ...]", mapping["fields"]),
            initial_state_checksum=cast("str", mapping["initial_state_checksum"]),
        )
        if mapping["num_qubits"] != spec.num_qubits or mapping["hamiltonian_checksum"] != spec.hamiltonian_checksum:
            msg = "TFIM Hamiltonian dimensions or dense matrix checksum changed."
            raise ValueError(msg)
        if require_checksum(mapping["content_checksum"], "TFIM Hamiltonian.content_checksum") != spec.content_checksum:
            msg = "TFIM Hamiltonian content checksum mismatch."
            raise ValueError(msg)
        return spec


@dataclass(frozen=True, slots=True)
class TargetBoundTFIMEnergyObjectiveSpec:
    """Authorized Phase II target specification bound to its TFIM objective."""

    target_instance_spec: TargetInstanceSpec
    target_manifest_checksum: str
    target_vector_checksum: str
    hamiltonian_binding: TFIMHamiltonianSpec
    schema_version: str = field(default=TARGET_BOUND_TFIM_OBJECTIVE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Verify target physics, identity, and computational-zero input."""
        if not isinstance(self.target_instance_spec, TargetInstanceSpec):
            msg = "target_instance_spec must be a TargetInstanceSpec."
            raise TypeError(msg)
        spec = self.target_instance_spec
        if spec.family_id != TFIM_FAMILY_ID:
            msg = "Target-bound energy ADAPT requires a TFIM target specification."
            raise ValueError(msg)
        if not isinstance(self.hamiltonian_binding, TFIMHamiltonianSpec):
            msg = "hamiltonian_binding must be a TFIMHamiltonianSpec."
            raise TypeError(msg)
        manifest_checksum = require_checksum(self.target_manifest_checksum, "target_manifest_checksum")
        vector_checksum = require_checksum(self.target_vector_checksum, "target_vector_checksum")
        parameters = spec.parameters
        expected_couplings = _float_tuple(parameters["couplings"], "target TFIM couplings", length=spec.qubit_count - 1)
        expected_fields = _float_tuple(parameters["fields"], "target TFIM fields", length=spec.qubit_count)
        expected_initial = _array_checksum(computational_zero_state(spec.qubit_count))
        if (
            self.hamiltonian_binding.couplings != expected_couplings
            or self.hamiltonian_binding.fields != expected_fields
            or self.hamiltonian_binding.initial_state_checksum != expected_initial
        ):
            msg = "Hamiltonian binding does not reproduce the authorized TFIM target specification."
            raise ValueError(msg)
        object.__setattr__(self, "target_manifest_checksum", manifest_checksum)
        object.__setattr__(self, "target_vector_checksum", vector_checksum)

    @classmethod
    def from_target(
        cls,
        target: MaterializedTarget,
        target_instance_spec: TargetInstanceSpec,
    ) -> TargetBoundTFIMEnergyObjectiveSpec:
        """Construct the exact target/Hamiltonian binding from authorized inputs."""
        if not isinstance(target, MaterializedTarget):
            msg = "target must be an authorized Phase II MaterializedTarget."
            raise TypeError(msg)
        if not isinstance(target_instance_spec, TargetInstanceSpec):
            msg = "target_instance_spec must be a TargetInstanceSpec."
            raise TypeError(msg)
        spec = target_instance_spec
        if (
            spec.family_id != TFIM_FAMILY_ID
            or target.family_id != TFIM_FAMILY_ID
            or target.target_instance_id != spec.target_instance_id
            or target.target_instance_spec_checksum != spec.content_checksum
            or target.population_config_checksum != spec.population_config_checksum
            or target.parameter_checksum != canonical_checksum(spec.parameters)
            or target.stratum_id != spec.stratum_id
            or target.qubit_count != spec.qubit_count
        ):
            msg = "Materialized target does not match the authorized TFIM target specification."
            raise ValueError(msg)
        couplings = _float_tuple(spec.parameters["couplings"], "target TFIM couplings")
        fields = _float_tuple(spec.parameters["fields"], "target TFIM fields")
        hamiltonian = TFIMHamiltonianSpec(
            couplings=couplings,
            fields=fields,
            initial_state_checksum=_array_checksum(computational_zero_state(spec.qubit_count)),
        )
        target_energy = tfim_energy(
            target.state_vector_copy(),
            dense_open_chain_tfim_hamiltonian(hamiltonian.couplings, hamiltonian.fields),
        )
        ground_energy = require_float(spec.parameters["ground_energy"], "target TFIM ground_energy")
        if not math.isclose(target_energy, ground_energy, rel_tol=1e-10, abs_tol=1e-10):
            msg = "Materialized target does not reproduce the authorized TFIM ground energy."
            raise ValueError(msg)
        return cls(
            target_instance_spec=spec,
            target_manifest_checksum=target.target_manifest_checksum,
            target_vector_checksum=target.vector_checksum,
            hamiltonian_binding=hamiltonian,
        )

    def _content_dict(self) -> dict[str, object]:
        """Return the complete target-bound energy objective document."""
        return {
            "schema_version": self.schema_version,
            "target_instance_spec": self.target_instance_spec.to_dict(),
            "target_manifest_checksum": self.target_manifest_checksum,
            "target_vector_checksum": self.target_vector_checksum,
            "hamiltonian_binding": self.hamiltonian_binding.to_dict(),
        }

    @property
    def content_checksum(self) -> str:
        """Checksum sealing target identity, physics, and Hamiltonian binding."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return the strict checksum-sealed objective binding."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, value: object) -> TargetBoundTFIMEnergyObjectiveSpec:
        """Decode and verify one target-bound TFIM objective."""
        mapping = require_mapping(value, "target-bound TFIM objective")
        require_exact_keys(mapping, _TARGET_BOUND_TFIM_OBJECTIVE_KEYS, "target-bound TFIM objective")
        if mapping["schema_version"] != TARGET_BOUND_TFIM_OBJECTIVE_SCHEMA_VERSION:
            msg = "target-bound TFIM objective uses an unsupported schema version."
            raise ValueError(msg)
        binding = cls(
            target_instance_spec=TargetInstanceSpec.from_dict(mapping["target_instance_spec"]),
            target_manifest_checksum=cast("str", mapping["target_manifest_checksum"]),
            target_vector_checksum=cast("str", mapping["target_vector_checksum"]),
            hamiltonian_binding=TFIMHamiltonianSpec.from_dict(mapping["hamiltonian_binding"]),
        )
        if require_checksum(mapping["content_checksum"], "target-bound objective.content_checksum") != (
            binding.content_checksum
        ):
            msg = "target-bound TFIM objective content checksum mismatch."
            raise ValueError(msg)
        return binding


def projector_infidelity(target_state: object, prepared_state: object) -> float:
    """Evaluate the actual pure-state projector cost ``1 - |<t|p>|^2``.

    Args:
        target_state: Normalized target statevector.
        prepared_state: Normalized candidate statevector of the same width.

    Returns:
        Projector infidelity, clipped only for roundoff at zero and one.
    """
    target, qubits = _normalized_state(target_state, "target_state")
    prepared, _ = _normalized_state(prepared_state, "prepared_state", num_qubits=qubits)
    fidelity = float(abs(np.vdot(target, prepared)) ** 2)
    return float(np.clip(1.0 - fidelity, 0.0, 1.0))


def tfim_energy(state: object, hamiltonian: object) -> float:
    """Evaluate a normalized state's expectation of a dense TFIM Hamiltonian.

    Args:
        state: Normalized statevector.
        hamiltonian: Finite Hermitian matrix of matching dimension.

    Returns:
        The real energy expectation.
    """
    vector, _ = _normalized_state(state, "state")
    try:
        matrix = np.asarray(hamiltonian, dtype=np.complex128)
    except (TypeError, ValueError) as error:
        msg = "hamiltonian must be convertible to a dense complex matrix."
        raise TypeError(msg) from error
    if matrix.shape != (vector.size, vector.size) or not np.all(np.isfinite(matrix)):
        msg = "hamiltonian shape or entries are invalid for the supplied state."
        raise ValueError(msg)
    if not np.allclose(matrix, matrix.conj().T, rtol=0.0, atol=1e-12):
        msg = "hamiltonian must be Hermitian."
        raise ValueError(msg)
    value = np.vdot(vector, matrix @ vector)
    if abs(value.imag) > 1e-10:
        msg = "Hermitian energy acquired a non-negligible imaginary component."
        raise ValueError(msg)
    return float(value.real)


def _apply_pauli(state: NDArray[np.complex128], operator: PoolOperator) -> NDArray[np.complex128]:
    """Apply one Pauli product without constructing a dense matrix."""
    result = np.zeros_like(state)
    for basis_index, amplitude in enumerate(state):
        output_index = basis_index
        phase = 1.0 + 0.0j
        for pauli, site in zip(operator.generator, operator.sites, strict=True):
            bit = (basis_index >> site) & 1
            if pauli == "x":
                output_index ^= 1 << site
            elif pauli == "y":
                output_index ^= 1 << site
                phase *= 1.0j if bit == 0 else -1.0j
            else:
                phase *= 1.0 if bit == 0 else -1.0
        result[output_index] = phase * amplitude
    return result


def operator_growth_state(
    num_qubits: int,
    operators: Sequence[PoolOperator],
    parameters: Sequence[float] | NDArray[np.float64],
    *,
    initial_state: object | None = None,
) -> NDArray[np.complex128]:
    """Apply an ordered sequence of Pauli rotations to a fixed input state.

    Args:
        num_qubits: Register width.
        operators: Selected pool operators in circuit order.
        parameters: One rotation angle per selected operator.
        initial_state: Optional normalized input; defaults to all-zero.

    Returns:
        The prepared dense statevector.
    """
    qubits = _strict_int(num_qubits, "num_qubits", minimum=1)
    selected = tuple(operators)
    if any(not isinstance(operator, PoolOperator) for operator in selected):
        msg = "operators must contain only PoolOperator records."
        raise TypeError(msg)
    if any(max(operator.sites) >= qubits for operator in selected):
        msg = "Selected operator lies outside the declared register."
        raise ValueError(msg)
    try:
        theta = np.asarray(parameters, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = "parameters must be convertible to a float64 vector."
        raise TypeError(msg) from error
    if theta.shape != (len(selected),) or not np.all(np.isfinite(theta)):
        msg = "parameters must contain one finite angle per selected operator."
        raise ValueError(msg)
    state = (
        computational_zero_state(qubits)
        if initial_state is None
        else _normalized_state(
            initial_state,
            "initial_state",
            num_qubits=qubits,
        )[0].copy()
    )
    for operator, angle in zip(selected, theta, strict=True):
        pauli_state = _apply_pauli(state, operator)
        state = math.cos(float(angle) / 2.0) * state - 1.0j * math.sin(float(angle) / 2.0) * pauli_state
    return np.asarray(state, dtype=np.complex128)


def materialize_operator_growth_circuit(
    num_qubits: int,
    operators: Sequence[PoolOperator],
) -> ParameterizedCircuit:
    """Materialize selected Pauli rotations under the frozen compiler contract.

    ``RX``, ``RY``, ``RZ``, ``RXX``, ``RYY``, and ``RZZ`` are emitted directly.
    The TFIM-only ``YZ`` and ``ZY`` rotations are represented exactly by an
    ``RX(pi/2)`` basis change on the Y site, one parameterized ``RZZ``, and the
    inverse ``RX(-pi/2)`` basis change.  Consequently the returned circuit can
    be passed directly to :func:`measure_circuit_resources` and the frozen
    Quantinuum compiler without extending its accepted logical gate set.

    Args:
        num_qubits: Register width.
        operators: Ordered selected pool operators.

    Returns:
        A parameterized circuit with one trainable parameter per operator.
    """
    qubits = _strict_int(num_qubits, "num_qubits", minimum=1)
    selected = tuple(operators)
    if any(not isinstance(operator, PoolOperator) for operator in selected):
        msg = "operators must contain only PoolOperator records."
        raise TypeError(msg)
    if any(max(operator.sites) >= qubits for operator in selected):
        msg = "Selected operator lies outside the declared register."
        raise ValueError(msg)
    operator_ids = tuple(operator.operator_id for operator in selected)
    if len(operator_ids) != len(set(operator_ids)):
        msg = "Operator-growth circuits cannot contain duplicate pool operators."
        raise ValueError(msg)

    gates: list[ParameterizedGate] = []
    for parameter_index, operator in enumerate(selected):
        if operator.generator in {"x", "y", "z", "xx", "yy", "zz"}:
            gates.append(
                ParameterizedGate(
                    name=f"r{operator.generator}",
                    sites=operator.sites,
                    param_index=parameter_index,
                    logical_gate_id=operator.operator_id,
                )
            )
            continue

        y_offset = 0 if operator.generator == "yz" else 1
        y_site = operator.sites[y_offset]
        gates.extend((
            ParameterizedGate(
                name="rx",
                sites=(y_site,),
                angle_offset=math.pi / 2.0,
                logical_gate_id=f"{operator.operator_id}_basis_before",
                noise_enabled=False,
            ),
            ParameterizedGate(
                name="rzz",
                sites=operator.sites,
                param_index=parameter_index,
                logical_gate_id=operator.operator_id,
            ),
            ParameterizedGate(
                name="rx",
                sites=(y_site,),
                angle_offset=-math.pi / 2.0,
                logical_gate_id=f"{operator.operator_id}_basis_after",
                noise_enabled=False,
            ),
        ))
    return ParameterizedCircuit(num_qubits=qubits, gates=gates, num_params=len(selected))


def _standard_noise_condition_checksum(
    noise_id: str,
    noise_definition_version: str,
    noise_strength_scale: float,
    tjm_dt: float,
    initial_state_checksum: str,
) -> str:
    """Return the complete frozen standard-noise runtime checksum."""
    return canonical_checksum({
        "noise_id": noise_id,
        "noise_definition_version": noise_definition_version,
        "noise_strength_scale": noise_strength_scale,
        "tjm_dt": tjm_dt,
        "placement": "logical_parameterized_gates",
        "apply_noise_to": "all",
        "noisy_gate_index_policy": "trainable_and_noise_enabled",
        "trajectory_update": "independent",
        "differentiate_jump_normalization": False,
        "use_crn_flag": False,
        "common_randomness_mechanism": "repeat_identical_seed_and_iteration_zero",
        "initial_state_checksum": initial_state_checksum,
        "truncation": {
            "max_bond_dim": None,
            "svd_threshold": 0.0,
            "trunc_mode": "discarded_weight",
            "min_bond_dim": 1,
        },
    })


def _standard_trajectory_ensemble_checksum(
    *,
    target_instance_id: str,
    target_vector_checksum: str,
    provider_checksum: str,
    objective_checksum: str,
    noise_condition_checksum: str,
    trajectory_count: int,
    trajectory_seed: int,
) -> str:
    """Return the fixed common-seed trajectory-stream identity."""
    return canonical_checksum({
        "derivation_version": "yaqs.state_preparation.phase2.operator_growth_common_seed.v1",
        "target_instance_id": target_instance_id,
        "target_vector_checksum": target_vector_checksum,
        "provider_checksum": provider_checksum,
        "objective_checksum": objective_checksum,
        "noise_condition_checksum": noise_condition_checksum,
        "trajectory_count": trajectory_count,
        "trajectory_seed": trajectory_seed,
    })


@dataclass(frozen=True, slots=True)
class OperatorGrowthTrainingProvenance:
    """Provider, objective, and fixed-trajectory identity for noisy growth."""

    target_instance_id: str
    target_family_id: str
    qubit_count: int
    target_vector_checksum: str
    initial_state_checksum: str
    optimization_block_id: str
    optimization_seed: int
    resource_stratum_id: str
    provider_id: str
    provider_checksum: str
    objective_id: str
    objective_checksum: str
    noise_id: str
    noise_definition_version: str
    noise_strength_scale: float
    tjm_dt: float
    noise_condition_checksum: str
    trajectory_count: int
    trajectory_seed: int
    trajectory_ensemble_checksum: str
    sampling_policy_id: str = _NOISY_SAMPLING_POLICY_ID
    trajectory_gate_counting_policy_id: str = _TRAJECTORY_GATE_COUNTING_POLICY_ID
    schema_version: str = field(default=OPERATOR_GROWTH_TRAINING_PROVENANCE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require a nonempty noisy ensemble and checksum-bound dependencies."""
        target_instance_id = require_string(self.target_instance_id, "target_instance_id")
        target_family_id = require_slug(self.target_family_id, "target_family_id")
        qubits = require_int(self.qubit_count, "qubit_count", minimum=1)
        target_checksum = require_checksum(self.target_vector_checksum, "target_vector_checksum")
        initial_checksum = require_checksum(self.initial_state_checksum, "initial_state_checksum")
        optimization_block_id = require_slug(self.optimization_block_id, "optimization_block_id")
        optimization_seed = require_int(self.optimization_seed, "optimization_seed")
        if optimization_seed >= 2**64:
            msg = "optimization_seed must fit an unsigned 64-bit integer."
            raise ValueError(msg)
        resource_stratum_id = require_slug(self.resource_stratum_id, "resource_stratum_id")
        if initial_checksum != _array_checksum(computational_zero_state(qubits)):
            msg = "Noisy operator growth requires the computational-zero initial state."
            raise ValueError(msg)
        provider_id = require_slug(self.provider_id, "provider_id")
        provider_checksum = require_checksum(self.provider_checksum, "provider_checksum")
        objective_id = require_slug(self.objective_id, "objective_id")
        objective_checksum = require_checksum(self.objective_checksum, "objective_checksum")
        noise_id = require_slug(self.noise_id, "noise_id")
        if noise_id not in STANDARD_NOISE_IDS:
            msg = "Operator-growth training provenance requires a standard fixed-rate noise ID."
            raise ValueError(msg)
        if self.noise_definition_version != FIXED_RATE_NOISE_DEFINITION_VERSION:
            msg = f"noise_definition_version must be {FIXED_RATE_NOISE_DEFINITION_VERSION!r}."
            raise ValueError(msg)
        strength = require_float(self.noise_strength_scale, "noise_strength_scale", minimum=0.0)
        dt = require_float(self.tjm_dt, "tjm_dt", minimum=0.0)
        if strength <= 0.0 or dt <= 0.0:
            msg = "noise_strength_scale and tjm_dt must be strictly positive."
            raise ValueError(msg)
        if (
            objective_id != PROJECTOR_COST_ID
            or objective_checksum
            != ProjectorObjectiveSpec(
                qubits,
                target_checksum,
                initial_checksum,
            ).content_checksum
        ):
            msg = "objective provenance does not reproduce the exact target projector binding."
            raise ValueError(msg)
        expected_provider = create_scaled_standard_noise_provider(noise_id, strength)
        if provider_id != f"scaled_standard_{noise_id}" or provider_checksum != expected_provider.content_checksum:
            msg = "provider provenance does not reproduce the standard scaled provider."
            raise ValueError(msg)
        for name in (
            "noise_condition_checksum",
            "trajectory_ensemble_checksum",
        ):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        object.__setattr__(
            self,
            "trajectory_count",
            require_int(self.trajectory_count, "trajectory_count", minimum=1),
        )
        seed = require_int(self.trajectory_seed, "trajectory_seed")
        if seed >= 2**64:
            msg = "trajectory_seed must fit an unsigned 64-bit integer."
            raise ValueError(msg)
        object.__setattr__(self, "trajectory_seed", seed)
        expected_noise_condition = _standard_noise_condition_checksum(
            noise_id,
            self.noise_definition_version,
            strength,
            dt,
            initial_checksum,
        )
        if self.noise_condition_checksum != expected_noise_condition:
            msg = "noise_condition_checksum does not reproduce the fixed standard-noise runtime."
            raise ValueError(msg)
        expected_ensemble = _standard_trajectory_ensemble_checksum(
            target_instance_id=target_instance_id,
            target_vector_checksum=target_checksum,
            provider_checksum=provider_checksum,
            objective_checksum=objective_checksum,
            noise_condition_checksum=expected_noise_condition,
            trajectory_count=self.trajectory_count,
            trajectory_seed=seed,
        )
        if self.trajectory_ensemble_checksum != expected_ensemble:
            msg = "trajectory_ensemble_checksum does not reproduce the common-seed stream."
            raise ValueError(msg)
        if self.sampling_policy_id != _NOISY_SAMPLING_POLICY_ID:
            msg = f"sampling_policy_id must be {_NOISY_SAMPLING_POLICY_ID!r}."
            raise ValueError(msg)
        if self.trajectory_gate_counting_policy_id != _TRAJECTORY_GATE_COUNTING_POLICY_ID:
            msg = "trajectory_gate_counting_policy_id must be the frozen logical-gate-application policy."
            raise ValueError(msg)
        object.__setattr__(self, "target_instance_id", target_instance_id)
        object.__setattr__(self, "target_family_id", target_family_id)
        object.__setattr__(self, "qubit_count", qubits)
        object.__setattr__(self, "target_vector_checksum", target_checksum)
        object.__setattr__(self, "initial_state_checksum", initial_checksum)
        object.__setattr__(self, "optimization_block_id", optimization_block_id)
        object.__setattr__(self, "optimization_seed", optimization_seed)
        object.__setattr__(self, "resource_stratum_id", resource_stratum_id)
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "provider_checksum", provider_checksum)
        object.__setattr__(self, "objective_id", objective_id)
        object.__setattr__(self, "objective_checksum", objective_checksum)
        object.__setattr__(self, "noise_id", noise_id)
        object.__setattr__(self, "noise_strength_scale", strength)
        object.__setattr__(self, "tjm_dt", dt)

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered noisy-training field."""
        return {
            "schema_version": self.schema_version,
            "target_instance_id": self.target_instance_id,
            "target_family_id": self.target_family_id,
            "qubit_count": self.qubit_count,
            "target_vector_checksum": self.target_vector_checksum,
            "initial_state_checksum": self.initial_state_checksum,
            "optimization_block_id": self.optimization_block_id,
            "optimization_seed": self.optimization_seed,
            "resource_stratum_id": self.resource_stratum_id,
            "provider_id": self.provider_id,
            "provider_checksum": self.provider_checksum,
            "objective_id": self.objective_id,
            "objective_checksum": self.objective_checksum,
            "noise_id": self.noise_id,
            "noise_definition_version": self.noise_definition_version,
            "noise_strength_scale": self.noise_strength_scale,
            "tjm_dt": self.tjm_dt,
            "noise_condition_checksum": self.noise_condition_checksum,
            "trajectory_count": self.trajectory_count,
            "trajectory_seed": self.trajectory_seed,
            "trajectory_ensemble_checksum": self.trajectory_ensemble_checksum,
            "sampling_policy_id": self.sampling_policy_id,
            "trajectory_gate_counting_policy_id": self.trajectory_gate_counting_policy_id,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum sealing the complete noisy-training identity."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return strict JSON-native noisy-training provenance."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, value: object) -> OperatorGrowthTrainingProvenance:
        """Decode and verify noisy-training provenance."""
        mapping = require_mapping(value, "operator-growth training provenance")
        require_exact_keys(mapping, _TRAINING_PROVENANCE_KEYS, "operator-growth training provenance")
        if mapping["schema_version"] != OPERATOR_GROWTH_TRAINING_PROVENANCE_SCHEMA_VERSION:
            msg = "operator-growth training provenance uses an unsupported schema version."
            raise ValueError(msg)
        provenance = cls(
            target_instance_id=cast("str", mapping["target_instance_id"]),
            target_family_id=cast("str", mapping["target_family_id"]),
            qubit_count=cast("int", mapping["qubit_count"]),
            target_vector_checksum=cast("str", mapping["target_vector_checksum"]),
            initial_state_checksum=cast("str", mapping["initial_state_checksum"]),
            optimization_block_id=cast("str", mapping["optimization_block_id"]),
            optimization_seed=cast("int", mapping["optimization_seed"]),
            resource_stratum_id=cast("str", mapping["resource_stratum_id"]),
            provider_id=cast("str", mapping["provider_id"]),
            provider_checksum=cast("str", mapping["provider_checksum"]),
            objective_id=cast("str", mapping["objective_id"]),
            objective_checksum=cast("str", mapping["objective_checksum"]),
            noise_id=cast("str", mapping["noise_id"]),
            noise_definition_version=cast("str", mapping["noise_definition_version"]),
            noise_strength_scale=cast("float", mapping["noise_strength_scale"]),
            tjm_dt=cast("float", mapping["tjm_dt"]),
            noise_condition_checksum=cast("str", mapping["noise_condition_checksum"]),
            trajectory_count=cast("int", mapping["trajectory_count"]),
            trajectory_seed=cast("int", mapping["trajectory_seed"]),
            trajectory_ensemble_checksum=cast("str", mapping["trajectory_ensemble_checksum"]),
            sampling_policy_id=cast("str", mapping["sampling_policy_id"]),
            trajectory_gate_counting_policy_id=cast("str", mapping["trajectory_gate_counting_policy_id"]),
        )
        if require_checksum(mapping["content_checksum"], "training provenance.content_checksum") != (
            provenance.content_checksum
        ):
            msg = "operator-growth training provenance content checksum mismatch."
            raise ValueError(msg)
        return provenance


@dataclass(frozen=True, slots=True)
class StandardFixedRateOperatorGrowthEvaluatorBinding:
    """Persisted authorized-target and standard-noise evaluator binding."""

    target_instance_id: str
    target_instance_spec_checksum: str
    population_config_checksum: str
    target_manifest_checksum: str
    parameter_checksum: str
    target_family_id: str
    target_stratum_id: str
    qubit_count: int
    norm: float
    target_vector_checksum: str
    training_provenance: OperatorGrowthTrainingProvenance
    schema_version: str = field(default=STANDARD_FIXED_RATE_OPERATOR_GROWTH_BINDING_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Verify complete target identity against noisy training provenance."""
        target_id = require_string(self.target_instance_id, "target_instance_id")
        for name in (
            "target_instance_spec_checksum",
            "population_config_checksum",
            "target_manifest_checksum",
            "parameter_checksum",
            "target_vector_checksum",
        ):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        family = require_slug(self.target_family_id, "target_family_id")
        stratum = require_slug(self.target_stratum_id, "target_stratum_id")
        qubits = require_int(self.qubit_count, "qubit_count", minimum=1)
        norm = require_float(self.norm, "norm", minimum=0.0)
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-12):
            msg = "Target-bound evaluator target norm must equal one."
            raise ValueError(msg)
        if not isinstance(self.training_provenance, OperatorGrowthTrainingProvenance):
            msg = "training_provenance must be OperatorGrowthTrainingProvenance."
            raise TypeError(msg)
        provenance = self.training_provenance
        if (
            provenance.target_instance_id != target_id
            or provenance.target_family_id != family
            or provenance.qubit_count != qubits
            or provenance.target_vector_checksum != self.target_vector_checksum
        ):
            msg = "Evaluator target identity differs from its training provenance."
            raise ValueError(msg)
        object.__setattr__(self, "target_instance_id", target_id)
        object.__setattr__(self, "target_family_id", family)
        object.__setattr__(self, "target_stratum_id", stratum)
        object.__setattr__(self, "qubit_count", qubits)
        object.__setattr__(self, "norm", norm)

    @classmethod
    def from_target(
        cls,
        target: MaterializedTarget | LegacyMaterializedTarget,
        provenance: OperatorGrowthTrainingProvenance,
    ) -> StandardFixedRateOperatorGrowthEvaluatorBinding:
        """Construct the persisted binding from one authorized evaluator target."""
        if not isinstance(target, (MaterializedTarget, LegacyMaterializedTarget)):
            msg = "target must be an authorized materialized target."
            raise TypeError(msg)
        return cls(
            target_instance_id=target.target_instance_id,
            target_instance_spec_checksum=target.target_instance_spec_checksum,
            population_config_checksum=target.population_config_checksum,
            target_manifest_checksum=target.target_manifest_checksum,
            parameter_checksum=target.parameter_checksum,
            target_family_id=target.family_id,
            target_stratum_id=target.stratum_id,
            qubit_count=target.qubit_count,
            norm=target.norm,
            target_vector_checksum=target.vector_checksum,
            training_provenance=provenance,
        )

    def target_identity_dict(self) -> dict[str, object]:
        """Return the exact MaterializedTarget-compatible identity document."""
        return {
            "target_instance_id": self.target_instance_id,
            "target_instance_spec_checksum": self.target_instance_spec_checksum,
            "population_config_checksum": self.population_config_checksum,
            "target_manifest_checksum": self.target_manifest_checksum,
            "parameter_checksum": self.parameter_checksum,
            "family_id": self.target_family_id,
            "stratum_id": self.target_stratum_id,
            "qubit_count": self.qubit_count,
            "norm": self.norm,
            "vector_checksum": self.target_vector_checksum,
        }

    def _content_dict(self) -> dict[str, object]:
        """Return every persisted evaluator binding field."""
        return {
            "schema_version": self.schema_version,
            "target_instance_id": self.target_instance_id,
            "target_instance_spec_checksum": self.target_instance_spec_checksum,
            "population_config_checksum": self.population_config_checksum,
            "target_manifest_checksum": self.target_manifest_checksum,
            "parameter_checksum": self.parameter_checksum,
            "target_family_id": self.target_family_id,
            "target_stratum_id": self.target_stratum_id,
            "qubit_count": self.qubit_count,
            "norm": self.norm,
            "target_vector_checksum": self.target_vector_checksum,
            "training_provenance": self.training_provenance.to_dict(),
        }

    @property
    def content_checksum(self) -> str:
        """Checksum sealing exact target and runtime provenance."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return the strict checksum-sealed evaluator binding."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, value: object) -> StandardFixedRateOperatorGrowthEvaluatorBinding:
        """Decode and verify one evaluator binding."""
        mapping = require_mapping(value, "standard fixed-rate operator-growth evaluator binding")
        require_exact_keys(mapping, _EVALUATOR_BINDING_KEYS, "operator-growth evaluator binding")
        if mapping["schema_version"] != STANDARD_FIXED_RATE_OPERATOR_GROWTH_BINDING_SCHEMA_VERSION:
            msg = "operator-growth evaluator binding uses an unsupported schema version."
            raise ValueError(msg)
        binding = cls(
            target_instance_id=cast("str", mapping["target_instance_id"]),
            target_instance_spec_checksum=cast("str", mapping["target_instance_spec_checksum"]),
            population_config_checksum=cast("str", mapping["population_config_checksum"]),
            target_manifest_checksum=cast("str", mapping["target_manifest_checksum"]),
            parameter_checksum=cast("str", mapping["parameter_checksum"]),
            target_family_id=cast("str", mapping["target_family_id"]),
            target_stratum_id=cast("str", mapping["target_stratum_id"]),
            qubit_count=cast("int", mapping["qubit_count"]),
            norm=cast("float", mapping["norm"]),
            target_vector_checksum=cast("str", mapping["target_vector_checksum"]),
            training_provenance=OperatorGrowthTrainingProvenance.from_dict(mapping["training_provenance"]),
        )
        if require_checksum(mapping["content_checksum"], "evaluator binding.content_checksum") != (
            binding.content_checksum
        ):
            msg = "operator-growth evaluator-binding checksum mismatch."
            raise ValueError(msg)
        return binding


@dataclass(frozen=True, slots=True)
class NoisyOperatorGrowthObjectiveRequest:
    """One immutable fixed-CRN noisy objective callback request."""

    training_provenance_checksum: str
    evaluation_index: int
    selected_operator_ids: tuple[str, ...]
    parameters: tuple[float, ...]
    circuit_resources_checksum: str
    logical_gate_count: int
    trajectory_count: int
    trajectory_ensemble_checksum: str
    schema_version: str = field(default=OPERATOR_GROWTH_OBJECTIVE_REQUEST_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate coordinates, parameters, resources, and trajectory identity."""
        object.__setattr__(
            self,
            "training_provenance_checksum",
            require_checksum(self.training_provenance_checksum, "training_provenance_checksum"),
        )
        object.__setattr__(self, "evaluation_index", require_int(self.evaluation_index, "evaluation_index"))
        selected = _string_tuple(self.selected_operator_ids, "selected_operator_ids")
        parameters = _float_tuple(self.parameters, "parameters", length=len(selected))
        object.__setattr__(self, "selected_operator_ids", selected)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(
            self,
            "circuit_resources_checksum",
            require_checksum(self.circuit_resources_checksum, "circuit_resources_checksum"),
        )
        object.__setattr__(self, "logical_gate_count", require_int(self.logical_gate_count, "logical_gate_count"))
        object.__setattr__(
            self,
            "trajectory_count",
            require_int(self.trajectory_count, "trajectory_count", minimum=1),
        )
        object.__setattr__(
            self,
            "trajectory_ensemble_checksum",
            require_checksum(self.trajectory_ensemble_checksum, "trajectory_ensemble_checksum"),
        )

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered callback coordinate."""
        return {
            "schema_version": self.schema_version,
            "training_provenance_checksum": self.training_provenance_checksum,
            "evaluation_index": self.evaluation_index,
            "selected_operator_ids": list(self.selected_operator_ids),
            "parameters": list(self.parameters),
            "circuit_resources_checksum": self.circuit_resources_checksum,
            "logical_gate_count": self.logical_gate_count,
            "trajectory_count": self.trajectory_count,
            "trajectory_ensemble_checksum": self.trajectory_ensemble_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum sealing callback parameters and trajectory identity."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON-native callback request."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, value: object) -> NoisyOperatorGrowthObjectiveRequest:
        """Decode and verify one callback request."""
        mapping = require_mapping(value, "noisy operator-growth objective request")
        require_exact_keys(mapping, _OBJECTIVE_REQUEST_KEYS, "noisy operator-growth objective request")
        if mapping["schema_version"] != OPERATOR_GROWTH_OBJECTIVE_REQUEST_SCHEMA_VERSION:
            msg = "noisy operator-growth objective request uses an unsupported schema version."
            raise ValueError(msg)
        request = cls(
            training_provenance_checksum=cast("str", mapping["training_provenance_checksum"]),
            evaluation_index=cast("int", mapping["evaluation_index"]),
            selected_operator_ids=cast("tuple[str, ...]", mapping["selected_operator_ids"]),
            parameters=cast("tuple[float, ...]", mapping["parameters"]),
            circuit_resources_checksum=cast("str", mapping["circuit_resources_checksum"]),
            logical_gate_count=cast("int", mapping["logical_gate_count"]),
            trajectory_count=cast("int", mapping["trajectory_count"]),
            trajectory_ensemble_checksum=cast("str", mapping["trajectory_ensemble_checksum"]),
        )
        if require_checksum(mapping["content_checksum"], "objective request.content_checksum") != (
            request.content_checksum
        ):
            msg = "noisy operator-growth objective-request checksum mismatch."
            raise ValueError(msg)
        return request


class NoisyOperatorGrowthObjective(Protocol):
    """Typed callback for an externally implemented noisy trajectory objective."""

    def __call__(
        self,
        request: NoisyOperatorGrowthObjectiveRequest,
        circuit: ParameterizedCircuit,
    ) -> float:
        """Return the noisy objective for the exact request and circuit."""


class StandardFixedRateNoisyOperatorGrowthEvaluator:
    """Concrete target-bound standard-noise objective for operator growth.

    The evaluator owns the authorized target, scaled standard provider, TJM
    options, computational-zero input, common trajectory seed, and every
    provenance checksum. Callers therefore cannot substitute an arbitrary
    callback or self-assert provider, objective, or trajectory identities.
    One evaluator is a one-shot sequential objective stream.
    """

    __slots__ = (
        "_locked",
        "_next_evaluation_index",
        "noise_condition_checksum",
        "noise_definition_version",
        "noise_id",
        "noise_strength_scale",
        "objective_spec",
        "optimization_block_id",
        "optimization_seed",
        "provider",
        "resource_stratum_id",
        "target",
        "tjm_dt",
        "training_provenance",
        "trajectory_count",
        "trajectory_seed",
    )

    def __setattr__(self, name: str, value: object) -> None:
        """Keep scientific bindings immutable after validated construction."""
        if getattr(self, "_locked", False) and name != "_next_evaluation_index":
            msg = "A target-bound operator-growth evaluator is immutable."
            raise AttributeError(msg)
        object.__setattr__(self, name, value)

    def __init__(
        self,
        target: MaterializedTarget | LegacyMaterializedTarget,
        *,
        optimization_block_id: str,
        optimization_seed: int,
        resource_stratum_id: str,
        noise_id: str,
        noise_definition_version: str,
        noise_strength_scale: float,
        tjm_dt: float,
        trajectory_count: int,
        trajectory_seed: int,
    ) -> None:
        """Validate and bind one complete fixed-rate noisy objective."""
        self._locked = False
        if not isinstance(target, (MaterializedTarget, LegacyMaterializedTarget)):
            msg = "target must be an authorized MaterializedTarget or LegacyMaterializedTarget."
            raise TypeError(msg)
        block_id = require_slug(optimization_block_id, "optimization_block_id")
        resolved_optimization_seed = require_int(optimization_seed, "optimization_seed")
        if resolved_optimization_seed >= 2**64:
            msg = "optimization_seed must fit an unsigned 64-bit integer."
            raise ValueError(msg)
        stratum_id = require_slug(resource_stratum_id, "resource_stratum_id")
        resolved_noise_id = require_slug(noise_id, "noise_id")
        if resolved_noise_id not in STANDARD_NOISE_IDS:
            msg = f"noise_id must be one of the standard fixed-rate profiles {STANDARD_NOISE_IDS!r}."
            raise ValueError(msg)
        if noise_definition_version != FIXED_RATE_NOISE_DEFINITION_VERSION:
            msg = f"noise_definition_version must be {FIXED_RATE_NOISE_DEFINITION_VERSION!r}."
            raise ValueError(msg)
        strength = require_float(noise_strength_scale, "noise_strength_scale", minimum=0.0)
        if strength <= 0.0:
            msg = "noise_strength_scale must be strictly positive."
            raise ValueError(msg)
        dt = require_float(tjm_dt, "tjm_dt", minimum=0.0)
        if dt <= 0.0:
            msg = "tjm_dt must be strictly positive."
            raise ValueError(msg)
        count = require_int(trajectory_count, "trajectory_count", minimum=1)
        seed = require_int(trajectory_seed, "trajectory_seed")
        if seed >= 2**64:
            msg = "trajectory_seed must fit an unsigned 64-bit integer."
            raise ValueError(msg)

        self.target = target
        self.optimization_block_id = block_id
        self.optimization_seed = resolved_optimization_seed
        self.resource_stratum_id = stratum_id
        self.noise_id = resolved_noise_id
        self.noise_definition_version = noise_definition_version
        self.noise_strength_scale = strength
        self.tjm_dt = dt
        self.trajectory_count = count
        self.trajectory_seed = seed
        self.provider: ScaledStandardNoiseProvider = create_scaled_standard_noise_provider(
            resolved_noise_id,
            strength,
        )
        target_vector = target.state_vector_copy()
        zero_state = computational_zero_state(target.qubit_count)
        self.objective_spec = ProjectorObjectiveSpec.from_states(target_vector, zero_state)
        self.noise_condition_checksum = _standard_noise_condition_checksum(
            self.noise_id,
            self.noise_definition_version,
            self.noise_strength_scale,
            self.tjm_dt,
            self.objective_spec.initial_state_checksum,
        )
        trajectory_ensemble_checksum = _standard_trajectory_ensemble_checksum(
            target_instance_id=target.target_instance_id,
            target_vector_checksum=target.vector_checksum,
            provider_checksum=self.provider.content_checksum,
            objective_checksum=self.objective_spec.content_checksum,
            noise_condition_checksum=self.noise_condition_checksum,
            trajectory_count=self.trajectory_count,
            trajectory_seed=self.trajectory_seed,
        )
        self.training_provenance = OperatorGrowthTrainingProvenance(
            target_instance_id=target.target_instance_id,
            target_family_id=target.family_id,
            qubit_count=target.qubit_count,
            target_vector_checksum=target.vector_checksum,
            initial_state_checksum=self.objective_spec.initial_state_checksum,
            optimization_block_id=self.optimization_block_id,
            optimization_seed=self.optimization_seed,
            resource_stratum_id=self.resource_stratum_id,
            provider_id=f"scaled_standard_{self.noise_id}",
            provider_checksum=self.provider.content_checksum,
            objective_id=self.objective_spec.objective_id,
            objective_checksum=self.objective_spec.content_checksum,
            noise_id=self.noise_id,
            noise_definition_version=self.noise_definition_version,
            noise_strength_scale=self.noise_strength_scale,
            tjm_dt=self.tjm_dt,
            noise_condition_checksum=self.noise_condition_checksum,
            trajectory_count=self.trajectory_count,
            trajectory_seed=self.trajectory_seed,
            trajectory_ensemble_checksum=trajectory_ensemble_checksum,
        )
        self._next_evaluation_index = 0
        self._locked = True

    @property
    def binding(self) -> StandardFixedRateOperatorGrowthEvaluatorBinding:
        """Return the exact persisted target and runtime binding document."""
        return StandardFixedRateOperatorGrowthEvaluatorBinding.from_target(
            self.target,
            self.training_provenance,
        )

    @property
    def content_checksum(self) -> str:
        """Seal the authorized target and complete runtime configuration."""
        return self.binding.content_checksum

    def _validate_request_and_circuit(
        self,
        request: NoisyOperatorGrowthObjectiveRequest,
        circuit: ParameterizedCircuit,
    ) -> None:
        """Reject any callback coordinate outside this exact objective stream."""
        if not isinstance(request, NoisyOperatorGrowthObjectiveRequest):
            msg = "request must be a NoisyOperatorGrowthObjectiveRequest."
            raise TypeError(msg)
        if not isinstance(circuit, ParameterizedCircuit):
            msg = "circuit must be a ParameterizedCircuit."
            raise TypeError(msg)
        if (
            request.training_provenance_checksum != self.training_provenance.content_checksum
            or request.trajectory_count != self.trajectory_count
            or request.trajectory_ensemble_checksum != self.training_provenance.trajectory_ensemble_checksum
            or request.evaluation_index != self._next_evaluation_index
            or circuit.num_qubits != self.target.qubit_count
            or circuit.num_params != len(request.parameters)
            or request.logical_gate_count != len(circuit.gates)
        ):
            msg = "Objective request or circuit does not match the target-bound evaluator."
            raise ValueError(msg)
        pool_by_id = {
            operator.operator_id: operator
            for operator in build_projector_operator_pool(self.target.qubit_count).operators
        }
        if any(operator_id not in pool_by_id for operator_id in request.selected_operator_ids):
            msg = "Objective request contains an operator outside the frozen projector pool."
            raise ValueError(msg)
        selected = tuple(pool_by_id[operator_id] for operator_id in request.selected_operator_ids)
        expected_circuit = materialize_operator_growth_circuit(self.target.qubit_count, selected)
        resources = measure_circuit_resources(circuit)
        expected_resources = measure_circuit_resources(expected_circuit)
        if (
            resources.content_checksum != request.circuit_resources_checksum
            or resources.content_checksum != expected_resources.content_checksum
        ):
            msg = "Objective circuit or resource checksum differs from the selected operator sequence."
            raise ValueError(msg)

    def __call__(
        self,
        request: NoisyOperatorGrowthObjectiveRequest,
        circuit: ParameterizedCircuit,
    ) -> float:
        """Evaluate one actual standard-noise fixed-seed trajectory mean."""
        self._validate_request_and_circuit(request, circuit)
        parameters = np.asarray(request.parameters, dtype=np.float64)
        if not circuit.gates:
            target = self.target.state_vector_copy()
            fidelity = float(abs(target[0]) ** 2)
            self._next_evaluation_index += 1
            return float(np.clip(1.0 - fidelity, 0.0, 1.0))
        noisy_gate_indices = tuple(
            index for index, gate in enumerate(circuit.gates) if gate.is_trainable and gate.noise_enabled
        )
        if not noisy_gate_indices:
            msg = "A nonempty noisy operator-growth circuit requires trainable noise-enabled gates."
            raise ValueError(msg)
        options = KrotovTJMOptions(
            num_trajectories=self.trajectory_count,
            random_seed=self.trajectory_seed,
            dt=self.tjm_dt,
            apply_noise_to="all",
            noisy_gate_indices=noisy_gate_indices,
            trajectory_update="independent",
            differentiate_jump_normalization=False,
            use_crn=False,
        )
        loss, mean_fidelity, trajectory_fidelities = noisy_state_preparation_metrics(
            circuit,
            parameters,
            self.target.state_vector_copy(),
            None,
            options,
            initial_state=MPS(circuit.num_qubits),
            truncation=KrotovTruncation(),
            iteration=0,
            noise_provider=self.provider,
        )
        if (
            len(trajectory_fidelities) != self.trajectory_count
            or not math.isfinite(loss)
            or not math.isfinite(mean_fidelity)
            or not math.isclose(loss, 1.0 - mean_fidelity, rel_tol=0.0, abs_tol=1e-12)
        ):
            msg = "Noisy state-preparation metrics returned inconsistent trajectory evidence."
            raise ValueError(msg)
        self._next_evaluation_index += 1
        return float(loss)


@dataclass(frozen=True, slots=True)
class OperatorGrowthApplicability:
    """Typed applicability and family-wide promotion status for one method."""

    method_id: str
    family_id: str
    status: Literal["applicable", "not_applicable"]
    reason: str
    promotion_eligible: bool
    structural_not_applicable_is_failure: bool = False
    schema_version: str = field(default=OPERATOR_GROWTH_APPLICABILITY_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate status against the frozen method scope."""
        method_id = require_slug(self.method_id, "method_id")
        family_id = require_slug(self.family_id, "family_id")
        if method_id not in _METHOD_IDS:
            msg = f"Unsupported operator-growth method {method_id!r}."
            raise ValueError(msg)
        expected_applicable = method_id == ADAPT_STYLE_METHOD_ID or family_id == TFIM_FAMILY_ID
        expected_status = "applicable" if expected_applicable else "not_applicable"
        expected_reason = (
            "family_wide_projector_objective"
            if method_id == ADAPT_STYLE_METHOD_ID
            else ("tfim_hamiltonian_available" if expected_applicable else "tfim_hamiltonian_unavailable_for_family")
        )
        if self.status != expected_status or self.reason != expected_reason:
            msg = "Applicability status or reason differs from the frozen method scope."
            raise ValueError(msg)
        promotion_eligible = require_bool(self.promotion_eligible, "promotion_eligible")
        if promotion_eligible and (method_id != ADAPT_STYLE_METHOD_ID or self.status != "applicable"):
            msg = "Only applicable, noisily trained family-wide projector growth can be promotion eligible."
            raise ValueError(msg)
        if require_bool(
            self.structural_not_applicable_is_failure,
            "structural_not_applicable_is_failure",
        ):
            msg = "Structural not-applicability must not be counted as optimizer failure."
            raise ValueError(msg)
        object.__setattr__(self, "method_id", method_id)
        object.__setattr__(self, "family_id", family_id)
        object.__setattr__(self, "promotion_eligible", promotion_eligible)

    @property
    def is_optimizer_failure(self) -> bool:
        """Whether this applicability record denotes optimizer failure."""
        return False

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered applicability field."""
        return {
            "schema_version": self.schema_version,
            "method_id": self.method_id,
            "family_id": self.family_id,
            "status": self.status,
            "reason": self.reason,
            "promotion_eligible": self.promotion_eligible,
            "structural_not_applicable_is_failure": self.structural_not_applicable_is_failure,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum sealing method scope and promotion status."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return the sealed applicability document."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, value: object) -> OperatorGrowthApplicability:
        """Decode and verify a strict applicability document."""
        mapping = require_mapping(value, "operator-growth applicability")
        require_exact_keys(mapping, _APPLICABILITY_KEYS, "operator-growth applicability")
        if mapping["schema_version"] != OPERATOR_GROWTH_APPLICABILITY_SCHEMA_VERSION:
            msg = "operator-growth applicability uses an unsupported schema version."
            raise ValueError(msg)
        record = cls(
            method_id=cast("str", mapping["method_id"]),
            family_id=cast("str", mapping["family_id"]),
            status=cast('Literal["applicable", "not_applicable"]', mapping["status"]),
            reason=cast("str", mapping["reason"]),
            promotion_eligible=cast("bool", mapping["promotion_eligible"]),
            structural_not_applicable_is_failure=cast("bool", mapping["structural_not_applicable_is_failure"]),
        )
        if require_checksum(mapping["content_checksum"], "applicability.content_checksum") != record.content_checksum:
            msg = "operator-growth applicability content checksum mismatch."
            raise ValueError(msg)
        return record


def operator_growth_applicability(
    method_id: str,
    family_id: str,
    *,
    noisy_training: bool = False,
) -> OperatorGrowthApplicability:
    """Return the frozen applicability decision before objective construction.

    Args:
        method_id: One of the two WP20 operator-growth method identifiers.
        family_id: Target family under consideration.
        noisy_training: Whether checksum-bound noisy training evidence will be
            attached. Analytic reference executions are never promotion
            eligible even for the family-wide method.

    Returns:
        A typed applicable or structural-not-applicable record.
    """
    method = require_slug(method_id, "method_id")
    family = require_slug(family_id, "family_id")
    noisy = require_bool(noisy_training, "noisy_training")
    if method not in _METHOD_IDS:
        msg = f"Unsupported operator-growth method {method!r}."
        raise ValueError(msg)
    applicable = method == ADAPT_STYLE_METHOD_ID or family == TFIM_FAMILY_ID
    return OperatorGrowthApplicability(
        method_id=method,
        family_id=family,
        status="applicable" if applicable else "not_applicable",
        reason=(
            "family_wide_projector_objective"
            if method == ADAPT_STYLE_METHOD_ID
            else "tfim_hamiltonian_available"
            if applicable
            else "tfim_hamiltonian_unavailable_for_family"
        ),
        promotion_eligible=method == ADAPT_STYLE_METHOD_ID and noisy,
    )


@dataclass(frozen=True, slots=True)
class OperatorGrowthWork:
    """Complete deterministic circuit-work ledger for operator growth."""

    forward_circuit_evaluations: int = 0
    backward_circuit_evaluations: int = 0
    objective_calls: int = 0
    gradient_calls: int = 0
    parameter_shift_evaluations: int = 0
    trajectory_gate_applications: int = 0
    total_sampled_trajectories: int = 0
    cross_trajectory_pairings: int = 0
    reoptimization_iterations: int = 0

    def __post_init__(self) -> None:
        """Require nonnegative built-in counters and accounting identities."""
        for name in _WORK_KEYS:
            object.__setattr__(self, name, _strict_int(getattr(self, name), name))
        if self.parameter_shift_evaluations != 2 * self.gradient_calls:
            msg = "Every scalar parameter-shift gradient must use exactly two evaluations."
            raise ValueError(msg)
        expected_minimum = self.objective_calls if self.total_sampled_trajectories == 0 else 0
        if self.forward_circuit_evaluations < expected_minimum:
            msg = "Every analytic objective call must account for one forward circuit evaluation."
            raise ValueError(msg)
        if self.total_sampled_trajectories and self.forward_circuit_evaluations != self.total_sampled_trajectories:
            msg = "Noisy objective work must count one forward evaluation per propagated trajectory."
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        """Return the complete JSON-native work ledger."""
        return {name: getattr(self, name) for name in sorted(_WORK_KEYS)}

    def to_wp20_work_ledger(self) -> WP20WorkLedger:
        """Project exact growth work onto the shared additive WP20 ledger.

        Analytic references contribute zero trajectory work, while concrete
        noisy operator growth contributes its mechanically retained sampled
        trajectories and logical gate applications.

        Returns:
            Shared detailed work with no hidden backward or pairing work.
        """
        return WP20WorkLedger(
            forward_circuit_evaluations=self.forward_circuit_evaluations,
            backward_circuit_evaluations=self.backward_circuit_evaluations,
            trajectory_gate_applications=self.trajectory_gate_applications,
            training_trajectories=self.total_sampled_trajectories,
            objective_calls=self.objective_calls,
            gradient_calls=self.gradient_calls,
            cross_trajectory_pairings=self.cross_trajectory_pairings,
        )

    @classmethod
    def from_dict(cls, value: object) -> OperatorGrowthWork:
        """Decode a strict complete work ledger."""
        mapping = require_mapping(value, "operator-growth work")
        require_exact_keys(mapping, _WORK_KEYS, "operator-growth work")
        return cls(**{name: cast("int", mapping[name]) for name in _WORK_KEYS})


@dataclass(frozen=True, slots=True)
class CandidateGradient:
    """One pool candidate's signed gradient and native feasibility."""

    operator_id: str
    pool_index: int
    gradient: float | None
    absolute_gradient: float | None
    native_two_qubit_increment: int
    native_cap_feasible: bool

    def __post_init__(self) -> None:
        """Validate one deterministic candidate record."""
        object.__setattr__(self, "operator_id", require_slug(self.operator_id, "operator_id"))
        object.__setattr__(self, "pool_index", _strict_int(self.pool_index, "pool_index"))
        gradient = _strict_optional_float(self.gradient, "gradient")
        magnitude = _strict_optional_float(self.absolute_gradient, "absolute_gradient")
        feasible = require_bool(self.native_cap_feasible, "native_cap_feasible")
        if feasible != (gradient is not None) or (gradient is None) != (magnitude is None):
            msg = "Only native-cap-feasible candidates may carry gradients."
            raise ValueError(msg)
        if gradient is not None and not math.isclose(
            cast("float", magnitude), abs(gradient), rel_tol=0.0, abs_tol=1e-15
        ):
            msg = "absolute_gradient must equal abs(gradient)."
            raise ValueError(msg)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "absolute_gradient", magnitude)
        object.__setattr__(
            self,
            "native_two_qubit_increment",
            _strict_int(self.native_two_qubit_increment, "native_two_qubit_increment"),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the strict candidate record."""
        return {
            "operator_id": self.operator_id,
            "pool_index": self.pool_index,
            "gradient": self.gradient,
            "absolute_gradient": self.absolute_gradient,
            "native_two_qubit_increment": self.native_two_qubit_increment,
            "native_cap_feasible": self.native_cap_feasible,
        }

    @classmethod
    def from_dict(cls, value: object) -> CandidateGradient:
        """Decode one candidate-gradient record."""
        mapping = require_mapping(value, "candidate gradient")
        require_exact_keys(mapping, _CANDIDATE_KEYS, "candidate gradient")
        return cls(
            operator_id=cast("str", mapping["operator_id"]),
            pool_index=cast("int", mapping["pool_index"]),
            gradient=cast("float | None", mapping["gradient"]),
            absolute_gradient=cast("float | None", mapping["absolute_gradient"]),
            native_two_qubit_increment=cast("int", mapping["native_two_qubit_increment"]),
            native_cap_feasible=cast("bool", mapping["native_cap_feasible"]),
        )


@dataclass(frozen=True, slots=True)
class OperatorGrowthStep:
    """One complete selection or stopping event in an operator-growth trace."""

    iteration: int
    event: Literal["selected", "gradient_stop", "native_cap_stop"]
    candidate_gradients: tuple[CandidateGradient, ...]
    selected_operator_id: str | None
    selected_gradient: float | None
    objective_before_reoptimization: float
    objective_after_reoptimization: float
    parameters: tuple[float, ...]
    native_two_qubit_counts_by_edge: tuple[int, ...]
    work_after_step: OperatorGrowthWork

    def __post_init__(self) -> None:
        """Validate trace-event consistency and freeze nested sequences."""
        object.__setattr__(self, "iteration", _strict_int(self.iteration, "iteration"))
        if self.event not in {"selected", "gradient_stop", "native_cap_stop"}:
            msg = "Unsupported operator-growth trace event."
            raise ValueError(msg)
        candidates = tuple(self.candidate_gradients)
        if any(not isinstance(item, CandidateGradient) for item in candidates):
            msg = "candidate_gradients must contain CandidateGradient records."
            raise TypeError(msg)
        object.__setattr__(self, "candidate_gradients", candidates)
        selected_id = self.selected_operator_id
        selected_gradient = _strict_optional_float(self.selected_gradient, "selected_gradient")
        if self.event == "selected":
            if selected_id is None or selected_gradient is None:
                msg = "A selected event requires an operator and signed gradient."
                raise ValueError(msg)
            selected_id = require_slug(selected_id, "selected_operator_id")
            if not any(item.operator_id == selected_id and item.gradient == selected_gradient for item in candidates):
                msg = "Selected operator and gradient must occur in the candidate ledger."
                raise ValueError(msg)
        elif selected_id is not None or selected_gradient is not None:
            msg = "Stopping events cannot carry a selected operator."
            raise ValueError(msg)
        object.__setattr__(self, "selected_operator_id", selected_id)
        object.__setattr__(self, "selected_gradient", selected_gradient)
        object.__setattr__(
            self,
            "objective_before_reoptimization",
            require_float(self.objective_before_reoptimization, "objective_before_reoptimization"),
        )
        object.__setattr__(
            self,
            "objective_after_reoptimization",
            require_float(self.objective_after_reoptimization, "objective_after_reoptimization"),
        )
        object.__setattr__(self, "parameters", _float_tuple(self.parameters, "parameters"))
        object.__setattr__(
            self,
            "native_two_qubit_counts_by_edge",
            _int_tuple(self.native_two_qubit_counts_by_edge, "native_two_qubit_counts_by_edge"),
        )
        if not isinstance(self.work_after_step, OperatorGrowthWork):
            msg = "work_after_step must be an OperatorGrowthWork record."
            raise TypeError(msg)

    def to_dict(self) -> dict[str, object]:
        """Return the complete JSON-native trace event."""
        return {
            "iteration": self.iteration,
            "event": self.event,
            "candidate_gradients": [item.to_dict() for item in self.candidate_gradients],
            "selected_operator_id": self.selected_operator_id,
            "selected_gradient": self.selected_gradient,
            "objective_before_reoptimization": self.objective_before_reoptimization,
            "objective_after_reoptimization": self.objective_after_reoptimization,
            "parameters": list(self.parameters),
            "native_two_qubit_counts_by_edge": list(self.native_two_qubit_counts_by_edge),
            "work_after_step": self.work_after_step.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> OperatorGrowthStep:
        """Decode one strict operator-growth event."""
        mapping = require_mapping(value, "operator-growth step")
        require_exact_keys(mapping, _STEP_KEYS, "operator-growth step")
        raw_candidates = mapping["candidate_gradients"]
        if not isinstance(raw_candidates, Sequence):
            msg = "candidate_gradients must be a sequence."
            raise TypeError(msg)
        return cls(
            iteration=cast("int", mapping["iteration"]),
            event=cast('Literal["selected", "gradient_stop", "native_cap_stop"]', mapping["event"]),
            candidate_gradients=tuple(CandidateGradient.from_dict(item) for item in raw_candidates),
            selected_operator_id=cast("str | None", mapping["selected_operator_id"]),
            selected_gradient=cast("float | None", mapping["selected_gradient"]),
            objective_before_reoptimization=cast("float", mapping["objective_before_reoptimization"]),
            objective_after_reoptimization=cast("float", mapping["objective_after_reoptimization"]),
            parameters=cast("tuple[float, ...]", mapping["parameters"]),
            native_two_qubit_counts_by_edge=cast("tuple[int, ...]", mapping["native_two_qubit_counts_by_edge"]),
            work_after_step=OperatorGrowthWork.from_dict(mapping["work_after_step"]),
        )


@dataclass(frozen=True, slots=True)
class DeterministicReoptimizationResult:
    """Typed return contract for an externally supplied deterministic callback."""

    parameters: tuple[float, ...]
    iterations: int
    gradient_calls: int

    def __post_init__(self) -> None:
        """Validate callback output and its reported work."""
        object.__setattr__(self, "parameters", _float_tuple(self.parameters, "parameters"))
        object.__setattr__(self, "iterations", _strict_int(self.iterations, "iterations"))
        object.__setattr__(self, "gradient_calls", _strict_int(self.gradient_calls, "gradient_calls"))


class DeterministicReoptimizer(Protocol):
    """Protocol for full retained-parameter deterministic reoptimization."""

    def __call__(
        self,
        objective: Callable[[NDArray[np.float64]], float],
        initial_parameters: NDArray[np.float64],
    ) -> DeterministicReoptimizationResult:
        """Return fully reoptimized parameters and exact gradient work."""


@dataclass(slots=True)
class _WorkCounter:
    """Mutable internal counter converted to immutable evidence snapshots."""

    forward_circuit_evaluations: int = 0
    objective_calls: int = 0
    gradient_calls: int = 0
    trajectory_gate_applications: int = 0
    total_sampled_trajectories: int = 0
    reoptimization_iterations: int = 0

    def record_objective(self, *, trajectory_count: int, logical_gate_count: int) -> None:
        """Record one analytic evaluation or one trajectory-mean evaluation."""
        self.objective_calls += 1
        self.forward_circuit_evaluations += max(1, trajectory_count)
        self.total_sampled_trajectories += trajectory_count
        self.trajectory_gate_applications += trajectory_count * logical_gate_count

    def snapshot(self) -> OperatorGrowthWork:
        """Return the current immutable complete ledger."""
        return OperatorGrowthWork(
            forward_circuit_evaluations=self.forward_circuit_evaluations,
            objective_calls=self.objective_calls,
            gradient_calls=self.gradient_calls,
            parameter_shift_evaluations=2 * self.gradient_calls,
            trajectory_gate_applications=self.trajectory_gate_applications,
            total_sampled_trajectories=self.total_sampled_trajectories,
            reoptimization_iterations=self.reoptimization_iterations,
        )


OperatorGrowthObjectiveBinding = ProjectorObjectiveSpec | TFIMHamiltonianSpec | TargetBoundTFIMEnergyObjectiveSpec


def _objective_binding_from_dict(value: object) -> OperatorGrowthObjectiveBinding:
    """Decode one exact objective binding by its frozen schema identity."""
    mapping = require_mapping(value, "operator-growth objective binding")
    schema_version = mapping.get("schema_version")
    if schema_version == PROJECTOR_OBJECTIVE_SCHEMA_VERSION:
        return ProjectorObjectiveSpec.from_dict(mapping)
    if schema_version == TFIM_HAMILTONIAN_SCHEMA_VERSION:
        return TFIMHamiltonianSpec.from_dict(mapping)
    if schema_version == TARGET_BOUND_TFIM_OBJECTIVE_SCHEMA_VERSION:
        return TargetBoundTFIMEnergyObjectiveSpec.from_dict(mapping)
    msg = "operator-growth objective binding uses an unsupported schema version."
    raise ValueError(msg)


def _operators_from_ids(pool: OperatorPoolSpec, operator_ids: tuple[str, ...]) -> tuple[PoolOperator, ...]:
    """Resolve an ordered operator sequence against one exact frozen pool."""
    by_id = {operator.operator_id: operator for operator in pool.operators}
    try:
        return tuple(by_id[operator_id] for operator_id in operator_ids)
    except KeyError as error:
        msg = "Selected operator identity is absent from the persisted exact pool."
        raise ValueError(msg) from error


def _edge_counts_for_operators(
    num_qubits: int,
    operators: Sequence[PoolOperator],
) -> tuple[int, ...]:
    """Derive native two-qubit counts per edge from an operator sequence."""
    counts = [0] * max(0, num_qubits - 1)
    for operator in operators:
        if len(operator.sites) == 2:
            counts[operator.sites[0]] += operator.native_two_qubit_gates
    return tuple(counts)


def _work_is_monotone(previous: OperatorGrowthWork, current: OperatorGrowthWork) -> bool:
    """Return whether every cumulative work field is nondecreasing."""
    return all(getattr(current, name) >= getattr(previous, name) for name in _WORK_KEYS)


def _validate_trace_against_exact_documents(
    pool: OperatorPoolSpec,
    spec: OperatorGrowthSpec,
    selected_operator_ids: tuple[str, ...],
    parameters: tuple[float, ...],
    edge_counts: tuple[int, ...],
    trace: tuple[OperatorGrowthStep, ...],
    initial_objective: float,
    termination_reason: str,
) -> tuple[PoolOperator, ...]:
    """Mechanically replay pool order, feasibility, selection, and resources."""
    selected: list[PoolOperator] = []
    selected_ids: set[str] = set()
    running_counts = [0] * max(0, pool.num_qubits - 1)
    previous_work = OperatorGrowthWork()
    previous_objective = initial_objective

    for step_index, step in enumerate(trace):
        if step.iteration != len(selected):
            msg = "Trace iteration does not equal the retained operator count."
            raise ValueError(msg)
        expected_candidates = tuple(
            (pool_index, operator)
            for pool_index, operator in enumerate(pool.operators)
            if operator.operator_id not in selected_ids
        )
        if len(step.candidate_gradients) != len(expected_candidates):
            msg = "Trace candidate ledger does not contain every unused pool operator."
            raise ValueError(msg)

        feasible_candidates: list[CandidateGradient] = []
        for candidate, (pool_index, operator) in zip(step.candidate_gradients, expected_candidates, strict=True):
            expected_feasible = True
            if len(operator.sites) == 2 and spec.native_two_qubit_cap_per_edge is not None:
                edge = operator.sites[0]
                expected_feasible = (
                    running_counts[edge] + operator.native_two_qubit_gates <= spec.native_two_qubit_cap_per_edge
                )
            if (
                candidate.operator_id != operator.operator_id
                or candidate.pool_index != pool_index
                or candidate.native_two_qubit_increment != operator.native_two_qubit_gates
                or candidate.native_cap_feasible != expected_feasible
            ):
                msg = "Trace candidate identity, ordering, native cost, or cap feasibility changed."
                raise ValueError(msg)
            if expected_feasible:
                feasible_candidates.append(candidate)

        if not _work_is_monotone(previous_work, step.work_after_step):
            msg = "Trace work snapshots must be cumulative and nondecreasing."
            raise ValueError(msg)
        previous_work = step.work_after_step

        if step.event == "selected":
            if not feasible_candidates:
                msg = "A selected trace event requires at least one feasible pool candidate."
                raise ValueError(msg)
            chosen = max(feasible_candidates, key=lambda candidate: cast("float", candidate.absolute_gradient))
            if (
                step.selected_operator_id != chosen.operator_id
                or step.selected_gradient != chosen.gradient
                or cast("float", chosen.absolute_gradient) <= spec.gradient_tolerance
            ):
                msg = "Trace selection differs from the frozen largest-gradient and tie-break rules."
                raise ValueError(msg)
            operator = pool.operators[chosen.pool_index]
            selected.append(operator)
            selected_ids.add(operator.operator_id)
            if len(operator.sites) == 2:
                running_counts[operator.sites[0]] += operator.native_two_qubit_gates
            if (
                len(step.parameters) != len(selected)
                or step.native_two_qubit_counts_by_edge != tuple(running_counts)
                or not math.isclose(
                    step.objective_before_reoptimization,
                    previous_objective,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                msg = "Selected trace endpoint does not reproduce its retained circuit prefix."
                raise ValueError(msg)
            previous_objective = step.objective_after_reoptimization
        else:
            if step_index != len(trace) - 1:
                msg = "A stopping trace event must be terminal."
                raise ValueError(msg)
            if (
                step.parameters != (() if not selected else trace[step_index - 1].parameters)
                or step.native_two_qubit_counts_by_edge != tuple(running_counts)
                or not math.isclose(
                    step.objective_before_reoptimization, previous_objective, rel_tol=0.0, abs_tol=1e-12
                )
                or not math.isclose(step.objective_after_reoptimization, previous_objective, rel_tol=0.0, abs_tol=1e-12)
            ):
                msg = "Stopping trace endpoint does not equal the retained circuit prefix."
                raise ValueError(msg)
            if step.event == "gradient_stop":
                if (
                    not feasible_candidates
                    or max(cast("float", candidate.absolute_gradient) for candidate in feasible_candidates)
                    > spec.gradient_tolerance
                ):
                    msg = "Gradient stop does not satisfy the persisted tolerance."
                    raise ValueError(msg)
            elif feasible_candidates:
                msg = "Native-cap stop still contains a feasible candidate."
                raise ValueError(msg)

    resolved = tuple(selected)
    if tuple(operator.operator_id for operator in resolved) != selected_operator_ids:
        msg = "Result selections do not equal the mechanically replayed trace."
        raise ValueError(msg)
    if parameters != (() if not trace else trace[-1].parameters):
        msg = "Result parameters do not equal the mechanically replayed trace endpoint."
        raise ValueError(msg)
    if edge_counts != tuple(running_counts):
        msg = "Result native per-edge counts do not equal the mechanically replayed trace."
        raise ValueError(msg)
    if termination_reason == "max_operators" and len(resolved) != spec.max_operators:
        msg = "max_operators termination does not reach the persisted maximum."
        raise ValueError(msg)
    if termination_reason == "pool_exhausted" and (
        len(resolved) != len(pool.operators) or spec.max_operators <= len(pool.operators)
    ):
        msg = "pool_exhausted termination does not exhaust the pool before the persisted maximum."
        raise ValueError(msg)
    if termination_reason == "gradient_tolerance" and (not trace or trace[-1].event != "gradient_stop"):
        msg = "gradient_tolerance termination lacks a matching terminal trace event."
        raise ValueError(msg)
    if termination_reason == "native_cap" and (not trace or trace[-1].event != "native_cap_stop"):
        msg = "native_cap termination lacks a matching terminal trace event."
        raise ValueError(msg)
    return resolved


def _expected_noisy_request_operator_sequences(
    spec: OperatorGrowthSpec,
    trace: tuple[OperatorGrowthStep, ...],
) -> tuple[tuple[str, ...], ...]:
    """Derive every internal-Adam noisy objective circuit from the trace."""
    expected: list[tuple[str, ...]] = [()]
    selected_prefix: tuple[str, ...] = ()
    for step in trace:
        for candidate in step.candidate_gradients:
            if candidate.native_cap_feasible:
                candidate_sequence = (*selected_prefix, candidate.operator_id)
                expected.extend((candidate_sequence, candidate_sequence))
        if step.event != "selected":
            continue
        assert step.selected_operator_id is not None
        selected_prefix = (*selected_prefix, step.selected_operator_id)
        # Appended-zero value, Adam baseline, every parameter-shift pair plus
        # post-update value, and the final selected objective are all explicit.
        expected.extend((selected_prefix, selected_prefix))
        for _ in range(spec.reoptimization_steps):
            for _ in selected_prefix:
                expected.extend((selected_prefix, selected_prefix))
            expected.append(selected_prefix)
        expected.append(selected_prefix)
    return tuple(expected)


@dataclass(frozen=True, slots=True)
class OperatorGrowthResult:
    """Checksum-sealed result with circuit, training, work, and applicability."""

    method_id: str
    algorithm_label: str
    status: Literal["completed", "not_applicable"]
    applicability: OperatorGrowthApplicability
    pool_checksum: str | None
    growth_spec_checksum: str | None
    objective_binding_checksum: str | None
    pool: OperatorPoolSpec | None
    growth_spec: OperatorGrowthSpec | None
    objective_binding: OperatorGrowthObjectiveBinding | None
    evaluator_binding: StandardFixedRateOperatorGrowthEvaluatorBinding | None
    selected_operator_ids: tuple[str, ...]
    parameters: tuple[float, ...]
    initial_objective: float | None
    final_objective: float | None
    native_two_qubit_counts_by_edge: tuple[int, ...]
    termination_reason: str
    trace: tuple[OperatorGrowthStep, ...]
    work: OperatorGrowthWork
    execution_mode: Literal["analytic_reference", "noisy_training"]
    training_provenance: OperatorGrowthTrainingProvenance | None
    objective_requests: tuple[NoisyOperatorGrowthObjectiveRequest, ...]
    circuit_resources: CircuitResourceMetrics | None
    schema_version: str = field(default=OPERATOR_GROWTH_RESULT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate exact bindings and mechanically replay all structural evidence."""
        if not isinstance(self.applicability, OperatorGrowthApplicability):
            msg = "applicability must be an OperatorGrowthApplicability record."
            raise TypeError(msg)
        if not isinstance(self.work, OperatorGrowthWork):
            msg = "work must be an OperatorGrowthWork record."
            raise TypeError(msg)
        if self.training_provenance is not None and not isinstance(
            self.training_provenance,
            OperatorGrowthTrainingProvenance,
        ):
            msg = "training_provenance must be OperatorGrowthTrainingProvenance or None."
            raise TypeError(msg)
        objective_requests = tuple(self.objective_requests)
        if any(not isinstance(request, NoisyOperatorGrowthObjectiveRequest) for request in objective_requests):
            msg = "objective_requests must contain NoisyOperatorGrowthObjectiveRequest records."
            raise TypeError(msg)
        if self.circuit_resources is not None and not isinstance(self.circuit_resources, CircuitResourceMetrics):
            msg = "circuit_resources must be CircuitResourceMetrics or None."
            raise TypeError(msg)
        if self.pool is not None and not isinstance(self.pool, OperatorPoolSpec):
            msg = "pool must be OperatorPoolSpec or None."
            raise TypeError(msg)
        if self.growth_spec is not None and not isinstance(self.growth_spec, OperatorGrowthSpec):
            msg = "growth_spec must be OperatorGrowthSpec or None."
            raise TypeError(msg)
        if self.objective_binding is not None and not isinstance(
            self.objective_binding,
            (ProjectorObjectiveSpec, TFIMHamiltonianSpec, TargetBoundTFIMEnergyObjectiveSpec),
        ):
            msg = "objective_binding must be an exact supported objective document or None."
            raise TypeError(msg)
        if self.evaluator_binding is not None and not isinstance(
            self.evaluator_binding,
            StandardFixedRateOperatorGrowthEvaluatorBinding,
        ):
            msg = "evaluator_binding must be StandardFixedRateOperatorGrowthEvaluatorBinding or None."
            raise TypeError(msg)
        method_id = require_slug(self.method_id, "method_id")
        if method_id not in _METHOD_IDS or self.applicability.method_id != method_id:
            msg = "Result method and applicability must identify the same supported method."
            raise ValueError(msg)
        expected_label = (
            "projector_operator_growth" if method_id == ADAPT_STYLE_METHOD_ID else "genuine_energy_adapt_vqe"
        )
        if require_slug(self.algorithm_label, "algorithm_label") != expected_label:
            msg = f"algorithm_label must be {expected_label!r}."
            raise ValueError(msg)
        if self.termination_reason not in _TERMINATION_REASONS:
            msg = "Unsupported operator-growth termination reason."
            raise ValueError(msg)
        selected = _string_tuple(self.selected_operator_ids, "selected_operator_ids")
        parameters = _float_tuple(self.parameters, "parameters", length=len(selected))
        edge_counts = _int_tuple(self.native_two_qubit_counts_by_edge, "native_two_qubit_counts_by_edge")
        trace = tuple(self.trace)
        if any(not isinstance(step, OperatorGrowthStep) for step in trace):
            msg = "trace must contain OperatorGrowthStep records."
            raise TypeError(msg)
        checksums = (self.pool_checksum, self.growth_spec_checksum, self.objective_binding_checksum)
        binding_documents = (self.pool, self.growth_spec, self.objective_binding, self.evaluator_binding)
        if self.status == "not_applicable":
            if self.applicability.status != "not_applicable" or self.termination_reason != "not_applicable":
                msg = "A not-applicable result requires matching applicability and termination."
                raise ValueError(msg)
            if (
                any(value is not None for value in checksums)
                or any(value is not None for value in binding_documents)
                or selected
                or parameters
                or trace
                or edge_counts
                or objective_requests
                or self.training_provenance is not None
                or self.circuit_resources is not None
            ):
                msg = "Structural not-applicability must return before pool or objective work."
                raise ValueError(msg)
            if self.execution_mode != "analytic_reference" or self.applicability.promotion_eligible:
                msg = "Structural not-applicability is an ineligible analytic-reference result."
                raise ValueError(msg)
            if (
                self.initial_objective is not None
                or self.final_objective is not None
                or self.work != OperatorGrowthWork()
            ):
                msg = "Structural not-applicability cannot contain objective values or work."
                raise ValueError(msg)
        elif self.status == "completed":
            if self.applicability.status != "applicable" or any(value is None for value in checksums):
                msg = "Completed growth requires applicable status and all provenance checksums."
                raise ValueError(msg)
            for index, value in enumerate(checksums):
                require_checksum(value, f"result checksum {index}")
            pool = self.pool
            spec = self.growth_spec
            objective_binding = self.objective_binding
            if pool is None or spec is None or objective_binding is None:
                msg = "Completed growth requires exact pool, growth-spec, and objective-binding documents."
                raise ValueError(msg)
            if (
                pool.method_id != method_id
                or pool.content_checksum != self.pool_checksum
                or spec.method_id != method_id
                or spec.pool_checksum != pool.content_checksum
                or spec.content_checksum != self.growth_spec_checksum
            ):
                msg = "Persisted pool or growth-spec documents do not reproduce their result bindings."
                raise ValueError(msg)
            _strict_optional_float(self.initial_objective, "initial_objective")
            _strict_optional_float(self.final_objective, "final_objective")
            if self.initial_objective is None or self.final_objective is None:
                msg = "Completed growth requires initial and final objective values."
                raise ValueError(msg)
            if self.circuit_resources is None:
                msg = "Completed growth requires mechanically derived circuit resources."
                raise ValueError(msg)
            if self.circuit_resources.qubit_count != pool.num_qubits:
                msg = "Circuit-resource width differs from the persisted exact pool."
                raise ValueError(msg)

            if method_id == ADAPT_STYLE_METHOD_ID:
                if not isinstance(objective_binding, ProjectorObjectiveSpec):
                    msg = "Projector operator growth requires an exact projector objective binding."
                    raise ValueError(msg)
                if objective_binding.num_qubits != pool.num_qubits:
                    msg = "Projector objective width differs from the persisted pool."
                    raise ValueError(msg)
            else:
                if isinstance(objective_binding, TFIMHamiltonianSpec):
                    hamiltonian_binding = objective_binding
                elif isinstance(objective_binding, TargetBoundTFIMEnergyObjectiveSpec):
                    hamiltonian_binding = objective_binding.hamiltonian_binding
                    target_spec = objective_binding.target_instance_spec
                    if (
                        target_spec.family_id != self.applicability.family_id
                        or target_spec.qubit_count != pool.num_qubits
                    ):
                        msg = "Target-bound TFIM objective differs from result family or pool width."
                        raise ValueError(msg)
                else:
                    msg = "Energy ADAPT requires a TFIM Hamiltonian objective binding."
                    raise ValueError(msg)
                if hamiltonian_binding.num_qubits != pool.num_qubits:
                    msg = "TFIM Hamiltonian width differs from the persisted pool."
                    raise ValueError(msg)

            assert self.initial_objective is not None
            selected_operators = _validate_trace_against_exact_documents(
                pool,
                spec,
                selected,
                parameters,
                edge_counts,
                trace,
                self.initial_objective,
                self.termination_reason,
            )
            expected_counts = _edge_counts_for_operators(pool.num_qubits, selected_operators)
            expected_resources = measure_circuit_resources(
                materialize_operator_growth_circuit(pool.num_qubits, selected_operators)
            )
            if edge_counts != expected_counts or self.circuit_resources != expected_resources:
                msg = "Circuit resources or native per-edge counts do not reproduce the selected exact operators."
                raise ValueError(msg)
            if trace:
                last_step = trace[-1]
                if (
                    last_step.parameters != parameters
                    or last_step.native_two_qubit_counts_by_edge != edge_counts
                    or last_step.work_after_step != self.work
                    or last_step.objective_after_reoptimization != self.final_objective
                ):
                    msg = "Result endpoint must equal the final trace snapshot."
                    raise ValueError(msg)
                expected_terminal_event = {
                    "gradient_tolerance": "gradient_stop",
                    "native_cap": "native_cap_stop",
                }.get(self.termination_reason)
                if expected_terminal_event is not None and last_step.event != expected_terminal_event:
                    msg = "Stopping reason does not match the final trace event."
                    raise ValueError(msg)
                if last_step.event == "gradient_stop" and self.termination_reason != "gradient_tolerance":
                    msg = "Gradient-stop trace event requires gradient_tolerance termination."
                    raise ValueError(msg)
                if last_step.event == "native_cap_stop" and self.termination_reason != "native_cap":
                    msg = "Native-cap trace event requires native_cap termination."
                    raise ValueError(msg)
            elif selected:
                msg = "A nonempty selected circuit requires selected trace events."
                raise ValueError(msg)

            if self.execution_mode == "analytic_reference":
                if self.training_provenance is not None or objective_requests or self.evaluator_binding is not None:
                    msg = "Analytic-reference growth cannot claim noisy objective provenance."
                    raise ValueError(msg)
                if self.work.total_sampled_trajectories or self.work.trajectory_gate_applications:
                    msg = "Analytic-reference growth cannot claim trajectory work."
                    raise ValueError(msg)
                if self.applicability.promotion_eligible:
                    msg = "Analytic-reference growth is ineligible for sealed noisy-method promotion."
                    raise ValueError(msg)
                if self.objective_binding_checksum != objective_binding.content_checksum:
                    msg = "Analytic objective checksum does not reproduce the persisted objective binding."
                    raise ValueError(msg)
            elif self.execution_mode == "noisy_training":
                provenance = self.training_provenance
                evaluator_binding = self.evaluator_binding
                if provenance is None or evaluator_binding is None or not objective_requests:
                    msg = "Noisy growth requires exact evaluator provenance and every objective request."
                    raise ValueError(msg)
                if (
                    method_id != ADAPT_STYLE_METHOD_ID
                    or not isinstance(objective_binding, ProjectorObjectiveSpec)
                    or not self.applicability.promotion_eligible
                ):
                    msg = "Only noisy family-wide projector growth is promotion eligible."
                    raise ValueError(msg)
                assert self.circuit_resources is not None
                if (
                    evaluator_binding.training_provenance != provenance
                    or evaluator_binding.target_family_id != self.applicability.family_id
                    or evaluator_binding.qubit_count != pool.num_qubits
                    or evaluator_binding.target_vector_checksum != objective_binding.target_state_checksum
                    or provenance.objective_id != objective_binding.objective_id
                    or provenance.objective_checksum != objective_binding.content_checksum
                    or provenance.initial_state_checksum != objective_binding.initial_state_checksum
                    or provenance.target_family_id != self.applicability.family_id
                    or provenance.qubit_count != self.circuit_resources.qubit_count
                ):
                    msg = "Noisy target, objective, evaluator, and runtime binding documents disagree."
                    raise ValueError(msg)
                expected_binding_checksum = canonical_checksum({
                    "projector_objective_checksum": objective_binding.content_checksum,
                    "training_provenance_checksum": provenance.content_checksum,
                    "evaluator_checksum": evaluator_binding.content_checksum,
                })
                if self.objective_binding_checksum != expected_binding_checksum:
                    msg = "Noisy objective checksum does not reproduce its exact persisted bindings."
                    raise ValueError(msg)
                expected_indices = tuple(range(len(objective_requests)))
                if tuple(request.evaluation_index for request in objective_requests) != expected_indices:
                    msg = "Noisy objective-request indices must be contiguous and ordered."
                    raise ValueError(msg)
                if any(
                    request.training_provenance_checksum != provenance.content_checksum
                    or request.trajectory_count != provenance.trajectory_count
                    or request.trajectory_ensemble_checksum != provenance.trajectory_ensemble_checksum
                    for request in objective_requests
                ):
                    msg = "Noisy objective requests differ from the sealed training provenance."
                    raise ValueError(msg)
                if len(objective_requests) != self.work.objective_calls:
                    msg = "Every noisy objective call must retain one exact request."
                    raise ValueError(msg)
                expected_request_sequences = _expected_noisy_request_operator_sequences(spec, trace)
                if tuple(request.selected_operator_ids for request in objective_requests) != expected_request_sequences:
                    msg = "Noisy objective requests do not follow the mechanically derived trace schedule."
                    raise ValueError(msg)
                for request in objective_requests:
                    request_operators = _operators_from_ids(pool, request.selected_operator_ids)
                    request_circuit = materialize_operator_growth_circuit(pool.num_qubits, request_operators)
                    request_resources = measure_circuit_resources(request_circuit)
                    if (
                        request.circuit_resources_checksum != request_resources.content_checksum
                        or request.logical_gate_count != len(request_circuit.gates)
                    ):
                        msg = "Noisy objective request does not reproduce its exact materialized circuit."
                        raise ValueError(msg)
                expected_trajectories = len(objective_requests) * provenance.trajectory_count
                expected_gate_applications = sum(
                    request.logical_gate_count * provenance.trajectory_count for request in objective_requests
                )
                if (
                    self.work.total_sampled_trajectories != expected_trajectories
                    or self.work.trajectory_gate_applications != expected_gate_applications
                ):
                    msg = "Noisy objective requests do not reproduce the recorded trajectory work."
                    raise ValueError(msg)
            else:
                msg = "execution_mode must be analytic_reference or noisy_training."
                raise ValueError(msg)
        else:
            msg = "status must be either 'completed' or 'not_applicable'."
            raise ValueError(msg)
        object.__setattr__(self, "method_id", method_id)
        object.__setattr__(self, "selected_operator_ids", selected)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "native_two_qubit_counts_by_edge", edge_counts)
        object.__setattr__(self, "trace", trace)
        object.__setattr__(self, "objective_requests", objective_requests)

    @property
    def promotion_eligible(self) -> bool:
        """Whether this family-wide method may enter the sealed promotion rule."""
        return self.applicability.promotion_eligible

    @property
    def is_optimizer_failure(self) -> bool:
        """Whether this result is an optimizer failure."""
        return False

    @property
    def wp20_work(self) -> WP20WorkLedger:
        """Shared additive work ledger for resource and Pareto comparisons."""
        return self.work.to_wp20_work_ledger()

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered result field."""
        return {
            "schema_version": self.schema_version,
            "method_id": self.method_id,
            "algorithm_label": self.algorithm_label,
            "status": self.status,
            "applicability": self.applicability.to_dict(),
            "pool_checksum": self.pool_checksum,
            "growth_spec_checksum": self.growth_spec_checksum,
            "objective_binding_checksum": self.objective_binding_checksum,
            "pool": None if self.pool is None else self.pool.to_dict(),
            "growth_spec": None if self.growth_spec is None else self.growth_spec.to_dict(),
            "objective_binding": None if self.objective_binding is None else self.objective_binding.to_dict(),
            "evaluator_binding": None if self.evaluator_binding is None else self.evaluator_binding.to_dict(),
            "selected_operator_ids": list(self.selected_operator_ids),
            "parameters": list(self.parameters),
            "initial_objective": self.initial_objective,
            "final_objective": self.final_objective,
            "native_two_qubit_counts_by_edge": list(self.native_two_qubit_counts_by_edge),
            "termination_reason": self.termination_reason,
            "trace": [step.to_dict() for step in self.trace],
            "work": self.work.to_dict(),
            "execution_mode": self.execution_mode,
            "training_provenance": (None if self.training_provenance is None else self.training_provenance.to_dict()),
            "objective_requests": [request.to_dict() for request in self.objective_requests],
            "circuit_resources": None if self.circuit_resources is None else self.circuit_resources.to_dict(),
        }

    @property
    def content_checksum(self) -> str:
        """Checksum sealing result values, trace, work, and applicability."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return the complete sealed result document."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, value: object) -> OperatorGrowthResult:
        """Decode and verify a strict result document."""
        mapping = freeze_json_mapping(require_mapping(value, "operator-growth result"), "operator-growth result")
        require_exact_keys(mapping, _RESULT_KEYS, "operator-growth result")
        if mapping["schema_version"] != OPERATOR_GROWTH_RESULT_SCHEMA_VERSION:
            msg = "operator-growth result uses an unsupported schema version."
            raise ValueError(msg)
        raw_trace = mapping["trace"]
        if not isinstance(raw_trace, Sequence):
            msg = "operator-growth result trace must be a sequence."
            raise TypeError(msg)
        raw_requests = mapping["objective_requests"]
        if isinstance(raw_requests, (str, bytes)) or not isinstance(raw_requests, Sequence):
            msg = "operator-growth objective_requests must be a sequence."
            raise TypeError(msg)
        raw_provenance = mapping["training_provenance"]
        raw_resources = mapping["circuit_resources"]
        raw_pool = mapping["pool"]
        raw_spec = mapping["growth_spec"]
        raw_objective_binding = mapping["objective_binding"]
        raw_evaluator_binding = mapping["evaluator_binding"]
        result = cls(
            method_id=cast("str", mapping["method_id"]),
            algorithm_label=cast("str", mapping["algorithm_label"]),
            status=cast('Literal["completed", "not_applicable"]', mapping["status"]),
            applicability=OperatorGrowthApplicability.from_dict(mapping["applicability"]),
            pool_checksum=cast("str | None", mapping["pool_checksum"]),
            growth_spec_checksum=cast("str | None", mapping["growth_spec_checksum"]),
            objective_binding_checksum=cast("str | None", mapping["objective_binding_checksum"]),
            pool=None if raw_pool is None else OperatorPoolSpec.from_dict(raw_pool),
            growth_spec=None if raw_spec is None else OperatorGrowthSpec.from_dict(raw_spec),
            objective_binding=(
                None if raw_objective_binding is None else _objective_binding_from_dict(raw_objective_binding)
            ),
            evaluator_binding=(
                None
                if raw_evaluator_binding is None
                else StandardFixedRateOperatorGrowthEvaluatorBinding.from_dict(raw_evaluator_binding)
            ),
            selected_operator_ids=cast("tuple[str, ...]", mapping["selected_operator_ids"]),
            parameters=cast("tuple[float, ...]", mapping["parameters"]),
            initial_objective=cast("float | None", mapping["initial_objective"]),
            final_objective=cast("float | None", mapping["final_objective"]),
            native_two_qubit_counts_by_edge=cast("tuple[int, ...]", mapping["native_two_qubit_counts_by_edge"]),
            termination_reason=cast("str", mapping["termination_reason"]),
            trace=tuple(OperatorGrowthStep.from_dict(item) for item in raw_trace),
            work=OperatorGrowthWork.from_dict(mapping["work"]),
            execution_mode=cast('Literal["analytic_reference", "noisy_training"]', mapping["execution_mode"]),
            training_provenance=(
                None if raw_provenance is None else OperatorGrowthTrainingProvenance.from_dict(raw_provenance)
            ),
            objective_requests=tuple(NoisyOperatorGrowthObjectiveRequest.from_dict(item) for item in raw_requests),
            circuit_resources=(None if raw_resources is None else CircuitResourceMetrics.from_dict(raw_resources)),
        )
        if (
            require_checksum(mapping["content_checksum"], "operator-growth result.content_checksum")
            != result.content_checksum
        ):
            msg = "operator-growth result content checksum mismatch."
            raise ValueError(msg)
        return result


def _parameter_shift_gradient(
    objective: Callable[[NDArray[np.float64]], float],
    parameters: NDArray[np.float64],
    parameter_index: int,
    work: _WorkCounter,
) -> float:
    """Evaluate one exact Pauli-rotation parameter-shift derivative."""
    plus = parameters.copy()
    minus = parameters.copy()
    plus[parameter_index] += np.pi / 2.0
    minus[parameter_index] -= np.pi / 2.0
    work.gradient_calls += 1
    return 0.5 * (objective(plus) - objective(minus))


def _internal_adam(
    objective: Callable[[NDArray[np.float64]], float],
    initial_parameters: NDArray[np.float64],
    spec: OperatorGrowthSpec,
    work: _WorkCounter,
) -> NDArray[np.float64]:
    """Fully reoptimize every retained angle with deterministic small Adam."""
    theta = initial_parameters.copy()
    first_moment = np.zeros_like(theta)
    second_moment = np.zeros_like(theta)
    best_theta = theta.copy()
    best_value = objective(theta)
    for step in range(1, spec.reoptimization_steps + 1):
        gradient = np.asarray(
            [_parameter_shift_gradient(objective, theta, index, work) for index in range(theta.size)],
            dtype=np.float64,
        )
        first_moment = spec.adam_beta1 * first_moment + (1.0 - spec.adam_beta1) * gradient
        second_moment = spec.adam_beta2 * second_moment + (1.0 - spec.adam_beta2) * gradient**2
        corrected_first = first_moment / (1.0 - spec.adam_beta1**step)
        corrected_second = second_moment / (1.0 - spec.adam_beta2**step)
        theta -= spec.learning_rate * corrected_first / (np.sqrt(corrected_second) + spec.adam_epsilon)
        value = objective(theta)
        if value < best_value:
            best_value = value
            best_theta = theta.copy()
        work.reoptimization_iterations += 1
    return best_theta


def _default_spec(
    pool: OperatorPoolSpec,
    *,
    gradient_tolerance: float | None,
    max_operators: int | None,
    native_two_qubit_cap_per_edge: int | None,
    reoptimization_steps: int | None,
    learning_rate: float | None,
) -> OperatorGrowthSpec:
    """Construct a pool-bound default spec from convenience overrides."""
    return OperatorGrowthSpec.for_pool(
        pool,
        gradient_tolerance=1e-10 if gradient_tolerance is None else gradient_tolerance,
        max_operators=min(16, len(pool.operators)) if max_operators is None else max_operators,
        native_two_qubit_cap_per_edge=native_two_qubit_cap_per_edge,
        reoptimization_steps=100 if reoptimization_steps is None else reoptimization_steps,
        learning_rate=0.08 if learning_rate is None else learning_rate,
    )


def _resolve_spec(
    pool: OperatorPoolSpec,
    growth_spec: OperatorGrowthSpec | None,
    *,
    gradient_tolerance: float | None,
    max_operators: int | None,
    native_two_qubit_cap_per_edge: int | None,
    reoptimization_steps: int | None,
    learning_rate: float | None,
) -> OperatorGrowthSpec:
    """Resolve either one supplied sealed spec or explicit convenience fields."""
    overrides = (
        gradient_tolerance,
        max_operators,
        native_two_qubit_cap_per_edge,
        reoptimization_steps,
        learning_rate,
    )
    if growth_spec is None:
        return _default_spec(
            pool,
            gradient_tolerance=gradient_tolerance,
            max_operators=max_operators,
            native_two_qubit_cap_per_edge=native_two_qubit_cap_per_edge,
            reoptimization_steps=reoptimization_steps,
            learning_rate=learning_rate,
        )
    if any(value is not None for value in overrides):
        msg = "Convenience optimizer options cannot be combined with growth_spec."
        raise ValueError(msg)
    if not isinstance(growth_spec, OperatorGrowthSpec):
        msg = "growth_spec must be an OperatorGrowthSpec or None."
        raise TypeError(msg)
    if growth_spec.method_id != pool.method_id or growth_spec.pool_checksum != pool.content_checksum:
        msg = "growth_spec is not bound to the supplied method and exact pool checksum."
        raise ValueError(msg)
    return growth_spec


def _not_applicable_result(applicability: OperatorGrowthApplicability) -> OperatorGrowthResult:
    """Return a zero-work structural-not-applicability result."""
    return OperatorGrowthResult(
        method_id=ENERGY_ADAPT_METHOD_ID,
        algorithm_label="genuine_energy_adapt_vqe",
        status="not_applicable",
        applicability=applicability,
        pool_checksum=None,
        growth_spec_checksum=None,
        objective_binding_checksum=None,
        pool=None,
        growth_spec=None,
        objective_binding=None,
        evaluator_binding=None,
        selected_operator_ids=(),
        parameters=(),
        initial_objective=None,
        final_objective=None,
        native_two_qubit_counts_by_edge=(),
        termination_reason="not_applicable",
        trace=(),
        work=OperatorGrowthWork(),
        execution_mode="analytic_reference",
        training_provenance=None,
        objective_requests=(),
        circuit_resources=None,
    )


def _run_operator_growth(
    pool: OperatorPoolSpec,
    spec: OperatorGrowthSpec,
    applicability: OperatorGrowthApplicability,
    objective_binding: OperatorGrowthObjectiveBinding,
    initial_state: NDArray[np.complex128],
    raw_objective: Callable[[tuple[PoolOperator, ...], NDArray[np.float64]], float] | None,
    reoptimizer: DeterministicReoptimizer | None,
    *,
    noisy_objective: NoisyOperatorGrowthObjective | None = None,
    training_provenance: OperatorGrowthTrainingProvenance | None = None,
    evaluator_binding: StandardFixedRateOperatorGrowthEvaluatorBinding | None = None,
) -> OperatorGrowthResult:
    """Execute the shared deterministic selection and reoptimization loop."""
    if spec.method_id != pool.method_id or spec.pool_checksum != pool.content_checksum:
        msg = "Growth spec does not bind the supplied exact pool."
        raise ValueError(msg)
    if applicability.status != "applicable" or applicability.method_id != pool.method_id:
        msg = "Applicable execution requires a matching applicability record."
        raise ValueError(msg)
    if reoptimizer is None and spec.reoptimization_rule_id != _INTERNAL_ADAM_ID:
        msg = "A non-internal reoptimization rule requires a deterministic callback."
        raise ValueError(msg)
    if reoptimizer is not None and spec.reoptimization_rule_id == _INTERNAL_ADAM_ID:
        msg = "A supplied callback requires a distinct sealed reoptimization_rule_id."
        raise ValueError(msg)
    noisy_mode = training_provenance is not None or noisy_objective is not None
    if noisy_mode != (training_provenance is not None and noisy_objective is not None):
        msg = "Noisy growth requires both training_provenance and noisy_objective."
        raise ValueError(msg)
    if noisy_mode != (evaluator_binding is not None):
        msg = "Noisy growth requires one exact evaluator binding; analytic growth forbids it."
        raise ValueError(msg)
    if noisy_mode == (raw_objective is not None):
        msg = "Supply exactly one analytic or noisy operator-growth objective."
        raise ValueError(msg)
    if noisy_mode and reoptimizer is not None:
        msg = "Promotion-eligible noisy growth requires the sealed, mechanically accounted internal Adam."
        raise ValueError(msg)
    if noisy_mode != applicability.promotion_eligible:
        msg = "Applicability promotion status must exactly match noisy projector training."
        raise ValueError(msg)

    del initial_state  # The checksum-bound closure owns the immutable input bytes.
    work = _WorkCounter()
    selected: list[PoolOperator] = []
    selected_ids: set[str] = set()
    parameters = np.empty(0, dtype=np.float64)
    edge_counts = [0] * max(0, pool.num_qubits - 1)
    trace: list[OperatorGrowthStep] = []
    objective_requests: list[NoisyOperatorGrowthObjectiveRequest] = []

    def objective(theta: NDArray[np.float64]) -> float:
        if theta.shape != (len(selected),) or not np.all(np.isfinite(theta)):
            msg = "Objective received invalid retained parameters."
            raise ValueError(msg)
        if training_provenance is None:
            assert raw_objective is not None
            work.record_objective(trajectory_count=0, logical_gate_count=0)
            value = float(raw_objective(tuple(selected), theta))
        else:
            assert noisy_objective is not None
            circuit = materialize_operator_growth_circuit(pool.num_qubits, selected)
            resources = measure_circuit_resources(circuit)
            request = NoisyOperatorGrowthObjectiveRequest(
                training_provenance_checksum=training_provenance.content_checksum,
                evaluation_index=len(objective_requests),
                selected_operator_ids=tuple(operator.operator_id for operator in selected),
                parameters=tuple(float(parameter) for parameter in theta),
                circuit_resources_checksum=resources.content_checksum,
                logical_gate_count=len(circuit.gates),
                trajectory_count=training_provenance.trajectory_count,
                trajectory_ensemble_checksum=training_provenance.trajectory_ensemble_checksum,
            )
            work.record_objective(
                trajectory_count=training_provenance.trajectory_count,
                logical_gate_count=len(circuit.gates),
            )
            value = float(noisy_objective(request, circuit))
            if measure_circuit_resources(circuit).content_checksum != resources.content_checksum:
                msg = "Noisy objective callback mutated the checksum-bound circuit."
                raise ValueError(msg)
            objective_requests.append(request)
        if not math.isfinite(value):
            msg = "Operator-growth objective returned a nonfinite value."
            raise ValueError(msg)
        return value

    initial_objective = objective(parameters)
    current_objective = initial_objective
    termination_reason = "max_operators"

    while len(selected) < spec.max_operators:
        candidates: list[CandidateGradient] = []
        feasible: list[tuple[int, PoolOperator, float]] = []
        for pool_index, candidate in enumerate(pool.operators):
            if candidate.operator_id in selected_ids:
                continue
            candidate_feasible = True
            if len(candidate.sites) == 2 and spec.native_two_qubit_cap_per_edge is not None:
                edge = candidate.sites[0]
                candidate_feasible = edge_counts[edge] + candidate.native_two_qubit_gates <= (
                    spec.native_two_qubit_cap_per_edge
                )
            if candidate_feasible:
                selected.append(candidate)
                appended = np.append(parameters, 0.0)
                gradient = _parameter_shift_gradient(objective, appended, appended.size - 1, work)
                selected.pop()
                feasible.append((pool_index, candidate, gradient))
                candidates.append(
                    CandidateGradient(
                        operator_id=candidate.operator_id,
                        pool_index=pool_index,
                        gradient=gradient,
                        absolute_gradient=abs(gradient),
                        native_two_qubit_increment=candidate.native_two_qubit_gates,
                        native_cap_feasible=True,
                    )
                )
            else:
                candidates.append(
                    CandidateGradient(
                        operator_id=candidate.operator_id,
                        pool_index=pool_index,
                        gradient=None,
                        absolute_gradient=None,
                        native_two_qubit_increment=candidate.native_two_qubit_gates,
                        native_cap_feasible=False,
                    )
                )

        if not candidates:
            termination_reason = "pool_exhausted"
            break
        if not feasible:
            termination_reason = "native_cap"
            trace.append(
                OperatorGrowthStep(
                    iteration=len(selected),
                    event="native_cap_stop",
                    candidate_gradients=tuple(candidates),
                    selected_operator_id=None,
                    selected_gradient=None,
                    objective_before_reoptimization=current_objective,
                    objective_after_reoptimization=current_objective,
                    parameters=tuple(float(value) for value in parameters),
                    native_two_qubit_counts_by_edge=tuple(edge_counts),
                    work_after_step=work.snapshot(),
                )
            )
            break

        # Python's max returns the first item on equal keys, hence exact frozen
        # pool-order tie breaking follows directly from the ordered scan.
        _pool_index, chosen, chosen_gradient = max(feasible, key=lambda item: abs(item[2]))
        if abs(chosen_gradient) <= spec.gradient_tolerance:
            termination_reason = "gradient_tolerance"
            trace.append(
                OperatorGrowthStep(
                    iteration=len(selected),
                    event="gradient_stop",
                    candidate_gradients=tuple(candidates),
                    selected_operator_id=None,
                    selected_gradient=None,
                    objective_before_reoptimization=current_objective,
                    objective_after_reoptimization=current_objective,
                    parameters=tuple(float(value) for value in parameters),
                    native_two_qubit_counts_by_edge=tuple(edge_counts),
                    work_after_step=work.snapshot(),
                )
            )
            break

        selected.append(chosen)
        selected_ids.add(chosen.operator_id)
        parameters = np.append(parameters, 0.0)
        if len(chosen.sites) == 2:
            edge_counts[chosen.sites[0]] += chosen.native_two_qubit_gates
        before_reoptimization = objective(parameters)

        if reoptimizer is None:
            parameters = _internal_adam(objective, parameters, spec, work)
        else:
            callback_initial = parameters.copy()
            callback_initial.setflags(write=False)
            calls_before_callback = work.objective_calls
            outcome = reoptimizer(objective, callback_initial)
            if not isinstance(outcome, DeterministicReoptimizationResult):
                msg = "Deterministic callback must return DeterministicReoptimizationResult."
                raise TypeError(msg)
            if len(outcome.parameters) != len(selected):
                msg = "Deterministic callback returned the wrong parameter count."
                raise ValueError(msg)
            callback_objective_calls = work.objective_calls - calls_before_callback
            if callback_objective_calls < 2 * outcome.gradient_calls:
                msg = "Callback gradient work is inconsistent with the frozen two-evaluation parameter-shift rule."
                raise ValueError(msg)
            parameters = np.asarray(outcome.parameters, dtype=np.float64)
            work.gradient_calls += outcome.gradient_calls
            work.reoptimization_iterations += outcome.iterations
        current_objective = objective(parameters)
        trace.append(
            OperatorGrowthStep(
                iteration=len(selected) - 1,
                event="selected",
                candidate_gradients=tuple(candidates),
                selected_operator_id=chosen.operator_id,
                selected_gradient=chosen_gradient,
                objective_before_reoptimization=before_reoptimization,
                objective_after_reoptimization=current_objective,
                parameters=tuple(float(value) for value in parameters),
                native_two_qubit_counts_by_edge=tuple(edge_counts),
                work_after_step=work.snapshot(),
            )
        )
    else:
        termination_reason = "max_operators"

    label = "projector_operator_growth" if pool.method_id == ADAPT_STYLE_METHOD_ID else "genuine_energy_adapt_vqe"
    circuit_resources = measure_circuit_resources(materialize_operator_growth_circuit(pool.num_qubits, selected))
    if noisy_mode:
        if (
            not isinstance(objective_binding, ProjectorObjectiveSpec)
            or training_provenance is None
            or evaluator_binding is None
        ):
            msg = "Noisy operator growth requires exact projector, provenance, and evaluator bindings."
            raise ValueError(msg)
        objective_binding_checksum = canonical_checksum({
            "projector_objective_checksum": objective_binding.content_checksum,
            "training_provenance_checksum": training_provenance.content_checksum,
            "evaluator_checksum": evaluator_binding.content_checksum,
        })
    else:
        objective_binding_checksum = objective_binding.content_checksum
    return OperatorGrowthResult(
        method_id=pool.method_id,
        algorithm_label=label,
        status="completed",
        applicability=applicability,
        pool_checksum=pool.content_checksum,
        growth_spec_checksum=spec.content_checksum,
        objective_binding_checksum=objective_binding_checksum,
        pool=pool,
        growth_spec=spec,
        objective_binding=objective_binding,
        evaluator_binding=evaluator_binding,
        selected_operator_ids=tuple(operator.operator_id for operator in selected),
        parameters=tuple(float(value) for value in parameters),
        initial_objective=initial_objective,
        final_objective=current_objective,
        native_two_qubit_counts_by_edge=tuple(edge_counts),
        termination_reason=termination_reason,
        trace=tuple(trace),
        work=work.snapshot(),
        execution_mode="noisy_training" if noisy_mode else "analytic_reference",
        training_provenance=training_provenance,
        objective_requests=tuple(objective_requests),
        circuit_resources=circuit_resources,
    )


def adapt_style_state_preparation(
    target_state: object,
    *,
    family_id: str = "gaussian_amplitude",
    initial_state: object | None = None,
    pool: OperatorPoolSpec | None = None,
    growth_spec: OperatorGrowthSpec | None = None,
    reoptimizer: DeterministicReoptimizer | None = None,
    gradient_tolerance: float | None = None,
    max_operators: int | None = None,
    native_two_qubit_cap_per_edge: int | None = None,
    reoptimization_steps: int | None = None,
    learning_rate: float | None = None,
) -> OperatorGrowthResult:
    """Run the dense analytic reference for projector/fidelity growth.

    This deterministic dense-state helper is useful for analytic tests and
    reference selection behavior. It does not perform the noisy training
    sealed for the Phase II candidate and is therefore never promotion
    eligible. Use :func:`noisy_adapt_style_state_preparation` for a typed,
    provenance-bound noisy execution.

    Args:
        target_state: Normalized target statevector.
        family_id: Target family identifier.
        initial_state: Optional fixed circuit input, defaulting to all-zero.
        pool: Optional exact projector pool.
        growth_spec: Optional exact pool-bound growth specification.
        reoptimizer: Optional deterministic full-parameter callback.
        gradient_tolerance: Convenience override when no spec is supplied.
        max_operators: Convenience maximum when no spec is supplied.
        native_two_qubit_cap_per_edge: Optional per-edge native RZZ cap.
        reoptimization_steps: Convenience internal-Adam step count.
        learning_rate: Convenience internal-Adam learning rate.

    Returns:
        A checksum-sealed operator-growth result.
    """
    applicability = operator_growth_applicability(ADAPT_STYLE_METHOD_ID, family_id)
    target, qubits = _normalized_state(target_state, "target_state")
    initial = (
        computational_zero_state(qubits)
        if initial_state is None
        else _normalized_state(
            initial_state,
            "initial_state",
            num_qubits=qubits,
        )[0]
    )
    objective_spec = ProjectorObjectiveSpec(
        qubits,
        _array_checksum(target),
        _array_checksum(initial),
    )
    resolved_pool = build_projector_operator_pool(qubits) if pool is None else pool
    if not isinstance(resolved_pool, OperatorPoolSpec) or (
        resolved_pool.method_id != ADAPT_STYLE_METHOD_ID or resolved_pool.num_qubits != qubits
    ):
        msg = "pool must be the exact projector pool for the target width."
        raise ValueError(msg)
    spec = _resolve_spec(
        resolved_pool,
        growth_spec,
        gradient_tolerance=gradient_tolerance,
        max_operators=max_operators,
        native_two_qubit_cap_per_edge=native_two_qubit_cap_per_edge,
        reoptimization_steps=reoptimization_steps,
        learning_rate=learning_rate,
    )

    def objective(selected: tuple[PoolOperator, ...], theta: NDArray[np.float64]) -> float:
        prepared = operator_growth_state(qubits, selected, theta, initial_state=initial)
        fidelity = float(abs(np.vdot(target, prepared)) ** 2)
        return float(np.clip(1.0 - fidelity, 0.0, 1.0))

    return _run_operator_growth(
        resolved_pool,
        spec,
        applicability,
        objective_spec,
        initial,
        objective,
        reoptimizer,
    )


def noisy_adapt_style_state_preparation(
    target: MaterializedTarget | LegacyMaterializedTarget,
    evaluator: StandardFixedRateNoisyOperatorGrowthEvaluator,
    *,
    pool: OperatorPoolSpec | None = None,
    growth_spec: OperatorGrowthSpec | None = None,
    gradient_tolerance: float | None = None,
    max_operators: int | None = None,
    native_two_qubit_cap_per_edge: int | None = None,
    reoptimization_steps: int | None = None,
    learning_rate: float | None = None,
) -> OperatorGrowthResult:
    """Run family-wide operator growth against a target-bound noisy evaluator.

    Only :class:`StandardFixedRateNoisyOperatorGrowthEvaluator` crosses this
    promotion-eligible boundary. It derives provider, objective, target, and
    common-trajectory identities itself and executes the actual standard-noise
    simulator, preventing caller-asserted checksums or generic callbacks.

    Args:
        target: Authorized materialized target owned by ``evaluator``.
        evaluator: Concrete standard fixed-rate noisy objective.
        pool: Optional exact projector pool.
        growth_spec: Optional exact pool-bound growth specification.
        gradient_tolerance: Convenience override when no spec is supplied.
        max_operators: Convenience maximum when no spec is supplied.
        native_two_qubit_cap_per_edge: Optional per-edge native RZZ cap.
        reoptimization_steps: Convenience internal-Adam step count.
        learning_rate: Convenience internal-Adam learning rate.

    Returns:
        Promotion-eligible noisy growth evidence with complete work accounting.
    """
    if not isinstance(target, (MaterializedTarget, LegacyMaterializedTarget)):
        msg = "target must be an authorized MaterializedTarget or LegacyMaterializedTarget."
        raise TypeError(msg)
    if type(evaluator) is not StandardFixedRateNoisyOperatorGrowthEvaluator:
        msg = "evaluator must be a StandardFixedRateNoisyOperatorGrowthEvaluator."
        raise TypeError(msg)
    if evaluator.target.identity_dict() != target.identity_dict():
        msg = "evaluator is not bound to the supplied authorized target."
        raise ValueError(msg)
    applicability = operator_growth_applicability(
        ADAPT_STYLE_METHOD_ID,
        target.family_id,
        noisy_training=True,
    )
    qubits = target.qubit_count
    initial = computational_zero_state(qubits)
    training_provenance = evaluator.training_provenance
    resolved_pool = build_projector_operator_pool(qubits) if pool is None else pool
    if not isinstance(resolved_pool, OperatorPoolSpec) or (
        resolved_pool.method_id != ADAPT_STYLE_METHOD_ID or resolved_pool.num_qubits != qubits
    ):
        msg = "pool must be the exact projector pool for the target width."
        raise ValueError(msg)
    spec = _resolve_spec(
        resolved_pool,
        growth_spec,
        gradient_tolerance=gradient_tolerance,
        max_operators=max_operators,
        native_two_qubit_cap_per_edge=native_two_qubit_cap_per_edge,
        reoptimization_steps=reoptimization_steps,
        learning_rate=learning_rate,
    )
    return _run_operator_growth(
        resolved_pool,
        spec,
        applicability,
        evaluator.objective_spec,
        initial,
        None,
        None,
        noisy_objective=evaluator,
        training_provenance=training_provenance,
        evaluator_binding=evaluator.binding,
    )


def run_standard_fixed_rate_noisy_operator_growth(
    target: MaterializedTarget | LegacyMaterializedTarget,
    *,
    optimization_block_id: str,
    optimization_seed: int,
    resource_stratum_id: str,
    noise_id: str,
    noise_definition_version: str,
    noise_strength_scale: float,
    tjm_dt: float,
    trajectory_count: int,
    trajectory_seed: int,
    pool: OperatorPoolSpec | None = None,
    growth_spec: OperatorGrowthSpec | None = None,
    gradient_tolerance: float | None = None,
    max_operators: int | None = None,
    native_two_qubit_cap_per_edge: int | None = None,
    reoptimization_steps: int | None = None,
    learning_rate: float | None = None,
) -> OperatorGrowthResult:
    """Construct and execute a complete standard fixed-rate noisy comparator.

    Args:
        target: Authorized Phase II or legacy materialized target.
        optimization_block_id: Exact paired optimization-block identity.
        optimization_seed: Unsigned 64-bit outer optimization seed.
        resource_stratum_id: Exact paired resource-stratum identity.
        noise_id: Standard fixed-rate noise identifier.
        noise_definition_version: Exact fixed-rate definition version.
        noise_strength_scale: Strictly positive provider strength multiplier.
        tjm_dt: Strictly positive TJM time step.
        trajectory_count: Positive training trajectory count per objective.
        trajectory_seed: Unsigned 64-bit common trajectory seed.
        pool: Optional exact projector pool.
        growth_spec: Optional exact pool-bound growth specification.
        gradient_tolerance: Convenience override when no spec is supplied.
        max_operators: Convenience maximum when no spec is supplied.
        native_two_qubit_cap_per_edge: Optional per-edge native RZZ cap.
        reoptimization_steps: Convenience internal-Adam step count.
        learning_rate: Convenience internal-Adam learning rate.

    Returns:
        Promotion-eligible result backed by actual standard-noise trajectories.
    """
    evaluator = StandardFixedRateNoisyOperatorGrowthEvaluator(
        target,
        optimization_block_id=optimization_block_id,
        optimization_seed=optimization_seed,
        resource_stratum_id=resource_stratum_id,
        noise_id=noise_id,
        noise_definition_version=noise_definition_version,
        noise_strength_scale=noise_strength_scale,
        tjm_dt=tjm_dt,
        trajectory_count=trajectory_count,
        trajectory_seed=trajectory_seed,
    )
    return noisy_adapt_style_state_preparation(
        target,
        evaluator,
        pool=pool,
        growth_spec=growth_spec,
        gradient_tolerance=gradient_tolerance,
        max_operators=max_operators,
        native_two_qubit_cap_per_edge=native_two_qubit_cap_per_edge,
        reoptimization_steps=reoptimization_steps,
        learning_rate=learning_rate,
    )


def energy_adapt_vqe(
    family_id: str,
    couplings: Sequence[float] | NDArray[np.float64] | None = None,
    fields: Sequence[float] | NDArray[np.float64] | None = None,
    *,
    initial_state: object | None = None,
    pool: OperatorPoolSpec | None = None,
    growth_spec: OperatorGrowthSpec | None = None,
    reoptimizer: DeterministicReoptimizer | None = None,
    gradient_tolerance: float | None = None,
    max_operators: int | None = None,
    native_two_qubit_cap_per_edge: int | None = None,
    reoptimization_steps: int | None = None,
    learning_rate: float | None = None,
) -> OperatorGrowthResult:
    """Run genuine Hamiltonian-energy ADAPT-VQE on the TFIM subset only.

    This caller-parameterized dense helper is retained for analytic reference
    tests and is never promotion eligible. Publishable Phase II target-bound
    evidence must use :func:`target_bound_energy_adapt_vqe` so family and
    Hamiltonian parameters are derived from an authorized typed target spec.

    Applicability is decided before inspecting couplings, fields, an initial
    state, a pool, or an objective configuration.  Thus non-TFIM cells return
    typed zero-work not-applicable evidence and are not optimizer failures.
    No target state is accepted: selection is bound exclusively to the frozen
    TFIM couplings and fields.

    Args:
        family_id: Target family identifier.
        couplings: Open-chain ``ZZ`` couplings for TFIM cells.
        fields: On-site transverse fields for TFIM cells.
        initial_state: Optional fixed input state, defaulting to all-zero.
        pool: Optional exact real-state TFIM pool.
        growth_spec: Optional exact pool-bound growth specification.
        reoptimizer: Optional deterministic full-parameter callback.
        gradient_tolerance: Convenience override when no spec is supplied.
        max_operators: Convenience maximum when no spec is supplied.
        native_two_qubit_cap_per_edge: Optional per-edge native RZZ cap.
        reoptimization_steps: Convenience internal-Adam step count.
        learning_rate: Convenience internal-Adam learning rate.

    Returns:
        A completed result for TFIM or typed zero-work not-applicable evidence.
    """
    applicability = operator_growth_applicability(ENERGY_ADAPT_METHOD_ID, family_id)
    if applicability.status == "not_applicable":
        return _not_applicable_result(applicability)

    if couplings is None or fields is None:
        msg = "Applicable TFIM energy growth requires couplings and fields."
        raise ValueError(msg)
    hamiltonian = dense_open_chain_tfim_hamiltonian(couplings, fields)
    qubits = int(np.asarray(fields).size)
    initial = (
        computational_zero_state(qubits)
        if initial_state is None
        else _normalized_state(
            initial_state,
            "initial_state",
            num_qubits=qubits,
        )[0]
    )
    objective_spec = TFIMHamiltonianSpec(
        couplings=tuple(float(value) for value in np.asarray(couplings, dtype=np.float64)),
        fields=tuple(float(value) for value in np.asarray(fields, dtype=np.float64)),
        initial_state_checksum=_array_checksum(initial),
    )
    resolved_pool = build_tfim_real_operator_pool(qubits) if pool is None else pool
    if not isinstance(resolved_pool, OperatorPoolSpec) or (
        resolved_pool.method_id != ENERGY_ADAPT_METHOD_ID or resolved_pool.num_qubits != qubits
    ):
        msg = "pool must be the exact real-state TFIM pool for the Hamiltonian width."
        raise ValueError(msg)
    spec = _resolve_spec(
        resolved_pool,
        growth_spec,
        gradient_tolerance=gradient_tolerance,
        max_operators=max_operators,
        native_two_qubit_cap_per_edge=native_two_qubit_cap_per_edge,
        reoptimization_steps=reoptimization_steps,
        learning_rate=learning_rate,
    )

    def objective(selected: tuple[PoolOperator, ...], theta: NDArray[np.float64]) -> float:
        prepared = operator_growth_state(qubits, selected, theta, initial_state=initial)
        value = np.vdot(prepared, hamiltonian @ prepared)
        return float(value.real)

    return _run_operator_growth(
        resolved_pool,
        spec,
        applicability,
        objective_spec,
        initial,
        objective,
        reoptimizer,
    )


def target_bound_energy_adapt_vqe(
    target: MaterializedTarget,
    target_instance_spec: TargetInstanceSpec,
    *,
    pool: OperatorPoolSpec | None = None,
    growth_spec: OperatorGrowthSpec | None = None,
    reoptimizer: DeterministicReoptimizer | None = None,
    gradient_tolerance: float | None = None,
    max_operators: int | None = None,
    native_two_qubit_cap_per_edge: int | None = None,
    reoptimization_steps: int | None = None,
    learning_rate: float | None = None,
) -> OperatorGrowthResult:
    """Run analytic energy ADAPT from one authorized typed TFIM target.

    Family, register width, couplings, fields, target identity, and the
    computational-zero input are all derived from the supplied Phase II target
    and its exact seed-bearing specification. No caller-provided Hamiltonian or
    family alias can cross this target-bound entry point. The dense analytic
    execution remains reference-only and is not promotion eligible.

    Args:
        target: Authorized materialized Phase II target.
        target_instance_spec: Exact manifest target specification for ``target``.
        pool: Optional exact real-state TFIM pool.
        growth_spec: Optional exact pool-bound growth specification.
        reoptimizer: Optional deterministic full-parameter callback.
        gradient_tolerance: Convenience override when no spec is supplied.
        max_operators: Convenience maximum when no spec is supplied.
        native_two_qubit_cap_per_edge: Optional per-edge native RZZ cap.
        reoptimization_steps: Convenience internal-Adam step count.
        learning_rate: Convenience internal-Adam learning rate.

    Returns:
        A target-bound, checksum-sealed analytic energy-ADAPT result.
    """
    objective_binding = TargetBoundTFIMEnergyObjectiveSpec.from_target(target, target_instance_spec)
    hamiltonian_binding = objective_binding.hamiltonian_binding
    qubits = target_instance_spec.qubit_count
    initial = computational_zero_state(qubits)
    hamiltonian = hamiltonian_binding.dense_matrix()
    applicability = operator_growth_applicability(
        ENERGY_ADAPT_METHOD_ID,
        target_instance_spec.family_id,
    )
    resolved_pool = build_tfim_real_operator_pool(qubits) if pool is None else pool
    if not isinstance(resolved_pool, OperatorPoolSpec) or (
        resolved_pool.method_id != ENERGY_ADAPT_METHOD_ID or resolved_pool.num_qubits != qubits
    ):
        msg = "pool must be the exact real-state TFIM pool for the authorized target width."
        raise ValueError(msg)
    spec = _resolve_spec(
        resolved_pool,
        growth_spec,
        gradient_tolerance=gradient_tolerance,
        max_operators=max_operators,
        native_two_qubit_cap_per_edge=native_two_qubit_cap_per_edge,
        reoptimization_steps=reoptimization_steps,
        learning_rate=learning_rate,
    )

    def objective(selected: tuple[PoolOperator, ...], theta: NDArray[np.float64]) -> float:
        prepared = operator_growth_state(qubits, selected, theta, initial_state=initial)
        value = np.vdot(prepared, hamiltonian @ prepared)
        return float(value.real)

    return _run_operator_growth(
        resolved_pool,
        spec,
        applicability,
        objective_binding,
        initial,
        objective,
        reoptimizer,
    )


__all__ = [
    "ADAPT_STYLE_METHOD_ID",
    "CONNECTIVITY_ID",
    "ENERGY_ADAPT_METHOD_ID",
    "OPERATOR_GROWTH_APPLICABILITY_SCHEMA_VERSION",
    "OPERATOR_GROWTH_OBJECTIVE_REQUEST_SCHEMA_VERSION",
    "OPERATOR_GROWTH_RESULT_SCHEMA_VERSION",
    "OPERATOR_GROWTH_SPEC_SCHEMA_VERSION",
    "OPERATOR_GROWTH_TRAINING_PROVENANCE_SCHEMA_VERSION",
    "OPERATOR_POOL_SCHEMA_VERSION",
    "PROJECTOR_COST_ID",
    "PROJECTOR_OBJECTIVE_SCHEMA_VERSION",
    "QUANTINUUM_NATIVE_POLICY_ID",
    "ROUTING_POLICY_ID",
    "STANDARD_FIXED_RATE_OPERATOR_GROWTH_BINDING_SCHEMA_VERSION",
    "STANDARD_FIXED_RATE_OPERATOR_GROWTH_EVALUATOR_SCHEMA_VERSION",
    "TARGET_BOUND_TFIM_OBJECTIVE_SCHEMA_VERSION",
    "TFIM_ENERGY_COST_ID",
    "TFIM_FAMILY_ID",
    "TFIM_HAMILTONIAN_SCHEMA_VERSION",
    "CandidateGradient",
    "DeterministicReoptimizationResult",
    "DeterministicReoptimizer",
    "NoisyOperatorGrowthObjective",
    "NoisyOperatorGrowthObjectiveRequest",
    "OperatorGrowthApplicability",
    "OperatorGrowthObjectiveBinding",
    "OperatorGrowthResult",
    "OperatorGrowthSpec",
    "OperatorGrowthStep",
    "OperatorGrowthTrainingProvenance",
    "OperatorGrowthWork",
    "OperatorPoolSpec",
    "PoolOperator",
    "ProjectorObjectiveSpec",
    "StandardFixedRateNoisyOperatorGrowthEvaluator",
    "StandardFixedRateOperatorGrowthEvaluatorBinding",
    "TFIMHamiltonianSpec",
    "TargetBoundTFIMEnergyObjectiveSpec",
    "adapt_style_state_preparation",
    "build_projector_operator_pool",
    "build_tfim_real_operator_pool",
    "computational_zero_state",
    "dense_open_chain_tfim_hamiltonian",
    "energy_adapt_vqe",
    "materialize_operator_growth_circuit",
    "noisy_adapt_style_state_preparation",
    "open_chain_tfim_hamiltonian",
    "operator_growth_applicability",
    "operator_growth_state",
    "projector_infidelity",
    "projector_operator_pool",
    "run_standard_fixed_rate_noisy_operator_growth",
    "target_bound_energy_adapt_vqe",
    "tfim_energy",
    "tfim_real_operator_pool",
]
