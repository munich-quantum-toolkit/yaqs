# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Method adapters for the state-preparation benchmarks."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, TypeVar, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import (
    KrotovOptions,
    KrotovResult,
    KrotovTruncation,
    ParameterizedCircuit,
    create_brickwall_matrix_product_disentangler_parameterized_circuit,
    state_preparation_metrics,
    train_krotov_state_preparation_batch,
)

from .schema import (
    AnsatzConfig,
    BenchmarkConfig,
    EvaluationConfig,
    InitializationConfig,
    OptimizerConfig,
    TargetSelection,
)
from .targets import TargetCollection, TargetRecord

if TYPE_CHECKING:
    from mqt.yaqs.core.methods.decompositions import TruncMode

KROTOV_METHOD_ID = "krotov"
KROTOV_METHOD_NAME = "Krotov"
KROTOV_METHOD_VERSION = "1"
KROTOV_CHECKPOINT_FORMAT = "yaqs.state_preparation.krotov_checkpoint.v1"
KROTOV_PARAMETER_LAYOUT_FORMAT = "yaqs.state_preparation.krotov_parameter_layout.v1"
TRAINING_IDENTITY_VERSION = "yaqs.state_preparation.training_identity.v1"
TRAINING_ID_PREFIX = "spt-v1-"

_KROTOV_OPTIMIZER_ID = "krotov"
_KROTOV_IMPLEMENTATION = "mqt.yaqs.optimization.train_krotov_state_preparation_batch"
_SUPPORTED_SCHEDULES = frozenset({"constant", "inverse", "exp"})
_KROTOV_HYPERPARAMETERS = frozenset({"step_size", "schedule"})
_TRACE_FIELDS = (
    "step",
    "phase",
    "loss",
    "fidelity",
    "step_size",
    "gradient_norm",
    "update_norm",
)
_CHECKPOINT_FIELDS = (
    "checkpoint_format",
    "method_id",
    "method_version",
    "parameter_layout_checksum",
    "num_qubits",
    "num_parameters",
    "theta",
)
_CHECKPOINT_ARCHIVE_MEMBERS = frozenset(f"{name}.npy" for name in _CHECKPOINT_FIELDS)
_CHECKPOINT_TEXT_FIELDS = frozenset({
    "checkpoint_format",
    "method_id",
    "method_version",
    "parameter_layout_checksum",
})
_CHECKPOINT_SCALAR_FIELDS = frozenset({"num_qubits", "num_parameters"})
_NPY_HEADER_ALLOWANCE = 4096
StatePreparationTarget: TypeAlias = TargetRecord | MPS | NDArray[np.complex128]
_ScheduleKind: TypeAlias = Literal["constant", "inverse", "exp"]
_TrainingFailurePhase: TypeAlias = Literal[
    "target_loading",
    "ansatz",
    "initialization",
    "optimization",
    "checkpoint",
]
_ResultT = TypeVar("_ResultT")
_StageT = TypeVar("_StageT")
_ARTIFACT_VALIDATION_TOKEN = object()


@runtime_checkable
class StatePreparationMethod(Protocol[_ResultT]):
    """Structural interface implemented by benchmark method adapters."""

    method_id: str
    method_name: str
    method_version: str

    def build_ansatz(self, num_qubits: int, ansatz: AnsatzConfig) -> ParameterizedCircuit:
        """Construct the shared logical ansatz."""
        ...

    def initialize_parameters(
        self,
        circuit: ParameterizedCircuit,
        initialization: InitializationConfig,
        *,
        checkpoint_root: Path | None = None,
    ) -> NDArray[np.float64]:
        """Return one fully resolved initial parameter vector."""
        ...

    def optimize_noiseless(
        self,
        circuit: ParameterizedCircuit,
        target: StatePreparationTarget,
        initial_parameters: NDArray[np.float64],
        optimizer: OptimizerConfig,
    ) -> _ResultT:
        """Optimize the ansatz without training noise."""
        ...

    def extract_final_parameters(self, result: _ResultT) -> NDArray[np.float64]:
        """Extract a detached final parameter vector."""
        ...

    def extract_training_fidelity(self, result: _ResultT) -> float:
        """Extract the final training fidelity."""
        ...

    def evaluate_noiseless(
        self,
        circuit: ParameterizedCircuit,
        parameters: NDArray[np.float64],
        target: StatePreparationTarget,
        *,
        evaluation: EvaluationConfig | None = None,
    ) -> float:
        """Evaluate final noiseless target-state fidelity."""
        ...

    def optimizer_metadata(self, optimizer: OptimizerConfig) -> dict[str, object]:
        """Return complete normalized optimizer metadata."""
        ...

    def serialize_checkpoint(self, circuit: ParameterizedCircuit, result: _ResultT) -> bytes:
        """Serialize one optimization result without writing it."""
        ...

    def deserialize_checkpoint(
        self,
        circuit: ParameterizedCircuit,
        payload: bytes,
        *,
        expected_checksum: str | None = None,
    ) -> NDArray[np.float64]:
        """Deserialize and validate one parameter checkpoint."""
        ...


def _canonical_json(value: object) -> str:
    """Serialize a JSON-native object deterministically.

    Returns:
        The canonical JSON document.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _require_exact_keys(value: Mapping[str, object], expected: frozenset[str], name: str) -> None:
    """Reject missing and unknown mapping keys.

    Raises:
        ValueError: If keys are missing or unknown.
    """
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing keys: {missing}")
        if unknown:
            details.append(f"unknown keys: {unknown}")
        msg = f"Invalid {name}: {'; '.join(details)}."
        raise ValueError(msg)


def _require_finite_float(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    """Return a finite real value with an optional lower bound.

    Returns:
        The normalized floating-point value.

    Raises:
        TypeError: If the value is not a non-Boolean real number.
        ValueError: If the value is non-finite or outside the bound.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        msg = f"{name} must be a real number, got {type(value).__name__}."
        raise TypeError(msg)
    try:
        result = float(value)
    except OverflowError as error:
        msg = f"{name} must be finite."
        raise ValueError(msg) from error
    if not math.isfinite(result):
        msg = f"{name} must be finite."
        raise ValueError(msg)
    if minimum is not None:
        invalid = result < minimum if minimum_inclusive else result <= minimum
        if invalid:
            relation = "at least" if minimum_inclusive else "greater than"
            msg = f"{name} must be {relation} {minimum}."
            raise ValueError(msg)
    return result


def _validated_parameter_vector(
    parameters: object,
    expected_count: int,
    name: str,
) -> NDArray[np.float64]:
    """Return a detached canonical parameter vector.

    Returns:
        A one-dimensional, finite, C-contiguous ``float64`` array.

    Raises:
        TypeError: If the value is not a real numeric array.
        ValueError: If its shape or values are invalid.
    """
    array = np.asarray(parameters)
    if array.dtype.kind not in {"f", "i", "u"}:
        msg = f"{name} must contain real numeric values."
        raise TypeError(msg)
    if array.shape != (expected_count,):
        msg = f"{name} must have shape ({expected_count},), got {array.shape}."
        raise ValueError(msg)
    result = np.ascontiguousarray(array, dtype=np.dtype("<f8"))
    if not np.all(np.isfinite(result)):
        msg = f"{name} must contain only finite values."
        raise ValueError(msg)
    return result.copy()


def _validated_fidelity(value: object, name: str) -> float:
    """Return a fidelity clipped only for roundoff at the physical boundary.

    Returns:
        A value in ``[0, 1]``.

    Raises:
        ValueError: If the value lies outside the physical range beyond
            numerical roundoff.
    """
    result = _require_finite_float(value, name)
    tolerance = 1e-12
    if result < -tolerance or result > 1.0 + tolerance:
        msg = f"{name} must lie in [0, 1]."
        raise ValueError(msg)
    return min(1.0, max(0.0, result))


def _target_state(target: StatePreparationTarget, num_qubits: int) -> MPS | NDArray[np.complex128]:
    """Resolve a target record without exposing its stored array.

    Returns:
        A target MPS or detached dense state vector.

    Raises:
        TypeError: If the target has an unsupported type.
        ValueError: If a target record and circuit use different qubit counts.
    """
    if isinstance(target, TargetRecord):
        if target.num_qubits != num_qubits:
            msg = f"Target record uses {target.num_qubits} qubits, but the circuit uses {num_qubits}."
            raise ValueError(msg)
        return target.state_vector_copy()
    if isinstance(target, MPS):
        return target
    if isinstance(target, np.ndarray):
        return np.asarray(target, dtype=np.complex128).copy()
    msg = f"target must be a TargetRecord, MPS, or ndarray, got {type(target).__name__}."
    raise TypeError(msg)


def _schedule_metadata(value: object) -> tuple[_ScheduleKind, float]:
    """Validate one high-level Krotov schedule specification.

    Returns:
        The schedule kind and decay.

    Raises:
        TypeError: If the schedule representation has an unsupported type.
        ValueError: If schedule keys or values are invalid.
    """
    if value is None:
        return "constant", 0.0
    if isinstance(value, str):
        kind: object = value
        decay = 0.0
    elif isinstance(value, Mapping):
        schedule = cast("Mapping[str, object]", value)
        allowed = frozenset({"kind", "decay"})
        unknown = sorted(set(schedule) - allowed)
        if unknown or "kind" not in schedule:
            details = []
            if "kind" not in schedule:
                details.append("missing key: 'kind'")
            if unknown:
                details.append(f"unknown keys: {unknown}")
            msg = f"Invalid Krotov schedule: {'; '.join(details)}."
            raise ValueError(msg)
        kind = schedule["kind"]
        decay = _require_finite_float(schedule.get("decay", 0.0), "schedule decay", minimum=0.0)
    else:
        msg = f"schedule must be a string or mapping, got {type(value).__name__}."
        raise TypeError(msg)
    if type(kind) is not str or kind not in _SUPPORTED_SCHEDULES:
        msg = f"schedule kind must be one of {sorted(_SUPPORTED_SCHEDULES)}."
        raise ValueError(msg)
    if kind == "constant" and decay > 0.0:
        msg = "A constant schedule must use zero decay."
        raise ValueError(msg)
    return cast("_ScheduleKind", kind), decay


def _validate_optimizer(optimizer: OptimizerConfig) -> tuple[float, _ScheduleKind, float]:
    """Validate the Krotov-specific optimizer configuration.

    Returns:
        The resolved step size, schedule kind, and schedule decay.

    Raises:
        TypeError: If ``optimizer`` has the wrong type.
        ValueError: If the optimizer requests unsupported behavior.
    """
    if not isinstance(optimizer, OptimizerConfig):
        msg = f"optimizer must be an OptimizerConfig, got {type(optimizer).__name__}."
        raise TypeError(msg)
    if optimizer.optimizer_id != _KROTOV_OPTIMIZER_ID:
        msg = f"KrotovStatePreparationMethod requires optimizer_id={_KROTOV_OPTIMIZER_ID!r}."
        raise ValueError(msg)
    if optimizer.train_trajectories_or_shots != 0 or optimizer.training_seed is not None:
        msg = "KrotovStatePreparationMethod v1 supports noiseless optimization only."
        raise ValueError(msg)
    hyperparameters = optimizer.hyperparameters
    unknown = sorted(set(hyperparameters) - _KROTOV_HYPERPARAMETERS)
    if unknown:
        msg = f"Unknown Krotov hyperparameters: {unknown}."
        raise ValueError(msg)
    step_size = _require_finite_float(
        hyperparameters.get("step_size", 0.1),
        "step_size",
        minimum=0.0,
        minimum_inclusive=False,
    )
    schedule, decay = _schedule_metadata(hyperparameters.get("schedule"))
    return step_size, schedule, decay


def _truncation(optimizer: OptimizerConfig) -> KrotovTruncation:
    """Translate benchmark truncation metadata to Krotov settings.

    Returns:
        The complete Krotov truncation configuration.
    """
    return KrotovTruncation(
        max_bond_dim=optimizer.max_bond_dimension,
        svd_threshold=optimizer.svd_threshold,
        trunc_mode=cast("TruncMode", optimizer.truncation_mode),
        min_bond_dim=optimizer.min_bond_dimension,
    )


def _evaluation_truncation(evaluation: EvaluationConfig | None) -> KrotovTruncation:
    """Translate independent evaluation truncation settings.

    Returns:
        Exact defaults or the complete configured truncation policy.

    Raises:
        TypeError: If ``evaluation`` has the wrong type.
    """
    if evaluation is None:
        return KrotovTruncation()
    if not isinstance(evaluation, EvaluationConfig):
        msg = f"evaluation must be an EvaluationConfig or None, got {type(evaluation).__name__}."
        raise TypeError(msg)
    return KrotovTruncation(
        max_bond_dim=evaluation.max_bond_dimension,
        svd_threshold=evaluation.svd_threshold,
        trunc_mode=cast("TruncMode", evaluation.truncation_mode),
        min_bond_dim=evaluation.min_bond_dimension,
    )


def _krotov_options(optimizer: OptimizerConfig) -> KrotovOptions:
    """Translate and validate one optimizer configuration.

    Returns:
        Full-batch Krotov options.
    """
    step_size, schedule, decay = _validate_optimizer(optimizer)
    return KrotovOptions(
        max_iterations=optimizer.max_iterations,
        switch_iteration=0,
        online_step_size=step_size,
        batch_step_size=step_size,
        online_schedule=schedule,
        batch_schedule=schedule,
        online_decay=decay,
        batch_decay=decay,
        seed=optimizer.optimizer_seed,
        truncation=_truncation(optimizer),
    )


def _validated_trace(trace: object) -> dict[str, list[float | int | str]]:
    """Validate and detach a complete state-preparation training trace.

    Returns:
        A detached trace with all required fields.

    Raises:
        TypeError: If a container or trace entry has the wrong type.
        ValueError: If fields, lengths, or numeric values are invalid.
    """
    if not isinstance(trace, Mapping):
        msg = f"Krotov trace must be a mapping, got {type(trace).__name__}."
        raise TypeError(msg)
    source = cast("Mapping[str, object]", trace)
    _require_exact_keys(source, frozenset(_TRACE_FIELDS), "Krotov trace")
    columns: dict[str, list[float | int | str]] = {}
    for name in _TRACE_FIELDS:
        values = source[name]
        if not isinstance(values, list):
            msg = f"Krotov trace field {name!r} must be a list."
            raise TypeError(msg)
        columns[name] = list(cast("list[float | int | str]", values))
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        msg = "Krotov trace fields must have one equal, nonzero length."
        raise ValueError(msg)

    for index, step in enumerate(columns["step"]):
        if isinstance(step, (bool, np.bool_)) or not isinstance(step, Integral) or step < 0:
            msg = f"Krotov trace step {index} must be a nonnegative integer."
            raise TypeError(msg)
        columns["step"][index] = int(step)
    expected_steps = list(range(len(columns["step"])))
    if columns["step"] != expected_steps:
        msg = f"Krotov trace steps must be consecutive from zero, got {columns['step']}."
        raise ValueError(msg)

    for index, phase in enumerate(columns["phase"]):
        if type(phase) is not str or not phase:
            msg = f"Krotov trace phase {index} must be a nonempty string."
            raise TypeError(msg)
    expected_phases = ["init", *(["batch"] * (len(columns["phase"]) - 1))]
    if columns["phase"] != expected_phases:
        msg = f"Krotov trace phases must be {expected_phases}, got {columns['phase']}."
        raise ValueError(msg)

    for name in ("step_size", "gradient_norm", "update_norm"):
        for index, value in enumerate(columns[name]):
            columns[name][index] = _require_finite_float(
                value,
                f"Krotov trace {name}[{index}]",
                minimum=0.0,
            )
        if not math.isclose(cast("float", columns[name][0]), 0.0, rel_tol=0.0, abs_tol=0.0):
            msg = f"Krotov trace {name}[0] must be zero for the initial point."
            raise ValueError(msg)

    for index, (loss, fidelity) in enumerate(
        zip(columns["loss"], columns["fidelity"], strict=True),
    ):
        validated_loss = _validated_fidelity(loss, f"Krotov trace loss[{index}]")
        validated_fidelity = _validated_fidelity(fidelity, f"Krotov trace fidelity[{index}]")
        if not math.isclose(validated_loss + validated_fidelity, 1.0, rel_tol=0.0, abs_tol=1e-10):
            msg = f"Krotov trace loss and fidelity at step {index} must sum to one."
            raise ValueError(msg)
        columns["loss"][index] = validated_loss
        columns["fidelity"][index] = validated_fidelity
    return columns


def _validated_result(result: object, *, expected_num_parameters: int | None = None) -> KrotovResult:
    """Validate and detach a Krotov optimization result.

    Returns:
        A detached, canonical result.

    Raises:
        TypeError: If the result or one of its fields has the wrong type.
        ValueError: If its parameters, bias, or trace are invalid.
    """
    if not isinstance(result, KrotovResult):
        msg = f"result must be a KrotovResult, got {type(result).__name__}."
        raise TypeError(msg)
    theta = np.asarray(result.theta)
    if theta.ndim != 1:
        msg = f"Krotov result theta must be one-dimensional, got shape {theta.shape}."
        raise ValueError(msg)
    parameter_count = len(theta) if expected_num_parameters is None else expected_num_parameters
    validated_theta = _validated_parameter_vector(theta, parameter_count, "Krotov result theta")
    bias = _require_finite_float(result.bias, "Krotov result bias")
    if not math.isclose(bias, 0.0, rel_tol=0.0, abs_tol=0.0):
        msg = "State-preparation Krotov results must have zero bias."
        raise ValueError(msg)
    return KrotovResult(theta=validated_theta, bias=0.0, trace=_validated_trace(result.trace))


def _parameter_layout_checksum(circuit: ParameterizedCircuit) -> str:
    """Return a stable digest of the data-free circuit and parameter layout.

    Returns:
        A prefixed SHA-256 checksum.

    Raises:
        ValueError: If a gate contains a data map, whose callable semantics
            cannot be serialized portably.
    """
    gates: list[dict[str, object]] = []
    for gate_index, gate in enumerate(circuit.gates):
        if gate.data_map is not None:
            msg = (
                f"Gate {gate_index} contains a data_map; state-preparation checkpoints support only data-free circuits."
            )
            raise ValueError(msg)
        gates.append({
            "name": gate.name,
            "sites": list(gate.sites),
            "param_index": gate.param_index,
            "angle_scale": _require_finite_float(gate.angle_scale, f"gate {gate_index} angle_scale"),
            "angle_offset": _require_finite_float(gate.angle_offset, f"gate {gate_index} angle_offset"),
            "fixed_params": [
                _require_finite_float(value, f"gate {gate_index} fixed_params[{parameter_index}]")
                for parameter_index, value in enumerate(gate.fixed_params)
            ],
        })
    payload = {
        "format": KROTOV_PARAMETER_LAYOUT_FORMAT,
        "num_qubits": circuit.num_qubits,
        "num_parameters": circuit.num_params,
        "gates": gates,
    }
    digest = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
    return f"sha256:{digest}"


def checkpoint_checksum(payload: bytes) -> str:
    """Return the benchmark checksum of serialized checkpoint bytes.

    Returns:
        ``sha256:`` followed by the lowercase hexadecimal digest.

    Raises:
        TypeError: If ``payload`` is not exact bytes.
    """
    if type(payload) is not bytes:
        msg = f"payload must be bytes, got {type(payload).__name__}."
        raise TypeError(msg)
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _verify_checkpoint_checksum(payload: bytes, expected_checksum: str) -> None:
    """Verify raw checkpoint bytes before parsing them.

    Raises:
        ValueError: If the expected checksum is malformed or does not match.
    """
    actual = checkpoint_checksum(payload)
    if (
        type(expected_checksum) is not str
        or len(expected_checksum) != len(actual)
        or not expected_checksum.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in expected_checksum.removeprefix("sha256:"))
    ):
        msg = "expected_checksum must be 'sha256:' followed by 64 lowercase hexadecimal characters."
        raise ValueError(msg)
    if actual != expected_checksum:
        msg = f"Checkpoint checksum mismatch: expected {expected_checksum}, computed {actual}."
        raise ValueError(msg)


def _checkpoint_text(archive: Mapping[str, NDArray[np.generic]], name: str) -> str:
    """Read one UTF-8 text field from a checkpoint archive.

    Returns:
        The decoded string.

    Raises:
        ValueError: If the field is not an exact byte vector containing UTF-8.
    """
    value = archive[name]
    if value.ndim != 1 or value.dtype != np.dtype("uint8"):
        msg = f"Checkpoint field {name!r} must be a one-dimensional uint8 array."
        raise ValueError(msg)
    try:
        return value.tobytes().decode("utf-8")
    except UnicodeDecodeError as error:
        msg = f"Checkpoint field {name!r} must contain valid UTF-8."
        raise ValueError(msg) from error


def _checkpoint_scalar_int(archive: Mapping[str, NDArray[np.generic]], name: str) -> int:
    """Read one exact integer scalar from a checkpoint archive.

    Returns:
        The scalar integer.

    Raises:
        ValueError: If the field is not an exact integer scalar.
    """
    value = archive[name]
    if value.shape != () or value.dtype.str != "<i8":
        msg = f"Checkpoint field {name!r} must be a little-endian int64 scalar."
        raise ValueError(msg)
    return int(value.item())


def _checkpoint_field_size_limits(expected_num_parameters: int) -> dict[str, int]:
    """Return raw NPY-member size limits for one checkpoint.

    Returns:
        A limit for every versioned checkpoint field.
    """
    return {
        **dict.fromkeys(_CHECKPOINT_TEXT_FIELDS | _CHECKPOINT_SCALAR_FIELDS, _NPY_HEADER_ALLOWANCE),
        "theta": expected_num_parameters * np.dtype("<f8").itemsize + _NPY_HEADER_ALLOWANCE,
    }


def _maximum_checkpoint_archive_size(expected_num_parameters: int) -> int:
    """Return the largest accepted raw checkpoint archive size.

    Returns:
        The byte limit including ZIP bookkeeping allowance.
    """
    return sum(_checkpoint_field_size_limits(expected_num_parameters).values()) + _NPY_HEADER_ALLOWANCE


def _maximum_legacy_payload_size(expected_num_parameters: int) -> int:
    """Return the largest accepted raw legacy NPY size.

    Returns:
        A conservative byte limit supporting ordinary extended numeric dtypes.
    """
    return expected_num_parameters * 32 + 65536


def _decode_npy_payload(
    payload: bytes,
    name: str,
    *,
    expected_shape: tuple[int, ...] | None = None,
    expected_dtype: np.dtype[np.generic] | None = None,
    max_elements: int | None = None,
) -> NDArray[np.generic]:
    """Decode an NPY array only after validating its header and exact byte size.

    Returns:
        A detached array.

    Raises:
        ValueError: If the NPY version, header, shape, dtype, storage order, or
            payload length is invalid.
    """
    buffer = io.BytesIO(payload)
    try:
        version = np.lib.format.read_magic(buffer)
    except (EOFError, TypeError, ValueError) as error:
        msg = f"{name} has an invalid NPY magic prefix."
        raise ValueError(msg) from error
    if version not in {(1, 0), (2, 0)}:
        msg = f"{name} uses unsupported NPY format version {version}."
        raise ValueError(msg)

    try:
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                buffer,
                max_header_size=_NPY_HEADER_ALLOWANCE,
            )
        else:
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                buffer,
                max_header_size=_NPY_HEADER_ALLOWANCE,
            )
    except (EOFError, TypeError, ValueError) as error:
        msg = f"{name} has an invalid NPY header."
        raise ValueError(msg) from error

    resolved_dtype = np.dtype(dtype)
    if resolved_dtype.hasobject:
        msg = f"{name} cannot contain object data."
        raise ValueError(msg)
    if fortran_order:
        msg = f"{name} must use C-order storage."
        raise ValueError(msg)
    if expected_shape is not None and shape != expected_shape:
        msg = f"{name} must have shape {expected_shape}, got {shape}."
        raise ValueError(msg)
    element_count = math.prod(shape)
    if max_elements is not None and element_count > max_elements:
        msg = f"{name} declares too many elements."
        raise ValueError(msg)
    if expected_dtype is not None and resolved_dtype != expected_dtype:
        msg = f"{name} must use dtype {expected_dtype.str}, got {resolved_dtype.str}."
        raise ValueError(msg)
    data_offset = buffer.tell()
    expected_payload_size = data_offset + element_count * resolved_dtype.itemsize
    if len(payload) != expected_payload_size:
        msg = f"{name} byte length does not match its NPY header."
        raise ValueError(msg)
    return (
        np
        .frombuffer(
            payload,
            dtype=resolved_dtype,
            count=element_count,
            offset=data_offset,
        )
        .reshape(shape)
        .copy()
    )


def _read_checkpoint_member_payloads(
    payload: bytes,
    field_limits: Mapping[str, int],
) -> dict[str, bytes]:
    """Validate the ZIP envelope and return bounded member bytes.

    Returns:
        Raw NPY payloads keyed by checkpoint field.

    Raises:
        ValueError: If members are duplicated, missing, compressed, encrypted,
            or oversized.
    """
    with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
        members = zipped.infolist()
        member_names = [member.filename for member in members]
        if len(member_names) != len(set(member_names)):
            msg = "Checkpoint archive contains duplicate ZIP members."
            raise ValueError(msg)
        if frozenset(member_names) != _CHECKPOINT_ARCHIVE_MEMBERS:
            msg = "Checkpoint archive members do not match the versioned format."
            raise ValueError(msg)
        for member in members:
            field_name = member.filename.removesuffix(".npy")
            if member.flag_bits & 0x1 or member.compress_type != zipfile.ZIP_STORED:
                msg = "Checkpoint archive members must be unencrypted and uncompressed."
                raise ValueError(msg)
            if member.file_size > field_limits[field_name]:
                msg = f"Checkpoint field {field_name!r} exceeds its allowed size."
                raise ValueError(msg)
        return {name: zipped.read(f"{name}.npy") for name in _CHECKPOINT_FIELDS}


def _read_npz_archive(
    payload: bytes,
    *,
    expected_num_parameters: int,
) -> dict[str, NDArray[np.generic]]:
    """Read an exact non-pickled checkpoint archive.

    Returns:
        Detached arrays keyed by checkpoint field.

    Raises:
        ValueError: If the ZIP or NPZ structure is invalid.
    """
    field_limits = _checkpoint_field_size_limits(expected_num_parameters)
    if len(payload) > _maximum_checkpoint_archive_size(expected_num_parameters):
        msg = "Checkpoint archive exceeds the size allowed by its circuit parameter count."
        raise ValueError(msg)

    try:
        member_payloads = _read_checkpoint_member_payloads(payload, field_limits)
    except (EOFError, KeyError, OSError, RuntimeError, zipfile.BadZipFile) as error:
        msg = "Checkpoint NPZ arrays could not be decoded safely."
        raise ValueError(msg) from error

    arrays: dict[str, NDArray[np.generic]] = {}
    for name in _CHECKPOINT_TEXT_FIELDS:
        arrays[name] = _decode_npy_payload(
            member_payloads[name],
            f"Checkpoint field {name!r}",
            expected_dtype=np.dtype("uint8"),
            max_elements=_NPY_HEADER_ALLOWANCE,
        )
        if arrays[name].ndim != 1:
            msg = f"Checkpoint field {name!r} must be one-dimensional."
            raise ValueError(msg)
    for name in _CHECKPOINT_SCALAR_FIELDS:
        arrays[name] = _decode_npy_payload(
            member_payloads[name],
            f"Checkpoint field {name!r}",
            expected_shape=(),
            expected_dtype=np.dtype("<i8"),
        )
    arrays["theta"] = _decode_npy_payload(
        member_payloads["theta"],
        "Checkpoint field 'theta'",
        expected_shape=(expected_num_parameters,),
        expected_dtype=np.dtype("<f8"),
    )
    return arrays


def _load_legacy_parameter_array(payload: bytes, expected_count: int) -> NDArray[np.float64]:
    """Load a checksum-verified legacy NPY warm-start vector.

    Returns:
        The detached parameter vector.

    Raises:
        ValueError: If the payload cannot be decoded safely or the vector is
            invalid.
    """
    if len(payload) > _maximum_legacy_payload_size(expected_count):
        msg = "Legacy warm-start payload exceeds the size allowed by its parameter count."
        raise ValueError(msg)
    try:
        loaded = _decode_npy_payload(
            payload,
            "Legacy warm-start payload",
            expected_shape=(expected_count,),
        )
    except (EOFError, OSError, TypeError, ValueError) as error:
        msg = "Warm-start payload is neither a valid Krotov NPZ checkpoint nor a safe NPY array."
        raise ValueError(msg) from error
    return _validated_parameter_vector(loaded, expected_count, "warm-start parameters")


def state_preparation_training_identity(
    method: StatePreparationMethod[_ResultT],
    config: BenchmarkConfig,
) -> dict[str, object]:
    """Return the canonical identity payload for one reusable training run.

    Test-noise settings, evaluation budgets, confidence intervals, and output
    paths are deliberately excluded. Warm-start content is identified by
    checksum rather than path spelling.

    Returns:
        A detached JSON-native training identity.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the adapter and configuration identities differ.
    """
    if not isinstance(method, StatePreparationMethod):
        msg = f"method must implement StatePreparationMethod, got {type(method).__name__}."
        raise TypeError(msg)
    if not isinstance(config, BenchmarkConfig):
        msg = f"config must be a BenchmarkConfig, got {type(config).__name__}."
        raise TypeError(msg)
    if config.method_id != method.method_id or config.method_version != method.method_version:
        msg = "Benchmark configuration method identity does not match the adapter."
        raise ValueError(msg)
    return {
        "identity_version": TRAINING_IDENTITY_VERSION,
        "method": {
            "method_id": method.method_id,
            "method_version": method.method_version,
        },
        "target": config.target.to_dict(),
        "ansatz": config.ansatz.to_dict(),
        "initialization": config.initialization.identity_dict(),
        "optimizer": config.optimizer.to_dict(),
        "training_noise": config.training_noise.to_dict(),
    }


def state_preparation_training_id(
    method: StatePreparationMethod[_ResultT],
    config: BenchmarkConfig,
) -> str:
    """Return a stable identifier shared by all evaluations of one training.

    Returns:
        The prefixed SHA-256 training identifier.
    """
    payload = state_preparation_training_identity(method, config)
    digest = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
    return f"{TRAINING_ID_PREFIX}{digest}"


class StatePreparationTrainingError(RuntimeError):
    """Failure from a classified stage of method-generic training."""

    failure_phase: _TrainingFailurePhase
    exception: Exception

    def __init__(self, failure_phase: _TrainingFailurePhase, exception: Exception) -> None:
        """Preserve an adapter exception together with its reporting phase."""
        message = str(exception) or type(exception).__name__
        super().__init__(f"{failure_phase} failed with {type(exception).__name__}: {message}")
        self.failure_phase = failure_phase
        self.exception = exception


@dataclass(frozen=True, slots=True, init=False)
class StatePreparationTrainingArtifact:
    """Detached output of one reusable state-preparation training run.

    The circuit and optimizer metadata are exposed as defensive copies.
    Parameters are backed by immutable bytes, so every evaluation observes the
    exact vector produced by the single optimization run.
    """

    training_id: str
    method_id: str
    method_name: str
    method_version: str
    training_fidelity: float
    checkpoint_payload: bytes = field(repr=False)
    checkpoint_checksum: str
    _circuit: ParameterizedCircuit = field(repr=False, compare=False)
    _parameter_bytes: bytes = field(repr=False)
    _optimizer_metadata_json: bytes = field(repr=False)

    def __init__(
        self,
        *,
        training_id: str,
        method_id: str,
        method_name: str,
        method_version: str,
        circuit: ParameterizedCircuit,
        parameters: NDArray[np.float64],
        training_fidelity: float,
        optimizer_metadata: Mapping[str, object],
        checkpoint_payload: bytes,
        _validation_token: object | None = None,
    ) -> None:
        """Validate and detach one factory-created trained artifact.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If an identity, fidelity, parameter vector, metadata
                document, or checkpoint is invalid.
        """
        if _validation_token is not _ARTIFACT_VALIDATION_TOKEN:
            msg = "StatePreparationTrainingArtifact instances must be created by train_state_preparation_method."
            raise ValueError(msg)
        expected_training_id_length = len(TRAINING_ID_PREFIX) + 64
        if (
            type(training_id) is not str
            or len(training_id) != expected_training_id_length
            or not training_id.startswith(TRAINING_ID_PREFIX)
            or any(character not in "0123456789abcdef" for character in training_id[len(TRAINING_ID_PREFIX) :])
        ):
            msg = f"training_id must be {TRAINING_ID_PREFIX!r} followed by 64 lowercase hexadecimal characters."
            raise ValueError(msg)
        for name, value in (
            ("method_id", method_id),
            ("method_name", method_name),
            ("method_version", method_version),
        ):
            if type(value) is not str or not value.strip():
                msg = f"{name} must be a nonempty string."
                raise ValueError(msg)
        if not isinstance(circuit, ParameterizedCircuit):
            msg = f"circuit must be a ParameterizedCircuit, got {type(circuit).__name__}."
            raise TypeError(msg)
        theta = _validated_parameter_vector(parameters, circuit.num_params, "trained parameters")
        fidelity = _validated_fidelity(training_fidelity, "training_fidelity")
        if not isinstance(optimizer_metadata, Mapping) or any(type(key) is not str for key in optimizer_metadata):
            msg = "optimizer_metadata must be a string-keyed mapping."
            raise TypeError(msg)
        metadata_json = _canonical_json(dict(optimizer_metadata)).encode("utf-8")
        if type(checkpoint_payload) is not bytes:
            msg = f"checkpoint_payload must be bytes, got {type(checkpoint_payload).__name__}."
            raise TypeError(msg)

        object.__setattr__(self, "training_id", training_id)
        object.__setattr__(self, "method_id", method_id)
        object.__setattr__(self, "method_name", method_name)
        object.__setattr__(self, "method_version", method_version)
        object.__setattr__(self, "training_fidelity", fidelity)
        object.__setattr__(self, "checkpoint_payload", checkpoint_payload)
        object.__setattr__(self, "checkpoint_checksum", checkpoint_checksum(checkpoint_payload))
        object.__setattr__(self, "_circuit", copy.deepcopy(circuit))
        object.__setattr__(self, "_parameter_bytes", theta.tobytes(order="C"))
        object.__setattr__(self, "_optimizer_metadata_json", metadata_json)

    @property
    def circuit(self) -> ParameterizedCircuit:
        """Defensive copy of the trained logical circuit."""
        return copy.deepcopy(self._circuit)

    @property
    def parameters(self) -> NDArray[np.float64]:
        """Read-only view of the exact trained parameter bytes."""
        return np.frombuffer(self._parameter_bytes, dtype=np.dtype("<f8"))

    def parameters_copy(self) -> NDArray[np.float64]:
        """Return a mutable copy of the trained parameter vector."""
        return self.parameters.copy()

    @property
    def optimizer_metadata(self) -> dict[str, object]:
        """Detached JSON-native optimizer metadata."""
        return cast("dict[str, object]", json.loads(self._optimizer_metadata_json))


def _load_selected_target(targets: TargetCollection, selection: TargetSelection) -> TargetRecord:
    """Resolve a target from the exact fixture bound into the configuration.

    Returns:
        The validated selected target record.

    Raises:
        TypeError: If ``targets`` is not a validated target collection.
        ValueError: If fixture provenance or the selected record differs from
            the resolved benchmark selection.
    """
    if not isinstance(targets, TargetCollection):
        msg = f"targets must be a TargetCollection, got {type(targets).__name__}."
        raise TypeError(msg)
    if targets.fixture_format != selection.fixture_format or targets.fixture_checksum != selection.fixture_checksum:
        msg = "TargetCollection provenance does not match the benchmark target selection."
        raise ValueError(msg)
    target = targets.load_target(selection.num_qubits, selection.target_id)
    if (
        target.num_qubits != selection.num_qubits
        or target.target_id != selection.target_id
        or target.seed != selection.target_seed
    ):
        msg = "TargetRecord does not match the benchmark target selection."
        raise ValueError(msg)
    return target


def _method_identity(method: StatePreparationMethod[_ResultT]) -> tuple[str, str, str]:
    """Return one validated adapter-identity snapshot.

    Returns:
        The method identifier, display name, and version.

    Raises:
        ValueError: If an identity field is empty or has the wrong type.
    """
    identity = (method.method_id, method.method_name, method.method_version)
    for name, value in zip(("method_id", "method_name", "method_version"), identity, strict=True):
        if type(value) is not str or not value.strip():
            msg = f"{name} must be a nonempty string."
            raise ValueError(msg)
    return identity


def _require_unchanged_method_identity(
    method: StatePreparationMethod[_ResultT],
    expected: tuple[str, str, str],
) -> None:
    """Reject an adapter that changes identity during a training run.

    Raises:
        ValueError: If the current identity differs from ``expected``.
    """
    if _method_identity(method) != expected:
        msg = "Method identity changed during state-preparation training."
        raise ValueError(msg)


def _run_training_stage(
    failure_phase: _TrainingFailurePhase,
    operation: Callable[[], _StageT],
) -> _StageT:
    """Run one operation and classify ordinary adapter exceptions.

    Returns:
        The operation result.

    Raises:
        StatePreparationTrainingError: If ``operation`` raises an exception.
    """
    try:
        return operation()
    except Exception as error:
        raise StatePreparationTrainingError(failure_phase, error) from error


def _training_identity_stage(
    method: StatePreparationMethod[_ResultT],
    config: BenchmarkConfig,
) -> tuple[tuple[str, str, str], str]:
    """Resolve one immutable method snapshot and canonical training ID.

    Returns:
        The identity snapshot and training ID.
    """
    identity = _method_identity(method)
    return identity, state_preparation_training_id(method, config)


def _build_ansatz_stage(
    method: StatePreparationMethod[_ResultT],
    config: BenchmarkConfig,
    identity: tuple[str, str, str],
) -> ParameterizedCircuit:
    """Build and validate the generic logical ansatz.

    Returns:
        The adapter's parameterized circuit.

    Raises:
        TypeError: If the adapter returns the wrong circuit type.
    """
    circuit = method.build_ansatz(config.target.num_qubits, config.ansatz)
    if not isinstance(circuit, ParameterizedCircuit):
        msg = f"build_ansatz must return a ParameterizedCircuit, got {type(circuit).__name__}."
        raise TypeError(msg)
    _require_unchanged_method_identity(method, identity)
    return circuit


def _initialize_training_stage(
    method: StatePreparationMethod[_ResultT],
    circuit: ParameterizedCircuit,
    config: BenchmarkConfig,
    identity: tuple[str, str, str],
    checkpoint_root: Path | None,
) -> NDArray[np.float64]:
    """Initialize and validate the exact optimizer input vector.

    Returns:
        A detached canonical parameter vector.
    """
    parameters = method.initialize_parameters(
        circuit,
        config.initialization,
        checkpoint_root=checkpoint_root,
    )
    initial = _validated_parameter_vector(parameters, circuit.num_params, "initial parameters")
    _require_unchanged_method_identity(method, identity)
    return initial


def _optimize_training_stage(
    method: StatePreparationMethod[_ResultT],
    circuit: ParameterizedCircuit,
    target: TargetRecord,
    initial: NDArray[np.float64],
    config: BenchmarkConfig,
    identity: tuple[str, str, str],
) -> tuple[_ResultT, NDArray[np.float64], float, Mapping[str, object]]:
    """Optimize once and validate every reusable result field.

    Returns:
        The opaque result, trained parameters, fidelity, and optimizer
        metadata.

    Raises:
        TypeError: If optimizer metadata is not a string-keyed mapping.
    """
    optimizer_metadata = method.optimizer_metadata(config.optimizer)
    if not isinstance(optimizer_metadata, Mapping) or any(type(key) is not str for key in optimizer_metadata):
        msg = "optimizer_metadata must be a string-keyed mapping."
        raise TypeError(msg)
    _canonical_json(dict(optimizer_metadata))
    result = method.optimize_noiseless(
        circuit,
        target,
        initial,
        config.optimizer,
    )
    parameters = method.extract_final_parameters(result)
    trained = _validated_parameter_vector(parameters, circuit.num_params, "trained parameters")
    training_fidelity = _validated_fidelity(
        method.extract_training_fidelity(result),
        "training_fidelity",
    )
    _require_unchanged_method_identity(method, identity)
    return result, trained, training_fidelity, optimizer_metadata


def _checkpoint_training_stage(
    method: StatePreparationMethod[_ResultT],
    circuit: ParameterizedCircuit,
    result: _ResultT,
    trained: NDArray[np.float64],
    training_fidelity: float,
    optimizer_metadata: Mapping[str, object],
    identity: tuple[str, str, str],
    training_id: str,
) -> StatePreparationTrainingArtifact:
    """Round-trip the checkpoint and construct the detached artifact.

    Returns:
        The validated reusable training artifact.

    Raises:
        ValueError: If the checkpoint changes the trained parameters.
    """
    checkpoint_payload = method.serialize_checkpoint(circuit, result)
    restored_parameters = method.deserialize_checkpoint(
        circuit,
        checkpoint_payload,
        expected_checksum=checkpoint_checksum(checkpoint_payload),
    )
    restored = _validated_parameter_vector(
        restored_parameters,
        circuit.num_params,
        "checkpoint round-trip parameters",
    )
    if restored.tobytes(order="C") != trained.tobytes(order="C"):
        msg = "Method checkpoint round-trip changed the trained parameters."
        raise ValueError(msg)
    _require_unchanged_method_identity(method, identity)
    return StatePreparationTrainingArtifact(
        training_id=training_id,
        method_id=identity[0],
        method_name=identity[1],
        method_version=identity[2],
        circuit=circuit,
        parameters=trained,
        training_fidelity=training_fidelity,
        optimizer_metadata=optimizer_metadata,
        checkpoint_payload=checkpoint_payload,
        _validation_token=_ARTIFACT_VALIDATION_TOKEN,
    )


def train_state_preparation_method(
    method: StatePreparationMethod[_ResultT],
    config: BenchmarkConfig,
    targets: TargetCollection,
    *,
    checkpoint_root: Path | None = None,
) -> StatePreparationTrainingArtifact:
    """Train one adapter exactly once and detach its reusable artifact.

    Test noise and evaluation settings participate in neither the optimization
    nor the artifact identity. Callers may therefore fan this artifact out to
    every test configuration sharing its ``training_id``. Stage failures are
    surfaced as :class:`StatePreparationTrainingError`.

    Returns:
        The immutable-parameter training artifact.
    """
    identity, training_id = _run_training_stage(
        "optimization",
        lambda: _training_identity_stage(method, config),
    )
    target = _run_training_stage(
        "target_loading",
        lambda: _load_selected_target(targets, config.target),
    )
    circuit = _run_training_stage(
        "ansatz",
        lambda: _build_ansatz_stage(method, config, identity),
    )
    initial = _run_training_stage(
        "initialization",
        lambda: _initialize_training_stage(method, circuit, config, identity, checkpoint_root),
    )
    result, trained, training_fidelity, optimizer_metadata = _run_training_stage(
        "optimization",
        lambda: _optimize_training_stage(method, circuit, target, initial, config, identity),
    )
    return _run_training_stage(
        "checkpoint",
        lambda: _checkpoint_training_stage(
            method,
            circuit,
            result,
            trained,
            training_fidelity,
            optimizer_metadata,
            identity,
            training_id,
        ),
    )


class KrotovStatePreparationMethod:
    """Adapter exposing noiseless full-batch YAQS Krotov state preparation."""

    method_id = KROTOV_METHOD_ID
    method_name = KROTOV_METHOD_NAME
    method_version = KROTOV_METHOD_VERSION

    def _validate_method_identity(self) -> None:
        """Reject subclasses that silently change checkpoint semantics.

        Raises:
            ValueError: If the adapter identity differs from the frozen v1
                implementation.
        """
        if (
            self.method_id != KROTOV_METHOD_ID
            or self.method_name != KROTOV_METHOD_NAME
            or self.method_version != KROTOV_METHOD_VERSION
        ):
            msg = "KrotovStatePreparationMethod identity is immutable."
            raise ValueError(msg)

    def build_ansatz(self, num_qubits: int, ansatz: AnsatzConfig) -> ParameterizedCircuit:
        """Construct the shared scalar-parameter BMPD ansatz.

        Returns:
            The logical parameterized circuit.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If the qubit count or adapter identity is invalid.
        """
        self._validate_method_identity()
        if isinstance(num_qubits, (bool, np.bool_)) or not isinstance(num_qubits, Integral):
            msg = f"num_qubits must be an integer, got {type(num_qubits).__name__}."
            raise TypeError(msg)
        if num_qubits < 1:
            msg = "num_qubits must be at least 1."
            raise ValueError(msg)
        if not isinstance(ansatz, AnsatzConfig):
            msg = f"ansatz must be an AnsatzConfig, got {type(ansatz).__name__}."
            raise TypeError(msg)
        return create_brickwall_matrix_product_disentangler_parameterized_circuit(
            int(num_qubits),
            ansatz.configured_bmpd_depth,
            initial_single_qubit_layer=ansatz.initial_single_qubit_layer,
        )

    def initialize_parameters(
        self,
        circuit: ParameterizedCircuit,
        initialization: InitializationConfig,
        *,
        checkpoint_root: Path | None = None,
    ) -> NDArray[np.float64]:
        """Initialize parameters without using NumPy's global random state.

        Uniform initialization samples ``[-scale, scale)``. Normal
        initialization uses mean zero and standard deviation ``scale``.
        Warm starts accept this adapter's versioned NPZ checkpoints and
        checksum-verified legacy numeric NPY vectors.

        Returns:
            The detached ``float64`` initial parameter vector.

        Raises:
            TypeError: If an argument or decoded array has the wrong type.
            ValueError: If initialization, checksum, checkpoint, or vector
                validation fails.
        """
        self._validate_method_identity()
        if not isinstance(circuit, ParameterizedCircuit):
            msg = f"circuit must be a ParameterizedCircuit, got {type(circuit).__name__}."
            raise TypeError(msg)
        if not isinstance(initialization, InitializationConfig):
            msg = f"initialization must be an InitializationConfig, got {type(initialization).__name__}."
            raise TypeError(msg)
        if checkpoint_root is not None and not isinstance(checkpoint_root, Path):
            msg = f"checkpoint_root must be a pathlib.Path or None, got {type(checkpoint_root).__name__}."
            raise TypeError(msg)

        if initialization.rule == "zeros":
            return np.zeros(circuit.num_params, dtype=np.float64)
        if initialization.rule in {"random_uniform", "random_normal"}:
            seed = cast("int", initialization.seed)
            scale = cast("float", initialization.scale)
            rng = np.random.Generator(np.random.PCG64(seed))
            if initialization.rule == "random_uniform":
                values = rng.uniform(-scale, scale, size=circuit.num_params)
            else:
                values = rng.normal(0.0, scale, size=circuit.num_params)
            return _validated_parameter_vector(values, circuit.num_params, "initial parameters")

        relative_path = cast("str", initialization.warm_start_path)
        root = Path.cwd() if checkpoint_root is None else checkpoint_root
        try:
            resolved_root = root.resolve()
            path = (resolved_root / relative_path).resolve()
        except (OSError, RuntimeError) as error:
            msg = f"Could not resolve warm-start checkpoint {relative_path!r}: {error}."
            raise ValueError(msg) from error
        if not path.is_relative_to(resolved_root):
            msg = f"Warm-start checkpoint {relative_path!r} resolves outside checkpoint_root."
            raise ValueError(msg)
        maximum_payload_size = max(
            _maximum_checkpoint_archive_size(circuit.num_params),
            _maximum_legacy_payload_size(circuit.num_params),
        )
        try:
            with path.open("rb") as checkpoint_file:
                payload = checkpoint_file.read(maximum_payload_size + 1)
        except OSError as error:
            msg = f"Could not read warm-start checkpoint {path}: {error}."
            raise ValueError(msg) from error
        if len(payload) > maximum_payload_size:
            msg = "Warm-start checkpoint exceeds the size allowed by its parameter count."
            raise ValueError(msg)
        expected_checksum = cast("str", initialization.warm_start_checksum)
        _verify_checkpoint_checksum(payload, expected_checksum)
        if zipfile.is_zipfile(io.BytesIO(payload)):
            return self.deserialize_checkpoint(circuit, payload, expected_checksum=expected_checksum)
        return _load_legacy_parameter_array(payload, circuit.num_params)

    def optimize_noiseless(
        self,
        circuit: ParameterizedCircuit,
        target: StatePreparationTarget,
        initial_parameters: NDArray[np.float64],
        optimizer: OptimizerConfig,
    ) -> KrotovResult:
        """Run the configured noiseless full-batch Krotov optimization.

        Returns:
            The validated Krotov result.

        Raises:
            TypeError: If an argument has the wrong type.
        """
        self._validate_method_identity()
        if not isinstance(circuit, ParameterizedCircuit):
            msg = f"circuit must be a ParameterizedCircuit, got {type(circuit).__name__}."
            raise TypeError(msg)
        initial = _validated_parameter_vector(initial_parameters, circuit.num_params, "initial_parameters")
        result = train_krotov_state_preparation_batch(
            circuit,
            _target_state(target, circuit.num_qubits),
            initial_theta=initial,
            options=_krotov_options(optimizer),
        )
        return _validated_result(result, expected_num_parameters=circuit.num_params)

    def extract_final_parameters(self, result: KrotovResult) -> NDArray[np.float64]:
        """Extract a detached final parameter vector.

        Returns:
            A fresh one-dimensional ``float64`` array.
        """
        self._validate_method_identity()
        return _validated_result(result).theta.copy()

    def final_parameters(self, result: KrotovResult) -> NDArray[np.float64]:
        """Alias for :meth:`extract_final_parameters`.

        Returns:
            A fresh one-dimensional ``float64`` array.
        """
        return self.extract_final_parameters(result)

    def extract_training_fidelity(self, result: KrotovResult) -> float:
        """Extract the final fidelity recorded by Krotov.

        Returns:
            The final training fidelity.
        """
        self._validate_method_identity()
        validated = _validated_result(result)
        return cast("float", validated.trace["fidelity"][-1])

    def training_fidelity(self, result: KrotovResult) -> float:
        """Alias for :meth:`extract_training_fidelity`.

        Returns:
            The final training fidelity.
        """
        return self.extract_training_fidelity(result)

    def evaluate_noiseless(
        self,
        circuit: ParameterizedCircuit,
        parameters: NDArray[np.float64],
        target: StatePreparationTarget,
        *,
        evaluation: EvaluationConfig | None = None,
    ) -> float:
        """Evaluate target fidelity without noise.

        Returns:
            The noiseless fidelity under the independent evaluation truncation.

        Raises:
            TypeError: If an argument has the wrong type.
        """
        self._validate_method_identity()
        if not isinstance(circuit, ParameterizedCircuit):
            msg = f"circuit must be a ParameterizedCircuit, got {type(circuit).__name__}."
            raise TypeError(msg)
        theta = _validated_parameter_vector(parameters, circuit.num_params, "parameters")
        _loss, fidelity = state_preparation_metrics(
            circuit,
            theta,
            _target_state(target, circuit.num_qubits),
            truncation=_evaluation_truncation(evaluation),
        )
        return _validated_fidelity(fidelity, "noiseless fidelity")

    def optimizer_metadata(self, optimizer: OptimizerConfig) -> dict[str, object]:
        """Return complete input and resolved Krotov optimizer metadata.

        Returns:
            A detached JSON-native mapping.
        """
        self._validate_method_identity()
        options = _krotov_options(optimizer)
        return {
            "implementation": _KROTOV_IMPLEMENTATION,
            "method_id": self.method_id,
            "method_version": self.method_version,
            "optimizer_config": optimizer.to_dict(),
            "resolved_options": {
                "variant": "batch",
                "max_iterations": options.max_iterations,
                "switch_iteration": options.switch_iteration,
                "online_step_size": options.online_step_size,
                "batch_step_size": options.batch_step_size,
                "online_schedule": options.online_schedule,
                "batch_schedule": options.batch_schedule,
                "online_decay": options.online_decay,
                "batch_decay": options.batch_decay,
                "seed": options.seed,
                "truncation": {
                    "max_bond_dim": options.truncation.max_bond_dim,
                    "svd_threshold": options.truncation.svd_threshold,
                    "trunc_mode": options.truncation.trunc_mode,
                    "min_bond_dim": options.truncation.min_bond_dim,
                },
            },
        }

    def serialize_checkpoint(self, circuit: ParameterizedCircuit, result: KrotovResult) -> bytes:
        """Serialize a deterministic versioned parameter checkpoint.

        Returns:
            Raw NPZ bytes. Reporting code owns filesystem writes.

        Raises:
            TypeError: If an argument has the wrong type.
        """
        self._validate_method_identity()
        if not isinstance(circuit, ParameterizedCircuit):
            msg = f"circuit must be a ParameterizedCircuit, got {type(circuit).__name__}."
            raise TypeError(msg)
        validated = _validated_result(result, expected_num_parameters=circuit.num_params)
        parameter_layout_checksum = _parameter_layout_checksum(circuit)
        buffer = io.BytesIO()
        np.savez(
            buffer,
            checkpoint_format=np.frombuffer(KROTOV_CHECKPOINT_FORMAT.encode(), dtype=np.uint8),
            method_id=np.frombuffer(self.method_id.encode(), dtype=np.uint8),
            method_version=np.frombuffer(self.method_version.encode(), dtype=np.uint8),
            parameter_layout_checksum=np.frombuffer(parameter_layout_checksum.encode(), dtype=np.uint8),
            num_qubits=np.asarray(circuit.num_qubits, dtype=np.dtype("<i8")),
            num_parameters=np.asarray(circuit.num_params, dtype=np.dtype("<i8")),
            theta=np.asarray(validated.theta, dtype=np.dtype("<f8")),
        )
        return buffer.getvalue()

    def deserialize_checkpoint(
        self,
        circuit: ParameterizedCircuit,
        payload: bytes,
        *,
        expected_checksum: str | None = None,
    ) -> NDArray[np.float64]:
        """Deserialize a strict Krotov parameter checkpoint.

        Returns:
            A detached validated parameter vector.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If checksum, archive, metadata, layout, or result
                validation fails.
        """
        self._validate_method_identity()
        if not isinstance(circuit, ParameterizedCircuit):
            msg = f"circuit must be a ParameterizedCircuit, got {type(circuit).__name__}."
            raise TypeError(msg)
        if type(payload) is not bytes:
            msg = f"payload must be bytes, got {type(payload).__name__}."
            raise TypeError(msg)
        if expected_checksum is not None:
            _verify_checkpoint_checksum(payload, expected_checksum)
        archive = _read_npz_archive(payload, expected_num_parameters=circuit.num_params)
        if _checkpoint_text(archive, "checkpoint_format") != KROTOV_CHECKPOINT_FORMAT:
            msg = "Unsupported Krotov checkpoint format."
            raise ValueError(msg)
        if _checkpoint_text(archive, "method_id") != self.method_id:
            msg = "Checkpoint method_id does not match this adapter."
            raise ValueError(msg)
        if _checkpoint_text(archive, "method_version") != self.method_version:
            msg = "Checkpoint method_version does not match this adapter."
            raise ValueError(msg)
        if _checkpoint_text(archive, "parameter_layout_checksum") != _parameter_layout_checksum(circuit):
            msg = "Checkpoint parameter layout does not match the circuit."
            raise ValueError(msg)
        if _checkpoint_scalar_int(archive, "num_qubits") != circuit.num_qubits:
            msg = "Checkpoint qubit count does not match the circuit."
            raise ValueError(msg)
        if _checkpoint_scalar_int(archive, "num_parameters") != circuit.num_params:
            msg = "Checkpoint parameter count does not match the circuit."
            raise ValueError(msg)

        theta_array = archive["theta"]
        if theta_array.dtype.str != "<f8":
            msg = "Checkpoint theta must use little-endian float64."
            raise ValueError(msg)
        return _validated_parameter_vector(theta_array, circuit.num_params, "checkpoint theta")

    def training_id(self, config: BenchmarkConfig) -> str:
        """Return the reusable training identifier for this method.

        Returns:
            A stable identifier independent of test noise and evaluation policy.
        """
        self._validate_method_identity()
        return state_preparation_training_id(self, config)


__all__ = [
    "KROTOV_CHECKPOINT_FORMAT",
    "KROTOV_METHOD_ID",
    "KROTOV_METHOD_NAME",
    "KROTOV_METHOD_VERSION",
    "KROTOV_PARAMETER_LAYOUT_FORMAT",
    "TRAINING_IDENTITY_VERSION",
    "TRAINING_ID_PREFIX",
    "KrotovStatePreparationMethod",
    "StatePreparationMethod",
    "StatePreparationTarget",
    "StatePreparationTrainingArtifact",
    "StatePreparationTrainingError",
    "checkpoint_checksum",
    "state_preparation_training_id",
    "state_preparation_training_identity",
    "train_state_preparation_method",
]
