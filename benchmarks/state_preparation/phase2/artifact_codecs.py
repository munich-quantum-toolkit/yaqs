# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Deterministic, bounded codecs for Phase II checkpoint artifacts."""

# Private helpers below are deliberately small validation/serialization
# primitives; their names and annotations document their narrow contracts.
# ruff: noqa: DOC201, DOC501

from __future__ import annotations

import hashlib
import io
import re
import struct
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast

import numpy as np

from .canonical import canonical_checksum, canonical_json, load_canonical_json_object, thaw_json
from .noisy_krotov import (
    KrotovWorkLedger,
    NoisyKrotovCheckpointSelection,
    NoisyKrotovResumeState,
    NoisyKrotovStageExecution,
)
from .validation import (
    require_checksum,
    require_exact_keys,
    require_float,
    require_int,
    require_mapping,
    require_slug,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray


STAGE_PARAMETER_CHECKPOINT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.stage_parameter_checkpoint.v1"
PHASE2_TRAJECTORY_SIDECAR_SCHEMA_VERSION = "yaqs.state_preparation.phase2.trajectory_fidelity_sidecar.v1"

MAX_STAGE_PARAMETER_COUNT: Final = 1_000_000
MAX_TRAJECTORY_FIDELITY_COUNT: Final = 1_000_000
PHASE2_MAP_ROLES: Final = (
    "training_trajectory",
    "checkpoint_validation",
    "pilot_evaluation",
    "screening_selection",
    "confirmatory_test",
)

_CHECKPOINT_MEMBERS = frozenset({"metadata.json", "selected_theta.npy", "final_theta.npy"})
_SIDECAR_MEMBERS = frozenset({"metadata.json", "fidelities.npy"})
_METADATA_LIMIT = 64 * 1024
_NPY_HEADER_LIMIT = 4096
_ZIP_OVERHEAD_LIMIT = 64 * 1024
_MAX_CHECKPOINT_ARCHIVE_SIZE = 2 * MAX_STAGE_PARAMETER_COUNT * 8 + 2 * _NPY_HEADER_LIMIT + _METADATA_LIMIT
_MAX_SIDECAR_ARCHIVE_SIZE = MAX_TRAJECTORY_FIDELITY_COUNT * 8 + _NPY_HEADER_LIMIT + _METADATA_LIMIT
_TRAINING_ID_PATTERN = re.compile(r"^phase2_training_[0-9a-f]{64}$")
_PIPELINE_PREFIX_PATTERN = re.compile(r"^phase2_pipeline_prefix_[0-9a-f]{64}$")
_EVALUATION_ROW_PATTERN = re.compile(r"^phase2_evaluation_[0-9a-f]{64}$")
_WORK_KEYS = frozenset({
    "objective_evaluations",
    "gradient_evaluations",
    "training_trajectories",
    "checkpoint_validation_trajectories",
    "test_trajectories",
    "trajectory_gate_applications",
})
_CHECKPOINT_METADATA_KEYS = frozenset({
    "schema_version",
    "pipeline_training_id",
    "pipeline_prefix_id",
    "stage_index",
    "stage_id",
    "stage_configuration_checksum",
    "circuit_binding_checksum",
    "provider_checksum",
    "objective_checksum",
    "stage_execution_checksum",
    "parameter_count",
    "selected_parameter_checksum",
    "final_parameter_checksum",
    "selected_global_iteration",
    "completed_global_iteration",
    "selected_checkpoint_validation_fidelity",
    "checkpoint_selection_checksum",
    "resume_state_checksum",
    "resume_cumulative_work",
    "resume_cumulative_cross_trajectory_pairings",
    "metadata_checksum",
})
_SIDECAR_METADATA_KEYS = frozenset({
    "schema_version",
    "evaluation_row_id",
    "pipeline_training_id",
    "map_role",
    "trajectory_count",
    "map_partitions",
    "fidelities_checksum",
    "metadata_checksum",
})
_MAP_PARTITION_KEYS = frozenset({"ensemble_id", "content_checksum", "trajectory_count"})


def artifact_checksum(payload: bytes) -> str:
    """Return the SHA-256 checksum of exact artifact bytes.

    Args:
        payload: Artifact bytes to identify.

    Returns:
        A prefixed lowercase SHA-256 checksum.

    Raises:
        TypeError: If ``payload`` is not exact bytes.
    """
    if type(payload) is not bytes:
        msg = f"payload must be bytes, got {type(payload).__name__}."
        raise TypeError(msg)
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _require_identifier(value: object, name: str, pattern: re.Pattern[str]) -> str:
    """Validate a stable hash-derived Phase II identifier."""
    if type(value) is not str or pattern.fullmatch(value) is None:
        msg = f"{name} does not match the required Phase II identity format."
        raise ValueError(msg)
    return value


def _optional_checksum(value: object, name: str) -> str | None:
    """Validate an optional checksum."""
    return None if value is None else require_checksum(value, name)


def _vector_bytes(vector: NDArray[np.float64]) -> bytes:
    """Return canonical little-endian float64 vector bytes."""
    return np.ascontiguousarray(vector, dtype=np.dtype("<f8")).tobytes(order="C")


def _vector_checksum(vector: NDArray[np.float64]) -> str:
    """Return the checksum convention used by the WP17 adapter."""
    return artifact_checksum(_vector_bytes(vector))


def _validated_vector(value: object, name: str) -> NDArray[np.float64]:
    """Validate and detach a finite one-dimensional float64 vector."""
    if not isinstance(value, np.ndarray):
        msg = f"{name} must be a NumPy array."
        raise TypeError(msg)
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or vector.size > MAX_STAGE_PARAMETER_COUNT:
        msg = f"{name} must contain between 1 and {MAX_STAGE_PARAMETER_COUNT} parameters."
        raise ValueError(msg)
    if not np.all(np.isfinite(vector)):
        msg = f"{name} must contain only finite values."
        raise ValueError(msg)
    return np.asarray(vector, dtype=np.dtype("<f8")).copy()


def _npy_payload(vector: NDArray[np.float64]) -> bytes:
    """Serialize a vector as deterministic NPY version-one bytes."""
    buffer = io.BytesIO()
    np.lib.format.write_array(
        buffer,
        np.ascontiguousarray(vector, dtype=np.dtype("<f8")),
        version=(1, 0),
        allow_pickle=False,
    )
    return buffer.getvalue()


def _decode_npy_vector(payload: bytes, name: str, *, maximum_count: int) -> NDArray[np.float64]:
    """Decode an exact bounded little-endian float64 NPY vector safely."""
    buffer = io.BytesIO(payload)
    try:
        version = np.lib.format.read_magic(buffer)
    except (EOFError, TypeError, ValueError) as error:
        msg = f"{name} has an invalid or unsupported NPY envelope."
        raise ValueError(msg) from error
    if version != (1, 0):
        msg = f"{name} must use NPY format version 1.0."
        raise ValueError(msg)
    try:
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
            buffer,
            max_header_size=_NPY_HEADER_LIMIT,
        )
    except (EOFError, TypeError, ValueError) as error:
        msg = f"{name} has an invalid or unsupported NPY envelope."
        raise ValueError(msg) from error
    resolved_dtype = np.dtype(dtype)
    if fortran_order or resolved_dtype != np.dtype("<f8") or len(shape) != 1:
        msg = f"{name} must be a C-order one-dimensional little-endian float64 array."
        raise ValueError(msg)
    count = shape[0]
    if type(count) is not int or count < 1 or count > maximum_count:
        msg = f"{name} declares an invalid element count."
        raise ValueError(msg)
    offset = buffer.tell()
    if len(payload) != offset + count * np.dtype("<f8").itemsize:
        msg = f"{name} byte length does not match its NPY header."
        raise ValueError(msg)
    vector = np.frombuffer(payload, dtype=np.dtype("<f8"), count=count, offset=offset).copy()
    if not np.all(np.isfinite(vector)):
        msg = f"{name} must contain only finite values."
        raise ValueError(msg)
    return cast("NDArray[np.float64]", vector)


def _zip_payload(members: Mapping[str, bytes]) -> bytes:
    """Create a deterministic uncompressed ZIP envelope."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, members[name])
    return buffer.getvalue()


def _bounded_metadata_payload(metadata: Mapping[str, object], name: str) -> bytes:
    """Encode metadata only when the corresponding decoder can accept it."""
    payload = canonical_json(metadata).encode()
    if len(payload) > _METADATA_LIMIT:
        msg = f"{name} metadata exceeds the {_METADATA_LIMIT}-byte codec limit."
        raise ValueError(msg)
    return payload


def _read_exact_zip(
    payload: bytes,
    *,
    expected_members: frozenset[str],
    member_limits: Mapping[str, int],
    maximum_archive_size: int,
    name: str,
) -> dict[str, bytes]:
    """Read a deterministic artifact ZIP after bounding its complete envelope."""
    if type(payload) is not bytes:
        msg = f"payload must be bytes, got {type(payload).__name__}."
        raise TypeError(msg)
    if len(payload) > maximum_archive_size + _ZIP_OVERHEAD_LIMIT:
        msg = f"{name} exceeds its maximum allowed size."
        raise ValueError(msg)
    if not payload.startswith(b"PK\x03\x04"):
        msg = f"{name} is not an exact ZIP artifact."
        raise ValueError(msg)
    end_record = payload.rfind(b"PK\x05\x06")
    if end_record < 0 or end_record + 22 != len(payload):
        msg = f"{name} contains a prefix, comment, or trailing data outside its ZIP records."
        raise ValueError(msg)
    disk_entries, total_entries = struct.unpack_from("<HH", payload, end_record + 8)
    if disk_entries != len(expected_members) or total_entries != len(expected_members):
        msg = f"{name} central-directory members count does not match the versioned format."
        raise ValueError(msg)
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except (EOFError, KeyError, OSError, RuntimeError, zipfile.BadZipFile) as error:
        msg = f"{name} could not be decoded safely."
        raise ValueError(msg) from error
    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or frozenset(names) != expected_members:
            msg = f"{name} members do not match the versioned format."
            raise ValueError(msg)
        if archive.comment:
            msg = f"{name} must not contain a ZIP comment."
            raise ValueError(msg)
        for info in infos:
            if info.flag_bits & 0x9 or info.compress_type != zipfile.ZIP_STORED:
                msg = f"{name} members must be unencrypted, descriptor-free, and uncompressed."
                raise ValueError(msg)
            if info.file_size != info.compress_size or info.file_size > member_limits[info.filename]:
                msg = f"{name} member {info.filename!r} exceeds its allowed size."
                raise ValueError(msg)
        try:
            return {info.filename: archive.read(info) for info in infos}
        except (EOFError, KeyError, OSError, RuntimeError, zipfile.BadZipFile) as error:
            msg = f"{name} members could not be read safely."
            raise ValueError(msg) from error


def _work_ledger(value: object, name: str) -> KrotovWorkLedger:
    """Construct a strict WP17 work ledger from JSON-native content."""
    mapping = require_mapping(value, name)
    require_exact_keys(mapping, _WORK_KEYS, name)
    counters = {key: require_int(mapping[key], f"{name}.{key}", minimum=0) for key in _WORK_KEYS}
    return KrotovWorkLedger(**counters)


@dataclass(frozen=True, slots=True, init=False)
class StageParameterCheckpoint:
    """Immutable selected/final parameter checkpoint for one Phase II stage."""

    pipeline_training_id: str
    pipeline_prefix_id: str
    stage_index: int
    stage_id: str
    stage_configuration_checksum: str
    circuit_binding_checksum: str | None
    provider_checksum: str | None
    objective_checksum: str | None
    stage_execution_checksum: str | None
    parameter_count: int
    selected_parameter_checksum: str
    final_parameter_checksum: str
    selected_global_iteration: int
    completed_global_iteration: int
    selected_checkpoint_validation_fidelity: float | None
    checkpoint_selection_checksum: str | None
    resume_state_checksum: str | None
    resume_cumulative_work: Mapping[str, int] | None
    resume_cumulative_cross_trajectory_pairings: int | None
    _selected_theta_bytes: bytes = field(repr=False)
    _final_theta_bytes: bytes = field(repr=False)
    schema_version: str = field(default=STAGE_PARAMETER_CHECKPOINT_SCHEMA_VERSION, init=False)

    def __init__(
        self,
        *,
        pipeline_training_id: str,
        pipeline_prefix_id: str,
        stage_index: int,
        stage_id: str,
        stage_configuration_checksum: str,
        selected_theta: NDArray[np.float64],
        final_theta: NDArray[np.float64],
        selected_global_iteration: int = 0,
        completed_global_iteration: int = 0,
        circuit_binding_checksum: str | None = None,
        provider_checksum: str | None = None,
        objective_checksum: str | None = None,
        stage_execution_checksum: str | None = None,
        resume_state: NoisyKrotovResumeState | None = None,
    ) -> None:
        """Validate identities and defensively snapshot both parameter states."""
        training_id = _require_identifier(pipeline_training_id, "pipeline_training_id", _TRAINING_ID_PATTERN)
        prefix_id = _require_identifier(pipeline_prefix_id, "pipeline_prefix_id", _PIPELINE_PREFIX_PATTERN)
        normalized_stage_index = require_int(stage_index, "stage_index", minimum=0)
        normalized_stage_id = require_slug(stage_id, "stage_id")
        stage_checksum = require_checksum(stage_configuration_checksum, "stage_configuration_checksum")
        selected = _validated_vector(selected_theta, "selected_theta")
        final = _validated_vector(final_theta, "final_theta")
        if selected.shape != final.shape:
            msg = "selected_theta and final_theta must have identical shapes."
            raise ValueError(msg)
        selected_iteration = require_int(selected_global_iteration, "selected_global_iteration", minimum=0)
        completed_iteration = require_int(completed_global_iteration, "completed_global_iteration", minimum=0)
        if selected_iteration > completed_iteration:
            msg = "selected_global_iteration cannot exceed completed_global_iteration."
            raise ValueError(msg)
        circuit_checksum = _optional_checksum(circuit_binding_checksum, "circuit_binding_checksum")
        provider = _optional_checksum(provider_checksum, "provider_checksum")
        objective = _optional_checksum(objective_checksum, "objective_checksum")
        execution_checksum = _optional_checksum(stage_execution_checksum, "stage_execution_checksum")
        if objective is not None and circuit_checksum is None:
            msg = "objective_checksum requires a circuit binding."
            raise ValueError(msg)
        if provider is not None and objective is None:
            msg = "provider_checksum requires complete noisy-resume provenance."
            raise ValueError(msg)

        selected_checksum = _vector_checksum(selected)
        final_checksum = _vector_checksum(final)
        validation_fidelity: float | None = None
        selection_checksum: str | None = None
        resume_checksum: str | None = None
        cumulative_work: Mapping[str, int] | None = None
        cumulative_pairings: int | None = None
        if resume_state is not None:
            if not isinstance(resume_state, NoisyKrotovResumeState):
                msg = "resume_state must be a NoisyKrotovResumeState or None."
                raise TypeError(msg)
            if circuit_checksum is None or objective is None:
                msg = "A noisy resume state requires circuit and objective provenance."
                raise ValueError(msg)
            expected_provenance = (stage_checksum, circuit_checksum, provider, objective)
            actual_provenance = (
                resume_state.stage_configuration_checksum,
                resume_state.circuit_binding_checksum,
                resume_state.provider_checksum,
                resume_state.objective_checksum,
            )
            if actual_provenance != expected_provenance:
                msg = "resume_state provenance does not match the checkpoint stage."
                raise ValueError(msg)
            if (
                resume_state.completed_global_iteration != completed_iteration
                or resume_state.final_parameter_checksum != final_checksum
            ):
                msg = "resume_state progress or final parameters do not match the checkpoint."
                raise ValueError(msg)
            selection = resume_state.checkpoint_selection
            if selection is None:
                if selected_iteration != completed_iteration or selected_checksum != final_checksum:
                    msg = "A resume without checkpoint selection must select the completed final state."
                    raise ValueError(msg)
            else:
                if (
                    selection.global_iteration != selected_iteration
                    or selection.parameter_checksum != selected_checksum
                    or not np.array_equal(selection.theta, selected)
                ):
                    msg = "resume_state checkpoint selection does not match selected_theta."
                    raise ValueError(msg)
                validation_fidelity = selection.validation_fidelity
                selection_checksum = selection.content_checksum
            resume_checksum = resume_state.content_checksum
            cumulative_work = MappingProxyType(resume_state.cumulative_work.to_dict())
            cumulative_pairings = resume_state.cumulative_cross_trajectory_pairings
        elif any(value is not None for value in (provider, objective)):
            msg = "Provider or objective provenance requires a complete resume_state."
            raise ValueError(msg)

        object.__setattr__(self, "pipeline_training_id", training_id)
        object.__setattr__(self, "pipeline_prefix_id", prefix_id)
        object.__setattr__(self, "stage_index", normalized_stage_index)
        object.__setattr__(self, "stage_id", normalized_stage_id)
        object.__setattr__(self, "stage_configuration_checksum", stage_checksum)
        object.__setattr__(self, "circuit_binding_checksum", circuit_checksum)
        object.__setattr__(self, "provider_checksum", provider)
        object.__setattr__(self, "objective_checksum", objective)
        object.__setattr__(self, "stage_execution_checksum", execution_checksum)
        object.__setattr__(self, "parameter_count", int(selected.size))
        object.__setattr__(self, "selected_parameter_checksum", selected_checksum)
        object.__setattr__(self, "final_parameter_checksum", final_checksum)
        object.__setattr__(self, "selected_global_iteration", selected_iteration)
        object.__setattr__(self, "completed_global_iteration", completed_iteration)
        object.__setattr__(self, "selected_checkpoint_validation_fidelity", validation_fidelity)
        object.__setattr__(self, "checkpoint_selection_checksum", selection_checksum)
        object.__setattr__(self, "resume_state_checksum", resume_checksum)
        object.__setattr__(self, "resume_cumulative_work", cumulative_work)
        object.__setattr__(self, "resume_cumulative_cross_trajectory_pairings", cumulative_pairings)
        object.__setattr__(self, "_selected_theta_bytes", _vector_bytes(selected))
        object.__setattr__(self, "_final_theta_bytes", _vector_bytes(final))
        object.__setattr__(self, "schema_version", STAGE_PARAMETER_CHECKPOINT_SCHEMA_VERSION)

    @classmethod
    def from_noisy_krotov(
        cls,
        *,
        pipeline_training_id: str,
        pipeline_prefix_id: str,
        execution: NoisyKrotovStageExecution,
    ) -> StageParameterCheckpoint:
        """Create a checkpoint from a successful WP17 stage execution.

        Args:
            pipeline_training_id: Stable full-pipeline training identity.
            pipeline_prefix_id: Stable prefix identity through this stage.
            execution: Successful in-memory WP17 execution.

        Returns:
            A detached selected/final checkpoint with complete resume evidence.
        """
        if not isinstance(execution, NoisyKrotovStageExecution):
            msg = "execution must be a NoisyKrotovStageExecution."
            raise TypeError(msg)
        resume_state = execution.resume_state
        return cls(
            pipeline_training_id=pipeline_training_id,
            pipeline_prefix_id=pipeline_prefix_id,
            stage_index=execution.stage_index,
            stage_id=execution.stage_id,
            stage_configuration_checksum=execution.stage_configuration_checksum,
            selected_theta=execution.selected_theta,
            final_theta=execution.final_theta,
            selected_global_iteration=execution.selected_global_iteration,
            completed_global_iteration=resume_state.completed_global_iteration,
            circuit_binding_checksum=execution.circuit_binding_checksum,
            provider_checksum=execution.provider_checksum,
            objective_checksum=execution.objective_checksum,
            stage_execution_checksum=execution.content_checksum,
            resume_state=resume_state,
        )

    @property
    def selected_theta(self) -> NDArray[np.float64]:
        """A detached writable selected parameter vector."""
        return np.frombuffer(self._selected_theta_bytes, dtype=np.dtype("<f8")).copy()

    @property
    def final_theta(self) -> NDArray[np.float64]:
        """A detached writable last-iteration parameter vector."""
        return np.frombuffer(self._final_theta_bytes, dtype=np.dtype("<f8")).copy()

    def _metadata_payload(self) -> dict[str, object]:
        """Return checksum-covered metadata without its checksum field."""
        return {
            "schema_version": self.schema_version,
            "pipeline_training_id": self.pipeline_training_id,
            "pipeline_prefix_id": self.pipeline_prefix_id,
            "stage_index": self.stage_index,
            "stage_id": self.stage_id,
            "stage_configuration_checksum": self.stage_configuration_checksum,
            "circuit_binding_checksum": self.circuit_binding_checksum,
            "provider_checksum": self.provider_checksum,
            "objective_checksum": self.objective_checksum,
            "stage_execution_checksum": self.stage_execution_checksum,
            "parameter_count": self.parameter_count,
            "selected_parameter_checksum": self.selected_parameter_checksum,
            "final_parameter_checksum": self.final_parameter_checksum,
            "selected_global_iteration": self.selected_global_iteration,
            "completed_global_iteration": self.completed_global_iteration,
            "selected_checkpoint_validation_fidelity": self.selected_checkpoint_validation_fidelity,
            "checkpoint_selection_checksum": self.checkpoint_selection_checksum,
            "resume_state_checksum": self.resume_state_checksum,
            "resume_cumulative_work": (
                None if self.resume_cumulative_work is None else dict(self.resume_cumulative_work)
            ),
            "resume_cumulative_cross_trajectory_pairings": self.resume_cumulative_cross_trajectory_pairings,
        }

    def metadata_dict(self) -> dict[str, object]:
        """Return detached checksum-sealed checkpoint metadata."""
        payload = self._metadata_payload()
        return {**payload, "metadata_checksum": canonical_checksum(payload)}

    def to_bytes(self) -> bytes:
        """Serialize this checkpoint into deterministic exact-member bytes."""
        return _zip_payload({
            "metadata.json": _bounded_metadata_payload(self.metadata_dict(), "Stage parameter checkpoint"),
            "selected_theta.npy": _npy_payload(self.selected_theta),
            "final_theta.npy": _npy_payload(self.final_theta),
        })

    @property
    def content_checksum(self) -> str:
        """Checksum of the exact deterministic checkpoint archive."""
        return artifact_checksum(self.to_bytes())

    @property
    def archive_checksum(self) -> str:
        """Alias for the exact deterministic archive checksum."""
        return self.content_checksum

    def to_noisy_krotov_resume_state(self) -> NoisyKrotovResumeState:
        """Reconstruct and verify the WP17 resume state stored in this checkpoint.

        Returns:
            A provenance-bound resume state carrying detached selection data.

        Raises:
            ValueError: If this is a generic transform-stage checkpoint.
        """
        if (
            self.circuit_binding_checksum is None
            or self.objective_checksum is None
            or self.resume_state_checksum is None
            or self.resume_cumulative_work is None
            or self.resume_cumulative_cross_trajectory_pairings is None
        ):
            msg = "This generic transform-stage checkpoint has no noisy Krotov resume state."
            raise ValueError(msg)
        selection: NoisyKrotovCheckpointSelection | None = None
        if self.selected_checkpoint_validation_fidelity is not None:
            selection = NoisyKrotovCheckpointSelection(
                stage_configuration_checksum=self.stage_configuration_checksum,
                circuit_binding_checksum=self.circuit_binding_checksum,
                provider_checksum=self.provider_checksum,
                objective_checksum=self.objective_checksum,
                global_iteration=self.selected_global_iteration,
                validation_fidelity=self.selected_checkpoint_validation_fidelity,
                theta=self.selected_theta,
            )
            if selection.content_checksum != self.checkpoint_selection_checksum:
                msg = "Stored checkpoint-selection provenance could not be reconstructed."
                raise ValueError(msg)
        ledger = _work_ledger(self.resume_cumulative_work, "resume_cumulative_work")
        resume = NoisyKrotovResumeState(
            stage_configuration_checksum=self.stage_configuration_checksum,
            circuit_binding_checksum=self.circuit_binding_checksum,
            provider_checksum=self.provider_checksum,
            objective_checksum=self.objective_checksum,
            completed_global_iteration=self.completed_global_iteration,
            final_parameter_checksum=self.final_parameter_checksum,
            checkpoint_selection=selection,
            cumulative_work=ledger,
            cumulative_cross_trajectory_pairings=self.resume_cumulative_cross_trajectory_pairings,
        )
        if resume.content_checksum != self.resume_state_checksum:
            msg = "Stored noisy Krotov resume provenance could not be reconstructed."
            raise ValueError(msg)
        return resume

    @classmethod
    def from_bytes(
        cls,
        payload: bytes,
        *,
        expected_checksum: str | None = None,
        expected_pipeline_training_id: str | None = None,
        expected_pipeline_prefix_id: str | None = None,
        expected_stage_configuration_checksum: str | None = None,
    ) -> StageParameterCheckpoint:
        """Decode and verify a bounded exact-member checkpoint archive.

        Args:
            payload: Exact archive bytes.
            expected_checksum: Optional expected checksum of the archive bytes.
            expected_pipeline_training_id: Optional expected full-pipeline identity.
            expected_pipeline_prefix_id: Optional expected stage-prefix identity.
            expected_stage_configuration_checksum: Optional expected stage config.

        Returns:
            A detached verified checkpoint.

        Raises:
            ValueError: If the archive, metadata, vectors, or expected bindings differ.
        """
        if expected_checksum is not None:
            checksum = require_checksum(expected_checksum, "expected_checksum")
            actual = artifact_checksum(payload)
            if checksum != actual:
                msg = f"Checkpoint checksum mismatch: expected {checksum}, computed {actual}."
                raise ValueError(msg)
        members = _read_exact_zip(
            payload,
            expected_members=_CHECKPOINT_MEMBERS,
            member_limits={
                "metadata.json": _METADATA_LIMIT,
                "selected_theta.npy": MAX_STAGE_PARAMETER_COUNT * 8 + _NPY_HEADER_LIMIT,
                "final_theta.npy": MAX_STAGE_PARAMETER_COUNT * 8 + _NPY_HEADER_LIMIT,
            },
            maximum_archive_size=_MAX_CHECKPOINT_ARCHIVE_SIZE,
            name="Stage parameter checkpoint",
        )
        try:
            metadata_text = members["metadata.json"].decode("utf-8")
        except UnicodeDecodeError as error:
            msg = "Checkpoint metadata is not valid UTF-8."
            raise ValueError(msg) from error
        metadata = load_canonical_json_object(metadata_text)
        require_exact_keys(metadata, _CHECKPOINT_METADATA_KEYS, "stage checkpoint metadata")
        if metadata["schema_version"] != STAGE_PARAMETER_CHECKPOINT_SCHEMA_VERSION:
            msg = f"schema_version must be {STAGE_PARAMETER_CHECKPOINT_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        supplied_metadata_checksum = require_checksum(metadata["metadata_checksum"], "metadata_checksum")
        metadata_payload = {key: thaw_json(value) for key, value in metadata.items() if key != "metadata_checksum"}
        if canonical_checksum(metadata_payload) != supplied_metadata_checksum:
            msg = "Stage checkpoint metadata checksum mismatch."
            raise ValueError(msg)

        selected = _decode_npy_vector(
            members["selected_theta.npy"],
            "selected_theta.npy",
            maximum_count=MAX_STAGE_PARAMETER_COUNT,
        )
        final = _decode_npy_vector(
            members["final_theta.npy"],
            "final_theta.npy",
            maximum_count=MAX_STAGE_PARAMETER_COUNT,
        )
        parameter_count = require_int(metadata["parameter_count"], "parameter_count", minimum=1)
        if parameter_count != len(selected) or parameter_count != len(final):
            msg = "Checkpoint parameter count does not match its vectors."
            raise ValueError(msg)
        if require_checksum(metadata["selected_parameter_checksum"], "selected_parameter_checksum") != _vector_checksum(
            selected
        ):
            msg = "Selected parameter checksum does not match selected_theta.npy."
            raise ValueError(msg)
        if require_checksum(metadata["final_parameter_checksum"], "final_parameter_checksum") != _vector_checksum(
            final
        ):
            msg = "Final parameter checksum does not match final_theta.npy."
            raise ValueError(msg)

        circuit_checksum = _optional_checksum(metadata["circuit_binding_checksum"], "circuit_binding_checksum")
        provider_checksum = _optional_checksum(metadata["provider_checksum"], "provider_checksum")
        objective_checksum = _optional_checksum(metadata["objective_checksum"], "objective_checksum")
        resume_checksum = _optional_checksum(metadata["resume_state_checksum"], "resume_state_checksum")
        selected_iteration = require_int(metadata["selected_global_iteration"], "selected_global_iteration", minimum=0)
        completed_iteration = require_int(
            metadata["completed_global_iteration"],
            "completed_global_iteration",
            minimum=0,
        )
        fidelity = (
            None
            if metadata["selected_checkpoint_validation_fidelity"] is None
            else require_float(
                metadata["selected_checkpoint_validation_fidelity"],
                "selected_checkpoint_validation_fidelity",
                minimum=0.0,
                maximum=1.0,
            )
        )
        resume_state: NoisyKrotovResumeState | None = None
        if resume_checksum is not None:
            if circuit_checksum is None or objective_checksum is None:
                msg = "Noisy checkpoint metadata is missing circuit or objective provenance."
                raise ValueError(msg)
            work = _work_ledger(metadata["resume_cumulative_work"], "resume_cumulative_work")
            pairings = require_int(
                metadata["resume_cumulative_cross_trajectory_pairings"],
                "resume_cumulative_cross_trajectory_pairings",
                minimum=0,
            )
            selection: NoisyKrotovCheckpointSelection | None = None
            if fidelity is not None:
                selection = NoisyKrotovCheckpointSelection(
                    stage_configuration_checksum=cast("str", metadata["stage_configuration_checksum"]),
                    circuit_binding_checksum=circuit_checksum,
                    provider_checksum=provider_checksum,
                    objective_checksum=objective_checksum,
                    global_iteration=selected_iteration,
                    validation_fidelity=fidelity,
                    theta=selected,
                )
                expected_selection = _optional_checksum(
                    metadata["checkpoint_selection_checksum"],
                    "checkpoint_selection_checksum",
                )
                if selection.content_checksum != expected_selection:
                    msg = "Checkpoint-selection checksum does not match selected parameters."
                    raise ValueError(msg)
            resume_state = NoisyKrotovResumeState(
                stage_configuration_checksum=cast("str", metadata["stage_configuration_checksum"]),
                circuit_binding_checksum=circuit_checksum,
                provider_checksum=provider_checksum,
                objective_checksum=objective_checksum,
                completed_global_iteration=completed_iteration,
                final_parameter_checksum=cast("str", metadata["final_parameter_checksum"]),
                checkpoint_selection=selection,
                cumulative_work=work,
                cumulative_cross_trajectory_pairings=pairings,
            )
            if resume_state.content_checksum != resume_checksum:
                msg = "Noisy Krotov resume-state checksum does not match checkpoint metadata."
                raise ValueError(msg)
        elif any(
            metadata[key] is not None
            for key in (
                "provider_checksum",
                "objective_checksum",
                "selected_checkpoint_validation_fidelity",
                "checkpoint_selection_checksum",
                "resume_cumulative_work",
                "resume_cumulative_cross_trajectory_pairings",
            )
        ):
            msg = "Generic checkpoint metadata contains incomplete noisy-resume fields."
            raise ValueError(msg)

        checkpoint = cls(
            pipeline_training_id=cast("str", metadata["pipeline_training_id"]),
            pipeline_prefix_id=cast("str", metadata["pipeline_prefix_id"]),
            stage_index=cast("int", metadata["stage_index"]),
            stage_id=cast("str", metadata["stage_id"]),
            stage_configuration_checksum=cast("str", metadata["stage_configuration_checksum"]),
            selected_theta=selected,
            final_theta=final,
            selected_global_iteration=selected_iteration,
            completed_global_iteration=completed_iteration,
            circuit_binding_checksum=circuit_checksum,
            provider_checksum=provider_checksum,
            objective_checksum=objective_checksum,
            stage_execution_checksum=cast("str | None", metadata["stage_execution_checksum"]),
            resume_state=resume_state,
        )
        if canonical_json(checkpoint.metadata_dict()) != canonical_json(metadata):
            msg = "Stage checkpoint metadata changed during normalization."
            raise ValueError(msg)
        expected_bindings = (
            ("pipeline_training_id", expected_pipeline_training_id),
            ("pipeline_prefix_id", expected_pipeline_prefix_id),
            ("stage_configuration_checksum", expected_stage_configuration_checksum),
        )
        for field_name, expected in expected_bindings:
            if expected is not None and getattr(checkpoint, field_name) != expected:
                msg = f"Checkpoint {field_name} does not match the requested resume context."
                raise ValueError(msg)
        return checkpoint


def _validated_map_partitions(
    value: object,
    *,
    expected_count: int,
) -> tuple[dict[str, object], ...]:
    """Validate an ordered map-to-trajectory partition covering every outcome."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = "map_partitions must be a sequence."
        raise TypeError(msg)
    partitions: list[dict[str, object]] = []
    for index, raw_partition in enumerate(value):
        mapping = require_mapping(raw_partition, f"map_partitions[{index}]")
        require_exact_keys(mapping, _MAP_PARTITION_KEYS, f"map_partitions[{index}]")
        partitions.append({
            "ensemble_id": require_slug(mapping["ensemble_id"], f"map_partitions[{index}].ensemble_id"),
            "content_checksum": require_checksum(
                mapping["content_checksum"],
                f"map_partitions[{index}].content_checksum",
            ),
            "trajectory_count": require_int(
                mapping["trajectory_count"],
                f"map_partitions[{index}].trajectory_count",
                minimum=1,
            ),
        })
    if not partitions or sum(cast("int", item["trajectory_count"]) for item in partitions) != expected_count:
        msg = "map_partitions must cover the trajectory outcomes exactly."
        raise ValueError(msg)
    if len({cast("str", item["ensemble_id"]) for item in partitions}) != len(partitions):
        msg = "map_partitions must not reuse an ensemble identity."
        raise ValueError(msg)
    if len({cast("str", item["content_checksum"]) for item in partitions}) != len(partitions):
        msg = "map_partitions must not reuse ensemble content."
        raise ValueError(msg)
    return tuple(partitions)


def _sidecar_metadata(
    *,
    evaluation_row_id: str,
    pipeline_training_id: str,
    map_role: str,
    trajectory_count: int,
    map_partitions: Sequence[Mapping[str, object]],
    fidelities_checksum: str,
) -> dict[str, object]:
    """Create checksum-sealed trajectory-sidecar metadata."""
    payload = {
        "schema_version": PHASE2_TRAJECTORY_SIDECAR_SCHEMA_VERSION,
        "evaluation_row_id": evaluation_row_id,
        "pipeline_training_id": pipeline_training_id,
        "map_role": map_role,
        "trajectory_count": trajectory_count,
        "map_partitions": [dict(partition) for partition in map_partitions],
        "fidelities_checksum": fidelities_checksum,
    }
    return {**payload, "metadata_checksum": canonical_checksum(payload)}


def create_phase2_trajectory_sidecar(
    *,
    evaluation_row_id: str,
    pipeline_training_id: str,
    map_role: str,
    map_partitions: Sequence[Mapping[str, object]],
    fidelities: Sequence[float],
) -> bytes:
    """Create a deterministic identity-bound Phase II fidelity sidecar.

    Args:
        evaluation_row_id: Stable Phase II evaluation-row identity.
        pipeline_training_id: Stable training-pipeline identity.
        map_role: Random-stream role used for these trajectories.
        map_partitions: Ordered ensemble identities, checksums, and trajectory
            counts corresponding to contiguous fidelity ranges.
        fidelities: Per-trajectory fidelities in canonical order.

    Returns:
        Deterministic exact-member sidecar bytes.
    """
    row_id = _require_identifier(evaluation_row_id, "evaluation_row_id", _EVALUATION_ROW_PATTERN)
    resolved_training_id = _require_identifier(
        pipeline_training_id,
        "pipeline_training_id",
        _TRAINING_ID_PATTERN,
    )
    if type(map_role) is not str or map_role not in PHASE2_MAP_ROLES:
        msg = f"map_role must be one of {PHASE2_MAP_ROLES!r}."
        raise ValueError(msg)
    if isinstance(fidelities, (str, bytes)) or not isinstance(fidelities, Sequence):
        msg = "fidelities must be a sequence of finite real values."
        raise TypeError(msg)
    try:
        values = np.asarray(fidelities, dtype=np.dtype("<f8"))
    except (TypeError, ValueError) as error:
        msg = "fidelities must be convertible to float64."
        raise TypeError(msg) from error
    if values.ndim != 1 or not 1 <= values.size <= MAX_TRAJECTORY_FIDELITY_COUNT:
        msg = f"fidelities must contain between 1 and {MAX_TRAJECTORY_FIDELITY_COUNT} values."
        raise ValueError(msg)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        msg = "fidelities must be finite and lie in [0, 1]."
        raise ValueError(msg)
    detached = np.asarray(values, dtype=np.dtype("<f8")).copy()
    partitions = _validated_map_partitions(map_partitions, expected_count=int(detached.size))
    metadata = _sidecar_metadata(
        evaluation_row_id=row_id,
        pipeline_training_id=resolved_training_id,
        map_role=map_role,
        trajectory_count=int(detached.size),
        map_partitions=partitions,
        fidelities_checksum=_vector_checksum(detached),
    )
    return _zip_payload({
        "metadata.json": _bounded_metadata_payload(metadata, "Phase II trajectory sidecar"),
        "fidelities.npy": _npy_payload(detached),
    })


def read_phase2_trajectory_sidecar(
    payload: bytes,
    *,
    expected_evaluation_row_id: str,
    expected_pipeline_training_id: str,
    expected_map_role: str,
    expected_map_partitions: Sequence[Mapping[str, object]],
    expected_count: int,
    expected_checksum: str | None = None,
) -> tuple[float, ...]:
    """Safely verify and decode a bounded Phase II fidelity sidecar.

    Args:
        payload: Exact sidecar bytes.
        expected_evaluation_row_id: Required evaluation-row identity.
        expected_pipeline_training_id: Required training identity.
        expected_map_role: Required random-stream role.
        expected_map_partitions: Required ordered map-to-trajectory partition.
        expected_count: Required number of trajectory outcomes.
        expected_checksum: Optional checksum of the exact sidecar bytes.

    Returns:
        The immutable ordered trajectory fidelities.

    Raises:
        ValueError: If identity, checksum, structure, or fidelity content differs.
    """
    row_id = _require_identifier(
        expected_evaluation_row_id,
        "expected_evaluation_row_id",
        _EVALUATION_ROW_PATTERN,
    )
    training_id = _require_identifier(
        expected_pipeline_training_id,
        "expected_pipeline_training_id",
        _TRAINING_ID_PATTERN,
    )
    if type(expected_map_role) is not str or expected_map_role not in PHASE2_MAP_ROLES:
        msg = f"expected_map_role must be one of {PHASE2_MAP_ROLES!r}."
        raise ValueError(msg)
    count = require_int(expected_count, "expected_count", minimum=1)
    if count > MAX_TRAJECTORY_FIDELITY_COUNT:
        msg = "expected_count exceeds the Phase II sidecar limit."
        raise ValueError(msg)
    partitions = _validated_map_partitions(expected_map_partitions, expected_count=count)
    if expected_checksum is not None:
        checksum = require_checksum(expected_checksum, "expected_checksum")
        actual = artifact_checksum(payload)
        if checksum != actual:
            msg = f"Trajectory sidecar checksum mismatch: expected {checksum}, computed {actual}."
            raise ValueError(msg)
    members = _read_exact_zip(
        payload,
        expected_members=_SIDECAR_MEMBERS,
        member_limits={
            "metadata.json": _METADATA_LIMIT,
            "fidelities.npy": count * 8 + _NPY_HEADER_LIMIT,
        },
        maximum_archive_size=min(_MAX_SIDECAR_ARCHIVE_SIZE, count * 8 + _NPY_HEADER_LIMIT + _METADATA_LIMIT),
        name="Phase II trajectory sidecar",
    )
    try:
        metadata_text = members["metadata.json"].decode("utf-8")
    except UnicodeDecodeError as error:
        msg = "Trajectory sidecar metadata is not valid UTF-8."
        raise ValueError(msg) from error
    metadata = load_canonical_json_object(metadata_text)
    require_exact_keys(metadata, _SIDECAR_METADATA_KEYS, "trajectory sidecar metadata")
    if metadata["schema_version"] != PHASE2_TRAJECTORY_SIDECAR_SCHEMA_VERSION:
        msg = f"schema_version must be {PHASE2_TRAJECTORY_SIDECAR_SCHEMA_VERSION!r}."
        raise ValueError(msg)
    supplied_metadata_checksum = require_checksum(metadata["metadata_checksum"], "metadata_checksum")
    metadata_payload = {key: thaw_json(value) for key, value in metadata.items() if key != "metadata_checksum"}
    if canonical_checksum(metadata_payload) != supplied_metadata_checksum:
        msg = "Trajectory sidecar metadata checksum mismatch."
        raise ValueError(msg)
    expected_identity = {
        "evaluation_row_id": row_id,
        "pipeline_training_id": training_id,
        "map_role": expected_map_role,
        "trajectory_count": count,
        "map_partitions": partitions,
    }
    for key, expected in expected_identity.items():
        if metadata[key] != expected:
            msg = f"Trajectory sidecar {key} does not match the expected evaluation context."
            raise ValueError(msg)
    values = _decode_npy_vector(members["fidelities.npy"], "fidelities.npy", maximum_count=count)
    if len(values) != count:
        msg = "Trajectory sidecar fidelity count does not match the expected budget."
        raise ValueError(msg)
    if require_checksum(metadata["fidelities_checksum"], "fidelities_checksum") != _vector_checksum(values):
        msg = "Trajectory sidecar fidelity checksum does not match fidelities.npy."
        raise ValueError(msg)
    if np.any(values < 0.0) or np.any(values > 1.0):
        msg = "Trajectory sidecar fidelities must lie in [0, 1]."
        raise ValueError(msg)
    return tuple(float(value) for value in values)


__all__ = [
    "MAX_STAGE_PARAMETER_COUNT",
    "MAX_TRAJECTORY_FIDELITY_COUNT",
    "PHASE2_MAP_ROLES",
    "PHASE2_TRAJECTORY_SIDECAR_SCHEMA_VERSION",
    "STAGE_PARAMETER_CHECKPOINT_SCHEMA_VERSION",
    "StageParameterCheckpoint",
    "artifact_checksum",
    "create_phase2_trajectory_sidecar",
    "read_phase2_trajectory_sidecar",
]
