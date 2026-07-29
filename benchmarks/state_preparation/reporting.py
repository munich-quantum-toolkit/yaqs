# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Atomic reporting, checkpointing, and resumability for benchmark runs."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import numpy as np
import scipy

from .evaluation import IndependentEvaluation
from .methods import StatePreparationTrainingArtifact
from .schema import (
    CSV_COLUMNS,
    RESULT_SCHEMA_VERSION,
    BenchmarkConfig,
    BenchmarkFailure,
    BenchmarkResult,
    benchmark_record_from_csv_row,
    benchmark_record_from_json,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

REPORT_MANIFEST_FORMAT = "yaqs.state_preparation.run_manifest.v1"
TRAJECTORY_SIDECAR_FORMAT = "yaqs.state_preparation.trajectory_sidecar.v1"
RESULTS_JSONL_NAME = "results.jsonl"
RESULTS_CSV_NAME = "results.csv"
MANIFEST_NAME = "manifest.json"
CHECKPOINT_DIRECTORY = "checkpoints"
TRAJECTORY_DIRECTORY = "trajectories"

_CHECKSUM_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_REQUIRED_SOFTWARE_VERSIONS = frozenset({"yaqs", "python", "numpy", "scipy"})
_SIDECAR_MEMBERS = frozenset({"format.npy", "run_id.npy", "training_id.npy", "repetition.npy", "fidelities.npy"})
_SIDECAR_NPY_ALLOWANCE = 4096
BenchmarkRecord = BenchmarkResult | BenchmarkFailure


def _canonical_json(value: object) -> str:
    """Serialize a JSON-native value deterministically.

    Returns:
        The canonical JSON document.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha256(payload: bytes) -> str:
    """Return a benchmark-formatted SHA-256 checksum.

    Returns:
        ``sha256:`` followed by a lowercase hexadecimal digest.
    """
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _validate_checksum(value: object, name: str) -> str:
    """Validate one benchmark checksum.

    Returns:
        The validated checksum.

    Raises:
        ValueError: If the checksum is malformed.
    """
    if type(value) is not str or _CHECKSUM_PATTERN.fullmatch(value) is None:
        msg = f"{name} must be 'sha256:' followed by 64 lowercase hexadecimal characters."
        raise ValueError(msg)
    return value


def _immutable_versions(value: object) -> Mapping[str, str]:
    """Validate and freeze software version provenance.

    Returns:
        An immutable string mapping.

    Raises:
        TypeError: If the input is not a string mapping.
        ValueError: If a required key or version is missing.
    """
    if not isinstance(value, Mapping):
        msg = "software_versions must be a mapping."
        raise TypeError(msg)
    versions: dict[str, str] = {}
    for key, version in value.items():
        if type(key) is not str or type(version) is not str:
            msg = "software_versions must contain only string keys and values."
            raise TypeError(msg)
        if not key or not version or key != key.strip() or version != version.strip():
            msg = "software_versions keys and values must be nonempty without surrounding whitespace."
            raise ValueError(msg)
        versions[key] = version
    missing = sorted(_REQUIRED_SOFTWARE_VERSIONS - set(versions))
    if missing:
        msg = f"software_versions is missing required keys: {missing}."
        raise ValueError(msg)
    return MappingProxyType(dict(sorted(versions.items())))


@dataclass(frozen=True, slots=True)
class RunProvenance:
    """Software and Git implementation fingerprint for one reporting process."""

    software_versions: Mapping[str, str]
    git_commit: str
    git_dirty: bool
    git_diff_checksum: str | None = None

    def __post_init__(self) -> None:
        """Validate and freeze the provenance snapshot.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If Git metadata is malformed or inconsistent.
        """
        object.__setattr__(self, "software_versions", _immutable_versions(self.software_versions))
        if type(self.git_commit) is not str or _GIT_COMMIT_PATTERN.fullmatch(self.git_commit) is None:
            msg = "git_commit must be a complete 40- or 64-character lowercase hexadecimal object ID."
            raise ValueError(msg)
        if type(self.git_dirty) is not bool:
            msg = f"git_dirty must be a bool, got {type(self.git_dirty).__name__}."
            raise TypeError(msg)
        checksum = (
            None
            if self.git_diff_checksum is None
            else _validate_checksum(
                self.git_diff_checksum,
                "git_diff_checksum",
            )
        )
        if self.git_dirty != (checksum is not None):
            msg = "git_diff_checksum is required exactly when git_dirty is true."
            raise ValueError(msg)
        object.__setattr__(self, "git_diff_checksum", checksum)

    @property
    def fingerprint(self) -> str:
        """Stable checksum of the complete implementation provenance."""
        return _sha256(_canonical_json(self.to_dict()).encode())

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-native provenance record."""
        return {
            "software_versions": dict(self.software_versions),
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "git_diff_checksum": self.git_diff_checksum,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RunProvenance:
        """Restore a strict provenance record.

        Returns:
            The validated provenance snapshot.

        Raises:
            ValueError: If keys differ from the versioned representation.
        """
        expected = {"software_versions", "git_commit", "git_dirty", "git_diff_checksum"}
        if set(data) != expected:
            msg = "RunProvenance fields do not match the versioned representation."
            raise ValueError(msg)
        return cls(
            software_versions=cast("Mapping[str, str]", data["software_versions"]),
            git_commit=cast("str", data["git_commit"]),
            git_dirty=cast("bool", data["git_dirty"]),
            git_diff_checksum=cast("str | None", data["git_diff_checksum"]),
        )

    @classmethod
    def from_record(cls, record: BenchmarkRecord) -> RunProvenance:
        """Extract provenance from a validated result-stream record.

        Returns:
            The record's provenance snapshot.
        """
        return cls(
            software_versions=cast("Mapping[str, str]", record.software_versions),
            git_commit=record.git_commit,
            git_dirty=record.git_dirty,
            git_diff_checksum=record.git_diff_checksum,
        )


def _run_git(repository_root: Path, *arguments: str) -> bytes:
    """Run one read-only Git command and return raw stdout.

    Returns:
        The command's stdout bytes.

    Raises:
        ValueError: If Git cannot inspect the repository.
    """
    git_executable = shutil.which("git")
    if git_executable is None:
        msg = "Could not inspect Git repository because the Git executable was not found."
        raise ValueError(msg)
    try:
        completed = subprocess.run(  # noqa: S603 -- resolved executable; no shell interpretation
            (git_executable, "-C", os.fspath(repository_root), *arguments),
            check=True,
            capture_output=True,
            shell=False,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        msg = f"Could not inspect Git repository {repository_root}: {error}."
        raise ValueError(msg) from error
    return completed.stdout


def _dirty_repository_payload(repository_root: Path, status: bytes) -> bytes:
    """Build a content-sensitive fingerprint payload for a dirty repository.

    Returns:
        Tracked changes, status metadata, and untracked-file content digests.
    """
    tracked_diff = _run_git(repository_root, "diff", "HEAD", "--binary", "--no-ext-diff")
    untracked = _run_git(repository_root, "ls-files", "--others", "--exclude-standard", "-z")
    payload = bytearray(b"status\0")
    payload.extend(status)
    payload.extend(b"\0tracked-diff\0")
    payload.extend(tracked_diff)
    for encoded_path in sorted(filter(None, untracked.split(b"\0"))):
        relative_path = os.fsdecode(encoded_path)
        path = repository_root / relative_path
        payload.extend(b"\0untracked\0")
        payload.extend(encoded_path)
        if path.is_symlink():
            content_digest = hashlib.sha256(os.fsencode(path.readlink())).digest()
        elif path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
            content_digest = digest.digest()
        else:
            content_digest = hashlib.sha256(b"<non-file>").digest()
        payload.extend(b"\0")
        payload.extend(content_digest)
    return bytes(payload)


def capture_run_provenance(repository_root: Path) -> RunProvenance:
    """Capture software versions and a content-sensitive Git fingerprint.

    Args:
        repository_root: Root of the Git checkout being executed.

    Returns:
        The validated provenance snapshot.

    Raises:
        TypeError: If ``repository_root`` is not a path.
    """
    if not isinstance(repository_root, Path):
        msg = f"repository_root must be a pathlib.Path, got {type(repository_root).__name__}."
        raise TypeError(msg)
    root = repository_root.resolve()
    commit = _run_git(root, "rev-parse", "HEAD").decode("ascii").strip()
    status = _run_git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    dirty = bool(status)
    diff_checksum = _sha256(_dirty_repository_payload(root, status)) if dirty else None
    try:
        yaqs_version = importlib.metadata.version("mqt.yaqs")
    except importlib.metadata.PackageNotFoundError:
        yaqs_version = importlib.metadata.version("mqt-yaqs")
    return RunProvenance(
        software_versions={
            "yaqs": yaqs_version,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        git_commit=commit,
        git_dirty=dirty,
        git_diff_checksum=diff_checksum,
    )


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Atomically replace one file with exact bytes in the same directory.

    Args:
        path: Destination file.
        payload: Exact bytes to publish.

    Raises:
        TypeError: If arguments have unsupported types.
    """
    if not isinstance(path, Path):
        msg = f"path must be a pathlib.Path, got {type(path).__name__}."
        raise TypeError(msg)
    if type(payload) is not bytes:
        msg = f"payload must be bytes, got {type(payload).__name__}."
        raise TypeError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path: Path | None = Path(temporary_name)
    try:
        _write_and_sync_descriptor(descriptor, payload)
        Path(temporary_path).replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    _sync_directory(path.parent)


def _write_and_sync_descriptor(descriptor: int, payload: bytes) -> None:
    """Write, flush, and close a temporary file descriptor."""
    with os.fdopen(descriptor, "wb") as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())


def _sync_directory(directory: Path) -> None:
    """Persist a directory entry update before returning."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _npy_payload(array: NDArray[np.generic]) -> bytes:
    """Serialize one non-object C-order array deterministically.

    Returns:
        NPY version-one bytes.
    """
    resolved = np.asarray(array)
    if not resolved.flags.c_contiguous:
        resolved = np.ascontiguousarray(resolved)
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, resolved, version=(1, 0), allow_pickle=False)
    return buffer.getvalue()


def _text_array(value: str) -> NDArray[np.uint8]:
    """Encode text as a one-dimensional immutable-byte array.

    Returns:
        The UTF-8 bytes as ``uint8``.
    """
    return np.frombuffer(value.encode(), dtype=np.uint8)


def create_trajectory_sidecar(
    *,
    run_id: str,
    training_id: str,
    repetition: int,
    fidelities: Sequence[float],
) -> bytes:
    """Create a deterministic compressed trajectory-fidelity sidecar.

    Returns:
        A versioned ZIP/NPZ-compatible byte payload.

    Raises:
        TypeError: If scalar identifiers or fidelities have unsupported types.
        ValueError: If repetition or fidelities are invalid.
    """
    for name, value in (("run_id", run_id), ("training_id", training_id)):
        if type(value) is not str or not value:
            msg = f"{name} must be a nonempty string."
            raise TypeError(msg)
    if type(repetition) is not int:
        msg = f"repetition must be an int, got {type(repetition).__name__}."
        raise TypeError(msg)
    if not 0 <= repetition <= 2**32 - 1:
        msg = f"repetition must lie in [0, {2**32 - 1}]."
        raise ValueError(msg)
    try:
        fidelity_array = np.asarray(fidelities, dtype=np.dtype("<f8"))
    except (TypeError, ValueError) as error:
        msg = "fidelities must be convertible to finite float64 values."
        raise TypeError(msg) from error
    if fidelity_array.ndim != 1 or not np.all(np.isfinite(fidelity_array)):
        msg = "fidelities must be a one-dimensional sequence of finite values."
        raise ValueError(msg)
    if np.any(fidelity_array < 0.0) or np.any(fidelity_array > 1.0):
        msg = "fidelities must lie in [0, 1]."
        raise ValueError(msg)

    members = {
        "format.npy": _npy_payload(_text_array(TRAJECTORY_SIDECAR_FORMAT)),
        "run_id.npy": _npy_payload(_text_array(run_id)),
        "training_id.npy": _npy_payload(_text_array(training_id)),
        "repetition.npy": _npy_payload(np.asarray(repetition, dtype=np.dtype("<u4"))),
        "fidelities.npy": _npy_payload(fidelity_array),
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, members[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return buffer.getvalue()


def _load_npy_member(
    archive: zipfile.ZipFile,
    name: str,
    *,
    maximum_size: int,
) -> NDArray[np.generic]:
    """Load one bounded non-object NPY sidecar member.

    Returns:
        A detached array.

    Raises:
        ValueError: If the member is oversized or invalid.
    """
    info = archive.getinfo(name)
    if info.file_size > maximum_size:
        msg = f"Trajectory sidecar member {name!r} exceeds its allowed size."
        raise ValueError(msg)
    try:
        value = np.load(io.BytesIO(archive.read(info)), allow_pickle=False)
    except (EOFError, OSError, TypeError, ValueError) as error:
        msg = f"Trajectory sidecar member {name!r} is not a safe NPY array."
        raise ValueError(msg) from error
    return np.asarray(value).copy()


def _decode_text_array(value: NDArray[np.generic], name: str) -> str:
    """Decode one strict UTF-8 ``uint8`` vector.

    Returns:
        The decoded text.

    Raises:
        ValueError: If shape, dtype, or encoding is invalid.
    """
    if value.ndim != 1 or value.dtype != np.dtype("uint8"):
        msg = f"Trajectory sidecar field {name!r} must be a one-dimensional uint8 array."
        raise ValueError(msg)
    try:
        return value.tobytes().decode()
    except UnicodeDecodeError as error:
        msg = f"Trajectory sidecar field {name!r} is not valid UTF-8."
        raise ValueError(msg) from error


def _decode_trajectory_archive(
    payload: bytes,
    *,
    expected_count: int,
) -> tuple[str, str, str, NDArray[np.generic], NDArray[np.generic]]:
    """Decode the bounded members of a trajectory archive.

    Returns:
        Format, run ID, training ID, repetition, and fidelity values.

    Raises:
        ValueError: If the archive structure or a member is invalid.
    """
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or frozenset(names) != _SIDECAR_MEMBERS:
            msg = "Trajectory sidecar members do not match the versioned format."
            raise ValueError(msg)
        if any(info.flag_bits & 0x1 for info in infos):
            msg = "Trajectory sidecar members must not be encrypted."
            raise ValueError(msg)
        text_limit = _SIDECAR_NPY_ALLOWANCE
        format_value = _decode_text_array(
            _load_npy_member(archive, "format.npy", maximum_size=text_limit),
            "format",
        )
        run_id = _decode_text_array(
            _load_npy_member(archive, "run_id.npy", maximum_size=text_limit),
            "run_id",
        )
        training_id = _decode_text_array(
            _load_npy_member(archive, "training_id.npy", maximum_size=text_limit),
            "training_id",
        )
        repetition = _load_npy_member(archive, "repetition.npy", maximum_size=text_limit)
        fidelities = _load_npy_member(
            archive,
            "fidelities.npy",
            maximum_size=expected_count * np.dtype("<f8").itemsize + _SIDECAR_NPY_ALLOWANCE,
        )
    return format_value, run_id, training_id, repetition, fidelities


def read_trajectory_sidecar(
    payload: bytes,
    *,
    expected_run_id: str,
    expected_training_id: str,
    expected_repetition: int,
    expected_count: int,
) -> tuple[float, ...]:
    """Verify and decode one compressed trajectory sidecar.

    Returns:
        The stored trajectory fidelities.

    Raises:
        TypeError: If ``payload`` is not bytes.
        ValueError: If the archive or its identity/count metadata is invalid.
    """
    if type(payload) is not bytes:
        msg = f"payload must be bytes, got {type(payload).__name__}."
        raise TypeError(msg)
    maximum_archive_size = expected_count * np.dtype("<f8").itemsize + 5 * _SIDECAR_NPY_ALLOWANCE
    if len(payload) > maximum_archive_size:
        msg = "Trajectory sidecar exceeds the size allowed by its expected sample count."
        raise ValueError(msg)
    try:
        decoded = _decode_trajectory_archive(payload, expected_count=expected_count)
    except (KeyError, OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        msg = "Trajectory sidecar could not be decoded safely."
        raise ValueError(msg) from error
    format_value, run_id, training_id, repetition, fidelities = decoded

    if format_value != TRAJECTORY_SIDECAR_FORMAT:
        msg = "Trajectory sidecar format is unsupported."
        raise ValueError(msg)
    if run_id != expected_run_id or training_id != expected_training_id:
        msg = "Trajectory sidecar identity does not match the result record."
        raise ValueError(msg)
    if repetition.shape != () or repetition.dtype != np.dtype("<u4") or int(repetition.item()) != expected_repetition:
        msg = "Trajectory sidecar repetition does not match the evaluation."
        raise ValueError(msg)
    if fidelities.shape != (expected_count,) or fidelities.dtype != np.dtype("<f8"):
        msg = "Trajectory sidecar fidelity shape or dtype does not match the evaluation budget."
        raise ValueError(msg)
    fidelity_values = cast("NDArray[np.float64]", fidelities)
    if not np.all(np.isfinite(fidelity_values)) or np.any(fidelity_values < 0.0) or np.any(fidelity_values > 1.0):
        msg = "Trajectory sidecar fidelities must be finite and lie in [0, 1]."
        raise ValueError(msg)
    return tuple(float(value) for value in fidelity_values)


class ReportingError(RuntimeError):
    """Base error for reporting-store reliability failures."""


class DuplicateRunError(ReportingError):
    """Raised when a run ID is written more than once without replacement."""


class ProvenanceMismatchError(ReportingError):
    """Raised when resume would reuse results from another implementation."""


class ArtifactVerificationError(ReportingError):
    """Raised when a stored checkpoint or sidecar fails verification."""


class BenchmarkReportStore:
    """Atomic canonical result stream with resumable derived artifacts."""

    def __init__(
        self,
        output_directory: Path,
        provenance: RunProvenance,
        *,
        overwrite: bool = False,
        allow_provenance_mismatch: bool = False,
    ) -> None:
        """Open or initialize one isolated benchmark output directory.

        Args:
            output_directory: Root for all managed benchmark outputs.
            provenance: Current software and Git fingerprint.
            overwrite: Remove all managed outputs before initialization.
            allow_provenance_mismatch: Explicitly allow reuse across different
                implementation fingerprints.

        Raises:
            TypeError: If an option has the wrong type.
            ProvenanceMismatchError: If stored provenance differs without an
                explicit override.
        """
        if not isinstance(output_directory, Path):
            msg = f"output_directory must be a pathlib.Path, got {type(output_directory).__name__}."
            raise TypeError(msg)
        if not isinstance(provenance, RunProvenance):
            msg = f"provenance must be a RunProvenance, got {type(provenance).__name__}."
            raise TypeError(msg)
        if type(overwrite) is not bool or type(allow_provenance_mismatch) is not bool:
            msg = "overwrite and allow_provenance_mismatch must be bool values."
            raise TypeError(msg)

        self.output_directory = output_directory.resolve()
        self.provenance = provenance
        self.results_jsonl_path = self.output_directory / RESULTS_JSONL_NAME
        self.results_csv_path = self.output_directory / RESULTS_CSV_NAME
        self.manifest_path = self.output_directory / MANIFEST_NAME
        self.checkpoint_directory = self.output_directory / CHECKPOINT_DIRECTORY
        self.trajectory_directory = self.output_directory / TRAJECTORY_DIRECTORY

        if overwrite:
            self._remove_managed_outputs()
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self._cleanup_temporary_files()
        manifest_history = self._read_manifest_history()
        records, recovered_partial = self._read_canonical_records()
        self._validate_unique_run_ids(records)
        self._records = records
        self._verify_all_artifacts()

        stored_provenances = self._record_provenances(records)
        stored_provenances.extend(manifest_history)
        unique_history = self._unique_provenances(stored_provenances)
        mismatch = any(item.fingerprint != provenance.fingerprint for item in unique_history)
        if mismatch and not allow_provenance_mismatch:
            msg = (
                "Stored benchmark results use a different Git/software provenance; "
                "set allow_provenance_mismatch=True to resume explicitly."
            )
            raise ProvenanceMismatchError(msg)
        self._provenance_history = self._unique_provenances([*unique_history, provenance])
        if recovered_partial or not self.results_jsonl_path.exists():
            self._write_jsonl()
        self._write_csv()
        self._write_manifest()

    @property
    def records(self) -> tuple[BenchmarkRecord, ...]:
        """Immutable snapshot of canonical records."""
        return tuple(self._records)

    @property
    def completed_run_ids(self) -> frozenset[str]:
        """Run IDs with successful canonical records."""
        return frozenset(record.run_id for record in self._records if isinstance(record, BenchmarkResult))

    @property
    def failed_run_ids(self) -> frozenset[str]:
        """Run IDs with failure canonical records."""
        return frozenset(record.run_id for record in self._records if isinstance(record, BenchmarkFailure))

    def is_completed(self, config_or_run_id: BenchmarkConfig | str) -> bool:
        """Return whether one stable run ID already has a successful row.

        Returns:
            Whether the run is complete.

        Raises:
            TypeError: If the argument is neither a config nor a run ID.
        """
        run_id = config_or_run_id.run_id if isinstance(config_or_run_id, BenchmarkConfig) else config_or_run_id
        if type(run_id) is not str:
            msg = f"config_or_run_id must be a BenchmarkConfig or str, got {type(config_or_run_id).__name__}."
            raise TypeError(msg)
        return run_id in self.completed_run_ids

    def write_success(
        self,
        *,
        config: BenchmarkConfig,
        artifact: StatePreparationTrainingArtifact,
        evaluation: IndependentEvaluation,
        optimization_wall_time_seconds: float,
        evaluation_wall_time_seconds: float,
        notes: str = "",
        replace: bool = False,
    ) -> BenchmarkResult:
        """Validate and atomically publish one successful result row.

        Returns:
            The canonical validated result.

        Raises:
            TypeError: If an argument has an unsupported type.
            ValueError: If the artifact, evaluation, and configuration disagree.
        """
        if not isinstance(config, BenchmarkConfig):
            msg = f"config must be a BenchmarkConfig, got {type(config).__name__}."
            raise TypeError(msg)
        self._require_writable_run(config.run_id, replace=replace)
        if not isinstance(artifact, StatePreparationTrainingArtifact):
            msg = f"artifact must be a StatePreparationTrainingArtifact, got {type(artifact).__name__}."
            raise TypeError(msg)
        if not isinstance(evaluation, IndependentEvaluation):
            msg = f"evaluation must be an IndependentEvaluation, got {type(evaluation).__name__}."
            raise TypeError(msg)
        if evaluation.training_id != artifact.training_id:
            msg = "Evaluation and training artifact identities differ."
            raise ValueError(msg)
        if evaluation.run_id != config.run_id:
            msg = "Evaluation and benchmark configuration run identities differ."
            raise ValueError(msg)

        checkpoint_relative = f"{CHECKPOINT_DIRECTORY}/{artifact.training_id}.npz"
        checkpoint_path = self._resolve_managed_relative(checkpoint_relative)
        sidecar_relative: str | None = None
        sidecar_payload: bytes | None = None
        sidecar_checksum: str | None = None
        if config.evaluation.store_trajectory_sidecar:
            if evaluation.trajectory_fidelities is None:
                msg = "A configured trajectory sidecar requires trajectory fidelities."
                raise ValueError(msg)
            sidecar_relative = f"{TRAJECTORY_DIRECTORY}/{config.run_id}.npz"
            sidecar_payload = create_trajectory_sidecar(
                run_id=config.run_id,
                training_id=artifact.training_id,
                repetition=evaluation.repetition,
                fidelities=evaluation.trajectory_fidelities,
            )
            sidecar_checksum = _sha256(sidecar_payload)

        result = BenchmarkResult(
            config=config,
            circuit_statistics=evaluation.circuit_statistics,
            train_fidelity=evaluation.train_fidelity,
            logical_test_noiseless_fidelity=evaluation.logical_test_noiseless_fidelity,
            native_pre_pruning_noiseless_fidelity=evaluation.native_pre_pruning_noiseless_fidelity,
            test_noiseless_fidelity=evaluation.test_noiseless_fidelity,
            test_noisy_fidelity=evaluation.test_noisy_fidelity,
            noisy_fidelity_standard_deviation=evaluation.noisy_fidelity_standard_deviation,
            noisy_fidelity_standard_error=evaluation.noisy_fidelity_standard_error,
            confidence_interval_lower=evaluation.confidence_interval_lower,
            confidence_interval_upper=evaluation.confidence_interval_upper,
            sampled_nonidentity_events=evaluation.sampled_nonidentity_events,
            optimization_wall_time_seconds=optimization_wall_time_seconds,
            evaluation_wall_time_seconds=evaluation_wall_time_seconds,
            software_versions=self.provenance.software_versions,
            git_commit=self.provenance.git_commit,
            git_dirty=self.provenance.git_dirty,
            git_diff_checksum=self.provenance.git_diff_checksum,
            parameter_checkpoint_path=checkpoint_relative,
            parameter_checkpoint_checksum=artifact.checkpoint_checksum,
            trajectory_sidecar_path=sidecar_relative,
            trajectory_sidecar_checksum=sidecar_checksum,
            notes=notes,
        )

        self._write_verified_artifact(
            checkpoint_path,
            artifact.checkpoint_payload,
            artifact.checkpoint_checksum,
            replace=False,
        )
        if sidecar_relative is not None and sidecar_payload is not None and sidecar_checksum is not None:
            self._write_verified_artifact(
                self._resolve_managed_relative(sidecar_relative),
                sidecar_payload,
                sidecar_checksum,
                replace=replace,
            )
        self._publish_record(result, replace=replace)
        return result

    def write_failure(
        self,
        *,
        config: BenchmarkConfig,
        failure_phase: str,
        exception: BaseException,
        wall_time_seconds: float,
        traceback: str | None = None,
        retryable: bool = False,
        attempt: int = 1,
        artifact: StatePreparationTrainingArtifact | None = None,
        notes: str = "",
        replace: bool = False,
    ) -> BenchmarkFailure:
        """Validate and atomically publish one explicit failure row.

        Returns:
            The canonical validated failure.

        Raises:
            TypeError: If an argument has an unsupported type.
        """
        if not isinstance(config, BenchmarkConfig):
            msg = f"config must be a BenchmarkConfig, got {type(config).__name__}."
            raise TypeError(msg)
        self._require_writable_run(config.run_id, replace=replace)
        checkpoint_relative: str | None = None
        checkpoint_checksum_value: str | None = None
        if artifact is not None:
            if not isinstance(artifact, StatePreparationTrainingArtifact):
                msg = f"artifact must be a StatePreparationTrainingArtifact or None, got {type(artifact).__name__}."
                raise TypeError(msg)
            checkpoint_relative = f"{CHECKPOINT_DIRECTORY}/{artifact.training_id}.npz"
            checkpoint_checksum_value = artifact.checkpoint_checksum

        failure = BenchmarkFailure.from_exception(
            config=config,
            failure_phase=failure_phase,
            exception=exception,
            traceback=traceback,
            retryable=retryable,
            attempt=attempt,
            wall_time_seconds=wall_time_seconds,
            software_versions=self.provenance.software_versions,
            git_commit=self.provenance.git_commit,
            git_dirty=self.provenance.git_dirty,
            git_diff_checksum=self.provenance.git_diff_checksum,
            parameter_checkpoint_path=checkpoint_relative,
            parameter_checkpoint_checksum=checkpoint_checksum_value,
            notes=notes,
        )
        if artifact is not None and checkpoint_relative is not None:
            self._write_verified_artifact(
                self._resolve_managed_relative(checkpoint_relative),
                artifact.checkpoint_payload,
                artifact.checkpoint_checksum,
                replace=False,
            )
        self._publish_record(failure, replace=replace)
        return failure

    def load_trajectory_fidelities(
        self,
        result: BenchmarkResult,
        *,
        repetition: int = 0,
    ) -> tuple[float, ...]:
        """Checksum-verify and load a result's optional trajectory sidecar.

        Returns:
            The stored trajectory fidelities.

        Raises:
            TypeError: If ``result`` is not a benchmark result.
            ArtifactVerificationError: If the sidecar is absent or invalid.
        """
        if not isinstance(result, BenchmarkResult):
            msg = f"result must be a BenchmarkResult, got {type(result).__name__}."
            raise TypeError(msg)
        if result.trajectory_sidecar_path is None or result.trajectory_sidecar_checksum is None:
            msg = "Benchmark result has no trajectory sidecar."
            raise ArtifactVerificationError(msg)
        path = self._resolve_managed_relative(result.trajectory_sidecar_path)
        payload = self._read_verified_artifact(path, result.trajectory_sidecar_checksum)
        try:
            return read_trajectory_sidecar(
                payload,
                expected_run_id=result.run_id,
                expected_training_id=Path(result.parameter_checkpoint_path).stem,
                expected_repetition=repetition,
                expected_count=result.config.evaluation.test_trajectories_or_shots,
            )
        except (TypeError, ValueError) as error:
            raise ArtifactVerificationError(str(error)) from error

    def _remove_managed_outputs(self) -> None:
        """Remove only versioned files and artifact directories."""
        for path in (self.results_jsonl_path, self.results_csv_path, self.manifest_path):
            path.unlink(missing_ok=True)
        for directory in (self.checkpoint_directory, self.trajectory_directory):
            if directory.exists():
                shutil.rmtree(directory)

    def _cleanup_temporary_files(self) -> None:
        """Remove abandoned atomic-write temporary files inside managed roots."""
        directories = (self.output_directory, self.checkpoint_directory, self.trajectory_directory)
        for directory in directories:
            if not directory.exists():
                continue
            for path in directory.glob(".*.tmp"):
                if path.is_file() or path.is_symlink():
                    path.unlink(missing_ok=True)

    def _resolve_managed_relative(self, relative_path: str) -> Path:
        """Resolve a schema-validated relative path under the output root.

        Returns:
            The resolved path under the managed output directory.

        Raises:
            ReportingError: If the path escapes the output directory.
        """
        path = (self.output_directory / relative_path).resolve()
        if not path.is_relative_to(self.output_directory):
            msg = f"Managed artifact path {relative_path!r} escapes the output directory."
            raise ReportingError(msg)
        return path

    def _read_manifest_history(self) -> list[RunProvenance]:
        """Read provenance history from an existing derived manifest.

        Returns:
            Validated provenance snapshots, or an empty list.

        Raises:
            ReportingError: If the existing manifest is invalid.
        """
        if not self.manifest_path.exists():
            return []
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            msg = f"Run manifest could not be decoded: {error}."
            raise ReportingError(msg) from error
        if not isinstance(data, dict) or data.get("manifest_format") != REPORT_MANIFEST_FORMAT:
            msg = "Run manifest format is unsupported."
            raise ReportingError(msg)
        history = data.get("provenance_history")
        if not isinstance(history, list):
            msg = "Run manifest provenance_history must be a list."
            raise ReportingError(msg)
        try:
            return [RunProvenance.from_dict(cast("Mapping[str, object]", item)) for item in history]
        except (TypeError, ValueError) as error:
            raise ReportingError(str(error)) from error

    def _read_canonical_records(self) -> tuple[list[BenchmarkRecord], bool]:
        """Read JSONL records, recovering only an unterminated partial tail.

        Returns:
            Validated records and whether a partial tail was recovered.

        Raises:
            ReportingError: If a complete canonical row is invalid.
        """
        if not self.results_jsonl_path.exists():
            return [], False
        try:
            payload = self.results_jsonl_path.read_bytes()
        except OSError as error:
            msg = f"Canonical result stream could not be read: {error}."
            raise ReportingError(msg) from error
        records: list[BenchmarkRecord] = []
        lines = payload.splitlines()
        recovered_partial = False
        for index, line in enumerate(lines):
            if not line.strip():
                msg = f"Canonical result stream contains an empty row at line {index + 1}."
                raise ReportingError(msg)
            try:
                records.append(benchmark_record_from_json(line.decode()))
            except (UnicodeDecodeError, TypeError, ValueError) as error:
                is_unterminated_tail = index == len(lines) - 1 and not payload.endswith(b"\n")
                if is_unterminated_tail:
                    recovered_partial = True
                    break
                msg = f"Canonical result stream is invalid at line {index + 1}: {error}."
                raise ReportingError(msg) from error
        if payload and not payload.endswith(b"\n") and not recovered_partial:
            recovered_partial = True
        return records, recovered_partial

    @staticmethod
    def _validate_unique_run_ids(records: Sequence[BenchmarkRecord]) -> None:
        """Reject duplicate stable run IDs in the canonical stream.

        Raises:
            DuplicateRunError: If a stable run ID appears more than once.
        """
        seen: set[str] = set()
        for record in records:
            if record.run_id in seen:
                msg = f"Canonical result stream contains duplicate run ID {record.run_id!r}."
                raise DuplicateRunError(msg)
            seen.add(record.run_id)

    @staticmethod
    def _record_provenances(records: Sequence[BenchmarkRecord]) -> list[RunProvenance]:
        """Return provenance snapshots carried by canonical records."""
        return [RunProvenance.from_record(record) for record in records]

    @staticmethod
    def _unique_provenances(provenances: Sequence[RunProvenance]) -> list[RunProvenance]:
        """Deduplicate provenance snapshots by canonical fingerprint.

        Returns:
            Unique snapshots ordered by fingerprint.
        """
        unique: dict[str, RunProvenance] = {}
        for provenance in provenances:
            unique.setdefault(provenance.fingerprint, provenance)
        return [unique[key] for key in sorted(unique)]

    def _verify_all_artifacts(self) -> None:
        """Verify every checkpoint and sidecar referenced by canonical rows."""
        for record in self._records:
            if record.parameter_checkpoint_path is not None and record.parameter_checkpoint_checksum is not None:
                path = self._resolve_managed_relative(record.parameter_checkpoint_path)
                self._read_verified_artifact(path, record.parameter_checkpoint_checksum)
            if isinstance(record, BenchmarkResult) and record.trajectory_sidecar_path is not None:
                assert record.trajectory_sidecar_checksum is not None
                path = self._resolve_managed_relative(record.trajectory_sidecar_path)
                self._read_verified_artifact(path, record.trajectory_sidecar_checksum)

    @staticmethod
    def _read_verified_artifact(path: Path, expected_checksum: str) -> bytes:
        """Read and checksum-verify one stored artifact.

        Returns:
            The verified artifact bytes.

        Raises:
            ArtifactVerificationError: If the artifact is unreadable or corrupt.
        """
        try:
            payload = path.read_bytes()
        except OSError as error:
            msg = f"Required benchmark artifact {path} could not be read: {error}."
            raise ArtifactVerificationError(msg) from error
        actual = _sha256(payload)
        if actual != expected_checksum:
            msg = f"Artifact checksum mismatch for {path}: expected {expected_checksum}, computed {actual}."
            raise ArtifactVerificationError(msg)
        return payload

    def _write_verified_artifact(
        self,
        path: Path,
        payload: bytes,
        expected_checksum: str,
        *,
        replace: bool,
    ) -> None:
        """Publish one artifact or verify an identical existing copy.

        Raises:
            ArtifactVerificationError: If an existing or new artifact is inconsistent.
        """
        _validate_checksum(expected_checksum, "expected_checksum")
        actual = _sha256(payload)
        if actual != expected_checksum:
            msg = f"Artifact payload checksum mismatch: expected {expected_checksum}, computed {actual}."
            raise ArtifactVerificationError(msg)
        if path.exists() and not replace:
            existing = self._read_verified_artifact(path, expected_checksum)
            if existing != payload:
                msg = f"Existing artifact {path} differs despite its expected checksum."
                raise ArtifactVerificationError(msg)
            return
        atomic_write_bytes(path, payload)

    def _require_writable_run(self, run_id: str, *, replace: bool) -> None:
        """Reject duplicate rows unless explicit replacement is requested.

        Raises:
            TypeError: If ``replace`` is not a boolean.
            DuplicateRunError: If the run exists and replacement is disabled.
        """
        if type(replace) is not bool:
            msg = f"replace must be a bool, got {type(replace).__name__}."
            raise TypeError(msg)
        if not replace and any(record.run_id == run_id for record in self._records):
            msg = f"Run ID {run_id!r} is already present in the canonical result stream."
            raise DuplicateRunError(msg)

    def _publish_record(self, record: BenchmarkRecord, *, replace: bool) -> None:
        """Atomically publish a validated record, then rebuild derivatives."""
        benchmark_record_from_json(record.to_json())
        if replace:
            records = [existing for existing in self._records if existing.run_id != record.run_id]
        else:
            records = list(self._records)
        records.append(record)
        self._validate_unique_run_ids(records)
        previous = self._records
        self._records = records
        try:
            self._write_jsonl()
        except Exception:
            self._records = previous
            raise
        self._write_csv()
        self._write_manifest()

    def _write_jsonl(self) -> None:
        """Atomically rewrite the canonical JSON Lines stream."""
        payload = "".join(f"{record.to_json()}\n" for record in self._records).encode()
        atomic_write_bytes(self.results_jsonl_path, payload)

    def _write_csv(self) -> None:
        """Atomically rebuild the convenience CSV from canonical records."""
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for record in self._records:
            writer.writerow(record.to_csv_row())
        atomic_write_bytes(self.results_csv_path, buffer.getvalue().encode())

    def _manifest_dict(self) -> dict[str, object]:
        """Return the complete derived run manifest."""
        successful = sorted(record.run_id for record in self._records if isinstance(record, BenchmarkResult))
        failed = sorted(record.run_id for record in self._records if isinstance(record, BenchmarkFailure))
        return {
            "manifest_format": REPORT_MANIFEST_FORMAT,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "canonical_result_stream": RESULTS_JSONL_NAME,
            "derived_csv": RESULTS_CSV_NAME,
            "checkpoint_directory": CHECKPOINT_DIRECTORY,
            "trajectory_directory": TRAJECTORY_DIRECTORY,
            "record_count": len(self._records),
            "successful_run_ids": successful,
            "failed_run_ids": failed,
            "active_provenance_fingerprint": self.provenance.fingerprint,
            "provenance_history": [item.to_dict() for item in self._provenance_history],
        }

    def _write_manifest(self) -> None:
        """Atomically rebuild the derived manifest."""
        atomic_write_bytes(self.manifest_path, f"{_canonical_json(self._manifest_dict())}\n".encode())


def read_csv_records(path: Path) -> tuple[BenchmarkRecord, ...]:
    """Read and validate a complete derived benchmark CSV.

    Returns:
        The success/failure records in file order.

    Raises:
        TypeError: If ``path`` is not a path.
        ValueError: If the CSV is unreadable or invalid.
    """
    if not isinstance(path, Path):
        msg = f"path must be a pathlib.Path, got {type(path).__name__}."
        raise TypeError(msg)
    try:
        with path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames != list(CSV_COLUMNS):
                msg = "Benchmark CSV header does not match the stable union schema."
                raise ValueError(msg)
            return tuple(benchmark_record_from_csv_row(row) for row in reader)
    except OSError as error:
        msg = f"Benchmark CSV could not be read: {error}."
        raise ValueError(msg) from error


__all__ = [
    "CHECKPOINT_DIRECTORY",
    "MANIFEST_NAME",
    "REPORT_MANIFEST_FORMAT",
    "RESULTS_CSV_NAME",
    "RESULTS_JSONL_NAME",
    "TRAJECTORY_DIRECTORY",
    "TRAJECTORY_SIDECAR_FORMAT",
    "ArtifactVerificationError",
    "BenchmarkRecord",
    "BenchmarkReportStore",
    "DuplicateRunError",
    "ProvenanceMismatchError",
    "ReportingError",
    "RunProvenance",
    "atomic_write_bytes",
    "capture_run_provenance",
    "create_trajectory_sidecar",
    "read_csv_records",
    "read_trajectory_sidecar",
]
