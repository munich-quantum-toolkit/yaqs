# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Explicitly opt-in execution of the pinned WP19 historical reproduction.

The command in this module is intentionally separate from ordinary benchmark
and test entry points.  It runs the five historical targets serially, limits
every discoverable numerical-library thread pool to one worker, and delegates
all scientific persistence to the WP18 artifact store, executor, and evaluator.
"""

# The private orchestration helpers share the strict public job contract, and
# Git is invoked only with fixed read-only arguments assembled in this module.
# ruff: noqa: BLE001, DOC201, DOC501, S603, S607, T201

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from filelock import FileLock, Timeout
from threadpoolctl import threadpool_info, threadpool_limits

from benchmarks.state_preparation.noise import (
    FIXED_RATE_NOISE_DEFINITION_VERSION,
    HISTORICAL_FIXED_RATE_NOISE_ID,
    create_historical_fixed_rate_noise_provider,
)
from benchmarks.state_preparation.reporting import atomic_write_bytes
from mqt.yaqs.optimization import (
    KROTOV_LEGACY_TRAJECTORY_SEED_DERIVATION_VERSION,
    KrotovTJMOptions,
    KrotovTruncation,
    noisy_state_preparation_metrics,
    sample_krotov_fixed_map_ensemble,
    state_preparation_metrics,
)

from .artifacts import PIPELINE_CONFIG_NAME, Phase2ArtifactStore
from .canonical import (
    canonical_checksum,
    canonical_json,
    load_canonical_json_object,
    seal_mapping,
    thaw_json,
    verify_sealed_mapping,
)
from .evaluator import MaterializedCircuitPayload, ParallelPhase2Evaluator, PipelineEvaluationMeasurement
from .execution import Phase2PipelineExecutor, PipelineExecutionFailure
from .historical_reproduction import (
    LEGACY_LAYERWISE_METHOD_ID,
    LEGACY_REPRODUCTION_TARGET_SEEDS,
    LayerwiseMaterializedCircuit,
    LegacyReproductionOutcome,
    LegacyReproductionReport,
    compare_legacy_reproduction,
    decode_layerwise_materialized_circuit,
    encode_layerwise_materialized_circuit,
)
from .layerwise_bmpd import (
    LEGACY_EVALUATION_SEED,
    LEGACY_EVALUATION_TRAJECTORY_COUNT,
    LayerwiseBMPDStageRunner,
    create_bmpd_circuit_binding,
    resolve_layerwise_bmpd_crn_legacy_v1_pipeline,
)
from .legacy_targets import load_legacy_target_collection
from .pipeline import (
    PipelineBenchmarkFailure,
    PipelineBenchmarkResult,
    PipelineEvaluationConfig,
    TrainingPipelineResult,
)
from .resumability import ExecutionSourceEntry, ResumabilityFingerprint, capture_resumability_fingerprint
from .validation import require_checksum, require_git_commit

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np
    from numpy.typing import NDArray

    from mqt.yaqs.optimization import (
        KrotovFixedMapEnsemble,
    )

    from .artifacts import StageFailureArtifact
    from .legacy_targets import LegacyMaterializedTarget


HISTORICAL_REPRODUCTION_JOB_SCHEMA_VERSION = "yaqs.state_preparation.phase2.historical_reproduction_job.v1"
HISTORICAL_REPRODUCTION_SOURCE_MANIFEST_SCHEMA_VERSION = (
    "yaqs.state_preparation.phase2.historical_reproduction_source_manifest.v1"
)
HISTORICAL_REPRODUCTION_JOB_ID = "wp19_layerwise_bmpd_crn_legacy_v1"
HISTORICAL_REPRODUCTION_REPORT_NAME = "historical_reproduction_report.json"
HISTORICAL_REPRODUCTION_RUNTIME_NAME = "historical_reproduction_runtime.json"
HISTORICAL_REPRODUCTION_SOURCE_MANIFEST_NAME = "historical_reproduction_source_manifest.json"
HISTORICAL_REPRODUCTION_LOCK_NAME = ".historical-reproduction.lock"
HISTORICAL_REPRODUCTION_LOCK_TIMEOUT_SECONDS = 30.0
HISTORICAL_REPRODUCTION_TARGET_DIRECTORY = "targets"
HISTORICAL_REPRODUCTION_THREAD_LIMIT = 1
HISTORICAL_REPRODUCTION_EVALUATION_WORKERS = 1
HISTORICAL_REPRODUCTION_SUCCESS_EXIT_CODE = 0
HISTORICAL_REPRODUCTION_DISCREPANCY_EXIT_CODE = 1
HISTORICAL_REPRODUCTION_FAILURE_EXIT_CODE = 2
HISTORICAL_REPRODUCTION_TOLERANCE = 1.0e-6
HISTORICAL_REPRODUCTION_TOLERANCE_RATIONALE = (
    "Absolute tolerance for floating-point drift across the pinned NumPy, SciPy, BLAS/LAPACK, and YAQS runtime; "
    "larger changes remain explicit discrepancies."
)

_RUNTIME_KEYS = frozenset({
    "schema_version",
    "job_id",
    "method_id",
    "target_seeds",
    "target_execution",
    "evaluation_workers",
    "numerical_thread_limit",
    "evaluation_noise_id",
    "evaluation_noise_definition_version",
    "evaluation_trajectory_count",
    "evaluation_seed",
    "trajectory_seed_derivation_version",
    "starting_commit",
    "python_implementation",
    "python_version",
    "platform",
    "machine",
    "dependency_versions",
    "thread_environment",
    "active_threadpools",
    "source_manifest_checksum",
    "content_checksum",
})
_SOURCE_MANIFEST_KEYS = frozenset({
    "schema_version",
    "starting_commit",
    "dependency_versions",
    "entries",
    "method_implementation_checksum",
    "lockfile_checksum",
    "study_protocol_checksum",
    "tracked_source_manifest_checksum",
    "dependency_versions_checksum",
    "content_checksum",
})
_THREAD_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
_DEPENDENCIES = ("mqt.yaqs", "numpy", "scipy", "threadpoolctl")
_SEALED_INPUT_PATHS = (
    Path("benchmarks/state_preparation/phase2/data/initial_preregistration_v1.json"),
    Path("benchmarks/state_preparation/phase2/data/legacy_evidence_audit_v1.json"),
    Path("benchmarks/state_preparation/phase2/data/legacy_tfim_targets_v1.json"),
)
_LOCKFILE_PATHS = (Path("pyproject.toml"), Path("uv.lock"))


@dataclass(frozen=True, slots=True)
class HistoricalReproductionSourceManifest:
    """Prefix-independent launch snapshot shared by all five WP19 targets."""

    starting_commit: str
    dependency_versions: Mapping[str, str]
    entries: tuple[ExecutionSourceEntry, ...]
    schema_version: str = field(default=HISTORICAL_REPRODUCTION_SOURCE_MANIFEST_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Normalize the snapshot through the strict WP18 fingerprint model."""
        probe = ResumabilityFingerprint(
            starting_commit=self.starting_commit,
            pipeline_prefix_id=f"phase2_pipeline_prefix_{'0' * 64}",
            dependency_versions=self.dependency_versions,
            entries=self.entries,
        )
        object.__setattr__(self, "starting_commit", probe.starting_commit)
        object.__setattr__(self, "dependency_versions", probe.dependency_versions)
        object.__setattr__(self, "entries", probe.entries)

    def _probe_fingerprint(self) -> ResumabilityFingerprint:
        """Return a validated fingerprint used only for derived group checksums."""
        return ResumabilityFingerprint(
            starting_commit=self.starting_commit,
            pipeline_prefix_id=f"phase2_pipeline_prefix_{'0' * 64}",
            dependency_versions=self.dependency_versions,
            entries=self.entries,
        )

    @property
    def method_implementation_checksum(self) -> str:
        """Checksum of every tracked execution-source entry."""
        return self._probe_fingerprint().method_implementation_checksum

    @property
    def lockfile_checksum(self) -> str:
        """Checksum of every tracked dependency lockfile entry."""
        return self._probe_fingerprint().lockfile_checksum

    @property
    def study_protocol_checksum(self) -> str:
        """Checksum of every tracked sealed-input entry."""
        return self._probe_fingerprint().study_protocol_checksum

    @property
    def tracked_source_manifest_checksum(self) -> str:
        """Checksum of all canonically ordered source-manifest entries."""
        return self._probe_fingerprint().tracked_source_manifest_checksum

    @property
    def dependency_versions_checksum(self) -> str:
        """Checksum of the launch dependency-version mapping."""
        return self._probe_fingerprint().dependency_versions_checksum

    def _content_dict(self) -> dict[str, object]:
        """Return all checksum-covered job snapshot fields."""
        return {
            "schema_version": self.schema_version,
            "starting_commit": self.starting_commit,
            "dependency_versions": dict(self.dependency_versions),
            "entries": [entry.to_dict() for entry in self.entries],
            "method_implementation_checksum": self.method_implementation_checksum,
            "lockfile_checksum": self.lockfile_checksum,
            "study_protocol_checksum": self.study_protocol_checksum,
            "tracked_source_manifest_checksum": self.tracked_source_manifest_checksum,
            "dependency_versions_checksum": self.dependency_versions_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the prefix-independent launch snapshot."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return the checksum-sealed canonical source manifest."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> HistoricalReproductionSourceManifest:
        """Decode and verify one strict source snapshot."""
        mapping = verify_sealed_mapping(
            data,
            expected_keys=_SOURCE_MANIFEST_KEYS,
            name="historical reproduction source manifest",
        )
        if mapping["schema_version"] != HISTORICAL_REPRODUCTION_SOURCE_MANIFEST_SCHEMA_VERSION:
            msg = "Historical reproduction source manifest uses an unsupported schema version."
            raise ValueError(msg)
        raw_entries = mapping["entries"]
        if isinstance(raw_entries, (str, bytes)) or not isinstance(raw_entries, Sequence):
            msg = "Historical reproduction source-manifest entries must be a sequence."
            raise TypeError(msg)
        result = cls(
            starting_commit=cast("str", mapping["starting_commit"]),
            dependency_versions=cast("Mapping[str, str]", mapping["dependency_versions"]),
            entries=tuple(ExecutionSourceEntry.from_dict(item) for item in raw_entries),
        )
        for name in (
            "method_implementation_checksum",
            "lockfile_checksum",
            "study_protocol_checksum",
            "tracked_source_manifest_checksum",
            "dependency_versions_checksum",
            "content_checksum",
        ):
            if mapping[name] != getattr(result, name):
                msg = f"Serialized {name} does not match the reconstructed historical source manifest."
                raise ValueError(msg)
        return result

    @classmethod
    def from_fingerprint(cls, fingerprint: ResumabilityFingerprint) -> HistoricalReproductionSourceManifest:
        """Discard only a target-specific pipeline prefix from a WP18 fingerprint."""
        if not isinstance(fingerprint, ResumabilityFingerprint):
            msg = "fingerprint must be a ResumabilityFingerprint."
            raise TypeError(msg)
        return cls(
            starting_commit=fingerprint.starting_commit,
            dependency_versions=fingerprint.dependency_versions,
            entries=fingerprint.entries,
        )

    @classmethod
    def from_json(cls, payload: str) -> HistoricalReproductionSourceManifest:
        """Decode canonical JSON into one verified source snapshot."""
        return cls.from_dict(load_canonical_json_object(payload))

    def require_fingerprint_match(self, fingerprint: ResumabilityFingerprint) -> None:
        """Require every prefix-independent fingerprint field to match this launch snapshot."""
        if not isinstance(fingerprint, ResumabilityFingerprint):
            msg = "fingerprint must be a ResumabilityFingerprint."
            raise TypeError(msg)
        mismatches = tuple(
            name
            for name, stored, current in (
                ("starting_commit", self.starting_commit, fingerprint.starting_commit),
                ("dependency_versions", dict(self.dependency_versions), dict(fingerprint.dependency_versions)),
                ("entries", self.entries, fingerprint.entries),
                (
                    "method_implementation",
                    self.method_implementation_checksum,
                    fingerprint.method_implementation_checksum,
                ),
                ("lockfiles", self.lockfile_checksum, fingerprint.lockfile_checksum),
                ("study_protocol", self.study_protocol_checksum, fingerprint.study_protocol_checksum),
                (
                    "tracked_source_manifest",
                    self.tracked_source_manifest_checksum,
                    fingerprint.tracked_source_manifest_checksum,
                ),
                (
                    "dependency_versions_checksum",
                    self.dependency_versions_checksum,
                    fingerprint.dependency_versions_checksum,
                ),
            )
            if stored != current
        )
        if mismatches:
            msg = f"Historical reproduction launch snapshot changed in: {', '.join(mismatches)}."
            raise ValueError(msg)

    def fingerprint_for_pipeline(self, pipeline_prefix_id: str) -> ResumabilityFingerprint:
        """Derive a WP18 fingerprint while changing only its pipeline prefix."""
        return ResumabilityFingerprint(
            starting_commit=self.starting_commit,
            pipeline_prefix_id=pipeline_prefix_id,
            dependency_versions=self.dependency_versions,
            entries=self.entries,
        )


@dataclass(frozen=True, slots=True)
class HistoricalTargetRequest:
    """One fixed target request passed to the serial target executor."""

    target_seed: int
    output_directory: Path
    repository_root: Path
    source_manifest: HistoricalReproductionSourceManifest
    resumability_fingerprint: ResumabilityFingerprint
    resume: bool
    overwrite: bool

    def __post_init__(self) -> None:
        """Validate and defensively freeze execution inputs."""
        if type(self.target_seed) is not int or self.target_seed not in LEGACY_REPRODUCTION_TARGET_SEEDS:
            msg = f"target_seed must be one of {LEGACY_REPRODUCTION_TARGET_SEEDS!r}."
            raise ValueError(msg)
        for name in ("output_directory", "repository_root"):
            if not isinstance(getattr(self, name), Path):
                msg = f"{name} must be a pathlib.Path."
                raise TypeError(msg)
        if not isinstance(self.source_manifest, HistoricalReproductionSourceManifest):
            msg = "source_manifest must be a HistoricalReproductionSourceManifest."
            raise TypeError(msg)
        self.source_manifest.require_fingerprint_match(self.resumability_fingerprint)
        pipeline = resolve_layerwise_bmpd_crn_legacy_v1_pipeline(self.target_seed)
        expected_prefix = pipeline.prefix_id(len(pipeline.stages) - 1)
        if self.resumability_fingerprint.pipeline_prefix_id != expected_prefix:
            msg = "resumability_fingerprint does not identify the requested target pipeline."
            raise ValueError(msg)
        if type(self.resume) is not bool or type(self.overwrite) is not bool or (self.resume and self.overwrite):
            msg = "resume and overwrite must be mutually exclusive bool values."
            raise ValueError(msg)


TargetExecutor = Callable[[HistoricalTargetRequest], LegacyReproductionOutcome]


class HistoricalReproductionConcurrentExecutionError(RuntimeError):
    """Raised when another process owns the complete WP19 job output."""


def _run_git(repository_root: Path, *arguments: str) -> str:
    """Run one read-only Git query and return stripped UTF-8 output."""
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        msg = f"Could not resolve pinned Git provenance for the historical reproduction: {error}."
        raise RuntimeError(msg) from error
    return result.stdout.decode("utf-8", errors="strict").strip()


def _dependency_versions() -> Mapping[str, str]:
    """Return the exact dependency versions included in every target fingerprint."""
    return MappingProxyType({
        **{name: version(name) for name in _DEPENDENCIES},
        "python": platform.python_version(),
    })


def _tracked_execution_sources(repository_root: Path) -> tuple[Path, ...]:
    """Return all tracked Python sources in the scientific benchmark and YAQS core."""
    payload = _run_git(
        repository_root,
        "ls-files",
        "-z",
        "--",
        "benchmarks/state_preparation",
        "src/mqt/yaqs",
    )
    paths = tuple(Path(item) for item in payload.split("\0") if item and Path(item).suffix == ".py")
    if not paths:
        msg = "No tracked scientific execution sources were found."
        raise RuntimeError(msg)
    return paths


def _capture_target_resumability_fingerprint(
    *,
    repository_root: Path,
    output_directory: Path,
    starting_commit: str,
    dependency_versions: Mapping[str, str],
    execution_source_paths: Sequence[Path],
    target_seed: int,
) -> ResumabilityFingerprint:
    """Capture the active WP18 fingerprint for one resolved target pipeline."""
    pipeline = resolve_layerwise_bmpd_crn_legacy_v1_pipeline(target_seed)
    return capture_resumability_fingerprint(
        repository_root,
        output_root=output_directory,
        starting_commit=starting_commit,
        pipeline_prefix_id=pipeline.prefix_id(len(pipeline.stages) - 1),
        dependency_versions=dependency_versions,
        execution_source_paths=execution_source_paths,
        lockfile_paths=_LOCKFILE_PATHS,
        sealed_input_paths=_SEALED_INPUT_PATHS,
    )


def _normalized_threadpools() -> tuple[dict[str, object], ...]:
    """Return deterministic active numerical-library thread-pool provenance."""
    fields = (
        "user_api",
        "internal_api",
        "prefix",
        "filepath",
        "version",
        "num_threads",
        "threading_layer",
        "architecture",
    )
    normalized = tuple({name: entry.get(name) for name in fields} for entry in threadpool_info())
    if any(entry["num_threads"] != HISTORICAL_REPRODUCTION_THREAD_LIMIT for entry in normalized):
        msg = "A numerical-library thread pool did not honor the pinned one-thread limit."
        raise RuntimeError(msg)
    return tuple(
        sorted(
            normalized,
            key=lambda entry: (
                str(entry["internal_api"]),
                str(entry["prefix"]),
                str(entry["filepath"]),
            ),
        )
    )


def capture_historical_reproduction_runtime(
    *,
    starting_commit: str,
    dependency_versions: Mapping[str, str],
    source_manifest_checksum: str,
) -> dict[str, object]:
    """Capture and checksum-seal the active serial, thread-pinned runtime."""
    commit = require_git_commit(starting_commit, "starting_commit")
    dependencies = dict(sorted(dependency_versions.items()))
    if any(
        type(key) is not str or type(value) is not str or not key or not value for key, value in dependencies.items()
    ):
        msg = "dependency_versions must map nonempty strings to nonempty strings."
        raise ValueError(msg)
    manifest_checksum = require_checksum(source_manifest_checksum, "source_manifest_checksum")
    return seal_mapping({
        "schema_version": HISTORICAL_REPRODUCTION_JOB_SCHEMA_VERSION,
        "job_id": HISTORICAL_REPRODUCTION_JOB_ID,
        "method_id": LEGACY_LAYERWISE_METHOD_ID,
        "target_seeds": list(LEGACY_REPRODUCTION_TARGET_SEEDS),
        "target_execution": "serial",
        "evaluation_workers": HISTORICAL_REPRODUCTION_EVALUATION_WORKERS,
        "numerical_thread_limit": HISTORICAL_REPRODUCTION_THREAD_LIMIT,
        "evaluation_noise_id": HISTORICAL_FIXED_RATE_NOISE_ID,
        "evaluation_noise_definition_version": FIXED_RATE_NOISE_DEFINITION_VERSION,
        "evaluation_trajectory_count": LEGACY_EVALUATION_TRAJECTORY_COUNT,
        "evaluation_seed": LEGACY_EVALUATION_SEED,
        "trajectory_seed_derivation_version": KROTOV_LEGACY_TRAJECTORY_SEED_DERIVATION_VERSION,
        "starting_commit": commit,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependency_versions": dependencies,
        "thread_environment": {name: os.environ.get(name) for name in _THREAD_ENVIRONMENT_NAMES},
        "active_threadpools": list(_normalized_threadpools()),
        "source_manifest_checksum": manifest_checksum,
    })


def _verified_runtime_document(document: object) -> Mapping[str, object]:
    """Verify one runtime document and all frozen WP19 execution fields."""
    mapping = verify_sealed_mapping(
        document,
        expected_keys=_RUNTIME_KEYS,
        name="historical reproduction runtime",
    )
    fixed = {
        "schema_version": HISTORICAL_REPRODUCTION_JOB_SCHEMA_VERSION,
        "job_id": HISTORICAL_REPRODUCTION_JOB_ID,
        "method_id": LEGACY_LAYERWISE_METHOD_ID,
        "target_seeds": LEGACY_REPRODUCTION_TARGET_SEEDS,
        "target_execution": "serial",
        "evaluation_workers": HISTORICAL_REPRODUCTION_EVALUATION_WORKERS,
        "numerical_thread_limit": HISTORICAL_REPRODUCTION_THREAD_LIMIT,
        "evaluation_noise_id": HISTORICAL_FIXED_RATE_NOISE_ID,
        "evaluation_noise_definition_version": FIXED_RATE_NOISE_DEFINITION_VERSION,
        "evaluation_trajectory_count": LEGACY_EVALUATION_TRAJECTORY_COUNT,
        "evaluation_seed": LEGACY_EVALUATION_SEED,
        "trajectory_seed_derivation_version": KROTOV_LEGACY_TRAJECTORY_SEED_DERIVATION_VERSION,
    }
    for name, expected in fixed.items():
        if mapping[name] != expected:
            msg = f"Historical runtime field {name!r} differs from the pinned WP19 job."
            raise ValueError(msg)
    require_git_commit(mapping["starting_commit"], "starting_commit")
    for name in ("dependency_versions", "thread_environment"):
        if not isinstance(mapping[name], Mapping):
            msg = f"{name} must be a mapping."
            raise TypeError(msg)
    if not isinstance(mapping["active_threadpools"], Sequence):
        msg = "active_threadpools must be a sequence."
        raise TypeError(msg)
    require_checksum(mapping["source_manifest_checksum"], "source_manifest_checksum")
    return mapping


def _safe_output_root(output_root: Path, repository_root: Path) -> Path:
    """Resolve a dedicated output root and reject broad or aliased targets."""
    if not isinstance(output_root, Path) or not isinstance(repository_root, Path):
        msg = "output_root and repository_root must be pathlib.Path values."
        raise TypeError(msg)
    if output_root.is_symlink():
        msg = "The historical reproduction output root must not be a symbolic link."
        raise ValueError(msg)
    root = output_root.resolve()
    repository = repository_root.resolve()
    broad_roots = {Path(root.anchor), Path.home().resolve(), repository, *repository.parents}
    if root in broad_roots:
        msg = "Choose a dedicated output directory, not a filesystem, home, repository, or repository-ancestor root."
        raise ValueError(msg)
    git_directory = repository / ".git"
    if root == git_directory or git_directory in root.parents:
        msg = "The historical reproduction output cannot be placed inside .git."
        raise ValueError(msg)
    if root.exists() and not root.is_dir():
        msg = "The historical reproduction output root must be a directory."
        raise ValueError(msg)
    return root


@contextmanager
def _exclusive_historical_job_lock(output_root: Path, repository_root: Path) -> Iterator[Path]:
    """Hold one process-level lock across the complete root orchestration.

    Yields:
        The validated dedicated output root while the job lock is owned.
    """
    root = _safe_output_root(output_root, repository_root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / HISTORICAL_REPRODUCTION_LOCK_NAME
    if lock_path.is_symlink() or (lock_path.exists() and not lock_path.is_file()):
        msg = "Historical reproduction job lock must be a regular file, never a symbolic link."
        raise ValueError(msg)
    lock = FileLock(lock_path, timeout=HISTORICAL_REPRODUCTION_LOCK_TIMEOUT_SECONDS)
    try:
        lock.acquire()
    except Timeout as error:
        msg = "Another process currently owns this historical reproduction job; retry after it finishes."
        raise HistoricalReproductionConcurrentExecutionError(msg) from error
    try:
        yield root
    finally:
        lock.release()


def _prepare_output_root(
    output_root: Path,
    repository_root: Path,
    *,
    resume: bool,
    overwrite: bool,
) -> Path:
    """Apply non-destructive root-level new/resume/overwrite semantics."""
    if type(resume) is not bool or type(overwrite) is not bool:
        msg = "resume and overwrite must be bool values."
        raise TypeError(msg)
    if resume and overwrite:
        msg = "resume and overwrite are mutually exclusive."
        raise ValueError(msg)
    root = _safe_output_root(output_root, repository_root)
    runtime_path = root / HISTORICAL_REPRODUCTION_RUNTIME_NAME
    report_path = root / HISTORICAL_REPRODUCTION_REPORT_NAME
    source_manifest_path = root / HISTORICAL_REPRODUCTION_SOURCE_MANIFEST_NAME
    for path, label in (
        (runtime_path, "runtime"),
        (report_path, "report"),
        (source_manifest_path, "source manifest"),
    ):
        if path.is_symlink() or (path.exists() and not path.is_file()):
            msg = f"The managed historical {label} path must be a regular file."
            raise ValueError(msg)
    if resume:
        if not root.is_dir() or not runtime_path.is_file() or not source_manifest_path.is_file():
            msg = "resume=True requires an existing verified historical reproduction manifest and runtime."
            raise ValueError(msg)
    elif (
        not overwrite
        and root.exists()
        and any(path.name != HISTORICAL_REPRODUCTION_LOCK_NAME for path in root.iterdir())
    ):
        msg = "Existing historical reproduction output requires resume=True or overwrite=True."
        raise ValueError(msg)
    root.mkdir(parents=True, exist_ok=True)
    target_root = root / HISTORICAL_REPRODUCTION_TARGET_DIRECTORY
    if target_root.is_symlink() or (target_root.exists() and not target_root.is_dir()):
        msg = "The managed historical target directory must be a real directory."
        raise ValueError(msg)
    target_root.mkdir(exist_ok=True)
    if overwrite:
        report_path.unlink(missing_ok=True)
    return root


def _write_or_verify_source_manifest(
    root: Path,
    manifest: HistoricalReproductionSourceManifest,
    *,
    resume: bool,
) -> None:
    """Persist one launch snapshot or require exact equality on resume."""
    path = root / HISTORICAL_REPRODUCTION_SOURCE_MANIFEST_NAME
    if path.is_symlink() or (path.exists() and not path.is_file()):
        msg = "The managed historical source-manifest path must be a regular file."
        raise ValueError(msg)
    if resume:
        try:
            stored = HistoricalReproductionSourceManifest.from_json(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, TypeError, ValueError) as error:
            msg = f"Could not verify the retained historical source manifest: {error}."
            raise ValueError(msg) from error
        if stored != manifest:
            msg = "The active launch snapshot differs from the retained historical source manifest."
            raise ValueError(msg)
        return
    atomic_write_bytes(path, f"{manifest.to_json()}\n".encode())


def _require_runtime_manifest_binding(
    runtime: Mapping[str, object],
    manifest: HistoricalReproductionSourceManifest,
) -> None:
    """Require one runtime to identify the exact launch snapshot it records."""
    if runtime["source_manifest_checksum"] != manifest.content_checksum:
        msg = "Historical runtime does not bind the active source manifest."
        raise ValueError(msg)
    if runtime["starting_commit"] != manifest.starting_commit:
        msg = "Historical runtime and source manifest disagree on the starting commit."
        raise ValueError(msg)
    if thaw_json(runtime["dependency_versions"]) != thaw_json(dict(manifest.dependency_versions)):
        msg = "Historical runtime and source manifest disagree on dependency versions."
        raise ValueError(msg)


def _write_or_verify_runtime(
    root: Path,
    runtime: Mapping[str, object],
    source_manifest: HistoricalReproductionSourceManifest,
    *,
    resume: bool,
) -> None:
    """Persist a new runtime root or require exact equality on resume."""
    path = root / HISTORICAL_REPRODUCTION_RUNTIME_NAME
    if path.is_symlink() or (path.exists() and not path.is_file()):
        msg = "The managed historical runtime path must be a regular file."
        raise ValueError(msg)
    verified = _verified_runtime_document(runtime)
    _require_runtime_manifest_binding(verified, source_manifest)
    if resume:
        try:
            stored = _verified_runtime_document(load_canonical_json_object(path.read_text(encoding="utf-8")))
        except (OSError, UnicodeError, TypeError, ValueError) as error:
            msg = f"Could not verify the retained historical runtime: {error}."
            raise ValueError(msg) from error
        if thaw_json(stored) != thaw_json(verified):
            msg = "The active pinned runtime differs from the retained historical reproduction runtime."
            raise ValueError(msg)
        _require_runtime_manifest_binding(stored, source_manifest)
        return
    atomic_write_bytes(path, f"{canonical_json(verified)}\n".encode())


def _require_report_artifact_bindings(
    report: LegacyReproductionReport,
    runtime: Mapping[str, object],
    source_manifest: HistoricalReproductionSourceManifest,
) -> None:
    """Require the report, runtime, rows, and per-target WP18 roots to agree."""
    runtime_checksum = require_checksum(runtime["content_checksum"], "runtime.content_checksum")
    if report.source_manifest_checksum != source_manifest.content_checksum:
        msg = "Historical report does not bind the active source manifest."
        raise ValueError(msg)
    if report.runtime_checksum != runtime_checksum:
        msg = "Historical report does not bind the active runtime."
        raise ValueError(msg)
    _require_runtime_manifest_binding(runtime, source_manifest)
    for comparison in report.target_comparisons:
        pipeline = resolve_layerwise_bmpd_crn_legacy_v1_pipeline(comparison.target_seed)
        expected = source_manifest.fingerprint_for_pipeline(
            pipeline.prefix_id(len(pipeline.stages) - 1)
        ).content_checksum
        if comparison.outcome.runtime_fingerprint_checksum != expected:
            msg = (
                f"Historical report source record for seed {comparison.target_seed} does not bind its WP18 fingerprint."
            )
            raise ValueError(msg)


def _load_verified_provenance(
    root: Path,
) -> tuple[HistoricalReproductionSourceManifest, Mapping[str, object]]:
    """Load and cross-check the persisted launch manifest and runtime."""
    manifest_path = root / HISTORICAL_REPRODUCTION_SOURCE_MANIFEST_NAME
    runtime_path = root / HISTORICAL_REPRODUCTION_RUNTIME_NAME
    for path, label in ((manifest_path, "source manifest"), (runtime_path, "runtime")):
        if path.is_symlink() or not path.is_file():
            msg = f"Historical reproduction {label} must be a regular file."
            raise ValueError(msg)
    manifest = HistoricalReproductionSourceManifest.from_json(manifest_path.read_text(encoding="utf-8"))
    runtime = _verified_runtime_document(load_canonical_json_object(runtime_path.read_text(encoding="utf-8")))
    _require_runtime_manifest_binding(runtime, manifest)
    return manifest, runtime


def verify_historical_reproduction_artifacts(output_root: Path) -> LegacyReproductionReport:
    """Load and verify the persisted manifest-runtime-report binding chain."""
    if not isinstance(output_root, Path):
        msg = "output_root must be a pathlib.Path."
        raise TypeError(msg)
    root = output_root.resolve()
    manifest, runtime = _load_verified_provenance(root)
    report_path = root / HISTORICAL_REPRODUCTION_REPORT_NAME
    if report_path.is_symlink() or not report_path.is_file():
        msg = "Historical reproduction report must be a regular file."
        raise ValueError(msg)
    report = LegacyReproductionReport.from_json(report_path.read_text(encoding="utf-8"))
    _require_report_artifact_bindings(report, runtime, manifest)
    return report


def _verify_retained_report_if_present(
    root: Path,
    runtime: Mapping[str, object],
    source_manifest: HistoricalReproductionSourceManifest,
) -> None:
    """Reject a tampered completed report before a resume can replace it."""
    path = root / HISTORICAL_REPRODUCTION_REPORT_NAME
    if not path.exists():
        return
    if path.is_symlink() or not path.is_file():
        msg = "The retained historical report must be a regular file."
        raise ValueError(msg)
    try:
        report = LegacyReproductionReport.from_json(path.read_text(encoding="utf-8"))
        _require_report_artifact_bindings(report, runtime, source_manifest)
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        msg = f"Could not verify the retained historical report: {error}."
        raise ValueError(msg) from error


def _measure_call(callback: Callable[[], object]) -> tuple[object, float, int]:
    """Run one scientific callback and return value, elapsed time, and peak bytes."""
    owns_tracing = not tracemalloc.is_tracing()
    if owns_tracing:
        tracemalloc.start()
        tracemalloc.reset_peak()
    before_current, _ = tracemalloc.get_traced_memory()
    started = time.perf_counter()
    try:
        result = callback()
    finally:
        elapsed = time.perf_counter() - started
        current, peak = tracemalloc.get_traced_memory()
        measured_peak = max(0, current - before_current, peak - before_current)
        if owns_tracing:
            tracemalloc.stop()
    return result, elapsed, measured_peak


def _provider_checksum(provider: object) -> str:
    """Return one provider checksum without relying on a private adapter helper."""
    checksum = getattr(provider, "content_checksum", None)
    if type(checksum) is not str:
        msg = "The historical noise provider must expose a stable content_checksum."
        raise TypeError(msg)
    return checksum


def evaluate_historical_materialized_circuit(
    config: PipelineEvaluationConfig,
    runtime_circuit: object,
    target: LegacyMaterializedTarget,
) -> PipelineEvaluationMeasurement:
    """Evaluate one selected legacy circuit with exact seeds zero through 499."""
    if not isinstance(runtime_circuit, LayerwiseMaterializedCircuit):
        msg = "The historical evaluator requires a verified LayerwiseMaterializedCircuit."
        raise TypeError(msg)
    binding = runtime_circuit.circuit_binding
    circuit = binding.circuit
    theta = runtime_circuit.selected_parameters
    target_vector = target.state_vector_copy()
    provider = create_historical_fixed_rate_noise_provider()
    provider_checksum = _provider_checksum(provider)
    truncation = KrotovTruncation(
        max_bond_dim=config.max_bond_dimension,
        svd_threshold=config.svd_threshold,
        trunc_mode=config.truncation_mode,
        min_bond_dim=config.min_bond_dimension,
    )
    tjm_options = KrotovTJMOptions(
        num_trajectories=LEGACY_EVALUATION_TRAJECTORY_COUNT,
        random_seed=LEGACY_EVALUATION_SEED,
        dt=1.0,
        apply_noise_to="all",
        noisy_gate_indices=binding.noisy_gate_indices,
        trajectory_update="independent",
        differentiate_jump_normalization=False,
        use_crn=False,
    )

    def run() -> tuple[float, tuple[float, ...], KrotovFixedMapEnsemble]:
        ensemble = sample_krotov_fixed_map_ensemble(
            circuit,
            theta,
            None,
            truncation,
            provider,
            tjm_options,
            role="pilot_evaluation",
            resolved_seed=LEGACY_EVALUATION_SEED,
            stage_index=5,
            stage_id="legacy_evaluation",
            stage_configuration_checksum=config.configuration_checksum,
            circuit_checksum=config.materialized_circuit_checksum,
            provider_checksum=provider_checksum,
            ensemble_index=0,
            refresh_index=0,
            global_iteration_start=0,
            legacy_linear_seed=True,
            legacy_compact_replay=False,
        )
        _, noiseless_fidelity = state_preparation_metrics(
            circuit,
            theta,
            target_vector,
            truncation=truncation,
        )
        _, _, trajectory_fidelities = noisy_state_preparation_metrics(
            circuit,
            theta,
            target_vector,
            None,
            tjm_options,
            truncation=truncation,
            iteration=0,
            fixed_noise_maps=ensemble.replay_maps(),
            noise_provider=provider,
        )
        return noiseless_fidelity, tuple(trajectory_fidelities), ensemble

    measured, wall_time, peak_memory = _measure_call(run)
    noiseless_fidelity, trajectory_fidelities, ensemble = cast(
        "tuple[float, tuple[float, ...], KrotovFixedMapEnsemble]",
        measured,
    )
    gate_count = len(circuit.gates)
    return PipelineEvaluationMeasurement(
        noiseless_fidelity=noiseless_fidelity,
        trajectory_fidelities=trajectory_fidelities,
        sampled_nonidentity_events=ensemble.nonidentity_event_count,
        provider_checksum=provider_checksum,
        normalized_work={
            "objective_evaluations": 2,
            "gradient_evaluations": 0,
            "training_trajectories": 0,
            "checkpoint_validation_trajectories": 0,
            "test_trajectories": LEGACY_EVALUATION_TRAJECTORY_COUNT,
            "trajectory_gate_applications": 2 * LEGACY_EVALUATION_TRAJECTORY_COUNT * gate_count,
        },
        fixed_map_ensembles=(ensemble,),
        wall_time_seconds=wall_time,
        peak_memory_bytes=peak_memory,
    )


def _stage_failure_outcome(target_seed: int, failure: StageFailureArtifact) -> LegacyReproductionOutcome:
    """Project one persisted training failure into the five-row comparison."""
    return LegacyReproductionOutcome(
        target_seed=target_seed,
        status="failure",
        computed_fidelity=None,
        source_record_id=failure.failure_id,
        source_record_checksum=failure.content_checksum,
        runtime_fingerprint_checksum=failure.runtime_fingerprint_checksum,
        failure_type=failure.exception_type,
        failure_message=f"Training stage {failure.stage_id} ({failure.phase}) failed: {failure.message}",
    )


def _unexpected_failure_outcome(
    target_seed: int,
    error: Exception,
    runtime_fingerprint_checksum: str,
) -> LegacyReproductionOutcome:
    """Preserve a pre-evaluation orchestration failure without inventing a value."""
    failure_type = type(error).__name__
    message = str(error) or failure_type
    checksum = canonical_checksum({
        "schema_version": "yaqs.state_preparation.phase2.historical_job_failure.v1",
        "target_seed": target_seed,
        "failure_type": failure_type,
        "message": message,
        "runtime_fingerprint_checksum": runtime_fingerprint_checksum,
    })
    return LegacyReproductionOutcome(
        target_seed=target_seed,
        status="failure",
        computed_fidelity=None,
        source_record_id=f"wp19_job_failure_seed_{target_seed}",
        source_record_checksum=checksum,
        runtime_fingerprint_checksum=runtime_fingerprint_checksum,
        failure_type=failure_type,
        failure_message=message,
    )


def _materialized_payload(
    pipeline: TrainingPipelineResult,
    parameters: NDArray[np.float64],
) -> MaterializedCircuitPayload:
    """Encode one final depth-four circuit and retain measured codec resources."""
    binding = create_bmpd_circuit_binding(pipeline.config.qubit_count, 4)
    measured, wall_time, peak_memory = _measure_call(
        lambda: encode_layerwise_materialized_circuit(binding, parameters),
    )
    return MaterializedCircuitPayload(
        serialized_bytes=cast("bytes", measured),
        wall_time_seconds=wall_time,
        peak_memory_bytes=peak_memory,
    )


def validate_historical_evaluation_record(
    *,
    target_seed: int,
    record: PipelineBenchmarkResult | PipelineBenchmarkFailure,
    planned_config: PipelineEvaluationConfig,
    pipeline: TrainingPipelineResult,
    expected_runtime_fingerprint_checksum: str,
) -> PipelineBenchmarkResult | PipelineBenchmarkFailure:
    """Require an evaluation record to be the exact planned historical row.

    This boundary prevents a valid row from another target, repetition, noise
    profile, or trajectory budget from being relabelled by the five-row report.
    """
    if type(target_seed) is not int or target_seed not in LEGACY_REPRODUCTION_TARGET_SEEDS:
        msg = f"target_seed must be one of {LEGACY_REPRODUCTION_TARGET_SEEDS!r}."
        raise ValueError(msg)
    if not isinstance(record, (PipelineBenchmarkResult, PipelineBenchmarkFailure)):
        msg = "record must be a PipelineBenchmarkResult or PipelineBenchmarkFailure."
        raise TypeError(msg)
    if not isinstance(planned_config, PipelineEvaluationConfig):
        msg = "planned_config must be a PipelineEvaluationConfig."
        raise TypeError(msg)
    if not isinstance(pipeline, TrainingPipelineResult):
        msg = "pipeline must be a TrainingPipelineResult."
        raise TypeError(msg)
    expected_runtime = require_checksum(
        expected_runtime_fingerprint_checksum,
        "expected_runtime_fingerprint_checksum",
    )

    pipeline_config = pipeline.config
    expected_target_id = f"legacy_tfim_seed_{target_seed}"
    if (
        pipeline_config.method_id != LEGACY_LAYERWISE_METHOD_ID
        or pipeline_config.target_namespace != "legacy_reproduction"
        or pipeline_config.target_instance_id != expected_target_id
        or pipeline_config.qubit_count != 8
        or pipeline_config.optimization_seed != target_seed
        or pipeline_config.data_role != "secondary_benchmark"
    ):
        msg = "The completed pipeline does not identify the requested historical target and method."
        raise ValueError(msg)
    if record.runtime_fingerprint_checksum != expected_runtime:
        msg = "The evaluator record does not bind the verified target WP18 fingerprint."
        raise ValueError(msg)

    planned_config.validate_against_pipeline(pipeline)
    record.config.validate_against_pipeline(pipeline)
    if (
        record.evaluation_row_id != planned_config.evaluation_row_id
        or record.config.configuration_checksum != planned_config.configuration_checksum
        or record.config.to_dict() != planned_config.to_dict()
    ):
        msg = "The evaluator returned a row other than the exact planned historical evaluation."
        raise ValueError(msg)

    expected_seed_domain = pipeline_config.seed_domains["pilot_evaluation"]
    historical_policy = (
        planned_config.test_noise_id == HISTORICAL_FIXED_RATE_NOISE_ID
        and planned_config.noise_definition_version == FIXED_RATE_NOISE_DEFINITION_VERSION
        and planned_config.noise_strength_scale is not None
        and float(planned_config.noise_strength_scale).hex() == (1.0).hex()
        and planned_config.tjm_dt is not None
        and float(planned_config.tjm_dt).hex() == (1.0).hex()
        and planned_config.evaluation_seed == LEGACY_EVALUATION_SEED
        and planned_config.evaluation_seed_domain == expected_seed_domain
        and planned_config.repetition == 0
        and planned_config.trajectory_budget == LEGACY_EVALUATION_TRAJECTORY_COUNT
        and planned_config.evaluation_policy == "fixed_sample"
        and planned_config.confidence_level is None
        and planned_config.confidence_interval_method is None
        and planned_config.sidecar_storage_policy == "trajectory_fidelities"
    )
    if not historical_policy:
        msg = "The planned row does not implement the fixed historical noise, seed-zero, 500-trajectory policy."
        raise ValueError(msg)
    return record


def execute_historical_target(request: HistoricalTargetRequest) -> LegacyReproductionOutcome:
    """Execute and persist one of the five q8 targets through the WP18 stack."""
    collection = load_legacy_target_collection()
    target = collection.target(f"legacy_tfim_seed_{request.target_seed}")
    pipeline = resolve_layerwise_bmpd_crn_legacy_v1_pipeline(request.target_seed)
    fingerprint = request.resumability_fingerprint
    request.source_manifest.require_fingerprint_match(fingerprint)
    if fingerprint.pipeline_prefix_id != pipeline.prefix_id(len(pipeline.stages) - 1):
        msg = "Verified target fingerprint does not match the resolved pipeline prefix."
        raise ValueError(msg)
    store = Phase2ArtifactStore(
        request.output_directory,
        pipeline,
        fingerprint,
        resume=request.resume,
        overwrite=request.overwrite,
    )
    stage_runner = LayerwiseBMPDStageRunner(pipeline, target)
    execution = Phase2PipelineExecutor(store).execute(
        stage_runner,
        circuit_statistics=stage_runner.circuit_statistics,
    )
    if isinstance(execution, PipelineExecutionFailure):
        if execution.failure.runtime_fingerprint_checksum != fingerprint.content_checksum:
            msg = "Training failure does not bind the verified target WP18 fingerprint."
            raise ValueError(msg)
        return _stage_failure_outcome(request.target_seed, execution.failure)

    parameters = store.load_final_parameters()
    planned_payload = encode_layerwise_materialized_circuit(
        create_bmpd_circuit_binding(pipeline.qubit_count, 4),
        parameters,
    )
    payload_checksum = f"sha256:{hashlib.sha256(planned_payload).hexdigest()}"
    final_stage = pipeline.stages[-1]
    evaluation_config = PipelineEvaluationConfig.for_pipeline(
        pipeline=execution,
        materialized_circuit_checksum=payload_checksum,
        test_noise_id=HISTORICAL_FIXED_RATE_NOISE_ID,
        noise_definition_version=FIXED_RATE_NOISE_DEFINITION_VERSION,
        noise_strength_scale=1.0,
        tjm_dt=1.0,
        evaluation_seed=LEGACY_EVALUATION_SEED,
        evaluation_seed_domain=cast("str", pipeline.seed_domains["pilot_evaluation"]),
        repetition=0,
        trajectory_budget=LEGACY_EVALUATION_TRAJECTORY_COUNT,
        evaluation_policy="fixed_sample",
        confidence_level=None,
        confidence_interval_method=None,
        sidecar_storage_policy="trajectory_fidelities",
        max_bond_dimension=final_stage.max_bond_dimension,
        svd_threshold=final_stage.svd_threshold,
        truncation_mode=final_stage.truncation_mode,
        min_bond_dimension=final_stage.min_bond_dimension,
    )
    evaluator = ParallelPhase2Evaluator(store, decode_layerwise_materialized_circuit)

    def materialize(
        pipeline_result: TrainingPipelineResult,
        selected_parameters: NDArray[np.float64],
    ) -> MaterializedCircuitPayload:
        payload = _materialized_payload(pipeline_result, selected_parameters)
        if payload.serialized_bytes != planned_payload:
            msg = "Final circuit materialization changed after evaluation planning."
            raise ValueError(msg)
        return payload

    records = evaluator.evaluate(
        (evaluation_config,),
        materialize,
        lambda config, circuit: evaluate_historical_materialized_circuit(config, circuit, target),
        max_workers=HISTORICAL_REPRODUCTION_EVALUATION_WORKERS,
    )
    if len(records) != 1:
        msg = "The historical evaluator must return exactly one planned row per target."
        raise RuntimeError(msg)
    record = validate_historical_evaluation_record(
        target_seed=request.target_seed,
        record=records[0],
        planned_config=evaluation_config,
        pipeline=execution,
        expected_runtime_fingerprint_checksum=fingerprint.content_checksum,
    )
    return LegacyReproductionOutcome.from_pipeline_record(request.target_seed, record)


def run_historical_reproduction_job(
    output_root: Path,
    *,
    execute_expensive: bool,
    resume: bool = False,
    overwrite: bool = False,
    repository_root: Path | None = None,
    target_executor: TargetExecutor = execute_historical_target,
) -> LegacyReproductionReport:
    """Run the exact five-target WP19 job and write its mechanical comparison.

    Args:
        output_root: Dedicated artifact directory chosen by the caller.
        execute_expensive: Must be exactly ``True``; there is no implicit run.
        resume: Verify and continue every existing per-target store.
        overwrite: Replace only known managed outputs in each target store.
        repository_root: Git worktree root, normally detected from this module.
        target_executor: Injection seam for structural tests; the CLI always uses
            :func:`execute_historical_target`.

    Returns:
        The checksum-sealed five-row comparison report.
    """
    if type(execute_expensive) is not bool or not execute_expensive:
        msg = "The q8 historical reproduction requires explicit execute_expensive=True opt-in."
        raise ValueError(msg)
    if not callable(target_executor):
        msg = "target_executor must be callable."
        raise TypeError(msg)
    repository = Path(__file__).resolve().parents[3] if repository_root is None else repository_root.resolve()
    with _exclusive_historical_job_lock(output_root, repository) as locked_root:
        root = _prepare_output_root(locked_root, repository, resume=resume, overwrite=overwrite)
        starting_commit = _run_git(repository, "rev-parse", "HEAD")
        dependencies = _dependency_versions()
        source_paths = _tracked_execution_sources(repository)
        target_root = root / HISTORICAL_REPRODUCTION_TARGET_DIRECTORY
        launch_seed = LEGACY_REPRODUCTION_TARGET_SEEDS[0]
        launch_fingerprint = _capture_target_resumability_fingerprint(
            repository_root=repository,
            output_directory=target_root / f"seed_{launch_seed}",
            starting_commit=starting_commit,
            dependency_versions=dependencies,
            execution_source_paths=source_paths,
            target_seed=launch_seed,
        )
        source_manifest = HistoricalReproductionSourceManifest.from_fingerprint(launch_fingerprint)
        _write_or_verify_source_manifest(root, source_manifest, resume=resume)

        outcomes: list[LegacyReproductionOutcome] = []
        with threadpool_limits(limits=HISTORICAL_REPRODUCTION_THREAD_LIMIT):
            runtime = capture_historical_reproduction_runtime(
                starting_commit=starting_commit,
                dependency_versions=dependencies,
                source_manifest_checksum=source_manifest.content_checksum,
            )
            _write_or_verify_runtime(root, runtime, source_manifest, resume=resume)
            if resume:
                _verify_retained_report_if_present(root, runtime, source_manifest)
            for target_seed in LEGACY_REPRODUCTION_TARGET_SEEDS:
                target_output = target_root / f"seed_{target_seed}"
                fingerprint = _capture_target_resumability_fingerprint(
                    repository_root=repository,
                    output_directory=target_output,
                    starting_commit=starting_commit,
                    dependency_versions=dependencies,
                    execution_source_paths=source_paths,
                    target_seed=target_seed,
                )
                source_manifest.require_fingerprint_match(fingerprint)
                expected_fingerprint = source_manifest.fingerprint_for_pipeline(fingerprint.pipeline_prefix_id)
                if fingerprint != expected_fingerprint:
                    msg = "Target WP18 fingerprint was not derived solely by changing the launch pipeline prefix."
                    raise ValueError(msg)
                if target_output.is_symlink() or (target_output.exists() and not target_output.is_dir()):
                    msg = f"Managed target output for seed {target_seed} must be a real directory."
                    outcomes.append(
                        _unexpected_failure_outcome(
                            target_seed,
                            ValueError(msg),
                            fingerprint.content_checksum,
                        )
                    )
                    continue
                target_resume = resume and (target_output / PIPELINE_CONFIG_NAME).is_file()
                request = HistoricalTargetRequest(
                    target_seed=target_seed,
                    output_directory=target_output,
                    repository_root=repository,
                    source_manifest=source_manifest,
                    resumability_fingerprint=fingerprint,
                    resume=target_resume,
                    overwrite=overwrite,
                )
                try:
                    outcome = target_executor(request)
                except Exception as error:
                    outcome = _unexpected_failure_outcome(
                        target_seed,
                        error,
                        fingerprint.content_checksum,
                    )
                if not isinstance(outcome, LegacyReproductionOutcome) or outcome.target_seed != target_seed:
                    msg = "target_executor returned an outcome for the wrong target seed."
                    raise ValueError(msg)
                if outcome.runtime_fingerprint_checksum != fingerprint.content_checksum:
                    msg = "target_executor outcome does not bind the verified target WP18 fingerprint."
                    raise ValueError(msg)
                outcomes.append(outcome)

        runtime_checksum = require_checksum(runtime["content_checksum"], "runtime.content_checksum")
        report = compare_legacy_reproduction(
            outcomes,
            tolerance=HISTORICAL_REPRODUCTION_TOLERANCE,
            tolerance_rationale=HISTORICAL_REPRODUCTION_TOLERANCE_RATIONALE,
            source_manifest_checksum=source_manifest.content_checksum,
            runtime_checksum=runtime_checksum,
        )

        final_seed = LEGACY_REPRODUCTION_TARGET_SEEDS[-1]
        final_fingerprint = _capture_target_resumability_fingerprint(
            repository_root=repository,
            output_directory=target_root / f"seed_{final_seed}",
            starting_commit=starting_commit,
            dependency_versions=dependencies,
            execution_source_paths=source_paths,
            target_seed=final_seed,
        )
        source_manifest.require_fingerprint_match(final_fingerprint)
        expected_final = source_manifest.fingerprint_for_pipeline(final_fingerprint.pipeline_prefix_id)
        if final_fingerprint != expected_final:
            msg = "Final WP18 fingerprint was not derived from the launch snapshot."
            raise ValueError(msg)
        stored_manifest, stored_runtime = _load_verified_provenance(root)
        if stored_manifest != source_manifest or thaw_json(stored_runtime) != thaw_json(runtime):
            msg = "Persisted historical provenance changed during target execution."
            raise ValueError(msg)
        _require_report_artifact_bindings(report, stored_runtime, stored_manifest)
        report_path = root / HISTORICAL_REPRODUCTION_REPORT_NAME
        if report_path.is_symlink() or (report_path.exists() and not report_path.is_file()):
            msg = "The managed historical report path must be a regular file."
            raise ValueError(msg)
        atomic_write_bytes(report_path, f"{report.to_json()}\n".encode())
        if verify_historical_reproduction_artifacts(root) != report:
            msg = "Persisted historical reproduction report changed during final verification."
            raise ValueError(msg)
        return report


def _parser() -> argparse.ArgumentParser:
    """Build the deliberately explicit command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True, help="Dedicated output directory.")
    parser.add_argument(
        "--execute-expensive",
        action="store_true",
        help="Required acknowledgement for five q8 optimizations and 2,500 evaluation trajectories.",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--resume", action="store_true", help="Verify and continue existing target stores.")
    modes.add_argument("--overwrite", action="store_true", help="Replace only managed artifacts in target stores.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the opt-in CLI and return a conventional process status."""
    arguments = _parser().parse_args(argv)
    try:
        report = run_historical_reproduction_job(
            arguments.output_root,
            execute_expensive=arguments.execute_expensive,
            resume=arguments.resume,
            overwrite=arguments.overwrite,
        )
    except Exception as error:
        print(f"Historical reproduction failed: {error}", file=sys.stderr)
        return HISTORICAL_REPRODUCTION_FAILURE_EXIT_CODE
    if any(comparison.outcome.status == "failure" for comparison in report.target_comparisons):
        return HISTORICAL_REPRODUCTION_FAILURE_EXIT_CODE
    if report.classification != "reproduced":
        return HISTORICAL_REPRODUCTION_DISCREPANCY_EXIT_CODE
    return HISTORICAL_REPRODUCTION_SUCCESS_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HISTORICAL_REPRODUCTION_DISCREPANCY_EXIT_CODE",
    "HISTORICAL_REPRODUCTION_EVALUATION_WORKERS",
    "HISTORICAL_REPRODUCTION_FAILURE_EXIT_CODE",
    "HISTORICAL_REPRODUCTION_JOB_ID",
    "HISTORICAL_REPRODUCTION_JOB_SCHEMA_VERSION",
    "HISTORICAL_REPRODUCTION_LOCK_NAME",
    "HISTORICAL_REPRODUCTION_LOCK_TIMEOUT_SECONDS",
    "HISTORICAL_REPRODUCTION_REPORT_NAME",
    "HISTORICAL_REPRODUCTION_RUNTIME_NAME",
    "HISTORICAL_REPRODUCTION_SOURCE_MANIFEST_NAME",
    "HISTORICAL_REPRODUCTION_SOURCE_MANIFEST_SCHEMA_VERSION",
    "HISTORICAL_REPRODUCTION_SUCCESS_EXIT_CODE",
    "HISTORICAL_REPRODUCTION_TARGET_DIRECTORY",
    "HISTORICAL_REPRODUCTION_THREAD_LIMIT",
    "HISTORICAL_REPRODUCTION_TOLERANCE",
    "HISTORICAL_REPRODUCTION_TOLERANCE_RATIONALE",
    "HistoricalReproductionConcurrentExecutionError",
    "HistoricalReproductionSourceManifest",
    "HistoricalTargetRequest",
    "TargetExecutor",
    "capture_historical_reproduction_runtime",
    "evaluate_historical_materialized_circuit",
    "execute_historical_target",
    "main",
    "run_historical_reproduction_job",
    "validate_historical_evaluation_record",
    "verify_historical_reproduction_artifacts",
]
