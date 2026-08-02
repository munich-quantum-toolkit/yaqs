# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Sealed WP19 reconstructions of the five historical disordered-TFIM targets.

The historical experiment retained the target generator but not its materialized
vectors, generated parameters, eigenvalues, or runtime fingerprint.  Consequently,
the checked-in vectors in this module's companion data file are explicitly labelled
WP19 reconstructed references.  They are never represented as archived vectors.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
import os
import platform
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import scipy

from .canonical import (
    canonical_checksum,
    canonical_json,
    freeze_json_mapping,
    load_canonical_json_object,
    thaw_json_mapping,
    verify_sealed_mapping,
)
from .pipeline import (
    LEGACY_REPRODUCTION_MANIFEST_CHECKSUM,
    LEGACY_REPRODUCTION_TARGET_IDS,
    fixture_target_spec_checksum,
)
from .validation import (
    require_bool,
    require_checksum,
    require_exact_keys,
    require_float,
    require_git_commit,
    require_int,
    require_mapping,
    require_string,
    require_string_sequence,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


LEGACY_TARGET_COLLECTION_SCHEMA_VERSION = "yaqs.state_preparation.phase2.legacy_target_collection.v1"
LEGACY_TARGET_RECORD_SCHEMA_VERSION = "yaqs.state_preparation.phase2.legacy_materialized_target.v1"
LEGACY_TARGET_GENERATOR_SCHEMA_VERSION = "yaqs.state_preparation.phase2.legacy_tfim_generator.v1"
LEGACY_TARGET_RUNTIME_SCHEMA_VERSION = "yaqs.state_preparation.phase2.legacy_reconstruction_runtime.v1"

LEGACY_TARGET_COLLECTION_ID = "legacy_tfim_targets_v1"
LEGACY_TARGET_NAMESPACE = "legacy_reproduction"
LEGACY_TARGET_REFERENCE_STATUS = "wp19_reconstructed_reference"
LEGACY_TARGET_FAMILY_ID = "tfim_ground_state"
LEGACY_TARGET_STRATUM_ID = "legacy_disordered"
LEGACY_TARGET_QUBIT_COUNT = 8
LEGACY_TARGET_SEEDS = (100, 200, 300, 400, 500)

LEGACY_TARGET_SOURCE_COMMIT = "fb621e2deb4da6f8ba16d3e48d05077d8e2b8809"
LEGACY_TARGET_SOURCE_PATH = "experiments/re_evaluate_benchmarks.py"
LEGACY_TARGET_SOURCE_CHECKSUM = "sha256:f597a8335df6de381f6740f59e19c12485acf03aef852fa2b0ceabde7067a2fd"

LEGACY_TARGET_PHASE_INVARIANT_ATOL = 1e-10
LEGACY_TARGET_PHASE_INVARIANT_RTOL = 1e-10

MAX_LEGACY_TARGET_COLLECTION_BYTES = 64 * 1024
MAX_LEGACY_TARGET_VECTOR_DECODED_BYTES = 2**LEGACY_TARGET_QUBIT_COUNT * np.dtype("<c16").itemsize
MAX_LEGACY_TARGET_VECTOR_BASE64_BYTES = 4 * ((MAX_LEGACY_TARGET_VECTOR_DECODED_BYTES + 2) // 3)

DEFAULT_LEGACY_TARGET_COLLECTION_PATH = Path(__file__).with_name("data") / "legacy_tfim_targets_v1.json"
TRUSTED_LEGACY_TARGET_COLLECTION_CHECKSUM = "sha256:06f1f31c1ba6373837f2b41b6824fd9d45107b12a8c11b8973f4cf0af83954f4"

LEGACY_TARGET_GENERATOR = freeze_json_mapping(
    {
        "schema_version": LEGACY_TARGET_GENERATOR_SCHEMA_VERSION,
        "rng": "numpy.random.RandomState",
        "coupling_draw": {
            "distribution": "uniform",
            "low": 0.8,
            "high": 1.2,
            "size": "qubit_count_minus_one",
        },
        "field_draw": {
            "distribution": "uniform",
            "low": 0.8,
            "high": 1.2,
            "size": "qubit_count",
        },
        "draw_order": ["couplings", "fields"],
        "hamiltonian": "negative_sum_J_i_Z_i_Z_i_plus_1_minus_sum_h_i_X_i",
        "tensor_product_order": "generator_site_zero_is_most_significant_basis_bit",
        "diagonalization": "numpy.linalg.eigh",
        "eigenvector_selection": "eigenvectors_column_zero",
        "global_phase_convention": "none_historical_eigensolver_output",
    },
    "legacy target generator",
)

LEGACY_TARGET_POPULATION_CONFIG_CHECKSUM = canonical_checksum({
    "schema_version": "yaqs.state_preparation.phase2.legacy_target_population_config.v1",
    "namespace": LEGACY_TARGET_NAMESPACE,
    "qubit_count": LEGACY_TARGET_QUBIT_COUNT,
    "target_seeds": list(LEGACY_TARGET_SEEDS),
    "generator": LEGACY_TARGET_GENERATOR,
})

LEGACY_TARGET_PHASE_COMPARISON = freeze_json_mapping(
    {
        "metric": "phase_aligned_amplitude_allclose",
        "absolute_tolerance": LEGACY_TARGET_PHASE_INVARIANT_ATOL,
        "relative_tolerance": LEGACY_TARGET_PHASE_INVARIANT_RTOL,
        "rationale": (
            "The tolerance permits floating-point eigensolver and BLAS/LAPACK drift while rejecting "
            "scientifically different state vectors."
        ),
    },
    "legacy target phase comparison",
)

LEGACY_TARGET_MISSING_PROVENANCE = (
    (
        "The archived target vectors, couplings, fields, energies, and target checksums were not retained; "
        "this collection contains WP19 reconstructed references generated from the commit-addressed source."
    ),
    (
        "The archived Python patch version, NumPy/SciPy builds, BLAS/LAPACK build, thread settings, platform, "
        "and hardware fingerprint were not retained; reconstruction_runtime describes only the WP19 environment."
    ),
)

_RUNTIME_LIBRARY_KEYS = frozenset({
    "name",
    "version",
    "detection_method",
    "found",
    "include_directory",
    "lib_directory",
})
_RUNTIME_KEYS = frozenset({
    "schema_version",
    "python_implementation",
    "python_version",
    "numpy_version",
    "scipy_version",
    "operating_system",
    "platform",
    "machine",
    "processor",
    "byteorder",
    "blas",
    "lapack",
})
_VECTOR_ENCODING_KEYS = frozenset({"encoding", "dtype", "amplitude_count", "data_base64"})
_TARGET_KEYS = frozenset({
    "schema_version",
    "target_instance_id",
    "target_instance_spec_checksum",
    "population_config_checksum",
    "target_manifest_checksum",
    "parameter_checksum",
    "family_id",
    "stratum_id",
    "qubit_count",
    "seed",
    "couplings",
    "fields",
    "ground_energy",
    "reference_status",
    "archived_vector_checksum",
    "norm",
    "vector_checksum",
    "state_vector",
    "content_checksum",
})
_COLLECTION_KEYS = frozenset({
    "schema_version",
    "collection_id",
    "namespace",
    "reference_status",
    "archived_state_vectors_retained",
    "source_commit",
    "source_path",
    "source_content_checksum",
    "generator",
    "population_config_checksum",
    "phase_comparison",
    "reconstruction_runtime",
    "missing_provenance",
    "targets",
    "content_checksum",
})
_TARGET_CREATION_SENTINEL = object()


def _require_exact_float_sequence(
    value: object,
    name: str,
    *,
    length: int,
) -> tuple[float, ...]:
    """Return a finite exact-float tuple of the required length.

    Raises:
        TypeError: If the value is not a sequence of built-in floats.
        ValueError: If the sequence length differs or a value is nonfinite.
    """
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        msg = f"{name} must be a sequence of floats."
        raise TypeError(msg)
    values = tuple(require_float(item, f"{name}[{index}]") for index, item in enumerate(value))
    if len(values) != length:
        msg = f"{name} must contain exactly {length} values."
        raise ValueError(msg)
    return values


def _raw_vector_checksum(vector_bytes: bytes) -> str:
    """Return the MaterializedTarget-compatible raw vector checksum."""
    return f"sha256:{hashlib.sha256(vector_bytes).hexdigest()}"


def _canonical_vector_bytes(vector: NDArray[np.complex128], name: str) -> bytes:
    """Validate a statevector and return canonical little-endian complex128 bytes.

    Returns:
        Canonical raw vector bytes.

    Raises:
        TypeError: If ``vector`` is not an exact complex128 NumPy array.
        ValueError: If its shape, amplitudes, or norm are invalid.
    """
    if not isinstance(vector, np.ndarray) or vector.dtype != np.dtype(np.complex128):
        msg = f"{name} must be an exact complex128 NumPy array."
        raise TypeError(msg)
    if vector.shape != (2**LEGACY_TARGET_QUBIT_COUNT,) or not np.all(np.isfinite(vector)):
        msg = f"{name} must be finite with shape ({2**LEGACY_TARGET_QUBIT_COUNT},)."
        raise ValueError(msg)
    canonical = np.ascontiguousarray(vector, dtype=np.dtype("<c16"))
    norm = float(np.linalg.norm(canonical))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-12):
        msg = f"{name} norm must equal one, got {norm!r}."
        raise ValueError(msg)
    return canonical.tobytes(order="C")


def _decode_vector(value: object) -> NDArray[np.complex128]:
    """Decode and strictly validate one canonical vector payload.

    Returns:
        Detached native-endian complex128 amplitudes.

    Raises:
        ValueError: If the encoding metadata or Base64 payload is invalid.
    """
    mapping = require_mapping(value, "legacy target state_vector")
    require_exact_keys(mapping, _VECTOR_ENCODING_KEYS, "legacy target state_vector")
    if mapping["encoding"] != "base64" or mapping["dtype"] != "<c16":
        msg = "legacy target state_vector must use canonical base64 little-endian complex128 encoding."
        raise ValueError(msg)
    amplitude_count = require_int(mapping["amplitude_count"], "state_vector.amplitude_count", minimum=1)
    expected_size = amplitude_count * np.dtype("<c16").itemsize
    if amplitude_count != 2**LEGACY_TARGET_QUBIT_COUNT or expected_size > MAX_LEGACY_TARGET_VECTOR_DECODED_BYTES:
        msg = (
            f"state_vector.amplitude_count must equal {2**LEGACY_TARGET_QUBIT_COUNT} "
            "and remain within the decoded size bound."
        )
        raise ValueError(msg)
    encoded = require_string(mapping["data_base64"], "state_vector.data_base64")
    if len(encoded) != MAX_LEGACY_TARGET_VECTOR_BASE64_BYTES:
        msg = (
            "state_vector.data_base64 must have exactly the bounded encoded size "
            f"{MAX_LEGACY_TARGET_VECTOR_BASE64_BYTES}."
        )
        raise ValueError(msg)
    try:
        encoded_bytes = encoded.encode("ascii", errors="strict")
        decoded = base64.b64decode(encoded_bytes, validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as error:
        msg = "state_vector.data_base64 is not valid canonical Base64."
        raise ValueError(msg) from error
    if base64.b64encode(decoded).decode("ascii") != encoded:
        msg = "state_vector.data_base64 is not in canonical padded Base64 form."
        raise ValueError(msg)
    if len(decoded) != expected_size:
        msg = f"state_vector decoded byte length must equal the size bound {MAX_LEGACY_TARGET_VECTOR_DECODED_BYTES}."
        raise ValueError(msg)
    return np.frombuffer(decoded, dtype=np.dtype("<c16")).astype(np.complex128, copy=True)


def _encode_vector(vector_bytes: bytes) -> dict[str, object]:
    """Return the canonical serialized vector envelope."""
    return {
        "encoding": "base64",
        "dtype": "<c16",
        "amplitude_count": 2**LEGACY_TARGET_QUBIT_COUNT,
        "data_base64": base64.b64encode(vector_bytes).decode("ascii"),
    }


def _parameter_checksum(seed: int, couplings: tuple[float, ...], fields: tuple[float, ...]) -> str:
    """Return the legacy generator-parameter commitment used by target identities."""
    return canonical_checksum({
        "generator_schema_version": LEGACY_TARGET_GENERATOR_SCHEMA_VERSION,
        "seed": seed,
        "couplings": list(couplings),
        "fields": list(fields),
    })


def _runtime_library_record(value: object, name: str) -> Mapping[str, object]:
    """Validate and freeze one BLAS or LAPACK provenance record.

    Returns:
        Immutable normalized build provenance.
    """
    mapping = require_mapping(value, name)
    require_exact_keys(mapping, _RUNTIME_LIBRARY_KEYS, name)
    normalized = {
        "name": require_string(mapping["name"], f"{name}.name"),
        "version": require_string(mapping["version"], f"{name}.version"),
        "detection_method": require_string(mapping["detection_method"], f"{name}.detection_method"),
        "found": require_bool(mapping["found"], f"{name}.found"),
        "include_directory": require_string(mapping["include_directory"], f"{name}.include_directory"),
        "lib_directory": require_string(mapping["lib_directory"], f"{name}.lib_directory"),
    }
    return freeze_json_mapping(normalized, name)


@dataclass(frozen=True, slots=True)
class LegacyReconstructionRuntime:
    """Runtime provenance for one WP19 target reconstruction."""

    python_implementation: str
    python_version: str
    numpy_version: str
    scipy_version: str
    operating_system: str
    platform: str
    machine: str
    processor: str
    byteorder: str
    blas: Mapping[str, object]
    lapack: Mapping[str, object]
    schema_version: str = field(default=LEGACY_TARGET_RUNTIME_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate and freeze all runtime and numerical-library fields.

        Raises:
            ValueError: If a version, platform, byte order, or build record is invalid.
        """
        for name in (
            "python_implementation",
            "python_version",
            "numpy_version",
            "scipy_version",
            "operating_system",
            "platform",
            "machine",
            "processor",
        ):
            object.__setattr__(self, name, require_string(getattr(self, name), name))
        if self.byteorder not in {"little", "big"}:
            msg = "byteorder must be 'little' or 'big'."
            raise ValueError(msg)
        object.__setattr__(self, "blas", _runtime_library_record(self.blas, "blas"))
        object.__setattr__(self, "lapack", _runtime_library_record(self.lapack, "lapack"))

    def to_dict(self) -> dict[str, object]:
        """Return detached JSON-native reconstruction provenance."""
        return {
            "schema_version": self.schema_version,
            "python_implementation": self.python_implementation,
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "scipy_version": self.scipy_version,
            "operating_system": self.operating_system,
            "platform": self.platform,
            "machine": self.machine,
            "processor": self.processor,
            "byteorder": self.byteorder,
            "blas": thaw_json_mapping(self.blas),
            "lapack": thaw_json_mapping(self.lapack),
        }

    @classmethod
    def from_dict(cls, data: object) -> LegacyReconstructionRuntime:
        """Construct provenance from an exact versioned JSON object.

        Returns:
            Validated immutable runtime provenance.

        Raises:
            ValueError: If the runtime schema version is unsupported.
        """
        mapping = require_mapping(data, "legacy reconstruction runtime")
        require_exact_keys(mapping, _RUNTIME_KEYS, "legacy reconstruction runtime")
        if mapping["schema_version"] != LEGACY_TARGET_RUNTIME_SCHEMA_VERSION:
            msg = f"schema_version must be {LEGACY_TARGET_RUNTIME_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        return cls(
            python_implementation=cast("str", mapping["python_implementation"]),
            python_version=cast("str", mapping["python_version"]),
            numpy_version=cast("str", mapping["numpy_version"]),
            scipy_version=cast("str", mapping["scipy_version"]),
            operating_system=cast("str", mapping["operating_system"]),
            platform=cast("str", mapping["platform"]),
            machine=cast("str", mapping["machine"]),
            processor=cast("str", mapping["processor"]),
            byteorder=cast("str", mapping["byteorder"]),
            blas=cast("Mapping[str, object]", mapping["blas"]),
            lapack=cast("Mapping[str, object]", mapping["lapack"]),
        )


def _numpy_build_dependency(name: str) -> dict[str, object]:
    """Extract a stable BLAS/LAPACK subset from NumPy's build configuration.

    Returns:
        JSON-native numerical-library build metadata.
    """
    config = cast("Mapping[str, object]", getattr(np.__config__, "CONFIG", {}))
    dependencies = cast("Mapping[str, object]", config.get("Build Dependencies", {}))
    raw = cast("Mapping[str, object]", dependencies.get(name, {}))
    return {
        "name": str(raw.get("name", "unknown")),
        "version": str(raw.get("version", "unknown")),
        "detection_method": str(raw.get("detection method", "unknown")),
        "found": bool(raw.get("found", False)),
        "include_directory": str(raw.get("include directory", "unknown")),
        "lib_directory": str(raw.get("lib directory", "unknown")),
    }


def capture_legacy_reconstruction_runtime() -> LegacyReconstructionRuntime:
    """Capture the current runtime fields required to interpret a reconstruction.

    Returns:
        Current Python, numerical-library, and platform provenance.
    """
    return LegacyReconstructionRuntime(
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        operating_system=platform.system(),
        platform=platform.platform(),
        machine=platform.machine() or "unknown",
        processor=platform.processor() or "unknown",
        byteorder=sys.byteorder,
        blas=_numpy_build_dependency("blas"),
        lapack=_numpy_build_dependency("lapack"),
    )


@dataclass(frozen=True, slots=True)
class LegacyRegeneratedTarget:
    """One direct execution of the exact historical target generator."""

    seed: int
    couplings: tuple[float, ...]
    fields: tuple[float, ...]
    ground_energy: float
    _state_vector_bytes: bytes = field(repr=False)

    def state_vector_copy(self) -> NDArray[np.complex128]:
        """Return a detached writable copy of the regenerated vector."""
        return np.frombuffer(self._state_vector_bytes, dtype=np.dtype("<c16")).astype(np.complex128, copy=True)


def regenerate_legacy_tfim_target(seed: int) -> LegacyRegeneratedTarget:
    """Execute the archived q8 TFIM generator with its exact ordering semantics.

    Args:
        seed: One of the five historical target seeds.

    Returns:
        Generated parameters, ground energy, and raw eigensolver vector.

    Raises:
        ValueError: If ``seed`` is not one of the historical seeds.
    """
    normalized_seed = require_int(seed, "seed")
    if normalized_seed not in LEGACY_TARGET_SEEDS:
        msg = f"seed must be one of the five historical seeds {LEGACY_TARGET_SEEDS!r}."
        raise ValueError(msg)

    rng = np.random.RandomState(normalized_seed)
    couplings_array = rng.uniform(0.8, 1.2, size=LEGACY_TARGET_QUBIT_COUNT - 1)
    fields_array = rng.uniform(0.8, 1.2, size=LEGACY_TARGET_QUBIT_COUNT)

    dimension = 2**LEGACY_TARGET_QUBIT_COUNT
    hamiltonian = np.zeros((dimension, dimension), dtype=np.complex128)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    identity = np.eye(2, dtype=np.complex128)

    for site in range(LEGACY_TARGET_QUBIT_COUNT - 1):
        operators = [identity] * LEGACY_TARGET_QUBIT_COUNT
        operators[site] = pauli_z
        operators[site + 1] = pauli_z
        term = operators[0]
        for operator in operators[1:]:
            term = np.kron(term, operator)
        hamiltonian -= couplings_array[site] * term

    for site in range(LEGACY_TARGET_QUBIT_COUNT):
        operators = [identity] * LEGACY_TARGET_QUBIT_COUNT
        operators[site] = pauli_x
        term = operators[0]
        for operator in operators[1:]:
            term = np.kron(term, operator)
        hamiltonian -= fields_array[site] * term

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    vector = cast("NDArray[np.complex128]", eigenvectors[:, 0])
    return LegacyRegeneratedTarget(
        seed=normalized_seed,
        couplings=tuple(float(value) for value in couplings_array),
        fields=tuple(float(value) for value in fields_array),
        ground_energy=float(eigenvalues[0]),
        _state_vector_bytes=_canonical_vector_bytes(vector, "regenerated state vector"),
    )


@dataclass(frozen=True, slots=True, init=False)
class LegacyMaterializedTarget:
    """Immutable reconstructed target with the MaterializedTarget identity interface."""

    target_instance_id: str
    target_instance_spec_checksum: str
    population_config_checksum: str
    target_manifest_checksum: str
    parameter_checksum: str
    family_id: str
    stratum_id: str
    qubit_count: int
    seed: int
    couplings: tuple[float, ...]
    fields: tuple[float, ...]
    ground_energy: float
    reference_status: str
    archived_vector_checksum: None
    norm: float
    vector_checksum: str
    _state_vector_bytes: bytes = field(repr=False)
    schema_version: str = field(default=LEGACY_TARGET_RECORD_SCHEMA_VERSION, init=False)

    def __init__(
        self,
        *,
        seed: int,
        couplings: Sequence[float],
        fields: Sequence[float],
        ground_energy: float,
        vector: NDArray[np.complex128],
        target_instance_id: str,
        target_instance_spec_checksum: str,
        population_config_checksum: str,
        target_manifest_checksum: str,
        parameter_checksum: str,
        family_id: str,
        stratum_id: str,
        qubit_count: int,
        reference_status: str,
        archived_vector_checksum: None,
        vector_checksum: str,
        recorded_norm: float | None,
        _marker: object,
    ) -> None:
        """Validate and freeze one reconstructed legacy target.

        Raises:
            ValueError: If any fixed identity, checksum, shape, or norm differs.
        """
        if _marker is not _TARGET_CREATION_SENTINEL:
            msg = "LegacyMaterializedTarget records may only be created by the strict decoder or reconstructor."
            raise ValueError(msg)
        normalized_seed = require_int(seed, "seed")
        if normalized_seed not in LEGACY_TARGET_SEEDS:
            msg = f"seed must be one of {LEGACY_TARGET_SEEDS!r}."
            raise ValueError(msg)
        expected_id = f"legacy_tfim_seed_{normalized_seed}"
        if require_string(target_instance_id, "target_instance_id") != expected_id:
            msg = f"target_instance_id must be {expected_id!r} for seed {normalized_seed}."
            raise ValueError(msg)
        if require_int(qubit_count, "qubit_count", minimum=2) != LEGACY_TARGET_QUBIT_COUNT:
            msg = f"qubit_count must equal {LEGACY_TARGET_QUBIT_COUNT}."
            raise ValueError(msg)
        if family_id != LEGACY_TARGET_FAMILY_ID or stratum_id != LEGACY_TARGET_STRATUM_ID:
            msg = "family_id and stratum_id must identify the frozen legacy disordered-TFIM fixture."
            raise ValueError(msg)
        if reference_status != LEGACY_TARGET_REFERENCE_STATUS:
            msg = f"reference_status must be {LEGACY_TARGET_REFERENCE_STATUS!r}."
            raise ValueError(msg)
        if archived_vector_checksum is not None:
            msg = "archived_vector_checksum must be null because no archived vector was retained."
            raise ValueError(msg)

        normalized_couplings = _require_exact_float_sequence(
            couplings,
            "couplings",
            length=LEGACY_TARGET_QUBIT_COUNT - 1,
        )
        normalized_fields = _require_exact_float_sequence(fields, "fields", length=LEGACY_TARGET_QUBIT_COUNT)
        normalized_energy = require_float(ground_energy, "ground_energy")
        expected_parameter_checksum = _parameter_checksum(normalized_seed, normalized_couplings, normalized_fields)
        supplied_parameter_checksum = require_checksum(parameter_checksum, "parameter_checksum")
        if supplied_parameter_checksum != expected_parameter_checksum:
            msg = (
                "parameter_checksum does not match seed, couplings, and fields: "
                f"expected {expected_parameter_checksum}, got {supplied_parameter_checksum}."
            )
            raise ValueError(msg)

        expected_spec_checksum = fixture_target_spec_checksum(
            "legacy_reproduction",
            expected_id,
            LEGACY_TARGET_QUBIT_COUNT,
        )
        supplied_spec_checksum = require_checksum(
            target_instance_spec_checksum,
            "target_instance_spec_checksum",
        )
        if supplied_spec_checksum != expected_spec_checksum:
            msg = f"target_instance_spec_checksum must equal the frozen fixture commitment {expected_spec_checksum}."
            raise ValueError(msg)
        if require_checksum(population_config_checksum, "population_config_checksum") != (
            LEGACY_TARGET_POPULATION_CONFIG_CHECKSUM
        ):
            msg = "population_config_checksum must equal the frozen legacy generator-population commitment."
            raise ValueError(msg)
        if require_checksum(target_manifest_checksum, "target_manifest_checksum") != (
            LEGACY_REPRODUCTION_MANIFEST_CHECKSUM
        ):
            msg = "target_manifest_checksum must equal the WP15 legacy reproduction manifest commitment."
            raise ValueError(msg)

        vector_bytes = _canonical_vector_bytes(vector, "legacy target vector")
        computed_norm = float(np.linalg.norm(vector))
        normalized_norm = computed_norm if recorded_norm is None else require_float(recorded_norm, "norm", minimum=0.0)
        if not math.isclose(computed_norm, normalized_norm, rel_tol=0.0, abs_tol=1e-12):
            msg = f"norm changed during vector decoding: expected {normalized_norm}, got {computed_norm}."
            raise ValueError(msg)
        expected_vector_checksum = _raw_vector_checksum(vector_bytes)
        supplied_vector_checksum = require_checksum(vector_checksum, "vector_checksum")
        if supplied_vector_checksum != expected_vector_checksum:
            msg = (
                "vector_checksum does not match the canonical state bytes: "
                f"expected {expected_vector_checksum}, got {supplied_vector_checksum}."
            )
            raise ValueError(msg)

        object.__setattr__(self, "target_instance_id", expected_id)
        object.__setattr__(self, "target_instance_spec_checksum", supplied_spec_checksum)
        object.__setattr__(self, "population_config_checksum", LEGACY_TARGET_POPULATION_CONFIG_CHECKSUM)
        object.__setattr__(self, "target_manifest_checksum", LEGACY_REPRODUCTION_MANIFEST_CHECKSUM)
        object.__setattr__(self, "parameter_checksum", supplied_parameter_checksum)
        object.__setattr__(self, "family_id", LEGACY_TARGET_FAMILY_ID)
        object.__setattr__(self, "stratum_id", LEGACY_TARGET_STRATUM_ID)
        object.__setattr__(self, "qubit_count", LEGACY_TARGET_QUBIT_COUNT)
        object.__setattr__(self, "seed", normalized_seed)
        object.__setattr__(self, "couplings", normalized_couplings)
        object.__setattr__(self, "fields", normalized_fields)
        object.__setattr__(self, "ground_energy", normalized_energy)
        object.__setattr__(self, "reference_status", LEGACY_TARGET_REFERENCE_STATUS)
        object.__setattr__(self, "archived_vector_checksum", None)
        object.__setattr__(self, "norm", normalized_norm)
        object.__setattr__(self, "vector_checksum", supplied_vector_checksum)
        object.__setattr__(self, "_state_vector_bytes", vector_bytes)
        object.__setattr__(self, "schema_version", LEGACY_TARGET_RECORD_SCHEMA_VERSION)

    def state_vector_copy(self) -> NDArray[np.complex128]:
        """Return a detached writable copy without imposing a global phase."""
        return np.frombuffer(self._state_vector_bytes, dtype=np.dtype("<c16")).astype(np.complex128, copy=True)

    def identity_dict(self) -> dict[str, object]:
        """Return the same agreement-ledger fields as ``MaterializedTarget``."""
        return {
            "target_instance_id": self.target_instance_id,
            "target_instance_spec_checksum": self.target_instance_spec_checksum,
            "population_config_checksum": self.population_config_checksum,
            "target_manifest_checksum": self.target_manifest_checksum,
            "parameter_checksum": self.parameter_checksum,
            "family_id": self.family_id,
            "stratum_id": self.stratum_id,
            "qubit_count": self.qubit_count,
            "norm": self.norm,
            "vector_checksum": self.vector_checksum,
        }

    def _content_dict(self) -> dict[str, object]:
        """Return the complete checksum-covered target payload."""
        return {
            "schema_version": self.schema_version,
            "target_instance_id": self.target_instance_id,
            "target_instance_spec_checksum": self.target_instance_spec_checksum,
            "population_config_checksum": self.population_config_checksum,
            "target_manifest_checksum": self.target_manifest_checksum,
            "parameter_checksum": self.parameter_checksum,
            "family_id": self.family_id,
            "stratum_id": self.stratum_id,
            "qubit_count": self.qubit_count,
            "seed": self.seed,
            "couplings": list(self.couplings),
            "fields": list(self.fields),
            "ground_energy": self.ground_energy,
            "reference_status": self.reference_status,
            "archived_vector_checksum": self.archived_vector_checksum,
            "norm": self.norm,
            "vector_checksum": self.vector_checksum,
            "state_vector": _encode_vector(self._state_vector_bytes),
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete reconstructed target record."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return a sealed detached JSON-native target record."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> LegacyMaterializedTarget:
        """Decode and checksum-verify an exact target record.

        Returns:
            Validated immutable target materialization.

        Raises:
            ValueError: If the schema, seal, or target identity differs.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_TARGET_KEYS, name="legacy materialized target")
        if mapping["schema_version"] != LEGACY_TARGET_RECORD_SCHEMA_VERSION:
            msg = f"schema_version must be {LEGACY_TARGET_RECORD_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        vector = _decode_vector(mapping["state_vector"])
        target = cls(
            seed=cast("int", mapping["seed"]),
            couplings=cast("Sequence[float]", mapping["couplings"]),
            fields=cast("Sequence[float]", mapping["fields"]),
            ground_energy=cast("float", mapping["ground_energy"]),
            vector=vector,
            target_instance_id=cast("str", mapping["target_instance_id"]),
            target_instance_spec_checksum=cast("str", mapping["target_instance_spec_checksum"]),
            population_config_checksum=cast("str", mapping["population_config_checksum"]),
            target_manifest_checksum=cast("str", mapping["target_manifest_checksum"]),
            parameter_checksum=cast("str", mapping["parameter_checksum"]),
            family_id=cast("str", mapping["family_id"]),
            stratum_id=cast("str", mapping["stratum_id"]),
            qubit_count=cast("int", mapping["qubit_count"]),
            reference_status=cast("str", mapping["reference_status"]),
            archived_vector_checksum=cast("None", mapping["archived_vector_checksum"]),
            vector_checksum=cast("str", mapping["vector_checksum"]),
            recorded_norm=cast("float", mapping["norm"]),
            _marker=_TARGET_CREATION_SENTINEL,
        )
        supplied_checksum = cast("str", mapping["content_checksum"])
        if target.content_checksum != supplied_checksum:
            msg = (
                "Legacy materialized-target checksum changed during normalization: "
                f"expected {supplied_checksum}, got {target.content_checksum}."
            )
            raise ValueError(msg)
        return target


def reconstruct_legacy_materialized_target(seed: int) -> LegacyMaterializedTarget:
    """Create one explicitly labelled WP19 reconstruction from the archived generator.

    Returns:
        Immutable reconstructed target with complete fixture identity.
    """
    regenerated = regenerate_legacy_tfim_target(seed)
    vector = regenerated.state_vector_copy()
    target_id = f"legacy_tfim_seed_{regenerated.seed}"
    return LegacyMaterializedTarget(
        seed=regenerated.seed,
        couplings=regenerated.couplings,
        fields=regenerated.fields,
        ground_energy=regenerated.ground_energy,
        vector=vector,
        target_instance_id=target_id,
        target_instance_spec_checksum=fixture_target_spec_checksum(
            "legacy_reproduction",
            target_id,
            LEGACY_TARGET_QUBIT_COUNT,
        ),
        population_config_checksum=LEGACY_TARGET_POPULATION_CONFIG_CHECKSUM,
        target_manifest_checksum=LEGACY_REPRODUCTION_MANIFEST_CHECKSUM,
        parameter_checksum=_parameter_checksum(regenerated.seed, regenerated.couplings, regenerated.fields),
        family_id=LEGACY_TARGET_FAMILY_ID,
        stratum_id=LEGACY_TARGET_STRATUM_ID,
        qubit_count=LEGACY_TARGET_QUBIT_COUNT,
        reference_status=LEGACY_TARGET_REFERENCE_STATUS,
        archived_vector_checksum=None,
        vector_checksum=_raw_vector_checksum(_canonical_vector_bytes(vector, "reconstructed state vector")),
        recorded_norm=None,
        _marker=_TARGET_CREATION_SENTINEL,
    )


@dataclass(frozen=True, slots=True)
class PhaseInvariantComparison:
    """Numerical result of an explicit-tolerance phase-invariant comparison."""

    matches: bool
    absolute_tolerance: float
    relative_tolerance: float
    overlap_magnitude: float
    phase_factor: complex
    maximum_absolute_error: float
    l2_error: float


def compare_statevectors_phase_invariant(
    reference: NDArray[np.complex128],
    candidate: NDArray[np.complex128],
    *,
    absolute_tolerance: float = LEGACY_TARGET_PHASE_INVARIANT_ATOL,
    relative_tolerance: float = LEGACY_TARGET_PHASE_INVARIANT_RTOL,
) -> PhaseInvariantComparison:
    """Compare normalized q8 vectors after removing the candidate's global phase.

    Args:
        reference: Stored reconstructed reference vector.
        candidate: Independently regenerated candidate vector.
        absolute_tolerance: Elementwise absolute tolerance.
        relative_tolerance: Elementwise relative tolerance.

    Returns:
        Immutable diagnostics containing the explicit tolerances and errors.
    """
    reference_bytes = _canonical_vector_bytes(reference, "reference")
    candidate_bytes = _canonical_vector_bytes(candidate, "candidate")
    normalized_reference = np.frombuffer(reference_bytes, dtype=np.dtype("<c16")).astype(np.complex128, copy=True)
    normalized_candidate = np.frombuffer(candidate_bytes, dtype=np.dtype("<c16")).astype(np.complex128, copy=True)
    atol = require_float(absolute_tolerance, "absolute_tolerance", minimum=0.0)
    rtol = require_float(relative_tolerance, "relative_tolerance", minimum=0.0)
    overlap = complex(np.vdot(normalized_candidate, normalized_reference))
    phase_factor = overlap / abs(overlap) if overlap else 1.0 + 0j
    aligned = normalized_candidate * phase_factor
    difference = normalized_reference - aligned
    return PhaseInvariantComparison(
        matches=bool(np.allclose(normalized_reference, aligned, atol=atol, rtol=rtol)),
        absolute_tolerance=atol,
        relative_tolerance=rtol,
        overlap_magnitude=float(abs(np.vdot(normalized_reference, normalized_candidate))),
        phase_factor=phase_factor,
        maximum_absolute_error=float(np.max(np.abs(difference))),
        l2_error=float(np.linalg.norm(difference)),
    )


def validate_legacy_target_regeneration(
    target: LegacyMaterializedTarget,
) -> PhaseInvariantComparison:
    """Regenerate one target under the sealed comparison policy and reject discrepancies.

    Returns:
        Phase-invariant vector comparison diagnostics.

    Raises:
        TypeError: If ``target`` has the wrong record type.
        ValueError: If regenerated evidence differs beyond the declared tolerance.
    """
    if not isinstance(target, LegacyMaterializedTarget):
        msg = f"target must be a LegacyMaterializedTarget, got {type(target).__name__}."
        raise TypeError(msg)
    regenerated = regenerate_legacy_tfim_target(target.seed)
    if regenerated.couplings != target.couplings or regenerated.fields != target.fields:
        msg = f"Legacy generator parameters differ for {target.target_instance_id!r}."
        raise ValueError(msg)
    atol = LEGACY_TARGET_PHASE_INVARIANT_ATOL
    rtol = LEGACY_TARGET_PHASE_INVARIANT_RTOL
    if not math.isclose(regenerated.ground_energy, target.ground_energy, rel_tol=rtol, abs_tol=atol):
        msg = (
            f"Legacy ground energy differs for {target.target_instance_id!r}: "
            f"stored={target.ground_energy}, regenerated={regenerated.ground_energy}."
        )
        raise ValueError(msg)
    comparison = compare_statevectors_phase_invariant(
        target.state_vector_copy(),
        regenerated.state_vector_copy(),
        absolute_tolerance=atol,
        relative_tolerance=rtol,
    )
    if not comparison.matches:
        msg = (
            f"Legacy target {target.target_instance_id!r} differs after phase alignment: "
            f"max_abs_error={comparison.maximum_absolute_error}, l2_error={comparison.l2_error}, "
            f"atol={atol}, rtol={rtol}."
        )
        raise ValueError(msg)
    return comparison


@dataclass(frozen=True, slots=True)
class LegacyTargetCollection:
    """Strictly sealed collection of exactly five WP19 reconstructed references."""

    reconstruction_runtime: LegacyReconstructionRuntime
    targets: tuple[LegacyMaterializedTarget, ...]
    missing_provenance: tuple[str, ...] = LEGACY_TARGET_MISSING_PROVENANCE
    schema_version: str = field(default=LEGACY_TARGET_COLLECTION_SCHEMA_VERSION, init=False)
    collection_id: str = field(default=LEGACY_TARGET_COLLECTION_ID, init=False)
    namespace: str = field(default=LEGACY_TARGET_NAMESPACE, init=False)
    reference_status: str = field(default=LEGACY_TARGET_REFERENCE_STATUS, init=False)
    archived_state_vectors_retained: bool = field(default=False, init=False)
    source_commit: str = field(default=LEGACY_TARGET_SOURCE_COMMIT, init=False)
    source_path: str = field(default=LEGACY_TARGET_SOURCE_PATH, init=False)
    source_content_checksum: str = field(default=LEGACY_TARGET_SOURCE_CHECKSUM, init=False)
    population_config_checksum: str = field(default=LEGACY_TARGET_POPULATION_CONFIG_CHECKSUM, init=False)

    def __post_init__(self) -> None:
        """Validate the exact ordered fixture population and missing-provenance disclosure.

        Raises:
            TypeError: If runtime or target records have the wrong types.
            ValueError: If the collection is incomplete, reordered, or mislabelled.
        """
        if not isinstance(self.reconstruction_runtime, LegacyReconstructionRuntime):
            msg = "reconstruction_runtime must be a LegacyReconstructionRuntime."
            raise TypeError(msg)
        targets = tuple(self.targets)
        if not all(isinstance(target, LegacyMaterializedTarget) for target in targets):
            msg = "targets must contain only LegacyMaterializedTarget records."
            raise TypeError(msg)
        target_ids = tuple(target.target_instance_id for target in targets)
        seeds = tuple(target.seed for target in targets)
        if target_ids != LEGACY_REPRODUCTION_TARGET_IDS or seeds != LEGACY_TARGET_SEEDS:
            msg = "targets must contain exactly q8 seeds 100, 200, 300, 400, and 500 in frozen order."
            raise ValueError(msg)
        missing = require_string_sequence(
            self.missing_provenance,
            "missing_provenance",
            minimum_length=1,
            unique=True,
        )
        if missing != LEGACY_TARGET_MISSING_PROVENANCE:
            msg = "missing_provenance must preserve the reviewed archived-evidence limitations verbatim."
            raise ValueError(msg)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "missing_provenance", missing)

    def _content_dict(self) -> dict[str, object]:
        """Return the complete checksum-covered collection payload."""
        return {
            "schema_version": self.schema_version,
            "collection_id": self.collection_id,
            "namespace": self.namespace,
            "reference_status": self.reference_status,
            "archived_state_vectors_retained": self.archived_state_vectors_retained,
            "source_commit": self.source_commit,
            "source_path": self.source_path,
            "source_content_checksum": self.source_content_checksum,
            "generator": thaw_json_mapping(LEGACY_TARGET_GENERATOR),
            "population_config_checksum": self.population_config_checksum,
            "phase_comparison": thaw_json_mapping(LEGACY_TARGET_PHASE_COMPARISON),
            "reconstruction_runtime": self.reconstruction_runtime.to_dict(),
            "missing_provenance": list(self.missing_provenance),
            "targets": [target.to_dict() for target in self.targets],
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the entire collection and all vector bytes."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return a sealed detached JSON-native collection."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical sealed JSON text."""
        return canonical_json(self.to_dict())

    def target(self, target_instance_id: str) -> LegacyMaterializedTarget:
        """Return one target by stable identifier.

        Raises:
            KeyError: If the target identifier is absent.
        """
        target_id = require_string(target_instance_id, "target_instance_id")
        for target in self.targets:
            if target.target_instance_id == target_id:
                return target
        raise KeyError(target_id)

    @classmethod
    def from_dict(cls, data: object) -> LegacyTargetCollection:
        """Construct and checksum-verify an exact collection document.

        Returns:
            Validated immutable five-target collection.

        Raises:
            TypeError: If targets are not represented by a sequence.
            ValueError: If a fixed field, nested seal, or collection seal differs.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_COLLECTION_KEYS, name="legacy target collection")
        fixed_values = {
            "schema_version": LEGACY_TARGET_COLLECTION_SCHEMA_VERSION,
            "collection_id": LEGACY_TARGET_COLLECTION_ID,
            "namespace": LEGACY_TARGET_NAMESPACE,
            "reference_status": LEGACY_TARGET_REFERENCE_STATUS,
            "archived_state_vectors_retained": False,
            "source_commit": LEGACY_TARGET_SOURCE_COMMIT,
            "source_path": LEGACY_TARGET_SOURCE_PATH,
            "source_content_checksum": LEGACY_TARGET_SOURCE_CHECKSUM,
            "population_config_checksum": LEGACY_TARGET_POPULATION_CONFIG_CHECKSUM,
        }
        for name, expected in fixed_values.items():
            if mapping[name] != expected:
                msg = f"{name} must be the frozen legacy target value {expected!r}."
                raise ValueError(msg)
        if mapping["generator"] != LEGACY_TARGET_GENERATOR:
            msg = "generator differs from the exact archived RandomState/J-then-h/MSB-kron/eigh semantics."
            raise ValueError(msg)
        if mapping["phase_comparison"] != LEGACY_TARGET_PHASE_COMPARISON:
            msg = "phase_comparison differs from the declared WP19 comparison rule and tolerance."
            raise ValueError(msg)
        require_git_commit(mapping["source_commit"], "source_commit")
        require_checksum(mapping["source_content_checksum"], "source_content_checksum")
        require_bool(mapping["archived_state_vectors_retained"], "archived_state_vectors_retained")
        target_values = mapping["targets"]
        if not isinstance(target_values, Sequence):
            msg = "targets must be a sequence."
            raise TypeError(msg)
        collection = cls(
            reconstruction_runtime=LegacyReconstructionRuntime.from_dict(mapping["reconstruction_runtime"]),
            targets=tuple(LegacyMaterializedTarget.from_dict(value) for value in target_values),
            missing_provenance=cast("tuple[str, ...]", mapping["missing_provenance"]),
        )
        supplied_checksum = cast("str", mapping["content_checksum"])
        if collection.content_checksum != supplied_checksum:
            msg = (
                "Legacy target-collection checksum changed during normalization: "
                f"expected {supplied_checksum}, got {collection.content_checksum}."
            )
            raise ValueError(msg)
        return collection

    @classmethod
    def from_json(cls, payload: str) -> LegacyTargetCollection:
        """Construct a collection from canonical sealed JSON text.

        Returns:
            Validated immutable five-target collection.

        Raises:
            TypeError: If ``payload`` is not text.
            ValueError: If UTF-8 content is empty, oversized, or invalid.
        """
        if type(payload) is not str:
            msg = f"payload must be a string, got {type(payload).__name__}."
            raise TypeError(msg)
        if not payload or len(payload) > MAX_LEGACY_TARGET_COLLECTION_BYTES:
            msg = "Legacy target collection JSON must be nonempty and within the trusted file size bound."
            raise ValueError(msg)
        if len(payload.encode("utf-8")) > MAX_LEGACY_TARGET_COLLECTION_BYTES:
            msg = "Legacy target collection UTF-8 JSON exceeds the trusted file size bound."
            raise ValueError(msg)
        return cls.from_dict(load_canonical_json_object(payload))


def reconstruct_legacy_target_collection(
    runtime: LegacyReconstructionRuntime | None = None,
) -> LegacyTargetCollection:
    """Reconstruct all five targets and attach current or supplied runtime provenance.

    Returns:
        Newly reconstructed five-target collection.

    Raises:
        TypeError: If supplied runtime provenance has the wrong record type.
    """
    resolved_runtime = capture_legacy_reconstruction_runtime() if runtime is None else runtime
    if not isinstance(resolved_runtime, LegacyReconstructionRuntime):
        msg = "runtime must be a LegacyReconstructionRuntime or None."
        raise TypeError(msg)
    return LegacyTargetCollection(
        reconstruction_runtime=resolved_runtime,
        targets=tuple(reconstruct_legacy_materialized_target(seed) for seed in LEGACY_TARGET_SEEDS),
    )


def _read_bounded_regular_file(path: Path) -> bytes:
    """Read the collection through a bounded, alias-resistant regular-file descriptor.

    Returns:
        At most :data:`MAX_LEGACY_TARGET_COLLECTION_BYTES` exact file bytes.

    Raises:
        ValueError: If the path is a symlink, is not regular, changes identity, cannot be read, or exceeds the bound.
    """
    for parent in path.parents:
        try:
            parent_metadata = parent.lstat()
        except OSError:
            continue
        if stat.S_ISLNK(parent_metadata.st_mode):
            msg = "Legacy target collection path components must never be symbolic links."
            raise ValueError(msg)
    try:
        path_metadata = path.lstat()
    except OSError as error:
        msg = f"Could not inspect canonical legacy target collection {path}: {error}."
        raise ValueError(msg) from error
    if stat.S_ISLNK(path_metadata.st_mode):
        msg = "Legacy target collection path must be a regular file, never a symbolic link."
        raise ValueError(msg)
    if not stat.S_ISREG(path_metadata.st_mode):
        msg = "Legacy target collection path must identify a regular file."
        raise ValueError(msg)

    flags = os.O_RDONLY
    for optional_flag in ("O_CLOEXEC", "O_NOFOLLOW", "O_NONBLOCK", "O_BINARY"):
        flags |= cast("int", getattr(os, optional_flag, 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        msg = f"Could not open canonical legacy target collection {path}: {error}."
        raise ValueError(msg) from error
    try:
        descriptor_metadata = os.fstat(descriptor)
    except OSError as error:
        os.close(descriptor)
        msg = f"Could not inspect opened legacy target collection {path}: {error}."
        raise ValueError(msg) from error
    if not stat.S_ISREG(descriptor_metadata.st_mode):
        os.close(descriptor)
        msg = "Opened legacy target collection descriptor must identify a regular file."
        raise ValueError(msg)
    if (path_metadata.st_dev, path_metadata.st_ino) != (
        descriptor_metadata.st_dev,
        descriptor_metadata.st_ino,
    ):
        os.close(descriptor)
        msg = "Legacy target collection path identity changed while it was being opened."
        raise ValueError(msg)
    try:
        stream = os.fdopen(descriptor, "rb", closefd=True)
    except OSError as error:
        os.close(descriptor)
        msg = f"Could not wrap opened legacy target collection {path}: {error}."
        raise ValueError(msg) from error
    try:
        with stream:
            payload = stream.read(MAX_LEGACY_TARGET_COLLECTION_BYTES + 1)
    except OSError as error:
        msg = f"Could not read canonical legacy target collection {path}: {error}."
        raise ValueError(msg) from error
    if not payload or len(payload) > MAX_LEGACY_TARGET_COLLECTION_BYTES:
        msg = "Legacy target collection file must be nonempty and within the trusted file size bound."
        raise ValueError(msg)
    return payload


def load_legacy_target_collection(
    path: Path = DEFAULT_LEGACY_TARGET_COLLECTION_PATH,
) -> LegacyTargetCollection:
    """Load the trusted checked-in canonical legacy target collection.

    Returns:
        Trusted immutable five-target collection.

    Raises:
        TypeError: If ``path`` is not a :class:`~pathlib.Path`.
        ValueError: If the path is aliased or non-regular, or the file is invalid or differs from the trusted checksum.
    """
    if not isinstance(path, Path):
        msg = f"path must be a pathlib.Path, got {type(path).__name__}."
        raise TypeError(msg)
    payload = _read_bounded_regular_file(path)
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        msg = "Legacy target collection file must contain canonical UTF-8 JSON."
        raise ValueError(msg) from error
    collection = LegacyTargetCollection.from_json(text)
    if collection.content_checksum != TRUSTED_LEGACY_TARGET_COLLECTION_CHECKSUM:
        msg = (
            "Checked-in legacy target-collection digest differs from the trusted runtime constant: "
            f"expected {TRUSTED_LEGACY_TARGET_COLLECTION_CHECKSUM}, got {collection.content_checksum}."
        )
        raise ValueError(msg)
    return collection


__all__ = [
    "DEFAULT_LEGACY_TARGET_COLLECTION_PATH",
    "LEGACY_TARGET_COLLECTION_ID",
    "LEGACY_TARGET_COLLECTION_SCHEMA_VERSION",
    "LEGACY_TARGET_FAMILY_ID",
    "LEGACY_TARGET_GENERATOR",
    "LEGACY_TARGET_GENERATOR_SCHEMA_VERSION",
    "LEGACY_TARGET_MISSING_PROVENANCE",
    "LEGACY_TARGET_NAMESPACE",
    "LEGACY_TARGET_PHASE_COMPARISON",
    "LEGACY_TARGET_PHASE_INVARIANT_ATOL",
    "LEGACY_TARGET_PHASE_INVARIANT_RTOL",
    "LEGACY_TARGET_POPULATION_CONFIG_CHECKSUM",
    "LEGACY_TARGET_QUBIT_COUNT",
    "LEGACY_TARGET_RECORD_SCHEMA_VERSION",
    "LEGACY_TARGET_REFERENCE_STATUS",
    "LEGACY_TARGET_RUNTIME_SCHEMA_VERSION",
    "LEGACY_TARGET_SEEDS",
    "LEGACY_TARGET_SOURCE_CHECKSUM",
    "LEGACY_TARGET_SOURCE_COMMIT",
    "LEGACY_TARGET_SOURCE_PATH",
    "MAX_LEGACY_TARGET_COLLECTION_BYTES",
    "MAX_LEGACY_TARGET_VECTOR_BASE64_BYTES",
    "MAX_LEGACY_TARGET_VECTOR_DECODED_BYTES",
    "TRUSTED_LEGACY_TARGET_COLLECTION_CHECKSUM",
    "LegacyMaterializedTarget",
    "LegacyReconstructionRuntime",
    "LegacyRegeneratedTarget",
    "LegacyTargetCollection",
    "PhaseInvariantComparison",
    "capture_legacy_reconstruction_runtime",
    "compare_statevectors_phase_invariant",
    "load_legacy_target_collection",
    "reconstruct_legacy_materialized_target",
    "reconstruct_legacy_target_collection",
    "regenerate_legacy_tfim_target",
    "validate_legacy_target_regeneration",
]
