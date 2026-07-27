# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Validated access to the checked-in state-preparation target fixtures."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, NoReturn, cast

import numpy as np

from .constants import (
    SUPPORTED_QUBIT_COUNTS,
    TARGET_FIXTURE_FORMAT,
    TARGET_GENERATION_SEEDS,
    TARGET_IDS,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

DEFAULT_TARGET_PATH = Path(__file__).parents[1] / "state_preparation_target_states.json"
NORM_ATOL = 1e-12
GROUND_ENERGY_ATOL = 1e-10
TFIM_EIGENSTATE_ATOL = 1e-10

_ROOT_KEYS = frozenset({
    "format",
    "generated_by",
    "complex_encoding",
    "basis_order",
    "global_phase",
    "numpy_version",
    "scipy_version",
    "qubit_counts",
    "target_ids",
    "targets",
})
_RECORD_KEYS = frozenset({"target_id", "num_qubits", "seed", "parameters", "norm", "state_vector"})
_ROOT_TEXT_METADATA = MappingProxyType({
    "generated_by": "benchmarks/generate_state_preparation_targets.py",
    "complex_encoding": "[real, imaginary]",
    "basis_order": "little_endian: amplitude index k has qubit i equal to bit i of k",
    "global_phase": "fixed so the largest-magnitude amplitude is positive real",
})
_GAUSSIAN_TEXT_PARAMETERS = MappingProxyType({
    "amplitude_encoding": "psi(x) = sqrt(f(x))",
    "distribution": "normal",
    "grid": "quantics endpoint-excluded grid on [0, 1): x=sum_{i=1}^N s_i*2**(-i)",
    "quantics_bit_order": "qubit 0 is s_1 and has weight 2**(-1); qubit i has weight 2**(-(i+1))",
    "source": "arXiv:2602.12042 Section 3.2.1 Probability distributions",
    "support_interval": "[0, 1)",
})
_TFIM_SPECS = MappingProxyType({
    "tfim_ferro": ("ferromagnetic", 1729, 0.5),
    "tfim_critical": ("critical", 2718, 1.0),
    "tfim_para": ("paramagnetic", 3141, 1.5),
})
_TFIM_TEXT_PARAMETERS = MappingProxyType({
    "boundary_conditions": "open",
    "finite_size_note": (
        "h/J=1 is the thermodynamic critical point; this record stores the finite-size open-chain ground state"
    ),
    "hamiltonian": "-J*sum_i Z_i Z_{i+1} - h*sum_i X_i",
    "model": "uniform_1d_transverse_field_ising_model",
    "pauli_z_eigenvalue_convention": "Z|b_i> = (-1)**b_i |b_i>",
    "site_order": "site i is qubit i and bit i of the little-endian basis index",
})
_HAAR_PARAMETERS = MappingProxyType({
    "distribution": "independent standard normal real and imaginary parts",
    "random_generator": "numpy.random.default_rng",
})
_MPS_TEXT_PARAMETERS = MappingProxyType({
    "random_generator": "numpy.random.default_rng",
    "quimb_reference": "qtn.MPS_rand_state(L=num_qubits, bond_dim=max_bond_dimension, normalize=True)",
    "tensor_distribution": "independent standard normal real entries",
})
_MPS_BOND_DIMENSIONS = MappingProxyType({
    "random_mps_bond2": 2,
    "random_mps_bond3": 3,
})


def _immutable_json(value: object) -> object:
    """Return an immutable, detached representation of JSON data."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _immutable_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_json(item) for item in value)
    return value


def _reject_json_constant(value: str) -> NoReturn:
    """Reject nonstandard JSON constants such as NaN and Infinity.

    Raises:
        ValueError: Always, because ``value`` is not a finite JSON number.
    """
    msg = f"Non-finite JSON constant {value!r} is not supported."
    raise ValueError(msg)


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while rejecting duplicate member names.

    Returns:
        A mapping containing the parsed members.

    Raises:
        ValueError: If a member name occurs more than once.
    """
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            msg = f"Duplicate JSON object key {key!r}."
            raise ValueError(msg)
        result[key] = value
    return result


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    """Validate and return a string-keyed mapping.

    Returns:
        The validated mapping.

    Raises:
        ValueError: If the value is not a string-keyed mapping.
    """
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        msg = f"{name} must be a JSON object with string keys."
        raise ValueError(msg)
    return cast("Mapping[str, object]", value)


def _require_exact_keys(value: Mapping[str, object], expected: frozenset[str], name: str) -> None:
    """Require a mapping to contain exactly the versioned schema keys.

    Raises:
        ValueError: If fields are missing or unsupported fields are present.
    """
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        msg = f"{name} fields do not match the v1 schema: missing={missing!r}, extra={extra!r}."
        raise ValueError(msg)


def _require_exact_int(value: object, name: str) -> int:
    """Validate an integer without accepting its Boolean subtype.

    Returns:
        The validated integer.

    Raises:
        TypeError: If the value is not an exact integer.
    """
    if type(value) is not int:
        msg = f"{name} must be an integer."
        raise TypeError(msg)
    return value


def _require_finite_real(value: object, name: str) -> float:
    """Validate and normalize a finite JSON real number.

    Returns:
        The value converted to a finite float.

    Raises:
        ValueError: If the value is non-finite or outside the finite float range.
        TypeError: If the value is not an integer or float, or is a Boolean.
    """
    if type(value) not in {int, float}:
        msg = f"{name} must be a JSON number."
        raise TypeError(msg)
    try:
        result = float(cast("int | float", value))
    except OverflowError as error:
        msg = f"{name} must be representable as a finite float."
        raise ValueError(msg) from error
    if not math.isfinite(result):
        msg = f"{name} must be finite."
        raise ValueError(msg)
    return result


def _require_exact_text(parameters: Mapping[str, object], expected: Mapping[str, str], name: str) -> None:
    """Validate fixed textual metadata.

    Raises:
        ValueError: If a textual value differs from the versioned definition.
    """
    for key, expected_value in expected.items():
        if parameters[key] != expected_value or type(parameters[key]) is not str:
            msg = f"{name}[{key!r}] must be {expected_value!r}."
            raise ValueError(msg)


def _require_real_sequence(
    value: object,
    name: str,
    *,
    length: int,
    expected_value: float,
) -> None:
    """Validate a fixed-length JSON array of equal finite real values.

    Raises:
        ValueError: If the length or any numeric value is incorrect.
        TypeError: If the value is not a JSON array or contains a non-number.
    """
    if not isinstance(value, list):
        msg = f"{name} must be a JSON array."
        raise TypeError(msg)
    if len(value) != length:
        msg = f"{name} must contain exactly {length} values."
        raise ValueError(msg)
    for index, item in enumerate(value):
        actual = _require_finite_real(item, f"{name}[{index}]")
        if not math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=0.0):
            msg = f"{name}[{index}] must be {expected_value}."
            raise ValueError(msg)


def _validate_gaussian_parameters(parameters: Mapping[str, object]) -> None:
    """Validate the fixed Gaussian target definition.

    Raises:
        ValueError: If a parameter differs from the benchmark definition.
    """
    expected_keys = frozenset((*_GAUSSIAN_TEXT_PARAMETERS, "mean", "standard_deviation"))
    _require_exact_keys(parameters, expected_keys, "Gaussian parameters")
    _require_exact_text(parameters, _GAUSSIAN_TEXT_PARAMETERS, "Gaussian parameters")
    if not math.isclose(
        _require_finite_real(parameters["mean"], "Gaussian parameters['mean']"),
        0.5,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        msg = "Gaussian parameters['mean'] must be 0.5."
        raise ValueError(msg)
    if not math.isclose(
        _require_finite_real(parameters["standard_deviation"], "Gaussian parameters['standard_deviation']"),
        0.1,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        msg = "Gaussian parameters['standard_deviation'] must be 0.1."
        raise ValueError(msg)


def _validate_haar_parameters(parameters: Mapping[str, object]) -> None:
    """Validate the fixed dense random-state generation rule."""
    _require_exact_keys(parameters, frozenset(_HAAR_PARAMETERS), "Haar-random parameters")
    _require_exact_text(parameters, _HAAR_PARAMETERS, "Haar-random parameters")


def _validate_mps_parameters(parameters: Mapping[str, object], *, num_qubits: int, target_id: str) -> None:
    """Validate one random-MPS generation definition.

    Raises:
        ValueError: If a bond dimension or fixed parameter is incorrect.
        TypeError: If a parameter has an invalid type.
    """
    expected_keys = frozenset((*_MPS_TEXT_PARAMETERS, "bond_dimensions", "max_bond_dimension"))
    _require_exact_keys(parameters, expected_keys, "Random-MPS parameters")
    _require_exact_text(parameters, _MPS_TEXT_PARAMETERS, "Random-MPS parameters")
    bond_dimension = _MPS_BOND_DIMENSIONS[target_id]
    if _require_exact_int(parameters["max_bond_dimension"], "max_bond_dimension") != bond_dimension:
        msg = f"Random-MPS target {target_id!r} must use maximum bond dimension {bond_dimension}."
        raise ValueError(msg)
    raw_bond_dimensions = parameters["bond_dimensions"]
    if not isinstance(raw_bond_dimensions, list):
        msg = "bond_dimensions must be a JSON array."
        raise TypeError(msg)
    expected_bond_dimensions = (1, *([bond_dimension] * (num_qubits - 1)), 1)
    actual_bond_dimensions = tuple(
        _require_exact_int(value, f"bond_dimensions[{index}]") for index, value in enumerate(raw_bond_dimensions)
    )
    if actual_bond_dimensions != expected_bond_dimensions:
        msg = f"bond_dimensions must be {expected_bond_dimensions!r} for target {(num_qubits, target_id)!r}."
        raise ValueError(msg)


def _tfim_energy_and_residual(
    vector: NDArray[np.complex128],
    *,
    num_qubits: int,
    transverse_field: float,
) -> tuple[float, float]:
    """Compute the open-chain uniform TFIM energy and eigenstate residual.

    Returns:
        The real energy expectation and ``||H|psi> - E|psi>||``.

    Raises:
        ValueError: If the numerical expectation has a material imaginary part.
    """
    basis_indices = np.arange(vector.size, dtype=np.int64)
    diagonal = np.zeros(vector.size, dtype=np.float64)
    for site in range(num_qubits - 1):
        left_bits = (basis_indices >> site) & 1
        right_bits = (basis_indices >> (site + 1)) & 1
        diagonal -= np.where(left_bits == right_bits, 1.0, -1.0)
    hamiltonian_vector = diagonal * vector
    for site in range(num_qubits):
        flipped_indices = basis_indices ^ (1 << site)
        hamiltonian_vector -= transverse_field * vector[flipped_indices]
    expectation = np.vdot(vector, hamiltonian_vector)
    if abs(expectation.imag) > GROUND_ENERGY_ATOL:
        msg = f"TFIM energy expectation has unexpected imaginary part {expectation.imag}."
        raise ValueError(msg)
    energy = float(expectation.real)
    residual = float(np.linalg.norm(hamiltonian_vector - energy * vector))
    return energy, residual


def _validate_tfim_parameters(
    parameters: Mapping[str, object],
    *,
    num_qubits: int,
    target_id: str,
    vector: NDArray[np.complex128],
) -> None:
    """Validate one TFIM target definition and its energy metadata.

    Raises:
        ValueError: If any fixed definition or derived energy is incorrect.
    """
    expected_keys = frozenset({
        *_TFIM_TEXT_PARAMETERS,
        "eigensolver_initial_vector_seed",
        "ground_energy",
        "h_over_j",
        "j_coupling",
        "j_couplings",
        "regime",
        "transverse_field",
        "transverse_fields",
    })
    _require_exact_keys(parameters, expected_keys, "TFIM parameters")
    _require_exact_text(parameters, _TFIM_TEXT_PARAMETERS, "TFIM parameters")
    expected_regime, eigensolver_base_seed, expected_field = _TFIM_SPECS[target_id]
    if parameters["regime"] != expected_regime or type(parameters["regime"]) is not str:
        msg = f"TFIM parameters['regime'] must be {expected_regime!r}."
        raise ValueError(msg)
    expected_eigensolver_seed = eigensolver_base_seed + 10_000 * num_qubits
    actual_eigensolver_seed = _require_exact_int(
        parameters["eigensolver_initial_vector_seed"],
        "eigensolver_initial_vector_seed",
    )
    if actual_eigensolver_seed != expected_eigensolver_seed:
        msg = f"eigensolver_initial_vector_seed must be {expected_eigensolver_seed}."
        raise ValueError(msg)
    for name, expected_value in (
        ("h_over_j", expected_field),
        ("j_coupling", 1.0),
        ("transverse_field", expected_field),
    ):
        if not math.isclose(
            _require_finite_real(parameters[name], name),
            expected_value,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            msg = f"TFIM parameter {name!r} must be {expected_value}."
            raise ValueError(msg)
    _require_real_sequence(
        parameters["j_couplings"],
        "j_couplings",
        length=num_qubits - 1,
        expected_value=1.0,
    )
    _require_real_sequence(
        parameters["transverse_fields"],
        "transverse_fields",
        length=num_qubits,
        expected_value=expected_field,
    )
    stored_energy = _require_finite_real(parameters["ground_energy"], "ground_energy")
    expected_energy, eigenstate_residual = _tfim_energy_and_residual(
        vector,
        num_qubits=num_qubits,
        transverse_field=expected_field,
    )
    if not math.isclose(stored_energy, expected_energy, rel_tol=0.0, abs_tol=GROUND_ENERGY_ATOL):
        msg = (
            f"TFIM ground_energy {stored_energy} does not match the state-vector expectation "
            f"{expected_energy} within absolute tolerance {GROUND_ENERGY_ATOL}."
        )
        raise ValueError(msg)
    if eigenstate_residual > TFIM_EIGENSTATE_ATOL:
        msg = (
            f"TFIM target {(num_qubits, target_id)!r} has eigenstate residual {eigenstate_residual}, "
            f"expected at most {TFIM_EIGENSTATE_ATOL}."
        )
        raise ValueError(msg)


def _validate_parameters(
    parameters: Mapping[str, object],
    *,
    num_qubits: int,
    target_id: str,
    vector: NDArray[np.complex128],
) -> Mapping[str, object]:
    """Validate and deeply freeze target-specific generation parameters.

    Returns:
        An immutable detached parameter mapping.
    """
    if target_id == "gaussian_mu0p5_sigma0p1":
        _validate_gaussian_parameters(parameters)
    elif target_id in _TFIM_SPECS:
        _validate_tfim_parameters(parameters, num_qubits=num_qubits, target_id=target_id, vector=vector)
    elif target_id.startswith("haar_random_"):
        _validate_haar_parameters(parameters)
    else:
        _validate_mps_parameters(parameters, num_qubits=num_qubits, target_id=target_id)
    return cast("Mapping[str, object]", _immutable_json(copy.deepcopy(dict(parameters))))


@dataclass(frozen=True, slots=True, init=False)
class TargetRecord:
    """One validated target state and its generation metadata."""

    num_qubits: int
    target_id: str
    seed: int | None
    parameters: Mapping[str, object]
    norm: float
    _state_vector_bytes: bytes = field(repr=False)

    def __init__(
        self,
        num_qubits: int,
        target_id: str,
        seed: int | None,
        parameters: Mapping[str, object],
        norm: float,
        state_vector: NDArray[np.complex128],
    ) -> None:
        """Validate and deeply freeze the target record.

        Raises:
            TypeError: If a core field has an invalid type.
            ValueError: If the record violates a target invariant.
        """
        num_qubits = _require_exact_int(num_qubits, "num_qubits")
        if type(target_id) is not str:
            msg = "target_id must be a string."
            raise TypeError(msg)
        _validate_filter(num_qubits, target_id)
        if seed is not None:
            seed = _require_exact_int(seed, "seed")
        expected_seed = TARGET_GENERATION_SEEDS[target_id]
        if seed != expected_seed:
            msg = f"Target {(num_qubits, target_id)!r} must use generation seed {expected_seed!r}."
            raise ValueError(msg)
        if not isinstance(state_vector, np.ndarray):
            msg = "state_vector must be a NumPy array."
            raise TypeError(msg)
        vector = np.asarray(state_vector, dtype=np.complex128)
        if vector.ndim != 1 or vector.shape != (2**num_qubits,):
            msg = f"Target {(num_qubits, target_id)!r} must contain exactly {2**num_qubits} amplitudes."
            raise ValueError(msg)
        if not np.all(np.isfinite(vector)):
            msg = f"Target {(num_qubits, target_id)!r} must contain finite amplitudes."
            raise ValueError(msg)
        vector_bytes = vector.tobytes()
        vector = np.frombuffer(vector_bytes, dtype=np.complex128)
        actual_norm = float(np.linalg.norm(vector))
        if not math.isclose(actual_norm, 1.0, rel_tol=0.0, abs_tol=NORM_ATOL):
            msg = (
                f"Target {(num_qubits, target_id)!r} has norm {actual_norm}, "
                f"expected 1 within absolute tolerance {NORM_ATOL}."
            )
            raise ValueError(msg)
        pivot = vector[int(np.argmax(np.abs(vector)))]
        if pivot.real <= 0.0 or abs(pivot.imag) > NORM_ATOL:
            msg = (
                f"Target {(num_qubits, target_id)!r} does not use the declared global-phase convention: "
                "its largest-magnitude amplitude must be positive real."
            )
            raise ValueError(msg)
        norm = _require_finite_real(norm, f"Stored norm for target {(num_qubits, target_id)!r}")
        if not math.isclose(norm, actual_norm, rel_tol=0.0, abs_tol=NORM_ATOL):
            msg = f"Stored norm for target {(num_qubits, target_id)!r} does not match its state vector."
            raise ValueError(msg)
        parameters = _require_mapping(parameters, "parameters")
        frozen_parameters = _validate_parameters(
            parameters,
            num_qubits=num_qubits,
            target_id=target_id,
            vector=vector,
        )
        object.__setattr__(self, "num_qubits", num_qubits)
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "norm", norm)
        object.__setattr__(self, "parameters", frozen_parameters)
        object.__setattr__(self, "_state_vector_bytes", vector_bytes)

    @property
    def key(self) -> tuple[int, str]:
        """Composite fixture key."""
        return self.num_qubits, self.target_id

    @property
    def state_vector(self) -> NDArray[np.complex128]:
        """Read-only view with independently mutable array metadata."""
        return np.frombuffer(self._state_vector_bytes, dtype=np.complex128)

    def state_vector_copy(self) -> NDArray[np.complex128]:
        """Return a mutable defensive copy of the target vector."""
        return np.frombuffer(self._state_vector_bytes, dtype=np.complex128).copy()


@dataclass(frozen=True, slots=True)
class TargetCollection:
    """An immutable validated collection of target fixtures."""

    fixture_format: str
    fixture_checksum: str
    metadata: Mapping[str, object]
    records: tuple[TargetRecord, ...]
    _index: Mapping[tuple[int, str], TargetRecord] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Build the immutable composite-key index.

        Raises:
            TypeError: If a collection member is not a target record.
            ValueError: If the format, checksum, or composite-key set is invalid.
        """
        if self.fixture_format != TARGET_FIXTURE_FORMAT or type(self.fixture_format) is not str:
            msg = f"Target collection format must be {TARGET_FIXTURE_FORMAT!r}."
            raise ValueError(msg)
        if (
            type(self.fixture_checksum) is not str
            or not self.fixture_checksum.startswith("sha256:")
            or len(self.fixture_checksum) != 71
            or any(character not in "0123456789abcdef" for character in self.fixture_checksum.removeprefix("sha256:"))
        ):
            msg = "Target collection checksum must be a lowercase sha256 checksum."
            raise ValueError(msg)
        if not isinstance(self.metadata, Mapping):
            msg = "Target collection metadata must be a mapping."
            raise TypeError(msg)
        frozen_metadata = cast("Mapping[str, object]", _immutable_json(copy.deepcopy(dict(self.metadata))))
        records = tuple(self.records)
        index: dict[tuple[int, str], TargetRecord] = {}
        for record in records:
            if not isinstance(record, TargetRecord):
                msg = "Target collection records must contain only TargetRecord instances."
                raise TypeError(msg)
            key = (record.num_qubits, record.target_id)
            if key in index:
                msg = f"Duplicate target record {key!r}."
                raise ValueError(msg)
            index[key] = record
        expected_keys = {(num_qubits, target_id) for num_qubits in SUPPORTED_QUBIT_COUNTS for target_id in TARGET_IDS}
        if index.keys() != expected_keys:
            missing = sorted(expected_keys - index.keys())
            extra = sorted(index.keys() - expected_keys)
            msg = f"Target fixture key set is incomplete: missing={missing!r}, extra={extra!r}."
            raise ValueError(msg)
        object.__setattr__(self, "metadata", frozen_metadata)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "_index", MappingProxyType(index))

    @property
    def format(self) -> str:
        """Fixture format alias matching the JSON envelope field."""
        return self.fixture_format

    @property
    def checksum(self) -> str:
        """Raw-file checksum alias for concise reporting."""
        return self.fixture_checksum

    def load_target(self, num_qubits: int, target_id: str) -> TargetRecord:
        """Look up one target by its unique key.

        Returns:
            The matching immutable target record.
        """
        _validate_filter(num_qubits, target_id)
        return self._index[num_qubits, target_id]

    def iter_targets(self, *, num_qubits: int | None = None, target_id: str | None = None) -> Iterator[TargetRecord]:
        """Iterate over targets, optionally filtered by either key component.

        Returns:
            An iterator in fixture order.
        """
        _validate_filter(num_qubits, target_id)
        return (
            record
            for record in self.records
            if (num_qubits is None or record.num_qubits == num_qubits)
            and (target_id is None or record.target_id == target_id)
        )


def _validate_filter(num_qubits: int | None, target_id: str | None) -> None:
    """Validate optional lookup filters.

    Raises:
        ValueError: If either supplied filter is unsupported.
    """
    if num_qubits is not None and (type(num_qubits) is not int or num_qubits not in SUPPORTED_QUBIT_COUNTS):
        msg = f"Unsupported qubit count {num_qubits!r}; expected one of {SUPPORTED_QUBIT_COUNTS}."
        raise ValueError(msg)
    if target_id is not None and (not isinstance(target_id, str) or target_id not in TARGET_IDS):
        msg = f"Unsupported target identifier {target_id!r}."
        raise ValueError(msg)


def _decode_state_vector(value: object, *, num_qubits: int, key: tuple[int, str]) -> NDArray[np.complex128]:
    """Decode and validate a complex-pair state vector.

    Returns:
        An immutable complex vector.

    Raises:
        ValueError: If the encoding, dimension, amplitudes, or norm is invalid.
    """
    if not isinstance(value, list) or len(value) != 2**num_qubits:
        msg = f"Target {key!r} must contain exactly {2**num_qubits} amplitudes."
        raise ValueError(msg)
    amplitudes: list[complex] = []
    for index, pair in enumerate(value):
        if not isinstance(pair, list) or len(pair) != 2:
            msg = f"Amplitude {index} of target {key!r} must be a [real, imaginary] pair."
            raise ValueError(msg)
        real, imaginary = pair
        try:
            finite_real = _require_finite_real(real, f"Amplitude {index} real component")
            finite_imaginary = _require_finite_real(imaginary, f"Amplitude {index} imaginary component")
        except TypeError as error:
            msg = f"Amplitude {index} of target {key!r} must contain two JSON numbers."
            raise ValueError(msg) from error
        except ValueError as error:
            msg = f"Amplitude {index} of target {key!r} must be finite."
            raise ValueError(msg) from error
        amplitudes.append(complex(finite_real, finite_imaginary))
    decoded = np.asarray(amplitudes, dtype=np.complex128)
    vector = np.frombuffer(decoded.tobytes(), dtype=np.complex128)
    norm = float(np.linalg.norm(vector))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=NORM_ATOL):
        msg = f"Target {key!r} has norm {norm}, expected 1 within absolute tolerance {NORM_ATOL}."
        raise ValueError(msg)
    return vector


def _parse_record(value: object) -> TargetRecord:
    """Parse and validate one target record.

    Returns:
        The validated immutable record.

    Raises:
        ValueError: If a required field or scientific invariant is invalid.
        TypeError: If a record field has an invalid type.
    """
    record = _require_mapping(value, "target record")
    _require_exact_keys(record, _RECORD_KEYS, "Target record")
    num_qubits = _require_exact_int(record["num_qubits"], "num_qubits")
    target_id = record["target_id"]
    seed = record["seed"]
    parameters = _require_mapping(record["parameters"], "parameters")
    stored_norm = record["norm"]
    encoded_vector = record["state_vector"]
    if type(target_id) is not str:
        msg = "target_id must be a string."
        raise TypeError(msg)
    _validate_filter(num_qubits, target_id)
    expected_seed = TARGET_GENERATION_SEEDS[target_id]
    if seed is not None:
        seed = _require_exact_int(seed, "seed")
    if seed != expected_seed:
        msg = f"Target {(num_qubits, target_id)!r} must use generation seed {expected_seed!r}."
        raise ValueError(msg)
    vector = _decode_state_vector(encoded_vector, num_qubits=num_qubits, key=(num_qubits, target_id))
    stored_norm_value = _require_finite_real(stored_norm, f"Stored norm for target {(num_qubits, target_id)!r}")
    return TargetRecord(
        num_qubits=num_qubits,
        target_id=target_id,
        seed=seed,
        parameters=parameters,
        norm=stored_norm_value,
        state_vector=vector,
    )


def _validate_root(root: Mapping[str, object]) -> None:
    """Validate the versioned target-fixture envelope.

    Raises:
        TypeError: If a declaration has an invalid JSON type.
        ValueError: If metadata or declarations differ from the v1 contract.
    """
    _require_exact_keys(root, _ROOT_KEYS, "Target fixture")
    if root["format"] != TARGET_FIXTURE_FORMAT or type(root["format"]) is not str:
        msg = f"Unsupported target fixture format {root['format']!r}; expected {TARGET_FIXTURE_FORMAT!r}."
        raise ValueError(msg)
    _require_exact_text(root, _ROOT_TEXT_METADATA, "Target fixture")
    for version_name in ("numpy_version", "scipy_version"):
        version = root[version_name]
        if type(version) is not str or not version.strip():
            msg = f"Target fixture field {version_name!r} must be a nonempty string."
            raise ValueError(msg)
    qubit_counts = root["qubit_counts"]
    if (
        not isinstance(qubit_counts, list)
        or any(type(value) is not int for value in qubit_counts)
        or tuple(qubit_counts) != SUPPORTED_QUBIT_COUNTS
    ):
        msg = f"Target fixture qubit_counts must be {list(SUPPORTED_QUBIT_COUNTS)!r}."
        raise ValueError(msg)
    target_ids = root["target_ids"]
    if (
        not isinstance(target_ids, list)
        or any(type(value) is not str for value in target_ids)
        or tuple(target_ids) != TARGET_IDS
    ):
        msg = f"Target fixture target_ids must be {list(TARGET_IDS)!r}."
        raise ValueError(msg)
    if not isinstance(root["targets"], list):
        msg = "Target fixture field 'targets' must be an array."
        raise TypeError(msg)


def load_target_collection(path: Path | None = None) -> TargetCollection:
    """Load and fully validate a target fixture collection.

    Args:
        path: JSON fixture path. The checked-in fixture is used by default.

    Returns:
        An immutable validated collection.

    Raises:
        ValueError: If the fixture cannot be read or fails validation.
    """
    fixture_path = DEFAULT_TARGET_PATH if path is None else Path(path)
    try:
        encoded_document = fixture_path.read_bytes()
    except OSError as error:
        msg = f"Could not load target fixture {fixture_path}: {error}"
        raise ValueError(msg) from error
    try:
        document_text = encoded_document.decode("utf-8")
    except UnicodeError as error:
        msg = f"Could not decode target fixture {fixture_path} as UTF-8."
        raise ValueError(msg) from error
    try:
        document = json.loads(
            document_text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        msg = f"Could not parse target fixture {fixture_path}: {error}"
        raise ValueError(msg) from error
    root = _require_mapping(document, "target fixture")
    _validate_root(root)
    raw_records = cast("list[object]", root["targets"])
    records = tuple(_parse_record(value) for value in raw_records)
    metadata = {key: copy.deepcopy(value) for key, value in root.items() if key != "targets"}
    return TargetCollection(
        fixture_format=TARGET_FIXTURE_FORMAT,
        fixture_checksum=f"sha256:{hashlib.sha256(encoded_document).hexdigest()}",
        metadata=cast("Mapping[str, object]", _immutable_json(metadata)),
        records=records,
    )


def load_target(num_qubits: int, target_id: str) -> TargetRecord:
    """Load one target from the checked-in fixture.

    Returns:
        The matching immutable target record.
    """
    return load_target_collection().load_target(num_qubits, target_id)


def iter_targets(*, num_qubits: int | None = None, target_id: str | None = None) -> Iterator[TargetRecord]:
    """Iterate over checked-in targets with optional filters.

    Returns:
        An iterator in fixture order.
    """
    return load_target_collection().iter_targets(num_qubits=num_qubits, target_id=target_id)


__all__ = [
    "DEFAULT_TARGET_PATH",
    "NORM_ATOL",
    "TargetCollection",
    "TargetRecord",
    "iter_targets",
    "load_target",
    "load_target_collection",
]
