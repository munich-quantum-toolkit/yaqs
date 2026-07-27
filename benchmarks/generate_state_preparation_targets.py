# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Generate the fixed target states for the state-preparation benchmark suite."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator, eigsh

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

QUBIT_COUNTS = (6, 12)
TARGET_IDS = (
    "gaussian_mu0p5_sigma0p1",
    "tfim_ferro",
    "tfim_critical",
    "tfim_para",
    "haar_random_1",
    "haar_random_2",
    "haar_random_3",
    "random_mps_bond2",
    "random_mps_bond3",
)

DEFAULT_OUTPUT_PATH = Path(__file__).with_name("state_preparation_target_states.json")
GAUSSIAN_MEAN = 0.5
GAUSSIAN_STANDARD_DEVIATION = 0.1
STATE_VECTOR_RTOL = 1e-9
STATE_VECTOR_ATOL = 1e-10
NORM_ATOL = 1e-12
GROUND_ENERGY_ATOL = 1e-10
VERSION_METADATA_KEYS = ("numpy_version", "scipy_version")


@dataclass(frozen=True, slots=True)
class TfimSpec:
    """Parameters for one uniform transverse-field Ising target."""

    target_id: str
    regime: str
    eigensolver_base_seed: int
    h_over_j: float


@dataclass(frozen=True, slots=True)
class RandomStateSpec:
    """Parameters for one dense random target state."""

    target_id: str
    seed: int


@dataclass(frozen=True, slots=True)
class RandomMpsSpec:
    """Parameters for one random MPS target state."""

    target_id: str
    seed: int
    bond_dimension: int


TFIM_SPECS = (
    TfimSpec(target_id="tfim_ferro", regime="ferromagnetic", eigensolver_base_seed=1729, h_over_j=0.5),
    TfimSpec(target_id="tfim_critical", regime="critical", eigensolver_base_seed=2718, h_over_j=1.0),
    TfimSpec(target_id="tfim_para", regime="paramagnetic", eigensolver_base_seed=3141, h_over_j=1.5),
)
HAAR_RANDOM_SPECS = (
    RandomStateSpec(target_id="haar_random_1", seed=4001),
    RandomStateSpec(target_id="haar_random_2", seed=4002),
    RandomStateSpec(target_id="haar_random_3", seed=4003),
)
RANDOM_MPS_SPECS = (
    RandomMpsSpec(target_id="random_mps_bond2", seed=5002, bond_dimension=2),
    RandomMpsSpec(target_id="random_mps_bond3", seed=5003, bond_dimension=3),
)


class _TfimLinearOperator(LinearOperator):
    """Sparse matrix-free representation of the open-chain TFIM Hamiltonian."""

    def __init__(
        self,
        diagonal: NDArray[np.float64],
        fields: NDArray[np.float64],
        flipped_indices: tuple[NDArray[np.int64], ...],
    ) -> None:
        self._diagonal = diagonal
        self._fields = fields
        self._flipped_indices = flipped_indices
        super().__init__(dtype=np.dtype(np.float64), shape=(diagonal.size, diagonal.size))

    def _matvec(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply the TFIM Hamiltonian to a vector.

        Returns:
            Matrix-vector product.
        """
        result = self._diagonal * x
        for field, flipped in zip(self._fields, self._flipped_indices, strict=True):
            result -= field * x[flipped]
        return result


def _normalize_and_fix_global_phase(state: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Normalize a state and choose a deterministic global phase.

    Returns:
        Normalized copy of the input state with fixed global phase.

    Raises:
        ValueError: If the input state is the zero vector.
    """
    norm = np.linalg.norm(state)
    if norm == 0:
        msg = "Cannot normalize the zero vector."
        raise ValueError(msg)

    normalized = np.asarray(state / norm, dtype=np.complex128)
    pivot = int(np.argmax(np.abs(normalized)))
    pivot_value = normalized[pivot]
    if np.abs(pivot_value) > 0:
        normalized *= np.conjugate(pivot_value) / np.abs(pivot_value)
    return np.asarray(normalized / np.linalg.norm(normalized), dtype=np.complex128)


def _paper_quantics_grid(num_qubits: int) -> NDArray[np.float64]:
    """Return the paper's quantics grid on [0, 1).

    The output is ordered according to the JSON basis convention: amplitude
    index ``k`` has qubit ``i`` equal to bit ``i`` of ``k``. To match the paper,
    qubit 0 is the coarsest quantics bit ``s_1``, so qubit ``i`` has weight
    ``2**(-(i + 1))``.

    Returns:
        Quantics grid values ordered by the little-endian JSON basis convention.
    """
    dim = 2**num_qubits
    indices = np.arange(dim, dtype=np.uint64)
    grid = np.zeros(dim, dtype=np.float64)
    for qubit in range(num_qubits):
        bits = ((indices >> np.uint64(qubit)) & np.uint64(1)).astype(np.float64)
        grid += bits * (2.0 ** (-(qubit + 1)))
    return grid


def _gaussian_state(num_qubits: int) -> NDArray[np.complex128]:
    """Return the paper-style Gaussian probability-distribution target.

    This implements Section 3.2.1 of arXiv:2602.12042 for ``N=num_qubits``,
    but as a dense state vector instead of an MPS/TCI approximation. The paper
    encodes ``psi(x)=sqrt(f(x))``, where ``f`` is a normal distribution with
    mean 0.5 and standard deviation 0.1, on the quantics grid
    ``x=sum_{i=1}^N s_i 2^{-i}`` in ``[0, 1)``.

    Returns:
        Normalized dense Gaussian target state.
    """
    grid = _paper_quantics_grid(num_qubits)
    mu = GAUSSIAN_MEAN
    sigma = GAUSSIAN_STANDARD_DEVIATION

    # Square root of the Gaussian probability density. The constant prefactor
    # is omitted because the state is normalized below.
    amplitudes = np.exp(-((grid - mu) ** 2) / (4.0 * sigma**2))
    return _normalize_and_fix_global_phase(np.asarray(amplitudes, dtype=np.complex128))


def _tfim_parameters(num_qubits: int, spec: TfimSpec) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return uniform open-chain TFIM couplings and transverse fields."""
    couplings = np.ones(num_qubits - 1, dtype=np.float64)
    fields = np.full(num_qubits, spec.h_over_j, dtype=np.float64)
    return couplings, fields


def _tfim_ground_state(
    num_qubits: int, spec: TfimSpec
) -> tuple[NDArray[np.complex128], float, NDArray[np.float64], NDArray[np.float64]]:
    """Return the lowest-energy state of the open-boundary uniform 1D TFIM."""
    couplings, fields = _tfim_parameters(num_qubits, spec)
    dim = 2**num_qubits
    basis_indices = np.arange(dim, dtype=np.int64)

    diagonal = np.zeros(dim, dtype=np.float64)
    for site, coupling in enumerate(couplings):
        left_bits = (basis_indices >> site) & 1
        right_bits = (basis_indices >> (site + 1)) & 1
        diagonal -= coupling * np.where(left_bits == right_bits, 1.0, -1.0)

    flipped_indices = tuple(basis_indices ^ (1 << site) for site in range(num_qubits))

    operator = _TfimLinearOperator(diagonal, fields, flipped_indices)
    rng = np.random.default_rng(spec.eigensolver_base_seed + 10_000 * num_qubits)
    initial_vector = rng.standard_normal(dim)
    initial_vector /= np.linalg.norm(initial_vector)
    eigenvalues, eigenvectors = eigsh(
        operator,
        k=1,
        which="SA",
        v0=initial_vector,
        tol=1e-13,
        maxiter=20_000,
    )
    state = _normalize_and_fix_global_phase(np.asarray(eigenvectors[:, 0], dtype=np.complex128))
    return state, float(eigenvalues[0]), couplings, fields


def _dense_haar_random_state(num_qubits: int, seed: int) -> NDArray[np.complex128]:
    """Return a normalized dense complex Gaussian random state."""
    rng = np.random.default_rng(seed)
    dim = 2**num_qubits
    state = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    return _normalize_and_fix_global_phase(np.asarray(state, dtype=np.complex128))


def _mps_bond_dimensions(num_qubits: int, max_bond_dimension: int) -> tuple[int, ...]:
    """Return Quimb-style open-boundary bond dimensions for a qubit MPS."""
    return (1, *([max_bond_dimension] * (num_qubits - 1)), 1)


def _random_mps_state(
    num_qubits: int, seed: int, bond_dimension: int
) -> tuple[NDArray[np.complex128], tuple[int, ...]]:
    """Return a dense state vector from Quimb-style random real MPS tensors."""
    rng = np.random.default_rng(seed)
    bond_dimensions = _mps_bond_dimensions(num_qubits, bond_dimension)
    tensors = [
        np.asarray(
            rng.standard_normal((2, bond_dimensions[site], bond_dimensions[site + 1])),
            dtype=np.float64,
        )
        for site in range(num_qubits)
    ]

    dense_tensor = tensors[0][:, 0, :]
    for tensor in tensors[1:]:
        dense_tensor = np.tensordot(dense_tensor, tensor, axes=([-1], [1]))
    dense_tensor = np.squeeze(dense_tensor, axis=-1)

    # The JSON vector uses little-endian computational-basis indexing:
    # amplitude k belongs to the bit string with qubit i equal to bit i of k.
    state = np.transpose(dense_tensor, axes=tuple(reversed(range(num_qubits)))).reshape(-1)
    return _normalize_and_fix_global_phase(np.asarray(state, dtype=np.complex128)), bond_dimensions


def _state_to_json_vector(state: NDArray[np.complex128]) -> list[list[float]]:
    """Encode a complex state vector as JSON-compatible [real, imag] pairs.

    Returns:
        JSON-compatible list of real-imaginary amplitude pairs.
    """
    return [[float(amplitude.real), float(amplitude.imag)] for amplitude in state]


def json_vector_to_state(vector: Sequence[Sequence[float]]) -> NDArray[np.complex128]:
    """Decode a JSON [real, imag] vector into a complex NumPy array.

    Returns:
        Complex NumPy state vector.
    """
    return np.asarray([complex(pair[0], pair[1]) for pair in vector], dtype=np.complex128)


def generate_target_records(num_qubits: Sequence[int] = QUBIT_COUNTS) -> list[dict[str, object]]:
    """Generate all target-state records for the requested qubit counts.

    Returns:
        List of JSON-serializable target-state records.
    """
    records: list[dict[str, object]] = []

    for n_qubits in num_qubits:
        gaussian = _gaussian_state(n_qubits)
        records.append({
            "target_id": "gaussian_mu0p5_sigma0p1",
            "num_qubits": n_qubits,
            "seed": None,
            "parameters": {
                "amplitude_encoding": "psi(x) = sqrt(f(x))",
                "distribution": "normal",
                "grid": "quantics endpoint-excluded grid on [0, 1): x=sum_{i=1}^N s_i*2**(-i)",
                "mean": GAUSSIAN_MEAN,
                "quantics_bit_order": "qubit 0 is s_1 and has weight 2**(-1); qubit i has weight 2**(-(i+1))",
                "source": "arXiv:2602.12042 Section 3.2.1 Probability distributions",
                "standard_deviation": GAUSSIAN_STANDARD_DEVIATION,
                "support_interval": "[0, 1)",
            },
            "norm": float(np.linalg.norm(gaussian)),
            "state_vector": _state_to_json_vector(gaussian),
        })

        for spec in TFIM_SPECS:
            state, energy, couplings, fields = _tfim_ground_state(n_qubits, spec)
            records.append({
                "target_id": spec.target_id,
                "num_qubits": n_qubits,
                "seed": None,
                "parameters": {
                    "boundary_conditions": "open",
                    "eigensolver_initial_vector_seed": spec.eigensolver_base_seed + 10_000 * n_qubits,
                    "finite_size_note": "h/J=1 is the thermodynamic critical point; "
                    "this record stores the finite-size open-chain ground state",
                    "ground_energy": float(energy),
                    "h_over_j": spec.h_over_j,
                    "hamiltonian": "-J*sum_i Z_i Z_{i+1} - h*sum_i X_i",
                    "j_coupling": 1.0,
                    "j_couplings": [float(value) for value in couplings],
                    "model": "uniform_1d_transverse_field_ising_model",
                    "pauli_z_eigenvalue_convention": "Z|b_i> = (-1)**b_i |b_i>",
                    "regime": spec.regime,
                    "site_order": "site i is qubit i and bit i of the little-endian basis index",
                    "transverse_field": spec.h_over_j,
                    "transverse_fields": [float(value) for value in fields],
                },
                "norm": float(np.linalg.norm(state)),
                "state_vector": _state_to_json_vector(state),
            })

        for spec in HAAR_RANDOM_SPECS:
            state = _dense_haar_random_state(n_qubits, spec.seed)
            records.append({
                "target_id": spec.target_id,
                "num_qubits": n_qubits,
                "seed": spec.seed,
                "parameters": {
                    "distribution": "independent standard normal real and imaginary parts",
                    "random_generator": "numpy.random.default_rng",
                },
                "norm": float(np.linalg.norm(state)),
                "state_vector": _state_to_json_vector(state),
            })

        for spec in RANDOM_MPS_SPECS:
            state, bond_dimensions = _random_mps_state(n_qubits, spec.seed, spec.bond_dimension)
            records.append({
                "target_id": spec.target_id,
                "num_qubits": n_qubits,
                "seed": spec.seed,
                "parameters": {
                    "bond_dimensions": list(bond_dimensions),
                    "max_bond_dimension": spec.bond_dimension,
                    "random_generator": "numpy.random.default_rng",
                    "quimb_reference": "qtn.MPS_rand_state(L=num_qubits, bond_dim=max_bond_dimension, normalize=True)",
                    "tensor_distribution": "independent standard normal real entries",
                },
                "norm": float(np.linalg.norm(state)),
                "state_vector": _state_to_json_vector(state),
            })

    return records


def generate_target_data(num_qubits: Sequence[int] = QUBIT_COUNTS) -> dict[str, object]:
    """Generate the complete JSON-serializable target-state payload.

    Returns:
        Complete JSON-serializable target-state payload.
    """
    return {
        "format": "yaqs.state_preparation_targets.v1",
        "generated_by": "benchmarks/generate_state_preparation_targets.py",
        "complex_encoding": "[real, imaginary]",
        "basis_order": "little_endian: amplitude index k has qubit i equal to bit i of k",
        "global_phase": "fixed so the largest-magnitude amplitude is positive real",
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "qubit_counts": list(num_qubits),
        "target_ids": list(TARGET_IDS),
        "targets": generate_target_records(num_qubits),
    }


def _numeric_values_match(actual: object, expected: object, *, atol: float, rtol: float = 0.0) -> bool:
    """Return whether two finite scalar values agree within the requested tolerance."""
    if (
        isinstance(actual, bool)
        or isinstance(expected, bool)
        or not isinstance(actual, (int, float))
        or not isinstance(expected, (int, float))
    ):
        return False
    actual_float = float(actual)
    expected_float = float(expected)
    return bool(
        np.isfinite(actual_float)
        and np.isfinite(expected_float)
        and np.isclose(actual_float, expected_float, atol=atol, rtol=rtol)
    )


def _state_vectors_match(actual: object, expected: object) -> bool:
    """Return whether two JSON-encoded state vectors agree numerically."""
    try:
        actual_array = np.asarray(actual, dtype=np.float64)
        expected_array = np.asarray(expected, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        actual_array.shape == expected_array.shape
        and actual_array.ndim == 2
        and actual_array.shape[1:] == (2,)
        and np.all(np.isfinite(actual_array))
        and np.all(np.isfinite(expected_array))
        and np.allclose(
            actual_array,
            expected_array,
            atol=STATE_VECTOR_ATOL,
            rtol=STATE_VECTOR_RTOL,
        )
    )


def target_data_matches(actual: object, expected: dict[str, object]) -> bool:
    """Compare stored target data with generated data while ignoring environment-only variation.

    NumPy and SciPy versions are retained as generation provenance but do not
    determine freshness. State-vector amplitudes, recorded norms, and TFIM
    ground energies are derived numerical values and are compared with tight
    tolerances. All benchmark-defining metadata and parameters must match
    exactly.

    Returns:
        Whether the stored payload is a fresh numerical match.
    """
    if not isinstance(actual, dict):
        return False

    actual_payload = copy.deepcopy(actual)
    expected_payload = copy.deepcopy(expected)
    for key in VERSION_METADATA_KEYS:
        actual_version = actual_payload.pop(key, None)
        expected_payload.pop(key, None)
        if not isinstance(actual_version, str) or not actual_version:
            return False

    actual_targets = actual_payload.pop("targets", None)
    expected_targets = expected_payload.pop("targets", None)
    if (
        actual_payload != expected_payload
        or not isinstance(actual_targets, list)
        or not isinstance(expected_targets, list)
    ):
        return False
    if len(actual_targets) != len(expected_targets):
        return False

    for actual_record, expected_record in zip(actual_targets, expected_targets, strict=True):
        if not isinstance(actual_record, dict) or not isinstance(expected_record, dict):
            return False
        actual_record_copy = copy.deepcopy(actual_record)
        expected_record_copy = copy.deepcopy(expected_record)

        actual_vector = actual_record_copy.pop("state_vector", None)
        expected_vector = expected_record_copy.pop("state_vector", None)
        if not _state_vectors_match(actual_vector, expected_vector):
            return False

        actual_norm = actual_record_copy.pop("norm", None)
        expected_norm = expected_record_copy.pop("norm", None)
        if not _numeric_values_match(actual_norm, expected_norm, atol=NORM_ATOL):
            return False

        actual_parameters = actual_record_copy.get("parameters")
        expected_parameters = expected_record_copy.get("parameters")
        if isinstance(expected_parameters, dict) and "ground_energy" in expected_parameters:
            if not isinstance(actual_parameters, dict):
                return False
            actual_energy = actual_parameters.pop("ground_energy", None)
            expected_energy = expected_parameters.pop("ground_energy", None)
            if not _numeric_values_match(actual_energy, expected_energy, atol=GROUND_ENERGY_ATOL):
                return False

        if actual_record_copy != expected_record_copy:
            return False

    return True


def main(argv: Sequence[str] | None = None) -> int:
    """Generate or check the state-preparation target-state JSON file.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the JSON target-state file to write or check.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Return a non-zero exit code if the output file is missing or stale.",
    )
    args = parser.parse_args(argv)

    payload = generate_target_data()
    encoded_payload = json.dumps(payload, indent=2)
    expected_text = f"{encoded_payload}\n"

    if args.check:
        try:
            stored_payload = json.loads(args.output.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return 1
        return 0 if target_data_matches(stored_payload, payload) else 1

    args.output.write_text(expected_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
