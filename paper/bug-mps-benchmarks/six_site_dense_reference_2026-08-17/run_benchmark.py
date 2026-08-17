#!/usr/bin/env python3
"""Reproduce the manuscript's uncompressed six-site BUG refinement table.

The dense Hamiltonian is assembled independently in the site-0-LSB ordering
used by :meth:`MPS.to_vec`.  The three columns exercise one center-augmented
endpoint sweep, two alternating center-augmented endpoint sweeps, and two
alternating sweeps with explicit previous-basis retention, respectively.
No variant calls MPS compression or normalization.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np

from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.methods.bug import bug_sweep, prepare_canonical_site_tensors
from mqt.yaqs.core.methods.decompositions import left_qr
from mqt.yaqs.core.methods.tdvp.primitives import update_right_environment, update_site


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]

LENGTH = 6
TOTAL_TIME = 0.4
KRYLOV_TOL = 1e-12
INITIAL_BASIS_STRING = "010011"
DT_GRID = (0.1, 0.05, 0.025, 0.0125, 0.00625)

JX = (0.37, 0.51, 0.29, 0.63, 0.43)
JY = (0.31, 0.47, 0.39, 0.55, 0.35)
JZ = (0.61, 0.33, 0.49, 0.41, 0.57)
HX = (0.23, -0.17, 0.31, 0.11, -0.29, 0.19)
HY = (0.07, -0.11, 0.13, -0.05, 0.09, -0.03)
HZ = (-0.07, 0.13, 0.05, -0.19, 0.17, 0.27)

VARIANT_ONE = "one_sweep_center"
VARIANT_CENTER = "two_sweeps_center"
VARIANT_PREVIOUS = "two_sweeps_previous_basis"
VARIANTS = (VARIANT_ONE, VARIANT_CENTER, VARIANT_PREVIOUS)

Sweep = Callable[..., None]


def parse_csv_floats(value: str) -> tuple[float, ...]:
    """Parse a comma-separated list of positive floating-point values."""
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        msg = "At least one positive timestep is required."
        raise argparse.ArgumentTypeError(msg)
    return values


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dts", type=parse_csv_floats, default=DT_GRID)
    parser.add_argument("--total-time", type=float, default=TOTAL_TIME)
    parser.add_argument("--output", type=Path, default=HERE / "raw_results.json")
    return parser.parse_args()


def pauli_terms() -> list[tuple[float, str]]:
    """Return the asymmetric fixture as indexed Pauli strings."""
    terms: list[tuple[float, str]] = []
    for label, coefficients in (("X", JX), ("Y", JY), ("Z", JZ)):
        terms.extend((coefficient, f"{label}{site} {label}{site + 1}") for site, coefficient in enumerate(coefficients))
    for label, coefficients in (("X", HX), ("Y", HY), ("Z", HZ)):
        terms.extend((coefficient, f"{label}{site}") for site, coefficient in enumerate(coefficients))
    return terms


def fixture_mpo() -> MPO:
    """Build the exact finite-state-machine MPO without an SVD compression sweep."""
    mpo = MPO()
    mpo.from_pauli_sum(terms=pauli_terms(), length=LENGTH, n_sweeps=0)
    return mpo


def _kron_term(local_operators: dict[int, np.ndarray]) -> np.ndarray:
    """Construct one dense Pauli term in ``O_5 kron ... kron O_0`` order."""
    identity = np.eye(2, dtype=np.complex128)
    matrix = np.ones((1, 1), dtype=np.complex128)
    for site in reversed(range(LENGTH)):
        matrix = np.kron(matrix, local_operators.get(site, identity))
    return np.asarray(matrix, dtype=np.complex128)


def independent_dense_hamiltonian(*, reflected: bool = False) -> np.ndarray:
    """Assemble the fixture independently of the MPO implementation.

    When ``reflected`` is true, each physical site ``i`` is replaced by
    ``LENGTH - 1 - i`` while retaining the independent Kronecker assembly.
    """
    paulis = {
        "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        "Z": np.diag([1, -1]).astype(np.complex128),
    }
    hamiltonian = np.zeros((2**LENGTH, 2**LENGTH), dtype=np.complex128)
    reflect_site = (lambda site: LENGTH - 1 - site) if reflected else (lambda site: site)

    for label, coefficients in (("X", JX), ("Y", JY), ("Z", JZ)):
        for site, coefficient in enumerate(coefficients):
            local = {
                reflect_site(site): paulis[label],
                reflect_site(site + 1): paulis[label],
            }
            hamiltonian += coefficient * _kron_term(local)
    for label, coefficients in (("X", HX), ("Y", HY), ("Z", HZ)):
        for site, coefficient in enumerate(coefficients):
            hamiltonian += coefficient * _kron_term({reflect_site(site): paulis[label]})
    return hamiltonian


def initial_state() -> MPS:
    """Return the product state whose string is indexed from site 0 upward."""
    return MPS(LENGTH, state="basis", basis_string=INITIAL_BASIS_STRING)


def dense_reference(hamiltonian: np.ndarray, initial_vector: np.ndarray, final_time: float) -> np.ndarray:
    """Propagate with an independently diagonalized dense Hamiltonian."""
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    coefficients = eigenvectors.conj().T @ initial_vector
    return np.asarray(eigenvectors @ (np.exp(-1j * final_time * eigenvalues) * coefficients), dtype=np.complex128)


def phase_aligned_state_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return the normalized state distance minimized over global phase."""
    denominator = float(np.linalg.norm(reference) * np.linalg.norm(candidate))
    if denominator == 0:
        return math.inf
    overlap = float(abs(np.vdot(reference, candidate)) / denominator)
    return math.sqrt(max(0.0, 2.0 - 2.0 * min(overlap, 1.0)))


def explicit_previous_basis_sweep(
    state: MPS,
    mpo: MPO,
    *,
    dt: float,
    krylov_tol: float,
) -> None:
    """Apply one uncompressed endpoint sweep retaining the previous block basis.

    This differs from :func:`bug_sweep` only at internal trial-basis stacks.  It
    retains the transported previous block basis ``old_basis_current`` instead
    of the coefficient-bearing working center.  The root update, environments,
    QR convention, and Krylov routine are shared with the production kernel.
    """
    if mpo.length != state.length:
        msg = "MPS and Hamiltonian must have the same number of sites"
        raise ValueError(msg)
    state.assert_center(0, context="explicit_previous_basis_sweep")

    canonical_centers, left_environments = prepare_canonical_site_tensors(state, mpo)
    right_dimension = state.tensors[-1].shape[2]
    right_environment = np.eye(right_dimension, dtype=np.complex128).reshape(right_dimension, 1, right_dimension)
    deeper_overlap = np.eye(right_dimension, dtype=np.complex128)

    for site in range(state.length - 1, 0, -1):
        working_center = canonical_centers[site]
        predictor = update_site(
            left_environments[site],
            right_environment,
            mpo.tensors[site],
            working_center,
            dt,
            krylov_tol=krylov_tol,
        )
        old_basis_current = np.asarray(
            np.tensordot(state.tensors[site], deeper_overlap, axes=(2, 0)),
            dtype=np.complex128,
        )
        updated_basis, _ = left_qr(np.concatenate((old_basis_current, predictor), axis=1))
        deeper_overlap = np.asarray(
            np.tensordot(old_basis_current, updated_basis.conj(), axes=([0, 2], [0, 2])),
            dtype=np.complex128,
        )
        state.tensors[site] = updated_basis
        canonical_centers[site - 1] = np.asarray(
            np.tensordot(canonical_centers[site - 1], deeper_overlap, axes=(2, 0)),
            dtype=np.complex128,
        )
        right_environment = update_right_environment(
            updated_basis,
            updated_basis,
            mpo.tensors[site],
            right_environment,
        )

    state.tensors[0] = update_site(
        left_environments[0],
        right_environment,
        mpo.tensors[0],
        canonical_centers[0],
        dt,
        krylov_tol=krylov_tol,
    )
    state.set_center(0)


def alternating_endpoint_step(state: MPS, mpo: MPO, *, dt: float, sweep: Sweep) -> None:
    """Apply two uncompressed endpoint sweeps of duration ``dt / 2``."""
    state.assert_center(0, context="alternating endpoint step entry")
    sweep(state, mpo, dt=dt / 2, krylov_tol=KRYLOV_TOL)

    # Moving the center is a gauge-only QR operation.  Reflection then maps the
    # right endpoint to index zero, satisfying the second sweep's entry contract.
    state.shift_center_to(state.length - 1, decomposition="QR")
    state.flip_network()
    state.assert_center(0, context="reflected alternating endpoint step entry")
    sweep(state, mpo.reflected(), dt=dt / 2, krylov_tol=KRYLOV_TOL)
    state.flip_network()
    state.assert_center(state.length - 1, context="alternating endpoint step exit")


def evolve_variant(initial: MPS, mpo: MPO, *, dt: float, final_time: float, variant: str) -> MPS:
    """Evolve a clone of ``initial`` with one table-column schedule."""
    steps_float = final_time / dt
    steps = int(round(steps_float))
    if not math.isclose(steps_float, steps, rel_tol=0.0, abs_tol=1e-12):
        msg = f"final_time={final_time} is not an integer multiple of dt={dt}"
        raise ValueError(msg)
    if variant not in VARIANTS:
        msg = f"Unknown variant: {variant}"
        raise ValueError(msg)

    state = deepcopy(initial)
    for _ in range(steps):
        if state.orthogonality_center != 0:
            state.shift_center_to(0, decomposition="QR")
        if variant == VARIANT_ONE:
            bug_sweep(state, mpo, dt=dt, krylov_tol=KRYLOV_TOL)
        else:
            sweep = bug_sweep if variant == VARIANT_CENTER else explicit_previous_basis_sweep
            alternating_endpoint_step(state, mpo, dt=dt, sweep=sweep)
    return state


def bond_profile(state: MPS) -> list[int]:
    """Return all virtual bond dimensions, including the two boundaries."""
    return [state.tensors[0].shape[1], *(tensor.shape[2] for tensor in state.tensors)]


def current_git_commit() -> str:
    """Return the enclosing Git commit, or ``UNKNOWN`` outside a worktree."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def run_benchmark(*, dts: tuple[float, ...], final_time: float) -> dict[str, Any]:
    """Run all requested timesteps and return the JSON-serializable payload."""
    mpo = fixture_mpo()
    dense_hamiltonian = independent_dense_hamiltonian()
    reflected_dense = independent_dense_hamiltonian(reflected=True)
    fixture_initial = initial_state()
    initial_vector = fixture_initial.to_vec().copy()
    initial_tensors = [tensor.copy() for tensor in fixture_initial.tensors]
    reference = dense_reference(dense_hamiltonian, initial_vector, final_time)

    prepared_initial = deepcopy(fixture_initial)
    prepared_snapshot = [tensor.copy() for tensor in prepared_initial.tensors]
    prepare_canonical_site_tensors(prepared_initial, mpo)
    preparation_preserves_input = all(
        np.array_equal(before, after)
        for before, after in zip(prepared_snapshot, prepared_initial.tensors, strict=True)
    )

    dense_norm = float(np.linalg.norm(dense_hamiltonian))
    reflected_norm = float(np.linalg.norm(reflected_dense))
    structural = {
        "mpo_dense_relative_frobenius_error": float(
            np.linalg.norm(mpo.to_matrix_mps_order() - dense_hamiltonian) / dense_norm
        ),
        "reflected_mpo_dense_relative_frobenius_error": float(
            np.linalg.norm(mpo.reflected().to_matrix_mps_order() - reflected_dense) / reflected_norm
        ),
        "site_ordering_gap": float(np.linalg.norm(mpo.to_matrix() - dense_hamiltonian) / dense_norm),
        "reflection_asymmetry_residual": float(np.linalg.norm(reflected_dense - dense_hamiltonian) / dense_norm),
        "dense_reference_norm_error": float(abs(np.vdot(reference, reference).real - 1.0)),
        "initial_state_nonzero_index": int(np.flatnonzero(np.abs(initial_vector) > 0.5)[0]),
        "expected_initial_state_nonzero_index": int(INITIAL_BASIS_STRING[::-1], 2),
        "preparation_preserves_input_tensors": preparation_preserves_input,
    }

    runs: dict[str, Any] = {}
    all_finite = True
    all_endpoints_restored = True
    for dt in dts:
        variant_vectors: dict[str, np.ndarray] = {}
        variant_results: dict[str, Any] = {}
        for variant in VARIANTS:
            state = evolve_variant(fixture_initial, mpo, dt=dt, final_time=final_time, variant=variant)
            vector = state.to_vec().copy()
            variant_vectors[variant] = vector
            expected_center = 0 if variant == VARIANT_ONE else LENGTH - 1
            finite = bool(np.all(np.isfinite(vector)))
            endpoint_restored = state.orthogonality_center == expected_center
            all_finite &= finite
            all_endpoints_restored &= endpoint_restored
            profile = bond_profile(state)
            variant_results[variant] = {
                "phase_aligned_state_error": phase_aligned_state_error(reference, vector),
                "norm": float(np.linalg.norm(vector)),
                "orthogonality_center": state.orthogonality_center,
                "expected_orthogonality_center": expected_center,
                "endpoint_restored": endpoint_restored,
                "all_values_finite": finite,
                "bond_profile": profile,
                "maximum_bond_dimension": max(profile),
            }
        runs[format(dt, ".8g")] = {
            "steps": int(round(final_time / dt)),
            "variants": variant_results,
            "two_sweep_variant_phase_aligned_difference": phase_aligned_state_error(
                variant_vectors[VARIANT_CENTER],
                variant_vectors[VARIANT_PREVIOUS],
            ),
        }

    structural["all_results_finite"] = all_finite
    structural["all_endpoints_restored"] = all_endpoints_restored
    structural["initial_input_tensors_preserved"] = all(
        np.array_equal(before, after)
        for before, after in zip(initial_tensors, fixture_initial.tensors, strict=True)
    )

    return {
        "schema_version": 1,
        "git_commit": current_git_commit(),
        "protocol": {
            "length": LENGTH,
            "total_time": final_time,
            "timesteps": list(dts),
            "initial_basis_string_site_0_first": INITIAL_BASIS_STRING,
            "site_0_vector_order": "least_significant_bit",
            "dense_kronecker_order": "O_5 tensor ... tensor O_0",
            "krylov_tolerance": KRYLOV_TOL,
            "compression": False,
            "normalization": False,
            "bond_cap": None,
            "one_sweep": "Phi_h,c^(R->L)",
            "two_sweeps_center": "Phi_h/2,c^(L->R) after Phi_h/2,c^(R->L)",
            "two_sweeps_previous_basis": "same composition with explicit previous-basis retention",
            "coefficients": {
                "Jx": JX,
                "Jy": JY,
                "Jz": JZ,
                "hx": HX,
                "hy": HY,
                "hz": HZ,
            },
        },
        "structural_checks": structural,
        "runs": runs,
    }


def print_table(payload: dict[str, Any]) -> None:
    """Print the refinement table in a compact Markdown form."""
    print("| h | One sweep, center | Two sweeps, center | Two sweeps, previous basis |")
    print("|---:|---:|---:|---:|")
    for dt, run in payload["runs"].items():
        variants = run["variants"]
        print(
            f"| {float(dt):.5f} "
            f"| {variants[VARIANT_ONE]['phase_aligned_state_error']:.6e} "
            f"| {variants[VARIANT_CENTER]['phase_aligned_state_error']:.6e} "
            f"| {variants[VARIANT_PREVIOUS]['phase_aligned_state_error']:.6e} |"
        )


def main() -> None:
    """Run the benchmark, save all data, and print its table."""
    args = parse_args()
    payload = run_benchmark(dts=args.dts, final_time=args.total_time)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_table(payload)
    print(f"\nSaved complete results to {args.output}")


if __name__ == "__main__":
    main()
