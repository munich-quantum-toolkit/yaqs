# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""YAQS MPS dense-vector vs Qiskit reference convention helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector

from mqt.yaqs.core.data_structures.mps import MPS

from scripts.benchmark_utils import (
    _expectation_via_mps_swaps,
    _fid_err_vec,
)

ReferenceConvention = Literal["direct", "bit_reversed"]
ObservableCategory = Literal["z_single", "x_single", "zz_nn", "zz_lr", "other"]


def bit_reverse_index(i: int, n: int) -> int:
    """Reverse little-endian qubit order for an amplitude index."""
    return int(f"{i:0{n}b}"[::-1], 2)


def bit_reverse_vec(vec: np.ndarray, n: int) -> np.ndarray:
    """Permute statevector amplitudes under qubit-index reversal."""
    dim = 1 << n
    out = np.zeros(dim, dtype=np.complex128)
    for i in range(dim):
        out[bit_reverse_index(i, n)] = vec[i]
    return out


def align_reference_vec(ref: np.ndarray, n: int, convention: ReferenceConvention) -> np.ndarray:
    if convention == "bit_reversed":
        return bit_reverse_vec(ref, n)
    return ref


def fidelity_between(a: np.ndarray, b: np.ndarray) -> float:
    return float(max(0.0, 1.0 - _fid_err_vec(a, b)))


def compare_statevectors(
    mps_vec: np.ndarray,
    ref_vec: np.ndarray,
    *,
    n: int,
) -> tuple[float, float, ReferenceConvention]:
    """Return (fidelity_direct, fidelity_bit_reversed, chosen_convention)."""
    f_direct = fidelity_between(ref_vec, mps_vec)
    f_rev = fidelity_between(bit_reverse_vec(ref_vec, n), mps_vec)
    if f_rev > f_direct + 1e-12:
        return f_direct, f_rev, "bit_reversed"
    return f_direct, f_rev, "direct"


def statevector_expectation(
    vec: np.ndarray,
    n: int,
    *,
    label: str,
    sites: list[int],
) -> float:
    """Expectation value using Qiskit Pauli string on *vec* with site labels = MPS indices."""
    pauli = ["I"] * n
    for p, s in zip(label, sites, strict=True):
        pauli[n - 1 - s] = p
    val = Statevector(vec).expectation_value(Pauli("".join(pauli)))
    return float(np.real(complex(val)))


def mps_expectation_via_dense(mps: MPS, n: int, *, label: str, sites: list[int]) -> float:
    """Exact expectation from ``mps.to_vec()`` using the Qiskit Pauli convention."""
    vec = np.asarray(mps.to_vec(), dtype=np.complex128)
    return statevector_expectation(vec, n, label=label, sites=sites)


def observable_errors_against_reference(
    mps: MPS,
    ref_vec: np.ndarray,
    observables: list[tuple[str, str, list[int]]],
    *,
    n: int,
    convention: ReferenceConvention,
) -> tuple[float, float, float]:
    ref = align_reference_vec(ref_vec, n, convention)
    errs: list[float] = []
    for _name, label, sites in observables:
        got = mps_expectation_via_dense(mps, n, label=label, sites=sites)
        ref_val = statevector_expectation(ref, n, label=label, sites=sites)
        errs.append(abs(got - ref_val))
    if not errs:
        return 0.0, 0.0, 0.0
    arr = np.array(errs, dtype=np.float64)
    return float(np.mean(arr)), float(np.max(arr)), float(np.sqrt(np.mean(arr**2)))


def _observable_category(name: str) -> ObservableCategory:
    if name.startswith("Z("):
        return "z_single"
    if name.startswith("X("):
        return "x_single"
    if name.startswith("ZZ_nn"):
        return "zz_nn"
    if name.startswith("ZZ_lr"):
        return "zz_lr"
    return "other"


def observable_errors_grouped(
    mps: MPS,
    ref_vec: np.ndarray,
    observables: list[tuple[str, str, list[int]]],
    *,
    n: int,
    convention: ReferenceConvention,
) -> dict[ObservableCategory, tuple[float, float, float]]:
    """Per-category mean/max/RMS absolute observable error vs *ref_vec*."""
    ref = align_reference_vec(ref_vec, n, convention)
    grouped: dict[ObservableCategory, list[float]] = defaultdict(list)
    for name, label, sites in observables:
        got = mps_expectation_via_dense(mps, n, label=label, sites=sites)
        ref_val = statevector_expectation(ref, n, label=label, sites=sites)
        grouped[_observable_category(name)].append(abs(got - ref_val))

    out: dict[ObservableCategory, tuple[float, float, float]] = {}
    for cat in ("z_single", "x_single", "zz_nn", "zz_lr", "other"):
        errs = grouped.get(cat, [])
        if not errs:
            out[cat] = (0.0, 0.0, 0.0)
            continue
        arr = np.array(errs, dtype=np.float64)
        out[cat] = float(np.mean(arr)), float(np.max(arr)), float(np.sqrt(np.mean(arr**2)))
    return out


def check_high_fidelity_observable_consistency(
    fidelity: float | None,
    max_obs_error: float | None,
    *,
    fid_tol: float = 1e-10,
    obs_tol: float = 1e-5,
) -> str:
    """Return a warning message when fidelity and aggregate observable error disagree."""
    if fidelity is None or max_obs_error is None:
        return ""
    if fidelity > 1.0 - fid_tol and max_obs_error > obs_tol:
        return "High fidelity but nonzero observable error: check observable convention."
    return ""


def observable_errors_both_conventions(
    mps: MPS,
    ref_vec: np.ndarray,
    observables: list[tuple[str, str, list[int]]],
    *,
    n: int,
) -> tuple[float, float, float, float, ReferenceConvention]:
    mean_d, max_d, rms_d = observable_errors_against_reference(
        mps, ref_vec, observables, n=n, convention="direct"
    )
    mean_r, max_r, rms_r = observable_errors_against_reference(
        mps, ref_vec, observables, n=n, convention="bit_reversed"
    )
    if max_r + 1e-15 < max_d:
        return mean_d, max_d, mean_r, max_r, "direct"
    if max_d + 1e-15 < max_r:
        return mean_d, max_d, mean_r, max_r, "bit_reversed"
    return mean_d, max_d, mean_r, max_r, "direct"


def qiskit_reference_vec(qc: QuantumCircuit) -> np.ndarray:
    return np.asarray(Statevector(qc).data, dtype=np.complex128)
