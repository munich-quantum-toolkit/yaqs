# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Minimal YAQS production-code helpers for the structural checks.

The helpers construct the gate inputs and diagnostics directly with the
current ``DigitalSimParams`` API, keeping this validation campaign independent
of the other manuscript benchmarks.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from mqt.yaqs.core import linalg
from mqt.yaqs.core.data_structures.mpo_utils import resolve_lr_tensor
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import DigitalSimParams
from mqt.yaqs.digital.digital_tjm import apply_two_qubit_gate

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class DiscardedWeightTracker:
    """Accumulate SVD discarded weight during gate application."""

    per_gate: float = 0.0
    events: list[float] = field(default_factory=list)

    def reset_gate(self) -> None:
        self.per_gate = 0.0

    def record(self, s_vec: np.ndarray, keep: int) -> None:
        total = float(np.sum(np.square(s_vec)))
        if total <= 0.0:
            return
        discarded = float(np.sum(np.square(s_vec[keep:])) / total)
        self.per_gate += discarded
        self.events.append(discarded)


@contextmanager
def track_discarded_weight(tracker: DiscardedWeightTracker) -> Iterator[None]:
    original = linalg.truncate

    def wrapped(
        s_vec: np.ndarray,
        *,
        mode: str,
        threshold: float,
        max_bond_dim: int | None = None,
        min_keep: int = 1,
    ) -> int:
        keep = original(
            s_vec,
            mode=mode,
            threshold=threshold,
            max_bond_dim=max_bond_dim,
            min_keep=min_keep,
        )
        tracker.record(np.asarray(s_vec), keep)
        return keep

    linalg.truncate = wrapped  # type: ignore[assignment]
    try:
        yield
    finally:
        linalg.truncate = original


def make_gate(gate_type: str, theta: float, q0: int, q1: int):
    from mqt.yaqs.core.libraries.gate_library import GateLibrary

    factory = getattr(GateLibrary, gate_type)
    gate = factory([theta])
    gate.set_sites(q0, q1)
    return gate


def make_dag_node(gate_type: str, theta: float, q0: int, q1: int, length: int):
    qc = QuantumCircuit(length)
    getattr(qc, gate_type)(theta, q0, q1)
    return next(iter(circuit_to_dag(qc).topological_op_nodes()))


def digital_params(chi: int, *, gate_mode: str, tdvp_sweeps: int) -> DigitalSimParams:
    return DigitalSimParams(
        observables=[],
        preset="exact",
        gate_mode=gate_mode,  # type: ignore[arg-type]
        svd_threshold=1e-13,
        max_bond_dim=chi,
        krylov_tol=1e-12,
        tdvp_sweeps=tdvp_sweeps,
        tdvp_mode="2site",
        trunc_mode="discarded_weight",
        get_state=True,
    )


def apply_full_tdvp(
    initial_mps: MPS,
    node: Any,
    *,
    chi: int,
    substeps: int,
    tracker: DiscardedWeightTracker | None = None,
) -> tuple[MPS, float]:
    """Apply one gate with ``gate_mode='full-tdvp'``."""
    params = digital_params(chi, gate_mode="full-tdvp", tdvp_sweeps=substeps)
    state = copy.deepcopy(initial_mps)
    if tracker is not None:
        tracker.reset_gate()
        with track_discarded_weight(tracker):
            apply_two_qubit_gate(state, node, params)
        return state, tracker.per_gate
    apply_two_qubit_gate(state, node, params)
    return state, float("nan")


def random_mps(length: int, bond_profile: list[int], rng: np.random.Generator) -> MPS:
    tensors = []
    for site in range(length):
        chi_l = bond_profile[site]
        chi_r = bond_profile[site + 1]
        real = rng.standard_normal((2, chi_l, chi_r))
        imag = rng.standard_normal((2, chi_l, chi_r))
        tensors.append((real + 1j * imag).astype(np.complex128))
    mps = MPS(length, tensors=tensors)
    mps.set_canonical_form(length // 2, decomposition="SVD")
    mps.normalize(form="B", decomposition="SVD")
    return mps


def mps_bond_profile(mps: MPS) -> list[int]:
    profile = [int(mps.tensors[0].shape[1])]
    profile.extend(int(tensor.shape[2]) for tensor in mps.tensors)
    return profile


def phase_align(reference: np.ndarray, state: np.ndarray) -> np.ndarray:
    phase = np.vdot(state, reference)
    if abs(phase) > 0.0:
        return state * (phase / abs(phase))
    return state


def normalized_infidelity(exact: np.ndarray, approx: np.ndarray) -> float:
    e = np.asarray(exact, dtype=np.complex128).reshape(-1)
    a = np.asarray(approx, dtype=np.complex128).reshape(-1)
    overlap = float(abs(np.vdot(e, a)) ** 2)
    ne = float(np.real(np.vdot(e, e)))
    na = float(np.real(np.vdot(a, a)))
    return 1.0 - overlap / (ne * na)


def apply_gate_dense_yaqs(vec: np.ndarray, length: int, q0: int, q1: int, gate) -> np.ndarray:
    """Apply a YAQS two-qubit gate matrix to a YAQS ``to_vec()`` statevector (LSB)."""
    left, right = min(q0, q1), max(q0, q1)
    u = resolve_lr_tensor(gate, left, right)
    u4 = np.asarray(u, dtype=np.complex128).reshape(4, 4)
    psi = vec.reshape([2] * length)
    psi = np.transpose(psi, list(reversed(range(length))))
    tmp = np.tensordot(u4.reshape(2, 2, 2, 2), psi, axes=([2, 3], [left, right]))
    remaining = [i for i in range(length) if i not in {left, right}]
    dest = [0] * length
    dest[left] = 0
    dest[right] = 1
    for k, site in enumerate(remaining):
        dest[site] = 2 + k
    out = np.transpose(tmp, dest)
    out = np.transpose(out, list(reversed(range(length))))
    return out.reshape(-1)
