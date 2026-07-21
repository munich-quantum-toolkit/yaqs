# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Gate execution helpers for the main-text single RZZ benchmark."""

from __future__ import annotations

import copy
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from mqt.yaqs.core import linalg
from mqt.yaqs.core.data_structures.mpo_utils import resolve_lr_tensor
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
from mqt.yaqs.digital.digital_tjm import apply_two_qubit_gate

if TYPE_CHECKING:
    from collections.abc import Iterator

L_DEFAULT = 12
TARGET_BOND_PROFILE = [1, 2, 4, 8, 8, 8, 8, 8, 8, 8, 4, 2, 1]
SVD_THRESHOLD = 1e-13
KRYLOV_TOL = 1e-12
TRUNC_MODE = "discarded_weight"
TDVP_MODE = "2site"
PRESET = "exact"


@dataclass
class DiscardedWeightTracker:
    """Accumulate SVD discarded weight during gate application."""

    per_gate: float = 0.0
    cumulative: float = 0.0
    events: list[float] = field(default_factory=list)

    def reset_gate(self) -> None:
        self.per_gate = 0.0

    def record(self, s_vec: np.ndarray, keep: int) -> None:
        total = float(np.sum(np.square(s_vec)))
        if total <= 0.0:
            return
        discarded = float(np.sum(np.square(s_vec[keep:])) / total)
        self.per_gate += discarded
        self.cumulative += discarded
        self.events.append(discarded)


@contextmanager
def track_discarded_weight(tracker: DiscardedWeightTracker) -> Iterator[None]:
    """Instrument ``linalg.truncate`` to record discarded weight."""
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


def _params(chi: int, *, gate_mode: str = "tdvp", tdvp_sweeps: int = 1) -> StrongSimParams:
    return StrongSimParams(
        observables=[],
        preset=PRESET,
        gate_mode=gate_mode,  # type: ignore[arg-type]
        svd_threshold=SVD_THRESHOLD,
        max_bond_dim=chi,
        krylov_tol=KRYLOV_TOL,
        tdvp_sweeps=tdvp_sweeps,
        tdvp_mode=TDVP_MODE,
        trunc_mode=TRUNC_MODE,
        get_state=False,
    )


def make_gate(gate_type: str, theta: float, q0: int, q1: int):
    """Build a YAQS gate object with sites set."""
    from mqt.yaqs.core.libraries.gate_library import GateLibrary

    factory = getattr(GateLibrary, gate_type)
    gate = factory([theta])
    gate.set_sites(q0, q1)
    return gate


def make_dag_node(gate_type: str, theta: float, q0: int, q1: int, length: int):
    """Create a DAG op node for a single two-qubit gate."""
    qc = QuantumCircuit(length)
    getattr(qc, gate_type)(theta, q0, q1)
    return next(iter(circuit_to_dag(qc).topological_op_nodes()))


def random_mps(length: int, bond_profile: list[int], rng: np.random.Generator) -> MPS:
    """Construct a generic complex MPS with the requested bond profile."""
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


def bond_profile(mps: MPS) -> list[int]:
    """Full bond-dimension profile ``[chi_0, ..., chi_L]``."""
    profile = [int(mps.tensors[0].shape[1])]
    profile.extend(int(tensor.shape[2]) for tensor in mps.tensors)
    return profile


def fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized state fidelity ``|⟨a|b⟩|² / (‖a‖² ‖b‖²)``."""
    return normalized_state_fidelity(a, b)["fidelity_normalized"]


def normalized_state_fidelity(
    exact: np.ndarray,
    approx: np.ndarray,
    *,
    clip_tol: float = 1e-12,
) -> dict[str, float]:
    """Compute normalized state fidelity without modifying either vector.

    Uses
    ``F = |⟨e|a⟩|² / (⟨e|e⟩ ⟨a|a⟩)`` and ``I = 1 - F``.
    Values outside ``[0, 1]`` by more than ``clip_tol`` raise ``ValueError``;
    smaller floating-point excursions are clipped into range.

    Args:
        exact: Reference statevector.
        approx: Approximate statevector (may be unnormalized).
        clip_tol: Allowed floating-point excursion before raising.

    Returns:
        Provenance dictionary with raw overlap, norms, normalized fidelity,
        infidelity, and L2 norm loss ``1 - ‖a‖/‖e‖``.
    """
    e = np.asarray(exact, dtype=np.complex128).reshape(-1)
    a = np.asarray(approx, dtype=np.complex128).reshape(-1)
    overlap_squared_raw = float(abs(np.vdot(e, a)) ** 2)
    norm_squared_exact = float(np.real(np.vdot(e, e)))
    norm_squared_approx = float(np.real(np.vdot(a, a)))
    if norm_squared_exact <= 0.0 or norm_squared_approx <= 0.0:
        msg = (
            "normalized_state_fidelity requires nonzero norms; "
            f"got ‖e‖²={norm_squared_exact}, ‖a‖²={norm_squared_approx}"
        )
        raise ValueError(msg)
    fidelity_normalized = overlap_squared_raw / (norm_squared_exact * norm_squared_approx)
    if fidelity_normalized < -clip_tol or fidelity_normalized > 1.0 + clip_tol:
        msg = f"Fidelity {fidelity_normalized} outside [0, 1] by more than {clip_tol}"
        raise ValueError(msg)
    fidelity_normalized = float(min(1.0, max(0.0, fidelity_normalized)))
    infidelity_normalized = 1.0 - fidelity_normalized
    norm_exact = float(np.sqrt(norm_squared_exact))
    norm_approx = float(np.sqrt(norm_squared_approx))
    norm_loss = 1.0 - (norm_approx / norm_exact)
    return {
        "overlap_squared_raw": overlap_squared_raw,
        "norm_squared_exact": norm_squared_exact,
        "norm_squared_approx": norm_squared_approx,
        "fidelity_normalized": fidelity_normalized,
        "infidelity_normalized": infidelity_normalized,
        "norm_loss": float(norm_loss),
        "norm_exact": norm_exact,
        "norm_approx": norm_approx,
    }


def phase_align(reference: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Align global phase of ``state`` to ``reference``."""
    phase = np.vdot(state, reference)
    if abs(phase) > 0.0:
        return state * (phase / abs(phase))
    return state


def apply_gate_to_dense_state(
    vec: np.ndarray,
    gate_matrix: np.ndarray,
    q0: int,
    q1: int,
    num_qubits: int,
) -> np.ndarray:
    """Apply a 4x4 two-qubit gate matrix to a dense statevector (LSB site ordering)."""
    left, right = min(q0, q1), max(q0, q1)
    u4 = np.asarray(gate_matrix, dtype=np.complex128).reshape(4, 4)
    psi = vec.reshape([2] * num_qubits)
    psi = np.transpose(psi, list(reversed(range(num_qubits))))
    tmp = np.tensordot(u4.reshape(2, 2, 2, 2), psi, axes=([2, 3], [left, right]))
    remaining = [i for i in range(num_qubits) if i not in {left, right}]
    dest = [0] * num_qubits
    dest[left] = 0
    dest[right] = 1
    for k, site in enumerate(remaining):
        dest[site] = 2 + k
    out = np.transpose(tmp, dest)
    out = np.transpose(out, list(reversed(range(num_qubits))))
    return out.reshape(-1)


def apply_two_qubit_dense(vec: np.ndarray, length: int, q0: int, q1: int, gate) -> np.ndarray:
    """Apply a two-qubit gate matrix from YAQS to a dense statevector."""
    left, right = min(q0, q1), max(q0, q1)
    u = resolve_lr_tensor(gate, left, right)
    return apply_gate_to_dense_state(vec, u.reshape(4, 4), q0, q1, length)


def gate_matrix(gate_type: str, theta: float) -> np.ndarray:
    """Return the dense 4x4 gate matrix from the YAQS implementation."""
    gate = make_gate(gate_type, theta, 0, 1)
    return np.asarray(gate.matrix, dtype=np.complex128)


def substep_unit_check(gate_type: str, theta: float, substeps: int) -> float:
    """Max entrywise error between full gate and ``s`` subgates at ``theta/s``."""
    full = gate_matrix(gate_type, theta)
    sub = gate_matrix(gate_type, theta / substeps)
    composed = np.linalg.matrix_power(sub, substeps)
    return float(np.max(np.abs(composed - full)))


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


def apply_method(
    initial_mps: MPS,
    node,
    *,
    method: str,
    chi: int,
    substeps: int,
    tracker: DiscardedWeightTracker | None = None,
) -> tuple[MPS, float, float]:
    """Apply one gate with the requested method."""
    gate_modes = {
        "hybrid_tdvp": "tdvp",
        "full_tdvp": "full-tdvp",
        "tebd_swap": "swaps",
        "mpo_zipup": "mpo",
    }
    if method not in gate_modes:
        msg = f"Unknown method {method!r}"
        raise ValueError(msg)
    params = _params(chi, gate_mode=gate_modes[method], tdvp_sweeps=substeps)
    state = copy.deepcopy(initial_mps)
    if tracker is not None:
        tracker.reset_gate()
    ctx = track_discarded_weight(tracker) if tracker is not None else _NullContext()
    with ctx:
        t0 = time.perf_counter()
        apply_two_qubit_gate(state, node, params)
        runtime = time.perf_counter() - t0
    discarded = tracker.per_gate if tracker is not None else float("nan")
    return state, runtime, discarded


def param_count_from_profile(profile: list[int], length: int) -> int:
    """Parameter count from a bond profile and qubit count."""
    total = 0
    for site in range(length):
        total += 2 * profile[site] * profile[site + 1]
    return total


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_initial_state(seed: int) -> dict[str, Any]:
    """Build the fixed random MPS initial state for the benchmark."""
    rng = np.random.default_rng(seed)
    mps = random_mps(L_DEFAULT, list(TARGET_BOND_PROFILE), rng)
    vec = mps.to_vec()
    diff = float(np.max(np.abs(vec - mps.to_vec())))
    if diff >= 1e-12:
        msg = f"Initial MPS/dense mismatch for seed {seed}: {diff:.3e}"
        raise RuntimeError(msg)
    return {
        "mps": mps,
        "vec": vec.astype(np.complex128, copy=False),
        "bond_profile": bond_profile(mps),
    }


class LockError(RuntimeError):
    """Raised when the output directory is locked."""


class DirectoryLock:
    """Prevent concurrent writers to the same output directory."""

    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / ".benchmark.lock"
        self.acquired = False

    def acquire(self) -> None:
        if self.path.exists():
            payload = _read_lock_file(self.path)
            msg = (
                f"Output directory is locked: {self.path}\n"
                f"  pid={payload.get('pid')}, started={payload.get('started')}\n"
                "Another benchmark process may be running. "
                "If the lock is stale, delete the lock file manually."
            )
            raise LockError(msg)
        payload = {"pid": os.getpid(), "started": datetime.now(UTC).isoformat()}
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self.acquired = True

    def release(self) -> None:
        if self.acquired and self.path.exists():
            self.path.unlink(missing_ok=True)
        self.acquired = False


class RunLogger:
    """Line-buffered run.log writer."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file: TextIO = Path(path).open("a", encoding="utf-8", buffering=1)

    def log(self, message: str) -> None:
        stamp = datetime.now(UTC).isoformat()
        self._file.write(f"[{stamp}] {message}\n")
        self._file.flush()

    def close(self) -> None:
        self._file.flush()
        self._file.close()


def _read_lock_file(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
