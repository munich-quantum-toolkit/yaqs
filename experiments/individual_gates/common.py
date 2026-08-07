# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Shared helpers for the self-contained individual-gates campaign."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from config import (
    BOND_PROFILE,
    GATE_LIBRARY_SPLIT_CUTOFF,
    KRYLOV_TOL,
    METHOD_TO_GATE_MODE,
    MIN_KEEP,
    POSITIVE_WEIGHT_EPS,
    REPO_ROOT,
    SVD_THRESHOLD,
    TDVP_MODE,
    TRUNC_MODE,
    N,
)
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Statevector

from mqt.yaqs.core import linalg
from mqt.yaqs.core.data_structures.mpo_utils import resolve_lr_tensor
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import DigitalSimParams
from mqt.yaqs.core.libraries.gate_library import GateLibrary, X, Y, Z
from mqt.yaqs.digital.digital_tjm import apply_two_qubit_gate, construct_generator_mpo

if TYPE_CHECKING:
    from collections.abc import Iterator

    from qiskit.dagcircuit import DAGOpNode

PAULI = {
    "x": np.asarray(X().matrix, dtype=np.complex128),
    "y": np.asarray(Y().matrix, dtype=np.complex128),
    "z": np.asarray(Z().matrix, dtype=np.complex128),
}
I2 = np.eye(2, dtype=np.complex128)


@dataclass
class DiscardedWeightTracker:
    """Accumulate SVD discarded weight and retention diagnostics."""

    per_gate: float = 0.0
    events: list[float] = field(default_factory=list)
    min_kept_singular: list[float] = field(default_factory=list)
    min_keep_args: list[int] = field(default_factory=list)
    keep_counts: list[int] = field(default_factory=list)
    singular_lists: list[list[float]] = field(default_factory=list)
    # True when discarded-weight threshold alone would have kept more than
    # max_bond_dim (i.e. the hard cap removed positive singular weight).
    cap_truncation_events: list[bool] = field(default_factory=list)

    def reset_gate(self) -> None:
        self.per_gate = 0.0
        self.events.clear()
        self.min_kept_singular.clear()
        self.min_keep_args.clear()
        self.keep_counts.clear()
        self.singular_lists.clear()
        self.cap_truncation_events.clear()

    def record(
        self,
        s_vec: np.ndarray,
        keep: int,
        *,
        min_keep: int,
        max_bond_dim: int | None,
        keep_without_cap: int | None = None,
    ) -> None:
        s = np.asarray(s_vec, dtype=np.float64).reshape(-1)
        total = float(np.sum(np.square(s)))
        if total > 0.0:
            discarded = float(np.sum(np.square(s[keep:])) / total)
            self.per_gate += discarded
            self.events.append(discarded)
        else:
            self.events.append(0.0)
        self.min_keep_args.append(int(min_keep))
        self.keep_counts.append(int(keep))
        self.singular_lists.append([float(v) for v in s])
        if keep > 0 and s.size > 0:
            self.min_kept_singular.append(float(s[keep - 1]))
        else:
            self.min_kept_singular.append(0.0)
        if max_bond_dim is None or keep_without_cap is None:
            self.cap_truncation_events.append(False)
        else:
            # Cap truncation: unconstrained keep exceeds the hard bond cap and
            # the discarded tail carries positive weight.
            tail = float(np.sum(np.square(s[keep:]))) if keep < s.size else 0.0
            self.cap_truncation_events.append(
                keep_without_cap > max_bond_dim and tail > POSITIVE_WEIGHT_EPS * max(total, 1.0)
            )

    @property
    def positive_weight_truncated(self) -> bool:
        return any(e > POSITIVE_WEIGHT_EPS for e in self.events)

    @property
    def cap_truncation_occurred(self) -> bool:
        return any(self.cap_truncation_events)


@contextmanager
def track_truncate(tracker: DiscardedWeightTracker) -> Iterator[None]:
    """Instrument ``linalg.truncate`` for discarded weight and ``min_keep``."""
    original = linalg.truncate

    def wrapped(
        s_vec: np.ndarray,
        *,
        mode: str,
        threshold: float,
        max_bond_dim: int | None = None,
        min_keep: int = 1,
    ) -> int:
        keep_without_cap = original(
            s_vec,
            mode=mode,
            threshold=threshold,
            max_bond_dim=None,
            min_keep=min_keep,
        )
        keep = original(
            s_vec,
            mode=mode,
            threshold=threshold,
            max_bond_dim=max_bond_dim,
            min_keep=min_keep,
        )
        tracker.record(
            np.asarray(s_vec),
            keep,
            min_keep=min_keep,
            max_bond_dim=max_bond_dim,
            keep_without_cap=keep_without_cap,
        )
        return keep

    linalg.truncate = wrapped  # type: ignore[assignment]
    try:
        yield
    finally:
        linalg.truncate = original


_GIT_DIFF_HASH_CACHE: str | None = None
_GIT_REVISION_CACHE: dict[str, str] | None = None


def git_diff_hash() -> str:
    """SHA256 of the exact binary working-tree diff (cached per process).

    Uses ``git diff HEAD`` (tracked modifications). Empty diff → hash of empty
    bytes. Computed at most once per process because the dirty tree can be large.
    """
    global _GIT_DIFF_HASH_CACHE
    if _GIT_DIFF_HASH_CACHE is not None:
        return _GIT_DIFF_HASH_CACHE
    try:
        # Stream through sha256 to avoid holding a multi-MB diff in memory twice.
        with subprocess.Popen(
            ["git", "diff", "HEAD"],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ) as proc:
            assert proc.stdout is not None
            h = hashlib.sha256()
            while True:
                chunk = proc.stdout.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
            proc.wait(timeout=120)
            _GIT_DIFF_HASH_CACHE = h.hexdigest()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        _GIT_DIFF_HASH_CACHE = "unavailable"
    return _GIT_DIFF_HASH_CACHE


def git_revision() -> dict[str, str]:
    """Return YAQS git metadata for manifests and row provenance (cached)."""
    global _GIT_REVISION_CACHE
    if _GIT_REVISION_CACHE is not None:
        return dict(_GIT_REVISION_CACHE)
    info = {
        "git_commit": "unknown",
        "git_dirty": "unknown",
        "git_diff_hash": "unavailable",
    }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        info["git_commit"] = commit
        info["git_dirty"] = "true" if dirty else "false"
        info["git_diff_hash"] = git_diff_hash()
    except (OSError, subprocess.CalledProcessError):
        pass
    _GIT_REVISION_CACHE = info
    return dict(info)


def git_revision_for_hash() -> dict[str, str]:
    """Stable git fields for task IDs (excludes dirty-diff hash).

    Task hashes historically include only ``git_commit`` and ``git_dirty``.
    Adding ``git_diff_hash`` would invalidate all resumable task IDs and force
    recomputation of the existing Pauli campaign.
    """
    g = git_revision()
    return {"git_commit": g["git_commit"], "git_dirty": g["git_dirty"]}


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("numpy", "scipy", "qiskit"):
        try:
            mod = __import__(name)
            versions[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            versions[name] = "unavailable"
    try:
        import mqt.yaqs as yaqs

        versions["mqt.yaqs"] = str(getattr(yaqs, "__version__", "unknown"))
    except Exception:
        versions["mqt.yaqs"] = "unavailable"
    return versions


def digital_params(
    chi: int,
    *,
    method: str,
    n_sub: int,
    svd_threshold: float | None = None,
    krylov_tol: float | None = None,
) -> DigitalSimParams:
    """Build ``DigitalSimParams`` for one approximate method."""
    return DigitalSimParams(
        observables=[],
        preset="exact",
        gate_mode=METHOD_TO_GATE_MODE[method],  # type: ignore[arg-type]
        svd_threshold=SVD_THRESHOLD if svd_threshold is None else float(svd_threshold),
        max_bond_dim=int(chi),
        krylov_tol=KRYLOV_TOL if krylov_tol is None else float(krylov_tol),
        tdvp_sweeps=int(n_sub),
        tdvp_mode=TDVP_MODE,
        trunc_mode=TRUNC_MODE,
        get_state=True,
    )


def numerical_settings_dict(
    *,
    chi: int,
    method: str,
    n_sub: int,
    svd_threshold: float | None = None,
    krylov_tol: float | None = None,
) -> dict[str, Any]:
    thr = SVD_THRESHOLD if svd_threshold is None else float(svd_threshold)
    kt = KRYLOV_TOL if krylov_tol is None else float(krylov_tol)
    return {
        "N": N,
        "bond_profile": list(BOND_PROFILE),
        "chi_max": int(chi),
        "method": method,
        "gate_mode": METHOD_TO_GATE_MODE[method],
        "n_sub": int(n_sub),
        "svd_threshold": thr,
        "krylov_tol": kt,
        "trunc_mode": TRUNC_MODE,
        "tdvp_mode": TDVP_MODE,
        "min_keep": MIN_KEEP,
        "gate_library_split_cutoff": GATE_LIBRARY_SPLIT_CUTOFF,
        "dtype": "complex128",
    }


def task_id_from_payload(payload: dict[str, Any]) -> str:
    """Stable SHA256 task ID over the complete configuration payload."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


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


def prepare_initial_state(seed: int) -> dict[str, Any]:
    """Exact-rank complex MPS for ``seed`` with the campaign bond profile."""
    rng = np.random.default_rng(int(seed))
    mps = random_mps(N, list(BOND_PROFILE), rng)
    vec = mps.to_vec().astype(np.complex128, copy=False)
    return {
        "seed": int(seed),
        "mps": mps,
        "vec": vec,
        "bond_profile": mps_bond_profile(mps),
        "norm": float(np.linalg.norm(vec)),
    }


def mps_bond_profile(mps: MPS) -> list[int]:
    profile = [int(mps.tensors[0].shape[1])]
    profile.extend(int(tensor.shape[2]) for tensor in mps.tensors)
    return profile


def final_max_bond(profile: list[int]) -> int:
    """Maximum bond in the final MPS profile (not an intra-update peak)."""
    return int(max(profile)) if profile else 1


def final_param_count(profile: list[int], length: int = N) -> int:
    """Parameter count of the final MPS profile (not an intra-update peak)."""
    return int(sum(2 * profile[i] * profile[i + 1] for i in range(length)))


# Backward-compatible aliases used by older call sites during migration.
peak_bond = final_max_bond
param_count = final_param_count


def phase_align(reference: np.ndarray, state: np.ndarray) -> np.ndarray:
    phase = np.vdot(state, reference)
    if abs(phase) > 0.0:
        return state * (phase / abs(phase))
    return state


def normalized_state_fidelity(exact: np.ndarray, approx: np.ndarray) -> dict[str, float]:
    e = np.asarray(exact, dtype=np.complex128).reshape(-1)
    a = np.asarray(approx, dtype=np.complex128).reshape(-1)
    overlap = float(abs(np.vdot(e, a)) ** 2)
    ne = float(np.real(np.vdot(e, e)))
    na = float(np.real(np.vdot(a, a)))
    if ne <= 0.0 or na <= 0.0:
        msg = f"Nonzero norms required; got ‖e‖²={ne}, ‖a‖²={na}"
        raise ValueError(msg)
    fid = overlap / (ne * na)
    if fid < -1e-12 or fid > 1.0 + 1e-12:
        msg = f"Fidelity {fid} outside [0,1] beyond roundoff"
        raise ValueError(msg)
    fid = float(min(1.0, max(0.0, fid)))
    return {
        "fidelity_normalized": fid,
        "infidelity_normalized": 1.0 - fid,
        "norm_exact": float(np.sqrt(ne)),
        "norm_approx": float(np.sqrt(na)),
        "norm_drift": float(np.sqrt(na) - np.sqrt(ne)),
    }


def state_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Phase-aligned L2 distance between two statevectors."""
    aa = np.asarray(a, dtype=np.complex128).reshape(-1)
    bb = phase_align(aa, np.asarray(b, dtype=np.complex128).reshape(-1))
    return float(np.linalg.norm(aa - bb))


def independent_r_pp_matrix(gate_type: str, theta: float) -> np.ndarray:
    """``R_PP(θ) = cos(θ/2) I - i sin(θ/2) P⊗P`` (= ``expm(-i θ P⊗P / 2)``)."""
    p = PAULI[gate_type[-1]]
    pp = np.kron(p, p)
    return (np.cos(theta / 2.0) * np.eye(4) - 1j * np.sin(theta / 2.0) * pp).astype(np.complex128)


def dense_h_cx(control: int, target: int, length: int = N) -> np.ndarray:
    """Dense ``H_CX = (π/4)(I-Z_c)⊗(I-X_t)`` on ``length`` qubits (YAQS LSB ``to_vec``)."""
    dim = 2**length
    h = np.zeros((dim, dim), dtype=np.complex128)
    iz = I2 - PAULI["z"]
    ix = I2 - PAULI["x"]
    # Build via Kronecker on bits with site 0 = LSB of YAQS to_vec.
    for i in range(dim):
        bits = [(i >> q) & 1 for q in range(length)]
        for j in range(dim):
            jbits = [(j >> q) & 1 for q in range(length)]
            if any(bits[q] != jbits[q] for q in range(length) if q not in (control, target)):
                continue
            # Matrix element of (π/4) iz_c ⊗ ix_t
            c_bra, c_ket = bits[control], jbits[control]
            t_bra, t_ket = bits[target], jbits[target]
            h[i, j] = (np.pi / 4.0) * iz[c_bra, c_ket] * ix[t_bra, t_ket]
    return h


def two_site_h_cx() -> np.ndarray:
    """Local 4×4 generator with control=qubit0, target=qubit1 in layout order."""
    return (np.pi / 4.0) * np.kron(I2 - PAULI["z"], I2 - PAULI["x"])


def cx_matrix() -> np.ndarray:
    return np.asarray(GateLibrary.cx().matrix, dtype=np.complex128)


def make_pauli_gate(gate_type: str, theta: float, q0: int, q1: int):
    factory = getattr(GateLibrary, gate_type)
    gate = factory([float(theta)])
    gate.set_sites(int(q0), int(q1))
    return gate


def make_cx_gate(control: int, target: int):
    gate = GateLibrary.cx()
    gate.set_sites(int(control), int(target))
    return gate


def make_pauli_dag_node(gate_type: str, theta: float, q0: int, q1: int, length: int = N) -> DAGOpNode:
    qc = QuantumCircuit(length)
    getattr(qc, gate_type)(float(theta), int(q0), int(q1))
    return next(iter(circuit_to_dag(qc).topological_op_nodes()))


def make_cx_dag_node(control: int, target: int, length: int = N) -> DAGOpNode:
    qc = QuantumCircuit(length)
    qc.cx(int(control), int(target))
    return next(iter(circuit_to_dag(qc).topological_op_nodes()))


def apply_gate_dense_yaqs(vec: np.ndarray, length: int, q0: int, q1: int, gate) -> np.ndarray:
    """Apply a YAQS two-qubit gate to a YAQS ``to_vec()`` statevector."""
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


def apply_cx_dense_qiskit(vec_yaqs: np.ndarray, control: int, target: int, length: int = N) -> np.ndarray:
    """Independent Qiskit Statevector CX on a YAQS LSB vector.

    Converts YAQS ``to_vec`` (site 0 = LSB) to Qiskit's little-endian convention
    (qubit 0 = least significant), applies CX, and converts back.
    """
    # YAQS and Qiskit both use qubit 0 as the least-significant amplitude bit.
    sv = Statevector(vec_yaqs)
    qc = QuantumCircuit(length)
    qc.cx(int(control), int(target))
    out = sv.evolve(qc)
    return np.asarray(out.data, dtype=np.complex128)


def _mpo_site_matrix(tensor: np.ndarray) -> np.ndarray:
    """Extract the physical (2×2) matrix from a bond-dimension-1 MPO site tensor."""
    t = np.asarray(tensor, dtype=np.complex128)
    if t.shape == (2, 2, 1, 1):
        return t[:, :, 0, 0]
    if t.shape == (1, 1, 2, 2):
        return t[0, 0]
    msg = f"Unexpected MPO site shape {t.shape}"
    raise ValueError(msg)


def mpo_to_dense(mpo_tensors: list[np.ndarray]) -> np.ndarray:
    """Contract a length-N MPO with bond dims 1 into a dense operator (YAQS LSB)."""
    # YAQS custom MPO sites are (d, d, Dl, Dr). Site 0 is the LSB of to_vec.
    acc = _mpo_site_matrix(mpo_tensors[0])
    for site in range(1, len(mpo_tensors)):
        w = _mpo_site_matrix(mpo_tensors[site])
        acc = np.kron(w, acc)
    return acc


def generator_mpo_dense(gate, length: int = N) -> np.ndarray:
    mpo, _, _ = construct_generator_mpo(gate, length)
    return mpo_to_dense(list(mpo.tensors))


def apply_method(
    initial_mps: MPS,
    node: DAGOpNode,
    *,
    method: str,
    chi: int,
    n_sub: int,
    svd_threshold: float | None = None,
    krylov_tol: float | None = None,
    tracker: DiscardedWeightTracker | None = None,
) -> tuple[MPS, float]:
    """Apply one gate with the requested approximate method."""
    params = digital_params(
        chi,
        method=method,
        n_sub=n_sub,
        svd_threshold=svd_threshold,
        krylov_tol=krylov_tol,
    )
    state = copy.deepcopy(initial_mps)
    if tracker is not None:
        tracker.reset_gate()
        with track_truncate(tracker):
            apply_two_qubit_gate(state, node, params)
        return state, float(tracker.per_gate)
    apply_two_qubit_gate(state, node, params)
    return state, float("nan")


def conventional_median(values: list[float] | np.ndarray) -> float:
    """Median via ``numpy.median`` (mean of the two middle values for even n)."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class DirectoryLock:
    """Prevent concurrent writers to the same output directory."""

    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / ".campaign.lock"
        self.acquired = False

    def acquire(self) -> None:
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            msg = f"Output directory locked: {self.path} (pid={payload.get('pid')}, started={payload.get('started')})"
            raise RuntimeError(msg)
        payload = {"pid": os.getpid(), "started": datetime.now(UTC).isoformat()}
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self.acquired = True

    def release(self) -> None:
        if self.acquired and self.path.exists():
            self.path.unlink(missing_ok=True)
        self.acquired = False


def utc_now() -> str:
    return datetime.now(UTC).isoformat()
