# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Shared builders, instrumentation, and method runners for the generator-rank pilot.

Tests whether complete-generator TDVP becomes advantageous when the generator MPO
is compact (D_H small) but the layer unitary has much larger operator bond
dimension (D_U). Two candidates: a 2D QAOA/Ising cost layer and a collective
one-axis-twisting (OAT) entangler.

Conventions (documented, validated in ``run_pilot.py --validate``):
* Library gate ``Rzz(theta) = exp(-i (theta/2) Z⊗Z)``; a QAOA cost layer
  ``U_C(gamma) = exp(-i gamma sum_edges ZZ)`` therefore uses edge gates with
  ``theta = 2*gamma`` (equivalently ``gamma = theta/2``).
* Library gate ``Rxx(theta) = exp(-i (theta/2) X⊗X)``; the OAT layer
  ``exp(-i kappa/(N-1) sum_{i<j} X_i X_j)`` uses pair gates ``theta = 2*kappa/(N-1)``.
* TDVP evolves ``exp(-i H_mpo * 1)`` in ``n`` fractional substeps (production
  digital semantics), so the generator MPO carries the full prefactor
  (``gamma`` per edge / ``kappa/(N-1)`` per pair).
* Numerical settings are the validated corrected-benchmark production values:
  ``svd_threshold=1e-13`` (discarded_weight), ``krylov_tol=1e-12``,
  gate-library ``split_tensor`` hard cutoff ``1e-14`` (fixed, not swept).
"""

from __future__ import annotations

import copy
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import path_setup  # noqa: F401
from gate_runtime import (
    KRYLOV_TOL,
    SVD_THRESHOLD,
    TRUNC_MODE,
    _params,
    apply_two_qubit_dense,
    make_dag_node,
    make_gate,
    normalized_state_fidelity,
)

from mqt.yaqs.core import linalg
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
from mqt.yaqs.core.methods.tdvp import integrators
from mqt.yaqs.core.methods.tdvp.tdvp import tdvp
from mqt.yaqs.digital.digital_tjm import apply_two_qubit_gate

if TYPE_CHECKING:
    from collections.abc import Iterator

# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------


@dataclass
class PilotTracker:
    """Records SVD/truncation events and local-evolver calls during one run."""

    n_svd: int = 0
    max_svd_elements: int = 0
    n_truncate: int = 0
    total_discarded: float = 0.0
    max_event_discarded: float = 0.0
    n_evolver_calls: int = 0
    per_step_discarded: list[float] = field(default_factory=list)
    _step_acc: float = 0.0

    def start_step(self) -> None:
        self._step_acc = 0.0

    def end_step(self) -> None:
        self.per_step_discarded.append(self._step_acc)

    @property
    def max_step_discarded(self) -> float:
        return max(self.per_step_discarded, default=0.0)


@contextmanager
def track_pilot(tracker: PilotTracker) -> Iterator[None]:
    """Instrument ``linalg.svd``/``linalg.truncate`` and TDVP evolver calls."""
    orig_svd = linalg.svd
    orig_trunc = linalg.truncate
    orig_site = integrators.update_site
    orig_bond = integrators.update_bond

    def svd_wrap(matrix: np.ndarray, *args: Any, **kwargs: Any):
        tracker.n_svd += 1
        tracker.max_svd_elements = max(tracker.max_svd_elements, int(matrix.shape[0]) * int(matrix.shape[1]))
        return orig_svd(matrix, *args, **kwargs)

    def trunc_wrap(s_vec: np.ndarray, **kwargs: Any) -> int:
        keep = orig_trunc(s_vec, **kwargs)
        tracker.n_truncate += 1
        s = np.asarray(s_vec)
        total = float(np.sum(np.square(s)))
        if total > 0.0:
            disc = float(np.sum(np.square(s[keep:])) / total)
            tracker.total_discarded += disc
            tracker._step_acc += disc  # noqa: SLF001
            tracker.max_event_discarded = max(tracker.max_event_discarded, disc)
        return keep

    def site_wrap(*args: Any, **kwargs: Any):
        tracker.n_evolver_calls += 1
        return orig_site(*args, **kwargs)

    def bond_wrap(*args: Any, **kwargs: Any):
        tracker.n_evolver_calls += 1
        return orig_bond(*args, **kwargs)

    linalg.svd = svd_wrap  # type: ignore[assignment]
    linalg.truncate = trunc_wrap  # type: ignore[assignment]
    integrators.update_site = site_wrap  # type: ignore[assignment]
    integrators.update_bond = bond_wrap  # type: ignore[assignment]
    try:
        yield
    finally:
        linalg.svd = orig_svd  # type: ignore[assignment]
        linalg.truncate = orig_trunc  # type: ignore[assignment]
        integrators.update_site = orig_site  # type: ignore[assignment]
        integrators.update_bond = orig_bond  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def snake_index(row: int, col: int, width: int) -> int:
    """Snake (boustrophedon) MPS ordering, matching the corrected 2D benchmarks."""
    if row % 2 == 0:
        return row * width + col
    return row * width + (width - 1 - col)


def grid_edges(width: int) -> list[tuple[int, int]]:
    """Open W×W square-lattice edges in snake indices (horizontal then vertical)."""
    edges: list[tuple[int, int]] = []
    for row in range(width):
        edges.extend((snake_index(row, col, width), snake_index(row, col + 1, width)) for col in range(width - 1))
    for col in range(width):
        edges.extend((snake_index(row, col, width), snake_index(row + 1, col, width)) for row in range(width - 1))
    return edges


def qaoa_gate_list(width: int, theta: float, ordering: str) -> list[tuple[str, float, int, int]]:
    """Edge Rzz gates for one cost layer. All gates commute; ordering affects truncation only."""
    horiz: list[tuple[int, int]] = []
    vert: list[tuple[int, int]] = []
    for row in range(width):
        horiz.extend((snake_index(row, col, width), snake_index(row, col + 1, width)) for col in range(width - 1))
    for col in range(width):
        vert.extend((snake_index(row, col, width), snake_index(row + 1, col, width)) for row in range(width - 1))
    if ordering == "horiz_first":
        pairs = horiz + vert
    elif ordering == "vert_first":
        pairs = vert + horiz
    else:
        msg = f"Unknown QAOA ordering {ordering!r}"
        raise ValueError(msg)
    return [("rzz", theta, a, b) for a, b in pairs]


def oat_gate_list(n: int, kappa: float, ordering: str) -> list[tuple[str, float, int, int]]:
    """All-to-all Rxx gates equivalent to the OAT layer. theta = 2*kappa/(N-1)."""
    theta = 2.0 * kappa / (n - 1)
    pairs: list[tuple[int, int]] = []
    if ordering == "lexicographic":
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    elif ordering == "by_distance":
        for dist in range(1, n):
            pairs.extend((i, i + dist) for i in range(n - dist))
    else:
        msg = f"Unknown OAT ordering {ordering!r}"
        raise ValueError(msg)
    return [("rxx", theta, a, b) for a, b in pairs]


def qaoa_generator_mpo(width: int, gamma: float) -> MPO:
    """Generator MPO for the cost layer: gamma * sum_edges Z_i Z_j."""
    terms = [(gamma, f"Z{a} Z{b}") for a, b in grid_edges(width)]
    mpo = MPO()
    mpo.from_pauli_sum(terms=terms, length=width * width, tol=1e-12, n_sweeps=2)
    return mpo


def oat_generator_mpo(n: int, kappa: float) -> MPO:
    """Generator MPO for OAT: kappa/(N-1) * sum_{i<j} X_i X_j."""
    coeff = kappa / (n - 1)
    terms = [(coeff, f"X{i} X{j}") for i in range(n) for j in range(i + 1, n)]
    mpo = MPO()
    mpo.from_pauli_sum(terms=terms, length=n, tol=1e-12, n_sweeps=2)
    return mpo


def mpo_max_bond(mpo: MPO) -> int:
    """Maximum virtual bond dimension of an MPO (convention-safe: max of both virtual legs)."""
    return max(max(int(t.shape[2]), int(t.shape[3])) for t in mpo.tensors)


def build_layer_unitary_mpo(gates: list[tuple[str, float, int, int]], length: int, tol: float = 1e-12) -> MPO:
    """Multiply all gate MPOs into an identity MPO, compressed at ``tol`` (uncapped)."""
    target = MPO.identity(length)
    for name, theta, q0, q1 in gates:
        gate = make_gate(name, theta, q0, q1)
        MPO.from_gate(gate, length).multiply(target, compress=True, tol=tol, max_bond_dim=None, n_sweeps=1)
    return target


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


def dense_initial(length: int, kind: str) -> np.ndarray:
    if kind == "x+":
        return np.full(2**length, 2.0 ** (-length / 2.0), dtype=np.complex128)
    if kind == "zeros":
        vec = np.zeros(2**length, dtype=np.complex128)
        vec[0] = 1.0
        return vec
    msg = f"Unknown initial kind {kind!r}"
    raise ValueError(msg)


def dense_reference(length: int, kind: str, gates: list[tuple[str, float, int, int]]) -> np.ndarray:
    """Exact dense state via sequential exact gate application (convention-safe)."""
    vec = dense_initial(length, kind)
    for name, theta, q0, q1 in gates:
        gate = make_gate(name, theta, q0, q1)
        vec = apply_two_qubit_dense(vec, length, q0, q1, gate)
    return vec


def exact_sim_params() -> StrongSimParams:
    """Uncapped, effectively exact gate-application settings (threshold 1e-15)."""
    return StrongSimParams(
        observables=[],
        preset="exact",
        gate_mode="mpo",
        svd_threshold=1e-15,
        max_bond_dim=None,
        krylov_tol=KRYLOV_TOL,
        tdvp_sweeps=1,
        tdvp_mode="2site",
        trunc_mode="discarded_weight",
        get_state=False,
    )


def exact_mps_reference(length: int, kind: str, gates: list[tuple[str, float, int, int]]) -> MPS:
    """Exact MPS via uncapped gatewise MPO application (for sizes without dense refs)."""
    state = MPS(length, state=kind if kind != "zeros" else "zeros")
    params = exact_sim_params()
    for name, theta, q0, q1 in gates:
        node = make_dag_node(name, theta, q0, q1, length)
        apply_two_qubit_gate(state, node, params)
    norm = float(np.sqrt(np.real(state.scalar_product(state))))
    state.tensors[0] /= norm
    return state


def mps_fidelity(exact: MPS, approx: MPS) -> float:
    """Normalized fidelity between two MPSs via scalar products."""
    ee = float(np.real(exact.scalar_product(exact)))
    aa = float(np.real(approx.scalar_product(approx)))
    ov = complex(exact.scalar_product(approx))
    return float(abs(ov) ** 2 / (ee * aa))


def bond_profile(mps: MPS) -> list[int]:
    return [int(t.shape[2]) for t in mps.tensors[:-1]]


def param_count(mps: MPS) -> int:
    return int(sum(int(np.prod(t.shape)) for t in mps.tensors))


# ---------------------------------------------------------------------------
# Method runners (each returns metrics dict; timing excludes reference work)
# ---------------------------------------------------------------------------


def _finish(
    record: dict[str, Any],
    state: MPS,
    tracker: PilotTracker,
    runtime: float,
    *,
    exact_vec: np.ndarray | None,
    exact_mps: MPS | None,
    peak_params: int,
    peak_bond: int,
    est_transient_elements: int,
) -> dict[str, Any]:
    if exact_vec is not None:
        # clip_tol=1e-9: near-exact runs can land at 1 + O(1e-12) from rounding.
        fid = normalized_state_fidelity(exact_vec, state.to_vec(), clip_tol=1e-9)
        fidelity = fid["fidelity_normalized"]
        norm = fid["norm_approx"]
    else:
        assert exact_mps is not None
        fidelity = mps_fidelity(exact_mps, state)
        norm = float(np.sqrt(np.real(state.scalar_product(state))))
    prof = bond_profile(state)
    record.update({
        "infidelity": 1.0 - fidelity,
        "fidelity": fidelity,
        "state_norm": norm,
        "final_max_bond": max(prof) if prof else 1,
        "peak_max_bond": max(peak_bond, max(prof) if prof else 1),
        "final_param_count": param_count(state),
        "peak_param_count": max(peak_params, param_count(state)),
        "est_transient_elements": max(est_transient_elements, tracker.max_svd_elements),
        "total_discarded_weight": tracker.total_discarded,
        "max_step_discarded_weight": tracker.max_step_discarded,
        "max_event_discarded_weight": tracker.max_event_discarded,
        "n_svd": tracker.n_svd,
        "n_evolver_calls": tracker.n_evolver_calls,
        "runtime_s": runtime,
    })
    return record


def run_tdvp_layer(
    initial: MPS,
    h_mpo: MPO,
    *,
    chi: int,
    substeps: int,
    exact_vec: np.ndarray | None,
    exact_mps: MPS | None,
) -> dict[str, Any]:
    """Complete-generator 2TDVP: one call, n fractional substeps (production semantics)."""
    state = copy.deepcopy(initial)
    params = _params(chi, gate_mode="tdvp", tdvp_sweeps=substeps)
    tracker = PilotTracker()
    tracker.start_step()
    t0 = time.perf_counter()
    with track_pilot(tracker):
        tdvp(state, h_mpo, params)
    runtime = time.perf_counter() - t0
    tracker.end_step()
    rec: dict[str, Any] = {"method": "tdvp_layer", "ordering": "", "substeps": substeps}
    return _finish(
        rec, state, tracker, runtime,
        exact_vec=exact_vec, exact_mps=exact_mps,
        peak_params=0, peak_bond=0, est_transient_elements=0,
    )


def run_mpo_gatewise(
    initial: MPS,
    gates: list[tuple[str, float, int, int]],
    *,
    chi: int,
    ordering: str,
    exact_vec: np.ndarray | None,
    exact_mps: MPS | None,
) -> dict[str, Any]:
    """Corrected production gatewise route (NN gates -> TEBD, long-range -> MPO+compress)."""
    length = initial.length
    state = copy.deepcopy(initial)
    params = _params(chi, gate_mode="mpo", tdvp_sweeps=1)
    tracker = PilotTracker()
    peak_params = 0
    peak_bond = 0
    est_transient = 0
    nodes = [make_dag_node(name, theta, q0, q1, length) for name, theta, q0, q1 in gates]
    t0 = time.perf_counter()
    with track_pilot(tracker):
        for node, (_, _, q0, q1) in zip(nodes, gates, strict=True):
            tracker.start_step()
            # Estimated pre-compression size: gate-MPO rank 2 on support bonds.
            prof = bond_profile(state)
            lo, hi = min(q0, q1), max(q0, q1)
            inter = list(prof)
            for b in range(lo, hi):
                inter[b] = min(inter[b] * 2, 2 ** min(b + 1, length - 1 - b))
            est = 2 * sum(
                (inter[i - 1] if i > 0 else 1) * (inter[i] if i < length - 1 else 1) for i in range(length)
            )
            est_transient = max(est_transient, est)
            apply_two_qubit_gate(state, node, params)
            tracker.end_step()
            peak_params = max(peak_params, param_count(state))
            prof = bond_profile(state)
            peak_bond = max(peak_bond, max(prof) if prof else 1)
    runtime = time.perf_counter() - t0
    rec: dict[str, Any] = {"method": "mpo_gatewise", "ordering": ordering, "substeps": 0}
    return _finish(
        rec, state, tracker, runtime,
        exact_vec=exact_vec, exact_mps=exact_mps,
        peak_params=peak_params, peak_bond=peak_bond, est_transient_elements=est_transient,
    )


def run_mpo_layer(
    initial: MPS,
    layer_mpo: MPO,
    *,
    chi: int,
    exact_vec: np.ndarray | None,
    exact_mps: MPS | None,
) -> dict[str, Any]:
    """Complete-layer unitary MPO applied once, then repaired compression at chi.

    The layer MPO is prebuilt (its construction cost is reported separately);
    timing covers application + compression only.
    """
    state = copy.deepcopy(initial)
    tracker = PilotTracker()
    t0 = time.perf_counter()
    with track_pilot(tracker):
        tracker.start_step()
        layer_mpo.multiply(state, sim_params=None, compress=False)
        intermediate_params = param_count(state)
        intermediate_bond = max(bond_profile(state), default=1)
        state.compress(SVD_THRESHOLD, max_bond_dim=chi, trunc_mode=TRUNC_MODE)
        tracker.end_step()
    runtime = time.perf_counter() - t0
    rec: dict[str, Any] = {"method": "mpo_layer", "ordering": "", "substeps": 0}
    return _finish(
        rec, state, tracker, runtime,
        exact_vec=exact_vec, exact_mps=exact_mps,
        peak_params=intermediate_params, peak_bond=intermediate_bond,
        est_transient_elements=intermediate_params,
    )


def run_variational_layer(
    target_exact: MPS,
    init_state: MPS,
    *,
    chi: int,
    exact_vec: np.ndarray | None,
    exact_mps: MPS | None,
) -> dict[str, Any]:
    """Variational fit of a chi-capped MPS to the exact layer-applied state."""
    from variational import variational_fit

    tracker = PilotTracker()
    t0 = time.perf_counter()
    with track_pilot(tracker):
        tracker.start_step()
        result = variational_fit(
            copy.deepcopy(target_exact),
            copy.deepcopy(init_state),
            chi=chi,
            max_sweeps=8,
            residual_tol=1e-8,
        )
        tracker.end_step()
    runtime = time.perf_counter() - t0
    rec: dict[str, Any] = {
        "method": "variational_layer",
        "ordering": "",
        "substeps": 0,
        "notes": f"sweeps={result.sweeps} converged={result.converged}",
    }
    return _finish(
        rec, result.state, tracker, runtime,
        exact_vec=exact_vec, exact_mps=exact_mps,
        peak_params=0, peak_bond=0, est_transient_elements=0,
    )


def run_oracle(
    exact_reference_mps: MPS,
    *,
    chi: int,
    exact_vec: np.ndarray | None,
    exact_mps: MPS | None,
) -> dict[str, Any]:
    """Diagnostic (not algorithmic): sequential rank-chi truncation of the exact state."""
    state = copy.deepcopy(exact_reference_mps)
    tracker = PilotTracker()
    t0 = time.perf_counter()
    with track_pilot(tracker):
        tracker.start_step()
        state.compress(1e-16, max_bond_dim=chi, trunc_mode="discarded_weight")
        tracker.end_step()
    runtime = time.perf_counter() - t0
    rec: dict[str, Any] = {"method": "oracle_compress", "ordering": "", "substeps": 0}
    return _finish(
        rec, state, tracker, runtime,
        exact_vec=exact_vec, exact_mps=exact_mps,
        peak_params=0, peak_bond=0, est_transient_elements=0,
    )


def settings_note() -> dict[str, Any]:
    return {
        "svd_threshold": SVD_THRESHOLD,
        "trunc_mode": TRUNC_MODE,
        "krylov_tol": KRYLOV_TOL,
        "gate_library_split_tensor_hard_cutoff": 1e-14,
        "tdvp_mode": "2site",
        "layer_mpo_tol": 1e-12,
        "exact_gatewise_threshold": 1e-15,
    }
