# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Gate application and trajectory metrics for fixed-resource circuits."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import path_setup  # ruff: ignore[unused-import]
from circuits import GateOp, TrotterStep, neel_basis_string
from config import NUM_QUBITS, TDVP_SUBSTEPS
from gate_runtime import (
    DiscardedWeightTracker,
    apply_two_qubit_dense,
    bond_profile,
    make_gate,
    param_count_from_profile,
    phase_align,
    track_discarded_weight,
)
from qiskit.quantum_info import Statevector

from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.state_utils import product_state_vector
from mqt.yaqs.digital.digital_tjm import apply_single_qubit_gate, apply_two_qubit_gate


@dataclass
class TrajectoryState:
    """Mutable simulation state for one method."""

    mps: MPS
    vec: np.ndarray
    cumulative_runtime_s: float = 0.0
    peak_bond: int = 1
    peak_memory_bytes: int = 0
    peak_param_count: int = 0
    peak_intermediate_elements: int = 0
    step_discarded: float = 0.0
    step_compression: float = 0.0
    failed: bool = False
    failure_message: str = ""
    variational_init: str = ""
    variational_converged: bool | None = None
    variational_sweeps: int | None = None
    num_qubits: int = NUM_QUBITS


def initial_mps(
    model: str,
    *,
    num_qubits: int | None = None,
    num_rows: int | None = None,
    num_cols: int | None = None,
) -> MPS:
    from config import NUM_COLS as NC
    from config import NUM_ROWS as NR

    nq = num_qubits if num_qubits is not None else NUM_QUBITS
    if model == "ising":
        return MPS(nq, state="zeros")
    rows = num_rows if num_rows is not None else NR
    cols = num_cols if num_cols is not None else NC
    return MPS(nq, state="basis", basis_string=neel_basis_string(num_rows=rows, num_cols=cols))


def initial_vector(
    model: str,
    *,
    num_qubits: int | None = None,
    num_rows: int | None = None,
    num_cols: int | None = None,
) -> np.ndarray:
    from config import NUM_COLS as NC
    from config import NUM_ROWS as NR

    nq = num_qubits if num_qubits is not None else NUM_QUBITS
    if model == "ising":
        return product_state_vector(nq, "zeros", 2)
    rows = num_rows if num_rows is not None else NR
    cols = num_cols if num_cols is not None else NC
    return product_state_vector(nq, "basis", 2, basis_string=neel_basis_string(num_rows=rows, num_cols=cols))


def apply_gate_dense(vec: np.ndarray, gate: GateOp, *, num_qubits: int = NUM_QUBITS) -> np.ndarray:
    if len(gate.qubits) == 1:
        from qiskit.circuit import QuantumCircuit
        from qiskit.quantum_info import Statevector as QSV

        mini = QuantumCircuit(num_qubits)
        getattr(mini, gate.name)(gate.theta, gate.qubits[0])
        return np.asarray(QSV(vec).evolve(mini).data, dtype=np.complex128)
    g = make_gate(gate.name, gate.theta, gate.qubits[0], gate.qubits[1])
    return apply_two_qubit_dense(vec, num_qubits, gate.qubits[0], gate.qubits[1], g)


def _gate_params(
    chi: int,
    method: str,
    *,
    tdvp_substeps: int = TDVP_SUBSTEPS,
    svd_threshold: float | None = None,
):
    from gate_runtime import _params

    modes = {
        "hybrid_tdvp": "tdvp",
        "full_tdvp": "full-tdvp",  # TDVP window update for every two-qubit gate (incl. NN)
        "tebd_swap": "swaps",
        "mpo_zipup": "mpo",
    }
    return _params(
        chi,
        gate_mode=modes[method],
        tdvp_sweeps=tdvp_substeps,
        svd_threshold=svd_threshold,
    )


def _apply_variational_two_qubit(state: TrajectoryState, node, *, chi: int) -> None:
    """Apply repaired variational MPO (input / zip-up inits only; no TDVP init).

    On numerical residual-monotonicity failure, fall back to the repaired zip-up
    state for that gate and record the failure reason (do not abort the circuit).
    """
    import sys
    from pathlib import Path

    sg = Path(__file__).resolve().parents[1] / "single_gate"
    if str(sg) not in sys.path:
        sys.path.append(str(sg))
    from gate_runtime import _params
    from variational import apply_variational_mpo_gate

    from mqt.yaqs.core.data_structures.mpo import MPO
    from mqt.yaqs.digital.digital_tjm import convert_dag_to_tensor_algorithm

    try:
        result = apply_variational_mpo_gate(
            state.mps,
            node,
            chi=chi,
            max_sweeps=4,
            residual_tol=1e-8,
            require_exact_when_chi_ge=None,
        )
        state.mps = result.state
        state.variational_init = result.best_initializer
        state.variational_converged = result.converged
        state.variational_sweeps = result.sweeps
        state.step_compression += float(result.objective_final)
    except RuntimeError as exc:
        gate = convert_dag_to_tensor_algorithm(node)[0]
        fallback = copy.deepcopy(state.mps)
        mpo = MPO.from_gate(gate, fallback.length)
        mpo.multiply(fallback, sim_params=_params(chi, gate_mode="mpo", tdvp_sweeps=1), compress=True)
        state.mps = fallback
        state.variational_init = "zipup_fallback"
        state.variational_converged = False
        state.variational_sweeps = 0
        state.failure_message = f"variational_fallback:{exc}"
        # Do not set state.failed — zip-up fallback keeps the circuit running.


def apply_gate_mps(
    state: TrajectoryState,
    gate: GateOp,
    *,
    method: str,
    chi: int,
    tdvp_substeps: int = TDVP_SUBSTEPS,
    svd_threshold: float | None = None,
) -> None:
    if state.failed:
        return
    nq = state.num_qubits
    node = gate.to_dag_node(nq)
    tracker = DiscardedWeightTracker()
    t0 = time.perf_counter()
    try:
        if len(gate.qubits) == 1:
            apply_single_qubit_gate(state.mps, node)
            state.step_discarded += tracker.per_gate
        elif method == "variational_mpo":
            _apply_variational_two_qubit(state, node, chi=chi)
        else:
            with track_discarded_weight(tracker):
                apply_two_qubit_gate(
                    state.mps,
                    node,
                    _gate_params(
                        chi, method, tdvp_substeps=tdvp_substeps, svd_threshold=svd_threshold
                    ),
                )
            state.step_discarded += tracker.per_gate
    except Exception as exc:  # ruff: ignore[blind-except]
        state.failed = True
        state.failure_message = str(exc)
        state.cumulative_runtime_s += time.perf_counter() - t0
        return
    state.cumulative_runtime_s += time.perf_counter() - t0
    prof = bond_profile(state.mps)
    state.peak_bond = max(state.peak_bond, max(prof) if prof else 1)
    params = param_count_from_profile(prof, nq)
    state.peak_param_count = max(state.peak_param_count, params)
    state.peak_memory_bytes = max(state.peak_memory_bytes, params * 16)


def apply_trotter_step_dense(
    vec: np.ndarray, step: TrotterStep, *, num_qubits: int = NUM_QUBITS
) -> np.ndarray:
    out = vec
    for gate in step.gates:
        out = apply_gate_dense(out, gate, num_qubits=num_qubits)
    return out


def apply_trotter_step_mps(
    state: TrajectoryState,
    step: TrotterStep,
    *,
    method: str,
    chi: int,
    tdvp_substeps: int = TDVP_SUBSTEPS,
    update_vec: bool = True,
    svd_threshold: float | None = None,
) -> None:
    state.step_discarded = 0.0
    state.step_compression = 0.0
    for gate in step.gates:
        apply_gate_mps(
            state,
            gate,
            method=method,
            chi=chi,
            tdvp_substeps=tdvp_substeps,
            svd_threshold=svd_threshold,
        )
    if update_vec:
        state.vec = state.mps.to_vec().astype(np.complex128, copy=False)


def compute_metrics(
    exact_vec: np.ndarray,
    approx_vec: np.ndarray,
    *,
    state: TrajectoryState,
    model: str,
    method: str,
    chi: int,
    trotter_step: int,
    time: float,
    step_runtime_s: float,
) -> dict[str, Any]:
    ex_norm = float(np.linalg.norm(exact_vec))
    ap_norm = float(np.linalg.norm(approx_vec))
    exact_n = exact_vec / ex_norm if ex_norm > 0 else exact_vec
    approx_n = approx_vec / ap_norm if ap_norm > 0 else approx_vec
    raw_fid = float(abs(np.vdot(exact_n, approx_n)) ** 2)
    # Preserve roundoff-level negative (1-F); do not floor here.
    raw_infidelity = 1.0 - raw_fid
    infidelity = raw_infidelity
    aligned = phase_align(exact_n, approx_n)
    vec_dist = float(np.linalg.norm(aligned - exact_n))
    nq = state.num_qubits
    prof = bond_profile(state.mps)
    params = param_count_from_profile(prof, nq)
    mem_bytes = params * 16
    return {
        "model": model,
        "method": method,
        "chi_max": chi,
        "trotter_step": trotter_step,
        "time": time,
        "infidelity": infidelity,
        "fidelity": raw_fid,
        "state_norm": ap_norm,
        "norm_drift": abs(ap_norm - 1.0),
        "phase_aligned_distance": vec_dist,
        "current_max_bond": max(prof) if prof else 1,
        "peak_max_bond": state.peak_bond,
        "bond_profile": prof,
        "param_count": params,
        "memory_bytes": mem_bytes,
        "peak_param_count": state.peak_param_count,
        "peak_memory_bytes": state.peak_memory_bytes,
        "largest_intermediate_elements": state.peak_intermediate_elements,
        "step_runtime_s": step_runtime_s,
        "cumulative_runtime_s": state.cumulative_runtime_s,
        "discarded_weight_step": state.step_discarded,
        "compression_residual_step": state.step_compression,
        "variational_init": state.variational_init,
        "variational_converged": state.variational_converged,
        "variational_sweeps": state.variational_sweeps,
        "failed": int(state.failed),
        "failure_message": state.failure_message,
        "tdvp_substeps": "",  # filled by caller
    }


def attach_tdvp_substeps(row: dict[str, Any], tdvp_substeps: int) -> dict[str, Any]:
    out = dict(row)
    out["tdvp_substeps"] = int(tdvp_substeps)
    return out


def qiskit_reference(model: str, *, timesteps: int) -> np.ndarray:
    from circuits import build_qiskit_circuit

    qc = build_qiskit_circuit(model, timesteps=timesteps)
    init = initial_vector(model)
    if timesteps == 0:
        return init
    return np.asarray(Statevector(init).evolve(qc).data, dtype=np.complex128)
