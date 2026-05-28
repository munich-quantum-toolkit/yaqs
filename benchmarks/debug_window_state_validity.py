#!/usr/bin/env python
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compare TDVP on extracted vs fresh vs dense-rebuilt window states.

Run:

    uv run python -m benchmarks.debug_window_state_validity
"""

from __future__ import annotations

import copy

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Statevector

from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.methods.tdvp import two_site_tdvp
from mqt.yaqs.digital.digital_tjm import apply_window, construct_generator_mpo
from mqt.yaqs.digital.utils.dag_utils import convert_dag_to_tensor_algorithm


def vector_to_mps(vec: np.ndarray, n: int, *, svd_threshold: float = 1e-14) -> MPS:
    """Exact open-boundary MPS from dense vector via successive SVDs."""
    psi = np.asarray(vec, dtype=np.complex128).reshape([2] * n)
    tensors: list[np.ndarray] = []
    chi_left = 1
    psi_mat = psi.reshape(2, -1)

    for site in range(n - 1):
        u, s, vh = np.linalg.svd(psi_mat, full_matrices=False)
        keep = s > svd_threshold
        if not np.any(keep):
            keep = slice(0, 1)
        u = u[:, keep]
        s = s[keep]
        vh = vh[keep, :]

        chi_right = u.shape[1]
        tensors.append(u.reshape(2, chi_left, chi_right))

        psi_mat = (np.diag(s) @ vh).reshape(chi_right * 2, -1)
        chi_left = chi_right

    tensors.append(psi_mat.reshape(2, chi_left, 1))
    return MPS(length=n, tensors=[np.asarray(t, dtype=np.complex128) for t in tensors])


def _gate_from_single_gate_circuit(qc: QuantumCircuit):
    dag = circuit_to_dag(qc)
    nodes = list(dag.topological_op_nodes())
    if len(nodes) != 1:
        raise ValueError("Expected exactly one gate.")
    return convert_dag_to_tensor_algorithm(nodes[0])[0]


def _mpo_from_tensors(tensors) -> MPO:
    mpo = MPO()
    mpo.custom(list(tensors), transpose=False)
    return mpo


def _tensor_norm_delta(before: list[np.ndarray], after: list[np.ndarray]) -> float:
    b = np.asarray([np.linalg.norm(t) for t in before], dtype=float)
    a = np.asarray([np.linalg.norm(t) for t in after], dtype=float)
    return float(np.linalg.norm(a - b))


def run_case(*, n: int, sites: tuple[int, int], theta: float, window: tuple[int, int]) -> None:
    qc_full = QuantumCircuit(n)
    qc_full.ryy(theta, sites[0], sites[1])
    gate = _gate_from_single_gate_circuit(qc_full)

    # Full MPO and windowed MPO
    full_mpo, first_site, last_site = construct_generator_mpo(gate, n)
    sliced_mpo = _mpo_from_tensors(full_mpo.tensors[window[0] : window[1] + 1])

    # Local window parameters
    win_len = window[1] - window[0] + 1
    local_sites = (sites[0] - window[0], sites[1] - window[0])

    local_gate = copy.deepcopy(gate)
    local_gate.set_sites(local_sites[0], local_sites[1])
    local_mpo, *_ = construct_generator_mpo(local_gate, win_len)

    # Reference: Qiskit local window
    qc_local = QuantumCircuit(win_len)
    qc_local.ryy(theta, local_sites[0], local_sites[1])
    ref = np.asarray(Statevector(qc_local).data, dtype=np.complex128)

    params = StrongSimParams(
        preset="exact",
        get_state=True,
        svd_threshold=1e-14,
        gate_mode="hybrid",
        tdvp_sweeps=1,
        tdvp_circuit_full_sweep=False,
    )

    # A) Extracted-window path
    full_state = State(n, initial="zeros", representation="mps", pad=None)
    full_mps = full_state.mps
    full_mps_copy = copy.deepcopy(full_mps)
    short_state_a, short_mpo_a, win_a = apply_window(full_mps_copy, full_mpo, first_site, last_site, window_size=0)
    # Override to the requested window explicitly
    short_state_a = MPS(length=win_len, tensors=full_mps_copy.tensors[window[0] : window[1] + 1])
    short_mpo_a = sliced_mpo

    a_before = [t.copy() for t in short_state_a.tensors]
    try:
        a_vec_before = np.asarray(short_state_a.to_vec(), dtype=np.complex128)
    except Exception:
        a_vec_before = None
    two_site_tdvp(short_state_a, short_mpo_a, params)
    a_after = [t.copy() for t in short_state_a.tensors]
    try:
        a_vec_after = np.asarray(short_state_a.to_vec(), dtype=np.complex128)
    except Exception:
        a_vec_after = None

    # B) Fresh local zeros state
    short_state_b = MPS(length=win_len, state="zeros", pad=None)
    b_before = [t.copy() for t in short_state_b.tensors]
    b_vec_before = np.asarray(short_state_b.to_vec(), dtype=np.complex128)
    two_site_tdvp(short_state_b, local_mpo, params)
    b_after = [t.copy() for t in short_state_b.tensors]
    b_vec_after = np.asarray(short_state_b.to_vec(), dtype=np.complex128)

    # C) Dense-rebuilt window state from A (if A has a vector)
    short_state_c = None
    c_vec_after = None
    c_delta_t = None
    c_delta_v = None
    if a_vec_before is not None:
        short_state_c = vector_to_mps(a_vec_before, win_len, svd_threshold=1e-14)
        c_before = [t.copy() for t in short_state_c.tensors]
        c_vec_before = np.asarray(short_state_c.to_vec(), dtype=np.complex128)
        two_site_tdvp(short_state_c, local_mpo, params)
        c_after = [t.copy() for t in short_state_c.tensors]
        c_vec_after = np.asarray(short_state_c.to_vec(), dtype=np.complex128)
        c_delta_t = _tensor_norm_delta(c_before, c_after)
        c_delta_v = float(np.linalg.norm(c_vec_after - c_vec_before))

    def fid_err(v: np.ndarray) -> float:
        return float(1.0 - abs(np.vdot(ref, v)) ** 2)

    print(f"\n=== RYY theta={theta} global_sites={sites} window={window} local_sites={local_sites} ===")
    print("A: extracted-window")
    print("  boundary shapes:", tuple(short_state_a.tensors[0].shape), tuple(short_state_a.tensors[-1].shape))
    print("  tensor_norm_delta:", _tensor_norm_delta(a_before, a_after))
    if a_vec_before is not None and a_vec_after is not None:
        print("  statevec_norm_delta:", float(np.linalg.norm(a_vec_after - a_vec_before)))
        print("  fid_err vs local Qiskit:", fid_err(a_vec_after))
    else:
        print("  statevec_norm_delta: (unavailable)")

    print("B: fresh local zeros")
    print("  boundary shapes:", tuple(short_state_b.tensors[0].shape), tuple(short_state_b.tensors[-1].shape))
    print("  tensor_norm_delta:", _tensor_norm_delta(b_before, b_after))
    print("  statevec_norm_delta:", float(np.linalg.norm(b_vec_after - b_vec_before)))
    print("  fid_err vs local Qiskit:", fid_err(b_vec_after))

    if short_state_c is not None and c_vec_after is not None:
        print("C: dense-rebuilt from A")
        print("  boundary shapes:", tuple(short_state_c.tensors[0].shape), tuple(short_state_c.tensors[-1].shape))
        print("  tensor_norm_delta:", c_delta_t)
        print("  statevec_norm_delta:", c_delta_v)
        print("  fid_err vs local Qiskit:", fid_err(c_vec_after))


def main() -> None:
    n = 10
    sites = (3, 8)
    theta = 0.25
    for window in ((3, 8), (2, 9)):
        run_case(n=n, sites=sites, theta=theta, window=window)


if __name__ == "__main__":
    main()

