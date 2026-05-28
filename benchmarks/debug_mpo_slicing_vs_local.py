#!/usr/bin/env python
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Debug whether slicing a full generator MPO yields a valid standalone window MPO.

Run:

    uv run python -m benchmarks.debug_mpo_slicing_vs_local
"""

from __future__ import annotations

import copy

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.digital.digital_tjm import construct_generator_mpo
from mqt.yaqs.digital.utils.dag_utils import convert_dag_to_tensor_algorithm


def _gate_from_qiskit(qc: QuantumCircuit):
    dag = circuit_to_dag(qc)
    nodes = list(dag.topological_op_nodes())
    if len(nodes) != 1:
        raise ValueError("Expected exactly one gate.")
    return convert_dag_to_tensor_algorithm(nodes[0])[0]


def _mpo_from_tensors(tensors) -> MPO:
    mpo = MPO()
    mpo.custom(list(tensors), transpose=False)
    return mpo


def compare(gate_name: str, theta: float, n: int, sites: tuple[int, int], window: tuple[int, int]) -> None:
    qc = QuantumCircuit(n)
    getattr(qc, gate_name)(theta, sites[0], sites[1])
    gate = _gate_from_qiskit(qc)

    full_mpo, *_ = construct_generator_mpo(gate, n)
    sliced = _mpo_from_tensors(full_mpo.tensors[window[0] : window[1] + 1])

    # Build local MPO directly on the window length.
    local_len = window[1] - window[0] + 1
    local_gate = copy.deepcopy(gate)
    local_gate.set_sites(sites[0] - window[0], sites[1] - window[0])
    local_mpo, *_ = construct_generator_mpo(local_gate, local_len)

    a = sliced.to_matrix()
    b = local_mpo.to_matrix()

    print(f"\n=== {gate_name} theta={theta} sites={sites} window={window} (len={local_len}) ===")
    print("sliced tensor shapes:", [tuple(t.shape) for t in sliced.tensors])
    print("local  tensor shapes:", [tuple(t.shape) for t in local_mpo.tensors])
    print("sliced boundary dims:", sliced.tensors[0].shape[2:], sliced.tensors[-1].shape[2:])
    print("local  boundary dims:", local_mpo.tensors[0].shape[2:], local_mpo.tensors[-1].shape[2:])
    print("||dense(sliced)||:", float(np.linalg.norm(a)))
    print("||dense(local )||:", float(np.linalg.norm(b)))
    print("||dense diff||  :", float(np.linalg.norm(a - b)))

    # Action on |0...0>
    psi0 = np.zeros((2**local_len,), dtype=np.complex128)
    psi0[0] = 1.0
    v_s = a @ psi0
    v_l = b @ psi0
    print("||H_sliced |0>||:", float(np.linalg.norm(v_s)))
    print("||H_local  |0>||:", float(np.linalg.norm(v_l)))
    print("||delta vec||   :", float(np.linalg.norm(v_s - v_l)))


def main() -> None:
    n = 10
    sites = (3, 8)

    # Window variants.
    for window in ((2, 9), (3, 8)):
        for gate_name in ("rxx", "ryy", "rzz"):
            compare(gate_name, 0.25, n, sites, window)


if __name__ == "__main__":
    main()

