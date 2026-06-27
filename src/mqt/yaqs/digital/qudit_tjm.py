# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Qudit Tensor Jump Method.

Native qudit (variable per-site dimension) counterpart to
:mod:`mqt.yaqs.digital.digital_tjm`. Drives the TJM loop using
:class:`~mqt.yaqs.digital.utils.qudit_dag_utils.QuditDAG` instead of Qiskit's ``DAGCircuit``,
and applies gates via ``node.gate.to_matrix()`` directly instead of
:class:`~mqt.yaqs.core.libraries.gate_library.BaseGate` (which rejects non-power-of-2
dimensions).

This module requires ``mqt-qudits`` (an optional dependency) and must never be imported
statically from a module that is part of the default ``mqt.yaqs`` import path; load it
dynamically (e.g. via :mod:`importlib`) instead.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import opt_einsum as oe

from ..core.data_structures.simulation_parameters import WeakSimParams
from ..core.methods.decompositions import merge_two_site, split_two_site
from ..core.methods.dissipation import apply_dissipation
from ..core.methods.stochastic_process import stochastic_process
from ..core.random_utils import make_trajectory_rng
from .digital_tjm import create_local_noise_model
from .utils.qudit_dag_utils import circuit_to_dag

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..core.data_structures.mps import MPS
    from ..core.data_structures.noise_model import NoiseModel
    from ..core.data_structures.simulation_parameters import StrongSimParams
    from .utils.qudit_dag_utils import QuditDAG, QuditOpNode


def process_layer_qudit(dag: QuditDAG) -> tuple[list[QuditOpNode], list[QuditOpNode], list[QuditOpNode]]:
    """Categorize the current front layer into single- and two-qudit gate groups.

    Mirrors :func:`mqt.yaqs.digital.digital_tjm.process_layer`'s even/odd grouping, which is
    pure site-index parity for non-overlapping nearest-neighbor sweep scheduling and is
    independent of physical dimension.

    Args:
        dag: Source qudit DAG.

    Returns:
        ``(single_qudit_nodes, even_nodes, odd_nodes)``, each sorted by lowest target qudit
        index to minimize orthogonality-center movement.

    Raises:
        NotImplementedError: If a node acts on more than two qudits.
    """
    current_layer = dag.front_layer()
    single_qudit_nodes = []
    even_nodes = []
    odd_nodes = []

    for node in current_layer:
        if len(node.target_qudits) == 1:
            single_qudit_nodes.append(node)
        elif len(node.target_qudits) == 2:
            if min(node.target_qudits) % 2 == 0:
                even_nodes.append(node)
            else:
                odd_nodes.append(node)
        else:
            raise NotImplementedError

    single_qudit_nodes.sort(key=lambda node: node.target_qudits[0])
    even_nodes.sort(key=lambda node: min(node.target_qudits))
    odd_nodes.sort(key=lambda node: min(node.target_qudits))

    return single_qudit_nodes, even_nodes, odd_nodes


def apply_single_qudit_gate(state: MPS, node: QuditOpNode) -> None:
    """Apply a single-qudit gate to the MPS in place.

    Args:
        state: MPS updated in place.
        node: DAG node for the single-qudit gate.
    """
    site = node.target_qudits[0]
    matrix = node.gate.to_matrix()
    state.tensors[site] = oe.contract("ab, bcd->acd", matrix, state.tensors[site])


def _apply_adjacent_two_qudit_gate(
    state: MPS,
    left_site: int,
    right_site: int,
    d_left: int,
    d_right: int,
    u: NDArray[np.complex128],
    sim_params: StrongSimParams | WeakSimParams,
) -> None:
    merged = merge_two_site(state.tensors[left_site], state.tensors[right_site])
    theta = u.reshape(d_left, d_right, d_left, d_right)
    theta_old = merged.reshape(d_left, d_right, merged.shape[1], merged.shape[2])
    theta_new = oe.contract("ijkl,klab->ijab", theta, theta_old)
    merged_new = theta_new.reshape(d_left * d_right, merged.shape[1], merged.shape[2])

    new_left, new_right = split_two_site(
        merged_new,
        [d_left, d_right],
        svd_distribution="right",
        trunc_mode=sim_params.trunc_mode,
        threshold=sim_params.svd_threshold,
        max_bond_dim=sim_params.max_bond_dim,
        min_keep=1,
    )
    state.tensors[left_site] = new_left
    state.tensors[right_site] = new_right


def _apply_swap(state: MPS, site: int, sim_params: StrongSimParams | WeakSimParams) -> None:

    d_left = state.tensors[site].shape[0]
    d_right = state.tensors[site + 1].shape[0]

    swap = np.zeros((d_right * d_left, d_left * d_right), dtype=np.complex128)
    for k in range(d_left):
        for m in range(d_right):
            swap[m * d_left + k, k * d_right + m] = 1.0
    swap = swap.reshape(d_right, d_left, d_left, d_right)

    merged = merge_two_site(state.tensors[site], state.tensors[site + 1])
    theta_old = merged.reshape(d_left, d_right, merged.shape[1], merged.shape[2])
    theta_new = oe.contract("ijkl, klab->ijab", swap, theta_old)
    merged_new = theta_new.reshape(d_right * d_left, merged.shape[1], merged.shape[2])

    new_left, new_right = split_two_site(
        merged_new,
        [d_right, d_left],
        svd_distribution="right",
        trunc_mode=sim_params.trunc_mode,
        threshold=sim_params.svd_threshold,
        max_bond_dim=sim_params.max_bond_dim,
        min_keep=1,
    )
    state.tensors[site] = new_left
    state.tensors[site + 1] = new_right


def apply_two_qudit_gate(state: MPS, node: QuditOpNode, sim_params: StrongSimParams | WeakSimParams) -> tuple[int, int]:
    """Apply a two-qudit gate, inserting a SWAP network for non-adjacent target qudits.

    Args:
        state: MPS updated in place.
        node: DAG node for the two-qudit gate.
        sim_params: Truncation settings forwarded to the merge/split steps.

    Returns:
        ``(first_site, last_site)`` spanning the gate support, for downstream local noise.
    """
    u, d_first, d_last, first_site, last_site = _gate_matrix_in_ascending_site_order(node)

    if last_site - first_site == 1:
        _apply_adjacent_two_qudit_gate(state, first_site, last_site, d_first, d_last, u, sim_params)
        return first_site, last_site

    for site in range(last_site - 1, first_site, -1):
        _apply_swap(state, site, sim_params)

    _apply_adjacent_two_qudit_gate(state, first_site, first_site + 1, d_first, d_last, u, sim_params)

    for site in range(first_site + 1, last_site):
        _apply_swap(state, site, sim_params)

    return first_site, last_site


def _gate_matrix_in_ascending_site_order(node: QuditOpNode) -> tuple[NDArray[np.complex128], int, int, int, int]:
    t0 = node.target_qudits[0]
    t1 = node.target_qudits[1]
    d0 = node.dimensions[0]
    d1 = node.dimensions[1]

    u = node.gate.to_matrix()
    if t0 < t1:
        return u, d0, d1, t0, t1
    u4 = u.reshape(d0, d1, d0, d1)
    u4 = np.transpose(u4, (1, 0, 3, 2))
    u_reordered = np.ascontiguousarray(u4).reshape(d1 * d0, d1 * d0)
    return u_reordered, d1, d0, t1, t0


def qudit_tjm(
    args: tuple[int, MPS, NoiseModel | None, StrongSimParams | WeakSimParams, object],
) -> tuple[NDArray[np.float64], None, MPS | None]:
    """Simulate a qudit circuit using the Tensor Jump Method (single trajectory).

    Mirrors :func:`mqt.yaqs.digital.digital_tjm.digital_tjm`'s control flow, driven by
    :class:`~mqt.yaqs.digital.utils.qudit_dag_utils.QuditDAG` instead of Qiskit's
    ``DAGCircuit``. MVP scope: no observable evaluation (see project issue), no
    diagnostics recording, single trajectory.

    Args:
        args: ``(traj_idx, initial_state, noise_model, sim_params, circuit)``. ``circuit`` is
            an ``mqt.qudits.quantum_circuit.QuantumCircuit``, untyped here since this module
            must not be referenced by ``ty``-checked callers without going through
            :mod:`importlib`.

    Returns:
        ``(None, None, final_state)``: the first two slots mirror ``digital_tjm``'s
        observable-results/diagnostics slots (both unused in the MVP); ``final_state`` is the
        final MPS if ``sim_params.get_state``, else ``None``.
    """
    traj_idx, initial_state, noise_model, sim_params, circuit = args
    if isinstance(sim_params, WeakSimParams):
        msg = "Shot-based qudit simulation (WeakSimParams) is not implemented yet"
        raise NotImplementedError(msg)

    state = copy.deepcopy(initial_state)
    dag = circuit_to_dag(circuit)
    rng = make_trajectory_rng(traj_idx, base_seed=sim_params.random_seed)

    while dag.op_nodes():
        single_nodes, even_nodes, odd_nodes = process_layer_qudit(dag)

        for node in single_nodes:
            apply_single_qudit_gate(state, node)
            dag.remove_op_node(node)

        for group in (even_nodes, odd_nodes):
            for node in group:
                first_site, last_site = apply_two_qudit_gate(state, node, sim_params)

                if noise_model is None or all(proc["strength"] == 0 for proc in noise_model.processes):
                    state.normalize(form="B", decomposition="QR")
                else:
                    local_noise_model = create_local_noise_model(noise_model, first_site, last_site)
                    apply_dissipation(state, local_noise_model, dt=1, sim_params=sim_params)
                    state = stochastic_process(state, local_noise_model, dt=1, sim_params=sim_params, rng=rng)

                dag.remove_op_node(node)

    results = np.zeros((len(sim_params.observables), 1))
    state.evaluate_observables(sim_params, results, column_index=0)
    final = state if sim_params.get_state else None
    return results, None, final
