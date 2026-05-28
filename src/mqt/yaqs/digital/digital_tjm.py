# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Digital Tensor Jump Method.

This module provides functions for simulating quantum circuits using the Tensor Jump Method (TJM). It includes
utilities for converting quantum circuits to DAG representations, processing gate layers, applying gates to
matrix product states (MPS) and constructing generator MPOs.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import opt_einsum as oe
from qiskit.converters import circuit_to_dag

from ..core.data_structures.mpo import MPO
from ..core.data_structures.mps import MPS
from ..core.data_structures.noise_model import NoiseModel
from ..core.data_structures.simulation_parameters import (
    StrongSimParams,
    WeakSimParams,
)
from ..core.libraries.gate_library import GateLibrary
from ..core.methods.decompositions import merge_two_site, split_two_site
from ..core.methods.dissipation import apply_dissipation
from ..core.methods.stochastic_process import stochastic_process
from ..core.methods.tdvp import (
    initialize_right_environments,
    merge_mpo_tensors,
    project_site,
    two_site_tdvp,
    update_left_environment,
    update_site,
)
from ..core.random_utils import make_trajectory_rng
from .utils.dag_utils import convert_dag_to_tensor_algorithm

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from qiskit.circuit import QuantumCircuit
    from qiskit.dagcircuit import DAGCircuit, DAGOpNode

    from ..core.data_structures.simulation_parameters import GateMode
    from ..core.libraries.gate_library import BaseGate
    from ..core.methods.decompositions import TruncMode


def create_local_noise_model(noise_model: NoiseModel, first_site: int, last_site: int) -> NoiseModel:
    """Create local noise model.

    Create a local noise model from a global noise model for a given gate.

    Args:
        noise_model (NoiseModel): The global noise model.
        first_site (int): The first site of the gate.
        last_site (int): The last site of the gate.

    Returns:
        NoiseModel: The local noise model.
    """
    affected_sites = [first_site, last_site]

    local_processes = [
        process
        for process in noise_model.processes
        if process["sites"] == affected_sites or process["sites"] == [first_site] or process["sites"] == [last_site]
    ]
    return NoiseModel(local_processes)


def process_layer(dag: DAGCircuit) -> tuple[list[DAGOpNode], list[DAGOpNode], list[DAGOpNode], list[DAGOpNode]]:
    """Process quantum circuit layer before applying to MPS.

    Processes the current layer of a DAGCircuit and categorizes nodes into single-qubit, even-indexed two-qubit,
    and odd-indexed two-qubit gates.

    Args:
        dag (DAGCircuit): The directed acyclic graph representing the quantum circuit.

    Returns:
        tuple[list[DAGOpNode], list[DAGOpNode], list[DAGOpNode], list[DAGOpNode]]: A tuple containing four lists:
            - single_qubit_nodes: Nodes corresponding to single-qubit gates.
            - even_nodes: Nodes corresponding to two-qubit gates where the lower qubit index is even.
            - odd_nodes: Nodes corresponding to two-qubit gates where the lower qubit index is odd.
            - measure_barriers: Labelled barriers ("SAMPLE_OBSERVABLES") used as sampling points.

    Raises:
        NotImplementedError: If a node with more than two qubits is encountered.
    """
    # Extract the current layer
    current_layer = dag.front_layer()

    # Prepare groups for even/odd two-qubit gates.
    single_qubit_nodes = []
    even_nodes = []
    odd_nodes = []
    measure_barriers = []

    # Separate the current layer into single-qubit and two-qubit gates.
    for node in current_layer:
        name = node.op.name

        # Drop measurements completely.
        if name == "measure":
            dag.remove_op_node(node)
            continue

        # Keep ONLY barriers with label "SAMPLE_OBSERVABLES" (case-insensitive). Remove all other barriers.
        if name == "barrier":
            label = getattr(node.op, "label", None)
            if label is not None and str(label).upper() == "SAMPLE_OBSERVABLES":
                measure_barriers.append(node)
            else:
                dag.remove_op_node(node)
            continue

        if len(node.qargs) == 1:
            single_qubit_nodes.append(node)
        elif len(node.qargs) == 2:
            # Group two-qubit gates by even/odd based on the lower qubit index.
            q0, q1 = node.qargs[0]._index, node.qargs[1]._index  # noqa: SLF001
            if min(q0, q1) % 2 == 0:
                even_nodes.append(node)
            else:
                odd_nodes.append(node)
        else:
            raise NotImplementedError

    # Sort the nodes to minimize orthogonality center movement (zig-zag optimization)
    single_qubit_nodes.sort(key=lambda node: node.qargs[0]._index)  # noqa: SLF001
    even_nodes.sort(key=lambda node: min(node.qargs[0]._index, node.qargs[1]._index))  # noqa: SLF001
    odd_nodes.sort(key=lambda node: min(node.qargs[0]._index, node.qargs[1]._index))  # noqa: SLF001

    return single_qubit_nodes, even_nodes, odd_nodes, measure_barriers


def apply_single_qubit_gate(state: MPS, node: DAGOpNode) -> None:
    """Apply single qubit gate.

    This function applies a single-qubit gate to the MPS, used during circuit simulation.

    Parameters:
    state (MPS): The matrix product state (MPS) representing the quantum state.
    node (DAGOpNode): The directed acyclic graph (DAG) operation node representing the gate to be applied.
    """
    gate = convert_dag_to_tensor_algorithm(node)[0]
    state.tensors[gate.sites[0]] = oe.contract("ab, bcd->acd", gate.tensor, state.tensors[gate.sites[0]])


def construct_generator_mpo(gate: BaseGate, length: int) -> tuple[MPO, int, int]:
    """Construct Generator MPO.

    Constructs a Matrix Product Operator (MPO) representation of a generator for a given gate over a
    specified length.

    Args:
        gate: The gate containing the generator and the sites it acts on.
        length: The total number of sites in the system.

    Returns:
        A tuple containing the constructed MPO, the first site index, and the last site index.
    """
    tensors = []

    if gate.sites[0] < gate.sites[1]:
        first_gen = 0
        second_gen = 1
    else:
        first_gen = 1
        second_gen = 0

    first_site = gate.sites[first_gen]
    last_site = gate.sites[second_gen]
    for site in range(length):
        if site == first_site:
            w = np.zeros((1, 1, 2, 2), dtype=complex)
            w[0, 0] = gate.generator[first_gen]
            tensors.append(w)
        elif site == last_site:
            w = np.zeros((1, 1, 2, 2), dtype=complex)
            w[0, 0] = gate.generator[second_gen]
            tensors.append(w)
        else:
            w = np.zeros((1, 1, 2, 2), dtype=complex)
            w[0, 0] = np.eye(2)
            tensors.append(w)

    mpo = MPO()
    mpo.custom(tensors)
    return mpo, first_site, last_site


def apply_window(state: MPS, mpo: MPO, first_site: int, last_site: int, window_size: int) -> tuple[MPS, MPO, list[int]]:
    """Apply Window.

    Apply a window to the given MPS and MPO for a local update.

    Args:
        state (MPS): The matrix product state (MPS) to be updated.
        mpo (MPO): The matrix product operator (MPO) to be applied.
        first_site (int): The index of the first site in the window.
        last_site (int): The index of the last site in the window.
        window_size: Number of sites on each side of first and last site

    Returns:
        tuple[MPS, MPO, list[int]]: A tuple containing the shortened MPS, the shortened MPO, and the window indices.
    """
    # Define a window for a local update.
    window = [first_site - window_size, last_site + window_size]
    window[0] = max(window[0], 0)
    window[1] = min(window[1], state.length - 1)

    # Shift the orthogonality center for sites before the window.
    for i in range(window[0]):
        state.shift_orthogonality_center_right(i)

    short_mpo = MPO()
    short_mpo.custom(mpo.tensors[window[0] : window[1] + 1], transpose=False)
    assert window[1] - window[0] + 1 > 1, "MPS cannot be length 1"
    short_state = MPS(length=window[1] - window[0] + 1, tensors=state.tensors[window[0] : window[1] + 1])

    return short_state, short_mpo, window


def _gate_tensor_left_right_order(
    gate: BaseGate,
    left_site: int,
    right_site: int,
) -> NDArray[np.complex128]:
    """Return ``gate.tensor`` with axes ordered as (left site, right site).

    Args:
        gate: Two-qubit gate with ``sites`` and ``tensor`` already set.
        left_site: Lower MPS site index.
        right_site: Higher MPS site index.

    Returns:
        Gate tensor ``U[out_l, out_r, in_l, in_r]`` for the left/right ordering.

    Raises:
        ValueError: If ``gate.sites`` does not match ``(left_site, right_site)``.
    """
    if gate.sites[0] == left_site and gate.sites[1] == right_site:
        return gate.tensor
    if gate.sites[0] == right_site and gate.sites[1] == left_site:
        return np.transpose(gate.tensor, (1, 0, 3, 2))
    msg = f"Gate sites {gate.sites!r} are not consistent with MPS sites ({left_site}, {right_site})."
    raise ValueError(msg)


@dataclass(frozen=True)
class TangentVisibility:
    r"""Visibility diagnostics for a local TDVP projector.

    Attributes:
        projected_norm: Estimate of \\(\\lVert P_T H_g |\\psi\rangle\rVert\\) (max over adjacent pairs).
        update_delta_norm: Optional estimate of the local TDVP update strength \\(\\lVert\\Delta A\rVert\\).
        generator_norm: Estimate of \\(\\lVert H_g |\\psi\rangle\rVert\\) for Pauli rotations.
        projected_ratio: ``projected_norm / generator_norm`` (clamped for safety in routing logic).
        update_delta_ratio: ``update_delta_norm / generator_norm`` if estimated.
        is_blind: Whether the projected generator ratio is below the configured blindness tolerance.
        is_weakly_visible: Whether the projected generator ratio is below the configured safety tolerance.
        recommended_route: Suggested update mechanism (``"tdvp"`` or ``"pauli_enriched"``).
        route_reason: Human-readable reason for the recommendation.
    """

    projected_norm: float
    update_delta_norm: float | None
    generator_norm: float
    projected_ratio: float
    update_delta_ratio: float | None
    is_blind: bool
    is_weakly_visible: bool
    recommended_route: str
    route_reason: str


@dataclass(frozen=True)
class PauliRouteDecision:
    """Routing decision for long-range Pauli rotations."""

    route: str  # "tdvp" or "pauli_enriched"
    reason: str
    visibility: TangentVisibility
    candidate_fidelity_error: float | None
    candidate_norm_error: float | None


def mps_overlap(left: MPS, right: MPS) -> complex:
    r"""Contract ``<left|right>`` exactly (no dense conversion).

    Args:
        left: Left MPS \\(|\\psi\rangle\\).
        right: Right MPS \\(|\\phi\rangle\\).

    Returns:
        The complex overlap \\(\\langle\\psi|\\phi\rangle\\).

    Raises:
        ValueError: If the MPS lengths do not match.
    """
    if left.length != right.length:
        msg = "MPS lengths must match for overlap."
        raise ValueError(msg)
    env = np.ones((1, 1), dtype=np.complex128)
    for n in range(left.length):
        a = np.asarray(left.tensors[n], dtype=np.complex128)  # (d, Dl, Dr)
        b = np.asarray(right.tensors[n], dtype=np.complex128)  # (d, Dl, Dr)
        env = np.einsum("pab,ac,pcd->bd", np.conjugate(a), env, b, optimize=True)
    return complex(env.reshape(()))


def mps_norm_squared(state: MPS) -> float:
    r"""Compute ``<state|state>`` exactly (no dense conversion).

    Args:
        state: MPS \\(|\\psi\rangle\\).

    Returns:
        The real scalar norm-squared \\(\\langle\\psi|\\psi\rangle\\).
    """
    return float(np.real(mps_overlap(state, state)))


def decide_long_range_pauli_route(
    state: MPS,
    gate: BaseGate,
    sim_params: StrongSimParams | WeakSimParams,
) -> PauliRouteDecision:
    """Decide routing for LR Pauli rotations using projection defect.

    Fast rule:
        projection_defect = 1 - min(projected_ratio, 1)
        route to TDVP if projection_defect <= tdvp_projection_defect_tol, else enrichment.

    ``tangent_blindness_tol`` is only used to label the extreme case where the
    projected generator is essentially zero.

    Args:
        state: Current MPS state.
        gate: Two-qubit long-range Pauli rotation gate.
        sim_params: Simulation parameters holding the routing tolerances.

    Returns:
        The routing decision and the underlying visibility diagnostics.
    """
    visibility = estimate_local_tdvp_projected_norm(
        state,
        gate,
        sim_params,
        window_size=1,
        estimate_update_delta=False,
    )
    blind_tol = float(getattr(sim_params, "tangent_blindness_tol", 1e-12))
    defect_tol = float(getattr(sim_params, "tdvp_projection_defect_tol", 1e-4))

    projected_ratio = float(visibility.projected_ratio)
    projection_defect = max(0.0, 1.0 - min(projected_ratio, 1.0))

    if projection_defect > defect_tol:
        if projected_ratio < blind_tol:
            reason = "tangent-blind: projected generator nearly zero"
        else:
            reason = (
                "projection-incomplete: projection defect "
                f"{projection_defect:.6g} above defect tol {defect_tol:.6g}"
            )
        return PauliRouteDecision(
            route="pauli_enriched",
            reason=reason,
            visibility=visibility,
            candidate_fidelity_error=None,
            candidate_norm_error=None,
        )
    return PauliRouteDecision(
        route="tdvp",
        reason=(
            "projection-complete: projection defect "
            f"{projection_defect:.6g} below defect tol {defect_tol:.6g}"
        ),
        visibility=visibility,
        candidate_fidelity_error=None,
        candidate_norm_error=None,
    )


def estimate_local_tdvp_projected_norm(
    state: MPS,
    gate: BaseGate,
    sim_params: StrongSimParams | WeakSimParams,
    *,
    window_size: int = 1,
    estimate_update_delta: bool = False,
) -> TangentVisibility:
    """Estimate whether the local 2TDVP projector can "see" the gate generator.

    Uses max-norm over adjacent-pair `project_site` values as a cheap diagnostic.
    Optionally also estimates the local update strength via `update_site` without mutating tensors.

    Returns:
        Visibility diagnostics for the current state and gate.
    """
    probe_state = copy.deepcopy(state)
    mpo, first_site, last_site = construct_generator_mpo(gate, probe_state.length)
    short_state, short_mpo, _window = apply_window(probe_state, mpo, first_site, last_site, window_size)

    # Environments as in TDVP sweeps.
    num_sites = short_mpo.length
    right_blocks = initialize_right_environments(short_state, short_mpo)

    left_blocks = [np.empty((0, 0, 0), dtype=np.complex128) for _ in range(num_sites)]
    left_virtual_dim = short_state.tensors[0].shape[1]
    mpo_left_dim = short_mpo.tensors[0].shape[2]
    left_identity = np.zeros((left_virtual_dim, mpo_left_dim, left_virtual_dim), dtype=right_blocks[0].dtype)
    for i in range(left_virtual_dim):
        for a in range(mpo_left_dim):
            left_identity[i, a, i] = 1
    left_blocks[0] = left_identity

    projected_max = 0.0
    update_delta_max_f = 0.0
    for i in range(num_sites - 1):
        merged_tensor = merge_two_site(short_state.tensors[i], short_state.tensors[i + 1])
        merged_mpo = merge_mpo_tensors(short_mpo.tensors[i], short_mpo.tensors[i + 1])
        proj = project_site(left_blocks[i], right_blocks[i + 1], merged_mpo, merged_tensor)
        projected_max = max(projected_max, float(np.linalg.norm(proj)))
        if estimate_update_delta:
            updated = update_site(
                left_blocks[i],
                right_blocks[i + 1],
                merged_mpo,
                merged_tensor,
                float(getattr(sim_params, "dt", 1.0)),
                krylov_tol=float(sim_params.krylov_tol),
            )
            update_delta_max_f = max(update_delta_max_f, float(np.linalg.norm(updated - merged_tensor)))
        if i + 1 < num_sites - 1:
            left_blocks[i + 1] = update_left_environment(
                short_state.tensors[i], short_state.tensors[i], short_mpo.tensors[i], left_blocks[i]
            )

    theta = float(getattr(gate, "theta", 0.0))
    generator_norm = abs(theta) / 2.0
    projected_ratio = projected_max / max(generator_norm, 1e-300)
    update_delta_max = update_delta_max_f if estimate_update_delta else None
    update_delta_ratio = None if update_delta_max is None else update_delta_max / max(generator_norm, 1e-300)

    blind_tol = float(getattr(sim_params, "tangent_blindness_tol", 1e-12))
    safety_tol = getattr(sim_params, "tdvp_visibility_safety_tol", None)
    safety_tol_f = None if safety_tol is None else float(safety_tol)

    is_blind = projected_ratio < blind_tol
    is_weakly_visible = (not is_blind) and (safety_tol_f is not None) and (projected_ratio < safety_tol_f)
    if is_blind:
        recommended_route = "pauli_enriched"
        route_reason = "tangent-blind: projected generator nearly zero"
    elif is_weakly_visible:
        recommended_route = "pauli_enriched"
        route_reason = "weakly visible: projected ratio below configured safety threshold"
    else:
        recommended_route = "tdvp"
        route_reason = "visible"
    return TangentVisibility(
        projected_norm=projected_max,
        update_delta_norm=update_delta_max,
        generator_norm=generator_norm,
        projected_ratio=projected_ratio,
        update_delta_ratio=update_delta_ratio,
        is_blind=is_blind,
        is_weakly_visible=is_weakly_visible,
        recommended_route=recommended_route,
        route_reason=route_reason,
    )


def add_mps_linear_combination(a: complex, state_a: MPS, b: complex, state_b: MPS) -> MPS:
    r"""Exact direct-sum MPS for ``a|A> + b|B>`` (no compression).

    Args:
        a: Scalar multiplier for ``state_a``.
        state_a: First MPS \\(|A\rangle\\).
        b: Scalar multiplier for ``state_b``.
        state_b: Second MPS \\(|B\rangle\\).

    Returns:
        An exact direct-sum MPS representing ``a|A> + b|B>``.

    Raises:
        ValueError: If the MPS lengths or physical dimensions do not match.
    """
    if state_a.length != state_b.length:
        msg = "A and B must have the same length."
        raise ValueError(msg)

    tensors: list[NDArray[np.complex128]] = []
    for n in range(state_a.length):
        a_t = np.asarray(state_a.tensors[n], dtype=np.complex128)
        b_t = np.asarray(state_b.tensors[n], dtype=np.complex128)
        d_a, l_a, r_a = a_t.shape
        d_b, l_b, r_b = b_t.shape
        if d_a != d_b:
            msg = "Physical dimensions must match."
            raise ValueError(msg)

        if n == 0:
            # Left boundary is assumed to be 1; concatenate along right bond.
            c_t = np.concatenate([a * a_t, b * b_t], axis=2)
        elif n == state_a.length - 1:
            # Right boundary is assumed to be 1; concatenate along left bond.
            c_t = np.concatenate([a_t, b_t], axis=1)
        else:
            c_t = np.zeros((d_a, l_a + l_b, r_a + r_b), dtype=np.complex128)
            c_t[:, :l_a, :r_a] = a_t
            c_t[:, l_a:, r_a:] = b_t
        tensors.append(c_t)

    return MPS(length=state_a.length, tensors=tensors)


def compress_mps_svd_sweep(state: MPS, sim_params: StrongSimParams | WeakSimParams) -> None:
    """In-place MPS compression via a single left-to-right SVD sweep."""
    if state.length <= 1:
        return
    # Bring into a stable canonical form before truncation.
    state.normalize(form="B", decomposition="QR")

    for i in range(state.length - 1):
        a = state.tensors[i]
        b = state.tensors[i + 1]
        merged = merge_two_site(a, b)
        a_new, b_new = split_two_site(
            merged,
            [a.shape[0], b.shape[0]],
            svd_distribution="right",
            trunc_mode=cast("TruncMode", sim_params.trunc_mode),
            threshold=sim_params.svd_threshold,
            max_bond_dim=sim_params.max_bond_dim,
            min_bond_dim=sim_params.min_bond_dim,
        )
        state.tensors[i], state.tensors[i + 1] = a_new, b_new

    state.normalize(form="B", decomposition="QR")


def apply_pauli_product_rotation_enriched(
    state: MPS,
    gate: BaseGate,
    sim_params: StrongSimParams | WeakSimParams,
    *,
    record_stats: bool = True,
) -> tuple[int, int]:
    """Apply ``rxx/ryy/rzz`` via exact Pauli-product generator enrichment.

    Uses the identity:
        exp(-i * theta * P_i P_j / 2) |psi>
        = cos(theta/2) |psi> - i sin(theta/2) P_i P_j |psi>

    Args:
        state: MPS updated in place.
        gate: Two-qubit Pauli rotation gate.
        sim_params: Truncation settings used for post-update compression.
        record_stats: If ``True``, updates diagnostic counters on ``sim_params``.

    Returns:
        ``(left_site, right_site)`` spanning the gate support in MPS order.

    Raises:
        ValueError: If ``gate.name`` is not a supported Pauli rotation.
    """
    if gate.name not in {"rxx", "ryy", "rzz"}:
        msg = f"Unsupported gate for enriched update: {gate.name!r}"
        raise ValueError(msg)

    site0, site1 = gate.sites
    left_site = min(site0, site1)
    right_site = max(site0, site1)
    theta = float(getattr(gate, "theta", 0.0))

    if gate.name == "rxx":
        pauli = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    elif gate.name == "ryy":
        pauli = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    else:  # rzz
        pauli = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    branch = copy.deepcopy(state)
    branch.tensors[site0] = oe.contract("ab, bcd->acd", pauli, branch.tensors[site0])
    branch.tensors[site1] = oe.contract("ab, bcd->acd", pauli, branch.tensors[site1])

    a = complex(np.cos(theta / 2.0))
    b = complex(-1j * np.sin(theta / 2.0))
    combined = add_mps_linear_combination(a, state, b, branch)
    if record_stats:
        # Metadata counters for diagnostics/direct runs.
        sim_params.enriched_pauli_count = int(getattr(sim_params, "enriched_pauli_count", 0)) + 1
    # Compress to avoid direct-sum bond blowup.
    compress_mps_svd_sweep(combined, sim_params)

    state.tensors = combined.tensors
    return left_site, right_site


def apply_two_qubit_gate_tdvp(
    state: MPS,
    gate: BaseGate,
    sim_params: StrongSimParams | WeakSimParams,
) -> tuple[int, int]:
    """Apply a two-qubit gate via generator MPO and two-site TDVP.

    Args:
        state: MPS updated in place.
        gate: Internal gate object from the gate library.
        sim_params: Truncation and Krylov settings for TDVP.

    Returns:
        ``(first_site, last_site)`` spanning the gate support in MPS order.
    """
    mpo, first_site, last_site = construct_generator_mpo(gate, state.length)

    window_size = 1
    short_state, short_mpo, window = apply_window(state, mpo, first_site, last_site, window_size)
    two_site_tdvp(short_state, short_mpo, sim_params)
    for i in range(window[0], window[1] + 1):
        state.tensors[i] = short_state.tensors[i - window[0]]

    return first_site, last_site


def apply_two_qubit_gate_tebd(
    state: MPS,
    gate: BaseGate,
    sim_params: StrongSimParams | WeakSimParams,
) -> tuple[int, int]:
    """Apply a two-qubit gate via TEBD/SVD, inserting adjacent SWAPs if needed.

    Args:
        state: MPS updated in place.
        gate: Internal gate object from the gate library.
        sim_params: Truncation settings shared with TDVP/MPS splitting.

    Returns:
        ``(left_site, right_site)`` spanning the gate support in MPS order.
    """

    def apply_swap(site_left: int) -> None:
        swap_gate = GateLibrary.swap()
        swap_gate.set_sites(site_left, site_left + 1)
        apply_two_qubit_gate_tebd(state, swap_gate, sim_params)

    site0, site1 = gate.sites[0], gate.sites[1]
    if abs(site0 - site1) != 1:
        left = min(site0, site1)
        right = max(site0, site1)

        for i in range(right - 1, left, -1):
            apply_swap(i)

        gate_adj = copy.deepcopy(gate)
        if site0 == left:
            gate_adj.set_sites(left, left + 1)
        else:
            gate_adj.set_sites(left + 1, left)
        apply_two_qubit_gate_tebd(state, gate_adj, sim_params)

        for i in range(left + 1, right):
            apply_swap(i)

        return left, right

    left_site = min(site0, site1)
    right_site = max(site0, site1)
    u_gate = _gate_tensor_left_right_order(gate, left_site, right_site)

    left_tensor = state.tensors[left_site]
    right_tensor = state.tensors[right_site]
    d_left, d_right = left_tensor.shape[0], right_tensor.shape[0]

    merged = merge_two_site(left_tensor, right_tensor)
    theta_4 = merged.reshape(d_left, d_right, merged.shape[1], merged.shape[2])
    theta_new = np.asarray(oe.contract("ijkl,klab->ijab", u_gate, theta_4), dtype=np.complex128)
    merged_new = theta_new.reshape(d_left * d_right, merged.shape[1], merged.shape[2])

    new_left, new_right = split_two_site(
        merged_new,
        [d_left, d_right],
        svd_distribution="right",
        trunc_mode=cast("TruncMode", sim_params.trunc_mode),
        threshold=sim_params.svd_threshold,
        max_bond_dim=sim_params.max_bond_dim,
        min_bond_dim=sim_params.min_bond_dim,
    )
    state.tensors[left_site] = new_left
    state.tensors[right_site] = new_right
    return left_site, right_site


def apply_two_qubit_gate(state: MPS, node: DAGOpNode, sim_params: StrongSimParams | WeakSimParams) -> tuple[int, int]:
    """Apply a two-qubit gate using the configured two-qubit gate mode.

    Args:
        state: MPS updated in place.
        node: DAG node for the two-qubit gate.
        sim_params: Simulation parameters including ``gate_mode``.

    Returns:
        ``(first_site, last_site)`` for downstream local noise handling.

    Raises:
        ValueError: If ``gate_mode`` is unknown.
    """
    gate = convert_dag_to_tensor_algorithm(node)[0]
    site0, site1 = gate.sites[0], gate.sites[1]
    is_nearest_neighbor = abs(site0 - site1) == 1
    gate_mode: GateMode = getattr(sim_params, "gate_mode", "hybrid")

    if gate_mode == "tdvp":
        sim_params.tdvp_lr_count = int(getattr(sim_params, "tdvp_lr_count", 0)) + 1
        return apply_two_qubit_gate_tdvp(state, gate, sim_params)

    if gate_mode == "tebd":
        sim_params.tebd_nn_count = int(getattr(sim_params, "tebd_nn_count", 0)) + 1
        return apply_two_qubit_gate_tebd(state, gate, sim_params)

    if gate_mode == "hybrid":
        if is_nearest_neighbor:
            sim_params.tebd_nn_count = int(getattr(sim_params, "tebd_nn_count", 0)) + 1
            return apply_two_qubit_gate_tebd(state, gate, sim_params)
        if gate.name in {"rxx", "ryy", "rzz"}:
            # Route accounting attached to the returned MPS (Simulator deep-copies sim_params).
            stats_any = getattr(state, "route_stats", None)
            if not isinstance(stats_any, dict):
                stats_any = {"tdvp_lr_pauli": 0, "enriched_lr_pauli": 0, "ratios": []}
                state.route_stats = stats_any
            stats = cast("dict[str, object]", stats_any)

            decision = decide_long_range_pauli_route(state, gate, sim_params)
            sim_params.last_lr_visibility = decision.visibility
            sim_params.last_lr_route = decision.route
            sim_params.last_lr_decision = decision
            defect_tol = float(getattr(sim_params, "tdvp_projection_defect_tol", 1e-3))
            projection_defect = max(0.0, 1.0 - min(float(decision.visibility.projected_ratio), 1.0))
            ratios_any = stats.get("ratios")
            if not isinstance(ratios_any, list):
                ratios_any = []
                stats["ratios"] = ratios_any
            ratios = cast("list[dict[str, object]]", ratios_any)
            ratios.append({
                "gate": gate.name,
                "sites": tuple(gate.sites),
                "projected_ratio": decision.visibility.projected_ratio,
                "projected_norm": decision.visibility.projected_norm,
                "generator_norm": decision.visibility.generator_norm,
                "projection_defect": projection_defect,
                "tdvp_projection_defect_tol": defect_tol,
                "update_delta_ratio": decision.visibility.update_delta_ratio,
                "candidate_fidelity_error": decision.candidate_fidelity_error,
                "candidate_norm_error": decision.candidate_norm_error,
                "route": decision.route,
                "reason": decision.reason,
            })

            if decision.route == "pauli_enriched":
                stats["enriched_lr_pauli"] = int(stats.get("enriched_lr_pauli", 0)) + 1
                return apply_pauli_product_rotation_enriched(state, gate, sim_params)
            stats["tdvp_lr_pauli"] = int(stats.get("tdvp_lr_pauli", 0)) + 1
            sim_params.tdvp_lr_count = int(getattr(sim_params, "tdvp_lr_count", 0)) + 1
        else:
            sim_params.tdvp_lr_count = int(getattr(sim_params, "tdvp_lr_count", 0)) + 1
        return apply_two_qubit_gate_tdvp(state, gate, sim_params)

    msg = f"Unknown gate_mode: {gate_mode!r}"
    raise ValueError(msg)


def digital_tjm(
    args: tuple[int, MPS, NoiseModel | None, StrongSimParams | WeakSimParams, QuantumCircuit],
) -> tuple[NDArray[np.float64] | dict[int, int], NDArray[np.float64] | None, MPS | None]:
    """Digital Tensor Jump Method.

    Simulates a quantum circuit using the Tensor Jump Method.

    Args:
        args: A tuple containing the following elements:
            - An index or identifier, primarily for parallelization
            - The initial state of the system represented as a Matrix Product MPS.
            - The noise model to be applied during the simulation, or None if no noise is to be applied.
            - Parameters for the simulation, either for strong or weak simulation.
            - The quantum circuit to be simulated.

    Returns:
        The results of the simulation.
        If StrongSimParams are used, the results are the measured observables.
        If WeakSimParams are used, the results are the measurement outcomes for each shot.
    """
    traj_idx, initial_state, noise_model, sim_params, circuit = args

    state = copy.deepcopy(initial_state)
    dag = circuit_to_dag(circuit)
    diagnostics: NDArray[np.float64] | None = None

    # Initialize results depending on simulation type
    if isinstance(sim_params, StrongSimParams):
        num_cols = (sim_params.num_mid_measurements + 2) if sim_params.sample_layers else 1
        diagnostics = np.zeros((3, num_cols), dtype=np.float64)
        if sim_params.sample_layers:
            results = np.zeros((len(sim_params.sorted_observables), sim_params.num_mid_measurements + 2))
            state.record_diagnostics(diagnostics, 0)
            state.evaluate_observables(sim_params, results, 0)
        else:
            results = np.zeros((len(sim_params.sorted_observables), 1))

    rng = make_trajectory_rng(traj_idx, base_seed=sim_params.random_seed)

    col_idx = 0
    canonical_form_lost = False
    while dag.op_nodes():
        single_qubit_nodes, even_nodes, odd_nodes, measure_barriers = process_layer(dag)

        for node in single_qubit_nodes:
            apply_single_qubit_gate(state, node)
            dag.remove_op_node(node)
            if not dag.op_nodes():
                canonical_form_lost = True

        # Process two-qubit gates in even/odd sweeps.
        for _, group in [("even", even_nodes), ("odd", odd_nodes)]:
            for node in group:
                first_site, last_site = apply_two_qubit_gate(state, node, sim_params)

                if noise_model is None or all(proc["strength"] == 0 for proc in noise_model.processes):
                    # Normalizes state
                    state.normalize(form="B", decomposition="QR")
                else:
                    local_noise_model = create_local_noise_model(noise_model, first_site, last_site)
                    apply_dissipation(state, local_noise_model, dt=1, sim_params=sim_params)
                    state = stochastic_process(state, local_noise_model, dt=1, sim_params=sim_params, rng=rng)

                dag.remove_op_node(node)

        # Process measurement barriers (only when sampling layers in strong sim)
        if isinstance(sim_params, StrongSimParams) and sim_params.sample_layers:
            for measure_barrier in measure_barriers:
                dag.remove_op_node(measure_barrier)
                col_idx += 1
                assert diagnostics is not None
                state.record_diagnostics(diagnostics, col_idx)
                state.evaluate_observables(sim_params, results, col_idx)

    if isinstance(sim_params, WeakSimParams):
        per_call_shots = _per_call_shots(sim_params)
        if not noise_model or all(proc["strength"] == 0 for proc in noise_model.processes):
            counts = state.measure_shots(per_call_shots)
            final = state if sim_params.get_state else None
            return counts, None, final
        return state.measure_shots(shots=1), None, state if sim_params.get_state else None

    if canonical_form_lost:
        state.normalize(form="B", decomposition="QR")

    assert isinstance(sim_params, StrongSimParams)
    assert diagnostics is not None
    final_col = results.shape[1] - 1
    state.record_diagnostics(diagnostics, final_col)
    state.evaluate_observables(sim_params, results, final_col)
    final = state if sim_params.get_state else None
    return results, diagnostics, final


def _per_call_shots(sim_params: WeakSimParams) -> int:
    """Return shots for this worker call (may differ from ``sim_params.shots`` when noisy).

    Returns:
        Number of shots for the current worker invocation.
    """
    try:
        from mqt.yaqs.simulator import WORKER_CTX  # noqa: PLC0415

        return int(WORKER_CTX["per_call_shots"])
    except KeyError:
        return sim_params.shots
