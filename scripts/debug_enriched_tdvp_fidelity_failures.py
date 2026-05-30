#!/usr/bin/env python
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Diagnose enriched/projection-defect TDVP fidelity failures vs Qiskit.

Run:

    uv run python -m scripts.debug_enriched_tdvp_fidelity_failures

Outputs:
    results/enriched_tdvp_fidelity_failure_debug.csv
    results/enriched_tdvp_fidelity_failure_debug.md
"""

from __future__ import annotations

import copy
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from qiskit.circuit import CircuitInstruction, QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.quantum_info import Statevector

from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.methods.tdvp import two_site_tdvp
from mqt.yaqs.digital.digital_tjm import (
    add_mps_linear_combination,
    apply_pauli_product_rotation_enriched,
    apply_single_qubit_gate,
    apply_two_qubit_gate_tebd,
    apply_two_qubit_gate_tdvp,
    apply_window,
    compress_mps_svd_sweep,
    construct_generator_mpo,
    decide_long_range_pauli_route,
    estimate_local_tdvp_projected_norm,
    make_pauli_product_branch,
    mps_overlap,
)
from mqt.yaqs.digital.utils.dag_utils import convert_dag_to_tensor_algorithm

from scripts.benchmark_enriched_tdvp_vs_tebd import (
    Case,
    _apply_two_qubit_enriched_tdvp,
    _count_circuit_gates,
    _fid_err_vec,
    _ising_2d_row_major,
    _mean_bond_dim,
    _pauli_obs_errors,
    _prep_initial_state,
    _qiskit_vec,
    _route_counts,
)

MethodName = Literal[
    "qiskit_reference",
    "tebd_swaps",
    "current_enriched_tdvp",
    "force_all_lr_pauli_enrichment",
    "force_all_lr_pauli_tdvp",
]
LrPauliRoute = Literal["current_router", "force_enrichment", "force_tdvp", "all_enrichment"]
RunOrder = Literal["circuit_data", "dag_topo"]

TARGET_CIRCUIT_NAMES: tuple[str, ...] = (
    "ising2d_3x3_h1_dt0.1_L4_plus",
    "ising2d_3x3_h0.5_dt0.1_L4_plus",
    "ising2d_3x3_h1_dt0.1_L4_random_product",
    "ising2d_3x3_h2_dt0.1_L2_random_product",
    "ising2d_3x3_h0.5_dt0.1_L2_all_zero",
)

DEFAULT_DEFECT_TOL = 1e-4
DEFAULT_SVD_THRESHOLD = 1e-12


@dataclass(frozen=True)
class GateStep:
    """One non-barrier gate in explicit ``QuantumCircuit.data`` or DAG order."""

    prefix_index: int
    node: DAGOpNode
    instruction: CircuitInstruction | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _results_dir() -> Path:
    d = _repo_root() / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _parse_ising2d_name(name: str) -> tuple[int, int, float, float, int, str]:
    m = re.fullmatch(
        r"ising2d_(\d+)x(\d+)_h([\d.]+)_dt([\d.]+)_L(\d+)_(\w+)",
        name,
    )
    if m is None:
        msg = f"Cannot parse circuit name: {name!r}"
        raise ValueError(msg)
    lx, ly, h_s, dt_s, layers_s, init = m.groups()
    return int(lx), int(ly), float(h_s), float(dt_s), int(layers_s), init


def _build_case(circuit_name: str) -> Case:
    lx, ly, h, dt, layers, init = _parse_ising2d_name(circuit_name)
    j = 1.0
    n = lx * ly
    qc_body, edge_types = _ising_2d_row_major(lx=lx, ly=ly, j=j, h=h, dt=dt, layers=layers, periodic_x=True)
    prep = QuantumCircuit(n)
    _prep_initial_state(prep, init, seed=0)  # type: ignore[arg-type]
    qc = prep.compose(qc_body)
    return Case(
        family="ising_2d",
        model="ising",
        geometry="2d",
        boundary_condition="row_major_periodic_x",
        lx=lx,
        ly=ly,
        n_qubits=n,
        initial_state=init,
        dt=dt,
        layers_or_depth=layers,
        circuit_name=circuit_name,
        qc=qc,
        edge_types=edge_types,
    )


def _make_params(
    *,
    svd_threshold: float = DEFAULT_SVD_THRESHOLD,
    max_bond_dim: int | None = None,
    defect_tol: float = DEFAULT_DEFECT_TOL,
    compress_after_enrichment: bool = True,
) -> StrongSimParams:
    params = StrongSimParams(
        preset="exact",
        get_state=True,
        svd_threshold=svd_threshold,
        max_bond_dim=max_bond_dim,
        krylov_tol=1e-12,
        gate_mode="hybrid",
        tangent_blindness_tol=1e-12,
        tdvp_projection_defect_tol=float(defect_tol),
        tdvp_visibility_safety_tol=None,
        tdvp_pauli_consistency_check=False,
        tdvp_pauli_consistency_tol=1e-10,
    )
    params.compress_after_enrichment = compress_after_enrichment  # type: ignore[attr-defined]
    return params


def _instruction_to_node(qc: QuantumCircuit, ci: CircuitInstruction) -> DAGOpNode:
    mini = QuantumCircuit(qc.num_qubits)
    qubits = list(getattr(ci, "qubits", None) or getattr(ci, "qargs", ()))
    clbits = list(getattr(ci, "clbits", ()) or ())
    mini.append(ci.operation, qubits, clbits)
    dag = circuit_to_dag(mini)
    return next(dag.topological_op_nodes())


def iter_gate_steps(qc: QuantumCircuit, *, order: RunOrder = "circuit_data") -> list[GateStep]:
    if order == "circuit_data":
        steps: list[GateStep] = []
        idx = 0
        for ci in qc.data:
            if ci.operation.name == "barrier":
                continue
            steps.append(GateStep(prefix_index=idx, node=_instruction_to_node(qc, ci), instruction=ci))
            idx += 1
        return steps

    dag = circuit_to_dag(qc)
    steps = []
    idx = 0
    for node in dag.topological_op_nodes():
        if node.op.name == "barrier":
            continue
        steps.append(GateStep(prefix_index=idx, node=node, instruction=None))
        idx += 1
    return steps


def build_prefix_circuit(qc: QuantumCircuit, num_gates: int, *, order: RunOrder = "circuit_data") -> QuantumCircuit:
    """First ``num_gates`` non-barrier gates in ``circuit_data`` or DAG topological order."""
    sub = QuantumCircuit(qc.num_qubits)
    if order == "circuit_data":
        count = 0
        for ci in qc.data:
            if ci.operation.name == "barrier":
                continue
            if count >= num_gates:
                break
            qubits = list(getattr(ci, "qubits", None) or getattr(ci, "qargs", ()))
            clbits = list(getattr(ci, "clbits", ()) or ())
            sub.append(ci.operation, qubits, clbits)
            count += 1
        return sub

    for step in iter_gate_steps(qc, order="dag_topo")[:num_gates]:
        sub.append(step.node.op, list(step.node.qargs), [])
    return sub


def _phase_aligned_l2_error(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 1.0
    denom = float(np.vdot(b, b).real)
    if denom < 1e-300:
        return float(np.linalg.norm(a))
    alpha = np.vdot(a, b) / denom
    return float(np.linalg.norm(a - alpha * b))


def _statevector_norm_error(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-300 or nb < 1e-300:
        return 1.0
    return float(np.linalg.norm(a / na - b / nb))


def _metrics_vs_ref(mps: MPS, ref_vec: np.ndarray, case: Case) -> dict[str, float | None]:
    vec = np.asarray(mps.to_vec(), dtype=np.complex128)
    obs_max, obs_mean, obs_l2, _worst = _pauli_obs_errors(case, mps=mps, ref_vec=ref_vec, ref_mps=None)
    return {
        "fidelity_error": _fid_err_vec(ref_vec, vec),
        "statevector_norm_error": _statevector_norm_error(vec, ref_vec),
        "phase_aligned_l2_error": _phase_aligned_l2_error(vec, ref_vec),
        "pauli_obs_max_error": obs_max,
        "max_bond": float(mps.get_max_bond()),
        "mean_bond": _mean_bond_dim(mps),
    }


def _init_mps(n: int) -> MPS:
    return State(n, initial="zeros", representation="mps").mps


def _reset_route_stats(mps: MPS) -> None:
    mps.route_stats = {"tdvp_lr_pauli": 0, "enriched_lr_pauli": 0, "ratios": []}  # type: ignore[attr-defined]


def _lr_route_decision(
    mps: MPS,
    gate,
    params: StrongSimParams,
    lr_mode: LrPauliRoute,
) -> tuple[str, str, float, float]:
    if lr_mode in {"force_enrichment", "all_enrichment"}:
        vis = estimate_local_tdvp_projected_norm(mps, gate, params, window_size=1, estimate_update_delta=False)
        pr = float(vis.projected_ratio)
        defect = max(0.0, 1.0 - min(pr, 1.0))
        return "pauli_enriched", f"forced: {lr_mode}", pr, defect
    if lr_mode == "force_tdvp":
        vis = estimate_local_tdvp_projected_norm(mps, gate, params, window_size=1, estimate_update_delta=False)
        pr = float(vis.projected_ratio)
        defect = max(0.0, 1.0 - min(pr, 1.0))
        return "tdvp", "forced: force_tdvp", pr, defect
    decision = decide_long_range_pauli_route(mps, gate, params)
    pr = float(decision.visibility.projected_ratio)
    defect = max(0.0, 1.0 - min(pr, 1.0))
    return decision.route, decision.reason, pr, defect


def apply_pauli_enriched_optional_compress(
    state: MPS,
    gate,
    sim_params: StrongSimParams,
    *,
    compress: bool,
) -> None:
    """Exact Pauli-product enrichment with optional post-update compression."""
    if gate.name not in {"rxx", "ryy", "rzz"}:
        msg = f"Unsupported gate: {gate.name!r}"
        raise ValueError(msg)

    site0, site1 = gate.sites
    theta = float(getattr(gate, "theta", 0.0))
    if gate.name == "rxx":
        pauli = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    elif gate.name == "ryy":
        pauli = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    else:
        pauli = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    branch = make_pauli_product_branch(state, gate)
    a = complex(np.cos(theta / 2.0))
    b = complex(-1j * np.sin(theta / 2.0))
    combined = add_mps_linear_combination(a, state, b, branch)
    if compress:
        compress_mps_svd_sweep(combined, sim_params)
    state.tensors = combined.tensors


def apply_two_qubit_gate_tdvp_window(
    state: MPS,
    gate,
    sim_params: StrongSimParams,
    *,
    window_size: int,
    num_sweeps: int = 1,
) -> tuple[int, int]:
    """TDVP with configurable window and sweep count (diagnostic only)."""
    mpo, first_site, last_site = construct_generator_mpo(gate, state.length)
    short_state, short_mpo, window = apply_window(state, mpo, first_site, last_site, window_size)
    for _ in range(max(1, int(num_sweeps))):
        two_site_tdvp(short_state, short_mpo, sim_params)
    for i in range(window[0], window[1] + 1):
        state.tensors[i] = short_state.tensors[i - window[0]]
    return first_site, last_site


def _apply_two_qubit(
    mps: MPS,
    node: DAGOpNode,
    params: StrongSimParams,
    *,
    method: MethodName | None = None,
    lr_mode: LrPauliRoute = "current_router",
) -> tuple[str | None, str | None, float | None, float | None]:
    """Apply one 2q gate; return route metadata for LR Pauli gates."""
    gate = convert_dag_to_tensor_algorithm(node)[0]
    i, j = int(gate.sites[0]), int(gate.sites[1])
    is_lr = abs(i - j) != 1
    route_used: str | None = None
    route_reason: str | None = None
    projected_ratio: float | None = None
    projection_defect: float | None = None

    if len(node.qargs) == 1:
        apply_single_qubit_gate(mps, node)
        return route_used, route_reason, projected_ratio, projection_defect

    if not is_lr:
        apply_two_qubit_gate_tebd(mps, gate, params)
        return route_used, route_reason, projected_ratio, projection_defect

    if gate.name not in {"rxx", "ryy", "rzz"}:
        apply_two_qubit_gate_tdvp(mps, gate, params)
        return route_used, route_reason, projected_ratio, projection_defect

    if method == "tebd_swaps":
        apply_two_qubit_gate_tebd(mps, gate, params)
        return route_used, route_reason, projected_ratio, projection_defect

    if method == "force_all_lr_pauli_tdvp":
        route_used, route_reason = "tdvp", "method: force_all_lr_pauli_tdvp"
        vis = estimate_local_tdvp_projected_norm(mps, gate, params, window_size=1, estimate_update_delta=False)
        projected_ratio = float(vis.projected_ratio)
        projection_defect = max(0.0, 1.0 - min(projected_ratio, 1.0))
        apply_two_qubit_gate_tdvp(mps, gate, params)
        return route_used, route_reason, projected_ratio, projection_defect

    if method in {"force_all_lr_pauli_enrichment", "current_enriched_tdvp"}:
        if method == "force_all_lr_pauli_enrichment":
            lr_mode = "force_enrichment"
        route_used, route_reason, projected_ratio, projection_defect = _lr_route_decision(mps, gate, params, lr_mode)
        compress = bool(getattr(params, "compress_after_enrichment", True))
        if route_used == "pauli_enriched":
            apply_pauli_enriched_optional_compress(mps, gate, params, compress=compress)
        else:
            apply_two_qubit_gate_tdvp(mps, gate, params)
        return route_used, route_reason, projected_ratio, projection_defect

    # current_enriched_tdvp via benchmark helper (DAG router + stats)
    if method == "current_enriched_tdvp":
        _apply_two_qubit_enriched_tdvp(mps, node, params)
        stats = getattr(mps, "route_stats", None)
        if isinstance(stats, dict) and stats.get("ratios"):
            last = stats["ratios"][-1]
            route_used = str(last.get("route"))
            route_reason = str(last.get("reason"))
            projected_ratio = float(last.get("projected_ratio"))
            projection_defect = float(last.get("projection_defect"))
        return route_used, route_reason, projected_ratio, projection_defect

    raise AssertionError(f"Unhandled method {method!r}")


def _run_circuit(
    case: Case,
    *,
    method: MethodName,
    params: StrongSimParams,
    order: RunOrder = "dag_topo",
) -> tuple[MPS, float]:
    mps = _init_mps(case.n_qubits)
    if method in {"current_enriched_tdvp", "force_all_lr_pauli_enrichment", "force_all_lr_pauli_tdvp"}:
        _reset_route_stats(mps)

    steps = iter_gate_steps(case.qc, order=order)
    t0 = time.perf_counter()
    for step in steps:
        node = step.node
        if len(node.qargs) == 1:
            apply_single_qubit_gate(mps, node)
            continue
        if method == "qiskit_reference":
            raise AssertionError("qiskit_reference is not an MPS method")
        _apply_two_qubit(mps, node, params, method=method)
    return mps, float(time.perf_counter() - t0)


def debug_gate_prefix_divergence(
    qc: QuantumCircuit,
    method: MethodName,
    sim_params: StrongSimParams,
    *,
    case: Case | None = None,
    max_prefix: int | None = None,
    lr_mode: LrPauliRoute = "current_router",
    order: RunOrder = "dag_topo",
) -> list[dict[str, Any]]:
    """Step through gates; compare each prefix to Qiskit (same order as simulator)."""
    if case is None:
        case = _build_case("ising2d_3x3_h0.5_dt0.1_L2_all_zero")  # placeholder; caller should pass case

    ref_full = _qiskit_vec(qc) if qc.num_qubits <= 14 else None
    if ref_full is None:
        return []

    steps = iter_gate_steps(qc, order=order)
    if max_prefix is not None:
        steps = steps[: max_prefix]

    mps = _init_mps(qc.num_qubits)
    if method == "current_enriched_tdvp":
        _reset_route_stats(mps)

    rows: list[dict[str, Any]] = []
    for k, step in enumerate(steps, start=1):
        node = step.node
        gate = convert_dag_to_tensor_algorithm(node)[0]
        is_1q = len(node.qargs) == 1
        if is_1q:
            is_nn = True
            is_lr = False
        else:
            i, j = int(gate.sites[0]), int(gate.sites[1])
            is_nn = abs(i - j) == 1
            is_lr = not is_nn
        route_used = ""
        route_reason = ""
        projected_ratio = float("nan")
        projection_defect = float("nan")

        if len(node.qargs) == 1:
            apply_single_qubit_gate(mps, node)
        else:
            if method == "tebd_swaps":
                apply_two_qubit_gate_tebd(mps, gate, sim_params)
            elif method in {"force_all_lr_pauli_enrichment", "force_all_lr_pauli_tdvp"}:
                _apply_two_qubit(mps, node, sim_params, method=method)
            elif method == "current_enriched_tdvp":
                if is_lr and gate.name in {"rxx", "ryy", "rzz"}:
                    decision = decide_long_range_pauli_route(mps, gate, sim_params)
                    route_used = decision.route
                    route_reason = decision.reason
                    projected_ratio = float(decision.visibility.projected_ratio)
                    projection_defect = max(0.0, 1.0 - min(projected_ratio, 1.0))
                    _apply_two_qubit_enriched_tdvp(mps, node, sim_params)
                elif is_nn:
                    apply_two_qubit_gate_tebd(mps, gate, sim_params)
                else:
                    apply_two_qubit_gate_tdvp(mps, gate, sim_params)
            else:
                raise AssertionError(method)

        prefix_qc = build_prefix_circuit(qc, k, order=order)
        ref_prefix = _qiskit_vec(prefix_qc)
        vec = np.asarray(mps.to_vec(), dtype=np.complex128)
        fid_err = _fid_err_vec(ref_prefix, vec)
        phase_l2 = _phase_aligned_l2_error(vec, ref_prefix)

        row = {
            "circuit_name": case.circuit_name,
            "method": method,
            "gate_order": order,
            "prefix_index": step.prefix_index,
            "gate_index_1based": k,
            "gate_name": gate.name if not is_1q else node.op.name,
            "sites": str(tuple(gate.sites)) if not is_1q else str([int(node.qargs[0]._index)]),  # noqa: SLF001
            "is_nearest_neighbor": is_nn,
            "is_long_range": is_lr,
            "route_used": route_used,
            "route_reason": route_reason,
            "projected_ratio": projected_ratio,
            "projection_defect": projection_defect,
            "tdvp_projection_defect_tol": float(sim_params.tdvp_projection_defect_tol),
            "fidelity_error_after_gate": fid_err,
            "phase_aligned_l2_error_after_gate": phase_l2,
            "max_bond_after_gate": int(mps.get_max_bond()),
            "mean_bond_after_gate": _mean_bond_dim(mps),
        }
        rows.append(row)

        if fid_err > 1e-10:
            print(
                f"[prefix] {case.circuit_name} gate {k} ({row['gate_name']} {row['sites']}): "
                f"fid_err={fid_err:.3e} route={route_used!r} defect={projection_defect}"
            )
        if fid_err > 1e-6 and not any(r.get("first_exceeds_1e_6") for r in rows[:-1]):
            row["first_exceeds_1e_6"] = True
        if fid_err > 1e-4 and not any(r.get("first_exceeds_1e_4") for r in rows[:-1]):
            row["first_exceeds_1e_4"] = True

    return rows


def _dense_apply_two_qubit(vec: np.ndarray, n: int, node: DAGOpNode) -> np.ndarray:
    """Exact statevector update for one gate via Qiskit evolution."""
    qargs = [node.qargs[0]._index, node.qargs[1]._index]  # noqa: SLF001
    evolved = Statevector(vec).evolve(node.op, qargs)
    return np.asarray(evolved.data, dtype=np.complex128)


def route_local_ab_test(
    case: Case,
    *,
    prefix_index: int,
    sim_params: StrongSimParams,
    order: RunOrder = "dag_topo",
) -> dict[str, Any]:
    """A/B: TDVP vs enrichment vs TEBD vs dense on the first bad gate."""
    steps = iter_gate_steps(case.qc, order=order)
    if prefix_index >= len(steps):
        msg = f"prefix_index {prefix_index} out of range"
        raise ValueError(msg)

    # Prefix before bad gate.
    mps_base = _init_mps(case.n_qubits)
    for step in steps[:prefix_index]:
        node = step.node
        if len(node.qargs) == 1:
            apply_single_qubit_gate(mps_base, node)
        else:
            _apply_two_qubit(mps_base, node, sim_params, method="current_enriched_tdvp")

    prefix_qc = build_prefix_circuit(case.qc, prefix_index, order=order)
    ref_before = _qiskit_vec(prefix_qc)
    vec_before = np.asarray(mps_base.to_vec(), dtype=np.complex128)
    fid_before = _fid_err_vec(ref_before, vec_before)

    bad_step = steps[prefix_index]
    gate = convert_dag_to_tensor_algorithm(bad_step.node)[0]
    route_used, _reason, pr, defect = _lr_route_decision(mps_base, gate, sim_params, "current_router")

    ref_after_qc = build_prefix_circuit(case.qc, prefix_index + 1, order=order)
    ref_after = _qiskit_vec(ref_after_qc)

    results: dict[str, float] = {}

    def _run_variant(label: str, apply_fn) -> None:
        trial = copy.deepcopy(mps_base)
        apply_fn(trial)
        vec = np.asarray(trial.to_vec(), dtype=np.complex128)
        results[f"{label}_fidelity_error_after_gate"] = _fid_err_vec(ref_after, vec)

    _run_variant("tdvp", lambda s: apply_two_qubit_gate_tdvp(s, gate, sim_params))
    _run_variant(
        "enrichment",
        lambda s: apply_pauli_enriched_optional_compress(
            s, gate, sim_params, compress=bool(getattr(sim_params, "compress_after_enrichment", True))
        ),
    )
    _run_variant("tebd", lambda s: apply_two_qubit_gate_tebd(s, gate, sim_params))

    if case.n_qubits <= 14:
        vec_dense = _dense_apply_two_qubit(vec_before, case.n_qubits, bad_step.node)
        results["dense_exact_fidelity_error_after_gate"] = _fid_err_vec(ref_after, vec_dense)
    else:
        results["dense_exact_fidelity_error_after_gate"] = float("nan")

    i, j = int(gate.sites[0]), int(gate.sites[1])
    return {
        "circuit_name": case.circuit_name,
        "prefix_index": prefix_index,
        "gate_name": gate.name,
        "sites": str(tuple(gate.sites)),
        "input_fidelity_to_qiskit_before_gate": fid_before,
        "projected_ratio_before_gate": pr,
        "projection_defect_before_gate": defect,
        "route_chosen_by_current_router": route_used,
        "tdvp_fidelity_error_after_gate": results["tdvp_fidelity_error_after_gate"],
        "enrichment_fidelity_error_after_gate": results["enrichment_fidelity_error_after_gate"],
        "tebd_fidelity_error_after_gate": results["tebd_fidelity_error_after_gate"],
        "dense_exact_fidelity_error_after_gate": results["dense_exact_fidelity_error_after_gate"],
        "is_long_range": abs(i - j) != 1,
    }


def long_range_route_audit(
    case: Case,
    sim_params: StrongSimParams,
    *,
    method: MethodName = "current_enriched_tdvp",
    order: RunOrder = "dag_topo",
) -> list[dict[str, Any]]:
    """Per LR Pauli gate: fidelity delta vs Qiskit prefix."""
    steps = iter_gate_steps(case.qc, order=order)
    mps = _init_mps(case.n_qubits)
    rows: list[dict[str, Any]] = []

    for k, step in enumerate(steps, start=1):
        node = step.node
        gate = convert_dag_to_tensor_algorithm(node)[0]
        if len(node.qargs) != 2:
            apply_single_qubit_gate(mps, node)
            continue
        i, j = int(gate.sites[0]), int(gate.sites[1])
        if abs(i - j) == 1 or gate.name not in {"rxx", "ryy", "rzz"}:
            if method == "tebd_swaps":
                apply_two_qubit_gate_tebd(mps, gate, sim_params)
            elif abs(i - j) == 1:
                apply_two_qubit_gate_tebd(mps, gate, sim_params)
            else:
                apply_two_qubit_gate_tdvp(mps, gate, sim_params)
            continue

        ref_before = _qiskit_vec(build_prefix_circuit(case.qc, k - 1, order=order))
        fid_before = _fid_err_vec(ref_before, np.asarray(mps.to_vec(), dtype=np.complex128))
        max_bond_before = int(mps.get_max_bond())

        route, reason, pr, defect = _lr_route_decision(mps, gate, sim_params, "current_router")
        _apply_two_qubit(mps, node, sim_params, method=method)

        ref_after = _qiskit_vec(build_prefix_circuit(case.qc, k, order=order))
        fid_after = _fid_err_vec(ref_after, np.asarray(mps.to_vec(), dtype=np.complex128))

        rows.append(
            {
                "circuit_name": case.circuit_name,
                "prefix_index": step.prefix_index,
                "gate": gate.name,
                "sites": str(tuple(gate.sites)),
                "route": route,
                "reason": reason,
                "projected_ratio": pr,
                "projection_defect": defect,
                "defect_tol": float(sim_params.tdvp_projection_defect_tol),
                "max_bond_before": max_bond_before,
                "max_bond_after": int(mps.get_max_bond()),
                "fidelity_error_before": fid_before,
                "fidelity_error_after": fid_after,
                "delta_fidelity_error": fid_after - fid_before,
            }
        )

    rows.sort(key=lambda r: float(r["delta_fidelity_error"]), reverse=True)
    return rows


def tdvp_local_sweep_window_diagnostic(
    case: Case,
    *,
    prefix_index: int,
    sim_params: StrongSimParams,
    order: RunOrder = "dag_topo",
) -> list[dict[str, Any]]:
    steps = iter_gate_steps(case.qc, order=order)
    mps = _init_mps(case.n_qubits)
    for step in steps[:prefix_index]:
        node = step.node
        if len(node.qargs) == 1:
            apply_single_qubit_gate(mps, node)
        else:
            _apply_two_qubit(mps, node, sim_params, method="current_enriched_tdvp")

    bad = steps[prefix_index]
    gate = convert_dag_to_tensor_algorithm(bad.node)[0]
    ref_after = _qiskit_vec(build_prefix_circuit(case.qc, prefix_index + 1, order=order))

    vis = estimate_local_tdvp_projected_norm(mps, gate, sim_params, window_size=1, estimate_update_delta=False)
    mpo, first_site, last_site = construct_generator_mpo(gate, mps.length)
    _, short_mpo, window = apply_window(mps, mpo, first_site, last_site, 1)
    nontrivial = int(short_mpo.length)

    rows: list[dict[str, Any]] = []
    n = mps.length
    full_ws = max(first_site, n - 1 - last_site)
    for n_sweeps in (1, 2, 4, 8):
        for ws in (1, 2, 4, full_ws):
            trial = copy.deepcopy(mps)
            apply_two_qubit_gate_tdvp_window(trial, gate, sim_params, window_size=ws, num_sweeps=n_sweeps)
            fid = _fid_err_vec(ref_after, np.asarray(trial.to_vec(), dtype=np.complex128))
            rows.append(
                {
                    "circuit_name": case.circuit_name,
                    "prefix_index": prefix_index,
                    "gate_name": gate.name,
                    "sites": str(tuple(gate.sites)),
                    "tdvp_sweeps": n_sweeps,
                    "window_size": ws,
                    "fidelity_error_after_gate": fid,
                    "projected_ratio": float(vis.projected_ratio),
                    "projection_defect": max(0.0, 1.0 - min(float(vis.projected_ratio), 1.0)),
                    "nontrivial_mpo_sites_in_window": nontrivial,
                    "max_bond_before": int(mps.get_max_bond()),
                    "max_bond_after": int(trial.get_max_bond()),
                }
            )
    return rows


def compare_gate_orderings(case: Case) -> list[dict[str, Any]]:
    """Canonical gate list: Qiskit data vs DAG topological."""
    data_steps = iter_gate_steps(case.qc, order="circuit_data")
    dag_steps = iter_gate_steps(case.qc, order="dag_topo")
    rows: list[dict[str, Any]] = []
    n = max(len(data_steps), len(dag_steps))
    for idx in range(n):
        d = data_steps[idx] if idx < len(data_steps) else None
        g = dag_steps[idx] if idx < len(dag_steps) else None
        if d is None or g is None:
            rows.append({"index": idx, "mismatch": True})
            continue
        gd = convert_dag_to_tensor_algorithm(d.node)[0]
        gg = convert_dag_to_tensor_algorithm(g.node)[0]
        assert d.instruction is not None
        q_d = [case.qc.find_bit(q).index for q in d.instruction.qubits]
        q_g = [case.qc.find_bit(q).index for q in g.node.qargs]
        rows.append(
            {
                "index": idx,
                "qiskit_gate_name": d.instruction.operation.name,
                "qiskit_qargs": str(q_d),
                "internal_gate_name": gd.name,
                "internal_sites": str(tuple(gd.sites)),
                "parameters": str(list(d.instruction.operation.params)),
                "dag_matches_data": (
                    d.instruction.operation.name == g.node.op.name
                    and q_d == q_g
                    and list(d.instruction.operation.params) == list(g.node.op.params)
                ),
            }
        )
    return rows


def _first_bad_gate(prefix_rows: list[dict[str, Any]], threshold: float = 1e-10) -> dict[str, Any] | None:
    for row in prefix_rows:
        if float(row["fidelity_error_after_gate"]) > threshold:
            return row
    return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if v is None else v) for k, v in row.items()})


def _render_md(
    *,
    method_rows: list[dict[str, Any]],
    prefix_summaries: list[dict[str, Any]],
    ab_rows: list[dict[str, Any]],
    tol_rows: list[dict[str, Any]],
    enrichment_rows: list[dict[str, Any]],
    compression_rows: list[dict[str, Any]],
    tdvp_sweep_rows: list[dict[str, Any]],
    ordering_mismatches: dict[str, int],
    conclusions: list[str],
) -> str:
    lines: list[str] = ["# Enriched TDVP fidelity failure diagnosis\n"]

    lines.append("## Summary\n\n")
    for r in method_rows:
        fid = r.get("fidelity_error")
        if isinstance(fid, (int, float)):
            lines.append(
                f"- `{r['circuit_name']}` / `{r['method']}`: fidelity_error={fid:.3e} "
                f"(TEBD ref fid={r.get('tebd_fidelity_error', 'n/a')})\n"
            )
        else:
            lines.append(f"- `{r['circuit_name']}` / `{r['method']}`\n")

    lines.append("\n## First divergence table\n\n")
    lines.append("circuit | first_bad_gate | gate | sites | route | projected_ratio | projection_defect | fid_before | fid_after\n")
    lines.append("---|---:|---|---|---|---:|---:|---:|---:\n")
    for s in prefix_summaries:
        lines.append(
            f"`{s['circuit_name']}` | {s.get('gate_index_1based', '')} | {s.get('gate_name', '')} | "
            f"{s.get('sites', '')} | {s.get('route_used', '')} | {s.get('projected_ratio', '')} | "
            f"{s.get('projection_defect', '')} | {s.get('fid_before', '')} | {s.get('fid_after', '')}\n"
        )

    lines.append("\n## Route-local A/B tests\n\n")
    if ab_rows:
        lines.append("circuit | gate | TDVP | enrichment | TEBD | dense | router\n")
        lines.append("---|---|---:|---:|---:|---:|---\n")
        for r in ab_rows:
            lines.append(
                f"`{r['circuit_name']}` | {r['gate_name']}{r['sites']} | "
                f"{r['tdvp_fidelity_error_after_gate']:.3e} | {r['enrichment_fidelity_error_after_gate']:.3e} | "
                f"{r['tebd_fidelity_error_after_gate']:.3e} | {r['dense_exact_fidelity_error_after_gate']:.3e} | "
                f"{r['route_chosen_by_current_router']}\n"
            )
    else:
        lines.append("No A/B rows recorded.\n")

    lines.append("\n## Tolerance sweep\n\n")
    if tol_rows:
        lines.append("circuit | defect_tol | fidelity_error | tdvp_lr | enriched_lr | first_bad_gate\n")
        lines.append("---|---:|---:|---:|---:|---:\n")
        for r in tol_rows:
            lines.append(
                f"`{r['circuit_name']}` | {r['defect_tol']} | {r['fidelity_error']:.3e} | "
                f"{r['tdvp_lr_pauli_count']} | {r['enriched_lr_pauli_count']} | {r.get('first_bad_gate', '')}\n"
            )

    lines.append("\n## All-enrichment baseline\n\n")
    for r in enrichment_rows:
        lines.append(f"- `{r['circuit_name']}`: fidelity_error={r['fidelity_error']:.3e}\n")

    lines.append("\n## Compression sweep\n")
    if compression_rows:
        lines.append("circuit | mode | compress | svd_threshold | max_bond | fidelity_error\n")
        lines.append("---|---|---|---:|---:|---:\n")
        for r in compression_rows:
            lines.append(
                f"`{r['circuit_name']}` | {r['mode']} | {r['compress']} | {r['svd_threshold']} | "
                f"{r['max_bond_dim']} | {r['fidelity_error']:.3e}"
            )

    lines.append("\n## TDVP local diagnostics\n")
    if tdvp_sweep_rows:
        best = min(tdvp_sweep_rows, key=lambda r: float(r["fidelity_error_after_gate"]))
        lines.append(
            f"Best sweep/window for first bad gate: sweeps={best['tdvp_sweeps']}, "
            f"window={best['window_size']}, fid={best['fidelity_error_after_gate']:.3e}\n"
        )

    lines.append("\n## Gate ordering\n")
    for cname, n_mis in ordering_mismatches.items():
        lines.append(f"- `{cname}`: DAG vs circuit.data mismatches at index: {n_mis}\n")

    lines.append("\n## Conclusion\n")
    lines.append("Likely cause:\n")
    for c in conclusions:
        lines.append(f"- {c}\n")
    return "".join(lines)


def main() -> None:
    out_dir = _results_dir()
    csv_path = out_dir / "enriched_tdvp_fidelity_failure_debug.csv"
    md_path = out_dir / "enriched_tdvp_fidelity_failure_debug.md"
    fast = os.environ.get("YAQS_FID_DEBUG_FAST", "").strip() not in {"", "0", "false", "False"}
    prefix_order: RunOrder = "dag_topo"

    cases = [_build_case(name) for name in TARGET_CIRCUIT_NAMES]
    all_csv_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    prefix_summaries: list[dict[str, Any]] = []
    ab_rows: list[dict[str, Any]] = []
    tol_rows: list[dict[str, Any]] = []
    enrichment_rows: list[dict[str, Any]] = []
    compression_rows: list[dict[str, Any]] = []
    tdvp_sweep_rows: list[dict[str, Any]] = []
    ordering_mismatches: dict[str, int] = {}
    conclusions: list[str] = []

    methods: list[MethodName] = [
        "qiskit_reference",
        "tebd_swaps",
        "current_enriched_tdvp",
        "force_all_lr_pauli_enrichment",
        "force_all_lr_pauli_tdvp",
    ]

    for case in cases:
        if case.n_qubits > 14:
            print(f"Skipping {case.circuit_name}: n={case.n_qubits} > 14 for Qiskit reference")
            continue

        ref_vec = _qiskit_vec(case.qc)
        params = _make_params()

        # --- Method comparison (DAG order matches benchmark) ---
        tebd_fid: float | None = None
        for method in methods:
            if method == "qiskit_reference":
                m = {
                    "section": "method_comparison",
                    "circuit_name": case.circuit_name,
                    "method": method,
                    "fidelity_error": 0.0,
                    "statevector_norm_error": 0.0,
                    "phase_aligned_l2_error": 0.0,
                    "pauli_obs_max_error": 0.0,
                    "max_bond": 0.0,
                    "mean_bond": 0.0,
                    "wall_time": 0.0,
                }
                method_rows.append(m)
                all_csv_rows.append(m)
                continue

            mps, wall = _run_circuit(case, method=method, params=params, order="dag_topo")
            met = _metrics_vs_ref(mps, ref_vec, case)
            tdvp_n, enr_n = _route_counts(mps)
            row = {
                "section": "method_comparison",
                "circuit_name": case.circuit_name,
                "method": method,
                "defect_tol": DEFAULT_DEFECT_TOL,
                "svd_threshold": DEFAULT_SVD_THRESHOLD,
                "wall_time": wall,
                "tdvp_lr_pauli_count": tdvp_n,
                "enriched_lr_pauli_count": enr_n,
                **met,
            }
            if method == "tebd_swaps":
                tebd_fid = float(met["fidelity_error"]) if met["fidelity_error"] is not None else None
                row["tebd_fidelity_error"] = tebd_fid
            else:
                row["tebd_fidelity_error"] = tebd_fid
            method_rows.append(row)
            all_csv_rows.append(row)
            print(
                f"[method] {case.circuit_name} {method}: fid={met['fidelity_error']:.3e} "
                f"obs_max={met['pauli_obs_max_error']:.3e}"
            )

        # --- Prefix divergence (circuit_data order) ---
        prefix_rows = debug_gate_prefix_divergence(
            case.qc,
            "current_enriched_tdvp",
            params,
            case=case,
            order=prefix_order,
        )
        if not fast:
            data_prefix = debug_gate_prefix_divergence(
                case.qc,
                "current_enriched_tdvp",
                params,
                case=case,
                order="circuit_data",
            )
            for pr in data_prefix:
                pr["section"] = "prefix_divergence_circuit_data"
                all_csv_rows.append(pr)
        for pr in prefix_rows:
            pr["section"] = "prefix_divergence"
            all_csv_rows.append(pr)

        first_bad = _first_bad_gate(prefix_rows, 1e-10)
        if first_bad:
            prev = prefix_rows[first_bad["gate_index_1based"] - 2] if first_bad["gate_index_1based"] >= 2 else None
            fid_before = float(prev["fidelity_error_after_gate"]) if prev else 0.0
            summary = {
                "circuit_name": case.circuit_name,
                "gate_index_1based": first_bad["gate_index_1based"],
                "gate_name": first_bad["gate_name"],
                "sites": first_bad["sites"],
                "route_used": first_bad["route_used"],
                "projected_ratio": first_bad["projected_ratio"],
                "projection_defect": first_bad["projection_defect"],
                "fid_before": fid_before,
                "fid_after": first_bad["fidelity_error_after_gate"],
            }
            prefix_summaries.append(summary)
            all_csv_rows.append({"section": "first_bad_gate", **summary})

            ab = route_local_ab_test(
                case, prefix_index=int(first_bad["prefix_index"]), sim_params=params, order=prefix_order
            )
            ab["section"] = "route_ab"
            ab_rows.append(ab)
            all_csv_rows.append(ab)

            if first_bad.get("route_used") == "tdvp":
                tdvp_sweep_rows.extend(
                    tdvp_local_sweep_window_diagnostic(
                        case,
                        prefix_index=int(first_bad["prefix_index"]),
                        sim_params=params,
                        order=prefix_order,
                    )
                )
                for tr in tdvp_sweep_rows[-20:]:
                    tr["section"] = "tdvp_sweep_window"
                    all_csv_rows.append(tr)

        # --- LR route audit ---
        audit = long_range_route_audit(case, params)
        for a in audit[:15]:
            a["section"] = "lr_route_audit"
            all_csv_rows.append(a)

        # --- Defect tolerance sweep ---
        for tol in (1e-4, 1e-6, 1e-8, 1e-10, 0.0):
            p_tol = _make_params(defect_tol=tol)
            mps, _ = _run_circuit(case, method="current_enriched_tdvp", params=p_tol, order="dag_topo")
            met = _metrics_vs_ref(mps, ref_vec, case)
            tdvp_n, enr_n = _route_counts(mps)
            tr = {
                "section": "tolerance_sweep",
                "circuit_name": case.circuit_name,
                "defect_tol": tol,
                "fidelity_error": met["fidelity_error"],
                "tdvp_lr_pauli_count": tdvp_n,
                "enriched_lr_pauli_count": enr_n,
                "first_bad_gate": None,
            }
            tol_rows.append(tr)
            all_csv_rows.append(tr)

        # --- All-enrichment baseline ---
        p_all = _make_params()
        mps_all, _ = _run_circuit(
            case, method="force_all_lr_pauli_enrichment", params=p_all, order="dag_topo"
        )
        met_all = _metrics_vs_ref(mps_all, ref_vec, case)
        enrichment_rows.append({"circuit_name": case.circuit_name, **met_all})
        all_csv_rows.append({"section": "all_enrichment", "circuit_name": case.circuit_name, **met_all})

        # --- Compression sweep (skip control and fast mode) ---
        if not fast and not case.circuit_name.endswith("_all_zero"):
            svd_values = (1e-14, 1e-12, 1e-9)
            chi_values: tuple[int | None, ...] = (None, 16)
            for mode, lr_mode in (("current_router", "current_router"), ("all_enrichment", "force_enrichment")):
                for compress in (True, False):
                    for svd_th in svd_values:
                        for chi in chi_values:
                            p_c = _make_params(
                                svd_threshold=svd_th, max_bond_dim=chi, compress_after_enrichment=compress
                            )
                            mps_c = _init_mps(case.n_qubits)
                            for step in iter_gate_steps(case.qc, order="dag_topo"):
                                node = step.node
                                if len(node.qargs) == 1:
                                    apply_single_qubit_gate(mps_c, node)
                                    continue
                                gate = convert_dag_to_tensor_algorithm(node)[0]
                                if abs(gate.sites[0] - gate.sites[1]) == 1:
                                    apply_two_qubit_gate_tebd(mps_c, gate, p_c)
                                    continue
                                if gate.name in {"rxx", "ryy", "rzz"}:
                                    route, _, _, _ = _lr_route_decision(mps_c, gate, p_c, lr_mode)  # type: ignore[arg-type]
                                    if route == "pauli_enriched":
                                        apply_pauli_enriched_optional_compress(
                                            mps_c, gate, p_c, compress=compress
                                        )
                                    else:
                                        apply_two_qubit_gate_tdvp(mps_c, gate, p_c)
                                else:
                                    apply_two_qubit_gate_tdvp(mps_c, gate, p_c)
                            fid = _fid_err_vec(ref_vec, np.asarray(mps_c.to_vec(), dtype=np.complex128))
                            cr = {
                                "section": "compression_sweep",
                                "circuit_name": case.circuit_name,
                                "mode": mode,
                                "compress": compress,
                                "svd_threshold": svd_th,
                                "max_bond_dim": chi,
                                "fidelity_error": fid,
                            }
                            compression_rows.append(cr)
                            all_csv_rows.append(cr)

        # --- Gate ordering (first failing circuit only) ---
        if case.circuit_name == TARGET_CIRCUIT_NAMES[0]:
            ord_rows = compare_gate_orderings(case)
            n_mis = sum(1 for r in ord_rows if not r.get("dag_matches_data", True))
            ordering_mismatches[case.circuit_name] = n_mis
            for r in ord_rows:
                r["section"] = "gate_ordering"
                r["circuit_name"] = case.circuit_name
                all_csv_rows.append(r)

    # --- Auto conclusions ---
    for case in cases:
        if case.n_qubits > 14:
            continue
        by_method = {r["method"]: r for r in method_rows if r["circuit_name"] == case.circuit_name}
        cur = by_method.get("current_enriched_tdvp", {})
        enr = by_method.get("force_all_lr_pauli_enrichment", {})
        tebd = by_method.get("tebd_swaps", {})
        if not cur or not enr:
            continue
        cur_fid = float(cur.get("fidelity_error", 1.0))
        enr_fid = float(enr.get("fidelity_error", 1.0))
        tebd_fid = float(tebd.get("fidelity_error", 1.0))
        if tebd_fid < 1e-10 and cur_fid > 1e-6 and enr_fid < 1e-10:
            conclusions.append(
                f"`{case.circuit_name}`: exact enrichment matches Qiskit; current router fails "
                "→ TDVP-routed LR Pauli gates are the likely culprit."
            )
        elif enr_fid > 1e-8:
            conclusions.append(
                f"`{case.circuit_name}`: all-enrichment does not match Qiskit "
                "→ check enrichment, compression, or gate conventions."
            )
        elif cur_fid > 1e-6 and enr_fid < 1e-10:
            conclusions.append(f"`{case.circuit_name}`: router sends unsafe gates to TDVP.")

    if not conclusions:
        conclusions.append("No strong automatic classification; inspect per-circuit tables in CSV.")

    _write_csv(csv_path, all_csv_rows)
    md_path.write_text(
        _render_md(
            method_rows=method_rows,
            prefix_summaries=prefix_summaries,
            ab_rows=ab_rows,
            tol_rows=tol_rows,
            enrichment_rows=enrichment_rows,
            compression_rows=compression_rows,
            tdvp_sweep_rows=tdvp_sweep_rows,
            ordering_mismatches=ordering_mismatches,
            conclusions=conclusions,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
