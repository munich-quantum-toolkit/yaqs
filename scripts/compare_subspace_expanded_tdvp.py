#!/usr/bin/env python
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compare subspace-expanded TDVP vs enrichment/router baselines.

Run:

    uv run python -m scripts.compare_subspace_expanded_tdvp

Outputs:
    results/subspace_expanded_tdvp_comparison.csv
    results/subspace_expanded_tdvp_comparison.md
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Pauli, Statevector

from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.digital.digital_tjm import (
    apply_pauli_product_rotation_enriched,
    apply_pauli_product_rotation_subspace_expanded_tdvp,
    apply_pauli_product_rotation_subspace_expanded_tdvp_with_diagnostics,
    apply_single_qubit_gate,
    apply_two_qubit_gate_tebd,
    apply_two_qubit_gate_tdvp,
    decide_long_range_pauli_route,
    estimate_local_tdvp_projected_norm,
    mps_overlap,
)
from mqt.yaqs.digital.utils.dag_utils import convert_dag_to_tensor_algorithm


MethodName = Literal[
    "plain_tdvp",
    "current_projection_defect_router",
    "exact_pauli_enrichment",
    "subspace_expanded_tdvp",
]


@dataclass(frozen=True)
class CircuitCase:
    name: str
    family: str
    n_qubits: int
    depth: int
    qc: QuantumCircuit


@dataclass(frozen=True)
class Row:
    circuit_name: str
    method: str
    n_qubits: int
    depth: int
    fidelity_error: float | None
    pauli_obs_max_error: float
    pauli_obs_mean_error: float
    pauli_obs_l2_error: float
    worst_observable: str | None
    max_bond_before: int
    max_bond_after_expansion: int | None
    max_bond_final: int
    mean_bond_final: float
    sum_chi_cubed: float
    wall_time_s: float
    projected_ratio_before_expansion: float | None
    projection_defect_before_expansion: float | None
    projected_ratio_after_expansion: float | None
    projection_defect_after_expansion: float | None
    state_delta_from_expansion: float | None
    route_summary: str | None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _results_dir() -> Path:
    d = _repo_root() / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _qiskit_vec(qc: QuantumCircuit) -> np.ndarray:
    return np.asarray(Statevector(qc).data, dtype=np.complex128)


def _fid_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(max(0.0, 1.0 - abs(np.vdot(a, b)) ** 2))


def _statevector_expect_pauli(state: np.ndarray, n: int, which: str, sites: list[int]) -> complex:
    label = ["I"] * n
    if which in {"X", "Y", "Z"}:
        label[n - 1 - sites[0]] = which
    else:
        p = which[0]
        label[n - 1 - sites[0]] = p
        label[n - 1 - sites[1]] = p
    return complex(Statevector(state).expectation_value(Pauli("".join(label))))


def _observable_set(qc: QuantumCircuit, *, max_all_sites_n: int = 16) -> list[tuple[str, Observable, int | list[int]]]:
    n = qc.num_qubits
    if n <= max_all_sites_n:
        sites_1q = list(range(n))
    else:
        core = {0, n - 1, n // 2, max(0, n // 2 - 1)}
        sites_1q = sorted(core | set(range(0, n, 2)))

    obs: list[tuple[str, Observable, int | list[int]]] = []
    for i in sites_1q:
        obs.append((f"X({i})", Observable("x", i), i))
        obs.append((f"Y({i})", Observable("y", i), i))
        obs.append((f"Z({i})", Observable("z", i), i))
    for i in sorted({max(0, n // 2 - 2), max(0, n // 2 - 1), n // 2}):
        if i + 1 >= n:
            continue
        obs.append((f"XX({i},{i+1})", Observable("xx", [i, i + 1]), [i, i + 1]))
        obs.append((f"YY({i},{i+1})", Observable("yy", [i, i + 1]), [i, i + 1]))
        obs.append((f"ZZ({i},{i+1})", Observable("zz", [i, i + 1]), [i, i + 1]))
    return obs


def _pauli_obs_errors_vs_statevector(qc: QuantumCircuit, mps: MPS, ref_vec: np.ndarray) -> tuple[float, float, float, str | None]:
    obs = _observable_set(qc)
    errs: list[tuple[str, float]] = []
    n = qc.num_qubits
    for name, ob, _sites in obs:
        ket_op = MPS(length=mps.length, tensors=[np.asarray(t, dtype=np.complex128) for t in mps.tensors])
        ket_op.apply_local(ob)
        got = float(np.real(mps_overlap(mps, ket_op)))

        if name.startswith(("X(", "Y(", "Z(")):
            which = name[0]
            idx = int(name.split("(")[1].split(")")[0])
            ref = float(np.real(_statevector_expect_pauli(ref_vec, n, which, [idx])))
        else:
            which = name[:2]
            ij = name.split("(")[1].split(")")[0]
            i_s, j_s = ij.split(",")
            ref = float(np.real(_statevector_expect_pauli(ref_vec, n, which, [int(i_s), int(j_s)])))

        errs.append((name, abs(got - ref)))

    if not errs:
        return 0.0, 0.0, 0.0, None
    worst_name, worst_val = max(errs, key=lambda x: x[1])
    vals = np.array([e for _, e in errs], dtype=np.float64)
    return float(worst_val), float(np.mean(vals)), float(np.linalg.norm(vals)), str(worst_name)


def _mps_cost_metrics(mps: MPS) -> tuple[int, float, float]:
    max_bond = int(mps.get_max_bond())
    bond_dims = [int(np.asarray(t).shape[2]) for t in mps.tensors[:-1]]
    mean_bond = float(np.mean(bond_dims)) if bond_dims else 1.0
    sum_chi_cubed = float(sum((chi**3 for chi in bond_dims)))
    return max_bond, mean_bond, sum_chi_cubed


def _route_stats_from_state(mps: MPS) -> str | None:
    stats = getattr(mps, "route_stats", None)
    if not isinstance(stats, dict):
        return None
    ratios = stats.get("ratios", None)
    if not isinstance(ratios, list):
        return None
    return json.dumps(ratios, separators=(",", ":"), sort_keys=True)


def _apply_two_qubit_custom(
    mps: MPS,
    gate_node,
    params: StrongSimParams,
    *,
    method: MethodName,
    defect_tol: float,
) -> tuple[int, int]:
    gate = convert_dag_to_tensor_algorithm(gate_node)[0]
    site0, site1 = gate.sites
    is_nn = abs(site0 - site1) == 1

    if is_nn:
        return apply_two_qubit_gate_tebd(mps, gate, params)

    if gate.name in {"rxx", "ryy", "rzz"}:
        if method == "plain_tdvp":
            return apply_two_qubit_gate_tdvp(mps, gate, params)
        if method == "exact_pauli_enrichment":
            return apply_pauli_product_rotation_enriched(mps, gate, params, record_stats=False)
        if method == "subspace_expanded_tdvp":
            # Diagnostics are handled by the caller; this keeps the gate application
            # single-shot (no double expansion).
            return apply_pauli_product_rotation_subspace_expanded_tdvp(mps, gate, params)
        if method == "current_projection_defect_router":
            params.tdvp_projection_defect_tol = float(defect_tol)
            decision = decide_long_range_pauli_route(mps, gate, params)
            if decision.route == "pauli_enriched":
                return apply_pauli_product_rotation_enriched(mps, gate, params, record_stats=False)
            return apply_two_qubit_gate_tdvp(mps, gate, params)
        raise AssertionError(f"Unknown method: {method}")

    # Non-Pauli LR gates: always TDVP.
    return apply_two_qubit_gate_tdvp(mps, gate, params)


def _run_method_on_circuit(
    case: CircuitCase,
    *,
    method: MethodName,
    defect_tol_router: float = 1e-4,
) -> Row:
    params = StrongSimParams(
        preset="exact",
        gate_mode="hybrid",
        svd_threshold=1e-14,
        max_bond_dim=None,
        krylov_tol=1e-12,
        tangent_blindness_tol=1e-12,
        tdvp_projection_defect_tol=float(defect_tol_router),
        tdvp_visibility_safety_tol=None,
        tdvp_pauli_consistency_check=False,
        get_state=True,
    )

    st = State(case.n_qubits, initial="zeros", representation="mps")
    mps = st.mps
    dag = circuit_to_dag(case.qc)

    max_bond_before = int(mps.get_max_bond())
    max_bond_after_expansion: int | None = None
    projected_ratio_before_expansion: float | None = None
    projection_defect_before_expansion: float | None = None
    projected_ratio_after_expansion: float | None = None
    projection_defect_after_expansion: float | None = None
    state_delta_from_expansion: float | None = None

    t0 = time.perf_counter()
    for node in dag.topological_op_nodes():
        if len(node.qargs) == 1:
            apply_single_qubit_gate(mps, node)
            continue
        if len(node.qargs) != 2:
            raise AssertionError(f"Unexpected {len(node.qargs)}-qubit op: {node.op.name}")

        gate = convert_dag_to_tensor_algorithm(node)[0]
        if (not (abs(gate.sites[0] - gate.sites[1]) == 1)) and gate.name in {"rxx", "ryy", "rzz"}:
            if method == "subspace_expanded_tdvp":
                # Record expansion diagnostics for the first LR Pauli gate in the circuit.
                if projected_ratio_before_expansion is None:
                    _sites, diag = apply_pauli_product_rotation_subspace_expanded_tdvp_with_diagnostics(mps, gate, params)
                    projected_ratio_before_expansion = diag.projected_ratio_before
                    projection_defect_before_expansion = diag.projection_defect_before
                    projected_ratio_after_expansion = diag.projected_ratio_after
                    projection_defect_after_expansion = diag.projection_defect_after
                    state_delta_from_expansion = diag.state_delta_from_expansion
                    max_bond_after_expansion = diag.max_bond_after_expansion
                    continue
                # Subsequent LR Pauli gates: use the helper (no per-gate diagnostics).
                apply_pauli_product_rotation_subspace_expanded_tdvp(mps, gate, params)
                continue

        _apply_two_qubit_custom(mps, node, params, method=method, defect_tol=defect_tol_router)

    wall = float(time.perf_counter() - t0)

    # Reference metrics.
    fidelity_error: float | None = None
    pauli_max = 0.0
    pauli_mean = 0.0
    pauli_l2 = 0.0
    worst_obs: str | None = None
    if case.n_qubits <= 14:
        ref_vec = _qiskit_vec(case.qc)
        vec = np.asarray(mps.to_vec(), dtype=np.complex128)
        fidelity_error = _fid_err(ref_vec, vec)
        pauli_max, pauli_mean, pauli_l2, worst_obs = _pauli_obs_errors_vs_statevector(case.qc, mps, ref_vec)

    max_bond_final, mean_bond_final, sum_chi_cubed = _mps_cost_metrics(mps)
    return Row(
        circuit_name=case.name,
        method=str(method),
        n_qubits=case.n_qubits,
        depth=case.depth,
        fidelity_error=fidelity_error,
        pauli_obs_max_error=pauli_max,
        pauli_obs_mean_error=pauli_mean,
        pauli_obs_l2_error=pauli_l2,
        worst_observable=worst_obs,
        max_bond_before=max_bond_before,
        max_bond_after_expansion=max_bond_after_expansion,
        max_bond_final=max_bond_final,
        mean_bond_final=mean_bond_final,
        sum_chi_cubed=sum_chi_cubed,
        wall_time_s=wall,
        projected_ratio_before_expansion=projected_ratio_before_expansion,
        projection_defect_before_expansion=projection_defect_before_expansion,
        projected_ratio_after_expansion=projected_ratio_after_expansion,
        projection_defect_after_expansion=projection_defect_after_expansion,
        state_delta_from_expansion=state_delta_from_expansion,
        route_summary=_route_stats_from_state(mps),
    )


def _build_cases() -> list[CircuitCase]:
    cases: list[CircuitCase] = []

    qc = QuantumCircuit(8)
    qc.rxx(0.25, 1, 6)
    cases.append(CircuitCase("tangent_blind_rxx_8q", "minimal", 8, 1, qc))

    qc = QuantumCircuit(8)
    qc.ryy(0.25, 1, 6)
    cases.append(CircuitCase("tangent_blind_ryy_8q", "minimal", 8, 1, qc))

    qc = QuantumCircuit(8)
    qc.rzz(0.25, 1, 6)
    cases.append(CircuitCase("sanity_rzz_8q", "minimal", 8, 1, qc))

    qc = QuantumCircuit(8)
    qc.rx(np.pi / 2, 6)
    qc.ryy(0.25, 1, 6)
    cases.append(CircuitCase("endpoint_prepared_ryy_8q", "minimal", 8, 2, qc))

    qc = QuantumCircuit(8)
    qc.ry(np.pi / 2, 6)
    qc.rxx(0.25, 1, 6)
    cases.append(CircuitCase("endpoint_prepared_rxx_8q", "minimal", 8, 2, qc))

    qc = QuantumCircuit(10)
    qc.rxx(0.21, 2, 9)
    qc.ryy(0.25, 3, 8)
    cases.append(CircuitCase("stack_rxx_ryy_vacuum_10q", "stacks", 10, 2, qc))

    qc = QuantumCircuit(12)
    qc.ry(np.pi / 4, 1)
    qc.ry(np.pi / 4, 4)
    qc.ry(np.pi / 4, 7)
    qc.rzz(0.19, 1, 10)
    qc.rzz(0.27, 4, 11)
    qc.rzz(0.33, 0, 7)
    qc.rxx(0.21, 2, 9)
    qc.ryy(0.25, 3, 8)
    cases.append(CircuitCase("lr_stack_mixed_12q", "stacks", 12, 8, qc))

    # Small random subset (n<=12 so Qiskit ref works).
    seeds = list(range(5))
    n_values = [8, 10, 12]
    depth_values = [4, 8]
    axes = ["rxx", "ryy", "rzz"]
    for n in n_values:
        for depth in depth_values:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                qc = QuantumCircuit(n)
                for _layer in range(depth):
                    for q in range(n):
                        ang = float(rng.uniform(-0.5, 0.5))
                        gate = rng.choice(["rx", "ry", "rz"])
                        getattr(qc, gate)(ang, q)
                    while True:
                        i = int(rng.integers(0, n))
                        j = int(rng.integers(0, n))
                        if i != j and abs(i - j) >= 3:
                            break
                    ax = str(rng.choice(axes))
                    theta = float(rng.uniform(-0.7, 0.7))
                    if ax == "rzz":
                        qc.ry(np.pi / 6, i)
                        qc.ry(np.pi / 7, j)
                    getattr(qc, ax)(theta, i, j)
                cases.append(CircuitCase(f"rand_n{n}_d{depth}_s{seed}", "random", n, depth, qc))

    return cases


def _render_md(rows: list[Row]) -> str:
    lines: list[str] = []
    lines.append("# Subspace-expanded TDVP comparison\n")
    lines.append("Strict: fidelity_error<=1e-10 and pauli_obs_max_error<=1e-10 (when fidelity available).\n")
    lines.append("Practical: pauli_obs_max_error<=1e-3.\n")
    lines.append("## Summary table (per circuit/method)\n")
    lines.append(
        "circuit | method | fid_err | obs_max | obs_mean | obs_l2 | worst_obs | bond_before | bond_after_expand | bond_final | wall_s\n"
    )
    lines.append("---|---|---:|---:|---:|---:|---|---:|---:|---:|---:\n")
    for r in rows:
        lines.append(
            f"`{r.circuit_name}` | {r.method} | "
            f"{('—' if r.fidelity_error is None else f'{r.fidelity_error:.3e}')} | "
            f"{r.pauli_obs_max_error:.3e} | {r.pauli_obs_mean_error:.3e} | {r.pauli_obs_l2_error:.3e} | "
            f"{('—' if r.worst_observable is None else r.worst_observable)} | "
            f"{r.max_bond_before} | {('—' if r.max_bond_after_expansion is None else r.max_bond_after_expansion)} | "
            f"{r.max_bond_final} | {r.wall_time_s:.2f}"
        )
    lines.append("\n## Expansion diagnostics (subspace_expanded_tdvp only)\n")
    lines.append("circuit | d_before | d_after | state_delta_from_expansion | bond_after_expand\n")
    lines.append("---|---:|---:|---:|---:\n")
    for r in rows:
        if r.method != "subspace_expanded_tdvp":
            continue
        lines.append(
            f"`{r.circuit_name}` | "
            f"{('—' if r.projection_defect_before_expansion is None else f'{r.projection_defect_before_expansion:.3e}')} | "
            f"{('—' if r.projection_defect_after_expansion is None else f'{r.projection_defect_after_expansion:.3e}')} | "
            f"{('—' if r.state_delta_from_expansion is None else f'{r.state_delta_from_expansion:.3e}')} | "
            f"{('—' if r.max_bond_after_expansion is None else str(r.max_bond_after_expansion))}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    cases = _build_cases()
    methods: list[MethodName] = [
        "plain_tdvp",
        "current_projection_defect_router",
        "exact_pauli_enrichment",
        "subspace_expanded_tdvp",
    ]

    out_dir = _results_dir()
    csv_path = out_dir / "subspace_expanded_tdvp_comparison.csv"
    md_path = out_dir / "subspace_expanded_tdvp_comparison.md"

    rows: list[Row] = []
    for case in cases:
        for method in methods:
            rows.append(_run_method_on_circuit(case, method=method, defect_tol_router=1e-4))

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(Row.__annotations__.keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)

    md_path.write_text(_render_md(rows), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

