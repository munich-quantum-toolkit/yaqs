#!/usr/bin/env python
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Benchmark enriched TDVP (projection-defect routed) vs TEBD+SWAPs.

Run:

    uv run python -m scripts.benchmark_enriched_tdvp_vs_tebd

Outputs:
    results/enriched_tdvp_vs_tebd.csv
    results/enriched_tdvp_vs_tebd.md
    results/enriched_tdvp_vs_tebd_bond_history.csv (optional; set YAQS_BOND_HISTORY=1)
"""

from __future__ import annotations

import copy
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector

from mqt.yaqs import Simulator
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.libraries.gate_library import GateLibrary
from mqt.yaqs.digital.digital_tjm import (
    apply_pauli_product_rotation_enriched,
    apply_single_qubit_gate,
    apply_two_qubit_gate_tebd,
    apply_two_qubit_gate_tdvp,
    decide_long_range_pauli_route,
    mps_overlap,
)
from mqt.yaqs.digital.utils.dag_utils import convert_dag_to_tensor_algorithm


MethodName = Literal["enriched_tdvp", "tebd_swaps"]
FamilyName = Literal["ising_2d", "xxx_2d", "ising_1d_periodic", "xxx_1d_periodic", "random_lr_pauli"]
InitialStateName = Literal["all_zero", "plus", "neel", "random_product"]


@dataclass(frozen=True)
class Case:
    family: str
    model: str
    geometry: str
    boundary_condition: str
    lx: int | None
    ly: int | None
    n_qubits: int
    initial_state: str
    dt: float
    layers_or_depth: int
    circuit_name: str
    qc: QuantumCircuit
    edge_types: dict[tuple[int, int, str], str]


@dataclass(frozen=True)
class Row:
    family: str
    model: str
    geometry: str
    boundary_condition: str
    lx: int | None
    ly: int | None
    n_qubits: int
    initial_state: str
    dt: float
    layers: int
    circuit_name: str
    method: str
    defect_tol: float | None
    svd_threshold: float
    max_bond_dim_setting: int | None
    reference_method: str | None
    fidelity_error: float | None
    pauli_obs_max_error: float | None
    pauli_obs_mean_error: float | None
    pauli_obs_l2_error: float | None
    worst_observable: str | None
    observable_ordering_suspect: bool
    strict_pass: bool | None
    practical_pass: bool | None
    max_bond_observed: int
    mean_bond_observed: float
    sum_chi_cubed: float
    wall_time_s: float
    num_1q_gates: int
    num_nn_2q_gates: int
    num_lr_pauli_gates: int
    num_swaps_inserted: int
    tdvp_lr_pauli_count: int | None
    enriched_lr_pauli_count: int | None
    tdvp_lr_pauli_fraction: float | None
    enriched_lr_pauli_fraction: float | None
    route_summary: str


@dataclass(frozen=True)
class BondHistoryRow:
    circuit_name: str
    method: str
    defect_tol: float | None
    max_bond_dim_setting: int | None
    layer: int
    gate_index: int
    gate_name: str
    sites: str
    is_long_range: bool
    max_bond: int
    mean_bond: float
    sum_chi_cubed: float


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _results_dir() -> Path:
    d = _repo_root() / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _site(lx: int, x: int, y: int) -> int:
    return y * lx + x


def _grid_edges(
    *,
    lx: int,
    ly: int,
    periodic_x: bool,
) -> list[tuple[int, int, str]]:
    edges: list[tuple[int, int, str]] = []
    for y in range(ly):
        for x in range(lx - 1):
            a = _site(lx, x, y)
            b = _site(lx, x + 1, y)
            edges.append((a, b, "horizontal_nn"))
        if periodic_x and lx > 2:
            a = _site(lx, lx - 1, y)
            b = _site(lx, 0, y)
            edges.append((a, b, "horizontal_periodic_lr"))
    for y in range(ly - 1):
        for x in range(lx):
            a = _site(lx, x, y)
            b = _site(lx, x, y + 1)
            edges.append((a, b, "vertical_lr"))
    return edges


def _prep_initial_state(qc: QuantumCircuit, name: InitialStateName, *, seed: int = 0) -> None:
    n = qc.num_qubits
    if name == "all_zero":
        return
    if name == "neel":
        for i in range(n):
            if i % 2 == 1:
                qc.x(i)
        return
    if name == "plus":
        for i in range(n):
            qc.ry(np.pi / 2, i)
        return
    if name == "random_product":
        rng = np.random.default_rng(seed)
        for i in range(n):
            qc.rx(float(rng.uniform(-np.pi, np.pi)), i)
            qc.ry(float(rng.uniform(-np.pi, np.pi)), i)
            qc.rz(float(rng.uniform(-np.pi, np.pi)), i)
        return
    raise AssertionError(f"Unknown initial state: {name}")


def _ising_2d_row_major(
    *,
    lx: int,
    ly: int,
    j: float,
    h: float,
    dt: float,
    layers: int,
    periodic_x: bool,
) -> tuple[QuantumCircuit, dict[tuple[int, int, str], str]]:
    n = lx * ly
    qc = QuantumCircuit(n)
    edges = _grid_edges(lx=lx, ly=ly, periodic_x=periodic_x)
    edge_types: dict[tuple[int, int, str], str] = {}
    for _ in range(layers):
        for a, b, et in edges:
            qc.rzz(2 * j * dt, a, b)
            i, k = (a, b) if a < b else (b, a)
            edge_types[(i, k, "rzz")] = et
        for q in range(n):
            qc.rx(2 * h * dt, q)
        qc.barrier()
    return qc, edge_types


def _xxx_2d_row_major(
    *,
    lx: int,
    ly: int,
    j: float,
    dt: float,
    layers: int,
    periodic_x: bool,
) -> tuple[QuantumCircuit, dict[tuple[int, int, str], str]]:
    n = lx * ly
    qc = QuantumCircuit(n)
    edges = _grid_edges(lx=lx, ly=ly, periodic_x=periodic_x)
    edge_types: dict[tuple[int, int, str], str] = {}
    for _ in range(layers):
        for a, b, et in edges:
            for name in ("rxx", "ryy", "rzz"):
                getattr(qc, name)(2 * j * dt, a, b)
                i, k = (a, b) if a < b else (b, a)
                edge_types[(i, k, name)] = et
        qc.barrier()
    return qc, edge_types


def _ising_1d_periodic(*, n: int, j: float, h: float, dt: float, layers: int) -> tuple[QuantumCircuit, dict[tuple[int, int, str], str]]:
    qc = QuantumCircuit(n)
    edge_types: dict[tuple[int, int, str], str] = {}
    for _ in range(layers):
        for i in range(n):
            k = (i + 1) % n
            qc.rzz(2 * j * dt, i, k)
            a, b = (i, k) if i < k else (k, i)
            edge_types[(a, b, "rzz")] = "periodic_lr" if (a, b) == (0, n - 1) else "horizontal_nn"
        for q in range(n):
            qc.rx(2 * h * dt, q)
        qc.barrier()
    return qc, edge_types


def _xxx_1d_periodic(*, n: int, j: float, dt: float, layers: int) -> tuple[QuantumCircuit, dict[tuple[int, int, str], str]]:
    qc = QuantumCircuit(n)
    edge_types: dict[tuple[int, int, str], str] = {}
    for _ in range(layers):
        for i in range(n):
            k = (i + 1) % n
            for name in ("rxx", "ryy", "rzz"):
                getattr(qc, name)(2 * j * dt, i, k)
                a, b = (i, k) if i < k else (k, i)
                edge_types[(a, b, name)] = "periodic_lr" if (a, b) == (0, n - 1) else "horizontal_nn"
        qc.barrier()
    return qc, edge_types


def _random_lr_pauli(
    *,
    n: int,
    depth: int,
    seed: int,
    axes: tuple[str, ...] = ("rxx", "ryy", "rzz"),
) -> tuple[QuantumCircuit, dict[tuple[int, int, str], str]]:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    edge_types: dict[tuple[int, int, str], str] = {}
    for _ in range(depth):
        for q in range(n):
            ang = float(rng.uniform(-0.5, 0.5))
            gate = str(rng.choice(["rx", "ry", "rz"]))
            getattr(qc, gate)(ang, q)

        while True:
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n))
            if i != j and abs(i - j) >= 3:
                break
        name = str(rng.choice(list(axes)))
        theta = float(rng.uniform(-0.7, 0.7))
        getattr(qc, name)(theta, i, j)
        a, b = (i, j) if i < j else (j, i)
        edge_types[(a, b, name)] = "random_lr"
        qc.barrier()
    return qc, edge_types


def _qiskit_vec(qc: QuantumCircuit) -> np.ndarray:
    return np.asarray(Statevector(qc).data, dtype=np.complex128)


def _fid_err_vec(a: np.ndarray, b: np.ndarray) -> float:
    return float(max(0.0, 1.0 - abs(np.vdot(a, b)) ** 2))


def _mps_fidelity_error(a: MPS, b: MPS) -> float:
    oa = mps_overlap(a, a)
    ob = mps_overlap(b, b)
    if abs(oa) < 1e-300 or abs(ob) < 1e-300:
        return 1.0
    ov = mps_overlap(a, b)
    return float(max(0.0, 1.0 - abs(ov) ** 2 / (float(np.real(oa)) * float(np.real(ob)))))


def _mean_bond_dim(mps: MPS) -> float:
    bonds = [int(t.shape[1]) for t in mps.tensors[1:]]
    return float(np.mean(bonds)) if bonds else 1.0


def _sum_chi_cubed(mps: MPS) -> float:
    bonds = [int(t.shape[1]) for t in mps.tensors[1:]]
    return float(sum((chi**3 for chi in bonds)))


def _num_swaps_for_gate(i: int, j: int) -> int:
    """SWAP count inserted by TEBD routing for a non-adjacent (i,j) gate."""
    d = abs(i - j)
    if d <= 1:
        return 0
    return 2 * (d - 1)


def _count_circuit_gates(qc: QuantumCircuit) -> tuple[int, int, int, int]:
    num_1q = 0
    num_nn_2q = 0
    num_lr_pauli = 0
    num_swaps_inserted = 0
    for ci in qc.data:
        inst = ci.operation
        if inst.name == "barrier":
            continue
        qargs = ci.qubits
        if len(qargs) == 1:
            num_1q += 1
            continue
        if len(qargs) != 2:
            continue
        i = qc.find_bit(qargs[0]).index
        j = qc.find_bit(qargs[1]).index
        if abs(i - j) == 1:
            num_nn_2q += 1
        else:
            if inst.name in {"rxx", "ryy", "rzz"}:
                num_lr_pauli += 1
            num_swaps_inserted += _num_swaps_for_gate(i, j)
    return num_1q, num_nn_2q, num_lr_pauli, num_swaps_inserted


def _route_summary(mps: MPS) -> str:
    stats = getattr(mps, "route_stats", None)
    if not isinstance(stats, dict):
        return "{}"
    return json.dumps(stats, separators=(",", ":"), sort_keys=True, default=str)


def _route_counts(mps: MPS) -> tuple[int | None, int | None]:
    stats = getattr(mps, "route_stats", None)
    if not isinstance(stats, dict):
        return None, None
    tdvp = stats.get("tdvp_lr_pauli", None)
    enr = stats.get("enriched_lr_pauli", None)
    return (None if tdvp is None else int(tdvp), None if enr is None else int(enr))


def _expectation_via_mps_swaps(
    psi: MPS,
    *,
    gate_name: str,
    sites: list[int],
) -> float:
    """Compute <psi|O|psi> for 1q/2q Pauli-product O (swap-to-adjacency)."""
    if len(sites) == 1:
        ket = copy.deepcopy(psi)
        ket.apply_local(Observable(gate_name, sites[0]))
        return float(np.real(mps_overlap(psi, ket)))

    i, j = sorted([int(sites[0]), int(sites[1])])
    ket = copy.deepcopy(psi)
    sp = StrongSimParams(preset="exact", svd_threshold=1e-16, max_bond_dim=None, gate_mode="tebd", get_state=True)

    for k in range(j - 1, i, -1):
        sw = GateLibrary.swap()
        sw.set_sites(k, k + 1)
        apply_two_qubit_gate_tebd(ket, sw, sp)

    ket.apply_local(Observable(gate_name, [i, i + 1]))

    for k in range(i, j - 1):
        sw = GateLibrary.swap()
        sw.set_sites(k, k + 1)
        apply_two_qubit_gate_tebd(ket, sw, sp)

    return float(np.real(mps_overlap(psi, ket)))


def _statevector_expectation(n: int, vec: np.ndarray, *, label: str, sites: list[int]) -> float:
    pauli = ["I"] * n
    for p, s in zip(label, sites, strict=True):
        pauli[n - 1 - s] = p
    val = Statevector(vec).expectation_value(Pauli("".join(pauli)))
    return float(np.real(complex(val)))


def _observable_set(case: Case) -> list[tuple[str, str, list[int]]]:
    """Return (name, label, sites) with label in {'X','Y','Z','XX','YY','ZZ'}."""
    n = case.n_qubits
    obs: list[tuple[str, str, list[int]]] = []

    # 1q on all sites (sample if large).
    if n <= 20:
        sites_1q = list(range(n))
    else:
        core = {0, n - 1, n // 2, max(0, n // 2 - 1)}
        sampled = set(range(0, n, 2))
        sites_1q = sorted(core | sampled)
    for i in sites_1q:
        obs.append((f"X({i})", "X", [i]))
        obs.append((f"Y({i})", "Y", [i]))
        obs.append((f"Z({i})", "Z", [i]))

    # Include a small set of model-edge correlators: prefer LR vertical and periodic wraps.
    seen_pairs: set[tuple[int, int]] = set()
    for (a, b, gname), et in sorted(case.edge_types.items()):
        if gname not in {"rxx", "ryy", "rzz"}:
            continue
        if "vertical" not in et and "periodic" not in et and "random_lr" not in et:
            continue
        if (a, b) in seen_pairs:
            continue
        seen_pairs.add((a, b))
        if case.model == "ising":
            obs.append((f"ZZ_{et}({a},{b})", "ZZ", [a, b]))
        else:
            obs.append((f"XX_{et}({a},{b})", "XX", [a, b]))
            obs.append((f"YY_{et}({a},{b})", "YY", [a, b]))
            obs.append((f"ZZ_{et}({a},{b})", "ZZ", [a, b]))
        if len(seen_pairs) >= 12:
            break

    return obs


def _pauli_obs_errors(
    case: Case,
    *,
    mps: MPS,
    ref_vec: np.ndarray | None,
    ref_mps: MPS | None,
) -> tuple[float | None, float | None, float | None, str | None]:
    obs = _observable_set(case)
    if ref_vec is None and ref_mps is None:
        return None, None, None, None

    errs: list[tuple[str, float]] = []
    for name, label, sites in obs:
        got = _expectation_via_mps_swaps(mps, gate_name=label.lower(), sites=sites)
        if ref_vec is not None:
            ref = _statevector_expectation(case.n_qubits, ref_vec, label=label, sites=sites)
        else:
            assert ref_mps is not None
            ref = _expectation_via_mps_swaps(ref_mps, gate_name=label.lower(), sites=sites)
        errs.append((name, abs(got - ref)))

    worst_name, worst_val = max(errs, key=lambda x: x[1]) if errs else (None, 0.0)
    vals = np.array([v for _, v in errs], dtype=np.float64)
    return float(worst_val), float(np.mean(vals)), float(np.linalg.norm(vals)), worst_name


def _apply_two_qubit_enriched_tdvp(
    mps: MPS,
    node,
    params: StrongSimParams,
) -> None:
    gate = convert_dag_to_tensor_algorithm(node)[0]
    i, j = gate.sites
    if abs(i - j) == 1:
        apply_two_qubit_gate_tebd(mps, gate, params)
        return
    if gate.name in {"rxx", "ryy", "rzz"}:
        decision = decide_long_range_pauli_route(mps, gate, params)
        # Mirror the per-gate route metadata used by the Simulator hybrid path.
        stats_any = getattr(mps, "route_stats", None)
        if not isinstance(stats_any, dict):
            stats_any = {"tdvp_lr_pauli": 0, "enriched_lr_pauli": 0, "ratios": []}
            mps.route_stats = stats_any
        stats = stats_any
        ratios_any = stats.get("ratios")
        if not isinstance(ratios_any, list):
            ratios_any = []
            stats["ratios"] = ratios_any
        defect_tol = float(getattr(params, "tdvp_projection_defect_tol", 1e-3))
        projected_ratio = float(decision.visibility.projected_ratio)
        projection_defect = max(0.0, 1.0 - min(projected_ratio, 1.0))
        ratios_any.append(
            {
                "gate": gate.name,
                "sites": tuple(gate.sites),
                "projected_ratio": projected_ratio,
                "projection_defect": projection_defect,
                "tdvp_projection_defect_tol": defect_tol,
                "route": decision.route,
                "reason": decision.reason,
            }
        )
        if decision.route == "pauli_enriched":
            apply_pauli_product_rotation_enriched(mps, gate, params, record_stats=False)
            stats["enriched_lr_pauli"] = int(stats.get("enriched_lr_pauli", 0)) + 1
            return
        stats["tdvp_lr_pauli"] = int(stats.get("tdvp_lr_pauli", 0)) + 1
        apply_two_qubit_gate_tdvp(mps, gate, params)
        return
    apply_two_qubit_gate_tdvp(mps, gate, params)


def _run_stepwise(
    case: Case,
    *,
    method: MethodName,
    defect_tol: float | None,
    svd_threshold: float,
    max_bond_dim: int | None,
    record_bond_history: bool,
) -> tuple[MPS, float, list[BondHistoryRow], int]:
    """Run circuit gate-by-gate to allow bond-history logging."""
    params = StrongSimParams(
        preset="exact",
        get_state=True,
        svd_threshold=svd_threshold,
        max_bond_dim=max_bond_dim,
        krylov_tol=1e-12,
        gate_mode="hybrid" if method == "enriched_tdvp" else "tebd",
        tangent_blindness_tol=1e-12,
        tdvp_projection_defect_tol=1e-3 if defect_tol is None else float(defect_tol),
        tdvp_visibility_safety_tol=None,
        tdvp_pauli_consistency_check=False,
        tdvp_pauli_consistency_tol=1e-10,
    )

    st = State(case.n_qubits, initial="zeros", representation="mps")
    mps = st.mps
    dag = case.qc.to_instruction().definition  # type: ignore[assignment]
    # Fallback: DAG order from circuit itself.
    dag = None
    from qiskit.converters import circuit_to_dag

    dag = circuit_to_dag(case.qc)

    bond_rows: list[BondHistoryRow] = []
    layer = 0
    gate_index = 0
    swaps_inserted = 0

    t0 = time.perf_counter()
    for node in dag.topological_op_nodes():
        if node.op.name == "barrier":
            layer += 1
            continue
        if len(node.qargs) == 1:
            apply_single_qubit_gate(mps, node)
        elif len(node.qargs) == 2:
            gate = convert_dag_to_tensor_algorithm(node)[0]
            i, j = gate.sites
            is_lr = abs(i - j) != 1
            if method == "tebd_swaps":
                swaps_inserted += _num_swaps_for_gate(i, j)
                apply_two_qubit_gate_tebd(mps, gate, params)
            else:
                _apply_two_qubit_enriched_tdvp(mps, node, params)
        else:
            raise AssertionError(f"Unsupported op arity: {len(node.qargs)} for {node.op.name}")

        if record_bond_history and len(node.qargs) == 2:
            gate = convert_dag_to_tensor_algorithm(node)[0]
            i, j = gate.sites
            bond_rows.append(
                BondHistoryRow(
                    circuit_name=case.circuit_name,
                    method=method,
                    defect_tol=defect_tol,
                    max_bond_dim_setting=max_bond_dim,
                    layer=layer,
                    gate_index=gate_index,
                    gate_name=gate.name,
                    sites=str(tuple(gate.sites)),
                    is_long_range=abs(i - j) != 1,
                    max_bond=int(mps.get_max_bond()),
                    mean_bond=_mean_bond_dim(mps),
                    sum_chi_cubed=_sum_chi_cubed(mps),
                )
            )
        gate_index += 1

    wall = float(time.perf_counter() - t0)
    return mps, wall, bond_rows, swaps_inserted


def _run_simulator(
    qc: QuantumCircuit,
    *,
    gate_mode: str,
    svd_threshold: float,
    max_bond_dim: int | None,
    tdvp_projection_defect_tol: float | None,
) -> tuple[MPS, float]:
    params = StrongSimParams(
        preset="exact",
        get_state=True,
        svd_threshold=svd_threshold,
        max_bond_dim=max_bond_dim,
        krylov_tol=1e-12,
        gate_mode=gate_mode,
        tangent_blindness_tol=1e-12,
        tdvp_projection_defect_tol=1e-3 if tdvp_projection_defect_tol is None else float(tdvp_projection_defect_tol),
        tdvp_visibility_safety_tol=None,
        tdvp_pauli_consistency_check=False,
        tdvp_pauli_consistency_tol=1e-10,
    )
    sim = Simulator()
    init = State(qc.num_qubits, initial="zeros", representation="mps")
    t0 = time.perf_counter()
    result = sim.run(init, qc, params)
    wall = float(time.perf_counter() - t0)
    assert result.output_state is not None
    return result.output_state.mps, wall


def _build_cases() -> list[Case]:
    # Runtime knobs.
    max_cases_env = os.environ.get("YAQS_ENRICHED_VS_TEBD_MAX_CASES", "").strip()
    max_cases = None if max_cases_env in {"", "0"} else int(max_cases_env)
    if max_cases is None:
        max_cases = 24

    j = 1.0
    dt_values = [0.1]
    layers_values = [2, 4]
    initial_states: list[InitialStateName] = ["all_zero", "plus", "neel", "random_product"]

    cases: list[Case] = []

    # 2D structured (main story).
    grids = [(3, 3), (3, 4), (4, 4), (4, 5)]
    h_values = [0.5, 1.0, 2.0]
    for lx, ly in grids:
        n = lx * ly
        for h in h_values:
            for dt in dt_values:
                for layers in layers_values:
                    for init in initial_states:
                        qc, edge_types = _ising_2d_row_major(lx=lx, ly=ly, j=j, h=h, dt=dt, layers=layers, periodic_x=True)
                        prep = QuantumCircuit(n)
                        _prep_initial_state(prep, init, seed=0)
                        qc = prep.compose(qc)
                        cases.append(
                            Case(
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
                                circuit_name=f"ising2d_{lx}x{ly}_h{h:g}_dt{dt:g}_L{layers}_{init}",
                                qc=qc,
                                edge_types=edge_types,
                            )
                        )
                        if len(cases) >= max_cases:
                            return cases[:max_cases]

    for lx, ly in grids:
        n = lx * ly
        for dt in dt_values:
            for layers in layers_values:
                for init in initial_states:
                    qc, edge_types = _xxx_2d_row_major(lx=lx, ly=ly, j=j, dt=dt, layers=layers, periodic_x=True)
                    prep = QuantumCircuit(n)
                    _prep_initial_state(prep, init, seed=0)
                    qc = prep.compose(qc)
                    cases.append(
                        Case(
                            family="xxx_2d",
                            model="xxx",
                            geometry="2d",
                            boundary_condition="row_major_periodic_x",
                            lx=lx,
                            ly=ly,
                            n_qubits=n,
                            initial_state=init,
                            dt=dt,
                            layers_or_depth=layers,
                            circuit_name=f"xxx2d_{lx}x{ly}_dt{dt:g}_L{layers}_{init}",
                            qc=qc,
                            edge_types=edge_types,
                        )
                    )
                    if len(cases) >= max_cases:
                        return cases[:max_cases]

    # 1D periodic sanity.
    n_values_1d = [8, 12]
    for n in n_values_1d:
        for dt in dt_values:
            for layers in [2, 4]:
                for init in initial_states:
                    qc, edge_types = _ising_1d_periodic(n=n, j=j, h=1.0, dt=dt, layers=layers)
                    prep = QuantumCircuit(n)
                    _prep_initial_state(prep, init, seed=0)
                    qc = prep.compose(qc)
                    cases.append(
                        Case(
                            family="ising_1d_periodic",
                            model="ising",
                            geometry="1d",
                            boundary_condition="periodic",
                            lx=None,
                            ly=None,
                            n_qubits=n,
                            initial_state=init,
                            dt=dt,
                            layers_or_depth=layers,
                            circuit_name=f"ising1dP_n{n}_dt{dt:g}_L{layers}_{init}",
                            qc=qc,
                            edge_types=edge_types,
                        )
                    )
                    if len(cases) >= max_cases:
                        return cases[:max_cases]

                for init in initial_states:
                    qc, edge_types = _xxx_1d_periodic(n=n, j=j, dt=dt, layers=layers)
                    prep = QuantumCircuit(n)
                    _prep_initial_state(prep, init, seed=0)
                    qc = prep.compose(qc)
                    cases.append(
                        Case(
                            family="xxx_1d_periodic",
                            model="xxx",
                            geometry="1d",
                            boundary_condition="periodic",
                            lx=None,
                            ly=None,
                            n_qubits=n,
                            initial_state=init,
                            dt=dt,
                            layers_or_depth=layers,
                            circuit_name=f"xxx1dP_n{n}_dt{dt:g}_L{layers}_{init}",
                            qc=qc,
                            edge_types=edge_types,
                        )
                    )
                    if len(cases) >= max_cases:
                        return cases[:max_cases]

    # Random LR Pauli stress subset.
    for n in [8, 10, 12, 14]:
        for depth in [4, 8, 16]:
            for seed in range(3):
                qc, edge_types = _random_lr_pauli(n=n, depth=depth, seed=seed)
                prep = QuantumCircuit(n)
                _prep_initial_state(prep, "random_product", seed=seed)
                qc = prep.compose(qc)
                cases.append(
                    Case(
                        family="random_lr_pauli",
                        model="pauli",
                        geometry="1d",
                        boundary_condition="random",
                        lx=None,
                        ly=None,
                        n_qubits=n,
                        initial_state="random_product",
                        dt=0.0,
                        layers_or_depth=depth,
                        circuit_name=f"rand_lr_pauli_n{n}_d{depth}_s{seed}",
                        qc=qc,
                        edge_types=edge_types,
                    )
                )
                if len(cases) >= max_cases:
                    return cases[:max_cases]
    return cases[:max_cases]


def _render_md(rows: list[Row]) -> str:
    lines: list[str] = []
    lines.append("# Enriched TDVP vs TEBD+SWAP benchmark\n")
    lines.append("## Method summary\n")
    lines.append(
        "- **enriched_tdvp**: `gate_mode='hybrid'` with LR Pauli routing by projection defect "
        r"\(d = 1 - \min(\mathrm{projected\_ratio}, 1)\): TDVP if \(d \le \epsilon\), else exact Pauli-product enrichment.\n"
    )
    lines.append("- **tebd_swaps**: `gate_mode='tebd'` and LR gates handled by SWAP networks.\n")
    lines.append("Practical pass: pauli_obs_max_error <= 1e-3. Strict: fid<=1e-10 and obs_max<=1e-10 (when available).\n")

    lines.append("## Fixed bond dimension scaling (summary)\n")
    lines.append("family | n | layers | chi | method | eps | fid_err | obs_max | max_bond | swaps | wall_s\n")
    lines.append("---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:\n")
    for r in rows:
        eps = "—" if r.defect_tol is None else f"{r.defect_tol:.0e}"
        chi = "—" if r.max_bond_dim_setting is None else str(r.max_bond_dim_setting)
        fid = "—" if r.fidelity_error is None else f"{r.fidelity_error:.3e}"
        obs = "—" if r.pauli_obs_max_error is None else f"{r.pauli_obs_max_error:.3e}"
        lines.append(
            f"{r.family} | {r.n_qubits} | {r.layers} | {chi} | {r.method} | {eps} | "
            f"{fid} | {obs} | {r.max_bond_observed} | {r.num_swaps_inserted} | {r.wall_time_s:.2f}"
        )

    lines.append("\n## Route statistics (enriched_tdvp only)\n")
    lines.append("family | eps | lr_pauli | tdvp_frac | enriched_frac\n")
    lines.append("---|---:|---:|---:|---:\n")
    for r in rows:
        if r.method != "enriched_tdvp":
            continue
        eps = "—" if r.defect_tol is None else f"{r.defect_tol:.0e}"
        tdvp_frac = "—" if r.tdvp_lr_pauli_fraction is None else f"{r.tdvp_lr_pauli_fraction:.3f}"
        enr_frac = "—" if r.enriched_lr_pauli_fraction is None else f"{r.enriched_lr_pauli_fraction:.3f}"
        lines.append(f"{r.family} | {eps} | {r.num_lr_pauli_gates} | {tdvp_frac} | {enr_frac}")

    lines.append("\n## Practical accuracy failures (obs_max > 1e-3)\n")
    bad = [r for r in rows if (r.pauli_obs_max_error or 0.0) > 1e-3]
    if not bad:
        lines.append("None.\n")
    else:
        lines.append("circuit | method | chi | eps | obs_max | fid_err | worst_obs | suspect\n")
        lines.append("---|---|---:|---:|---:|---:|---|---\n")
        for r in bad:
            chi = "—" if r.max_bond_dim_setting is None else str(r.max_bond_dim_setting)
            eps = "—" if r.defect_tol is None else f"{r.defect_tol:.0e}"
            fid = "—" if r.fidelity_error is None else f"{r.fidelity_error:.3e}"
            lines.append(
                f"`{r.circuit_name}` | {r.method} | {chi} | {eps} | {r.pauli_obs_max_error:.3e} | {fid} | "
                f"{r.worst_observable} | {r.observable_ordering_suspect}"
            )

    lines.append("\n## Recommended conclusion\n")
    lines.append(
        "Use the fixed-bond rows (same `chi`) to compare error-vs-budget and wall-time. "
        "Pay special attention to 2D XXX (`xxx_2d`) where vertical edges generate many long-range "
        "RXX/RYY/RZZ gates.\n"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    out_dir = _results_dir()
    csv_path = out_dir / "enriched_tdvp_vs_tebd.csv"
    md_path = out_dir / "enriched_tdvp_vs_tebd.md"
    bond_hist_path = out_dir / "enriched_tdvp_vs_tebd_bond_history.csv"

    record_bond_history = os.environ.get("YAQS_BOND_HISTORY", "").strip() not in {"", "0", "false", "False"}

    # Benchmark regimes (exact-ish + fixed bond).
    max_bond_values = [16, 32, 64, 128, 256]
    if os.environ.get("YAQS_ENRICHED_VS_TEBD_FAST", "").strip() not in {"", "0", "false", "False"}:
        max_bond_values = [32, 64]

    defect_tols = [1e-4, 1e-6]
    cases = _build_cases()

    rows: list[Row] = []
    bond_rows: list[BondHistoryRow] = []

    for case in cases:
        # References: Qiskit for n<=14, else strict TEBD+SWAP reference if feasible for moderate sizes.
        ref_vec = _qiskit_vec(case.qc) if case.n_qubits <= 14 else None
        ref_mps = None
        ref_method = None
        if ref_vec is None and case.n_qubits <= 14:
            ref_mps, _ = _run_simulator(case.qc, gate_mode="tebd", svd_threshold=1e-12, max_bond_dim=None, tdvp_projection_defect_tol=None)
            ref_method = "tebd_swaps_strict"
        if ref_vec is not None:
            ref_method = "qiskit_statevector"

        num_1q, num_nn_2q, num_lr_pauli, swaps_est = _count_circuit_gates(case.qc)

        # Exact-ish correctness: only for small systems.
        if case.n_qubits <= 14:
            for method in ("enriched_tdvp", "tebd_swaps"):
                if method == "enriched_tdvp":
                    for eps in defect_tols:
                        mps, wall, b_hist, swaps = _run_stepwise(
                            case,
                            method="enriched_tdvp",
                            defect_tol=eps,
                            svd_threshold=1e-12,
                            max_bond_dim=None,
                            record_bond_history=record_bond_history,
                        )
                        bond_rows.extend(b_hist)
                        fid_err = None
                        if ref_vec is not None:
                            fid_err = _fid_err_vec(ref_vec, np.asarray(mps.to_vec(), dtype=np.complex128))
                        elif ref_mps is not None:
                            fid_err = _mps_fidelity_error(ref_mps, mps)

                        obs_max, obs_mean, obs_l2, worst = _pauli_obs_errors(case, mps=mps, ref_vec=ref_vec, ref_mps=ref_mps)
                        suspect = bool((fid_err is not None) and (fid_err < 1e-8) and (obs_max is not None) and (obs_max > 1e-2))
                        strict_pass = None if (fid_err is None or obs_max is None) else bool((fid_err <= 1e-10) and (obs_max <= 1e-10))
                        practical_pass = None if obs_max is None else bool(obs_max <= 1e-3)
                        tdvp_c, enr_c = _route_counts(mps)
                        tdvp_frac = None
                        enr_frac = None
                        if tdvp_c is not None and enr_c is not None and (tdvp_c + enr_c) > 0:
                            tdvp_frac = float(tdvp_c / (tdvp_c + enr_c))
                            enr_frac = float(enr_c / (tdvp_c + enr_c))
                        rows.append(
                            Row(
                                family=case.family,
                                model=case.model,
                                geometry=case.geometry,
                                boundary_condition=case.boundary_condition,
                                lx=case.lx,
                                ly=case.ly,
                                n_qubits=case.n_qubits,
                                initial_state=case.initial_state,
                                dt=case.dt,
                                layers=case.layers_or_depth,
                                circuit_name=case.circuit_name,
                                method="enriched_tdvp",
                                defect_tol=eps,
                                svd_threshold=1e-12,
                                max_bond_dim_setting=None,
                                reference_method=ref_method,
                                fidelity_error=fid_err,
                                pauli_obs_max_error=obs_max,
                                pauli_obs_mean_error=obs_mean,
                                pauli_obs_l2_error=obs_l2,
                                worst_observable=worst,
                                observable_ordering_suspect=suspect,
                                strict_pass=strict_pass,
                                practical_pass=practical_pass,
                                max_bond_observed=int(mps.get_max_bond()),
                                mean_bond_observed=_mean_bond_dim(mps),
                                sum_chi_cubed=_sum_chi_cubed(mps),
                                wall_time_s=wall,
                                num_1q_gates=num_1q,
                                num_nn_2q_gates=num_nn_2q,
                                num_lr_pauli_gates=num_lr_pauli,
                                num_swaps_inserted=0,
                                tdvp_lr_pauli_count=tdvp_c,
                                enriched_lr_pauli_count=enr_c,
                                tdvp_lr_pauli_fraction=tdvp_frac,
                                enriched_lr_pauli_fraction=enr_frac,
                                route_summary=_route_summary(mps),
                            )
                        )
                else:
                    mps, wall, b_hist, swaps = _run_stepwise(
                        case,
                        method="tebd_swaps",
                        defect_tol=None,
                        svd_threshold=1e-12,
                        max_bond_dim=None,
                        record_bond_history=record_bond_history,
                    )
                    bond_rows.extend(b_hist)
                    fid_err = None
                    if ref_vec is not None:
                        fid_err = _fid_err_vec(ref_vec, np.asarray(mps.to_vec(), dtype=np.complex128))
                    elif ref_mps is not None:
                        fid_err = _mps_fidelity_error(ref_mps, mps)
                    obs_max, obs_mean, obs_l2, worst = _pauli_obs_errors(case, mps=mps, ref_vec=ref_vec, ref_mps=ref_mps)
                    suspect = bool((fid_err is not None) and (fid_err < 1e-8) and (obs_max is not None) and (obs_max > 1e-2))
                    strict_pass = None if (fid_err is None or obs_max is None) else bool((fid_err <= 1e-10) and (obs_max <= 1e-10))
                    practical_pass = None if obs_max is None else bool(obs_max <= 1e-3)
                    rows.append(
                        Row(
                            family=case.family,
                            model=case.model,
                            geometry=case.geometry,
                            boundary_condition=case.boundary_condition,
                            lx=case.lx,
                            ly=case.ly,
                            n_qubits=case.n_qubits,
                            initial_state=case.initial_state,
                            dt=case.dt,
                            layers=case.layers_or_depth,
                            circuit_name=case.circuit_name,
                            method="tebd_swaps",
                            defect_tol=None,
                            svd_threshold=1e-12,
                            max_bond_dim_setting=None,
                            reference_method=ref_method,
                            fidelity_error=fid_err,
                            pauli_obs_max_error=obs_max,
                            pauli_obs_mean_error=obs_mean,
                            pauli_obs_l2_error=obs_l2,
                            worst_observable=worst,
                            observable_ordering_suspect=suspect,
                            strict_pass=strict_pass,
                            practical_pass=practical_pass,
                            max_bond_observed=int(mps.get_max_bond()),
                            mean_bond_observed=_mean_bond_dim(mps),
                            sum_chi_cubed=_sum_chi_cubed(mps),
                            wall_time_s=wall,
                            num_1q_gates=num_1q,
                            num_nn_2q_gates=num_nn_2q,
                            num_lr_pauli_gates=num_lr_pauli,
                            num_swaps_inserted=swaps,
                            tdvp_lr_pauli_count=None,
                            enriched_lr_pauli_count=None,
                            tdvp_lr_pauli_fraction=None,
                            enriched_lr_pauli_fraction=None,
                            route_summary="{}",
                        )
                    )

        # Fixed-bond scaling regime.
        for chi in max_bond_values:
            for eps in defect_tols:
                mps, wall, b_hist, swaps = _run_stepwise(
                    case,
                    method="enriched_tdvp",
                    defect_tol=eps,
                    svd_threshold=1e-9,
                    max_bond_dim=chi,
                    record_bond_history=record_bond_history,
                )
                bond_rows.extend(b_hist)
                fid_err = None
                if ref_vec is not None:
                    fid_err = _fid_err_vec(ref_vec, np.asarray(mps.to_vec(), dtype=np.complex128))
                elif ref_mps is not None and case.n_qubits <= 14:
                    fid_err = _mps_fidelity_error(ref_mps, mps)
                obs_max, obs_mean, obs_l2, worst = _pauli_obs_errors(case, mps=mps, ref_vec=ref_vec, ref_mps=ref_mps)
                suspect = bool((fid_err is not None) and (fid_err < 1e-8) and (obs_max is not None) and (obs_max > 1e-2))
                strict_pass = None if (fid_err is None or obs_max is None) else bool((fid_err <= 1e-10) and (obs_max <= 1e-10))
                practical_pass = None if obs_max is None else bool(obs_max <= 1e-3)
                tdvp_c, enr_c = _route_counts(mps)
                tdvp_frac = None
                enr_frac = None
                if tdvp_c is not None and enr_c is not None and (tdvp_c + enr_c) > 0:
                    tdvp_frac = float(tdvp_c / (tdvp_c + enr_c))
                    enr_frac = float(enr_c / (tdvp_c + enr_c))
                rows.append(
                    Row(
                        family=case.family,
                        model=case.model,
                        geometry=case.geometry,
                        boundary_condition=case.boundary_condition,
                        lx=case.lx,
                        ly=case.ly,
                        n_qubits=case.n_qubits,
                        initial_state=case.initial_state,
                        dt=case.dt,
                        layers=case.layers_or_depth,
                        circuit_name=case.circuit_name,
                        method="enriched_tdvp",
                        defect_tol=eps,
                        svd_threshold=1e-9,
                        max_bond_dim_setting=chi,
                        reference_method=ref_method,
                        fidelity_error=fid_err,
                        pauli_obs_max_error=obs_max,
                        pauli_obs_mean_error=obs_mean,
                        pauli_obs_l2_error=obs_l2,
                        worst_observable=worst,
                        observable_ordering_suspect=suspect,
                        strict_pass=strict_pass,
                        practical_pass=practical_pass,
                        max_bond_observed=int(mps.get_max_bond()),
                        mean_bond_observed=_mean_bond_dim(mps),
                        sum_chi_cubed=_sum_chi_cubed(mps),
                        wall_time_s=wall,
                        num_1q_gates=num_1q,
                        num_nn_2q_gates=num_nn_2q,
                        num_lr_pauli_gates=num_lr_pauli,
                        num_swaps_inserted=0,
                        tdvp_lr_pauli_count=tdvp_c,
                        enriched_lr_pauli_count=enr_c,
                        tdvp_lr_pauli_fraction=tdvp_frac,
                        enriched_lr_pauli_fraction=enr_frac,
                        route_summary=_route_summary(mps),
                    )
                )

            mps, wall, b_hist, swaps = _run_stepwise(
                case,
                method="tebd_swaps",
                defect_tol=None,
                svd_threshold=1e-9,
                max_bond_dim=chi,
                record_bond_history=record_bond_history,
            )
            bond_rows.extend(b_hist)
            fid_err = None
            if ref_vec is not None:
                fid_err = _fid_err_vec(ref_vec, np.asarray(mps.to_vec(), dtype=np.complex128))
            elif ref_mps is not None and case.n_qubits <= 14:
                fid_err = _mps_fidelity_error(ref_mps, mps)
            obs_max, obs_mean, obs_l2, worst = _pauli_obs_errors(case, mps=mps, ref_vec=ref_vec, ref_mps=ref_mps)
            suspect = bool((fid_err is not None) and (fid_err < 1e-8) and (obs_max is not None) and (obs_max > 1e-2))
            strict_pass = None if (fid_err is None or obs_max is None) else bool((fid_err <= 1e-10) and (obs_max <= 1e-10))
            practical_pass = None if obs_max is None else bool(obs_max <= 1e-3)
            rows.append(
                Row(
                    family=case.family,
                    model=case.model,
                    geometry=case.geometry,
                    boundary_condition=case.boundary_condition,
                    lx=case.lx,
                    ly=case.ly,
                    n_qubits=case.n_qubits,
                    initial_state=case.initial_state,
                    dt=case.dt,
                    layers=case.layers_or_depth,
                    circuit_name=case.circuit_name,
                    method="tebd_swaps",
                    defect_tol=None,
                    svd_threshold=1e-9,
                    max_bond_dim_setting=chi,
                    reference_method=ref_method,
                    fidelity_error=fid_err,
                    pauli_obs_max_error=obs_max,
                    pauli_obs_mean_error=obs_mean,
                    pauli_obs_l2_error=obs_l2,
                    worst_observable=worst,
                    observable_ordering_suspect=suspect,
                    strict_pass=strict_pass,
                    practical_pass=practical_pass,
                    max_bond_observed=int(mps.get_max_bond()),
                    mean_bond_observed=_mean_bond_dim(mps),
                    sum_chi_cubed=_sum_chi_cubed(mps),
                    wall_time_s=wall,
                    num_1q_gates=num_1q,
                    num_nn_2q_gates=num_nn_2q,
                    num_lr_pauli_gates=num_lr_pauli,
                    num_swaps_inserted=swaps,
                    tdvp_lr_pauli_count=None,
                    enriched_lr_pauli_count=None,
                    tdvp_lr_pauli_fraction=None,
                    enriched_lr_pauli_fraction=None,
                    route_summary="{}",
                )
            )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(Row.__annotations__.keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    md_path.write_text(_render_md(rows), encoding="utf-8")

    if record_bond_history:
        with bond_hist_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(BondHistoryRow.__annotations__.keys()))
            w.writeheader()
            for r in bond_rows:
                w.writerow(asdict(r))

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    if record_bond_history:
        print(f"Wrote {bond_hist_path}")


if __name__ == "__main__":
    main()

