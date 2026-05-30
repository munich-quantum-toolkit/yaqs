# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared helpers for TDVP benchmark scripts and debug tooling."""

from __future__ import annotations

import copy
import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector

from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.libraries.gate_library import GateLibrary
from mqt.yaqs.digital.digital_tjm import (
    apply_pauli_product_rotation_enriched,
    apply_two_qubit_gate_tebd,
    apply_two_qubit_gate_tdvp,
    decide_long_range_pauli_route,
    mps_overlap,
)
from mqt.yaqs.digital.utils.dag_utils import convert_dag_to_tensor_algorithm

if TYPE_CHECKING:
    from mqt.yaqs.core.data_structures.mps import MPS

InitialStateName = Literal["all_zero", "plus", "neel", "random_product"]


@dataclass(frozen=True)
class Case:
    """Minimal circuit case record used by debug scripts."""

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
    msg = f"Unknown initial state: {name}"
    raise AssertionError(msg)


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
            edge_types[i, k, "rzz"] = et
        for q in range(n):
            qc.rx(2 * h * dt, q)
        qc.barrier()
    return qc, edge_types


def _qiskit_vec(qc: QuantumCircuit) -> np.ndarray:
    return np.asarray(Statevector(qc).data, dtype=np.complex128)


def _fid_err_vec(a: np.ndarray, b: np.ndarray) -> float:
    return float(max(0.0, 1.0 - abs(np.vdot(a, b)) ** 2))


def _mean_bond_dim(mps: MPS) -> float:
    bonds = [int(t.shape[1]) for t in mps.tensors[1:]]
    return float(np.mean(bonds)) if bonds else 1.0


def _sum_chi_cubed(mps: MPS) -> float:
    bonds = [int(t.shape[1]) for t in mps.tensors[1:]]
    return float(sum(chi**3 for chi in bonds))


def _num_swaps_for_gate(i: int, j: int) -> int:
    """SWAP count inserted by TEBD routing for a non-adjacent (i,j) gate."""
    d = abs(i - j)
    if d <= 1:
        return 0
    return 2 * (d - 1)


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

    if n <= 20:
        sites_1q = list(range(n))
    else:
        core = {0, n - 1, n // 2, max(0, n // 2 - 1)}
        sampled = set(range(0, n, 2))
        sites_1q = sorted(core | sampled)
    for i in sites_1q:
        obs.extend(((f"X({i})", "X", [i]), (f"Y({i})", "Y", [i]), (f"Z({i})", "Z", [i])))

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
            obs.extend((
                (f"XX_{et}({a},{b})", "XX", [a, b]),
                (f"YY_{et}({a},{b})", "YY", [a, b]),
                (f"ZZ_{et}({a},{b})", "ZZ", [a, b]),
            ))
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

    worst_name, worst_val = max(errs, key=operator.itemgetter(1)) if errs else (None, 0.0)
    vals = np.array([v for _, v in errs], dtype=np.float64)
    return float(worst_val), float(np.mean(vals)), float(np.linalg.norm(vals)), worst_name


def _apply_two_qubit_enriched_tdvp(
    mps: MPS,
    node,
    params: StrongSimParams,
) -> None:
    """Stepwise hybrid_pauli-style gate apply (for debug scripts)."""
    gate = convert_dag_to_tensor_algorithm(node)[0]
    i, j = gate.sites
    if abs(i - j) == 1:
        apply_two_qubit_gate_tebd(mps, gate, params)
        return
    if gate.name in {"rxx", "ryy", "rzz"}:
        decision = decide_long_range_pauli_route(mps, gate, params)
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
        ratios_any.append({
            "gate": gate.name,
            "sites": tuple(gate.sites),
            "projected_ratio": projected_ratio,
            "projection_defect": projection_defect,
            "tdvp_projection_defect_tol": defect_tol,
            "route": decision.route,
            "reason": decision.reason,
        })
        if decision.route == "pauli_enriched":
            apply_pauli_product_rotation_enriched(mps, gate, params, record_stats=False)
            stats["enriched_lr_pauli"] = int(stats.get("enriched_lr_pauli", 0)) + 1
            return
        stats["tdvp_lr_pauli"] = int(stats.get("tdvp_lr_pauli", 0)) + 1
        apply_two_qubit_gate_tdvp(mps, gate, params)
        return
    apply_two_qubit_gate_tdvp(mps, gate, params)


def _route_counts(mps: MPS) -> tuple[int | None, int | None]:
    stats = getattr(mps, "route_stats", None)
    if not isinstance(stats, dict):
        return None, None
    tdvp = stats.get("tdvp_lr_pauli", None)
    enr = stats.get("enriched_lr_pauli", None)
    return (None if tdvp is None else int(tdvp), None if enr is None else int(enr))
