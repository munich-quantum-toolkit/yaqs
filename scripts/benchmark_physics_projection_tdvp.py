#!/usr/bin/env python
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Physics benchmark: projection-aware generator TDVP with Pauli enrichment.

Run:

    uv run python -m scripts.benchmark_physics_projection_tdvp

Outputs:
    results/physics_projection_tdvp_benchmark.csv
    results/physics_projection_tdvp_benchmark.md
"""

from __future__ import annotations

import copy
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector

from mqt.yaqs import Simulator
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.libraries.gate_library import GateLibrary
from mqt.yaqs.digital.digital_tjm import apply_two_qubit_gate_tebd, mps_overlap


MethodName = Literal[
    "adaptive_hybrid_defect_1e-4",
    "adaptive_hybrid_defect_1e-6",
    "tebd_swaps_reference",
]

InitialStateName = Literal["all_zero", "neel", "plus", "random_product"]


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
    h_or_field: float
    j: float
    dt: float
    layers: int
    circuit_name: str
    qc: QuantumCircuit
    # Optional mapping for 2q gate edge classification: (min(i,j), max(i,j), gate_name) -> edge_type.
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
    h_or_field: float
    j: float
    dt: float
    layers: int
    circuit_name: str
    method: str
    defect_tol: float | None
    reference_method: str | None
    fidelity_error: float | None
    pauli_obs_max_error: float | None
    pauli_obs_mean_error: float | None
    pauli_obs_l2_error: float | None
    worst_observable: str | None
    strict_pass: bool | None
    practical_pass: bool | None
    max_bond_dim: int
    mean_bond_dim: float
    sum_chi_cubed: float
    wall_time_s: float
    num_1q_gates: int
    num_nn_2q_gates: int
    num_lr_pauli_gates: int
    num_lr_pauli_horizontal: int
    num_lr_pauli_vertical: int
    num_lr_pauli_periodic: int
    tdvp_lr_pauli_count: int | None
    enriched_lr_pauli_count: int | None
    tdvp_lr_pauli_fraction: float | None
    enriched_lr_pauli_fraction: float | None
    route_summary: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _results_dir() -> Path:
    d = _repo_root() / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


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
    return qc, edge_types


def _site(lx: int, x: int, y: int) -> int:
    return y * lx + x


def _grid_edges(
    *,
    lx: int,
    ly: int,
    periodic_x: bool,
    periodic_y: bool,
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
    if periodic_y and ly > 2:
        for x in range(lx):
            a = _site(lx, x, ly - 1)
            b = _site(lx, x, 0)
            edges.append((a, b, "vertical_periodic_lr"))
    return edges


def _ising_2d_row_major(
    *,
    lx: int,
    ly: int,
    j: float,
    h: float,
    dt: float,
    layers: int,
    periodic_x: bool,
    periodic_y: bool,
) -> tuple[QuantumCircuit, dict[tuple[int, int, str], str]]:
    n = lx * ly
    qc = QuantumCircuit(n)
    edges = _grid_edges(lx=lx, ly=ly, periodic_x=periodic_x, periodic_y=periodic_y)
    edge_types: dict[tuple[int, int, str], str] = {}
    for _ in range(layers):
        for a, b, edge_type in edges:
            qc.rzz(2 * j * dt, a, b)
            i, k = (a, b) if a < b else (b, a)
            edge_types[(i, k, "rzz")] = edge_type
        for q in range(n):
            qc.rx(2 * h * dt, q)
    return qc, edge_types


def _xxx_2d_row_major(
    *,
    lx: int,
    ly: int,
    j: float,
    dt: float,
    layers: int,
    periodic_x: bool,
    periodic_y: bool,
) -> tuple[QuantumCircuit, dict[tuple[int, int, str], str]]:
    n = lx * ly
    qc = QuantumCircuit(n)
    edges = _grid_edges(lx=lx, ly=ly, periodic_x=periodic_x, periodic_y=periodic_y)
    edge_types: dict[tuple[int, int, str], str] = {}
    for _ in range(layers):
        for a, b, edge_type in edges:
            for name in ("rxx", "ryy", "rzz"):
                getattr(qc, name)(2 * j * dt, a, b)
                i, k = (a, b) if a < b else (b, a)
                edge_types[(i, k, name)] = edge_type
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


def _count_gates_and_lr_pauli(case: Case) -> tuple[int, int, int, int, int, int]:
    num_1q = 0
    num_nn_2q = 0
    num_lr_pauli = 0
    lr_h = 0
    lr_v = 0
    lr_p = 0
    qc = case.qc
    for ci in qc.data:
        inst = ci.operation
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
            continue
        if inst.name in {"rxx", "ryy", "rzz"}:
            num_lr_pauli += 1
            a, b = (i, j) if i < j else (j, i)
            et = case.edge_types.get((a, b, inst.name), "unknown_lr")
            if "vertical" in et:
                lr_v += 1
            elif "periodic" in et:
                lr_p += 1
            else:
                lr_h += 1
    return num_1q, num_nn_2q, num_lr_pauli, lr_h, lr_v, lr_p


def _expectation_via_mps_swaps(
    psi: MPS,
    *,
    gate_name: str,
    sites: list[int],
) -> float:
    """Compute <psi|O|psi> for 1q or 2q Pauli-product O by swapping to adjacency.

    Supports Observable gates (x,y,z,xx,yy,zz) on arbitrary pairs by:
        swap → apply adjacent observable → swap back → overlap.
    """
    obs = Observable(gate_name, sites if len(sites) == 2 else sites[0])
    if len(sites) == 1:
        ket = copy.deepcopy(psi)
        ket.apply_local(obs)
        return float(np.real(mps_overlap(psi, ket)))

    i, j = sorted([int(sites[0]), int(sites[1])])
    ket = copy.deepcopy(psi)

    # Use exact-ish TEBD for swaps: no truncation.
    sp = StrongSimParams(preset="exact", svd_threshold=1e-16, max_bond_dim=None, gate_mode="tebd", get_state=True)

    # Move logical site j left to i+1.
    for k in range(j - 1, i, -1):
        sw = GateLibrary.swap()
        sw.set_sites(k, k + 1)
        apply_two_qubit_gate_tebd(ket, sw, sp)

    ket.apply_local(Observable(gate_name, [i, i + 1]))

    # Undo swaps.
    for k in range(i + 1, j):
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

    # Single-site X/Y/Z on all sites (sample if very large).
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

    # Nearest-neighbor XX/YY/ZZ along MPS-adjacent sites.
    for i in range(n - 1):
        obs.append((f"XX({i},{i+1})", "XX", [i, i + 1]))
        obs.append((f"YY({i},{i+1})", "YY", [i, i + 1]))
        obs.append((f"ZZ({i},{i+1})", "ZZ", [i, i + 1]))

    # Periodic boundary correlators where meaningful.
    if "periodic" in case.boundary_condition and n >= 3:
        obs.append((f"XX({n-1},0)", "XX", [n - 1, 0]))
        obs.append((f"YY({n-1},0)", "YY", [n - 1, 0]))
        obs.append((f"ZZ({n-1},0)", "ZZ", [n - 1, 0]))

    # For 2D: include a small set of vertical and periodic-wrap edge correlators.
    if case.lx is not None and case.ly is not None:
        # Include correlators on each distinct edge type (sampled) to see where errors land.
        added: set[tuple[int, int]] = set()
        for (a, b, gname), et in sorted(case.edge_types.items()):
            if gname != "rzz":
                continue
            if "vertical" not in et and "periodic" not in et:
                continue
            if (a, b) in added:
                continue
            added.add((a, b))
            obs.append((f"ZZ[{et}]({a},{b})", "ZZ", [a, b]))
            if len(added) >= 8:
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


def _build_cases() -> list[Case]:
    full = os.environ.get("YAQS_PHYSICS_BENCH_FULL", "").strip() not in {"", "0", "false", "False"}
    max_cases_env = os.environ.get("YAQS_PHYSICS_BENCH_MAX_CASES", "").strip()
    max_cases = None if max_cases_env in {"", "0"} else int(max_cases_env)
    if (not full) and max_cases is None:
        # Keep runtime manageable by default; override via YAQS_PHYSICS_BENCH_MAX_CASES
        # or YAQS_PHYSICS_BENCH_FULL.
        max_cases = 48

    j = 1.0
    dt_values = [0.05, 0.1, 0.2] if full else [0.1]
    layers_values_1d = [1, 2, 4, 8, 16] if full else [1, 4, 8]
    layers_values_2d = [1, 2, 4, 8] if full else [1, 2, 4]

    # 1D periodic families (small number of LR gates per layer; sanity + route fractions).
    n_values_1d = [8, 10, 12, 14, 16, 20, 32] if full else [8, 12, 16]
    h_values_1d = [0.5, 1.0, 1.5] if full else [1.0]

    # 2D families (vertical edges become long-range under row-major MPS ordering).
    grids = [(3, 3), (3, 4), (4, 3), (4, 4)] if full else [(3, 3), (3, 4)]
    h_values_2d = [0.5, 1.0, 2.0, 3.0] if full else [1.0, 2.0]

    initial_states: list[InitialStateName] = ["all_zero", "neel", "plus", "random_product"]

    cases: list[Case] = []
    for n in n_values_1d:
        for h in h_values_1d:
            for dt in dt_values:
                for layers in layers_values_1d:
                    for init in initial_states:
                        if (not full) and init == "random_product" and n > 16:
                            continue
                        qc, edge_types = _ising_1d_periodic(n=n, j=j, h=h, dt=dt, layers=layers)
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
                                h_or_field=h,
                                j=j,
                                dt=dt,
                                layers=layers,
                                circuit_name=f"ising_1d_periodic_n{n}_h{h:g}_dt{dt:g}_L{layers}_{init}",
                                qc=qc,
                                edge_types=edge_types,
                            )
                        )

                    # XXX periodic (field-free).
                    for init in initial_states:
                        if (not full) and init == "random_product" and n > 16:
                            continue
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
                                h_or_field=0.0,
                                j=j,
                                dt=dt,
                                layers=layers,
                                circuit_name=f"xxx_1d_periodic_n{n}_dt{dt:g}_L{layers}_{init}",
                                qc=qc,
                                edge_types=edge_types,
                            )
                        )
                    if max_cases is not None and len(cases) >= max_cases:
                        return cases[:max_cases]

    # 2D row-major (periodic wraps optional; enable y wraps only for a subset to keep gate counts bounded).
    for lx, ly in grids:
        n = lx * ly
        for h in h_values_2d:
            for dt in dt_values:
                for layers in layers_values_2d:
                    for init in initial_states:
                        if (not full) and init == "random_product" and n > 12:
                            continue
                        qc, edge_types = _ising_2d_row_major(
                            lx=lx,
                            ly=ly,
                            j=j,
                            h=h,
                            dt=dt,
                            layers=layers,
                            periodic_x=True,
                            periodic_y=False,
                        )
                        prep = QuantumCircuit(n)
                        _prep_initial_state(prep, init, seed=0)
                        qc = prep.compose(qc)
                        cases.append(
                            Case(
                                family="ising_2d_row_major",
                                model="ising",
                                geometry="2d",
                                boundary_condition="periodic_x",
                                lx=lx,
                                ly=ly,
                                n_qubits=n,
                                initial_state=init,
                                h_or_field=h,
                                j=j,
                                dt=dt,
                                layers=layers,
                                circuit_name=f"ising_2d_{lx}x{ly}_px_n{n}_h{h:g}_dt{dt:g}_L{layers}_{init}",
                                qc=qc,
                                edge_types=edge_types,
                            )
                        )
                        if max_cases is not None and len(cases) >= max_cases:
                            return cases[:max_cases]

        for dt in dt_values:
            for layers in layers_values_2d:
                for init in initial_states:
                    if (not full) and init == "random_product" and n > 12:
                        continue
                    qc, edge_types = _xxx_2d_row_major(
                        lx=lx,
                        ly=ly,
                        j=j,
                        dt=dt,
                        layers=layers,
                        periodic_x=True,
                        periodic_y=False,
                    )
                    prep = QuantumCircuit(n)
                    _prep_initial_state(prep, init, seed=0)
                    qc = prep.compose(qc)
                    cases.append(
                        Case(
                            family="xxx_2d_row_major",
                            model="xxx",
                            geometry="2d",
                            boundary_condition="periodic_x",
                            lx=lx,
                            ly=ly,
                            n_qubits=n,
                            initial_state=init,
                            h_or_field=0.0,
                            j=j,
                            dt=dt,
                            layers=layers,
                            circuit_name=f"xxx_2d_{lx}x{ly}_px_n{n}_dt{dt:g}_L{layers}_{init}",
                            qc=qc,
                            edge_types=edge_types,
                        )
                    )
                    if max_cases is not None and len(cases) >= max_cases:
                        return cases[:max_cases]
    return cases


def _render_md(rows: list[Row]) -> str:
    lines: list[str] = []
    lines.append("# Physics benchmark: projection-aware generator TDVP\n")
    lines.append("## Method\n")
    lines.append(
        "Routing rule: TDVP if projection defect "
        r"\(d = 1 - \min(\mathrm{projected\_ratio}, 1)\) is <= epsilon; "
        "exact Pauli-product enrichment otherwise.\n"
    )
    lines.append("Practical pass: pauli_obs_max_error <= 1e-3.\n")
    lines.append("Strict pass: fidelity_error <= 1e-10 AND pauli_obs_max_error <= 1e-10 (when reference available).\n")

    lines.append("## Summary (grouped)\n")
    lines.append("family | geometry | boundary | init | eps | n | dt | layers | lr_pauli | tdvp_frac | obs_max | fid_err\n")
    lines.append("---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:\n")
    for r in rows:
        if not r.method.startswith("adaptive_hybrid"):
            continue
        eps = "—" if r.defect_tol is None else f"{r.defect_tol:.0e}"
        tdvp_frac = "—" if r.tdvp_lr_pauli_fraction is None else f"{r.tdvp_lr_pauli_fraction:.3f}"
        obs_max = "—" if r.pauli_obs_max_error is None else f"{r.pauli_obs_max_error:.3e}"
        fid = "—" if r.fidelity_error is None else f"{r.fidelity_error:.3e}"
        lines.append(
            f"{r.family} | {r.geometry} | {r.boundary_condition} | {r.initial_state} | {eps} | "
            f"{r.n_qubits} | {r.dt:g} | {r.layers} | {r.num_lr_pauli_gates} | {tdvp_frac} | {obs_max} | {fid}"
        )

    lines.append("\n## TDVP vs enrichment usage\n")
    lines.append("family | geometry | boundary | init | eps | lr_pauli | tdvp | enriched | tdvp_frac\n")
    lines.append("---|---|---|---|---:|---:|---:|---:|---:\n")
    for r in rows:
        if not r.method.startswith("adaptive_hybrid"):
            continue
        eps = "—" if r.defect_tol is None else f"{r.defect_tol:.0e}"
        tdvp = "—" if r.tdvp_lr_pauli_count is None else str(r.tdvp_lr_pauli_count)
        enr = "—" if r.enriched_lr_pauli_count is None else str(r.enriched_lr_pauli_count)
        tdvp_frac = "—" if r.tdvp_lr_pauli_fraction is None else f"{r.tdvp_lr_pauli_fraction:.3f}"
        lines.append(
            f"{r.family} | {r.geometry} | {r.boundary_condition} | {r.initial_state} | {eps} | "
            f"{r.num_lr_pauli_gates} | {tdvp} | {enr} | {tdvp_frac}"
        )

    lines.append("\n## Practical accuracy failures (obs_max > 1e-3)\n")
    bad = [r for r in rows if r.method.startswith("adaptive_hybrid") and (r.pauli_obs_max_error or 0.0) > 1e-3]
    if not bad:
        lines.append("None.\n")
    else:
        lines.append("circuit | method | eps | obs_max | worst_obs | tdvp_frac\n")
        lines.append("---|---|---:|---:|---|---:\n")
        for r in bad:
            eps = "—" if r.defect_tol is None else f"{r.defect_tol:.0e}"
            tdvp_frac = "—" if r.tdvp_lr_pauli_fraction is None else f"{r.tdvp_lr_pauli_fraction:.3f}"
            lines.append(
                f"`{r.circuit_name}` | {r.method} | {eps} | {r.pauli_obs_max_error:.3e} | "
                f"{r.worst_observable} | {tdvp_frac}"
            )

    lines.append("\n## Strict accuracy failures (when reference available)\n")
    strict_bad = [
        r
        for r in rows
        if r.method.startswith("adaptive_hybrid")
        and r.fidelity_error is not None
        and r.pauli_obs_max_error is not None
        and ((r.fidelity_error > 1e-10) or (r.pauli_obs_max_error > 1e-10))
    ]
    if not strict_bad:
        lines.append("None.\n")
    else:
        lines.append("circuit | method | eps | fid_err | obs_max | worst_obs\n")
        lines.append("---|---|---:|---:|---:|---\n")
        for r in strict_bad:
            eps = "—" if r.defect_tol is None else f"{r.defect_tol:.0e}"
            lines.append(
                f"`{r.circuit_name}` | {r.method} | {eps} | {r.fidelity_error:.3e} | "
                f"{r.pauli_obs_max_error:.3e} | {r.worst_observable}"
            )

    lines.append("\n## Recommended setting\n")
    lines.append(
        "Compare eps=1e-4 vs eps=1e-6: eps=1e-4 should route more LR Pauli gates through TDVP, "
        "eps=1e-6 is more conservative. Use the TDVP fractions above plus practical accuracy "
        "failures to justify the default.\n"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    cases = _build_cases()
    out_dir = _results_dir()
    csv_path = out_dir / "physics_projection_tdvp_benchmark.csv"
    md_path = out_dir / "physics_projection_tdvp_benchmark.md"

    rows: list[Row] = []
    for case in cases:
        # References:
        ref_vec = _qiskit_vec(case.qc) if case.n_qubits <= 14 else None
        ref_mps = None
        ref_method = None
        if ref_vec is None and case.n_qubits <= 20:
            # Use TEBD+SWAP reference for moderate sizes; skip for the largest cases.
            ref_mps, _ref_wall = _run_simulator(
                case.qc, gate_mode="tebd", svd_threshold=1e-12, max_bond_dim=None, tdvp_projection_defect_tol=None
            )
            ref_method = "tebd_swaps_reference"

        num_1q, num_nn_2q, num_lr_pauli, lr_h, lr_v, lr_p = _count_gates_and_lr_pauli(case)

        for method, defect_tol in (
            ("adaptive_hybrid_defect_1e-4", 1e-4),
            ("adaptive_hybrid_defect_1e-6", 1e-6),
        ):
            mps, wall = _run_simulator(
                case.qc,
                gate_mode="hybrid",
                svd_threshold=1e-12,
                max_bond_dim=None,
                tdvp_projection_defect_tol=defect_tol,
            )

            fid_err = None
            if ref_vec is not None:
                vec = np.asarray(mps.to_vec(), dtype=np.complex128)
                fid_err = _fid_err_vec(ref_vec, vec)
                reference_method = "qiskit_statevector"
            elif ref_mps is not None:
                fid_err = _mps_fidelity_error(ref_mps, mps)
                reference_method = ref_method
            else:
                reference_method = None

            obs_max, obs_mean, obs_l2, worst = _pauli_obs_errors(case, mps=mps, ref_vec=ref_vec, ref_mps=ref_mps)
            strict_pass = None
            practical_pass = None
            if fid_err is not None and obs_max is not None:
                strict_pass = bool((fid_err <= 1e-10) and (obs_max <= 1e-10))
                practical_pass = bool(obs_max <= 1e-3)

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
                    h_or_field=case.h_or_field,
                    j=case.j,
                    dt=case.dt,
                    layers=case.layers,
                    circuit_name=case.circuit_name,
                    method=method,
                    defect_tol=float(defect_tol),
                    reference_method=reference_method,
                    fidelity_error=fid_err,
                    pauli_obs_max_error=obs_max,
                    pauli_obs_mean_error=obs_mean,
                    pauli_obs_l2_error=obs_l2,
                    worst_observable=worst,
                    strict_pass=strict_pass,
                    practical_pass=practical_pass,
                    max_bond_dim=int(mps.get_max_bond()),
                    mean_bond_dim=_mean_bond_dim(mps),
                    sum_chi_cubed=_sum_chi_cubed(mps),
                    wall_time_s=wall,
                    num_1q_gates=num_1q,
                    num_nn_2q_gates=num_nn_2q,
                    num_lr_pauli_gates=num_lr_pauli,
                    num_lr_pauli_horizontal=lr_h,
                    num_lr_pauli_vertical=lr_v,
                    num_lr_pauli_periodic=lr_p,
                    tdvp_lr_pauli_count=tdvp_c,
                    enriched_lr_pauli_count=enr_c,
                    tdvp_lr_pauli_fraction=tdvp_frac,
                    enriched_lr_pauli_fraction=enr_frac,
                    route_summary=_route_summary(mps),
                )
            )

        # Also record a TEBD+SWAP row (no defect tol, no route stats).
        ref_state, wall_ref = _run_simulator(
            case.qc, gate_mode="tebd", svd_threshold=1e-12, max_bond_dim=None, tdvp_projection_defect_tol=None
        )
        obs_max_r, obs_mean_r, obs_l2_r, worst_r = _pauli_obs_errors(case, mps=ref_state, ref_vec=ref_vec, ref_mps=None)
        fid_err_r = None if ref_vec is None else _fid_err_vec(ref_vec, np.asarray(ref_state.to_vec(), dtype=np.complex128))
        strict_pass_r = None
        practical_pass_r = None
        if fid_err_r is not None and obs_max_r is not None:
            strict_pass_r = bool((fid_err_r <= 1e-10) and (obs_max_r <= 1e-10))
            practical_pass_r = bool(obs_max_r <= 1e-3)
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
                h_or_field=case.h_or_field,
                j=case.j,
                dt=case.dt,
                layers=case.layers,
                circuit_name=case.circuit_name,
                method="tebd_swaps_reference",
                defect_tol=None,
                reference_method="qiskit_statevector" if ref_vec is not None else None,
                fidelity_error=fid_err_r,
                pauli_obs_max_error=obs_max_r,
                pauli_obs_mean_error=obs_mean_r,
                pauli_obs_l2_error=obs_l2_r,
                worst_observable=worst_r,
                strict_pass=strict_pass_r,
                practical_pass=practical_pass_r,
                max_bond_dim=int(ref_state.get_max_bond()),
                mean_bond_dim=_mean_bond_dim(ref_state),
                sum_chi_cubed=_sum_chi_cubed(ref_state),
                wall_time_s=wall_ref,
                num_1q_gates=num_1q,
                num_nn_2q_gates=num_nn_2q,
                num_lr_pauli_gates=num_lr_pauli,
                num_lr_pauli_horizontal=lr_h,
                num_lr_pauli_vertical=lr_v,
                num_lr_pauli_periodic=lr_p,
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
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

