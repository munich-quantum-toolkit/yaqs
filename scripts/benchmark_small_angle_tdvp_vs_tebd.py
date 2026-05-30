#!/usr/bin/env python
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Benchmark small-angle local-generator 2TDVP vs TEBD+SWAP (no Pauli enrichment).

Run:

    uv run python -m scripts.benchmark_small_angle_tdvp_vs_tebd

Outputs:
    results/small_angle_tdvp_vs_tebd.csv
    results/small_angle_tdvp_vs_tebd.md
    results/small_angle_tdvp_vs_tebd_bond_history.csv (optional; YAQS_BOND_HISTORY=1)

Environment:
    YAQS_SMALL_ANGLE_BENCH_FULL=1     larger parameter sweep
    YAQS_SMALL_ANGLE_BENCH_MAX_CASES=N cap total (circuit, method, chi) rows
    YAQS_SMALL_ANGLE_BENCH_MAX_CIRCUITS=N cap circuits (stratified across families)
    YAQS_SMALL_ANGLE_INCLUDE_FULL_H_TDVP=1  include not-implemented full-H rows
    YAQS_BOND_HISTORY=1               write bond history CSV

Script-level routing (does not change production defaults in ``digital_tjm``):

    LR_PAULI_ROUTE = "tdvp_only"
"""

from __future__ import annotations

import csv
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.digital.digital_tjm import (
    apply_single_qubit_gate,
    apply_two_qubit_gate_tebd,
    apply_two_qubit_gate_tdvp,
    mps_overlap,
)
from mqt.yaqs.digital.utils.dag_utils import convert_dag_to_tensor_algorithm

from scripts.benchmark_enriched_tdvp_vs_tebd import (
    _expectation_via_mps_swaps,
    _fid_err_vec,
    _grid_edges,
    _mean_bond_dim,
    _num_swaps_for_gate,
    _prep_initial_state,
    _qiskit_vec,
    _statevector_expectation,
    _sum_chi_cubed,
)

MethodName = Literal["local_generator_tdvp", "tebd_swaps", "full_hamiltonian_tdvp"]
InitialStateName = Literal["all_zero", "plus", "neel", "random_product"]

BENCH_FULL = os.environ.get("YAQS_SMALL_ANGLE_BENCH_FULL", "").strip() not in {"", "0", "false", "False"}
MAX_ROWS = int(os.environ.get("YAQS_SMALL_ANGLE_BENCH_MAX_CASES", "0" if BENCH_FULL else "48"))
MAX_CIRCUITS = int(os.environ.get("YAQS_SMALL_ANGLE_BENCH_MAX_CIRCUITS", "0" if BENCH_FULL else "8"))
RECORD_BOND_HISTORY = os.environ.get("YAQS_BOND_HISTORY", "").strip() not in {"", "0", "false", "False"}
INCLUDE_FULL_H_TDVP = os.environ.get("YAQS_SMALL_ANGLE_INCLUDE_FULL_H_TDVP", "").strip() not in {
    "",
    "0",
    "false",
    "False",
}

# Internal benchmark routing: LR Pauli gates always use 2TDVP; enrichment never called.
LR_PAULI_ROUTE = "tdvp_only"

REF_LARGE_CHI = 256

DT_VALUES = [0.0005, 0.00125, 0.0025, 0.005, 0.0125, 0.025] if BENCH_FULL else [0.0025, 0.005, 0.0125, 0.025]
LAYERS_VALUES = [10, 20, 50, 100] if BENCH_FULL else [10, 20]
H_VALUES = [0.5, 1.0, 2.0] if BENCH_FULL else [0.5, 1.0]
INITIAL_STATES: list[InitialStateName] = ["all_zero", "plus", "neel", "random_product"]
GRIDS = [(3, 3), (3, 4), (4, 4), (4, 5), (5, 5)] if BENCH_FULL else [(3, 3), (3, 4), (4, 4)]
MAX_BOND_VALUES: list[int | None] = (
    [16, 32, 64, 128, 256, None] if BENCH_FULL else [16, 32, 64, 128]
)
SVD_THRESHOLD_FIXED = 1e-9
SVD_THRESHOLD_EXACT = 1e-12

# Power-law family
PL_N = [12, 16, 20, 32, 48] if BENCH_FULL else [12, 16, 20]
PL_ALPHA = [1.5, 2.0, 3.0] if BENCH_FULL else [2.0, 3.0]
PL_MAX_RANGE: list[int | None] = [3, 5, 8, None] if BENCH_FULL else [5, None]
PL_H = [0.5, 1.0]
PL_DT = [0.00125, 0.0025, 0.005, 0.0125, 0.025] if BENCH_FULL else [0.0025, 0.005, 0.0125]
PL_LAYERS = [10, 20, 50, 100] if BENCH_FULL else [10, 20]

# QAOA exploratory (small-angle only)
QAOA_GAMMA = [0.001, 0.0025, 0.005, 0.01, 0.025] if BENCH_FULL else [0.005, 0.01, 0.025]
QAOA_BETA = [0.001, 0.0025, 0.005, 0.01, 0.025] if BENCH_FULL else [0.005, 0.01]
QAOA_DEPTH = [10, 20, 50] if BENCH_FULL else [10, 20]
QAOA_N = 9 if not BENCH_FULL else 16

J_DEFAULT = 1.0


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
    theta_values: tuple[float, ...]
    extra: dict[str, Any]


@dataclass(frozen=True)
class Row:
    family: str
    model: str
    geometry: str
    n_qubits: int
    lx: int | None
    ly: int | None
    initial_state: str
    dt: float
    theta_max: float
    theta_mean: float
    layers: int
    method: str
    circuit_name: str
    svd_threshold: float
    max_bond_dim_setting: int | None
    reference_method: str | None
    fidelity_error: float | None
    observable_error_max: float | None
    observable_error_mean: float | None
    energy_error: float | None
    magnetization_error: float | None
    strict_pass: bool | None
    practical_pass: bool | None
    observable_ordering_suspect: bool
    max_bond_observed: int
    mean_bond_observed: float
    sum_chi_cubed: float
    wall_time_s: float
    num_1q_gates: int
    num_nn_2q_gates: int
    num_lr_pauli_gates: int
    num_swaps_inserted: int
    tdvp_lr_pauli_count: int
    enriched_lr_pauli_count: int
    tdvp_lr_pauli_fraction: float
    worst_observable: str | None
    status: str


@dataclass(frozen=True)
class BondHistoryRow:
    family: str
    circuit_name: str
    method: str
    max_bond_dim_setting: int | None
    layer: int
    max_bond: int
    mean_bond: float
    sum_chi_cubed: float
    wall_time_elapsed_s: float


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _results_dir() -> Path:
    d = _repo_root() / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _site(lx: int, x: int, y: int) -> int:
    return y * lx + x


def _thetas_from_qc(qc: QuantumCircuit) -> tuple[float, ...]:
    thetas: list[float] = []
    for ci in qc.data:
        if ci.operation.name in {"rxx", "ryy", "rzz"} and ci.operation.params:
            thetas.append(float(ci.operation.params[0]))
    return tuple(thetas)


def _theta_stats(thetas: tuple[float, ...]) -> tuple[float, float]:
    if not thetas:
        return 0.0, 0.0
    arr = np.array(thetas, dtype=np.float64)
    return float(np.max(arr)), float(np.mean(arr))


def _ising_2d_circuit(
    *,
    lx: int,
    ly: int,
    j: float,
    h: float,
    dt: float,
    layers: int,
) -> tuple[QuantumCircuit, dict[tuple[int, int, str], str]]:
    n = lx * ly
    qc = QuantumCircuit(n)
    edges = _grid_edges(lx=lx, ly=ly, periodic_x=False)
    edge_types: dict[tuple[int, int, str], str] = {}
    for _ in range(layers):
        for a, b, et in edges:
            theta = 2.0 * j * dt
            qc.rzz(theta, a, b)
            i, k = (a, b) if a < b else (b, a)
            edge_types[(i, k, "rzz")] = et
        for q in range(n):
            qc.rx(2.0 * h * dt, q)
        qc.barrier()
    return qc, edge_types


def _xxx_2d_circuit(
    *,
    lx: int,
    ly: int,
    j: float,
    dt: float,
    layers: int,
) -> tuple[QuantumCircuit, dict[tuple[int, int, str], str]]:
    n = lx * ly
    qc = QuantumCircuit(n)
    edges = _grid_edges(lx=lx, ly=ly, periodic_x=False)
    edge_types: dict[tuple[int, int, str], str] = {}
    for _ in range(layers):
        for a, b, et in edges:
            ang = 2.0 * j * dt
            for name in ("rxx", "ryy", "rzz"):
                getattr(qc, name)(ang, a, b)
                i, k = (a, b) if a < b else (b, a)
                edge_types[(i, k, name)] = et
        qc.barrier()
    return qc, edge_types


def _power_law_ising_circuit(
    *,
    n: int,
    alpha: float,
    max_range: int | None,
    j: float,
    h: float,
    dt: float,
    layers: int,
) -> tuple[QuantumCircuit, dict[tuple[int, int, str], str]]:
    qc = QuantumCircuit(n)
    edge_types: dict[tuple[int, int, str], str] = {}
    for _ in range(layers):
        for i in range(n):
            for j_idx in range(i + 1, n):
                sep = j_idx - i
                if max_range is not None and sep > max_range:
                    continue
                jij = j / float(sep**alpha)
                theta = 2.0 * jij * dt
                qc.rzz(theta, i, j_idx)
                edge_types[(i, j_idx, "rzz")] = "power_law_lr"
        for q in range(n):
            qc.rx(2.0 * h * dt, q)
        qc.barrier()
    return qc, edge_types


def _qaoa_layer(
    qc: QuantumCircuit,
    *,
    edges: list[tuple[int, int]],
    weights: list[float],
    gamma: float,
    beta: float,
    edge_types: dict[tuple[int, int, str], str],
    edge_tag: str,
) -> None:
    n = qc.num_qubits
    for (a, b), w in zip(edges, weights, strict=True):
        qc.rzz(2.0 * gamma * w, a, b)
        i, k = (a, b) if a < b else (b, a)
        edge_types[(i, k, "rzz")] = edge_tag
    for q in range(n):
        qc.rx(2.0 * beta, q)
    qc.barrier()


def _qaoa_circuit(
    *,
    n: int,
    edges: list[tuple[int, int]],
    weights: list[float],
    gamma: float,
    beta: float,
    depth: int,
    edge_tag: str,
) -> tuple[QuantumCircuit, dict[tuple[int, int, str], str]]:
    qc = QuantumCircuit(n)
    edge_types: dict[tuple[int, int, str], str] = {}
    for _ in range(depth):
        _qaoa_layer(qc, edges=edges, weights=weights, gamma=gamma, beta=beta, edge_types=edge_types, edge_tag=edge_tag)
    return qc, edge_types


def _qaoa_grid_circuit(
    *,
    lx: int,
    ly: int,
    gamma: float,
    beta: float,
    depth: int,
) -> tuple[QuantumCircuit, dict[tuple[int, int, str], str]]:
    n = lx * ly
    raw = _grid_edges(lx=lx, ly=ly, periodic_x=False)
    edges = [(a, b) for a, b, _ in raw]
    weights = [1.0] * len(edges)
    return _qaoa_circuit(n=n, edges=edges, weights=weights, gamma=gamma, beta=beta, depth=depth, edge_tag="grid")


def _random_geometric_edges(*, n: int, radius: float, seed: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, 1.0, size=(n, 2))
    edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if float(np.linalg.norm(pos[i] - pos[j])) <= radius:
                edges.append((i, j))
    return edges


def _power_law_chain_edges(*, n: int, alpha: float, max_range: int | None) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            sep = j - i
            if max_range is not None and sep > max_range:
                continue
            edges.append((i, j))
    return edges


def _select_cases_stratified(cases: list[Case]) -> list[Case]:
    """When capped, round-robin across families so quick runs include power-law / 2D / XXX."""
    if MAX_CIRCUITS <= 0 or len(cases) <= MAX_CIRCUITS:
        return cases
    by_family: dict[str, list[Case]] = defaultdict(list)
    for case in cases:
        by_family[case.family].append(case)
    families = sorted(by_family)
    selected: list[Case] = []
    idx = 0
    while len(selected) < MAX_CIRCUITS:
        added = False
        for fam in families:
            pool = by_family[fam]
            if idx < len(pool):
                selected.append(pool[idx])
                added = True
                if len(selected) >= MAX_CIRCUITS:
                    break
        if not added:
            break
        idx += 1
    return selected


def _make_case(
    *,
    family: str,
    model: str,
    geometry: str,
    circuit_name: str,
    qc: QuantumCircuit,
    edge_types: dict,
    lx: int | None,
    ly: int | None,
    initial_state: InitialStateName,
    dt: float,
    layers: int,
    extra: dict[str, Any] | None = None,
) -> Case:
    prep = QuantumCircuit(qc.num_qubits)
    _prep_initial_state(prep, initial_state, seed=0)
    full = prep.compose(qc)
    thetas = _thetas_from_qc(full)
    return Case(
        family=family,
        model=model,
        geometry=geometry,
        boundary_condition="open",
        lx=lx,
        ly=ly,
        n_qubits=full.num_qubits,
        initial_state=initial_state,
        dt=dt,
        layers_or_depth=layers,
        circuit_name=circuit_name,
        qc=full,
        edge_types=edge_types,
        theta_values=thetas,
        extra=extra or {},
    )


def _build_cases() -> list[Case]:
    cases: list[Case] = []

    for lx, ly in GRIDS:
        for h in H_VALUES:
            for dt in DT_VALUES:
                for layers in LAYERS_VALUES:
                    for init in INITIAL_STATES:
                        qc, et = _ising_2d_circuit(lx=lx, ly=ly, j=J_DEFAULT, h=h, dt=dt, layers=layers)
                        cases.append(
                            _make_case(
                                family="ising_2d",
                                model="ising",
                                geometry="2d_row_major",
                                circuit_name=f"ising2d_{lx}x{ly}_h{h:g}_dt{dt:g}_L{layers}_{init}",
                                qc=qc,
                                edge_types=et,
                                lx=lx,
                                ly=ly,
                                initial_state=init,
                                dt=dt,
                                layers=layers,
                            )
                        )
    for lx, ly in GRIDS:
        for dt in DT_VALUES:
            for layers in LAYERS_VALUES:
                for init in INITIAL_STATES:
                    qc, et = _xxx_2d_circuit(lx=lx, ly=ly, j=J_DEFAULT, dt=dt, layers=layers)
                    cases.append(
                        _make_case(
                            family="xxx_2d",
                            model="xxx",
                            geometry="2d_row_major",
                            circuit_name=f"xxx2d_{lx}x{ly}_dt{dt:g}_L{layers}_{init}",
                            qc=qc,
                            edge_types=et,
                            lx=lx,
                            ly=ly,
                            initial_state=init,
                            dt=dt,
                            layers=layers,
                        )
                    )
    for n in PL_N:
        for alpha in PL_ALPHA:
            for max_range in PL_MAX_RANGE:
                for h in PL_H:
                    for dt in PL_DT:
                        for layers in PL_LAYERS:
                            for init in INITIAL_STATES:
                                mr_tag = "inf" if max_range is None else str(max_range)
                                qc, et = _power_law_ising_circuit(
                                    n=n,
                                    alpha=alpha,
                                    max_range=max_range,
                                    j=J_DEFAULT,
                                    h=h,
                                    dt=dt,
                                    layers=layers,
                                )
                                cases.append(
                                    _make_case(
                                        family="power_law_ising",
                                        model="ising",
                                        geometry="1d_chain",
                                        circuit_name=(
                                            f"plising_n{n}_a{alpha:g}_R{mr_tag}_h{h:g}_dt{dt:g}_L{layers}_{init}"
                                        ),
                                        qc=qc,
                                        edge_types=et,
                                        lx=None,
                                        ly=None,
                                        initial_state=init,
                                        dt=dt,
                                        layers=layers,
                                        extra={"alpha": alpha, "max_range": max_range},
                                    )
                                )
    # Exploratory small-angle QAOA (full sweep only)
    if BENCH_FULL:
        lx, ly = 3, 3
        for gamma in QAOA_GAMMA:
            for beta in QAOA_BETA:
                for depth in QAOA_DEPTH:
                    qc, et = _qaoa_grid_circuit(lx=lx, ly=ly, gamma=gamma, beta=beta, depth=depth)
                    cases.append(
                        _make_case(
                            family="qaoa_small_angle",
                            model="qaoa",
                            geometry="2d_grid",
                            circuit_name=f"qaoa_grid_{lx}x{ly}_g{gamma:g}_b{beta:g}_d{depth}",
                            qc=qc,
                            edge_types=et,
                            lx=lx,
                            ly=ly,
                            initial_state="plus",
                            dt=gamma,
                            layers=depth,
                            extra={"gamma": gamma, "beta": beta, "graph": "grid"},
                        )
                    )
        n_geo = QAOA_N
        geo_edges = _random_geometric_edges(n=n_geo, radius=0.45, seed=7)
        geo_weights = [1.0] * len(geo_edges)
        for gamma in QAOA_GAMMA[:3]:
            for beta in QAOA_BETA[:3]:
                for depth in QAOA_DEPTH[:2]:
                    qc, et = _qaoa_circuit(
                        n=n_geo,
                        edges=geo_edges,
                        weights=geo_weights,
                        gamma=gamma,
                        beta=beta,
                        depth=depth,
                        edge_tag="geometric",
                    )
                    cases.append(
                        _make_case(
                            family="qaoa_small_angle",
                            model="qaoa",
                            geometry="random_geometric",
                            circuit_name=f"qaoa_geo_n{n_geo}_g{gamma:g}_b{beta:g}_d{depth}",
                            qc=qc,
                            edge_types=et,
                            lx=None,
                            ly=None,
                            initial_state="plus",
                            dt=gamma,
                            layers=depth,
                            extra={"gamma": gamma, "beta": beta, "graph": "geometric"},
                        )
                    )
        n_pl = min(QAOA_N, 16)
        pl_edges = _power_law_chain_edges(n=n_pl, alpha=2.0, max_range=6)
        pl_weights = [1.0 / float((j - i) ** 2) for i, j in pl_edges]
        for gamma in QAOA_GAMMA[:3]:
            for beta in QAOA_BETA[:3]:
                for depth in QAOA_DEPTH[:2]:
                    qc, et = _qaoa_circuit(
                        n=n_pl,
                        edges=pl_edges,
                        weights=pl_weights,
                        gamma=gamma,
                        beta=beta,
                        depth=depth,
                        edge_tag="power_law",
                    )
                    cases.append(
                        _make_case(
                            family="qaoa_small_angle",
                            model="qaoa",
                            geometry="power_law",
                            circuit_name=f"qaoa_pl_n{n_pl}_g{gamma:g}_b{beta:g}_d{depth}",
                            qc=qc,
                            edge_types=et,
                            lx=None,
                            ly=None,
                            initial_state="plus",
                            dt=gamma,
                            layers=depth,
                            extra={"gamma": gamma, "beta": beta, "graph": "power_law"},
                        )
                    )

    return _select_cases_stratified(cases)


def _expect_pauli(
    case: Case,
    state: MPS | np.ndarray,
    *,
    label: str,
    sites: list[int],
) -> float:
    if isinstance(state, np.ndarray):
        return _statevector_expectation(case.n_qubits, state, label=label, sites=sites)
    return _expectation_via_mps_swaps(state, gate_name=label.lower(), sites=sites)


def _sites_for_magnetization(n: int) -> list[int]:
    if n <= 24:
        return list(range(n))
    return sorted({0, n - 1, n // 2, max(0, n // 2 - 1)})


def _unique_zz_edges(case: Case) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for (a, b, gname), _et in case.edge_types.items():
        if gname != "rzz":
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _observable_errors(
    case: Case,
    mps: MPS,
    ref_vec: np.ndarray | None,
    ref_mps: MPS | None,
) -> tuple[float, float, float, float, str | None]:
    if ref_vec is None and ref_mps is None:
        return float("nan"), float("nan"), float("nan"), float("nan"), None

    ref_state: MPS | np.ndarray = ref_vec if ref_vec is not None else ref_mps  # type: ignore[assignment]
    errs: list[tuple[str, float, str]] = []

    sites_1q = _sites_for_magnetization(case.n_qubits)
    mx_got = float(np.mean([_expect_pauli(case, mps, label="X", sites=[i]) for i in sites_1q]))
    mx_ref = float(np.mean([_expect_pauli(case, ref_state, label="X", sites=[i]) for i in sites_1q]))
    mz_got = float(np.mean([_expect_pauli(case, mps, label="Z", sites=[i]) for i in sites_1q]))
    mz_ref = float(np.mean([_expect_pauli(case, ref_state, label="Z", sites=[i]) for i in sites_1q]))
    errs.append(("magnetization_x", abs(mx_got - mx_ref), "magnetization"))
    errs.append(("magnetization_z", abs(mz_got - mz_ref), "magnetization"))

    if case.family == "power_law_ising":
        alpha = float(case.extra.get("alpha", 2.0))
        w_got = 0.0
        w_ref = 0.0
        lr_pair: tuple[int, int] | None = None
        lr_sep = -1
        for a, b in _unique_zz_edges(case):
            sep = abs(b - a)
            jij = J_DEFAULT / float(sep**alpha)
            zz_g = _expect_pauli(case, mps, label="ZZ", sites=[a, b])
            zz_r = _expect_pauli(case, ref_state, label="ZZ", sites=[a, b])
            w_got += jij * zz_g
            w_ref += jij * zz_r
            if sep > lr_sep:
                lr_sep = sep
                lr_pair = (a, b)
        errs.append(("weighted_zz_energy", abs(w_got - w_ref), "energy"))
        if lr_pair is not None:
            a, b = lr_pair
            zz_g = _expect_pauli(case, mps, label="ZZ", sites=[a, b])
            zz_r = _expect_pauli(case, ref_state, label="ZZ", sites=[a, b])
            errs.append((f"ZZ_lr({a},{b})", abs(zz_g - zz_r), "energy"))
    elif case.model == "xxx":
        axis_means_got: dict[str, float] = {}
        axis_means_ref: dict[str, float] = {}
        for axis in ("XX", "YY", "ZZ"):
            pairs = [(a, b) for (a, b, g), _ in case.edge_types.items() if g == axis.lower()]
            if not pairs:
                continue
            vals_g = [_expect_pauli(case, mps, label=axis, sites=[a, b]) for a, b in pairs]
            vals_r = [_expect_pauli(case, ref_state, label=axis, sites=[a, b]) for a, b in pairs]
            axis_means_got[axis] = float(np.mean(vals_g))
            axis_means_ref[axis] = float(np.mean(vals_r))
            errs.append((f"energy_{axis.lower()}", abs(axis_means_got[axis] - axis_means_ref[axis]), "energy"))
        if axis_means_got:
            tot_g = sum(axis_means_got.values())
            tot_r = sum(axis_means_ref.values())
            errs.append(("total_energy", abs(tot_g - tot_r), "energy"))
    elif case.model == "qaoa":
        for (a, b, _) in list(case.edge_types.items())[:8]:
            zz_g = _expect_pauli(case, mps, label="ZZ", sites=[a, b])
            zz_r = _expect_pauli(case, ref_state, label="ZZ", sites=[a, b])
            errs.append((f"ZZ({a},{b})", abs(zz_g - zz_r), "energy"))
    else:
        zz_edges = _unique_zz_edges(case)
        if zz_edges:
            zz_g = float(np.mean([_expect_pauli(case, mps, label="ZZ", sites=[a, b]) for a, b in zz_edges]))
            zz_r = float(np.mean([_expect_pauli(case, ref_state, label="ZZ", sites=[a, b]) for a, b in zz_edges]))
            errs.append(("energy_zz_mean", abs(zz_g - zz_r), "energy"))
        vertical = [(a, b) for (a, b, g), et in case.edge_types.items() if g == "rzz" and "vertical" in et]
        if vertical:
            a, b = vertical[0]
            zz_g = _expect_pauli(case, mps, label="ZZ", sites=[a, b])
            zz_r = _expect_pauli(case, ref_state, label="ZZ", sites=[a, b])
            errs.append((f"ZZ_vertical({a},{b})", abs(zz_g - zz_r), "energy"))

    if not errs:
        return 0.0, 0.0, 0.0, 0.0, None

    worst_name, worst_val, _worst_cat = max(errs, key=lambda x: x[1])
    vals = np.array([e for _, e, _ in errs], dtype=np.float64)
    energy_errs = [e for _, e, c in errs if c == "energy"]
    mag_errs = [e for _, e, c in errs if c == "magnetization"]
    energy_err = float(np.max(energy_errs)) if energy_errs else 0.0
    mag_err = float(np.max(mag_errs)) if mag_errs else 0.0
    return float(worst_val), float(np.mean(vals)), energy_err, mag_err, worst_name


def _apply_two_qubit_local_generator_tdvp(
    mps: MPS,
    node,
    params: StrongSimParams,
) -> tuple[int, int, int]:
    """Apply ``LR_PAULI_ROUTE`` (tdvp_only): NN via TEBD; LR Pauli via 2TDVP; no enrichment."""
    if LR_PAULI_ROUTE != "tdvp_only":
        msg = f"Unsupported LR_PAULI_ROUTE={LR_PAULI_ROUTE!r}"
        raise ValueError(msg)
    gate = convert_dag_to_tensor_algorithm(node)[0]
    i, j = int(gate.sites[0]), int(gate.sites[1])
    swaps = 0
    if abs(i - j) == 1:
        apply_two_qubit_gate_tebd(mps, gate, params)
        return swaps, 0, 0

    if gate.name in {"rxx", "ryy", "rzz"}:
        apply_two_qubit_gate_tdvp(mps, gate, params)
        return swaps, 1, 0

    apply_two_qubit_gate_tdvp(mps, gate, params)
    return swaps, 1, 0


def _count_circuit_gates(qc: QuantumCircuit) -> tuple[int, int, int, int]:
    num_1q = 0
    num_nn = 0
    num_lr = 0
    swaps = 0
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
            num_nn += 1
        else:
            if inst.name in {"rxx", "ryy", "rzz"}:
                num_lr += 1
            swaps += _num_swaps_for_gate(i, j)
    return num_1q, num_nn, num_lr, swaps


def _run_circuit(
    case: Case,
    *,
    method: MethodName,
    svd_threshold: float,
    max_bond_dim: int | None,
) -> tuple[MPS | None, float, list[BondHistoryRow], int, dict[str, int], str]:
    """Returns (mps, wall_time, bond_history, swaps_inserted, route_counts, status)."""
    if method == "full_hamiltonian_tdvp":
        return None, 0.0, [], 0, {"tdvp_lr": 0, "enriched_lr": 0}, "not_implemented"

    params = StrongSimParams(
        preset="exact",
        get_state=True,
        svd_threshold=svd_threshold,
        max_bond_dim=max_bond_dim,
        krylov_tol=1e-12,
        gate_mode="tebd" if method == "tebd_swaps" else "hybrid",
        tdvp_projection_defect_tol=1.0,
        tangent_blindness_tol=1e-12,
        tdvp_pauli_consistency_check=False,
    )

    mps = State(case.n_qubits, initial="zeros", representation="mps").mps
    mps.route_stats = {"tdvp_lr_pauli": 0, "enriched_lr_pauli": 0, "ratios": []}

    dag = circuit_to_dag(case.qc)
    bond_rows: list[BondHistoryRow] = []
    layer = 0
    swaps_inserted = 0
    tdvp_lr = 0
    enriched_lr = 0

    t0 = time.perf_counter()
    for node in dag.topological_op_nodes():
        if node.op.name == "barrier":
            if RECORD_BOND_HISTORY:
                bond_rows.append(
                    BondHistoryRow(
                        family=case.family,
                        circuit_name=case.circuit_name,
                        method=method,
                        max_bond_dim_setting=max_bond_dim,
                        layer=layer,
                        max_bond=int(mps.get_max_bond()),
                        mean_bond=_mean_bond_dim(mps),
                        sum_chi_cubed=_sum_chi_cubed(mps),
                        wall_time_elapsed_s=float(time.perf_counter() - t0),
                    )
                )
            layer += 1
            continue

        if len(node.qargs) == 1:
            apply_single_qubit_gate(mps, node)
            continue

        gate = convert_dag_to_tensor_algorithm(node)[0]
        i, j = int(gate.sites[0]), int(gate.sites[1])
        if method == "tebd_swaps":
            swaps_inserted += _num_swaps_for_gate(i, j)
            apply_two_qubit_gate_tebd(mps, gate, params)
        else:
            sw, t_lr, e_lr = _apply_two_qubit_local_generator_tdvp(mps, node, params)
            swaps_inserted += sw
            tdvp_lr += t_lr
            enriched_lr += e_lr

    wall = float(time.perf_counter() - t0)
    return mps, wall, bond_rows, swaps_inserted, {"tdvp_lr": tdvp_lr, "enriched_lr": enriched_lr}, "ok"


def _reference_for_case(
    case: Case,
    *,
    svd_threshold: float,
) -> tuple[np.ndarray | None, MPS | None, str | None]:
    if case.n_qubits <= 14:
        return _qiskit_vec(case.qc), None, "qiskit_statevector"
    mps, _, _, _, _, status = _run_circuit(
        case,
        method="tebd_swaps",
        svd_threshold=svd_threshold,
        max_bond_dim=None,
    )
    if status == "ok" and mps is not None:
        return None, mps, "tebd_swaps_strict"
    mps_hi, _, _, _, _, status_hi = _run_circuit(
        case,
        method="local_generator_tdvp",
        svd_threshold=SVD_THRESHOLD_EXACT,
        max_bond_dim=REF_LARGE_CHI,
    )
    if status_hi == "ok" and mps_hi is not None:
        return None, mps_hi, "local_generator_tdvp_chi256"
    return None, None, None


def _row_from_run(
    case: Case,
    *,
    method: MethodName,
    svd_threshold: float,
    max_bond_dim: int | None,
    mps: MPS | None,
    wall: float,
    route_counts: dict[str, int],
    status: str,
    ref_vec: np.ndarray | None,
    ref_mps: MPS | None,
    ref_method: str | None,
) -> Row:
    theta_max, theta_mean = _theta_stats(case.theta_values)

    if status != "ok" or mps is None:
        return Row(
            family=case.family,
            model=case.model,
            geometry=case.geometry,
            n_qubits=case.n_qubits,
            lx=case.lx,
            ly=case.ly,
            initial_state=case.initial_state,
            dt=case.dt,
            theta_max=theta_max,
            theta_mean=theta_mean,
            layers=case.layers_or_depth,
            method=method,
            circuit_name=case.circuit_name,
            svd_threshold=svd_threshold,
            max_bond_dim_setting=max_bond_dim,
            reference_method=ref_method,
            fidelity_error=None,
            observable_error_max=None,
            observable_error_mean=None,
            energy_error=None,
            magnetization_error=None,
            strict_pass=None,
            practical_pass=None,
            observable_ordering_suspect=False,
            max_bond_observed=0,
            mean_bond_observed=0.0,
            sum_chi_cubed=0.0,
            wall_time_s=wall,
            num_1q_gates=0,
            num_nn_2q_gates=0,
            num_lr_pauli_gates=0,
            num_swaps_inserted=0,
            tdvp_lr_pauli_count=0,
            enriched_lr_pauli_count=0,
            tdvp_lr_pauli_fraction=0.0,
            worst_observable=None,
            status=status,
        )

    fid_err: float | None = None
    if ref_vec is not None:
        fid_err = _fid_err_vec(ref_vec, np.asarray(mps.to_vec(), dtype=np.complex128))
    elif ref_mps is not None:
        fid_err = float(
            max(
                0.0,
                1.0
                - abs(mps_overlap(ref_mps, mps)) ** 2
                / (float(np.real(mps_overlap(ref_mps, ref_mps))) * float(np.real(mps_overlap(mps, mps)))),
            )
        )

    obs_max, obs_mean, energy_err, mag_err, worst_obs = _observable_errors(case, mps, ref_vec, ref_mps)
    ordering_suspect = (
        fid_err is not None and fid_err < 1e-8 and obs_max is not None and obs_max > 1e-2
    )

    num_1q, num_nn, num_lr, swaps_est = _count_circuit_gates(case.qc)
    tdvp_n = route_counts["tdvp_lr"]
    enr_n = route_counts["enriched_lr"]
    lr_total = tdvp_n + enr_n
    tdvp_frac = float(tdvp_n / lr_total) if lr_total > 0 else 0.0

    strict = None if fid_err is None else (fid_err <= 1e-10 and obs_max <= 1e-10)
    practical = None if obs_max is None else obs_max <= 1e-3

    return Row(
        family=case.family,
        model=case.model,
        geometry=case.geometry,
        n_qubits=case.n_qubits,
        lx=case.lx,
        ly=case.ly,
        initial_state=case.initial_state,
        dt=case.dt,
        theta_max=theta_max,
        theta_mean=theta_mean,
        layers=case.layers_or_depth,
        method=method,
        circuit_name=case.circuit_name,
        svd_threshold=svd_threshold,
        max_bond_dim_setting=max_bond_dim,
        reference_method=ref_method,
        fidelity_error=fid_err,
        observable_error_max=obs_max,
        observable_error_mean=obs_mean,
        energy_error=energy_err,
        magnetization_error=mag_err,
        strict_pass=strict,
        practical_pass=practical,
        observable_ordering_suspect=ordering_suspect,
        max_bond_observed=int(mps.get_max_bond()),
        mean_bond_observed=_mean_bond_dim(mps),
        sum_chi_cubed=_sum_chi_cubed(mps),
        wall_time_s=wall,
        num_1q_gates=num_1q,
        num_nn_2q_gates=num_nn,
        num_lr_pauli_gates=num_lr,
        num_swaps_inserted=swaps_est,
        tdvp_lr_pauli_count=tdvp_n,
        enriched_lr_pauli_count=enr_n,
        tdvp_lr_pauli_fraction=tdvp_frac,
        worst_observable=worst_obs,
        status=status,
    )


def _write_csv(path: Path, rows: list[Any]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        d = row if isinstance(row, dict) else row.__dict__
        for k in d:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            d = row if isinstance(row, dict) else row.__dict__
            w.writerow({k: ("" if d.get(k) is None else d[k]) for k in keys})


def _render_md(rows: list[Row], bond_hist: list[BondHistoryRow]) -> str:
    lines: list[str] = [
        "# Small-angle local-generator TDVP vs TEBD+SWAP\n\n",
        "## Purpose\n\n",
        "This benchmark targets the regime where TDVP is expected to be useful: "
        "small-angle long-range generator evolution (θ ≤ 0.05), with **Pauli enrichment disabled**.\n\n",
        "## Methods\n\n",
        "- **local_generator_tdvp**: NN gates via TEBD/SVD; LR `rxx/ryy/rzz` via 2TDVP only "
        f"(`LR_PAULI_ROUTE={LR_PAULI_ROUTE!r}`, enrichment disabled).\n",
        "- **tebd_swaps**: all two-qubit gates via TEBD+SWAP.\n",
        "- **full_hamiltonian_tdvp**: scaffolded (not implemented in YAQS digital path).\n\n",
    ]

    ok_rows = [r for r in rows if r.status == "ok"]
    thetas = [r.theta_max for r in ok_rows]
    if thetas:
        lines.append("## Small-angle sanity check\n\n")
        lines.append(f"- θ_max across rows: **{max(thetas):.4g}** (design cap 0.05; no θ=0.2).\n")
        lines.append(
            f"- θ_mean range: {min(r.theta_mean for r in ok_rows):.4g} – "
            f"{max(r.theta_mean for r in ok_rows):.4g}\n"
        )
        n_focus = sum(1 for r in ok_rows if r.theta_max <= 0.025)
        lines.append(f"- Rows with θ_max ≤ 0.025: **{n_focus}** / {len(ok_rows)}\n\n")

    lines.append("## Accuracy vs bond dimension\n\n")
    lines.append("| family | method | max_bond | obs_max (median) | wall_s (median) | sum_χ³ (median) |\n")
    lines.append("|---|---|---:|---:|---:|---:|\n")
    groups: dict[tuple[str, str, int | None], list[Row]] = defaultdict(list)
    for r in rows:
        if r.status != "ok" or r.observable_ordering_suspect:
            continue
        groups[(r.family, r.method, r.max_bond_dim_setting)].append(r)

    for (fam, meth, chi), grp in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2] or 0)):
        obs_vals = [float(g.observable_error_max) for g in grp if g.observable_error_max is not None]
        wall_vals = [g.wall_time_s for g in grp]
        chi_vals = [g.sum_chi_cubed for g in grp]
        if not obs_vals:
            continue
        chi_s = "∞" if chi is None else str(chi)
        lines.append(
            f"| {fam} | {meth} | {chi_s} | {float(np.median(obs_vals)):.3e} | "
            f"{float(np.median(wall_vals)):.2f} | {float(np.median(chi_vals)):.1f} |\n"
        )

    lines.append("\n## Bond growth\n\n")
    if bond_hist:
        lines.append("Per-layer bond history in `small_angle_tdvp_vs_tebd_bond_history.csv`.\n\n")
        for meth in ("local_generator_tdvp", "tebd_swaps"):
            layers = [b.layer for b in bond_hist if b.method == meth]
            if not layers:
                continue
            max_bonds = [b.max_bond for b in bond_hist if b.method == meth]
            chi_cubes = [b.sum_chi_cubed for b in bond_hist if b.method == meth]
            lines.append(
                f"- `{meth}`: final-layer median max_χ={float(np.median(max_bonds)):.0f}, "
                f"median sum_χ³={float(np.median(chi_cubes)):.0f}\n"
            )
    else:
        lines.append("Bond history not recorded (set `YAQS_BOND_HISTORY=1`).\n\n")

    lines.append("## Runtime\n\n")
    for meth in ("local_generator_tdvp", "tebd_swaps"):
        walls = [r.wall_time_s for r in ok_rows if r.method == meth and not r.observable_ordering_suspect]
        if walls:
            lines.append(f"- `{meth}` median wall_time_s: **{float(np.median(walls)):.2f}**\n")

    lines.append("\n## Physics observables\n\n")
    lines.append("Rows with `observable_ordering_suspect=True` are excluded from observable conclusions.\n\n")

    lines.append("## Power-law long-range results\n\n")
    pl = [r for r in rows if r.family == "power_law_ising" and r.status == "ok" and not r.observable_ordering_suspect]
    if pl:
        for meth in ("local_generator_tdvp", "tebd_swaps"):
            vals = [r.observable_error_max for r in pl if r.method == meth and r.observable_error_max is not None]
            if vals:
                lines.append(f"- `{meth}` median obs_max on power-law cases: **{float(np.median(vals)):.3e}**\n")

    lines.append("\n## 2D results\n\n")
    d2 = [r for r in rows if r.family in {"ising_2d", "xxx_2d"} and r.status == "ok"]
    if d2:
        for meth in ("local_generator_tdvp", "tebd_swaps"):
            vals = [r.observable_error_max for r in d2 if r.method == meth and r.observable_error_max is not None]
            if vals:
                lines.append(f"- `{meth}` median obs_max on 2D cases: **{float(np.median(vals)):.3e}**\n")

    lines.append("\n## Recommended conclusion\n\n")
    lines.append(_conclusion(rows))
    return "".join(lines)


def _conclusion(rows: list[Row]) -> str:
    ok = [r for r in rows if r.status == "ok" and not r.observable_ordering_suspect]
    if not ok:
        return "No valid comparable rows.\n"

    def median_obs(method: str, family: str | None = None) -> float | None:
        subset = [r for r in ok if r.method == method and (family is None or r.family == family)]
        vals = [float(r.observable_error_max) for r in subset if r.observable_error_max is not None]
        return None if not vals else float(np.median(vals))

    def median_chi(method: str) -> float | None:
        vals = [float(r.sum_chi_cubed) for r in ok if r.method == method]
        return None if not vals else float(np.median(vals))

    def median_wall(method: str) -> float | None:
        vals = [float(r.wall_time_s) for r in ok if r.method == method]
        return None if not vals else float(np.median(vals))

    tdvp_obs = median_obs("local_generator_tdvp")
    tebd_obs = median_obs("tebd_swaps")
    tdvp_chi = median_chi("local_generator_tdvp")
    tebd_chi = median_chi("tebd_swaps")
    tdvp_wall = median_wall("local_generator_tdvp")
    tebd_wall = median_wall("tebd_swaps")
    pl_tdvp = median_obs("local_generator_tdvp", "power_law_ising")
    pl_tebd = median_obs("tebd_swaps", "power_law_ising")

    lines: list[str] = []
    if tdvp_obs is not None and tebd_obs is not None:
        if tdvp_obs < tebd_obs * 0.9:
            lines.append("- At fixed bond, **local-generator TDVP tends toward lower observable error** than TEBD+SWAP on several families.\n")
        elif tdvp_obs > tebd_obs * 1.1:
            lines.append("- At fixed bond, **TEBD+SWAP tends toward lower observable error** than local-generator TDVP.\n")
        else:
            lines.append("- Observable accuracy is **mixed / comparable** between methods at fixed bond.\n")

    if tdvp_chi is not None and tebd_chi is not None and tdvp_chi < tebd_chi:
        lines.append("- TDVP often shows **lower sum_χ³ (memory proxy)** than TEBD+SWAP.\n")
    if tdvp_wall is not None and tebd_wall is not None and tdvp_wall < tebd_wall * 0.9:
        lines.append("- TDVP is **faster** than TEBD on a subset of deep/long-range cases.\n")
    elif tdvp_wall is not None and tebd_wall is not None and tdvp_wall > tebd_wall * 1.1:
        lines.append("- TEBD+SWAP is **faster** than local-generator TDVP in this sweep.\n")

    if pl_tdvp is not None and pl_tebd is not None:
        lines.append(
            f"- Power-law long-range Ising: median obs_max TDVP={pl_tdvp:.3e}, TEBD={pl_tebd:.3e}.\n"
        )

    enr = sum(r.enriched_lr_pauli_count for r in ok if r.method == "local_generator_tdvp")
    if enr == 0:
        lines.append("- **enriched_lr_pauli_count = 0** for local_generator_tdvp (enrichment disabled as intended).\n")
    else:
        lines.append(f"- WARNING: enriched_lr_pauli_count={enr} (should be 0).\n")

    lines.append("- **full_hamiltonian_tdvp** is not implemented; do not compare until available.\n")
    return "".join(lines)


def main() -> None:
    out_dir = _results_dir()
    csv_path = out_dir / "small_angle_tdvp_vs_tebd.csv"
    md_path = out_dir / "small_angle_tdvp_vs_tebd.md"
    bond_path = out_dir / "small_angle_tdvp_vs_tebd_bond_history.csv"

    cases = _build_cases()
    methods: list[MethodName] = ["local_generator_tdvp", "tebd_swaps"]
    if INCLUDE_FULL_H_TDVP:
        methods.append("full_hamiltonian_tdvp")

    rows: list[Row] = []
    bond_rows: list[BondHistoryRow] = []
    row_budget = MAX_ROWS if MAX_ROWS > 0 else None

    done = False
    for case in cases:
        if done:
            break
        ref_vec, ref_mps, ref_method = _reference_for_case(case, svd_threshold=SVD_THRESHOLD_EXACT)

        for max_bond in MAX_BOND_VALUES:
            if done:
                break
            svd = SVD_THRESHOLD_EXACT if max_bond is None else SVD_THRESHOLD_FIXED
            for method in methods:
                if row_budget is not None and len(rows) >= row_budget:
                    done = True
                    break
                mps, wall, bhist, swaps, routes, status = _run_circuit(
                    case,
                    method=method,
                    svd_threshold=svd,
                    max_bond_dim=max_bond,
                )
                bond_rows.extend(bhist)
                rows.append(
                    _row_from_run(
                        case,
                        method=method,
                        svd_threshold=svd,
                        max_bond_dim=max_bond,
                        mps=mps,
                        wall=wall,
                        route_counts=routes,
                        status=status,
                        ref_vec=ref_vec,
                        ref_mps=ref_mps,
                        ref_method=ref_method,
                    )
                )
                print(
                    f"[{len(rows)}] {case.circuit_name} {method} chi={max_bond} "
                    f"status={status} obs={rows[-1].observable_error_max}"
                )

    _write_csv(csv_path, rows)
    if RECORD_BOND_HISTORY:
        _write_csv(bond_path, bond_rows)
    md_path.write_text(_render_md(rows, bond_rows), encoding="utf-8")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")
    if RECORD_BOND_HISTORY:
        print(f"Wrote {bond_path}")


if __name__ == "__main__":
    main()
