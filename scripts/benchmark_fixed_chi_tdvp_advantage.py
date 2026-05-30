#!/usr/bin/env python
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Fixed-bond advantage benchmark: local-generator TDVP vs TEBD+SWAP (power-law Ising focus).

Run:

    uv run python -m scripts.benchmark_fixed_chi_tdvp_advantage

Outputs:
    results/fixed_chi_tdvp_advantage.csv
    results/fixed_chi_tdvp_advantage_summary.csv
    results/fixed_chi_tdvp_advantage.md
    results/fixed_chi_tdvp_advantage_bond_history.csv (YAQS_BOND_HISTORY=1)

Environment:
    YAQS_FIXED_CHI_STAGE=0|1|2|3   benchmark stage (default 0)
    YAQS_FIXED_CHI_PROFILE=stage1_plus  narrow Stage-1 plus-state power-law grid
    YAQS_FIXED_CHI_MAX_RANGE_ONLY=8|none  restrict max_range (8 first, then none)
    YAQS_FIXED_CHI_MAX_CASES=N     cap circuit configs (0 = no cap)
    YAQS_BOND_HISTORY=1            per-layer bond CSV
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
    apply_pauli_product_rotation_enriched,
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
from scripts.benchmark_small_angle_tdvp_vs_tebd import (
    LR_PAULI_ROUTE,
    _apply_two_qubit_local_generator_tdvp,
    _ising_2d_circuit,
    _power_law_ising_circuit,
    _xxx_2d_circuit,
)

MethodName = Literal["local_generator_tdvp", "tebd_swaps", "exact_pauli_reference"]
InitialStateName = Literal["plus", "random_product", "all_zero", "neel"]

STAGE = int(os.environ.get("YAQS_FIXED_CHI_STAGE", "0"))
PROFILE = os.environ.get("YAQS_FIXED_CHI_PROFILE", "").strip()
MAX_RANGE_ONLY = os.environ.get("YAQS_FIXED_CHI_MAX_RANGE_ONLY", "").strip().lower()
MAX_CASES = int(os.environ.get("YAQS_FIXED_CHI_MAX_CASES", "0"))
RECORD_BOND_HISTORY = os.environ.get("YAQS_BOND_HISTORY", "").strip() not in {"", "0", "false", "False"}

SVD_THRESHOLD = 1e-9
SVD_THRESHOLD_REF = 1e-12
J_DEFAULT = 1.0

RECOVERY_TDVP_CHIS = (16, 24, 32)

STAGE_CONFIGS: dict[int, dict[str, Any]] = {
    0: {
        "families": ["power_law_ising"],
        "n_values": [12],
        "alpha_values": [1.5, 2.0],
        "max_range_values": [5, 8],
        "h_values": [0.5],
        "dt_values": [0.00125, 0.0025, 0.005],
        "layers_values": [10, 20],
        "initial_states": ["plus", "random_product"],
        "max_bond_dim_values": [8, 12, 16, 24, 32, 48, 64],
        "grids": [],
        "reference": "qiskit",
    },
    1: {
        "families": ["power_law_ising"],
        "n_values": [16, 20],
        "alpha_values": [1.5, 2.0, 3.0],
        "max_range_values": [5, 8, None],
        "h_values": [0.5, 1.0],
        "dt_values": [0.00125, 0.0025, 0.005],
        "layers_values": [20, 50],
        "initial_states": ["plus", "random_product"],
        "max_bond_dim_values": [8, 12, 16, 24, 32, 48, 64, 96, 128],
        "grids": [],
        "reference": "strict_tebd",
    },
    2: {
        "families": ["power_law_ising"],
        "n_values": [20, 32],
        "alpha_values": [1.5, 2.0],
        "max_range_values": [8, None],
        "h_values": [0.5],
        "dt_values": [0.00125, 0.0025],
        "layers_values": [50, 100],
        "initial_states": ["plus"],
        "max_bond_dim_values": [16, 24, 32, 48, 64, 96, 128],
        "grids": [],
        "reference": "strict_tebd",
    },
    3: {
        "families": ["ising_2d", "xxx_2d"],
        "n_values": [],
        "alpha_values": [],
        "max_range_values": [],
        "h_values": [0.5],
        "dt_values": [0.00125, 0.0025, 0.005],
        "layers_values": [20, 50],
        "initial_states": ["plus", "random_product"],
        "max_bond_dim_values": [16, 32, 64, 128],
        "grids": [(3, 4), (4, 4)],
        "reference": "strict_tebd",
    },
}

# Narrow Stage-1 grid: power-law Ising, plus state only (Stage 0 cleanest TDVP-advantage regime).
PROFILE_CONFIGS: dict[str, dict[str, Any]] = {
    "stage1_plus": {
        "families": ["power_law_ising"],
        "n_values": [16, 20],
        "alpha_values": [1.5, 2.0],
        "max_range_values": [8, None],
        "h_values": [0.5],
        "dt_values": [0.00125, 0.0025],
        "layers_values": [20, 50],
        "initial_states": ["plus"],
        "max_bond_dim_values": [16, 24, 32, 48, 64, 96, 128],
        "grids": [],
        "reference": "strict_tebd",
    },
}


def _active_config(stage: int) -> dict[str, Any]:
    cfg = dict(STAGE_CONFIGS[stage])
    if PROFILE in PROFILE_CONFIGS:
        cfg.update(PROFILE_CONFIGS[PROFILE])
    if MAX_RANGE_ONLY in {"8", "r8", "max8"}:
        cfg["max_range_values"] = [8]
    elif MAX_RANGE_ONLY in {"none", "inf", "all", "null"}:
        cfg["max_range_values"] = [None]
    return cfg


def _output_stem() -> str:
    if PROFILE == "stage1_plus":
        stem = "fixed_chi_tdvp_advantage_stage1_plus"
        if MAX_RANGE_ONLY in {"8", "r8", "max8"}:
            return f"{stem}_mr8"
        if MAX_RANGE_ONLY in {"none", "inf", "all", "null"}:
            return f"{stem}_mrinf"
        return stem
    return "fixed_chi_tdvp_advantage"


@dataclass(frozen=True)
class Case:
    stage: int
    family: str
    model: str
    circuit_name: str
    circuit_key: str
    n_qubits: int
    alpha: float | None
    max_range: int | None
    h: float
    dt: float
    layers: int
    initial_state: str
    qc: QuantumCircuit
    edge_types: dict[tuple[int, int, str], str]
    theta_max: float
    theta_mean: float
    theta_min_nonzero: float
    num_lr_pauli_gates: int
    num_nn_2q_gates: int
    num_zz_terms: int


@dataclass
class ObsMetrics:
    fidelity_error: float | None
    weighted_zz_energy_error: float
    energy_density_error: float
    magnetization_x_error: float
    magnetization_z_error: float
    long_range_corr_error_max: float
    observable_error_max: float
    observable_error_mean: float
    energy_error: float
    practical_pass_1e3: bool
    practical_pass_1e2: bool


@dataclass(frozen=True)
class Row:
    stage: int
    family: str
    model: str
    n_qubits: int
    alpha: float | None
    max_range: int | None
    initial_state: str
    dt: float
    theta_max: float
    theta_mean: float
    theta_min_nonzero: float
    layers: int
    method: str
    circuit_name: str
    circuit_key: str
    svd_threshold: float
    max_bond_dim_setting: int | None
    reference_method: str | None
    fidelity_error: float | None
    observable_error_max: float | None
    observable_error_mean: float | None
    energy_error: float | None
    energy_density_error: float | None
    magnetization_x_error: float | None
    magnetization_z_error: float | None
    long_range_corr_error_max: float | None
    practical_pass_1e3: bool
    practical_pass_1e2: bool
    max_bond_observed: int
    mean_bond_observed: float
    sum_chi_cubed: float
    wall_time_s: float
    num_swaps_inserted: int
    num_lr_pauli_gates: int
    num_nn_2q_gates: int
    tdvp_lr_pauli_count: int
    enriched_lr_pauli_count: int
    tdvp_lr_pauli_fraction: float
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


@dataclass(frozen=True)
class SummaryRow:
    stage: int
    circuit_name: str
    circuit_key: str
    tdvp_chi: int
    tdvp_energy_density_error: float | None
    tdvp_observable_error_max: float | None
    tebd_same_chi_energy_density_error: float | None
    tebd_same_chi_observable_error_max: float | None
    energy_error_ratio_tebd_over_tdvp: float | None
    obs_error_ratio_tebd_over_tdvp: float | None
    tebd_recovery_chi_energy: str
    tebd_recovery_chi_observable: str
    recovery_ratio_energy: float | None
    recovery_ratio_observable: float | None
    tdvp_max_layers_practical_1e3: int | None
    tebd_max_layers_practical_1e3: int | None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _results_dir() -> Path:
    d = _repo_root() / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _theta_stats(qc: QuantumCircuit) -> tuple[float, float, float]:
    thetas = [
        float(ci.operation.params[0])
        for ci in qc.data
        if ci.operation.name in {"rxx", "ryy", "rzz"} and ci.operation.params
    ]
    if not thetas:
        return 0.0, 0.0, 0.0
    arr = np.array(thetas, dtype=np.float64)
    nz = arr[arr > 0]
    theta_min = float(np.min(nz)) if len(nz) else 0.0
    return float(np.max(arr)), float(np.mean(arr)), theta_min


def _count_gates(qc: QuantumCircuit) -> tuple[int, int, int]:
    num_nn = 0
    num_lr = 0
    swaps = 0
    for ci in qc.data:
        if ci.operation.name == "barrier" or len(ci.qubits) != 2:
            continue
        i = qc.find_bit(ci.qubits[0]).index
        j = qc.find_bit(ci.qubits[1]).index
        if ci.operation.name in {"rxx", "ryy", "rzz"}:
            if abs(i - j) == 1:
                num_nn += 1
            else:
                num_lr += 1
                swaps += _num_swaps_for_gate(i, j)
    return num_nn, num_lr, swaps


def _zz_term_list(n: int, alpha: float, max_range: int | None) -> list[tuple[int, int, float]]:
    terms: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            sep = j - i
            if max_range is not None and sep > max_range:
                continue
            jij = J_DEFAULT / float(sep**alpha)
            terms.append((i, j, jij))
    return terms


def _long_range_pairs(n: int) -> list[tuple[int, int]]:
    pairs = [(0, n // 2), (0, n - 1), (n // 4, (3 * n) // 4)]
    out: list[tuple[int, int]] = []
    for a, b in pairs:
        if a == b or a < 0 or b < 0 or a >= n or b >= n:
            continue
        out.append((min(a, b), max(a, b)))
    return out


def _make_power_law_case(
    *,
    stage: int,
    n: int,
    alpha: float,
    max_range: int | None,
    h: float,
    dt: float,
    layers: int,
    initial_state: InitialStateName,
) -> Case:
    qc_body, edge_types = _power_law_ising_circuit(
        n=n, alpha=alpha, max_range=max_range, j=J_DEFAULT, h=h, dt=dt, layers=layers
    )
    prep = QuantumCircuit(n)
    _prep_initial_state(prep, initial_state, seed=0)
    qc = prep.compose(qc_body)
    theta_max, theta_mean, theta_min = _theta_stats(qc)
    num_nn, num_lr, _ = _count_gates(qc)
    mr_tag = "inf" if max_range is None else str(max_range)
    name = f"pl_s{stage}_n{n}_a{alpha:g}_R{mr_tag}_h{h:g}_dt{dt:g}_L{layers}_{initial_state}"
    key = f"pl_n{n}_a{alpha:g}_R{mr_tag}_h{h:g}_dt{dt:g}_{initial_state}"
    terms = _zz_term_list(n, alpha, max_range)
    return Case(
        stage=stage,
        family="power_law_ising",
        model="ising",
        circuit_name=name,
        circuit_key=key,
        n_qubits=n,
        alpha=alpha,
        max_range=max_range,
        h=h,
        dt=dt,
        layers=layers,
        initial_state=initial_state,
        qc=qc,
        edge_types=edge_types,
        theta_max=theta_max,
        theta_mean=theta_mean,
        theta_min_nonzero=theta_min,
        num_lr_pauli_gates=num_lr,
        num_nn_2q_gates=0,
        num_zz_terms=len(terms),
    )


def _make_2d_case(
    *,
    stage: int,
    family: str,
    lx: int,
    ly: int,
    h: float,
    dt: float,
    layers: int,
    initial_state: InitialStateName,
) -> Case:
    if family == "ising_2d":
        qc_body, edge_types = _ising_2d_circuit(lx=lx, ly=ly, j=J_DEFAULT, h=h, dt=dt, layers=layers)
        model = "ising"
    else:
        qc_body, edge_types = _xxx_2d_circuit(lx=lx, ly=ly, j=J_DEFAULT, dt=dt, layers=layers)
        model = "xxx"
    n = lx * ly
    prep = QuantumCircuit(n)
    _prep_initial_state(prep, initial_state, seed=0)
    qc = prep.compose(qc_body)
    theta_max, theta_mean, theta_min = _theta_stats(qc)
    num_nn, num_lr, _ = _count_gates(qc)
    name = f"{family}_s{stage}_{lx}x{ly}_h{h:g}_dt{dt:g}_L{layers}_{initial_state}"
    key = f"{family}_{lx}x{ly}_h{h:g}_dt{dt:g}_{initial_state}"
    return Case(
        stage=stage,
        family=family,
        model=model,
        circuit_name=name,
        circuit_key=key,
        n_qubits=n,
        alpha=None,
        max_range=None,
        h=h,
        dt=dt,
        layers=layers,
        initial_state=initial_state,
        qc=qc,
        edge_types=edge_types,
        theta_max=theta_max,
        theta_mean=theta_mean,
        theta_min_nonzero=theta_min,
        num_lr_pauli_gates=num_lr,
        num_nn_2q_gates=num_nn,
        num_zz_terms=0,
    )


def _build_cases(stage: int, cfg: dict[str, Any]) -> list[Case]:
    cases: list[Case] = []

    if "power_law_ising" in cfg["families"]:
        for n in cfg["n_values"]:
            for alpha in cfg["alpha_values"]:
                for max_range in cfg["max_range_values"]:
                    for h in cfg["h_values"]:
                        for dt in cfg["dt_values"]:
                            for layers in cfg["layers_values"]:
                                for init in cfg["initial_states"]:
                                    cases.append(
                                        _make_power_law_case(
                                            stage=stage,
                                            n=n,
                                            alpha=alpha,
                                            max_range=max_range,
                                            h=h,
                                            dt=dt,
                                            layers=layers,
                                            initial_state=init,
                                        )
                                    )

    for family in ("ising_2d", "xxx_2d"):
        if family not in cfg["families"]:
            continue
        for lx, ly in cfg["grids"]:
            for h in cfg["h_values"]:
                for dt in cfg["dt_values"]:
                    for layers in cfg["layers_values"]:
                        for init in cfg["initial_states"]:
                            cases.append(
                                _make_2d_case(
                                    stage=stage,
                                    family=family,
                                    lx=lx,
                                    ly=ly,
                                    h=h,
                                    dt=dt,
                                    layers=layers,
                                    initial_state=init,
                                )
                            )

    if MAX_CASES > 0 and len(cases) > MAX_CASES:
        return cases[:MAX_CASES]
    return cases


def _expect(
    case: Case,
    state: MPS | np.ndarray,
    *,
    label: str,
    sites: list[int],
) -> float:
    if isinstance(state, np.ndarray):
        return _statevector_expectation(case.n_qubits, state, label=label, sites=sites)
    return _expectation_via_mps_swaps(state, gate_name=label.lower(), sites=sites)


def _obs_metrics(case: Case, mps: MPS, ref: MPS | np.ndarray, *, fid_err: float | None) -> ObsMetrics:
    n = case.n_qubits
    sites = list(range(n)) if n <= 32 else sorted({0, n - 1, n // 2, max(0, n // 2 - 1)})

    mx_g = float(np.mean([_expect(case, mps, label="X", sites=[i]) for i in sites]))
    mx_r = float(np.mean([_expect(case, ref, label="X", sites=[i]) for i in sites]))
    mz_g = float(np.mean([_expect(case, mps, label="Z", sites=[i]) for i in sites]))
    mz_r = float(np.mean([_expect(case, ref, label="Z", sites=[i]) for i in sites]))
    mx_err = abs(mx_g - mx_r)
    mz_err = abs(mz_g - mz_r)

    w_g = 0.0
    w_r = 0.0
    if case.family == "power_law_ising" and case.alpha is not None:
        for i, j, jij in _zz_term_list(n, case.alpha, case.max_range):
            w_g += jij * _expect(case, mps, label="ZZ", sites=[i, j])
            w_r += jij * _expect(case, ref, label="ZZ", sites=[i, j])
        n_terms = max(case.num_zz_terms, 1)
    else:
        pairs = sorted({(min(a, b), max(a, b)) for (a, b, g), _ in case.edge_types.items() if g == "rzz"})
        for a, b in pairs:
            w_g += _expect(case, mps, label="ZZ", sites=[a, b])
            w_r += _expect(case, ref, label="ZZ", sites=[a, b])
        n_terms = max(len(pairs), 1)

    w_err = abs(w_g - w_r)
    dens_err = abs(w_g / n_terms - w_r / n_terms)

    lr_errs: list[float] = []
    if case.family == "power_law_ising":
        for a, b in _long_range_pairs(n):
            lr_errs.append(abs(_expect(case, mps, label="ZZ", sites=[a, b]) - _expect(case, ref, label="ZZ", sites=[a, b])))
    lr_max = float(max(lr_errs)) if lr_errs else 0.0

    per_obs = [w_err, dens_err, mx_err, mz_err, lr_max]
    obs_max = float(max(per_obs))
    obs_mean = float(np.mean(per_obs))

    pass_1e3 = dens_err <= 1e-3 and obs_max <= 1e-3
    pass_1e2 = dens_err <= 1e-2 and obs_max <= 1e-2

    return ObsMetrics(
        fidelity_error=fid_err,
        weighted_zz_energy_error=w_err,
        energy_density_error=dens_err,
        magnetization_x_error=mx_err,
        magnetization_z_error=mz_err,
        long_range_corr_error_max=lr_max,
        observable_error_max=obs_max,
        observable_error_mean=obs_mean,
        energy_error=w_err,
        practical_pass_1e3=pass_1e3,
        practical_pass_1e2=pass_1e2,
    )


def _run_circuit(
    case: Case,
    *,
    method: MethodName,
    max_bond_dim: int | None,
    svd_threshold: float = SVD_THRESHOLD,
) -> tuple[MPS | None, float, list[BondHistoryRow], int, dict[str, int], str]:
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
    dag = circuit_to_dag(case.qc)
    bond_rows: list[BondHistoryRow] = []
    layer = 0
    swaps = 0
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
            swaps += _num_swaps_for_gate(i, j)
            apply_two_qubit_gate_tebd(mps, gate, params)
        elif method == "exact_pauli_reference":
            if gate.name in {"rxx", "ryy", "rzz"}:
                apply_pauli_product_rotation_enriched(mps, gate, params, record_stats=False)
                if abs(i - j) != 1:
                    enriched_lr += 1
            else:
                apply_two_qubit_gate_tebd(mps, gate, params)
        else:
            sw, t_lr, e_lr = _apply_two_qubit_local_generator_tdvp(mps, node, params)
            swaps += sw
            tdvp_lr += t_lr
            enriched_lr += e_lr

    wall = float(time.perf_counter() - t0)
    routes = {"tdvp_lr": tdvp_lr, "enriched_lr": enriched_lr}
    return mps, wall, bond_rows, swaps, routes, "ok"


def _resolve_reference(
    case: Case,
    *,
    stage_cfg: dict[str, Any],
) -> tuple[MPS | np.ndarray | None, str | None]:
    ref_pref = stage_cfg.get("reference", "strict_tebd")

    if case.n_qubits <= 14 and ref_pref == "qiskit":
        return _qiskit_vec(case.qc), "qiskit_statevector"

    candidates: list[tuple[str, MPS | np.ndarray]] = []

    mps_tebd, _, _, _, _, st = _run_circuit(
        case, method="tebd_swaps", max_bond_dim=None, svd_threshold=SVD_THRESHOLD_REF
    )
    if st == "ok" and mps_tebd is not None:
        candidates.append(("strict_tebd_reference", mps_tebd))

    mps_enr, _, _, _, _, st_e = _run_circuit(
        case, method="exact_pauli_reference", max_bond_dim=None, svd_threshold=SVD_THRESHOLD_REF
    )
    if st_e == "ok" and mps_enr is not None:
        candidates.append(("optional_exact_pauli_reference", mps_enr))

    if not candidates:
        return None, None

    if len(candidates) == 1:
        return candidates[0][1], candidates[0][0]

    if case.n_qubits <= 14:
        qref = _qiskit_vec(case.qc)
        best_name = candidates[0][0]
        best_state = candidates[0][1]
        best_score = float("inf")
        for name, state in candidates:
            assert isinstance(state, MPS)
            m = _obs_metrics(case, state, qref, fid_err=None)
            score = m.observable_error_max
            if score < best_score:
                best_score = score
                best_name = name
                best_state = state
        return best_state, f"best_available_reference({best_name})"

    return candidates[0][1], candidates[0][0]


def _row_from_run(
    case: Case,
    *,
    method: MethodName,
    max_bond_dim: int | None,
    mps: MPS | None,
    wall: float,
    routes: dict[str, int],
    status: str,
    ref: MPS | np.ndarray | None,
    ref_method: str | None,
) -> Row:
    if status != "ok" or mps is None or ref is None:
        return Row(
            stage=case.stage,
            family=case.family,
            model=case.model,
            n_qubits=case.n_qubits,
            alpha=case.alpha,
            max_range=case.max_range,
            initial_state=case.initial_state,
            dt=case.dt,
            theta_max=case.theta_max,
            theta_mean=case.theta_mean,
            theta_min_nonzero=case.theta_min_nonzero,
            layers=case.layers,
            method=method,
            circuit_name=case.circuit_name,
            circuit_key=case.circuit_key,
            svd_threshold=SVD_THRESHOLD,
            max_bond_dim_setting=max_bond_dim,
            reference_method=ref_method,
            fidelity_error=None,
            observable_error_max=None,
            observable_error_mean=None,
            energy_error=None,
            energy_density_error=None,
            magnetization_x_error=None,
            magnetization_z_error=None,
            long_range_corr_error_max=None,
            practical_pass_1e3=False,
            practical_pass_1e2=False,
            max_bond_observed=0,
            mean_bond_observed=0.0,
            sum_chi_cubed=0.0,
            wall_time_s=wall,
            num_swaps_inserted=0,
            num_lr_pauli_gates=case.num_lr_pauli_gates,
            num_nn_2q_gates=case.num_nn_2q_gates,
            tdvp_lr_pauli_count=0,
            enriched_lr_pauli_count=0,
            tdvp_lr_pauli_fraction=0.0,
            status=status,
        )

    fid: float | None = None
    if isinstance(ref, np.ndarray):
        fid = _fid_err_vec(ref, np.asarray(mps.to_vec(), dtype=np.complex128))
    else:
        denom = float(np.real(mps_overlap(ref, ref))) * float(np.real(mps_overlap(mps, mps)))
        if denom > 0:
            fid = float(max(0.0, 1.0 - abs(mps_overlap(ref, mps)) ** 2 / denom))

    metrics = _obs_metrics(case, mps, ref, fid_err=fid)
    lr_total = routes["tdvp_lr"] + routes["enriched_lr"]
    tdvp_frac = float(routes["tdvp_lr"] / lr_total) if lr_total > 0 else 0.0
    _, _, swaps_est = _count_gates(case.qc)

    return Row(
        stage=case.stage,
        family=case.family,
        model=case.model,
        n_qubits=case.n_qubits,
        alpha=case.alpha,
        max_range=case.max_range,
        initial_state=case.initial_state,
        dt=case.dt,
        theta_max=case.theta_max,
        theta_mean=case.theta_mean,
        theta_min_nonzero=case.theta_min_nonzero,
        layers=case.layers,
        method=method,
        circuit_name=case.circuit_name,
        circuit_key=case.circuit_key,
        svd_threshold=SVD_THRESHOLD,
        max_bond_dim_setting=max_bond_dim,
        reference_method=ref_method,
        fidelity_error=metrics.fidelity_error,
        observable_error_max=metrics.observable_error_max,
        observable_error_mean=metrics.observable_error_mean,
        energy_error=metrics.energy_error,
        energy_density_error=metrics.energy_density_error,
        magnetization_x_error=metrics.magnetization_x_error,
        magnetization_z_error=metrics.magnetization_z_error,
        long_range_corr_error_max=metrics.long_range_corr_error_max,
        practical_pass_1e3=metrics.practical_pass_1e3,
        practical_pass_1e2=metrics.practical_pass_1e2,
        max_bond_observed=int(mps.get_max_bond()),
        mean_bond_observed=_mean_bond_dim(mps),
        sum_chi_cubed=_sum_chi_cubed(mps),
        wall_time_s=wall,
        num_swaps_inserted=swaps_est,
        num_lr_pauli_gates=case.num_lr_pauli_gates,
        num_nn_2q_gates=case.num_nn_2q_gates,
        tdvp_lr_pauli_count=routes["tdvp_lr"],
        enriched_lr_pauli_count=routes["enriched_lr"],
        tdvp_lr_pauli_fraction=tdvp_frac,
        status=status,
    )


def _safe_ratio(tebd_err: float | None, tdvp_err: float | None) -> float | None:
    if tebd_err is None or tdvp_err is None:
        return None
    return float(tebd_err / max(tdvp_err, 1e-15))


def _recovery_chi(
    tebd_by_chi: dict[int, Row],
    *,
    tdvp_err: float,
    metric: Literal["energy_density_error", "observable_error_max"],
) -> str:
    for chi in sorted(tebd_by_chi):
        row = tebd_by_chi[chi]
        val = getattr(row, metric)
        if val is not None and val <= tdvp_err:
            return str(chi)
    return ">max_tested"


def _build_summary(rows: list[Row], *, stage: int) -> list[SummaryRow]:
    by_circuit: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        if r.status != "ok" or r.method not in {"local_generator_tdvp", "tebd_swaps"}:
            continue
        by_circuit[r.circuit_name].append(r)

    summaries: list[SummaryRow] = []

    depth_tdvp: dict[tuple[str, int | None], int] = {}
    depth_tebd: dict[tuple[str, int | None], int] = {}
    for r in rows:
        if r.status != "ok" or r.max_bond_dim_setting is None:
            continue
        key = (r.circuit_key, r.max_bond_dim_setting)
        if r.practical_pass_1e3:
            if r.method == "local_generator_tdvp":
                depth_tdvp[key] = max(depth_tdvp.get(key, 0), r.layers)
            elif r.method == "tebd_swaps":
                depth_tebd[key] = max(depth_tebd.get(key, 0), r.layers)

    for cname, grp in by_circuit.items():
        tdvp_rows = {r.max_bond_dim_setting: r for r in grp if r.method == "local_generator_tdvp"}
        tebd_rows = {r.max_bond_dim_setting: r for r in grp if r.method == "tebd_swaps"}
        if not tdvp_rows or not tebd_rows:
            continue
        ck = grp[0].circuit_key
        for tdvp_chi in RECOVERY_TDVP_CHIS:
            tdvp_r = tdvp_rows.get(tdvp_chi)
            if tdvp_r is None:
                continue
            tebd_same = tebd_rows.get(tdvp_chi)
            tebd_by_chi = {int(k): v for k, v in tebd_rows.items() if k is not None}

            rec_e = ">max_tested"
            rec_o = ">max_tested"
            if tdvp_r.energy_density_error is not None:
                rec_e = _recovery_chi(
                    tebd_by_chi, tdvp_err=tdvp_r.energy_density_error, metric="energy_density_error"
                )
            if tdvp_r.observable_error_max is not None:
                rec_o = _recovery_chi(tebd_by_chi, tdvp_err=tdvp_r.observable_error_max, metric="observable_error_max")

            def _ratio_val(rec: str) -> float | None:
                if rec.startswith(">") or not rec.isdigit():
                    return None
                return float(int(rec)) / float(tdvp_chi)

            summaries.append(
                SummaryRow(
                    stage=stage,
                    circuit_name=cname,
                    circuit_key=ck,
                    tdvp_chi=tdvp_chi,
                    tdvp_energy_density_error=tdvp_r.energy_density_error,
                    tdvp_observable_error_max=tdvp_r.observable_error_max,
                    tebd_same_chi_energy_density_error=tebd_same.energy_density_error if tebd_same else None,
                    tebd_same_chi_observable_error_max=tebd_same.observable_error_max if tebd_same else None,
                    energy_error_ratio_tebd_over_tdvp=_safe_ratio(
                        tebd_same.energy_density_error if tebd_same else None,
                        tdvp_r.energy_density_error,
                    ),
                    obs_error_ratio_tebd_over_tdvp=_safe_ratio(
                        tebd_same.observable_error_max if tebd_same else None,
                        tdvp_r.observable_error_max,
                    ),
                    tebd_recovery_chi_energy=rec_e,
                    tebd_recovery_chi_observable=rec_o,
                    recovery_ratio_energy=_ratio_val(rec_e),
                    recovery_ratio_observable=_ratio_val(rec_o),
                    tdvp_max_layers_practical_1e3=depth_tdvp.get((ck, tdvp_chi)),
                    tebd_max_layers_practical_1e3=depth_tebd.get((ck, tdvp_chi)),
                )
            )
    return summaries


def _write_csv(path: Path, rows: list[Any]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        d = row.__dict__ if hasattr(row, "__dict__") else row
        for k in d:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            d = row.__dict__ if hasattr(row, "__dict__") else row
            w.writerow({k: ("" if d.get(k) is None else d[k]) for k in keys})


def _mr_label(max_range: int | None) -> str:
    return "inf" if max_range is None else str(max_range)


def _plus_ok(rows: list[Row]) -> list[Row]:
    return [r for r in rows if r.status == "ok" and r.initial_state == "plus"]


def _render_plus_focus_tables(
    plus: list[Row],
    summaries: list[SummaryRow],
) -> str:
    lines: list[str] = ["\n## Plus-state focus (`initial_state=plus`)\n\n"]

    lines.append("### 1. Same-χ comparison\n\n")
    lines.append(
        "| n | α | R | dt | L | χ | TDVP dens | TEBD dens | TDVP obs | TEBD obs | TEBD/TDVP obs |\n"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    by_circuit_chi: dict[tuple[str, int | None], dict[str, Row]] = defaultdict(dict)
    for r in plus:
        if r.max_bond_dim_setting is None:
            continue
        by_circuit_chi[(r.circuit_name, r.max_bond_dim_setting)][r.method] = r

    highlight_gt2 = 0
    highlight_gt10 = 0
    for (cname, chi), meths in sorted(by_circuit_chi.items(), key=lambda x: (x[0][0], x[0][1] or 0)):
        t = meths.get("local_generator_tdvp")
        e = meths.get("tebd_swaps")
        if t is None or e is None:
            continue
        ratio = _safe_ratio(e.observable_error_max, t.observable_error_max)
        if ratio is not None and ratio > 2:
            highlight_gt2 += 1
        if ratio is not None and ratio > 10:
            highlight_gt10 += 1
        mr = _mr_label(t.max_range)
        lines.append(
            f"| {t.n_qubits} | {t.alpha} | {mr} | {t.dt:g} | {t.layers} | {chi} | "
            f"{t.energy_density_error:.3e} | {e.energy_density_error:.3e} | "
            f"{t.observable_error_max:.3e} | {e.observable_error_max:.3e} | "
            f"{ratio if ratio is not None else ''} |\n"
        )
    lines.append(f"\nPairs with TEBD/TDVP obs ratio **> 2**: {highlight_gt2}; **> 10**: {highlight_gt10}.\n\n")

    lines.append("### 2. TEBD recovery χ (main paper metric)\n\n")
    lines.append(
        "| circuit | TDVP χ | TDVP dens | rec_χ (energy) | ratio_E | rec_χ (obs) | ratio_obs |\n"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for s in sorted(summaries, key=lambda x: (x.circuit_name, x.tdvp_chi)):
        lines.append(
            f"| `{s.circuit_name}` | {s.tdvp_chi} | {s.tdvp_energy_density_error:.3e} | "
            f"{s.tebd_recovery_chi_energy} | "
            f"{s.recovery_ratio_energy if s.recovery_ratio_energy is not None else ''} | "
            f"{s.tebd_recovery_chi_observable} | "
            f"{s.recovery_ratio_observable if s.recovery_ratio_observable is not None else ''} |\n"
        )

    lines.append("\n### 3. Depth advantage (practical thresholds)\n\n")
    lines.append("Pass: energy_density ≤ 1e-3 and observable_max ≤ 1e-2.\n\n")
    lines.append("| n | α | R | dt | χ | method | L=20 pass | L=50 pass |\n")
    lines.append("|---:|---:|---:|---:|---:|---|---|---|\n")
    keys: dict[tuple[int, float, str, float, int, str], dict[int, bool]] = defaultdict(dict)
    for r in plus:
        if r.max_bond_dim_setting is None:
            continue
        mr = _mr_label(r.max_range)
        key = (r.n_qubits, float(r.alpha or 0), mr, r.dt, int(r.max_bond_dim_setting), r.method)
        pass_l = r.energy_density_error is not None and r.energy_density_error <= 1e-3
        pass_l = pass_l and r.observable_error_max is not None and r.observable_error_max <= 1e-2
        keys[key][r.layers] = pass_l
    for key, layer_pass in sorted(keys.items()):
        n, alpha, mr, dt, chi, meth = key
        lines.append(
            f"| {n} | {alpha} | {mr} | {dt:g} | {chi} | {meth} | "
            f"{'yes' if layer_pass.get(20) else 'no'} | {'yes' if layer_pass.get(50) else 'no'} |\n"
        )

    lines.append("\n### 4. Bond / SWAP overhead\n\n")
    lines.append("| circuit | χ | method | max_χ | mean_χ | sum_χ³ | wall_s | swaps |\n")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|\n")
    for r in sorted(plus, key=lambda x: (x.circuit_name, x.max_bond_dim_setting or 0, x.method)):
        lines.append(
            f"| `{r.circuit_name}` | {r.max_bond_dim_setting} | {r.method} | "
            f"{r.max_bond_observed} | {r.mean_bond_observed:.1f} | {r.sum_chi_cubed:.0f} | "
            f"{r.wall_time_s:.1f} | {r.num_swaps_inserted} |\n"
        )
    return "".join(lines)


def _render_md(
    rows: list[Row],
    summaries: list[SummaryRow],
    bond_hist: list[BondHistoryRow],
    *,
    stage: int,
    output_stem: str,
) -> str:
    ok = [r for r in rows if r.status == "ok"]
    profile_note = f", profile=`{PROFILE}`" if PROFILE else ""
    lines = [
        "# Fixed-bond advantage: local-generator TDVP vs TEBD+SWAP\n\n",
        f"**Stage {stage}** (`YAQS_FIXED_CHI_STAGE`{profile_note}). "
        f"`LR_PAULI_ROUTE={LR_PAULI_ROUTE!r}`; Pauli enrichment disabled for TDVP path.\n\n",
        "## Claim tested\n\n",
        "At fixed χ, local-generator TDVP can be more accurate than TEBD+SWAP for small-angle "
        "long-range Hamiltonian evolution, because SWAP networks inflate intermediate entanglement.\n\n",
        "## Method\n\n",
        "- **local_generator_tdvp**: NN → TEBD; LR Pauli → 2TDVP only.\n",
        "- **tebd_swaps**: all 2q → TEBD+SWAP.\n",
        "- Reference: Qiskit (n≤14), strict TEBD (χ=∞), optional exact Pauli enrichment.\n\n",
    ]

    if stage == 0:
        lines.append("## Sanity checks (Qiskit reference)\n\n")
    q_rows = [r for r in ok if r.reference_method == "qiskit_statevector"]
    if q_rows:
        for chi in (16, 24, 32):
            tdvp = [r for r in q_rows if r.method == "local_generator_tdvp" and r.max_bond_dim_setting == chi]
            tebd = [r for r in q_rows if r.method == "tebd_swaps" and r.max_bond_dim_setting == chi]
            if tdvp and tebd:
                lines.append(
                    f"- χ={chi}: median energy_density err TDVP="
                    f"{float(np.median([r.energy_density_error for r in tdvp if r.energy_density_error is not None])):.3e}, "
                    f"TEBD={float(np.median([r.energy_density_error for r in tebd if r.energy_density_error is not None])):.3e}\n"
                )

    lines.append("\n## Same-χ accuracy comparison\n\n")
    lines.append("| χ | TDVP median obs_max | TEBD median obs_max | TDVP wins (count) |\n")
    lines.append("|---:|---:|---:|---:|\n")
    for chi in sorted({r.max_bond_dim_setting for r in ok if r.max_bond_dim_setting}):
        tdvp = [r for r in ok if r.method == "local_generator_tdvp" and r.max_bond_dim_setting == chi]
        tebd = [r for r in ok if r.method == "tebd_swaps" and r.max_bond_dim_setting == chi]
        if not tdvp or not tebd:
            continue
        t_med = float(np.median([r.observable_error_max for r in tdvp if r.observable_error_max is not None]))
        e_med = float(np.median([r.observable_error_max for r in tebd if r.observable_error_max is not None]))
        wins = 0
        for t in tdvp:
            e = next((x for x in tebd if x.circuit_name == t.circuit_name), None)
            if (
                e is not None
                and t.observable_error_max is not None
                and e.observable_error_max is not None
                and t.observable_error_max < e.observable_error_max
            ):
                wins += 1
        lines.append(f"| {chi} | {t_med:.3e} | {e_med:.3e} | {wins} |\n")

    lines.append("\n## TEBD recovery χ\n\n")
    lines.append("Smallest TEBD χ with error ≤ TDVP at χ∈{16,24,32}. See `fixed_chi_tdvp_advantage_summary.csv`.\n\n")
    strong = [s for s in summaries if s.recovery_ratio_observable is not None and s.recovery_ratio_observable >= 2.0]
    if strong:
        lines.append("Cases with recovery ratio ≥ 2 (observable):\n\n")
        for s in strong[:10]:
            lines.append(
                f"- `{s.circuit_name}` χ_TDVP={s.tdvp_chi}: recovery_χ={s.tebd_recovery_chi_observable} "
                f"(ratio={s.recovery_ratio_observable:.1f})\n"
            )
    else:
        lines.append("- No cases with recovery ratio ≥ 2 in this stage.\n")

    lines.append("\n## Bond growth and SWAP overhead\n\n")
    for meth in ("local_generator_tdvp", "tebd_swaps"):
        sub = [r for r in ok if r.method == meth]
        if sub:
            lines.append(
                f"- `{meth}`: median sum_χ³={float(np.median([r.sum_chi_cubed for r in sub])):.0f}, "
                f"median max_χ_obs={float(np.median([r.max_bond_observed for r in sub])):.0f}\n"
            )

    lines.append("\n## Runtime\n\n")
    for meth in ("local_generator_tdvp", "tebd_swaps"):
        walls = [r.wall_time_s for r in ok if r.method == meth]
        if walls:
            lines.append(f"- `{meth}` median wall_time_s: {float(np.median(walls)):.2f}\n")

    lines.append("\n## Best examples / counterexamples\n\n")
    pairs: list[tuple[Row, Row]] = []
    by_name_chi: dict[tuple[str, int | None], dict[str, Row]] = defaultdict(dict)
    for r in ok:
        if r.max_bond_dim_setting is None:
            continue
        by_name_chi[(r.circuit_name, r.max_bond_dim_setting)][r.method] = r
    for meths in by_name_chi.values():
        if "local_generator_tdvp" in meths and "tebd_swaps" in meths:
            pairs.append((meths["local_generator_tdvp"], meths["tebd_swaps"]))

    tdvp_wins = [
        (t, e)
        for t, e in pairs
        if t.observable_error_max is not None
        and e.observable_error_max is not None
        and t.observable_error_max < e.observable_error_max * 0.9
    ]
    tebd_wins = [
        (t, e)
        for t, e in pairs
        if t.observable_error_max is not None
        and e.observable_error_max is not None
        and e.observable_error_max < t.observable_error_max * 0.9
    ]
    if tdvp_wins:
        lines.append("**TDVP better (same χ):**\n\n")
        for t, e in sorted(tdvp_wins, key=lambda p: p[1].observable_error_max / max(p[0].observable_error_max or 1e-15, 1e-15), reverse=True)[:5]:
            ratio = _safe_ratio(e.observable_error_max, t.observable_error_max)
            lines.append(
                f"- `{t.circuit_name}` χ={t.max_bond_dim_setting}: "
                f"obs TDVP={t.observable_error_max:.3e} TEBD={e.observable_error_max:.3e} "
                f"(ratio={ratio:.1f})\n"
            )
    if tebd_wins:
        lines.append("\n**TEBD better (same χ):**\n\n")
        for t, e in tebd_wins[:5]:
            lines.append(
                f"- `{t.circuit_name}` χ={t.max_bond_dim_setting}: "
                f"obs TDVP={t.observable_error_max:.3e} TEBD={e.observable_error_max:.3e}\n"
            )

    if PROFILE == "stage1_plus":
        lines.append(_render_plus_focus_tables(_plus_ok(rows), summaries))

    lines.append("\n## Conclusion\n\n")
    lines.append(_conclusion(ok, summaries))
    if bond_hist:
        lines.append(f"\nPer-layer bond history: `{output_stem}_bond_history.csv`.\n")
    return "".join(lines)


def _conclusion(ok: list[Row], summaries: list[SummaryRow]) -> str:
    if not ok:
        return "No successful rows.\n"

    same_chi_tdvp_better = 0
    same_chi_tebd_better = 0
    for t in ok:
        if t.method != "local_generator_tdvp":
            continue
        e = next(
            (
                r
                for r in ok
                if r.circuit_name == t.circuit_name
                and r.method == "tebd_swaps"
                and r.max_bond_dim_setting == t.max_bond_dim_setting
            ),
            None,
        )
        if e is None or t.observable_error_max is None or e.observable_error_max is None:
            continue
        if t.observable_error_max < e.observable_error_max:
            same_chi_tdvp_better += 1
        elif e.observable_error_max < t.observable_error_max:
            same_chi_tebd_better += 1

    recovery_ge2 = sum(1 for s in summaries if (s.recovery_ratio_observable or 0) >= 2.0)
    enr = sum(r.enriched_lr_pauli_count for r in ok if r.method == "local_generator_tdvp")

    lines = [
        f"- Same-χ pairs: TDVP lower obs error in **{same_chi_tdvp_better}** cases; "
        f"TEBD lower in **{same_chi_tebd_better}**.\n",
        f"- Recovery ratio ≥ 2 (TEBD needs ≥2× χ to match TDVP): **{recovery_ge2}** summary rows.\n",
    ]
    if enr == 0:
        lines.append("- `enriched_lr_pauli_count = 0` for local_generator_tdvp ✓\n")

    if same_chi_tdvp_better > same_chi_tebd_better and recovery_ge2 > 0:
        lines.append(
            "\n**Partial support:** fixed-bond TDVP advantage appears in a subset of power-law / plus-state cases; "
            "run stage 1+ for stronger evidence.\n"
        )
    elif same_chi_tebd_better >= same_chi_tdvp_better:
        lines.append(
            "\n**Not supported in this stage:** TEBD+SWAP is as accurate or better at the same χ on most cases tested.\n"
        )
    else:
        lines.append("\n**Inconclusive:** rerun with larger stage or more χ values.\n")
    return "".join(lines)


def main() -> None:
    if STAGE not in STAGE_CONFIGS:
        msg = f"Unknown YAQS_FIXED_CHI_STAGE={STAGE}; choose from {sorted(STAGE_CONFIGS)}"
        raise SystemExit(msg)

    cfg = _active_config(STAGE)
    cases = _build_cases(STAGE, cfg)
    chi_values: list[int | None] = list(cfg["max_bond_dim_values"])
    methods: list[MethodName] = ["local_generator_tdvp", "tebd_swaps"]
    stem = _output_stem()

    out = _results_dir()
    rows: list[Row] = []
    bond_hist: list[BondHistoryRow] = []

    mr_vals = cfg.get("max_range_values", [])
    print(
        f"Stage {STAGE} profile={PROFILE or 'default'}: {len(cases)} circuits, "
        f"max_range={mr_vals}, chi={chi_values}, LR_PAULI_ROUTE={LR_PAULI_ROUTE!r}, stem={stem}"
    )

    for case in cases:
        ref, ref_method = _resolve_reference(case, stage_cfg=cfg)
        if ref is None:
            print(f"SKIP {case.circuit_name}: no reference")
            continue

        for chi in chi_values:
            for method in methods:
                mps, wall, bh, _, routes, status = _run_circuit(case, method=method, max_bond_dim=chi)
                bond_hist.extend(bh)
                row = _row_from_run(
                    case,
                    method=method,
                    max_bond_dim=chi,
                    mps=mps,
                    wall=wall,
                    routes=routes,
                    status=status,
                    ref=ref,
                    ref_method=ref_method,
                )
                rows.append(row)
                print(
                    f"[{len(rows)}] {case.circuit_name} {method} chi={chi} "
                    f"dens_err={row.energy_density_error} obs={row.observable_error_max}"
                )

    summaries = _build_summary(rows, stage=STAGE)
    csv_path = out / f"{stem}.csv"
    sum_path = out / f"{stem}_summary.csv"
    md_path = out / f"{stem}.md"
    bond_path = out / f"{stem}_bond_history.csv"

    _write_csv(csv_path, rows)
    _write_csv(sum_path, summaries)
    md_path.write_text(_render_md(rows, summaries, bond_hist, stage=STAGE, output_stem=stem), encoding="utf-8")
    if RECORD_BOND_HISTORY:
        _write_csv(bond_path, bond_hist)

    print(f"\nWrote {csv_path} ({len(rows)} rows)")
    print(f"Wrote {sum_path} ({len(summaries)} rows)")
    print(f"Wrote {md_path}")
    if RECORD_BOND_HISTORY:
        print(f"Wrote {bond_path}")


if __name__ == "__main__":
    main()
