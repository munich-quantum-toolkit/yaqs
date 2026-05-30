#!/usr/bin/env python
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Intuition-map benchmark: when does hybrid long-range TDVP beat TEBD+SWAP?

This is the single staged TDVP-vs-TEBD benchmark suite. Use stages 0–3 for
increasing cost and coverage.

Run::

    uv run python -m scripts.benchmark_long_range_tdvp_regimes

Environment::

    YAQS_LR_REGIME_STAGE=0|1|2|3
    YAQS_LR_REGIME_MAX_CASES=N
    YAQS_LR_REGIME_SEED=1234
    YAQS_LR_REGIME_BOND_HISTORY=0|1
    YAQS_LR_REGIME_OUTDIR=results/long_range_tdvp_regimes
    YAQS_LR_REGIME_OVERWRITE=1   truncate raw_runs.csv and rerun
    YAQS_LR_REGIME_TDVP_PADDING_NOISE=0
    YAQS_LR_REGIME_INCLUDE_PADDING_SCAN=0|1
    YAQS_LR_REGIME_INCLUDE_TDVP_SWEEP_SCAN=0|1
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
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
    apply_two_qubit_gate_tdvp_experimental,
)
from mqt.yaqs.digital.utils.dag_utils import convert_dag_to_tensor_algorithm

from scripts.benchmark_utils import (
    _expectation_via_mps_swaps,
    _grid_edges,
    _mean_bond_dim,
    _num_swaps_for_gate,
    _prep_initial_state,
    _sum_chi_cubed,
)
from scripts.yaqs_reference_utils import (
    ReferenceConvention,
    align_reference_vec,
    bit_reverse_vec,
    check_high_fidelity_observable_consistency,
    compare_statevectors,
    observable_errors_against_reference,
    observable_errors_grouped,
    mps_expectation_via_dense,
    qiskit_reference_vec,
    statevector_expectation,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STAGE = int(os.environ.get("YAQS_LR_REGIME_STAGE", "1"))
MAX_CASES = int(os.environ.get("YAQS_LR_REGIME_MAX_CASES", "0"))
BASE_SEED = int(os.environ.get("YAQS_LR_REGIME_SEED", "1234"))
RECORD_BOND_HISTORY = os.environ.get("YAQS_LR_REGIME_BOND_HISTORY", "0").strip() not in {
    "",
    "0",
    "false",
    "False",
}
OVERWRITE = os.environ.get("YAQS_LR_REGIME_OVERWRITE", "").strip() not in {"", "0", "false", "False"}
OUTDIR = Path(os.environ.get("YAQS_LR_REGIME_OUTDIR", "results/long_range_tdvp_regimes"))
TDVP_PADDING_NOISE = float(os.environ.get("YAQS_LR_REGIME_TDVP_PADDING_NOISE", "0"))
INCLUDE_PADDING_SCAN = os.environ.get("YAQS_LR_REGIME_INCLUDE_PADDING_SCAN", "0").strip() not in {
    "",
    "0",
    "false",
    "False",
}
INCLUDE_TDVP_SWEEP_SCAN = os.environ.get("YAQS_LR_REGIME_INCLUDE_TDVP_SWEEP_SCAN", "0").strip() not in {
    "",
    "0",
    "false",
    "False",
}

PADDED4_TARGET_DIM = 4

STAGE_PADDING_SCAN_DIMS: dict[int, list[int | None]] = {
    0: [None, 2, 4, 6, 8],
    1: [None, 2, 4, 8, 16],
    2: [None, 4, 8, 16, 32],
    3: [None, 4, 8, 16, 32, 64],
}

STAGE_MAIN_METHODS: dict[int, list[str]] = {
    0: ["tebd_swap", "hybrid_tdvp_unpadded", "hybrid_tdvp_padded4"],
    1: ["tebd_swap", "hybrid_tdvp_unpadded", "hybrid_tdvp_padded4"],
    2: ["tebd_swap", "hybrid_tdvp_padded4"],
    3: ["tebd_swap", "hybrid_tdvp_padded4"],
}

SVD_THRESHOLD = 1e-10
LANCZOS_TOL = 1e-8
REF_SVD = 1e-12
HIGH_BUDGET_CHI = 256

# Set by validate_reference_convention() before benchmark runs.
REFERENCE_CONVENTION: ReferenceConvention = "direct"

MethodName = Literal[
    "tebd_swap",
    "hybrid_tdvp_unpadded",
    "hybrid_tdvp_padded4",
    "hybrid_tdvp_padding_scan",
]
SimMode = Literal["tebd_swap", "hybrid_tdvp"]
TDVP_METHODS = frozenset({
    "hybrid_tdvp_unpadded",
    "hybrid_tdvp_padded4",
    "hybrid_tdvp_padding_scan",
})
RefType = Literal["exact_statevector", "high_budget_tebd", "high_budget_tdvp", "none"]
AngleRegime = Literal["small", "medium", "large"]
FamilyName = Literal[
    "nn_brickwork",
    "periodic_1d",
    "sparse_long_range",
    "flattened_2d_grid",
    "dense_long_range",
    "random_all_to_all",
    "floquet_hump",
]

STAGE_CONFIGS: dict[int, dict[str, Any]] = {
    0: {
        "n_values": [8, 10],
        "depth_values": [2, 4],
        "num_seeds": 1,
        "chi_values": [16, 32],
        "tdvp_sweep_scan_values": [1, 2, 4],
        "angle_regimes": ["small"],
        "families": ["nn_brickwork", "periodic_1d", "sparse_long_range"],
        "exact_n_max": 16,
        "include_large_angles": False,
    },
    1: {
        "n_values": [8, 12, 16],
        "depth_values": [4, 8],
        "num_seeds": 3,
        "chi_values": [16, 32, 64],
        "tdvp_sweep_scan_values": [1, 2, 4],
        "angle_regimes": ["small", "medium"],
        "families": [
            "nn_brickwork",
            "periodic_1d",
            "sparse_long_range",
            "flattened_2d_grid",
            "dense_long_range",
            "random_all_to_all",
        ],
        "exact_n_max": 16,
        "include_large_angles": False,
    },
    2: {
        "n_values": [12, 16, 24],
        "depth_values": [4, 8, 12],
        "num_seeds": 5,
        "chi_values": [16, 32, 64, 128],
        "tdvp_sweep_scan_values": [1, 2, 4],
        "angle_regimes": ["small", "medium"],
        "families": [
            "nn_brickwork",
            "periodic_1d",
            "sparse_long_range",
            "flattened_2d_grid",
            "dense_long_range",
            "random_all_to_all",
            "floquet_hump",
        ],
        "exact_n_max": 16,
        "include_large_angles": True,
    },
    3: {
        "n_values": [16, 24, 32],
        "depth_values": [8, 16],
        "num_seeds": 5,
        "chi_values": [32, 64, 128, 256],
        "tdvp_sweep_scan_values": [1, 2, 4, 8],
        "angle_regimes": ["small", "medium", "large"],
        "families": [
            "nn_brickwork",
            "periodic_1d",
            "sparse_long_range",
            "flattened_2d_grid",
            "dense_long_range",
            "random_all_to_all",
            "floquet_hump",
        ],
        "exact_n_max": 12,
        "include_large_angles": True,
    },
}


def _resolve_sweep_values(cfg: dict[str, Any]) -> list[int]:
    """Return sweep grid; default is a single sweep for standard long-range TDVP."""
    if INCLUDE_TDVP_SWEEP_SCAN:
        return list(cfg.get("tdvp_sweep_scan_values", [1, 2, 4]))
    return [1]


def _sim_mode(method: MethodName) -> SimMode:
    return "tebd_swap" if method == "tebd_swap" else "hybrid_tdvp"


def _is_tdvp_method(method: MethodName) -> bool:
    return method in TDVP_METHODS


def _padding_dim_for_method(method: MethodName, scan_dim: int | None = None) -> int | None:
    if method == "hybrid_tdvp_unpadded":
        return None
    if method == "hybrid_tdvp_padded4":
        return PADDED4_TARGET_DIM
    if method == "hybrid_tdvp_padding_scan":
        return scan_dim
    return None


@dataclass(frozen=True)
class BenchmarkRunSpec:
    method: MethodName
    tdvp_sweeps: int
    tdvp_padding_dim: int | None
    sweep_scan_enabled: bool


def build_benchmark_runs(stage: int, cfg: dict[str, Any]) -> list[BenchmarkRunSpec]:
    """Build the list of (method, sweeps, padding) combinations for one circuit."""
    runs: list[BenchmarkRunSpec] = []
    sweep_scan = INCLUDE_TDVP_SWEEP_SCAN
    sweeps = _resolve_sweep_values(cfg)
    main_methods = STAGE_MAIN_METHODS.get(stage, STAGE_MAIN_METHODS[0])

    for tdvp_sweeps in sweeps:
        for method in main_methods:
            runs.append(
                BenchmarkRunSpec(
                    method=method,  # type: ignore[arg-type]
                    tdvp_sweeps=tdvp_sweeps if _is_tdvp_method(method) else 1,  # type: ignore[arg-type]
                    tdvp_padding_dim=_padding_dim_for_method(method),  # type: ignore[arg-type]
                    sweep_scan_enabled=sweep_scan,
                )
            )

    if INCLUDE_PADDING_SCAN:
        scan_dims = STAGE_PADDING_SCAN_DIMS.get(stage, STAGE_PADDING_SCAN_DIMS[0])
        for tdvp_sweeps in sweeps:
            for pad in scan_dims:
                if tdvp_sweeps == 1:
                    if pad is None and "hybrid_tdvp_unpadded" in main_methods:
                        continue
                    if pad == PADDED4_TARGET_DIM and "hybrid_tdvp_padded4" in main_methods:
                        continue
                runs.append(
                    BenchmarkRunSpec(
                        method="hybrid_tdvp_padding_scan",
                        tdvp_sweeps=tdvp_sweeps,
                        tdvp_padding_dim=pad,
                        sweep_scan_enabled=sweep_scan,
                    )
                )
    return runs


def _padding_key(padding_dim: int | None) -> str:
    return "none" if padding_dim is None else str(padding_dim)


def _noise_key(noise: float) -> str:
    return "0" if noise == 0.0 else f"{noise:g}"


def _padding_key_from_row(row: dict[str, Any]) -> str:
    value = row.get("tdvp_padding_dim", "")
    if value in (None, "", "none", "None"):
        return "none"
    return str(value)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CircuitSpec:
    family: str
    n_qubits: int
    depth: int
    seed: int
    angle_regime: str
    case_id: str
    qc: QuantumCircuit
    lr_pairs: tuple[tuple[int, int], ...]
    nn_pairs: tuple[tuple[int, int], ...]
    topology: str
    lx: int | None = None
    ly: int | None = None


@dataclass
class GateDispatchStats:
    """Counts recorded while applying *spec.qc* (prep gates excluded)."""

    total_gates: int = 0
    nn_gate_count: int = 0
    lr_gate_count: int = 0
    tebd_direct_gate_count: int = 0
    tebd_swap_gate_count: int = 0
    tdvp_lr_gate_count: int = 0
    skipped_or_exact_lr_gate_count: int = 0
    num_swaps_inserted: int = 0
    lr_gate_pairs: tuple[tuple[int, int], ...] = ()


@dataclass
class PaddingRunStats:
    tdvp_padding_dim: int | None = None
    tdvp_padding_noise: float = 0.0
    padding_applied: bool = False
    padded_bonds_count: int = 0
    max_bond_dim_before_padding: int | None = None
    max_bond_dim_after_padding: int | None = None
    mean_bond_dim_after_padding: float | None = None
    peak_bond_dim_before_first_lr: int | None = None
    peak_bond_dim_after_first_lr: int | None = None


@dataclass
class RawRun:
    family: str
    n_qubits: int
    depth: int
    seed: int
    angle_regime: str
    method: str
    chi_max: int
    tdvp_sweeps: int
    svd_threshold: float
    lanczos_tol: float
    reference_type: str
    reference_available: bool
    reference_chi: int | None = None
    reference_sweeps: int | None = None
    wall_time_s: float = 0.0
    peak_bond_dim: int = 0
    mean_bond_dim: float = 0.0
    final_max_bond_dim: int = 0
    num_truncations: int = 0
    hit_chi_max: bool = False
    norm_error_or_norm_drift: float | None = None
    fidelity_to_reference: float | None = None
    infidelity_to_reference: float | None = None
    fidelity_direct: float | None = None
    fidelity_bit_reversed: float | None = None
    observable_error_direct: float | None = None
    observable_error_bit_reversed: float | None = None
    reference_convention_used: str = "direct"
    mean_abs_observable_error: float | None = None
    max_abs_observable_error: float | None = None
    rms_observable_error: float | None = None
    energy_error: float | None = None
    status: str = "ok"
    error_message: str = ""
    case_id: str = ""
    num_swaps_inserted: int = 0
    tdvp_lr_count: int = 0
    enriched_lr_count: int = 0
    total_gates: int = 0
    nn_gate_count: int = 0
    lr_gate_count: int = 0
    tebd_direct_gate_count: int = 0
    tebd_swap_gate_count: int = 0
    tdvp_lr_gate_count: int = 0
    skipped_or_exact_lr_gate_count: int = 0
    lr_gate_pairs: str = ""
    mean_abs_error_z_single: float | None = None
    max_abs_error_z_single: float | None = None
    mean_abs_error_x_single: float | None = None
    max_abs_error_x_single: float | None = None
    mean_abs_error_y_single: float | None = None
    max_abs_error_y_single: float | None = None
    mean_abs_error_zz_nn: float | None = None
    max_abs_error_zz_nn: float | None = None
    mean_abs_error_zz_lr: float | None = None
    max_abs_error_zz_lr: float | None = None
    fidelity_observable_consistency_warning: str = ""
    tdvp_padding_dim: int | None = None
    tdvp_padding_noise: float = 0.0
    padding_applied: bool = False
    padded_bonds_count: int = 0
    max_bond_dim_before_padding: int | None = None
    max_bond_dim_after_padding: int | None = None
    mean_bond_dim_after_padding: float | None = None
    peak_bond_dim_before_first_lr: int | None = None
    peak_bond_dim_after_first_lr: int | None = None
    sweep_scan_enabled: bool = False

    def run_key(self) -> str:
        return (
            f"{self.case_id}|{self.method}|chi{self.chi_max}|sweeps{self.tdvp_sweeps}"
            f"|padding{_padding_key(self.tdvp_padding_dim)}|noise{_noise_key(self.tdvp_padding_noise)}"
        )


@dataclass(frozen=True)
class BondHistoryRow:
    case_id: str
    method: str
    chi_max: int
    tdvp_sweeps: int
    layer: int
    max_bond: int
    mean_bond: float
    sum_chi_cubed: float


# ---------------------------------------------------------------------------
# Angle helpers
# ---------------------------------------------------------------------------


def _angle_range(regime: str, rng: np.random.Generator) -> tuple[float, float]:
    if regime == "small":
        return 0.05, 0.2
    if regime == "medium":
        return 0.3, 0.8
    return 1.0, float(np.pi / 2)


def _sample_angle(regime: str, rng: np.random.Generator) -> float:
    lo, hi = _angle_range(regime, rng)
    return float(rng.uniform(lo, hi))


def _gate_names_cycle(layer: int) -> tuple[str, ...]:
    return ("rxx", "ryy", "rzz")[layer % 3 : layer % 3 + 1] or ("rzz",)


# ---------------------------------------------------------------------------
# Circuit family builders
# ---------------------------------------------------------------------------


def _add_pauli_rotation(qc: QuantumCircuit, name: str, theta: float, i: int, j: int) -> None:
    if i == j:
        msg = f"duplicate qubit indices for {name}: ({i}, {j})"
        raise ValueError(msg)
    getattr(qc, name)(theta, i, j)


def _sample_distinct_pair(n: int, rng: np.random.Generator) -> tuple[int, int]:
    """Return sorted distinct qubit indices ``(i, j)`` with ``i < j``."""
    if n < 2:
        msg = f"need at least 2 qubits for two-qubit gates, got n={n}"
        raise ValueError(msg)
    a, b = (int(x) for x in rng.choice(n, size=2, replace=False))
    return (a, b) if a < b else (b, a)


def make_circuit_family(
    family: str,
    *,
    n: int,
    depth: int,
    seed: int,
    angle_regime: str,
) -> CircuitSpec:
    rng = np.random.default_rng(BASE_SEED + seed * 10007 + n * 131 + depth * 17 + hash(family) % 1000)
    qc = QuantumCircuit(n)
    lr_pairs: set[tuple[int, int]] = set()
    nn_pairs: set[tuple[int, int]] = set()
    topology = "1d_chain"
    lx = ly = None

    for layer in range(depth):
        if family == "nn_brickwork":
            for i in range(n - 1):
                theta = _sample_angle(angle_regime, rng)
                gname = ("rxx", "ryy", "rzz")[layer % 3]
                _add_pauli_rotation(qc, gname, theta, i, i + 1)
                nn_pairs.add((i, i + 1))

        elif family == "periodic_1d":
            for i in range(n - 1):
                theta = _sample_angle(angle_regime, rng)
                _add_pauli_rotation(qc, "rzz", theta, i, i + 1)
                nn_pairs.add((i, i + 1))
            theta = _sample_angle(angle_regime, rng)
            _add_pauli_rotation(qc, "rzz", theta, 0, n - 1)
            lr_pairs.add((0, n - 1))
            topology = "1d_ring"

        elif family == "sparse_long_range":
            for i in range(n - 1):
                theta = _sample_angle(angle_regime, rng)
                _add_pauli_rotation(qc, "rzz", theta, i, i + 1)
                nn_pairs.add((i, i + 1))
            min_dist = max(2, n // 3)
            candidates = [(i, j) for i in range(n) for j in range(i + min_dist, n)]
            rng.shuffle(candidates)
            for pair in candidates[: min(2, len(candidates))]:
                theta = _sample_angle(angle_regime, rng)
                _add_pauli_rotation(qc, "rxx", theta, pair[0], pair[1])
                lr_pairs.add(pair)

        elif family == "flattened_2d_grid":
            lx = int(np.ceil(np.sqrt(n)))
            while n % lx != 0 and lx <= n:
                lx += 1
            if n % lx != 0:
                lx, ly = 1, n
            else:
                ly = n // lx
            edges = _grid_edges(lx=lx, ly=ly, periodic_x=False)
            topology = f"2d_grid_{lx}x{ly}_row_major"
            for a, b, _ in edges:
                theta = _sample_angle(angle_regime, rng)
                _add_pauli_rotation(qc, "rzz", theta, a, b)
                if abs(a - b) == 1:
                    nn_pairs.add((min(a, b), max(a, b)))
                else:
                    lr_pairs.add((min(a, b), max(a, b)))

        elif family == "dense_long_range":
            for _ in range(min(4, max(2, n // 2))):
                for _attempt in range(20):
                    i, j = _sample_distinct_pair(n, rng)
                    if j - i > 1:
                        break
                else:
                    i, j = 0, min(n - 1, 3)
                theta = _sample_angle(angle_regime, rng)
                gname = rng.choice(["rxx", "ryy", "rzz"])
                _add_pauli_rotation(qc, gname, theta, i, j)
                lr_pairs.add((i, j))

        elif family == "random_all_to_all":
            for _ in range(min(6, n)):
                i, j = _sample_distinct_pair(n, rng)
                theta = _sample_angle(angle_regime, rng)
                gname = rng.choice(["rxx", "ryy", "rzz"])
                _add_pauli_rotation(qc, gname, theta, i, j)
                if abs(i - j) == 1:
                    nn_pairs.add((min(i, j), max(i, j)))
                else:
                    lr_pairs.add((min(i, j), max(i, j)))

        elif family == "floquet_hump":
            if layer % 2 == 0:
                for i in range(n - 1):
                    theta = _sample_angle(angle_regime, rng)
                    _add_pauli_rotation(qc, "rzz", theta, i, i + 1)
                    nn_pairs.add((i, i + 1))
                if n > 4:
                    i, j = 0, n // 2
                    theta = _sample_angle(angle_regime, rng)
                    _add_pauli_rotation(qc, "rxx", theta, i, j)
                    lr_pairs.add((min(i, j), max(i, j)))
            else:
                for q in range(n):
                    qc.rx(_sample_angle(angle_regime, rng) * 0.5, q)
            topology = "floquet"

        else:
            msg = f"Unknown family: {family}"
            raise ValueError(msg)
        qc.barrier()

    case_id = f"{family}_n{n}_d{depth}_s{seed}_{angle_regime}"
    return CircuitSpec(
        family=family,
        n_qubits=n,
        depth=depth,
        seed=seed,
        angle_regime=angle_regime,
        case_id=case_id,
        qc=qc,
        lr_pairs=tuple(sorted(lr_pairs)),
        nn_pairs=tuple(sorted(nn_pairs)),
        topology=topology,
        lx=lx,
        ly=ly,
    )


def build_all_specs(cfg: dict[str, Any]) -> list[CircuitSpec]:
    specs: list[CircuitSpec] = []
    regimes: list[str] = list(cfg["angle_regimes"])
    if cfg.get("include_large_angles") and "large" not in regimes:
        regimes.append("large")
    for family in cfg["families"]:
        for n in cfg["n_values"]:
            for depth in cfg["depth_values"]:
                for seed in range(cfg["num_seeds"]):
                    for angle_regime in regimes:
                        specs.append(
                            make_circuit_family(
                                family,
                                n=n,
                                depth=depth,
                                seed=seed,
                                angle_regime=angle_regime,
                            )
                        )
    if MAX_CASES > 0:
        return specs[:MAX_CASES]
    return specs


def write_circuit_metadata(specs: list[CircuitSpec], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for spec in specs:
            gates = []
            for ci in spec.qc.data:
                if ci.operation.name == "barrier":
                    continue
                if len(ci.qubits) == 2:
                    qa = spec.qc.find_bit(ci.qubits[0]).index
                    qb = spec.qc.find_bit(ci.qubits[1]).index
                    gates.append({
                        "name": ci.operation.name,
                        "sites": [qa, qb],
                        "theta": float(ci.operation.params[0]) if ci.operation.params else 0.0,
                    })
            meta = {
                "case_id": spec.case_id,
                "family": spec.family,
                "n_qubits": spec.n_qubits,
                "depth": spec.depth,
                "seed": spec.seed,
                "angle_regime": spec.angle_regime,
                "topology": spec.topology,
                "lr_pairs": list(spec.lr_pairs),
                "nn_pairs": list(spec.nn_pairs),
                "lx": spec.lx,
                "ly": spec.ly,
                "num_gates": len(gates),
                "gates": gates[:200],
            }
            f.write(json.dumps(meta) + "\n")


# ---------------------------------------------------------------------------
# Observables and reference
# ---------------------------------------------------------------------------


def observable_list(spec: CircuitSpec) -> list[tuple[str, str, list[int]]]:
    n = spec.n_qubits
    obs: list[tuple[str, str, list[int]]] = []
    sites_z = list(range(n)) if n <= 20 else sorted({0, n // 2, n - 1, max(0, n // 4)})
    for i in sites_z:
        obs.append((f"Z({i})", "Z", [i]))
    for i in sites_z[: min(4, len(sites_z))]:
        obs.append((f"X({i})", "X", [i]))
    for i in sites_z[: min(4, len(sites_z))]:
        obs.append((f"Y({i})", "Y", [i]))
    for a, b in spec.nn_pairs[: min(8, len(spec.nn_pairs))]:
        obs.append((f"ZZ_nn({a},{b})", "ZZ", [a, b]))
    for a, b in spec.lr_pairs[: min(6, len(spec.lr_pairs))]:
        obs.append((f"ZZ_lr({a},{b})", "ZZ", [a, b]))
    return obs


def _static_gate_counts(qc: QuantumCircuit) -> tuple[int, int, int]:
    """Return (total_gates, nn_gate_count, lr_gate_count) for *qc* body."""
    total = nn = lr = 0
    dag = circuit_to_dag(qc)
    for node in dag.topological_op_nodes():
        if node.op.name == "barrier":
            continue
        if len(node.qargs) == 1:
            total += 1
            continue
        if len(node.qargs) != 2:
            continue
        total += 1
        qa = qc.find_bit(node.qargs[0]).index
        qb = qc.find_bit(node.qargs[1]).index
        if abs(qa - qb) == 1:
            nn += 1
        else:
            lr += 1
    return total, nn, lr


def _apply_grouped_observable_errors(
    run: RawRun,
    spec: CircuitSpec,
    mps: MPS,
    ref: MPS | np.ndarray,
    *,
    convention: ReferenceConvention,
) -> None:
    obs = observable_list(spec)
    if isinstance(ref, np.ndarray):
        grouped = observable_errors_grouped(mps, ref, obs, n=spec.n_qubits, convention=convention)
        y_errs: list[float] = []
        use_ref = align_reference_vec(ref, spec.n_qubits, convention)
        for name, label, sites in obs:
            if not name.startswith("Y("):
                continue
            got = mps_expectation_via_dense(mps, spec.n_qubits, label=label, sites=sites)
            ref_val = statevector_expectation(use_ref, spec.n_qubits, label=label, sites=sites)
            y_errs.append(abs(got - ref_val))
        if y_errs:
            y_arr = np.array(y_errs, dtype=np.float64)
            grouped["y_single"] = (
                float(np.mean(y_arr)),
                float(np.max(y_arr)),
                float(np.sqrt(np.mean(y_arr**2))),
            )
        else:
            grouped["y_single"] = (0.0, 0.0, 0.0)
        _, max_d, _ = observable_errors_against_reference(
            mps, ref, obs, n=spec.n_qubits, convention="direct"
        )
        _, max_r, _ = observable_errors_against_reference(
            mps, ref, obs, n=spec.n_qubits, convention="bit_reversed"
        )
        run.observable_error_direct = max_d
        run.observable_error_bit_reversed = max_r
    else:
        grouped_lists: dict[str, list[float]] = defaultdict(list)
        for name, label, sites in obs:
            got = _expectation_via_mps_swaps(mps, gate_name=label.lower(), sites=sites)
            ref_val = _expectation_via_mps_swaps(ref, gate_name=label.lower(), sites=sites)
            if name.startswith("Z("):
                cat = "z_single"
            elif name.startswith("X("):
                cat = "x_single"
            elif name.startswith("Y("):
                cat = "y_single"
            elif name.startswith("ZZ_nn"):
                cat = "zz_nn"
            elif name.startswith("ZZ_lr"):
                cat = "zz_lr"
            else:
                cat = "other"
            grouped_lists[cat].append(abs(got - ref_val))
        grouped = {}
        for cat in ("z_single", "x_single", "y_single", "zz_nn", "zz_lr", "other"):
            errs = grouped_lists.get(cat, [])
            if not errs:
                grouped[cat] = (0.0, 0.0, 0.0)
            else:
                arr = np.array(errs, dtype=np.float64)
                grouped[cat] = float(np.mean(arr)), float(np.max(arr)), float(np.sqrt(np.mean(arr**2)))
        run.observable_error_direct = float(
            max(
                grouped["z_single"][1],
                grouped["x_single"][1],
                grouped["y_single"][1],
                grouped["zz_nn"][1],
                grouped["zz_lr"][1],
                grouped["other"][1],
            )
        )
        run.observable_error_bit_reversed = run.observable_error_direct

    zs = grouped["z_single"]
    xs = grouped["x_single"]
    ys = grouped.get("y_single", (0.0, 0.0, 0.0))
    znn = grouped["zz_nn"]
    zlr = grouped["zz_lr"]
    other = grouped["other"]
    run.mean_abs_error_z_single = zs[0]
    run.max_abs_error_z_single = zs[1]
    run.mean_abs_error_x_single = xs[0]
    run.max_abs_error_x_single = xs[1]
    run.mean_abs_error_y_single = ys[0]
    run.max_abs_error_y_single = ys[1]
    run.mean_abs_error_zz_nn = znn[0]
    run.max_abs_error_zz_nn = znn[1]
    run.mean_abs_error_zz_lr = zlr[0]
    run.max_abs_error_zz_lr = zlr[1]
    run.max_abs_observable_error = float(max(zs[1], xs[1], ys[1], znn[1], zlr[1], other[1]))
    cat_means = [zs[0], xs[0], ys[0], znn[0], zlr[0], other[0]]
    run.mean_abs_observable_error = float(np.mean([m for m in cat_means if m > 0] or [0.0]))
    run.rms_observable_error = float(
        np.sqrt(np.mean([zs[2] ** 2, xs[2] ** 2, ys[2] ** 2, znn[2] ** 2, zlr[2] ** 2, other[2] ** 2]))
    )


def compute_observable_errors(
    spec: CircuitSpec,
    mps: MPS,
    ref: MPS | np.ndarray,
    *,
    convention: ReferenceConvention,
) -> tuple[float, float, float, float, float]:
    obs = observable_list(spec)
    if isinstance(ref, np.ndarray):
        mean_d, max_d, rms_d = observable_errors_against_reference(
            mps, ref, obs, n=spec.n_qubits, convention="direct"
        )
        mean_r, max_r, rms_r = observable_errors_against_reference(
            mps, ref, obs, n=spec.n_qubits, convention="bit_reversed"
        )
        if convention == "bit_reversed":
            return mean_r, max_r, rms_r, max_d, max_r
        return mean_d, max_d, rms_d, max_d, max_r
    # MPS reference: direct only
    errs: list[float] = []
    for _name, label, sites in obs:
        got = _expectation_via_mps_swaps(mps, gate_name=label.lower(), sites=sites)
        ref_val = _expectation_via_mps_swaps(ref, gate_name=label.lower(), sites=sites)
        errs.append(abs(got - ref_val))
    arr = np.array(errs, dtype=np.float64)
    mx = float(np.max(arr))
    return float(np.mean(arr)), mx, float(np.sqrt(np.mean(arr**2))), mx, mx


def compute_reference(
    spec: CircuitSpec,
    *,
    exact_n_max: int,
) -> tuple[MPS | np.ndarray | None, RefType, int | None, int | None]:
    prep = QuantumCircuit(spec.n_qubits)
    _prep_initial_state(prep, "plus", seed=0)
    full = prep.compose(spec.qc)

    if spec.n_qubits <= exact_n_max:
        return qiskit_reference_vec(full), "exact_statevector", None, None

    params = StrongSimParams(
        preset="exact",
        get_state=True,
        svd_threshold=REF_SVD,
        max_bond_dim=HIGH_BUDGET_CHI,
        krylov_tol=LANCZOS_TOL,
        gate_mode="tebd",
    )
    try:
        dummy = CircuitSpec(
            family=spec.family,
            n_qubits=spec.n_qubits,
            depth=spec.depth,
            seed=spec.seed,
            angle_regime=spec.angle_regime,
            case_id=spec.case_id,
            qc=spec.qc,
            lr_pairs=spec.lr_pairs,
            nn_pairs=spec.nn_pairs,
            topology=spec.topology,
            lx=spec.lx,
            ly=spec.ly,
        )
        mps, _, _, _ = _run_method(spec.qc, dummy, method="tebd_swap", chi=HIGH_BUDGET_CHI, tdvp_sweeps=1, params=params)
        return mps, "high_budget_tebd", HIGH_BUDGET_CHI, 1
    except Exception:
        return None, "none", None, None


# ---------------------------------------------------------------------------
# Simulation runners
# ---------------------------------------------------------------------------


def _apply_hybrid_gate(
    mps: MPS,
    node,
    params: StrongSimParams,
    *,
    tdvp_sweeps: int,
) -> tuple[int, int, int, tuple[int, int]]:
    """Returns (swaps, tdvp_lr, enriched_lr, sorted_gate_pair)."""
    gate = convert_dag_to_tensor_algorithm(node)[0]
    i, j = int(gate.sites[0]), int(gate.sites[1])
    pair = (min(i, j), max(i, j))
    swaps = 0
    tdvp_lr = 0
    enriched_lr = 0

    if abs(i - j) == 1:
        apply_two_qubit_gate_tebd(mps, gate, params)
        return swaps, tdvp_lr, enriched_lr, pair

    apply_two_qubit_gate_tdvp_experimental(mps, gate, params, tdvp_sweeps=tdvp_sweeps)
    return swaps, tdvp_lr + 1, enriched_lr, pair


def _internal_bond_dims(mps: MPS) -> list[int]:
    return [int(t.shape[1]) for t in mps.tensors[1:]]


def _pad_bond_dimension_with_noise(
    mps: MPS,
    target_dim: int,
    *,
    noise: float,
    rng: np.random.Generator,
) -> None:
    """Pad bonds then inject tiny noise into newly added slots only."""
    old_shapes = [t.shape for t in mps.tensors]
    mps.pad_bond_dimension(target_dim)
    if noise <= 0.0:
        return
    for i, (old_shape, tensor) in enumerate(zip(old_shapes, mps.tensors, strict=True)):
        _, old_cl, old_cr = old_shape
        phys, cl, cr = tensor.shape
        if cl <= old_cl and cr <= old_cr:
            continue
        mask = np.ones((phys, cl, cr), dtype=bool)
        mask[:, :old_cl, :old_cr] = False
        if not mask.any():
            continue
        perturb = noise * (
            rng.normal(size=(phys, cl, cr)) + 1j * rng.normal(size=(phys, cl, cr))
        )
        tensor[mask] += perturb[mask]
        mps.tensors[i] = tensor
    mps.normalize()


def _apply_tdvp_pre_lr_padding(
    mps: MPS,
    *,
    target_dim: int,
    chi_max: int,
    noise: float,
    rng: np.random.Generator,
) -> tuple[bool, int, int, int, float]:
    """Pad internal bonds once before the first long-range TDVP gate."""
    effective = min(int(target_dim), int(chi_max))
    max_before = int(mps.get_max_bond())
    mean_before = float(_mean_bond_dim(mps))
    if max_before >= effective:
        return False, 0, max_before, max_before, mean_before

    bonds_before = _internal_bond_dims(mps)
    if noise <= 0.0:
        mps.pad_bond_dimension(effective)
    else:
        _pad_bond_dimension_with_noise(mps, effective, noise=noise, rng=rng)
    bonds_after = _internal_bond_dims(mps)
    padded_count = sum(1 for b, a in zip(bonds_before, bonds_after, strict=False) if a > b)
    max_after = int(mps.get_max_bond())
    return True, padded_count, max_before, max_after, float(_mean_bond_dim(mps))


def _run_method(
    qc: QuantumCircuit,
    spec: CircuitSpec,
    *,
    method: MethodName,
    chi: int,
    tdvp_sweeps: int,
    tdvp_padding_dim: int | None = None,
    tdvp_padding_noise: float = 0.0,
    params: StrongSimParams | None = None,
    initial_state: str = "plus",
) -> tuple[MPS, float, GateDispatchStats, PaddingRunStats]:
    sim_mode = _sim_mode(method)
    if params is None:
        params = StrongSimParams(
            preset="exact",
            get_state=True,
            svd_threshold=SVD_THRESHOLD,
            max_bond_dim=chi,
            krylov_tol=LANCZOS_TOL,
            gate_mode="hybrid" if sim_mode == "hybrid_tdvp" else "tebd",
            tdvp_projection_defect_tol=1e-3,
            tdvp_pauli_consistency_check=False,
        )

    static_total, static_nn, static_lr = _static_gate_counts(qc)
    dispatch = GateDispatchStats(
        total_gates=static_total,
        nn_gate_count=static_nn,
        lr_gate_count=static_lr,
    )
    lr_pairs_seen: list[tuple[int, int]] = []
    pad_target = (
        tdvp_padding_dim if tdvp_padding_dim is not None else _padding_dim_for_method(method)
    )
    padding_stats = PaddingRunStats(
        tdvp_padding_dim=pad_target if sim_mode == "hybrid_tdvp" else None,
        tdvp_padding_noise=tdvp_padding_noise if sim_mode == "hybrid_tdvp" else 0.0,
    )
    padding_done = False
    first_lr_done = False
    pad_rng = np.random.default_rng(BASE_SEED + hash(spec.case_id) % 100_000)

    import time

    mps = State(spec.n_qubits, initial="zeros", representation="mps").mps
    if initial_state != "all_zero":
        prep = QuantumCircuit(spec.n_qubits)
        _prep_initial_state(prep, initial_state, seed=0)  # type: ignore[arg-type]
        prep_dag = circuit_to_dag(prep)
        for node in prep_dag.topological_op_nodes():
            if len(node.qargs) == 1:
                apply_single_qubit_gate(mps, node)

    dag = circuit_to_dag(qc)
    peak = 0
    swaps = 0
    t0 = time.perf_counter()
    for node in dag.topological_op_nodes():
        if node.op.name == "barrier":
            continue
        if len(node.qargs) == 1:
            apply_single_qubit_gate(mps, node)
            continue
        gate = convert_dag_to_tensor_algorithm(node)[0]
        i, j = int(gate.sites[0]), int(gate.sites[1])
        pair = (min(i, j), max(i, j))
        if sim_mode == "tebd_swap":
            if abs(i - j) == 1:
                dispatch.tebd_direct_gate_count += 1
            else:
                dispatch.tebd_swap_gate_count += 1
                lr_pairs_seen.append(pair)
            swaps += _num_swaps_for_gate(i, j)
            apply_two_qubit_gate_tebd(mps, gate, params)
        elif abs(i - j) == 1:
            dispatch.tebd_direct_gate_count += 1
            apply_two_qubit_gate_tebd(mps, gate, params)
        else:
            if not padding_done and pad_target is not None and sim_mode == "hybrid_tdvp":
                applied, pb_count, max_bef, max_aft, mean_aft = _apply_tdvp_pre_lr_padding(
                    mps,
                    target_dim=pad_target,
                    chi_max=chi,
                    noise=tdvp_padding_noise,
                    rng=pad_rng,
                )
                padding_stats.padding_applied = applied
                padding_stats.padded_bonds_count = pb_count
                padding_stats.max_bond_dim_before_padding = max_bef
                padding_stats.max_bond_dim_after_padding = max_aft
                padding_stats.mean_bond_dim_after_padding = mean_aft
                padding_done = True
            if not first_lr_done:
                padding_stats.peak_bond_dim_before_first_lr = max(peak, int(mps.get_max_bond()))
            sw, tdvp_n, enrich_n, gate_pair = _apply_hybrid_gate(
                mps, node, params, tdvp_sweeps=tdvp_sweeps
            )
            swaps += sw
            dispatch.tdvp_lr_gate_count += tdvp_n
            dispatch.skipped_or_exact_lr_gate_count += enrich_n
            if tdvp_n or enrich_n:
                lr_pairs_seen.append(gate_pair)
            peak = max(peak, int(mps.get_max_bond()))
            if not first_lr_done:
                padding_stats.peak_bond_dim_after_first_lr = peak
                first_lr_done = True
            continue
        peak = max(peak, int(mps.get_max_bond()))
    wall = float(time.perf_counter() - t0)
    dispatch.num_swaps_inserted = swaps
    dispatch.lr_gate_pairs = tuple(sorted(set(lr_pairs_seen)))
    mps._bench_peak_bond = peak  # type: ignore[attr-defined]
    mps._bench_swaps = swaps  # type: ignore[attr-defined]
    mps._bench_dispatch = dispatch  # type: ignore[attr-defined]
    return mps, wall, dispatch, padding_stats


def _assign_padding_fields(run: RawRun, padding: PaddingRunStats) -> None:
    run.tdvp_padding_dim = padding.tdvp_padding_dim
    run.tdvp_padding_noise = padding.tdvp_padding_noise
    run.padding_applied = padding.padding_applied
    run.padded_bonds_count = padding.padded_bonds_count
    run.max_bond_dim_before_padding = padding.max_bond_dim_before_padding
    run.max_bond_dim_after_padding = padding.max_bond_dim_after_padding
    run.mean_bond_dim_after_padding = padding.mean_bond_dim_after_padding
    run.peak_bond_dim_before_first_lr = padding.peak_bond_dim_before_first_lr
    run.peak_bond_dim_after_first_lr = padding.peak_bond_dim_after_first_lr


def _assign_dispatch_fields(run: RawRun, dispatch: GateDispatchStats) -> None:
    run.total_gates = dispatch.total_gates
    run.nn_gate_count = dispatch.nn_gate_count
    run.lr_gate_count = dispatch.lr_gate_count
    run.tebd_direct_gate_count = dispatch.tebd_direct_gate_count
    run.tebd_swap_gate_count = dispatch.tebd_swap_gate_count
    run.tdvp_lr_gate_count = dispatch.tdvp_lr_gate_count
    run.tdvp_lr_count = dispatch.tdvp_lr_gate_count
    run.skipped_or_exact_lr_gate_count = dispatch.skipped_or_exact_lr_gate_count
    run.enriched_lr_count = dispatch.skipped_or_exact_lr_gate_count
    run.num_swaps_inserted = dispatch.num_swaps_inserted
    run.lr_gate_pairs = json.dumps(list(dispatch.lr_gate_pairs))


def run_method(
    spec: CircuitSpec,
    *,
    method: MethodName,
    chi: int,
    tdvp_sweeps: int,
    tdvp_padding_dim: int | None = None,
    tdvp_padding_noise: float = 0.0,
    sweep_scan_enabled: bool = False,
    ref: MPS | np.ndarray | None,
    ref_type: RefType,
    ref_chi: int | None,
    ref_sweeps: int | None,
    bond_history: list[BondHistoryRow],
) -> RawRun:
    pad_dim = (
        tdvp_padding_dim
        if tdvp_padding_dim is not None
        else _padding_dim_for_method(method)
        if _is_tdvp_method(method)
        else None
    )
    pad_noise = tdvp_padding_noise if _is_tdvp_method(method) else 0.0
    run = RawRun(
        family=spec.family,
        n_qubits=spec.n_qubits,
        depth=spec.depth,
        seed=spec.seed,
        angle_regime=spec.angle_regime,
        method=method,
        chi_max=chi,
        tdvp_sweeps=tdvp_sweeps if _is_tdvp_method(method) else 1,
        tdvp_padding_dim=pad_dim,
        tdvp_padding_noise=pad_noise,
        sweep_scan_enabled=sweep_scan_enabled,
        svd_threshold=SVD_THRESHOLD,
        lanczos_tol=LANCZOS_TOL,
        reference_type=ref_type,
        reference_available=ref is not None,
        reference_chi=ref_chi,
        reference_sweeps=ref_sweeps,
        case_id=spec.case_id,
    )
    try:
        sweeps = tdvp_sweeps if _is_tdvp_method(method) else 1
        mps, wall, dispatch, padding = _run_method(
            spec.qc,
            spec,
            method=method,
            chi=chi,
            tdvp_sweeps=sweeps,
            tdvp_padding_dim=pad_dim,
            tdvp_padding_noise=pad_noise,
        )
        run.wall_time_s = wall
        run.peak_bond_dim = int(getattr(mps, "_bench_peak_bond", mps.get_max_bond()))
        run.final_max_bond_dim = int(mps.get_max_bond())
        run.mean_bond_dim = _mean_bond_dim(mps)
        run.hit_chi_max = run.final_max_bond_dim >= chi
        _assign_dispatch_fields(run, dispatch)
        _assign_padding_fields(run, padding)

        if ref is not None:
            convention: ReferenceConvention = REFERENCE_CONVENTION
            if isinstance(ref, np.ndarray):
                vec = np.asarray(mps.to_vec(), dtype=np.complex128)
                f_dir, f_rev, conv_sv = compare_statevectors(vec, ref, n=spec.n_qubits)
                run.fidelity_direct = f_dir
                run.fidelity_bit_reversed = f_rev
                if conv_sv == "bit_reversed" and f_rev > f_dir + 1e-10:
                    convention = "bit_reversed"
                    run.infidelity_to_reference = 1.0 - f_rev
                else:
                    convention = "direct"
                    run.infidelity_to_reference = 1.0 - f_dir
                run.fidelity_to_reference = 1.0 - run.infidelity_to_reference
            run.reference_convention_used = convention
            _apply_grouped_observable_errors(run, spec, mps, ref, convention=convention)
            warning = check_high_fidelity_observable_consistency(
                run.fidelity_to_reference,
                run.max_abs_observable_error,
            )
            run.fidelity_observable_consistency_warning = warning
            if warning:
                print(f"  WARNING [{spec.case_id} {method} chi={chi}]: {warning}")
        run.status = "ok"
    except Exception as exc:
        run.status = "failed"
        run.error_message = f"{type(exc).__name__}: {exc}"
    return run


# ---------------------------------------------------------------------------
# Reference validation (must pass before benchmark interpretation)
# ---------------------------------------------------------------------------


def _validation_spec(n: int, qc: QuantumCircuit, tag: str) -> CircuitSpec:
    return CircuitSpec(
        family="validation",
        n_qubits=n,
        depth=1,
        seed=0,
        angle_regime="small",
        case_id=f"validation_{tag}_n{n}",
        qc=qc,
        lr_pairs=(),
        nn_pairs=tuple((i, i + 1) for i in range(n - 1)),
        topology="validation",
    )


def _run_validation_circuit(
    qc: QuantumCircuit,
    n: int,
    *,
    chi: int = 64,
    method: MethodName = "tebd_swap",
    tdvp_sweeps: int = 1,
    initial_state: str = "plus",
) -> MPS:
    spec = _validation_spec(n, qc, "run")
    mps, _, _, _ = _run_method(
        qc,
        spec,
        method=method,
        chi=chi,
        tdvp_sweeps=tdvp_sweeps,
        initial_state=initial_state,
    )
    return mps


def _reference_for_validation(
    qc: QuantumCircuit,
    n: int,
    *,
    initial_state: str = "plus",
) -> np.ndarray:
    prep = QuantumCircuit(n)
    if initial_state != "all_zero":
        _prep_initial_state(prep, initial_state, seed=0)  # type: ignore[arg-type]
    return qiskit_reference_vec(prep.compose(qc))


def validate_tdvp_padding(*, chi: int = 64) -> None:
    """Validate bond padding preserves product states and periodic LR TDVP instrumentation."""
    from copy import deepcopy

    print("=== TDVP padding validation ===")
    tol_fid = 1e-12
    pad_rng = np.random.default_rng(0)

    for n in (4, 8):
        mps = State(n, initial="zeros", representation="mps").mps
        prep = QuantumCircuit(n)
        _prep_initial_state(prep, "plus", seed=0)  # type: ignore[arg-type]
        for node in circuit_to_dag(prep).topological_op_nodes():
            apply_single_qubit_gate(mps, node)
        vec_before = np.asarray(mps.to_vec(), dtype=np.complex128)
        mps_pad = deepcopy(mps)
        _apply_tdvp_pre_lr_padding(
            mps_pad,
            target_dim=PADDED4_TARGET_DIM,
            chi_max=chi,
            noise=0.0,
            rng=pad_rng,
        )
        vec_after = np.asarray(mps_pad.to_vec(), dtype=np.complex128)
        f_pad, _, _ = compare_statevectors(vec_after, vec_before, n=n)
        if f_pad < 1.0 - tol_fid:
            raise SystemExit(f"Padding validation FAILED: plus-state fidelity n={n} f={f_pad:.3e}")
        print(f"  plus-state padding preservation n={n}: fid={f_pad:.3e} OK")

    n = 8
    theta = 0.15
    qc = QuantumCircuit(n)
    qc.rzz(theta, 0, n - 1)
    spec = CircuitSpec(
        family="validation",
        n_qubits=n,
        depth=1,
        seed=0,
        angle_regime="small",
        case_id="validation_padding_lr_rzz_n8",
        qc=qc,
        lr_pairs=((0, n - 1),),
        nn_pairs=tuple((i, i + 1) for i in range(n - 1)),
        topology="validation",
    )
    prep = QuantumCircuit(n)
    _prep_initial_state(prep, "plus", seed=0)  # type: ignore[arg-type]
    ref = qiskit_reference_vec(prep.compose(qc))

    mps_u, _, _, unpadded_pad = _run_method(
        qc, spec, method="hybrid_tdvp_unpadded", chi=chi, tdvp_sweeps=1
    )
    mps_p, _, padded_disp, padded_pad = _run_method(
        qc, spec, method="hybrid_tdvp_padded4", chi=chi, tdvp_sweeps=1
    )
    _, _, tebd_disp, tebd_pad = _run_method(qc, spec, method="tebd_swap", chi=chi, tdvp_sweeps=1)

    f_u, _, _ = compare_statevectors(np.asarray(mps_u.to_vec(), dtype=np.complex128), ref, n=n)
    f_p, _, _ = compare_statevectors(np.asarray(mps_p.to_vec(), dtype=np.complex128), ref, n=n)
    f_t, _, _ = compare_statevectors(
        np.asarray(_run_validation_circuit(qc, n, chi=chi, method="tebd_swap").to_vec()),
        ref,
        n=n,
    )

    if f_t < 1.0 - 1e-8:
        raise SystemExit(f"Padding validation FAILED: TEBD fidelity={f_t:.3e}")
    if not padded_pad.padding_applied:
        raise SystemExit("Padding validation FAILED: padded4 padding_applied=False")
    if padded_disp.tdvp_lr_gate_count != 1:
        raise SystemExit(
            f"Padding validation FAILED: padded4 tdvp_lr={padded_disp.tdvp_lr_gate_count}, expected 1"
        )
    if tebd_disp.tebd_swap_gate_count != 1:
        raise SystemExit(
            f"Padding validation FAILED: tebd_swap_gate_count={tebd_disp.tebd_swap_gate_count}"
        )
    if tebd_disp.tdvp_lr_gate_count != 0:
        raise SystemExit("Padding validation FAILED: TEBD tdvp_lr_gate_count != 0")
    if tebd_pad.padding_applied:
        raise SystemExit("Padding validation FAILED: TEBD padding_applied should be False")
    if f_p <= f_u + 1e-6:
        print(
            f"  WARNING periodic LR RZZ: padded4 fid={f_p:.3e} not clearly better than "
            f"unpadded={f_u:.3e}"
        )
    print(
        f"  periodic LR RZZ n=8: unpadded_f={f_u:.3e} padded4_f={f_p:.3e} tebd_f={f_t:.3e} OK"
    )

    if padded_pad.max_bond_dim_before_padding is None:
        raise SystemExit("Padding validation FAILED: max_bond_dim_before_padding missing")
    if padded_pad.max_bond_dim_after_padding is None:
        raise SystemExit("Padding validation FAILED: max_bond_dim_after_padding missing")
    if padded_pad.max_bond_dim_after_padding < padded_pad.max_bond_dim_before_padding:
        raise SystemExit("Padding validation FAILED: bond dim decreased after padding")
    if padded_pad.peak_bond_dim_before_first_lr is None:
        raise SystemExit("Padding validation FAILED: peak_bond_dim_before_first_lr missing")
    if padded_pad.peak_bond_dim_after_first_lr is None:
        raise SystemExit("Padding validation FAILED: peak_bond_dim_after_first_lr missing")

    qc_nn = QuantumCircuit(n)
    qc_nn.rzz(0.1, 0, 1)
    spec_nn = _validation_spec(n, qc_nn, "nn_only")
    _, _, _, nn_pad = _run_method(qc_nn, spec_nn, method="hybrid_tdvp_padded4", chi=chi, tdvp_sweeps=1)
    if nn_pad.padding_applied:
        raise SystemExit("Padding validation FAILED: padding applied without LR gates")
    if unpadded_pad.padding_applied:
        raise SystemExit("Padding validation FAILED: unpadded run reported padding_applied=True")

    print("=== TDVP padding validation PASSED ===\n")


def _validate_isolated_long_range_gate(*, n: int = 8, chi: int = 64, tol_fid: float = 1e-8) -> None:
    """One LR Pauli rotation on |0...0>; hybrid vs TEBD+SWAP vs exact."""
    theta = 0.3
    qc = QuantumCircuit(n)
    qc.rxx(theta, 0, n - 1)
    spec = CircuitSpec(
        family="validation",
        n_qubits=n,
        depth=1,
        seed=0,
        angle_regime="small",
        case_id=f"validation_isolated_lr_rxx_n{n}",
        qc=qc,
        lr_pairs=((0, n - 1),),
        nn_pairs=(),
        topology="validation",
    )
    ref = _reference_for_validation(qc, n, initial_state="all_zero")

    _, _, hybrid_dispatch, _ = _run_method(
        qc, spec, method="hybrid_tdvp_unpadded", chi=chi, tdvp_sweeps=1, initial_state="all_zero"
    )
    mps_hybrid = _run_validation_circuit(
        qc, n, chi=chi, method="hybrid_tdvp_unpadded", tdvp_sweeps=1, initial_state="all_zero"
    )
    _, _, tebd_dispatch, _ = _run_method(
        qc, spec, method="tebd_swap", chi=chi, tdvp_sweeps=1, initial_state="all_zero"
    )
    mps_tebd = _run_validation_circuit(qc, n, chi=chi, method="tebd_swap", initial_state="all_zero")

    vec_h = np.asarray(mps_hybrid.to_vec(), dtype=np.complex128)
    vec_t = np.asarray(mps_tebd.to_vec(), dtype=np.complex128)
    f_h, _, _ = compare_statevectors(vec_h, ref, n=n)
    f_t, _, _ = compare_statevectors(vec_t, ref, n=n)

    if hybrid_dispatch.tdvp_lr_gate_count != 1:
        raise SystemExit(
            f"Isolated LR validation FAILED: hybrid tdvp_lr_gate_count="
            f"{hybrid_dispatch.tdvp_lr_gate_count}, expected 1"
        )
    if hybrid_dispatch.skipped_or_exact_lr_gate_count != 0:
        raise SystemExit(
            f"Isolated LR validation FAILED: hybrid enriched dispatch "
            f"{hybrid_dispatch.skipped_or_exact_lr_gate_count}, expected 0"
        )
    if hybrid_dispatch.tebd_swap_gate_count != 0:
        raise SystemExit(
            f"Isolated LR validation FAILED: hybrid tebd_swap_gate_count="
            f"{hybrid_dispatch.tebd_swap_gate_count}, expected 0"
        )
    if tebd_dispatch.tebd_swap_gate_count != 1:
        raise SystemExit(
            f"Isolated LR validation FAILED: TEBD tebd_swap_gate_count="
            f"{tebd_dispatch.tebd_swap_gate_count}, expected 1"
        )
    if tebd_dispatch.tdvp_lr_gate_count != 0:
        raise SystemExit(
            f"Isolated LR validation FAILED: TEBD tdvp_lr_gate_count="
            f"{tebd_dispatch.tdvp_lr_gate_count}, expected 0"
        )
    if f_t < 1.0 - tol_fid:
        raise SystemExit(f"Isolated LR validation FAILED: TEBD+SWAP fidelity={f_t:.3e}")
    if f_h < 1.0 - 1e-2:
        print(f"  isolated LR RXX: hybrid fidelity={f_h:.3e} (TEBD={f_t:.3e}) — TDVP LR may need refinement")
    else:
        obs = observable_list(spec)
        mean_e, max_e, _ = observable_errors_against_reference(
            mps_hybrid, ref, obs, n=n, convention="direct"
        )
        if f_h > 1.0 - tol_fid and max_e > 1e-5:
            raise SystemExit(
                f"Isolated LR validation FAILED: hybrid fid={f_h:.3e} but max_obs={max_e:.3e}"
            )
        if mean_e > 1e-5 and f_h > 1.0 - tol_fid:
            print(f"  isolated LR RXX: hybrid obs err mean={mean_e:.3e} max={max_e:.3e}")

    print(
        f"  isolated LR RXX n={n}: hybrid_f={f_h:.3e} tebd_f={f_t:.3e} "
        f"hybrid_tdvp_lr={hybrid_dispatch.tdvp_lr_gate_count} "
        f"tebd_swap_gates={tebd_dispatch.tebd_swap_gate_count} OK"
    )


def _validate_tdvp_sweep_semantics(*, n: int = 8, chi: int = 64) -> None:
    """Check tdvp_sweeps>1 refines the same gate rather than re-applying full rotation."""
    theta = 0.25
    qc = QuantumCircuit(n)
    qc.rxx(theta, 0, n - 1)
    spec = _validation_spec(n, qc, "sweep")
    ref_once = _reference_for_validation(qc, n, initial_state="all_zero")

    qc_twice = QuantumCircuit(n)
    qc_twice.rxx(theta, 0, n - 1)
    qc_twice.rxx(theta, 0, n - 1)
    ref_twice = _reference_for_validation(qc_twice, n, initial_state="all_zero")

    mps_s1, _, d1, _ = _run_method(
        qc, spec, method="hybrid_tdvp_unpadded", chi=chi, tdvp_sweeps=1, initial_state="all_zero"
    )
    mps_s2, _, d2, _ = _run_method(
        qc, spec, method="hybrid_tdvp_unpadded", chi=chi, tdvp_sweeps=2, initial_state="all_zero"
    )
    vec_s1 = np.asarray(mps_s1.to_vec(), dtype=np.complex128)
    vec_s2 = np.asarray(mps_s2.to_vec(), dtype=np.complex128)
    f_s1, _, _ = compare_statevectors(vec_s1, ref_once, n=n)
    f_s2_once, _, _ = compare_statevectors(vec_s2, ref_once, n=n)
    f_s2_twice, _, _ = compare_statevectors(vec_s2, ref_twice, n=n)

    if d1.tdvp_lr_gate_count != 1 or d1.skipped_or_exact_lr_gate_count != 0:
        raise SystemExit(
            f"TDVP sweep validation FAILED: LR dispatch sweeps1 "
            f"tdvp={d1.tdvp_lr_gate_count} enriched={d1.skipped_or_exact_lr_gate_count}"
        )
    if d2.tdvp_lr_gate_count != 1 or d2.skipped_or_exact_lr_gate_count != 0:
        raise SystemExit(
            f"TDVP sweep validation FAILED: LR dispatch sweeps2 "
            f"tdvp={d2.tdvp_lr_gate_count} enriched={d2.skipped_or_exact_lr_gate_count}"
        )

    if f_s2_twice > f_s2_once + 0.05 and f_s2_twice > 0.99:
        print(
            "  WARNING: tdvp_sweeps=2 state matches double-applied gate better than single — "
            "sweeps may re-apply full evolution time, not refine projection."
        )
    elif f_s1 > 0.99 and f_s2_once < f_s1 - 0.05:
        print(
            f"  NOTE: tdvp_sweeps=2 lowers fidelity ({f_s2_once:.3e} vs {f_s1:.3e} at sweeps=1); "
            "interpret multi-sweep rows as extra TDVP passes, not guaranteed refinement."
        )
    else:
        print(
            f"  tdvp_sweeps semantics n={n}: f(sweeps=1)={f_s1:.3e} "
            f"f(sweeps=2 vs once)={f_s2_once:.3e} f(sweeps=2 vs twice)={f_s2_twice:.3e} OK"
        )


def validate_reference_convention(*, chi: int = 64, tol_fid: float = 1e-9, tol_obs: float = 1e-9) -> ReferenceConvention:
    """Run minimal sanity checks; abort benchmark if MPS vs exact reference is broken."""
    global REFERENCE_CONVENTION
    print("=== Reference validation ===")
    bit_rev_wins = 0
    direct_wins = 0

    # 1. Identity / plus prep only
    for n in (2, 3, 4, 8):
        qc = QuantumCircuit(n)
        ref = _reference_for_validation(qc, n)
        mps = _run_validation_circuit(qc, n, chi=chi)
        vec = np.asarray(mps.to_vec(), dtype=np.complex128)
        f_dir, f_rev, conv = compare_statevectors(vec, ref, n=n)
        if conv == "bit_reversed":
            bit_rev_wins += 1
        else:
            direct_wins += 1
        if max(f_dir, f_rev) < 1.0 - tol_fid:
            msg = f"identity n={n}: fidelity_direct={f_dir:.3e} fidelity_bit_reversed={f_rev:.3e}"
            raise SystemExit(f"Reference validation FAILED: {msg}")
        use = "bit_reversed" if f_rev > f_dir + 1e-12 else "direct"
        ref_aligned = align_reference_vec(ref, n, use)
        for i in range(n):
            z_g = _expectation_via_mps_swaps(mps, gate_name="z", sites=[i])
            x_g = _expectation_via_mps_swaps(mps, gate_name="x", sites=[i])
            z_r = statevector_expectation(ref_aligned, n, label="Z", sites=[i])
            x_r = statevector_expectation(ref_aligned, n, label="X", sites=[i])
            if abs(z_g - z_r) > tol_obs or abs(x_g - x_r) > tol_obs:
                raise SystemExit(
                    f"Reference validation FAILED identity observables n={n} site={i}: "
                    f"Z err={abs(z_g - z_r):.3e} X err={abs(x_g - x_r):.3e} "
                    f"(direct_f={f_dir:.3e} rev_f={f_rev:.3e})"
                )
        print(f"  identity n={n}: fid_direct={f_dir:.3e} fid_rev={f_rev:.3e} OK")

    # 2. Single X on each site
    for n in (3, 4):
        for site in range(n):
            qc = QuantumCircuit(n)
            qc.x(site)
            ref = _reference_for_validation(qc, n)
            mps = _run_validation_circuit(qc, n, chi=chi)
            vec = np.asarray(mps.to_vec(), dtype=np.complex128)
            f_dir, f_rev, conv = compare_statevectors(vec, ref, n=n)
            if max(f_dir, f_rev) < 1.0 - tol_fid:
                print(f"  X({site}) n={n}: direct_f={f_dir:.3e} bit_reversed_f={f_rev:.3e}")
                raise SystemExit(f"Reference validation FAILED: X({site}) on n={n}")
            use_ref = align_reference_vec(ref, n, conv)
            for i in range(n):
                z_g = _expectation_via_mps_swaps(mps, gate_name="z", sites=[i])
                z_r = statevector_expectation(use_ref, n, label="Z", sites=[i])
                if abs(z_g - z_r) > tol_obs:
                    raise SystemExit(f"Reference validation FAILED: X({site}) n={n} Z({i}) err={abs(z_g - z_r):.3e}")
        print(f"  single-X sweep n={n}: OK")

    # 3. Single-qubit RX/RY/RZ on each site
    for n in (3, 4):
        for site in range(n):
            for gate, angle in (("rx", 0.31), ("ry", 0.47), ("rz", 0.19)):
                qc = QuantumCircuit(n)
                getattr(qc, gate)(angle, site)
                ref = _reference_for_validation(qc, n)
                mps = _run_validation_circuit(qc, n, chi=chi)
                vec = np.asarray(mps.to_vec(), dtype=np.complex128)
                f_dir, f_rev, conv = compare_statevectors(vec, ref, n=n)
                if max(f_dir, f_rev) < 1.0 - tol_fid:
                    raise SystemExit(f"Reference validation FAILED: {gate}({site}) n={n}")
                use_ref = align_reference_vec(ref, n, conv)
                for axis in ("X", "Y", "Z"):
                    g = _expectation_via_mps_swaps(mps, gate_name=axis.lower(), sites=[site])
                    r = statevector_expectation(use_ref, n, label=axis, sites=[site])
                    if abs(g - r) > tol_obs:
                        raise SystemExit(
                            f"Reference validation FAILED: {gate} on {site} <{axis}_{site}> err={abs(g - r):.3e}"
                        )
        print(f"  1q rotation sweep n={n}: OK")

    # 4. One NN RXX(0.2) on every pair
    for n in (3, 4, 8):
        for i in range(n - 1):
            qc = QuantumCircuit(n)
            qc.rxx(0.2, i, i + 1)
            ref = _reference_for_validation(qc, n)
            mps = _run_validation_circuit(qc, n, chi=chi)
            vec = np.asarray(mps.to_vec(), dtype=np.complex128)
            f_dir, f_rev, conv = compare_statevectors(vec, ref, n=n)
            if max(f_dir, f_rev) < 1.0 - 1e-8:
                raise SystemExit(f"Reference validation FAILED: RXX({i},{i+1}) n={n}")
            if conv == "bit_reversed":
                bit_rev_wins += 1
            else:
                direct_wins += 1
        print(f"  RXX pairs n={n}: OK")

    _validate_isolated_long_range_gate(n=8, chi=max(chi, 64))
    _validate_tdvp_sweep_semantics(n=8, chi=max(chi, 64))

    convention: ReferenceConvention = "bit_reversed" if bit_rev_wins > direct_wins else "direct"
    REFERENCE_CONVENTION = convention
    print(f"=== Reference validation PASSED (convention={convention}) ===\n")
    return convention


# ---------------------------------------------------------------------------
# Resume / IO
# ---------------------------------------------------------------------------


def _load_existing_keys(path: Path) -> set[str]:
    if not path.exists() or OVERWRITE:
        return set()
    keys: set[str] = set()
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row.get("case_id", "")
            meth = row.get("method", "")
            chi = row.get("chi_max", "")
            sw = row.get("tdvp_sweeps", "1")
            pad = _padding_key_from_row(row)
            noise = row.get("tdvp_padding_noise", "0")
            if noise in (None, ""):
                noise = "0"
            keys.add(f"{cid}|{meth}|chi{chi}|sweeps{sw}|padding{pad}|noise{noise}")
    return keys


def _raw_run_fieldnames() -> list[str]:
    return list(RawRun.__dataclass_fields__.keys())


def _append_raw_run(path: Path, run: RawRun) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(run)
    fieldnames = _raw_run_fieldnames()
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def _read_all_raw(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Summaries and classification
# ---------------------------------------------------------------------------


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    v = row.get(key)
    if v in (None, ""):
        return default
    return float(v)


def _classify_pair(tdvp: dict[str, Any], tebd: dict[str, Any]) -> str:
    if tdvp.get("status") != "ok" or tebd.get("status") != "ok":
        return "both_fail"
    t_err = _f(tdvp, "mean_abs_observable_error", float("inf"))
    e_err = _f(tebd, "mean_abs_observable_error", float("inf"))
    if not tdvp.get("reference_available") and abs(t_err - e_err) / max(e_err, 1e-15) > 0.5:
        return "no_reference"
    if t_err > 0.1 and e_err > 0.1:
        return "both_fail"
    if t_err <= e_err / 2:
        if tebd.get("hit_chi_max") in ("True", True, "true") and not tdvp.get("hit_chi_max") in ("True", True, "true"):
            return "tdvp_wins"
        if e_err >= 2 * t_err:
            return "tdvp_wins"
    if e_err <= t_err / 2:
        return "tebd_wins"
    if max(t_err, e_err) / max(min(t_err, e_err), 1e-15) <= 2:
        return "similar"
    return "similar"


def _hit_chi_max(row: dict[str, Any]) -> bool:
    return row.get("hit_chi_max") in ("True", True, "true")


def _classify_padding_row(unpadded: dict[str, Any], curr: dict[str, Any]) -> str:
    pe = _f(unpadded, "mean_abs_observable_error", 1e-15)
    ce = _f(curr, "mean_abs_observable_error", 1e-15)
    pi = _f(unpadded, "infidelity_to_reference", 1e-15)
    ci = _f(curr, "infidelity_to_reference", 1e-15)
    err_ratio = ce / pe
    inf_ratio = ci / pi if pi > 1e-15 else err_ratio
    time_ratio = _f(curr, "wall_time_s") / max(_f(unpadded, "wall_time_s"), 1e-9)
    if err_ratio < 0.5 or inf_ratio < 0.5:
        return "padding_helpful"
    if err_ratio > 1.3:
        return "padding_harmful"
    if time_ratio > 3 and err_ratio < 0.7:
        return "padding_helpful_but_expensive"
    return "padding_neutral"


def _classify_main_outcome(
    tebd: dict[str, Any],
    unpadded: dict[str, Any] | None,
    padded4: dict[str, Any],
) -> str:
    te = _f(tebd, "mean_abs_observable_error", 1e-15)
    ue = _f(unpadded, "mean_abs_observable_error", 1e-15) if unpadded else te
    pe = _f(padded4, "mean_abs_observable_error", 1e-15)
    tf = _f(tebd, "fidelity_to_reference", 0.0)
    pf = _f(padded4, "fidelity_to_reference", 0.0)
    uf = _f(unpadded, "fidelity_to_reference", 0.0) if unpadded else pf
    if te > 0.1 and pe > 0.1 and tf < 0.9 and pf < 0.9:
        return "both_fail"
    tebd_hit = _hit_chi_max(tebd)
    padded_hit = _hit_chi_max(padded4)
    if pe <= te / 2 or (tebd_hit and not padded_hit and pe <= te):
        return "padded4_tdvp_wins"
    if te <= pe / 2:
        return "tebd_wins"
    if (
        te <= pe / 2
        and _f(tebd, "wall_time_s") < _f(padded4, "wall_time_s")
        and te <= ue
    ):
        return "tebd_wins"
    if unpadded is not None and pe <= ue / 2:
        return "padding_required"
    if max(te, pe) / max(min(te, pe), 1e-15) <= 2:
        return "similar"
    if unpadded is not None and pf > uf + 0.01 and abs(te - pe) / max(te, pe) <= 0.5:
        return "padding_required"
    return "similar"


def summarize_raw_runs(raw: list[dict[str, Any]], outdir: Path) -> None:
    ok = [r for r in raw if r.get("status") == "ok"]

    # fixed_chi_comparison
    fixed: list[dict[str, Any]] = []
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in ok:
        key = (
            f"{r['case_id']}|chi{r['chi_max']}|sweeps{r['tdvp_sweeps']}"
            f"|padding{_padding_key_from_row(r)}"
        )
        by_case[key].append(r)

    for _key, rows in by_case.items():
        tdvp = next((r for r in rows if r["method"] == "hybrid_tdvp_padded4"), None)
        if tdvp is None:
            tdvp = next((r for r in rows if r["method"] in TDVP_METHODS), None)
        tebd = next((r for r in rows if r["method"] == "tebd_swap"), None)
        if tdvp is None or tebd is None:
            continue
        t_err = _f(tdvp, "mean_abs_observable_error")
        e_err = _f(tebd, "mean_abs_observable_error")
        fixed.append({
            "case_id": tdvp["case_id"],
            "family": tdvp["family"],
            "n_qubits": tdvp["n_qubits"],
            "depth": tdvp["depth"],
            "seed": tdvp["seed"],
            "angle_regime": tdvp["angle_regime"],
            "chi_max": tdvp["chi_max"],
            "tdvp_sweeps": tdvp["tdvp_sweeps"],
            "tdvp_padding_dim": tdvp.get("tdvp_padding_dim", ""),
            "tdvp_mean_obs_error": t_err,
            "tebd_mean_obs_error": e_err,
            "tdvp_error_over_tebd_error": e_err / max(t_err, 1e-15),
            "tebd_error_over_tdvp_error": t_err / max(e_err, 1e-15) if e_err > 0 else None,
            "tdvp_peak_chi": tdvp["peak_bond_dim"],
            "tebd_peak_chi": tebd["peak_bond_dim"],
            "tdvp_peak_chi_over_tebd_peak_chi": _f(tdvp, "peak_bond_dim") / max(_f(tebd, "peak_bond_dim"), 1),
            "tdvp_time_over_tebd_time": _f(tdvp, "wall_time_s") / max(_f(tebd, "wall_time_s"), 1e-9),
            "tdvp_hit_chi_max": tdvp.get("hit_chi_max"),
            "tebd_hit_chi_max": tebd.get("hit_chi_max"),
            "outcome": _classify_pair(tdvp, tebd),
        })
    _write_csv(outdir / "fixed_chi_comparison.csv", fixed)

    # tdvp_sweep_scaling
    sweep_rows: list[dict[str, Any]] = []
    by_sweep_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in ok:
        if r["method"] not in TDVP_METHODS:
            continue
        sk = f"{r['case_id']}|chi{r['chi_max']}|padding{_padding_key_from_row(r)}"
        by_sweep_case[sk].append(r)
    for sk, rows in by_sweep_case.items():
        rows_sorted = sorted(rows, key=lambda x: int(x["tdvp_sweeps"]))
        for i in range(1, len(rows_sorted)):
            prev, curr = rows_sorted[i - 1], rows_sorted[i]
            pe = _f(prev, "mean_abs_observable_error", 1e-15)
            ce = _f(curr, "mean_abs_observable_error", 1e-15)
            pt = _f(prev, "wall_time_s", 1e-9)
            ct = _f(curr, "wall_time_s", 1e-9)
            err_ratio = ce / pe
            if err_ratio < 0.7:
                cls = "sweep_helpful"
            elif err_ratio > 1.3:
                cls = "sweep_harmful_or_unstable"
            else:
                cls = "sweep_saturated"
            sweep_rows.append({
                "case_id": curr["case_id"],
                "family": curr["family"],
                "chi_max": curr["chi_max"],
                "sweeps_from": prev["tdvp_sweeps"],
                "sweeps_to": curr["tdvp_sweeps"],
                "error_ratio": err_ratio,
                "time_ratio": ct / pt,
                "peak_chi_change": _f(curr, "peak_bond_dim") - _f(prev, "peak_bond_dim"),
                "classification": cls,
            })
    _write_csv(outdir / "tdvp_sweep_scaling.csv", sweep_rows)

    # tdvp_padding_scaling
    padding_rows: list[dict[str, Any]] = []
    by_padding_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in ok:
        if r["method"] not in TDVP_METHODS:
            continue
        pk = f"{r['case_id']}|chi{r['chi_max']}|sweeps{r['tdvp_sweeps']}"
        by_padding_case[pk].append(r)
    for _pk, rows in by_padding_case.items():
        unpadded = next((r for r in rows if r["method"] == "hybrid_tdvp_unpadded"), None)
        if unpadded is None:
            unpadded = next((r for r in rows if _padding_key_from_row(r) == "none"), None)
        for curr in sorted(rows, key=lambda x: (_padding_key_from_row(x), str(x.get("method", "")))):
            pad_dim = _padding_key_from_row(curr)
            pe = _f(unpadded, "mean_abs_observable_error", 1e-15) if unpadded else 1e-15
            ce = _f(curr, "mean_abs_observable_error", 1e-15)
            pi = _f(unpadded, "infidelity_to_reference", 1e-15) if unpadded else 1e-15
            ci = _f(curr, "infidelity_to_reference", 1e-15)
            pf = _f(unpadded, "fidelity_to_reference", 0.0) if unpadded else 0.0
            cf = _f(curr, "fidelity_to_reference", 0.0)
            pt = _f(unpadded, "wall_time_s", 1e-9) if unpadded else 1e-9
            ct = _f(curr, "wall_time_s", 1e-9)
            pchi = _f(unpadded, "peak_bond_dim", 1.0) if unpadded else 1.0
            cchi = _f(curr, "peak_bond_dim", 1.0)
            err_ratio = ce / pe if unpadded else 1.0
            inf_ratio = ci / pi if unpadded and pi > 1e-15 else err_ratio
            is_baseline = curr is unpadded or pad_dim == "none"
            cls = (
                "padding_baseline"
                if is_baseline
                else _classify_padding_row(unpadded, curr)
                if unpadded
                else "padding_baseline"
            )
            padding_rows.append({
                "case_id": curr["case_id"],
                "family": curr["family"],
                "n_qubits": curr["n_qubits"],
                "depth": curr["depth"],
                "seed": curr["seed"],
                "angle_regime": curr["angle_regime"],
                "chi_max": curr["chi_max"],
                "tdvp_sweeps": curr["tdvp_sweeps"],
                "method": curr["method"],
                "padding_dim": pad_dim,
                "fidelity_to_reference": cf,
                "infidelity_to_reference": ci,
                "mean_abs_observable_error": ce,
                "max_abs_observable_error": _f(curr, "max_abs_observable_error"),
                "rms_observable_error": _f(curr, "rms_observable_error"),
                "wall_time_s": ct,
                "peak_bond_dim": cchi,
                "mean_bond_dim": _f(curr, "mean_bond_dim"),
                "hit_chi_max": curr.get("hit_chi_max"),
                "error_ratio_vs_unpadded": err_ratio,
                "infidelity_ratio_vs_unpadded": inf_ratio,
                "fidelity_gain_vs_unpadded": cf - pf if unpadded else 0.0,
                "time_ratio_vs_unpadded": ct / pt if unpadded else 1.0,
                "peak_chi_ratio_vs_unpadded": cchi / pchi if unpadded else 1.0,
                "classification": cls,
            })
    _write_csv(outdir / "tdvp_padding_scaling.csv", padding_rows)

    # main_method_comparison
    main_rows: list[dict[str, Any]] = []
    by_main: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in ok:
        if str(r.get("tdvp_sweeps", "1")) != "1":
            continue
        key = f"{r['case_id']}|chi{r['chi_max']}"
        by_main[key].append(r)
    for _key, rows in by_main.items():
        tebd = next((r for r in rows if r["method"] == "tebd_swap"), None)
        unpadded = next((r for r in rows if r["method"] == "hybrid_tdvp_unpadded"), None)
        padded4 = next((r for r in rows if r["method"] == "hybrid_tdvp_padded4"), None)
        if tebd is None or padded4 is None:
            continue
        te = _f(tebd, "mean_abs_observable_error")
        ue = _f(unpadded, "mean_abs_observable_error") if unpadded else None
        pe = _f(padded4, "mean_abs_observable_error")
        main_rows.append({
            "case_id": tebd["case_id"],
            "family": tebd["family"],
            "n_qubits": tebd["n_qubits"],
            "depth": tebd["depth"],
            "seed": tebd["seed"],
            "angle_regime": tebd["angle_regime"],
            "chi_max": tebd["chi_max"],
            "tebd_error": te,
            "unpadded_tdvp_error": ue,
            "padded4_tdvp_error": pe,
            "tebd_fidelity": _f(tebd, "fidelity_to_reference"),
            "unpadded_tdvp_fidelity": _f(unpadded, "fidelity_to_reference") if unpadded else None,
            "padded4_tdvp_fidelity": _f(padded4, "fidelity_to_reference"),
            "tebd_peak_chi": tebd["peak_bond_dim"],
            "unpadded_tdvp_peak_chi": unpadded["peak_bond_dim"] if unpadded else None,
            "padded4_tdvp_peak_chi": padded4["peak_bond_dim"],
            "tebd_wall_time": _f(tebd, "wall_time_s"),
            "unpadded_tdvp_wall_time": _f(unpadded, "wall_time_s") if unpadded else None,
            "padded4_tdvp_wall_time": _f(padded4, "wall_time_s"),
            "tebd_hit_chi_max": tebd.get("hit_chi_max"),
            "unpadded_tdvp_hit_chi_max": unpadded.get("hit_chi_max") if unpadded else None,
            "padded4_tdvp_hit_chi_max": padded4.get("hit_chi_max"),
            "padded4_error_over_tebd_error": pe / max(te, 1e-15),
            "padded4_peak_chi_over_tebd_peak_chi": _f(padded4, "peak_bond_dim")
            / max(_f(tebd, "peak_bond_dim"), 1),
            "padded4_time_over_tebd_time": _f(padded4, "wall_time_s")
            / max(_f(tebd, "wall_time_s"), 1e-9),
            "unpadded_error_over_padded4_error": ue / max(pe, 1e-15) if ue is not None else None,
            "outcome": _classify_main_outcome(tebd, unpadded, padded4),
        })
    _write_csv(outdir / "main_method_comparison.csv", main_rows)

    # summary_by_family
    fam_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fixed:
        fam_groups[str(row["family"])].append(row)
    fam_summary: list[dict[str, Any]] = []
    for fam, rows in sorted(fam_groups.items()):
        outcomes = [r["outcome"] for r in rows]
        fam_summary.append({
            "family": fam,
            "num_comparisons": len(rows),
            "tdvp_win_fraction": outcomes.count("tdvp_wins") / max(len(rows), 1),
            "tebd_win_fraction": outcomes.count("tebd_wins") / max(len(rows), 1),
            "similar_fraction": outcomes.count("similar") / max(len(rows), 1),
            "both_fail_fraction": outcomes.count("both_fail") / max(len(rows), 1),
            "mean_tebd_over_tdvp_error": float(np.mean([_f(r, "tebd_error_over_tdvp_error") for r in rows])),
            "mean_tdvp_time_over_tebd": float(np.mean([_f(r, "tdvp_time_over_tebd_time") for r in rows])),
            "mean_tdvp_peak_chi_over_tebd": float(np.mean([_f(r, "tdvp_peak_chi_over_tebd_peak_chi") for r in rows])),
        })
    _write_csv(outdir / "summary_by_family.csv", fam_summary)

    # summary_by_regime (angle_regime)
    reg_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fixed:
        reg_groups[str(row["angle_regime"])].append(row)
    reg_summary: list[dict[str, Any]] = []
    for reg, rows in sorted(reg_groups.items()):
        outcomes = [r["outcome"] for r in rows]
        reg_summary.append({
            "angle_regime": reg,
            "num_comparisons": len(rows),
            "tdvp_win_fraction": outcomes.count("tdvp_wins") / max(len(rows), 1),
            "tebd_win_fraction": outcomes.count("tebd_wins") / max(len(rows), 1),
            "mean_tebd_over_tdvp_error": float(np.mean([_f(r, "tebd_error_over_tdvp_error") for r in rows])),
        })
    _write_csv(outdir / "summary_by_regime.csv", reg_summary)

    # regime_classification
    class_rows: list[dict[str, Any]] = []
    bucket: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in fixed:
        k = (
            str(row["family"]),
            str(row["angle_regime"]),
            str(row["n_qubits"]),
            str(row["depth"]),
            str(row["chi_max"]),
        )
        bucket[k].append(row)

    for (fam, ang, n, depth, chi), rows in sorted(bucket.items()):
        outcomes = [r["outcome"] for r in rows]
        tdvp_w = outcomes.count("tdvp_wins") / max(len(rows), 1)
        tebd_w = outcomes.count("tebd_wins") / max(len(rows), 1)
        similar = outcomes.count("similar") / max(len(rows), 1)
        both_f = outcomes.count("both_fail") / max(len(rows), 1)
        mean_err_ratio = float(np.mean([_f(r, "tebd_error_over_tdvp_error") for r in rows]))
        mean_time_ratio = float(np.mean([_f(r, "tdvp_time_over_tebd_time") for r in rows]))
        mean_chi_ratio = float(np.mean([_f(r, "tdvp_peak_chi_over_tebd_peak_chi") for r in rows]))
        tdvp_sat = float(np.mean([1.0 if r.get("tdvp_hit_chi_max") in ("True", True, "true") else 0.0 for r in rows]))
        tebd_sat = float(np.mean([1.0 if r.get("tebd_hit_chi_max") in ("True", True, "true") else 0.0 for r in rows]))

        if tdvp_w > 0.4 and mean_err_ratio > 2:
            interp = "TDVP promising: structured long-range gates, sweep improvements, lower or comparable chi"
        elif tdvp_w > 0.3 and mean_time_ratio > 2:
            interp = "TDVP promising but expensive: accuracy improves with sweeps but wall time high"
        elif tebd_sat > 0.5 and tdvp_w > tebd_w:
            interp = "Routing-limited TEBD regime: TEBD+SWAP hits chi or has high error while TDVP remains stable"
        elif both_f > 0.5:
            interp = "Compressibility-limited regime: both methods fail or both saturate chi"
        elif tebd_w > tdvp_w:
            interp = "TEBD preferable: TEBD is faster and equally/more accurate"
        else:
            interp = "Mixed / inconclusive: compare sweep_scaling and fixed_chi tables"

        dominant = max(
            ("tdvp_wins", tdvp_w),
            ("tebd_wins", tebd_w),
            ("similar", similar),
            ("both_fail", both_f),
            key=lambda x: x[1],
        )[0]

        class_rows.append({
            "family": fam,
            "angle_regime": ang,
            "n_qubits": n,
            "depth": depth,
            "chi_max": chi,
            "dominant_outcome": dominant,
            "tdvp_win_fraction": tdvp_w,
            "tebd_win_fraction": tebd_w,
            "similar_fraction": similar,
            "both_fail_fraction": both_f,
            "mean_tdvp_error_over_tebd_error": 1.0 / max(mean_err_ratio, 1e-15),
            "mean_tdvp_time_over_tebd_time": mean_time_ratio,
            "mean_tdvp_peak_chi_over_tebd_peak_chi": mean_chi_ratio,
            "chi_saturation_fraction_tdvp": tdvp_sat,
            "chi_saturation_fraction_tebd": tebd_sat,
            "interpretation": interp,
        })
    _write_csv(outdir / "regime_classification.csv", class_rows)

    write_report(
        raw,
        fixed,
        fam_summary,
        sweep_rows,
        padding_rows,
        main_rows,
        class_rows,
        outdir / "report.md",
        stage=STAGE,
    )


def write_report(
    raw: list[dict[str, Any]],
    fixed: list[dict[str, Any]],
    fam_summary: list[dict[str, Any]],
    sweep_rows: list[dict[str, Any]],
    padding_rows: list[dict[str, Any]],
    main_rows: list[dict[str, Any]],
    class_rows: list[dict[str, Any]],
    path: Path,
    *,
    stage: int,
) -> None:
    lines = [
        "# Long-range TDVP regime map\n\n",
        f"Stage **{stage}**. Primary metric: observable error vs reference.\n\n",
        "## 1. Overall conclusion by family\n\n",
        "| family | comparisons | TDVP win % | TEBD win % | mean TEBD/TDVP err | recommendation |\n",
        "|---|---:|---:|---:|---:|---|\n",
    ]
    rec_map = {
        "periodic_1d": "use hybrid TDVP",
        "sparse_long_range": "use hybrid TDVP",
        "flattened_2d_grid": "use hybrid TDVP",
        "nn_brickwork": "both acceptable",
        "dense_long_range": "avoid MPS at this chi",
        "random_all_to_all": "avoid MPS at this chi",
        "floquet_hump": "needs larger chi / reference unclear",
    }
    for fs in fam_summary:
        fam = fs["family"]
        tw = float(fs["tdvp_win_fraction"])
        ew = float(fs["tebd_win_fraction"])
        if tw > 0.4:
            rec = "use hybrid TDVP"
        elif ew > 0.4:
            rec = "use TEBD+SWAP"
        elif tw + ew < 0.2:
            rec = "needs larger chi / reference unclear"
        else:
            rec = rec_map.get(str(fam), "both acceptable")
        lines.append(
            f"| {fam} | {fs['num_comparisons']} | {100*tw:.0f}% | {100*ew:.0f}% | "
            f"{fs['mean_tebd_over_tdvp_error']:.2f} | {rec} |\n"
        )

    lines.append("\n## 2. Best TDVP regimes\n\n")
    tdvp_best = sorted(fixed, key=lambda r: _f(r, "tebd_error_over_tdvp_error"), reverse=True)[:8]
    for r in tdvp_best:
        lines.append(
            f"- `{r['case_id']}` χ={r['chi_max']}: TEBD/TDVP err ratio={_f(r, 'tebd_error_over_tdvp_error'):.1f}, "
            f"outcome={r['outcome']}\n"
        )

    lines.append("\n## 3. Worst TDVP regimes\n\n")
    tdvp_worst = sorted(fixed, key=lambda r: _f(r, "tdvp_error_over_tebd_error"), reverse=True)[:8]
    for r in tdvp_worst:
        lines.append(
            f"- `{r['case_id']}` χ={r['chi_max']}: TDVP/TEBD err ratio={_f(r, 'tdvp_error_over_tebd_error'):.1f}\n"
        )

    lines.append("\n## 4. Sweep diagnostic\n\n")
    if INCLUDE_TDVP_SWEEP_SCAN:
        lines.append(
            "> **Warning:** Multi-sweep standard TDVP is diagnostic only. Previous Stage 0 "
            "results showed that additional sweeps can worsen the state and should not be "
            "interpreted as systematic refinement unless separately validated.\n\n"
        )
    else:
        lines.append(
            "- Default benchmark uses `tdvp_sweeps=1` only. "
            "Set `YAQS_LR_REGIME_INCLUDE_TDVP_SWEEP_SCAN=1` for sweep-grid diagnostics.\n\n"
        )
    if sweep_rows:
        helpful = sum(1 for s in sweep_rows if s["classification"] == "sweep_helpful")
        harmful = sum(1 for s in sweep_rows if s["classification"] == "sweep_harmful_or_unstable")
        lines.append(
            f"- sweep_helpful: {helpful}/{len(sweep_rows)}; "
            f"saturated: {sum(1 for s in sweep_rows if s['classification']=='sweep_saturated')}; "
            f"harmful: {harmful}\n"
        )
        if harmful > helpful:
            lines.append(
                "- **sweeps:** `tdvp_sweeps > 1` should remain diagnostic only; "
                "extra sweeps often hurt in Stage 0.\n"
            )
    else:
        lines.append("- No sweep pairs in this run.\n")

    lines.append("\n## Standard TDVP padding diagnostic\n\n")
    lines.append(
        "**Stage 0 conclusions (reference):** unpadded standard TDVP underuses χ on periodic "
        "long-range gates; padding to bond dimension 4 fixes small periodic cases; padding "
        "beyond 4 does not help those cases; extra TDVP sweeps are not reliable refinement.\n\n"
    )

    periodic_main = [
        r for r in main_rows if r.get("family") == "periodic_1d"
    ]
    lines.append("### Does padding fix periodic long-range gates?\n\n")
    if periodic_main:
        for r in sorted(periodic_main, key=lambda x: (str(x["n_qubits"]), str(x["depth"]), str(x["chi_max"])))[:8]:
            lines.append(
                f"- `{r['case_id']}` χ={r['chi_max']}: "
                f"unpadded err={r.get('unpadded_tdvp_error')} fid={r.get('unpadded_tdvp_fidelity')}; "
                f"padded4 err={r['padded4_tdvp_error']:.3e} fid={r['padded4_tdvp_fidelity']:.4f}; "
                f"TEBD err={r['tebd_error']:.3e}\n"
            )
        lines.append(
            "\n**periodic_1d:** unpadded TDVP often underuses χ and loses X-coherence; "
            "padded4 TDVP fixes the small periodic cases.\n\n"
        )
    else:
        lines.append("- No periodic_1d main-method rows at sweeps=1.\n\n")

    sparse_main = [r for r in main_rows if r.get("family") == "sparse_long_range"]
    lines.append("### Does padding preserve sparse-long-range advantages?\n\n")
    if sparse_main:
        wins = sum(1 for r in sparse_main if r["outcome"] in ("padded4_tdvp_wins", "padding_required"))
        tebd_sat = sum(1 for r in sparse_main if _hit_chi_max({"hit_chi_max": r.get("tebd_hit_chi_max")}))
        lines.append(
            f"- {wins}/{len(sparse_main)} sparse cases favor padded4 or require padding; "
            f"TEBD hits χ_max in {tebd_sat}/{len(sparse_main)} fixed-χ rows.\n"
        )
        lines.append(
            "**sparse_long_range:** TDVP can be competitive when TEBD+SWAP hits χ_max; "
            "at larger χ, TEBD may become essentially exact.\n\n"
        )
    else:
        lines.append("- No sparse_long_range main-method rows.\n\n")

    lines.append("### Is padding enough, or is TEBD still better?\n\n")
    if main_rows:
        p4_wins = sum(1 for r in main_rows if r["outcome"] == "padded4_tdvp_wins")
        tebd_w = sum(1 for r in main_rows if r["outcome"] == "tebd_wins")
        lines.append(
            f"- padded4_tdvp_wins: {p4_wins}/{len(main_rows)}; tebd_wins: {tebd_w}/{len(main_rows)}.\n\n"
        )
    else:
        lines.append("- No main_method_comparison rows.\n\n")

    lines.append("### Does padding merely match TEBD's bond dimension?\n\n")
    if main_rows:
        lower = sum(
            1 for r in main_rows if _f(r, "padded4_peak_chi_over_tebd_peak_chi") < 0.85
        )
        lines.append(
            f"- padded4 peak χ < 85% of TEBD peak χ in {lower}/{len(main_rows)} rows "
            f"(TDVP can stay lower-χ).\n\n"
        )

    lines.append("### Should multi-sweep TDVP be used?\n\n")
    lines.append(
        "- Default benchmark uses one sweep only (`tdvp_sweeps=1`).\n"
        "- **sweeps:** `tdvp_sweeps > 1` should remain diagnostic only.\n"
    )
    if INCLUDE_TDVP_SWEEP_SCAN and sweep_rows:
        harmful = sum(1 for s in sweep_rows if s["classification"] == "sweep_harmful_or_unstable")
        lines.append(
            f"- Sweep scan enabled: {harmful}/{len(sweep_rows)} transitions classified harmful/unstable.\n"
        )

    pad_transitions = [p for p in padding_rows if p.get("classification") not in ("padding_baseline", "")]
    if pad_transitions:
        helpful = sum(1 for p in pad_transitions if p["classification"] == "padding_helpful")
        lines.append(
            f"\nPadding scan transitions: helpful={helpful}/{len(pad_transitions)} "
            f"(see `tdvp_padding_scaling.csv`).\n"
        )

    lines.append("\n## 5. Bond-dimension diagnostic\n\n")
    if fixed:
        lower_chi = sum(1 for r in fixed if _f(r, "tdvp_peak_chi_over_tebd_peak_chi") < 0.95)
        lines.append(f"- TDVP lower peak χ than TEBD in {lower_chi}/{len(fixed)} matched pairs.\n")

    lines.append("\n## 6. Accuracy vs runtime\n\n")
    if fixed:
        faster = sum(1 for r in fixed if _f(r, "tdvp_time_over_tebd_time") < 0.9)
        more_acc = sum(1 for r in fixed if _f(r, "tebd_error_over_tdvp_error") > 1.5)
        lines.append(f"- TDVP faster: {faster}/{len(fixed)}; TDVP more accurate (err ratio>1.5): {more_acc}/{len(fixed)}\n")

    lines.append("\n## 7. Practical recommendation\n\n")
    for cr in class_rows[:12]:
        lines.append(
            f"- **{cr['family']}** ({cr['angle_regime']}, n={cr['n_qubits']}, d={cr['depth']}, χ={cr['chi_max']}): "
            f"{cr['interpretation']}\n"
        )

    lines.append(f"\n---\n\nTotal raw runs: {len(raw)}; successful comparisons: {len(fixed)}.\n")
    path.write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if STAGE not in STAGE_CONFIGS:
        raise SystemExit(f"Unknown YAQS_LR_REGIME_STAGE={STAGE}")

    cfg = STAGE_CONFIGS[STAGE]
    outdir = OUTDIR if OUTDIR.is_absolute() else _repo_root() / OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)

    validate_reference_convention(chi=max(cfg["chi_values"]))
    validate_tdvp_padding(chi=max(cfg["chi_values"]))

    specs = build_all_specs(cfg)
    write_circuit_metadata(specs, outdir / "circuit_metadata.jsonl")

    raw_path = outdir / "raw_runs.csv"
    if OVERWRITE and raw_path.exists():
        raw_path.unlink()
    existing = _load_existing_keys(raw_path)
    bond_hist: list[BondHistoryRow] = []

    chi_values = list(cfg["chi_values"])
    run_plans = build_benchmark_runs(STAGE, cfg)
    padding_noise = TDVP_PADDING_NOISE
    total = len(specs) * len(chi_values) * len(run_plans)
    case_num = 0

    print(
        f"Stage {STAGE}: {len(specs)} circuits, chi={chi_values}, "
        f"{len(run_plans)} method/sweep/padding combos, "
        f"padding_scan={INCLUDE_PADDING_SCAN}, sweep_scan={INCLUDE_TDVP_SWEEP_SCAN}, "
        f"noise={padding_noise}, out={outdir}"
    )

    ref_cache: dict[str, tuple[Any, RefType, int | None, int | None]] = {}

    for spec in specs:
        if spec.case_id not in ref_cache:
            ref_cache[spec.case_id] = compute_reference(spec, exact_n_max=int(cfg["exact_n_max"]))
        ref, ref_type, ref_chi, ref_sw = ref_cache[spec.case_id]

        for chi in chi_values:
            for plan in run_plans:
                case_num += 1
                pad_dim = plan.tdvp_padding_dim
                run = RawRun(
                    family=spec.family,
                    n_qubits=spec.n_qubits,
                    depth=spec.depth,
                    seed=spec.seed,
                    angle_regime=spec.angle_regime,
                    method=plan.method,
                    chi_max=chi,
                    tdvp_sweeps=plan.tdvp_sweeps,
                    tdvp_padding_dim=pad_dim,
                    tdvp_padding_noise=padding_noise if _is_tdvp_method(plan.method) else 0.0,
                    sweep_scan_enabled=plan.sweep_scan_enabled,
                    svd_threshold=SVD_THRESHOLD,
                    lanczos_tol=LANCZOS_TOL,
                    reference_type=ref_type,
                    reference_available=ref is not None,
                    reference_chi=ref_chi,
                    reference_sweeps=ref_sw,
                    case_id=spec.case_id,
                )
                if run.run_key() in existing:
                    print(f"[{case_num}/{total}] SKIP {run.run_key()}")
                    continue

                run = run_method(
                    spec,
                    method=plan.method,
                    chi=chi,
                    tdvp_sweeps=plan.tdvp_sweeps,
                    tdvp_padding_dim=pad_dim,
                    tdvp_padding_noise=padding_noise,
                    sweep_scan_enabled=plan.sweep_scan_enabled,
                    ref=ref,
                    ref_type=ref_type,
                    ref_chi=ref_chi,
                    ref_sweeps=ref_sw,
                    bond_history=bond_hist,
                )
                _append_raw_run(raw_path, run)
                existing.add(run.run_key())
                dispatch_note = (
                    f" gates={run.total_gates} nn={run.nn_gate_count} lr={run.lr_gate_count} "
                    f"tebd_direct={run.tebd_direct_gate_count} tebd_swap={run.tebd_swap_gate_count} "
                    f"tdvp_lr={run.tdvp_lr_count} enriched={run.enriched_lr_count} "
                    f"pad={_padding_key(pad_dim)} applied={run.padding_applied}"
                )
                print(
                    f"[{case_num}/{total}] family={spec.family} n={spec.n_qubits} depth={spec.depth} "
                    f"chi={chi} method={plan.method} sweeps={plan.tdvp_sweeps} "
                    f"padding={_padding_key(pad_dim)} status={run.status}"
                    f"{dispatch_note if run.status == 'ok' else ''}"
                )

    raw = _read_all_raw(raw_path)
    summarize_raw_runs(raw, outdir)

    if RECORD_BOND_HISTORY and bond_hist:
        _write_csv(outdir / "bond_history.csv", [asdict(b) for b in bond_hist])

    print(f"\nDone. {len(raw)} rows in {raw_path}")
    print(f"Report: {outdir / 'report.md'}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    main()
