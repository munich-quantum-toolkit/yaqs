#!/usr/bin/env python
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Benchmark suite for the adaptive generator-enriched TDVP hybrid method.

Run:

    uv run python -m scripts.benchmark_adaptive_generator_tdvp

Outputs:
    results/adaptive_generator_tdvp_benchmark.csv
    results/adaptive_generator_tdvp_benchmark.md
"""

from __future__ import annotations

import csv
import json
import os
import time
import copy
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

MethodName = Literal["adaptive_hybrid", "tebd_swaps"]


@dataclass(frozen=True)
class BenchmarkRow:
    family: str
    n_qubits: int
    depth_or_layers: int
    circuit_name: str
    method: str
    reference_method: str
    fidelity_error: float | None
    pauli_obs_max_error: float | None
    pauli_obs_mean_error: float | None
    pauli_obs_l2_error: float | None
    worst_observable: str | None
    max_bond_dim: int | None
    mean_bond_dim: float | None
    sum_chi_cubed: int | None
    wall_time_s: float
    tdvp_lr_pauli_count: int | None
    enriched_lr_pauli_count: int | None
    tebd_nn_count: int
    num_long_range_gates: int
    num_long_range_pauli_gates: int
    route_summary: str


def _ensure_results_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _qiskit_vec(qc: QuantumCircuit) -> np.ndarray:
    return np.asarray(Statevector(qc).data, dtype=np.complex128)


def _fid_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(max(0.0, 1.0 - abs(np.vdot(a, b)) ** 2))


def _count_circuit_gates(qc: QuantumCircuit) -> tuple[int, int, int]:
    """Return (nn_two_qubit_count, lr_two_qubit_count, lr_pauli_count)."""
    nn = 0
    lr = 0
    lr_pauli = 0
    for ci in qc.data:
        inst = ci.operation
        qargs = ci.qubits
        if len(qargs) != 2:
            continue
        i = qc.find_bit(qargs[0]).index
        j = qc.find_bit(qargs[1]).index
        if abs(i - j) == 1:
            nn += 1
        else:
            lr += 1
            if inst.name in {"rxx", "ryy", "rzz"}:
                lr_pauli += 1
    return nn, lr, lr_pauli


def _run_method(
    qc: QuantumCircuit,
    *,
    method: MethodName,
    svd_threshold: float,
    max_bond_dim: int | None,
    krylov_tol: float = 1e-12,
    tdvp_projection_defect_tol: float = 1e-3,
) -> tuple[np.ndarray, MPS, float]:
    params = StrongSimParams(
        preset="exact",
        get_state=True,
        svd_threshold=svd_threshold,
        max_bond_dim=max_bond_dim,
        krylov_tol=krylov_tol,
        gate_mode="hybrid" if method == "adaptive_hybrid" else "tebd",
        tangent_blindness_tol=1e-12,
        tdvp_projection_defect_tol=tdvp_projection_defect_tol,
        tdvp_visibility_safety_tol=None,
        tdvp_pauli_consistency_check=False,
        tdvp_pauli_consistency_tol=1e-10,
    )
    sim = Simulator()
    init = State(qc.num_qubits, initial="zeros", representation="mps")
    t0 = time.perf_counter()
    result = sim.run(init, qc, params)
    dt = time.perf_counter() - t0
    assert result.output_state is not None
    mps = result.output_state.mps
    vec = np.asarray(mps.to_vec(), dtype=np.complex128) if qc.num_qubits <= 14 else np.array([], dtype=np.complex128)
    return vec, mps, float(dt)


def _mean_bond_dim(mps: MPS) -> float:
    bonds = [int(t.shape[1]) for t in mps.tensors[1:]]
    if not bonds:
        return 1.0
    return float(sum(bonds) / len(bonds))


def _observable_set(qc: QuantumCircuit, *, max_all_sites_n: int = 20) -> list[tuple[str, Observable, int | list[int]]]:
    n = qc.num_qubits
    # For large n, sample to keep runtime bounded.
    if n <= max_all_sites_n:
        sites_1q = list(range(n))
    else:
        core = {0, n - 1, n // 2, max(0, n // 2 - 1)}
        step = 2
        sampled = set(range(0, n, step))
        sites_1q = sorted(core | sampled)

    obs: list[tuple[str, Observable, int | list[int]]] = []
    for i in sites_1q:
        obs.append((f"X({i})", Observable("x", i), i))
        obs.append((f"Y({i})", Observable("y", i), i))
        obs.append((f"Z({i})", Observable("z", i), i))

    # Nearest-neighbor correlators near center.
    for i in sorted({max(0, n // 2 - 2), max(0, n // 2 - 1), n // 2}):
        if i + 1 >= n:
            continue
        obs.append((f"XX({i},{i+1})", Observable("xx", [i, i + 1]), [i, i + 1]))
        obs.append((f"YY({i},{i+1})", Observable("yy", [i, i + 1]), [i, i + 1]))
        obs.append((f"ZZ({i},{i+1})", Observable("zz", [i, i + 1]), [i, i + 1]))
    return obs


def _statevector_expect_pauli(state: np.ndarray, n: int, which: str, sites: list[int]) -> complex:
    """Expectation of a Pauli string (uses Qiskit Pauli ordering)."""
    if which not in {"X", "Y", "Z", "XX", "YY", "ZZ"}:
        raise ValueError(which)
    label = ["I"] * n
    if which in {"X", "Y", "Z"}:
        label[n - 1 - sites[0]] = which
    else:
        p = which[0]
        label[n - 1 - sites[0]] = p
        label[n - 1 - sites[1]] = p
    return complex(Statevector(state).expectation_value(Pauli("".join(label))))


def _observable_errors(
    qc: QuantumCircuit,
    *,
    mps: MPS,
    reference_vec: np.ndarray | None,
    max_all_sites_n: int = 20,
) -> tuple[float | None, float | None, float | None, str | None]:
    obs = _observable_set(qc, max_all_sites_n=max_all_sites_n)
    if not obs:
        return None, None, None, None

    errs: list[float] = []
    worst_name: str | None = None
    worst_val = -1.0

    def mps_overlap(left: MPS, right: MPS) -> complex:
        env = np.ones((1, 1), dtype=np.complex128)
        for n_site in range(left.length):
            a = np.asarray(left.tensors[n_site], dtype=np.complex128)
            b = np.asarray(right.tensors[n_site], dtype=np.complex128)
            env = np.einsum("pab,ac,pcd->bd", np.conjugate(a), env, b, optimize=True)
        return complex(env.reshape(()))

    n = qc.num_qubits
    for name, ob, sites in obs:
        ket_op = copy.deepcopy(mps)
        ket_op.apply_local(ob)
        got = complex(mps_overlap(mps, ket_op))
        got_r = float(np.real(got))
        if reference_vec is None:
            continue

        # Reference from statevector for these Pauli observables.
        if name.startswith(("X(", "Y(", "Z(")):
            which = name[0]
            ref = _statevector_expect_pauli(reference_vec, n, which, [int(name.split("(")[1].split(")")[0])])
        else:
            which = name[:2]
            ij = name.split("(")[1].split(")")[0]
            i_s, j_s = ij.split(",")
            ref = _statevector_expect_pauli(reference_vec, n, which, [int(i_s), int(j_s)])
        ref_r = float(np.real(ref))
        err = abs(got_r - ref_r)
        errs.append(err)
        if err > worst_val:
            worst_val = err
            worst_name = f"{name}: got={got_r:.6g}, ref={ref_r:.6g}, err={err:.3g}"

    if reference_vec is None or not errs:
        return None, None, None, None
    max_err = float(max(errs))
    mean_err = float(sum(errs) / len(errs))
    l2_err = float(np.sqrt(float(np.sum(np.square(np.asarray(errs, dtype=np.float64))))))
    return max_err, mean_err, l2_err, worst_name


def _route_summary(mps: MPS) -> tuple[int | None, int | None, str]:
    stats = getattr(mps, "route_stats", None)
    if not isinstance(stats, dict):
        return None, None, ""
    tdvp_c = int(stats.get("tdvp_lr_pauli", 0))
    enr_c = int(stats.get("enriched_lr_pauli", 0))
    ratios = stats.get("ratios", [])
    try:
        summary = json.dumps(ratios, separators=(",", ":"), sort_keys=True)
    except TypeError:
        summary = json.dumps([str(x) for x in ratios], separators=(",", ":"), sort_keys=True)
    return tdvp_c, enr_c, summary


def _bench_one(
    rows: list[BenchmarkRow],
    *,
    family: str,
    circuit_name: str,
    qc: QuantumCircuit,
    depth_or_layers: int,
    compare_tebd_swaps: bool,
    svd_threshold_exact: float = 1e-14,
    svd_threshold_budget: float = 1e-9,
    budgets: list[int | None] | None = None,
) -> None:
    if budgets is None:
        budgets = [None]

    nn2q, lr2q, lr_pauli = _count_circuit_gates(qc)
    reference_vec = _qiskit_vec(qc) if qc.num_qubits <= 14 else None

    # Always run adaptive hybrid.
    for max_bond in budgets:
        vec, mps, wall = _run_method(
            qc,
            method="adaptive_hybrid",
            svd_threshold=svd_threshold_exact,
            max_bond_dim=max_bond,
            tdvp_projection_defect_tol=float(os.environ.get("YAQS_TDVP_DEFECT_TOL", "1e-3")),
        )
        fid = None if reference_vec is None else _fid_err(reference_vec, vec)
        max_err, mean_err, l2_err, worst = _observable_errors(qc, mps=mps, reference_vec=reference_vec)
        tdvp_c, enr_c, route_json = _route_summary(mps)
        rows.append(
            BenchmarkRow(
                family=family,
                n_qubits=qc.num_qubits,
                depth_or_layers=depth_or_layers,
                circuit_name=circuit_name,
                method=f"adaptive_hybrid(max_bond={max_bond},svd={svd_threshold_exact})",
                reference_method="qiskit_statevector" if reference_vec is not None else "none",
                fidelity_error=fid,
                pauli_obs_max_error=max_err,
                pauli_obs_mean_error=mean_err,
                pauli_obs_l2_error=l2_err,
                worst_observable=worst,
                max_bond_dim=int(mps.get_max_bond()),
                mean_bond_dim=_mean_bond_dim(mps),
                sum_chi_cubed=int(mps.get_cost()),
                wall_time_s=wall,
                tdvp_lr_pauli_count=tdvp_c,
                enriched_lr_pauli_count=enr_c,
                tebd_nn_count=nn2q,
                num_long_range_gates=lr2q,
                num_long_range_pauli_gates=lr_pauli,
                route_summary=route_json,
            )
        )

    if not compare_tebd_swaps:
        return

    # Compare TEBD+SWAPs under "exact-ish" and budget settings.
    for max_bond, svd_thr in [(None, svd_threshold_exact), (64, svd_threshold_budget), (128, svd_threshold_budget)]:
        vec, mps, wall = _run_method(qc, method="tebd_swaps", svd_threshold=svd_thr, max_bond_dim=max_bond)
        fid = None if reference_vec is None else _fid_err(reference_vec, vec)
        max_err, mean_err, l2_err, worst = _observable_errors(qc, mps=mps, reference_vec=reference_vec)
        tdvp_c, enr_c, route_json = _route_summary(mps)
        rows.append(
            BenchmarkRow(
                family=family,
                n_qubits=qc.num_qubits,
                depth_or_layers=depth_or_layers,
                circuit_name=circuit_name,
                method=f"tebd_swaps(max_bond={max_bond},svd={svd_thr})",
                reference_method="qiskit_statevector" if reference_vec is not None else "none",
                fidelity_error=fid,
                pauli_obs_max_error=max_err,
                pauli_obs_mean_error=mean_err,
                pauli_obs_l2_error=l2_err,
                worst_observable=worst,
                max_bond_dim=int(mps.get_max_bond()),
                mean_bond_dim=_mean_bond_dim(mps),
                sum_chi_cubed=int(mps.get_cost()),
                wall_time_s=wall,
                tdvp_lr_pauli_count=tdvp_c,
                enriched_lr_pauli_count=enr_c,
                tebd_nn_count=nn2q,
                num_long_range_gates=lr2q,
                num_long_range_pauli_gates=lr_pauli,
                route_summary=route_json,
            )
        )


def _circuits_correctness() -> list[tuple[str, str, int, QuantumCircuit]]:
    out: list[tuple[str, str, int, QuantumCircuit]] = []

    qc = QuantumCircuit(8)
    qc.ryy(0.25, 1, 6)
    out.append(("correctness", "tangent_blind_ryy_8q", 1, qc))

    qc = QuantumCircuit(8)
    qc.rx(np.pi / 2, 6)
    qc.ryy(0.25, 1, 6)
    out.append(("correctness", "endpoint_prepared_ryy_8q", 2, qc))

    qc = QuantumCircuit(8)
    qc.rzz(0.25, 1, 6)
    out.append(("correctness", "rzz_8q", 1, qc))

    qc = QuantumCircuit(10)
    qc.rxx(0.21, 2, 9)
    qc.ryy(0.25, 3, 8)
    out.append(("correctness", "mixed_stack_10q", 2, qc))

    qc = QuantumCircuit(12)
    qc.ry(np.pi / 4, 1)
    qc.ry(np.pi / 4, 4)
    qc.ry(np.pi / 4, 7)
    qc.rzz(0.19, 1, 10)
    qc.rzz(0.27, 4, 11)
    qc.rzz(0.33, 0, 7)
    qc.rxx(0.21, 2, 9)
    qc.ryy(0.25, 3, 8)
    out.append(("correctness", "hard_mixed_stack_12q", 8, qc))

    return out


def _circuits_position_distance() -> list[tuple[str, str, int, QuantumCircuit]]:
    out: list[tuple[str, str, int, QuantumCircuit]] = []
    theta = 0.25
    for n in (10, 14):
        pairs = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (0, n - 1), (1, n - 2), (2, n - 3)]
        for gate in ("rxx", "ryy", "rzz"):
            for i, j in pairs:
                qc = QuantumCircuit(n)
                if gate == "rzz":
                    qc.ry(np.pi / 4, i)
                    qc.ry(np.pi / 5, j)
                getattr(qc, gate)(theta, i, j)
                out.append(("posdist", f"{gate}_n{n}_{i}_{j}", 1, qc))
    return out


def _circuits_repeated_layers() -> list[tuple[str, str, int, QuantumCircuit]]:
    out: list[tuple[str, str, int, QuantumCircuit]] = []
    rng = np.random.default_rng(0)
    for n in (12, 16, 20):
        edges_disjoint = [(0, n - 1), (2, n - 3), (4, n - 5)]
        edges_overlap = [(1, n - 2), (2, n - 3), (1, n - 3)]
        for L in (1, 2, 4, 8):
            # all rzz
            qc = QuantumCircuit(n)
            for _layer in range(L):
                for i, j in edges_disjoint:
                    qc.ry(np.pi / 6, i)
                    qc.ry(np.pi / 7, j)
                    qc.rzz(0.17, i, j)
            out.append(("layers", f"all_rzz_disjoint_n{n}_L{L}", L, qc))

            # mixed disjoint
            qc = QuantumCircuit(n)
            axes = ["rxx", "ryy", "rzz"]
            for _layer in range(L):
                for (i, j), ax in zip(edges_disjoint, axes, strict=False):
                    if ax == "rzz":
                        qc.ry(np.pi / 6, i)
                        qc.ry(np.pi / 7, j)
                    getattr(qc, ax)(0.19, i, j)
            out.append(("layers", f"mixed_disjoint_n{n}_L{L}", L, qc))

            # overlapping noncommuting
            qc = QuantumCircuit(n)
            for _layer in range(L):
                i, j = edges_overlap[rng.integers(0, len(edges_overlap))]
                ax = axes[rng.integers(0, len(axes))]
                if ax == "rzz":
                    qc.ry(np.pi / 6, i)
                    qc.ry(np.pi / 7, j)
                getattr(qc, ax)(float(rng.choice([0.11, 0.17, 0.23])), i, j)
            out.append(("layers", f"overlap_random_axes_n{n}_L{L}", L, qc))
    return out


def _render_markdown(rows: list[BenchmarkRow]) -> str:
    def fmt(x: Any) -> str:
        if x is None:
            return "—"
        if isinstance(x, float):
            return f"{x:.3e}"
        return str(x)

    lines: list[str] = []
    lines.append("# Adaptive Generator-Enriched TDVP Benchmark\n")
    lines.append("## Method summary\n")
    lines.append(
        "- single-qubit gates: direct tensor contraction\n"
        "- nearest-neighbor two-qubit gates: TEBD/SVD\n"
        "- long-range non-Pauli gates: local TDVP\n"
        "- long-range Pauli rotations (rxx/ryy/rzz): adaptive routing\n"
        "  - TDVP if estimated local projection defect <= ε\n"
        "  - otherwise exact generator enrichment\n"
    )

    lines.append("## Correctness diagnostics\n")
    for r in rows:
        if r.family != "correctness":
            continue
        lines.append(f"- **{r.circuit_name}** `{r.method}`: fidelity={fmt(r.fidelity_error)}, "
                     f"pauli_max={fmt(r.pauli_obs_max_error)}, max_bond={fmt(r.max_bond_dim)}, "
                     f"tdvp_lr_pauli={fmt(r.tdvp_lr_pauli_count)}, enriched_lr_pauli={fmt(r.enriched_lr_pauli_count)}")

    lines.append("\n## Route statistics\n")
    total_tdvp = sum((r.tdvp_lr_pauli_count or 0) for r in rows if r.method.startswith("adaptive_hybrid"))
    total_enr = sum((r.enriched_lr_pauli_count or 0) for r in rows if r.method.startswith("adaptive_hybrid"))
    lines.append(f"- Total LR Pauli routed to TDVP: **{total_tdvp}**")
    lines.append(f"- Total LR Pauli routed to enrichment: **{total_enr}**\n")

    lines.append("## Position/distance sweep\n")
    lines.append("See CSV for full per-pair detail (includes per-gate route JSON).\n")

    lines.append("## Mixed-stack stability\n")
    lines.append("The previously failing mixed stacks are included in correctness diagnostics and repeated-layer families.\n")

    lines.append("## TEBD+SWAP comparison\n")
    lines.append("Rows include both `adaptive_hybrid(...)` and `tebd_swaps(...)` for selected circuits.\n")

    lines.append("## Paper-scale circuits\n")
    lines.append("This script includes repeated-layer stress families up to n=20. Add manuscript circuits here as needed.\n")

    lines.append("## Summary\n")
    lines.append(
        "This benchmark reports fidelity (when Qiskit statevector is feasible), Pauli observable errors, "
        "bond-dimension cost proxies, wall time, and per-long-range-Pauli route decisions.\n"
    )
    return "\n".join(lines)


def main() -> None:
    out_dir = _ensure_results_dir()
    circuits: list[tuple[str, str, int, QuantumCircuit]] = []
    circuits += _circuits_correctness()
    circuits += _circuits_position_distance()
    circuits += _circuits_repeated_layers()

    thresholds = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 1e-4, 1e-6]
    summary: list[dict[str, Any]] = []

    for thr in thresholds:
        os.environ["YAQS_TDVP_DEFECT_TOL"] = str(thr)
        rows: list[BenchmarkRow] = []

        t_total0 = time.perf_counter()
        for family, name, depth_or_layers, qc in circuits:
            compare_tebd = (qc.num_qubits <= 14) and (family in {"correctness", "layers"})
            budgets = [None] if family != "layers" else [None, 64, 128]
            _bench_one(
                rows,
                family=family,
                circuit_name=name,
                qc=qc,
                depth_or_layers=depth_or_layers,
                compare_tebd_swaps=compare_tebd,
                budgets=budgets,
            )
        wall_total = float(time.perf_counter() - t_total0)

        # Only judge Qiskit-feasible adaptive rows.
        adaptive = [r for r in rows if r.method.startswith("adaptive_hybrid") and r.reference_method == "qiskit_statevector"]
        passed = [r for r in adaptive if r.fidelity_error is not None and r.fidelity_error < 1e-10]
        failed = [r for r in adaptive if r.fidelity_error is not None and r.fidelity_error >= 1e-10]
        fid_max = float(max((r.fidelity_error or 0.0) for r in adaptive)) if adaptive else 0.0
        pauli_max = float(max((r.pauli_obs_max_error or 0.0) for r in adaptive)) if adaptive else 0.0
        tdvp_total = int(sum((r.tdvp_lr_pauli_count or 0) for r in rows if r.method.startswith("adaptive_hybrid")))
        enr_total = int(sum((r.enriched_lr_pauli_count or 0) for r in rows if r.method.startswith("adaptive_hybrid")))

        summary.append(
            {
                "threshold": thr,
                "num_passed_circuits": len(passed),
                "num_failed_circuits": len(failed),
                "fidelity_error_max": fid_max,
                "pauli_obs_max_error_max": pauli_max,
                "tdvp_lr_pauli_count": tdvp_total,
                "enriched_lr_pauli_count": enr_total,
                "wall_time_s_total": wall_total,
            }
        )

        suffix = str(thr).replace(".", "p").replace("-", "m")
        csv_path = out_dir / f"adaptive_generator_tdvp_benchmark_defect{suffix}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(BenchmarkRow.__annotations__.keys()))
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))

        md_path = out_dir / f"adaptive_generator_tdvp_benchmark_defect{suffix}.md"
        md_path.write_text(_render_markdown(rows), encoding="utf-8")
        print(f"Wrote {csv_path}")
        print(f"Wrote {md_path}")

    # Print summary.
    print("\n=== accept-ratio summary ===")
    for s in summary:
        print(
            "defect_tol",
            s["threshold"],
            "passed",
            s["num_passed_circuits"],
            "failed",
            s["num_failed_circuits"],
            "fid_max",
            f"{s['fidelity_error_max']:.3e}",
            "pauli_max",
            f"{s['pauli_obs_max_error_max']:.3e}",
            "tdvp_lr_pauli",
            s["tdvp_lr_pauli_count"],
            "enriched_lr_pauli",
            s["enriched_lr_pauli_count"],
            "wall_s",
            f"{s['wall_time_s_total']:.2f}",
        )


if __name__ == "__main__":
    # Avoid noisy Windows OpenMP warnings in some environments.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

