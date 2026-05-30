#!/usr/bin/env python
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Stability sweep for tdvp_projection_defect_tol.

Run:

    uv run python -m scripts.check_projection_accept_ratio_stability

Outputs:
    results/projection_defect_tol_stability.csv
    results/projection_defect_tol_stability.md
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector

from mqt.yaqs import Simulator
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.data_structures.state import State


@dataclass(frozen=True)
class CircuitCase:
    family: str
    name: str
    n_qubits: int
    depth_or_layers: int
    qc: QuantumCircuit


@dataclass(frozen=True)
class ThresholdSummary:
    defect_tol: float
    num_circuits: int
    strict_passed: int
    strict_failed: int
    practical_passed: int
    practical_failed: int
    max_fidelity_error: float | None
    max_pauli_obs_error: float
    mean_fidelity_error: float | None
    mean_pauli_obs_error: float
    tdvp_lr_pauli_count: int
    enriched_lr_pauli_count: int
    tdvp_fraction: float
    enriched_fraction: float
    total_wall_time_s: float
    mean_max_bond_dim: float
    mean_sum_chi_cubed: float
    worst_case_circuit: str | None
    worst_case_family: str | None
    worst_case_gate_route_summary: str | None
    recommend_default: str | None
    recommend_reason: str | None


@dataclass(frozen=True)
class CircuitMetrics:
    fidelity_error: float | None
    pauli_obs_max_error: float
    pauli_obs_mean_error: float
    pauli_obs_l2_error: float
    worst_observable: str | None


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


def _mps_overlap(left: MPS, right: MPS) -> complex:
    env = np.ones((1, 1), dtype=np.complex128)
    for n_site in range(left.length):
        a = np.asarray(left.tensors[n_site], dtype=np.complex128)
        b = np.asarray(right.tensors[n_site], dtype=np.complex128)
        env = np.einsum("pab,ac,pcd->bd", np.conjugate(a), env, b, optimize=True)
    return complex(env.reshape(()))


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


def _pauli_obs_errors_vs_statevector(qc: QuantumCircuit, mps: MPS, ref_vec: np.ndarray) -> CircuitMetrics:
    obs = _observable_set(qc)
    errs: list[tuple[str, float]] = []
    n = qc.num_qubits
    for name, ob, _sites in obs:
        ket_op = MPS(length=mps.length, tensors=[np.asarray(t, dtype=np.complex128) for t in mps.tensors])
        ket_op.apply_local(ob)
        got = float(np.real(_mps_overlap(mps, ket_op)))

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
        return CircuitMetrics(
            fidelity_error=None,
            pauli_obs_max_error=0.0,
            pauli_obs_mean_error=0.0,
            pauli_obs_l2_error=0.0,
            worst_observable=None,
        )
    worst_name, worst_val = max(errs, key=lambda x: x[1])
    vals = np.array([e for _, e in errs], dtype=np.float64)
    return CircuitMetrics(
        fidelity_error=None,
        pauli_obs_max_error=float(worst_val),
        pauli_obs_mean_error=float(np.mean(vals)),
        pauli_obs_l2_error=float(np.linalg.norm(vals)),
        worst_observable=str(worst_name),
    )


def _pauli_obs_errors_vs_mps(qc: QuantumCircuit, mps: MPS, ref_mps: MPS) -> CircuitMetrics:
    obs = _observable_set(qc)
    errs: list[tuple[str, float]] = []
    for name, ob, _sites in obs:
        ket_op = MPS(length=mps.length, tensors=[np.asarray(t, dtype=np.complex128) for t in mps.tensors])
        ket_op.apply_local(ob)
        got = float(np.real(_mps_overlap(mps, ket_op)))

        ref_ket_op = MPS(length=ref_mps.length, tensors=[np.asarray(t, dtype=np.complex128) for t in ref_mps.tensors])
        ref_ket_op.apply_local(ob)
        ref = float(np.real(_mps_overlap(ref_mps, ref_ket_op)))
        errs.append((name, abs(got - ref)))
    if not errs:
        return CircuitMetrics(
            fidelity_error=None,
            pauli_obs_max_error=0.0,
            pauli_obs_mean_error=0.0,
            pauli_obs_l2_error=0.0,
            worst_observable=None,
        )
    worst_name, worst_val = max(errs, key=lambda x: x[1])
    vals = np.array([e for _, e in errs], dtype=np.float64)
    return CircuitMetrics(
        fidelity_error=None,
        pauli_obs_max_error=float(worst_val),
        pauli_obs_mean_error=float(np.mean(vals)),
        pauli_obs_l2_error=float(np.linalg.norm(vals)),
        worst_observable=str(worst_name),
    )


def _mps_cost_metrics(mps: MPS) -> tuple[int, float]:
    max_bond = int(mps.get_max_bond())
    bond_dims = [int(np.asarray(t).shape[2]) for t in mps.tensors[:-1]]
    sum_chi_cubed = float(sum((chi**3 for chi in bond_dims)))
    return max_bond, sum_chi_cubed


def _run_adaptive(
    qc: QuantumCircuit,
    *,
    defect_tol: float,
    svd_threshold: float = 1e-14,
    max_bond_dim: int | None = None,
) -> tuple[MPS, float]:
    params = StrongSimParams(
        preset="exact",
        get_state=True,
        svd_threshold=svd_threshold,
        max_bond_dim=max_bond_dim,
        krylov_tol=1e-12,
        gate_mode="hybrid",
        tangent_blindness_tol=1e-12,
        tdvp_projection_defect_tol=float(defect_tol),
    )
    sim = Simulator()
    init = State(qc.num_qubits, initial="zeros", representation="mps")
    t0 = time.perf_counter()
    result = sim.run(init, qc, params)
    dt = float(time.perf_counter() - t0)
    assert result.output_state is not None
    return result.output_state.mps, dt


def _run_tebd_swaps_reference(qc: QuantumCircuit, *, svd_threshold: float = 1e-14) -> MPS:
    params = StrongSimParams(
        preset="exact",
        get_state=True,
        svd_threshold=svd_threshold,
        max_bond_dim=None,
        gate_mode="tebd",
    )
    sim = Simulator()
    init = State(qc.num_qubits, initial="zeros", representation="mps")
    result = sim.run(init, qc, params)
    assert result.output_state is not None
    return result.output_state.mps


def _route_stats(mps: MPS) -> tuple[int, int, list[dict[str, Any]]]:
    stats = getattr(mps, "route_stats", None)
    if not isinstance(stats, dict):
        return 0, 0, []
    tdvp_c = int(stats.get("tdvp_lr_pauli", 0))
    enr_c = int(stats.get("enriched_lr_pauli", 0))
    ratios_any = stats.get("ratios", [])
    if not isinstance(ratios_any, list):
        return tdvp_c, enr_c, []
    ratios: list[dict[str, Any]] = [r for r in ratios_any if isinstance(r, dict)]
    return tdvp_c, enr_c, ratios


def _failed_tdvp_gate_defect_extrema(ratios: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    tdvp_projected_ratios: list[float] = []
    tdvp_projection_defects: list[float] = []
    for r in ratios:
        if r.get("route") != "tdvp":
            continue
        pr = r.get("projected_ratio", None)
        pd = r.get("projection_defect", None)
        if pr is not None:
            tdvp_projected_ratios.append(float(pr))
        if pd is not None:
            tdvp_projection_defects.append(float(pd))
    max_pr = None if not tdvp_projected_ratios else float(max(tdvp_projected_ratios))
    min_def = None if not tdvp_projection_defects else float(min(tdvp_projection_defects))
    return max_pr, min_def


def _build_cases() -> list[CircuitCase]:
    cases: list[CircuitCase] = []

    # Base diagnostic suite (from benchmark script).
    qc = QuantumCircuit(8)
    qc.ryy(0.25, 1, 6)
    cases.append(CircuitCase("diagnostic", "tangent_blind_ryy_8q", 8, 1, qc))

    qc = QuantumCircuit(8)
    qc.rx(np.pi / 2, 6)
    qc.ryy(0.25, 1, 6)
    cases.append(CircuitCase("diagnostic", "endpoint_prepared_ryy_8q", 8, 2, qc))

    qc = QuantumCircuit(10)
    qc.ry(np.pi / 4, 1)
    qc.ry(np.pi / 5, 6)
    qc.rzz(0.25, 1, 6)
    cases.append(CircuitCase("diagnostic", "known_rzz_ratio_0809_n10_1_6", 10, 3, qc))

    # Angle sweep.
    theta_values = [0.01, 0.05, 0.1, 0.25, 0.5, 0.9, 1.3]
    for n in (8, 10, 12, 14):
        pairs = [(0, n - 1), (1, n - 2), (2, n - 3), (n // 3, 2 * n // 3)]
        for gate in ("rxx", "ryy", "rzz"):
            for theta in theta_values:
                for i, j in pairs:
                    qc = QuantumCircuit(n)
                    if gate == "rzz":
                        qc.ry(np.pi / 4, i)
                        qc.ry(np.pi / 5, j)
                    getattr(qc, gate)(theta, i, j)
                    cases.append(CircuitCase("angle", f"{gate}_n{n}_th{theta}_{i}_{j}", n, 1, qc))

    # Endpoint phi sweep.
    phi_values = np.linspace(0, np.pi / 2, 9)
    n = 10
    i, j = 1, 8
    theta = 0.25
    for phi in phi_values:
        qc = QuantumCircuit(n)
        qc.ry(phi, i)
        qc.rxx(theta, i, j)
        cases.append(CircuitCase("phi", f"rxx_phi{phi:.3f}", n, 2, qc))

        qc = QuantumCircuit(n)
        qc.rx(phi, i)
        qc.ryy(theta, i, j)
        cases.append(CircuitCase("phi", f"ryy_phi{phi:.3f}", n, 2, qc))

        qc = QuantumCircuit(n)
        qc.ry(phi, i)
        qc.rzz(theta, i, j)
        cases.append(CircuitCase("phi", f"rzz_phi{phi:.3f}", n, 2, qc))

    # Mixed-axis repeated layers (deterministic).
    for n in (12, 14, 16):
        edges = [(0, n - 1), (1, n - 2), (2, n - 3), (n // 3, 2 * n // 3)]
        axes = ["rxx", "ryy", "rzz", "rxx", "rzz", "ryy"]
        for L in (1, 2, 4, 8):
            qc = QuantumCircuit(n)
            for _layer in range(L):
                for a, (u, v) in zip(axes, edges * 2, strict=False):
                    if a == "rzz":
                        qc.ry(np.pi / 6, u)
                        qc.ry(np.pi / 7, v)
                    getattr(qc, a)(0.19, u, v)
            cases.append(CircuitCase("layers", f"mixed_axes_n{n}_L{L}", n, L, qc))

    # Random Pauli circuits.
    seeds = list(range(20))
    n_values = [8, 10, 12, 14]
    depth_values = [4, 8, 16]
    axes = ["rxx", "ryy", "rzz"]
    for n in n_values:
        for depth in depth_values:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                qc = QuantumCircuit(n)
                for _layer in range(depth):
                    # Optional 1q layer.
                    for q in range(n):
                        ang = float(rng.uniform(-0.5, 0.5))
                        gate = rng.choice(["rx", "ry", "rz"])
                        getattr(qc, gate)(ang, q)

                    # One LR Pauli gate per layer.
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
                cases.append(CircuitCase("random", f"rand_n{n}_d{depth}_s{seed}", n, depth, qc))

    return cases


def _render_md(rows: list[ThresholdSummary]) -> str:
    lines: list[str] = []
    lines.append("# Projection defect tolerance stability\n")
    lines.append(
        "Routing rule:\n\n"
        r"- compute \(r = ||P_T H_g|\psi>|| / ||H_g|\psi>||\)" "\n"
        r"- compute projection defect \(d = \max(0, 1 - \min(r, 1))\)" "\n"
        r"- route to TDVP if \(d \le \varepsilon\), else route to enrichment" "\n"
    )
    lines.append("## Threshold sweep summary\n")
    lines.append(
        "defect_tol | strict pass/fail | practical pass/fail | max fid err | max obs err | TDVP count | enrichment count | wall time\n"
    )
    lines.append("---|---:|---:|---:|---:|---:|---:|---:\n")
    for r in rows:
        lines.append(
            f"{r.defect_tol:.1e} | {r.strict_passed}/{r.strict_failed} | {r.practical_passed}/{r.practical_failed} | "
            f"{('—' if r.max_fidelity_error is None else f'{r.max_fidelity_error:.3e}')} | {r.max_pauli_obs_error:.3e} | "
            f"{r.tdvp_lr_pauli_count} | {r.enriched_lr_pauli_count} | {r.total_wall_time_s:.2f}"
        )
    lines.append("\n## Worst cases\n")
    for r in rows:
        if r.strict_failed == 0 and r.practical_failed == 0:
            continue
        lines.append(f"- **eps={r.defect_tol:.1e}** worst: `{r.worst_case_circuit}` ({r.worst_case_family})")
        if r.worst_case_gate_route_summary:
            lines.append(f"  - route summary: `{r.worst_case_gate_route_summary}`")
    lines.append("\n## Recommendation\n")
    best = next((x for x in rows if x.recommend_default), None)
    if best is None:
        lines.append("- No recommendation computed.")
    else:
        lines.append(f"- **Recommended default: {best.recommend_default}**")
        lines.append(f"- **Reason**: {best.recommend_reason}")
    lines.append("\n## Interpretation\n")
    lines.append("- Strict failures indicate deviation from machine-precision Qiskit agreement.")
    lines.append("- Practical failures indicate observable error above **1e-3**.")
    lines.append("\n## Route usage\n")
    for r in rows:
        lines.append(f"- At defect_tol = **{r.defect_tol:.1e}**:")
        lines.append(f"  - TDVP-routed LR Pauli gates: {r.tdvp_lr_pauli_count}")
        lines.append(f"  - enriched LR Pauli gates: {r.enriched_lr_pauli_count}")
        lines.append(f"  - TDVP fraction: {r.tdvp_fraction:.3f}")
        lines.append(f"  - enriched fraction: {r.enriched_fraction:.3f}")
    return "\n".join(lines) + "\n"


def main() -> None:
    defect_tols = [1e-2, 5e-3, 1e-3, 1e-4, 1e-6]
    cases = _build_cases()

    out_dir = _results_dir()
    csv_path = out_dir / "projection_defect_tol_stability.csv"
    md_path = out_dir / "projection_defect_tol_stability.md"

    summaries: list[ThresholdSummary] = []

    strict_fid_tol = 1e-10
    strict_obs_tol = 1e-10
    practical_obs_tol = 1e-3
    practical_fid_tol = 1e-5

    for eps in defect_tols:
        t0 = time.perf_counter()
        fid_errs: list[float] = []
        obs_max_errs: list[float] = []
        obs_mean_errs: list[float] = []
        tdvp_total = 0
        enr_total = 0
        worst_case = None
        worst_family = None
        worst_route = None
        worst_score = -1.0
        strict_passed = 0
        strict_failed = 0
        practical_passed = 0
        practical_failed = 0
        max_bonds: list[int] = []
        sum_chi_cubeds: list[float] = []

        for case in cases:
            # Reference:
            ref_vec = _qiskit_vec(case.qc) if case.n_qubits <= 14 else None

            mps, _dt = _run_adaptive(case.qc, defect_tol=eps)
            tdvp_c, enr_c, ratios = _route_stats(mps)
            tdvp_total += tdvp_c
            enr_total += enr_c
            max_bond, sum_chi_cubed = _mps_cost_metrics(mps)
            max_bonds.append(max_bond)
            sum_chi_cubeds.append(sum_chi_cubed)

            if ref_vec is not None:
                vec = np.asarray(mps.to_vec(), dtype=np.complex128)
                fid = _fid_err(ref_vec, vec)
                metrics = _pauli_obs_errors_vs_statevector(case.qc, mps, ref_vec)
                fid_errs.append(fid)
                metrics = replace(metrics, fidelity_error=fid)
                obs_max_errs.append(metrics.pauli_obs_max_error)
                obs_mean_errs.append(metrics.pauli_obs_mean_error)

                strict_ok = metrics.pauli_obs_max_error <= strict_obs_tol and fid <= strict_fid_tol
                practical_ok = metrics.pauli_obs_max_error <= practical_obs_tol
                _ = fid <= practical_fid_tol
                if strict_ok:
                    strict_passed += 1
                else:
                    strict_failed += 1
                if practical_ok:
                    practical_passed += 1
                else:
                    practical_failed += 1

                score = max(fid, metrics.pauli_obs_max_error)
                if score > worst_score:
                    worst_score = score
                    worst_case = case.name
                    worst_family = case.family
                    worst_route = json.dumps(ratios, separators=(",", ":"), sort_keys=True)
            else:
                # Compare observables to strict TEBD+SWAPs where feasible.
                # Keep this bounded: only do for n<=16 to avoid excessive runtime.
                if case.n_qubits <= 16:
                    ref_mps = _run_tebd_swaps_reference(case.qc)
                    metrics = _pauli_obs_errors_vs_mps(case.qc, mps, ref_mps)
                    obs_max_errs.append(metrics.pauli_obs_max_error)
                    obs_mean_errs.append(metrics.pauli_obs_mean_error)
                    strict_ok = metrics.pauli_obs_max_error <= strict_obs_tol
                    practical_ok = metrics.pauli_obs_max_error <= practical_obs_tol
                    if strict_ok:
                        strict_passed += 1
                    else:
                        strict_failed += 1
                    if practical_ok:
                        practical_passed += 1
                    else:
                        practical_failed += 1

                    score = metrics.pauli_obs_max_error
                    if score > worst_score:
                        worst_score = score
                        worst_case = case.name
                        worst_family = case.family
                        worst_route = json.dumps(ratios, separators=(",", ":"), sort_keys=True)

        wall = float(time.perf_counter() - t0)
        max_fid = None if not fid_errs else float(max(fid_errs))
        mean_fid = None if not fid_errs else float(np.mean(fid_errs))
        max_obs = float(max(obs_max_errs)) if obs_max_errs else 0.0
        mean_obs = float(np.mean(obs_mean_errs)) if obs_mean_errs else 0.0
        tdvp_frac = float(tdvp_total) / float(max(tdvp_total + enr_total, 1))
        enr_frac = float(enr_total) / float(max(tdvp_total + enr_total, 1))

        summaries.append(
            ThresholdSummary(
                defect_tol=float(eps),
                num_circuits=len(cases),
                strict_passed=int(strict_passed),
                strict_failed=int(strict_failed),
                practical_passed=int(practical_passed),
                practical_failed=int(practical_failed),
                max_fidelity_error=max_fid,
                max_pauli_obs_error=max_obs,
                mean_fidelity_error=mean_fid,
                mean_pauli_obs_error=mean_obs,
                tdvp_lr_pauli_count=tdvp_total,
                enriched_lr_pauli_count=enr_total,
                total_wall_time_s=wall,
                worst_case_circuit=worst_case,
                worst_case_family=worst_family,
                worst_case_gate_route_summary=worst_route,
                tdvp_fraction=tdvp_frac,
                enriched_fraction=enr_frac,
                mean_max_bond_dim=float(np.mean(max_bonds)) if max_bonds else 0.0,
                mean_sum_chi_cubed=float(np.mean(sum_chi_cubeds)) if sum_chi_cubeds else 0.0,
                recommend_default=None,
                recommend_reason=None,
            )
        )

    # Recommendation: based on practical observable accuracy, not strict correctness.
    passing = [s for s in summaries if s.practical_failed == 0]
    rec: ThresholdSummary | None = None
    if passing:
        rec = sorted(passing, key=lambda s: (s.defect_tol, -s.tdvp_lr_pauli_count))[0]
        reason = (
            f"zero failures with defect_tol={rec.defect_tol:.1e}; "
            f"tdvp_lr_pauli={rec.tdvp_lr_pauli_count}, enriched_lr_pauli={rec.enriched_lr_pauli_count}"
        )
        # Replace the stored objects with the recommendation embedded.
        new_summaries: list[ThresholdSummary] = []
        for s in summaries:
            if s.defect_tol == rec.defect_tol:
                new_summaries.append(
                    replace(
                        s,
                        recommend_default=f"{rec.defect_tol:.1e}",
                        recommend_reason=reason,
                    )
                )
            else:
                new_summaries.append(s)
        summaries = new_summaries
    else:
        # No threshold in the sweep achieved zero failures. Make this explicit.
        worst = sorted(summaries, key=lambda s: (s.practical_failed, s.max_pauli_obs_error))[0]
        reason = (
            "no defect_tol in sweep achieved zero failures; "
            f"best observed was defect_tol={worst.defect_tol:.1e} with "
            f"{worst.practical_failed} practical failures (max obs {worst.max_pauli_obs_error:.3e})."
        )
        summaries = [
            replace(s, recommend_default="none", recommend_reason=reason) if s is summaries[0] else s for s in summaries
        ]

    # Write CSV.
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ThresholdSummary.__annotations__.keys()))
        w.writeheader()
        for s in summaries:
            w.writerow(s.__dict__)

    md_path.write_text(_render_md(summaries), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    if rec is not None:
        print(f"Recommended default: {rec.defect_tol:.1e}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

