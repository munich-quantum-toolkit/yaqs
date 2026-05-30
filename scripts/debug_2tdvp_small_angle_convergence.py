#!/usr/bin/env python
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""2TDVP small-angle convergence diagnostic for long-range Pauli gates.

Run:

    uv run python -m scripts.debug_2tdvp_small_angle_convergence

Outputs:
    results/2tdvp_small_angle_convergence.csv
    results/2tdvp_small_angle_convergence.md
"""

from __future__ import annotations

import copy
import csv
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RZZGate
from qiskit.quantum_info import Statevector

from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.digital.digital_tjm import (
    apply_pauli_product_rotation_enriched,
    apply_single_qubit_gate,
    apply_two_qubit_gate_tebd,
    apply_two_qubit_gate_tdvp_experimental,
)
from mqt.yaqs.digital.utils.dag_utils import convert_dag_to_tensor_algorithm

from scripts.benchmark_enriched_tdvp_vs_tebd import (
    Case,
    _apply_two_qubit_enriched_tdvp,
    _fid_err_vec,
    _ising_2d_row_major,
    _prep_initial_state,
)
from scripts.debug_enriched_tdvp_fidelity_failures import (
    build_prefix_circuit,
    iter_gate_steps,
)

Row = dict[str, Any]

INCLUDE_SWEEP_16 = os.environ.get("YAQS_SMALL_ANGLE_FULL", "").strip() not in {"", "0", "false", "False"}

THETA_VALUES: tuple[float, ...] = (
    2e-1,
    1e-1,
    5e-2,
    2.5e-2,
    1.25e-2,
    6.25e-3,
    3.125e-3,
    1.5625e-3,
)

M_VALUES: tuple[int, ...] = (1, 2, 4, 8, 16, 32)

DEFAULT_SVD_THRESHOLD = 1e-12

TDVP_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "current_local",
        "window_size": 1,
        "tdvp_sweeps": 1,
        "tdvp_circuit_full_sweep": False,
    },
    {
        "name": "local_full_sweep",
        "window_size": 1,
        "tdvp_sweeps": 1,
        "tdvp_circuit_full_sweep": True,
    },
    {
        "name": "local_many_sweeps",
        "window_size": 1,
        "tdvp_sweeps": 8,
        "tdvp_circuit_full_sweep": True,
    },
    {
        "name": "full_chain_many_sweeps",
        "window_size": "full",
        "tdvp_sweeps": 8,
        "tdvp_circuit_full_sweep": True,
    },
]

if INCLUDE_SWEEP_16:
    TDVP_CONFIGS.append(
        {
            "name": "full_chain_16_sweeps",
            "window_size": "full",
            "tdvp_sweeps": 16,
            "tdvp_circuit_full_sweep": True,
        }
    )

SUBSTEP_TDVP_CONFIG = TDVP_CONFIGS[3]  # full_chain_many_sweeps


@dataclass(frozen=True)
class TargetGate:
    circuit_name: str
    prefix_gate_count: int
    gate_name: str
    sites: tuple[int, int]
    label: str


TARGET_GATES: tuple[TargetGate, ...] = (
    TargetGate("ising2d_3x3_h1_dt0.1_L4_plus", 47, "rzz", (4, 7), "h1_L4_plus_rzz_4_7"),
    TargetGate("ising2d_3x3_h1_dt0.1_L4_random_product", 65, "rzz", (4, 7), "random_rzz_4_7"),
    TargetGate("ising2d_3x3_h2_dt0.1_L2_random_product", 65, "rzz", (4, 7), "h2_L2_random_rzz_4_7"),
    TargetGate("ising2d_3x3_h0.5_dt0.1_L2_all_zero", 34, "rzz", (2, 5), "control_rzz_2_5"),
)


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
    n = lx * ly
    qc_body, edge_types = _ising_2d_row_major(lx=lx, ly=ly, j=1.0, h=h, dt=dt, layers=layers, periodic_x=True)
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


def _make_params() -> StrongSimParams:
    return StrongSimParams(
        preset="exact",
        get_state=True,
        svd_threshold=DEFAULT_SVD_THRESHOLD,
        max_bond_dim=None,
        krylov_tol=1e-12,
        gate_mode="hybrid",
        tangent_blindness_tol=1e-12,
        tdvp_projection_defect_tol=1e-4,
        tdvp_visibility_safety_tol=None,
        tdvp_pauli_consistency_check=False,
    )


def _phase_aligned_l2_error(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.vdot(b, b).real)
    if denom < 1e-300:
        return float(np.linalg.norm(a))
    alpha = np.vdot(a, b) / denom
    return float(np.linalg.norm(a - alpha * b))


def _state_delta_norm(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-300 or nb < 1e-300:
        return 1.0
    denom = float(np.vdot(b, b).real)
    alpha = np.vdot(a, b) / denom if denom > 1e-300 else 0.0
    return float(np.linalg.norm(a / na - alpha * b / nb))


def _apply_prefix(case: Case, num_gates: int) -> MPS:
    params = _make_params()
    mps = State(case.n_qubits, initial="zeros", representation="mps").mps
    for step in iter_gate_steps(case.qc, order="dag_topo")[:num_gates]:
        node = step.node
        if len(node.qargs) == 1:
            apply_single_qubit_gate(mps, node)
            continue
        gate = convert_dag_to_tensor_algorithm(node)[0]
        if abs(gate.sites[0] - gate.sites[1]) == 1:
            apply_two_qubit_gate_tebd(mps, gate, params)
        else:
            _apply_two_qubit_enriched_tdvp(mps, node, params)
    return mps


def _gate_at_theta(target: TargetGate, theta: float):
    from mqt.yaqs.core.libraries.gate_library import GateLibrary

    gate = getattr(GateLibrary, target.gate_name)([float(theta)])
    gate.set_sites(*target.sites)
    return gate


def _reference_statevector(case: Case, target: TargetGate, theta: float) -> np.ndarray:
    after = build_prefix_circuit(case.qc, target.prefix_gate_count, order="dag_topo")
    i, j = target.sites
    getattr(after, target.gate_name)(float(theta), i, j)
    return np.asarray(Statevector(after).data, dtype=np.complex128)


def _eval_vs_reference(mps: MPS, ref_vec: np.ndarray) -> tuple[float, float, float]:
    vec = np.asarray(mps.to_vec(), dtype=np.complex128)
    return (
        _fid_err_vec(ref_vec, vec),
        _phase_aligned_l2_error(vec, ref_vec),
        _state_delta_norm(vec, ref_vec),
    )


def _apply_method(
    prefix_mps: MPS,
    gate,
    params: StrongSimParams,
    method: Literal["tebd", "enrichment"],
) -> tuple[MPS, float]:
    trial = copy.deepcopy(prefix_mps)
    t0 = time.perf_counter()
    if method == "tebd":
        apply_two_qubit_gate_tebd(trial, gate, params)
    else:
        apply_pauli_product_rotation_enriched(trial, gate, params, record_stats=False)
    wall = float(time.perf_counter() - t0)
    return trial, wall


def _theta_list(original_theta: float) -> list[float]:
    thetas = list(THETA_VALUES)
    if not any(abs(t - original_theta) < 1e-15 * max(1.0, abs(original_theta)) for t in thetas):
        thetas.append(original_theta)
    return sorted(set(thetas), reverse=True)


def _fit_scaling(
    thetas: list[float],
    errors: list[float],
    *,
    original_theta: float,
) -> tuple[float, float, float, bool]:
    """Return (p, error_at_original, error_at_smallest, monotonic_decrease)."""
    if not errors:
        return float("nan"), float("nan"), float("nan"), False

    pairs = sorted(zip(thetas, errors, strict=True), key=lambda x: x[0], reverse=True)
    err_by_theta = {t: e for t, e in pairs}
    err_orig = err_by_theta.get(original_theta, pairs[0][1] if pairs else float("nan"))
    err_small = pairs[-1][1] if pairs else float("nan")
    monotonic = all(pairs[i][1] >= pairs[i + 1][1] for i in range(len(pairs) - 1))

    valid = [(t, e) for t, e in pairs if 1e-14 < e < 1e-2 and t > 0]
    if len(valid) < 2:
        return float("nan"), err_orig, err_small, monotonic

    log_t = np.array([math.log(t) for t, _ in valid], dtype=np.float64)
    log_e = np.array([math.log(e) for _, e in valid], dtype=np.float64)
    slope, _intercept = np.polyfit(log_t, log_e, 1)
    return float(slope), err_orig, err_small, monotonic


def _classify_target(
    *,
    estimated_order_p: float,
    error_smallest: float,
    config_name: str,
    monotonic: bool,
) -> str:
    if config_name != "full_chain_many_sweeps" and config_name != "full_chain_16_sweeps":
        return "inconclusive"
    if math.isnan(estimated_order_p):
        if error_smallest > 1e-6:
            return "likely TDVP implementation bug"
        return "inconclusive"
    if estimated_order_p >= 1.5 and error_smallest <= 1e-6 and monotonic:
        return "finite-step integrator error"
    if estimated_order_p >= 1.0 and monotonic and error_smallest < 1e-4:
        return "finite-step integrator error"
    if error_smallest > 1e-4 and not monotonic:
        return "likely TDVP implementation bug"
    if error_smallest > 1e-6 and estimated_order_p < 0.5:
        return "likely TDVP implementation bug"
    if error_smallest > 1e-6 and monotonic and estimated_order_p >= 1.0:
        return "window/environment issue"
    return "inconclusive"


def _write_csv(path: Path, rows: list[Row]) -> None:
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
    scaling_summaries: list[Row],
    substep_rows: list[Row],
    reference_rows: list[Row],
    classifications: dict[str, str],
    final_answers: list[str],
) -> str:
    lines: list[str] = ["# 2TDVP small-angle convergence diagnostic\n\n", "## Summary\n\n"]
    for s in scaling_summaries:
        lines.append(
            f"- **{s['target_label']}** / `{s['tdvp_config']}`: "
            f"error {s['error_at_original_theta']:.3e} → {s['error_at_smallest_theta']:.3e}, "
            f"p≈{s['estimated_order_p']}, monotonic={s['monotonic_decrease']}, "
            f"class={classifications.get(s['target_label'], '?')}\n"
        )

    lines.append("\n## Angle scaling tables\n\n")
    lines.append(
        "| target | tdvp_config | error_original | error_smallest | estimated_order_p | monotonic |\n"
    )
    lines.append("|---|---|---:|---:|---:|---|\n")
    for s in scaling_summaries:
        p = s["estimated_order_p"]
        p_s = "NaN" if isinstance(p, float) and math.isnan(p) else f"{float(p):.2f}"
        lines.append(
            f"| {s['target_label']} | {s['tdvp_config']} | {float(s['error_at_original_theta']):.3e} | "
            f"{float(s['error_at_smallest_theta']):.3e} | {p_s} | {s['monotonic_decrease']} |\n"
        )

    lines.append("\n## Substep convergence\n\n")
    lines.append("| target | m | error | wall_time | max_bond |\n")
    lines.append("|---|---|---:|---:|---:|\n")
    for r in substep_rows:
        lines.append(
            f"| {r['target_label']} | {r['m']} | {float(r['fidelity_error']):.3e} | "
            f"{float(r['wall_time_s']):.3f} | {r['max_bond_after']} |\n"
        )

    lines.append("\n## Reference sanity checks\n\n")
    lines.append("| target | theta | tebd | enrichment | dense |\n")
    lines.append("|---|---|---:|---:|---:|\n")
    for r in reference_rows:
        lines.append(
            f"| {r['target_label']} | {float(r['theta']):.4g} | {float(r['tebd_error']):.3e} | "
            f"{float(r['enrichment_error']):.3e} | {float(r['dense_exact_error']):.3e} |\n"
        )

    lines.append("\n## Interpretation\n\n")
    for label, cls in classifications.items():
        lines.append(f"- **{label}**: {cls}\n")

    lines.append("\n## Final answers\n\n")
    for a in final_answers:
        lines.append(f"{a}\n")
    return "".join(lines)


def main() -> None:
    out_dir = _results_dir()
    csv_path = out_dir / "2tdvp_small_angle_convergence.csv"
    md_path = out_dir / "2tdvp_small_angle_convergence.md"

    all_rows: list[Row] = []
    scaling_summaries: list[Row] = []
    substep_rows: list[Row] = []
    reference_rows: list[Row] = []
    classifications: dict[str, str] = {}

    cases = {t.circuit_name: _build_case(t.circuit_name) for t in TARGET_GATES}
    params = _make_params()

    for target in TARGET_GATES:
        case = cases[target.circuit_name]
        print(f"\n=== {target.label} ===")
        prefix_mps = _apply_prefix(case, target.prefix_gate_count)
        steps = iter_gate_steps(case.qc, order="dag_topo")
        template_gate = convert_dag_to_tensor_algorithm(steps[target.prefix_gate_count].node)[0]
        original_theta = float(getattr(template_gate, "theta", 0.0))
        max_bond_before = int(prefix_mps.get_max_bond())

        thetas = _theta_list(original_theta)

        # Reference sanity at original theta
        ref_at_orig = _reference_statevector(case, target, original_theta)
        gate_orig_ref = _gate_at_theta(target, original_theta)
        i_ref, j_ref = sorted(target.sites)
        vec_pre = np.asarray(prefix_mps.to_vec(), dtype=np.complex128)
        dense_vec_orig = np.asarray(
            Statevector(vec_pre).evolve(RZZGate(float(original_theta)), [i_ref, j_ref]).data,
            dtype=np.complex128,
        )
        dense_fid_orig = _fid_err_vec(ref_at_orig, dense_vec_orig)
        tebd_trial, _ = _apply_method(prefix_mps, gate_orig_ref, params, "tebd")
        tebd_fid_orig, _, _ = _eval_vs_reference(tebd_trial, ref_at_orig)
        enr_trial, _ = _apply_method(prefix_mps, gate_orig_ref, params, "enrichment")
        enr_fid_orig, _, _ = _eval_vs_reference(enr_trial, ref_at_orig)
        reference_rows.append(
            {
                "target_label": target.label,
                "theta": original_theta,
                "tebd_error": tebd_fid_orig,
                "enrichment_error": enr_fid_orig,
                "dense_exact_error": dense_fid_orig,
            }
        )

        for theta in thetas:
            gate = _gate_at_theta(target, theta)
            ref_vec = _reference_statevector(case, target, theta)
            theta_rel = theta / original_theta if original_theta != 0 else float("nan")

            # Per-theta references
            tebd_trial, _ = _apply_method(prefix_mps, gate, params, "tebd")
            tebd_fid, tebd_l2, tebd_delta = _eval_vs_reference(tebd_trial, ref_vec)
            enr_trial, _ = _apply_method(prefix_mps, gate, params, "enrichment")
            enr_fid, enr_l2, enr_delta = _eval_vs_reference(enr_trial, ref_vec)

            i, j = sorted(target.sites)
            vec_pre = np.asarray(prefix_mps.to_vec(), dtype=np.complex128)
            dense_vec = np.asarray(
                Statevector(vec_pre).evolve(RZZGate(float(theta)), [i, j]).data,
                dtype=np.complex128,
            )
            dense_fid = _fid_err_vec(ref_vec, dense_vec)

            for method, trial, fid, l2, delta, wall in (
                (
                    "tebd",
                    tebd_trial,
                    tebd_fid,
                    tebd_l2,
                    tebd_delta,
                    0.0,
                ),
                (
                    "enrichment",
                    enr_trial,
                    enr_fid,
                    enr_l2,
                    enr_delta,
                    0.0,
                ),
                (
                    "dense_exact",
                    prefix_mps,
                    dense_fid,
                    _phase_aligned_l2_error(dense_vec, ref_vec),
                    _state_delta_norm(dense_vec, ref_vec),
                    0.0,
                ),
            ):
                all_rows.append(
                    {
                        "section": "angle_sweep",
                        "circuit": case.circuit_name,
                        "target_label": target.label,
                        "target_gate": target.gate_name,
                        "sites": str(target.sites),
                        "theta": theta,
                        "theta_relative_to_original": theta_rel,
                        "method": method,
                        "tdvp_config": "",
                        "fidelity_error": fid,
                        "phase_aligned_l2_error": l2,
                        "state_delta_norm": delta,
                        "tebd_error": tebd_fid,
                        "enrichment_error": enr_fid,
                        "dense_exact_error": dense_fid,
                        "max_bond_before": max_bond_before,
                        "max_bond_after": max_bond_before if method == "dense_exact" else int(trial.get_max_bond()),
                        "wall_time_s": wall,
                    }
                )

            for cfg in TDVP_CONFIGS:
                trial = copy.deepcopy(prefix_mps)
                t0 = time.perf_counter()
                apply_two_qubit_gate_tdvp_experimental(
                    trial,
                    gate,
                    params,
                    window_size=cfg["window_size"],
                    tdvp_sweeps=int(cfg["tdvp_sweeps"]),
                    substeps=1,
                    tdvp_circuit_full_sweep=bool(cfg["tdvp_circuit_full_sweep"]),
                )
                wall = float(time.perf_counter() - t0)
                fid, l2, delta = _eval_vs_reference(trial, ref_vec)
                all_rows.append(
                    {
                        "section": "angle_sweep",
                        "circuit": case.circuit_name,
                        "target_label": target.label,
                        "target_gate": target.gate_name,
                        "sites": str(target.sites),
                        "theta": theta,
                        "theta_relative_to_original": theta_rel,
                        "method": "2tdvp",
                        "tdvp_config": cfg["name"],
                        "window_size": cfg["window_size"],
                        "tdvp_sweeps": cfg["tdvp_sweeps"],
                        "tdvp_circuit_full_sweep": cfg["tdvp_circuit_full_sweep"],
                        "fidelity_error": fid,
                        "phase_aligned_l2_error": l2,
                        "state_delta_norm": delta,
                        "tdvp_error": fid,
                        "tebd_error": tebd_fid,
                        "enrichment_error": enr_fid,
                        "dense_exact_error": dense_fid,
                        "max_bond_before": max_bond_before,
                        "max_bond_after": int(trial.get_max_bond()),
                        "wall_time_s": wall,
                    }
                )

        # Scaling per TDVP config
        for cfg in TDVP_CONFIGS:
            cfg_rows = [
                r
                for r in all_rows
                if r.get("section") == "angle_sweep"
                and r.get("target_label") == target.label
                and r.get("method") == "2tdvp"
                and r.get("tdvp_config") == cfg["name"]
            ]
            thetas_fit = [float(r["theta"]) for r in cfg_rows]
            errs_fit = [float(r["fidelity_error"]) for r in cfg_rows]
            p, err_orig, err_small, monotonic = _fit_scaling(
                thetas_fit, errs_fit, original_theta=original_theta
            )
            summary = {
                "section": "scaling_summary",
                "target_label": target.label,
                "circuit": case.circuit_name,
                "tdvp_config": cfg["name"],
                "estimated_order_p": p,
                "error_at_original_theta": err_orig,
                "error_at_smallest_theta": err_small,
                "monotonic_decrease": monotonic,
            }
            scaling_summaries.append(summary)
            all_rows.append(summary)

        # Classify using full_chain_many_sweeps
        fc = next(s for s in scaling_summaries if s["target_label"] == target.label and s["tdvp_config"] == "full_chain_many_sweeps")
        classifications[target.label] = _classify_target(
            estimated_order_p=float(fc["estimated_order_p"]),
            error_smallest=float(fc["error_at_smallest_theta"]),
            config_name="full_chain_many_sweeps",
            monotonic=bool(fc["monotonic_decrease"]),
        )

        # Substep equivalence at original theta
        gate_orig = _gate_at_theta(target, original_theta)
        ref_orig = _reference_statevector(case, target, original_theta)
        enr_trial, _ = _apply_method(prefix_mps, gate_orig, params, "enrichment")
        enr_fid, _, _ = _eval_vs_reference(enr_trial, ref_orig)

        for m in M_VALUES:
            trial = copy.deepcopy(prefix_mps)
            t0 = time.perf_counter()
            apply_two_qubit_gate_tdvp_experimental(
                trial,
                gate_orig,
                params,
                window_size=SUBSTEP_TDVP_CONFIG["window_size"],
                tdvp_sweeps=int(SUBSTEP_TDVP_CONFIG["tdvp_sweeps"]),
                substeps=m,
                tdvp_circuit_full_sweep=bool(SUBSTEP_TDVP_CONFIG["tdvp_circuit_full_sweep"]),
            )
            wall = float(time.perf_counter() - t0)
            fid, _, _ = _eval_vs_reference(trial, ref_orig)
            row = {
                "section": "substep",
                "circuit": case.circuit_name,
                "target_label": target.label,
                "target_gate": target.gate_name,
                "sites": str(target.sites),
                "theta": original_theta,
                "m": m,
                "theta_per_substep": original_theta / m,
                "fidelity_error": fid,
                "enrichment_error": enr_fid,
                "wall_time_s": wall,
                "max_bond_before": max_bond_before,
                "max_bond_after": int(trial.get_max_bond()),
            }
            substep_rows.append(row)
            all_rows.append(row)
            print(f"  substep m={m}: fid={fid:.3e} (enrichment={enr_fid:.3e})")

    # Final narrative answers
    final_answers: list[str] = []
    for target in TARGET_GATES:
        fc = next(
            s
            for s in scaling_summaries
            if s["target_label"] == target.label and s["tdvp_config"] == "full_chain_many_sweeps"
        )
        p = float(fc["estimated_order_p"])
        final_answers.append(
            f"**{target.label}**: θ smallest error={float(fc['error_at_smallest_theta']):.3e}, "
            f"θ original error={float(fc['error_at_original_theta']):.3e}, p≈{p:.2f}."
        )

    final_answers.extend(
        [
            "1. **θ→0**: See per-target monotonic flag and error_at_smallest_theta in scaling table.",
            "2. **Full-chain vs local**: Compare `full_chain_many_sweeps` vs `current_local` columns.",
            "3. **Substepping**: See substep table vs enrichment at original θ.",
            "4. **Order p**: From log-log fit on TDVP rows with 1e-14 < error < 1e-2.",
            "5. **Parameter vs bug**: Classifications in Interpretation section.",
            "6. **Production TDVP route**: Only viable if full-chain+substeps reach ≤1e-6 at circuit θ; else use enrichment.",
        ]
    )

    _write_csv(csv_path, all_rows)
    md_path.write_text(
        _render_md(
            scaling_summaries=scaling_summaries,
            substep_rows=substep_rows,
            reference_rows=reference_rows,
            classifications=classifications,
            final_answers=final_answers,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
