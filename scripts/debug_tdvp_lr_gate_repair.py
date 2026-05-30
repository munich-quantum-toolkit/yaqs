#!/usr/bin/env python
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""TDVP repair diagnostic for long-range Pauli gate failures.

Run:

    uv run python -m scripts.debug_tdvp_lr_gate_repair

Outputs:
    results/tdvp_lr_gate_repair.csv
    results/tdvp_lr_gate_repair.md
"""

from __future__ import annotations

import copy
import csv
import itertools
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Statevector

from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.digital.digital_tjm import (
    WindowSizeSpec,
    apply_pauli_product_rotation_enriched,
    apply_single_qubit_gate,
    apply_two_qubit_gate_tebd,
    apply_two_qubit_gate_tdvp,
    apply_two_qubit_gate_tdvp_experimental,
    decide_long_range_pauli_route,
)
from mqt.yaqs.digital.utils.dag_utils import convert_dag_to_tensor_algorithm

from scripts.benchmark_enriched_tdvp_vs_tebd import (
    Case,
    _apply_two_qubit_enriched_tdvp,
    _fid_err_vec,
    _ising_2d_row_major,
    _mean_bond_dim,
    _pauli_obs_errors,
    _prep_initial_state,
    _qiskit_vec,
)
from scripts.debug_enriched_tdvp_fidelity_failures import (
    build_prefix_circuit,
    iter_gate_steps,
)

GridRow = dict[str, Any]

FAST_GRID = os.environ.get("YAQS_TDVP_REPAIR_FULL", "").strip() in {"", "0", "false", "False"}

TDVP_SWEEPS_VALUES = [1, 2, 4, 8] if FAST_GRID else [1, 2, 4, 8, 16]
SUBSTEPS_VALUES = [1, 2, 4, 8] if FAST_GRID else [1, 2, 4, 8, 16]
WINDOW_VALUES: list[WindowSizeSpec] = [1, 2, 4, "full"] if FAST_GRID else [1, 2, 3, 4, "interval", "full"]
FULL_SWEEP_VALUES = [False, True]

DEFAULT_DEFECT_TOL = 1e-4
DEFAULT_SVD_THRESHOLD = 1e-12


@dataclass(frozen=True)
class TargetGate:
    circuit_name: str
    prefix_gate_count: int
    gate_name: str
    sites: tuple[int, int]
    label: str


TARGET_GATES: tuple[TargetGate, ...] = (
    TargetGate("ising2d_3x3_h1_dt0.1_L4_plus", 47, "rzz", (4, 7), "h1_L4_plus_rzz_4_7"),
    TargetGate("ising2d_3x3_h0.5_dt0.1_L4_plus", 43, "rzz", (2, 5), "h05_L4_plus_rzz_2_5"),
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
    from qiskit.circuit import QuantumCircuit

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
        tdvp_projection_defect_tol=DEFAULT_DEFECT_TOL,
        tdvp_visibility_safety_tol=None,
        tdvp_pauli_consistency_check=False,
    )


def _phase_aligned_l2_error(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.vdot(b, b).real)
    if denom < 1e-300:
        return float(np.linalg.norm(a))
    alpha = np.vdot(a, b) / denom
    return float(np.linalg.norm(a - alpha * b))


def _init_mps(n: int) -> MPS:
    return State(n, initial="zeros", representation="mps").mps


def _apply_prefix(case: Case, num_gates: int) -> MPS:
    params = _make_params()
    mps = _init_mps(case.n_qubits)
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


def _dense_apply_gate(vec: np.ndarray, n: int, node) -> np.ndarray:
    qargs = [node.qargs[0]._index, node.qargs[1]._index]  # noqa: SLF001
    evolved = Statevector(vec).evolve(node.op, qargs)
    return np.asarray(evolved.data, dtype=np.complex128)


def _eval_after_gate(
    mps: MPS,
    case: Case,
    *,
    prefix_gate_count: int,
) -> tuple[float, float]:
    ref = _qiskit_vec(build_prefix_circuit(case.qc, prefix_gate_count + 1, order="dag_topo"))
    vec = np.asarray(mps.to_vec(), dtype=np.complex128)
    return _fid_err_vec(ref, vec), _phase_aligned_l2_error(vec, ref)


def _baseline_errors(
    prefix_mps: MPS,
    case: Case,
    target: TargetGate,
    params: StrongSimParams,
) -> dict[str, float]:
    steps = iter_gate_steps(case.qc, order="dag_topo")
    bad_node = steps[target.prefix_gate_count].node
    gate = convert_dag_to_tensor_algorithm(bad_node)[0]
    theta = float(getattr(gate, "theta", 0.0))

    out: dict[str, float] = {"theta": theta}

    trial = copy.deepcopy(prefix_mps)
    apply_two_qubit_gate_tdvp(trial, gate, params)
    fid, l2 = _eval_after_gate(trial, case, prefix_gate_count=target.prefix_gate_count)
    out["plain_tdvp_error"] = fid
    out["plain_tdvp_l2"] = l2

    trial = copy.deepcopy(prefix_mps)
    apply_pauli_product_rotation_enriched(trial, gate, params, record_stats=False)
    fid, _ = _eval_after_gate(trial, case, prefix_gate_count=target.prefix_gate_count)
    out["enrichment_error"] = fid

    trial = copy.deepcopy(prefix_mps)
    apply_two_qubit_gate_tebd(trial, gate, params)
    fid, _ = _eval_after_gate(trial, case, prefix_gate_count=target.prefix_gate_count)
    out["tebd_error"] = fid

    vec_before = np.asarray(prefix_mps.to_vec(), dtype=np.complex128)
    vec_dense = _dense_apply_gate(vec_before, case.n_qubits, bad_node)
    ref = _qiskit_vec(build_prefix_circuit(case.qc, target.prefix_gate_count + 1, order="dag_topo"))
    out["dense_exact_error"] = _fid_err_vec(ref, vec_dense)

    return out


def _grid_rows_for_target(
    case: Case,
    target: TargetGate,
    baselines: dict[str, float],
) -> list[GridRow]:
    params = _make_params()
    prefix_mps = _apply_prefix(case, target.prefix_gate_count)
    steps = iter_gate_steps(case.qc, order="dag_topo")
    gate = convert_dag_to_tensor_algorithm(steps[target.prefix_gate_count].node)[0]
    max_bond_before = int(prefix_mps.get_max_bond())

    rows: list[GridRow] = []
    for sweeps, substeps, window, full_sweep in itertools.product(
        TDVP_SWEEPS_VALUES,
        SUBSTEPS_VALUES,
        WINDOW_VALUES,
        FULL_SWEEP_VALUES,
    ):
        trial = copy.deepcopy(prefix_mps)
        t0 = time.perf_counter()
        apply_two_qubit_gate_tdvp_experimental(
            trial,
            gate,
            params,
            window_size=window,
            tdvp_sweeps=sweeps,
            substeps=substeps,
            tdvp_circuit_full_sweep=full_sweep,
        )
        wall = float(time.perf_counter() - t0)
        fid, l2 = _eval_after_gate(trial, case, prefix_gate_count=target.prefix_gate_count)
        rows.append(
            {
                "section": "grid",
                "circuit": case.circuit_name,
                "target_label": target.label,
                "target_gate_index": target.prefix_gate_count,
                "gate": gate.name,
                "sites": str(tuple(gate.sites)),
                "theta": float(getattr(gate, "theta", 0.0)),
                "window_size": window,
                "tdvp_sweeps": sweeps,
                "substeps": substeps,
                "tdvp_circuit_full_sweep": full_sweep,
                "fidelity_error_after_gate": fid,
                "phase_aligned_l2_error_after_gate": l2,
                "max_bond_before": max_bond_before,
                "max_bond_after": int(trial.get_max_bond()),
                "mean_bond_after": _mean_bond_dim(trial),
                "wall_time_s": wall,
                **baselines,
            }
        )
    return rows


def _pick_best(rows: list[GridRow]) -> dict[str, GridRow | None]:
    grid = [r for r in rows if r.get("section") == "grid"]
    if not grid:
        return {"best": None, "fastest_practical": None, "lowest_bond_practical": None}

    def key_fid(r: GridRow) -> float:
        return float(r["fidelity_error_after_gate"])

    best = min(grid, key=key_fid)
    practical = [r for r in grid if float(r["fidelity_error_after_gate"]) <= 1e-6]
    fastest = min(practical, key=lambda r: float(r["wall_time_s"])) if practical else None
    lowest_bond = (
        min(practical, key=lambda r: (int(r["max_bond_after"]), float(r["fidelity_error_after_gate"])))
        if practical
        else None
    )
    return {"best": best, "fastest_practical": fastest, "lowest_bond_practical": lowest_bond}


@dataclass(frozen=True)
class RepairPolicy:
    name: str
    tdvp_sweeps: int
    substeps: int
    window_size: WindowSizeSpec
    tdvp_circuit_full_sweep: bool


def _run_full_circuit(
    case: Case,
    *,
    policy: Literal["current_router", "repaired_tdvp", "all_enrichment", "tebd_swaps"],
    repair: RepairPolicy | None = None,
) -> dict[str, Any]:
    params = _make_params()
    mps = _init_mps(case.n_qubits)
    t0 = time.perf_counter()

    for step in iter_gate_steps(case.qc, order="dag_topo"):
        node = step.node
        if len(node.qargs) == 1:
            apply_single_qubit_gate(mps, node)
            continue
        gate = convert_dag_to_tensor_algorithm(node)[0]
        is_nn = abs(gate.sites[0] - gate.sites[1]) == 1

        if policy == "tebd_swaps":
            apply_two_qubit_gate_tebd(mps, gate, params)
            continue

        if is_nn:
            apply_two_qubit_gate_tebd(mps, gate, params)
            continue

        if gate.name not in {"rxx", "ryy", "rzz"}:
            apply_two_qubit_gate_tdvp(mps, gate, params)
            continue

        if policy == "all_enrichment":
            apply_pauli_product_rotation_enriched(mps, gate, params, record_stats=False)
            continue

        decision = decide_long_range_pauli_route(mps, gate, params)
        if policy == "current_router":
            if decision.route == "pauli_enriched":
                apply_pauli_product_rotation_enriched(mps, gate, params, record_stats=False)
            else:
                apply_two_qubit_gate_tdvp(mps, gate, params)
            continue

        assert policy == "repaired_tdvp" and repair is not None
        if decision.route == "pauli_enriched":
            apply_pauli_product_rotation_enriched(mps, gate, params, record_stats=False)
        else:
            apply_two_qubit_gate_tdvp_experimental(
                mps,
                gate,
                params,
                window_size=repair.window_size,
                tdvp_sweeps=repair.tdvp_sweeps,
                substeps=repair.substeps,
                tdvp_circuit_full_sweep=repair.tdvp_circuit_full_sweep,
            )

    wall = float(time.perf_counter() - t0)
    ref_vec = _qiskit_vec(case.qc)
    vec = np.asarray(mps.to_vec(), dtype=np.complex128)
    obs_max, _, _, _ = _pauli_obs_errors(case, mps=mps, ref_vec=ref_vec, ref_mps=None)
    return {
        "section": "full_circuit",
        "circuit": case.circuit_name,
        "policy": policy,
        "repair_name": None if repair is None else repair.name,
        "fidelity_error": _fid_err_vec(ref_vec, vec),
        "pauli_obs_max_error": obs_max,
        "max_bond": int(mps.get_max_bond()),
        "mean_bond": _mean_bond_dim(mps),
        "wall_time_s": wall,
    }


def _write_csv(path: Path, rows: list[GridRow]) -> None:
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
    summaries: list[dict[str, Any]],
    grid_best: dict[str, dict[str, GridRow | None]],
    full_rows: list[dict[str, Any]],
    hypotheses: list[str],
) -> str:
    lines: list[str] = ["# TDVP LR Pauli gate repair diagnostic\n\n", "## Summary\n\n"]
    lines.append("| circuit | target | plain TDVP | enrichment | best repair fid | strict pass |\n")
    lines.append("|---|---|---:|---:|---:|---|\n")
    for s in summaries:
        lines.append(
            f"| `{s['circuit']}` | {s['target_label']} | {s['plain_tdvp_error']:.3e} | "
            f"{s['enrichment_error']:.3e} | {s['best_fidelity']:.3e} | {s['strict_pass']} |\n"
        )

    lines.append("\n## Single-gate repair grid\n\n")
    for label, picks in grid_best.items():
        best = picks.get("best")
        if best is None:
            continue
        lines.append(
            f"### {label}\n\n"
            f"- Best: sweeps={best['tdvp_sweeps']}, substeps={best['substeps']}, "
            f"window={best['window_size']}, full_sweep={best['tdvp_circuit_full_sweep']}, "
            f"fid={float(best['fidelity_error_after_gate']):.3e}, wall={float(best['wall_time_s']):.3f}s, "
            f"max_bond={best['max_bond_after']}\n\n"
        )

    lines.append("## Sweep effects\n\n")
    lines.append("(See CSV for full grids; mini-tables per target in CSV `section=grid`.)\n\n")

    lines.append("## Full-circuit validation\n\n")
    lines.append("| circuit | policy | fidelity | obs_max | max_bond | wall_s |\n")
    lines.append("|---|---|---:|---:|---:|---:|\n")
    for r in full_rows:
        lines.append(
            f"| `{r['circuit']}` | {r['policy']} | {float(r['fidelity_error']):.3e} | "
            f"{float(r['pauli_obs_max_error']):.3e} | {r['max_bond']} | {float(r['wall_time_s']):.2f} |\n"
        )

    lines.append("\n## Conclusion\n\n")
    for h in hypotheses:
        lines.append(f"- {h}\n")
    return "".join(lines)


def _hypothesis_notes(grid_rows: list[GridRow], baselines: dict[str, float]) -> list[str]:
    grid = [r for r in grid_rows if r.get("section") == "grid"]
    if not grid:
        return ["No grid data collected."]

    notes: list[str] = []
    plain = float(baselines["plain_tdvp_error"])
    enr = float(baselines["enrichment_error"])
    best = min(float(r["fidelity_error_after_gate"]) for r in grid)

    def min_fid(filter_fn) -> float:
        subset = [r for r in grid if filter_fn(r)]
        return min((float(r["fidelity_error_after_gate"]) for r in subset), default=plain)

    h1 = min(min_fid(lambda r: int(r["tdvp_sweeps"]) > 1 and int(r["substeps"]) == 1), plain)
    h2 = min(min_fid(lambda r: int(r["substeps"]) > 1 and int(r["tdvp_sweeps"]) == 1), plain)
    h3 = min(min_fid(lambda r: r["window_size"] in {2, 3, 4, "full", "interval"}), plain)
    h4 = min(min_fid(lambda r: bool(r["tdvp_circuit_full_sweep"])), plain)

    if h1 < plain * 0.1 and h1 <= 1e-6:
        notes.append("H1 (more sweeps): supported — increasing tdvp_sweeps reduces error materially.")
    elif h1 < plain:
        notes.append("H1 (more sweeps): partially supported — sweeps help but may not reach 1e-6 alone.")
    else:
        notes.append("H1 (more sweeps): not supported — more sweeps did not beat plain TDVP.")

    if h2 < plain * 0.1 and h2 <= 1e-6:
        notes.append("H2 (substepping): supported — angle subdivision fixes the gate.")
    elif h2 < plain:
        notes.append("H2 (substepping): partially supported.")
    else:
        notes.append("H2 (substepping): not supported.")

    if h3 < plain * 0.1 and h3 <= 1e-6:
        notes.append("H3 (larger window): supported.")
    elif h3 < plain:
        notes.append("H3 (larger window): partially supported.")
    else:
        notes.append("H3 (larger window): not supported.")

    if h4 < plain * 0.1 and h4 <= 1e-6:
        notes.append("H4 (full sweep): supported.")
    elif h4 < plain:
        notes.append("H4 (full sweep): partially supported.")
    else:
        notes.append("H4 (full sweep): not supported.")

    if enr <= 1e-10 and best > 1e-6:
        notes.append("All-enrichment remains exact; repaired TDVP does not match — keep enrichment for production.")
    elif best <= 1e-10:
        notes.append("A repaired TDVP setting reaches strict single-gate accuracy; validate full-circuit rows.")
    elif best <= 1e-6:
        notes.append("Practical single-gate repair (<=1e-6) may be achievable; check full-circuit validation.")

    if best > 1e-6 and min(h3, h4, h1, h2) > 1e-3:
        notes.append("H5/H6: only enrichment/TEBD are reliable — local TDVP may be unsuitable on these states.")

    return notes


def main() -> None:
    out_dir = _results_dir()
    csv_path = out_dir / "tdvp_lr_gate_repair.csv"
    md_path = out_dir / "tdvp_lr_gate_repair.md"

    all_rows: list[GridRow] = []
    summaries: list[dict[str, Any]] = []
    grid_best: dict[str, dict[str, GridRow | None]] = {}
    repair_policies: dict[str, RepairPolicy] = {}
    full_rows: list[dict[str, Any]] = []

    unique_circuits = sorted({t.circuit_name for t in TARGET_GATES})
    cases = {name: _build_case(name) for name in unique_circuits}

    for target in TARGET_GATES:
        case = cases[target.circuit_name]
        print(f"\n=== {target.label} ({case.circuit_name}) ===")
        params = _make_params()
        prefix_mps = _apply_prefix(case, target.prefix_gate_count)
        baselines = _baseline_errors(prefix_mps, case, target, params)
        print(
            f"  baselines: plain_tdvp={baselines['plain_tdvp_error']:.3e} "
            f"enrichment={baselines['enrichment_error']:.3e} tebd={baselines['tebd_error']:.3e}"
        )

        all_rows.append({"section": "baseline", "target_label": target.label, **baselines})

        grid = _grid_rows_for_target(case, target, baselines)
        all_rows.extend(grid)
        picks = _pick_best(grid)
        grid_best[target.label] = picks
        best = picks["best"]
        best_fid = float(best["fidelity_error_after_gate"]) if best else float("nan")
        summaries.append(
            {
                "circuit": case.circuit_name,
                "target_label": target.label,
                "plain_tdvp_error": baselines["plain_tdvp_error"],
                "enrichment_error": baselines["enrichment_error"],
                "best_fidelity": best_fid,
                "strict_pass": best_fid <= 1e-10,
            }
        )
        if best is not None:
            repair_policies[target.circuit_name] = RepairPolicy(
                name=f"best_{target.label}",
                tdvp_sweeps=int(best["tdvp_sweeps"]),
                substeps=int(best["substeps"]),
                window_size=best["window_size"],  # type: ignore[arg-type]
                tdvp_circuit_full_sweep=bool(best["tdvp_circuit_full_sweep"]),
            )

    for cname in unique_circuits:
        case = cases[cname]
        if case.n_qubits > 14:
            continue
        batch = [
            _run_full_circuit(case, policy="tebd_swaps"),
            _run_full_circuit(case, policy="all_enrichment"),
            _run_full_circuit(case, policy="current_router"),
        ]
        repair = repair_policies.get(cname)
        if repair is not None:
            row = _run_full_circuit(case, policy="repaired_tdvp", repair=repair)
            row["repair_name"] = repair.name
            batch.append(row)
        full_rows.extend(batch)
        all_rows.extend(batch)

    hypotheses: list[str] = []
    for target in TARGET_GATES:
        baselines = next(r for r in all_rows if r.get("section") == "baseline" and r.get("target_label") == target.label)
        grid = [r for r in all_rows if r.get("section") == "grid" and r.get("target_label") == target.label]
        hypotheses.extend(_hypothesis_notes(grid, baselines))

    _write_csv(csv_path, all_rows)
    md_path.write_text(
        _render_md(
            summaries=summaries,
            grid_best=grid_best,
            full_rows=[r for r in full_rows if r.get("section") == "full_circuit"],
            hypotheses=list(dict.fromkeys(hypotheses)),
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
