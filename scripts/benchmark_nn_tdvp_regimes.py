#!/usr/bin/env python
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""NN brickwork benchmark: TEBD vs all-TDVP on nearest-neighbor-only circuits.

Every two-qubit gate in the ``tdvp_all`` method uses standard TDVP (including
nearest-neighbor gates). Compare against ``tebd_nn``, which applies TEBD to all
two-qubit gates. Only ``nn_brickwork`` circuits are included.

Run::

    uv run python -m scripts.benchmark_nn_tdvp_regimes

Environment::

    YAQS_NN_TDVP_STAGE=0|1
    YAQS_NN_TDVP_MAX_CASES=N
    YAQS_NN_TDVP_OUTDIR=results/nn_tdvp_regimes
    YAQS_NN_TDVP_OVERWRITE=1
    YAQS_NN_TDVP_INCLUDE_SWEEP_SCAN=0|1
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from dataclasses import asdict
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

from scripts.benchmark_long_range_tdvp_regimes import (
    REFERENCE_CONVENTION,
    CircuitSpec,
    GateDispatchStats,
    RawRun,
    _apply_grouped_observable_errors,
    _assign_dispatch_fields,
    _static_gate_counts,
    build_all_specs,
    compute_reference,
    validate_reference_convention,
    write_circuit_metadata,
)
from scripts.benchmark_utils import _mean_bond_dim, _prep_initial_state
from scripts.yaqs_reference_utils import (
    ReferenceConvention,
    check_high_fidelity_observable_consistency,
    compare_statevectors,
)

STAGE = int(os.environ.get("YAQS_NN_TDVP_STAGE", "0"))
MAX_CASES = int(os.environ.get("YAQS_NN_TDVP_MAX_CASES", "0"))
OVERWRITE = os.environ.get("YAQS_NN_TDVP_OVERWRITE", "").strip() not in {"", "0", "false", "False"}
OUTDIR = Path(os.environ.get("YAQS_NN_TDVP_OUTDIR", "results/nn_tdvp_regimes"))
INCLUDE_SWEEP_SCAN = os.environ.get("YAQS_NN_TDVP_INCLUDE_SWEEP_SCAN", "0").strip() not in {
    "",
    "0",
    "false",
    "False",
}

SVD_THRESHOLD = 1e-10
LANCZOS_TOL = 1e-8

MethodName = Literal["tebd_nn", "tdvp_all"]

NN_STAGE_CONFIGS: dict[int, dict[str, Any]] = {
    0: {
        "n_values": [8, 10],
        "depth_values": [2, 4, 8],
        "num_seeds": 1,
        "chi_values": [16, 32],
        "tdvp_sweep_scan_values": [1, 2, 4],
        "angle_regimes": ["small"],
        "families": ["nn_brickwork"],
        "exact_n_max": 16,
        "include_large_angles": False,
    },
    1: {
        "n_values": [8, 12, 16],
        "depth_values": [4, 8, 12],
        "num_seeds": 3,
        "chi_values": [16, 32, 64],
        "tdvp_sweep_scan_values": [1, 2, 4],
        "angle_regimes": ["small", "medium"],
        "families": ["nn_brickwork"],
        "exact_n_max": 16,
        "include_large_angles": False,
    },
}


def _resolve_sweep_values(cfg: dict[str, Any]) -> list[int]:
    if INCLUDE_SWEEP_SCAN:
        return list(cfg.get("tdvp_sweep_scan_values", [1, 2, 4]))
    return [1]


def build_nn_specs(cfg: dict[str, Any]) -> list[CircuitSpec]:
    """Build ``nn_brickwork`` specs only."""
    nn_cfg = {**cfg, "families": ["nn_brickwork"]}
    specs = build_all_specs(nn_cfg)
    if MAX_CASES > 0:
        return specs[:MAX_CASES]
    return specs


def _run_nn_method(
    qc: QuantumCircuit,
    spec: CircuitSpec,
    *,
    method: MethodName,
    chi: int,
    tdvp_sweeps: int,
) -> tuple[MPS, float, GateDispatchStats]:
    """Run NN-only circuits with TEBD or TDVP on every two-qubit gate."""
    import time

    gate_mode = "tdvp" if method == "tdvp_all" else "tebd"
    params = StrongSimParams(
        preset="exact",
        get_state=True,
        svd_threshold=SVD_THRESHOLD,
        max_bond_dim=chi,
        krylov_tol=LANCZOS_TOL,
        gate_mode=gate_mode,
        tdvp_projection_defect_tol=1e-3,
        tdvp_pauli_consistency_check=False,
    )

    static_total, static_nn, static_lr = _static_gate_counts(qc)
    dispatch = GateDispatchStats(
        total_gates=static_total,
        nn_gate_count=static_nn,
        lr_gate_count=static_lr,
    )

    mps = State(spec.n_qubits, initial="zeros", representation="mps").mps
    prep = QuantumCircuit(spec.n_qubits)
    _prep_initial_state(prep, "plus", seed=0)  # type: ignore[arg-type]
    for node in circuit_to_dag(prep).topological_op_nodes():
        if len(node.qargs) == 1:
            apply_single_qubit_gate(mps, node)

    peak = 0
    t0 = time.perf_counter()
    for node in circuit_to_dag(qc).topological_op_nodes():
        if node.op.name == "barrier":
            continue
        if len(node.qargs) == 1:
            apply_single_qubit_gate(mps, node)
            continue
        gate = convert_dag_to_tensor_algorithm(node)[0]
        i, j = int(gate.sites[0]), int(gate.sites[1])
        if method == "tebd_nn":
            dispatch.tebd_direct_gate_count += 1
            apply_two_qubit_gate_tebd(mps, gate, params)
        else:
            apply_two_qubit_gate_tdvp_experimental(mps, gate, params, tdvp_sweeps=tdvp_sweeps)
            dispatch.tdvp_lr_gate_count += 1
        peak = max(peak, int(mps.get_max_bond()))
    wall = float(time.perf_counter() - t0)
    mps._bench_peak_bond = peak  # type: ignore[attr-defined]
    return mps, wall, dispatch


def run_nn_method(
    spec: CircuitSpec,
    *,
    method: MethodName,
    chi: int,
    tdvp_sweeps: int,
    sweep_scan_enabled: bool,
    ref: MPS | np.ndarray | None,
    ref_type: str,
    ref_chi: int | None,
    ref_sweeps: int | None,
) -> RawRun:
    sweeps = tdvp_sweeps if method == "tdvp_all" else 1
    run = RawRun(
        family=spec.family,
        n_qubits=spec.n_qubits,
        depth=spec.depth,
        seed=spec.seed,
        angle_regime=spec.angle_regime,
        method=method,
        chi_max=chi,
        tdvp_sweeps=sweeps,
        svd_threshold=SVD_THRESHOLD,
        lanczos_tol=LANCZOS_TOL,
        reference_type=ref_type,
        reference_available=ref is not None,
        reference_chi=ref_chi,
        reference_sweeps=ref_sweeps,
        case_id=spec.case_id,
        sweep_scan_enabled=sweep_scan_enabled,
    )
    try:
        mps, wall, dispatch = _run_nn_method(
            spec.qc, spec, method=method, chi=chi, tdvp_sweeps=sweeps
        )
        run.wall_time_s = wall
        run.peak_bond_dim = int(getattr(mps, "_bench_peak_bond", mps.get_max_bond()))
        run.final_max_bond_dim = int(mps.get_max_bond())
        run.mean_bond_dim = _mean_bond_dim(mps)
        run.hit_chi_max = run.final_max_bond_dim >= chi
        _assign_dispatch_fields(run, dispatch)
        run.tdvp_lr_count = dispatch.tdvp_lr_gate_count if method == "tdvp_all" else 0
        run.enriched_lr_count = 0

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


def _run_key(run: RawRun) -> str:
    return f"{run.case_id}|{run.method}|chi{run.chi_max}|sweeps{run.tdvp_sweeps}"


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    v = row.get(key)
    if v in (None, ""):
        return default
    return float(v)


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


def _classify_nn_pair(tdvp: dict[str, Any], tebd: dict[str, Any]) -> str:
    if tdvp.get("status") != "ok" or tebd.get("status") != "ok":
        return "both_fail"
    t_err = _f(tdvp, "mean_abs_observable_error", float("inf"))
    e_err = _f(tebd, "mean_abs_observable_error", float("inf"))
    if t_err > 0.1 and e_err > 0.1:
        return "both_fail"
    if t_err <= e_err / 2:
        return "tdvp_wins"
    if e_err <= t_err / 2:
        return "tebd_wins"
    return "similar"


def summarize_nn_runs(raw: list[dict[str, Any]], outdir: Path) -> None:
    ok = [r for r in raw if r.get("status") == "ok"]
    comparisons: list[dict[str, Any]] = []
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in ok:
        key = f"{r['case_id']}|chi{r['chi_max']}|sweeps{r['tdvp_sweeps']}"
        by_case[key].append(r)

    for _key, rows in by_case.items():
        tdvp = next((r for r in rows if r["method"] == "tdvp_all"), None)
        tebd = next((r for r in rows if r["method"] == "tebd_nn"), None)
        if tdvp is None or tebd is None:
            continue
        t_err = _f(tdvp, "mean_abs_observable_error")
        e_err = _f(tebd, "mean_abs_observable_error")
        comparisons.append({
            "case_id": tdvp["case_id"],
            "n_qubits": tdvp["n_qubits"],
            "depth": tdvp["depth"],
            "seed": tdvp["seed"],
            "angle_regime": tdvp["angle_regime"],
            "chi_max": tdvp["chi_max"],
            "tdvp_sweeps": tdvp["tdvp_sweeps"],
            "tebd_error": e_err,
            "tdvp_error": t_err,
            "tebd_fidelity": _f(tebd, "fidelity_to_reference"),
            "tdvp_fidelity": _f(tdvp, "fidelity_to_reference"),
            "tebd_peak_chi": tebd["peak_bond_dim"],
            "tdvp_peak_chi": tdvp["peak_bond_dim"],
            "tdvp_peak_chi_over_tebd_peak_chi": _f(tdvp, "peak_bond_dim")
            / max(_f(tebd, "peak_bond_dim"), 1),
            "tebd_wall_time": _f(tebd, "wall_time_s"),
            "tdvp_wall_time": _f(tdvp, "wall_time_s"),
            "tdvp_time_over_tebd_time": _f(tdvp, "wall_time_s")
            / max(_f(tebd, "wall_time_s"), 1e-9),
            "tebd_error_over_tdvp_error": e_err / max(t_err, 1e-15),
            "tdvp_error_over_tebd_error": t_err / max(e_err, 1e-15),
            "tebd_hit_chi_max": tebd.get("hit_chi_max"),
            "tdvp_hit_chi_max": tdvp.get("hit_chi_max"),
            "tdvp_gate_count": tdvp.get("tdvp_lr_gate_count"),
            "tebd_gate_count": tebd.get("tebd_direct_gate_count"),
            "outcome": _classify_nn_pair(tdvp, tebd),
        })
    _write_csv(outdir / "nn_method_comparison.csv", comparisons)

    sweep_rows: list[dict[str, Any]] = []
    by_sweep: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in ok:
        if r["method"] != "tdvp_all":
            continue
        sk = f"{r['case_id']}|chi{r['chi_max']}"
        by_sweep[sk].append(r)
    for _sk, rows in by_sweep.items():
        rows_sorted = sorted(rows, key=lambda x: int(x["tdvp_sweeps"]))
        for i in range(1, len(rows_sorted)):
            prev, curr = rows_sorted[i - 1], rows_sorted[i]
            pe = _f(prev, "mean_abs_observable_error", 1e-15)
            ce = _f(curr, "mean_abs_observable_error", 1e-15)
            err_ratio = ce / pe
            if err_ratio < 0.7:
                cls = "sweep_helpful"
            elif err_ratio > 1.3:
                cls = "sweep_harmful_or_unstable"
            else:
                cls = "sweep_saturated"
            sweep_rows.append({
                "case_id": curr["case_id"],
                "chi_max": curr["chi_max"],
                "sweeps_from": prev["tdvp_sweeps"],
                "sweeps_to": curr["tdvp_sweeps"],
                "error_ratio": err_ratio,
                "classification": cls,
            })
    _write_csv(outdir / "tdvp_sweep_scaling.csv", sweep_rows)

    _write_nn_report(comparisons, sweep_rows, raw, outdir / "report.md", stage=STAGE)


def _write_nn_report(
    comparisons: list[dict[str, Any]],
    sweep_rows: list[dict[str, Any]],
    raw: list[dict[str, Any]],
    path: Path,
    *,
    stage: int,
) -> None:
    lines = [
        "# NN brickwork: TEBD vs all-TDVP\n\n",
        f"Stage **{stage}**. Circuits: ``nn_brickwork`` only. "
        "``tdvp_all`` applies TDVP to every two-qubit gate (including NN).\n\n",
        "## Summary\n\n",
    ]
    if comparisons:
        tdvp_w = sum(1 for c in comparisons if c["outcome"] == "tdvp_wins")
        tebd_w = sum(1 for c in comparisons if c["outcome"] == "tebd_wins")
        similar = sum(1 for c in comparisons if c["outcome"] == "similar")
        lines.append(
            f"- Comparisons: {len(comparisons)}; TDVP wins: {tdvp_w}; "
            f"TEBD wins: {tebd_w}; similar: {similar}\n"
        )
        faster = sum(1 for c in comparisons if _f(c, "tdvp_time_over_tebd_time") < 0.9)
        more_acc = sum(1 for c in comparisons if _f(c, "tebd_error_over_tdvp_error") > 1.5)
        lines.append(
            f"- TDVP faster than TEBD: {faster}/{len(comparisons)}; "
            f"TDVP more accurate (err ratio>1.5): {more_acc}/{len(comparisons)}\n"
        )
        lower_chi = sum(
            1 for c in comparisons if _f(c, "tdvp_peak_chi_over_tebd_peak_chi") < 0.95
        )
        lines.append(
            f"- TDVP lower peak χ than TEBD: {lower_chi}/{len(comparisons)}\n\n"
        )
        best_tdvp = sorted(comparisons, key=lambda c: _f(c, "tebd_error_over_tdvp_error"), reverse=True)[:6]
        lines.append("## Largest TDVP advantage (TEBD/TDVP error ratio)\n\n")
        for c in best_tdvp:
            lines.append(
                f"- `{c['case_id']}` χ={c['chi_max']}: ratio={_f(c, 'tebd_error_over_tdvp_error'):.2f}, "
                f"outcome={c['outcome']}\n"
            )
        best_tebd = sorted(comparisons, key=lambda c: _f(c, "tdvp_error_over_tebd_error"), reverse=True)[:6]
        lines.append("\n## Largest TEBD advantage (TDVP/TEBD error ratio)\n\n")
        for c in best_tebd:
            lines.append(
                f"- `{c['case_id']}` χ={c['chi_max']}: ratio={_f(c, 'tdvp_error_over_tebd_error'):.2f}\n"
            )
    else:
        lines.append("- No matched TEBD/TDVP pairs.\n")

    lines.append("\n## Sweep diagnostic\n\n")
    if INCLUDE_SWEEP_SCAN:
        lines.append(
            "> Multi-sweep TDVP is diagnostic only; extra sweeps may worsen NN circuits.\n\n"
        )
    if sweep_rows:
        harmful = sum(1 for s in sweep_rows if s["classification"] == "sweep_harmful_or_unstable")
        helpful = sum(1 for s in sweep_rows if s["classification"] == "sweep_helpful")
        lines.append(f"- sweep_helpful: {helpful}; sweep_harmful: {harmful}\n")
    else:
        lines.append("- No sweep pairs (default uses ``tdvp_sweeps=1`` only).\n")

    lines.append(f"\n---\n\nTotal raw runs: {len(raw)}.\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    if STAGE not in NN_STAGE_CONFIGS:
        raise SystemExit(f"Unknown YAQS_NN_TDVP_STAGE={STAGE}")

    cfg = NN_STAGE_CONFIGS[STAGE]
    outdir = OUTDIR if OUTDIR.is_absolute() else Path(__file__).resolve().parents[1] / OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)

    validate_reference_convention(chi=max(cfg["chi_values"]))

    specs = build_nn_specs(cfg)
    write_circuit_metadata(specs, outdir / "circuit_metadata.jsonl")

    raw_path = outdir / "raw_runs.csv"
    if OVERWRITE and raw_path.exists():
        raw_path.unlink()

    existing: set[str] = set()
    if raw_path.exists() and not OVERWRITE:
        with raw_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing.add(
                    f"{row.get('case_id')}|{row.get('method')}|chi{row.get('chi_max')}"
                    f"|sweeps{row.get('tdvp_sweeps', '1')}"
                )

    chi_values = list(cfg["chi_values"])
    sweep_values = _resolve_sweep_values(cfg)
    methods: list[MethodName] = ["tebd_nn", "tdvp_all"]
    sweep_scan = INCLUDE_SWEEP_SCAN
    total = len(specs) * len(chi_values) * (1 + len(sweep_values))
    case_num = 0

    print(
        f"NN TDVP stage {STAGE}: {len(specs)} nn_brickwork circuits, chi={chi_values}, "
        f"sweeps={sweep_values}, sweep_scan={sweep_scan}, out={outdir}"
    )

    ref_cache: dict[str, tuple[Any, str, int | None, int | None]] = {}

    fieldnames = list(RawRun.__dataclass_fields__.keys())
    for spec in specs:
        if spec.case_id not in ref_cache:
            ref, ref_type, ref_chi, ref_sw = compute_reference(
                spec, exact_n_max=int(cfg["exact_n_max"])
            )
            ref_cache[spec.case_id] = (ref, ref_type, ref_chi, ref_sw)
        ref, ref_type, ref_chi, ref_sw = ref_cache[spec.case_id]

        for chi in chi_values:
            for method in methods:
                sweeps_list = sweep_values if method == "tdvp_all" else [1]
                for tdvp_sweeps in sweeps_list:
                    case_num += 1
                    run = run_nn_method(
                        spec,
                        method=method,
                        chi=chi,
                        tdvp_sweeps=tdvp_sweeps,
                        sweep_scan_enabled=sweep_scan,
                        ref=ref,
                        ref_type=ref_type,
                        ref_chi=ref_chi,
                        ref_sweeps=ref_sw,
                    )
                    key = _run_key(run)
                    if key in existing:
                        print(f"[{case_num}/{total}] SKIP {key}")
                        continue

                    write_header = not raw_path.exists() or raw_path.stat().st_size == 0
                    with raw_path.open("a", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                        if write_header:
                            w.writeheader()
                        w.writerow({k: asdict(run).get(k, "") for k in fieldnames})
                    existing.add(key)
                    note = (
                        f" gates={run.total_gates} tebd={run.tebd_direct_gate_count} "
                        f"tdvp={run.tdvp_lr_gate_count}"
                    )
                    print(
                        f"[{case_num}/{total}] {spec.case_id} n={spec.n_qubits} d={spec.depth} "
                        f"chi={chi} method={method} sweeps={run.tdvp_sweeps} status={run.status}"
                        f"{note if run.status == 'ok' else ''}"
                    )

    raw: list[dict[str, Any]] = []
    if raw_path.exists():
        with raw_path.open(newline="", encoding="utf-8") as f:
            raw = list(csv.DictReader(f))
    summarize_nn_runs(raw, outdir)
    print(f"\nDone. {len(raw)} rows in {raw_path}")
    print(f"Report: {outdir / 'report.md'}")


if __name__ == "__main__":
    main()
