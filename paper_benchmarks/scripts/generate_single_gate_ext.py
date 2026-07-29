# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Stage 3a (generate missing data only): corrected single-gate extension.

Extends the corrected single-gate campaign (RZZ, seed 11; protocol
compress_rightcanon_ltr+var_multistart+tdvp_n1_v1) to the missing cells while
keeping every locked convention identical:

  * angle sweep for gates rxx/ryy/rzz x seeds {11,22,33} x chi {8,12,16} on
    the exact corrected angle grid (RZZ/seed-11 cells are reused, not re-run);
  * a separate theta=0 identity row per (gate, seed, chi, method);
  * the exact-limit substep study: x = 1/4, chi_max = 32 (nonbinding),
    n in {1,...,256}, all three gates, with phase-aligned self-convergence
    against the highest-substep result.

Deterministic, sqlite-checkpointed (task_id = sha256 of the task payload),
resume-safe, parallel over (gate, seed, chi) units with PB_WORKERS workers,
BLAS pinned to one thread per worker.

Usage:
    uv run python paper_benchmarks/scripts/generate_single_gate_ext.py
"""

from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from pb_common import (
    LOGS_DIR,
    RAW_NEW_DIR,
    SG_ANGLE_TDVP_SUBSTEPS,
    SG_CHI_GRID,
    SG_GATES,
    SG_L,
    SG_Q0,
    SG_Q1,
    SG_SEEDS,
    SUBSTEP_STUDY_CHI,
    SUBSTEP_STUDY_NS,
    SUBSTEP_STUDY_X,
    add_experiment_path,
    limit_blas_threads,
    worker_count,
)

limit_blas_threads()
add_experiment_path("single_gate")

import numpy as np  # ruff: ignore[module-import-not-at-top-of-file]

BENCHMARK_ID = "paper_benchmarks_single_gate_ext_v1"
DB_PATH = RAW_NEW_DIR / "single_gate_ext.sqlite"
CSV_PATH = RAW_NEW_DIR / "single_gate_ext.csv"

COLUMNS = [
    "task_id", "task_type", "gate_type", "seed", "q0", "q1", "separation",
    "method", "chi_max", "chi0", "theta", "x_fraction", "special_angle",
    "substeps", "infidelity", "fidelity", "overlap_squared_raw",
    "norm_squared_exact", "norm_squared_approx", "norm_loss",
    "phase_aligned_error_exact", "phase_aligned_error_selfref",
    "fidelity_definition", "max_bond", "bond_profile", "param_count",
    "runtime_s", "norm_before", "norm_after", "norm_drift",
    "discarded_weight", "variational_converged", "variational_failed",
    "failure_message", "benchmark_id",
]


def task_id(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def open_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL")
    cols = ", ".join(f"{c}" for c in COLUMNS)
    conn.execute(f"CREATE TABLE IF NOT EXISTS results ({cols}, PRIMARY KEY (task_id))")
    conn.commit()
    return conn


def angle_grid() -> list[tuple[float, bool]]:
    from config import build_generic_angle_grid, build_special_angles

    x_gen, _ = build_generic_angle_grid()
    x_spec, _ = build_special_angles()
    return [(float(x), False) for x in x_gen] + [(float(x), True) for x in x_spec]


def run_one_method(
    initial: dict[str, Any],
    *,
    gate_type: str,
    method: str,
    theta: float,
    chi: int,
    substeps: int,
) -> dict[str, Any]:
    """Gate-parameterized port of experiments/single_gate/core.run_method."""
    from gate_runtime import (
        DiscardedWeightTracker,
        apply_gate_to_dense_state,
        apply_method,
        bond_profile,
        make_dag_node,
        make_gate,
        normalized_state_fidelity,
        param_count_from_profile,
        phase_align,
    )
    from variational import apply_variational_mpo_gate, tt_svd_from_vec

    gate = make_gate(gate_type, theta, SG_Q0, SG_Q1)
    g4 = np.asarray(gate.matrix, dtype=np.complex128)
    exact_vec = apply_gate_to_dense_state(initial["vec"], g4, SG_Q0, SG_Q1, SG_L)
    tracker = DiscardedWeightTracker()
    failure_message = ""
    var_converged = None
    var_failed = None
    t0 = time.perf_counter()
    if method == "no_update":
        approx_vec = initial["vec"]
        prof = initial["bond_profile"]
        param_count = 0
    elif method == "ttsvd_candidate":
        tt = tt_svd_from_vec(exact_vec, SG_L, chi)
        approx_vec = tt.to_vec()
        prof = bond_profile(tt)
        param_count = param_count_from_profile(prof, SG_L)
        failure_message = "independent_ttsvd_candidate"
    elif method == "variational_mpo":
        node = make_dag_node(gate_type, theta, SG_Q0, SG_Q1, SG_L)
        vres = apply_variational_mpo_gate(copy.deepcopy(initial["mps"]), node, chi=chi)
        approx_vec = vres.state.to_vec()
        prof = bond_profile(vres.state)
        param_count = param_count_from_profile(prof, SG_L)
        var_converged = vres.converged
        var_failed = vres.failed
        if vres.failed:
            failure_message = "variational_failed"
        elif vres.best_initializer:
            failure_message = f"best_init={vres.best_initializer}"
    else:
        node = make_dag_node(gate_type, theta, SG_Q0, SG_Q1, SG_L)
        state, _, _ = apply_method(
            initial["mps"], node, method=method, chi=chi, substeps=substeps, tracker=tracker,
        )
        approx_vec = state.to_vec()
        prof = bond_profile(state)
        param_count = param_count_from_profile(prof, SG_L)
    runtime = time.perf_counter() - t0
    approx_vec = np.asarray(approx_vec, dtype=np.complex128)
    metrics = normalized_state_fidelity(exact_vec, approx_vec)
    unit_approx = approx_vec / np.linalg.norm(approx_vec)
    unit_exact = exact_vec / np.linalg.norm(exact_vec)
    pa_err = float(np.linalg.norm(phase_align(unit_exact, unit_approx) - unit_exact))
    norm_after = float(np.linalg.norm(approx_vec))
    return {
        "infidelity": metrics["infidelity_normalized"],
        "fidelity": metrics["fidelity_normalized"],
        "overlap_squared_raw": metrics["overlap_squared_raw"],
        "norm_squared_exact": metrics["norm_squared_exact"],
        "norm_squared_approx": metrics["norm_squared_approx"],
        "norm_loss": metrics["norm_loss"],
        "phase_aligned_error_exact": pa_err,
        "fidelity_definition": "normalized_state_fidelity_v2",
        "max_bond": int(max(prof)),
        "bond_profile": json.dumps([int(b) for b in prof]),
        "param_count": int(param_count),
        "runtime_s": float(runtime),
        "norm_before": float(np.linalg.norm(initial["vec"])),
        "norm_after": norm_after,
        "norm_drift": abs(1.0 - norm_after),
        "discarded_weight": float(tracker.cumulative),
        "variational_converged": None if var_converged is None else int(var_converged),
        "variational_failed": None if var_failed is None else int(var_failed),
        "failure_message": failure_message,
        "_approx_vec": approx_vec,
    }


ANGLE_METHODS = (
    "hybrid_tdvp", "tebd_swap", "mpo_zipup", "variational_mpo", "no_update", "ttsvd_candidate",
)


def angle_unit(args: tuple[str, int, int, set[str]]) -> list[dict[str, Any]]:
    """Worker: one (gate, seed, chi) angle-sweep unit incl. theta=0 rows."""
    limit_blas_threads()
    gate_type, seed, chi, done = args
    from gate_runtime import prepare_initial_state

    initial = prepare_initial_state(seed)
    rows: list[dict[str, Any]] = []
    grid = [(0.0, False, "theta_zero")] + [
        (x, special, "angle_sweep") for x, special in angle_grid()
    ]
    for x, special, ttype in grid:
        theta = float(2.0 * np.pi * x)
        for method in ANGLE_METHODS:
            substeps = SG_ANGLE_TDVP_SUBSTEPS if method == "hybrid_tdvp" else (
                0 if method in {"no_update", "ttsvd_candidate"} else 1
            )
            payload = {
                "benchmark_id": BENCHMARK_ID,
                "task_type": ttype,
                "gate_type": gate_type,
                "seed": seed,
                "method": method,
                "chi_max": chi,
                "x_fraction": x,
                "substeps": substeps,
            }
            tid = task_id(payload)
            if tid in done:
                continue
            res = run_one_method(
                initial, gate_type=gate_type, method=method, theta=theta, chi=chi,
                substeps=substeps,
            )
            res.pop("_approx_vec")
            rows.append({
                "task_id": tid,
                "task_type": ttype,
                "gate_type": gate_type,
                "seed": seed,
                "q0": SG_Q0,
                "q1": SG_Q1,
                "separation": SG_Q1 - SG_Q0,
                "method": method,
                "chi_max": chi,
                "chi0": 8,
                "theta": theta,
                "x_fraction": x,
                "special_angle": int(special),
                "substeps": substeps,
                "phase_aligned_error_selfref": None,
                "benchmark_id": BENCHMARK_ID,
                **res,
            })
    return rows


def substep_unit(args: tuple[str, set[str]]) -> list[dict[str, Any]]:
    """Worker: exact-limit substep study for one gate at x=1/4, chi=32."""
    limit_blas_threads()
    gate_type, done = args
    from gate_runtime import phase_align, prepare_initial_state

    seed = 11
    initial = prepare_initial_state(seed)
    theta = float(2.0 * np.pi * SUBSTEP_STUDY_X)
    results: dict[int, tuple[str, dict[str, Any]]] = {}
    for n in SUBSTEP_STUDY_NS:
        payload = {
            "benchmark_id": BENCHMARK_ID,
            "task_type": "substep_study",
            "gate_type": gate_type,
            "seed": seed,
            "method": "hybrid_tdvp",
            "chi_max": SUBSTEP_STUDY_CHI,
            "x_fraction": SUBSTEP_STUDY_X,
            "substeps": n,
        }
        tid = task_id(payload)
        res = run_one_method(
            initial, gate_type=gate_type, method="hybrid_tdvp", theta=theta,
            chi=SUBSTEP_STUDY_CHI, substeps=n,
        )
        results[n] = (tid, res)
    n_ref = max(SUBSTEP_STUDY_NS)
    ref_vec = results[n_ref][1]["_approx_vec"]
    ref_unit = ref_vec / np.linalg.norm(ref_vec)
    rows = []
    for n, (tid, res) in sorted(results.items()):
        vec = res.pop("_approx_vec")
        unit = vec / np.linalg.norm(vec)
        selfref = float(np.linalg.norm(phase_align(ref_unit, unit) - ref_unit))
        if tid in done:
            continue
        rows.append({
            "task_id": tid,
            "task_type": "substep_study",
            "gate_type": gate_type,
            "seed": seed,
            "q0": SG_Q0,
            "q1": SG_Q1,
            "separation": SG_Q1 - SG_Q0,
            "method": "hybrid_tdvp",
            "chi_max": SUBSTEP_STUDY_CHI,
            "chi0": 8,
            "theta": theta,
            "x_fraction": SUBSTEP_STUDY_X,
            "special_angle": 1,
            "substeps": n,
            "phase_aligned_error_selfref": selfref,
            "benchmark_id": BENCHMARK_ID,
            **res,
        })
    return rows


def export_csv(conn: sqlite3.Connection) -> None:
    import csv

    cur = conn.execute(f"SELECT {', '.join(COLUMNS)} FROM results ORDER BY task_type, gate_type, seed, chi_max, x_fraction, method, substeps")
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        writer.writerows(cur.fetchall())


def main() -> int:
    t_start = time.perf_counter()
    conn = open_db()
    done = {row[0] for row in conn.execute("SELECT task_id FROM results")}
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "generate_single_gate_ext.log"

    units: list[tuple[str, tuple]] = []
    for gate_type in SG_GATES:
        for seed in SG_SEEDS:
            if gate_type == "rzz" and seed == 11:
                continue  # reuse corrected campaign rows; do not re-run
            units.extend(("angle", (gate_type, seed, chi, done)) for chi in SG_CHI_GRID)
    units.extend(("substep", (gate_type, done)) for gate_type in SG_GATES)

    n_workers = worker_count()
    print(f"{len(units)} work units, {len(done)} completed tasks in checkpoint, "
          f"{n_workers} workers")
    inserted = 0
    failures = 0
    with log_path.open("a", encoding="utf-8") as log, ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for kind, args in units:
            fn = angle_unit if kind == "angle" else substep_unit
            futures[pool.submit(fn, args)] = (kind, args[:3] if kind == "angle" else args[:1])
        for fut in as_completed(futures):
            kind, key = futures[fut]
            try:
                rows = fut.result()
            except Exception as exc:  # log and continue; checkpoint stays valid
                failures += 1
                msg = f"UNIT FAILED {kind} {key}: {exc!r}"
                print(msg, flush=True)
                log.write(msg + "\n")
                continue
            placeholders = ", ".join("?" for _ in COLUMNS)
            for row in rows:
                conn.execute(
                    f"INSERT OR IGNORE INTO results ({', '.join(COLUMNS)}) VALUES ({placeholders})",
                    [row.get(c) for c in COLUMNS],
                )
            conn.commit()
            inserted += len(rows)
            msg = f"unit done {kind} {key}: +{len(rows)} rows"
            print(msg, flush=True)
            log.write(msg + "\n")

    export_csv(conn)
    total = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    conn.close()
    wall = time.perf_counter() - t_start
    print(f"done: +{inserted} new rows, {total} total, {failures} failed units, "
          f"wall {wall:.1f}s -> {CSV_PATH}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
