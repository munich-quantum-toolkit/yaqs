# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Fresh-process worker: one (method, χ) TFIM trajectory."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import path_setup  # noqa: F401
from circuits import build_ising_schedule
from config import (
    BYTES_PER_COMPLEX128,
    DT,
    OUTPUT_DIR,
    RELIABILITY_THRESHOLD,
    TARGET_STEPS,
    TDVP_SUBSTEPS,
    apply_thread_limits,
)
from store import FrontierStore
from trajectory import (
    TrajectoryState,
    apply_trotter_step_dense,
    apply_trotter_step_mps,
    compute_metrics,
    initial_mps,
    initial_vector,
)


def _metrics_to_raw(row: dict[str, Any], *, source: str, tag: str) -> dict[str, Any]:
    peak_params = int(row.get("peak_param_count") or (int(row.get("peak_memory_bytes", 0)) // BYTES_PER_COMPLEX128))
    return {
        "method": row["method"],
        "chi_max": int(row["chi_max"]),
        "trotter_step": int(row["trotter_step"]),
        "time": float(row["time"]),
        "infidelity": float(row["infidelity"]),
        "state_norm": float(row["state_norm"]),
        "cumulative_runtime_s": float(row["cumulative_runtime_s"]),
        "step_runtime_s": float(row["step_runtime_s"]),
        "current_max_bond": int(row["current_max_bond"]),
        "peak_max_bond": int(row["peak_max_bond"]),
        "param_count": int(row["param_count"]),
        "peak_param_count": peak_params,
        "memory_bytes": int(row["memory_bytes"]),
        "peak_memory_bytes": int(row.get("peak_memory_bytes", peak_params * BYTES_PER_COMPLEX128)),
        "discarded_weight_step": float(row.get("discarded_weight_step", 0.0) or 0.0),
        "largest_intermediate_elements": int(row.get("largest_intermediate_elements", 0) or 0),
        "failed": int(row.get("failed", 0) or 0),
        "failure_message": str(row.get("failure_message", "") or ""),
        "converged": 1,
        "source": source,
        "tag": tag,
    }


def precompute_exact(*, timesteps: int, path: Path) -> np.ndarray:
    if path.exists():
        exact = np.load(path)
        if exact.shape[0] >= timesteps + 1:
            return exact
    vec = initial_vector("ising")
    schedule = build_ising_schedule(timesteps=timesteps)
    out = np.zeros((timesteps + 1, vec.size), dtype=np.complex128)
    out[0] = vec
    for i, step in enumerate(schedule, start=1):
        vec = apply_trotter_step_dense(vec, step)
        out[i] = vec
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out


def run_trajectory(
    *,
    method: str,
    chi: int,
    exact: np.ndarray,
    db_path: Path,
    tag: str = "main",
    source: str = "generated",
    stop_steps: int = TARGET_STEPS,
    stop_after_crossing: bool = True,
    warm_up: bool = False,
) -> list[dict[str, Any]]:
    """Run or resume one trajectory; algorithm timing excludes fidelity/IO."""
    apply_thread_limits()
    store = FrontierStore(db_path)
    try:
        existing = store.fetch_steps(method=method, chi=chi, tag=tag)
        if existing:
            crossed = any(float(r["infidelity"]) >= RELIABILITY_THRESHOLD for r in existing if int(float(r["trotter_step"])) > 0)
            max_s = max(int(float(r["trotter_step"])) for r in existing)
            if (stop_after_crossing and crossed) or max_s >= stop_steps:
                return existing

        schedule = build_ising_schedule(timesteps=stop_steps)
        st = TrajectoryState(mps=initial_mps("ising"), vec=initial_vector("ising"))

        if warm_up:
            warm = TrajectoryState(mps=initial_mps("ising"), vec=initial_vector("ising"))
            apply_trotter_step_mps(warm, schedule[0], method=method, chi=chi, tdvp_substeps=TDVP_SUBSTEPS)

        completed = store.max_step(method, chi, tag=tag)
        if completed < 0:
            # Step 0 baseline (not part of algorithm runtime).
            row = compute_metrics(
                exact[0],
                st.vec,
                state=st,
                model="ising",
                method=method,
                chi=chi,
                trotter_step=0,
                time=0.0,
                step_runtime_s=0.0,
            )
            store.insert_step(_metrics_to_raw(row, source=source, tag=tag))
            completed = 0

        for s in range(1, completed + 1):
            apply_trotter_step_mps(st, schedule[s - 1], method=method, chi=chi, tdvp_substeps=TDVP_SUBSTEPS)

        while completed < stop_steps:
            existing = store.fetch_steps(method=method, chi=chi, tag=tag)
            if stop_after_crossing and any(
                int(float(r["trotter_step"])) > 0 and float(r["infidelity"]) >= RELIABILITY_THRESHOLD for r in existing
            ):
                break

            next_step = completed + 1
            t0 = time.perf_counter()
            apply_trotter_step_mps(
                st,
                schedule[next_step - 1],
                method=method,
                chi=chi,
                tdvp_substeps=TDVP_SUBSTEPS,
                update_vec=False,
            )
            step_rt = time.perf_counter() - t0
            # Fidelity evaluation (dense vectorization + overlap) excluded from algorithm runtime.
            st.vec = st.mps.to_vec().astype(np.complex128, copy=False)
            row = compute_metrics(
                exact[next_step],
                st.vec,
                state=st,
                model="ising",
                method=method,
                chi=chi,
                trotter_step=next_step,
                time=next_step * DT,
                step_runtime_s=step_rt,
            )
            # Re-assert cumulative runtime from algorithm-only counter.
            row["cumulative_runtime_s"] = st.cumulative_runtime_s
            row["step_runtime_s"] = step_rt
            store.insert_step(_metrics_to_raw(row, source=source, tag=tag))
            completed = next_step
            print(
                f"  [{tag}] {method}/χ={chi} step={next_step} "
                f"t={next_step * DT:.1f} inf={row['infidelity']:.3e} "
                f"peakP={row['peak_param_count']} rt={step_rt:.2f}s",
                flush=True,
            )
            if st.failed:
                break
            if stop_after_crossing and float(row["infidelity"]) >= RELIABILITY_THRESHOLD:
                break

        return store.fetch_steps(method=method, chi=chi, tag=tag)
    finally:
        store.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resource-frontier single-trajectory worker.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--chi", type=int, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--tag", default="main")
    parser.add_argument("--source", default="generated")
    parser.add_argument("--stop-steps", type=int, default=TARGET_STEPS)
    parser.add_argument("--no-stop-after-crossing", action="store_true")
    parser.add_argument("--warm-up", action="store_true")
    parser.add_argument("--status-json", type=Path, default=None)
    args = parser.parse_args(argv)

    apply_thread_limits()
    exact = precompute_exact(timesteps=args.stop_steps, path=args.exact)
    rows = run_trajectory(
        method=args.method,
        chi=args.chi,
        exact=exact,
        db_path=args.db,
        tag=args.tag,
        source=args.source,
        stop_steps=args.stop_steps,
        stop_after_crossing=not args.no_stop_after_crossing,
        warm_up=args.warm_up,
    )
    status = {
        "method": args.method,
        "chi": args.chi,
        "tag": args.tag,
        "n_rows": len(rows),
        "max_step": max((int(float(r["trotter_step"])) for r in rows), default=-1),
        "max_infidelity": max((float(r["infidelity"]) for r in rows if int(float(r["trotter_step"])) > 0), default=0.0),
    }
    if args.status_json is not None:
        args.status_json.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status), flush=True)
    return 0


if __name__ == "__main__":
    # Ensure this package directory is importable when spawned as a script.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
