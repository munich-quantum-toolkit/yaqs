# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Fresh-process worker: one (χ, n_substeps) TFIM TDVP trajectory."""

from __future__ import annotations

import argparse
import json
import shutil
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
    RESOURCE_FRONTIER_OUTPUT,
    TARGET_STEPS,
    apply_thread_limits,
    cache_key,
    config_hash,
)
from store import ConvergenceStore
from trajectory import (
    TrajectoryState,
    apply_trotter_step_dense,
    apply_trotter_step_mps,
    compute_metrics,
    initial_mps,
    initial_vector,
)


def precompute_exact(*, timesteps: int, path: Path) -> np.ndarray:
    # Prefer shared resource-frontier cache when present and long enough.
    shared = RESOURCE_FRONTIER_OUTPUT / f"exact_ising_t{timesteps}.npy"
    if shared.exists() and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(shared, path)
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


def _vec_dir(key: str) -> Path:
    d = OUTPUT_DIR / "statevectors" / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_trajectory(
    *,
    chi: int,
    substeps: int,
    exact: np.ndarray,
    db_path: Path,
    stop_steps: int = TARGET_STEPS,
) -> list[dict[str, Any]]:
    apply_thread_limits()
    ch = config_hash()
    key = cache_key(chi=chi, substeps=substeps)
    store = ConvergenceStore(db_path)
    try:
        existing = store.fetch_steps(chi=chi, substeps=substeps, config_hash=ch)
        if existing and store.max_step(chi, substeps, ch) >= stop_steps:
            return existing

        schedule = build_ising_schedule(timesteps=stop_steps)
        st = TrajectoryState(mps=initial_mps("ising"), vec=initial_vector("ising"))
        completed = store.max_step(chi, substeps, ch)
        vdir = _vec_dir(key)

        if completed < 0:
            row = compute_metrics(
                exact[0],
                st.vec,
                state=st,
                model="ising",
                method="hybrid_tdvp",
                chi=chi,
                trotter_step=0,
                time=0.0,
                step_runtime_s=0.0,
            )
            store.insert_step(
                {
                    "chi_max": chi,
                    "tdvp_substeps": substeps,
                    "config_hash": ch,
                    "cache_key": key,
                    "trotter_step": 0,
                    "time": 0.0,
                    "infidelity": float(row["infidelity"]),
                    "state_norm": float(row["state_norm"]),
                    "peak_max_bond": int(row["peak_max_bond"]),
                    "peak_param_count": int(row["peak_param_count"]),
                    "param_count": int(row["param_count"]),
                    "cumulative_runtime_s": 0.0,
                    "step_runtime_s": 0.0,
                    "discarded_weight_step": 0.0,
                    "failed": 0,
                    "failure_message": "",
                    "krylov_failed": 0,
                }
            )
            np.save(vdir / "step_000.npy", st.vec.astype(np.complex128, copy=False))
            completed = 0

        for s in range(1, completed + 1):
            apply_trotter_step_mps(
                st,
                schedule[s - 1],
                method="hybrid_tdvp",
                chi=chi,
                tdvp_substeps=substeps,
                update_vec=False,
            )

        while completed < stop_steps:
            next_step = completed + 1
            t0 = time.perf_counter()
            apply_trotter_step_mps(
                st,
                schedule[next_step - 1],
                method="hybrid_tdvp",
                chi=chi,
                tdvp_substeps=substeps,
                update_vec=False,
            )
            step_rt = time.perf_counter() - t0
            st.vec = st.mps.to_vec().astype(np.complex128, copy=False)
            row = compute_metrics(
                exact[next_step],
                st.vec,
                state=st,
                model="ising",
                method="hybrid_tdvp",
                chi=chi,
                trotter_step=next_step,
                time=next_step * DT,
                step_runtime_s=step_rt,
            )
            krylov_failed = int(
                st.failed and ("krylov" in (st.failure_message or "").lower() or "lanczos" in (st.failure_message or "").lower() or "max" in (st.failure_message or "").lower())
            )
            store.insert_step(
                {
                    "chi_max": chi,
                    "tdvp_substeps": substeps,
                    "config_hash": ch,
                    "cache_key": key,
                    "trotter_step": next_step,
                    "time": next_step * DT,
                    "infidelity": float(row["infidelity"]),
                    "state_norm": float(row["state_norm"]),
                    "peak_max_bond": int(row["peak_max_bond"]),
                    "peak_param_count": int(row["peak_param_count"]),
                    "param_count": int(row["param_count"]),
                    "cumulative_runtime_s": float(st.cumulative_runtime_s),
                    "step_runtime_s": step_rt,
                    "discarded_weight_step": float(row.get("discarded_weight_step", 0.0) or 0.0),
                    "failed": int(st.failed),
                    "failure_message": st.failure_message or "",
                    "krylov_failed": krylov_failed,
                }
            )
            np.save(vdir / f"step_{next_step:03d}.npy", st.vec)
            completed = next_step
            print(
                f"  [χ={chi}, n={substeps}] step={next_step} "
                f"inf={row['infidelity']:.3e} peakP={row['peak_param_count']} "
                f"rt={step_rt:.2f}s failed={int(st.failed)}",
                flush=True,
            )
            if st.failed:
                break

        return store.fetch_steps(chi=chi, substeps=substeps, config_hash=ch)
    finally:
        store.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TDVP substep convergence worker.")
    parser.add_argument("--chi", type=int, required=True)
    parser.add_argument("--substeps", type=int, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--stop-steps", type=int, default=TARGET_STEPS)
    parser.add_argument("--status-json", type=Path, default=None)
    args = parser.parse_args(argv)

    apply_thread_limits()
    exact = precompute_exact(timesteps=args.stop_steps, path=args.exact)
    rows = run_trajectory(
        chi=args.chi,
        substeps=args.substeps,
        exact=exact,
        db_path=args.db,
        stop_steps=args.stop_steps,
    )
    status = {
        "chi": args.chi,
        "substeps": args.substeps,
        "config_hash": config_hash(),
        "cache_key": cache_key(chi=args.chi, substeps=args.substeps),
        "n_rows": len(rows),
        "max_step": max((int(float(r["trotter_step"])) for r in rows), default=-1),
        "bytes_per_complex128": BYTES_PER_COMPLEX128,
    }
    if args.status_json is not None:
        args.status_json.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status), flush=True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
