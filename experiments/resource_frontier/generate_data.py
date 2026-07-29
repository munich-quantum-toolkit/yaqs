# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Generate / ingest TFIM trajectories for the resource frontier."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import path_setup  # noqa: F401
from config import (
    BYTES_PER_COMPLEX128,
    CHI_HIGH,
    CHI_INGEST,
    DT,
    FIXED_RESOURCES_OUTPUT,
    METHODS,
    OUTPUT_DIR,
    RELIABILITY_THRESHOLD,
    TARGET_STEPS,
    TIMING_NEAR_OPTIMAL_FRAC,
    TIMING_REPEATS,
    apply_thread_limits,
    production_config,
)
from store import FrontierStore, RAW_COLUMNS, save_json, write_csv
from worker_run import precompute_exact


def _python() -> str:
    return sys.executable


def _spawn_worker(
    *,
    method: str,
    chi: int,
    db: Path,
    exact: Path,
    tag: str,
    source: str,
    stop_steps: int = TARGET_STEPS,
    stop_after_crossing: bool = True,
    warm_up: bool = False,
) -> dict[str, Any]:
    """Run one trajectory in a fresh interpreter process (no worker pool)."""
    worker = Path(__file__).resolve().parent / "worker_run.py"
    status_json = OUTPUT_DIR / f"status_{method}_chi{chi}_{tag}.json"
    cmd = [
        _python(),
        str(worker),
        "--method",
        method,
        "--chi",
        str(chi),
        "--db",
        str(db),
        "--exact",
        str(exact),
        "--tag",
        tag,
        "--source",
        source,
        "--stop-steps",
        str(stop_steps),
        "--status-json",
        str(status_json),
    ]
    if not stop_after_crossing:
        cmd.append("--no-stop-after-crossing")
    if warm_up:
        cmd.append("--warm-up")
    print(f"SPAWN {method}/χ={chi} tag={tag}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, check=False, cwd=str(worker.parent))
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Worker failed for {method}/χ={chi} tag={tag} (rc={proc.returncode})")
    status: dict[str, Any] = {"method": method, "chi": chi, "tag": tag, "wall_s": elapsed}
    if status_json.exists():
        import json

        status.update(json.loads(status_json.read_text(encoding="utf-8")))
    return status


def _reliable_through(rows: list[dict[str, Any]], n: int) -> bool:
    by_step = {int(float(r["trotter_step"])): r for r in rows}
    for k in range(1, n + 1):
        if k not in by_step:
            return False
        if int(float(by_step[k].get("failed", 0) or 0)):
            return False
        if float(by_step[k]["infidelity"]) >= RELIABILITY_THRESHOLD:
            return False
    return True


def _max_reliable_step(rows: list[dict[str, Any]]) -> int:
    best = 0
    for n in range(1, TARGET_STEPS + 1):
        if _reliable_through(rows, n):
            best = n
        else:
            break
    return best


def ingest_fixed_resources_chi32(store: FrontierStore) -> list[str]:
    """Load χ=32 TFIM rows from fixed_resources for cross-check (tag=ingest_ref).

    Main frontier runs are regenerated so peak MPS parameters are measured after every gate.
    Existing fixed_resources files are never modified.
    """
    path = FIXED_RESOURCES_OUTPUT / "trajectories.csv"
    notes: list[str] = []
    if not path.exists():
        notes.append(f"Missing {path}; no χ=32 ingest cross-check.")
        return notes
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    n_in = 0
    for r in rows:
        if r.get("model") != "ising":
            continue
        if int(float(r["chi_max"])) != 32:
            continue
        method = r["method"]
        if method not in METHODS:
            continue
        step = int(float(r["trotter_step"]))
        if step > TARGET_STEPS:
            continue
        params = int(float(r["param_count"]))
        store.insert_step(
            {
                "method": method,
                "chi_max": 32,
                "trotter_step": step,
                "time": float(r["time"]),
                "infidelity": float(r["infidelity"]),
                "state_norm": float(r["state_norm"]),
                "cumulative_runtime_s": float(r["cumulative_runtime_s"]),
                "step_runtime_s": float(r["step_runtime_s"]),
                "current_max_bond": int(float(r["current_max_bond"])),
                "peak_max_bond": int(float(r["peak_max_bond"])),
                "param_count": params,
                "peak_param_count": params,
                "memory_bytes": int(float(r["memory_bytes"])),
                "peak_memory_bytes": params * BYTES_PER_COMPLEX128,
                "discarded_weight_step": float(r.get("discarded_weight_step") or 0.0),
                "largest_intermediate_elements": 0,
                "failed": int(float(r.get("failed") or 0)),
                "failure_message": r.get("failure_message") or "",
                "converged": 1,
                "source": "fixed_resources/trajectories.csv",
                "tag": "ingest_ref",
            }
        )
        n_in += 1
    notes.append(
        f"Cross-check ingest: {n_in} χ=32 TFIM rows from {path.name} → tag=ingest_ref "
        "(not used for frontier peaks; main runs regenerated for gate-level peak accounting)."
    )
    notes.append(
        "Other χ∈CHI_INGEST had no retained per-step files after fixed_resources cleanup; "
        "regenerated with matching 4×4 TFIM configuration."
    )
    return notes


def _trajectory_complete(store: FrontierStore, method: str, chi: int) -> bool:
    rows = store.fetch_steps(method=method, chi=chi, tag="main")
    if not rows:
        return False
    if _max_reliable_step(rows) >= TARGET_STEPS:
        return True
    # Completed if crossed or reached target steps.
    max_s = max(int(float(r["trotter_step"])) for r in rows)
    crossed = any(int(float(r["trotter_step"])) > 0 and float(r["infidelity"]) >= RELIABILITY_THRESHOLD for r in rows)
    return crossed or max_s >= TARGET_STEPS


def generate_main_runs(*, resume: bool = True) -> dict[str, Any]:
    apply_thread_limits()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    db = OUTPUT_DIR / "raw_runs.sqlite"
    exact_path = OUTPUT_DIR / f"exact_ising_t{TARGET_STEPS}.npy"
    store = FrontierStore(db)
    notes = ingest_fixed_resources_chi32(store)
    store.set_meta("ingest_notes", " | ".join(notes))
    store.close()

    precompute_exact(timesteps=TARGET_STEPS, path=exact_path)

    spawned: list[dict[str, Any]] = []
    # Ingest ladder: all methods × CHI_INGEST (skip complete).
    for method in METHODS:
        for chi in CHI_INGEST:
            store = FrontierStore(db)
            done = resume and _trajectory_complete(store, method, chi)
            store.close()
            if done:
                print(f"SKIP complete {method}/χ={chi}", flush=True)
                continue
            spawned.append(
                _spawn_worker(
                    method=method,
                    chi=chi,
                    db=db,
                    exact=exact_path,
                    tag="main",
                    source="generated",
                )
            )

    # Adaptive high-χ for TEBD/MPO only.
    high_spawned: list[dict[str, Any]] = []
    for method in ("tebd_swap", "mpo_zipup"):
        store = FrontierStore(db)
        rows = store.fetch_steps(method=method, tag="main")
        store.close()
        # If any existing χ already reaches step 15, skip further high χ.
        best = 0
        for chi in CHI_INGEST:
            best = max(best, _max_reliable_step([r for r in rows if int(float(r["chi_max"])) == chi]))
        if best >= TARGET_STEPS:
            print(f"SKIP high-χ {method}: already reliable through n={TARGET_STEPS}", flush=True)
            continue
        for chi in CHI_HIGH:
            store = FrontierStore(db)
            rows_chi = store.fetch_steps(method=method, chi=chi, tag="main")
            done = resume and _trajectory_complete(store, method, chi)
            store.close()
            if done:
                print(f"SKIP complete {method}/χ={chi}", flush=True)
            else:
                high_spawned.append(
                    _spawn_worker(
                        method=method,
                        chi=chi,
                        db=db,
                        exact=exact_path,
                        tag="main",
                        source="generated_high_chi",
                    )
                )
            store = FrontierStore(db)
            rows_chi = store.fetch_steps(method=method, chi=chi, tag="main")
            store.close()
            if _max_reliable_step(rows_chi) >= TARGET_STEPS:
                print(f"STOP high-χ {method} after χ={chi} (reliable through n={TARGET_STEPS})", flush=True)
                break

    store = FrontierStore(db)
    all_rows = store.fetch_steps(tag="main")
    store.close()
    write_csv(OUTPUT_DIR / "raw_runs.csv", all_rows, fieldnames=list(RAW_COLUMNS))
    save_json(OUTPUT_DIR / "config.json", production_config())
    return {
        "notes": notes,
        "spawned": spawned,
        "high_spawned": high_spawned,
        "n_raw_rows": len(all_rows),
    }


def run_timing_repeats(candidate_cfgs: list[tuple[str, int]]) -> list[dict[str, Any]]:
    """Controlled median timing for runtime-frontier candidates."""
    apply_thread_limits()
    db = OUTPUT_DIR / "raw_runs.sqlite"
    exact_path = OUTPUT_DIR / f"exact_ising_t{TARGET_STEPS}.npy"
    precompute_exact(timesteps=TARGET_STEPS, path=exact_path)
    records: list[dict[str, Any]] = []
    for method, chi in candidate_cfgs:
        for rep in range(TIMING_REPEATS):
            tag = f"timing_r{rep}"
            # Clear previous timing tag rows for clean remeasure? Keep resume within repeat.
            _spawn_worker(
                method=method,
                chi=chi,
                db=db,
                exact=exact_path,
                tag=tag,
                source="timing_repeat",
                stop_after_crossing=True,
                warm_up=(rep == 0),
            )
            store = FrontierStore(db)
            rows = store.fetch_steps(method=method, chi=chi, tag=tag)
            store.close()
            for r in rows:
                records.append(
                    {
                        "method": method,
                        "chi_max": chi,
                        "repeat": rep,
                        "trotter_step": int(float(r["trotter_step"])),
                        "time": float(r["time"]),
                        "infidelity": float(r["infidelity"]),
                        "cumulative_runtime_s": float(r["cumulative_runtime_s"]),
                        "peak_param_count": int(float(r["peak_param_count"])),
                        "failed": int(float(r.get("failed", 0) or 0)),
                    }
                )
    write_csv(OUTPUT_DIR / "timing_repeats.csv", records)
    return records


def select_timing_candidates(raw_rows: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """Configs that are runtime-optimal or within 20% for any target step."""
    from build_frontier import preliminary_runtime_mins

    mins = preliminary_runtime_mins(raw_rows)
    selected: set[tuple[str, int]] = set()
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in raw_rows:
        key = (str(r["method"]), int(float(r["chi_max"])))
        by_key.setdefault(key, []).append(r)

    for n, per_method in mins.items():
        for method, info in per_method.items():
            if info is None:
                continue
            selected.add((method, int(info["chi_max"])))
            best_rt = float(info["runtime_s"])
            for (m, chi), rows in by_key.items():
                if m != method:
                    continue
                if not _reliable_through(rows, n):
                    continue
                row_n = next(r for r in rows if int(float(r["trotter_step"])) == n)
                rt = float(row_n["cumulative_runtime_s"])
                if rt <= best_rt * (1.0 + TIMING_NEAR_OPTIMAL_FRAC):
                    selected.add((m, chi))
    return sorted(selected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate resource-frontier TFIM data.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-timing", action="store_true")
    args = parser.parse_args(argv)
    summary = generate_main_runs(resume=not args.no_resume)
    print(f"Main runs complete: {summary['n_raw_rows']} rows", flush=True)
    if not args.skip_timing:
        store = FrontierStore(OUTPUT_DIR / "raw_runs.sqlite")
        raw = store.fetch_steps(tag="main")
        store.close()
        cands = select_timing_candidates(raw)
        print(f"Timing candidates: {cands}", flush=True)
        run_timing_repeats(cands)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
