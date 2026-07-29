# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Supplementary RSS working-memory validation for frontier configs."""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

import path_setup  # noqa: F401
from build_frontier import _is_present, build_memory_frontier, largest_common_reliable_step
from circuits import build_ising_schedule
from config import BYTES_PER_COMPLEX128, MIB, OUTPUT_DIR, TARGET_STEPS, TDVP_SUBSTEPS, apply_thread_limits
from store import FrontierStore, write_csv
from trajectory import TrajectoryState, apply_trotter_step_mps, initial_mps, initial_vector
from worker_run import precompute_exact


def _rss_bytes() -> int:
    # Linux: ru_maxrss is kilobytes.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def run_one(*, method: str, chi: int, n_steps: int, exact: np.ndarray) -> dict[str, Any]:
    apply_thread_limits()
    # Load reference before baseline.
    _ = exact[0]
    baseline = _rss_bytes()
    schedule = build_ising_schedule(timesteps=n_steps)
    st = TrajectoryState(mps=initial_mps("ising"), vec=initial_vector("ising"))
    peak_rss = baseline
    for i in range(n_steps):
        apply_trotter_step_mps(
            st, schedule[i], method=method, chi=chi, tdvp_substeps=TDVP_SUBSTEPS, update_vec=False
        )
        peak_rss = max(peak_rss, _rss_bytes())
    return {
        "method": method,
        "chi_max": chi,
        "target_step": n_steps,
        "baseline_rss_bytes": baseline,
        "peak_rss_bytes": peak_rss,
        "peak_rss_increase_bytes": peak_rss - baseline,
        "peak_rss_increase_MiB": (peak_rss - baseline) / MIB,
        "peak_param_count": st.peak_param_count,
        "peak_mps_storage_bytes": st.peak_param_count * BYTES_PER_COMPLEX128,
        "peak_mps_storage_MiB": (st.peak_param_count * BYTES_PER_COMPLEX128) / MIB,
        "largest_intermediate_elements": st.peak_intermediate_elements,
        "ordering_note": "",
        "common_target_step": n_steps,
    }


def _spawn_one(*, method: str, chi: int, n_steps: int, exact_path: Path) -> dict[str, Any]:
    """Fresh interpreter process per configuration."""
    script = Path(__file__).resolve()
    out_json = OUTPUT_DIR / f"working_memory_{method}_chi{chi}.json"
    cmd = [
        sys.executable,
        str(script),
        "--worker",
        "--method",
        method,
        "--chi",
        str(chi),
        "--n-steps",
        str(n_steps),
        "--exact",
        str(exact_path),
        "--out-json",
        str(out_json),
    ]
    print(f"Working-memory check {method}/χ={chi} through n={n_steps}", flush=True)
    proc = subprocess.run(cmd, check=False, cwd=str(script.parent))
    if proc.returncode != 0 or not out_json.exists():
        raise RuntimeError(f"Working-memory worker failed for {method}/χ={chi}")
    return json.loads(out_json.read_text(encoding="utf-8"))


def run_validation() -> list[dict[str, Any]]:
    store = FrontierStore(OUTPUT_DIR / "raw_runs.sqlite")
    raw = store.fetch_steps(tag="main")
    store.close()
    mem = build_memory_frontier(raw)
    n_star = largest_common_reliable_step(mem)
    if n_star <= 0:
        raise RuntimeError("No common reliable target step for working-memory check.")

    exact_path = OUTPUT_DIR / f"exact_ising_t{TARGET_STEPS}.npy"
    precompute_exact(timesteps=TARGET_STEPS, path=exact_path)

    rows: list[dict[str, Any]] = []
    for method in ("hybrid_tdvp", "tebd_swap", "mpo_zipup"):
        sel = next(
            (
                r
                for r in mem
                if r["method"] == method
                and int(float(r["target_step"])) == n_star
                and _is_present(r)
            ),
            None,
        )
        if sel is None:
            continue
        chi = int(float(sel["chi_max"]))
        rows.append(_spawn_one(method=method, chi=chi, n_steps=n_star, exact_path=exact_path))

    if rows:
        by_mps = sorted(rows, key=lambda r: float(r["peak_mps_storage_MiB"]))
        by_rss = sorted(rows, key=lambda r: float(r["peak_rss_increase_MiB"]))
        mps_order = [r["method"] for r in by_mps]
        rss_order = [r["method"] for r in by_rss]
        note = (
            "agree"
            if mps_order == rss_order
            else (
                f"DISAGREE: MPS order {mps_order} vs RSS order {rss_order}; "
                "do not claim overall peak-memory advantage from MPS parameters alone."
            )
        )
        for r in rows:
            r["ordering_note"] = note
            r["common_target_step"] = n_star
    write_csv(OUTPUT_DIR / "working_memory_validation.csv", rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Working-memory RSS validation.")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--method", default="")
    parser.add_argument("--chi", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=0)
    parser.add_argument("--exact", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.worker:
        apply_thread_limits()
        exact = np.load(args.exact)
        row = run_one(method=args.method, chi=args.chi, n_steps=args.n_steps, exact=exact)
        args.out_json.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(row), flush=True)
        return 0

    rows = run_validation()
    print(f"Wrote {len(rows)} working-memory rows")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
