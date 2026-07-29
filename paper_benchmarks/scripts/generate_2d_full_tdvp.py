# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Generate 2D (4x4) circuit trajectories with full_tdvp at chi=32.

Unlike the corrected-campaign hybrid_tdvp method (TEBD for nearest-neighbour
gates, gate-local TDVP only for long-range gates), full_tdvp routes every
two-qubit gate through the gate-local TDVP window update
(gate_mode="full-tdvp", n=2 fractional-time substeps).

Reuses the corrected dense exact references unmodified.

Usage:
    uv run python paper_benchmarks/scripts/generate_2d_full_tdvp.py
"""

from __future__ import annotations

import csv
import time
from concurrent.futures import ProcessPoolExecutor

from pb_common import (
    CIRCUIT_CHI_MAIN,
    CIRCUIT_TDVP_SUBSTEPS,
    CIRCUIT_TIMESTEPS,
    LOGS_DIR,
    RAW_DIR,
    RAW_NEW_DIR,
    add_experiment_path,
    limit_blas_threads,
    worker_count,
)

limit_blas_threads()
add_experiment_path("fixed_resources")

import numpy as np  # noqa: E402

OUT_DIR = RAW_NEW_DIR / "circuits_2d_full_tdvp"
MODELS = ("ising", "heisenberg")
METHOD = "full_tdvp"


def _job(model: str) -> str:
    from generate_corrected import CSV_FIELDS, run_trajectory

    out = OUT_DIR / f"{model}_chi{CIRCUIT_CHI_MAIN}_{METHOD}.csv"
    if out.exists():
        with out.open(encoding="utf-8") as fh:
            nrows = sum(1 for _ in fh) - 1
        if nrows >= CIRCUIT_TIMESTEPS + 1:
            return f"skip {model} {METHOD}: complete ({nrows} rows)"
        out.unlink()

    exact_path = RAW_DIR / "circuits_corrected" / f"exact_{model}_t{CIRCUIT_TIMESTEPS}.npy"
    exact = np.load(exact_path)
    t0 = time.perf_counter()
    rows = run_trajectory(
        model=model,
        method=METHOD,
        chi=CIRCUIT_CHI_MAIN,
        timesteps=CIRCUIT_TIMESTEPS,
        exact=exact,
        tdvp_substeps=CIRCUIT_TDVP_SUBSTEPS,
        stop_after_crossing=False,
    )
    tmp = out.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.rename(out)
    wall = time.perf_counter() - t0
    return f"{model} {METHOD}: {len(rows)} rows in {wall:.1f}s -> {out.name}"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log = (LOGS_DIR / "generate_2d_full_tdvp.log").open("a", encoding="utf-8", buffering=1)
    print(
        f"=== 2D full_tdvp trajectories (chi={CIRCUIT_CHI_MAIN}, "
        f"{CIRCUIT_TIMESTEPS} steps, n={CIRCUIT_TDVP_SUBSTEPS} on ALL 2q gates) ===",
        flush=True,
    )
    exit_code = 0
    with ProcessPoolExecutor(max_workers=min(worker_count(2), len(MODELS))) as pool:
        futures = {pool.submit(_job, m): m for m in MODELS}
        for fut, model in futures.items():
            try:
                msg = fut.result()
            except Exception as exc:  # noqa: BLE001
                exit_code = 1
                msg = f"JOB FAILED {model}: {exc!r}"
            print(msg, flush=True)
            log.write(msg + "\n")
    log.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
