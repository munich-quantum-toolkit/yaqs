# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Stage 3b (generate missing data only): full Heisenberg chi=32 trajectories.

The corrected circuit campaign early-stopped Heisenberg chi=32 trajectories
one step after crossing epsilon=1e-2 (TDVP/TEBD: step 1; zip-up: step 3).
Figure 3(b) plots infidelity versus physical time, so this stage re-runs the
identical corrected configuration (same schedule builders, initial states,
dt=0.1, TDVP n=2 on long-range gates only, SVD discarded_weight @ 1e-13)
for the full 30 Trotter steps without early stopping. The stored corrected
exact reference exact_heisenberg_t30.npy is reused unmodified.

Checkpointing: one CSV per method under raw_new/heisenberg_chi32_full/;
a completed method (31 rows) is skipped on resume.

Usage:
    uv run python paper_benchmarks/scripts/generate_heisenberg_traj.py
"""

from __future__ import annotations

import csv
import time

from pb_common import (
    CIRCUIT_CHI_MAIN,
    CIRCUIT_TDVP_SUBSTEPS,
    CIRCUIT_TIMESTEPS,
    LOGS_DIR,
    RAW_DIR,
    RAW_NEW_DIR,
    add_experiment_path,
    limit_blas_threads,
)

limit_blas_threads()
add_experiment_path("fixed_resources")

import numpy as np  # ruff: ignore[module-import-not-at-top-of-file]

OUT_DIR = RAW_NEW_DIR / "heisenberg_chi32_full"
METHODS = ("hybrid_tdvp", "tebd_swap", "mpo_zipup")


def main() -> int:
    from generate_corrected import CSV_FIELDS, run_trajectory

    exact = np.load(RAW_DIR / "circuits_corrected" / "exact_heisenberg_t30.npy")
    assert exact.shape[0] >= CIRCUIT_TIMESTEPS + 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log = (LOGS_DIR / "generate_heisenberg_traj.log").open("a", encoding="utf-8", buffering=1)

    exit_code = 0
    for method in METHODS:
        out = OUT_DIR / f"heisenberg_chi{CIRCUIT_CHI_MAIN}_{method}.csv"
        if out.exists():
            with out.open(encoding="utf-8") as fh:
                nrows = sum(1 for _ in fh) - 1
            if nrows >= CIRCUIT_TIMESTEPS + 1:
                print(f"skip {method}: complete ({nrows} rows)")
                continue
            out.unlink()  # partial file: regenerate deterministically
        t0 = time.perf_counter()
        print(f"running heisenberg chi={CIRCUIT_CHI_MAIN} {method} "
              f"({CIRCUIT_TIMESTEPS} steps, n={CIRCUIT_TDVP_SUBSTEPS}) ...", flush=True)
        try:
            rows = run_trajectory(
                model="heisenberg",
                method=method,
                chi=CIRCUIT_CHI_MAIN,
                timesteps=CIRCUIT_TIMESTEPS,
                exact=exact,
                tdvp_substeps=CIRCUIT_TDVP_SUBSTEPS,
                stop_after_crossing=False,
            )
        except Exception as exc:
            exit_code = 1
            log.write(f"METHOD FAILED {method}: {exc!r}\n")
            print(f"METHOD FAILED {method}: {exc!r}")
            continue
        tmp = out.with_suffix(".csv.tmp")
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        tmp.rename(out)
        wall = time.perf_counter() - t0
        msg = f"{method}: {len(rows)} rows in {wall:.1f}s -> {out.name}"
        print(msg, flush=True)
        log.write(msg + "\n")
    log.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
