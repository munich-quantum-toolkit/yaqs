# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Deterministic-repeat helper: re-run Heisenberg chi=32 step 1 (all methods).

Must run as its own process so that BLAS threads are pinned before numpy is
first imported (the stored trajectories were generated the same way).
Prints a JSON dict {method: step1_infidelity}.
"""

from __future__ import annotations

from pb_common import RAW_DIR, add_experiment_path, limit_blas_threads

limit_blas_threads()
add_experiment_path("fixed_resources")


import json  # ruff: ignore[module-import-not-at-top-of-file]
import sys  # ruff: ignore[module-import-not-at-top-of-file]

import numpy as np  # ruff: ignore[module-import-not-at-top-of-file]


def main() -> int:
    from generate_corrected import run_trajectory

    exact = np.load(RAW_DIR / "circuits_corrected" / "exact_heisenberg_t30.npy")
    out = {}
    for m in ("hybrid_tdvp", "tebd_swap", "mpo_zipup"):
        rows = run_trajectory(
            model="heisenberg", method=m, chi=32, timesteps=1, exact=exact,
            tdvp_substeps=2, stop_after_crossing=False,
        )
        out[m] = float(rows[1]["infidelity"])
    sys.stdout.write(json.dumps(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
