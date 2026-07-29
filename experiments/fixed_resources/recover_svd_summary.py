# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Recover SVD diagnostic summary from a completed run log after CSV write failure."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "output_svd_diagnostic"
LOG = Path.home() / ".cursor/projects/home-aaron-Github-yaqs/terminals/631555.txt"


def main() -> int:
    OUT.mkdir(exist_ok=True)
    log = LOG.read_text(encoding="utf-8")
    pat = re.compile(
        r"=== (\S+) χ=(\d+) τ=([0-9eE.+-]+) ===\n"
        r"  Tε=([0-9.]+) peakχ=(\d+) params=(\d+) f_cut=([0-9.]+) f_chi=([0-9.]+)"
    )
    rows = []
    for m in pat.finditer(log):
        method, chi, tau, te, peak, params, fc, fx = m.groups()
        rows.append(
            {
                "method": method,
                "chi_max": int(chi),
                "tau": float(tau),
                "T_eps": float(te),
                "n_eps": int(round(float(te) / 0.1)),
                "crossed": 1,
                "right_censored": 0,
                "peak_actual_chi": int(peak),
                "peak_param_count": int(params),
                "runtime_s": "",
                "total_discarded_weight": "",
                "n_truncation_events": "",
                "fraction_cutoff_limited": float(fc),
                "fraction_chi_limited": float(fx),
                "final_infidelity": "",
                "source": "parsed_from_completed_run_log",
            }
        )
    print(f"parsed {len(rows)} summary rows")
    if len(rows) != 30:
        raise SystemExit(f"expected 30 rows, got {len(rows)}")

    with (OUT / "svd_cutoff_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    note = (
        "All 30 (method, χ, τ) runs completed successfully. "
        "Writing trajectories/events crashed on a DictWriter field mismatch; "
        "summary was recovered from the run log. Event-level CSVs and spectra "
        "are stubs — re-run `run_svd_cutoff_diagnostic.py` only if needed."
    )
    (OUT / "DATA_NOTE.md").write_text(note + "\n", encoding="utf-8")
    for name in ("svd_cutoff_trajectories.csv", "svd_truncation_events.csv"):
        with (OUT / name).open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["note"])
            writer.writeheader()
            writer.writerow({"note": note})
    np.savez_compressed(OUT / "representative_spectra.npz")
    sem = {
        "production_trunc_mode": "discarded_weight",
        "tau_meaning": (
            "Cumulative discarded squared singular-value weight; "
            "retained_rank=min(keep_cutoff, chi_max), min_keep=1."
        ),
        "gate_library_split_tensor_hard_cutoff": 1e-14,
        "corrected_benchmark_svd_threshold": 1e-13,
        "diagnostic_reference_tau": 1e-14,
        "reuse_1e-14_from_corrected": False,
    }
    (OUT / "cutoff_semantics.json").write_text(json.dumps(sem, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT / 'svd_cutoff_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
