# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""SVD-cutoff diagnostic for the corrected 4×4 TFIM circuit benchmark.

Production truncation semantics (held fixed as the mode; only the numeric
threshold τ is swept):

* ``trunc_mode = "discarded_weight"``
* ``τ = svd_threshold`` = cumulative discarded squared singular-value weight
  (sum of ``s_i²`` of discarded singular values, compared against τ; at least
  ``min_keep=1`` retained; then capped by ``χmax``).
* Retained rank: ``min(rank_allowed_by_cutoff, χmax)`` with ≥1 singular value.

This is **not** a relative ``s_i/s_max`` cutoff and **not** the gate-library
``hard_cutoff`` used only when building MPO tensors from gates
(``split_tensor``, fixed at ``1e-14`` and not swept here).

The corrected production benchmark used ``svd_threshold=1e-13``. The diagnostic
reference τ=1e-14 is therefore re-run (configuration is not identical).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from circuits import build_ising_schedule
from config import (
    CORRECTED_OUTPUT_DIR,
    DT,
    PACKAGE_DIR,
    RELIABILITY_THRESHOLD,
    TMAX_INITIAL,
    timesteps_for_tmax,
)
from horizons import reliable_horizon
from svd_instrumentation import SVDDiagnosticTracker, track_svd_events
from trajectory import (
    TrajectoryState,
    apply_gate_mps,
    compute_metrics,
    initial_mps,
    initial_vector,
)

OUTPUT_DIR = PACKAGE_DIR / "output_svd_diagnostic"
CUTOFFS = (1e-14, 1e-12, 1e-9, 1e-6, 1e-3)
CHI_VALUES = (16, 32)
METHODS = ("hybrid_tdvp", "tebd_swap", "mpo_zipup")
TDVP_N = 2
METHOD_LABELS = {
    "hybrid_tdvp": "TDVP",
    "tebd_swap": "TEBD+SWAP",
    "mpo_zipup": "MPO zip-up",
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_one(
    *,
    method: str,
    chi: int,
    tau: float,
    exact: np.ndarray,
    timesteps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    schedule = build_ising_schedule(timesteps=timesteps)
    st = TrajectoryState(mps=copy.deepcopy(initial_mps("ising")), vec=initial_vector("ising").copy())
    tracker = SVDDiagnosticTracker()
    traj_rows: list[dict[str, Any]] = []

    row0 = compute_metrics(
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
    row0["tdvp_substeps"] = TDVP_N
    row0["svd_threshold"] = tau
    row0["trunc_mode"] = "discarded_weight"
    traj_rows.append(row0)

    first_lr_captured = False
    for step_idx, step in enumerate(schedule, start=1):
        t_phys = step_idx * DT
        # Request per-step spectrum snapshot (first truncate in this step).
        tracker.request_spectrum(f"step_{step_idx}")
        if abs(t_phys - 1.0) < 1e-12:
            tracker.request_spectrum("near_t1")
        t_before = st.cumulative_runtime_s
        st.step_discarded = 0.0
        with track_svd_events(tracker):
            for g_idx, gate in enumerate(step.gates):
                is_lr = len(gate.qubits) == 2 and abs(gate.qubits[0] - gate.qubits[1]) != 1
                if is_lr and not first_lr_captured:
                    tracker.request_spectrum("first_long_range")
                    first_lr_captured = True
                tracker.set_context(
                    method=method,
                    threshold=tau,
                    chi_max=chi,
                    trotter_step=step_idx,
                    time=t_phys,
                    gate_name=gate.name,
                    gate_qubits=gate.qubits,
                    gate_index=g_idx,
                    is_long_range=is_lr,
                    trunc_mode="discarded_weight",
                )
                apply_gate_mps(
                    st,
                    gate,
                    method=method,
                    chi=chi,
                    tdvp_substeps=TDVP_N,
                    svd_threshold=tau,
                )
        st.vec = st.mps.to_vec().astype(np.complex128, copy=False)
        step_runtime = st.cumulative_runtime_s - t_before
        row = compute_metrics(
            exact[step_idx],
            st.vec,
            state=st,
            model="ising",
            method=method,
            chi=chi,
            trotter_step=step_idx,
            time=t_phys,
            step_runtime_s=step_runtime,
        )
        row["tdvp_substeps"] = TDVP_N
        row["svd_threshold"] = tau
        row["trunc_mode"] = "discarded_weight"
        # Cumulative discarded weight from instrumented events this run.
        row["cumulative_discarded_weight_events"] = float(
            sum(e.discarded_weight for e in tracker.events)
        )
        traj_rows.append(row)
        if st.failed:
            break

    event_rows = [asdict(e) for e in tracker.events]
    spectra = dict(tracker.spectra)
    return traj_rows, event_rows, spectra


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    out: Path = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    semantics = {
        "production_trunc_mode": "discarded_weight",
        "tau_meaning": (
            "Cumulative discarded squared singular-value weight: discard the "
            "smallest singular values until sum_{discarded} s_i^2 >= τ, then "
            "apply retained_rank = min(keep_cutoff, χmax) with min_keep=1."
        ),
        "not_relative_s_over_smax": True,
        "not_absolute_hard_cutoff_for_mps": True,
        "gate_library_split_tensor_hard_cutoff": 1e-14,
        "gate_library_note": (
            "MPO construction from gates still uses hard_cutoff=1e-14 in "
            "split_tensor; that cutoff is held fixed and is not the swept τ."
        ),
        "corrected_benchmark_svd_threshold": 1e-13,
        "diagnostic_reference_tau": 1e-14,
        "reuse_1e-14_from_corrected": False,
        "reuse_reason": "Corrected production used svd_threshold=1e-13, not 1e-14.",
        "held_fixed": {
            "krylov_tol": 1e-12,
            "tdvp_substeps": TDVP_N,
            "tdvp_mode": "2site",
            "trunc_mode": "discarded_weight",
            "dt": DT,
            "model": "ising_4x4",
        },
        "cutoffs": list(CUTOFFS),
        "chi_values": list(CHI_VALUES),
        "methods": list(METHODS),
    }
    (out / "cutoff_semantics.json").write_text(json.dumps(semantics, indent=2) + "\n", encoding="utf-8")

    timesteps = timesteps_for_tmax(TMAX_INITIAL)
    exact_path = CORRECTED_OUTPUT_DIR / f"exact_ising_t{timesteps}.npy"
    if not exact_path.exists():
        raise SystemExit(f"Missing exact reference {exact_path}; run generate_corrected first.")
    exact = np.load(exact_path)
    if exact.shape[0] < timesteps + 1:
        raise SystemExit("Exact reference too short.")

    all_traj: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    all_spectra: dict[str, np.ndarray] = {}
    summary: list[dict[str, Any]] = []

    for chi in CHI_VALUES:
        for method in METHODS:
            for tau in CUTOFFS:
                print(f"=== {method} χ={chi} τ={tau:g} ===", flush=True)
                traj, events, spectra = run_one(
                    method=method, chi=chi, tau=tau, exact=exact, timesteps=timesteps
                )
                for r in traj:
                    r["chi_max"] = chi
                for e in events:
                    e["chi_max"] = chi
                all_traj.extend(traj)
                all_events.extend(events)

                h = reliable_horizon(traj, epsilon=RELIABILITY_THRESHOLD, dt=DT)
                n_eps = int(h["n_eps"])
                if chi == 32:
                    prefix = f"{method}_chi{chi}_tau{tau:g}"
                    if "first_long_range" in spectra:
                        all_spectra[f"{prefix}_first_long_range"] = spectra["first_long_range"]
                    if "near_t1" in spectra:
                        all_spectra[f"{prefix}_near_t1"] = spectra["near_t1"]
                    # Last reliable step spectrum.
                    key = f"step_{n_eps}" if n_eps > 0 else "step_1"
                    if key in spectra:
                        all_spectra[f"{prefix}_last_reliable"] = spectra[key]
                    elif "step_1" in spectra:
                        all_spectra[f"{prefix}_last_reliable"] = spectra["step_1"]

                n_ev = max(len(events), 1)
                n_cut = sum(1 for e in events if e["limiter"] == "cutoff")
                n_chi = sum(1 for e in events if e["limiter"] == "chi_max")
                summary.append(
                    {
                        "method": method,
                        "chi_max": chi,
                        "tau": tau,
                        "T_eps": h["T_eps"],
                        "n_eps": n_eps,
                        "crossed": h["crossed"],
                        "right_censored": h["right_censored"],
                        "peak_actual_chi": max(int(r["peak_max_bond"]) for r in traj),
                        "peak_param_count": max(int(r["peak_param_count"]) for r in traj),
                        "runtime_s": max(float(r["cumulative_runtime_s"]) for r in traj),
                        "total_discarded_weight": float(sum(e["discarded_weight"] for e in events)),
                        "n_truncation_events": len(events),
                        "fraction_cutoff_limited": n_cut / n_ev,
                        "fraction_chi_limited": n_chi / n_ev,
                        "final_infidelity": float(traj[-1]["infidelity"]),
                    }
                )
                print(
                    f"  Tε={h['T_eps']:.2f} peakχ={summary[-1]['peak_actual_chi']} "
                    f"params={summary[-1]['peak_param_count']} "
                    f"f_cut={summary[-1]['fraction_cutoff_limited']:.2f} "
                    f"f_chi={summary[-1]['fraction_chi_limited']:.2f}",
                    flush=True,
                )

    _write_csv(out / "svd_cutoff_trajectories.csv", all_traj)
    _write_csv(out / "svd_truncation_events.csv", all_events)
    _write_csv(out / "svd_cutoff_summary.csv", summary)
    np.savez_compressed(out / "representative_spectra.npz", **all_spectra)
    (out / "config.json").write_text(
        json.dumps(
            {
                **semantics,
                "timesteps": timesteps,
                "reliability_threshold": RELIABILITY_THRESHOLD,
                "exact_reference": str(exact_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote outputs under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
