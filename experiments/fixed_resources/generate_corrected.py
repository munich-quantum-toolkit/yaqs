# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Regenerate corrected fixed-χ circuit benchmark from repaired implementations."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from circuits import build_heisenberg_schedule, build_ising_schedule
from config import (
    CHI_CONTROL,
    CHI_HEISENBERG,
    CHI_HORIZON,
    CHI_MAIN,
    CONTROL_INF_TOL_EXACT,
    CONTROL_INF_TOL_TDVP,
    CONTROL_STEPS,
    CORRECTED_OUTPUT_DIR,
    DT,
    METHODS,
    RELIABILITY_THRESHOLD,
    SUBDIVISION_HEIS_STEPS,
    SUBDIVISION_NS,
    SUBDIVISION_TFIM_STEPS,
    THRESHOLD_SENSITIVITY,
    TMAX_INITIAL,
    TDVP_SUBSTEPS,
    production_config,
    timesteps_for_tmax,
)
from horizons import reliable_horizon
from trajectory import (
    TrajectoryState,
    apply_trotter_step_dense,
    apply_trotter_step_mps,
    attach_tdvp_substeps,
    compute_metrics,
    initial_mps,
    initial_vector,
)

CSV_FIELDS = [
    "model",
    "method",
    "chi_max",
    "trotter_step",
    "time",
    "infidelity",
    "fidelity",
    "state_norm",
    "norm_drift",
    "phase_aligned_distance",
    "current_max_bond",
    "peak_max_bond",
    "param_count",
    "memory_bytes",
    "peak_param_count",
    "peak_memory_bytes",
    "step_runtime_s",
    "cumulative_runtime_s",
    "discarded_weight_step",
    "compression_residual_step",
    "tdvp_substeps",
    "variational_init",
    "variational_converged",
    "variational_sweeps",
    "failed",
    "failure_message",
]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    use_fields = fields or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=use_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in use_fields})


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def precompute_exact(model: str, timesteps: int, path: Path) -> np.ndarray:
    if path.exists():
        arr = np.load(path)
        if arr.shape[0] >= timesteps + 1:
            return arr
    schedule = (
        build_ising_schedule(timesteps=timesteps)
        if model == "ising"
        else build_heisenberg_schedule(timesteps=timesteps)
    )
    vec = initial_vector(model)
    out = np.zeros((timesteps + 1, vec.size), dtype=np.complex128)
    out[0] = vec
    for i, step in enumerate(schedule, start=1):
        vec = apply_trotter_step_dense(vec, step)
        out[i] = vec
        print(f"  exact {model} step {i}/{timesteps}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out


def run_trajectory(
    *,
    model: str,
    method: str,
    chi: int,
    timesteps: int,
    exact: np.ndarray,
    tdvp_substeps: int,
    stop_after_crossing: bool = False,
    epsilon: float = RELIABILITY_THRESHOLD,
) -> list[dict[str, Any]]:
    """Run one independent trajectory from a deep-copied initial state."""
    schedule = (
        build_ising_schedule(timesteps=timesteps)
        if model == "ising"
        else build_heisenberg_schedule(timesteps=timesteps)
    )
    st = TrajectoryState(mps=copy.deepcopy(initial_mps(model)), vec=initial_vector(model).copy())
    rows: list[dict[str, Any]] = []
    row0 = attach_tdvp_substeps(
        compute_metrics(
            exact[0],
            st.vec,
            state=st,
            model=model,
            method=method,
            chi=chi,
            trotter_step=0,
            time=0.0,
            step_runtime_s=0.0,
        ),
        tdvp_substeps,
    )
    rows.append(row0)
    for step_idx, step in enumerate(schedule, start=1):
        t_before = st.cumulative_runtime_s
        apply_trotter_step_mps(
            st, step, method=method, chi=chi, tdvp_substeps=tdvp_substeps, update_vec=True
        )
        step_runtime = st.cumulative_runtime_s - t_before
        if st.failed:
            row = attach_tdvp_substeps(
                compute_metrics(
                    exact[step_idx],
                    st.vec if st.vec is not None else exact[step_idx],
                    state=st,
                    model=model,
                    method=method,
                    chi=chi,
                    trotter_step=step_idx,
                    time=step_idx * DT,
                    step_runtime_s=step_runtime,
                ),
                tdvp_substeps,
            )
            rows.append(row)
            break
        row = attach_tdvp_substeps(
            compute_metrics(
                exact[step_idx],
                st.vec,
                state=st,
                model=model,
                method=method,
                chi=chi,
                trotter_step=step_idx,
                time=step_idx * DT,
                step_runtime_s=step_runtime,
            ),
            tdvp_substeps,
        )
        rows.append(row)
        if stop_after_crossing and float(row["infidelity"]) >= epsilon:
            break
    return rows


def choose_subdivision(sub_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick smallest n where doubling leaves the horizon effectively unchanged."""

    def traj(model: str, n: int) -> list[dict[str, Any]]:
        return [r for r in sub_rows if r["model"] == model and int(float(r["tdvp_substeps"])) == n]

    def neps(model: str, n: int) -> int:
        rows = traj(model, n)
        if not rows:
            return -1
        return int(reliable_horizon(rows, epsilon=RELIABILITY_THRESHOLD, dt=DT)["n_eps"])

    available = sorted({int(float(r["tdvp_substeps"])) for r in sub_rows})
    table = []
    for n in available:
        table.append(
            {
                "n": n,
                "tfim_n_eps": neps("ising", n),
                "heis_n_eps": neps("heisenberg", n),
                "tfim_inf_step1": float(
                    next(r["infidelity"] for r in traj("ising", n) if int(float(r["trotter_step"])) == 1)
                ),
                "heis_inf_step1": float(
                    next(
                        r["infidelity"] for r in traj("heisenberg", n) if int(float(r["trotter_step"])) == 1
                    )
                ),
            }
        )

    pairs = [(n, available[i + 1]) for i, n in enumerate(available[:-1]) if available[i + 1] == 2 * n]
    # Prefer exact horizon agreement under doubling (Δnε=0), else allow Δ≤1.
    chosen = available[0]
    reason = f"fallback smallest available n={chosen}"
    for prefer_exact in (True, False):
        for n, n2 in pairs:
            d_tfim = abs(neps("ising", n) - neps("ising", n2))
            d_heis = abs(neps("heisenberg", n) - neps("heisenberg", n2))
            same_class = (neps("ising", n) == 0) == (neps("ising", n2) == 0) and (
                neps("heisenberg", n) == 0
            ) == (neps("heisenberg", n2) == 0)
            ok = same_class and (
                (d_tfim == 0 and d_heis == 0)
                if prefer_exact
                else (d_tfim <= 1 and d_heis <= 1)
            )
            if ok:
                chosen = n
                reason = (
                    f"smallest n with doubling-stable horizon "
                    f"(n={n} vs {n2}: ΔTFIM={d_tfim}, ΔHeis={d_heis}, prefer_exact={prefer_exact})"
                )
                return {"chosen_n": chosen, "reason": reason, "table": table}
    return {"chosen_n": chosen, "reason": reason, "table": table}


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _has_traj(rows: list[dict[str, Any]], *, model: str, method: str, chi: int, n: int) -> bool:
    return any(
        r["model"] == model
        and r["method"] == method
        and int(float(r["chi_max"])) == chi
        and int(float(r["tdvp_substeps"])) == n
        for r in rows
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Corrected fixed-χ circuit regeneration")
    parser.add_argument("--output-dir", type=Path, default=CORRECTED_OUTPUT_DIR)
    parser.add_argument("--skip-variational", action="store_true")
    parser.add_argument("--skip-control", action="store_true")
    parser.add_argument("--skip-subdivision", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tdvp-substeps", type=int, default=None, help="Override chosen n")
    args = parser.parse_args(argv)
    out: Path = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("=== Precompute exact references ===", flush=True)
    tmax_steps = timesteps_for_tmax(TMAX_INITIAL)
    exact_ising = precompute_exact("ising", tmax_steps, out / f"exact_ising_t{tmax_steps}.npy")
    exact_heis = precompute_exact("heisenberg", tmax_steps, out / f"exact_heisenberg_t{tmax_steps}.npy")

    # --- Subdivision validation ---
    sub_path = out / "circuit_subdivision_validation.csv"
    sub_rows: list[dict[str, Any]] = _load_csv_rows(sub_path)
    if not args.skip_subdivision:
        print("=== Subdivision validation ===", flush=True)
        if not args.resume:
            sub_rows = []
        for n in SUBDIVISION_NS:
            if not _has_traj(sub_rows, model="ising", method="hybrid_tdvp", chi=CHI_MAIN, n=n):
                print(f" TFIM χ={CHI_MAIN} n={n} steps={SUBDIVISION_TFIM_STEPS}", flush=True)
                sub_rows.extend(
                    run_trajectory(
                        model="ising",
                        method="hybrid_tdvp",
                        chi=CHI_MAIN,
                        timesteps=SUBDIVISION_TFIM_STEPS,
                        exact=exact_ising,
                        tdvp_substeps=n,
                    )
                )
                _write_csv(sub_path, sub_rows, CSV_FIELDS)
            else:
                print(f" SKIP TFIM n={n}", flush=True)
            if not _has_traj(sub_rows, model="heisenberg", method="hybrid_tdvp", chi=CHI_MAIN, n=n):
                print(f" Heisenberg χ={CHI_MAIN} n={n} steps={SUBDIVISION_HEIS_STEPS}", flush=True)
                sub_rows.extend(
                    run_trajectory(
                        model="heisenberg",
                        method="hybrid_tdvp",
                        chi=CHI_MAIN,
                        timesteps=SUBDIVISION_HEIS_STEPS,
                        exact=exact_heis,
                        tdvp_substeps=n,
                    )
                )
                _write_csv(sub_path, sub_rows, CSV_FIELDS)
            else:
                print(f" SKIP Heisenberg n={n}", flush=True)
        _write_csv(sub_path, sub_rows, CSV_FIELDS)
        choice = choose_subdivision(sub_rows)
    else:
        print("=== Subdivision skipped; using existing CSV / override ===", flush=True)
        choice = choose_subdivision(sub_rows) if sub_rows else {"chosen_n": TDVP_SUBSTEPS, "reason": "config default", "table": []}
    n_prod = int(args.tdvp_substeps) if args.tdvp_substeps is not None else int(choice["chosen_n"])
    choice["production_n"] = n_prod
    _save_json(out / "subdivision_choice.json", choice)
    print(f"Chosen TDVP n={n_prod}: {choice['reason']}", flush=True)

    # --- Production trajectories ---
    results_path = out / "circuit_results_corrected.csv"
    all_rows: list[dict[str, Any]] = _load_csv_rows(results_path) if args.resume else []

    print("=== TFIM horizon scan ===", flush=True)
    for chi in CHI_HORIZON:
        for method in METHODS:
            if _has_traj(all_rows, model="ising", method=method, chi=chi, n=n_prod):
                print(f"  SKIP ising {method} χ={chi}", flush=True)
                continue
            print(f"  ising {method} χ={chi}", flush=True)
            full = chi == CHI_MAIN
            rows = run_trajectory(
                model="ising",
                method=method,
                chi=chi,
                timesteps=tmax_steps if full else min(tmax_steps, 40),
                exact=exact_ising,
                tdvp_substeps=n_prod,
                stop_after_crossing=not full,
            )
            all_rows.extend(rows)
            _write_csv(results_path, all_rows, CSV_FIELDS)

    print("=== Heisenberg scan ===", flush=True)
    heis_multi = False
    for chi in CHI_HEISENBERG:
        for method in METHODS:
            if _has_traj(all_rows, model="heisenberg", method=method, chi=chi, n=n_prod):
                print(f"  SKIP heisenberg {method} χ={chi}", flush=True)
                existing = [
                    r
                    for r in all_rows
                    if r["model"] == "heisenberg"
                    and r["method"] == method
                    and int(float(r["chi_max"])) == chi
                ]
                h = reliable_horizon(existing, epsilon=RELIABILITY_THRESHOLD, dt=DT)
                if int(h["n_eps"]) >= 1:
                    heis_multi = True
                continue
            print(f"  heisenberg {method} χ={chi}", flush=True)
            rows = run_trajectory(
                model="heisenberg",
                method=method,
                chi=chi,
                timesteps=min(tmax_steps, 20),
                exact=exact_heis,
                tdvp_substeps=n_prod,
                stop_after_crossing=True,
            )
            all_rows.extend(rows)
            _write_csv(results_path, all_rows, CSV_FIELDS)
            h = reliable_horizon(rows, epsilon=RELIABILITY_THRESHOLD, dt=DT)
            if int(h["n_eps"]) >= 1:
                heis_multi = True

    # --- Variational control ---
    var_rows: list[dict[str, Any]] = []
    if not args.skip_variational:
        print("=== Variational control ===", flush=True)
        print(f"  TFIM variational χ={CHI_MAIN} full traj", flush=True)
        t0 = time.perf_counter()
        try:
            var_rows.extend(
                run_trajectory(
                    model="ising",
                    method="variational_mpo",
                    chi=CHI_MAIN,
                    timesteps=tmax_steps,
                    exact=exact_ising,
                    tdvp_substeps=n_prod,
                    stop_after_crossing=False,
                )
            )
            print(f"  TFIM variational done in {time.perf_counter() - t0:.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  TFIM variational failed: {exc}", flush=True)
        for chi in CHI_HEISENBERG:
            print(f"  Heisenberg variational χ={chi} first step", flush=True)
            try:
                var_rows.extend(
                    run_trajectory(
                        model="heisenberg",
                        method="variational_mpo",
                        chi=chi,
                        timesteps=1,
                        exact=exact_heis,
                        tdvp_substeps=n_prod,
                        stop_after_crossing=True,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  Heisenberg variational χ={chi} failed: {exc}", flush=True)

    # --- High-χ control ---
    if not args.skip_control:
        print("=== High-χ control (Ising χ=256) ===", flush=True)
        control_rows: list[dict[str, Any]] = []
        for method in METHODS:
            rows = run_trajectory(
                model="ising",
                method=method,
                chi=CHI_CONTROL,
                timesteps=CONTROL_STEPS,
                exact=exact_ising,
                tdvp_substeps=n_prod,
            )
            control_rows.extend(rows)
            for r in rows:
                if int(r["trotter_step"]) == 0:
                    continue
                inf = float(r["infidelity"])
                tol = CONTROL_INF_TOL_TDVP if method == "hybrid_tdvp" else CONTROL_INF_TOL_EXACT
                ok = inf <= tol
                print(f"  {method} step {r['trotter_step']}: 1-F={inf:.3e} tol={tol:g} ok={ok}", flush=True)
                if not ok:
                    msg = f"Control failed for {method} at χ={CHI_CONTROL}: infidelity={inf}"
                    raise RuntimeError(msg)
        _write_csv(out / "control_chi256.csv", control_rows, CSV_FIELDS)

    # Deterministic repeat of representative points
    print("=== Deterministic repeat (TFIM χ=32, 5 steps) ===", flush=True)
    repeat_rows = []
    for method in METHODS:
        a = run_trajectory(
            model="ising",
            method=method,
            chi=CHI_MAIN,
            timesteps=5,
            exact=exact_ising,
            tdvp_substeps=n_prod,
        )
        b = run_trajectory(
            model="ising",
            method=method,
            chi=CHI_MAIN,
            timesteps=5,
            exact=exact_ising,
            tdvp_substeps=n_prod,
        )
        for ra, rb in zip(a, b, strict=True):
            if abs(float(ra["infidelity"]) - float(rb["infidelity"])) > 1e-14:
                msg = f"Non-deterministic repeat for {method} step {ra['trotter_step']}"
                raise RuntimeError(msg)
        repeat_rows.extend(a)
    _write_csv(out / "deterministic_repeat.csv", repeat_rows, CSV_FIELDS)

    _write_csv(out / "circuit_results_corrected.csv", all_rows, CSV_FIELDS)
    _write_csv(out / "variational_circuit_control.csv", var_rows, CSV_FIELDS)

    # Horizons
    horizon_rows: list[dict[str, Any]] = []
    sens_rows: list[dict[str, Any]] = []
    for model in ("ising", "heisenberg"):
        for method in METHODS:
            chis = CHI_HORIZON if model == "ising" else CHI_HEISENBERG
            for chi in chis:
                traj = [
                    r
                    for r in all_rows
                    if r["model"] == model and r["method"] == method and int(r["chi_max"]) == chi
                ]
                if not traj:
                    continue
                for eps in THRESHOLD_SENSITIVITY:
                    h = reliable_horizon(traj, epsilon=eps, dt=DT)
                    peak = max(int(r["peak_max_bond"]) for r in traj)
                    params = max(int(r["peak_param_count"]) for r in traj)
                    runtime = max(float(r["cumulative_runtime_s"]) for r in traj)
                    rec = {
                        "model": model,
                        "method": method,
                        "chi_max": chi,
                        "tdvp_substeps": n_prod,
                        "peak_max_bond": peak,
                        "peak_param_count": params,
                        "cumulative_runtime_s": runtime,
                        **h,
                    }
                    sens_rows.append(rec)
                    if abs(eps - RELIABILITY_THRESHOLD) < 1e-15:
                        horizon_rows.append(rec)
        # variational horizons for TFIM control
        for method in ("variational_mpo",):
            traj = [r for r in var_rows if r["model"] == model and r["method"] == method]
            if not traj:
                continue
            # group by chi
            chis = sorted({int(r["chi_max"]) for r in traj})
            for chi in chis:
                sub = [r for r in traj if int(r["chi_max"]) == chi]
                h = reliable_horizon(sub, epsilon=RELIABILITY_THRESHOLD, dt=DT)
                horizon_rows.append(
                    {
                        "model": model,
                        "method": method,
                        "chi_max": chi,
                        "tdvp_substeps": n_prod,
                        "peak_max_bond": max(int(r["peak_max_bond"]) for r in sub),
                        "peak_param_count": max(int(r["peak_param_count"]) for r in sub),
                        "cumulative_runtime_s": max(float(r["cumulative_runtime_s"]) for r in sub),
                        **h,
                    }
                )

    _write_csv(out / "circuit_horizons_corrected.csv", horizon_rows)
    _write_csv(out / "threshold_sensitivity.csv", sens_rows)
    _save_json(
        out / "config.json",
        {
            **production_config(tmax=TMAX_INITIAL, tdvp_substeps=n_prod),
            "heisenberg_has_multistep_horizon": heis_multi,
            "subdivision": choice,
        },
    )
    print(f"Wrote results under {out}")
    print(f"heisenberg_has_multistep_horizon={heis_multi}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
