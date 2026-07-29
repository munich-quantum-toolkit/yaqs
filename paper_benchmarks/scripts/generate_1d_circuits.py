# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Stage 3c (generate new data): 1D chain circuit benchmarks at chi=32.

Two 16-site open-boundary chains with the same locked conventions as the 2D
benchmark (second-order Suzuki-Trotter bond/2, field, bond/2; dt=0.1;
30 steps; ferromagnetic sign convention; dense exact reference of the
identical Trotter circuit):

  ising_1d       1D TFIM, H = -J sum Z_i Z_{i+1} - g sum X_i with J = g = 1,
                 initial state |0...0>.
  heisenberg_1d  1D XXX Heisenberg, H = -J sum (XX+YY+ZZ) - h sum Z with
                 J = h = 1, Neel initial state |0101...>.

All two-qubit gates are nearest-neighbour. The TDVP method here is
"full_tdvp" (gate_mode="full-tdvp"): every two-qubit gate is applied through
the gate-local TDVP window update with n=2 fractional-time substeps -- no
direct TEBD contraction for nearest-neighbour gates. TEBD+SWAP (no SWAPs
actually needed in 1D) and MPO zip-up use the same production truncation
semantics (discarded_weight @ 1e-13, chi_max=32).

Checkpointing: one CSV per (model, method) under raw_new/circuits_1d/; a
completed trajectory (31 rows) is skipped on resume. Exact references are
cached as .npy in the same directory.

Usage:
    uv run python paper_benchmarks/scripts/generate_1d_circuits.py
"""

from __future__ import annotations

import copy
import csv
import time
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Any

from pb_common import (
    CIRCUIT_1D_L,
    CIRCUIT_1D_METHODS,
    CIRCUIT_1D_MODELS,
    CIRCUIT_CHI_MAIN,
    CIRCUIT_DT,
    CIRCUIT_TDVP_SUBSTEPS,
    CIRCUIT_TIMESTEPS,
    LOGS_DIR,
    RAW_NEW_DIR,
    add_experiment_path,
    limit_blas_threads,
    worker_count,
)

limit_blas_threads()
add_experiment_path("fixed_resources")

import numpy as np  # ruff: ignore[module-import-not-at-top-of-file]

if TYPE_CHECKING:
    from circuits import TrotterStep

OUT_DIR = RAW_NEW_DIR / "circuits_1d"

# Model parameters (locked; ferromagnetic convention as in the 2D benchmark).
ISING_1D_J = 1.0
ISING_1D_G = 1.0
HEIS_1D_J = 1.0
HEIS_1D_H = 1.0


def _bond_pairs(length: int) -> list[tuple[int, int]]:
    """Even bonds then odd bonds (1D analogue of the 2D edge colouring)."""
    pairs = [(i, i + 1) for i in range(0, length - 1, 2)]
    pairs += [(i, i + 1) for i in range(1, length - 1, 2)]
    return pairs


def build_ising_1d_schedule(*, timesteps: int, length: int) -> tuple:
    """Second-order Trotter for the 1D TFIM: bond(dt/2), field(dt), bond(dt/2)."""
    from circuits import GateOp, TrotterStep

    alpha = -2.0 * CIRCUIT_DT * ISING_1D_G
    beta = -2.0 * CIRCUIT_DT * ISING_1D_J
    half = beta / 2.0

    def bond_layer() -> list:
        return [GateOp("rzz", (q1, q2), half) for q1, q2 in _bond_pairs(length)]

    field = [GateOp("rx", (q,), alpha) for q in range(length)]
    steps = []
    for idx in range(timesteps):
        gates = bond_layer() + field + bond_layer()
        steps.append(TrotterStep(index=idx, gates=tuple(gates)))
    return tuple(steps)


def build_heisenberg_1d_schedule(*, timesteps: int, length: int) -> tuple:
    """Second-order Trotter for the 1D XXX chain with field h (rz layer)."""
    from circuits import GateOp, TrotterStep

    theta = -2.0 * CIRCUIT_DT * HEIS_1D_J
    theta_z = -2.0 * CIRCUIT_DT * HEIS_1D_H
    half = theta / 2.0

    def bond_layer() -> list:
        gates = []
        for gate_name in ("rzz", "rxx", "ryy"):  # same axis order as the 2D benchmark
            gates += [GateOp(gate_name, (q1, q2), half) for q1, q2 in _bond_pairs(length)]
        return gates

    field = [GateOp("rz", (q,), theta_z) for q in range(length)]
    steps = []
    for idx in range(timesteps):
        gates = bond_layer() + field + bond_layer()
        steps.append(TrotterStep(index=idx, gates=tuple(gates)))
    return tuple(steps)


def build_schedule(model: str, *, timesteps: int, length: int) -> tuple:
    if model == "ising_1d":
        return build_ising_1d_schedule(timesteps=timesteps, length=length)
    return build_heisenberg_1d_schedule(timesteps=timesteps, length=length)


def neel_1d_string(length: int) -> str:
    return "".join("0" if i % 2 == 0 else "1" for i in range(length))


def initial_mps_1d(model: str, length: int):
    from mqt.yaqs.core.data_structures.mps import MPS

    if model == "ising_1d":
        return MPS(length, state="zeros")
    return MPS(length, state="basis", basis_string=neel_1d_string(length))


def initial_vector_1d(model: str, length: int) -> np.ndarray:
    from mqt.yaqs.core.data_structures.state_utils import product_state_vector

    if model == "ising_1d":
        return product_state_vector(length, "zeros", 2)
    return product_state_vector(length, "basis", 2, basis_string=neel_1d_string(length))


def precompute_exact_1d(model: str) -> np.ndarray:
    """Dense reference states after each Trotter step (identical circuit)."""
    from trajectory import apply_trotter_step_dense

    path = OUT_DIR / f"exact_{model}_t{CIRCUIT_TIMESTEPS}.npy"
    if path.exists():
        arr = np.load(path)
        if arr.shape[0] >= CIRCUIT_TIMESTEPS + 1:
            return arr
    schedule = build_schedule(model, timesteps=CIRCUIT_TIMESTEPS, length=CIRCUIT_1D_L)
    vec = initial_vector_1d(model, CIRCUIT_1D_L)
    out = np.zeros((CIRCUIT_TIMESTEPS + 1, vec.size), dtype=np.complex128)
    out[0] = vec
    for i, step in enumerate(schedule, start=1):
        vec = apply_trotter_step_dense(vec, step, num_qubits=CIRCUIT_1D_L)
        out[i] = vec
        print(f"  exact {model} step {i}/{CIRCUIT_TIMESTEPS}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out


def run_trajectory_1d(model: str, method: str, exact: np.ndarray) -> list[dict[str, Any]]:
    """One full 30-step trajectory (no early stopping)."""
    from trajectory import (
        TrajectoryState,
        apply_trotter_step_mps,
        attach_tdvp_substeps,
        compute_metrics,
    )

    schedule = build_schedule(model, timesteps=CIRCUIT_TIMESTEPS, length=CIRCUIT_1D_L)
    st = TrajectoryState(
        mps=copy.deepcopy(initial_mps_1d(model, CIRCUIT_1D_L)),
        vec=initial_vector_1d(model, CIRCUIT_1D_L).copy(),
        num_qubits=CIRCUIT_1D_L,
    )
    rows: list[dict[str, Any]] = []

    def record(step_idx: int, step_runtime: float) -> None:
        rows.append(
            attach_tdvp_substeps(
                compute_metrics(
                    exact[step_idx],
                    st.vec,
                    state=st,
                    model=model,
                    method=method,
                    chi=CIRCUIT_CHI_MAIN,
                    trotter_step=step_idx,
                    time=step_idx * CIRCUIT_DT,
                    step_runtime_s=step_runtime,
                ),
                CIRCUIT_TDVP_SUBSTEPS,
            )
        )

    record(0, 0.0)
    for step_idx, step in enumerate(schedule, start=1):
        t_before = st.cumulative_runtime_s
        apply_trotter_step_mps(
            st,
            step,
            method=method,
            chi=CIRCUIT_CHI_MAIN,
            tdvp_substeps=CIRCUIT_TDVP_SUBSTEPS,
            update_vec=True,
        )
        record(step_idx, st.cumulative_runtime_s - t_before)
        if st.failed:
            break
    return rows


def _job(model: str, method: str) -> str:
    """Worker entry: run one trajectory and write its checkpoint CSV."""
    from generate_corrected import CSV_FIELDS

    out = OUT_DIR / f"{model}_chi{CIRCUIT_CHI_MAIN}_{method}.csv"
    if out.exists():
        with out.open(encoding="utf-8") as fh:
            nrows = sum(1 for _ in fh) - 1
        if nrows >= CIRCUIT_TIMESTEPS + 1:
            return f"skip {model} {method}: complete ({nrows} rows)"
        out.unlink()  # partial file: regenerate deterministically
    exact = np.load(OUT_DIR / f"exact_{model}_t{CIRCUIT_TIMESTEPS}.npy")
    t0 = time.perf_counter()
    rows = run_trajectory_1d(model, method, exact)
    tmp = out.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.rename(out)
    wall = time.perf_counter() - t0
    return f"{model} {method}: {len(rows)} rows in {wall:.1f}s -> {out.name}"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log = (LOGS_DIR / "generate_1d_circuits.log").open("a", encoding="utf-8", buffering=1)

    print("=== Precompute 1D exact references ===", flush=True)
    for model in CIRCUIT_1D_MODELS:
        precompute_exact_1d(model)

    jobs = [(model, method) for model in CIRCUIT_1D_MODELS for method in CIRCUIT_1D_METHODS]
    print(f"=== Running {len(jobs)} trajectories "
          f"(chi={CIRCUIT_CHI_MAIN}, {CIRCUIT_TIMESTEPS} steps, "
          f"TDVP n={CIRCUIT_TDVP_SUBSTEPS} on ALL two-qubit gates) ===", flush=True)
    exit_code = 0
    with ProcessPoolExecutor(max_workers=min(worker_count(6), len(jobs))) as pool:
        futures = {pool.submit(_job, m, meth): (m, meth) for m, meth in jobs}
        for fut, (m, meth) in futures.items():
            try:
                msg = fut.result()
            except Exception as exc:  # noqa: BLE001
                exit_code = 1
                msg = f"JOB FAILED {m} {meth}: {exc!r}"
            print(msg, flush=True)
            log.write(msg + "\n")
    log.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
