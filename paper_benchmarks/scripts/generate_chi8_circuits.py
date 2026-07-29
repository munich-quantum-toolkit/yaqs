# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Generate the circuit suite at dt=0.1, chi_max=8 (full_tdvp on all 2q gates).

Same models / schedules / physical window as the production chi=32 campaign,
but with a tighter bond-dimension cap. Reuses existing dense exact references.

Usage:
    uv run python paper_benchmarks/scripts/generate_chi8_circuits.py
"""

from __future__ import annotations

import copy
import csv
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from pb_common import (
    CIRCUIT_1D_L,
    CIRCUIT_DT,
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

CHI = 8
DT = CIRCUIT_DT  # 0.1
TIMESTEPS = CIRCUIT_TIMESTEPS  # 30
OUT_DIR = RAW_NEW_DIR / "circuits_chi8"

MODELS_1D = ("ising_1d", "heisenberg_1d")
MODELS_2D = ("ising", "heisenberg")
METHODS = ("full_tdvp", "tebd_swap", "mpo_zipup")

ISING_J = ISING_G = 1.0
HEIS_J = 1.0
HEIS_1D_H = 1.0
HEIS_2D_H = 0.0


def _bond_pairs_1d(length: int) -> list[tuple[int, int]]:
    pairs = [(i, i + 1) for i in range(0, length - 1, 2)]
    pairs += [(i, i + 1) for i in range(1, length - 1, 2)]
    return pairs


def _site_index(row: int, col: int, *, num_cols: int = 4) -> int:
    if row % 2 == 0:
        return row * num_cols + col
    return row * num_cols + (num_cols - 1 - col)


def _ising_2d_bond_gates(beta: float) -> list:
    from circuits import GateOp

    gates: list = []
    for row in range(4):
        for col in range(0, 3, 2):
            gates.append(GateOp("rzz", (_site_index(row, col), _site_index(row, col + 1)), beta))
        for col in range(1, 3, 2):
            gates.append(GateOp("rzz", (_site_index(row, col), _site_index(row, col + 1)), beta))
    for col in range(4):
        for row in range(0, 3, 2):
            gates.append(GateOp("rzz", (_site_index(row, col), _site_index(row + 1, col)), beta))
        for row in range(1, 3, 2):
            gates.append(GateOp("rzz", (_site_index(row, col), _site_index(row + 1, col)), beta))
    return gates


def _heisenberg_2d_bond_gates(half: float) -> list:
    from circuits import GateOp

    gates: list = []
    for gate_name in ("rzz", "rxx", "ryy"):
        for row in range(4):
            for col in range(0, 3, 2):
                gates.append(
                    GateOp(gate_name, (_site_index(row, col), _site_index(row, col + 1)), half)
                )
            for col in range(1, 3, 2):
                gates.append(
                    GateOp(gate_name, (_site_index(row, col), _site_index(row, col + 1)), half)
                )
        for col in range(4):
            for row in range(0, 3, 2):
                gates.append(
                    GateOp(gate_name, (_site_index(row, col), _site_index(row + 1, col)), half)
                )
            for row in range(1, 3, 2):
                gates.append(
                    GateOp(gate_name, (_site_index(row, col), _site_index(row + 1, col)), half)
                )
    return gates


def build_schedule(model: str) -> tuple:
    from circuits import GateOp, TrotterStep

    steps = []
    if model == "ising_1d":
        alpha = -2.0 * DT * ISING_G
        half = -2.0 * DT * ISING_J / 2.0
        field = [GateOp("rx", (q,), alpha) for q in range(CIRCUIT_1D_L)]
        for idx in range(TIMESTEPS):
            bond = [GateOp("rzz", (a, b), half) for a, b in _bond_pairs_1d(CIRCUIT_1D_L)]
            steps.append(TrotterStep(index=idx, gates=tuple(bond + field + bond)))
    elif model == "heisenberg_1d":
        half = -2.0 * DT * HEIS_J / 2.0
        theta_z = -2.0 * DT * HEIS_1D_H
        field = [GateOp("rz", (q,), theta_z) for q in range(CIRCUIT_1D_L)]
        for idx in range(TIMESTEPS):
            bond = []
            for name in ("rzz", "rxx", "ryy"):
                bond += [GateOp(name, (a, b), half) for a, b in _bond_pairs_1d(CIRCUIT_1D_L)]
            steps.append(TrotterStep(index=idx, gates=tuple(bond + field + bond)))
    elif model == "ising":
        alpha = -2.0 * DT * ISING_G
        half = -2.0 * DT * ISING_J / 2.0
        field = [
            GateOp("rx", (_site_index(r, c),), alpha) for r in range(4) for c in range(4)
        ]
        for idx in range(TIMESTEPS):
            bond = _ising_2d_bond_gates(half)
            steps.append(TrotterStep(index=idx, gates=tuple(bond + field + bond)))
    else:
        half = -2.0 * DT * HEIS_J / 2.0
        theta_z = -2.0 * DT * HEIS_2D_H
        field = (
            [GateOp("rz", (_site_index(r, c),), theta_z) for r in range(4) for c in range(4)]
            if abs(theta_z) > 1e-15
            else []
        )
        for idx in range(TIMESTEPS):
            bond = _heisenberg_2d_bond_gates(half)
            steps.append(TrotterStep(index=idx, gates=tuple(bond + field + bond)))
    return tuple(steps)


def num_qubits(model: str) -> int:
    return CIRCUIT_1D_L if model.endswith("_1d") else 16


def initial_mps(model: str):
    from circuits import neel_basis_string
    from mqt.yaqs.core.data_structures.mps import MPS

    nq = num_qubits(model)
    if model in ("ising_1d", "ising"):
        return MPS(nq, state="zeros")
    if model == "heisenberg_1d":
        basis = "".join("0" if i % 2 == 0 else "1" for i in range(nq))
        return MPS(nq, state="basis", basis_string=basis)
    return MPS(nq, state="basis", basis_string=neel_basis_string(num_rows=4, num_cols=4))


def initial_vector(model: str) -> np.ndarray:
    from circuits import neel_basis_string
    from mqt.yaqs.core.data_structures.state_utils import product_state_vector

    nq = num_qubits(model)
    if model in ("ising_1d", "ising"):
        return product_state_vector(nq, "zeros", 2)
    if model == "heisenberg_1d":
        basis = "".join("0" if i % 2 == 0 else "1" for i in range(nq))
        return product_state_vector(nq, "basis", 2, basis_string=basis)
    return product_state_vector(
        nq, "basis", 2, basis_string=neel_basis_string(num_rows=4, num_cols=4)
    )


def load_exact(model: str) -> np.ndarray:
    """Reuse existing dense exact references at dt=0.1, 30 steps."""
    if model.endswith("_1d"):
        path = RAW_NEW_DIR / "circuits_1d" / f"exact_{model}_t{TIMESTEPS}.npy"
    else:
        path = RAW_DIR / "circuits_corrected" / f"exact_{model}_t{TIMESTEPS}.npy"
    if not path.exists():
        msg = f"Missing exact reference {path}"
        raise FileNotFoundError(msg)
    arr = np.load(path)
    assert arr.shape[0] >= TIMESTEPS + 1
    return arr


def run_trajectory(model: str, method: str, exact: np.ndarray) -> list[dict[str, Any]]:
    from trajectory import (
        TrajectoryState,
        apply_trotter_step_mps,
        attach_tdvp_substeps,
        compute_metrics,
    )

    schedule = build_schedule(model)
    nq = num_qubits(model)
    st = TrajectoryState(
        mps=copy.deepcopy(initial_mps(model)),
        vec=initial_vector(model).copy(),
        num_qubits=nq,
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
                    chi=CHI,
                    trotter_step=step_idx,
                    time=step_idx * DT,
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
            chi=CHI,
            tdvp_substeps=CIRCUIT_TDVP_SUBSTEPS,
            update_vec=True,
        )
        record(step_idx, st.cumulative_runtime_s - t_before)
        if st.failed:
            break
    return rows


def _job(model: str, method: str) -> str:
    from generate_corrected import CSV_FIELDS

    out = OUT_DIR / f"{model}_chi{CHI}_{method}.csv"
    if out.exists():
        with out.open(encoding="utf-8") as fh:
            nrows = sum(1 for _ in fh) - 1
        if nrows >= TIMESTEPS + 1:
            return f"skip {model} {method}: complete ({nrows} rows)"
        out.unlink()
    exact = load_exact(model)
    t0 = time.perf_counter()
    rows = run_trajectory(model, method, exact)
    tmp = out.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.rename(out)
    wall = time.perf_counter() - t0
    final = float(rows[-1]["infidelity"]) if rows else float("nan")
    return (
        f"{model} {method}: {len(rows)} rows in {wall:.1f}s "
        f"final 1-F={final:.3e} peakχ={rows[-1]['peak_max_bond']} -> {out.name}"
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log = (LOGS_DIR / "generate_chi8_circuits.log").open("a", encoding="utf-8", buffering=1)

    jobs = [(m, meth) for m in (*MODELS_1D, *MODELS_2D) for meth in METHODS]
    print(
        f"=== chi={CHI}, dt={DT}, timesteps={TIMESTEPS}, "
        f"full_tdvp n={CIRCUIT_TDVP_SUBSTEPS}: {len(jobs)} jobs ===",
        flush=True,
    )
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
