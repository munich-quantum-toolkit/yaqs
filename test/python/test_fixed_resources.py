# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Tests for the fixed-resource 2D circuit benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

FIXED_DIR = Path(__file__).resolve().parents[2] / "experiments" / "fixed_resources"
sys.path.insert(0, str(FIXED_DIR))

from circuits import (  # noqa: E402
    build_heisenberg_schedule,
    build_ising_schedule,
    build_qiskit_circuit,
    neel_basis_string,
)
from config import CHI_MAIN, DT, METHODS, NUM_QUBITS, OUTPUT_DIR, T0_INFIDELITY_TOL  # noqa: E402
from trajectory import (  # noqa: E402
    TrajectoryState,
    apply_trotter_step_dense,
    apply_trotter_step_mps,
    compute_metrics,
    initial_mps,
    initial_vector,
    qiskit_reference,
)


def test_neel_basis_is_checkerboard() -> None:
    s = neel_basis_string()
    assert len(s) == NUM_QUBITS
    assert set(s) == {"0", "1"}


def test_second_order_schedule_gate_count() -> None:
    ising = build_ising_schedule(timesteps=1)[0]
    heis = build_heisenberg_schedule(timesteps=1)[0]
    assert len(ising.gates) == 64
    assert len(heis.gates) == 144


def test_exact_matches_qiskit_one_step() -> None:
    for model in ("ising", "heisenberg"):
        step = build_ising_schedule(timesteps=1)[0] if model == "ising" else build_heisenberg_schedule(timesteps=1)[0]
        rebuilt = apply_trotter_step_dense(initial_vector(model), step)
        ref = qiskit_reference(model, timesteps=1)
        assert float(np.max(np.abs(rebuilt - ref))) < 1e-10


def test_t0_infidelity_all_methods() -> None:
    for model in ("ising", "heisenberg"):
        exact = initial_vector(model)
        for method in METHODS:
            st = TrajectoryState(mps=initial_mps(model), vec=initial_vector(model))
            row = compute_metrics(
                exact,
                st.vec,
                state=st,
                model=model,
                method=method,
                chi=CHI_MAIN,
                trotter_step=0,
                time=0.0,
                step_runtime_s=0.0,
            )
            assert row["infidelity"] <= T0_INFIDELITY_TOL


def test_one_step_smoke_all_methods() -> None:
    model = "ising"
    step = build_ising_schedule(timesteps=1)[0]
    exact = apply_trotter_step_dense(initial_vector(model), step)
    for method in METHODS:
        st = TrajectoryState(mps=initial_mps(model), vec=initial_vector(model))
        apply_trotter_step_mps(st, step, method=method, chi=CHI_MAIN)
        st.vec = st.mps.to_vec()
        row = compute_metrics(
            exact,
            st.vec,
            state=st,
            model=model,
            method=method,
            chi=CHI_MAIN,
            trotter_step=1,
            time=DT,
            step_runtime_s=0.0,
        )
        assert not np.isnan(row["infidelity"])
        assert row["infidelity"] >= 0.0


def test_qiskit_circuit_matches_schedule() -> None:
    qc = build_qiskit_circuit("ising", timesteps=2)
    assert qc.num_qubits == NUM_QUBITS
    assert qc.size() == len(build_ising_schedule(timesteps=2)[0].gates) * 2


def test_fixed_resources_outputs_exist() -> None:
    out = OUTPUT_DIR
    if not (out / "results.sqlite").exists():
        pytest.skip("fixed-resource benchmark not run")
    for name in (
        "trajectories.csv",
        "summary.csv",
        "config.json",
        "validation.md",
        "fixed_resources.pdf",
        "fixed_resources.png",
    ):
        assert (out / name).is_file(), name
