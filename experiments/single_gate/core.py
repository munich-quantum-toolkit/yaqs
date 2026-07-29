# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Core execution helpers for the main-text single RZZ gate benchmark."""

from __future__ import annotations

import copy
import resource
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from config import GATE_TYPE, Q0, Q1
from gate_runtime import (
    L_DEFAULT,
    DiscardedWeightTracker,
    apply_method,
    apply_two_qubit_dense,
    bond_profile,
    make_dag_node,
    make_gate,
    normalized_state_fidelity,
    param_count_from_profile,
    track_discarded_weight,
)
from variational import VariationalResult, apply_variational_mpo_gate

FIDELITY_DEFINITION = "normalized_state_fidelity_v2"


@dataclass(frozen=True)
class RunResult:
    """Outcome of one gate-application trial."""

    infidelity: float
    fidelity: float
    overlap_squared_raw: float
    norm_squared_exact: float
    norm_squared_approx: float
    fidelity_normalized: float
    infidelity_normalized: float
    norm_loss: float
    fidelity_definition: str
    max_bond: int
    bond_profile: list[int]
    param_count: int
    runtime_s: float
    peak_memory_mb: float
    norm_before: float
    norm_after: float
    discarded_weight: float
    variational_converged: bool | None = None
    variational_failed: bool | None = None
    failure_message: str = ""


def _peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_maxrss) / 1024.0


def exact_reference(initial_vec: np.ndarray, theta: float) -> np.ndarray:
    gate = make_gate(GATE_TYPE, theta, Q0, Q1)
    return apply_two_qubit_dense(initial_vec, L_DEFAULT, Q0, Q1, gate)


def run_method(
    initial_mps,
    initial_vec: np.ndarray,
    *,
    theta: float,
    method: str,
    chi: int,
    substeps: int,
) -> RunResult:
    """Apply one method and compare against the dense exact reference."""
    exact_vec = exact_reference(initial_vec, theta)
    norm_before = float(np.linalg.norm(initial_vec))
    node = make_dag_node(GATE_TYPE, theta, Q0, Q1, L_DEFAULT)
    tracker = DiscardedWeightTracker()
    t0 = time.perf_counter()
    mem0 = _peak_memory_mb()
    vres: VariationalResult | None = None
    failure_message = ""
    if method == "variational_mpo":
        vres = apply_variational_mpo_gate(copy.deepcopy(initial_mps), node, chi=chi)
        state = vres.state
        if vres.failed:
            failure_message = "variational_failed"
        elif not vres.converged:
            failure_message = "variational_not_converged"
        elif vres.failure_reason:
            failure_message = vres.failure_reason
        elif vres.best_initializer:
            failure_message = f"best_init={vres.best_initializer}"
    else:
        with track_discarded_weight(tracker):
            state, _runtime_inner, _dw = apply_method(
                initial_mps, node, method=method, chi=chi, substeps=substeps, tracker=tracker
            )
    runtime = time.perf_counter() - t0
    peak_memory_mb = max(mem0, _peak_memory_mb())
    approx_vec = state.to_vec().astype(np.complex128, copy=False)
    metrics = normalized_state_fidelity(exact_vec, approx_vec)
    prof = bond_profile(state)
    return RunResult(
        infidelity=metrics["infidelity_normalized"],
        fidelity=metrics["fidelity_normalized"],
        overlap_squared_raw=metrics["overlap_squared_raw"],
        norm_squared_exact=metrics["norm_squared_exact"],
        norm_squared_approx=metrics["norm_squared_approx"],
        fidelity_normalized=metrics["fidelity_normalized"],
        infidelity_normalized=metrics["infidelity_normalized"],
        norm_loss=metrics["norm_loss"],
        fidelity_definition=FIDELITY_DEFINITION,
        max_bond=max(prof),
        bond_profile=prof,
        param_count=param_count_from_profile(prof, L_DEFAULT),
        runtime_s=runtime,
        peak_memory_mb=peak_memory_mb,
        norm_before=norm_before,
        norm_after=float(np.linalg.norm(approx_vec)),
        discarded_weight=tracker.per_gate if method != "variational_mpo" else 0.0,
        variational_converged=None if vres is None else vres.converged,
        variational_failed=None if vres is None else vres.failed,
        failure_message=failure_message,
    )


def result_row(
    *,
    task_type: str,
    method: str,
    chi_max: int,
    theta: float,
    x_fraction: float,
    special_angle: bool,
    substeps: int,
    result: RunResult,
) -> dict[str, Any]:
    return {
        "task_type": task_type,
        "method": method,
        "chi_max": chi_max,
        "theta": theta,
        "x_fraction": x_fraction,
        "special_angle": int(special_angle),
        "substeps": substeps,
        "infidelity": result.infidelity,
        "fidelity": result.fidelity,
        "overlap_squared_raw": result.overlap_squared_raw,
        "norm_squared_exact": result.norm_squared_exact,
        "norm_squared_approx": result.norm_squared_approx,
        "fidelity_normalized": result.fidelity_normalized,
        "infidelity_normalized": result.infidelity_normalized,
        "norm_loss": result.norm_loss,
        "fidelity_definition": result.fidelity_definition,
        "max_bond": result.max_bond,
        "bond_profile": result.bond_profile,
        "param_count": result.param_count,
        "runtime_s": result.runtime_s,
        "peak_memory_mb": result.peak_memory_mb,
        "norm_before": result.norm_before,
        "norm_after": result.norm_after,
        "discarded_weight": result.discarded_weight,
        "variational_converged": result.variational_converged,
        "variational_failed": result.variational_failed,
        "failure_message": result.failure_message,
    }
