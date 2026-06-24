# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared helpers for lightweight process-tensor experiment benchmarks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from mqt.yaqs.characterization.memory.reference.exact import evaluate_exact_probe_set_with_diagnostics
from mqt.yaqs.characterization.memory.probing.probe import ProbeSet, analyze_v_matrix, sample_split_cut_probes
from mqt.yaqs.characterization.memory.probing.v_matrix import (
    build_weighted_v_matrix,
    center_past_rows,
    prepare_branch_weights,
)
from mqt.yaqs.characterization.memory.combs.surrogates.utils import _random_pure_state
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

# Paper / legacy experiment defaults (reduced k for basic runs).
L_DEFAULT = 2
K_DEFAULT = 10
DT_DEFAULT = 0.1
G_DEFAULT = 1.0
BRANCH_WEIGHT_BETA = 1.0


def parse_int_list(spec: str) -> list[int]:
    vals = [int(tok.strip()) for tok in spec.split(",") if tok.strip()]
    if not vals:
        msg = "expected at least one integer"
        raise ValueError(msg)
    return sorted(set(vals))


def parse_float_list(spec: str) -> list[float]:
    vals = [float(tok.strip()) for tok in spec.split(",") if tok.strip()]
    if not vals:
        msg = "expected at least one float"
        raise ValueError(msg)
    return vals


def list_initial_states_sys_env0(*, length: int, n_seeds: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Site 0 random pure state; remaining sites in |0⟩."""
    if n_seeds < 1:
        msg = "n_seeds must be >= 1"
        raise ValueError(msg)
    if n_seeds == 1:
        psi = np.zeros(2**length, dtype=np.complex128)
        psi[0] = 1.0 + 0.0j
        return [psi]
    z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    out: list[np.ndarray] = []
    for _ in range(n_seeds):
        psi_sys = _random_pure_state(rng).astype(np.complex128)
        psi = psi_sys
        for _ in range(length - 1):
            psi = np.kron(psi, z)
        nrm = float(np.linalg.norm(psi))
        out.append(psi / max(nrm, 1e-15))
    return out


def linear_weighted_metrics(
    *,
    probe_set: ProbeSet,
    op: MPO,
    sim_params: AnalogSimParams,
    psi0: np.ndarray,
    parallel: bool,
) -> dict[str, float | int]:
    """Past-centered S_V with V = w^β ρ (β=1) using causal cut branch weights."""
    pauli_xyz_ij, weights_ij, _ = evaluate_exact_probe_set_with_diagnostics(
        probe_set=probe_set,
        operator=op,
        sim_params=sim_params,
        initial_psi=psi0,
        parallel=parallel,
    )
    w_clean, _ = prepare_branch_weights(weights_ij, log_warnings=False)
    v_w = build_weighted_v_matrix(pauli_xyz_ij, w_clean, BRANCH_WEIGHT_BETA)
    v_c = center_past_rows(v_w)
    ana = analyze_v_matrix(v_w, v_c)
    return {
        "entropy": float(ana["entropy"]),
        "delta_norm": float(ana["delta_norm"]),
        "rank": int(ana["rank"]),
        "weight_min": float(np.min(weights_ij)),
        "weight_max": float(np.max(weights_ij)),
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def sample_probe_set(
    *,
    cut: int,
    k: int,
    n_pasts: int,
    n_futures: int,
    seed: int,
    unitary_ensemble: str,
    intervention_mode: str = "unitary_break_mp",
) -> ProbeSet:
    rng = np.random.default_rng(int(seed) + 10_000 * int(cut))
    return sample_split_cut_probes(
        cut=int(cut),
        k=int(k),
        n_pasts=int(n_pasts),
        n_futures=int(n_futures),
        rng=rng,
        unitary_ensemble=unitary_ensemble,
        intervention_mode=intervention_mode,
    )
