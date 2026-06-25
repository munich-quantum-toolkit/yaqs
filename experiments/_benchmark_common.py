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
from typing import TYPE_CHECKING, Any

import numpy as np
from _benchmark_memory import linear_weighted_metrics as _linear_weighted_metrics

from mqt.yaqs.characterization.memory.combs.surrogates.utils import _random_pure_state
from mqt.yaqs.characterization.memory.diagnostics.probe import ProbeSet, sample_split_cut_probes

if TYPE_CHECKING:
    from pathlib import Path

# Paper / legacy experiment defaults.
L_DEFAULT = 2
L_PAPER = 6
K_DEFAULT = 10
K_PAPER = 20
DT_DEFAULT = 0.1
G_DEFAULT = 1.0
BRANCH_WEIGHT_BETA = 1.0
J_SWEEP_PAPER = [0.05 * i for i in range(41)]  # 0.0 ... 2.0

# Fixed-window delayed-break geometry (gap / ell benchmarks).
PAST_LEN_FIXED = 15
FUTURE_LEN_FIXED = 5
ELL_MAX_GAP = 24
ELL_MAX_ELL = 15
SV_THRESHOLD_DEFAULT = 1e-2
PANEL2_FIXED_TAUS: tuple[int, ...] = (0, 1, 2, 4, 8)


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


def load_summary_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def linear_weighted_metrics(
    *,
    probe_set: ProbeSet,
    op: Any,
    sim_params: Any,
    psi0: np.ndarray,
    parallel: bool,
) -> dict[str, float | int]:
    """Past-centered S_V with weighted memory matrix (beta=1) using mc-process branch weights."""
    return _linear_weighted_metrics(
        probe_set=probe_set,
        op=op,
        sim_params=sim_params,
        psi0=psi0,
        parallel=parallel,
        branch_weight_beta=BRANCH_WEIGHT_BETA,
    )


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


__all__ = [
    "BRANCH_WEIGHT_BETA",
    "DT_DEFAULT",
    "ELL_MAX_ELL",
    "ELL_MAX_GAP",
    "FUTURE_LEN_FIXED",
    "G_DEFAULT",
    "J_SWEEP_PAPER",
    "K_DEFAULT",
    "K_PAPER",
    "L_DEFAULT",
    "L_PAPER",
    "PANEL2_FIXED_TAUS",
    "PAST_LEN_FIXED",
    "SV_THRESHOLD_DEFAULT",
    "linear_weighted_metrics",
    "list_initial_states_sys_env0",
    "load_summary_csv",
    "parse_float_list",
    "parse_int_list",
    "sample_probe_set",
    "write_summary_csv",
    "write_summary_json",
]
