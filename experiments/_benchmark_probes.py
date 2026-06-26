# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Experiment-only split-cut probe geometries (gap / ell paper figures)."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from mqt.yaqs.characterization.memory.operational_memory.samples import (
    ProbeSet,
    _sample_cut_measurement_only,
    _sample_cut_preparation_only,
    _sample_probe_step,
    _sample_step,
    resolve_unitary_sampler,
)

PAST_LEN_FIXED = 15
FUTURE_LEN_FIXED = 5


def sample_split_delayed_break_probes(
    *,
    left_cut: int,
    tau: int,
    k: int,
    n_pasts: int,
    n_futures: int,
    rng: np.random.Generator,
    sigma_ref: np.ndarray | None = None,
    intervention_mode: str = "unitary_break_mp",
    unitary_ensemble: str = "haar",
) -> tuple[ProbeSet, list[list[Any]]]:
    """Delayed causal-break probes: past + break + identity bridge + future."""
    c_left = int(left_cut)
    tt = int(tau)
    kk = int(k)
    if tt < 0:
        msg = f"tau must be >= 0, got {tau}"
        raise ValueError(msg)
    if not (1 <= c_left <= kk):
        msg = f"left_cut must satisfy 1 <= left_cut <= k, got {left_cut}, k={k}"
        raise ValueError(msg)
    c_right = c_left + tt + 1
    if c_right > kk:
        tau_max = max(0, kk - c_left - 1)
        msg = f"tau must satisfy 0 <= tau <= {tau_max}, got {tau}"
        raise ValueError(msg)
    past_full = c_left - 1
    future_tail = kk - c_right

    mode = str(intervention_mode).strip().lower()
    if mode not in {"unitary_break_mp", "measure_prepare"}:
        msg = f"intervention_mode must be 'unitary_break_mp' or 'measure_prepare', got {intervention_mode!r}"
        raise ValueError(msg)
    unitary_sampler = resolve_unitary_sampler(unitary_ensemble)

    past_pairs: list[list[Any]] = []
    past_cut_meas: list[np.ndarray] = []
    for _ in range(n_pasts):
        pairs_i: list[Any] = []
        for _ in range(past_full):
            _feat, step = _sample_probe_step(rng, intervention_mode=mode, unitary_sampler=unitary_sampler)
            pairs_i.append(step)
        _feat_m, psi_m = _sample_cut_measurement_only(rng)
        past_cut_meas.append(psi_m)
        past_pairs.append(pairs_i)

    future_prep_cut: list[np.ndarray] = []
    future_pairs: list[list[Any]] = []
    for _ in range(n_futures):
        _feat_p, psi_p = _sample_cut_preparation_only(rng)
        future_prep_cut.append(psi_p)
        pairs_j: list[Any] = []
        for _ in range(future_tail):
            _feat, step = _sample_probe_step(rng, intervention_mode=mode, unitary_sampler=unitary_sampler)
            pairs_j.append(step)
        future_pairs.append(pairs_j)

    z0 = np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    sigma_ref_vec = z0 if sigma_ref is None else np.asarray(sigma_ref, dtype=np.complex128).reshape(2)
    nrm = float(np.linalg.norm(sigma_ref_vec))
    sigma_ref_vec /= max(nrm, 1e-15)
    u_id = np.eye(2, dtype=np.complex128)
    bridge_common: list[dict[str, Any]] = [{"type": "unitary", "U": u_id} for _ in range(tt)]

    all_pairs: list[list[Any]] = []
    for i in range(n_pasts):
        for j in range(n_futures):
            full: list[Any] = []
            full.extend(past_pairs[i])
            psi_m = np.asarray(past_cut_meas[i], dtype=np.complex128)
            full.append((psi_m, sigma_ref_vec))
            full.extend(bridge_common)
            full.append({"type": "prepare_only", "psi_prep": np.asarray(future_prep_cut[j], dtype=np.complex128)})
            full.extend(future_pairs[j])
            if len(full) != kk:
                msg = f"internal: delayed-break sequence length mismatch, got {len(full)} expected {kk}"
                raise RuntimeError(msg)
            all_pairs.append(full)

    past_features = np.zeros((n_pasts, max(1, past_full + 1), 32), dtype=np.float32)
    future_features = np.zeros((n_futures, max(1, 1 + tt + future_tail), 32), dtype=np.float32)
    return ProbeSet(
        cut=c_left,
        k=kk,
        past_features=past_features,
        future_features=future_features,
        past_pairs=past_pairs,
        past_cut_meas=past_cut_meas,
        future_prep_cut=future_prep_cut,
        future_pairs=future_pairs,
    ), all_pairs


def sample_base_past_future_ensemble(
    *,
    n_pasts: int,
    n_futures: int,
    rng: np.random.Generator,
    past_len: int = PAST_LEN_FIXED,
    future_len: int = FUTURE_LEN_FIXED,
    intervention_mode: str = "unitary_break_mp",
    unitary_ensemble: str = "haar",
) -> tuple[list[list[Any]], list[np.ndarray], list[np.ndarray], list[list[Any]]]:
    """Draw shared past/future ensembles for ell-sweep benchmarks."""
    mode = str(intervention_mode).strip().lower()
    if mode not in {"unitary_break_mp", "measure_prepare"}:
        msg = f"intervention_mode must be 'unitary_break_mp' or 'measure_prepare', got {intervention_mode!r}"
        raise ValueError(msg)
    unitary_sampler = resolve_unitary_sampler(unitary_ensemble)

    past_pairs: list[list[Any]] = []
    past_cut_meas: list[np.ndarray] = []
    for _ in range(n_pasts):
        pairs_i: list[Any] = []
        for _ in range(past_len):
            if mode == "measure_prepare":
                _feat, pair = _sample_step(rng)
                pairs_i.append(pair)
            else:
                pairs_i.append({"type": "unitary", "U": unitary_sampler(rng)})
        _feat_m, psi_m = _sample_cut_measurement_only(rng)
        past_cut_meas.append(psi_m)
        past_pairs.append(pairs_i)

    future_prep_cut: list[np.ndarray] = []
    future_pairs: list[list[Any]] = []
    for _ in range(n_futures):
        _feat_p, psi_p = _sample_cut_preparation_only(rng)
        future_prep_cut.append(psi_p)
        pairs_j: list[Any] = []
        for _ in range(future_len):
            if mode == "measure_prepare":
                _feat, pair = _sample_step(rng)
                pairs_j.append(pair)
            else:
                pairs_j.append({"type": "unitary", "U": unitary_sampler(rng)})
        future_pairs.append(pairs_j)

    return past_pairs, past_cut_meas, future_prep_cut, future_pairs


def probe_set_for_ell(
    *,
    past_pairs: list[list[Any]],
    past_cut_meas: list[np.ndarray],
    future_prep_cut: list[np.ndarray],
    future_pairs: list[list[Any]],
    ell: int,
    past_len: int = PAST_LEN_FIXED,
    future_len: int = FUTURE_LEN_FIXED,
) -> tuple[ProbeSet, list[list[Any]]]:
    """Build a probe set with ``ell`` zero-reset slots after the left break."""
    left_cut = int(past_len + 1)
    ell_i = int(ell)
    k_this = int(past_len + 1 + ell_i + 1 + future_len)
    z0 = np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    n_p = len(past_pairs)
    n_f = len(future_pairs)
    all_pairs: list[list[Any]] = []
    for i in range(n_p):
        for j in range(n_f):
            full: list[Any] = []
            full.extend(copy.deepcopy(past_pairs[i]))
            psi_m = np.asarray(past_cut_meas[i], dtype=np.complex128)
            full.append((psi_m, z0))
            full.extend((z0, z0) for _ in range(ell_i))
            full.append({"type": "prepare_only", "psi_prep": np.asarray(future_prep_cut[j], dtype=np.complex128)})
            full.extend(copy.deepcopy(future_pairs[j]))
            if len(full) != k_this:
                msg = f"internal: sequence length mismatch, got {len(full)} expected {k_this}"
                raise RuntimeError(msg)
            all_pairs.append(full)

    past_features = np.zeros((n_p, max(1, past_len + 1), 32), dtype=np.float32)
    future_features = np.zeros((n_f, max(1, 1 + ell_i + future_len), 32), dtype=np.float32)
    return ProbeSet(
        cut=left_cut,
        k=k_this,
        past_features=past_features,
        future_features=future_features,
        past_pairs=copy.deepcopy(past_pairs),
        past_cut_meas=[np.asarray(x, dtype=np.complex128) for x in past_cut_meas],
        future_prep_cut=[np.asarray(x, dtype=np.complex128) for x in future_prep_cut],
        future_pairs=copy.deepcopy(future_pairs),
    ), all_pairs
