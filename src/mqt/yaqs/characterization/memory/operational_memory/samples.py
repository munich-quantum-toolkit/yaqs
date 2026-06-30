# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Split-cut probe sampling and :class:`ProbeSet`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..shared.encoding import extract_ket
from ..shared.interventions import (
    _sample_mp,
    encode_choi_features,
    encode_unitary_choi,
    resolve_unitary_sampler,
    sample_intervention_parts,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(slots=True)
class ProbeSet:
    """Sampled split-cut probes for a fixed cut and sequence length.

    Attributes:
        cut: Causal cut index ``c`` (1-based).
        num_interventions: Total intervention steps per probe sequence.
        past_features: Choi features for past branches, shape ``(n_pasts, c, 32)``.
        future_features: Choi features for future branches, shape ``(n_futures, k - c + 1, 32)``.
        past_pairs: Intervention steps before the cut (per past index).
        past_cut_meas: Measurement kets at the cut (per past index).
        future_prep_cut: Preparation kets at the cut (per future index).
        future_pairs: Intervention steps after the cut (per future index).
    """

    cut: int
    num_interventions: int
    past_features: np.ndarray
    future_features: np.ndarray
    past_pairs: list[list[Any]]
    past_cut_meas: list[np.ndarray]
    future_prep_cut: list[np.ndarray]
    future_pairs: list[list[Any]]


def sample_cut_measurement(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample the cut measurement branch (effect only).

    Args:
        rng: NumPy random generator.

    Returns:
        Tuple ``(choi_features, psi_meas)``.
    """
    _rho_prep, effect, _feat = sample_intervention_parts(rng)
    psi_meas = extract_ket(effect)
    feat = encode_choi_features(np.eye(2, dtype=np.complex128) * 0.5, effect)
    return feat.astype(np.float32), psi_meas


def sample_cut_preparation(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample the cut preparation branch (state only).

    Args:
        rng: NumPy random generator.

    Returns:
        Tuple ``(choi_features, psi_prep)``.
    """
    rho_prep, _effect, _feat = sample_intervention_parts(rng)
    psi_prep = extract_ket(rho_prep)
    feat = encode_choi_features(rho_prep, np.eye(2, dtype=np.complex128))
    return feat.astype(np.float32), psi_prep


def sample_probe(
    rng: np.random.Generator,
    *,
    intervention_mode: str,
    unitary_sampler: Callable[[np.random.Generator], np.ndarray] | None,
) -> tuple[np.ndarray, Any]:
    """Sample one within-sequence intervention step.

    Args:
        rng: NumPy random generator.
        intervention_mode: ``"measure_prepare"`` or unitary-break mode.
        unitary_sampler: Callable ``rng -> U`` for unitary-break modes.

    Returns:
        Tuple ``(choi_features, step)`` where ``step`` is an MP pair or unitary dict.

    Raises:
        ValueError: If ``unitary_sampler`` is missing for unitary-break modes.
    """
    if intervention_mode == "measure_prepare":
        feat, pair = _sample_mp(rng)
        return feat, pair
    if unitary_sampler is None:
        msg = "unitary_sampler is required for unitary-break intervention modes."
        raise ValueError(msg)
    u = unitary_sampler(rng)
    return encode_unitary_choi(u), {"type": "unitary", "U": u}


def sample_probes(
    *,
    cut: int,
    num_interventions: int,
    n_pasts: int,
    n_futures: int,
    rng: np.random.Generator,
    intervention_mode: str = "split_cut_unitary",
    unitary_ensemble: str = "haar",
) -> ProbeSet:
    """Sample random split-cut past/future probe ensembles.

    Args:
        cut: Causal cut index ``c``.
        num_interventions: Total sequence length.
        n_pasts: Number of past probe branches.
        n_futures: Number of future probe branches.
        rng: NumPy random generator.
        intervention_mode: ``"split_cut_unitary"`` or ``"measure_prepare"``.
        unitary_ensemble: ``"haar"`` or ``"clifford"`` (unitary-break modes only).

    Returns:
        Populated :class:`ProbeSet`.

    Raises:
        ValueError: If ``cut`` or ``intervention_mode`` is invalid.
    """
    if not (1 <= cut <= num_interventions):
        msg = f"cut must satisfy 1 <= cut <= num_interventions, got cut={cut}, num_interventions={num_interventions}"
        raise ValueError(msg)
    mode = str(intervention_mode).strip().lower()
    if mode not in {"split_cut_unitary", "measure_prepare"}:
        msg = f"intervention_mode must be 'split_cut_unitary' or 'measure_prepare', got {intervention_mode!r}"
        raise ValueError(msg)
    unitary_sampler = resolve_unitary_sampler(unitary_ensemble) if mode == "split_cut_unitary" else None
    past_full = cut - 1
    future_full = num_interventions - cut

    past_features = np.empty((n_pasts, past_full + 1, 32), dtype=np.float32)
    past_pairs: list[list[Any]] = []
    past_cut_meas: list[np.ndarray] = []
    for i in range(n_pasts):
        pairs_i: list[Any] = []
        for t in range(past_full):
            feat, step = sample_probe(rng, intervention_mode=mode, unitary_sampler=unitary_sampler)
            past_features[i, t] = feat
            pairs_i.append(step)
        feat_m, psi_m = sample_cut_measurement(rng)
        past_features[i, past_full] = feat_m
        past_cut_meas.append(psi_m)
        past_pairs.append(pairs_i)

    future_features = np.empty((n_futures, 1 + future_full, 32), dtype=np.float32)
    future_prep_cut: list[np.ndarray] = []
    future_pairs: list[list[Any]] = []
    for j in range(n_futures):
        feat_p, psi_p = sample_cut_preparation(rng)
        future_features[j, 0] = feat_p
        future_prep_cut.append(psi_p)
        pairs_j: list[Any] = []
        for t in range(future_full):
            feat, step = sample_probe(rng, intervention_mode=mode, unitary_sampler=unitary_sampler)
            future_features[j, 1 + t] = feat
            pairs_j.append(step)
        future_pairs.append(pairs_j)

    return ProbeSet(
        cut=cut,
        num_interventions=num_interventions,
        past_features=past_features,
        future_features=future_features,
        past_pairs=past_pairs,
        past_cut_meas=past_cut_meas,
        future_prep_cut=future_prep_cut,
        future_pairs=future_pairs,
    )
