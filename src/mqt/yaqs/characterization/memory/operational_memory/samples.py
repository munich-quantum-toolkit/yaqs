# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Split-cut probe sampling and :class:`ProbeSet`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ....core._validation import validate_integer
from ..shared.encoding import extract_ket
from ..shared.interventions import (
    DEFAULT_INTERVENTION_STYLE,
    InterventionStyle,
    _sample_mp,
    encode_choi_features,
    encode_unitary_choi,
    normalize_style,
    resolve_unitary_sampler,
    sample_intervention_parts,
)


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
    intervention_style: InterventionStyle,
) -> tuple[np.ndarray, Any]:
    """Sample one within-sequence intervention step.

    Args:
        rng: NumPy random generator.
        intervention_style: ``"haar"``, ``"clifford"``, or ``"measure_prepare"``.

    Returns:
        Tuple ``(choi_features, step)`` where ``step`` is an MP pair or unitary dict.
    """
    if intervention_style == "measure_prepare":
        feat, pair = _sample_mp(rng)
        return feat, pair
    u = resolve_unitary_sampler(intervention_style)(rng)
    return encode_unitary_choi(u), {"type": "unitary", "U": u}


def sample_probes(
    *,
    cut: int,
    num_interventions: int,
    n_pasts: int,
    n_futures: int,
    rng: np.random.Generator,
    intervention_style: str = DEFAULT_INTERVENTION_STYLE,
) -> ProbeSet:
    """Sample random split-cut past/future probe ensembles.

    Args:
        cut: Integer causal cut index ``c`` in ``[1, num_interventions]``.
        num_interventions: Positive integer total sequence length.
        n_pasts: Positive integer number of past probe branches.
        n_futures: Positive integer number of future probe branches.
        rng: NumPy random generator.
        intervention_style: ``"haar"``, ``"clifford"``, or ``"measure_prepare"``.

    Returns:
        Populated :class:`ProbeSet`.

    Raises:
        ValueError: If a count is not positive or ``cut`` exceeds ``num_interventions``.
    """
    resolved_num_interventions = validate_integer(num_interventions, name="num_interventions", minimum=1)
    resolved_cut = validate_integer(cut, name="cut", minimum=1)
    resolved_n_pasts = validate_integer(n_pasts, name="n_pasts", minimum=1)
    resolved_n_futures = validate_integer(n_futures, name="n_futures", minimum=1)
    if resolved_cut > resolved_num_interventions:
        msg = (
            "cut must satisfy 1 <= cut <= num_interventions, "
            f"got cut={resolved_cut}, num_interventions={resolved_num_interventions}"
        )
        raise ValueError(msg)
    style = normalize_style(intervention_style)
    past_full = resolved_cut - 1
    future_full = resolved_num_interventions - resolved_cut

    past_features = np.empty((resolved_n_pasts, past_full + 1, 32), dtype=np.float32)
    past_pairs: list[list[Any]] = []
    past_cut_meas: list[np.ndarray] = []
    for i in range(resolved_n_pasts):
        pairs_i: list[Any] = []
        for t in range(past_full):
            feat, step = sample_probe(rng, intervention_style=style)
            past_features[i, t] = feat
            pairs_i.append(step)
        feat_m, psi_m = sample_cut_measurement(rng)
        past_features[i, past_full] = feat_m
        past_cut_meas.append(psi_m)
        past_pairs.append(pairs_i)

    future_features = np.empty((resolved_n_futures, 1 + future_full, 32), dtype=np.float32)
    future_prep_cut: list[np.ndarray] = []
    future_pairs: list[list[Any]] = []
    for j in range(resolved_n_futures):
        feat_p, psi_p = sample_cut_preparation(rng)
        future_features[j, 0] = feat_p
        future_prep_cut.append(psi_p)
        pairs_j: list[Any] = []
        for t in range(future_full):
            feat, step = sample_probe(rng, intervention_style=style)
            future_features[j, 1 + t] = feat
            pairs_j.append(step)
        future_pairs.append(pairs_j)

    return ProbeSet(
        cut=resolved_cut,
        num_interventions=resolved_num_interventions,
        past_features=past_features,
        future_features=future_features,
        past_pairs=past_pairs,
        past_cut_meas=past_cut_meas,
        future_prep_cut=future_prep_cut,
        future_pairs=future_pairs,
    )
