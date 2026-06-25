# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""User-facing intervention specifications for memory characterization."""

# ruff: noqa: PLC2701, DOC201, ANN202 -- bridges internal probe/surrogate helpers

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np

from mqt.yaqs.characterization.memory.combs.surrogates.utils import (
    _sample_random_intervention_parts,
    _sample_random_intervention_sequence,
)
from mqt.yaqs.characterization.memory.diagnostics.probe import (
    _psi_from_rank1_projector,
    _sample_random_clifford_unitary,
    _sample_random_unitary,
    _unitary_to_choi_features,
)

InterventionKind = Literal["haar", "clifford", "measure_prepare"]
InterventionSlot = str | dict[str, Any]
InterventionSequence = Sequence[InterventionSlot] | InterventionKind


def _normalize_intervention_kind(kind: str) -> InterventionKind:
    """Validate a user intervention kind string.

    Args:
        kind: ``"haar"``, ``"clifford"``, or ``"measure_prepare"``.

    Returns:
        Normalized intervention kind.

    Raises:
        ValueError: If ``kind`` is unsupported.
    """
    key = str(kind).strip().lower()
    if key in {"haar", "clifford", "measure_prepare"}:
        return cast("InterventionKind", key)
    msg = f"interventions must be 'haar', 'clifford', or 'measure_prepare', got {kind!r}."
    raise ValueError(msg)


def probe_kwargs_from_interventions(interventions: str) -> dict[str, str]:
    """Map user intervention names to internal split-cut probe keyword arguments."""
    kind = _normalize_intervention_kind(interventions)
    if kind == "measure_prepare":
        return {"intervention_mode": "measure_prepare", "unitary_ensemble": "haar"}
    ensemble = "clifford" if kind == "clifford" else "haar"
    return {"intervention_mode": "unitary_break_mp", "unitary_ensemble": ensemble}


def _unitary_sampler(kind: InterventionKind, rng: np.random.Generator):
    if kind == "clifford":
        return _sample_random_clifford_unitary
    return _sample_random_unitary


def _encode_slot(slot: InterventionSlot, rng: np.random.Generator) -> tuple[Any, np.ndarray]:
    """Encode one intervention slot to a simulator step and Choi feature row.

    Returns:
        Tuple ``(step, choi_features)`` where ``step`` is an MP pair or unitary dict.
    """
    if isinstance(slot, dict):
        if "unitary" not in slot:
            msg = "dict intervention slots must contain key 'unitary'."
            raise ValueError(msg)
        u = np.asarray(slot["unitary"], dtype=np.complex128).reshape(2, 2)
        return {"type": "unitary", "U": u}, _unitary_to_choi_features(u)
    kind = _normalize_intervention_kind(str(slot))
    if kind == "measure_prepare":
        rho_prep, effect, feat = _sample_random_intervention_parts(rng)
        psi_meas = _psi_from_rank1_projector(effect)
        psi_prep = _psi_from_rank1_projector(rho_prep)
        return (psi_meas, psi_prep), feat
    u = _unitary_sampler(kind, rng)(rng)
    return {"type": "unitary", "U": u}, _unitary_to_choi_features(u)


def _expand_intervention_sequence(
    spec: InterventionSequence,
    *,
    k: int,
    rng: np.random.Generator,
) -> list[InterventionSlot]:
    """Expand a scalar spec or per-slot list to length ``k``."""
    kk = int(k)
    if isinstance(spec, str):
        kind = _normalize_intervention_kind(spec)
        return [kind] * kk
    slots = list(spec)
    if len(slots) == 1 and kk > 1:
        return [slots[0]] * kk
    if len(slots) != kk:
        msg = f"intervention sequence length must be k={kk}, got {len(slots)}."
        raise ValueError(msg)
    return slots


def encode_sequence(
    spec: InterventionSequence,
    *,
    k: int,
    rng: np.random.Generator,
) -> tuple[list[Any], np.ndarray]:
    """Encode a user intervention sequence for simulation or surrogate inference.

    Returns:
        ``(psi_pairs, choi_features)`` with ``choi_features`` shaped ``(k, 32)``.
    """
    slots = _expand_intervention_sequence(spec, k=k, rng=rng)
    steps: list[Any] = []
    rows: list[np.ndarray] = []
    for slot in slots:
        step, feat = _encode_slot(slot, rng)
        steps.append(step)
        rows.append(feat)
    return steps, np.stack(rows, axis=0).astype(np.float32)


def _sample_training_sequence(
    k: int,
    interventions: InterventionKind,
    rng: np.random.Generator,
) -> tuple[list[Any], np.ndarray]:
    """Sample one training intervention sequence of length ``k``."""
    if interventions == "measure_prepare":
        maps, choi = _sample_random_intervention_sequence(int(k), rng)
        steps: list[Any] = []
        for emap in maps:
            psi_meas = _psi_from_rank1_projector(emap.effect)
            psi_prep = _psi_from_rank1_projector(emap.rho_prep)
            steps.append((psi_meas, psi_prep))
        return steps, choi
    return encode_sequence(interventions, k=int(k), rng=rng)
