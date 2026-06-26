# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""User-facing intervention specifications for memory characterization."""

# ruff: noqa: PLC2701, ANN202 -- bridges internal probe/surrogate helpers

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np

from mqt.yaqs.characterization.memory.combs.surrogates.utils import (
    sample_intervention_parts,
    sample_intervention_sequence,
)
from mqt.yaqs.characterization.memory.diagnostics.probe import (
    _sample_random_clifford_unitary,
    _sample_random_unitary,
    encode_unitary_choi,
    extract_ket,
)

InterventionKind = Literal["haar", "clifford", "measure_prepare"]
InterventionSlot = str | dict[str, Any]
InterventionSequence = Sequence[InterventionSlot] | InterventionKind


def normalize_kind(kind: str) -> InterventionKind:
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


def map_probe_kwargs(interventions: str) -> dict[str, str]:
    """Map user intervention names to internal split-cut probe keyword arguments.

    Args:
        interventions: ``"haar"``, ``"clifford"``, or ``"measure_prepare"``.

    Returns:
        Dict with ``intervention_mode`` and ``unitary_ensemble`` keys for probing.
    """
    kind = normalize_kind(interventions)
    if kind == "measure_prepare":
        return {"intervention_mode": "measure_prepare", "unitary_ensemble": "haar"}
    ensemble = "clifford" if kind == "clifford" else "haar"
    return {"intervention_mode": "unitary_break_mp", "unitary_ensemble": ensemble}


def _unitary_sampler(kind: InterventionKind, rng: np.random.Generator):
    """Return the unitary sampler callable for an intervention kind.

    Args:
        kind: ``"haar"`` or ``"clifford"``.
        rng: Unused; present for call-site symmetry.

    Returns:
        Callable ``rng -> U``.
    """
    if kind == "clifford":
        return _sample_random_clifford_unitary
    return _sample_random_unitary


def encode_slot(slot: InterventionSlot, rng: np.random.Generator) -> tuple[Any, np.ndarray]:
    """Encode one intervention slot to a simulator step and Choi feature row.

    Args:
        slot: Intervention kind string, ``{"unitary": U}`` dict, or expanded slot.
        rng: NumPy random generator for stochastic slots.

    Returns:
        Tuple ``(step, choi_features)`` where ``step`` is an MP pair or unitary dict.

    Raises:
        ValueError: If a dict slot lacks ``unitary`` or the kind is unsupported.
    """
    if isinstance(slot, dict):
        if "unitary" not in slot:
            msg = "dict intervention slots must contain key 'unitary'."
            raise ValueError(msg)
        u = np.asarray(slot["unitary"], dtype=np.complex128).reshape(2, 2)
        return {"type": "unitary", "U": u}, encode_unitary_choi(u)
    kind = normalize_kind(str(slot))
    if kind == "measure_prepare":
        rho_prep, effect, feat = sample_intervention_parts(rng)
        psi_meas = extract_ket(effect)
        psi_prep = extract_ket(rho_prep)
        return (psi_meas, psi_prep), feat
    u = _unitary_sampler(kind, rng)(rng)
    return {"type": "unitary", "U": u}, encode_unitary_choi(u)


def expand_sequence(
    spec: InterventionSequence,
    *,
    k: int,
    rng: np.random.Generator,
) -> list[InterventionSlot]:
    """Expand a scalar spec or per-slot list to length ``k``.

    Args:
        spec: Per-slot list or scalar intervention kind.
        k: Required sequence length.
        rng: Unused for string specs; reserved for future stochastic expansion.

    Returns:
        List of ``k`` intervention slots.

    Raises:
        ValueError: If an explicit list length does not match ``k``.
    """
    kk = int(k)
    if isinstance(spec, str):
        kind = normalize_kind(spec)
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

    Args:
        spec: Intervention sequence or scalar kind.
        k: Sequence length.
        rng: NumPy random generator.

    Returns:
        Tuple ``(steps, choi_features)`` with ``choi_features`` shaped ``(k, 32)``.
    """
    slots = expand_sequence(spec, k=k, rng=rng)
    steps: list[Any] = []
    rows: list[np.ndarray] = []
    for slot in slots:
        step, feat = encode_slot(slot, rng)
        steps.append(step)
        rows.append(feat)
    return steps, np.stack(rows, axis=0).astype(np.float32)


def sample_train_sequence(
    k: int,
    interventions: InterventionKind,
    rng: np.random.Generator,
) -> tuple[list[Any], np.ndarray]:
    """Sample one training intervention sequence of length ``k``.

    Args:
        k: Sequence length.
        interventions: Intervention kind for all slots.
        rng: NumPy random generator.

    Returns:
        Tuple ``(steps, choi_features)`` suitable for surrogate training rollouts.
    """
    if interventions == "measure_prepare":
        maps, choi = sample_intervention_sequence(int(k), rng)
        steps: list[Any] = []
        for emap in maps:
            psi_meas = extract_ket(emap.effect)
            psi_prep = extract_ket(emap.rho_prep)
            steps.append((psi_meas, psi_prep))
        return steps, choi
    return encode_sequence(interventions, k=int(k), rng=rng)
