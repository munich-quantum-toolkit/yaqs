# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""User-facing intervention specifications for memory characterization."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

import numpy as np

from mqt.yaqs.characterization.memory.backends.surrogates.utils import (
    sample_intervention_parts,
    sample_intervention_sequence,
)
from mqt.yaqs.characterization.memory.operational_memory.samples import (
    _sample_random_clifford_unitary,  # noqa: PLC2701
    _sample_random_unitary,  # noqa: PLC2701
    encode_unitary_choi,
    extract_ket,
)

InterventionStyle = Literal["haar", "clifford", "measure_prepare"]
DEFAULT_INTERVENTION_STYLE: InterventionStyle = "haar"
Intervention = str | dict[str, Any]
InterventionSequence = Sequence[Intervention] | InterventionStyle


def normalize_style(style: str) -> InterventionStyle:
    """Validate a user intervention style string.

    Args:
        style: ``"haar"``, ``"clifford"``, or ``"measure_prepare"``.

    Returns:
        Normalized intervention style.

    Raises:
        ValueError: If ``style`` is unsupported.
    """
    key = str(style).strip().lower()
    if key in {"haar", "clifford", "measure_prepare"}:
        return cast("InterventionStyle", key)
    msg = f"style must be 'haar', 'clifford', or 'measure_prepare', got {style!r}."
    raise ValueError(msg)


def map_probe_kwargs(style: str) -> dict[str, str]:
    """Map user intervention style to internal split-cut probe keyword arguments.

    Args:
        style: ``"haar"``, ``"clifford"``, or ``"measure_prepare"``.

    Returns:
        Dict with ``intervention_mode`` and ``unitary_ensemble`` keys for probing.
    """
    resolved = normalize_style(style)
    if resolved == "measure_prepare":
        return {"intervention_mode": "measure_prepare", "unitary_ensemble": "haar"}
    ensemble = "clifford" if resolved == "clifford" else "haar"
    return {"intervention_mode": "split_cut_unitary", "unitary_ensemble": ensemble}


def _unitary_sampler(
    intervention_style: InterventionStyle, _rng: np.random.Generator
) -> Callable[[np.random.Generator], np.ndarray]:
    """Return the unitary sampler callable for an intervention style.

    Args:
        style: ``"haar"`` or ``"clifford"``.
        _rng: Unused; present for call-site symmetry.

    Returns:
        Callable ``rng -> U``.
    """
    if intervention_style == "clifford":
        return _sample_random_clifford_unitary
    return _sample_random_unitary


def encode_intervention(slot: Intervention, rng: np.random.Generator) -> tuple[Any, np.ndarray]:
    """Encode one intervention slot to a simulator step and Choi feature row.

    Args:
        slot: Intervention style string, ``{"unitary": U}`` dict, or expanded slot.
        rng: NumPy random generator for stochastic slots.

    Returns:
        Tuple ``(step, choi_features)`` where ``step`` is an MP pair or unitary dict.

    Raises:
        ValueError: If a dict slot lacks ``unitary`` or the style is unsupported.
    """
    if isinstance(slot, dict):
        if "unitary" not in slot:
            msg = "dict intervention slots must contain key 'unitary'."
            raise ValueError(msg)
        u = np.asarray(slot["unitary"], dtype=np.complex128).reshape(2, 2)
        if not np.allclose(u.conj().T @ u, np.eye(2, dtype=np.complex128), atol=1e-8):
            msg = "dict intervention 'unitary' must be a 2x2 unitary matrix."
            raise ValueError(msg)
        return {"type": "unitary", "U": u}, encode_unitary_choi(u)
    resolved = normalize_style(str(slot))
    if resolved == "measure_prepare":
        rho_prep, effect, feat = sample_intervention_parts(rng)
        psi_meas = extract_ket(effect)
        psi_prep = extract_ket(rho_prep)
        return (psi_meas, psi_prep), feat
    u = _unitary_sampler(resolved, rng)(rng)
    return {"type": "unitary", "U": u}, encode_unitary_choi(u)


def expand_interventions(
    spec: InterventionSequence,
    *,
    num_interventions: int,
    _rng: np.random.Generator,
) -> list[Intervention]:
    """Expand a scalar spec or per-slot list to length ``k``.

    Args:
        spec: Per-slot list or scalar intervention style.
        k: Required sequence length.
        _rng: Unused for string specs; reserved for future stochastic expansion.

    Returns:
        List of ``k`` intervention slots.

    Raises:
        ValueError: If an explicit list length does not match ``k``.
    """
    if isinstance(spec, str):
        resolved = normalize_style(spec)
        return [resolved] * num_interventions
    slots = list(spec)
    if len(slots) == 1 and num_interventions > 1:
        return [slots[0]] * num_interventions
    if len(slots) != num_interventions:
        msg = (
            f"intervention sequence length must be num_interventions={num_interventions}, "
            f"got {len(slots)}."
        )
        raise ValueError(msg)
    return slots


def encode_interventions(
    spec: InterventionSequence,
    *,
    num_interventions: int,
    rng: np.random.Generator,
) -> tuple[list[Any], np.ndarray]:
    """Encode a user intervention sequence for simulation or surrogate inference.

    Args:
        spec: Intervention sequence or scalar style.
        k: Sequence length.
        rng: NumPy random generator.

    Returns:
        Tuple ``(steps, choi_features)`` with ``choi_features`` shaped ``(k, 32)``.
    """
    slots = expand_interventions(spec, num_interventions=num_interventions, _rng=rng)
    steps: list[Any] = []
    rows: list[np.ndarray] = []
    for slot in slots:
        step, feat = encode_intervention(slot, rng)
        steps.append(step)
        rows.append(feat)
    return steps, np.stack(rows, axis=0).astype(np.float32)


def sample_train_interventions(
    num_interventions: int,
    intervention_style: InterventionStyle,
    rng: np.random.Generator,
) -> tuple[list[Any], np.ndarray]:
    """Sample one training intervention sequence of length ``k``.

    Args:
        k: Sequence length.
        style: Intervention style for all slots.
        rng: NumPy random generator.

    Returns:
        Tuple ``(steps, choi_features)`` suitable for surrogate training sequences.
    """
    if intervention_style == "measure_prepare":
        maps, choi = sample_intervention_sequence(int(num_interventions), rng)
        steps: list[Any] = []
        for emap in maps:
            psi_meas = extract_ket(emap.effect)
            psi_prep = extract_ket(emap.rho_prep)
            steps.append((psi_meas, psi_prep))
        return steps, choi
    return encode_interventions(intervention_style, num_interventions=int(num_interventions), rng=rng)
