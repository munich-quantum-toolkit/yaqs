# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Intervention maps, Choi encoding, sampling, and user-facing sequence specs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, cast

import numpy as np

from .encoding import _flatten_choi4, extract_ket

InterventionStyle = Literal["haar", "clifford", "measure_prepare"]
DEFAULT_INTERVENTION_STYLE: InterventionStyle = "haar"
Intervention = str | dict[str, Any]
InterventionSequence = Sequence[Intervention] | InterventionStyle


@dataclass(frozen=True, slots=True)
class InterventionMap:
    """Rank-1 CP map ``rho -> Tr(effect @ rho) * rho_prep`` with exposed parts."""

    rho_prep: np.ndarray
    effect: np.ndarray

    def __call__(self, rho: np.ndarray) -> np.ndarray:
        """Apply the rank-1 intervention map to a single-qubit density matrix.

        Args:
            rho: ``2 x 2`` density matrix before the map.

        Returns:
            Updated single-qubit density matrix after the rank-1 map.
        """
        r = np.asarray(rho, dtype=np.complex128).reshape(2, 2)
        return np.trace(self.effect @ r) * self.rho_prep


def assemble_choi(rho_prep: np.ndarray, effect: np.ndarray) -> np.ndarray:
    r"""Build the 4x4 Choi matrix for one rank-1 intervention.

    For the continuous surrogate encoding, one timestep intervention is represented by the Choi
    matrix ``J = kron(rho_prep, effect.T)``.

    Args:
        rho_prep: ``2 x 2`` preparation density matrix.
        effect: ``2 x 2`` measurement effect matrix.

    Returns:
        Complex :math:`4\times 4` Choi matrix.
    """
    rp = np.asarray(rho_prep, dtype=np.complex128).reshape(2, 2)
    ef = np.asarray(effect, dtype=np.complex128).reshape(2, 2)
    return np.kron(rp, ef.T).astype(np.complex128)


def encode_choi_features(rho_prep: np.ndarray, effect: np.ndarray) -> np.ndarray:
    """Encode an intervention's Choi matrix into the standard 32-float feature row.

    Args:
        rho_prep: ``2 x 2`` preparation density matrix.
        effect: ``2 x 2`` measurement effect matrix.

    Returns:
        Float32 feature vector of shape ``(32,)``.
    """
    return _flatten_choi4(assemble_choi(rho_prep, effect)).astype(np.float32)


def sample_pure_state(rng: np.random.Generator) -> np.ndarray:
    """Sample a random single-qubit pure state.

    Args:
        rng: Random number generator.

    Returns:
        A normalized state vector of shape ``(2,)`` with dtype complex128.
    """
    v = rng.standard_normal(2) + 1j * rng.standard_normal(2)
    n = float(np.linalg.norm(v))
    if n < 1e-15:
        return np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    return (v / n).astype(np.complex128)


def sample_rank1_projector(rng: np.random.Generator) -> np.ndarray:
    """Sample a random rank-1 projector (pure-state density matrix).

    Args:
        rng: Random number generator.

    Returns:
        A ``2 x 2`` rank-1 density matrix for a random pure state.
    """
    psi = sample_pure_state(rng)
    return np.outer(psi, psi.conj()).astype(np.complex128)


def sample_intervention_parts(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample one continuous intervention as (prep, effect) plus its fused Choi features.

    Args:
        rng: Random number generator.

    Returns:
        Tuple ``(rho_prep, effect, choi_features)`` where ``choi_features`` has shape ``(32,)``.
    """
    rho_prep = sample_rank1_projector(rng)
    effect = sample_rank1_projector(rng)
    feat = encode_choi_features(rho_prep, effect)
    return rho_prep, effect, feat


def _sample_random_intervention(
    rng: np.random.Generator,
) -> tuple[InterventionMap, np.ndarray, np.ndarray, np.ndarray]:
    """Sample one continuous CP intervention map and its Choi matrix.

    Args:
        rng: Random number generator.

    Returns:
        Tuple ``(emap, rho_prep, effect, choi_mat)``.
    """
    rho_prep, effect_mat, _feat = sample_intervention_parts(rng)
    emap = InterventionMap(rho_prep=rho_prep, effect=effect_mat)
    choi_mat = assemble_choi(rho_prep, effect_mat)
    return emap, rho_prep, effect_mat, choi_mat


def sample_intervention_sequence(
    num_interventions: int,
    rng: np.random.Generator,
) -> tuple[list[InterventionMap], np.ndarray]:
    """Sample fresh interventions and return maps + per-step Choi features.

    Args:
        num_interventions: Number of intervention steps.
        rng: Random number generator.

    Returns:
        Tuple ``(maps, choi_features)`` where ``maps`` has length ``num_interventions`` and
        ``choi_features`` has shape ``(num_interventions, 32)``.
    """
    maps: list[InterventionMap] = []
    rows: list[np.ndarray] = []
    for _ in range(int(num_interventions)):
        emap, _rho_prep, _effect, choi_mat = _sample_random_intervention(rng)
        maps.append(emap)
        rows.append(_flatten_choi4(choi_mat))
    return maps, np.stack(rows, axis=0).astype(np.float32)


def _sample_random_unitary(rng: np.random.Generator) -> np.ndarray:
    """Sample a Haar-random ``2 x 2`` unitary.

    Args:
        rng: Random number generator.

    Returns:
        Complex unitary matrix of shape ``(2, 2)``.
    """
    a = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    q, r = np.linalg.qr(a)
    d = np.diag(r)
    phases = np.ones_like(d, dtype=np.complex128)
    nz = np.abs(d) > 1e-15
    phases[nz] = d[nz] / np.abs(d[nz])
    u = q @ np.diag(phases)
    return np.asarray(u, dtype=np.complex128)


@lru_cache(maxsize=1)
def enumerate_clifford_unitaries() -> tuple[np.ndarray, ...]:
    """Enumerate the 24 single-qubit Clifford unitaries (cached).

    Returns:
        Tuple of ``2 x 2`` unitary matrices.
    """
    h = (1.0 / np.sqrt(2.0)) * np.asarray([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
    s = np.asarray([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)
    gens = (h, s)
    eye = np.eye(2, dtype=np.complex128)
    elems: list[np.ndarray] = [eye]
    queue: list[np.ndarray] = [eye]
    while queue:
        u = queue.pop(0)
        for g in gens:
            v = g @ u
            flat = v.reshape(-1)
            idx = int(np.argmax(np.abs(flat)))
            ref = flat[idx]
            if np.abs(ref) > 1e-15:
                v *= np.exp(-1j * np.angle(ref))
            if not any(np.allclose(v, w, atol=1e-12, rtol=0.0) for w in elems):
                elems.append(v)
                queue.append(v)
        if len(elems) >= 24 and not queue:
            break
    return tuple(elems[:24])


def _sample_random_clifford_unitary(rng: np.random.Generator) -> np.ndarray:
    """Sample a uniformly random single-qubit Clifford gate.

    Args:
        rng: Random number generator.

    Returns:
        Complex unitary matrix of shape ``(2, 2)``.
    """
    cliffords = enumerate_clifford_unitaries()
    idx = int(rng.integers(0, len(cliffords)))
    return np.asarray(cliffords[idx], dtype=np.complex128).copy()


def encode_unitary_choi(u: np.ndarray) -> np.ndarray:
    """Encode a unitary as a 32-dimensional Choi feature row.

    Args:
        u: ``2 x 2`` unitary matrix.

    Returns:
        Float32 feature vector of shape ``(32,)``.
    """
    uu = np.asarray(u, dtype=np.complex128).reshape(2, 2)
    vec_u = uu.reshape(4, order="F")
    choi = np.outer(vec_u, vec_u.conj()).astype(np.complex128)
    return _flatten_choi4(choi).astype(np.float32)


def resolve_unitary_sampler(style: str) -> Callable[[np.random.Generator], np.ndarray]:
    """Map a unitary intervention style to a sampling callable.

    Args:
        style: ``"haar"`` or ``"clifford"``.

    Returns:
        Callable ``rng -> U`` that draws a single-qubit unitary.

    Raises:
        ValueError: If ``style`` is not a unitary intervention style.
    """
    resolved = normalize_style(style)
    if resolved == "measure_prepare":
        msg = f"intervention style must be 'haar' or 'clifford' for unitary sampling, got {style!r}."
        raise ValueError(msg)
    return _sample_random_clifford_unitary if resolved == "clifford" else _sample_random_unitary


def _sample_mp(rng: np.random.Generator) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Sample one measure-prepare intervention and its Choi features.

    Args:
        rng: Random number generator.

    Returns:
        Tuple ``(choi_features, (psi_meas, psi_prep))``.
    """
    rho_prep, effect, feat = sample_intervention_parts(rng)
    psi_meas = extract_ket(effect)
    psi_prep = extract_ket(rho_prep)
    return feat.astype(np.float32), (psi_meas, psi_prep)


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
        feat, (psi_meas, psi_prep) = _sample_mp(rng)
        return (psi_meas, psi_prep), feat
    u = resolve_unitary_sampler(resolved)(rng)
    return {"type": "unitary", "U": u}, encode_unitary_choi(u)


def expand_interventions(
    spec: InterventionSequence,
    *,
    num_interventions: int,
    _rng: np.random.Generator,
) -> list[Intervention]:
    """Expand a scalar spec or per-slot list to length ``num_interventions``.

    Args:
        spec: Per-slot list or scalar intervention style.
        num_interventions: Required sequence length.
        _rng: Unused for string specs; reserved for future stochastic expansion.

    Returns:
        List of ``num_interventions`` intervention slots.

    Raises:
        ValueError: If an explicit list length does not match ``num_interventions``.
    """
    if isinstance(spec, str):
        resolved = normalize_style(spec)
        return [resolved] * num_interventions
    slots = list(spec)
    if len(slots) == 1 and num_interventions > 1:
        return [slots[0]] * num_interventions
    if len(slots) != num_interventions:
        msg = f"intervention sequence length must be num_interventions={num_interventions}, got {len(slots)}."
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
        num_interventions: Sequence length.
        rng: NumPy random generator.

    Returns:
        Tuple ``(steps, choi_features)`` with ``choi_features`` shaped ``(num_interventions, 32)``.
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
    """Sample one training intervention sequence of length ``num_interventions``.

    Args:
        num_interventions: Sequence length.
        intervention_style: Intervention style for all slots.
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
