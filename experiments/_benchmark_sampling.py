# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Intervention sampling helpers for paper benchmark probe geometries only."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from mqt.yaqs.characterization.memory.combs.surrogates.utils import _sample_random_intervention_parts


def _psi_from_rank1_projector(projector: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(np.asarray(projector, dtype=np.complex128).reshape(2, 2))
    idx = int(np.argmax(eigvals.real))
    psi = eigvecs[:, idx]
    norm = float(np.linalg.norm(psi))
    if norm < 1e-15:
        return np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    return (psi / norm).astype(np.complex128)


def sample_step(rng: np.random.Generator) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    rho_prep, effect, feat = _sample_random_intervention_parts(rng)
    psi_meas = _psi_from_rank1_projector(effect)
    psi_prep = _psi_from_rank1_projector(rho_prep)
    return feat.astype(np.float32), (psi_meas, psi_prep)


def sample_random_unitary(rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    q, r = np.linalg.qr(a)
    d = np.diag(r)
    phases = np.ones_like(d, dtype=np.complex128)
    nz = np.abs(d) > 1e-15
    phases[nz] = d[nz] / np.abs(d[nz])
    u = q @ np.diag(phases.conj())
    return np.asarray(u, dtype=np.complex128)


@lru_cache(maxsize=1)
def _single_qubit_clifford_group() -> tuple[np.ndarray, ...]:
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


def sample_random_clifford_unitary(rng: np.random.Generator) -> np.ndarray:
    cliffords = _single_qubit_clifford_group()
    idx = int(rng.integers(0, len(cliffords)))
    return np.asarray(cliffords[idx], dtype=np.complex128)


def sample_cut_measurement_only(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    _rho_prep, effect, feat = _sample_random_intervention_parts(rng)
    psi_meas = _psi_from_rank1_projector(effect)
    return feat.astype(np.float32), psi_meas


def sample_cut_preparation_only(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    rho_prep, _effect, feat = _sample_random_intervention_parts(rng)
    psi_prep = _psi_from_rank1_projector(rho_prep)
    return feat.astype(np.float32), psi_prep


def sample_probe_step(
    rng: np.random.Generator,
    *,
    intervention_mode: str,
    unitary_sampler: Any,
) -> tuple[np.ndarray, Any]:
    if intervention_mode == "measure_prepare":
        feat, pair = sample_step(rng)
        return feat, pair
    u = unitary_sampler(rng)
    return np.zeros(32, dtype=np.float32), {"type": "unitary", "U": u}


def resolve_unitary_sampler(unitary_ensemble: str):
    ensemble = str(unitary_ensemble).strip().lower()
    if ensemble not in {"haar", "clifford"}:
        msg = f"unitary_ensemble must be 'haar' or 'clifford', got {unitary_ensemble!r}"
        raise ValueError(msg)
    return sample_random_unitary if ensemble == "haar" else sample_random_clifford_unitary
