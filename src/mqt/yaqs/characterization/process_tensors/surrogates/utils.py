# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Helpers used by surrogate data generation (not the exhaustive process-tensor pipeline)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.encoding import _flatten_choi4_to_real32


def _initial_mcwf_state_from_rho0(
    rho: np.ndarray,
    length: int,
    *,
    rng: np.random.Generator | None = None,
    init_mode: str = "eigenstate",
    return_eig_sample: bool = False,
) -> Any:
    if rho.size != 4:
        msg = "rho must be a 2x2 reduced density matrix."
        raise ValueError(msg)
    rho = np.asarray(rho, dtype=np.complex128).reshape(2, 2)
    rho = 0.5 * (rho + rho.conj().T)
    w, v = np.linalg.eigh(rho)
    w = np.maximum(w.real, 0.0)
    s = float(w.sum())
    w = w / s if s > 1e-15 else np.array([1.0, 0.0], dtype=np.float64)

    if init_mode not in {"eigenstate", "purified"}:
        msg = f"init_mode must be 'eigenstate' or 'purified', got {init_mode!r}"
        raise ValueError(msg)

    if init_mode == "eigenstate":
        if rng is None:
            rng = np.random.default_rng()
        idx = int(rng.choice(2, p=w))
        p = float(w[idx])
        v_idx = v[:, idx].astype(np.complex128)
        if length <= 1:
            psi = v_idx
        else:
            env0 = np.array([1.0, 0.0], dtype=np.complex128)
            env_state = env0
            for _ in range(length - 2):
                env_state = np.kron(env_state, env0)
            psi = np.kron(v_idx, env_state)
        if return_eig_sample:
            return psi, idx, p
        return psi

    if length <= 1:
        psi = np.zeros(2, dtype=np.complex128)
        for i in range(2):
            if w[i] > 1e-15:
                psi += np.sqrt(w[i]) * v[:, i].astype(np.complex128)
        nrm = float(np.linalg.norm(psi))
        psi /= max(nrm, 1e-15)
        return (psi, 0, float(w[0])) if return_eig_sample else psi

    psi_2 = np.zeros(4, dtype=np.complex128)
    for i in range(2):
        if w[i] < 1e-15:
            continue
        anc = np.zeros(2, dtype=np.complex128)
        anc[i] = 1.0
        psi_2 += np.sqrt(w[i]) * np.kron(v[:, i].astype(np.complex128), anc)
    nrm = float(np.linalg.norm(psi_2))
    if nrm < 1e-15:
        psi_2 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    else:
        psi_2 /= nrm
    psi = psi_2
    for _ in range(length - 2):
        psi = np.kron(psi, np.array([1.0, 0.0], dtype=np.complex128))
    return (psi, 0, float(w[0])) if return_eig_sample else psi


def build_initial_psi(
    rho_in: np.ndarray,
    *,
    length: int,
    rng: np.random.Generator,
    init_mode: str,
    return_eig_sample: bool = False,
) -> Any:
    """Pure MCWF state vector on ``length`` qubits whose site-0 reduced state follows ``rho_in``."""
    return _initial_mcwf_state_from_rho0(
        rho_in,
        length,
        rng=rng,
        init_mode=init_mode,
        return_eig_sample=return_eig_sample,
    )


def _random_density_matrix(rng: np.random.Generator) -> np.ndarray:
    """Sample a Haar-inspired random physical 2x2 density matrix (Ginibre / normalize)."""
    a = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    rho = a @ a.conj().T
    tr = float(np.trace(rho).real)
    rho /= max(tr, 1e-15)
    return 0.5 * (rho + rho.conj().T)


def _random_pure_state(rng: np.random.Generator) -> np.ndarray:
    """Sample a Haar-random single-qubit pure state vector |psi> (shape (2,), complex128)."""
    v = rng.standard_normal(2) + 1j * rng.standard_normal(2)
    n = float(np.linalg.norm(v))
    if n < 1e-15:
        return np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    return (v / n).astype(np.complex128)


def _random_rank1_projector(rng: np.random.Generator) -> np.ndarray:
    psi = _random_pure_state(rng)
    return np.outer(psi, psi.conj()).astype(np.complex128)


def _sample_random_intervention(
    rng: np.random.Generator,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """Sample one continuous CP intervention map and its Choi matrix.

    Returns:
        emap: callable map ``emap(rho) = Tr(E @ rho) * rho_prep``
        rho_prep: 2x2 rank-1 density matrix
        E: 2x2 rank-1 measurement effect
        J: 4x4 Choi matrix ``kron(rho_prep, E.T)``
    """
    rho_prep = _random_rank1_projector(rng)
    E = _random_rank1_projector(rng)

    def emap(rho: np.ndarray, rho_prep: np.ndarray = rho_prep, E: np.ndarray = E) -> np.ndarray:
        r = np.asarray(rho, dtype=np.complex128).reshape(2, 2)
        return np.trace(E @ r) * rho_prep

    emap.rho_prep = rho_prep
    emap.effect = E

    J = np.kron(rho_prep, E.T).astype(np.complex128)
    return emap, rho_prep, E, J


def _sample_random_intervention_sequence(
    k: int,
    rng: np.random.Generator,
) -> tuple[list[Any], np.ndarray]:
    """Sample k fresh interventions and return maps + per-step Choi features.

    Returns:
        maps: length-k list of callables ``rho -> Tr(E_t rho) * rho_prep_t``
        choi_features: shape ``(k, 32)``, each row from :func:`~mqt.yaqs.characterization.process_tensors.core.encoding._flatten_choi4_to_real32`
    """
    maps: list[Any] = []
    rows: list[np.ndarray] = []
    for _ in range(int(k)):
        emap, _rho_prep, _E, J = _sample_random_intervention(rng)
        maps.append(emap)
        rows.append(_flatten_choi4_to_real32(J))
    return maps, np.stack(rows, axis=0).astype(np.float32)
