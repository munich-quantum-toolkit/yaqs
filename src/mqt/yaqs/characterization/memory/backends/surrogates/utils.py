# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""MCWF initial-state helpers for surrogate training data generation."""

from __future__ import annotations

import numpy as np


def _initial_mcwf_state_from_rho0(
    rho: np.ndarray,
    length: int,
    *,
    rng: np.random.Generator | None = None,
    init_mode: str = "eigenstate",
    return_eig_sample: bool = False,
) -> np.ndarray | tuple[np.ndarray, int, float]:
    """Construct a pure MCWF state consistent with a given reduced density matrix."""
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
        if int(np.sum(w > 1e-12)) > 1:
            msg = "purified init_mode requires a pure single-qubit state when length <= 1."
            raise ValueError(msg)
        psi = np.zeros(2, dtype=np.complex128)
        for i in range(2):
            if w[i] > 1e-15:
                psi += np.sqrt(w[i]) * v[:, i].astype(np.complex128)
        nrm = float(np.linalg.norm(psi))
        psi /= max(nrm, 1e-15)
        if return_eig_sample:
            if rng is None:
                rng = np.random.default_rng()
            idx = int(rng.choice(2, p=w))
            return psi, idx, float(w[idx])
        return psi

    psi_2 = np.zeros(4, dtype=np.complex128)
    for i in range(2):
        if w[i] < 1e-15:
            continue
        aux_ket = np.zeros(2, dtype=np.complex128)
        aux_ket[i] = 1.0
        psi_2 += np.sqrt(w[i]) * np.kron(v[:, i].astype(np.complex128), aux_ket)
    nrm = float(np.linalg.norm(psi_2))
    if nrm < 1e-15:
        psi_2 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    else:
        psi_2 /= nrm
    psi = psi_2
    for _ in range(length - 2):
        psi = np.kron(psi, np.array([1.0, 0.0], dtype=np.complex128))
    if return_eig_sample:
        if rng is None:
            rng = np.random.default_rng()
        idx = int(rng.choice(2, p=w))
        return psi, idx, float(w[idx])
    return psi


def sample_initial_psi(
    rho_in: np.ndarray,
    *,
    length: int,
    rng: np.random.Generator,
    init_mode: str,
    return_eig_sample: bool = False,
) -> np.ndarray | tuple[np.ndarray, int, float]:
    """Build an initial MCWF pure state for simulation."""
    return _initial_mcwf_state_from_rho0(
        rho_in,
        length,
        rng=rng,
        init_mode=init_mode,
        return_eig_sample=return_eig_sample,
    )


def sample_density_matrix(rng: np.random.Generator) -> np.ndarray:
    """Sample a random physical 2x2 density matrix."""
    a = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    rho = a @ a.conj().T
    tr = float(np.trace(rho).real)
    rho /= max(tr, 1e-15)
    return 0.5 * (rho + rho.conj().T)
