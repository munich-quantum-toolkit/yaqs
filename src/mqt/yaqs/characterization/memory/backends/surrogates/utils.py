# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Helpers used by surrogate data generation (not the exhaustive process-tensor pipeline)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...shared.encoding import _flatten_choi4


@dataclass(frozen=True, slots=True)
class InterventionMap:
    """Rank-1 CP map ``rho -> Tr(effect @ rho) * rho_prep`` with exposed parts."""

    rho_prep: np.ndarray
    effect: np.ndarray

    def __call__(self, rho: np.ndarray) -> np.ndarray:
        """Apply the rank-1 intervention map to a single-qubit density matrix.

        Returns:
            Updated single-qubit density matrix after the rank-1 map.
        """
        r = np.asarray(rho, dtype=np.complex128).reshape(2, 2)
        return np.trace(self.effect @ r) * self.rho_prep


def _initial_mcwf_state_from_rho0(
    rho: np.ndarray,
    length: int,
    *,
    rng: np.random.Generator | None = None,
    init_mode: str = "eigenstate",
    return_eig_sample: bool = False,
) -> np.ndarray | tuple[np.ndarray, int, float]:
    """Construct a pure MCWF state consistent with a given reduced density matrix.

    This helper returns a state vector on ``length`` qubits such that the reduced state on
    site 0 follows ``rho`` under the selected initialization strategy.

    Args:
        rho: Reduced 2x2 density matrix for site 0.
        length: Total number of qubits in the state vector.
        rng: Random number generator used for sampling eigenstates when ``init_mode="eigenstate"``.
        init_mode: Initialization strategy. ``"eigenstate"`` samples an eigenvector of ``rho``;
            ``"purified"`` returns a simple purification-based state.
        return_eig_sample: If ``True``, also return the sampled eigen-index and probability.

    Returns:
        If ``return_eig_sample=False``: complex state vector of shape ``(2**length,)`` (or ``(2,)`` when
        ``length<=1``). If ``return_eig_sample=True``: ``(psi, idx, p)`` where ``idx`` is the chosen
        eigen-index and ``p`` its probability.

    Raises:
        ValueError: If ``rho`` is not 2x2 or if ``init_mode`` is invalid.
    """
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
        return (psi, 0, float(w[0])) if return_eig_sample else psi

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
    return (psi, 0, float(w[0])) if return_eig_sample else psi


def sample_initial_psi(
    rho_in: np.ndarray,
    *,
    length: int,
    rng: np.random.Generator,
    init_mode: str,
    return_eig_sample: bool = False,
) -> np.ndarray | tuple[np.ndarray, int, float]:
    """Build an initial MCWF pure state for simulation.

    Args:
        rho_in: Reduced 2x2 density matrix on site 0.
        length: Total number of qubits in the simulated chain.
        rng: Random number generator used for sampling.
        init_mode: Initialization mode (see :func:`_initial_mcwf_state_from_rho0`).
        return_eig_sample: Whether to return extra eigen-sampling info.

    Returns:
        State vector (and optionally eigen-sampling info), see :func:`_initial_mcwf_state_from_rho0`.
    """
    return _initial_mcwf_state_from_rho0(
        rho_in,
        length,
        rng=rng,
        init_mode=init_mode,
        return_eig_sample=return_eig_sample,
    )


def sample_density_matrix(rng: np.random.Generator) -> np.ndarray:
    """Sample a random physical 2x2 density matrix.

    Args:
        rng: Random number generator.

    Returns:
        A Hermitian, trace-1 2x2 density matrix sampled via a Ginibre construction.
    """
    a = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    rho = a @ a.conj().T
    tr = float(np.trace(rho).real)
    rho /= max(tr, 1e-15)
    return 0.5 * (rho + rho.conj().T)


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
        A 2x2 rank-1 density matrix for a random pure state.
    """
    psi = sample_pure_state(rng)
    return np.outer(psi, psi.conj()).astype(np.complex128)


def _sample_random_intervention(
    rng: np.random.Generator,
) -> tuple[InterventionMap, np.ndarray, np.ndarray, np.ndarray]:
    """Sample one continuous CP intervention map and its Choi matrix.

    Args:
        rng: Random number generator.

    Returns:
        emap: callable map ``emap(rho) = Tr(E @ rho) * rho_prep``
        rho_prep: 2x2 rank-1 density matrix
        E: 2x2 rank-1 measurement effect
        J: 4x4 Choi matrix ``kron(rho_prep, E.T)``
    """
    rho_prep, effect_mat, _feat = sample_intervention_parts(rng)
    emap = InterventionMap(rho_prep=rho_prep, effect=effect_mat)

    choi_mat = assemble_choi(rho_prep, effect_mat)
    return emap, rho_prep, effect_mat, choi_mat


def assemble_choi(rho_prep: np.ndarray, effect: np.ndarray) -> np.ndarray:
    r"""Build the 4x4 Choi matrix for one rank-1 intervention.

    For the continuous surrogate encoding, one timestep intervention is represented by the Choi matrix
    ``J = kron(rho_prep, effect.T)``.

    Returns:
        Complex :math:`4\times 4` Choi matrix.
    """
    rp = np.asarray(rho_prep, dtype=np.complex128).reshape(2, 2)
    ef = np.asarray(effect, dtype=np.complex128).reshape(2, 2)
    return np.kron(rp, ef.T).astype(np.complex128)


def encode_choi_features(rho_prep: np.ndarray, effect: np.ndarray) -> np.ndarray:
    """Encode an intervention's Choi matrix into the standard 32-float feature row.

    Returns:
        Float32 feature vector of shape ``(32,)``.
    """
    return _flatten_choi4(assemble_choi(rho_prep, effect)).astype(np.float32)


def sample_intervention_parts(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample one continuous intervention as (prep, effect) plus its fused Choi features.

    Returns:
        rho_prep: 2x2 rank-1 preparation density matrix.
        effect: 2x2 rank-1 measurement effect (projector).
        choi_features: float32 feature row of shape (32,) for ``kron(rho_prep, effect.T)``.
    """
    rho_prep = sample_rank1_projector(rng)
    effect = sample_rank1_projector(rng)
    feat = encode_choi_features(rho_prep, effect)
    return rho_prep, effect, feat


def sample_intervention_sequence(
    k: int,
    rng: np.random.Generator,
) -> tuple[list[InterventionMap], np.ndarray]:
    """Sample k fresh interventions and return maps + per-step Choi features.

    Args:
        k: Number of intervention steps.
        rng: Random number generator.

    Returns:
        maps: length-k list of callables ``rho -> Tr(E_t rho) * rho_prep_t``
        choi_features: shape ``(k, 32)``, each row from
        :func:`~mqt.yaqs.characterization.memory.shared.encoding._flatten_choi4`
    """
    maps: list[InterventionMap] = []
    rows: list[np.ndarray] = []
    for _ in range(int(k)):
        emap, _rho_prep, _effect, choi_mat = _sample_random_intervention(rng)
        maps.append(emap)
        rows.append(_flatten_choi4(choi_mat))
    return maps, np.stack(rows, axis=0).astype(np.float32)
