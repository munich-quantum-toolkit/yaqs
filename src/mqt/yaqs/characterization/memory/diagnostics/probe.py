# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Split-cut process diagnostics (V-matrix construction and metrics)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import numpy as np

from mqt.yaqs.core.parallel_utils import merge_execution_config

from ..combs.core.encoding import _flatten_choi4_to_real32
from ..combs.surrogates.utils import _sample_random_intervention_parts
from .v_matrix import build_weighted_v_matrix, center_past_rows, prepare_branch_weights


def _psi_from_rank1_projector(projector: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(np.asarray(projector, dtype=np.complex128).reshape(2, 2))
    idx = int(np.argmax(eigvals.real))
    psi = eigvecs[:, idx]
    norm = float(np.linalg.norm(psi))
    if norm < 1e-15:
        return np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    return (psi / norm).astype(np.complex128)


@dataclass(slots=True)
class ProbeSet:
    """Sampled split-cut probes for a fixed cut/sequence length."""

    cut: int
    k: int
    past_features: np.ndarray
    future_features: np.ndarray
    past_pairs: list[list[Any]]
    past_cut_meas: list[np.ndarray]
    future_prep_cut: list[np.ndarray]
    future_pairs: list[list[Any]]
    # Optional explicit sequence grid for non-standard constructions.
    all_pairs_grid: list[list[Any]] | None = None
    n_pasts_grid: int | None = None
    n_futures_grid: int | None = None


class ProbeProcess(Protocol):
    """Minimal backend contract for object-first process probing."""

    def evaluate_probe_set(self, probe_set: ProbeSet) -> np.ndarray:
        """Return probe responses shaped ``(n_pasts, n_futures, 3)`` (Pauli :math:`x,y,z`)."""


def _sample_step(rng: np.random.Generator) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Sample one intervention step feature row and MP pair.

    Returns:
        Tuple ``(choi_features, (psi_meas, psi_prep))``.
    """
    rho_prep, effect, feat = _sample_random_intervention_parts(rng)
    psi_meas = _psi_from_rank1_projector(effect)
    psi_prep = _psi_from_rank1_projector(rho_prep)
    return feat.astype(np.float32), (psi_meas, psi_prep)


def _sample_random_unitary(rng: np.random.Generator) -> np.ndarray:
    r"""Sample a Haar-like random single-qubit unitary via complex QR.

    Returns:
        Complex :math:`2\times 2` unitary matrix.
    """
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
    """Enumerate the 24 single-qubit Clifford unitaries from generators H and S.

    Returns:
        Tuple of 24 single-qubit Clifford unitary matrices.
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
            # normalize global phase using first significant entry
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
    r"""Sample a uniformly random element from the 24 single-qubit Cliffords.

    Returns:
        Complex :math:`2\times 2` Clifford unitary matrix.
    """
    cliffords = _single_qubit_clifford_group()
    idx = int(rng.integers(0, len(cliffords)))
    return np.asarray(cliffords[idx], dtype=np.complex128)


def _sample_depolarizing_pauli_unitary(rng: np.random.Generator) -> np.ndarray:
    r"""Sample a Pauli unitary for a stochastic unraveling of the depolarizing channel.

    Returns:
        One of ``I, X, Y, Z`` as a :math:`2\times 2` unitary.
    """
    idx = int(rng.integers(0, 4))
    if idx == 0:  # I
        return np.eye(2, dtype=np.complex128)
    if idx == 1:  # X
        return np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    if idx == 2:  # Y
        return np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    # Z
    return np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def _unitary_to_choi_features(u: np.ndarray) -> np.ndarray:
    """Encode unitary channel Choi matrix as the standard 32-float row.

    Returns:
        Float32 feature vector of shape ``(32,)``.
    """
    uu = np.asarray(u, dtype=np.complex128).reshape(2, 2)
    vec_u = uu.reshape(4, order="F")
    choi = np.outer(vec_u, vec_u.conj()).astype(np.complex128)
    return _flatten_choi4_to_real32(choi).astype(np.float32)


def _sample_cut_measurement_only(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    _rho_prep, effect, feat = _sample_random_intervention_parts(rng)
    psi_meas = _psi_from_rank1_projector(effect)
    return feat.astype(np.float32), psi_meas


def _sample_cut_preparation_only(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    rho_prep, _effect, feat = _sample_random_intervention_parts(rng)
    psi_prep = _psi_from_rank1_projector(rho_prep)
    return feat.astype(np.float32), psi_prep


def sample_split_cut_probes(
    *,
    cut: int,
    k: int,
    n_pasts: int,
    n_futures: int,
    rng: np.random.Generator,
    intervention_mode: str = "unitary_break_mp",
    unitary_ensemble: str = "haar",
) -> ProbeSet:
    """Sample split-cut probes.

    Modes:
    - ``unitary_break_mp`` (default): random unitary at non-break steps, MP at break.
    - ``measure_prepare``: legacy all-random MP at all steps.

    For ``unitary_break_mp``, ``unitary_ensemble`` controls non-break unitaries:
    - ``haar`` (default): Haar-like random unitary via QR.
    - ``clifford``: uniformly random single-qubit Clifford.

    Returns:
        Probe set with sampled past/future features and pairs.

    Raises:
        ValueError: If ``cut``, ``intervention_mode``, or ``unitary_ensemble`` is invalid.
    """
    c = int(cut)
    kk = int(k)
    if not (1 <= c <= kk):
        msg = f"cut must satisfy 1 <= cut <= k, got cut={cut}, k={k}"
        raise ValueError(msg)
    past_full = c - 1
    future_full = kk - c

    mode = str(intervention_mode).strip().lower()
    if mode not in {"unitary_break_mp", "measure_prepare"}:
        msg = f"intervention_mode must be 'unitary_break_mp' or 'measure_prepare', got {intervention_mode!r}"
        raise ValueError(msg)
    ensemble = str(unitary_ensemble).strip().lower()
    if ensemble not in {"haar", "clifford"}:
        msg = f"unitary_ensemble must be 'haar' or 'clifford', got {unitary_ensemble!r}"
        raise ValueError(msg)
    unitary_sampler = _sample_random_unitary if ensemble == "haar" else _sample_random_clifford_unitary

    past_features = np.empty((n_pasts, past_full + 1, 32), dtype=np.float32)
    past_pairs: list[list[Any]] = []
    past_cut_meas: list[np.ndarray] = []
    for i in range(n_pasts):
        pairs_i: list[Any] = []
        for t in range(past_full):
            if mode == "measure_prepare":
                feat, pair = _sample_step(rng)
                past_features[i, t] = feat
                pairs_i.append(pair)
            else:
                u = unitary_sampler(rng)
                past_features[i, t] = _unitary_to_choi_features(u)
                pairs_i.append({"type": "unitary", "U": u})
        feat_m, psi_m = _sample_cut_measurement_only(rng)
        past_features[i, past_full] = feat_m
        past_cut_meas.append(psi_m)
        past_pairs.append(pairs_i)

    future_features = np.empty((n_futures, 1 + future_full, 32), dtype=np.float32)
    future_prep_cut: list[np.ndarray] = []
    future_pairs: list[list[Any]] = []
    for j in range(n_futures):
        feat_p, psi_p = _sample_cut_preparation_only(rng)
        future_features[j, 0] = feat_p
        future_prep_cut.append(psi_p)
        pairs_j: list[Any] = []
        for t in range(future_full):
            if mode == "measure_prepare":
                feat, pair = _sample_step(rng)
                future_features[j, 1 + t] = feat
                pairs_j.append(pair)
            else:
                u = unitary_sampler(rng)
                future_features[j, 1 + t] = _unitary_to_choi_features(u)
                pairs_j.append({"type": "unitary", "U": u})
        future_pairs.append(pairs_j)

    return ProbeSet(
        cut=c,
        k=kk,
        past_features=past_features,
        future_features=future_features,
        past_pairs=past_pairs,
        past_cut_meas=past_cut_meas,
        future_prep_cut=future_prep_cut,
        future_pairs=future_pairs,
    )


def sample_split_gap_probes(
    *,
    cut: int,
    gap: int,
    k: int,
    n_pasts: int,
    n_futures: int,
    rng: np.random.Generator,
    intervention_mode: str = "unitary_break_mp",
    unitary_ensemble: str = "haar",
) -> ProbeSet:
    """Sample split probes with a depolarizing memory gap after the break.

    ``gap`` inserts that many stochastic depolarizing slots between the break intervention
    and the remaining future region. For ``gap=0`` this reduces exactly to
    :func:`sample_split_cut_probes`.

    Returns:
        Probe set with a depolarizing memory gap after the break.

    Raises:
        ValueError: If geometry or mode parameters are invalid.
    """
    c = int(cut)
    g = int(gap)
    kk = int(k)
    if g < 0:
        msg = f"gap must be >= 0, got {gap}"
        raise ValueError(msg)
    if not (1 <= c <= kk):
        msg = f"cut must satisfy 1 <= cut <= k, got cut={cut}, k={k}"
        raise ValueError(msg)
    if c + g + 1 > kk:
        msg = f"require cut + gap + 1 <= k, got cut={c}, gap={g}, k={kk}"
        raise ValueError(msg)
    if g == 0:
        return sample_split_cut_probes(
            cut=c,
            k=kk,
            n_pasts=n_pasts,
            n_futures=n_futures,
            rng=rng,
            intervention_mode=intervention_mode,
            unitary_ensemble=unitary_ensemble,
        )

    past_full = c - 1
    future_full = kk - c - g
    mode = str(intervention_mode).strip().lower()
    if mode not in {"unitary_break_mp", "measure_prepare"}:
        msg = f"intervention_mode must be 'unitary_break_mp' or 'measure_prepare', got {intervention_mode!r}"
        raise ValueError(
            msg
        )
    ensemble = str(unitary_ensemble).strip().lower()
    if ensemble not in {"haar", "clifford"}:
        msg = f"unitary_ensemble must be 'haar' or 'clifford', got {unitary_ensemble!r}"
        raise ValueError(msg)
    unitary_sampler = _sample_random_unitary if ensemble == "haar" else _sample_random_clifford_unitary

    past_features = np.empty((n_pasts, past_full + 1, 32), dtype=np.float32)
    past_pairs: list[list[Any]] = []
    past_cut_meas: list[np.ndarray] = []
    for i in range(n_pasts):
        pairs_i: list[Any] = []
        for t in range(past_full):
            if mode == "measure_prepare":
                feat, pair = _sample_step(rng)
                past_features[i, t] = feat
                pairs_i.append(pair)
            else:
                u = unitary_sampler(rng)
                past_features[i, t] = _unitary_to_choi_features(u)
                pairs_i.append({"type": "unitary", "U": u})
        feat_m, psi_m = _sample_cut_measurement_only(rng)
        past_features[i, past_full] = feat_m
        past_cut_meas.append(psi_m)
        past_pairs.append(pairs_i)

    future_features = np.empty((n_futures, 1 + g + future_full, 32), dtype=np.float32)
    future_prep_cut: list[np.ndarray] = []
    future_pairs: list[list[Any]] = []
    for j in range(n_futures):
        feat_p, psi_p = _sample_cut_preparation_only(rng)
        future_features[j, 0] = feat_p
        future_prep_cut.append(psi_p)
        pairs_j: list[Any] = []
        # Depolarizing buffer slots.
        for t in range(g):
            u_dep = _sample_depolarizing_pauli_unitary(rng)
            future_features[j, 1 + t] = _unitary_to_choi_features(u_dep)
            pairs_j.append({"type": "unitary", "U": u_dep})
        # Remaining future region.
        for t in range(future_full):
            if mode == "measure_prepare":
                feat, pair = _sample_step(rng)
                future_features[j, 1 + g + t] = feat
                pairs_j.append(pair)
            else:
                u = unitary_sampler(rng)
                future_features[j, 1 + g + t] = _unitary_to_choi_features(u)
                pairs_j.append({"type": "unitary", "U": u})
        future_pairs.append(pairs_j)

    return ProbeSet(
        cut=c,
        k=kk,
        past_features=past_features,
        future_features=future_features,
        past_pairs=past_pairs,
        past_cut_meas=past_cut_meas,
        future_prep_cut=future_prep_cut,
        future_pairs=future_pairs,
    )


def sample_split_symmetric_gap_probes(
    *,
    center_cut: int,
    ell: int,
    k: int,
    n_pasts: int,
    n_futures: int,
    rng: np.random.Generator,
    intervention_mode: str = "unitary_break_mp",
    unitary_ensemble: str = "haar",
) -> ProbeSet:
    """Sample probes for a **hard-reset memory-gap** (symmetric system-erasure block).

    ``ell`` is the symmetric half-width around the midpoint cut; the blocked region has width
    ``2*ell+1`` from ``c_left=center_cut-ell`` to ``c_right=center_cut+ell``. Between the row
    label (left measurement + fixed reset) and the column label (right preparation), the same
    deterministic ``reset_only`` steps are inserted for every ``(i,j)`` — a repeated system clamp,
    not an exact depolarizing channel. For ``ell=0`` this reduces to ordinary split-cut probing at
    ``center_cut``.

    Returns:
        Probe set with a symmetric hard-reset memory gap.

    Raises:
        ValueError: If geometry or mode parameters are invalid.
        RuntimeError: If an internal sequence length mismatch occurs.
    """
    c0 = int(center_cut)
    e = int(ell)
    kk = int(k)
    if e < 0:
        msg = f"ell must be >= 0, got {ell}"
        raise ValueError(msg)
    if not (1 <= c0 <= kk):
        msg = f"center_cut must satisfy 1 <= center_cut <= k, got {center_cut}, k={k}"
        raise ValueError(msg)
    ell_max = min(c0 - 1, kk - c0)
    if e > ell_max:
        msg = f"ell must satisfy 0 <= ell <= {ell_max}, got {ell}"
        raise ValueError(msg)
    if e == 0:
        return sample_split_cut_probes(
            cut=c0,
            k=kk,
            n_pasts=n_pasts,
            n_futures=n_futures,
            rng=rng,
            intervention_mode=intervention_mode,
            unitary_ensemble=unitary_ensemble,
        )

    c_left = c0 - e
    c_right = c0 + e
    # Timesteps strictly between left boundary (measure+reset) and right boundary (prepare).
    blocked_steps = max(0, c_right - c_left - 1)
    past_full = c_left - 1
    future_tail = kk - c_right

    mode = str(intervention_mode).strip().lower()
    if mode not in {"unitary_break_mp", "measure_prepare"}:
        msg = f"intervention_mode must be 'unitary_break_mp' or 'measure_prepare', got {intervention_mode!r}"
        raise ValueError(
            msg
        )
    ensemble = str(unitary_ensemble).strip().lower()
    if ensemble not in {"haar", "clifford"}:
        msg = f"unitary_ensemble must be 'haar' or 'clifford', got {unitary_ensemble!r}"
        raise ValueError(msg)
    unitary_sampler = _sample_random_unitary if ensemble == "haar" else _sample_random_clifford_unitary

    past_pairs: list[list[Any]] = []
    past_cut_meas: list[np.ndarray] = []
    for _i in range(n_pasts):
        pairs_i: list[Any] = []
        for _t in range(past_full):
            if mode == "measure_prepare":
                _feat, pair = _sample_step(rng)
                pairs_i.append(pair)
            else:
                u = unitary_sampler(rng)
                pairs_i.append({"type": "unitary", "U": u})
        _feat_m, psi_m = _sample_cut_measurement_only(rng)
        past_cut_meas.append(psi_m)
        past_pairs.append(pairs_i)

    future_prep_cut: list[np.ndarray] = []
    future_pairs: list[list[Any]] = []
    for _j in range(n_futures):
        _feat_p, psi_p = _sample_cut_preparation_only(rng)
        future_prep_cut.append(psi_p)
        pairs_j: list[Any] = []
        for _t in range(future_tail):
            if mode == "measure_prepare":
                _feat, pair = _sample_step(rng)
                pairs_j.append(pair)
            else:
                u = unitary_sampler(rng)
                pairs_j.append({"type": "unitary", "U": u})
        future_pairs.append(pairs_j)

    # Build explicit sequence grid:
    # past ... ; left boundary (row label) ; symmetric system-erasure block ; right boundary (column label) ; future ...
    all_pairs: list[list[Any]] = []
    z0 = np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    # Common middle block: identical hard resets for every (i, j); no sampling, no row/column dependence.
    erased_block_common: list[dict[str, Any]] = [{"type": "reset_only", "psi_reset": z0} for _ in range(blocked_steps)]
    for i in range(n_pasts):
        for j in range(n_futures):
            full: list[Any] = []
            full.extend(past_pairs[i])
            # Left boundary: measurement-defined row label, then fixed reset independent of row.
            psi_m = np.asarray(past_cut_meas[i], dtype=np.complex128)
            full.append({"type": "measure_only", "psi_meas": psi_m, "psi_reset": z0})
            # Interior erased slots: same channel sequence for all (i, j).
            full.extend(erased_block_common)
            # Right boundary: prepare-only column label (handled by evaluator step semantics).
            full.append({"type": "prepare_only", "psi_prep": np.asarray(future_prep_cut[j], dtype=np.complex128)})
            full.extend(future_pairs[j])
            if len(full) != kk:
                msg = f"internal: symmetric gap sequence length mismatch, got {len(full)} expected {kk}"
                raise RuntimeError(msg)
            all_pairs.append(full)

    # Features are not consumed by exact evaluator, but keep placeholder arrays for consistency.
    past_features = np.zeros((n_pasts, max(1, past_full + 1), 32), dtype=np.float32)
    future_features = np.zeros((n_futures, max(1, 1 + blocked_steps + future_tail), 32), dtype=np.float32)
    return ProbeSet(
        cut=c_left,
        k=kk,
        past_features=past_features,
        future_features=future_features,
        past_pairs=past_pairs,
        past_cut_meas=past_cut_meas,
        future_prep_cut=future_prep_cut,
        future_pairs=future_pairs,
        all_pairs_grid=all_pairs,
        n_pasts_grid=int(n_pasts),
        n_futures_grid=int(n_futures),
    )


def sample_split_delayed_break_probes(
    *,
    left_cut: int,
    tau: int,
    k: int,
    n_pasts: int,
    n_futures: int,
    rng: np.random.Generator,
    sigma_ref: np.ndarray | None = None,
    intervention_mode: str = "unitary_break_mp",
    unitary_ensemble: str = "haar",
) -> ProbeSet:
    """Sample probes for delayed causal-break memory-length benchmark.

    Construction for each full sequence:
    ``past(i) + [left break: (E_m(i), sigma_ref)] + [identity]^tau + [prepare_only sigma_p(j)] + future(j)``.
    The ``tau`` bridge is common to all ``(i,j)`` entries and independent of row/column labels.

    Returns:
        Probe set with a delayed break and optional memory gap.

    Raises:
        ValueError: If geometry or mode parameters are invalid.
        RuntimeError: If an internal sequence length mismatch occurs.
    """
    c_left = int(left_cut)
    tt = int(tau)
    kk = int(k)
    if tt < 0:
        msg = f"tau must be >= 0, got {tau}"
        raise ValueError(msg)
    if not (1 <= c_left <= kk):
        msg = f"left_cut must satisfy 1 <= left_cut <= k, got {left_cut}, k={k}"
        raise ValueError(msg)
    c_right = c_left + tt + 1
    if c_right > kk:
        tau_max = max(0, kk - c_left - 1)
        msg = f"tau must satisfy 0 <= tau <= {tau_max}, got {tau}"
        raise ValueError(msg)
    past_full = c_left - 1
    future_tail = kk - c_right

    mode = str(intervention_mode).strip().lower()
    if mode not in {"unitary_break_mp", "measure_prepare"}:
        msg = f"intervention_mode must be 'unitary_break_mp' or 'measure_prepare', got {intervention_mode!r}"
        raise ValueError(
            msg
        )
    ensemble = str(unitary_ensemble).strip().lower()
    if ensemble not in {"haar", "clifford"}:
        msg = f"unitary_ensemble must be 'haar' or 'clifford', got {unitary_ensemble!r}"
        raise ValueError(msg)
    unitary_sampler = _sample_random_unitary if ensemble == "haar" else _sample_random_clifford_unitary

    past_pairs: list[list[Any]] = []
    past_cut_meas: list[np.ndarray] = []
    for _i in range(n_pasts):
        pairs_i: list[Any] = []
        for _t in range(past_full):
            if mode == "measure_prepare":
                _feat, pair = _sample_step(rng)
                pairs_i.append(pair)
            else:
                u = unitary_sampler(rng)
                pairs_i.append({"type": "unitary", "U": u})
        _feat_m, psi_m = _sample_cut_measurement_only(rng)
        past_cut_meas.append(psi_m)
        past_pairs.append(pairs_i)

    future_prep_cut: list[np.ndarray] = []
    future_pairs: list[list[Any]] = []
    for _j in range(n_futures):
        _feat_p, psi_p = _sample_cut_preparation_only(rng)
        future_prep_cut.append(psi_p)
        pairs_j: list[Any] = []
        for _t in range(future_tail):
            if mode == "measure_prepare":
                _feat, pair = _sample_step(rng)
                pairs_j.append(pair)
            else:
                u = unitary_sampler(rng)
                pairs_j.append({"type": "unitary", "U": u})
        future_pairs.append(pairs_j)

    z0 = np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    sigma_ref_vec = z0 if sigma_ref is None else np.asarray(sigma_ref, dtype=np.complex128).reshape(2)
    nrm = float(np.linalg.norm(sigma_ref_vec))
    sigma_ref_vec /= max(nrm, 1e-15)
    u_id = np.eye(2, dtype=np.complex128)
    bridge_common: list[dict[str, Any]] = [{"type": "unitary", "U": u_id} for _ in range(tt)]

    all_pairs: list[list[Any]] = []
    for i in range(n_pasts):
        for j in range(n_futures):
            full: list[Any] = []
            full.extend(past_pairs[i])
            psi_m = np.asarray(past_cut_meas[i], dtype=np.complex128)
            full.append((psi_m, sigma_ref_vec))
            full.extend(bridge_common)
            full.append({"type": "prepare_only", "psi_prep": np.asarray(future_prep_cut[j], dtype=np.complex128)})
            full.extend(future_pairs[j])
            if len(full) != kk:
                msg = f"internal: delayed-break sequence length mismatch, got {len(full)} expected {kk}"
                raise RuntimeError(msg)
            all_pairs.append(full)

    past_features = np.zeros((n_pasts, max(1, past_full + 1), 32), dtype=np.float32)
    future_features = np.zeros((n_futures, max(1, 1 + tt + future_tail), 32), dtype=np.float32)
    return ProbeSet(
        cut=c_left,
        k=kk,
        past_features=past_features,
        future_features=future_features,
        past_pairs=past_pairs,
        past_cut_meas=past_cut_meas,
        future_prep_cut=future_prep_cut,
        future_pairs=future_pairs,
        all_pairs_grid=all_pairs,
        n_pasts_grid=int(n_pasts),
        n_futures_grid=int(n_futures),
    )


def build_all_pairs_grid(probe_set: ProbeSet) -> tuple[list[list[Any]], int, int]:
    """Construct full sequence pair grid from split-cut probes.

    Returns:
        Tuple ``(all_pairs, n_pasts, n_futures)``.

    Raises:
        RuntimeError: If an internal sequence length mismatch occurs.
    """
    if probe_set.all_pairs_grid is not None:
        npg = int(probe_set.n_pasts_grid) if probe_set.n_pasts_grid is not None else len(probe_set.past_pairs)
        nfg = int(probe_set.n_futures_grid) if probe_set.n_futures_grid is not None else len(probe_set.future_pairs)
        return probe_set.all_pairs_grid, npg, nfg
    c = int(probe_set.cut)
    kk = int(probe_set.k)
    n_p = len(probe_set.past_pairs)
    n_f = len(probe_set.future_pairs)
    all_pairs: list[list[Any]] = []
    for i in range(n_p):
        for j in range(n_f):
            full: list[Any] = [probe_set.past_pairs[i][t] for t in range(c - 1)]
            full.append((probe_set.past_cut_meas[i], probe_set.future_prep_cut[j]))
            full.extend(probe_set.future_pairs[j][t] for t in range(kk - c))
            if len(full) != kk:
                msg = "internal: full sequence length mismatch"
                raise RuntimeError(msg)
            all_pairs.append(full)
    return all_pairs, n_p, n_f


def probe_sequence(probe_set: ProbeSet, i: int, j: int) -> list[Any]:
    """Full intervention sequence for probe-grid entry ``(i, j)``."""
    c = int(probe_set.cut)
    kk = int(probe_set.k)
    full: list[Any] = [probe_set.past_pairs[i][t] for t in range(c - 1)]
    full.append((probe_set.past_cut_meas[i], probe_set.future_prep_cut[j]))
    full.extend(probe_set.future_pairs[j][t] for t in range(kk - c))
    return full


_RHO0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)


def _rank1_prob(rho: np.ndarray, psi: np.ndarray) -> float:
    r = np.asarray(rho, dtype=np.complex128).reshape(2, 2)
    ket = np.asarray(psi, dtype=np.complex128).reshape(2)
    return float(np.real(np.vdot(ket, r @ ket)))


def _step_probability(rho: np.ndarray, step: Any) -> float:
    if isinstance(step, dict):
        step_type = str(step.get("type", "")).lower()
        if step_type in {"unitary", "depolarizing_pauli", "prepare_only", "reset_only"}:
            return 1.0
        if step_type == "measure_only":
            psi_meas = np.asarray(step["psi_meas"], dtype=np.complex128).reshape(2)
            return _rank1_prob(rho, psi_meas)
        msg = f"Unsupported probe step type: {step_type!r}"
        raise ValueError(msg)
    psi_meas, _ = step
    return _rank1_prob(rho, psi_meas)


def _apply_step(rho: np.ndarray, step: Any) -> np.ndarray:
    r = np.asarray(rho, dtype=np.complex128).reshape(2, 2)
    if isinstance(step, dict):
        step_type = str(step.get("type", "")).lower()
        if step_type == "unitary":
            u = np.asarray(step["U"], dtype=np.complex128).reshape(2, 2)
            out = u @ r @ u.conj().T
        elif step_type == "depolarizing_pauli":
            u = np.asarray(step["U"], dtype=np.complex128).reshape(2, 2)
            out = u @ r @ u.conj().T
        elif step_type == "measure_only":
            z0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
            psi_meas = np.asarray(step["psi_meas"], dtype=np.complex128).reshape(2)
            psi_reset = np.asarray(step.get("psi_reset", z0), dtype=np.complex128).reshape(2)
            prob = _rank1_prob(r, psi_meas)
            ket = psi_reset / max(float(np.linalg.norm(psi_reset)), 1e-15)
            out = np.outer(ket, ket.conj()) if prob > 1e-15 else np.zeros((2, 2), dtype=np.complex128)
        elif step_type == "prepare_only":
            psi_prep = np.asarray(step["psi_prep"], dtype=np.complex128).reshape(2)
            ket = psi_prep / max(float(np.linalg.norm(psi_prep)), 1e-15)
            out = np.outer(ket, ket.conj())
        elif step_type == "reset_only":
            psi_r = np.asarray(step["psi_reset"], dtype=np.complex128).reshape(2)
            ket = psi_r / max(float(np.linalg.norm(psi_r)), 1e-15)
            out = np.outer(ket, ket.conj())
        else:
            msg = f"Unsupported probe step type: {step_type!r}"
            raise ValueError(msg)
    else:
        psi_meas, psi_prep = step
        prob = _rank1_prob(r, psi_meas)
        ket = np.asarray(psi_prep, dtype=np.complex128).reshape(2)
        ket = ket / max(float(np.linalg.norm(ket)), 1e-15)
        out = np.outer(ket, ket.conj()) if prob > 1e-15 else np.zeros((2, 2), dtype=np.complex128)
    tr = np.trace(out)
    if abs(tr) > 1e-15:
        out = out / tr
    return out


def rollout_branch_weight(steps: list[Any], *, cut: int) -> float:
    """Product of step probabilities through ``cut`` (inclusive, paper semantics)."""
    rho = _RHO0.copy()
    weight = 1.0
    for t in range(min(int(cut), len(steps))):
        sp = _step_probability(rho, steps[t])
        weight *= sp
        if weight < 1e-15:
            return float(weight)
        rho = _apply_step(rho, steps[t])
    return float(weight)


def branch_weights_ij(probe_set: ProbeSet) -> np.ndarray:
    """Causal cut weights ``w_ij = prod(step_probs[:cut])`` for each probe-grid row.

    Under split-cut sampling the weight is constant across future columns for fixed past row ``i``.
    """
    n_p = len(probe_set.past_pairs)
    n_f = len(probe_set.future_pairs)
    c = int(probe_set.cut)
    w = np.empty((n_p, n_f), dtype=np.float64)
    for i in range(n_p):
        w_i = rollout_branch_weight(probe_sequence(probe_set, i, 0), cut=c)
        w[i, :] = w_i
    return w


def build_weighted_v_from_probe(
    pauli_xyz_ij: np.ndarray,
    weights_ij: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build paper-weighted V with past centering (Eq. 14, beta=1)."""
    w_clean, _ = prepare_branch_weights(weights_ij)
    v = build_weighted_v_matrix(pauli_xyz_ij, w_clean, beta=1.0)
    return v, center_past_rows(v)


def build_v_matrix(pauli_xyz_ij: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten Pauli features ``(n_p, n_f, 3)`` into rows of :math:`V` (order preserved).

    Returns:
        Tuple ``(v_raw, v_centered_past)``.
    """
    n_p, n_f, d_out = pauli_xyz_ij.shape
    v = pauli_xyz_ij.reshape(n_p, n_f * d_out).astype(np.float64)
    v_centered = v - v.mean(axis=0, keepdims=True)
    return v, v_centered


def pairwise_row_distances(v: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances between rows of ``v``.

    Returns:
        Symmetric distance matrix of shape ``(n_rows, n_rows)``.
    """
    n = int(v.shape[0])
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            d[i, j] = float(np.linalg.norm(v[i] - v[j]))
    return d


def analyze_v_matrix(
    v: np.ndarray,
    v_centered: np.ndarray,
    *,
    discarded_weight_threshold: float | None = 1e-12,
    min_keep: int = 1,
) -> dict[str, Any]:
    """Analyze raw and past-centered V matrices.

    Returns:
        Dictionary with Frobenius norms, singular-value spectrum stats, and entropy.
    """
    fro_v_sq = float(np.linalg.norm(v, ord="fro") ** 2)
    fro_c_sq = float(np.linalg.norm(v_centered, ord="fro") ** 2)
    delta_norm = float(fro_c_sq / fro_v_sq) if fro_v_sq > 0.0 else 0.0

    s_full = np.linalg.svd(v_centered, compute_uv=False).astype(np.float64)
    s = s_full.copy()

    total_weight = float(np.sum(s_full**2))
    discarded_weight = 0.0
    discarded_fraction = 0.0

    discarded_weight_threshold = 1e-12
    if s.size and discarded_weight_threshold is not None and total_weight > 0.0:
        thr = max(float(discarded_weight_threshold), 0.0)
        min_keep_eff = max(1, min(int(min_keep), int(s.size)))

        tail_cumsum = np.cumsum(s_full[::-1] ** 2)
        keep = s_full.size

        for idx, tail_weight in enumerate(tail_cumsum, start=1):
            tail_fraction = float(tail_weight / total_weight)
            candidate_keep = s_full.size - idx
            if tail_fraction > thr:
                keep = max(candidate_keep + 1, min_keep_eff)
                break
        else:
            keep = min_keep_eff

        s = s_full[:keep]
        discarded_weight = float(np.sum(s_full[keep:] ** 2))
        discarded_fraction = discarded_weight / total_weight if total_weight > 0.0 else 0.0

    kept_weight = float(np.sum(s**2))
    if kept_weight <= 0.0:
        entropy = 0.0
        p = np.array([], dtype=np.float64)
    else:
        p = (s**2) / kept_weight
        q = np.clip(p, 1e-30, 1.0)
        entropy = float(-np.sum(q * np.log(q)))

    tol_full = 1e-10 * max(1.0, float(s_full[0]) if s_full.size else 1.0)
    rank_full = int(np.sum(s_full > tol_full))

    tol_kept = 1e-10 * max(1.0, float(s[0]) if s.size else 1.0)
    rank_kept = int(np.sum(s > tol_kept))

    dmat = pairwise_row_distances(v)
    tri = dmat[np.triu_indices(dmat.shape[0], k=1)]

    return {
        "delta": fro_c_sq,
        "delta_norm": delta_norm,
        "singular_values": s,
        "singular_values_full": s_full,
        "sv_truncated_count": int(s.size),
        "sv_discarded_count": int(max(0, s_full.size - s.size)),
        "sv_discarded_weight": discarded_weight,
        "sv_discarded_fraction": discarded_fraction,
        "sv_discarded_weight_threshold": (
            None if discarded_weight_threshold is None else float(discarded_weight_threshold)
        ),
        "entropy": entropy,
        "rank": rank_kept,
        "rank_full": rank_full,
        "row_distances": dmat,
        "max_row_distance": float(np.max(tri)) if tri.size else 0.0,
        "mean_row_distance": float(np.mean(tri)) if tri.size else 0.0,
        "median_row_distance": float(np.median(tri)) if tri.size else 0.0,
    }


def probe_process(
    *,
    process: ProbeProcess,
    cut: int,
    k: int,
    n_pasts: int = 32,
    n_futures: int = 32,
    rng: np.random.Generator | None = None,
    probe_set: ProbeSet | None = None,
    return_v: bool = False,
    intervention_mode: str = "unitary_break_mp",
    unitary_ensemble: str = "haar",
    parallel: bool | None = None,
) -> dict[str, Any]:
    """Probe a process via split-cut probes and return V-matrix diagnostics.

    Returns:
        Dictionary with Pauli responses, entropy/spectrum metrics, and the probe set.
        Optionally includes raw ``V`` matrices when ``return_v=True``.
    """
    if parallel is not None:
        from ..reference.exact import ExactProbeProcess

        if isinstance(process, ExactProbeProcess):
            process._execution = merge_execution_config(  # noqa: SLF001
                process._execution,
                parallel=parallel,
            )
    if probe_set is None:
        if rng is None:
            rng = np.random.default_rng()
        probe_set = sample_split_cut_probes(
            cut=cut,
            k=k,
            n_pasts=n_pasts,
            n_futures=n_futures,
            rng=rng,
            intervention_mode=intervention_mode,
            unitary_ensemble=unitary_ensemble,
        )
    weighted = getattr(process, "evaluate_probe_set_with_weights", None)
    if callable(weighted):
        pauli_xyz_ij, weights_ij = weighted(probe_set)
        pauli_xyz_ij = np.asarray(pauli_xyz_ij, dtype=np.float32)
        weights_ij = np.asarray(weights_ij, dtype=np.float64)
        v, v_centered = build_weighted_v_from_probe(pauli_xyz_ij, weights_ij)
    else:
        pauli_xyz_ij = process.evaluate_probe_set(probe_set).astype(np.float32)
        weights_ij = None
        v, v_centered = build_v_matrix(pauli_xyz_ij)
    ana = analyze_v_matrix(v, v_centered)
    out: dict[str, Any] = {
        "pauli_xyz_ij": pauli_xyz_ij,
        **ana,
        "probe_set": probe_set,
    }
    if weights_ij is not None:
        out["weights_ij"] = weights_ij
    if return_v:
        out["V"] = v
        out["V_centered"] = v_centered
    return out
