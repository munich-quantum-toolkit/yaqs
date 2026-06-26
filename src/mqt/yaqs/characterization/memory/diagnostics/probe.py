# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Split-cut probes, branch weights, and probe_process orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import numpy as np

from mqt.yaqs.core.parallel_utils import merge_execution_config

from ..combs.core.encoding import _flatten_choi4
from ..combs.surrogates.utils import sample_intervention_parts
from .memory_matrix import (
    assemble_memory_matrix,
    assemble_weighted_matrix_from_probe,
    compute_spectrum,
)

_RHO0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)


@dataclass(slots=True)
class ProbeSet:
    """Sampled split-cut probes for a fixed cut and sequence length.

    Attributes:
        cut: Causal cut index ``c`` (1-based).
        k: Total intervention steps per probe sequence.
        past_features: Choi features for past branches, shape ``(n_pasts, c, 32)``.
        future_features: Choi features for future branches, shape ``(n_futures, k - c + 1, 32)``.
        past_pairs: Intervention steps before the cut (per past index).
        past_cut_meas: Measurement kets at the cut (per past index).
        future_prep_cut: Preparation kets at the cut (per future index).
        future_pairs: Intervention steps after the cut (per future index).
        all_pairs_grid: Optional pre-built full grid (experiments only).
        n_pasts_grid: Grid past count when ``all_pairs_grid`` is set.
        n_futures_grid: Grid future count when ``all_pairs_grid`` is set.
    """

    cut: int
    k: int
    past_features: np.ndarray
    future_features: np.ndarray
    past_pairs: list[list[Any]]
    past_cut_meas: list[np.ndarray]
    future_prep_cut: list[np.ndarray]
    future_pairs: list[list[Any]]
    all_pairs_grid: list[list[Any]] | None = None
    n_pasts_grid: int | None = None
    n_futures_grid: int | None = None


class _ProbeProcess(Protocol):
    """Minimal backend contract for object-first process probing."""

    def evaluate_probes(self, probe_set: ProbeSet) -> np.ndarray:
        """Return probe responses shaped ``(n_pasts, n_futures, 4)`` (Pauli tomography)."""


def extract_ket(projector: np.ndarray) -> np.ndarray:
    """Extract a normalized ket from a rank-one projector.

    Args:
        projector: ``2 x 2`` Hermitian rank-one projector or density matrix.

    Returns:
        Normalized state vector of length 2; falls back to ``|0>`` if degenerate.
    """
    eigvals, eigvecs = np.linalg.eigh(np.asarray(projector, dtype=np.complex128).reshape(2, 2))
    idx = int(np.argmax(eigvals.real))
    psi = eigvecs[:, idx]
    norm = float(np.linalg.norm(psi))
    if norm < 1e-15:
        return np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    return (psi / norm).astype(np.complex128)


def _sample_step(rng: np.random.Generator) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Sample one measure-prepare intervention and its Choi features.

    Args:
        rng: NumPy random generator.

    Returns:
        Tuple ``(choi_features, (psi_meas, psi_prep))``.
    """
    rho_prep, effect, feat = sample_intervention_parts(rng)
    psi_meas = extract_ket(effect)
    psi_prep = extract_ket(rho_prep)
    return feat.astype(np.float32), (psi_meas, psi_prep)


def _sample_random_unitary(rng: np.random.Generator) -> np.ndarray:
    """Sample a Haar-random ``2 x 2`` unitary.

    Args:
        rng: NumPy random generator.

    Returns:
        Complex unitary matrix.
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
def enumerate_clifford_unitaries() -> tuple[np.ndarray, ...]:
    """Return the 24 single-qubit Clifford unitaries (cached).

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
        rng: NumPy random generator.

    Returns:
        Complex unitary matrix.
    """
    cliffords = enumerate_clifford_unitaries()
    idx = int(rng.integers(0, len(cliffords)))
    return np.asarray(cliffords[idx], dtype=np.complex128)


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


def _sample_cut_measurement_only(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample the cut measurement branch (effect only).

    Args:
        rng: NumPy random generator.

    Returns:
        Tuple ``(choi_features, psi_meas)``.
    """
    _rho_prep, effect, feat = sample_intervention_parts(rng)
    psi_meas = extract_ket(effect)
    return feat.astype(np.float32), psi_meas


def _sample_cut_preparation_only(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample the cut preparation branch (state only).

    Args:
        rng: NumPy random generator.

    Returns:
        Tuple ``(choi_features, psi_prep)``.
    """
    rho_prep, _effect, feat = sample_intervention_parts(rng)
    psi_prep = extract_ket(rho_prep)
    return feat.astype(np.float32), psi_prep


def _sample_probe_step(
    rng: np.random.Generator,
    *,
    intervention_mode: str,
    unitary_sampler: Any,
) -> tuple[np.ndarray, Any]:
    """Sample one within-sequence intervention step.

    Args:
        rng: NumPy random generator.
        intervention_mode: ``"measure_prepare"`` or unitary-break mode.
        unitary_sampler: Callable ``rng -> U`` for unitary-break modes.

    Returns:
        Tuple ``(choi_features, step)`` where ``step`` is an MP pair or unitary dict.
    """
    if intervention_mode == "measure_prepare":
        feat, pair = _sample_step(rng)
        return feat, pair
    u = unitary_sampler(rng)
    return encode_unitary_choi(u), {"type": "unitary", "U": u}


def resolve_unitary_sampler(unitary_ensemble: str):
    """Map ensemble name to a unitary sampling callable.

    Args:
        unitary_ensemble: ``"haar"`` or ``"clifford"``.

    Returns:
        Callable ``rng -> U``.

    Raises:
        ValueError: If ``unitary_ensemble`` is unsupported.
    """
    ensemble = str(unitary_ensemble).strip().lower()
    if ensemble not in {"haar", "clifford"}:
        msg = f"unitary_ensemble must be 'haar' or 'clifford', got {unitary_ensemble!r}"
        raise ValueError(msg)
    return _sample_random_unitary if ensemble == "haar" else _sample_random_clifford_unitary


def assemble_probe_sequence(probe_set: ProbeSet, i: int, j: int) -> list[Any]:
    """Build the full intervention sequence for probe-grid entry ``(i, j)``.

    Args:
        probe_set: Sampled split-cut probes.
        i: Past index.
        j: Future index.

    Returns:
        Intervention sequence of length ``probe_set.k``.
    """
    c = int(probe_set.cut)
    kk = int(probe_set.k)
    full: list[Any] = [probe_set.past_pairs[i][t] for t in range(c - 1)]
    full.append((probe_set.past_cut_meas[i], probe_set.future_prep_cut[j]))
    full.extend(probe_set.future_pairs[j][t] for t in range(kk - c))
    return full


def assemble_probe_grid(probe_set: ProbeSet) -> tuple[list[list[Any]], int, int]:
    """Construct the full ``(past, future)`` sequence pair grid.

    Args:
        probe_set: Sampled split-cut probes.

    Returns:
        Tuple ``(all_pairs, n_pasts, n_futures)``.

    Raises:
        RuntimeError: If an assembled sequence length does not match ``k``.
    """
    if probe_set.all_pairs_grid is not None:
        npg = int(probe_set.n_pasts_grid) if probe_set.n_pasts_grid is not None else len(probe_set.past_pairs)
        nfg = int(probe_set.n_futures_grid) if probe_set.n_futures_grid is not None else len(probe_set.future_pairs)
        return probe_set.all_pairs_grid, npg, nfg
    n_p = len(probe_set.past_pairs)
    n_f = len(probe_set.future_pairs)
    kk = int(probe_set.k)
    all_pairs: list[list[Any]] = []
    for i in range(n_p):
        for j in range(n_f):
            full = assemble_probe_sequence(probe_set, i, j)
            if len(full) != kk:
                msg = "internal: full sequence length mismatch"
                raise RuntimeError(msg)
            all_pairs.append(full)
    return all_pairs, n_p, n_f


def compute_born_prob(rho: np.ndarray, psi: np.ndarray) -> float:
    """Return Born probability ``<psi|rho|psi>`` for a rank-one effect.

    Args:
        rho: ``2 x 2`` density matrix.
        psi: Length-2 ket.

    Returns:
        Real probability in ``[0, 1]``.
    """
    r = np.asarray(rho, dtype=np.complex128).reshape(2, 2)
    ket = np.asarray(psi, dtype=np.complex128).reshape(2)
    return float(np.real(np.vdot(ket, r @ ket)))


def _step_probability(rho: np.ndarray, step: Any) -> float:
    """Measurement probability for one intervention step.

    Args:
        rho: ``2 x 2`` state before the step.
        step: MP pair, unitary dict, or structured probe step.

    Returns:
        Step probability used in branch-weight rollout.

    Raises:
        ValueError: If the step type is unsupported.
    """
    if isinstance(step, dict):
        step_type = str(step.get("type", "")).lower()
        if step_type in {"unitary", "depolarizing_pauli", "prepare_only", "reset_only"}:
            return 1.0
        if step_type == "measure_only":
            psi_meas = np.asarray(step["psi_meas"], dtype=np.complex128).reshape(2)
            return compute_born_prob(rho, psi_meas)
        msg = f"Unsupported probe step type: {step_type!r}"
        raise ValueError(msg)
    psi_meas, _ = step
    return compute_born_prob(rho, psi_meas)


def _apply_step(rho: np.ndarray, step: Any) -> np.ndarray:
    """Apply one intervention step to a single-qubit state.

    Args:
        rho: ``2 x 2`` density matrix before the step.
        step: MP pair, unitary dict, or structured probe step.

    Returns:
        Normalized ``2 x 2`` state after the step.

    Raises:
        ValueError: If the step type is unsupported.
    """
    r = np.asarray(rho, dtype=np.complex128).reshape(2, 2)
    if isinstance(step, dict):
        step_type = str(step.get("type", "")).lower()
        if step_type in {"unitary", "depolarizing_pauli"}:
            u = np.asarray(step["U"], dtype=np.complex128).reshape(2, 2)
            out = u @ r @ u.conj().T
        elif step_type == "measure_only":
            z0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
            psi_meas = np.asarray(step["psi_meas"], dtype=np.complex128).reshape(2)
            psi_reset = np.asarray(step.get("psi_reset", z0), dtype=np.complex128).reshape(2)
            prob = compute_born_prob(r, psi_meas)
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
        prob = compute_born_prob(r, psi_meas)
        ket = np.asarray(psi_prep, dtype=np.complex128).reshape(2)
        ket /= max(float(np.linalg.norm(ket)), 1e-15)
        out = np.outer(ket, ket.conj()) if prob > 1e-15 else np.zeros((2, 2), dtype=np.complex128)
    tr = np.trace(out)
    if abs(tr) > 1e-15:
        out /= tr
    return out


def compute_branch_weight(steps: list[Any], *, cut: int) -> float:
    """Analytic branch weight from product of step probabilities up to ``cut``.

    Args:
        steps: Full intervention sequence.
        cut: Causal cut index.

    Returns:
        Cumulative branch weight ``prod_t p_t`` for ``t < cut``.
    """
    rho = _RHO0.copy()
    weight = 1.0
    for t in range(min(int(cut), len(steps))):
        sp = _step_probability(rho, steps[t])
        weight *= sp
        if weight < 1e-15:
            return float(weight)
        rho = _apply_step(rho, steps[t])
    return float(weight)


def compute_branch_weights(probe_set: ProbeSet) -> np.ndarray:
    r"""Analytic branch weights :math:`w_{\alpha,m}` at the causal cut.

    Args:
        probe_set: Sampled split-cut probes.

    Returns:
        Array of shape ``(n_pasts, n_futures)`` constant across future columns per past.
    """
    n_p = len(probe_set.past_pairs)
    n_f = len(probe_set.future_pairs)
    c = int(probe_set.cut)
    w = np.empty((n_p, n_f), dtype=np.float64)
    for i in range(n_p):
        w_i = compute_branch_weight(assemble_probe_sequence(probe_set, i, 0), cut=c)
        w[i, :] = w_i
    return w


def sample_probes(
    *,
    cut: int,
    k: int,
    n_pasts: int,
    n_futures: int,
    rng: np.random.Generator,
    intervention_mode: str = "unitary_break_mp",
    unitary_ensemble: str = "haar",
) -> ProbeSet:
    """Sample random split-cut past/future probe ensembles.

    Args:
        cut: Causal cut index ``c``.
        k: Total sequence length.
        n_pasts: Number of past probe branches.
        n_futures: Number of future probe branches.
        rng: NumPy random generator.
        intervention_mode: ``"unitary_break_mp"`` or ``"measure_prepare"``.
        unitary_ensemble: ``"haar"`` or ``"clifford"`` (unitary-break modes only).

    Returns:
        Populated :class:`ProbeSet`.

    Raises:
        ValueError: If ``cut`` or ``intervention_mode`` is invalid.
    """
    c = int(cut)
    kk = int(k)
    if not (1 <= c <= kk):
        msg = f"cut must satisfy 1 <= cut <= k, got cut={cut}, k={k}"
        raise ValueError(msg)
    mode = str(intervention_mode).strip().lower()
    if mode not in {"unitary_break_mp", "measure_prepare"}:
        msg = f"intervention_mode must be 'unitary_break_mp' or 'measure_prepare', got {intervention_mode!r}"
        raise ValueError(msg)
    unitary_sampler = resolve_unitary_sampler(unitary_ensemble)
    past_full = c - 1
    future_full = kk - c

    past_features = np.empty((n_pasts, past_full + 1, 32), dtype=np.float32)
    past_pairs: list[list[Any]] = []
    past_cut_meas: list[np.ndarray] = []
    for i in range(n_pasts):
        pairs_i: list[Any] = []
        for t in range(past_full):
            feat, step = _sample_probe_step(rng, intervention_mode=mode, unitary_sampler=unitary_sampler)
            past_features[i, t] = feat
            pairs_i.append(step)
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
            feat, step = _sample_probe_step(rng, intervention_mode=mode, unitary_sampler=unitary_sampler)
            future_features[j, 1 + t] = feat
            pairs_j.append(step)
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


def run_probe_diagnostics(
    *,
    process: _ProbeProcess,
    cut: int,
    k: int,
    n_pasts: int = 32,
    n_futures: int = 32,
    rng: np.random.Generator | None = None,
    probe_set: ProbeSet | None = None,
    return_raw: bool = False,
    intervention_mode: str = "unitary_break_mp",
    unitary_ensemble: str = "haar",
    parallel: bool | None = None,
) -> dict[str, Any]:
    """Run split-cut probing and assemble memory-matrix diagnostics.

    Args:
        process: Backend implementing :meth:`evaluate_probes`.
        cut: Causal cut index.
        k: Total sequence length.
        n_pasts: Past probe count when sampling internally.
        n_futures: Future probe count when sampling internally.
        rng: RNG for internal probe sampling.
        probe_set: Pre-sampled probes (optional).
        return_raw: If True, include uncentered ``memory_matrix_raw``.
        intervention_mode: Passed to internal sampling.
        unitary_ensemble: Passed to internal sampling.
        parallel: Override parallelism for exact backends.

    Returns:
        Dict with ``entropy``, ``rank``, ``singular_values``, ``memory_matrix``,
        ``probe_set``, and optional ``weights_ij``.
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
        probe_set = sample_probes(
            cut=cut,
            k=k,
            n_pasts=n_pasts,
            n_futures=n_futures,
            rng=rng,
            intervention_mode=intervention_mode,
            unitary_ensemble=unitary_ensemble,
        )
    weighted = getattr(process, "evaluate_probes_weighted", None)
    if callable(weighted):
        pauli_xyz_ij, weights_ij = weighted(probe_set)
        pauli_xyz_ij = np.asarray(pauli_xyz_ij, dtype=np.float32)
        weights_ij = np.asarray(weights_ij, dtype=np.float64)
        m_raw, memory_matrix = assemble_weighted_matrix_from_probe(pauli_xyz_ij, weights_ij)
    else:
        pauli_xyz_ij = process.evaluate_probes(probe_set).astype(np.float32)
        weights_ij = None
        m_raw, memory_matrix = assemble_memory_matrix(pauli_xyz_ij)
    ana = compute_spectrum(memory_matrix)
    out: dict[str, Any] = {
        "pauli_xyz_ij": pauli_xyz_ij,
        **ana,
        "probe_set": probe_set,
        "memory_matrix": memory_matrix,
    }
    if weights_ij is not None:
        out["weights_ij"] = weights_ij
    if return_raw:
        out["memory_matrix_raw"] = m_raw
    return out
