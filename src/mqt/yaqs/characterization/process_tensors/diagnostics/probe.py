"""Process probing diagnostics (split-cut V-matrix construction and metrics)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import numpy as np

from ..core.encoding import _flatten_choi4_to_real32
from ..surrogates.utils import _sample_random_intervention_parts


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


class ProbeProcess(Protocol):
    """Minimal backend contract for object-first process probing."""

    def evaluate_probe_set(self, probe_set: ProbeSet) -> np.ndarray:
        """Return probe responses shaped ``(n_pasts, n_futures, 3)`` (Pauli :math:`x,y,z`)."""


def _sample_step(rng: np.random.Generator) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    rho_prep, effect, feat = _sample_random_intervention_parts(rng)
    psi_meas = _psi_from_rank1_projector(effect)
    psi_prep = _psi_from_rank1_projector(rho_prep)
    return feat.astype(np.float32), (psi_meas, psi_prep)


def _sample_random_unitary(rng: np.random.Generator) -> np.ndarray:
    """Sample a Haar-like random single-qubit unitary via complex QR."""
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
    """Enumerate the 24 single-qubit Clifford unitaries from generators H and S."""
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
                v = v * np.exp(-1j * np.angle(ref))
            if not any(np.allclose(v, w, atol=1e-12, rtol=0.0) for w in elems):
                elems.append(v)
                queue.append(v)
        if len(elems) >= 24 and not queue:
            break
    return tuple(elems[:24])


def _sample_random_clifford_unitary(rng: np.random.Generator) -> np.ndarray:
    """Sample a uniformly random element from the 24 single-qubit Cliffords."""
    cliffords = _single_qubit_clifford_group()
    idx = int(rng.integers(0, len(cliffords)))
    return np.asarray(cliffords[idx], dtype=np.complex128)


def _sample_depolarizing_pauli_unitary(rng: np.random.Generator) -> np.ndarray:
    """Sample a Pauli unitary for a stochastic unraveling of the depolarizing channel."""
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
    """Encode unitary channel Choi matrix as the standard 32-float row."""
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
    """
    c = int(cut)
    g = int(gap)
    kk = int(k)
    if g < 0:
        raise ValueError(f"gap must be >= 0, got {gap}")
    if not (1 <= c <= kk):
        raise ValueError(f"cut must satisfy 1 <= cut <= k, got cut={cut}, k={k}")
    if c + g + 1 > kk:
        raise ValueError(f"require cut + gap + 1 <= k, got cut={c}, gap={g}, k={kk}")
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
        raise ValueError(f"intervention_mode must be 'unitary_break_mp' or 'measure_prepare', got {intervention_mode!r}")
    ensemble = str(unitary_ensemble).strip().lower()
    if ensemble not in {"haar", "clifford"}:
        raise ValueError(f"unitary_ensemble must be 'haar' or 'clifford', got {unitary_ensemble!r}")
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


def build_all_pairs_grid(probe_set: ProbeSet) -> tuple[list[list[Any]], int, int]:
    """Construct full sequence pair grid from split-cut probes."""
    c = int(probe_set.cut)
    kk = int(probe_set.k)
    n_p = len(probe_set.past_pairs)
    n_f = len(probe_set.future_pairs)
    all_pairs: list[list[Any]] = []
    for i in range(n_p):
        for j in range(n_f):
            full: list[Any] = []
            for t in range(c - 1):
                full.append(probe_set.past_pairs[i][t])
            full.append((probe_set.past_cut_meas[i], probe_set.future_prep_cut[j]))
            for t in range(kk - c):
                full.append(probe_set.future_pairs[j][t])
            if len(full) != kk:
                raise RuntimeError("internal: full sequence length mismatch")
            all_pairs.append(full)
    return all_pairs, n_p, n_f


def build_v_matrix(pauli_xyz_ij: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten Pauli features ``(n_p, n_f, 3)`` into rows of :math:`V` (order preserved)."""
    n_p, n_f, d_out = pauli_xyz_ij.shape
    v = pauli_xyz_ij.reshape(n_p, n_f * d_out).astype(np.float64)
    v_centered = v - v.mean(axis=0, keepdims=True)
    return v, v_centered


def pairwise_row_distances(v: np.ndarray) -> np.ndarray:
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
    fro_v_sq = float(np.linalg.norm(v, ord="fro") ** 2)
    fro_c_sq = float(np.linalg.norm(v_centered, ord="fro") ** 2)
    delta_norm = float(fro_c_sq / fro_v_sq) if fro_v_sq > 0.0 else 0.0

    s_full = np.linalg.svd(v_centered, compute_uv=False).astype(np.float64)
    s = s_full.copy()

    total_weight = float(np.sum(s_full**2))
    discarded_weight = 0.0
    discarded_fraction = 0.0

    discarded_weight_threshold = 0
    if s.size and discarded_weight_threshold is not None and total_weight > 0.0:
        thr = max(float(discarded_weight_threshold), 0.0)
        min_keep_eff = max(1, min(int(min_keep), int(s.size)))

        tail_cumsum = np.cumsum((s_full[::-1] ** 2))
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
        discarded_fraction = (
            discarded_weight / total_weight if total_weight > 0.0 else 0.0
        )

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
) -> dict[str, Any]:
    """Probe a process via split-cut probes and return V-matrix diagnostics."""
    if probe_set is None:
        if rng is None:
            rng = np.random.default_rng()
        probe_set = sample_split_cut_probes(cut=cut, k=k, n_pasts=n_pasts, n_futures=n_futures, rng=rng)
    pauli_xyz_ij = process.evaluate_probe_set(probe_set).astype(np.float32)
    v, v_centered = build_v_matrix(pauli_xyz_ij)
    ana = analyze_v_matrix(v, v_centered)
    out: dict[str, Any] = {
        "pauli_xyz_ij": pauli_xyz_ij,
        **ana,
        "probe_set": probe_set,
    }
    if return_v:
        out["V"] = v
        out["V_centered"] = v_centered
    return out

