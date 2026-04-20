"""Process probing diagnostics (split-cut V-matrix construction and metrics)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

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
    past_pairs: list[list[tuple[np.ndarray, np.ndarray]]]
    past_cut_meas: list[np.ndarray]
    future_prep_cut: list[np.ndarray]
    future_pairs: list[list[tuple[np.ndarray, np.ndarray]]]


class ProbeProcess(Protocol):
    """Minimal backend contract for object-first process probing."""

    def evaluate_probe_set(self, probe_set: ProbeSet) -> np.ndarray:
        """Return packed outputs shaped ``(n_pasts, n_futures, d_out)``."""


def _sample_step(rng: np.random.Generator) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    rho_prep, effect, feat = _sample_random_intervention_parts(rng)
    psi_meas = _psi_from_rank1_projector(effect)
    psi_prep = _psi_from_rank1_projector(rho_prep)
    return feat.astype(np.float32), (psi_meas, psi_prep)


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
) -> ProbeSet:
    """Sample split-cut probe sets (past-measurement and future-preparation at cut)."""
    c = int(cut)
    kk = int(k)
    if not (1 <= c <= kk):
        msg = f"cut must satisfy 1 <= cut <= k, got cut={cut}, k={k}"
        raise ValueError(msg)
    past_full = c - 1
    future_full = kk - c

    past_features = np.empty((n_pasts, past_full + 1, 32), dtype=np.float32)
    past_pairs: list[list[tuple[np.ndarray, np.ndarray]]] = []
    past_cut_meas: list[np.ndarray] = []
    for i in range(n_pasts):
        pairs_i: list[tuple[np.ndarray, np.ndarray]] = []
        for t in range(past_full):
            feat, pair = _sample_step(rng)
            past_features[i, t] = feat
            pairs_i.append(pair)
        feat_m, psi_m = _sample_cut_measurement_only(rng)
        past_features[i, past_full] = feat_m
        past_cut_meas.append(psi_m)
        past_pairs.append(pairs_i)

    future_features = np.empty((n_futures, 1 + future_full, 32), dtype=np.float32)
    future_prep_cut: list[np.ndarray] = []
    future_pairs: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for j in range(n_futures):
        feat_p, psi_p = _sample_cut_preparation_only(rng)
        future_features[j, 0] = feat_p
        future_prep_cut.append(psi_p)
        pairs_j: list[tuple[np.ndarray, np.ndarray]] = []
        for t in range(future_full):
            feat, pair = _sample_step(rng)
            future_features[j, 1 + t] = feat
            pairs_j.append(pair)
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


def build_all_pairs_grid(probe_set: ProbeSet) -> tuple[list[list[tuple[np.ndarray, np.ndarray]]], int, int]:
    """Construct full sequence pair grid from split-cut probes."""
    c = int(probe_set.cut)
    kk = int(probe_set.k)
    n_p = len(probe_set.past_pairs)
    n_f = len(probe_set.future_pairs)
    all_pairs: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for i in range(n_p):
        for j in range(n_f):
            full: list[tuple[np.ndarray, np.ndarray]] = []
            for t in range(c - 1):
                full.append(probe_set.past_pairs[i][t])
            full.append((probe_set.past_cut_meas[i], probe_set.future_prep_cut[j]))
            for t in range(kk - c):
                full.append(probe_set.future_pairs[j][t])
            if len(full) != kk:
                raise RuntimeError("internal: full sequence length mismatch")
            all_pairs.append(full)
    return all_pairs, n_p, n_f


def build_v_matrix(rho8_ij: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_p, n_f, d_out = rho8_ij.shape
    v = rho8_ij.reshape(n_p, n_f * d_out).astype(np.float64)
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

    discarded_weight_threshold = 1e-16
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
    rho8_ij = process.evaluate_probe_set(probe_set).astype(np.float32)
    v, v_centered = build_v_matrix(rho8_ij)
    ana = analyze_v_matrix(v, v_centered)
    out: dict[str, Any] = {
        "rho8_ij": rho8_ij,
        **ana,
        "probe_set": probe_set,
    }
    if return_v:
        out["V"] = v
        out["V_centered"] = v_centered
    return out

