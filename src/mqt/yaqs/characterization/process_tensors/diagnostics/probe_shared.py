"""Build split-cut :class:`ProbeSet` from shared per-slot pools (cut-comparable probes)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .probe import (
    ProbeSet,
    _sample_cut_measurement_only,
    _sample_cut_preparation_only,
    _sample_step,
)


@dataclass(slots=True)
class SharedProbePools:
    """Master pools drawn once; each :class:`ProbeSet` for a given ``cut`` slices the same slot arrays."""

    k: int
    n_pasts: int
    n_futures: int
    full_pool: list[list[tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]]]
    meas_feat_psi: list[list[tuple[np.ndarray, np.ndarray]]]
    prep_feat_psi: list[list[tuple[np.ndarray, np.ndarray]]]


def draw_shared_probe_pools(
    *,
    k: int,
    n_pasts: int,
    n_futures: int,
    rng: np.random.Generator,
) -> SharedProbePools:
    """Draw all pools with a single RNG stream (one call per benchmark configuration)."""
    kk = int(k)
    mx = max(int(n_pasts), int(n_futures))
    full_pool: list[list[tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]]] = [
        [_sample_step(rng) for _ in range(mx)] for _t in range(kk)
    ]
    meas_feat_psi: list[list[tuple[np.ndarray, np.ndarray]]] = [
        [_sample_cut_measurement_only(rng) for _ in range(n_pasts)] for _t in range(kk)
    ]
    prep_feat_psi: list[list[tuple[np.ndarray, np.ndarray]]] = [
        [_sample_cut_preparation_only(rng) for _ in range(n_futures)] for _t in range(kk)
    ]
    return SharedProbePools(
        k=kk,
        n_pasts=int(n_pasts),
        n_futures=int(n_futures),
        full_pool=full_pool,
        meas_feat_psi=meas_feat_psi,
        prep_feat_psi=prep_feat_psi,
    )


def probe_set_from_shared_pools(pools: SharedProbePools, *, cut: int) -> ProbeSet:
    """Assemble a :class:`ProbeSet` for ``cut`` from pre-drawn pools (no additional RNG)."""
    c = int(cut)
    kk = pools.k
    n_pasts = pools.n_pasts
    n_futures = pools.n_futures
    if not (1 <= c <= kk):
        msg = f"cut must satisfy 1 <= cut <= k, got cut={cut}, k={kk}"
        raise ValueError(msg)
    past_full = c - 1
    future_full = kk - c

    past_features = np.empty((n_pasts, past_full + 1, 32), dtype=np.float32)
    past_pairs: list[list[tuple[np.ndarray, np.ndarray]]] = []
    past_cut_meas: list[np.ndarray] = []
    for i in range(n_pasts):
        pairs_i: list[tuple[np.ndarray, np.ndarray]] = []
        for t in range(past_full):
            feat, pair = pools.full_pool[t][i]
            past_features[i, t] = feat
            pairs_i.append(pair)
        feat_m, psi_m = pools.meas_feat_psi[c - 1][i]
        past_features[i, past_full] = feat_m
        past_cut_meas.append(psi_m)
        past_pairs.append(pairs_i)

    future_features = np.empty((n_futures, 1 + future_full, 32), dtype=np.float32)
    future_prep_cut: list[np.ndarray] = []
    future_pairs: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for j in range(n_futures):
        feat_p, psi_p = pools.prep_feat_psi[c - 1][j]
        future_features[j, 0] = feat_p
        future_prep_cut.append(psi_p)
        pairs_j: list[tuple[np.ndarray, np.ndarray]] = []
        for t in range(future_full):
            slot_idx = c + t
            feat, pair = pools.full_pool[slot_idx][j]
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


def build_probe_set_from_shared_pools(
    *,
    cut: int,
    k: int,
    n_pasts: int,
    n_futures: int,
    rng: np.random.Generator,
) -> ProbeSet:
    """Convenience: draw pools and assemble for ``cut`` (one RNG draw; not cut-comparable across calls)."""
    pools = draw_shared_probe_pools(k=k, n_pasts=n_pasts, n_futures=n_futures, rng=rng)
    return probe_set_from_shared_pools(pools, cut=cut)
