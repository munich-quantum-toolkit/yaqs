# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exhaustive split-cut probe catalogs from discrete tomography bases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..backends.tomography.basis import TomographyBasis, assemble_fixed_basis
from ..shared.interventions import encode_choi_features
from .samples import ProbeSet


@dataclass(frozen=True, slots=True)
class PastSetting:
    """One conditioned-past label ``(alpha, m)`` at causal cut ``c``."""

    alpha: int
    m_idx: int
    past_pairs: tuple[tuple[Any, ...], ...]
    past_cut_meas: np.ndarray


@dataclass(frozen=True, slots=True)
class FutureSetting:
    """One future label ``(p, beta)`` at causal cut ``c``."""

    p_idx: int
    beta: int
    future_prep_cut: np.ndarray
    future_pairs: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True, slots=True)
class FullProbeCatalog:
    """Exhaustive past/future probe settings for a split-cut geometry.

    Attributes:
        cut: Causal cut index ``c``.
        num_interventions: Horizon ``k``.
        past_settings: Ordered list ``P_full`` of past labels.
        future_settings: Ordered list ``F_full`` of future labels.
        n_intervention_maps: Number of discrete CP maps per leg (16 for tetrahedral).
        n_cut_states: Number of cut preparation/measurement states (4 for tetrahedral).
        intervention_tomographically_complete: Whether the map set spans single-qubit CPTP.
        output_informationally_complete: Whether XYZ tomography spans qubit observables.
    """

    cut: int
    num_interventions: int
    past_settings: tuple[PastSetting, ...]
    future_settings: tuple[FutureSetting, ...]
    n_intervention_maps: int
    n_cut_states: int
    intervention_tomographically_complete: bool
    output_informationally_complete: bool


def _mp_step_from_index(
    alpha: int,
    basis_set: list[tuple[str, np.ndarray, np.ndarray]],
    choi_indices: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a discrete Choi index to a measure/prepare pair step."""
    prep_idx, meas_idx = choi_indices[int(alpha)]
    psi_meas = np.asarray(basis_set[int(meas_idx)][1], dtype=np.complex128).copy()
    psi_prep = np.asarray(basis_set[int(prep_idx)][1], dtype=np.complex128).copy()
    return psi_meas, psi_prep


def enumerate_full_probe_catalog(
    *,
    cut: int,
    num_interventions: int,
    basis: TomographyBasis | str = "tetrahedral",
    basis_seed: int | None = None,
) -> FullProbeCatalog:
    """Enumerate ``P_full`` and ``F_full`` from the discrete tomography alphabet.

    Each within-sequence leg uses one of 16 rank-1 CP maps from the tetrahedral
    (or standard) four-state preparation/measurement product. Cut measurement and
    preparation kets are drawn from the same four reference states.

    Args:
        cut: Causal cut index ``c``.
        num_interventions: Horizon ``k``.
        basis: Tomography basis name.
        basis_seed: Optional seed when ``basis="random"``.

    Returns:
        Catalog with ``len(P_full) = 16**(c-1) * 4`` and ``len(F_full) = 4 * 16**(k-c)``.

    Raises:
        ValueError: If ``cut`` is invalid.
    """
    if not (1 <= cut <= num_interventions):
        msg = (
            f"cut must satisfy 1 <= cut <= num_interventions, got cut={cut}, "
            f"num_interventions={num_interventions}"
        )
        raise ValueError(msg)
    basis_set, _choi_mats, choi_indices, _feat = assemble_fixed_basis(
        basis=basis,
        basis_seed=basis_seed,
    )
    n_maps = len(choi_indices)
    n_states = len(basis_set)
    past_full = cut - 1
    future_full = num_interventions - cut

    past_settings: list[PastSetting] = []
    if past_full == 0:
        for m_idx in range(n_states):
            past_settings.append(
                PastSetting(
                    alpha=-1,
                    m_idx=m_idx,
                    past_pairs=(),
                    past_cut_meas=np.asarray(basis_set[m_idx][1], dtype=np.complex128).copy(),
                )
            )
    else:
        from itertools import product

        alpha_axes = [range(n_maps)] * past_full
        for alpha_tuple in product(*alpha_axes):
            alpha_label = int(alpha_tuple[0]) if past_full == 1 else int(sum(a * n_maps**i for i, a in enumerate(alpha_tuple)))
            pairs = tuple(_mp_step_from_index(a, basis_set, choi_indices) for a in alpha_tuple)
            for m_idx in range(n_states):
                past_settings.append(
                    PastSetting(
                        alpha=alpha_label,
                        m_idx=m_idx,
                        past_pairs=pairs,
                        past_cut_meas=np.asarray(basis_set[m_idx][1], dtype=np.complex128).copy(),
                    )
                )

    future_settings: list[FutureSetting] = []
    if future_full == 0:
        for p_idx in range(n_states):
            future_settings.append(
                FutureSetting(
                    p_idx=p_idx,
                    beta=-1,
                    future_prep_cut=np.asarray(basis_set[p_idx][1], dtype=np.complex128).copy(),
                    future_pairs=(),
                )
            )
    else:
        from itertools import product

        beta_axes = [range(n_maps)] * future_full
        for p_idx in range(n_states):
            for beta_tuple in product(*beta_axes):
                beta_label = int(beta_tuple[0]) if future_full == 1 else int(
                    sum(b * n_maps**i for i, b in enumerate(beta_tuple))
                )
                pairs = tuple(_mp_step_from_index(b, basis_set, choi_indices) for b in beta_tuple)
                future_settings.append(
                    FutureSetting(
                        p_idx=p_idx,
                        beta=beta_label,
                        future_prep_cut=np.asarray(basis_set[p_idx][1], dtype=np.complex128).copy(),
                        future_pairs=pairs,
                    )
                )

    return FullProbeCatalog(
        cut=int(cut),
        num_interventions=int(num_interventions),
        past_settings=tuple(past_settings),
        future_settings=tuple(future_settings),
        n_intervention_maps=n_maps,
        n_cut_states=n_states,
        intervention_tomographically_complete=n_maps >= 16,
        output_informationally_complete=True,
    )


def build_probe_set_from_catalog(
    catalog: FullProbeCatalog,
    past_indices: np.ndarray | list[int],
    future_indices: np.ndarray | list[int],
) -> ProbeSet:
    """Build a :class:`ProbeSet` from catalog rows selected by index.

    Args:
        catalog: Exhaustive probe catalog.
        past_indices: Row indices into ``catalog.past_settings``.
        future_indices: Row indices into ``catalog.future_settings``.

    Returns:
        Probe set whose past/future branches are nested subsets of ``P_full`` / ``F_full``.
    """
    past_idx = [int(i) for i in past_indices]
    future_idx = [int(i) for i in future_indices]
    n_pasts = len(past_idx)
    n_futures = len(future_idx)
    past_full = catalog.cut - 1
    future_full = catalog.num_interventions - catalog.cut

    past_features = np.empty((n_pasts, past_full + 1, 32), dtype=np.float32)
    past_pairs: list[list[Any]] = []
    past_cut_meas: list[np.ndarray] = []
    for row, i in enumerate(past_idx):
        setting = catalog.past_settings[i]
        pairs_i: list[Any] = [pair for pair in setting.past_pairs]
        for t, step in enumerate(pairs_i):
            psi_m, psi_p = step
            past_features[row, t] = encode_choi_features(
                np.outer(psi_p, psi_p.conj()),
                np.outer(psi_m, psi_m.conj()),
            )
        cut_slot = past_full if past_full else 0
        past_features[row, cut_slot] = encode_choi_features(
            np.eye(2, dtype=np.complex128) * 0.5,
            np.outer(setting.past_cut_meas, setting.past_cut_meas.conj()),
        )
        past_cut_meas.append(setting.past_cut_meas.copy())
        past_pairs.append(pairs_i)

    future_features = np.empty((n_futures, 1 + future_full, 32), dtype=np.float32)
    future_prep_cut: list[np.ndarray] = []
    future_pairs: list[list[Any]] = []
    for row, j in enumerate(future_idx):
        setting = catalog.future_settings[j]
        feat_p = encode_choi_features(
            np.outer(setting.future_prep_cut, setting.future_prep_cut.conj()),
            np.eye(2, dtype=np.complex128),
        )
        future_features[row, 0] = feat_p
        future_prep_cut.append(setting.future_prep_cut.copy())
        pairs_j = [pair for pair in setting.future_pairs]
        for t, step in enumerate(pairs_j):
            psi_m, psi_p = step
            future_features[row, 1 + t] = encode_choi_features(
                np.outer(psi_p, psi_p.conj()),
                np.outer(psi_m, psi_m.conj()),
            )
        future_pairs.append(pairs_j)

    return ProbeSet(
        cut=catalog.cut,
        num_interventions=catalog.num_interventions,
        past_features=past_features,
        future_features=future_features,
        past_pairs=past_pairs,
        past_cut_meas=past_cut_meas,
        future_prep_cut=future_prep_cut,
        future_pairs=future_pairs,
    )


def nested_probe_indices(
    catalog: FullProbeCatalog,
    budget: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nested past/future index subsets of size ``budget``.

    Args:
        catalog: Exhaustive probe catalog.
        budget: Number of past and future settings to retain.
        seed: RNG seed for the shared permutation across all couplings ``J``.

    Returns:
        Tuple ``(past_indices, future_indices)`` each of length ``min(budget, |P_full|)``.
    """
    rng = np.random.default_rng(int(seed))
    n_p = len(catalog.past_settings)
    n_f = len(catalog.future_settings)
    m_p = min(int(budget), n_p)
    m_f = min(int(budget), n_f)
    past_order = rng.permutation(n_p)
    future_order = rng.permutation(n_f)
    return past_order[:m_p], future_order[:m_f]
