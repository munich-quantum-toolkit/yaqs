# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Probe sequence grid assembly for split-cut operational memory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .samples import ProbeSet


def assemble_probe_sequence(probe_set: ProbeSet, i: int, j: int) -> list[Any]:
    """Build the full intervention sequence for probe-grid entry ``(i, j)``.

    Args:
        probe_set: Sampled split-cut probes.
        i: Past index.
        j: Future index.

    Returns:
        Intervention sequence of length ``probe_set.k``.

    Raises:
        ValueError: If past/future branch lengths are inconsistent or the assembled
            sequence length does not equal ``probe_set.k``.
    """
    c = int(probe_set.cut)
    kk = int(probe_set.k)
    past_len = c - 1
    future_len = kk - c
    past_pairs = probe_set.past_pairs[i]
    future_pairs = probe_set.future_pairs[j]
    if len(past_pairs) != past_len:
        msg = f"past_pairs[{i}] length {len(past_pairs)} != cut-1={past_len}"
        raise ValueError(msg)
    if len(future_pairs) != future_len:
        msg = f"future_pairs[{j}] length {len(future_pairs)} != k-cut={future_len}"
        raise ValueError(msg)
    full: list[Any] = list(past_pairs)
    full.append((probe_set.past_cut_meas[i], probe_set.future_prep_cut[j]))
    full.extend(future_pairs)
    if len(full) != kk:
        msg = f"assembled probe sequence length {len(full)} != k={kk}"
        raise ValueError(msg)
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
