# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Probe sequence grid assembly for split-cut operational memory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .samples import ProbeSet

_Z0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)


def assemble_probe_sequence(probe_set: ProbeSet, i: int, j: int) -> list[Any]:
    """Build the full intervention sequence for probe-grid entry ``(i, j)``.

    Args:
        probe_set: Sampled split-cut probes.
        i: Past index.
        j: Future index.

    Returns:
        Intervention sequence of length ``probe_set.k``.

    Raises:
        ValueError: If past/future branch lengths, cut-branch array sizes, or the assembled
            sequence length do not match ``probe_set`` metadata.
    """
    cut = probe_set.cut
    sequence_length = probe_set.k
    past_len = cut - 1
    future_len = sequence_length - cut
    n_pasts = len(probe_set.past_pairs)
    n_futures = len(probe_set.future_pairs)
    past_pairs = probe_set.past_pairs[i]
    future_pairs = probe_set.future_pairs[j]
    if len(past_pairs) != past_len:
        msg = f"past_pairs[{i}] length {len(past_pairs)} != cut-1={past_len}"
        raise ValueError(msg)
    if len(future_pairs) != future_len:
        msg = f"future_pairs[{j}] length {len(future_pairs)} != k-cut={future_len}"
        raise ValueError(msg)
    if len(probe_set.past_cut_meas) != n_pasts:
        msg = f"past_cut_meas length {len(probe_set.past_cut_meas)} != n_pasts={n_pasts}"
        raise ValueError(msg)
    if len(probe_set.future_prep_cut) != n_futures:
        msg = f"future_prep_cut length {len(probe_set.future_prep_cut)} != n_futures={n_futures}"
        raise ValueError(msg)
    full: list[Any] = list(past_pairs)
    full.append((probe_set.past_cut_meas[i], probe_set.future_prep_cut[j]))
    full.extend(future_pairs)
    if len(full) != sequence_length:
        msg = f"assembled probe sequence length {len(full)} != k={sequence_length}"
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
    n_pasts = len(probe_set.past_pairs)
    n_futures = len(probe_set.future_pairs)
    sequence_length = probe_set.k
    all_pairs: list[list[Any]] = []
    for i in range(n_pasts):
        for j in range(n_futures):
            full = assemble_probe_sequence(probe_set, i, j)
            if len(full) != sequence_length:
                msg = "internal: full sequence length mismatch"
                raise RuntimeError(msg)
            all_pairs.append(full)
    return all_pairs, n_pasts, n_futures


def delayed_sequence_length(*, k: int, delay: int) -> int:
    """Compute the physical sequence length with reset delay at the causal break.

    Args:
        k: Base split-cut sequence length when ``delay=0``.
        delay: Number of ``(|0>, |0>)`` soft-reset slots inserted at the break.

    Returns:
        ``k + delay + 1`` when ``delay > 0``; otherwise ``k``.

    Raises:
        ValueError: If ``delay`` is negative.
    """
    if delay < 0:
        msg = f"delay must be >= 0, got {delay}"
        raise ValueError(msg)
    return k + delay + 1 if delay > 0 else k


def assemble_delayed_probe_sequence(probe_set: ProbeSet, i: int, j: int, *, delay: int = 0) -> list[Any]:
    """Build a probe sequence with optional soft-reset delay at the causal break.

    Args:
        probe_set: Sampled split-cut probes at base length ``k``.
        i: Past index.
        j: Future index.
        delay: Number of ``(|0>, |0>)`` slots to insert at the break.

    Returns:
        Intervention sequence of length :func:`delayed_sequence_length`.

    Raises:
        ValueError: If past/future branch lengths, cut-branch array sizes, or the assembled
            length do not match.

    Note:
        When ``delay=0``, the result matches :func:`assemble_probe_sequence`. For
        ``delay > 0``, the cut step becomes ``(meas, |0>)``, followed by ``delay``
        ``(|0>, |0>)`` slots, then ``(|0>, prep)`` before the future unitaries.
    """
    if delay == 0:
        return assemble_probe_sequence(probe_set, i, j)
    cut = probe_set.cut
    sequence_length = probe_set.k
    past_len = cut - 1
    future_len = sequence_length - cut
    n_pasts = len(probe_set.past_pairs)
    n_futures = len(probe_set.future_pairs)
    past_pairs = probe_set.past_pairs[i]
    future_pairs = probe_set.future_pairs[j]
    if len(past_pairs) != past_len:
        msg = f"past_pairs[{i}] length {len(past_pairs)} != cut-1={past_len}"
        raise ValueError(msg)
    if len(future_pairs) != future_len:
        msg = f"future_pairs[{j}] length {len(future_pairs)} != k-cut={future_len}"
        raise ValueError(msg)
    if len(probe_set.past_cut_meas) != n_pasts:
        msg = f"past_cut_meas length {len(probe_set.past_cut_meas)} != n_pasts={n_pasts}"
        raise ValueError(msg)
    if len(probe_set.future_prep_cut) != n_futures:
        msg = f"future_prep_cut length {len(probe_set.future_prep_cut)} != n_futures={n_futures}"
        raise ValueError(msg)
    full: list[Any] = list(past_pairs)
    full.append((probe_set.past_cut_meas[i], _Z0))
    full.extend((_Z0, _Z0) for _ in range(delay))
    full.append((_Z0, probe_set.future_prep_cut[j]))
    full.extend(future_pairs)
    expected = delayed_sequence_length(k=sequence_length, delay=delay)
    if len(full) != expected:
        msg = f"assembled delayed sequence length {len(full)} != k+delay+1={expected}"
        raise ValueError(msg)
    return full


def assemble_delayed_probe_grid(
    probe_set: ProbeSet,
    *,
    delay: int = 0,
) -> tuple[list[list[Any]], int, int]:
    """Construct the full probe grid with optional reset delay at the causal break.

    Args:
        probe_set: Sampled split-cut probes at base length ``k``.
        delay: Number of ``(|0>, |0>)`` soft-reset slots to insert at the break.

    Returns:
        Tuple ``(all_pairs, n_pasts, n_futures)``.

    Raises:
        RuntimeError: If an assembled sequence length does not match the expected length.
    """
    if delay == 0:
        return assemble_probe_grid(probe_set)
    n_pasts = len(probe_set.past_pairs)
    n_futures = len(probe_set.future_pairs)
    expected = delayed_sequence_length(k=probe_set.k, delay=delay)
    all_pairs: list[list[Any]] = []
    for i in range(n_pasts):
        for j in range(n_futures):
            full = assemble_delayed_probe_sequence(probe_set, i, j, delay=delay)
            if len(full) != expected:
                msg = "internal: delayed sequence length mismatch"
                raise RuntimeError(msg)
            all_pairs.append(full)
    return all_pairs, n_pasts, n_futures
