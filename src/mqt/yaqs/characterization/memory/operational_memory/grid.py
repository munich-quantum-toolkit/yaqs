# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Probe sequence grid assembly for split-cut operational memory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from ..shared.encoding import SITE0_KET

if TYPE_CHECKING:
    from .samples import ProbeSet


class _ProbeBranches(NamedTuple):
    """Validated past/future branch slices for one probe-grid entry."""

    cut: int
    num_interventions: int
    past_pairs: list[Any]
    future_pairs: list[Any]


def _validated_probe_branches(probe_set: ProbeSet, i: int, j: int) -> _ProbeBranches:
    """Validate probe-set metadata and return branch slices for ``(i, j)``.

    Args:
        probe_set: Sampled split-cut probes.
        i: Past index.
        j: Future index.

    Returns:
        Validated branch context.

    Raises:
        ValueError: If branch lengths or cut-branch array sizes do not match metadata.
    """
    cut = probe_set.cut
    num_interventions = probe_set.num_interventions
    past_len = cut - 1
    future_len = num_interventions - cut
    n_pasts = len(probe_set.past_pairs)
    n_futures = len(probe_set.future_pairs)
    past_pairs = probe_set.past_pairs[i]
    future_pairs = probe_set.future_pairs[j]
    if len(past_pairs) != past_len:
        msg = f"past_pairs[{i}] length {len(past_pairs)} != cut-1={past_len}"
        raise ValueError(msg)
    if len(future_pairs) != future_len:
        msg = f"future_pairs[{j}] length {len(future_pairs)} != num_interventions-cut={future_len}"
        raise ValueError(msg)
    if len(probe_set.past_cut_meas) != n_pasts:
        msg = f"past_cut_meas length {len(probe_set.past_cut_meas)} != n_pasts={n_pasts}"
        raise ValueError(msg)
    if len(probe_set.future_prep_cut) != n_futures:
        msg = f"future_prep_cut length {len(probe_set.future_prep_cut)} != n_futures={n_futures}"
        raise ValueError(msg)
    return _ProbeBranches(cut, num_interventions, past_pairs, future_pairs)


def compute_delayed_length(*, num_interventions: int, delay: int) -> int:
    """Compute the physical sequence length with reset delay at the causal break.

    Args:
        num_interventions: Base split-cut sequence length when ``delay=0``.
        delay: Number of ``(|0>, |0>)`` soft-reset slots inserted at the break.

    Returns:
        ``num_interventions + delay + 1`` when ``delay > 0``; otherwise ``num_interventions``.

    Raises:
        ValueError: If ``delay`` is negative.
    """
    if delay < 0:
        msg = f"delay must be >= 0, got {delay}"
        raise ValueError(msg)
    return num_interventions + delay + 1 if delay > 0 else num_interventions


def _append_cut_steps(
    full: list[Any],
    probe_set: ProbeSet,
    *,
    i: int,
    j: int,
    delay: int,
) -> None:
    """Append causal-break steps, optionally with soft-reset delay slots."""
    if delay == 0:
        full.append((probe_set.past_cut_meas[i], probe_set.future_prep_cut[j]))
        return
    full.append((probe_set.past_cut_meas[i], SITE0_KET))
    full.extend((SITE0_KET, SITE0_KET) for _ in range(delay))
    full.append((SITE0_KET, probe_set.future_prep_cut[j]))


def assemble_probe_sequence(probe_set: ProbeSet, i: int, j: int, *, delay: int = 0) -> list[Any]:
    """Build the full intervention sequence for probe-grid entry ``(i, j)``.

    Args:
        probe_set: Sampled split-cut probes.
        i: Past index.
        j: Future index.
        delay: Number of ``(|0>, |0>)`` soft-reset slots to insert at the break.

    Returns:
        Intervention sequence of length :func:`compute_delayed_length`.

    Raises:
        ValueError: If ``delay`` is negative, branch lengths or cut-branch array sizes do not
            match metadata, or the assembled length does not match the expected length.

    Note:
        When ``delay=0``, the cut step is ``(meas, prep)``. For ``delay > 0``, the cut becomes
        ``(meas, |0>)``, followed by ``delay`` ``(|0>, |0>)`` slots, then ``(|0>, prep)`` before
        the future unitaries.
    """
    branches = _validated_probe_branches(probe_set, i, j)
    full: list[Any] = list(branches.past_pairs)
    _append_cut_steps(full, probe_set, i=i, j=j, delay=delay)
    full.extend(branches.future_pairs)
    expected = compute_delayed_length(num_interventions=branches.num_interventions, delay=delay)
    if len(full) != expected:
        if delay == 0:
            msg = f"assembled probe sequence length {len(full)} != num_interventions={expected}"
        else:
            msg = f"assembled delayed sequence length {len(full)} != num_interventions+delay+1={expected}"
        raise ValueError(msg)
    return full


def assemble_probe_grid(
    probe_set: ProbeSet,
    *,
    delay: int = 0,
) -> tuple[list[list[Any]], int, int]:
    """Construct the full ``(past, future)`` sequence pair grid.

    Args:
        probe_set: Sampled split-cut probes.
        delay: Number of ``(|0>, |0>)`` soft-reset slots to insert at the causal break.

    Returns:
        Tuple ``(all_pairs, n_pasts, n_futures)``.

    Raises:
        ValueError: If ``delay`` is negative.
        RuntimeError: If an assembled sequence length does not match the expected length.
    """
    n_pasts = len(probe_set.past_pairs)
    n_futures = len(probe_set.future_pairs)
    expected = compute_delayed_length(num_interventions=probe_set.num_interventions, delay=delay)
    all_pairs: list[list[Any]] = []
    for i in range(n_pasts):
        for j in range(n_futures):
            full = assemble_probe_sequence(probe_set, i, j, delay=delay)
            if len(full) != expected:
                msg = "internal: sequence length mismatch"
                raise RuntimeError(msg)
            all_pairs.append(full)
    return all_pairs, n_pasts, n_futures
