# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""SVD truncation instrumentation for the cutoff diagnostic."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from mqt.yaqs.core import linalg


@dataclass
class TruncationContext:
    """Metadata attached to the next ``linalg.truncate`` calls."""

    method: str = ""
    threshold: float = 0.0
    chi_max: int = 0
    trotter_step: int = 0
    time: float = 0.0
    gate_name: str = ""
    gate_qubits: tuple[int, ...] = ()
    gate_index: int = -1
    is_long_range: bool = False
    trunc_mode: str = "discarded_weight"


@dataclass
class TruncationEvent:
    """One SVD truncation event."""

    method: str
    threshold: float
    chi_max: int
    trotter_step: int
    time: float
    gate_name: str
    gate_qubits: str
    gate_index: int
    is_long_range: int
    bond_hint: int
    pre_rank: int
    retained_rank: int
    keep_by_cutoff: int
    s_max: float
    s_min_retained: float
    s_max_discarded: float
    discarded_weight: float
    discarded_weight_cutoff: float
    discarded_weight_chi: float
    limiter: str  # cutoff | chi_max | neither
    trunc_mode: str


def _keep_counts(
    s: np.ndarray,
    *,
    mode: str,
    threshold: float,
    max_bond_dim: int | None,
    min_keep: int,
    truncate_fn,
) -> tuple[int, int]:
    keep_cutoff = int(
        truncate_fn(s, mode=mode, threshold=threshold, max_bond_dim=None, min_keep=min_keep)
    )
    keep = int(
        truncate_fn(
            s,
            mode=mode,
            threshold=threshold,
            max_bond_dim=max_bond_dim,
            min_keep=min_keep,
        )
    )
    return keep_cutoff, keep


@dataclass
class SVDDiagnosticTracker:
    """Accumulate detailed truncation statistics by monkeypatching ``linalg.truncate``."""

    events: list[TruncationEvent] = field(default_factory=list)
    spectra: dict[str, np.ndarray] = field(default_factory=dict)
    context: TruncationContext = field(default_factory=TruncationContext)
    _capture_spectrum_keys: set[str] = field(default_factory=set)

    def set_context(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self.context, key, value)

    def request_spectrum(self, key: str) -> None:
        self._capture_spectrum_keys.add(key)


@contextmanager
def track_svd_events(tracker: SVDDiagnosticTracker) -> Iterator[None]:
    """Instrument ``linalg.truncate`` while preserving production keep semantics."""
    original = linalg.truncate

    def wrapped(
        s_vec: np.ndarray,
        *,
        mode: str,
        threshold: float,
        max_bond_dim: int | None = None,
        min_keep: int = 1,
    ) -> int:
        s = np.asarray(s_vec, dtype=float).reshape(-1)
        n = int(s.size)
        total = float(np.sum(np.square(s)))
        keep_cutoff, keep = _keep_counts(
            s,
            mode=mode,
            threshold=threshold,
            max_bond_dim=max_bond_dim,
            min_keep=min_keep,
            truncate_fn=original,
        )
        if total > 0.0:
            disc_cut = float(np.sum(np.square(s[keep_cutoff:]))) / total
            disc_chi = float(np.sum(np.square(s[keep:keep_cutoff]))) / total if keep < keep_cutoff else 0.0
            disc = float(np.sum(np.square(s[keep:]))) / total
        else:
            disc_cut = disc_chi = disc = 0.0
        if keep < keep_cutoff:
            limiter = "chi_max"
        elif keep_cutoff < n:
            limiter = "cutoff"
        else:
            limiter = "neither"
        ctx = tracker.context
        tracker.events.append(
            TruncationEvent(
                method=ctx.method,
                threshold=float(threshold),
                chi_max=int(ctx.chi_max if max_bond_dim is None else max_bond_dim),
                trotter_step=int(ctx.trotter_step),
                time=float(ctx.time),
                gate_name=ctx.gate_name,
                gate_qubits="-".join(str(q) for q in ctx.gate_qubits),
                gate_index=int(ctx.gate_index),
                is_long_range=int(ctx.is_long_range),
                bond_hint=int(ctx.gate_qubits[0]) if ctx.gate_qubits else -1,
                pre_rank=n,
                retained_rank=int(keep),
                keep_by_cutoff=int(keep_cutoff),
                s_max=float(s[0]) if n else 0.0,
                s_min_retained=float(s[keep - 1]) if keep > 0 else 0.0,
                s_max_discarded=float(s[keep]) if keep < n else 0.0,
                discarded_weight=disc,
                discarded_weight_cutoff=disc_cut,
                discarded_weight_chi=disc_chi,
                limiter=limiter,
                trunc_mode=str(mode),
            )
        )
        for key in list(tracker._capture_spectrum_keys):
            if key not in tracker.spectra:
                tracker.spectra[key] = s.copy()
                tracker._capture_spectrum_keys.discard(key)
                break
        return keep

    linalg.truncate = wrapped  # type: ignore[assignment]
    try:
        yield
    finally:
        linalg.truncate = original
