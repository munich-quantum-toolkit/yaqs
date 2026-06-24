# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Operational memory readouts (entropy / singular spectrum) for comb types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .probe import ProbeSet, branch_weights_ij, probe_process

if TYPE_CHECKING:
    from numpy.random import Generator


def _hash_probe_kwargs(probe_kwargs: dict[str, Any]) -> tuple[Any, ...]:
    items: list[tuple[str, Any]] = []
    for key in sorted(probe_kwargs):
        val = probe_kwargs[key]
        if isinstance(val, ProbeSet):
            items.append((key, ("probe_set", id(val))))
        else:
            items.append((key, val))
    return tuple(items)


class OperationalMemoryMixin:
    """Mixin providing ``entropy``, ``singular_values``, and ``rank`` via split-cut probing."""

    _operational_memory_cache: dict[str, Any] | None

    def evaluate_probe_set_with_weights(self, probe_set: ProbeSet) -> tuple[np.ndarray, np.ndarray]:
        """Return Pauli responses and causal cut branch weights for paper-aligned V assembly."""
        pauli = np.asarray(self.evaluate_probe_set(probe_set), dtype=np.float32)
        return pauli, branch_weights_ij(probe_set)

    def clear_operational_memory_cache(self) -> None:
        """Drop cached split-cut V-matrix diagnostics (e.g. after refitting a surrogate)."""
        self._operational_memory_cache = None

    def _default_cut(self, k: int) -> int:
        return (k + 1) // 2

    def _resolve_cut(self, cut: int | None) -> int:
        k = self._k_for_probe()
        c = self._default_cut(k) if cut is None else int(cut)
        if not (1 <= c <= k):
            msg = f"cut must satisfy 1 <= cut <= k ({k}), got {c}."
            raise ValueError(msg)
        return c

    def _k_for_probe(self) -> int:
        msg = "OperationalMemoryMixin requires _k_for_probe() on the host class."
        raise NotImplementedError(msg)

    def _memory_cache_key(
        self,
        cut: int,
        *,
        n_pasts: int,
        n_futures: int,
        rng: Generator | None,
        probe_kwargs: dict[str, Any],
    ) -> tuple[Any, ...]:
        if rng is not None:
            rng_token = ("rng", id(rng))
        else:
            rng_token = ("auto",)
        return (
            int(cut),
            int(n_pasts),
            int(n_futures),
            rng_token,
            _hash_probe_kwargs(probe_kwargs),
        )

    def _probe_memory_diagnostics(
        self,
        cut: int,
        *,
        n_pasts: int,
        n_futures: int,
        rng: Generator | None,
        probe_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        extra = dict(probe_kwargs or {})
        key = self._memory_cache_key(
            cut,
            n_pasts=n_pasts,
            n_futures=n_futures,
            rng=rng,
            probe_kwargs=extra,
        )
        cache = getattr(self, "_operational_memory_cache", None)
        if cache is not None and cache.get("key") == key:
            return cache["out"]

        out = probe_process(
            process=self,
            cut=cut,
            k=self._k_for_probe(),
            n_pasts=n_pasts,
            n_futures=n_futures,
            rng=rng,
            return_v=True,
            **extra,
        )
        self._operational_memory_cache = {"key": key, "out": out}
        return out

    def entropy(
        self,
        cut: int | None = None,
        *,
        n_pasts: int = 32,
        n_futures: int = 32,
        rng: Generator | None = None,
        **probe_kwargs: Any,
    ) -> float:
        c = self._resolve_cut(cut)
        out = self._probe_memory_diagnostics(
            c,
            n_pasts=n_pasts,
            n_futures=n_futures,
            rng=rng,
            probe_kwargs=probe_kwargs,
        )
        return float(out["entropy"])

    def singular_values(
        self,
        cut: int | None = None,
        *,
        n_pasts: int = 32,
        n_futures: int = 32,
        rng: Generator | None = None,
        **probe_kwargs: Any,
    ) -> np.ndarray:
        c = self._resolve_cut(cut)
        out = self._probe_memory_diagnostics(
            c,
            n_pasts=n_pasts,
            n_futures=n_futures,
            rng=rng,
            probe_kwargs=probe_kwargs,
        )
        return np.asarray(out["singular_values_full"])

    def rank(
        self,
        cut: int | None = None,
        *,
        n_pasts: int = 32,
        n_futures: int = 32,
        rng: Generator | None = None,
        **probe_kwargs: Any,
    ) -> int:
        """Operational memory rank from the same centered V matrix as :meth:`entropy`."""
        c = self._resolve_cut(cut)
        out = self._probe_memory_diagnostics(
            c,
            n_pasts=n_pasts,
            n_futures=n_futures,
            rng=rng,
            probe_kwargs=probe_kwargs,
        )
        return int(out["rank"])

    @property
    def operational_v_matrices(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Cached ``(V, V_centered)`` from the last split-cut diagnostic, if any."""
        cache = getattr(self, "_operational_memory_cache", None)
        if cache is None:
            return None
        out = cache["out"]
        v = out.get("V")
        v_c = out.get("V_centered")
        if v is None or v_c is None:
            return None
        return np.asarray(v), np.asarray(v_c)
