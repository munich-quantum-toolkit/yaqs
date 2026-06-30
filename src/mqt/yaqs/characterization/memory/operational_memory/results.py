# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Typed results for split-cut operational memory characterization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class _CutResult:
    r"""Internal per-cut storage for :class:`CharacterizationResult`.

    Attributes:
        cut: Causal cut index.
        entropy: Cross-cut memory entropy :math:`S_V(c)`.
        modes: Effective mode number :math:`R(c)=\exp(S_V(c))`.
        singular_values: Singular spectrum (possibly tail-truncated for entropy).
        response_matrix: Past-row-centered weighted response matrix.
        probe_set: Optional :class:`~mqt.yaqs.characterization.memory.operational_memory.samples.ProbeSet`.
    """

    cut: int
    entropy: float
    modes: float
    singular_values: np.ndarray
    response_matrix: np.ndarray
    probe_set: Any | None = None


@dataclass
class CharacterizationResult:
    """Operational memory diagnostics at one or more temporal cuts.

    Returned by :class:`~mqt.yaqs.memory_characterizer.MemoryCharacterizer.characterize`.
    """

    by_cut: dict[int, _CutResult]

    def _resolve_cut(self, cut: int | None) -> int:
        """Resolve an explicit cut or the sole stored cut.

        Args:
            cut: Causal cut index, or ``None`` when exactly one cut is stored.

        Returns:
            Resolved cut index.

        Raises:
            ValueError: If ``cut`` is omitted and multiple cuts are stored.
        """
        if cut is not None:
            cut_index = cut
            if cut_index not in self.by_cut:
                msg = f"cut {cut_index} is not stored in this result (available: {sorted(self.by_cut)})."
                raise ValueError(msg)
            return cut_index
        if len(self.by_cut) != 1:
            msg = "cut is required when the result holds multiple cuts."
            raise ValueError(msg)
        return int(next(iter(self.by_cut)))

    def entropy(self, cut: int | None = None) -> float:
        """Cross-cut memory entropy :math:`S_V(c)` (natural log of mode weights).

        Args:
            cut: Causal cut index. Optional when exactly one cut is stored.

        Returns:
            Entropy :math:`S_V(c)`.
        """
        c = self._resolve_cut(cut)
        return float(self.by_cut[c].entropy)

    def modes(self, cut: int | None = None) -> float:
        r"""Effective mode number :math:`R(c)=\exp(S_V(c))`.

        Args:
            cut: Causal cut index. Optional when exactly one cut is stored.

        Returns:
            Effective mode number :math:`R(c)`.
        """
        c = self._resolve_cut(cut)
        return float(self.by_cut[c].modes)

    def singular_values(self, cut: int | None = None) -> np.ndarray:
        """Singular spectrum of the response matrix at ``cut``.

        Args:
            cut: Causal cut index. Optional when exactly one cut is stored.

        Returns:
            Singular values (possibly tail-truncated for entropy).
        """
        c = self._resolve_cut(cut)
        return np.asarray(self.by_cut[c].singular_values)

    def response_matrix(self, cut: int | None = None) -> np.ndarray:
        r"""Past-row-centered weighted response matrix at ``cut``.

        Args:
            cut: Causal cut index. Optional when exactly one cut is stored.

        Returns:
            Centered response matrix :math:`\\widetilde{V}(c)`.
        """
        c = self._resolve_cut(cut)
        return np.asarray(self.by_cut[c].response_matrix)

    def probes(self, cut: int | None = None) -> dict[str, Any]:
        """Export probe arrays used at ``cut`` for logging or cross-backend reuse.

        Args:
            cut: Causal cut index. Optional when exactly one cut is stored.

        Returns:
            Dict with keys ``cut``, ``num_interventions``, ``past_features``, and ``future_features``.

        Raises:
            ValueError: If no probe data was recorded for the resolved cut.
        """
        c = self._resolve_cut(cut)
        entry = self.by_cut[c]
        if entry.probe_set is None:
            msg = f"No probe data recorded for cut={c}."
            raise ValueError(msg)
        ps = entry.probe_set
        return {
            "cut": int(ps.cut),
            "num_interventions": int(ps.num_interventions),
            "past_features": np.asarray(ps.past_features),
            "future_features": np.asarray(ps.future_features),
        }

    def summary(self) -> str:
        """Human-readable summary of entropy and modes per cut.

        Returns:
            One-line summary for a single cut, or a fixed-width table for several cuts.
        """
        if len(self.by_cut) == 1:
            c = next(iter(self.by_cut))
            d = self.by_cut[c]
            return f"cut={c}: S_V={d.entropy:.4f}, modes={d.modes:.3f}"
        lines = ["cut  S_V    modes"]
        for c in sorted(self.by_cut):
            d = self.by_cut[c]
            lines.append(f"{c:4d} {d.entropy:10.4f} {d.modes:8.3f}")
        return "\n".join(lines)


def parse_cut_result(out: dict[str, Any], *, cut: int) -> _CutResult:
    """Build one per-cut result entry from a probe-process output dict.

    Args:
        out: Output of :func:`~mqt.yaqs.characterization.memory.operational_memory.run.run_operational_memory`.
        cut: Causal cut index.

    Returns:
        Internal per-cut storage object.

    Raises:
        ValueError: If ``response_matrix`` is missing from ``out``.
    """
    response_matrix = out.get("response_matrix")
    if response_matrix is None:
        msg = "probe output missing response_matrix."
        raise ValueError(msg)
    return _CutResult(
        cut=int(cut),
        entropy=float(out["entropy"]),
        modes=float(out["modes"]),
        singular_values=np.asarray(out["singular_values"]),
        response_matrix=np.asarray(response_matrix),
        probe_set=out.get("probe_set"),
    )


def pack_result(out: dict[str, Any], *, cut: int) -> CharacterizationResult:
    """Wrap a single-cut probe dict as :class:`CharacterizationResult`.

    Args:
        out: Probe-process output dict.
        cut: Causal cut index.

    Returns:
        Result holding exactly one cut.
    """
    return CharacterizationResult(by_cut={int(cut): parse_cut_result(out, cut=cut)})


def merge_cut_results(results: dict[int, CharacterizationResult]) -> CharacterizationResult:
    """Merge single-cut characterization results into one multi-cut object.

    Args:
        results: Mapping ``cut -> CharacterizationResult`` with one cut each.

    Returns:
        Combined result.

    Raises:
        ValueError: If any partial result holds more than one cut.
    """
    by_cut: dict[int, _CutResult] = {}
    for cut_key in sorted(results):
        part = results[cut_key]
        if len(part.by_cut) != 1:
            msg = "merge expects each CharacterizationResult to hold exactly one cut."
            raise ValueError(msg)
        inner_cut = next(iter(part.by_cut))
        if int(cut_key) != int(inner_cut):
            msg = f"merge cut key {cut_key} does not match partial result cut {inner_cut}."
            raise ValueError(msg)
        by_cut[int(cut_key)] = part.by_cut[inner_cut]
    return CharacterizationResult(by_cut=by_cut)
