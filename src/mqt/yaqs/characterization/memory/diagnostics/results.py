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
    cut: int
    entropy: float
    rank: int
    singular_values: np.ndarray
    memory_matrix: np.ndarray


@dataclass
class CharacterizationResult:
    """Operational memory diagnostics at one or more temporal cuts.

    Returned by :class:`~mqt.yaqs.memory_characterizer.MemoryCharacterizer.characterize`.
    """

    by_cut: dict[int, _CutResult]

    def entropy(self, cut: int) -> float:
        """Cross-cut memory entropy :math:`S_V(c)` at ``cut`` (natural log of mode weights)."""
        return float(self.by_cut[int(cut)].entropy)

    def rank(self, cut: int) -> int:
        """Effective number of resolved memory modes at ``cut`` (paper :math:`R(c)=\\exp(S_V(c))` scale)."""
        return int(self.by_cut[int(cut)].rank)

    def singular_values(self, cut: int) -> np.ndarray:
        """Singular spectrum of the memory matrix at ``cut``."""
        return np.asarray(self.by_cut[int(cut)].singular_values)

    def memory_matrix(self, cut: int) -> np.ndarray:
        """Past-row-centered weighted memory matrix at ``cut``."""
        return np.asarray(self.by_cut[int(cut)].memory_matrix)

    def summary(self) -> str:
        """Human-readable summary of entropy and rank per cut."""
        if len(self.by_cut) == 1:
            c = next(iter(self.by_cut))
            d = self.by_cut[c]
            return f"cut={c}: S_V={d.entropy:.4f}, modes≈{d.rank}"
        lines = ["cut  S_V    modes"]
        for c in sorted(self.by_cut):
            d = self.by_cut[c]
            lines.append(f"{c:4d} {d.entropy:10.4f} {d.rank:5d}")
        return "\n".join(lines)


def _cut_from_probe_dict(out: dict[str, Any], *, cut: int) -> _CutResult:
    v_centered = out.get("V_centered")
    if v_centered is None:
        msg = "probe output missing V_centered required for memory_matrix."
        raise ValueError(msg)
    return _CutResult(
        cut=int(cut),
        entropy=float(out["entropy"]),
        rank=int(out["rank"]),
        singular_values=np.asarray(out.get("singular_values_full", out["singular_values"])),
        memory_matrix=np.asarray(v_centered),
    )


def _result_from_probe_dict(out: dict[str, Any], *, cut: int) -> CharacterizationResult:
    return CharacterizationResult(by_cut={int(cut): _cut_from_probe_dict(out, cut=cut)})


def _merge_results(results: dict[int, CharacterizationResult]) -> CharacterizationResult:
    by_cut: dict[int, _CutResult] = {}
    for cut_key in sorted(results):
        part = results[cut_key]
        if len(part.by_cut) != 1:
            msg = "merge expects each CharacterizationResult to hold exactly one cut."
            raise ValueError(msg)
        by_cut[int(cut_key)] = part.by_cut[int(cut_key)]
    return CharacterizationResult(by_cut=by_cut)


__all__ = ["CharacterizationResult"]
