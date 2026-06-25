# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Typed results for split-cut V-matrix memory diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mqt.yaqs.characterization.memory.combs.surrogates.model import TransformerComb

    from .probe import ProbeSet


@dataclass(slots=True)
class CutDiagnostics:
    """Per-cut V-matrix diagnostics from split-cut probing."""

    cut: int
    entropy: float
    rank: int
    singular_values: np.ndarray
    singular_values_full: np.ndarray
    pauli_xyz_ij: np.ndarray
    probe_set: ProbeSet
    weights: np.ndarray | None = None
    V: np.ndarray | None = None
    V_centered: np.ndarray | None = None
    delta_norm: float | None = None

    @classmethod
    def from_probe_process_dict(cls, out: dict[str, Any], *, cut: int) -> CutDiagnostics:
        """Build from the internal :func:`probe_process` return dictionary.

        Returns:
            Populated :class:`CutDiagnostics` for ``cut``.
        """
        return cls(
            cut=int(cut),
            entropy=float(out["entropy"]),
            rank=int(out["rank"]),
            singular_values=np.asarray(out["singular_values"]),
            singular_values_full=np.asarray(out["singular_values_full"]),
            pauli_xyz_ij=np.asarray(out["pauli_xyz_ij"]),
            probe_set=out["probe_set"],
            weights=None if out.get("weights_ij") is None else np.asarray(out["weights_ij"]),
            V=None if out.get("V") is None else np.asarray(out["V"]),
            V_centered=None if out.get("V_centered") is None else np.asarray(out["V_centered"]),
            delta_norm=None if out.get("delta_norm") is None else float(out["delta_norm"]),
        )


@dataclass
class ProbeResult:
    """V-matrix diagnostics (entropy, rank, singular spectrum) at one or more cuts.

    Returned by :class:`~mqt.yaqs.memory_characterizer.MemoryCharacterizer` ``probe*``
    methods. Use :meth:`entropy`, :meth:`rank`, and :meth:`singular_values` for the
    primary readouts; :attr:`by_cut` exposes per-cut detail including raw ``V`` matrices.
    """

    by_cut: dict[int, CutDiagnostics]
    model: TransformerComb | None = None
    _extra: dict[str, Any] = field(default_factory=dict, repr=False)

    def entropy(self, cut: int) -> float:
        """Bond entropy :math:`S_V(c)` in nats at ``cut``.

        Returns:
            Bond entropy in nats.
        """
        return float(self.by_cut[int(cut)].entropy)

    def rank(self, cut: int) -> int:
        """Operational rank at ``cut``.

        Returns:
            Operational rank estimate.
        """
        return int(self.by_cut[int(cut)].rank)

    def singular_values(self, cut: int) -> np.ndarray:
        """Singular spectrum of the centered V matrix at ``cut``.

        Returns:
            1D array of singular values.
        """
        return np.asarray(self.by_cut[int(cut)].singular_values)

    @property
    def cut(self) -> int:
        """Single cut key when exactly one cut is present.

        Returns:
            The sole cut index in ``by_cut``.

        Raises:
            ValueError: If more than one cut is stored.
        """
        if len(self.by_cut) != 1:
            msg = f"ProbeResult.cut requires exactly one cut, got {len(self.by_cut)}."
            raise ValueError(msg)
        return next(iter(self.by_cut))

    @classmethod
    def from_single_cut(cls, diag: CutDiagnostics, *, model: TransformerComb | None = None) -> ProbeResult:
        """Wrap a single :class:`CutDiagnostics` as a :class:`ProbeResult`.

        Returns:
            Single-cut diagnostics object.
        """
        return cls(by_cut={int(diag.cut): diag}, model=model)

    @classmethod
    def from_probe_process_dict(
        cls,
        out: dict[str, Any],
        *,
        cut: int,
        model: TransformerComb | None = None,
    ) -> ProbeResult:
        """Build a single-cut result from :func:`probe_process` output.

        Returns:
            Single-cut diagnostics object.
        """
        return cls.from_single_cut(CutDiagnostics.from_probe_process_dict(out, cut=cut), model=model)

    @classmethod
    def merge(
        cls,
        results: dict[int, ProbeResult],
        *,
        model: TransformerComb | None = None,
    ) -> ProbeResult:
        """Merge per-cut :class:`ProbeResult` instances into one multi-cut result.

        Returns:
            Combined multi-cut :class:`ProbeResult`.

        Raises:
            ValueError: If any input does not contain exactly one cut.
        """
        by_cut: dict[int, CutDiagnostics] = {}
        resolved_model = model
        for cut_key in sorted(results):
            part = results[cut_key]
            if len(part.by_cut) != 1:
                msg = "merge() expects each ProbeResult to hold exactly one cut."
                raise ValueError(msg)
            by_cut[int(cut_key)] = part.by_cut[int(cut_key)]
            if resolved_model is None:
                resolved_model = part.model
        return cls(by_cut=by_cut, model=resolved_model)

    def summary(self) -> str:
        """Human-readable summary of entropy and rank per cut.

        Returns:
            Summary string.
        """
        if len(self.by_cut) == 1:
            c = next(iter(self.by_cut))
            d = self.by_cut[c]
            return f"cut={c}: S_V={d.entropy:.4f} nats, rank={d.rank}"
        lines = ["cut  S_V (nats)  rank"]
        for c in sorted(self.by_cut):
            d = self.by_cut[c]
            lines.append(f"{c:4d} {d.entropy:10.4f} {d.rank:5d}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary (for notebooks and debugging).

        Returns:
            JSON-friendly summary dictionary.
        """
        return {
            "by_cut": {
                str(cut): {
                    "entropy": d.entropy,
                    "rank": d.rank,
                    "singular_values": d.singular_values,
                    "delta_norm": d.delta_norm,
                }
                for cut, d in self.by_cut.items()
            },
            "model": None if self.model is None else repr(self.model),
            **self._extra,
        }


__all__ = ["CutDiagnostics", "ProbeResult"]
