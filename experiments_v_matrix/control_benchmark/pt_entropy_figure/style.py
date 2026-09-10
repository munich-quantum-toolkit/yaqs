"""Matplotlib styling aligned with the reference main-text figure."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .layouts import FigureLayout


def configure_matplotlib(layout: FigureLayout | None = None) -> None:
    """APS-oriented defaults; optional ``layout`` scales typography for export size."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib as mpl

    if layout is None:
        from .layouts import STANDARD_LAYOUT

        layout = STANDARD_LAYOUT

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": layout.savefig_pad_inches,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": layout.font_size,
            "axes.labelsize": layout.axes_labelsize,
            "axes.titlesize": layout.axes_labelsize,
            "xtick.labelsize": layout.tick_labelsize,
            "ytick.labelsize": layout.tick_labelsize,
            "legend.fontsize": layout.legend_fontsize,
            "axes.linewidth": layout.spine_width,
            "xtick.major.width": layout.tick_width,
            "ytick.major.width": layout.tick_width,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "lines.linewidth": max(layout.panel_a_line_lw_mpo, 0.5),
        }
    )
