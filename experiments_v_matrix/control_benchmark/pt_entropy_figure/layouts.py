"""Figure layout presets for draft and PRX Quantum single-column export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .constants import OUTPUT_STEM

# APS / PRX Quantum single-column width (8.6 cm = 3 3/8 in).
PRX_SINGLE_COLUMN_WIDTH_IN = 3.375

Topology = Literal["side_by_side", "stacked"]
PanelLabelCorner = Literal["top_left", "top_right"]
JLegendPlacement = Literal["b", "c", "between_bc"]


@dataclass(frozen=True)
class FigureLayout:
    """Geometry and typography for one figure variant."""

    name: str
    output_stem: str
    figsize: tuple[float, float]
    topology: Topology
    width_ratios: tuple[float, float]
    height_ratios: tuple[float, float]
    wspace: float
    hspace: float
    font_size: float
    axes_labelsize: float
    tick_labelsize: float
    legend_fontsize: float
    panel_label_size: float
    panel_a_ylabel_size: float
    panel_a_ylabel: str
    spectrum_ylabel_size: float
    cut_annotate_size: float
    panel_a_line_lw_sv: float
    panel_a_line_lw_mpo: float
    panel_a_marker_ms_sv: float
    panel_a_marker_ms_mpo: float
    spectrum_marker_ms: float
    spectrum_line_lw: float
    tick_length: float
    tick_width: float
    spine_width: float
    panel_a_qty_legend_anchor: tuple[float, float]
    panel_a_cut_legend_anchor: tuple[float, float]
    panel_a_cut_legend_fontsize: float
    j_legend_anchor: tuple[float, float]
    j_legend_fontsize: float
    j_legend_ncol: int
    j_legend_placement: JLegendPlacement
    savefig_pad_inches: float
    panel_a_label_corner: PanelLabelCorner
    spectrum_label_corner: PanelLabelCorner
    panel_label_bbox: bool
    panel_label_xy: tuple[float, float]
    png_dpi: int


STANDARD_LAYOUT = FigureLayout(
    name="standard",
    output_stem=OUTPUT_STEM,
    figsize=(7.1, 2.95),
    topology="side_by_side",
    width_ratios=(1.08, 1.0),
    height_ratios=(1.0, 1.0),
    wspace=0.08,
    hspace=0.12,
    font_size=8.5,
    axes_labelsize=10.5,
    tick_labelsize=8.5,
    legend_fontsize=8.0,
    panel_label_size=11.5,
    panel_a_ylabel_size=10.5,
    panel_a_ylabel="Entropy",
    spectrum_ylabel_size=8.5,
    cut_annotate_size=7.5,
    panel_a_line_lw_sv=1.8,
    panel_a_line_lw_mpo=1.6,
    panel_a_marker_ms_sv=4.8,
    panel_a_marker_ms_mpo=4.6,
    spectrum_marker_ms=4.2,
    spectrum_line_lw=1.2,
    tick_length=3.5,
    tick_width=0.75,
    spine_width=0.85,
    panel_a_qty_legend_anchor=(0.72, 0.02),
    panel_a_cut_legend_anchor=(0.98, 0.02),
    panel_a_cut_legend_fontsize=8.0,
    j_legend_anchor=(0.03, 0.04),
    j_legend_fontsize=6.0,
    j_legend_ncol=2,
    j_legend_placement="b",
    savefig_pad_inches=0.02,
    panel_a_label_corner="top_left",
    spectrum_label_corner="top_right",
    panel_label_bbox=True,
    panel_label_xy=(0.03, 0.97),
    png_dpi=400,
)

# Inspired by experiments_v_matrix/cut_vs_j.py: dominant left panel, two right
# cross-sections, width_ratios 2:1, tight wspace/hspace, top-left panel tags.
_CUT_VS_J_HEIGHT_IN = 4.3
_CUT_VS_J_WIDTH_IN = 8.2
_PRX_HEIGHT_IN = _CUT_VS_J_HEIGHT_IN * (PRX_SINGLE_COLUMN_WIDTH_IN / _CUT_VS_J_WIDTH_IN)

PRX_SINGLE_COLUMN_LAYOUT = FigureLayout(
    name="prx_single_column",
    output_stem="sv_entropy_spectra_prx1col",
    figsize=(PRX_SINGLE_COLUMN_WIDTH_IN, _PRX_HEIGHT_IN),
    topology="side_by_side",
    width_ratios=(2.0, 1.0),
    height_ratios=(1.0, 1.0),
    wspace=0.06,
    hspace=0.10,
    font_size=6.5,
    axes_labelsize=8.0,
    tick_labelsize=6.0,
    legend_fontsize=5.5,
    panel_label_size=6.0,
    panel_a_ylabel_size=8.0,
    panel_a_ylabel="Entropy",
    spectrum_ylabel_size=6.5,
    cut_annotate_size=5.5,
    panel_a_line_lw_sv=0.9,
    panel_a_line_lw_mpo=0.8,
    panel_a_marker_ms_sv=3.4,
    panel_a_marker_ms_mpo=3.2,
    spectrum_marker_ms=3.0,
    spectrum_line_lw=0.75,
    tick_length=2.8,
    tick_width=0.55,
    spine_width=0.65,
    panel_a_qty_legend_anchor=(0.76, 0.02),
    panel_a_cut_legend_anchor=(0.98, 0.02),
    panel_a_cut_legend_fontsize=4.8,
    j_legend_anchor=(0.04, 0.05),
    j_legend_fontsize=4.5,
    j_legend_ncol=4,
    j_legend_placement="between_bc",
    savefig_pad_inches=0.02,
    panel_a_label_corner="top_left",
    spectrum_label_corner="top_right",
    panel_label_bbox=False,
    panel_label_xy=(0.04, 0.955),
    png_dpi=600,
)
