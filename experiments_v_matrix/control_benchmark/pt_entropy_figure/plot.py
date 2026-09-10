"""Publication-quality figure: S_V versus S_MPO with split spectrum panels."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from .constants import (
    CUT_COLORS,
    CUTS,
    J_COLORS,
    OUTPUT_STEM,
    PANEL_A_J_MIN,
    PANEL_A_MARKER_EVERY,
    PANEL_A_Y_FLOOR,
    PANEL_B_Y_DISPLAY_FLOOR,
    PANEL_BC_Y_CEIL,
    SPECTRUM_CUT,
    SPECTRUM_JS,
)
from .data import (
    EntropyPoint,
    QuantityKind,
    SpectrumCurve,
    entropy_series,
    panel_b_mode_span,
    resolved_mask,
)
from .layouts import STANDARD_LAYOUT, FigureLayout
from .style import configure_matplotlib


def _plot_line_markers(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    ls: str,
    marker: str,
    lw: float = 1.6,
    ms: float = 4.6,
    markevery: int = 1,
) -> None:
    mask = np.isfinite(y) & (y > 0.0)
    if not np.any(mask):
        return
    xv, yv = x[mask], y[mask]
    ax.plot(xv, yv, ls=ls, color=color, lw=lw, zorder=2, solid_capstyle="round")
    ax.plot(
        xv,
        yv,
        linestyle="none",
        marker=marker,
        color=color,
        ms=ms,
        mfc=color,
        mec="0.15",
        mew=0.75,
        markevery=markevery,
        zorder=3,
    )


def _style_log_axis(
    ax,
    layout: FigureLayout,
    *,
    xlabel: str,
    ylabel: str,
    panel_label: str,
    ylabel_size: float,
    show_xlabel: bool = True,
    panel_label_corner: Literal["top_left", "top_right"] | None = None,
) -> None:
    from matplotlib.ticker import LogFormatterMathtext, LogLocator

    corner = panel_label_corner or layout.panel_a_label_corner
    ax.set_yscale("log")
    if show_xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, fontsize=ylabel_size)
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.tick_params(
        direction="in",
        top=True,
        right=True,
        which="both",
        length=layout.tick_length,
        width=layout.tick_width,
        labelsize=layout.tick_labelsize,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(layout.spine_width)
    if corner == "top_right":
        label_x, label_ha = 0.97, "right"
    else:
        label_x, label_ha = layout.panel_label_xy[0], "left"
    label_y = layout.panel_label_xy[1]
    text_kwargs: dict = {
        "transform": ax.transAxes,
        "fontsize": layout.panel_label_size,
        "fontweight": "bold",
        "va": "top",
        "ha": label_ha,
    }
    if layout.panel_label_bbox:
        text_kwargs["bbox"] = {
            "boxstyle": "round,pad=0.12",
            "fc": "white",
            "ec": "none",
            "alpha": 0.9,
        }
    ax.text(label_x if corner == "top_right" else label_x, label_y, panel_label, **text_kwargs)


def _panel_a_legends(ax, layout: FigureLayout) -> None:
    from matplotlib.lines import Line2D

    style_ref = "0.40"
    qty_handles = [
        Line2D(
            [0],
            [0],
            color=style_ref,
            ls="-",
            marker="o",
            lw=layout.panel_a_line_lw_sv,
            ms=layout.panel_a_marker_ms_sv,
            mfc=style_ref,
            mec="0.15",
            mew=0.75,
            label=r"$S_V$",
        ),
        Line2D(
            [0],
            [0],
            color=style_ref,
            ls="--",
            marker="s",
            lw=layout.panel_a_line_lw_mpo,
            ms=layout.panel_a_marker_ms_mpo,
            mfc=style_ref,
            mec="0.15",
            mew=0.75,
            label=r"$S_{\mathrm{MPO}}$",
        ),
    ]
    cut_handles = [Line2D([0], [0], color=CUT_COLORS[c], lw=layout.panel_a_line_lw_sv, label=rf"$c={c}$") for c in CUTS]
    leg_qty = ax.legend(
        handles=qty_handles,
        frameon=False,
        loc="lower right",
        bbox_to_anchor=layout.panel_a_qty_legend_anchor,
        fontsize=layout.legend_fontsize,
        handlelength=2.4,
    )
    ax.add_artist(leg_qty)
    ax.legend(
        handles=cut_handles,
        frameon=False,
        loc="lower right",
        bbox_to_anchor=layout.panel_a_cut_legend_anchor,
        fontsize=layout.panel_a_cut_legend_fontsize,
        title="Cut",
        title_fontsize=layout.panel_a_cut_legend_fontsize,
        handlelength=1.5,
    )


def _panel_a_ylim(points: list[EntropyPoint]) -> tuple[float, float]:
    """Log-scale limits that emphasize the physical bulk, not J=0 numerical zeros."""
    bulk = [
        value
        for p in points
        for value in (p.s_v, p.s_mpo)
        if p.j >= PANEL_A_J_MIN and value > PANEL_A_Y_FLOOR
    ]
    if not bulk:
        return 1e-7, 1e-1
    y_lo = max(min(bulk) / np.sqrt(10.0), PANEL_A_Y_FLOOR)
    y_hi = max(bulk) * np.sqrt(10.0)
    return y_lo, y_hi


def _draw_panel_a(ax, points: list[EntropyPoint], layout: FigureLayout) -> None:
    for cut in CUTS:
        color = CUT_COLORS[cut]
        j_pt, s_pt = entropy_series(points, cut=cut, quantity="S_MPO")
        j_sv, s_sv = entropy_series(points, cut=cut, quantity="S_V")
        _plot_line_markers(
            ax,
            j_pt,
            s_pt,
            color=color,
            ls="--",
            marker="s",
            lw=layout.panel_a_line_lw_mpo,
            ms=layout.panel_a_marker_ms_mpo,
            markevery=PANEL_A_MARKER_EVERY,
        )
        _plot_line_markers(
            ax,
            j_sv,
            s_sv,
            color=color,
            ls="-",
            marker="o",
            lw=layout.panel_a_line_lw_sv,
            ms=layout.panel_a_marker_ms_sv,
            markevery=PANEL_A_MARKER_EVERY,
        )

    _style_log_axis(
        ax,
        layout,
        xlabel=r"Coupling $J$",
        ylabel=layout.panel_a_ylabel,
        panel_label="(a)",
        ylabel_size=layout.panel_a_ylabel_size,
        panel_label_corner=layout.panel_a_label_corner,
    )
    j_all = [p.j for p in points if p.j > 0.0]
    if j_all:
        ax.set_xlim(0.0, max(j_all) + 0.05)
    y_lo, y_hi = _panel_a_ylim(points)
    ax.set_ylim(y_lo, y_hi)
    _panel_a_legends(ax, layout)


def _quantity_marker(quantity: QuantityKind) -> str:
    return "o" if quantity == "S_V" else "s"


def _quantity_ylabel(quantity: QuantityKind) -> str:
    if quantity == "S_V":
        return r"$S_V$: $p_i$"
    return r"$S_{\mathrm{MPO}}$: $p_i$"


def _collect_spectrum_y_vals(curves: list[SpectrumCurve], n_show: int) -> list[float]:
    """Gather plotted spectrum weights from a curve subset."""
    y_vals: list[float] = []
    for curve in curves:
        n = min(n_show, curve.weights.size)
        s_head = curve.singular_values[:n]
        p_head = curve.weights[:n]
        keep = resolved_mask(s_head) & (p_head > 0.0)
        if not np.any(keep):
            continue
        y_vals.extend(p_head[keep].tolist())
    return y_vals


def _spectrum_panel_ylim(y_vals: list[float]) -> tuple[float, float]:
    """Log-scale limits from plotted spectrum weights (tight zoom on decay)."""
    if not y_vals:
        return PANEL_B_Y_DISPLAY_FLOOR, PANEL_BC_Y_CEIL
    y_lo = max(min(y_vals) / np.sqrt(10.0), PANEL_B_Y_DISPLAY_FLOOR)
    y_hi = min(max(y_vals) * np.sqrt(10.0), PANEL_BC_Y_CEIL)
    return y_lo, y_hi


def _draw_j_legend_between(fig, ax_b, ax_c, layout: FigureLayout) -> None:
    """Horizontal J legend in the gap between spectrum panels (b) and (c)."""
    from matplotlib.lines import Line2D

    j_handles = [
        Line2D([0], [0], color=J_COLORS[jv], lw=2.0, label=rf"$J={jv:g}$")
        for jv in SPECTRUM_JS
    ]
    fig.canvas.draw()
    bbox_b = ax_b.get_position()
    bbox_c = ax_c.get_position()
    x_center = (bbox_b.x0 + bbox_b.x1) / 2.0
    y_center = (bbox_b.y0 + bbox_c.y1) / 2.0
    leg = fig.legend(
        handles=j_handles,
        loc="center",
        bbox_to_anchor=(x_center, y_center),
        bbox_transform=fig.transFigure,
        frameon=True,
        fancybox=False,
        edgecolor="none",
        facecolor="white",
        framealpha=0.92,
        fontsize=layout.j_legend_fontsize,
        handlelength=1.0,
        handletextpad=0.3,
        borderpad=0.12,
        labelspacing=0.15,
        columnspacing=0.45,
        ncol=layout.j_legend_ncol,
    )


def _annotate_spectrum_cut(ax, layout: FigureLayout) -> None:
    ax.annotate(
        rf"$c={SPECTRUM_CUT}$",
        xy=(0.04, 0.05),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=layout.cut_annotate_size,
    )


def _draw_j_legend(ax, layout: FigureLayout) -> None:
    from matplotlib.lines import Line2D

    j_handles = [
        Line2D([0], [0], color=J_COLORS[jv], lw=2.0, label=rf"$J={jv:g}$")
        for jv in SPECTRUM_JS
    ]
    leg = ax.legend(
        handles=j_handles,
        loc="lower left",
        bbox_to_anchor=layout.j_legend_anchor,
        bbox_transform=ax.transAxes,
        frameon=True,
        fancybox=False,
        edgecolor="none",
        facecolor="white",
        framealpha=0.92,
        fontsize=layout.j_legend_fontsize,
        handlelength=1.0,
        handletextpad=0.3,
        borderpad=0.15,
        labelspacing=0.15,
        columnspacing=0.4,
        ncol=layout.j_legend_ncol,
    )


def _draw_spectrum_panel(
    ax,
    curves: list[SpectrumCurve],
    layout: FigureLayout,
    *,
    quantity: QuantityKind,
    panel_label: str,
    n_show: int,
    show_xlabel: bool,
    y_lim: tuple[float, float] | None = None,
) -> None:
    marker = _quantity_marker(quantity)
    for curve in curves:
        color = J_COLORS[curve.j]
        n = min(n_show, curve.weights.size)
        s_head = curve.singular_values[:n]
        p_head = curve.weights[:n]
        keep = resolved_mask(s_head) & (p_head > 0.0)
        if not np.any(keep):
            continue
        idx = np.arange(1, n + 1)[keep]
        pv = p_head[keep]
        _plot_points_with_fit(
            ax,
            idx,
            pv,
            color=color,
            marker=marker,
            ms=layout.spectrum_marker_ms,
            lw=layout.spectrum_line_lw,
        )

    _style_log_axis(
        ax,
        layout,
        xlabel=r"Mode index $i$",
        ylabel=_quantity_ylabel(quantity),
        panel_label=panel_label,
        ylabel_size=layout.spectrum_ylabel_size,
        show_xlabel=show_xlabel,
        panel_label_corner=layout.spectrum_label_corner,
    )
    ax.set_xscale("linear")
    ax.set_xlim(0.4, max(n_show, 1) + 0.6)
    mode_ticks = np.arange(1, n_show + 1)
    ax.set_xticks(mode_ticks)
    from matplotlib.ticker import FuncFormatter

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{int(round(x))}"))
    if not show_xlabel:
        ax.tick_params(labelbottom=False)
    if y_lim is not None:
        ax.set_ylim(*y_lim)


def _draw_spectrum_panels(
    fig,
    ax_b,
    ax_c,
    curves: list[SpectrumCurve],
    layout: FigureLayout,
) -> None:
    """Stacked spectrum panels: (b) S_V, (c) S_MPO."""
    n_show = panel_b_mode_span(curves)
    sv_curves = [c for c in curves if c.quantity == "S_V"]
    spt_curves = [c for c in curves if c.quantity == "S_MPO"]
    y_vals = _collect_spectrum_y_vals(sv_curves, n_show) + _collect_spectrum_y_vals(spt_curves, n_show)
    y_lim = _spectrum_panel_ylim(y_vals)
    _draw_spectrum_panel(
        ax_b,
        sv_curves,
        layout,
        quantity="S_V",
        panel_label="(b)",
        n_show=n_show,
        show_xlabel=False,
        y_lim=y_lim,
    )
    if layout.j_legend_placement == "b":
        _draw_j_legend(ax_b, layout)
    _draw_spectrum_panel(
        ax_c,
        spt_curves,
        layout,
        quantity="S_MPO",
        panel_label="(c)",
        n_show=n_show,
        show_xlabel=True,
        y_lim=y_lim,
    )
    if layout.j_legend_placement == "c":
        _draw_j_legend(ax_c, layout)
    if layout.j_legend_placement == "between_bc":
        _draw_j_legend_between(fig, ax_b, ax_c, layout)
    _annotate_spectrum_cut(ax_c, layout)


def _plot_points_with_fit(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    marker: str,
    ms: float = 4.6,
    lw: float = 1.2,
) -> None:
    """Scatter resolved spectrum points and overlay a smooth log-space interpolant."""
    mask = np.isfinite(y) & (y > 0.0)
    if not np.any(mask):
        return
    xv, yv = x[mask], y[mask]
    order = np.argsort(xv)
    xv, yv = xv[order], yv[order]
    ax.plot(
        xv,
        yv,
        linestyle="none",
        marker=marker,
        color=color,
        ms=ms,
        mfc=color,
        mec="0.15",
        mew=0.75,
        zorder=3,
    )
    if xv.size >= 2:
        xf = np.linspace(float(xv[0]), float(xv[-1]), 200)
        yf = np.exp(np.interp(xf, xv, np.log(yv)))
        ax.plot(xf, yf, "-", color=color, lw=lw, alpha=0.85, zorder=2, solid_capstyle="round")


def _make_axes(fig, layout: FigureLayout):
    """Gridspec axes following the chosen layout topology."""
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=list(layout.width_ratios),
        height_ratios=list(layout.height_ratios),
        wspace=layout.wspace,
        hspace=layout.hspace,
    )
    if layout.topology == "side_by_side":
        ax_a = fig.add_subplot(gs[:, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[1, 1], sharex=ax_b, sharey=ax_b)
        return ax_a, ax_b, ax_c
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1], sharex=ax_b, sharey=ax_b)
    return ax_a, ax_b, ax_c


def build_figure(
    points: list[EntropyPoint],
    curves: list[SpectrumCurve],
    *,
    layout: FigureLayout | None = None,
):
    """Create the figure: (a) entropy sweep; stacked (b)/(c) spectra."""
    import matplotlib.pyplot as plt

    layout = layout or STANDARD_LAYOUT
    configure_matplotlib(layout)
    fig = plt.figure(figsize=layout.figsize, layout="constrained")
    ax_a, ax_b, ax_c = _make_axes(fig, layout)
    _draw_panel_a(ax_a, points, layout)
    _draw_spectrum_panels(fig, ax_b, ax_c, curves, layout)
    return fig


def save_figure(
    fig,
    output_dir: Path,
    *,
    stem: str = OUTPUT_STEM,
    dpi: int = 400,
    pad_inches: float | None = None,
) -> tuple[Path, Path]:
    """Write vector PDF and high-resolution PNG exports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    pad = 0.02 if pad_inches is None else pad_inches
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=pad)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=pad)
    return pdf_path, png_path
