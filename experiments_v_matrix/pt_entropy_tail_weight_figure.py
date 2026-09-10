#!/usr/bin/env python3
r"""Two-panel main-text comparison figures for the temporal-entropy diagnostics.

Panel (a) is shared by both figures: full response entropy ``S_V^{full}`` (centered response
matrix) and the causal-block process-tensor entropy ``S_PT^{cb}`` -- the temporal
operator-Schmidt entropy of the process tensor -- versus coupling J across cuts c=1,2,3.

Two panel-(b) variants are produced:
  * tail-weight variant (``entropy_and_tail_weight_vs_J_refined``): 1 - p1 versus J.
  * representative-spectrum variant (``entropy_and_representative_spectrum``): the normalized
    singular weights p_i at a single strong-coupling point (default J=6, c=2).

Manuscript caption basis (representative-spectrum variant):
    (a) Full response entropy S_V^{full} and causal-block process-tensor entropy S_PT^{cb}
    versus coupling strength J. The quantities track each other at weak coupling but separate
    at stronger coupling. (b) Normalized singular weights at c=2 and J=6. The process tensor
    remains strongly concentrated in its leading operator-Schmidt mode, whereas the centered
    response develops appreciable subleading modes.

Reads diagnostic outputs produced by ``pt_temporal_entropy_diagnostics.py``. Main-sweep points
are joined against validation.csv on (J, cut) at the reference run configuration; a point is
plotted only if its validation row is valid. Invalid points are omitted entirely (no lines
drawn across them) and reported to the console. Stored data is never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from common import configure_matplotlib_prl

CUTS = (1, 2, 3)
CUT_COLORS = {1: "#009E73", 2: "#0072B2", 3: "#D55E00"}

# Object mapping: (csv object_type, display label, line style, marker)
QUANTITIES = (
    ("response_centered", r"$S_V^{\mathrm{full}}$", "-", "o"),
    ("process_tensor", r"$S_{\mathrm{PT}}^{\mathrm{cb}}$", "--", "s"),
)

# Distinct (non-cut) colors for the two objects in the representative-spectrum panel.
SPECTRUM_STYLE = {
    "response_centered": {
        "color": "#D81B60",
        "ls": "-",
        "marker": "o",
        "label": r"$S_V^{\mathrm{full}}$ (centered response)",
    },
    "process_tensor": {
        "color": "#1E88E5",
        "ls": "--",
        "marker": "s",
        "label": r"$S_{\mathrm{PT}}^{\mathrm{cb}}$ (process tensor)",
    },
}

TAIL_FLOOR = 1e-16
WEIGHT_FLOOR = 1e-16  # numerically meaningful mode threshold for the spectrum panel
CUMULATIVE_CAP = 1.0 - 1e-12
MIN_MODES = 10
REF_POINT = (6.0, 2)  # (J, cut) representative strong-coupling point
REF_BUILD = {"k": 3, "L": 6, "dt": 0.1}

# Three-spectra panel: weak-coupling and strong-coupling reference points at cut c=2.
THREE_CUT = 2
THREE_WEAK_J = 1.0
THREE_STRONG_J = 6.0
THREE_CUM_CAP = 1.0 - 1e-8
THREE_MAX_MODES = 8
THREE_MIN_MODES = 6
# Display floor for the three-spectra panel: modes below this are numerical noise and are not
# drawn, so the y-axis is not stretched down to the ~1e-16 tail. Stored data is untouched.
THREE_DISPLAY_FLOOR = 1e-8
# (color, line style, marker) per curve; distinct colors, cut is NOT color-encoded here.
THREE_WEAK_RESP_STYLE = {"color": "#E69F00", "ls": "-", "marker": "o"}
THREE_STRONG_RESP_STYLE = {"color": "#D81B60", "ls": "-", "marker": "o"}
THREE_STRONG_PT_STYLE = {"color": "#1E88E5", "ls": "--", "marker": "s"}

TAIL_STEM = "entropy_and_tail_weight_vs_J_refined"
SPECTRUM_STEM = "entropy_and_representative_spectrum"
THREE_STEM = "entropy_and_three_spectra"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _invalid_main_keys(validation_csv: Path) -> set[tuple[float, int]]:
    """Return ``(J, cut)`` main-sweep coordinates flagged invalid in validation.csv."""
    invalid: set[tuple[float, int]] = set()
    if not validation_csv.is_file():
        return invalid
    for row in _load_rows(validation_csv):
        # Restrict to the reference (main-sweep) build coordinates.
        if (
            int(float(row["k"])) != REF_BUILD["k"]
            or int(float(row["L"])) != REF_BUILD["L"]
        ):
            continue
        if abs(float(row["dt"]) - REF_BUILD["dt"]) > 1e-12:
            continue
        if not _truthy(row.get("valid", "True")):
            invalid.add((float(row["J"]), int(float(row["cut"]))))
    return invalid


def _report_excluded(
    scalar_rows: list[dict[str, Any]], invalid_keys: set[tuple[float, int]]
) -> None:
    excluded_report: list[str] = []
    for jv, cut in sorted(invalid_keys):
        excluded_report.append(f"  excluded (validation.csv): J={jv:g}, c={cut}")
    for row in scalar_rows:
        if row.get("run_kind") != "main":
            continue
        if not _truthy(row.get("valid", "True")):
            key = (float(row["J"]), int(float(row["cut"])), row.get("object_type"))
            excluded_report.append(
                f"  excluded (scalar row invalid): J={key[0]:g}, c={key[1]}, object={key[2]}"
            )
    if excluded_report:
        print("Validation mask excluded the following main-sweep points:")
        for line in sorted(set(excluded_report)):
            print(line)
    else:
        print("Validation mask: all main-sweep points valid; none excluded.")


def _main_series(
    scalar_rows: list[dict[str, Any]],
    *,
    object_type: str,
    cut: int,
    invalid_keys: set[tuple[float, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sorted (J, entropy, p1) arrays for valid main-sweep points."""
    picked: list[tuple[float, float, float]] = []
    for row in scalar_rows:
        if row.get("run_kind") != "main":
            continue
        if row.get("object_type") != object_type:
            continue
        if int(float(row["cut"])) != cut:
            continue
        jv = float(row["J"])
        if not _truthy(row.get("valid", "True")):
            continue
        if (jv, cut) in invalid_keys:
            continue
        picked.append((jv, float(row["entropy"]), float(row["p1"])))
    picked.sort(key=lambda t: t[0])
    if not picked:
        return np.empty(0), np.empty(0), np.empty(0)
    arr = np.asarray(picked, dtype=np.float64)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def _apply_paper_rcparams(plt) -> None:  # noqa: ANN001
    configure_matplotlib_prl()
    # Paper-consistent type sizes: axis labels 20-24 pt, ticks 16-19 pt, legend 16-19 pt,
    # panel labels 22-26 pt. Fonts are embedded (Type 42) for the manuscript workflow.
    plt.rcParams.update(
        {
            "font.size": 18.0,
            "axes.labelsize": 22.0,
            "xtick.labelsize": 18.0,
            "ytick.labelsize": 18.0,
            "legend.fontsize": 18.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _plot_line_markers(ax, x, y, *, color, ls, marker, lw=2.4, ms=7.0) -> None:  # noqa: ANN001
    """Plot a connected line plus markers; caller supplies contiguous (no-gap) data."""
    if len(x) == 0:
        return
    ax.plot(x, y, ls=ls, color=color, lw=lw, zorder=2, solid_capstyle="round")
    ax.plot(
        x,
        y,
        linestyle="none",
        marker=marker,
        color=color,
        ms=ms,
        mfc=color,
        mec="0.15",
        mew=0.9,
        zorder=3,
    )


def _style_log_axis(
    ax, *, xlabel, ylabel, label, labelsize, label_xy=(0.035, 0.955), label_va="top"
) -> None:  # noqa: ANN001
    from matplotlib.ticker import LogFormatterMathtext, LogLocator

    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=22.0)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.tick_params(
        direction="in", top=True, right=True, which="both", length=4.5, width=0.9
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    # Panel label; a subtle white box keeps it legible over any nearby data.
    ax.text(
        label_xy[0],
        label_xy[1],
        label,
        transform=ax.transAxes,
        fontsize=24.0,
        fontweight="bold",
        va=label_va,
        ha="left",
        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85},
    )


def _draw_panel_a(ax, scalar_rows, invalid_keys) -> None:  # noqa: ANN001
    """Panel (a): entropy vs J for both objects and all cuts (log y-axis)."""
    for cut in CUTS:
        color = CUT_COLORS[cut]
        for object_type, _label, ls, marker in QUANTITIES:
            j, entropy, _p1 = _main_series(
                scalar_rows, object_type=object_type, cut=cut, invalid_keys=invalid_keys
            )
            # Drop the J=0 point so the weak-coupling ~0 entropy does not collapse the log axis;
            # do not connect across excluded/missing points.
            mask = j > 1e-12
            _plot_line_markers(
                ax, j[mask], entropy[mask], color=color, ls=ls, marker=marker
            )
    _style_log_axis(
        ax, xlabel=r"Coupling $J$", ylabel="Entropy (nats)", label="(a)", labelsize=22.0
    )


def _shared_top_legend(fig) -> None:  # noqa: ANN001
    from matplotlib.lines import Line2D

    cut_handles = [
        Line2D([0], [0], color=CUT_COLORS[c], lw=2.8, label=rf"$c={c}$") for c in CUTS
    ]
    qty_handles = [
        Line2D(
            [0],
            [0],
            color="0.35",
            ls=ls,
            marker=marker,
            lw=2.4,
            ms=7.0,
            mfc="0.35",
            mec="0.15",
            mew=0.9,
            label=label,
        )
        for _obj, label, ls, marker in QUANTITIES
    ]
    fig.legend(
        handles=cut_handles + qty_handles,
        loc="outside upper center",
        ncol=len(cut_handles) + len(qty_handles),
        frameon=False,
        handlelength=2.4,
        columnspacing=1.5,
        handletextpad=0.5,
        borderaxespad=0.2,
    )


def _make_figure(plt):  # noqa: ANN001
    # Both panels share the same physical size; constrained_layout packs the shared legend
    # tightly above the axes without a large empty band.
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.0, 5.6), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.06, h_pad=0.04, wspace=0.05, hspace=0.0)
    return fig, ax_a, ax_b


def build_tail_weight_figure(out_dir: Path, *, png_dpi: int = 400) -> None:
    """Panel (b) = subleading spectral weight 1 - p1 versus J."""
    import matplotlib.pyplot as plt

    scalar_rows = _load_rows(out_dir / "scalar_diagnostics.csv")
    invalid_keys = _invalid_main_keys(out_dir / "validation.csv")
    _report_excluded(scalar_rows, invalid_keys)

    _apply_paper_rcparams(plt)
    fig, ax_a, ax_b = _make_figure(plt)
    _draw_panel_a(ax_a, scalar_rows, invalid_keys)

    for cut in CUTS:
        color = CUT_COLORS[cut]
        for object_type, _label, ls, marker in QUANTITIES:
            j, _entropy, p1 = _main_series(
                scalar_rows, object_type=object_type, cut=cut, invalid_keys=invalid_keys
            )
            mask = j > 1e-12
            # Plotting-only floor for the log axis; stored p1/tail values are untouched.
            tail = np.maximum(1.0 - p1[mask], TAIL_FLOOR)
            _plot_line_markers(ax_b, j[mask], tail, color=color, ls=ls, marker=marker)
    _style_log_axis(
        ax_b,
        xlabel=r"Coupling $J$",
        ylabel=r"Subleading weight, $1 - p_1$",
        label="(b)",
        labelsize=20.0,
    )
    _shared_top_legend(fig)
    _save(fig, out_dir, TAIL_STEM, png_dpi=png_dpi)
    plt.close(fig)


def _spectrum_key(j: float, cut: int, object_type: str) -> str:
    return (
        f"J{j:g}_cut{cut}_k{REF_BUILD['k']}_L{REF_BUILD['L']}"
        f"_dt{REF_BUILD['dt']:g}__{object_type}__sv"
    )


def _select_representative(
    npz, invalid_keys: set[tuple[float, int]], index_entries: list[dict[str, Any]]
) -> tuple[float, int]:
    """Return the representative (J, cut). Prefer REF_POINT if valid and present; otherwise
    fall back to the largest valid J at the same cut that has stored spectra for both objects.
    """
    j_ref, cut_ref = REF_POINT

    def _available(j: float, cut: int) -> bool:
        return all(
            _spectrum_key(j, cut, obj) in npz
            for obj in ("response_centered", "process_tensor")
        )

    if (j_ref, cut_ref) not in invalid_keys and _available(j_ref, cut_ref):
        return j_ref, cut_ref

    # Candidate J values with stored spectra at the reference cut.
    candidate_js = sorted(
        {
            float(e["J"])
            for e in index_entries
            if int(e["cut"]) == cut_ref
            and e["object_type"] in ("response_centered", "process_tensor")
        },
        reverse=True,
    )
    for j in candidate_js:
        if (j, cut_ref) not in invalid_keys and _available(j, cut_ref):
            print(
                f"Representative point J={j_ref:g}, c={cut_ref} unavailable/invalid; "
                f"falling back to largest valid J={j:g}, c={cut_ref}."
            )
            return j, cut_ref
    msg = f"No valid spectrum available at cut c={cut_ref}."
    raise RuntimeError(msg)


def _normalized_weights(npz, j: float, cut: int, object_type: str) -> np.ndarray:
    """Load raw singular values and return p_i = s_i^2 / sum_j s_j^2, sorted descending."""
    sv = np.asarray(npz[_spectrum_key(j, cut, object_type)], dtype=np.float64)
    p = sv**2 / np.sum(sv**2)
    return np.sort(p)[::-1]


def _n_modes_to_plot(p: np.ndarray) -> int:
    """Number of leading modes to consider for display: all with p_i > WEIGHT_FLOOR, but span
    at least MIN_MODES when available. Never pads beyond the stored spectrum length (no zero
    padding). Sub-floor modes within this span are masked out at plot time.
    """
    n_thresh = int(np.count_nonzero(p > WEIGHT_FLOOR))
    return int(min(len(p), max(n_thresh, min(MIN_MODES, len(p)))))


def build_spectrum_figure(out_dir: Path, *, png_dpi: int = 400) -> None:
    """Panel (b) = normalized singular weights at a representative strong-coupling point."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    scalar_rows = _load_rows(out_dir / "scalar_diagnostics.csv")
    invalid_keys = _invalid_main_keys(out_dir / "validation.csv")
    _report_excluded(scalar_rows, invalid_keys)

    npz = np.load(out_dir / "spectra_selected.npz")
    with (out_dir / "spectra_index.json").open(encoding="utf-8") as f:
        index_entries = json.load(f)
    j_rep, cut_rep = _select_representative(npz, invalid_keys, index_entries)
    print(f"Representative singular-spectrum point: J={j_rep:g}, c={cut_rep}")

    _apply_paper_rcparams(plt)
    fig, ax_a, ax_b = _make_figure(plt)
    _draw_panel_a(ax_a, scalar_rows, invalid_keys)

    max_modes = 0
    for object_type in ("response_centered", "process_tensor"):
        style = SPECTRUM_STYLE[object_type]
        p = _normalized_weights(npz, j_rep, cut_rep, object_type)
        n = _n_modes_to_plot(p)
        idx = np.arange(1, len(p) + 1)[:n]
        pv = p[:n]
        # Only plot numerically meaningful modes (p_i > floor); no zero padding.
        keep = pv > WEIGHT_FLOOR
        _plot_line_markers(
            ax_b,
            idx[keep],
            pv[keep],
            color=style["color"],
            ls=style["ls"],
            marker=style["marker"],
        )
        max_modes = max(max_modes, int(idx[keep][-1]) if np.any(keep) else 0)
        print(f"  {object_type}: plotted {int(keep.sum())} modes (p1={p[0]:.4g})")

    _style_log_axis(
        ax_b,
        xlabel=r"Mode index $i$",
        ylabel=r"Normalized singular weight $p_i$",
        label="(b)",
        labelsize=20.0,
    )
    ax_b.set_xlim(0.4, max(MIN_MODES, max_modes) + 0.6)
    ax_b.set_ylim(WEIGHT_FLOOR * 0.5, 2.0)
    ax_b.set_xscale("linear")
    ax_b.annotate(
        rf"$c={cut_rep},\;J={j_rep:g}$",
        xy=(0.97, 0.955),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=18.0,
    )
    spec_handles = [
        Line2D(
            [0],
            [0],
            color=SPECTRUM_STYLE[obj]["color"],
            ls=SPECTRUM_STYLE[obj]["ls"],
            marker=SPECTRUM_STYLE[obj]["marker"],
            lw=2.4,
            ms=7.0,
            mfc=SPECTRUM_STYLE[obj]["color"],
            mec="0.15",
            mew=0.9,
            label=SPECTRUM_STYLE[obj]["label"],
        )
        for obj in ("response_centered", "process_tensor")
    ]
    ax_b.legend(
        handles=spec_handles,
        loc="lower left",
        frameon=False,
        fontsize=16.0,
        handlelength=2.2,
        borderaxespad=0.4,
    )
    # Shared top legend documents the panel-(a) cut/quantity encodings only.
    _shared_top_legend(fig)
    _save(fig, out_dir, SPECTRUM_STEM, png_dpi=png_dpi)
    plt.close(fig)


def _available_js(
    index_entries: list[dict[str, Any]], cut: int, object_type: str
) -> list[float]:
    """Sorted (ascending) J values with stored spectra for the given cut and object type."""
    return sorted(
        {
            float(e["J"])
            for e in index_entries
            if int(e["cut"]) == cut and e["object_type"] == object_type
        }
    )


def _select_spectrum_point(
    npz,
    invalid_keys: set[tuple[float, int]],
    index_entries: list[dict[str, Any]],
    *,
    cut: int,
    object_type: str,
    target_j: float,
    prefer: str,
) -> float:
    """Return a valid J for (cut, object_type). Prefer ``target_j`` when valid and present;
    otherwise fall back per ``prefer``: ``"nearest_weak"`` picks the valid J closest to the
    target (ties favour smaller J), ``"largest"`` picks the largest valid J.
    """

    def _ok(j: float) -> bool:
        return (j, cut) not in invalid_keys and _spectrum_key(
            j, cut, object_type
        ) in npz

    if _ok(target_j):
        return target_j

    candidates = [j for j in _available_js(index_entries, cut, object_type) if _ok(j)]
    if not candidates:
        msg = f"No valid spectrum for object={object_type} at cut c={cut}."
        raise RuntimeError(msg)
    if prefer == "largest":
        chosen = max(candidates)
    else:  # nearest_weak: closest to target, ties favour the smaller (weaker) coupling
        chosen = min(candidates, key=lambda j: (abs(j - target_j), j))
    print(
        f"Requested {object_type} J={target_j:g}, c={cut} unavailable/invalid; "
        f"falling back to J={chosen:g} (rule: {prefer})."
    )
    return chosen


def _n_modes_three(p: np.ndarray) -> int:
    """Per-curve mode count: smallest N with cumulative weight >= THREE_CUM_CAP, capped at
    THREE_MAX_MODES. Never exceeds the stored spectrum length (no zero padding).
    """
    cumulative = np.cumsum(p)
    n_cum = int(np.searchsorted(cumulative, THREE_CUM_CAP) + 1)
    return int(min(len(p), min(n_cum, THREE_MAX_MODES)))


def build_three_spectra_figure(out_dir: Path, *, png_dpi: int = 400) -> None:
    """Panel (b) = three normalized singular spectra at c=2: weak-coupling response,
    strong-coupling response, and strong-coupling process tensor.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    scalar_rows = _load_rows(out_dir / "scalar_diagnostics.csv")
    invalid_keys = _invalid_main_keys(out_dir / "validation.csv")
    _report_excluded(scalar_rows, invalid_keys)

    npz = np.load(out_dir / "spectra_selected.npz")
    with (out_dir / "spectra_index.json").open(encoding="utf-8") as f:
        index_entries = json.load(f)

    j_weak_resp = _select_spectrum_point(
        npz,
        invalid_keys,
        index_entries,
        cut=THREE_CUT,
        object_type="response_centered",
        target_j=THREE_WEAK_J,
        prefer="nearest_weak",
    )
    j_strong_resp = _select_spectrum_point(
        npz,
        invalid_keys,
        index_entries,
        cut=THREE_CUT,
        object_type="response_centered",
        target_j=THREE_STRONG_J,
        prefer="largest",
    )
    j_strong_pt = _select_spectrum_point(
        npz,
        invalid_keys,
        index_entries,
        cut=THREE_CUT,
        object_type="process_tensor",
        target_j=THREE_STRONG_J,
        prefer="largest",
    )

    # (J, object_type, style, label) for the three curves.
    curves = (
        (
            j_weak_resp,
            "response_centered",
            THREE_WEAK_RESP_STYLE,
            rf"$S_V^{{\mathrm{{full}}}}$, $c={THREE_CUT}$, $J={j_weak_resp:g}$",
        ),
        (
            j_strong_resp,
            "response_centered",
            THREE_STRONG_RESP_STYLE,
            rf"$S_V^{{\mathrm{{full}}}}$, $c={THREE_CUT}$, $J={j_strong_resp:g}$",
        ),
        (
            j_strong_pt,
            "process_tensor",
            THREE_STRONG_PT_STYLE,
            rf"$S_{{\mathrm{{PT}}}}^{{\mathrm{{cb}}}}$, $c={THREE_CUT}$, $J={j_strong_pt:g}$",
        ),
    )

    _apply_paper_rcparams(plt)
    fig, ax_a, ax_b = _make_figure(plt)
    _draw_panel_a(ax_a, scalar_rows, invalid_keys)

    weights = [_normalized_weights(npz, j, THREE_CUT, obj) for j, obj, _s, _l in curves]
    n_show = max(_n_modes_three(p) for p in weights)
    n_show = max(n_show, min(THREE_MIN_MODES, min(len(p) for p in weights)))

    handles: list[Any] = []
    for (j, obj, style, label), p in zip(curves, weights):
        n = min(n_show, len(p))
        idx = np.arange(1, n + 1)
        pv = p[:n]
        # Plot only physically meaningful modes (above the numerical-noise display floor);
        # no zero padding.
        keep = pv > THREE_DISPLAY_FLOOR
        _plot_line_markers(
            ax_b,
            idx[keep],
            pv[keep],
            color=style["color"],
            ls=style["ls"],
            marker=style["marker"],
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                ls=style["ls"],
                marker=style["marker"],
                lw=2.4,
                ms=7.0,
                mfc=style["color"],
                mec="0.15",
                mew=0.9,
                label=label,
            )
        )
        print(
            f"  {obj} J={j:g}, c={THREE_CUT}: p1={p[0]:.5f}, 1-p1={1.0 - p[0]:.3e}, "
            f"plotted {int(keep.sum())} modes"
        )

    # Leading modes occupy the top-left; place the "(b)" label at the lower-left instead.
    _style_log_axis(
        ax_b,
        xlabel=r"Mode index $i$",
        ylabel=r"Normalized singular weight $p_i$",
        label="(b)",
        labelsize=20.0,
        label_xy=(0.035, 0.05),
        label_va="bottom",
    )
    ax_b.set_xlim(0.4, n_show + 0.6)
    # Bound the y-axis to the meaningful weight range (avoid the deep numerical-noise tail).
    ax_b.set_ylim(THREE_DISPLAY_FLOOR * 0.5, 2.0)
    ax_b.set_xscale("linear")
    ax_b.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # Small annotation, kept clear of the leading modes and the legend.
    ax_b.annotate(
        rf"$c={THREE_CUT}$",
        xy=(0.5, 0.05),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=18.0,
    )
    ax_b.legend(
        handles=handles,
        loc="upper right",
        frameon=False,
        fontsize=15.0,
        handlelength=2.2,
        borderaxespad=0.4,
        labelspacing=0.35,
    )
    # Shared top legend documents the panel-(a) cut/quantity encodings only.
    _shared_top_legend(fig)
    _save(fig, out_dir, THREE_STEM, png_dpi=png_dpi)
    plt.close(fig)


def _save(fig, out_dir: Path, stem: str, *, png_dpi: int) -> None:  # noqa: ANN001
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = fig_dir / f"{stem}.pdf"
    png_path = fig_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("save/diagnostics_temporal_entropy"),
        help="Diagnostics directory containing the CSV/NPZ/JSON diagnostic outputs.",
    )
    p.add_argument(
        "--figure",
        choices=("three", "spectrum", "tail", "all"),
        default="three",
        help="Which panel-(b) variant to build (default: three-spectra comparison).",
    )
    p.add_argument("--png-dpi", type=int, default=400)
    args = p.parse_args()
    out_dir = args.out_dir.resolve()
    if args.figure in ("tail", "all"):
        build_tail_weight_figure(out_dir, png_dpi=int(args.png_dpi))
    if args.figure in ("spectrum", "all"):
        build_spectrum_figure(out_dir, png_dpi=int(args.png_dpi))
    if args.figure in ("three", "all"):
        build_three_spectra_figure(out_dir, png_dpi=int(args.png_dpi))


if __name__ == "__main__":
    main()
