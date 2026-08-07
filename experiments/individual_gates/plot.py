# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Publication main-text figure for the individual-gates campaign.

(a) Aggregated Pauli-angle results (9 gate–state cases)
(b) Forward CNOT(2→11) vs χ_max
(c,d) Fixed-cap RXX infidelity and convergence (three-state aggregate)
(e,f) Fixed-cap CNOT infidelity and convergence (three-state aggregate)

Uses existing CSVs only. Does not generate a supplementary figure.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common import conventional_median  # noqa: E402
from config import (  # noqa: E402
    CNOT_RANK_CHI_VALUES,
    EXPERIMENT_DIR,
    OUTPUT_DIR,
    Q0,
    Q1,
    REFINEMENT_CHI,
    REFINEMENT_CONTROL,
    REFINEMENT_TARGET,
    REPO_ROOT,
    N,
)

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, NullFormatter
    from matplotlib.transforms import ScaledTranslation
except ModuleNotFoundError as exc:  # pragma: no cover
    msg = "matplotlib is required; install with: uv pip install matplotlib"
    raise SystemExit(msg) from exc

FIGURES_DIR = REPO_ROOT / "experiments" / "figures"
FIGURE_STEM = "figure_individual_gates_main_text"
# Obsolete; retained on disk but not regenerated or referenced.

FIG_WIDTH_MM = 180.0
FIG_HEIGHT_MM = 70.0
MM_TO_IN = 1.0 / 25.4
DPI = 600

ANGLE_XLABEL = r"$\theta/(2\pi)$"
ROTATION_YLABEL = r"Infidelity $1-F$ (Rotation)"
CNOT_YLABEL = r"Infidelity $1-F$ (CNOT)"
RXX_REFINEMENT_YLABEL = r"$1-F$ ($R_{XX}$)"
CNOT_REFINEMENT_YLABEL = r"$1-F$ (CNOT)"

# High-contrast journal palette; marker and line styles provide redundant encoding.
COLOR_TDVP = "#E64B35"
COLOR_MPO = "#00A087"
COLOR_TEBD = "#3C5488"
COLOR_VARIATIONAL = "#CC79A7"

METHODS = ("gate_local_2tdvp", "variational_mpo", "mpo_zipup", "tebd_swap")
METHOD_LABELS = {
    "gate_local_2tdvp": "TDVP",
    "variational_mpo": "Variational MPO",
    "mpo_zipup": "MPO",
    "tebd_swap": "TEBD+SWAP",
}
METHOD_STYLES = {
    "gate_local_2tdvp": {
        "color": COLOR_TDVP,
        "marker": "o",
        "linestyle": "-",
        "fillstyle": "full",
    },
    "mpo_zipup": {
        "color": COLOR_MPO,
        "marker": "s",
        "linestyle": "-.",
        "fillstyle": "full",
    },
    "variational_mpo": {
        "color": COLOR_VARIATIONAL,
        "marker": "D",
        "linestyle": ":",
        "fillstyle": "full",
        "band_alpha": 0.13,
    },
    "tebd_swap": {
        "color": COLOR_TEBD,
        "marker": "^",
        "linestyle": "--",
        "fillstyle": "none",
    },
}

PANEL_LABELS = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")
PANEL_LABEL_ZORDER = 100
PRECISION_THRESHOLD = 1e-14
GUIDE_X_MIN = 1e-4
GUIDE_X_MAX = 1e-2
MAIN_CHI = 8
CNOT_CURVE_CHI = (8, 10, 12, 14, 16)
ROUND_OFF_FLOOR = 1e-16
PAULI_MARKER_X_OFFSET_PT = {
    "gate_local_2tdvp": 3.0,
    "variational_mpo": 0.0,
    "mpo_zipup": -3.0,
    "tebd_swap": 0.0,
}
CNOT_RANK_MARKER_X_OFFSET = {
    "gate_local_2tdvp": 0.18,
    "variational_mpo": 0.06,
    "mpo_zipup": -0.06,
    "tebd_swap": -0.18,
}


def _apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
            "font.size": 7.8,
            "axes.labelsize": 9.0,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.72,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#202020",
            "text.color": "#202020",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "lines.linewidth": 1.6,
            "lines.markersize": 4.4,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _validated_variational_plot_rows(
    variational_rows: list[dict[str, str]],
    campaign_rows: list[dict[str, str]],
    cnot_rank_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Validate the separate control data and convert them to plotting rows."""
    if len(variational_rows) != 87:
        msg = f"Expected 87 variational controls, found {len(variational_rows)}."
        raise ValueError(msg)
    pauli_mpo = {
        (row["gate"], int(row["seed"]), float(row["x"]), int(row["chi_max"])): row
        for row in campaign_rows
        if row["family"] == "pauli" and row["method"] == "mpo_zipup" and int(float(row["n_sub"])) == 1
    }
    cnot_mpo = {
        (
            int(row["control"]),
            int(row["target"]),
            int(row["seed"]),
            int(row["chi_max"]),
        ): row
        for row in cnot_rank_rows
        if row["family"] == "cnot_rank" and row["method"] == "mpo_zipup" and int(float(row["n_sub"])) == 1
    }
    pauli_plot: list[dict[str, str]] = []
    cnot_plot: list[dict[str, str]] = []
    rxx_references: list[dict[str, str]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in variational_rows:
        if row["converged"].lower() != "true":
            msg = f"Nonconverged variational control {row['task_id']}."
            raise ValueError(msg)
        family = row["family"]
        if family == "pauli":
            key = (row["gate"], int(row["seed"]), float(row["x"]), int(row["chi_max"]))
            direct = pauli_mpo.get(key)
        elif family == "cnot_rank":
            key = (
                int(row["control"]),
                int(row["target"]),
                int(row["seed"]),
                int(row["chi_max"]),
            )
            direct = cnot_mpo.get(key)
        else:
            msg = f"Unexpected variational family {family!r}."
            raise ValueError(msg)
        identity = (family, *key)
        if identity in seen or direct is None:
            msg = f"Missing or duplicate direct-MPO match for {identity}."
            raise ValueError(msg)
        seen.add(identity)
        if not np.isclose(
            float(row["mpo_infidelity"]),
            float(direct["infidelity_normalized"]),
            rtol=0.0,
            atol=2e-12,
        ):
            msg = f"Direct-MPO mismatch for variational control {identity}."
            raise ValueError(msg)
        converted = {
            "family": family,
            "gate": row["gate"],
            "control": row["control"],
            "target": row["target"],
            "x": row["x"],
            "seed": row["seed"],
            "chi_max": row["chi_max"],
            "method": "variational_mpo",
            "n_sub": "1",
            "infidelity_normalized": row["variational_infidelity"],
        }
        if family == "pauli":
            pauli_plot.append(converted)
            if row["gate"] == "rxx" and np.isclose(float(row["x"]), 1e-2):
                rxx_references.append(
                    {
                        "method": "variational_mpo",
                        "n_sub": "1",
                        "infidelity": row["variational_infidelity"],
                    }
                )
        else:
            cnot_plot.append(converted)
    if len(pauli_plot) != 72 or len(cnot_plot) != 15 or len(rxx_references) != 3:
        msg = "Variational controls do not cover the expected 72 Pauli, 15 CNOT, and 3 RXX-reference cells."
        raise ValueError(msg)
    return pauli_plot, cnot_plot, rxx_references


def _panel_label(ax: Any, label: str) -> None:
    ax.text(
        0.03,
        0.97,
        label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="top",
        zorder=PANEL_LABEL_ZORDER,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.6},
    )


def aggregated_pauli_series(
    rows: list[dict[str, str]],
    *,
    method: str,
    chi: int = MAIN_CHI,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool 3 Pauli gates × 3 states at each angle (9 cases)."""
    bucket: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if row["family"] != "pauli":
            continue
        if row["method"] != method or int(row["chi_max"]) != chi:
            continue
        if int(float(row["n_sub"])) != 1:
            continue
        bucket[float(row["x"])].append(float(row["infidelity_normalized"]))
    xs = np.array(sorted(bucket), dtype=float)
    med = np.array([conventional_median(bucket[x]) for x in xs], dtype=float)
    ymin = np.array([min(bucket[x]) for x in xs], dtype=float)
    ymax = np.array([max(bucket[x]) for x in xs], dtype=float)
    return xs, med, ymin, ymax


def _plot_pauli_panel(ax: Any, rows: list[dict[str, str]]) -> None:
    plot_order = ["tebd_swap", "mpo_zipup", "variational_mpo", "gate_local_2tdvp"]
    series = {m: aggregated_pauli_series(rows, method=m) for m in plot_order}

    # Small ∝ θ² guide (not a fit).
    xs_g, ys_g, _, _ = series["gate_local_2tdvp"]
    mask = (xs_g >= GUIDE_X_MIN) & (xs_g <= GUIDE_X_MAX) & (ys_g > PRECISION_THRESHOLD)
    if np.any(mask):
        xf, yf = xs_g[mask], ys_g[mask]
        ref_i = len(xf) // 2
        guide_x = np.logspace(np.log10(GUIDE_X_MIN), np.log10(GUIDE_X_MAX), 20)
        guide_y = yf[ref_i] * (guide_x / xf[ref_i]) ** 2 * 0.45
        ax.plot(
            guide_x,
            guide_y,
            linestyle=(0, (2.0, 1.2)),
            color="0.2",
            linewidth=1.0,
            zorder=1,
        )
        ax.text(
            float(guide_x[-2]),
            float(guide_y[-2]) * 0.35,
            r"$\propto\theta^2$",
            fontsize=8.0,
            color="0.2",
            ha="left",
            va="top",
            zorder=2,
        )

    for method in plot_order:
        style = METHOD_STYLES[method]
        xs, med, ymin, ymax = series[method]
        pos = xs > 0.0
        if np.any(pos):
            ax.fill_between(
                xs[pos],
                np.maximum(ymin[pos], 1e-18),
                np.maximum(ymax[pos], 1e-18),
                color=style["color"],
                alpha=style.get("band_alpha", 0.18),
                linewidth=0,
                zorder=2,
            )
            ys = med[pos]
            ax.plot(
                xs[pos],
                ys,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.65,
                zorder=10,
            )
            ax.plot(
                xs[pos],
                ys,
                linestyle="none",
                marker=style["marker"],
                color=style["color"],
                fillstyle=style["fillstyle"],
                markerfacecolor="white" if method == "tebd_swap" else style["color"],
                markeredgecolor=style["color"],
                markeredgewidth=0.9,
                markersize=4.5,
                transform=ax.transData
                + ScaledTranslation(
                    PAULI_MARKER_X_OFFSET_PT[method] / 72.0,
                    0.0,
                    ax.figure.dpi_scale_trans,
                ),
                zorder=11,
            )


def cnot_rank_stats(
    rows: list[dict[str, str]],
    *,
    method: str,
    n_sub: int,
) -> dict[int, tuple[float, float, float]]:
    bucket: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] != method or int(float(row["n_sub"])) != n_sub:
            continue
        bucket[int(row["chi_max"])].append(float(row["infidelity_normalized"]))
    return {chi: (conventional_median(vals), min(vals), max(vals)) for chi, vals in sorted(bucket.items())}


def _range_band_series(
    ax: Any,
    stats: dict[int, tuple[float, float, float]],
    *,
    color: str,
    marker: str,
    linestyle: str,
    fillstyle: str,
    chis: tuple[int, ...],
    marker_x_offset: float = 0.0,
    zorder: int = 10,
) -> None:
    xs = np.array([c for c in chis if c in stats], dtype=float)
    if xs.size == 0:
        return
    med = np.maximum(np.array([stats[int(c)][0] for c in xs]), ROUND_OFF_FLOOR)
    lo = np.maximum(np.array([stats[int(c)][1] for c in xs]), ROUND_OFF_FLOOR)
    hi = np.maximum(np.array([stats[int(c)][2] for c in xs]), ROUND_OFF_FLOOR)
    ax.fill_between(
        xs,
        lo,
        hi,
        color=color,
        alpha=0.18,
        linewidth=0,
        zorder=zorder - 2,
    )
    ax.plot(
        xs,
        med,
        color=color,
        linestyle=linestyle,
        linewidth=1.65,
        zorder=zorder,
    )
    ax.plot(
        xs + marker_x_offset,
        med,
        linestyle="none",
        marker=marker,
        markersize=4.5,
        fillstyle=fillstyle,
        markerfacecolor=color if fillstyle == "full" else "white",
        markeredgecolor=color,
        markeredgewidth=0.9,
        zorder=zorder + 1,
    )


def _plot_cnot_rank_panel(ax: Any, rows: list[dict[str, str]]) -> None:
    tebd = cnot_rank_stats(rows, method="tebd_swap", n_sub=1)
    mpo = cnot_rank_stats(rows, method="mpo_zipup", n_sub=1)
    variational = cnot_rank_stats(rows, method="variational_mpo", n_sub=1)
    tdvp1 = cnot_rank_stats(rows, method="gate_local_2tdvp", n_sub=1)

    _range_band_series(
        ax,
        tebd,
        color=COLOR_TEBD,
        marker="^",
        linestyle=METHOD_STYLES["tebd_swap"]["linestyle"],
        fillstyle=METHOD_STYLES["tebd_swap"]["fillstyle"],
        chis=CNOT_CURVE_CHI,
        marker_x_offset=CNOT_RANK_MARKER_X_OFFSET["tebd_swap"],
        zorder=8,
    )
    _range_band_series(
        ax,
        mpo,
        color=COLOR_MPO,
        marker="s",
        linestyle=METHOD_STYLES["mpo_zipup"]["linestyle"],
        fillstyle="full",
        chis=CNOT_CURVE_CHI,
        marker_x_offset=CNOT_RANK_MARKER_X_OFFSET["mpo_zipup"],
        zorder=9,
    )
    _range_band_series(
        ax,
        variational,
        color=COLOR_VARIATIONAL,
        marker="D",
        linestyle=METHOD_STYLES["variational_mpo"]["linestyle"],
        fillstyle="full",
        chis=CNOT_CURVE_CHI,
        marker_x_offset=CNOT_RANK_MARKER_X_OFFSET["variational_mpo"],
        zorder=10,
    )
    _range_band_series(
        ax,
        tdvp1,
        color=COLOR_TDVP,
        marker="o",
        linestyle="-",
        fillstyle="full",
        chis=CNOT_CURVE_CHI,
        marker_x_offset=CNOT_RANK_MARKER_X_OFFSET["gate_local_2tdvp"],
        zorder=11,
    )


def _cnot_method_stats(rows: list[dict[str, str]], *, method: str) -> tuple[float, float, float] | None:
    def _match(row: dict[str, str], family: str) -> bool:
        if row.get("family") != family or row["method"] != method:
            return False
        if int(row["chi_max"]) != REFINEMENT_CHI:
            return False
        if int(float(row["n_sub"])) != 1:
            return False
        control = row.get("control", "")
        target = row.get("target", "")
        if control not in {"", str(REFINEMENT_CONTROL)} and int(control) != REFINEMENT_CONTROL:
            return False
        return not (target not in {"", str(REFINEMENT_TARGET)} and int(target) != REFINEMENT_TARGET)

    for family in ("cnot_rank", "cnot"):
        vals = [float(row["infidelity_normalized"]) for row in rows if _match(row, family)]
        if vals:
            return conventional_median(vals), min(vals), max(vals)
    return None


def _plot_cnot_refinement_panels(
    ax: Any,
    convergence_ax: Any,
    reference_rows: list[dict[str, str]],
    refinement_rows: list[dict[str, str]],
) -> None:
    for method in ("mpo_zipup", "variational_mpo", "tebd_swap"):
        stats = _cnot_method_stats(reference_rows, method=method)
        if stats is None:
            continue
        style = METHOD_STYLES[method]
        med, lo, hi = stats
        ax.axhspan(lo, hi, color=style["color"], alpha=0.13, linewidth=0, zorder=0)
        ax.axhline(
            med,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.5,
            zorder=1,
        )

    infidelity: dict[int, list[float]] = defaultdict(list)
    adjacent: dict[int, list[float]] = defaultdict(list)
    for row in refinement_rows:
        n_sub = int(float(row["n_sub"]))
        infidelity[n_sub].append(float(row["infidelity_vs_exact"]))
        adj = row.get("adjacent_refinement_distance", "")
        if adj not in {"", None}:
            adjacent[n_sub].append(float(adj))

    ns = np.array(sorted(infidelity), dtype=float)
    inf = np.array([conventional_median(infidelity[int(n)]) for n in ns], dtype=float)
    inf_lo = np.array([min(infidelity[int(n)]) for n in ns], dtype=float)
    inf_hi = np.array([max(infidelity[int(n)]) for n in ns], dtype=float)

    ax.fill_between(ns, inf_lo, inf_hi, color=COLOR_TDVP, alpha=0.18, linewidth=0, zorder=1)
    ax.plot(ns, inf, color=COLOR_TDVP, linewidth=1.65, zorder=2)
    ax.plot(
        ns,
        inf,
        linestyle="none",
        marker="o",
        color=COLOR_TDVP,
        markersize=4.1,
        markeredgewidth=0.85,
        zorder=3,
    )

    # Keep endpoint accuracy and self-convergence visually separate. The
    # adjacent distance ends at n_sub=512 (its partner uses 1024 substeps).
    adj_ns = np.array(sorted(adjacent), dtype=float)
    adj = np.array([conventional_median(adjacent[int(n)]) for n in adj_ns], dtype=float)
    adj_lo = np.array([min(adjacent[int(n)]) for n in adj_ns], dtype=float)
    adj_hi = np.array([max(adjacent[int(n)]) for n in adj_ns], dtype=float)
    inset = convergence_ax
    inset.fill_between(adj_ns, adj_lo, adj_hi, color=COLOR_TDVP, alpha=0.18, linewidth=0, zorder=1)
    inset.plot(adj_ns, adj, color=COLOR_TDVP, linewidth=1.4, linestyle="-", zorder=2)
    inset.plot(
        adj_ns,
        adj,
        linestyle="none",
        marker="o",
        color=COLOR_TDVP,
        markersize=3.4,
        markeredgewidth=0.8,
        zorder=3,
    )
    inset.set_xscale("log", base=2)
    inset.set_yscale("log")
    inset.set_xlim(0.8, 700)
    inset.set_ylim(1.2e-4, 0.32)
    inset.set_xticks([1, 16, 256])
    inset.set_xticklabels(["1", "16", "256"])
    inset.set_yticks([1e-4, 1e-2, 1e-1])
    inset.set_xlabel(r"Substeps $n_{\mathrm{sub}}$")
    inset.set_ylabel(r"$D_n$")
    inset.grid(True, which="major", axis="y", color="0.92", linewidth=0.3)
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)


def _plot_rxx_refinement_panels(ax: Any, convergence_ax: Any, rows: list[dict[str, str]]) -> None:
    for method in ("mpo_zipup", "variational_mpo", "tebd_swap"):
        vals = [float(row["infidelity"]) for row in rows if row["method"] == method and int(float(row["n_sub"])) == 1]
        med, lo, hi = conventional_median(vals), min(vals), max(vals)
        style = METHOD_STYLES[method]
        ax.axhspan(lo, hi, color=style["color"], alpha=0.13, linewidth=0, zorder=0)
        ax.axhline(med, color=style["color"], linestyle=style["linestyle"], linewidth=1.5, zorder=1)

    infidelity: dict[int, list[float]] = defaultdict(list)
    adjacent: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] != "gate_local_2tdvp":
            continue
        n_sub = int(float(row["n_sub"]))
        infidelity[n_sub].append(float(row["infidelity"]))
        adj = row.get("adjacent_refinement_distance", "")
        if adj not in {"", None}:
            adjacent[n_sub].append(float(adj))

    ns = np.array(sorted(infidelity), dtype=float)
    inf = np.array([conventional_median(infidelity[int(n)]) for n in ns], dtype=float)
    inf_lo = np.array([min(infidelity[int(n)]) for n in ns], dtype=float)
    inf_hi = np.array([max(infidelity[int(n)]) for n in ns], dtype=float)
    ax.fill_between(ns, inf_lo, inf_hi, color=COLOR_TDVP, alpha=0.18, linewidth=0, zorder=1)
    ax.plot(
        ns,
        inf,
        color=COLOR_TDVP,
        linewidth=1.65,
        marker="o",
        markersize=4.1,
        markeredgewidth=0.85,
        zorder=3,
    )

    adj_ns = np.array(sorted(adjacent), dtype=float)
    adj = np.array([conventional_median(adjacent[int(n)]) for n in adj_ns], dtype=float)
    adj_lo = np.array([min(adjacent[int(n)]) for n in adj_ns], dtype=float)
    adj_hi = np.array([max(adjacent[int(n)]) for n in adj_ns], dtype=float)
    inset = convergence_ax
    inset.fill_between(adj_ns, adj_lo, adj_hi, color=COLOR_TDVP, alpha=0.18, linewidth=0, zorder=1)
    inset.plot(
        adj_ns,
        adj,
        color=COLOR_TDVP,
        linewidth=1.4,
        marker="o",
        markersize=3.4,
        markeredgewidth=0.8,
        zorder=2,
    )
    inset.set_xscale("log", base=2)
    inset.set_yscale("log")
    inset.set_xlim(0.8, 700)
    inset.set_ylim(5e-8, 2e-4)
    inset.set_xticks([1, 16, 256])
    inset.set_xticklabels(["1", "16", "256"])
    inset.set_yticks([1e-7, 1e-5, 1e-4])
    inset.set_xlabel(r"Substeps $n_{\mathrm{sub}}$")
    inset.set_ylabel(r"$D_n$")
    inset.grid(True, which="major", axis="y", color="0.92", linewidth=0.3)
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)


def plot_main_figure(
    campaign_rows: list[dict[str, str]],
    cnot_rank_rows: list[dict[str, str]],
    refinement_rows: list[dict[str, str]],
    rxx_refinement_rows: list[dict[str, str]],
) -> Any:
    _apply_style()
    fig = plt.figure(figsize=(FIG_WIDTH_MM * MM_TO_IN, FIG_HEIGHT_MM * MM_TO_IN))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.12, 1.02, 1.52), wspace=0.40)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    refinement_grid = gs[0, 2].subgridspec(2, 2, wspace=0.55, hspace=0.38)
    ax_c = fig.add_subplot(refinement_grid[0, 0])
    ax_d = fig.add_subplot(refinement_grid[0, 1])
    ax_e = fig.add_subplot(refinement_grid[1, 0])
    ax_f = fig.add_subplot(refinement_grid[1, 1])

    # --- (a) Aggregated Pauli ---
    _plot_pauli_panel(ax_a, campaign_rows)
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlim(7.5e-5, 0.35)
    pauli_vals = [
        float(r["infidelity_normalized"])
        for r in campaign_rows
        if r["family"] == "pauli" and int(r["chi_max"]) == MAIN_CHI and float(r["infidelity_normalized"]) > 0
    ]
    y_hi_a = min(1.0, max(pauli_vals) * 1.5) if pauli_vals else 1.0
    ax_a.set_ylim(2e-8, y_hi_a)
    ax_a.set_xlabel(ANGLE_XLABEL)
    ax_a.set_ylabel(ROTATION_YLABEL)
    ax_a.set_xticks([1e-4, 1e-3, 1e-2, 1e-1])
    ax_a.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax_a.xaxis.set_minor_formatter(NullFormatter())
    ax_a.yaxis.set_major_locator(LogLocator(base=10.0))
    ax_a.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax_a.yaxis.set_minor_formatter(NullFormatter())
    ax_a.grid(True, which="major", axis="y", color="#E6E8EB", linewidth=0.45)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    handles_a = [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLES[m]["color"],
            marker=METHOD_STYLES[m]["marker"],
            linestyle=METHOD_STYLES[m]["linestyle"],
            fillstyle=METHOD_STYLES[m]["fillstyle"],
            markerfacecolor="white" if m == "tebd_swap" else METHOD_STYLES[m]["color"],
            markeredgecolor=METHOD_STYLES[m]["color"],
            markeredgewidth=0.9,
            markersize=4.5,
            label=METHOD_LABELS[m],
        )
        for m in METHODS
    ]
    ax_a.legend(
        handles=handles_a,
        loc="lower right",
        frameon=False,
        fontsize=6.8,
        handlelength=1.5,
        ncol=1,
        labelspacing=0.22,
    )
    _panel_label(ax_a, PANEL_LABELS[0])

    # --- (b) CNOT rank ---
    _plot_cnot_rank_panel(ax_b, cnot_rank_rows)
    ax_b.set_yscale("log")
    ax_b.set_xlim(7.3, 16.7)
    ax_b.set_xticks(list(CNOT_RANK_CHI_VALUES))
    ax_b.set_xticklabels([str(c) for c in CNOT_RANK_CHI_VALUES])
    ax_b.set_ylim(5e-4, 0.7)
    ax_b.set_xlabel(r"$\chi_{\max}$")
    ax_b.set_ylabel(CNOT_YLABEL)
    ax_b.yaxis.set_major_locator(LogLocator(base=10.0))
    ax_b.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax_b.yaxis.set_minor_formatter(NullFormatter())
    ax_b.grid(True, which="major", axis="y", color="#E6E8EB", linewidth=0.45)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    _panel_label(ax_b, PANEL_LABELS[1])

    # --- (c,d) Fixed-cap RXX refinement ---
    _plot_rxx_refinement_panels(ax_c, ax_d, rxx_refinement_rows)
    ax_c.set_xscale("log", base=2)
    ax_c.set_yscale("log")
    ax_c.set_xlim(0.8, 1400)
    ax_c.set_ylim(2.8e-4, 0.5)
    refinement_ticks = (1, 32, 1024)
    ax_c.set_xticks(list(refinement_ticks))
    ax_c.set_xticklabels([])
    ax_c.set_ylabel(RXX_REFINEMENT_YLABEL)
    ax_c.yaxis.set_major_locator(LogLocator(base=10.0))
    ax_c.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax_c.yaxis.set_minor_formatter(NullFormatter())
    ax_c.grid(True, which="major", axis="y", color="#E6E8EB", linewidth=0.45)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    _panel_label(ax_c, PANEL_LABELS[2])
    ax_d.set_xlabel("")
    ax_d.set_xticklabels([])
    _panel_label(ax_d, PANEL_LABELS[3])

    # --- (e,f) Fixed-cap CNOT refinement ---
    ref_rows = cnot_rank_rows + campaign_rows
    _plot_cnot_refinement_panels(ax_e, ax_f, ref_rows, refinement_rows)
    ax_e.set_xscale("log", base=2)
    ax_e.set_yscale("log")
    ax_e.set_xlim(0.8, 1400)
    ax_e.set_ylim(2e-2, 0.5)
    ax_e.set_xticks(list(refinement_ticks))
    ax_e.set_xticklabels([str(v) for v in refinement_ticks])
    ax_e.set_xlabel(r"Substeps $n_{\mathrm{sub}}$")
    ax_e.set_ylabel(CNOT_REFINEMENT_YLABEL)
    ax_e.set_yticks([0.03, 0.1, 0.3])
    ax_e.set_yticklabels(["0.03", "0.1", "0.3"])
    ax_e.yaxis.set_minor_formatter(NullFormatter())
    ax_e.grid(True, which="major", axis="y", color="#E6E8EB", linewidth=0.45)
    ax_e.spines["top"].set_visible(False)
    ax_e.spines["right"].set_visible(False)
    _panel_label(ax_e, PANEL_LABELS[4])
    _panel_label(ax_f, PANEL_LABELS[5])

    fig.subplots_adjust(left=0.067, right=0.995, bottom=0.165, top=0.95)
    return fig


def write_main_caption(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Individual-gates main-text figure caption (draft)",
                "",
                "\\textbf{Single-gate accuracy and path refinement.}",
                f"Long-range gates between sites {Q0 + 1} and {Q1 + 1} of $N={N}$ MPS are compared with",
                "dense application of the same gate. (a) Median infidelity for $R_{XX}$,",
                "$R_{YY}$, and $R_{ZZ}$ at $\\chi_{\\max}=8$ and $n_{\\mathrm{sub}}=1$; bands",
                "span three gates and three states and are not confidence intervals. The",
                "$\\theta^2$ line is a guide, not a fit. The identity control is reported",
                "separately because $\\theta=0$ cannot lie on the logarithmic axis; TEBD+SWAP",
                "still executes the routing sequence in this diagnostic. (b) CNOT infidelity",
                "versus $\\chi_{\\max}$ with an effectively zero SVD threshold and",
                "$n_{\\mathrm{sub}}=1$ for TDVP. Symbols show three-state medians and bands",
                "their full ranges. Variational MPO is the locally converged alternating-sweep",
                "endpoint fit; all 87 controls converged without fallback. Markers in (a,b)",
                "are offset where needed to expose overlapping medians. Curves and bands",
                "remain at the stated coordinates.",
                "The curves include the $\\chi_{\\max}=16$ endpoints,",
                "which lie below the displayed range and therefore continue out of frame.",
                "(c,d) Fixed-cap $R_{XX}$ infidelity and convergence at",
                "$\\theta/(2\\pi)=10^{-2}$; (e,f) the corresponding CNOT results, all at",
                "$\\chi_{\\max}=8$. Curves and horizontal lines show three-state medians,",
                "with full-range bands. Panels (d,f) show",
                "$D(n)=\\min_\\phi\\|\\Psi_n-e^{i\\phi}\\Psi_{2n}\\|_2$, where $n$ is the",
                "number of substeps. Refinement stabilizes",
                "the cap-constrained TDVP path without driving it to the exact gate endpoint.",
                "The states are normalized to roundoff, so",
                "$D(n)=\\sqrt{2-2|\\langle\\Psi_n|\\Psi_{2n}\\rangle|}$; this is a",
                "phase-aligned state-vector distance, not an infidelity.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _save_figure(fig: Any, stem: str, figures_dir: Path, output_dir: Path) -> None:
    """Write the canonical figure; derived campaign-local copies are unnecessary."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    pdf = figures_dir / f"{stem}.pdf"
    png = figures_dir / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=DPI)
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")
    output_dir.mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args(argv)

    campaign_csv = args.output_dir / "campaign_rows.csv"
    cnot_csv = args.output_dir / "cnot_rank_rows.csv"
    refinement_csv = args.output_dir / "refinement_rows.csv"
    rxx_refinement_csv = args.output_dir / "rxx_refinement_comparison_rows.csv"
    variational_csv = args.output_dir / "variational_mpo_control" / "single_gate_rows.csv"
    for path in (campaign_csv, cnot_csv, refinement_csv, rxx_refinement_csv, variational_csv):
        if not path.is_file():
            raise SystemExit(f"Missing {path}")

    campaign_rows = _read_csv(campaign_csv)
    cnot_rows = _read_csv(cnot_csv)
    pauli_variational, cnot_variational, rxx_variational = _validated_variational_plot_rows(
        _read_csv(variational_csv),
        campaign_rows,
        cnot_rows,
    )
    fig = plot_main_figure(
        campaign_rows + pauli_variational,
        cnot_rows + cnot_variational,
        _read_csv(refinement_csv),
        _read_csv(rxx_refinement_csv) + rxx_variational,
    )
    _save_figure(fig, FIGURE_STEM, args.figures_dir, args.output_dir)
    plt.close(fig)

    caption_path = args.output_dir / f"{FIGURE_STEM}_caption.md"
    write_main_caption(caption_path)
    print(f"Wrote {caption_path}")

    _ = EXPERIMENT_DIR
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
