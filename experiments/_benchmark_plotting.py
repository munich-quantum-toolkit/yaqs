# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Matplotlib helpers for operational-memory experiment benchmarks."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Fixed log color limits for heatmaps (values clip to ends of the scale).
HEATMAP_COLOR_VMIN = 1e-3
HEATMAP_COLOR_VMAX = 3.0

# Representative cuts / couplings for multi-panel figures.
PANEL2_FIXED_CUTS: tuple[int, ...] = (10, 15, 19)
PANEL3_TARGET_JS: tuple[float, ...] = (0.4, 1.0, 2.0)
PANEL2_FIXED_TAUS: tuple[int, ...] = (0, 1, 2, 4, 8)
PANEL_TARGET_JS_ELL: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)


def configure_matplotlib_prl() -> None:
    """Physical Review Letters–oriented rc (single-column width, serif)."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "grid.alpha": 0.0,
            "lines.linewidth": 1.5,
            "lines.markersize": 3.5,
        }
    )


def configure_matplotlib_simple() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "axes.linewidth": 0.8,
            "grid.alpha": 0.25,
            "font.family": "sans-serif",
            "mathtext.default": "it",
        }
    )


def savefig_base(fig: object, path_stem: Path, *, dpi: int = 300) -> None:
    import matplotlib.pyplot as plt

    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _truncate_cmap(name: str, lo: float = 0.10, hi: float = 0.85, n: int = 256):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    base = plt.get_cmap(name)
    return LinearSegmentedColormap.from_list(
        f"{name}_trunc_{lo:.2f}_{hi:.2f}",
        base(np.linspace(lo, hi, n)),
    )


def _mesh_edges(arr: np.ndarray) -> np.ndarray:
    if arr.size >= 2:
        d = float(np.median(np.diff(arr)))
        return np.concatenate([[arr[0] - d / 2], (arr[:-1] + arr[1:]) / 2, [arr[-1] + d / 2]])
    return np.array([arr[0] - 0.5, arr[0] + 0.5])


def plot_entropy_vs_j(rows: list[dict[str, float | int]], out_stem: Path, *, k: int) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    if not rows:
        return
    configure_matplotlib_simple()
    cuts = sorted({int(r["cut"]) for r in rows})
    j_vals = sorted({float(r["J"]) for r in rows})
    c_mid = int(k) // 2

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=min(cuts), vmax=max(cuts))
    fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.0), constrained_layout=True)

    for c in cuts:
        sub = sorted((r for r in rows if int(r["cut"]) == c), key=lambda r: float(r["J"]))
        xs = [float(r["J"]) for r in sub]
        ys = [float(r["entropy"]) for r in sub]
        color = cmap(norm(c))
        lw = 2.8 if c == c_mid else 1.4
        label = f"c={c} (k/2 baseline)" if c == c_mid else None
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=0.9 if c == c_mid else 0.85, label=label)

    ax.set_xlabel(r"$J$")
    ax.set_ylabel(r"$S_V$ (linear $w_{ij}$)")
    ax.set_xticks(j_vals)
    ax.grid(True, axis="y")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.95)
    cbar.ax.set_ylabel("cut c")
    if any(c == c_mid for c in cuts):
        ax.legend(frameon=False, loc="best")
    savefig_base(fig, out_stem)


def plot_entropy_heatmap_cut_vs_j(rows: list[dict[str, str | float | int]], out_stem: Path) -> None:
    """Three-panel figure: heatmap cut×J plus two cross-sections."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator

    if not rows:
        return

    cuts = sorted({int(float(r["cut"])) for r in rows})
    j_vals = sorted({float(r["J"]) for r in rows})
    k_max = max(cuts)
    z = np.full((len(cuts), len(j_vals)), np.nan, dtype=np.float64)
    for r in rows:
        z[cuts.index(int(float(r["cut"]))), j_vals.index(float(r["J"]))] = float(r["entropy"])

    j_arr = np.asarray(j_vals, dtype=np.float64)
    c_arr = np.asarray(cuts, dtype=np.float64)
    j_edges = _mesh_edges(j_arr)
    c_edges = _mesh_edges(c_arr)

    def nearest_j(target: float) -> float:
        return float(j_arr[int(np.argmin(np.abs(j_arr - float(target))))])

    configure_matplotlib_prl()
    fig = plt.figure(figsize=(8.2, 4.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.06, hspace=0.10)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])

    z_plot = np.where(
        np.isfinite(z),
        np.where(z <= 0.0, HEATMAP_COLOR_VMIN * 0.1, np.maximum(z, HEATMAP_COLOR_VMIN)),
        np.nan,
    )
    z_mesh = np.ma.masked_invalid(np.transpose(z_plot))
    norm = LogNorm(vmin=HEATMAP_COLOR_VMIN, vmax=HEATMAP_COLOR_VMAX)
    cmap = _truncate_cmap("magma", 0.08, 0.78).copy()
    cmap.set_under(color="black")
    cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))

    im = ax0.pcolormesh(
        c_edges,
        j_edges,
        z_mesh,
        cmap=cmap,
        norm=norm,
        shading="auto",
        linewidth=0,
        edgecolors="none",
        antialiased=False,
        rasterized=True,
    )
    ax0.set_xlabel(r"Causal cut $c$")
    ax0.set_ylabel(r"Coupling $J$")
    cbar = fig.colorbar(im, ax=ax0, shrink=0.92, pad=0.012, aspect=18)
    cbar.ax.set_title(r"$S_V$", fontsize=16, pad=3)
    cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    panel2_cuts = [c for c in PANEL2_FIXED_CUTS if c in cuts]
    if len(panel2_cuts) < 3:
        idx = np.linspace(0, len(cuts) - 1, min(5, len(cuts))).round().astype(int)
        panel2_cuts = [cuts[i] for i in idx]
    panel3_js = [nearest_j(v) for v in PANEL3_TARGET_JS]

    panel2_cmap = _truncate_cmap("Blues", 0.35, 0.95)
    panel2_norm = Normalize(vmin=1.0, vmax=float(k_max))
    for c_sel in panel2_cuts:
        sub = sorted((r for r in rows if int(float(r["cut"])) == c_sel), key=lambda r: float(r["J"]))
        if not sub:
            continue
        ax1.semilogy(
            [float(r["J"]) for r in sub],
            [max(float(r["entropy"]), HEATMAP_COLOR_VMIN) for r in sub],
            color=panel2_cmap(panel2_norm(float(c_sel))),
            lw=2.0,
            marker="o",
            ms=3.8,
            label=rf"$c={c_sel}$",
        )
    ax1.set_xlabel(r"Coupling $J$")
    ax1.set_ylabel(r"$S_V$")
    ax1.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax1.yaxis.set_minor_locator(NullLocator())
    ax1.grid(True, which="major", axis="y", alpha=0.12, linewidth=0.45)
    ax1.legend(frameon=False, loc="lower right", fontsize=7.5)

    panel3_cmap = _truncate_cmap("Reds", 0.30, 0.95)
    panel3_norm = Normalize(vmin=0.0, vmax=2.0)
    for j_use in sorted(panel3_js):
        ji = j_vals.index(j_use)
        ys = [max(float(z[ci, ji]), HEATMAP_COLOR_VMIN) for ci in range(len(cuts))]
        ax2.semilogy(cuts, ys, lw=1.9, marker="o", ms=5.0, color=panel3_cmap(panel3_norm(j_use)), label=rf"$J={j_use:g}$")
    ax2.set_xlabel(r"Causal cut $c$")
    ax2.set_ylabel(r"$S_V$")
    ax2.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax2.yaxis.set_minor_locator(NullLocator())
    ax2.grid(True, which="major", axis="y", alpha=0.12, linewidth=0.45)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=3, frameon=False, fontsize=8.0)

    y_vals = [max(float(r["entropy"]), HEATMAP_COLOR_VMIN) for r in rows]
    y_hi = min(1.0, max(HEATMAP_COLOR_VMIN * 1.2, float(np.nanmax(y_vals)) * 1.25 if y_vals else 1.0))
    ax1.set_ylim(HEATMAP_COLOR_VMIN, y_hi)
    ax2.set_ylim(HEATMAP_COLOR_VMIN, y_hi)

    for ax, tag in ((ax0, "(a)"), (ax1, "(b)"), (ax2, "(c)")):
        ax.text(0.04, 0.955, tag, transform=ax.transAxes, va="top", ha="left", fontsize=13, fontweight="bold")

    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    prl_stem = out_stem.with_name(f"{out_stem.name}_prl")
    fig.savefig(prl_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    fig.savefig(prl_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)


def plot_entropy_heatmap_middlecut_vs_l_k(rows: list[dict[str, str | float | int]], out_stem: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator

    if not rows:
        return
    configure_matplotlib_prl()
    ls = sorted({int(float(r["L"])) for r in rows})
    ks = sorted({int(float(r["k"])) for r in rows})
    z = np.full((len(ls), len(ks)), np.nan, dtype=np.float64)
    for r in rows:
        z[ls.index(int(float(r["L"]))), ks.index(int(float(r["k"])))] = float(r["entropy"])

    l_arr = np.asarray(ls, dtype=np.float64)
    k_arr = np.asarray(ks, dtype=np.float64)
    l_edges = _mesh_edges(l_arr)
    k_edges = _mesh_edges(k_arr)

    fig = plt.figure(figsize=(8.2, 4.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.06, hspace=0.10)
    ax0, ax1, ax2 = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])

    z_plot = np.where(np.isfinite(z), np.where(z <= 0.0, HEATMAP_COLOR_VMIN * 0.1, np.maximum(z, HEATMAP_COLOR_VMIN)), np.nan)
    z_mesh = np.ma.masked_invalid(np.transpose(z_plot))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_under(color="black")
    im = ax0.pcolormesh(
        l_edges,
        k_edges,
        z_mesh,
        cmap=cmap,
        norm=LogNorm(vmin=HEATMAP_COLOR_VMIN, vmax=HEATMAP_COLOR_VMAX),
        shading="auto",
        rasterized=True,
    )
    ax0.set_xlabel(r"Total sites $L$")
    ax0.set_ylabel(r"Timestep count $k$")
    cbar = fig.colorbar(im, ax=ax0, shrink=0.92, pad=0.012, aspect=18)
    cbar.ax.set_title(r"$S_V$", fontsize=16, pad=3)
    cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    rep_ls = [v for v in (2, 4, 6, 8, 10) if v in ls] or ls[:3]
    panel2_cmap = plt.get_cmap("Blues")
    panel2_norm = Normalize(vmin=float(min(ls)), vmax=float(max(ls)))
    for lv in rep_ls:
        sub = sorted((r for r in rows if int(float(r["L"])) == lv), key=lambda r: int(float(r["k"])))
        ax1.semilogy(
            [int(float(r["k"])) for r in sub],
            [max(float(r["entropy"]), HEATMAP_COLOR_VMIN) for r in sub],
            color=panel2_cmap(panel2_norm(float(lv))),
            lw=1.8,
            marker="o",
            ms=3.6,
            label=rf"$L={lv}$",
        )
    ax1.set_xlabel(r"Timestep count $k$")
    ax1.set_ylabel(r"$S_V$")
    ax1.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax1.legend(frameon=False, fontsize=7.0)

    rep_ks = [v for v in (10, 20, 40, 80, 100) if v in ks] or ks[:: max(1, len(ks) // 3)]
    panel3_cmap = plt.get_cmap("Reds")
    panel3_norm = Normalize(vmin=float(min(ks)), vmax=float(max(ks)))
    for kv in rep_ks:
        ki = ks.index(kv)
        ys = [max(float(z[li, ki]), HEATMAP_COLOR_VMIN) for li in range(len(ls))]
        ax2.semilogy(ls, ys, color=panel3_cmap(panel3_norm(float(kv))), lw=1.8, marker="o", ms=4.0, label=rf"$k={kv}$")
    ax2.set_xlabel(r"Total sites $L$")
    ax2.set_ylabel(r"$S_V$")
    ax2.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax2.legend(frameon=False, fontsize=7.0)

    y_hi = min(1.0, max(HEATMAP_COLOR_VMIN * 1.2, float(np.nanmax([float(r["entropy"]) for r in rows])) * 1.25))
    ax1.set_ylim(HEATMAP_COLOR_VMIN, y_hi)
    ax2.set_ylim(HEATMAP_COLOR_VMIN, y_hi)

    for ax, tag in ((ax0, "(a)"), (ax1, "(b)"), (ax2, "(c)")):
        ax.text(0.04, 0.955, tag, transform=ax.transAxes, va="top", ha="left", fontsize=13, fontweight="bold")

    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)


def plot_convergence_sv_vs_m(summary_rows: list[dict[str, str | float | int]], out_stem: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.ticker import LogLocator, NullLocator

    configure_matplotlib_prl()
    fig, ax = plt.subplots(1, 1, figsize=(3.35, 2.45), constrained_layout=True)
    js = sorted({float(r["J"]) for r in summary_rows})
    norm = Normalize(vmin=min(js), vmax=max(js)) if js else Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap("viridis")
    m_vals = sorted({int(float(r["m"])) for r in summary_rows})

    for jv in js:
        sub = sorted([r for r in summary_rows if abs(float(r["J"]) - jv) < 1e-12], key=lambda r: int(float(r["m"])))
        xs = np.asarray([int(float(r["m"])) for r in sub], dtype=np.float64)
        mu = np.clip(np.asarray([float(r["entropy_mean"]) for r in sub], dtype=np.float64), 1e-30, None)
        col = cmap(norm(jv))
        ax.semilogy(xs, mu, color=col, lw=0.9, marker="o", ms=2.2, alpha=0.9)

    ax.set_xlabel(r"Probe budget $m$ ($N_p=N_f$)")
    ax.set_ylabel(r"$S_V$")
    ax.set_xticks(m_vals)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(True, which="major", axis="y", alpha=0.1, linewidth=0.3)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.9).ax.set_title(r"$J$", pad=2)
    fig.savefig(out_stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_spectrum_and_rank_vs_j(
    *,
    rank_rows: list[dict[str, str | float | int]],
    spectrum_probs: dict[str, dict[str, np.ndarray]],
    out_stem: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    configure_matplotlib_prl()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.2, 2.45), constrained_layout=True, gridspec_kw={"width_ratios": [1.15, 1.0]})

    js = sorted(float(k) for k in spectrum_probs)
    max_len = max((int(np.asarray(spectrum_probs[f"{j:g}"]["p_mean"]).size) for j in js), default=1)
    z = np.full((len(js), max_len), np.nan, dtype=np.float64)
    for i, jv in enumerate(js):
        p = np.asarray(spectrum_probs[f"{jv:g}"]["p_mean"], dtype=np.float64)
        z[i, : p.size] = np.clip(p, 1e-30, None)

    x = np.arange(1, max_len + 1, dtype=np.float64)
    y = np.asarray(js, dtype=np.float64)
    x_edges = np.concatenate(([0.5], 0.5 * (x[:-1] + x[1:]), [x[-1] + 0.5])) if x.size > 1 else np.array([0.5, 1.5])
    y_edges = _mesh_edges(y) if y.size else np.array([-0.05, 0.05])

    im = ax0.pcolormesh(x_edges, y_edges, z, cmap="cividis", norm=LogNorm(vmin=1e-16, vmax=1.0), shading="auto", rasterized=True)
    ax0.set_xlabel(r"Mode index $n$")
    ax0.set_ylabel(r"$J$")
    fig.colorbar(im, ax=ax0, pad=0.02, shrink=0.9).ax.set_title(r"$p_n$", pad=2)

    rr = sorted(rank_rows, key=lambda r: float(r["J"]))
    xs = np.asarray([float(r["J"]) for r in rr], dtype=np.float64)
    mu = np.asarray([float(r["R_mean"]) for r in rr], dtype=np.float64)
    sd = np.asarray([float(r.get("R_std", 0.0)) for r in rr], dtype=np.float64)
    ax1.plot(xs, mu, color="#1f77b4", lw=1.4, marker="o", ms=2.6)
    ax1.fill_between(xs, np.clip(mu - sd, 0.0, None), mu + sd, color="#1f77b4", alpha=0.14, linewidth=0)
    ax1.set_xlabel(r"$J$")
    ax1.set_ylabel(r"$R=\#\{s_n>10^{-16}\}$")
    ax1.grid(True, axis="y", alpha=0.1, linewidth=0.3)

    for ax, tag in ((ax0, "(a)"), (ax1, "(b)")):
        ax.text(0.04, 0.955, tag, transform=ax.transAxes, va="top", ha="left", fontsize=7.3, fontweight="semibold")

    fig.savefig(out_stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_figure_prl(fig: object, out_stem: Path) -> None:
    """Save PDF/PNG plus ``_prl`` variants (mc-process figure naming)."""
    import matplotlib.pyplot as plt

    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    prl_stem = out_stem.with_name(f"{out_stem.name}_prl")
    fig.savefig(prl_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    fig.savefig(prl_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)


def last_tau_above_for_panel_b(
    z: np.ndarray, taus: list[int], ji: int, *, threshold: float
) -> tuple[float, str]:
    """Last τ (in increasing order) with S_V ≥ threshold for panel (c) of the gap figure."""
    if not taus:
        return float("nan"), "nodata"
    thr = float(threshold)
    has_any = False
    last_above: float | None = None
    for gi, t in enumerate(taus):
        v = z[gi, ji]
        if not np.isfinite(v):
            continue
        has_any = True
        if float(v) >= thr:
            last_above = float(t)
    if not has_any:
        return float("nan"), "nodata"
    if last_above is not None:
        return last_above, "hit"
    return 0.0, "censored"


def plot_entropy_heatmap_gap_vs_j(rows: list[dict[str, str | float | int]], out_stem: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator

    if not rows:
        return
    configure_matplotlib_prl()
    taus = sorted({int(float(r["tau"])) for r in rows})
    j_vals = sorted({float(r["J"]) for r in rows})
    z = np.full((len(taus), len(j_vals)), np.nan, dtype=np.float64)
    for r in rows:
        gi = taus.index(int(float(r["tau"])))
        ji = j_vals.index(float(r["J"]))
        z[gi, ji] = float(r["entropy"])

    g_arr = np.asarray(taus, dtype=np.float64)
    j_arr = np.asarray(j_vals, dtype=np.float64)
    g_edges = (
        np.concatenate([[g_arr[0] - 0.5], 0.5 * (g_arr[:-1] + g_arr[1:]), [g_arr[-1] + 0.5]])
        if len(g_arr) > 1
        else np.array([-0.5, 0.5])
    )
    if len(j_arr) > 1:
        dj = float(np.median(np.diff(j_arr)))
        j_edges = np.concatenate([[j_arr[0] - dj / 2], 0.5 * (j_arr[:-1] + j_arr[1:]), [j_arr[-1] + dj / 2]])
    else:
        j_edges = np.array([j_arr[0] - 0.05, j_arr[0] + 0.05])

    fig = plt.figure(figsize=(8.2, 4.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.06, hspace=0.10)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])

    z_plot = np.where(np.isfinite(z), np.where(z <= 0.0, HEATMAP_COLOR_VMIN * 0.1, np.maximum(z, HEATMAP_COLOR_VMIN)), np.nan)
    z_mesh = np.ma.masked_invalid(np.transpose(z_plot))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_under(color="black")
    cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
    im = ax0.pcolormesh(
        g_edges,
        j_edges,
        z_mesh,
        cmap=cmap,
        norm=LogNorm(vmin=HEATMAP_COLOR_VMIN, vmax=HEATMAP_COLOR_VMAX),
        shading="auto",
        linewidth=0,
        rasterized=True,
    )
    ax0.set_xlabel(r"Delay $\tau$")
    ax0.set_ylabel(r"Coupling $J$")
    cbar = fig.colorbar(im, ax=ax0, shrink=0.92, pad=0.012, aspect=18)
    cbar.ax.set_title(r"$S_V$", fontsize=16, pad=3)
    cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    panel2_gaps = [g for g in PANEL2_FIXED_TAUS if g in taus]
    panel2_cmap = plt.get_cmap("Blues")
    panel2_norm = Normalize(vmin=0.0, vmax=float(max(taus) if taus else 1))
    for g in panel2_gaps:
        sub = sorted((r for r in rows if int(float(r["tau"])) == int(g)), key=lambda r: float(r["J"]))
        ax1.semilogy(
            [float(r["J"]) for r in sub],
            [max(float(r["entropy"]), HEATMAP_COLOR_VMIN) for r in sub],
            color=panel2_cmap(panel2_norm(float(g))),
            lw=1.7,
            marker="o",
            ms=3.8,
            label=rf"$\tau={g}$",
        )
    ax1.set_xlabel(r"Coupling $J$")
    ax1.set_ylabel(r"$S_V$")
    ax1.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax1.yaxis.set_minor_locator(NullLocator())
    ax1.legend(frameon=False, fontsize=7.0, loc="upper right")

    panel3_js = [float(j_arr[int(np.argmin(np.abs(j_arr - t)))]) for t in PANEL3_TARGET_JS]
    panel3_cmap = plt.get_cmap("Reds")
    panel3_norm = Normalize(vmin=0.0, vmax=2.0)
    for jv in panel3_js:
        ji = j_vals.index(jv)
        ys = np.asarray([max(float(z[gi, ji]), HEATMAP_COLOR_VMIN) for gi in range(len(taus))], dtype=np.float64)
        ax2.semilogy(taus, ys, lw=1.8, marker="o", ms=4.6, color=panel3_cmap(panel3_norm(jv)), label=rf"$J={jv:g}$")
    ax2.set_xlabel(r"Delay $\tau$")
    ax2.set_ylabel(r"$S_V$")
    ax2.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax2.legend(frameon=False, fontsize=7.0, loc="upper right")

    y_side = [max(float(r["entropy"]), HEATMAP_COLOR_VMIN) for r in rows if int(float(r["tau"])) in panel2_gaps]
    y_hi = min(1.0, max(HEATMAP_COLOR_VMIN * 1.2, (float(np.nanmax(y_side)) * 1.25 if y_side else 1.0)))
    ax1.set_ylim(HEATMAP_COLOR_VMIN, y_hi)
    ax2.set_ylim(HEATMAP_COLOR_VMIN, y_hi)

    for ax, tag in ((ax0, "(a)"), (ax1, "(b)"), (ax2, "(c)")):
        ax.text(0.04, 0.955, tag, transform=ax.transAxes, va="top", ha="left", fontsize=13, fontweight="bold")

    save_figure_prl(fig, out_stem)


def plot_entropy_heatmap_tau_j_pair(
    rows: list[dict[str, str | float | int]],
    out_stem: Path,
    *,
    sv_threshold: float = 1e-2,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator

    from _benchmark_common import SV_THRESHOLD_DEFAULT

    if not rows:
        return
    configure_matplotlib_prl()
    thr = float(sv_threshold if sv_threshold is not None else SV_THRESHOLD_DEFAULT)
    taus = sorted({int(float(r["tau"])) for r in rows})
    j_vals = sorted({float(r["J"]) for r in rows})
    j_arr = np.asarray(j_vals, dtype=np.float64)
    g_arr = np.asarray(taus, dtype=np.float64)
    z = np.full((len(taus), len(j_vals)), np.nan, dtype=np.float64)
    for r in rows:
        gi = taus.index(int(float(r["tau"])))
        ji = j_vals.index(float(r["J"]))
        z[gi, ji] = float(r["entropy"])

    g_edges = (
        np.concatenate([[g_arr[0] - 0.5], 0.5 * (g_arr[:-1] + g_arr[1:]), [g_arr[-1] + 0.5]])
        if len(g_arr) > 1
        else np.array([-0.5, 0.5])
    )
    if len(j_arr) > 1:
        dj = float(np.median(np.diff(j_arr)))
        j_edges = np.concatenate([[j_arr[0] - dj / 2], 0.5 * (j_arr[:-1] + j_arr[1:]), [j_arr[-1] + dj / 2]])
    else:
        j_edges = np.array([j_arr[0] - 0.05, j_arr[0] + 0.05])

    fig = plt.figure(figsize=(8.3, 4.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.06, hspace=0.10)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])

    cmap = plt.get_cmap("magma").copy()
    cmap.set_under(color="black")
    cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
    z_plot = np.where(
        np.isfinite(z),
        np.where(z <= 0.0, HEATMAP_COLOR_VMIN * 0.1, np.maximum(z, HEATMAP_COLOR_VMIN)),
        np.nan,
    )
    z_mesh = np.ma.masked_invalid(np.transpose(z_plot))
    im = ax0.pcolormesh(
        g_edges,
        j_edges,
        z_mesh,
        cmap=cmap,
        norm=LogNorm(vmin=HEATMAP_COLOR_VMIN, vmax=HEATMAP_COLOR_VMAX),
        shading="auto",
        linewidth=0,
        rasterized=True,
    )
    ax0.set_xlabel(r"Delay $\tau$")
    ax0.set_ylabel(r"Coupling $J$")
    cbar = fig.colorbar(im, ax=ax0, shrink=0.90, pad=0.015, aspect=20)
    cbar.ax.set_title(r"$S_V$", fontsize=14, pad=2)
    cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    panel3_js = [float(j_arr[int(np.argmin(np.abs(j_arr - t)))]) for t in PANEL3_TARGET_JS]
    panel3_cmap = plt.get_cmap("Reds")
    panel3_norm = Normalize(vmin=0.0, vmax=2.0)
    for jv in panel3_js:
        ji = j_vals.index(jv)
        ys = np.asarray([max(float(z[gi, ji]), HEATMAP_COLOR_VMIN) for gi in range(len(taus))], dtype=np.float64)
        ax1.semilogy(taus, ys, lw=1.8, marker="o", ms=4.2, color=panel3_cmap(panel3_norm(jv)), label=rf"$J={jv:g}$")
    ax1.set_xlabel(r"Delay $\tau$")
    ax1.set_ylabel(r"$S_V$")
    ax1.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ref_thr = max(thr, 1e-30)
    ax1.axhline(ref_thr, color="0.35", ls="--", lw=0.9, label=rf"$S_{{\rm thr}}={ref_thr:.0e}$")
    ax1.legend(frameon=False, fontsize=6.8, loc="upper right")

    thr_main = max(thr, 1e-30)
    thr_lo, thr_hi = 0.5 * thr_main, 2.0 * thr_main

    def _tau_curve(threshold: float) -> np.ndarray:
        return np.asarray(
            [last_tau_above_for_panel_b(z, taus, ji, threshold=threshold)[0] for ji in range(len(j_vals))],
            dtype=np.float64,
        )

    y_lo, y_main, y_hi = _tau_curve(thr_lo), _tau_curve(thr_main), _tau_curve(thr_hi)
    mb = np.isfinite(y_lo) & np.isfinite(y_hi)
    mm = np.isfinite(y_main)
    if np.any(mb):
        ax2.fill_between(j_arr[mb], y_lo[mb], y_hi[mb], color="0.70", alpha=0.30, linewidth=0.0)
    if np.any(mm):
        ax2.plot(j_arr[mm], y_main[mm], color="0.15", lw=1.8, marker="o", ms=3.6, label=rf"main: $S_{{\rm thr}}={thr_main:.0e}$")
    ax2.set_xlabel(r"Coupling $J$")
    ax2.set_ylabel(r"Operational memory horizon $\tau_{\rm mem}$")
    tau_max = float(max(taus) if taus else 0.0)
    ax2.set_ylim(-0.5, max(1.0, tau_max + 0.5))
    ax2.axhline(tau_max, color="0.45", ls="--", lw=0.8)
    ax2.legend(frameon=False, fontsize=6.3, loc="upper left")

    y_side = [max(float(r["entropy"]), HEATMAP_COLOR_VMIN) for r in rows]
    y_hi_plot = min(1.0, max(HEATMAP_COLOR_VMIN * 1.2, (float(np.nanmax(y_side)) * 1.25 if y_side else 1.0)))
    ax1.set_ylim(HEATMAP_COLOR_VMIN, y_hi_plot)

    for ax, tag in ((ax0, "(a)"), (ax1, "(b)"), (ax2, "(c)")):
        ax.text(0.04, 0.955, tag, transform=ax.transAxes, va="top", ha="left", fontsize=12, fontweight="bold")

    save_figure_prl(fig, out_stem)


def plot_entropy_vs_ell(rows: list[dict[str, str | float | int]], out_stem: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator

    if not rows:
        return
    configure_matplotlib_prl()
    ell_key = "ell" if "ell" in rows[0] else "tau"
    ells = sorted({int(float(r[ell_key])) for r in rows})
    j_arr = np.asarray(sorted({float(r["J"]) for r in rows}), dtype=np.float64)
    ell_arr = np.asarray(ells, dtype=np.float64)

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.2), constrained_layout=True)
    target_js = [float(j_arr[int(np.argmin(np.abs(j_arr - t)))]) for t in PANEL_TARGET_JS_ELL]
    target_js = list(dict.fromkeys(target_js))
    cmap = plt.get_cmap("Reds")
    norm = Normalize(vmin=0.0, vmax=2.0)

    for jv in target_js:
        sub = sorted((r for r in rows if float(r["J"]) == float(jv)), key=lambda r: int(float(r[ell_key])))
        if not sub:
            continue
        ax.semilogy(
            [int(float(r[ell_key])) for r in sub],
            [max(float(r["entropy"]), HEATMAP_COLOR_VMIN) for r in sub],
            lw=1.9,
            marker="o",
            ms=3.8,
            color=cmap(norm(jv)),
            label=rf"$J={jv:g}$",
        )

    ax.set_xlabel(r"Zero resets $\ell$")
    ax.set_ylabel(r"$S_V$")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.grid(True, which="major", axis="y", alpha=0.10, linewidth=0.35)
    if len(ell_arr) > 1:
        ax.set_xlim(ell_arr[0] - 0.4, ell_arr[-1] + 0.4)
    y_all = [max(float(r["entropy"]), HEATMAP_COLOR_VMIN) for r in rows]
    ax.set_ylim(HEATMAP_COLOR_VMIN, min(1.0, max(HEATMAP_COLOR_VMIN * 1.2, float(np.nanmax(y_all)) * 1.25)))
    ax.legend(frameon=False, fontsize=7.0, loc="upper right")
    ax.text(0.04, 0.955, "(a)", transform=ax.transAxes, va="top", ha="left", fontsize=12, fontweight="bold")

    save_figure_prl(fig, out_stem)
