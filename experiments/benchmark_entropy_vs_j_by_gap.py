#!/usr/bin/env python3
"""Exact benchmark: fixed-window delayed break sweep ``S_V(ell, J)``.

Fix the number of timesteps before/after the break and sweep the number of in-between delayed
slots ``ell``. Construction per sequence:
``past(15) + [measure, prepare |0>] + [identity]^ell + [prepare_only] + future(5)``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from benchmark_entropy_vs_j_by_cut import (
    BRANCH_WEIGHT_BETA,
    DT_FIXED,
    G_FIXED,
    HEATMAP_COLOR_VMAX,
    HEATMAP_COLOR_VMIN,
    J_SWEEP_DEFAULT,
    K_FIXED,
    L_FIXED,
    _configure_matplotlib_prl_figure,
    _linear_weighted_metrics,
    _list_initial_states_sys_env0,
    _load_summary_csv,
)
from mqt.yaqs.characterization.process_tensors.diagnostics.probe import sample_split_delayed_break_probes
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

K_FIXED = 50
CENTER_CUT = K_FIXED // 2
TAU_MAX = max(0, K_FIXED - CENTER_CUT - 1)
TAU_DEFAULT = tuple(range(0, TAU_MAX + 1))
PANEL2_FIXED_TAUS = (0, 1, 2, 4, 8)
PANEL3_TARGET_JS = (0.4, 1.0, 2.0)
PAST_LEN_FIXED = 15
FUTURE_LEN_FIXED = 5
ELL_MAX_FIXED = 24
ELL_DEFAULT = tuple(range(0, ELL_MAX_FIXED + 1))

# For ``fig_entropy_heatmap_tau_j_pair`` panel (b): the last :math:`\tau` in increasing :math:`\tau`
# order for which :math:`S_V(\tau, J) >` this value; if none in the grid, the panel uses ``0``.
SV_THRESHOLD_DEFAULT = 1e-2


def _last_tau_above_for_panel_b(
    z: np.ndarray, taus: list[int], ji: int, *, threshold: float
) -> tuple[float, str]:
    """(y, kind) for the tau–J pair panel: ``hit`` = last (in increasing ``tau`` order) with
    ``S_V >= threshold``; ``censored`` = at least one finite value but no ``tau`` with ``S_V >= thr``,
    ``y = 0``; ``nodata`` = all entropies missing for this ``J``.
    """
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-pasts", type=int, default=32)
    p.add_argument("--n-futures", type=int, default=32)
    p.add_argument("--ells", type=str, default=",".join(str(g) for g in ELL_DEFAULT))
    p.add_argument("--taus", type=str, default="", help="Deprecated alias for --ells.")
    p.add_argument("--gaps", type=str, default="", help="Deprecated alias for --ells.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--out-dir", type=Path, default=Path("benchmark_entropy_vs_j_by_gap_results"))
    p.add_argument("--parallel", action="store_true", default=True)
    p.add_argument("--no-parallel", dest="parallel", action="store_false")
    p.add_argument("--unitary-ensemble", type=str, default="haar", choices=("haar", "clifford"))
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--summary-csv", type=Path, default=None)
    p.add_argument(
        "--sv-threshold",
        type=float,
        default=SV_THRESHOLD_DEFAULT,
        help=(
            "For tau–J pair panel (b): last tau with S_V above this; else 0. "
            f"Default: {SV_THRESHOLD_DEFAULT}."
        ),
    )
    return p.parse_args()


def _parse_int_list(spec: str) -> list[int]:
    vals = sorted({int(tok.strip()) for tok in spec.split(",") if tok.strip()})
    if not vals:
        raise ValueError("expected at least one ell")
    for g in vals:
        if g < 0 or g > ELL_MAX_FIXED:
            raise ValueError(f"ell must satisfy 0 <= ell <= {ELL_MAX_FIXED}, got {g}")
    return vals


def _write_summary_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def plot_entropy_heatmap_gap_vs_j(rows: list[dict[str, str | float | int]], out_stem: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator

    if not rows:
        return
    _configure_matplotlib_prl_figure()
    taus = sorted({int(float(r["tau"])) for r in rows})
    j_vals = sorted({float(r["J"]) for r in rows})
    z = np.full((len(taus), len(j_vals)), np.nan, dtype=np.float64)
    for r in rows:
        gi = taus.index(int(float(r["tau"])))
        ji = j_vals.index(float(r["J"]))
        z[gi, ji] = float(r["entropy"])

    g_arr = np.asarray(taus, dtype=np.float64)
    j_arr = np.asarray(j_vals, dtype=np.float64)
    g_edges = np.concatenate([[g_arr[0] - 0.5], 0.5 * (g_arr[:-1] + g_arr[1:]), [g_arr[-1] + 0.5]]) if len(g_arr) > 1 else np.array([-0.5, 0.5])
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
    im = ax0.pcolormesh(g_edges, j_edges, z_mesh, cmap=cmap, norm=LogNorm(vmin=HEATMAP_COLOR_VMIN, vmax=HEATMAP_COLOR_VMAX), shading="auto", linewidth=0, edgecolors="none", antialiased=False, rasterized=True)
    ax0.set_xlabel(r"Delay $\tau$")
    ax0.set_ylabel(r"Coupling $J$")
    ax0.grid(False)
    cbar = fig.colorbar(im, ax=ax0, shrink=0.92, pad=0.012, aspect=18)
    cbar.ax.set_title(r"$S_V$", fontsize=16, pad=3)
    cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    cbar.ax.tick_params(length=3.0, width=0.7, labelsize=13)

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
            markeredgewidth=0.0,
            alpha=0.95,
            label=rf"$\tau={g}$",
        )
    ax1.set_xlabel(r"Coupling $J$")
    ax1.set_ylabel(r"$S_V$")
    ax1.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax1.yaxis.set_minor_locator(NullLocator())
    ax1.grid(True, which="major", axis="y", alpha=0.10, linewidth=0.35)
    ax1.legend(frameon=False, fontsize=7.0, handlelength=1.6, borderaxespad=0.2)

    panel3_js = [float(j_arr[int(np.argmin(np.abs(j_arr - t)))]) for t in PANEL3_TARGET_JS]
    panel3_cmap = plt.get_cmap("Reds")
    panel3_norm = Normalize(vmin=0.0, vmax=2.0)
    for jv in panel3_js:
        ji = j_vals.index(jv)
        ys = np.asarray([max(float(z[gi, ji]), HEATMAP_COLOR_VMIN) for gi in range(len(taus))], dtype=np.float64)
        ax2.semilogy(taus, ys, lw=1.8, marker="o", ms=4.6, markeredgewidth=0.0, color=panel3_cmap(panel3_norm(jv)), alpha=0.92, label=rf"$J={jv:g}$")
    ax2.set_xlabel(r"Delay $\tau$")
    ax2.set_ylabel(r"$S_V$")
    ax2.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax2.yaxis.set_minor_locator(NullLocator())
    ax2.grid(True, which="major", axis="y", alpha=0.10, linewidth=0.35)
    ax2.legend(frameon=False, fontsize=7.0, handlelength=1.4, borderaxespad=0.2)

    y_side = [max(float(r["entropy"]), HEATMAP_COLOR_VMIN) for r in rows if int(float(r["tau"])) in panel2_gaps]
    y_hi = min(1.0, max(HEATMAP_COLOR_VMIN * 1.2, (float(np.nanmax(y_side)) * 1.25 if y_side else 1.0)))
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


def plot_entropy_vs_ell(rows: list[dict[str, str | float | int]], out_stem: Path) -> None:
    """Plot representative ``S_V`` vs ``ell`` curves at fixed couplings ``J``."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator

    if not rows:
        return
    _configure_matplotlib_prl_figure()
    ell_key = "ell" if ("ell" in rows[0]) else "tau"
    ells = sorted({int(float(r[ell_key])) for r in rows})
    j_vals = sorted({float(r["J"]) for r in rows})
    ell_arr = np.asarray(ells, dtype=np.float64)
    j_arr = np.asarray(j_vals, dtype=np.float64)

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.2), constrained_layout=True)
    ax.set_facecolor("white")
    target_js = [float(j_arr[int(np.argmin(np.abs(j_arr - t)))]) for t in PANEL3_TARGET_JS]
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
            markeredgewidth=0.0,
            color=cmap(norm(jv)),
            alpha=0.94,
            label=rf"$J={jv:g}$",
        )

    ax.set_xlabel(r"Delay $\ell$")
    ax.set_ylabel(r"$S_V$")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.grid(True, which="major", axis="y", alpha=0.10, linewidth=0.35)
    if len(ell_arr) > 1:
        ax.set_xlim(ell_arr[0] - 0.4, ell_arr[-1] + 0.4)
    y_all = [max(float(r["entropy"]), HEATMAP_COLOR_VMIN) for r in rows]
    y_hi = min(1.0, max(HEATMAP_COLOR_VMIN * 1.2, (float(np.nanmax(y_all)) * 1.25 if y_all else 1.0)))
    ax.set_ylim(HEATMAP_COLOR_VMIN, y_hi)
    ax.legend(frameon=False, fontsize=7.0, handlelength=1.4, borderaxespad=0.2, loc="upper right")
    ax.text(0.04, 0.955, "(a)", transform=ax.transAxes, va="top", ha="left", fontsize=12, fontweight="bold")

    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    prl_stem = out_stem.with_name(f"{out_stem.name}_prl")
    fig.savefig(prl_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    fig.savefig(prl_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)


def plot_entropy_heatmap_tau_j_pair(
    rows: list[dict[str, str | float | int]],
    out_stem: Path,
    *,
    sv_threshold: float = SV_THRESHOLD_DEFAULT,
) -> None:
    """Three-panel figure:
    (a) :math:`S_V` heatmap in :math:`(\\tau, J)`;
    (b) :math:`S_V(\\tau)` at fixed :math:`J`;
    (c) operational memory horizon :math:`\\tau_{\\rm mem}(J)` with robustness band.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator

    if not rows:
        return
    _configure_matplotlib_prl_figure()
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

    z_plot = np.where(
        np.isfinite(z),
        np.where(z <= 0.0, HEATMAP_COLOR_VMIN * 0.1, np.maximum(z, HEATMAP_COLOR_VMIN)),
        np.nan,
    )
    z_mesh = np.ma.masked_invalid(np.transpose(z_plot))

    fig = plt.figure(figsize=(8.3, 4.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.06, hspace=0.10)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])
    ax0.set_facecolor("white")
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")

    cmap = plt.get_cmap("magma").copy()
    cmap.set_under(color="black")
    cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
    sv_norm = LogNorm(vmin=HEATMAP_COLOR_VMIN, vmax=HEATMAP_COLOR_VMAX)
    im = ax0.pcolormesh(
        g_edges,
        j_edges,
        z_mesh,
        cmap=cmap,
        norm=sv_norm,
        shading="auto",
        linewidth=0,
        edgecolors="none",
        antialiased=False,
        rasterized=True,
    )
    ax0.set_xlabel(r"Delay $\tau$")
    ax0.set_ylabel(r"Coupling $J$")
    ax0.grid(False)
    cbar = fig.colorbar(im, ax=ax0, shrink=0.90, pad=0.015, aspect=20)
    cbar.ax.set_title(r"$S_V$", fontsize=14, pad=2)
    cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    cbar.ax.tick_params(length=2.5, width=0.6, labelsize=11)

    # Panel (b): S_V vs tau for fixed representative J values.
    panel3_js = [float(j_arr[int(np.argmin(np.abs(j_arr - t)))]) for t in PANEL3_TARGET_JS]
    panel3_cmap = plt.get_cmap("Reds")
    panel3_norm = Normalize(vmin=0.0, vmax=2.0)
    for jv in panel3_js:
        ji = j_vals.index(jv)
        ys = np.asarray([max(float(z[gi, ji]), HEATMAP_COLOR_VMIN) for gi in range(len(taus))], dtype=np.float64)
        ax1.semilogy(
            taus,
            ys,
            lw=1.8,
            marker="o",
            ms=4.2,
            markeredgewidth=0.0,
            color=panel3_cmap(panel3_norm(jv)),
            alpha=0.92,
            label=rf"$J={jv:g}$",
        )
    ax1.set_xlabel(r"Delay $\tau$")
    ax1.set_ylabel(r"$S_V$")
    ax1.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax1.yaxis.set_minor_locator(NullLocator())
    ax1.grid(True, which="major", axis="y", alpha=0.10, linewidth=0.35)
    ref_thr = max(float(sv_threshold), 1e-30)
    ax1.axhline(ref_thr, color="0.35", ls="--", lw=0.9, alpha=0.9, label=rf"$S_{{\rm thr}}={ref_thr:.0e}$")
    ax1.legend(frameon=False, fontsize=6.8, handlelength=1.25, borderaxespad=0.2, loc="upper right")

    # Panel (c): main operational horizon curve + robustness band.
    thr_main = max(float(sv_threshold), 1e-30)
    thr_lo = 0.5 * thr_main
    thr_hi = 2.0 * thr_main

    def _tau_curve_for_threshold(thr: float) -> np.ndarray:
        vals: list[float] = []
        for ji in range(len(j_vals)):
            yv, _ = _last_tau_above_for_panel_b(z, taus, ji, threshold=float(thr))
            vals.append(yv)
        return np.asarray(vals, dtype=np.float64)

    y_lo = _tau_curve_for_threshold(thr_lo)
    y_main = _tau_curve_for_threshold(thr_main)
    y_hi = _tau_curve_for_threshold(thr_hi)
    y_band_min = np.minimum(y_lo, y_hi)
    y_band_max = np.maximum(y_lo, y_hi)
    mb = np.isfinite(y_band_min) & np.isfinite(y_band_max)
    mm = np.isfinite(y_main)
    if np.any(mb):
        ax2.fill_between(
            j_arr[mb],
            y_band_min[mb],
            y_band_max[mb],
            color="0.70",
            alpha=0.30,
            linewidth=0.0,
            label=rf"band: {thr_lo:.0e}–{thr_hi:.0e}",
        )
    if np.any(mm):
        ax2.plot(
            j_arr[mm],
            y_main[mm],
            color="0.15",
            lw=1.8,
            marker="o",
            ms=3.6,
            markeredgewidth=0.0,
            label=rf"main: $S_{{\rm thr}}={thr_main:.0e}$",
        )
    ax2.set_xlabel(r"Coupling $J$")
    ax2.grid(True, which="major", axis="y", alpha=0.10, linewidth=0.35)
    ax2.grid(True, which="major", axis="x", alpha=0.10, linewidth=0.35)
    tau_max = float(max(taus) if taus else 0.0)
    ax2.set_ylim(-0.5, max(1.0, tau_max + 0.5))
    if len(j_arr) > 1:
        ax2.set_xlim(j_arr[0] - 0.02 * (j_arr[-1] - j_arr[0]), j_arr[-1] + 0.02 * (j_arr[-1] - j_arr[0]))
    ax2.axhline(tau_max, color="0.45", ls="--", lw=0.8, alpha=0.95)
    ax2.text(
        0.98,
        0.98,
        rf"$\tau_{{\max}}={int(tau_max)}$",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=6.6,
        color="0.30",
    )
    ax2.set_ylabel(
        r"Operational memory horizon $\tau_{\rm mem}$",
        fontsize=10,
    )
    ax2.legend(frameon=False, fontsize=6.3, handlelength=1.3, borderaxespad=0.2, loc="upper left")

    y_side = [max(float(r["entropy"]), HEATMAP_COLOR_VMIN) for r in rows if int(float(r["tau"])) in taus]
    y_hi = min(1.0, max(HEATMAP_COLOR_VMIN * 1.2, (float(np.nanmax(y_side)) * 1.25 if y_side else 1.0)))
    ax1.set_ylim(HEATMAP_COLOR_VMIN, y_hi)

    for ax, tag in ((ax0, "(a)"), (ax1, "(b)"), (ax2, "(c)")):
        ax.text(0.04, 0.955, tag, transform=ax.transAxes, va="top", ha="left", fontsize=12, fontweight="bold")

    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    prl_stem = out_stem.with_name(f"{out_stem.name}_prl")
    fig.savefig(prl_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    fig.savefig(prl_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.plot_only):
        csv_path = Path(args.summary_csv) if args.summary_csv is not None else out_dir / "summary.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"summary CSV not found: {csv_path}")
        rows_raw = _load_summary_csv(csv_path)
        plot_entropy_vs_ell(rows_raw, out_dir / "fig_entropy_vs_ell")
        print(
            f"Wrote figure: {out_dir / 'fig_entropy_vs_ell.pdf'}",
            flush=True,
        )
        return

    ell_spec = str(args.ells).strip()
    if not ell_spec:
        ell_spec = str(args.taus).strip() or str(args.gaps).strip()
    if (str(args.taus).strip() or str(args.gaps).strip()) and not str(args.ells).strip():
        print("Using deprecated --taus/--gaps alias; please use --ells.", flush=True)
    ells = _parse_int_list(ell_spec)
    n_seeds = int(args.n_seeds)
    init_rng = np.random.default_rng(int(args.seed) + 77_777)
    initial_list = _list_initial_states_sys_env0(n_seeds=n_seeds, rng=init_rng)
    np.save(out_dir / "initial_states.npy", np.stack(initial_list, axis=0))

    rows: list[dict[str, float | int]] = []
    for ell in ells:
        left_cut = int(PAST_LEN_FIXED + 1)
        right_cut = int(left_cut + ell + 1)
        k_this = int(PAST_LEN_FIXED + 1 + ell + 1 + FUTURE_LEN_FIXED)
        print(
            f"ell={ell:2d}, left_cut={left_cut:2d}, right_cut={right_cut:2d}, k={k_this:2d}, "
            f"past={PAST_LEN_FIXED}, future={FUTURE_LEN_FIXED}",
            flush=True,
        )
        probe_rng = np.random.default_rng(int(args.seed) + 10_000 * int(ell))
        probe_set = sample_split_delayed_break_probes(
            left_cut=int(left_cut),
            tau=int(ell),
            k=int(k_this),
            n_pasts=int(args.n_pasts),
            n_futures=int(args.n_futures),
            rng=probe_rng,
            sigma_ref=np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128),
            unitary_ensemble=str(args.unitary_ensemble),
        )
        for jv in J_SWEEP_DEFAULT:
            op = MPO.ising(length=L_FIXED, J=float(jv), g=G_FIXED)
            sim_params = AnalogSimParams(dt=DT_FIXED, solver="MCWF", show_progress=False)
            entropies: list[float] = []
            delta_norms: list[float] = []
            ranks: list[int] = []
            for psi0 in initial_list:
                m = _linear_weighted_metrics(
                    probe_set=probe_set,
                    op=op,
                    sim_params=sim_params,
                    psi0=psi0,
                    parallel=bool(args.parallel),
                )
                entropies.append(float(m["entropy"]))
                delta_norms.append(float(m["delta_norm"]))
                ranks.append(int(m["rank"]))
            rows.append(
                {
                    "L": L_FIXED,
                    "k": int(k_this),
                    "dt": DT_FIXED,
                    "g": G_FIXED,
                    "left_cut": int(left_cut),
                    "tau": int(ell),
                    "ell": int(ell),
                    "right_cut": int(right_cut),
                    "past_len_fixed": int(PAST_LEN_FIXED),
                    "future_len_fixed": int(FUTURE_LEN_FIXED),
                    "J": float(jv),
                    "n_pasts": int(args.n_pasts),
                    "n_futures": int(args.n_futures),
                    "n_seeds": n_seeds,
                    "branch_weight_beta": BRANCH_WEIGHT_BETA,
                    "entropy": float(np.mean(entropies)),
                    "entropy_std": float(np.std(entropies, ddof=1)) if len(entropies) > 1 else 0.0,
                    "delta_norm": float(np.mean(delta_norms)),
                    "rank": int(round(float(np.mean(ranks)))),
                }
            )
            print(f"ell={ell:2d}, J={jv:>4.2f}, S_mean={rows[-1]['entropy']:.6e}", flush=True)

    _write_summary_csv(out_dir / "summary.csv", rows)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    plot_entropy_vs_ell(rows, out_dir / "fig_entropy_vs_ell")
    print(f"Wrote results to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()

