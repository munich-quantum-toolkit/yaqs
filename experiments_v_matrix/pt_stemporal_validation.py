#!/usr/bin/env python3
"""Compare S_temporal (causal-block entropy from direct PT-MPO) with exhaustive S_PT^cb."""

from __future__ import annotations

import argparse
import csv
import textwrap
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

from common import DT_DEFAULT, G_DEFAULT, L_DEFAULT, characterizer, configure_matplotlib_prl, ising_chain, sim_params, write_csv
from mqt.yaqs.characterization.memory.backends.tomography.process_tensors import (
    MPOProcessTensor,
    causal_block_operator_entropy,
)
from mqt.yaqs.characterization.memory.operational_memory.full_basis import (
    build_probe_set_from_catalog,
    enumerate_full_probe_catalog,
)
from pt_cut_reference import PROFILE_JS, _sv_metrics
from pt_response_process_tensor_main import CUTS, CUT_COLORS, CUT_MARKERS, K, _clip_entropy

ATOL = 1e-15
DEFAULT_MAX_BOND_DIM = 64  # fast; use --exact for agreement at L=6, k=3
DEFAULT_MPO_TOL = 1e-10
DEFAULT_CB_RTOL = 1e-10
DEFAULT_COMPRESS_EVERY = 16
DEFAULT_J_STEP = 0.2
DEFAULT_J_MAX = 4.0
MAIN_DATA_NAME = "stemporal_main_data.csv"


def _load_reference(path: Path) -> dict[tuple[int, float], dict[str, float]]:
    """Load exhaustive ``S_PT_cb`` reference keyed by ``(cut, J)``."""
    ref: dict[tuple[int, float], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (int(float(row["cut"])), float(row["J"]))
            ref[key] = {
                "S_PT_cb": float(row["S_PT_cb"]),
                "S_V_full": float(row["S_V_full"]),
                "process_tensor_trace": float(row["process_tensor_trace"]),
            }
    return ref


def _build_direct_mpo(
    jv: float,
    mc: Any,
    params: Any,
    timesteps: list[float],
    *,
    max_bond_dim: int | None,
    tol: float,
    compress_every: int,
) -> MPOProcessTensor:
    ham = ising_chain(length=L_DEFAULT, j=float(jv), g=G_DEFAULT)
    ham.ensure_encoded("mpo")
    return cast(
        MPOProcessTensor,
        mc.build_process_tensor(
            ham,
            params,
            timesteps=timesteps,
            return_type="mpo",
            method="direct",
            max_bond_dim=max_bond_dim,
            tol=tol,
            compress_every=compress_every,
        ),
    )


def _stemporal_entropy(
    upsilon: np.ndarray,
    cut: int,
    *,
    rtol: float,
    weight_tol: float,
) -> float:
    """Causal-block operator entropy with explicit Schmidt ``rtol`` threshold."""
    cb = causal_block_operator_entropy(upsilon, K, cut, rtol=rtol, weight_tol=weight_tol)
    singular_values = np.asarray(cb["singular_values"], dtype=np.float64)
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        return 0.0
    keep = singular_values > rtol * singular_values[0]
    s = singular_values[keep]
    weights = s**2
    total = float(weights.sum())
    if total <= weight_tol:
        return 0.0
    p = weights / total
    p = p[p > weight_tol]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def _relative_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), ATOL)
    return abs(a - b) / denom


def run_validation(
    *,
    j_values: list[float],
    reference_csv: Path,
    max_bond_dim: int | None,
    tol: float,
    compress_every: int,
    cb_rtol: float,
    weight_tol: float,
) -> list[dict[str, float | int | str]]:
    ref = _load_reference(reference_csv)
    mc = characterizer(parallel=False)
    params = sim_params(dt=DT_DEFAULT)
    timesteps = [DT_DEFAULT] * (K + 1)
    rows: list[dict[str, float | int | str]] = []

    for jv in j_values:
        t0 = time.perf_counter()
        print(
            f"J={jv:g}  direct MPO (chi<={max_bond_dim}, tol={tol:.0e}, compress_every={compress_every})",
            flush=True,
        )
        pt_mpo = _build_direct_mpo(
            jv,
            mc,
            params,
            timesteps,
            max_bond_dim=max_bond_dim,
            tol=tol,
            compress_every=compress_every,
        )
        upsilon = pt_mpo.to_matrix()
        trace = float(np.trace(upsilon).real)
        for cut in CUTS:
            base = ref.get((cut, float(jv)))
            if base is None:
                continue
            s_temporal = _clip_entropy(_stemporal_entropy(upsilon, cut, rtol=cb_rtol, weight_tol=weight_tol))
            s_ref = float(base["S_PT_cb"])
            rel = _relative_diff(s_temporal, s_ref)
            rows.append(
                {
                    "cut": int(cut),
                    "J": float(jv),
                    "k": int(K),
                    "L": int(L_DEFAULT),
                    "S_temporal": s_temporal,
                    "S_PT_cb_exhaustive": s_ref,
                    "abs_diff": abs(s_temporal - s_ref),
                    "rel_diff": rel,
                    "process_tensor_trace": trace,
                    "trace_exhaustive": float(base["process_tensor_trace"]),
                    "max_bond_dim": -1 if max_bond_dim is None else int(max_bond_dim),
                    "mpo_tol": float(tol),
                    "cb_rtol": float(cb_rtol),
                }
            )
        print(f"  done in {time.perf_counter() - t0:.1f}s", flush=True)

    return rows


def _j_grid(*, j_max: float, j_step: float) -> list[float]:
    n = int(round(j_max / j_step))
    return [_normalize_j(i * j_step, j_step=j_step) for i in range(n + 1)]


def _normalize_j(jv: float, *, j_step: float = DEFAULT_J_STEP) -> float:
    """Snap ``J`` to the coupling grid (avoids ``0.2*i`` vs ``round(i*0.2, 12)`` mismatches)."""
    return round(round(float(jv) / j_step) * j_step, 12)


def _dedupe_main_rows(rows: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
    by_key: dict[tuple[int, float], dict[str, float | int]] = {}
    for row in rows:
        key = (int(row["cut"]), _normalize_j(float(row["J"])))
        by_key[key] = {**row, "cut": key[0], "J": key[1]}
    out = list(by_key.values())
    out.sort(key=lambda r: (int(r["cut"]), float(r["J"])))
    return out


def _load_main_rows(path: Path) -> list[dict[str, float | int]]:
    if not path.is_file():
        return []
    rows: list[dict[str, float | int]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "cut": int(float(row["cut"])),
                    "J": _normalize_j(float(row["J"])),
                    "k": int(K),
                    "L": int(L_DEFAULT),
                    "S_temporal": _clip_entropy(float(row["S_temporal"])),
                    "S_V_full": _clip_entropy(float(row["S_V_full"])),
                }
            )
    rows.sort(key=lambda r: (int(r["cut"]), float(r["J"])))
    return _dedupe_main_rows(rows)


def _seed_main_rows(
    *,
    main_csv: Path,
    stemporal_csv: Path,
    reference_csv: Path,
) -> list[dict[str, float | int]]:
    """Bootstrap main-plot rows from legacy CSVs when ``stemporal_main_data.csv`` is absent."""
    existing = _load_main_rows(main_csv)
    if existing:
        return existing
    if stemporal_csv.is_file() and reference_csv.is_file():
        rows = _merge_legacy_rows(stemporal_csv, reference_csv)
        write_csv(main_csv, rows)
        return rows
    return []


def _merge_legacy_rows(
    stemporal_csv: Path,
    reference_csv: Path,
) -> list[dict[str, float | int]]:
    sv_by_key: dict[tuple[int, float], float] = {}
    with reference_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (int(float(row["cut"])), _normalize_j(float(row["J"])))
            sv_by_key[key] = float(row["S_V_full"])
    merged: list[dict[str, float | int]] = []
    with stemporal_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cut = int(float(row["cut"]))
            jv = _normalize_j(float(row["J"]))
            if (cut, jv) not in sv_by_key:
                continue
            merged.append(
                {
                    "cut": cut,
                    "J": jv,
                    "k": int(K),
                    "L": int(L_DEFAULT),
                    "S_temporal": _clip_entropy(float(row["S_temporal"])),
                    "S_V_full": _clip_entropy(sv_by_key[(cut, jv)]),
                }
            )
    merged.sort(key=lambda r: (int(r["cut"]), float(r["J"])))
    return merged


def run_main_extend(
    *,
    j_values: list[float],
    existing: list[dict[str, float | int]],
    max_bond_dim: int | None,
    tol: float,
    compress_every: int,
    cb_rtol: float,
    weight_tol: float,
) -> list[dict[str, float | int]]:
    """Compute ``S_temporal`` and ``S_V^full`` from direct PT-MPO for missing ``(cut, J)``."""
    have = {(int(r["cut"]), _normalize_j(float(r["J"]))) for r in existing}
    mc = characterizer(parallel=False)
    params = sim_params(dt=DT_DEFAULT)
    timesteps = [DT_DEFAULT] * (K + 1)
    out = list(existing)

    for jv in j_values:
        jv = _normalize_j(jv)
        needed_cuts = [cut for cut in CUTS if (cut, jv) not in have]
        if not needed_cuts:
            continue
        t0 = time.perf_counter()
        print(
            f"J={jv:g}  main sweep (chi<={max_bond_dim}, tol={tol:.0e}, cuts={needed_cuts})",
            flush=True,
        )
        pt_mpo = _build_direct_mpo(
            jv,
            mc,
            params,
            timesteps,
            max_bond_dim=max_bond_dim,
            tol=tol,
            compress_every=compress_every,
        )
        pt_dense = pt_mpo.to_dense()
        upsilon = pt_mpo.to_matrix()
        for cut in needed_cuts:
            catalog = enumerate_full_probe_catalog(cut=cut, num_interventions=K)
            probe_full = build_probe_set_from_catalog(
                catalog,
                np.arange(len(catalog.past_settings), dtype=np.int64),
                np.arange(len(catalog.future_settings), dtype=np.int64),
            )
            sv_full, _norm = _sv_metrics(probe_full, pt_dense)
            s_temporal = _clip_entropy(_stemporal_entropy(upsilon, cut, rtol=cb_rtol, weight_tol=weight_tol))
            row = {
                "cut": int(cut),
                "J": jv,
                "k": int(K),
                "L": int(L_DEFAULT),
                "S_temporal": s_temporal,
                "S_V_full": _clip_entropy(float(sv_full)),
            }
            out.append(row)
            have.add((cut, jv))
        print(f"  done in {time.perf_counter() - t0:.1f}s", flush=True)

    out.sort(key=lambda r: (int(r["cut"]), float(r["J"])))
    return out


def _saturation_note(rows: list[dict[str, float | int]], *, cut: int = 2) -> str:
    """Heuristic: relative increase of ``S_V`` over the last three ``J`` points."""
    sub = sorted([r for r in rows if int(r["cut"]) == cut], key=lambda r: float(r["J"]))
    if len(sub) < 3:
        return f"c={cut}: insufficient points for saturation check"
    tail = sub[-3:]
    s0, s1, s2 = (float(r["S_V_full"]) for r in tail)
    j0, j1, j2 = (float(r["J"]) for r in tail)
    rel01 = (s1 - s0) / max(s0, 1e-30)
    rel12 = (s2 - s1) / max(s1, 1e-30)
    return (
        f"c={cut}: ΔS/S over [{j0:g},{j1:g}] = {rel01:.2%}, "
        f"over [{j1:g},{j2:g}] = {rel12:.2%}"
    )


def _merge_main_rows(main_csv: Path) -> list[dict[str, float | int]]:
    """Load rows for main-figure plotting."""
    return _load_main_rows(main_csv)


def _plot_main_stemporal_figure(rows: list[dict[str, float | int]], *, out_dir: Path) -> None:
    """Two-panel main figure: ``S_temporal`` vs ``S_V^full`` (same layout as process-tensor main)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogFormatterMathtext, LogLocator

    configure_matplotlib_prl()
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    retained = list(CUTS)
    pos_rows = [r for r in rows if int(r["cut"]) in retained and float(r["J"]) > 1e-12]
    pos_sv = [float(r["S_V_full"]) for r in pos_rows if float(r["S_V_full"]) > 0.0]
    pos_st = [float(r["S_temporal"]) for r in pos_rows if float(r["S_temporal"]) > 0.0]
    all_pos = pos_sv + pos_st
    y_lo = min(all_pos) / np.sqrt(10.0) if all_pos else 1e-9
    y_lo = max(y_lo, 1e-10)
    y_hi = max(all_pos) * np.sqrt(10.0) if all_pos else 4e-2
    y_hi = max(y_hi, 4e-2)

    j_all = sorted({float(r["J"]) for r in pos_rows})
    j_min, j_max = min(j_all), max(j_all)

    fig = plt.figure(figsize=(7.1, 2.95))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.0], wspace=0.24, left=0.09, right=0.88, bottom=0.17, top=0.95)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    def _plot_style_series(
        ax,
        x: np.ndarray,
        y: np.ndarray,
        *,
        color: str,
        ls: str,
        marker: str,
        lw: float,
        ms: float,
        line_z: int,
        marker_z: int,
    ) -> None:
        ax.plot(x, y, ls=ls, color=color, lw=lw, zorder=line_z, solid_capstyle="round")
        ax.plot(
            x,
            y,
            linestyle="none",
            marker=marker,
            color=color,
            ms=ms,
            mfc=color,
            mec="0.15",
            mew=0.75,
            zorder=marker_z,
        )

    for cut in retained:
        sub = sorted(
            [r for r in rows if int(r["cut"]) == cut and float(r["J"]) > 1e-12],
            key=lambda r: float(r["J"]),
        )
        j = np.asarray([float(r["J"]) for r in sub], dtype=np.float64)
        st = np.asarray([float(r["S_temporal"]) for r in sub], dtype=np.float64)
        svf = np.asarray([float(r["S_V_full"]) for r in sub], dtype=np.float64)
        color = CUT_COLORS[cut]
        _plot_style_series(
            ax_a,
            j,
            st,
            color=color,
            ls="--",
            marker="D",
            lw=1.6,
            ms=4.6,
            line_z=2,
            marker_z=3,
        )
        _plot_style_series(
            ax_a,
            j,
            svf,
            color=color,
            ls="-",
            marker="o",
            lw=1.8,
            ms=4.8,
            line_z=3,
            marker_z=4,
        )

    ax_a.set_yscale("log")
    ax_a.set_xlabel(r"Coupling $J$")
    ax_a.set_ylabel("Information measure (nats)")
    ax_a.set_xlim(0.12, j_max + 0.08)
    ax_a.set_ylim(y_lo, y_hi)
    ax_a.yaxis.set_major_formatter(LogFormatterMathtext())
    ax_a.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax_a.tick_params(direction="in", top=True, right=True, which="both", length=3.5, width=0.75)
    for spine in ax_a.spines.values():
        spine.set_linewidth(0.85)
    ax_a.text(0.03, 0.97, "(a)", transform=ax_a.transAxes, fontsize=12.5, fontweight="bold", va="top")

    style_ref = "0.40"
    qty_handles = [
        Line2D(
            [0],
            [0],
            color=style_ref,
            ls="-",
            marker="o",
            lw=1.8,
            ms=4.8,
            mfc=style_ref,
            mec="0.15",
            mew=0.75,
            label=r"$S_V^{\mathrm{full}}$",
        ),
        Line2D(
            [0],
            [0],
            color=style_ref,
            ls="--",
            marker="D",
            lw=1.6,
            ms=4.6,
            mfc=style_ref,
            mec="0.15",
            mew=0.75,
            label=r"$S_{\mathrm{temporal}}$",
        ),
    ]
    cut_handles = [Line2D([0], [0], color=CUT_COLORS[c], lw=1.8, label=rf"$c={c}$") for c in retained]
    leg_qty = ax_a.legend(
        handles=qty_handles,
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(0.72, 0.02),
        fontsize=8.0,
        handlelength=2.4,
    )
    ax_a.add_artist(leg_qty)
    ax_a.legend(
        handles=cut_handles,
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        fontsize=8.0,
        title="Cut",
        title_fontsize=8.0,
        handlelength=1.8,
    )

    norm = Normalize(vmin=j_min, vmax=j_max)
    cmap = plt.colormaps["plasma"]
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    scatters = []
    for cut in retained:
        sub = sorted(
            [r for r in rows if int(r["cut"]) == cut and float(r["J"]) > 1e-12],
            key=lambda r: float(r["J"]),
        )
        x = np.asarray([float(r["S_temporal"]) for r in sub], dtype=np.float64)
        y = np.asarray([float(r["S_V_full"]) for r in sub], dtype=np.float64)
        j = np.asarray([float(r["J"]) for r in sub], dtype=np.float64)
        ax_b.plot(x, y, "-", color="0.78", lw=0.9, alpha=0.62, zorder=1)
        sc = ax_b.scatter(
            x,
            y,
            c=j,
            cmap=cmap,
            norm=norm,
            marker=CUT_MARKERS[cut],
            s=30,
            edgecolors="0.15",
            linewidths=0.8,
            zorder=3,
        )
        scatters.append(sc)

    ax_b.set_xlabel(r"$S_{\mathrm{temporal}}$")
    ax_b.set_ylabel(r"$S_V^{\mathrm{full}}$")
    ax_b.xaxis.set_major_formatter(LogFormatterMathtext())
    ax_b.yaxis.set_major_formatter(LogFormatterMathtext())
    ax_b.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax_b.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax_b.tick_params(direction="in", top=True, right=True, which="both", length=3.5, width=0.75)
    for spine in ax_b.spines.values():
        spine.set_linewidth(0.85)
    ax_b.text(0.03, 0.97, "(b)", transform=ax_b.transAxes, fontsize=12.5, fontweight="bold", va="top")

    shape_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=CUT_MARKERS[c],
            color="0.25",
            ms=5.2,
            mew=0.75,
            mec="0.15",
            label=rf"$c={c}$",
        )
        for c in retained
    ]
    ax_b.legend(
        handles=shape_handles,
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        fontsize=8.0,
        title="Cut",
        title_fontsize=8.0,
        handlelength=1.2,
    )

    fig.canvas.draw()
    pos_b = ax_b.get_position()
    cax = fig.add_axes([pos_b.x1 + 0.010, pos_b.y0, 0.012, pos_b.height])
    cbar = fig.colorbar(scatters[0], cax=cax)
    cbar.set_label(r"Coupling $J$", fontsize=9.5, labelpad=2)
    cbar.set_ticks(
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        if j_max >= 3.5
        else ([0.5, 1.0, 1.5, 2.0] if j_max >= 1.5 else [j_min, (j_min + j_max) / 2, j_max])
    )
    cbar.ax.tick_params(length=3, width=0.6, labelsize=8.5)

    stem = out_dir / "pt_response_stemporal_main"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)


def _plot_comparison(rows: list[dict[str, float | int | str]], *, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogFormatterMathtext, LogLocator

    configure_matplotlib_prl()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"wspace": 0.32})
    colors = {1: "#009E73", 2: "#0072B2", 3: "#D55E00"}
    floor = 1e-12

    ax = axes[0]
    for cut in CUTS:
        sub = sorted([r for r in rows if int(r["cut"]) == cut], key=lambda r: float(r["J"]))
        j = np.asarray([float(r["J"]) for r in sub], dtype=np.float64)
        s_ref = np.maximum(np.asarray([float(r["S_PT_cb_exhaustive"]) for r in sub], dtype=np.float64), floor)
        s_tmp = np.maximum(np.asarray([float(r["S_temporal"]) for r in sub], dtype=np.float64), floor)
        ax.plot(j, s_ref, "o-", color=colors[cut], lw=1.5, ms=4, label=rf"ex. $c={cut}$")
        ax.plot(j, s_tmp, "x--", color=colors[cut], lw=1.2, ms=4, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel(r"Coupling $J$")
    ax.set_ylabel("Entropy (nats)")
    ax.set_title(rf"$S_{{\mathrm{{temporal}}}}$ vs $S_{{\mathrm{{PT}}}}^{{\mathrm{{cb}}}}$", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.text(0.03, 0.97, "(a)", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")

    ax2 = axes[1]
    for cut in CUTS:
        sub = sorted([r for r in rows if int(r["cut"]) == cut], key=lambda r: float(r["J"]))
        j = np.asarray([float(r["J"]) for r in sub], dtype=np.float64)
        rel = np.maximum(np.asarray([float(r["rel_diff"]) for r in sub], dtype=np.float64), floor)
        ax2.plot(j, rel, "o-", color=colors[cut], lw=1.5, ms=4, label=rf"$c={cut}$")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"Coupling $J$")
    ax2.set_ylabel(r"Relative difference")
    ax2.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax2.yaxis.set_major_formatter(LogFormatterMathtext())
    ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax2.text(0.03, 0.97, "(b)", transform=ax2.transAxes, fontsize=12, fontweight="bold", va="top")

    stem = out_dir / "fig_stemporal_vs_exhaustive"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", dpi=600)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=600)
    plt.close(fig)


def _write_report(
    rows: list[dict[str, float | int | str]],
    *,
    out_dir: Path,
    max_bond_dim: int | None,
    tol: float,
    cb_rtol: float,
) -> str:
    pos = [r for r in rows if float(r["J"]) > 1e-12]
    max_rel = max(float(r["rel_diff"]) for r in pos) if pos else 0.0
    med_rel = float(np.median([float(r["rel_diff"]) for r in pos])) if pos else 0.0
    max_abs = max(float(r["abs_diff"]) for r in pos) if pos else 0.0
    lines = [
        "S_temporal (direct PT-MPO) vs exhaustive S_PT^cb",
        "=" * 48,
        f"Parameters: L={L_DEFAULT}, k={K}, g={G_DEFAULT}, dt={DT_DEFAULT}",
        f"Direct build: method=direct, return_type=mpo, max_bond_dim={max_bond_dim!r}, tol={tol:.3e}",
        f"Causal-block SVD rtol: {cb_rtol:.3e}",
        "",
        f"Max relative diff (J>0): {max_rel:.3e}",
        f"Median relative diff (J>0): {med_rel:.3e}",
        f"Max absolute diff (J>0): {max_abs:.3e}",
        "",
        "Per (cut, J):",
    ]
    for r in sorted(rows, key=lambda x: (int(x["cut"]), float(x["J"]))):
        lines.append(
            f"  c={int(r['cut']):d} J={float(r['J']):5.3g}  "
            f"S_temporal={float(r['S_temporal']):12.6e}  "
            f"S_PT_cb={float(r['S_PT_cb_exhaustive']):12.6e}  "
            f"rel={float(r['rel_diff']):.3e}"
        )
    lines.append("")
    lines.append(
        textwrap.dedent(
            """
            S_temporal: causal-block operator entropy on upsilon = direct_MPO.to_matrix(),
            keeping Schmidt values s_i > rtol * s_0 before normalizing weights.
            """
        ).strip()
    )
    report = "\n".join(lines)
    (out_dir / "stemporal_validation_report.txt").write_text(report + "\n", encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("save/pt_cut_reference"))
    p.add_argument(
        "--reference-csv",
        type=Path,
        default=None,
        help="Exhaustive S_PT_cb reference (default: pt_response_process_tensor_main_data.csv).",
    )
    p.add_argument("--j-values", type=str, default=None)
    p.add_argument(
        "--max-bond-dim",
        type=int,
        default=DEFAULT_MAX_BOND_DIM,
        help=f"MPO bond cap for direct build (default {DEFAULT_MAX_BOND_DIM}; use 0 for exact).",
    )
    p.add_argument("--tol", type=float, default=DEFAULT_MPO_TOL, help="MPO compression SVD tolerance.")
    p.add_argument(
        "--cb-rtol",
        type=float,
        default=DEFAULT_CB_RTOL,
        help="Relative Schmidt threshold s_i > rtol*s_0 for S_temporal.",
    )
    p.add_argument("--weight-tol", type=float, default=1e-30, help="Weight floor in S_temporal entropy.")
    p.add_argument("--compress-every", type=int, default=DEFAULT_COMPRESS_EVERY)
    p.add_argument("--smoke", action="store_true", help="Run J=0,1,2 only.")
    p.add_argument("--exact", action="store_true", help="Exact direct build (max_bond_dim=None; ~2 min/J at L=6,k=3).")
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Plot main figure from existing stemporal_main_data.csv (skip recomputation).",
    )
    p.add_argument(
        "--extend-main",
        action="store_true",
        help="Extend main-plot data with direct PT-MPO up to --j-max.",
    )
    p.add_argument("--j-max", type=float, default=DEFAULT_J_MAX, help="Maximum coupling J (default 4.0).")
    p.add_argument("--j-step", type=float, default=DEFAULT_J_STEP, help="J grid step (default 0.2).")
    args = p.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_csv = args.reference_csv or (out_dir / "pt_response_process_tensor_main_data.csv")
    stemporal_csv = out_dir / "stemporal_validation.csv"
    main_csv = out_dir / MAIN_DATA_NAME

    if args.plot_only:
        rows = _load_main_rows(main_csv)
        if not rows:
            rows = _seed_main_rows(main_csv=main_csv, stemporal_csv=stemporal_csv, reference_csv=ref_csv)
        if not rows:
            raise SystemExit(f"Missing {main_csv}; run --extend-main first.")
        _plot_main_stemporal_figure(rows, out_dir=out_dir)
        print(f"Wrote {out_dir / 'pt_response_stemporal_main.png'}", flush=True)
        return

    if args.extend_main:
        if args.exact or args.max_bond_dim == 0:
            max_bond_dim = None
        else:
            max_bond_dim = int(args.max_bond_dim)
        if max_bond_dim is not None and max_bond_dim < 128:
            print(
                f"WARNING: max_bond_dim={max_bond_dim} may disagree with reference at L={L_DEFAULT}, k={K}. "
                "Use --exact for production data.",
                flush=True,
            )
        existing = _seed_main_rows(main_csv=main_csv, stemporal_csv=stemporal_csv, reference_csv=ref_csv)
        j_values = _j_grid(j_max=float(args.j_max), j_step=float(args.j_step))
        rows = run_main_extend(
            j_values=j_values,
            existing=existing,
            max_bond_dim=max_bond_dim,
            tol=float(args.tol),
            compress_every=int(args.compress_every),
            cb_rtol=float(args.cb_rtol),
            weight_tol=float(args.weight_tol),
        )
        write_csv(main_csv, rows)
        _plot_main_stemporal_figure(rows, out_dir=out_dir)
        print(_saturation_note(rows, cut=2), flush=True)
        print(_saturation_note(rows, cut=3), flush=True)
        print(f"Wrote {main_csv}", flush=True)
        print(f"Wrote {out_dir / 'pt_response_stemporal_main.png'}", flush=True)
        return

    if args.j_values:
        j_values = [float(x.strip()) for x in args.j_values.split(",") if x.strip()]
    elif args.smoke:
        j_values = [0.0, 1.0, 2.0]
    else:
        j_values = list(PROFILE_JS["default"])

    if args.exact or args.max_bond_dim == 0:
        max_bond_dim = None
    else:
        max_bond_dim = int(args.max_bond_dim)

    if max_bond_dim is not None and max_bond_dim < 128:
        print(
            f"WARNING: max_bond_dim={max_bond_dim} may disagree with exhaustive S_PT^cb at L={L_DEFAULT}, k={K}. "
            "Use --exact for validation.",
            flush=True,
        )

    rows = run_validation(
        j_values=j_values,
        reference_csv=ref_csv,
        max_bond_dim=max_bond_dim,
        tol=float(args.tol),
        compress_every=int(args.compress_every),
        cb_rtol=float(args.cb_rtol),
        weight_tol=float(args.weight_tol),
    )
    write_csv(out_dir / "stemporal_validation.csv", rows)
    report = _write_report(
        rows,
        out_dir=out_dir,
        max_bond_dim=max_bond_dim,
        tol=float(args.tol),
        cb_rtol=float(args.cb_rtol),
    )
    _plot_comparison(rows, out_dir=out_dir)
    main_rows = _merge_main_rows(main_csv) if main_csv.is_file() else _seed_main_rows(
        main_csv=main_csv, stemporal_csv=stemporal_csv, reference_csv=ref_csv
    )
    if main_rows:
        _plot_main_stemporal_figure(main_rows, out_dir=out_dir)
    print(report, flush=True)
    print(f"\nWrote {out_dir / 'fig_stemporal_vs_exhaustive.png'}", flush=True)
    print(f"Wrote {out_dir / 'pt_response_stemporal_main.png'}", flush=True)


if __name__ == "__main__":
    main()
