#!/usr/bin/env python3
"""Main-text figure: S_PT^cb versus S_V^full across causal cuts c=1,2,3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np

from common import (
    DT_DEFAULT,
    G_DEFAULT,
    L_DEFAULT,
    characterizer,
    configure_matplotlib_prl,
    ising_chain,
    sim_params,
    write_csv,
)
from mqt.yaqs.characterization.memory.backends.tomography.process_tensors import DenseProcessTensor
from mqt.yaqs.characterization.memory.operational_memory.full_basis import (
    build_probe_set_from_catalog,
    enumerate_full_probe_catalog,
)
from pt_cut_reference import (
    PROFILE_JS,
    _causal_break_error,
    _sv_metrics,
)

CUTS = (1, 2, 3)
K = 3
CLIP_NEG = 1e-10
J0_TOL = 1e-10
TRACE_TOL = 1e-10
MINEV_TOL = -1e-10
CB_TOL = 1e-10

CUT_COLORS = {
    1: "#009E73",
    2: "#0072B2",
    3: "#D55E00",
}
CUT_MARKERS = {
    1: "^",
    2: "o",
    3: "s",
}


def _clip_entropy(value: float) -> float:
    if -CLIP_NEG < value < 0.0:
        return 0.0
    return float(value)


def _load_csv_rows(path: Path) -> list[dict[str, float | int]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows: list[dict[str, float | int]] = []
        for row in csv.DictReader(f):
            out: dict[str, float | int] = {}
            for k, v in row.items():
                if k in {"cut", "k", "L"}:
                    out[k] = int(float(v))
                else:
                    out[k] = float(v)
            rows.append(out)
        return rows


def _load_c2_sptcb_by_j(summary_cb: Path) -> dict[float, float]:
    """Load ``S_PT^cb`` for ``c=2`` from a prior causal-block benchmark CSV."""
    by_j: dict[float, float] = {}
    if not summary_cb.exists():
        return by_j
    with summary_cb.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(float(row["cut"])) != 2 or int(float(row["k"])) != K:
                continue
            if int(float(row["L"])) != L_DEFAULT:
                continue
            by_j[float(row["J"])] = float(row["S_PT_cb"])
    return by_j


def _evaluate_cut_j(
    *,
    cut: int,
    jv: float,
    mc: Any,
    params: Any,
    timesteps: list[float],
    pt: DenseProcessTensor | None = None,
) -> dict[str, float | int]:
    """Compute S_PT^cb and S_V^full for one (cut, J) pair."""
    catalog = enumerate_full_probe_catalog(cut=cut, num_interventions=K)
    n_p = len(catalog.past_settings)
    n_f = len(catalog.future_settings)
    probe_full = build_probe_set_from_catalog(
        catalog,
        np.arange(n_p, dtype=np.int64),
        np.arange(n_f, dtype=np.int64),
    )
    if pt is None:
        ham = ising_chain(length=L_DEFAULT, j=float(jv), g=G_DEFAULT)
        ham.ensure_encoded("mpo")
        pt = cast(
            DenseProcessTensor,
            mc.build_process_tensor(
                ham,
                params,
                timesteps=timesteps,
                return_type="dense",
                method="exhaustive",
                compress_every=1,
            ),
        )
    cb = pt.causal_block_operator_entropy(cut)
    sv_full, _norm = _sv_metrics(probe_full, pt)
    mat = pt.to_matrix()
    min_eval = float(np.min(np.linalg.eigvalsh(0.5 * (mat + mat.conj().T)).real))
    return {
        "cut": int(cut),
        "J": float(jv),
        "k": int(K),
        "L": int(L_DEFAULT),
        "S_PT_cb": _clip_entropy(float(cast("float", cb["entropy"]))),
        "S_V_full": _clip_entropy(float(sv_full)),
        "process_tensor_trace": float(np.trace(mat).real),
        "minimum_eigenvalue": min_eval,
        "causal_break_error": float(_causal_break_error(pt, K, cut)),
        "len_P_full": int(n_p),
        "len_F_full": int(n_f),
    }


def _build_pt(jv: float, mc: Any, params: Any, timesteps: list[float]) -> DenseProcessTensor:
    ham = ising_chain(length=L_DEFAULT, j=float(jv), g=G_DEFAULT)
    ham.ensure_encoded("mpo")
    return cast(
        DenseProcessTensor,
        mc.build_process_tensor(
            ham,
            params,
            timesteps=timesteps,
            return_type="dense",
            method="exhaustive",
            compress_every=1,
        ),
    )


def _merge_c2_from_summary_cb(
    rows: list[dict[str, float | int]],
    *,
    summary_cb: Path,
    main_csv: Path | None,
) -> list[dict[str, float | int]]:
    """Fill ``c=2`` rows from cached causal-block benchmark when possible."""
    spt_by_j = _load_c2_sptcb_by_j(summary_cb)
    if not spt_by_j:
        return rows
    main_by_j: dict[float, dict[str, float | int]] = {}
    if main_csv and main_csv.exists():
        for row in _load_csv_rows(main_csv):
            if int(row["cut"]) == 2:
                main_by_j[float(row["J"])] = row
    merged: list[dict[str, float | int]] = []
    for row in rows:
        if int(row["cut"]) != 2 or float(row["J"]) not in spt_by_j:
            merged.append(row)
            continue
        jv = float(row["J"])
        base = main_by_j.get(jv, row)
        merged.append(
            {
                "cut": 2,
                "J": jv,
                "k": int(K),
                "L": int(L_DEFAULT),
                "S_PT_cb": _clip_entropy(float(spt_by_j[jv])),
                "S_V_full": _clip_entropy(float(base["S_V_full"])),
                "process_tensor_trace": float(base.get("process_tensor_trace", row.get("process_tensor_trace", 8.0))),
                "minimum_eigenvalue": float(base.get("minimum_eigenvalue", row.get("minimum_eigenvalue", 0.0))),
                "causal_break_error": float(base.get("causal_break_error", row.get("causal_break_error", 0.0))),
                "len_P_full": int(base.get("len_P_full", 64)),
                "len_F_full": int(base.get("len_F_full", 64)),
            }
        )
    return merged


def run_benchmark(
    j_values: list[float],
    *,
    out_dir: Path,
    summary_cb_csv: Path | None,
    main_csv: Path | None,
) -> list[dict[str, float | int]]:
    """Evaluate all (cut, J) pairs; reuse cached ``c=2`` ``S_PT^cb`` when available."""
    mc = characterizer(parallel=False)
    params = sim_params(dt=DT_DEFAULT)
    timesteps = [DT_DEFAULT] * (K + 1)
    spt_c2 = _load_c2_sptcb_by_j(summary_cb_csv) if summary_cb_csv else {}
    main_c2: dict[float, dict[str, float | int]] = {}
    if main_csv and main_csv.exists():
        for row in _load_csv_rows(main_csv):
            if int(row["cut"]) == 2:
                main_c2[float(row["J"])] = row

    rows: list[dict[str, float | int]] = []
    t0 = time.perf_counter()
    for jv in j_values:
        pt: DenseProcessTensor | None = None
        for cut in CUTS:
            if cut == 2 and float(jv) in spt_c2:
                base = main_c2.get(float(jv), {})
                rows.append(
                    {
                        "cut": 2,
                        "J": float(jv),
                        "k": int(K),
                        "L": int(L_DEFAULT),
                        "S_PT_cb": _clip_entropy(float(spt_c2[float(jv)])),
                        "S_V_full": _clip_entropy(float(base.get("S_V_full", 0.0))),
                        "process_tensor_trace": float(base.get("process_tensor_trace", 8.0)),
                        "minimum_eigenvalue": float(base.get("minimum_eigenvalue", 0.0)),
                        "causal_break_error": float(base.get("causal_break_error", 0.0)),
                        "len_P_full": int(base.get("len_P_full", 64)),
                        "len_F_full": int(base.get("len_F_full", 64)),
                    }
                )
                continue
            if pt is None:
                print(f"  J={jv:g}", flush=True)
                pt = _build_pt(jv, mc, params, timesteps)
            print(f"    cut={cut}", flush=True)
            rows.append(_evaluate_cut_j(cut=cut, jv=jv, mc=mc, params=params, timesteps=timesteps, pt=pt))
    rows.sort(key=lambda r: (int(r["cut"]), float(r["J"])))
    print(f"Benchmark finished in {time.perf_counter() - t0:.1f}s", flush=True)
    return rows


def _augment_rows_with_sptcb(
    rows: list[dict[str, float | int]],
    *,
    summary_cb_csv: Path,
) -> list[dict[str, float | int]]:
    """Ensure every row has ``S_PT^cb``, computing only missing cuts."""
    spt_c2 = _load_c2_sptcb_by_j(summary_cb_csv)
    by_j_cut = {(float(r["J"]), int(r["cut"])): dict(r) for r in rows}
    mc = characterizer(parallel=False)
    params = sim_params(dt=DT_DEFAULT)
    timesteps = [DT_DEFAULT] * (K + 1)
    out: list[dict[str, float | int]] = []
    for jv in sorted({float(r["J"]) for r in rows}):
        pt: DenseProcessTensor | None = None
        for cut in CUTS:
            row = by_j_cut.get((jv, cut))
            if row is None:
                continue
            if cut == 2 and jv in spt_c2:
                row["S_PT_cb"] = _clip_entropy(float(spt_c2[jv]))
                out.append(row)
                continue
            if "S_PT_cb" in row:
                out.append(row)
                continue
            if pt is None:
                print(f"  augment J={jv:g}", flush=True)
                pt = _build_pt(jv, mc, params, timesteps)
            print(f"    augment cut={cut}", flush=True)
            fresh = _evaluate_cut_j(cut=cut, jv=jv, mc=mc, params=params, timesteps=timesteps, pt=pt)
            row.update(
                {
                    k: fresh[k]
                    for k in ("S_PT_cb", "process_tensor_trace", "minimum_eigenvalue", "causal_break_error")
                }
            )
            out.append(row)
    out.sort(key=lambda r: (int(r["cut"]), float(r["J"])))
    return out


def _validate_rows(rows: list[dict[str, float | int]]) -> None:
    """Assert per-cut J=0 and PT sanity checks."""
    for cut in CUTS:
        sub = [r for r in rows if int(r["cut"]) == cut]
        j0 = next(r for r in sub if abs(float(r["J"])) < 1e-12)
        struct_note = ""
        if cut == 1:
            struct_note = " (c=1: no pre-break intervention history in catalog)"
        assert abs(float(j0["S_PT_cb"])) < J0_TOL, (cut, j0["S_PT_cb"])
        assert abs(float(j0["S_V_full"])) < J0_TOL, (cut, j0["S_V_full"], struct_note)
        for row in sub:
            assert abs(float(row["process_tensor_trace"]) - 8.0) < TRACE_TOL, row
            assert float(row["minimum_eigenvalue"]) > MINEV_TOL, row


def _causal_break_summary(rows: list[dict[str, float | int]]) -> float:
    """Return the maximum causal-break probe error across all rows."""
    return max(float(r["causal_break_error"]) for r in rows)


def _print_table(rows: list[dict[str, float | int]]) -> None:
    header = (
        f"{'cut':>4}  {'J':>6}  {'S_PT_cb':>14}  {'S_V_full':>14}  "
        f"{'trace':>12}  {'min_eval':>12}  {'cb_err':>12}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for row in rows:
        print(
            f"{int(row['cut']):4d}  {float(row['J']):6.3g}  "
            f"{float(row['S_PT_cb']):14.6e}  {float(row['S_V_full']):14.6e}  "
            f"{float(row['process_tensor_trace']):12.6f}  "
            f"{float(row['minimum_eigenvalue']):12.3e}  "
            f"{float(row['causal_break_error']):12.3e}",
            flush=True,
        )


def _select_retained_cuts(rows: list[dict[str, float | int]]) -> tuple[list[int], dict[int, str]]:
    """Decide which cuts appear in the main figure."""
    reasons: dict[int, str] = {}
    retained: list[int] = []
    for cut in CUTS:
        sub = [r for r in rows if int(r["cut"]) == cut]
        sv_all_zero = all(abs(float(r["S_V_full"])) < J0_TOL for r in sub)
        ipt_all_zero = all(abs(float(r["S_PT_cb"])) < J0_TOL for r in sub)
        pos = [r for r in sub if float(r["J"]) > 1e-12]
        sv_range = max(float(r["S_V_full"]) for r in pos) - min(float(r["S_V_full"]) for r in pos)
        ipt_range = max(float(r["S_PT_cb"]) for r in pos) - min(float(r["S_PT_cb"]) for r in pos)
        if cut == 1 and sv_all_zero:
            reasons[cut] = "omitted: S_V^full identically zero (no nontrivial pre-break history at c=1)"
            continue
        if ipt_all_zero and sv_all_zero:
            reasons[cut] = "omitted: both quantities identically zero for all J"
            continue
        if sv_range < 1e-14 and ipt_range < 1e-14:
            reasons[cut] = "omitted: no J-dependent variation above numerical floor"
            continue
        retained.append(cut)
        reasons[cut] = (
            f"retained: S_PT^cb range={ipt_range:.3e}, S_V^full range={sv_range:.3e} over J>0"
        )
    return retained, reasons


def _bootstrap_log_slope(x: np.ndarray, y: np.ndarray, *, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    log_x = np.log10(x)
    log_y = np.log10(y)
    n = x.size
    slopes: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        b, _a = np.polyfit(log_x[idx], log_y[idx], 1)
        slopes.append(float(b))
    arr = np.asarray(slopes, dtype=np.float64)
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def _fit_diagnostics(
    rows: list[dict[str, float | int]],
    retained: list[int],
) -> dict[str, Any]:
    from scipy import stats as sp_stats

    diag: dict[str, Any] = {"per_cut": {}, "pooled": {}, "show_pooled_fit": False}
    slopes: dict[int, float] = {}
    slope_ci: dict[int, tuple[float, float]] = {}
    spearman: dict[int, float] = {}

    for cut in retained:
        sub = [r for r in rows if int(r["cut"]) == cut and float(r["J"]) > 1e-12]
        x = np.asarray([float(r["S_PT_cb"]) for r in sub], dtype=np.float64)
        y = np.asarray([float(r["S_V_full"]) for r in sub], dtype=np.float64)
        log_x = np.log10(x)
        log_y = np.log10(y)
        slope, intercept = np.polyfit(log_x, log_y, 1)
        alpha_mean, alpha_lo, alpha_hi = _bootstrap_log_slope(x, y, seed=1000 + cut)
        pearson = float(sp_stats.pearsonr(log_x, log_y).statistic)
        rho = float(sp_stats.spearmanr(x, y).statistic)
        slopes[cut] = float(slope)
        slope_ci[cut] = (alpha_lo, alpha_hi)
        spearman[cut] = rho
        diag["per_cut"][cut] = {
            "alpha": float(slope),
            "alpha_bootstrap_mean": alpha_mean,
            "alpha_ci_95": (alpha_lo, alpha_hi),
            "pearson_log": pearson,
            "spearman": rho,
            "intercept": float(intercept),
        }

    if len(retained) >= 1:
        sub = [r for r in rows if int(r["cut"]) in retained and float(r["J"]) > 1e-12]
        x = np.asarray([float(r["S_PT_cb"]) for r in sub], dtype=np.float64)
        y = np.asarray([float(r["S_V_full"]) for r in sub], dtype=np.float64)
        log_x = np.log10(x)
        log_y = np.log10(y)
        slope, intercept = np.polyfit(log_x, log_y, 1)
        alpha_mean, alpha_lo, alpha_hi = _bootstrap_log_slope(x, y, seed=9999)
        pearson = float(sp_stats.pearsonr(log_x, log_y).statistic)
        rho = float(sp_stats.spearmanr(x, y).statistic)
        diag["pooled"] = {
            "alpha": float(slope),
            "alpha_bootstrap_mean": alpha_mean,
            "alpha_ci_95": (alpha_lo, alpha_hi),
            "pearson_log": pearson,
            "spearman": rho,
            "intercept": float(intercept),
        }

        spearman_ok = all(spearman[c] > 0.98 for c in retained)
        slope_ok = True
        if len(retained) >= 2:
            svals = [slopes[c] for c in retained]
            for i, ci in enumerate(retained):
                for cj in retained[i + 1 :]:
                    rel_diff = abs(slopes[ci] - slopes[cj]) / max(slopes[ci], slopes[cj], 1e-30)
                    lo_i, hi_i = slope_ci[ci]
                    lo_j, hi_j = slope_ci[cj]
                    overlap = not (hi_i < lo_j or hi_j < lo_i)
                    if rel_diff >= 0.10 and not overlap:
                        slope_ok = False
        pooled_slope = float(slope)
        pooled_intercept = float(intercept)
        log_pred = pooled_intercept + pooled_slope * log_x
        residuals = log_y - log_pred
        cut_means = []
        for cut in retained:
            idx = np.asarray([int(r["cut"]) == cut for r in sub], dtype=bool)
            cut_means.append(float(np.mean(residuals[idx])))
        residual_spread = max(cut_means) - min(cut_means) if cut_means else 0.0
        residual_ok = residual_spread < 0.12
        diag["collapse_tests"] = {
            "spearman_all_gt_0.98": spearman_ok,
            "slope_agreement": slope_ok,
            "residual_cut_spread_log10": residual_spread,
            "residual_separation_ok": residual_ok,
        }
        diag["show_pooled_fit"] = bool(spearman_ok and slope_ok and residual_ok)

    return diag


def _plot_main_figure(
    rows: list[dict[str, float | int]],
    retained: list[int],
    *,
    out_dir: Path,
    fit_diag: dict[str, Any],
) -> None:
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

    _ = fit_diag  # diagnostics only; never drawn on the figure

    pos_rows = [r for r in rows if int(r["cut"]) in retained and float(r["J"]) > 1e-12]
    pos_sv = [float(r["S_V_full"]) for r in pos_rows if float(r["S_V_full"]) > 0.0]
    pos_ipt = [float(r["S_PT_cb"]) for r in pos_rows if float(r["S_PT_cb"]) > 0.0]
    all_pos = pos_sv + pos_ipt
    y_lo = min(all_pos) / np.sqrt(10.0) if all_pos else 1e-9
    y_lo = max(y_lo, 1e-10)
    y_hi = 4e-2

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
        spt = np.asarray([float(r["S_PT_cb"]) for r in sub], dtype=np.float64)
        svf = np.asarray([float(r["S_V_full"]) for r in sub], dtype=np.float64)
        color = CUT_COLORS[cut]
        _plot_style_series(
            ax_a,
            j,
            spt,
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
    ax_a.set_xlim(0.12, j_max + 0.05)
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
            label=r"$S_{\mathrm{PT}}^{\mathrm{cb}}$",
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
        x = np.asarray([float(r["S_PT_cb"]) for r in sub], dtype=np.float64)
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

    ax_b.set_xlabel(r"$S_{\mathrm{PT}}^{\mathrm{cb}}$")
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
    cbar.set_ticks([0.5, 1.0, 1.5, 2.0] if j_max >= 1.5 else [j_min, (j_min + j_max) / 2, j_max])
    cbar.ax.tick_params(length=3, width=0.6, labelsize=8.5)

    stem = out_dir / "pt_response_process_tensor_main"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)


def _write_report(
    path: Path,
    *,
    rows: list[dict[str, float | int]],
    retained: list[int],
    cut_reasons: dict[int, str],
    fit_diag: dict[str, Any],
    file_meta: list[str],
) -> None:
    lines = [
        "Process-tensor main-figure validation report",
        "=========================================",
        f"Parameters: L={L_DEFAULT}, k={K}, g={G_DEFAULT}, dt={DT_DEFAULT}",
        "",
        "Cut retention:",
    ]
    for cut in CUTS:
        lines.append(f"  c={cut}: {cut_reasons.get(cut, 'not evaluated')}")
    lines.append(f"Retained cuts for figure: {retained}")
    lines.append("")
    lines.append(f"Max causal-break probe error: {max(float(r['causal_break_error']) for r in rows):.3e}")
    lines.append("(Swap-probe upper bound; Choi trace and min-eigenvalue checks enforced separately.)")
    lines.append("")
    lines.append("J=0 validation (all cuts):")
    for cut in CUTS:
        j0 = next(r for r in rows if int(r["cut"]) == cut and abs(float(r["J"])) < 1e-12)
        lines.append(
            f"  c={cut}: S_PT^cb={float(j0['S_PT_cb']):.3e}, S_V^full={float(j0['S_V_full']):.3e}"
        )
    lines.append("")
    lines.append("Fit diagnostics (not shown on figure unless collapse tests pass):")
    for cut, info in sorted(fit_diag.get("per_cut", {}).items()):
        ci = info["alpha_ci_95"]
        lines.append(
            f"  c={cut}: alpha={info['alpha']:.4f}, CI=[{ci[0]:.4f}, {ci[1]:.4f}], "
            f"Pearson r_log={info['pearson_log']:.6f}, Spearman={info['spearman']:.6f}"
        )
    if fit_diag.get("pooled"):
        p = fit_diag["pooled"]
        ci = p["alpha_ci_95"]
        lines.append(
            f"  pooled: alpha={p['alpha']:.4f}, CI=[{ci[0]:.4f}, {ci[1]:.4f}], "
            f"Pearson r_log={p['pearson_log']:.6f}, Spearman={p['spearman']:.6f}"
        )
    if fit_diag.get("collapse_tests"):
        lines.append(f"  collapse tests: {fit_diag['collapse_tests']}")
    lines.append(f"  pooled fit plotted: {fit_diag.get('show_pooled_fit', False)}")
    lines.append("")
    lines.append("Output file checksums:")
    lines.extend(file_meta)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("save/pt_cut_reference"))
    p.add_argument("--profile", choices=("smoke", "default"), default="default")
    p.add_argument("--data-csv", type=Path, default=None, help="Existing CSV to load instead of recomputing.")
    p.add_argument("--summary-cb-csv", type=Path, default=None, help="Reuse c=2 S_PT^cb from summary_cb.csv.")
    args = p.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "pt_response_process_tensor_main_data.csv"
    summary_cb = args.summary_cb_csv or (out_dir / "summary_cb.csv")

    if args.data_csv and args.data_csv.exists():
        rows = _load_csv_rows(args.data_csv)
    elif data_path.exists():
        rows = _load_csv_rows(data_path)
    else:
        j_values = list(PROFILE_JS[str(args.profile)])
        print(f"Running multi-cut benchmark: cuts={CUTS}, J points={len(j_values)}", flush=True)
        rows = run_benchmark(
            j_values,
            out_dir=out_dir,
            summary_cb_csv=summary_cb,
            main_csv=data_path if data_path.exists() else None,
        )

    if any("S_PT_cb" not in r for r in rows):
        print("Augmenting rows with S_PT^cb...", flush=True)
        rows = _augment_rows_with_sptcb(rows, summary_cb_csv=summary_cb)

    _validate_rows(rows)
    cb_max = _causal_break_summary(rows)
    _print_table(rows)
    print(f"\nMax causal-break probe error (Monte Carlo): {cb_max:.3e}", flush=True)

    retained, cut_reasons = _select_retained_cuts(rows)
    print("\nCut retention:", flush=True)
    for cut in CUTS:
        print(f"  c={cut}: {cut_reasons.get(cut, '')}", flush=True)
    print(f"Retained: {retained}", flush=True)

    export_rows = [
        {
            "cut": int(r["cut"]),
            "J": float(r["J"]),
            "S_PT_cb": float(r["S_PT_cb"]),
            "S_V_full": float(r["S_V_full"]),
            "process_tensor_trace": float(r["process_tensor_trace"]),
            "minimum_eigenvalue": float(r["minimum_eigenvalue"]),
            "causal_break_error": float(r["causal_break_error"]),
        }
        for r in rows
    ]
    write_csv(data_path, export_rows)

    fit_diag = _fit_diagnostics(rows, retained)
    _plot_main_figure(rows, retained, out_dir=out_dir, fit_diag=fit_diag)

    report_path = out_dir / "pt_response_process_tensor_main_report.txt"
    file_meta: list[str] = []
    for name in (
        "pt_response_process_tensor_main.pdf",
        "pt_response_process_tensor_main.svg",
        "pt_response_process_tensor_main.png",
        "pt_response_process_tensor_main_data.csv",
    ):
        path = out_dir / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        file_meta.append(f"  {name}: mtime={mtime}, sha256={digest}")

    _write_report(
        report_path,
        rows=rows,
        retained=retained,
        cut_reasons=cut_reasons,
        fit_diag=fit_diag,
        file_meta=file_meta,
    )
    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    mtime = datetime.fromtimestamp(report_path.stat().st_mtime, tz=timezone.utc).isoformat()
    file_meta.append(f"  pt_response_process_tensor_main_report.txt: mtime={mtime}, sha256={digest}")

    print("\nWrote main figure to", out_dir, flush=True)
    for line in file_meta:
        print(line, flush=True)


if __name__ == "__main__":
    main()
