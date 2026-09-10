#!/usr/bin/env python3
"""Publication-quality PRX figure: S_V^full versus I_PT with subsampling statistics."""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path
from typing import Any, cast

import numpy as np

from common import (
    BETA,
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
from mqt.yaqs.characterization.memory.operational_memory.response_matrix import (
    assemble_response_matrix,
    response_matrix_entropy,
)
from mqt.yaqs.characterization.memory.operational_memory.run import evaluate_probes_with_weights

# Okabe–Ito (panel-a hierarchy uses local overrides in _plot_main_figure)
COLOR_IPT = "#0072B2"
COLOR_SVFULL = "#D55E00"
COLOR_SV8 = "#E69F00"

N_SUBSETS = 100
MASTER_SEED = 20260704
CONVERGENCE_JS = (0.4, 1.0, 2.0)
CONVERGENCE_BUDGETS = (4, 8, 16, 32, 64)
CLIP_NEG = 1e-10


def _clip_entropy(value: float) -> float:
    if -CLIP_NEG < value < 0.0:
        return 0.0
    return float(value)


def _load_summary(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: float(v) if k not in {"cut", "k", "L"} else int(v) for k, v in row.items()} for row in csv.DictReader(f)]


def _validate_rows(rows: list[dict[str, float]]) -> dict[str, Any]:
    j = np.asarray([r["J"] for r in rows], dtype=np.float64)
    ipt = np.asarray([_clip_entropy(r["I_PT"]) for r in rows])
    svf = np.asarray([_clip_entropy(r["S_V_full"]) for r in rows])
    sv64 = np.asarray([_clip_entropy(r["S_V_64"]) for r in rows])
    trace = np.asarray([r["process_tensor_trace"] for r in rows])
    minev = np.asarray([r["process_tensor_min_eigenvalue"] for r in rows])

    j0 = np.isclose(j, 0.0)
    checks = {
        "I_PT_J0": float(ipt[j0].max()) if np.any(j0) else float("nan"),
        "S_V_full_J0": float(svf[j0].max()) if np.any(j0) else float("nan"),
        "max_abs_SV_full_minus_SV64": float(np.max(np.abs(svf - sv64))),
        "max_trace_deviation": float(np.max(np.abs(trace - 8.0))),
        "min_eigenvalue": float(np.min(minev)),
    }
    assert checks["I_PT_J0"] < 1e-10, checks
    assert checks["S_V_full_J0"] < 1e-10, checks
    assert checks["max_trace_deviation"] < 1e-10, checks
    assert checks["min_eigenvalue"] > -1e-10, checks
    assert checks["max_abs_SV_full_minus_SV64"] < 1e-12, checks
    return checks


def _ensure_response_cache(
    rows: list[dict[str, float]],
    cache_path: Path,
    *,
    cut: int,
    k: int,
) -> dict[str, np.ndarray]:
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        return {key: data[key] for key in data.files}

    catalog = enumerate_full_probe_catalog(cut=cut, num_interventions=k)
    n_p = len(catalog.past_settings)
    probe_full = build_probe_set_from_catalog(catalog, np.arange(n_p), np.arange(n_p))
    mc = characterizer(parallel=False)
    params = sim_params(dt=DT_DEFAULT)
    timesteps = [DT_DEFAULT] * (k + 1)

    j_vals: list[float] = []
    pauli_stack: list[np.ndarray] = []
    weight_stack: list[np.ndarray] = []
    for row in rows:
        jv = float(row["J"])
        print(f"  caching response arrays J={jv:g}", flush=True)
        ham = ising_chain(length=L_DEFAULT, j=jv, g=G_DEFAULT)
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
        pauli, weights = evaluate_probes_with_weights(pt, probe_full)
        j_vals.append(jv)
        pauli_stack.append(np.asarray(pauli, dtype=np.float64))
        weight_stack.append(np.asarray(weights, dtype=np.float64))

    out = {
        "J": np.asarray(j_vals, dtype=np.float64),
        "pauli": np.stack(pauli_stack, axis=0),
        "weights": np.stack(weight_stack, axis=0),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **out)
    return out


def _draw_subset_indices(
    n: int,
    m: int,
    *,
    n_subsets: int,
    master_seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    master = np.random.default_rng(master_seed)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_subsets):
        child = np.random.default_rng(int(master.integers(0, 2**31 - 1)))
        past = np.sort(child.choice(n, m, replace=False))
        fut = np.sort(child.choice(n, m, replace=False))
        out.append((past, fut))
    return out


def _sv_from_arrays(
    pauli: np.ndarray,
    weights: np.ndarray,
    past_idx: np.ndarray,
    future_idx: np.ndarray,
) -> float:
    p = pauli[np.ix_(past_idx, future_idx)]
    w = weights[np.ix_(past_idx, future_idx)]
    _raw, response = assemble_response_matrix(p, w, beta=BETA, center=True)
    return _clip_entropy(response_matrix_entropy(response))


def _subsampling_stats(
    cache: dict[str, np.ndarray],
    rows: list[dict[str, float]],
    *,
    m: int,
    n_subsets: int,
    master_seed: int,
    subset_draws: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> list[dict[str, float]]:
    n = int(cache["pauli"].shape[1])
    draws = subset_draws or _draw_subset_indices(n, m, n_subsets=n_subsets, master_seed=master_seed)
    stats_rows: list[dict[str, float]] = []
    for i, row in enumerate(rows):
        jv = float(row["J"])
        j_idx = int(np.where(np.isclose(cache["J"], jv))[0][0])
        pauli = cache["pauli"][j_idx]
        weights = cache["weights"][j_idx]
        samples = np.asarray([_sv_from_arrays(pauli, weights, pi, fi) for pi, fi in draws], dtype=np.float64)
        samples = np.asarray([_clip_entropy(v) for v in samples], dtype=np.float64)
        stats_rows.append(
            {
                "J": jv,
                "m": float(m),
                "median": float(np.median(samples)),
                "p16": float(np.percentile(samples, 16)),
                "p84": float(np.percentile(samples, 84)),
                "p05": float(np.percentile(samples, 5)),
                "p95": float(np.percentile(samples, 95)),
                "S_V_full": _clip_entropy(float(row["S_V_full"])),
            }
        )
    return stats_rows


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


def _verify_rendered_figure(
    png_path: Path,
    *,
    y_floor: float,
    checks: dict[str, Any],
) -> dict[str, Any]:
    """Programmatic checks on saved figure metadata and PNG bytes."""
    import hashlib
    from datetime import datetime, timezone

    png_path = png_path.resolve()
    data = png_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    mtime = datetime.fromtimestamp(png_path.stat().st_mtime, tz=timezone.utc).isoformat()

    verify: dict[str, Any] = {
        "png_sha256": digest,
        "png_mtime_utc": mtime,
        "png_bytes": len(data),
        **checks,
    }
    assert checks["panel_a_connected_lines"] == 3, verify
    assert checks["panel_a_has_fill_band"], verify
    assert checks["panel_a_no_sv64_curve"], verify
    assert checks["y_min"] >= 1e-8, verify
    assert checks["y_min"] < 1e-6, verify
    assert not checks["j0_connected_to_j02"], verify
    assert checks["panel_b_legend_entries"] == 0, verify
    assert checks["panel_b_positive_points_only"], verify
    return verify


def _plot_main_figure(
    rows: list[dict[str, float]],
    stats8: list[dict[str, float]],
    *,
    out_dir: Path,
    y_floor: float,
    fit_meta: dict[str, float],
) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
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

    color_ipt = COLOR_IPT
    color_svfull = "#B22222"
    color_sv8 = "#F5A623"
    band_alpha = 0.30
    panel_label_size = 10.5

    j_all = np.asarray([r["J"] for r in rows], dtype=np.float64)
    pos = j_all > 1e-12
    j = j_all[pos]
    ipt = np.asarray([_clip_entropy(r["I_PT"]) for r in rows], dtype=np.float64)[pos]
    svf = np.asarray([_clip_entropy(r["S_V_full"]) for r in rows], dtype=np.float64)[pos]

    med8 = np.asarray([s["median"] for s in stats8 if s["J"] > 1e-12], dtype=np.float64)
    p16 = np.asarray([s["p16"] for s in stats8 if s["J"] > 1e-12], dtype=np.float64)
    p84 = np.asarray([s["p84"] for s in stats8 if s["J"] > 1e-12], dtype=np.float64)

    fig = plt.figure(figsize=(7.15, 2.95))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.15, 1.0],
        wspace=0.28,
        left=0.08,
        right=0.90,
        bottom=0.17,
        top=0.95,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    y_top = max(float(np.max(np.concatenate([ipt, svf, p84]))), y_floor) * 2.5

    def _plot_pos(
        ax,
        x,
        y,
        *,
        color,
        marker,
        lw,
        ms,
        ls="-",
        mfc=None,
        mec=None,
        alpha=1.0,
        zorder=3,
    ):
        yp = np.maximum(y, y_floor)
        ax.plot(
            x,
            yp,
            ls=ls,
            color=color,
            marker=marker,
            lw=lw,
            ms=ms,
            alpha=alpha,
            zorder=zorder,
            mfc=mfc or color,
            mec=mec or color,
            mew=0.85,
        )

    _plot_pos(ax_a, j, ipt, color=color_ipt, marker="o", lw=2.0, ms=6.0, zorder=4)
    _plot_pos(ax_a, j, svf, color=color_svfull, marker="D", lw=1.8, ms=6.0, zorder=3)
    _plot_pos(
        ax_a,
        j,
        med8,
        color=color_sv8,
        marker="s",
        lw=1.3,
        ms=5.0,
        mfc="none",
        mec=color_sv8,
        ls="--",
        alpha=0.95,
        zorder=2,
    )
    ax_a.fill_between(
        j,
        np.maximum(p16, y_floor),
        np.maximum(p84, y_floor),
        color=color_sv8,
        alpha=band_alpha,
        linewidth=0,
        edgecolor="none",
        zorder=1,
    )

    ax_a.plot(
        0.0,
        y_floor,
        marker="v",
        color="0.40",
        ms=5.0,
        mfc="0.40",
        mec="0.15",
        mew=0.8,
        linestyle="none",
        zorder=4,
        clip_on=False,
    )
    ax_a.text(
        0.04,
        0.10,
        r"$J=0$: $S_V^{\mathrm{full}}=I_{\mathrm{PT}}=0$",
        transform=ax_a.transAxes,
        fontsize=7.8,
        color="0.35",
        va="bottom",
        ha="left",
    )

    ax_a.set_yscale("log")
    ax_a.set_xlabel(r"Coupling $J$")
    ax_a.set_ylabel("Entropy (nats)")
    ax_a.set_xlim(-0.02, float(j_all.max()) + 0.05)
    ax_a.set_ylim(y_floor, y_top)
    ax_a.yaxis.set_major_formatter(LogFormatterMathtext())
    ax_a.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    legend_handles = [
        Line2D([0], [0], color=color_ipt, marker="o", lw=2.0, ms=6.0, label=r"$I_{\mathrm{PT}}$"),
        Line2D([0], [0], color=color_svfull, marker="D", lw=1.8, ms=6.0, label=r"$S_V^{\mathrm{full}}$"),
        Line2D(
            [0],
            [0],
            color=color_sv8,
            marker="s",
            lw=1.3,
            ms=5.0,
            ls="--",
            mfc="none",
            mec=color_sv8,
            label=r"$S_V^{(8\times 8)}$ median",
        ),
        Patch(facecolor=color_sv8, alpha=band_alpha, edgecolor="none", label="16–84% interval"),
    ]
    ax_a.legend(handles=legend_handles, frameon=False, loc="lower right", fontsize=8.0)
    ax_a.text(0.03, 0.97, "(a)", transform=ax_a.transAxes, fontsize=panel_label_size, fontweight="bold", va="top")
    for spine in ax_a.spines.values():
        spine.set_linewidth(0.85)
    ax_a.tick_params(direction="in", top=True, right=True, which="both", length=3.5, width=0.75)

    norm = Normalize(vmin=float(j.min()), vmax=float(j.max()))
    cmap = plt.colormaps["plasma"]
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    sc = ax_b.scatter(
        ipt,
        svf,
        c=j,
        cmap=cmap,
        norm=norm,
        s=52,
        edgecolors="0.15",
        linewidths=0.85,
        zorder=3,
    )
    if ipt.size >= 2:
        x_line = np.geomspace(float(ipt.min()), float(ipt.max()), 100)
        y_line = 10 ** (fit_meta["intercept"] + fit_meta["slope"] * np.log10(x_line))
        ax_b.plot(x_line, y_line, "--", color="0.45", lw=1.2, zorder=2)
    ax_b.set_xlabel(r"$I_{\mathrm{PT}}$")
    ax_b.set_ylabel(r"$S_V^{\mathrm{full}}$")
    ax_b.xaxis.set_major_formatter(LogFormatterMathtext())
    ax_b.yaxis.set_major_formatter(LogFormatterMathtext())
    ax_b.text(0.03, 0.97, "(b)", transform=ax_b.transAxes, fontsize=panel_label_size, fontweight="bold", va="top")
    for spine in ax_b.spines.values():
        spine.set_linewidth(0.85)
    ax_b.tick_params(direction="in", top=True, right=True, which="both", length=3.5, width=0.75)

    fig.canvas.draw()
    pos_b = ax_b.get_position()
    cax = fig.add_axes([pos_b.x1 + 0.012, pos_b.y0, 0.012, pos_b.height])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label(r"Coupling $J$", fontsize=9.5, labelpad=2)
    cbar.set_ticks([0.5, 1.0, 1.5, 2.0])
    cbar.ax.tick_params(length=3, width=0.6, labelsize=8.5)

    ax_b.text(
        0.03,
        0.03,
        r"$J=0$: both vanish",
        transform=ax_b.transAxes,
        fontsize=7.8,
        color="0.35",
        va="bottom",
        ha="left",
        zorder=5,
    )

    plot_checks = {
        "panel_a_connected_lines": sum(1 for ln in ax_a.lines if len(ln.get_xdata()) > 1),
        "panel_a_j0_markers": sum(
            1 for ln in ax_a.lines if len(ln.get_xdata()) == 1 and ln.get_linestyle() in {"none", "None", ""}
        ),
        "panel_a_has_fill_band": len(ax_a.collections) >= 1,
        "panel_a_no_sv64_curve": all("64" not in ln.get_label() for ln in ax_a.lines),
        "y_min": float(ax_a.get_ylim()[0]),
        "y_max": float(ax_a.get_ylim()[1]),
        "j0_connected_to_j02": any(
            len(ln.get_xdata()) > 1 and np.isclose(ln.get_xdata()[0], 0.0) for ln in ax_a.lines
        ),
        "panel_b_legend_entries": len(ax_b.get_legend().get_texts()) if ax_b.get_legend() else 0,
        "panel_b_positive_points_only": bool(np.all(ipt > 0) and np.all(svf > 0)),
        "panel_b_point_count": int(ipt.size),
        "panel_b_no_fit_stats_annotation": True,
    }

    stem = out_dir / "pt_response_comparison_prx"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)
    return plot_checks


def _plot_convergence_supplement(
    conv_rows: list[dict[str, float]],
    ratio_rows: list[dict[str, float]],
    *,
    out_dir: Path,
    y_floor: float,
) -> None:
    import matplotlib.pyplot as plt

    configure_matplotlib_prl()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"wspace": 0.32})

    for jv in CONVERGENCE_JS:
        sub = [r for r in conv_rows if abs(r["J"] - jv) < 1e-9]
        sub = sorted(sub, key=lambda r: r["m"])
        m = np.asarray([r["m"] for r in sub], dtype=np.float64)
        med = np.asarray([max(r["median_rel_error"], y_floor) for r in sub], dtype=np.float64)
        p16 = np.asarray([max(r["p16_rel_error"], y_floor) for r in sub], dtype=np.float64)
        p84 = np.asarray([max(r["p84_rel_error"], y_floor) for r in sub], dtype=np.float64)
        ax1.plot(m, med, "o-", ms=4, lw=1.4, label=rf"$J={jv:g}$")
        ax1.fill_between(m, p16, p84, alpha=0.18, linewidth=0)

        sub2 = [r for r in ratio_rows if abs(r["J"] - jv) < 1e-9]
        sub2 = sorted(sub2, key=lambda r: r["m"])
        m2 = np.asarray([r["m"] for r in sub2], dtype=np.float64)
        rmed = np.asarray([r["median_ratio"] for r in sub2], dtype=np.float64)
        r16 = np.asarray([r["p16_ratio"] for r in sub2], dtype=np.float64)
        r84 = np.asarray([r["p84_ratio"] for r in sub2], dtype=np.float64)
        ax2.plot(m2, rmed, "o-", ms=4, lw=1.4, label=rf"$J={jv:g}$")
        ax2.fill_between(m2, r16, r84, alpha=0.18, linewidth=0)

    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel(r"Probe budget $m$")
    ax1.set_ylabel(r"median$|S_V^{(m)}-S_V^{\mathrm{full}}|/S_V^{\mathrm{full}}$")
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_title(r"Relative error", fontsize=9)

    ax2.axhline(1.0, color="0.5", lw=0.8, ls=":")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel(r"Probe budget $m$")
    ax2.set_ylabel(r"$S_V^{(m)}/S_V^{\mathrm{full}}$ (median)")
    ax2.legend(frameon=False, fontsize=8)
    ax2.set_title(r"Finite-budget ratio", fontsize=9)

    out = out_dir / "pt_response_convergence_supplement.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02, dpi=600)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("save/pt_cut_reference"))
    p.add_argument("--summary-csv", type=Path, default=None)
    p.add_argument("--n-subsets", type=int, default=N_SUBSETS)
    p.add_argument("--master-seed", type=int, default=MASTER_SEED)
    args = p.parse_args()

    out_dir = args.data_dir.resolve()
    summary_path = args.summary_csv or (out_dir / "summary_sv_full.csv")
    rows = _load_summary(summary_path)
    cut = int(rows[0]["cut"])
    k = int(rows[0]["k"])

    checks = _validate_rows(rows)
    print("Validation checks passed:", checks, flush=True)

    cache_path = out_dir / "response_arrays.npz"
    if not cache_path.exists():
        print("Building response-array cache (one PT evaluation per J)...", flush=True)
    cache = _ensure_response_cache(rows, cache_path, cut=cut, k=k)

    draws8 = _draw_subset_indices(64, 8, n_subsets=args.n_subsets, master_seed=args.master_seed)
    stats8 = _subsampling_stats(cache, rows, m=8, n_subsets=args.n_subsets, master_seed=args.master_seed, subset_draws=draws8)

    all_sub_stats: list[dict[str, float]] = []
    draws_by_m: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {8: draws8}
    for m in CONVERGENCE_BUDGETS:
        if m == 8:
            continue
        if m == 64:
            draws_by_m[m] = [(np.arange(64), np.arange(64))]
        else:
            draws_by_m[m] = _draw_subset_indices(64, m, n_subsets=args.n_subsets, master_seed=args.master_seed + m)
    for m in CONVERGENCE_BUDGETS:
        all_sub_stats.extend(
            _subsampling_stats(
                cache,
                rows,
                m=m,
                n_subsets=args.n_subsets,
                master_seed=args.master_seed,
                subset_draws=draws_by_m[m],
            )
        )

    pos_sv = [max(_clip_entropy(r["S_V_full"]), 0.0) for r in rows if r["J"] > 1e-12]
    y_floor = max(1e-7, min(pos_sv) * 0.35 if pos_sv else 1e-7)

    # Fit panel (b)
    j_pos = np.asarray([r["J"] for r in rows if r["J"] > 1e-12], dtype=np.float64)
    ipt_pos = np.asarray([max(_clip_entropy(r["I_PT"]), y_floor) for r in rows if r["J"] > 1e-12])
    sv_pos = np.asarray([max(_clip_entropy(r["S_V_full"]), y_floor) for r in rows if r["J"] > 1e-12])
    from scipy import stats as sp_stats

    log_i = np.log10(ipt_pos)
    log_s = np.log10(sv_pos)
    slope, intercept = np.polyfit(log_i, log_s, 1)
    alpha_mean, alpha_lo, alpha_hi = _bootstrap_log_slope(ipt_pos, sv_pos, seed=args.master_seed)
    pearson = float(sp_stats.pearsonr(log_i, log_s).statistic)
    spearman = float(sp_stats.spearmanr(ipt_pos, sv_pos).statistic)
    fit_meta = {"slope": float(slope), "intercept": float(intercept), "alpha_mean": alpha_mean, "alpha_lo": alpha_lo, "alpha_hi": alpha_hi, "pearson_log": pearson, "spearman": spearman}

    # Convergence statistics
    conv_stats: list[dict[str, float]] = []
    ratio_stats: list[dict[str, float]] = []
    for m in CONVERGENCE_BUDGETS:
        if m == 64:
            draws = [(np.arange(64), np.arange(64))]
        else:
            draws = _draw_subset_indices(64, m, n_subsets=args.n_subsets, master_seed=args.master_seed + m)
        for jv in CONVERGENCE_JS:
            row = next(r for r in rows if abs(float(r["J"]) - jv) < 1e-9)
            j_idx = int(np.where(np.isclose(cache["J"], jv))[0][0])
            sv_full = max(_clip_entropy(float(row["S_V_full"])), y_floor)
            pauli = cache["pauli"][j_idx]
            weights = cache["weights"][j_idx]
            samples = np.asarray([_sv_from_arrays(pauli, weights, pi, fi) for pi, fi in draws], dtype=np.float64)
            rel = np.abs(samples - sv_full) / sv_full if sv_full > 0 else np.zeros_like(samples)
            ratio = samples / sv_full if sv_full > 0 else np.ones_like(samples)
            conv_stats.append(
                {
                    "J": float(jv),
                    "m": float(m),
                    "median_rel_error": float(np.median(rel)),
                    "p16_rel_error": float(np.percentile(rel, 16)),
                    "p84_rel_error": float(np.percentile(rel, 84)),
                }
            )
            ratio_stats.append(
                {
                    "J": float(jv),
                    "m": float(m),
                    "median_ratio": float(np.median(ratio)),
                    "p16_ratio": float(np.percentile(ratio, 16)),
                    "p84_ratio": float(np.percentile(ratio, 84)),
                }
            )

    # Export data
    plot_rows = []
    for row, s8 in zip(rows, stats8, strict=True):
        plot_rows.append(
            {
                "J": row["J"],
                "I_PT": _clip_entropy(row["I_PT"]),
                "S_V_full": _clip_entropy(row["S_V_full"]),
                "S_V_8_median": s8["median"],
                "S_V_8_p16": s8["p16"],
                "S_V_8_p84": s8["p84"],
                "S_V_8_p05": s8["p05"],
                "S_V_8_p95": s8["p95"],
                "process_tensor_trace": row["process_tensor_trace"],
                "process_tensor_min_eigenvalue": row["process_tensor_min_eigenvalue"],
            }
        )
    write_csv(out_dir / "pt_response_comparison_data.csv", plot_rows)
    write_csv(out_dir / "pt_response_subsampling_statistics.csv", all_sub_stats)

    plot_checks = _plot_main_figure(rows, stats8, out_dir=out_dir, y_floor=y_floor, fit_meta=fit_meta)
    _plot_convergence_supplement(conv_stats, ratio_stats, out_dir=out_dir, y_floor=y_floor)

    import hashlib
    from datetime import datetime, timezone

    output_files = [
        out_dir / "pt_response_comparison_prx.pdf",
        out_dir / "pt_response_comparison_prx.svg",
        out_dir / "pt_response_comparison_prx.png",
        out_dir / "pt_response_comparison_data.csv",
        out_dir / "pt_response_subsampling_statistics.csv",
        out_dir / "pt_response_convergence_supplement.pdf",
    ]
    file_meta: list[str] = []
    for path in output_files:
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        file_meta.append(f"  {path.name}: mtime={mtime}, sha256={digest}")
    verify = _verify_rendered_figure(
        out_dir / "pt_response_comparison_prx.png",
        y_floor=y_floor,
        checks=plot_checks,
    )

    report = textwrap.dedent(
        f"""
        PRX response-comparison validation report
        ========================================
        Full past basis size |P_full| = 64
        Full future basis size |F_full| = 64
        Complete response ensemble = 64 x 64 (S_V^full = S_V at m=64)

        J=0 checks:
          I_PT(J=0) = {checks['I_PT_J0']:.3e}
          S_V^full(J=0) = {checks['S_V_full_J0']:.3e}
          max |trace - 8| = {checks['max_trace_deviation']:.3e}
          min Choi eigenvalue = {checks['min_eigenvalue']:.3e}
          max |S_V^full - S_V_64| = {checks['max_abs_SV_full_minus_SV64']:.3e}

        Subsampling: N_subsets = {args.n_subsets}, master_seed = {args.master_seed}
        Log-log fit (J>0): alpha = {slope:.3f} (bootstrap 95% CI [{alpha_lo:.3f}, {alpha_hi:.3f}])
        Pearson r_log = {pearson:.6f}, Spearman rho = {spearman:.6f}
        Plotting floor y_lo = {y_floor:.3e}

        Figure verification:
          panel (a) connected curves = {plot_checks['panel_a_connected_lines']} (expect 3)
          panel (a) uncertainty band present = {plot_checks['panel_a_has_fill_band']}
          panel (a) no S_V(64) curve = {plot_checks['panel_a_no_sv64_curve']}
          J=0 not connected to J=0.2 = {not plot_checks['j0_connected_to_j02']}
          panel (b) legend entries = {plot_checks['panel_b_legend_entries']} (expect 0)
          panel (b) positive-J points = {plot_checks['panel_b_point_count']}
        """
    ).strip()
    report_lines = [report, "", "Output file checksums:"]
    report_lines.extend(file_meta)
    (out_dir / "pt_response_comparison_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(report, flush=True)
    print("Output file checksums:", flush=True)
    for line in file_meta:
        print(line, flush=True)
    print(f"Figure verification passed: {verify}", flush=True)
    print(f"Wrote figures to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
