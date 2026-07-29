# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Publication 1×3 fixed bond-dimension circuit figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from config import (
    CHI_HEISENBERG,
    CHI_HORIZON,
    CHI_MAIN,
    CORRECTED_OUTPUT_DIR,
    DT,
    FIGURE_STEM,
    FIGURES_DIR,
    METHODS,
    PLOT_FLOOR,
    RELIABILITY_THRESHOLD,
    TFIM_TRAJ_PLOT_TMAX,
)
from matplotlib.lines import Line2D

FIG_WIDTH_MM = 180.0
FIG_HEIGHT_MM = 58.0
MM_TO_IN = 1.0 / 25.4
DPI = 600

# Match corrected single-gate figure.
METHOD_LABELS = {
    "hybrid_tdvp": r"TDVP",
    "tebd_swap": "TEBD+SWAP",
    "mpo_zipup": "MPO zip-up",
    "variational_mpo": "Variational MPO",
}
METHOD_STYLES = {
    "hybrid_tdvp": {"color": "#E31A1C", "marker": "o", "linestyle": "-", "fillstyle": "full"},
    "tebd_swap": {"color": "#1F78B4", "marker": "^", "linestyle": "-", "fillstyle": "full"},
    "mpo_zipup": {"color": "#33A02C", "marker": "s", "linestyle": "-", "fillstyle": "full"},
    "variational_mpo": {"color": "#FF7F00", "marker": "D", "linestyle": "--", "fillstyle": "none"},
}


def _apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.5,
            "lines.linewidth": 0.9,
            "lines.markersize": 3.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _method_handles(methods: tuple[str, ...]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLES[m]["color"],
            marker=METHOD_STYLES[m]["marker"],
            linestyle=METHOD_STYLES[m]["linestyle"],
            fillstyle=METHOD_STYLES[m]["fillstyle"],
            markeredgecolor=METHOD_STYLES[m]["color"],
            markeredgewidth=0.5,
            markersize=3.5,
            label=METHOD_LABELS[m],
        )
        for m in methods
    ]


def plot_figure(
    *,
    horizons: list[dict[str, Any]],
    traj_rows: list[dict[str, Any]],
    heis_rows: list[dict[str, Any]],
    heis_mode: str,
    methods: tuple[str, ...],
    tdvp_n: int,
) -> plt.Figure:
    _apply_style()
    fig = plt.figure(figsize=(FIG_WIDTH_MM * MM_TO_IN, FIG_HEIGHT_MM * MM_TO_IN))
    ax_a = fig.add_subplot(1, 3, 1)
    ax_b = fig.add_subplot(1, 3, 2)
    ax_c = fig.add_subplot(1, 3, 3)

    # (a) TFIM T_ε vs χ
    for method in methods:
        style = METHOD_STYLES[method]
        pts = sorted(
            [h for h in horizons if h["model"] == "ising" and h["method"] == method],
            key=lambda h: int(h["chi_max"]),
        )
        if not pts:
            continue
        xs = np.array([int(p["chi_max"]) for p in pts], dtype=float)
        ys = np.array([float(p["T_eps"]) for p in pts], dtype=float)
        ax_a.plot(xs, ys, color=style["color"], linestyle=style["linestyle"], zorder=2, linewidth=0.85)
        for p in pts:
            cens = int(p.get("right_censored", 0) or 0)
            ax_a.plot(
                int(p["chi_max"]),
                float(p["T_eps"]),
                linestyle="none",
                marker="^" if cens else style["marker"],
                color=style["color"],
                fillstyle=style["fillstyle"],
                markeredgecolor=style["color"],
                markeredgewidth=0.5,
                markersize=3.2 if cens else 2.8,
                zorder=3,
            )
    ax_a.set_xscale("log", base=2)
    ax_a.set_xticks(list(CHI_HORIZON))
    ax_a.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_a.set_xlim(CHI_HORIZON[0] / 1.2, CHI_HORIZON[-1] * 1.2)
    ax_a.set_xlabel(r"$\chi_{\max}$")
    ax_a.set_ylabel(rf"$T_{{\varepsilon}}$ ($\varepsilon={RELIABILITY_THRESHOLD:g}$)")
    ax_a.grid(True, axis="y", color="0.92", linewidth=0.35)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # (b) TFIM trajectory at χ=32
    y_lo = 1e-4
    y_hi = 2.0
    for method in methods:
        style = METHOD_STYLES[method]
        subset = sorted(
            [
                r
                for r in traj_rows
                if r["model"] == "ising"
                and r["method"] == method
                and int(float(r["chi_max"])) == CHI_MAIN
                and float(r["time"]) <= TFIM_TRAJ_PLOT_TMAX + 1e-12
            ],
            key=lambda r: float(r["time"]),
        )
        if not subset:
            continue
        ts = np.array([float(r["time"]) for r in subset])
        raw = np.array([float(r["infidelity"]) for r in subset])
        ys = np.maximum(raw, PLOT_FLOOR)
        ax_b.plot(ts, ys, color=style["color"], linestyle=style["linestyle"], zorder=2)
        ax_b.plot(
            ts,
            ys,
            linestyle="none",
            marker=style["marker"],
            color=style["color"],
            fillstyle=style["fillstyle"],
            markeredgecolor=style["color"],
            markeredgewidth=0.5,
            markersize=2.8,
            zorder=3,
        )
        for t, inf in zip(ts, raw, strict=True):
            if t <= 0:
                continue
            if inf >= RELIABILITY_THRESHOLD:
                ax_b.plot(t, max(inf, PLOT_FLOOR), marker="x", color=style["color"], markersize=3.5, zorder=4)
                break
    ax_b.axhline(RELIABILITY_THRESHOLD, color="0.45", linestyle="--", linewidth=0.6, zorder=1)
    ax_b.text(
        TFIM_TRAJ_PLOT_TMAX * 0.98,
        RELIABILITY_THRESHOLD * 0.85,
        r"$\varepsilon=10^{-2}$",
        color="0.35",
        fontsize=5.5,
        va="top",
        ha="right",
    )
    ax_b.set_yscale("log")
    ax_b.set_xlim(0.0, TFIM_TRAJ_PLOT_TMAX)
    ax_b.set_ylim(y_lo, y_hi)
    ax_b.set_xlabel(r"Time $t$")
    ax_b.set_ylabel(r"Infidelity ($1{-}F$)")
    ax_b.legend(handles=_method_handles(methods), loc="lower right", frameon=False, fontsize=5.5, handlelength=1.3)
    ax_b.grid(True, which="major", axis="y", color="0.92", linewidth=0.35)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # (c) Heisenberg
    if heis_mode == "one_step":
        for method in methods:
            style = METHOD_STYLES[method]
            pts = sorted(
                [
                    r
                    for r in heis_rows
                    if r["method"] == method and int(float(r["trotter_step"])) == 1
                ],
                key=lambda r: int(float(r["chi_max"])),
            )
            if not pts:
                continue
            xs = np.array([int(float(p["chi_max"])) for p in pts], dtype=float)
            ys = np.maximum(np.array([float(p["infidelity"]) for p in pts]), PLOT_FLOOR)
            ax_c.plot(xs, ys, color=style["color"], linestyle=style["linestyle"], zorder=2, linewidth=0.85)
            ax_c.plot(
                xs,
                ys,
                linestyle="none",
                marker=style["marker"],
                color=style["color"],
                fillstyle=style["fillstyle"],
                markeredgecolor=style["color"],
                markeredgewidth=0.5,
                markersize=2.8,
                zorder=3,
            )
        ax_c.axhline(RELIABILITY_THRESHOLD, color="0.45", linestyle="--", linewidth=0.6, zorder=1)
        ax_c.set_yscale("log")
        ax_c.set_ylabel(r"Infidelity ($1{-}F$) at $\Delta t$")
        ax_c.set_ylim(y_lo, y_hi)
    else:
        for method in methods:
            style = METHOD_STYLES[method]
            pts = sorted(
                [h for h in horizons if h["model"] == "heisenberg" and h["method"] == method],
                key=lambda h: int(h["chi_max"]),
            )
            if not pts:
                continue
            xs = np.array([int(p["chi_max"]) for p in pts], dtype=float)
            ys = np.array([float(p["T_eps"]) for p in pts], dtype=float)
            ax_c.plot(xs, ys, color=style["color"], linestyle=style["linestyle"], zorder=2, linewidth=0.85)
            for p in pts:
                cens = int(p.get("right_censored", 0) or 0)
                ax_c.plot(
                    int(p["chi_max"]),
                    float(p["T_eps"]),
                    linestyle="none",
                    marker="^" if cens else style["marker"],
                    color=style["color"],
                    fillstyle=style["fillstyle"],
                    markeredgecolor=style["color"],
                    markeredgewidth=0.5,
                    markersize=3.2 if cens else 2.8,
                    zorder=3,
                )
        ax_c.set_ylabel(rf"$T_{{\varepsilon}}$ ($\varepsilon={RELIABILITY_THRESHOLD:g}$)")
    ax_c.set_xscale("log", base=2)
    ax_c.set_xticks([c for c in CHI_HEISENBERG if c in (2, 4, 8, 16, 32, 64, 128)])
    ax_c.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_c.set_xlim(CHI_HEISENBERG[0] / 1.2, CHI_HEISENBERG[-1] * 1.2)
    ax_c.set_xlabel(r"$\chi_{\max}$")
    ax_c.grid(True, which="major", axis="y", color="0.92", linewidth=0.35)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    for ax, lab in zip((ax_a, ax_b, ax_c), ("(a)", "(b)", "(c)"), strict=True):
        ax.text(0.03, 0.97, lab, transform=ax.transAxes, fontsize=8.0, fontweight="bold", va="top")

    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.18, top=0.92, wspace=0.38)
    fig.text(
        0.5,
        0.02,
        rf"Fixed $\chi_{{\max}}$ comparison; TDVP $n={tdvp_n}$ substeps. Not a fixed-memory study.",
        ha="center",
        fontsize=5.5,
        color="0.35",
    )
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=CORRECTED_OUTPUT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args(argv)
    out = args.output_dir.resolve()
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    tdvp_n = int(cfg["tdvp_substeps"])
    heis_multi = bool(cfg.get("heisenberg_has_multistep_horizon", False))

    results = _load_csv(out / "circuit_results_corrected.csv")
    horizons_raw = _load_csv(out / "circuit_horizons_corrected.csv")
    methods = METHODS
    # Include variational in main figure only if config requests it.
    if cfg.get("include_variational_in_main"):
        methods = (*METHODS, "variational_mpo")

    horizons = [
        {
            "model": r["model"],
            "method": r["method"],
            "chi_max": int(float(r["chi_max"])),
            "T_eps": float(r["T_eps"]),
            "right_censored": int(float(r.get("right_censored", 0) or 0)),
        }
        for r in horizons_raw
        if abs(float(r.get("epsilon", RELIABILITY_THRESHOLD)) - RELIABILITY_THRESHOLD) < 1e-15
        or "epsilon" not in r
        or r.get("epsilon", "") == ""
    ]
    # threshold_sensitivity embeds epsilon; horizons file from generate includes it
    if horizons and "epsilon" in horizons_raw[0]:
        horizons = [
            {
                "model": r["model"],
                "method": r["method"],
                "chi_max": int(float(r["chi_max"])),
                "T_eps": float(r["T_eps"]),
                "right_censored": int(float(r.get("right_censored", 0) or 0)),
            }
            for r in horizons_raw
            if abs(float(r["epsilon"]) - RELIABILITY_THRESHOLD) < 1e-15
        ]

    heis_mode = "horizon" if heis_multi else "one_step"
    fig = plot_figure(
        horizons=horizons,
        traj_rows=results,
        heis_rows=[r for r in results if r["model"] == "heisenberg"],
        heis_mode=heis_mode,
        methods=methods,
        tdvp_n=tdvp_n,
    )
    figs = args.figures_dir.resolve()
    figs.mkdir(parents=True, exist_ok=True)
    pdf = figs / f"{FIGURE_STEM}.pdf"
    png = figs / f"{FIGURE_STEM}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=DPI)
    # Also copy into corrected output
    fig.savefig(out / f"{FIGURE_STEM}.pdf")
    fig.savefig(out / f"{FIGURE_STEM}.png", dpi=DPI)
    plt.close(fig)
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")
    print(f"heis_mode={heis_mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
