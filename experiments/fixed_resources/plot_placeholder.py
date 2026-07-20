# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Publication 1×3 fixed-resource placeholder Results figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from config import DT, METHODS, OUTPUT_DIR, PLOT_FLOOR, RELIABILITY_THRESHOLD
from matplotlib.lines import Line2D

FIG_WIDTH_MM = 180.0
FIG_HEIGHT_MM = 58.0
MM_TO_IN = 1.0 / 25.4
DPI = 600
Y_FLOOR_DISPLAY = PLOT_FLOOR / 3.0

METHOD_LABELS = {
    "hybrid_tdvp": "TDVP",
    "tebd_swap": "TEBD+SWAP",
    "mpo_zipup": "MPO zip-up",
}
METHOD_STYLES = {
    "hybrid_tdvp": {"color": "#0072B2", "marker": "o", "linestyle": "-", "fillstyle": "full"},
    "tebd_swap": {"color": "#D55E00", "marker": "^", "linestyle": "-", "fillstyle": "full"},
    "mpo_zipup": {"color": "#009E73", "marker": "s", "linestyle": "-", "fillstyle": "full"},
}

TFIM_TMAX = 2.0
HORIZON_CHI = (2, 4, 8, 12, 16, 24, 32, 48, 64)
HORIZON_MAJOR_TICKS = (2, 4, 8, 16, 32, 64)
ONE_STEP_CHI = (2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)
ONE_STEP_MAJOR_TICKS = (2, 4, 8, 16, 32, 64, 128)
CONVERGED_TDVP_SUBSTEPS = 4

TRAJ_CSV = OUTPUT_DIR / "trajectories.csv"
TFIM_DENSE_CSV = OUTPUT_DIR / "tfim_horizon_dense.csv"
HEIS_DENSE_CSV = OUTPUT_DIR / "heisenberg_one_step_dense.csv"
PLACEHOLDER_PDF = OUTPUT_DIR / "fixed_resources_placeholder.pdf"
PLACEHOLDER_PNG = OUTPUT_DIR / "fixed_resources_placeholder.png"
PLACEHOLDER_MD = OUTPUT_DIR / "fixed_resources_placeholder.md"


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


def _display_y(val: float) -> float:
    return max(float(val), PLOT_FLOOR)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _method_handles() -> list[Line2D]:
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
        for m in METHODS
    ]


def _annotate_epsilon(ax: plt.Axes, *, x_frac: float = 0.98, ha: str = "right") -> None:
    """Place ε label on the underside of the dashed threshold line."""
    xmin, xmax = ax.get_xlim()
    if ax.get_xscale() == "log":
        x = np.exp(np.log(xmin) + x_frac * (np.log(xmax) - np.log(xmin)))
    else:
        x = xmin + x_frac * (xmax - xmin)
    # Slightly below the threshold on a log axis.
    y = RELIABILITY_THRESHOLD * 0.85
    ax.text(
        x,
        y,
        r"$\varepsilon=10^{-2}$",
        color="0.35",
        fontsize=5.5,
        va="top",
        ha=ha,
        clip_on=False,
        zorder=5,
    )


def load_tfim_horizons() -> list[dict[str, Any]]:
    if not TFIM_DENSE_CSV.exists():
        raise SystemExit(f"Missing {TFIM_DENSE_CSV}")
    rows = _load_csv(TFIM_DENSE_CSV)
    out: list[dict[str, Any]] = []
    for r in rows:
        model = r.get("model", "ising")
        if model not in ("", "ising"):
            continue
        chi = int(float(r["chi_max"]))
        if chi not in HORIZON_CHI:
            continue
        if r["method"] not in METHODS:
            continue
        te = r.get("T_eps") or r.get("first_crossing_time") or ""
        censored = int(float(r.get("right_censored", 0) or 0))
        if te == "" and not censored:
            continue
        t_eps = float(te) if te != "" else TFIM_TMAX
        out.append(
            {
                "method": r["method"],
                "chi_max": chi,
                "n_eps": int(round(t_eps / DT)),
                "right_censored": censored,
            }
        )
    return out


def load_heisenberg_one_step() -> list[dict[str, Any]]:
    if not HEIS_DENSE_CSV.exists():
        raise SystemExit(f"Missing {HEIS_DENSE_CSV}")
    return [
        {
            "method": r["method"],
            "chi_max": int(float(r["chi_max"])),
            "time": float(r.get("time", DT) or DT),
            "infidelity": float(r["infidelity"]),
            "source": r.get("source", ""),
            "tdvp_substeps": r.get("tdvp_substeps", ""),
        }
        for r in _load_csv(HEIS_DENSE_CSV)
        if r["method"] in METHODS and int(float(r["chi_max"])) in ONE_STEP_CHI
    ]


def plot_placeholder(
    traj_rows: list[dict[str, str]],
    horizons: list[dict[str, Any]],
    one_step: list[dict[str, Any]],
) -> plt.Figure:
    _apply_style()
    fig = plt.figure(figsize=(FIG_WIDTH_MM * MM_TO_IN, FIG_HEIGHT_MM * MM_TO_IN))
    ax_a = fig.add_subplot(1, 3, 1)
    ax_b = fig.add_subplot(1, 3, 2)
    ax_c = fig.add_subplot(1, 3, 3, sharey=ax_b)

    y_lo_shared = 1e-4
    y_hi_shared = 2.0
    n_eps_hi = int(round(TFIM_TMAX / DT))

    # (a) 4×4 TFIM nε vs χ
    for method in METHODS:
        style = METHOD_STYLES[method]
        pts = sorted(
            [h for h in horizons if h["method"] == method],
            key=lambda h: h["chi_max"],
        )
        if not pts:
            continue
        xs = np.array([p["chi_max"] for p in pts], dtype=float)
        ys = np.array([p["n_eps"] for p in pts], dtype=float)
        ax_a.plot(xs, ys, color=style["color"], linestyle=style["linestyle"], zorder=2, linewidth=0.85)
        for p in pts:
            cens = int(p.get("right_censored", 0))
            ax_a.plot(
                p["chi_max"],
                p["n_eps"],
                linestyle="none",
                marker="^" if cens else style["marker"],
                color=style["color"],
                fillstyle=style["fillstyle"],
                markeredgecolor=style["color"],
                markeredgewidth=0.5,
                markersize=3.0 if cens else 2.8,
                zorder=3,
            )
    ax_a.set_xscale("log", base=2)
    ax_a.set_xticks(list(HORIZON_MAJOR_TICKS))
    ax_a.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_a.set_xlim(HORIZON_CHI[0] / 1.2, HORIZON_CHI[-1] * 1.2)
    ax_a.set_ylim(0.0, 18.0)
    ax_a.set_xlabel(r"$\chi_{\max}$")
    ax_a.set_ylabel(r"$n_\varepsilon$ (Trotter steps)")
    ax_a.set_title("4×4 TFIM")
    ax_a.grid(True, axis="y", color="0.92", linewidth=0.35)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # (b) 4×4 TFIM infidelity vs Trotter step at χ=32
    max_step = n_eps_hi
    for method in METHODS:
        style = METHOD_STYLES[method]
        subset = sorted(
            [
                r
                for r in traj_rows
                if r["model"] == "ising"
                and r["method"] == method
                and int(float(r["chi_max"])) == 32
                and float(r["time"]) <= TFIM_TMAX + 1e-12
            ],
            key=lambda r: float(r["time"]),
        )
        if not subset:
            continue
        steps = np.array(
            [
                int(float(r["trotter_step"])) if r.get("trotter_step", "") != "" else int(round(float(r["time"]) / DT))
                for r in subset
            ],
            dtype=float,
        )
        raw = np.array([float(r["infidelity"]) for r in subset])
        below = raw < y_lo_shared
        ys = np.array([max(_display_y(v), y_lo_shared) for v in raw])
        ax_b.plot(steps, ys, color=style["color"], linestyle=style["linestyle"], zorder=2)
        ax_b.plot(
            steps[~below],
            ys[~below],
            linestyle="none",
            marker=style["marker"],
            color=style["color"],
            fillstyle=style["fillstyle"],
            markeredgecolor=style["color"],
            markeredgewidth=0.5,
            markersize=2.8,
            zorder=3,
        )
        if np.any(below):
            ax_b.plot(
                steps[below],
                np.full(int(np.sum(below)), y_lo_shared),
                linestyle="none",
                marker="v",
                color=style["color"],
                markersize=2.4,
                zorder=3,
            )
        for step, inf in zip(steps, raw, strict=True):
            if inf >= RELIABILITY_THRESHOLD:
                ax_b.plot(
                    step,
                    max(_display_y(inf), y_lo_shared),
                    marker="x",
                    color=style["color"],
                    markersize=3.0,
                    zorder=4,
                )
                break
    ax_b.axhline(RELIABILITY_THRESHOLD, color="0.45", linestyle="--", linewidth=0.6, zorder=1)
    ax_b.set_yscale("log")
    ax_b.set_xlim(0.0, max_step)
    ax_b.set_ylim(y_lo_shared, y_hi_shared)
    ax_b.set_xlabel(rf"Trotter step ($\Delta t={DT:g}$)")
    ax_b.set_ylabel(r"Infidelity ($1{-}F$)")
    ax_b.set_title(r"4×4 TFIM ($\chi_{\max}=32$)")
    _annotate_epsilon(ax_b, x_frac=0.98, ha="right")
    ax_b.legend(
        handles=_method_handles(),
        loc="lower right",
        frameon=False,
        fontsize=5.5,
        handlelength=1.3,
        borderaxespad=0.4,
        labelspacing=0.25,
    )
    ax_b.grid(True, which="major", axis="y", color="0.92", linewidth=0.35)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # (c) 4×4 Heisenberg one Trotter step
    for method in METHODS:
        style = METHOD_STYLES[method]
        pts = sorted(
            [r for r in one_step if r["method"] == method],
            key=lambda r: int(r["chi_max"]),
        )
        if not pts:
            continue
        xs = np.array([int(p["chi_max"]) for p in pts], dtype=float)
        ys = np.array([max(_display_y(float(p["infidelity"])), y_lo_shared) for p in pts], dtype=float)
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
    ax_c.set_xscale("log", base=2)
    ax_c.set_yscale("log")
    ax_c.set_xticks(list(ONE_STEP_MAJOR_TICKS))
    ax_c.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_c.set_xlim(ONE_STEP_CHI[0] / 1.2, ONE_STEP_CHI[-1] * 1.2)
    ax_c.set_ylim(y_lo_shared, y_hi_shared)
    ax_c.set_xlabel(r"$\chi_{\max}$")
    ax_c.set_ylabel(r"Infidelity ($1{-}F$)")
    ax_c.set_title("4×4 Heisenberg, one Trotter step")
    ax_c.grid(True, which="major", axis="y", color="0.92", linewidth=0.35)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.tick_params(labelleft=True)

    for ax, lab in zip((ax_a, ax_b, ax_c), ("(a)", "(b)", "(c)"), strict=True):
        ax.text(0.03, 0.97, lab, transform=ax.transAxes, fontsize=8.0, fontweight="bold", va="top")

    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.15, top=0.88, wspace=0.38)
    return fig


def write_markdown(one_step: list[dict[str, Any]], horizons: list[dict[str, Any]]) -> None:
    by_chi: dict[int, dict[str, float]] = {chi: {} for chi in ONE_STEP_CHI}
    for r in one_step:
        by_chi[int(r["chi_max"])][r["method"]] = float(r["infidelity"])

    caption = (
        "Circuit accuracy and method regimes under a fixed bond-dimension cap on 4×4 lattices. "
        "(a) Reliability horizon nε (Trotter steps to ε) for the 4×4 TFIM versus χmax. "
        f"(b) Infidelity versus Trotter step (Δt={DT:g}) for the 4×4 TFIM at χmax=32. "
        "(c) Infidelity after one Trotter step of the 4×4 Heisenberg circuit across χmax, "
        "showing the crossover between TDVP at constrained χ and explicit gate application "
        "at larger χ. Heisenberg TDVP uses four substeps."
    )

    lines = [
        "# Fixed-resource Results placeholder figure",
        "",
        "## Figure caption",
        "",
        caption,
        "",
        "## Layout",
        "Full-width 1×3 figure. Panels (a) and (c) use dense χ scans; (b) uses the validated χ=32 TFIM trajectory.",
        "",
        "### (a) 4×4 TFIM horizon",
        f"- Dense χ ∈ {list(HORIZON_CHI)} from `{TFIM_DENSE_CSV.name}`",
        f"- Vertical axis: $n_\\varepsilon$ (Trotter steps; $T_\\varepsilon/\\Delta t$, $\\Delta t={DT:g}$)",
        "",
    ]
    for method in METHODS:
        vals = [
            f"χ={h['chi_max']}:{h['n_eps']}"
            + ("↑" if int(h.get("right_censored", 0)) else "")
            for h in sorted(horizons, key=lambda x: x["chi_max"])
            if h["method"] == method
        ]
        lines.append(f"- {METHOD_LABELS[method]}: {', '.join(vals)}")

    lines += [
        "",
        "### (b) 4×4 TFIM trajectory",
        f"- Infidelity vs Trotter step (Δt={DT:g}) at χmax=32 (`trajectories.csv`)",
        "",
        "### (c) 4×4 Heisenberg, one Trotter step",
        f"- Dense χ ∈ {list(ONE_STEP_CHI)}; TDVP n={CONVERGED_TDVP_SUBSTEPS}",
        f"- Source: `{HEIS_DENSE_CSV.name}`",
        "",
        "| χmax | TDVP | TEBD+SWAP | MPO zip-up |",
        "|---:|---:|---:|---:|",
    ]
    for chi in ONE_STEP_CHI:
        vals = by_chi.get(chi, {})
        if not vals:
            continue
        lines.append(
            f"| {chi} | {vals.get('hybrid_tdvp', float('nan')):.4e} | "
            f"{vals.get('tebd_swap', float('nan')):.4e} | "
            f"{vals.get('mpo_zipup', float('nan')):.4e} |"
        )

    lines += [
        "",
        "## Outputs",
        f"- `{PLACEHOLDER_PDF.name}` / `{PLACEHOLDER_PNG.name}`",
        f"- `{PLACEHOLDER_MD.name}`",
        "",
        "## Notes",
        "- Incomplete TFIM χ=128 not resumed; no variational MPO.",
    ]
    PLACEHOLDER_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {PLACEHOLDER_MD}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build fixed-resource placeholder Results figure.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    out = args.output_dir.resolve()
    global TRAJ_CSV, PLACEHOLDER_PDF, PLACEHOLDER_PNG, PLACEHOLDER_MD
    global TFIM_DENSE_CSV, HEIS_DENSE_CSV
    TRAJ_CSV = out / "trajectories.csv"
    TFIM_DENSE_CSV = out / "tfim_horizon_dense.csv"
    HEIS_DENSE_CSV = out / "heisenberg_one_step_dense.csv"
    PLACEHOLDER_PDF = out / "fixed_resources_placeholder.pdf"
    PLACEHOLDER_PNG = out / "fixed_resources_placeholder.png"
    PLACEHOLDER_MD = out / "fixed_resources_placeholder.md"

    if not TRAJ_CSV.exists():
        raise SystemExit(f"Missing {TRAJ_CSV}")

    one_step = load_heisenberg_one_step()
    traj_rows = _load_csv(TRAJ_CSV)
    horizons = load_tfim_horizons()
    write_markdown(one_step, horizons)
    fig = plot_placeholder(traj_rows, horizons, one_step)
    fig.savefig(PLACEHOLDER_PDF)
    fig.savefig(PLACEHOLDER_PNG, dpi=DPI)
    plt.close(fig)
    print(f"Wrote {PLACEHOLDER_PDF}")
    print(f"Wrote {PLACEHOLDER_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
