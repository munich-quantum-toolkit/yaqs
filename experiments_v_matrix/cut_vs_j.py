#!/usr/bin/env python3
"""Memory entropy :math:`S_V` vs coupling ``J`` for multiple causal cuts.

Fixed setup:
- L=6 sites (Ising chain)
- k=20 instrument slots
- dt=0.1
- g=1.0
- J sweep: 0.0, 0.2, ..., 2.0

For each cut c, this script computes the past-centered singular-value entropy with **linear
branch weighting** fixed at :math:`\\beta=1` (:math:`V_{i,(j,\\alpha)} = w_{ij}[\\rho_{ij}]_\\alpha`,
with cumulative weights from the exact diagnostic rollout). It plots :math:`S_V(J)`. By default
it uses three initial states (random system on site 0, environment |0⟩ on site 1) and averages
scalar metrics across seeds. A gradient colormap encodes cut index, and the midpoint cut c=k/2
is highlighted as a baseline.

Outputs include a **three-panel PRL-style figure**: (1) heatmap of :math:`S_V` (:math:`c` vs :math:`J`, log colors
on :math:`[10^{-5}, 10^0]` with values below :math:`10^{-5}` clipped to the scale floor); (2) :math:`S_V` vs :math:`J`
for **representative cuts** ``PANEL2_FIXED_CUTS``; (3) :math:`S_V` vs :math:`c` for **representative couplings**
``PANEL3_TARGET_JS`` with nearest available :math:`J` from the sweep. Regenerate from ``summary.csv`` via
``--plot-heatmap-only`` (optional ``--summary-csv PATH``).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from common import (
    BETA,
    DT_DEFAULT,
    G_DEFAULT,
    HEATMAP_VMAX,
    HEATMAP_VMIN,
    J_SWEEP,
    K_DEFAULT,
    L_DEFAULT,
    PANEL2_CUTS,
    PANEL3_JS,
    characterize,
    characterizer,
    configure_matplotlib,
    configure_matplotlib_prl,
    initial_states_sys_env0,
    ising_chain,
    load_csv,
    mean_metrics,
    metrics_from,
    sample_cut_probes,
    save_figure,
    sim_params,
    write_csv,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-pasts", type=int, default=32)
    p.add_argument("--n-futures", type=int, default=32)
    p.add_argument(
        "--cuts",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20",
        help='Comma-separated cuts c to run (1..20), e.g. "1,4,7,10,13,16,20".',
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        dest="n_seeds",
        help=(
            "Number of initial states: env fixed to |0>, system (site 0) random for n_seeds>1; "
            "n_seeds=1 uses |0...0>. Scalar metrics are averaged across seeds."
        ),
    )
    p.add_argument(
        "--j-values",
        type=str,
        default="",
        help="Optional comma-separated J sweep (default: full 0..2 step 0.05).",
    )
    p.add_argument("--out-dir", type=Path, default=Path("benchmark_entropy_vs_j_by_cut_results"))
    p.add_argument("--parallel", action="store_true", default=True)
    p.add_argument("--no-parallel", dest="parallel", action="store_false")
    p.add_argument(
        "--unitary-ensemble",
        type=str,
        default="haar",
        choices=("haar", "clifford"),
        help="Non-break random unitary ensemble for unitary_break_mp probes.",
    )
    p.add_argument(
        "--plot-heatmap-only",
        action="store_true",
        dest="plot_heatmap_only",
        help="Only build the cut×J entropy heatmap from --summary-csv (no simulations).",
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Input summary for --plot-heatmap-only (default: <out-dir>/summary.csv).",
    )
    p.add_argument(
        "--no-center",
        action="store_true",
        help="Skip past-row centering when assembling the response matrix.",
    )
    return p.parse_args()


def _parse_int_list(spec: str) -> list[int]:
    vals = [int(tok.strip()) for tok in spec.split(",") if tok.strip()]
    if not vals:
        raise ValueError("cuts must contain at least one integer.")
    uniq = sorted(set(vals))
    for c in uniq:
        if not (1 <= c <= K_DEFAULT):
            raise ValueError(f"cut must satisfy 1 <= cut <= {K_DEFAULT}, got {c}.")
    return uniq


def _plot_entropy_vs_j(rows: list[dict[str, float | int]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    if not rows:
        return
    cuts = sorted({int(r["cut"]) for r in rows})
    j_vals = sorted({float(r["J"]) for r in rows})
    c_mid = K_DEFAULT // 2

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=min(cuts), vmax=max(cuts))

    fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.0), constrained_layout=True)

    for c in cuts:
        sub = sorted((r for r in rows if int(r["cut"]) == c), key=lambda r: float(r["J"]))
        xs = [float(r["J"]) for r in sub]
        ys = [float(r["entropy"]) for r in sub]
        color = cmap(norm(c))
        lw = 2.8 if c == c_mid else 1.4
        alpha = 1.0 if c == c_mid else 0.85
        label = f"c={c} (k/2 baseline)" if c == c_mid else None
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha, label=label, zorder=3 if c == c_mid else 2)

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

    save_figure(fig, out_dir / "fig_entropy_vs_j_by_cut")


def plot_entropy_heatmap_cut_vs_j(
    rows: list[dict[str, str | float | int]],
    out_stem: Path,
) -> None:
    """Hierarchy layout: dominant heatmap with two supporting cross-sections."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
    from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullLocator

    if not rows:
        return

    def _truncate_cmap(name: str, lo: float = 0.10, hi: float = 0.85, n: int = 256) -> LinearSegmentedColormap:
        base = plt.get_cmap(name)
        return LinearSegmentedColormap.from_list(
            f"{name}_trunc_{lo:.2f}_{hi:.2f}",
            base(np.linspace(lo, hi, n)),
        )

    cuts = sorted({int(float(r["cut"])) for r in rows})
    j_vals = sorted({float(r["J"]) for r in rows})
    z = np.full((len(cuts), len(j_vals)), np.nan, dtype=np.float64)
    for r in rows:
        ci = cuts.index(int(float(r["cut"])))
        ji = j_vals.index(float(r["J"]))
        z[ci, ji] = float(r["entropy"])

    j_arr = np.asarray(j_vals, dtype=np.float64)
    if j_arr.size >= 2:
        dj = float(np.median(np.diff(j_arr)))
        j_edges = np.concatenate(
            [[j_arr[0] - dj / 2], (j_arr[:-1] + j_arr[1:]) / 2, [j_arr[-1] + dj / 2]]
        )
    else:
        j_edges = np.array([j_arr[0] - 0.1, j_arr[0] + 0.1])

    c_arr = np.asarray(cuts, dtype=np.float64)
    if c_arr.size >= 2:
        dc = float(np.median(np.diff(c_arr)))
        c_edges = np.concatenate(
            [[c_arr[0] - dc / 2], (c_arr[:-1] + c_arr[1:]) / 2, [c_arr[-1] + dc / 2]]
        )
    else:
        c_edges = np.array([c_arr[0] - 0.5, c_arr[0] + 0.5])

    def nearest_j(target: float) -> float:
        return float(j_arr[int(np.argmin(np.abs(j_arr - float(target))))])

    configure_matplotlib_prl()
    fig = plt.figure(figsize=(8.2, 4.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.06, hspace=0.10)
    ax0 = fig.add_subplot(gs[:, 0])  # dominant heatmap
    ax1 = fig.add_subplot(gs[0, 1])  # S vs J
    ax2 = fig.add_subplot(gs[1, 1])  # S vs c
    ax0.set_facecolor("white")
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")

    # Show exact zeros explicitly in black via colormap "under" color.
    # Positive values remain on the log color scale [HEATMAP_COLOR_VMIN, HEATMAP_COLOR_VMAX].
    z_plot = np.where(
        np.isfinite(z),
        np.where(z <= 0.0, HEATMAP_VMIN * 0.1, np.maximum(z, HEATMAP_VMIN)),
        np.nan,
    )
    z_mesh = np.ma.masked_invalid(np.transpose(z_plot))

    norm = LogNorm(vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX)
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
    cbar.ax.tick_params(length=3.0, width=0.7, labelsize=13)
    cbar.outline.set_linewidth(0.8)

    ax0.set_xlim(c_edges[0], c_edges[-1])
    ax0.set_ylim(j_edges[0], j_edges[-1])

    ax0.set_xticks([1, 5, 10, 15, 20])
    ax0.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax0.grid(False)

    for spine in ax0.spines.values():
        spine.set_linewidth(0.9)
    ax0.tick_params(which="major", direction="in", top=True, right=True, length=4.2, width=0.8, labelsize=15)
    ax0.xaxis.label.set_size(20)
    ax0.yaxis.label.set_size(20)

    # Representative slices.
    panel2_pref = [1, 5, 10, 15, 20]
    panel2_cuts = [c for c in panel2_pref if c in cuts]
    if len(panel2_cuts) < 4:
        panel2_all = [int(c) for c in cuts if 1 <= int(c) <= 20]
        if panel2_all:
            idx = np.linspace(0, len(panel2_all) - 1, min(5, len(panel2_all))).round().astype(int)
            panel2_cuts = [panel2_all[i] for i in idx]
    panel3_js = [nearest_j(v) for v in PANEL3_JS]

    # Heatmap slice guides.
    for c_sel in panel2_cuts:
        ax0.axvline(float(c_sel), color="white", lw=0.55, ls="--", alpha=0.10, zorder=3)
    guide_cmap = plt.get_cmap("Reds")
    guide_norm = Normalize(vmin=0.0, vmax=2.0)
    for j_sel in panel3_js:
        ax0.axhline(float(j_sel), color=guide_cmap(guide_norm(j_sel)), lw=0.55, ls="--", alpha=0.12, zorder=3)

    # Panel (b): S_V vs J for all cuts (continuous colormap).
    panel2_vals = sorted(panel2_cuts)
    panel2_cmap = _truncate_cmap("Blues", 0.35, 0.95)
    panel2_norm = Normalize(vmin=1.0, vmax=20.0)
    panel2_color_map = {c: panel2_cmap(panel2_norm(float(c))) for c in panel2_vals}
    for c_sel in panel2_vals:
        sub = sorted((r for r in rows if int(float(r["cut"])) == c_sel), key=lambda r: float(r["J"]))
        if not sub:
            continue
        xs = [float(r["J"]) for r in sub]
        ys = [max(float(r["entropy"]), HEATMAP_VMIN) for r in sub]
        ax1.semilogy(
            xs,
            ys,
            color=panel2_color_map[c_sel],
            lw=2.0,
            marker="o",
            ms=3.8,
            markeredgewidth=0.0,
            alpha=0.95,
            label=rf"$c={c_sel}$",
        )
    ax1.set_xlabel(r"Coupling $J$")
    ax1.set_ylabel(r"$S_V$")
    ax1.tick_params(which="major", direction="in", top=True, right=True, length=3.8, width=0.8, labelsize=12)
    ax1.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax1.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax1.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax1.yaxis.set_minor_locator(NullLocator())
    ax1.grid(True, which="major", axis="y", alpha=0.12, linewidth=0.45)
    for s in ax1.spines.values():
        s.set_linewidth(0.9)
    ax1.legend(frameon=False, loc="lower right", fontsize=7.5, handlelength=1.5, borderaxespad=0.2, labelspacing=0.2)
    ax1.xaxis.label.set_size(17)
    ax1.yaxis.label.set_size(17)

    # Panel (c): S_V vs c for all couplings (raw discrete points only).
    xs_c = list(cuts)
    panel3_vals = sorted(panel3_js)
    panel3_cmap = _truncate_cmap("Reds", 0.30, 0.95)
    panel3_norm = Normalize(vmin=0.0, vmax=2.0)

    def _marker_for_j(jv: float) -> str:
        if abs(jv - 0.4) < 1e-9:
            return "o"
        if abs(jv - 1.0) < 1e-9:
            return "s"
        if abs(jv - 2.0) < 1e-9:
            return "^"
        return "o"

    for j_use in panel3_vals:
        ji = j_vals.index(j_use)
        ys_s = np.asarray([max(float(z[ci, ji]), HEATMAP_VMIN) for ci in range(len(cuts))], dtype=np.float64)
        col = panel3_cmap(panel3_norm(j_use))
        ax2.semilogy(
            xs_c,
            ys_s,
            ls="-",
            lw=1.9,
            marker=_marker_for_j(j_use),
            ms=5.0,
            markeredgewidth=0.0,
            alpha=0.92,
            color=col,
            label=rf"$J={j_use:g}$",
        )
    ax2.set_xlabel(r"Causal cut $c$")
    ax2.set_ylabel(r"$S_V$")
    ax2.tick_params(which="major", direction="in", top=True, right=True, length=3.8, width=0.8, labelsize=12)
    ax2.set_xticks([4, 8, 12, 16, 20])
    ax2.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax2.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax2.yaxis.set_minor_locator(NullLocator())
    ax2.grid(True, which="major", axis="y", alpha=0.12, linewidth=0.45)
    for s in ax2.spines.values():
        s.set_linewidth(0.9)
    ax2.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        frameon=False,
        fontsize=8.0,
        handlelength=1.25,
        columnspacing=0.9,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    ax2.xaxis.label.set_size(17)
    ax2.yaxis.label.set_size(17)

    # Match y-limits across side panels for fair comparison.
    y_floor = 1e-3
    y_vals_side: list[float] = []
    for c_sel in panel2_cuts:
        for r in rows:
            if int(float(r["cut"])) == c_sel:
                y_vals_side.append(max(float(r["entropy"]), HEATMAP_VMIN))
    for j_use in panel3_js:
        ji = j_vals.index(j_use)
        y_vals_side.extend([max(float(z[ci, ji]), HEATMAP_VMIN) for ci in range(len(cuts))])
    if y_vals_side:
        y_hi = min(1.0, max(y_floor * 1.2, float(np.nanmax(y_vals_side)) * 1.25))
    else:
        y_hi = 1.0
    ax1.set_ylim(y_floor, y_hi)
    ax2.set_ylim(y_floor, y_hi)

    for ax, tag in ((ax0, "(a)"), (ax1, "(b)"), (ax2, "(c)")):
        ax.text(
            0.04,
            0.955,
            tag,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=13,
            fontweight="bold",
        )

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

    if bool(args.plot_heatmap_only):
        csv_path = Path(args.summary_csv) if args.summary_csv is not None else out_dir / "summary.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"summary CSV not found: {csv_path}")
        rows_raw = load_csv(csv_path)
        plot_entropy_heatmap_cut_vs_j(rows_raw, out_dir / "fig_entropy_heatmap_cut_vs_J")
        print(f"Wrote heatmap: {out_dir / 'fig_entropy_heatmap_cut_vs_J.pdf'}", flush=True)
        return

    cuts = _parse_int_list(str(args.cuts))
    n_seeds = int(args.n_seeds)
    j_values = (
        [float(tok.strip()) for tok in str(args.j_values).split(",") if tok.strip()]
        if str(args.j_values).strip()
        else list(J_SWEEP)
    )
    if n_seeds < 1:
        raise ValueError("n-seeds must be >= 1.")
    init_rng = np.random.default_rng(int(args.seed) + 77_777)
    initial_list = initial_states_sys_env0(length=L_DEFAULT, n_seeds=n_seeds, rng=init_rng)
    np.save(out_dir / "initial_states.npy", np.stack(initial_list, axis=0))

    mc = characterizer(parallel=bool(args.parallel))
    params = sim_params(dt=DT_DEFAULT)

    center = not bool(args.no_center)
    print("=== cut_vs_j: S_V vs J by cut ===", flush=True)
    print(
        f"L={L_DEFAULT}, k={K_DEFAULT}, dt={DT_DEFAULT}, g={G_DEFAULT}, "
        f"N_p={args.n_pasts}, N_f={args.n_futures}, n_seeds={n_seeds}, cuts={cuts}, "
        f"beta={BETA}, center={center}",
        flush=True,
    )

    rows: list[dict[str, float | int]] = []

    for cut in cuts:
        probe_set = sample_cut_probes(
            cut=int(cut),
            k=K_DEFAULT,
            n_pasts=int(args.n_pasts),
            n_futures=int(args.n_futures),
            seed=int(args.seed),
            style=str(args.unitary_ensemble),
        )
        for jv in j_values:
            ham = ising_chain(length=L_DEFAULT, j=float(jv), g=G_DEFAULT)
            per_seed = []
            for psi0 in initial_list:
                result = characterize(
                    mc,
                    ham,
                    params,
                    k=K_DEFAULT,
                    cut=int(cut),
                    n_pasts=int(args.n_pasts),
                    n_futures=int(args.n_futures),
                    probe_set=probe_set,
                    initial_psi=psi0,
                    style=str(args.unitary_ensemble),
                    center=center,
                )
                per_seed.append(metrics_from(result, int(cut)))
            agg = mean_metrics(per_seed)
            row: dict[str, float | int] = {
                "L": L_DEFAULT,
                "k": K_DEFAULT,
                "dt": DT_DEFAULT,
                "g": G_DEFAULT,
                "cut": int(cut),
                "J": float(jv),
                "n_pasts": int(args.n_pasts),
                "n_futures": int(args.n_futures),
                "n_seeds": n_seeds,
                "branch_weight_beta": BETA,
                "centered": int(center),
                "entropy": float(agg["entropy"]),
                "entropy_std": float(agg["entropy_std"]),
                "delta_norm": float(agg["delta_norm"]),
                "rank": int(agg["rank"]),
            }
            rows.append(row)
            print(f"cut={cut:2d}, J={jv:>3.1f}, S_mean={row['entropy']:.6e}", flush=True)

    write_csv(out_dir / "summary.csv", rows)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    configure_matplotlib()
    _plot_entropy_vs_j(rows, out_dir)
    plot_entropy_heatmap_cut_vs_j(rows, out_dir / "fig_entropy_heatmap_cut_vs_J")
    print(f"\nWrote results to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()

