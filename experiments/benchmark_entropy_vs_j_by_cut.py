#!/usr/bin/env python3
"""Minimal exact benchmark: plot S_V vs J for multiple cut values.

Fixed setup:
- L=2 sites
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

from mqt.yaqs.characterization.process_tensors.diagnostics.exact import evaluate_exact_probe_set_with_diagnostics
from mqt.yaqs.characterization.process_tensors.diagnostics.probe import (
    analyze_v_matrix,
    sample_split_cut_probes,
)
from mqt.yaqs.characterization.process_tensors.diagnostics.v_matrix_diag import (
    build_weighted_v_matrix,
    center_past_rows,
    prepare_branch_weights,
)
from mqt.yaqs.characterization.process_tensors.surrogates.utils import _random_pure_state
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

L_FIXED = 6
K_FIXED = 20
DT_FIXED = 0.1
G_FIXED = 1.0
J_SWEEP_DEFAULT = [0.2 * i for i in range(11)]  # 0.0 ... 2.0

# Fixed linear weighting (β=1); matches branch-weight β=1 in the exact V-matrix benchmark.
BRANCH_WEIGHT_BETA = 1.0

# Fixed log color limits for the cut×J heatmap (values clip to ends of the scale).
HEATMAP_COLOR_VMIN = 1e-4
HEATMAP_COLOR_VMAX = 1.0

# Panel 2: $S_V$ vs $J$ at these fixed cuts $c$.
PANEL2_FIXED_CUTS: tuple[int, ...] = (10, 15, 19)

# Panel 3: $S_V$ vs $c$ at these representative $J$ targets (nearest $J$ in the summary sweep).
PANEL3_TARGET_JS: tuple[float, ...] = (0.4, 1.0, 2.0)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-pasts", type=int, default=8)
    p.add_argument("--n-futures", type=int, default=8)
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
    p.add_argument("--out-dir", type=Path, default=Path("benchmark_entropy_vs_j_by_cut_results"))
    p.add_argument("--parallel", action="store_true", default=True)
    p.add_argument("--no-parallel", dest="parallel", action="store_false")
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
    return p.parse_args()


def _parse_int_list(spec: str) -> list[int]:
    vals = [int(tok.strip()) for tok in spec.split(",") if tok.strip()]
    if not vals:
        raise ValueError("cuts must contain at least one integer.")
    uniq = sorted(set(vals))
    for c in uniq:
        if not (1 <= c <= K_FIXED):
            raise ValueError(f"cut must satisfy 1 <= cut <= {K_FIXED}, got {c}.")
    return uniq


def _linear_weighted_metrics(
    *,
    probe_set,
    op,
    sim_params,
    psi0: np.ndarray,
    parallel: bool,
) -> dict[str, float | int]:
    """Past-centered S_V entropy, delta_norm, rank with V = w^β ρ, β = BRANCH_WEIGHT_BETA."""
    pauli_xyz_ij, weights_ij, _ = evaluate_exact_probe_set_with_diagnostics(
        probe_set=probe_set,
        operator=op,
        sim_params=sim_params,
        initial_psi=psi0,
        parallel=parallel,
    )
    w_clean, _ = prepare_branch_weights(weights_ij, log_warnings=False)
    v_w = build_weighted_v_matrix(pauli_xyz_ij, w_clean, BRANCH_WEIGHT_BETA)
    v_c = center_past_rows(v_w)
    ana = analyze_v_matrix(v_w, v_c)
    return {
        "entropy": float(ana["entropy"]),
        "delta_norm": float(ana["delta_norm"]),
        "rank": int(ana["rank"]),
    }


def _list_initial_states_sys_env0(*, n_seeds: int, rng: np.random.Generator) -> list[np.ndarray]:
    """System = site 0; environment = site 1 fixed to |0⟩ (L=2)."""
    if n_seeds < 1:
        raise ValueError("n_seeds must be >= 1.")
    if n_seeds == 1:
        psi = np.zeros(2**L_FIXED, dtype=np.complex128)
        psi[0] = 1.0 + 0.0j
        return [psi]
    out: list[np.ndarray] = []
    z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    for _ in range(n_seeds):
        psi_sys = _random_pure_state(rng).astype(np.complex128)
        psi = psi_sys
        for _ in range(L_FIXED - 1):
            psi = np.kron(psi, z)
        nrm = float(np.linalg.norm(psi))
        out.append(psi / max(nrm, 1e-15))
    return out


def _write_summary_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _load_summary_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _configure_matplotlib_pr_heatmap() -> None:
    """APS/Physical Review–style defaults for publication figures (serif, crisp axes)."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "grid.alpha": 0.0,
        }
    )


def _configure_matplotlib_prl_figure() -> None:
    """Tighter Physical Review Letters–oriented rc (single-column width, 8 pt class)."""
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
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.4,
            "ytick.major.width": 0.4,
            "xtick.minor.width": 0.3,
            "ytick.minor.width": 0.3,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "grid.alpha": 0.0,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.0,
        }
    )


def _configure_matplotlib() -> None:
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
            "grid.linewidth": 0.6,
            "font.family": "sans-serif",
            "mathtext.default": "it",
        }
    )


def _savefig_base(fig: object, path_stem: Path) -> None:
    import matplotlib.pyplot as plt

    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_entropy_vs_j(rows: list[dict[str, float | int]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    if not rows:
        return
    cuts = sorted({int(r["cut"]) for r in rows})
    j_vals = sorted({float(r["J"]) for r in rows})
    c_mid = K_FIXED // 2

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

    _savefig_base(fig, out_dir / "fig_entropy_vs_j_by_cut")


def plot_entropy_heatmap_cut_vs_j(
    rows: list[dict[str, str | float | int]],
    out_stem: Path,
) -> None:
    """PRL-oriented three-panel figure with dominant heatmap and compact supporting slices."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import LogLocator, MaxNLocator, NullLocator

    if not rows:
        return

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

    _configure_matplotlib_prl_figure()
    # Two-column PRL width with panel (a) visually dominant.
    fig_w_in = 7.2
    fig_h_in = 2.25
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(fig_w_in, fig_h_in),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.35, 1.0, 1.0], "wspace": 0.06},
    )
    ax0, ax1, ax2 = axes

    z_plot = np.ma.masked_where(~np.isfinite(z) | (z <= 0), z)
    z_mesh = np.ma.transpose(z_plot)

    norm = LogNorm(vmin=HEATMAP_COLOR_VMIN, vmax=HEATMAP_COLOR_VMAX)
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))

    im = ax0.pcolormesh(
        c_edges,
        j_edges,
        z_mesh,
        cmap=cmap,
        norm=norm,
        shading="auto",
        linewidth=0,
        rasterized=True,
    )
    ax0.set_xlabel(r"Causal cut $c$")
    ax0.set_ylabel(r"Coupling $J$")

    cbar = fig.colorbar(im, ax=ax0, shrink=0.9, pad=0.015, aspect=24)
    cbar.ax.set_ylabel(r"$S_V$", rotation=90, labelpad=6)
    cbar.ax.tick_params(length=2.5, width=0.4)
    cbar.outline.set_linewidth(0.4)

    ax0.set_xlim(c_edges[0], c_edges[-1])
    ax0.set_ylim(j_edges[0], j_edges[-1])

    if len(cuts) <= 25:
        ax0.set_xticks(cuts)
    j_tick_step = 0.5
    j_ticks = np.arange(np.ceil(j_arr.min() / j_tick_step) * j_tick_step, j_arr.max() + 0.5 * j_tick_step, j_tick_step)
    if j_ticks.size > 1:
        ax0.set_yticks(j_ticks)

    for spine in ax0.spines.values():
        spine.set_linewidth(0.5)

    panel2_colors = ("#1f4e79", "#3b7ea1", "#4c956c", "#9aa44f", "#6b7280")
    for ic, c_sel in enumerate(PANEL2_FIXED_CUTS):
        if c_sel not in cuts:
            continue
        sub = sorted((r for r in rows if int(float(r["cut"])) == c_sel), key=lambda r: float(r["J"]))
        if not sub:
            continue
        xs = [float(r["J"]) for r in sub]
        ys = [max(float(r["entropy"]), 1e-30) for r in sub]
        ax1.semilogy(
            xs,
            ys,
            color=panel2_colors[ic % len(panel2_colors)],
            lw=1.1,
            label=rf"$c={c_sel}$",
        )
    ax1.set_xlabel(r"Coupling $J$")
    ax1.set_ylabel(r"$S_V$")
    ax1.set_ylim(1e-3, 1)
    ax1.tick_params(which="both", direction="in", top=True, right=True, length=2.5, width=0.35)
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
    ax1.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax1.yaxis.set_minor_locator(NullLocator())
    ax1.grid(True, which="major", axis="y", alpha=0.1, linewidth=0.3)
    for s in ax1.spines.values():
        s.set_linewidth(0.45)
    leg1 = ax1.legend(
        title="Cuts",
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="0.55",
        framealpha=0.85,
        borderpad=0.22,
        handlelength=1.4,
        handletextpad=0.35,
        borderaxespad=0.18,
    )
    leg1.get_title().set_fontsize(5.8)
    leg1.get_frame().set_linewidth(0.35)
    for t in leg1.get_texts():
        t.set_fontsize(6.1)

    xs_c = list(cuts)
    panel3_colors = ("#8c564b", "#1f77b4", "#2a9d8f", "#4c4c4c")
    seen_j: set[float] = set()
    color_i = 0
    for j_tgt in PANEL3_TARGET_JS:
        j_use = nearest_j(j_tgt)
        if j_use in seen_j:
            continue
        seen_j.add(j_use)
        ji = j_vals.index(j_use)
        ys_s = [max(float(z[ci, ji]), 1e-30) for ci in range(len(cuts))]
        lbl = f"$J={j_use:g}$"
        ax2.semilogy(
            xs_c,
            ys_s,
            color=panel3_colors[color_i % len(panel3_colors)],
            lw=1.25,
            label=lbl,
        )
        color_i += 1
    ax2.set_xlabel(r"Causal cut $c$")
    ax2.set_ylabel(r"$S_V$")
    ax2.set_ylim(1e-3, 1)
    ax2.tick_params(which="both", direction="in", top=True, right=True, length=2.5, width=0.35)
    if len(cuts) <= 25:
        ax2.set_xticks(cuts)
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
    ax2.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax2.yaxis.set_minor_locator(NullLocator())
    ax2.grid(True, which="major", axis="y", alpha=0.1, linewidth=0.3)
    for s in ax2.spines.values():
        s.set_linewidth(0.45)
    leg2 = ax2.legend(
        title="Couplings",
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="0.55",
        framealpha=0.85,
        borderpad=0.22,
        handlelength=1.4,
        handletextpad=0.35,
        borderaxespad=0.18,
    )
    leg2.get_title().set_fontsize(5.8)
    leg2.get_frame().set_linewidth(0.35)
    for t in leg2.get_texts():
        t.set_fontsize(6.1)

    for ax, tag in zip(axes, ("(a)", "(b)", "(c)"), strict=True):
        ax.text(
            0.035,
            0.955,
            tag,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7.3,
            fontweight="semibold",
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
        rows_raw = _load_summary_csv(csv_path)
        plot_entropy_heatmap_cut_vs_j(rows_raw, out_dir / "fig_entropy_heatmap_cut_vs_J")
        print(f"Wrote heatmap: {out_dir / 'fig_entropy_heatmap_cut_vs_J.pdf'}", flush=True)
        return

    cuts = _parse_int_list(str(args.cuts))
    n_seeds = int(args.n_seeds)
    if n_seeds < 1:
        raise ValueError("n-seeds must be >= 1.")
    init_rng = np.random.default_rng(int(args.seed) + 77_777)
    initial_list = _list_initial_states_sys_env0(n_seeds=n_seeds, rng=init_rng)
    np.save(out_dir / "initial_states.npy", np.stack(initial_list, axis=0))

    print("=== Benchmark: S_V vs J by cut (linear branch weights, β=1) ===", flush=True)
    print(
        f"L={L_FIXED}, k={K_FIXED}, dt={DT_FIXED}, g={G_FIXED}, "
        f"N_p={args.n_pasts}, N_f={args.n_futures}, n_seeds={n_seeds}, cuts={cuts}, "
        f"branch_weight_beta={BRANCH_WEIGHT_BETA}",
        flush=True,
    )

    rows: list[dict[str, float | int]] = []

    for cut in cuts:
        probe_rng = np.random.default_rng(int(args.seed) + 10_000 * int(cut))
        probe_set = sample_split_cut_probes(
            cut=int(cut),
            k=K_FIXED,
            n_pasts=int(args.n_pasts),
            n_futures=int(args.n_futures),
            rng=probe_rng,
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
            row: dict[str, float | int] = {
                "L": L_FIXED,
                "k": K_FIXED,
                "dt": DT_FIXED,
                "g": G_FIXED,
                "cut": int(cut),
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
            rows.append(row)
            print(
                f"cut={cut:2d}, J={jv:>3.1f}, S_mean={row['entropy']:.6e}",
                flush=True,
            )

    _write_summary_csv(out_dir / "summary.csv", rows)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    _configure_matplotlib()
    _plot_entropy_vs_j(rows, out_dir)
    plot_entropy_heatmap_cut_vs_j(rows, out_dir / "fig_entropy_heatmap_cut_vs_J")
    print(f"\nWrote results to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()

