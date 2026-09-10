#!/usr/bin/env python3
"""Probe-budget convergence: :math:`S_V` vs grid size ``m`` over a dense ``J`` sweep.

Uses :class:`~mqt.yaqs.MemoryCharacterizer` with one full ``m_max`` probe evaluation per draw,
then slices prefix grids to estimate convergence (exact-backend path for prefix reuse).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from common import (
    DT_DEFAULT,
    G_DEFAULT,
    K_DEFAULT,
    L_DEFAULT,
    characterizer,
    configure_matplotlib_prl,
    entropy_from_responses,
    evaluate_weighted_probes,
    initial_states_sys_env0,
    ising_chain,
    load_csv,
    parse_float_list,
    parse_int_list,
    sim_params,
    slice_probe_prefix,
    write_csv,
)
from mqt.yaqs.characterization.memory.operational_memory.samples import sample_probes

M_GRID_DEFAULT = (2,3,4,5,6,8,10,12,16,24,32,48,64)
DENSE_JS_DEFAULT = tuple(round(0.05 * i, 10) for i in range(41))


def run_benchmark(args: argparse.Namespace) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    m_values = sorted(parse_int_list(args.m_values))
    m_max = int(max(m_values))
    conv_js = parse_float_list(args.convergence_js)

    init_rng = np.random.default_rng(int(args.seed) + 77_777)
    initial_list = initial_states_sys_env0(length=L_DEFAULT, n_seeds=int(args.n_seeds), rng=init_rng)
    np.save(out_dir / "initial_states.npy", np.stack(initial_list, axis=0))

    mc = characterizer(parallel=bool(args.parallel))
    params = sim_params(dt=DT_DEFAULT)
    detail_rows: list[dict[str, float | int]] = []
    shape_check_done = False

    for j in conv_js:
        ham = ising_chain(length=L_DEFAULT, j=float(j), g=G_DEFAULT)
        for draw in range(int(args.probe_draws)):
            draw_seed = int(args.seed) + 100_000 * int(args.cut) + 10 * int(round(100 * j)) + draw
            probe_set_max = sample_probes(
                cut=int(args.cut),
                k=K_DEFAULT,
                n_pasts=m_max,
                n_futures=m_max,
                rng=np.random.default_rng(draw_seed),
                intervention_mode="unitary_break_mp",
                unitary_ensemble="haar",
            )
            entropies_by_m: dict[int, list[float]] = {int(m): [] for m in m_values}

            for psi0 in initial_list:
                pauli_xyz_max, weights_max = evaluate_weighted_probes(
                    mc,
                    ham,
                    params,
                    probe_set=probe_set_max,
                    initial_psi=psi0,
                )

                if not shape_check_done:
                    pshape = np.asarray(pauli_xyz_max).shape
                    if len(pshape) < 2 or pshape[0] != m_max or pshape[1] != m_max:
                        raise ValueError(
                            f"unexpected pauli_xyz_max shape={pshape}, expected first two axes = ({m_max}, {m_max})"
                        )
                    shape_check_done = True

                for m in m_values:
                    pauli_sub, w_sub = slice_probe_prefix(pauli_xyz_max, weights_max, int(m))
                    entropies_by_m[int(m)].append(entropy_from_responses(pauli_sub, w_sub))

            for m in m_values:
                detail_rows.append(
                    {
                        "cut": int(args.cut),
                        "J": float(j),
                        "m": int(m),
                        "probe_draw": int(draw),
                        "entropy": float(np.mean(entropies_by_m[int(m)])),
                    }
                )

    summary_rows: list[dict[str, float | int]] = []
    grouped: dict[tuple[int, float], list[dict[str, float | int]]] = {}
    for row in detail_rows:
        grouped.setdefault((int(row["m"]), float(row["J"])), []).append(row)

    for (m, jv), rs in sorted(grouped.items()):
        ent = np.asarray([float(r["entropy"]) for r in rs], dtype=np.float64)
        summary_rows.append(
            {
                "cut": int(args.cut),
                "J": float(jv),
                "m": int(m),
                "entropy_mean": float(np.mean(ent)),
                "entropy_std": float(np.std(ent, ddof=1)) if ent.size > 1 else 0.0,
                "entropy_sem": float(np.std(ent, ddof=1) / np.sqrt(float(ent.size))) if ent.size > 1 else 0.0,
            }
        )

    write_csv(out_dir / "convergence_detail.csv", detail_rows)
    write_csv(out_dir / "convergence_summary.csv", summary_rows)
    return detail_rows, summary_rows


def plot_from_saved(*, summary_rows: list[dict[str, str | float | int]], out_stem: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.ticker import FixedLocator, LogFormatter, LogLocator, NullFormatter, NullLocator

    def _truncate_cmap(name: str, lo: float, hi: float, n: int = 256) -> LinearSegmentedColormap:
        base = plt.get_cmap(name)
        return LinearSegmentedColormap.from_list(f"{name}_trunc_{lo:.2f}_{hi:.2f}", base(np.linspace(lo, hi, n)))

    configure_matplotlib_prl()
    fig, ax = plt.subplots(1, 1, figsize=(3.35, 2.45), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    js = sorted({float(r["J"]) for r in summary_rows})
    norm = Normalize(vmin=0.0, vmax=2.0)
    cmap = _truncate_cmap("Reds", 0.35, 0.95)
    m_vals = sorted({int(float(r["m"])) for r in summary_rows})
    all_js = js
    highlight_targets = [0.4, 1.0, 2.0]
    highlight_js = sorted({float(min(js, key=lambda x: abs(x - t))) for t in highlight_targets}) if js else []

    # Background family: thin lines, no markers, no uncertainty shading.
    for jv in all_js:
        sub = sorted([r for r in summary_rows if abs(float(r["J"]) - jv) < 1e-12], key=lambda r: int(float(r["m"])))
        xs = np.asarray([int(float(r["m"])) for r in sub], dtype=np.float64)
        mu = np.asarray([float(r["entropy_mean"]) for r in sub], dtype=np.float64)
        ys = np.clip(mu, 1e-30, None)
        col = cmap(norm(jv))
        mask = xs >= 3.0
        if np.any(mask):
            ax.semilogy(xs[mask], ys[mask], color=col, lw=0.8, alpha=0.38, zorder=1)

    # Representative highlights: thicker lines + markers.
    for jv in highlight_js:
        sub = sorted([r for r in summary_rows if abs(float(r["J"]) - jv) < 1e-12], key=lambda r: int(float(r["m"])))
        xs = np.asarray([int(float(r["m"])) for r in sub], dtype=np.float64)
        mu = np.asarray([float(r["entropy_mean"]) for r in sub], dtype=np.float64)
        ys = np.clip(mu, 1e-30, None)
        col = cmap(norm(jv))
        mask = xs >= 3.0
        if np.any(mask):
            ax.semilogy(
                xs[mask],
                ys[mask],
                color=col,
                lw=1.7,
                marker="o",
                ms=4.0,
                markeredgewidth=0.0,
                alpha=0.98,
                zorder=3,
            )

    ax.set_xlabel(r"Probe budget $m$ ($N_p=N_f$)")
    ax.set_ylabel(r"$S_V$")
    ax.set_xscale("log", base=2)
    major_ticks = [2, 4, 8, 16, 32, 64]
    major_ticks = [t for t in major_ticks if m_vals and min(m_vals) <= t <= max(m_vals)]
    if major_ticks:
        ax.xaxis.set_major_locator(FixedLocator(major_ticks))
    ax.xaxis.set_major_formatter(LogFormatter(base=2, labelOnlyBase=False))
    minor_ticks = [3, 5, 6, 10, 12, 24, 48]
    minor_ticks = [t for t in minor_ticks if m_vals and min(m_vals) <= t <= max(m_vals)]
    if minor_ticks:
        ax.xaxis.set_minor_locator(FixedLocator(minor_ticks))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(True, which="major", axis="y", alpha=0.08, linewidth=0.3)
    ax.grid(False, axis="x")
    ax.set_ylim(1e-6, 1)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.9)
    cbar.ax.set_title(r"$J$", pad=2)

    fig.savefig(out_stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_error_from_saved(*, summary_rows: list[dict[str, str | float | int]], out_stem: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.ticker import FixedLocator, LogFormatter, LogLocator, NullFormatter, NullLocator

    def _truncate_cmap(name: str, lo: float, hi: float, n: int = 256) -> LinearSegmentedColormap:
        base = plt.get_cmap(name)
        return LinearSegmentedColormap.from_list(f"{name}_trunc_{lo:.2f}_{hi:.2f}", base(np.linspace(lo, hi, n)))

    configure_matplotlib_prl()
    fig, ax = plt.subplots(1, 1, figsize=(3.35, 2.45), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    js = sorted({float(r["J"]) for r in summary_rows})
    norm = Normalize(vmin=0.0, vmax=2.0)
    cmap = _truncate_cmap("Reds", 0.35, 0.95)
    m_vals = sorted({int(float(r["m"])) for r in summary_rows})
    m_ref = max(m_vals) if m_vals else 0
    all_js = js
    highlight_targets = [0.4, 1.0, 2.0]
    highlight_js = sorted({float(min(js, key=lambda x: abs(x - t))) for t in highlight_targets}) if js else []

    # Background family.
    for jv in all_js:
        sub = sorted([r for r in summary_rows if abs(float(r["J"]) - jv) < 1e-12], key=lambda r: int(float(r["m"])))
        if not sub:
            continue
        ref_rows = [r for r in sub if int(float(r["m"])) == int(m_ref)]
        if not ref_rows:
            continue
        ref_mu = float(ref_rows[0]["entropy_mean"])
        xs = np.asarray([int(float(r["m"])) for r in sub], dtype=np.float64)
        mu = np.asarray([float(r["entropy_mean"]) for r in sub], dtype=np.float64)
        denom = max(abs(ref_mu), 1e-30)
        rel_err = np.abs(mu - ref_mu) / denom
        ys = np.clip(rel_err, 1e-30, None)
        col = cmap(norm(jv))
        mask = xs >= 3.0
        if np.any(mask):
            ax.semilogy(xs[mask], ys[mask], color=col, lw=0.8, alpha=0.38, zorder=1)

    # Representative highlights.
    for jv in highlight_js:
        sub = sorted([r for r in summary_rows if abs(float(r["J"]) - jv) < 1e-12], key=lambda r: int(float(r["m"])))
        if not sub:
            continue
        ref_rows = [r for r in sub if int(float(r["m"])) == int(m_ref)]
        if not ref_rows:
            continue
        ref_mu = float(ref_rows[0]["entropy_mean"])
        xs = np.asarray([int(float(r["m"])) for r in sub], dtype=np.float64)
        mu = np.asarray([float(r["entropy_mean"]) for r in sub], dtype=np.float64)
        denom = max(abs(ref_mu), 1e-30)
        ys = np.clip(np.abs(mu - ref_mu) / denom, 1e-30, None)
        col = cmap(norm(jv))
        mask = xs >= 3.0
        if np.any(mask):
            ax.semilogy(
                xs[mask],
                ys[mask],
                color=col,
                lw=1.7,
                marker="o",
                ms=4.0,
                markeredgewidth=0.0,
                alpha=0.98,
                zorder=3,
            )
    ax.set_ylim(1e-2, 1)
    ax.set_xlabel(r"Probe budget $m$ ($N_p=N_f$)")
    ax.set_ylabel(r"$|S_V(m)-S_V(m_{\mathrm{ref}})|/|S_V(m_{\mathrm{ref}})|$")
    ax.set_xscale("log", base=2)
    major_ticks = [2, 4, 8, 16, 32, 64]
    major_ticks = [t for t in major_ticks if m_vals and min(m_vals) <= t <= max(m_vals)]
    if major_ticks:
        ax.xaxis.set_major_locator(FixedLocator(major_ticks))
    ax.xaxis.set_major_formatter(LogFormatter(base=2, labelOnlyBase=False))
    minor_ticks = [3, 5, 6, 10, 12, 24, 48]
    minor_ticks = [t for t in minor_ticks if m_vals and min(m_vals) <= t <= max(m_vals)]
    if minor_ticks:
        ax.xaxis.set_minor_locator(FixedLocator(minor_ticks))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(True, which="major", axis="y", alpha=0.08, linewidth=0.3)
    ax.grid(False, axis="x")
    # Reference slope guide for visual convergence comparison.
    x_ref = np.asarray([m for m in m_vals if m >= 3], dtype=np.float64)
    if x_ref.size:
        y_ref = 1.0 / np.sqrt(x_ref)
        ax.semilogy(x_ref, y_ref, color="black", lw=1.2, alpha=0.9)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.9)
    cbar.ax.set_title(r"$J$", pad=2)

    fig.savefig(out_stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("results/convergence"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--parallel", action="store_true", default=True)
    p.add_argument("--no-parallel", dest="parallel", action="store_false")
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--cut", type=int, default=10)
    p.add_argument("--probe-draws", type=int, default=5)
    p.add_argument("--m-values", type=str, default=",".join(str(v) for v in M_GRID_DEFAULT))
    p.add_argument("--convergence-js", type=str, default=",".join(str(v) for v in DENSE_JS_DEFAULT))
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--benchmark-only", action="store_true")
    p.add_argument("--convergence-summary-csv", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.plot_only):
        summary_csv = args.convergence_summary_csv if args.convergence_summary_csv is not None else out_dir / "convergence_summary.csv"
        rows = load_csv(summary_csv)
        plot_from_saved(summary_rows=rows, out_stem=out_dir / "fig_convergence_sv_vs_m_prl")
        plot_error_from_saved(summary_rows=rows, out_stem=out_dir / "fig_convergence_error_sv_vs_m_prl")
        print(f"Wrote figure: {(out_dir / 'fig_convergence_sv_vs_m_prl').with_suffix('.pdf')}", flush=True)
        print(f"Wrote figure: {(out_dir / 'fig_convergence_error_sv_vs_m_prl').with_suffix('.pdf')}", flush=True)
        return

    _, summary_rows = run_benchmark(args)
    if not bool(args.benchmark_only):
        plot_from_saved(summary_rows=summary_rows, out_stem=out_dir / "fig_convergence_sv_vs_m_prl")
        plot_error_from_saved(summary_rows=summary_rows, out_stem=out_dir / "fig_convergence_error_sv_vs_m_prl")
        print(f"Wrote figure: {(out_dir / 'fig_convergence_sv_vs_m_prl').with_suffix('.pdf')}", flush=True)
        print(f"Wrote figure: {(out_dir / 'fig_convergence_error_sv_vs_m_prl').with_suffix('.pdf')}", flush=True)
    print(f"Wrote tables to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
