#!/usr/bin/env python3
"""Exact benchmark: middle-cut entropy heatmap ``S_V(L, k)`` at fixed ``J=1``.

This uses the ordinary split-cut probe construction and the same exact weighted-V pipeline as the
standard cut benchmark:
- probes from :func:`sample_split_cut_probes`
- exact rollout + branch weights
- linear weighting ``beta=1``
- past-row centering
- singular-value entropy ``S_V``.
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
    _configure_matplotlib_prl_figure,
    _linear_weighted_metrics,
)
from mqt.yaqs.characterization.process_tensors.diagnostics.probe import ProbeSet, sample_split_cut_probes
from mqt.yaqs.characterization.process_tensors.surrogates.utils import _random_pure_state
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

J_FIXED = 1.0
L_VALUES_DEFAULT = tuple(range(2, 11))
K_VALUES_DEFAULT = tuple(range(2, 101, 2))
REP_LS = (2, 4, 6, 8, 10)
REP_KS = (10, 20, 40, 80, 100)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-pasts", type=int, default=16)
    p.add_argument("--n-futures", type=int, default=16)
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--l-values", type=str, default=",".join(str(v) for v in L_VALUES_DEFAULT))
    p.add_argument("--k-values", type=str, default=",".join(str(v) for v in K_VALUES_DEFAULT))
    p.add_argument("--out-dir", type=Path, default=Path("benchmark_entropy_middlecut_vs_L_k_results"))
    p.add_argument("--parallel", action="store_true", default=True)
    p.add_argument("--no-parallel", dest="parallel", action="store_false")
    p.add_argument("--unitary-ensemble", type=str, default="haar", choices=("haar", "clifford"))
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--summary-csv", type=Path, default=None)
    return p.parse_args()


def _parse_int_list(spec: str, *, name: str, min_v: int, max_v: int) -> list[int]:
    vals = sorted({int(tok.strip()) for tok in spec.split(",") if tok.strip()})
    if not vals:
        raise ValueError(f"expected at least one {name} value")
    for v in vals:
        if v < min_v or v > max_v:
            raise ValueError(f"{name} must satisfy {min_v} <= {name} <= {max_v}, got {v}")
    return vals


def _list_initial_states_sys_env0_for_length(*, L: int, n_seeds: int, rng: np.random.Generator) -> list[np.ndarray]:
    """System = site 0; all environment sites fixed to |0>."""
    if L < 2:
        raise ValueError("L must be >= 2")
    if n_seeds < 1:
        raise ValueError("n_seeds must be >= 1")
    if n_seeds == 1:
        psi = np.zeros(2**L, dtype=np.complex128)
        psi[0] = 1.0 + 0.0j
        return [psi]
    z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    out: list[np.ndarray] = []
    for _ in range(n_seeds):
        psi_sys = _random_pure_state(rng).astype(np.complex128)
        psi = psi_sys
        for _ in range(L - 1):
            psi = np.kron(psi, z)
        out.append(psi / max(float(np.linalg.norm(psi)), 1e-15))
    return out


def _write_summary_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _load_summary_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_entropy_heatmap_middlecut_vs_l_k(rows: list[dict[str, str | float | int]], out_stem: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator

    if not rows:
        return
    _configure_matplotlib_prl_figure()
    ls = sorted({int(float(r["L"])) for r in rows})
    ks = sorted({int(float(r["k"])) for r in rows})
    z = np.full((len(ls), len(ks)), np.nan, dtype=np.float64)
    for r in rows:
        li = ls.index(int(float(r["L"])))
        ki = ks.index(int(float(r["k"])))
        z[li, ki] = float(r["entropy"])

    l_arr = np.asarray(ls, dtype=np.float64)
    k_arr = np.asarray(ks, dtype=np.float64)
    l_edges = (
        np.concatenate([[l_arr[0] - 0.5], 0.5 * (l_arr[:-1] + l_arr[1:]), [l_arr[-1] + 0.5]])
        if len(l_arr) > 1
        else np.array([l_arr[0] - 0.5, l_arr[0] + 0.5])
    )
    if len(k_arr) > 1:
        dk = float(np.median(np.diff(k_arr)))
        k_edges = np.concatenate([[k_arr[0] - dk / 2], 0.5 * (k_arr[:-1] + k_arr[1:]), [k_arr[-1] + dk / 2]])
    else:
        k_edges = np.array([k_arr[0] - 1.0, k_arr[0] + 1.0])

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
    im = ax0.pcolormesh(l_edges, k_edges, z_mesh, cmap=cmap, norm=LogNorm(vmin=HEATMAP_COLOR_VMIN, vmax=HEATMAP_COLOR_VMAX), shading="auto", linewidth=0, edgecolors="none", antialiased=False, rasterized=True)
    ax0.set_xlabel(r"Total sites $L$")
    ax0.set_ylabel(r"Timestep count $k$")
    cbar = fig.colorbar(im, ax=ax0, shrink=0.92, pad=0.012, aspect=18)
    cbar.ax.set_title(r"$S_V$", fontsize=16, pad=3)
    cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    cbar.ax.tick_params(length=3.0, width=0.7, labelsize=13)
    ax0.grid(False)

    rep_ls = [v for v in REP_LS if v in ls]
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
            markeredgewidth=0.0,
            alpha=0.95,
            label=rf"$L={lv}$",
        )
    ax1.set_xlabel(r"Timestep count $k$")
    ax1.set_ylabel(r"$S_V$")
    ax1.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax1.yaxis.set_minor_locator(NullLocator())
    ax1.grid(True, which="major", axis="y", alpha=0.12, linewidth=0.4)
    ax1.legend(frameon=False, fontsize=7.0, handlelength=1.4, borderaxespad=0.2)

    rep_ks = [v for v in REP_KS if v in ks]
    panel3_cmap = plt.get_cmap("Reds")
    panel3_norm = Normalize(vmin=float(min(ks)), vmax=float(max(ks)))
    for kv in rep_ks:
        ki = ks.index(kv)
        ys = np.asarray([max(float(z[li, ki]), HEATMAP_COLOR_VMIN) for li in range(len(ls))], dtype=np.float64)
        ax2.semilogy(
            ls,
            ys,
            color=panel3_cmap(panel3_norm(float(kv))),
            lw=1.8,
            marker="o",
            ms=4.0,
            markeredgewidth=0.0,
            alpha=0.95,
            label=rf"$k={kv}$",
        )
    ax2.set_xlabel(r"Total sites $L$")
    ax2.set_ylabel(r"$S_V$")
    ax2.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax2.yaxis.set_minor_locator(NullLocator())
    ax2.grid(True, which="major", axis="y", alpha=0.12, linewidth=0.4)
    ax2.legend(frameon=False, fontsize=7.0, handlelength=1.4, borderaxespad=0.2)

    y_vals = [max(float(r["entropy"]), HEATMAP_COLOR_VMIN) for r in rows]
    y_hi = min(1.0, max(HEATMAP_COLOR_VMIN * 1.2, float(np.nanmax(y_vals)) * 1.25 if y_vals else 1.0))
    ax1.set_ylim(HEATMAP_COLOR_VMIN, y_hi)
    ax2.set_ylim(HEATMAP_COLOR_VMIN, y_hi)

    for ax, tag in ((ax0, "(a)"), (ax1, "(b)"), (ax2, "(c)")):
        ax.text(0.04, 0.955, tag, transform=ax.transAxes, va="top", ha="left", fontsize=13, fontweight="bold")

    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=600)
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
        plot_entropy_heatmap_middlecut_vs_l_k(rows_raw, out_dir / "fig_entropy_heatmap_middlecut_vs_L_k")
        print(f"Wrote heatmap: {out_dir / 'fig_entropy_heatmap_middlecut_vs_L_k.pdf'}", flush=True)
        return

    l_values = _parse_int_list(str(args.l_values), name="L", min_v=2, max_v=64)
    k_values = _parse_int_list(str(args.k_values), name="k", min_v=2, max_v=200)
    if int(args.n_seeds) < 1:
        raise ValueError("n-seeds must be >= 1")

    probe_set_by_k: dict[int, ProbeSet] = {}
    for k in k_values:
        cut = max(1, int(k) // 2)
        probe_rng = np.random.default_rng(int(args.seed) + 10_000 * int(k))
        probe_set_by_k[int(k)] = sample_split_cut_probes(
            cut=int(cut),
            k=int(k),
            n_pasts=int(args.n_pasts),
            n_futures=int(args.n_futures),
            rng=probe_rng,
            unitary_ensemble=str(args.unitary_ensemble),
        )

    initial_states_by_l: dict[int, list[np.ndarray]] = {}
    for L in l_values:
        init_rng = np.random.default_rng(int(args.seed) + 1_000_000 * int(L))
        initial_states_by_l[int(L)] = _list_initial_states_sys_env0_for_length(
            L=int(L), n_seeds=int(args.n_seeds), rng=init_rng
        )

    if 2 in k_values:
        print("sanity: k=2 -> cut=1", flush=True)
    if 20 in k_values:
        print("sanity: k=20 -> cut=10", flush=True)
    print(
        f"Running middle-cut benchmark: L={l_values}, k={k_values}, J={J_FIXED}, g={G_FIXED}, dt={DT_FIXED}, "
        f"n_pasts={args.n_pasts}, n_futures={args.n_futures}, n_seeds={args.n_seeds}",
        flush=True,
    )

    rows: list[dict[str, float | int]] = []
    for L in l_values:
        initial_list = initial_states_by_l[int(L)]
        op = MPO.ising(length=int(L), J=float(J_FIXED), g=G_FIXED)
        sim_params = AnalogSimParams(dt=DT_FIXED, solver="MCWF", show_progress=False)
        for k in k_values:
            cut = max(1, int(k) // 2)
            probe_set = probe_set_by_k[int(k)]
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
                "L": int(L),
                "env_sites": int(L) - 1,
                "k": int(k),
                "cut": int(cut),
                "J": float(J_FIXED),
                "g": G_FIXED,
                "dt": DT_FIXED,
                "n_pasts": int(args.n_pasts),
                "n_futures": int(args.n_futures),
                "n_seeds": int(args.n_seeds),
                "branch_weight_beta": BRANCH_WEIGHT_BETA,
                "entropy": float(np.mean(entropies)),
                "entropy_std": float(np.std(entropies, ddof=1)) if len(entropies) > 1 else 0.0,
                "delta_norm": float(np.mean(delta_norms)),
                "rank": int(round(float(np.mean(ranks)))),
            }
            rows.append(row)
            print(f"L={L:2d}, k={k:3d}, cut={cut:2d}, S_mean={row['entropy']:.6e}", flush=True)

    _write_summary_csv(out_dir / "summary.csv", rows)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    plot_entropy_heatmap_middlecut_vs_l_k(rows, out_dir / "fig_entropy_heatmap_middlecut_vs_L_k")
    print(f"Wrote results to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()

