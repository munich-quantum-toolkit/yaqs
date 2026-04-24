#!/usr/bin/env python3
"""Finite-size scaling benchmark: entropy vs bath size L with shared probe sets.

Uses the same pipeline as the entropy benchmarks: exact probes, linear branch weights (β=1),
Pauli (x,y,z) embedding, past-row–centered :math:`V`, and singular-value entropy :math:`S_V`
from the **full** spectrum of the centered matrix.

**Probe reuse:** one ``ProbeSet`` per cut ``c`` is sampled from RNG ``(seed + c)`` only and reused
for every ``(L, J)``, so cut profiles are directly comparable across system sizes and couplings.

Outputs:
- ``finite_size_detail.csv``: one row per ``(L, J, c)`` with ``entropy`` (mean over seeds).
- ``finite_size_summary.csv``: one row per ``(L, J)`` with ``peak_entropy``, ``integrated_entropy``,
  ``mean_entropy``, ``peak_cut``.
- ``fig_finite_size_scaling_summary.pdf`` / ``.png``: 2×2 figure (peak, integrated, mean vs ``L``;
  cut profile ``S_V(c)`` at ``--profile-j`` for ``--profile-l``).
- ``fig_peak_cut_vs_L.pdf`` / ``.png``: ``peak_cut`` vs ``L`` at fixed ``--profile-j``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from mqt.yaqs.characterization.process_tensors.diagnostics.exact import evaluate_exact_probe_set_with_diagnostics
from mqt.yaqs.characterization.process_tensors.diagnostics.probe import ProbeSet, sample_split_cut_probes
from mqt.yaqs.characterization.process_tensors.diagnostics.v_matrix_diag import (
    build_weighted_v_matrix,
    center_past_rows,
    prepare_branch_weights,
)
from mqt.yaqs.characterization.process_tensors.surrogates.utils import _random_pure_state
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

K_FIXED = 20
DT_FIXED = 0.1
G_FIXED = 1.0
BRANCH_WEIGHT_BETA = 1.0

L_LIST_DEFAULT = (2, 3, 4, 5, 6, 7, 8, 9, 10)
J_LIST_DEFAULT = (0.5, 1.0, 1.5, 2.0)
PROFILE_L_DEFAULT = (2, 3, 4, 5, 6, 7, 8)
PROFILE_J_DEFAULT = 1.0


def _weighted_centered_entropy(
    *,
    probe_set,
    op: MPO,
    sim_params: AnalogSimParams,
    psi0: np.ndarray,
    parallel: bool,
) -> float:
    """Past-centered :math:`S_V` from full SVD spectrum of centered weighted :math:`V`."""
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
    vc = np.asarray(v_c, dtype=np.float64)
    s = np.linalg.svd(vc, compute_uv=False).astype(np.float64)
    p = s * s
    tot = float(np.sum(p))
    if tot <= 0.0:
        return 0.0
    p = p / tot
    q = np.clip(p, 1e-300, 1.0)
    return float(-np.sum(q * np.log(q)))


def _initial_states(*, L: int, n_seeds: int, rng: np.random.Generator) -> list[np.ndarray]:
    z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    if n_seeds < 1:
        raise ValueError("n_seeds must be >= 1.")
    if n_seeds == 1:
        psi = np.zeros(2**L, dtype=np.complex128)
        psi[0] = 1.0 + 0.0j
        return [psi]
    out: list[np.ndarray] = []
    for _ in range(n_seeds):
        psi_sys = _random_pure_state(rng).astype(np.complex128)
        psi = psi_sys
        for _ in range(L - 1):
            psi = np.kron(psi, z)
        nrm = float(np.linalg.norm(psi))
        out.append(psi / max(nrm, 1e-15))
    return out


def _build_probe_sets(
    *,
    cuts: list[int],
    k: int,
    n_pasts: int,
    n_futures: int,
    seed: int,
) -> dict[int, ProbeSet]:
    """One probe set per cut; RNG depends only on global ``seed`` and ``c``."""
    out: dict[int, object] = {}
    for c in cuts:
        rng = np.random.default_rng(int(seed) + int(c))
        out[int(c)] = sample_split_cut_probes(
            cut=int(c),
            k=int(k),
            n_pasts=int(n_pasts),
            n_futures=int(n_futures),
            rng=rng,
        )
    return out


def _parse_int_tuple(spec: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in spec.split(",") if x.strip())


def _parse_float_tuple(spec: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in spec.split(",") if x.strip())


def _write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _configure_matplotlib() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.5,
        }
    )


def _col_j(j: float) -> str:
    if abs(j - 0.5) < 1e-9:
        return "#0072B2"
    if abs(j - 1.0) < 1e-9:
        return "#D55E00"
    if abs(j - 2.0) < 1e-9:
        return "#009E73"
    return "#333333"


def plot_finite_size_summary_figure(
    summary_rows: list[dict[str, str | float | int]],
    detail_rows: list[dict[str, str | float | int]],
    *,
    out_stem: Path,
    profile_j: float,
    profile_ls: tuple[int, ...],
) -> None:
    import matplotlib.pyplot as plt

    _configure_matplotlib()
    js = sorted({float(r["J"]) for r in summary_rows})
    ls_all = sorted({int(float(r["L"])) for r in summary_rows})

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.2), constrained_layout=True)
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    def _plot_metric_vs_l(ax, key: str, ylabel: str, tag: str) -> None:
        for jv in js:
            xs: list[int] = []
            ys: list[float] = []
            for L in ls_all:
                sub = [r for r in summary_rows if int(float(r["L"])) == L and abs(float(r["J"]) - jv) < 1e-9]
                if not sub:
                    continue
                xs.append(L-1)
                ys.append(float(sub[0][key]))
            if xs:
                pts = sorted(zip(xs, ys), key=lambda p: p[0])
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    marker="o",
                    lw=1.2,
                    color=_col_j(jv),
                    label=rf"$J={jv:g}$",
                )
        ax.set_xlabel(r"$Environment Size$")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, loc="best", fontsize=7)
        ax.grid(True, alpha=0.12, linewidth=0.4)
        ax.text(0.04, 0.95, tag, transform=ax.transAxes, va="top", ha="left", fontsize=9, fontweight="semibold")

    _plot_metric_vs_l(ax_a, "peak_entropy", r"$\max_c S_V(c)$", "(a)")
    _plot_metric_vs_l(ax_b, "integrated_entropy", r"$\sum_c S_V(c)$", "(b)")
    _plot_metric_vs_l(ax_c, "mean_entropy", r"$\frac{1}{k}\sum_c S_V(c)$", "(c)")

    j_key = profile_j
    for L in profile_ls:
        sub = [r for r in detail_rows if int(float(r["L"])) == L and abs(float(r["J"]) - j_key) < 1e-9]
        sub = sorted(sub, key=lambda r: int(float(r["c"])))
        if not sub:
            continue
        cc = [int(float(r["c"])) for r in sub]
        sv = [float(r["entropy"]) for r in sub]
        ax_d.plot(cc, sv, marker="o", lw=1.1, ms=2.8, label=rf"$L={L}$")
    ax_d.set_xlabel(r"Cut $c$")
    ax_d.set_ylabel(r"$S_V(c)$")
    ax_d.legend(title=rf"$J={j_key:g}$", frameon=False, loc="best", fontsize=6.5, ncol=2)
    ax_d.grid(True, alpha=0.12, linewidth=0.4)
    ax_d.text(0.04, 0.95, "(d)", transform=ax_d.transAxes, va="top", ha="left", fontsize=9, fontweight="semibold")

    fig.savefig(out_stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_peak_cut_vs_l(
    summary_rows: list[dict[str, str | float | int]],
    *,
    out_stem: Path,
    profile_j: float,
) -> None:
    import matplotlib.pyplot as plt

    _configure_matplotlib()
    sub = [r for r in summary_rows if abs(float(r["J"]) - profile_j) < 1e-9]
    sub = sorted(sub, key=lambda r: int(float(r["L"])))
    if not sub:
        return
    Ls = [int(float(r["L"])) for r in sub]
    pcs = [int(float(r["peak_cut"])) for r in sub]
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.6), constrained_layout=True)
    ax.plot(Ls, pcs, marker="s", lw=1.2, color="#333333")
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$c_{\max}$")
    ax.set_title(rf"$c_{{\max}}(L)$ at $J={profile_j:g}$", fontsize=10)
    ax.grid(True, alpha=0.12, linewidth=0.4)
    fig.savefig(out_stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_integrated_entropy_vs_l(
    summary_rows: list[dict[str, str | float | int]],
    *,
    out_stem: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    _configure_matplotlib()
    js = sorted({float(r["J"]) for r in summary_rows})
    ls_all = sorted({int(float(r["L"])) for r in summary_rows})

    def _truncate_cmap(name: str, lo: float, hi: float, n: int = 256) -> LinearSegmentedColormap:
        base = plt.get_cmap(name)
        return LinearSegmentedColormap.from_list(f"{name}_trunc_{lo:.2f}_{hi:.2f}", base(np.linspace(lo, hi, n)))

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.55), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    j_norm = Normalize(vmin=0.0, vmax=2.0)
    j_cmap = _truncate_cmap("Reds", 0.30, 0.95)

    for jv in js:
        xs: list[int] = []
        ys: list[float] = []
        for L in ls_all:
            sub = [r for r in summary_rows if int(float(r["L"])) == L and abs(float(r["J"]) - jv) < 1e-9]
            if not sub:
                continue
            xs.append(L-1)
            ys.append(float(sub[0]["integrated_entropy"]))
        if xs:
            pts = sorted(zip(xs, ys), key=lambda p: p[0])
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                marker="o",
                lw=1.7,
                ms=5.0,
                markeredgewidth=0.0,
                color=j_cmap(j_norm(jv)),
                label=rf"$J={jv:g}$",
            )
    ax.set_xlabel(r"Environmental Sites")
    ax.set_ylabel(r"Integrated Memory $\sum_c S_V(c)$")
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
    ax.legend(frameon=False, loc="upper right", fontsize=7.0, handlelength=1.8, borderaxespad=0.25)
    ax.grid(True, axis="y", alpha=0.10, linewidth=0.35)
    ax.grid(False, axis="x")
    ax.tick_params(direction="in", which="both", top=True, right=True, length=3.5, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.15)
    fig.savefig(out_stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def run_benchmark(args: argparse.Namespace) -> tuple[Path, Path]:
    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    L_list = tuple(_parse_int_tuple(args.l_values))
    J_list = tuple(_parse_float_tuple(args.j_values))
    cuts = list(range(1, int(args.k) + 1))
    n_seeds = int(args.n_seeds)
    k_eff = int(args.k)
    init_rng = np.random.default_rng(int(args.seed) + 91_091)

    probe_sets = _build_probe_sets(
        cuts=cuts,
        k=int(args.k),
        n_pasts=int(args.n_pasts),
        n_futures=int(args.n_futures),
        seed=int(args.seed),
    )

    detail_rows: list[dict[str, float | int]] = []
    summary_rows: list[dict[str, float | int]] = []

    for L in L_list:
        initial_list = _initial_states(L=L, n_seeds=n_seeds, rng=init_rng)
        np.save(out_dir / f"initial_states_L{L}.npy", np.stack(initial_list, axis=0))
        for J in J_list:
            op = MPO.ising(length=int(L), J=float(J), g=float(args.g))
            sim_params = AnalogSimParams(dt=float(args.dt), solver="MCWF", show_progress=False)
            mean_ent: dict[int, float] = {}
            for c in cuts:
                probe_set = probe_sets[int(c)]
                ents: list[float] = []
                for psi0 in initial_list:
                    e = _weighted_centered_entropy(
                        probe_set=probe_set,
                        op=op,
                        sim_params=sim_params,
                        psi0=psi0,
                        parallel=bool(args.parallel),
                    )
                    ents.append(e)
                mean_ent[int(c)] = float(np.mean(ents))

            peak_cut = int(max(mean_ent, key=lambda cc: mean_ent[cc]))
            peak_entropy = float(mean_ent[peak_cut])
            integrated_entropy = float(sum(mean_ent[c] for c in cuts))
            mean_entropy = float(integrated_entropy / k_eff)

            summary_rows.append(
                {
                    "L": int(L),
                    "J": float(J),
                    "peak_entropy": peak_entropy,
                    "integrated_entropy": integrated_entropy,
                    "mean_entropy": mean_entropy,
                    "peak_cut": peak_cut,
                    "k": k_eff,
                    "dt": float(args.dt),
                    "g": float(args.g),
                    "n_pasts": int(args.n_pasts),
                    "n_futures": int(args.n_futures),
                    "n_seeds": n_seeds,
                    "branch_weight_beta": BRANCH_WEIGHT_BETA,
                }
            )

            for c in cuts:
                detail_rows.append(
                    {
                        "L": int(L),
                        "J": float(J),
                        "c": int(c),
                        "entropy": float(mean_ent[int(c)]),
                        "k": k_eff,
                        "dt": float(args.dt),
                        "g": float(args.g),
                        "n_pasts": int(args.n_pasts),
                        "n_futures": int(args.n_futures),
                        "n_seeds": n_seeds,
                        "branch_weight_beta": BRANCH_WEIGHT_BETA,
                    }
                )
            print(
                f"L={L} J={J:g} S_max={peak_entropy:.5f} S_int={integrated_entropy:.4f} S_mean={mean_entropy:.5f} c_max={peak_cut}",
                flush=True,
            )

    detail_path = out_dir / "finite_size_detail.csv"
    summary_path = out_dir / "finite_size_summary.csv"
    _write_csv(detail_path, detail_rows)
    _write_csv(summary_path, summary_rows)
    (out_dir / "finite_size_detail.json").write_text(json.dumps(detail_rows, indent=2))
    (out_dir / "finite_size_summary.json").write_text(json.dumps(summary_rows, indent=2))
    print(f"Wrote {detail_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    return detail_path, summary_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("benchmark_finite_size_scaling_results"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--parallel", action="store_true", default=True)
    p.add_argument("--no-parallel", dest="parallel", action="store_false")
    p.add_argument("--k", type=int, default=K_FIXED, help="Instrument slots (cuts 1..k).")
    p.add_argument("--dt", type=float, default=DT_FIXED)
    p.add_argument("--g", type=float, default=G_FIXED)
    p.add_argument("--l-values", type=str, default=",".join(str(x) for x in L_LIST_DEFAULT))
    p.add_argument("--j-values", type=str, default=",".join(str(x) for x in J_LIST_DEFAULT))
    p.add_argument("--n-pasts", type=int, default=32)
    p.add_argument("--n-futures", type=int, default=32)
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--profile-j", type=float, default=PROFILE_J_DEFAULT)
    p.add_argument("--profile-l", type=str, default=",".join(str(x) for x in PROFILE_L_DEFAULT))
    p.add_argument("--plot-only", action="store_true", help="Only build figures from CSVs.")
    p.add_argument(
        "--plot-integrated-only",
        action="store_true",
        help="Only render integrated entropy vs L from existing summary CSV.",
    )
    p.add_argument(
        "--detail-csv",
        type=Path,
        default=None,
        help="Detail CSV for --plot-only (default: <out-dir>/finite_size_detail.csv).",
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Summary CSV for --plot-only (default: <out-dir>/finite_size_summary.csv).",
    )
    p.add_argument("--benchmark-only", action="store_true", help="Run simulations only, skip figures.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_ls = _parse_int_tuple(args.profile_l)

    if args.plot_only:
        detail_path = args.detail_csv if args.detail_csv is not None else out_dir / "finite_size_detail.csv"
        summary_path = args.summary_csv if args.summary_csv is not None else out_dir / "finite_size_summary.csv"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Summary CSV not found: {summary_path}")
        summary_raw = _load_csv(summary_path)
        if args.plot_integrated_only:
            plot_integrated_entropy_vs_l(summary_raw, out_stem=out_dir / "fig_integrated_entropy_vs_L")
            print(f"Wrote {out_dir / 'fig_integrated_entropy_vs_L.pdf'}", flush=True)
            return
        if not detail_path.is_file():
            raise FileNotFoundError(f"Detail CSV not found: {detail_path}")
        detail_raw = _load_csv(detail_path)
        plot_finite_size_summary_figure(
            summary_raw,
            detail_raw,
            out_stem=out_dir / "fig_finite_size_scaling_summary",
            profile_j=float(args.profile_j),
            profile_ls=profile_ls,
        )
        plot_peak_cut_vs_l(
            summary_raw,
            out_stem=out_dir / "fig_peak_cut_vs_L",
            profile_j=float(args.profile_j),
        )
        print(f"Wrote {out_dir / 'fig_finite_size_scaling_summary.pdf'}", flush=True)
        print(f"Wrote {out_dir / 'fig_peak_cut_vs_L.pdf'}", flush=True)
        return

    _, _ = run_benchmark(args)
    if not args.benchmark_only:
        detail_raw = _load_csv(out_dir / "finite_size_detail.csv")
        summary_raw = _load_csv(out_dir / "finite_size_summary.csv")
        plot_finite_size_summary_figure(
            summary_raw,
            detail_raw,
            out_stem=out_dir / "fig_finite_size_scaling_summary",
            profile_j=float(args.profile_j),
            profile_ls=profile_ls,
        )
        plot_peak_cut_vs_l(
            summary_raw,
            out_stem=out_dir / "fig_peak_cut_vs_L",
            profile_j=float(args.profile_j),
        )
        print(f"Wrote {out_dir / 'fig_finite_size_scaling_summary.pdf'}", flush=True)
        print(f"Wrote {out_dir / 'fig_peak_cut_vs_L.pdf'}", flush=True)
    print(f"Done. Results in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
