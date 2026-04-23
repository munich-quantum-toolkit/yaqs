#!/usr/bin/env python3
"""Modes benchmark: singular-value spectrum and entropy-based effective modes R(J).

R(J) is plotted as :math:`\\exp(S_V)`, where :math:`S_V` is singular-value entropy.
Pipeline: exact probes -> weighted V (beta=1) -> past-row centering -> SVD.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from benchmark_entropy_vs_j_by_cut import (
    BRANCH_WEIGHT_BETA,
    DT_FIXED,
    G_FIXED,
    K_FIXED,
    L_FIXED,
    _configure_matplotlib_prl_figure,
    _list_initial_states_sys_env0,
)
from mqt.yaqs.characterization.process_tensors.diagnostics.exact import evaluate_exact_probe_set_with_diagnostics
from mqt.yaqs.characterization.process_tensors.diagnostics.probe import sample_split_cut_probes
from mqt.yaqs.characterization.process_tensors.diagnostics.v_matrix_diag import (
    build_weighted_v_matrix,
    center_past_rows,
    prepare_branch_weights,
)
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

DENSE_JS_DEFAULT = tuple[float, ...](round(0.05 * i, 10) for i in range(41))
RANK_TOL = 1e-16


def _parse_int_list(spec: str) -> list[int]:
    vals = [int(tok.strip()) for tok in spec.split(",") if tok.strip()]
    if not vals:
        raise ValueError("expected at least one integer")
    return vals


def _parse_float_list(spec: str) -> list[float]:
    vals = [float(tok.strip()) for tok in spec.split(",") if tok.strip()]
    if not vals:
        raise ValueError("expected at least one float")
    return vals


def _write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _aggregate_variable_length(vectors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not vectors:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    max_len = max(int(v.size) for v in vectors)
    arr = np.full((len(vectors), max_len), np.nan, dtype=np.float64)
    for i, v in enumerate(vectors):
        arr[i, : v.size] = v
    return np.nanmean(arr, axis=0), (np.nanstd(arr, axis=0, ddof=1) if len(vectors) > 1 else np.zeros(max_len))


def _weighted_centered_singular_values(*, probe_set, op: MPO, sim_params: AnalogSimParams, psi0: np.ndarray, parallel: bool) -> np.ndarray:
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
    return np.linalg.svd(np.asarray(v_c, dtype=np.float64), compute_uv=False).astype(np.float64)


def run_benchmark(args: argparse.Namespace) -> tuple[list[dict[str, float | int]], dict[str, dict[str, np.ndarray]]]:
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_js = _parse_float_list(args.spectrum_js)
    cuts = _parse_int_list(args.cuts)

    init_rng = np.random.default_rng(int(args.seed) + 77_777)
    initial_list = _list_initial_states_sys_env0(n_seeds=int(args.n_seeds), rng=init_rng)
    sim_params = AnalogSimParams(dt=DT_FIXED, solver="MCWF", show_progress=False)

    spectrum_probs: dict[str, dict[str, np.ndarray]] = {}
    rank_rows: list[dict[str, float | int]] = []

    for cut in cuts:
        print("Cut", cut)
        for jv in spec_js:
            op = MPO.ising(length=L_FIXED, J=float(jv), g=G_FIXED)
            per_draw_vectors: list[np.ndarray] = []
            per_draw_rank: list[float] = []
            per_draw_singulars: list[np.ndarray] = []
            for draw in range(int(args.spectrum_draws)):
                probe_seed = int(args.seed) + 900_000 + 100_000 * int(cut) + 100 * int(round(100 * jv)) + draw
                probe_set = sample_split_cut_probes(
                    cut=int(cut),
                    k=K_FIXED,
                    n_pasts=int(args.m_spectrum),
                    n_futures=int(args.m_spectrum),
                    rng=np.random.default_rng(probe_seed),
                )
                seed_vectors: list[np.ndarray] = []
                seed_ranks: list[float] = []
                seed_singulars: list[np.ndarray] = []
                for psi0 in initial_list:
                    s = _weighted_centered_singular_values(
                        probe_set=probe_set,
                        op=op,
                        sim_params=sim_params,
                        psi0=psi0,
                        parallel=bool(args.parallel),
                    )
                    p = s * s
                    ps = float(np.sum(p))
                    if ps <= 0.0:
                        seed_vectors.append(np.zeros(0, dtype=np.float64))
                    else:
                        seed_vectors.append(p / ps)
                    seed_ranks.append(float(np.sum(s > float(args.rank_tol))))
                    seed_singulars.append(np.asarray(s, dtype=np.float64))
                mean_seed, _ = _aggregate_variable_length(seed_vectors)
                per_draw_vectors.append(mean_seed)
                per_draw_rank.append(float(np.mean(seed_ranks)))
                mean_singular_seed, _ = _aggregate_variable_length(seed_singulars)
                per_draw_singulars.append(mean_singular_seed)

            p_mean, p_std = _aggregate_variable_length(per_draw_vectors)
            s_mean, s_std = _aggregate_variable_length(per_draw_singulars)
            r_effective = int(np.sum(np.asarray(s_mean, dtype=np.float64) > float(args.rank_tol)))
            key = f"c{int(cut)}_J{jv:g}"
            spectrum_probs[key] = {"p_mean": p_mean, "p_std": p_std, "s_mean": s_mean, "s_std": s_std}
            rank_rows.append(
                {
                    "J": float(jv),
                    "R_mean": float(r_effective),
                    "R_std": float(np.std(per_draw_rank, ddof=1)) if len(per_draw_rank) > 1 else 0.0,
                    "rank_tol": float(args.rank_tol),
                    "cut": int(cut),
                    "m_spectrum": int(args.m_spectrum),
                }
            )

    _write_csv(out_dir / "spectrum_rank_summary.csv", rank_rows)
    payload: dict[str, np.ndarray] = {}
    for key, val in spectrum_probs.items():
        payload[f"{key}_mean"] = val["p_mean"]
        payload[f"{key}_std"] = val["p_std"]
        payload[f"{key}_smean"] = val["s_mean"]
        payload[f"{key}_sstd"] = val["s_std"]
    np.savez_compressed(out_dir / "spectrum_probs.npz", **payload)
    return rank_rows, spectrum_probs


def plot_from_saved(
    *,
    rank_rows: list[dict[str, str | float | int]],
    spectrum_probs: dict[str, dict[str, np.ndarray]],
    out_stem: Path,
    rank_tol: float,
    plot_cuts: list[int] | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    def _truncate_cmap(name: str, lo: float, hi: float, n: int = 256) -> LinearSegmentedColormap:
        base = plt.get_cmap(name)
        return LinearSegmentedColormap.from_list(f"{name}_trunc_{lo:.2f}_{hi:.2f}", base(np.linspace(lo, hi, n)))

    _configure_matplotlib_prl_figure()
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.7), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    cuts_all = sorted({int(float(r["cut"])) for r in rank_rows})
    requested_cuts = [1, 5, 10] if plot_cuts is None else plot_cuts
    cuts = [c for c in cuts_all if c in requested_cuts]
    js = sorted({float(r["J"]) for r in rank_rows})

    # Primary panel: all requested cuts.
    cut_cmap = _truncate_cmap("Blues", 0.35, 0.95)
    cut_norm = Normalize(vmin=1.0, vmax=20.0)
    cut_color: dict[int, tuple[float, float, float, float]] = {c: cut_cmap(cut_norm(float(c))) for c in cuts}
    for cut in cuts:
        mu_vals: list[float] = []
        x_vals: list[float] = []
        for jv in js:
            key = f"c{cut}_J{jv:g}"
            if key not in spectrum_probs:
                continue
            p = np.asarray(spectrum_probs[key]["p_mean"], dtype=np.float64)
            ps = float(np.sum(p))
            if ps <= 0.0:
                mu_vals.append(1.0)
            else:
                q = np.clip(p / ps, 1e-30, 1.0)
                sv = float(-np.sum(q * np.log(q)))
                mu_vals.append(float(np.exp(sv)))
            x_vals.append(float(jv))
        if not x_vals:
            continue
        ax.plot(
            np.asarray(x_vals, dtype=np.float64),
            np.asarray(mu_vals, dtype=np.float64),
            color=cut_color[cut],
            lw=1.7,
            marker="o",
            ms=4.0,
            alpha=0.95,
            label=rf"$c={cut}$",
        )
    ax.set_xlabel(r"$J$")
    ax.set_ylabel(r"$R=\exp(S_V)$")
    ax.set_xlim(0.0, float(max(js)) if js else 2.0)
    y_all: list[float] = []
    for cut in cuts:
        for jv in js:
            key = f"c{cut}_J{jv:g}"
            if key not in spectrum_probs:
                continue
            p = np.asarray(spectrum_probs[key]["p_mean"], dtype=np.float64)
            ps = float(np.sum(p))
            if ps <= 0.0:
                y_all.append(1.0)
            else:
                q = np.clip(p / ps, 1e-30, 1.0)
                y_all.append(float(np.exp(-np.sum(q * np.log(q)))))
    y_hi = float(max(y_all)) if y_all else 2.0
    ax.set_ylim(0.98, max(1.05, y_hi * 1.04))
    ax.grid(True, axis="y", alpha=0.06, linewidth=0.3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    ax.tick_params(direction="in", which="both", top=True, right=True, length=3.2, width=0.7)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.90, 0.995),
        frameon=False,
        fontsize=6.6,
        handlelength=1.5,
        borderaxespad=0.2,
        labelspacing=0.2,
    )

    # Inset: selected spectra for c=10 only.
    inset = ax.inset_axes((0.16, 0.56, 0.36, 0.34))
    inset.set_facecolor("white")
    inset_cut = 10
    j_inset = [0.5, 1.0, 1.5, 2.0]
    available = [jv for jv in j_inset if f"c{inset_cut}_J{jv:g}" in spectrum_probs]
    j_norm = Normalize(vmin=0.0, vmax=2.0)
    j_cmap = _truncate_cmap("Reds", 0.30, 0.95)
    inset_colors: dict[float, tuple[float, float, float, float]] = {}
    for jv in available:
        p = np.asarray(spectrum_probs[f"c{inset_cut}_J{jv:g}"]["p_mean"], dtype=np.float64)
        if p.size == 0:
            continue
        n = np.arange(1, p.size + 1, dtype=np.float64)
        col = j_cmap(j_norm(jv))
        inset_colors[jv] = col
        inset.plot(n, np.clip(p, 1e-30, None), color=col, lw=0.9, alpha=0.95, label=rf"$J={jv:g}$")
    inset.set_yscale("log")
    inset.set_ylim(1e-16, 1.0)
    inset.set_xlim(1, 30)
    inset.set_title(r"$c=10$", fontsize=6.6, pad=1.5)
    inset.set_xlabel(r"$n$", labelpad=1)
    inset.set_ylabel(r"$p_n$", labelpad=1)
    inset.tick_params(axis="both", which="major", labelsize=6)
    inset.grid(False)
    inset.legend(loc="upper right", frameon=False, fontsize=5.7, handlelength=1.5, borderaxespad=0.2, labelspacing=0.2)

    # Highlight c=10 values used by inset directly on the main curve.
    for jv in available:
        key = f"c{inset_cut}_J{jv:g}"
        if key not in spectrum_probs:
            continue
        p = np.asarray(spectrum_probs[key]["p_mean"], dtype=np.float64)
        ps = float(np.sum(p))
        if ps <= 0.0:
            rv = 1.0
        else:
            q = np.clip(p / ps, 1e-30, 1.0)
            rv = float(np.exp(-np.sum(q * np.log(q))))
        ax.plot(
            [float(jv)],
            [rv],
            ls="None",
            marker="o",
            ms=4.6,
            color=inset_colors.get(jv, "black"),
            markeredgecolor="black",
            markeredgewidth=0.2,
            zorder=6,
        )

    fig.savefig(out_stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("benchmark_entropy_convergence_and_spectrum_results"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--parallel", action="store_true", default=True)
    p.add_argument("--no-parallel", dest="parallel", action="store_false")
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--cut", type=int, default=10)
    p.add_argument("--cuts", type=str, default="1,5,10,15,20")
    p.add_argument("--m-spectrum", type=int, default=64)
    p.add_argument("--spectrum-draws", type=int, default=1)
    p.add_argument("--spectrum-js", type=str, default=",".join(str(v) for v in DENSE_JS_DEFAULT))
    p.add_argument("--rank-tol", type=float, default=RANK_TOL)
    p.add_argument("--plot-cuts", type=str, default="")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--benchmark-only", action="store_true")
    p.add_argument("--spectrum-rank-csv", type=Path, default=None)
    p.add_argument("--spectrum-npz", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cuts = _parse_int_list(args.plot_cuts) if str(args.plot_cuts).strip() else None

    if bool(args.plot_only):
        spectrum_rank_csv = args.spectrum_rank_csv if args.spectrum_rank_csv is not None else out_dir / "spectrum_rank_summary.csv"
        spectrum_npz = args.spectrum_npz if args.spectrum_npz is not None else out_dir / "spectrum_probs.npz"
        rank_rows = _load_csv(spectrum_rank_csv)
        loaded = np.load(spectrum_npz)
        probs: dict[str, dict[str, np.ndarray]] = {}
        for key in loaded.files:
            if not key.endswith("_mean"):
                continue
            root = key[:-5]
            if key.endswith("_smean"):
                continue
            if root.endswith("_s"):
                continue
            if root.startswith("J"):
                # Backward compatibility: old single-cut format.
                jlabel = root[1:]
                probs[f"c{int(args.cut)}_J{jlabel}"] = {
                    "p_mean": np.asarray(loaded[key], dtype=np.float64),
                    "p_std": np.asarray(loaded[f"J{jlabel}_std"], dtype=np.float64),
                    "s_mean": np.asarray(loaded[f"J{jlabel}_smean"], dtype=np.float64) if f"J{jlabel}_smean" in loaded.files else np.sqrt(np.clip(np.asarray(loaded[key], dtype=np.float64), 0.0, None)),
                    "s_std": np.asarray(loaded[f"J{jlabel}_sstd"], dtype=np.float64) if f"J{jlabel}_sstd" in loaded.files else np.zeros_like(np.asarray(loaded[key], dtype=np.float64), dtype=np.float64),
                }
                continue
            # New multi-cut format: c<cut>_J<j>
            ctag = root
            if f"{ctag}_std" not in loaded.files:
                continue
            p_mean = np.asarray(loaded[key], dtype=np.float64)
            has_s_mean = f"{ctag}_smean" in loaded.files
            has_s_std = f"{ctag}_sstd" in loaded.files
            probs[ctag] = {
                "p_mean": p_mean,
                "p_std": np.asarray(loaded[f"{ctag}_std"], dtype=np.float64),
                "s_mean": np.asarray(loaded[f"{ctag}_smean"], dtype=np.float64) if has_s_mean else np.sqrt(np.clip(p_mean, 0.0, None)),
                "s_std": np.asarray(loaded[f"{ctag}_sstd"], dtype=np.float64) if has_s_std else np.zeros_like(p_mean, dtype=np.float64),
            }
        plot_from_saved(
            rank_rows=rank_rows,
            spectrum_probs=probs,
            out_stem=out_dir / "fig_spectrum_and_rank_vs_j_prl",
            rank_tol=float(args.rank_tol),
            plot_cuts=plot_cuts,
        )
        print(f"Wrote figure: {(out_dir / 'fig_spectrum_and_rank_vs_j_prl').with_suffix('.pdf')}", flush=True)
        return

    rank_rows, spectrum_probs = run_benchmark(args)
    if not bool(args.benchmark_only):
        plot_from_saved(
            rank_rows=rank_rows,
            spectrum_probs=spectrum_probs,
            out_stem=out_dir / "fig_spectrum_and_rank_vs_j_prl",
            rank_tol=float(args.rank_tol),
            plot_cuts=plot_cuts,
        )
        print(f"Wrote figure: {(out_dir / 'fig_spectrum_and_rank_vs_j_prl').with_suffix('.pdf')}", flush=True)
    print(f"Wrote tables to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
