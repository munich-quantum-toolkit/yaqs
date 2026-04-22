#!/usr/bin/env python3
"""Modes benchmark: singular-value spectrum and effective rank R(J).

R(J) is defined as the number of singular values above 1e-16.
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

    init_rng = np.random.default_rng(int(args.seed) + 77_777)
    initial_list = _list_initial_states_sys_env0(n_seeds=int(args.n_seeds), rng=init_rng)
    sim_params = AnalogSimParams(dt=DT_FIXED, solver="MCWF", show_progress=False)

    spectrum_probs: dict[str, dict[str, np.ndarray]] = {}
    rank_rows: list[dict[str, float | int]] = []

    for jv in spec_js:
        op = MPO.ising(length=L_FIXED, J=float(jv), g=G_FIXED)
        per_draw_vectors: list[np.ndarray] = []
        per_draw_rank: list[float] = []
        per_draw_singulars: list[np.ndarray] = []
        for draw in range(int(args.spectrum_draws)):
            probe_seed = int(args.seed) + 900_000 + 100_000 * int(args.cut) + 100 * int(round(100 * jv)) + draw
            probe_set = sample_split_cut_probes(
                cut=int(args.cut),
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
        spectrum_probs[f"{jv:g}"] = {"p_mean": p_mean, "p_std": p_std, "s_mean": s_mean, "s_std": s_std}
        rank_rows.append(
            {
                "J": float(jv),
                "R_mean": float(r_effective),
                "R_std": float(np.std(per_draw_rank, ddof=1)) if len(per_draw_rank) > 1 else 0.0,
                "rank_tol": float(args.rank_tol),
                "cut": int(args.cut),
                "m_spectrum": int(args.m_spectrum),
            }
        )

    _write_csv(out_dir / "spectrum_rank_summary.csv", rank_rows)
    np.savez_compressed(
        out_dir / "spectrum_probs.npz",
        **{f"J{j:g}_mean": v["p_mean"] for j, v in {float(k): val for k, val in spectrum_probs.items()}.items()},
        **{f"J{j:g}_std": v["p_std"] for j, v in {float(k): val for k, val in spectrum_probs.items()}.items()},
        **{f"J{j:g}_smean": v["s_mean"] for j, v in {float(k): val for k, val in spectrum_probs.items()}.items()},
        **{f"J{j:g}_sstd": v["s_std"] for j, v in {float(k): val for k, val in spectrum_probs.items()}.items()},
    )
    return rank_rows, spectrum_probs


def plot_from_saved(
    *,
    rank_rows: list[dict[str, str | float | int]],
    spectrum_probs: dict[str, dict[str, np.ndarray]],
    out_stem: Path,
    rank_tol: float,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    _configure_matplotlib_prl_figure()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.2, 2.45), constrained_layout=True, gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.08})

    js = sorted(float(k) for k in spectrum_probs)
    norm = Normalize(vmin=min(js) if js else 0.0, vmax=max(js) if js else 1.0)
    cmap = plt.get_cmap("viridis")
    for jv in js:
        spec = spectrum_probs[f"{jv:g}"]
        p = np.asarray(spec["p_mean"], dtype=np.float64)
        if p.size == 0:
            continue
        n = np.arange(1, p.size + 1, dtype=np.float64)
        ax0.plot(n, np.clip(p, 1e-30, None), color=cmap(norm(jv)), lw=1.1, alpha=0.95)

    ax0.set_xlabel(r"Mode index $n$")
    ax0.set_ylabel(r"$p_n$")
    ax0.set_yscale("log")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax0, pad=0.02, shrink=0.9)
    cbar.ax.set_title(r"$J$", pad=2)

    xs = np.asarray(js, dtype=np.float64)
    mu = np.asarray(
        [float(np.sum(np.asarray(spectrum_probs[f"{jv:g}"]["s_mean"], dtype=np.float64) > float(rank_tol))) for jv in js],
        dtype=np.float64,
    )
    ax1.plot(xs, mu, color="#1f77b4", lw=1.4, marker="o", ms=2.6)
    ax1.set_xlabel(r"$J$")
    ax1.set_ylabel(rf"$R=\#\{{s_n>{rank_tol:.0e}\}}$")
    ax1.grid(True, axis="y", alpha=0.1, linewidth=0.3)

    for ax, tag in ((ax0, "(a)"), (ax1, "(b)")):
        ax.text(0.04, 0.955, tag, transform=ax.transAxes, va="top", ha="left", fontsize=7.3, fontweight="semibold")

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
    p.add_argument("--m-spectrum", type=int, default=64)
    p.add_argument("--spectrum-draws", type=int, default=1)
    p.add_argument("--spectrum-js", type=str, default=",".join(str(v) for v in DENSE_JS_DEFAULT))
    p.add_argument("--rank-tol", type=float, default=RANK_TOL)
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--benchmark-only", action="store_true")
    p.add_argument("--spectrum-rank-csv", type=Path, default=None)
    p.add_argument("--spectrum-npz", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.plot_only):
        spectrum_rank_csv = args.spectrum_rank_csv if args.spectrum_rank_csv is not None else out_dir / "spectrum_rank_summary.csv"
        spectrum_npz = args.spectrum_npz if args.spectrum_npz is not None else out_dir / "spectrum_probs.npz"
        rank_rows = _load_csv(spectrum_rank_csv)
        loaded = np.load(spectrum_npz)
        probs: dict[str, dict[str, np.ndarray]] = {}
        for key in loaded.files:
            if not key.startswith("J") or not key.endswith("_mean"):
                continue
            jlabel = key[1:-5]
            p_mean = np.asarray(loaded[key], dtype=np.float64)
            has_s_mean = f"J{jlabel}_smean" in loaded.files
            has_s_std = f"J{jlabel}_sstd" in loaded.files
            probs[jlabel] = {
                "p_mean": p_mean,
                "p_std": np.asarray(loaded[f"J{jlabel}_std"], dtype=np.float64),
                "s_mean": np.asarray(loaded[f"J{jlabel}_smean"], dtype=np.float64) if has_s_mean else np.sqrt(np.clip(p_mean, 0.0, None)),
                "s_std": np.asarray(loaded[f"J{jlabel}_sstd"], dtype=np.float64) if has_s_std else np.zeros_like(p_mean, dtype=np.float64),
            }
        plot_from_saved(rank_rows=rank_rows, spectrum_probs=probs, out_stem=out_dir / "fig_spectrum_and_rank_vs_j_prl", rank_tol=float(args.rank_tol))
        print(f"Wrote figure: {(out_dir / 'fig_spectrum_and_rank_vs_j_prl').with_suffix('.pdf')}", flush=True)
        return

    rank_rows, spectrum_probs = run_benchmark(args)
    if not bool(args.benchmark_only):
        plot_from_saved(
            rank_rows=rank_rows,
            spectrum_probs=spectrum_probs,
            out_stem=out_dir / "fig_spectrum_and_rank_vs_j_prl",
            rank_tol=float(args.rank_tol),
        )
        print(f"Wrote figure: {(out_dir / 'fig_spectrum_and_rank_vs_j_prl').with_suffix('.pdf')}", flush=True)
    print(f"Wrote tables to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
