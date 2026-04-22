#!/usr/bin/env python3
"""Separated Figure-2 benchmarks:
1) convergence S_V vs probe budget m over dense J sweep,
2) singular-value spectrum and effective mode rank R(J).

Pipeline is unchanged: exact probes -> weighted V (beta=1) -> past-row centering -> SVD.
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

M_GRID_DEFAULT = (4, 8, 16, 32, 64)
DENSE_JS_DEFAULT = tuple(round(0.05 * i, 10) for i in range(41))
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


def _weighted_centered_singular_values(
    *,
    probe_set,
    op: MPO,
    sim_params: AnalogSimParams,
    psi0: np.ndarray,
    parallel: bool,
) -> np.ndarray:
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


def _entropy_from_singular_values(s: np.ndarray) -> float:
    p = np.asarray(s, dtype=np.float64) ** 2
    ps = float(np.sum(p))
    if ps <= 0.0:
        return 0.0
    q = p / ps
    q = np.clip(q, 1e-30, 1.0)
    return float(-np.sum(q * np.log(q)))


def _aggregate_variable_length(vectors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not vectors:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    max_len = max(int(v.size) for v in vectors)
    arr = np.full((len(vectors), max_len), np.nan, dtype=np.float64)
    for i, v in enumerate(vectors):
        arr[i, : v.size] = v
    return np.nanmean(arr, axis=0), (np.nanstd(arr, axis=0, ddof=1) if len(vectors) > 1 else np.zeros(max_len))


def run_convergence_benchmark(args: argparse.Namespace) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    m_values = _parse_int_list(args.m_values)
    conv_js = _parse_float_list(args.convergence_js)

    init_rng = np.random.default_rng(int(args.seed) + 77_777)
    initial_list = _list_initial_states_sys_env0(n_seeds=int(args.n_seeds), rng=init_rng)
    np.save(out_dir / "initial_states.npy", np.stack(initial_list, axis=0))

    sim_params = AnalogSimParams(dt=DT_FIXED, solver="MCWF", show_progress=False)
    detail_rows: list[dict[str, float | int]] = []

    m_max = int(max(m_values))
    for j in conv_js:
        op = MPO.ising(length=L_FIXED, J=float(j), g=G_FIXED)
        for draw in range(int(args.probe_draws)):
            draw_seed = int(args.seed) + 100_000 * int(args.cut) + 10 * int(round(100 * j)) + draw
            base_probe_set = sample_split_cut_probes(
                cut=int(args.cut),
                k=K_FIXED,
                n_pasts=m_max,
                n_futures=m_max,
                rng=np.random.default_rng(draw_seed),
            )
            for m in m_values:
                mm = int(m)
                draw_entropy: list[float] = []
                for psi0 in initial_list:
                    # Reuse the same probe set prefixes for this draw.
                    s_full = _weighted_centered_singular_values(
                        probe_set=sample_split_cut_probes(
                            cut=int(args.cut),
                            k=K_FIXED,
                            n_pasts=mm,
                            n_futures=mm,
                            rng=np.random.default_rng(draw_seed),
                        ),
                        op=op,
                        sim_params=sim_params,
                        psi0=psi0,
                        parallel=bool(args.parallel),
                    )
                    draw_entropy.append(_entropy_from_singular_values(s_full))
                detail_rows.append(
                    {
                        "cut": int(args.cut),
                        "J": float(j),
                        "m": mm,
                        "probe_draw": int(draw),
                        "entropy": float(np.mean(draw_entropy)),
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

    _write_csv(out_dir / "convergence_detail.csv", detail_rows)
    _write_csv(out_dir / "convergence_summary.csv", summary_rows)
    return detail_rows, summary_rows


def run_spectrum_rank_benchmark(args: argparse.Namespace) -> tuple[list[dict[str, float | int]], dict[str, dict[str, np.ndarray]]]:
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
                seed_ranks.append(float(np.sum(s > RANK_TOL)))
            mean_seed, _ = _aggregate_variable_length(seed_vectors)
            per_draw_vectors.append(mean_seed)
            per_draw_rank.append(float(np.mean(seed_ranks)))

        p_mean, p_std = _aggregate_variable_length(per_draw_vectors)
        spectrum_probs[f"{jv:g}"] = {"p_mean": p_mean, "p_std": p_std}
        rank_rows.append(
            {
                "J": float(jv),
                "R_mean": float(np.mean(per_draw_rank)),
                "R_std": float(np.std(per_draw_rank, ddof=1)) if len(per_draw_rank) > 1 else 0.0,
                "rank_tol": float(RANK_TOL),
                "cut": int(args.cut),
                "m_spectrum": int(args.m_spectrum),
            }
        )

    _write_csv(out_dir / "spectrum_rank_summary.csv", rank_rows)
    np.savez_compressed(
        out_dir / "spectrum_probs.npz",
        **{f"J{j:g}_mean": v["p_mean"] for j, v in {float(k): val for k, val in spectrum_probs.items()}.items()},
        **{f"J{j:g}_std": v["p_std"] for j, v in {float(k): val for k, val in spectrum_probs.items()}.items()},
    )
    return rank_rows, spectrum_probs


def plot_convergence_from_saved(*, summary_rows: list[dict[str, str | float | int]], out_stem: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.ticker import LogLocator, NullLocator

    _configure_matplotlib_prl_figure()
    fig, ax = plt.subplots(1, 1, figsize=(3.35, 2.45), constrained_layout=True)
    js = sorted({float(r["J"]) for r in summary_rows})
    norm = Normalize(vmin=min(js), vmax=max(js)) if js else Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap("viridis")
    m_vals = sorted({int(float(r["m"])) for r in summary_rows})

    for jv in js:
        sub = sorted([r for r in summary_rows if abs(float(r["J"]) - jv) < 1e-12], key=lambda r: int(float(r["m"])))
        xs = np.asarray([int(float(r["m"])) for r in sub], dtype=np.float64)
        mu = np.asarray([float(r["entropy_mean"]) for r in sub], dtype=np.float64)
        sem = np.asarray([float(r.get("entropy_sem", 0.0)) for r in sub], dtype=np.float64)
        ys = np.clip(mu, 1e-30, None)
        lo = np.clip(mu - sem, 1e-30, None)
        hi = np.clip(mu + sem, 1e-30, None)
        col = cmap(norm(jv))
        ax.semilogy(xs, ys, color=col, lw=0.9, marker="o", ms=2.2, alpha=0.9)
        ax.fill_between(xs, lo, hi, color=col, alpha=0.08, linewidth=0)

    ax.set_xlabel(r"Probe budget $m$ ($N_p=N_f$)")
    ax.set_ylabel(r"$S_V$")
    ax.set_xticks(m_vals)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(True, which="major", axis="y", alpha=0.1, linewidth=0.3)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.9)
    cbar.ax.set_title(r"$J$", pad=2)

    fig.savefig(out_stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_spectrum_rank_from_saved(
    *,
    rank_rows: list[dict[str, str | float | int]],
    spectrum_probs: dict[str, dict[str, np.ndarray]],
    out_stem: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.cm import ScalarMappable

    _configure_matplotlib_prl_figure()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.2, 2.45), constrained_layout=True, gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.08})

    js = sorted(float(k) for k in spectrum_probs)
    max_len = max((int(np.asarray(spectrum_probs[f"{j:g}"]["p_mean"]).size) for j in js), default=1)
    z = np.full((len(js), max_len), np.nan, dtype=np.float64)
    for i, jv in enumerate(js):
        p = np.asarray(spectrum_probs[f"{jv:g}"]["p_mean"], dtype=np.float64)
        z[i, : p.size] = np.clip(p, 1e-30, None)

    x = np.arange(1, max_len + 1, dtype=np.float64)
    y = np.asarray(js, dtype=np.float64)
    x_edges = np.concatenate(([0.5], 0.5 * (x[:-1] + x[1:]), [x[-1] + 0.5]))
    if y.size >= 2:
        dy = float(np.median(np.diff(y)))
        y_edges = np.concatenate(([y[0] - dy / 2], 0.5 * (y[:-1] + y[1:]), [y[-1] + dy / 2]))
    else:
        y_edges = np.array([y[0] - 0.05, y[0] + 0.05]) if y.size else np.array([-0.05, 0.05])

    im = ax0.pcolormesh(x_edges, y_edges, z, cmap="cividis", norm=LogNorm(vmin=1e-16, vmax=1.0), shading="auto", rasterized=True)
    ax0.set_xlabel(r"Mode index $n$")
    ax0.set_ylabel(r"$J$")
    cbar = fig.colorbar(im, ax=ax0, pad=0.02, shrink=0.9)
    cbar.ax.set_title(r"$p_n$", pad=2)

    rr = sorted(rank_rows, key=lambda r: float(r["J"]))
    xs = np.asarray([float(r["J"]) for r in rr], dtype=np.float64)
    mu = np.asarray([float(r["R_mean"]) for r in rr], dtype=np.float64)
    sd = np.asarray([float(r.get("R_std", 0.0)) for r in rr], dtype=np.float64)
    ax1.plot(xs, mu, color="#1f77b4", lw=1.4, marker="o", ms=2.6)
    ax1.fill_between(xs, np.clip(mu - sd, 0.0, None), mu + sd, color="#1f77b4", alpha=0.14, linewidth=0)
    ax1.set_xlabel(r"$J$")
    ax1.set_ylabel(r"$R=\#\{s_n>10^{-16}\}$")
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
    p.add_argument("--cut", type=int, default=15)

    p.add_argument("--run-convergence", action="store_true", default=True)
    p.add_argument("--no-run-convergence", dest="run_convergence", action="store_false")
    p.add_argument("--run-spectrum-rank", action="store_true", default=True)
    p.add_argument("--no-run-spectrum-rank", dest="run_spectrum_rank", action="store_false")

    p.add_argument("--probe-draws", type=int, default=8)
    p.add_argument("--m-values", type=str, default=",".join(str(v) for v in M_GRID_DEFAULT))
    p.add_argument("--convergence-js", type=str, default=",".join(str(v) for v in DENSE_JS_DEFAULT))

    p.add_argument("--m-spectrum", type=int, default=64)
    p.add_argument("--spectrum-draws", type=int, default=3)
    p.add_argument("--spectrum-js", type=str, default=",".join(str(v) for v in DENSE_JS_DEFAULT))

    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--benchmark-only", action="store_true")
    p.add_argument("--convergence-summary-csv", type=Path, default=None)
    p.add_argument("--spectrum-rank-csv", type=Path, default=None)
    p.add_argument("--spectrum-npz", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.plot_only):
        if bool(args.run_convergence):
            summary_csv = args.convergence_summary_csv if args.convergence_summary_csv is not None else out_dir / "convergence_summary.csv"
            rows = _load_csv(summary_csv)
            plot_convergence_from_saved(summary_rows=rows, out_stem=out_dir / "fig_convergence_sv_vs_m_prl")
            print(f"Wrote figure: {(out_dir / 'fig_convergence_sv_vs_m_prl').with_suffix('.pdf')}", flush=True)

        if bool(args.run_spectrum_rank):
            spectrum_rank_csv = args.spectrum_rank_csv if args.spectrum_rank_csv is not None else out_dir / "spectrum_rank_summary.csv"
            spectrum_npz = args.spectrum_npz if args.spectrum_npz is not None else out_dir / "spectrum_probs.npz"
            rank_rows = _load_csv(spectrum_rank_csv)
            loaded = np.load(spectrum_npz)
            probs: dict[str, dict[str, np.ndarray]] = {}
            for key in loaded.files:
                if not key.startswith("J") or not key.endswith("_mean"):
                    continue
                jlabel = key[1:-5]
                probs[jlabel] = {
                    "p_mean": np.asarray(loaded[key], dtype=np.float64),
                    "p_std": np.asarray(loaded[f"J{jlabel}_std"], dtype=np.float64),
                }
            plot_spectrum_rank_from_saved(rank_rows=rank_rows, spectrum_probs=probs, out_stem=out_dir / "fig_spectrum_and_rank_vs_j_prl")
            print(f"Wrote figure: {(out_dir / 'fig_spectrum_and_rank_vs_j_prl').with_suffix('.pdf')}", flush=True)
        return

    if bool(args.run_convergence):
        _, summary_rows = run_convergence_benchmark(args)
        if not bool(args.benchmark_only):
            plot_convergence_from_saved(summary_rows=summary_rows, out_stem=out_dir / "fig_convergence_sv_vs_m_prl")
            print(f"Wrote figure: {(out_dir / 'fig_convergence_sv_vs_m_prl').with_suffix('.pdf')}", flush=True)

    if bool(args.run_spectrum_rank):
        rank_rows, spectrum_probs = run_spectrum_rank_benchmark(args)
        if not bool(args.benchmark_only):
            plot_spectrum_rank_from_saved(rank_rows=rank_rows, spectrum_probs=spectrum_probs, out_stem=out_dir / "fig_spectrum_and_rank_vs_j_prl")
            print(f"Wrote figure: {(out_dir / 'fig_spectrum_and_rank_vs_j_prl').with_suffix('.pdf')}", flush=True)

    print(f"Wrote tables to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
