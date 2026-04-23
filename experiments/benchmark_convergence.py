#!/usr/bin/env python3
"""Convergence benchmark: S_V vs probe budget m over dense J sweep.

Pipeline: exact probes -> weighted V (beta=1) -> past-row centering -> SVD entropy.
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

M_GRID_DEFAULT = (4, 8, 16, 32, 64)
DENSE_JS_DEFAULT = tuple(round(0.05 * i, 10) for i in range(41))


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


def _slice_probe_prefix(pauli_xyz_ij: np.ndarray, weights_ij: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Slice raw evaluated responses to an m x m probe prefix (slice first, then center)."""
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")

    p = np.asarray(pauli_xyz_ij)
    if p.ndim < 2:
        raise ValueError(f"expected pauli_xyz_ij with at least 2 dimensions, got shape={p.shape}")
    if p.shape[0] < m or p.shape[1] < m:
        raise ValueError(f"cannot slice pauli_xyz_ij shape={p.shape} to m={m}")
    p_sub = p[:m, :m, ...]

    w = np.asarray(weights_ij)
    # Most common: branch weights per past/future pair.
    if w.ndim >= 2 and w.shape[0] >= m and w.shape[1] >= m:
        w_sub = w[:m, :m, ...]
    # Fallback: weights only along past-branch axis.
    elif w.shape[0] >= m:
        w_sub = w[:m, ...]
    else:
        raise ValueError(f"cannot slice weights_ij shape={w.shape} to m={m}")

    return np.asarray(p_sub), np.asarray(w_sub)


def _entropy_from_sliced_raw(pauli_xyz_sub: np.ndarray, weights_sub: np.ndarray) -> float:
    """Compute S_V from already-sliced raw probe responses and weights."""
    v_w = build_weighted_v_matrix(pauli_xyz_sub, weights_sub, BRANCH_WEIGHT_BETA)
    v_c = center_past_rows(v_w)
    s = np.linalg.svd(np.asarray(v_c, dtype=np.float64), compute_uv=False).astype(np.float64)
    p = s * s
    ps = float(np.sum(p))
    if ps <= 0.0:
        return 0.0
    q = np.clip(p / ps, 1e-30, 1.0)
    return float(-np.sum(q * np.log(q)))


def run_benchmark(args: argparse.Namespace) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    m_values = sorted(_parse_int_list(args.m_values))
    m_max = int(max(m_values))
    conv_js = _parse_float_list(args.convergence_js)

    init_rng = np.random.default_rng(int(args.seed) + 77_777)
    initial_list = _list_initial_states_sys_env0(n_seeds=int(args.n_seeds), rng=init_rng)
    np.save(out_dir / "initial_states.npy", np.stack(initial_list, axis=0))

    sim_params = AnalogSimParams(dt=DT_FIXED, solver="MCWF", show_progress=False)
    detail_rows: list[dict[str, float | int]] = []
    shape_check_done = False

    for j in conv_js:
        op = MPO.ising(length=L_FIXED, J=float(j), g=G_FIXED)
        for draw in range(int(args.probe_draws)):
            draw_seed = int(args.seed) + 100_000 * int(args.cut) + 10 * int(round(100 * j)) + draw
            probe_set_max = sample_split_cut_probes(
                cut=int(args.cut),
                k=K_FIXED,
                n_pasts=m_max,
                n_futures=m_max,
                rng=np.random.default_rng(draw_seed),
            )
            entropies_by_m: dict[int, list[float]] = {int(m): [] for m in m_values}

            for psi0 in initial_list:
                pauli_xyz_max, weights_max, _ = evaluate_exact_probe_set_with_diagnostics(
                    probe_set=probe_set_max,
                    operator=op,
                    sim_params=sim_params,
                    initial_psi=psi0,
                    parallel=bool(args.parallel),
                )
                w_clean_max, _ = prepare_branch_weights(weights_max, log_warnings=False)

                if not shape_check_done:
                    pshape = np.asarray(pauli_xyz_max).shape
                    if len(pshape) < 2 or pshape[0] != m_max or pshape[1] != m_max:
                        raise ValueError(
                            f"unexpected pauli_xyz_max shape={pshape}, expected first two axes = ({m_max}, {m_max})"
                        )
                    shape_check_done = True

                for m in m_values:
                    pauli_sub, w_sub = _slice_probe_prefix(pauli_xyz_max, w_clean_max, int(m))
                    entropy = _entropy_from_sliced_raw(pauli_sub, w_sub)
                    entropies_by_m[int(m)].append(float(entropy))

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

    _write_csv(out_dir / "convergence_detail.csv", detail_rows)
    _write_csv(out_dir / "convergence_summary.csv", summary_rows)
    return detail_rows, summary_rows


def plot_from_saved(*, summary_rows: list[dict[str, str | float | int]], out_stem: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
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
    from matplotlib.colors import Normalize
    from matplotlib.ticker import LogLocator, NullLocator

    _configure_matplotlib_prl_figure()
    fig, ax = plt.subplots(1, 1, figsize=(3.35, 2.45), constrained_layout=True)
    js = sorted({float(r["J"]) for r in summary_rows})
    norm = Normalize(vmin=min(js), vmax=max(js)) if js else Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap("viridis")
    m_vals = sorted({int(float(r["m"])) for r in summary_rows})
    m_ref = max(m_vals) if m_vals else 0

    for jv in js:
        sub = sorted([r for r in summary_rows if abs(float(r["J"]) - jv) < 1e-12], key=lambda r: int(float(r["m"])))
        if not sub:
            continue
        ref_rows = [r for r in sub if int(float(r["m"])) == int(m_ref)]
        if not ref_rows:
            continue
        ref_mu = float(ref_rows[0]["entropy_mean"])
        xs = np.asarray([int(float(r["m"])) for r in sub], dtype=np.float64)
        mu = np.asarray([float(r["entropy_mean"]) for r in sub], dtype=np.float64)
        sem = np.asarray([float(r.get("entropy_sem", 0.0)) for r in sub], dtype=np.float64)
        err = np.abs(mu - ref_mu)
        lo = np.clip(err - sem, 1e-30, None)
        hi = np.clip(err + sem, 1e-30, None)
        ys = np.clip(err, 1e-30, None)
        col = cmap(norm(jv))
        ax.semilogy(xs, ys, color=col, lw=0.9, marker="o", ms=2.2, alpha=0.9)
        ax.fill_between(xs, lo, hi, color=col, alpha=0.08, linewidth=0)

    ax.set_xlabel(r"Probe budget $m$ ($N_p=N_f$)")
    ax.set_ylabel(r"$|S_V(m)-S_V(m_{\mathrm{ref}})|$")
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("benchmark_entropy_convergence_and_spectrum_results"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--parallel", action="store_true", default=True)
    p.add_argument("--no-parallel", dest="parallel", action="store_false")
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--cut", type=int, default=15)
    p.add_argument("--probe-draws", type=int, default=8)
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
        rows = _load_csv(summary_csv)
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
