#!/usr/bin/env python3
"""Figure-2 style benchmarks: probe convergence and singular-value spectrum / rank.

Outputs:
- ``fig_convergence_sv_vs_m_prl`` — S_V vs probe budget m
- ``fig_spectrum_and_rank_vs_j_prl`` — mode probabilities vs J and effective rank R(J)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from _benchmark_common import (
    BRANCH_WEIGHT_BETA,
    DT_DEFAULT,
    G_DEFAULT,
    list_initial_states_sys_env0,
    parse_float_list,
    parse_int_list,
)
from _benchmark_plotting import plot_convergence_sv_vs_m, plot_spectrum_and_rank_vs_j
from mqt.yaqs.characterization.memory.diagnostics.probe import sample_split_cut_probes
from mqt.yaqs.characterization.memory.reference.exact import evaluate_exact_probe_set_with_diagnostics
from mqt.yaqs.characterization.memory.diagnostics.v_matrix import (
    build_weighted_v_matrix,
    center_past_rows,
    prepare_branch_weights,
)
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

RANK_TOL = 1e-16


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--length", type=int, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--cut", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--parallel", action="store_true", default=True)
    p.add_argument("--no-parallel", dest="parallel", action="store_false")
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--run-convergence", action="store_true", default=True)
    p.add_argument("--no-run-convergence", dest="run_convergence", action="store_false")
    p.add_argument("--run-spectrum-rank", action="store_true", default=True)
    p.add_argument("--no-run-spectrum-rank", dest="run_spectrum_rank", action="store_false")
    p.add_argument("--probe-draws", type=int, default=None)
    p.add_argument("--m-values", type=str, default=None)
    p.add_argument("--convergence-js", type=str, default=None)
    p.add_argument("--m-spectrum", type=int, default=None)
    p.add_argument("--spectrum-draws", type=int, default=None)
    p.add_argument("--spectrum-js", type=str, default=None)
    p.add_argument("--plot-only", action="store_true")
    return p.parse_args()


def _resolve_config(args: argparse.Namespace) -> dict[str, object]:
    if args.quick:
        length, k, cut = 2, 8, 4
        m_values = "4,8,12,16"
        conv_js = "0.0,0.5,1.0,1.5,2.0"
        spec_js = conv_js
        probe_draws = 2
        m_spectrum = 16
        spectrum_draws = 2
        out_dir = Path("benchmark_entropy_convergence_and_spectrum_quick_results")
    else:
        length, k, cut = 6, 20, 15
        m_values = "4,8,16,32,64"
        conv_js = ",".join(str(round(0.05 * i, 2)) for i in range(41))
        spec_js = conv_js
        probe_draws = 8
        m_spectrum = 64
        spectrum_draws = 3
        out_dir = Path("benchmark_entropy_convergence_and_spectrum_results")
    return {
        "length": args.length if args.length is not None else length,
        "k": args.k if args.k is not None else k,
        "cut": args.cut if args.cut is not None else cut,
        "m_values": parse_int_list(args.m_values if args.m_values else m_values),
        "conv_js": parse_float_list(args.convergence_js if args.convergence_js else conv_js),
        "spec_js": parse_float_list(args.spectrum_js if args.spectrum_js else spec_js),
        "probe_draws": args.probe_draws if args.probe_draws is not None else probe_draws,
        "m_spectrum": args.m_spectrum if args.m_spectrum is not None else m_spectrum,
        "spectrum_draws": args.spectrum_draws if args.spectrum_draws is not None else spectrum_draws,
        "out_dir": Path(args.out_dir if args.out_dir is not None else out_dir),
    }


def _weighted_centered_singular_values(*, probe_set, op, sim_params, psi0, parallel: bool) -> np.ndarray:
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
    q = np.clip(p / ps, 1e-30, 1.0)
    return float(-np.sum(q * np.log(q)))


def _write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_convergence(cfg: dict[str, object], args: argparse.Namespace) -> list[dict[str, float | int]]:
    out_dir = cfg["out_dir"].resolve()
    length = int(cfg["length"])
    k = int(cfg["k"])
    cut = int(cfg["cut"])
    m_values = cfg["m_values"]
    conv_js = cfg["conv_js"]
    m_max = int(max(m_values))

    init_rng = np.random.default_rng(int(args.seed) + 77_777)
    initial_list = list_initial_states_sys_env0(length=length, n_seeds=int(args.n_seeds), rng=init_rng)
    sim_params = AnalogSimParams(dt=DT_DEFAULT)
    detail_rows: list[dict[str, float | int]] = []

    for jv in conv_js:
        op = MPO.ising(length=length, J=float(jv), g=G_DEFAULT)
        for draw in range(int(cfg["probe_draws"])):
            draw_seed = int(args.seed) + 100_000 * cut + 10 * int(round(100 * jv)) + draw
            for m in m_values:
                probe_set = sample_split_cut_probes(
                    cut=cut,
                    k=k,
                    n_pasts=int(m),
                    n_futures=int(m),
                    rng=np.random.default_rng(draw_seed),
                )
                draw_entropy = [
                    _entropy_from_singular_values(
                        _weighted_centered_singular_values(
                            probe_set=probe_set,
                            op=op,
                            sim_params=sim_params,
                            psi0=psi0,
                            parallel=bool(args.parallel),
                        )
                    )
                    for psi0 in initial_list
                ]
                detail_rows.append({"cut": cut, "J": float(jv), "m": int(m), "probe_draw": draw, "entropy": float(np.mean(draw_entropy))})
                print(f"conv: J={jv:.1f}, m={m}, S_V={detail_rows[-1]['entropy']:.6e}", flush=True)

    summary: list[dict[str, float | int]] = []
    grouped: dict[tuple[int, float], list[float]] = {}
    for row in detail_rows:
        grouped.setdefault((int(row["m"]), float(row["J"])), []).append(float(row["entropy"]))
    for (m, jv), vals in sorted(grouped.items()):
        summary.append({"cut": cut, "J": jv, "m": m, "entropy_mean": float(np.mean(vals)), "entropy_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0})

    _write_csv(out_dir / "convergence_detail.csv", detail_rows)
    _write_csv(out_dir / "convergence_summary.csv", summary)
    return summary


def run_spectrum_rank(cfg: dict[str, object], args: argparse.Namespace) -> tuple[list[dict[str, float | int]], dict[str, dict[str, np.ndarray]]]:
    out_dir = cfg["out_dir"].resolve()
    length = int(cfg["length"])
    k = int(cfg["k"])
    cut = int(cfg["cut"])
    spec_js = cfg["spec_js"]
    m_spec = int(cfg["m_spectrum"])

    init_rng = np.random.default_rng(int(args.seed) + 77_777)
    initial_list = list_initial_states_sys_env0(length=length, n_seeds=int(args.n_seeds), rng=init_rng)
    sim_params = AnalogSimParams(dt=DT_DEFAULT)
    spectrum_probs: dict[str, dict[str, np.ndarray]] = {}
    rank_rows: list[dict[str, float | int]] = []

    for jv in spec_js:
        op = MPO.ising(length=length, J=float(jv), g=G_DEFAULT)
        per_draw_vectors: list[np.ndarray] = []
        per_draw_rank: list[float] = []
        for draw in range(int(cfg["spectrum_draws"])):
            probe_seed = int(args.seed) + 900_000 + 100_000 * cut + 100 * int(round(100 * jv)) + draw
            probe_set = sample_split_cut_probes(
                cut=cut,
                k=k,
                n_pasts=m_spec,
                n_futures=m_spec,
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
                seed_vectors.append(p / ps if ps > 0.0 else np.zeros(0, dtype=np.float64))
                seed_ranks.append(float(np.sum(s > RANK_TOL)))
            max_len = max(v.size for v in seed_vectors)
            arr = np.full((len(seed_vectors), max_len), np.nan)
            for i, v in enumerate(seed_vectors):
                arr[i, : v.size] = v
            per_draw_vectors.append(np.nanmean(arr, axis=0))
            per_draw_rank.append(float(np.mean(seed_ranks)))

        max_len = max(v.size for v in per_draw_vectors)
        arr = np.full((len(per_draw_vectors), max_len), np.nan)
        for i, v in enumerate(per_draw_vectors):
            arr[i, : v.size] = v
        p_mean = np.nanmean(arr, axis=0)
        spectrum_probs[f"{jv:g}"] = {"p_mean": p_mean, "p_std": np.nanstd(arr, axis=0, ddof=1) if len(per_draw_vectors) > 1 else np.zeros_like(p_mean)}
        rank_rows.append({"J": float(jv), "R_mean": float(np.mean(per_draw_rank)), "R_std": float(np.std(per_draw_rank, ddof=1)) if len(per_draw_rank) > 1 else 0.0, "cut": cut, "m_spectrum": m_spec})
        print(f"spectrum: J={jv:.1f}, R={rank_rows[-1]['R_mean']:.2f}", flush=True)

    _write_csv(out_dir / "spectrum_rank_summary.csv", rank_rows)
    np.savez_compressed(
        out_dir / "spectrum_probs.npz",
        **{f"J{j:g}_mean": spectrum_probs[f"{j:g}"]["p_mean"] for j in spec_js},
        **{f"J{j:g}_std": spectrum_probs[f"{j:g}"]["p_std"] for j in spec_js},
    )
    return rank_rows, spectrum_probs


def main() -> None:
    args = _parse_args()
    cfg = _resolve_config(args)
    out_dir = cfg["out_dir"].resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.plot_only):
        if bool(args.run_convergence):
            plot_convergence_sv_vs_m(_load_csv(out_dir / "convergence_summary.csv"), out_dir / "fig_convergence_sv_vs_m_prl")
        if bool(args.run_spectrum_rank):
            loaded = np.load(out_dir / "spectrum_probs.npz")
            probs: dict[str, dict[str, np.ndarray]] = {}
            for key in loaded.files:
                if key.endswith("_mean"):
                    jlabel = key[1:-5]
                    probs[jlabel] = {"p_mean": np.asarray(loaded[key]), "p_std": np.asarray(loaded[f"J{jlabel}_std"])}
            plot_spectrum_and_rank_vs_j(rank_rows=_load_csv(out_dir / "spectrum_rank_summary.csv"), spectrum_probs=probs, out_stem=out_dir / "fig_spectrum_and_rank_vs_j_prl")
        print(f"Wrote figures under: {out_dir}", flush=True)
        return

    if bool(args.run_convergence):
        summary = run_convergence(cfg, args)
        plot_convergence_sv_vs_m(summary, out_dir / "fig_convergence_sv_vs_m_prl")
    if bool(args.run_spectrum_rank):
        rank_rows, spectrum_probs = run_spectrum_rank(cfg, args)
        plot_spectrum_and_rank_vs_j(rank_rows=rank_rows, spectrum_probs=spectrum_probs, out_stem=out_dir / "fig_spectrum_and_rank_vs_j_prl")
    print(f"Wrote results to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
