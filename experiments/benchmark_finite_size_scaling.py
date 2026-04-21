#!/usr/bin/env python3
"""Finite-size scaling benchmark: peak entropy, cut profiles, and 99% spectral rank vs bath size L.

Uses the same pipeline as the entropy benchmarks: exact probes, linear branch weights (β=1),
Pauli (x,y,z) embedding, past-row–centered :math:`V`, and singular-value entropy :math:`S_V`
from the **full** spectrum of the centered matrix (no SVD truncation for the reported entropy).

Outputs:
- ``finite_size_detail.csv``: rows per (L, J, c) with mean entropy over seeds, plus
  ``peak_entropy``, ``peak_cut``, ``n_0_99`` (from :math:`V` at ``peak_cut``, mean over seeds).
- ``fig_finite_size_scaling.pdf`` / ``.png``: 3-panel figure (peak vs :math:`1/L`, profile at J=1,
  :math:`n_{0.99}` vs L).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from mqt.yaqs.characterization.process_tensors.diagnostics.exact import evaluate_exact_probe_set_with_diagnostics
from mqt.yaqs.characterization.process_tensors.diagnostics.probe import sample_split_cut_probes
from mqt.yaqs.characterization.process_tensors.diagnostics.v_matrix_diag import (
    build_weighted_v_matrix,
    center_past_rows,
    prepare_branch_weights,
)
from mqt.yaqs.characterization.process_tensors.surrogates.utils import _random_pure_state
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

# Fixed physics (match main entropy benchmarks unless overridden).
K_FIXED = 20
DT_FIXED = 0.1
G_FIXED = 1.0
BRANCH_WEIGHT_BETA = 1.0

L_LIST_DEFAULT = (2, 4, 6, 8, 10)
J_LIST_DEFAULT = (0.5, 1.0, 2.0)
# Panel 2: cut profile at fixed J for these L only.
PROFILE_L_DEFAULT = (2, 6, 10)
PROFILE_J_DEFAULT = 1.0


def _spectral_entropy_and_n99(v_centered: np.ndarray, *, mass: float = 0.99) -> tuple[float, int]:
    """Full SVD on centered V: normalized :math:`p_\\ell \\propto s_\\ell^2`, entropy and rank at ``mass``."""
    vc = np.asarray(v_centered, dtype=np.float64)
    s = np.linalg.svd(vc, compute_uv=False).astype(np.float64)
    p = s * s
    tot = float(np.sum(p))
    if tot <= 0.0:
        return 0.0, 0
    p = p / tot
    q = np.clip(p, 1e-300, 1.0)
    entropy = float(-np.sum(q * np.log(q)))
    cum = np.cumsum(p)
    target = float(mass)
    n99 = int(np.searchsorted(cum, target, side="left") + 1)
    return entropy, min(n99, int(p.size))


def _weighted_centered_entropy_n99(
    *,
    probe_set,
    op: MPO,
    sim_params: AnalogSimParams,
    psi0: np.ndarray,
    parallel: bool,
) -> tuple[float, int]:
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
    return _spectral_entropy_and_n99(v_c)


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
            "legend.fontsize": 8,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.5,
        }
    )


def plot_finite_size_figure(
    rows: list[dict[str, str | float | int]],
    *,
    out_stem: Path,
    profile_j: float,
    profile_ls: tuple[int, ...],
) -> None:
    import matplotlib.pyplot as plt

    _configure_matplotlib()
    # Aggregate peaks: (L, J) -> peak_entropy, peak_cut, n_0_99
    peaks: dict[tuple[int, float], tuple[float, int, float]] = {}
    for r in rows:
        key = (int(float(r["L"])), float(r["J"]))
        peaks[key] = (
            float(r["peak_entropy"]),
            int(float(r["peak_cut"])),
            float(r["n_0_99"]),
        )

    js = sorted({float(r["J"]) for r in rows})
    ls_all = sorted({int(float(r["L"])) for r in rows})

    def _col(j: float) -> str:
        if abs(j - 0.5) < 1e-9:
            return "#0072B2"
        if abs(j - 1.0) < 1e-9:
            return "#D55E00"
        if abs(j - 2.0) < 1e-9:
            return "#009E73"
        return "#333333"

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), constrained_layout=True, gridspec_kw={"wspace": 0.28})
    ax0, ax1, ax2 = axes

    # (a) S_V^max vs 1/L
    for jv in js:
        xs: list[float] = []
        ys: list[float] = []
        for L in ls_all:
            t = peaks.get((L, jv))
            if t is None:
                continue
            xs.append(1.0 / L)
            ys.append(t[0])
        if xs:
            pts = sorted(zip(xs, ys), key=lambda p: p[0])
            ax0.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                marker="o",
                lw=1.25,
                color=_col(jv),
                label=rf"$J={jv:g}$",
            )
    ax0.set_xlabel(r"$1/L$")
    ax0.set_ylabel(r"$\max_c S_V(c)$")
    ax0.legend(frameon=False, loc="best", fontsize=8)
    ax0.text(0.04, 0.95, "(a)", transform=ax0.transAxes, va="top", ha="left", fontsize=9, fontweight="semibold")

    # (b) S_V(c) for J=profile_j, L in profile_ls
    j_key = profile_j
    for L in profile_ls:
        sub = [r for r in rows if int(float(r["L"])) == L and abs(float(r["J"]) - j_key) < 1e-9]
        sub = sorted(sub, key=lambda r: int(float(r["c"])))
        if not sub:
            continue
        cc = [int(float(r["c"])) for r in sub]
        sv = [float(r["entropy"]) for r in sub]
        ax1.plot(cc, sv, marker="o", lw=1.2, ms=3.0, label=rf"$L={L}$")
    ax1.set_xlabel(r"Cut $c$")
    ax1.set_ylabel(r"$S_V(c)$")
    ax1.legend(title=rf"$J={j_key:g}$", frameon=False, loc="best", fontsize=8)
    ax1.text(0.04, 0.95, "(b)", transform=ax1.transAxes, va="top", ha="left", fontsize=9, fontweight="semibold")

    # (c) n_0.99 vs L
    for jv in js:
        xs: list[int] = []
        ys: list[float] = []
        for L in ls_all:
            t = peaks.get((L, jv))
            if t is None:
                continue
            xs.append(L)
            ys.append(t[2])
        if xs:
            ax2.plot(xs, ys, marker="s", lw=1.25, color=_col(jv), label=rf"$J={jv:g}$")
    ax2.set_xlabel(r"$L$")
    ax2.set_ylabel(r"$n_{0.99}$")
    ax2.legend(frameon=False, loc="best", fontsize=8)
    ax2.text(0.04, 0.95, "(c)", transform=ax2.transAxes, va="top", ha="left", fontsize=9, fontweight="semibold")

    for ax in axes:
        ax.grid(True, alpha=0.12, linewidth=0.4)

    fig.savefig(out_stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def run_benchmark(args: argparse.Namespace) -> tuple[list[dict[str, float | int]], Path]:
    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    L_list = tuple(_parse_int_tuple(args.l_values))
    J_list = tuple(_parse_float_tuple(args.j_values))
    cuts = list(range(1, int(args.k) + 1))
    n_seeds = int(args.n_seeds)
    init_rng = np.random.default_rng(int(args.seed) + 91_091)

    rows: list[dict[str, float | int]] = []

    for L in L_list:
        initial_list = _initial_states(L=L, n_seeds=n_seeds, rng=init_rng)
        np.save(out_dir / f"initial_states_L{L}.npy", np.stack(initial_list, axis=0))
        for J in J_list:
            op = MPO.ising(length=int(L), J=float(J), g=float(args.g))
            sim_params = AnalogSimParams(dt=float(args.dt), solver="MCWF", show_progress=False)
            # mean entropy per cut over seeds
            mean_ent: dict[int, float] = {c: 0.0 for c in cuts}
            n99_at_peak: list[float] = []

            for c in cuts:
                probe_rng = np.random.default_rng(int(args.seed) + 1_000 * L + 50 * int(10 * J) + c)
                probe_set = sample_split_cut_probes(
                    cut=int(c),
                    k=int(args.k),
                    n_pasts=int(args.n_pasts),
                    n_futures=int(args.n_futures),
                    rng=probe_rng,
                )
                ents: list[float] = []
                for psi0 in initial_list:
                    e, _ = _weighted_centered_entropy_n99(
                        probe_set=probe_set,
                        op=op,
                        sim_params=sim_params,
                        psi0=psi0,
                        parallel=bool(args.parallel),
                    )
                    ents.append(e)
                mean_ent[c] = float(np.mean(ents))

            peak_cut = max(mean_ent, key=lambda cc: mean_ent[cc])
            peak_entropy = float(mean_ent[peak_cut])

            probe_rng_peak = np.random.default_rng(int(args.seed) + 1_000 * L + 50 * int(10 * J) + peak_cut)
            probe_peak = sample_split_cut_probes(
                cut=int(peak_cut),
                k=int(args.k),
                n_pasts=int(args.n_pasts),
                n_futures=int(args.n_futures),
                rng=probe_rng_peak,
            )
            for psi0 in initial_list:
                _e, n99 = _weighted_centered_entropy_n99(
                    probe_set=probe_peak,
                    op=op,
                    sim_params=sim_params,
                    psi0=psi0,
                    parallel=bool(args.parallel),
                )
                n99_at_peak.append(float(n99))
            n99_mean = float(np.mean(n99_at_peak))

            for c in cuts:
                rows.append(
                    {
                        "L": int(L),
                        "J": float(J),
                        "c": int(c),
                        "entropy": float(mean_ent[c]),
                        "peak_entropy": peak_entropy,
                        "peak_cut": int(peak_cut),
                        "n_0_99": n99_mean,
                        "k": int(args.k),
                        "dt": float(args.dt),
                        "g": float(args.g),
                        "n_pasts": int(args.n_pasts),
                        "n_futures": int(args.n_futures),
                        "n_seeds": n_seeds,
                        "branch_weight_beta": BRANCH_WEIGHT_BETA,
                    }
                )
            print(
                f"L={L} J={J:g} peak S_V={peak_entropy:.5f} at c={peak_cut} n_0.99={n99_mean:.2f}",
                flush=True,
            )

    csv_path = out_dir / "finite_size_detail.csv"
    _write_csv(csv_path, rows)
    (out_dir / "finite_size_detail.json").write_text(json.dumps(rows, indent=2))
    print(f"Wrote {csv_path}", flush=True)
    return rows, csv_path


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
    p.add_argument("--n-pasts", type=int, default=16)
    p.add_argument("--n-futures", type=int, default=16)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--profile-j", type=float, default=PROFILE_J_DEFAULT)
    p.add_argument("--profile-l", type=str, default=",".join(str(x) for x in PROFILE_L_DEFAULT))
    p.add_argument("--plot-only", action="store_true", help="Only build figure from CSV.")
    p.add_argument(
        "--detail-csv",
        type=Path,
        default=None,
        help="Input CSV for --plot-only (default: <out-dir>/finite_size_detail.csv).",
    )
    p.add_argument("--benchmark-only", action="store_true", help="Run simulations only, skip figure.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_ls = _parse_int_tuple(args.profile_l)

    if args.plot_only:
        csv_path = args.detail_csv if args.detail_csv is not None else out_dir / "finite_size_detail.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        raw = _load_csv(csv_path)
        plot_finite_size_figure(
            raw,
            out_stem=out_dir / "fig_finite_size_scaling",
            profile_j=float(args.profile_j),
            profile_ls=profile_ls,
        )
        print(f"Wrote figure: {out_dir / 'fig_finite_size_scaling.pdf'}", flush=True)
        return

    _, _ = run_benchmark(args)
    if not args.benchmark_only:
        csv_path = out_dir / "finite_size_detail.csv"
        raw = _load_csv(csv_path)
        plot_finite_size_figure(
            raw,
            out_stem=out_dir / "fig_finite_size_scaling",
            profile_j=float(args.profile_j),
            profile_ls=profile_ls,
        )
        print(f"Wrote figure: {out_dir / 'fig_finite_size_scaling.pdf'}", flush=True)
    print(f"Done. Results in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
