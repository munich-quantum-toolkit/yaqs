#!/usr/bin/env python3
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

r"""Effective memory modes: singular-value spectrum and :math:`R(J)=\\exp(S_V)`.

Uses :class:`~mqt.yaqs.MemoryCharacterizer.characterize` for exact split-cut diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
from common import (
    DT_DEFAULT,
    G_DEFAULT,
    K_DEFAULT,
    L_DEFAULT,
    characterize,
    characterizer,
    compute_spectrum,
    configure_matplotlib_prl,
    initial_states_sys_env0,
    ising_chain,
    load_csv,
    parse_float_list,
    parse_int_list,
    sim_params,
    singular_value_probs,
    write_csv,
)

from mqt.yaqs.characterization.memory.operational_memory.samples import sample_probes

DENSE_JS_DEFAULT = tuple[float, ...](round(0.05 * i, 10) for i in range(41))
RANK_TOL = 1e-16
SPECTRUM_THRESHOLD = 1e-12
LOGGER = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _j_tag(j_value: float) -> str:
    return f"{float(j_value):.12g}".replace("-", "m").replace(".", "p")


def _analyze_response(
    response_matrix: np.ndarray,
    *,
    n_pasts: int,
    n_futures: int,
) -> dict[str, np.ndarray | float | int]:
    """Validate the IXYZ response and construct matched historical diagnostics."""
    response = np.asarray(response_matrix, dtype=np.float64)
    expected_shape = (4 * int(n_futures), int(n_pasts))
    if response.shape != expected_shape or not np.all(np.isfinite(response)):
        msg = f"invalid response matrix: expected finite {expected_shape}, got {response.shape}"
        raise AssertionError(msg)

    weighted_ixyz = response.reshape(n_futures, 4, n_pasts).transpose(2, 0, 1)
    weights = weighted_ixyz[:, :, 0]
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0) or np.any(weights > 1.0 + 1e-8):
        msg = "complete retained-record weights must be finite probabilities in (0, 1]"
        raise AssertionError(msg)
    if not np.allclose(weights, weights[:, :1], rtol=1e-10, atol=1e-12):
        msg = "unitary future controls must leave each history weight constant across futures"
        raise AssertionError(msg)

    pauli_ixyz = weighted_ixyz / weights[:, :, np.newaxis]
    if not np.allclose(pauli_ixyz[:, :, 0], 1.0, rtol=1e-10, atol=1e-12):
        msg = "reconstructed identity coefficients must equal one"
        raise AssertionError(msg)
    rebuilt = (pauli_ixyz * weights[:, :, np.newaxis]).transpose(1, 2, 0).reshape(expected_shape)
    if not np.allclose(rebuilt, response, rtol=1e-12, atol=1e-14):
        msg = "IXYZ response reconstruction does not match the public result"
        raise AssertionError(msg)

    current = compute_spectrum(response, discarded_weight_threshold=SPECTRUM_THRESHOLD)
    singular_values = np.asarray(current["singular_values_full"], dtype=np.float64)
    singular_values_t = np.linalg.svd(response.T, compute_uv=False)
    if not np.allclose(singular_values, singular_values_t, rtol=1e-11, atol=1e-13):
        msg = "response spectrum changed under transpose"
        raise AssertionError(msg)

    raw_xyz_response = (pauli_ixyz[:, :, 1:] * weights[:, :, np.newaxis]).reshape(n_pasts, 3 * n_futures)
    centered_xyz_response = raw_xyz_response - raw_xyz_response.mean(axis=0, keepdims=True)
    raw_xyz = compute_spectrum(raw_xyz_response, discarded_weight_threshold=SPECTRUM_THRESHOLD)
    centered_xyz = compute_spectrum(centered_xyz_response, discarded_weight_threshold=SPECTRUM_THRESHOLD)
    return {
        "response_matrix": response,
        "pauli_ixyz": pauli_ixyz,
        "weights": weights,
        "singular_values": singular_values,
        "singular_values_raw_xyz": np.asarray(raw_xyz["singular_values_full"], dtype=np.float64),
        "singular_values_centered_xyz": np.asarray(centered_xyz["singular_values_full"], dtype=np.float64),
        "entropy": float(current["entropy"]),
        "modes": float(current["modes"]),
        "entropy_raw_xyz": float(raw_xyz["entropy"]),
        "modes_raw_xyz": float(raw_xyz["modes"]),
        "entropy_centered_xyz": float(centered_xyz["entropy"]),
        "modes_centered_xyz": float(centered_xyz["modes"]),
        "resolved_rank": int(np.sum(singular_values > RANK_TOL)),
    }


def _aggregate_variable_length(vectors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not vectors:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    max_len = max(int(v.size) for v in vectors)
    arr = np.full((len(vectors), max_len), np.nan, dtype=np.float64)
    for i, v in enumerate(vectors):
        arr[i, : v.size] = v
    return np.nanmean(arr, axis=0), (np.nanstd(arr, axis=0, ddof=1) if len(vectors) > 1 else np.zeros(max_len))


def run_benchmark(
    args: argparse.Namespace,
) -> tuple[list[dict[str, float | int | str]], dict[str, dict[str, np.ndarray]]]:
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    if bool(args.save_raw):
        raw_dir.mkdir(parents=True, exist_ok=True)
    spec_js = parse_float_list(args.spectrum_js)
    cuts = parse_int_list(args.cuts)

    init_rng = np.random.default_rng(int(args.seed) + 77_777)
    initial_list = initial_states_sys_env0(length=L_DEFAULT, n_seeds=int(args.n_seeds), rng=init_rng)
    initial_states_path = out_dir / "initial_states.npy"
    np.save(initial_states_path, np.stack(initial_list))
    mc = characterizer(parallel=bool(args.parallel), max_workers=args.max_workers)
    params = sim_params(dt=DT_DEFAULT)

    spectrum_probs: dict[str, dict[str, np.ndarray]] = {}
    rank_rows: list[dict[str, float | int | str]] = []

    for cut in cuts:
        LOGGER.info("Starting cut %d", cut)
        for jv in spec_js:
            ham = ising_chain(length=L_DEFAULT, j=float(jv), g=G_DEFAULT)
            per_draw_vectors: list[np.ndarray] = []
            per_draw_rank: list[float] = []
            per_draw_singulars: list[np.ndarray] = []
            per_draw_entropy: list[float] = []
            per_draw_modes: list[float] = []
            per_draw_entropy_raw_xyz: list[float] = []
            per_draw_modes_raw_xyz: list[float] = []
            per_draw_entropy_centered_xyz: list[float] = []
            per_draw_modes_centered_xyz: list[float] = []
            raw_responses: list[np.ndarray] = []
            raw_pauli: list[np.ndarray] = []
            raw_weights: list[np.ndarray] = []
            raw_singulars: list[np.ndarray] = []
            raw_singulars_xyz: list[np.ndarray] = []
            raw_singulars_centered_xyz: list[np.ndarray] = []
            probe_past_features: list[np.ndarray] = []
            probe_future_features: list[np.ndarray] = []
            for draw in range(int(args.spectrum_draws)):
                probe_seed = int(args.seed) + 900_000 + 100_000 * int(cut) + 100 * round(100 * jv) + draw
                probe_set = sample_probes(
                    cut=int(cut),
                    num_interventions=K_DEFAULT,
                    n_pasts=int(args.m_spectrum),
                    n_futures=int(args.m_spectrum),
                    rng=np.random.default_rng(probe_seed),
                    intervention_style="haar",
                )
                probe_past_features.append(np.asarray(probe_set.past_features))
                probe_future_features.append(np.asarray(probe_set.future_features))
                seed_vectors: list[np.ndarray] = []
                seed_ranks: list[float] = []
                seed_singulars: list[np.ndarray] = []
                seed_entropy: list[float] = []
                seed_modes: list[float] = []
                seed_entropy_raw_xyz: list[float] = []
                seed_modes_raw_xyz: list[float] = []
                seed_entropy_centered_xyz: list[float] = []
                seed_modes_centered_xyz: list[float] = []
                for psi0 in initial_list:
                    result = characterize(
                        mc,
                        ham,
                        params,
                        k=K_DEFAULT,
                        cut=int(cut),
                        n_pasts=int(args.m_spectrum),
                        n_futures=int(args.m_spectrum),
                        probe_set=probe_set,
                        initial_psi=psi0,
                    )
                    analysis = _analyze_response(
                        result.response_matrix(int(cut)),
                        n_pasts=int(args.m_spectrum),
                        n_futures=int(args.m_spectrum),
                    )
                    s = np.asarray(analysis["singular_values"], dtype=np.float64)
                    seed_vectors.append(singular_value_probs(s))
                    seed_ranks.append(float(np.sum(s > float(args.rank_tol))))
                    seed_singulars.append(s)
                    seed_entropy.append(float(analysis["entropy"]))
                    seed_modes.append(float(analysis["modes"]))
                    seed_entropy_raw_xyz.append(float(analysis["entropy_raw_xyz"]))
                    seed_modes_raw_xyz.append(float(analysis["modes_raw_xyz"]))
                    seed_entropy_centered_xyz.append(float(analysis["entropy_centered_xyz"]))
                    seed_modes_centered_xyz.append(float(analysis["modes_centered_xyz"]))
                    raw_responses.append(np.asarray(analysis["response_matrix"], dtype=np.float64))
                    raw_pauli.append(np.asarray(analysis["pauli_ixyz"], dtype=np.float64))
                    raw_weights.append(np.asarray(analysis["weights"], dtype=np.float64))
                    raw_singulars.append(s)
                    raw_singulars_xyz.append(np.asarray(analysis["singular_values_raw_xyz"], dtype=np.float64))
                    raw_singulars_centered_xyz.append(
                        np.asarray(analysis["singular_values_centered_xyz"], dtype=np.float64)
                    )
                mean_seed, _ = _aggregate_variable_length(seed_vectors)
                per_draw_vectors.append(mean_seed)
                per_draw_rank.append(float(np.mean(seed_ranks)))
                mean_singular_seed, _ = _aggregate_variable_length(seed_singulars)
                per_draw_singulars.append(mean_singular_seed)
                per_draw_entropy.append(float(np.mean(seed_entropy)))
                per_draw_modes.append(float(np.mean(seed_modes)))
                per_draw_entropy_raw_xyz.append(float(np.mean(seed_entropy_raw_xyz)))
                per_draw_modes_raw_xyz.append(float(np.mean(seed_modes_raw_xyz)))
                per_draw_entropy_centered_xyz.append(float(np.mean(seed_entropy_centered_xyz)))
                per_draw_modes_centered_xyz.append(float(np.mean(seed_modes_centered_xyz)))

            p_mean, p_std = _aggregate_variable_length(per_draw_vectors)
            s_mean, s_std = _aggregate_variable_length(per_draw_singulars)
            key = f"c{int(cut)}_J{jv:g}"
            spectrum_probs[key] = {"p_mean": p_mean, "p_std": p_std, "s_mean": s_mean, "s_std": s_std}
            raw_file = ""
            raw_sha256 = ""
            if bool(args.save_raw):
                raw_path = raw_dir / f"cut_{int(cut):02d}_J_{_j_tag(jv)}.npz"
                np.savez_compressed(
                    raw_path,
                    response_matrix=np.stack(raw_responses),
                    pauli_ixyz_ij=np.stack(raw_pauli),
                    weights_ij=np.stack(raw_weights),
                    singular_values_full=np.stack(raw_singulars),
                    singular_values_raw_xyz_full=np.stack(raw_singulars_xyz),
                    singular_values_centered_xyz_full=np.stack(raw_singulars_centered_xyz),
                    probe_past_features=np.stack(probe_past_features),
                    probe_future_features=np.stack(probe_future_features),
                )
                raw_file = str(raw_path.relative_to(out_dir))
                raw_sha256 = _sha256(raw_path)

            modes_mean = float(np.mean(per_draw_modes))
            modes_std = float(np.std(per_draw_modes, ddof=1)) if len(per_draw_modes) > 1 else 0.0
            rank_rows.append({
                "J": float(jv),
                "entropy_mean": float(np.mean(per_draw_entropy)),
                "entropy_std": float(np.std(per_draw_entropy, ddof=1)) if len(per_draw_entropy) > 1 else 0.0,
                "modes_mean": modes_mean,
                "modes_std": modes_std,
                "entropy_old_raw_xyz_mean": float(np.mean(per_draw_entropy_raw_xyz)),
                "modes_old_raw_xyz_mean": float(np.mean(per_draw_modes_raw_xyz)),
                "entropy_old_centered_xyz_mean": float(np.mean(per_draw_entropy_centered_xyz)),
                "modes_old_centered_xyz_mean": float(np.mean(per_draw_modes_centered_xyz)),
                "resolved_rank_mean": float(np.mean(per_draw_rank)),
                "resolved_rank_std": (float(np.std(per_draw_rank, ddof=1)) if len(per_draw_rank) > 1 else 0.0),
                "rank_tol": float(args.rank_tol),
                "spectrum_discarded_weight_threshold": SPECTRUM_THRESHOLD,
                "cut": int(cut),
                "m_spectrum": int(args.m_spectrum),
                "basis": "IXYZ",
                "orientation": "future_rows_history_columns",
                "weight_scope": "complete_retained_record",
                "centered": 0,
                "probability_power": 1,
                "raw_file": raw_file,
                "raw_sha256": raw_sha256,
            })
            LOGGER.info(
                "cut=%2d J=%.2f R_IXYZ=%.9f R_raw_XYZ=%.9f R_centered_XYZ=%.9f",
                cut,
                jv,
                modes_mean,
                float(np.mean(per_draw_modes_raw_xyz)),
                float(np.mean(per_draw_modes_centered_xyz)),
            )

    write_csv(out_dir / "spectrum_rank_summary.csv", rank_rows)
    payload: dict[str, np.ndarray] = {}
    for key, val in spectrum_probs.items():
        payload[f"{key}_mean"] = val["p_mean"]
        payload[f"{key}_std"] = val["p_std"]
        payload[f"{key}_smean"] = val["s_mean"]
        payload[f"{key}_sstd"] = val["s_std"]
    np.savez_compressed(out_dir / "spectrum_probs.npz", **payload)

    repo = Path(__file__).resolve().parents[1]
    manifest = {
        "integration_commit": _git_output(repo, "rev-parse", "HEAD"),
        "response_matrix_update_commit": _git_output(repo, "rev-parse", "response-matrix-update"),
        "response_matrix_tests_commit": _git_output(repo, "rev-parse", "response-matrix-tests"),
        "git_status": _git_output(repo, "status", "--short"),
        "python": sys.version,
        "settings": {
            "L": L_DEFAULT,
            "k": K_DEFAULT,
            "dt": DT_DEFAULT,
            "g": G_DEFAULT,
            "cuts": cuts,
            "J_values": spec_js,
            "n_pasts": int(args.m_spectrum),
            "n_futures": int(args.m_spectrum),
            "n_seeds": int(args.n_seeds),
            "seed": int(args.seed),
            "spectrum_draws": int(args.spectrum_draws),
            "unitary_ensemble": "haar",
            "parallel": bool(args.parallel),
            "max_workers": args.max_workers,
            "save_raw": bool(args.save_raw),
        },
        "response_matrix_contract": {
            "basis": "IXYZ",
            "orientation": "future_rows_history_columns",
            "weight_scope": "complete_retained_record",
            "centered": False,
            "probability_power": 1,
            "spectrum_discarded_weight_threshold": SPECTRUM_THRESHOLD,
            "inset_spectrum": "full_normalized_squared_singular_values",
        },
        "initial_states_sha256": _sha256(initial_states_path),
        "completed_points": len(rank_rows),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return rank_rows, spectrum_probs


def plot_from_saved(
    *,
    rank_rows: list[dict[str, str | float | int]],
    spectrum_probs: dict[str, dict[str, np.ndarray]],
    out_stem: Path,
    plot_cuts: list[int] | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    def _truncate_cmap(name: str, lo: float, hi: float, n: int = 256) -> LinearSegmentedColormap:
        base = plt.get_cmap(name)
        return LinearSegmentedColormap.from_list(f"{name}_trunc_{lo:.2f}_{hi:.2f}", base(np.linspace(lo, hi, n)))

    configure_matplotlib_prl()
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.7), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    cuts_all = sorted({int(float(r["cut"])) for r in rank_rows})
    requested_cuts = [1, 5, 10] if plot_cuts is None else plot_cuts
    cuts = [c for c in cuts_all if c in requested_cuts]
    js = sorted({float(r["J"]) for r in rank_rows})
    modes_by_point = {
        (int(float(r["cut"])), round(float(r["J"]), 12)): float(r["modes_mean"])
        for r in rank_rows
        if "modes_mean" in r and str(r["modes_mean"]).strip()
    }

    def _resolved_modes(cut: int, j_value: float, probabilities: np.ndarray) -> float:
        stored = modes_by_point.get((int(cut), round(float(j_value), 12)))
        if stored is not None:
            return stored
        total = float(np.sum(probabilities))
        if total <= 0.0:
            return 1.0
        normalized = np.clip(probabilities / total, 1e-30, 1.0)
        return float(np.exp(-np.sum(normalized * np.log(normalized))))

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
            mu_vals.append(_resolved_modes(cut, jv, p))
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
            y_all.append(_resolved_modes(cut, jv, p))
    y_hi = float(max(y_all)) if y_all else 2.0
    y_span = max(y_hi - 1.0, 0.01)
    ax.set_ylim(1.0 - 0.05 * y_span, 1.0 + 1.12 * y_span)
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
        rv = _resolved_modes(inset_cut, jv, p)
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
    p.add_argument("--out-dir", type=Path, default=Path("results/modes"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--parallel", action="store_true", default=True)
    p.add_argument("--no-parallel", dest="parallel", action="store_false")
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--cut", type=int, default=10)
    p.add_argument("--cuts", type=str, default="1,5,10,15,20")
    p.add_argument("--m-spectrum", type=int, default=64)
    p.add_argument("--spectrum-draws", type=int, default=1)
    p.add_argument("--spectrum-js", type=str, default=",".join(str(v) for v in DENSE_JS_DEFAULT))
    p.add_argument("--rank-tol", type=float, default=RANK_TOL)
    p.add_argument("--save-raw", action="store_true")
    p.add_argument("--plot-cuts", type=str, default="")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--benchmark-only", action="store_true")
    p.add_argument("--spectrum-rank-csv", type=Path, default=None)
    p.add_argument("--spectrum-npz", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cuts = parse_int_list(args.plot_cuts) if str(args.plot_cuts).strip() else None

    if bool(args.plot_only):
        spectrum_rank_csv = (
            args.spectrum_rank_csv if args.spectrum_rank_csv is not None else out_dir / "spectrum_rank_summary.csv"
        )
        spectrum_npz = args.spectrum_npz if args.spectrum_npz is not None else out_dir / "spectrum_probs.npz"
        rank_rows = load_csv(spectrum_rank_csv)
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
                    "s_mean": np.asarray(loaded[f"J{jlabel}_smean"], dtype=np.float64)
                    if f"J{jlabel}_smean" in loaded.files
                    else np.sqrt(np.clip(np.asarray(loaded[key], dtype=np.float64), 0.0, None)),
                    "s_std": np.asarray(loaded[f"J{jlabel}_sstd"], dtype=np.float64)
                    if f"J{jlabel}_sstd" in loaded.files
                    else np.zeros_like(np.asarray(loaded[key], dtype=np.float64), dtype=np.float64),
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
                "s_mean": np.asarray(loaded[f"{ctag}_smean"], dtype=np.float64)
                if has_s_mean
                else np.sqrt(np.clip(p_mean, 0.0, None)),
                "s_std": np.asarray(loaded[f"{ctag}_sstd"], dtype=np.float64)
                if has_s_std
                else np.zeros_like(p_mean, dtype=np.float64),
            }
        plot_from_saved(
            rank_rows=rank_rows,
            spectrum_probs=probs,
            out_stem=out_dir / "fig_spectrum_and_rank_vs_j_prl",
            plot_cuts=plot_cuts,
        )
        LOGGER.info("Wrote figure: %s", (out_dir / "fig_spectrum_and_rank_vs_j_prl").with_suffix(".pdf"))
        return

    rank_rows, spectrum_probs = run_benchmark(args)
    if not bool(args.benchmark_only):
        plot_from_saved(
            rank_rows=rank_rows,
            spectrum_probs=spectrum_probs,
            out_stem=out_dir / "fig_spectrum_and_rank_vs_j_prl",
            plot_cuts=plot_cuts,
        )
        LOGGER.info("Wrote figure: %s", (out_dir / "fig_spectrum_and_rank_vs_j_prl").with_suffix(".pdf"))


if __name__ == "__main__":
    main()
