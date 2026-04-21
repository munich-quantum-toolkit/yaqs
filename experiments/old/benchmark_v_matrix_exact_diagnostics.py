#!/usr/bin/env python3
"""Diagnostic-only benchmark for the split-cut exact V-matrix pipeline.

We already inspected the code path and found no obvious split-cut indexing bug, but there is a
strong suspicion that low-probability branches can terminate early while still contributing
unweighted final states to the V-matrix benchmark, and that centering over past rows may amplify
this for asymmetric cuts. This script measures whether early-cut pathology is dominated by ignored
branch weights, centering, asymmetric probe sampling, or cut-to-cut probe realization differences.

Does **not** change the main estimator or physics; diagnostic outputs only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from mqt.yaqs.characterization.process_tensors.diagnostics.exact import evaluate_exact_probe_set_with_diagnostics
from mqt.yaqs.characterization.process_tensors.diagnostics.probe import ProbeSet, sample_split_cut_probes
from mqt.yaqs.characterization.process_tensors.diagnostics.probe_shared import (
    draw_shared_probe_pools,
    probe_set_from_shared_pools,
)
from mqt.yaqs.characterization.process_tensors.diagnostics.v_matrix_diag import (
    apply_weight_mask_and_scales,
    build_v_variants,
    correlation_terminated_vs_entry_norm,
    correlation_weight_vs_entry_norm,
    delta_norm_of_centered,
    entry_centered_block_norms,
    matrix_diagnostic_metrics,
    pairwise_row_distances,
    row_distance_summary,
    summarize_weight_by_index,
    traces_flat_to_ij_arrays,
    weight_threshold_fractions,
)
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

# Fixed diagnostic panel defaults
DIAG_L = 2
DIAG_K = 20
DIAG_DT = 0.1
DIAG_CUTS = (5, 10, 15)
CASES: tuple[tuple[str, float, float], ...] = (
    ("trivial", 0.0, 0.0),
    ("markov_ref", 0.0, 1.0),
    ("nm_mid", 1.0, 1.0),
)
PROBE_BUDGETS_MAIN: tuple[tuple[int, int], ...] = ((32, 32), (16, 64), (64, 16))

ASYMMETRY_GRID: tuple[tuple[int, int], ...] = (
    (16, 16),
    (16, 32),
    (16, 64),
    (32, 16),
    (32, 32),
    (32, 64),
    (64, 16),
    (64, 32),
    (64, 64),
)
ASYMMETRY_CUTS = (5, 10, 15)
ASYMMETRY_CASES: tuple[tuple[str, float, float], ...] = (("trivial", 0.0, 0.0), ("markov_ref", 0.0, 1.0))

WEIGHT_THRESHOLDS = (1e-12, 1e-10, 1e-8, 1e-6)
MASK_THRESHOLDS = (1e-12, 1e-10, 1e-8)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("benchmark_v_matrix_exact_diagnostics_results"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=1, dest="n_seeds")
    p.add_argument("--parallel", action="store_true", default=True)
    p.add_argument("--no-parallel", dest="parallel", action="store_false")
    p.add_argument("--skip-main-panel", action="store_true", help="Skip main panel + cut-comparison modes.")
    p.add_argument("--skip-asymmetry", action="store_true", help="Skip (n_p,n_f) asymmetry grid.")
    p.add_argument("--skip-plots", action="store_true")
    return p.parse_args()


def configure_matplotlib() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.labelsize": 9,
            "font.size": 9,
        }
    )


def savefig_both(fig: Any, path_stem: Path) -> None:
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")


def _initial_state_zero(L: int) -> np.ndarray:
    psi = np.zeros(2**L, dtype=np.complex128)
    psi[0] = 1.0 + 0.0j
    return psi


def _mix_seed(base: int, *parts: int) -> int:
    """Deterministic seed mixing (avoid salted :func:`hash`)."""
    x = int(base) % (2**31)
    for p in parts:
        x = (x * 1_000_003 + int(p)) % (2**31)
    return x


def _tag_int(s: str) -> int:
    """Stable small int from string (for RNG diversification)."""
    return zlib.crc32(s.encode("utf-8")) & 0x7FFFFFFF


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, bool | float | int | str) or obj is None:
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


def _metrics_dict_from_result(m: dict[str, Any]) -> dict[str, float]:
    return {
        "frobenius": float(m["frobenius"]),
        "frobenius_sq": float(m["frobenius_sq"]),
        "entropy_sv": float(m["entropy_sv"]),
        "rank_tol": int(m["rank_tol"]),
        "participation_ratio": float(m["participation_ratio"]),
        "mean_row_norm": float(m["mean_row_norm"]),
        "std_row_norm": float(m["std_row_norm"]),
        "mean_col_norm": float(m["mean_col_norm"]),
        "std_col_norm": float(m["std_col_norm"]),
    }


def run_diagnostic_point(
    *,
    case_name: str,
    J: float,
    g: float,
    cut: int,
    k: int,
    dt: float,
    L: int,
    n_pasts: int,
    n_futures: int,
    probe_set: ProbeSet,
    initial_psi: np.ndarray,
    parallel: bool,
    out_point: Path,
    cut_comparison_mode: str,
) -> dict[str, Any]:
    out_point.mkdir(parents=True, exist_ok=True)
    op = MPO.ising(length=int(L), J=float(J), g=float(g))
    sim_params = AnalogSimParams(dt=float(dt), solver="MCWF", show_progress=False)

    pauli_xyz, weights_ij, traces = evaluate_exact_probe_set_with_diagnostics(
        probe_set=probe_set,
        operator=op,
        sim_params=sim_params,
        initial_psi=initial_psi,
        parallel=parallel,
    )
    n_p, n_f = int(n_pasts), int(n_futures)
    fields = traces_flat_to_ij_arrays(traces, n_p=n_p, n_f=n_f)

    np.save(out_point / "pauli_xyz_ij.npy", pauli_xyz)
    np.save(out_point / "weights.npy", weights_ij)
    np.save(out_point / "terminated_early.npy", fields["terminated_early"])
    np.save(out_point / "break_step.npy", fields["break_step"])
    np.save(out_point / "num_steps_completed.npy", fields["num_steps_completed"])
    np.save(out_point / "min_step_prob.npy", fields["min_step_prob"])
    np.save(out_point / "any_prob_skipped_renormalize.npy", fields["any_prob_skipped_renormalize"])

    wflat = weights_ij.reshape(-1)
    frac_early = float(np.mean(fields["terminated_early"]))
    frac_skip_rn = float(np.mean(fields["any_prob_skipped_renormalize"]))
    w_thr = weight_threshold_fractions(weights_ij, list(WEIGHT_THRESHOLDS))
    mean_w = float(np.mean(wflat))
    med_w = float(np.median(wflat))

    variants = build_v_variants(pauli_xyz)
    v_raw = variants["V_raw"]
    v_cp = variants["V_centered_past"]
    v_cg = variants["V_centered_global"]
    v_crc = variants["V_centered_rowcol"]

    np.save(out_point / "V_raw.npy", v_raw)
    np.save(out_point / "V_centered_past.npy", v_cp)
    np.save(out_point / "V_centered_global.npy", v_cg)
    np.save(out_point / "V_centered_rowcol.npy", v_crc)

    m_raw = matrix_diagnostic_metrics(v_raw, "V_raw")
    m_cp = matrix_diagnostic_metrics(v_cp, "V_centered_past")
    m_cg = matrix_diagnostic_metrics(v_cg, "V_centered_global")
    m_crc = matrix_diagnostic_metrics(v_crc, "V_centered_rowcol")

    np.save(out_point / "singular_values_raw.npy", m_raw["singular_values"])
    np.save(out_point / "singular_values_centered.npy", m_cp["singular_values"])

    d_raw = pairwise_row_distances(v_raw)
    d_cp = pairwise_row_distances(v_cp)
    np.save(out_point / "row_distances_raw.npy", d_raw)
    np.save(out_point / "row_distances_centered.npy", d_cp)
    sum_raw = row_distance_summary(v_raw)
    sum_cp = row_distance_summary(v_cp)

    entry_rho_norm = np.linalg.norm(pauli_xyz, axis=2)
    centered_blk = entry_centered_block_norms(v_cp, n_f, d=3)

    corr_w = correlation_weight_vs_entry_norm(weights_ij, centered_blk)
    corr_te = correlation_terminated_vs_entry_norm(fields["terminated_early"], centered_blk)

    delta_norm = delta_norm_of_centered(v_raw, v_cp)

    weighted_diag: dict[str, dict[str, float]] = {}
    for mthr in MASK_THRESHOLDS:
        wvars = apply_weight_mask_and_scales(v_raw, weights_ij, mask_threshold=mthr)
        for key, mat in wvars.items():
            tag = f"{key}_thr{mthr:g}"
            weighted_diag[tag] = _metrics_dict_from_result(matrix_diagnostic_metrics(mat, tag))

    summary: dict[str, Any] = {
        "case_name": case_name,
        "cut": int(cut),
        "cut_comparison_mode": cut_comparison_mode,
        "J": float(J),
        "g": float(g),
        "L": int(L),
        "k": int(k),
        "dt": float(dt),
        "n_pasts": n_p,
        "n_futures": n_f,
        "fraction_terminated_early": frac_early,
        "fraction_any_prob_skipped_renormalize": frac_skip_rn,
        **w_thr,
        "mean_cumulative_weight": mean_w,
        "median_cumulative_weight": med_w,
        "metrics_V_raw": _metrics_dict_from_result(m_raw),
        "metrics_V_centered_past": _metrics_dict_from_result(m_cp),
        "metrics_V_centered_global": _metrics_dict_from_result(m_cg),
        "metrics_V_centered_rowcol": _metrics_dict_from_result(m_crc),
        "row_distance_raw": sum_raw,
        "row_distance_centered_past": sum_cp,
        "delta_norm_centered_past": float(delta_norm),
        "corr_weight_vs_centered_block_norm": float(corr_w),
        "corr_terminated_early_vs_centered_block_norm": float(corr_te),
        "summarize_weights": summarize_weight_by_index(weights_ij),
        "mean_centered_block_norm_by_past": [float(x) for x in np.mean(centered_blk, axis=1)],
        "mean_centered_block_norm_by_future": [float(x) for x in np.mean(centered_blk, axis=0)],
        "weighted_variants": weighted_diag,
        "heuristic_control_ok": bool(
            case_name in ("trivial", "markov_ref")
            and frac_early < 0.25
            and m_cp["entropy_sv"] < 3.0
        ),
    }
    (out_point / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2))
    return summary


def plot_point_heatmaps(
    *,
    out_plot: Path,
    weights: np.ndarray,
    terminated: np.ndarray,
    entry_norms: np.ndarray,
    s_raw: np.ndarray,
    s_centered: np.ndarray,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    out_plot.parent.mkdir(parents=True, exist_ok=True)
    w = np.clip(weights, 1e-30, None)
    logw = np.log10(w)

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8))
    fig.suptitle(title, fontsize=11)

    im0 = axes[0, 0].imshow(logw, aspect="auto", origin="lower")
    axes[0, 0].set_title(r"$\log_{10}$(cumulative weight)")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(terminated.astype(float), aspect="auto", origin="lower", vmin=0, vmax=1)
    axes[0, 1].set_title("terminated_early")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[1, 0].imshow(entry_norms, aspect="auto", origin="lower")
    axes[1, 0].set_title(r"$\|\rho_8\|$ per entry")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    ax = axes[1, 1]
    idx = np.arange(1, len(s_raw) + 1)
    ax.plot(idx, s_raw / max(float(s_raw[0]), 1e-30), label="raw", marker="o", ms=3)
    ax.plot(idx, s_centered / max(float(s_centered[0]), 1e-30), label="centered (past)", marker="s", ms=3)
    ax.set_yscale("log")
    ax.set_xlabel("singular index")
    ax.set_ylabel("σ / σ_1")
    ax.legend()
    ax.set_title("Singular value decay (normalized)")
    ax.grid(True, alpha=0.3)

    for ax in axes.ravel():
        ax.set_xlabel("future j")
    axes[0, 0].set_ylabel("past i")
    axes[1, 0].set_ylabel("past i")

    plt.tight_layout()
    savefig_both(fig, out_plot)
    plt.close(fig)


def plot_cut_comparison_lines(rows: list[dict[str, Any]], out_path: Path, mode: str) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return
    cuts = sorted({int(r["cut"]) for r in rows})
    cases = sorted({str(r["case_name"]) for r in rows})

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    fig.suptitle(f"Cut comparison ({mode})", fontsize=11)

    def series_for(case: str, getter: Any) -> list[float]:
        out: list[float] = []
        for c in cuts:
            rr = [r for r in rows if r["case_name"] == case and int(r["cut"]) == c]
            if not rr:
                out.append(float("nan"))
            else:
                out.append(float(getter(rr[0])))
        return out

    for case in cases:
        axes[0, 0].plot(
            cuts,
            series_for(case, lambda r: r["metrics_V_centered_past"]["entropy_sv"]),
            marker="o",
            label=case,
        )
        axes[0, 1].plot(
            cuts,
            series_for(case, lambda r: r["metrics_V_raw"]["entropy_sv"]),
            marker="o",
            label=case,
        )
        axes[1, 0].plot(cuts, series_for(case, lambda r: r["fraction_terminated_early"]), marker="o", label=case)
        axes[1, 1].plot(cuts, series_for(case, lambda r: r["median_cumulative_weight"]), marker="o", label=case)

    axes[0, 0].set_ylabel("entropy (centered)")
    axes[0, 0].set_title("Centered entropy vs cut")
    axes[0, 1].set_ylabel("entropy (raw)")
    axes[0, 1].set_title("Raw entropy vs cut")
    axes[1, 0].set_ylabel("fraction early")
    axes[1, 0].set_title("Early termination vs cut")
    axes[1, 1].set_ylabel("median weight")
    axes[1, 1].set_title("Median cumulative weight vs cut")
    for ax in axes.ravel():
        ax.set_xlabel("cut")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    plt.tight_layout()
    savefig_both(fig, out_path)
    plt.close(fig)


def plot_asymmetry_heatmaps(
    grid_data: dict[tuple[int, int], dict[str, Any]],
    out_path: Path,
    *,
    case_name: str,
    cut: int,
) -> None:
    import matplotlib.pyplot as plt

    np_list = sorted({t[0] for t in grid_data})
    nf_list = sorted({t[1] for t in grid_data})
    mat_ent = np.full((len(np_list), len(nf_list)), np.nan)
    mat_early = np.full((len(np_list), len(nf_list)), np.nan)
    mat_dn = np.full((len(np_list), len(nf_list)), np.nan)
    for (i, j), d in grid_data.items():
        ii = np_list.index(i)
        jj = nf_list.index(j)
        mat_ent[ii, jj] = float(d["metrics_V_centered_past"]["entropy_sv"])
        mat_early[ii, jj] = float(d["fraction_terminated_early"])
        mat_dn[ii, jj] = float(d["delta_norm_centered_past"])

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    fig.suptitle(f"Asymmetry grid case={case_name} cut={cut}", fontsize=11)
    for ax, mat, title in zip(
        axes,
        [mat_ent, mat_early, mat_dn],
        ["entropy (centered)", "fraction terminated early", "delta_norm"],
        strict=True,
    ):
        im = ax.imshow(mat, aspect="auto", origin="lower")
        ax.set_xticks(range(len(nf_list)))
        ax.set_xticklabels([str(x) for x in nf_list])
        ax.set_yticks(range(len(np_list)))
        ax.set_yticklabels([str(x) for x in np_list])
        ax.set_xlabel("n_futures")
        ax.set_ylabel("n_pasts")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    savefig_both(fig, out_path)
    plt.close(fig)


def print_interpretation(all_rows: list[dict[str, Any]]) -> None:
    """Print compact answers to diagnostic questions from aggregated summaries."""
    print("\n=== Diagnostic interpretation (from this run) ===\n")

    mp = [r for r in all_rows if r.get("benchmark_segment") == "main_panel"]
    asym = [r for r in all_rows if r.get("benchmark_segment") == "asymmetry_grid"]

    def _avg(key: str, rows: list[dict[str, Any]], *, nested: tuple[str, str] | None = None) -> float:
        if not rows:
            return float("nan")
        if nested:
            a, b = nested
            return float(np.mean([float(r[a][b]) for r in rows]))
        return float(np.mean([float(r[key]) for r in rows]))

    print("1) Association of bad early-cut behavior with early termination / tiny weights:")
    for mode in ("cut_comparison_fresh_probes", "cut_comparison_shared_probe_source"):
        sub = [r for r in mp if r.get("cut_comparison_mode") == mode and r.get("case_name") == "markov_ref"]
        by_cut: dict[int, list[dict[str, Any]]] = {}
        for r in sub:
            by_cut.setdefault(int(r["cut"]), []).append(r)
        if not by_cut:
            continue
        print(f"   [{mode}] markov_ref — mean fraction_terminated_early by cut:")
        for c in sorted(by_cut):
            rows = by_cut[c]
            fe = _avg("fraction_terminated_early", rows)
            mw = _avg("median_cumulative_weight", rows)
            print(f"      cut={c}: mean frac_early={fe:.4f}, mean median_weight={mw:.3e}")
        print(
            "   If frac_early rises as cut decreases (5 vs 15) while median_weight falls, truncation/weight "
            "effects are strongly implicated.\n"
        )

    print("2) Raw V vs centering (main panel, all cases/modes):")
    if mp:
        er = _avg("entropy_sv", mp, nested=("metrics_V_raw", "entropy_sv"))
        ec = _avg("entropy_sv", mp, nested=("metrics_V_centered_past", "entropy_sv"))
        dn = _avg("delta_norm_centered_past", mp)
        print(f"   Mean entropy_sv: raw={er:.4f}, centered_past={ec:.4f}; mean delta_norm={dn:.4f}")
        print(
            "   Large raw entropy with modest delta_norm suggests the pathology is already in V; "
            "large delta_norm means centering dominates the Frobenius energy.\n"
        )

    print("3) Control sensitivity to n_pasts vs n_futures (asymmetry grid, trivial + markov_ref):")
    if asym:
        for case in ("trivial", "markov_ref"):
            sub = [r for r in asym if r.get("case_name") == case]
            if not sub:
                continue
            ents = np.array([float(r["metrics_V_centered_past"]["entropy_sv"]) for r in sub])
            nfs = np.array([int(r["n_futures"]) for r in sub])
            nps = np.array([int(r["n_pasts"]) for r in sub])
            print(
                f"   {case}: centered entropy range [{float(np.min(ents)):.3f}, {float(np.max(ents)):.3f}], "
                f"corr(entropy, n_pasts)={float(np.corrcoef(nps, ents)[0,1]) if len(sub)>2 else float('nan'):.3f}, "
                f"corr(entropy, n_futures)={float(np.corrcoef(nfs, ents)[0,1]) if len(sub)>2 else float('nan'):.3f}"
            )
        print()

    print("4) Cut dependence vs probe realization (shared vs fresh):")
    for case in ("markov_ref", "trivial"):
        fresh = [r for r in mp if r.get("case_name") == case and r.get("cut_comparison_mode") == "cut_comparison_fresh_probes"]
        shared = [r for r in mp if r.get("case_name") == case and r.get("cut_comparison_mode") == "cut_comparison_shared_probe_source"]
        if not fresh or not shared:
            continue

        def std_ent_across_cuts(rows: list[dict[str, Any]]) -> float:
            by_cut: dict[int, list[float]] = {}
            for r in rows:
                by_cut.setdefault(int(r["cut"]), []).append(float(r["metrics_V_centered_past"]["entropy_sv"]))
            vals = [float(np.mean(v)) for v in by_cut.values()]
            return float(np.std(vals)) if len(vals) > 1 else 0.0

        sf = std_ent_across_cuts(fresh)
        ss = std_ent_across_cuts(shared)
        print(
            f"   {case}: std over cuts of (mean centered entropy per cut, averaged over probe budgets) — "
            f"fresh={sf:.4f}, shared={ss:.4f} "
            f"(if shared ≪ fresh, shared probe source reduces cut-to-cut noise).\n"
        )

    print("5) Masking / sqrt(weight) / linear weight (inspect per-point summary.json weighted_variants):")
    mr_mp = [r for r in mp if r.get("case_name") == "markov_ref"]
    if mr_mp:
        r0 = mr_mp[0]
        wv = r0.get("weighted_variants", {})
        base = float(r0["metrics_V_centered_past"]["entropy_sv"])
        print(f"   Example markov_ref point: baseline centered entropy={base:.4f}")
        for k in sorted(wv.keys()):
            if "weighted_sqrt" in k or "masked" in k:
                print(f"      {k}: entropy_sv={float(wv[k]['entropy_sv']):.4f}")
    print()


def main() -> None:
    args = _parse_args()
    configure_matplotlib()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if int(args.n_seeds) != 1:
        print("Warning: diagnostic benchmark uses n_seeds=1; overriding.")
    L, k, dt = DIAG_L, DIAG_K, DIAG_DT
    initial_psi = _initial_state_zero(L)

    all_rows: list[dict[str, Any]] = []

    # --- Main panel: two cut-comparison modes ---
    if not args.skip_main_panel:
        for mode in ("cut_comparison_fresh_probes", "cut_comparison_shared_probe_source"):
            mode_dir = out_root / "main_panel" / mode
            mode_rows: list[dict[str, Any]] = []
            for np_, nf_ in PROBE_BUDGETS_MAIN:
                for case_name, J, g in CASES:
                    pools = None
                    if mode == "cut_comparison_shared_probe_source":
                        rng_p = np.random.default_rng(_mix_seed(int(args.seed), 17, _tag_int(case_name) % 10_000, np_, nf_))
                        pools = draw_shared_probe_pools(k=k, n_pasts=np_, n_futures=nf_, rng=rng_p)

                    for cut in DIAG_CUTS:
                        if mode == "cut_comparison_fresh_probes":
                            rng = np.random.default_rng(_mix_seed(int(args.seed), 10007, cut, _tag_int(case_name) % 10_000, np_, nf_))
                            probe_set = sample_split_cut_probes(
                                cut=cut, k=k, n_pasts=np_, n_futures=nf_, rng=rng
                            )
                        else:
                            assert pools is not None
                            probe_set = probe_set_from_shared_pools(pools, cut=cut)

                        out_point = mode_dir / f"cut_{cut}" / f"{case_name}" / f"np{np_}_nf{nf_}"
                        summary = run_diagnostic_point(
                            case_name=case_name,
                            J=J,
                            g=g,
                            cut=cut,
                            k=k,
                            dt=dt,
                            L=L,
                            n_pasts=np_,
                            n_futures=nf_,
                            probe_set=probe_set,
                            initial_psi=initial_psi,
                            parallel=bool(args.parallel),
                            out_point=out_point,
                            cut_comparison_mode=mode,
                        )
                        summary["benchmark_segment"] = "main_panel"
                        summary["n_pasts"] = np_
                        summary["n_futures"] = nf_
                        all_rows.append(summary)
                        mode_rows.append(summary)

                        if not args.skip_plots:
                            pauli_xyz = np.load(out_point / "pauli_xyz_ij.npy")
                            w = np.load(out_point / "weights.npy")
                            term = np.load(out_point / "terminated_early.npy")
                            s_raw = np.load(out_point / "singular_values_raw.npy")
                            s_c = np.load(out_point / "singular_values_centered.npy")
                            en = np.linalg.norm(pauli_xyz, axis=2)
                            plot_point_heatmaps(
                                out_plot=out_point / "fig_diagnostics",
                                weights=w,
                                terminated=term,
                                entry_norms=en,
                                s_raw=s_raw,
                                s_centered=s_c,
                                title=f"{mode} {case_name} cut={cut} np={np_} nf={nf_}",
                            )

            plot_cut_comparison_lines(
                mode_rows,
                out_root / f"fig_cut_comparison_{mode}",
                mode=mode,
            )

    # --- Asymmetry grid ---
    asymmetry_index: dict[tuple[str, int], dict[tuple[int, int], dict[str, Any]]] = {}
    if not args.skip_asymmetry:
        asym_dir = out_root / "asymmetry_grid"
        for case_name, J, g in ASYMMETRY_CASES:
            for cut in ASYMMETRY_CUTS:
                grid: dict[tuple[int, int], dict[str, Any]] = {}
                for np_, nf_ in ASYMMETRY_GRID:
                    rng = np.random.default_rng(_mix_seed(int(args.seed), 9001, cut, _tag_int(case_name) % 10_000, np_, nf_))
                    probe_set = sample_split_cut_probes(
                        cut=cut, k=k, n_pasts=np_, n_futures=nf_, rng=rng
                    )
                    out_point = asym_dir / f"cut_{cut}" / case_name / f"np{np_}_nf{nf_}"
                    summary = run_diagnostic_point(
                        case_name=case_name,
                        J=J,
                        g=g,
                        cut=cut,
                        k=k,
                        dt=dt,
                        L=L,
                        n_pasts=np_,
                        n_futures=nf_,
                        probe_set=probe_set,
                        initial_psi=initial_psi,
                        parallel=bool(args.parallel),
                        out_point=out_point,
                        cut_comparison_mode="asymmetry_grid",
                    )
                    summary["benchmark_segment"] = "asymmetry_grid"
                    all_rows.append(summary)
                    grid[(np_, nf_)] = summary
                asymmetry_index[(case_name, cut)] = grid
                if not args.skip_plots:
                    plot_asymmetry_heatmaps(
                        grid,
                        asym_dir / f"fig_asymmetry_{case_name}_cut_{cut}",
                        case_name=case_name,
                        cut=cut,
                    )

    global_csv = out_root / "summary_all.csv"
    _write_summary_csv(global_csv, all_rows)
    (out_root / "summary_all.json").write_text(json.dumps(_json_safe(all_rows), indent=2))

    print_interpretation(all_rows)
    print(f"\nWrote {global_csv}")


def _flatten_summary_row(r: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested metrics for CSV."""
    flat: dict[str, Any] = {}
    for k, v in r.items():
        if k == "metrics_V_raw" and isinstance(v, dict):
            for k2, v2 in v.items():
                flat[f"raw_{k2}"] = v2
        elif k == "metrics_V_centered_past" and isinstance(v, dict):
            for k2, v2 in v.items():
                flat[f"centered_{k2}"] = v2
        elif k in ("weighted_variants", "summarize_weights", "metrics_V_centered_global", "metrics_V_centered_rowcol"):
            continue
        elif isinstance(v, dict | list):
            continue
        else:
            flat[k] = v
    return flat


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    flat_rows = [_flatten_summary_row(r) for r in rows]
    keys: list[str] = []
    seen: set[str] = set()
    for row in flat_rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(flat_rows)


if __name__ == "__main__":
    main()
