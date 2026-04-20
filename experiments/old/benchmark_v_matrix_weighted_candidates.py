#!/usr/bin/env python3
"""Weight-aware V-matrix candidate metrics and paired benchmark (additive; does not replace the main metric).

Implements :math:`V^{(\\beta)}_{i,(j,\\alpha)} = w_{ij}^{\\beta} [\\rho_{ij}]_\\alpha` for
:math:`\\beta \\in \\{0, 1/2, 1\\}`, keeps past-row centering as in the current pipeline, and
compares **unweighted** vs **weighted_sqrt** vs **weighted_linear** over cuts 5/10/15 with controls
``trivial`` / ``markov_ref`` and ``nm_mid``. The goal is to test whether weighting by branch
probability removes spurious early-cut / control entropy from the unweighted construction.

The primary corrected candidate is **past-centered singular-value entropy** of
``centered_past( w^{1/2} \\otimes \\rho )`` (i.e. ``weighted_sqrt`` then centered), labeled
``candidate_main`` in outputs.

Does not change the published default metric or the original exact benchmark script.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import warnings
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
    analyze_weight_scheme_pair,
    build_weighted_v_candidate_triple,
)
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

PANEL_L = 2
PANEL_K = 20
PANEL_DT = 0.1
PANEL_CUTS = (5, 10, 15)
CASES: tuple[tuple[str, float, float], ...] = (
    ("trivial", 0.0, 0.0),
    ("markov_ref", 0.0, 1.0),
    ("nm_mid", 1.0, 1.0),
)
PROBE_BUDGETS: tuple[tuple[int, int], ...] = ((32, 32), (16, 64), (64, 16))

SCHEME_KEYS = ("unweighted", "weighted_sqrt", "weighted_linear")
SCHEME_LABELS = {
    "unweighted": r"$\beta=0$ (unweighted)",
    "weighted_sqrt": r"$\beta=1/2$ (weighted_sqrt)",
    "weighted_linear": r"$\beta=1$ (weighted_linear)",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("benchmark_v_matrix_weighted_candidates_results"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--parallel", action="store_true", default=True)
    p.add_argument("--no-parallel", dest="parallel", action="store_false")
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
    x = int(base) % (2**31)
    for p in parts:
        x = (x * 1_000_003 + int(p)) % (2**31)
    return x


def _tag_int(s: str) -> int:
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


def _compact_metrics(m: dict[str, Any]) -> dict[str, float]:
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


def _control_heuristic(ent_c: float, case: str) -> dict[str, bool]:
    if case in ("trivial", "markov_ref"):
        return {"low_entropy_control": bool(ent_c < 2.5), "very_low_entropy": bool(ent_c < 1.5)}
    return {"low_entropy_control": True, "very_low_entropy": True}


def run_weighted_point(
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

    rho8, weights_ij, _traces = evaluate_exact_probe_set_with_diagnostics(
        probe_set=probe_set,
        operator=op,
        sim_params=sim_params,
        initial_psi=initial_psi,
        parallel=parallel,
    )

    np.save(out_point / "rho8_ij.npy", rho8)
    np.save(out_point / "weights_ij.npy", weights_ij)

    raw_by_name, centered_by_name, wmeta = build_weighted_v_candidate_triple(rho8, weights_ij)
    if wmeta.get("weight_data_invalid"):
        warnings.warn(
            f"Weight sanity: invalid entries nan={wmeta['nan_count']} "
            f"inf={wmeta['posinf_count'] + wmeta['neginf_count']}",
            stacklevel=2,
        )
    (out_point / "weight_preparation_meta.json").write_text(json.dumps(_json_safe(wmeta), indent=2))

    per_scheme: dict[str, Any] = {}
    for sk in SCHEME_KEYS:
        v_raw = raw_by_name[sk]
        v_c = centered_by_name[sk]
        np.save(out_point / f"V_{sk}_raw.npy", v_raw)
        np.save(out_point / f"V_{sk}_centered_past.npy", v_c)
        an = analyze_weight_scheme_pair(v_raw, v_c, scheme_name=sk)
        np.save(out_point / f"singular_values_{sk}_centered_past.npy", an["singular_values_centered_past"])
        np.save(out_point / f"singular_values_{sk}_raw.npy", an["singular_values_raw"])
        np.save(out_point / f"row_distances_{sk}_centered_past.npy", an["pairwise_row_distances_centered_past"])
        ent_c = float(an["metrics_centered_past"]["entropy_sv"])
        per_scheme[sk] = {
            "metrics_raw": _compact_metrics(an["metrics_raw"]),
            "metrics_centered_past": _compact_metrics(an["metrics_centered_past"]),
            "delta_norm": float(an["delta_norm"]),
            "row_distance_raw": an["row_distance_raw"],
            "row_distance_centered_past": an["row_distance_centered_past"],
            "control_heuristic": _control_heuristic(ent_c, case_name),
        }

    candidate_main = {
        "definition": "entropy_sv of centered_past( w_ij^0.5 * rho_flat ) — same as weighted_sqrt centered_past",
        "metrics_centered_past": per_scheme["weighted_sqrt"]["metrics_centered_past"],
        "metrics_raw": per_scheme["weighted_sqrt"]["metrics_raw"],
        "delta_norm": per_scheme["weighted_sqrt"]["delta_norm"],
    }

    summary: dict[str, Any] = {
        "case_name": case_name,
        "cut": int(cut),
        "cut_comparison_mode": cut_comparison_mode,
        "J": float(J),
        "g": float(g),
        "L": int(L),
        "k": int(k),
        "dt": float(dt),
        "n_pasts": int(n_pasts),
        "n_futures": int(n_futures),
        "weight_preparation": wmeta,
        "variants": per_scheme,
        "candidate_main": candidate_main,
    }
    (out_point / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2))
    return summary


def _write_global_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    flat: list[dict[str, Any]] = []
    for r in rows:
        f: dict[str, Any] = {k: v for k, v in r.items() if k != "variants" and k != "candidate_main"}
        for sk in SCHEME_KEYS:
            prefix = f"{sk}_"
            vdict = r.get("variants", {}).get(sk, {})
            if not vdict:
                continue
            f[prefix + "entropy_centered"] = vdict["metrics_centered_past"]["entropy_sv"]
            f[prefix + "rank_centered"] = vdict["metrics_centered_past"]["rank_tol"]
            f[prefix + "entropy_raw"] = vdict["metrics_raw"]["entropy_sv"]
            f[prefix + "delta_norm"] = vdict["delta_norm"]
            f[prefix + "frobenius_centered"] = vdict["metrics_centered_past"]["frobenius"]
        if "candidate_main" in r:
            f["candidate_main_entropy"] = r["candidate_main"]["metrics_centered_past"]["entropy_sv"]
        flat.append(f)

    keys: list[str] = []
    seen: set[str] = set()
    for row in flat:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(flat)


def plot_sv_decay_centered(
    out_path: Path,
    *,
    title: str,
    s_u: np.ndarray,
    s_s: np.ndarray,
    s_l: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for s, lab in zip([s_u, s_s, s_l], SCHEME_KEYS, strict=True):
        s = np.asarray(s, dtype=np.float64).ravel()
        if s.size == 0:
            continue
        ax.semilogy(np.arange(1, len(s) + 1), np.maximum(s / max(float(s[0]), 1e-30), 1e-16), label=SCHEME_LABELS[lab])
    ax.set_xlabel("singular index")
    ax.set_ylabel(r"$\sigma_k / \sigma_1$")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig_both(fig, out_path)
    plt.close(fig)


def plot_grouped_cases_entropy(
    out_path: Path,
    *,
    title: str,
    rows_at_point: list[dict[str, Any]],
) -> None:
    """Grouped bars: x = case, groups = scheme, value = centered entropy."""
    import matplotlib.pyplot as plt

    cases_order = ["trivial", "markov_ref", "nm_mid"]
    x = np.arange(len(cases_order))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, sk in enumerate(SCHEME_KEYS):
        vals = []
        for cn in cases_order:
            rr = [r for r in rows_at_point if r.get("case_name") == cn]
            vals.append(float(rr[0]["variants"][sk]["metrics_centered_past"]["entropy_sv"]) if rr else float("nan"))
        ax.bar(x + (i - 1) * width, vals, width, label=SCHEME_LABELS[sk])
    ax.set_xticks(x)
    ax.set_xticklabels(cases_order)
    ax.set_ylabel("entropy (centered)")
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    savefig_both(fig, out_path)
    plt.close(fig)


def plot_grouped_cases_rank_delta(
    out_path: Path,
    *,
    title: str,
    rows_at_point: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    cases_order = ["trivial", "markov_ref", "nm_mid"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(cases_order))
    width = 0.25
    for i, sk in enumerate(SCHEME_KEYS):
        ranks = []
        dns = []
        for cn in cases_order:
            rr = [r for r in rows_at_point if r.get("case_name") == cn]
            if rr:
                ranks.append(float(rr[0]["variants"][sk]["metrics_centered_past"]["rank_tol"]))
                dns.append(float(rr[0]["variants"][sk]["delta_norm"]))
            else:
                ranks.append(float("nan"))
                dns.append(float("nan"))
        axes[0].bar(x + (i - 1) * width, ranks, width, label=SCHEME_LABELS[sk])
        axes[1].bar(x + (i - 1) * width, dns, width, label=SCHEME_LABELS[sk])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cases_order)
    axes[0].set_ylabel("rank (tol)")
    axes[0].set_title("Centered rank")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cases_order)
    axes[1].set_ylabel("delta_norm")
    axes[1].set_title("delta_norm (same variant)")
    for ax in axes:
        ax.legend(fontsize=6)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    savefig_both(fig, out_path)
    plt.close(fig)


def plot_entropy_vs_cut(
    out_path: Path,
    *,
    title: str,
    rows: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    cuts = sorted({int(r["cut"]) for r in rows})
    fig, ax = plt.subplots(figsize=(7, 4))
    for sk in SCHEME_KEYS:
        ys = []
        for c in cuts:
            rr = [r for r in rows if int(r["cut"]) == c]
            ys.append(float(np.mean([float(r["variants"][sk]["metrics_centered_past"]["entropy_sv"]) for r in rr])))
        ax.plot(cuts, ys, marker="o", label=SCHEME_LABELS[sk])
    ax.set_xlabel("cut")
    ax.set_ylabel("centered entropy (mean over probe budgets)")
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig_both(fig, out_path)
    plt.close(fig)


def plot_budget_heatmap(
    out_path: Path,
    *,
    title: str,
    rows: list[dict[str, Any]],
    case_name: str,
    metric_fn: Any,
) -> None:
    """Rows = scheme, cols = probe budgets; metric averaged over cuts and both probe modes."""
    import matplotlib.pyplot as plt

    budget_labels = ["32×32", "16×64", "64×16"]
    budgets = [(32, 32), (16, 64), (64, 16)]
    mat = np.full((len(SCHEME_KEYS), len(budgets)), np.nan)
    for i, sk in enumerate(SCHEME_KEYS):
        for j, (np_, nf_) in enumerate(budgets):
            rr = [
                r
                for r in rows
                if r.get("case_name") == case_name
                and int(r["n_pasts"]) == np_
                and int(r["n_futures"]) == nf_
            ]
            if rr:
                mat[i, j] = float(np.mean([metric_fn(r, sk) for r in rr]))
    fig, ax = plt.subplots(figsize=(7, 3.2))
    im = ax.imshow(mat, aspect="auto", origin="lower")
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels(budget_labels)
    ax.set_yticks(range(len(SCHEME_KEYS)))
    ax.set_yticklabels([SCHEME_LABELS[s] for s in SCHEME_KEYS])
    ax.set_xlabel("probe budget")
    ax.set_ylabel("weighting")
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    savefig_both(fig, out_path)
    plt.close(fig)


def aggregate_stats(all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-scheme aggregates for scoring and interpretation."""
    out: dict[str, Any] = {}
    for sk in SCHEME_KEYS:
        trivial = [r for r in all_rows if r.get("case_name") == "trivial"]
        markov = [r for r in all_rows if r.get("case_name") == "markov_ref"]
        nm = [r for r in all_rows if r.get("case_name") == "nm_mid"]

        def ent(rows: list[dict[str, Any]]) -> list[float]:
            return [float(r["variants"][sk]["metrics_centered_past"]["entropy_sv"]) for r in rows]

        e_triv = ent(trivial)
        e_mar = ent(markov)
        e_nm = ent(nm)
        e_ctrl = [float(r["variants"][sk]["metrics_centered_past"]["entropy_sv"]) for r in trivial + markov]
        out[sk] = {
            "mean_entropy_trivial": float(np.mean(e_triv)) if e_triv else float("nan"),
            "mean_entropy_markov_ref": float(np.mean(e_mar)) if e_mar else float("nan"),
            "max_entropy_control": float(np.max(e_ctrl)) if e_ctrl else float("nan"),
            "mean_entropy_control": float(np.mean(e_ctrl)) if e_ctrl else float("nan"),
            "mean_entropy_nm_mid": float(np.mean(e_nm)) if e_nm else float("nan"),
            "max_entropy_nm_mid": float(np.max(e_nm)) if e_nm else float("nan"),
        }
    return out


def score_scheme(stats: dict[str, float]) -> float:
    """Higher is better: low control entropy, low max control, retain nm_mid signal."""
    mc = float(stats["mean_entropy_control"])
    xc = float(stats["max_entropy_control"])
    nm = float(stats["mean_entropy_nm_mid"])
    if np.isnan(mc):
        return float("-inf")
    return -mc - 0.25 * xc + 0.4 * nm


def print_interpretation(all_rows: list[dict[str, Any]], agg: dict[str, Any]) -> None:
    print("\n=== Weighted candidate benchmark — interpretation ===\n")
    a = agg
    print("Aggregate (centered entropy, all cuts/budgets/modes):")
    for sk in SCHEME_KEYS:
        s = a[sk]
        print(
            f"  {sk}: mean_control={s['mean_entropy_control']:.4f}, max_control={s['max_entropy_control']:.4f}, "
            f"mean_nm_mid={s['mean_entropy_nm_mid']:.4f}"
        )

    scores = {sk: score_scheme(a[sk]) for sk in SCHEME_KEYS}
    winner = max(scores, key=lambda k: scores[k])
    print("\nAutomatic selection (score = -mean_control - 0.25*max_control + 0.4*mean_nm_mid):")
    for sk in SCHEME_KEYS:
        print(f"  {sk}: score={scores[sk]:.4f}")
    print(f"\n  Recommended candidate (this run): {winner}  (candidate_main uses weighted_sqrt)\n")

    print("1) Do weighted variants reduce spurious entropy in `trivial` (mean centered entropy)?")
    print(
        f"   trivial: unweighted={a['unweighted']['mean_entropy_trivial']:.4f}, "
        f"sqrt={a['weighted_sqrt']['mean_entropy_trivial']:.4f}, linear={a['weighted_linear']['mean_entropy_trivial']:.4f}"
    )
    print(
        f"2) Same for `markov_ref`: "
        f"{a['unweighted']['mean_entropy_markov_ref']:.4f} / {a['weighted_sqrt']['mean_entropy_markov_ref']:.4f} / {a['weighted_linear']['mean_entropy_markov_ref']:.4f}"
    )
    print(
        f"3) Best preservation of `nm_mid` signal (mean entropy): "
        f"{a['unweighted']['mean_entropy_nm_mid']:.4f} / {a['weighted_sqrt']['mean_entropy_nm_mid']:.4f} / {a['weighted_linear']['mean_entropy_nm_mid']:.4f}"
    )
    print(
        "4) Does weighted_sqrt outperform unweighted and linear on composite score? "
        f"sqrt vs unweighted: {'yes' if scores['weighted_sqrt'] > scores['unweighted'] else 'no'}, "
        f"sqrt vs linear: {'yes' if scores['weighted_sqrt'] > scores['weighted_linear'] else 'no'}"
    )
    print("5) Consistency across cuts/budgets: see fig_entropy_vs_cut and budget heatmaps in --out-dir.\n")


def main() -> None:
    args = _parse_args()
    configure_matplotlib()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    initial_psi = _initial_state_zero(PANEL_L)
    all_summaries: list[dict[str, Any]] = []

    for mode in ("cut_comparison_fresh_probes", "cut_comparison_shared_probe_source"):
        mode_dir = out_root / mode
        for np_, nf_ in PROBE_BUDGETS:
            pools = None
            if mode == "cut_comparison_shared_probe_source":
                rng_p = np.random.default_rng(_mix_seed(int(args.seed), 17, _tag_int("pools") % 10_000, np_, nf_))
                pools = draw_shared_probe_pools(k=PANEL_K, n_pasts=np_, n_futures=nf_, rng=rng_p)

            for cut in PANEL_CUTS:
                if mode == "cut_comparison_fresh_probes":
                    for case_name, J, g in CASES:
                        rng = np.random.default_rng(
                            _mix_seed(int(args.seed), 10007, cut, _tag_int(case_name) % 10_000, np_, nf_)
                        )
                        probe_set = sample_split_cut_probes(
                            cut=cut, k=PANEL_K, n_pasts=np_, n_futures=nf_, rng=rng
                        )
                        out_point = mode_dir / f"cut_{cut}" / f"np{np_}_nf{nf_}" / case_name
                        summ = run_weighted_point(
                            case_name=case_name,
                            J=J,
                            g=g,
                            cut=cut,
                            k=PANEL_K,
                            dt=PANEL_DT,
                            L=PANEL_L,
                            n_pasts=np_,
                            n_futures=nf_,
                            probe_set=probe_set,
                            initial_psi=initial_psi,
                            parallel=bool(args.parallel),
                            out_point=out_point,
                            cut_comparison_mode=mode,
                        )
                        all_summaries.append(summ)
                else:
                    assert pools is not None
                    for case_name, J, g in CASES:
                        probe_set = probe_set_from_shared_pools(pools, cut=cut)
                        out_point = mode_dir / f"cut_{cut}" / f"np{np_}_nf{nf_}" / case_name
                        summ = run_weighted_point(
                            case_name=case_name,
                            J=J,
                            g=g,
                            cut=cut,
                            k=PANEL_K,
                            dt=PANEL_DT,
                            L=PANEL_L,
                            n_pasts=np_,
                            n_futures=nf_,
                            probe_set=probe_set,
                            initial_psi=initial_psi,
                            parallel=bool(args.parallel),
                            out_point=out_point,
                            cut_comparison_mode=mode,
                        )
                        all_summaries.append(summ)

    _write_global_csv(out_root / "summary_all.csv", all_summaries)
    (out_root / "summary_all.json").write_text(json.dumps(_json_safe(all_summaries), indent=2))

    agg = aggregate_stats(all_summaries)
    (out_root / "aggregate_stats.json").write_text(json.dumps(_json_safe(agg), indent=2))

    if not args.skip_plots:
        plot_dir = out_root / "figures"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # SV decay per saved point
        for mode in ("cut_comparison_fresh_probes", "cut_comparison_shared_probe_source"):
            for np_, nf_ in PROBE_BUDGETS:
                for cut in PANEL_CUTS:
                    for case_name, _, _ in CASES:
                        pdir = out_root / mode / f"cut_{cut}" / f"np{np_}_nf{nf_}" / case_name
                        if not (pdir / "summary.json").exists():
                            continue
                        s_u = np.load(pdir / "singular_values_unweighted_centered_past.npy")
                        s_s = np.load(pdir / "singular_values_weighted_sqrt_centered_past.npy")
                        s_l = np.load(pdir / "singular_values_weighted_linear_centered_past.npy")
                        plot_sv_decay_centered(
                            plot_dir / f"sv_decay_{mode}_c{cut}_np{np_}_nf{nf_}_{case_name}",
                            title=f"SV decay {mode} cut={cut} {case_name} np={np_} nf={nf_}",
                            s_u=s_u,
                            s_s=s_s,
                            s_l=s_l,
                        )

        for mode in ("cut_comparison_fresh_probes", "cut_comparison_shared_probe_source"):
            for np_, nf_ in PROBE_BUDGETS:
                for cut in PANEL_CUTS:
                    rows_here = [
                        r
                        for r in all_summaries
                        if r.get("cut_comparison_mode") == mode
                        and int(r["cut"]) == cut
                        and int(r["n_pasts"]) == np_
                        and int(r["n_futures"]) == nf_
                    ]
                    if len(rows_here) < 3:
                        continue
                    plot_grouped_cases_entropy(
                        plot_dir / f"cases_entropy_{mode}_c{cut}_np{np_}_nf{nf_}",
                        title=f"Centered entropy by case ({mode} cut={cut} np={np_} nf={nf_})",
                        rows_at_point=rows_here,
                    )
                    plot_grouped_cases_rank_delta(
                        plot_dir / f"cases_rank_delta_{mode}_c{cut}_np{np_}_nf{nf_}",
                        title=f"Rank & delta_norm ({mode} cut={cut} np={np_} nf={nf_})",
                        rows_at_point=rows_here,
                    )

        for mode in ("cut_comparison_fresh_probes", "cut_comparison_shared_probe_source"):
            for case_name, _, _ in CASES:
                rows_c = [r for r in all_summaries if r.get("cut_comparison_mode") == mode and r.get("case_name") == case_name]
                if rows_c:
                    plot_entropy_vs_cut(
                        plot_dir / f"entropy_vs_cut_{mode}_{case_name}",
                        title=f"Centered entropy vs cut ({mode}) {case_name}",
                        rows=rows_c,
                    )

        for case_name, _, _ in CASES:
            plot_budget_heatmap(
                plot_dir / f"heatmap_budget_entropy_{case_name}",
                title=f"Mean centered entropy over cuts & modes — {case_name}",
                rows=all_summaries,
                case_name=case_name,
                metric_fn=lambda r, sk: float(r["variants"][sk]["metrics_centered_past"]["entropy_sv"]),
            )
            plot_budget_heatmap(
                plot_dir / f"heatmap_budget_rank_{case_name}",
                title=f"Mean centered rank — {case_name}",
                rows=all_summaries,
                case_name=case_name,
                metric_fn=lambda r, sk: float(r["variants"][sk]["metrics_centered_past"]["rank_tol"]),
            )

    print_interpretation(all_summaries, agg)
    print(f"Done. Outputs under {out_root.resolve()}")


if __name__ == "__main__":
    main()
