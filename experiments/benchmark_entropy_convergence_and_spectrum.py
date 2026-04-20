#!/usr/bin/env python3
"""Figure-2 benchmark: probe-budget convergence and weighted-centered singular-value spectra.

This script reuses the same metric pipeline as ``benchmark_entropy_vs_j_by_cut.py``:
exact probe evaluation -> weighted ``V`` with ``beta=1`` -> past-row centering -> SVD diagnostics.
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
from mqt.yaqs.characterization.process_tensors.diagnostics.probe import ProbeSet, analyze_v_matrix, sample_split_cut_probes
from mqt.yaqs.characterization.process_tensors.diagnostics.v_matrix_diag import (
    build_weighted_v_matrix,
    center_past_rows,
    prepare_branch_weights,
)
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

M_GRID_DEFAULT = (4, 8, 16, 32, 64, 128) # (2, 3, 4, 5, 6, 7, 8, 10, 12, 16)
CONVERGENCE_JS_DEFAULT = (0.0, 1.0, 2.0)
SPECTRUM_JS_DEFAULT = (0.4, 1.0, 2.0)


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


def _weighted_centered_probabilities(
    *,
    probe_set,
    op: MPO,
    sim_params: AnalogSimParams,
    psi0: np.ndarray,
    parallel: bool,
) -> np.ndarray:
    rho8_ij, weights_ij, _ = evaluate_exact_probe_set_with_diagnostics(
        probe_set=probe_set,
        operator=op,
        sim_params=sim_params,
        initial_psi=psi0,
        parallel=parallel,
    )
    w_clean, _ = prepare_branch_weights(weights_ij, log_warnings=False)
    v_w = build_weighted_v_matrix(rho8_ij, w_clean, BRANCH_WEIGHT_BETA)
    v_c = center_past_rows(v_w)
    ana = analyze_v_matrix(v_w, v_c)
    s = np.asarray(ana["singular_values"], dtype=np.float64)
    p = s * s
    p_sum = float(np.sum(p))
    if p_sum <= 0.0:
        return np.zeros(0, dtype=np.float64)
    return p / p_sum


def _aggregate_variable_length(vectors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not vectors:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    max_len = max(int(v.size) for v in vectors)
    arr = np.full((len(vectors), max_len), np.nan, dtype=np.float64)
    for i, v in enumerate(vectors):
        arr[i, : v.size] = v
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0, ddof=1) if len(vectors) > 1 else np.zeros(max_len)


def _permute_probe_set_once(probe_set: ProbeSet, rng: np.random.Generator) -> ProbeSet:
    """Randomly permute past/future probe lists once, preserving iid prefix slices."""
    ip = rng.permutation(int(probe_set.past_features.shape[0]))
    jf = rng.permutation(int(probe_set.future_features.shape[0]))
    return ProbeSet(
        cut=int(probe_set.cut),
        k=int(probe_set.k),
        past_features=np.asarray(probe_set.past_features[ip], dtype=np.float32),
        future_features=np.asarray(probe_set.future_features[jf], dtype=np.float32),
        past_pairs=[probe_set.past_pairs[int(i)] for i in ip],
        past_cut_meas=[probe_set.past_cut_meas[int(i)] for i in ip],
        future_prep_cut=[probe_set.future_prep_cut[int(j)] for j in jf],
        future_pairs=[probe_set.future_pairs[int(j)] for j in jf],
    )


def _metrics_from_singular_values(s: np.ndarray) -> dict[str, float]:
    s = np.asarray(s, dtype=np.float64)
    p = s * s
    ps = float(np.sum(p))
    if ps <= 0.0:
        return {
            "effective_rank": 1.0,
            "participation_ratio": 1.0,
            "r95": 1.0,
            "p1": 0.0,
            "p2": 0.0,
            "p3": 0.0,
            "p4": 0.0,
            "p5": 0.0,
        }
    q = p / ps
    c = np.cumsum(q)
    r95 = float(int(np.searchsorted(c, 0.95) + 1))
    return {
        "effective_rank": float(np.exp(-np.sum(np.clip(q, 1e-30, 1.0) * np.log(np.clip(q, 1e-30, 1.0))))),
        "participation_ratio": float(1.0 / np.sum(q * q)),
        "r95": r95,
        "p1": float(q[0]) if q.size >= 1 else 0.0,
        "p2": float(q[1]) if q.size >= 2 else 0.0,
        "p3": float(q[2]) if q.size >= 3 else 0.0,
        "p4": float(q[3]) if q.size >= 4 else 0.0,
        "p5": float(q[4]) if q.size >= 5 else 0.0,
    }


def run_benchmark(args: argparse.Namespace) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]], dict[str, dict[str, np.ndarray]]]:
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    m_values = _parse_int_list(args.m_values)
    m_ref = int(args.m_ref) if args.m_ref is not None else int(max(m_values))
    conv_js = _parse_float_list(args.convergence_js)
    spec_js = _parse_float_list(args.spectrum_js)
    if m_ref not in m_values:
        raise ValueError(f"m_ref={m_ref} must be in m_values")

    init_rng = np.random.default_rng(int(args.seed) + 77_777)
    initial_list = _list_initial_states_sys_env0(n_seeds=int(args.n_seeds), rng=init_rng)
    np.save(out_dir / "initial_states.npy", np.stack(initial_list, axis=0))

    sim_params = AnalogSimParams(dt=DT_FIXED, solver="MCWF", show_progress=False)
    detail_rows: list[dict[str, float | int]] = []

    m_max = int(max(m_values))
    for j in conv_js:
        op = MPO.ising(length=L_FIXED, J=float(j), g=G_FIXED)
        for draw in range(int(args.probe_draws)):
            draw_seed = int(args.seed) + 100_000 * int(args.cut) + 10 * int(round(10 * j)) + draw
            base_probe_set = sample_split_cut_probes(
                cut=int(args.cut),
                k=K_FIXED,
                n_pasts=int(m_max),
                n_futures=int(m_max),
                rng=np.random.default_rng(draw_seed),
            )
            perm_rng = np.random.default_rng(draw_seed + 33_337)
            full_probe_set = _permute_probe_set_once(base_probe_set, perm_rng)

            # One exact evaluation per (J, draw, initial state) at full m_max.
            full_eval: list[tuple[np.ndarray, np.ndarray]] = []
            for psi0 in initial_list:
                rho_full, w_full, _ = evaluate_exact_probe_set_with_diagnostics(
                    probe_set=full_probe_set,
                    operator=op,
                    sim_params=sim_params,
                    initial_psi=psi0,
                    parallel=bool(args.parallel),
                )
                full_eval.append((rho_full, w_full))

            # Nested prefixes from the same full draw.
            for m in m_values:
                draw_entropy: list[float] = []
                draw_rank: list[float] = []
                draw_delta: list[float] = []
                draw_erank: list[float] = []
                draw_pr: list[float] = []
                draw_r95: list[float] = []
                draw_p1: list[float] = []
                draw_p2: list[float] = []
                draw_p3: list[float] = []
                draw_p4: list[float] = []
                draw_p5: list[float] = []
                mm = int(m)
                for rho_full, w_full in full_eval:
                    rho_sub = np.asarray(rho_full[:mm, :mm, :], dtype=np.float64)
                    w_sub = np.asarray(w_full[:mm, :mm], dtype=np.float64)
                    w_clean, _ = prepare_branch_weights(w_sub, log_warnings=False)
                    v_w = build_weighted_v_matrix(rho_sub, w_clean, BRANCH_WEIGHT_BETA)
                    v_c = center_past_rows(v_w)
                    ana = analyze_v_matrix(v_w, v_c)
                    draw_entropy.append(float(ana["entropy"]))
                    draw_rank.append(float(ana["rank"]))
                    draw_delta.append(float(ana["delta_norm"]))
                    extra = _metrics_from_singular_values(np.asarray(ana["singular_values"], dtype=np.float64))
                    draw_erank.append(float(extra["effective_rank"]))
                    draw_pr.append(float(extra["participation_ratio"]))
                    draw_r95.append(float(extra["r95"]))
                    draw_p1.append(float(extra["p1"]))
                    draw_p2.append(float(extra["p2"]))
                    draw_p3.append(float(extra["p3"]))
                    draw_p4.append(float(extra["p4"]))
                    draw_p5.append(float(extra["p5"]))

                detail_rows.append(
                    {
                        "cut": int(args.cut),
                        "J": float(j),
                        "m": int(m),
                        "probe_draw": int(draw),
                        "entropy": float(np.mean(draw_entropy)),
                        "rank": float(np.mean(draw_rank)),
                        "delta_norm": float(np.mean(draw_delta)),
                        "effective_rank": float(np.mean(draw_erank)),
                        "participation_ratio": float(np.mean(draw_pr)),
                        "r95": float(np.mean(draw_r95)),
                        "p1": float(np.mean(draw_p1)),
                        "p2": float(np.mean(draw_p2)),
                        "p3": float(np.mean(draw_p3)),
                        "p4": float(np.mean(draw_p4)),
                        "p5": float(np.mean(draw_p5)),
                    }
                )

    summary_rows: list[dict[str, float | int]] = []
    grouped: dict[tuple[int, float, int], list[dict[str, float | int]]] = {}
    for row in detail_rows:
        grouped.setdefault((int(row["cut"]), float(row["J"]), int(row["m"])), []).append(row)
    ref_entropy: dict[tuple[int, float], float] = {}
    for (cut, jv, m), rs in grouped.items():
        if m == int(m_ref):
            ref_entropy[(cut, jv)] = float(np.mean([float(r["entropy"]) for r in rs]))
    for (cut, jv, m), rs in sorted(grouped.items()):
        ent = np.asarray([float(r["entropy"]) for r in rs], dtype=np.float64)
        rank = np.asarray([float(r["rank"]) for r in rs], dtype=np.float64)
        dn = np.asarray([float(r["delta_norm"]) for r in rs], dtype=np.float64)
        ref = ref_entropy.get((cut, jv), float("nan"))
        err_abs = float(abs(np.mean(ent) - ref)) if np.isfinite(ref) else float("nan")
        err_rel = err_abs / max(abs(ref), 1e-30) if np.isfinite(ref) and abs(ref) > 0 else err_abs
        summary_rows.append(
            {
                "cut": int(cut),
                "J": float(jv),
                "m": int(m),
                "entropy_mean": float(np.mean(ent)),
                "entropy_std": float(np.std(ent, ddof=1)) if ent.size > 1 else 0.0,
                "entropy_sem": float(np.std(ent, ddof=1) / np.sqrt(float(ent.size))) if ent.size > 1 else 0.0,
                "rank_mean": float(np.mean(rank)),
                "delta_norm_mean": float(np.mean(dn)),
                "entropy_err_vs_ref": float(err_rel),
            }
        )

    # Add aggregated nested metrics over draws for each (J,m).
    grouped2: dict[tuple[int, float, int], list[dict[str, float | int]]] = {}
    for row in detail_rows:
        grouped2.setdefault((int(row["cut"]), float(row["J"]), int(row["m"])), []).append(row)
    nested_summary_rows: list[dict[str, float | int]] = []
    for (cut, jv, m), rs in sorted(grouped2.items()):
        def ms(key: str) -> tuple[float, float]:
            arr = np.asarray([float(r[key]) for r in rs], dtype=np.float64)
            mu = float(np.mean(arr))
            sem = float(np.std(arr, ddof=1) / np.sqrt(float(arr.size))) if arr.size > 1 else 0.0
            return mu, sem
        entropy_mu, entropy_sem = ms("entropy")
        er_mu, er_sem = ms("effective_rank")
        pr_mu, pr_sem = ms("participation_ratio")
        r95_mu, r95_sem = ms("r95")
        p1_mu, p1_sem = ms("p1")
        p2_mu, p2_sem = ms("p2")
        p3_mu, p3_sem = ms("p3")
        p4_mu, p4_sem = ms("p4")
        p5_mu, p5_sem = ms("p5")
        nested_summary_rows.append(
            {
                "cut": int(cut),
                "J": float(jv),
                "m": int(m),
                "n_draws": int(len(rs)),
                "entropy_mean": entropy_mu,
                "entropy_sem": entropy_sem,
                "effective_rank_mean": er_mu,
                "effective_rank_sem": er_sem,
                "participation_ratio_mean": pr_mu,
                "participation_ratio_sem": pr_sem,
                "r95_mean": r95_mu,
                "r95_sem": r95_sem,
                "p1_mean": p1_mu,
                "p1_sem": p1_sem,
                "p2_mean": p2_mu,
                "p2_sem": p2_sem,
                "p3_mean": p3_mu,
                "p3_sem": p3_sem,
                "p4_mean": p4_mu,
                "p4_sem": p4_sem,
                "p5_mean": p5_mu,
                "p5_sem": p5_sem,
            }
        )

    spectrum_probs: dict[str, dict[str, np.ndarray]] = {}
    for jv in spec_js:
        op = MPO.ising(length=L_FIXED, J=float(jv), g=G_FIXED)
        per_draw_vectors: list[np.ndarray] = []
        for draw in range(int(args.spectrum_draws)):
            probe_seed = int(args.seed) + 900_000 + 100_000 * int(args.cut) + 100 * int(round(10 * jv)) + draw
            probe_set = sample_split_cut_probes(
                cut=int(args.cut),
                k=K_FIXED,
                n_pasts=int(args.m_spectrum),
                n_futures=int(args.m_spectrum),
                rng=np.random.default_rng(probe_seed),
            )
            seed_vectors = [
                _weighted_centered_probabilities(
                    probe_set=probe_set,
                    op=op,
                    sim_params=sim_params,
                    psi0=psi0,
                    parallel=bool(args.parallel),
                )
                for psi0 in initial_list
            ]
            mean_seed, _ = _aggregate_variable_length(seed_vectors)
            per_draw_vectors.append(mean_seed)
        p_mean, p_std = _aggregate_variable_length(per_draw_vectors)
        spectrum_probs[f"{jv:g}"] = {"p_mean": p_mean, "p_std": p_std}

    _write_csv(out_dir / "convergence_detail.csv", detail_rows)
    _write_csv(out_dir / "convergence_summary.csv", summary_rows)
    _write_csv(out_dir / "nested_probe_draw_metrics.csv", detail_rows)
    _write_csv(out_dir / "nested_probe_aggregated_summary.csv", nested_summary_rows)
    np.savez_compressed(
        out_dir / "spectrum_probs.npz",
        **{
            f"J{j:g}_mean": v["p_mean"]
            for j, v in {float(k): val for k, val in spectrum_probs.items()}.items()
        },
        **{
            f"J{j:g}_std": v["p_std"]
            for j, v in {float(k): val for k, val in spectrum_probs.items()}.items()
        },
    )
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "cut": int(args.cut),
                "L": L_FIXED,
                "k": K_FIXED,
                "dt": DT_FIXED,
                "g": G_FIXED,
                "branch_weight_beta": BRANCH_WEIGHT_BETA,
                "m_values": m_values,
                "convergence_js": conv_js,
                "spectrum_js": spec_js,
                "m_ref": int(m_ref),
                "m_spectrum": int(args.m_spectrum),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return detail_rows, summary_rows, spectrum_probs


def _print_convergence_interpretation(summary_rows: list[dict[str, float | int]], m_ref: int) -> None:
    def select(jv: float, m: int) -> dict[str, float | int] | None:
        for row in summary_rows:
            if abs(float(row["J"]) - jv) < 1e-12 and int(row["m"]) == int(m):
                return row
        return None

    ctrl_ok = True
    for row in summary_rows:
        if abs(float(row["J"])) < 1e-12 and float(row["entropy_mean"]) > 1e-8:
            ctrl_ok = False
            break
    print("\n=== Convergence interpretation ===", flush=True)
    print(f"- Control (J=0) numerically negligible at all m: {'yes' if ctrl_ok else 'no'}", flush=True)
    for jv in (1.0, 2.0):
        r64 = select(jv, 64)
        rref = select(jv, int(m_ref))
        if r64 is None or rref is None:
            continue
        e64 = float(r64["entropy_mean"])
        eref = float(rref["entropy_mean"])
        rel = abs(e64 - eref) / max(abs(eref), 1e-30)
        within_err = abs(e64 - eref) <= (float(r64["entropy_std"]) + float(rref["entropy_std"]))
        plateau = (rel <= 0.10) or within_err
        print(f"- J={jv:g}: |S(64)-S({m_ref})|/S({m_ref})={rel:.3f}, within combined std={'yes' if within_err else 'no'}, plateau={'yes' if plateau else 'no'}", flush=True)


def plot_from_saved(
    *,
    detail_rows: list[dict[str, str | float | int]],
    summary_rows: list[dict[str, str | float | int]],
    spectrum_probs: dict[str, dict[str, np.ndarray]],
    out_stem: Path,
    m_ref: int = 128,
    leading_modes: int = 10,
    save_supp_full_tail: bool = True,
    plot_relative_error: bool = False,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullLocator

    _configure_matplotlib_prl_figure()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
        }
    )
    plt.rcParams["lines.solid_joinstyle"] = "round"
    plt.rcParams["lines.solid_capstyle"] = "round"
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.35), constrained_layout=True, gridspec_kw={"width_ratios": [1.2, 1.0], "wspace": 0.07})
    plt.subplots_adjust(left=0.14, right=0.96, bottom=0.14, top=0.96)
    ax0, ax1 = axes
    conv_colors = {1.0: "#1f77b4", 2.0: "#d55e00"}
    conv_js = [j for j in sorted({float(r["J"]) for r in summary_rows}) if abs(j - 1.0) < 1e-12 or abs(j - 2.0) < 1e-12]
    m_vals = sorted({int(float(r["m"])) for r in summary_rows})
    interacting_vals: list[float] = []
    for jv in conv_js:
        color = conv_colors.get(jv, "#4c4c4c")
        sub = sorted(
            [r for r in summary_rows if abs(float(r["J"]) - jv) < 1e-12],
            key=lambda r: int(float(r["m"])),
        )
        xs = np.asarray([int(float(r["m"])) for r in sub], dtype=np.float64)
        mu = np.asarray([float(r["entropy_mean"]) for r in sub], dtype=np.float64)
        sem = np.asarray([float(r.get("entropy_sem", 0.0)) for r in sub], dtype=np.float64)
        ys = np.clip(mu, 1e-30, None)
        lo = np.clip(mu - sem, 1e-30, None)
        hi = np.clip(mu + sem, 1e-30, None)
        interacting_vals.extend(ys.tolist())
        ax0.semilogy(xs, ys, color=color, marker="o", lw=1.1, ms=2.8, label=rf"$J={jv:g}$")
        ax0.fill_between(xs, lo, hi, color=color, alpha=0.13, linewidth=0)
    ax0.set_xlabel(r"Probe budget $m$ ($N_p=N_f$)")
    ax0.set_ylabel(r"$S_V$")
    ax0.set_xticks(m_vals)
    if interacting_vals:
        arr = np.asarray(interacting_vals, dtype=np.float64)
        y_lo = 1e-3
        y_hi = float(np.quantile(arr, 0.98)) * 1.4
        if y_hi > y_lo:
            ax0.set_ylim(y_lo, y_hi)
    ax0.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax0.yaxis.set_minor_locator(NullLocator())
    ax0.grid(True, which="major", axis="y", alpha=0.1, linewidth=0.3)
    leg0 = ax0.legend(title=r"$J$", loc="lower right", frameon=True, fancybox=False, edgecolor="0.55", framealpha=0.85, borderpad=0.22)
    leg0.get_title().set_fontsize(5.8)
    leg0.get_frame().set_linewidth(0.35)
    for t in leg0.get_texts():
        t.set_fontsize(6.1)

    # Colorblind-safe, high-contrast palette.
    spec_colors = {"0.4": "#0072B2", "1": "#D55E00", "2": "#009E73"}
    main_mode_max = 30
    for j_label in sorted(spectrum_probs.keys(), key=lambda s: float(s)):
        p_mean = np.asarray(spectrum_probs[j_label]["p_mean"], dtype=np.float64)
        if p_mean.size == 0:
            continue
        n_keep = min(int(leading_modes), int(p_mean.size))
        idx_head = np.arange(1, n_keep + 1, dtype=np.float64)
        ax1.semilogy(
            idx_head,
            np.clip(p_mean[:n_keep], 1e-30, None),
            ls="-",
            lw=1.6,
            marker="o",
            ms=2.2,
            alpha=0.95,
            markeredgewidth=0.0,
            color=spec_colors.get(j_label, "#4c4c4c"),
            label=rf"$J={float(j_label):g}$",
        )
    ax1.set_xlabel(r"Mode index $n$")
    ax1.set_ylabel(r"$p_n$")
    ax1.set_xlim(1, 20)
    ax1.tick_params(direction="in", which="both", top=True, right=True)
    ax1.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax1.yaxis.set_minor_locator(NullLocator())
    ax1.set_ylim(1e-17, 1)
    ax1.grid(False)
    leg1 = ax1.legend(
        loc="upper right",
        frameon=False,
        fontsize=11,
        handlelength=2.5,
    )
    for s in ax1.spines.values():
        s.set_linewidth(0.9)
    for ax, tag in zip(axes, ("(a)", "(b)"), strict=True):
        ax.text(0.04, 0.955, tag, transform=ax.transAxes, va="top", ha="left", fontsize=7.3, fontweight="semibold")
        for s in ax.spines.values():
            s.set_linewidth(0.45)
    fig.savefig(out_stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    if plot_relative_error:
        fig_e, ax_e = plt.subplots(1, 1, figsize=(3.15, 2.35), constrained_layout=True)
        for jv in conv_js:
            color = conv_colors.get(jv, "#4c4c4c")
            sub = sorted(
                [r for r in summary_rows if abs(float(r["J"]) - jv) < 1e-12],
                key=lambda r: int(float(r["m"])),
            )
            if not sub:
                continue
            ref_row = next((r for r in sub if int(float(r["m"])) == int(m_ref)), sub[-1])
            ref = float(ref_row["entropy_mean"])
            xs = np.asarray([int(float(r["m"])) for r in sub], dtype=np.float64)
            ys = np.asarray([abs(float(r["entropy_mean"]) - ref) for r in sub], dtype=np.float64)
            ax_e.semilogy(xs, np.clip(ys, 1e-30, None), color=color, marker="o", lw=1.1, ms=2.8, label=rf"$J={jv:g}$")
        ax_e.set_xlabel(r"Probe budget $m$ ($N_p=N_f$)")
        ax_e.set_ylabel(r"$|S_V(m)-S_V(m_{\max})|$")
        ax_e.set_xticks(m_vals)
        ax_e.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        ax_e.yaxis.set_minor_locator(NullLocator())
        ax_e.grid(True, which="major", axis="y", alpha=0.1, linewidth=0.3)
        ax_e.legend(frameon=True, fancybox=False, edgecolor="0.55", framealpha=0.85, borderpad=0.22)
        for s in ax_e.spines.values():
            s.set_linewidth(0.45)
        err_stem = out_stem.with_name(f"{out_stem.name}_error")
        fig_e.savefig(err_stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight", pad_inches=0.02)
        fig_e.savefig(err_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig_e)

    if save_supp_full_tail:
        fig_s, ax_s = plt.subplots(1, 1, figsize=(3.15, 2.35), constrained_layout=True)
        for j_label in sorted(spectrum_probs.keys(), key=lambda s: float(s)):
            p_mean = np.asarray(spectrum_probs[j_label]["p_mean"], dtype=np.float64)
            if p_mean.size == 0:
                continue
            idx = np.arange(1, p_mean.size + 1, dtype=np.float64)
            ax_s.semilogy(
                idx[:main_mode_max],
                np.clip(p_mean[:main_mode_max], 1e-30, None),
                ls="-",
                lw=1.6,
                marker="o",
                ms=2.2,
                alpha=0.95,
                markeredgewidth=0.0,
                color=spec_colors.get(j_label, "#4c4c4c"),
                label=rf"$J={float(j_label):g}$",
            )
        ax_s.set_xlabel(r"Mode index $n$")
        ax_s.set_ylabel(r"$p_n$")
        ax_s.set_xlim(1, 20)
        ax_s.tick_params(direction="in", which="both", top=True, right=True)
        ax_s.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        ax_s.yaxis.set_minor_locator(NullLocator())
        ax_s.set_ylim(1e-17, 1)
        ax_s.grid(False)
        ax_s.legend(
            loc="upper right",
            frameon=False,
            fontsize=11,
            handlelength=2.5,
        )
        for s in ax_s.spines.values():
            s.set_linewidth(0.9)
        supp_stem = out_stem.with_name("fig_spectrum_full_tail_supp")
        fig_s.savefig(supp_stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight", pad_inches=0.02)
        fig_s.savefig(supp_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig_s)


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
    p.add_argument("--m-ref", type=int, default=None, help="Reference budget; defaults to max(m-values).")
    p.add_argument("--convergence-js", type=str, default=",".join(str(v) for v in CONVERGENCE_JS_DEFAULT))
    p.add_argument("--spectrum-js", type=str, default=",".join(str(v) for v in SPECTRUM_JS_DEFAULT))
    p.add_argument("--m-spectrum", type=int, default=128)
    p.add_argument("--spectrum-draws", type=int, default=3)
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--benchmark-only", action="store_true")
    p.add_argument("--leading-modes", type=int, default=10)
    p.add_argument("--no-supp-full-tail", dest="supp_full_tail", action="store_false")
    p.set_defaults(supp_full_tail=True)
    p.add_argument("--detail-csv", type=Path, default=None)
    p.add_argument("--summary-csv", type=Path, default=None)
    p.add_argument("--spectrum-npz", type=Path, default=None)
    p.add_argument("--plot-relative-error", action="store_true", help="Also save |S(m)-S(m_max)| plot.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    m_values = _parse_int_list(args.m_values)
    m_ref = int(args.m_ref) if args.m_ref is not None else int(max(m_values))

    if bool(args.plot_only):
        detail_csv = args.detail_csv if args.detail_csv is not None else out_dir / "convergence_detail.csv"
        summary_csv = args.summary_csv if args.summary_csv is not None else out_dir / "convergence_summary.csv"
        spectrum_npz = args.spectrum_npz if args.spectrum_npz is not None else out_dir / "spectrum_probs.npz"
        detail_raw = _load_csv(detail_csv)
        rows_raw = _load_csv(summary_csv)
        loaded = np.load(spectrum_npz)
        probs: dict[str, dict[str, np.ndarray]] = {}
        for key in loaded.files:
            if not key.startswith("J") or not key.endswith("_mean"):
                continue
            jlabel = key[1:-5]
            probs[jlabel] = {"p_mean": np.asarray(loaded[key], dtype=np.float64), "p_std": np.asarray(loaded[f"J{jlabel}_std"], dtype=np.float64)}
        out_stem = out_dir / "fig_convergence_and_spectrum_prl_v2"
        plot_from_saved(
            detail_rows=detail_raw,
            summary_rows=rows_raw,
            spectrum_probs=probs,
            out_stem=out_stem,
            m_ref=int(m_ref),
            leading_modes=int(args.leading_modes),
            save_supp_full_tail=bool(args.supp_full_tail),
            plot_relative_error=bool(args.plot_relative_error),
        )
        print(f"Wrote figure: {out_stem.with_suffix('.pdf')}", flush=True)
        return

    detail_rows, summary_rows, spectrum_probs = run_benchmark(args)
    _print_convergence_interpretation(summary_rows, int(m_ref))
    if not bool(args.benchmark_only):
        out_stem = out_dir / "fig_convergence_and_spectrum_prl_v2"
        plot_from_saved(
            detail_rows=detail_rows,
            summary_rows=summary_rows,
            spectrum_probs=spectrum_probs,
            out_stem=out_stem,
            m_ref=int(m_ref),
            leading_modes=int(args.leading_modes),
            save_supp_full_tail=bool(args.supp_full_tail),
            plot_relative_error=bool(args.plot_relative_error),
        )
        print(f"Wrote figure: {out_stem.with_suffix('.pdf')}", flush=True)
    print(f"Wrote tables to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()

