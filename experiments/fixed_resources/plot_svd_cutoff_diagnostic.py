# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Plot SVD-cutoff diagnostic figure and write interpretation report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from config import PACKAGE_DIR, RELIABILITY_THRESHOLD
from run_svd_cutoff_diagnostic import CUTOFFS, METHOD_LABELS, METHODS, OUTPUT_DIR

FIG_WIDTH_MM = 180.0
FIG_HEIGHT_MM = 110.0
MM_TO_IN = 1.0 / 25.4
DPI = 600

METHOD_STYLES = {
    "hybrid_tdvp": {"color": "#E31A1C", "marker": "o"},
    "tebd_swap": {"color": "#1F78B4", "marker": "^"},
    "mpo_zipup": {"color": "#33A02C", "marker": "s"},
}
CHI_MARK = {16: "full", 32: "full"}
CHI_ALPHA = {16: 0.45, 32: 1.0}


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.5,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def plot_figure(summary: list[dict[str, Any]], spectra: dict[str, np.ndarray], out: Path) -> None:
    _apply_style()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FIG_WIDTH_MM * MM_TO_IN, FIG_HEIGHT_MM * MM_TO_IN),
    )
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    for method in METHODS:
        style = METHOD_STYLES[method]
        for chi in (16, 32):
            pts = sorted(
                [r for r in summary if r["method"] == method and int(r["chi_max"]) == chi],
                key=lambda r: float(r["tau"]),
            )
            if not pts:
                continue
            xs = np.array([float(p["tau"]) for p in pts])
            te = np.array([float(p["T_eps"]) for p in pts])
            params = np.array([float(p["peak_param_count"]) for p in pts])
            alpha = CHI_ALPHA[chi]
            ls = "-" if chi == 32 else "--"
            label = f"{METHOD_LABELS[method]}, χ={chi}"
            ax_a.plot(xs, te, color=style["color"], marker=style["marker"], alpha=alpha, linestyle=ls, label=label)
            ax_b.plot(xs, params, color=style["color"], marker=style["marker"], alpha=alpha, linestyle=ls)
            ax_c.plot(
                params,
                te,
                color=style["color"],
                marker=style["marker"],
                alpha=alpha,
                linestyle="none",
                label=label if chi == 32 else None,
            )

    for ax in (ax_a, ax_b):
        ax.set_xscale("log")
        ax.set_xlabel(r"Cutoff $\tau$ (discarded weight)")
        ax.set_xticks(list(CUTOFFS))
        ax.set_xticklabels([f"$10^{{{int(np.log10(t))}}}$" for t in CUTOFFS])
    ax_a.set_ylabel(rf"$T_{{\varepsilon}}$ ($\varepsilon={RELIABILITY_THRESHOLD:g}$)")
    ax_b.set_ylabel("Peak MPS parameter count")
    ax_c.set_xlabel("Peak MPS parameter count")
    ax_c.set_ylabel(rf"$T_{{\varepsilon}}$ ($\varepsilon={RELIABILITY_THRESHOLD:g}$)")
    ax_a.legend(frameon=False, fontsize=5.0, ncol=2, loc="best")
    ax_c.legend(frameon=False, fontsize=5.5, loc="best")

    # (d) representative spectra at χ=32 (optional if npz populated)
    plotted_spec = False
    for method in METHODS:
        key = f"{method}_chi32_tau1e-14_near_t1"
        alt = f"{method}_chi32_tau1e-14_first_long_range"
        spec = spectra.get(key, spectra.get(alt))
        if spec is None:
            continue
        style = METHOD_STYLES[method]
        s = np.asarray(spec, dtype=float)
        s = s[s > 0]
        if s.size == 0:
            continue
        ax_d.semilogy(np.arange(1, s.size + 1), s, color=style["color"], label=METHOD_LABELS[method])
        plotted_spec = True
    if plotted_spec:
        for tau in CUTOFFS:
            ax_d.axhline(np.sqrt(tau), color="0.6", linestyle=":", linewidth=0.6)
        ax_d.set_xlabel("Singular-value index")
        ax_d.set_ylabel(r"$s_i$ (χ=32, near $t{=}1$ or first LR)")
        ax_d.legend(frameon=False, fontsize=5.5)
        ax_d.text(
            0.98,
            0.05,
            r"dotted: $\sqrt{\tau}$ guides",
            transform=ax_d.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.0,
            color="0.4",
        )
    else:
        ax_d.text(
            0.5,
            0.5,
            "Spectra unavailable\n(re-run diagnostic for event-level capture)",
            transform=ax_d.transAxes,
            ha="center",
            va="center",
            fontsize=7,
            color="0.4",
        )
        ax_d.set_xticks([])
        ax_d.set_yticks([])

    for ax, lab in zip((ax_a, ax_b, ax_c, ax_d), ("(a)", "(b)", "(c)", "(d)"), strict=True):
        ax.text(0.02, 0.97, lab, transform=ax.transAxes, fontweight="bold", va="top", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        r"TFIM SVD-cutoff diagnostic (discarded-weight $\tau$; TDVP $n{=}2$)",
        fontsize=8,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "figure_svd_cutoff_diagnostic.pdf")
    fig.savefig(out / "figure_svd_cutoff_diagnostic.png", dpi=DPI)
    plt.close(fig)


def write_report(summary: list[dict[str, Any]], semantics: dict[str, Any], out: Path) -> None:
    def rows(method: str, chi: int) -> list[dict[str, Any]]:
        return sorted(
            [r for r in summary if r["method"] == method and int(r["chi_max"]) == chi],
            key=lambda r: float(r["tau"]),
        )

    lines = [
        "# SVD-cutoff diagnostic report",
        "",
        "## Cutoff semantics",
        "",
        f"- Production mode: **`{semantics['production_trunc_mode']}`**.",
        f"- τ meaning: {semantics['tau_meaning']}",
        f"- Gate-library `split_tensor` hard cutoff held fixed at `{semantics['gate_library_split_tensor_hard_cutoff']}`.",
        f"- Corrected benchmark used `svd_threshold={semantics['corrected_benchmark_svd_threshold']}`; "
        "diagnostic reference τ=1e-14 was **re-run** (not reused).",
        "- Krylov tol, TDVP n=2, χmax, Δt, and circuit held fixed.",
        "",
        "## Summary table (ε=1e-2)",
        "",
        "| method | χmax | τ | Tε | peak χ | peak params | runtime [s] | Σ disc. wt | f_cut | f_χ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(summary, key=lambda x: (x["method"], int(x["chi_max"]), float(x["tau"]))):
        runtime = r.get("runtime_s", "")
        runtime_s = f"{float(runtime):.2f}" if runtime not in ("", None) else "—"
        disc = r.get("total_discarded_weight", "")
        disc_s = f"{float(disc):.3e}" if disc not in ("", None) else "—"
        lines.append(
            f"| {r['method']} | {r['chi_max']} | {float(r['tau']):.0e} | {float(r['T_eps']):.2f} | "
            f"{r['peak_actual_chi']} | {r['peak_param_count']} | {runtime_s} | "
            f"{disc_s} | {float(r['fraction_cutoff_limited']):.2f} | "
            f"{float(r['fraction_chi_limited']):.2f} |"
        )

    # Interpretation helpers
    def te(method: str, chi: int, tau: float) -> float:
        for r in rows(method, chi):
            if abs(float(r["tau"]) - tau) < 1e-30 * max(1.0, abs(tau)):
                return float(r["T_eps"])
            if abs(np.log10(float(r["tau"])) - np.log10(tau)) < 1e-9:
                return float(r["T_eps"])
        return float("nan")

    def peak_params(method: str, chi: int, tau: float) -> int:
        for r in rows(method, chi):
            if abs(np.log10(float(r["tau"])) - np.log10(tau)) < 1e-9:
                return int(r["peak_param_count"])
        return -1

    def peak_chi(method: str, chi: int, tau: float) -> int:
        for r in rows(method, chi):
            if abs(np.log10(float(r["tau"])) - np.log10(tau)) < 1e-9:
                return int(r["peak_actual_chi"])
        return -1

    ref = 1e-14
    indist = []
    for method in METHODS:
        for chi in (16, 32):
            t14 = te(method, chi, 1e-14)
            for tau in (1e-12, 1e-9):
                dt = abs(te(method, chi, tau) - t14)
                dp = abs(peak_params(method, chi, tau) - peak_params(method, chi, 1e-14))
                indist.append((method, chi, tau, dt, dp, dt <= 0.05 and dp == 0))

    # When does cutoff reduce actual bond dims?
    first_reduce = {}
    for method in METHODS:
        for chi in (16, 32):
            base = peak_chi(method, chi, 1e-14)
            first = None
            for tau in CUTOFFS:
                if peak_chi(method, chi, tau) < base:
                    first = tau
                    break
            first_reduce[(method, chi)] = first

    # Ranking at each tau for chi=32
    ranking_lines = []
    for tau in CUTOFFS:
        order = sorted(METHODS, key=lambda m: te(m, 32, tau), reverse=True)
        ranking_lines.append(
            f"- τ={tau:g}: " + " > ".join(f"{METHOD_LABELS[m]}({te(m,32,tau):.2f})" for m in order)
        )

    # chi vs cutoff limited
    avg_f = {
        method: (
            float(np.mean([float(r["fraction_cutoff_limited"]) for r in summary if r["method"] == method])),
            float(np.mean([float(r["fraction_chi_limited"]) for r in summary if r["method"] == method])),
        )
        for method in METHODS
    }

    through_1e9 = all(
        abs(te(m, chi, tau) - te(m, chi, ref)) <= 0.1000001
        and peak_params(m, chi, tau) == peak_params(m, chi, ref)
        for m in METHODS
        for chi in (16, 32)
        for tau in (1e-12, 1e-9)
    )

    lines += [
        "",
        "## Interpretation",
        "",
        "### Are 1e-12 and 1e-9 numerically indistinguishable from 1e-14?",
        "",
    ]
    if through_1e9:
        lines.append(
            "**Yes** for this TFIM scope: horizons and peak parameter counts match the τ=1e-14 "
            "reference through τ=1e-9 (within ≤ one sampled step in Tε and exact param-count equality)."
        )
    else:
        lines.append("**Not fully.** Per-(method,χ,τ) deviations vs 1e-14:")
        for method, chi, tau, dt, dp, ok in indist:
            lines.append(
                f"- {method} χ={chi} τ={tau:g}: ΔTε={dt:.3f}, Δparams={dp}"
                + (" (match)" if ok else " (**differs**)")
            )

    lines += ["", "### At what τ does the cutoff begin reducing actual bond dimensions?", ""]
    for (method, chi), first in first_reduce.items():
        if first is None:
            lines.append(f"- {METHOD_LABELS[method]} χ={chi}: no reduction vs τ=1e-14 on this grid.")
        else:
            lines.append(f"- {METHOD_LABELS[method]} χ={chi}: first reduction at τ={first:g}.")

    lines += [
        "",
        "### Does reduced memory compensate for any loss of horizon?",
        "",
    ]
    # Compare Tε/params at aggressive vs reference for zip-up and tdvp at chi=32
    for method in ("hybrid_tdvp", "mpo_zipup"):
        t_ref, p_ref = te(method, 32, 1e-14), peak_params(method, 32, 1e-14)
        t_agg, p_agg = te(method, 32, 1e-3), peak_params(method, 32, 1e-3)
        lines.append(
            f"- {METHOD_LABELS[method]} χ=32: τ=1e-14 → Tε={t_ref:.2f}, params={p_ref}; "
            f"τ=1e-3 → Tε={t_agg:.2f}, params={p_agg} "
            f"(ΔTε={t_agg - t_ref:+.2f}, params×{p_agg / max(p_ref, 1):.2f})."
        )
    lines.append(
        "If Tε drops while params shrink, memory savings do **not** compensate for "
        "horizon loss under a fixed-ε reliability criterion."
    )

    lines += [
        "",
        "### Does the TDVP-versus-zip-up ranking change (also by parameter count)?",
        "",
        *ranking_lines,
        "",
    ]
    # param-count comparison at chi=32
    for tau in CUTOFFS:
        tdvp_t, zip_t = te("hybrid_tdvp", 32, tau), te("mpo_zipup", 32, tau)
        tdvp_p, zip_p = peak_params("hybrid_tdvp", 32, tau), peak_params("mpo_zipup", 32, tau)
        lines.append(
            f"- τ={tau:g}, χ=32 by params: TDVP params={tdvp_p} (Tε={tdvp_t:.2f}), "
            f"zip-up params={zip_p} (Tε={zip_t:.2f})."
        )

    lines += [
        "",
        "### Are runs primarily χ-limited or cutoff-limited?",
        "",
    ]
    for method, (fc, fx) in avg_f.items():
        lines.append(
            f"- {METHOD_LABELS[method]}: mean fraction cutoff-limited={fc:.2f}, χ-limited={fx:.2f}."
        )

    lines += [
        "",
        "## Stopping rule",
        "",
    ]
    if through_1e9:
        lines.append(
            "Results are unchanged through τ=1e-9 and only change at looser thresholds "
            "(1e-6 / 1e-3). **Stop here**; do not search for a τ that restores a TDVP advantage."
        )
    else:
        lines.append(
            "Differences already appear at τ≤1e-9 for some runs; see table. Still do **not** "
            "tune τ to restore a TDVP advantage."
        )

    lines += [
        "",
        "## Outputs",
        "",
        "- `svd_cutoff_trajectories.csv`",
        "- `svd_truncation_events.csv`",
        "- `svd_cutoff_summary.csv`",
        "- `representative_spectra.npz`",
        "- `figure_svd_cutoff_diagnostic.{pdf,png}`",
        "- `cutoff_semantics.json`",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'report.md'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    out = args.output_dir.resolve()
    summary = _load_csv(out / "svd_cutoff_summary.csv")
    semantics = json.loads((out / "cutoff_semantics.json").read_text(encoding="utf-8"))
    spectra_path = out / "representative_spectra.npz"
    spectra = dict(np.load(spectra_path)) if spectra_path.exists() else {}
    plot_figure(summary, spectra, out)
    write_report(summary, semantics, out)
    print(f"Wrote {out / 'figure_svd_cutoff_diagnostic.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
