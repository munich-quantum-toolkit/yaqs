# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Analyze the generator-rank pilot: decision-rule screening, report, diagnostic plots.

Usage:
    uv run python experiments/generator_rank_pilot/analyze.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import operator

import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
CSV_PATH = OUTPUT_DIR / "pilot_results.csv"
REPORT_PATH = OUTPUT_DIR / "report.md"

FLOOR = 1e-16
BASELINE_METHODS = ("mpo_gatewise", "mpo_layer", "variational_layer")  # non-SWAP, non-oracle


def load() -> list[dict[str, Any]]:
    rows = []
    with CSV_PATH.open() as fh:
        for r in csv.DictReader(fh):
            if not r["infidelity"]:
                continue
            for k in ("infidelity", "runtime_s", "total_discarded_weight"):
                r[k] = float(r[k])
            for k in ("chi", "substeps", "final_max_bond", "peak_max_bond",
                      "peak_param_count", "est_transient_elements", "length"):
                r[k] = int(r[k])
            r["infidelity"] = max(r["infidelity"], FLOOR)
            rows.append(r)
    return rows


def config_key(r: dict[str, Any]) -> tuple[str, str, str]:
    return (r["candidate"], r["size_label"], r["angle"])


def tdvp_converged(rows: list[dict[str, Any]], chi: int) -> dict[str, Any] | None:
    """Best (largest-n) TDVP run at this chi, with convergence delta vs n/2."""
    runs = {r["substeps"]: r for r in rows if r["method"] == "tdvp_layer" and r["chi"] == chi}
    if not runs:
        return None
    n_max = max(runs)
    best = dict(runs[n_max])
    prev = runs.get(n_max // 2)
    best["conv_delta_vs_prev_n"] = abs(best["infidelity"] - prev["infidelity"]) if prev else float("nan")
    return best


def screen_config(rows: list[dict[str, Any]]) -> list[str]:
    """Apply the pre-registered decision rule per chi; return findings lines."""
    lines = []
    chis = sorted({r["chi"] for r in rows})
    for chi in chis:
        td = tdvp_converged(rows, chi)
        if td is None:
            continue
        base = [r for r in rows if r["method"] in BASELINE_METHODS and r["chi"] == chi]
        if not base:
            continue
        best_base = min(base, key=operator.itemgetter("infidelity"))
        # Matched-error comparison: does converged TDVP reach <= best-baseline error
        # with >=2x lower peak params and runtime within 2x, or >=2x lower runtime at
        # matched error and memory, or >=10x accuracy at matched resources?
        mem_adv = best_base["peak_param_count"] / max(td["peak_param_count"], 1)
        trans_adv = best_base["est_transient_elements"] / max(td["est_transient_elements"], 1)
        rt_ratio = td["runtime_s"] / max(best_base["runtime_s"], 1e-9)
        acc_ratio = best_base["infidelity"] / td["infidelity"]
        passes = (
            (acc_ratio >= 1.0 and mem_adv >= 2.0 and rt_ratio <= 2.0)
            or (acc_ratio >= 1.0 and rt_ratio <= 0.5 and mem_adv >= 1.0)
            or (acc_ratio >= 10.0 and mem_adv >= 0.5 and rt_ratio <= 2.0)
        )
        lines.append(
            f"chi={chi}: TDVP(n={td['substeps']}) inf={td['infidelity']:.2e} "
            f"(Δ vs n/2: {td['conv_delta_vs_prev_n']:.1e}) vs best baseline "
            f"{best_base['method']}/{best_base['ordering']} inf={best_base['infidelity']:.2e}; "
            f"acc x{acc_ratio:.2g}, peak-param x{mem_adv:.2g}, transient x{trans_adv:.2g}, "
            f"runtime x{rt_ratio:.2g} -> {'PASS' if passes else 'no'}"
        )
    return lines


def method_series(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Best run per (display-method, chi) for Pareto plots."""
    series: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        if r["method"] == "tdvp_layer":
            label = "tdvp (best n)"
        elif r["method"] == "mpo_gatewise":
            label = "mpo gatewise (best order)"
        else:
            label = r["method"]
        cur = series[label].get(r["chi"])
        if cur is None or r["infidelity"] < cur["infidelity"]:
            series[label][r["chi"]] = r
    return {k: [v[c] for c in sorted(v)] for k, v in series.items()}


STYLE = {
    "tdvp (best n)": {"color": "tab:red", "marker": "o"},
    "mpo gatewise (best order)": {"color": "tab:blue", "marker": "s"},
    "mpo_layer": {"color": "tab:green", "marker": "^"},
    "variational_layer": {"color": "tab:purple", "marker": "v"},
    "oracle_compress": {"color": "black", "marker": "x", "linestyle": "--"},
}


def make_plots(groups: dict[tuple[str, str, str], list[dict[str, Any]]]) -> None:
    for xfield, fname, xlabel in (
        ("peak_param_count", "infidelity_vs_peak_params.png", "peak MPS parameter count"),
        ("runtime_s", "infidelity_vs_runtime.png", "wall-clock time [s] (excl. reference/layer-MPO build)"),
    ):
        keys = sorted(groups)
        ncol = 4
        nrow = (len(keys) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False)
        for ax, key in zip(axes.flat, keys, strict=False):
            for label, pts in method_series(groups[key]).items():
                xs = [max(p[xfield], 1e-4 if xfield == "runtime_s" else 1) for p in pts]
                ys = [p["infidelity"] for p in pts]
                ax.plot(xs, ys, label=label, alpha=0.8, **STYLE.get(label, {}))
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{key[0]} {key[1]} angle={key[2]}", fontsize=10)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel("infidelity (floored 1e-16)", fontsize=8)
            ax.grid(True, which="both", alpha=0.25)
        for ax in axes.flat[len(keys):]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / fname, dpi=140)
        plt.close(fig)


INTERPRETATION = """\
## Interpretation

Both candidates are **negative** under the pre-registered rule. The intended rank
separation D_H << D_U was genuinely realized (QAOA 5x5: D_H=7 vs D_U=32 = 2^b at
b=5 crossing edges; OAT: D_H=3 vs D_U up to 47), so the hypothesis received a fair
test and failed.

### QAOA/Ising cost layer: TDVP fails by projection, not by resources

Complete-generator 2TDVP stalls at bond profile [2,2,2,4,...] regardless of
chi (4..64) and substeps (converged in n; n=4 vs n=8 delta ~1e-3), plateauing at
infidelity 0.127 (4x4, gamma=0.15) to 0.78 (5x5, gamma=0.35) while gatewise and
complete-layer MPO routes reach the exact state at chi = 2^b. Total discarded
weight during TDVP is ~1e-29 and the stall persists bit-identically at
svd_threshold=1e-30 (see run log), so it is intrinsic tangent-space projection
error, not truncation: for the diagonal ZZ generator acting on |+>^N, the exact
evolution preserves <Z_i>=0, and the projected effective generators transmit
almost no entanglement across bonds, confining the TDVP flow to a low-rank
invariant submanifold. Direct MPO application has no such obstruction. This is
the *opposite* of a TDVP-favoring regime, despite maximal D_H/D_U separation.

### OAT collective entangler: TDVP works, but the complete-layer MPO removes any advantage

Substep-converged TDVP tracks the oracle representability bound well in the
representability-limited regime (e.g. N=20, kappa=0.5, chi=4: TDVP n=8 reaches
1.34e-5 vs oracle 1.30e-5) and decisively beats *sequential gatewise* application
of the 190 equivalent Rxx gates (7.8e-3 at chi=4). However, the complete-layer
MPO applied once and compressed matches the oracle at every chi, at comparable
peak parameter count (the initial state is a product state, so the uncompressed
MPO x MPS intermediate is only D_U wide) and ~25-50x lower wall-clock time than
TDVP n=8. Per the pre-registered rule ("if a complete-layer or oracle compression
removes the apparent advantage, record the candidate as negative"), OAT is
negative. The gatewise-vs-aggregated gap is itself a real observation, but the
winning aggregated method is the MPO layer, not TDVP.

### Cross-chi matched-error check

Equal-chi screening is sufficient here: the complete-layer MPO baseline attains
oracle-level accuracy at every chi in every configuration, so no alternative chi
pairing can produce a TDVP accuracy-resource Pareto advantage; TDVP's only edge
(smaller transient than the MPO x MPS expansion) is offset by 20-80x runtime and
vanishes against the layer route applied to product states.

### Is this the paper's local-TDVP method?

No. Both candidates require the generator MPO to span the entire chain (the 2D
cost layer couples every snake bond; OAT is all-to-all), so the "window" of the
paper's local two-site TDVP construction is the full chain and the algorithm
reduces to standard global MPO-TDVP (a projector-splitting sweep under a
Hamiltonian-like MPO). Even a positive result here would have supported standard
MPO-TDVP practice, not the manuscript's local gate-application contribution.
Combined with the negative screen, this closes the D_H << D_U direction: no
further candidates in this family are warranted, and the PRA rewrite should
proceed on the honest method-selection storyline.
"""


def main() -> None:
    rows = load()
    meta = json.loads((OUTPUT_DIR / "meta.json").read_text())
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[config_key(r)].append(r)
    make_plots(groups)

    lines: list[str] = []
    lines.extend(("# Generator-rank pilot: compact generator (D_H) vs high-rank layer unitary (D_U)\n", "Automated screening of `pilot_results.csv` against the pre-registered decision rule. See the run log and `meta.json` for validations and fixed numerical settings.\n", "## Fixed settings\n"))
    for k, v in meta["settings"].items():
        lines.append(f"- `{k}` = {v}")
    lines.extend(("- BLAS/OMP threads pinned to 1 for all timed runs", "- angle conventions: " + json.dumps(meta["angle_convention"], indent=0) + "\n", "## Rank separation actually realized\n", "| candidate | size | angle | D_H | D_U (tol 1e-12) | exact state max bond |", "|---|---|---|---|---|---|"))
    for key in sorted(groups):
        g = groups[key][0]
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {g['d_h']} | {g['d_u']} | {g['exact_max_bond']} |")
    lines.extend(("", "## Per-configuration screening (converged TDVP vs best non-SWAP MPO/layer baseline)\n", "Rule: PASS requires >=2x lower peak params at matched-or-better error with runtime within 2x, or >=2x lower runtime at matched error/memory, or >=10x accuracy at matched resources. Oracle (`oracle_compress`) is a representability diagnostic, not a baseline.\n"))
    any_pass = False
    for key in sorted(groups):
        lines.append(f"### {key[0]} {key[1]} angle={key[2]}\n")
        for ln in screen_config(groups[key]):
            lines.append(f"- {ln}")
            any_pass = any_pass or ln.endswith("PASS")
        lines.append("")

    lines.extend(("## Screen outcome\n", f"- Any configuration passing the decision rule: **{'YES' if any_pass else 'NO'}**", "- Plots: `infidelity_vs_peak_params.png`, `infidelity_vs_runtime.png`\n", INTERPRETATION))

    REPORT_PATH.write_text("\n".join(lines))
    print(f"Wrote {REPORT_PATH}")  # noqa: T201
    print(f"Any PASS: {any_pass}")  # noqa: T201


if __name__ == "__main__":
    main()
