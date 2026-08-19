# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Exploratory RXX counterpart of the main-text CNOT refinement panel."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common import (  # noqa: E402
    DiscardedWeightTracker,
    apply_gate_dense_yaqs,
    apply_method,
    conventional_median,
    make_pauli_dag_node,
    make_pauli_gate,
    normalized_state_fidelity,
    prepare_initial_state,
    state_distance,
)
from config import (  # noqa: E402
    EFFECTIVE_ZERO_SVD_THRESHOLD,
    OUTPUT_DIR,
    Q0,
    Q1,
    REFINEMENT_CHI,
    REFINEMENT_FINE_N_SUB,
    REFINEMENT_N_SUB,
    REFINEMENT_SEEDS,
    REPO_ROOT,
    N,
    theta_from_x,
)
from plot import (  # noqa: E402
    COLOR_MPO,
    COLOR_TDVP,
    COLOR_TEBD,
    METHOD_STYLES,
    MM_TO_IN,
    _apply_style,
)

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required") from exc

X = 1e-2
THETA = theta_from_x(X)
N_VALUES = tuple(REFINEMENT_N_SUB) + (REFINEMENT_FINE_N_SUB,)
METHODS = ("gate_local_2tdvp", "mpo_zipup", "tebd_swap")
CSV_PATH = OUTPUT_DIR / "rxx_refinement_comparison_rows.csv"
FIGURE_STEM = "figure_individual_gates_rxx_refinement_comparison"
FIGURES_DIR = REPO_ROOT / "experiments" / "figures"


def run_comparison() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    states_root = OUTPUT_DIR / "rxx_refinement_states"

    for seed in REFINEMENT_SEEDS:
        initial = prepare_initial_state(seed)
        gate = make_pauli_gate("rxx", THETA, Q0, Q1)
        node = make_pauli_dag_node("rxx", THETA, Q0, Q1)
        exact = apply_gate_dense_yaqs(initial["vec"], N, Q0, Q1, gate)
        state_dir = states_root / f"seed_{seed}"
        state_dir.mkdir(parents=True, exist_ok=True)

        for method in METHODS:
            method_n_values = N_VALUES if method == "gate_local_2tdvp" else (1,)
            for n_sub in method_n_values:
                tracker = DiscardedWeightTracker()
                final_mps, _ = apply_method(
                    initial["mps"],
                    node,
                    method=method,
                    chi=REFINEMENT_CHI,
                    n_sub=n_sub,
                    svd_threshold=EFFECTIVE_ZERO_SVD_THRESHOLD,
                    tracker=tracker,
                )
                vec = final_mps.to_vec().astype(np.complex128)
                metrics = normalized_state_fidelity(exact, vec)
                if method == "gate_local_2tdvp":
                    np.save(state_dir / f"n{n_sub}.npy", vec)
                rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "n_sub": n_sub,
                        "infidelity": metrics["infidelity_normalized"],
                        "adjacent_refinement_distance": "",
                    }
                )

        for n_sub in REFINEMENT_N_SUB:
            vec = np.load(state_dir / f"n{n_sub}.npy")
            vec2 = np.load(state_dir / f"n{2 * n_sub}.npy")
            for row in rows:
                if (
                    row["seed"] == seed
                    and row["method"] == "gate_local_2tdvp"
                    and row["n_sub"] == n_sub
                ):
                    row["adjacent_refinement_distance"] = state_distance(vec, vec2)
                    break

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=("seed", "method", "n_sub", "infidelity", "adjacent_refinement_distance"),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {CSV_PATH}")
    return rows


def read_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def stats(values: list[float]) -> tuple[float, float, float]:
    return conventional_median(values), min(values), max(values)


def plot_comparison(rows: list[dict[str, object] | dict[str, str]]) -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(68.0 * MM_TO_IN, 70.0 * MM_TO_IN))

    endpoint: dict[str, tuple[float, float, float]] = {}
    for method, color in (("mpo_zipup", COLOR_MPO), ("tebd_swap", COLOR_TEBD)):
        vals = [
            float(row["infidelity"])
            for row in rows
            if row["method"] == method and int(row["n_sub"]) == 1
        ]
        med, lo, hi = stats(vals)
        endpoint[method] = (med, lo, hi)
        style = METHOD_STYLES[method]
        ax.axhspan(lo, hi, color=color, alpha=0.10, linewidth=0, zorder=0)
        ax.axhline(med, color=color, linestyle=style["linestyle"], linewidth=1.05, zorder=1)

    infidelity: dict[int, list[float]] = defaultdict(list)
    adjacent: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] != "gate_local_2tdvp":
            continue
        n_sub = int(row["n_sub"])
        infidelity[n_sub].append(float(row["infidelity"]))
        adj = row["adjacent_refinement_distance"]
        if adj not in {"", None}:
            adjacent[n_sub].append(float(adj))

    ns = np.array(sorted(infidelity), dtype=float)
    med = np.array([stats(infidelity[int(n)])[0] for n in ns])
    lo = np.array([stats(infidelity[int(n)])[1] for n in ns])
    hi = np.array([stats(infidelity[int(n)])[2] for n in ns])
    ax.fill_between(ns, lo, hi, color=COLOR_TDVP, alpha=0.14, linewidth=0, zorder=1)
    ax.plot(ns, med, color=COLOR_TDVP, linewidth=1.05, marker="o", markersize=3.0, zorder=3)

    adj_ns = np.array(sorted(adjacent), dtype=float)
    adj_med = np.array([stats(adjacent[int(n)])[0] for n in adj_ns])
    adj_lo = np.array([stats(adjacent[int(n)])[1] for n in adj_ns])
    adj_hi = np.array([stats(adjacent[int(n)])[2] for n in adj_ns])
    inset = ax.inset_axes([0.57, 0.42, 0.39, 0.23])
    inset.fill_between(adj_ns, adj_lo, adj_hi, color=COLOR_TDVP, alpha=0.14, linewidth=0)
    inset.plot(adj_ns, adj_med, color=COLOR_TDVP, linewidth=0.9, marker="o", markersize=2.5)
    inset.set_xscale("log", base=2)
    inset.set_yscale("log")
    inset.set_xlim(0.8, 700)
    inset.set_ylim(max(float(np.min(adj_lo)) * 0.5, 1e-12), float(np.max(adj_hi)) * 2.0)
    inset.set_xticks([1, 16, 256])
    inset.set_xticklabels(["1", "16", "256"])
    inset.set_xlabel(r"$n_{\mathrm{sub}}$", fontsize=5.2, labelpad=0.2)
    inset.set_ylabel(r"$D_n$", fontsize=5.2, labelpad=0.8)
    inset.tick_params(axis="both", which="major", labelsize=4.9, length=2.0, pad=1.0)
    inset.tick_params(axis="both", which="minor", length=1.2)
    inset.grid(True, which="major", axis="y", color="0.92", linewidth=0.3)
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(0.8, 1400)
    all_main = np.concatenate((lo, hi, np.array([v for triple in endpoint.values() for v in triple])))
    ax.set_ylim(max(float(np.min(all_main)) * 0.45, 1e-10), min(float(np.max(all_main)) * 1.7, 1.0))
    ticks = (1, 4, 16, 64, 256, 1024)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(v) for v in ticks])
    ax.set_xlabel(r"Projection substeps $n_{\mathrm{sub}}$")
    ax.set_ylabel(r"Infidelity $1-F$ (Rotation)")
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", axis="y", color="0.92", linewidth=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.4}
    label_x = 900.0
    ax.text(label_x, endpoint["tebd_swap"][0] * 1.03, "TEBD+SWAP", color=COLOR_TEBD, fontsize=5.8,
            ha="right", va="bottom", bbox=label_box)
    ax.text(label_x, endpoint["mpo_zipup"][0] * 1.08, "Direct MPO", color=COLOR_MPO, fontsize=5.8,
            ha="right", va="bottom", bbox=label_box)
    ax.text(label_x, med[-1] * 0.92, "Projection", color=COLOR_TDVP, fontsize=5.8,
            ha="right", va="top", bbox=label_box)

    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.17, top=0.96)
    for directory in (FIGURES_DIR, OUTPUT_DIR / "figures"):
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / f"{FIGURE_STEM}.pdf")
        fig.savefig(directory / f"{FIGURE_STEM}.png", dpi=600)
        print(f"Wrote {directory / f'{FIGURE_STEM}.pdf'}")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args(argv)
    if args.plot_only:
        if not CSV_PATH.is_file():
            raise SystemExit(f"Missing {CSV_PATH}")
        rows = read_rows()
    else:
        rows = run_comparison()
    plot_comparison(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
