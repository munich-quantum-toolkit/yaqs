# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Reproduce the appendix probe-budget convergence figure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from mqt.yaqs.characterization.memory.operational_memory.samples import sample_probes
from updated_experiments.experiments.common import (
    DT_DEFAULT,
    G_DEFAULT,
    K_DEFAULT,
    L_DEFAULT,
    characterizer,
    entropy_from_responses,
    evaluate_weighted_probes,
    initial_states_sys_env0,
    ising_chain,
    load_csv,
    parse_float_list,
    parse_int_list,
    sim_params,
    slice_probe_prefix,
)

M_VALUES = (4, 8, 16, 32, 64)
J_VALUES = tuple(round(0.2 * i, 10) for i in range(11))
IMPLEMENTATION_COMMIT = "c6744d1ff97ad163b53682e71dda858a3454718b"


def run(args: argparse.Namespace) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    """Run one maximum-size grid per coupling and draw, then reuse its prefixes."""
    m_values = sorted(parse_int_list(args.m_values))
    j_values = parse_float_list(args.j_values)
    m_max = max(m_values)
    psi0 = initial_states_sys_env0(
        length=L_DEFAULT,
        n_seeds=1,
        rng=np.random.default_rng(args.seed),
    )[0]
    mc = characterizer(parallel=args.parallel, max_workers=args.max_workers)
    params = sim_params(dt=DT_DEFAULT)
    detail: list[dict[str, float | int]] = []

    for j_value in j_values:
        ham = ising_chain(length=L_DEFAULT, j=j_value, g=G_DEFAULT)
        for draw in range(args.probe_draws):
            draw_seed = args.seed + 100_000 * args.cut + 10 * round(100 * j_value) + draw
            probes = sample_probes(
                cut=args.cut,
                num_interventions=K_DEFAULT,
                n_pasts=m_max,
                n_futures=m_max,
                rng=np.random.default_rng(draw_seed),
                intervention_style="haar",
            )
            pauli_ixyz, weights = evaluate_weighted_probes(
                mc,
                ham,
                params,
                probe_set=probes,
                initial_psi=psi0,
            )
            for m in m_values:
                response_prefix, weight_prefix = slice_probe_prefix(pauli_ixyz, weights, m)
                detail.append({
                    "cut": args.cut,
                    "J": j_value,
                    "m": m,
                    "probe_draw": draw,
                    "entropy": entropy_from_responses(response_prefix, weight_prefix),
                })
        print(f"completed J={j_value:g}", flush=True)

    summary: list[dict[str, float | int]] = []
    for m in m_values:
        for j_value in j_values:
            values = np.asarray(
                [float(row["entropy"]) for row in detail if row["m"] == m and row["J"] == j_value],
                dtype=np.float64,
            )
            summary.append({
                "cut": args.cut,
                "J": j_value,
                "m": m,
                "entropy_mean": float(values.mean()),
                "entropy_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "entropy_sem": float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0,
            })
    return detail, summary


def plot(summary: list[dict[str, str | float | int]], output_stem: Path) -> tuple[float, float]:
    """Plot the stored convergence table using the paper's compact style."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.ticker import FixedLocator, LogFormatter, LogLocator, NullFormatter, NullLocator

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.0,
    })
    fig, ax = plt.subplots(figsize=(3.35, 2.45), constrained_layout=True)
    js = sorted({float(row["J"]) for row in summary})
    m_values = sorted({int(float(row["m"])) for row in summary})
    highlights = {min(js, key=lambda value: abs(value - target)) for target in (0.4, 1.0, 2.0)}
    base = plt.get_cmap("Reds")
    cmap = LinearSegmentedColormap.from_list("Reds_truncated", base(np.linspace(0.35, 0.95, 256)))
    norm = Normalize(vmin=0.0, vmax=2.0)

    positive: list[float] = []
    for j_value in js:
        rows = sorted(
            (row for row in summary if abs(float(row["J"]) - j_value) < 1e-12),
            key=lambda row: int(float(row["m"])),
        )
        x = np.asarray([int(float(row["m"])) for row in rows], dtype=np.float64)
        y = np.asarray([float(row["entropy_mean"]) for row in rows], dtype=np.float64)
        positive.extend(y[y > 1e-15].tolist())
        mask = x >= 3
        highlighted = j_value in highlights
        ax.semilogy(
            x[mask],
            np.clip(y[mask], 1e-30, None),
            color=cmap(norm(j_value)),
            lw=1.7 if highlighted else 0.8,
            marker="o" if highlighted else None,
            ms=4.0,
            markeredgewidth=0.0,
            alpha=0.98 if highlighted else 0.38,
            zorder=3 if highlighted else 1,
        )

    floor = 10.0 ** np.floor(np.log10(min(positive))) if positive else 1e-7
    ceiling = 10.0 ** np.ceil(np.log10(max(positive))) if positive else 1.0
    ax.set_ylim(floor, ceiling)
    ax.set_xlabel(r"Probe budget $m$ ($N_p=N_f$)")
    ax.set_ylabel(r"$S_V$")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(m_values))
    ax.xaxis.set_major_formatter(LogFormatter(base=2, labelOnlyBase=False))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(True, which="major", axis="y", alpha=0.08, linewidth=0.3)
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02, shrink=0.9)
    colorbar.ax.set_title(r"$J$", pad=2)
    fig.savefig(output_stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return floor, ceiling


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("updated_experiments/results/probe_budget"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cut", type=int, default=10)
    parser.add_argument("--probe-draws", type=int, default=5)
    parser.add_argument("--m-values", default=",".join(map(str, M_VALUES)))
    parser.add_argument("--j-values", default=",".join(map(str, J_VALUES)))
    parser.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "convergence_summary.csv"
    started = time.perf_counter()
    if args.plot_only:
        summary = load_csv(summary_path)
    else:
        detail, summary = run(args)
        _write_csv(output_dir / "convergence_detail.csv", detail)
        _write_csv(summary_path, summary)
    limits = plot(summary, output_dir / "probe_budget")
    if not args.plot_only:
        artifacts = [
            output_dir / "convergence_detail.csv",
            summary_path,
            output_dir / "probe_budget.pdf",
            output_dir / "probe_budget.png",
        ]
        manifest = {
            "figure": "appendix probe budget",
            "implementation_git_commit": IMPLEMENTATION_COMMIT,
            "configuration": {
                "L": L_DEFAULT,
                "k": K_DEFAULT,
                "dt": DT_DEFAULT,
                "g": G_DEFAULT,
                "cut": args.cut,
                "m_values": parse_int_list(args.m_values),
                "J_values": parse_float_list(args.j_values),
                "probe_draws": args.probe_draws,
                "seed": args.seed,
                "probe_style": "haar",
                "centered": False,
                "response_channels": "IXYZ",
                "response_orientation": "future_rows_history_columns",
                "weight_scope": "complete_retained_record",
            },
            "plot_y_limits": limits,
            "runtime_seconds": time.perf_counter() - started,
            "artifacts": {path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)} for path in artifacts},
        }
        (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote probe-budget figure to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
