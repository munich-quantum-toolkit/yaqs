# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Reproduce the appendix finite-environment consistency figure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from updated_experiments.experiments.common import (
    DT_DEFAULT,
    G_DEFAULT,
    K_DEFAULT,
    characterize,
    characterizer,
    ising_chain,
    load_csv,
    metrics_from,
    parse_float_list,
    parse_int_list,
    sample_cut_probes,
    sim_params,
)

L_VALUES = tuple(range(2, 11))
J_VALUES = (0.5, 1.0, 1.5, 2.0)
CUTS = (5, 10, 15)
IMPLEMENTATION_COMMIT = "c6744d1ff97ad163b53682e71dda858a3454718b"


def run(args: argparse.Namespace) -> list[dict[str, float | int]]:
    """Evaluate only the three cuts displayed in the appendix figure."""
    lengths = parse_int_list(args.l_values)
    couplings = parse_float_list(args.j_values)
    cuts = parse_int_list(args.cuts)
    probes = {
        cut: sample_cut_probes(
            cut=cut,
            k=K_DEFAULT,
            n_pasts=args.n_pasts,
            n_futures=args.n_futures,
            seed=args.seed,
        )
        for cut in cuts
    }
    mc = characterizer(parallel=args.parallel, max_workers=args.max_workers)
    params = sim_params(dt=DT_DEFAULT)
    rows: list[dict[str, float | int]] = []

    for length in lengths:
        psi0 = np.zeros(2**length, dtype=np.complex128)
        psi0[0] = 1.0
        for coupling in couplings:
            ham = ising_chain(length=length, j=coupling, g=G_DEFAULT)
            for cut in cuts:
                result = characterize(
                    mc,
                    ham,
                    params,
                    k=K_DEFAULT,
                    cut=cut,
                    n_pasts=args.n_pasts,
                    n_futures=args.n_futures,
                    probe_set=probes[cut],
                    initial_psi=psi0,
                )
                rows.append({
                    "L": length,
                    "environment_sites": length - 1,
                    "J": coupling,
                    "cut": cut,
                    "entropy": float(metrics_from(result, cut)["entropy"]),
                })
            print(f"completed L={length}, J={coupling:g}", flush=True)
    return rows


def plot(rows: list[dict[str, str | float | int]], output_stem: Path) -> tuple[float, float]:
    """Plot the three fixed-cut panels in the paper's full-width style."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })
    cuts = sorted({int(float(row["cut"])) for row in rows})
    couplings = sorted({float(row["J"]) for row in rows})
    environments = sorted({int(float(row["environment_sites"])) for row in rows})
    base = plt.get_cmap("Reds")
    cmap = LinearSegmentedColormap.from_list("Reds_truncated", base(np.linspace(0.30, 0.95, 256)))
    norm = Normalize(vmin=0.0, vmax=2.0)
    maximum = max(float(row["entropy"]) for row in rows)
    limits = (0.0, 1.08 * maximum)

    fig, axes_raw = plt.subplots(1, len(cuts), figsize=(8.2, 2.85), constrained_layout=True, sharey=True)
    axes = np.atleast_1d(axes_raw)
    tags = tuple(f"({chr(ord('a') + index)})" for index in range(len(cuts)))
    for ax, cut, tag in zip(axes, cuts, tags, strict=True):
        for coupling in couplings:
            selected = sorted(
                (row for row in rows if int(float(row["cut"])) == cut and abs(float(row["J"]) - coupling) < 1e-12),
                key=lambda row: int(float(row["environment_sites"])),
            )
            ax.plot(
                [int(float(row["environment_sites"])) for row in selected],
                [float(row["entropy"]) for row in selected],
                marker="o",
                linewidth=1.7,
                markersize=4.0,
                markeredgewidth=0.0,
                color=cmap(norm(coupling)),
                label=rf"$J={coupling:g}$",
            )
        ax.set_title(rf"$c={cut}$")
        ax.set_xlabel("Environmental sites")
        ax.set_xticks(environments)
        ax.set_ylim(*limits)
        ax.grid(True, axis="y", alpha=0.10, linewidth=0.35)
        ax.text(0.04, 0.955, tag, transform=ax.transAxes, va="top", fontsize=11, fontweight="bold")
    axes[0].set_ylabel(r"$S_V(c)$")
    axes[-1].legend(frameon=False, loc="upper right", handlelength=1.6, borderaxespad=0.2)
    fig.savefig(output_stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return limits


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    """Write portable LF-terminated data."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of an artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("updated_experiments/results/finite_size"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--l-values", default=",".join(map(str, L_VALUES)))
    parser.add_argument("--j-values", default=",".join(map(str, J_VALUES)))
    parser.add_argument("--cuts", default=",".join(map(str, CUTS)))
    parser.add_argument("--n-pasts", type=int, default=32)
    parser.add_argument("--n-futures", type=int, default=32)
    parser.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "finite_size.csv"
    started = time.perf_counter()
    if args.plot_only:
        rows = load_csv(data_path)
    else:
        rows = run(args)
        write_csv(data_path, rows)
    limits = plot(rows, output_dir / "finite_size")
    if not args.plot_only:
        artifacts = [data_path, output_dir / "finite_size.pdf", output_dir / "finite_size.png"]
        manifest = {
            "figure": "appendix finite environment",
            "implementation_git_commit": IMPLEMENTATION_COMMIT,
            "configuration": {
                "L_values": parse_int_list(args.l_values),
                "environment_sites": [length - 1 for length in parse_int_list(args.l_values)],
                "J_values": parse_float_list(args.j_values),
                "cuts": parse_int_list(args.cuts),
                "k": K_DEFAULT,
                "dt": DT_DEFAULT,
                "g": G_DEFAULT,
                "n_pasts": args.n_pasts,
                "n_futures": args.n_futures,
                "seed": args.seed,
                "probe_style": "haar",
                "centered": False,
                "response_channels": "IXYZ",
                "response_orientation": "future_rows_history_columns",
                "weight_scope": "complete_retained_record",
            },
            "plot_y_limits": limits,
            "runtime_seconds": time.perf_counter() - started,
            "artifacts": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in artifacts},
        }
        (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote finite-size figure to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
