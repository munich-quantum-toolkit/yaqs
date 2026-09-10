# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Reproduce Figure 5, the conditioned-reset persistence profile."""

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
    L_DEFAULT,
    build_ell_delay_probes,
    characterize_custom_sequences,
    characterizer,
    ising_chain,
    load_csv,
    metrics_from,
    sample_ell_base_ensemble,
    sim_params,
)

J_VALUES = (0.5, 1.0, 1.5, 2.0)
ELL_VALUES = tuple(range(16))
PAST_LEN = 15
FUTURE_LEN = 5
IMPLEMENTATION_COMMIT = "c6744d1ff97ad163b53682e71dda858a3454718b"


def parse_int_list(specification: str) -> list[int]:
    """Parse a comma-separated integer list."""
    return [int(token.strip()) for token in specification.split(",") if token.strip()]


def run(args: argparse.Namespace) -> list[dict[str, float | int]]:
    """Evaluate the four coupling profiles using one shared probe ensemble."""
    ells = parse_int_list(args.ells)
    mc = characterizer(parallel=args.parallel, max_workers=args.max_workers)
    params = sim_params(dt=DT_DEFAULT)
    psi0 = np.zeros(2**L_DEFAULT, dtype=np.complex128)
    psi0[0] = 1.0
    probe_rng = np.random.default_rng(args.seed + 999_991)
    past_pairs, past_cut_meas, future_prep_cut, future_pairs = sample_ell_base_ensemble(
        n_pasts=args.n_pasts,
        n_futures=args.n_futures,
        rng=probe_rng,
        past_len=PAST_LEN,
        future_len=FUTURE_LEN,
        style="haar",
    )
    rows: list[dict[str, float | int]] = []
    left_cut = PAST_LEN + 1

    for ell in ells:
        probes, intervention_steps = build_ell_delay_probes(
            past_pairs=past_pairs,
            past_cut_meas=past_cut_meas,
            future_prep_cut=future_prep_cut,
            future_pairs=future_pairs,
            past_len=PAST_LEN,
            future_len=FUTURE_LEN,
            ell=ell,
        )
        for coupling in J_VALUES:
            result = characterize_custom_sequences(
                mc,
                ising_chain(length=L_DEFAULT, j=coupling, g=G_DEFAULT),
                params,
                probe_set=probes,
                psi_pairs_list=intervention_steps,
                initial_psi=psi0,
                cut=left_cut,
            )
            rows.append({
                "ell": ell,
                "J": coupling,
                "entropy": float(metrics_from(result, left_cut)["entropy"]),
                "k": PAST_LEN + ell + FUTURE_LEN + 2,
                "left_cut": left_cut,
                "right_cut": left_cut + ell + 1,
            })
        print(f"completed ell={ell}", flush=True)
    return rows


def plot(rows: list[dict[str, str | float | int]], output_stem: Path) -> tuple[float, float]:
    """Plot the persistence curves with the original Figure 5 styling."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })
    couplings = sorted({float(row["J"]) for row in rows})
    ells = sorted({int(float(row["ell"])) for row in rows})
    positive = [float(row["entropy"]) for row in rows if float(row["entropy"]) > 1e-15]
    floor = 10.0 ** np.floor(np.log10(min(positive)))
    ceiling = min(1.0, 10.0 ** np.ceil(np.log10(max(positive))))
    cmap = plt.get_cmap("Reds")
    norm = Normalize(vmin=0.0, vmax=2.0)

    fig, ax = plt.subplots(figsize=(5.0, 3.2), constrained_layout=True)
    for coupling in couplings:
        selected = sorted(
            (row for row in rows if abs(float(row["J"]) - coupling) < 1e-12),
            key=lambda row: int(float(row["ell"])),
        )
        ax.semilogy(
            [int(float(row["ell"])) for row in selected],
            [max(float(row["entropy"]), floor) for row in selected],
            linewidth=1.9,
            marker="o",
            markersize=3.8,
            markeredgewidth=0.0,
            color=cmap(norm(coupling)),
            alpha=0.94,
            label=rf"$J={coupling:g}$",
        )
    ax.set_xlabel(r"Delay $\ell$")
    ax.set_ylabel(r"$S_V$")
    ax.set_xlim(min(ells) - 0.4, max(ells) + 0.4)
    ax.set_ylim(floor, ceiling)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.grid(True, which="major", axis="y", alpha=0.10, linewidth=0.35)
    ax.legend(frameon=False, fontsize=7.0, handlelength=1.4, borderaxespad=0.2, loc="upper right")
    fig.savefig(output_stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return floor, ceiling


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    """Write portable LF-terminated data."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    """Return an artifact's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("updated_experiments/results/figure5"))
    parser.add_argument("--ells", default=",".join(map(str, ELL_VALUES)))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-pasts", type=int, default=64)
    parser.add_argument("--n-futures", type=int, default=64)
    parser.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "memory_persistence.csv"
    started = time.perf_counter()
    if args.plot_only:
        rows = load_csv(data_path)
    else:
        rows = run(args)
        write_csv(data_path, rows)
    limits = plot(rows, output_dir / "memory_length")
    if not args.plot_only:
        artifacts = [data_path, output_dir / "memory_length.pdf", output_dir / "memory_length.png"]
        manifest = {
            "figure": 5,
            "implementation_git_commit": IMPLEMENTATION_COMMIT,
            "configuration": {
                "L": L_DEFAULT,
                "J_values": list(J_VALUES),
                "ell_values": parse_int_list(args.ells),
                "past_length": PAST_LEN,
                "future_length": FUTURE_LEN,
                "n_pasts": args.n_pasts,
                "n_futures": args.n_futures,
                "initial_state": "|0>^6",
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
    print(f"Wrote Figure 5 to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
