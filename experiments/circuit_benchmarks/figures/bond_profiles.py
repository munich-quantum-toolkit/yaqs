# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License
"""Plot a single-column grid of fixed-cap, step-end MPS bond profiles."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from experiments.circuit_benchmarks.config import (
    CAMPAIGN_ID,
    CHI_MAIN,
    N_STEPS,
    OUTPUT_DIR,
    REPO_ROOT,
    N,
)
from experiments.circuit_benchmarks.plotting import (
    CASE_ORDER,
    apply_style,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

FIGURE_STEM = "figure_circuit_bond_profiles"
DATA_FILENAME = "bond_profiles.csv"
FIGURE_WIDTH_MM = 86.0
FIGURE_HEIGHT_MM = 125.0
MM_TO_IN = 1.0 / 25.4
DPI = 600

PROFILE_METHOD_ORDER = ("gate_local_2tdvp", "mpo_contract_compress", "tebd_swap")
PROFILE_METHOD_LABELS = {
    "gate_local_2tdvp": "TDVP",
    "mpo_contract_compress": "MPO",
    "tebd_swap": "TEBD+SWAP",
}
PROFILE_ROW_LABELS = {
    "ising_1d": "(a)\n1D\nIsing",
    "heisenberg_1d": "(b)\n1D\nHeisenberg",
    "ising_2d": "(c)\n" + "$4\\times4$\nIsing",
    "heisenberg_2d": "(d)\n" + "$4\\times4$\nHeisenberg",
}
PROFILE_COLOR_TICKS = (1, 8, 16, 24, 32)
PROFILE_COLORMAP = "plasma"
CASE_LAST_STEP = {
    "ising_1d": 27,
    "heisenberg_1d": 6,
    "ising_2d": 6,
    "heisenberg_2d": 1,
}
CASE_X_TICKS = {
    "ising_1d": (0, 9, 18, 27),
    "heisenberg_1d": (0, 2, 4, 6),
    "ising_2d": (0, 2, 4, 6),
    "heisenberg_2d": (0, 1),
}

TaskKey = tuple[str, str]


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object from ``path``."""
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        msg = f"Expected a JSON object in {path}."
        raise TypeError(msg)
    return value


def _current_trajectory_tasks(output_dir: Path) -> dict[TaskKey, dict[str, Any]]:
    """Return the current successful traced trajectory task for every panel."""
    manifest = _load_json(output_dir / "manifest.json")
    source_hash = manifest.get("source_hash")
    tasks: dict[TaskKey, dict[str, Any]] = {}
    task_dir = output_dir / "tasks" / "trajectories"
    for path in sorted(task_dir.glob("*.json")):
        task = _load_json(path)
        payload = task.get("payload", {})
        spec = payload.get("spec", {}) if isinstance(payload, dict) else {}
        if (
            task.get("status") != "success"
            or payload.get("campaign_id") != CAMPAIGN_ID
            or payload.get("source_hash") != source_hash
            or spec.get("run_family") != "trajectories"
            or int(spec.get("chi_max", -1)) != CHI_MAIN
            or int(spec.get("steps", -1)) != N_STEPS
            or spec.get("trace_resources") is not True
        ):
            continue
        key = (str(spec.get("case")), str(spec.get("method")))
        if key in tasks:
            msg = f"Multiple current trajectory tasks found for {key}."
            raise RuntimeError(msg)
        tasks[key] = task

    expected = {(case, method) for case in CASE_ORDER for method in PROFILE_METHOD_ORDER}
    if set(tasks) != expected:
        missing = sorted(expected - set(tasks))
        extra = sorted(set(tasks) - expected)
        msg = f"Incomplete current trajectory set: missing={missing}, extra={extra}."
        raise RuntimeError(msg)
    return tasks


def _internal_bond_profile(profile: Sequence[int], *, n_sites: int = N) -> np.ndarray:
    """Validate a full MPS bond profile and return its internal cuts."""
    values = np.asarray(profile, dtype=np.int64)
    if values.shape != (n_sites + 1,):
        msg = f"Expected {n_sites + 1} MPS bonds, received shape {values.shape}."
        raise ValueError(msg)
    if values[0] != 1 or values[-1] != 1:
        msg = "Open-boundary MPS profiles must have unit boundary bonds."
        raise ValueError(msg)
    if np.any(values < 1):
        msg = "MPS bond dimensions must be positive."
        raise ValueError(msg)
    return values[1:-1]


def _stack_step_profiles(
    profiles: Mapping[int, np.ndarray],
    *,
    n_steps: int = N_STEPS,
) -> np.ndarray:
    """Stack one unique profile for every step from zero through ``n_steps``."""
    expected = set(range(n_steps + 1))
    if set(profiles) != expected:
        missing = sorted(expected - set(profiles))
        extra = sorted(set(profiles) - expected)
        msg = f"Incomplete step-end profile sequence: missing={missing}, extra={extra}."
        raise RuntimeError(msg)
    return np.stack([profiles[step] for step in range(n_steps + 1)])


def _task_profile_matrix(task: Mapping[str, Any], output_dir: Path) -> np.ndarray:
    """Load the complete step-end internal-bond matrix for one trajectory task."""
    task_id = str(task["task_id"])
    checkpoint_path = output_dir / "checkpoints" / f"{task_id}.jsonl.gz"
    if not checkpoint_path.is_file():
        msg = f"Missing checkpoint stream {checkpoint_path}."
        raise FileNotFoundError(msg)

    profiles: dict[int, np.ndarray] = {}
    with gzip.open(checkpoint_path, mode="rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            step = int(row.get("step", -1))
            checkpoint = row.get("checkpoint")
            is_endpoint = (step == 0 and checkpoint == "initial") or (1 <= step <= N_STEPS and checkpoint == "step_end")
            if not is_endpoint:
                continue
            if step in profiles:
                msg = f"Duplicate step-end profile for task {task_id}, step {step}."
                raise RuntimeError(msg)
            profiles[step] = _internal_bond_profile(row["bond_dimensions"])
    return _stack_step_profiles(profiles)


def _load_profile_table(path: Path) -> dict[TaskKey, np.ndarray]:
    """Load the portable, panel-level bond-profile source table."""
    profiles: dict[TaskKey, dict[int, dict[int, int]]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["case"], row["method"])
            step = int(row["step"])
            bond = int(row["bond"])
            profiles.setdefault(key, {}).setdefault(step, {})[bond] = int(row["bond_dimension"])

    matrices: dict[TaskKey, np.ndarray] = {}
    for key, by_step in profiles.items():
        step_profiles = {
            step: np.asarray([bonds[bond] for bond in range(1, N)], dtype=np.int64)
            for step, bonds in by_step.items()
        }
        matrices[key] = _stack_step_profiles(step_profiles)
    expected = {(case, method) for case in CASE_ORDER for method in PROFILE_METHOD_ORDER}
    if set(matrices) != expected:
        msg = f"Incomplete bond-profile source table: missing={sorted(expected - set(matrices))}."
        raise RuntimeError(msg)
    return matrices


def _write_profile_table(path: Path, profiles: Mapping[TaskKey, np.ndarray]) -> None:
    """Write a portable source table after extracting resumable checkpoints."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("case", "method", "step", "bond", "bond_dimension"))
        for (case, method), matrix in sorted(profiles.items()):
            for step, profile in enumerate(matrix):
                for bond, dimension in enumerate(profile, start=1):
                    writer.writerow((case, method, step, bond, int(dimension)))


def load_profile_matrices(
    output_dir: Path,
    *,
    refresh_data: bool = False,
) -> dict[TaskKey, np.ndarray]:
    """Load all 12 profiles, optionally refreshing the portable source table."""
    source_path = output_dir / DATA_FILENAME
    if source_path.is_file() and not refresh_data:
        return _load_profile_table(source_path)
    tasks = _current_trajectory_tasks(output_dir)
    profiles = {key: _task_profile_matrix(task, output_dir) for key, task in tasks.items()}
    _write_profile_table(source_path, profiles)
    return profiles


def _step_edges(last_step: int) -> np.ndarray:
    """Return complete cell edges centered on integer Trotter steps."""
    if last_step < 1:
        msg = "A bond-profile heatmap requires at least one evolved step."
        raise ValueError(msg)
    return np.arange(last_step + 2, dtype=float) - 0.5


def _bond_edges() -> np.ndarray:
    """Return cell edges for the 15 internal MPS cuts."""
    return np.arange(0.5, N + 0.5, dtype=float)


def _cropped_profile(matrix: np.ndarray, case_key: str) -> np.ndarray:
    """Return the informative rank-growth transient for one circuit."""
    expected = (N_STEPS + 1, N - 1)
    if matrix.shape != expected:
        msg = f"Expected profile shape {expected}, received {matrix.shape}."
        raise ValueError(msg)
    return matrix[: CASE_LAST_STEP[case_key] + 1]


def _style_profile_axis(
    axis: plt.Axes,
    *,
    case_key: str,
    show_x: bool,
    show_y: bool,
) -> None:
    """Apply compact shared-axis styling to one profile heatmap."""
    last_step = CASE_LAST_STEP[case_key]
    axis.set_xlim(-0.5, last_step + 0.5)
    axis.set_ylim(0.5, N - 0.5)
    axis.set_xticks(CASE_X_TICKS[case_key])
    axis.set_yticks((1, 4, 8, 12, 15))
    axis.tick_params(
        which="both",
        direction="out",
        width=0.7,
        length=2.2,
        labelbottom=show_x,
        labelleft=show_y,
    )
    for spine in axis.spines.values():
        spine.set_linewidth(0.7)


def create_figure(profiles: Mapping[TaskKey, np.ndarray]) -> plt.Figure:
    """Build the four-circuit by three-method single-column figure."""
    apply_style()
    mpl.rcParams["axes.titlesize"] = 8.0
    figure = plt.figure(
        figsize=(FIGURE_WIDTH_MM * MM_TO_IN, FIGURE_HEIGHT_MM * MM_TO_IN),
    )
    grid = figure.add_gridspec(
        len(CASE_ORDER),
        len(PROFILE_METHOD_ORDER) + 2,
        width_ratios=(0.70, 0.14, 1.0, 1.0, 1.0),
        wspace=0.16,
        hspace=0.30,
    )

    cut_label_axis = figure.add_subplot(grid[:, 1])
    cut_label_axis.axis("off")
    cut_label_axis.text(
        -0.55,
        0.5,
        r"Bond $b$",
        ha="center",
        va="center",
        rotation=90,
        fontsize=9.0,
    )

    mesh: mpl.collections.QuadMesh | None = None
    profile_axes: list[plt.Axes] = []
    for row_index, case in enumerate(CASE_ORDER):
        label_axis = figure.add_subplot(grid[row_index, 0])
        label_axis.axis("off")
        label_axis.text(
            0.50,
            0.5,
            PROFILE_ROW_LABELS[case],
            ha="center",
            va="center",
            fontsize=6.8,
            linespacing=1.05,
        )
        for column, method in enumerate(PROFILE_METHOD_ORDER):
            axis = figure.add_subplot(grid[row_index, column + 2])
            matrix = _cropped_profile(profiles[(case, method)], case)
            mesh = axis.pcolormesh(
                _step_edges(CASE_LAST_STEP[case]),
                _bond_edges(),
                matrix.T,
                cmap=PROFILE_COLORMAP,
                norm=Normalize(vmin=1, vmax=CHI_MAIN),
                shading="flat",
                rasterized=True,
            )
            _style_profile_axis(
                axis,
                case_key=case,
                show_x=True,
                show_y=column == 0,
            )
            if row_index == 0:
                axis.set_title(PROFILE_METHOD_LABELS[method], pad=2.5)
            profile_axes.append(axis)

    if mesh is None:
        msg = "No profile heatmaps were created."
        raise RuntimeError(msg)
    figure.subplots_adjust(top=0.82, bottom=0.085, left=0.01, right=0.985)
    color_axis = figure.add_axes((0.315, 0.91, 0.60, 0.018))
    colorbar = figure.colorbar(
        mesh,
        cax=color_axis,
        orientation="horizontal",
        ticks=PROFILE_COLOR_TICKS,
    )
    colorbar.ax.set_xticklabels(["1", "8", "16", "24", "32 (cap)"])
    colorbar.ax.tick_params(width=0.7, length=2.2)
    colorbar.outline.set_linewidth(0.7)
    colorbar.ax.set_title(r"Bond dimension $\chi_b$", fontsize=8.0, pad=2.0)

    plot_center = (
        profile_axes[0].get_position().x0 + profile_axes[len(PROFILE_METHOD_ORDER) - 1].get_position().x1
    ) / 2
    figure.supxlabel(r"Trotter steps $n$", x=plot_center, y=0.018)
    return figure


def caption() -> str:
    """Return the manuscript-ready figure caption."""
    return (
        "MPS bond profiles during fixed-cap circuit evolution at chi_max=32. Rows (a)--(d) show the four "
        "model and geometry combinations; columns show TDVP, MPO, and TEBD+SWAP. TDVP denotes gate-local "
        "two-site TDVP, and MPO denotes routing-free MPO contract-and-truncate. "
        "The displayed step ranges end at n=27, 6, "
        "6, and 1 from top to bottom. Colors encode the retained bond dimensions linearly, with the terminal "
        "color marking the imposed cap. Profiles are recorded after complete Trotter "
        "steps, not exact Schmidt ranks or transient working ranks, and do not by themselves establish "
        "accuracy. For the two-dimensional systems, b "
        "indexes cuts of the snake-ordered MPS. TDVP fills the available bond space later in the "
        "first three cases, whereas every method reaches the stable cap-limited profile after one step for "
        "the 4x4 Heisenberg circuit."
    )


def main(argv: list[str] | None = None) -> None:
    """Load frozen data and write the vector/raster profile figure and caption."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "figures",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Rebuild bond_profiles.csv from the current trajectory checkpoints.",
    )
    args = parser.parse_args(argv)

    profiles = load_profile_matrices(args.output_dir, refresh_data=args.refresh_data)
    figure = create_figure(profiles)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figures_dir / f"{FIGURE_STEM}.pdf", dpi=DPI)
    figure.savefig(args.figures_dir / f"{FIGURE_STEM}.png", dpi=DPI)
    plt.close(figure)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{FIGURE_STEM}_caption.md").write_text(
        caption() + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
