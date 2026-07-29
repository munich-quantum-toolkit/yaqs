# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Shared helpers for the paper_benchmarks pipeline.

All stages import paths, method styles, and small utilities from here so that
figure conventions and directory layout stay consistent.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PB_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PB_DIR.parent
RAW_DIR = PB_DIR / "raw"
RAW_NEW_DIR = PB_DIR / "raw_new"
PROCESSED_DIR = PB_DIR / "processed"
FIGURES_DIR = PB_DIR / "figures"
TABLES_DIR = PB_DIR / "tables"
LOGS_DIR = PB_DIR / "logs"
CONFIGS_DIR = PB_DIR / "configs"

SINGLE_GATE_DIR = REPO_DIR / "experiments" / "single_gate"
FIXED_RESOURCES_DIR = REPO_DIR / "experiments" / "fixed_resources"

# Locked corrected-campaign conventions (see configs/locked_single_gate.json).
SG_L = 12
SG_SEED_MAIN = 11
SG_SEEDS = (11, 22, 33)
SG_Q0, SG_Q1 = 2, 9
SG_CHI0 = 8
SG_CHI_GRID = (8, 12, 16)
SG_GATES = ("rxx", "ryy", "rzz")
SG_METHODS = ("hybrid_tdvp", "tebd_swap", "mpo_zipup", "variational_mpo")
SG_ANGLE_TDVP_SUBSTEPS = 1  # corrected production value (repair protocol tdvp_n1_v1)

# Exact-limit substep study (predeclared; theta/(2pi) = 1/4 is a corrected special angle).
SUBSTEP_STUDY_X = 0.25
SUBSTEP_STUDY_CHI = 32  # provably nonbinding: exact bond after one long-range gate <= 16
SUBSTEP_STUDY_NS = (1, 2, 4, 8, 16, 32, 64, 128, 256)

CIRCUIT_CHI_MAIN = 32
CIRCUIT_DT = 0.1
CIRCUIT_TIMESTEPS = 30
CIRCUIT_TDVP_SUBSTEPS = 2

# 1D chain circuit benchmark (same dt/steps/chi as the 2D benchmark).
# TFIM: J = g = 1; XXX Heisenberg: J = h = 1. All two-qubit gates are
# nearest-neighbour, and the TDVP method routes every one of them through the
# gate-local TDVP window update (gate_mode="full-tdvp", method "full_tdvp").
CIRCUIT_1D_L = 16
CIRCUIT_1D_MODELS = ("ising_1d", "heisenberg_1d")
CIRCUIT_1D_METHODS = ("full_tdvp", "tebd_swap", "mpo_zipup")

# PRA figure geometry.
SINGLE_COL_IN = 3.375
DOUBLE_COL_IN = 7.0
PNG_DPI = 450

# Method identity (Okabe-Ito, colorblind safe), locked across every figure.
METHOD_STYLES: dict[str, dict[str, Any]] = {
    "hybrid_tdvp": {
        "label": "hybrid gate-local TDVP",
        "label_single": "gate-local TDVP",
        "color": "#D55E00",  # vermilion
        "marker": "o",
        "linestyle": "-",
    },
    # Same visual identity as hybrid_tdvp: it is the same gate-local TDVP
    # update, applied to every two-qubit gate (1D circuits, all gates NN).
    "full_tdvp": {
        "label": "gate-local TDVP (all gates)",
        "label_single": "gate-local TDVP",
        "color": "#D55E00",  # vermilion
        "marker": "o",
        "linestyle": "-",
    },
    "tebd_swap": {
        "label": "TEBD+SWAP",
        "label_single": "TEBD+SWAP",
        "color": "#0072B2",  # blue
        "marker": "^",
        "linestyle": "--",
    },
    "mpo_zipup": {
        "label": "MPO zip-up",
        "label_single": "MPO zip-up",
        "color": "#009E73",  # bluish green
        "marker": "s",
        "linestyle": "-.",
    },
    "variational_mpo": {
        "label": "variational MPO",
        "label_single": "variational MPO",
        "color": "#CC79A7",  # reddish purple
        "marker": "D",
        "linestyle": ":",
    },
    "no_update": {
        "label": "no update",
        "label_single": "no update",
        "color": "0.45",
        "marker": "",
        "linestyle": (0, (1.2, 1.2)),
    },
}
REFERENCE_COLOR = "0.35"

GATE_LABELS = {"rxx": r"$R_{XX}$", "ryy": r"$R_{YY}$", "rzz": r"$R_{ZZ}$"}

PLOT_FLOOR = 1e-16  # display floor only; stored data are never modified


def apply_pra_style() -> None:
    """RevTeX-compatible matplotlib style shared by all figures."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.8,
        "lines.markeredgewidth": 0.7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.0,
        "xtick.minor.size": 1.7,
        "ytick.major.size": 3.0,
        "ytick.minor.size": 1.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.45,
        "ytick.minor.width": 0.45,
        "legend.frameon": False,
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def panel_label(ax, text: str) -> None:
    """Consistent panel label position: top-left inside the axes."""
    ax.text(
        0.03, 0.965, text, transform=ax.transAxes,
        ha="left", va="top", fontsize=9, fontweight="bold",
    )


def save_figure(fig, stem: str) -> tuple[Path, Path]:
    """Save vector PDF plus PNG preview into figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES_DIR / f"{stem}.pdf"
    png = FIGURES_DIR / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=PNG_DPI)
    return pdf, png


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_DIR, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def limit_blas_threads(n: int = 1) -> None:
    """Avoid BLAS oversubscription in worker processes."""
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(var, str(n))


def add_experiment_path(name: str) -> None:
    """Put one experiment package directory on sys.path.

    Only ever add a single experiment directory per process: both
    experiments/single_gate and experiments/fixed_resources define a module
    named ``config`` and must not be imported into the same interpreter.
    """
    path = str(REPO_DIR / "experiments" / name)
    if path not in sys.path:
        sys.path.insert(0, path)


def worker_count(default: int = 4) -> int:
    return max(1, int(os.environ.get("PB_WORKERS", default)))
