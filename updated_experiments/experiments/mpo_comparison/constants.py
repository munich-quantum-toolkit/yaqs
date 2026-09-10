# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared constants for the updated response/PT-MPO comparison figure."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PACKAGE_DIR.parents[1] / "results" / "figure4"
DEFAULT_DATA_CSV = DEFAULT_OUTPUT_DIR / "entropy_vs_J.csv"
DEFAULT_SPECTRA_NPZ = DEFAULT_OUTPUT_DIR / "spectra.npz"

CUTS = (1, 2, 3)
SPECTRUM_CUT = 2

# Ising-chain benchmark defaults (L=6, k=3, dt=0.1, g=1).
L_DEFAULT = 6
K_DEFAULT = 3
DT_DEFAULT = 0.1
G_DEFAULT = 1.0
SPECTRUM_DISCARDED_WEIGHT_THRESHOLD = 1e-12

# Numerical thresholds matching the diagnostic pipeline.
RANK_RTOL = 1e-12
WEIGHT_TOL = 1e-30
CUMULATIVE_CAP = 1.0 - 1e-8
SPECTRUM_MAX_MODES = 8
SPECTRUM_MIN_MODES = 6
SPECTRUM_SUM_TOL = 1e-10

# Panel-(b) coupling grid and display floors for log-axis limits (display only).
SPECTRUM_JS = (0.1, 1.0, 2.0, 4.0)
PANEL_A_Y_FLOOR = 1e-7  # ignore numerically negligible entropies when setting y-limits
PANEL_A_J_MIN = 0.2  # exclude weak-coupling numerical floor from y-limit selection
PANEL_A_MARKER_EVERY = 5  # plot every Nth marker in panel (a); line uses all points
PANEL_B_Y_DISPLAY_FLOOR = 1e-18  # spectrum y-limit floor (display only; does not clip weights)
PANEL_BC_Y_CEIL = 1.05  # probabilities sum to unity; cap ymax for tighter zoom

CUT_COLORS = {1: "#009E73", 2: "#0072B2", 3: "#D55E00"}


def _j_colors_from_reds() -> dict[float, str]:
    """Sample the matplotlib Reds colormap across ``SPECTRUM_JS`` (weak to strong J)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    cmap = plt.get_cmap("Reds")
    if len(SPECTRUM_JS) == 1:
        return {SPECTRUM_JS[0]: to_hex(cmap(0.7))}
    lo, hi = 0.35, 0.95  # skip near-white for legibility on white background
    return {jv: to_hex(cmap(lo + (hi - lo) * i / (len(SPECTRUM_JS) - 1))) for i, jv in enumerate(SPECTRUM_JS)}


J_COLORS = _j_colors_from_reds()

OUTPUT_STEM = "mpo_comparison"
