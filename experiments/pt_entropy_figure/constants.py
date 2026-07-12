"""Shared constants for the S_V versus S_MPO comparison figure."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_CSV = PACKAGE_DIR / "data" / "entropy_vs_J.csv"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "output"

CUTS = (1, 2, 3)
SPECTRUM_CUT = 2

# Ising-chain benchmark defaults (L=6, k=3, dt=0.1, g=1).
L_DEFAULT = 6
K_DEFAULT = 3
DT_DEFAULT = 0.1
G_DEFAULT = 1.0
BETA = 1.0

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
    return {
        jv: to_hex(cmap(lo + (hi - lo) * i / (len(SPECTRUM_JS) - 1)))
        for i, jv in enumerate(SPECTRUM_JS)
    }


J_COLORS = _j_colors_from_reds()

OUTPUT_STEM = "sv_ptcb_entropy_spectra"
