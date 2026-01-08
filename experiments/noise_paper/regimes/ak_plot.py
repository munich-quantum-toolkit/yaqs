from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PhasePoint:
    """A labeled point to overlay on the phase diagram."""
    alpha: float
    kappa: float
    label: str = ""
    marker: str = "o"
    ms: float = 6.0  # marker size


def plot_alpha_kappa_phase_prx(
    *,
    alpha_range: Tuple[float, float] = (1.0, 16.0),
    kappa_range: Tuple[float, float] = (1.0, 1e6),
    n: int = 600,
    points: Optional[Sequence[PhasePoint]] = None,
    annotate_points: bool = True,
    title: Optional[str] = None,
    savepath: Optional[str] = None,
) -> None:
    r"""
    Publication-ready (alpha, kappa) phase diagram.

    Assumptions / definitions (choose A as higher-chi, lower-N; B as lower-chi, higher-N):
      alpha = chi_A / chi_B >= 1
      kappa = N_B / N_A >= 1

    Regime logic:
      - Thread-limited:     A faster if alpha^3 < kappa, else B faster.
      - Memory-limited:     A faster if alpha^5 < kappa, else B faster.
      -> "Flip region" where decision depends on regime: alpha^3 < kappa < alpha^5

    Notes:
      - This is a *conceptual* phase diagram; it is independent of absolute L, chi, N.
      - Overlay points (alpha,kappa) computed from your benchmarks to make it data-driven.
    """
    a0, a1 = alpha_range
    k0, k1 = kappa_range
    if a0 < 1.0:
        raise ValueError("alpha_range[0] must be >= 1 because alpha = chi_A/chi_B >= 1.")
    if k0 < 1.0:
        raise ValueError("kappa_range[0] must be >= 1 because kappa = N_B/N_A >= 1.")
    if a1 <= a0 or k1 <= k0:
        raise ValueError("Invalid ranges: ensure max > min for both axes.")

    # --------- Style tweaks for paper figures ----------
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,   # embed fonts nicely
        "ps.fonttype": 42,
    })

    # --------- Grid in log space ----------
    a = np.logspace(np.log10(a0), np.log10(a1), n)
    k3 = a**3
    k5 = a**5

    # Shading regions via fill_between on log axes works fine as long as we clip
    # Region definitions:
    #   B wins both: kappa < alpha^3
    #   Flip:        alpha^3 < kappa < alpha^5
    #   A wins both: kappa > alpha^5
    k3_clip = np.clip(k3, k0, k1)
    k5_clip = np.clip(k5, k0, k1)

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    # ax.set_xscale("log")
    ax.set_yscale("log")

    # --- Shaded regions (no explicit colors requested; matplotlib defaults) ---
    # Use alpha for transparency; don't set explicit colors to keep styling neutral.
    # Region: B wins (below k3)
    ax.fill_between(a, k0, k3_clip, alpha=0.10, linewidth=0)

    # Region: Flip (between k3 and k5)
    ax.fill_between(a, k3_clip, k5_clip, alpha=0.08, linewidth=0)

    # Region: A wins (above k5)
    ax.fill_between(a, k5_clip, k1, alpha=0.10, linewidth=0)

    # --- Boundary curves ---
    ax.plot(a, k3, lw=2.0, label=r"$\kappa=\alpha^3$ (thread-limited boundary)")
    ax.plot(a, k5, lw=2.0, label=r"$\kappa=\alpha^5$ (memory-limited boundary)")

    # --- Region labels (placed at representative points) ---
    # Choose points safely inside the plotting window.
    # B-wins region label: pick alpha close to 1, kappa a bit above 1
    ax.text(
        x=3.5,
        y=5, # min(max(1.5, k0 * 1.8), k1 / 50),
        s="Unraveling B faster",
        ha="center",
        va="center",
        fontweight="bold",
    )

    ax.text(
        x=4,
        y=200, # min(max(1.5, k0 * 1.8), k1 / 50),
        s="A faster if thread-limited\nB faster if memory-limited",
        ha="center",
        va="center",
        fontweight="bold",
    )

    # A-wins region label: pick alpha large and kappa above a^5
    ax.text(
        x=2,
        y=300, # min(max(1.5, k0 * 1.8), k1 / 50),
        s="Unraveling A faster",
        ha="center",
        va="center",
        fontweight="bold",
    )

    # --- Axes labels ---
    ax.set_xlabel(r"$\alpha=\chi_A/\chi_B$  (bond dimension inflation)")
    ax.set_ylabel(r"$\kappa=N_B/N_A$  (trajectory inflation)")

    ax.set_xlim(a0, a1)
    ax.set_ylim(k0, k1)

    # --- Grid: subtle, publication style ---
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.35)

    # --- Optional overlay points ---
    if points:
        for p in points:
            ax.plot(
                p.alpha,
                p.kappa,
                p.marker,
                ms=p.ms,
                color="#7A1E1E",
                markeredgecolor="black",  # optional, improves contrast
                zorder=5,
            )
            if annotate_points and p.label:
                ax.annotate(
                    p.label,
                    (p.alpha, p.kappa),
                    textcoords="offset points",
                    xytext=(1, 15),
                    ha="center",
                    va="center",
                    rotation=30,
                    rotation_mode="anchor",
                )


    # --- Legend ---
    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()

    if savepath:
        # Recommended: PDF or SVG for vector export
        fig.savefig(savepath, bbox_inches="tight")
    plt.show()


# ---------------- Example usage ----------------
if __name__ == "__main__":
    demo_points = [
        PhasePoint(alpha=2.0, kappa=10.0, label="$(\\alpha,\\kappa)=(2,10)$", marker="s"),
        # Add your measured points here...
        # PhasePoint(alpha=1.6, kappa=5.0, label="Ising, dt=0.1", marker="o"),
    ]
    plot_alpha_kappa_phase_prx(
        alpha_range=(1.0, 10),
        kappa_range=(1.0, 1e3),
        points=demo_points,
        savepath=None,   # e.g. "alpha_kappa_phase.pdf"
    )
