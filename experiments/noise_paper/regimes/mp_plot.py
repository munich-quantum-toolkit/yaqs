from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


@dataclass(frozen=True)
class MPPoint:
    """Optional point overlay on the (m, P) plot."""
    m: float        # memory in units of M_B (concurrent B trajectories)
    P: int
    label: str = ""
    marker: str = "o"
    ms: float = 7.0


def plot_m_P_decision_map(
    *,
    alpha: float,
    kappa: float,
    # --- plot ranges ---
    m_range: Tuple[float, float] = (0.5, 1e3),
    P_range: Tuple[int, int] = (1, 256),
    n_m: int = 350,
    n_P: int = 260,
    # --- overlay ---
    points: Optional[Sequence[MPPoint]] = None,
    annotate_points: bool = True,
    title: Optional[str] = None,
    savepath: Optional[str] = None,
    # --- secondary y-axis in GB (optional) ---
    M_B_gb: Optional[float] = None,
) -> None:
    """
    Decision map in (m, P) space for fixed (alpha, kappa), with optional secondary y-axis in GB.
    """
    if alpha < 1.0:
        raise ValueError("alpha must be >= 1.")
    if kappa < 1.0:
        raise ValueError("kappa must be >= 1.")

    m_min, m_max = m_range
    P_min, P_max = P_range
    if m_min <= 0 or m_max <= m_min:
        raise ValueError("Invalid m_range. Must satisfy 0 < m_min < m_max.")
    if P_min < 1 or P_max <= P_min:
        raise ValueError("Invalid P_range. Must satisfy 1 <= P_min < P_max.")

    # --------- Figure style ----------
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # --- Grid ---
    m_vals = np.logspace(np.log10(m_min), np.log10(m_max), n_m)
    # Keep x linear but sample more near low P with log-spaced sampling
    P_vals = np.unique(
        np.round(np.logspace(np.log10(P_min), np.log10(P_max), n_P)).astype(int)
    )
    P_grid, m_grid = np.meshgrid(P_vals, m_vals)

    # --- Effective batching ---
    P_B = np.minimum(P_grid, m_grid)
    P_A = np.minimum(P_grid, m_grid / (alpha**2))

    eps = 1e-12
    R = (1.0 / kappa) * (P_B / np.maximum(P_A, eps)) * (alpha**3)
    feasible_A = m_grid >= alpha**2
    A_faster = (R < 1.0) & feasible_A

    # --- Discrete colormap (2 classes) ---
    # 0 = B faster (or A infeasible), 1 = A faster
    cmap = ListedColormap(["#3b0f70", "#fde725"])  # dark purple / bright yellow (high contrast)
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    Z = A_faster.astype(int)

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.set_yscale("log")

    ax.pcolormesh(P_grid, m_vals[:, None], Z, shading="auto", cmap=cmap, norm=norm)

    ax.set_xlabel("Threads available $P$")
    ax.set_ylabel(r"Memory budget $m=M/M_B$  (concurrent B trajectories)")
    ax.set_xlim(P_min, P_max)
    ax.set_ylim(m_min, m_max)

    if title is None:
        title = rf"Decision map in $(m,P)$ for $\alpha={alpha:g}$, $\kappa={kappa:g}$"
    ax.set_title(title)

    # --- Secondary y-axis ---
    if M_B_gb is not None:
        if M_B_gb <= 0:
            raise ValueError("M_B_gb must be positive (GB per B-trajectory).")

        def m_to_gb(m: float) -> float:
            return m * M_B_gb

        def gb_to_m(M: float) -> float:
            return M / M_B_gb

        secax = ax.secondary_yaxis("right", functions=(m_to_gb, gb_to_m))
        secax.set_ylabel(r"Memory $M$ (GB)")

        ax.text(
            0.02, 0.02,
            rf"Secondary axis uses $M_B \approx {M_B_gb:.3g}$ GB per B trajectory",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.90),
        )

    # --- Boundaries ---
    P_line = np.linspace(P_min, P_max, 800)

    # Use consistent linewidths and line styles
    ax.plot(P_line, np.full_like(P_line, alpha**2),
            lw=2.0, ls="-", color="#2c7fb8",
            label=r"Feasible(A): $m=\alpha^2$")

    ax.plot(P_line, (alpha**2) * P_line,
            lw=2.0, ls="-", color="#f03b20",
            label=r"Thread-limited: $m=\alpha^2 P$")

    ax.plot(P_line, P_line,
            lw=2.0, ls="--", color="#31a354",
            label=r"Memory-limited: $m=P$")

    ax.plot(P_line, (alpha**5 / kappa) * P_line,
            lw=2.0, ls="-.", color="#ff7f00",
            label=r"Decision (mixed): $m=(\alpha^5/\kappa)P$")

    # --- Overlay points ---
    if points:
        for p in points:
            ax.plot(
                p.P, p.m,
                p.marker,
                ms=p.ms,
                color="#7A1E1E",
                markeredgecolor="black",
                markeredgewidth=0.6,
                zorder=5,
            )
            if annotate_points and p.label:
                ax.annotate(
                    p.label,
                    (p.P, p.m),
                    textcoords="offset points",
                    xytext=(8, 6),
                    ha="left",
                    va="bottom",
                    rotation=30,
                    rotation_mode="anchor",
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.88),
                    zorder=6,
                )

    # --- Grid: PRX clean (major only) ---
    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.grid(False, which="minor")

    # --- Legend: include region meaning + boundaries ---
    region_handles = [
        Patch(facecolor=cmap(0), edgecolor="none", label="B faster (or A infeasible)"),
        Patch(facecolor=cmap(1), edgecolor="none", label="A faster"),
    ]
    boundary_legend = ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.add_artist(boundary_legend)
    ax.legend(handles=region_handles, loc="upper left", frameon=True, fontsize=9, title="Regions")

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    plt.show()


# -------- Example usage --------
if __name__ == "__main__":
    # Heisenberg
    L = 48
    chi_B = 1024

    # Ising
    # L = 64
    # chi_B = 512

    bytes_per_complex = 16  # complex128
    overhead = 2.0          # illustrative; calibrate from measurement if possible
    M_B_gb = overhead * (L * chi_B**2 * bytes_per_complex) / 1e9

    # (1.8, 8), (2.2, 14), (2.7, 30), (3.2, 55), (4.0, 120)
    plot_m_P_decision_map(
        alpha=1.8,
        kappa=8.0,
        m_range=(0.5, 1e3),
        P_range=(1, 64),
        M_B_gb=M_B_gb,
        points=[
            MPPoint(m=8.0 / M_B_gb,   P=4,  label="Edge device"),
            MPPoint(m=16.0 / M_B_gb,  P=8,  label="Laptop"),
            MPPoint(m=24.0 / M_B_gb,  P=16, label="GPU"),
            MPPoint(m=64.0 / M_B_gb,  P=16, label="Desktop"),
            MPPoint(m=256.0 / M_B_gb, P=64, label="HPC node"),
        ],
        savepath=None,
    )
