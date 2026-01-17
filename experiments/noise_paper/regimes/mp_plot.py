from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patheffects as pe


@dataclass(frozen=True)
class MPPoint:
    """Optional point overlay on the (m, P) plot."""
    m: float
    P: int
    label: str = ""          # kept for caption use; not drawn on-plot
    marker: str = "s"
    ms: float = 6.0


def plot_m_P_decision_map_prx(
    *,
    alpha: float,
    kappa: float,
    m_range: Tuple[float, float] = (0.5, 1e3),
    P_range: Tuple[int, int] = (1, 64),
    n_m: int = 350,
    n_P: int = 260,
    points: Optional[Sequence[MPPoint]] = None,
    annotate_points: bool = True,   # numbers only
    savepath: Optional[str] = None,
    M_B_gb: Optional[float] = None,
) -> None:
    """Single-column PRX-style decision map in (m, P) for fixed (alpha, kappa)."""
    if alpha < 1.0:
        raise ValueError("alpha must be >= 1.")
    if kappa < 1.0:
        raise ValueError("kappa must be >= 1.")

    m_min, m_max = m_range
    P_min, P_max = P_range
    if not (0.0 < m_min < m_max):
        raise ValueError("Invalid m_range. Must satisfy 0 < m_min < m_max.")
    if not (1 <= P_min < P_max):
        raise ValueError("Invalid P_range. Must satisfy 1 <= P_min < P_max.")

    # --- PRX/APS-ish style (match sister plot) ---
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.6,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "mathtext.fontset": "stix",
        }
    )

    # APS single-column width ~3.375"
    fig, ax = plt.subplots(figsize=(3.375, 2.75), constrained_layout=True)

    # --- grids ---
    m_vals = np.logspace(np.log10(m_min), np.log10(m_max), n_m)
    P_vals = np.unique(np.round(np.logspace(np.log10(P_min), np.log10(P_max), n_P)).astype(int))
    P_grid, m_grid = np.meshgrid(P_vals, m_vals)

    # --- decision map ---
    P_B = np.minimum(P_grid, m_grid)
    P_A = np.minimum(P_grid, m_grid / (alpha**2))

    eps = 1e-12
    R = (1.0 / kappa) * (P_B / np.maximum(P_A, eps)) * (alpha**3)
    feasible_A = m_grid >= alpha**2
    Z = ((R < 1.0) & feasible_A).astype(int)

    # --- background (print-friendly, 2-class) ---
    cmap = ListedColormap(["#E6E6E6", "#C9D6E8"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    ax.set_yscale("log")
    ax.pcolormesh(P_grid, m_vals[:, None], Z, shading="auto", cmap=cmap, norm=norm)

    ax.set_xlabel(r"Threads $P$")
    ax.set_ylabel(r"Memory $m=M/M_B$")
    ax.set_xlim(P_min, P_max)
    ax.set_ylim(m_min, m_max)

    # in-panel parameter label
    ax.text(
        0.02,
        0.98,
        rf"$\alpha={alpha:g},\ \kappa={kappa:g}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2),
    )

    # --- secondary y-axis (GB) ---
    if M_B_gb is not None:
        if M_B_gb <= 0:
            raise ValueError("M_B_gb must be positive (GB per B-trajectory).")

        def m_to_gb(m: float) -> float:
            return m * M_B_gb

        def gb_to_m(M: float) -> float:
            return M / M_B_gb

        secax = ax.secondary_yaxis("right", functions=(m_to_gb, gb_to_m))
        secax.set_ylabel(r"$M$ (GB)")
        secax.tick_params(labelsize=8.0)

    # --- boundary lines ---
    P_line = np.linspace(P_min, P_max, 800)
    lw = 1.1

    l_feas, = ax.plot(P_line, np.full_like(P_line, alpha**2), lw=lw, ls="-",  label="Feasible (A)")
    l_thr,  = ax.plot(P_line, (alpha**2) * P_line,           lw=lw, ls="--", label="Thread-limited")
    l_mem,  = ax.plot(P_line, P_line,                         lw=lw, ls=":",  label="Memory-limited")
    l_mix,  = ax.plot(P_line, (alpha**5 / kappa) * P_line,    lw=lw, ls="-.", label="Crossover")

    # --- overlay points: numbers INSIDE squares, with stroke (fixed) ---
    if points:
        for i, p in enumerate(points, start=1):
            ax.plot(
                p.P, p.m,
                p.marker,
                ms=p.ms,
                color="#7A1E1E",
                markeredgecolor="black",
                markeredgewidth=0.7,
                zorder=7,
                clip_on=False,   # helps edge cases at boundaries
            )

            if annotate_points:
                # small inward nudge if we're hugging an axis edge
                x = p.P
                y = p.m
                if p.P >= P_max * 0.95:
                    x = p.P - 0.8
                if p.P <= P_min * 1.2:
                    x = p.P + 0.6
                if p.m >= m_max / 1.1:
                    y = p.m / 1.08

                ax.text(
                    x, y, f"{i}",
                    ha="center", va="center",
                    fontsize=7.2,
                    fontweight="bold",
                    color="white",
                    path_effects=[pe.withStroke(linewidth=1.2, foreground="black")],
                    zorder=8,
                    clip_on=False,
                )

    # --- grid ---
    ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.55)
    ax.grid(False, which="minor")

    # --- legend inside plot (2x2), slightly higher to avoid clipping ---
    ax.legend(
        handles=[l_feas, l_thr, l_mem, l_mix],
        loc="lower center",
        bbox_to_anchor=(0.50, 0.065),  # was 0.05; reduce edge overlap
        ncol=2,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.0,
        handletextpad=0.6,
        labelspacing=0.35,
        borderaxespad=0.0,
        fontsize=7.2,
    )

    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    L = 512
    chi_B = 256
    bytes_per_complex = 16 # Could also be 8
    overhead = 2.0
    M_B_gb = overhead * (L * chi_B**2 * bytes_per_complex) / 1e9

    plot_m_P_decision_map_prx(
        alpha=1.8,
        kappa=8.0,
        m_range=(0.5, 1e3),
        P_range=(1, 68),
        M_B_gb=M_B_gb,
        points=[
            MPPoint(m=8.0  / M_B_gb, P=4),    # Edge / embedded
            MPPoint(m=16.0 / M_B_gb, P=8),    # Developer laptop
            MPPoint(m=64.0 / M_B_gb, P=16),   # Research desktop
            MPPoint(m=128.0 / M_B_gb, P=32),  # Small server / workstation
            MPPoint(m=256.0 / M_B_gb, P=64),  # HPC node
        ],
        savepath=None,
    )
