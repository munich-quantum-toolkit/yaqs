from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, List, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches


@dataclass(frozen=True)
class MPPoint:
    """Optional point overlay on the (M, P) plot."""
    M_gb: float              # Physical memory in GB
    P: int
    label: str = ""          # kept for caption use; not drawn on-plot
    marker: str = "s"
    ms: float = 8.5          # Increased to 8.5


def draw_m_P_decision_map_ax(
    ax: plt.Axes,
    *,
    alpha: float,
    kappa: float,
    M_range: Tuple[float, float], # Range in GB
    P_range: Tuple[int, int],
    M_B_gb: float,                # Mandatory now
    n_m: int = 350,
    n_P: int = 260,
    points: Optional[Sequence[MPPoint]] = None,
    annotate_points: bool = True,
    # Toggles for triptych
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_right_axis: bool = True, # Ignored, right axis removed
    show_region_labels: bool = False, 
    show_crossover: bool = True,  
    show_dominance_key: bool = False, # New toggle
    panel_title: str = "",
) -> None:
    """Helper: draws the decision map on an existing axis (Physical Memory)."""

    M_min, M_max = M_range
    P_min, P_max = P_range
    if not (0.0 < M_min < M_max):
        raise ValueError("Invalid M_range (GB). Must satisfy 0 < M_min < M_max.")
    if not (1 <= P_min < P_max):
        raise ValueError("Invalid P_range. Must satisfy 1 <= P_min < P_max.")

    # --- grids ---
    M_vals = np.logspace(np.log10(M_min), np.log10(M_max), n_m)
    P_vals = np.unique(np.round(np.logspace(np.log10(P_min), np.log10(P_max), n_P)).astype(int))
    P_grid, M_grid = np.meshgrid(P_vals, M_vals)
    
    # Calculate equivalent dimensionless m for decision logic
    m_grid = M_grid / M_B_gb

    # --- decision map (logic uses m_grid) ---
    P_B = np.minimum(P_grid, m_grid)
    P_A = np.minimum(P_grid, m_grid / (alpha**2))

    eps = 1e-12
    R = (1.0 / kappa) * (P_B / np.maximum(P_A, eps)) * (alpha**3)
    feasible_A = m_grid >= alpha**2
    Z = ((R < 1.0) & feasible_A).astype(int)

    # --- background ---
    color_B_orig = "#C9D6E8"
    color_A_orig = "#D8EBD2"
    
    cmap = ListedColormap([color_B_orig, color_A_orig])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    ax.set_yscale("log")
    # Plot using M_vals (GB)
    ax.pcolormesh(P_grid, M_vals[:, None], Z, shading="auto", cmap=cmap, norm=norm, alpha=0.6)

    # 1) Dominance Key (Panel A only usually)
    if show_dominance_key:
        # Create small custom legend for dominance
        # "draw two small rectangles ... label them A faster and B faster"
        # "Place it in an empty corner (e.g., top-right) with fontsize ~7.5, no box"
        
        # We can use ax.legend with custom handles
        patch_A = mpatches.Patch(color=color_A_orig, label="A faster", alpha=0.6, ec="black", lw=0.5) 
        patch_B = mpatches.Patch(color=color_B_orig, label="B faster", alpha=0.6, ec="black", lw=0.5)
        
        # Use a separate legend instance or add patches to plot?
        # Standard legend is easiest.
        leg_dom = ax.legend(
            handles=[patch_A, patch_B],
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            fontsize=7.5,
            frameon=False,
            handlelength=1.2,
            handletextpad=0.5,
            borderaxespad=0.2
        )
        # We might need to add this manually if we want multiple legends?
        # But we don't have other legends on this axis usually.

    if show_xlabel:
        ax.set_xlabel(r"Threads $P$")
    if show_ylabel:
        ax.set_ylabel(r"Memory $M$ (GB)") 

    ax.set_xlim(P_min, P_max)
    ax.set_ylim(M_min, M_max)

    # Panel title (top-left)
    if panel_title:
        ax.text(
            0.02, 0.96, panel_title,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=9.0, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.0, pad=0.0)
        )

    # Parameter text
    y_param = 0.88 if panel_title else 0.98
    ax.text(
        0.02,
        y_param,
        rf"$\alpha={alpha:g},\ \kappa={kappa:g}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=0.7),
    )

    # --- boundary lines (Physical M) ---
    P_line = np.linspace(P_min, P_max, 800)
    
    lw_feas = 1.1
    lw_limit = 1.3
    lw_mix = 2.3
    
    col_feas = "#333333"
    col_thr  = "#2E7D32"
    col_mem  = "#1565C0"
    col_x    = "#D84315"

    # Feasible (A): M = (alpha^2) * M_B
    ax.plot(P_line, np.full_like(P_line, (alpha**2) * M_B_gb),
            lw=lw_feas, color=col_feas, ls="-", label="Feasible (A)")

    # Thread-limited: M = (alpha^2) * M_B * P
    ax.plot(P_line, (alpha**2) * M_B_gb * P_line,
            lw=lw_limit, color=col_thr, ls="--", label="Thread-limited")

    # Memory-limited: M = M_B * P
    ax.plot(P_line, M_B_gb * P_line,
            lw=lw_limit, color=col_mem, linestyle=(0, (1, 1)), label="Memory-limited")

    # Crossover: M = (alpha^5 / kappa) * M_B * P
    if show_crossover:
        ax.plot(P_line, (alpha**5 / kappa) * M_B_gb * P_line,
                lw=lw_mix, color=col_x, linestyle=(0, (6, 3)), label="Crossover")
    # else:
    #     # 2) "B dominates" text removed per request
    #     pass

    # --- overlay points (Physical M) ---
    if points:
        for i, p in enumerate(points, start=1):
            
            marker_color = "#FF9800" 
            
            # 3) Center numbers
            # "Stop using the manual y-shift... Place number text at exactly (p.P, p.M)"
            # "Only apply small x/y nudges when marker is near axes boundaries"
            
            x_marker = p.P
            y_marker = p.M_gb

            ax.plot(
                x_marker, y_marker,
                p.marker,
                ms=p.ms,
                color=marker_color,
                markeredgecolor="black",
                markeredgewidth=1.0, # Slightly thicker edge? Or keeping 0.8? User said "reduce text stroke" but for marker edge "Slightly increase marker size". 
                                     # Let's keep 0.8 or maybe 1.0 looks cleaner with larger marker.
                zorder=9,
            )

            if annotate_points:
                x_text = x_marker
                y_text = y_marker*0.98
                
                # Small boundary nudges only
                # If marker is very close to edge, maybe shift text slightly 
                # but "Place number text at exactly (p.P, p.M) with ha=center, va=center" implies strict centering.
                # "Only apply small x/y nudges when marker is near axes boundaries" 
                # -> Implicitly this means shifting ONLY for boundary cases.
                
                # Check boundaries.
                # If too close to right edge P_max
                if p.P >= P_max * 0.95:
                    x_text = p.P * 0.9 # Shift text slightly left? Or shift marker? 
                    # "Only apply small x/y nudges" -> usually applied to text placement relative to marker or BOTH.
                    # Given the "Place number text at exactly (p.P, p.M)" instruction is strong, 
                    # I will interpret "nudges" as moving the *whole assembly* or just text if it clips.
                    # But points are fixed hardware points. I shouldn't move the data point.
                    # If I move the text, it's no longer centered.
                    # I will stick to exact centering unless it CLIPS the plot area.
                    # The clipping logic in matplotlib `clip_on=False` handles visibility.
                    # Aesthetic nudge:
                    # If marker is cut off, maybe that's bad.
                    # I will respect "Place number text at exactly... ha=center, va=center" as the primary rule.
                    pass 
                
                ax.text(
                    x_text, y_text, f"{i}",
                    ha="center", va="center",
                    fontsize=7.0,
                    fontweight="bold",
                    color="white",
                    # Reduced stroke width 1.2 -> 1.0
                    path_effects=[pe.withStroke(linewidth=1.0, foreground="black")],
                    zorder=10,
                    clip_on=False,
                )

    # --- grid ---
    ax.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.4)
    ax.grid(False, which="minor")


def plot_m_P_decision_map_prx(
    *,
    alpha: float,
    kappa: float,
    m_range: Tuple[float, float] = (0.5, 1e3),
    P_range: Tuple[int, int] = (1, 64),
    n_m: int = 350,
    n_P: int = 260,
    points: Optional[Sequence[MPPoint]] = None,
    annotate_points: bool = True,   
    savepath: Optional[str] = None,
    M_B_gb: Optional[float] = None,
) -> None:
    """Legacy wrapper. Requires M_B_gb now to convert m_range to M_range."""
    if M_B_gb is None:
        raise ValueError("M_B_gb is required for physical axis conversion.")

    M_range = (m_range[0] * M_B_gb, m_range[1] * M_B_gb)
    
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

    fig, ax = plt.subplots(figsize=(3.375, 2.75), constrained_layout=True)

    draw_m_P_decision_map_ax(
        ax,
        alpha=alpha,
        kappa=kappa,
        M_range=M_range, 
        P_range=P_range,
        n_m=n_m,
        n_P=n_P,
        points=points,
        annotate_points=annotate_points,
        M_B_gb=M_B_gb,
        show_xlabel=True,
        show_ylabel=True,
        show_region_labels=True, 
        show_crossover=True,
        show_dominance_key=True,
        panel_title="",
    )

    h, l = ax.get_legend_handles_labels()
    # Filter out dominance patches from main legend if mixed
    # But get_legend_handles_labels() might capture them.
    # Actually, main legend is mostly lines. Patches are A/B faster.
    # We probably want lines here.
    leg_handles = [x for x in h if isinstance(x, plt.Line2D)]
    leg_labels = [l[i] for i, x in enumerate(h) if isinstance(x, plt.Line2D)]

    ax.legend(
        handles=leg_handles, labels=leg_labels,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.04),
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
        fig.savefig(savepath, dpi=300)
    plt.show()


def plot_m_P_decision_map_triptych_prx(
    scenarios: Sequence[dict],
    M_range: Tuple[float, float], # Physical GB
    P_range: Tuple[int, int],
    points: Optional[Sequence[MPPoint]], # Physical GB
    M_B_gb: float,
    savepath: Optional[str] = None,
) -> None:
    """
    Creates a single-column figure with three vertically stacked decision maps.
    """
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

    # 3.375" width. Height ~7.6"
    fig, axes = plt.subplots(3, 1, figsize=(3.375, 7.6), sharex=True, sharey=True)

    if len(scenarios) != 3:
        raise ValueError("Expected exactly 3 scenarios for the triptych.")

    # Manual margin adjustment for legend + supxlabel
    # bottom=0.13 as requested
    fig.subplots_adjust(left=0.16, right=0.96, top=0.98, bottom=0.13, hspace=0.05)

    for i, ax in enumerate(axes):
        s = scenarios[i]
        
        is_bottom = (i == 2)
        is_middle = (i == 1)
        
        show_cross = (i != 2)
        
        # Dominance key on ALL panels now
        show_dom = True

        draw_m_P_decision_map_ax(
            ax,
            alpha=s["alpha"],
            kappa=s["kappa"],
            M_range=M_range,
            P_range=P_range,
            points=points,
            M_B_gb=M_B_gb, 
            show_xlabel=False, 
            show_ylabel=is_middle,
            show_right_axis=False,
            show_region_labels=False, 
            show_crossover=show_cross,
            show_dominance_key=show_dom,
            panel_title=s.get("panel_title", ""),
        )
    
    # Shared X-label (supxlabel)
    # Move closer to axis (y=0.085 instead of 0.06)
    fig.supxlabel(r"Threads $P$", y=0.085, fontsize=9.0)

    # Figure-level legend
    # Need lines, not patches (which are in panel a's legend now)
    # Axes 0 has patches in legend handles? Maybe.
    # Axes 2 (bottom) certainly has the lines but NO PATCHES.
    # Use axes[0] for crossover? Axes 2 has no crossover plotted!
    # Axes 0 has crossover AND patches.
    # We want to exclude patches from the bottom figure legend.
    
    h0, l0 = axes[0].get_legend_handles_labels()
    # Filter for Line2D
    lines = [h for h in h0 if isinstance(h, plt.Line2D)]
    labels = [l0[i] for i, h in enumerate(h0) if isinstance(h, plt.Line2D)]
    
    fig.legend(
        lines, labels,
        loc="lower center",
        bbox_to_anchor=(0.56, 0.01),
        ncol=2, 
        frameon=False,
        handlelength=2.2,
        columnspacing=1.5,
        handletextpad=0.5,
        labelspacing=0.3,
        fontsize=7.2,
    )

    if savepath:
        fig.savefig(savepath, dpi=300)
    plt.show()


if __name__ == "__main__":
    L = 1024
    chi_B = 256
    bytes_per_complex = 16 
    overhead = 1.0
    M_B_gb = overhead * (L * chi_B**2 * bytes_per_complex) / 1e9

    scenarios = [
        {"alpha": 2.0, "kappa": 11.0, "panel_title": "(a) Depolarizing"},
        {"alpha": 1.5, "kappa": 3.8,  "panel_title": "(b) Dephasing"},
        {"alpha": 1.5, "kappa": 0.9,  "panel_title": "(c) Bit-flip"},
    ]

    points = [
        MPPoint(M_gb=8.0,   P=4),    # Edge / embedded
        MPPoint(M_gb=16.0,  P=8),    # Developer laptop
        MPPoint(M_gb=64.0,  P=16),   # Research desktop
        MPPoint(M_gb=128.0, P=32),   # Small server / workstation
        MPPoint(M_gb=256.0, P=64),   # HPC node
    ]

    plot_m_P_decision_map_triptych_prx(
        scenarios=scenarios,
        M_range=(3, 1e3), 
        P_range=(1, 68),
        points=points,
        M_B_gb=M_B_gb,
        savepath="mp_triptych.pdf"
    )
