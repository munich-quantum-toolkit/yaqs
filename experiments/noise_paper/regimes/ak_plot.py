from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PhasePoint:
    """A labeled point to overlay on the phase diagram."""
    alpha: float
    kappa: float
    label: str = ""
    marker: str = "s"   # keep square as the default marker
    ms: float = 6.0


def plot_alpha_kappa_phase_prx(
    *,
    alpha_range: Tuple[float, float] = (1.0, 5.0),
    kappa_range: Tuple[float, float] = (1.0, 1e3),
    n: int = 500,
    logx: bool = False,
    points: Optional[Sequence[PhasePoint]] = None,
    annotate_points: bool = True,
    title: Optional[str] = None,
    savepath: Optional[str] = None,
) -> None:
    a0, a1 = alpha_range
    k0, k1 = kappa_range
    if a0 < 1.0 or k0 < 1.0:
        raise ValueError("Require alpha_range[0] >= 1 and kappa_range[0] >= 1.")
    if a1 <= a0 or k1 <= k0:
        raise ValueError("Invalid ranges: ensure max > min for both axes.")

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "mathtext.fontset": "stix",
        }
    )

    fig, ax = plt.subplots(figsize=(3.375, 2.55), constrained_layout=True)

    if logx:
        a = np.logspace(np.log10(a0), np.log10(a1), n)
        ax.set_xscale("log")
    else:
        a = np.linspace(a0, a1, n)

    k3 = a**3
    k5 = a**5

    ax.set_yscale("log")
    ax.set_xlim(a0, a1)
    ax.set_ylim(k0, k1)

    k3c = np.clip(k3, k0, k1)
    k5c = np.clip(k5, k0, k1)

    ax.fill_between(a, k0, k3c, alpha=0.12, linewidth=0)
    ax.fill_between(a, k3c, k5c, alpha=0.08, linewidth=0)
    ax.fill_between(a, k5c, k1, alpha=0.12, linewidth=0)

    ax.plot(a, k3, lw=1.2, label=r"$\kappa=\alpha^3$")
    ax.plot(a, k5, lw=1.2, label=r"$\kappa=\alpha^5$")

    x_b = min(a1, max(a0, (a0 + 0.55 * (a1 - a0)))) if not logx else np.sqrt(a0 * (a0 * (a1 / a0) ** 0.65))
    x_m = min(a1, max(a0, (a0 + 0.62 * (a1 - a0)))) if not logx else np.sqrt(a0 * (a0 * (a1 / a0) ** 0.75))
    x_a = min(a1, max(a0, (a0 + 0.2 * (a1 - a0)))) if not logx else np.sqrt(a0 * (a0 * (a1 / a0) ** 0.55))

    k3_b, k5_b = x_b**3, x_b**5
    k3_m, k5_m = x_m**3, x_m**5
    k3_a, k5_a = x_a**3, x_a**5

    def clamp(y: float) -> float:
        return float(min(max(y, k0 * 1.05), k1 / 1.05))

    y_b = clamp(k3_b / 6.0)
    y_m = clamp(np.sqrt(k3_m * k5_m) * 1.15)
    y_a = clamp(k5_a * 10.0)

    label_box = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5)

    ax.text(
        x_b, y_b, "B faster",
        ha="center", va="center",
        fontweight="bold", fontsize=8.0,
        bbox=label_box,
    )

    # Better label than "Depends" + rotate 45° to match the diagonal band
    middle_label = "Regime-dependent"
    ax.text(
        x_m, y_m, middle_label,
        ha="center", va="center",
        fontweight="bold", fontsize=7.4,
        rotation=25,
        rotation_mode="anchor",
        bbox=label_box,
    )

    ax.text(
        x_a, y_a, "A faster",
        ha="center", va="center",
        fontweight="bold", fontsize=8.0,
        bbox=label_box,
    )

    ax.set_xlabel(r"$\alpha=\chi_A/\chi_B$  (bond dimension inflation)")
    ax.set_ylabel(r"$\kappa=N_B/N_A$  (trajectory inflation)")

    ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)

    if title:
        ax.set_title(title, pad=2.0)

    if points:
        for p in points:
            ax.plot(
                p.alpha, p.kappa, p.marker,
                ms=p.ms,
                color="#7A1E1E",
                markeredgecolor="black",
                markeredgewidth=0.7,
                zorder=6,
            )
            if annotate_points and p.label:
                ax.annotate(
                    p.label,
                    (p.alpha, p.kappa),
                    textcoords="offset points",
                    xytext=(5, 1),
                    ha="left",
                    va="center",
                    fontsize=7.6,
                    rotation=30,
                    rotation_mode="anchor",
                )

    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        borderpad=0.3,
        handlelength=1.8,
        handletextpad=0.6,
        labelspacing=0.25,
    )

    if savepath:
        fig.savefig(savepath, dpi=300)
    plt.show()


if __name__ == "__main__":
    # demo_points = [
    #     PhasePoint(alpha=1.8, kappa=5.0, label=r"$(\alpha,\kappa)=(1.8,5)$", marker="o"),
    # ]
    plot_alpha_kappa_phase_prx(
        alpha_range=(1.0, 5.0),
        kappa_range=(1.0, 1e3),
        # points=demo_points,
        savepath="ak_plot.pdf",  # e.g. "alpha_kappa_phase.pdf"
    )
