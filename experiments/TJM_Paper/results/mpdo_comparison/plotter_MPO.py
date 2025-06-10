import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

def plot_MPDO_data():
    # ── Global LaTeX styling ─────────────────────────────────────────────────
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{newtxtext}\usepackage{newtxmath}",
        "lines.linewidth": 1.5,
    })

    # ── Outer Grid: 4 rows, 2 columns ─────────────────────────────────────────
    fig = plt.figure(figsize=(7.2, 3.6))
    outer_gs = GridSpec(4, 2, figure=fig,
                        width_ratios=[3, 2],
                        wspace=0.3,
                        hspace=0.35)

    # ── Nested Grid in left column ─────────────────────────────────────────────
    left_gs = GridSpecFromSubplotSpec(
        4, 2,
        subplot_spec=outer_gs[:, 0],
        width_ratios=[1, 0.05],
        wspace=0.05,
        hspace=0.35
    )

    # Create 4 heatmap axes and one colorbar axis
    heatmap_axes = [fig.add_subplot(left_gs[i, 0]) for i in range(4)]
    cbar_ax      = fig.add_subplot(left_gs[:, 1])

    # ── Filenames + titles ────────────────────────────────────────────────────
    files_and_titles = [
        ("30L_Bond2.pickle",  "$\\chi = 2$"),
        ("30L_Bond4.pickle",  "$\\chi = 4$"),
        ("30L_Bond8.pickle",  "$\\chi = 8$"),
        # ("30L_Bond16.pickle", "$\\chi = 16$"),
        # ("30L_Bond32.pickle", "$\\chi = 32$"),
        ("lindblad_mpo_results.pkl", "MPO ($D_s=400$)"),
    ]

    # ── Preload each heatmap (shape T×L) into all_hmaps ───────────────────────
    all_hmaps = []
    for (filename, _) in files_and_titles:
        with open(filename, "rb") as f:
            data = pickle.load(f)

        if "z_expectation_values_mpo" in data:
            hmap = np.array(data["z_expectation_values_mpo"])  # (T, L)
        else:
            hmap = np.array([obs.results for obs in data["sim_params"].observables])  # (L, T)

        all_hmaps.append(hmap)

    # ── Common heatmap settings ─────────────────────────────────────────────────
    L      = 30
    t_max  = 10.0
    xticks = list(range(0, 11))  # 0, 1, ..., 10

    for i, ax in enumerate(heatmap_axes):
        ax.set_ylim(L, 1)  # invert y so site=1 is at top
        ax.set_yticks([1, 10, 20, 30])
        ax.set_yticklabels([1, 10, 20, 30], fontsize=10)

        # Vertical dashed line at Jt=4
        # ax.axvline(4.0, color="white", linestyle="--", linewidth=2)
        ax.set_ylabel("Site", fontsize=12)
        if i == 3:
            ax.set_xlim(0, t_max)
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(x) for x in xticks], fontsize=10)
            ax.set_xlabel(r"Time $(Jt)$", fontsize=12)

            # # Add "Truncated" / "Exact" around Jt=4
            # ax.text(4.2, 13, "Truncated", color="white", fontsize=11,
            #         va="top", ha="left")
            # ax.text(3.8, 13, "Exact", color="white", fontsize=11,
            #         va="top", ha="right")
        else:
            ax.set_xticks([])

        # Panel letter
        letter = chr(ord('a') + i)
        ax.text(0.01, 0.95, f"({letter})",
                transform=ax.transAxes,
                fontsize=8, fontweight="bold",
                va="top", ha="left")

    # ── Plot each heatmap + set title ───────────────────────────────────────────
    for ax, hmap, (_, title) in zip(heatmap_axes, all_hmaps, files_and_titles):
        ax.imshow(
            hmap,
            aspect="auto",
            vmin=-1, vmax=1,
            cmap="viridis",
            extent=[0, t_max, L, 1]
        )
        # Move the row label into the title
        ax.set_title(title, fontsize=10, pad=4)

    # ── Shared vertical colorbar ───────────────────────────────────────────────
    norm = plt.Normalize(vmin=-1, vmax=1)
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
        cax=cbar_ax,
        orientation="vertical"
    )
    cb.ax.set_title(r"$\langle Z\rangle$", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    # ── Right column: time series for sites 1, 5, 10, 15 ─────────────────────
    sites = [1, 5, 10, 15]
    side_axes = [fig.add_subplot(outer_gs[i, 1]) for i in range(4)]

    colors    = ["lightcoral", "red", "darkred", "black"]
    linestyles = ["-", "-", "-", "--"]  # dashed for MPO

    for idx, (ax, site) in enumerate(zip(side_axes, sites)):
        for (hmap, (_, title), col, ls) in zip(all_hmaps, files_and_titles, colors, linestyles):
            hmap = hmap.T  # (T, L)
            times = np.linspace(0, t_max, hmap.shape[0])
            ts = hmap[:, site - 1]
            label = "MPO" if "MPO" in title else title
            ax.plot(times, ts, color=col, linestyle=ls, label=label, linewidth=1.5)

        ax.set_xlim(0, t_max)
        ax.set_ylim(-0.1, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels([0, 0.5, 1], fontsize=10)

        if idx == 3:
            ax.set_xticks([0, 2, 4, 6, 8, 10])
            ax.set_xticklabels([0, 2, 4, 6, 8, 10], fontsize=10)
            ax.set_xlabel(r"Time $(Jt)$", fontsize=12)
        else:
            ax.set_xticks([])

        # Move ylabel into the subplot title
        ax.set_title(fr"$\langle Z^{{[{site}]}}\rangle$", fontsize=10, pad=4)

        # Panel letters (e–h)
        letter = chr(ord('e') + idx)
        ax.text(0.99, 0.95, f"({letter})",
                transform=ax.transAxes,
                fontsize=8, fontweight="bold",
                va="top", ha="right")

    # ── Combined legend on the right side, outside of the plots ─────────────────
    # Remove individual legends
    for ax in side_axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    legend_lines = []
    legend_labels = []
    for col, lbl, ls in zip(colors, [r"$\chi=2$", r"$\chi=4$", r"$\chi=8$", "MPO"], linestyles):
        legend_lines.append(plt.Line2D([], [], color=col, linestyle=ls, linewidth=1.5))
        legend_labels.append(lbl)

    # Place the combined legend to the right of all subplots
    fig.legend(
        legend_lines, legend_labels,
        loc="center right",
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        fontsize=8,
        frameon=False
    )


    # ── Final layout adjustments ───────────────────────────────────────────────
    fig.subplots_adjust(left=0.075, right=0.9, top=0.95, bottom=0.125)
    plt.savefig("Benchmark_MPOComparison.pdf", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    plot_MPDO_data()
