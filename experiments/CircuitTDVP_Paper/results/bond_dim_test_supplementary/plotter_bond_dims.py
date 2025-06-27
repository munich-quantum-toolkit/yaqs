import matplotlib.pyplot as plt
import numpy as np
import operator
import pickle
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_bond_heatmaps(
    methods=("TEBD", "TDVP"),
    file_specs=(
        ("heisenberg_bonds.pickle", "Heisenberg"),
        ("periodic_heisenberg_bonds.pickle", "Periodic Heisenberg"),
        ("2d_ising_bond.pickle", "2D Ising")
    ),
    cmap="viridis",
    log_scale=False,
    figsize=(7.2, 6.5)
) -> None:
    # styling
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
    })
    cmap_obj = plt.get_cmap(cmap)
    cmap_obj.set_bad('white')

    def extract_obs(results, method):
        data = sorted(results.get(method, []), key=lambda x: x[0])
        if not data:
            return np.array([]), np.array([])
        t, _, ev = zip(*data)
        return np.array(t), np.array(ev)

    def extract_matrix(results, method):
        data = sorted(results.get(method, []), key=lambda x: x[0])
        if not data:
            return np.empty((0, 0))
        return np.vstack([b for _, b, _ in data])

    # first pass: gather for global norm
    combined_all = []
    for fname, _ in file_specs:
        with open(fname, 'rb') as f:
            res = pickle.load(f)['results']
        m1, m2 = methods
        A = extract_matrix(res, m1)
        B = extract_matrix(res, m2)
        if A.size and B.size:
            combined_all.append(np.vstack([B.T, A.T]))

    if log_scale and combined_all:
        all_vals = np.concatenate([c[np.isfinite(c)].ravel() for c in combined_all])
        vmin, vmax = all_vals.min(), all_vals.max()
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    # create the 3×N grid with tighter spacing
    ncols = len(file_specs)
    fig, axes = plt.subplots(
        3, ncols, figsize=figsize,
        # sharex='col',
        gridspec_kw={'hspace': 0.3, 'wspace': 0.2}
    )

    for col, (fname, title) in enumerate(file_specs):
        with open(fname, 'rb') as f:
            res = pickle.load(f)['results']
        m1, m2 = methods
        ts1, obs1 = extract_obs(res, m1)
        ts2, obs2 = extract_obs(res, m2)
        mat1 = extract_matrix(res, m1)
        mat2 = extract_matrix(res, m2)

        nan_rows = np.where(np.isnan(mat2).all(axis=1))[0]
        cutoff = nan_rows[0]+1 if nan_rows.size else mat2.shape[0]
        nan_rows = np.where(np.isnan(mat1).all(axis=1))[0]
        cutoff_tebd = nan_rows[0] if nan_rows.size else mat1.shape[0]

        # (a) expectation
        ax0 = axes[0, col]
        ax0.plot(ts1, obs1, '^--', label=m1, zorder=3, linewidth=1.5)
        ax0.plot(ts2, obs2, 'o-', label=m2, zorder=2,  markeredgecolor='black', linewidth=1.5)
        ax0.set_title(title, pad=6, fontsize=10)

        if log_scale:
            ax0.set_yscale('log')
        ax0.set_xlim(0, cutoff)
        if col == 0:
            ax0.set_ylabel('$\\langle X_{24} X_{25} \\rangle$')
            ax0.legend(frameon=False, loc='lower center')
        ax0.axvline(cutoff_tebd, color='gray', linestyle='--', lw=1)

        if col == 0:
            ax0.set_xticks([10, 20, 30, 40, 50, 60, 70])
            ax0.set_xticklabels([10, 20, 30, 40, 50, 60, 70])
        if col == 1:
            ax0.set_xticks([5, 10, 15, 20, 25, 30])
            ax0.set_xticklabels([5, 10, 15, 20, 25, 30])
        # ax1.set_xlabel('Trotter steps')
        if col == 2:
            ax0.set_xlim(1, cutoff)

        # (b) heatmap
        ax1 = axes[1, col]
        combined = np.vstack([mat2.T, mat1.T]) if (mat1.size and mat2.size) else np.empty((0,0))
        im = ax1.imshow(combined, origin='lower', aspect='auto',
                        cmap=cmap_obj, interpolation='none', vmin=2, vmax=512)
        ax1.axhline(mat1.shape[1]-0.5, color='white', lw=1)
        ax1.set_xlim(0, cutoff)
        if col == 0:
            ax1.set_ylabel('Bond index')
            ax1.set_xticks([10, 20, 30, 40, 50, 60, 70])
            ax1.set_xticklabels([10, 20, 30, 40, 50, 60, 70])
        if col == 1:
            ax1.set_xticks([5, 10, 15, 20, 25, 30])
            ax1.set_xticklabels([5, 10, 15, 20, 25, 30])
        # ax1.set_xlabel('Trotter steps')
        if col == 2:
            ax1.set_xticks([1, 3, 5, 7])
            ax1.set_xticklabels([2, 4, 6, 8])
        ax1.set_yticks([0, mat1.shape[1]//2, mat1.shape[1]-3, mat1.shape[1]+3, 1.5*mat1.shape[1], 2*mat1.shape[1]-1])
        ax1.set_yticklabels([1, mat1.shape[1]//2, mat1.shape[1], 1, mat1.shape[1]//2, mat1.shape[1]])
        ax1.axvline(cutoff_tebd-0.5, color='gray', linestyle='--', lw=1)

        # --- right after your im = ax1.imshow(…) call ---
        # annotate TEBD at the top‐left
        ax1.text(
            0.015,        # x position, 1% from left
            0.95,        # y position, 95% from bottom (i.e. top)
            m1,          # that's "TEBD"
            transform=ax1.transAxes,
            color='white',
            fontsize=8,
            fontweight='bold',
            va='top',
            ha='left'
        )
        # annotate TDVP just below center
        ax1.text(
            0.015,        # same x
            0.425,        # 45% from bottom, just under the middle
            m2,          # that's "TDVP"
            transform=ax1.transAxes,
            color='white',
            fontsize=8,
            fontweight='bold',
            va='center',
            ha='left'
        )
        # --- then the rest of your code continues ---

        # (c) total bond
        ax2 = axes[2, col]
        tot1 = np.nansum(mat1**3, axis=1)
        tot2 = np.nansum(mat2**3, axis=1)
        mask1 = (ts1 < cutoff) & (tot1 > 0)
        mask2 = (ts2 < cutoff) & (tot2 > 0)
        ax2.plot(ts1[mask1], tot1[mask1], '^--', zorder=3, label="TEBD", linewidth=1.5)
        ax2.plot(ts2[mask2], tot2[mask2], 'o-', zorder=2, markeredgecolor='black', label="TDVP", linewidth=1.5)
        ax2.set_yscale('linear')
        ax2.set_xlim(0, cutoff)
        if col == 0:
            ax2.set_ylabel('Runtime cost $\\sum_j \\chi_j^3$')
        ax2.set_xlabel('Trotter steps')
        ax2.axvline(cutoff_tebd, color='gray', linestyle='--', lw=1)
        if col == 0:
            ax2.set_xticks([10, 20, 30, 40, 50, 60, 70])
            ax2.set_xticklabels([10, 20, 30, 40, 50, 60, 70])
            ax2.legend(frameon=False, loc='upper left')
        if col == 1:
            ax2.set_xticks([5, 10, 15, 20, 25, 30])
            ax2.set_xticklabels([5, 10, 15, 20, 25, 30])
            # ax2.set_yticklabels([])
        if col == 2:
            ax2.set_xlim(1, cutoff)
            # ax2.set_yticklabels([])
        # ax1.set_xlabel('Trotter steps')
        # if col == 2:
        #     ax2.set_xticks([2, 4, 6, 8])
        #     ax2.set_xticklabels([2, 4, 6, 8])
        ax2.set_ylim(1, 4e9)

    # inset colorbar in the last heatmap without shifting anything
    ax_cb_target = axes[1, -1]
    cax = inset_axes(ax_cb_target,
                     width="10%", height="90%",
                     loc='upper right', borderpad=0.1)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.set_title('$\\chi$')

    cb.ax.tick_params(direction='out', length=3)

    # clean up spines & ticks
    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out', length=3, width=1)

    # panel letters
    fig.text(0.02, 0.95, '(a)', fontweight='bold', fontsize=10)
    fig.text(0.02, 0.62, '(b)', fontweight='bold', fontsize=10)
    fig.text(0.02, 0.29, '(c)', fontweight='bold', fontsize=10)
    # squeeze margins just a bit tighter
    # plt.tight_layout(rect=[0,0,1,1], pad=1.0)
    fig.savefig("results.pdf", format="pdf", dpi=600)

    plt.show()


if __name__ == "__main__":
    plot_bond_heatmaps()
