import os
import pickle
import re
import math
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx

# --- your edge‐generator unchanged ---
def generate_graph_edges(graph_type: str, n: int) -> List[Tuple[int, int]]:
    if graph_type == "line":
        return [(i, i+1) for i in range(n-1)]
    elif graph_type == "2d_grid":
        rows = cols = int(n**0.5)
        assert rows*cols == n, "n must be a perfect square"
        def idx(r,c): return r*cols + c
        edges = []
        for r in range(rows):
            for c in range(cols):
                if c+1 < cols:     edges.append((idx(r,c),   idx(r,c+1)))
                if r+1 < rows:     edges.append((idx(r,c),   idx(r+1,c)))
        return edges
    elif graph_type == "star":
        return [(0,i) for i in range(1,n)]
    elif graph_type == "ring":
        return [(i,(i+1)%n) for i in range(n)]
    elif graph_type.startswith("erdos_renyi_d"):
        m = re.match(r"erdos_renyi_d(\d+)", graph_type)
        d = int(m.group(1))
        if n*d % 2 != 0:
            raise ValueError("n*d must be even")
        G = nx.random_regular_graph(d,n)
        return list(G.edges())
    elif graph_type == "binary_tree":
        edges = []
        for i in range(n):
            for ch in (2*i+1, 2*i+2):
                if ch < n: edges.append((i,ch))
        return edges
    else:
        raise ValueError(f"Unknown: {graph_type}")


# --- NP‐style rcParams ---
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman"],
    "font.size":        8,
    "axes.linewidth":   0.8,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon":   False,
    "figure.dpi":       300,
})

filenames   = ["line_graph_states.pickle",
               "star_graph_states.pickle",
               "ring_graph_states.pickle"]
graph_types = ["line", "star", "ring"]

fig, axes = plt.subplots(
    2, 3,
    figsize=(7.0, 3.6),
    gridspec_kw={"hspace":0.4}
)

letters = ["(a)","(b)"]

for col, (fname, gtype) in enumerate(zip(filenames, graph_types)):
    # load + sort
    with open(fname, "rb") as f:
        all_results = pickle.load(f)
    all_results = sorted(all_results, key=lambda e: e["n"])
    ns = [e["n"] for e in all_results]
    n_max = 16 # ns[-1]

    # ========== TOP ROW: network layout ==========
    axN = axes[0, col]
    G = nx.Graph()
    G.add_edges_from(generate_graph_edges(gtype, n_max))

    # choose or build a layout
    if gtype == "ring":
        pos = nx.circular_layout(G)
    elif gtype == "line":
        # horizontal line
        pos = {i:(i, 0.0) for i in G.nodes()}
    elif gtype == "star":
        # central node at (0,0), leaves evenly on unit circle
        leaves = [n for n in G.nodes() if n != 0]
        L = len(leaves)
        pos = {0:(0,0)}
        for idx, leaf in enumerate(leaves):
            θ = 2*math.pi * idx / L
            pos[leaf] = (math.cos(θ), math.sin(θ))
    else:
        pos = nx.spring_layout(G, seed=42)

    # draw edges then nodes
    nx.draw_networkx_edges(G, pos,
                           ax=axN,
                           edge_color="black",
                           width=0.6)
    nx.draw_networkx_nodes(G, pos,
                           ax=axN,
                           node_size=30,
                           node_color="black")

    # fix aspect & limits so nothing collapses
    if gtype == "line":
        axN.set_aspect("auto")
        axN.set_xlim(-1, n_max)
        axN.set_ylim(-1, 1)
    else:
        axN.set_aspect("equal")
        axN.margins(0.1)

    axN.axis("off")
    axN.set_title(f"{gtype.title()}, $n={n_max}$", pad=4)
    if col == 0:
        axN.text(-0.2, 1.05, letters[col],
                transform=axN.transAxes,
                fontsize=10, fontweight="bold")

    # ========== BOTTOM ROW: bonds vs n ==========
    axB = axes[1, col]
    total_tebd = []
    total_tdvp = []
    for entry in all_results:
        res = entry["results"]
        if "TEBD" in res and res["TEBD"]:
            b, _, _ = res["TEBD"][-1]
            total_tebd.append(sum(b))
        else:
            total_tebd.append(float("nan"))
        if "TDVP" in res and res["TDVP"]:
            b, _ = res["TDVP"][-1]
            total_tdvp.append(sum(b))
        else:
            total_tdvp.append(float("nan"))

    axB.plot(ns, total_tebd, marker='', linestyle='--', zorder=3, markeredgecolor='black', label="TEBD", linewidth=1.5)
    axB.plot(ns, total_tdvp, marker='', linestyle='-', zorder=2, markeredgecolor='black', label="TDVP", linewidth=1.5)
    axB.tick_params(direction="in", which="both")
    if col == 0:
        axB.set_ylabel("Total bond dimension")
    axB.set_xlabel("Qubits, $N$")
    if col == 0:
        axB.legend(fontsize=7)
    if col == 0:
        axB.text(-0.2, 1.05, letters[col+1],
                transform=axB.transAxes,
                fontsize=10, fontweight="bold")
fig.savefig("results_graph_states.pdf", format="pdf", dpi=600)
plt.tight_layout()
plt.show()
