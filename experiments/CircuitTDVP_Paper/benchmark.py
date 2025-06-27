from __future__ import annotations

import numpy as np
import pickle

from supp_benchmarks.benchmark_utils import benchmark
from mqt.yaqs.core.libraries.circuit_library import create_heisenberg_circuit, create_2d_ising_circuit

from qiskit import QuantumCircuit, transpile
import os

def run_benchmark():
    max_bond = 4096
    min_bond = 2
    out_dir = "results/supp_benchmarks"
    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir("supp_benchmarks/circuits"):
        if not fname.endswith(".qasm"):
            continue

        path = os.path.join("supp_benchmarks/circuits", fname)
        # load the QASM file
        circ = QuantumCircuit.from_qasm_file(path)
        qc_clean = QuantumCircuit(*circ.qregs)

        for instr, qargs, cargs in circ.data:
            # skip any instr that uses classical bits, e.g. Measure, Barrier on clbits, etc.
            if cargs:
                continue
            # optionally also skip named barriers on qubits:
            if instr.name == "barrier":
                continue
            # now we can append safely—the qargs are the same Qubit instances
            qc_clean.append(instr, qargs, [])
            qc_clean.barrier()
        # for instr, qargs, _ in qc_clean.data:
        #     if len(qargs) == 2:
        #         q0 = qargs[0].index
        #         q1 = qargs[1].index
        #         print(f"{instr.name} between qubit {q0} and {q1}, distance = {abs(q0 - q1)}")
        from qiskit.transpiler import Layout
        layout = Layout({qc_clean.qubits[i]: i for i in range(circ.num_qubits)})

        transpiled = transpile(
            qc_clean,
            basis_gates=['rx', 'rz', 'rzz'],
            optimization_level=0,  # 🔒 No decomposition or reordering
            layout_method='trivial',  # 🔒 Use the original qubit layout
            routing_method='none',  # 🔒 No routing = preserve gate connectivity
            initial_layout=layout,
        )

        print(f"Running benchmark on {fname}…")
        results = benchmark(
            transpiled,
            min_bond_dim=min_bond,
            bond_dim_limit=max_bond,
            break_on_exceed=True
        )

        base, _ = os.path.splitext(fname)
        out_file = os.path.join(out_dir, f"{base}.pickle")
        with open(out_file, "wb") as f:
            pickle.dump({"results": results}, f)

if __name__ == "__main__":
    run_benchmark()


# def run_benchmark_2d_cluster_states():
#     max_bond = 512
#     min_bond = 16
#     # Example grid sizes (you can extend or modify this list)
#     grid_sizes = []
#     for i in range(2, 9):
#         for j in range(2, 9):
#             grid_sizes.append((i, j))
#     # grid_sizes = [(2, 3), (3, 3), (3, 5), (4, 6), (5, 5)]  # row × col
#     out_dir = "results/supp_benchmarks"
#     os.makedirs(out_dir, exist_ok=True)

#     for rows, cols in grid_sizes:
#         n = rows * cols
#         fname = f"cluster2d_{rows}x{cols}"

#         circ = QuantumCircuit(n, name=fname)

#         # Step 1: Hadamard on all qubits
#         for i in range(n):
#             circ.h(i)

#         # Map (row, col) to flat index
#         def idx(r, c):
#             return r * cols + c

#         # Step 2: CZ with right and bottom neighbors
#         for r in range(rows):
#             for c in range(cols):
#                 q = idx(r, c)
#                 if c < cols - 1:  # right neighbor
#                     circ.cz(q, idx(r, c + 1))
#                 if r < rows - 1:  # bottom neighbor
#                     circ.cz(q, idx(r + 1, c))

#         # Optional cleaning loop
#         qc_clean = QuantumCircuit(*circ.qregs)
#         for instr, qargs, cargs in circ.data:
#             if cargs or instr.name == "barrier":
#                 continue
#             qc_clean.append(instr, qargs, [])
#             qc_clean.barrier()

#         transpiled = transpile(
#             qc_clean,
#             basis_gates=['rx', 'ry', 'rz', 'h', 'cx'],
#             optimization_level=0
#         )

#         print(f"Running benchmark on {fname}…")
#         results = benchmark(
#             transpiled,
#             min_bond_dim=min_bond,
#             bond_dim_limit=max_bond,
#             break_on_exceed=True
#         )

#         out_file = os.path.join(out_dir, f"{fname}.pickle")
#         with open(out_file, "wb") as f:
#             pickle.dump({"results": results}, f)

# from qiskit import QuantumCircuit
# from typing import List, Tuple

# def create_graph_state(n: int, edges: List[Tuple[int, int]], apply_hadamards: bool = True, name: str = "graph_state"):
#     """
#     Create a graph state on n qubits based on the given edge list.

#     Parameters:
#     - n: Number of qubits
#     - edges: List of (i, j) edges representing CZ connections between qubits i and j
#     - apply_hadamards: Whether to apply Hadamard gates on all qubits at the beginning
#     - name: Name of the quantum circuit

#     Returns:
#     - QuantumCircuit object
#     """
#     circ = QuantumCircuit(n, name=name)

#     if apply_hadamards:
#         for i in range(n):
#             circ.h(i)

#     for i, j in edges:
#         circ.cz(i, j)

#     return circ

# import os
# import pickle
# from qiskit import QuantumCircuit, transpile
# from typing import List, Tuple

# def create_graph_state(n: int, edges: List[Tuple[int, int]], apply_hadamards: bool = True, name: str = "graph_state"):
#     circ = QuantumCircuit(n, name=name)
#     if apply_hadamards:
#         for i in range(n):
#             circ.h(i)

#     for i, j in edges:
#         circ.cz(i, j)

#     return circ

# from typing import List, Tuple
# import networkx as nx
# import re

# def generate_graph_edges(graph_type: str, n: int) -> List[Tuple[int, int]]:
#     if graph_type == "line":
#         return [(i, i+1) for i in range(n-1)]

#     elif graph_type == "2d_grid":
#         rows = cols = int(n**0.5)
#         assert rows * cols == n, "n must be a perfect square for 2D grid"
#         def idx(r, c): return r * cols + c
#         edges = []
#         for r in range(rows):
#             for c in range(cols):
#                 if c < cols - 1:
#                     edges.append((idx(r, c), idx(r, c + 1)))
#                 if r < rows - 1:
#                     edges.append((idx(r, c), idx(r + 1, c)))
#         return edges

#     elif graph_type == "star":
#         return [(0, i) for i in range(1, n)]

#     elif graph_type == "ring":
#         return [(i, (i + 1) % n) for i in range(n)]

#     elif graph_type.startswith("erdos_renyi_d"):
#         # Extract degree from the string, e.g. "erdos_renyi_d3"
#         match = re.match(r"erdos_renyi_d(\d+)", graph_type)
#         if not match:
#             raise ValueError(f"Invalid ER graph type format: {graph_type}")
#         d = int(match.group(1))
#         if n * d % 2 != 0:
#             raise ValueError(f"Cannot construct d-regular graph with odd n*d (n={n}, d={d})")
#         G = nx.random_regular_graph(d, n)
#         return list(G.edges())
#     elif graph_type == "binary_tree":
#         edges = []
#         for i in range(n):
#             left = 2 * i + 1
#             right = 2 * i + 2
#             if left < n:
#                 edges.append((i, left))
#             if right < n:
#                 edges.append((i, right))
#         return edges

#     else:
#         raise ValueError(f"Unknown graph type: {graph_type}")


# def run_benchmark_graph_states():
#     graph_types = ["ring"]
#     qubit_sizes = range(3, 64)
#     max_bond = 512
#     min_bond = 2

#     out_dir = "results/supp_benchmarks"
#     os.makedirs(out_dir, exist_ok=True)

#     all_results = []   # will hold dicts of {graph_type, n, results}

#     for graph_type in graph_types:
#         for n in qubit_sizes:
#             try:
#                 edges = generate_graph_edges(graph_type, n)
#             except AssertionError as e:
#                 print(f"Skipping {graph_type} with n={n}: {e}")
#                 continue

#             circ = create_graph_state(n, edges, name=f"{graph_type}_{n}")
#             transpiled = transpile(
#                 circ,
#                 basis_gates=['rzz', 'rx', 'rz'],
#                 optimization_level=0
#             )

#             print(f"Running benchmark on graph: {graph_type}, qubits: {n}")
#             results = benchmark(
#                 transpiled,
#                 min_bond_dim=min_bond,
#                 bond_dim_limit=max_bond,
#                 break_on_exceed=True
#             )

#             all_results.append({
#                 "graph_type": graph_type,
#                 "n": n,
#                 "results": results
#             })

#     # save everything in one pickle
#     out_file = os.path.join(out_dir, "star_graph_states.pickle")
#     with open(out_file, "wb") as f:
#         pickle.dump(all_results, f)


# if __name__ == "__main__":
#     run_benchmark_graph_states()
