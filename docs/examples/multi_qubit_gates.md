---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
  execution_timeout: 300
---

```{code-cell} ipython3
:tags: [remove-cell]
%config InlineBackend.figure_formats = ['svg']
```

# Multi-Qubit Gates

Gates acting on three or more qubits translate and apply natively in circuit
simulation. `ccx` (Toffoli), `ccz`, and `cswap` (Fredkin) are hardcoded
{class}`~mqt.yaqs.core.libraries.gate_library.GateLibrary` classes; any other
multi-qubit unitary (for example a Qiskit `UnitaryGate` or a multi-controlled
gate with a matrix representation) is translated through the matrix fallback.

Routing follows the two-qubit dispatch {cite:p}`sander2025_CircuitTDVP`: in the
TDVP gate modes, gates with a product-form generator (`ccx`, `ccz`) are applied
through a generator MPO and a local two-site TDVP window; all other cases —
including `cswap`, matrix-backed gates, and `gate_mode="swaps"` — apply the gate
as an extended MPO followed by a compression sweep.

## Exactness check

A Toffoli circuit reproduces the Qiskit statevector at the exact preset:

```{code-cell} ipython3
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from mqt.yaqs import DigitalSimParams, Observable, Simulator, State
from mqt.yaqs.core.libraries.gate_library import Z

qc = QuantumCircuit(3)
qc.h(0)
qc.h(1)
qc.ry(0.9, 2)
qc.ccx(0, 1, 2)

sim_params = DigitalSimParams(observables=[Observable(Z(), 0)], preset="exact", get_state=True)
result = Simulator(parallel=False, show_progress=False).run(State(3, initial="zeros"), qc, sim_params)
vec = np.asarray(result.output_state.mps.to_vec())
ref = np.asarray(Statevector(qc).data)
print(f"fidelity vs Qiskit: {abs(np.vdot(ref, vec)) ** 2:.15f}")
```

## Native gate vs. two-qubit decomposition under a bond budget

A long-range multi-controlled gate is one gate for the native path (here a
three-control `mcx`, translated through the matrix fallback), but a sequence of
long-range `cx` gates after transpilation to a two-qubit basis. Under a bond
budget every gate application truncates, and the decomposition's *intermediate*
states are far more entangled than the circuit's actual output: the exact final
state below has bond dimension 4, while the decomposition passes through states
that require much larger bonds. The comparison runs the same circuit both ways
at increasing `max_bond_dim` against the exact statevector.

```{code-cell} ipython3
from qiskit import transpile

num_qubits = 10


def build_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    for q in range(num_qubits):
        if q % 2 == 0:
            qc.h(q)
        else:
            qc.ry(0.5 + 0.08 * q, q)
    qc.mcx([0, 3, 6], 9)
    for q in range(num_qubits):
        qc.rz(0.25 + 0.04 * q, q)
    qc.mcx([1, 4, 7], 8)
    return qc


def run_infidelity(qc: QuantumCircuit, reference: np.ndarray, max_bond_dim: int) -> float:
    sim_params = DigitalSimParams(
        observables=[Observable(Z(), 0)],
        gate_mode="mpo",
        max_bond_dim=max_bond_dim,
        svd_threshold=1e-12,
        get_state=True,
    )
    result = Simulator(parallel=False, show_progress=False).run(
        State(num_qubits, initial="zeros"), qc, sim_params
    )
    vec = np.asarray(result.output_state.mps.to_vec())
    return 1.0 - abs(np.vdot(reference, vec)) ** 2


native = build_circuit()
decomposed = transpile(native, basis_gates=["cx", "u"], optimization_level=0)
reference = np.asarray(Statevector(native).data)

num_entangling_native = sum(1 for inst in native.data if inst.operation.num_qubits > 1)
num_entangling_decomposed = sum(1 for inst in decomposed.data if inst.operation.num_qubits > 1)
print(f"entangling gates: {num_entangling_native} native vs. {num_entangling_decomposed} decomposed")

bond_dims = [1, 2, 3, 4, 6, 8, 12, 16]
infidelity_native = [run_infidelity(native, reference, chi) for chi in bond_dims]
infidelity_decomposed = [run_infidelity(decomposed, reference, chi) for chi in bond_dims]
```

```{code-cell} ipython3
import matplotlib.pyplot as plt

floor = 1e-16
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(bond_dims, np.maximum(infidelity_native, floor), "o-", label="native multi-qubit gates")
ax.semilogy(bond_dims, np.maximum(infidelity_decomposed, floor), "s--", label="cx + u decomposition")
ax.set_xlabel(r"bond budget $\chi$ (max_bond_dim)")
ax.set_ylabel("infidelity vs. exact statevector")
ax.set_title("Long-range multi-controlled gates under a bond budget")
ax.legend()
fig.tight_layout()
```

The native path reaches machine precision as soon as the budget covers the exact
output state (here $\chi = 4$), because each gate is applied as a single MPO
contraction and only the final, weakly entangled state is truncated. The
decomposition still carries a finite error there — its intermediate states
exceed the budget — and needs a larger budget to recover.

## Notes

```{note}
- `gate_mode="swaps"` applies gates on three or more qubits through the gate-MPO
  path; there is no SWAP-network decomposition for them.
- `cswap` carries no generator (the SWAP part of the gate has no single-product
  form), so it always uses the gate-MPO path, like `swap`.
- The generator-window path applies a windowed two-site TDVP update, whose
  single-substep accuracy is state dependent; increase
  `DigitalSimParams(tdvp_sweeps=...)` to reduce the time-discretization error.
- Equivalence checking of circuits with gates on more than two qubits requires
  `representation="matrix"`; the MPO backend rejects them.
```
