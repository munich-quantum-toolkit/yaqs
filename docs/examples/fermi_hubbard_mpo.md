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

# 1D Fermi-Hubbard Hamiltonian

This example shows how to build a 1D Fermi-Hubbard Hamiltonian for analog simulation using
{meth}`~mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian.fermi_hubbard_1d` (backed by an MPO internally).
You can also call {meth}`~mqt.yaqs.core.data_structures.mpo.MPO.fermi_hubbard_1d` directly and wrap the result
with {meth}`~mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian.from_mpo` for custom workflows.

YAQS supports two representations:

- **Fermionic sites** (default): one site with local dimension 4 per physical lattice site.
  Ladder operators act on a composite ↑/↓ basis per site; this is not a Jordan–Wigner qubit chain
  across sites, but matches the standard tensor-product embedding of site Fock spaces.
- **Jordan-Wigner Pauli chain** (`jordan_wigner=True`): qubits in the order 1↑, 1↓, 2↑, 2↓, … with
  local dimension 2 and full JW signs between spin orbitals. Use this mode for Pauli-string /
  qubit simulators.

The Hamiltonian (open boundaries, no chemical potential) is

$$
H = -t \sum_{i,\sigma} \left(c^\dagger_{i,\sigma} c_{i+1,\sigma} + \mathrm{h.c.}\right)
+ U \sum_i n_{i,\uparrow} n_{i,\downarrow}.
$$

## 1. Fermionic Hamiltonian

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian

num_sites = 4
t = 1.0
u = 0.5

H = Hamiltonian.fermi_hubbard_1d(num_sites, t=t, u=u)
print(f"sites={H.length}, local dim={H.mpo.physical_dimension}, matrix shape={H.mpo.to_matrix().shape}")
```

The single-site basis is $|0\rangle, |\!\downarrow\rangle, |\!\uparrow\rangle, |\!\uparrow\downarrow\rangle$ (NumPy `kron` ordering for $|\!\uparrow\rangle \otimes |\!\downarrow\rangle$).

## 2. Jordan-Wigner Hamiltonian

For the same model on $L$ physical sites, pass `length=2 * L` spin orbitals:

```{code-cell} ipython3
num_orbitals = 2 * num_sites

H_jw = Hamiltonian.fermi_hubbard_1d(num_orbitals, t=t, u=u, jordan_wigner=True)
print(f"orbitals={H_jw.length}, local dim={H_jw.mpo.physical_dimension}, matrix shape={H_jw.mpo.to_matrix().shape}")
```

## 3. Short analog simulation

Evolve a two-site fermionic chain in the vacuum $|00\rangle$ and track the probability of remaining in that sector. Fermionic sites use local dimension 4, so pass `physical_dimensions=[4, 4]` on {class}`~mqt.yaqs.core.data_structures.state.State`.

```{code-cell} ipython3
from mqt.yaqs import Simulator
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.data_structures.state import State

hubbard_sites = 2
H_small = Hamiltonian.fermi_hubbard_1d(hubbard_sites, t=1.0, u=0.5)
psi0 = State(hubbard_sites, initial="zeros", physical_dimensions=[4, 4])

params = AnalogSimParams(
    observables=[Observable("00"), Observable("11")],
    elapsed_time=0.2,
    dt=0.05,
    preset="fast",
    sample_timesteps=True,
)

sim = Simulator(show_progress=False)
result = sim.run(psi0, H_small, params)
times = params.times
p_vacuum = [float(v) for v in result.expectation_values[0]]
p_down_down = [float(v) for v in result.expectation_values[1]]
print("P(vacuum |00⟩):", p_vacuum)
print("P(|↓⟩⊗|↓⟩) basis string '11':", p_down_down)
```

Hopping spreads population out of the vacuum; `"11"` indexes $|{\downarrow}\rangle \otimes |{\downarrow}\rangle$, not double occupancy (index `3` = $|{\uparrow\downarrow}\rangle$ per site).

```{code-cell} ipython3
---
mystnb:
  image:
    width: 70%
    align: center
---
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(times, p_vacuum, "o-", label=r"$P(|00\rangle)$ vacuum")
ax.plot(times, p_down_down, "s--", label=r"$P(|\!\downarrow\rangle^{\otimes 2})$ ('11')")
ax.set_xlabel("time")
ax.set_ylabel("probability")
ax.set_title("Two-site Fermi–Hubbard from vacuum")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

## 4. Relation to the Trotter circuit helper

{func}`~mqt.yaqs.core.libraries.circuit_library.create_1d_fermi_hubbard_circuit` builds a **digital** Trotter circuit on separate ↑ and ↓ registers and can include a chemical potential $\mu$.
The MPO factories above target the **analog** Hamiltonian without $\mu$ and use either fermionic operators or an interleaved JW layout.

For digital simulation of the circuit model, use the circuit API; for tensor-network evolution of the Hubbard Hamiltonian, use `Hamiltonian.fermi_hubbard_1d` with {meth}`~mqt.yaqs.Simulator.run`.

## Related topics

- {doc}`analog_simulation` — TJM workflow and noise models
- {doc}`circuit_simulation` — digital Trotter circuits via {class}`qiskit.circuit.QuantumCircuit`
- {doc}`simulation_parameters` — presets and truncation for analog runs
