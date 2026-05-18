---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
  execution_timeout: 300
---

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

## Fermionic Hamiltonian

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian

num_sites = 4
t = 1.0
u = 0.5

H = Hamiltonian.fermi_hubbard_1d(num_sites, t=t, u=u)
print(f"sites={H.length}, local dim={H.mpo.physical_dimension}, matrix shape={H.mpo.to_matrix().shape}")
```

The single-site basis is $|0\rangle, |\!\downarrow\rangle, |\!\uparrow\rangle, |\!\uparrow\downarrow\rangle$ (NumPy `kron` ordering for $|\!\uparrow\rangle \otimes |\!\downarrow\rangle$).

## Jordan-Wigner Hamiltonian

For the same model on $L$ physical sites, pass `length=2 * L` spin orbitals:

```{code-cell} ipython3
num_orbitals = 2 * num_sites

H_jw = Hamiltonian.fermi_hubbard_1d(num_orbitals, t=t, u=u, jordan_wigner=True)
print(f"orbitals={H_jw.length}, local dim={H_jw.mpo.physical_dimension}, matrix shape={H_jw.mpo.to_matrix().shape}")
```

## Relation to the Trotter circuit helper

{func}`~mqt.yaqs.core.libraries.circuit_library.create_1d_fermi_hubbard_circuit` builds a **digital** Trotter circuit on separate ↑ and ↓ registers and can include a chemical potential $\mu$.
The MPO factories above target the **analog** Hamiltonian without $\mu$ and use either fermionic operators or an interleaved JW layout.

For digital simulation of the circuit model, use the circuit API; for tensor-network evolution of the Hubbard Hamiltonian, use `Hamiltonian.fermi_hubbard_1d` with {func}`~mqt.yaqs.simulator.run`.
