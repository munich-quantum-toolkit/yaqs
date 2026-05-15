---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
  execution_timeout: 300
---

# 1D Fermi-Hubbard MPO

This example shows how to build a 1D Fermi-Hubbard Hamiltonian as an MPO using {class}`~mqt.yaqs.core.data_structures.networks.MPO.fermi_hubbard_1d`.

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

## Fermionic MPO

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.networks import MPO

num_sites = 4
t = 1.0
u = 0.5

h_mpo = MPO.fermi_hubbard_1d(num_sites, t=t, u=u)
print(f"sites={h_mpo.length}, local dim={h_mpo.physical_dimension}, matrix shape={h_mpo.to_matrix().shape}")
```

The single-site basis is $|0\rangle, |\!\downarrow\rangle, |\!\uparrow\rangle, |\!\uparrow\downarrow\rangle$ (NumPy `kron` ordering for $|\!\uparrow\rangle \otimes |\!\downarrow\rangle$).

## Jordan-Wigner MPO

For the same model on $L$ physical sites, pass `length=2 * L` spin orbitals:

```{code-cell} ipython3
num_orbitals = 2 * num_sites

h_jw = MPO.fermi_hubbard_1d(num_orbitals, t=t, u=u, jordan_wigner=True)
print(f"orbitals={h_jw.length}, local dim={h_jw.physical_dimension}, matrix shape={h_jw.to_matrix().shape}")
```

## Relation to the Trotter circuit helper

{func}`~mqt.yaqs.core.libraries.circuit_library.create_1d_fermi_hubbard_circuit` builds a **digital** Trotter circuit on separate ↑ and ↓ registers and can include a chemical potential $\mu$.
The MPO factories above target the **analog** Hamiltonian without $\mu$ and use either fermionic operators or an interleaved JW layout.

For digital simulation of the circuit model, use the circuit API; for tensor-network evolution of the Hubbard Hamiltonian, use `MPO.fermi_hubbard_1d`.
