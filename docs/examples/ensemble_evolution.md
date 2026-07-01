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

# Ensemble Evolution

Use this page when you need **two-time correlators** (`multi_time_observables` on {class}`~mqt.yaqs.core.data_structures.simulation_parameters.AnalogSimParams`) or **ensemble averages** over `list[State]` inputs—for dynamical typicality studies, transport correlators, and finite-temperature observables estimated from random pure states.

This page demonstrates workflows for computing two-time correlations in a deterministic (noiseless, unitary) ensemble in YAQS.
The focus is on compact, executable examples:

- Single-state auto/two-time correlations.
- Ensemble-averaged correlations (typicality view).
- Small periodic spin-current transport example.

## 1. Unitary analog evolution primer

In unitary analog evolution, we have no noise or tensor jumps.
Omit `noise_model` in {meth}`~mqt.yaqs.Simulator.run` (it defaults to `None`).

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

from mqt.yaqs import AnalogSimParams, Hamiltonian, Observable, Simulator, State
from mqt.yaqs.core.libraries.gate_library import BaseGate

sim = Simulator(show_progress=False)
```

```{code-cell} ipython3
L = 6
Jxx = 1.0
delta = 0.7
h_x = 0.4

# Open XXZ + transverse field: H = Jxx ∑_r (S^x_r S^x_{r+1} + S^y_r S^y_{r+1}) + Δ ∑_r S^z_r S^z_{r+1} + h_x ∑_r S^x_r
# (Pauli convention in code: S^α = σ^α/2, matching two_body prefactors 0.25 * Jxx / Δ.)
H_open = Hamiltonian.pauli(
    length=L,
    two_body=[(0.25 * Jxx, "X", "X"), (0.25 * Jxx, "Y", "Y"), (0.25 * delta, "Z", "Z")],
    one_body=[(0.5 * h_x, "X")],
    bc="open",
)

mid = L // 2
psi0 = State(L, initial="haar-random", pad=2)

primer_params = AnalogSimParams(
    observables=[Observable("z", mid)],
    elapsed_time=5.0,
    dt=0.15,
    max_bond_dim=64,
    svd_threshold=1e-10,
    sample_timesteps=True,
)

result_primer = sim.run(psi0, H_open, primer_params)
times_primer = primer_params.times
zexp_primer = result_primer.expectation_values[0]
```

```{code-cell} ipython3
fig, ax = plt.subplots(1, 1, figsize=(5.4, 3.2))
ax.plot(times_primer, zexp_primer, marker="o", ms=3)
ax.set_xlabel("t")
ax.set_ylabel(r"$\langle S^z_m(t)\rangle$")
ax.set_title("Single-state unitary evolution")
ax.grid(alpha=0.3)
plt.show()
```

## 2. Two-time Correlations

For an initial state $|\psi(0)\rangle$ and unitary $U(t)$:

- Autocorrelation (for one observable $O$):
  \[
  C\_{OO}(t) = \langle \psi(0)| U^\dagger(t)\, O\, U(t)\, O |\psi(0)\rangle.
  \]
- Generic two-time correlation (probe $A$ and kick $B$):
  \[
  C\_{AB}(t) = \langle \psi(0)| U^\dagger(t)\, A\, U(t)\, B |\psi(0)\rangle.
  \]

These quantities probe dynamical memory and relaxation.
They are standard observables in **dynamical quantum typicality (DQT)** and related finite-temperature dynamics studies, where one compares single-trajectory and ensemble-averaged behavior.

The unitary-ensemble backend computes `multi_time_observables` pairs for `list[State]` inputs (each with `representation="mps"`, the default).
Autocorrelation is the special case where both the observables are the same `(O, O)`. For a single-state demonstration, we pass a list with one element.

```{code-cell} ipython3
sz_mid = Observable("z", mid)
sx_mid = Observable("x", mid)

single_state_params = AnalogSimParams(
    observables=[],
    elapsed_time=5.0,
    dt=0.15,
    max_bond_dim=64,
    svd_threshold=1e-10,
    sample_timesteps=True,
    multi_time_observables=[(sz_mid, sz_mid), (sz_mid, sx_mid)],  # row 0: C_zz(t), row 1: C_zx(t)
)

sim = Simulator(show_progress=False)
result_single = sim.run(
    [State(L, initial="haar-random", pad=2)], H_open, single_state_params
)

t_single = result_single.multi_time_times
czz_single = result_single.multi_time_results[0]
czx_single = result_single.multi_time_results[1]
```

```{code-cell} ipython3
fig, ax = plt.subplots(1, 1, figsize=(5.8, 3.4))
ax.plot(t_single, np.real(czz_single), "o-", label=r"$C_{zz}(t)$")
ax.plot(t_single, np.real(czx_single), "s--", label=r"$C_{zx}(t)$")
ax.set_xlabel("t")
ax.set_ylabel(r"$C_{ab}(t)$")
ax.set_title("Single-state two-time correlations")
ax.legend()
ax.grid(alpha=0.3)
plt.show()
```

## 3. Typicality view: from one state to an ensemble

In dynamical typicality studies, one often averages correlations over an ensemble of initial states.
Under certain thermalisation guarantees, one can show that the typical relaxation behavior of _any_ state can be represented by an ensemble average of the expectation over randomly initialised states.
For sufficiently rich ensembles, this can approximate high-temperature traces and reveal robust transport trends.

YAQS supports this directly by passing `list[State]` into `Simulator.run`.
Each member evolves independently, which, when parallelized by the unitary backend, offers computational advantage to calculate these variables.

```{code-cell} ipython3
num_states = 8
ensemble_states = [State(L, initial="haar-random", pad=2) for _ in range(num_states)]

ensemble_params = AnalogSimParams(
    observables=[],
    elapsed_time=5.0,
    dt=0.15,
    max_bond_dim=64,
    svd_threshold=1e-10,
    sample_timesteps=True,
    multi_time_observables=[
        (Observable("z", mid), Observable("z", mid)),  # C_zz(t) autocorrelation
        (Observable("z", mid), Observable("x", mid)),  # C_zx(t)
    ],
)

result_ens = sim.run(ensemble_states, H_open, ensemble_params)
t_ens = result_ens.multi_time_times
czz_ens = result_ens.multi_time_results[0]
czx_ens = result_ens.multi_time_results[1]
```

```{code-cell} ipython3
fig, ax = plt.subplots(1, 1, figsize=(5.8, 3.4))
ax.plot(t_ens, np.real(czz_ens), "o-", label=r"ensemble $C_{zz}(t)$")
ax.plot(t_ens, np.real(czx_ens), "s--", label=r"ensemble $C_{zx}(t)$")
ax.set_xlabel("t")
ax.set_ylabel(r"$\overline{C}_{ab}(t)$")
ax.set_title(f"Typicality-style ensemble average of $C_{{zz}}(t)$ and $C_{{zx}}(t)$ (N={num_states})")
ax.legend()
ax.grid(alpha=0.3)
plt.show()
```

In this illustrative run, the ensemble-averaged $C_{zz}(t)$ appears to decay toward zero while $C_{zx}(t)$ stays comparatively close to zero over the sampled window; other runs may show different behavior.

## 4. Spin transport example: periodic spin-current autocorrelation

For periodic XXZ chains, define local bond current

<!-- prettier-ignore-start -->
\[
j_r = J_{xx} \bigl(S_r^x S_{r+1}^y - S_r^y S_{r+1}^x\bigr)
\]
<!-- prettier-ignore-end -->

and total current $J = \sum_r j_r$.
The normalized autocorrelator

<!-- prettier-ignore-start -->
\[
C_{JJ}(t) = \frac{1}{L}\,\langle J(t)\,J(0)\rangle
\]
<!-- prettier-ignore-end -->

can be assembled from all bond-pair two-time correlators.
Such current autocorrelations are central to linear-response spin transport; dynamical typicality makes it practical to estimate high-temperature ensemble quantities from a few random pure-state trajectories ([Steiningeweg _et al._, Phys. Rev. Lett. **112**, 120601 (2014)](https://doi.org/10.1103/PhysRevLett.112.120601)).
For finite-temperature Drude weights, diffusion, and integrable XXZ phenomenology—including the role of conservation laws—see the review ([Bertini _et al._, Rev. Mod. Phys. **93**, 025003 (2021)](https://doi.org/10.1103/RevModPhys.93.025003)).

```{code-cell} ipython3
def spin_current_bond_matrix(j_coupling: float) -> np.ndarray:
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    return 0.25 * j_coupling * (np.kron(x, y) - np.kron(y, x))


def periodic_bonds(length: int) -> list[tuple[int, int]]:
    return [(i, (i + 1) % length) for i in range(length)]


def current_observables(length: int, j_coupling: float) -> list[Observable]:
    j_mat = spin_current_bond_matrix(j_coupling)
    gate = BaseGate(j_mat)
    return [Observable(gate, sites=[i, j]) for i, j in periodic_bonds(length)]
```

```{code-cell} ipython3
Ltr = 6
Jxx = 1.0
deltas = [0.1, 0.5, 1.5]
t_final = 5.0
dt = 0.15
n_transport_states = 4

states_transport = [State(Ltr, initial="haar-random", pad=2) for _ in range(n_transport_states)]
bond_obs = current_observables(Ltr, Jxx)
pairs_jj = [(a, b) for a in bond_obs for b in bond_obs]

transport_curves: dict[float, np.ndarray] = {}
t_transport = None
for d in deltas:
    h_periodic = Hamiltonian.pauli(
        length=Ltr,
        two_body=[(0.25 * Jxx, "X", "X"), (0.25 * Jxx, "Y", "Y"), (0.25 * d, "Z", "Z")],
        one_body=[],
        bc="periodic",
    )
    sp = AnalogSimParams(
        observables=[],
        elapsed_time=t_final,
        dt=dt,
        max_bond_dim=64,
        svd_threshold=1e-10,
        sample_timesteps=True,
        multi_time_observables=pairs_jj,
    )
    result_transport = sim.run(states_transport, h_periodic, sp)
    t_transport = result_transport.multi_time_times
    c_jj = np.real(np.sum(result_transport.multi_time_results, axis=0) / Ltr)
    transport_curves[d] = c_jj
```

```{code-cell} ipython3
fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.5))
for d in deltas:
    ax.plot(t_transport, transport_curves[d], marker="o", ms=3, label=rf"$\Delta={d}$")
ax.set_xlabel("t")
ax.set_ylabel(r"$C_{JJ}(t)$")
ax.set_title("Periodic XXZ spin-current autocorrelation (small illustrative setup)")
ax.legend()
ax.grid(alpha=0.3)
plt.show()
```

This finite-size, short-time run already shows different relaxation trends for different anisotropies.
In the thermodynamic limit and Kubo picture, the long-time behavior of $C_{JJ}(t)$ is tied to the spin Drude weight and to ballistic versus diffusive transport in the XXZ chain; [Bertini _et al._, Rev. Mod. Phys. **93**, 025003 (2021)](https://doi.org/10.1103/RevModPhys.93.025003) summarizes the established finite-temperature picture (including subtleties at $\Delta=1$ and in finite systems).
The illustrative curves here use small $L$ and a handful of Haar-random states; larger-scale or higher-accuracy studies follow typicality, as in [Steiningeweg _et al._, Phys. Rev. Lett. **112**, 120601 (2014)](https://doi.org/10.1103/PhysRevLett.112.120601).

:::{tip} Practical notes: scaling runs and MPS entanglement

- Scale gradually: `L`, ensemble size, `dt`, `elapsed_time`, and `max_bond_dim`.
- Enable ensemble parallelization (`Simulator(parallel=True)`) when you have many initial states.
- **MPS entanglement:** under unitary evolution, entanglement entropy and required bond dimension typically **grow** with time (until truncation or saturation). For longer times or larger $L$, increase `max_bond_dim`, tighten `svd_threshold` only with care, or shorten the window so the MPS remains an accurate ansatz for your observable.

:::

## Related topics

- {doc}`analog_simulation` — noisy and unitary TJM evolution
- {doc}`state_initialization` — Haar-random and ensemble `list[State]` inputs
- {doc}`simulator_initialization` — `Simulator(parallel=True)` for ensemble runs
