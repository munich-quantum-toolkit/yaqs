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

# Two-Time Correlators in Unitary Analog Simulation

This page demonstrates workflows for computing two-time correlators in deterministic unitary analog evolution in YAQS.
The focus is on compact, executable examples:

- single-state correlators,
- ensemble-averaged correlators (typicality view),
- and a small periodic spin-current transport example.

## 1. Unitary analog evolution primer

In unitary analog evolution, we have no noise or tensor jumps.
Pass `noise_model=None` to `simulator.run`.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import BaseGate, Z, X, Y
```

```{code-cell} ipython3
L = 6
Jxx = 1.0
delta = 0.7
h_x = 0.4

# Open XXZ + transverse field: H = Jxx ∑_r (S^x_r S^x_{r+1} + S^y_r S^y_{r+1}) + Δ ∑_r S^z_r S^z_{r+1} + h_x ∑_r S^x_r
# (Pauli convention in code: S^α = σ^α/2, matching two_body prefactors 0.25 * Jxx / Δ.)
H_open = MPO.hamiltonian(
    length=L,
    two_body=[(0.25 * Jxx, "X", "X"), (0.25 * Jxx, "Y", "Y"), (0.25 * delta, "Z", "Z")],
    one_body=[(0.5 * h_x, "X")],
    bc="open",
)

mid = L // 2
psi0 = MPS(L, state="haar-random", pad=2)

primer_params = AnalogSimParams(
    observables=[Observable(Z(), mid)],
    elapsed_time=5.0,
    dt=0.15,
    max_bond_dim=64,
    threshold=1e-10,
    sample_timesteps=True,
    show_progress=False,
)

simulator.run(psi0, H_open, primer_params, noise_model=None, parallel=False)
times_primer = primer_params.times
zexp_primer = primer_params.observables[0].results
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

## 2. Two-time Correlators

For an initial state $|\psi(0)\rangle$ and unitary $U(t)$:

- Autocorrelator (for one observable $O$):
  \[
  C\_{OO}(t) = \langle \psi(0)| U^\dagger(t)\, O\, U(t)\, O |\psi(0)\rangle.
  \]
- Generic two-time correlator (probe $A$ and kick $B$):
  \[
  C\_{AB}(t) = \langle \psi(0)| U^\dagger(t)\, A\, U(t)\, B |\psi(0)\rangle.
  \]

These quantities probe dynamical memory and relaxation.
They are standard observables in **dynamical quantum typicality (DQT)** and related finite-temperature dynamics studies, where one compares single-trajectory and ensemble-averaged behavior.

The unitary-ensemble backend computes `two_time_correlators` for `list[MPS]` inputs.
For a single-state demonstration, we pass a list with one element.

```{code-cell} ipython3
sz_mid = Observable(Z(), mid)
sx_mid = Observable(X(), mid)

single_state_params = AnalogSimParams(
    observables=[],
    elapsed_time=5.0,
    dt=0.15,
    max_bond_dim=64,
    threshold=1e-10,
    sample_timesteps=True,
    show_progress=False,
    compute_autocorrelator=True,
    autocorrelator_observable=sz_mid,
    two_time_correlators=[(sz_mid, sx_mid)],  # C_zx(t) = <Sz(t) Sx(0)>
)

simulator.run([MPS(L, state="haar-random", pad=2)], H_open, single_state_params, noise_model=None, parallel=False)

t_single = single_state_params.autocorrelator_times
czz_single = single_state_params.autocorrelator_results
czx_single = single_state_params.two_time_correlator_results[0]
```

```{code-cell} ipython3
fig, ax = plt.subplots(1, 1, figsize=(5.8, 3.4))
ax.plot(t_single, np.real(czz_single), "o-", label=r"$C_{zz}(t)$")
ax.plot(t_single, np.real(czx_single), "s--", label=r"$C_{zx}(t)$")
ax.set_xlabel("t")
ax.set_ylabel(r"$C_{ab}(t)$")
ax.set_title("Single-state two-time correlators")
ax.legend()
ax.grid(alpha=0.3)
plt.show()
```

## 3. Typicality view: from one state to an ensemble

In dynamical typicality studies, one often averages correlators over an ensemble of initial states.
Under certain thermalisation guarantees, one can show that the typical relaxation behavior of _any_ state can be represented by an ensemble average of the expectation over randomly initialised states.
For sufficiently rich ensembles, this can approximate high-temperature traces and reveal robust transport trends.

YAQS supports this directly by passing `list[MPS]` into `simulator.run`.
Each member evolves independently, which, when parallelized by the unitary backend, offers computational advantage to calculate these variables.

```{code-cell} ipython3
num_states = 8
ensemble_states = [MPS(L, state="haar-random", pad=2) for _ in range(num_states)]

ensemble_params = AnalogSimParams(
    observables=[],
    elapsed_time=5.0,
    dt=0.15,
    max_bond_dim=64,
    threshold=1e-10,
    sample_timesteps=True,
    show_progress=False,
    compute_autocorrelator=True,
    autocorrelator_observable=Observable(Z(), mid),
    two_time_correlators=[(Observable(Z(), mid), Observable(X(), mid))],
)

simulator.run(ensemble_states, H_open, ensemble_params, noise_model=None, parallel=False)
t_ens = ensemble_params.autocorrelator_times
czz_ens = ensemble_params.autocorrelator_results
czx_ens = ensemble_params.two_time_correlator_results[0]
```

```{code-cell} ipython3
fig, ax = plt.subplots(1, 1, figsize=(5.8, 3.4))
ax.plot(t_ens, np.real(czz_ens), "o-", label=r"ensemble $C_{zz}(t)$")
ax.plot(t_ens, np.real(czx_ens), "s--", label=r"ensemble $C_{zx}(t)$")
ax.set_xlabel("t")
ax.set_ylabel(r"$\overline{C}_{ab}(t)$")
ax.set_title(f"Typicality-style ensemble average (N={num_states})")
ax.legend()
ax.grid(alpha=0.3)
plt.show()
```

In the above plot, we see that on computing the ensemble average of the two-time correlators, the $C_{zz}(t)$ converges monotonically to zero, and the $C_{xz}(t)$ stays relatively around zero in contrast to the values we saw in a single MPS trajectory earlier.

## 4. Spin transport example: periodic spin-current autocorrelator

For periodic XXZ chains, define local bond current
\[
j*r = J*{xx} (S*r^x S*{r+1}^y - S*r^y S*{r+1}^x),
\]
and total current $J = \sum_r j_r$.
The normalized autocorrelator
\[
C\_{JJ}(t) = \frac{1}{L}\,\langle J(t)\,J(0)\rangle
\]
can be assembled from all bond-pair two-time correlators.
Such current autocorrelators are central to linear-response spin transport; dynamical typicality makes it practical to estimate high-temperature ensemble quantities from a few random pure-state trajectories {cite:p}`steinigeweg2014_prl_spin_current`.
For finite-temperature Drude weights, diffusion, and integrable XXZ phenomenology—including the role of conservation laws—see the review {cite:p}`bertini2020_arxiv_1d_transport_review`.

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

states_transport = [MPS(Ltr, state="haar-random", pad=2) for _ in range(n_transport_states)]
bond_obs = current_observables(Ltr, Jxx)
pairs_jj = [(a, b) for a in bond_obs for b in bond_obs]

transport_curves: dict[float, np.ndarray] = {}
t_transport = None
for d in deltas:
    h_periodic = MPO.hamiltonian(
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
        threshold=1e-10,
        sample_timesteps=True,
        show_progress=False,
        two_time_correlators=pairs_jj,
    )
    simulator.run(states_transport, h_periodic, sp, noise_model=None, parallel=False)
    assert sp.two_time_correlator_results is not None
    t_transport = sp.two_time_correlator_times
    c_jj = np.real(np.sum(sp.two_time_correlator_results, axis=0) / Ltr)
    transport_curves[d] = c_jj
```

```{code-cell} ipython3
fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.5))
for d in deltas:
    ax.plot(t_transport, transport_curves[d], marker="o", ms=3, label=rf"$\Delta={d}$")
ax.set_xlabel("t")
ax.set_ylabel(r"$C_{JJ}(t)$")
ax.set_title("Periodic XXZ spin-current autocorrelator (small illustrative setup)")
ax.legend()
ax.grid(alpha=0.3)
plt.show()
```

This finite-size, short-time run already shows different relaxation trends for different anisotropies.
In the thermodynamic limit and Kubo picture, the long-time behavior of $C_{JJ}(t)$ is tied to the spin Drude weight and to ballistic versus diffusive transport in the XXZ chain; {cite:p}`bertini2020_arxiv_1d_transport_review` summarizes the established finite-temperature picture (including subtleties at $\Delta=1$ and in finite systems).
The illustrative curves here use small $L$ and a handful of Haar-random states; larger-scale or higher-accuracy studies follow typicality, as seen in {cite:p}`steinigeweg2014_prl_spin_current`.

:::{tip} Practical notes: scaling runs and MPS entanglement

- Scale gradually: `L`, ensemble size, `dt`, `elapsed_time`, and `max_bond_dim`.
- Enable ensemble parallelization (`parallel=True` in `simulator.run`) when you have many initial states.
- **MPS entanglement:** under unitary evolution, entanglement entropy and required bond dimension typically **grow** with time (until truncation or saturation). For longer times or larger $L$, increase `max_bond_dim`, tighten `threshold` only with care, or shorten the window so the MPS remains an accurate ansatz for your observable.

:::
