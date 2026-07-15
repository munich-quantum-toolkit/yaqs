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

# Trapped-Ion Position-Grid Emulation

This example evolves a **single ion** on a finite position grid with
{meth}`~mqt.yaqs.core.data_structures.mpo.MPO.trapped_ion`. Each ion is one MPO site;
the local Hilbert space is the grid itself. The Hamiltonian combines a
finite-difference kinetic term and a harmonic trap—see {doc}`hamiltonians` for the
factory API and two-ion Coulomb extensions.

We initialize a displaced harmonic-oscillator wavepacket in a static central well.
In the continuum limit, its center follows $\langle x(t)\rangle = x_0 \cos(\omega t)$,
so after half a trap period it reaches the opposite turning point.

## 1. Hamiltonian and initial state

```{code-cell} ipython3
import numpy as np

from mqt.yaqs import Hamiltonian, MPO, State

omega = 1.0
initial_displacement = 1.0
half_period = np.pi / omega

positions = np.linspace(-8.0, 8.0, 33)
grid_dim = len(positions)

initial_grid_state = np.exp(-0.5 * (positions - initial_displacement) ** 2).astype(np.complex128)
initial_grid_state /= np.linalg.norm(initial_grid_state)

hamiltonian = Hamiltonian.from_mpo(MPO.trapped_ion(positions, masses=[1.0], omega=omega))
state = State(length=1, vector=initial_grid_state, physical_dimensions=[grid_dim])
```

## 2. Noiseless evolution to $T/2$

```{code-cell} ipython3
from mqt.yaqs import AnalogSimParams, Simulator

params = AnalogSimParams(
    observables=[],
    elapsed_time=half_period,
    dt=half_period / 16,
    max_bond_dim=None,
    svd_threshold=1e-12,
    krylov_tol=1e-12,
    preset="exact",
    get_state=True,
    sample_timesteps=False,
)

result = Simulator(show_progress=False).run(state, hamiltonian, params)
final_state = result.output_state.vector
final_x = float(np.sum(positions * np.abs(final_state) ** 2))
```

The final $\langle x\rangle$ is close to $-x_0$ but not exact because the simulation uses a
finite grid and a finite-difference kinetic operator.

```{code-cell} ipython3
print(f"Initial <x>       = {initial_displacement:.6f}")
print(f"Final <x> at T/2  = {final_x:.6f}")
print(f"Continuum target  = {-initial_displacement:.6f}")
```

## 3. Wavepacket at $t=0$ and $t=T/2$

```{code-cell} ipython3
---
mystnb:
  image:
    width: 90%
    align: center
---
import matplotlib.pyplot as plt

prob_initial = np.abs(initial_grid_state) ** 2
prob_final = np.abs(final_state) ** 2

fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), layout="constrained", sharey=True)
axes[0].fill_between(positions, prob_initial, alpha=0.35, color="tab:blue")
axes[0].plot(positions, prob_initial, color="tab:blue", lw=1.5)
axes[0].set_title(r"$t = 0$")
axes[0].set_xlabel(r"$x$")
axes[0].set_ylabel(r"$|\psi(x)|^2$")
axes[0].grid(alpha=0.3)

axes[1].fill_between(positions, prob_final, alpha=0.35, color="tab:orange")
axes[1].plot(positions, prob_final, color="tab:orange", lw=1.5)
axes[1].set_title(rf"$t = T/2$")
axes[1].set_xlabel(r"$x$")
axes[1].grid(alpha=0.3)

fig.suptitle("Harmonic wavepacket reflection on a position grid")
plt.show()
```

## Related topics

- {doc}`hamiltonians` — `MPO.trapped_ion` parameters and two-ion Coulomb channels
- {doc}`transmon_emulation` — another mixed-dimensional hardware model
- {doc}`analog_simulation` — analog time evolution and noise models
- {doc}`state_initialization` — custom `physical_dimensions` and manual vectors
