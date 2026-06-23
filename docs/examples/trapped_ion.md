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

This example builds a static single-ion Hamiltonian with
{meth}`~mqt.yaqs.core.data_structures.mpo.MPO.trapped_ion`. Each ion is one MPO
site, and the local basis is a finite position grid. The Hamiltonian contains a
finite-difference kinetic term and a harmonic trap.

We initialize a displaced harmonic-oscillator ground-state wavepacket in a static
central well. In the continuum limit, its center follows
$\langle x(t)\rangle = x_0 \cos(\omega t)$, so after half a trap period it reaches
the opposite turning point.

## Static central well

```{code-cell} ipython3
import numpy as np

from mqt.yaqs import AnalogSimParams, Hamiltonian, MPO, Simulator, State

omega = 1.0
initial_displacement = 1.0
half_period = np.pi / omega

positions = np.linspace(-8.0, 8.0, 33)
grid_dim = len(positions)

initial_grid_state = np.exp(-0.5 * (positions - initial_displacement) ** 2).astype(np.complex128)
initial_grid_state /= np.linalg.norm(initial_grid_state)

hamiltonian = Hamiltonian.from_mpo(MPO.trapped_ion(positions, masses=[1.0], omega=omega))
state = State(length=1, vector=initial_grid_state, physical_dimensions=[grid_dim])

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

result = Simulator(parallel=False, show_progress=False).run(state, hamiltonian, params)
assert result.output_state is not None
final_state = result.output_state.vector
final_x = float(np.sum(positions * np.abs(final_state) ** 2))

print(f"Initial <x>       = {initial_displacement:.6f}")
print(f"Final <x> at T/2  = {final_x:.6f}")
print(f"Continuum target  = {-initial_displacement:.6f}")
```

The final value is close to $-x_0$ but not exact because the simulation uses a
finite grid and a finite-difference kinetic operator.

## Related topics

- {doc}`transmon_emulation` — another mixed-dimensional hardware model
- {doc}`analog_simulation` — analog time evolution and noise models
- {doc}`state_initialization` — custom `physical_dimensions` and manual vectors
