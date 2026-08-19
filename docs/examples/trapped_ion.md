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

# Static and Moving Trapped-Ion Position-Grid Emulation

This example evolves a **single ion** on a finite position grid with
{meth}`~mqt.yaqs.core.data_structures.mpo.MPO.trapped_ion`. Each ion is one MPO
site; the local Hilbert space is the grid itself. The Hamiltonian combines a
finite-difference kinetic term and a harmonic trap—see {doc}`hamiltonians` for
the factory API and two-ion Coulomb extensions.

We first initialize a displaced harmonic-oscillator wavepacket in a static
central well. In the continuum limit, its center follows
$\langle x(t)\rangle = x_0 \cos(\omega t)$, so after half a trap period it
reaches the opposite turning point.

## 1. Hamiltonian and initial state

```{code-cell} ipython3
import numpy as np

from mqt.yaqs import Hamiltonian, MPO, Observable, State

omega = 1.0
initial_displacement = 1.0
half_period = np.pi / omega

positions = np.linspace(-8.0, 8.0, 33)
grid_dim = len(positions)

initial_grid_state = np.exp(-0.5 * (positions - initial_displacement) ** 2).astype(np.complex128)
initial_grid_state /= np.linalg.norm(initial_grid_state)

hamiltonian = Hamiltonian.from_mpo(MPO.trapped_ion(positions, masses=[1.0], omega=omega))
state = State(length=1, vector=initial_grid_state, physical_dimensions=[grid_dim])
position_observable = Observable("position", 0, positions=positions)
```

## 2. Noiseless evolution to $T/2$

```{code-cell} ipython3
from mqt.yaqs import AnalogSimParams, Simulator

params = AnalogSimParams(
    observables=[position_observable],
    elapsed_time=half_period,
    dt=half_period / 16,
    max_bond_dim=None,
    svd_threshold=1e-12,
    krylov_tol=1e-12,
    preset="exact",
    get_state=True,
    sample_timesteps=True,
)

result = Simulator(show_progress=False).run(state, hamiltonian, params)
final_state = result.output_state.vector
position_expectation = np.real(result.expectation_values[0])
final_x = float(position_expectation[-1])
```

The position observable is a custom one-site matrix on the grid basis. The final
$\langle x\rangle$ is close to $-x_0$ but not exact because the simulation uses
a finite grid and a finite-difference kinetic operator.

```{code-cell} ipython3
print(f"Initial <x>       = {initial_displacement:.6f}")
print(f"Final <x> at T/2  = {final_x:.6f}")
print(f"Continuum target  = {-initial_displacement:.6f}")
```

## 3. Wavepacket over time

```{code-cell} ipython3
---
mystnb:
  image:
    width: 90%
    align: center
---
import matplotlib.pyplot as plt

dense_hamiltonian = hamiltonian.to_matrix()
eigenvalues, eigenvectors = np.linalg.eigh(dense_hamiltonian)
coefficients = eigenvectors.conj().T @ initial_grid_state
phases = np.exp(-1j * eigenvalues[:, None] * params.times[None, :])
states = eigenvectors @ (coefficients[:, None] * phases)
probability_density = np.abs(states) ** 2

fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
image = ax.imshow(
    probability_density,
    aspect="auto",
    origin="lower",
    extent=(params.times[0], params.times[-1], positions[0], positions[-1]),
    cmap="viridis",
)
ax.plot(params.times, position_expectation, color="white", lw=1.4, label=r"$\langle x\rangle$")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$x$")
ax.set_title("Position-grid wavepacket density")
ax.legend(loc="upper right")
fig.colorbar(image, ax=ax, label=r"$|\psi(x,t)|^2$")
plt.show()
```

## 4. Transport in a moving harmonic well

A moving trap is a piecewise Hamiltonian: one static well per transport step,
then a hold at the target. Here a linear trajectory transports the well from
$q\nobreak=\nobreak-1$ to $q\nobreak=\nobreak1$.

```{code-cell} ipython3
transport_positions = np.linspace(-6.0, 6.0, 25)
transport_grid_dim = len(transport_positions)
start_center = -1.0
target_center = 1.0
transport_duration = 10.0
hold_duration = 5.0
dt = 0.25


def trap_at(trap_center: float) -> Hamiltonian:
    return Hamiltonian.from_mpo(
        MPO.trapped_ion(
            transport_positions,
            masses=[1.0],
            omega=omega,
            trap_center=trap_center,
        )
    )


n_transport = round(transport_duration / dt)
transport_pieces = [
    (trap_at(start_center + (target_center - start_center) * (step / n_transport)), dt)
    for step in range(n_transport)
]
moving_hamiltonian = Hamiltonian.piecewise([
    *transport_pieces,
    (trap_at(target_center), hold_duration),
])

transport_wavepacket = np.exp(-0.5 * (transport_positions - start_center) ** 2).astype(np.complex128)
transport_wavepacket /= np.linalg.norm(transport_wavepacket)
transport_state = State(
    length=1,
    tensors=[transport_wavepacket.reshape(transport_grid_dim, 1, 1)],
    physical_dimensions=[transport_grid_dim],
)
transport_position = Observable("position", 0, positions=transport_positions)
transport_params = AnalogSimParams(
    observables=[transport_position],
    elapsed_time=transport_duration + hold_duration,
    dt=dt,
    tdvp_sweeps=2,
    max_bond_dim=None,
    svd_threshold=1e-12,
    krylov_tol=1e-12,
    preset="exact",
    sample_timesteps=True,
)

transport_result = Simulator(parallel=False, show_progress=False).run(
    transport_state,
    moving_hamiltonian,
    transport_params,
)
transport_expectation = np.real(transport_result.expectation_values[0])
scheduled_centers = np.asarray(
    [start_center + (target_center - start_center) * min(time / transport_duration, 1.0) for time in transport_params.times]
)
hold_mask = transport_params.times >= transport_duration
residual_motion = np.max(np.abs(transport_expectation[hold_mask] - target_center))
assert residual_motion > 0.1
print(f"Maximum displacement from the target during the hold: {residual_motion:.3f}")
```

Each transport interval uses a fixed trap center. During the hold the well stays
at the target while the ion continues to move.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7.2, 3.4), layout="constrained")
ax.plot(transport_params.times, scheduled_centers, "--", label=r"trap center $q(t)$")
ax.plot(transport_params.times, transport_expectation, label=r"ion $\langle x(t)\rangle$")
ax.axvline(transport_duration, color="0.6", ls=":", label="end of transport")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$x$")
ax.set_title("Residual motion after linear trap transport")
ax.legend()
plt.show()
```

The ion does not end at rest: during the hold, $q(t)$ remains fixed while
$\langle x(t)\rangle$ oscillates around the target. This linear protocol is
intentionally idealized and highly suboptimal: although the trap position is
continuous, its velocity changes discontinuously at the start and end. The
resulting residual motion can degrade the fidelity of subsequent operations in
practice, illustrating the importance of smooth, optimized control ramps.

This example uses dimensionless units with $hbar=m=omega=1$. For dimensional
inputs, use compatible time and energy units. A finer grid, smaller `dt`, and
slower trajectory reduce spatial and non-adiabatic transport errors.

## Related topics

- {doc}`hamiltonians` — `MPO.trapped_ion` parameters and two-ion Coulomb
  channels
- {doc}`transmon_emulation` — another mixed-dimensional hardware model
- {doc}`analog_simulation` — analog time evolution and noise models
- {doc}`state_initialization` — custom `physical_dimensions` and manual vectors
