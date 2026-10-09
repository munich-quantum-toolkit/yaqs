---
file_format: mystnb
kernelspec:
  name: python3
language_info:
  name: python
mystnb:
  number_source_lines: true
  execution_timeout: 120
---

# Noise Models

A {class}`~mqt.yaqs.NoiseModel` describes how a system loses energy, gains
excitations, or suffers other disturbances during a simulation. Assemble named
jump operators or supply your own matrices, then pass the model to
`Simulator.run`. Use fixed strengths for a calibrated model, or distributions to
represent static variation between runs.

## Define the noise processes

Each process has a `name`, a list of `sites`, and a `strength`. Site indices
start at zero. This four-qubit model combines local relaxation and dephasing:

```{code-cell} python
from mqt.yaqs import NoiseModel

length = 4
noise = NoiseModel([
    {"name": "lowering", "sites": [site], "strength": 0.1}
    for site in range(length)
] + [
    {"name": "pauli_z", "sites": [site], "strength": 0.02}
    for site in range(length)
])
```

The named operators act on qubits. Names are case-sensitive:

| Process name                          | Effect                                                                | Sites                             |
| ------------------------------------- | --------------------------------------------------------------------- | --------------------------------- |
| `"lowering"`                          | Relaxation from $\lvert1\rangle$ to $\lvert0\rangle$.                 | One.                              |
| `"raising"`                           | Excitation from $\lvert0\rangle$ to $\lvert1\rangle$.                 | One.                              |
| `"pauli_x"`, `"pauli_y"`, `"pauli_z"` | Pauli errors; $Z$ causes dephasing. Aliases: `"x"`, `"y"`, `"z"`.     | One.                              |
| `"lowering_two"`, `"raising_two"`     | Joint relaxation $\lvert11\rangle\to\lvert00\rangle$, or the reverse. | Two adjacent sites.               |
| `"crosstalk_xx"`, `"crosstalk_xy"`, … | Correlated Pauli errors; any pair of `x`, `y`, and `z`.               | Two; see the support table below. |

Joint relaxation is one two-site jump. To model independent relaxation on two
sites, supply two `"lowering"` processes instead.

## Interpret the strengths

For `Simulator`, `strength` is a finite, nonnegative Lindblad rate $\gamma$. Use
inverse units of the simulation time. YAQS supplies the factor $\sqrt{\gamma}$;
pass the unscaled operator as `matrix`. Operators need not be Hermitian or
unitary, and YAQS does not normalize them.

Analog noise acts over the physical time steps set by `AnalogSimParams.dt`.
Circuit noise uses a unit noise step after each multi-qubit gate, with only the
processes whose sites lie within that gate's qubits. Single-qubit gates,
barriers, and idle qubits do not create noise opportunities. Circuit strengths
therefore do not represent hardware gate durations or direct error
probabilities. See {doc}`circuit_observables` for a worked example.

:::{important}
`EquivalenceChecker` interprets resolved strengths as direct per-opportunity
branch probabilities and imposes its own probability constraints. A simulator
rate cannot be reused as a checker probability without choosing a conversion.
See {ref}`equivalence-noise-model`.
:::

:::{dropdown} Relate rates to relaxation and dephasing times
For a jump operator $L$, the rate multiplies the dissipator

$$
\gamma\mathcal{D}[L](\rho)=\gamma\left(
L\rho L^\dagger-\tfrac12\{L^\dagger L,\rho\}\right).
$$

With only `"lowering"` and no Hamiltonian, the excited-state population decays
as $e^{-\gamma t}$, so $\gamma=1/T_1$. With only `"pauli_z"`, the off-diagonal
density-matrix entries decay as $e^{-2\gamma t}$, so $\gamma=1/(2T_\phi)$. These
conventions matter when converting measured lifetimes to strengths. Scaling $L$
by a factor $c$ scales its dissipator by $|c|^2$.

Negative rates, including time-local descriptions with temporarily negative
coefficients, are not supported. Negative or complex matrix entries remain valid
parts of a jump operator.
:::

## Run and inspect the model

Pass the model with the initial state, Hamiltonian, and simulation parameters.
This short Ising evolution uses the default MPS representation and averages
eight trajectories:

```{code-cell} python
from mqt.yaqs import AnalogSimParams, Hamiltonian, Observable, Simulator, State

state = State(length, initial="ones")
hamiltonian = Hamiltonian.ising(length, J=1.0, g=0.5)
params = AnalogSimParams(
    observables=[Observable("z", site) for site in range(length)],
    elapsed_time=0.2,
    dt=0.05,
    num_traj=8,
    random_seed=7,
)

sim = Simulator(show_progress=False)
result = sim.run(state, hamiltonian, params, noise_model=noise)
print(result.expectation_values[0].shape)
```

The shape is `(5,)`: five sampled times, including the initial time.
`result.expectation_values` contains one such array per observable, in the
supplied order. Parallel execution remains enabled; the documentation hides
progress bars. Eight trajectories suffice to demonstrate the call, but
scientific results need a convergence check. The model used in the run is
available as `result.noise_model`.

## Represent static variation

Replace a scalar strength with a distribution dictionary. This example gives
each qubit an independent log-normal relaxation rate:

```{code-cell} python
import numpy as np

variable_noise = NoiseModel([
    {
        "name": "lowering",
        "sites": [site],
        "strength": {"distribution": "lognormal", "mean": np.log(0.1), "std": 0.4},
    }
    for site in range(length)
])

resolved_noise = variable_noise.sample(rng=7)
print([process["strength"] for process in resolved_noise.processes])
```

{meth}`~mqt.yaqs.NoiseModel.sample` returns a new model with concrete rates; it
leaves the original model unchanged. For `"lognormal"`, `mean` and `std`
describe the normal distribution of $\log\gamma$, so the median rate above is
`0.1`.

| `distribution`       | Meaning of `mean` and `std`                                | Treatment of negative draws                                     |
| -------------------- | ---------------------------------------------------------- | --------------------------------------------------------------- |
| `"lognormal"`        | Mean and standard deviation of $\log\gamma$.               | All draws are positive.                                         |
| `"normal"`           | Mean and standard deviation of a normal rate distribution. | Clamped to zero with a warning.                                 |
| `"truncated_normal"` | Parameters of the underlying normal distribution.          | Sampled from that distribution restricted to nonnegative rates. |

Passing `variable_noise` directly to `Simulator.run` draws one rate per process
at the start of the run. All trajectories share those rates. This represents
static disorder: rates do not change during the evolution or between
trajectories. To average over disorder, perform separate runs with separate
draws; increasing `num_traj` only improves the trajectory average for one draw.

To compare setups using the same rates, pass a resolved model:

```{code-cell} python
resolved_result = sim.run(state, hamiltonian, params, noise_model=resolved_noise)
print([process["strength"] for process in resolved_result.noise_model.processes])
```

The printed rates match the earlier sample. You can also reuse a previous run's
`result.noise_model`. Setting `params.random_seed` fixes automatic disorder
draws and trajectory random streams for a fixed setup. It does not seed
independently prepared random states or final shot sampling, or guarantee
identical numbers across software versions and platforms. For successive manual
disorder draws, reuse a NumPy `Generator`; repeating `.sample(rng=7)` repeats
the same draw.

:::{dropdown} Distribution parameters and limits
`mean` and `std` default to zero when omitted; specify both to make the intended
distribution clear. `mean` must be finite, and `std` must be finite and
nonnegative. A zero-width normal or truncated normal resolves to `max(0, mean)`;
a zero-width log-normal resolves to `exp(mean)`.

There is no default distribution or upper rate limit. Choose a distribution from
the variation you intend to model, and choose the time step to resolve the
resulting dynamics. Clamping a normal distribution creates a point mass at zero;
truncating it renormalizes the positive part instead. Neither choice creates a
time-dependent noise model.
:::

(noise-custom-operators)=

## Supply a custom operator

Add `matrix` to override the library lookup; `name` then serves as an
identifier. A one-site operator must be a finite, square matrix matching that
site's local dimension. This explicit matrix gives the same jump as
`"lowering"`:

```{code-cell} python
sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
custom_noise = NoiseModel([
    {"name": "relaxation", "sites": [0], "strength": 0.1, "matrix": sigma_minus},
    {"name": "pauli_z", "sites": [1], "strength": 0.02},
])
```

You can mix custom and named processes. For higher local dimensions, supply an
operator in that local basis. A truncated oscillator with levels $0,1,2$ uses
the annihilation operator

```{code-cell} python
annihilation = np.diag(np.sqrt(np.arange(1, 3)), k=1)
qutrit_noise = NoiseModel([
    {"name": "loss", "sites": [0], "strength": 0.1, "matrix": annihilation},
])
```

This matrix is $3\times3$ and requires a dimension-three site. See
{doc}`transmon_emulation` for a device example with such operators.

## Choose a supported combination

The state representation selects the analog backend. One-site and adjacent
two-site process matrices work with all three analog representations. Long-range
processes need the choices below:

| Workflow                         | One-site and adjacent two-site noise                    | Non-adjacent two-site noise                                         |
| -------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------- |
| Analog MPS (TJM)                 | Supported.                                              | Products of Pauli operators, up to a unit-modulus phase per factor. |
| Analog vector (MCWF)             | Supported.                                              | Custom factor pairs supported.                                      |
| Analog density matrix (Lindblad) | Supported.                                              | Custom factor pairs supported.                                      |
| Circuit simulation (MPS)         | Supported on the gate qubits at each noise opportunity. | Not supported.                                                      |

All operators must match the site's local dimension. Named Pauli operators are
$2\times2$; supply custom matrices for other dimensions. YAQS checks dimensions
and site bounds against the state when the simulation starts. See
{doc}`representation_comparison` to choose a representation.

Noisy MPS, vector, and circuit runs cannot retain a final pure state with
`get_state=True`; noisy density-matrix runs can retain their final mixed state.
The analog `list[State]` ensemble workflow does not support process noise. See
{doc}`simulation_parameters` for output choices.

:::{dropdown} Construct adjacent and long-range two-site processes
For adjacent sites, supply a `matrix` acting on their joint basis. Use ascending
site order: the first tensor factor acts on the first listed site. If a matrix
was written for descending sites, swap both its input and output tensor-factor
axes as well as reversing the site list.

```python
adjacent_noise = NoiseModel([
    {
        "name": "joint_relaxation",
        "sites": [0, 1],
        "strength": 0.05,
        "matrix": np.kron(sigma_minus, sigma_minus),
    },
])
```

For non-adjacent sites, the names `"crosstalk_xy"` and
`"longrange_crosstalk_xy"` both construct the factors $X$ and $Y$. Any pair of
`x`, `y`, and `z` is accepted:

```python
long_range_pauli = NoiseModel([
    {"name": "longrange_crosstalk_xy", "sites": [0, 3], "strength": 0.05},
])
```

For a custom non-adjacent process, supply two local `factors` in the order of
the listed sites. This lowering-and-dephasing product requires an analog vector
or density-matrix simulation:

```python
long_range_custom = NoiseModel([
    {
        "name": "correlated_loss",
        "sites": [0, 3],
        "strength": 0.05,
        "factors": (sigma_minus, np.diag([1, -1])),
    },
])
```

YAQS sorts the sites and reorders the factors together. Use `matrix` for
adjacent sites and `factors` for non-adjacent sites; do not provide both.
:::

(noise-scheduled-jumps)=

## Apply a scheduled jump

Use `scheduled_jumps` for an operator applied at a specified analog time. Each
entry gives a `time`, `sites`, and a library `name`, or a custom `matrix` with
an identifying name. Scheduled events have no `strength`: the operator is
applied when the simulation reaches its time.

This four-site example isolates the event by setting the Hamiltonian to zero. An
$X$ flip at $t=0.1$ changes $\langle Z_0\rangle$ from $+1$ to $-1$:

```{code-cell} python
scheduled_noise = NoiseModel(scheduled_jumps=[
    {"time": 0.1, "sites": [0], "name": "x"},
])
jump_state = State(length, initial="zeros")
zero_hamiltonian = Hamiltonian.ising(length, J=0.0, g=0.0)
jump_params = AnalogSimParams(
    observables=[Observable("z", 0)],
    elapsed_time=0.2,
    dt=0.05,
    num_traj=1,
    order=1,
)
jump_result = sim.run(jump_state, zero_hamiltonian, jump_params, noise_model=scheduled_noise)
print(jump_result.expectation_values[0])
```

The five values are `[1, 1, -1, -1, -1]`, sampled at times
`[0, 0.05, 0.1, 0.15, 0.2]`. Event-only runs are deterministic, so one
trajectory suffices; they can also retain the final MPS with `get_state=True`.

Scheduled jumps require a single MPS `State` and `AnalogSimParams(order=1)`.
MCWF, Lindblad, order-2 TJM, circuit runs, and `list[State]` ensembles do not
support them. Event times must lie on the simulation time grid, including its
endpoints. Two-site events must act on adjacent sites; custom matrices require
ascending site order.

:::{important}
At a matching time after zero, scheduled operators replace the ordinary
stochastic-jump draw for that step; process dissipation still runs. For control
pulses that also sample stochastic noise at pulse times, use an
{doc}`analog-digital program <digital_analog_simulation>`.
:::

:::{dropdown} Custom scheduled operators and event order
Add `matrix` to supply a custom operator. For example, this schedules a $\pi/2$
rotation about $Y$:

```python
ry_pi2 = np.array([[1, -1], [1, 1]], dtype=complex) / np.sqrt(2)
custom_event = NoiseModel(
    scheduled_jumps=[
        {"time": 0.1, "sites": [0], "name": "ry_pi2", "matrix": ry_pi2},
    ]
)
```

Matrices must be finite, square, and match the selected sites. Scheduled events
accept `matrix`, not long-range `factors`. Operators need not be unitary; YAQS
normalizes the state after the events and rejects a zero or nonfinite norm. A
non-unitary scheduled operator is a prescribed normalized state update, not a
randomly sampled Lindblad channel.

At time zero, events act before dissipation and the initial measurement. At
later grid points, including the final time, the order is Hamiltonian evolution,
process dissipation, all matching scheduled operators, normalization, then
measurement. Events at the same time follow their order in `scheduled_jumps`.
Inside a `SimulationProgram`, event times use the local clock of an analog run;
see {doc}`digital_analog_simulation` for segment boundaries and noise overrides.
:::

## Next steps

Use {doc}`analog_simulation` or {doc}`circuit_observables` to see how noise
changes measured dynamics. The device guides show how to combine noise with
{doc}`transmon_emulation` and {doc}`trapped_ion`. For deterministic control
sequences that combine gates with analog evolution, see
{doc}`digital_analog_simulation`.
