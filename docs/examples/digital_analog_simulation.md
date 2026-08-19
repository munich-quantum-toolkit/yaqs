---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
  execution_timeout: 600
---

```{code-cell} ipython3
:tags: [remove-cell]
%config InlineBackend.figure_formats = ['svg']
```

# Digital–Analog Simulation

Some simulations need both analog parts (continuous Hamiltonian evolution) and
digital parts (quantum circuits). A YAQS {class}`~mqt.yaqs.SimulationProgram`
accepts an ordered list of `(operator, params)` pairs and runs them as one
program. YAQS passes the evolving state from one segment to the next, keeps each
noisy trajectory continuous across segment boundaries, and returns a normal
{class}`~mqt.yaqs.Result` with stitched `times` and `expectation_values`.

To demonstrate this workflow, the following example:

1. prepares a phase-sensitive state with a digital operation;
2. evolves it continuously under a simple static-$Z$ Hamiltonian that
   accumulates phase;
3. optionally inserts an instantaneous digital pulse between two analog parts;
4. runs both programs with and without noise.

## 1. Create the digital operations

One qubit is enough for this demonstration. A Hadamard gate prepares the
phase-sensitive state $\lvert+\rangle$ from the initial state $\lvert0\rangle$
(`"zeros"`, specified later on). An $X$ gate will act as the midpoint refocusing
pulse and, when repeated at the end, return the sequence to its original frame.

```{code-cell} ipython3
from qiskit.circuit import QuantumCircuit

number_of_qubits = 1

preparation = QuantumCircuit(number_of_qubits)
preparation.h(0);

refocusing_pulse = QuantumCircuit(number_of_qubits)
refocusing_pulse.x(0);
```

These are normal Qiskit circuits. They will appear as the first entry of each
digital pair in the program.

## 2. Configure the parameters

Segment parameters carry timing, truncation, gate-mode, and digital `shots`.
Observables, `random_seed`, and `get_state` belong on the program; `num_traj` is
either unanimous on the segments or set on the program.

```{code-cell} ipython3
from mqt.yaqs import AnalogSimParams, DigitalSimParams, Hamiltonian, Observable

half_duration = 0.7
number_of_trajectories = 256

hamiltonian = Hamiltonian.pauli(
    length=number_of_qubits,
    one_body=[(1.1, "z")],
)
x_observable = Observable("x", 0)
analog_parameters = AnalogSimParams(
    elapsed_time=half_duration,
    dt=0.05,
    sample_timesteps=True,
)
digital_parameters = DigitalSimParams(sample_layers=True)
```

`elapsed_time` is the duration of each analog segment, while `dt` controls its
time-step size. With `sample_layers=True`, digital segments also record the
shared observable at circuit entry and exit.

## 3. Put the operations into programs

A program is an ordered list of `(operator, params)` pairs. The operator type
selects the mode: a {class}`~qiskit.circuit.QuantumCircuit` (or an OpenQASM
string / file path) with {class}`~mqt.yaqs.DigitalSimParams`, or a
{class}`~mqt.yaqs.Hamiltonian` with {class}`~mqt.yaqs.AnalogSimParams`.

```{code-cell} ipython3
from mqt.yaqs import SimulationProgram

free_evolution = SimulationProgram(
    [
        (preparation, digital_parameters),
        (hamiltonian, analog_parameters),
        (hamiltonian, analog_parameters),
    ],
    observables=[x_observable],
    num_traj=number_of_trajectories,
    random_seed=7,
)
```

The second program inserts the refocusing pulse between the analog intervals and
repeats it at the end as a frame correction.

```{code-cell} ipython3
evolution_with_hahn_echo = SimulationProgram(
    [
        (preparation, digital_parameters),
        (hamiltonian, analog_parameters),
        (refocusing_pulse, digital_parameters),
        (hamiltonian, analog_parameters),
        (refocusing_pulse, digital_parameters),
    ],
    observables=[x_observable],
    num_traj=number_of_trajectories,
    random_seed=7,
)
```

There is no need to run these segments individually or manually extract and
resubmit an intermediate state. YAQS carries the state through each complete
list in order.

## 4. Run the programs with and without noise

Create a simulator and an initial state, then pass each program to
{meth}`~mqt.yaqs.Simulator.run`.

```{code-cell} ipython3
from mqt.yaqs import NoiseModel, Simulator, State

simulator = Simulator(parallel=False, show_progress=False)
initial_state = State(number_of_qubits, initial="zeros")

free_noiseless = simulator.run(initial_state, free_evolution)
echo_noiseless = simulator.run(initial_state, evolution_with_hahn_echo)
```

Adding noise does not require changing either program. Pass YAQS's built-in
Markovian `pauli_z` dephasing process to the same calls.

```{code-cell} ipython3
dephasing_noise = NoiseModel(
    [{"name": "pauli_z", "sites": [0], "strength": 0.15}]
)

free_noisy = simulator.run(
    initial_state,
    free_evolution,
    noise_model=dephasing_noise,
)
echo_noisy = simulator.run(
    initial_state,
    evolution_with_hahn_echo,
    noise_model=dephasing_noise,
)
```

During a noisy run, each trajectory passes through the complete program on one
worker (one MPS and one RNG) before YAQS averages the recorded observables.
Parallelism is over trajectories, not over segments.

You can also pass the pair list directly to `run`, which builds a
`SimulationProgram` under the hood:

```python
result = simulator.run(
    initial_state,
    [(preparation, digital_parameters), (hamiltonian, analog_parameters)],
    observables=[x_observable],
    num_traj=number_of_trajectories,
    random_seed=7,
)
```

## 5. OpenQASM, scheduled jumps, and per-segment results

Programs reuse the same **operator / params / noise** objects as MPS TJM analog
and digital runs. A few knobs are program-owned or unsupported here: the initial
state must be MPS; observables, `random_seed`, and `get_state` belong on the
program; `multi_time_observables` are not supported.

**Digital operators.** A segment may take a
{class}`~qiskit.circuit.QuantumCircuit`, or an OpenQASM string / file path, with
{class}`~mqt.yaqs.DigitalSimParams`. `shots` stay on the digital segment params:

```python
program = SimulationProgram(
    [("prep.qasm", digital_parameters), (hamiltonian, analog_parameters)],
    observables=[x_observable],
)
```

**Scheduled jumps.** Attach them through a {class}`~mqt.yaqs.NoiseModel` as in
{doc}`scheduled_jumps` (analog MPS TJM, `order=1`, times on that analog run's
`dt` grid). Jump times are relative to the start of the analog run that carries
the model. Consecutive analog segments that share one interval schedule also
share that clock; a digital gate starts a new analog run. Use a third-tuple
noise override when only one segment should fire them:

```python
jumps = NoiseModel(scheduled_jumps=[{"time": 0.1, "sites": [0], "name": "x"}])
program = SimulationProgram(
    [
        (preparation, digital_parameters),
        (hamiltonian, analog_parameters, jumps),
    ],
    observables=[x_observable],
)
```

**Trajectory count.** Set `num_traj=` on the program, or use the same `num_traj`
on every segment. If segments disagree, pass an explicit program value.

**Results.** The outer result stitches `result.times` and
`result.expectation_values`. Outer `result.counts` is the histogram from the
last segment that recorded shots. Each segment also keeps an ordinary
{class}`~mqt.yaqs.Result` at `result.segment_results[i]`.

## 6. Compare the four results

Program results look like ordinary analog/digital results: use `result.times`
and `result.expectation_values`. Digital samples sit at their physical time
offset (operations are instantaneous on the program timeline), so repeated
timestamps around pulses are expected.

```{code-cell} ipython3
times = echo_noiseless.times
signal = echo_noiseless.expectation_values[0]
```

We now plot the results, indicating noisy simulation with dashed lines.

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
import matplotlib.pyplot as plt

colors = plt.colormaps["viridis"]([0.2, 0.75])
traces = [
    (free_noiseless, "free evolution, noiseless", colors[0], "-"),
    (echo_noiseless, "with digital pulses, noiseless", colors[1], "-"),
    (free_noisy, "free evolution, noisy", colors[0], "--"),
    (echo_noisy, "with digital pulses, noisy", colors[1], "--"),
]

fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
for result, label, color, line_style in traces:
    ax.plot(
        result.times,
        result.expectation_values[0],
        "o",
        color=color,
        linestyle=line_style,
        markersize=2.5,
        label=label,
    )

ax.axvline(half_duration, color="0.65", linewidth=1, label="digital pulse")
ax.set(
    xlabel="Time",
    ylabel=r"Phase-sensitive signal $\langle X\rangle$",
    ylim=(-1.05, 1.05),
)
ax.legend(ncols=2)
plt.show()
```

This simulation combines continuous Hamiltonian evolution and instantaneous
digital operations in one digital–analog program. In this simple application,
the digital midpoint pulse reverses the coherent phase accumulated during the
analog evolution, a phase-cancellation technique known as a Hahn echo. With
Markovian dephasing, the coherent phase is still cancelled, but the pulse cannot
recover lost signal contrast.

## Useful things to know

- Segments run in the order they appear in `SimulationProgram`.
- Analog-only Hamiltonian quenches belong in {doc}`hamiltonians`
  (`Hamiltonian.piecewise`). Use a program when analog evolution sits in a
  protocol with digital gates.
- Observables, `random_seed`, and `get_state` are program-wide; leave them unset
  on segment params. `num_traj` may be unanimous on segments or set on the
  program. `shots` stay on digital segment params.
- YAQS passes the state between segments automatically and does not mutate the
  input state you give `Simulator.run`.
- A noise model passed to `Simulator.run` is inherited by every segment unless
  that segment supplies a third-tuple override (`None` inherits; an empty
  `NoiseModel()` disables noise).
- Noisy trajectories and their RNG streams remain continuous across segment
  boundaries.
- A noiseless program can retain its final state with
  `SimulationProgram(..., get_state=True)`.
- Digital segments may operate on qubit sites in a heterogeneous state while
  non-qubit sites remain spectators. Digital gates themselves currently require
  qubit targets.
- Independent trajectories may run in parallel with `Simulator(parallel=True)`.

## Related topics

- {doc}`hamiltonians` — analog quench with `Hamiltonian.piecewise`
- {doc}`analog_simulation` — standalone noisy analog evolution
- {doc}`circuit_observables` — standalone digital circuit simulation and
  OpenQASM
- {doc}`scheduled_jumps` — deterministic jumps on an analog time grid
- {doc}`simulation_parameters` — simulation accuracy and output controls
