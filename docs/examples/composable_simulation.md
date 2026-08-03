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

# Composable Analog–Digital Simulation

Ramsey and Hahn-echo experiments alternate short control operations with
continuous evolution. They are a natural use case for
{class}`~mqt.yaqs.SimulationProgram`: digital segments prepare, refocus, and
read out the spins, while analog segments describe their evolution between
those operations. YAQS transfers one evolving state through the complete
sequence, so users do not need to extract and resubmit intermediate states.

This example compares free induction under a static $Z$ detuning with a Hahn
echo. The echo has an exact physical expectation: an ideal $X$ pulse reverses
the phase accumulated under $Z$, and a final frame correction makes the full
sequence the identity.

## 1. Build the digital operations

We use two noninteracting spins, initially in $\lvert00\rangle$. Hadamard gates
prepare $\lvert++\rangle$ and later convert transverse coherence back into a
computational-basis signal. The central $X$ gates are the Hahn-echo pulse. A
second set of $X$ gates after the analog evolution corrects the known final
frame, making the refocused operation an identity rather than an identity only
up to $X$.

```{code-cell} ipython3
from qiskit.circuit import QuantumCircuit

length = 2

preparation = QuantumCircuit(length)
preparation.h(range(length))

echo_pulse = QuantumCircuit(length)
echo_pulse.x(range(length))

analysis = QuantumCircuit(length)
analysis.h(range(length))
```

These operations remain ordinary Qiskit circuits inside `DigitalSegment`.
Names such as “preparation” and “echo pulse” belong to this experiment; YAQS's
core interface only needs to distinguish digital and analog simulation.

## 2. Configure analog evolution and observations

The analog Hamiltonian is a uniform static $Z$ detuning. We sample the mean
transverse magnetization during each half of the experiment and request a $Z$
readout after the final digital analysis operation.

```{code-cell} ipython3
from mqt.yaqs import AnalogSimParams, DigitalSimParams, Hamiltonian, Observable

half_duration = 0.4
detuning = 1.1
transverse_observables = [Observable("x", site) for site in range(length)]
readout_observables = [Observable("z", site) for site in range(length)]


def evolution_parameters() -> AnalogSimParams:
    return AnalogSimParams(
        observables=transverse_observables,
        elapsed_time=half_duration,
        dt=0.05,
        max_bond_dim=8,
        svd_threshold=1e-12,
        order=2,
        sample_timesteps=True,
    )


def detuning_hamiltonian(*, coupling: float = 0.0) -> Hamiltonian:
    return Hamiltonian.heisenberg(
        length,
        Jx=0.0,
        Jy=0.0,
        Jz=coupling,
        h=detuning,
    )
```

## 3. Compose free-induction and echo programs

Both programs use the same preparation, analog Hamiltonian, evolution time,
and readout. Their only physical difference is whether the echo and frame
correction pulses are present.

```{code-cell} ipython3
from mqt.yaqs import AnalogSegment, DigitalSegment, SimulationProgram


def coherence_program(*, include_echo: bool, coupling: float = 0.0) -> SimulationProgram:
    hamiltonian = detuning_hamiltonian(coupling=coupling)
    segments = [
        DigitalSegment(preparation),
        AnalogSegment(hamiltonian, sim_params=evolution_parameters()),
    ]
    if include_echo:
        segments.append(DigitalSegment(echo_pulse))
    segments.append(AnalogSegment(hamiltonian, sim_params=evolution_parameters()))
    if include_echo:
        segments.append(DigitalSegment(echo_pulse))
    segments.append(
        DigitalSegment(
            analysis,
            sim_params=DigitalSimParams(observables=readout_observables),
        )
    )
    return SimulationProgram(segments, get_state=True)
```

The final correction deliberately reuses the same digital operation as the
central pulse. For $U_\delta(\tau)=\exp(-i\tau\delta Z)$,

$$
X U_\delta(\tau) X U_\delta(\tau) = I,
$$

because $XZX=-Z$. The cancellation holds for every static detuning $\delta$.

## 4. Run the experiment

We run the refocused program, its no-pulse control, and an interacting control.
The $ZZ$ interaction in the final program is not reversed by a global $X$
pulse, because $(X\otimes X)(Z\otimes Z)(X\otimes X)=Z\otimes Z$.

```{code-cell} ipython3
import numpy as np

from mqt.yaqs import Simulator, State

simulator = Simulator(parallel=False, show_progress=False)
initial_state = State(length, initial="zeros")
initial_vector = initial_state.mps.to_vec().copy()

echo_result = simulator.run(
    initial_state,
    coherence_program(include_echo=True),
)
no_pulse_result = simulator.run(
    initial_state,
    coherence_program(include_echo=False),
)
interacting_echo_result = simulator.run(
    initial_state,
    coherence_program(include_echo=True, coupling=0.7),
)
```

The runtime always propagates intermediate states, independently of whether an
individual segment requests `get_state`. The program-level flag retains the
final state as `result.output_state`; setting it to `False` would hide that
state without interrupting handoff.

The complete noninteracting echo returns the initial state and produces unit
$Z$ readout after the analysis gates:

```{code-cell} ipython3
echo_final_vector = echo_result.output_state.mps.to_vec()
echo_fidelity = abs(np.vdot(initial_vector, echo_final_vector)) ** 2
echo_readout = np.mean(
    [values[-1].real for values in echo_result.segment_results[-1].expectation_values]
)

np.testing.assert_allclose(echo_fidelity, 1.0, atol=1e-10)
np.testing.assert_allclose(echo_readout, 1.0, atol=1e-10)
```

## 5. Compare transverse coherence

Results remain in program order at `result.segment_results[i]`. Each analog
result has a local time axis beginning at zero and a `time_offset` locating it
within the full sequence. Digital operations are instantaneous in this model
and therefore advance no physical time.

```{code-cell} ipython3
def transverse_magnetization_trace(program_result):
    analog_results = [
        segment for segment in program_result.segment_results if segment.segment_type == "analog"
    ]
    times = np.concatenate(
        [
            analog_results[0].times + analog_results[0].time_offset,
            analog_results[1].times[1:] + analog_results[1].time_offset,
        ]
    )
    magnetization = np.concatenate(
        [
            np.mean(np.asarray(analog_results[0].expectation_values).real, axis=0),
            np.mean(np.asarray(analog_results[1].expectation_values).real, axis=0)[1:],
        ]
    )
    return times, magnetization
```

The first sample of the second analog segment duplicates the preceding physical
time. The helper removes that duplicate when constructing a display trace;
the segment-scoped results remain the authoritative, lossless output.

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
for program_result, label in [
    (echo_result, "static detuning + echo"),
    (no_pulse_result, "static detuning, no pulse"),
    (interacting_echo_result, r"detuning + $ZZ$ + echo"),
]:
    times, magnetization = transverse_magnetization_trace(program_result)
    ax.plot(times, magnetization, label=label)

ax.axvline(half_duration, color="black", linestyle=":", label="echo time")
ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
ax.set(xlabel="Program time", ylabel=r"Mean transverse magnetization $\langle X\rangle$")
ax.legend()
plt.show()
```

Only the noninteracting echo returns to unit transverse magnetization. The
no-pulse trace continues its detuning-induced rotation, while the interacting
trace shows dynamics that the global pulse cannot refocus.

## 6. From static detuning to correlated noise

This noiseless example is also the template for a noisy Hahn-echo experiment.
A quasi-static noise model should sample one detuning per trajectory and retain
that same realization across both analog segments. The echo then cancels every
trajectory before ensemble averaging, so increasing the trajectory count makes
the estimated operation converge to identity.

That behavior is different from ordinary Markovian dephasing. An ideal echo can
refocus coherent or sufficiently slow correlated $Z$ fluctuations, but it
cannot reverse an irreversible Markovian decay envelope. Sampling error also
does not change the underlying refocusing; more trajectories only improve the
estimate.

Noisy whole-program trajectories are not yet supported by this interface. When
they are added, noise history and random-number state must remain continuous
across segment boundaries. Resampling the detuning independently for the two
halves would describe a different physical process and would destroy the exact
cancellation.

## Current boundaries

Composable programs currently execute noiseless, static-Hamiltonian segments
on qubit MPS states. Noise across a complete trajectory, heterogeneous local
dimensions, compact long schedules, and parameter-scheduled Hamiltonians are
not yet supported. The segment specification and private compiled execution
path allow those capabilities to be added without changing the ordered user
workflow shown above.

## Related topics

- {doc}`analog_simulation` — standalone noisy analog evolution
- {doc}`circuit_observables` — standalone digital circuit simulation
- {doc}`simulation_parameters` — simulation accuracy and output controls
