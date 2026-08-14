# Circuit trajectory noise

Digital circuit trajectories use the same standard
{class}`~mqt.yaqs.core.data_structures.noise_model.NoiseModel` as other YAQS
simulations. Each process supplies its support, Lindblad strength, and jump
operator. YAQS resolves built-in process names to an operator `matrix` (or to
per-site `factors` for supported long-range processes).

After each ideal circuit gate, YAQS selects the processes whose complete support
is contained in the sites touched by that gate. Their resolved operators become
internal gate-local trajectory operations:

1. apply the ideal circuit gate;
2. apply the existing TJM no-jump evolution for the relevant processes;
3. sample and apply a jump operator using the trajectory-local random number
   generator;
4. continue with the next ideal gate.

This rule applies after both one-qubit and multi-qubit gates. A one-site process
can therefore be relevant after any gate touching that site. A two-site process
is relevant only when the gate touches both process sites; idle and spectator
sites do not receive unrelated noise.

## Standard API

Construct and pass a normal `NoiseModel`; no circuit-specific noise-model type
is required:

```python
from qiskit.circuit import QuantumCircuit

from mqt.yaqs import DigitalSimParams, NoiseModel, Observable, Simulator, State

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)

noise_model = NoiseModel(
    processes=[
        {"name": "pauli_z", "sites": [0], "strength": 0.02},
        {"name": "lowering", "sites": [1], "strength": 0.05},
    ]
)

sim_params = DigitalSimParams(
    observables=[Observable("z", 0), Observable("z", 1)],
    num_traj=32,
    random_seed=7,
)

result = Simulator(show_progress=False).run(
    State(2, initial="zeros"),
    circuit,
    sim_params,
    noise_model=noise_model,
)
```

`DigitalSimParams.random_seed` deterministically derives one generator for each
trajectory. That generator is reused for every gate-local noise step in the
trajectory.

## Strength semantics

`strength` keeps its standard YAQS meaning: it is the nonnegative Lindblad rate
$\gamma$ multiplying the process jump operator. A digital gate-local noise step
uses the existing TJM circuit convention of a unit timestep. The no-jump
exponential, state-dependent total jump probability, categorical jump weights,
and post-jump normalization are therefore exactly those already used by YAQS.
YAQS does not reinterpret `strength` as a Bernoulli probability and does not
perform a separate $\gamma\mapsto p$ conversion.

Both unitary operators such as Pauli matrices and non-unitary operators such as
`lowering` use this same infrastructure; the process matrix determines the
operation.

## Related topics

- {doc}`realistic_noise_models` — built-in and custom noise processes
- {doc}`circuit_observables` — circuit observables and trajectory settings
- {doc}`simulation_parameters` — reproducible seeds and accuracy presets
