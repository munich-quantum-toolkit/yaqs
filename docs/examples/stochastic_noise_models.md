# Stochastic circuit noise models

YAQS provides two gate-local digital noise models used in the accompanying paper
experiments:

- {class}`~mqt.yaqs.XYZPauliNoiseModel` samples a local XYZ Pauli channel.
- {class}`~mqt.yaqs.XBasisDissipativeNoiseModel` samples a complete Kraus
  channel in the X basis.

Pass either model to {meth}`~mqt.yaqs.Simulator.run` for a standalone circuit
simulation. They are deliberately separate from {class}`~mqt.yaqs.NoiseModel`.
The latter stores continuous-time Lindblad rates $\gamma$ and is evolved using a
timestep; the models on this page store a stochastic gate-local parameter $p$.
YAQS never converts between $p$ and $\gamma$.

## XYZ Pauli model

For one qubit, {class}`~mqt.yaqs.XYZPauliNoiseModel` uses $p\in[0,1]$ as the
total local Pauli-error probability. It samples one mutually exclusive channel
member after an eligible gate:

$$
P(I)=1-p,\qquad
P(X)=P(Y)=P(Z)=\frac{p}{3}.
$$

At $p=0$, the model follows the exact ideal-circuit path. After a two-qubit
gate, each touched qubit receives an independent sample from the same local
channel.

```python
from mqt.yaqs import XYZPauliNoiseModel

pauli_noise = XYZPauliNoiseModel(p=0.02)
pauli_noise.probabilities
```

## X-basis dissipative channel

{class}`~mqt.yaqs.XBasisDissipativeNoiseModel` uses the X-basis damping
parameter $p\in[0,1]$ in the Kraus pair

$$
K_0=|+\rangle\langle+|+\sqrt{1-p}|-\rangle\langle-|,
\qquad
K_1=\sqrt{p}|+\rangle\langle-|.
$$

This maps $|-\rangle$ irreversibly toward $|+\rangle$. The channel is applied
after every supported gate to every touched qubit; $p$ parametrizes its Kraus
operators rather than an outer Bernoulli decision. Two-qubit gates therefore
receive two sequential, state-dependent Kraus samples in gate order. Each branch
is selected with $q_i=\lVert K_i|\psi\rangle\rVert^2$ and the selected
trajectory is normalized by $1/\sqrt{q_i}$. A trajectory-local NumPy generator
derived from `DigitalSimParams.random_seed` supplies every draw.

```python
import numpy as np
from qiskit.circuit import QuantumCircuit

from mqt.yaqs import (
    DigitalSimParams,
    Observable,
    Simulator,
    State,
    XBasisDissipativeNoiseModel,
)

circuit = QuantumCircuit(1)
circuit.h(0)

params = DigitalSimParams(
    observables=[Observable("x", 0), Observable("z", 0)],
    num_traj=16,
    random_seed=7,
    preset="exact",
)
noise = XBasisDissipativeNoiseModel(p=0.2)
result = Simulator(show_progress=False).run(State(1), circuit, params, noise)

[float(np.real(values[0])) for values in result.expectation_values]
```

Both stochastic models currently support standalone digital simulations with
one- and two-qubit gates. Analog evolution and multi-segment
{class}`~mqt.yaqs.SimulationProgram` execution continue to use
{class}`~mqt.yaqs.NoiseModel`.

## Related topics

- {doc}`circuit_observables` — circuit observables and trajectory settings
- {doc}`simulation_parameters` — reproducible seeds and accuracy presets
- {doc}`realistic_noise_models` — continuous Lindblad noise and static disorder
