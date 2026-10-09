---
file_format: mystnb
kernelspec:
  name: python3
language_info:
  name: python
mystnb:
  number_source_lines: true
  execution_timeout: 900
---

# Predicting Non-Markovian Dynamics

A control pulse changes a quantum system and its later interaction with the
environment. Predicting that response usually requires evolving the joint system
and environment again for each control sequence. A surrogate learns from
simulated sequences so that we can query new controls without repeating that
evolution.

Here we train on random controls, then predict how a chosen rotation changes the
final coherence of a probe qubit. We extend {doc}`quickstart` by comparing two
environment couplings and checking each prediction against direct Hamiltonian
evolution. Two qubits keep the reference calculation small; this example teaches
the workflow rather than demonstrating a speed advantage.

```{note}
**Experimental feature.** Surrogate modeling is not yet supported by a published
YAQS paper. Validate predictions for your controls and time horizon against
reference simulations or measurements. This example uses two interventions; it
does not establish reliable prediction for long control protocols.
```

Install the PyTorch extra with `uv pip install "mqt.yaqs[torch]"`. The example
also uses Matplotlib. Run the cells in order in a notebook; for a script, use
the entry-point guard in {doc}`simulator_initialization`.

## 1. Choose the system and control times

Site 0 is the probe, and site 1 is an unobserved environment qubit. Their
Hamiltonian is

$$
H=-JZ_0Z_1-g(X_0+X_1), \qquad g=0.5.
$$

The environment starts in $|0\rangle$. During training we vary the probe
preparation and apply random single-qubit rotations to the probe. The joint
state evolves between rotations, so the environment can retain information about
earlier controls.

```{code-cell} python
import numpy as np
import torch

from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer

num_steps = 2
interval = 0.6
schedule = [0.0, interval, interval]
couplings = [0.3, 1.0]
field = 0.5
hamiltonians = {coupling: Hamiltonian.ising(2, J=coupling, g=field) for coupling in couplings}
params = AnalogSimParams(elapsed_time=interval, dt=interval, preset="fast")
characterizer = MemoryCharacterizer(show_progress=False)
```

`timesteps` contains one more duration than there are interventions. The initial
`0.0` means that the first intervention occurs immediately after preparation.
Evolution for $0.6$ follows each intervention, giving a final time of $1.2$ in
units with $\hbar=1$. These durations become part of the training problem;
`predict` does not accept a new time grid.

We use the same schedule at weak coupling, $J=0.3$, and stronger coupling,
$J=1$. Each Hamiltonian needs its own model. Coupling strength is not an input
to the trained surrogate. The public training path also does not accept a
`NoiseModel`; this comparison changes environmental coupling rather than adding
a Lindblad noise channel.

## 2. Train on random control sequences

`sample` generates a validation dataset. `train` generates a separate training
dataset and fits the surrogate. Setting `intervention_style="haar"` draws random
single-qubit unitaries at both control times. The default initialization samples
a pure probe state from each random density matrix's eigenstates; the
environment remains in $|0\rangle$.

```{code-cell} python
models = {}
for coupling, hamiltonian in hamiltonians.items():
    torch.manual_seed(7)
    validation = characterizer.sample(
        hamiltonian, params, num_interventions=num_steps, n=256, seed=99,
        timesteps=schedule, intervention_style="haar",
    )
    models[coupling] = characterizer.train(
        hamiltonian, params, num_interventions=num_steps, n=4096, seed=7,
        timesteps=schedule, intervention_style="haar",
        model_kwargs={"d_model": 64, "num_layers": 2, "dim_ff": 128},
        train_kwargs={"epochs": 400, "lr": 1e-3, "device": "cpu", "val_dataset": validation},
    )
```

The seeds separate training and validation sequences. At the end of training,
YAQS restores the model with the lowest validation loss across the 400 epochs.
The validation set therefore selects the model; it is not an independent test of
the predictions below. The small architecture and CPU setting bound this
example's training cost. Other hardware, seeds, and PyTorch versions can give
different errors.

The resulting model learns a mapping from the initial probe state and control
sequence to reduced probe states. It does not reconstruct the environment's
state. Both the environment preparation and the two-intervention horizon stay
fixed throughout this example.

## 3. Predict the response to a chosen pulse

Prepare the probe in $|+\rangle=(|0\rangle+|1\rangle)/\sqrt{2}$. Apply the
identity at the first control time, let the joint system evolve for $0.6$, then
apply $R_z(\theta)$ to the probe. The surrogate predicts its state after the
second evolution interval. We sweep the angle while using the same model.

```{code-cell} python
plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
rho0 = np.outer(plus, plus.conj())
identity = np.eye(2, dtype=complex)
pulse_angles = np.linspace(0, 2 * np.pi, 61)
sequences = [
    [
        {"unitary": identity},
        {"unitary": np.diag(np.exp(-0.5j * angle * np.array([1, -1])))},
    ]
    for angle in pulse_angles
]
predictions = {
    coupling: np.stack([characterizer.predict(model, rho0, sequence) for sequence in sequences])
    for coupling, model in models.items()
}
```

`predict` returns a complex array of shape `(2, 2)`. Each stacked sweep has
shape `(61, 2, 2)`, with the first axis following `pulse_angles`. These chosen
sequences were not supplied during training; the model must generalize from its
random controls. At $\theta=0$ the sequence gives free evolution, and
$\theta=\pi$ gives a phase flip halfway through the evolution.

The following figure uses the stronger coupling. Its left panel projects the
final states onto the equatorial Bloch plane, with coordinates
$(\langle X\rangle,\langle Y\rangle)$. The right panel plots coherence
$C=2|\rho_{01}|$, which is the distance from the origin in that plane for a
physical qubit state. All points describe the same final time; the colored curve
is a control-angle sweep, not a trajectory through time.

```{code-cell} python
:tags: [hide-input]
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats

set_matplotlib_formats("svg")
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 10, "axes.labelsize": 11, "axes.linewidth": 0.7,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "legend.frameon": False, "figure.constrained_layout.use": True,
    "savefig.bbox": "tight", "svg.fonttype": "none",
})
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

predicted_states = predictions[1.0]
predicted_coherence = 2 * np.abs(predicted_states[:, 0, 1])
no_pulse_coherence = predicted_coherence[0]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1, 1.4]})

# Show the final states projected onto the equatorial Bloch plane.
bloch_xy = np.column_stack((
    2 * predicted_states[:, 0, 1].real,
    -2 * predicted_states[:, 0, 1].imag,
))
points = bloch_xy[:, None, :]
segments = np.concatenate((points[:-1], points[1:]), axis=1)
trajectory = LineCollection(segments, cmap="twilight_shifted",
                            norm=plt.Normalize(0, 2 * np.pi), linewidth=2.6)
trajectory.set_array((pulse_angles[:-1] + pulse_angles[1:]) / 2)
axes[0].add_patch(Circle((0, 0), 1, facecolor="0.97", edgecolor="0.75", linewidth=0.8))
axes[0].add_patch(Circle((0, 0), 0.5, fill=False, edgecolor="0.85", linewidth=0.6))
axes[0].axhline(0, color="0.85", linewidth=0.6)
axes[0].axvline(0, color="0.85", linewidth=0.6)
axes[0].add_collection(trajectory)
axes[0].plot(*bloch_xy[0], "o", color="0.3", markerfacecolor="white", markersize=6)
axes[0].set(xlabel=r"$\langle X\rangle$", ylabel=r"$\langle Y\rangle$",
            xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), aspect="equal",
            xticks=[-1, 0, 1], yticks=[-1, 0, 1])
axes[0].set_title("(a) Final probe state", loc="left", fontsize=11)
colorbar = fig.colorbar(trajectory, ax=axes[0], orientation="horizontal",
                       shrink=0.8, pad=0.08, aspect=25, ticks=[0, np.pi, 2 * np.pi])
colorbar.ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
colorbar.set_label(r"Pulse angle $\theta$")

axes[1].fill_between(pulse_angles, no_pulse_coherence, predicted_coherence,
                     where=predicted_coherence >= no_pulse_coherence,
                     interpolate=True, color="#0072B2", alpha=0.15)
axes[1].fill_between(pulse_angles, no_pulse_coherence, predicted_coherence,
                     where=predicted_coherence < no_pulse_coherence,
                     interpolate=True, color="#D55E00", alpha=0.2)
axes[1].plot(pulse_angles, predicted_coherence, color="#0072B2", linewidth=2.2,
             label="With control pulse")
axes[1].axhline(no_pulse_coherence, color="0.4", linestyle="--", linewidth=1.1,
                label="Free evolution")
axes[1].set(xlabel=r"Pulse angle $\theta$", ylabel=r"Final coherence $2|\rho_{01}|$",
            xlim=(0, 2 * np.pi), ylim=(0, 1),
            xticks=[0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
            xticklabels=["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
axes[1].set_title("(b) Predicted coherence", loc="left", fontsize=11)
axes[1].legend(loc="upper right", fontsize=9)
plt.show()
```

**A control pulse changes the final coherence.** The open circle marks free
evolution. Shading shows changes relative to that prediction. An instantaneous
$Z$ rotation preserves coherence magnitude at the moment it is applied; the
differences here arise during the subsequent joint evolution. A prediction alone
does not show whether the model has learned that response accurately, so we next
compare it with a reference.

(short-horizon-validation)=

## 4. Check the predictions against joint evolution

For two qubits, we can build the Hamiltonian directly with NumPy and propagate
with SciPy's matrix exponential. This reference uses neither the surrogate nor
YAQS's evolution routines. With site 0 as the least significant bit, the joint
initial vector is $|0\rangle_{\mathrm{env}}\otimes|+\rangle_{\mathrm{probe}}$,
and a probe rotation acts as $I\otimes R_z(\theta)$.

```{code-cell} python
from scipy.linalg import expm

pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
pauli_z = np.diag([1.0, -1.0])
initial_joint = np.kron([1, 0], plus)
references = {}
for coupling in couplings:
    dense_hamiltonian = -coupling * np.kron(pauli_z, pauli_z) - field * (
        np.kron(identity, pauli_x) + np.kron(pauli_x, identity)
    )
    evolution = expm(-1j * interval * dense_hamiltonian)
    states = []
    for sequence in sequences:
        pulse = sequence[1]["unitary"]
        joint = evolution @ np.kron(identity, pulse) @ evolution @ initial_joint
        amplitudes = joint.reshape(2, 2)
        states.append(amplitudes.T @ amplitudes.conj())
    references[coupling] = np.stack(states)
```

Tracing out the environment gives the reference probe density matrix. Compare
the full matrix as well as the plotted coherence. We use half the trace norm of
the matrix difference, which equals trace distance when both matrices are
normalized physical states.

```{code-cell} python
matrix_errors = {}
for coupling in couplings:
    predicted = predictions[coupling]
    reference = references[coupling]
    matrix_errors[coupling] = 0.5 * np.sum(np.abs(np.linalg.eigvalsh(predicted - reference)), axis=1)
    trace_error = np.max(np.abs(np.trace(predicted, axis1=1, axis2=2) - 1))
    hermiticity_error = np.max(np.abs(predicted - predicted.conj().swapaxes(1, 2)))
    minimum_eigenvalue = np.min(np.linalg.eigvalsh(predicted))
    coherence_rmse = np.sqrt(np.mean((2 * np.abs(predicted[:, 0, 1]) - 2 * np.abs(reference[:, 0, 1])) ** 2))
    print(
        f"J={coupling:g}: max matrix error={matrix_errors[coupling].max():.4f}, "
        f"coherence RMSE={coherence_rmse:.4f}\n"
        f"  max trace error={trace_error:.4f}, "
        f"Hermiticity error={hermiticity_error:.1e}, min eigenvalue={minimum_eigenvalue:.4f}"
    )
```

The public API makes each returned estimate Hermitian, so a zero Hermiticity
error is expected. It does not enforce unit trace or positivity. The printed
checks expose normalization error and any negative eigenvalues; we do not
renormalize the predictions, clip their eigenvalues, or project them onto
physical states. A positive minimum eigenvalue alone is insufficient when the
trace differs from one.

## 5. Compare environmental couplings

The second model asks whether the same control has a different effect when the
probe couples more weakly to its environment. Plot both coherence sweeps with
their independent references, then show where the full predicted matrices differ
from those references.

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
colors = ["#D55E00", "#0072B2"]
for coupling, color in zip(couplings, colors, strict=True):
    coherence = 2 * np.abs(predictions[coupling][:, 0, 1])
    exact_coherence = 2 * np.abs(references[coupling][:, 0, 1])
    axes[0].plot(pulse_angles, coherence, color=color, linewidth=2, label=rf"Surrogate, $J={coupling:g}$")
    axes[0].plot(pulse_angles[::5], exact_coherence[::5], "o", color=color,
                 markerfacecolor="white", markersize=4, markeredgewidth=1)
    axes[1].plot(pulse_angles, matrix_errors[coupling], color=color, linewidth=2, label=rf"$J={coupling:g}$")
axes[0].plot([], [], "o", color="0.3", markerfacecolor="white", markersize=4, label="Joint evolution")
axes[0].set(ylabel=r"Final coherence $2|\rho_{01}|$")
axes[1].set(ylabel=r"Matrix error $\frac{1}{2}\|\rho_{\rm pred}-\rho_{\rm ref}\|_1$")
for ax in axes:
    ax.set(xlabel=r"Pulse angle $\theta$", xlim=(0, 2 * np.pi),
           xticks=[0, np.pi, 2 * np.pi], xticklabels=["0", r"$\pi$", r"$2\pi$"])
    ax.legend(fontsize=8, loc="best")
axes[1].set_ylim(bottom=0)
axes[0].set_title("(a) Coupling changes the control response", loc="left", fontsize=10)
axes[1].set_title("(b) Error against joint evolution", loc="left", fontsize=10)
plt.show()
```

**The same pulse produces different responses at the two couplings.** Solid
curves show surrogate predictions; open circles show direct evolution. The error
panel and printed state checks bound what we can infer from those curves.
Accuracy on random validation sequences does not certify a chosen pulse family,
and this two-step comparison says nothing about longer sequences. Repeat the
reference checks when changing the Hamiltonian, environment preparation, control
family, or time horizon.

## Other supported options

`predict(model, rho0, sequence, return_sequence=True)` returns an array of shape
`(num_interventions, 2, 2)`. Its entries describe the probe after each
intervention and its following evolution interval, not a continuous time trace.
Predictions for earlier steps still use the trained schedule.

For other control families, `intervention_style` also accepts `"clifford"` and
`"measure_prepare"` when sampling or training. Prediction sequences accept
unitary dictionaries as above, or style strings that draw random controls. A
`"measure_prepare"` draw selects a rank-one measurement outcome and a
replacement state. Match the training controls to the intended queries and
validate other interventions separately; this example tests only unitary
controls. See {doc}`characterization` for memory diagnostics.

Short process tensors provide another reference through `build_process_tensor`
and the same `predict` call. They start from the joint all-zero state, and their
input must match `process_tensor.initial_rho`. They therefore do not directly
match the $|+\rangle$ preparation used here. The default uncapped MPO
construction grows as `16**num_interventions`; dense tomography also grows
exponentially. Use these references for short horizons and see
{doc}`characterization` for approximation limits, QMI, and CMI. Those
process-tensor diagnostics are distinct from prediction accuracy and from the
response-matrix memory spectrum.
