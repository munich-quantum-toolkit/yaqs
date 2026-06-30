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

# Operational Memory Characterization

**Operational memory** quantifies how much history a black-box quantum process retains across a temporal cut `c`.
YAQS estimates the cross-cut memory entropy `S_V(c)`, an effective mode number, and the singular spectrum of the centered response matrix from **split-cut probes**.

## Typical workflow

1. **Train** a surrogate for fast dynamics: `model = mc.train(ham, params, num_interventions=...)`.
2. **Predict** site-0 reduced states with the surrogate (production) or a reference process tensor (small `num_interventions` validation).
3. **Characterize** operational memory with the Hamiltonian (primary metric) or the same quantity via surrogate/process-tensor backends.

```python
from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer

mc = MemoryCharacterizer(representation="auto", parallel=True)

num_interventions = 4
cut = 2  # causal break after step cut-1; future probes use steps cut+1 … num_interventions

model = mc.train(ham, params, num_interventions=num_interventions, n=500)
rho = mc.predict(model, rho0, sequence, num_interventions=num_interventions)

memory = mc.characterize(ham, params, cut=cut, num_interventions=num_interventions)  # primary metric
surrogate_memory = mc.characterize(model, cut=cut, num_interventions=num_interventions)

pt = mc.build_process_tensor(ham, params, timesteps=[0.1] * (num_interventions + 1), return_type="dense")
rho_ref = mc.predict(pt, rho0, sequence, num_interventions=num_interventions)
```

## Verb × backend

|                    | **predict**                            | **characterize**                    |
| ------------------ | -------------------------------------- | ----------------------------------- |
| **Surrogate**      | Primary production dynamics            | Same `S_V(c)` via surrogate process |
| **Hamiltonian**    | —                                      | Primary memory metric               |
| **Reference PT**   | Primary reference dynamics (small `num_interventions`) | Optional reference metric |

## Setup

```{code-cell} ipython3
import numpy as np

from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer

ham = Hamiltonian.ising(length=1, J=1.0, g=0.5)
params = AnalogSimParams(dt=0.1, max_bond_dim=12, order=1)
mc = MemoryCharacterizer(parallel=False, show_progress=False)
rho0 = np.eye(2, dtype=np.complex128) / 2.0
```

## Predict with a trained surrogate

```{code-cell} ipython3
---
tags: [remove-output]
---
try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    model = mc.train(
        ham,
        params,
        num_interventions=1,
        n=12,
        train_kwargs={"epochs": 2, "batch_size": 4},
        model_kwargs={"d_model": 32, "nhead": 2, "num_layers": 1, "dim_ff": 64},
    )
    rho_out = mc.predict(model, rho0, "haar", num_interventions=1)
    print(f"trace(rho) = {np.trace(rho_out).real:.4f}")
else:
    print("torch not installed; skip surrogate path in doc build")
```

`ProcessTensorSurrogate` is a **causal transformer** conditioned on `rho0` and per-step intervention encodings: sinusoidal positional encoding and a causal attention mask ensure step `t` only sees steps `≤ t`; each step predicts the site-0 reduced density matrix.

Training fixes `num_interventions` on the model. At inference, `mc.predict(model, rho0, sequence, num_interventions=n_prime)` may use a shorter or longer sequence because the Transformer encoder is length-agnostic. Predictions are most accurate for `n_prime` up to the trained horizon; accuracy generally decreases when `n_prime` exceeds it.

For custom architecture knobs or direct dataset access, use `mc.sample(...)` and instantiate {class}`~mqt.yaqs.characterization.memory.backends.surrogates.model.ProcessTensorSurrogate` yourself; user-facing dynamics should still go through `mc.predict`.

### Intervention styles

`train`, `sample`, and `characterize` accept `intervention_style=` (default `"haar"`):

- **`"haar"`** (default): Haar-random unitaries on sequence legs; measure/prepare at the causal cut only (paper V-matrix / `experiments/` standard).
- **`"measure_prepare"`**: rank-1 measure–prepare CP maps on every leg — opt-in for tomography-aligned studies.
- **`"clifford"`**: uniformly random single-qubit Clifford gates on legs.

For `predict`, pass the same style string as the third argument, or an explicit per-leg intervention list.

## Predict with a reference process tensor

```{warning}
`build_process_tensor` scales as `16**num_interventions`. Use only for reference dynamics at very small horizons.
```

```{code-cell} ipython3
---
tags: [remove-output]
---
pt = mc.build_process_tensor(ham, params, timesteps=[0.1, 0.1], return_type="dense")
rho_ref = mc.predict(pt, rho0, "haar", num_interventions=1)
print(f"trace(rho_ref) = {np.trace(rho_ref).real:.4f}")
```

`rho0` must match `pt.initial_rho` (site-0 reference after `U_0` from `|0⟩^⊗L`). In the noiseless path, `num_trajectories` is ignored (set `noise_model` to enable stochastic trajectories).

## Characterize with a Hamiltonian

```{code-cell} ipython3
---
tags: [remove-output]
---
memory = mc.characterize(ham, params, cut=1, num_interventions=2, n_pasts=6, n_futures=6)
print(memory.summary())
print(f"S_V = {memory.entropy(1):.4f}, R = {memory.modes(1):.3f}")
```

Use `preset="quick"`, `"balanced"`, or `"accurate"` for built-in probe-grid sizes, or override with `n_pasts` / `n_futures`.

### Reset delay at the causal break

Pass `delay=N` to insert `N` soft-reset slots `(|0>, |0>)` at the causal cut.
Each slot projects the probe site onto `|0>` and reprepares `|0>` while the rest of
the chain (the environment) continues evolving — useful for studying how long a reset
bridge must be before future probes lose sensitivity to the past.

`num_interventions` and `cut` are unchanged from the standard split-cut geometry; the physical sequence
length becomes `num_interventions + delay + 1`. Reuse the same `probe_set` when sweeping `delay`.

```{code-cell} ipython3
---
tags: [remove-output]
---
ham_delay = Hamiltonian.ising(length=6, J=1.0, g=1.0)
params_delay = AnalogSimParams(dt=0.1)
mc_delay = MemoryCharacterizer(parallel=False, show_progress=False)
anchor = mc_delay.characterize(
    ham_delay,
    params_delay,
    num_interventions=6,
    cut=4,
    delay=0,
    n_pasts=6,
    n_futures=6,
    rng=np.random.default_rng(999_991),
)
for delay in (0, 1, 2):
    result = mc_delay.characterize(
        ham_delay,
        params_delay,
        num_interventions=6,
        cut=4,
        delay=delay,
        probe_set=anchor,
    )
    print(f"delay={delay}  S_V={result.entropy(4):.4f}")
```

`delay > 0` is supported for Hamiltonian (exact) characterize only.

## Characterize with a surrogate

`characterize(model, ...)` evaluates the **same** operational memory quantity at cut `c`, using the surrogate as the process backend. It is not a training-quality score.

```{code-cell} ipython3
---
tags: [remove-output]
---
if torch is not None:
    surrogate_memory = mc.characterize(model, cut=1, num_interventions=1, n_pasts=4, n_futures=4)
    print(surrogate_memory.summary())
else:
    print("torch not installed; skip surrogate characterize in doc build")
```

## Reference process-tensor characterize (optional)

```{code-cell} ipython3
---
tags: [remove-output]
---
ref = mc.characterize(pt, cut=1, num_interventions=1, n_pasts=4, n_futures=4)
print(ref.summary())
```

## Reading `CharacterizationResult`

| Access                      | Meaning                                                          |
| --------------------------- | ---------------------------------------------------------------- |
| `result.entropy(c)`         | Cross-cut memory entropy `S_V(c)` (natural log of mode weights)  |
| `result.modes(c)`            | Effective mode number `R(c) = exp(S_V(c))`                       |
| `result.singular_values(c)` | Singular spectrum of the centered response matrix at cut `c`     |
| `result.memory_matrix(c)`   | Centered response matrix :math:`\widetilde{V}(c)`                |
| `result.probes(c)`          | Probe feature arrays used at cut `c` (for logging or inspection) |
| `result.summary()`          | Human-readable entropy/modes table                                |

## Representation

`MemoryCharacterizer(representation="auto")` mirrors `Simulator`: `"vector"` uses MCWF, `"mps"` uses TJM.
With `"auto"`, vector is chosen when `hamiltonian.length <= vector_max_qubits` (default 10).

## Theory: response matrix construction

Operational memory quantifies how many independent ways the conditioned past remains visible in accessible future responses across a temporal cut `c`.

Following the black-box split-cut construction:

1. Sample past interventions :math:`\alpha=(U_1,\ldots,U_{c-1})` and future interventions :math:`\beta=(V_{c+1},\ldots,V_k)`.
2. Insert a **causal break** at step `c`: measure an effect :math:`E_m` on the past side and prepare :math:`\sigma_p` on the future side.
3. For each conditioned past :math:`(\alpha,m)` and future setting :math:`(p,\beta)`, record the break weight :math:`w_{\alpha,m}` and the output response :math:`\mathbf{r}(\rho_{\mathrm{out}})` — Pauli tomography :math:`(\langle I\rangle,\langle X\rangle,\langle Y\rangle,\langle Z\rangle)` with :math:`\langle I\rangle=1` for physical states. The cross-cut matrix :math:`\widetilde{V}(c)` uses the :math:`X,Y,Z` components only.
4. Assemble the weighted response tensor, **center** over the past index to remove the Markovian background, reshape to :math:`\widetilde{V}(c)`, and take the entropy :math:`S_V(c)` of the normalized singular-value weights. Report :math:`R(c)=\exp(S_V(c))` via `result.modes(c)`.

### Break weights :math:`w_{\alpha,m}`

| Backend                        | How `weights_ij` are obtained                                                                                                  |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| **Hamiltonian**                | Product of simulated intervention probabilities along each probe sequence through cut `c` (traced full-state simulation).      |
| **Process tensor / surrogate** | Analytic product through cut `c` from the site-0 :math:`\vert 0\rangle\langle 0\vert` reference path (branch-weight path). |

In all cases the weight depends only on the conditioned past row `i` (constant across future columns `j` for a fixed probe grid).

### Same probes across backends

Run one backend first, then pass that result to `characterize(..., probe_set=...)` so both use the same past/future ensemble. Inspect sampled arrays with `result.probes(cut)`.

```{code-cell} ipython3
---
tags: [remove-output]
---
cut, num_interventions = 2, 2
pt_compare = mc.build_process_tensor(
    ham,
    params,
    timesteps=[0.1, 0.1, 0.1],
    return_type="dense",
)
ham_result = mc.characterize(ham, params, cut=cut, num_interventions=num_interventions, n_pasts=6, n_futures=5)
pt_result = mc.characterize(pt_compare, cut=cut, num_interventions=num_interventions, probe_set=ham_result)
print(f"Hamiltonian S_V = {ham_result.entropy(cut):.4f}")
print(f"PT S_V       = {pt_result.entropy(cut):.4f}")
```

When comparing Hamiltonian and process-tensor backends, align `cut`, `num_interventions`, and the probe grid via `probe_set`.

### Spectrum and cut sweep

```{code-cell} ipython3
---
tags: [remove-output]
---
import matplotlib.pyplot as plt

pt_sweep = mc.build_process_tensor(
    ham,
    params,
    timesteps=[0.1, 0.1, 0.1],
    return_type="dense",
)
num_interventions = 2
sv = mc.characterize(
    pt_sweep,
    cut=2,
    num_interventions=num_interventions,
    n_pasts=8,
    n_futures=8,
    rng=np.random.default_rng(1),
).singular_values(2)

fig, ax = plt.subplots(figsize=(5, 3))
ax.semilogy(sv, "o-")
ax.set_xlabel("mode index")
ax.set_ylabel("singular value")
ax.set_title("Spectrum of Ṽ(c) at cut c=2")
fig.tight_layout()
```

## Internal package layout (developers)

User code should call :class:`~mqt.yaqs.memory_characterizer.MemoryCharacterizer` only. Lower-level modules live under `mqt.yaqs.characterization.memory`:

| Path                   | Role                                                                                                |
| ---------------------- | --------------------------------------------------------------------------------------------------- |
| `operational_memory/`  | Split-cut protocol: `sample_probes`, `assemble_probe_grid`, `run_operational_memory`                |
| `shared/`              | Choi/rho encoding, intervention-step helpers, site-0 MCWF/TJM utilities                             |
| `backends/exact.py`    | :class:`~mqt.yaqs.characterization.memory.backends.exact.ExactBackend` for Hamiltonian characterize |
| `backends/tomography/` | Reference dense/MPO process tensors                                                                 |
| `backends/sequences/`  | `simulate_sequences`, pool workers for the unified process-tensor schedule                          |
| `backends/surrogates/` | `ProcessTensorSurrogate`, `SeqTrace` training traces, `sample_train_dataset`                        |

Verb-first naming is used throughout (`compute_*`, `assemble_*`, `simulate_*`, `encode_*`).

## Related topics

- {doc}`reference_process_tensors` — dense/MPO reference process tensors, predict validation, QMI/CMI
