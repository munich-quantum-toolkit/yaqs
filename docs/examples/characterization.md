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

Training fixes `num_interventions` on the model. At inference, `mc.predict(model, rho0, sequence, num_interventions=n_prime)` may use a shorter or longer sequence because the Transformer encoder is length-agnostic. Predictions are most accurate for `n_prime` up to the trained horizon; accuracy generally decreases when `n_prime` exceeds it. See {doc}`process_tensor_surrogates` for architecture details.

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

## Developer modules

Lower-level split-cut helpers live under `mqt.yaqs.characterization.memory` (`operational_memory`,
`shared`, `backends`). See {doc}`operational_memory` for the internal layout and verb-first API names.

## Related topics

- {doc}`process_tensor_surrogates` — surrogate training and Transformer structure
- {doc}`reference_process_tensors` — reference process-tensor predict and validation
- {doc}`operational_memory` — construction details and theory (advanced)
