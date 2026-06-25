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

**Operational memory** quantifies how much history an open quantum process retains at a temporal cut `c`.
YAQS estimates bond entropy `S_V(c)`, operational rank, and the memory-matrix singular spectrum from **split-cut probes**.

## Typical workflow

1. **Train** a surrogate for fast dynamics: `model = mc.train(ham, params, k=...)`.
2. **Predict** site-0 reduced states with the surrogate (production) or a reference comb (small-`k` gold).
3. **Characterize** operational memory with the Hamiltonian (primary metric) or the same quantity via surrogate/comb backends.

```python
from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer

mc = MemoryCharacterizer(representation="auto", parallel=True)

model = mc.train(ham, params, k=4, n=500)
rho = mc.predict(model, rho0, sequence, k=4)

memory = mc.characterize(ham, params, k=4, cut=2)  # primary metric
surrogate_memory = mc.characterize(model, cut=2, k=4)  # same quantity, surrogate backend

comb = mc.build_comb(ham, params, timesteps=[0.1] * 4, return_type="dense")
rho_ref = mc.predict(comb, rho0, sequence, k=4)  # gold dynamics (small k)
```

## Verb × backend

|                    | **predict**                       | **characterize**                    |
| ------------------ | --------------------------------- | ----------------------------------- |
| **Surrogate**      | Primary production dynamics       | Same `S_V(c)` via surrogate process |
| **Hamiltonian**    | —                                 | Primary memory metric               |
| **Reference comb** | Primary gold dynamics (small `k`) | Optional gold metric                |

## Setup

```{code-cell} ipython3
from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer

ham = Hamiltonian.ising(length=1, J=1.0, g=0.5)
params = AnalogSimParams(dt=0.1, max_bond_dim=12, order=1)
mc = MemoryCharacterizer(parallel=False, show_progress=False)
```

## Predict with a trained surrogate

```{code-cell} ipython3
---
tags: [remove-output]
---
import numpy as np

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    model = mc.train(
        ham,
        params,
        k=1,
        n=12,
        interventions="measure_prepare",
        train_kwargs={"epochs": 2, "batch_size": 4},
        model_kwargs={"d_model": 32, "nhead": 2, "num_layers": 1, "dim_ff": 64},
    )
    rho0 = np.eye(2, dtype=np.complex128) / 2.0
    rho_out = mc.predict(model, rho0, "measure_prepare", k=1)
    print(f"trace(rho) = {np.trace(rho_out).real:.4f}")
else:
    print("torch not installed; skip surrogate path in doc build")
```

Training fixes the rollout horizon `k` on the model. At inference, `mc.predict(model, ..., k=k_prime)` may use a shorter or longer sequence; accuracy is best at the trained horizon and degrades when extrapolating beyond it. See {doc}`process_tensor_surrogates` for architecture details.

## Predict with a reference comb

```{warning}
`build_comb` scales as `16^k`. Use only for gold dynamics at very small `k`.
```

```{code-cell} ipython3
---
tags: [remove-output]
---
comb = mc.build_comb(ham, params, timesteps=[0.1], return_type="dense", num_trajectories=20)
rho_ref = mc.predict(comb, rho0, "measure_prepare", k=1)
print(f"trace(rho_ref) = {np.trace(rho_ref).real:.4f}")
```

`rho0` is accepted for API symmetry but **not used** — the comb contracts from the tomographic reference state.

## Characterize with a Hamiltonian

```{code-cell} ipython3
---
tags: [remove-output]
---
memory = mc.characterize(ham, params, k=1, cut=1, n_pasts=6, n_futures=6)
print(memory.summary())
print(f"S_V = {memory.entropy(1):.4f} nats, rank = {memory.rank(1)}")
```

Use `preset="quick"`, `"balanced"`, or `"accurate"` for built-in probe-grid sizes, or override with `n_pasts` / `n_futures`.

## Characterize with a surrogate

`characterize(model, ...)` evaluates the **same** operational memory quantity at cut `c`, using the surrogate as the process backend. It is not a training-quality score.

```{code-cell} ipython3
---
tags: [remove-output]
---
if torch is not None:
    surrogate_memory = mc.characterize(model, cut=1, k=1, n_pasts=4, n_futures=4)
    print(surrogate_memory.summary())
else:
    print("torch not installed; skip surrogate characterize in doc build")
```

## Reference comb characterize (optional)

```{code-cell} ipython3
---
tags: [remove-output]
---
ref = mc.characterize(comb, cut=1, k=1, n_pasts=4, n_futures=4)
print(ref.summary())
```

## Reading `CharacterizationResult`

| Access                      | Meaning                                  |
| --------------------------- | ---------------------------------------- |
| `result.entropy(c)`         | Bond entropy `S_V(c)` in nats            |
| `result.rank(c)`            | Operational rank at cut `c`              |
| `result.singular_values(c)` | Memory-matrix spectrum at cut `c`        |
| `result.memory_matrix(c)`   | Past-row-centered weighted memory matrix |
| `result.summary()`          | Human-readable entropy/rank table        |

## Representation

`MemoryCharacterizer(representation="auto")` mirrors `Simulator`: `"vector"` uses MCWF, `"mps"` uses TJM.
With `"auto"`, vector is chosen when `hamiltonian.length <= vector_max_qubits` (default 10).

## Related topics

- {doc}`process_tensor_surrogates` — advanced surrogate training and Transformer structure
- {doc}`reference_exact_combs` — reference comb predict and validation
- {doc}`operational_memory` — V-matrix theory (advanced)
