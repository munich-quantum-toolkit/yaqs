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

## Mental model

1. **Predict** reduced-state dynamics under intervention sequences — primary production workflow.
2. **Characterize** V-matrix memory metrics when you need `S_V`, rank, or the memory matrix.
3. **Build** artifacts when needed — `train()` (surrogate) or `build_comb()` (reference comb at small `k`).

```python
from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer

mc = MemoryCharacterizer(representation="auto", parallel=True)

# Predict (surrogate — production)
model = mc.train(ham, params, k=4, n=80, interventions="measure_prepare")
rho = mc.predict(model, rho0, "measure_prepare", k=4)

# Characterize (memory metrics)
memory = mc.characterize(model, cut=2, preset="balanced", interventions="haar")

# Hamiltonian black-box metric (research; not tomographic ground truth)
metric = mc.characterize(ham, params, k=4, cut=2, interventions="haar")

# Reference comb (validation only; scales as 16^k)
comb = mc.build_comb(ham, params, timesteps=[0.1], return_type="dense")
ref = mc.characterize(comb, cut=1)
```

## Three backends

| Backend            | Build          | Characterize                       | Typical use                                   |
| ------------------ | -------------- | ---------------------------------- | --------------------------------------------- |
| **Surrogate**      | `train()`      | `characterize(model)`              | Production predictions and fast metrics       |
| **Hamiltonian**    | —              | `characterize(ham, params, k=...)` | Research V-matrix metric from full simulation |
| **Reference comb** | `build_comb()` | `characterize(comb)`               | Tomographic gold standard at very small `k`   |

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

Training defaults to `interventions="measure_prepare"`; characterization defaults to `interventions="haar"`.

## Characterize memory metrics

```{code-cell} ipython3
---
tags: [remove-output]
---
exact = mc.characterize(ham, params, k=1, cut=1, n_pasts=6, n_futures=6)
print(exact.summary())
print(f"S_V = {exact.entropy(1):.4f} nats, rank = {exact.rank(1)}")
```

Use `preset="quick"`, `"balanced"`, or `"accurate"` for built-in probe-grid sizes, or override with `n_pasts` / `n_futures`.

## Reference comb validation

```{warning}
`build_comb` scales as `16^k`. Use only for validation at very small `k`.
```

```{code-cell} ipython3
---
tags: [remove-output]
---
comb = mc.build_comb(ham, params, timesteps=[0.1], return_type="dense", num_trajectories=20)
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

- {doc}`process_tensor_surrogates` — advanced surrogate training
- {doc}`operational_memory` — V-matrix theory
- {doc}`reference_exact_combs` — reference comb details
