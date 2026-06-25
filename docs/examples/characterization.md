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
YAQS estimates bond entropy `S_V(c)`, operational rank, and the V-matrix singular spectrum from **split-cut probes**.

## Mental model

1. **Build** artifacts when needed — `train()` (surrogate), `build_comb()` (reference comb).
2. **Probe** to get V-matrix diagnostics — every `probe*` method returns the same
   `ProbeResult` with `.entropy(c)`, `.rank(c)`, `.singular_values(c)`.

```python
from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer

mc = MemoryCharacterizer(parallel=True)

# Path A — exact simulator (no torch)
exact = mc.probe_exact(ham, params, cut=2, k=2, n_pasts=16, n_futures=16)

# Path B — train separately, then probe; or use characterize() shortcut
model = mc.train(ham, params, k=4, n=80)
result = mc.probe(model, cut=2, k=4, n_pasts=16, n_futures=16)
# result = mc.characterize(ham, params, k=4, n=80)  # train + probe in one call

# Path C — from pre-computed responses
rebuilt = MemoryCharacterizer.probe_from_responses(pauli_xyz, weights, probe_set, cut=1)

# Path D — build comb separately, then probe
comb = mc.build_comb(ham, params, timesteps=[0.1], return_type="dense")
ref = mc.probe(comb, cut=1, k=1, n_pasts=16, n_futures=16)
```

## The three `probe*` methods

All three assemble the **same weighted V matrix** and return a **`ProbeResult`**. They differ only in **where probe responses come from**:

| Method | Responses from | Typical use |
|--------|------------------|-------------|
| `probe_exact(ham, params, …)` | Full **simulator** rollouts | Ground-truth `S_V`; validate surrogates |
| `probe(target, …)` | A **model you built** (`train` / `build_comb`) | Surrogate or reference comb probing |
| `probe_from_responses(…)` | **Arrays you already have** | External data or re-analysis |

`train`, `sample`, and `build_comb` return models or combs — **not** diagnostics. Always finish with a `probe*` call (or use `characterize` / `characterize_comb` shortcuts).

## Setup

```{code-cell} ipython3
from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer

ham = Hamiltonian.ising(length=1, J=1.0, g=0.5)
params = AnalogSimParams(dt=0.1, max_bond_dim=12, order=1)
mc = MemoryCharacterizer(parallel=False, show_progress=False)
```

## Path A — `probe_exact()`

```{code-cell} ipython3
---
tags: [remove-output]
---
exact = mc.probe_exact(ham, params, cut=1, k=1, n_pasts=6, n_futures=6)
print(exact.summary())
print(f"S_V = {exact.entropy(1):.4f} nats, rank = {exact.rank(1)}")
```

## Path B — `train()` then `probe()`

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
        k=1,
        n=12,
        train_kwargs={"epochs": 2, "batch_size": 4},
        model_kwargs={"d_model": 32, "nhead": 2, "num_layers": 1, "dim_ff": 64},
    )
    result = mc.probe(model, cut=1, k=1, n_pasts=6, n_futures=6)
    print(result.summary())
else:
    print("torch not installed; skip surrogate path in doc build")
```

One-call shortcut: `mc.characterize(ham, params, k=1, n=12, …)` trains and probes in one step.

## Path C — `probe_from_responses()`

```{code-cell} ipython3
---
tags: [remove-output]
---
from mqt.yaqs.characterization.memory.diagnostics.probe import sample_split_cut_probes
import numpy as np

rng = np.random.default_rng(0)
probe_set = sample_split_cut_probes(cut=1, k=1, n_pasts=4, n_futures=4, rng=rng)
exact = mc.probe_exact(ham, params, cut=1, k=1, probe_set=probe_set)
cut = exact.by_cut[1]
assert cut.weights is not None
rebuilt = MemoryCharacterizer.probe_from_responses(cut.pauli_xyz_ij, cut.weights, probe_set, cut=1)
print(f"exact S_V={exact.entropy(1):.4f}, rebuilt S_V={rebuilt.entropy(1):.4f}")
```

## Path D — `build_comb()` then `probe()`

```{warning}
`build_comb` scales as `16^k`. Use only for validation at very small `k`.
```

```{code-cell} ipython3
---
tags: [remove-output]
---
comb = mc.build_comb(ham, params, timesteps=[0.1], return_type="dense", num_trajectories=20)
ref = mc.probe(comb, cut=1, k=1, n_pasts=4, n_futures=4)
print(ref.summary())
```

## Reading `ProbeResult`

| Access | Meaning |
|--------|---------|
| `result.entropy(c)` | Bond entropy `S_V(c)` in nats |
| `result.rank(c)` | Operational rank at cut `c` |
| `result.singular_values(c)` | Centered V spectrum at cut `c` |
| `result.by_cut[c]` | Per-cut detail (raw `V`, weights, probe grid, …) |
| `result.model` | Trained surrogate when using `characterize()` |

## Related topics

- {doc}`process_tensor_surrogates` — advanced surrogate training
- {doc}`operational_memory` — V-matrix theory
- {doc}`reference_exact_combs` — reference comb details
