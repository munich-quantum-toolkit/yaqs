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

# Reference Exact Combs (small `k`)

```{warning}
This page is for **reference validation only**. Exhaustive tomography scales as ``16^k``.
For production memory metrics, start with {doc}`characterization` (surrogate or Hamiltonian characterize paths).
```

```{note}
For walkthroughs of operational memory workflows, see {doc}`characterization`.
```

Reference comb construction reconstructs the full multi-time comb matrix :math:`\Upsilon` by
simulating every discrete intervention sequence. Use it to validate surrogate or exact-probe
results at very small `k`.

## 1. Define the system

```{code-cell} ipython3
from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer

hamiltonian = Hamiltonian.ising(length=2, J=1.0, g=0.5)
sim_params = AnalogSimParams(dt=0.1, max_bond_dim=16, order=1)
mc = MemoryCharacterizer(parallel=False, show_progress=False)
```

## 2. Dense comb

```{code-cell} ipython3
---
tags: [remove-output]
---
comb_single = mc.build_comb(
    hamiltonian,
    sim_params,
    timesteps=[0.1],
    num_trajectories=60,
    return_type="dense",
)
print(f"Comb Choi matrix shape: {comb_single.to_matrix().shape}")
```

## 3. MPO comb

```{code-cell} ipython3
---
tags: [remove-output]
---
comb_mpo = mc.build_comb(
    hamiltonian,
    sim_params,
    timesteps=[0.1],
    num_trajectories=60,
    return_type="mpo",
    compress_every=1,
)
print(type(comb_mpo).__name__, comb_mpo.to_dense().to_matrix().shape)
```

## 4. Probe the reference comb

```{code-cell} ipython3
---
tags: [remove-output]
---
result = mc.characterize(comb_single, cut=1, k=1, n_pasts=6, n_futures=6)
print(result.summary())
```

## Related topics

- {doc}`characterization` — primary memory characterization guide
- {doc}`operational_memory` — V-matrix theory
