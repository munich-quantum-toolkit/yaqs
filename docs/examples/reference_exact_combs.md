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
Exhaustive tomography scales as ``16^k``. For production dynamics use a trained surrogate; for small-``k`` gold dynamics and metrics, use the reference comb paths below.
```

```{note}
For the main characterization funnel, see {doc}`characterization`.
```

Reference comb construction reconstructs the full multi-time comb matrix :math:`\Upsilon` by
simulating every discrete intervention sequence. At very small `k`, use it for **gold dynamics**
(`mc.predict(comb, ...)`) and optional memory-metric cross-checks (`mc.characterize(comb, ...)`).

## 1. Define the system

```{code-cell} ipython3
from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer
import numpy as np

hamiltonian = Hamiltonian.ising(length=2, J=1.0, g=0.5)
sim_params = AnalogSimParams(dt=0.1, max_bond_dim=16, order=1)
mc = MemoryCharacterizer(parallel=False, show_progress=False)
rho0 = np.eye(2, dtype=np.complex128) / 2.0
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

## 4. Predict with the reference comb

`mc.predict(comb, rho0, sequence, k=...)` returns site-0 reduced-state dynamics from the tomographic comb.
`rho0` is accepted for API symmetry but **not used** (fixed product reference state).

```{code-cell} ipython3
---
tags: [remove-output]
---
rho_ref = mc.predict(comb_single, rho0, "measure_prepare", k=1)
print(f"trace(rho_ref) = {np.trace(rho_ref).real:.4f}")
```

## 5. Characterize (optional cross-check)

```{code-cell} ipython3
---
tags: [remove-output]
---
result = mc.characterize(comb_single, cut=1, k=1, n_pasts=6, n_futures=6)
print(result.summary())
```

## 6. Validation sketch

At small `k`, compare surrogate predictions against the reference comb:

```python
model = mc.train(hamiltonian, sim_params, k=1, n=40)
rho_sure = mc.predict(model, rho0, "measure_prepare", k=1)
rho_comb = mc.predict(comb_single, rho0, "measure_prepare", k=1)
# Compare rho_sure vs rho_comb (e.g. trace distance or fidelity)
```

## Related topics

- {doc}`characterization` — primary memory characterization guide
- {doc}`process_tensor_surrogates` — surrogate training
- {doc}`operational_memory` — V-matrix theory (advanced)
