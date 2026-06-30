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

# Reference Exact Process Tensors (small `num_interventions`)

```{warning}
Exhaustive tomography scales as ``16**num_interventions``. For production dynamics use a trained surrogate; for small-horizon reference dynamics and metrics, use the reference process-tensor paths below.
```

```{note}
For the main characterization funnel, see {doc}`characterization`.
```

Reference process tensor construction reconstructs the full multi-time process tensor matrix :math:`\Upsilon` by
simulating every discrete intervention sequence. At very small `num_interventions`, use it for **reference dynamics**
(`mc.predict(pt, ...)`) and optional memory-metric cross-checks (`mc.characterize(pt, ...)`).

## 1. Define the system

```{code-cell} ipython3
from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer
import numpy as np

hamiltonian = Hamiltonian.ising(length=2, J=1.0, g=0.5)
sim_params = AnalogSimParams(dt=0.1, max_bond_dim=16, order=1)
mc = MemoryCharacterizer(parallel=False, show_progress=False)
rho0 = np.eye(2, dtype=np.complex128) / 2.0
```

## 2. Dense process tensor

```{code-cell} ipython3
---
tags: [remove-output]
---
pt_single = mc.build_process_tensor(
    hamiltonian,
    sim_params,
    timesteps=[0.1],
    return_type="dense",
)
print(f"Process tensor Choi matrix shape: {pt_single.to_matrix().shape}")
```

## 3. MPO process tensor

```{code-cell} ipython3
---
tags: [remove-output]
---
pt_mpo = mc.build_process_tensor(
    hamiltonian,
    sim_params,
    timesteps=[0.1],
    return_type="mpo",
    compress_every=1,
)
print(type(pt_mpo).__name__, pt_mpo.to_dense().to_matrix().shape)
```

## 4. Predict with the reference process tensor

`mc.predict(pt, rho0, sequence, num_interventions=...)` returns site-0 reduced-state dynamics from the tomographic process tensor.
`rho0` is accepted for API symmetry but **not used** (fixed product reference state).

```{code-cell} ipython3
---
tags: [remove-output]
---
rho_ref = mc.predict(pt_single, rho0, "haar", num_interventions=1)
print(f"trace(rho_ref) = {np.trace(rho_ref).real:.4f}")
```

## 5. Characterize (optional cross-check)

```{code-cell} ipython3
---
tags: [remove-output]
---
result = mc.characterize(pt_single, cut=1, num_interventions=1, n_pasts=6, n_futures=6)
print(result.summary())
```

## 6. Validation sketch

At small `num_interventions`, compare surrogate predictions against the reference process tensor:

```python
model = mc.train(hamiltonian, sim_params, num_interventions=1, n=40)
rho_sure = mc.predict(model, rho0, "haar", num_interventions=1)
rho_pt = mc.predict(pt_single, rho0, "haar", num_interventions=1)
# Compare rho_sure vs rho_pt (e.g. trace distance or fidelity)
```

## Related topics

- {doc}`characterization` — primary memory characterization guide
- {doc}`process_tensor_surrogates` — surrogate training
- {doc}`operational_memory` — memory matrix theory (advanced)
