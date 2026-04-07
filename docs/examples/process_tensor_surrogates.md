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

# Process Tensor Surrogates (TransformerComb)

This example shows how to train and use the **TransformerComb** surrogate model for process-tensor workflows.

## What the surrogate is intended to do

The surrogate learns a fast approximation of the **site-0 reduced state rollout** produced by the simulator under
random intervention sequences.

- **Input**: a packed representation of the initial site-0 density matrix (`rho0`) and a per-step feature vector (`E_t`)
  describing the intervention at each step.
- **Output**: predicted packed reduced density matrices for site 0 at each step.

This is useful when you want to:

- generate many rollouts quickly after a one-time training cost,
- compare simulation settings / regimes via predicted reduced dynamics,
- run parameter sweeps where full backend simulation would be too expensive.

The surrogate is **not** a replacement for tomography:

- it does **not** reconstruct the full comb matrix \( \Upsilon \),
- it predicts only the reduced dynamics for the training data distribution (model/solver/timestep regime).

## 1. Define the system

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

operator = MPO.ising(length=2, J=1.0, g=1.0)

sim_params = AnalogSimParams(
    solver="MCWF",  # also works with "TJM" (surrogate generation is noiseless)
    dt=0.1,
    order=1,
    max_bond_dim=16,
)
```

## 2. Generate training data

`generate_data` samples random interventions and simulates rollouts. It returns a PyTorch `TensorDataset`
with tensors `(E_features, rho0, rho_seq)`:

- `E_features`: `(n, k, d_e)`
- `rho0`: `(n, 8)` (packed `rho8`)
- `rho_seq`: `(n, k, 8)` (packed `rho8` per step)

```{code-cell} ipython3
from mqt.yaqs import generate_data

train_ds = generate_data(
    operator,
    sim_params,
    k=4,          # number of intervention steps
    n=80,         # number of sampled sequences
    seed=0,
    parallel=True,
    show_progress=False,
)

E, rho0, tgt = train_ds.tensors
print(E.shape, rho0.shape, tgt.shape)
```

## 3. Train a TransformerComb explicitly

If you want full control of architecture and training arguments, instantiate `TransformerComb` directly and call `fit`.

```{code-cell} ipython3
import torch
from mqt.yaqs import TransformerComb

d_e = int(E.shape[-1])
model = TransformerComb(
    d_e=d_e,
    d_rho=8,
    d_model=128,
    nhead=4,
    num_layers=3,
    dim_ff=256,
    dropout=0.0,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.fit(
    train_ds,
    epochs=50,
    lr=2e-3,
    batch_size=64,
    device=device,
)
```

## 4. Predict rollouts

`predict` expects the same shapes as training:

- `E_features`: `(B, k, d_e)`
- `rho0`: `(B, 8)`

```{code-cell} ipython3
pred = model.predict(E[:5], rho0[:5], device=device, return_numpy=True)
print(pred.shape)  # (5, k, 8)
```

## 5. One-call helper: create_surrogate

`create_surrogate` wraps `generate_data -> TransformerComb -> fit` for a quick end-to-end workflow.

```{code-cell} ipython3
from mqt.yaqs import create_surrogate

quick_model = create_surrogate(
    operator,
    sim_params,
    k=4,
    n=80,
    seed=0,
    parallel=True,
    show_progress=False,
    model_kwargs={"d_model": 128, "nhead": 4, "num_layers": 3},
    train_kwargs={"epochs": 50, "lr": 2e-3, "batch_size": 64},
)
```

