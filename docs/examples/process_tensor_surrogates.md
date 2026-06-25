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

```{note}
Start with {doc}`characterization` for the main predict/characterize funnel. This page covers advanced training knobs and Transformer internals.
```

This example shows how to train and use the **TransformerComb** surrogate model.

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

## Transformer structure and sequence length

`TransformerComb` is a **causal transformer** over per-step intervention features, conditioned on the initial packed `rho0`:

- sinusoidal positional encoding and a causal attention mask ensure step `t` only sees steps `≤ t`,
- each step emits a packed `rho8` reduced density matrix for site 0.

Training via `mc.train(..., k=...)` fixes the rollout length and stores `sequence_length` on the model after `fit`.

**Inference flexibility:** `mc.predict(model, rho0, seq, k=k_prime)` may use `k_prime` shorter or longer than the training `k` because the encoder is length-agnostic. Accuracy is best at or near the trained horizon and diminishes when extrapolating beyond it.

For user-facing dynamics, prefer `MemoryCharacterizer.predict`; the low-level `model.predict(E, rho0)` API remains for batch training shapes.

## 1. Define the system

```{code-cell} ipython3
from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer

hamiltonian = Hamiltonian.ising(length=2, J=1.0, g=1.0)

sim_params = AnalogSimParams(
    dt=0.1,
    order=1,
    max_bond_dim=16,
)
mc = MemoryCharacterizer(parallel=False, show_progress=False)
```

## 2. Generate training data

`sample` samples random interventions and simulates rollouts. It returns a PyTorch `TensorDataset`
with tensors `(E_features, rho0, rho_seq)`:

- `E_features`: `(n, k, d_e)`
- `rho0`: `(n, 8)` (packed `rho8`)
- `rho_seq`: `(n, k, 8)` (packed `rho8` per step)

```{code-cell} ipython3
try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    train_ds = mc.sample(
        hamiltonian,
        sim_params,
        k=4,          # number of intervention steps
        n=80,         # number of sampled sequences
        seed=0,
        parallel=True,
        show_progress=False,
        interventions="measure_prepare",
    )

    E, rho0, tgt = train_ds.tensors
    print(E.shape, rho0.shape, tgt.shape)
else:
    print("torch not installed; skip surrogate data generation in doc build")
```

## 3. Train a TransformerComb explicitly

If you want full control of architecture and training arguments, instantiate `TransformerComb` directly and call `fit`.

```{code-cell} ipython3
if torch is not None:
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
if torch is not None:
    pred = model.predict(E[:5], rho0[:5], device=device, return_numpy=True)
    print(pred.shape)  # (5, k, 8)
```

For single-sequence user workflows, use `mc.predict(model, rho0, sequence, k=...)`.

## 5. Operational memory via surrogate backend

After training, `mc.characterize(model, ...)` quantifies operational memory at cut `c` using the surrogate as the process backend — the same `S_V` quantity as Hamiltonian characterize, not a training-quality score.

```{code-cell} ipython3
if torch is not None:
    memory = mc.characterize(model, cut=2, k=4, n_pasts=8, n_futures=8)
    print(f"S_V(2) = {memory.entropy(2):.4f} nats")
    print(f"singular spectrum length: {memory.singular_values(2).size}")
```

The surrogate learns reduced-state rollouts; it does **not** reconstruct the full comb matrix :math:`\Upsilon`.

## 6. One-call training via `MemoryCharacterizer.train`

`MemoryCharacterizer.train` wraps `sample` → `TransformerComb` → `fit` for a quick end-to-end workflow.

```{code-cell} ipython3
if torch is not None:
    quick_model = mc.train(
        hamiltonian,
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

## Related topics

- {doc}`characterization` — main predict/characterize funnel
- {doc}`reference_exact_combs` — exact comb tomography for small `k`
- {doc}`operational_memory` — V-matrix theory (advanced)
