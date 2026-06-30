---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
  execution_timeout: 600
---

# Process Tensor Surrogates (ProcessTensorSurrogate)

```{code-cell} ipython3
:tags: [remove-cell]
%config InlineBackend.figure_formats = ['svg']
```

```{note}
Start with {doc}`characterization` for the main predict/characterize funnel. This page covers surrogate training details and optional architecture knobs.
```

A **surrogate** learns to approximate site-0 reduced-state dynamics under intervention sequences — the same object you query in production via `mc.predict(model, rho0, sequence, num_interventions=...)`.

- **Input:** initial site-0 density matrix `rho0` and an intervention sequence (preset string or explicit step list).
- **Output:** predicted site-0 density matrix after the sequence (or after each step if requested).

The surrogate does **not** reconstruct the full process-tensor Choi matrix :math:`\Upsilon`; it predicts reduced dynamics only for the training regime (model, solver, timestep).

## Transformer structure and sequence length

`ProcessTensorSurrogate` is a **causal transformer** conditioned on `rho0` and per-step intervention encodings:

- sinusoidal positional encoding and a causal attention mask ensure step `t` only sees steps `≤ t`,
- each step predicts the site-0 reduced density matrix.

`mc.train(..., num_interventions=...)` fixes the number of intervention steps used during training and stores that horizon on the model.

**Inference flexibility:** `mc.predict(model, rho0, seq, num_interventions=n_prime)` may use `n_prime` shorter or longer than the training horizon because the encoder is length-agnostic. Accuracy is best for `n_prime` up to the trained value and generally decreases when `n_prime` exceeds it.

## Train and predict

```{code-cell} ipython3
from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer
import numpy as np

hamiltonian = Hamiltonian.ising(length=2, J=1.0, g=1.0)
sim_params = AnalogSimParams(dt=0.1, order=1, max_bond_dim=16)
mc = MemoryCharacterizer(parallel=False, show_progress=False)

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    model = mc.train(
        hamiltonian,
        sim_params,
        num_interventions=4,
        n=80,
        seed=0,
        model_kwargs={"d_model": 128, "nhead": 4, "num_layers": 3},
        train_kwargs={"epochs": 50, "lr": 2e-3, "batch_size": 64},
    )
    rho0 = np.eye(2, dtype=np.complex128) / 2.0
    rho_out = mc.predict(model, rho0, "haar", num_interventions=4)
    print(f"trace(rho) = {np.trace(rho_out).real:.4f}")
else:
    print("torch not installed; skip surrogate path in doc build")
```

## Characterize with the surrogate backend

`mc.characterize(model, cut=c, num_interventions=..., ...)` evaluates the same cross-cut memory entropy :math:`S_V(c)` as Hamiltonian characterize, using the surrogate as the black-box process.

```{code-cell} ipython3
if torch is not None:
    memory = mc.characterize(model, cut=2, num_interventions=4, n_pasts=8, n_futures=8)
    print(f"S_V(2) = {memory.entropy(2):.4f}")
    print(f"R(2) = {memory.modes(2):.3f}")
```

## Advanced: custom architecture and `sample`

For full control over architecture, training loops, or batched tensor shapes, use `mc.sample(...)` to generate a PyTorch dataset and instantiate {class}`~mqt.yaqs.characterization.memory.backends.surrogates.model.ProcessTensorSurrogate` directly. Training data are built from simulated :class:`~mqt.yaqs.characterization.memory.backends.surrogates.data.SeqTrace` records via :func:`~mqt.yaqs.characterization.memory.backends.surrogates.workflow.sample_train_dataset`. User-facing dynamics should still go through `mc.predict`.

## Related topics

- {doc}`characterization` — main predict/characterize funnel
- {doc}`reference_process_tensors` — reference process tensor at small `num_interventions`
- {doc}`operational_memory` — response-matrix construction (advanced)
