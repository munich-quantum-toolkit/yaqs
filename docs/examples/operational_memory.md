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

# Operational Memory Diagnostics

```{note}
This page is **not** the main on-ramp. For predict/characterize workflows start with {doc}`characterization`.
Here we cover V-matrix theory, `probe_process`, and cross-backend validation.
```

**Operational memory** quantifies how much history an open quantum process retains at a temporal cut `c`.
YAQS estimates it from **split-cut probes**: random interventions on the past and future sides of the cut, assembled into a weighted **V matrix** with causal branch weights :math:`w_{ij} = \prod_t p(\text{step}_t)` through cut `c` (starting from :math:`|0\rangle\langle 0|`).
Past-row centering and the singular spectrum of the centered V yield bond entropy :math:`S_V(c)` and an **operational rank**.

Use {class}`~mqt.yaqs.memory_characterizer.MemoryCharacterizer` for all memory metrics:

```python
result = mc.characterize(target, cut=c, preset="balanced")
result.entropy(c)
result.memory_matrix(c)
```

This page also covers the low-level `probe_process` return dictionary, plots, and validation against exact rollouts.

For comb construction see {doc}`reference_exact_combs`; for surrogate training see {doc}`process_tensor_surrogates`.

## 1. Setup

```{code-cell} ipython3
import numpy as np

from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer

hamiltonian = Hamiltonian.ising(length=2, J=1.0, g=0.5)
sim_params = AnalogSimParams(dt=0.1, max_bond_dim=12, order=1)
mc = MemoryCharacterizer(parallel=False, show_progress=False)
rng = np.random.default_rng(0)
```

## 2. Reference combs

```{code-cell} ipython3
---
tags: [remove-output]
---
comb_dense = mc.build_comb(
    hamiltonian,
    sim_params,
    timesteps=[0.1, 0.1],
    num_trajectories=60,
    return_type="dense",
)

comb_mpo = mc.build_comb(
    hamiltonian,
    sim_params,
    timesteps=[0.1, 0.1],
    num_trajectories=60,
    return_type="mpo",
    compress_every=1,
)

print("Dense:", type(comb_dense).__name__)
print("MPO:", type(comb_mpo).__name__)
```

## 3. Characterize memory metrics

```{code-cell} ipython3
---
tags: [remove-output]
---
cut, n_p, n_f = 2, 8, 8

result = mc.characterize(comb_dense, cut=cut, k=2, n_pasts=n_p, n_futures=n_f, rng=rng)
ent = result.entropy(cut)
sv = result.singular_values(cut)
rk = result.rank(cut)
v_c = result.memory_matrix(cut)

print(f"S_V({cut}) = {ent:.4f} nats")
print(f"spectrum length = {sv.size}, operational rank = {rk}")
print(f"memory matrix shape {v_c.shape}")
```

When `cut` is omitted on `characterize()`, the default interior cut `(k + 1) // 2` is used.

## 4. Low-level API — `probe_process`

Use {func}`~mqt.yaqs.characterization.memory.diagnostics.probe.probe_process` for full control and inspection of intermediate objects.

```{code-cell} ipython3
---
tags: [remove-output]
---
from mqt.yaqs.characterization.memory.diagnostics.probe import probe_process

out = probe_process(
    process=comb_dense,
    cut=2,
    k=2,
    n_pasts=8,
    n_futures=8,
    return_v=True,
    rng=np.random.default_rng(0),
)

for key in sorted(out):
    val = out[key]
    if hasattr(val, "shape"):
        print(f"{key:24s} array{val.shape}")
    elif isinstance(val, float):
        print(f"{key:24s} {val:.6f}")
    else:
        print(f"{key:24s} {type(val).__name__}")
```

Key quantities:

| Key                    | Meaning                                                    |
| ---------------------- | ---------------------------------------------------------- |
| `pauli_xyz_ij`         | Probe responses `(n_pasts, n_futures, 3)`                  |
| `weights_ij`           | Causal cut branch weights on comb backends                 |
| `V`, `V_centered`      | Weighted V and past-row-centered V (`return_v=True`)       |
| `entropy`              | Bond entropy :math:`S_V(c)` in nats                        |
| `singular_values_full` | Full SVD spectrum of centered V                            |
| `rank`                 | Operational rank from spectrum threshold                   |
| `delta_norm`           | :math:`\|V_c\|_F^2 / \|V\|_F^2` — effect of past centering |
| `probe_set`            | Frozen probe grid for reproducible re-runs                 |

Manual V assembly (equivalent to characterize on weighted backends):

```{code-cell} ipython3
---
tags: [remove-output]
---
from mqt.yaqs.characterization.memory.diagnostics.probe import (
    analyze_v_matrix,
    build_weighted_v_from_probe,
)

pauli = out["pauli_xyz_ij"]
weights = out["weights_ij"]
v, v_c = build_weighted_v_from_probe(pauli, weights)
ana = analyze_v_matrix(v, v_c)
print(f"manual S_V = {ana['entropy']:.4f}, comb S_V = {out['entropy']:.4f}")
```

## 5. Singular spectrum and cut sweep

```{code-cell} ipython3
---
tags: [remove-output]
---
import matplotlib.pyplot as plt

sv = mc.characterize(
    comb_dense,
    cut=2,
    k=2,
    n_pasts=8,
    n_futures=8,
    rng=np.random.default_rng(1),
).singular_values(2)
fig, ax = plt.subplots(figsize=(5, 3))
ax.semilogy(sv, "o-")
ax.set_xlabel("index")
ax.set_ylabel("singular value")
ax.set_title("Centered V spectrum at cut c=2")
fig.tight_layout()
```

```{code-cell} ipython3
---
tags: [remove-output]
---
k = 2
cuts = range(1, k + 1)
ents = [
    mc.characterize(comb_dense, cut=c, k=k, n_pasts=8, n_futures=8, rng=np.random.default_rng(c)).entropy(c)
    for c in cuts
]
ranks = [
    mc.characterize(comb_dense, cut=c, k=k, n_pasts=8, n_futures=8, rng=np.random.default_rng(c)).rank(c)
    for c in cuts
]

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].plot(list(cuts), ents, "o-")
axes[0].set_xlabel("cut c")
axes[0].set_ylabel("S_V (nats)")
axes[0].set_title("Bond entropy vs cut")

axes[1].plot(list(cuts), ranks, "s-")
axes[1].set_xlabel("cut c")
axes[1].set_ylabel("operational rank")
axes[1].set_title("Rank vs cut")
fig.tight_layout()
```

## 6. Comb vs simulator reference

For validation, compare comb probing against full MCWF rollouts via {mod}`~mqt.yaqs.characterization.memory.reference.exact` (benchmark / test tooling).

```{code-cell} ipython3
---
tags: [remove-output]
---
from mqt.yaqs.characterization.memory.diagnostics.probe import sample_split_cut_probes
from mqt.yaqs.characterization.memory.reference.exact import (
    evaluate_exact_probe_set_with_diagnostics,
)

probe_set = sample_split_cut_probes(cut=2, k=2, n_pasts=6, n_futures=5, rng=np.random.default_rng(3))
psi0 = np.zeros(4, dtype=np.complex128)
psi0[0] = 1.0

pauli_e, weights_e, _ = evaluate_exact_probe_set_with_diagnostics(
    probe_set=probe_set,
    operator=hamiltonian.mpo,
    sim_params=sim_params,
    initial_psi=psi0,
    parallel=False,
)
v_e, v_c_e = build_weighted_v_from_probe(pauli_e, weights_e)
out_exact = analyze_v_matrix(v_e, v_c_e)

out_comb = probe_process(process=comb_dense, cut=2, k=2, probe_set=probe_set)
print(f"exact S_V = {out_exact['entropy']:.4f}, comb S_V = {out_comb['entropy']:.4f}")
```

## 7. Open-system sweep (doc-sized)

Lightweight :math:`S_V` vs coupling :math:`J` at fixed cuts (inspired by the experiment benchmarks).

```{code-cell} ipython3
---
tags: [remove-output]
---
js = [0.0, 1.0, 2.0]
cuts = [1, 2]
rows = []
for jv in js:
    ham_j = Hamiltonian.ising(length=1, J=jv, g=0.0)
    comb_j = mc.build_comb(
        ham_j,
        AnalogSimParams(dt=0.1, max_bond_dim=8, order=1),
        timesteps=[0.1, 0.1],
        num_trajectories=40,
        return_type="dense",
    )
    for cut in cuts:
        ent_j = mc.characterize(
            comb_j,
            cut=cut,
            k=2,
            n_pasts=8,
            n_futures=8,
            rng=np.random.default_rng(int(jv * 10 + cut)),
        ).entropy(cut)
        rows.append({"J": jv, "cut": cut, "S_V": ent_j})

for row in rows:
    print(f"J={row['J']:.1f}, cut={row['cut']}, S_V={row['S_V']:.4f}")
```

## Related topics

- {doc}`reference_exact_combs` — dense and MPO reference comb construction
- {doc}`process_tensor_surrogates` — surrogate combs for larger `k`
- {doc}`realistic_noise_models` — attaching open-system noise to tomography runs
