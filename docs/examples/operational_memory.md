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
Here we spell out the response-matrix construction and show cut sweeps / backend comparisons.
```

**Operational memory** quantifies how many independent ways the conditioned past remains visible in accessible future responses across a temporal cut `c`.

Following the black-box construction in the cross-cut memory framework:

1. Sample past interventions :math:`\alpha=(U_1,\ldots,U_{c-1})` and future interventions :math:`\beta=(V_{c+1},\ldots,V_k)`.
2. Insert a **causal break** at step `c`: measure an effect :math:`E_m` on the past side and prepare :math:`\sigma_p` on the future side.
3. For each conditioned past :math:`(\alpha,m)` and future setting :math:`(p,\beta)`, record the break weight :math:`w_{\alpha,m}` (the probability of that conditioned past at the cut) and the output response :math:`\mathbf{r}(\rho_{\mathrm{out}})` — Pauli tomography :math:`(\langle I\rangle,\langle X\rangle,\langle Y\rangle,\langle Z\rangle)` with :math:`\langle I\rangle=1` for physical states. The cross-cut matrix :math:`\widetilde{V}(c)` uses the :math:`X,Y,Z` components only (identical to the legacy three-component pipeline).
4. Assemble the weighted response tensor, **center** over the past index to remove the Markovian background, reshape to :math:`\widetilde{V}(c)`, and take the entropy :math:`S_V(c)` of the normalized singular-value weights. Report :math:`R(c)=\exp(S_V(c))` via `result.rank(c)`.

Use {class}`~mqt.yaqs.memory_characterizer.MemoryCharacterizer` for all memory metrics:

```python
result = mc.characterize(target, cut=c, k=k, preset="balanced")
result.entropy(c)  # S_V(c)
result.singular_values(c)  # spectrum of Ṽ(c)
result.rank(c)  # R(c) = exp(S_V(c))
```

### Break weights :math:`w_{\alpha,m}`

| Backend              | How `weights_ij` are obtained                                                                                                          |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Hamiltonian**      | Product of simulated intervention probabilities along each probe sequence through cut `c` (from traced full-state simulation).       |
| **Comb / surrogate** | Analytic product through cut `c` from the site-0 :math:`\vert 0\rangle\langle 0\vert` reference path (internal branch-weight path). |

In all cases the weight depends only on the conditioned past row `i` (constant across future columns `j` for a fixed probe grid).

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
cut, k = 2, 2
n_p, n_f = 8, 8

result = mc.characterize(comb_dense, cut=cut, k=k, n_pasts=n_p, n_futures=n_f, rng=rng)
ent = result.entropy(cut)
sv = result.singular_values(cut)
rk = result.rank(cut)
v_c = result.memory_matrix(cut)

print(f"S_V({cut}) = {ent:.4f}, R({cut}) = {rk:.3f}")
print(f"spectrum length = {sv.size}")
print(f"centered response matrix shape {v_c.shape}")
```

When `cut` is omitted on `characterize()`, the default interior cut `(k + 1) // 2` is used.

## 4. Reproducible probes across backends

Run one backend first, then pass that result to `characterize(..., probe_set=...)` so both use the same past/future ensemble. Inspect the sampled arrays with `result.probes(cut)`.

```{code-cell} ipython3
---
tags: [remove-output]
---
cut, k = 2, 2

ham_result = mc.characterize(hamiltonian, sim_params, cut=cut, k=k, n_pasts=6, n_futures=5, rng=rng)
comb_result = mc.characterize(comb_dense, cut=cut, k=k, probe_set=ham_result)
print(f"Hamiltonian S_V = {ham_result.entropy(cut):.4f}")
print(f"Comb S_V       = {comb_result.entropy(cut):.4f}")
print("past feature shape:", ham_result.probes(cut)["past_features"].shape)
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
ax.set_xlabel("mode index")
ax.set_ylabel("singular value")
ax.set_title("Spectrum of Ṽ(c) at cut c=2")
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
axes[0].set_ylabel("S_V(c)")
axes[0].set_title("Memory entropy vs cut")

axes[1].plot(list(cuts), ranks, "s-")
axes[1].set_xlabel("cut c")
axes[1].set_ylabel("R(c)")
axes[1].set_title("Effective modes vs cut")
fig.tight_layout()
```

## 6. Coupling sweep (doc-sized)

Lightweight :math:`S_V(c)` vs coupling :math:`J` at fixed cuts.

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

## Internal package layout (developers)

User code should call :class:`~mqt.yaqs.memory_characterizer.MemoryCharacterizer` only. Lower-level modules live under ``mqt.yaqs.characterization.memory``:

| Path | Role |
| ---- | ---- |
| ``operational_memory/`` | Split-cut protocol: ``sample_probes``, ``assemble_probe_grid``, ``run_operational_memory`` |
| ``shared/`` | Choi/rho encoding and site-0 MCWF/TJM helpers (not ``mqt.yaqs.core``) |
| ``backends/exact.py`` | :class:`~mqt.yaqs.characterization.memory.backends.exact.ExactBackend` for Hamiltonian characterize |
| ``backends/tomography/`` | Reference dense/MPO combs |
| ``backends/surrogates/`` | ``TransformerComb``, ``simulate_sequences``, ``SeqTrace`` training traces |

Verb-first naming is used throughout (``compute_*``, ``assemble_*``, ``simulate_*``, ``encode_*``).

## Related topics

- {doc}`characterization` — main user funnel
- {doc}`reference_exact_combs` — dense and MPO reference comb construction
- {doc}`process_tensor_surrogates` — surrogate training
