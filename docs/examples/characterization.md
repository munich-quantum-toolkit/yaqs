---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
  execution_timeout: 900
---

```{code-cell} ipython3
:tags: [remove-cell]
%config InlineBackend.figure_formats = ['svg']
```

# Probing Environmental Memory

Open quantum systems in YAQS couple a **probe qubit** (site 0) to an **environment** simulated by the remaining chain.
**Environmental memory** measures how long the environment keeps past control and measurement choices relevant for future probe responses, evaluated at a temporal cut $c$ in a sequence of interventions.

Use {meth}`~mqt.yaqs.memory_characterizer.MemoryCharacterizer.characterize` to probe the process: assemble the weighted **response matrix** $\widetilde{V}(c)$, then read $S_V(c)$, $R(c)=\exp(S_V(c))$, and the mode spectrum.
For fast dynamics under control sequences, see {doc}`memory_surrogate`.

## Setup

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np

from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer
from mqt.yaqs.characterization.memory.shared.utils import make_zero_psi

length = 4
ham = Hamiltonian.ising(length=length, J=1.0, g=1.0)
params = AnalogSimParams(dt=0.1, max_bond_dim=16, order=1)
mc = MemoryCharacterizer(show_progress=False)
psi0 = make_zero_psi(length)
```

Throughout, `num_interventions` is the probe-sequence length $k$ and `cut` is the causal-break index $c$ (the break sits at step $c-1$; future legs use steps $c+1,\ldots,k$).
Use $k>1$ and an interior cut so both past and future probe legs contribute to $\widetilde{V}(c)$.

## Characterize with the Hamiltonian backend

The full chain (system + environment) is simulated for each probe sequence.
This is the reference memory metric when you have a microscopic open-system model.

```{code-cell} ipython3
cut, num_interventions = 4, 6
ham_result = mc.characterize(
    ham,
    params,
    cut=cut,
    num_interventions=num_interventions,
    n_pasts=8,
    n_futures=8,
    initial_psi=psi0,
    rng=np.random.default_rng(42),
)

sv = ham_result.singular_values(cut)
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].semilogy(sv, "o-")
axes[0].set_xlabel("mode index")
axes[0].set_ylabel("singular value")
axes[0].set_title(r"Memory spectrum at cut $c=4$")

v = ham_result.response_matrix(cut)
im = axes[1].imshow(np.abs(v), aspect="auto", cmap="viridis")
axes[1].set_title(r"$|\widetilde{V}(c)|$")
axes[1].set_xlabel("future probe index")
axes[1].set_ylabel("past probe index")
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
fig.suptitle(
    rf"$S_V(c={cut})={ham_result.entropy(cut):.3f}$, "
    rf"$R(c)={ham_result.modes(cut):.2f}$",
    y=1.02,
)
fig.tight_layout()
```

Use `preset="quick"`, `"balanced"`, or `"accurate"` for default probe-grid sizes, or set `n_pasts` / `n_futures` explicitly.

### Reading `CharacterizationResult`

| Access                      | Meaning                                                          |
| --------------------------- | ---------------------------------------------------------------- |
| `result.entropy(c)`         | Environmental memory entropy $S_V(c)$                            |
| `result.modes(c)`           | Effective memory modes $R(c)=\exp(S_V(c))$                       |
| `result.singular_values(c)` | Mode spectrum at cut $c$ (how many independent past branches remain visible) |
| `result.response_matrix(c)` | Cross-cut memory matrix $\widetilde{V}(c)$ (see theory below)    |
| `result.probes(c)`          | Probe arrays used at cut $c$ (for reuse or inspection)           |
| `result.summary()`          | Human-readable table of entropies and modes                      |

(memory-theory)=

## Theory: split-cut probing

Environmental memory asks: across a grid of past and future control settings on the probe, how many independent ways does the **environment** still correlate past choices with accessible future responses?

The split-cut protocol:

1. Sample past control legs $\alpha=(U_1,\ldots,U_{c-1})$ and future legs $\beta=(V_{c+1},\ldots,V_k)$ on the probe.
2. Insert a **causal break** at step $c$: measure on the past side and prepare on the future side while the environment continues to evolve.
3. For each grid entry, simulate the open system, record probe weights and the output Pauli vector $\mathbf{r}=(\langle X\rangle,\langle Y\rangle,\langle Z\rangle)$.
4. Assemble the weighted probe responses into $\widetilde{V}(c)$ and compute $S_V(c)$ from the normalized mode spectrum.

Hamiltonian `characterize` obtains weights from simulated intervention probabilities through cut $c$ (MCWF or TJM/MPS, per `representation`).
Surrogate and exact-reference backends use the same probing protocol with analytic weights on the reference probe path.

### Coupling strength and memory

Stronger Ising coupling $J$ between the probe and the environment typically increases cross-cut memory.
Reuse one `probe_set` when sweeping $J$:

```{code-cell} ipython3
j_values = np.linspace(0.0, 2.0, 9)
anchor = mc.characterize(
    Hamiltonian.ising(length=length, J=0.0, g=1.0),
    params,
    num_interventions=num_interventions,
    cut=cut,
    n_pasts=8,
    n_futures=8,
    initial_psi=psi0,
    rng=np.random.default_rng(42),
)
entropies = []
for j in j_values:
    result = mc.characterize(
        Hamiltonian.ising(length=length, J=float(j), g=1.0),
        params,
        num_interventions=num_interventions,
        cut=cut,
        probe_set=anchor,
    )
    entropies.append(result.entropy(cut))

fig, ax = plt.subplots(figsize=(5.5, 3))
ax.plot(j_values, entropies, "o-")
ax.set_xlabel(r"Ising coupling $J$")
ax.set_ylabel(r"$S_V(c)$")
ax.set_title(r"Environmental memory grows with probe-environment coupling")
fig.tight_layout()
```

### Intervention styles

`characterize` accepts `intervention_style=` (default `"haar"`):

- **`"haar"`** — random unitaries on sequence legs; measure/prepare only at the causal cut.
- **`"measure_prepare"`** — rank-1 measure–prepare maps on every leg.
- **`"clifford"`** — random single-qubit Clifford gates on legs.

Pass `probe_set=` from a Hamiltonian run so surrogate or exact-reference backends evaluate the **same** probe ensemble ({doc}`memory_surrogate`).

(reset-delay)=

## Memory persistence: reset delay at the causal break

Pass `delay=N` to insert $N$ soft-reset slots $(\lvert 0\rangle, \lvert 0\rangle)$ at the causal cut while the **environment** keeps evolving.
Extra reset time lets the environment decouple from the past before future controls act, so $S_V(c)$ often **decreases** at strong probe-environment coupling (weaker coupling can show the opposite trend).

The logical `num_interventions` and `cut` are unchanged; the physical sequence length becomes `num_interventions + delay + 1`.
Reuse the same `probe_set` when sweeping `delay`.
`delay > 0` is supported for Hamiltonian characterize only.

```{code-cell} ipython3
delay_length = 6
ham_delay = Hamiltonian.ising(length=delay_length, J=2.0, g=1.0)
params_delay = AnalogSimParams(dt=0.1)
mc_delay = MemoryCharacterizer(show_progress=False)
delay_cut = 4
delay_k = 6
anchor_delay = mc_delay.characterize(
    ham_delay,
    params_delay,
    num_interventions=delay_k,
    cut=delay_cut,
    delay=0,
    n_pasts=6,
    n_futures=6,
    initial_psi=make_zero_psi(delay_length),
    rng=np.random.default_rng(999_991),
)
delays = [0, 1, 2, 3]
delay_entropies = []
for delay in delays:
    result = mc_delay.characterize(
        ham_delay,
        params_delay,
        num_interventions=delay_k,
        cut=delay_cut,
        delay=delay,
        probe_set=anchor_delay,
    )
    delay_entropies.append(result.entropy(delay_cut))

fig, ax = plt.subplots(figsize=(4.5, 3))
ax.plot(delays, delay_entropies, "s-")
ax.set_xlabel("reset delay at causal cut")
ax.set_ylabel(r"$S_V(c)$")
ax.set_title(r"Strong coupling: memory erodes with longer reset delay")
ax.set_xticks(delays)
fig.tight_layout()
```

## Representation

`MemoryCharacterizer(representation="auto")` mirrors `Simulator`: `"vector"` selects MCWF, `"mps"` selects TJM for the **environment** chain.
With `"auto"`, MCWF is used when `hamiltonian.length <= vector_max_qubits` (default 10).

## Related topics

- {doc}`quickstart` — minimal characterize and surrogate predict snippets
- {doc}`memory_surrogate` — train a surrogate, predict dynamics, validate against exact references
- API reference: :class:`~mqt.yaqs.memory_characterizer.MemoryCharacterizer`
