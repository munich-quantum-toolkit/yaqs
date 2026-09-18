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

Open quantum systems in YAQS couple a **probe qubit** (site 0) to an
**environment** simulated by the remaining chain. **Environmental memory**
measures how long the environment keeps past control and measurement choices
relevant for future probe responses, evaluated at a temporal cut $c$ in a
sequence of interventions.

Memory characterization currently supports qubit Hamiltonians only.

Use {meth}`~mqt.yaqs.memory_characterizer.MemoryCharacterizer.characterize` to
probe **operational memory**: assemble the **response matrix** $V(c)$, then read
$S_V(c)$, $R(c)=\exp(S_V(c))$, and the mode spectrum.

Alternatively, build a process tensor (default: direct MPO) and call
`compute_temporal_entropy` for **temporal entanglement** $S_{PT}(c)$ of the
multi-time process itself — a distinct quantity from $S_V(c)$. For fast dynamics
under control sequences, see {doc}`memory_surrogate`.

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

Throughout, `num_interventions` is the probe-sequence length $k$ and `cut` is
the causal-break index $c$ (the break sits at step $c-1$; future legs use steps
$c+1,\ldots,k$). Use $k>1$ and an interior cut so both past and future probe
legs contribute to $V(c)$.

## Characterize with the Hamiltonian backend

The full chain (system + environment) is simulated for each probe sequence. This
is the reference memory metric when you have a microscopic open-system model.

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
axes[1].set_title(r"$|V(c)|$")
axes[1].set_xlabel("history index")
axes[1].set_ylabel("future probe and response channel")
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
fig.suptitle(
    rf"$S_V(c={cut})={ham_result.entropy(cut):.3f}$, "
    rf"$R(c)={ham_result.modes(cut):.2f}$",
    y=1.02,
)
fig.tight_layout()
```

Use `preset="quick"`, `"balanced"`, or `"accurate"` for default probe-grid
sizes, or set `n_pasts` / `n_futures` explicitly.

### Reading `CharacterizationResult`

| Access                             | Meaning                                                                    |
| ---------------------------------- | -------------------------------------------------------------------------- |
| `result.entropy(c)`                | Environmental memory entropy $S_V(c)$                                      |
| `result.modes(c)`                  | Effective memory modes $R(c)=\exp(S_V(c))$                                 |
| `result.singular_values(c)`        | Resolution-retained spectrum used to compute $S_V(c)$                      |
| `result.singular_values_full(c)`   | Every compact-SVD value, including zero and unresolved tail values         |
| `result.left_singular_vectors(c)`  | All compact-SVD future-response directions as columns                      |
| `result.right_singular_vectors(c)` | All compact-SVD history-combination directions as columns                  |
| `result.response_matrix(c)`        | $V(c)$ with $4N_f$ IXYZ future-response rows and $N_h$ history columns     |
| `result.probes(c)`                 | Probe arrays used at cut $c$ (for reuse or inspection)                     |
| `result.summary()`                 | Human-readable table of entropies and modes                                |

(memory-theory)=

## Theory: split-cut probing

Environmental memory asks: across a grid of past and future control settings on
the probe, how many independent ways does the **environment** still correlate
past choices with accessible future responses?

The split-cut protocol:

1. Sample past control legs $\alpha=(U_1,\ldots,U_{c-1})$ and future legs
   $\beta=(V_{c+1},\ldots,V_k)$ on the probe.
2. Insert a **causal break** at step $c$: measure on the past side and prepare
   on the future side while the environment continues to evolve.
3. For each grid entry, simulate the open system, record the joint probability
   of the retained outcomes and the normalized final-system Pauli response
   $\mathbf{r}=(\langle I\rangle,\langle X\rangle,\langle Y\rangle,\langle Z\rangle)$.
4. Assemble the sampled response coefficients into $V(c)$. Its rows contain one
   $(I,X,Y,Z)$ block per future probe, its columns label conditioned histories,
   and $V_{(j,I),i}=w_{ij}$ for normalized output states. Compute $S_V(c)$ from
   the normalized mode spectrum.

For an SVD $V=U\Sigma W^\dagger$, each column pair associated with a retained,
nonzero singular value defines a response mode: the column of $U$ gives the
future-response direction, while the column of $W$ gives a combination of
conditioned histories. With `U = result.left_singular_vectors(c)`,
`s = result.singular_values_full(c)`, and
`W = result.right_singular_vectors(c)`, the full factors satisfy
`V = U @ np.diag(s) @ W.conj().T`. The full factors also contain directions
paired with exact zeros or an unresolved numerical tail. Do not interpret those
directions as resolved memory modes. Singular vectors are also not unique inside
a degenerate singular subspace.

Hamiltonian `characterize` obtains joint probabilities of the retained outcomes
from the simulated intervention sequence (MCWF or TJM/MPS, per
`representation`). Process-tensor backends obtain the same probabilities from
the trace of each subnormalized contraction, while surrogates estimate them from
their predicted pre-intervention reduced states.

### Coupling strength and memory

Stronger Ising coupling $J$ between the probe and the environment typically
increases cross-cut memory. Reuse one `probe_set` when sweeping $J$:

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

- **`"haar"`** — random unitaries on sequence legs; measure/prepare only at the
  causal cut.
- **`"measure_prepare"`** — rank-1 measure–prepare maps on every leg.
- **`"clifford"`** — random single-qubit Clifford gates on legs.

Pass `probe_set=` from a Hamiltonian run so surrogate or exact-reference
backends evaluate the **same** probe ensemble ({doc}`memory_surrogate`).
Surrogate characterization also requires `initial_rho=`: the site-0 density
matrix after the schedule's initial evolution segment and before its first
intervention. For a surrogate trained against a reference process tensor, use
that tensor's `initial_rho`.

(reset-delay)=

## Memory persistence: conditioned reset delay

Pass `delay=N`, for any $N\geq0$, to use the conditioned-reset protocol from
Figure 5 of the response-matrix paper. The intervention at the history boundary
applies the selected measurement and prepares $\lvert0\rangle$. YAQS then
inserts $N$ selected-zero reset slots $(\lvert0\rangle,\lvert0\rangle)$ and
applies a second selected-zero measurement before the sampled future
preparation. The environment keeps evolving between these interventions. The
selected history outcome and every selected-zero outcome contribute to the
complete branch probability.

The two boundary interventions remain separate at `delay=0`. The physical
sequence length is therefore `num_interventions + delay + 1` for every explicit
delay. Omitting `delay` uses the standard one-step causal break
`(selected_history_measurement, sampled_future_preparation)` instead. This keeps
ordinary characterization aligned across Hamiltonian, process-tensor, and
surrogate backends.

Extra reset time lets the environment decouple from the past before future
controls act, so $S_V(c)$ often decreases at strong probe-environment coupling.
Weaker coupling can show a nonmonotonic profile. The example below uses a
smaller probe grid and shorter sequences than the paper campaign, but it uses
the same conditioned-reset geometry. Reuse the same `probe_set` across the delay
sweep. An explicit `delay` is supported for Hamiltonian characterization only.

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

`MemoryCharacterizer(representation="auto")` mirrors `Simulator`: `"vector"`
selects MCWF, `"mps"` selects TJM for the **environment** chain. With `"auto"`,
MCWF is used when `hamiltonian.length <= vector_max_qubits` (default 10).

## Temporal entanglement from a process tensor

Operational memory ($S_V$) comes from probe responses. **Temporal entanglement**
$S_{PT}(c)$ is computed directly from a process tensor at the same causal cut.
By default, `build_process_tensor` uses direct MPO construction
(`return_type="mpo"`). Pass `return_type="dense"` for exhaustive tomography
(required for `noise_model`):

```{code-cell} ipython3
k = 2
cut_pt = 1
timesteps = [0.1] * (k + 1)

pt_mpo = mc.build_process_tensor(
    ham,
    params,
    timesteps=timesteps,
)
pt_dense = mc.build_process_tensor(
    ham,
    params,
    timesteps=timesteps,
    return_type="dense",
)

s_mpo = pt_mpo.compute_temporal_entropy(cut_pt)
s_dense = pt_dense.compute_temporal_entropy(cut_pt)
print(
    f"S_PT(c={cut_pt}): mpo={s_mpo['entropy']:.4f}, "
    f"dense={s_dense['entropy']:.4f}, schmidt_rank={s_mpo['schmidt_rank']}"
)

# The same exact process tensor also supports operational memory via characterize:
pt_result = mc.characterize(
    pt_mpo,
    cut=cut_pt,
    num_interventions=k,
    n_pasts=6,
    n_futures=6,
    rng=np.random.default_rng(7),
)
print(f"S_V(c={cut_pt}) from process-tensor probes: {pt_result.entropy(cut_pt):.4f}")
```

Dense and default direct MPO construction agree on $S_{PT}$ for small $k$. The
supported direct path is uncapped: at intervention leg $k$, it can retain up to
$16^k$ histories and construct the same number of rank-one terms. Use it only
for short horizons. `compress_every` limits an accumulation batch; it does not
limit the number of histories. Dense tomography has the same $16^k$ sequence
count and is required when you use `noise_model`.

`MPOProcessTensor.compute_temporal_entropy()` currently converts the complete
MPO to a dense matrix. The matrix alone uses $64\,16^k$ bytes for $k$
intervention legs, before decomposition workspace. On a typical workstation,
restrict this calculation to about five legs. The matrix uses 64 MiB at five
legs and 1 GiB at six legs.

```{warning}
Passing a finite `max_bond_dim` enables experimental direct-MPO truncation and
emits a `RuntimeWarning`. This uncontrolled approximation can change the process
tensor and does not preserve positivity or causal normalization. Do not use a
capped result as a stable scientific reference.
```

Operational characterization requires each contracted branch to be Hermitian and
positive semidefinite, with trace in $[0,1]$. Response assembly also checks that
each normalized qubit response lies in the Bloch ball. QMI and CMI always
normalize the process tensor and validate positive semidefiniteness. Remove an
experimental finite cap by setting `max_bond_dim=None`, or use a sufficiently
accurate dense reconstruction when you need $S_V$ from a process tensor.
`characterize(pt, ...)` uses native MPO `evaluate_probes_with_weights` without
densifying the V-matrix path.

## Related topics

- {doc}`quickstart` — minimal characterize and surrogate predict snippets
- {doc}`memory_surrogate` — train a surrogate, predict dynamics, validate
  against exact references
- API reference: :class:`~mqt.yaqs.memory_characterizer.MemoryCharacterizer`
