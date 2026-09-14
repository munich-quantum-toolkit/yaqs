# Updated response-matrix experiments

This directory contains the Figure 2 through Figure 5 and appendix consistency
experiments after the response-matrix update, together with an exact
quantum-memory-witness benchmark. The updated matrix is uncentered, uses
future-response rows and history columns, includes the `I, X, Y, Z` channels,
and uses complete retained-record probabilities.

The stored Figure 2 and Figure 3 results use the paper settings: `L=6`, `k=20`,
`dt=0.1`, `g=1`, the initial state `|0>^6`, 64 history probes, 64 future probes,
seed 0, and one Haar-probe draw. The Figure 2 sweep uses all cuts from 1 through
20. Both sweeps use `J=0,0.05,...,2`.

Figure 4 uses the original small-process comparison: `L=6`, `k=3`, `dt=0.1`,
`g=1`, exhaustive tetrahedral probe grids, cuts `1,2,3`, and `J=0,0.2,...,6`.
Its spectrum panels use cut 2 at `J=0.1,1,2,4`.

Figure 5 uses `L=6`, `dt=0.1`, `g=1`, a fixed 15-step history and five-step
future, 64 history and future probes, `J=0.5,1,1.5,2`, and conditioned-reset
bridge lengths `ell=0,...,15`. The initial state is `|0>^6`.

The appendix probe-budget sweep uses `L=6`, `k=20`, `dt=0.1`, `g=1`, cut 10,
five Haar-probe draws, `m=4,8,16,32,64`, and `J=0,0.2,...,2`.

The appendix finite-size sweep uses `L=2,...,10`, `k=20`, `dt=0.1`, `g=1`, cuts
`5,10,15`, `J=0.5,1,1.5,2`, and 32 history and future probes.

The quantum-memory benchmark evolves six Pauli eigenstate inputs through an
exact two-qubit SWAP-reset-SWAP protocol. It uses 101 depolarization values,
including `p=2/3` exactly, and constructs the raw `4 x 6` response matrix in
fixed `I, X, Y, Z` row and `+X, -X, +Y, -Y, +Z, -Z` column order. No finite-shot
sampling is used.

## Contents

- `experiments/common.py`: shared simulation and plotting functions.
- `experiments/cut_vs_j.py`: Figure 2 cut-resolved memory landscape.
- `experiments/modes.py`: Figure 3 effective modes and singular spectra.
- `experiments/mpo_comparison`: Figure 4 response/PT-MPO campaign and plotting
  package. Its local `full_basis.py` preserves the probe catalog removed from
  the current public YAQS API.
- `experiments/probe_budget.py`: appendix probe-budget convergence sweep.
- `experiments/finite_size.py`: appendix finite-environment sweep at the three
  displayed cuts.
- `experiments/memory_persistence.py`: Figure 5 conditioned-reset persistence
  sweep.
- `experiments/quantum_memory_witness.py`: exact noisy SWAP-reset-SWAP
  quantum-memory benchmark, response entropy, and fixed linear witness.
- `results/figure2`: the Figure 2 PDF and PNG, scalar data, initial state, and
  run manifest.
- `results/figure3`: the Figure 3 PDF and PNG, scalar data, full plotted
  spectra, initial state, and run manifest.
- `results/figure4`: the Figure 4 PDF and PNG, 93-point entropy table, full
  plotted spectra, qualitative checks, and run manifest.
- `results/figure5`: the Figure 5 PDF and PNG, plotted scalar data, and run
  manifest.
- `results/probe_budget`: the appendix PDF and PNG, per-draw and summary data,
  and run manifest.
- `results/finite_size`: the appendix PDF and PNG, plotted scalar data, and run
  manifest.
- `results/quantum_memory_witness`: the witness PDF and PNG, scalar CSV, raw
  response arrays, and run manifest.

The Figure 2 logarithmic color range is `1e-4` to `5e-1`, which preserves the
visual balance of the earlier figure at the updated entropy scale. Its black
`J=0` row with white hatching and matching colorbar swatch denote the exact
zero-entropy result. The Figure 3 main axis scales automatically from the updated
effective-mode values. Its inset keeps the full normalized squared
singular-value spectrum and displays values down to `1e-16`.

Figure 4 retains the original data-driven logarithmic styling. With the updated
values, panel (a) spans `1e-7` to approximately `2.70e-2`; the spectrum panels
remain at `1e-18` to `1.05` and display six modes.

The probe-budget figure spans `1e-5` to `1e-1`. Its old/new matched log-value
correlation is `0.9939`: the absolute entropy decreases, while the coupling
ordering and broad stabilization with increasing budget remain qualitatively the
same.

The finite-size figure rescales its shared vertical axis from 0 to approximately
`0.209`. Its old/new log-value correlations are `0.9869`, `0.9906`, and `0.9939`
at cuts 5, 10, and 15. The smallest environments remain the most size-dependent,
followed by broad plateaus after several environmental sites.

Figure 5 spans `1e-4` to `1e-1`. It retains the original qualitative profile:
the weak-coupling curve is nonmonotonic, the `J=1` curve has a small late
increase, and the `J=1.5` and `J=2` curves cross and approach similar values by
`ell=15`.

## Certifying quantum memory from response data

The intended paper subsection title is
`\subsection{Certifying quantum memory from response data}`. The benchmark's
simulated output states, response singular values, entropy, and witness agree
with the independent analytic predictions to floating-point precision. The
witness crosses zero at `p=2/3`; `S_V(2/3) = 0.434944202258` while
`S_V(1) = 0`. Thus `p<2/3` certifies quantum memory under the stated classical
feed-forward model, `2/3 <= p < 1` is classically reproducible for this known
benchmark, and `p=1` is memoryless. A nonnegative witness is generally
inconclusive outside this benchmark-specific classification.

The updated figures retain the qualitative conclusions of the earlier figures:

- The response is at the numerical baseline at `J=0` and near the first causal
  cut.
- The response grows with coupling and is largest over interior causal cuts.
- The effective mode number remains close to one at the boundary and increases
  at interior cuts.
- Increasing the coupling transfers weight from the leading singular mode to
  subleading modes.

As shape-only checks, the updated Figure 2 and matched earlier landscape have a
log-value correlation of `0.9988`. The updated and earlier Figure 3 entropy
sweeps have correlations of `0.9647` at cut 5 and `0.9667` at cut 10 over
`J>=0.5`.

The absolute entropies and effective mode numbers are smaller than in the
centered XYZ figures. The stored matched centered-XYZ diagnostics reproduce the
earlier Figure 3, so the scale reduction comes from removing centering and
adding the identity channel. Matrix transposition does not change the singular
spectrum. Complete-record weighting also does not change these two benchmarks
because their future controls are deterministic.

Figure 4 retains the broad weak-to-intermediate-coupling shape, with log-value
correlations of `0.9704`, `0.9701`, and `0.9841` at cuts 1, 2, and 3. It does
not retain the earlier strong-coupling interpretation. The updated response
entropy peaks near `J=4.4-4.6` at the interior cuts and then decreases, remains
below the PT-MPO entropy, and redistributes less selected spectral weight than
the PT-MPO spectrum between `J=0.1` and `J=4`. The current Figure 4 caption and
discussion therefore need revision; rescaling cannot restore their previous
claim.

## Replot the stored results

Run these commands from the repository root:

```bash
python updated_experiments/experiments/cut_vs_j.py \
  --plot-heatmap-only \
  --out-dir updated_experiments/results/figure2

python updated_experiments/experiments/modes.py \
  --plot-only \
  --out-dir updated_experiments/results/figure3

python -m updated_experiments.experiments.mpo_comparison.run \
  --plot-only \
  --output-dir updated_experiments/results/figure4

python -m updated_experiments.experiments.probe_budget \
  --plot-only \
  --output-dir updated_experiments/results/probe_budget

python -m updated_experiments.experiments.finite_size \
  --plot-only \
  --output-dir updated_experiments/results/finite_size

python -m updated_experiments.experiments.memory_persistence \
  --plot-only \
  --output-dir updated_experiments/results/figure5

python -m updated_experiments.experiments.quantum_memory_witness \
  --plot-only \
  --output-dir updated_experiments/results/quantum_memory_witness
```

## Repeat the simulations

The simulation commands require the implementation from
`response-matrix-update`. The stored Figure 2--5 and appendix runs used
implementation commit `c6744d1f` with experiment commit `1919c615` through
temporary integration commits recorded in their `run_manifest.json` files. The
quantum-memory manifest separately records the exact imported YAQS source file,
source hash, and implementation commit.

```bash
python updated_experiments/experiments/cut_vs_j.py \
  --n-pasts 64 \
  --n-futures 64 \
  --seed 0 \
  --n-seeds 1 \
  --unitary-ensemble haar \
  --parallel \
  --max-workers 8 \
  --save-raw \
  --out-dir results/updated_figure2

python updated_experiments/experiments/modes.py \
  --m-spectrum 64 \
  --cuts 1,5,10 \
  --seed 0 \
  --n-seeds 1 \
  --spectrum-draws 1 \
  --parallel \
  --max-workers 8 \
  --save-raw \
  --out-dir results/updated_figure3

python -m updated_experiments.experiments.mpo_comparison.run \
  --output-dir results/updated_figure4

python -m updated_experiments.experiments.probe_budget \
  --parallel \
  --max-workers 8 \
  --output-dir results/updated_probe_budget

python -m updated_experiments.experiments.finite_size \
  --parallel \
  --max-workers 8 \
  --output-dir results/updated_finite_size

python -m updated_experiments.experiments.memory_persistence \
  --parallel \
  --max-workers 8 \
  --output-dir results/updated_figure5

python -m updated_experiments.experiments.quantum_memory_witness \
  --output-dir updated_experiments/results/quantum_memory_witness
```

The complete per-point raw matrices occupy approximately 185 MB and are not
tracked here. The CSV, JSON, and NPZ files contain the data required to redraw
Figures 2 and 3. The manifests and summary rows retain the names and SHA-256
hashes of the raw verification files from the recorded runs.

The exact dense process tensors used for Figure 4 are also not retained. Its CSV
and NPZ contain every value needed to redraw the figure, and its manifest
records the implementation commit, configuration, timings, and artifact hashes.
