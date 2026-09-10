# Updated response-matrix experiments

This directory contains the Figure 2 and Figure 3 experiments after the
response-matrix update. The updated matrix is uncentered, uses future-response
rows and history columns, includes the `I, X, Y, Z` channels, and uses complete
retained-record probabilities.

The stored results use the paper settings: `L=6`, `k=20`, `dt=0.1`, `g=1`, the
initial state `|0>^6`, 64 history probes, 64 future probes, seed 0, and one
Haar-probe draw. The Figure 2 sweep uses all cuts from 1 through 20. Both sweeps
use `J=0,0.05,...,2`.

## Contents

- `experiments/common.py`: shared simulation and plotting functions.
- `experiments/cut_vs_j.py`: Figure 2 cut-resolved memory landscape.
- `experiments/modes.py`: Figure 3 effective modes and singular spectra.
- `results/figure2`: the Figure 2 PDF and PNG, scalar data, initial state, and
  run manifest.
- `results/figure3`: the Figure 3 PDF and PNG, scalar data, full plotted
  spectra, initial state, and run manifest.

The Figure 2 logarithmic color range is `1e-4` to `5e-1`, which preserves the
visual balance of the earlier figure at the updated entropy scale. The Figure 3
main axis scales automatically from the updated effective-mode values. Its inset
keeps the full normalized squared singular-value spectrum and displays values
down to `1e-16`.

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

## Replot the stored results

Run these commands from the repository root:

```bash
python updated_experiments/experiments/cut_vs_j.py \
  --plot-heatmap-only \
  --out-dir updated_experiments/results/figure2

python updated_experiments/experiments/modes.py \
  --plot-only \
  --out-dir updated_experiments/results/figure3
```

## Repeat the simulations

The simulation commands require the implementation from
`response-matrix-update`. The stored runs used implementation commit `c6744d1f`
with experiment commit `1919c615` through temporary integration commits recorded
in each `run_manifest.json`.

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
```

The complete per-point raw matrices occupy approximately 185 MB and are not
tracked here. The CSV, JSON, and NPZ files contain the data required to redraw
both figures. The manifests and summary rows retain the names and SHA-256 hashes
of the raw verification files from the recorded runs.
