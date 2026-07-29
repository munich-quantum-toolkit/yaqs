# Hybrid vs full gate-local TDVP — difference report

**Date:** 2026-07-25  
**Question:** What changes when nearest-neighbour (NN) two-qubit gates also
go through the gate-local TDVP window update (`gate_mode="full-tdvp"`,
method `full_tdvp`, \(n=2\)), instead of the hybrid routing (TEBD for NN,
TDVP only for long-range)?

## Scope of the re-run

| Figure | Effect of switching to full TDVP |
| --- | --- |
| Fig 1 (schematic) | Unchanged (conceptual). |
| Fig 2 (single-gate) | **Unchanged.** Gates act on sites \((2,9)\) (separation 7). Hybrid and full share the same TDVP window path for non-NN gates. |
| Fig 3 (circuits) | **2D panels (c,d) re-plotted with `full_tdvp`.** 1D panels (a,b) already used full TDVP. |
| Fig S1, S2 | Unchanged (single-gate grids / substeps). |
| Fig S3 (resources) | **2D columns re-plotted with `full_tdvp`.** |
| Diagnostic overlay | New: `figures/fig_hybrid_vs_full_tdvp.{pdf,png}` (hybrid vs full vs zip-up on 2D). |

New raw data: `raw_new/circuits_2d_full_tdvp/{ising,heisenberg}_chi32_full_tdvp.csv`
(~7 min wall: Ising 162 s, Heisenberg 394 s). Hybrid trajectories retained
for comparison.

Gate mix per second-order Trotter step on the \(4\times4\) snake lattice:

| Model | 1-qubit | NN 2-qubit | Long-range 2-qubit |
| --- | ---: | ---: | ---: |
| TFIM | 16 | **30** | 18 |
| Heisenberg | 0 | **90** | 54 |

So hybrid applied TDVP to only ~38% of the two-qubit gates; full applies it to all of them.

## Quantitative differences (χ_max = 32)

### 2D transverse-field Ising — material improvement

| \(t\) | hybrid \(1-F\) | full \(1-F\) | full / hybrid |
| ---: | ---: | ---: | ---: |
| 0.1 | \(1.95\times10^{-5}\) | \(1.19\times10^{-5}\) | 0.61 |
| 0.5 | \(4.48\times10^{-4}\) | \(1.98\times10^{-4}\) | 0.44 |
| 1.0 | \(1.82\times10^{-3}\) | \(6.44\times10^{-4}\) | 0.35 |
| 2.0 | \(8.72\times10^{-2}\) | \(2.61\times10^{-2}\) | 0.30 |
| 3.0 | \(1.69\times10^{-1}\) | \(5.39\times10^{-2}\) | 0.32 |

- ε = 10⁻² crossing: hybrid at \(t=1.4\), full at \(t=1.6\).
- Max \(|\Delta(1-F)|\) over the trajectory: \(0.115\); median \(0.023\).
- Mean signed \(\Delta\) (full − hybrid): \(-0.040\) (full systematically lower).
- Bond-dimension saturation: both hit χ = 32 early (hybrid \(t=0.4\), full \(t=0.3\)); resource curves remain similar once the cap binds.
- Relative to MPO zip-up: under hybrid, zip-up eventually overtook TDVP after \(t\sim1.5\); under full TDVP, gate-local TDVP stays competitive with (and often below) zip-up through mid-to-late times on this lattice.

**Interpretation.** On TFIM, most two-qubit gates are NN. Routing those through the projected TDVP update (instead of a truncated TEBD contraction) reduces accumulated projection / truncation error enough to cut late-time infidelity by roughly a factor of three at this cap.

### 2D Heisenberg — modest improvement, still a hard regime

| \(t\) | hybrid \(1-F\) | full \(1-F\) | full / hybrid |
| ---: | ---: | ---: | ---: |
| 0.1 | \(1.81\times10^{-2}\) | \(1.84\times10^{-2}\) | 1.02 |
| 0.5 | \(0.300\) | \(0.224\) | 0.75 |
| 1.0 | \(0.600\) | \(0.494\) | 0.82 |
| 2.0 | \(0.786\) | \(0.669\) | 0.85 |
| 3.0 | \(0.843\) | \(0.732\) | 0.87 |

- Both cross ε = 10⁻² at the first step (\(t=0.1\)).
- Max \(|\Delta(1-F)|\): \(0.122\); median \(0.108\).
- Full is better after the first step, but both trajectories are already deep in the unreliable regime; MPO zip-up remains the most accurate early on.
- Bond dimension saturates at χ = 32 by \(t=0.1\) for both.

**Interpretation.** Entanglement growth is so rapid that the NN-gate update choice does not change the qualitative story: fixed-χ compressed dynamics fail early. Full TDVP trims late-time infidelity by ~10–25% relative, which does not restore a usable horizon.

### 1D circuits — already full TDVP (no change)

The 1D TFIM and 1D XXX panels were already generated with `full_tdvp` (all gates NN). Re-plotting leaves them unchanged. In 1D, TEBD+SWAP and MPO zip-up coincide (direct SVD); full TDVP is the method that differs from that baseline, and that is what Fig 3(a,b) and Fig S3(a,b,e,f) already show.

### Single-gate (Fig 2) — no change

Long-range Pauli rotations on \((2,9)\): hybrid and full are the same code path. Re-running Fig 2 / S1 / S2 produces identical figures.

## Effect on the paper storyline

1. **Claim “gate-local TDVP is a valid routing-free update”** is unchanged; full TDVP strengthens the numerical evidence on 2D TFIM by removing the hybrid’s silent TEBD fallback on NN bonds.
2. **Favorable vs difficult regimes.** The favorable 2D TFIM regime becomes *more* favorable under full TDVP (later ε-crossing, ~3× lower final \(1-F\)). The difficult 2D Heisenberg regime remains difficult — full TDVP does not create a usable horizon there.
3. **Hybrid vs full as a methods choice.** Hybrid was a pragmatic optimization (cheap TEBD on NN). On this TFIM instance it was also *less accurate*. For a methods-and-analysis paper that wants a clean statement of the projected-dynamics update, full TDVP is the more consistent definition; hybrid can be retained as an optional performance trade-off, not as the primary production method.
4. **Resource story (Fig S3).** Switching 2D to full TDVP does not materially change the peak-bond / peak-parameter plots once χ_max binds; the main visible change is in infidelity (Fig 3), not in the resource envelope.

## Files

- Raw: `raw_new/circuits_2d_full_tdvp/`
- Comparison CSVs: `processed/hybrid_vs_full_tdvp_{ising,heisenberg}.csv`
- Overlay figure: `figures/fig_hybrid_vs_full_tdvp.{pdf,png}`
- Updated main figures: `figures/fig3_circuits.*`, `figures/figS3_circuit_resources.*`
