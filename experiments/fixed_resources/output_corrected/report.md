# Corrected fixed-χ circuit benchmark report

**Archive:** `archive/pre_repair_20260723T145513Z/`  
**Corrected data:** `output_corrected/`  
**Figure:** `experiments/figures/figure_circuit_fixed_chi.{pdf,png}`

This is a **fixed bond-dimension-cap** comparison against the identical dense Trotter
circuit. It is **not** a fixed-memory or runtime-frontier study.

## Preconditions

- Single-gate repair suite `tests/experiments/test_single_gate_repairs.py`: **11 passed**
  (right-canonical `MPS.compress`, SVD cutoff `1e-14`, infinitesimal continuity,
  no silent shape exceptions).
- All circuit methods regenerated from scratch with deep-copied initial states.

## TDVP subdivision choice

Prior `circuit_subdivision_check.md` only documented a failed import and did not
use this figure configuration, so the grid was re-run.

| n | TFIM \(n_\varepsilon\) (χ=32) | Heisenberg \(n_\varepsilon\) (χ=32) | Heis \(1{-}F(\Delta t)\) |
|--:|--:|--:|--:|
| 1 | 12 | 0 | 2.16×10⁻² |
| 2 | 13 | 0 | 1.81×10⁻² |
| 4 | 13 | 0 | 1.64×10⁻² |
| 8 | 13 | 0 | 1.58×10⁻² |

**Chosen uniform \(n=2\)**: smallest \(n\) with exact horizon agreement under doubling
(\(n=2\) vs \(n=4\): ΔTFIM=0, ΔHeis=0). \(n=1\) shortens the TFIM horizon by one
sampled step; \(n\ge4\) does not change the conclusion. Source:
`circuit_subdivision_validation.csv`, `subdivision_choice.json`.

Hybrid routing (unchanged): nearest-neighbour two-qubit gates → TEBD; non-adjacent
gates → 2-site TDVP with `tdvp_sweeps=n`.

## Horizon definition (corrected)

\[
T_\varepsilon=\max\{t_m:\,1{-}F(t_j)<\varepsilon\ \forall j\le m\},
\]
with \(T_\varepsilon=0\) if the first circuit step fails, and right-censoring if the
run ends before crossing. Main \(\varepsilon=10^{-2}\).  
**Note:** the archived CSV used \(T_\varepsilon=\) first-crossing time (one step later).

## Old vs corrected TFIM \(T_\varepsilon\) (\(\varepsilon=10^{-2}\))

Archived “old \(T\)” below is the previous **first-crossing** time; corrected values
use the last-reliable definition. Qualitative comparison uses the corrected numbers.

| χ | Old TDVP (cross) | Corr. TDVP \(T_\varepsilon\) | Old zip-up (cross) | Corr. zip-up \(T_\varepsilon\) | Old TEBD (cross) | Corr. TEBD |
|--:|--:|--:|--:|--:|--:|
| 8 | 0.7 | **0.5** | 0.3 | **0.6** | 0.2 | **0.1** |
| 16 | 1.0 | **0.9** | 0.5 | **1.0** | 0.3 | **0.2** |
| 32 | 1.3 | **1.3** | 0.4 | **1.5** | 0.3 | **0.2** |
| 48 | 1.5 | **1.4** | 0.6 | **2.0** | 0.3 | **0.2** |
| 64 | 1.6 | **1.7** | 0.6 | **2.9** | 0.3 | **0.2** |

At χ=32 trajectory (panel b): zip-up crosses after TDVP; TEBD fails by \(t\approx0.3\).

## Does the central fixed-χ conclusion survive?

**No — not in the previous form.**

Previous claim: local TDVP extends the reliable TFIM horizon relative to routed TEBD
and MPO zip-up under a fixed χcap.

Corrected finding:

1. TDVP still **dominates TEBD+SWAP** on TFIM at every simulated χ.
2. After repairing `MPS.compress` / zip-up, **MPO zip-up meets or exceeds TDVP**
   on TFIM for \(\chi_{\max}\gtrsim 8\) and clearly leads at large χ
   (e.g. χ=64: zip-up \(T_\varepsilon=2.9\) vs TDVP \(1.7\)).
3. The old zip-up horizons were **artefactually short**; they must not be cited.

Defensible revised claim: under a fixed χcap, hybrid TDVP remains a stable
compressed circuit update that beats swap-routed TEBD, but **repaired MPO zip-up
is competitive and often superior on this TFIM benchmark**; method ranking must
use the repaired implementations.

## Heisenberg (panel c)

Corrected data **no longer** support an all-methods one-step failure plot.

| Method | Behaviour |
|---|---|
| TDVP | \(T_\varepsilon=0\) for all χ≤128 (first-step \(1{-}F>\varepsilon\); plateau ~1.8×10⁻²) |
| TEBD+SWAP | \(T_\varepsilon=0\) until χ=128 (\(T_\varepsilon=0.1\)) |
| MPO zip-up | Multi-step horizons from χ≳12; \(T_\varepsilon=0.6\) at χ=128 |

Panel (c) is therefore a **horizon vs χ** plot. TDVP remains the limitation/control
on Heisenberg; repaired zip-up develops a short but real horizon.

## Threshold sensitivity

See `threshold_sensitivity.csv`. At \(\varepsilon\in\{10^{-3},10^{-2},10^{-1}\}\) the
TFIM ordering **TEBD ≪ TDVP ≲ zip-up** (zip-up ≥ TDVP at mid/high χ) is stable;
absolute horizons shrink/grow with ε as expected.

## Validation

- Exact reference: dense identical Trotter circuit (`exact_*_t30.npy`).
- χ=256 Ising control (2 steps): TEBD/MPO within `1e-8`; TDVP within `1e-3`.
- Deterministic repeat of TFIM χ=32, 5 steps: bit-stable infidelities.
- Horizons derived from raw `circuit_results_corrected.csv`, not from the figure.

## Variational MPO control

Attempted on TFIM χ=32 and Heisenberg first step over the χ grid
(`variational_circuit_control.csv`, first pass). Local residual-monotonicity
checks aborted mid-gate (numerical residual noise), so that pass did **not**
produce a trustworthy circuit horizon. A zip-up-fallback rerun was started but
stopped for wall-clock; it is **not** required for the main conclusion because:

- even a successful variational fit is not expected to beat repaired zip-up by
  enough to change the TFIM headline (zip-up already leads TDVP);
- the main figure therefore retains **three methods** (TDVP, TEBD+SWAP, zip-up).

Do **not** promote variational into the main panel from the aborted control.

## Proposed caption

> Reliable circuit simulation under a fixed bond-dimension cap on 4×4 lattices
> (exact reference = identical second-order Trotter circuit; Δt=0.1; TDVP uses
> n=2 fractional-time substeps on long-range gates only).
> (a) TFIM reliability horizon \(T_\varepsilon\) at \(\varepsilon=10^{-2}\) versus
> \(\chi_{\max}\). After repairing MPO compression, zip-up meets or exceeds TDVP
> for \(\chi_{\max}\gtrsim8\), while TEBD+SWAP remains short-horizon.
> (b) TFIM infidelity versus time at \(\chi_{\max}=32\), with the ε threshold marked.
> (c) Heisenberg \(T_\varepsilon\) versus \(\chi_{\max}\): TDVP stays at
> \(T_\varepsilon=0\) up to χ=128, whereas repaired zip-up develops a multi-step
> horizon. This panel is a fixed-χ comparison, not a fixed-memory study.
