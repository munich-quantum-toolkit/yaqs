# Resource frontier validation

## Figure caption

Resource requirements for reliable 4×4 TFIM circuit simulation. For each target time, the plotted value is minimized over all tested bond-dimension caps satisfying 1-F<10⁻² at every preceding Trotter step. (a) Minimum peak retained MPS parameter count. (b) Measured runtime trade-off, using median wall-clock timings from three isolated repetitions on fixed hardware. TDVP combines a smaller retained representation with competitive runtime at intermediate times, while its measured cost rises sharply at later times when the runtime-minimizing reliable χmax increases (32→48 at t=1.3; 48→64 at t=1.5).

## Panel roles

- **(a)** measures retained MPS representation size (Pmax), not process RSS.
- **(b)** is the **measured runtime trade-off**: actual wall-clock medians from three isolated repetitions on fixed hardware with fixed thread settings (see Configuration). It is not a parameter-derived or theoretical complexity estimate.

## Definition of Pmax

$P_{\max}=\max_t \sum_i d_i\,\chi_{i-1}(t)\,\chi_i(t)$.

The maximum is evaluated over every retained MPS after every gate, not only at complete Trotter-step boundaries. This quantity measures representation size of the stored MPS and is independent of transient process RSS.

## Configuration
- Benchmark: `resource_frontier_tfim_4x4`
- Grid: 4×4 TFIM, Δt=0.1, ε=0.01, target steps=1…15
- Methods: hybrid_tdvp, tebd_swap, mpo_zipup (no variational MPO)
- Git commit: `ceab7c207d25d9dc869943dc8b82f4488973a91d`
- Platform: `Linux-6.8.0-90-generic-x86_64-with-glibc2.39` / `x86_64`
- Python: `3.14.2`
- Packages: numpy=2.4.6, qiskit=2.5.0, mqt.yaqs=0.6.1.dev2+gceab7c207
- Thread env: `{'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}`

## Data provenance
- Cross-check ingest: 48 χ=32 TFIM rows from trajectories.csv → tag=ingest_ref (not used for frontier peaks; main runs regenerated for gate-level peak accounting). | Other χ∈CHI_INGEST had no retained per-step files after fixed_resources cleanup; regenerated with matching 4×4 TFIM configuration.
- Ingest cross-check rows (tag=ingest_ref): 48
- Main generated rows (tag=main): 253
- CHI_INGEST regenerated: [2, 4, 8, 12, 16, 24, 32, 48, 64]
- High-χ TEBD/MPO values present: [96, 128, 192, 256]
- Timing repeat rows: 747

## Reachability
- hybrid_tdvp: max reliable n=15 (need 15)
- tebd_swap: max reliable n=15 (need 15)
- mpo_zipup: max reliable n=15 (need 15)
- All methods reach step 15: yes

## MPS representation frontier (Pmax)

Primary representation-size result: minimum peak retained MPS parameter count among reliable χmax.

### n=1 (t=0.1)
- hybrid_tdvp: χ=2, Pmax=120
- tebd_swap: χ=2, Pmax=120
- mpo_zipup: χ=8, Pmax=1448
- Parameter ratio tebd_swap/TDVP @ n=1: 1.000
- Parameter ratio mpo_zipup/TDVP @ n=1: 12.067
### n=5 (t=0.5)
- hybrid_tdvp: χ=8, Pmax=1448
- tebd_swap: χ=192, Pmax=141992
- mpo_zipup: χ=64, Pmax=56808
- Parameter ratio tebd_swap/TDVP @ n=5: 98.061
- Parameter ratio mpo_zipup/TDVP @ n=5: 39.232
### n=10 (t=1)
- hybrid_tdvp: χ=24, Pmax=9128
- tebd_swap: χ=256, Pmax=174760
- mpo_zipup: χ=192, Pmax=286184
- Parameter ratio tebd_swap/TDVP @ n=10: 19.145
- Parameter ratio mpo_zipup/TDVP @ n=10: 31.352
### n=15 (t=1.5)
- hybrid_tdvp: χ=64, Pmax=43688
- tebd_swap: χ=256, Pmax=174760
- mpo_zipup: χ=256, Pmax=433640
- Parameter ratio tebd_swap/TDVP @ n=15: 4.000
- Parameter ratio mpo_zipup/TDVP @ n=15: 9.926

### Matched-time parameter ratios (summary)
- At t=0.5: TEBD/MPO require approximately 39–98× more parameters than TDVP.
- At t=1.0: approximately 19–31× more parameters than TDVP.
- At t=1.5: approximately 4–10× more parameters than TDVP.

## Measured runtime trade-off

Panel (b) reports median cumulative wall-clock runtime over three controlled repetitions (IQR shown in the figure).

### n=1 (t=0.1)
- hybrid_tdvp: χ=24, R*=0.0989 s (source=timing_median)
- tebd_swap: χ=2, R*=0.02739 s (source=timing_median)
- mpo_zipup: χ=128, R*=0.04825 s (source=timing_median)
- Runtime ratio tebd_swap/TDVP @ n=1: 0.277
- Runtime ratio mpo_zipup/TDVP @ n=1: 0.488
### n=5 (t=0.5)
- hybrid_tdvp: χ=12, R*=0.7546 s (source=timing_median)
- tebd_swap: χ=192, R*=1.808 s (source=timing_median)
- mpo_zipup: χ=64, R*=1.555 s (source=timing_median)
- Runtime ratio tebd_swap/TDVP @ n=5: 2.396
- Runtime ratio mpo_zipup/TDVP @ n=5: 2.061
### n=10 (t=1)
- hybrid_tdvp: χ=24, R*=3.541 s (source=timing_median)
- tebd_swap: χ=256, R*=5 s (source=timing_median)
- mpo_zipup: χ=192, R*=21.1 s (source=timing_median)
- Runtime ratio tebd_swap/TDVP @ n=10: 1.412
- Runtime ratio mpo_zipup/TDVP @ n=10: 5.959
### n=15 (t=1.5)
- hybrid_tdvp: χ=64, R*=263.5 s (source=timing_median)
- tebd_swap: χ=256, R*=8.208 s (source=timing_median)
- mpo_zipup: χ=256, R*=52.93 s (source=timing_median)
- Runtime ratio tebd_swap/TDVP @ n=15: 0.031
- Runtime ratio mpo_zipup/TDVP @ n=15: 0.201

## Late-time TDVP measured-runtime increase

Diagnostic based solely on existing timing repetitions (no new simulations): `tdvp_late_runtime_diagnostic.csv` / `.md`.

Outcome B

At t=1.3 and t=1.5 the runtime-minimizing reliable TDVP χmax increases (32→48 and 48→64). Within each selected configuration all three repetitions agree (IQR ≪ the frontier jump). The increase is therefore a **configuration-switch effect (outcome B)**, not a single-repetition timing artifact. It is associated with the larger selected χmax and with growth of retained bonds and effective local TDVP problems, and is **consistent with the increasing cost of local TDVP updates** (no operation-level profiling is claimed).

## Interpretation

- TDVP reaches matched reliable times with a substantially smaller retained MPS representation.
- Through the intermediate-time regime, TDVP is both compact and competitive in measured runtime.
- At later times, TDVP still retains the smallest MPS but becomes slower than TEBD+SWAP and MPO zip-up.
- The representation-size and runtime frontiers quantify distinct resources. TDVP’s compact representation does not guarantee lower wall-clock cost once the local projected updates become expensive.

Avoided claims: smaller MPS ⇒ faster simulation; parameter count predicts runtime; TDVP is uniformly faster; the runtime frontier is an abstract complexity estimate.

## Implementation-specific working-memory diagnostic

Process RSS is a supplementary, implementation-specific diagnostic. It is **not** used to construct or validate the MPS representation frontier or the measured runtime trade-off. RSS includes transient Krylov, contraction and SVD workspace and may reverse the ordering implied by retained MPS parameters.

- hybrid_tdvp/χ=64: ΔRSS=627.719 MiB, retained MPS parameters=43688 (0.6666 MiB equivalent at 16 B/element)
- tebd_swap/χ=256: ΔRSS=42.281 MiB, retained MPS parameters=174760 (2.6666 MiB equivalent at 16 B/element)
- mpo_zipup/χ=256: ΔRSS=113.965 MiB, retained MPS parameters=433640 (6.6168 MiB equivalent at 16 B/element)
- Ordering note: DISAGREE: MPS order ['hybrid_tdvp', 'tebd_swap', 'mpo_zipup'] vs RSS order ['tebd_swap', 'mpo_zipup', 'hybrid_tdvp']; do not claim overall peak-memory advantage from MPS parameters alone.
- Because RSS ordering disagrees with Pmax ordering, do not claim that TDVP uses less total peak process memory.

## Failures / exclusions
- Failed step rows: 0
- MPS-representation frontier nondecreasing violations: 0
- Runtime frontier nondecreasing violations: 0
