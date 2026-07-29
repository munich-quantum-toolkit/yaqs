# Generator-rank pilot: compact generator (D_H) vs high-rank layer unitary (D_U)

Automated screening of `pilot_results.csv` against the pre-registered decision rule. See the run log and `meta.json` for validations and fixed numerical settings.

## Fixed settings

- `svd_threshold` = 1e-13
- `trunc_mode` = discarded_weight
- `krylov_tol` = 1e-12
- `gate_library_split_tensor_hard_cutoff` = 1e-14
- `tdvp_mode` = 2site
- `layer_mpo_tol` = 1e-12
- `exact_gatewise_threshold` = 1e-15
- BLAS/OMP threads pinned to 1 for all timed runs
- angle conventions: {
"qaoa": "edge Rzz(theta)=exp(-i theta/2 ZZ); layer U_C=exp(-i gamma H_C) with gamma=theta/2; angle column = theta (0.3, 0.7) => gamma in {0.15, 0.35}",
"oat": "pair Rxx(theta)=exp(-i theta/2 XX) with theta=2*kappa/(N-1); angle column = kappa"
}

## Rank separation actually realized

| candidate | size | angle | D_H | D_U (tol 1e-12) | exact state max bond |
|---|---|---|---|---|---|
| oat | N16 | 0.5 | 3 | 9 | 8 |
| oat | N16 | 1.5 | 3 | 9 | 9 |
| oat | N20 | 0.5 | 3 | 11 | 8 |
| oat | N20 | 1.5 | 3 | 47 | 11 |
| qaoa | 4x4 | 0.3 | 6 | 16 | 16 |
| qaoa | 4x4 | 0.7 | 6 | 16 | 16 |
| qaoa | 5x5 | 0.3 | 7 | 32 | 32 |
| qaoa | 5x5 | 0.7 | 7 | 32 | 32 |

## Per-configuration screening (converged TDVP vs best non-SWAP MPO/layer baseline)

Rule: PASS requires >=2x lower peak params at matched-or-better error with runtime within 2x, or >=2x lower runtime at matched error/memory, or >=10x accuracy at matched resources. Oracle (`oracle_compress`) is a representability diagnostic, not a baseline.

### oat N16 angle=0.5

- chi=2: TDVP(n=8) inf=8.23e-03 (Δ vs n/2: 1.8e-04) vs best baseline mpo_layer/ inf=7.24e-03; acc x0.88, peak-param x8, transient x60, runtime x34 -> no
- chi=4: TDVP(n=8) inf=9.68e-06 (Δ vs n/2: 2.8e-06) vs best baseline mpo_layer/ inf=9.40e-06; acc x0.97, peak-param x2.4, transient x15, runtime x40 -> no
- chi=8: TDVP(n=8) inf=2.12e-07 (Δ vs n/2: 3.2e-06) vs best baseline mpo_layer/ inf=1.00e-16; acc x4.7e-10, peak-param x1, transient x4.3, runtime x43 -> no
- chi=16: TDVP(n=8) inf=2.12e-07 (Δ vs n/2: 3.2e-06) vs best baseline mpo_layer/ inf=1.00e-16; acc x4.7e-10, peak-param x1, transient x4.3, runtime x43 -> no

### oat N16 angle=1.5

- chi=2: TDVP(n=8) inf=2.34e-01 (Δ vs n/2: 3.8e-03) vs best baseline mpo_layer/ inf=1.74e-01; acc x0.75, peak-param x8, transient x60, runtime x34 -> no
- chi=4: TDVP(n=8) inf=2.00e-02 (Δ vs n/2: 5.5e-04) vs best baseline mpo_layer/ inf=1.37e-02; acc x0.69, peak-param x2.3, transient x15, runtime x45 -> no
- chi=8: TDVP(n=8) inf=1.77e-05 (Δ vs n/2: 2.6e-04) vs best baseline mpo_layer/ inf=5.56e-07; acc x0.031, peak-param x0.75, transient x3.8, runtime x54 -> no
- chi=16: TDVP(n=8) inf=1.72e-05 (Δ vs n/2: 2.6e-04) vs best baseline mpo_gatewise/lexicographic inf=1.00e-16; acc x5.8e-12, peak-param x1.3, transient x18, runtime x0.5 -> no

### oat N20 angle=0.5

- chi=2: TDVP(n=8) inf=8.20e-03 (Δ vs n/2: 6.0e-04) vs best baseline mpo_layer/ inf=7.36e-03; acc x0.9, peak-param x12, transient x1.1e+02, runtime x35 -> no
- chi=4: TDVP(n=8) inf=1.34e-05 (Δ vs n/2: 2.8e-06) vs best baseline mpo_layer/ inf=1.30e-05; acc x0.97, peak-param x3.3, transient x28, runtime x40 -> no
- chi=8: TDVP(n=8) inf=2.18e-07 (Δ vs n/2: 3.3e-06) vs best baseline mpo_layer/ inf=1.47e-12; acc x6.7e-06, peak-param x1.2, transient x6.9, runtime x44 -> no
- chi=16: TDVP(n=8) inf=2.18e-07 (Δ vs n/2: 3.3e-06) vs best baseline mpo_layer/ inf=8.44e-13; acc x3.9e-06, peak-param x1.2, transient x6.9, runtime x43 -> no
- chi=32: TDVP(n=16) inf=1.37e-08 (Δ vs n/2: 2.0e-07) vs best baseline mpo_layer/ inf=8.44e-13; acc x6.1e-05, peak-param x1.2, transient x6.9, runtime x82 -> no

### oat N20 angle=1.5

- chi=2: TDVP(n=8) inf=2.30e-01 (Δ vs n/2: 2.1e-03) vs best baseline mpo_layer/ inf=1.72e-01; acc x0.75, peak-param x1.6e+02, transient x1.5e+03, runtime x26 -> no
- chi=4: TDVP(n=8) inf=2.22e-02 (Δ vs n/2: 9.2e-05) vs best baseline mpo_layer/ inf=1.47e-02; acc x0.67, peak-param x46, transient x3.9e+02, runtime x33 -> no
- chi=8: TDVP(n=8) inf=2.97e-05 (Δ vs n/2: 2.6e-04) vs best baseline mpo_layer/ inf=1.15e-05; acc x0.39, peak-param x14, transient x97, runtime x39 -> no
- chi=16: TDVP(n=8) inf=1.77e-05 (Δ vs n/2: 2.7e-04) vs best baseline mpo_layer/ inf=4.55e-15; acc x2.6e-10, peak-param x9.3, transient x47, runtime x39 -> no
- chi=32: TDVP(n=8) inf=1.77e-05 (Δ vs n/2: 2.7e-04) vs best baseline mpo_gatewise/lexicographic inf=1.00e-16; acc x5.7e-12, peak-param x1.5, transient x27, runtime x0.31 -> no

### qaoa 4x4 angle=0.3

- chi=2: TDVP(n=8) inf=1.84e-01 (Δ vs n/2: 3.0e-15) vs best baseline mpo_layer/ inf=1.86e-01; acc x1, peak-param x40, transient x3e+02, runtime x28 -> no
- chi=4: TDVP(n=8) inf=1.27e-01 (Δ vs n/2: 6.9e-04) vs best baseline mpo_layer/ inf=8.77e-02; acc x0.69, peak-param x28, transient x1.5e+02, runtime x28 -> no
- chi=8: TDVP(n=8) inf=1.27e-01 (Δ vs n/2: 6.9e-04) vs best baseline mpo_layer/ inf=5.84e-03; acc x0.046, peak-param x28, transient x1.5e+02, runtime x27 -> no
- chi=16: TDVP(n=8) inf=1.27e-01 (Δ vs n/2: 6.9e-04) vs best baseline mpo_gatewise/horiz_first inf=1.00e-16; acc x7.9e-16, peak-param x28, transient x1.5e+02, runtime x3 -> no
- chi=32: TDVP(n=8) inf=1.27e-01 (Δ vs n/2: 6.9e-04) vs best baseline mpo_gatewise/horiz_first inf=1.00e-16; acc x7.9e-16, peak-param x28, transient x1.5e+02, runtime x3 -> no

### qaoa 4x4 angle=0.7

- chi=2: TDVP(n=8) inf=6.76e-01 (Δ vs n/2: 2.4e-15) vs best baseline mpo_layer/ inf=6.86e-01; acc x1, peak-param x40, transient x3e+02, runtime x28 -> no
- chi=4: TDVP(n=8) inf=5.29e-01 (Δ vs n/2: 2.0e-03) vs best baseline mpo_layer/ inf=4.27e-01; acc x0.81, peak-param x28, transient x1.5e+02, runtime x28 -> no
- chi=8: TDVP(n=8) inf=5.29e-01 (Δ vs n/2: 2.0e-03) vs best baseline mpo_layer/ inf=1.37e-01; acc x0.26, peak-param x28, transient x1.5e+02, runtime x28 -> no
- chi=16: TDVP(n=8) inf=5.29e-01 (Δ vs n/2: 2.0e-03) vs best baseline mpo_gatewise/vert_first inf=9.21e-15; acc x1.7e-14, peak-param x28, transient x1.8e+02, runtime x3.1 -> no
- chi=32: TDVP(n=8) inf=5.29e-01 (Δ vs n/2: 2.0e-03) vs best baseline mpo_gatewise/vert_first inf=9.21e-15; acc x1.7e-14, peak-param x28, transient x1.8e+02, runtime x3 -> no

### qaoa 5x5 angle=0.3

- chi=4: TDVP(n=8) inf=2.38e-01 (Δ vs n/2: 8.0e-04) vs best baseline mpo_layer/ inf=1.73e-01; acc x0.73, peak-param x1.3e+02, transient x1e+03, runtime x23 -> no
- chi=8: TDVP(n=8) inf=2.38e-01 (Δ vs n/2: 8.0e-04) vs best baseline mpo_layer/ inf=3.02e-02; acc x0.13, peak-param x1.3e+02, transient x1e+03, runtime x21 -> no
- chi=16: TDVP(n=8) inf=2.38e-01 (Δ vs n/2: 8.0e-04) vs best baseline mpo_layer/ inf=1.37e-03; acc x0.0058, peak-param x1.3e+02, transient x1e+03, runtime x17 -> no
- chi=32: TDVP(n=8) inf=2.38e-01 (Δ vs n/2: 8.0e-04) vs best baseline mpo_gatewise/horiz_first inf=1.00e-16; acc x4.2e-16, peak-param x1.3e+02, transient x1e+03, runtime x1.6 -> no
- chi=64: TDVP(n=8) inf=2.38e-01 (Δ vs n/2: 8.0e-04) vs best baseline mpo_gatewise/horiz_first inf=1.00e-16; acc x4.2e-16, peak-param x1.3e+02, transient x1e+03, runtime x1.5 -> no

### qaoa 5x5 angle=0.7

- chi=4: TDVP(n=8) inf=7.78e-01 (Δ vs n/2: 1.3e-03) vs best baseline mpo_layer/ inf=7.06e-01; acc x0.91, peak-param x1.3e+02, transient x1e+03, runtime x23 -> no
- chi=8: TDVP(n=8) inf=7.78e-01 (Δ vs n/2: 1.3e-03) vs best baseline mpo_layer/ inf=4.61e-01; acc x0.59, peak-param x1.3e+02, transient x1e+03, runtime x21 -> no
- chi=16: TDVP(n=8) inf=7.78e-01 (Δ vs n/2: 1.3e-03) vs best baseline mpo_layer/ inf=1.26e-01; acc x0.16, peak-param x1.3e+02, transient x1e+03, runtime x17 -> no
- chi=32: TDVP(n=8) inf=7.78e-01 (Δ vs n/2: 1.3e-03) vs best baseline mpo_gatewise/horiz_first inf=1.00e-16; acc x1.3e-16, peak-param x1.3e+02, transient x1e+03, runtime x1.6 -> no
- chi=64: TDVP(n=8) inf=7.78e-01 (Δ vs n/2: 1.3e-03) vs best baseline mpo_gatewise/horiz_first inf=1.00e-16; acc x1.3e-16, peak-param x1.3e+02, transient x1e+03, runtime x1.5 -> no

## Screen outcome

- Any configuration passing the decision rule: **NO**
- Plots: `infidelity_vs_peak_params.png`, `infidelity_vs_runtime.png`

## Interpretation

Both candidates are **negative** under the pre-registered rule. The intended rank
separation D_H << D_U was genuinely realized (QAOA 5x5: D_H=7 vs D_U=32 = 2^b at
b=5 crossing edges; OAT: D_H=3 vs D_U up to 47), so the hypothesis received a fair
test and failed.

### QAOA/Ising cost layer: TDVP fails by projection, not by resources

Complete-generator 2TDVP stalls at bond profile [2,2,2,4,...] regardless of
chi (4..64) and substeps (converged in n; n=4 vs n=8 delta ~1e-3), plateauing at
infidelity 0.127 (4x4, gamma=0.15) to 0.78 (5x5, gamma=0.35) while gatewise and
complete-layer MPO routes reach the exact state at chi = 2^b. Total discarded
weight during TDVP is ~1e-29 and the stall persists bit-identically at
svd_threshold=1e-30 (see run log), so it is intrinsic tangent-space projection
error, not truncation: for the diagonal ZZ generator acting on |+>^N, the exact
evolution preserves <Z_i>=0, and the projected effective generators transmit
almost no entanglement across bonds, confining the TDVP flow to a low-rank
invariant submanifold. Direct MPO application has no such obstruction. This is
the *opposite* of a TDVP-favoring regime, despite maximal D_H/D_U separation.

### OAT collective entangler: TDVP works, but the complete-layer MPO removes any advantage

Substep-converged TDVP tracks the oracle representability bound well in the
representability-limited regime (e.g. N=20, kappa=0.5, chi=4: TDVP n=8 reaches
1.34e-5 vs oracle 1.30e-5) and decisively beats *sequential gatewise* application
of the 190 equivalent Rxx gates (7.8e-3 at chi=4). However, the complete-layer
MPO applied once and compressed matches the oracle at every chi, at comparable
peak parameter count (the initial state is a product state, so the uncompressed
MPO x MPS intermediate is only D_U wide) and ~25-50x lower wall-clock time than
TDVP n=8. Per the pre-registered rule ("if a complete-layer or oracle compression
removes the apparent advantage, record the candidate as negative"), OAT is
negative. The gatewise-vs-aggregated gap is itself a real observation, but the
winning aggregated method is the MPO layer, not TDVP.

### Cross-chi matched-error check

Equal-chi screening is sufficient here: the complete-layer MPO baseline attains
oracle-level accuracy at every chi in every configuration, so no alternative chi
pairing can produce a TDVP accuracy-resource Pareto advantage; TDVP's only edge
(smaller transient than the MPO x MPS expansion) is offset by 20-80x runtime and
vanishes against the layer route applied to product states.

### Is this the paper's local-TDVP method?

No. Both candidates require the generator MPO to span the entire chain (the 2D
cost layer couples every snake bond; OAT is all-to-all), so the "window" of the
paper's local two-site TDVP construction is the full chain and the algorithm
reduces to standard global MPO-TDVP (a projector-splitting sweep under a
Hamiltonian-like MPO). Even a positive result here would have supported standard
MPO-TDVP practice, not the manuscript's local gate-application contribution.
Combined with the negative screen, this closes the D_H << D_U direction: no
further candidates in this family are warranted, and the PRA rewrite should
proceed on the honest method-selection storyline.
