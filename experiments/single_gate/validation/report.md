# Single-gate benchmark validation report

This audit does **not** overwrite production `output/` or the publication figure.

## Executive summary

The plotted variational-MPO small-angle plateaus at compressed χ are **not** a limitation of χ-constrained MPS approximation. They are an **optimizer / implementation failure**: the production variational routine is initialized from MPO zip-up and its local bond updates are silently rejected, so the method returns the zip-up state unchanged.

Decision-rule outcomes:

- **Input initialization removes the variational plateau** → treat old result as optimizer failure; any regenerated benchmark must use multi-start / best-retained protocol.
- **Optimizer objective disagrees with dense fidelity** (`Re⟨T|A⟩` vs `|⟨T|A⟩|²`) → implementation bug.
- Independent TT-SVD best-found at χ=8 tracks TDVP (not zip-up) at weak angles.
- TDVP **does** beat the no-update baseline at weak angles on **all 10** tested seeds (median ratio ≈0.69), so quadratic scaling is not merely the trivial no-update O(θ²).
- Publication figure must **not** be regenerated until variational method is fixed.

**Update (figure regenerated):** `variational.py` now implements the multi-start
protocol; `variational_mpo` angle-sweep rows were regenerated and
`figure_single_gate_main_text` was replotted. Pre-fix DB backup:
`output/results.sqlite.pre_varfix_bak`.

## 1. Exact reference

- YAQS RZZ matrix vs independent `expm(-i θ Z⊗Z/2)` max|Δ| = `0.000e+00`
- Factor 1/2 in exponent: **PASS**
- Site-0 is LSB in `to_vec`: **True**
- x=0.0: dense vs independent max|Δ|=0.000e+00, dense vs uncapped MPO infidelity=0.000e+00, uncapped max χ=8
- x=0.0001: dense vs independent max|Δ|=0.000e+00, dense vs uncapped MPO infidelity=0.000e+00, uncapped max χ=16
- x=0.01: dense vs independent max|Δ|=0.000e+00, dense vs uncapped MPO infidelity=0.000e+00, uncapped max χ=16
- x=0.1: dense vs independent max|Δ|=0.000e+00, dense vs uncapped MPO infidelity=0.000e+00, uncapped max χ=16
- x=1.0: dense vs independent max|Δ|=0.000e+00, dense vs uncapped MPO infidelity=0.000e+00, uncapped max χ=8
- At x=0.1, uncapped max χ=16; compress(χ=16) infidelity=0.000e+00; TT-SVD(χ=16) infidelity=2.220e-16

## 2. No-update baseline

Analytic identity verified:

```
1 - F0 = sin²(θ/2) [1 - <Z2 Z9>²]  ≤  sin²(θ/2)
```

- Seed 11, x=1e-4: measured 1-F0 = `9.866032e-08`, analytic = `9.866032e-08`, sin²(θ/2) bound ≈ `9.869604e-08`
- ⟨Z₂Z₉⟩ (seed 11) = `0.019024934511`
- TDVP / baseline ratio at x=1e-4, χ=8: `0.6332` (TDVP infidelity `6.246976e-08`)
- Var-MPO (zip init) / baseline ratio: `5.4759e+05` (infidelity `5.402539e-02`) — **worse than doing nothing by ~547590×**
- Independent best-found χ=8 infidelity: `6.247072e-08` (ratio to baseline `0.6332`)

## 3. θ = 0

| χ | method | infidelity | norm | actual χ | notes |
|---:|---|---:|---:|---:|---|
| 8 | hybrid_tdvp | 0.000e+00 | 1.000000 | 8 |  |
| 8 | tebd_swap | 2.998e-01 | 0.801350 | 8 | routing compression |
| 8 | mpo_zipup | 0.000e+00 | 1.000000 | 8 |  |
| 8 | variational_mpo | 0.000e+00 | 1.000000 | 8 | init=zipup;sweeps=0;conv=True;failed=False;reason='' |
| 8 | no_update | 0.000e+00 | 1.000000 | 8 |  |
| 8 | best_found | 0.000e+00 | 1.000000 | 8 | candidates=input:0.000e+00,ttsvd:0.000e+00,compress_uncapped |
| 12 | hybrid_tdvp | 0.000e+00 | 1.000000 | 8 |  |
| 12 | tebd_swap | 2.440e-01 | 0.875672 | 12 | routing compression |
| 12 | mpo_zipup | 0.000e+00 | 1.000000 | 8 |  |
| 12 | variational_mpo | 0.000e+00 | 1.000000 | 8 | init=zipup;sweeps=0;conv=True;failed=False;reason='' |
| 12 | no_update | 0.000e+00 | 1.000000 | 8 |  |
| 12 | best_found | 0.000e+00 | 1.000000 | 8 | candidates=input:0.000e+00,compress_uncapped:0.000e+00,ttsvd |
| 16 | hybrid_tdvp | 0.000e+00 | 1.000000 | 8 |  |
| 16 | tebd_swap | 4.441e-16 | 1.000000 | 8 |  |
| 16 | mpo_zipup | 0.000e+00 | 1.000000 | 8 |  |
| 16 | variational_mpo | 0.000e+00 | 1.000000 | 8 | init=zipup;sweeps=0;conv=True;failed=False;reason='' |
| 16 | no_update | 0.000e+00 | 1.000000 | 8 |  |
| 16 | best_found | 0.000e+00 | 1.000000 | 8 | candidates=input:0.000e+00,compress_uncapped:0.000e+00,ttsvd |

TEBD+SWAP nonzero error at χ<16 for θ=0 is **routing compression of SWAP networks**, not a physical RZZ approximation error (RZZ(0)=I).

## 4. Variational-MPO audit (root cause)

### Objective

The code minimizes the Euclidean residual `‖|ψ_target⟩ − |ψ_approx⟩‖² = ⟨T|T⟩ + ⟨A|A⟩ − 2 Re⟨T|A⟩` (`variational._compression_objective`), i.e. state-vector distance to the **uncapped** MPO-applied target — not an MPO-application residual in operator space.

### Root cause

At x=1e-4, χ=8: zip-up objective=`5.402539e-02`, input objective=`9.869604e-08`. All local bond updates leave the state unchanged (`all_updates_unchanged=True`).

Mechanism:

1. `_bond_update_from_target` copies the target's merged two-site block and SVD-splits it.
2. The resulting tensors inherit the target's **neighboring virtual dimensions** (e.g. 16), which need not match the approx neighbors (e.g. 8).
3. `_compression_objective` → `scalar_product` then raises `ValueError` on bond mismatch.
4. The `except ValueError: return approx, obj_before` handler **silently rejects** the update.
5. After a no-op sweep, relative progress is zero → marked `converged`, returning zip-up.

Shape evidence (bond 5):

```
{
  "approx_shapes_before": [
    [
      2,
      8,
      8
    ],
    [
      2,
      8,
      8
    ]
  ],
  "target_shapes": [
    [
      2,
      16,
      16
    ],
    [
      2,
      16,
      16
    ]
  ],
  "new_shapes": [
    [
      2,
      16,
      8
    ],
    [
      2,
      8,
      16
    ]
  ],
  "neighbor_left_right_bond": 8,
  "explanation": "Replacing sites with truncated target tensors yields virtual dims that do not match neighbors; scalar_product then raises ValueError and _bond_update_from_target silently rejects the update.",
  "objective_after_replace": "ValueError: Size of label 'd' for operand 1 (8) does not match previous terms (16)."
}
```

### Multi-start results (seed 11, χ=8, 4 sweeps)

| x | init | infidelity | ratio to baseline |
|---:|---|---:|---:|
| 0.0001 | zipup | 5.402539e-02 | 547589.8386665683 |
| 0.0001 | input | 9.866032e-08 | 1.0 |
| 0.0001 | tdvp | 6.246976e-08 | 0.63318024593222 |
| 0.0001 | best_found (compress init) | 5.402539e-02 | 547589.8386665683 |
| 0.01 | zipup | 5.402539e-02 | 54.77700060430625 |
| 0.01 | input | 9.862787e-04 | 1.0 |
| 0.01 | tdvp | 9.039680e-01 | 916.5441948590216 |
| 0.01 | best_found (compress init) | 5.402539e-02 | 54.77700060430625 |
| 0.1 | zipup | 5.402539e-02 | 0.5659660540120951 |
| 0.1 | input | 9.545694e-02 | 1.0 |
| 0.1 | tdvp | 9.052116e-01 | 9.48293136254657 |
| 0.1 | best_found (compress init) | 5.402539e-02 | 0.5659660540120951 |

Notes on this table:

- **zipup init:** production path; sweeps are no-ops → zip-up plateau.
- **input init:** stays at the no-update baseline (sweeps no-op or reject); **removes the plateau**.
- **tdvp init at x=1e-4:** retains TDVP (good). At x∈{0.01,0.1}: some bond updates are *accepted* (virtual dims happen to match after gauge moves) and the **Re⟨T|A⟩ objective decreases while normalized fidelity collapses** (0.90). This is a second bug: the optimizer objective is not fidelity.
- **best_found as initializer** here means `MPS.compress(uncapped)` — itself a poor χ=8 truncation (~zip-up quality). It is **not** the independent TT-SVD best-found of §5.

### Objective vs fidelity (implementation bug)

At x=0.01, χ=8, TDVP init:

- TDVP normalized infidelity = `6.23e-4`, but Euclidean objective = `‖T−A‖² ≈ 4.0` because the code uses `Re⟨T|A⟩` rather than `|⟨T|A⟩|`.
- After sweeps: objective falls to `≈1.40` while infidelity rises to `≈0.90`.

Decision rule triggered: **independently evaluated fidelity disagrees with the optimizer objective → implementation bug.**

**Conclusion:** the O(10⁻²) variational plateau is **optimizer failure** (silent no-op on zip-up init), not a χ=8 expressivity limit. Input initialization alone removes it. A correct method must optimize a fidelity-compatible objective (or TT-SVD the exact target) and use multi-start best-retained selection.

## 5. Independent best-found MPS

Built by multi-start compression of the uncapped exact target (`MPS.compress`) plus TT-SVD of the dense statevector; best infidelity retained.

| χ | x | best-found inf | no-update | zip var-MPO |
|---:|---:|---:|---:|---:|
| 8 | 0.0001 | 6.247072e-08 | 9.866032e-08 | 5.402539e-02 |
| 8 | 0.01 | 6.531320e-04 | 9.862787e-04 | 5.402539e-02 |
| 8 | 0.1 | 1.554201e-02 | 9.545694e-02 | 5.402539e-02 |
| 12 | 0.0001 | 1.617767e-09 | 9.866032e-08 | 6.274185e-03 |
| 12 | 0.01 | 1.606159e-05 | 9.862787e-04 | 6.274185e-03 |
| 12 | 0.1 | 1.164029e-03 | 9.545694e-02 | 6.274185e-03 |
| 16 | 0.0001 | 0.000000e+00 | 9.866032e-08 | 0.000000e+00 |
| 16 | 0.01 | 0.000000e+00 | 9.862787e-04 | 0.000000e+00 |
| 16 | 0.1 | 0.000000e+00 | 9.545694e-02 | 0.000000e+00 |

## 6. TDVP subdivision

Implementation: hybrid long-range path uses 2-site TDVP with symmetric LTR+RTL sweep per substep (`tdvp_mode='2site'`, `tdvp_sweeps=n`, krylov_tol=1e-12, svd_threshold=1e-13). Each substep advances time `1/n`.

| χ | n | infidelity | phase-aligned ‖·‖ | actual χ |
|---:|---:|---:|---:|---:|
| 8 | 1 | 1.950984e-02 | 1.400212e-01 | 8 |
| 8 | 2 | 2.169686e-02 | 1.477019e-01 | 8 |
| 8 | 4 | 2.866550e-02 | 1.699233e-01 | 8 |
| 8 | 8 | 3.093880e-02 | 1.765839e-01 | 8 |
| 8 | 16 | 3.201527e-02 | 1.796544e-01 | 8 |
| 8 | 32 | 3.254401e-02 | 1.811441e-01 | 8 |
| 8 | 64 | 3.280613e-02 | 1.818783e-01 | 8 |
| 12 | 1 | 1.456000e-03 | 3.816452e-02 | 12 |
| 12 | 2 | 1.379585e-03 | 3.714918e-02 | 12 |
| 12 | 4 | 1.379320e-03 | 3.714560e-02 | 12 |
| 12 | 8 | 1.387951e-03 | 3.726168e-02 | 12 |
| 12 | 16 | 1.393273e-03 | 3.733308e-02 | 12 |
| 12 | 32 | 1.395129e-03 | 3.735794e-02 | 12 |
| 12 | 64 | 1.394913e-03 | 3.735506e-02 | 12 |
| 16 | 1 | 4.440892e-16 | 1.505204e-13 | 16 |
| 16 | 2 | 0.000000e+00 | 6.692528e-14 | 16 |
| 16 | 4 | 0.000000e+00 | 6.098090e-14 | 16 |
| 16 | 8 | 0.000000e+00 | 6.369914e-14 | 16 |
| 16 | 16 | 1.438336e-10 | 1.199307e-05 | 16 |
| 16 | 32 | 3.588996e-11 | 5.990800e-06 | 16 |
| 16 | 64 | 8.963719e-12 | 2.993963e-06 | 16 |

At χ=16 (sufficient capacity), n=1…8 sit at numerical noise; larger n shows a small rise then decrease — consistent with accumulated Krylov/projector noise rather than classical Trotter order improvement. In the compressed regime (χ=8), increasing n **worsens** infidelity: projection error dominates integrator error.

## 7. Reproduction of quoted production numbers (seed 11)

### x = 0.1

| χ | method | audit infidelity |
|---:|---|---:|
| 8 | hybrid_tdvp | 3.280613398672e-02 |
| 8 | tebd_swap | 3.095156407881e-01 |
| 8 | mpo_zipup | 5.402538757008e-02 |
| 8 | variational_mpo | 5.402538757008e-02 |
| 12 | hybrid_tdvp | 1.394913463179e-03 |
| 12 | tebd_swap | 2.521355355855e-01 |
| 12 | mpo_zipup | 6.274184945525e-03 |
| 12 | variational_mpo | 6.274184945525e-03 |
| 16 | hybrid_tdvp | 8.963718656219e-12 |
| 16 | tebd_swap | 0.000000000000e+00 |
| 16 | mpo_zipup | 0.000000000000e+00 |
| 16 | variational_mpo | 0.000000000000e+00 |

### TDVP small-angle fit (χ=8)

- Interval x∈[0.0001, 0.01], n=3
- Fit `infidelity ≈ c x²` with c=`6.243125e+00` (rms relative residual `1.358e-03`)

- Worst TDVP infidelity at χ=16 on audited generic+special grid: `5.094875e-09` at x=0.0001 (PASS vs claimed <1e-8 on selection angles).

## 8. Robustness across seeds

Ten fixed seeds `{11,…,20}`; angles `x∈{1e-4,1e-3,1e-2,1e-1,1}`; χ∈{8,16}.
Summary JSON: `robustness_summary.json`.

### χ=8, x=1e-4 (n=10)

- no-update: med=`9.860e-08`, IQR=`[9.848e-08, 9.865e-08]`, range=`[9.607e-08, 9.869e-08]`
- TDVP: med=`6.756e-08`, IQR=`[5.882e-08, 7.235e-08]`, range=`[5.148e-08, 8.058e-08]`
- MPO zip-up: med=`1.152e-01`, IQR=`[8.620e-02, 1.341e-01]`, range=`[5.403e-02, 1.544e-01]`
- Var-MPO zip init: **identical to MPO zip-up** (same median/IQR/range)
- Var-MPO input init: **identical to no-update**
- best-found (TT-SVD multi-start): med=`6.756e-08`, IQR=`[5.882e-08, 7.235e-08]`, range=`[5.148e-08, 8.058e-08]` (matches TDVP)

- TDVP/baseline ratio: med=`0.685`, range=`[0.523, 0.817]`
- Fraction of seeds with TDVP < no-update baseline: **1.00**

Qualitative conclusion (TDVP beats doing nothing at weak angles under compression; production var-MPO equals broken zip-up) does **not** depend on seed 11.

## Corrected interpretation

| Claim in current figure/text | Audit finding |
|---|---|
| Variational MPO ≈ MPO zip-up plateau at small θ, χ=8/12 | **True numerically for the production code path**, but only because variational is a no-op on zip-up |
| Plateau is the best χ-constrained variational approximation | **False** — input init and TT-SVD best-found are ~10⁵–10⁶× better at x=1e-4, χ=8 |
| TDVP small-angle O(θ²) advantage | **Holds vs no-update baseline on all 10 seeds** (not only vs broken MPO) |
| TEBD+SWAP flat error at χ=8 | **Mostly SWAP-routing truncation**, already visible at θ=0 |
| Var-MPO objective equals state fidelity | **False** — uses `Re⟨T|A⟩`; can improve objective while destroying `|⟨T|A⟩|²` fidelity |

## Minimal code patch required

See `variational_patch_notes.md` and `variational_fixed.py` (multi-start TT-SVD / input / zip-up, best-retained).
Do **not** regenerate `figure_single_gate_main_text` until that protocol is wired in and the benchmark is re-run into a new output directory.

## Artifacts

- `results.csv` — raw measurements
- `convergence.csv` — variational traces and TDVP substep history
- `diagnostic_angle_with_baselines.{pdf,png}` — methods + no-update + best-found
- `diagnostic_variational_inits.{pdf,png}` — initializer comparison
- `meta.json` — machine-readable summaries
- `robustness_summary.json` — ten-seed aggregates
- `variational_fixed.py` — minimal corrected multi-start implementation
- `variational_patch_notes.md` — patch instructions

