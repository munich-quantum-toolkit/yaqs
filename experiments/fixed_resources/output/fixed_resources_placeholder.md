# Fixed-resource Results placeholder figure

## Figure caption

Circuit accuracy and method regimes under a fixed bond-dimension cap on 4×4 lattices. (a) Reliability horizon nε (Trotter steps to ε) for the 4×4 TFIM versus χmax. (b) Infidelity versus Trotter step (Δt=0.1) for the 4×4 TFIM at χmax=32. (c) Infidelity after one Trotter step of the 4×4 Heisenberg circuit across χmax, showing the crossover between TDVP at constrained χ and explicit gate application at larger χ. Heisenberg TDVP uses four substeps.

## Layout
Full-width 1×3 figure. Panels (a) and (c) use dense χ scans; (b) uses the validated χ=32 TFIM trajectory.

### (a) 4×4 TFIM horizon
- Dense χ ∈ [2, 4, 8, 12, 16, 24, 32, 48, 64] from `tfim_horizon_dense.csv`
- Vertical axis: $n_\varepsilon$ (Trotter steps; $T_\varepsilon/\Delta t$, $\Delta t=0.1$)

- TDVP: χ=2:3, χ=4:4, χ=8:7, χ=12:7, χ=16:10, χ=24:12, χ=32:13, χ=48:15, χ=64:16
- TEBD+SWAP: χ=2:3, χ=4:3, χ=8:2, χ=12:2, χ=16:3, χ=24:3, χ=32:3, χ=48:3, χ=64:3
- MPO zip-up: χ=2:1, χ=4:1, χ=8:3, χ=12:4, χ=16:5, χ=24:4, χ=32:4, χ=48:6, χ=64:6

### (b) 4×4 TFIM trajectory
- Infidelity vs Trotter step (Δt=0.1) at χmax=32 (`trajectories.csv`)

### (c) 4×4 Heisenberg, one Trotter step
- Dense χ ∈ [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]; TDVP n=4
- Source: `heisenberg_one_step_dense.csv`

| χmax | TDVP | TEBD+SWAP | MPO zip-up |
|---:|---:|---:|---:|
| 2 | 2.8601e-01 | 5.3278e-01 | 9.9633e-01 |
| 4 | 1.5984e-01 | 4.2093e-01 | 9.9999e-01 |
| 8 | 6.1692e-02 | 3.6172e-01 | 2.6459e-01 |
| 12 | 2.1392e-02 | 2.8644e-01 | 2.4297e-01 |
| 16 | 1.9825e-02 | 2.2968e-01 | 1.2919e-01 |
| 24 | 1.6816e-02 | 2.3128e-01 | 1.5800e-01 |
| 32 | 1.6447e-02 | 1.2832e-01 | 8.9457e-02 |
| 48 | 1.6399e-02 | 8.4413e-02 | 4.2584e-02 |
| 64 | 1.6338e-02 | 4.3990e-02 | 4.6657e-02 |
| 96 | 1.6216e-02 | 3.2686e-02 | 4.4908e-03 |
| 128 | 1.6226e-02 | 1.0223e-03 | 1.2744e-03 |

## Outputs
- `fixed_resources_placeholder.pdf` / `fixed_resources_placeholder.png`
- `fixed_resources_placeholder.md`

## Notes
- Incomplete TFIM χ=128 not resumed; no variational MPO.
