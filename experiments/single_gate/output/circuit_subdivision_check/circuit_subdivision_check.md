# Circuit subdivision check (corrected fixed-χ figure)

Re-run after `MPS.compress` / zip-up repairs on the exact 4×4 TFIM & Heisenberg
Strange circuits used by `experiments/fixed_resources`.

## Protocol

- χ=32, TDVP only, \(n\in\{1,2,4,8\}\) (16/64 omitted after plateau visible)
- TFIM: 20 Trotter steps (\(t\le2\)); Heisenberg: first step for subdivision,
  multi-step horizons measured in production
- Horizon: last-reliable \(T_\varepsilon\) at \(\varepsilon=10^{-2}\)

## Table

| n | TFIM \(n_\varepsilon\) | Heis \(n_\varepsilon\) | Heis \(1{-}F(\Delta t)\) |
|--:|--:|--:|--:|
| 1 | 12 | 0 | 2.155×10⁻² |
| 2 | 13 | 0 | 1.808×10⁻² |
| 4 | 13 | 0 | 1.645×10⁻² |
| 8 | 13 | 0 | 1.577×10⁻² |

## Choice

**Uniform circuit baseline: \(n=2\).**

Smallest \(n\) with doubling-stable TFIM horizon (\(n=2\) vs \(4\): Δ\(n_\varepsilon\)=0).
Do **not** adopt \(n=1\) from the isolated-gate study: it shortens the TFIM horizon
by one step here. Larger \(n\) does not change the scientific ranking.

Raw data: `experiments/fixed_resources/output_corrected/circuit_subdivision_validation.csv`
