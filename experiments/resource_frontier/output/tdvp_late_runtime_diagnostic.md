# TDVP late-time runtime diagnostic

Uses existing `timing_repeats.csv`, `runtime_frontier.csv` and `raw_runs.csv` only.
Reliability threshold ε=0.01, Δt=0.1.

## Overall classification

**Outcome B**

The large late-time increases in the TDVP measured-runtime frontier are primarily **configuration-switch effects**: the runtime-minimizing reliable χmax increases (32→48 at t=1.3; 48→64 at t=1.5) because smaller χmax caps cease to satisfy 1-F<10⁻² at every preceding step. Within each newly selected χmax, all three controlled repetitions agree (IQR ≪ the frontier jump), so this is not a timing artifact.

## Per-target findings

- t=1.3 (n=13): selected χmax=48 (prev χmax=32); median=70.257s, Q1=69.507s, Q3=71.898s, IQR=2.391s; median incremental step=8.222s; peak_param=27304, peak_bond=48; class=B. Frontier χmax switches 32→48 at n=13; median growth 58.37s ≫ IQR 2.39s.
- t=1.4 (n=14): selected χmax=48 (prev χmax=48); median=79.219s, Q1=78.051s, Q3=80.626s, IQR=2.575s; median incremental step=8.494s; peak_param=27304, peak_bond=48; class=A. All repetitions show growth; median growth 8.96s ≫ IQR 2.57s; χmax unchanged (48).
- t=1.5 (n=15): selected χmax=64 (prev χmax=48); median=263.500s, Q1=260.524s, Q3=264.955s, IQR=4.431s; median incremental step=26.258s; peak_param=43688, peak_bond=64; class=B. Frontier χmax switches 48→64 at n=15; median growth 184.28s ≫ IQR 4.43s.

## χmax sequence on the TDVP runtime frontier

| n | t | χmax | median R* (s) | IQR (s) |
|---:|---:|---:|---:|---:|
| 10 | 1 | 24 | 3.541 | 0.060 |
| 11 | 1.1 | 24 | 3.999 | 0.059 |
| 12 | 1.2 | 32 | 11.884 | 4.767 |
| 13 | 1.3 | 48 | 70.257 | 4.783 |
| 14 | 1.4 | 48 | 79.219 | 5.150 |
| 15 | 1.5 | 64 | 263.500 | 8.863 |

## Interpretation note

The late-time TDVP cost increase is associated with the larger selected χmax required for reliability and with growth of retained bonds / effective local TDVP problems. Phrase causally as consistent with the increasing cost of local TDVP updates; do not imply smooth intrinsic scaling at fixed χmax across these jumps.

Wrote `tdvp_late_runtime_diagnostic.csv`.
