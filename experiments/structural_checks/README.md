# Structural checks

This deterministic suite supplies Table I. It evaluates the summed
instantaneous fixed-rank and two-site projector actions on the same contiguous
gate-support window, their exterior cancellations, nearest-neighbor exactness,
and the minimal-support long-range obstruction. The exterior diagnostic includes
the individually nonzero two-site terms that straddle the window boundaries.
It does not claim that a finite full-chain projector-splitting sweep equals a
windowed sweep.

The dense projector fixtures are independent of the production TDVP
implementation. The production updater is invoked only for the
\(\lvert0000\rangle\), \(R_{XX}(\pi/2)\) stall control. Gate ordering and dense
gate conventions are also checked by the independent individual-gate
validation suite.

## Site conventions

- CSV and code sites are zero-based.
- The generated LaTeX table uses one-based manuscript labels.
- Dense statevectors use site 0 as the slowest tensor axis.

## Outputs

| Artifact | Purpose |
| --- | --- |
| `projector_checks.csv` | fixed-rank locality, two-site locality, and nearest-neighbor exactness |
| `exterior_cancellation.csv` | nonzero exterior terms and their pairwise cancellation |
| `obstruction_checks.csv` | analytical obstruction and four production-stall configurations |
| `summary.json` | configuration, residual maxima, and validation status |
| `table_structural.tex` | theorem-facing manuscript table |
| `table_production_stall.tex` | compact source for the production-stall prose |

## Reproduce

```bash
uv run python -m experiments.structural_checks.run
uv run pytest experiments/structural_checks/test_structural_checks.py -q
```
