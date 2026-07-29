# Affected-benchmark inventory (post compress / zip-up / variational repair)

Repairs landed in:

- `src/mqt/yaqs/core/data_structures/mps.py` — `MPS.compress` (right-canonize with QR, then LTR truncate)
- `src/mqt/yaqs/core/libraries/gate_library.py` — `split_tensor` hard cutoff `1e-6 → 1e-14`
- `experiments/single_gate/variational.py` — multi-start projected variational MPO

## Datasets

| Pipeline | Status |
|---|---|
| `experiments/single_gate/` | Regenerated; archive `archive/pre_repair_*` |
| `experiments/fixed_resources/` | **Regenerated** in `output_corrected/`; archive `archive/pre_repair_20260723T145513Z/` |
| `experiments/resource_frontier/` | **Still needs regeneration** after accepting fixed_resources |
| `experiments/convergence/` | Optional; subdivision re-validated in fixed_resources (`n=2`) |

## Do not reuse

Archived pre-repair CSVs retain broken zip-up horizons and must not mix into corrected figures.
