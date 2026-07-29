# Notes for the variational-MPO fix (now applied)

## Root cause (historical)

`_bond_update_from_target` silently no-oped (bond-dimension mismatch → swallowed
`ValueError`), so production `apply_variational_mpo_gate` returned zip-up
unchanged. The Euclidean objective also used `Re⟨T|A⟩` rather than fidelity.

## Fix applied

`experiments/single_gate/variational.py` now uses multi-start best-retained
selection by normalized state infidelity among:

- MPO zip-up
- unchanged input MPS
- TT-SVD of the uncapped exact target at the requested χ

Angle-sweep `variational_mpo` rows were regenerated (backup:
`output/results.sqlite.pre_varfix_bak`). The publication figure was replotted
from the corrected database.

Meta key: `variational_protocol=multistart_best_retained_ttsvd_input_zipup_v1`.
