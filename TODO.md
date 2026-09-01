# Orthogonality-center TODO

The tracked center is a correctness invariant:

- `orthogonality_center == c` means that the full MPS is mixed-canonical at site
  `c`.
- `orthogonality_center is None` means that the gauge is unknown.
- A tensor mutation must preserve this invariant, update the center, or set it
  to `None`.
- A center-dependent contraction or truncation must first establish the required
  gauge.

## P0: Correctness defects

- [ ] Canonicalize unknown-gauge states before public TDVP evolution.
  - Locations: `src/mqt/yaqs/core/methods/tdvp/tdvp.py::tdvp` and
    `evolve_window`.
  - Problem: `tdvp` checks for center `0` only when the center is known. The
    sweep kernels still assume a center-0 gauge when it is `None`.
  - Fix: canonicalize at site `0` when the center is unknown. Make
    `evolve_window` enforce or establish the same precondition.
  - Regression: compare genuinely gauge-equivalent MPS inputs for 1-site,
    2-site, and dynamic TDVP. Include a dynamic input whose bond dimension
    exceeds `max_bond_dim`, and verify the returned center with
    `check_canonical_form()`.

- [ ] Move the center onto each scheduled two-site jump before a capped split.
  - Location:
    `src/mqt/yaqs/core/methods/scheduled_jumps.py::apply_scheduled_jumps`.
  - Problem: an arbitrary adjacent pair is merged and truncated without first
    making its environment isometric. The result depends on the input gauge even
    though the returned state is later normalized and has a valid center.
  - Fix: if the center does not cover `[i, j]`, shift it to the nearest pair
    site or canonicalize there when the gauge is unknown. Do this for every
    matched jump because an earlier jump can change the gauge.
  - Regression: use an identity two-site jump and `max_bond_dim=1` on two
    gauge-equivalent states. The current output fidelity can fall to `0.02318`;
    the centered result must match the optimal Schmidt truncation. Also test the
    uncapped control.

- [ ] Recenter before applying the selected stochastic adjacent two-site jump.
  - Location:
    `src/mqt/yaqs/core/methods/stochastic_process.py::stochastic_process`.
  - Problem: `create_probability_distribution` sweeps a known center to the last
    site. The selected process can then truncate an arbitrary earlier pair
    without restoring the pair gauge.
  - Fix: establish a center on the selected pair immediately before the capped
    split. Share the same helper and contract as scheduled jumps if practical.
  - Regression: force selection of an identity jump on an earlier pair and
    compare gauge-equivalent inputs under a bond cap. Include the uncapped
    control and assert that the returned center is genuine.

## P1: Internal invariant violations

- [ ] Invalidate or correctly propagate the center in fixed-bond-dimension
      synchronization.
  - Locations: `src/mqt/yaqs/core/methods/tdvp/sweep_utils.py::_sync_bond_dim`,
    `_align_bond`, and `_cap_bonds`.
  - Problem: a `sqrt` SVD split or zero-padding can destroy mixed-canonical form
    while leaving the old center set. A direct truncation has produced global
    norm `0.89248` while the center-local norm returned `1.0`.
  - Fix: set the center to `None` after a gauge-breaking change, or use a
    directional split that proves and records the new center. Re-establish the
    required gauge before any later center-dependent operation.
  - Regression: cover truncation, padding, no-op, digital, and analog paths.
    Check both the physical result and
    `orthogonality_center in check_canonical_form()` whenever the center is
    non-`None`.

- [ ] Clear the center when `ensure_internal_bond_dims` adds null workspace.
  - Location:
    `src/mqt/yaqs/core/data_structures/mps.py::ensure_internal_bond_dims`.
  - Problem: zero-padding can make `check_canonical_form()` return no valid
    center while the previous center remains set.
  - Fix: preserve the center for a no-op, but set it to `None` after any actual
    padding unless canonicality is explicitly restored.
  - Regression: verify that padding clears the center, a no-op preserves it, and
    the represented dense state is unchanged.

- [ ] Stop using `orthogonality_center` as the compression return-target field.
  - Location: `src/mqt/yaqs/core/data_structures/mpo.py::MPO._multiply_mps`.
  - Problem: after contracting an MPO into every tensor, the method temporarily
    sets the center to the last site even though the state is not canonical.
    `MPS.compress` currently reads it only as the site to restore, then
    immediately canonicalizes, so normal returns are correct but the metadata is
    temporarily false.
  - Fix: pass a private restore target to compression without claiming that it
    is the current center, or otherwise keep the center `None` until
    canonicalization establishes it.
  - Regression: retain the existing dense-result, capped-compression, and
    genuine-final-center checks.

- [ ] Remove the temporary center-as-cursor use in `measure_single_shot` (low
      priority).
  - Location: `src/mqt/yaqs/core/data_structures/mps.py::measure_single_shot`.
  - Problem: the temporary state discards the measured prefix logically but
    leaves its tensors in place. It then records the next site as the full-MPS
    center even though the temporary tensor chain can have inconsistent bonds.
    The object is not exposed and the sampling algorithm only uses the active
    suffix, but the center field no longer has its documented meaning.
  - Fix: track the active suffix or current site in a local variable instead of
    writing center metadata on the incomplete temporary representation.
  - Regression: compare the full sampled bitstring distribution of an entangled
    MPS with a dense reference, not only the first-site marginal.

## P2: API hardening

- [ ] Validate center-helper arguments before changing tensors or metadata.
  - Locations: `MPS.set_center`, `shift_center_to`,
    `shift_orthogonality_center_right`, `shift_orthogonality_center_left`, and
    `set_canonical_form`.
  - Problem: an unsupported decomposition can leave tensors unchanged while
    advancing the center. Center and site targets also lack consistent bounds
    checks.
  - Fix: reject decompositions other than `"QR"` and `"SVD"`, and reject
    out-of-range sites before mutation. Keep `set_center(None)` as the explicit
    invalidation operation.
  - Regression: assert that invalid inputs raise without changing tensors,
    orientation, physical dimensions, or center metadata.

- [ ] Make the two-site split precondition explicit and harder to misuse.
  - Location: `MPS.update_center_after_split` and all callers.
  - Problem: the helper assigns a center from the singular-value distribution
    but cannot know whether the pair was in a valid mixed-canonical environment
    before truncation. This allowed the scheduled and stochastic jump defects.
  - Fix: validate site bounds, adjacency, and the SVD distribution. Centralize
    “ensure center covers pair, split, update center” behavior, or document and
    assert a clear precondition at internal call sites.
  - Regression: exercise calls with a center left of, on, right of, and unknown
    relative to the pair.

- [ ] Defensively enforce the documented precondition of local-only
      contractions.
  - Locations: `MPS.local_expect` and local-site `MPS.scalar_product` calls.
  - Problem: these fast contractions are correct only when the center covers the
    requested site or pair. High-level callers currently establish that gauge,
    but direct calls can silently return a gauge-dependent value.
  - Fix: validate `check_covers_sites()` or keep these routines private and
    route public use through `expect` and `norm`.
  - Regression: direct off-center calls must either recenter safely or raise a
    clear error.

## Documented boundaries, not current defects

- Direct assignment to `MPS.tensors` bypasses tracking. The class documentation
  already requires callers to use a mutator or call `set_center(None)`.
- `set_center` deliberately changes metadata without canonicalizing. Callers
  remain responsible for supplying a truthful center; bounds validation is still
  required.
- `apply_long_range_gate_mpo` already canonicalizes an unknown gauge and moves a
  known center into the gate window.
- TEBD gate application, digital TDVP windows, dissipation, BUG, `apply_local`,
  entropy and Schmidt-spectrum evaluation, `expect`, `evaluate_observables`, and
  `norm(site)`, in-place single-site `measure`, `_adjacent_jump_weight`, and
  `_renormalize` currently establish, preserve, or invalidate the center
  correctly on their supported paths.
