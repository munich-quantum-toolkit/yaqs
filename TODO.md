# MPO expectation values for 1.0

## Goal

Add one MPS-level operation that computes the expectation value of a full-chain
MPO. Use this operation to measure Hamiltonian energy and individual long-range
correlations.

MPO-backed `Observable` objects remain a post-1.0 feature.

## 1. Define the supported contract

- [x] Add `MPS.expect_mpo(operator: MPO) -> np.complex128`.
- [x] Compute the stored-state contraction $\langle\psi|W|\psi\rangle$ directly.
- [x] Accept full-chain MPOs only. Require the MPO length to equal the MPS
      length.
- [x] Do not mutate the MPS or MPO.
- [x] Do not apply the MPO to a copied MPS.
- [x] Do not densify or compress the MPO.
- [x] Do not normalize the MPS implicitly.
- [x] Return the raw complex value. A general MPO need not be Hermitian.
- [x] Keep `MPS.expect(Observable(...))` unchanged.

## 2. Validate the MPO tensor chain

- [x] Validate that the tensor count and MPO length match the MPS length.
- [x] Require each MPO tensor to have four axes in
      `(phys_out, phys_in, left_bond, right_bond)` order.
- [x] Require both physical dimensions to match the corresponding MPS tensor.
- [x] Require neighboring virtual bond dimensions to match.
- [x] Require both outer virtual bond dimensions to be one.
- [x] Validate the full tensor chain before contraction so malformed MPO
      dimensions cannot broadcast silently.
- [x] Use tensor shapes as the source of truth. Do not rely on the scalar
      `MPO.physical_dimension` metadata.
- [x] Keep the final check that the contraction has no open virtual bonds.

## 3. Support Hamiltonian energy

- [x] Use the existing `Hamiltonian.ensure_mpo()` and `Hamiltonian.mpo` APIs:

  ```python
  hamiltonian.ensure_mpo()
  energy = state.mps.expect_mpo(hamiltonian.mpo)
  ```

- [x] Verify preset and manually supplied static Hamiltonians.
- [x] State that the result is the expectation value of the materialized MPO.
- [x] For a piecewise Hamiltonian, require the user to select the applicable
      static piece.
- [x] Do not add an `energy()` wrapper unless it provides a distinct contract.

## 4. Support individual long-range correlations

- [ ] Use the existing `MPO.from_pauli_sum` builder for Pauli products and
      strings:

  ```python
  correlation = MPO()
  correlation.from_pauli_sum(
      terms=[(1.0, "Z0 Z7")],
      length=state.mps.length,
      n_sweeps=0,
  )
  value = state.mps.expect_mpo(correlation)
  ```

- [ ] Also verify a bond-one product created with `MPO.from_local_ops`.
- [ ] Document that a connected correlation is obtained by subtracting the
      product of the two one-site expectations.
- [ ] Keep arbitrary separated-site dense matrices out of this change. Such an
      operator needs an operator-Schmidt decomposition and a separate cutoff and
      site-order contract.
- [ ] Keep optimized all-pairs correlation matrices out of this change. They
      need reusable left and right environments.

## 5. Add focused tests

Add tests to `tests/core/data_structures/test_mps.py`, which mirrors the owning
source file.

- [ ] Compare small random MPS-MPO contractions with independent dense
      references.
- [x] Test a known Ising or Heisenberg energy.
- [ ] Test a separated-site correlation such as $Z_0Z_{L-1}$.
- [ ] Test a Pauli string with more than two nonidentity factors.
- [x] Test a non-Hermitian MPO whose expectation value is complex.
- [ ] Test a zero MPO.
- [x] Test an unnormalized MPS and verify the expected amplitude-squared
      scaling.
- [ ] Test known and unknown orthogonality-center metadata.
- [x] Confirm that neither input changes.
- [x] Confirm that the implementation does not call MPO densification,
      multiplication, or compression.
- [x] Test invalid tensor count, rank, physical dimensions, boundaries, and
      internal bonds.
- [x] Test mixed local dimensions if they are part of the declared contract.
- [ ] Keep dense references independent of the contraction under test.

## 6. Document and release the 1.0 feature

- [ ] Add energy and long-range-correlation examples to the Hamiltonian
      documentation.
- [ ] Document the full-chain-only, raw-complex, and no-implicit-normalization
      contracts.
- [ ] Explain that local `Observable` measurements keep their existing fast
      path.
- [ ] Add a `CHANGELOG.md` entry with the pull request and author links.
- [ ] Disclose AI assistance in the pull request description.

## 1.0 non-goals

- `Observable(MPO)` or another MPO-observable wrapper.
- MPO observables in `AnalogSimParams`, `DigitalSimParams`, or
  `SimulationProgram`.
- Time-resolved MPO measurements during simulation.
- Vector or density-matrix backend support for MPO measurements.
- Changes to bitstrings, entropy, Schmidt spectra, or result storage.
- Automatic conversion of every local observable into an MPO.
- Global MPO Hermiticity checks or new MPO compression policies.

## Post-1.0: make MPOs first-class observables

- [ ] Choose the public construction API, such as `Observable.from_mpo(mpo)`.
- [ ] Define one authoritative operator representation. Do not keep mutable
      matrix, factor, and cached-MPO definitions that can disagree.
- [ ] Define copy and ownership rules for caller-supplied MPO tensors.
- [ ] Consolidate MPO structural validation and audit every public MPO builder
      against it.
- [ ] Define a scale-safe Hermiticity policy, or state clearly that callers are
      responsible for supplying Hermitian observable MPOs.
- [ ] Preserve direct local contraction for one-site and adjacent two-site
      observables.
- [ ] Add MPS-backed, time-resolved simulator support before adding other
      representations.
- [ ] Define and document the backend support matrix before adding vector or
      density-matrix conversions.
- [ ] Add focused integration tests for observable ordering, worker copies,
      noise workflows, and program segments without changing unrelated
      diagnostic behavior.

## Validation

- [x] Run `uv run pytest tests/core/data_structures/test_mps.py -q`.
- [ ] Run `uvx nox -s lint` after each batch of changes.
- [x] Run `uv run pytest` before handoff.
- [x] Run `git diff --check`.
