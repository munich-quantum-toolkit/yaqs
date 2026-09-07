# Observable API implementation plan

Make observable definitions independent of gates. Preserve the familiar
`Observable(operator, sites)` interface and support Hermitian operators supplied
as local matrices, operators on several sites, Pauli sums, or matrix product
operators (MPOs).

## Status and execution order

Chunks 1 through 3 are complete. MPS measurement support does not yet mean that
every backend can measure the new operators.

| Chunk                             | Status                                        | Completion boundary                                     |
| --------------------------------- | --------------------------------------------- | ------------------------------------------------------- |
| 1. Observable definitions         | Complete                                      | Gate-independent construction and metadata              |
| 2. MPO construction               | Complete                                      | Validated, reusable operator data                       |
| 3. MPS contraction                | Complete                                      | Direct and batched MPS measurements accept general MPOs |
| 4. Backend and result integration | Preparation is connected; measurement remains | Supported backends and result paths agree               |
| 5. Performance and documentation  | Pending; use the pre-3A commit as baseline    | Measured cost, complete examples, and release checks    |

Work in order: **5A → 4A → 4B → 4C → 5B → 5C**. Each subsection is a reviewable
implementation batch. Complete its acceptance checks before proceeding. In 5A,
use the commit immediately before 3A for the old direct contraction and the
commit immediately before 3B for the old batched dispatch. Finish the
performance comparison after integration.

## Design requirements

- Accept only Hermitian operators, within a documented numerical tolerance. Keep
  expectation values real.
- Preserve common calls such as `Observable("z", 0)`,
  `Observable("zz", [0, 1])`, and `Observable(matrix, 0)`.
- Use MPOs as the general operator representation for MPS measurement. Keep
  compact local definitions and efficient local contractions.
- Separate entropy and Schmidt spectra from linear operators internally. Keep
  their existing `Observable(...)` syntax as requests for state diagnostics.
- Remove `BaseGate` inputs, the `gate=` keyword, and the `.gate` attribute.
  Breaking changes are allowed. Do not add compatibility adapters, deprecation
  stages, or migration guidance.
- Keep the identity name `id`, with no aliases. Keep `h`, `cx`, `cz`, and `swap`
  as gate names; their Hermitian matrices remain valid custom inputs. Use
  `ObservableType` and `.type` consistently in code and Google-style docstrings.
- Preserve observable order, sampling behavior, and result shapes for supported
  measurements.
- Keep this work focused on observables and their measurement paths.

## Target interface

These constructors and general MPS measurements are implemented. Consistent
backend support remains in chunk 4.

```python
from mqt.yaqs import Observable

# Existing construction remains familiar.
Observable("z", 0)
Observable("zz", [0, 1])
Observable("position", 0, positions=grid)
Observable(local_matrix, 0)
Observable("101")
Observable("entropy", [1, 2])
Observable("schmidt_spectrum", [1, 2])

# General MPS measurement is available; backend integration is the next step.
Observable("zz", [0, 5])
Observable(custom_matrix, [1, 3])
Observable(custom_mpo)
Observable.from_pauli_sum(
    terms=[(0.5, "Z0 Z3"), (0.2, "X1")],
    length=4,
)
```

An integer site selects one local space. A matrix on a single site can have any
supported local dimension; a 4-by-4 matrix on one site must not be interpreted
as a two-qubit operator. A site list identifies the spaces of an operator on
several sites. An MPO supplied without sites represents the full chain and must
match the state length and each local dimension.

## 1. Separate definitions from gates

- [x] Add `core/data_structures/observable.py` and export `Observable` from
  `mqt.yaqs`. Update internal imports and remove the class definition from
  `simulation_parameters.py`.
- [x] Add `core/libraries/observable_library.py` with explicit named operator
  factories. Include Pauli operators, identity, Pauli products, projectors,
  and position, plus separate factories for state diagnostics.
- [x] Replace gate metadata with observable metadata for operator data, sites,
  names, and diagnostic requests. Keep per-run values on `Result`.
- [x] Remove observable-only definitions from `gate_library.py`. Share basic
      operator data through a neutral module only where needed to avoid
      duplicate definitions. Preserve definitions that circuit or Hamiltonian
      code still uses.
- [x] Resolve names explicitly. Interpret valid binary strings as projectors;
  reject unknown names and malformed strings with clear errors.
- [x] Validate factory arguments and reject non-Hermitian named operators and
      custom matrices. A real expectation on one state is not a Hermiticity
      test.
- [x] Update constructor and import tests in the corresponding test tree. Cover
      valid syntax, local dimensions, removed gate inputs, and invalid
      arguments.

## 2. Build and validate operator MPOs

Implemented in `core/data_structures/observable.py` and
`core/data_structures/mpo.py`:

- [x] Define and document matrix basis order, site-list order, and MPO tensor
      order before implementing conversions. Preserve supported local
      conventions. Handle reversed sites by permuting operator axes with the
      sites.
- [x] Resolve system-dependent dimensions when preparing a run. Validate site
      bounds, duplicate sites, matrix sizes, and physical dimensions before
      workers start. Direct measurement calls must perform equivalent
      validation.
- [x] Build compact MPOs for named and custom operators. Insert identity tensors
      across gaps for operators on nonadjacent sites. Avoid allocating a
      separate full-chain MPO for every local observable.
- [x] Accept existing MPO objects and add `Observable.from_pauli_sum(...)` using
      the existing MPO builders where their contracts fit. Validate Hermiticity
      of the represented sum.
- [x] Validate supplied MPO tensors, finite values, boundary bonds, neighboring
  bond dimensions, and square physical legs.
- [x] Check Hermiticity of the complete MPO through tensor contractions without
  constructing a full Hilbert-space matrix. Do not require each tensor to be
  Hermitian. Specify scale-aware tolerances and handle zero operators.
- [x] Avoid silent observable truncation. Audit default cutoffs in reused MPO
  builders and make any approximation explicit.
- [x] Copy caller-owned matrices and MPOs. Keep prepared operator data
      consistent when observables are reused with different states or simulation
      parameters.
- [x] Test conversions against independent small dense operators. Include
  asymmetric operators, complex Hermitian matrices, nonadjacent and reversed
  sites, mixed local dimensions, and malformed MPOs.

### 2A. Close construction and validation contracts

- [x] Audit the existing tests against the completed checklist. Add tests only
      for missing contracts or concrete regressions; retain working
      construction.
- [x] Check matrix and MPO Hermiticity tolerances near their acceptance limits.
      Document the norm used by each check. Include zero operators, small
      coefficients, cancellation in Pauli sums, and rescaled equivalent MPO
      gauges. Verify that compression of the temporary Hermiticity residual
      cannot hide a violation of the declared tolerance or alter the stored
      operator.
- [x] Define the mutation rules for source and prepared observables. Verify
  caller-data isolation, reuse with different state dimensions, and repeated
  preparation. Ensure cached preparation cannot silently use stale operator
  data under the supported mutation rules.
- [x] Audit preparation in `simulator.py`, `simulation_program.py`, and direct
  measurement entry points. Invalid support and dimensions must fail before
  evolution or worker launch. Diagnostic cuts and backend capabilities are
  completed in chunk 4.
- [x] Make the construction documentation state the current measurement limits.
      Check `to_mpo()` documentation for compact support and `mpo_sites`;
      callers must not mistake a local MPO for a full-chain operator.

Acceptance: the construction contracts pass in
`tests/core/data_structures/test_observable.py` and `test_mpo.py`, with targeted
simulator and program tests for validation before execution. This batch does not
need to add general measurement support.

## 3. Evaluate general MPO expectations

`mixed_expectation()` contracts compact and full-chain MPOs directly.
`MPS.expect()` and `evaluate_observables()` use that contraction for general
operators and retain the local contraction path for small adjacent operators.

### 3A. Add the direct contraction

- [x] Contract `bra`, prepared MPO tensors, and `ket` directly. Support both
  full-chain MPOs and compact MPOs with identity action outside `mpo_sites`.
  Include the outer state environments when the gauge is unknown.
- [x] Validate matching chain lengths and local dimensions. Allow bra and ket to
      have different virtual bond dimensions and orthogonality centers.
- [x] Return the raw complex matrix element `<bra|O|ket>`. Do not apply the MPO
      to a copied state, truncate bonds, normalize inputs, or build a dense
      operator. A Hermitian operator can have a complex mixed matrix element.
- [x] Preserve both states, all observable tensors, and center metadata. The
  contraction must also work when the center is unknown.
- [x] Replace the apply-to-copy path in `MPS.mixed_expectation()` with the
      direct contraction while keeping its public call form.

Acceptance: independent dense references in `test_mps.py` cover distinct bra and
ket, complex Hermitian operators, mixed local dimensions, reversed and
nonadjacent sites, general MPOs, and zero operators. Check the scaling rule
`<a bra|O|b ket> = conj(a) * b * <bra|O|ket>` and exact input preservation.

### 3B. Connect direct and batched MPS measurements

- [x] Route `MPS.expect()` and `evaluate_observables()` through the general
      contraction for long-range correlations, operators on several sites, Pauli
      sums, and supplied MPOs. Remove local-site assumptions from general
      dispatch.
- [x] Preserve fast one-site and adjacent two-site evaluation when the gauge
      permits it. Reuse center shifts on a working copy for many local
      measurements. Compare the cost with the pre-3B baseline in chunks 5A and
      5B.
- [x] Keep local and general normalization conventions consistent. Direct
      expectation values return `<psi|O|psi>` and scale by `abs(a)**2` when the
      state is scaled by `a`; simulation backends keep their existing
      normalization flow.
- [x] Prepare operator data once per run and reuse it across time steps and
      trajectories. Direct calls must still validate unprepared inputs. Keep
      state contraction environments local to the current measurement state.
- [x] Use one real-result check for operator expectations: reject non-finite
  values and excessive absolute imaginary residuals using a documented,
  scale-aware tolerance. Do not discard a negative imaginary residual or use
  assertions to validate user inputs. Keep mixed matrix elements complex.

Acceptance: direct and batched results agree with dense references and each
other for known, displaced, and unknown centers, normalized and rescaled states,
tiny coefficients, and zero expectations. Existing local measurement tests pass,
and general measurements preserve the source state.

## 4. Integrate backends, diagnostics, and results

- [x] Replace observable gate metadata in consumers and connect state-dependent
  preparation to simulator workers and program compilation.

Remaining limits: analog embedding requires local matrices on at most two sites;
simulation parameters reject mixed bitstring/operator requests; dense analog
backends reject bitstrings. Diagnostic placeholders can still produce zero
results in dense backend measurement paths.

### 4A. Support operators in each backend

- [ ] Extend `analog/utils.py` embedding and the `mcwf.py` and `lindblad.py`
  preprocessors to accept all prepared Hermitian operators. Keep conversions
  outside measurement loops and preserve efficient sparse local embeddings.
- [ ] Use the existing `MPO.to_matrix_mps_order()` and `to_sparse_matrix()`
      where appropriate. Account for compact support and mixed local dimensions.
      `MPO.to_matrix()` uses a different basis order; do not pass its output to
      a state-vector backend without the required permutation.
- [ ] Compute vector expectations as `<psi|O|psi>` and density-matrix
      expectations as `Tr(O rho)`. Apply the same real-result validation as the
      MPS path. Dense or sparse operator construction belongs only in backends
      that need it.
- [ ] Verify that digital MPS and analog MPS entry points reach the general
  measurement path, including each supported analog solver.

Acceptance: small deterministic simulations agree with independent references
across MPS, vector MCWF, and density-matrix Lindblad backends. Include an
asymmetric complex operator, nonadjacent and reversed support, an operator on
three sites, a Pauli sum, and a supplied MPO. Exercise mixed dimensions wherever
the state and evolution backend support them.

### 4B. Handle projectors and diagnostics explicitly

- [ ] Give bitstring requests a linear-projector representation while retaining
      efficient MPS amplitude, vector amplitude, and density-diagonal
      evaluation. Preserve the convention that the first character refers to
      site 0, and validate the requested basis state against local dimensions.
- [ ] Allow bitstrings alongside operator observables. Remove the simulation
      parameter restriction and dense-backend rejection only when those paths
      can evaluate and aggregate the mixed requests correctly.
- [ ] Validate entropy and Schmidt-spectrum cuts during preparation: two
      adjacent, distinct, in-range sites. Keep diagnostics separate from
      operator contraction and retain their supported result semantics.
- [ ] State which backend/diagnostic combinations are supported. Reject other
      combinations before execution, including direct backend calls; remove
      placeholder zero results. Do not introduce a new mixed-state entropy
      meaning under the existing diagnostic name.

Acceptance: mixed projector/operator requests agree across backends. Supported
diagnostics retain their values and shapes; invalid cuts and unsupported
combinations raise explicit errors before evolution.

### 4C. Verify sampling, workers, and results

- [ ] Preserve user-list order and duplicate entries through sorting and result
  lookup. Full-chain observables must not require a local site index. Use
  observable type metadata for dispatch instead of special name checks.
- [ ] Cover final measurements, intermediate analog times, digital sampling
      barriers, noisy trajectories, and runs that request observables with
      shots.
- [ ] Check prepared-observable serialization and reuse in process workers.
  Preserve real expectation buffers, trajectory statistics, time axes, and
  diagnostic arrays through aggregation.
- [ ] Check `SimulationProgram` segment preparation, repeated measurements,
  segment boundaries, result stitching, and supported state representations.
- [ ] Audit remaining observable consumers, including characterization helpers,
  for assumptions about a local matrix, a site index, or one scalar result.

Acceptance: tests in `tests/analog/`, `tests/digital/`,
`tests/core/data_structures/`, `tests/test_simulator.py`, and
`tests/test_simulation_program.py` cover the affected contracts. Test worker
logic directly where possible and include a process-worker serialization check.
Use fixed seeds or controlled trajectories for noisy regression tests.

## 5. Validate performance and document the API

### 5A. Capture the pre-3A baseline

- [ ] Record repeatable runtime and peak-memory measurements for one local
  observable, all single-site observables, and many adjacent two-site
  observables. Include known and unknown MPS centers.
- [ ] Record the commit, environment, seeds, chain length, local dimensions,
      state bond dimensions, observable count, and thread limits. Separate
      operator preparation from repeated measurement cost; use warmups and
      repeated runs.

Acceptance: save the benchmark procedure and results so the same workloads can
be rerun after integration. Use bounded workloads that fit available memory.

### 5B. Compare performance after chunk 4

- [ ] Repeat 5A and investigate avoidable local-measurement regressions before
  declaring completion. Record runtime and memory separately.
- [ ] Measure long-range products, Pauli sums, and full-chain MPO expectations
      while varying chain length, MPS bond dimension, and MPO bond dimension.
      Check that general measurement does not construct a modified MPS or a
      dense Hilbert-space operator; keep dense references limited to small
      cases.
- [ ] Confirm that repeated time steps and trajectories reuse prepared operators
  without retaining stale state environments or growing memory over time.

Acceptance: retain reproducible measurements and explain remaining costs. Avoid
flaky wall-clock thresholds in ordinary unit tests.

### 5C. Complete documentation and release checks

- [ ] Update `docs/examples/simulation_parameters.md` and API docstrings with
  working examples for named operators, custom matrices, nonadjacent sites,
  supplied MPOs, Pauli sums, mixed projectors, and diagnostics.
- [ ] Document Hermiticity and real-result tolerances, normalization, tensor and
      basis order, compact MPO support, dimension checks, and reuse/mutation
      rules. Include a backend support table and diagnostic result semantics.
- [ ] Check public imports and Google-style docstrings. Keep `ObservableType`,
      `id`, and gate-independent definitions consistent across code and
      examples.
- [ ] Complete `CHANGELOG.md` and `UPGRADING.md` for the final feature. Include
      diagnostics and MPO inputs in accepted-input descriptions and distinguish
      construction from measurement support. Use required PR and author
      references; no compatibility period or migration tutorial is needed.
- [ ] Run the relevant integration tests and full test suite after the targeted
      checks pass. Build documentation with `uvx nox --non-interactive -s docs`
      and verify the examples against supported execution paths.
- [ ] Run `uvx nox -s lint` with all hooks passing before submission. Record any
      unavailable check as a blocker, rather than treating a skipped hook as a
      pass.

For every implementation batch, add or update tests in the owning component's
test tree, run the affected tests, and run `uvx nox -s lint`. Update affected
docstrings and user-facing notes with the behavior change; 5C is the final
audit.

## Completion criteria

- [x] Standard named and custom Hermitian observables use the same public
  construction pattern without depending on gate objects.
- [ ] Long-range correlations, custom operators on several sites, and supplied
  MPOs agree with independent references across supported backends.
- [ ] Non-Hermitian operators and invalid dimensions fail clearly before
  evolution begins.
- [ ] Measurements preserve the state and retain local-observable efficiency.
- [ ] Diagnostics and projectors have explicit behavior across sampling and
  result aggregation paths.
- [ ] Tests, examples, release notes, and required checks cover the final API.
