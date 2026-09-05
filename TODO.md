# Observable API implementation plan

Make observable definitions independent of gates. Preserve the familiar
`Observable(operator, sites)` interface and support Hermitian operators supplied
as local matrices, operators on several sites, Pauli sums, or matrix product
operators (MPOs).

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
- Preserve observable order, sampling behavior, and result shapes for supported
  measurements.
- Keep this work focused on observables and their measurement paths.

## Target interface

The examples below describe the intended API; the extensions still need to be
implemented.

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

# Extend operator support.
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

- [ ] Add `core/data_structures/observable.py` and export `Observable` from
  `mqt.yaqs`. Update internal imports and remove the class definition from
  `simulation_parameters.py`.
- [ ] Add `core/libraries/observable_library.py` with explicit named operator
  factories. Include Pauli operators, identity, Pauli products, projectors,
  position, and other supported Hermitian named operators.
- [ ] Replace gate metadata with observable metadata for operator data, sites,
  names, and diagnostic requests. Keep per-run values on `Result`.
- [ ] Remove observable-only definitions from `gate_library.py`. Share basic
      operator data through a neutral module only where needed to avoid
      duplicate definitions. Preserve definitions that circuit or Hamiltonian
      code still uses.
- [ ] Resolve names explicitly. Interpret valid binary strings as projectors;
  reject unknown names and malformed strings with clear errors.
- [ ] Validate factory arguments and reject non-Hermitian named operators and
      custom matrices. A real expectation on one state is not a Hermiticity
      test.
- [ ] Update constructor and import tests in the corresponding test tree. Cover
      valid syntax, local dimensions, removed gate inputs, and invalid
      arguments.

## 2. Build and validate operator MPOs

- [ ] Define and document matrix basis order, site-list order, and MPO tensor
      order before implementing conversions. Preserve supported local
      conventions. Handle reversed sites by permuting operator axes with the
      sites.
- [ ] Resolve system-dependent dimensions when preparing a run. Validate site
      bounds, duplicate sites, matrix sizes, and physical dimensions before
      workers start. Direct measurement calls must perform equivalent
      validation.
- [ ] Build compact MPOs for named and custom operators. Insert identity tensors
      across gaps for operators on nonadjacent sites. Avoid allocating a
      separate full-chain MPO for every local observable.
- [ ] Accept existing MPO objects and add `Observable.from_pauli_sum(...)` using
      the existing MPO builders where their contracts fit. Validate Hermiticity
      of the represented sum.
- [ ] Validate supplied MPO tensors, finite values, boundary bonds, neighboring
  bond dimensions, and square physical legs.
- [ ] Check Hermiticity of the complete MPO through tensor contractions without
  constructing a full Hilbert-space matrix. Do not require each tensor to be
  Hermitian. Specify scale-aware tolerances and handle zero operators.
- [ ] Avoid silent observable truncation. Audit default cutoffs in reused MPO
  builders and make any approximation explicit.
- [ ] Copy caller-owned matrices and MPOs. Keep prepared operator data
      consistent when observables are reused with different states or simulation
      parameters.
- [ ] Test conversions against independent small dense operators. Include
  asymmetric operators, complex Hermitian matrices, nonadjacent and reversed
  sites, mixed local dimensions, and malformed MPOs.

## 3. Evaluate general MPO expectations

- [ ] Add direct MPS-MPO contraction for `bra * operator * ket`. Use this for
  general expectation values and mixed matrix elements where needed.
- [ ] Route long-range correlations, operators on several sites, Pauli sums, and
      supplied MPOs through the general contraction.
- [ ] Preserve fast one-site and adjacent two-site evaluation. Reuse suitable
  environments or center shifts when measuring several local observables.
- [ ] Make measurement independent of the MPS gauge. Preserve the input state,
  observable tensors, and orthogonality-center metadata.
- [ ] Keep normalization conventions consistent between local and general
  contractions. Test normalized and rescaled states.
- [ ] Prepare reusable operator data once per run, outside time-step and
  trajectory measurement loops.
- [ ] Check the absolute imaginary residual with a documented tolerance before
  storing a real result. Use explicit exceptions instead of assertions for
  invalid user input.
- [ ] Test against independent dense matrix elements on small product and
      entangled states. Cover known and unknown gauges, weak operator
      coefficients, zero operators, and state preservation.

## 4. Integrate backends, diagnostics, and results

- [ ] Replace `.gate` access in MPS helpers, analog embedding, simulation
  parameters, result aggregation, and all other observable consumers.
- [ ] Support the same Hermitian operators in digital MPS simulation, analog MPS
      trajectories, vector MCWF, and density-matrix Lindblad simulation.
- [ ] Convert to sparse or dense operators only when the backend needs those
      forms. Keep MPS measurement free of full Hilbert-space matrix
      construction.
- [ ] Verify basis order across backends. `MPO.to_matrix()` and MPS state-vector
  conversions currently use different site significance conventions.
- [ ] Integrate final sampling, intermediate analog times, digital sampling
  barriers, noisy trajectories, and `SimulationProgram` segments.
- [ ] Preserve user-list order, including duplicate observables, when workers
  choose a different evaluation order. Handle full-chain operators without
  assuming every observable has a local site index.
- [ ] Represent bitstring projectors as operators, retain efficient probability
  evaluation, and allow them alongside other observables.
- [ ] Dispatch entropy and Schmidt spectra as diagnostics with validated cuts
      and their existing aggregation semantics. Raise clear errors for
      unsupported backend/diagnostic combinations instead of returning
      placeholder zeros.
- [ ] Keep expectation buffers real and preserve trajectory and time-axis
  semantics. Confirm prepared observables work with process workers.
- [ ] Add backend comparison and integration tests for sampling, aggregation,
  observable ordering, program stitching, and mixed measurement requests.

## 5. Validate performance and document the API

- [ ] Measure the cost of many local observables before and after the change.
      Use the same states and operators to detect avoidable runtime or memory
      growth.
- [ ] Check that long-range and full-chain MPO measurements use tensor
  contractions without building a modified state or dense operator.
- [ ] Update API documentation and examples for named observables, custom local
  matrices, operators on several sites, MPOs, Pauli sums, and diagnostics.
- [ ] Document Hermiticity tolerances, site and basis order, dimension checks,
  supported backends, and diagnostic result semantics.
- [ ] Remove documentation that presents arbitrary circuit gates as observables.
- [ ] Record the feature and breaking API changes in `CHANGELOG.md` and
  `UPGRADING.md`. Use the required PR and author references when available;
  no migration tutorial or compatibility period is needed.
- [ ] Run targeted tests for each implementation batch. Keep tests in the
  component's corresponding `tests/` directory and test supported behavior.
- [ ] Run `uvx nox -s lint` after every change batch. Build the affected
  documentation and run the relevant integration tests before handoff.

## Completion criteria

- [ ] Standard named and custom Hermitian observables use the same public
  construction pattern without depending on gate objects.
- [ ] Long-range correlations, custom operators on several sites, and supplied
  MPOs agree with independent references across supported backends.
- [ ] Non-Hermitian operators and invalid dimensions fail clearly before
  evolution begins.
- [ ] Measurements preserve the state and retain local-observable efficiency.
- [ ] Diagnostics and projectors have explicit behavior across sampling and
  result aggregation paths.
- [ ] Tests, examples, release notes, and required checks cover the final API.
