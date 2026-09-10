# Response-matrix paper alignment

## Goal

Align the YAQS operational-memory response matrix with the definition used in
the response-matrix paper. The paper defines

\[ (V_c)_{(j,\alpha),i} = p_{ij}\,\operatorname{Tr}(P_\alpha\rho_{ij}), \qquad
P_\alpha\in\{I,X,Y,Z\}, \]

where `i` labels a conditioned history, `j` labels a future probe, and `p_ij` is
the probability of every retained outcome in the complete record. Rows therefore
label future records and columns label histories.

Use these branches as follows:

- Implement the method, API, tests, and documentation on
  `response-matrix-update`.
- Use the experiment campaign committed on `response-matrix-tests` for
  integration and numerical verification.
- Record both commit hashes for each verification run.

The quantum-memory witness experiment is not part of this update. The updated
response-matrix contract must, however, provide the data that the witness will
need later.

## Chunk 1: Remove centering from the response-matrix method

- [x] Make the raw, branch-weighted matrix the canonical response matrix.
- [x] Remove centering from `assemble_response_matrix`.
- [x] Remove the `center` argument from the canonical assembly path.
- [x] Remove or deprecate `center_rows` as a public operational-memory helper.
- [x] Replace the `(response_matrix_raw, response_matrix)` return pair with one
      canonical response matrix. If compatibility requires a transition period,
      document the temporary behavior and its removal date.
- [x] Remove the redundant `return_raw` option and `response_matrix_raw` result
      entry from the public characterization path.
- [x] Update `compute_spectrum` and result docstrings so that they accept and
      describe the raw response matrix, not a centered matrix.
- [x] Keep any centered quantity outside the paper-facing method and give it a
      distinct diagnostic name.

## 2. Use the paper's matrix orientation

- [x] Keep evaluated probe data in its current three-dimensional layout:
      `(n_histories, n_future_probes, n_output_channels)`.
- [x] Assemble the public two-dimensional matrix with shape
      `(n_future_probes * n_output_channels, n_histories)`.
- [x] Use the following element mapping:

```python
response_matrix[j * n_output_channels + alpha, i] = weights[i, j] * pauli[i, j, alpha]
```

- [x] Order each future probe's output channels as `I, X, Y, Z`.
- [x] Document that left singular vectors describe future-response directions
      and right singular vectors describe combinations of histories.
- [x] Update examples and plot labels that place histories on the row axis.
- [ ] Add an upgrade note: the old public matrix is the transpose of the new
      matrix, apart from the other changes in this update.

The transpose alone must not change singular values, rank, Frobenius norm, or
response entropy. It changes the meaning of singular vectors and all direct
matrix indexing.

## 3. Include the identity response

- [x] Preserve all four tomography channels during assembly. Do not reduce
      `I, X, Y, Z` data to `X, Y, Z`.
- [x] Require a four-channel input for the canonical paper-facing method, or
      provide an explicit and validated conversion for older three-channel
      inputs. Do not infer ambiguous input semantics silently.
- [x] Multiply the identity expectation by the branch weight. The identity entry
      is `p_ij`, not `1`.
- [x] Update the formal future-record dimension from `3 * n_future_probes` to
      `4 * n_future_probes` wherever applicable.
- [x] Verify that the identity rows provide the deterministic normalization
      direction required by the paper's history-branch probability and rank-one
      result.

## 4. Use complete retained-outcome weights

This requirement does not change the current unitary-future benchmarks. It is
required for selected future outcomes, the reset-bridge experiment, and the
later witness experiment.

- [ ] Define `weights[i, j]` as the probability of all retained outcomes in the
      history and future record represented by matrix entry `(j, i)`.
- [ ] For deterministic future operations, verify that this reduces to the
      history-branch probability.
- [ ] For non-trace-preserving future operations, include their outcome
      probabilities instead of stopping the product at the causal cut.
- [ ] Prefer returning a subnormalized final branch from a backend. If a backend
      returns a normalized state and a separate probability, verify that their
      product equals the subnormalized response entry.
- [ ] Keep the canonical construction linear in probabilities. The paper's
      response matrix uses `beta = 1`; treat other powers as separately named
      diagnostics if YAQS retains them.

## 5. Update the public API and documentation

- [x] Update the response-matrix implementation and exports.
- [x] Update `run_memory_characterization` and `CharacterizationResult`.
- [x] Rename variables and text that call four-channel data `pauli_xyz`.
- [x] Update the characterization and quick-start documentation.
- [x] State the exact shape, channel order, axis meaning, and weighting rule in
      every public return-value description.
- [ ] Update `CHANGELOG.md` and `UPGRADING.md` because the matrix values, shape,
      orientation, and public API change.

Likely files include:

- `src/mqt/yaqs/characterization/memory/operational_memory/response_matrix.py`
- `src/mqt/yaqs/characterization/memory/operational_memory/run.py`
- `src/mqt/yaqs/characterization/memory/operational_memory/results.py`
- `src/mqt/yaqs/characterization/memory/operational_memory/__init__.py`
- `docs/examples/characterization.md`
- `docs/examples/quickstart.md`

## 6. Add contract tests on `response-matrix-update`

- [x] Use a non-square sentinel input, such as two histories and three future
      probes, to test every output index and prevent an unnoticed transpose.
- [x] Assert the output shape `(4 * n_future_probes, n_histories)`.
- [x] Assert the exact mapping
      `V[4 * j + alpha, i] == weights[i, j] * pauli[i, j, alpha]`.
- [x] Assert that no mean subtraction occurs.
- [x] Assert that the identity entry equals the weighted branch probability.
- [x] Add a maximally mixed output test. The `X, Y, Z` entries may vanish, but
      the response matrix must remain nonzero because of `I`.
- [x] Add a memoryless example whose raw response matrix has rank one.
- [x] Verify that transposing the old raw XYZ block preserves its singular
      values before adding the identity block.
- [ ] Test a retained future outcome whose probability depends on both history
      and future indices.
- [x] Update public-run and result-container tests for the single canonical
      response matrix.
- [x] Remove tests that require centering or require IXYZ and XYZ inputs to
      produce the same matrix.

Run targeted tests during implementation:

```bash
uv run pytest tests/characterization/memory/operational_memory/test_response_matrix.py
uv run pytest tests/characterization/memory/operational_memory/test_run.py
uv run pytest tests/characterization/memory/operational_memory/test_results.py
uv run pytest tests/test_memory_characterizer.py
```

Then run the repository checks:

```bash
uvx nox -s lint
uvx nox -s tests
```

## 7. Verify with `response-matrix-tests`

- [ ] Create a temporary integration worktree or branch that contains the
      `response-matrix-tests` campaign and the completed
      `response-matrix-update` implementation.
- [ ] Fix experiment adapters that still request or assume centered matrices.
- [ ] Use one explicit singular-value resolution rule for all compared runs.
- [ ] Run a small non-square smoke case and compare selected entries with the
      defining formula before running the full campaign.
- [ ] Verify that an orientation-only transpose preserves the old scalar
      spectrum.
- [ ] Verify that removing centering and adding the identity channel produce the
      expected new spectrum. Do not compare these as orientation-only changes.
- [ ] Rerun the landscape, mode, convergence, reset-bridge, finite-size, and
      process-tensor comparison campaigns at the settings stated in the paper.
- [ ] Correct the reset-bridge sequence and include the retained bridge-outcome
      probabilities before accepting its new result.
- [ ] Save the raw IXYZ responses, complete weights, probe definitions, random
      seeds, resolution rule, YAQS commit, and experiment commit with each run.
- [ ] Regenerate every affected table and figure from the verified outputs.

## Acceptance criteria

- [ ] The primary YAQS response matrix is raw and uncentered.
- [ ] Its rows are future records and its columns are histories.
- [ ] It contains weighted `I, X, Y, Z` responses in a documented order.
- [ ] Its weights include every retained outcome represented by an entry.
- [ ] Unit tests check matrix values and semantics, not only singular values.
- [ ] Documentation, changelog, and upgrade guidance describe the breaking
      change.
- [ ] Targeted tests, the full test suite, and lint pass.
- [ ] The `response-matrix-tests` campaign reproduces the updated paper figures
      from recorded inputs and commit hashes.
