# Noisy Equivalence Checking TODO

## Chunk 1: Fix `NoiseModel` handling

- [x] Derive sampled Pauli gates from normalized `matrix`/`factors`, not the
  process name.
- [x] Preserve descending-site crosstalk ordering and handle supported
  `longrange_crosstalk_*` processes consistently.
- [x] Validate sites and operator dimensions against the circuit before running
  trajectories.
- [x] Reject unsupported processes explicitly instead of silently ignoring them.
- [x] Add regressions for custom matrix overrides, descending crosstalk, and
  out-of-range sites.

## Chunk 2: Match YAQS circuit-noise semantics

- [x] Use the same noise-opportunity rule as the digital Simulator.
- [x] Do not noise single-qubit gates.
- [x] Apply noise after gates on two or more qubits, including CCX, when the
  selected equivalence backend supports the gate.
- [x] Share the opportunity logic so the Simulator and checker cannot drift.
- [x] Test H, CX, and CCX circuits explicitly.

## Chunk 3: Fix ensemble result semantics

- [x] Report noisy-channel process fidelity by averaging squared trajectory
  overlaps, rather than averaging root overlaps.
- [x] Keep noisy `equivalent` as a finite-sample threshold result, not an exact
  equivalence certificate.
- [x] Return a fidelity error estimate, with sensible behavior for `num_traj=1`.
- [x] Update the noisy result type and tests accordingly while leaving noiseless
  checks unchanged.

## Chunk 4: Harden parallelism and seeding

- [x] Cap worker processes at `min(num_traj, resolved_workers)`.
- [x] Make `parallel=False` keep noisy checks serial.
- [x] Reject negative seeds consistently with other YAQS APIs.
- [x] Replace the degenerate serial/process parity test with one that can detect
  different sampled trajectories.

## Chunk 5: Update documentation and release notes

- [ ] Describe the feature as stochastic Pauli-channel comparison, not general
  noisy-channel equivalence.
- [ ] Document the corrected metric, uncertainty, supported noise processes, and
  gate-opportunity rule.
- [ ] Correct the relative-operator order to `U_noisy^dagger U_ideal`.
- [ ] Credit both `@yiranwang-phys` and `@aaronleesander` in `CHANGELOG.md`.
- [ ] Remove unrelated `simulator.py` changes if they are no longer needed.

## Final validation

- [ ] Run the focused equivalence and digital-noise tests.
- [ ] Run `uvx nox -s lint`.
- [ ] Run the full test suite and documentation build.
- [ ] Confirm all PR checks pass on the final commit.
