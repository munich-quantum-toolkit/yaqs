# Noisy Equivalence Checking TODO

## Chunk 1: Fix `NoiseModel` handling

- [ ] Derive sampled Pauli gates from normalized `matrix`/`factors`, not the
  process name.
- [ ] Preserve descending-site crosstalk ordering and handle supported
  `longrange_crosstalk_*` processes consistently.
- [ ] Validate sites and operator dimensions against the circuit before running
  trajectories.
- [ ] Reject unsupported processes explicitly instead of silently ignoring them.
- [ ] Add regressions for custom matrix overrides, descending crosstalk, and
  out-of-range sites.

## Chunk 2: Match YAQS circuit-noise semantics

- [ ] Use the same noise-opportunity rule as the digital Simulator.
- [ ] Do not noise single-qubit gates.
- [ ] Apply noise after gates on two or more qubits, including CCX, when the
  selected equivalence backend supports the gate.
- [ ] Share the opportunity logic so the Simulator and checker cannot drift.
- [ ] Test H, CX, and CCX circuits explicitly.

## Chunk 3: Fix ensemble result semantics

- [ ] Report noisy-channel process fidelity by averaging squared trajectory
  overlaps, rather than averaging root overlaps.
- [ ] Do not present a finite Monte Carlo sample as an equivalence certificate.
- [ ] Return an uncertainty estimate and a clearly named statistical decision,
  including sensible behavior for `num_traj=1`.
- [ ] Update the noisy result type and tests accordingly while leaving noiseless
  checks unchanged.

## Chunk 4: Harden parallelism and seeding

- [ ] Cap worker processes at `min(num_traj, resolved_workers)`.
- [ ] Make `parallel=False` keep noisy checks serial.
- [ ] Reject negative seeds consistently with other YAQS APIs.
- [ ] Replace the degenerate serial/process parity test with one that can detect
  different sampled trajectories.

## Chunk 5: Update documentation and release notes

- [ ] Describe the feature as stochastic Pauli-channel comparison, not general
  noisy-channel equivalence.
- [ ] Document the corrected metric, uncertainty, supported noise processes, and
  gate-opportunity rule.
- [ ] Correct the relative-operator order to
  `U_noisy^dagger U_ideal`.
- [ ] Credit both `@yiranwang-phys` and `@aaronleesander` in
  `CHANGELOG.md`.
- [ ] Remove unrelated `simulator.py` changes if they are no longer needed.

## Final validation

- [ ] Run the focused equivalence and digital-noise tests.
- [ ] Run `uvx nox -s lint`.
- [ ] Run the full test suite and documentation build.
- [ ] Confirm all PR checks pass on the final commit.
