# Anti-Gravity Repository Summary

This file is a quick orientation note for an AI coding agent working on noisy-circuit
state preparation with Krotov-style optimization in MQT YAQS. It intentionally avoids
experiment outcomes, parameter sweeps, and attempted solution ideas. Use it to find the
implemented building blocks faster.

## Project Shape

MQT YAQS is a Python package under `src/mqt/yaqs` for tensor-network simulation of
open quantum systems and noisy quantum circuits. The package uses a `src` layout,
`uv` for environments, `nox` for common tasks, `pytest` for tests, and Sphinx/MyST for
documentation.

Important top-level locations:

- `src/mqt/yaqs/`: package source.
- `tests/`: unit and integration tests.
- `docs/`: user documentation and examples.
- `experiments/`: exploratory scripts and generated artifacts; useful as examples, but
  not part of the stable package API.
- `pyproject.toml`: dependencies, pytest config, ruff config, and tooling groups.
- `AGENTS.md`: repository-specific development commands and contribution rules.

## Main Public Entry Points

- `mqt.yaqs.Simulator`: high-level simulation API.
- `mqt.yaqs.State`: user-facing initial-state specification.
- `mqt.yaqs.Result`: simulation result container.
- `mqt.yaqs.EquivalenceChecker`: circuit equivalence checker.
- `mqt.yaqs.optimization`: Krotov and parameterized-circuit functionality.

The central simulation dispatch is in `src/mqt/yaqs/simulator.py`. `Simulator.run(...)`
accepts a `State`, an operator (`Hamiltonian` for analog simulation or `QuantumCircuit`
for digital simulation), optional `NoiseModel`, and one of the simulation-parameter
classes from `core/data_structures/simulation_parameters.py`.

## Core Data Structures

Look in `src/mqt/yaqs/core/data_structures/` for the objects that most simulation and
optimization code uses:

- `state.py`: `State`, with MPS, dense vector, and density-matrix representations.
- `mps.py`: matrix product state implementation.
- `mpo.py`: matrix product operator implementation.
- `hamiltonian.py`: Hamiltonian wrapper for analog simulation.
- `noise_model.py`: `NoiseModel`, including local, two-site, crosstalk, long-range, and
  sampled-strength noise descriptions.
- `simulation_parameters.py`: `AnalogSimParams`, `StrongSimParams`, `WeakSimParams`,
  `Observable`, simulation presets, and digital gate modes.
- `result.py`: `Result` and aggregation helpers for trajectories, observables,
  diagnostics, and measurement counts.

## Simulation Backends

Analog simulation code is in `src/mqt/yaqs/analog/`:

- `analog_tjm.py`: tensor jump method on MPS.
- `mcwf.py`: Monte Carlo wave-function simulation.
- `lindblad.py`: density-matrix Lindblad simulation.
- `ensemble.py`: deterministic unitary ensemble path.
- `utils.py`: dense/sparse embedding helpers for operators and observables.

Digital noisy-circuit simulation code is in `src/mqt/yaqs/digital/`:

- `digital_tjm.py`: circuit tensor-jump backend, Qiskit DAG layer processing, one- and
  two-qubit gate application, local noise handling, measurements, and strong/weak modes.
- `utils/dag_utils.py`: temporal-zone and DAG helper logic.
- `utils/contraction_utils.py`: MPO update and circuit-equivalence contraction logic.
- `utils/matrix_utils.py`: dense matrix helper routines for small circuits.
- `utils/scheduler_utils.py`: disjoint gate batching helpers.

For noisy state-preparation work, the digital backend and the Krotov optimizer both
rely on consistent gate-site ordering and MPS/MPO conventions. Check neighboring tests
before changing those conventions.

## Gate, Noise, and Circuit Libraries

Look in `src/mqt/yaqs/core/libraries/`:

- `gate_library.py`: gate classes and matrices used throughout the simulator and
  optimizer.
- `noise_library.py`: built-in jump/noise operators.
- `circuit_library.py`: helper constructors for model circuits and ansatz-style
  circuits, including matrix-product-disentangler related builders.
- `circuit_library_utils.py`: shared circuit-library helpers.

These files are useful when adding a new ansatz, changing supported gates, or mapping
between Qiskit circuits and YAQS gate primitives.

## Krotov Optimization Area

The relevant implementation lives in `src/mqt/yaqs/optimization/`:

- `parameterized_circuit.py`: lightweight gate-list representation for parameterized
  circuits. It defines `ParameterizedGate`, `ParameterizedCircuit`, derivative
  operators for supported one-parameter gates, and factory helpers for sequential and
  brickwall matrix-product-disentangler circuits.
- `krotov.py`: Krotov-style discrete adjoint optimizer on the MPS backend. It contains
  truncation/readout/options dataclasses, forward and backward sweeps, sample
  contributions, state-preparation losses/metrics, noisy trajectory handling, and
  online/batch/hybrid training entry points.
- `__init__.py`: exported optimization API.

Names to search for in `krotov.py`:

- `KrotovTruncation`, `KrotovReadout`, `KrotovOptions`, `KrotovTJMOptions`.
- `KrotovNoiseMap`, `KrotovTrajectory`, `KrotovResult`.
- `forward_states`, `forward_tjm_trajectory`, `backward_costates`.
- `state_preparation_loss`, `state_preparation_metrics`,
  `state_preparation_contribution`.
- `noisy_state_preparation_loss`, `noisy_state_preparation_metrics`,
  `noisy_state_preparation_contribution`,
  `noisy_state_preparation_cross_contribution`.
- `train_krotov_state_preparation_online`,
  `train_krotov_state_preparation_batch`,
  `train_krotov_state_preparation_hybrid`.
- `train_krotov_noisy_state_preparation_online`,
  `train_krotov_noisy_state_preparation_batch`,
  `train_krotov_noisy_state_preparation_hybrid`.

## Tests To Read First

Useful tests for this task:

- `tests/optimization/test_krotov.py`: Krotov optimizer behavior, state-preparation
  helpers, noisy state-preparation helpers, and parameterized-circuit interactions.
- `tests/optimization/test_supervised_qml_benchmarks.py`: benchmark helper coverage for
  optimization-related experiment support.
- `tests/core/libraries/test_circuit_library.py`: ansatz and circuit-library coverage.
- `tests/digital/test_digital_tjm.py`: digital noisy-circuit behavior, gate modes,
  long-range gates, measurements, and MPS-vs-Qiskit consistency checks.
- `tests/core/data_structures/test_noise_model.py`: noise-model validation and sampling.
- `tests/core/data_structures/test_state.py`: state initialization and representation
  behavior.

## Documentation To Check

Useful docs:

- `docs/examples/state_initialization.md`
- `docs/examples/simulator_initialization.md`
- `docs/examples/simulation_parameters.md`
- `docs/examples/strong_circuit_simulation.md`
- `docs/examples/weak_circuit_simulation.md`
- `docs/examples/equivalence_checking.md`
- `docs/tooling.md`
- `docs/contributing.md`

The API reference is generated from source via Sphinx autoapi.

## Development Commands

Common commands from `AGENTS.md`:

```console
uv sync
uv run pytest
uvx nox -s tests
uvx nox -s lint
uvx nox --non-interactive -s docs
```

Prefer targeted tests while iterating, then run the repository lint session before
submitting changes.

## Repo-State Notes For Agents

At the time this note was written, the active branch was `Noisy-State-Preparation`.
The working tree contained local modifications and untracked experiment files. Treat
those as existing user work unless explicitly instructed otherwise.
