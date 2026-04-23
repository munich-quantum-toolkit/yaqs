# CLAUDE.md

This file provides guidance for Claude Code when working on MQT YAQS.
See [AGENTS.md](AGENTS.md) for the canonical developer workflow (setup, linting, testing commands).

## Project overview

MQT YAQS (**Y**et **A**nother **Q**uantum **S**imulator) simulates open quantum systems and noisy
quantum circuits using tensor-network (MPS/MPO) methods. The primary simulation engine is the
Tensor Jump Method (TJM). It is part of the Munich Quantum Toolkit (MQT) and lives under the
`mqt.yaqs` namespace package.

## Source layout

```
src/mqt/yaqs/
  simulator.py                    # top-level run() entry point
  core/
    data_structures/
      networks.py                 # MPS / MPO classes
      noise_model.py              # NoiseModel (process dicts + jump operators)
      simulation_parameters.py   # AnalogSimParams, DigitalSimParams, Observable
    libraries/
      gate_library.py             # BaseGate hierarchy + GateLibrary registry
      noise_library.py            # pre-built NoiseModel factories
      circuit_library.py          # Qiskit-circuit helpers
    methods/
      tdvp.py / tdvp_numba.py     # TDVP time evolution (Numba-accelerated)
      stochastic_process.py       # quantum-jump stochastic trajectories
      dissipation.py              # Lindblad dissipator application
      decompositions.py           # gate decompositions into MPO form
  analog/                         # Hamiltonian (Lindblad / MCWF) solvers
  digital/                        # noisy-circuit simulator + equivalence checker
  characterization/               # process-tensor tomography
tests/                            # mirrors src/ structure exactly
```

## Key abstractions

### Gates (`core/libraries/gate_library.py`)

- `BaseGate` is the root class. Every operator is a `BaseGate`.
- `local_dim` defaults to 2 (qubit); pass a larger value for qutrits / bosonic modes.
- **`set_sites()` is lazy**: for single-qubit gates it just records the site index.
  For two-qubit gates it also reshapes `tensor` to `(2,2,2,2)`, computes `generator`,
  and builds `mpo_tensors` via `extend_gate()`. Accessing `mpo_tensors` before calling
  `set_sites()` raises `AttributeError`.
- `GateLibrary` stores **class references**, not instances — use `GateLibrary.rx([theta])`,
  not `GateLibrary.rx`.
- `Crosstalk(gate1, gate2)` is the Kronecker product `gate1 ⊗ gate2`.
  `swapped_matrix` holds `gate2 ⊗ gate1` for reversed site ordering.

### Noise model (`core/data_structures/noise_model.py`)

Each process is a dict with mandatory keys `name`, `sites`, `strength`.
`NoiseModel.__init__` auto-fills:

- `matrix` — for single-site processes and adjacent two-site processes.
- `factors` — for long-range two-site `Crosstalk` processes (tuple of two 1-site matrices).

`name` can be a `GateLibrary` attribute string or a `BaseGate` instance directly.
`strength` can be a plain float or a dict describing a distribution
(`normal`, `lognormal`, `truncated_normal`) for static disorder; call `.sample()` to draw.

### Networks (`core/data_structures/networks.py`)

- `MPS` — Matrix Product State; represents the quantum state.
- `MPO` — Matrix Product Operator; represents Hamiltonians, gates, and observables.
- MPO tensors have shape `(phys_out, phys_in, bond_left, bond_right)`.

## Development conventions

- **Docstrings**: Google style, enforced by ruff. See existing classes for examples.
- **No inline comments** unless the reason is non-obvious (a workaround, a hidden invariant).
- **Type annotations** on every function; `ty` (not mypy) is the type checker.
- **Tests**: mirror the source path — `src/mqt/yaqs/core/foo.py` → `tests/core/test_foo.py`.
  Use `assert_array_equal` / `assert_allclose` from `numpy.testing` for array comparisons.
- **Parallel tests**: pytest runs with `--numprocesses=auto` (xdist). Coverage requires
  `--override-ini="addopts="` to disable parallelism.
- **No `assert` in production code** outside of deliberate guard clauses in `NoiseModel`.
- **Template files**: never edit files whose first line starts with
  `This file has been generated from an external template`.

## Running checks

```bash
# Lint + format (runs all pre-commit hooks locally)
uvx nox -s lint

# Tests (single module, fast)
python -m pytest tests/core/libraries/test_gate_library.py --override-ini="addopts="

# Full test suite
uv run pytest

# Coverage for one module
python -m pytest tests/core/libraries/test_gate_library.py \
  --override-ini="addopts=" \
  --cov=mqt.yaqs.core.libraries.gate_library \
  --cov-report=term-missing
```

## Commit messages

Every commit must include:

```
Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```
