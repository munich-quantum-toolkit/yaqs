# Circuit benchmarks

This campaign supplies Figures 2--4 and the circuit-level numerical controls.
It compares gate-local two-site TDVP, MPO contract-and-truncate, TEBD+SWAP, and
the bounded variational-MPO controls on 16-site Ising and Heisenberg circuits.

## Physical protocol

- Open 16-site chains and open \(4\times4\) square lattices in snake MPS order.
- Ising: \(H=-\sum_{\langle ij\rangle}Z_iZ_j-\sum_iX_i\), initialized in
  \(\lvert0\cdots0\rangle\).
- Heisenberg: \(H=-\sum_{\langle ij\rangle}(X_iX_j+Y_iY_j+Z_iZ_j)\), initialized
  in a Neel/checkerboard product state.
- Symmetric second-order Trotter circuits with \(\Delta t=0.1\).
- Dense references execute the identical ordered gate list, so the reported
  infidelity excludes Trotter error.

The exact schedules, update modes, cap grids, tolerances, and timing rules are
defined in `config.py`, `circuits.py`, and the campaign-specific configuration
files.

## Figure 2: fixed-cap trajectories

```bash
uv run python -m experiments.circuit_benchmarks.long_trajectories.run
uv run python -m experiments.circuit_benchmarks.long_trajectories.timing
uv run python -m experiments.circuit_benchmarks.long_trajectories.variational_control
uv run python -m experiments.circuit_benchmarks.long_trajectories.plot
```

The source data are `long_trajectories/output/trajectory_rows.csv`, the raw
and summarized timing rows, and the variational trajectory and censor records.

## Figure 3: retained bond profiles

The base driver creates exact references and traced fixed-cap trajectories:

```bash
uv run python -m experiments.circuit_benchmarks.run --stage exact
uv run python -m experiments.circuit_benchmarks.run --stage resolution
uv run python -m experiments.circuit_benchmarks.run --stage trajectories
uv run python -m experiments.circuit_benchmarks.run --stage aggregate
uv run python -m experiments.circuit_benchmarks.figures.bond_profiles --refresh-data
```

The refresh option replaces `output/bond_profiles.csv` from the newly generated
checkpoint streams. Without it, the plotting command uses the retained portable
table directly.

## Figure 4 and controls

```bash
uv run python -m experiments.circuit_benchmarks.run --stage frontier
uv run python -m experiments.circuit_benchmarks.run --stage aggregate
uv run python -m experiments.circuit_benchmarks.extensions.fixed_horizon_refinement --stage all
uv run python -m experiments.circuit_benchmarks.extensions.fixed_horizon_cap_timing
uv run python -m experiments.circuit_benchmarks.extensions.fixed_horizon_tolerance_summary
uv run python -m experiments.circuit_benchmarks.extensions.substep_control
uv run python -m experiments.circuit_benchmarks.extensions.svd_threshold_control
uv run python -m experiments.circuit_benchmarks.extensions.variational_control
uv run python -m experiments.circuit_benchmarks.figures.fixed_horizon_cap_sweep
```

The compact tables under `output/fixed_horizon_refinement/`,
`output/svd_threshold_control/`, and `output/variational_mpo_control/`
support the figure and the threshold, substep, and variational-MPO statements
in the text. All timed primary-method runs use one warm-up and three
single-thread measurements; variational-MPO timings are single observations.

## Tests

```bash
uv run pytest -n 0 \
  tests/experiments/test_circuit_benchmark_analysis.py \
  tests/experiments/test_circuit_benchmark_schedules.py \
  tests/experiments/test_circuit_bond_profile_plot.py \
  tests/experiments/test_circuit_fixed_horizon_cap_sweep_plot.py \
  tests/experiments/test_circuit_fixed_horizon_cap_timing.py \
  tests/experiments/test_circuit_fixed_horizon_refinement.py \
  tests/experiments/test_circuit_fixed_horizon_tolerance_summary.py \
  tests/experiments/test_circuit_long_trajectory_figures.py \
  tests/experiments/test_circuit_resource_tracing.py \
  tests/experiments/test_svd_threshold_control.py \
  tests/experiments/test_circuit_substep_control.py -q
```
