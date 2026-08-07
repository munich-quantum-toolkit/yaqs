# Manuscript experiments

This directory contains the complete benchmark and validation layer for
*Gate-local projected evolution for quantum circuit simulation with matrix
product states*. The production tensor-network implementation remains in the
normal MQT YAQS library; this directory contains the paper-specific protocols,
drivers, analysis, source data, and final figures.

## Result map

| Manuscript result | Code and source data |
| --- | --- |
| Table I: structural checks | `structural_checks/` |
| Figure 1: individual gates | `individual_gates/` |
| Figure 2: fixed-cap circuit trajectories | `circuit_benchmarks/long_trajectories/` |
| Figure 3: retained bond profiles | `circuit_benchmarks/figures/bond_profiles.py` and `circuit_benchmarks/output/bond_profiles.csv` |
| Figure 4: fixed-horizon accuracy and cost | `circuit_benchmarks/figures/fixed_horizon_cap_sweep.py` and `circuit_benchmarks/output/fixed_horizon_refinement/` |
| Variational-MPO controls | `variational_mpo.py` and the campaign-specific control drivers |
| SVD-threshold and TDVP-substep controls | `circuit_benchmarks/extensions/` and the corresponding compact output tables |

The four publication figures are collected in `figures/`. The CSV, JSON, and
compressed CSV files under each campaign's `output/` directory are the source
data used by the plotting and manuscript checks. Dense statevectors,
content-addressed task caches, checkpoint streams, and temporary solver states
are deliberately not stored because the drivers regenerate them.

## Reproduce

Create the YAQS development environment with `uv sync`. Each campaign README
lists the full protocol and commands. The final analysis-only steps are:

```bash
uv run python -m experiments.structural_checks.run
uv run python experiments/individual_gates/analyze.py
uv run python experiments/individual_gates/plot.py
uv run python -m experiments.circuit_benchmarks.long_trajectories.plot
uv run python -m experiments.circuit_benchmarks.figures.bond_profiles
uv run python -m experiments.circuit_benchmarks.figures.fixed_horizon_cap_sweep
```

Run the focused experiment tests with:

```bash
uv run pytest -n 0 tests/experiments \
  experiments/individual_gates/test_individual_gates.py \
  experiments/structural_checks/test_structural_checks.py -q
```

The numerical protocol, random seeds, tolerances, timing boundaries, and
hardware are specified in the manuscript and mirrored by the campaign
configuration files.
