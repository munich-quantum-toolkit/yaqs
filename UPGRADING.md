# Upgrade Guide

This document describes breaking changes and how to upgrade. For a complete list of changes including minor and patch releases, please refer to the [changelog](CHANGELOG.md).

## [Unreleased]

### Analog `simulator.run` now accepts `list[MPS]` for unitary ensembles

For analog simulations, `simulator.run` now supports passing a list of initial states:

```python
simulator.run([state_0, state_1, ...], hamiltonian_mpo, sim_params, noise_model=None)
```

This mode performs deterministic unitary evolution for each initial state in parallel
and aggregates observables across the ensemble.

Important: `list[MPS]` with noisy analog simulation is not supported in this release.
Use either:

- `list[MPS]` with `noise_model=None` (unitary ensemble), or
- a single `MPS` with noisy simulation (existing stochastic trajectory workflow).

### Optional autocorrelator outputs in `AnalogSimParams`

`AnalogSimParams` now includes optional autocorrelator controls:

- `compute_autocorrelator: bool`
- `autocorrelator_observable: Observable | None`

When enabled in unitary ensemble mode, the mean autocorrelator trajectory is written to
`sim_params.autocorrelator_results` with time grid in `sim_params.autocorrelator_times`.

The evaluated quantity matches the dense mixed amplitude
`⟨Uψ| O U O |ψ⟩` (equivalently `vdot(Uψ, O @ (U @ (O @ ψ)))` in NumPy conventions), where the first operator
is applied to the ket **before** time evolution and the second is inserted as in standard two-time correlators.
Earlier versions incorrectly conjugated this ordering at `t>0` by using `⟨U O ψ| O |U ψ⟩`.

### Optional two-time correlator outputs in `AnalogSimParams`

`AnalogSimParams` also supports explicit two-time correlator pairs via:

- `two_time_correlators: list[tuple[Observable, Observable]]`

For each pair `(A, B)`, the unitary-ensemble backend evaluates `⟨ψ(t)| A |φ_B(t)⟩` with
`|φ_B(t)⟩ = U(t) B |ψ(0)⟩`, and stores the ensemble mean in
`sim_params.two_time_correlator_results` with corresponding time grid in
`sim_params.two_time_correlator_times`.

### New `state="haar-random"` MPS initializer

An additional MPS initializer is available: `MPS(..., state="haar-random")`.
It constructs an entangled MPS from Haar-random isometries and uses `pad` as a bond-dimension cap for this mode.
If `pad` is not provided, it defaults to bond dimension 1.

### End of support for x86 macOS systems

Starting with this release, we can no longer guarantee support for x86 macOS systems.
x86 macOS systems are no longer tested in our CI and we can no longer guarantee that MQT YAQS installs and runs correctly on them.

## [0.3.2]

### End of support for Python 3.9

Starting with this release, MQT YAQS no longer supports Python 3.9.
This is in line with the scheduled end of life of the version.
As a result, MQT YAQS is no longer tested under Python 3.9 and requires Python 3.10 or later.

<!-- Version links -->

[Unreleased]: https://github.com/munich-quantum-toolkit/yaqs/compare/v0.3.3...HEAD
[0.3.2]: https://github.com/munich-quantum-toolkit/yaqs/compare/v0.3.1...v0.3.2
