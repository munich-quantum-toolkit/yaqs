<!-- Entries in each category are sorted by merge time, with the latest PRs appearing first. -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on a mixture of [Keep a Changelog] and [Common Changelog].
This project adheres to [Semantic Versioning], with the exception that minor releases may include breaking changes.

## [Unreleased]

### Added

- added deterministic unitary analog ensemble evolution by allowing `simulator.run` to accept `list[MPS]` as `initial_state`
- added optional autocorrelator evaluation (`compute_autocorrelator`) for unitary analog ensemble runs
- added optional two-time correlator evaluation (`two_time_correlators`) for unitary analog ensemble runs, with results written to `two_time_correlator_results` on the simulation time grid
- added periodic-wrap support for two-site observables on sites `(L-1, 0)` in `apply_observable_inplace` (full ring spin-current correlators)
- added `unitary_ensemble_spin_current_row_worker` to accumulate bond-bond spin-current matrix elements per ensemble member
- added benchmark script `tests/analog/bench_xxz_transverse_unitary_ensemble_autocorrelator.py` for transverse-field XXZ (Pauli) chains with QuSpin ED comparison (`YAQS_J`, `YAQS_DELTA`, `YAQS_FIELD`; set `YAQS_DELTA=YAQS_J` for XXX)
- added script `scripts/plot_unitary_ensemble_autocorrelator_loglog_multi_N.py` to run default `simulator.run` unitary-ensemble autocorrelators for multiple chain lengths and save a combined log–log figure (`YAQS_LENGTHS`, `YAQS_NUM_STATES`, `YAQS_T_FINAL`, …)
- added `MPS(..., state="haar-random")` initializer using Haar-random isometries with optional bond-dimension cap via `pad`
- added process tomography for non-Markovian noise ([#344]) ([**@aaronleesander**])
- added ability to measure in X or Y basis ([#339]) ([**@aaronleesander**])
- added Arnoldi iteration as alternative to Lanczos method ([#338]) ([**@aaronleesander**])
- added Monte Carlo Wavefunction solver ([#333]) ([**@aaronleesander**])
- added MPO.from_matrix() method ([#331]) ([**@lucello**])
- added Lindblad solver for small systems ([#330]) ([**@aaronleesander**])
- Noise model strengths can now be sampled from a normal distribution ([#329]) ([**@aaronleesander**])

### Changed

- XXZ spin-current benchmark now computes the **full** periodic `J` autocorrelator in YAQS (all bond pairs, including cross terms and the wrap bond), matching exact diagonalization.
- changed Lindblad and MCWF solvers to use sparse implementation ([#338]) ([**@aaronleesander**])

### Fixed

- fixed analog unitary-ensemble autocorrelator time ordering to match `⟨Uψ|O U O|ψ⟩` / `vdot(Uψ, O U O ψ)` (the bra was previously taken on the `O|ψ⟩` branch, altering dynamics at `t>0`)
- fixed numba attempting to parallelize in already parallelized processes ([#337]) ([**@aaronleesander**])
- fixed potential memory leak in parallelization ([#336]) ([**@aaronleesander**])

## [0.4.0] - 2026-02-05

### Added

- added ability to schedule jumps ([#319]) ([**@aaronleesander**])
- added Bose-Hubbard Hamiltonian option ([#309]) ([**@lucello**])
- added multi-threading setting for noise-free ones ([#316]) ([**@aaronleesander**])
- Minor improvements to TDVP performance ([#311]) ([**@aaronleesander**])
- ⚡️ Improve Lanczos iteration and use numba for significant speedup ([#310]) ([**@aaronleesander**])

### Changed

- ♻️ Change Pauli summation to use finite state machine construction ([#308]) ([**@aaronleesander**])
- 🔧 Replace `mypy` with `ty` ([#304]) ([**@denialhaag**])

### Removed

- 👷‍♂️ Stop testing on x86 macOS systems ([#310]) ([**@aaronleesander**])

## [0.3.3] - 2026-01-12

### Added

- updated bib and readme with Nature Communications publication ([#298]) ([**@aaronleesander**])
- updates MPO class to allow construction of arbitrary Pauli Hamiltonians ([#216]) ([**@aaronleesander**, **@thilomueller**])
- added faster paths for TDVP dense effective Hamiltonian ([#280]) ([**@aaronleesander**])
- added more stable and faster SVD and QR implementation ([#278]) ([**@aaronleesander**])
- TDVP now utilizes a dense Hamiltonian for small tensor sizes (adjusted by global variable DENSE_THRESHOLD) ([#261]) ([**@aaronleesander**])
- Updated readability of TDVP subfunctions and Lanczos method ([#261]) ([**@aaronleesander**])

### Fixed

- Fixed bug where two-site dissipative processes were computed in loop ([#290]) ([**@aaronleesander**])
- Fixed a truncation bug in TDVP sometimes leading to over-truncation ([#274]) ([**@aaronleesander**])
- Updated Lanczos method's vdot order to match mathematical expectations ([#261]) ([**@aaronleesander**])

## [0.3.2] - 2025-10-16

_If you are upgrading: please see [`UPGRADING.md`](UPGRADING.md#032)._

### Added

- ✨ Make it possible to return final state without setting dummy observables ([#214]) ([**@aaronleesander**])
- 👷 Enable testing on Python 3.14 ([#212]) ([**@denialhaag**])

### Removed

- 🔥 Drop support for Python 3.9 ([#180]) ([**@denialhaag**])

### Fixed

- 🐛 Fix project bar of simulator ([#213]) ([**@aaronleesander**])

## [0.3.1] - 2025-08-29

_📚 Refer to the [GitHub Release Notes](https://github.com/munich-quantum-toolkit/yaqs/releases) for previous changelogs._

<!-- Version links -->

[Unreleased]: https://github.com/munich-quantum-toolkit/yaqs/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/munich-quantum-toolkit/yaqs/releases/tag/v0.4.0
[0.3.3]: https://github.com/munich-quantum-toolkit/yaqs/releases/tag/v0.3.3
[0.3.2]: https://github.com/munich-quantum-toolkit/yaqs/releases/tag/v0.3.2
[0.3.1]: https://github.com/munich-quantum-toolkit/yaqs/releases/tag/v0.3.1

<!-- PR links -->

[#344]: https://github.com/munich-quantum-toolkit/yaqs/pull/344
[#339]: https://github.com/munich-quantum-toolkit/yaqs/pull/339
[#338]: https://github.com/munich-quantum-toolkit/yaqs/pull/338
[#337]: https://github.com/munich-quantum-toolkit/yaqs/pull/337
[#336]: https://github.com/munich-quantum-toolkit/yaqs/pull/336
[#333]: https://github.com/munich-quantum-toolkit/yaqs/pull/333
[#331]: https://github.com/munich-quantum-toolkit/yaqs/pull/331
[#330]: https://github.com/munich-quantum-toolkit/yaqs/pull/330
[#329]: https://github.com/munich-quantum-toolkit/yaqs/pull/329
[#319]: https://github.com/munich-quantum-toolkit/yaqs/pull/319
[#316]: https://github.com/munich-quantum-toolkit/yaqs/pull/316
[#311]: https://github.com/munich-quantum-toolkit/yaqs/pull/311
[#310]: https://github.com/munich-quantum-toolkit/yaqs/pull/310
[#309]: https://github.com/munich-quantum-toolkit/yaqs/pull/309
[#308]: https://github.com/munich-quantum-toolkit/yaqs/pull/308
[#304]: https://github.com/munich-quantum-toolkit/yaqs/pull/304
[#298]: https://github.com/munich-quantum-toolkit/yaqs/pull/298
[#216]: https://github.com/munich-quantum-toolkit/yaqs/pull/216
[#290]: https://github.com/munich-quantum-toolkit/yaqs/pull/290
[#280]: https://github.com/munich-quantum-toolkit/yaqs/pull/280
[#278]: https://github.com/munich-quantum-toolkit/yaqs/pull/278
[#274]: https://github.com/munich-quantum-toolkit/yaqs/pull/274
[#261]: https://github.com/munich-quantum-toolkit/yaqs/pull/261
[#214]: https://github.com/munich-quantum-toolkit/yaqs/pull/214
[#213]: https://github.com/munich-quantum-toolkit/yaqs/pull/213
[#212]: https://github.com/munich-quantum-toolkit/yaqs/pull/212
[#180]: https://github.com/munich-quantum-toolkit/yaqs/pull/180

<!-- Contributor -->

[**@denialhaag**]: https://github.com/denialhaag
[**@aaronleesander**]: https://github.com/aaronleesander
[**@thilomueller**]: https://github.com/thilomueller
[**@lucello**]: https://github.com/lucello

<!-- General links -->

[Keep a Changelog]: https://keepachangelog.com/en/1.1.0/
[Common Changelog]: https://common-changelog.org
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html
[GitHub Release Notes]: https://github.com/munich-quantum-toolkit/yaqs/releases
