<!-- Entries in each category are sorted by merge time, with the latest PRs appearing first. -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on a mixture of [Keep a Changelog] and [Common Changelog].
This project adheres to [Semantic Versioning], with the exception that minor releases may include breaking changes.

## [Unreleased]

### Added

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

[unreleased]: https://github.com/munich-quantum-toolkit/yaqs/compare/v0.3.2...HEAD
[0.3.2]: https://github.com/munich-quantum-toolkit/yaqs/releases/tag/v0.3.2
[0.3.1]: https://github.com/munich-quantum-toolkit/yaqs/releases/tag/v0.3.1

<!-- PR links -->

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

<!-- General links -->

[Keep a Changelog]: https://keepachangelog.com/en/1.1.0/
[Common Changelog]: https://common-changelog.org
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html
[GitHub Release Notes]: https://github.com/munich-quantum-toolkit/yaqs/releases
