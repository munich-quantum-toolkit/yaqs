# Upgrade Guide

This document describes breaking changes and how to upgrade. For a complete list of changes including minor and patch releases, please refer to the [changelog](CHANGELOG.md).

## [Unreleased]

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
[0.3.3]: https://github.com/munich-quantum-toolkit/yaqs/releases/tag/v0.3.3
[0.3.2]: https://github.com/munich-quantum-toolkit/yaqs/compare/v0.3.1...v0.3.2
