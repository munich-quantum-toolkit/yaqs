# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Public package-surface regression tests for WP22H operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarks.state_preparation import phase2
from benchmarks.state_preparation.phase2 import (
    ceremony_store,
    execution_registry,
    operational_ceremony,
    operational_ceremony_runner,
)

if TYPE_CHECKING:
    from types import ModuleType

WP22H_PUBLIC_MODULES: tuple[ModuleType, ...] = (
    execution_registry,
    ceremony_store,
    operational_ceremony,
    operational_ceremony_runner,
)
PACKAGE_PRIVATE_CLI_EXPORTS = frozenset({"build_argument_parser", "main"})


def test_wp22h_public_module_exports_are_closed_over_the_phase2_package() -> None:
    """Every non-CLI WP22H export is the identical package-level object."""
    package_exports = set(phase2.__all__)
    wp22h_exports: set[str] = set()

    for module in WP22H_PUBLIC_MODULES:
        module_exports = set(module.__all__)
        assert len(module_exports) == len(module.__all__)
        assert module_exports <= set(vars(module))
        package_module_exports = module_exports - PACKAGE_PRIVATE_CLI_EXPORTS
        wp22h_exports.update(package_module_exports)

        for name in package_module_exports:
            assert name in package_exports
            assert getattr(phase2, name) is getattr(module, name)

    assert wp22h_exports <= package_exports
    assert PACKAGE_PRIVATE_CLI_EXPORTS.isdisjoint(package_exports)
    assert set(operational_ceremony_runner.__all__) >= PACKAGE_PRIVATE_CLI_EXPORTS
