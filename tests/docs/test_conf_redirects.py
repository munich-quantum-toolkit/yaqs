# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Regression tests for Sphinx redirect configuration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import types


def _load_docs_conf() -> types.ModuleType:
    """Load ``docs/conf.py`` as a module.

    Returns:
        Loaded Sphinx configuration module.

    Raises:
        RuntimeError: If the configuration file cannot be loaded.
    """
    conf_path = Path(__file__).resolve().parents[2] / "docs" / "conf.py"
    spec = importlib.util.spec_from_file_location("docs_conf", conf_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load docs configuration from {conf_path}."
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_example_redirects() -> None:
    """Legacy example URLs still map to the strong-simulation targets."""
    conf = _load_docs_conf()
    redirects = conf.redirects

    assert redirects["examples/circuit_simulation"] == "examples/strong_simulation.html"
    assert redirects["examples/strong_circuit_simulation"] == "examples/strong_simulation.html"
    assert (
        redirects["examples/sample_observable_digital_tjm"] == "examples/strong_simulation.html#mid-circuit-observables"
    )
