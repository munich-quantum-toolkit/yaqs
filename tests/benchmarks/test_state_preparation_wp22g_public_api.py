# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Public package-surface regression tests for WP22G confirmation custody."""

from __future__ import annotations

from benchmarks.state_preparation import phase2
from benchmarks.state_preparation.phase2.confirmatory_study import validate_confirmatory_novelty
from benchmarks.state_preparation.phase2.confirmatory_study_store import (
    CONFIRMATORY_STUDY_DIRECTORY_NAME,
    confirmation_output_has_interrupted_attempt,
    confirmation_output_has_owned_state,
    publish_locked_confirmatory_study_snapshot,
    validate_initial_locked_confirmatory_study_snapshot,
    validate_locked_confirmatory_study_output,
)
from benchmarks.state_preparation.phase2.production_executors import (
    PersistedConfirmationResourceLimitError,
    validate_confirmation_resource_limits,
    validate_existing_confirmation_outcome,
)


def test_wp22g_public_package_exports_are_available() -> None:
    """The Phase II package exposes every public WP22G custody operation."""
    expected_symbols = {
        "PersistedConfirmationResourceLimitError": PersistedConfirmationResourceLimitError,
        "confirmation_output_has_interrupted_attempt": confirmation_output_has_interrupted_attempt,
        "confirmation_output_has_owned_state": confirmation_output_has_owned_state,
        "publish_locked_confirmatory_study_snapshot": publish_locked_confirmatory_study_snapshot,
        "validate_confirmation_resource_limits": validate_confirmation_resource_limits,
        "validate_confirmatory_novelty": validate_confirmatory_novelty,
        "validate_existing_confirmation_outcome": validate_existing_confirmation_outcome,
        "validate_initial_locked_confirmatory_study_snapshot": validate_initial_locked_confirmatory_study_snapshot,
        "validate_locked_confirmatory_study_output": validate_locked_confirmatory_study_output,
    }
    assert phase2.CONFIRMATORY_STUDY_DIRECTORY_NAME == CONFIRMATORY_STUDY_DIRECTORY_NAME
    assert "CONFIRMATORY_STUDY_DIRECTORY_NAME" in phase2.__all__
    for name, symbol in expected_symbols.items():
        assert getattr(phase2, name) is symbol
        assert name in phase2.__all__
