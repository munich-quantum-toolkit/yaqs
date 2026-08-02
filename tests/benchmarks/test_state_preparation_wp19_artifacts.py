# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Focused artifact-binding tests for the WP19 historical reproduction."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest

from benchmarks.state_preparation.phase2.artifacts import Phase2ArtifactStore
from benchmarks.state_preparation.phase2.canonical import thaw_json_mapping
from benchmarks.state_preparation.phase2.layerwise_bmpd import resolve_layerwise_bmpd_crn_legacy_v1_pipeline
from benchmarks.state_preparation.phase2.legacy_targets import load_legacy_target_collection
from benchmarks.state_preparation.phase2.noisy_krotov import NoisyKrotovObjectiveBinding
from benchmarks.state_preparation.phase2.pipeline import (
    PHASE1_FIXTURE_MANIFEST_CHECKSUM,
    fixture_target_spec_checksum,
)
from mqt.yaqs.core.data_structures.mps import MPS
from tests.benchmarks.test_state_preparation_phase2_pipeline import _pipeline, _template

if TYPE_CHECKING:
    from collections.abc import Mapping

    from benchmarks.state_preparation.phase2.pipeline import TrainingPipelineConfig


def _validate_binding(
    pipeline: TrainingPipelineConfig,
    binding: NoisyKrotovObjectiveBinding,
) -> None:
    """Invoke the publication guard without creating unrelated filesystem state."""
    store = object.__new__(Phase2ArtifactStore)
    store.pipeline = pipeline
    store._validate_wp17_objective_binding(  # noqa: SLF001
        binding,
        objective_checksum=binding.objective_checksum,
    )


def _phase1_pipeline() -> TrainingPipelineConfig:
    """Resolve an immutable Phase I fixture for namespace-isolation checks.

    Returns:
        A fully validated Phase I secondary-fixture pipeline.
    """
    template = replace(_template(), target_scope_id="phase1_fixture")
    target_id = "gaussian_mu0p5_sigma0p1"
    return template.resolve(
        target_namespace="phase1_fixture",
        target_manifest=None,
        target_instance_id=target_id,
        target_population_manifest_checksum=PHASE1_FIXTURE_MANIFEST_CHECKSUM,
        target_instance_spec_checksum=fixture_target_spec_checksum("phase1_fixture", target_id, 6),
        target_family_id="gaussian_amplitude",
        target_stratum_id="interior",
        qubit_count=6,
        optimization_block_id="wp19_phase1_cross_namespace",
        optimization_seed=1,
        data_role="secondary_benchmark",
    )


def test_wp19_artifact_binding_accepts_only_the_trusted_legacy_target() -> None:
    """The exact checked-in reconstructed target closes the legacy objective ledger."""
    pipeline = resolve_layerwise_bmpd_crn_legacy_v1_pipeline(100)
    target = load_legacy_target_collection().target(pipeline.target_instance_id)
    binding = NoisyKrotovObjectiveBinding.from_inputs(target, None, num_qubits=pipeline.qubit_count)

    _validate_binding(pipeline, binding)


@pytest.mark.parametrize("raw_target_kind", ["ndarray", "mps"])
def test_wp19_artifact_binding_rejects_untyped_target_operands(raw_target_kind: str) -> None:
    """Raw arrays and MPS values cannot impersonate a sealed reconstructed target."""
    pipeline = resolve_layerwise_bmpd_crn_legacy_v1_pipeline(100)
    trusted = load_legacy_target_collection().target(pipeline.target_instance_id)
    raw_target = trusted.state_vector_copy() if raw_target_kind == "ndarray" else MPS(8)
    binding = NoisyKrotovObjectiveBinding.from_inputs(raw_target, None, num_qubits=8)

    with pytest.raises(ValueError, match="sealed reconstructed target identity"):
        _validate_binding(pipeline, binding)


def test_wp19_artifact_binding_rejects_wrong_and_forged_legacy_identities() -> None:
    """Another sealed row and a caller-forged mapping both fail before publication."""
    collection = load_legacy_target_collection()
    pipeline = resolve_layerwise_bmpd_crn_legacy_v1_pipeline(100)
    configured = collection.target(pipeline.target_instance_id)
    wrong_target = collection.target("legacy_tfim_seed_200")
    wrong_binding = NoisyKrotovObjectiveBinding.from_inputs(wrong_target, None, num_qubits=8)
    with pytest.raises(ValueError, match="trusted sealed legacy target"):
        _validate_binding(pipeline, wrong_binding)

    genuine = NoisyKrotovObjectiveBinding.from_inputs(configured, None, num_qubits=8)
    forged_identity = thaw_json_mapping(cast("Mapping[str, object]", genuine.materialized_target_identity))
    forged_identity["family_id"] = "caller_forged_family"
    forged = NoisyKrotovObjectiveBinding(
        target_state_checksum=genuine.target_state_checksum,
        initial_state_policy=genuine.initial_state_policy,
        initial_state_checksum=genuine.initial_state_checksum,
        materialized_target_identity=forged_identity,
    )
    with pytest.raises(ValueError, match="trusted sealed legacy target"):
        _validate_binding(pipeline, forged)


@pytest.mark.parametrize("pipeline_kind", ["phase1", "phase2"])
def test_wp19_artifact_binding_cannot_cross_target_namespaces(pipeline_kind: str) -> None:
    """A legacy identity is unauthorized in both Phase I and Phase II target namespaces."""
    target = load_legacy_target_collection().target("legacy_tfim_seed_100")
    pipeline = _phase1_pipeline() if pipeline_kind == "phase1" else _pipeline()
    binding = NoisyKrotovObjectiveBinding.from_inputs(target, None, num_qubits=8)

    with pytest.raises(ValueError, match=r"authorized materialized Phase II target|configured pipeline target"):
        _validate_binding(pipeline, binding)


def test_wp19_artifact_binding_rejects_nonzero_initial_state() -> None:
    """Historical publication retains the computational-zero initial-state policy."""
    pipeline = resolve_layerwise_bmpd_crn_legacy_v1_pipeline(100)
    target = load_legacy_target_collection().target(pipeline.target_instance_id)
    binding = NoisyKrotovObjectiveBinding.from_inputs(target, MPS(8, state="x+"), num_qubits=8)

    with pytest.raises(ValueError, match="computational-zero initial-state policy"):
        _validate_binding(pipeline, binding)
