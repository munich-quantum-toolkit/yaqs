# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for WP22 candidate binding and held-out screening promotion."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest

from benchmarks.state_preparation.phase2.artifact_codecs import create_phase2_trajectory_sidecar
from benchmarks.state_preparation.phase2.protocol import (
    InitialPreregistration,
    ScreeningCell,
    load_initial_preregistration,
)
from benchmarks.state_preparation.phase2.result_custody import TrajectoryFidelityEvidence
from benchmarks.state_preparation.phase2.screening import (
    ADAPT_STYLE_PUBLICATION_METHOD_ID,
    IMPACT_PRUNING_PUBLICATION_METHOD_ID,
    OperatorGrowthScreeningTemplate,
    ProductionScreeningSourceRecord,
    ScreeningSourceRecord,
    VerifiedScreeningOutcome,
    WP18ScreeningSourceArtifact,
    WP22CandidateConfiguration,
    build_production_screening_evidence_from_records,
    build_screening_evidence,
    build_screening_manifest,
)
from benchmarks.state_preparation.phase2.screening_design import (
    WP22CandidateConfiguration as FrozenWP22CandidateConfiguration,
)
from benchmarks.state_preparation.phase2.targets import (
    TargetPopulationManifest,
    build_target_population_config,
    create_target_population_manifest,
    role_master_entropy_commitment,
)
from tests.benchmarks.wp22_screening_test_support import (
    candidate_configurations,
    complete_screening_sources,
    production_screening_records_fixture,
    production_screening_source,
    wp18_source,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

_SCREENING_MASTER = bytes(reversed(range(32)))


def _checksum(label: str) -> str:
    """Return one deterministic prefixed SHA-256 checksum."""
    return f"sha256:{hashlib.sha256(label.encode()).hexdigest()}"


@pytest.fixture(scope="module")
def preregistration() -> InitialPreregistration:
    """Load the trusted Phase II preregistration.

    Returns:
        The trusted protocol artifact.
    """
    return load_initial_preregistration()


@pytest.fixture(scope="module")
def screening_targets(preregistration: InitialPreregistration) -> TargetPopulationManifest:
    """Build the deterministic primary q6 screening population.

    Returns:
        The complete seed-bearing screening manifest.
    """
    config = build_target_population_config(
        preregistration,
        "screening_selection",
        role_master_entropy_commitment=role_master_entropy_commitment(_SCREENING_MASTER),
        population_scope="primary_q6",
    )
    return create_target_population_manifest(config, preregistration, _SCREENING_MASTER)


def _candidate_configurations(
    preregistration: InitialPreregistration,
) -> tuple[WP22CandidateConfiguration, ...]:
    """Build one strict synthetic implementation binding per frozen method.

    Returns:
        The nine typed candidate configurations.
    """
    return candidate_configurations(preregistration)


def test_candidate_alias_and_operator_template_are_explicit_and_sealed() -> None:
    """The publication pruning alias and operator-growth identity cannot drift."""
    assert WP22CandidateConfiguration is FrozenWP22CandidateConfiguration
    preregistration = load_initial_preregistration()
    candidates = _candidate_configurations(preregistration)
    impact = next(item for item in candidates if item.method_id == IMPACT_PRUNING_PUBLICATION_METHOD_ID)
    assert impact.implementation_method_id == "topdown_impact_iterative"
    assert WP22CandidateConfiguration.from_json(impact.to_json()) == impact

    with pytest.raises(ValueError, match="publication methods must retain"):
        replace(
            next(item for item in candidates if item.method_id == "spsa_layerwise"),
            implementation_method_id="parameter_shift_adam_layerwise",
        )
    with pytest.raises(ValueError, match="iterative-impact/noisy-CRN"):
        replace(impact, publication_mapping={})

    operator = OperatorGrowthScreeningTemplate(
        pool_policy_id="nearest_neighbor_pool",
        growth_policy_id="largest_projector_gradient",
        max_operators=12,
        reoptimization_steps=20,
        gradient_threshold=0.001,
        training_trajectory_count=256,
        native_two_qubit_cap_per_edge=12.0,
    )
    assert OperatorGrowthScreeningTemplate.from_json(operator.to_json()) == operator


def test_screening_manifest_has_exact_preregistered_cartesian_cardinality(
    preregistration: InitialPreregistration,
    screening_targets: TargetPopulationManifest,
) -> None:
    """Nine methods by 48 targets by three seeds produce 1,296 observations."""
    candidates = _candidate_configurations(preregistration)
    manifest = build_screening_manifest(
        preregistration,
        screening_targets,
        candidates,
        optimization_seeds=(101, 202, 303),
        screening_seed_root=7,
    )
    assert len(manifest.candidates) == 9
    assert len(manifest.cells) == 144
    assert len(manifest.candidates) * len(manifest.cells) == 1_296
    assert {cell.qubit_count for cell in manifest.cells} == {6}
    assert len({cell.screening_seed for cell in manifest.cells}) == 144


def _success_outcome(
    target_manifest: TargetPopulationManifest,
    candidate: WP22CandidateConfiguration,
    cell: ScreeningCell,
    fidelity: float,
) -> ScreeningSourceRecord:
    """Return one complete typed WP18 screening source."""
    return wp18_source(load_initial_preregistration(), target_manifest, candidate, cell, fidelity)


def test_promotion_uses_only_complete_outer_screening_evidence(
    preregistration: InitialPreregistration,
    screening_targets: TargetPopulationManifest,
) -> None:
    """The mechanical rule promotes the sole held-out improvement and rejects omissions."""
    candidates = _candidate_configurations(preregistration)
    manifest = build_screening_manifest(
        preregistration,
        screening_targets,
        candidates,
        optimization_seeds=(1, 2, 3),
        screening_seed_root=11,
    )
    outcomes = complete_screening_sources(
        preregistration,
        screening_targets,
        candidates,
        manifest.cells,
        promoted_method_id=ADAPT_STYLE_PUBLICATION_METHOD_ID,
    )
    evidence, decision = build_screening_evidence(preregistration, manifest, outcomes)
    assert len(evidence.observations) == 1_296
    assert decision.promoted_method_id == ADAPT_STYLE_PUBLICATION_METHOD_ID
    assert not decision.null_fallback

    with pytest.raises(TypeError, match="manifest-reopened ProductionScreeningSourceRecord"):
        build_production_screening_evidence_from_records(
            preregistration,
            manifest,
            cast("Sequence[ProductionScreeningSourceRecord]", outcomes),
            evidence_id="self_authored_sources_must_not_promote",
        )

    with pytest.raises(ValueError, match="Cartesian universe"):
        build_screening_evidence(preregistration, manifest, outcomes[:-1])
    with pytest.raises(TypeError, match="complete WP18ScreeningSourceArtifact"):
        cast("Callable[[object], ScreeningSourceRecord]", ScreeningSourceRecord)(object())
    projections = cast(
        "tuple[ScreeningSourceRecord, ...]",
        tuple(source.verified_outcome() for source in outcomes),
    )
    with pytest.raises(TypeError, match="authoritative"):
        build_screening_evidence(
            preregistration,
            manifest,
            projections,
        )


def test_production_screening_record_reopens_success_failure_and_rejects_wrong_custody(
    preregistration: InitialPreregistration,
    screening_targets: TargetPopulationManifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Representative E manifests enforce role, count, source, and first attempt."""
    monkeypatch.setattr(
        "benchmarks.state_preparation.phase2.production_executors.os.fsync",
        lambda _descriptor: None,
    )
    candidates = _candidate_configurations(preregistration)
    manifest = build_screening_manifest(
        preregistration,
        screening_targets,
        candidates,
        optimization_seeds=(1, 2, 3),
        screening_seed_root=29,
    )
    success = production_screening_source(
        manifest.candidates[0],
        manifest.cells[0],
        tmp_path / "success",
        fidelity=0.81,
    )
    failure = production_screening_source(
        manifest.candidates[0],
        manifest.cells[1],
        tmp_path / "failure",
        fidelity=None,
    )
    assert success.result_custody.mean_fidelity == pytest.approx(0.81)
    assert success.promotion_observation().status == "success"
    assert failure.result_custody.trajectory_fidelities is None
    assert failure.promotion_observation().status == "failure"

    with pytest.raises(ValueError, match="role"):
        production_screening_source(
            manifest.candidates[0],
            manifest.cells[2],
            tmp_path / "wrong-role",
            fidelity=0.8,
            raw_data_role="development",
        )
    with pytest.raises(ValueError, match="fixed trajectory count"):
        production_screening_source(
            manifest.candidates[0],
            manifest.cells[3],
            tmp_path / "wrong-count",
            fidelity=0.8,
            raw_trajectory_count=3,
        )
    with pytest.raises(ValueError, match="execution closure"):
        production_screening_source(
            manifest.candidates[0],
            manifest.cells[4],
            tmp_path / "wrong-source",
            fidelity=0.8,
            source_fingerprint_checksum=_checksum("foreign source fingerprint"),
        )
    with pytest.raises(ValueError, match=r"attempt|filename"):
        production_screening_source(
            manifest.candidates[0],
            manifest.cells[5],
            tmp_path / "wrong-attempt",
            fidelity=0.8,
            outcome_attempt=2,
        )


def test_complete_production_screening_universe_promotes_mechanically(
    preregistration: InitialPreregistration,
    screening_targets: TargetPopulationManifest,
) -> None:
    """All 1,296 typed promotion rows retain distinct production identities."""
    candidates = _candidate_configurations(preregistration)
    manifest = build_screening_manifest(
        preregistration,
        screening_targets,
        candidates,
        optimization_seeds=(1, 2, 3),
        screening_seed_root=31,
    )
    records = production_screening_records_fixture(
        manifest,
        target_manifest=screening_targets,
        fixed_trajectory_count=2,
        execution_source_manifest_checksum=_checksum("complete screen execution source"),
        promoted_method_id=ADAPT_STYLE_PUBLICATION_METHOD_ID,
    )
    evidence, decision = build_production_screening_evidence_from_records(
        preregistration,
        manifest,
        records,
        evidence_id="complete_manifest_reopened_production_screen",
    )
    assert len(records) == len(evidence.observations) == 1_296
    assert len({record.result_custody.reference.content_checksum for record in records}) == 1_296
    assert decision.promoted_method_id == ADAPT_STYLE_PUBLICATION_METHOD_ID
    assert not decision.null_fallback
    with pytest.raises(ValueError, match="1,296-cell Cartesian universe"):
        build_production_screening_evidence_from_records(
            preregistration,
            manifest,
            records[:-1],
            evidence_id="incomplete_production_screen",
        )


def test_screening_outcome_roundtrip_recomputes_work_and_resources(
    screening_targets: TargetPopulationManifest,
) -> None:
    """Source roundtrips retain raw evidence and derive the projection."""
    candidate = _candidate_configurations(load_initial_preregistration())[0]
    target = screening_targets.instances[0]
    cell = ScreeningCell(
        cell_id="screening_cell_test",
        family_id=target.family_id,
        stratum_id=target.stratum_id,
        qubit_count=target.qubit_count,
        target_instance_id=target.target_instance_id,
        optimization_seed=7,
        screening_seed=123,
    )
    source = _success_outcome(screening_targets, candidate, cell, 0.75)
    assert ScreeningSourceRecord.from_json(source.to_json()).to_dict() == source.to_dict()
    outcome = source.verified_outcome()
    assert VerifiedScreeningOutcome.from_json(outcome.to_json()) == outcome
    with pytest.raises(ValueError, match="mechanically derived"):
        replace(outcome, normalized_work=1.0)
    with pytest.raises(ValueError, match="compiler-derived"):
        replace(outcome, resource_value=1.0)


def test_pipeline_screening_source_derives_mean_from_raw_trajectory_evidence(
    screening_targets: TargetPopulationManifest,
) -> None:
    """WP18 custody dereferences exact sidecar bytes and rejects same-mean substitutes."""
    preregistration = load_initial_preregistration()
    candidate = _candidate_configurations(preregistration)[0]
    target = screening_targets.instances[0]
    cell = ScreeningCell(
        cell_id="screening_cell_authenticated_sidecar",
        family_id=target.family_id,
        stratum_id=target.stratum_id,
        qubit_count=target.qubit_count,
        target_instance_id=target.target_instance_id,
        optimization_seed=101,
        screening_seed=909,
    )
    source = wp18_source(preregistration, screening_targets, candidate, cell, 0.80)
    artifact = cast("WP18ScreeningSourceArtifact", source.source_artifact)

    assert source.verified_outcome().noisy_fidelity == pytest.approx(0.80)
    assert WP18ScreeningSourceArtifact.from_json(artifact.to_json()).to_dict() == artifact.to_dict()
    assert source.trajectory_evidence is not None
    partitions = tuple(
        {
            "ensemble_id": item.ensemble_id,
            "content_checksum": item.content_checksum,
            "trajectory_count": item.trajectory_count,
        }
        for item in artifact.evaluation_maps
    )
    substituted = create_phase2_trajectory_sidecar(
        evaluation_row_id=artifact.record.evaluation_row_id,
        pipeline_training_id=artifact.record.config.pipeline_training_id,
        map_role="screening_selection",
        map_partitions=partitions,
        fidelities=(0.70, 0.90),
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        replace(artifact, trajectory_sidecar_payload=substituted)

    fabricated = TrajectoryFidelityEvidence(
        evaluation_context_checksum=source.trajectory_evidence.evaluation_context_checksum,
        data_role="screening_selection",
        evaluation_seed=cell.screening_seed,
        trajectory_fidelities=(0.70, 0.90),
    )
    with pytest.raises(TypeError, match="unexpected keyword"):
        cast("Callable[..., ScreeningSourceRecord]", ScreeningSourceRecord.from_pipeline_record)(
            trajectory_evidence=fabricated,
            candidate=candidate,
            cell=cell,
            template=artifact.template,
            pipeline_result=artifact.pipeline_result,
            record=artifact.record,
            work_ledger=artifact.work_ledger,
            circuit_resources=artifact.circuit_resources,
            evaluation_evidence=artifact.evaluation_evidence,
            materialization=artifact.materialization,
            preregistration=preregistration,
        )
