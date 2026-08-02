# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Golden compatibility checks protecting Phase I from Phase II changes."""

from __future__ import annotations

import hashlib
import json

from benchmarks.state_preparation import (
    CONFIG_SCHEMA_VERSION,
    CSV_COLUMNS,
    KROTOV_CHECKPOINT_FORMAT,
    KROTOV_METHOD_ID,
    KROTOV_METHOD_VERSION,
    KROTOV_PARAMETER_LAYOUT_FORMAT,
    NOISE_DEFINITION_VERSION,
    REPORT_MANIFEST_FORMAT,
    RESULT_SCHEMA_VERSION,
    TARGET_FIXTURE_FORMAT,
    TRAINING_IDENTITY_VERSION,
    TRAJECTORY_SIDECAR_FORMAT,
    KrotovStatePreparationMethod,
    load_target_collection,
    state_preparation_training_id,
)
from benchmarks.state_preparation import phase2 as phase2_api
from benchmarks.state_preparation.runner import (
    RUNNER_CONFIGURATION_FORMAT,
    build_benchmark_matrix,
    parse_arguments,
    resolve_options,
)
from benchmarks.state_preparation.schema import RUN_IDENTITY_VERSION


def test_phase2_public_api_exports_wp16_records() -> None:
    """The Phase II package exposes the versioned WP16 construction API."""
    expected = {
        "ExternalCheckpointRef",
        "Phase2TargetRef",
        "PipelineBenchmarkFailure",
        "PipelineBenchmarkResult",
        "PipelineEvaluationConfig",
        "TargetPopulationConfig",
        "TargetPopulationManifest",
        "TrainingPipelineConfig",
        "TrainingPipelineTemplate",
        "TrainingStageConfig",
        "create_target_population_manifest",
        "validate_screening_resolution",
    }
    assert expected <= set(phase2_api.__all__)
    assert len(phase2_api.__all__) == len(set(phase2_api.__all__))
    assert all(getattr(phase2_api, name) is not None for name in expected)


def test_phase1_schema_and_method_identities_remain_literal_v1_values() -> None:
    """Phase II additions must not rename or rev existing Phase I formats."""
    assert {
        "benchmark_config": CONFIG_SCHEMA_VERSION,
        "benchmark_result": RESULT_SCHEMA_VERSION,
        "run_identity": RUN_IDENTITY_VERSION,
        "training_identity": TRAINING_IDENTITY_VERSION,
        "noise_definition": NOISE_DEFINITION_VERSION,
        "target_fixture": TARGET_FIXTURE_FORMAT,
        "runner_configuration": RUNNER_CONFIGURATION_FORMAT,
        "checkpoint": KROTOV_CHECKPOINT_FORMAT,
        "parameter_layout": KROTOV_PARAMETER_LAYOUT_FORMAT,
        "report_manifest": REPORT_MANIFEST_FORMAT,
        "trajectory_sidecar": TRAJECTORY_SIDECAR_FORMAT,
    } == {
        "benchmark_config": "yaqs.state_preparation.config.v1",
        "benchmark_result": "yaqs.state_preparation.result.v1",
        "run_identity": "yaqs.state_preparation.run_identity.v1",
        "training_identity": "yaqs.state_preparation.training_identity.v1",
        "noise_definition": "yaqs.state_preparation.noise.v1",
        "target_fixture": "yaqs.state_preparation_targets.v1",
        "runner_configuration": "yaqs.state_preparation.runner_config.v1",
        "checkpoint": "yaqs.state_preparation.krotov_checkpoint.v1",
        "parameter_layout": "yaqs.state_preparation.krotov_parameter_layout.v1",
        "report_manifest": "yaqs.state_preparation.run_manifest.v1",
        "trajectory_sidecar": "yaqs.state_preparation.trajectory_sidecar.v1",
    }
    assert (KROTOV_METHOD_ID, KROTOV_METHOD_VERSION) == ("krotov", "1")


def test_phase1_target_fixture_checksum_and_cardinality_remain_frozen() -> None:
    """The checked-in target population must retain its exact Phase I bytes."""
    targets = load_target_collection()

    assert targets.fixture_format == "yaqs.state_preparation_targets.v1"
    assert targets.fixture_checksum == "sha256:49948fe4e63f652169c603e5e03f32f8a66ad70daa25091ee7cdf83644287d11"
    assert len(targets.records) == 18


def test_phase1_preset_result_and_training_cardinalities_remain_frozen() -> None:
    """Every Phase I preset must retain its train-once and evaluation fan-out."""
    targets = load_target_collection()
    method = KrotovStatePreparationMethod()
    expected_cardinalities = {
        "smoke": (12, 1),
        "minimum": (108, 18),
        "full": (216, 18),
    }

    for preset, (expected_results, expected_trainings) in expected_cardinalities.items():
        options = resolve_options(parse_arguments(["--preset", preset, "--dry-run"]))
        matrix = build_benchmark_matrix(options, targets)
        training_ids = {state_preparation_training_id(method, config) for config in matrix}

        assert len(matrix) == expected_results
        assert len({config.run_id for config in matrix}) == expected_results
        assert len(training_ids) == expected_trainings


def test_phase1_representative_run_and_training_ids_remain_frozen() -> None:
    """A canonical training and two evaluation fan-outs retain their hashes."""
    targets = load_target_collection()
    options = resolve_options(parse_arguments(["--preset", "full", "--dry-run"]))
    matrix = build_benchmark_matrix(options, targets)
    noiseless = matrix[0]
    last_noisy_fanout = matrix[11]
    method = KrotovStatePreparationMethod()

    assert (noiseless.target.num_qubits, noiseless.target.target_id, noiseless.test_noise.noise_id) == (
        6,
        "gaussian_mu0p5_sigma0p1",
        "noiseless",
    )
    assert last_noisy_fanout.target == noiseless.target
    assert last_noisy_fanout.test_noise.noise_id == "depolarizing_1s2s_all"
    assert noiseless.run_id == "spr-v1-e47827d2fd8391d6df6899683a7264420bb42300ac2481ce88ddfbf86a756361"
    assert last_noisy_fanout.run_id == "spr-v1-7b7e993b5e1a8242f1112d89875cd92ff48688da59c5836b1022639f4006631d"
    assert (
        state_preparation_training_id(
            method,
            noiseless,
        )
        == "spt-v1-de1ffaca3a067ae67dcdbb959855d8a21b841388bda150ad05887fbbc2e59a5d"
    )
    assert state_preparation_training_id(method, last_noisy_fanout) == state_preparation_training_id(
        method,
        noiseless,
    )
    assert len(noiseless.to_json()) == 1336
    assert hashlib.sha256(noiseless.to_json().encode()).hexdigest() == (
        "ee253930bc4299e989b57bee3e59b7eaec6c08f90627251c1b08562c79b9366a"
    )
    assert len(last_noisy_fanout.to_json()) == 1359
    assert hashlib.sha256(last_noisy_fanout.to_json().encode()).hexdigest() == (
        "fecea1ee279d273f6f3f81e85bfee5a1148bd19204d1a78c7381221feaecaa7d"
    )


def test_phase1_csv_schema_hash_and_cardinality_remain_frozen() -> None:
    """The Phase I CSV column sequence must remain byte-for-byte stable."""
    canonical_columns = json.dumps(
        list(CSV_COLUMNS),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()

    assert len(CSV_COLUMNS) == 47
    assert len(set(CSV_COLUMNS)) == 47
    assert f"sha256:{hashlib.sha256(canonical_columns).hexdigest()}" == (
        "sha256:dbf3bb1bce3eaf2f82adbca362e968b27380f403e97946433f186719c8e1a588"
    )
