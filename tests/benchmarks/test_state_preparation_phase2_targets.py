# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for versioned, externally custodied Phase II target populations."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from benchmarks.state_preparation.constants import TARGET_IDS as PHASE_I_TARGET_IDS
from benchmarks.state_preparation.phase2 import protocol as protocol_module
from benchmarks.state_preparation.phase2 import targets as target_module
from benchmarks.state_preparation.phase2.canonical import canonical_checksum, canonical_json, seal_mapping
from benchmarks.state_preparation.phase2.protocol import (
    PRIMARY_FAMILY_STRATA,
    PRIMARY_TARGET_FAMILIES,
    ConfirmationAuthorization,
    InitialPreregistration,
    ScreeningCandidateRef,
    ScreeningCell,
    ScreeningManifest,
    load_initial_preregistration,
)
from benchmarks.state_preparation.phase2.targets import (
    PHASE_II_TARGET_ID_PREFIX,
    TARGET_GENERATOR_SCHEMA_VERSION,
    TRUSTED_TARGET_ALLOCATION_POLICY_CHECKSUM,
    TRUSTED_TARGET_NUMERIC_POLICY_CHECKSUM,
    TRUSTED_TARGET_POPULATION_POLICY_CHECKSUM,
    TRUSTED_TARGET_RNG_POLICY_CHECKSUM,
    MaterializedTarget,
    TargetAllocation,
    TargetInstanceSpec,
    TargetMaterializationAuthorization,
    TargetPopulationCommitment,
    TargetPopulationConfig,
    TargetPopulationManifest,
    TargetPopulationMaterialization,
    authorize_target_materialization,
    build_target_population_config,
    create_target_population_manifest,
    materialize_target_population,
    role_master_entropy_commitment,
    verify_screening_target_population,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

ROLE_MASTER = bytes(range(32))
SCREENING_MASTER = bytes(reversed(range(32)))
CONFIRMATORY_MASTER = bytes((index * 7) % 256 for index in range(32))
GENERATION_RUNTIME_AVAILABLE = np.__version__ == "2.4.6"


@pytest.fixture(scope="module")
def preregistration() -> InitialPreregistration:
    """Return the trusted checked-in WP15 preregistration."""
    return load_initial_preregistration()


@pytest.fixture(scope="module")
def development_config(preregistration: InitialPreregistration) -> TargetPopulationConfig:
    """Return the exact q6 development-population config."""
    return build_target_population_config(
        preregistration,
        "development",
        role_master_entropy_commitment=role_master_entropy_commitment(ROLE_MASTER),
    )


@pytest.fixture(scope="module")
def development_manifest(
    preregistration: InitialPreregistration,
    development_config: TargetPopulationConfig,
) -> TargetPopulationManifest:
    """Return a deterministic seed-bearing development manifest."""
    if not GENERATION_RUNTIME_AVAILABLE:
        pytest.skip("Target-generation goldens require preregistered NumPy 2.4.6.")
    return create_target_population_manifest(development_config, preregistration, ROLE_MASTER)


@pytest.fixture(scope="module")
def development_materialization(
    preregistration: InitialPreregistration,
    development_config: TargetPopulationConfig,
    development_manifest: TargetPopulationManifest,
) -> TargetPopulationMaterialization:
    """Return the authorized vectors for the development manifest."""
    authorization = authorize_target_materialization(
        preregistration,
        development_config,
        development_manifest,
        ROLE_MASTER,
    )
    return materialize_target_population(
        development_config,
        preregistration,
        development_manifest,
        ROLE_MASTER,
        authorization,
    )


@pytest.fixture(scope="module")
def primary_screening_config(preregistration: InitialPreregistration) -> TargetPopulationConfig:
    """Return the isolated primary-q6 screening config."""
    return build_target_population_config(
        preregistration,
        "screening_selection",
        role_master_entropy_commitment=role_master_entropy_commitment(SCREENING_MASTER),
        population_scope="primary_q6",
    )


@pytest.fixture(scope="module")
def primary_screening_targets(
    preregistration: InitialPreregistration,
    primary_screening_config: TargetPopulationConfig,
) -> TargetPopulationManifest:
    """Return the isolated primary-q6 screening target manifest."""
    if not GENERATION_RUNTIME_AVAILABLE:
        pytest.skip("Target-generation goldens require preregistered NumPy 2.4.6.")
    return create_target_population_manifest(primary_screening_config, preregistration, SCREENING_MASTER)


@pytest.fixture(scope="module")
def confirmatory_config(preregistration: InitialPreregistration) -> TargetPopulationConfig:
    """Return the minimum exact q6 confirmatory-population config."""
    return build_target_population_config(
        preregistration,
        "confirmatory",
        role_master_entropy_commitment=role_master_entropy_commitment(CONFIRMATORY_MASTER),
        confirmatory_target_count_by_family=dict.fromkeys(PRIMARY_TARGET_FAMILIES, 24),
    )


@pytest.fixture(scope="module")
def confirmatory_manifest(
    preregistration: InitialPreregistration,
    confirmatory_config: TargetPopulationConfig,
) -> TargetPopulationManifest:
    """Return the minimum custodied confirmatory target manifest."""
    if not GENERATION_RUNTIME_AVAILABLE:
        pytest.skip("Target-generation goldens require preregistered NumPy 2.4.6.")
    return create_target_population_manifest(confirmatory_config, preregistration, CONFIRMATORY_MASTER)


def _checksum(label: str) -> str:
    """Return a valid deterministic test checksum."""
    return f"sha256:{hashlib.sha256(label.encode('utf-8')).hexdigest()}"


def _screening_manifest_for_targets(
    preregistration: InitialPreregistration,
    targets: TargetPopulationManifest,
    *,
    omitted_target_id: str | None = None,
    mismatched_first_identity: bool = False,
    extra_target_id: str | None = None,
    target_manifest_checksum: str | None = None,
) -> ScreeningManifest:
    """Build WP15 screening cells resolving the supplied WP16 targets.

    Returns:
        A complete three-optimization-seed screening manifest.
    """
    cells: list[ScreeningCell] = []
    screening_seed = 800_000
    mismatched_target_id = targets.instances[0].target_instance_id if mismatched_first_identity else None
    for spec in targets.instances:
        if spec.target_instance_id == omitted_target_id:
            continue
        for optimization_index, optimization_seed in enumerate((101, 102, 103), start=1):
            screening_seed += 1
            family_id = spec.family_id
            stratum_id = spec.stratum_id
            if spec.target_instance_id == mismatched_target_id:
                family_id = "haar_random"
                stratum_id = "dense_complex"
            cells.append(
                ScreeningCell(
                    cell_id=f"{spec.target_instance_id}_optimization_{optimization_index}",
                    family_id=family_id,
                    stratum_id=stratum_id,
                    qubit_count=spec.qubit_count,
                    target_instance_id=spec.target_instance_id,
                    optimization_seed=optimization_seed,
                    screening_seed=screening_seed,
                )
            )
    if extra_target_id is not None:
        for optimization_index, optimization_seed in enumerate((101, 102, 103), start=1):
            screening_seed += 1
            cells.append(
                ScreeningCell(
                    cell_id=f"{extra_target_id}_optimization_{optimization_index}",
                    family_id="gaussian_amplitude",
                    stratum_id="interior",
                    qubit_count=6,
                    target_instance_id=extra_target_id,
                    optimization_seed=optimization_seed,
                    screening_seed=screening_seed,
                )
            )
    candidate = ScreeningCandidateRef(
        configuration_schema_version="phase2_test_configuration_v1",
        configuration_checksum=_checksum("layerwise v2"),
        method_id="layerwise_bmpd_crn_v2",
        noisy_training=True,
        resource_stratum_id="depth4_equivalent",
        matching_projection_checksum=_checksum("matching projection"),
    )
    return ScreeningManifest(
        manifest_id="phase2_screening_manifest_target_link_test",
        preregistration_checksum=preregistration.content_checksum,
        screening_target_manifest_checksum=target_manifest_checksum or targets.content_checksum,
        evaluation_policy_checksum=_checksum("evaluation policy"),
        resource_policy_checksum=_checksum("resource policy"),
        baseline_configuration_checksum=candidate.configuration_checksum,
        candidates=(candidate,),
        cells=tuple(cells),
    )


def _forged_confirmation_authorization(
    preregistration_checksum: str,
    target_manifest_checksum: str,
) -> ConfirmationAuthorization:
    """Create an exact typed token for linkage tests.

    Returns:
        A confirmation token carrying the requested manifest checksum.
    """
    return ConfirmationAuthorization(
        preregistration_checksum=preregistration_checksum,
        final_seal_checksum=_checksum("final seal"),
        target_manifest_checksum=target_manifest_checksum,
        execution_source_checksum=_checksum("execution source"),
        _marker=protocol_module._AUTHORIZATION_SENTINEL,  # noqa: SLF001
    )


def test_preregistration_policy_links_are_exact(preregistration: InitialPreregistration) -> None:
    """WP16 binds the exact sealed target, RNG, numeric, and allocation policies."""
    policy = preregistration.target_population_policy
    assert (
        preregistration.target_population_configuration_checksum
        == TRUSTED_TARGET_POPULATION_POLICY_CHECKSUM
        == "sha256:67720a4ab54ed515e8affe543ed35b199cea98fe977fe3da60541640477c5d7e"
    )
    assert canonical_checksum(policy["rng_policy"]) == TRUSTED_TARGET_RNG_POLICY_CHECKSUM
    assert (
        canonical_checksum(policy["numeric_policy"])
        == TRUSTED_TARGET_NUMERIC_POLICY_CHECKSUM
        == "sha256:34ef8cc1de502c1cb9a0699a1bad5cba1ec28e243124b1ea2a336a00c21c670a"
    )
    assert canonical_checksum(policy["role_allocation_policy"]) == TRUSTED_TARGET_ALLOCATION_POLICY_CHECKSUM
    assert policy["generator_schema_version"] == TARGET_GENERATOR_SCHEMA_VERSION
    numeric_policy = cast("Mapping[str, object]", policy["numeric_policy"])
    assert numeric_policy["manifest_spectrum_solver"] == "numpy_linalg_eigvalsh_dense_hermitian"
    assert numeric_policy["authorized_state_solver"] == "numpy_linalg_eigh_dense_hermitian"
    assert numeric_policy["spectrum_agreement_rtol"] == pytest.approx(1e-13, rel=0.0, abs=0.0)
    assert numeric_policy["spectrum_agreement_atol"] == pytest.approx(1e-13, rel=0.0, abs=0.0)


def test_role_master_commitment_is_exact_and_non_revealing() -> None:
    """The config binds a raw 32-byte key through SHA-256 without exposing it."""
    assert (
        role_master_entropy_commitment(ROLE_MASTER)
        == "sha256:630dcd2966c4336691125448bbb25b4ff412a49c732db2c8abc1b8581bd710dd"
    )
    assert role_master_entropy_commitment(ROLE_MASTER.hex()) == role_master_entropy_commitment(ROLE_MASTER)
    with pytest.raises(ValueError, match="exactly 32 bytes"):
        role_master_entropy_commitment(b"short")
    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        role_master_entropy_commitment("AA" * 32)
    with pytest.raises(TypeError, match="32 bytes"):
        role_master_entropy_commitment(bytearray(32))  # ty: ignore[invalid-argument-type]


def test_config_allocations_are_exact_and_q6_q12_are_separate(
    preregistration: InitialPreregistration,
    development_config: TargetPopulationConfig,
    primary_screening_config: TargetPopulationConfig,
) -> None:
    """Development, primary screening, and secondary screening cannot mix."""
    assert development_config.population_scope == "primary_q6"
    assert sum(allocation.instance_count for allocation in development_config.allocations) == 48
    assert {allocation.qubit_count for allocation in development_config.allocations} == {6}
    assert sum(allocation.instance_count for allocation in primary_screening_config.allocations) == 48
    secondary = build_target_population_config(
        preregistration,
        "screening_selection",
        role_master_entropy_commitment=_checksum("secondary screening key"),
        population_scope="secondary_q12",
    )
    assert sum(allocation.instance_count for allocation in secondary.allocations) == 24
    assert {allocation.qubit_count for allocation in secondary.allocations} == {12}
    assert secondary.content_checksum != primary_screening_config.content_checksum
    assert secondary.population_id != primary_screening_config.population_id


def test_checkpoint_validation_and_invalid_scope_cannot_create_populations(
    preregistration: InitialPreregistration,
) -> None:
    """Only the three preregistered target-instance roles are accepted."""
    commitment = role_master_entropy_commitment(ROLE_MASTER)
    with pytest.raises(ValueError, match="data_role"):
        build_target_population_config(
            preregistration,
            "checkpoint_validation",
            role_master_entropy_commitment=commitment,
        )
    with pytest.raises(ValueError, match="primary_q6"):
        build_target_population_config(
            preregistration,
            "development",
            role_master_entropy_commitment=commitment,
            population_scope="secondary_q12",
        )
    with pytest.raises(ValueError, match="primary_q6"):
        build_target_population_config(
            preregistration,
            "confirmatory",
            role_master_entropy_commitment=commitment,
            population_scope="secondary_q12",
            confirmatory_target_count_by_family=dict.fromkeys(PRIMARY_TARGET_FAMILIES, 24),
        )


@pytest.mark.parametrize("count", [18, 25, 102])
def test_confirmatory_counts_enforce_frozen_bounds(
    preregistration: InitialPreregistration,
    count: int,
) -> None:
    """Confirmatory family totals must be balanced 24..96 in increments of six."""
    with pytest.raises(ValueError, match="24 through 96"):
        build_target_population_config(
            preregistration,
            "confirmatory",
            role_master_entropy_commitment=_checksum(f"confirmatory {count}"),
            confirmatory_target_count_by_family=dict.fromkeys(PRIMARY_TARGET_FAMILIES, count),
        )


def test_confirmatory_counts_are_equal_across_families(preregistration: InitialPreregistration) -> None:
    """Unequal family totals cannot enter the primary confirmatory population."""
    counts = dict.fromkeys(PRIMARY_TARGET_FAMILIES, 24)
    counts["haar_random"] = 30
    with pytest.raises(ValueError, match="equal across families"):
        build_target_population_config(
            preregistration,
            "confirmatory",
            role_master_entropy_commitment=_checksum("unequal confirmatory"),
            confirmatory_target_count_by_family=counts,
        )


def test_config_and_role_identity_change_with_key_and_role(
    preregistration: InitialPreregistration,
    development_config: TargetPopulationConfig,
) -> None:
    """Role and role-key commitments are inside the population identity."""
    changed_key = build_target_population_config(
        preregistration,
        "development",
        role_master_entropy_commitment=role_master_entropy_commitment(SCREENING_MASTER),
    )
    screening = build_target_population_config(
        preregistration,
        "screening_selection",
        role_master_entropy_commitment=role_master_entropy_commitment(ROLE_MASTER),
    )
    assert len({development_config.content_checksum, changed_key.content_checksum, screening.content_checksum}) == 3
    assert len({development_config.population_id, changed_key.population_id, screening.population_id}) == 3


def test_hmac_seed_and_big_endian_component_rng_golden(
    development_config: TargetPopulationConfig,
    development_manifest: TargetPopulationManifest,
) -> None:
    """Canonical UTF-8 HMAC derivation and big-endian SeedSequence input are stable."""
    allocation = development_config.allocations[0]
    seed_identity = {
        "random_stream_domain": "target_generation",
        "substream": "instance_seed",
        "generator_schema_version": TARGET_GENERATOR_SCHEMA_VERSION,
        "population_config_checksum": development_config.content_checksum,
        "data_role": development_config.data_role,
        "family_id": allocation.family_id,
        "stratum_id": allocation.stratum_id,
        "qubit_count": allocation.qubit_count,
        "instance_index": 0,
    }
    digest = hmac.new(
        ROLE_MASTER,
        canonical_json(seed_identity).encode("utf-8"),
        hashlib.sha256,
    ).digest()[:16]
    expected_seed = digest.hex()
    assert expected_seed == "e77597e44019f7dfc7d3fe7f2a352261"
    matching = next(spec for spec in development_manifest.instances if spec.instance_seed == expected_seed)
    component_identity = {
        "random_stream_domain": "target_generation",
        "generator_schema_version": TARGET_GENERATOR_SCHEMA_VERSION,
        "target_instance_id": matching.target_instance_id,
        "component_substream": "parameters_mean",
    }
    component_digest = hmac.new(
        ROLE_MASTER,
        canonical_json(component_identity).encode("utf-8"),
        hashlib.sha256,
    ).digest()[:16]
    entropy = int.from_bytes(component_digest, byteorder="big", signed=False)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
    assert float(cast("float", matching.parameters["mean"])).hex() == float(rng.uniform(0.3, 0.7)).hex()


def test_manifest_is_deterministic_seed_bearing_and_vector_free(
    preregistration: InitialPreregistration,
    development_config: TargetPopulationConfig,
    development_manifest: TargetPopulationManifest,
) -> None:
    """The custodied document commits seeds and parameters, never amplitudes or the key."""
    repeated = create_target_population_manifest(development_config, preregistration, ROLE_MASTER)
    assert repeated.to_json() == development_manifest.to_json()
    assert repeated.content_checksum == development_manifest.content_checksum
    payload = development_manifest.to_json()
    assert "instance_seed" in payload
    assert '"parameters"' in payload
    assert "state_vector" not in payload
    assert "amplitudes" not in payload
    assert ROLE_MASTER.hex() not in payload
    assert development_manifest.role_master_entropy_commitment == role_master_entropy_commitment(ROLE_MASTER)


def test_manifest_generation_never_constructs_or_normalizes_vectors(
    monkeypatch: pytest.MonkeyPatch,
    preregistration: InitialPreregistration,
    development_config: TargetPopulationConfig,
    development_manifest: TargetPopulationManifest,
) -> None:
    """The pre-authorization path uses TFIM spectra but no target eigenvector operation."""

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Manifest generation attempted to construct or normalize a target vector.")

    monkeypatch.setattr(target_module.np.linalg, "eigh", forbidden)
    monkeypatch.setattr(target_module, "_normalize_and_fix_phase", forbidden)
    repeated = create_target_population_manifest(development_config, preregistration, ROLE_MASTER)
    assert repeated.to_json() == development_manifest.to_json()


def test_complete_role_manifests_match_pinned_goldens(
    development_config: TargetPopulationConfig,
    development_manifest: TargetPopulationManifest,
    primary_screening_config: TargetPopulationConfig,
    primary_screening_targets: TargetPopulationManifest,
    confirmatory_config: TargetPopulationConfig,
    confirmatory_manifest: TargetPopulationManifest,
) -> None:
    """Every complete role manifest has one pinned full-payload identity."""
    artifacts = {
        "development": (development_config, development_manifest),
        "screening_selection": (primary_screening_config, primary_screening_targets),
        "confirmatory": (confirmatory_config, confirmatory_manifest),
    }
    expected = {
        "development": {
            "config_checksum": "sha256:b5a9d5953db1ef07625b9af4d483dde8f1f087b8196c66b158a7d82253e07baf",
            "population_id": (
                "phase2_target_population_595a9a06916111187b76a6e21e06b5a4a9aa9a9f5ac5a6cc9e494fe10f2f074c"
            ),
            "manifest_id": "phase2_target_population_a81d29e1f4d5a561670d7164c73ede0320f7c52ff3244c1c1130357024003131",
            "manifest_checksum": "sha256:b02d04e587c9cd48902dc406bd004556d2818f60cc4c58d63aae43629fe80ec4",
            "json_length": 35112,
            "json_checksum": "sha256:712873e4c09bb4c83984c589d001f3017da3f72c0487efa97f9e05bf3409b343",
        },
        "screening_selection": {
            "config_checksum": "sha256:cd4fd3fae1a167ba3b6906e3af1131d17223f6301eb0fcd4c8f00d46ede0704c",
            "population_id": (
                "phase2_target_population_31ab287f181cbe19ee079aa476242d3fd058571a5269b7a05ae0d167fef31cb5"
            ),
            "manifest_id": "phase2_target_population_2b94d4c67c60612bd2b7d78d1d2ed67e96169751ef7cea2be2168ac70bd86399",
            "manifest_checksum": "sha256:c0f325ab6ca45fdd11e6001af70ce726e389cbd4aa570ee7d30a7a4e171544c5",
            "json_length": 35526,
            "json_checksum": "sha256:f05c318f67b2f24f1afb5dd9440f2eb51f71be3d94f2d1da0a094c01155e4379",
        },
        "confirmatory": {
            "config_checksum": "sha256:1b4cbac7adf10c8dde87369658ddf4060cb61bce28d3b7be9a263eb72cab96c1",
            "population_id": (
                "phase2_target_population_b21cc0bf552899b13790e67647d02ff3c2a284552ff76444e2e1de9468cbc213"
            ),
            "manifest_id": "phase2_target_population_b182e430ec2a5cc30ebc2a2473d48b95d52f835f1ffc4c7b348ca1fe8e4609a0",
            "manifest_checksum": "sha256:e461088181f9568e5de029e76282a313f81de3f2cc1f2810e84eed0a0b985c09",
            "json_length": 68926,
            "json_checksum": "sha256:97295df5c2267a8962b5fcabeceb3d66474e7145cbb9a76169af03941ea6c2cd",
        },
    }
    for role, (config, manifest) in artifacts.items():
        payload = manifest.to_json().encode("utf-8")
        assert config.content_checksum == expected[role]["config_checksum"]
        assert config.population_id == expected[role]["population_id"]
        assert manifest.manifest_id == expected[role]["manifest_id"]
        assert manifest.content_checksum == expected[role]["manifest_checksum"]
        assert len(payload) == expected[role]["json_length"]
        assert f"sha256:{hashlib.sha256(payload).hexdigest()}" == expected[role]["json_checksum"]


def test_manifest_covers_all_families_strata_and_exact_allocations(
    development_manifest: TargetPopulationManifest,
) -> None:
    """Every preregistered family/stratum cell occurs at its exact target count."""
    represented = {(spec.family_id, spec.stratum_id) for spec in development_manifest.instances}
    expected = {(family_id, stratum_id) for family_id, strata in PRIMARY_FAMILY_STRATA.items() for stratum_id in strata}
    assert represented == expected
    for family_id, strata in PRIMARY_FAMILY_STRATA.items():
        counts = [
            sum(
                spec.family_id == family_id and spec.stratum_id == stratum_id for spec in development_manifest.instances
            )
            for stratum_id in strata
        ]
        assert len(set(counts)) == 1
        assert sum(counts) == 12


def test_instance_ids_are_unique_across_all_roles_and_not_phase_i(
    primary_screening_targets: TargetPopulationManifest,
    development_manifest: TargetPopulationManifest,
    confirmatory_manifest: TargetPopulationManifest,
) -> None:
    """Stable Phase II IDs are unique and pairwise disjoint across every target role."""
    ids_by_role = {
        "development": {spec.target_instance_id for spec in development_manifest.instances},
        "screening_selection": {spec.target_instance_id for spec in primary_screening_targets.instances},
        "confirmatory": {spec.target_instance_id for spec in confirmatory_manifest.instances},
    }
    for role, identifiers in ids_by_role.items():
        manifest = {
            "development": development_manifest,
            "screening_selection": primary_screening_targets,
            "confirmatory": confirmatory_manifest,
        }[role]
        assert len(identifiers) == len(manifest.instances)
    assert ids_by_role["development"].isdisjoint(ids_by_role["screening_selection"])
    assert ids_by_role["development"].isdisjoint(ids_by_role["confirmatory"])
    assert ids_by_role["screening_selection"].isdisjoint(ids_by_role["confirmatory"])
    all_ids = set().union(*ids_by_role.values())
    assert all(
        target_id.startswith(PHASE_II_TARGET_ID_PREFIX) and len(target_id) == len(PHASE_II_TARGET_ID_PREFIX) + 64
        for target_id in all_ids
    )
    assert all_ids.isdisjoint(PHASE_I_TARGET_IDS)


def test_config_spec_manifest_and_commitment_json_round_trips(
    development_config: TargetPopulationConfig,
    development_manifest: TargetPopulationManifest,
) -> None:
    """Every public target schema round-trips through strict canonical JSON."""
    spec = development_manifest.instances[0]
    commitment = development_manifest.public_commitment()
    assert TargetPopulationConfig.from_json(development_config.to_json()) == development_config
    assert TargetInstanceSpec.from_json(spec.to_json()) == spec
    assert TargetPopulationManifest.from_json(development_manifest.to_json()) == development_manifest
    assert TargetPopulationCommitment.from_json(commitment.to_json()) == commitment
    assert set(commitment.to_dict()) == {
        "schema_version",
        "target_manifest_checksum",
        "target_count_by_family",
        "content_checksum",
    }


def test_public_commitment_exposes_only_checksum_and_family_counts(
    development_manifest: TargetPopulationManifest,
) -> None:
    """No seed, target identifier, parameter, role, scope, or key leaks publicly."""
    payload = development_manifest.public_commitment().to_json()
    for forbidden in (
        "instance_seed",
        "target_instance_id",
        "parameters",
        "data_role",
        "population_scope",
        "role_master_entropy",
    ):
        assert forbidden not in payload
    assert json.loads(payload)["target_count_by_family"] == dict.fromkeys(sorted(PRIMARY_TARGET_FAMILIES), 12)


def test_target_records_are_deeply_immutable(
    development_config: TargetPopulationConfig,
    development_manifest: TargetPopulationManifest,
) -> None:
    """Frozen records and nested parameter mappings reject mutation."""
    with pytest.raises(FrozenInstanceError):
        development_config.data_role = "confirmatory"  # ty: ignore[invalid-assignment]
    with pytest.raises(TypeError):
        development_manifest.instances[0].parameters["changed"] = True  # ty: ignore[invalid-assignment]


def test_strict_json_rejects_noncanonical_duplicate_extra_and_nonfinite_data(
    development_config: TargetPopulationConfig,
    development_manifest: TargetPopulationManifest,
) -> None:
    """Target readers reject ambiguous and extended documents."""
    with pytest.raises(ValueError, match="canonical"):
        TargetPopulationConfig.from_json(json.dumps(development_config.to_dict(), indent=2))
    duplicated = '{"schema_version":"duplicate",' + development_config.to_json()[1:]
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        TargetPopulationConfig.from_json(duplicated)
    extended = development_manifest.to_dict()
    extended.pop("content_checksum")
    extended["state_vector"] = [[1.0, 0.0]]
    with pytest.raises(ValueError, match="fields do not match"):
        TargetPopulationManifest.from_dict(seal_mapping(extended))
    spec_data = development_manifest.instances[0].to_dict()
    cast("dict[str, object]", spec_data["parameters"])["mean"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        TargetInstanceSpec.from_dict(spec_data)
    with pytest.raises(TypeError, match="int"):
        TargetAllocation(
            family_id="gaussian_amplitude",
            stratum_id="interior",
            qubit_count=True,
            instance_count=1,
        )


@pytest.mark.parametrize("reserved_id", [PHASE_I_TARGET_IDS[0], "legacy_target_100", "phase1_target_deadbeef"])
def test_phase_i_and_legacy_ids_are_rejected(
    development_manifest: TargetPopulationManifest,
    reserved_id: str,
) -> None:
    """A resealed target spec cannot cross the Phase I or legacy namespace."""
    data = development_manifest.instances[0].to_dict()
    data.pop("content_checksum")
    data["target_instance_id"] = reserved_id
    with pytest.raises(ValueError, match="Phase II namespace"):
        TargetInstanceSpec.from_dict(seal_mapping(data))


def test_instance_specs_and_authorizations_are_factory_guarded(
    development_manifest: TargetPopulationManifest,
) -> None:
    """Direct construction cannot bypass target or materialization factories."""
    spec = development_manifest.instances[0]
    with pytest.raises(ValueError, match="factories"):
        TargetInstanceSpec(
            data_role=spec.data_role,
            population_config_checksum=spec.population_config_checksum,
            family_id=spec.family_id,
            stratum_id=spec.stratum_id,
            qubit_count=spec.qubit_count,
            instance_seed=spec.instance_seed,
            target_instance_id=spec.target_instance_id,
            parameters=spec.parameters,
            _marker=object(),
        )
    with pytest.raises(ValueError, match="may only be issued"):
        TargetMaterializationAuthorization(
            preregistration_checksum=_checksum("prereg"),
            population_config_checksum=_checksum("config"),
            target_manifest_checksum=_checksum("manifest"),
            data_role="development",
            _marker=object(),
        )


@pytest.mark.skipif(
    not GENERATION_RUNTIME_AVAILABLE,
    reason="Key-to-manifest generation requires preregistered NumPy 2.4.6.",
)
def test_wrong_role_master_is_rejected_before_generation(
    preregistration: InitialPreregistration,
    development_config: TargetPopulationConfig,
) -> None:
    """A different master key cannot silently change one config's population."""
    with pytest.raises(ValueError, match="does not match"):
        create_target_population_manifest(development_config, preregistration, SCREENING_MASTER)


def test_tampered_parameter_manifest_cannot_authorize(
    preregistration: InitialPreregistration,
    development_config: TargetPopulationConfig,
    development_manifest: TargetPopulationManifest,
) -> None:
    """A self-consistently resealed sampled parameter must still regenerate exactly."""
    original = next(spec for spec in development_manifest.instances if spec.family_id == "gaussian_amplitude")
    data = original.to_dict()
    data.pop("content_checksum")
    parameters = cast("dict[str, object]", data["parameters"])
    parameters["mean"] = cast("float", parameters["mean"]) + 1e-6
    tampered_spec = TargetInstanceSpec.from_dict(seal_mapping(data))
    instances = tuple(
        sorted(
            (
                tampered_spec if spec.target_instance_id == original.target_instance_id else spec
                for spec in development_manifest.instances
            ),
            key=lambda spec: (
                PRIMARY_TARGET_FAMILIES.index(spec.family_id),
                PRIMARY_FAMILY_STRATA[spec.family_id].index(spec.stratum_id),
                spec.qubit_count,
                spec.target_instance_id,
            ),
        )
    )
    tampered_manifest = TargetPopulationManifest(
        preregistration_checksum=development_manifest.preregistration_checksum,
        population_config_checksum=development_manifest.population_config_checksum,
        data_role=development_manifest.data_role,
        population_scope=development_manifest.population_scope,
        role_master_entropy_commitment=development_manifest.role_master_entropy_commitment,
        allocations=development_manifest.allocations,
        instances=instances,
    )
    with pytest.raises(ValueError, match="not the exact deterministic result"):
        authorize_target_materialization(
            preregistration,
            development_config,
            tampered_manifest,
            ROLE_MASTER,
        )


def test_manifest_rejects_missing_instance(
    development_manifest: TargetPopulationManifest,
) -> None:
    """A manifest cannot omit even one allocation member."""
    with pytest.raises(ValueError, match="exact Cartesian allocation"):
        TargetPopulationManifest(
            preregistration_checksum=development_manifest.preregistration_checksum,
            population_config_checksum=development_manifest.population_config_checksum,
            data_role=development_manifest.data_role,
            population_scope=development_manifest.population_scope,
            role_master_entropy_commitment=development_manifest.role_master_entropy_commitment,
            allocations=development_manifest.allocations,
            instances=development_manifest.instances[1:],
        )


def test_runtime_numpy_version_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    preregistration: InitialPreregistration,
    development_config: TargetPopulationConfig,
) -> None:
    """Generation refuses a runtime other than preregistered NumPy 2.4.6."""
    monkeypatch.setattr(target_module.np, "__version__", "2.4.5")
    with pytest.raises(RuntimeError, match=r"requires NumPy 2\.4\.6"):
        create_target_population_manifest(development_config, preregistration, ROLE_MASTER)


def test_tfim_rejection_uses_strict_gap_and_attempt_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A degenerate attempt is rejected and exhaustion occurs after attempt 99."""
    calls = 0

    def reject_once(_hamiltonian: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == 1:
            return np.zeros(4, dtype=np.float64)
        return np.arange(4, dtype=np.float64)

    monkeypatch.setattr(target_module.np.linalg, "eigvalsh", reject_once)
    parameters = target_module._tfim_parameter_record(  # noqa: SLF001
        ROLE_MASTER,
        f"{PHASE_II_TARGET_ID_PREFIX}{'0' * 64}",
        "critical",
        2,
    )
    assert parameters["attempt_index"] == 1
    assert cast("float", parameters["ground_state_gap"]) > cast("float", parameters["gap_threshold"])

    calls = 0

    def always_degenerate(_hamiltonian: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.zeros(4, dtype=np.float64)

    monkeypatch.setattr(target_module.np.linalg, "eigvalsh", always_degenerate)
    with pytest.raises(RuntimeError, match="exhausted"):
        target_module._tfim_parameter_record(  # noqa: SLF001
            ROLE_MASTER,
            f"{PHASE_II_TARGET_ID_PREFIX}{'1' * 64}",
            "critical",
            2,
        )
    assert calls == 100


def test_tfim_parameters_follow_regime_formula_and_gap_rule(
    development_manifest: TargetPopulationManifest,
) -> None:
    """Every disordered TFIM sample uses h=r*(0.8+0.4v) and passes its gap test."""
    for spec in development_manifest.instances:
        if spec.family_id != "tfim_ground_state":
            continue
        ratio = {"ferromagnetic": 0.5, "critical": 1.0, "paramagnetic": 1.5}[spec.stratum_id]
        couplings = cast("tuple[float, ...]", spec.parameters["couplings"])
        fields = cast("tuple[float, ...]", spec.parameters["fields"])
        assert all(0.8 <= value < 1.2 for value in couplings)
        assert all(0.8 * ratio <= value < 1.2 * ratio for value in fields)
        assert cast("float", spec.parameters["ground_state_gap"]) > cast(
            "float",
            spec.parameters["gap_threshold"],
        )


def test_materialization_has_manifest_parameter_and_little_endian_vector_agreement(
    development_config: TargetPopulationConfig,
    development_manifest: TargetPopulationManifest,
    development_materialization: TargetPopulationMaterialization,
) -> None:
    """Every immutable vector links the exact config, spec, parameters, and manifest."""
    assert development_materialization.target_manifest_checksum == development_manifest.content_checksum
    assert len(development_materialization.targets) == len(development_manifest.instances)
    spec_by_id = {spec.target_instance_id: spec for spec in development_manifest.instances}
    for target in development_materialization.targets:
        spec = spec_by_id[target.target_instance_id]
        vector = target.state_vector_copy()
        expected_bytes = np.ascontiguousarray(vector, dtype=np.dtype("<c16")).tobytes(order="C")
        assert target.population_config_checksum == development_config.content_checksum
        assert target.target_instance_spec_checksum == spec.content_checksum
        assert target.parameter_checksum == canonical_checksum(spec.parameters)
        assert target.vector_checksum == f"sha256:{hashlib.sha256(expected_bytes).hexdigest()}"
        assert vector.dtype == np.dtype(np.complex128)
        assert math.isclose(float(np.linalg.norm(vector)), 1.0, rel_tol=0.0, abs_tol=1e-12)
        pivot = vector[int(np.argmax(np.abs(vector)))]
        assert pivot.real > 0.0
        assert abs(pivot.imag) <= 1e-12


def test_representative_vectors_match_all_seven_stratum_goldens(
    development_manifest: TargetPopulationManifest,
    development_materialization: TargetPopulationMaterialization,
) -> None:
    """The first deterministic target in every family stratum has a frozen vector digest."""
    expected = {
        ("gaussian_amplitude", "interior"): (
            "phase2_target_03604b78221cea7d50427c2edbda9e53e68597f1799d7491b6ae890d425f9cd9",
            "sha256:a34093e9e1b9d8c0fbb911c34c5824006cfb4f3e50c546cfa466c0a3d16febc6",
        ),
        ("tfim_ground_state", "ferromagnetic"): (
            "phase2_target_767c7c95dcb8ace14c71cbdd44868e91f65458520e2c5d68c2795ece962d049c",
            "sha256:18d966fea976c3fc93d5559e2c38a242a48ca83d2bcaf338f7e03e6bf8de1e33",
        ),
        ("tfim_ground_state", "critical"): (
            "phase2_target_09f02925e37833dcf747f5f30275665e52a7aa92d70d7e325c43ac53ed8a8695",
            "sha256:06db341ef66cff4051029ccbf55bab846c31bb4f9e2d29f380c50cb95d0c2f23",
        ),
        ("tfim_ground_state", "paramagnetic"): (
            "phase2_target_0a0988be83294fb5df2a786ff9cfdb51ee4190aa98fc528b59305d353ca890f3",
            "sha256:c426be7d223841e314ef599ce5063e7c63573693c28b534b5b97e949fa19dc95",
        ),
        ("haar_random", "dense_complex"): (
            "phase2_target_036b5fbc49d8a94679f154ab274437c8f992033e458085e89c920b9b5d733f71",
            "sha256:b726c9be9354283caf7bce47b960a2f1e277263f5fd61e146c79696c5587f4b5",
        ),
        ("random_mps", "bond2"): (
            "phase2_target_0a6625ae794961334fa0348e095d069064c40f41733a5cc429aaf38fb6711a2a",
            "sha256:dacd71de7f3f90a6f6b5556ec41afb2d6d8a5ee48e4e99a196b40edafb463f35",
        ),
        ("random_mps", "bond3"): (
            "phase2_target_069631b9ca78facffc4c095edc5652fbbd687e4c100d98e2899cac7e9e267cf0",
            "sha256:455bae620236c4d080927ad3b5393ce7f0c2bd2120f52122fb9414d48e74929d",
        ),
    }
    observed = {}
    for family_id, stratum_id in expected:
        spec = next(
            spec
            for spec in development_manifest.instances
            if spec.family_id == family_id and spec.stratum_id == stratum_id
        )
        target = development_materialization.target(spec.target_instance_id)
        observed[family_id, stratum_id] = (spec.target_instance_id, target.vector_checksum)
    assert observed == expected
    assert development_materialization.content_checksum == (
        "sha256:a558f4a1519c3d754cec32256754f715dff8c69a353c0aeabdbba7fc2631b92f"
    )


def test_materialized_vector_copies_do_not_mutate_stored_bytes(
    development_materialization: TargetPopulationMaterialization,
) -> None:
    """Callers receive detached vectors rather than mutable internal storage."""
    target = development_materialization.targets[0]
    original = target.state_vector_copy()
    changed = target.state_vector_copy()
    changed[0] += 1.0
    np.testing.assert_array_equal(target.state_vector_copy(), original)


def test_gaussian_uses_bit_reversed_endpoint_excluded_grid(
    development_manifest: TargetPopulationManifest,
    development_materialization: TargetPopulationMaterialization,
) -> None:
    """Gaussian amplitudes use x_k=sum bit_i(k)*2**(-(i+1)), not arange/2**n."""
    spec = next(spec for spec in development_manifest.instances if spec.family_id == "gaussian_amplitude")
    vector = development_materialization.target(spec.target_instance_id).state_vector_copy()
    indices = np.arange(2**spec.qubit_count, dtype=np.uint64)
    grid = np.zeros(indices.size, dtype=np.float64)
    for site in range(spec.qubit_count):
        grid += ((indices >> np.uint64(site)) & np.uint64(1)).astype(np.float64) * (2.0 ** (-(site + 1)))
    mean = cast("float", spec.parameters["mean"])
    width = cast("float", spec.parameters["width"])
    expected = np.exp(-((grid - mean) ** 2) / (4.0 * width**2)).astype(np.complex128)
    expected /= np.linalg.norm(expected)
    np.testing.assert_array_equal(grid[:4], np.asarray([0.0, 0.5, 0.25, 0.75]))
    assert math.isclose(float(grid.max()), 63 / 64, rel_tol=0.0, abs_tol=0.0)
    np.testing.assert_allclose(vector, expected, rtol=0.0, atol=1e-15)


def test_tfim_vectors_are_ground_states_of_recorded_hamiltonians(
    development_manifest: TargetPopulationManifest,
    development_materialization: TargetPopulationMaterialization,
) -> None:
    """Each TFIM vector agrees with its recorded couplings, fields, and energy."""
    for spec in development_manifest.instances:
        if spec.family_id != "tfim_ground_state":
            continue
        vector = development_materialization.target(spec.target_instance_id).state_vector_copy()
        couplings = np.asarray(spec.parameters["couplings"], dtype=np.float64)
        fields = np.asarray(spec.parameters["fields"], dtype=np.float64)
        hamiltonian = target_module._tfim_hamiltonian(couplings, fields)  # noqa: SLF001
        energy = cast("float", spec.parameters["ground_energy"])
        assert np.linalg.norm(hamiltonian @ vector - energy * vector) < 1e-10


def test_random_mps_bonds_canonicalization_and_schmidt_ranks(
    development_manifest: TargetPopulationManifest,
    development_materialization: TargetPopulationMaterialization,
) -> None:
    """Random-MPS shapes follow the sealed bonds and dense states respect them."""
    expected_bonds = {
        "bond2": (1, 2, 2, 2, 2, 2, 1),
        "bond3": (1, 2, 3, 3, 3, 2, 1),
    }
    for stratum_id, bonds in expected_bonds.items():
        spec = next(
            spec
            for spec in development_manifest.instances
            if spec.family_id == "random_mps" and spec.stratum_id == stratum_id
        )
        assert cast("tuple[int, ...]", spec.parameters["bond_dimensions"]) == bonds
        vector = development_materialization.target(spec.target_instance_id).state_vector_copy()
        for cut in range(1, spec.qubit_count):
            coefficient_matrix = vector.reshape(2 ** (spec.qubit_count - cut), 2**cut)
            rank = np.linalg.matrix_rank(coefficient_matrix, tol=1e-11)
            assert rank <= bonds[cut]


def test_haar_vectors_are_dense_complex_and_family_substreams_differ(
    development_manifest: TargetPopulationManifest,
    development_materialization: TargetPopulationMaterialization,
) -> None:
    """Separate real/imaginary PCG64 streams produce dense complex Haar targets."""
    haar_specs = [spec for spec in development_manifest.instances if spec.family_id == "haar_random"]
    vectors = [development_materialization.target(spec.target_instance_id).state_vector_copy() for spec in haar_specs]
    assert all(np.count_nonzero(vector.real) == vector.size for vector in vectors)
    assert all(np.count_nonzero(vector.imag) == vector.size for vector in vectors)
    haar_checksums = {
        target.vector_checksum for target in development_materialization.targets if target.family_id == "haar_random"
    }
    assert len(haar_checksums) == 12


@pytest.mark.skipif(
    not GENERATION_RUNTIME_AVAILABLE,
    reason="Confirmatory manifest generation requires preregistered NumPy 2.4.6.",
)
def test_confirmatory_authorization_must_match_exact_manifest(
    preregistration: InitialPreregistration,
    confirmatory_config: TargetPopulationConfig,
    confirmatory_manifest: TargetPopulationManifest,
) -> None:
    """The WP15 token is required and bound to the exact confirmatory manifest checksum."""
    with pytest.raises(TypeError, match="ConfirmationAuthorization"):
        authorize_target_materialization(
            preregistration,
            confirmatory_config,
            confirmatory_manifest,
            CONFIRMATORY_MASTER,
        )
    wrong = _forged_confirmation_authorization(preregistration.content_checksum, _checksum("wrong targets"))
    with pytest.raises(ValueError, match="not bound"):
        authorize_target_materialization(
            preregistration,
            confirmatory_config,
            confirmatory_manifest,
            CONFIRMATORY_MASTER,
            wrong,
        )
    matching = _forged_confirmation_authorization(
        preregistration.content_checksum,
        confirmatory_manifest.content_checksum,
    )
    authorization = authorize_target_materialization(
        preregistration,
        confirmatory_config,
        confirmatory_manifest,
        CONFIRMATORY_MASTER,
        matching,
    )
    assert authorization.target_manifest_checksum == confirmatory_manifest.content_checksum


def test_unauthorized_confirmation_fails_before_manifest_regeneration(
    monkeypatch: pytest.MonkeyPatch,
    preregistration: InitialPreregistration,
    confirmatory_config: TargetPopulationConfig,
    confirmatory_manifest: TargetPopulationManifest,
) -> None:
    """A missing or mismatched final-seal token cannot trigger held-out regeneration work."""

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Unauthorized confirmation attempted to regenerate the target manifest.")

    monkeypatch.setattr(target_module, "create_target_population_manifest", forbidden)
    with pytest.raises(TypeError, match="ConfirmationAuthorization"):
        authorize_target_materialization(
            preregistration,
            confirmatory_config,
            confirmatory_manifest,
            CONFIRMATORY_MASTER,
        )
    wrong = _forged_confirmation_authorization(preregistration.content_checksum, _checksum("wrong targets"))
    with pytest.raises(ValueError, match="not bound"):
        authorize_target_materialization(
            preregistration,
            confirmatory_config,
            confirmatory_manifest,
            CONFIRMATORY_MASTER,
            wrong,
        )


def test_confirmation_token_cannot_authorize_development(
    preregistration: InitialPreregistration,
    development_config: TargetPopulationConfig,
    development_manifest: TargetPopulationManifest,
) -> None:
    """A confirmatory token cannot be replayed into a nonconfirmatory role."""
    token = _forged_confirmation_authorization(
        preregistration.content_checksum,
        development_manifest.content_checksum,
    )
    with pytest.raises(ValueError, match="cannot be reused"):
        authorize_target_materialization(
            preregistration,
            development_config,
            development_manifest,
            ROLE_MASTER,
            token,
        )


def test_materialization_authorization_is_bound_to_exact_inputs(
    preregistration: InitialPreregistration,
    development_config: TargetPopulationConfig,
    development_manifest: TargetPopulationManifest,
) -> None:
    """An authorization cannot be replayed after config or manifest substitution."""
    authorization = authorize_target_materialization(
        preregistration,
        development_config,
        development_manifest,
        ROLE_MASTER,
    )
    changed_config = build_target_population_config(
        preregistration,
        "development",
        role_master_entropy_commitment=role_master_entropy_commitment(SCREENING_MASTER),
    )
    changed_manifest = create_target_population_manifest(changed_config, preregistration, SCREENING_MASTER)
    with pytest.raises(ValueError, match="not bound"):
        materialize_target_population(
            changed_config,
            preregistration,
            changed_manifest,
            SCREENING_MASTER,
            authorization,
        )


def test_wp15_screening_manifest_resolves_exact_wp16_population(
    preregistration: InitialPreregistration,
    primary_screening_targets: TargetPopulationManifest,
) -> None:
    """Optimizer repetitions collapse to exactly the sealed WP16 target set."""
    screening = _screening_manifest_for_targets(preregistration, primary_screening_targets)
    verify_screening_target_population(screening, primary_screening_targets)


def test_wp15_wp16_link_rejects_checksum_missing_extra_and_metadata_changes(
    preregistration: InitialPreregistration,
    primary_screening_targets: TargetPopulationManifest,
) -> None:
    """Every checksum and target-resolution edge in the cross-package link is enforced."""
    wrong_checksum = _screening_manifest_for_targets(
        preregistration,
        primary_screening_targets,
        target_manifest_checksum=_checksum("wrong target manifest"),
    )
    with pytest.raises(ValueError, match="not sealed"):
        verify_screening_target_population(wrong_checksum, primary_screening_targets)
    omitted_id = primary_screening_targets.instances[0].target_instance_id
    omitted = _screening_manifest_for_targets(
        preregistration,
        primary_screening_targets,
        omitted_target_id=omitted_id,
    )
    with pytest.raises(ValueError, match="target sets differ"):
        verify_screening_target_population(omitted, primary_screening_targets)
    extra = _screening_manifest_for_targets(
        preregistration,
        primary_screening_targets,
        extra_target_id=f"{PHASE_II_TARGET_ID_PREFIX}{'f' * 64}",
    )
    with pytest.raises(ValueError, match="target sets differ"):
        verify_screening_target_population(extra, primary_screening_targets)
    mismatched = _screening_manifest_for_targets(
        preregistration,
        primary_screening_targets,
        mismatched_first_identity=True,
    )
    with pytest.raises(ValueError, match=r"inconsistent|target sets differ"):
        verify_screening_target_population(mismatched, primary_screening_targets)


def test_non_screening_population_cannot_feed_wp15_primary_screening(
    preregistration: InitialPreregistration,
    primary_screening_targets: TargetPopulationManifest,
    development_manifest: TargetPopulationManifest,
) -> None:
    """Only a screening-selection primary-q6 manifest may supply promotion targets."""
    screening = _screening_manifest_for_targets(
        preregistration,
        primary_screening_targets,
        target_manifest_checksum=development_manifest.content_checksum,
    )
    with pytest.raises(ValueError, match="screening_selection primary_q6"):
        verify_screening_target_population(screening, development_manifest)


def test_materialized_target_and_population_are_factory_validated(
    development_manifest: TargetPopulationManifest,
    development_materialization: TargetPopulationMaterialization,
) -> None:
    """Materialized records reject direct fabrication, subsets, and reordered ledgers."""
    spec = development_manifest.instances[0]
    vector = development_materialization.target(spec.target_instance_id).state_vector_copy()
    with pytest.raises(ValueError, match="authorized materializer"):
        MaterializedTarget(spec, development_manifest.content_checksum, vector, _marker=object())
    with pytest.raises(ValueError, match="authorized materializer"):
        TargetPopulationMaterialization(development_manifest, development_materialization.targets, _marker=object())
    with pytest.raises(ValueError, match="exact ordered instance specifications"):
        TargetPopulationMaterialization(
            development_manifest,
            development_materialization.targets[:-1],
            _marker=target_module._MATERIALIZATION_SENTINEL,  # noqa: SLF001
        )
    with pytest.raises(ValueError, match="exact ordered instance specifications"):
        TargetPopulationMaterialization(
            development_manifest,
            tuple(reversed(development_materialization.targets)),
            _marker=target_module._MATERIALIZATION_SENTINEL,  # noqa: SLF001
        )
