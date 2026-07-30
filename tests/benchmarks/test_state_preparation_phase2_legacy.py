# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Phase II canonical JSON and the commit-addressed legacy audit."""

from __future__ import annotations

import csv
import io
import json
import math
import shutil
import subprocess
from dataclasses import replace
from statistics import fmean, stdev
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import pytest

from benchmarks.state_preparation.phase2.canonical import (
    canonical_checksum,
    canonical_json,
    freeze_json,
    load_canonical_json_object,
    seal_mapping,
    thaw_json,
)
from benchmarks.state_preparation.phase2.legacy import (
    DEFAULT_LEGACY_AUDIT_PATH,
    LEGACY_CLASSIFICATIONS,
    LegacyArtifactRef,
    LegacyClaimAudit,
    LegacyEvidenceAudit,
    load_legacy_evidence_audit,
    verify_legacy_evidence_sources,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


AUDIT_CHECKSUM = "sha256:a294080bf54a62b2bad0df85faa2f75ade5098b6a9afd84dc81fbb29bafdda1c"
LEGACY_COMMIT = "fb621e2deb4da6f8ba16d3e48d05077d8e2b8809"
SHA_ZERO = f"sha256:{'0' * 64}"
BLOB_ZERO = "0" * 40
REPOSITORY_ROOT = DEFAULT_LEGACY_AUDIT_PATH.parents[4]

EXPECTED_CLASSIFICATIONS = {
    "shared_protocol_five_target_arithmetic": "reproduced",
    "historical_simulation_reproduction": "unreproducible",
    "bottom_up_method_identity": "discrepant",
    "topdown_pruning_identity": "discrepant",
    "hardware_execution_identity": "discrepant",
    "evaluation_seed_identity": "discrepant",
    "crn_numerical_claim": "discrepant",
    "state_five_failure_cause": "unreproducible",
    "method_budget_equivalence": "discrepant",
    "manuscript_figure_copy_provenance": "reproduced",
}


def _fixture_document() -> dict[str, Any]:
    """Return a detached mutable copy of the checked-in audit document."""
    return cast("dict[str, Any]", json.loads(DEFAULT_LEGACY_AUDIT_PATH.read_text(encoding="utf-8")))


def _reseal(document: dict[str, Any]) -> None:
    """Update a mutable audit document's outer content checksum."""
    payload = {key: value for key, value in document.items() if key != "content_checksum"}
    document["content_checksum"] = canonical_checksum(payload)


def _replace_artifact(audit: LegacyEvidenceAudit, replacement: LegacyArtifactRef) -> LegacyEvidenceAudit:
    """Return an audit with one artifact replaced by matching identity."""
    artifacts = tuple(
        replacement if artifact.artifact_id == replacement.artifact_id else artifact for artifact in audit.artifacts
    )
    return replace(audit, artifacts=artifacts)


def _read_commit_addressed_blob(artifact: LegacyArtifactRef) -> bytes:
    """Read an audited blob directly from Git rather than from the worktree.

    Returns:
        Exact bytes stored under the artifact's historical Git object identifier.
    """
    git_executable = shutil.which("git")
    assert git_executable is not None
    result = subprocess.run(  # noqa: S603 -- executable and audited blob identifier are validated
        [git_executable, "-C", str(REPOSITORY_ROOT), "cat-file", "blob", artifact.git_blob_id],
        check=True,
        capture_output=True,
    )
    return result.stdout


def test_default_audit_checksum_classifications_and_arithmetic() -> None:
    """The checked-in audit must retain its reviewed seal and numerical inventory."""
    audit = load_legacy_evidence_audit()

    assert audit.source_commit == LEGACY_COMMIT
    assert audit.content_checksum == AUDIT_CHECKSUM
    assert len(audit.artifacts) == 19
    assert len(audit.claims) == 10
    assert {claim.claim_id: claim.classification for claim in audit.claims} == EXPECTED_CLASSIFICATIONS
    assert frozenset(EXPECTED_CLASSIFICATIONS.values()) == frozenset(LEGACY_CLASSIFICATIONS)

    claim = audit.claim("shared_protocol_five_target_arithmetic")
    assert claim.configuration is not None
    method_fidelities = cast("Mapping[str, tuple[float, ...]]", claim.configuration["method_noisy_fidelities"])
    method_means = cast("Mapping[str, float]", claim.configuration["method_means"])
    layerwise = method_fidelities["layerwise_bmpd_crn_legacy_v1"]

    assert layerwise == (
        0.7714404528882339,
        0.8133339085571137,
        0.8112718858400323,
        0.7482008490029097,
        0.8082234870125117,
    )
    for method_id, fidelities in method_fidelities.items():
        assert method_means[method_id] == fmean(fidelities)
    assert float(method_means["layerwise_bmpd_crn_legacy_v1"]).hex() == float("0.7904941166601602").hex()
    assert claim.configuration["layerwise_sample_standard_deviation"] == stdev(layerwise)

    seed_policy = cast("Mapping[str, object]", claim.configuration["evaluation_trajectory_seed_policy"])
    assert seed_policy["base_seed"] == 0
    assert seed_policy["effective_seed_range_end"] == 499
    assert seed_policy["effective_seeds"] == (0, 1, 2, 3, 4)


def test_audited_fidelities_and_means_derive_from_historical_csv_blob() -> None:
    """Claimed values must equal an independent parse of the commit-addressed CSV."""
    audit = load_legacy_evidence_audit()
    artifact = audit.artifact("result_rigorous_csv")
    assert artifact.repo_path == "experiments/results/rigorous_benchmark_5states.csv"
    rows = tuple(csv.DictReader(io.StringIO(_read_commit_addressed_blob(artifact).decode("utf-8"))))

    audited_method_ids = {
        "Standard VQA": "standard_vqa",
        "ADAPT-VQE": "layerwise_bmpd_crn_legacy_v1",
        "Top-Down Krotov": "topdown_magnitude_pruning_legacy_v1",
    }
    derived: dict[str, list[float]] = {method_id: [] for method_id in audited_method_ids.values()}
    derived_seeds: dict[str, list[int]] = {method_id: [] for method_id in audited_method_ids.values()}
    for row in rows:
        method_id = audited_method_ids[row["Method"]]
        derived[method_id].append(float(row["True_Hardware_Fidelity"]))
        derived_seeds[method_id].append(int(row["State_Seed"]))

    claim = audit.claim("shared_protocol_five_target_arithmetic")
    assert claim.configuration is not None
    audited_fidelities = cast("Mapping[str, tuple[float, ...]]", claim.configuration["method_noisy_fidelities"])
    audited_means = cast("Mapping[str, float]", claim.configuration["method_means"])
    target_seeds = cast("tuple[int, ...]", claim.configuration["target_seeds"])

    assert len(rows) == 15
    for method_id, values in derived.items():
        assert tuple(derived_seeds[method_id]) == target_seeds
        assert tuple(values) == audited_fidelities[method_id]
        assert fmean(values) == audited_means[method_id]
    assert (
        stdev(derived["layerwise_bmpd_crn_legacy_v1"]) == (claim.configuration["layerwise_sample_standard_deviation"])
    )


def test_audited_crn_values_derive_from_historical_csv_blob() -> None:
    """The CRN discrepancy must be grounded in the archived trace's final row."""
    audit = load_legacy_evidence_audit()
    artifact = audit.artifact("result_crn_csv")
    assert artifact.repo_path == "experiments/crn_comparison.csv"
    rows = tuple(csv.DictReader(io.StringIO(_read_commit_addressed_blob(artifact).decode("utf-8"))))
    final = rows[-1]

    claim = audit.claim("crn_numerical_claim")
    assert claim.configuration is not None
    audited = cast("Mapping[str, float]", claim.configuration["final_fidelities"])
    csv_columns = {
        "noiseless": "Noiseless",
        "noisy_independent": "Noisy_Independent",
        "noisy_cross": "Noisy_Cross",
        "noisy_cross_crn": "Noisy_Cross_CRN",
    }

    assert int(final["Iteration"]) == claim.configuration["final_iteration"] == 20
    assert {name: float(final[column]) for name, column in csv_columns.items()} == audited


def test_default_audit_canonical_round_trip_and_detached_serialization() -> None:
    """Canonical serialization must round-trip without exposing mutable audit state."""
    audit = load_legacy_evidence_audit()
    encoded = audit.to_json()

    assert encoded == DEFAULT_LEGACY_AUDIT_PATH.read_text(encoding="utf-8").removesuffix("\n")
    assert LegacyEvidenceAudit.from_json(encoded) == audit
    assert LegacyEvidenceAudit.from_dict(audit.to_dict()) == audit

    claim = audit.claim("shared_protocol_five_target_arithmetic")
    assert isinstance(claim.configuration, MappingProxyType)
    assert claim.configuration is not None
    with pytest.raises(TypeError):
        cast("dict[str, object]", claim.configuration)["qubits"] = 99

    detached = audit.to_dict()
    detached_claims = cast("list[dict[str, Any]]", detached["claims"])
    arithmetic = next(item for item in detached_claims if item["claim_id"] == claim.claim_id)
    detached_configuration = cast("dict[str, Any]", arithmetic["configuration"])
    detached_configuration["qubits"] = 99
    assert claim.configuration["qubits"] == 8


def test_all_commit_addressed_legacy_sources_verify() -> None:
    """Every one of the nineteen references must resolve and match both digests."""
    audit = load_legacy_evidence_audit()

    verified = verify_legacy_evidence_sources(audit, REPOSITORY_ROOT)

    assert verified == tuple(artifact.artifact_id for artifact in audit.artifacts)
    assert len(verified) == 19


def test_canonical_json_freezes_sorts_normalizes_and_thaws() -> None:
    """Canonical JSON must be immutable internally and deterministic externally."""
    source = {"z": [1, {"negative_zero": -0.0}], "a": "Größe"}

    frozen = freeze_json(source)
    assert isinstance(frozen, MappingProxyType)
    source["z"] = []
    assert thaw_json(frozen) == {"a": "Größe", "z": [1, {"negative_zero": 0.0}]}
    assert canonical_json(frozen) == '{"a":"Größe","z":[1,{"negative_zero":0.0}]}'
    assert canonical_checksum({"b": 2, "a": 1}) == canonical_checksum({"a": 1, "b": 2})


def test_seal_mapping_is_detached_and_verifiable_by_audit_checksum_rules() -> None:
    """Sealing must not mutate or retain aliases to its source payload."""
    nested = [1, 2]
    source: dict[str, object] = {"name": "audit", "values": nested}

    sealed = seal_mapping(source)
    nested.append(3)

    assert source == {"name": "audit", "values": [1, 2, 3]}
    assert sealed["values"] == [1, 2]
    assert sealed["content_checksum"] == canonical_checksum({"name": "audit", "values": [1, 2]})
    with pytest.raises(ValueError, match="must not be present"):
        seal_mapping(sealed)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_canonical_json_loader_rejects_nonfinite_constants(constant: str) -> None:
    """Nonstandard non-finite JSON constants must never enter a sealed document."""
    with pytest.raises(ValueError, match="Nonstandard JSON constant"):
        load_canonical_json_object(f'{{"value":{constant}}}')


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_canonical_json_encoder_rejects_nonfinite_floats(value: float) -> None:
    """The encoder must reject non-finite Python floats before hashing."""
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json({"value": value})


def test_canonical_json_loader_rejects_duplicate_keys_even_when_nested() -> None:
    """Duplicate object members must not be silently resolved by the JSON decoder."""
    with pytest.raises(ValueError, match="Duplicate JSON key 'value'"):
        load_canonical_json_object('{"outer":{"value":1,"value":2}}')


@pytest.mark.parametrize(
    ("payload", "error", "message"),
    [
        ('{"b":2, "a":1}', ValueError, "not in canonical form"),
        ('{"a":1}\\n\\n', ValueError, "Could not decode"),
        ("[]", TypeError, "top level must be an object"),
    ],
)
def test_canonical_json_loader_rejects_noncanonical_documents(
    payload: str,
    error: type[Exception],
    message: str,
) -> None:
    """Only one canonical top-level object and at most one newline are accepted."""
    with pytest.raises(error, match=message):
        load_canonical_json_object(payload)


def test_outer_audit_tampering_is_detected() -> None:
    """Changing checksum-covered content without resealing must invalidate the audit."""
    document = _fixture_document()
    document["audit_id"] = "tampered-audit"

    with pytest.raises(ValueError, match="content checksum mismatch"):
        LegacyEvidenceAudit.from_dict(document)


def test_inner_configuration_tampering_is_detected_after_outer_reseal() -> None:
    """An attacker cannot bypass claim checksums by recomputing only the outer seal."""
    document = _fixture_document()
    claims = cast("list[dict[str, Any]]", document["claims"])
    arithmetic = next(item for item in claims if item["claim_id"] == "shared_protocol_five_target_arithmetic")
    configuration = cast("dict[str, object]", arithmetic["configuration"])
    configuration["qubits"] = 9
    _reseal(document)

    with pytest.raises(ValueError, match="configuration_checksum mismatch"):
        LegacyEvidenceAudit.from_dict(document)


def test_source_blob_and_content_tampering_are_detected() -> None:
    """Commit verification must independently check Git identity and SHA-256 content."""
    audit = load_legacy_evidence_audit()
    artifact = audit.artifacts[0]

    wrong_blob = replace(artifact, git_blob_id=BLOB_ZERO)
    with pytest.raises(ValueError, match="blob mismatch"):
        verify_legacy_evidence_sources(_replace_artifact(audit, wrong_blob), REPOSITORY_ROOT)

    wrong_checksum = replace(artifact, content_checksum=SHA_ZERO)
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_legacy_evidence_sources(_replace_artifact(audit, wrong_checksum), REPOSITORY_ROOT)


@pytest.mark.parametrize(
    "repo_path",
    [
        "/absolute/path",
        "../outside",
        "nested/../outside",
        "nested//file",
        "nested\\file",
        "./file",
        "file/",
    ],
)
def test_artifact_rejects_invalid_repository_paths(repo_path: str) -> None:
    """Commit-addressed evidence paths must be normalized and traversal-free."""
    with pytest.raises(ValueError, match="normalized relative POSIX path"):
        LegacyArtifactRef(
            artifact_id="invalid-path",
            repo_path=repo_path,
            source_commit=LEGACY_COMMIT,
            git_blob_id="a" * 40,
            content_checksum=f"sha256:{'a' * 64}",
            role="result",
        )


def test_source_verification_rejects_missing_commit_path() -> None:
    """A syntactically valid reference must still resolve at the frozen commit."""
    audit = load_legacy_evidence_audit()
    artifact = replace(audit.artifacts[0], repo_path="missing/legacy-evidence.bin")

    with pytest.raises(ValueError, match="Could not resolve legacy artifact"):
        verify_legacy_evidence_sources(_replace_artifact(audit, artifact), REPOSITORY_ROOT)


def test_audit_rejects_unknown_claim_and_environment_references() -> None:
    """Every claim and environment identifier must resolve inside the same audit."""
    audit = load_legacy_evidence_audit()
    claim = audit.claim("bottom_up_method_identity")
    invalid_claim = replace(claim, artifact_ids=("missing_artifact",))
    invalid_claims = tuple(invalid_claim if item.claim_id == claim.claim_id else item for item in audit.claims)

    with pytest.raises(ValueError, match="references unknown artifacts"):
        replace(audit, claims=invalid_claims)
    with pytest.raises(ValueError, match="environment_lock_artifact_id"):
        replace(audit, environment_lock_artifact_id="missing_environment")
    with pytest.raises(ValueError, match="environment artifact"):
        replace(audit, environment_lock_artifact_id="result_rigorous_csv")


def test_audit_rejects_duplicate_and_cross_commit_records() -> None:
    """Audit identities must be unique and all artifacts must address one commit."""
    audit = load_legacy_evidence_audit()

    with pytest.raises(ValueError, match="unique artifact_id"):
        replace(audit, artifacts=(*audit.artifacts, audit.artifacts[0]))
    with pytest.raises(ValueError, match="unique claim_id"):
        replace(audit, claims=(*audit.claims, audit.claims[0]))

    cross_commit = replace(audit.artifacts[0], source_commit="a" * 40)
    with pytest.raises(ValueError, match="must use the audit source_commit"):
        _replace_artifact(audit, cross_commit)


def test_claim_configuration_checksum_and_null_pairing_are_enforced() -> None:
    """Configuration evidence and its checksum must form one internally sealed pair."""
    audit = load_legacy_evidence_audit()
    claim = audit.claim("bottom_up_method_identity")

    with pytest.raises(ValueError, match="configuration_checksum mismatch"):
        replace(claim, configuration_checksum=SHA_ZERO)
    with pytest.raises(ValueError, match="must be null"):
        LegacyClaimAudit(
            claim_id="null-configuration",
            statement="No configuration was retained.",
            classification="unreproducible",
            artifact_ids=(audit.artifacts[0].artifact_id,),
            manuscript_locations=(),
            configuration=None,
            configuration_checksum=SHA_ZERO,
            limitations=("The configuration is unavailable.",),
        )
    with pytest.raises(ValueError, match="required for every audited claim"):
        LegacyClaimAudit(
            claim_id="missing-configuration",
            statement="No configuration was recorded.",
            classification="unreproducible",
            artifact_ids=(audit.artifacts[0].artifact_id,),
            manuscript_locations=(),
            configuration=None,
            configuration_checksum=None,
            limitations=("The configuration is unavailable.",),
        )


@pytest.mark.parametrize("record_kind", ["artifact_missing", "claim_extra", "audit_extra"])
def test_legacy_records_reject_nonversioned_fields(record_kind: str) -> None:
    """Exact field sets prevent silent acceptance of misspelled or future fields."""
    document = _fixture_document()
    if record_kind == "artifact_missing":
        artifact = cast("list[dict[str, object]]", document["artifacts"])[0]
        artifact.pop("role")
        with pytest.raises(ValueError, match="fields do not match the schema"):
            LegacyArtifactRef.from_dict(artifact)
    elif record_kind == "claim_extra":
        claim = cast("list[dict[str, object]]", document["claims"])[0]
        claim["unexpected"] = True
        with pytest.raises(ValueError, match="fields do not match the schema"):
            LegacyClaimAudit.from_dict(claim)
    else:
        document["unexpected"] = True
        with pytest.raises(ValueError, match="fields do not match the schema"):
            LegacyEvidenceAudit.from_dict(document)


def test_checked_in_figure_pairs_are_byte_identical_at_the_legacy_commit() -> None:
    """The audit must identify exact experiment/manuscript copies, not similar plots."""
    audit = load_legacy_evidence_audit()
    claim = audit.claim("manuscript_figure_copy_provenance")
    assert claim.configuration is not None
    pairs = cast("tuple[tuple[str, str], ...]", claim.configuration["matching_pairs"])

    for experiment_id, manuscript_id in pairs:
        experiment = audit.artifact(experiment_id)
        manuscript = audit.artifact(manuscript_id)
        assert experiment.git_blob_id == manuscript.git_blob_id
        assert experiment.content_checksum == manuscript.content_checksum

    assert audit.artifact("figure_topdown_experiment").repo_path == ("experiments/results/pruning_tradeoff_tfim.png")
    assert audit.artifact("figure_topdown_manuscript").repo_path == ("manuscript/figures/pruning_tradeoff_tfim.png")
