# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the sealed WP19 reconstructed historical target collection."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import stat
from dataclasses import FrozenInstanceError, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from benchmarks.state_preparation.phase2 import legacy_targets as legacy_target_module
from benchmarks.state_preparation.phase2.canonical import canonical_checksum
from benchmarks.state_preparation.phase2.legacy_targets import (
    DEFAULT_LEGACY_TARGET_COLLECTION_PATH,
    LEGACY_TARGET_COLLECTION_SCHEMA_VERSION,
    LEGACY_TARGET_GENERATOR,
    LEGACY_TARGET_MISSING_PROVENANCE,
    LEGACY_TARGET_PHASE_INVARIANT_ATOL,
    LEGACY_TARGET_PHASE_INVARIANT_RTOL,
    LEGACY_TARGET_POPULATION_CONFIG_CHECKSUM,
    LEGACY_TARGET_REFERENCE_STATUS,
    LEGACY_TARGET_RUNTIME_SCHEMA_VERSION,
    LEGACY_TARGET_SEEDS,
    LEGACY_TARGET_SOURCE_CHECKSUM,
    LEGACY_TARGET_SOURCE_COMMIT,
    LEGACY_TARGET_SOURCE_PATH,
    MAX_LEGACY_TARGET_COLLECTION_BYTES,
    MAX_LEGACY_TARGET_VECTOR_BASE64_BYTES,
    MAX_LEGACY_TARGET_VECTOR_DECODED_BYTES,
    TRUSTED_LEGACY_TARGET_COLLECTION_CHECKSUM,
    LegacyMaterializedTarget,
    LegacyReconstructionRuntime,
    LegacyTargetCollection,
    capture_legacy_reconstruction_runtime,
    compare_statevectors_phase_invariant,
    load_legacy_target_collection,
    regenerate_legacy_tfim_target,
    validate_legacy_target_regeneration,
)
from benchmarks.state_preparation.phase2.pipeline import (
    LEGACY_REPRODUCTION_MANIFEST_CHECKSUM,
    LEGACY_REPRODUCTION_TARGET_IDS,
    fixture_target_spec_checksum,
)

if TYPE_CHECKING:
    from pathlib import Path


COLLECTION_CHECKSUM = "sha256:06f1f31c1ba6373837f2b41b6824fd9d45107b12a8c11b8973f4cf0af83954f4"
EXPECTED_VECTOR_CHECKSUMS = (
    "sha256:ff693beaa54cc47b10f7ec18148f2b90597cd6e8c9aa1b4eed71a2b64fb5965a",
    "sha256:f95cd6fce3718d210dda6f2bd8fb0265b8a0d04d74628cfe35fc8077261e1aed",
    "sha256:4f5d67b6d63c18a0f115bf38c4a27b3a79a1a53cf5082818b912324831690870",
    "sha256:a748d62bce69208264fc9cf7633385ae56d94bd4042d4604b1d4ef4df257c045",
    "sha256:ffa9e362fb77bb5519150b2bb9e01a68f86f557172158cc591d0bcd81360b2ec",
)
EXPECTED_COUPLING_BYTE_DIGESTS = (
    "d10330880aab480a74fb7fa4e9069a2d627e80f38e6e175da4998a73b3591279",
    "1216307ca253233921628cafc87019402b004d3d94df43b31ab9f153d0ae6e7c",
    "f7db39083f015d5181d760747438b223bca09dcc2954f3177fce28ccb2d0aa95",
    "634f3104a3efb7a6de5f118c1ae652eb3a5780a8c3ab93269cc8677f4b2b9b9d",
    "03314bdf6a486b2fd9247e3d87b035c8ed2a57a1e4551cb4064bf10f9ecabe57",
)
EXPECTED_FIELD_BYTE_DIGESTS = (
    "7a3eb5e8d8619660387b9c0302b38dd36f5ea9dd9e14849ebd1ecf1cb6a2cd48",
    "4d241305ca8eb266d2e169eceacc1a01a74a18185d72e903c3dfa8c22657dcf2",
    "8a346c2721fea1b2af60740adaf6bd7cbc8e9d085d6ce52970bb0275d216b425",
    "34818c8fe093d533248316198efe07b012dcaf1a2456478b7e4e23f8d0950016",
    "e53d6b36ffad21745b61179927f5cb70f19a917e03309992e0c6c7eb8b045780",
)
EXPECTED_GROUND_ENERGIES = (
    -9.463589977998124,
    -10.51953098464069,
    -9.906092033737044,
    -9.723211452879116,
    -9.552962782390237,
)
MATERIALIZED_IDENTITY_KEYS = {
    "target_instance_id",
    "target_instance_spec_checksum",
    "population_config_checksum",
    "target_manifest_checksum",
    "parameter_checksum",
    "family_id",
    "stratum_id",
    "qubit_count",
    "norm",
    "vector_checksum",
}


def _fixture_document() -> dict[str, Any]:
    """Return a mutable detached copy of the canonical target document."""
    return cast("dict[str, Any]", json.loads(DEFAULT_LEGACY_TARGET_COLLECTION_PATH.read_text(encoding="utf-8")))


def _reseal(document: dict[str, Any]) -> None:
    """Recompute one record's canonical ``content_checksum`` field."""
    payload = {key: value for key, value in document.items() if key != "content_checksum"}
    document["content_checksum"] = canonical_checksum(payload)


def _little_endian_float_digest(values: tuple[float, ...]) -> str:
    """Return the SHA-256 digest of canonical little-endian float64 values."""
    payload = np.ascontiguousarray(values, dtype=np.dtype("<f8")).tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def test_checked_in_collection_has_exact_identity_and_truthful_status() -> None:
    """The trusted fixture must contain only the five isolated q8 WP19 reconstructions."""
    collection = load_legacy_target_collection()

    assert collection.schema_version == LEGACY_TARGET_COLLECTION_SCHEMA_VERSION
    assert collection.content_checksum == COLLECTION_CHECKSUM == TRUSTED_LEGACY_TARGET_COLLECTION_CHECKSUM
    assert DEFAULT_LEGACY_TARGET_COLLECTION_PATH.stat().st_size < MAX_LEGACY_TARGET_COLLECTION_BYTES
    assert collection.source_commit == LEGACY_TARGET_SOURCE_COMMIT
    assert collection.source_path == LEGACY_TARGET_SOURCE_PATH
    assert collection.source_content_checksum == LEGACY_TARGET_SOURCE_CHECKSUM
    assert collection.reference_status == LEGACY_TARGET_REFERENCE_STATUS == "wp19_reconstructed_reference"
    assert not collection.archived_state_vectors_retained
    assert collection.missing_provenance == LEGACY_TARGET_MISSING_PROVENANCE
    assert "not retained" in " ".join(collection.missing_provenance)

    assert tuple(target.target_instance_id for target in collection.targets) == LEGACY_REPRODUCTION_TARGET_IDS
    assert tuple(target.seed for target in collection.targets) == LEGACY_TARGET_SEEDS
    assert tuple(target.vector_checksum for target in collection.targets) == EXPECTED_VECTOR_CHECKSUMS
    assert tuple(target.ground_energy for target in collection.targets) == EXPECTED_GROUND_ENERGIES
    for target in collection.targets:
        assert target.qubit_count == 8
        assert target.family_id == "tfim_ground_state"
        assert target.stratum_id == "legacy_disordered"
        assert target.reference_status == LEGACY_TARGET_REFERENCE_STATUS
        assert target.archived_vector_checksum is None
        assert target.target_manifest_checksum == LEGACY_REPRODUCTION_MANIFEST_CHECKSUM
        assert target.population_config_checksum == LEGACY_TARGET_POPULATION_CONFIG_CHECKSUM
        assert target.target_instance_spec_checksum == fixture_target_spec_checksum(
            "legacy_reproduction",
            target.target_instance_id,
            8,
        )
        assert math.isclose(target.norm, 1.0, rel_tol=0.0, abs_tol=1e-12)
        assert set(target.identity_dict()) == MATERIALIZED_IDENTITY_KEYS


def test_generator_contract_freezes_randomstate_draw_order_kron_order_and_eigh() -> None:
    """The generator metadata must spell out every historically relevant ordering choice."""
    generator = LEGACY_TARGET_GENERATOR

    assert generator["rng"] == "numpy.random.RandomState"
    assert generator["draw_order"] == ("couplings", "fields")
    assert generator["tensor_product_order"] == "generator_site_zero_is_most_significant_basis_bit"
    assert generator["diagonalization"] == "numpy.linalg.eigh"
    assert generator["eigenvector_selection"] == "eigenvectors_column_zero"
    assert generator["global_phase_convention"] == "none_historical_eigensolver_output"

    for seed in LEGACY_TARGET_SEEDS:
        target = load_legacy_target_collection().target(f"legacy_tfim_seed_{seed}")
        rng = np.random.RandomState(seed)
        assert tuple(float(value) for value in rng.uniform(0.8, 1.2, size=7)) == target.couplings
        assert tuple(float(value) for value in rng.uniform(0.8, 1.2, size=8)) == target.fields


def test_parameter_draws_match_stable_structural_goldens() -> None:
    """RandomState parameter draws must match portable little-endian byte goldens."""
    collection = load_legacy_target_collection()

    assert tuple(_little_endian_float_digest(target.couplings) for target in collection.targets) == (
        EXPECTED_COUPLING_BYTE_DIGESTS
    )
    assert tuple(_little_endian_float_digest(target.fields) for target in collection.targets) == (
        EXPECTED_FIELD_BYTE_DIGESTS
    )


def test_reconstruction_runtime_records_numerical_build_and_platform_provenance() -> None:
    """The checked-in reconstruction must identify Python, libraries, builds, and platform."""
    runtime = load_legacy_target_collection().reconstruction_runtime

    assert runtime.schema_version == LEGACY_TARGET_RUNTIME_SCHEMA_VERSION
    assert runtime.python_implementation == "CPython"
    assert runtime.python_version == "3.11.9"
    assert runtime.numpy_version == "2.4.6"
    assert runtime.scipy_version == "1.17.1"
    assert runtime.operating_system == "Darwin"
    assert runtime.platform == "macOS-26.6-arm64-arm-64bit"
    assert runtime.machine == "arm64"
    assert runtime.processor == "arm"
    assert runtime.byteorder == "little"
    for numerical_library in (runtime.blas, runtime.lapack):
        assert numerical_library["name"] == "accelerate"
        assert numerical_library["detection_method"] == "system"
        assert numerical_library["found"] is True
        assert numerical_library["version"] == "unknown"

    current = capture_legacy_reconstruction_runtime()
    assert current.python_version
    assert current.numpy_version
    assert current.scipy_version
    assert current.platform
    assert set(current.blas) == {
        "name",
        "version",
        "detection_method",
        "found",
        "include_directory",
        "lib_directory",
    }


def test_canonical_round_trip_and_materialized_target_immutability() -> None:
    """Serialization and vector access must not expose mutable collection state."""
    collection = load_legacy_target_collection()
    encoded = collection.to_json()

    assert encoded == DEFAULT_LEGACY_TARGET_COLLECTION_PATH.read_text(encoding="utf-8").removesuffix("\n")
    assert LegacyTargetCollection.from_json(encoded) == collection
    assert LegacyTargetCollection.from_dict(collection.to_dict()) == collection

    target = collection.targets[0]
    original = target.state_vector_copy()
    detached = target.state_vector_copy()
    assert detached.flags.writeable  # spellchecker:disable-line
    detached[0] = 0.0
    assert np.array_equal(target.state_vector_copy(), original)
    field_name = "seed"
    with pytest.raises(FrozenInstanceError):
        setattr(target, field_name, 999)


def test_phase_invariant_comparison_accepts_global_phase_and_rejects_perturbation() -> None:
    """Comparison must remove only global phase and retain the declared tight tolerance."""
    reference = load_legacy_target_collection().targets[0].state_vector_copy()
    phase = 0.713
    shifted = np.asarray(reference * np.exp(1j * phase), dtype=np.complex128)

    comparison = compare_statevectors_phase_invariant(reference, shifted)
    assert comparison.matches
    assert comparison.absolute_tolerance == LEGACY_TARGET_PHASE_INVARIANT_ATOL
    assert comparison.relative_tolerance == LEGACY_TARGET_PHASE_INVARIANT_RTOL
    assert comparison.overlap_magnitude == pytest.approx(1.0, abs=1e-12)
    assert comparison.phase_factor == pytest.approx(np.exp(-1j * phase), abs=1e-12)
    assert comparison.maximum_absolute_error < 1e-12

    perturbed = reference.copy()
    perturbed[0] += 1e-5
    perturbed /= np.linalg.norm(perturbed)
    mismatch = compare_statevectors_phase_invariant(reference, perturbed)
    assert not mismatch.matches
    assert mismatch.maximum_absolute_error > LEGACY_TARGET_PHASE_INVARIANT_ATOL


def test_all_five_targets_regenerate_phase_invariantly_within_declared_tolerance() -> None:
    """Fresh exact-generator executions must reproduce every stored reference numerically."""
    collection = load_legacy_target_collection()

    for target in collection.targets:
        comparison = validate_legacy_target_regeneration(target)
        assert comparison.matches
        assert comparison.maximum_absolute_error <= LEGACY_TARGET_PHASE_INVARIANT_ATOL
        regenerated = regenerate_legacy_tfim_target(target.seed)
        assert regenerated.couplings == target.couplings
        assert regenerated.fields == target.fields
        assert regenerated.ground_energy == pytest.approx(target.ground_energy, abs=1e-10, rel=1e-10)


def test_phase_invariant_regeneration_does_not_require_raw_eigensolver_checksum() -> None:
    """A phase-shifted eigensolver result must pass despite having different raw bytes."""
    target = load_legacy_target_collection().targets[1]
    candidate = np.asarray(target.state_vector_copy() * 1j, dtype=np.complex128)
    candidate_checksum = f"sha256:{hashlib.sha256(candidate.astype('<c16').tobytes()).hexdigest()}"

    assert candidate_checksum != target.vector_checksum
    assert compare_statevectors_phase_invariant(target.state_vector_copy(), candidate).matches


def test_regeneration_validator_cannot_bypass_the_sealed_tolerance() -> None:
    """The evidence validator must expose no caller-controlled tolerance override."""
    target = load_legacy_target_collection().targets[0]
    validator = cast("Any", validate_legacy_target_regeneration)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        validator(target, absolute_tolerance=1.0, relative_tolerance=1.0)


def test_collection_size_bounds_reject_before_json_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Oversized text and files must fail without invoking the canonical JSON parser."""

    def forbidden_parser(_payload: str) -> None:
        msg = "oversized payload reached the JSON parser"
        raise AssertionError(msg)

    monkeypatch.setattr(legacy_target_module, "load_canonical_json_object", forbidden_parser)
    oversized = "x" * (MAX_LEGACY_TARGET_COLLECTION_BYTES + 1)
    with pytest.raises(ValueError, match="size bound"):
        LegacyTargetCollection.from_json(oversized)

    oversized_path = tmp_path / "oversized_legacy_targets.json"
    oversized_path.write_bytes(oversized.encode("ascii"))
    with pytest.raises(ValueError, match="size bound"):
        load_legacy_target_collection(oversized_path)


def test_loader_rejects_symlink_before_opening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A symlink to the trusted bytes must fail before any descriptor is opened."""
    symlink = tmp_path / "legacy_targets_symlink.json"
    try:
        symlink.symlink_to(DEFAULT_LEGACY_TARGET_COLLECTION_PATH)
    except OSError as error:
        pytest.skip(f"Symlink creation is unavailable: {error}")

    def forbidden_open(_path: object, _flags: int) -> int:
        msg = "symlink reached os.open"
        raise AssertionError(msg)

    monkeypatch.setattr(legacy_target_module.os, "open", forbidden_open)
    with pytest.raises(ValueError, match="never a symbolic link"):
        load_legacy_target_collection(symlink)


def test_loader_rejects_symlinked_parent_before_opening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A regular file reached through a symlinked directory is still an aliased path."""
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    target = real_directory / "legacy_targets.json"
    target.write_bytes(DEFAULT_LEGACY_TARGET_COLLECTION_PATH.read_bytes())
    symlinked_directory = tmp_path / "alias"
    try:
        symlinked_directory.symlink_to(real_directory, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Directory symlink creation is unavailable: {error}")

    def forbidden_open(_path: object, _flags: int) -> int:
        msg = "symlinked parent reached os.open"
        raise AssertionError(msg)

    monkeypatch.setattr(legacy_target_module.os, "open", forbidden_open)
    with pytest.raises(ValueError, match="components must never be symbolic links"):
        load_legacy_target_collection(symlinked_directory / target.name)


def test_loader_rejects_fifo_before_opening_without_a_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A FIFO with no writer must be classified by lstat and can never block in open/read."""
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is unavailable on this platform.")
    fifo = tmp_path / "legacy_targets_fifo.json"
    os.mkfifo(fifo)

    def forbidden_open(_path: object, _flags: int) -> int:
        msg = "FIFO reached os.open"
        raise AssertionError(msg)

    monkeypatch.setattr(legacy_target_module.os, "open", forbidden_open)
    with pytest.raises(ValueError, match="regular file"):
        load_legacy_target_collection(fifo)


def test_loader_uses_nofollow_and_checks_opened_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regular-file precheck must be repeated on an alias-resistant opened descriptor."""
    real_open = os.open
    captured_flags: list[int] = []

    def recording_open(path: Path, flags: int) -> int:
        captured_flags.append(flags)
        return real_open(path, flags)

    monkeypatch.setattr(legacy_target_module.os, "open", recording_open)
    assert load_legacy_target_collection().content_checksum == COLLECTION_CHECKSUM
    assert len(captured_flags) == 1
    if hasattr(os, "O_NOFOLLOW"):
        assert captured_flags[0] & os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        assert captured_flags[0] & os.O_NONBLOCK

    real_fstat = os.fstat

    def fifo_descriptor(descriptor: int) -> os.stat_result:
        metadata = list(real_fstat(descriptor))
        metadata[stat.ST_MODE] = stat.S_IFIFO | 0o600
        return os.stat_result(metadata)

    monkeypatch.setattr(legacy_target_module.os, "fstat", fifo_descriptor)
    with pytest.raises(ValueError, match="descriptor must identify a regular file"):
        load_legacy_target_collection()


def test_base64_size_bounds_reject_before_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Encoded and declared decoded sizes must be bounded before Base64 allocation."""
    document = _fixture_document()
    target = cast("list[dict[str, Any]]", document["targets"])[0]
    vector = cast("dict[str, Any]", target["state_vector"])

    def forbidden_decoder(_payload: object, *, validate: bool) -> None:
        del validate
        msg = "oversized payload reached the Base64 decoder"
        raise AssertionError(msg)

    monkeypatch.setattr(legacy_target_module.base64, "b64decode", forbidden_decoder)
    vector["data_base64"] = "A" * (MAX_LEGACY_TARGET_VECTOR_BASE64_BYTES + 4)
    _reseal(target)
    with pytest.raises(ValueError, match="bounded encoded size"):
        LegacyMaterializedTarget.from_dict(target)

    target = cast("list[dict[str, Any]]", _fixture_document()["targets"])[0]
    vector = cast("dict[str, Any]", target["state_vector"])
    vector["amplitude_count"] = MAX_LEGACY_TARGET_VECTOR_DECODED_BYTES
    _reseal(target)
    with pytest.raises(ValueError, match="decoded size bound"):
        LegacyMaterializedTarget.from_dict(target)


def test_base64_decoded_size_is_exactly_bounded() -> None:
    """A maximum-length Base64 string that expands past 4096 bytes must be rejected."""
    target = cast("list[dict[str, Any]]", _fixture_document()["targets"])[0]
    vector = cast("dict[str, Any]", target["state_vector"])
    vector["data_base64"] = "A" * MAX_LEGACY_TARGET_VECTOR_BASE64_BYTES
    _reseal(target)

    with pytest.raises(ValueError, match=f"size bound {MAX_LEGACY_TARGET_VECTOR_DECODED_BYTES}"):
        LegacyMaterializedTarget.from_dict(target)


def test_outer_and_nested_checksum_tampering_is_detected() -> None:
    """Outer, parameter, and vector commitments must detect independent modifications."""
    outer_tamper = _fixture_document()
    outer_tamper["collection_id"] = "tampered_collection"
    with pytest.raises(ValueError, match="content checksum mismatch"):
        LegacyTargetCollection.from_dict(outer_tamper)

    parameter_tamper = _fixture_document()
    parameter_target = cast("list[dict[str, Any]]", parameter_tamper["targets"])[0]
    couplings = cast("list[float]", parameter_target["couplings"])
    couplings[0] += 1e-6
    _reseal(parameter_target)
    _reseal(parameter_tamper)
    with pytest.raises(ValueError, match="parameter_checksum does not match"):
        LegacyTargetCollection.from_dict(parameter_tamper)

    vector_tamper = _fixture_document()
    vector_target = cast("list[dict[str, Any]]", vector_tamper["targets"])[0]
    vector_envelope = cast("dict[str, Any]", vector_target["state_vector"])
    vector_bytes = bytearray(base64.b64decode(cast("str", vector_envelope["data_base64"]), validate=True))
    vector_bytes[0] ^= 1
    vector_envelope["data_base64"] = base64.b64encode(vector_bytes).decode("ascii")
    _reseal(vector_target)
    _reseal(vector_tamper)
    with pytest.raises(ValueError, match="vector_checksum does not match"):
        LegacyTargetCollection.from_dict(vector_tamper)


def test_resealing_cannot_mislabel_reconstructed_vectors_as_archived() -> None:
    """The schema must reject an invented archived-vector checksum even after resealing."""
    document = _fixture_document()
    target = cast("list[dict[str, Any]]", document["targets"])[0]
    target["archived_vector_checksum"] = target["vector_checksum"]
    _reseal(target)
    _reseal(document)

    with pytest.raises(ValueError, match="no archived vector was retained"):
        LegacyTargetCollection.from_dict(document)


@pytest.mark.parametrize("mutation", ["remove", "reorder"])
def test_collection_requires_exact_five_seed_order(mutation: str) -> None:
    """Cardinality and ordering are part of the immutable legacy fixture identity."""
    collection = load_legacy_target_collection()
    targets = collection.targets[:-1] if mutation == "remove" else tuple(reversed(collection.targets))

    with pytest.raises(ValueError, match="exactly q8 seeds"):
        replace(collection, targets=targets)


def test_strict_nested_schemas_reject_missing_and_extra_fields() -> None:
    """Every collection, target, runtime, and vector envelope has an exact field set."""
    document = _fixture_document()
    document["unexpected"] = True
    with pytest.raises(ValueError, match="fields do not match the schema"):
        LegacyTargetCollection.from_dict(document)

    target = cast("list[dict[str, Any]]", _fixture_document()["targets"])[0]
    target["unexpected"] = True
    with pytest.raises(ValueError, match="fields do not match the schema"):
        LegacyMaterializedTarget.from_dict(target)

    runtime = load_legacy_target_collection().reconstruction_runtime.to_dict()
    runtime.pop("scipy_version")
    with pytest.raises(ValueError, match="fields do not match the schema"):
        LegacyReconstructionRuntime.from_dict(runtime)

    vector_target = cast("list[dict[str, Any]]", _fixture_document()["targets"])[0]
    vector = cast("dict[str, Any]", vector_target["state_vector"])
    vector["unexpected"] = True
    _reseal(vector_target)
    with pytest.raises(ValueError, match="fields do not match the schema"):
        LegacyMaterializedTarget.from_dict(vector_target)


def test_loader_rejects_a_valid_but_unreviewed_collection_seal(tmp_path: Path) -> None:
    """Only the exact reviewed companion file may pass the public trusted loader."""
    collection = load_legacy_target_collection()
    changed_runtime = replace(collection.reconstruction_runtime, platform="unreviewed-platform")
    changed_collection = replace(collection, reconstruction_runtime=changed_runtime)
    changed_path = tmp_path / "legacy_targets.json"
    changed_path.write_text(f"{changed_collection.to_json()}\n", encoding="utf-8")

    assert LegacyTargetCollection.from_json(changed_collection.to_json()) == changed_collection
    with pytest.raises(ValueError, match="trusted runtime constant"):
        load_legacy_target_collection(changed_path)


@pytest.mark.parametrize("seed", [-1, 0, 99, 101, 501])
def test_regenerator_rejects_nonhistorical_seeds(seed: int) -> None:
    """The legacy fixture generator API must stay confined to the five audited targets."""
    with pytest.raises(ValueError, match="seed must"):
        regenerate_legacy_tfim_target(seed)
