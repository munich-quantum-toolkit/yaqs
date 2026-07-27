# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for validated state-preparation target loading."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from benchmarks.state_preparation import (
    SUPPORTED_QUBIT_COUNTS,
    TARGET_FIXTURE_FORMAT,
    TARGET_GENERATION_SEEDS,
    TARGET_IDS,
    TargetCollection,
    TargetRecord,
    TargetSelection,
    iter_targets,
    load_target,
    load_target_collection,
)
from benchmarks.state_preparation.targets import DEFAULT_TARGET_PATH

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def fixture_document() -> dict[str, Any]:
    """Return an independently mutable fixture document."""
    return cast("dict[str, Any]", json.loads(DEFAULT_TARGET_PATH.read_text(encoding="utf-8")))


def _write_fixture(tmp_path: Path, document: object) -> Path:
    """Write one JSON fixture in an isolated directory.

    Returns:
        Path to the encoded fixture.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "targets.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _write_raw_fixture(tmp_path: Path, document: bytes) -> Path:
    """Write raw fixture bytes for decoder failure tests.

    Returns:
        Path to the raw fixture.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "targets.json"
    path.write_bytes(document)
    return path


def _records(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the mutable target records from a fixture document."""
    records = document["targets"]
    assert isinstance(records, list)
    return cast("list[dict[str, Any]]", records)


def _record(document: dict[str, Any], target_id: str, num_qubits: int = 6) -> dict[str, Any]:
    """Return one mutable raw target record by composite key."""
    return next(
        record
        for record in _records(document)
        if record["target_id"] == target_id and record["num_qubits"] == num_qubits
    )


def _thaw_json(value: object) -> object:
    """Convert immutable loader metadata back to detached JSON-native data.

    Returns:
        Detached JSON-native data.
    """
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def test_load_all_targets_and_preserve_every_metadata_field(fixture_document: dict[str, Any]) -> None:
    """The checked-in fixture loads all records without losing metadata."""
    collection = load_target_collection()

    expected_keys = {(num_qubits, target_id) for num_qubits in SUPPORTED_QUBIT_COUNTS for target_id in TARGET_IDS}
    assert len(collection.records) == 18
    assert {record.key for record in collection.records} == expected_keys
    assert collection.fixture_format == TARGET_FIXTURE_FORMAT
    assert collection.format == TARGET_FIXTURE_FORMAT

    expected_metadata = {key: value for key, value in fixture_document.items() if key != "targets"}
    assert _thaw_json(collection.metadata) == expected_metadata
    raw_index = {
        (cast("int", record["num_qubits"]), cast("str", record["target_id"])): record
        for record in _records(fixture_document)
    }
    for record in collection.records:
        raw_record = raw_index[record.key]
        assert record.seed == raw_record["seed"]
        assert record.norm == raw_record["norm"]
        assert _thaw_json(record.parameters) == raw_record["parameters"]
        assert record.state_vector.shape == (2**record.num_qubits,)
        assert record.state_vector.dtype == np.complex128
        assert np.linalg.norm(record.state_vector) == pytest.approx(1.0, abs=1e-12)


def test_fixture_checksum_connects_to_target_selection() -> None:
    """Raw-byte fixture provenance can directly populate the Work Package 1 schema."""
    collection = load_target_collection()
    expected_checksum = f"sha256:{hashlib.sha256(DEFAULT_TARGET_PATH.read_bytes()).hexdigest()}"

    assert collection.fixture_checksum == expected_checksum
    assert collection.checksum == expected_checksum
    record = collection.load_target(6, "haar_random_1")
    selection = TargetSelection(
        num_qubits=record.num_qubits,
        target_id=record.target_id,
        target_seed=record.seed,
        fixture_format=collection.fixture_format,
        fixture_checksum=collection.fixture_checksum,
    )
    assert selection.fixture_checksum == expected_checksum


def test_module_and_collection_lookup_apis() -> None:
    """Both public API layers return the requested immutable records."""
    collection = load_target_collection()

    from_collection = collection.load_target(12, "tfim_critical")
    from_module = load_target(12, "tfim_critical")
    assert isinstance(from_collection, TargetRecord)
    assert from_collection.key == (12, "tfim_critical")
    np.testing.assert_array_equal(from_module.state_vector, from_collection.state_vector)
    assert len(tuple(iter_targets())) == 18


def test_filter_targets_and_preserve_fixture_order(fixture_document: dict[str, Any], tmp_path: Path) -> None:
    """Iteration filters independently and preserves the source record order."""
    document = copy.deepcopy(fixture_document)
    _records(document).reverse()
    collection = load_target_collection(_write_fixture(tmp_path, document))

    assert [record.key for record in collection.records] == [
        (cast("int", raw["num_qubits"]), cast("str", raw["target_id"])) for raw in _records(document)
    ]
    assert len(tuple(collection.iter_targets(num_qubits=6))) == 9
    assert len(tuple(collection.iter_targets(target_id="tfim_critical"))) == 2
    assert [record.key for record in collection.iter_targets(num_qubits=12, target_id="haar_random_1")] == [
        (12, "haar_random_1")
    ]


@pytest.mark.parametrize("num_qubits", [True, 6.0, 7])
def test_reject_invalid_qubit_filters(num_qubits: object) -> None:
    """Lookup filters accept only exact supported integer qubit counts."""
    collection = load_target_collection()
    with pytest.raises(ValueError, match="Unsupported qubit count"):
        tuple(collection.iter_targets(num_qubits=cast("Any", num_qubits)))


@pytest.mark.parametrize("target_id", [1, "", "unknown"])
def test_reject_invalid_target_filters(target_id: object) -> None:
    """Lookup filters accept only known string target identifiers."""
    collection = load_target_collection()
    with pytest.raises(ValueError, match="Unsupported target identifier"):
        collection.load_target(6, cast("Any", target_id))


def test_records_and_nested_metadata_are_irreversibly_immutable() -> None:
    """Frozen records use private bytes and recursively frozen JSON."""
    collection = load_target_collection()
    record = collection.load_target(6, "random_mps_bond2")

    with pytest.raises(ValueError, match="read-only"):
        record.state_vector[0] = 0
    with pytest.raises(ValueError, match=r"cannot set .* flag"):
        record.state_vector.setflags(write=True)
    with pytest.raises(TypeError):
        record.parameters["new"] = "value"  # ty: ignore[invalid-assignment]
    bond_dimensions = record.parameters["bond_dimensions"]
    assert isinstance(bond_dimensions, tuple)
    with pytest.raises(TypeError):
        bond_dimensions[0] = 2  # ty: ignore[invalid-assignment]
    with pytest.raises(TypeError):
        collection.metadata["new"] = "value"  # ty: ignore[invalid-assignment]
    with pytest.raises(FrozenInstanceError):
        record.seed = 2  # ty: ignore[invalid-assignment]

    mutable = record.state_vector_copy()
    mutable[0] = 0
    assert record.state_vector[0] != 0


def test_array_metadata_mutation_does_not_change_the_stored_vector() -> None:
    """Each array view has independent shape and dtype metadata."""
    record = load_target_collection().load_target(6, "haar_random_1")
    expected = record.state_vector.copy()

    reshaped_view = record.state_vector
    reshaped_view.shape = (8, 8)
    retyped_view = record.state_vector
    retyped_view.dtype = np.float64  # ty: ignore[invalid-assignment]  # NumPy supports runtime dtype mutation.

    assert record.state_vector.shape == (64,)
    assert record.state_vector.dtype == np.complex128
    np.testing.assert_array_equal(record.state_vector, expected)


@pytest.mark.parametrize(
    ("raw_document", "message"),
    [
        (b"{", "Could not parse"),
        (b'{"value": NaN}', "Non-finite JSON constant"),
        (b'{"format": "one", "format": "two"}', "Duplicate JSON object key"),
        (b"\xff", "UTF-8"),
    ],
)
def test_reject_malformed_encoded_fixtures(tmp_path: Path, raw_document: bytes, message: str) -> None:
    """Malformed JSON, duplicate keys, non-finite constants, and invalid UTF-8 fail cleanly."""
    with pytest.raises(ValueError, match=message):
        load_target_collection(_write_raw_fixture(tmp_path, raw_document))


def test_reject_missing_or_unreadable_fixture(tmp_path: Path) -> None:
    """Filesystem failures are exposed as contextual validation errors."""
    with pytest.raises(ValueError, match="Could not load target fixture"):
        load_target_collection(tmp_path / "missing.json")
    with pytest.raises(ValueError, match="Could not load target fixture"):
        load_target_collection(tmp_path)


def test_reject_non_object_root(tmp_path: Path) -> None:
    """The fixture envelope must be a JSON object."""
    with pytest.raises(ValueError, match="target fixture must be a JSON object"):
        load_target_collection(_write_fixture(tmp_path, []))


@pytest.mark.parametrize("field", ["format", "basis_order", "numpy_version", "targets"])
def test_reject_missing_root_fields(tmp_path: Path, fixture_document: dict[str, Any], field: str) -> None:
    """Every field in the v1 fixture envelope is required."""
    del fixture_document[field]
    with pytest.raises(ValueError, match="Target fixture fields do not match"):
        load_target_collection(_write_fixture(tmp_path, fixture_document))


def test_reject_unknown_root_field(tmp_path: Path, fixture_document: dict[str, Any]) -> None:
    """V1 envelope extensions require a format-version change."""
    fixture_document["unexpected"] = "metadata"
    with pytest.raises(ValueError, match=r"extra=.*unexpected"):
        load_target_collection(_write_fixture(tmp_path, fixture_document))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format", "v2", "Unsupported target fixture format"),
        ("generated_by", "other.py", "generated_by"),
        ("complex_encoding", "complex", "complex_encoding"),
        ("basis_order", "big_endian", "basis_order"),
        ("global_phase", "arbitrary", "global_phase"),
        ("numpy_version", "", "nonempty string"),
        ("scipy_version", 1, "nonempty string"),
        ("qubit_counts", [12, 6], "qubit_counts"),
        ("qubit_counts", [6.0, 12], "qubit_counts"),
        ("target_ids", list(reversed(TARGET_IDS)), "target_ids"),
        ("targets", {}, "must be an array"),
    ],
)
def test_reject_invalid_root_metadata(
    tmp_path: Path,
    fixture_document: dict[str, Any],
    field: str,
    value: object,
    message: str,
) -> None:
    """Fixture-level declarations are strict and versioned."""
    fixture_document[field] = value
    error = TypeError if field == "targets" else ValueError
    with pytest.raises(error, match=message):
        load_target_collection(_write_fixture(tmp_path, fixture_document))


@pytest.mark.parametrize("remove_index", [0, -1])
def test_reject_missing_record(tmp_path: Path, fixture_document: dict[str, Any], remove_index: int) -> None:
    """A fixture must contain every expected composite key."""
    _records(fixture_document).pop(remove_index)
    with pytest.raises(ValueError, match="key set is incomplete"):
        load_target_collection(_write_fixture(tmp_path, fixture_document))


def test_reject_duplicate_record(tmp_path: Path, fixture_document: dict[str, Any]) -> None:
    """A fixture cannot repeat a composite key."""
    records = _records(fixture_document)
    records.append(copy.deepcopy(records[0]))
    with pytest.raises(ValueError, match="Duplicate target"):
        load_target_collection(_write_fixture(tmp_path, fixture_document))


def test_reject_non_object_record(tmp_path: Path, fixture_document: dict[str, Any]) -> None:
    """Every target array element must be a JSON object."""
    cast("list[Any]", fixture_document["targets"])[0] = "not-an-object"
    with pytest.raises(ValueError, match="target record must be a JSON object"):
        load_target_collection(_write_fixture(tmp_path, fixture_document))


@pytest.mark.parametrize("field", ["target_id", "num_qubits", "seed", "parameters", "norm", "state_vector"])
def test_reject_missing_record_fields(tmp_path: Path, fixture_document: dict[str, Any], field: str) -> None:
    """Every v1 record field is required."""
    del _records(fixture_document)[0][field]
    with pytest.raises(ValueError, match="Target record fields do not match"):
        load_target_collection(_write_fixture(tmp_path, fixture_document))


def test_reject_unknown_record_field(tmp_path: Path, fixture_document: dict[str, Any]) -> None:
    """Unknown record fields are not silently discarded."""
    _records(fixture_document)[0]["unexpected"] = "metadata"
    with pytest.raises(ValueError, match=r"extra=.*unexpected"):
        load_target_collection(_write_fixture(tmp_path, fixture_document))


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("num_qubits", True, TypeError, "must be an integer"),
        ("num_qubits", 6.0, TypeError, "must be an integer"),
        ("num_qubits", 7, ValueError, "Unsupported qubit count"),
        ("target_id", 1, TypeError, "must be a string"),
        ("target_id", "unknown", ValueError, "Unsupported target identifier"),
    ],
)
def test_reject_invalid_record_keys(
    tmp_path: Path,
    fixture_document: dict[str, Any],
    field: str,
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    """Composite record keys use strict supported types and values."""
    _records(fixture_document)[0][field] = value
    with pytest.raises(error, match=message):
        load_target_collection(_write_fixture(tmp_path, fixture_document))


@pytest.mark.parametrize(
    ("target_id", "seed", "error"),
    [
        ("gaussian_mu0p5_sigma0p1", 99, ValueError),
        ("haar_random_1", 4002, ValueError),
        ("haar_random_1", 4001.0, TypeError),
        ("haar_random_1", True, TypeError),
    ],
)
def test_reject_invalid_target_seeds(
    tmp_path: Path,
    fixture_document: dict[str, Any],
    target_id: str,
    seed: object,
    error: type[Exception],
) -> None:
    """Every target identifier is bound to its exact generation seed and type."""
    _record(fixture_document, target_id)["seed"] = seed
    with pytest.raises(error, match="seed"):
        load_target_collection(_write_fixture(tmp_path, fixture_document))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mean", 0.6, "mean"),
        ("standard_deviation", True, "JSON number"),
        ("distribution", "uniform", "distribution"),
    ],
)
def test_reject_invalid_gaussian_parameters(
    tmp_path: Path,
    fixture_document: dict[str, Any],
    field: str,
    value: object,
    message: str,
) -> None:
    """Gaussian metadata exactly matches the fixed target definition."""
    _record(fixture_document, "gaussian_mu0p5_sigma0p1")["parameters"][field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        load_target_collection(_write_fixture(tmp_path, fixture_document))


@pytest.mark.parametrize("change", ["missing", "extra", "non-object"])
def test_reject_invalid_parameter_object_shape(
    tmp_path: Path,
    fixture_document: dict[str, Any],
    change: str,
) -> None:
    """Target parameter objects cannot be missing fields, extended, or replaced."""
    record = _record(fixture_document, "gaussian_mu0p5_sigma0p1")
    parameters = cast("dict[str, Any]", record["parameters"])
    if change == "missing":
        del parameters["source"]
    elif change == "extra":
        parameters["unexpected"] = "value"
    else:
        record["parameters"] = []
    with pytest.raises((TypeError, ValueError), match=r"parameters|Parameters"):
        load_target_collection(_write_fixture(tmp_path, fixture_document))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("distribution", "uniform", "distribution"),
        ("random_generator", "legacy", "random_generator"),
    ],
)
def test_reject_invalid_haar_parameters(
    tmp_path: Path,
    fixture_document: dict[str, Any],
    field: str,
    value: object,
    message: str,
) -> None:
    """Dense random targets retain their exact generation rule."""
    _record(fixture_document, "haar_random_1")["parameters"][field] = value
    with pytest.raises(ValueError, match=message):
        load_target_collection(_write_fixture(tmp_path, fixture_document))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_bond_dimension", 3, "maximum bond dimension"),
        ("max_bond_dimension", 2.0, "must be an integer"),
        ("bond_dimensions", [1, 2, 2, 1], "bond_dimensions"),
        ("tensor_distribution", "complex", "tensor_distribution"),
    ],
)
def test_reject_invalid_mps_parameters(
    tmp_path: Path,
    fixture_document: dict[str, Any],
    field: str,
    value: object,
    message: str,
) -> None:
    """Random-MPS metadata uses the exact seed, bond profile, and distribution."""
    _record(fixture_document, "random_mps_bond2")["parameters"][field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        load_target_collection(_write_fixture(tmp_path, fixture_document))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("regime", "critical", "regime"),
        ("eigensolver_initial_vector_seed", 1729, "eigensolver_initial_vector_seed"),
        ("h_over_j", 1.0, "h_over_j"),
        ("j_couplings", [1.0], "j_couplings"),
        ("transverse_fields", [0.5], "transverse_fields"),
        ("ground_energy", -5.0, "ground_energy"),
    ],
)
def test_reject_invalid_tfim_parameters(
    tmp_path: Path,
    fixture_document: dict[str, Any],
    field: str,
    value: object,
    message: str,
) -> None:
    """TFIM parameters agree with the identifier, qubit count, and loaded state."""
    _record(fixture_document, "tfim_ferro")["parameters"][field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        load_target_collection(_write_fixture(tmp_path, fixture_document))


def test_reject_tfim_state_that_is_not_an_eigenstate(tmp_path: Path, fixture_document: dict[str, Any]) -> None:
    """A normalized vector with matching energy expectation still must be a TFIM eigenstate."""
    record = _record(fixture_document, "tfim_ferro")
    record["state_vector"] = [[1.0, 0.0], *([[0.0, 0.0]] * 63)]
    record["norm"] = 1.0
    record["parameters"]["ground_energy"] = -5.0
    with pytest.raises(ValueError, match="eigenstate residual"):
        load_target_collection(_write_fixture(tmp_path, fixture_document))


def test_reject_vector_with_noncanonical_global_phase(tmp_path: Path, fixture_document: dict[str, Any]) -> None:
    """The declared largest-amplitude phase convention is enforced."""
    record = _record(fixture_document, "haar_random_1")
    record["state_vector"] = [[-real, -imaginary] for real, imaginary in record["state_vector"]]
    with pytest.raises(ValueError, match="global-phase convention"):
        load_target_collection(_write_fixture(tmp_path, fixture_document))


@pytest.mark.parametrize(
    ("state_vector", "message"),
    [
        ({}, "amplitudes"),
        ([[1.0]], "amplitudes"),
        ([1.0] * 64, r"\[real, imaginary\] pair"),
        ([[1.0, 0.0, 0.0]] * 64, r"\[real, imaginary\] pair"),
        ([[1.0, "bad"]] * 64, "JSON numbers"),
        ([[True, 0.0]] * 64, "JSON numbers"),
        ([[10**400, 0.0]] + [[0.0, 0.0]] * 63, "finite"),
        ([[1.0, 0.0]] * 63, "exactly 64"),
        ([[0.0, 0.0]] * 64, "has norm"),
        ([[0.5, 0.0]] + [[0.0, 0.0]] * 63, "has norm"),
    ],
)
def test_reject_invalid_vectors(
    tmp_path: Path,
    fixture_document: dict[str, Any],
    state_vector: object,
    message: str,
) -> None:
    """Malformed, overflowed, incorrectly sized, and unnormalized vectors fail."""
    _records(fixture_document)[0]["state_vector"] = state_vector
    with pytest.raises(ValueError, match=message):
        load_target_collection(_write_fixture(tmp_path, fixture_document))


@pytest.mark.parametrize(
    ("norm", "error", "message"),
    [
        (True, TypeError, "JSON number"),
        ("1.0", TypeError, "JSON number"),
        (10**400, ValueError, "finite float"),
        (0.5, ValueError, "does not match"),
    ],
)
def test_reject_invalid_stored_norms(
    tmp_path: Path,
    fixture_document: dict[str, Any],
    norm: object,
    error: type[Exception],
    message: str,
) -> None:
    """The recorded norm is finite, non-Boolean, and consistent with the vector."""
    _records(fixture_document)[0]["norm"] = norm
    with pytest.raises(error, match=message):
        load_target_collection(_write_fixture(tmp_path, fixture_document))


def test_target_collection_constructor_defends_its_index() -> None:
    """Direct collection construction cannot bypass format, checksum, or key-set validation."""
    collection = load_target_collection()

    with pytest.raises(ValueError, match="format"):
        TargetCollection("wrong", collection.fixture_checksum, collection.metadata, collection.records)
    with pytest.raises(ValueError, match="checksum"):
        TargetCollection(collection.fixture_format, "sha256:bad", collection.metadata, collection.records)
    with pytest.raises(TypeError, match="TargetRecord"):
        TargetCollection(
            collection.fixture_format,
            collection.fixture_checksum,
            collection.metadata,
            cast("Any", (object(),)),
        )
    with pytest.raises(ValueError, match="Duplicate"):
        TargetCollection(
            collection.fixture_format,
            collection.fixture_checksum,
            collection.metadata,
            (*collection.records, collection.records[0]),
        )
    with pytest.raises(ValueError, match="incomplete"):
        TargetCollection(
            collection.fixture_format,
            collection.fixture_checksum,
            collection.metadata,
            collection.records[:-1],
        )


def test_target_generation_seed_table_covers_loader_records() -> None:
    """Every loaded record seed matches the single shared target definition table."""
    for record in load_target_collection().records:
        assert record.seed == TARGET_GENERATION_SEEDS[record.target_id]
