# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the fixed state-preparation benchmark targets."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from benchmarks.generate_state_preparation_targets import (
    GAUSSIAN_MEAN,
    GAUSSIAN_STANDARD_DEVIATION,
    TARGET_IDS,
    TFIM_SPECS,
    _gaussian_state,  # ruff: ignore[import-private-name] - tests intentionally lock private generator details.
    _mps_bond_dimensions,  # ruff: ignore[import-private-name] - tests intentionally lock private generator details.
    _paper_quantics_grid,  # ruff: ignore[import-private-name] - tests intentionally lock private generator details.
    _random_mps_state,  # ruff: ignore[import-private-name] - tests intentionally lock private generator details.
    _tfim_parameters,  # ruff: ignore[import-private-name] - tests intentionally lock private generator details.
    generate_target_data,
    generate_target_records,
    json_vector_to_state,
    main,
    target_data_matches,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

TARGET_FILE = Path(__file__).resolve().parents[2] / "benchmarks" / "state_preparation_target_states.json"


def _load_target_records() -> list[dict[str, object]]:
    payload = cast("dict[str, object]", json.loads(TARGET_FILE.read_text(encoding="utf-8")))
    return cast("list[dict[str, object]]", payload["targets"])


def _state_from_record(record: dict[str, object]) -> np.ndarray:
    encoded = cast("Sequence[Sequence[float]]", record["state_vector"])
    return json_vector_to_state(encoded)


def test_paper_quantics_grid_excludes_endpoint() -> None:
    """The Gaussian grid is endpoint-excluded on [0, 1)."""
    for num_qubits in (6, 12):
        grid = _paper_quantics_grid(num_qubits)
        np.testing.assert_equal(np.min(grid), 0.0)
        np.testing.assert_equal(np.max(grid), 1.0 - 2.0 ** (-num_qubits))


def test_paper_quantics_grid_uses_little_endian_coarse_bit_order() -> None:
    """Qubit 0 is the paper's first, coarsest quantics bit."""
    grid = _paper_quantics_grid(3)
    np.testing.assert_equal(grid[0b001], 0.5)
    np.testing.assert_equal(grid[0b010], 0.25)
    np.testing.assert_equal(grid[0b100], 0.125)
    np.testing.assert_equal(grid[0b111], 0.875)


def test_gaussian_state_reproduces_normalized_probability_density() -> None:
    """Gaussian measurement probabilities match the paper-style target distribution."""
    for num_qubits in (6, 12):
        state = _gaussian_state(num_qubits)
        assert state.shape == (2**num_qubits,)
        np.testing.assert_allclose(np.linalg.norm(state), 1.0, atol=1e-12)
        np.testing.assert_allclose(state.imag, 0.0, atol=1e-15)
        assert np.all(state.real >= -1e-15)

        grid = _paper_quantics_grid(num_qubits)
        expected_probs = np.exp(-((grid - GAUSSIAN_MEAN) ** 2) / (2.0 * GAUSSIAN_STANDARD_DEVIATION**2))
        expected_probs /= np.sum(expected_probs)
        actual_probs = np.abs(state) ** 2
        np.testing.assert_allclose(actual_probs, expected_probs, rtol=1e-12, atol=1e-14)


def test_random_mps_uses_quimb_style_bond_dimensions() -> None:
    """Random MPS targets use Quimb's open-chain internal bond convention."""
    assert _mps_bond_dimensions(6, 2) == (1, 2, 2, 2, 2, 2, 1)
    assert _mps_bond_dimensions(6, 3) == (1, 3, 3, 3, 3, 3, 1)
    assert _mps_bond_dimensions(12, 3) == (1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1)


def test_random_mps_states_are_real_normalized_dense_vectors() -> None:
    """Random MPS targets match Quimb's default real-valued tensor convention."""
    for num_qubits in (6, 12):
        for bond_dimension, seed in ((2, 5002), (3, 5003)):
            state, bond_dimensions = _random_mps_state(num_qubits, seed, bond_dimension)
            assert bond_dimensions == _mps_bond_dimensions(num_qubits, bond_dimension)
            assert state.shape == (2**num_qubits,)
            np.testing.assert_allclose(np.linalg.norm(state), 1.0, atol=1e-12)
            np.testing.assert_allclose(state.imag, 0.0, atol=1e-15)


def test_tfim_parameters_are_uniform() -> None:
    """TFIM targets use clean open-chain couplings and fields."""
    for num_qubits in (6, 12):
        for spec in TFIM_SPECS:
            couplings, fields = _tfim_parameters(num_qubits, spec)
            assert couplings.shape == (num_qubits - 1,)
            assert fields.shape == (num_qubits,)
            np.testing.assert_allclose(couplings, 1.0)
            np.testing.assert_allclose(fields, spec.h_over_j)


def test_tfim_records_describe_clean_uniform_model() -> None:
    """TFIM target metadata and vectors describe the clean uniform model."""
    records = generate_target_records((6,))
    tfim_records = [record for record in records if cast("str", record["target_id"]).startswith("tfim_")]
    assert len(tfim_records) == 3

    for record in tfim_records:
        assert record["seed"] is None
        params = cast("dict[str, object]", record["parameters"])
        assert params["model"] == "uniform_1d_transverse_field_ising_model"
        assert params["boundary_conditions"] == "open"
        assert "disorder_strength" not in params

        h_over_j = cast("float", params["h_over_j"])
        j_couplings = np.asarray(cast("Sequence[float]", params["j_couplings"]), dtype=np.float64)
        transverse_fields = np.asarray(cast("Sequence[float]", params["transverse_fields"]), dtype=np.float64)
        np.testing.assert_allclose(j_couplings, 1.0)
        np.testing.assert_allclose(transverse_fields, h_over_j)

        state = _state_from_record(record)
        assert state.shape == (2 ** cast("int", record["num_qubits"]),)
        assert np.all(np.isfinite(state))
        np.testing.assert_allclose(np.linalg.norm(state), 1.0, atol=1e-12)


def test_target_state_file_has_expected_shape_and_normalization() -> None:
    """The checked-in target file contains all 18 normalized benchmark states."""
    records = _load_target_records()
    assert len(records) == 18

    for num_qubits in (6, 12):
        target_ids = {cast("str", record["target_id"]) for record in records if record["num_qubits"] == num_qubits}
        assert target_ids == set(TARGET_IDS)

    for record in records:
        num_qubits = cast("int", record["num_qubits"])
        state = _state_from_record(record)
        assert state.shape == (2**num_qubits,)
        np.testing.assert_allclose(np.linalg.norm(state), 1.0, atol=1e-12)

        pivot = int(np.argmax(np.abs(state)))
        assert abs(state[pivot].imag) < 1e-12
        assert state[pivot].real > 0


def test_checked_in_targets_match_generator() -> None:
    """All checked-in targets and benchmark metadata are reproducible."""
    checked_in = json.loads(TARGET_FILE.read_text(encoding="utf-8"))
    assert target_data_matches(checked_in, generate_target_data())


def test_target_data_matching_ignores_provenance_versions_and_roundoff() -> None:
    """Freshness ignores environment versions and harmless numerical variation."""
    expected = generate_target_data((6,))
    actual = copy.deepcopy(expected)
    actual["numpy_version"] = "different-numpy-version"
    actual["scipy_version"] = "different-scipy-version"

    records = cast("list[dict[str, object]]", actual["targets"])
    gaussian_vector = cast("list[list[float]]", records[0]["state_vector"])
    gaussian_vector[0][0] += 1e-12
    tfim_parameters = cast("dict[str, object]", records[1]["parameters"])
    tfim_parameters["ground_energy"] = cast("float", tfim_parameters["ground_energy"]) + 1e-12

    assert target_data_matches(actual, expected)


def test_target_data_matching_rejects_stale_data() -> None:
    """Freshness rejects benchmark parameter, state-vector, and schema changes."""
    expected = generate_target_data((6,))

    stale_parameter = copy.deepcopy(expected)
    parameter_records = cast("list[dict[str, object]]", stale_parameter["targets"])
    tfim_parameters = cast("dict[str, object]", parameter_records[1]["parameters"])
    tfim_parameters["h_over_j"] = 0.75
    assert not target_data_matches(stale_parameter, expected)

    stale_vector = copy.deepcopy(expected)
    vector_records = cast("list[dict[str, object]]", stale_vector["targets"])
    state_vector = cast("list[list[float]]", vector_records[0]["state_vector"])
    state_vector[0][0] += 1e-6
    assert not target_data_matches(stale_vector, expected)

    missing_version = copy.deepcopy(expected)
    del missing_version["scipy_version"]
    assert not target_data_matches(missing_version, expected)


def test_cli_check_accepts_checked_in_targets() -> None:
    """The CLI semantic freshness check accepts the checked-in fixture."""
    assert main(["--output", str(TARGET_FILE), "--check"]) == 0
