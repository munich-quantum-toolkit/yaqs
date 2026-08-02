# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for WP19 historical-reproduction evidence and payload codecs."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import operator
from dataclasses import replace
from statistics import fmean
from typing import TYPE_CHECKING

import numpy as np
import pytest

from benchmarks.state_preparation.phase2.canonical import canonical_checksum, canonical_json
from benchmarks.state_preparation.phase2.historical_reproduction import (
    LEGACY_LAYERWISE_METHOD_ID,
    LEGACY_REPRODUCTION_TARGET_SEEDS,
    MAX_LAYERWISE_MATERIALIZED_CIRCUIT_BYTES,
    LayerwiseMaterializedCircuit,
    LegacyReproductionOutcome,
    LegacyReproductionReport,
    compare_legacy_reproduction,
    decode_layerwise_materialized_circuit,
    encode_layerwise_materialized_circuit,
    load_archived_layerwise_reference,
)
from benchmarks.state_preparation.phase2.noisy_krotov import NoisyKrotovCircuitBinding
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


_RATIONALE = "Allow bounded floating-point drift from the pinned eigensolver and tensor-network runtime."
_SOURCE_MANIFEST_CHECKSUM = "sha256:" + "d" * 64
_RUNTIME_CHECKSUM = "sha256:" + "e" * 64


def _checksum(index: int) -> str:
    """Return a distinct well-formed checksum for a synthetic row."""
    return f"sha256:{index:064x}"


def _successes(fidelities: Sequence[float]) -> tuple[LegacyReproductionOutcome, ...]:
    """Build five synthetic successful computed rows.

    Returns:
        The five outcomes in canonical target-seed order.
    """
    assert len(fidelities) == 5
    return tuple(
        LegacyReproductionOutcome(
            target_seed=seed,
            status="success",
            computed_fidelity=float(fidelity),
            source_record_id=f"phase2_evaluation_seed_{seed}",
            source_record_checksum=_checksum(index + 1),
            runtime_fingerprint_checksum=_checksum(index + 101),
        )
        for index, (seed, fidelity) in enumerate(zip(LEGACY_REPRODUCTION_TARGET_SEEDS, fidelities, strict=True))
    )


def _compare(
    outcomes: Sequence[LegacyReproductionOutcome],
    *,
    tolerance: float,
    tolerance_rationale: str = _RATIONALE,
) -> LegacyReproductionReport:
    """Compare synthetic rows with fixed valid provenance links.

    Returns:
        The checksum-bound synthetic comparison report.
    """
    return compare_legacy_reproduction(
        outcomes,
        tolerance=tolerance,
        tolerance_rationale=tolerance_rationale,
        source_manifest_checksum=_SOURCE_MANIFEST_CHECKSUM,
        runtime_checksum=_RUNTIME_CHECKSUM,
    )


def _binding() -> NoisyKrotovCircuitBinding:
    """Return a small deterministic logical circuit binding."""
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("rx", (0,), param_index=0, logical_gate_id="rx_0"),
            ParameterizedGate("ry", (1,), param_index=1, logical_gate_id="ry_1"),
            ParameterizedGate("rzz", (0, 1), param_index=2, logical_gate_id="rzz_0_1"),
        ],
        num_params=3,
    )
    return NoisyKrotovCircuitBinding(circuit, "wp19_layerwise_payload_d1")


def _reseal(document: dict[str, object]) -> None:
    """Recompute the outer checksum of a mutable test document."""
    document["content_checksum"] = canonical_checksum({
        key: value for key, value in document.items() if key != "content_checksum"
    })


def test_archived_reference_is_loaded_from_the_trusted_audit_with_golden_mean() -> None:
    """The five numerical references retain their audited CSV provenance."""
    reference = load_archived_layerwise_reference()

    assert reference.method_id == LEGACY_LAYERWISE_METHOD_ID
    assert reference.target_seeds == LEGACY_REPRODUCTION_TARGET_SEEDS
    assert reference.fidelities == (
        0.7714404528882339,
        0.8133339085571137,
        0.8112718858400323,
        0.7482008490029097,
        0.8082234870125117,
    )
    assert float(reference.reference_mean).hex() == float("0.7904941166601602").hex()
    assert reference.reference_mean == fmean(reference.fidelities)
    assert reference.csv_artifact_id == "result_rigorous_csv"
    assert reference.csv_repo_path == "experiments/results/rigorous_benchmark_5states.csv"
    assert reference.csv_git_blob_id == "6f0a4fab48fa62751076095bb30f6804b741876b"
    assert reference.content_checksum == load_archived_layerwise_reference().content_checksum


def test_exact_synthetic_reproduction_round_trips_canonically() -> None:
    """Five supplied matching rows produce one stable reproduced report."""
    reference = load_archived_layerwise_reference()
    report = _compare(
        _successes(reference.fidelities),
        tolerance=1e-12,
    )

    assert report.classification == "reproduced"
    assert report.computed_mean == reference.reference_mean
    assert report.mean_delta is not None
    assert report.absolute_mean_delta is not None
    assert float(report.mean_delta).hex() == (0.0).hex()
    assert float(report.absolute_mean_delta).hex() == (0.0).hex()
    assert all(
        comparison.delta is not None and float(comparison.delta).hex() == (0.0).hex()
        for comparison in report.target_comparisons
    )
    assert all(comparison.within_tolerance for comparison in report.target_comparisons)
    assert report.to_json() == canonical_json(report.to_dict())
    assert LegacyReproductionReport.from_json(report.to_json()) == report
    assert LegacyReproductionReport.from_dict(report.to_dict()) == report


def test_synthetic_mismatch_uses_only_computed_rows_and_never_copies_references() -> None:
    """Mismatching supplied values remain the report's values and mean."""
    computed = (0.11, 0.12, 0.13, 0.14, 0.15)
    report = _compare(
        _successes(computed),
        tolerance=1e-6,
    )

    assert report.classification == "discrepant"
    assert tuple(item.outcome.computed_fidelity for item in report.target_comparisons) == computed
    assert tuple(item.reference_fidelity for item in report.target_comparisons) != computed
    assert report.computed_mean == fmean(computed)
    assert report.reference_mean == fmean(report.archived_reference.fidelities)
    assert report.computed_mean is not None
    assert report.mean_delta == report.computed_mean - report.reference_mean


def test_failure_is_retained_and_prevents_a_fabricated_five_target_mean() -> None:
    """A failed row remains explicit and cannot receive its reference value."""
    reference = load_archived_layerwise_reference()
    outcomes = list(_successes(reference.fidelities))
    outcomes[2] = LegacyReproductionOutcome(
        target_seed=300,
        status="failure",
        computed_fidelity=None,
        source_record_id="phase2_evaluation_seed_300_failure",
        source_record_checksum=_checksum(30),
        runtime_fingerprint_checksum=_checksum(130),
        failure_type="NumericalError",
        failure_message="Optimization produced a non-finite update.",
    )

    report = _compare(outcomes, tolerance=1e-6)
    failed = report.target_comparisons[2]

    assert report.classification == "discrepant"
    assert report.computed_mean is None
    assert report.mean_delta is None
    assert report.absolute_mean_delta is None
    assert failed.outcome.status == "failure"
    assert failed.outcome.computed_fidelity is None
    assert failed.delta is None
    assert failed.absolute_delta is None
    assert not failed.within_tolerance
    assert LegacyReproductionReport.from_json(report.to_json()) == report


@pytest.mark.parametrize(
    "mutator",
    [
        operator.itemgetter(slice(-1)),
        lambda outcomes: [*outcomes[:1], outcomes[0], *outcomes[2:]],
        lambda outcomes: list(reversed(outcomes)),
    ],
    ids=("missing", "duplicate", "reordered"),
)
def test_missing_duplicate_or_reordered_target_outcomes_are_rejected(
    mutator: Callable[[list[LegacyReproductionOutcome]], list[LegacyReproductionOutcome]],
) -> None:
    """The report input is the exact ordered five-target universe."""
    reference = load_archived_layerwise_reference()
    outcomes = list(_successes(reference.fidelities))
    changed = mutator(outcomes)

    with pytest.raises(ValueError, match="all five targets exactly once"):
        _compare(changed, tolerance=1e-6)


def test_duplicate_source_records_are_rejected_even_across_distinct_seeds() -> None:
    """Two target outcomes cannot project the same retained evidence record."""
    reference = load_archived_layerwise_reference()
    outcomes = list(_successes(reference.fidelities))
    outcomes[1] = replace(outcomes[1], source_record_id=outcomes[0].source_record_id)

    with pytest.raises(ValueError, match="reuse a source-record identity"):
        _compare(outcomes, tolerance=1e-6)


def test_duplicate_target_runtime_fingerprints_are_rejected() -> None:
    """Each outcome must bind its own target-prefix-specific WP18 fingerprint."""
    reference = load_archived_layerwise_reference()
    outcomes = list(_successes(reference.fidelities))
    outcomes[1] = replace(
        outcomes[1],
        runtime_fingerprint_checksum=outcomes[0].runtime_fingerprint_checksum,
    )

    with pytest.raises(ValueError, match="reuse a target runtime-fingerprint identity"):
        _compare(outcomes, tolerance=1e-6)


def test_resealed_report_cannot_forge_the_derived_classification() -> None:
    """A valid outer checksum cannot authorize inconsistent arithmetic."""
    report = _compare(
        _successes((0.1, 0.1, 0.1, 0.1, 0.1)),
        tolerance=1e-9,
    )
    document = report.to_dict()
    document["classification"] = "reproduced"
    _reseal(document)

    with pytest.raises(ValueError, match="not derived"):
        LegacyReproductionReport.from_dict(document)


def test_report_and_outcome_validation_reject_nonfinite_or_incomplete_values() -> None:
    """Tolerance and mutually exclusive success/failure fields are strict."""
    reference = load_archived_layerwise_reference()
    outcomes = _successes(reference.fidelities)

    with pytest.raises(ValueError, match="finite"):
        _compare(outcomes, tolerance=math.inf)
    with pytest.raises(ValueError, match="strictly positive"):
        _compare(outcomes, tolerance=0.0)
    with pytest.raises(ValueError, match="requires a fidelity"):
        LegacyReproductionOutcome(
            target_seed=100,
            status="success",
            computed_fidelity=None,
            source_record_id="phase2_evaluation_missing",
            source_record_checksum=_checksum(50),
            runtime_fingerprint_checksum=_checksum(150),
        )
    with pytest.raises(ValueError, match="forbids a fidelity"):
        LegacyReproductionOutcome(
            target_seed=100,
            status="failure",
            computed_fidelity=0.5,
            source_record_id="phase2_evaluation_invalid_failure",
            source_record_checksum=_checksum(51),
            runtime_fingerprint_checksum=_checksum(151),
            failure_type="Error",
            failure_message="failed",
        )


def test_layerwise_materialized_circuit_codec_is_deterministic_and_detached() -> None:
    """Binding metadata and selected theta round-trip as exact bounded bytes."""
    binding = _binding()
    theta = np.array([0.25, -0.5, 0.75], dtype=np.float64)

    payload = encode_layerwise_materialized_circuit(binding, theta)
    assert payload == encode_layerwise_materialized_circuit(binding, theta.copy())
    decoded = decode_layerwise_materialized_circuit(payload)

    assert decoded.circuit_binding.content_checksum == binding.content_checksum
    assert np.array_equal(decoded.selected_parameters, theta)
    assert decoded.to_bytes() == payload
    assert decoded.payload_checksum == f"sha256:{hashlib.sha256(payload).hexdigest()}"
    detached = decoded.selected_parameters
    detached[0] = 99.0
    assert np.array_equal(decoded.selected_parameters, theta)


def test_layerwise_materialized_circuit_codec_rejects_tampering_after_reseal() -> None:
    """Changing selected bytes cannot preserve their sealed parameter checksum."""
    payload = encode_layerwise_materialized_circuit(
        _binding(),
        np.array([0.25, -0.5, 0.75], dtype=np.float64),
    )
    document = json.loads(payload)
    parameter_document = document["selected_parameters"]
    parameter_document["data_base64"] = base64.b64encode(
        np.asarray([0.1, 0.2, 0.3], dtype=np.dtype("<f8")).tobytes()
    ).decode("ascii")
    _reseal(document)
    tampered = canonical_json(document).encode()

    with pytest.raises(ValueError, match="checksum"):
        decode_layerwise_materialized_circuit(tampered)


def test_layerwise_materialized_circuit_codec_rejects_bounds_and_noncanonical_bytes() -> None:
    """The decoder fails before allocating from oversized or ambiguous input."""
    with pytest.raises(ValueError, match="size bound"):
        decode_layerwise_materialized_circuit(b"x" * (MAX_LAYERWISE_MATERIALIZED_CIRCUIT_BYTES + 1))
    with pytest.raises(ValueError, match="canonical"):
        decode_layerwise_materialized_circuit(b'{ "schema_version": "invalid" }')
    with pytest.raises(ValueError, match="finite"):
        LayerwiseMaterializedCircuit(_binding(), np.array([0.0, np.nan, 0.0]))
