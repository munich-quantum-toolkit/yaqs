# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""State-preparation benchmark definitions and orchestration."""

from __future__ import annotations

from .constants import (
    BALLARIN_NOISE_ID,
    DEPHASING_NOISE_IDS,
    DEPOLARIZING_NOISE_IDS,
    NOISE_IDS,
    NOISELESS_NOISE_ID,
    STANDARD_NOISE_IDS,
    SUPPORTED_QUBIT_COUNTS,
    TARGET_FIXTURE_FORMAT,
    TARGET_GENERATION_SEEDS,
    TARGET_IDS,
)
from .noise import PauliDistribution, sample_local_pauli, sample_product_pauli_channel
from .schema import (
    CONFIDENCE_INTERVAL_METHODS,
    CONFIG_SCHEMA_VERSION,
    CSV_COLUMNS,
    NOISE_DEFINITION_VERSION,
    RESULT_SCHEMA_VERSION,
    AnsatzConfig,
    BenchmarkConfig,
    BenchmarkFailure,
    BenchmarkResult,
    CircuitStatistics,
    EvaluationConfig,
    InitializationConfig,
    NoiseConfig,
    OptimizerConfig,
    TargetSelection,
    benchmark_record_from_csv_row,
    benchmark_record_from_dict,
    benchmark_record_from_json,
)
from .targets import TargetCollection, TargetRecord, iter_targets, load_target, load_target_collection

__all__ = [
    "BALLARIN_NOISE_ID",
    "CONFIDENCE_INTERVAL_METHODS",
    "CONFIG_SCHEMA_VERSION",
    "CSV_COLUMNS",
    "DEPHASING_NOISE_IDS",
    "DEPOLARIZING_NOISE_IDS",
    "NOISELESS_NOISE_ID",
    "NOISE_DEFINITION_VERSION",
    "NOISE_IDS",
    "RESULT_SCHEMA_VERSION",
    "STANDARD_NOISE_IDS",
    "SUPPORTED_QUBIT_COUNTS",
    "TARGET_FIXTURE_FORMAT",
    "TARGET_GENERATION_SEEDS",
    "TARGET_IDS",
    "AnsatzConfig",
    "BenchmarkConfig",
    "BenchmarkFailure",
    "BenchmarkResult",
    "CircuitStatistics",
    "EvaluationConfig",
    "InitializationConfig",
    "NoiseConfig",
    "OptimizerConfig",
    "PauliDistribution",
    "TargetCollection",
    "TargetRecord",
    "TargetSelection",
    "benchmark_record_from_csv_row",
    "benchmark_record_from_dict",
    "benchmark_record_from_json",
    "iter_targets",
    "load_target",
    "load_target_collection",
    "sample_local_pauli",
    "sample_product_pauli_channel",
]
