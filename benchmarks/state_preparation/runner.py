# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Command-line orchestration for state-preparation benchmark matrices."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback as traceback_module
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, NoReturn, cast

from .constants import (
    BALLARIN_NOISE_ID,
    DEPHASING_NOISE_IDS,
    DEPOLARIZING_NOISE_IDS,
    NOISE_IDS,
    NOISELESS_NOISE_ID,
    STANDARD_NOISE_IDS,
    SUPPORTED_QUBIT_COUNTS,
    TARGET_IDS,
)
from .evaluation import evaluate_state_preparation_artifact
from .methods import (
    KROTOV_METHOD_ID,
    KROTOV_METHOD_VERSION,
    KrotovStatePreparationMethod,
    StatePreparationTrainingError,
    state_preparation_training_id,
    train_state_preparation_method,
)
from .reporting import RESULTS_JSONL_NAME, BenchmarkReportStore, ReportingError, capture_run_provenance
from .schema import (
    AnsatzConfig,
    BenchmarkConfig,
    EvaluationConfig,
    InitializationConfig,
    NoiseConfig,
    OptimizerConfig,
    TargetSelection,
)
from .targets import TargetCollection, load_target_collection

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Sequence

RUNNER_CONFIGURATION_FORMAT = "yaqs.state_preparation.runner_config.v1"
PRESET_NAMES = ("smoke", "minimum", "full")
METHOD_IDS = (KROTOV_METHOD_ID,)
MINIMUM_NOISE_IDS = (
    NOISELESS_NOISE_ID,
    BALLARIN_NOISE_ID,
    DEPHASING_NOISE_IDS[0],
    DEPHASING_NOISE_IDS[1],
    DEPOLARIZING_NOISE_IDS[0],
    DEPOLARIZING_NOISE_IDS[1],
)

_CONFIGURATION_KEYS = frozenset({
    "format",
    "preset",
    "num_qubits",
    "target_id",
    "target_ids",
    "noise_id",
    "noise_ids",
    "method",
    "methods",
    "num_layers",
    "initialization_seed",
    "initialization_seeds",
    "optimizer_iterations",
    "train_trajectories",
    "test_trajectories",
    "output_dir",
    "resume",
    "overwrite",
    "dry_run",
    "fail_fast",
})
_CONFIGURATION_ALIASES = {
    "target_id": "target_ids",
    "noise_id": "noise_ids",
    "method": "methods",
    "initialization_seed": "initialization_seeds",
}
_UINT64_MODULUS = 2**64


class RunnerConfigurationError(ValueError):
    """Raised when CLI or JSON runner configuration is invalid."""


@dataclass(frozen=True, slots=True)
class RunnerOptions:
    """Fully resolved non-scientific runner and matrix options."""

    preset: str
    num_qubits: tuple[int, ...]
    target_ids: tuple[str, ...]
    noise_ids: tuple[str, ...]
    methods: tuple[str, ...]
    num_layers: tuple[int, ...]
    initialization_seeds: tuple[int, ...]
    optimizer_iterations: int
    train_trajectories: int
    test_trajectories: int
    output_dir: Path
    resume: bool
    overwrite: bool
    dry_run: bool
    fail_fast: bool


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Aggregate outcome of one sequential matrix execution."""

    planned: int
    attempted: int
    succeeded: int
    failed: int
    skipped: int


def _reject_json_constant(value: str) -> NoReturn:
    """Reject nonstandard JSON numeric constants.

    Raises:
        RunnerConfigurationError: Always, because the constant is non-finite.
    """
    msg = f"Non-finite JSON constant {value!r} is not supported."
    raise RunnerConfigurationError(msg)


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while rejecting duplicate keys.

    Returns:
        The decoded object.

    Raises:
        RunnerConfigurationError: If a key occurs more than once.
    """
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            msg = f"Duplicate JSON configuration key {key!r}."
            raise RunnerConfigurationError(msg)
        result[key] = value
    return result


def load_configuration_file(path: Path) -> dict[str, object]:
    """Load a strict partial runner configuration.

    Returns:
        Validated top-level overrides.

    Raises:
        TypeError: If ``path`` is not a path.
        RunnerConfigurationError: If the file is unreadable or malformed.
    """
    if not isinstance(path, Path):
        msg = f"path must be a pathlib.Path, got {type(path).__name__}."
        raise TypeError(msg)
    try:
        payload = path.read_text(encoding="utf-8")
        decoded = json.loads(
            payload,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        msg = f"Could not read JSON configuration {path}: {error}."
        raise RunnerConfigurationError(msg) from error
    if not isinstance(decoded, dict):
        msg = "Runner configuration must be a JSON object."
        raise RunnerConfigurationError(msg)
    if any(type(key) is not str for key in decoded):
        msg = "Runner configuration keys must be strings."
        raise RunnerConfigurationError(msg)
    string_decoded = cast("dict[str, object]", decoded)
    unknown = sorted(set(string_decoded) - _CONFIGURATION_KEYS)
    if unknown:
        msg = f"Unknown runner configuration fields: {unknown}."
        raise RunnerConfigurationError(msg)
    configuration_format = string_decoded.get("format", RUNNER_CONFIGURATION_FORMAT)
    if configuration_format != RUNNER_CONFIGURATION_FORMAT:
        msg = f"format must be {RUNNER_CONFIGURATION_FORMAT!r}."
        raise RunnerConfigurationError(msg)
    for alias, canonical in _CONFIGURATION_ALIASES.items():
        if alias in string_decoded and canonical in string_decoded:
            msg = f"Configuration cannot contain both {alias!r} and {canonical!r}."
            raise RunnerConfigurationError(msg)
        if alias in string_decoded:
            string_decoded[canonical] = string_decoded.pop(alias)
    return string_decoded


def create_argument_parser() -> argparse.ArgumentParser:
    """Create the public command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.state_preparation.runner",
        description="Run reproducible state-preparation benchmark matrices.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=Path, help="JSON configuration file.")
    parser.add_argument("--preset", choices=PRESET_NAMES)
    parser.add_argument("--num-qubits", type=int, action="append")
    parser.add_argument("--target-id", dest="target_ids", action="append", metavar="TARGET_ID")
    parser.add_argument("--noise-id", dest="noise_ids", action="append", metavar="NOISE_ID")
    parser.add_argument("--method", dest="methods", action="append", metavar="METHOD")
    parser.add_argument("--num-layers", type=int, action="append")
    parser.add_argument(
        "--initialization-seed",
        dest="initialization_seeds",
        type=int,
        action="append",
        metavar="SEED",
    )
    parser.add_argument("--optimizer-iterations", type=int)
    parser.add_argument("--train-trajectories", type=int)
    parser.add_argument("--test-trajectories", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--overwrite", action="store_true", default=None)
    parser.add_argument("--dry-run", action="store_true", default=None)
    parser.add_argument("--fail-fast", action="store_true", default=None)
    return parser


def parse_arguments(arguments: Sequence[str] | None = None) -> Namespace:
    """Parse command-line arguments without resolving defaults.

    Returns:
        The parsed argument namespace.
    """
    return create_argument_parser().parse_args(arguments)


def _preset_defaults(preset: str) -> dict[str, object]:
    """Return detached defaults for one canonical preset.

    Returns:
        Mutable values ready for configuration overlays.

    Raises:
        RunnerConfigurationError: If the preset is unknown.
    """
    if preset == "smoke":
        return {
            "preset": preset,
            "num_qubits": [6],
            "target_ids": [TARGET_IDS[0]],
            "noise_ids": list(NOISE_IDS),
            "methods": [KROTOV_METHOD_ID],
            "num_layers": [0],
            "initialization_seeds": [11],
            "optimizer_iterations": 0,
            "train_trajectories": 0,
            "test_trajectories": 2,
            "output_dir": Path("state_preparation_results") / preset,
            "resume": False,
            "overwrite": False,
            "dry_run": False,
            "fail_fast": False,
        }
    if preset not in {"minimum", "full"}:
        msg = f"Unknown preset {preset!r}; expected one of {PRESET_NAMES}."
        raise RunnerConfigurationError(msg)
    return {
        "preset": preset,
        "num_qubits": list(SUPPORTED_QUBIT_COUNTS),
        "target_ids": list(TARGET_IDS),
        "noise_ids": list(MINIMUM_NOISE_IDS if preset == "minimum" else NOISE_IDS),
        "methods": [KROTOV_METHOD_ID],
        "num_layers": [2],
        "initialization_seeds": [11],
        "optimizer_iterations": 100,
        "train_trajectories": 0,
        "test_trajectories": 100,
        "output_dir": Path("state_preparation_results") / preset,
        "resume": False,
        "overwrite": False,
        "dry_run": False,
        "fail_fast": False,
    }


def _exact_int(value: object, name: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    """Validate one exact bounded integer.

    Returns:
        The validated integer.

    Raises:
        RunnerConfigurationError: If the value is not in range.
    """
    if type(value) is not int:
        msg = f"{name} must be an integer."
        raise RunnerConfigurationError(msg)
    result = value
    if result < minimum or (maximum is not None and result > maximum):
        upper = f", {maximum}" if maximum is not None else ""
        msg = f"{name} must lie in [{minimum}{upper}]."
        raise RunnerConfigurationError(msg)
    return result


def _sequence(value: object, name: str) -> list[object]:
    """Normalize one scalar or nonempty JSON/CLI selection sequence.

    Returns:
        A detached list.

    Raises:
        RunnerConfigurationError: If the value is an empty sequence.
    """
    if isinstance(value, (list, tuple)):
        if value:
            return list(value)
        msg = f"{name} must be a nonempty array."
        raise RunnerConfigurationError(msg)
    return [value]


def _ordered_identifiers(value: object, name: str, supported: Sequence[str]) -> tuple[str, ...]:
    """Validate, deduplicate, and canonically order identifiers.

    Returns:
        Selected identifiers in registry order.

    Raises:
        RunnerConfigurationError: If an identifier is unknown.
    """
    values = _sequence(value, name)
    if any(type(item) is not str for item in values):
        msg = f"{name} must contain only strings."
        raise RunnerConfigurationError(msg)
    selected = cast("set[str]", set(values))
    unknown = sorted(selected - set(supported))
    if unknown:
        msg = f"Unknown {name}: {unknown}; supported values are {tuple(supported)}."
        raise RunnerConfigurationError(msg)
    return tuple(identifier for identifier in supported if identifier in selected)


def _ordered_integers(
    value: object,
    name: str,
    *,
    supported: Sequence[int] | None = None,
    even: bool = False,
    maximum: int | None = None,
) -> tuple[int, ...]:
    """Validate, deduplicate, and order integer selections.

    Returns:
        Selected integers in deterministic order.

    Raises:
        RunnerConfigurationError: If an integer is unsupported.
    """
    values = tuple(_exact_int(item, name, maximum=maximum) for item in _sequence(value, name))
    selected = set(values)
    if supported is not None:
        unknown = sorted(selected - set(supported))
        if unknown:
            msg = f"Unsupported {name}: {unknown}; supported values are {tuple(supported)}."
            raise RunnerConfigurationError(msg)
        return tuple(item for item in supported if item in selected)
    if even and any(item % 2 for item in selected):
        msg = f"{name} values must be even because BMPD depth produces two layers."
        raise RunnerConfigurationError(msg)
    return tuple(sorted(selected))


def _json_output_path(value: object) -> Path:
    """Validate a JSON or CLI output path.

    Returns:
        The expanded absolute output path.

    Raises:
        RunnerConfigurationError: If the value is not a path string.
    """
    if isinstance(value, Path):
        path = value
    elif type(value) is str and value:
        path = Path(value)
    else:
        msg = "output_dir must be a nonempty path string."
        raise RunnerConfigurationError(msg)
    return path.expanduser().resolve()


def _boolean(value: object, name: str) -> bool:
    """Validate one Boolean option.

    Returns:
        The validated Boolean.

    Raises:
        RunnerConfigurationError: If the value is not a Boolean.
    """
    if type(value) is not bool:
        msg = f"{name} must be a boolean."
        raise RunnerConfigurationError(msg)
    return value


def resolve_options(namespace: Namespace) -> RunnerOptions:
    """Resolve preset defaults, JSON overrides, and CLI overrides.

    CLI values take precedence over the JSON file, which takes precedence over
    the selected preset.

    Returns:
        Fully validated runner options.

    Raises:
        RunnerConfigurationError: If any resolved option is invalid.
    """
    file_values = load_configuration_file(namespace.config) if namespace.config is not None else {}
    cli_preset = namespace.preset
    file_preset = file_values.get("preset")
    preset_value = cli_preset if cli_preset is not None else file_preset if file_preset is not None else "smoke"
    if type(preset_value) is not str or preset_value not in PRESET_NAMES:
        msg = f"preset must be one of {PRESET_NAMES}."
        raise RunnerConfigurationError(msg)

    resolved = _preset_defaults(preset_value)
    resolved.update({key: value for key, value in file_values.items() if key not in {"format", "preset"}})
    resolved["preset"] = preset_value
    for key in (
        "num_qubits",
        "target_ids",
        "noise_ids",
        "methods",
        "num_layers",
        "initialization_seeds",
        "optimizer_iterations",
        "train_trajectories",
        "test_trajectories",
        "output_dir",
        "resume",
        "overwrite",
        "dry_run",
        "fail_fast",
    ):
        cli_value = getattr(namespace, key)
        if cli_value is not None:
            resolved[key] = cli_value
    if namespace.resume and not namespace.overwrite:
        resolved["overwrite"] = False
    if namespace.overwrite and not namespace.resume:
        resolved["resume"] = False

    resume = _boolean(resolved["resume"], "resume")
    overwrite = _boolean(resolved["overwrite"], "overwrite")
    if resume and overwrite:
        msg = "resume and overwrite are mutually exclusive."
        raise RunnerConfigurationError(msg)
    train_trajectories = _exact_int(resolved["train_trajectories"], "train_trajectories")
    if train_trajectories != 0:
        msg = "The current Krotov benchmark performs noiseless training and requires train_trajectories=0."
        raise RunnerConfigurationError(msg)
    test_trajectories = _exact_int(resolved["test_trajectories"], "test_trajectories")
    noise_ids = _ordered_identifiers(resolved["noise_ids"], "noise_ids", NOISE_IDS)
    if any(noise_id != NOISELESS_NOISE_ID for noise_id in noise_ids) and test_trajectories == 0:
        msg = "Noisy benchmark cells require test_trajectories greater than zero."
        raise RunnerConfigurationError(msg)

    return RunnerOptions(
        preset=preset_value,
        num_qubits=_ordered_integers(
            resolved["num_qubits"],
            "num_qubits",
            supported=SUPPORTED_QUBIT_COUNTS,
        ),
        target_ids=_ordered_identifiers(resolved["target_ids"], "target_ids", TARGET_IDS),
        noise_ids=noise_ids,
        methods=_ordered_identifiers(resolved["methods"], "methods", METHOD_IDS),
        num_layers=_ordered_integers(resolved["num_layers"], "num_layers", even=True),
        initialization_seeds=_ordered_integers(
            resolved["initialization_seeds"],
            "initialization_seeds",
            maximum=_UINT64_MODULUS - 1,
        ),
        optimizer_iterations=_exact_int(resolved["optimizer_iterations"], "optimizer_iterations"),
        train_trajectories=train_trajectories,
        test_trajectories=test_trajectories,
        output_dir=_json_output_path(resolved["output_dir"]),
        resume=resume,
        overwrite=overwrite,
        dry_run=_boolean(resolved["dry_run"], "dry_run"),
        fail_fast=_boolean(resolved["fail_fast"], "fail_fast"),
    )


def _optimizer_seed(initialization_seed: int) -> int:
    """Return a deterministic seed distinct from initialization."""
    return (initialization_seed + 1) % _UINT64_MODULUS


def _test_seed(initialization_seed: int) -> int:
    """Return a deterministic test seed distinct from training streams."""
    return (initialization_seed + 2) % _UINT64_MODULUS


def _target_selection(targets: TargetCollection, num_qubits: int, target_id: str) -> TargetSelection:
    """Resolve one exact target fixture selection.

    Returns:
        The target metadata bound to fixture provenance.
    """
    target = targets.load_target(num_qubits, target_id)
    return TargetSelection(
        num_qubits=num_qubits,
        target_id=target_id,
        target_seed=target.seed,
        fixture_format=targets.fixture_format,
        fixture_checksum=targets.fixture_checksum,
    )


def _evaluation_config(options: RunnerOptions, noise_id: str, initialization_seed: int) -> EvaluationConfig:
    """Resolve one test evaluation policy.

    Returns:
        A schema-valid noiseless or noisy evaluation configuration.
    """
    if noise_id == NOISELESS_NOISE_ID:
        return EvaluationConfig(test_trajectories_or_shots=0, test_seed=None)
    confidence_level = 0.95 if options.test_trajectories >= 2 else None
    confidence_method = "normal_clipped" if confidence_level is not None else None
    return EvaluationConfig(
        test_trajectories_or_shots=options.test_trajectories,
        test_seed=_test_seed(initialization_seed),
        confidence_level=confidence_level,
        confidence_interval_method=confidence_method,
    )


def _noise_config(noise_id: str) -> NoiseConfig:
    """Resolve one canonical noise configuration.

    Returns:
        Noise metadata with the canonical TJM step where applicable.
    """
    return NoiseConfig(noise_id, tjm_dt=1.0 if noise_id in STANDARD_NOISE_IDS else None)


def build_benchmark_matrix(
    options: RunnerOptions,
    targets: TargetCollection | None = None,
) -> tuple[BenchmarkConfig, ...]:
    """Expand fully resolved options into deterministic result cells.

    Returns:
        The complete matrix in stable training/fan-out order.

    Raises:
        TypeError: If ``targets`` is not a target collection.
        RunnerConfigurationError: If expansion creates duplicate run IDs.
    """
    collection = load_target_collection() if targets is None else targets
    if not isinstance(collection, TargetCollection):
        msg = f"targets must be a TargetCollection, got {type(collection).__name__}."
        raise TypeError(msg)
    configs: list[BenchmarkConfig] = []
    for method_id in options.methods:
        for num_layers in options.num_layers:
            for initialization_seed in options.initialization_seeds:
                initialization = InitializationConfig(
                    rule="random_normal",
                    seed=initialization_seed,
                    scale=0.1,
                )
                optimizer = OptimizerConfig(
                    optimizer_id=method_id,
                    max_iterations=options.optimizer_iterations,
                    optimizer_seed=_optimizer_seed(initialization_seed),
                    hyperparameters={"step_size": 0.2, "schedule": {"kind": "constant"}},
                    train_trajectories_or_shots=options.train_trajectories,
                    training_seed=None,
                )
                for num_qubits in options.num_qubits:
                    for target_id in options.target_ids:
                        target = _target_selection(collection, num_qubits, target_id)
                        configs.extend(
                            BenchmarkConfig(
                                method_id=method_id,
                                method_version=KROTOV_METHOD_VERSION,
                                target=target,
                                ansatz=AnsatzConfig(num_layers // 2),
                                initialization=initialization,
                                optimizer=optimizer,
                                evaluation=_evaluation_config(options, noise_id, initialization_seed),
                                training_noise=NoiseConfig(NOISELESS_NOISE_ID),
                                test_noise=_noise_config(noise_id),
                            )
                            for noise_id in options.noise_ids
                        )
    run_ids = [config.run_id for config in configs]
    if len(run_ids) != len(set(run_ids)):
        msg = "Resolved benchmark matrix contains duplicate stable run IDs."
        raise RunnerConfigurationError(msg)
    return tuple(configs)


def _method_for_id(method_id: str) -> KrotovStatePreparationMethod:
    """Construct one registered state-preparation method.

    Returns:
        A fresh method adapter.

    Raises:
        RunnerConfigurationError: If no implementation is registered.
    """
    if method_id == KROTOV_METHOD_ID:
        return KrotovStatePreparationMethod()
    msg = f"No state-preparation method is registered for {method_id!r}."
    raise RunnerConfigurationError(msg)


def _training_groups(
    matrix: Sequence[BenchmarkConfig],
) -> tuple[tuple[str, KrotovStatePreparationMethod, tuple[BenchmarkConfig, ...]], ...]:
    """Group evaluation cells by reusable training identity.

    Returns:
        Stable groups carrying their method adapter and fan-out cells.
    """
    grouped: dict[tuple[str, str], list[BenchmarkConfig]] = defaultdict(list)
    methods: dict[str, KrotovStatePreparationMethod] = {}
    for config in matrix:
        if config.method_id not in methods:
            methods[config.method_id] = _method_for_id(config.method_id)
        method = methods[config.method_id]
        training_id = state_preparation_training_id(method, config)
        grouped[config.method_id, training_id].append(config)
    return tuple(
        (training_id, methods[method_id], tuple(configs)) for (method_id, training_id), configs in grouped.items()
    )


def print_resolved_matrix(
    options: RunnerOptions,
    matrix: Sequence[BenchmarkConfig],
    *,
    stream: IO[str] = sys.stdout,
) -> None:
    """Print the complete deterministic matrix before execution."""
    groups = _training_groups(matrix)
    header = {
        "matrix_format": RUNNER_CONFIGURATION_FORMAT,
        "preset": options.preset,
        "output_dir": str(options.output_dir),
        "result_rows": len(matrix),
        "training_jobs": len(groups),
    }
    print(json.dumps(header, sort_keys=True, separators=(",", ":")), file=stream)
    for config in matrix:
        print(config.to_json(), file=stream)


def _write_training_failures(
    *,
    store: BenchmarkReportStore,
    configs: Sequence[BenchmarkConfig],
    error: BaseException,
    phase: str,
    wall_time_seconds: float,
    traceback_text: str,
    fail_fast: bool,
    stream: IO[str],
) -> int:
    """Publish failures for cells blocked by one training failure.

    Returns:
        Number of failure rows written.
    """
    written = 0
    for config in configs:
        store.write_failure(
            config=config,
            failure_phase=phase,
            exception=error,
            wall_time_seconds=wall_time_seconds,
            traceback=traceback_text,
            retryable=True,
            replace=config.run_id in {record.run_id for record in store.records},
        )
        written += 1
        print(f"FAILED {config.run_id} {phase}: {error}", file=stream)
        if fail_fast:
            break
    return written


def execute_benchmark_matrix(
    options: RunnerOptions,
    matrix: Sequence[BenchmarkConfig],
    *,
    targets: TargetCollection | None = None,
    repository_root: Path | None = None,
    stream: IO[str] = sys.stdout,
) -> RunSummary:
    """Execute one matrix sequentially with train-once evaluation fan-out.

    Returns:
        Aggregate execution counts.

    Raises:
        TypeError: If ``repository_root`` is not a path.
        RunnerConfigurationError: If existing output requires resume or overwrite.
    """
    collection = load_target_collection() if targets is None else targets
    root = Path(__file__).parents[2] if repository_root is None else repository_root
    if not isinstance(root, Path):
        msg = f"repository_root must be a pathlib.Path, got {type(root).__name__}."
        raise TypeError(msg)
    canonical_path = options.output_dir / RESULTS_JSONL_NAME
    if canonical_path.exists() and not options.resume and not options.overwrite:
        msg = f"{canonical_path} already exists; pass --resume or --overwrite explicitly."
        raise RunnerConfigurationError(msg)
    provenance = capture_run_provenance(root, excluded_paths=(options.output_dir,))
    store = BenchmarkReportStore(
        options.output_dir,
        provenance,
        overwrite=options.overwrite,
    )
    completed = store.completed_run_ids if options.resume else frozenset()
    skipped = sum(config.run_id in completed for config in matrix)
    attempted = 0
    succeeded = 0
    failed = 0

    for training_id, method, configs in _training_groups(matrix):
        pending = tuple(config for config in configs if config.run_id not in completed)
        if not pending:
            print(f"SKIPPED training {training_id}: all {len(configs)} rows complete", file=stream)
            continue
        print(f"TRAINING {training_id} for {len(pending)} pending rows", file=stream)
        optimization_started = time.perf_counter()
        try:
            artifact = train_state_preparation_method(method, pending[0], collection)
        except Exception as error:  # noqa: BLE001 -- benchmark adapters may surface arbitrary domain failures
            optimization_time = time.perf_counter() - optimization_started
            traceback_text = traceback_module.format_exc()
            if isinstance(error, StatePreparationTrainingError):
                phase = error.failure_phase
                reported_error: BaseException = error.exception
            else:
                phase = "optimization"
                reported_error = error
            written = _write_training_failures(
                store=store,
                configs=pending,
                error=reported_error,
                phase=phase,
                wall_time_seconds=optimization_time,
                traceback_text=traceback_text,
                fail_fast=options.fail_fast,
                stream=stream,
            )
            attempted += written
            failed += written
            if options.fail_fast:
                break
            continue
        optimization_time = time.perf_counter() - optimization_started

        for config in pending:
            attempted += 1
            evaluation_started = time.perf_counter()
            try:
                evaluation = evaluate_state_preparation_artifact(method, artifact, config, collection)
            except Exception as error:  # noqa: BLE001 -- simulators may surface arbitrary domain failures
                evaluation_time = time.perf_counter() - evaluation_started
                traceback_text = traceback_module.format_exc()
                store.write_failure(
                    config=config,
                    artifact=artifact,
                    failure_phase="evaluation",
                    exception=error,
                    wall_time_seconds=optimization_time + evaluation_time,
                    traceback=traceback_text,
                    retryable=True,
                    replace=config.run_id in {record.run_id for record in store.records},
                )
                failed += 1
                print(f"FAILED {config.run_id} evaluation: {error}", file=stream)
                if options.fail_fast:
                    break
                continue
            evaluation_time = time.perf_counter() - evaluation_started
            try:
                store.write_success(
                    config=config,
                    artifact=artifact,
                    evaluation=evaluation,
                    optimization_wall_time_seconds=optimization_time,
                    evaluation_wall_time_seconds=evaluation_time,
                    replace=config.run_id in {record.run_id for record in store.records},
                )
            except Exception as error:
                traceback_text = traceback_module.format_exc()
                if store.is_completed(config):
                    raise
                store.write_failure(
                    config=config,
                    failure_phase="serialization",
                    exception=error,
                    wall_time_seconds=optimization_time + evaluation_time,
                    traceback=traceback_text,
                    retryable=True,
                    artifact=artifact,
                    replace=config.run_id in {record.run_id for record in store.records},
                )
                failed += 1
                print(f"FAILED {config.run_id} serialization: {error}", file=stream)
                if options.fail_fast:
                    break
            else:
                succeeded += 1
                print(f"COMPLETED {config.run_id}", file=stream)
        if options.fail_fast and failed:
            break

    summary = RunSummary(
        planned=len(matrix),
        attempted=attempted,
        succeeded=succeeded,
        failed=failed,
        skipped=skipped,
    )
    print(
        "SUMMARY "
        f"planned={summary.planned} attempted={summary.attempted} "
        f"succeeded={summary.succeeded} failed={summary.failed} skipped={summary.skipped}",
        file=stream,
    )
    return summary


def run(
    arguments: Sequence[str] | None = None,
    *,
    stdout: IO[str] = sys.stdout,
    stderr: IO[str] = sys.stderr,
) -> int:
    """Resolve and execute the CLI.

    Returns:
        Zero on success, one for recorded benchmark failures, or two for
        invalid configuration.
    """
    try:
        return _run_resolved(arguments, stdout=stdout)
    except (OSError, ReportingError, TypeError, ValueError) as error:
        print(f"error: {error}", file=stderr)
        return 2


def _run_resolved(arguments: Sequence[str] | None, *, stdout: IO[str]) -> int:
    """Resolve and execute valid CLI input.

    Returns:
        The process exit code.
    """
    options = resolve_options(parse_arguments(arguments))
    targets = load_target_collection()
    matrix = build_benchmark_matrix(options, targets)
    print_resolved_matrix(options, matrix, stream=stdout)
    if options.dry_run:
        print("DRY RUN: no output files were created.", file=stdout)
        return 0
    summary = execute_benchmark_matrix(options, matrix, targets=targets, stream=stdout)
    return 1 if summary.failed else 0


def main() -> None:
    """Run the module entry point.

    Raises:
        SystemExit: Always, with the CLI result code.
    """
    raise SystemExit(run())


if __name__ == "__main__":
    main()


__all__ = [
    "METHOD_IDS",
    "MINIMUM_NOISE_IDS",
    "PRESET_NAMES",
    "RUNNER_CONFIGURATION_FORMAT",
    "RunSummary",
    "RunnerConfigurationError",
    "RunnerOptions",
    "build_benchmark_matrix",
    "create_argument_parser",
    "execute_benchmark_matrix",
    "load_configuration_file",
    "main",
    "parse_arguments",
    "print_resolved_matrix",
    "resolve_options",
    "run",
]
