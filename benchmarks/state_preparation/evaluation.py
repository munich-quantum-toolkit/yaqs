# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent test evaluation for state-preparation benchmarks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from statistics import NormalDist
from typing import TYPE_CHECKING, TypeVar, cast

import numpy as np

from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import (
    GateNoiseProvider,
    KrotovNoiseMap,
    KrotovTJMOptions,
    KrotovTruncation,
    ParameterizedCircuit,
    forward_tjm_trajectory,
)

from .ballarin import (
    BallarinCircuitMaterialization,
    create_ballarin_noise_provider,
    materialize_ballarin_circuit,
)
from .circuits import compile_quantinuum_native
from .constants import BALLARIN_NOISE_ID, NOISELESS_NOISE_ID, STANDARD_NOISE_IDS
from .methods import (
    StatePreparationMethod,
    StatePreparationTrainingArtifact,
    state_preparation_training_id,
)
from .noise import create_standard_noise_provider
from .schema import BenchmarkConfig, CircuitStatistics
from .statistics import collect_circuit_statistics
from .targets import TargetCollection, TargetRecord

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.core.methods.decompositions import TruncMode

_ResultT = TypeVar("_ResultT")
_EMPTY_INPUT = np.array([], dtype=np.float64)


class SeedDomain(IntEnum):
    """Stable tags separating all benchmark random-number domains."""

    PARAMETER_INITIALIZATION = 0x50415241
    OPTIMIZER_ORDERING = 0x4F505449
    TRAINING_TRAJECTORY = 0x54524149
    TEST_TRAJECTORY = 0x54455354
    REPEATED_TEST_EVALUATION = 0x52455045


def derive_seed_sequence(
    resolved_seed: int,
    domain: SeedDomain,
    *,
    repetition: int = 0,
    sample_index: int = 0,
) -> np.random.SeedSequence:
    """Derive one stable, domain-separated NumPy seed sequence.

    Args:
        resolved_seed: Fully resolved unsigned 64-bit benchmark seed.
        domain: Randomness domain being derived.
        repetition: Independent repeated-evaluation index.
        sample_index: Index within the selected domain and repetition.

    Returns:
        A new deterministic :class:`numpy.random.SeedSequence`.

    Raises:
        TypeError: If an integer argument has the wrong type.
        ValueError: If an integer argument is outside its supported range.
    """
    for name, value, maximum in (
        ("resolved_seed", resolved_seed, 2**64 - 1),
        ("repetition", repetition, 2**32 - 1),
        ("sample_index", sample_index, 2**64 - 1),
    ):
        if type(value) is not int:
            msg = f"{name} must be an int, got {type(value).__name__}."
            raise TypeError(msg)
        if not 0 <= value <= maximum:
            msg = f"{name} must lie in [0, {maximum}]."
            raise ValueError(msg)
    if not isinstance(domain, SeedDomain):
        msg = f"domain must be a SeedDomain, got {type(domain).__name__}."
        raise TypeError(msg)

    return np.random.SeedSequence([
        resolved_seed & 0xFFFFFFFF,
        resolved_seed >> 32,
        int(domain),
        repetition,
        sample_index & 0xFFFFFFFF,
        sample_index >> 32,
    ])


@dataclass(frozen=True, slots=True)
class IndependentEvaluation:
    """Complete scientific output of one independent test evaluation."""

    training_id: str
    run_id: str
    repetition: int
    train_fidelity: float
    logical_test_noiseless_fidelity: float
    test_noiseless_fidelity: float
    test_noisy_fidelity: float
    circuit_statistics: CircuitStatistics
    native_pre_pruning_noiseless_fidelity: float | None = None
    noisy_fidelity_standard_deviation: float | None = None
    noisy_fidelity_standard_error: float | None = None
    confidence_interval_lower: float | None = None
    confidence_interval_upper: float | None = None
    sampled_nonidentity_events: int = 0
    trajectory_fidelities: tuple[float, ...] | None = None


def _evaluation_truncation(config: BenchmarkConfig) -> KrotovTruncation:
    """Translate benchmark evaluation truncation into Krotov settings.

    Returns:
        The complete evaluation truncation policy.
    """
    evaluation = config.evaluation
    return KrotovTruncation(
        max_bond_dim=evaluation.max_bond_dimension,
        svd_threshold=evaluation.svd_threshold,
        trunc_mode=cast("TruncMode", evaluation.truncation_mode),
        min_bond_dim=evaluation.min_bond_dimension,
    )


def _load_target(targets: TargetCollection, config: BenchmarkConfig) -> TargetRecord:
    """Load the exact fixture target selected by a benchmark configuration.

    Returns:
        The selected immutable target record.

    Raises:
        TypeError: If ``targets`` is not a target collection.
        ValueError: If target provenance differs from the configuration.
    """
    if not isinstance(targets, TargetCollection):
        msg = f"targets must be a TargetCollection, got {type(targets).__name__}."
        raise TypeError(msg)
    selection = config.target
    if targets.fixture_format != selection.fixture_format or targets.fixture_checksum != selection.fixture_checksum:
        msg = "TargetCollection provenance does not match the benchmark target selection."
        raise ValueError(msg)
    target = targets.load_target(selection.num_qubits, selection.target_id)
    if target.seed != selection.target_seed:
        msg = "TargetRecord seed does not match the benchmark target selection."
        raise ValueError(msg)
    return target


def _validate_inputs(
    method: StatePreparationMethod[_ResultT],
    artifact: StatePreparationTrainingArtifact,
    config: BenchmarkConfig,
) -> None:
    """Ensure the artifact is the one reusable training result for the config.

    Raises:
        TypeError: If an input has the wrong type.
        ValueError: If method or training identities do not match.
    """
    if not isinstance(method, StatePreparationMethod):
        msg = f"method must implement StatePreparationMethod, got {type(method).__name__}."
        raise TypeError(msg)
    if not isinstance(artifact, StatePreparationTrainingArtifact):
        msg = f"artifact must be a StatePreparationTrainingArtifact, got {type(artifact).__name__}."
        raise TypeError(msg)
    if not isinstance(config, BenchmarkConfig):
        msg = f"config must be a BenchmarkConfig, got {type(config).__name__}."
        raise TypeError(msg)
    expected_identity = (method.method_id, method.method_name, method.method_version)
    artifact_identity = (artifact.method_id, artifact.method_name, artifact.method_version)
    if artifact_identity != expected_identity:
        msg = "Training artifact method identity does not match the evaluation adapter."
        raise ValueError(msg)
    if artifact.training_id != state_preparation_training_id(method, config):
        msg = "Training artifact does not match the configuration's reusable training identity."
        raise ValueError(msg)


def _fidelity(target: MPS, state: MPS) -> float:
    """Return a numerically bounded pure-state fidelity.

    Returns:
        The fidelity clipped only within numerical roundoff.

    Raises:
        ValueError: If the computed fidelity is non-finite or nonphysical.
    """
    value = float(abs(target.scalar_product(state)) ** 2)
    if not math.isfinite(value) or value < -1e-12 or value > 1.0 + 1e-12:
        msg = f"Trajectory fidelity must lie in [0, 1], got {value!r}."
        raise ValueError(msg)
    return min(1.0, max(0.0, value))


def _test_trajectory_seed_sequence(
    resolved_seed: int,
    *,
    repetition: int,
    sample_index: int,
) -> np.random.SeedSequence:
    """Derive a test trajectory beneath an independent repetition stream.

    Returns:
        A seed sequence tagged with both repeated-evaluation and test-trajectory
        domains.
    """
    repetition_root = derive_seed_sequence(
        resolved_seed,
        SeedDomain.REPEATED_TEST_EVALUATION,
        repetition=repetition,
    )
    root_state = repetition_root.generate_state(4)
    return np.random.SeedSequence([
        *(int(word) for word in root_state),
        int(SeedDomain.TEST_TRAJECTORY),
        sample_index & 0xFFFFFFFF,
        sample_index >> 32,
    ])


def count_nonidentity_events(noise_maps: list[KrotovNoiseMap]) -> int:
    """Count sampled non-identity channel events in one trajectory.

    Returns:
        The number of sampled non-identity events.
    """
    count = 0
    for noise_map in noise_maps:
        if noise_map.channel_id == BALLARIN_NOISE_ID:
            count += sum(label != "I" for label in noise_map.outcome_labels)
        elif noise_map.is_identity is False or (
            noise_map.jump_process_index is not None and noise_map.is_identity is None
        ):
            count += 1
    return count


def _evaluate_trajectories(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: TargetRecord,
    config: BenchmarkConfig,
    provider: GateNoiseProvider,
    *,
    repetition: int,
) -> tuple[list[float], int]:
    """Run the exact configured trajectory budget with fresh per-trajectory RNGs.

    Returns:
        Trajectory fidelities and the total non-identity event count.
    """
    count = config.evaluation.test_trajectories_or_shots
    base_seed = cast("int", config.evaluation.test_seed)
    dt = config.test_noise.tjm_dt if config.test_noise.tjm_dt is not None else 1.0
    options = KrotovTJMOptions(num_trajectories=1, random_seed=base_seed, dt=dt)
    truncation = _evaluation_truncation(config)
    target_mps = MPS.from_statevector(target.state_vector_copy())
    fidelities: list[float] = []
    nonidentity_events = 0

    for sample_index in range(count):
        seed_sequence = _test_trajectory_seed_sequence(
            base_seed,
            repetition=repetition,
            sample_index=sample_index,
        )
        rng = np.random.Generator(np.random.PCG64(seed_sequence))
        trajectory = forward_tjm_trajectory(
            circuit,
            theta,
            _EMPTY_INPUT,
            MPS(circuit.num_qubits),
            truncation,
            None,
            options,
            rng,
            noise_provider=provider,
        )
        fidelities.append(_fidelity(target_mps, trajectory.states[-1]))
        nonidentity_events += count_nonidentity_events(trajectory.noise_maps)
    return fidelities, nonidentity_events


def _uncertainty(
    fidelities: list[float],
    config: BenchmarkConfig,
) -> tuple[float, float | None, float | None, float | None, float | None]:
    """Aggregate trajectory fidelities and the configured uncertainty estimate.

    Returns:
        Mean, sample standard deviation, standard error, and optional interval
        bounds.
    """
    mean = float(np.mean(fidelities))
    if len(fidelities) < 2:
        return mean, None, None, None, None
    standard_deviation = float(np.std(fidelities, ddof=1))
    standard_error = standard_deviation / math.sqrt(len(fidelities))
    if config.evaluation.confidence_level is None:
        return mean, standard_deviation, standard_error, None, None
    level = config.evaluation.confidence_level
    z_score = NormalDist().inv_cdf((1.0 + level) / 2.0)
    lower = max(0.0, mean - z_score * standard_error)
    upper = min(1.0, mean + z_score * standard_error)
    return mean, standard_deviation, standard_error, lower, upper


def _materialize_evaluation_circuit(
    logical_circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    config: BenchmarkConfig,
) -> tuple[
    ParameterizedCircuit,
    NDArray[np.float64],
    BallarinCircuitMaterialization | None,
]:
    """Return the representation required by the selected test noise."""
    if config.test_noise.noise_id != BALLARIN_NOISE_ID:
        return logical_circuit, theta, None
    compilation = compile_quantinuum_native(logical_circuit)
    materialization = materialize_ballarin_circuit(compilation, theta)
    return (
        materialization.to_parameterized_circuit(),
        np.zeros(0, dtype=np.float64),
        materialization,
    )


def evaluate_state_preparation_artifact(
    method: StatePreparationMethod[_ResultT],
    artifact: StatePreparationTrainingArtifact,
    config: BenchmarkConfig,
    targets: TargetCollection,
    *,
    repetition: int = 0,
) -> IndependentEvaluation:
    """Evaluate a trained artifact with independent noiseless and noisy tests.

    The API intentionally accepts no training trajectory maps. Every noisy
    trajectory receives a fresh RNG derived solely from the resolved test seed,
    repetition index, and trajectory index.

    Returns:
        Complete evaluation metrics ready for result reporting.

    Raises:
        TypeError: If an input or repetition has the wrong type.
        ValueError: If identities, target provenance, repetition, or metrics are
            invalid.
    """
    _validate_inputs(method, artifact, config)
    if type(repetition) is not int:
        msg = f"repetition must be an int, got {type(repetition).__name__}."
        raise TypeError(msg)
    if not 0 <= repetition <= 2**32 - 1:
        msg = f"repetition must lie in [0, {2**32 - 1}]."
        raise ValueError(msg)

    target = _load_target(targets, config)
    logical_circuit = artifact.circuit
    theta = artifact.parameters_copy()
    logical_noiseless = method.evaluate_noiseless(
        logical_circuit,
        theta,
        target,
        evaluation=config.evaluation,
    )

    evaluated_circuit, evaluated_theta, materialization = _materialize_evaluation_circuit(
        logical_circuit,
        theta,
        config,
    )
    test_noiseless = method.evaluate_noiseless(
        evaluated_circuit,
        evaluated_theta,
        target,
        evaluation=config.evaluation,
    )
    pre_pruning_fidelity: float | None = None
    if materialization is not None:
        compilation = compile_quantinuum_native(logical_circuit)
        pre_pruning_fidelity = method.evaluate_noiseless(
            compilation.circuit,
            theta,
            target,
            evaluation=config.evaluation,
        )

    statistics = collect_circuit_statistics(
        logical_circuit,
        config.ansatz,
        native_source=materialization,
        evaluated_representation="native" if materialization is not None else "logical",
    )
    if config.test_noise.noise_id == NOISELESS_NOISE_ID:
        return IndependentEvaluation(
            training_id=artifact.training_id,
            run_id=config.run_id,
            repetition=repetition,
            train_fidelity=artifact.training_fidelity,
            logical_test_noiseless_fidelity=logical_noiseless,
            test_noiseless_fidelity=test_noiseless,
            test_noisy_fidelity=test_noiseless,
            native_pre_pruning_noiseless_fidelity=pre_pruning_fidelity,
            circuit_statistics=statistics,
        )

    provider: GateNoiseProvider
    if config.test_noise.noise_id in STANDARD_NOISE_IDS:
        provider = create_standard_noise_provider(config.test_noise.noise_id)
    else:
        provider = create_ballarin_noise_provider()
    fidelities, nonidentity_events = _evaluate_trajectories(
        evaluated_circuit,
        evaluated_theta,
        target,
        config,
        provider,
        repetition=repetition,
    )
    mean, standard_deviation, standard_error, lower, upper = _uncertainty(fidelities, config)
    return IndependentEvaluation(
        training_id=artifact.training_id,
        run_id=config.run_id,
        repetition=repetition,
        train_fidelity=artifact.training_fidelity,
        logical_test_noiseless_fidelity=logical_noiseless,
        test_noiseless_fidelity=test_noiseless,
        test_noisy_fidelity=mean,
        native_pre_pruning_noiseless_fidelity=pre_pruning_fidelity,
        noisy_fidelity_standard_deviation=standard_deviation,
        noisy_fidelity_standard_error=standard_error,
        confidence_interval_lower=lower,
        confidence_interval_upper=upper,
        sampled_nonidentity_events=nonidentity_events,
        trajectory_fidelities=tuple(fidelities) if config.evaluation.store_trajectory_sidecar else None,
        circuit_statistics=statistics,
    )


__all__ = [
    "IndependentEvaluation",
    "SeedDomain",
    "count_nonidentity_events",
    "derive_seed_sequence",
    "evaluate_state_preparation_artifact",
]
