# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for state-preparation method adapters."""

from __future__ import annotations

import io
import zipfile
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from benchmarks.state_preparation import (
    KROTOV_CHECKPOINT_FORMAT,
    KROTOV_METHOD_ID,
    KROTOV_METHOD_NAME,
    KROTOV_METHOD_VERSION,
    TRAINING_ID_PREFIX,
    AnsatzConfig,
    BenchmarkConfig,
    BenchmarkFailure,
    EvaluationConfig,
    InitializationConfig,
    KrotovStatePreparationMethod,
    NoiseConfig,
    OptimizerConfig,
    StatePreparationMethod,
    StatePreparationTarget,
    StatePreparationTrainingArtifact,
    StatePreparationTrainingError,
    TargetCollection,
    TargetRecord,
    TargetSelection,
    checkpoint_checksum,
    load_target_collection,
    state_preparation_training_id,
    state_preparation_training_identity,
    train_state_preparation_method,
)
from mqt.yaqs.optimization import KrotovResult, ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

SHA_A = f"sha256:{'a' * 64}"
GIT_COMMIT = "1" * 40


def _optimizer(
    *,
    optimizer_id: str = "krotov",
    max_iterations: int = 0,
    optimizer_seed: int = 17,
    hyperparameters: Mapping[str, object] | None = None,
    train_trajectories_or_shots: int = 0,
    training_seed: int | None = None,
    max_bond_dimension: int | None = None,
    svd_threshold: float = 0.0,
    truncation_mode: str = "discarded_weight",
    min_bond_dimension: int = 1,
) -> OptimizerConfig:
    """Return a compact valid Krotov optimizer configuration."""
    resolved_hyperparameters = (
        {
            "step_size": 0.2,
            "schedule": {"kind": "constant"},
        }
        if hyperparameters is None
        else hyperparameters
    )
    return OptimizerConfig(
        optimizer_id=optimizer_id,
        max_iterations=max_iterations,
        optimizer_seed=optimizer_seed,
        hyperparameters=resolved_hyperparameters,
        train_trajectories_or_shots=train_trajectories_or_shots,
        training_seed=training_seed,
        max_bond_dimension=max_bond_dimension,
        svd_threshold=svd_threshold,
        truncation_mode=truncation_mode,
        min_bond_dimension=min_bond_dimension,
    )


def _config() -> BenchmarkConfig:
    """Return a valid noiseless benchmark configuration."""
    return BenchmarkConfig(
        method_id=KROTOV_METHOD_ID,
        method_version=KROTOV_METHOD_VERSION,
        target=TargetSelection(
            num_qubits=6,
            target_id="gaussian_mu0p5_sigma0p1",
            target_seed=None,
            fixture_format="yaqs.state_preparation_targets.v1",
            fixture_checksum=SHA_A,
        ),
        ansatz=AnsatzConfig(1, initial_single_qubit_layer=True),
        initialization=InitializationConfig(rule="random_normal", seed=11, scale=0.1),
        optimizer=_optimizer(),
        evaluation=EvaluationConfig(test_trajectories_or_shots=0, test_seed=None),
        training_noise=NoiseConfig("noiseless"),
        test_noise=NoiseConfig("noiseless"),
    )


def _fixture_training_problem(
    *,
    method_id: str = KROTOV_METHOD_ID,
    method_version: str = KROTOV_METHOD_VERSION,
) -> tuple[BenchmarkConfig, TargetCollection, TargetRecord]:
    """Return a config and matching validated production target record."""
    collection = load_target_collection()
    target = collection.load_target(6, "gaussian_mu0p5_sigma0p1")
    config = replace(
        _config(),
        method_id=method_id,
        method_version=method_version,
        target=TargetSelection(
            num_qubits=target.num_qubits,
            target_id=target.target_id,
            target_seed=target.seed,
            fixture_format=collection.fixture_format,
            fixture_checksum=collection.fixture_checksum,
        ),
    )
    return config, collection, target


def _single_qubit_problem() -> tuple[ParameterizedCircuit, np.ndarray]:
    """Return a small trainable circuit and reachable target vector."""
    circuit = ParameterizedCircuit(
        1,
        [
            ParameterizedGate("rz", (0,), param_index=0),
            ParameterizedGate("ry", (0,), param_index=1),
            ParameterizedGate("rz", (0,), param_index=2),
        ],
    )
    target = np.array([np.cos(0.3), np.sin(0.3)], dtype=np.complex128)
    return circuit, target


def _zero_iteration_result(
    method: KrotovStatePreparationMethod,
    circuit: ParameterizedCircuit,
    target: np.ndarray,
) -> KrotovResult:
    """Run a deterministic zero-iteration optimization.

    Returns:
        The zero-iteration Krotov result.
    """
    return method.optimize_noiseless(
        circuit,
        target,
        np.zeros(circuit.num_params),
        _optimizer(),
    )


class _FakeStatePreparationMethod:
    """Minimal structurally compatible second method used by contract tests."""

    method_id = "fake"
    method_name = "Fake"
    method_version = "1"
    calls = 0

    def _touch(self) -> None:
        """Record access through the instance contract."""
        self.calls += 1

    def build_ansatz(self, num_qubits: int, ansatz: AnsatzConfig) -> ParameterizedCircuit:
        """Return an empty circuit."""
        self._touch()
        del ansatz
        return ParameterizedCircuit(num_qubits, [])

    def initialize_parameters(
        self,
        circuit: ParameterizedCircuit,
        initialization: InitializationConfig,
        *,
        checkpoint_root: Path | None = None,
    ) -> np.ndarray:
        """Return an empty parameter vector."""
        self._touch()
        del initialization, checkpoint_root
        return np.zeros(circuit.num_params)

    def optimize_noiseless(
        self,
        circuit: ParameterizedCircuit,
        target: StatePreparationTarget,
        initial_parameters: np.ndarray,
        optimizer: OptimizerConfig,
    ) -> object:
        """Return the supplied parameters as an opaque fake result."""
        self._touch()
        del circuit, target, optimizer
        return initial_parameters.copy()

    def extract_final_parameters(self, result: object) -> np.ndarray:
        """Extract the fake parameter vector.

        Returns:
            A detached numeric vector.
        """
        self._touch()
        return np.asarray(result, dtype=np.float64).copy()

    def extract_training_fidelity(self, result: object) -> float:
        """Return a fake training fidelity."""
        self._touch()
        del result
        return 1.0

    def evaluate_noiseless(
        self,
        circuit: ParameterizedCircuit,
        parameters: np.ndarray,
        target: StatePreparationTarget,
        *,
        evaluation: EvaluationConfig | None = None,
    ) -> float:
        """Return a fake evaluation fidelity."""
        self._touch()
        del circuit, parameters, target, evaluation
        return 1.0

    def optimizer_metadata(self, optimizer: OptimizerConfig) -> dict[str, object]:
        """Return detached fake metadata."""
        self._touch()
        return optimizer.to_dict()

    def serialize_checkpoint(self, circuit: ParameterizedCircuit, result: object) -> bytes:
        """Return a fake checkpoint."""
        self._touch()
        del circuit, result
        return b"fake"

    def deserialize_checkpoint(
        self,
        circuit: ParameterizedCircuit,
        payload: bytes,
        *,
        expected_checksum: str | None = None,
    ) -> np.ndarray:
        """Return fake decoded parameters."""
        self._touch()
        del circuit, payload, expected_checksum
        return np.array([], dtype=np.float64)


def test_method_protocol_accepts_a_second_structural_adapter() -> None:
    """Generic training and fan-out exercise a second method without branches."""
    fake = _FakeStatePreparationMethod()
    typed_fake: StatePreparationMethod[object] = fake
    assert isinstance(fake, StatePreparationMethod)
    assert not isinstance(object(), StatePreparationMethod)

    config, collection, target = _fixture_training_problem(
        method_id=fake.method_id,
        method_version=fake.method_version,
    )
    artifact = train_state_preparation_method(typed_fake, config, collection)
    assert isinstance(artifact, StatePreparationTrainingArtifact)
    assert artifact.method_id == fake.method_id
    assert not artifact.parameters.flags.writeable

    before = artifact.parameters.tobytes()
    first = typed_fake.evaluate_noiseless(
        artifact.circuit,
        artifact.parameters,
        target,
        evaluation=config.evaluation,
    )
    second = typed_fake.evaluate_noiseless(
        artifact.circuit,
        artifact.parameters,
        target,
        evaluation=replace(config.evaluation),
    )
    assert first == second == pytest.approx(artifact.training_fidelity)
    assert artifact.training_fidelity == pytest.approx(1.0)
    assert artifact.parameters.tobytes() == before
    assert artifact.checkpoint_checksum == checkpoint_checksum(b"fake")
    assert fake.calls == 10


def test_training_artifacts_are_factory_only() -> None:
    """Public artifacts cannot claim arbitrary unverified checkpoint bytes."""
    circuit = ParameterizedCircuit(1, [])
    with pytest.raises(ValueError, match="train_state_preparation_method"):
        StatePreparationTrainingArtifact(
            training_id=f"{TRAINING_ID_PREFIX}{'a' * 64}",
            method_id="fake",
            method_name="Fake",
            method_version="1",
            circuit=circuit,
            parameters=np.array([], dtype=np.float64),
            training_fidelity=1.0,
            optimizer_metadata={},
            checkpoint_payload=b"not a checkpoint",
        )


def test_training_boundary_binds_target_provenance_and_failure_phases(tmp_path: Path) -> None:
    """The generic boundary rejects wrong fixtures and classifies stage errors."""
    collection = load_target_collection()
    method = KrotovStatePreparationMethod()

    with pytest.raises(StatePreparationTrainingError) as provenance_error:
        train_state_preparation_method(method, _config(), collection)
    assert provenance_error.value.failure_phase == "target_loading"
    assert isinstance(provenance_error.value.exception, ValueError)

    config, matching_collection, _target = _fixture_training_problem()
    missing_warm_start = replace(
        config,
        initialization=InitializationConfig(
            "warm_start",
            warm_start_path="missing.npy",
            warm_start_checksum=SHA_A,
        ),
    )
    with pytest.raises(StatePreparationTrainingError) as initialization_error:
        train_state_preparation_method(
            method,
            missing_warm_start,
            matching_collection,
            checkpoint_root=tmp_path,
        )
    assert initialization_error.value.failure_phase == "initialization"
    assert isinstance(initialization_error.value.exception, ValueError)
    assert initialization_error.value.__cause__ is initialization_error.value.exception


def test_training_boundary_rejects_mid_run_method_identity_changes() -> None:
    """An adapter cannot change the identity encoded by its training ID."""

    class MutatingFakeMethod(_FakeStatePreparationMethod):
        """Fake adapter that violates identity stability during initialization."""

        def initialize_parameters(
            self,
            circuit: ParameterizedCircuit,
            initialization: InitializationConfig,
            *,
            checkpoint_root: Path | None = None,
        ) -> np.ndarray:
            """Mutate the version after producing valid initial parameters.

            Returns:
                The otherwise valid fake parameter vector.
            """
            parameters = super().initialize_parameters(
                circuit,
                initialization,
                checkpoint_root=checkpoint_root,
            )
            self.method_version = "2"
            return parameters

    method = MutatingFakeMethod()
    config, collection, _target = _fixture_training_problem(
        method_id=method.method_id,
        method_version=method.method_version,
    )
    with pytest.raises(StatePreparationTrainingError) as error:
        train_state_preparation_method(method, config, collection)
    assert error.value.failure_phase == "initialization"
    assert "identity changed" in str(error.value.exception)


def test_krotov_identity_and_ansatz_construction() -> None:
    """The adapter exposes frozen identity and the exact shared BMPD ansatz."""
    method = KrotovStatePreparationMethod()
    assert method.method_id == KROTOV_METHOD_ID == "krotov"
    assert method.method_name == KROTOV_METHOD_NAME == "Krotov"
    assert method.method_version == KROTOV_METHOD_VERSION == "1"
    assert isinstance(method, StatePreparationMethod)

    no_product = method.build_ansatz(4, AnsatzConfig(1, initial_single_qubit_layer=False))
    with_product = method.build_ansatz(4, AnsatzConfig(1, initial_single_qubit_layer=True))
    zero_depth = method.build_ansatz(4, AnsatzConfig(0, initial_single_qubit_layer=False))
    assert no_product.num_qubits == 4
    assert no_product.num_params == 27
    assert with_product.num_params == 39
    assert zero_depth.num_params == 0
    assert zero_depth.gates == []


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ("random_uniform", [0.09117593162407173, -0.22308949059888866, -0.1398200636136943]),
        ("random_normal", [-0.24728033758696272, -0.0919466628669708, 0.32198131532231217]),
    ],
)
def test_random_initialization_is_local_and_deterministic(
    rule: str,
    expected: list[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Random initialization depends only on its dedicated generator seed."""
    method = KrotovStatePreparationMethod()
    circuit, _target = _single_qubit_problem()
    config = InitializationConfig(rule=rule, seed=123, scale=0.25)

    def reject_global_random(*args: object, **kwargs: object) -> None:
        """Fail if initialization calls a legacy global random function."""
        del args, kwargs
        pytest.fail("Initialization used NumPy's global random state.")

    monkeypatch.setattr(np.random, "uniform", reject_global_random)
    monkeypatch.setattr(np.random, "normal", reject_global_random)
    first = method.initialize_parameters(circuit, config)
    second = method.initialize_parameters(circuit, config)
    changed = method.initialize_parameters(circuit, replace(config, seed=124))

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, np.asarray(expected))
    assert not np.array_equal(first, changed)
    assert first.dtype == np.float64
    if rule == "random_uniform":
        assert np.all(first >= -0.25)
        assert np.all(first < 0.25)


def test_zero_initialization_handles_an_empty_parameter_vector() -> None:
    """Zero initialization is exact for ordinary and zero-depth circuits."""
    method = KrotovStatePreparationMethod()
    ordinary, _target = _single_qubit_problem()
    empty = ParameterizedCircuit(2, [])
    np.testing.assert_array_equal(
        method.initialize_parameters(ordinary, InitializationConfig("zeros")),
        np.zeros(3),
    )
    assert method.initialize_parameters(empty, InitializationConfig("zeros")).shape == (0,)


def test_zero_iteration_optimization_and_extraction_are_defensive() -> None:
    """A zero budget records only the initial point and leaves inputs untouched."""
    method = KrotovStatePreparationMethod()
    circuit, target = _single_qubit_problem()
    initial = np.array([0.1, -0.2, 0.3])
    snapshot = initial.copy()

    result = method.optimize_noiseless(circuit, target, initial, _optimizer())
    extracted = method.extract_final_parameters(result)
    fidelity = method.extract_training_fidelity(result)

    np.testing.assert_array_equal(initial, snapshot)
    np.testing.assert_array_equal(extracted, snapshot)
    assert result.trace["step"] == [0]
    assert result.trace["phase"] == ["init"]
    assert fidelity == pytest.approx(method.evaluate_noiseless(circuit, extracted, target))
    result_snapshot = result.theta.copy()
    extracted[0] = 99.0
    np.testing.assert_array_equal(result.theta, result_snapshot)


def test_one_iteration_optimization_improves_fidelity() -> None:
    """One full-batch iteration updates parameters and improves the target fidelity."""
    method = KrotovStatePreparationMethod()
    circuit, target = _single_qubit_problem()
    initial = np.zeros(3)
    optimizer = _optimizer(max_iterations=1)
    before = method.evaluate_noiseless(circuit, initial, target)

    result = method.optimize_noiseless(circuit, target, initial, optimizer)
    parameters = method.final_parameters(result)
    after = method.training_fidelity(result)

    assert result.trace["step"] == [0, 1]
    assert result.trace["phase"] == ["init", "batch"]
    assert not np.array_equal(parameters, initial)
    assert after > before
    assert method.evaluate_noiseless(circuit, parameters, target) == pytest.approx(after)


def test_optimizer_metadata_is_complete_resolved_and_detached() -> None:
    """Metadata retains every typed input and all resolved Krotov defaults."""
    method = KrotovStatePreparationMethod()
    optimizer = _optimizer(
        max_iterations=7,
        optimizer_seed=123,
        hyperparameters={
            "step_size": 0.3,
            "schedule": {"kind": "inverse", "decay": 0.2},
        },
        max_bond_dimension=8,
        svd_threshold=1e-10,
        min_bond_dimension=2,
    )

    metadata = method.optimizer_metadata(optimizer)
    assert metadata["implementation"] == "mqt.yaqs.optimization.train_krotov_state_preparation_batch"
    assert metadata["optimizer_config"] == optimizer.to_dict()
    resolved = cast("dict[str, object]", metadata["resolved_options"])
    assert resolved == {
        "variant": "batch",
        "max_iterations": 7,
        "switch_iteration": 0,
        "online_step_size": 0.3,
        "batch_step_size": 0.3,
        "online_schedule": "inverse",
        "batch_schedule": "inverse",
        "online_decay": 0.2,
        "batch_decay": 0.2,
        "seed": 123,
        "truncation": {
            "max_bond_dim": 8,
            "svd_threshold": 1e-10,
            "trunc_mode": "discarded_weight",
            "min_bond_dim": 2,
        },
    }
    cast("dict[str, object]", metadata["optimizer_config"])["max_iterations"] = 999
    assert cast("dict[str, object]", method.optimizer_metadata(optimizer)["optimizer_config"])["max_iterations"] == 7


@pytest.mark.parametrize(
    "optimizer",
    [
        _optimizer(optimizer_id="adam"),
        _optimizer(hyperparameters={"unknown": 1}),
        _optimizer(hyperparameters={"step_size": -0.1}),
        _optimizer(hyperparameters={"schedule": {"kind": "cyclic"}}),
        _optimizer(hyperparameters={"schedule": {"kind": "constant", "decay": 0.1}}),
        _optimizer(train_trajectories_or_shots=1, training_seed=3),
    ],
)
def test_optimizer_rejects_unsupported_behavior(optimizer: OptimizerConfig) -> None:
    """The adapter does not silently ignore method-specific configuration."""
    with pytest.raises((TypeError, ValueError)):
        KrotovStatePreparationMethod().optimizer_metadata(optimizer)


def test_optimizer_reports_overflowing_json_numbers_as_validation_errors() -> None:
    """Huge JSON integers cannot leak an implementation OverflowError."""
    optimizer = _optimizer(hyperparameters={"step_size": 10**1000})
    with pytest.raises(ValueError, match="finite"):
        KrotovStatePreparationMethod().optimizer_metadata(optimizer)


def test_evaluation_uses_the_independent_truncation_config() -> None:
    """The evaluation interface accepts evaluation rather than training policy."""
    method = KrotovStatePreparationMethod()
    circuit, target = _single_qubit_problem()
    evaluation = EvaluationConfig(
        test_trajectories_or_shots=0,
        test_seed=None,
        max_bond_dimension=1,
        svd_threshold=1e-12,
        min_bond_dimension=1,
    )
    fidelity = method.evaluate_noiseless(circuit, np.zeros(3), target, evaluation=evaluation)
    assert 0.0 <= fidelity <= 1.0


def test_real_target_record_runs_through_the_training_artifact_boundary() -> None:
    """The production target-loader record works without benchmark JSON parsing."""
    collection = load_target_collection()
    target = collection.load_target(6, "gaussian_mu0p5_sigma0p1")
    config = replace(
        _config(),
        target=TargetSelection(
            num_qubits=target.num_qubits,
            target_id=target.target_id,
            target_seed=target.seed,
            fixture_format=collection.fixture_format,
            fixture_checksum=collection.fixture_checksum,
        ),
        ansatz=AnsatzConfig(0, initial_single_qubit_layer=False),
        initialization=InitializationConfig("zeros"),
    )
    method = KrotovStatePreparationMethod()

    artifact = train_state_preparation_method(method, config, collection)

    assert artifact.parameters.shape == (0,)
    assert artifact.training_fidelity == pytest.approx(
        method.evaluate_noiseless(
            artifact.circuit,
            artifact.parameters,
            target,
            evaluation=config.evaluation,
        )
    )


def test_checkpoint_serialization_is_deterministic_and_round_trips() -> None:
    """Versioned parameter-checkpoint bytes are deterministic and portable."""
    method = KrotovStatePreparationMethod()
    circuit, target = _single_qubit_problem()
    result = _zero_iteration_result(method, circuit, target)

    first = method.serialize_checkpoint(circuit, result)
    second = method.serialize_checkpoint(circuit, result)
    checksum = checkpoint_checksum(first)
    restored = method.deserialize_checkpoint(circuit, first, expected_checksum=checksum)

    assert first == second
    assert checksum.startswith("sha256:")
    np.testing.assert_array_equal(restored, result.theta)
    result_snapshot = result.theta.copy()
    restored[0] = 44.0
    np.testing.assert_array_equal(result.theta, result_snapshot)
    with np.load(io.BytesIO(first), allow_pickle=False) as archive:
        assert archive["checkpoint_format"].tobytes().decode() == KROTOV_CHECKPOINT_FORMAT
        assert archive["num_qubits"].dtype.str == "<i8"
        assert archive["theta"].dtype.str == "<f8"
        assert "trace_json" not in archive
        assert "bias" not in archive


def test_checkpoint_rejects_checksum_tampering_and_another_layout() -> None:
    """Raw-byte integrity and parameter layout are checked independently."""
    method = KrotovStatePreparationMethod()
    circuit, target = _single_qubit_problem()
    payload = method.serialize_checkpoint(circuit, _zero_iteration_result(method, circuit, target))
    checksum = checkpoint_checksum(payload)
    tampered = payload[:-1] + bytes([payload[-1] ^ 1])
    other_layout = ParameterizedCircuit(
        1,
        [
            ParameterizedGate("rx", (0,), param_index=0),
            ParameterizedGate("ry", (0,), param_index=1),
            ParameterizedGate("rz", (0,), param_index=2),
        ],
    )

    with pytest.raises(ValueError, match="checksum"):
        method.deserialize_checkpoint(circuit, tampered, expected_checksum=checksum)
    with pytest.raises(ValueError, match="layout"):
        method.deserialize_checkpoint(other_layout, payload)


def test_checkpoint_layout_includes_fixed_gates_and_rejects_data_maps() -> None:
    """Checkpoint compatibility covers every data-free circuit operation."""
    method = KrotovStatePreparationMethod()
    source = ParameterizedCircuit(
        1,
        [
            ParameterizedGate("u", (0,), fixed_params=(0.1, 0.2, 0.3)),
            ParameterizedGate("ry", (0,), param_index=0),
        ],
    )
    receiver = ParameterizedCircuit(
        1,
        [
            ParameterizedGate("u", (0,), fixed_params=(0.4, 0.2, 0.3)),
            ParameterizedGate("ry", (0,), param_index=0),
        ],
    )
    target = np.array([1.0, 0.0], dtype=np.complex128)
    result = method.optimize_noiseless(source, target, np.zeros(1), _optimizer())
    payload = method.serialize_checkpoint(source, result)

    with pytest.raises(ValueError, match="layout"):
        method.deserialize_checkpoint(receiver, payload)

    with_data_map = ParameterizedCircuit(
        1,
        [ParameterizedGate("ry", (0,), param_index=0, data_map=lambda _input: 0.0)],
    )
    with pytest.raises(ValueError, match="data_map"):
        method.serialize_checkpoint(with_data_map, result)


def test_checkpoint_wraps_corrupt_members_and_rejects_compression() -> None:
    """Malformed or compressed archives fail at the strict format boundary."""
    method = KrotovStatePreparationMethod()
    circuit, target = _single_qubit_problem()
    payload = method.serialize_checkpoint(circuit, _zero_iteration_result(method, circuit, target))

    corrupted = bytearray(payload)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        member = archive.getinfo("theta.npy")
    header = member.header_offset
    filename_length = int.from_bytes(corrupted[header + 26 : header + 28], "little")
    extra_length = int.from_bytes(corrupted[header + 28 : header + 30], "little")
    data_offset = header + 30 + filename_length + extra_length
    corrupted[data_offset + member.file_size - 1] ^= 1
    corrupt_payload = bytes(corrupted)
    with pytest.raises(ValueError, match="decoded safely"):
        method.deserialize_checkpoint(
            circuit,
            corrupt_payload,
            expected_checksum=checkpoint_checksum(corrupt_payload),
        )

    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        fields = {name: archive[name] for name in archive.files}
    compressed_buffer = io.BytesIO()
    np.savez_compressed(compressed_buffer, **fields)
    with pytest.raises(ValueError, match="uncompressed"):
        method.deserialize_checkpoint(circuit, compressed_buffer.getvalue())


def test_changed_krotov_identity_cannot_reuse_v1_semantics() -> None:
    """Every concrete public boundary enforces the frozen method identity."""

    class ChangedKrotovMethod(KrotovStatePreparationMethod):
        """Deliberately invalid identity mutation."""

        method_version = "2"

    baseline = KrotovStatePreparationMethod()
    changed = ChangedKrotovMethod()
    circuit, target = _single_qubit_problem()
    result = _zero_iteration_result(baseline, circuit, target)
    payload = baseline.serialize_checkpoint(circuit, result)

    with pytest.raises(ValueError, match="identity"):
        changed.optimizer_metadata(_optimizer())
    with pytest.raises(ValueError, match="identity"):
        changed.serialize_checkpoint(circuit, result)
    with pytest.raises(ValueError, match="identity"):
        changed.deserialize_checkpoint(circuit, payload)
    with pytest.raises(ValueError, match="identity"):
        changed.training_id(replace(_config(), method_version="2"))


def test_warm_start_supports_versioned_npz_and_legacy_npy(tmp_path: Path) -> None:
    """Both current checkpoints and checksum-verified numeric NPY vectors load."""
    method = KrotovStatePreparationMethod()
    circuit, target = _single_qubit_problem()
    result = _zero_iteration_result(method, circuit, target)

    checkpoint_payload = method.serialize_checkpoint(circuit, result)
    checkpoint_path = tmp_path / "current.npz"
    checkpoint_path.write_bytes(checkpoint_payload)
    checkpoint_config = InitializationConfig(
        "warm_start",
        warm_start_path=checkpoint_path.name,
        warm_start_checksum=checkpoint_checksum(checkpoint_payload),
    )

    legacy_path = tmp_path / "legacy.npy"
    with legacy_path.open("wb") as legacy_file:
        np.save(legacy_file, result.theta)
    legacy_payload = legacy_path.read_bytes()
    legacy_config = InitializationConfig(
        "warm_start",
        warm_start_path=legacy_path.name,
        warm_start_checksum=checkpoint_checksum(legacy_payload),
    )

    np.testing.assert_array_equal(
        method.initialize_parameters(circuit, checkpoint_config, checkpoint_root=tmp_path),
        result.theta,
    )
    np.testing.assert_array_equal(
        method.initialize_parameters(circuit, legacy_config, checkpoint_root=tmp_path),
        result.theta,
    )


@pytest.mark.parametrize(
    "parameters",
    [
        np.array([0.0, 1.0]),
        np.array([0.0, np.nan, 1.0]),
        np.array([False, True, False]),
        np.array([0.0j, 0.0j, 0.0j]),
    ],
)
def test_legacy_warm_start_rejects_invalid_vectors(tmp_path: Path, parameters: np.ndarray) -> None:
    """Legacy arrays still obey exact shape, real-type, and finiteness rules."""
    path = tmp_path / "invalid.npy"
    with path.open("wb") as file:
        np.save(file, parameters)
    payload = path.read_bytes()
    config = InitializationConfig(
        "warm_start",
        warm_start_path=path.name,
        warm_start_checksum=checkpoint_checksum(payload),
    )
    circuit, _target = _single_qubit_problem()
    with pytest.raises((TypeError, ValueError)):
        KrotovStatePreparationMethod().initialize_parameters(circuit, config, checkpoint_root=tmp_path)


def test_warm_start_verifies_checksum_before_decoding(tmp_path: Path) -> None:
    """Malformed bytes with the wrong checksum fail at the integrity boundary."""
    path = tmp_path / "broken.npy"
    path.write_bytes(b"not a NumPy file")
    config = InitializationConfig(
        "warm_start",
        warm_start_path=path.name,
        warm_start_checksum=SHA_A,
    )
    circuit, _target = _single_qubit_problem()
    with pytest.raises(ValueError, match="checksum"):
        KrotovStatePreparationMethod().initialize_parameters(circuit, config, checkpoint_root=tmp_path)


@pytest.mark.parametrize(
    ("name", "payload", "message"),
    [
        ("empty.npy", b"", "safe NPY"),
        ("oversized.npy", b"x" * 70000, "exceeds the size"),
    ],
)
def test_warm_start_bounds_raw_files_before_decoding(
    tmp_path: Path,
    name: str,
    payload: bytes,
    message: str,
) -> None:
    """Empty and oversized files fail as bounded validation errors."""
    path = tmp_path / name
    path.write_bytes(payload)
    config = InitializationConfig(
        "warm_start",
        warm_start_path=path.name,
        warm_start_checksum=checkpoint_checksum(payload),
    )
    circuit, _target = _single_qubit_problem()
    with pytest.raises(ValueError, match=message):
        KrotovStatePreparationMethod().initialize_parameters(circuit, config, checkpoint_root=tmp_path)


def test_warm_start_validates_declared_npy_shape_before_allocation(tmp_path: Path) -> None:
    """A tiny NPY file cannot request allocation from an enormous fake shape."""
    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        buffer,
        {
            "descr": np.dtype("<f8").str,
            "fortran_order": False,
            "shape": (10**9,),
        },
    )
    payload = buffer.getvalue()
    path = tmp_path / "huge-shape.npy"
    path.write_bytes(payload)
    config = InitializationConfig(
        "warm_start",
        warm_start_path=path.name,
        warm_start_checksum=checkpoint_checksum(payload),
    )
    circuit, _target = _single_qubit_problem()
    with pytest.raises(ValueError, match="safe NPY"):
        KrotovStatePreparationMethod().initialize_parameters(circuit, config, checkpoint_root=tmp_path)


def test_live_results_reject_scientifically_impossible_traces() -> None:
    """Trace extraction rejects impossible steps, phases, metrics, and losses."""
    method = KrotovStatePreparationMethod()
    circuit, target = _single_qubit_problem()
    valid = _zero_iteration_result(method, circuit, target)

    mutations: list[tuple[str, float | int | str]] = [
        ("step", 1),
        ("phase", "bogus"),
        ("gradient_norm", -1.0),
        ("loss", 0.5),
    ]
    for field_name, value in mutations:
        trace = {name: list(entries) for name, entries in valid.trace.items()}
        trace[field_name][0] = value
        malformed = KrotovResult(theta=valid.theta, bias=valid.bias, trace=trace)
        with pytest.raises(ValueError, match="Krotov trace"):
            method.extract_final_parameters(malformed)


def test_training_identity_excludes_every_test_evaluation_choice() -> None:
    """All noisy evaluations of one trained artifact share one training ID."""
    method = KrotovStatePreparationMethod()
    baseline = _config()
    noisy = replace(
        baseline,
        evaluation=EvaluationConfig(
            test_trajectories_or_shots=32,
            test_seed=91,
            store_trajectory_sidecar=True,
            confidence_level=0.95,
            confidence_interval_method="normal_clipped",
        ),
        test_noise=NoiseConfig("dephasing_1s_all", tjm_dt=1.0),
    )
    another_evaluation = replace(
        noisy,
        evaluation=replace(
            noisy.evaluation,
            test_trajectories_or_shots=64,
            test_seed=92,
            store_trajectory_sidecar=False,
            confidence_level=None,
            confidence_interval_method=None,
        ),
        test_noise=NoiseConfig("depolarizing_2s_2q", tjm_dt=1.0),
    )

    training_id = method.training_id(baseline)
    assert training_id.startswith(TRAINING_ID_PREFIX)
    assert training_id == state_preparation_training_id(method, noisy)
    assert training_id == method.training_id(another_evaluation)
    assert baseline.run_id != noisy.run_id != another_evaluation.run_id

    identity = state_preparation_training_identity(method, noisy)
    assert "evaluation" not in identity
    assert "test_noise" not in identity


@pytest.mark.parametrize(
    "changed",
    [
        {"ansatz": AnsatzConfig(2, initial_single_qubit_layer=True)},
        {"initialization": InitializationConfig("random_normal", seed=12, scale=0.1)},
        {"optimizer": _optimizer(max_iterations=1)},
        {
            "target": TargetSelection(
                num_qubits=6,
                target_id="tfim_critical",
                target_seed=None,
                fixture_format="yaqs.state_preparation_targets.v1",
                fixture_checksum=SHA_A,
            )
        },
    ],
)
def test_training_identity_changes_with_each_training_input(changed: dict[str, object]) -> None:
    """Every optimization-bearing input participates in the training identity."""
    method = KrotovStatePreparationMethod()
    baseline = _config()
    assert method.training_id(baseline) != method.training_id(replace(baseline, **changed))


def test_adapter_failure_converts_to_a_typed_benchmark_failure() -> None:
    """Orchestration can preserve adapter exceptions without fake fidelities."""
    config, collection, _target = _fixture_training_problem()
    method = KrotovStatePreparationMethod()
    invalid = _optimizer(hyperparameters={"step_size": -1.0})
    invalid_config = replace(config, optimizer=invalid)

    try:
        train_state_preparation_method(method, invalid_config, collection)
    except StatePreparationTrainingError as training_error:
        failure = BenchmarkFailure.from_exception(
            config=invalid_config,
            failure_phase=training_error.failure_phase,
            exception=training_error.exception,
            software_versions={
                "yaqs": "0.0.dev0",
                "python": "3.11",
                "numpy": "2.0",
                "scipy": "1.0",
            },
            git_commit=GIT_COMMIT,
            git_dirty=False,
        )
    else:
        pytest.fail("Invalid optimizer unexpectedly succeeded.")

    assert failure.failure_phase == "optimization"
    assert failure.exception_type == "ValueError"
    assert "step_size" in failure.message
    assert failure.config == invalid_config
    assert BenchmarkFailure.from_json(failure.to_json()) == failure
