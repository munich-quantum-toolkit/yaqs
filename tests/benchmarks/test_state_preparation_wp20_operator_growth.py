# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for WP20 projector growth and genuine TFIM energy ADAPT-VQE."""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from benchmarks.state_preparation import phase2
from benchmarks.state_preparation.noise import FIXED_RATE_NOISE_DEFINITION_VERSION
from benchmarks.state_preparation.phase2.canonical import canonical_checksum
from benchmarks.state_preparation.phase2.legacy_targets import load_legacy_target_collection
from benchmarks.state_preparation.phase2.operator_growth import (
    ADAPT_STYLE_METHOD_ID,
    ENERGY_ADAPT_METHOD_ID,
    CandidateGradient,
    DeterministicReoptimizationResult,
    NoisyOperatorGrowthObjectiveRequest,
    OperatorGrowthResult,
    OperatorGrowthSpec,
    OperatorPoolSpec,
    PoolOperator,
    ProjectorObjectiveSpec,
    StandardFixedRateNoisyOperatorGrowthEvaluator,
    TargetBoundTFIMEnergyObjectiveSpec,
    TFIMHamiltonianSpec,
    adapt_style_state_preparation,
    build_projector_operator_pool,
    build_tfim_real_operator_pool,
    computational_zero_state,
    dense_open_chain_tfim_hamiltonian,
    energy_adapt_vqe,
    materialize_operator_growth_circuit,
    noisy_adapt_style_state_preparation,
    operator_growth_state,
    projector_infidelity,
    run_standard_fixed_rate_noisy_operator_growth,
    target_bound_energy_adapt_vqe,
    tfim_energy,
)
from benchmarks.state_preparation.phase2.protocol import load_initial_preregistration
from benchmarks.state_preparation.phase2.targets import (
    MaterializedTarget,
    TargetInstanceSpec,
    authorize_target_materialization,
    build_target_population_config,
    materialize_target_population,
    role_master_entropy_commitment,
)
from benchmarks.state_preparation.phase2.wp20_resources import measure_circuit_resources
from tests.benchmarks.test_state_preparation_phase2_pipeline import _screening_target_manifest

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit


_I2 = np.eye(2, dtype=np.complex128)
_OPTIMIZATION_BLOCK_ID = "operator-growth-test-block"
_OPTIMIZATION_SEED = 17
_RESOURCE_STRATUM_ID = "native-rzz-12"


def test_target_bound_noisy_operator_growth_is_public_phase2_api() -> None:
    """The publishable operator-growth path is available from the package root."""
    assert phase2.StandardFixedRateNoisyOperatorGrowthEvaluator is StandardFixedRateNoisyOperatorGrowthEvaluator
    assert phase2.noisy_adapt_style_state_preparation is noisy_adapt_style_state_preparation
    assert phase2.run_standard_fixed_rate_noisy_operator_growth is run_standard_fixed_rate_noisy_operator_growth
    assert phase2.target_bound_energy_adapt_vqe is target_bound_energy_adapt_vqe


@pytest.fixture(scope="module")
def phase2_tfim_case() -> tuple[MaterializedTarget, TargetInstanceSpec, TargetInstanceSpec]:
    """Return one authorized TFIM target, its spec, and a mismatched spec.

    Returns:
        The target, its matching specification, and another TFIM specification.
    """
    master_entropy = bytes(reversed(range(32)))
    preregistration = load_initial_preregistration()
    population = build_target_population_config(
        preregistration,
        "screening_selection",
        role_master_entropy_commitment=role_master_entropy_commitment(master_entropy),
        population_scope="primary_q6",
    )
    manifest = _screening_target_manifest()
    authorization = authorize_target_materialization(
        preregistration,
        population,
        manifest,
        master_entropy,
    )
    materialization = materialize_target_population(
        population,
        preregistration,
        manifest,
        master_entropy,
        authorization,
    )
    tfim_specs = tuple(spec for spec in manifest.instances if spec.family_id == "tfim_ground_state")
    target_spec = tfim_specs[0]
    return materialization.target(target_spec.target_instance_id), target_spec, tfim_specs[1]


def _embed_one_site(
    matrix: NDArray[np.complex128],
    site: int,
    num_qubits: int,
) -> NDArray[np.complex128]:
    """Embed a one-site matrix with site zero least significant.

    Returns:
        The full-register matrix.
    """
    embedded = np.asarray([[1.0]], dtype=np.complex128)
    for qubit in range(num_qubits - 1, -1, -1):
        embedded = np.kron(embedded, matrix if qubit == site else _I2)
    return np.asarray(embedded, dtype=np.complex128)


def _embed_two_sites(
    matrix: NDArray[np.complex128],
    first_site: int,
    second_site: int,
    num_qubits: int,
) -> NDArray[np.complex128]:
    """Embed a YAQS ascending-site two-qubit matrix.

    Returns:
        The full-register matrix.
    """
    dimension = 2**num_qubits
    embedded = np.zeros((dimension, dimension), dtype=np.complex128)
    for column in range(dimension):
        bits = [(column >> qubit) & 1 for qubit in range(num_qubits)]
        local_column = 2 * bits[first_site] + bits[second_site]
        for local_row in range(4):
            output_bits = bits.copy()
            output_bits[first_site] = (local_row >> 1) & 1
            output_bits[second_site] = local_row & 1
            row = sum(bit << qubit for qubit, bit in enumerate(output_bits))
            embedded[row, column] += matrix[local_row, local_column]
    return embedded


def _materialized_state(
    circuit: ParameterizedCircuit,
    parameters: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Apply a small materialized circuit to computational zero.

    Returns:
        The prepared dense statevector.
    """
    state = computational_zero_state(circuit.num_qubits)
    for gate in circuit.gates:
        matrix, sites = circuit.gate_matrix(gate, parameters)
        embedded = (
            _embed_one_site(matrix, sites[0], circuit.num_qubits)
            if len(sites) == 1
            else _embed_two_sites(matrix, sites[0], sites[1], circuit.num_qubits)
        )
        state = embedded @ state
    return np.asarray(state, dtype=np.complex128)


def test_projector_growth_q2_selects_ry0_and_improves_fidelity() -> None:
    """The analytic |00> to |+0> derivative must select RY on site zero."""
    target = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.complex128) / np.sqrt(2.0)

    result = adapt_style_state_preparation(
        target,
        max_operators=1,
        reoptimization_steps=60,
        learning_rate=0.08,
    )

    assert result.method_id == ADAPT_STYLE_METHOD_ID
    assert result.algorithm_label == "projector_operator_growth"
    assert "adapt_vqe" not in result.algorithm_label
    assert result.selected_operator_ids == ("ry_q0",)
    assert result.initial_objective == pytest.approx(0.5)
    assert result.final_objective is not None
    assert result.initial_objective is not None
    assert result.final_objective < 1e-4
    assert result.final_objective < result.initial_objective
    assert result.trace[0].selected_gradient == pytest.approx(-0.5)
    assert result.work.objective_calls == result.work.forward_circuit_evaluations
    assert result.work.parameter_shift_evaluations == 2 * result.work.gradient_calls
    assert result.work.backward_circuit_evaluations == 0
    assert result.wp20_work.objective_calls == result.work.objective_calls
    assert result.wp20_work.gradient_calls == result.work.gradient_calls
    assert result.wp20_work.total_sampled_trajectories == 0
    assert result.execution_mode == "analytic_reference"
    assert result.promotion_eligible is False
    assert result.circuit_resources is not None
    assert result.circuit_resources.trainable_parameter_count == 1
    assert result.circuit_resources.native_two_qubit_gates_per_chain_edge == (0,)
    assert OperatorGrowthResult.from_dict(result.to_dict()) == result

    pool = build_projector_operator_pool(2)
    selected = tuple(operator for operator in pool.operators if operator.operator_id in result.selected_operator_ids)
    prepared = operator_growth_state(2, selected, result.parameters)
    assert projector_infidelity(target, prepared) == pytest.approx(result.final_objective)


def test_energy_adapt_selects_stronger_ry0_and_lowers_actual_tfim_energy() -> None:
    """A stronger X0 field gives the largest appended-zero RY0 derivative."""
    couplings = (0.7,)
    fields = (1.2, 0.4)
    hamiltonian = dense_open_chain_tfim_hamiltonian(couplings, fields)
    expected = np.asarray(
        [
            [-0.7, -1.2, -0.4, 0.0],
            [-1.2, 0.7, 0.0, -0.4],
            [-0.4, 0.0, 0.7, -1.2],
            [0.0, -0.4, -1.2, -0.7],
        ],
        dtype=np.float64,
    )
    assert hamiltonian == pytest.approx(expected)

    result = energy_adapt_vqe(
        "tfim_ground_state",
        couplings,
        fields,
        max_operators=1,
        reoptimization_steps=60,
    )

    assert result.method_id == ENERGY_ADAPT_METHOD_ID
    assert result.algorithm_label == "genuine_energy_adapt_vqe"
    assert result.selected_operator_ids == ("ry_q0",)
    assert result.trace[0].selected_gradient == pytest.approx(-fields[0])
    assert result.initial_objective == pytest.approx(-couplings[0])
    assert result.final_objective is not None
    assert result.initial_objective is not None
    assert result.final_objective < result.initial_objective
    assert result.promotion_eligible is False

    pool = build_tfim_real_operator_pool(2)
    selected = tuple(operator for operator in pool.operators if operator.operator_id in result.selected_operator_ids)
    prepared = operator_growth_state(2, selected, result.parameters)
    assert tfim_energy(prepared, hamiltonian) == pytest.approx(result.final_objective)


def test_exact_gradient_ties_use_frozen_pool_order() -> None:
    """Equal TFIM fields must choose site zero, the lower frozen pool index."""
    result = energy_adapt_vqe(
        "tfim_ground_state",
        (0.3,),
        (0.8, 0.8),
        max_operators=1,
        reoptimization_steps=0,
    )

    assert result.selected_operator_ids == ("ry_q0",)
    candidates = result.trace[0].candidate_gradients
    ry0 = next(item for item in candidates if item.operator_id == "ry_q0")
    ry1 = next(item for item in candidates if item.operator_id == "ry_q1")
    assert ry0.absolute_gradient == pytest.approx(ry1.absolute_gradient, abs=1e-15)
    assert ry0.pool_index < ry1.pool_index


def test_q1_energy_selection_and_supplied_deterministic_reoptimizer() -> None:
    """Q1 selection works and an explicitly sealed callback may reoptimize all angles."""
    energy_result = energy_adapt_vqe(
        "tfim_ground_state",
        (),
        (0.9,),
        max_operators=1,
        reoptimization_steps=0,
    )
    assert energy_result.selected_operator_ids == ("ry_q0",)
    assert energy_result.trace[0].selected_gradient == pytest.approx(-0.9)

    target = np.asarray([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    pool = build_projector_operator_pool(1)
    spec = OperatorGrowthSpec.for_pool(
        pool,
        max_operators=1,
        reoptimization_steps=0,
        reoptimization_rule_id="analytic_test_callback_v1",
    )

    def reoptimizer(
        objective: Callable[[np.ndarray], float],
        initial_parameters: np.ndarray,
    ) -> DeterministicReoptimizationResult:
        assert initial_parameters.flags["W"] is False
        assert objective(initial_parameters) == pytest.approx(0.5)
        return DeterministicReoptimizationResult(parameters=(np.pi / 2.0,), iterations=1, gradient_calls=0)

    projector_result = adapt_style_state_preparation(
        target,
        pool=pool,
        growth_spec=spec,
        reoptimizer=reoptimizer,
    )
    assert projector_result.parameters == pytest.approx((np.pi / 2.0,))
    assert projector_result.final_objective == pytest.approx(0.0, abs=1e-14)
    assert projector_result.work.reoptimization_iterations == 1


def test_materialized_pauli_sequence_is_exact_and_has_mechanical_native_resources() -> None:
    """Direct and YZ/ZY rotations materialize exactly under frozen compilation."""
    projector_pool = build_projector_operator_pool(2)
    tfim_pool = build_tfim_real_operator_pool(2)
    by_id = {operator.operator_id: operator for operator in (*projector_pool.operators, *tfim_pool.operators)}
    operators = tuple(
        by_id[operator_id] for operator_id in ("rxx_q0_q1", "ryy_q0_q1", "rzz_q0_q1", "ryz_q0_q1", "rzy_q0_q1")
    )
    parameters = np.asarray([0.13, -0.29, 0.41, -0.53, 0.67], dtype=np.float64)
    circuit = materialize_operator_growth_circuit(2, operators)

    np.testing.assert_allclose(
        _materialized_state(circuit, parameters),
        operator_growth_state(2, operators, parameters),
        rtol=0.0,
        atol=2e-15,
    )
    sequence_resources = measure_circuit_resources(circuit)
    assert sequence_resources.trainable_parameter_count == len(operators)
    assert sequence_resources.logical_two_qubit_gates == len(operators)
    assert sequence_resources.native_two_qubit_gates == len(operators)
    assert sequence_resources.native_two_qubit_gates_per_chain_edge == (len(operators),)
    assert sequence_resources.native_one_qubit_gates == 12
    yz_gate_names = tuple(gate.name for gate in circuit.gates[-6:-3])
    zy_gate_names = tuple(gate.name for gate in circuit.gates[-3:])
    assert yz_gate_names == zy_gate_names == ("rx", "rzz", "rx")
    assert circuit.gates[-6].angle_offset == pytest.approx(np.pi / 2.0)
    assert circuit.gates[-4].angle_offset == pytest.approx(-np.pi / 2.0)
    assert circuit.gates[-3].angle_offset == pytest.approx(np.pi / 2.0)
    assert circuit.gates[-1].angle_offset == pytest.approx(-np.pi / 2.0)


def test_noisy_projector_growth_is_the_only_promotion_eligible_execution_mode() -> None:
    """Concrete empty-circuit execution is noisy, promotable, and fully accounted."""
    target = load_legacy_target_collection().target("legacy_tfim_seed_100")
    result = run_standard_fixed_rate_noisy_operator_growth(
        target,
        optimization_block_id=_OPTIMIZATION_BLOCK_ID,
        optimization_seed=_OPTIMIZATION_SEED,
        resource_stratum_id=_RESOURCE_STRATUM_ID,
        noise_id="dephasing_1s_1q",
        noise_definition_version=FIXED_RATE_NOISE_DEFINITION_VERSION,
        noise_strength_scale=1.0,
        tjm_dt=1.0,
        trajectory_count=4,
        trajectory_seed=91,
        max_operators=0,
    )

    assert result.selected_operator_ids == ()
    assert result.execution_mode == "noisy_training"
    assert result.promotion_eligible is True
    assert result.training_provenance is not None
    assert result.training_provenance.provider_id == "scaled_standard_dephasing_1s_1q"
    assert result.training_provenance.optimization_block_id == _OPTIMIZATION_BLOCK_ID
    assert result.training_provenance.optimization_seed == _OPTIMIZATION_SEED
    assert result.training_provenance.resource_stratum_id == _RESOURCE_STRATUM_ID
    assert len(result.objective_requests) == result.work.objective_calls == 1
    assert result.work.total_sampled_trajectories == 4
    assert result.work.forward_circuit_evaluations == 4
    assert result.work.trajectory_gate_applications == 0
    assert result.wp20_work.training_trajectories == result.work.total_sampled_trajectories
    assert result.initial_objective == pytest.approx(1.0 - abs(target.state_vector_copy()[0]) ** 2)
    assert result.pool is not None
    assert result.growth_spec is not None
    assert result.objective_binding is not None
    assert result.evaluator_binding is not None
    assert result.pool_checksum == result.pool.content_checksum
    assert result.growth_spec_checksum == result.growth_spec.content_checksum
    assert result.evaluator_binding.training_provenance == result.training_provenance
    assert OperatorGrowthResult.from_dict(result.to_dict()) == result


def test_target_bound_evaluator_executes_actual_standard_noise_with_common_seed() -> None:
    """Two identical materialized circuits receive deterministic common trajectories."""
    target = load_legacy_target_collection().target("legacy_tfim_seed_100")
    evaluator = StandardFixedRateNoisyOperatorGrowthEvaluator(
        target,
        optimization_block_id=_OPTIMIZATION_BLOCK_ID,
        optimization_seed=_OPTIMIZATION_SEED,
        resource_stratum_id=_RESOURCE_STRATUM_ID,
        noise_id="dephasing_1s_1q",
        noise_definition_version=FIXED_RATE_NOISE_DEFINITION_VERSION,
        noise_strength_scale=1.0,
        tjm_dt=0.9,
        trajectory_count=2,
        trajectory_seed=7,
    )
    operator = next(
        operator
        for operator in build_projector_operator_pool(target.qubit_count).operators
        if operator.operator_id == "ry_q0"
    )
    circuit = materialize_operator_growth_circuit(target.qubit_count, (operator,))
    resources = measure_circuit_resources(circuit)

    def request(index: int) -> NoisyOperatorGrowthObjectiveRequest:
        return NoisyOperatorGrowthObjectiveRequest(
            training_provenance_checksum=evaluator.training_provenance.content_checksum,
            evaluation_index=index,
            selected_operator_ids=(operator.operator_id,),
            parameters=(0.31,),
            circuit_resources_checksum=resources.content_checksum,
            logical_gate_count=len(circuit.gates),
            trajectory_count=evaluator.trajectory_count,
            trajectory_ensemble_checksum=evaluator.training_provenance.trajectory_ensemble_checksum,
        )

    first = evaluator(request(0), circuit)
    second = evaluator(request(1), materialize_operator_growth_circuit(target.qubit_count, (operator,)))
    assert first == pytest.approx(second, rel=0.0, abs=0.0)
    assert evaluator.training_provenance.provider_checksum == evaluator.provider.content_checksum
    assert evaluator.training_provenance.objective_checksum == evaluator.objective_spec.content_checksum
    wrong_operator = next(
        item for item in build_projector_operator_pool(target.qubit_count).operators if item.operator_id == "rz_q0"
    )
    with pytest.raises(ValueError, match="resource checksum"):
        evaluator(
            request(2),
            materialize_operator_growth_circuit(target.qubit_count, (wrong_operator,)),
        )
    with pytest.raises(AttributeError, match="immutable"):
        evaluator.noise_id = "depolarizing_1s_1q"


def test_noisy_growth_rejects_forged_callback_target_and_runtime_configuration() -> None:
    """Only the exact concrete target-bound evaluator reaches promotable execution."""
    collection = load_legacy_target_collection()
    target = collection.target("legacy_tfim_seed_100")
    evaluator = StandardFixedRateNoisyOperatorGrowthEvaluator(
        target,
        optimization_block_id=_OPTIMIZATION_BLOCK_ID,
        optimization_seed=_OPTIMIZATION_SEED,
        resource_stratum_id=_RESOURCE_STRATUM_ID,
        noise_id="dephasing_1s_1q",
        noise_definition_version=FIXED_RATE_NOISE_DEFINITION_VERSION,
        noise_strength_scale=1.0,
        tjm_dt=1.0,
        trajectory_count=2,
        trajectory_seed=7,
    )

    forged = cast("StandardFixedRateNoisyOperatorGrowthEvaluator", lambda _request, _circuit: 0.0)
    with pytest.raises(TypeError, match="StandardFixedRateNoisyOperatorGrowthEvaluator"):
        noisy_adapt_style_state_preparation(target, forged, max_operators=0)
    with pytest.raises(ValueError, match="not bound"):
        noisy_adapt_style_state_preparation(collection.target("legacy_tfim_seed_200"), evaluator, max_operators=0)
    custom_spec = OperatorGrowthSpec.for_pool(
        build_projector_operator_pool(target.qubit_count),
        max_operators=0,
        reoptimization_rule_id="unverifiable_external_reoptimizer_v1",
    )
    with pytest.raises(ValueError, match="non-internal reoptimization rule"):
        noisy_adapt_style_state_preparation(target, evaluator, growth_spec=custom_spec)
    with pytest.raises(ValueError, match="noise_definition_version"):
        StandardFixedRateNoisyOperatorGrowthEvaluator(
            target,
            optimization_block_id=_OPTIMIZATION_BLOCK_ID,
            optimization_seed=_OPTIMIZATION_SEED,
            resource_stratum_id=_RESOURCE_STRATUM_ID,
            noise_id="dephasing_1s_1q",
            noise_definition_version="wrong_version",
            noise_strength_scale=1.0,
            tjm_dt=1.0,
            trajectory_count=2,
            trajectory_seed=7,
        )


def test_applicable_result_rejects_native_per_edge_count_not_derived_from_resources() -> None:
    """Serialized operator counters cannot disagree with compiled resource evidence."""
    target = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    result = adapt_style_state_preparation(target, max_operators=0)
    with pytest.raises(ValueError, match="native per-edge counts"):
        replace(result, native_two_qubit_counts_by_edge=(1,))


def test_pool_definition_is_ordered_checksum_sealed_and_rejects_duplicates() -> None:
    """Pool serialization detects mutation and constructors reject duplication."""
    projector = build_projector_operator_pool(2)
    assert tuple(operator.operator_id for operator in projector.operators) == (
        "rx_q0",
        "ry_q0",
        "rz_q0",
        "rx_q1",
        "ry_q1",
        "rz_q1",
        "rxx_q0_q1",
        "ryy_q0_q1",
        "rzz_q0_q1",
    )
    assert OperatorPoolSpec.from_dict(projector.to_dict()) == projector

    mutated = projector.to_dict()
    mutated["content_checksum"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="content checksum mismatch"):
        OperatorPoolSpec.from_dict(mutated)

    with pytest.raises(ValueError, match="Duplicate operators"):
        replace(projector, operators=(projector.operators[0], projector.operators[0], *projector.operators[2:]))

    malformed = projector.to_dict()
    operators = cast("list[dict[str, object]]", malformed["operators"])
    operators[1] = dict(operators[0])
    with pytest.raises(ValueError, match="Duplicate operators"):
        OperatorPoolSpec.from_dict(malformed)

    tfim = build_tfim_real_operator_pool(2)
    assert tfim.one_qubit_generators == ("y",)
    assert tfim.two_qubit_generators == ("yz", "zy")
    assert tuple(operator.operator_id for operator in tfim.operators) == (
        "ry_q0",
        "ry_q1",
        "ryz_q0_q1",
        "rzy_q0_q1",
    )
    assert tfim.symmetry_restrictions == "real_state_odd_y_pauli_strings_only"


def test_tfim_selection_is_hamiltonian_bound_and_cannot_accept_a_target_state() -> None:
    """Energy selection provenance contains physics parameters but no target vector."""
    assert "target_state" not in inspect.signature(energy_adapt_vqe).parameters
    initial = computational_zero_state(2)
    binding = TFIMHamiltonianSpec(
        couplings=(0.6,),
        fields=(1.0, 0.2),
        initial_state_checksum=ProjectorObjectiveSpec.from_states(initial).initial_state_checksum,
    )
    document = binding.to_dict()
    assert document["target_state_binding"] == "forbidden_hamiltonian_parameters_only"
    assert "target_state_checksum" not in document
    assert TFIMHamiltonianSpec.from_dict(document) == binding

    first = energy_adapt_vqe(
        "tfim_ground_state",
        binding.couplings,
        binding.fields,
        max_operators=1,
        reoptimization_steps=0,
    )
    # An arbitrary state may change elsewhere without entering this API or its
    # Hamiltonian-bound selection identity.
    unrelated_target = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.complex128)
    unrelated_target[0] = 1.0
    unrelated_target[3] = 0.0
    second = energy_adapt_vqe(
        "tfim_ground_state",
        binding.couplings,
        binding.fields,
        max_operators=1,
        reoptimization_steps=0,
    )
    assert unrelated_target.tolist() == initial.tolist()
    assert first.objective_binding_checksum == second.objective_binding_checksum
    assert first.selected_operator_ids == second.selected_operator_ids == ("ry_q0",)


def test_target_bound_energy_adapt_derives_authorized_tfim_physics(
    phase2_tfim_case: tuple[MaterializedTarget, TargetInstanceSpec, TargetInstanceSpec],
) -> None:
    """The typed entry point derives all TFIM physics and rejects a mismatched spec."""
    target, target_spec, mismatched_spec = phase2_tfim_case
    result = target_bound_energy_adapt_vqe(
        target,
        target_spec,
        max_operators=1,
        reoptimization_steps=0,
    )

    assert result.status == "completed"
    assert result.promotion_eligible is False
    assert result.applicability.family_id == target_spec.family_id
    assert isinstance(result.objective_binding, TargetBoundTFIMEnergyObjectiveSpec)
    assert result.objective_binding.target_instance_spec == target_spec
    assert result.objective_binding.target_manifest_checksum == target.target_manifest_checksum
    assert result.objective_binding.target_vector_checksum == target.vector_checksum
    expected_couplings = tuple(cast("Sequence[float]", target_spec.parameters["couplings"]))
    expected_fields = tuple(cast("Sequence[float]", target_spec.parameters["fields"]))
    assert result.objective_binding.hamiltonian_binding.couplings == expected_couplings
    assert result.objective_binding.hamiltonian_binding.fields == expected_fields
    assert result.objective_binding_checksum == result.objective_binding.content_checksum
    assert OperatorGrowthResult.from_dict(result.to_dict()) == result
    assert "family_id" not in inspect.signature(target_bound_energy_adapt_vqe).parameters
    assert "couplings" not in inspect.signature(target_bound_energy_adapt_vqe).parameters
    assert "fields" not in inspect.signature(target_bound_energy_adapt_vqe).parameters

    with pytest.raises(ValueError, match="does not match"):
        target_bound_energy_adapt_vqe(target, mismatched_spec, max_operators=0)


def test_non_tfim_energy_adapt_is_structural_not_applicable_not_failure() -> None:
    """A non-TFIM cell returns typed zero-work N/A before objective inputs."""
    exploding = cast("tuple[float, ...]", object())
    result = energy_adapt_vqe("haar_random", exploding, exploding)

    assert result.status == "not_applicable"
    assert result.termination_reason == "not_applicable"
    assert result.applicability.status == "not_applicable"
    assert result.applicability.structural_not_applicable_is_failure is False
    assert result.applicability.is_optimizer_failure is False
    assert result.is_optimizer_failure is False
    assert result.promotion_eligible is False
    assert result.pool_checksum is None
    assert result.growth_spec_checksum is None
    assert result.objective_binding_checksum is None
    assert result.initial_objective is None
    assert result.final_objective is None
    assert result.trace == ()
    assert result.work.objective_calls == 0
    assert result.work.gradient_calls == 0
    assert result.work.forward_circuit_evaluations == 0
    assert OperatorGrowthResult.from_dict(result.to_dict()) == result


def test_native_cap_is_per_edge_and_infeasible_candidates_are_not_evaluated() -> None:
    """A zero per-edge cap preserves one-qubit scans and marks pair candidates."""
    target = np.asarray([1.0, 0.0, 0.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
    pool = build_projector_operator_pool(2)
    spec = OperatorGrowthSpec.for_pool(
        pool,
        max_operators=len(pool.operators),
        native_two_qubit_cap_per_edge=0,
        reoptimization_steps=0,
        gradient_tolerance=0.0,
    )
    result = adapt_style_state_preparation(target, pool=pool, growth_spec=spec)

    assert result.native_two_qubit_counts_by_edge == (0,)
    assert all(not operator_id.startswith(("rxx", "ryy", "rzz")) for operator_id in result.selected_operator_ids)
    terminal = result.trace[-1]
    if terminal.event == "native_cap_stop":
        blocked = tuple(item for item in terminal.candidate_gradients if not item.native_cap_feasible)
        assert blocked
        assert all(item.gradient is None and item.absolute_gradient is None for item in blocked)


def test_strict_nested_records_reject_inconsistent_gradient_and_result_mutation() -> None:
    """Nested trace semantics and outer checksums cannot be caller rewritten."""
    with pytest.raises(ValueError, match="absolute_gradient"):
        CandidateGradient(
            operator_id="ry_q0",
            pool_index=0,
            gradient=-0.4,
            absolute_gradient=0.3,
            native_two_qubit_increment=0,
            native_cap_feasible=True,
        )

    target = np.asarray([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    result = adapt_style_state_preparation(target, max_operators=1, reoptimization_steps=0)
    document = result.to_dict()
    document["content_checksum"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="content checksum mismatch"):
        OperatorGrowthResult.from_dict(document)

    trace_tamper = result.to_dict()
    trace = cast("list[dict[str, object]]", trace_tamper["trace"])
    candidates = cast("list[dict[str, object]]", trace[0]["candidate_gradients"])
    candidates[0]["pool_index"] = 1
    trace_tamper["content_checksum"] = canonical_checksum({
        key: value for key, value in trace_tamper.items() if key != "content_checksum"
    })
    with pytest.raises(ValueError, match="candidate identity"):
        OperatorGrowthResult.from_dict(trace_tamper)


def test_result_rejects_resealed_arbitrary_binding_checksums() -> None:
    """Exact persisted documents defeat the former opaque-checksum resealing attack."""
    target = load_legacy_target_collection().target("legacy_tfim_seed_100")
    result = run_standard_fixed_rate_noisy_operator_growth(
        target,
        optimization_block_id=_OPTIMIZATION_BLOCK_ID,
        optimization_seed=_OPTIMIZATION_SEED,
        resource_stratum_id=_RESOURCE_STRATUM_ID,
        noise_id="dephasing_1s_1q",
        noise_definition_version=FIXED_RATE_NOISE_DEFINITION_VERSION,
        noise_strength_scale=1.0,
        tjm_dt=1.0,
        trajectory_count=1,
        trajectory_seed=17,
        max_operators=0,
    )
    assert result.promotion_eligible is True
    document = result.to_dict()
    document["pool_checksum"] = "sha256:" + "1" * 64
    document["growth_spec_checksum"] = "sha256:" + "2" * 64
    document["objective_binding_checksum"] = "sha256:" + "3" * 64
    document["content_checksum"] = canonical_checksum({
        key: value for key, value in document.items() if key != "content_checksum"
    })

    with pytest.raises(ValueError, match="do not reproduce"):
        OperatorGrowthResult.from_dict(document)


def test_pool_operator_rejects_non_nearest_neighbor_and_wrong_native_cost() -> None:
    """Pool records cannot hide routing or substitute parameters for gate cost."""
    with pytest.raises(ValueError, match="nearest-neighbor"):
        PoolOperator("rxx_q0_q2", "xx", (0, 2), 1, "local_basis_rzz_local_basis")
    with pytest.raises(ValueError, match="exactly one native RZZ"):
        PoolOperator("rxx_q0_q1", "xx", (0, 1), 0, "local_basis_rzz_local_basis")
