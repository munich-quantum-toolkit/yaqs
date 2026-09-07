# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Cross-representation Hamiltonian source and backend regression tests."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse
from qiskit import QuantumCircuit

from mqt.yaqs import AnalogSimParams, DigitalSimParams, Hamiltonian, NoiseModel, Observable, Simulator, State
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import EvolutionMode
from mqt.yaqs.core.data_structures.state_utils import embed_one_site_operator


@pytest.fixture
def haar_state() -> tuple[MPS, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Haar-random 3-qubit MPS and dense snapshots.

    Returns:
        Tuple of ``(mps, psi, rho, tensors)``.
    """
    length = 3
    mps = MPS(length, state="haar-random", pad=4)
    psi = np.asarray(mps.to_vec(), dtype=np.complex128)
    rho = np.outer(psi, psi.conj())
    tensors = [np.asarray(t, dtype=np.complex128).copy() for t in mps.tensors]
    return mps, psi, rho, tensors


def _embed_operator_reference(operator: np.ndarray, sites: list[int], dimensions: list[int]) -> np.ndarray:
    """Embed an operator by direct enumeration in site-0-LSB vector order.

    Args:
        operator: Matrix whose tensor factors follow ``sites``.
        sites: Target sites in the matrix tensor-factor order.
        dimensions: Full-chain local dimensions.

    Returns:
        Full-chain operator in MPS vector order.
    """
    dimension = int(np.prod(dimensions))
    active_dimensions = [dimensions[site] for site in sites]
    spectators = [site for site in range(len(dimensions)) if site not in sites]
    embedded = np.zeros((dimension, dimension), dtype=np.complex128)
    basis_digits: list[list[int]] = []
    for index in range(dimension):
        digits: list[int] = []
        remainder = index
        for local_dimension in dimensions:
            digits.append(remainder % local_dimension)
            remainder //= local_dimension
        basis_digits.append(digits)
    for row, row_digits in enumerate(basis_digits):
        active_row = np.ravel_multi_index(tuple(row_digits[site] for site in sites), active_dimensions)
        for column, column_digits in enumerate(basis_digits):
            if any(row_digits[site] != column_digits[site] for site in spectators):
                continue
            active_column = np.ravel_multi_index(tuple(column_digits[site] for site in sites), active_dimensions)
            embedded[row, column] = operator[active_row, active_column]
    return embedded


def _deterministic_qubit_state() -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Return one deterministic three-qubit product state in all required forms."""
    local_vectors = [
        np.array([np.sqrt(0.7), 1j * np.sqrt(0.3)], dtype=np.complex128),
        np.array([0.5, np.sqrt(0.75)], dtype=np.complex128),
        np.array([np.sqrt(0.4), np.exp(0.3j) * np.sqrt(0.6)], dtype=np.complex128),
    ]
    tensors = [vector.reshape(2, 1, 1) for vector in local_vectors]
    vector = np.kron(local_vectors[2], np.kron(local_vectors[1], local_vectors[0]))
    return vector, np.outer(vector, vector.conj()), tensors


def _general_qubit_observables() -> tuple[list[Observable], list[np.ndarray]]:
    """Return general observable definitions and independent dense matrices."""
    identity = np.eye(2, dtype=np.complex128)
    x_op = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    y_op = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    z_op = np.diag([1, -1]).astype(np.complex128)
    rng = np.random.default_rng(91)
    raw_pair = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    pair = np.asarray(raw_pair + raw_pair.conj().T, dtype=np.complex128)
    raw_three = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    three = np.asarray(raw_three + raw_three.conj().T, dtype=np.complex128)
    pauli_sum = 0.4 * np.kron(x_op, np.kron(identity, z_op)) - 0.3 * np.kron(identity, np.kron(y_op, identity))
    supplied_mpo = np.kron(z_op, np.kron(y_op, x_op))
    observables = [
        Observable(pair, [0, 1]),
        Observable(pair, [2, 0]),
        Observable(three, [2, 0, 1]),
        Observable.from_pauli_sum(terms=[(0.4, "Z0 X2"), (-0.3, "Y1")], length=3),
        Observable(MPO.from_local_ops([x_op, y_op, z_op])),
    ]
    matrices = [
        _embed_operator_reference(pair, [0, 1], [2, 2, 2]),
        _embed_operator_reference(pair, [2, 0], [2, 2, 2]),
        _embed_operator_reference(three, [2, 0, 1], [2, 2, 2]),
        pauli_sum,
        supplied_mpo,
    ]
    return observables, matrices


def test_general_observables_agree_across_analog_representations() -> None:
    """MPS, MCWF, and Lindblad measure each general operator consistently."""
    vector, density_matrix, tensors = _deterministic_qubit_state()
    observables, matrices = _general_qubit_observables()
    expected = [float(np.real(np.vdot(vector, matrix @ vector))) for matrix in matrices]
    zero_hamiltonian = Hamiltonian.from_mpo(MPO.from_local_ops([np.zeros((2, 2)), np.eye(2), np.eye(2)]))
    simulator = Simulator(parallel=False, show_progress=False)
    params = AnalogSimParams(
        observables=observables,
        elapsed_time=0.1,
        dt=0.1,
        num_traj=1,
        sample_timesteps=False,
        max_bond_dim=None,
        svd_threshold=0.0,
    )
    states = [
        State(tensors=[tensor.copy() for tensor in tensors]),
        State(vector=vector.copy()),
        State(density_matrix=density_matrix.copy()),
    ]

    for state in states:
        result = simulator.run(state, zero_hamiltonian, params)
        actual = [float(np.real(values[-1])) for values in result.expectation_values]
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize(
    ("order", "evolution_mode"),
    [
        pytest.param(1, EvolutionMode.TDVP, id="order1_tdvp"),
        pytest.param(2, EvolutionMode.TDVP, id="order2_tdvp"),
        pytest.param(1, EvolutionMode.BUG, id="order1_bug"),
        pytest.param(2, EvolutionMode.BUG, id="order2_bug"),
    ],
)
def test_general_observable_reaches_each_mps_analog_solver(order: int, evolution_mode: EvolutionMode) -> None:
    """Each MPS analog solver reaches the general MPO measurement path."""
    vector, _density_matrix, tensors = _deterministic_qubit_state()
    observables, matrices = _general_qubit_observables()
    expected = float(np.real(np.vdot(vector, matrices[0] @ vector)))
    zero_hamiltonian = Hamiltonian.from_mpo(MPO.from_local_ops([np.zeros((2, 2)), np.eye(2), np.eye(2)]))
    params = AnalogSimParams(
        observables=[observables[0]],
        elapsed_time=0.1,
        dt=0.1,
        order=order,
        evolution_mode=evolution_mode,
        num_traj=1,
        sample_timesteps=False,
        max_bond_dim=None,
        svd_threshold=0.0,
    )

    result = Simulator(parallel=False, show_progress=False).run(
        State(tensors=[tensor.copy() for tensor in tensors]), zero_hamiltonian, params
    )

    assert float(np.real(result.expectation_values[0][-1])) == pytest.approx(expected, abs=1e-10)


def test_general_observable_reaches_digital_mps_measurement() -> None:
    """Digital MPS simulation measures a general prepared MPO observable."""
    vector, _density_matrix, tensors = _deterministic_qubit_state()
    observables, matrices = _general_qubit_observables()
    expected = float(np.real(np.vdot(vector, matrices[2] @ vector)))
    params = DigitalSimParams(observables=[observables[2]], num_traj=1, max_bond_dim=None, svd_threshold=0.0)

    result = Simulator(parallel=False, show_progress=False).run(
        State(tensors=[tensor.copy() for tensor in tensors]), QuantumCircuit(3), params
    )

    assert float(np.real(result.expectation_values[0][-1])) == pytest.approx(expected, abs=1e-10)


def test_general_observable_supports_mixed_dimensions_across_analog_backends() -> None:
    """All analog representations embed compact MPO support with mixed dimensions."""
    dimensions = [2, 3, 2]
    local_vectors = [
        np.array([np.sqrt(0.6), 1j * np.sqrt(0.4)], dtype=np.complex128),
        np.array([0.5, 0.5j, np.sqrt(0.5)], dtype=np.complex128),
        np.array([np.sqrt(0.3), np.exp(0.2j) * np.sqrt(0.7)], dtype=np.complex128),
    ]
    tensors = [vector.reshape(dimension, 1, 1) for vector, dimension in zip(local_vectors, dimensions, strict=True)]
    vector = np.asarray(np.kron(local_vectors[2], np.kron(local_vectors[1], local_vectors[0])), dtype=np.complex128)
    density_matrix = np.asarray(np.outer(vector, vector.conj()), dtype=np.complex128)
    rng = np.random.default_rng(92)
    raw = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    local_operator = np.asarray(raw + raw.conj().T, dtype=np.complex128)
    observable = Observable(local_operator, [2, 0])
    full_operator = _embed_operator_reference(local_operator, [2, 0], dimensions)
    expected = float(np.real(np.vdot(vector, full_operator @ vector)))
    zero_hamiltonian = MPO()
    zero_hamiltonian.custom(
        [
            np.zeros((2, 2, 1, 1), dtype=np.complex128),
            np.eye(3, dtype=np.complex128).reshape(3, 3, 1, 1),
            np.eye(2, dtype=np.complex128).reshape(2, 2, 1, 1),
        ],
        transpose=False,
    )
    hamiltonian = Hamiltonian.from_mpo(zero_hamiltonian)
    params = AnalogSimParams(
        observables=[observable],
        elapsed_time=0.1,
        dt=0.1,
        num_traj=1,
        sample_timesteps=False,
        max_bond_dim=None,
        svd_threshold=0.0,
    )
    states = [
        State(tensors=[tensor.copy() for tensor in tensors], physical_dimensions=dimensions),
        State(length=3, vector=vector.copy(), physical_dimensions=dimensions),
        State(length=3, density_matrix=density_matrix.copy(), physical_dimensions=dimensions),
    ]

    for state in states:
        result = Simulator(parallel=False, show_progress=False).run(state, hamiltonian, params)
        assert float(np.real(result.expectation_values[0][-1])) == pytest.approx(expected, abs=1e-10)


def test_haar_embedded_observables_match_mps(haar_state: tuple[MPS, np.ndarray, np.ndarray, list[np.ndarray]]) -> None:
    """Dense MCWF/Lindblad embeddings agree with ``MPS.expect`` on entangled states."""
    mps, psi, _rho, _tensors = haar_state
    length = mps.length
    for site in range(length):
        for name in ("x", "z"):
            obs = Observable(name, site)
            mps_val = mps.expect(obs)
            op = embed_one_site_operator(np.asarray(obs.matrix, dtype=np.complex128), length, site)
            embed_val = float(np.real(np.vdot(psi, op @ psi)))
            assert mps_val == pytest.approx(embed_val, abs=1e-9), f"{name} site {site}"


def test_noiseless_evolution_agrees_across_backends(
    haar_state: tuple[MPS, np.ndarray, np.ndarray, list[np.ndarray]],
) -> None:
    """MPS, MCWF, and Lindblad paths agree on noiseless Ising observables."""
    _mps, psi, rho, tensors = haar_state
    length = len(tensors)
    sim = Simulator(show_progress=False)
    hamiltonian = Hamiltonian.ising(length, J=1.0, g=0.5)

    obs_list = [Observable("z", s) for s in range(length)] + [Observable("x", 0)]
    params_mps = AnalogSimParams(observables=obs_list, elapsed_time=0.5, dt=0.05, max_bond_dim=32, svd_threshold=1e-10)
    params_dense = AnalogSimParams(observables=obs_list, elapsed_time=0.5, dt=0.05, num_traj=1)

    z_x_mps = sim.run(State(length, tensors=[t.copy() for t in tensors]), hamiltonian, params_mps, None)
    z_x_vec = sim.run(State(vector=psi.copy()), hamiltonian, params_dense, None)
    z_x_rho = sim.run(State(density_matrix=rho.copy()), hamiltonian, params_dense, None)

    for idx in range(len(obs_list)):
        mps_val = float(z_x_mps.expectation_values[idx][-1])
        vec_val = float(z_x_vec.expectation_values[idx][-1])
        rho_val = float(z_x_rho.expectation_values[idx][-1])
        assert vec_val == pytest.approx(rho_val, abs=1e-8), f"obs {idx} vector vs density_matrix"
        assert mps_val == pytest.approx(vec_val, abs=1e-5), f"obs {idx} mps vs vector"


def test_noisy_short_step_mps_vs_mcwf(
    haar_state: tuple[MPS, np.ndarray, np.ndarray, list[np.ndarray]],
) -> None:
    """Noisy one-step X expectation stays aligned between MPS and MCWF."""
    _mps, psi, _rho, tensors = haar_state
    length = len(tensors)
    sim = Simulator(show_progress=False)
    hamiltonian = Hamiltonian.ising(length, J=1.0, g=0.5)
    noise = NoiseModel([{"name": "pauli_z", "sites": [i], "strength": 0.2} for i in range(length)])
    obs = Observable("x", sites=[0])
    params = AnalogSimParams(
        observables=[obs],
        elapsed_time=0.05,
        dt=0.05,
        num_traj=1,
        max_bond_dim=16,
        random_seed=7,
    )

    mps_val = float(
        sim.run(State(length, tensors=[t.copy() for t in tensors]), hamiltonian, params, noise).expectation_values[0][
            -1
        ]
    )
    vec_val = float(sim.run(State(vector=psi.copy()), hamiltonian, params, noise).expectation_values[0][-1])
    assert mps_val == pytest.approx(vec_val, abs=1e-6)


def _final_obs(
    sim: Simulator,
    state: State,
    hamiltonian: Hamiltonian,
    params: AnalogSimParams,
) -> float:
    return float(sim.run(state, hamiltonian, params, None).expectation_values[0][-1])


@pytest.mark.parametrize("source", ["preset", "dense", "sparse"])
def test_hamiltonian_source_runs_on_all_state_representations(source: str) -> None:
    """Preset, dense, and sparse Hamiltonians work with mps, vector, and density_matrix."""
    length = 2
    ref = Hamiltonian.ising(length, J=1.0, g=0.5)
    dense = np.asarray(ref.to_matrix(), dtype=np.complex128)
    if source == "preset":
        hamiltonian = Hamiltonian.ising(length, J=1.0, g=0.5)
    elif source == "dense":
        hamiltonian = Hamiltonian(matrix=dense.copy())
    else:
        hamiltonian = Hamiltonian(sparse_matrix=scipy.sparse.csr_matrix(dense))

    sim = Simulator(show_progress=False)
    obs = Observable("z", sites=[0])
    params_mps = AnalogSimParams(observables=[obs], elapsed_time=0.3, dt=0.05, max_bond_dim=16, svd_threshold=1e-10)
    params_dense = AnalogSimParams(observables=[obs], elapsed_time=0.3, dt=0.05, num_traj=1)

    init = State(length, initial="zeros")
    psi = np.asarray(init.mps.to_vec(), dtype=np.complex128)
    rho = np.outer(psi, psi.conj())
    tensors = [np.asarray(t, dtype=np.complex128).copy() for t in init.mps.tensors]

    mps_val = _final_obs(sim, State(length, tensors=[t.copy() for t in tensors]), hamiltonian, params_mps)
    vec_val = _final_obs(sim, State(vector=psi.copy()), hamiltonian, params_dense)
    rho_val = _final_obs(sim, State(density_matrix=rho.copy()), hamiltonian, params_dense)

    assert vec_val == pytest.approx(rho_val, abs=1e-8)
    assert mps_val == pytest.approx(vec_val, abs=1e-5)

    # Cache reuse: both forms remain available and agree numerically.
    hamiltonian.ensure_mpo()
    hamiltonian.ensure_sparse()
    np.testing.assert_allclose(
        hamiltonian.mpo.to_matrix(),
        hamiltonian.sparse_matrix.toarray(),
        atol=1e-10,
    )


def test_heisenberg_noiseless_agrees_across_backends() -> None:
    """Heisenberg preset works with mps, vector, and density_matrix."""
    length = 3
    sim = Simulator(show_progress=False)
    hamiltonian = Hamiltonian.heisenberg(length, Jx=1.0, Jy=0.5, Jz=0.3, h=0.1)

    init = State(length, initial="x+")
    psi = np.asarray(init.mps.to_vec(), dtype=np.complex128)
    rho = np.outer(psi, psi.conj())
    tensors = [np.asarray(t, dtype=np.complex128).copy() for t in init.mps.tensors]

    obs_list = [Observable("z", s) for s in range(length)] + [Observable("x", 0)]
    params_mps = AnalogSimParams(observables=obs_list, elapsed_time=0.4, dt=0.05, max_bond_dim=32, svd_threshold=1e-10)
    params_dense = AnalogSimParams(observables=obs_list, elapsed_time=0.4, dt=0.05, num_traj=1)

    z_x_mps = sim.run(State(length, tensors=[t.copy() for t in tensors]), hamiltonian, params_mps, None)
    z_x_vec = sim.run(State(vector=psi.copy()), hamiltonian, params_dense, None)
    z_x_rho = sim.run(State(density_matrix=rho.copy()), hamiltonian, params_dense, None)

    for idx in range(len(obs_list)):
        mps_val = float(z_x_mps.expectation_values[idx][-1])
        vec_val = float(z_x_vec.expectation_values[idx][-1])
        rho_val = float(z_x_rho.expectation_values[idx][-1])
        assert vec_val == pytest.approx(rho_val, abs=1e-8), f"obs {idx} vector vs density_matrix"
        assert mps_val == pytest.approx(vec_val, abs=1e-5), f"obs {idx} mps vs vector"


def test_single_hamiltonian_reused_across_all_backends() -> None:
    """One Hamiltonian instance can be reused across mps, vector, and density_matrix runs."""
    length = 2
    hamiltonian = Hamiltonian.ising(length, J=1.0, g=0.5)
    sim = Simulator(show_progress=False)
    obs = Observable("z", sites=[0])
    params_mps = AnalogSimParams(observables=[obs], elapsed_time=0.2, dt=0.05, max_bond_dim=16)
    params_dense = AnalogSimParams(observables=[obs], elapsed_time=0.2, dt=0.05, num_traj=1)

    init = State(length, initial="zeros")
    psi = np.asarray(init.mps.to_vec(), dtype=np.complex128)
    rho = np.outer(psi, psi.conj())
    tensors = [np.asarray(t, dtype=np.complex128).copy() for t in init.mps.tensors]

    mps_val = _final_obs(sim, State(length, tensors=tensors), hamiltonian, params_mps)
    vec_val = _final_obs(sim, State(vector=psi), hamiltonian, params_dense)
    rho_val = _final_obs(sim, State(density_matrix=rho), hamiltonian, params_dense)

    assert vec_val == pytest.approx(rho_val, abs=1e-8)
    assert mps_val == pytest.approx(vec_val, abs=1e-5)
    np.testing.assert_allclose(
        hamiltonian.mpo.to_matrix(),
        hamiltonian.sparse_matrix.toarray(),
        atol=1e-10,
    )
