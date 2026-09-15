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
from scipy.linalg import expm

from mqt.yaqs import AnalogSimParams, Hamiltonian, NoiseModel, Observable, Simulator, State
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.state_utils import embed_one_site_operator
from tests.site_order_reference import embed_local_operator, mixed_radix_index


@pytest.fixture
def deterministic_state() -> tuple[MPS, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Deterministic entangled 3-qubit MPS and dense snapshots.

    Returns:
        Tuple of ``(mps, psi, rho, tensors)``.
    """
    length = 3
    rng = np.random.default_rng(20260915)
    shapes = ((2, 1, 2), (2, 2, 2), (2, 2, 1))
    tensors = [
        np.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape), dtype=np.complex128)
        for shape in shapes
    ]
    mps = MPS(length, tensors=tensors)
    mps.normalize("B")
    psi = np.asarray(mps.to_vec(), dtype=np.complex128)
    rho = np.outer(psi, psi.conj())
    tensors = [np.asarray(t, dtype=np.complex128).copy() for t in mps.tensors]
    return mps, psi, rho, tensors


def test_entangled_embedded_observables_match_mps(
    deterministic_state: tuple[MPS, np.ndarray, np.ndarray, list[np.ndarray]],
) -> None:
    """Dense MCWF/Lindblad embeddings agree with ``MPS.expect`` on entangled states."""
    mps, psi, _rho, _tensors = deterministic_state
    length = mps.length
    for site in range(length):
        for name in ("x", "z"):
            obs = Observable(name, site)
            mps_val = mps.expect(obs)
            op = embed_one_site_operator(np.asarray(obs.matrix, dtype=np.complex128), length, site)
            embed_val = float(np.real(np.vdot(psi, op @ psi)))
            assert mps_val == pytest.approx(embed_val, abs=1e-9), f"{name} site {site}"


@pytest.mark.parametrize(
    ("length", "sites", "basis_digits"),
    [
        (2, [0, 1], (1, 0)),
        (2, [1, 0], (0, 1)),
        (3, [2, 0], (0, 0, 1)),
        (3, [0, 2], (1, 0, 0)),
    ],
)
def test_asymmetric_two_site_observable_agrees_across_representations(
    length: int,
    sites: list[int],
    basis_digits: tuple[int, ...],
) -> None:
    """Adjacent and periodic observables follow their listed sites on all backends."""
    pauli_z = np.diag([1, -1]).astype(np.complex128)
    local_matrix = np.kron(pauli_z, np.eye(2, dtype=np.complex128))
    observable = Observable(local_matrix, sites)
    dimensions = (2,) * length
    dense_observable = embed_local_operator(local_matrix, tuple(sites), dimensions)
    initial_vector = np.zeros(2**length, dtype=np.complex128)
    initial_vector[mixed_radix_index(basis_digits, dimensions)] = 1.0
    expected = float(np.real(np.vdot(initial_vector, dense_observable @ initial_vector)))
    hamiltonian = Hamiltonian(matrix=np.zeros((2**length, 2**length), dtype=np.complex128))
    parameters = AnalogSimParams(
        observables=[observable],
        elapsed_time=0.1,
        dt=0.1,
        num_traj=1,
        max_bond_dim=None,
        svd_threshold=0.0,
        sample_timesteps=False,
    )

    results = []
    for representation in ("mps", "vector", "density_matrix"):
        state = State(
            length,
            initial="basis",
            basis_string="".join(str(digit) for digit in basis_digits),
            representation=representation,
        )
        result = Simulator(show_progress=False).run(state, hamiltonian, parameters, None)
        results.append(float(np.real(result.expectation_values[0][-1])))

    np.testing.assert_allclose(results, expected, atol=1e-12)


def test_noiseless_evolution_agrees_across_backends(
    deterministic_state: tuple[MPS, np.ndarray, np.ndarray, list[np.ndarray]],
) -> None:
    """MPS, MCWF, and Lindblad paths agree on noiseless Ising observables."""
    _mps, psi, rho, tensors = deterministic_state
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
    deterministic_state: tuple[MPS, np.ndarray, np.ndarray, list[np.ndarray]],
) -> None:
    """Noisy one-step X expectation stays aligned between MPS and MCWF."""
    _mps, psi, _rho, tensors = deterministic_state
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


@pytest.mark.parametrize("source", ["mpo", "tensors", "dense", "sparse"])
def test_asymmetric_hamiltonian_preserves_sites_across_representations(source: str) -> None:
    """Every Hamiltonian source evolves site 0 on all analog backends."""
    length = 2
    identity = np.eye(2, dtype=np.complex128)
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    dense = np.asarray(np.kron(identity, pauli_x), dtype=np.complex128)
    source_mpo = MPO.from_local_ops([pauli_x, identity])
    if source == "mpo":
        hamiltonian = Hamiltonian.from_mpo(source_mpo)
    elif source == "tensors":
        external_tensors = [tensor.transpose(2, 3, 0, 1) for tensor in source_mpo.tensors]
        hamiltonian = Hamiltonian(tensors=external_tensors)
    elif source == "dense":
        hamiltonian = Hamiltonian(matrix=dense.copy())
    else:
        hamiltonian = Hamiltonian(sparse_matrix=scipy.sparse.csr_matrix(dense))

    sim = Simulator(show_progress=False)
    observables = [Observable("z", 0), Observable("z", 1)]
    evolution_time = 0.2
    params = AnalogSimParams(
        observables=observables,
        elapsed_time=evolution_time,
        dt=evolution_time,
        num_traj=1,
        max_bond_dim=None,
        svd_threshold=1e-12,
        get_state=True,
        sample_timesteps=False,
    )
    initial = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    expected_vector = expm(-1j * evolution_time * dense) @ initial
    expected_density = np.outer(expected_vector, expected_vector.conj())
    expected_observables = [np.cos(2 * evolution_time), 1.0]

    for representation in ("mps", "vector", "density_matrix"):
        result = sim.run(
            State(length, initial="zeros", representation=representation),
            hamiltonian,
            params,
            None,
        )
        assert result.output_state is not None
        if representation == "mps":
            output = result.output_state.mps.to_vec()
            expected_state = expected_vector
        elif representation == "vector":
            output = result.output_state.vector
            expected_state = expected_vector
        else:
            output = result.output_state.density_matrix
            expected_state = expected_density
        np.testing.assert_allclose(output, expected_state, atol=1e-10)
        np.testing.assert_allclose(
            [values[-1] for values in result.expectation_values],
            expected_observables,
            atol=1e-10,
        )

    hamiltonian.ensure_mpo()
    hamiltonian.ensure_sparse()
    for converted in (
        hamiltonian.to_matrix(),
        hamiltonian.to_sparse_matrix().toarray(),
        hamiltonian.mpo.to_matrix(),
        hamiltonian.mpo.to_sparse_matrix().toarray(),
    ):
        np.testing.assert_allclose(converted, dense, atol=1e-12)


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
