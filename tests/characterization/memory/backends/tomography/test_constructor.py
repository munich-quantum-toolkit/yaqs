# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for build_process_tensor entry point."""

from __future__ import annotations

import sys
from typing import Any, cast

import numpy as np
import pytest

import mqt.yaqs.characterization.memory.backends.tomography.constructor as constructor_module
from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer
from mqt.yaqs.characterization.memory.backends.tomography import build_process_tensor
from mqt.yaqs.characterization.memory.backends.tomography.constructor import run_all_sequences
from mqt.yaqs.characterization.memory.operational_memory.samples import ProbeSet
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.noise_model import NoiseModel


def test_build_process_tensor_invalid_return_type_raises() -> None:
    """Unknown return_type values are rejected."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8)
    with pytest.raises(ValueError, match="Unknown return_type"):
        build_process_tensor(op, params, timesteps=[0.0, 0.0], return_type=cast("Any", "nope"))


def test_build_process_tensor_returns_dense_and_mpo_smoke() -> None:
    """build_process_tensor returns dense and MPO process-tensor wrappers."""
    ham = Hamiltonian.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8)
    mc = MemoryCharacterizer(parallel=False, show_progress=False)

    dense = mc.build_process_tensor(ham, params, timesteps=[0.0, 0.0], return_type="dense")
    assert dense.to_matrix().shape == (8, 8)

    mpo = mc.build_process_tensor(ham, params, timesteps=[0.0, 0.0], compress_every=1)
    mat = mpo.to_matrix()
    assert mat.shape == (8, 8)
    np.testing.assert_allclose(mat, dense.to_matrix(), atol=1e-8)


def test_build_process_tensor_rejects_k_zero() -> None:
    """Zero-step schedules are rejected for both MPO and dense construction."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8)
    with pytest.raises(ValueError, match="at least one intervention"):
        build_process_tensor(op, params, timesteps=[])
    with pytest.raises(ValueError, match="at least one intervention"):
        build_process_tensor(op, params, timesteps=[0.1])
    with pytest.raises(ValueError, match="No sequences for num_interventions=0"):
        build_process_tensor(op, params, timesteps=[], return_type="dense")
    with pytest.raises(ValueError, match="No sequences for num_interventions=0"):
        build_process_tensor(op, params, timesteps=[0.1], return_type="dense")


def test_build_process_tensor_parallel_smoke() -> None:
    """build_process_tensor runs with parallel execution enabled for dense and MPO."""
    ham = Hamiltonian.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8)
    mc = MemoryCharacterizer(parallel=True, max_workers=2, show_progress=False)
    dense = mc.build_process_tensor(ham, params, timesteps=[0.0, 0.0], return_type="dense")
    assert dense.to_matrix().shape == (8, 8)
    mpo = mc.build_process_tensor(ham, params, timesteps=[0.0, 0.0], compress_every=1)
    assert mpo.to_matrix().shape == (8, 8)


def test_build_process_tensor_stores_reference_initial_rho() -> None:
    """Built process tensors store the site-0 reference after U_0 evolution."""
    ham = Hamiltonian.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8)
    pt = MemoryCharacterizer(parallel=False, show_progress=False).build_process_tensor(
        ham, params, timesteps=[0.0, 0.0], return_type="dense"
    )
    np.testing.assert_allclose(pt.initial_rho, np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128), atol=1e-10)


def test_noisy_dense_process_tensor_matches_initial_excitation_channel() -> None:
    """Noisy tomography preserves retained-outcome probabilities and mixed conditional states."""
    duration = 0.1
    survival = 0.4
    num_trajectories = 256
    params = AnalogSimParams(dt=duration, max_bond_dim=8, random_seed=42)
    noise = NoiseModel([{"name": "raising", "sites": [0], "strength": -np.log(survival) / duration}])
    pt = MemoryCharacterizer(parallel=False, show_progress=False, representation="vector").build_process_tensor(
        Hamiltonian.ising(length=1, J=0.0, g=0.0),
        params,
        timesteps=[duration, 0.0],
        noise_model=noise,
        num_trajectories=num_trajectories,
        basis="standard",
        return_type="dense",
    )

    # Excitation from |0> leaves population exp(-gamma * duration) in |0>.
    reference_initial = np.diag([survival, 1.0 - survival]).astype(np.complex128)
    sampling_tolerance = 5.0 * np.sqrt(survival * (1.0 - survival) / num_trajectories)
    np.testing.assert_allclose(pt.initial_rho, reference_initial, atol=sampling_tolerance, rtol=0.0)
    probes = ProbeSet(
        cut=1,
        num_interventions=1,
        past_features=np.zeros((2, 1, 32), dtype=np.float32),
        future_features=np.zeros((2, 1, 32), dtype=np.float32),
        past_pairs=[[], []],
        past_cut_meas=[np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        future_prep_cut=[np.array([1.0, 1.0]) / np.sqrt(2.0), np.array([1.0, 1.0j]) / np.sqrt(2.0)],
        future_pairs=[[], []],
    )
    responses, weights = pt.evaluate_probes_with_weights(probes)
    expected_weights = np.array([[survival, survival], [1.0 - survival, 1.0 - survival]])
    expected_responses = np.array([[[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]]] * 2)
    np.testing.assert_allclose(weights, expected_weights, atol=sampling_tolerance, rtol=0.0)
    np.testing.assert_allclose(responses, expected_responses, atol=1e-12, rtol=0.0)

    rotation = np.array([[np.cos(0.2), -np.sin(0.2)], [np.sin(0.2), np.cos(0.2)]])
    outcome_operator = rotation @ np.diag([1.0, 0.5])

    def retain_outcome(rho: np.ndarray) -> np.ndarray:
        """Apply one outcome of a nonunitary filter followed by a rotation.

        Args:
            rho: Input system density matrix.

        Returns:
            Subnormalized output for the retained outcome.
        """
        return outcome_operator @ rho @ outcome_operator.conj().T

    reference = retain_outcome(reference_initial)
    reference /= np.trace(reference)
    # Propagate five binomial standard errors through conditional normalization.
    conditional_tolerance = 0.0
    for population in (survival - sampling_tolerance, survival + sampling_tolerance):
        bound = retain_outcome(np.diag([population, 1.0 - population]))
        bound /= np.trace(bound)
        conditional_tolerance = max(conditional_tolerance, float(np.max(np.abs(bound - reference))))
    prediction = pt.predict([retain_outcome])
    np.testing.assert_allclose(prediction, reference, atol=conditional_tolerance, rtol=0.0)
    assert np.trace(prediction) == pytest.approx(1.0, abs=1e-12)
    noiseless_output = retain_outcome(np.diag([1.0, 0.0]))
    assert np.linalg.norm(reference - noiseless_output) > 2.0 * conditional_tolerance


def test_build_process_tensor_validates_initial_rho_arg() -> None:
    """Optional initial_rho at build time is checked against the computed reference."""
    ham = Hamiltonian.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8)
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    ref = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    pt = mc.build_process_tensor(ham, params, timesteps=[0.0, 0.0], return_type="dense", initial_rho=ref)
    np.testing.assert_allclose(pt.initial_rho, ref, atol=1e-10)
    with pytest.raises(ValueError, match="rho0 does not match"):
        mc.build_process_tensor(
            ham,
            params,
            timesteps=[0.0, 0.0],
            return_type="dense",
            initial_rho=np.eye(2, dtype=np.complex128) / 2.0,
        )


def test_run_all_sequences_rejects_non_positive_num_trajectories_with_noise() -> None:
    """Noisy tomography rejects zero or negative trajectory counts before reference-state work."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8)
    noise_model = NoiseModel([{"name": "pauli_z", "sites": [0], "strength": 0.1}])
    with pytest.raises(ValueError, match="num_trajectories must be >= 1"):
        run_all_sequences(
            op,
            params,
            [0.0, 0.0],
            parallel=False,
            num_trajectories=0,
            noise_model=noise_model,
            show_progress=False,
        )
    with pytest.raises(ValueError, match="num_trajectories must be >= 1"):
        run_all_sequences(
            op,
            params,
            [0.0, 0.0],
            parallel=False,
            num_trajectories=-3,
            noise_model=noise_model,
            show_progress=False,
        )


@pytest.mark.parametrize("num_trajectories", [False, 1.5, "1"])
def test_run_all_sequences_rejects_non_integer_num_trajectories(num_trajectories: object) -> None:
    """The lower-level dense runner rejects booleans and floats before setup."""
    with pytest.raises(TypeError, match="num_trajectories must be an integer"):
        run_all_sequences(
            MPO.ising(length=1, J=0.0, g=0.0),
            AnalogSimParams(dt=0.1, max_bond_dim=8),
            [0.0, 0.0],
            parallel=False,
            num_trajectories=num_trajectories,  # ty: ignore[invalid-argument-type]
            show_progress=False,
        )


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"num_trajectories": 0}, ValueError, r"num_trajectories must be >= 1"),
        ({"num_trajectories": -1}, ValueError, r"num_trajectories must be >= 1"),
        ({"num_trajectories": False}, TypeError, r"num_trajectories must be an integer"),
        ({"num_trajectories": 1.5}, TypeError, r"num_trajectories must be an integer"),
        ({"num_trajectories": "1"}, TypeError, r"num_trajectories must be an integer"),
    ],
)
def test_build_process_tensor_validates_dense_sizes(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    """The exported dense constructor validates trajectory counts at its boundary."""
    monkeypatch.setattr(constructor_module, "_construct_data", pytest.fail)
    with pytest.raises(error, match=match):
        build_process_tensor(
            MPO.ising(length=1, J=0.0, g=0.0),
            AnalogSimParams(dt=0.1, max_bond_dim=8),
            timesteps=[0.0, 0.0],
            return_type="dense",
            **kwargs,  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"max_bond_dim": 0}, ValueError, r"max_bond_dim must be >= 1"),
        ({"max_bond_dim": -1}, ValueError, r"max_bond_dim must be >= 1"),
        ({"max_bond_dim": False}, TypeError, r"max_bond_dim must be an integer"),
        ({"max_bond_dim": 1.5}, TypeError, r"max_bond_dim must be an integer"),
        ({"max_bond_dim": "1"}, TypeError, r"max_bond_dim must be an integer"),
        ({"compress_every": 0}, ValueError, r"compress_every must be >= 1"),
        ({"compress_every": -1}, ValueError, r"compress_every must be >= 1"),
        ({"compress_every": False}, TypeError, r"compress_every must be an integer"),
        ({"compress_every": 1.5}, TypeError, r"compress_every must be an integer"),
        ({"compress_every": "1"}, TypeError, r"compress_every must be an integer"),
        ({"n_sweeps": -1}, ValueError, r"n_sweeps must be >= 0"),
        ({"n_sweeps": False}, TypeError, r"n_sweeps must be an integer"),
        ({"n_sweeps": 1.5}, TypeError, r"n_sweeps must be an integer"),
        ({"n_sweeps": "0"}, TypeError, r"n_sweeps must be an integer"),
    ],
)
def test_build_process_tensor_validates_direct_sizes(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    """The exported MPO constructor validates direct-compression sizes at its boundary."""
    monkeypatch.setitem(
        sys.modules,
        "mqt.yaqs.characterization.memory.backends.tomography.direct",
        None,
    )
    with pytest.raises(error, match=match):
        build_process_tensor(
            MPO.ising(length=1, J=0.0, g=0.0),
            AnalogSimParams(dt=0.1, max_bond_dim=8),
            timesteps=[0.0, 0.0],
            **kwargs,  # ty: ignore[invalid-argument-type]
        )


def test_tomography_constructors_accept_numpy_integer_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NumPy integer tomography sizes pass each directly callable constructor boundary."""
    operator = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8)

    def _stop_dense(*_args: object, **_kwargs: object) -> None:
        msg = "validated dense sizes"
        raise RuntimeError(msg)

    monkeypatch.setattr(constructor_module, "_construct_data", _stop_dense)
    with pytest.raises(RuntimeError, match="validated dense sizes"):
        build_process_tensor(
            operator,
            params,
            timesteps=[0.0, 0.0],
            return_type="dense",
            num_trajectories=np.int64(1),  # ty: ignore[invalid-argument-type]
        )

    def _stop_copy(_value: object) -> None:
        msg = "validated runner sizes"
        raise RuntimeError(msg)

    monkeypatch.setattr(constructor_module.copy, "deepcopy", _stop_copy)
    with pytest.raises(RuntimeError, match="validated runner sizes"):
        run_all_sequences(
            operator,
            params,
            [0.0, 0.0],
            parallel=False,
            num_trajectories=np.int64(1),  # ty: ignore[invalid-argument-type]
        )

    monkeypatch.setitem(
        sys.modules,
        "mqt.yaqs.characterization.memory.backends.tomography.direct",
        None,
    )
    with pytest.raises(ModuleNotFoundError):
        build_process_tensor(
            operator,
            params,
            timesteps=[0.0, 0.0],
            max_bond_dim=np.int64(2),  # ty: ignore[invalid-argument-type]
            compress_every=np.int64(1),  # ty: ignore[invalid-argument-type]
            n_sweeps=np.int64(0),  # ty: ignore[invalid-argument-type]
        )
