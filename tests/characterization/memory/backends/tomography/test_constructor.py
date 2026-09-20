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
