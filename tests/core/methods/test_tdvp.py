# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Time-Dependent Variational Principle (TDVP) methods in YAQS.

This module contains unit tests for verifying various components of TDVP-based methods,
including:

- Tensor decomposition (splitting and merging) routines for Matrix Product States (MPS)
   and Matrix Product Operators (MPO).
- Environment updates (left and right) and projection routines essential to efficient TDVP implementations.
- Single-site and two-site TDVP algorithms for evolving quantum states under given Hamiltonians.

The tests ensure that:
- Tensor reshaping and decomposition operations maintain numerical accuracy.
- Environment tensors are updated with correct shapes and dimensions.
- TDVP routines yield canonical MPS states with the correct orthogonality center.

These tests confirm the correctness and stability of TDVP-based simulations within the YAQS framework.
"""

# ignore non-lowercase variable names for physics notation
# ruff: noqa: N806

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.linalg import expm

from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable, StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.libraries.gate_library import GateLibrary, Z
from mqt.yaqs.core.methods.decompositions import merge_two_site, split_two_site
from mqt.yaqs.core.methods.tdvp import (
    _build_dense_effective_operator,  # noqa: PLC2701
    _run_sweeps,  # noqa: PLC2701
    _split_two_site_tdvp,  # noqa: PLC2701
    build_dense_heff_bond,
    build_dense_heff_site,
    merge_mpo_tensors,
    project_bond,
    project_site,
    tdvp,
    update_bond,
    update_left_environment,
    update_right_environment,
    update_site,
)
from mqt.yaqs.digital.digital_tjm import apply_two_qubit_gate_tdvp, apply_window, construct_generator_mpo

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.core.methods.decompositions import TruncMode

rng = np.random.default_rng()


def test_split_two_site_tdvp_left_right_sqrt() -> None:
    """Test splitting of an MPS tensor using different singular value distribution options.

    This test creates a random tensor A with shape (4, 3, 5), corresponding to (d0*d1, D0, D2)
    with d0 = d1 = 2, D0 = 3, and D2 = 5. For each SVD distribution option ("left", "right", "sqrt"),
    :func:`mqt.yaqs.core.methods.tdvp._split_two_site_tdvp` decomposes A into two tensors A0 and A1.
    The test then reconstructs A from A0 and A1 by undoing the transpose on A1 and contracting
    over the singular value index.
    """
    A = rng.random(size=(4, 3, 5)).astype(np.complex128)
    # Placeholder
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)], elapsed_time=0.2, dt=0.1, sample_timesteps=True, trunc_mode="relative"
    )
    physical_dimensions = [A.shape[0] // 2, A.shape[0] // 2]
    for distr in ["left", "right", "sqrt"]:
        A0, A1 = _split_two_site_tdvp(A, sim_params, physical_dimensions, distr, dynamic=False)
        # A0 should have shape (2, 3, r) and A1 should have shape (2, r, 5), where r is the effective rank.
        assert A0.ndim == 3
        assert A1.ndim == 3
        r = A0.shape[2]
        assert A1.shape[1] == r
        # Reconstruct A: first undo the transpose on A1 so that its shape becomes (r, 2, 5)
        A1_recon = A1.transpose((1, 0, 2))
        # Contract A0 (indices: d0, D0, r) with A1_recon (indices: r, d1, D2) over the rank index r.
        # This yields a tensor of shape (d0, d1, D0, D2). Then, we transpose to (d0*d1, D0, D2)
        A_recon = np.tensordot(A0, A1_recon, axes=(2, 0))  # shape (2, 3, 2, 5)
        A_recon = A_recon.transpose((0, 2, 1, 3)).reshape(4, 3, 5)
        np.testing.assert_allclose(A, A_recon, atol=1e-6)


def test_split_two_site_invalid_shape() -> None:
    """``split_two_site`` raises when the first axis does not match ``physical_dimensions``."""
    A = rng.random(size=(3, 3, 5)).astype(np.complex128)
    # Placeholder
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
    )
    physical_dimensions = [3, 3]
    with pytest.raises(
        ValueError, match=r"The first dimension of the tensor must be a combination of the given physical dimensions."
    ):
        split_two_site(
            A,
            physical_dimensions,
            svd_distribution="left",
            trunc_mode=cast("TruncMode", sim_params.trunc_mode),
            threshold=sim_params.svd_threshold,
            max_bond_dim=sim_params.max_bond_dim,
            min_bond_dim=sim_params.min_bond_dim,
        )


def test_merge_two_site() -> None:
    """Test :func:`mqt.yaqs.core.methods.decompositions.merge_two_site`.

    This test creates two tensors A0 and A1 with shapes (2, 3, 4) and (5, 4, 7), respectively.
    After merging, the expected shape is (10, 3, 7) because the contraction is over the bond
    between the two site tensors.
    """
    A0 = rng.random(size=(2, 3, 4)).astype(np.complex128)
    A1 = rng.random(size=(5, 4, 7)).astype(np.complex128)
    merged = merge_two_site(A0, A1)
    assert merged.shape == (10, 3, 7)


def test_merge_mpo_tensors() -> None:
    """Test the merge_mpo_tensors function.

    This test creates two 4D arrays A0 and A1 with shapes (2, 3, 4, 5) and (7, 8, 5, 9), respectively.
    After merging via merge_mpo_tensors, the expected shape is (14, 24, 4, 9).
    """
    A0 = rng.random(size=(2, 3, 4, 5)).astype(np.complex128)
    A1 = rng.random(size=(7, 8, 5, 9)).astype(np.complex128)
    merged = merge_mpo_tensors(A0, A1)
    assert merged.shape == (14, 24, 4, 9)


def test_update_right_environment() -> None:
    """Test the update_right_environment function.

    This test creates dummy arrays A, B, W, and R with compatible shapes for the contraction
    operations defined in update_right_environment. It then verifies that the resulting tensor
    has the expected shape (3, 8, 9).
    """
    A = rng.random(size=(2, 3, 4)).astype(np.complex128)
    R = rng.random(size=(4, 5, 6)).astype(np.complex128)
    W = rng.random(size=(7, 2, 8, 5)).astype(np.complex128)
    B = rng.random(size=(7, 9, 6)).astype(np.complex128)
    Rnext = update_right_environment(A, B, W, R)
    assert Rnext.shape == (3, 8, 9)


def test_update_left_environment() -> None:
    """Test the update_left_environment function.

    This test constructs dummy arrays A, B, W, and L with compatible shapes for the contraction.
    It then verifies that the output is a 3D tensor.
    """
    A = rng.random(size=(3, 4, 10)).astype(np.complex128)
    B = rng.random(size=(7, 6, 8)).astype(np.complex128)
    L_arr = rng.random(size=(4, 5, 6)).astype(np.complex128)
    W = rng.random(size=(7, 3, 5, 9)).astype(np.complex128)
    Rnext = update_left_environment(A, B, W, L_arr)
    assert Rnext.ndim == 3


def test_project_site() -> None:
    """Test the project_site function.

    This test creates dummy tensors A, R, W, and L with appropriate shapes and checks that
    the output of project_site is a 3D tensor.
    """
    A = rng.random(size=(2, 3, 4)).astype(np.complex128)
    R = rng.random(size=(4, 5, 6)).astype(np.complex128)
    W = rng.random(size=(7, 2, 8, 5)).astype(np.complex128)
    L_arr = rng.random(size=(3, 8, 9)).astype(np.complex128)
    out = project_site(L_arr, R, W, A)
    assert out.ndim == 3


def test_project_bond() -> None:
    """Test the project_bond function.

    This test creates a bond tensor C and dummy tensors L and R with compatible shapes,
    and verifies that the output has the expected shape (6, 5).
    """
    C = rng.random(size=(2, 3)).astype(np.complex128)
    R = rng.random(size=(3, 4, 5)).astype(np.complex128)
    L_arr = rng.random(size=(2, 4, 6)).astype(np.complex128)
    out = project_bond(L_arr, R, C)
    assert out.shape == (6, 5)


def test_update_site() -> None:
    """Test the update_site function.

    This test creates a dummy MPS tensor A (shape (2,2,4)), along with tensors L, R, and W,
    and applies update_site with a small time step and a fixed number of Lanczos iterations.
    The output should have the same shape as the input tensor A.
    """
    A = rng.random(size=(2, 2, 4)).astype(np.complex128)
    R = rng.random(size=(4, 1, 4)).astype(np.complex128)
    W = rng.random(size=(2, 2, 1, 1)).astype(np.complex128)
    L_arr = rng.random(size=(2, 1, 2)).astype(np.complex128)
    dt = 0.05
    out = update_site(L_arr, R, W, A, dt, krylov_tol=1e-12)
    assert out.shape == A.shape, f"Expected shape {A.shape}, got {out.shape}"


def test_update_bond() -> None:
    """Test the update_bond function.

    This test creates a square bond tensor C and compatible dummy tensors R and L.
    It applies update_bond and checks that the output shape matches that of C.
    """
    C = rng.random(size=(2, 2)).astype(np.complex128)
    R = rng.random(size=(2, 2, 2)).astype(np.complex128)
    L_arr = rng.random(size=(2, 2, 2)).astype(np.complex128)
    dt = 0.05
    out = update_bond(L_arr, R, C, dt, krylov_tol=1e-12)
    assert out.shape == C.shape, f"Expected shape {C.shape}, got {out.shape}"


def test_single_site_tdvp() -> None:
    """Test the single_site_TDVP function.

    This test initializes an Ising MPO and an MPS of length 5 (initialized to 'zeros'),
    along with AnalogSimParams configured for a single trajectory update.
    It runs single_site_TDVP and verifies that the MPS remains of the same length, all tensors are numpy arrays,
    and the MPS is left in a canonical form with the orthogonality center at site 0.
    """
    L = 5
    J = 1
    g = 0.5
    H = MPO.ising(L, J, g)

    state = MPS(L, state="zeros")
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
    )
    tdvp(state, H, sim_params, mode="1site")
    assert state.length == L
    for tensor in state.tensors:
        assert isinstance(tensor, np.ndarray)
    canonical_site = state.check_canonical_form()[0]
    assert canonical_site == 0, (
        f"MPS should be site-canonical at site 0 after single-site TDVP, but got canonical site: {canonical_site}"
    )


def test_two_site_tdvp() -> None:
    """Test the two_site_TDVP function.

    This test initializes an Ising MPO and an MPS of length 5, sets up AnalogSimParams,
    and runs two_site_TDVP. It checks that the MPS retains the correct number of tensors,
    that all tensors remain numpy arrays, and that the MPS is in canonical form with
    the orthogonality center at site 0.
    """
    L = 5
    J = 1
    g = 0.5
    H = MPO.ising(L, J, g)

    state = MPS(L, state="zeros")
    ref_mps = deepcopy(state)
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        krylov_tol=1e-12,
    )
    tdvp(state, H, sim_params, mode="2site")
    assert state.length == L
    for tensor in state.tensors:
        assert isinstance(tensor, np.ndarray)
    canonical_site = state.check_canonical_form()[0]
    assert canonical_site == 0, (
        f"MPS should be site-canonical at site 0 after two-site TDVP, but got canonical site: {canonical_site}"
    )
    # Check against exact evolution
    state_vec = ref_mps.to_vec()
    H_mat = H.to_matrix()
    U = expm(-1j * sim_params.dt * H_mat)
    state_vec = U @ state_vec
    found = state.to_vec()
    assert np.allclose(state_vec, found)


def test_two_site_tdvp_circuit_sweep_scaling() -> None:
    """Circuit-mode 2TDVP passes symmetric substeps when tdvp_sweeps>1."""
    L = 4
    H = MPO.ising(L, 1.0, 0.5)
    state = MPS(L, state="zeros")
    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], tdvp_sweeps=2, preset="exact")

    with patch("mqt.yaqs.core.methods.tdvp._two_site_tdvp_sweep") as mock_sweep:
        tdvp(state, H, sim_params, mode="2site")
        assert mock_sweep.call_count == 1
        sweep_plan = mock_sweep.call_args.kwargs["sweep_plan"]
        assert len(sweep_plan) == 2
        assert sweep_plan[0] == pytest.approx(0.5)
        assert sweep_plan[1] == pytest.approx(0.5)


def test_two_site_tdvp_circuit_single_sweep_uses_symmetric() -> None:
    """Circuit tdvp_sweeps=1 uses one symmetric substep, matching analog geometry."""
    L = 4
    H = MPO.ising(L, 1.0, 0.5)
    state = MPS(L, state="zeros")
    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], tdvp_sweeps=1, preset="exact")

    with patch("mqt.yaqs.core.methods.tdvp._two_site_tdvp_sweep") as mock_sweep:
        tdvp(state, H, sim_params, mode="2site")
        assert mock_sweep.call_args.kwargs["sweep_plan"] == [1.0]


def test_run_sweeps_invokes_substeps() -> None:
    """_run_sweeps batches symmetric substeps for analog and digital."""
    L = 3
    H = MPO.ising(L, 1.0, 0.5)
    state = MPS(L, state="zeros")

    captured_analog: list[float] = []

    def _capture_analog(*_args: object, sweep_plan: list[float] | None = None, **_kwargs: object) -> None:
        if sweep_plan is not None:
            captured_analog.extend(sweep_plan)

    analog_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.2,
        dt=0.1,
        tdvp_sweeps=3,
        sample_timesteps=True,
    )
    _run_sweeps(_capture_analog, state, H, analog_params)
    assert len(captured_analog) == 3
    for scale in captured_analog:
        assert scale == pytest.approx(1 / 3)

    captured_plan: list[float] = []

    def _capture_plan(*_args: object, sweep_plan: list[float] | None = None, **_kwargs: object) -> None:
        if sweep_plan is not None:
            captured_plan.extend(sweep_plan)

    digital_params = StrongSimParams(observables=[Observable(Z(), 0)], tdvp_sweeps=3, preset="exact")
    _run_sweeps(_capture_plan, state, H, digital_params)
    assert len(captured_plan) == 3
    for scale in captured_plan:
        assert scale == pytest.approx(1 / 3)

    captured_plan.clear()
    digital_one = StrongSimParams(observables=[Observable(Z(), 0)], tdvp_sweeps=1, preset="exact")
    _run_sweeps(_capture_plan, state, H, digital_one)
    assert captured_plan == [1.0]

    analog_one = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.1,
        dt=0.1,
        tdvp_sweeps=1,
        sample_timesteps=True,
    )
    captured_analog.clear()
    _run_sweeps(_capture_analog, state, H, analog_one)
    assert captured_analog == captured_plan


def test_two_site_tdvp_analog_sweep_preservation() -> None:
    """Analog 2TDVP with tdvp_sweeps>1 still integrates over the full dt per step."""
    L = 5
    J = 1
    g = 0.5
    H = MPO.ising(L, J, g)

    state = MPS(L, state="zeros")
    ref_mps = deepcopy(state)
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.2,
        dt=0.1,
        tdvp_sweeps=2,
        sample_timesteps=True,
        krylov_tol=1e-12,
    )
    tdvp(state, H, sim_params, mode="2site")

    state_vec = ref_mps.to_vec()
    H_mat = H.to_matrix()
    U = expm(-1j * sim_params.dt * H_mat)
    state_vec = U @ state_vec
    found = state.to_vec()
    assert np.allclose(state_vec, found)


def test_local_dynamic_tdvp_circuit_sweep_scaling() -> None:
    """Circuit-mode local dynamic TDVP honors tdvp_sweeps via a sweep plan."""
    L = 4
    H = MPO.ising(L, 1.0, 0.5)
    state = MPS(L, state="zeros")
    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], tdvp_sweeps=2, preset="exact")

    with patch("mqt.yaqs.core.methods.tdvp._local_dynamic_tdvp_sweep") as mock_sweep:
        tdvp(state, H, sim_params)
        assert mock_sweep.call_count == 1
        assert len(mock_sweep.call_args.kwargs["sweep_plan"]) == 2


def test_tdvp_mode_dispatch() -> None:
    """Tdvp routes each mode to the matching private sweep kernel."""
    L = 4
    H = MPO.ising(L, 1.0, 0.5)
    state = MPS(L, state="zeros")
    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], preset="exact")

    with patch("mqt.yaqs.core.methods.tdvp._single_site_tdvp_sweep") as mock_one:
        tdvp(state, H, sim_params, mode="1site")
        mock_one.assert_called_once()

    with patch("mqt.yaqs.core.methods.tdvp._two_site_tdvp_sweep") as mock_two:
        tdvp(state, H, sim_params, mode="2site")
        mock_two.assert_called_once()

    with patch("mqt.yaqs.core.methods.tdvp._local_dynamic_tdvp_sweep") as mock_dyn:
        tdvp(state, H, sim_params, mode="dynamic")
        mock_dyn.assert_called_once()


def test_tdvp_default_mode_is_dynamic() -> None:
    """Default tdvp mode uses local dynamic TDVP."""
    L = 4
    H = MPO.ising(L, 1.0, 0.5)
    state = MPS(L, state="zeros")
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=False,
    )

    with patch("mqt.yaqs.core.methods.tdvp._local_dynamic_tdvp_sweep") as mock_dyn:
        tdvp(state, H, sim_params)
        mock_dyn.assert_called_once()


def test_tdvp_dynamic_single_site_chain() -> None:
    """Dynamic mode on a one-site chain falls back to 1-site TDVP."""
    H = MPO.ising(1, 1.0, 0.5)
    state = MPS(1, state="zeros")
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.1,
        dt=0.1,
        sample_timesteps=False,
    )

    with patch("mqt.yaqs.core.methods.tdvp._single_site_tdvp_sweep") as mock_one:
        tdvp(state, H, sim_params)
        mock_one.assert_called_once()


def _rand_unitary_like(m: int, n: int, *, seed: int) -> NDArray[np.complex128]:
    rng_local = np.random.default_rng(seed)
    A = rng_local.normal(size=(m, n)) + 1j * rng_local.normal(size=(m, n))
    Q, _ = np.linalg.qr(A)
    Q = np.asarray(Q, dtype=np.complex128)
    return cast("NDArray[np.complex128]", Q[:, :n])


def _theta_from_singulars(s: NDArray[np.float64], m: int, n: int, *, seed: int) -> NDArray[np.complex128]:
    r = min(len(s), m, n)
    U = _rand_unitary_like(m, r, seed=seed)
    V = _rand_unitary_like(n, r, seed=seed + 1)
    S = np.diag(s[:r].astype(np.complex128))  # complex diag
    theta = (U @ S @ V.conj().T).astype(np.complex128, copy=False)
    return cast("NDArray[np.complex128]", theta)


def _as_input_tensor(theta: NDArray[np.complex128], d0: int, d1: int, d2: int, d3: int) -> NDArray[np.complex128]:
    t = theta.reshape(d0, d2, d1, d3).transpose(0, 2, 1, 3)  # (d0, d1, d2, d3)
    return cast("NDArray[np.complex128]", t.reshape(d0 * d1, d2, d3))


@pytest.mark.parametrize(
    ("svs", "threshold", "expected_keep"),
    [
        (np.array([1.0, 0.5, 0.1, 0.0100001]), 1e-4, 4),
        (np.array([1.0, 0.5, 0.01, 0.001]), 1e-4, 3),
        (np.array([1.0, 0.2, 0.2, 0.2]), 0.2**2 * 3, 1),
    ],
)
def test_split_truncation_discarded_weight_kept_count(
    svs: NDArray[np.float64], threshold: float, expected_keep: int
) -> None:
    """discarded_weight: keep count matches tail-power threshold; shapes consistent, robust at boundary."""
    d0, d1, D0, D2 = 2, 2, 3, 3
    theta = _theta_from_singulars(svs, d0 * D0, d1 * D2, seed=11)
    A_in = _as_input_tensor(theta, d0, d1, D0, D2)

    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.2,
        dt=0.1,
        min_bond_dim=1,
        svd_threshold=threshold,
        trunc_mode="discarded_weight",
        sample_timesteps=True,
    )

    A0, A1 = _split_two_site_tdvp(A_in, sim_params, [d0, d1], "sqrt", dynamic=True)
    keep = A0.shape[2]
    assert A1.shape[1] == keep

    # Scale-aware tolerance (handles tiny round-off differences robustly)
    total_power = float(np.sum(svs**2))
    tol = 64.0 * np.finfo(float).eps * max(1.0, total_power)

    # Is expected_keep exactly on the threshold within tolerance?
    tail_at_expected = svs[expected_keep:] if expected_keep < len(svs) else np.array([], dtype=svs.dtype)
    boundary_case = np.isclose(np.sum(tail_at_expected**2), threshold, rtol=0.0, atol=tol)

    if boundary_case:
        # Accept ±1 around expected_keep; SVD numerics can flip the decision at the boundary.
        acceptable = {expected_keep}
        if expected_keep > sim_params.min_bond_dim:
            acceptable.add(expected_keep - 1)
        if expected_keep < len(svs):
            acceptable.add(expected_keep + 1)
        assert keep in acceptable
    else:
        assert keep == expected_keep

    tail = svs[keep:] if keep < len(svs) else np.array([], dtype=svs.dtype)
    tail_power = float(np.sum(tail**2))

    # 1) chosen keep must satisfy tail <= threshold (within tolerance)
    assert tail_power <= threshold + tol or keep == len(svs)

    # 2) maximality: if we kept one fewer, tail would exceed threshold (unless keep is forced)
    if keep > sim_params.min_bond_dim:
        tail_prev = svs[keep - 1 :]  # discarding one extra singular value
        tail_prev_power = float(np.sum(tail_prev**2))
        assert tail_prev_power > threshold - tol


@pytest.mark.parametrize(
    ("svs", "rel_the", "expected_keep"),
    [
        # Keep all s_i strictly greater than rel_the * s_max
        (np.array([1.0, 0.6, 0.4, 0.1]), 0.5, 2),  # keep 1.0, 0.6
        (np.array([1.0, 0.99, 0.98]), 0.95, 3),  # keep all
        (np.array([1.0, 0.49, 0.3]), 0.5, 1),  # keep only 1.0
    ],
)
def test_split_truncation_relative_kept_count(svs: NDArray[np.float64], rel_the: float, expected_keep: int) -> None:
    """relative: keep count matches s_i/s_max >= threshold; shapes consistent."""
    d0, d1, D0, D2 = 2, 3, 2, 3
    theta = _theta_from_singulars(svs, d0 * D0, d1 * D2, seed=12)
    A_in = _as_input_tensor(theta, d0, d1, D0, D2)

    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.2,
        dt=0.1,
        min_bond_dim=1,
        svd_threshold=rel_the,
        trunc_mode="relative",
        sample_timesteps=True,
    )

    A0, A1 = _split_two_site_tdvp(A_in, sim_params, [d0, d1], "sqrt", dynamic=True)
    keep = A0.shape[2]
    assert keep == expected_keep
    assert A1.shape[1] == keep

    # Sanity around threshold boundary (strict >)
    smax = float(np.max(svs))
    if expected_keep > 0:
        assert np.all(svs[:expected_keep] / smax > rel_the)
    if expected_keep < len(svs):
        assert not (svs[expected_keep] / smax > rel_the)


def test_split_truncation_min_max_bond_enforced() -> None:
    """min_bond_dim/max_bond_dim are respected in both modes."""
    svs = np.array([1.0, 0.9, 0.8, 0.7])
    d0, d1, D0, D2 = 2, 2, 3, 3
    theta = _theta_from_singulars(svs, d0 * D0, d1 * D2, seed=13)
    A_in = _as_input_tensor(theta, d0, d1, D0, D2)

    # relative would keep >2, cap at max_bond_dim=2
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.2,
        dt=0.1,
        max_bond_dim=2,
        svd_threshold=0.5,
        trunc_mode="relative",
        sample_timesteps=True,
    )
    A0, A1 = _split_two_site_tdvp(A_in, sim_params, [d0, d1], "sqrt", dynamic=False)
    assert A0.shape[2] == 2
    assert A1.shape[1] == 2

    # discarded_weight would keep 1 for high threshold; min_bond_dim=2 lifts it to 2
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.2,
        dt=0.1,
        min_bond_dim=2,
        svd_threshold=2,
        trunc_mode="discarded_weight",
        sample_timesteps=True,
    )
    A0, A1 = _split_two_site_tdvp(A_in, sim_params, [d0, d1], "sqrt", dynamic=True)
    assert A0.shape[2] == 2
    assert A1.shape[1] == 2


@pytest.mark.parametrize("distr", ["left", "right", "sqrt"])
def test_split_truncation_distribution_reconstructs_optimal_rank(distr: str) -> None:
    """All SVD distribution choices reconstruct the optimal rank-k approximation."""
    svs = np.array([1.0, 0.7, 0.3, 0.1])
    d0, d1, D0, D2 = 2, 2, 3, 3
    theta = _theta_from_singulars(svs, d0 * D0, d1 * D2, seed=14)
    A_in = _as_input_tensor(theta, d0, d1, D0, D2)

    # Use a very permissive relative threshold so we keep k=4 (full rank) here;
    # the identity should still hold for any k produced.
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.2,
        dt=0.1,
        max_bond_dim=2,
        svd_threshold=0.5,
        trunc_mode="relative",
        sample_timesteps=True,
    )

    A0, A1 = _split_two_site_tdvp(A_in, sim_params, [d0, d1], distr, dynamic=True)
    k = A0.shape[2]

    L = A0.reshape(d0 * D0, k)
    R = A1.transpose(1, 0, 2).reshape(k, d1 * D2)
    theta_recon = L @ R

    # Compare with best rank-k SVD approximation of the original theta
    u, s, v = np.linalg.svd(theta, full_matrices=False)
    theta_opt_k = (u[:, :k] * s[:k]) @ v[:k, :]
    np.testing.assert_allclose(theta_recon, theta_opt_k, atol=1e-10, rtol=1e-8)


def test_dense_vs_project_site() -> None:
    """Dense H_eff should match the action of project_site on a local tensor."""
    # small random dims
    phys_dim, bond_left_dim, bond_right_dim = 2, 2, 3
    chi_a_left = chi_a_right = 2

    rng = np.random.default_rng(1234)

    ket = rng.normal(size=(phys_dim, bond_left_dim, bond_right_dim)) + 1j * rng.normal(
        size=(phys_dim, bond_left_dim, bond_right_dim)
    )
    left_env = rng.normal(size=(bond_left_dim, chi_a_left, bond_left_dim)) + 1j * rng.normal(
        size=(bond_left_dim, chi_a_left, bond_left_dim)
    )
    right_env = rng.normal(size=(bond_right_dim, chi_a_right, bond_right_dim)) + 1j * rng.normal(
        size=(bond_right_dim, chi_a_right, bond_right_dim)
    )
    op = rng.normal(size=(phys_dim, phys_dim, chi_a_left, chi_a_right)) + 1j * rng.normal(
        size=(phys_dim, phys_dim, chi_a_left, chi_a_right)
    )

    H_eff = _build_dense_effective_operator(
        project_site,
        (left_env, right_env, op),
        ket.shape,
    )

    y1 = project_site(left_env, right_env, op, ket).reshape(-1)
    y2 = H_eff @ ket.reshape(-1)

    np.testing.assert_allclose(y1, y2, atol=1e-12)


def test_dense_vs_project_bond() -> None:
    """Dense H_eff should match the action of project_bond on a bond tensor."""
    bond_left_dim, bond_right_dim = 3, 4
    chi_a = 2

    rng = np.random.default_rng(5678)

    C = rng.normal(size=(bond_left_dim, bond_right_dim)) + 1j * rng.normal(size=(bond_left_dim, bond_right_dim))
    left_env = rng.normal(size=(bond_left_dim, chi_a, bond_left_dim)) + 1j * rng.normal(
        size=(bond_left_dim, chi_a, bond_left_dim)
    )
    right_env = rng.normal(size=(bond_right_dim, chi_a, bond_right_dim)) + 1j * rng.normal(
        size=(bond_right_dim, chi_a, bond_right_dim)
    )

    H_eff = _build_dense_effective_operator(
        project_bond,
        (left_env, right_env),
        C.shape,
    )

    y1 = project_bond(left_env, right_env, C).reshape(-1)
    y2 = H_eff @ C.reshape(-1)

    np.testing.assert_allclose(y1, y2, atol=1e-12)


def test_build_dense_heff_site_matches_project_site() -> None:
    """build_dense_heff_site: vec(project_site(..., X)) == H_eff @ vec(X) for random small tensors.

    This test constructs random left/right environments and a local MPO tensor with compatible
    dimensions, builds the dense effective operator via build_dense_heff_site, and verifies
    that applying it to a flattened local tensor matches the explicit projector contraction.
    """
    # Small random dims
    phys_in, phys_out = 2, 2
    bond_left_dim, bond_right_dim = 3, 4
    chi_left, chi_right = 2, 3  # MPO virtual dims (left/right)

    rng = np.random.default_rng(4321)

    # X (ket) has shape (p, a, b)
    ket = rng.normal(size=(phys_in, bond_left_dim, bond_right_dim)) + 1j * rng.normal(
        size=(phys_in, bond_left_dim, bond_right_dim)
    )
    ket = np.asarray(ket, dtype=np.complex128)

    # left_env has shape (a, l, A)
    left_env = rng.normal(size=(bond_left_dim, chi_left, bond_left_dim)) + 1j * rng.normal(
        size=(bond_left_dim, chi_left, bond_left_dim)
    )
    left_env = np.asarray(left_env, dtype=np.complex128)

    # right_env has shape (b, r, B)
    right_env = rng.normal(size=(bond_right_dim, chi_right, bond_right_dim)) + 1j * rng.normal(
        size=(bond_right_dim, chi_right, bond_right_dim)
    )
    right_env = np.asarray(right_env, dtype=np.complex128)

    # op has shape (o, p, l, r)
    op = rng.normal(size=(phys_out, phys_in, chi_left, chi_right)) + 1j * rng.normal(
        size=(phys_out, phys_in, chi_left, chi_right)
    )
    op = np.asarray(op, dtype=np.complex128)

    H_eff = build_dense_heff_site(left_env, right_env, op)

    y1 = project_site(left_env, right_env, op, ket).reshape(-1)
    y2 = H_eff @ ket.reshape(-1)

    np.testing.assert_allclose(y1, y2, atol=1e-12)


def test_build_dense_heff_bond_matches_project_bond() -> None:
    """build_dense_heff_bond: vec(project_bond(..., C)) == H_eff @ vec(C) for random small tensors.

    This test constructs random left/right environments and a bond tensor with compatible
    dimensions, builds the dense effective operator via build_dense_heff_bond, and verifies
    that applying it to a flattened bond tensor matches the explicit projector contraction.
    """
    bond_left_dim, bond_right_dim = 3, 4
    chi = 2  # MPO virtual dim (shared index)

    rng = np.random.default_rng(8765)

    C = rng.normal(size=(bond_left_dim, bond_right_dim)) + 1j * rng.normal(size=(bond_left_dim, bond_right_dim))
    C = np.asarray(C, dtype=np.complex128)

    # left_env has shape (u, a, p)
    left_env = rng.normal(size=(bond_left_dim, chi, bond_left_dim)) + 1j * rng.normal(
        size=(bond_left_dim, chi, bond_left_dim)
    )
    left_env = np.asarray(left_env, dtype=np.complex128)

    # right_env has shape (v, a, w)
    right_env = rng.normal(size=(bond_right_dim, chi, bond_right_dim)) + 1j * rng.normal(
        size=(bond_right_dim, chi, bond_right_dim)
    )
    right_env = np.asarray(right_env, dtype=np.complex128)

    H_eff = build_dense_heff_bond(left_env, right_env)

    y1 = project_bond(left_env, right_env, C).reshape(-1)
    y2 = H_eff @ C.reshape(-1)

    np.testing.assert_allclose(y1, y2, atol=1e-12)


def test_build_dense_effective_operator_uses_generic_fallback() -> None:
    """_build_dense_effective_operator: uses generic fallback for unknown projector (basis-loop path).

    This test defines a simple linear projector that is *not* the canonical project_site/project_bond
    function object, calls _build_dense_effective_operator, and verifies:
      1) the projector was called exactly n_loc times (once per basis vector),
      2) the resulting dense operator has the expected shape.
    """
    rng_local = np.random.default_rng(2025)

    # Small tensor space: n_loc = 2 * 3 = 6
    tensor_shape = (2, 3)
    n_loc = int(np.prod(tensor_shape))

    # Define an explicit dense operator A and a projector implementing y = A @ vec(x)
    A = rng_local.normal(size=(n_loc, n_loc)) + 1j * rng_local.normal(size=(n_loc, n_loc))
    A = np.asarray(A, dtype=np.complex128)

    calls = {"n": 0}

    def custom_projector(mat: NDArray[np.complex128], x: NDArray[np.complex128]) -> NDArray[np.complex128]:
        calls["n"] += 1
        y = mat @ x.reshape(-1)
        return y.reshape(tensor_shape)

    H_eff = _build_dense_effective_operator(
        projector=custom_projector,  # not project_site / project_bond
        proj_args=(A,),
        tensor_shape=tensor_shape,
    )

    assert calls["n"] == n_loc, f"Expected {n_loc} projector calls (basis loop), got {calls['n']}"
    assert H_eff.shape == (n_loc, n_loc)


def test_build_dense_effective_operator_generic_fallback_correctness() -> None:
    """_build_dense_effective_operator: generic fallback reconstructs the operator exactly.

    This test uses a custom linear projector defined by an explicit matrix A acting on vec(x).
    The fallback builder should reconstruct A (up to floating-point roundoff), so H_eff @ vec(x)
    matches projector(x) for a random x.
    """
    rng_local = np.random.default_rng(2026)

    tensor_shape = (2, 2, 2)  # n_loc = 8
    n_loc = int(np.prod(tensor_shape))

    A = rng_local.normal(size=(n_loc, n_loc)) + 1j * rng_local.normal(size=(n_loc, n_loc))
    A = np.asarray(A, dtype=np.complex128)

    def custom_projector(mat: NDArray[np.complex128], x: NDArray[np.complex128]) -> NDArray[np.complex128]:
        y = mat @ x.reshape(-1)
        return y.reshape(tensor_shape)

    H_eff = _build_dense_effective_operator(
        projector=custom_projector,
        proj_args=(A,),
        tensor_shape=tensor_shape,
    )

    # Check action agreement on a random vector (stronger than only checking H_eff ~= A elementwise
    # if you later change internal vec conventions).
    x = rng_local.normal(size=n_loc) + 1j * rng_local.normal(size=n_loc)
    x = np.asarray(x, dtype=np.complex128)

    y1 = custom_projector(A, x.reshape(tensor_shape)).reshape(-1)
    y2 = H_eff @ x

    np.testing.assert_allclose(y1, y2, atol=1e-12)


def _digital_two_site_rzz_fidelity(
    length: int,
    theta: float,
    *,
    sites: tuple[int, int],
    initial: str = "x+",
) -> float:
    """Fidelity of ``mode='2site'`` digital TDVP against Qiskit for one RZZ gate.

    Args:
        length: Chain length.
        theta: RZZ angle.
        sites: Qubit indices for the gate.
        initial: Initial state label for :class:`~mqt.yaqs.core.data_structures.state.State`.

    Returns:
        Squared overlap fidelity against Qiskit.
    """
    qc = QuantumCircuit(length)
    qc.h(range(length))
    qc.rzz(theta, sites[0], sites[1])
    reference = np.asarray(Statevector(qc).data, dtype=np.complex128)

    prep = deepcopy(State(length, initial=initial).mps)
    gate = GateLibrary.rzz([theta])
    gate.set_sites(*sites)
    mpo, first_site, last_site = construct_generator_mpo(gate, length)
    window_state, window_mpo, _window = apply_window(prep, mpo, first_site, last_site, 1)
    sim_params = StrongSimParams(
        preset="exact",
        get_state=True,
        max_bond_dim=None,
        tdvp_sweeps=1,
        svd_threshold=1e-14,
        krylov_tol=1e-12,
    )
    tdvp(window_state, window_mpo, sim_params, mode="2site")
    return float(abs(np.vdot(reference, window_state.to_vec())) ** 2)


@pytest.mark.parametrize("theta", [0.1, 0.3, np.pi / 4])
def test_two_site_digital_rzz_l2_plus_exact(theta: float) -> None:
    """Two-site digital TDVP is exact on ``L=2``, ``|++⟩``, ``RZZ(0,1)`` without truncation."""
    fidelity = _digital_two_site_rzz_fidelity(2, theta, sites=(0, 1))
    assert fidelity == pytest.approx(1.0, abs=1e-12)


def test_two_site_digital_rzz_l2_haar_exact() -> None:
    """Two-site digital TDVP is exact on ``L=2`` Haar random states."""
    length = 2
    prep = MPS(length, state="haar-random")
    prep.normalize()
    theta = 0.3

    qc = QuantumCircuit(length)
    qc.initialize(prep.to_vec().tolist(), range(length))
    qc.rzz(theta, 0, 1)
    reference = np.asarray(Statevector(qc).data, dtype=np.complex128)

    gate = GateLibrary.rzz([theta])
    gate.set_sites(0, 1)
    mpo, first_site, last_site = construct_generator_mpo(gate, length)
    window_state, window_mpo, _window = apply_window(deepcopy(prep), mpo, first_site, last_site, 1)
    sim_params = StrongSimParams(
        preset="exact",
        get_state=True,
        max_bond_dim=None,
        tdvp_sweeps=1,
        svd_threshold=1e-14,
        krylov_tol=1e-12,
    )
    tdvp(window_state, window_mpo, sim_params, mode="2site")
    fidelity = float(abs(np.vdot(reference, window_state.to_vec())) ** 2)
    assert fidelity == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("gate_name", ["rzz", "rxx", "ryy"])
@pytest.mark.parametrize("theta", [0.1, 0.3, np.pi / 4])
def test_two_site_digital_pauli_pair_l2_plus_exact(gate_name: str, theta: float) -> None:
    """Two-site digital TDVP is exact for diagonal Pauli pair gates on ``L=2``."""
    length = 2
    qc = QuantumCircuit(length)
    qc.h(range(length))
    if gate_name == "rzz":
        qc.rzz(theta, 0, 1)
        gate = GateLibrary.rzz([theta])
    elif gate_name == "rxx":
        qc.rxx(theta, 0, 1)
        gate = GateLibrary.rxx([theta])
    else:
        qc.ryy(theta, 0, 1)
        gate = GateLibrary.ryy([theta])
    reference = np.asarray(Statevector(qc).data, dtype=np.complex128)

    prep = deepcopy(State(length, initial="x+").mps)
    gate.set_sites(0, 1)
    mpo, first_site, last_site = construct_generator_mpo(gate, length)
    window_state, window_mpo, _window = apply_window(prep, mpo, first_site, last_site, 1)
    sim_params = StrongSimParams(
        preset="exact",
        get_state=True,
        max_bond_dim=None,
        tdvp_sweeps=1,
        svd_threshold=1e-14,
        krylov_tol=1e-12,
    )
    tdvp(window_state, window_mpo, sim_params, mode="2site")
    fidelity = float(abs(np.vdot(reference, window_state.to_vec())) ** 2)
    assert fidelity == pytest.approx(1.0, abs=1e-12)
    assert window_state.norm() == pytest.approx(1.0, abs=1e-12)


def test_two_site_l2_rzz_applies_unit_evolution_time() -> None:
    """One digital 2TDVP substep on ``L=2`` must apply total generator time 1, not 2."""
    theta = 0.3
    prep = deepcopy(State(2, initial="x+").mps)
    gate = GateLibrary.rzz([theta])
    gate.set_sites(0, 1)
    mpo, first_site, last_site = construct_generator_mpo(gate, 2)
    window_state, window_mpo, _window = apply_window(prep, mpo, first_site, last_site, 1)
    sim_params = StrongSimParams(
        preset="exact",
        get_state=True,
        max_bond_dim=None,
        tdvp_sweeps=1,
        svd_threshold=1e-14,
        krylov_tol=1e-12,
    )

    recorded_dts: list[float] = []

    def _record_update_site(
        left_env: NDArray[np.complex128],
        right_env: NDArray[np.complex128],
        op: NDArray[np.complex128],
        ket: NDArray[np.complex128],
        dt: float,
        *,
        krylov_tol: float,
    ) -> NDArray[np.complex128]:
        recorded_dts.append(dt)
        return update_site(left_env, right_env, op, ket, dt, krylov_tol=krylov_tol)

    with patch("mqt.yaqs.core.methods.tdvp.update_site", side_effect=_record_update_site):
        tdvp(window_state, window_mpo, sim_params, mode="2site")

    assert recorded_dts == [pytest.approx(1.0)]


def test_lr_rzz_sweep_improves_fidelity() -> None:
    """More ``tdvp_sweeps`` should improve long-range RZZ accuracy on ``|+⟩^{⊗L}``."""
    length = 6
    theta = 0.3
    gate = GateLibrary.rzz([theta])
    gate.set_sites(0, length - 1)
    prep = State(length, initial="x+").mps
    qc = QuantumCircuit(length)
    qc.h(range(length))
    qc.rzz(theta, 0, length - 1)
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)

    def _fidelity(out: MPS) -> float:
        return float(abs(np.vdot(ref, out.to_vec())) ** 2)

    fidelities: list[float] = []
    for sweeps in (1, 16, 64):
        out = deepcopy(prep)
        params = StrongSimParams(
            preset="exact",
            get_state=True,
            max_bond_dim=None,
            tdvp_sweeps=sweeps,
            svd_threshold=1e-14,
            krylov_tol=1e-12,
        )
        apply_two_qubit_gate_tdvp(out, gate, params)
        fidelities.append(_fidelity(out))

    assert fidelities[-1] > fidelities[0] + 1e-6
