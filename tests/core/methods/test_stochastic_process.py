# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the stochastic_process module."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.core.methods.stochastic_process import (
    calculate_stochastic_factor,
    create_probability_distribution,
    stochastic_process,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

rng = np.random.default_rng()


def crandn(
    size: int | tuple[int, ...], *args: int, seed: np.random.Generator | int | None = None
) -> NDArray[np.complex128]:
    """Draw random samples from the standard complex normal distribution.

    Args:
        size (int |Tuple[int,...]): The size/shape of the output array.
        *args (int): Additional dimensions for the output array.
        seed (Generator | int): The seed for the random number generator.

    Returns:
        NDArray[np.complex128]: The array of random complex numbers.
    """
    if isinstance(size, int) and len(args) > 0:
        size = (size, *list(args))
    elif isinstance(size, int):
        size = (size,)
    rng = np.random.default_rng(seed)
    # 1/sqrt(2) is a normalization factor
    return (rng.standard_normal(size) + 1j * rng.standard_normal(size)) / np.sqrt(2)


def random_mps(shapes: list[tuple[int, int, int]], *, normalize: bool = True) -> MPS:
    """Create a random MPS with the given shapes.

    Args:
        shapes (List[Tuple[int, int, int]]): The shapes of the tensors in the
            MPS.
        normalize (bool): Whether to normalize the MPS.

    Returns:
        MPS: The random MPS.
    """
    tensors = [crandn(shape) for shape in shapes]
    mps = MPS(len(shapes), tensors=tensors)
    if normalize:
        mps.normalize()
    return mps


def test_calculate_stochastic_factor_zero_norm() -> None:
    """Test that the stochastic factor is zero for a norm-1 state at site 0.

    This test creates a normalized MPS and verifies that the stochastic factor
    computed by `calculate_stochastic_factor` is exactly zero, confirming correct
    behavior for states with unit norm at the first site.
    """
    state = random_mps([(2, 1, 2), (2, 2, 2), (2, 2, 1)])
    # Manually set norm to 1 at site 0
    state.normalize()
    factor = calculate_stochastic_factor(state)
    assert np.isclose(factor, 0.0), "Stochastic factor should be zero for normalized state."


def test_calculate_stochastic_factor_nontrivial() -> None:
    """Test stochastic factor is correct for a non-unit norm at site 0.

    This test artificially rescales the first tensor of an MPS, resulting in a non-unit
    norm, and checks that `calculate_stochastic_factor` returns the expected value
    (1 minus the actual norm of the first site).
    """
    state = random_mps([(2, 1, 2), (2, 2, 2), (2, 2, 1)])
    state.tensors[0] *= 0.8
    factor = calculate_stochastic_factor(state)
    expected = 1 - state.norm()
    assert np.isclose(factor, expected), "Stochastic factor does not match expectation."


def test_create_probability_distribution_no_noise() -> None:
    """Test probability distribution is empty when no noise model is provided.

    This test passes an empty noise model to `create_probability_distribution` and verifies
    that the resulting jump dictionary contains only empty lists for all fields, confirming
    correct behavior for noiseless systems.
    """
    state = random_mps([(2, 1, 2), (2, 2, 2), (2, 2, 1)])
    noise_model = NoiseModel([])
    dt = 0.1
    sim_params = AnalogSimParams(observables=[], elapsed_time=0.0, max_bond_dim=5, threshold=1e-10)
    out = create_probability_distribution(state, noise_model, dt, sim_params)
    assert all(len(v) == 0 for v in out.values()), "Output should be empty with no noise."


def test_create_probability_distribution_one_site() -> None:
    """Test probability distribution for a single 1-site jump operator.

    This test sets up a noise model with one local jump operator and checks that
    `create_probability_distribution` returns a distribution with one jump, correct site,
    correct probability normalization, and the correct strength.
    """
    state = random_mps([(2, 1, 2), (2, 2, 2), (2, 2, 1)])
    # Identity jump operator for simplicity
    id_op = np.eye(2)
    noise_model = NoiseModel([
        {"name": "relaxation", "sites": [1], "strength": 0.5, "matrix": id_op},
    ])
    dt = 0.1
    sim_params = AnalogSimParams(observables=[], elapsed_time=0.0, max_bond_dim=5, threshold=1e-10)
    out = create_probability_distribution(state, noise_model, dt, sim_params)
    # One jump, one site
    assert len(out["jumps"]) == 1
    assert len(out["sites"]) == 1
    assert out["sites"][0] == [1]
    assert np.isclose(sum(out["probabilities"]), 1.0)
    assert out["strengths"][0] == 0.5


def test_stochastic_process_no_jump() -> None:
    """Test that stochastic_process returns the state unchanged if no jump occurs.

    This test applies `stochastic_process` with None type noise model,
    and verifies that the MPS is unchanged after the operation.
    """
    state = random_mps([(2, 1, 2), (2, 2, 2), (2, 2, 1)])
    noise_model = None
    dt = 0.1
    sim_params = AnalogSimParams(observables=[], elapsed_time=0.0, max_bond_dim=5, threshold=1e-10)
    new_state = stochastic_process(state, noise_model, dt, sim_params)
    # Should still be the same type
    assert isinstance(new_state, MPS)
    # Should not modify tensors (deepcopy not strictly guaranteed but should be unchanged)
    for a, b in zip(new_state.tensors, state.tensors):
        np.testing.assert_allclose(a, b)


def test_stochastic_process_jump() -> None:
    """Test that stochastic_process triggers a jump.

    This test that triggers a jump in `stochastic_process` by rescaling the first tensor, then
    verifies that the returned MPS differs from the original, confirming that a jump was applied.
    """
    state = random_mps([(2, 1, 2), (2, 2, 2), (2, 2, 1)])
    state.tensors[0] *= 0.99
    noise_model = NoiseModel([
        {"name": "bitflip", "sites": [0], "strength": 1000.0},
    ])
    dt = 0.1
    sim_params = AnalogSimParams(observables=[], elapsed_time=0.0, max_bond_dim=5, threshold=1e-10)
    state_copy = copy.deepcopy(state)
    new_state = stochastic_process(state_copy, noise_model, dt, sim_params)
    # Should still be the same type
    assert isinstance(new_state, MPS)
    # Check that at least one tensor changed (jump applied)
    different = any(not np.allclose(a, b) for a, b in zip(new_state.tensors, state.tensors))
    assert different, "At least one tensor should have changed after jump."


def test_create_probability_distribution_two_site() -> None:
    """Test probability distribution for a single 2-site jump operator.

    This test uses a noise model containing a single two-site jump process and checks
    that `create_probability_distribution` produces a dictionary with one jump on the
    correct pair of sites and a normalized probability.
    """
    state = random_mps([(2, 1, 2), (2, 2, 2), (2, 2, 1)])
    # 2x2 identity operator (for simplicity, but normally should be 4x4, depends on your merge op!)
    np.eye(2)
    noise_model = NoiseModel([
        {"name": "crosstalk_xx", "sites": [0, 1], "strength": 0.2},
    ])
    dt = 0.1
    sim_params = AnalogSimParams(observables=[], elapsed_time=0.0, max_bond_dim=5, threshold=1e-10)
    out = create_probability_distribution(state, noise_model, dt, sim_params)
    assert len(out["jumps"]) == 1
    assert out["sites"][0] == [0, 1]
    assert np.isclose(sum(out["probabilities"]), 1.0)
