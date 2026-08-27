# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the stochastic_process module."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import numpy as np
import opt_einsum as oe
import pytest

from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.core.methods.decompositions import merge_two_site, split_two_site
from mqt.yaqs.core.methods.stochastic_process import (
    calculate_stochastic_factor,
    create_probability_distribution,
    stochastic_process,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

rng = np.random.default_rng()


def _always_jump_rng() -> np.random.Generator:
    """RNG stub that always triggers a jump and selects process index 0.

    Returns:
        A minimal RNG-like object for ``stochastic_process`` jump tests.
    """

    class _AlwaysJumpRng:
        @staticmethod
        def random() -> float:
            return 0.0

        @staticmethod
        def choice(size: int, p: list[float]) -> int:
            _ = (size, p)
            return 0

    return cast("np.random.Generator", _AlwaysJumpRng())


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
    # 1 / sqrt(2) is a normalization factor
    return np.asarray((rng.standard_normal(size) + 1j * rng.standard_normal(size)) / np.sqrt(2), dtype=np.complex128)


def random_mps(
    shapes: list[tuple[int, int, int]],
    *,
    normalize: bool = True,
    seed: np.random.Generator | int | None = None,
) -> MPS:
    """Create a random MPS with the given shapes.

    Args:
        shapes (List[Tuple[int, int, int]]): The shapes of the tensors in the
            MPS.
        normalize (bool): Whether to normalize the MPS.
        seed: Optional seed or generator for reproducible tensors.

    Returns:
        MPS: The random MPS.
    """
    rng = np.random.default_rng(seed)
    tensors = [crandn(shape, seed=rng) for shape in shapes]
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
    (1 minus the actual squared norm of the first site).
    """
    state = MPS(3, state="zeros")
    state.tensors[0] *= 0.8
    factor = calculate_stochastic_factor(state)
    expected = 1.0 - state.norm() ** 2
    assert np.isclose(factor, expected), "Stochastic factor does not match expectation."
    # Residual norm 0.8 ⇒ ||psi||^2 = 0.64 ⇒ MCWF factor 0.36 (previous squared-norm API).
    assert float(factor) == pytest.approx(0.36, abs=1e-12)


def test_adjacent_jump_weight_unknown_gauge_uses_euclidean_norm_squared() -> None:
    """Unknown-gauge adjacent weights match ``||L|psi>||^2`` via ``norm() ** 2``."""
    state = MPS(2, state="zeros")
    state.set_center(None)
    two_i = 2.0 * np.eye(4, dtype=np.complex128)
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)

    # Reproduce the unknown-gauge branch: untruncated split, then global ||ψ'||^2.
    merged = oe.contract("ab, bcd->acd", two_i, merge_two_site(state.tensors[0], state.tensors[1]))
    left, right = split_two_site(
        merged,
        [2, 2],
        svd_distribution="right",
        trunc_mode=sim_params.trunc_mode,
        threshold=0.0,
        max_bond_dim=None,
    )
    jumped = copy.deepcopy(state)
    jumped.tensors[0] = left
    jumped.tensors[1] = right
    jumped.set_center(None)
    assert float(jumped.norm() ** 2) == pytest.approx(4.0, abs=1e-12)

    noise_model = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 1.0},
        {"name": "scaled_i", "sites": [0, 1], "strength": 1.0, "matrix": two_i},
    ])
    _procs, probabilities = create_probability_distribution(state, noise_model, dt=1.0, sim_params=sim_params)
    np.testing.assert_allclose(probabilities, [0.2, 0.8], atol=1e-10)


def test_create_probability_distribution_no_noise() -> None:
    """Test probability distribution is empty when no noise model is provided.

    This test passes an empty noise model to `create_probability_distribution` and verifies
    that the resulting tuple contains empty lists for both processes and probabilities,
    confirming correct behavior for noiseless systems.
    """
    state = random_mps([(2, 1, 2), (2, 2, 2), (2, 2, 1)])
    noise_model = NoiseModel([])
    dt = 0.1
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)
    ordered_processes, probabilities = create_probability_distribution(state, noise_model, dt, sim_params)
    assert len(ordered_processes) == 0, "No processes should be computed with empty noise model."
    assert len(probabilities) == 0, "No probabilities should be computed with empty noise model."


def test_create_probability_distribution_one_site() -> None:
    """Test probability distribution for a single 1-site jump operator.

    This test sets up a noise model with one local jump operator and checks that
    `create_probability_distribution` returns one applicable process with correct site,
    correct probability normalization, and the correct strength.
    """
    state = random_mps([(2, 1, 2), (2, 2, 2), (2, 2, 1)])
    # Identity jump operator for simplicity
    id_op = np.eye(2)
    noise_model = NoiseModel([
        {"name": "lowering", "sites": [1], "strength": 0.5, "matrix": id_op},
    ])
    dt = 0.1
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)
    ordered_processes, probabilities = create_probability_distribution(state, noise_model, dt, sim_params)
    # One applicable process
    assert len(ordered_processes) == 1
    assert len(probabilities) == 1
    # Check process properties
    process = ordered_processes[0]
    assert process["sites"] == [1]
    assert process["strength"] == pytest.approx(0.5)
    # Check probability normalization
    assert np.isclose(sum(probabilities), 1.0)


def test_stochastic_process_no_jump() -> None:
    """Test that stochastic_process returns the state unchanged if no jump occurs.

    This test applies `stochastic_process` with None type noise model,
    and verifies that the MPS is unchanged after the operation.
    """
    state = random_mps([(2, 1, 2), (2, 2, 2), (2, 2, 1)])
    noise_model = None
    dt = 0.1
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)
    new_state = stochastic_process(state, noise_model, dt, sim_params)
    # Should still be the same type
    assert isinstance(new_state, MPS)
    # Should not modify tensors (deepcopy not strictly guaranteed but should be unchanged)
    for a, b in zip(new_state.tensors, state.tensors, strict=False):
        np.testing.assert_allclose(a, b)


def test_stochastic_process_jump() -> None:
    """Test that stochastic_process triggers a jump.

    This test that triggers a jump in `stochastic_process` by rescaling the first tensor, then
    verifies that the returned MPS differs from the original, confirming that a jump was applied.
    """
    state = random_mps([(2, 1, 2), (2, 2, 2), (2, 2, 1)])
    state.tensors[0] *= 0.99
    noise_model = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 1000.0},
    ])
    dt = 0.1
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)
    state_copy = copy.deepcopy(state)

    new_state = stochastic_process(
        state_copy,
        noise_model,
        dt,
        sim_params,
        rng=_always_jump_rng(),
    )
    # Should still be the same type
    assert isinstance(new_state, MPS)
    # Check that at least one tensor changed (jump applied)
    different = any(not np.allclose(a, b) for a, b in zip(new_state.tensors, state.tensors, strict=False))
    assert different, "At least one tensor should have changed after jump."
    assert new_state.orthogonality_center == 0


def test_create_probability_distribution_two_site() -> None:
    """Test probability distribution for a single 2-site jump operator.

    This test uses a noise model containing a single two-site jump process and checks
    that `create_probability_distribution` produces one applicable process on the
    correct pair of sites and a normalized probability.
    """
    state = random_mps([(2, 1, 2), (2, 2, 2), (2, 2, 1)])
    # 2x2 identity operator (for simplicity, but normally should be 4x4, depends on your merge op!)
    np.eye(2)
    noise_model = NoiseModel([
        {"name": "crosstalk_xx", "sites": [0, 1], "strength": 0.2},
    ])
    dt = 0.1
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)
    ordered_processes, probabilities = create_probability_distribution(state, noise_model, dt, sim_params)
    # One applicable process
    assert len(ordered_processes) == 1
    assert len(probabilities) == 1
    # Check process properties
    process = ordered_processes[0]
    assert process["sites"] == [0, 1]
    assert process["strength"] == pytest.approx(0.2)
    # Check probability normalization
    assert np.isclose(sum(probabilities), 1.0)


def test_create_probability_distribution_adjacent_non_pauli_two_site() -> None:
    """Adjacent non-Pauli two-site processes contribute normalized probabilities."""
    state = random_mps([(2, 1, 2), (2, 2, 2), (2, 2, 1)])
    lowering_left = np.kron(np.array([[0, 0], [1, 0]], dtype=np.complex128), np.eye(2))
    noise_model = NoiseModel([
        {"name": "custom_2site", "sites": [0, 1], "strength": 0.3, "matrix": lowering_left},
    ])
    dt = 0.1
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)
    ordered_processes, probabilities = create_probability_distribution(state, noise_model, dt, sim_params)
    assert len(ordered_processes) == 1
    assert len(probabilities) == 1
    assert np.isclose(sum(probabilities), 1.0)


def test_adjacent_non_pauli_pdf_matches_exact_weights() -> None:
    """Equal-rate X@0 and 2I@[0,1] yield PDF weights proportional to operator norms."""
    # Product |0>: ||X|0>||^2 = 1, ||(2I)|00>||^2 = 4 → weights 1:4 → [0.2, 0.8]
    state = MPS(2, state="zeros")
    two_i = 2.0 * np.eye(4, dtype=np.complex128)
    merged = oe.contract("ab, bcd->acd", two_i, merge_two_site(state.tensors[0], state.tensors[1]))
    assert float(np.vdot(merged, merged).real) == pytest.approx(4.0)
    noise_model = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 1.0},
        {"name": "scaled_i", "sites": [0, 1], "strength": 1.0, "matrix": two_i},
    ])
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)
    _procs, probabilities = create_probability_distribution(state, noise_model, dt=1.0, sim_params=sim_params)
    assert len(probabilities) == 2
    np.testing.assert_allclose(probabilities, [0.2, 0.8], atol=1e-10)


def test_adjacent_pdf_independent_of_max_bond_dim() -> None:
    """PDF weights use the untruncated post-jump block, not a truncated split."""
    # Both operators map |00> with ||L|psi>||^2 = 1, but only the Bell channel needs chi>1.
    # Truncating before the norm would bias the PDF to [2/3, 1/3].
    xx = np.kron(np.array([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]])).astype(np.complex128)
    bell_create = np.zeros((4, 4), dtype=np.complex128)
    bell_create[:, 0] = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
    bell_create[:, 1] = [0, 1, 0, 0]
    bell_create[:, 2] = [0, 0, 1, 0]
    bell_create[:, 3] = [1 / np.sqrt(2), 0, 0, -1 / np.sqrt(2)]
    state = MPS(2, state="zeros")
    noise_model = NoiseModel([
        {"name": "xx", "sites": [0, 1], "strength": 1.0, "matrix": xx},
        {"name": "bell", "sites": [0, 1], "strength": 1.0, "matrix": bell_create},
    ])
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0, max_bond_dim=1)
    _procs, probabilities = create_probability_distribution(state, noise_model, dt=1.0, sim_params=sim_params)
    np.testing.assert_allclose(probabilities, [0.5, 0.5], atol=1e-10)


def test_adjacent_pdf_unknown_gauge_uses_global_norm() -> None:
    """With a genuinely noncanonical MPS, adjacent PDF weights match global post-jump norms."""
    # Random tensors + non-unitary bond gauge (not merely canonical + set_center(None)).
    state = random_mps([(2, 1, 2), (2, 2, 2), (2, 2, 1)], normalize=False, seed=20260804)
    gauge = np.array([[1.5, 0.4], [0.0, 0.7]], dtype=np.complex128)
    state.tensors[0] = np.einsum("ijk,kl->ijl", state.tensors[0], gauge)
    state.tensors[1] = np.einsum("ij,jkl->ikl", np.linalg.inv(gauge), state.tensors[1])
    state.set_center(None)
    assert state.orthogonality_center is None

    two_i = 2.0 * np.eye(4, dtype=np.complex128)
    # Local Frobenius shortcuts disagree with the global norm under this gauge.
    global_norm_sq = float(state.norm() ** 2)
    local_t0 = float(np.vdot(state.tensors[0], state.tensors[0]).real)
    merged_2i = oe.contract("ab, bcd->acd", two_i, merge_two_site(state.tensors[0], state.tensors[1]))
    assert not np.isclose(local_t0, global_norm_sq, atol=1e-8)
    assert not np.isclose(float(np.vdot(merged_2i, merged_2i).real), 4.0 * global_norm_sq, atol=1e-8)

    noise_model = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 1.0},
        {"name": "scaled_i", "sites": [0, 1], "strength": 1.0, "matrix": two_i},
    ])
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)
    tensors_before = [t.copy() for t in state.tensors]
    _procs, probabilities = create_probability_distribution(state, noise_model, dt=1.0, sim_params=sim_params)
    # Pauli unitary: weight ∝ ||ψ||^2; 2I: weight ∝ 4||ψ||^2 → [0.2, 0.8] via global norms.
    np.testing.assert_allclose(probabilities, [0.2, 0.8], atol=1e-10)
    assert state.orthogonality_center is None
    for before, after in zip(tensors_before, state.tensors, strict=True):
        np.testing.assert_array_equal(before, after)


def test_zero_probability_weight_raises() -> None:
    """Non-finite/zero total jump weight raises a clear ValueError."""
    state = MPS(1, state="zeros")
    # Dissipate then jump with enormous rate so post-jump norms underflow.
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 2000.0}])
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)
    # Apply a strong dissipator-like shrink so stochastic factor triggers with empty weights.
    state.tensors[0] *= 0.0
    with pytest.raises(ValueError, match="zero or non-finite"):
        create_probability_distribution(state, noise_model, dt=1.0, sim_params=sim_params)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_probability_weight_raises(bad: float) -> None:
    """NaN and Inf jump weights raise the non-finite guard."""
    state = MPS(1, state="zeros")
    state.normalize("B")
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 1.0}])
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)
    # Inject the non-finite weight directly: planting 1e200 overflows in the norm
    # contraction and emits a platform-dependent RuntimeWarning under BLAS.
    with patch.object(MPS, "norm", return_value=bad), pytest.raises(ValueError, match="zero or non-finite"):
        create_probability_distribution(state, noise_model, dt=1.0, sim_params=sim_params)


def test_stochastic_process_jump_independent_of_process_order() -> None:
    """Jump application is independent of NoiseModel process list order."""
    dt = 0.1
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)

    def make_noise(order: str) -> NoiseModel:
        lowerings = [{"name": "lowering", "sites": [q], "strength": 0.6} for q in range(3)]
        dephasings = [{"name": "pauli_z", "sites": [q], "strength": 0.3} for q in range(3)]
        if order == "grouped":
            processes = lowerings + dephasings
        else:
            processes = [proc for q in range(3) for proc in (lowerings[q], dephasings[q])]
        return NoiseModel(processes)

    class _JumpAtIndex1:
        @staticmethod
        def random() -> float:
            return 0.0

        @staticmethod
        def choice(size: int, p: list[float]) -> int:
            _ = (size, p)
            return 1  # site-sweep index 1: pauli_z@0

    base = random_mps([(2, 1, 2), (2, 2, 2), (2, 2, 1)])
    base.tensors[0] *= 0.99  # non-unit norm so a jump is triggered
    grouped = stochastic_process(
        copy.deepcopy(base),
        make_noise("grouped"),
        dt,
        sim_params,
        rng=cast("np.random.Generator", _JumpAtIndex1()),
    )
    site_major = stochastic_process(
        copy.deepcopy(base),
        make_noise("site_major"),
        dt,
        sim_params,
        rng=cast("np.random.Generator", _JumpAtIndex1()),
    )
    for a, b in zip(grouped.tensors, site_major.tensors, strict=False):
        np.testing.assert_allclose(a, b)


def test_stochastic_process_jump_independent_of_process_order_mixed_channels() -> None:
    """Jump application is independent of list order for mixed 1- and 2-site channels."""
    num_qubits = 3
    noise_factor = 0.01
    dt = 0.1
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)

    def make_noise(order: str) -> NoiseModel:
        one_site = [{"name": "pauli_x", "sites": [i], "strength": noise_factor} for i in range(num_qubits)] + [
            {"name": "pauli_y", "sites": [i], "strength": noise_factor} for i in range(num_qubits)
        ]
        two_site = [
            {"name": "crosstalk_xx", "sites": [i, i + 1], "strength": noise_factor} for i in range(num_qubits - 1)
        ] + [{"name": "crosstalk_yy", "sites": [i, i + 1], "strength": noise_factor} for i in range(num_qubits - 1)]
        if order == "grouped":
            processes = list(one_site) + list(two_site)
        else:
            processes = []
            for i in range(num_qubits):
                processes.append({"name": "pauli_x", "sites": [i], "strength": noise_factor})
                processes.append({"name": "pauli_y", "sites": [i], "strength": noise_factor})
                if i < num_qubits - 1:
                    processes.append({"name": "crosstalk_xx", "sites": [i, i + 1], "strength": noise_factor})
                    processes.append({"name": "crosstalk_yy", "sites": [i, i + 1], "strength": noise_factor})
        return NoiseModel(processes)

    class _JumpAtIndex1:
        @staticmethod
        def random() -> float:
            return 0.0

        @staticmethod
        def choice(size: int, p: list[float]) -> int:
            _ = (size, p)
            return 1  # site-sweep index 1: pauli_y@0

    base = random_mps([(2, 1, 2), (2, 2, 2), (2, 2, 1)])
    base.tensors[0] *= 0.99
    grouped = stochastic_process(
        copy.deepcopy(base),
        make_noise("grouped"),
        dt,
        sim_params,
        rng=cast("np.random.Generator", _JumpAtIndex1()),
    )
    site_major = stochastic_process(
        copy.deepcopy(base),
        make_noise("site_major"),
        dt,
        sim_params,
        rng=cast("np.random.Generator", _JumpAtIndex1()),
    )
    for a, b in zip(grouped.tensors, site_major.tensors, strict=False):
        np.testing.assert_allclose(a, b)


def test_stochastic_process_no_jump_unknown_gauge() -> None:
    """A no-jump path with unknown gauge re-canonicalizes the MPS at site 0."""
    state = random_mps([(2, 1, 2), (2, 2, 2), (2, 2, 1)])
    state.set_center(None)
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)

    new_state = stochastic_process(state, None, dt=0.1, sim_params=sim_params)

    assert new_state.orthogonality_center == 0


def test_stochastic_process_empty_probabilities_after_jump() -> None:
    """A triggered jump with no applicable processes recenters at site 0 without applying a jump."""
    state = random_mps([(2, 1, 2), (2, 2, 2), (2, 2, 1)])
    state.tensors[0] *= 0.99
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 0.0}])
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)
    state_copy = copy.deepcopy(state)

    with patch(
        "mqt.yaqs.core.methods.stochastic_process.create_probability_distribution",
        return_value=([], []),
    ):
        new_state = stochastic_process(
            state_copy,
            noise_model,
            0.1,
            sim_params,
            rng=_always_jump_rng(),
        )
    assert new_state.orthogonality_center == 0


def test_stochastic_process_empty_probabilities_unknown_gauge() -> None:
    """Empty jump probabilities canonicalize at site 0 when the gauge is unknown."""
    state = random_mps([(2, 1, 2), (2, 2, 2), (2, 2, 1)])
    state.tensors[0] *= 0.99
    state.set_center(None)
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 0.0}])
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)

    with patch(
        "mqt.yaqs.core.methods.stochastic_process.create_probability_distribution",
        return_value=([], []),
    ):
        new_state = stochastic_process(
            state,
            noise_model,
            0.1,
            sim_params,
            rng=_always_jump_rng(),
        )
    assert new_state.orthogonality_center == 0


def test_create_probability_distribution_non_pauli_longrange_raises() -> None:
    """Non-Pauli long-range processes raise instead of being omitted from the jump PDF."""
    state = random_mps([(2, 1, 2), (2, 2, 2), (2, 2, 1)])
    lowering = np.array([[0, 0], [1, 0]], dtype=np.complex128)
    noise_model = NoiseModel([
        {"name": "custom_lr", "sites": [0, 2], "strength": 0.1, "factors": (lowering, lowering)},
    ])
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)
    with pytest.raises(NotImplementedError, match="Non-Pauli long-range"):
        create_probability_distribution(state, noise_model, 0.1, sim_params)


def test_stochastic_process_longrange_crosstalk_xy_jump() -> None:
    """Documented longrange_crosstalk_xy is treated as Pauli and can jump."""
    state = random_mps([(2, 1, 2), (2, 2, 2), (2, 2, 1)])
    state.tensors[0] *= 0.99
    noise_model = NoiseModel([
        {"name": "longrange_crosstalk_xy", "sites": [0, 2], "strength": 1000.0},
    ])
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)
    state_copy = copy.deepcopy(state)

    new_state = stochastic_process(
        state_copy,
        noise_model,
        0.1,
        sim_params,
        rng=_always_jump_rng(),
    )
    assert new_state.orthogonality_center == 0


def test_stochastic_process_adjacent_non_pauli_two_site_jump() -> None:
    """Stochastic jumps support adjacent non-Pauli two-site processes."""
    state = random_mps([(2, 1, 2), (2, 2, 2), (2, 2, 1)])
    state.tensors[0] *= 0.99
    lowering_left = np.kron(np.array([[0, 0], [1, 0]], dtype=np.complex128), np.eye(2))
    noise_model = NoiseModel([
        {"name": "custom_2site", "sites": [0, 1], "strength": 1000.0, "matrix": lowering_left},
    ])
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)
    state_copy = copy.deepcopy(state)

    new_state = stochastic_process(
        state_copy,
        noise_model,
        0.1,
        sim_params,
        rng=_always_jump_rng(),
    )
    assert new_state.orthogonality_center == 0
    assert any(not np.allclose(a, b) for a, b in zip(new_state.tensors, state.tensors, strict=False))


def test_stochastic_process_longrange_pauli_jump() -> None:
    """Long-range Pauli crosstalk jumps apply per-site factors and clear the gauge."""
    state = random_mps([(2, 1, 2), (2, 2, 2), (2, 2, 1)])
    state.tensors[0] *= 0.99
    noise_model = NoiseModel([
        {"name": "crosstalk_xx", "sites": [0, 2], "strength": 1000.0},
    ])
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)
    state_copy = copy.deepcopy(state)

    new_state = stochastic_process(
        state_copy,
        noise_model,
        0.1,
        sim_params,
        rng=_always_jump_rng(),
    )
    assert new_state.orthogonality_center == 0


def test_stochastic_process_non_adjacent_non_pauli_jump_raises() -> None:
    """Non-Pauli long-range two-site jumps are rejected during application."""
    state = random_mps([(2, 1, 2), (2, 2, 2), (2, 2, 1)])
    state.tensors[0] *= 0.99
    lowering_left = np.kron(np.array([[0, 0], [1, 0]], dtype=np.complex128), np.eye(2))
    noise_model = NoiseModel([
        {"name": "custom_2site", "sites": [0, 1], "strength": 1000.0, "matrix": lowering_left},
    ])
    noise_model.processes[0]["sites"] = [0, 2]
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)
    state_copy = copy.deepcopy(state)

    with (
        patch(
            "mqt.yaqs.core.methods.stochastic_process.create_probability_distribution",
            return_value=([noise_model.processes[0]], [1.0]),
        ),
        pytest.raises(ValueError, match="nearest-neighbor"),
    ):
        stochastic_process(
            state_copy,
            noise_model,
            0.1,
            sim_params,
            rng=_always_jump_rng(),
        )
