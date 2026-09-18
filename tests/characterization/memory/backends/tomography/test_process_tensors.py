# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff:file-ignore[private-member-access] -- white-box tests exercise private process-tensor prediction helpers

"""Tests for DenseProcessTensor and MPOProcessTensor wrappers."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer
from mqt.yaqs.characterization.memory.backends.exact import simulate_exact
from mqt.yaqs.characterization.memory.backends.tomography.constructor import build_process_tensor
from mqt.yaqs.characterization.memory.backends.tomography.process_tensors import (
    DenseProcessTensor,
    MPOProcessTensor,
    compute_entropy_dense,
    compute_temporal_entropy,
    convert_probe_callable,
    encode_map_choi,
    evaluate_probes,
    trace_partial_dense,
)
from mqt.yaqs.characterization.memory.operational_memory.grid import assemble_probe_sequence
from mqt.yaqs.characterization.memory.operational_memory.samples import sample_probes
from mqt.yaqs.characterization.memory.shared.encoding import encode_rho_pauli
from mqt.yaqs.characterization.memory.shared.intervention_steps import build_intervention_operator
from mqt.yaqs.characterization.memory.shared.interventions import InterventionMap
from mqt.yaqs.core.data_structures.mpo import MPO

_REF_RHO0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)


def _two_swap_process_matrix() -> np.ndarray:
    """Return the exact two-leg process that stores and returns the first output."""
    identity_ket = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128)
    identity_choi = np.outer(identity_ket, identity_ket.conj())
    return np.kron(np.kron(np.kron(identity_choi, _REF_RHO0), np.eye(2)), _REF_RHO0)


def _classical_memory_process_matrix() -> np.ndarray:
    """Return a causal two-leg process with one bit of past-final correlation."""
    diagonal = np.zeros((2, 2, 2, 2, 2), dtype=np.complex128)
    for first_input, first_output, last_output, last_input in np.ndindex(2, 2, 2, 2):
        diagonal[first_input, first_output, first_input, last_output, last_input] = 0.25
    return np.diag(diagonal.reshape(-1))


def _assert_causally_normalized(upsilon: np.ndarray, num_interventions: int) -> None:
    """Check deterministic-comb trace constraints without production helpers."""
    current = np.asarray(upsilon, dtype=np.complex128)
    np.testing.assert_allclose(np.trace(current), 2**num_interventions, atol=1e-10)

    for leg in range(num_interventions, 0, -1):
        earlier_dimension = 4 ** (leg - 1)
        past_dimension = 4**leg
        matrix = current.reshape(2, past_dimension, 2, past_dimension)
        traced_final = matrix[0, :, 0, :] + matrix[1, :, 1, :]
        unfused = traced_final.reshape(earlier_dimension, 2, 2, earlier_dimension, 2, 2)

        previous_tail = np.zeros(
            (earlier_dimension, 2, earlier_dimension, 2),
            dtype=np.complex128,
        )
        for output in range(2):
            previous_tail += 0.5 * unfused[:, output, :, :, output, :]
        for output in range(2):
            for output_bra in range(2):
                expected = previous_tail if output == output_bra else np.zeros_like(previous_tail)
                np.testing.assert_allclose(
                    unfused[:, output, :, :, output_bra, :],
                    expected,
                    atol=1e-10,
                )

        current = previous_tail.transpose(1, 0, 3, 2).reshape(
            2 * earlier_dimension,
            2 * earlier_dimension,
        )

    np.testing.assert_allclose(np.trace(current), 1.0, atol=1e-10)


def _tiny_mpo_process_tensor(*, num_interventions: int = 1) -> MPOProcessTensor:
    """Build a noiseless direct MPO process tensor for wrapper unit tests.

    Returns:
        MPO process tensor with the requested number of intervention legs.
    """
    ham = Hamiltonian.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8)
    timesteps = [0.0] * (num_interventions + 1)
    return cast(
        "MPOProcessTensor",
        build_process_tensor(
            ham.mpo,
            params,
            timesteps=timesteps,
            return_type="mpo",
            compress_every=1,
        ),
    )


def _single_site_mpo_pt(rho: np.ndarray) -> MPOProcessTensor:
    """Wrap a single-site output density matrix as a zero-intervention MPO PT.

    Returns:
        MPO process tensor whose only site encodes ``rho``.
    """
    tensors = [np.asarray(rho, dtype=np.complex128).reshape(2, 2, 1, 1)]
    mpo = MPO()
    mpo.custom(tensors, transpose=False)
    return MPOProcessTensor(mpo, [], initial_rho=_REF_RHO0.copy())


def _unchecked_mpo(tensors: list[np.ndarray], *, length: int | None = None) -> MPO:
    """Build an MPO without invoking the generic constructor assertions.

    Returns:
        MPO with the supplied raw tensor metadata.
    """
    mpo = MPO()
    mpo.tensors = tensors
    mpo.length = len(tensors) if length is None else length
    mpo.physical_dimension = int(tensors[0].shape[0]) if tensors else 0
    return mpo


def test_dense_process_tensor_predict_matches_branch_contraction() -> None:
    """The dense branch contraction matches the Choi contraction; predict normalizes."""
    ups = 0.25 * np.eye(2 * 4, dtype=np.complex128)
    timesteps = [0.1, 0.1]

    def id_map(rho: np.ndarray) -> np.ndarray:
        return rho

    pt = DenseProcessTensor(ups, timesteps)
    # Identity map Choi has trace 2; the scaled identity process gives rho = I/2.
    rho_branch = pt._contract_subnormalized_branch([id_map])
    np.testing.assert_allclose(rho_branch, 0.5 * np.eye(2, dtype=np.complex128), atol=1e-12)
    rho = pt.predict([id_map])
    np.testing.assert_allclose(np.trace(rho), 1.0, atol=1e-12)
    np.testing.assert_allclose(rho, rho_branch / np.trace(rho_branch), atol=1e-12)


def test_dense_process_tensor_predict_raises_on_length_mismatch() -> None:
    """DenseProcessTensor.predict rejects intervention lists whose length mismatches num_interventions."""
    ups = np.eye(2 * 4, dtype=np.complex128)
    pt = DenseProcessTensor(ups, [0.1, 0.1])

    def id_map(rho: np.ndarray) -> np.ndarray:
        return rho

    with pytest.raises(ValueError, match="DenseProcessTensor expects"):
        pt.predict([id_map, id_map])


@pytest.mark.parametrize(
    ("upsilon", "message"),
    [
        (np.zeros((2, 2, 1), dtype=np.complex128), "rank-2 matrix"),
        (np.zeros((2, 3), dtype=np.complex128), "must be square"),
        (np.eye(4, dtype=np.complex128), r"dimension must equal 2 \* 4\*\*k"),
        (np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=np.complex128), "only finite values"),
    ],
)
def test_dense_process_tensor_rejects_malformed_matrix(upsilon: np.ndarray, message: str) -> None:
    """Dense process tensors require a finite square matrix with an exact causal-leg dimension."""
    with pytest.raises(ValueError, match=message):
        DenseProcessTensor(upsilon, [])


def test_dense_process_tensor_requires_schedule_matching_intervention_legs() -> None:
    """The dense matrix dimension and schedule must encode the same intervention count."""
    upsilon = np.eye(8, dtype=np.complex128)
    with pytest.raises(ValueError, match="1 intervention legs requires 2 timesteps, got 1"):
        DenseProcessTensor(upsilon, [0.1])

    pt = DenseProcessTensor(upsilon, [0.1, 0.2])
    assert pt._num_interventions() == 1


def test_compute_entropy_dense_rejects_invalid_base() -> None:
    """Entropy helpers reject non-positive bases and base equal to 1."""
    rho = np.eye(2, dtype=np.complex128) * 0.5
    with pytest.raises(ValueError, match="entropy base"):
        compute_entropy_dense(rho, base=1)
    with pytest.raises(ValueError, match="entropy base"):
        compute_entropy_dense(rho, base=0)


def test_mpo_process_tensor_dense_and_sparse_use_causal_leg_order() -> None:
    """Process-tensor conversions keep the final output leg first."""
    rho_final = np.diag([1.0, 2.0]).astype(np.complex128)
    dual_operator = np.diag([3.0, 4.0, 5.0, 6.0]).astype(np.complex128)
    mpo = MPO()
    mpo.custom(
        [
            rho_final.reshape(2, 2, 1, 1),
            dual_operator.reshape(4, 4, 1, 1),
        ],
        transpose=False,
    )
    pt = MPOProcessTensor(mpo, [0.1, 0.1])
    expected = np.kron(rho_final, dual_operator)

    np.testing.assert_allclose(pt.to_matrix(), expected, atol=1e-12)
    np.testing.assert_allclose(pt.to_sparse_matrix().toarray(), expected, atol=1e-12)
    np.testing.assert_allclose(mpo.to_matrix(), np.kron(dual_operator, rho_final), atol=1e-12)


def test_mpo_process_tensor_requires_schedule_matching_intervention_legs() -> None:
    """The MPO site count and schedule must encode the same intervention count."""
    mpo = _unchecked_mpo([
        np.zeros((2, 2, 1, 1), dtype=np.complex128),
        np.zeros((4, 4, 1, 1), dtype=np.complex128),
    ])

    with pytest.raises(ValueError, match="1 intervention legs requires 2 timesteps, got 1"):
        MPOProcessTensor(mpo, [0.1])

    pt = MPOProcessTensor(mpo, [0.1, 0.2])
    assert pt.length == 2
    assert pt.timesteps == [0.1, 0.2]


@pytest.mark.parametrize(
    ("tensors", "length", "timesteps", "message"),
    [
        ([], 0, [], "at least one tensor"),
        ([np.zeros((2, 2, 1, 1), dtype=np.complex128)], 2, [], "length metadata"),
        ([np.zeros((2, 2, 1), dtype=np.complex128)], 1, [], "must be rank 4"),
        ([np.zeros((3, 3, 1, 1), dtype=np.complex128)], 1, [], "site 0 must have physical dimensions"),
        (
            [
                np.zeros((2, 2, 1, 1), dtype=np.complex128),
                np.zeros((2, 2, 1, 1), dtype=np.complex128),
            ],
            2,
            [0.1, 0.1],
            "site 1 must have physical dimensions",
        ),
        ([np.zeros((2, 2, 0, 1), dtype=np.complex128)], 1, [], "positive bond dimensions"),
        ([np.zeros((2, 2, 2, 1), dtype=np.complex128)], 1, [], "unit left and right boundary bonds"),
        ([np.zeros((2, 2, 1, 2), dtype=np.complex128)], 1, [], "unit left and right boundary bonds"),
        (
            [
                np.zeros((2, 2, 1, 2), dtype=np.complex128),
                np.zeros((4, 4, 3, 1), dtype=np.complex128),
            ],
            2,
            [0.1, 0.1],
            "bond mismatch",
        ),
        ([np.full((2, 2, 1, 1), np.nan, dtype=np.complex128)], 1, [], "only finite values"),
    ],
)
def test_mpo_process_tensor_rejects_malformed_structure(
    tensors: list[np.ndarray],
    length: int,
    timesteps: list[float],
    message: str,
) -> None:
    """MPO process tensors require finite causal sites with valid open bonds."""
    with pytest.raises(ValueError, match=message):
        MPOProcessTensor(_unchecked_mpo(tensors, length=length), timesteps)


def test_mpo_process_tensor_qmi_fallback_to_dense() -> None:
    """MPOProcessTensor.qmi should agree with DenseProcessTensor.qmi via dense fallback."""
    pt = _tiny_mpo_process_tensor(num_interventions=2)
    dense = pt.to_dense()

    for past in ("all", "first", "last"):
        assert pt.qmi(past=past) == pytest.approx(dense.qmi(past=past), abs=1e-12)


def test_mpo_process_tensor_predict_smoke_identity_map() -> None:
    """MPOProcessTensor.predict returns a physical density matrix for a trivial intervention."""
    pt = _tiny_mpo_process_tensor(num_interventions=1)

    def id_map(x: np.ndarray) -> np.ndarray:
        return x

    rho_out = pt.predict([id_map])
    assert rho_out.shape == (2, 2)
    np.testing.assert_allclose(rho_out, rho_out.conj().T, atol=1e-12)
    np.testing.assert_allclose(np.trace(rho_out).real, 1.0, atol=1e-12)


def test_mpo_process_tensor_predict_raises_on_empty_interventions() -> None:
    """Predict rejects empty interventions when num_interventions>0."""
    pt = _tiny_mpo_process_tensor(num_interventions=1)
    with pytest.raises(ValueError, match="interventions list must be non-empty"):
        pt.predict([])


def test_mpo_process_tensor_predict_zero_steps() -> None:
    """MPOProcessTensor.predict([]) returns the stored output when num_interventions=0."""
    rho = np.array([[0.6, 0.1 + 0.0j], [0.1 - 0.0j, 0.4]], dtype=np.complex128)
    pt = _single_site_mpo_pt(rho)
    rho_out = pt.predict([])
    np.testing.assert_allclose(rho_out, rho, atol=1e-10)


def test_mpo_process_tensor_predict_raises_on_length_mismatch() -> None:
    """Predict rejects intervention lists whose length mismatches the process tensor."""
    pt = _tiny_mpo_process_tensor(num_interventions=1)

    def id_map(x: np.ndarray) -> np.ndarray:
        return x

    with pytest.raises(ValueError, match="MPOProcessTensor length"):
        pt.predict([id_map, id_map])


def _tiny_process_tensor(*, num_interventions: int) -> DenseProcessTensor:
    """Build a noiseless 1-site process tensor for wrapper unit tests.

    Returns:
        Dense process-tensor wrapper for a trivial 1-site Ising chain.
    """
    ham = Hamiltonian.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8)
    timesteps = [0.0] * (num_interventions + 1)
    return cast(
        "DenseProcessTensor",
        MemoryCharacterizer(parallel=False, show_progress=False).build_process_tensor(
            ham, params, timesteps=timesteps, return_type="dense"
        ),
    )


def test_build_intervention_operator_dict_variants() -> None:
    """Structured probe steps normalize to unitaries or intervention maps."""
    z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    x = np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
    u = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    u_out = build_intervention_operator({"type": "unitary", "U": u})
    assert isinstance(u_out, np.ndarray)
    np.testing.assert_allclose(cast("np.ndarray", u_out), u)

    mo = build_intervention_operator({"type": "cut_measurement", "psi_meas": x})
    assert isinstance(mo, InterventionMap)
    assert mo.effect.shape == (2, 2)

    po = build_intervention_operator({"type": "cut_preparation", "psi_prep": x})
    assert isinstance(po, InterventionMap)
    assert po.rho_prep.shape == (2, 2)
    np.testing.assert_allclose(po.effect, np.eye(2), atol=1e-12)

    mp = build_intervention_operator((x, z))
    assert isinstance(mp, InterventionMap)
    assert mp.effect.shape == (2, 2)

    with pytest.raises(ValueError, match="Unsupported probe step"):
        build_intervention_operator({"type": "nope"})


def test_cut_preparation_map_independent_of_input_state() -> None:
    """cut_preparation applies unconditional preparation, not a |0>-conditioned effect."""
    plus = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2)
    step_map = convert_probe_callable({"type": "cut_preparation", "psi_prep": plus})
    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    rho1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    out0 = step_map(rho0)
    out1 = step_map(rho1)
    target = np.outer(plus, plus.conj())
    np.testing.assert_allclose(out0, target, atol=1e-12)
    np.testing.assert_allclose(out1, target, atol=1e-12)


def test_dense_process_tensor_predict_zero_steps() -> None:
    """DenseProcessTensor.predict([]) returns the stored output state when num_interventions=0."""
    rho = np.array([[0.2, 0.1 + 0.1j], [0.1 - 0.1j, 0.8]], dtype=np.complex128)
    pt = DenseProcessTensor(rho.reshape(2, 2), timesteps=[])
    rho_out = pt.predict([])
    np.testing.assert_allclose(rho_out, rho, atol=1e-12)


def test_dense_process_tensor_qmi_zero_steps() -> None:
    """QMI is zero when the process tensor has no past intervention legs."""
    rho = np.array([[0.7, 0.0], [0.0, 0.3]], dtype=np.complex128)
    pt = DenseProcessTensor(rho.reshape(2, 2), timesteps=[])
    assert pt.qmi(past="all") == pytest.approx(0.0)
    assert pt.qmi(past="first") == pytest.approx(0.0)
    assert pt.qmi(past="last") == pytest.approx(0.0)


def test_convert_probe_callable_unitary_and_map() -> None:
    """Callable conversion covers unitary matrices and intervention maps."""
    u = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    rho = np.eye(2, dtype=np.complex128) * 0.5
    u_map = convert_probe_callable({"type": "unitary", "U": u})
    np.testing.assert_allclose(u_map(rho), u @ rho @ u.conj().T, atol=1e-12)

    z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    mp_map = convert_probe_callable((z, z))
    out = mp_map(rho)
    assert out.shape == (2, 2)


def test_encode_map_choi_uses_output_then_input_order() -> None:
    """Choi encoding uses the output-input order required by prediction."""

    def id_map(rho: np.ndarray) -> np.ndarray:
        return rho

    identity_ket = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128)
    np.testing.assert_allclose(encode_map_choi(id_map), np.outer(identity_ket, identity_ket.conj()), atol=1e-12)

    rho_one = np.diag([0.0, 1.0]).astype(np.complex128)

    def reset_one(rho: np.ndarray) -> np.ndarray:
        return np.trace(rho) * rho_one

    np.testing.assert_allclose(encode_map_choi(reset_one), np.kron(rho_one, np.eye(2)), atol=1e-12)


def test_trace_partial_dense_and_entropy_edge_cases() -> None:
    """Partial trace and entropy helpers cover validation and degenerate inputs."""
    rho = np.asarray(
        np.kron(np.eye(2, dtype=np.complex128), np.eye(2, dtype=np.complex128)) * 0.25,
        dtype=np.complex128,
    )
    reduced = trace_partial_dense(rho, dims=[2, 2], keep=[0])
    assert reduced.shape == (2, 2)

    with pytest.raises(ValueError, match="keep indices"):
        trace_partial_dense(rho, dims=[2, 2], keep=[3])

    with pytest.raises(ValueError, match="trace greater than"):
        compute_entropy_dense(np.zeros((2, 2), dtype=np.complex128))
    pure = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    assert compute_entropy_dense(pure) == pytest.approx(0.0, abs=1e-12)


def test_compute_entropy_dense_rejects_nonphysical_inputs() -> None:
    """Entropy rejects negative spectral weight and matrices with near-zero trace."""
    with pytest.raises(ValueError, match="positive semidefinite"):
        compute_entropy_dense(np.diag([1.2, -0.2]).astype(np.complex128))
    with pytest.raises(ValueError, match="trace greater than"):
        compute_entropy_dense(np.diag([1e-14, 0.0]).astype(np.complex128))


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.ones(2, dtype=np.complex128), "nonempty square rank-2 matrix"),
        (np.array([[1.0, np.nan], [0.0, 0.0]], dtype=np.complex128), "only finite values"),
        (np.array([[0.5, 0.2], [0.0, 0.5]], dtype=np.complex128), "must be Hermitian"),
    ],
)
def test_compute_entropy_dense_rejects_malformed_inputs(matrix: np.ndarray, message: str) -> None:
    """Entropy rejects malformed matrices before diagonalization."""
    with pytest.raises(ValueError, match=message):
        compute_entropy_dense(matrix)


def test_compute_entropy_dense_clips_roundoff_and_respects_dimension_bound() -> None:
    """Roundoff-scale negative eigenvalues are clipped and accepted entropy stays bounded."""
    almost_pure = np.diag([1.0 + 1e-12, -1e-12]).astype(np.complex128)
    assert compute_entropy_dense(almost_pure) == pytest.approx(0.0, abs=1e-12)

    dimension = 4
    entropy = compute_entropy_dense(np.eye(dimension, dtype=np.complex128))
    assert 0.0 <= entropy <= np.log2(dimension)
    assert entropy == pytest.approx(np.log2(dimension))


def test_information_metrics_reject_non_psd_process_tensor() -> None:
    """QMI and CMI validate the process tensor before returning trivial small-horizon values."""
    pt = DenseProcessTensor(np.diag([1.2, -0.2]).astype(np.complex128), timesteps=[])
    with pytest.raises(ValueError, match="Upsilon must be positive semidefinite"):
        pt.qmi()
    with pytest.raises(ValueError, match="Upsilon must be positive semidefinite"):
        pt.cmi()


def test_information_metrics_reject_psd_but_noncausal_process_tensor() -> None:
    """A positive matrix must also satisfy the recursive process-tensor trace constraints."""
    noncausal = np.zeros((8, 8), dtype=np.complex128)
    noncausal[0, 0] = 2.0
    pt = DenseProcessTensor(noncausal, timesteps=[0.0, 0.0])

    with pytest.raises(ValueError, match="causal normalization"):
        pt.qmi()
    with pytest.raises(ValueError, match="causal normalization"):
        pt.cmi()

    # This two-leg operator satisfies the outer constraint but reduces to the
    # invalid one-leg operator above, so recursion must catch the inner leg.
    recursive_diagonal = np.zeros((2, 2, 2, 2, 2), dtype=np.complex128)
    for last_output in range(2):
        recursive_diagonal[0, 0, 0, last_output, 0] = 2.0
    recursive_pt = DenseProcessTensor(np.diag(recursive_diagonal.reshape(-1)), [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match=r"intervention leg 1\b"):
        recursive_pt.qmi()


def test_information_metrics_reject_zero_trace_and_invalid_base_before_early_return() -> None:
    """Trivial horizons do not bypass process-tensor or entropy-base validation."""
    zero = DenseProcessTensor(np.zeros((2, 2), dtype=np.complex128), timesteps=[])
    with pytest.raises(ValueError, match="Upsilon must have trace greater than"):
        zero.qmi()
    with pytest.raises(ValueError, match="Upsilon must have trace greater than"):
        zero.cmi()

    valid = DenseProcessTensor(_REF_RHO0, timesteps=[])
    with pytest.raises(ValueError, match="entropy base"):
        valid.qmi(base=1)
    with pytest.raises(ValueError, match="entropy base"):
        valid.cmi(base=1)


def test_information_metrics_match_classically_correlated_references() -> None:
    """QMI and CMI match exact classical correlations in valid process tensors."""
    one_leg_diagonal = np.zeros((2, 2, 2), dtype=np.complex128)
    for input_index, output_index in np.ndindex(2, 2):
        one_leg_diagonal[input_index, output_index, input_index] = 0.5
    one_leg = np.diag(one_leg_diagonal.reshape(-1))
    _assert_causally_normalized(one_leg, 1)
    pt_k1 = DenseProcessTensor(one_leg, [0.0, 0.0])
    assert pt_k1.qmi() == pytest.approx(1.0)

    two_leg = _classical_memory_process_matrix()
    _assert_causally_normalized(two_leg, 2)
    pt_k2 = DenseProcessTensor(two_leg, [0.0, 0.0, 0.0])
    assert pt_k2.qmi(past="all") == pytest.approx(1.0)
    assert pt_k2.qmi(past="first") == pytest.approx(1.0)
    qmi_last = pt_k2.qmi(past="last")
    assert qmi_last >= 0.0
    assert qmi_last == pytest.approx(0.0, abs=1e-12)
    assert pt_k2.cmi() == pytest.approx(1.0)

    scaled_pt = DenseProcessTensor(0.125 * two_leg, [0.0, 0.0, 0.0])
    assert scaled_pt.qmi(past="first") == pytest.approx(1.0)
    assert scaled_pt.cmi() == pytest.approx(1.0)


def test_information_metrics_repair_tolerated_spectral_roundoff_before_reduction() -> None:
    """Accepted negative roundoff is projected out before a partial trace can amplify it."""
    two_leg = _classical_memory_process_matrix()
    diagonal = np.diag(two_leg).real
    support = np.flatnonzero(diagonal > 0.0)
    null_space = np.flatnonzero(np.isclose(diagonal, 0.0))
    perturbation = 3.6e-10
    two_leg[null_space, null_space] -= perturbation
    two_leg[support, support] += perturbation
    pt = DenseProcessTensor(two_leg, [0.0, 0.0, 0.0])

    assert pt.qmi(past="first") == pytest.approx(1.0, abs=1e-10)
    assert pt.cmi() == pytest.approx(1.0, abs=1e-10)


def test_dense_process_tensor_qmi_and_cmi() -> None:
    """Information metrics match the exact values for a memoryless two-leg process."""
    pt_k1 = _tiny_process_tensor(num_interventions=1)
    pt_k2 = _tiny_process_tensor(num_interventions=2)

    assert pt_k1.cmi() == pytest.approx(0.0)
    assert pt_k2.qmi(past="all") == pytest.approx(2.0)
    assert pt_k2.qmi(past="first") == pytest.approx(0.0, abs=1e-12)
    assert pt_k2.qmi(past="last") == pytest.approx(2.0)
    assert pt_k2.cmi() == pytest.approx(0.0, abs=1e-12)

    with pytest.raises(ValueError, match="Unknown past"):
        pt_k2.qmi(past="middle")


def test_dense_and_mpo_process_tensors_match_exact_two_swap_memory_process() -> None:
    """Both constructors reproduce a two-SWAP process that returns the first operation output."""
    duration = np.pi / 4
    ham = Hamiltonian.heisenberg(length=2, Jx=1.0, Jy=1.0, Jz=1.0, h=0.0)
    params = AnalogSimParams(elapsed_time=duration, dt=duration, max_bond_dim=16)
    characterizer = MemoryCharacterizer(parallel=False, show_progress=False)
    timesteps = [0.0, duration, duration]
    dense = cast(
        "DenseProcessTensor",
        characterizer.build_process_tensor(ham, params, timesteps=timesteps, return_type="dense"),
    )
    mpo = cast(
        "MPOProcessTensor",
        characterizer.build_process_tensor(ham, params, timesteps=timesteps, return_type="mpo"),
    )
    expected = _two_swap_process_matrix()

    rotation = (np.eye(2, dtype=np.complex128) - 1j * np.array([[0.0, 1.0], [1.0, 0.0]])) / np.sqrt(2)
    flip = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    def rotate(rho: np.ndarray) -> np.ndarray:
        return rotation @ rho @ rotation.conj().T

    def flip_state(rho: np.ndarray) -> np.ndarray:
        return flip @ rho @ flip

    expected_output = rotate(_REF_RHO0)
    for process_tensor in (dense, mpo):
        matrix = process_tensor.to_matrix()
        np.testing.assert_allclose(matrix, expected, atol=1e-11)
        _assert_causally_normalized(matrix, 2)
        np.testing.assert_allclose(process_tensor.predict([rotate, flip_state]), expected_output, atol=1e-11)
        assert process_tensor.qmi(past="all") == pytest.approx(2.0, abs=1e-10)
        assert process_tensor.qmi(past="first") == pytest.approx(2.0, abs=1e-10)
        qmi_last = process_tensor.qmi(past="last")
        assert qmi_last >= 0.0
        assert qmi_last == pytest.approx(0.0, abs=1e-12)
        assert process_tensor.cmi() == pytest.approx(2.0, abs=1e-10)

    first_cut = compute_temporal_entropy(expected, 2, 1)
    second_cut = compute_temporal_entropy(expected, 2, 2)
    assert cast("int", first_cut["schmidt_rank"]) == 1
    assert float(cast("float", first_cut["entropy"])) == pytest.approx(0.0, abs=1e-12)
    assert cast("int", second_cut["schmidt_rank"]) == 4
    assert float(cast("float", second_cut["entropy"])) == pytest.approx(np.log(4.0), abs=1e-12)


def test_dense_process_tensor_evaluate_probes_smoke() -> None:
    """Dense process-tensor probe evaluation returns Pauli tomography coefficients."""
    pt = _tiny_process_tensor(num_interventions=1)
    probe_set = sample_probes(cut=1, num_interventions=1, n_pasts=2, n_futures=2, rng=np.random.default_rng(0))
    pauli = evaluate_probes(pt, probe_set)
    assert pauli.shape == (2, 2, 4)
    wrapped = pt.evaluate_probes(probe_set)
    np.testing.assert_allclose(wrapped, pauli)


def test_dense_process_tensor_responses_with_weights_reconstruct_subnormalized_tomography() -> None:
    """Normalized responses times joint probabilities recover each outcome branch."""
    pt = _tiny_process_tensor(num_interventions=2)
    probe_set = sample_probes(
        cut=1,
        num_interventions=2,
        n_pasts=2,
        n_futures=3,
        rng=np.random.default_rng(73),
        intervention_style="measure_prepare",
    )
    pauli, weights = pt.evaluate_probes_with_weights(probe_set)
    for i in range(2):
        for j in range(3):
            steps = assemble_probe_sequence(probe_set, i, j)
            branch = pt._contract_subnormalized_branch([convert_probe_callable(step) for step in steps])
            np.testing.assert_allclose(weights[i, j] * pauli[i, j], encode_rho_pauli(branch), atol=1e-12)
    wrapped_pauli, wrapped_weights = pt.evaluate_probes_with_weights(probe_set)
    np.testing.assert_allclose(wrapped_pauli, pauli)
    np.testing.assert_allclose(wrapped_weights, weights)


def test_dense_process_tensor_responses_with_weights_preserve_small_positive_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A small positive branch remains part of the response matrix."""
    probability = 1e-13
    rho_branch = probability * _REF_RHO0
    pt = DenseProcessTensor(np.eye(8, dtype=np.complex128), [0.0, 0.0])

    def _contract_small_branch(self: DenseProcessTensor, interventions: object) -> np.ndarray:
        _ = (self, interventions)
        return rho_branch

    monkeypatch.setattr(DenseProcessTensor, "_contract_subnormalized_branch", _contract_small_branch)
    probe_set = sample_probes(
        cut=1,
        num_interventions=1,
        n_pasts=1,
        n_futures=1,
        rng=np.random.default_rng(0),
    )

    pauli, weights = pt.evaluate_probes_with_weights(probe_set)

    assert weights[0, 0] == pytest.approx(probability, rel=1e-12, abs=0.0)
    np.testing.assert_allclose(pauli[0, 0], encode_rho_pauli(_REF_RHO0), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(
        weights[0, 0] * pauli[0, 0],
        encode_rho_pauli(rho_branch),
        rtol=1e-12,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("rho_branch", "message"),
    [
        (1e-13 * np.diag([1.2, -0.2]).astype(np.complex128), "conditional state must be positive semidefinite"),
        (
            1e-13 * np.array([[0.5, 0.2], [0.0, 0.5]], dtype=np.complex128),
            "Process-tensor branch must be Hermitian",
        ),
        (np.ones(2, dtype=np.complex128), r"branch must have shape \(2, 2\)"),
        (
            np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=np.complex128),
            "branch must contain only finite values",
        ),
    ],
)
def test_dense_process_tensor_responses_with_weights_reject_invalid_branches(
    monkeypatch: pytest.MonkeyPatch,
    rho_branch: np.ndarray,
    message: str,
) -> None:
    """Weighted responses reject tiny nonphysical and malformed branches."""
    pt = DenseProcessTensor(np.eye(8, dtype=np.complex128), [0.0, 0.0])

    def _contract_invalid_branch(self: DenseProcessTensor, interventions: object) -> np.ndarray:
        _ = (self, interventions)
        return rho_branch

    monkeypatch.setattr(DenseProcessTensor, "_contract_subnormalized_branch", _contract_invalid_branch)
    probe_set = sample_probes(
        cut=1,
        num_interventions=1,
        n_pasts=1,
        n_futures=1,
        rng=np.random.default_rng(0),
    )

    with pytest.raises(ValueError, match=message):
        pt.evaluate_probes_with_weights(probe_set)


def test_dense_process_tensor_responses_with_weights_represent_zero_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An impossible branch has zero weight and a neutral response placeholder."""
    pt = DenseProcessTensor(np.eye(8, dtype=np.complex128), [0.0, 0.0])

    def _contract_zero_branch(self: DenseProcessTensor, interventions: object) -> np.ndarray:
        _ = (self, interventions)
        return np.zeros((2, 2), dtype=np.complex128)

    monkeypatch.setattr(DenseProcessTensor, "_contract_subnormalized_branch", _contract_zero_branch)
    probe_set = sample_probes(
        cut=1,
        num_interventions=1,
        n_pasts=1,
        n_futures=1,
        rng=np.random.default_rng(0),
    )

    pauli, weights = pt.evaluate_probes_with_weights(probe_set)

    np.testing.assert_array_equal(weights, np.zeros((1, 1), dtype=np.float64))
    np.testing.assert_allclose(pauli[0, 0], np.array([1.0, 0.0, 0.0, 0.0]))


def test_dense_process_tensor_responses_with_weights_reject_nonzero_traceless_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero-probability branch cannot have a nonzero subnormalized output."""
    rho_branch = np.diag([1e-6, -1e-6]).astype(np.complex128)
    pt = DenseProcessTensor(np.eye(8, dtype=np.complex128), [0.0, 0.0])

    def _contract_traceless_branch(self: DenseProcessTensor, interventions: object) -> np.ndarray:
        _ = (self, interventions)
        return rho_branch

    monkeypatch.setattr(DenseProcessTensor, "_contract_subnormalized_branch", _contract_traceless_branch)
    probe_set = sample_probes(
        cut=1,
        num_interventions=1,
        n_pasts=1,
        n_futures=1,
        rng=np.random.default_rng(0),
    )

    with pytest.raises(ValueError, match="near-zero trace but a nonzero subnormalized output"):
        pt.evaluate_probes_with_weights(probe_set)


def test_dense_process_tensor_rejects_non_psd_conditional_state() -> None:
    """Prediction rejects a trace-one branch with a substantive negative eigenvalue."""
    pt = DenseProcessTensor(np.diag([1.2, -0.2]).astype(np.complex128), timesteps=[])

    with pytest.raises(ValueError, match="conditional state must be positive semidefinite"):
        pt.predict([])


def test_dense_process_tensor_rejects_non_hermitian_conditional_state() -> None:
    """Prediction rejects a conditional branch with a substantive anti-Hermitian part."""
    branch = np.array([[0.5, 0.2], [0.0, 0.5]], dtype=np.complex128)
    pt = DenseProcessTensor(branch, timesteps=[])

    with pytest.raises(ValueError, match="Process-tensor branch must be Hermitian"):
        pt.predict([])


def test_dense_process_tensor_selected_future_with_weights_matches_exact() -> None:
    """Outcome-branch traces reproduce exact joint probabilities for selected future outcomes."""
    ham = Hamiltonian.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8, order=1)
    pt = build_process_tensor(
        ham.mpo,
        params,
        timesteps=[0.0, 0.0, 0.0],
        parallel=False,
        return_type="dense",
    )
    assert isinstance(pt, DenseProcessTensor)
    probe_set = sample_probes(
        cut=1,
        num_interventions=2,
        n_pasts=2,
        n_futures=3,
        rng=np.random.default_rng(73),
        intervention_style="measure_prepare",
    )

    pauli_pt, weights_pt = pt.evaluate_probes_with_weights(probe_set)
    pauli_exact, weights_exact, _ = simulate_exact(
        probe_set=probe_set,
        operator=ham.mpo,
        sim_params=params,
        initial_psi=np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128),
        parallel=False,
    )

    assert np.any(np.ptp(weights_exact, axis=1) > 1e-8)
    np.testing.assert_allclose(weights_pt, weights_exact, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        weights_pt[..., np.newaxis] * pauli_pt,
        weights_exact[..., np.newaxis] * pauli_exact,
        rtol=1e-7,
        atol=1e-8,
    )


def test_default_mpo_selected_future_with_weights_matches_exact() -> None:
    """The supported direct MPO reproduces exact responses and joint probabilities."""
    ham = Hamiltonian.ising(length=2, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8, order=1)
    timesteps = [0.1, 0.1, 0.1]
    pt = build_process_tensor(
        ham.mpo,
        params,
        timesteps=timesteps,
        parallel=False,
        compress_every=1,
    )
    assert isinstance(pt, MPOProcessTensor)
    probe_set = sample_probes(
        cut=1,
        num_interventions=2,
        n_pasts=2,
        n_futures=2,
        rng=np.random.default_rng(73),
        intervention_style="measure_prepare",
    )

    pauli_pt, weights_pt = pt.evaluate_probes_with_weights(probe_set)
    initial_psi = np.zeros(4, dtype=np.complex128)
    initial_psi[0] = 1.0
    pauli_exact, weights_exact, _ = simulate_exact(
        probe_set=probe_set,
        operator=ham.mpo,
        sim_params=params,
        initial_psi=initial_psi,
        parallel=False,
    )

    np.testing.assert_allclose(weights_pt, weights_exact, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        weights_pt[..., np.newaxis] * pauli_pt,
        weights_exact[..., np.newaxis] * pauli_exact,
        rtol=1e-7,
        atol=1e-8,
    )


def test_mpo_probes_with_weights_reject_nonphysical_reconstruction(monkeypatch: pytest.MonkeyPatch) -> None:
    """An experimental capped MPO warns and rejects an invalid branch."""
    ham = Hamiltonian.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8)
    with pytest.warns(RuntimeWarning, match="experimental direct process-tensor truncation"):
        pt = cast(
            "MPOProcessTensor",
            build_process_tensor(
                ham.mpo,
                params,
                timesteps=[0.0, 0.0],
                max_bond_dim=1,
                compress_every=1,
                parallel=False,
            ),
        )

    def _contract_invalid_branch(self: MPOProcessTensor, interventions: object) -> np.ndarray:
        _ = (self, interventions)
        return -0.1 * _REF_RHO0

    monkeypatch.setattr(MPOProcessTensor, "_contract_subnormalized_branch", _contract_invalid_branch)
    probe_set = sample_probes(
        cut=1,
        num_interventions=1,
        n_pasts=1,
        n_futures=1,
        rng=np.random.default_rng(0),
        intervention_style="measure_prepare",
    )

    with pytest.raises(ValueError, match="branch trace must be a probability") as exc_info:
        pt.evaluate_probes_with_weights(probe_set)

    assert "Direct-MPO compression or tomography error" in str(exc_info.value)
    assert "max_bond_dim=None" in str(exc_info.value)
    assert "return_type='dense'" in str(exc_info.value)


def test_impossible_branch_predict_matches_between_dense_and_mpo() -> None:
    """Dense and MPO predictors preserve the same near-zero impossible branch."""
    mpo_pt = _tiny_mpo_process_tensor(num_interventions=1)
    dense_pt = mpo_pt.to_dense()
    zero = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    one = np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
    impossible = convert_probe_callable((one, zero))

    dense_prediction = dense_pt.predict([impossible])
    mpo_prediction = mpo_pt.predict([impossible])

    assert np.linalg.norm(dense_prediction) < 1e-12
    assert np.linalg.norm(mpo_prediction) < 1e-12
    np.testing.assert_allclose(mpo_prediction, dense_prediction, atol=1e-12)


def test_mpo_process_tensor_evaluate_probes_matches_dense_without_densifying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MPO probe evaluation matches dense and does not call :meth:`to_dense`."""
    ham = Hamiltonian.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8)
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    mpo_pt = cast(
        "MPOProcessTensor",
        mc.build_process_tensor(ham, params, timesteps=[0.0, 0.0, 0.0], return_type="mpo", compress_every=1),
    )
    dense_pt = mpo_pt.to_dense()

    probe_set = sample_probes(
        cut=1,
        num_interventions=2,
        n_pasts=2,
        n_futures=2,
        rng=np.random.default_rng(1),
        intervention_style="measure_prepare",
    )

    def _fail_to_dense(self: MPOProcessTensor) -> DenseProcessTensor:
        _ = self
        msg = "evaluate_probes must not densify the MPO process tensor"
        raise AssertionError(msg)

    monkeypatch.setattr(MPOProcessTensor, "to_dense", _fail_to_dense)
    mpo_responses = mpo_pt.evaluate_probes(probe_set)
    mpo_responses_with_weights, mpo_weights = mpo_pt.evaluate_probes_with_weights(probe_set)

    dense_responses = dense_pt.evaluate_probes(probe_set)
    dense_responses_with_weights, dense_weights = dense_pt.evaluate_probes_with_weights(probe_set)
    assert mpo_responses.shape == dense_responses.shape == (2, 2, 4)
    np.testing.assert_allclose(mpo_responses, dense_responses, atol=1e-6)
    np.testing.assert_allclose(mpo_responses_with_weights, dense_responses_with_weights, atol=1e-6)
    np.testing.assert_allclose(mpo_weights, dense_weights, atol=1e-6)

    # Restore before methods that still densify (qmi/cmi).
    monkeypatch.undo()
    assert isinstance(mpo_pt.cmi(), float)
    assert mpo_pt._num_interventions_for_probe() == 2


def test_compute_temporal_entropy_markov_j0() -> None:
    """Uncoupled Ising process has vanishing temporal entanglement at every cut."""
    ham = Hamiltonian.ising(length=6, J=0.0, g=1.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=64, order=1)
    pt = cast(
        "DenseProcessTensor",
        MemoryCharacterizer(parallel=False, show_progress=False).build_process_tensor(
            ham,
            params,
            timesteps=[0.1] * 4,
            return_type="dense",
        ),
    )
    for cut in (1, 2, 3):
        result = pt.compute_temporal_entropy(cut)
        assert cast("int", result["schmidt_rank"]) == 1
        assert float(cast("float", result["entropy"])) == pytest.approx(0.0, abs=1e-10)


def test_compute_temporal_entropy_correlated_j1() -> None:
    """Correlated process has positive temporal entanglement at the center cut."""
    ham = Hamiltonian.ising(length=6, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=64, order=1)
    pt = cast(
        "DenseProcessTensor",
        MemoryCharacterizer(parallel=False, show_progress=False).build_process_tensor(
            ham,
            params,
            timesteps=[0.1] * 4,
            return_type="dense",
        ),
    )
    result = pt.compute_temporal_entropy(2)
    assert cast("int", result["schmidt_rank"]) > 1
    assert float(cast("float", result["entropy"])) > 0.0


def test_compute_temporal_entropy_scale_invariant() -> None:
    """Overall scaling of upsilon does not change temporal entanglement."""
    k = 2
    ups = np.eye(2 * 4**k, dtype=np.complex128)
    base = float(cast("float", compute_temporal_entropy(ups, k, 1)["entropy"]))
    scaled = float(cast("float", compute_temporal_entropy(2.5 * ups, k, 1)["entropy"]))
    assert base == pytest.approx(scaled, abs=1e-12)
