# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff: noqa: SLF001 -- white-box tests exercise private process-tensor prediction helpers

"""Tests for DenseProcessTensor and MPOProcessTensor wrappers."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer
from mqt.yaqs.characterization.memory.backends.tomography.data import SequenceData
from mqt.yaqs.characterization.memory.backends.tomography.process_tensors import (
    DenseProcessTensor,
    MPOProcessTensor,
    compute_entropy_dense,
    convert_probe_callable,
    encode_cptp_choi,
    evaluate_dense_probes,
    trace_partial_dense,
)
from mqt.yaqs.characterization.memory.operational_memory.samples import sample_probes
from mqt.yaqs.characterization.memory.shared.intervention_steps import build_intervention_operator
from mqt.yaqs.characterization.memory.shared.interventions import InterventionMap
from mqt.yaqs.core.data_structures.mpo import MPO

_REF_RHO0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)


def test_dense_process_tensor_predict_matches_helper() -> None:
    """DenseProcessTensor._predict_raw matches the Choi contraction; predict physicalizes."""
    ups = np.eye(2 * 4, dtype=np.complex128)
    timesteps = [0.1]

    def id_map(rho: np.ndarray) -> np.ndarray:
        return rho

    pt = DenseProcessTensor(ups, timesteps)
    # Identity map Choi has trace 2; contract U = I with it gives unnormalized rho = 2*I
    rho_raw = pt._predict_raw([id_map])
    np.testing.assert_allclose(rho_raw, 2.0 * np.eye(2, dtype=np.complex128), atol=1e-12)
    rho = pt.predict([id_map])
    np.testing.assert_allclose(np.trace(rho), 1.0, atol=1e-12)
    np.testing.assert_allclose(rho, rho_raw / np.trace(rho_raw), atol=1e-12)


def test_dense_process_tensor_predict_raises_on_length_mismatch() -> None:
    """DenseProcessTensor.predict rejects intervention lists whose length mismatches num_interventions."""
    ups = np.eye(2 * 4, dtype=np.complex128)
    pt = DenseProcessTensor(ups, [0.1])

    def id_map(rho: np.ndarray) -> np.ndarray:
        return rho

    with pytest.raises(ValueError, match="DenseProcessTensor expects"):
        pt.predict([id_map, id_map])


def test_compute_entropy_dense_rejects_invalid_base() -> None:
    """Entropy helpers reject non-positive bases and base equal to 1."""
    rho = np.eye(2, dtype=np.complex128) * 0.5
    with pytest.raises(ValueError, match="entropy base"):
        compute_entropy_dense(rho, base=1)
    with pytest.raises(ValueError, match="entropy base"):
        compute_entropy_dense(rho, base=0)


def test_mpo_process_tensor_matrix_matches_dense() -> None:
    """MPOProcessTensor.to_matrix should match MPO.to_matrix()."""
    mpo = MPO.ising(length=1, J=1.0, g=0.5)
    timesteps: list[float] = [0.1]
    pt = MPOProcessTensor(mpo, timesteps)

    np.testing.assert_allclose(
        pt.to_matrix(),
        mpo.to_matrix(),
        atol=1e-12,
    )


def test_mpo_process_tensor_qmi_fallback_to_dense() -> None:
    """MPOProcessTensor.qmi should agree with DenseProcessTensor.qmi via dense fallback."""
    mpo = MPO.ising(length=1, J=1.0, g=0.5)
    timesteps: list[float] = [0.1]
    pt = MPOProcessTensor(mpo, timesteps)

    q1 = pt.qmi()
    q2 = pt.to_dense().qmi()
    assert abs(q1 - q2) < 1e-12


def test_mpo_process_tensor_predict_smoke_identity_map() -> None:
    """MPOProcessTensor.predict returns a physical density matrix for a trivial intervention."""
    rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    data = SequenceData(
        sequences=[(0,)],
        outputs=[rho],
        weights=[1.0],
        choi_basis=[np.eye(4, dtype=np.complex128)] * 16,
        choi_indices=[(0, 0)] * 16,
        choi_duals=[np.eye(4, dtype=np.complex128)] * 16,
        timesteps=[0.1],
        initial_rho=_REF_RHO0,
    )
    pt = data.to_mpo_process_tensor(compress_every=1)

    def id_map(x: np.ndarray) -> np.ndarray:
        return x

    rho_out = pt.predict([id_map])
    assert rho_out.shape == (2, 2)
    np.testing.assert_allclose(rho_out, rho_out.conj().T, atol=1e-12)
    np.testing.assert_allclose(np.trace(rho_out).real, 1.0, atol=1e-12)


def test_mpo_process_tensor_predict_raises_on_empty_interventions() -> None:
    """Predict rejects empty interventions when num_interventions>0."""
    data = SequenceData(
        sequences=[(0,)],
        outputs=[np.eye(2, dtype=np.complex128)],
        weights=[1.0],
        choi_basis=[np.eye(4, dtype=np.complex128)] * 16,
        choi_indices=[(0, 0)] * 16,
        choi_duals=[np.eye(4, dtype=np.complex128)] * 16,
        timesteps=[0.1],
        initial_rho=_REF_RHO0,
    )
    pt = data.to_mpo_process_tensor(compress_every=1)
    with pytest.raises(ValueError, match="interventions list must be non-empty"):
        pt.predict([])


def test_mpo_process_tensor_predict_zero_steps() -> None:
    """MPOProcessTensor.predict([]) returns the stored output when num_interventions=0."""
    rho = np.array([[0.6, 0.1 + 0.0j], [0.1 - 0.0j, 0.4]], dtype=np.complex128)
    data = SequenceData(
        sequences=[()],
        outputs=[rho],
        weights=[1.0],
        choi_basis=[],
        choi_indices=[],
        choi_duals=[],
        timesteps=[],
        initial_rho=_REF_RHO0,
    )
    pt = data.to_mpo_process_tensor(compress_every=1)
    rho_out = pt.predict([])
    np.testing.assert_allclose(rho_out, rho, atol=1e-10)


def test_mpo_process_tensor_predict_raises_on_length_mismatch() -> None:
    """Predict rejects intervention lists whose length mismatches the process tensor."""
    data = SequenceData(
        sequences=[(0,)],
        outputs=[np.eye(2, dtype=np.complex128)],
        weights=[1.0],
        choi_basis=[np.eye(4, dtype=np.complex128)] * 16,
        choi_indices=[(0, 0)] * 16,
        choi_duals=[np.eye(4, dtype=np.complex128)] * 16,
        timesteps=[0.1],
        initial_rho=_REF_RHO0,
    )
    pt = data.to_mpo_process_tensor(compress_every=1)

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


def test_encode_cptp_choi_identity() -> None:
    """Choi encoding round-trips the identity channel."""

    def id_map(rho: np.ndarray) -> np.ndarray:
        return rho

    choi = encode_cptp_choi(id_map)
    assert choi.shape == (4, 4)
    assert np.linalg.norm(choi - choi.conj().T) < 1e-10


def test_trace_partial_dense_and_entropy_edge_cases() -> None:
    """Partial trace and entropy helpers cover validation and degenerate inputs."""
    rho = np.kron(np.eye(2, dtype=np.complex128), np.eye(2, dtype=np.complex128)) * 0.25
    reduced = trace_partial_dense(rho, dims=[2, 2], keep=[0])
    assert reduced.shape == (2, 2)

    with pytest.raises(ValueError, match="keep indices"):
        trace_partial_dense(rho, dims=[2, 2], keep=[3])

    assert compute_entropy_dense(np.zeros((2, 2), dtype=np.complex128)) == pytest.approx(0.0)
    pure = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    assert compute_entropy_dense(pure) == pytest.approx(0.0, abs=1e-12)


def test_dense_process_tensor_qmi_and_cmi() -> None:
    """Information metrics run on small process tensors including past-leg variants."""
    pt_k1 = _tiny_process_tensor(num_interventions=1)
    pt_k2 = _tiny_process_tensor(num_interventions=2)

    assert pt_k1.cmi() == pytest.approx(0.0)

    q_all = pt_k2.qmi(past="all")
    q_last = pt_k2.qmi(past="last", assume_canonical=True)
    q_first = pt_k2.qmi(past="first")
    assert isinstance(q_all, float)
    assert isinstance(q_last, float)
    assert isinstance(q_first, float)

    cmi = pt_k2.cmi(assume_canonical=True)
    assert isinstance(cmi, float)

    with pytest.raises(ValueError, match="Unknown past"):
        pt_k2.qmi(past="middle")


def test_dense_process_tensor_evaluate_probes_smoke() -> None:
    """Dense process-tensor probe evaluation returns Pauli tomography coefficients."""
    pt = _tiny_process_tensor(num_interventions=1)
    probe_set = sample_probes(cut=1, num_interventions=1, n_pasts=2, n_futures=2, rng=np.random.default_rng(0))
    pauli = evaluate_dense_probes(pt, probe_set)
    assert pauli.shape == (2, 2, 4)
    wrapped = pt.evaluate_probes(probe_set)
    np.testing.assert_allclose(wrapped, pauli)


def test_mpo_process_tensor_evaluate_probes_and_cmi_delegates() -> None:
    """MPOProcessTensor wrappers delegate probe and information metrics to dense."""
    ham = Hamiltonian.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8)
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    mpo_pt = cast(
        "MPOProcessTensor",
        mc.build_process_tensor(ham, params, timesteps=[0.0, 0.0, 0.0], return_type="mpo", compress_every=1),
    )
    dense_pt = mpo_pt.to_dense()

    probe_set = sample_probes(cut=1, num_interventions=2, n_pasts=2, n_futures=2, rng=np.random.default_rng(1))
    mpo_pauli = mpo_pt.evaluate_probes(probe_set)
    dense_pauli = dense_pt.evaluate_probes(probe_set)
    assert mpo_pauli.shape == dense_pauli.shape == (2, 2, 4)

    assert isinstance(mpo_pt.cmi(), float)
    assert mpo_pt._num_interventions_for_probe() == 2
