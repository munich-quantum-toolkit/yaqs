from __future__ import annotations

import numpy as np

from mqt.yaqs.characterization.process_tensors.core.utils import (
    _get_rho_site_zero,
    _initialize_backend_state,
    _reconstruct_state,
)
from mqt.yaqs.core.data_structures.networks import MPO, MPS


def test_initialize_backend_state_mcwf_and_tjm() -> None:
    op = MPO.ising(length=1, J=1.0, g=0.5)

    state_mcwf = _initialize_backend_state(op, solver="MCWF")
    assert isinstance(state_mcwf, np.ndarray)
    assert state_mcwf.shape == (2**op.length,)

    state_tjm = _initialize_backend_state(op, solver="TJM")
    assert isinstance(state_tjm, MPS)
    assert state_tjm.length == op.length


def test_get_rho_site_zero_from_mps_and_vector() -> None:
    """Single-qubit density extraction should give a 2x2 PSD matrix with non-negative trace."""
    mps = MPS(length=1, state="zeros")
    rho_mps = _get_rho_site_zero(mps)
    assert rho_mps.shape == (2, 2)
    assert np.real(np.trace(rho_mps)) >= 0.0

    vec = np.zeros(2, dtype=np.complex128)
    vec[0] = 1.0
    rho_vec = _get_rho_site_zero(vec)
    np.testing.assert_allclose(rho_vec, np.array([[1.0, 0.0], [0.0, 0.0]]))


def test_reconstruct_state_from_expectations() -> None:
    """Check that _reconstruct_state inverts simple Pauli expectations for |0>."""
    from mqt.yaqs.core.libraries.gate_library import X, Y, Z

    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    rho0 = np.outer(psi0, psi0.conj())

    ex = np.trace(X().matrix @ rho0)
    ey = np.trace(Y().matrix @ rho0)
    ez = np.trace(Z().matrix @ rho0)

    rho_rec = _reconstruct_state({"x": ex, "y": ey, "z": ez})
    np.testing.assert_allclose(rho_rec, rho0, atol=1e-12)
