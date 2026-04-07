from __future__ import annotations

import numpy as np

from mqt.yaqs.characterization.process_tensors.core.encoding import (
    _flatten_choi4_to_real32,
    build_choi_feature_table,
    normalize_rho_from_backend_output,
    pack_rho8,
    unpack_rho8,
)


def test_flatten_choi4_to_real32_shape_and_dtype() -> None:
    j = np.eye(4, dtype=np.complex128)
    y = _flatten_choi4_to_real32(j)
    assert y.shape == (32,)
    assert y.dtype == np.float32


def test_build_choi_feature_table_shape() -> None:
    mats = [np.eye(4, dtype=np.complex128) for _ in range(16)]
    table = build_choi_feature_table(mats)
    assert table.shape == (16, 32)
    assert table.dtype == np.float32


def test_pack_unpack_roundtrip_hermitianized() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    rho = a @ a.conj().T
    rho = rho / np.trace(rho)
    y = pack_rho8(rho)
    assert y.shape == (8,)
    rho2 = unpack_rho8(y)
    np.testing.assert_allclose(rho2, rho2.conj().T, atol=1e-12)
    np.testing.assert_allclose(np.trace(rho2).real, 1.0, atol=1e-12)


def test_normalize_rho_from_backend_output_returns_physical_dm() -> None:
    rho_raw = np.array([[2.0 + 0.0j, 1.0 + 2.0j], [0.0 + 0.0j, 0.1 + 0.0j]], dtype=np.complex128)
    rho = normalize_rho_from_backend_output(rho_raw)
    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-12)
    np.testing.assert_allclose(np.trace(rho).real, 1.0, atol=1e-12)
    evals = np.linalg.eigvalsh(rho).real
    assert float(evals.min()) >= -1e-12

