from __future__ import annotations

import numpy as np

from mqt.yaqs.characterization.process_tensors.core.encoding import pack_rho8
from mqt.yaqs.characterization.process_tensors.core.metrics import (
    _mean_frobenius_mse_rho8,
    _mean_trace_distance_rho8,
    _rel_fro_error,
    _trace_distance,
)


def test_rel_fro_error_zero_for_equal_matrices() -> None:
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.complex128)
    assert _rel_fro_error(A, A.copy()) == 0.0


def test_rel_fro_error_scaling() -> None:
    A = np.eye(2, dtype=np.complex128)
    B = 2.0 * A
    # ||A - 2A||_F = ||A||_F, so relative error = 1
    assert np.isclose(_rel_fro_error(A, B), 0.5)


def test_trace_distance_basic_pure_states() -> None:
    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    rho1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    d = _trace_distance(rho0, rho1)
    # Orthogonal pure states → trace distance = 1
    assert np.isclose(d, 1.0)


def test_rho8_metrics_zero_when_equal() -> None:
    rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    y = pack_rho8(rho)[None, :]
    assert _mean_trace_distance_rho8(y, y) == 0.0
    assert _mean_frobenius_mse_rho8(y, y) == 0.0


def test_rho8_metrics_positive_for_different_states() -> None:
    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    rho1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    y0 = pack_rho8(rho0)[None, :]
    y1 = pack_rho8(rho1)[None, :]
    assert _mean_trace_distance_rho8(y0, y1) > 0.9
    assert _mean_frobenius_mse_rho8(y0, y1) > 0.0
