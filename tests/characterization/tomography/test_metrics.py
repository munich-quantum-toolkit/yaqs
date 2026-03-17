from __future__ import annotations

import numpy as np

from mqt.yaqs.characterization.tomography.metrics import rel_fro_error, trace_distance


def test_rel_fro_error_zero_for_equal_matrices() -> None:
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.complex128)
    assert rel_fro_error(A, A.copy()) == 0.0


def test_rel_fro_error_scaling() -> None:
    A = np.eye(2, dtype=np.complex128)
    B = 2.0 * A
    # ||A - 2A||_F = ||A||_F, so relative error = 1
    assert np.isclose(rel_fro_error(A, B), 0.5)


def test_trace_distance_basic_pure_states() -> None:
    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    rho1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    d = trace_distance(rho0, rho1)
    # Orthogonal pure states → trace distance = 1
    assert np.isclose(d, 1.0)