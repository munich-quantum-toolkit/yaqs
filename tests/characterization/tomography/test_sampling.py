from __future__ import annotations

import numpy as np

from mqt.yaqs.characterization.tomography.sampling import (
    _continuous_dual_step,
    _enumerate_sequences,
    _normalize_log_weights,
    _sample_haar_pure_state,
)


def test_enumerate_sequences_length() -> None:
    seqs_k1 = _enumerate_sequences(1)
    seqs_k2 = _enumerate_sequences(2)
    assert len(seqs_k1) == 16
    assert len(seqs_k2) == 16 * 16
    assert all(isinstance(s, tuple) for s in seqs_k2)


def test_sample_haar_pure_state_normalized() -> None:
    rng = np.random.default_rng(0)
    psi = _sample_haar_pure_state(rng)
    norm = np.linalg.norm(psi)
    assert psi.shape == (2,)
    assert np.isclose(norm, 1.0)


def test_continuous_dual_step_shape() -> None:
    rng = np.random.default_rng(1)
    psi_meas = _sample_haar_pure_state(rng)
    psi_prep = _sample_haar_pure_state(rng)
    D = _continuous_dual_step(psi_meas, psi_prep)
    assert D.shape == (4, 4)


def test_normalize_log_weights_returns_prob_vector() -> None:
    log_w = np.log(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    w_norm, log_w_sum = _normalize_log_weights(log_w)
    assert w_norm.shape == (3,)
    assert np.all(w_norm >= 0)
    assert np.isclose(w_norm.sum(), 1.0)
    exp_sum = float(np.sum(np.exp(log_w)))
    assert np.isclose(log_w_sum, np.log(exp_sum))