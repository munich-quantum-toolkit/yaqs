# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff:file-ignore[no-self-use, import-private-name] -- protocol-style dummy backend; white-box rollout test

"""Tests for operational-memory orchestration (:mod:`run`)."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from mqt.yaqs.characterization.memory.backends.exact import ExactBackend, simulate_exact
from mqt.yaqs.characterization.memory.backends.tomography import build_process_tensor
from mqt.yaqs.characterization.memory.backends.tomography.process_tensors import DenseProcessTensor, MPOProcessTensor
from mqt.yaqs.characterization.memory.operational_memory.branch_weights import (
    _compute_branch_weight_for_sequence,
    compute_branch_weights,
)
from mqt.yaqs.characterization.memory.operational_memory.response_matrix import (
    assemble_response_matrix,
    compute_spectrum,
)
from mqt.yaqs.characterization.memory.operational_memory.run import (
    OperationalMemoryBackend,
    evaluate_probes_with_weights,
    run_memory_characterization,
)
from mqt.yaqs.characterization.memory.operational_memory.samples import ProbeSet, sample_probes
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

_PSI0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)


def _params() -> AnalogSimParams:
    """Return analog simulation parameters for tight exact-backend tests.

    Returns:
        :class:`~mqt.yaqs.AnalogSimParams` with a small bond dimension and timestep.
    """
    return AnalogSimParams(dt=0.05, max_bond_dim=8, order=1)


def _diagnostics_final_weight(diagnostics: dict[str, object]) -> float:
    """Extract the final cumulative weight from simulation diagnostics.

    Returns:
        Final cumulative intervention weight.

    Raises:
        TypeError: If ``cumulative_weight_final`` is not numeric.
    """
    val = diagnostics["cumulative_weight_final"]
    if not isinstance(val, (int, float)):
        msg = "cumulative_weight_final must be numeric"
        raise TypeError(msg)
    return float(val)


def test_run_memory_characterization_uses_object_backend() -> None:
    """run_memory_characterization delegates evaluation to a user-supplied process object."""

    class DummyProcess:
        def evaluate_probes_weighted(self, probe_set: ProbeSet) -> tuple[np.ndarray, np.ndarray]:
            n_p = len(probe_set.past_pairs)
            n_f = len(probe_set.future_pairs)
            pauli_ixyz = np.zeros((n_p, n_f, 4), dtype=np.float32)
            pauli_ixyz[..., 0] = 1.0
            return pauli_ixyz, np.ones((n_p, n_f), dtype=np.float64)

    out = run_memory_characterization(
        process=DummyProcess(), cut=1, num_interventions=1, n_pasts=2, n_futures=3, rng=np.random.default_rng(7)
    )
    assert out["pauli_ixyz_ij"].shape == (2, 3, 4)
    assert out["response_matrix"].shape == (12, 2)
    assert "entropy" in out


def test_deterministic_future_weights_reduce_to_history_probability() -> None:
    """Unitary future probes do not change each history's retained probability."""
    rng = np.random.default_rng(3)
    probe_set = sample_probes(cut=2, num_interventions=3, n_pasts=5, n_futures=4, rng=rng)
    w = compute_branch_weights(probe_set)
    assert np.allclose(w.std(axis=1), 0.0, atol=1e-14)


def test_compute_branch_weight_from_steps() -> None:
    """Structured unitary steps yield unit branch weight."""
    z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    steps = [
        {"type": "unitary", "U": np.eye(2, dtype=np.complex128)},
        (z, z),
    ]
    assert _compute_branch_weight_for_sequence(steps) == pytest.approx(1.0)


def test_process_tensor_run_memory_characterization_returns_complete_weights() -> None:
    """Dense process-tensor orchestration returns positive complete-record weights."""
    rng = np.random.default_rng(0)
    op = MPO.ising(length=1, J=0.0, g=0.0)
    pt = build_process_tensor(
        op,
        _params(),
        timesteps=[0.05, 0.05],
        num_trajectories=20,
        parallel=False,
        return_type="dense",
    )
    out = run_memory_characterization(process=pt, cut=1, num_interventions=1, n_pasts=4, n_futures=3, rng=rng)
    assert "weights_ij" in out
    assert out["weights_ij"].shape == (4, 3)
    assert np.all(out["weights_ij"] > 0.0)


def test_analytic_weights_match_exact_for_trivial_dynamics() -> None:
    """Analytic branch weights match exact rollout at J=0."""
    rng = np.random.default_rng(11)
    op = MPO.ising(length=1, J=0.0, g=0.0)
    probe_set = sample_probes(
        cut=2,
        num_interventions=3,
        n_pasts=4,
        n_futures=3,
        rng=rng,
        intervention_style="haar",
    )
    w_analytic = compute_branch_weights(probe_set)
    _, w_exact, _ = simulate_exact(
        probe_set=probe_set,
        operator=op,
        sim_params=_params(),
        initial_psi=_PSI0,
        parallel=False,
    )
    np.testing.assert_allclose(w_analytic, w_exact, rtol=1e-10, atol=1e-12)


def test_dense_process_tensor_vs_exact_probe_entropy() -> None:
    """DenseProcessTensor weighted entropy agrees with exact rollout on small k."""
    rng = np.random.default_rng(42)
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = _params()
    pt = build_process_tensor(
        op,
        params,
        timesteps=[0.05, 0.05, 0.05],
        num_trajectories=50,
        parallel=False,
        return_type="dense",
    )
    assert isinstance(pt, DenseProcessTensor)
    probe_set = sample_probes(
        cut=2,
        num_interventions=2,
        n_pasts=5,
        n_futures=4,
        rng=rng,
        intervention_style="haar",
    )
    exact = ExactBackend(
        operator=op,
        sim_params=params,
        initial_psi=_PSI0,
        parallel=False,
    )
    pauli_e, weights_e, _ = simulate_exact(
        probe_set=probe_set,
        operator=exact.operator,
        sim_params=exact.sim_params,
        initial_psi=exact.initial_psi,
        parallel=exact.parallel,
    )
    response_matrix_e = assemble_response_matrix(pauli_e, weights_e)
    out_exact = compute_spectrum(response_matrix_e)
    out_pt = run_memory_characterization(process=pt, cut=2, num_interventions=2, probe_set=probe_set)
    assert out_pt["entropy"] == pytest.approx(out_exact["entropy"], rel=0.15, abs=0.05)


def test_mpo_process_tensor_entropy_matches_dense() -> None:
    """MPO and dense process-tensor backends yield the same entropy."""
    rng = np.random.default_rng(1)
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = _params()
    mpo_pt = build_process_tensor(
        op,
        params,
        timesteps=[0.05, 0.05],
        num_trajectories=40,
        parallel=False,
        return_type="mpo",
        compress_every=1,
    )
    assert isinstance(mpo_pt, MPOProcessTensor)
    dense = mpo_pt.to_dense()
    probe_set = sample_probes(cut=1, num_interventions=1, n_pasts=4, n_futures=3, rng=rng)
    out_mpo = run_memory_characterization(process=mpo_pt, cut=1, num_interventions=1, probe_set=probe_set)
    out_dense = run_memory_characterization(process=dense, cut=1, num_interventions=1, probe_set=probe_set)
    assert out_mpo["entropy"] == pytest.approx(out_dense["entropy"], rel=1e-10, abs=1e-10)


def test_evaluate_probes_with_weights_process_tensor_uses_subnormalized_weights() -> None:
    """Process tensors supply complete weights from subnormalized contractions."""
    rng = np.random.default_rng(2)
    op = MPO.ising(length=1, J=0.0, g=0.0)
    pt = build_process_tensor(
        op,
        _params(),
        timesteps=[0.05, 0.05],
        num_trajectories=20,
        parallel=False,
        return_type="dense",
    )
    probe_set = sample_probes(cut=1, num_interventions=1, n_pasts=3, n_futures=2, rng=rng)
    pauli, weights = evaluate_probes_with_weights(pt, probe_set)
    assert pauli.shape == (3, 2, 4)
    assert weights.shape == (3, 2)
    assert np.allclose(weights.std(axis=1), 0.0)


def test_evaluate_probes_with_weights_missing_method_raises() -> None:
    """Objects without weighted probe responses raise TypeError."""

    class NoProbes:
        pass

    probe_set = sample_probes(cut=1, num_interventions=1, n_pasts=2, n_futures=2, rng=np.random.default_rng(0))
    with pytest.raises(TypeError, match="evaluate_probes_weighted"):
        evaluate_probes_with_weights(cast("OperationalMemoryBackend", NoProbes()), probe_set)


def test_evaluate_probes_with_weights_rejects_normalized_only_backend() -> None:
    """Normalized responses alone cannot determine retained-outcome probabilities."""

    class UnweightedBackend:
        def evaluate_probes(self, probe_set: ProbeSet) -> np.ndarray:
            return np.zeros((len(probe_set.past_pairs), len(probe_set.future_pairs), 4), dtype=np.float64)

    probe_set = sample_probes(cut=1, num_interventions=1, n_pasts=2, n_futures=2, rng=np.random.default_rng(0))
    with pytest.raises(TypeError, match="normalized probe responses do not determine"):
        evaluate_probes_with_weights(cast("OperationalMemoryBackend", UnweightedBackend()), probe_set)


def test_evaluate_probes_with_weights_inherited_method() -> None:
    """Subclasses that inherit probe methods dispatch without TypeError."""

    class BaseBackend:
        def evaluate_probes_weighted(self, probe_set: ProbeSet) -> tuple[np.ndarray, np.ndarray]:
            n_p = len(probe_set.past_pairs)
            n_f = len(probe_set.future_pairs)
            pauli_ixyz = np.zeros((n_p, n_f, 4), dtype=np.float32)
            pauli_ixyz[..., 0] = 1.0
            return pauli_ixyz, np.ones((n_p, n_f), dtype=np.float64)

    class ChildBackend(BaseBackend):
        pass

    probe_set = sample_probes(cut=1, num_interventions=1, n_pasts=2, n_futures=2, rng=np.random.default_rng(0))
    pauli, weights = evaluate_probes_with_weights(cast("OperationalMemoryBackend", ChildBackend()), probe_set)
    assert pauli.shape == (2, 2, 4)
    assert weights.shape == (2, 2)


def test_run_memory_characterization_parallel_override_does_not_mutate_backend() -> None:
    """A one-shot parallel=False override must not change ExactBackend defaults."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    backend = ExactBackend(operator=op, sim_params=_params(), initial_psi=_PSI0, parallel=True)
    probe_set = sample_probes(cut=1, num_interventions=1, n_pasts=2, n_futures=2, rng=np.random.default_rng(0))
    run_memory_characterization(process=backend, cut=1, num_interventions=1, probe_set=probe_set, parallel=False)
    assert backend.parallel is True


def test_run_memory_characterization_returns_raw_matrix() -> None:
    """run_memory_characterization exposes one canonical raw response matrix."""
    rng = np.random.default_rng(9)
    op = MPO.ising(length=1, J=0.0, g=0.0)
    pt = build_process_tensor(
        op,
        _params(),
        timesteps=[0.05, 0.05],
        num_trajectories=20,
        parallel=False,
        return_type="dense",
    )
    out = run_memory_characterization(
        process=pt,
        cut=1,
        num_interventions=1,
        n_pasts=3,
        n_futures=2,
        rng=rng,
    )
    expected = assemble_response_matrix(out["pauli_ixyz_ij"], out["weights_ij"])
    np.testing.assert_allclose(out["response_matrix"], expected)
    np.testing.assert_allclose(out["response_matrix"][0::4], out["weights_ij"].T)
    assert "response_matrix_raw" not in out


def _entropy_from_cumulative_weights(
    probe_set: ProbeSet,
    op: MPO,
    params: AnalogSimParams,
    psi0: np.ndarray,
) -> float:
    """Entropy using cumulative_weight_final from exact simulation diagnostics.

    Args:
        probe_set: Split-cut probe bundle to simulate.
        op: Hamiltonian MPO.
        params: Analog simulation parameters.
        psi0: Initial state vector.

    Returns:
        Von Neumann entropy of the assembled memory matrix.
    """
    pauli, _, simulation_diagnostics = simulate_exact(
        probe_set=probe_set,
        operator=op,
        sim_params=params,
        initial_psi=psi0,
        parallel=False,
    )
    n_p, n_f = pauli.shape[:2]
    weights = np.zeros((n_p, n_f), dtype=np.float64)
    for ii in range(n_p):
        for jj in range(n_f):
            weights[ii, jj] = _diagnostics_final_weight(simulation_diagnostics[ii * n_f + jj])
    response_matrix = assemble_response_matrix(pauli, weights)
    return float(compute_spectrum(response_matrix)["entropy"])


def test_run_memory_characterization_matches_cumulative_weight_entropy() -> None:
    """Exact-backend orchestration agrees with cumulative-weight entropy at J=0."""
    rng = np.random.default_rng(4)
    op = MPO.ising(length=2, J=0.0, g=1.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8, order=1)
    probe_set = sample_probes(cut=2, num_interventions=4, n_pasts=4, n_futures=3, rng=rng)
    psi0 = np.zeros(4, dtype=np.complex128)
    psi0[0] = 1.0 + 0.0j
    backend = ExactBackend(operator=op, sim_params=params, initial_psi=psi0, parallel=False)
    out = run_memory_characterization(process=backend, cut=2, num_interventions=4, probe_set=probe_set)
    exp = _entropy_from_cumulative_weights(probe_set, op, params, psi0)
    assert out["entropy"] == pytest.approx(exp, rel=1e-10, abs=1e-10)
    np.testing.assert_allclose(out["response_matrix"][0::4], out["weights_ij"].T)


def test_run_memory_characterization_rejects_mismatched_probe_set() -> None:
    """Supplied probe_set must match the requested cut and k."""
    rng = np.random.default_rng(0)
    probe_set = sample_probes(cut=1, num_interventions=2, n_pasts=2, n_futures=2, rng=rng)

    class DummyProcess:
        def evaluate_probes_weighted(self, probe_set: ProbeSet) -> tuple[np.ndarray, np.ndarray]:
            n_p = len(probe_set.past_pairs)
            n_f = len(probe_set.future_pairs)
            return np.zeros((n_p, n_f, 4), dtype=np.float64), np.ones((n_p, n_f), dtype=np.float64)

    with pytest.raises(ValueError, match="probe_set was built for"):
        run_memory_characterization(process=DummyProcess(), cut=2, num_interventions=2, probe_set=probe_set)


def test_evaluate_probes_with_weights_preserves_float64() -> None:
    """Probe responses are not downcast to float32 before memory assembly."""

    class HighPrecisionBackend:
        def evaluate_probes_weighted(self, probe_set: ProbeSet) -> tuple[np.ndarray, np.ndarray]:
            n_p = len(probe_set.past_pairs)
            n_f = len(probe_set.future_pairs)
            out = np.zeros((n_p, n_f, 4), dtype=np.float64)
            out[..., 0] = 1.0
            out[..., 1] = 1e-7
            return out, np.ones((n_p, n_f), dtype=np.float64)

    probe_set = sample_probes(cut=1, num_interventions=1, n_pasts=1, n_futures=1, rng=np.random.default_rng(0))
    pauli, _weights = evaluate_probes_with_weights(HighPrecisionBackend(), probe_set)
    assert pauli.dtype == np.float64
    assert pauli[0, 0, 1] == pytest.approx(1e-7)


def test_run_memory_characterization_delay_rejects_negative() -> None:
    """Negative reset delay is rejected before simulation."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    backend = ExactBackend(operator=op, sim_params=_params(), initial_psi=_PSI0, parallel=False)
    with pytest.raises(ValueError, match="delay must be >= 0"):
        run_memory_characterization(process=backend, cut=1, num_interventions=2, delay=-1)


def test_run_memory_characterization_delay_rejects_process_tensor_backend() -> None:
    """Reset delay requires the exact sequence backend."""
    rng = np.random.default_rng(0)
    op = MPO.ising(length=1, J=0.0, g=0.0)
    pt = build_process_tensor(
        op,
        _params(),
        timesteps=[0.05, 0.05],
        num_trajectories=20,
        parallel=False,
        return_type="dense",
    )
    probe_set = sample_probes(cut=1, num_interventions=2, n_pasts=2, n_futures=2, rng=rng)
    with pytest.raises(ValueError, match="delay > 0 requires an exact Hamiltonian"):
        run_memory_characterization(process=pt, cut=1, num_interventions=2, probe_set=probe_set, delay=1)


def test_run_memory_characterization_delay_zero_matches_default() -> None:
    """Explicit delay=0 matches the default split-cut path."""
    rng = np.random.default_rng(6)
    op = MPO.ising(length=2, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8, order=1)
    psi0 = np.zeros(4, dtype=np.complex128)
    psi0[0] = 1.0 + 0.0j
    probe_set = sample_probes(cut=2, num_interventions=4, n_pasts=3, n_futures=2, rng=rng)
    backend = ExactBackend(operator=op, sim_params=params, initial_psi=psi0, parallel=False)
    out_default = run_memory_characterization(process=backend, cut=2, num_interventions=4, probe_set=probe_set)
    out_zero = run_memory_characterization(process=backend, cut=2, num_interventions=4, probe_set=probe_set, delay=0)
    assert out_zero["entropy"] == pytest.approx(out_default["entropy"], rel=1e-10, abs=1e-10)


def test_run_memory_characterization_delay_exact_returns_finite_entropy() -> None:
    """Exact backend accepts delay>0 and returns finite memory diagnostics."""
    rng = np.random.default_rng(7)
    op = MPO.ising(length=2, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8, order=1)
    psi0 = np.zeros(4, dtype=np.complex128)
    psi0[0] = 1.0 + 0.0j
    probe_set = sample_probes(cut=3, num_interventions=5, n_pasts=3, n_futures=2, rng=rng)
    backend = ExactBackend(operator=op, sim_params=params, initial_psi=psi0, parallel=False)
    out = run_memory_characterization(process=backend, cut=3, num_interventions=5, probe_set=probe_set, delay=2)
    assert np.isfinite(out["entropy"])
    assert out["pauli_ixyz_ij"].shape == (3, 2, 4)
