# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for simulator reference probe backends."""

from __future__ import annotations

import inspect
from typing import Any, cast

import numpy as np
import pytest

import mqt.yaqs.characterization.memory.backends.exact as exact_mod
from mqt.yaqs.characterization.memory.backends.exact import (
    ExactBackend,
    _branch_weights_from_simulation,  # noqa: PLC2701
    simulate_exact,
)
from mqt.yaqs.characterization.memory.operational_memory.samples import (
    ProbeSet,
    sample_cut_measurement,
    sample_cut_preparation,
    sample_probe,
    sample_probes,
)
from mqt.yaqs.characterization.memory.shared.encoding import SITE0_KET
from mqt.yaqs.characterization.memory.shared.utils import validate_stochastic_solver
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


def _product_initial_state(length: int) -> np.ndarray:
    """Build |0...0> as a length-``2**length`` state vector.

    Args:
        length: Number of qubits in the product state.

    Returns:
        Normalized computational-basis state vector.
    """
    psi = np.zeros(2**length, dtype=np.complex128)
    psi[0] = 1.0 + 0.0j
    return psi


def _sample_split_delayed_break_probes(
    *,
    left_cut: int,
    tau: int,
    num_interventions: int,
    n_pasts: int,
    n_futures: int,
    rng: np.random.Generator,
) -> tuple[ProbeSet, list[list[Any]]]:
    """Delayed causal-break probes: past + break + identity bridge + future.

    Returns:
        Tuple of probe set metadata and flat sequence grid for ``simulate_exact``.
    """
    past_len = left_cut - 1
    bridge_len = tau
    future_tail = num_interventions - (left_cut + bridge_len + 1)

    past_pairs: list[list[Any]] = []
    past_cut_meas: list[np.ndarray] = []
    for _ in range(n_pasts):
        pairs_i = [sample_probe(rng, intervention_style="haar")[1] for _ in range(past_len)]
        _feat_m, psi_m = sample_cut_measurement(rng)
        past_cut_meas.append(psi_m)
        past_pairs.append(pairs_i)

    future_prep_cut: list[np.ndarray] = []
    future_pairs: list[list[Any]] = []
    for _ in range(n_futures):
        _feat_p, psi_p = sample_cut_preparation(rng)
        future_prep_cut.append(psi_p)
        future_pairs.append([sample_probe(rng, intervention_style="haar")[1] for _ in range(future_tail)])

    z0 = SITE0_KET
    u_id = np.eye(2, dtype=np.complex128)
    bridge = [{"type": "unitary", "U": u_id} for _ in range(bridge_len)]
    all_pairs: list[list[Any]] = []
    for i in range(n_pasts):
        for j in range(n_futures):
            full = list(past_pairs[i])
            full.append((past_cut_meas[i], z0))
            full.extend(bridge)
            full.append((z0, np.asarray(future_prep_cut[j], dtype=np.complex128)))
            full.extend(future_pairs[j])
            all_pairs.append(full)

    probe_set = ProbeSet(
        cut=left_cut,
        num_interventions=num_interventions,
        past_features=np.zeros((n_pasts, max(1, past_len + 1), 32), dtype=np.float32),
        future_features=np.zeros((n_futures, max(1, 1 + bridge_len + future_tail), 32), dtype=np.float32),
        past_pairs=past_pairs,
        past_cut_meas=past_cut_meas,
        future_prep_cut=future_prep_cut,
        future_pairs=future_pairs,
    )
    return probe_set, all_pairs


def _make_minimal_probe_set(*, cut: int = 1, num_interventions: int = 1, n_p: int = 2, n_f: int = 3) -> ProbeSet:
    """Build a tiny ProbeSet with empty unitary legs and |0> cut kets.

    Args:
        cut: Causal cut index.
        num_interventions: Intervention sequence length.
        n_p: Number of past probe rows.
        n_f: Number of future probe rows.

    Returns:
        Probe set suitable for ExactBackend smoke tests.
    """
    z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    return ProbeSet(
        cut=cut,
        num_interventions=num_interventions,
        past_features=np.zeros((n_p, cut, 32), dtype=np.float32),
        future_features=np.zeros((n_f, num_interventions - cut + 1, 32), dtype=np.float32),
        past_pairs=[[] for _ in range(n_p)],
        past_cut_meas=[z.copy() for _ in range(n_p)],
        future_prep_cut=[z.copy() for _ in range(n_f)],
        future_pairs=[[] for _ in range(n_f)],
    )


def test_exact_run_memory_characterization_hides_static_ctx_parameter() -> None:
    """ExactBackend builds static context internally instead of exposing it."""
    sig = inspect.signature(ExactBackend.__init__)
    assert "static_ctx" not in sig.parameters
    assert "initial_psi" in sig.parameters


def test_exact_run_memory_characterization_builds_static_ctx_internally(monkeypatch: pytest.MonkeyPatch) -> None:
    """ExactBackend wires make_mcwf_static_context and simulate_sequences internally."""
    calls: dict[str, Any] = {}

    def _fake_make_ctx(operator: object, sim_params: object, noise_model: object | None = None) -> str:
        calls["ctx_args"] = (operator, sim_params, noise_model)
        return "CTX"

    def _fake_simulate_sequences(**kwargs) -> np.ndarray | tuple[np.ndarray, list[dict[str, object]]]:  # noqa: ANN003
        calls["simulate_kwargs"] = kwargs
        n_tot = len(kwargs["intervention_steps_list"])
        packed = np.zeros((n_tot, 8), dtype=np.float32)
        if kwargs.get("record_diagnostics"):
            simulation_diagnostics = cast(
                "list[dict[str, object]]",
                [{"step_probs": [1.0], "cumulative_weight_final": 1.0} for _ in range(n_tot)],
            )
            return packed, simulation_diagnostics
        return packed

    monkeypatch.setattr(exact_mod, "make_mcwf_static_context", _fake_make_ctx)
    monkeypatch.setattr(exact_mod, "simulate_sequences", _fake_simulate_sequences)

    op = MPO.ising(length=1, J=0.0, g=0.0)
    sim = AnalogSimParams(dt=0.1)
    psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    process = ExactBackend(operator=op, sim_params=sim, initial_psi=psi0, parallel=False)
    probe_set = _make_minimal_probe_set(cut=1, num_interventions=1, n_p=2, n_f=3)
    out = process.evaluate_probes(probe_set)

    assert out.shape == (2, 3, 4)
    assert calls["ctx_args"] == (op, sim, None)
    assert calls["simulate_kwargs"]["static_ctx"] == "CTX"


def test_exact_diagnostics_use_cut_branch_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """simulate_exact weights prod(step_probs[:cut])."""

    def _fake_simulate(**kwargs) -> tuple[np.ndarray, list[dict[str, object]]]:  # noqa: ANN003
        n_tot = len(kwargs["intervention_steps_list"])
        simulation_diagnostics = cast(
            "list[dict[str, object]]",
            [{"step_probs": [0.5, 0.8, 1.0], "cumulative_weight_final": 0.99} for _ in range(n_tot)],
        )
        return np.zeros((n_tot, 8), dtype=np.float32), simulation_diagnostics

    monkeypatch.setattr(exact_mod, "simulate_sequences", _fake_simulate)

    z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    probe_set = ProbeSet(
        cut=2,
        num_interventions=2,
        past_features=np.zeros((1, 2, 32), dtype=np.float32),
        future_features=np.zeros((1, 1, 32), dtype=np.float32),
        past_pairs=[[(z, z)]],
        past_cut_meas=[z.copy()],
        future_prep_cut=[z.copy()],
        future_pairs=[[]],
    )
    op = MPO.ising(length=1, J=0.0, g=0.0)
    sim = AnalogSimParams(dt=0.1)
    psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    _, weights, _ = simulate_exact(
        probe_set=probe_set,
        operator=op,
        sim_params=sim,
        initial_psi=psi0,
        parallel=False,
    )
    assert weights.shape == (1, 1)
    assert float(weights[0, 0]) == pytest.approx(0.4)


def test_exact_backend_rejects_invalid_solver() -> None:
    """Invalid solver strings fail at backend construction."""
    with pytest.raises(ValueError, match="solver must be"):
        validate_stochastic_solver("typo")


def test_exact_run_memory_characterization_parallel_smoke() -> None:
    """ExactBackend completes a tiny parallel rollout batch."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    sim = AnalogSimParams(dt=0.1)
    psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    process = ExactBackend(operator=op, sim_params=sim, initial_psi=psi0, parallel=True, show_progress=False)
    probe_set = _make_minimal_probe_set(cut=1, num_interventions=1, n_p=2, n_f=2)
    out = process.evaluate_probes(probe_set)
    assert out.shape == (2, 2, 4)


def test_delayed_break_custom_intervention_steps_list_geometry() -> None:
    """Gap geometry builds k-length sequences with an identity bridge of length tau."""
    rng = np.random.default_rng(1)
    left_cut, tau, k = 4, 2, 10
    probe_set, intervention_steps_list = _sample_split_delayed_break_probes(
        left_cut=left_cut,
        tau=tau,
        num_interventions=k,
        n_pasts=3,
        n_futures=2,
        rng=rng,
    )
    assert probe_set.cut == left_cut
    assert probe_set.num_interventions == k
    assert len(intervention_steps_list) == 6
    u_id = np.eye(2, dtype=np.complex128)
    for seq in intervention_steps_list:
        assert len(seq) == k
        bridge = seq[left_cut : left_cut + tau]
        assert all(step.get("type") == "unitary" and np.array_equal(step["U"], u_id) for step in bridge)
        assert isinstance(seq[left_cut + tau], tuple)


def test_delayed_break_soft_future_prepare_retains_past_response() -> None:
    """Right-cut preparation uses (|0>, sigma_p) so past rows differ in final tomography."""
    rng = np.random.default_rng(4)
    probe_set, intervention_steps_list = _sample_split_delayed_break_probes(
        left_cut=4,
        tau=0,
        num_interventions=10,
        n_pasts=8,
        n_futures=4,
        rng=rng,
    )
    op = MPO.ising(length=2, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=12, order=1)
    pauli, _, _ = simulate_exact(
        probe_set=probe_set,
        operator=op,
        sim_params=params,
        initial_psi=_product_initial_state(2),
        parallel=False,
        intervention_steps_list=intervention_steps_list,
    )
    past_std = float(np.std(pauli[:, 0, 1:4], axis=0).mean())
    future_std = float(np.std(pauli[0, :, 1:4], axis=0).mean())
    assert past_std > 1e-4
    assert future_std > 1e-4


def test_simulate_exact_accepts_custom_intervention_steps_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """simulate_exact uses a supplied intervention_steps_list instead of assemble_probe_grid."""

    def _fail_assemble(*_args: object, **_kwargs: object) -> None:
        msg = "assemble_probe_grid must not be called when intervention_steps_list is provided"
        raise AssertionError(msg)

    monkeypatch.setattr(exact_mod, "assemble_probe_grid", _fail_assemble)

    rng = np.random.default_rng(2)
    probe_set, intervention_steps_list = _sample_split_delayed_break_probes(
        left_cut=3,
        tau=1,
        num_interventions=8,
        n_pasts=3,
        n_futures=2,
        rng=rng,
    )
    op = MPO.ising(length=2, J=0.5, g=1.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=12, order=1)
    pauli, weights, simulation_diagnostics = simulate_exact(
        probe_set=probe_set,
        operator=op,
        sim_params=params,
        initial_psi=_product_initial_state(2),
        parallel=False,
        intervention_steps_list=intervention_steps_list,
    )
    assert pauli.shape[:2] == (3, 2)
    assert weights.shape == (3, 2)
    assert len(simulation_diagnostics) == 6
    assert all("cumulative_weight_final" in d for d in simulation_diagnostics)
    u_id = np.eye(2, dtype=np.complex128)
    for seq in intervention_steps_list:
        assert any(
            isinstance(step, dict) and step.get("type") == "unitary" and np.array_equal(step["U"], u_id) for step in seq
        )


def test_exact_backend_execution_config_override() -> None:
    """execution_config merges one-shot parallel overrides."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1)
    backend = ExactBackend(
        operator=op,
        sim_params=params,
        initial_psi=np.array([1.0, 0.0], dtype=np.complex128),
        parallel=True,
    )
    assert backend.execution_config().parallel is True
    assert backend.execution_config(parallel=False).parallel is False


def _diagnostics_final_weight(diagnostics: dict[str, object]) -> float:
    """Extract the final cumulative weight from simulation diagnostics.

    Args:
        diagnostics: Per-sequence diagnostics dict from :func:`simulate_exact`.

    Returns:
        Final cumulative branch weight as a float.

    Raises:
        TypeError: If ``cumulative_weight_final`` is not numeric.
    """
    val = diagnostics["cumulative_weight_final"]
    if not isinstance(val, (int, float)):
        msg = "cumulative_weight_final must be numeric"
        raise TypeError(msg)
    return float(val)


def _cumulative_weights_from_simulation_diagnostics(
    simulation_diagnostics: list[dict[str, object]],
    *,
    n_pasts: int,
    n_futures: int,
) -> np.ndarray:
    """Mirror experiments/_benchmark_memory.py cumulative_weight_final weighting.

    Args:
        simulation_diagnostics: Flat list of per-(past, future) diagnostics from
            :func:`simulate_exact`.
        n_pasts: Number of past probe rows.
        n_futures: Number of future probe columns.

    Returns:
        Branch-weight matrix of shape ``(n_pasts, n_futures)``.
    """
    n_p, n_f = n_pasts, n_futures
    weights = np.zeros((n_p, n_f), dtype=np.float64)
    for ii in range(n_p):
        for jj in range(n_f):
            weights[ii, jj] = _diagnostics_final_weight(simulation_diagnostics[ii * n_f + jj])
    return weights


_PSI0_L2 = np.zeros(4, dtype=np.complex128)
_PSI0_L2[0] = 1.0 + 0.0j


def test_simulation_branch_weights_match_cumulative_final_split_cut_unitary() -> None:
    """Paper metric path: cumulative_weight_final agrees with cut-truncated step_probs."""
    rng = np.random.default_rng(21)
    op = MPO.ising(length=2, J=0.5, g=1.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=12, order=1)
    probe_set = sample_probes(
        cut=3,
        num_interventions=5,
        n_pasts=4,
        n_futures=3,
        rng=rng,
        intervention_style="haar",
    )
    _, weights_cut, simulation_diagnostics = simulate_exact(
        probe_set=probe_set,
        operator=op,
        sim_params=params,
        initial_psi=_PSI0_L2,
        parallel=False,
    )
    w_cumulative = _cumulative_weights_from_simulation_diagnostics(
        simulation_diagnostics,
        n_pasts=len(probe_set.past_pairs),
        n_futures=len(probe_set.future_pairs),
    )
    w_sim = _branch_weights_from_simulation(
        simulation_diagnostics,
        n_pasts=len(probe_set.past_pairs),
        n_futures=len(probe_set.future_pairs),
        cut=probe_set.cut,
    )
    np.testing.assert_allclose(w_sim, weights_cut, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(w_cumulative, weights_cut, rtol=1e-10, atol=1e-12)


def test_exact_weights_positive_l2_quick_geometry() -> None:
    """L=2 paper quick geometry yields positive branch weights from exact rollouts."""
    rng = np.random.default_rng(44)
    op = MPO.ising(length=2, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=12, order=1)
    probe_set = sample_probes(
        cut=4,
        num_interventions=8,
        n_pasts=4,
        n_futures=3,
        rng=rng,
        intervention_style="haar",
    )
    _, weights, simulation_diagnostics = simulate_exact(
        probe_set=probe_set,
        operator=op,
        sim_params=params,
        initial_psi=_PSI0_L2,
        parallel=False,
    )
    assert np.all(weights > 0.0)
    assert np.allclose(weights.std(axis=1), 0.0)
    assert all(float(d["cumulative_weight_final"]) > 0.0 for d in simulation_diagnostics)


def test_simulate_exact_rejects_mismatched_intervention_steps_list() -> None:
    """Custom intervention_steps_list length must match the probe grid."""
    rng = np.random.default_rng(0)
    probe_set = sample_probes(cut=1, num_interventions=1, n_pasts=2, n_futures=2, rng=rng)
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1)
    with pytest.raises(ValueError, match="intervention_steps_list length"):
        simulate_exact(
            probe_set=probe_set,
            operator=op,
            sim_params=params,
            initial_psi=np.array([1.0, 0.0], dtype=np.complex128),
            parallel=False,
            intervention_steps_list=[[(np.array([1.0, 0.0]), np.array([1.0, 0.0]))]],
        )


def test_simulate_exact_preserves_float64_probe_coefficients() -> None:
    """Exact probe decoding keeps float64 precision through to the memory matrix."""
    rng = np.random.default_rng(0)
    probe_set = sample_probes(cut=1, num_interventions=1, n_pasts=2, n_futures=2, rng=rng)
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1)
    pauli, _weights, _simulation_diagnostics = simulate_exact(
        probe_set=probe_set,
        operator=op,
        sim_params=params,
        initial_psi=np.array([1.0, 0.0], dtype=np.complex128),
        parallel=False,
    )
    assert pauli.dtype == np.float64
