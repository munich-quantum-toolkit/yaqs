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
    simulate_exact,
)
from mqt.yaqs.characterization.memory.operational_memory.samples import (
    ProbeSet,
    _sample_cut_measurement_only,  # noqa: PLC2701
    _sample_cut_preparation_only,  # noqa: PLC2701
    _sample_probe_step,  # noqa: PLC2701
    resolve_unitary_sampler,
    sample_probes,
)
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
    k: int,
    n_pasts: int,
    n_futures: int,
    rng: np.random.Generator,
) -> tuple[ProbeSet, list[list[Any]]]:
    """Delayed causal-break probes: past + break + identity bridge + future.

    Returns:
        Tuple of probe set metadata and flat sequence grid for ``simulate_exact``.
    """
    c_left = int(left_cut)
    tt = int(tau)
    kk = int(k)
    past_full = c_left - 1
    future_tail = kk - (c_left + tt + 1)
    unitary_sampler = resolve_unitary_sampler("haar")

    past_pairs: list[list[Any]] = []
    past_cut_meas: list[np.ndarray] = []
    for _ in range(n_pasts):
        pairs_i = [
            _sample_probe_step(rng, intervention_mode="unitary_break_mp", unitary_sampler=unitary_sampler)[1]
            for _ in range(past_full)
        ]
        _feat_m, psi_m = _sample_cut_measurement_only(rng)
        past_cut_meas.append(psi_m)
        past_pairs.append(pairs_i)

    future_prep_cut: list[np.ndarray] = []
    future_pairs: list[list[Any]] = []
    for _ in range(n_futures):
        _feat_p, psi_p = _sample_cut_preparation_only(rng)
        future_prep_cut.append(psi_p)
        future_pairs.append([
            _sample_probe_step(rng, intervention_mode="unitary_break_mp", unitary_sampler=unitary_sampler)[1]
            for _ in range(future_tail)
        ])

    z0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    u_id = np.eye(2, dtype=np.complex128)
    bridge = [{"type": "unitary", "U": u_id} for _ in range(tt)]
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
        cut=c_left,
        k=kk,
        past_features=np.zeros((n_pasts, max(1, past_full + 1), 32), dtype=np.float32),
        future_features=np.zeros((n_futures, max(1, 1 + tt + future_tail), 32), dtype=np.float32),
        past_pairs=past_pairs,
        past_cut_meas=past_cut_meas,
        future_prep_cut=future_prep_cut,
        future_pairs=future_pairs,
    )
    return probe_set, all_pairs


def _make_minimal_probe_set(*, cut: int = 1, k: int = 1, n_p: int = 2, n_f: int = 3) -> ProbeSet:
    """Build a tiny ProbeSet with empty unitary legs and |0> cut kets.

    Args:
        cut: Causal cut index.
        k: Intervention sequence length.
        n_p: Number of past probe rows.
        n_f: Number of future probe rows.

    Returns:
        Probe set suitable for ExactBackend smoke tests.
    """
    z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    return ProbeSet(
        cut=cut,
        k=k,
        past_features=np.zeros((n_p, cut, 32), dtype=np.float32),
        future_features=np.zeros((n_f, k - cut + 1, 32), dtype=np.float32),
        past_pairs=[[] for _ in range(n_p)],
        past_cut_meas=[z.copy() for _ in range(n_p)],
        future_prep_cut=[z.copy() for _ in range(n_f)],
        future_pairs=[[] for _ in range(n_f)],
    )


def test_exact_run_operational_memory_hides_static_ctx_parameter() -> None:
    """ExactBackend builds static context internally instead of exposing it."""
    sig = inspect.signature(ExactBackend.__init__)
    assert "static_ctx" not in sig.parameters
    assert "initial_psi" in sig.parameters


def test_exact_run_operational_memory_builds_static_ctx_internally(monkeypatch: pytest.MonkeyPatch) -> None:
    """ExactBackend wires make_mcwf_static_context and simulate_sequences internally."""
    calls: dict[str, Any] = {}

    def _fake_make_ctx(operator: object, sim_params: object, noise_model: object | None = None) -> str:
        calls["ctx_args"] = (operator, sim_params, noise_model)
        return "CTX"

    def _fake_simulate_sequences(**kwargs) -> np.ndarray | tuple[np.ndarray, list[dict[str, object]]]:  # noqa: ANN003
        calls["simulate_kwargs"] = kwargs
        n_tot = len(kwargs["psi_pairs_list"])
        packed = np.zeros((n_tot, 8), dtype=np.float32)
        if kwargs.get("traced"):
            traces = cast(
                "list[dict[str, object]]",
                [{"step_probs": [1.0], "cumulative_weight_final": 1.0} for _ in range(n_tot)],
            )
            return packed, traces
        return packed

    monkeypatch.setattr(exact_mod, "make_mcwf_static_context", _fake_make_ctx)
    monkeypatch.setattr(exact_mod, "simulate_sequences", _fake_simulate_sequences)

    op = MPO.ising(length=1, J=0.0, g=0.0)
    sim = AnalogSimParams(dt=0.1)
    psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    process = ExactBackend(operator=op, sim_params=sim, initial_psi=psi0, parallel=False)
    probe_set = _make_minimal_probe_set(cut=1, k=1, n_p=2, n_f=3)
    out = process.evaluate_probes(probe_set)

    assert out.shape == (2, 3, 4)
    assert calls["ctx_args"] == (op, sim, None)
    assert calls["simulate_kwargs"]["static_ctx"] == "CTX"


def test_exact_diagnostics_use_cut_branch_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """simulate_exact weights prod(step_probs[:cut])."""

    def _fake_simulate(**kwargs) -> tuple[np.ndarray, list[dict[str, object]]]:  # noqa: ANN003
        n_tot = len(kwargs["psi_pairs_list"])
        traces = cast(
            "list[dict[str, object]]",
            [{"step_probs": [0.5, 0.8, 1.0], "cumulative_weight_final": 0.99} for _ in range(n_tot)],
        )
        return np.zeros((n_tot, 8), dtype=np.float32), traces

    monkeypatch.setattr(exact_mod, "simulate_sequences", _fake_simulate)

    z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    probe_set = ProbeSet(
        cut=2,
        k=2,
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


def test_exact_run_operational_memory_parallel_smoke() -> None:
    """ExactBackend completes a tiny parallel rollout batch."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    sim = AnalogSimParams(dt=0.1)
    psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    process = ExactBackend(operator=op, sim_params=sim, initial_psi=psi0, parallel=True, show_progress=False)
    probe_set = _make_minimal_probe_set(cut=1, k=1, n_p=2, n_f=2)
    out = process.evaluate_probes(probe_set)
    assert out.shape == (2, 2, 4)


def test_delayed_break_custom_psi_pairs_list_geometry() -> None:
    """Gap geometry builds k-length sequences with an identity bridge of length tau."""
    rng = np.random.default_rng(1)
    left_cut, tau, k = 4, 2, 10
    probe_set, psi_pairs_list = _sample_split_delayed_break_probes(
        left_cut=left_cut,
        tau=tau,
        k=k,
        n_pasts=3,
        n_futures=2,
        rng=rng,
    )
    assert probe_set.cut == left_cut
    assert probe_set.k == k
    assert len(psi_pairs_list) == 6
    u_id = np.eye(2, dtype=np.complex128)
    for seq in psi_pairs_list:
        assert len(seq) == k
        bridge = seq[left_cut : left_cut + tau]
        assert all(step.get("type") == "unitary" and np.array_equal(step["U"], u_id) for step in bridge)
        assert isinstance(seq[left_cut + tau], tuple)


def test_delayed_break_soft_future_prepare_retains_past_response() -> None:
    """Right-cut preparation uses (|0>, sigma_p) so past rows differ in final tomography."""
    rng = np.random.default_rng(4)
    probe_set, psi_pairs_list = _sample_split_delayed_break_probes(
        left_cut=4,
        tau=0,
        k=10,
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
        psi_pairs_list=psi_pairs_list,
    )
    past_std = float(np.std(pauli[:, 0, 1:4], axis=0).mean())
    future_std = float(np.std(pauli[0, :, 1:4], axis=0).mean())
    assert past_std > 1e-4
    assert future_std > 1e-4


def test_simulate_exact_accepts_custom_psi_pairs_list() -> None:
    """simulate_exact rolls out delayed-break probe grids when psi_pairs_list is supplied."""
    rng = np.random.default_rng(2)
    probe_set, psi_pairs_list = _sample_split_delayed_break_probes(
        left_cut=3,
        tau=1,
        k=8,
        n_pasts=3,
        n_futures=2,
        rng=rng,
    )
    op = MPO.ising(length=2, J=0.5, g=1.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=12, order=1)
    pauli, weights, traces = simulate_exact(
        probe_set=probe_set,
        operator=op,
        sim_params=params,
        initial_psi=_product_initial_state(2),
        parallel=False,
        psi_pairs_list=psi_pairs_list,
    )
    assert pauli.shape[:2] == (3, 2)
    assert weights.shape == (3, 2)
    assert len(traces) == 6
    assert all("cumulative_weight_final" in t for t in traces)


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


def test_simulate_exact_rejects_mismatched_psi_pairs_list() -> None:
    """Custom psi_pairs_list length must match the probe grid."""
    rng = np.random.default_rng(0)
    probe_set = sample_probes(cut=1, k=1, n_pasts=2, n_futures=2, rng=rng)
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1)
    with pytest.raises(ValueError, match="psi_pairs_list length"):
        simulate_exact(
            probe_set=probe_set,
            operator=op,
            sim_params=params,
            initial_psi=np.array([1.0, 0.0], dtype=np.complex128),
            parallel=False,
            psi_pairs_list=[[(np.array([1.0, 0.0]), np.array([1.0, 0.0]))]],
        )


def test_simulate_exact_preserves_float64_probe_coefficients() -> None:
    """Exact probe decoding keeps float64 precision through to the memory matrix."""
    rng = np.random.default_rng(0)
    probe_set = sample_probes(cut=1, k=1, n_pasts=2, n_futures=2, rng=rng)
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1)
    pauli, _weights, _traces = simulate_exact(
        probe_set=probe_set,
        operator=op,
        sim_params=params,
        initial_psi=np.array([1.0, 0.0], dtype=np.complex128),
        parallel=False,
    )
    assert pauli.dtype == np.float64
