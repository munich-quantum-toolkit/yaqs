# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for simulator reference probe backends."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import numpy as np

import mqt.yaqs.characterization.memory.reference.exact as exact_mod
from mqt.yaqs.characterization.memory.diagnostics.probe import ProbeSet
from mqt.yaqs.characterization.memory.reference.exact import (
    ExactProbeProcess,
    evaluate_exact_probe_set_with_diagnostics,
)
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

if TYPE_CHECKING:
    import pytest


def _make_minimal_probe_set(*, cut: int = 1, k: int = 1, n_p: int = 2, n_f: int = 3) -> ProbeSet:
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


def test_exact_probe_process_hides_static_ctx_parameter() -> None:
    """ExactProbeProcess builds static context internally instead of exposing it."""
    sig = inspect.signature(ExactProbeProcess.__init__)
    assert "static_ctx" not in sig.parameters
    assert "initial_psi" in sig.parameters


def test_exact_probe_process_builds_static_ctx_internally(monkeypatch: pytest.MonkeyPatch) -> None:
    """ExactProbeProcess wires make_mcwf_static_context and _simulate_sequences internally."""
    calls: dict[str, Any] = {}

    def _fake_make_ctx(operator: object, sim_params: object, noise_model: object | None = None) -> str:
        calls["ctx_args"] = (operator, sim_params, noise_model)
        return "CTX"

    def _fake_simulate_sequences(**kwargs) -> np.ndarray:  # noqa: ANN003
        calls["simulate_kwargs"] = kwargs
        n_tot = len(kwargs["psi_pairs_list"])
        return np.zeros((n_tot, 8), dtype=np.float32)

    monkeypatch.setattr(exact_mod, "make_mcwf_static_context", _fake_make_ctx)
    monkeypatch.setattr(exact_mod, "_simulate_sequences", _fake_simulate_sequences)

    op = MPO.ising(length=1, J=0.0, g=0.0)
    sim = AnalogSimParams(dt=0.1)
    psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    process = ExactProbeProcess(operator=op, sim_params=sim, initial_psi=psi0, parallel=False)
    probe_set = _make_minimal_probe_set(cut=1, k=1, n_p=2, n_f=3)
    out = process.evaluate_probe_set(probe_set)

    assert out.shape == (2, 3, 3)
    assert calls["ctx_args"] == (op, sim, None)
    assert calls["simulate_kwargs"]["static_ctx"] == "CTX"


def test_exact_diagnostics_use_cut_branch_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """evaluate_exact_probe_set_with_diagnostics weights prod(step_probs[:cut])."""

    def _fake_simulate(**kwargs) -> tuple[np.ndarray, list[dict[str, object]]]:  # noqa: ANN003
        n_tot = len(kwargs["psi_pairs_list"])
        traces = [{"step_probs": [0.5, 0.8, 1.0], "cumulative_weight_final": 0.4} for _ in range(n_tot)]
        return np.zeros((n_tot, 8), dtype=np.float32), traces

    monkeypatch.setattr(exact_mod, "simulate_final_states_with_diagnostics", _fake_simulate)

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
    _, weights, _ = evaluate_exact_probe_set_with_diagnostics(
        probe_set=probe_set,
        operator=op,
        sim_params=sim,
        initial_psi=psi0,
        parallel=False,
    )
    assert weights.shape == (1, 1)
    assert float(weights[0, 0]) == 0.4


def test_exact_probe_process_parallel_smoke() -> None:
    """ExactProbeProcess completes a tiny parallel rollout batch."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    sim = AnalogSimParams(dt=0.1)
    psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    process = ExactProbeProcess(operator=op, sim_params=sim, initial_psi=psi0, parallel=True, show_progress=False)
    probe_set = _make_minimal_probe_set(cut=1, k=1, n_p=2, n_f=2)
    out = process.evaluate_probe_set(probe_set)
    assert out.shape == (2, 2, 3)
