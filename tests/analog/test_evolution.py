# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for shared analog unitary-evolution dispatch."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

from mqt.yaqs import AnalogSimParams, Observable
from mqt.yaqs.analog import evolution as evolution_module
from mqt.yaqs.analog.analog_tjm import analog_tjm_1
from mqt.yaqs.analog.ensemble import ensemble_member_worker
from mqt.yaqs.analog.evolution import apply_unitary_evolution
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import EvolutionMode
from mqt.yaqs.core.libraries.gate_library import X, Z

GaugeKind = Literal["center_zero", "center_right", "unknown"]


def _gauge_variant(kind: GaugeKind) -> MPS:
    """Return one deterministic state in the requested gauge.

    Args:
        kind: Center at site zero, center at the right endpoint, or a genuinely
            non-canonical gauge with unknown center metadata.

    Returns:
        A gauge-equivalent normalized three-site MPS.
    """
    rng = np.random.default_rng(20260901)
    bonds = [1, 2, 2, 1]
    tensors = [
        (
            rng.normal(size=(2, bonds[site], bonds[site + 1])) + 1j * rng.normal(size=(2, bonds[site], bonds[site + 1]))
        ).astype(np.complex128)
        for site in range(3)
    ]
    state = MPS(3, tensors=tensors)
    state.normalize("B", decomposition="SVD")

    if kind == "center_right":
        state.shift_center_to(state.length - 1)
    elif kind == "unknown":
        before = state.to_vec()
        gauge = np.array([[0.4, 0.3], [0.0, 1.7]], dtype=np.complex128)
        state.tensors[1] = np.einsum("plr,rs->pls", state.tensors[1], gauge)
        state.tensors[2] = np.einsum("ab,pbr->par", np.linalg.inv(gauge), state.tensors[2])
        state.set_center(None)
        np.testing.assert_allclose(state.to_vec(), before, atol=1e-12)
        assert not state.check_canonical_form()

    return state


def _bug_params(*, multi_time: bool = False) -> AnalogSimParams:
    """Return small deterministic BUG parameters.

    Args:
        multi_time: Whether to request one ensemble correlator.

    Returns:
        Analog simulation parameters for one BUG step.
    """
    z0 = Observable(Z(), 0)
    return AnalogSimParams(
        observables=[z0],
        elapsed_time=0.05,
        dt=0.05,
        order=1,
        evolution_mode=EvolutionMode.BUG,
        sample_timesteps=True,
        get_state=True,
        max_bond_dim=8,
        svd_threshold=1e-12,
        multi_time_observables=[(z0, Observable(X(), 1))] if multi_time else None,
    )


def _fidelity(left: MPS, right: MPS) -> float:
    """Return normalized dense-state fidelity.

    Args:
        left: First MPS.
        right: Second MPS.

    Returns:
        Squared normalized overlap.
    """
    left_vec = left.to_vec()
    right_vec = right.to_vec()
    overlap = abs(np.vdot(left_vec, right_vec)) ** 2
    norm = float(np.vdot(left_vec, left_vec).real * np.vdot(right_vec, right_vec).real)
    return float(overlap / norm)


@pytest.mark.parametrize("gauge_kind", ["center_zero", "center_right", "unknown"])
def test_apply_unitary_evolution_prepares_bug_center(
    gauge_kind: GaugeKind,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared wrapper gives the strict BUG kernel a genuine center at site zero.

    Args:
        gauge_kind: Gauge assigned to the input state.
        monkeypatch: Pytest fixture used to inspect the BUG call boundary.
    """
    state = _gauge_variant(gauge_kind)
    before = state.to_vec()
    called = False

    def strict_bug(
        current: MPS,
        _hamiltonian: MPO,
        _sim_params: AnalogSimParams,
        *,
        normalize: bool = True,
    ) -> None:
        """Check the wrapper contract without evolving the state."""
        nonlocal called
        del normalize
        called = True
        assert current.orthogonality_center == 0
        assert 0 in current.check_canonical_form()
        np.testing.assert_allclose(current.to_vec(), before, atol=1e-12)

    monkeypatch.setattr(evolution_module, "bug", strict_bug)

    apply_unitary_evolution(state, MPO.ising(3, 1.0, 0.7), _bug_params())

    assert called


@pytest.mark.parametrize("gauge_kind", ["center_zero", "center_right", "unknown"])
def test_order_one_bug_is_gauge_independent(gauge_kind: GaugeKind) -> None:
    """Order-one BUG evolution gives the same state from equivalent input gauges.

    Args:
        gauge_kind: Gauge assigned to the candidate input.
    """
    hamiltonian = MPO.ising(3, 1.0, 0.7)
    params = _bug_params()
    _, _, reference = analog_tjm_1((0, _gauge_variant("center_zero"), None, params, hamiltonian))
    _, _, candidate = analog_tjm_1((0, _gauge_variant(gauge_kind), None, params, hamiltonian))

    assert reference is not None
    assert candidate is not None
    assert _fidelity(reference, candidate) == pytest.approx(1.0, abs=1e-11)
    assert candidate.orthogonality_center == 0
    assert 0 in candidate.check_canonical_form()


@pytest.mark.parametrize("gauge_kind", ["center_zero", "center_right", "unknown"])
def test_bug_ensemble_member_is_gauge_independent(gauge_kind: GaugeKind) -> None:
    """BUG ensemble observables and correlators do not depend on the input gauge.

    Args:
        gauge_kind: Gauge assigned to the candidate input.
    """
    hamiltonian = MPO.ising(3, 1.0, 0.7)
    params = _bug_params(multi_time=True)
    reference, _, reference_multi = ensemble_member_worker((0, _gauge_variant("center_zero"), params, hamiltonian))
    candidate, _, candidate_multi = ensemble_member_worker((0, _gauge_variant(gauge_kind), params, hamiltonian))

    np.testing.assert_allclose(candidate, reference, atol=1e-11)
    assert reference_multi is not None
    assert candidate_multi is not None
    np.testing.assert_allclose(candidate_multi, reference_multi, atol=1e-11)
