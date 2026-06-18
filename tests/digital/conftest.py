# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared helpers for digital TJM tests."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from mqt.yaqs import Observable, Simulator, State, StrongSimParams
from mqt.yaqs.core.libraries.gate_library import Z

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit

    from mqt.yaqs.core.data_structures.mps import MPS
    from mqt.yaqs.core.data_structures.simulation_parameters import GateMode


def _expect_mps(mps: MPS, obs: Observable) -> float:
    """Expectation value computed directly from an MPS.

    Args:
        mps: State in MPS form.
        obs: Observable to evaluate.

    Returns:
        Real expectation value ``<obs>``.
    """
    return float(mps.expect(obs))


def _expect_mps_like_evaluate_observables(mps: MPS, observables: list[Observable]) -> list[float]:
    """Compute expectations using the same center-shifting scheme as ``evaluate_observables``.

    Args:
        mps: State in MPS form.
        observables: Observables to evaluate.

    Returns:
        Expectation values in the same order as ``observables``.
    """
    temp = copy.deepcopy(mps)
    last_site = 0
    values: list[float] = []
    for obs in observables:
        idx = obs.sites[0] if isinstance(obs.sites, list) else obs.sites
        if idx > last_site:
            for site in range(last_site, idx):
                temp.shift_orthogonality_center_right(site)
            last_site = idx
        values.append(float(temp.expect(obs)))
    return values


def _fidelity(a: np.ndarray, b: np.ndarray) -> float:
    r"""Squared overlap fidelity \\(|\\langle a|b\\rangle|^2\\).

    Args:
        a: First state vector.
        b: Second state vector.

    Returns:
        Fidelity as a float in ``[0, 1]`` (up to numerical error).
    """
    return float(abs(np.vdot(a, b)) ** 2)


def _phase_align(reference: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Align ``state``'s global phase to ``reference``.

    Args:
        reference: Reference vector to align against.
        state: State vector to be phase-aligned.

    Returns:
        ``state`` with global phase aligned to ``reference``.
    """
    phase = np.vdot(state, reference)
    if abs(phase) > 0.0:
        return state * (phase / abs(phase))
    return state


def _run_strong_noiseless(
    qc: QuantumCircuit,
    *,
    gate_mode: GateMode = "tdvp",
    max_bond_dim: int | None = None,
    svd_threshold: float = 1e-12,
    get_state: bool = False,
    tdvp_sweeps: int = 1,
) -> float | np.ndarray:
    """Run a noiseless strong simulation.

    Returns:
        Final state vector if ``get_state=True``, otherwise ``<Z_0>``.
    """
    num_qubits = qc.num_qubits
    sim_params = StrongSimParams(
        observables=[Observable(Z(), 0)],
        gate_mode=gate_mode,
        preset="exact",
        svd_threshold=svd_threshold,
        max_bond_dim=max_bond_dim,
        get_state=get_state,
        tdvp_sweeps=tdvp_sweeps,
    )
    state = State(num_qubits, initial="zeros")
    result = Simulator(parallel=False, show_progress=False).run(state, qc, sim_params, None)
    if get_state:
        assert result.output_state is not None
        return result.output_state.mps.to_vec()
    exp = result.expectation_values[0]
    assert exp is not None
    return float(np.real(exp[0]))


def _physical_second_schmidt(vec: np.ndarray, length: int, cut: int) -> float:
    """Second normalized Schmidt value of a statevector at a physical cut.

    Returns:
        Normalized second Schmidt coefficient at ``cut``.
    """
    left_dim = 2 ** (cut + 1)
    right_dim = 2 ** (length - cut - 1)
    mat = vec.reshape(left_dim, right_dim)
    _u, s_vec, _v = np.linalg.svd(mat, full_matrices=False)
    s_arr = np.asarray(s_vec, dtype=np.float64)
    norm = float(np.linalg.norm(s_arr))
    if norm > 0.0:
        s_arr /= norm
    return float(s_arr[1]) if len(s_arr) > 1 else 0.0
