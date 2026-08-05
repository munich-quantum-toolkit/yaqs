# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the local noise model applied to multi-qubit gates in the digital TJM.

After every gate acting on two or more qubits, the digital TJM applies one noise
layer of unit duration (``apply_dissipation`` followed by ``stochastic_process``,
both at ``dt=1``), restricted to the noise processes whose sites all belong to the
gate's qubits (``create_local_noise_model``).  These tests pin that behavior:

* which processes are retained for contiguous, long-range, and permuted-qarg
  gates, and that two-qubit gates behave exactly as before;
* the operator content of the dissipation and jump steps against dense
  references, at machine precision;
* convergence of the sampled trajectory average to the exact expected channel
  of the noise layer, within the statistical floor;
* that one noise layer after the gate approximates the corresponding
  continuous noisy evolution, and that covering *all* gate qubits (rather than
  only the outermost two) is what makes it accurate.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Choi, Operator, SuperOp
from scipy.linalg import expm

from mqt.yaqs import DigitalSimParams, NoiseModel, Observable
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.libraries.gate_library import CCX, Z
from mqt.yaqs.core.methods.dissipation import apply_dissipation
from mqt.yaqs.core.methods.stochastic_process import create_probability_distribution
from mqt.yaqs.digital.digital_tjm import create_local_noise_model, digital_tjm

if TYPE_CHECKING:
    from collections.abc import Callable

I2 = np.eye(2, dtype=np.complex128)
PAULI_Z_MAT = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _embed_le(op: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """Embed a one-qubit operator at ``qubit`` on ``n`` qubits (little-endian kron).

    Returns:
        The embedded operator as a dense matrix.
    """
    ops = [I2] * n
    ops[qubit] = op
    full = ops[n - 1]
    for q in range(n - 2, -1, -1):
        full = np.kron(full, ops[q])
    return full


def _ry(theta: float) -> np.ndarray:
    """Real Y-rotation matrix.

    Returns:
        The 2x2 rotation matrix.
    """
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _dense_to_mps(vec: np.ndarray) -> MPS:
    """Exact (untruncated) MPS from a dense state vector in ``MPS.to_vec`` order.

    Returns:
        The MPS whose ``to_vec`` reproduces ``vec``.
    """
    n = int(np.log2(vec.size))
    psi = vec.reshape([2] * n)
    # MPS.to_vec is little-endian (site 0 = least significant bit): axis i of the
    # big-endian reshape corresponds to site n-1-i, so reverse to site order.
    psi = np.transpose(psi, axes=list(range(n))[::-1])
    tensors = []
    chi_l = 1
    m = psi.reshape(chi_l * 2, -1)
    for site in range(n):
        if site < n - 1:
            u, s, vh = np.linalg.svd(m, full_matrices=False)
            chi_r = len(s)
            tensors.append(u.reshape(chi_l, 2, chi_r).transpose(1, 0, 2))
            m = (np.diag(s) @ vh).reshape(chi_r * 2, -1)
            chi_l = chi_r
        else:
            tensors.append(m.reshape(chi_l, 2, 1).transpose(1, 0, 2))
    return MPS(n, tensors=tensors)


def _sample_processes() -> list[dict[str, object]]:
    """Global noise model spanning five sites with one- and two-site processes.

    Returns:
        The list of process dictionaries.
    """
    return [
        {"name": "pauli_x", "sites": [0], "strength": 0.01},
        {"name": "pauli_x", "sites": [1], "strength": 0.02},
        {"name": "pauli_x", "sites": [2], "strength": 0.03},
        {"name": "pauli_x", "sites": [3], "strength": 0.04},
        {"name": "pauli_x", "sites": [4], "strength": 0.05},
        {"name": "crosstalk_xx", "sites": [0, 1], "strength": 0.06},
        {"name": "crosstalk_xx", "sites": [1, 2], "strength": 0.07},
        {"name": "crosstalk_xx", "sites": [2, 3], "strength": 0.08},
        {"name": "crosstalk_xx", "sites": [1, 3], "strength": 0.09},
        {"name": "crosstalk_xx", "sites": [0, 4], "strength": 0.10},
    ]


def _process_keys(model: NoiseModel) -> set[tuple[str, tuple[int, ...], float]]:
    """Hashable (name, sites, strength) triples of a noise model.

    Returns:
        The set of process triples.
    """
    return {(str(p["name"]), tuple(p["sites"]), float(p["strength"])) for p in model.processes}


# ---------------------------------------------------------------------------
# which processes a gate's noise layer retains
# ---------------------------------------------------------------------------


def test_local_noise_model_contiguous_three_qubit() -> None:
    """A contiguous 3q gate keeps all 1q and in-support 2q processes, nothing else."""
    local = create_local_noise_model(NoiseModel(_sample_processes()), [0, 1, 2])
    expected = {
        ("pauli_x", (0,), 0.01),
        ("pauli_x", (1,), 0.02),
        ("pauli_x", (2,), 0.03),
        ("crosstalk_xx", (0, 1), 0.06),
        ("crosstalk_xx", (1, 2), 0.07),
    }
    assert _process_keys(local) == expected


def test_local_noise_model_long_range_interior_dropped() -> None:
    """ccx(0, 2, 4) support: idle sites 1 and 3 between the gate qubits get no noise."""
    local = create_local_noise_model(NoiseModel(_sample_processes()), [0, 2, 4])
    expected = {
        ("pauli_x", (0,), 0.01),
        ("pauli_x", (2,), 0.03),
        ("pauli_x", (4,), 0.05),
        ("crosstalk_xx", (0, 4), 0.10),
    }
    assert _process_keys(local) == expected


def test_local_noise_model_qarg_order_irrelevant() -> None:
    """The retained set depends on the site set, not on the qarg order."""
    model = NoiseModel(_sample_processes())
    assert _process_keys(create_local_noise_model(model, [2, 0, 1])) == _process_keys(
        create_local_noise_model(model, [0, 1, 2])
    )


def test_local_noise_model_two_qubit_unchanged() -> None:
    """For 2q gates the rule retains exactly what the earlier endpoint filter did.

    ``NoiseModel.__init__`` sorts two-site process sites, so the earlier
    ordered-equality filter and the current subset filter cannot disagree;
    this includes a process the user specified with descending sites.
    """
    processes = [*_sample_processes(), {"name": "crosstalk_xx", "sites": [3, 1], "strength": 0.11}]
    model = NoiseModel(processes)

    def endpoint_filter(first_site: int, last_site: int) -> set[tuple[str, tuple[int, ...], float]]:
        affected = [first_site, last_site]
        kept = [
            p
            for p in model.processes
            if p["sites"] == affected or p["sites"] == [first_site] or p["sites"] == [last_site]
        ]
        return {(str(p["name"]), tuple(p["sites"]), float(p["strength"])) for p in kept}

    for first, last in [(0, 1), (1, 2), (2, 3), (1, 3), (0, 4)]:
        assert endpoint_filter(first, last) == _process_keys(create_local_noise_model(model, [first, last]))


# ---------------------------------------------------------------------------
# operator content of the noise layer
# ---------------------------------------------------------------------------

_LAYER_PROCS: list[tuple[str, np.ndarray, int, float]] = [
    ("lowering", np.asarray(NoiseModel.get_operator("lowering"), dtype=np.complex128), 0, 0.013),
    ("pauli_z", PAULI_Z_MAT, 1, 0.007),
    ("pauli_x", np.array([[0, 1], [1, 0]], dtype=np.complex128), 2, 0.005),
]


def _layer_state() -> np.ndarray:
    """Rotated product state with strong control weight (the gate must act).

    Returns:
        The dense state vector.
    """
    v = np.eye(8, dtype=np.complex128)[:, 0]
    for q, theta in enumerate([1.9, 2.1, 0.9]):
        v = _embed_le(_ry(theta), q, 3) @ v
    return v


def test_dissipation_matches_dense_exponential() -> None:
    """``apply_dissipation`` equals K = expm(-dt/2 * sum_j gamma_j L_j^dag L_j)."""
    nm = NoiseModel([{"name": name, "sites": [s], "strength": g} for name, _, s, g in _LAYER_PROCS])
    sim_params = DigitalSimParams(observables=[Observable(Z(), 0)], preset="exact", num_traj=1, random_seed=0)
    v = _layer_state()
    h_eff = np.zeros((8, 8), dtype=np.complex128)
    for _, mat, s, g in _LAYER_PROCS:
        lf = _embed_le(mat, s, 3)
        h_eff += g * (lf.conj().T @ lf)
    expected = expm(-0.5 * h_eff) @ v

    mps = _dense_to_mps(v)
    apply_dissipation(mps, nm, dt=1, sim_params=sim_params)
    np.testing.assert_allclose(mps.to_vec(), expected, atol=1e-12)


def test_jump_probabilities_match_dense() -> None:
    """The categorical jump weights are gamma_m ||L_m K psi||^2, normalized."""
    nm = NoiseModel([{"name": name, "sites": [s], "strength": g} for name, _, s, g in _LAYER_PROCS])
    sim_params = DigitalSimParams(observables=[Observable(Z(), 0)], preset="exact", num_traj=1, random_seed=0)
    v = _layer_state()
    mps = _dense_to_mps(v)
    apply_dissipation(mps, nm, dt=1, sim_params=sim_params)
    phi = mps.to_vec()

    dense = np.array([g * np.linalg.norm(_embed_le(mat, s, 3) @ phi) ** 2 for _, mat, s, g in _LAYER_PROCS])
    dense /= dense.sum()
    ordered, probs = create_probability_distribution(copy.deepcopy(mps), nm, 1.0, sim_params)
    order = [(str(p["name"]), p["sites"][0]) for p in ordered]
    perm = [order.index((name, s)) for name, _, s, _ in _LAYER_PROCS]
    np.testing.assert_allclose(np.asarray(probs)[perm], dense, atol=1e-12)


# ---------------------------------------------------------------------------
# sampled trajectories converge to the exact expected channel
# ---------------------------------------------------------------------------


def test_noisy_ccx_trajectory_convergence() -> None:
    """Trajectory-averaged <Z_q> converges to the exact expected channel.

    The reference is the exact ensemble average of the noise layer (dissipation
    K, then one jump draw), composed by hand behind the CCX; the tolerance is a
    statistical floor ~5/sqrt(N_traj).  Guards: the CCX must change the
    pre-gate state and the noise effect must exceed the tolerance, so the test
    cannot pass with the gate or the noise silently inactive.
    """
    n, num_traj, gamma = 3, 1024, 0.4
    angles = [1.9, 2.1, 0.9]
    qc = QuantumCircuit(n)
    for q, theta in enumerate(angles):
        qc.ry(theta, q)
    qc.ccx(0, 1, 2)

    nm = NoiseModel([{"name": "lowering", "sites": [q], "strength": gamma} for q in range(n)])
    sim_params = DigitalSimParams(
        observables=[Observable(Z(), q) for q in range(n)],
        preset="exact",
        num_traj=num_traj,
        random_seed=7,
    )

    # dense reference
    v = np.eye(2**n, dtype=np.complex128)[:, 0]
    for q, theta in enumerate(angles):
        v = _embed_le(_ry(theta), q, n) @ v
    v_pre = v
    v_post = np.asarray(Operator(qc).data) @ np.eye(2**n)[:, 0]
    overlap = abs(np.vdot(v_pre, v_post))
    assert overlap < 0.9, "CCX acts trivially on this input; test would be vacuous"

    lower = np.asarray(NoiseModel.get_operator("lowering"), dtype=np.complex128)
    h_eff = sum(gamma * (_embed_le(lower, q, n).conj().T @ _embed_le(lower, q, n)) for q in range(n))
    phi = expm(-0.5 * h_eff) @ v_post
    rho = np.outer(phi, phi.conj())
    weights, jumps = [], []
    for q in range(n):
        lphi = _embed_le(lower, q, n) @ phi
        weights.append(gamma * np.linalg.norm(lphi) ** 2)
        jumps.append(gamma * np.outer(lphi, lphi.conj()))
    rho += ((1.0 - np.linalg.norm(phi) ** 2) / np.sum(weights)) * np.sum(jumps, axis=0)
    z_ref = np.array([np.real(np.trace(_embed_le(PAULI_Z_MAT, q, n) @ rho)) for q in range(n)])
    z_noiseless = np.array([np.real(np.vdot(v_post, _embed_le(PAULI_Z_MAT, q, n) @ v_post)) for q in range(n)])
    tol = 5.0 / np.sqrt(num_traj)
    assert np.max(np.abs(z_ref - z_noiseless)) > 2 * tol, "noise effect below the statistical floor; vacuous"

    initial = MPS(n, state="zeros")
    acc = np.zeros(n)
    for traj in range(num_traj):
        results, _diag, _counts, _final = digital_tjm((traj, initial, nm, sim_params, qc))
        assert results is not None
        acc += results[:, -1]
    z_traj = acc / num_traj
    np.testing.assert_allclose(z_traj, z_ref, atol=tol)


# ---------------------------------------------------------------------------
# one noise layer after the gate approximates continuous noisy evolution
# ---------------------------------------------------------------------------


def test_noise_layer_matches_continuous_evolution() -> None:
    """One noise layer on all gate qubits tracks the continuous noisy evolution.

    Reference: the exact Lindblad evolution generated by the CCX generator plus
    dephasing on all three qubits over one time unit.  The simulator's rule
    (perfect CCX, then exp(D) on the gate qubits) stays within a small splitting
    error of it (measured 5.5e-4 at gamma=1e-3, asserted with 2x headroom).
    Restricting the noise to the outer two qubits — the behavior before the
    noise model covered all gate qubits — is strictly worse, which is what
    makes covering every gate qubit the right choice.
    """
    n, gamma = 3, 1e-3
    qc = QuantumCircuit(n)
    qc.ccx(0, 1, 2)
    u_super = SuperOp(Operator(qc))

    # dephasing channel on one qubit (column stacking): D = gamma (Z (x) Z - id)
    dmat_1q = gamma * (np.kron(PAULI_Z_MAT.conj(), PAULI_Z_MAT) - np.eye(4))
    ch_1q = SuperOp(expm(dmat_1q))

    def with_noise(qubits: list[int]) -> SuperOp:
        ch = u_super
        for q in qubits:
            ch = ch.compose(ch_1q, qargs=[q])
        return ch

    # continuous reference: generator from the gate library, mapped little-endian
    gate = CCX()
    gate.set_sites(0, 1, 2)
    f0, f1, f2 = (np.asarray(f, dtype=np.complex128) for f in gate.generator)
    gmat = np.kron(f2, np.kron(f1, f0))
    np.testing.assert_allclose(expm(-1j * gmat), np.asarray(Operator(qc).data), atol=1e-12)
    eye = np.eye(2**n)
    liou = -1j * (np.kron(eye, gmat) - np.kron(gmat.T, eye))
    for q in range(n):
        zf = _embed_le(PAULI_Z_MAT, q, n)
        liou += gamma * (np.kron(zf.conj(), zf) - np.eye(4**n))
    continuous = SuperOp(expm(liou), input_dims=(2,) * n, output_dims=(2,) * n)

    def distance(a: SuperOp, b: SuperOp) -> float:
        delta = np.asarray(Choi(a).data - Choi(b).data)
        return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh((delta + delta.conj().T) / 2)))) / 2**n

    all_gate_qubits = distance(with_noise([0, 1, 2]), continuous)
    outer_only = distance(with_noise([0, 2]), continuous)
    assert all_gate_qubits < 1.1e-3  # measured 5.5e-4, 2x headroom
    assert outer_only > all_gate_qubits


# ---------------------------------------------------------------------------
# end to end: every gate shape hands its qubits to the noise layer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("builder", "expected_sites"),
    [
        pytest.param(lambda qc: qc.ccx(0, 1, 2), [0, 1, 2], id="contiguous-3q"),
        pytest.param(lambda qc: qc.ccx(0, 2, 4), [0, 2, 4], id="long-range-3q"),
        pytest.param(lambda qc: qc.ccx(2, 0, 1), [2, 0, 1], id="permuted-qargs-3q"),
        pytest.param(lambda qc: qc.cx(0, 1), [0, 1], id="nearest-neighbor-2q"),
        pytest.param(lambda qc: qc.cx(0, 3), [0, 3], id="long-range-2q"),
    ],
)
def test_noise_sites_end_to_end(builder: Callable[[QuantumCircuit], object], expected_sites: list[int]) -> None:
    """Each gate shape hands exactly its qarg indices to the local noise model."""
    n = 5
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.ry(1.9 - 0.2 * q, q)
    builder(qc)
    nm = NoiseModel([{"name": "pauli_x", "sites": [q], "strength": 0.01} for q in range(n)])
    sim_params = DigitalSimParams(
        observables=[Observable(Z(), 0)],
        preset="exact",
        num_traj=1,
        random_seed=0,
    )
    with patch(
        "mqt.yaqs.digital.digital_tjm.create_local_noise_model",
        wraps=create_local_noise_model,
    ) as mock_local:
        results, _diag, _counts, _final = digital_tjm((0, MPS(n, state="zeros"), nm, sim_params, qc))
    mock_local.assert_called_once()
    _model, sites = mock_local.call_args.args
    assert list(sites) == expected_sites
    local = create_local_noise_model(nm, sites)
    assert {tuple(p["sites"]) for p in local.processes} == {(s,) for s in expected_sites}
    assert results is not None
    assert np.all(np.isfinite(results))


def test_noise_sites_generator_path() -> None:
    """The TDVP/generator path uses the same noise sites as the MPO path."""
    n = 3
    qc = QuantumCircuit(n)
    for q, theta in enumerate([1.9, 2.1, 0.9]):
        qc.ry(theta, q)
    qc.ccx(0, 1, 2)
    nm = NoiseModel([{"name": "pauli_x", "sites": [q], "strength": 0.01} for q in range(n)])
    sim_params = DigitalSimParams(
        observables=[Observable(Z(), 0)],
        gate_mode="tdvp",
        preset="exact",
        num_traj=1,
        random_seed=0,
    )
    with patch(
        "mqt.yaqs.digital.digital_tjm.create_local_noise_model",
        wraps=create_local_noise_model,
    ) as mock_local:
        results, _diag, _counts, _final = digital_tjm((0, MPS(n, state="zeros"), nm, sim_params, qc))
    mock_local.assert_called_once()
    _model, sites = mock_local.call_args.args
    assert list(sites) == [0, 1, 2]
    assert results is not None
    assert np.all(np.isfinite(results))
