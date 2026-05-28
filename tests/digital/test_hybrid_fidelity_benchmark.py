# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Benchmark hybrid circuit simulation against Qiskit statevectors.

Circuit simulation pads product-state initial MPS bond dimensions to at least two
before applying gates. These probes check that hybrid mode (TEBD for NN gates,
TDVP for long-range gates) reaches unit fidelity vs Qiskit on small circuits
when that padding is in effect.

The probe set focuses on product states, directional CNOT layouts (including interior
long-range controls), and a mixed NN/long-range circuit. Deep stacks of arbitrary
rotations plus long-range entangling gates are omitted here because hybrid mode can
show larger accumulated numerical drift (~1e-3) unrelated to initial bond padding.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector

from mqt.yaqs import Simulator
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.libraries.gate_library import Z

_FIDELITY_TOL = 1e-8


def _fidelity(reference: np.ndarray, state: np.ndarray) -> float:
    """Squared overlap fidelity |<reference|state>|^2.

    Args:
        reference: Reference state vector.
        state: State vector to compare.

    Returns:
        Fidelity in ``[0, 1]``.
    """
    return float(abs(np.vdot(reference, state)) ** 2)


def _hybrid_statevector(qc: QuantumCircuit, *, tdvp_sweeps: int = 4) -> np.ndarray:
    """Noiseless hybrid simulation state vector (default initial-state padding).

    Args:
        qc: Circuit to simulate.
        tdvp_sweeps: TDVP sub-sweeps per long-range gate.

    Returns:
        Final state vector in little-endian order (site ``i`` = Qiskit qubit ``i``).
    """
    sim_params = StrongSimParams(
        observables=[Observable(Z(), 0)],
        gate_mode="hybrid",
        preset="exact",
        svd_threshold=1e-14,
        tdvp_sweeps=tdvp_sweeps,
        get_state=True,
    )
    state = State(qc.num_qubits, initial="zeros")
    result = Simulator(parallel=False, show_progress=False).run(state, qc, sim_params, None)
    assert result.output_state is not None
    return result.output_state.mps.to_vec()


def _build_hybrid_fidelity_probes() -> list[tuple[str, QuantumCircuit]]:
    """Named circuits exercising NN, long-range, and mixed gate layouts.

    Returns:
        ``(probe_id, circuit)`` pairs for parametrized benchmarks.
    """
    probes: list[tuple[str, QuantumCircuit]] = []

    for n in (3, 4):
        for site in range(n):
            qc = QuantumCircuit(n)
            qc.x(site)
            probes.append((f"product_x_{n}q_site{site}", qc))

    qc = QuantumCircuit(3)
    qc.rx(0.37, 0)
    qc.ry(0.51, 1)
    qc.rz(0.23, 2)
    qc.x(0)
    qc.h(2)
    probes.append(("single_qubit_rotations_3q", qc))

    for n, directed_cnots in [
        (3, [(0, 0, 1), (1, 1, 2), (0, 0, 2), (2, 2, 0)]),
        (4, [(1, 1, 2), (1, 1, 3), (0, 0, 3), (3, 3, 1), (1, 0, 1)]),
    ]:
        for h_site, ctrl, tgt in directed_cnots:
            qc = QuantumCircuit(n)
            qc.h(h_site)
            qc.cx(ctrl, tgt)
            probes.append((f"h_cx_{n}q_h{h_site}_c{ctrl}_t{tgt}", qc))

    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 3)
    qc.rzz(0.2, 2, 3)
    probes.append(("mixed_nn_lr_rzz_4q", qc))

    return probes


_HYBRID_FIDELITY_PROBES = _build_hybrid_fidelity_probes()
_PROBE_IDS = [probe_id for probe_id, _ in _HYBRID_FIDELITY_PROBES]


@pytest.mark.parametrize(
    ("probe_id", "qc"),
    _HYBRID_FIDELITY_PROBES,
    ids=_PROBE_IDS,
)
def test_hybrid_statevector_fidelity_per_probe(probe_id: str, qc: QuantumCircuit) -> None:
    """Hybrid simulation matches Qiskit for one benchmark probe (chi>=2 initial padding)."""
    del probe_id  # used only for pytest ids
    reference = np.asarray(Statevector(qc).data, dtype=np.complex128)
    simulated = _hybrid_statevector(qc)
    assert _fidelity(reference, simulated) == pytest.approx(1.0, abs=_FIDELITY_TOL)


def test_hybrid_fidelity_benchmark_all_probes() -> None:
    """Aggregate benchmark: every probe reaches unit fidelity under hybrid simulation."""
    worst_probe = ""
    worst_fidelity = 1.0
    failures: list[tuple[str, float]] = []

    for probe_id, qc in _HYBRID_FIDELITY_PROBES:
        reference = np.asarray(Statevector(qc).data, dtype=np.complex128)
        simulated = _hybrid_statevector(qc)
        fidelity = _fidelity(reference, simulated)
        if fidelity < worst_fidelity:
            worst_fidelity = fidelity
            worst_probe = probe_id
        if fidelity < 1.0 - _FIDELITY_TOL:
            failures.append((probe_id, fidelity))

    assert not failures, (
        f"hybrid fidelity below {1.0 - _FIDELITY_TOL} on {len(failures)} probe(s): {failures}; "
        f"worst overall {worst_probe}={worst_fidelity:.12g}"
    )
