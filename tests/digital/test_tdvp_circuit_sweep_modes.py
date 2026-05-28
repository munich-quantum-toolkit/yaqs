# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compare partial-LR vs symmetric LR+RL circuit TDVP via Pauli expectations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector

from mqt.yaqs import Simulator
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.libraries.gate_library import X, Y, Z

if TYPE_CHECKING:
    from mqt.yaqs.core.data_structures.simulation_parameters import GateMode


def _pauli_label(num_qubits: int, site: int, letter: str) -> str:
    """Build a Qiskit Pauli label for a single-site operator (site ``i`` = Qiskit qubit ``i``)."""
    chars = ["I"] * num_qubits
    chars[num_qubits - 1 - site] = letter
    return "".join(chars)


def _qiskit_expectation(qc: QuantumCircuit, obs: Observable) -> float:
    """Reference expectation value from Qiskit ``Statevector``.

    Args:
        qc: Circuit defining the reference state.
        obs: Single-qubit Pauli observable.

    Returns:
        Real expectation value.
    """
    site = obs.sites[0] if isinstance(obs.sites, list) else obs.sites
    letter = obs.gate.name.upper()
    pauli = Pauli(_pauli_label(qc.num_qubits, site, letter))
    return float(np.real(Statevector(qc).expectation_value(pauli)))


def _xyz_observable_panel(num_qubits: int) -> list[Observable]:
    """Pauli X, Y, and Z on every qubit."""
    observables: list[Observable] = []
    for site in range(num_qubits):
        observables.extend([Observable(X(), site), Observable(Y(), site), Observable(Z(), site)])
    return observables


@dataclass(frozen=True)
class ObservableError:
    """Absolute error on one Pauli expectation value."""

    name: str
    site: int | list[int]
    reference: float
    simulated: float

    @property
    def abs_error(self) -> float:
        """Absolute deviation from the reference expectation."""
        return abs(self.simulated - self.reference)


def _run_observable_errors(
    qc: QuantumCircuit,
    *,
    gate_mode: GateMode = "hybrid",
    tdvp_circuit_full_sweep: bool = False,
    tdvp_sweeps: int = 4,
) -> list[ObservableError]:
    """Simulate ``qc`` and compare Pauli expectations against Qiskit.

    Args:
        qc: Circuit to simulate.
        gate_mode: Two-qubit update mode.
        tdvp_circuit_full_sweep: Symmetric LR+RL vs partial LR circuit TDVP sweeps.
        tdvp_sweeps: TDVP sub-sweeps per long-range gate.

    Returns:
        Per-observable errors in ``result.observables`` order.
    """
    observables = _xyz_observable_panel(qc.num_qubits)
    sim_params = StrongSimParams(
        observables=observables,
        gate_mode=gate_mode,
        preset="exact",
        svd_threshold=1e-14,
        tdvp_sweeps=tdvp_sweeps,
        tdvp_circuit_full_sweep=tdvp_circuit_full_sweep,
    )
    state = State(qc.num_qubits, initial="zeros")
    result = Simulator(parallel=False, show_progress=False).run(state, qc, sim_params, None)

    errors: list[ObservableError] = []
    for index, obs in enumerate(result.observables):
        simulated = float(np.real(result.expectation_values[index][-1]))
        reference = _qiskit_expectation(qc, obs)
        site = obs.sites[0] if isinstance(obs.sites, list) else obs.sites
        errors.append(
            ObservableError(
                name=obs.gate.name,
                site=site,
                reference=reference,
                simulated=simulated,
            )
        )
    return errors


def _max_observable_error(errors: list[ObservableError]) -> float:
    """Largest absolute expectation error in a panel."""
    return max(entry.abs_error for entry in errors)


def _worst_observable(errors: list[ObservableError]) -> ObservableError:
    """Observable with the largest absolute error."""
    return max(errors, key=lambda entry: entry.abs_error)


def test_partial_lr_observable_panel_on_interior_long_range_cnot() -> None:
    """Default partial LR matches Qiskit on X/Y/Z expectations for one interior LR CNOT."""
    qc = QuantumCircuit(4)
    qc.h(1)
    qc.cx(1, 3)
    errors = _run_observable_errors(qc, tdvp_circuit_full_sweep=False)
    assert _max_observable_error(errors) == pytest.approx(0.0, abs=1e-8)


def test_mid_control_partial_lr_y_error_vs_tebd() -> None:
    """Partial LR misses Y expectations on control/target; TEBD is exact."""
    qc = QuantumCircuit(8)
    qc.h(4)
    qc.cx(4, 0)

    partial = _run_observable_errors(qc, gate_mode="hybrid", tdvp_circuit_full_sweep=False)
    tebd = _run_observable_errors(qc, gate_mode="tebd")

    worst_partial = _worst_observable(partial)
    assert worst_partial.name == "y"
    assert worst_partial.site in {0, 4}
    assert worst_partial.abs_error == pytest.approx(0.3827, abs=0.01)
    assert _max_observable_error(tebd) == pytest.approx(0.0, abs=1e-8)


def test_mid_control_symmetric_lr_rl_y_errors_are_large() -> None:
    """Symmetric LR+RL drives Y expectations to ±1 instead of 0 on ``H(mid); CX(mid, 0)``."""
    qc = QuantumCircuit(8)
    qc.h(4)
    qc.cx(4, 0)

    partial = _run_observable_errors(qc, tdvp_circuit_full_sweep=False)
    symmetric = _run_observable_errors(qc, tdvp_circuit_full_sweep=True)

    assert _max_observable_error(partial) == pytest.approx(0.3827, abs=0.01)
    assert _max_observable_error(symmetric) == pytest.approx(1.0, abs=0.02)
    worst_sym = _worst_observable(symmetric)
    assert worst_sym.name == "y"
    assert worst_sym.abs_error == pytest.approx(1.0, abs=0.02)


def test_double_long_range_observable_errors_vs_tebd() -> None:
    """Two interior LR CNOTs: hybrid Y/X errors are O(1); TEBD matches Qiskit."""
    qc = QuantumCircuit(6)
    qc.h(1)
    qc.cx(1, 5)
    qc.h(2)
    qc.cx(2, 4)

    partial = _run_observable_errors(qc, gate_mode="hybrid", tdvp_circuit_full_sweep=False)
    symmetric = _run_observable_errors(qc, gate_mode="hybrid", tdvp_circuit_full_sweep=True)
    tebd = _run_observable_errors(qc, gate_mode="tebd")

    assert _max_observable_error(partial) == pytest.approx(1.0, abs=0.05)
    assert _max_observable_error(symmetric) == pytest.approx(1.0, abs=0.05)
    assert _max_observable_error(tebd) == pytest.approx(0.0, abs=1e-8)

    worst_partial = _worst_observable(partial)
    assert worst_partial.name in {"y", "x"}
    assert worst_partial.abs_error >= 0.5


def test_observable_error_summary_table() -> None:
    """Aggregate worst-case observable errors across probe circuits (diagnostic bounds)."""
    probes: list[tuple[str, QuantumCircuit]] = [
        ("interior_lr_4q", QuantumCircuit(4)),
        ("mid_ctrl_8q", QuantumCircuit(8)),
        ("double_lr_6q", QuantumCircuit(6)),
    ]
    probes[0][1].h(1)
    probes[0][1].cx(1, 3)
    probes[1][1].h(4)
    probes[1][1].cx(4, 0)
    probes[2][1].h(1)
    probes[2][1].cx(1, 5)
    probes[2][1].h(2)
    probes[2][1].cx(2, 4)

    bounds = {
        ("interior_lr_4q", "partial"): 1e-8,
        ("interior_lr_4q", "symmetric"): 1e-8,
        ("mid_ctrl_8q", "partial"): 0.75,
        ("mid_ctrl_8q", "symmetric"): 1.0,
        ("double_lr_6q", "partial"): 1.0,
        ("double_lr_6q", "symmetric"): 1.0,
    }

    for probe_id, qc in probes:
        for mode, full_sweep in [("partial", False), ("symmetric", True)]:
            errors = _run_observable_errors(qc, tdvp_circuit_full_sweep=full_sweep)
            max_err = _max_observable_error(errors)
            assert max_err <= bounds[(probe_id, mode)] + 1e-6, (
                f"{probe_id}/{mode}: max observable error {max_err:.4f} exceeds bound; "
                f"worst={_worst_observable(errors)}"
            )
