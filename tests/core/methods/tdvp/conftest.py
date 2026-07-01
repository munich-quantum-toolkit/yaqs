# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

r"""Shared helpers for TDVP unit and regression tests.

PR TDVP regression smoke (fast subset)::

    uv run pytest tests/core/methods/tdvp/test_tdvp.py \\
                  tests/core/methods/tdvp/test_integrators.py \\
                  tests/digital/test_digital_tjm.py \\
                  -m "tdvp_regression and not slow"
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Literal
from unittest.mock import patch

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector

from mqt.yaqs import Simulator, State, StrongSimParams
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.libraries.gate_library import GateLibrary
from mqt.yaqs.digital.digital_tjm import apply_two_qubit_gate_tdvp

if TYPE_CHECKING:
    import pytest

NORM_TOL = 1e-6
EXACT_FID_TOL = 1e-12

GateName = Literal["rzz", "rxx", "ryy"]


def pytest_configure(config: pytest.Config) -> None:
    """Register TDVP regression markers."""
    config.addinivalue_line(
        "markers",
        "tdvp_regression: circuit TDVP infrastructure stability regressions",
    )
    config.addinivalue_line(
        "markers",
        "slow: tests that need many TDVP sweeps or large L",
    )


Z_TOL = 1e-8
# Endpoint |+⟩ RZZ global fidelity under production 2TDVP window path (local ⟨Z_i⟩ still exact).
PLUS_LR_RZZ_GLOBAL_FID = 0.9776682445628022
# Entangled Haar prep: 1-sweep LR RZZ may drift slightly on minimum-deps / ARM Krylov stacks.
HAAR_LR_PREP_SEED = 42
HAAR_LR_PREP_PAD = 4
HAAR_LR_Z_TOL = 1e-6
HAAR_LR_FID_ABS = 0.03  # vs same-prep Qiskit reference
HAAR_LR_FID_FLOOR = 0.97


def _z_expectation(vec: np.ndarray, site: int) -> float:
    """Single-site ``⟨Z⟩`` from a state vector (Qiskit little-endian convention).

    Args:
        vec: State vector whose length is a power of two (dimension ``2^n``).
        site: Qubit index for the ``Z`` expectation.

    Returns:
        Real expectation value ``⟨Z_site⟩``.

    Raises:
        ValueError: If ``vec.size`` is not a power of two.

    """
    if vec.size & (vec.size - 1) != 0:
        msg = f"State vector length must be a power of two, got {vec.size}."
        raise ValueError(msg)
    num_qubits = int(np.log2(vec.size))
    label = ["I"] * num_qubits
    label[num_qubits - 1 - site] = "Z"
    return float(np.real(Statevector(vec).expectation_value(Pauli("".join(label)))))


def _assert_z_observables_match(
    ref_vec: np.ndarray,
    vec: np.ndarray,
    length: int,
    *,
    tol: float = Z_TOL,
) -> None:
    """Assert all single-site ``⟨Z_i⟩`` match between two state vectors.

    Args:
        ref_vec: Reference state vector.
        vec: Candidate state vector.
        length: Number of qubits compared.
        tol: Maximum allowed absolute error per site.

    """
    for site in range(length):
        err = abs(_z_expectation(vec, site) - _z_expectation(ref_vec, site))
        assert err < tol, f"site={site} Δ⟨Z⟩={err}"


def _fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the squared overlap ``|⟨a|b⟩|²`` between two state vectors.

    Args:
        a: Reference state vector.
        b: Candidate state vector.

    Returns:
        Real fidelity in ``[0, 1]``.

    """
    return float(abs(np.vdot(a, b)) ** 2)


def _bond_second_schmidt(mps: MPS, bond: int) -> float:
    """Return the second Schmidt coefficient across an internal bond.

    Args:
        mps: MPS whose spectrum is measured.
        bond: Internal bond index ``0 <= b < length - 1``.

    Returns:
        Second normalized Schmidt value, or ``0.0`` when unavailable.

    """
    spec = mps.get_schmidt_spectrum([bond, bond + 1])
    vals = np.asarray(spec[~np.isnan(spec)], dtype=np.float64)
    norm = float(np.sum(vals**2))
    if norm > 0.0:
        vals /= np.sqrt(norm)
    return float(vals[1]) if len(vals) > 1 else 0.0


def _max_bond(mps: MPS) -> int:
    """Return the largest internal bond dimension of an MPS.

    Args:
        mps: MPS whose bond dimensions are read.

    Returns:
        Maximum bond dimension, or ``1`` for a single-site chain.

    """
    return max(mps.bond_dimensions()) if mps.length > 1 else 1


def _tdvp_params(*, max_bond_dim: int | None, tdvp_sweeps: int) -> StrongSimParams:
    """Build tight digital TDVP parameters for regression tests.

    Args:
        max_bond_dim: Optional bond-dimension cap.
        tdvp_sweeps: Number of symmetric TDVP substeps per gate.

    Returns:
        Strong-simulation parameters with exact preset and tight tolerances.

    """
    return StrongSimParams(
        preset="exact",
        get_state=True,
        max_bond_dim=max_bond_dim,
        tdvp_sweeps=tdvp_sweeps,
        svd_threshold=1e-14,
        krylov_tol=1e-12,
    )


def _qiskit_plus_rzz_reference(length: int, theta: float, *, sites: tuple[int, int]) -> np.ndarray:
    """Build exact Qiskit reference for RZZ on ``|+⟩^L``.

    Args:
        length: Number of qubits.
        theta: RZZ rotation angle.
        sites: Qubit indices passed to :meth:`QuantumCircuit.rzz`.

    Returns:
        State vector after preparing ``|+⟩^L`` and applying one RZZ gate.

    """
    qc = QuantumCircuit(length)
    qc.h(range(length))
    qc.rzz(theta, sites[0], sites[1])
    return np.asarray(Statevector(qc).data, dtype=np.complex128)


def _haar_random_mps(
    length: int,
    *,
    pad: int = HAAR_LR_PREP_PAD,
    seed: int = HAAR_LR_PREP_SEED,
) -> MPS:
    """Build a reproducible entangled Haar-random MPS for LR regression tests.

    Args:
        length: Chain length.
        pad: Target maximum internal bond dimension for Haar isometries.
        seed: NumPy RNG seed for the Haar draw.

    Returns:
        Fresh MPS initialized with ``state="haar-random"``.

    """
    with patch("numpy.random.default_rng", return_value=np.random.default_rng(seed)):
        return MPS(length, state="haar-random", pad=pad)


def _prep_state(name: str, length: int) -> MPS:
    """Prepare a named initial MPS for TDVP regression tests.

    Args:
        name: One of ``"plus"``, ``"zeros"``, ``"haar"``, or ``"low_depth"``.
        length: Chain length.

    Returns:
        Initial MPS for the requested label.

    Raises:
        ValueError: If ``name`` is not supported.

    """
    if name == "plus":
        return State(length, initial="x+").mps
    if name == "zeros":
        return State(length, initial="zeros").mps
    if name == "haar":
        return State(length, initial="haar-random").mps
    if name == "low_depth":
        prep_qc = QuantumCircuit(length)
        for i in range(0, length, 2):
            prep_qc.h(i)
        for i in range(length - 1):
            prep_qc.cx(i, i + 1)
        params = StrongSimParams(
            preset="exact",
            get_state=True,
            max_bond_dim=8,
            gate_mode="mpo",
            svd_threshold=1e-14,
            krylov_tol=1e-12,
        )
        result = Simulator(parallel=False, show_progress=False).run(
            State(length, initial="zeros"), prep_qc, params, None
        )
        assert result.output_state is not None
        return result.output_state.mps
    msg = f"Unknown initial state {name!r}"
    raise ValueError(msg)


def _apply_lr_gate(
    mps: MPS,
    gate_name: str,
    theta: float,
    *,
    max_bond_dim: int | None,
    sweeps: int,
) -> MPS:
    """Apply one long-range two-qubit Pauli rotation via windowed TDVP.

    Args:
        mps: Input MPS; not modified in place.
        gate_name: One of ``"rzz"``, ``"rxx"``, or ``"ryy"``.
        theta: Rotation angle.
        max_bond_dim: Optional bond-dimension cap.
        sweeps: Number of TDVP substeps.

    Returns:
        Deep-copied MPS after the gate application.

    Raises:
        ValueError: If ``gate_name`` is not supported.

    """
    if gate_name == "rzz":
        gate = GateLibrary.rzz([theta])
    elif gate_name == "rxx":
        gate = GateLibrary.rxx([theta])
    elif gate_name == "ryy":
        gate = GateLibrary.ryy([theta])
    else:
        msg = f"Unknown gate {gate_name!r}"
        raise ValueError(msg)
    gate.set_sites(0, mps.length - 1)
    out = copy.deepcopy(mps)
    apply_two_qubit_gate_tdvp(out, gate, _tdvp_params(max_bond_dim=max_bond_dim, tdvp_sweeps=sweeps))
    return out


def assert_mps_bond_invariants(mps: MPS, *, max_bond_dim: int | None = None) -> None:
    """Check neighbor tensor virtual dimensions match and respect an optional χ cap.

    Args:
        mps: MPS whose bond shapes are validated.
        max_bond_dim: Optional hard cap checked after shape consistency.

    """
    mps.assert_bond_shapes_consistent(max_bond_dim=max_bond_dim)
    if max_bond_dim is not None:
        assert all(dim <= max_bond_dim for dim in mps.bond_dimensions())


def _reliable_sweeps(length: int) -> int:
    """Legacy sweep count used by older regression tests (production default is 1).

    Returns:
        TDVP sweep count for tests that still sweep to convergence.

    """
    return 256 if length >= 10 else 64


def _qiskit_two_site_reference(
    length: int,
    gate_name: GateName,
    theta: float,
    *,
    sites: tuple[int, int],
    initial: str = "x+",
    prep_vec: np.ndarray | None = None,
) -> np.ndarray:
    """Exact Qiskit reference for one two-qubit Pauli rotation gate.

    Returns:
        State vector after evolving ``initial`` (or ``prep_vec``) with one gate.

    Raises:
        ValueError: If ``initial`` is not a supported label.

    """
    qc = QuantumCircuit(length)
    if prep_vec is not None:
        qc.initialize(prep_vec.tolist(), range(length))
    elif initial == "x+":
        qc.h(range(length))
    elif initial == "zeros":
        pass
    else:
        msg = f"Unknown initial {initial!r}"
        raise ValueError(msg)
    if gate_name == "rzz":
        qc.rzz(theta, sites[0], sites[1])
    elif gate_name == "rxx":
        qc.rxx(theta, sites[0], sites[1])
    else:
        qc.ryy(theta, sites[0], sites[1])
    return np.asarray(Statevector(qc).data, dtype=np.complex128)


def _double_theta_reference(length: int, theta: float, *, sites: tuple[int, int]) -> np.ndarray:
    """Build reference state from applying RZZ twice (the old 2θ bug).

    Returns:
        State vector after two identical RZZ applications on |+⟩^L.

    """
    qc = QuantumCircuit(length)
    qc.h(range(length))
    qc.rzz(theta, sites[0], sites[1])
    qc.rzz(theta, sites[0], sites[1])
    return np.asarray(Statevector(qc).data, dtype=np.complex128)


def _plus_rzz_overlap(_length: int, theta: float) -> float:
    """Return |⟨+|ψ⟩|² after RZZ(0, L-1, θ) on |+⟩^L (global phase ignored).

    Returns:
        Squared overlap with the uniform superposition state.

    """
    return math.cos(theta / 2.0) ** 2


def _run_circuit(
    prep: MPS,
    qc: QuantumCircuit,
    *,
    max_bond_dim: int,
    sweeps: int,
) -> MPS:
    """Evolve a prepared MPS through a Qiskit circuit with TDVP gates.

    Args:
        prep: Initial MPS copied before simulation.
        qc: Circuit executed with ``gate_mode="tdvp"``.
        max_bond_dim: Bond-dimension cap for the simulation.
        sweeps: Number of TDVP substeps per gate.

    Returns:
        Output MPS from the simulator run.

    """
    init = State(prep.length, tensors=[t.copy() for t in prep.tensors])
    params = StrongSimParams(
        preset="exact",
        get_state=True,
        max_bond_dim=max_bond_dim,
        tdvp_sweeps=sweeps,
        gate_mode="tdvp",
        svd_threshold=1e-14,
        krylov_tol=1e-12,
    )
    result = Simulator(parallel=False, show_progress=False).run(init, qc, params, None)
    assert result.output_state is not None
    return result.output_state.mps
