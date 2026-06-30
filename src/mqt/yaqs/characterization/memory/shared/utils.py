# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""MCWF/TJM state evolution helpers shared by process-tensor tomography and surrogate workflows.

Distinct from :mod:`mqt.yaqs.simulator`. See
:mod:`mqt.yaqs.characterization.memory.backends.tomography.data` for **sequence** vs **trajectory**
terminology.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from mqt.yaqs.analog.analog_tjm import analog_tjm_1, analog_tjm_2
from mqt.yaqs.analog.mcwf import MCWFContext, mcwf, preprocess_mcwf
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable
from mqt.yaqs.core.libraries.gate_library import X, Y, Z

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.core.data_structures.mpo import MPO
    from mqt.yaqs.core.data_structures.noise_model import NoiseModel
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

StochasticSolver = Literal["MCWF", "TJM"]
CharacterizerRepresentation = Literal["vector", "mps", "auto"]
DEFAULT_VECTOR_MAX_QUBITS = 10


def resolve_characterizer_representation(
    chain_length: int,
    representation: CharacterizerRepresentation,
    *,
    vector_max_qubits: int = DEFAULT_VECTOR_MAX_QUBITS,
) -> Literal["vector", "mps"]:
    """Resolve ``auto`` to ``vector`` (MCWF) or ``mps`` (TJM) by chain length.

    Args:
        chain_length: Number of qubits in the Hamiltonian chain.
        representation: User representation selection.
        vector_max_qubits: Inclusive upper qubit count for ``auto`` → ``vector``.

    Returns:
        Resolved ``"vector"`` or ``"mps"``.

    Raises:
        ValueError: If ``representation`` is invalid.
    """
    rep = str(representation).strip().lower()
    if rep == "vector":
        return "vector"
    if rep == "mps":
        return "mps"
    if rep == "auto":
        return "vector" if int(chain_length) <= int(vector_max_qubits) else "mps"
    msg = f"representation must be 'vector', 'mps', or 'auto', got {representation!r}."
    raise ValueError(msg)


def representation_to_solver(rep: Literal["vector", "mps"]) -> StochasticSolver:
    """Map characterizer representation to the internal stochastic solver name.

    Args:
        rep: Characterizer representation (``"vector"`` or ``"mps"``).

    Returns:
        ``"MCWF"`` for ``vector`` and ``"TJM"`` for ``mps``.
    """
    return "MCWF" if rep == "vector" else "TJM"


def validate_stochastic_solver(solver: StochasticSolver | str | None) -> StochasticSolver:
    """Validate and normalize a stochastic solver name.

    Args:
        solver: Solver override, or ``None`` for the MCWF default.

    Returns:
        ``"MCWF"`` or ``"TJM"``.

    Raises:
        ValueError: If ``solver`` is not ``"MCWF"`` or ``"TJM"``.
    """
    if solver is None:
        return "MCWF"
    if solver not in {"MCWF", "TJM"}:
        msg = f"solver must be 'MCWF' or 'TJM', got {solver!r}."
        raise ValueError(msg)
    return cast("StochasticSolver", solver)


def resolve_stochastic_solver(
    sim_params: AnalogSimParams,
    *,
    solver: StochasticSolver | None = None,
    representation: CharacterizerRepresentation | None = None,
    chain_length: int | None = None,
    vector_max_qubits: int = DEFAULT_VECTOR_MAX_QUBITS,
) -> StochasticSolver:
    """Return the stochastic unraveling backend for process-tensor schedule simulation.

    Returns:
        Resolved solver name ``"MCWF"`` or ``"TJM"``.

    Raises:
        ValueError: If ``representation`` is set without ``chain_length``.
    """
    if solver is not None:
        return validate_stochastic_solver(solver)
    if representation is not None:
        if chain_length is None:
            msg = "chain_length is required when representation= is passed."
            raise ValueError(msg)
        rep = resolve_characterizer_representation(
            int(chain_length),
            representation,
            vector_max_qubits=vector_max_qubits,
        )
        return representation_to_solver(rep)
    legacy = getattr(sim_params, "solver", None)
    if legacy in {"MCWF", "TJM"}:
        return legacy
    return "MCWF"


def _reprepare_site_zero_forced(
    mps: MPS,
    proj_state: NDArray[np.complex128],
    new_state: NDArray[np.complex128],
) -> float:
    """Project site 0 and reprepare it to a new state (MPS backend).

    Args:
        mps: State in MPS form (modified in-place).
        proj_state: Single-qubit ket to project onto (shape (2,)).
        new_state: Single-qubit ket to reprepare to (shape (2,)).

    Returns:
        The projection probability.
    """
    mps.set_canonical_form(orthogonality_center=0)
    t_mps = mps.tensors[0]
    env_vec = np.einsum("s c, s -> c", t_mps[:, 0, :], proj_state.conj())
    prob = float(np.linalg.norm(env_vec) ** 2)
    if prob > 1e-15:
        env_vec /= np.sqrt(prob)
    d, chi = new_state.shape[0], env_vec.shape[0]
    new_tensor = np.zeros((d, 1, chi), dtype=np.complex128)
    for s in range(d):
        new_tensor[s, 0, :] = new_state[s] * env_vec
    mps.tensors[0] = new_tensor
    final_norm = mps.norm()
    if abs(final_norm) > 1e-15:
        mps.tensors[0] /= final_norm
    return prob


def _reprepare_site_zero_vector_forced(
    state_vec: NDArray[np.complex128],
    proj_state: NDArray[np.complex128],
    new_state: NDArray[np.complex128],
) -> tuple[NDArray[np.complex128], float]:
    """Project+reprepare site 0 for a dense state vector backend.

    Args:
        state_vec: Full state vector of shape ``(2**L,)``.
        proj_state: Single-qubit ket to project onto (shape (2,)).
        new_state: Single-qubit ket to reprepare to (shape (2,)).

    Returns:
        Tuple ``(new_state_vec, prob)`` where ``new_state_vec`` is the updated full state vector and
        ``prob`` is the projection probability.
    """
    psi_reshaped = state_vec.reshape(2, state_vec.shape[0] // 2)
    env_vec = proj_state.conj() @ psi_reshaped
    prob = float(np.linalg.norm(env_vec) ** 2)
    if prob > 1e-15:
        env_vec /= np.sqrt(prob)
    return np.outer(new_state, env_vec).flatten(), prob


def assemble_state_from_expectations(expectations: dict[str, float]) -> NDArray[np.complex128]:
    """Reconstruct a single-qubit density matrix from Pauli expectations.

    Args:
        expectations: Mapping with keys ``"x"``, ``"y"``, ``"z"`` for Pauli expectations.

    Returns:
        2x2 complex density matrix.
    """
    eye = np.eye(2, dtype=complex)
    return 0.5 * (
        eye + expectations["x"] * X().matrix + expectations["y"] * Y().matrix + expectations["z"] * Z().matrix
    )


def extract_site0_rho(state: MPS | NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Extract the site-0 reduced density matrix.

    Args:
        state: Either an MPS state or a dense state vector.

    Returns:
        2x2 complex reduced density matrix on site 0.
    """
    if isinstance(state, np.ndarray):
        rho = np.reshape(state, (2, -1))
        return rho @ rho.conj().T
    assert isinstance(state, MPS)
    trace = float(state.norm() ** 2)
    if trace < 1e-15:
        return np.zeros((2, 2), dtype=np.complex128)
    rx = state.expect(Observable(X(), sites=[0]))
    ry = state.expect(Observable(Y(), sites=[0]))
    rz = state.expect(Observable(Z(), sites=[0]))
    return trace * assemble_state_from_expectations({"x": rx / trace, "y": ry / trace, "z": rz / trace})


def _initialize_backend_state(operator: MPO, solver: str) -> MPS | NDArray[np.complex128]:
    """Initialize the all-zeros state for a given backend solver.

    Args:
        operator: Hamiltonian MPO (used for chain length).
        solver: Backend solver name (e.g. ``"MCWF"`` or ``"TJM"``).

    Returns:
        Dense state vector for MCWF or an MPS for TJM.
    """
    if solver == "MCWF":
        psi = np.zeros(2**operator.length, dtype=np.complex128)
        psi[0] = 1.0
        return psi
    return MPS(length=operator.length, state="zeros")


def _reprepare_backend_state_forced(
    state: MPS | NDArray[np.complex128],
    proj_state: NDArray[np.complex128],
    new_state: NDArray[np.complex128],
    solver: str,
) -> tuple[MPS | NDArray[np.complex128], float]:
    """Project+reprepare site 0 using the given solver backend.

    Args:
        state: Current backend state.
        proj_state: Single-qubit ket to project onto (shape (2,)).
        new_state: Single-qubit ket to reprepare to (shape (2,)).
        solver: Backend solver name.

    Returns:
        Tuple ``(state_out, prob)`` where ``state_out`` is the updated backend state and ``prob`` is
        the projection probability.

    Raises:
        TypeError: If ``state`` is incompatible with ``solver``.
    """
    if solver == "MCWF":
        if not isinstance(state, np.ndarray):
            msg = f"MCWF solver requires dense NDArray state, got {type(state)}."
            raise TypeError(msg)
        state_vec = cast("NDArray[np.complex128]", np.asarray(state, dtype=np.complex128))
        return _reprepare_site_zero_vector_forced(state_vec, proj_state, new_state)
    assert isinstance(state, MPS)
    new_mps = copy.deepcopy(state)
    prob = _reprepare_site_zero_forced(new_mps, proj_state, new_state)
    return new_mps, prob


def _single_qubit_unitary_mapping_basis0_to_ket(psi: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Return a 2x2 unitary whose first column is ``psi`` (normalized computational |0> -> ``psi``).

    Args:
        psi: Target single-qubit ket (length 2).

    Returns:
        Unitary matrix mapping the computational ``|0>`` basis state to ``psi``.
    """
    p = np.asarray(psi, dtype=np.complex128).reshape(2)
    nrm = float(np.linalg.norm(p))
    p = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128) if nrm < 1e-15 else p / nrm
    a, b = p[0], p[1]
    return np.array([[a, -np.conj(b)], [b, np.conj(a)]], dtype=np.complex128)


_SITE0_KET = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)


def _apply_cut_preparation_step(
    state: MPS | NDArray[np.complex128],
    psi_prep: NDArray[np.complex128],
    solver: str,
    *,
    chain_length: int,
) -> tuple[MPS | NDArray[np.complex128], float]:
    """Apply a ``cut_preparation`` intervention on site 0.

    * ``chain_length == 1``: unconditional assignment to ``|ψ⟩`` (process-tensor reference).
    * ``chain_length > 1``: project site 0 onto ``|0⟩``, reprepare to ``|ψ⟩``; other
      sites are unchanged.

    Branch weight is always ``1.0``.

    Args:
        state: Current backend state.
        psi_prep: Target site-0 ket.
        solver: Backend solver name.
        chain_length: Number of qubits in the chain.

    Returns:
        Tuple ``(state_out, 1.0)``.

    Raises:
        TypeError: If ``solver`` is ``"MCWF"``, ``chain_length == 1``, and ``state`` is not a
            dense :class:`~numpy.ndarray`.
    """
    p = np.asarray(psi_prep, dtype=np.complex128).reshape(2)
    nrm = float(np.linalg.norm(p))
    p = _SITE0_KET if nrm < 1e-15 else p / nrm
    if int(chain_length) == 1:
        if solver == "MCWF":
            if not isinstance(state, np.ndarray):
                msg = f"MCWF solver requires dense NDArray state, got {type(state)}."
                raise TypeError(msg)
            return p.astype(np.complex128).copy(), 1.0
        u = _single_qubit_unitary_mapping_basis0_to_ket(p)
        new_mps = MPS(length=1, state="zeros")
        t0 = np.asarray(new_mps.tensors[0], dtype=np.complex128)
        new_mps.tensors[0] = np.einsum("ab,bcd->acd", u, t0)
        return new_mps, 1.0
    state_out, _prob = _reprepare_backend_state_forced(state, _SITE0_KET, p, solver)
    return state_out, 1.0


def _apply_backend_unitary_site_zero(
    state: MPS | NDArray[np.complex128],
    unitary: NDArray[np.complex128],
    solver: str,
) -> MPS | NDArray[np.complex128]:
    """Apply a single-qubit unitary on site 0 without introducing measurement weight.

    Args:
        state: Current backend state.
        unitary: Single-qubit unitary matrix.
        solver: Backend solver name.

    Returns:
        Updated backend state after applying the unitary on site 0.
    """
    u = np.asarray(unitary, dtype=np.complex128).reshape(2, 2)
    if solver == "MCWF":
        assert isinstance(state, np.ndarray)
        psi = np.asarray(state, dtype=np.complex128).reshape(2, -1)
        return (u @ psi).reshape(-1)
    assert isinstance(state, MPS)
    new_mps = copy.deepcopy(state)
    t0 = np.asarray(new_mps.tensors[0], dtype=np.complex128)
    new_mps.tensors[0] = np.einsum("ab,bcd->acd", u, t0)
    return new_mps


def _evolve_backend_state(
    state: MPS | NDArray[np.complex128],
    operator: MPO,
    noise_model: NoiseModel | None,
    step_params: AnalogSimParams,
    solver: str,
    traj_idx: int = 0,
    static_ctx: MCWFContext | None = None,
) -> MPS | NDArray[np.complex128]:
    """Evolve a backend state forward in time by one segment.

    Args:
        state: Current backend state (dense vector for MCWF, MPS for TJM).
        operator: Hamiltonian MPO.
        noise_model: Optional noise model; ``None`` for deterministic evolution.
        step_params: Simulation parameters for this step (duration and time grid are read here).
        solver: Backend solver name.
        traj_idx: MCWF trajectory index (used for deterministic seeding in the backend).
        static_ctx: Optional preprocessed MCWF context.

    Returns:
        Updated backend state after evolution.

    Raises:
        TypeError: If ``state`` is incompatible with ``solver``.
        RuntimeError: If the backend fails to produce an output state.
    """
    if solver == "MCWF":
        if not isinstance(state, np.ndarray):
            msg = f"MCWF solver requires dense NDArray state, got {type(state)}."
            raise TypeError(msg)
        if static_ctx is None:
            static_ctx = make_mcwf_static_context(operator, step_params, noise_model=noise_model)
        dynamic_ctx = copy.copy(static_ctx)
        dynamic_ctx.psi_initial = cast("NDArray[np.complex128]", np.asarray(state, dtype=np.complex128))
        dynamic_ctx.sim_params = step_params
        _, _, out = mcwf((traj_idx, dynamic_ctx))
        if out is None:
            msg = "MCWF backend returned None state."
            raise RuntimeError(msg)
        return out

    if not isinstance(state, MPS):
        msg = f"TJM solver requires MPS state, got {type(state)}."
        raise TypeError(msg)
    step_params_tjm = copy.copy(step_params)
    step_params_tjm.get_state = True
    backend = analog_tjm_1 if step_params_tjm.order == 1 else analog_tjm_2
    _, _, out = backend((traj_idx, state, noise_model, step_params_tjm, operator))
    if out is None:
        msg = "TJM backend returned None state."
        raise RuntimeError(msg)
    return out


def make_mcwf_static_context(
    operator: MPO,
    sim_params: AnalogSimParams,
    *,
    noise_model: NoiseModel | None = None,
) -> MCWFContext:
    """Build a reusable MCWF preprocessing context.

    Args:
        operator: Hamiltonian MPO.
        sim_params: Simulation parameters.
        noise_model: Optional noise model.

    Returns:
        A backend-specific preprocessing context suitable for reuse across many worker calls.
    """
    dummy_mps = MPS(length=operator.length, state="zeros")
    return preprocess_mcwf(dummy_mps, operator, noise_model, sim_params)
