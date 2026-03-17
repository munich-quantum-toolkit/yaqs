# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Backend state evolution and tomography sequence workers."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from mqt.yaqs.analog.analog_tjm import analog_tjm_1, analog_tjm_2
from mqt.yaqs.analog.mcwf import mcwf, preprocess_mcwf
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable
from mqt.yaqs.core.libraries.gate_library import X, Y, Z
from mqt.yaqs.simulator import WORKER_CTX

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.core.data_structures.noise_model import NoiseModel
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


def _reprepare_site_zero_forced(
    mps: MPS,
    proj_state: NDArray[np.complex128],
    new_state: NDArray[np.complex128],
) -> float:
    """Project site 0 onto proj_state and reprepare new_state (in-place). Returns prob."""
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
    """Reprepare site 0 for a dense vector state. Returns (new_state_vec, prob)."""
    psi_reshaped = state_vec.reshape(2, state_vec.shape[0] // 2)
    env_vec = proj_state.conj() @ psi_reshaped
    prob = float(np.linalg.norm(env_vec) ** 2)
    if prob > 1e-15:
        env_vec /= np.sqrt(prob)
    return np.outer(new_state, env_vec).flatten(), prob


def _reconstruct_state(expectations: dict[str, float]) -> NDArray[np.complex128]:
    """Reconstruct single-qubit density matrix from Pauli expectations."""
    eye = np.eye(2, dtype=complex)
    return 0.5 * (
        eye
        + expectations["x"] * X().matrix
        + expectations["y"] * Y().matrix
        + expectations["z"] * Z().matrix
    )


def _get_rho_site_zero(state: MPS | NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Extract single-qubit (site 0) density matrix from MPS or dense vector."""
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
    return trace * _reconstruct_state({"x": rx / trace, "y": ry / trace, "z": rz / trace})


def _initialize_backend_state(
    operator: MPO, solver: str
) -> MPS | NDArray[np.complex128]:
    """Initialise |0...0> state for the given solver."""
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
    """Reprepare site 0 for the given solver. Returns (new_state, prob)."""
    if solver == "MCWF":
        assert isinstance(state, np.ndarray)
        return _reprepare_site_zero_vector_forced(state, proj_state, new_state)
    assert isinstance(state, MPS)
    new_mps = copy.deepcopy(state)
    prob = _reprepare_site_zero_forced(new_mps, proj_state, new_state)
    return new_mps, prob


def _evolve_backend_state(
    state: MPS | NDArray[np.complex128],
    operator: MPO,
    noise_model: NoiseModel | None,
    step_params: AnalogSimParams,
    solver: str,
    traj_idx: int = 0,
    static_ctx: Any = None,
) -> MPS | NDArray[np.complex128]:
    """Evolve state for one step using the given solver."""
    if solver == "MCWF":
        if not isinstance(state, np.ndarray):
            msg = f"MCWF solver requires dense NDArray state, got {type(state)}."
            raise TypeError(msg)
        if static_ctx is None:
            dummy_mps = MPS(length=operator.length, state="zeros")
            static_ctx = preprocess_mcwf(dummy_mps, operator, noise_model, step_params)
        dynamic_ctx = copy.copy(static_ctx)
        dynamic_ctx.psi_initial = state
        dynamic_ctx.sim_params = step_params
        dynamic_ctx.output_state = None
        mcwf((traj_idx, dynamic_ctx))
        out = dynamic_ctx.output_state
        if out is None:
            raise RuntimeError("MCWF backend returned None state.")
        return cast("NDArray[np.complex128]", out)

    if not isinstance(state, MPS):
        msg = f"TJM solver requires MPS state, got {type(state)}."
        raise TypeError(msg)
    backend = analog_tjm_1 if step_params.order == 1 else analog_tjm_2
    step_params.output_state = None
    backend((traj_idx, state, noise_model, step_params, operator))
    out = step_params.output_state
    if out is None:
        raise RuntimeError("TJM backend returned None state.")
    return cast("MPS", out)


def _tomography_sequence_worker(
    job_idx: int,
    payload: dict[str, Any] | None = None,
) -> tuple[int, int, NDArray[np.complex128], float]:
    """Execute one tomography trajectory given pre-specified (psi_meas, psi_prep) pairs."""
    ctx = payload if payload is not None else WORKER_CTX
    num_trajectories: int = ctx["num_trajectories"]
    s_idx: int = job_idx // num_trajectories
    traj_idx: int = job_idx % num_trajectories
    psi_pairs = ctx["psi_pairs"][s_idx]
    operator = ctx["operator"]
    sim_params = ctx["sim_params"]
    timesteps: list[float] = ctx["timesteps"]
    noise_model = ctx["noise_model"]

    if noise_model is None:
        assert num_trajectories == 1, "num_trajectories must be 1 when noise_model is None."

    solver = sim_params.solver
    current_state = _initialize_backend_state(operator, solver)
    weight = 1.0

    for step_i, (psi_meas, psi_prep) in enumerate(psi_pairs):
        current_state, step_prob = _reprepare_backend_state_forced(
            current_state, psi_meas, psi_prep, solver
        )
        weight *= step_prob
        if weight < 1e-15:
            break
        duration = timesteps[step_i]
        step_params = copy.deepcopy(sim_params)
        step_params.elapsed_time = duration
        step_params.num_traj = 1
        step_params.show_progress = False
        step_params.get_state = True
        n_steps = max(1, int(np.round(duration / step_params.dt)))
        step_params.times = np.linspace(0, n_steps * step_params.dt, n_steps + 1)
        static_ctx = ctx.get("mcwf_static_ctx")
        current_state = _evolve_backend_state(
            current_state,
            operator,
            noise_model,
            step_params,
            solver,
            traj_idx=traj_idx,
            static_ctx=static_ctx,
        )

    rho_final = _get_rho_site_zero(current_state)
    return (s_idx, traj_idx, rho_final, weight)


def _sis_evolve_worker(job_idx: int) -> MPS | NDArray[np.complex128]:
    """Worker function for parallel SIS particle evolution."""
    from mqt.yaqs.simulator import WORKER_CTX as SIS_CTX

    state = SIS_CTX["prep_states"][job_idx]
    duration = SIS_CTX["duration"]
    static_ctx = SIS_CTX["static_ctx"]
    sim_params = SIS_CTX["sim_params"]
    operator = SIS_CTX["operator"]
    noise_model = SIS_CTX["noise_model"]
    solver = sim_params.solver
    sp = copy.deepcopy(sim_params)
    sp.elapsed_time = duration
    sp.num_traj = 1
    sp.get_state = True
    sp.show_progress = False
    n_steps = max(1, int(np.round(duration / sp.dt)))
    sp.times = np.linspace(0, n_steps * sp.dt, n_steps + 1)
    return _evolve_backend_state(
        state, operator, noise_model, sp, solver, traj_idx=0, static_ctx=static_ctx
    )
