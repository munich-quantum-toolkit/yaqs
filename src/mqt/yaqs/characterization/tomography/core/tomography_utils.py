# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Tomography backend state evolution (distinct from :mod:`mqt.yaqs.simulator`).

See :mod:`mqt.yaqs.characterization.tomography.estimate.sampling` for **sequence** vs **trajectory**
(noise-model realization) terminology.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from mqt.yaqs.analog.analog_tjm import analog_tjm_1, analog_tjm_2
from mqt.yaqs.analog.mcwf import mcwf, preprocess_mcwf
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import X, Y, Z
from .predictor_encoding import normalize_rho_from_backend_output, pack_rho8

if TYPE_CHECKING:
    from numpy.typing import NDArray


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


def make_mcwf_static_context(
    operator: MPO,
    sim_params: AnalogSimParams,
    *,
    noise_model: NoiseModel | None = None,
) -> Any:
    """Shared ``preprocess_mcwf`` context for batch tomography / predictor workers (dummy reference MPS)."""
    dummy_mps = MPS(length=operator.length, state="zeros")
    return preprocess_mcwf(dummy_mps, operator, noise_model, sim_params)


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
            static_ctx = make_mcwf_static_context(operator, step_params, noise_model=noise_model)
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


def _rollout_intervention_steps(
    current_state: MPS | NDArray[np.complex128],
    *,
    psi_pairs: list[tuple[np.ndarray, np.ndarray]],
    step_durations: list[float],
    operator: MPO,
    operators_per_step: list[MPO] | None,
    sim_params: AnalogSimParams,
    noise_model: NoiseModel | None,
    solver: str,
    traj_idx: int,
    default_static_ctx: Any,
    static_ctx_per_step: list[Any] | None,
    record_step_states: bool,
) -> tuple[MPS | NDArray[np.complex128], float, list[np.ndarray] | None]:
    """Shared prep → evolve loop along one **sequence** of interventions.

    When ``record_step_states`` is False (exact-comb pipeline), only the final state matters.
    When True (surrogate **sequence rollout**), returns packed site-0 densities after each segment
    (length ``k``, padded if the weight collapses early). This is not the same word as **trajectory**
    in :mod:`~mqt.yaqs.characterization.tomography.estimate.sampling` (stochastic MCWF/TJM runs).
    """
    k = len(psi_pairs)
    if len(step_durations) != k:
        msg = f"step_durations length {len(step_durations)} != number of psi_pairs {k}."
        raise ValueError(msg)
    if operators_per_step is not None and len(operators_per_step) != k:
        msg = f"operators_per_step length must match psi_pairs ({k})."
        raise ValueError(msg)
    if static_ctx_per_step is not None and len(static_ctx_per_step) != k:
        msg = f"static_ctx_per_step length must match psi_pairs ({k})."
        raise ValueError(msg)

    weight = 1.0
    rho_steps: list[np.ndarray] = []
    last_packed: np.ndarray | None = None
    if record_step_states:
        rho0_raw = _get_rho_site_zero(current_state)
        last_packed = pack_rho8(normalize_rho_from_backend_output(rho0_raw)).astype(np.float32)

    for step_i, (psi_meas, psi_prep) in enumerate(psi_pairs):
        current_state, step_prob = _reprepare_backend_state_forced(
            current_state, psi_meas, psi_prep, solver
        )
        weight *= step_prob
        if weight < 1e-15:
            if record_step_states:
                while len(rho_steps) < k and last_packed is not None:
                    rho_steps.append(last_packed.copy())
            break

        duration = step_durations[step_i]
        op_step = operator if operators_per_step is None else operators_per_step[step_i]
        static_ctx = default_static_ctx if static_ctx_per_step is None else static_ctx_per_step[step_i]

        step_params = copy.deepcopy(sim_params)
        step_params.elapsed_time = duration
        step_params.num_traj = 1
        step_params.show_progress = False
        step_params.get_state = True
        n_steps = max(1, int(np.round(duration / step_params.dt)))
        step_params.times = np.linspace(0, n_steps * step_params.dt, n_steps + 1)

        current_state = _evolve_backend_state(
            current_state,
            op_step,
            noise_model,
            step_params,
            solver,
            traj_idx=traj_idx,
            static_ctx=static_ctx,
        )

        if record_step_states:
            rho_step = _get_rho_site_zero(current_state)
            rho_norm = normalize_rho_from_backend_output(rho_step)
            last_packed = pack_rho8(rho_norm).astype(np.float32)
            rho_steps.append(last_packed)

    if record_step_states:
        while len(rho_steps) < k and last_packed is not None:
            rho_steps.append(last_packed.copy())
        return current_state, weight, rho_steps
    return current_state, weight, None


def build_initial_psi(
    rho_in: np.ndarray,
    *,
    length: int,
    rng: np.random.Generator,
    init_mode: str,
    return_eig_sample: bool = False,
) -> Any:
    """Pure MCWF state vector on ``length`` qubits whose site-0 reduced state follows ``rho_in``."""
    return _initial_mcwf_state_from_rho0(
        rho_in,
        length,
        rng=rng,
        init_mode=init_mode,
        return_eig_sample=return_eig_sample,
    )


def _initial_mcwf_state_from_rho0(
    rho: np.ndarray,
    length: int,
    *,
    rng: np.random.Generator | None = None,
    init_mode: str = "eigenstate",
    return_eig_sample: bool = False,
) -> Any:
    if rho.size != 4:
        msg = "rho must be a 2x2 reduced density matrix."
        raise ValueError(msg)
    rho = np.asarray(rho, dtype=np.complex128).reshape(2, 2)
    rho = 0.5 * (rho + rho.conj().T)
    w, v = np.linalg.eigh(rho)
    w = np.maximum(w.real, 0.0)
    s = float(w.sum())
    if s > 1e-15:
        w = w / s
    else:
        w = np.array([1.0, 0.0], dtype=np.float64)

    if init_mode not in {"eigenstate", "purified"}:
        raise ValueError(f"init_mode must be 'eigenstate' or 'purified', got {init_mode!r}")

    if init_mode == "eigenstate":
        if rng is None:
            rng = np.random.default_rng()
        idx = int(rng.choice(2, p=w))
        p = float(w[idx])
        v_idx = v[:, idx].astype(np.complex128)
        if length <= 1:
            psi = v_idx
        else:
            env0 = np.array([1.0, 0.0], dtype=np.complex128)
            env_state = env0
            for _ in range(length - 2):
                env_state = np.kron(env_state, env0)
            psi = np.kron(v_idx, env_state)
        if return_eig_sample:
            return psi, idx, p
        return psi

    if length <= 1:
        psi = np.zeros(2, dtype=np.complex128)
        for i in range(2):
            if w[i] > 1e-15:
                psi += np.sqrt(w[i]) * v[:, i].astype(np.complex128)
        nrm = float(np.linalg.norm(psi))
        psi = psi / max(nrm, 1e-15)
        return (psi, 0, float(w[0])) if return_eig_sample else psi

    psi_2 = np.zeros(4, dtype=np.complex128)
    for i in range(2):
        if w[i] < 1e-15:
            continue
        anc = np.zeros(2, dtype=np.complex128)
        anc[i] = 1.0
        psi_2 += np.sqrt(w[i]) * np.kron(v[:, i].astype(np.complex128), anc)
    nrm = float(np.linalg.norm(psi_2))
    if nrm < 1e-15:
        psi_2 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    else:
        psi_2 /= nrm
    psi = psi_2
    for _ in range(length - 2):
        psi = np.kron(psi, np.array([1.0, 0.0], dtype=np.complex128))
    return (psi, 0, float(w[0])) if return_eig_sample else psi
