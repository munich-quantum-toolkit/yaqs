"""
Compute (alpha, kappa) for *bit-flip* Lindblad noise (jump operator X on each site)
for a 16-site Heisenberg-XXX chain, using YAQS TDVP trajectories.

Target:
- L = 16
- gamma = 0.2
- dt = 0.05
- T = 5.0

Interpretation (matches your paper framing):
- Compare two *physically equivalent* unravelings of the SAME Lindblad generator:
    U1: "jump" unraveling with local pauli_x jump operators
    U2: "measurement" unraveling that reproduces the same Lindbladian

- alpha := chi_max(U2) / chi_max(U1)   (bond-dimension inflation)
- kappa := N_req(U2) / N_req(U1)       (sampling inflation)
  where N_req(U) is the number of trajectories needed to reach a target statistical
  error on a chosen observable at a chosen time (default: final-time XX_mid).

This script is written to be robust to small API differences by:
- Trying multiple ways to extract chi_max
- Trying multiple ways to extract per-trajectory samples
  (if YAQS doesn't expose per-trajectory values directly, it falls back to running
   many num_traj=1 runs to get samples.)

Adjust the "EXTRACTORS" section if your YAQS build stores these differently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Any
import math
import numpy as np

from mqt.yaqs.core.data_structures.networks import MPS, MPO
from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import XX
from mqt.yaqs.core.data_structures.noise_model import NoiseModel


# ----------------------------
# Problem setup
# ----------------------------
L = 12
J = 1.0
h = 1.0
gamma = 0.01
dt = 0.05
T = 2.0

# Observable used for sampling convergence (edit if you prefer Z or something else)
OBS = [Observable(XX(), [L // 2, L // 2 + 1])] + [Observable("max_bond")]  # XX on the middle bond


# ----------------------------
# Heisenberg MPO
# ----------------------------
def build_heisenberg_mpo(L: int, J: float, h: float) -> MPO:
    """
    Build the Hamiltonian MPO.

    NOTE: YAQS function names can vary. Common patterns I've seen:
      - init_heisenberg_xxx(L, J, h)
      - init_xxx(L, J, h)
      - init_heisenberg(L, J, h)
    Adjust this function to the initializer your YAQS version exposes.
    """
    H = MPO.heisenberg(L, J, J, J, h)
    # H = MPO.ising(L, J, h)
    return H


# ----------------------------
# Noise models: same Lindbladian, different unravelings
# ----------------------------
def noise_u1_bitflip_jump(L: int, gamma: float) -> NoiseModel:
    """
    U1: Jump unraveling with collapse operator sqrt(gamma) X_i.
    In YAQS NoiseModel schema, this is typically name='pauli_x' with strength=gamma.
    """
    return NoiseModel(
        [{"name": "pauli_z", "sites": [i], "strength": gamma} for i in range(L)]
    )


def noise_u2_bitflip_measurement(L: int, gamma: float) -> NoiseModel:
    """
    U2: Measurement unraveling reproducing the same Lindbladian as pauli_x jumps.

    In your depolarizing example you used strength=2*gamma for the (measure_0/1, measure_x_0/1, ...)
    channels. For pure X noise, it's typically the pair ('measure_x_0', 'measure_x_1') with strength=2*gamma
    per site to match the same Lindblad superoperator normalization.

    If your YAQS naming differs (e.g. 'measureX0', 'measure_x_plus', etc.), update here.
    """
    return NoiseModel(
        [{"name": name, "sites": [i], "strength": 2.0 * gamma}
         for i in range(L)
         for name in ["measure_0", "measure_1"]]
    )


# ----------------------------
# YAQS run helpers
# ----------------------------
@dataclass
class RunResult:
    obs_time_series: np.ndarray          # ensemble average over time (len = n_t)
    chi_max: Optional[float]             # maximum bond dimension during run (if extractable)
    traj_samples_final: Optional[np.ndarray]  # per-trajectory samples at final time if extractable


def run_yaqs(
    H: MPO,
    noise_model: NoiseModel,
    *,
    L: int,
    dt: float,
    T: float,
    num_traj: int,
    seed: Optional[int] = None,
) -> RunResult:
    """
    Run YAQS for a given unraveling and return:
    - ensemble average observable time series
    - chi_max (best effort)
    - per-trajectory final-time samples (best effort)
    """
    if seed is not None:
        np.random.seed(seed)

    state = MPS(state="Neel", length=L)

    sim_params = AnalogSimParams(
        observables=OBS,
        elapsed_time=T,
        dt=dt,
        num_traj=num_traj,
        # keep your defaults; edit if needed
        threshold=1e-6,
        trunc_mode="discarded_weight",
        order=2,
        sample_timesteps=True,
    )

    simulator.run(state, H, sim_params, noise_model=noise_model)

    # Ensemble-average time series (this is what your current code uses)
    obs_avg = np.array(sim_params.observables[0].results, dtype=float)

    # ---- EXTRACTORS (best-effort) ----
    # 1) chi_max
    chi_max = sim_params.observables[1].results[-1]
    # common places where chi might live: sim_params.max_bond, sim_params.chi_max,
    # state.max_bond_dim, or an attached observable. Try several.

    # 2) per-trajectory final-time samples (needed for kappa)
    # Some builds store per-trajectory data, e.g. observables[0].traj_results or .samples.
    traj_samples_final = None
    obs0 = sim_params.observables[0]
    for attr in ["trajectories"]:
        if hasattr(obs0, attr):
            arr = getattr(obs0, attr)
            # Expect shape (num_traj, n_t) or list-of-lists; take final time
            try:
                arr_np = np.array(arr, dtype=float)
                if arr_np.ndim == 2 and arr_np.shape[0] == num_traj:
                    traj_samples_final = arr_np[:, -1]
                elif arr_np.ndim == 1 and arr_np.shape[0] == num_traj:
                    traj_samples_final = arr_np
            except Exception:
                pass
            if traj_samples_final is not None:
                break

    return RunResult(
        obs_time_series=obs_avg,
        chi_max=chi_max,
        traj_samples_final=traj_samples_final,
    )


def collect_final_time_samples_via_single_traj(
    H: MPO,
    noise_model: NoiseModel,
    *,
    L: int,
    dt: float,
    T: float,
    N: int,
    seed0: int = 1234,
) -> np.ndarray:
    """
    Fallback sampler if YAQS doesn't expose per-trajectory values in a num_traj>1 run.

    Runs N times with num_traj=1 and records the final-time observable.
    This is slower but always works if the simulator is deterministic per seed.
    """
    samples = np.empty(N, dtype=float)
    for k in range(N):
        rr = run_yaqs(H, noise_model, L=L, dt=dt, T=T, num_traj=1, seed=seed0 + k)
        samples[k] = float(rr.observables[0].results[-1])
    return samples


# ----------------------------
# kappa estimation by a target standard error
# ----------------------------
def required_trajectories_for_target_se(
    samples: np.ndarray,
    target_se: float,
) -> int:
    """
    Given a pilot set of i.i.d. samples, estimate the number of trajectories N
    needed to reach standard error <= target_se:
        SE = s / sqrt(N)
    """
    if samples.size < 2:
        raise ValueError("Need at least 2 samples to estimate variance.")
    s = float(np.std(samples, ddof=1))
    if s == 0.0:
        return 1
    N_req = int(math.ceil((s / target_se) ** 2))
    return max(1, N_req)


# ----------------------------
# Main: compute alpha and kappa
# ----------------------------
def main() -> None:
    H = build_heisenberg_mpo(L, J, h)

    u1 = noise_u1_bitflip_jump(L, gamma)
    u2 = noise_u2_bitflip_measurement(L, gamma)

    # --- pick a statistical convergence target ---
    # Choose a reasonable small SE for a [-1,1]-bounded observable like XX.
    # If you want to match your paper's exact criterion (e.g., relative error, time-averaged),
    # replace this with that criterion.
    target_se = 0.01  # e.g. 0.025 absolute standard error at final time

    # --- pilot runs to get chi_max and pilot variance ---
    # Use a moderate pilot N to estimate variance
    pilot_N = 200

    rr1 = run_yaqs(H, u1, L=L, dt=dt, T=T, num_traj=pilot_N, seed=1)
    rr2 = run_yaqs(H, u2, L=L, dt=dt, T=T, num_traj=pilot_N, seed=2)

    # chi_max: if not extractable, we still print what we can and mark alpha as None
    chi1 = rr1.chi_max
    chi2 = rr2.chi_max
    alpha = None if (chi1 is None or chi2 is None) else (chi1 / chi2)

    # sampling: get per-trajectory final samples
    samp1 = rr1.traj_samples_final
    samp2 = rr2.traj_samples_final

    if samp1 is None:
        print("[info] Could not extract per-trajectory samples for U1 from YAQS; falling back to num_traj=1 sampling.")
        samp1 = collect_final_time_samples_via_single_traj(H, u1, L=L, dt=dt, T=T, N=pilot_N, seed0=10_000)

    if samp2 is None:
        print("[info] Could not extract per-trajectory samples for U2 from YAQS; falling back to num_traj=1 sampling.")
        samp2 = collect_final_time_samples_via_single_traj(H, u2, L=L, dt=dt, T=T, N=pilot_N, seed0=20_000)

    N1 = required_trajectories_for_target_se(samp1, target_se=target_se)
    N2 = required_trajectories_for_target_se(samp2, target_se=target_se)
    kappa = N2 / N1

    # --- report ---
    print("\n=== Bit-flip (X) noise: alpha/kappa at fixed (gamma, dt, T) ===")
    print(f"L={L}, gamma={gamma}, dt={dt}, T={T}")
    print(f"Observable for kappa: XX on sites [{L//2}, {L//2+1}] at final time")
    print(f"Target standard error (final time): {target_se:.4g}")
    print("")
    print("U1 (jump):   noise = pauli_x @ each site, strength=gamma")
    print("U2 (meas.):  noise = measure_x_0/1 @ each site, strength=2*gamma")
    print("")
    print(f"Pilot N = {pilot_N}")
    print(f"Estimated N_req(U1) = {N1}")
    print(f"Estimated N_req(U2) = {N2}")
    print(f"kappa = N_req(U2) / N_req(U1) = {kappa:.6g}")
    print("")
    if alpha is None:
        print("alpha = (chi_max(U2)/chi_max(U1)) could not be computed automatically.")
        print(f"Extracted chi_max(U1) = {chi1}")
        print(f"Extracted chi_max(U2) = {chi2}")
        print("-> Update the chi_max extractor in run_yaqs() to match your YAQS build.")
    else:
        print(f"chi_max(U1) = {chi1}")
        print(f"chi_max(U2) = {chi2}")
        print(f"alpha = chi_max(U1) / chi_max(U2) = {alpha:.6g}")

    # Optional: print final-time ensemble averages as a sanity check
    print("")
    print(f"Ensemble avg XX_final(U1) = {rr1.obs_time_series[-1]:.6g}")
    print(f"Ensemble avg XX_final(U2) = {rr2.obs_time_series[-1]:.6g}")


if __name__ == "__main__":
    main()
