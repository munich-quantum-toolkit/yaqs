"""
Compute (alpha, kappa) for Lindblad noise unravelings using YAQS TDVP trajectories.

This script compares "physically equivalent" unravelings of the SAME Lindblad generator
(e.g., Jump vs. Measurement) to compute hardware overhead metrics:

- alpha := chi_max(U2) / chi_max(U1)   (bond-dimension inflation)
- kappa := N_req(U2) / N_req(U1)       (sampling inflation)
  where N_req(U) is the number of trajectories needed to reach a target statistical
  error on a chosen observable at a chosen time.

Target Scenario:
- Heisenberg XXX chain
- Bit-flip noise (X operators)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, List, Any
import math
import numpy as np

from mqt.yaqs.core.data_structures.networks import MPS, MPO
from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import X, Z
from mqt.yaqs.core.data_structures.noise_model import NoiseModel


# ----------------------------
# Configuration & Dataclasses
# ----------------------------

@dataclass
class UnravelingConfig:
    """Configuration for a specific noise unraveling."""
    name: str
    noise_model_factory: Callable[[int, float], NoiseModel]
    description: str = ""


@dataclass
class NoiseScenario:
    """A scenario comparing two unravelings for a specific noise type."""
    name: str
    u1: UnravelingConfig  # Reference (e.g. Jump)
    u2: UnravelingConfig  # Comparison (e.g. Measurement)


@dataclass
class SimulationResult:
    """Results from a single unraveling simulation."""
    name: str
    obs_time_series: np.ndarray          # ensemble average over time
    chi_max: float                       # maximum bond dimension reached
    traj_samples_final: np.ndarray       # per-trajectory samples at final time
    final_error_estimate: float = 0.0    # standard error of the mean at final time


# ----------------------------
# Problem Setup & Defaults
# ----------------------------
DEFAULT_L = 16
DEFAULT_J = 1.0
DEFAULT_H = 1.0
DEFAULT_GAMMA = 0.5
DEFAULT_DT = 0.1
DEFAULT_T = 2.0
DEFAULT_TARGET_SE = 0.001
DEFAULT_PILOT_N = 200


def get_default_observables(L: int) -> List[Observable]:
    """Return default observables: XX on middle bond and max_bond."""
    return [
        Observable(Z(), [L // 2]),
        Observable("max_bond")
    ]


# ----------------------------
# Hamiltonian & Noise Builders
# ----------------------------

def build_heisenberg_mpo(L: int, J: float, h: float) -> MPO:
    """Build the Heisenberg Hamiltonian MPO."""
    return MPO.heisenberg(L, J, J, J, h)
    # return MPO.ising(L, J, h)


def noise_jump_bitflip(L: int, gamma: float) -> NoiseModel:
    """Diffusive/Jump unraveling with collapse operator sqrt(gamma) X_i."""
    return NoiseModel(
        [{"name": "pauli_z", "sites": [i], "strength": gamma} for i in range(L)]
    )


def noise_meas_bitflip(L: int, gamma: float) -> NoiseModel:
    """Measurement unraveling reproducing the same Lindbladian as pauli_x jumps."""
    # Strength = 2*gamma to match Lindblad term 2*gamma * (D[P+] + D[P-]) = gamma * D[X]
    return NoiseModel(
        [{"name": name, "sites": [i], "strength": 2.0 * gamma}
         for i in range(L)
         for name in ["measure_0", "measure_1"]]
    )


# ----------------------------
# Extraction Helpers
# ----------------------------

def extract_chi_max(sim_params: AnalogSimParams) -> float:
    """Extract max bond dimension from simulation results."""
    # Assuming the second observable is always "max_bond" as set in get_default_observables
    # available in sim_params.observables[1].results
    max_bond_obs = sim_params.observables[1]
    if max_bond_obs.results is not None:
        return float(np.max(max_bond_obs.results))
    return 0.0


def extract_traj_samples(sim_params: AnalogSimParams) -> np.ndarray:
    """Extract per-trajectory final-time samples."""
    # trajectories is (num_traj, num_timesteps)
    obs = sim_params.observables[0]
    return np.real(obs.trajectories[:, -1]).astype(float)


# ----------------------------
# Simulation Logic
# ----------------------------

def run_unraveling_pilot(
    H: MPO,
    config: UnravelingConfig,
    base_params: dict[str, Any],
    pilot_N: int,
    seed: int
) -> SimulationResult:
    """Run a pilot simulation for a single unraveling."""
    L = base_params["L"]
    gamma = base_params["gamma"]
    
    # Instantiate noise model
    noise_model = config.noise_model_factory(L, gamma)
    
    # Setup Simulation
    np.random.seed(seed)
    # We use a known product state to ensure consistency
    state = MPS(state="Neel", length=L)
    
    sim_params = AnalogSimParams(
        observables=get_default_observables(L),
        elapsed_time=base_params["T"],
        dt=base_params["dt"],
        num_traj=pilot_N,
        threshold=1e-6,
        trunc_mode="discarded_weight",
        order=2,
        sample_timesteps=False,
    )
    
    simulator.run(state, H, sim_params, noise_model=noise_model)
    print("Obs val", sim_params.observables[0].results)

    # Extraction
    obs_avg = np.array(sim_params.observables[0].results, dtype=float)
    chi_max = extract_chi_max(sim_params)
    samples = extract_traj_samples(sim_params)
    
    # Stats
    std = np.std(samples, ddof=1) if samples.size > 1 else 0.0
    sem = std / np.sqrt(pilot_N) if pilot_N > 0 else 0.0
    
    return SimulationResult(
        name=config.name,
        obs_time_series=obs_avg,
        chi_max=chi_max,
        traj_samples_final=samples,
        final_error_estimate=sem
    )


def calculate_n_req(samples: np.ndarray, target_se: float) -> int:
    """Estimate required trajectories for target standard error."""
    if samples is None or samples.size < 2:
        return 1
    std = np.std(samples, ddof=1)
    if std == 0:
        return 1
    return max(1, int(math.ceil((std / target_se) ** 2)))


# ----------------------------
# Comparison Logic
# ----------------------------

def compare_scenarios(
    scenarios: List[NoiseScenario],
    L: int = DEFAULT_L,
    J: float = DEFAULT_J,
    h: float = DEFAULT_H,
    gamma: float = DEFAULT_GAMMA,
    dt: float = DEFAULT_DT,
    T: float = DEFAULT_T,
    pilot_N: int = DEFAULT_PILOT_N,
    target_se: float = DEFAULT_TARGET_SE
) -> None:
    """Run comparison for a list of noise scenarios."""
    
    print(f"\n=== Noise Scenario Comparison (L={L}, gamma={gamma}, T={T}) ===")
    print(f"Goal: Target SE = {target_se} for primary observable")
    
    H = build_heisenberg_mpo(L, J, h)
    base_params = {"L": L, "J": J, "h": h, "gamma": gamma, "dt": dt, "T": T}
    
    print("\n" + "="*80)
    print(f"{'Scenario':<15} | {'Unraveling':<10} | {'ChiMax':<8} | {'N_req':<8} | {'Alpha (X1/X2)':<15} | {'Kappa (N2/N1)':<15}")
    print("="*80)

    for i, scenario in enumerate(scenarios):
        # Run U1
        res1 = run_unraveling_pilot(H, scenario.u1, base_params, pilot_N, seed=i*100 + 1)
        n1 = calculate_n_req(res1.traj_samples_final, target_se)
        
        # Run U2
        res2 = run_unraveling_pilot(H, scenario.u2, base_params, pilot_N, seed=i*100 + 2)
        n2 = calculate_n_req(res2.traj_samples_final, target_se)
        
        # Calculate Metrics
        # alpha = Chi_1 / Chi_2 (as per user request)
        alpha = res1.chi_max / res2.chi_max if res2.chi_max > 0 else float('inf')
        
        # kappa = N_2 / N_1
        kappa = n2 / n1 if n1 > 0 else float('inf')
        
        # Print Result Row 1 (Scenario + U1 stats)
        print(f"{scenario.name:<15} | {scenario.u1.name:<10} | {res1.chi_max:<8.1f} | {n1:<8} | {'':<15} | {'':<15}")
        
        # Print Result Row 2 (U2 stats + Comparative Metrics)
        print(f"{'':<15} | {scenario.u2.name:<10} | {res2.chi_max:<8.1f} | {n2:<8} | {alpha:<15.3f} | {kappa:<15.3f}")
        print("-" * 80)


# ----------------------------
# Main Execution
# ----------------------------

def main() -> None:
    # Define scenarios to compare
    scenarios = [
        NoiseScenario(
            name="BitFlip",
            u1=UnravelingConfig(
                name="Jump",
                noise_model_factory=noise_jump_bitflip,
                description="Standard Pauli-X jumps"
            ),
            u2=UnravelingConfig(
                name="Meas",
                noise_model_factory=noise_meas_bitflip,
                description="Measurement based X-noise"
            )
        ),
        # Add more scenarios here as needed, e.g.:
        # NoiseScenario(name="Dephasing", u1=..., u2=...)
    ]
    
    compare_scenarios(
        scenarios,
        L=DEFAULT_L,
        T=DEFAULT_T,
        gamma=DEFAULT_GAMMA,
        pilot_N=DEFAULT_PILOT_N,
        target_se=DEFAULT_TARGET_SE
    )

if __name__ == "__main__":
    main()
