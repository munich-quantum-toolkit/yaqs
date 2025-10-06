#!/usr/bin/env python3
"""
Theoretical variance comparison for 1-qubit Pauli-X noise.

Simulates a simple 1-qubit circuit with Z gate (no effect on |0⟩) and Pauli-X noise,
comparing simulation results with theoretical variance formulas for different unraveling methods.
"""

import sys
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli

# Import YAQS components
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams, Observable
from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs import simulator

from worker_functions.yaqs_simulator import run_yaqs, build_noise_models
from worker_functions.qiskit_simulators import run_qiskit_exact, run_qiskit_mps

# Import Qiskit noise components
from qiskit_aer.noise import NoiseModel as QiskitNoiseModel, PauliLindbladError


def build_simple_circuit() -> QuantumCircuit:
    """Build a simple 1-qubit circuit with identity gates that noise can act on."""
    qc = QuantumCircuit(2)
    qc.rxx(0.001*np.pi/2, 0, 1)
    return qc



# def theoretical_variance_formulas(t_layers: np.ndarray, gamma: float) -> Dict[str, np.ndarray]:
#     """
#     Compute theoretical variance formulas for different unraveling methods.
    
#     For 2-qubit system with CZ gates and Pauli-X noise, the theoretical formulas
#     are approximations since the full 2-qubit case is more complex.
    
#     Args:
#         t_layers: Physical times at each layer (t_ℓ)
#         gamma: Noise rate
        
#     Returns:
#         Dictionary with theoretical variances for each method
#     """
#     # For 2-qubit system with CZ gates and noise, we approximate using
#     # the 1-qubit formulas as a baseline (the actual 2-qubit case is more complex)
    
#     # # Common expectation value: E[X_ℓ] = e^(-2γt_ℓ) (starting from |0⟩ with ⟨Z⟩ = +1)
#     # expectation = np.exp(-2 * gamma * t_layers)
    
#     # # Unitary X jump (standard) - scaled for 2-qubit case
#     # var_standard = 1 - np.exp(-4 * gamma * t_layers)
    
#     # # Projector jumps - scaled for 2-qubit case
#     # var_projector = expectation * (1 - expectation)
    
#     # # Analog Gaussian kicks - scaled for 2-qubit case
#     # var_gaussian = 0.5 + 0.5 * np.exp(-8 * gamma * t_layers) - np.exp(-4 * gamma * t_layers)
    
#     # # Analog two-point (discrete approximation) - scaled for 2-qubit case
#     # var_unitary_2pt = 0.5 + 0.5 * np.exp(-8 * gamma * t_layers) - np.exp(-4 * gamma * t_layers)

#     # Effective flip rate for local Z_i (anticommuting channels): r_i = γ_Xi + γ_XX = 2γ
#     r_i = 2.0 * gamma

#     # Ensemble mean of <Z_i>
#     m = np.exp(-2.0 * r_i * t_layers)  # = exp(-4γ t)

#     # Projector unraveling (absorbing on first anti-commute projector jump; rate 2 r_i = 4γ)
#     var_projector = m * (1.0 - m)  # = e^{-4γ t}(1 - e^{-4γ t})

#     # Standard (unitary) unraveling → telegraph ±1 with flip rate r_i
#     var_standard = 1.0 - np.exp(-4.0 * r_i * t_layers)  # = 1 - e^{-8γ t}

#     # # Analog 2-pt / Gaussian: stationary variance = 1/4 for this symmetric 2q case.
#     # var_plateau = 0.25 * np.ones_like(t_layers)

#     # Analog Gaussian kicks - scaled for 2-qubit case
#     var_gaussian = 0.25 * np.ones_like(t_layers)
    
#     # Analog two-point (discrete approximation) - scaled for 2-qubit case
#     var_unitary_2pt = 0.25 * np.ones_like(t_layers)
    
#     return {
#         "standard": var_standard,
#         "projector": var_projector,
#         "unitary_gauss": var_gaussian,
#         "unitary_2pt": var_unitary_2pt,
#     }

def theoretical_variance_formulas(t_layers: np.ndarray, gamma: float,
                                  *, scheme_params: dict[str, float] | None = None) -> dict[str, np.ndarray]:
    """
    Exact two-qubit IX/XI/XX identity-layer variance curves for <Z_i>.

    scheme_params:
      - for 2pt:  {"s": s}  with s = sin^2(theta0) used in your analog init
      - for gauss:{"s": s}  with s = (1 - exp(-2 sigma^2))/2 used in your analog init
      If omitted, we assume hazard_gain=g=3 and no cap -> s = 1/g = 1/3 for both analog laws.
    """
    if scheme_params is None:
        s_analog = 1.0 / 3.0  # matches hazard_gain=3, cap=0
    else:
        s_analog = float(scheme_params.get("s", 1.0/3.0))
    t = np.asarray(t_layers, dtype=float)

    # mean is scheme-independent
    mean_sq = np.exp(-8.0 * gamma * t)

    # standard (telegraph with flip-rate 2γ)
    var_std = 1.0 - mean_sq

    # projector (absorbing with rate 4γ)
    var_proj = np.exp(-4.0 * gamma * t) * (1.0 - np.exp(-4.0 * gamma * t))

    # analog 2-point: Var = 1/4 + 1/2 e^{-8γ(1-s)t} + 1/4 e^{-16γ(1-s)t} - e^{-8γ t}
    s = s_analog
    var_2pt = 0.25 + 0.5*np.exp(-8.0*gamma*(1.0 - s)*t) + 0.25*np.exp(-16.0*gamma*(1.0 - s)*t) - mean_sq

    # analog Gaussian:
    # b = (γ/s) * [1 - (1 - 2s)^4]/2 = γ * (1 + e^{-2σ^2} + e^{-4σ^2} + e^{-6σ^2})
    one_minus_2s = 1.0 - 2.0*s
    B = (1.0 - one_minus_2s**4) / 2.0
    b = (gamma / max(s, 1e-16)) * B
    var_gauss = 0.25 + 0.5*np.exp(-2.0*b*t) + 0.25*np.exp(-4.0*b*t) - mean_sq

    return {
        "standard": var_std,
        "projector": var_proj,
        "unitary_2pt": var_2pt,
        "unitary_gauss": var_gauss,
    }


def plot_comparison(
    t_layers: np.ndarray,
    theoretical_vars: Dict[str, np.ndarray],
    simulation_vars: Dict[str, np.ndarray],
    simulation_expectations: Dict[str, np.ndarray],
    gamma: float,
) -> None:
    """Plot theoretical vs simulation variances for 2-qubit system."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Variance comparison (Qubit 0)
    ax1.set_title(f"Variance Comparison - Qubit 0 (γ={gamma})")
    ax1.set_xlabel("Physical Time (t)")
    ax1.set_ylabel("Variance")
    ax1.grid(True, alpha=0.3)
    
    colors = ["blue", "red", "green", "orange", "purple", "brown"]
    
    for i, (method, theoretical_var) in enumerate(theoretical_vars.items()):
        color = colors[i % len(colors)]
        
        # Theoretical
        ax1.plot(t_layers, theoretical_var, "--", color=color, 
                label=f"{method} (theoretical)", linewidth=2)
        
        # Simulation
        if method in simulation_vars:
            sim_var_data = simulation_vars[method]
            # Handle 2D arrays (qubits × time) - show qubit 0
            if hasattr(sim_var_data, 'shape') and len(sim_var_data.shape) == 2:
                sim_var_1d = sim_var_data[0, :]  # Qubit 0
            else:
                sim_var_1d = sim_var_data
            
            # Adjust t_layers to match the length of sim_var_1d
            if len(sim_var_1d) == len(t_layers) - 1:
                t_plot = t_layers[1:]
            else:
                t_plot = t_layers
                
            ax1.plot(t_plot, sim_var_1d, "-", color=color,
                    label=f"{method} (simulation)", linewidth=1.5, alpha=0.8)
    
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    
    # Plot 2: Variance comparison (Qubit 1)
    ax2.set_title(f"Variance Comparison - Qubit 1 (γ={gamma})")
    ax2.set_xlabel("Physical Time (t)")
    ax2.set_ylabel("Variance")
    ax2.grid(True, alpha=0.3)
    
    for i, (method, theoretical_var) in enumerate(theoretical_vars.items()):
        color = colors[i % len(colors)]
        
        # Theoretical
        ax2.plot(t_layers, theoretical_var, "--", color=color, 
                label=f"{method} (theoretical)", linewidth=2)
        
        # Simulation
        if method in simulation_vars:
            sim_var_data = simulation_vars[method]
            # Handle 2D arrays (qubits × time) - show qubit 1
            if hasattr(sim_var_data, 'shape') and len(sim_var_data.shape) == 2:
                sim_var_1d = sim_var_data[1, :]  # Qubit 1
            else:
                sim_var_1d = sim_var_data
            
            # Adjust t_layers to match the length of sim_var_1d
            if len(sim_var_1d) == len(t_layers) - 1:
                t_plot = t_layers[1:]
            else:
                t_plot = t_layers
                
            ax2.plot(t_plot, sim_var_1d, "-", color=color,
                    label=f"{method} (simulation)", linewidth=1.5, alpha=0.8)
    
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    
    # Plot 3: Expectation value comparison (Qubit 0)
    ax3.set_title(f"Expectation Value Comparison - Qubit 0 (γ={gamma})")
    ax3.set_xlabel("Physical Time (t)")
    ax3.set_ylabel("⟨Z⟩")
    ax3.grid(True, alpha=0.3)
    
    # Use exact simulation as theoretical reference
    if "exact" in simulation_expectations:
        exact_exp = simulation_expectations["exact"]
        if hasattr(exact_exp, 'shape') and len(exact_exp.shape) == 2:
            exact_exp_0 = exact_exp[0, :]  # Qubit 0
        else:
            exact_exp_0 = exact_exp
        
        # Add initial state (t=0) with expectation value 1.0 for |0⟩ state
        if len(exact_exp_0) == len(t_layers) - 1:
            exact_exp_with_init = np.concatenate([[1.0], exact_exp_0])
        else:
            exact_exp_with_init = exact_exp_0
            
        ax3.plot(t_layers, exact_exp_with_init, "--", color="black", 
                label="exact (reference)", linewidth=2)
    
    # Simulation expectations
    for i, (method, exp_vals) in enumerate(simulation_expectations.items()):
        if method == "exact":  # Skip exact as it's already plotted as reference
            continue
        color = colors[i % len(colors)]
        # Handle 2D arrays (qubits × time) - show qubit 0
        if hasattr(exp_vals, 'shape') and len(exp_vals.shape) == 2:
            exp_vals_1d = exp_vals[0, :]  # Qubit 0
        else:
            exp_vals_1d = exp_vals
        
        # Adjust t_layers to match the length of exp_vals_1d
        if len(exp_vals_1d) == len(t_layers) - 1:
            # Add initial state (t=0) with expectation value 1.0 for |0⟩ state
            exp_vals_with_init = np.concatenate([[1.0], exp_vals_1d])
            t_plot = t_layers
        else:
            exp_vals_with_init = exp_vals_1d
            t_plot = t_layers
            
        ax3.plot(t_plot, exp_vals_with_init, "-", color=color,
                label=f"{method}", linewidth=1.5, alpha=0.8)
    
    ax3.legend()
    ax3.set_ylim(0, 1.1)
    
    # Plot 4: Expectation value comparison (Qubit 1)
    ax4.set_title(f"Expectation Value Comparison - Qubit 1 (γ={gamma})")
    ax4.set_xlabel("Physical Time (t)")
    ax4.set_ylabel("⟨Z⟩")
    ax4.grid(True, alpha=0.3)
    
    # Use exact simulation as theoretical reference
    if "exact" in simulation_expectations:
        exact_exp = simulation_expectations["exact"]
        if hasattr(exact_exp, 'shape') and len(exact_exp.shape) == 2:
            exact_exp_1 = exact_exp[1, :]  # Qubit 1
        else:
            exact_exp_1 = exact_exp
        
        # Add initial state (t=0) with expectation value 1.0 for |0⟩ state
        if len(exact_exp_1) == len(t_layers) - 1:
            exact_exp_with_init = np.concatenate([[1.0], exact_exp_1])
        else:
            exact_exp_with_init = exact_exp_1
            
        ax4.plot(t_layers, exact_exp_with_init, "--", color="black", 
                label="exact (reference)", linewidth=2)
    
    # Simulation expectations
    for i, (method, exp_vals) in enumerate(simulation_expectations.items()):
        if method == "exact":  # Skip exact as it's already plotted as reference
            continue
        color = colors[i % len(colors)]
        # Handle 2D arrays (qubits × time) - show qubit 1
        if hasattr(exp_vals, 'shape') and len(exp_vals.shape) == 2:
            exp_vals_1d = exp_vals[1, :]  # Qubit 1
        else:
            exp_vals_1d = exp_vals
        
        # Adjust t_layers to match the length of exp_vals_1d
        if len(exp_vals_1d) == len(t_layers) - 1:
            # Add initial state (t=0) with expectation value 1.0 for |0⟩ state
            exp_vals_with_init = np.concatenate([[1.0], exp_vals_1d])
            t_plot = t_layers
        else:
            exp_vals_with_init = exp_vals_1d
            t_plot = t_layers
            
        ax4.plot(t_plot, exp_vals_with_init, "-", color=color,
                label=f"{method}", linewidth=1.5, alpha=0.8)
    
    ax4.legend()
    ax4.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.show()


def print_comparison_table(
    t_layers: np.ndarray,
    theoretical_vars: Dict[str, np.ndarray],
    simulation_vars: Dict[str, np.ndarray],
    gamma: float,
) -> None:
    """Print a comparison table of theoretical vs simulation variances for 2-qubit system."""
    print(f"\nVariance Comparison (γ={gamma}) - 2-Qubit System")
    print("=" * 100)
    print(f"{'Time':<8} {'Qubit':<6} {'Method':<12} {'Theoretical':<12} {'Simulation':<12} {'Error':<12}")
    print("-" * 100)
    
    # Sample a few time points
    time_indices = np.linspace(0, len(t_layers)-1, min(5, len(t_layers)), dtype=int)
    
    for t_idx in time_indices:
        t_val = t_layers[t_idx]
        print(f"{t_val:<8.2f}")
        
        for qubit in [0, 1]:
            for method in theoretical_vars.keys():
                theo_var = float(theoretical_vars[method][t_idx])
                sim_var_array = simulation_vars.get(method, np.zeros_like(t_layers))
                
                # Handle 2D arrays (qubits × time) - show specific qubit
                if hasattr(sim_var_array, 'shape') and len(sim_var_array.shape) == 2:
                    sim_var = float(sim_var_array[qubit, t_idx])
                elif hasattr(sim_var_array, 'shape') and len(sim_var_array.shape) == 1:
                    sim_var = float(sim_var_array[t_idx])
                else:
                    sim_var = float(sim_var_array)
                
                error = abs(theo_var - sim_var)
                
                print(f"{'':8} {qubit:<6} {method:<12} {theo_var:<12.6f} {sim_var:<12.6f} {error:<12.6f}")
        print()


if __name__ == "__main__":
    # Parameters
    L = 2
    gamma = 0.1  # Noise rate
    num_layers = 70
    num_traj = 1000
    dt = 1.0  # Time step per layer
    
    # Physical times at each layer
    t_layers = np.arange(num_layers + 1) * dt
    
    print(f"Running theoretical variance comparison...")
    print(f"Parameters: γ={gamma}, num_layers={num_layers}, num_traj={num_traj}")
    print(f"Physical times: {t_layers}")
    
    # Build circuit and noise models
    basis_circuit = QuantumCircuit(L)
    basis_circuit.rxx(0.0, 0, 1)


    processes = [{"name": "pauli_x", "sites": [0], "strength": gamma}] + [{"name": "crosstalk_xx", "sites": [0, 1], "strength": gamma}] + [{"name": "pauli_x", "sites": [1], "strength": gamma}] 
    yaqs_noise_models = build_noise_models(processes)
    noise_model = QiskitNoiseModel()
    # Pauli-X error with rate gamma
    x_error = PauliLindbladError([Pauli("XX"), Pauli("IX"), Pauli("XI")], [gamma, gamma, gamma])
    noise_model.add_all_qubit_quantum_error(x_error, ["rxx"]) 
    
    # Compute theoretical variances
    theoretical_vars = theoretical_variance_formulas(t_layers, gamma)
    
    # Run simulations
    simulation_vars = {}
    simulation_expectations = {}
    
    print("\nRunning YAQS simulations...")
    method_names = ["standard", "projector", "unitary_2pt", "unitary_gauss"]
    for i, yaqs_noise_model in enumerate(yaqs_noise_models):
        method = method_names[i]
        print(f"  {method}...")
        exp_vals, _ , vars = run_yaqs(basis_circuit, L, num_layers, yaqs_noise_model, num_traj=num_traj)
        simulation_vars[method] = vars
        simulation_expectations[method] = exp_vals
    
    print("Running Qiskit simulations...")
    # Run exact density matrix simulation as theoretical reference
    exp_vals_exact, _ = run_qiskit_exact(L, num_layers, basis_circuit, noise_model)
    simulation_expectations["exact"] = exp_vals_exact
    
    # Run MPS simulation
    exp_vals, _ , vars = run_qiskit_mps(L, num_layers, basis_circuit, noise_model, num_traj=num_traj)
    simulation_vars["qiskit"] = vars
    simulation_expectations["qiskit_mps"] = exp_vals
    
    # Print comparison table
    print_comparison_table(t_layers, theoretical_vars, simulation_vars, gamma)
    
    # Plot results
    plot_comparison(t_layers, theoretical_vars, simulation_vars, simulation_expectations, gamma)
    
    print("Comparison complete!")


