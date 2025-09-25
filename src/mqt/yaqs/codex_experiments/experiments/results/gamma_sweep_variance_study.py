#!/usr/bin/env python3
"""
Gamma Sweep Variance Study for Nature Physics Publication

This script runs the theoretical variance comparison experiment across three different
gamma values (0.1, 0.01, 0.001) and creates publication-quality plots showing both
variance and mean expectation values for all simulation methods.

Author: Generated for MQT-YAQS project
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, List, Tuple
import sys
import os
import pickle
from datetime import datetime

# Add current directory to path for imports
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from worker_functions.qiskit_simulators import run_qiskit_exact, run_qiskit_mps
from worker_functions.yaqs_simulator import run_yaqs, build_noise_models
from worker_functions.circuits import build_basis_noncommuting
from theoretical_variance_comparison import theoretical_variance_formulas
from qiskit_aer.noise import NoiseModel as QiskitNoiseModel, PauliLindbladError
from qiskit.quantum_info import Pauli

def build_qiskit_noise_model(gamma: float) -> QiskitNoiseModel:
    """Build Qiskit noise model for the 2-qubit system."""
    noise_model = QiskitNoiseModel()
    # Pauli-X error with rate gamma on both qubits and crosstalk
    x_error = PauliLindbladError([Pauli("XX"), Pauli("IX"), Pauli("XI")], [gamma, gamma, gamma])
    noise_model.add_all_qubit_quantum_error(x_error, ["rxx"]) 
    return noise_model

# Use basic matplotlib styling
plt.style.use('default')

def run_gamma_sweep_experiment(gamma_values: List[float], L: int = 2, 
                              num_layers: int = 70, num_traj: int = 1000) -> Dict:
    """
    Run the variance experiment across multiple gamma values.
    
    Args:
        gamma_values: List of gamma values to test
        L: Number of qubits
        num_layers: Number of circuit layers
        num_traj: Number of trajectories for stochastic methods
        
    Returns:
        Dictionary containing results for each gamma value
    """
    results = {}
    
    for gamma in gamma_values:
        print(f"\n{'='*60}")
        print(f"Running experiment for γ = {gamma}")
        print(f"{'='*60}")
        
        # Build circuit and noise models
        from qiskit import QuantumCircuit
        basis_circuit = QuantumCircuit(L)
        basis_circuit.rxx(0.0, 0, 1)  # Identity circuit with RXX gate
        
        processes = [
            {"name": "pauli_x", "sites": [0], "strength": gamma},
            {"name": "crosstalk_xx", "sites": [0, 1], "strength": gamma},
            {"name": "pauli_x", "sites": [1], "strength": gamma}
        ]
        yaqs_noise_models = build_noise_models(processes)
        qiskit_noise_model = build_qiskit_noise_model(gamma)
        
        # Physical times
        t_layers = np.arange(num_layers + 1, dtype=float)
        
        # Storage for results
        simulation_vars = {}
        simulation_expectations = {}
        
        # Run YAQS simulations
        print("Running YAQS simulations...")
        method_names = ["standard", "projector", "unitary_2pt", "unitary_gauss"]
        for i, (method, yaqs_noise_model) in enumerate(zip(method_names, yaqs_noise_models)):
            print(f"  {method}...")
            exp_vals, _, vars = run_yaqs(basis_circuit, L, num_layers, yaqs_noise_model, num_traj=num_traj)
            simulation_vars[method] = vars
            simulation_expectations[method] = exp_vals
        
        # Run Qiskit simulations
        print("Running Qiskit simulations...")
        # Exact simulation as reference
        exp_vals_exact, _ = run_qiskit_exact(L, num_layers, basis_circuit, qiskit_noise_model)
        simulation_expectations["exact"] = exp_vals_exact
        
        # MPS simulation
        exp_vals, _, vars = run_qiskit_mps(L, num_layers, basis_circuit, qiskit_noise_model, num_traj=num_traj)
        simulation_vars["qiskit"] = vars
        simulation_expectations["qiskit_mps"] = exp_vals
        
        # Compute theoretical variances
        theoretical_vars = theoretical_variance_formulas(t_layers, gamma)
        
        # Store results
        results[gamma] = {
            't_layers': t_layers,
            'simulation_vars': simulation_vars,
            'simulation_expectations': simulation_expectations,
            'theoretical_vars': theoretical_vars
        }
        
        print(f"Completed γ = {gamma}")
    
    return results

def create_publication_plots(results: Dict, gamma_values: List[float], 
                           save_path: str = "gamma_sweep_variance_study.pdf") -> None:
    """
    Create basic plots for the gamma sweep experiment.
    
    Args:
        results: Results dictionary from run_gamma_sweep_experiment
        gamma_values: List of gamma values tested
        save_path: Path to save the plot
    """
    # Define colors and styles for different methods
    method_colors = {
        'exact': 'black',
        'standard': 'blue',
        'projector': 'red', 
        'unitary_2pt': 'green',
        'unitary_gauss': 'orange',
        'qiskit_mps': 'purple'
    }
    
    method_styles = {
        'exact': '--',
        'standard': '-',
        'projector': '-',
        'unitary_2pt': '-',
        'unitary_gauss': '-',
        'qiskit_mps': '-'
    }
    
    method_labels = {
        'exact': 'Exact',
        'standard': 'Standard',
        'projector': 'Projector',
        'unitary_2pt': 'Unitary 2-pt',
        'unitary_gauss': 'Unitary Gauss',
        'qiskit_mps': 'Qiskit MPS'
    }
    
    # Create figure with 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot variance (top row)
    for col, gamma in enumerate(gamma_values):
        ax = axes[0, col]
        data = results[gamma]
        t_layers = data['t_layers']
        simulation_vars = data['simulation_vars']
        theoretical_vars = data['theoretical_vars']
        
        # Plot theoretical variances
        for method, theoretical_var in theoretical_vars.items():
            if method in method_colors:
                ax.plot(t_layers, theoretical_var, '--', 
                       color=method_colors[method], linewidth=2, alpha=0.7,
                       label=f"Theoretical {method_labels[method]}")
        
        # Plot simulation variances (Qubit 1 only)
        for method, sim_vars in simulation_vars.items():
            # Map 'qiskit' to 'qiskit_mps' for consistency
            plot_method = 'qiskit_mps' if method == 'qiskit' else method
            if plot_method in method_colors:
                # Handle 2D arrays (qubits × time) - show qubit 1
                if hasattr(sim_vars, 'shape') and len(sim_vars.shape) == 2:
                    sim_vars_1d = sim_vars[1, :]  # Qubit 1
                else:
                    sim_vars_1d = sim_vars
                
                ax.plot(t_layers, sim_vars_1d, method_styles[plot_method], 
                       color=method_colors[plot_method], linewidth=2.5, alpha=0.9,
                       label=f"Simulation {method_labels[plot_method]}")
        
        ax.set_title(f'γ = {gamma}')
        ax.set_xlabel('Physical Time (t)')
        if col == 0:
            ax.set_ylabel('Variance')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        # Add legend only to the first subplot
        if col == 0:
            ax.legend(loc='upper right')
    
    # Plot mean expectation values (bottom row)
    for col, gamma in enumerate(gamma_values):
        ax = axes[1, col]
        data = results[gamma]
        t_layers = data['t_layers']
        simulation_expectations = data['simulation_expectations']
        
        # Plot exact simulation as reference
        if 'exact' in simulation_expectations:
            exact_exp = simulation_expectations['exact']
            if hasattr(exact_exp, 'shape') and len(exact_exp.shape) == 2:
                exact_exp_1 = exact_exp[1, :]  # Qubit 1
            else:
                exact_exp_1 = exact_exp
            
            # Add initial state (t=0) with expectation value 1.0 for |0⟩ state
            if len(exact_exp_1) == len(t_layers) - 1:
                exact_exp_with_init = np.concatenate([[1.0], exact_exp_1])
            else:
                exact_exp_with_init = exact_exp_1
                
            ax.plot(t_layers, exact_exp_with_init, method_styles['exact'], 
                   color=method_colors['exact'], linewidth=3, alpha=0.8,
                   label=method_labels['exact'])
        
        # Plot simulation expectations (Qubit 1 only)
        for method, exp_vals in simulation_expectations.items():
            if method == 'exact':  # Skip exact as it's already plotted
                continue
            if method in method_colors:
                # Handle 2D arrays (qubits × time) - show qubit 1
                if hasattr(exp_vals, 'shape') and len(exp_vals.shape) == 2:
                    exp_vals_1d = exp_vals[1, :]  # Qubit 1
                else:
                    exp_vals_1d = exp_vals
                
                # Add initial state (t=0) with expectation value 1.0 for |0⟩ state
                if len(exp_vals_1d) == len(t_layers) - 1:
                    exp_vals_with_init = np.concatenate([[1.0], exp_vals_1d])
                else:
                    exp_vals_with_init = exp_vals_1d
                    
                ax.plot(t_layers, exp_vals_with_init, method_styles[method], 
                       color=method_colors[method], linewidth=2.5, alpha=0.9,
                       label=method_labels[method])
        
        ax.set_title(f'γ = {gamma}')
        ax.set_xlabel('Physical Time (t)')
        if col == 0:
            ax.set_ylabel('⟨Z₁⟩')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        # Add legend only to the first subplot
        if col == 0:
            ax.legend(loc='upper right')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the plot
    #plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    # print(f"Plot saved as {save_path}")
    
    # Also save as PNG for quick viewing
    png_path = save_path.replace('.pdf', '.png')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f"Plot also saved as {png_path}")
    
    plt.show()

def save_results_to_pickle(results: Dict, gamma_values: List[float], 
                          filename: str = None) -> str:
    """
    Save the simulation results to a pickle file.
    
    Args:
        results: Results dictionary from run_gamma_sweep_experiment
        gamma_values: List of gamma values tested
        filename: Optional filename. If None, generates timestamped filename
        
    Returns:
        The filename used for saving
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gamma_sweep_results_{timestamp}.pkl"
    
    # Create metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'gamma_values': gamma_values,
        'num_traj': results[gamma_values[0]]['simulation_vars']['standard'].shape[0] if 'standard' in results[gamma_values[0]]['simulation_vars'] else 0,
        'num_layers': len(results[gamma_values[0]]['t_layers']) - 1,
        'num_qubits': 2
    }
    
    # Combine results and metadata
    save_data = {
        'metadata': metadata,
        'results': results
    }
    
    # Save to pickle file
    with open(filename, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"Results saved to {filename}")
    return filename

def load_results_from_pickle(filename: str) -> Tuple[Dict, List[float], Dict]:
    """
    Load simulation results from a pickle file.
    
    Args:
        filename: Path to the pickle file
        
    Returns:
        Tuple of (results, gamma_values, metadata)
    """
    with open(filename, 'rb') as f:
        save_data = pickle.load(f)
    
    metadata = save_data['metadata']
    results = save_data['results']
    gamma_values = metadata['gamma_values']
    
    print(f"Results loaded from {filename}")
    print(f"  Timestamp: {metadata['timestamp']}")
    print(f"  Gamma values: {gamma_values}")
    print(f"  Trajectories: {metadata['num_traj']}")
    print(f"  Layers: {metadata['num_layers']}")
    print(f"  Qubits: {metadata['num_qubits']}")
    
    return results, gamma_values, metadata

def print_summary_statistics(results: Dict, gamma_values: List[float]) -> None:
    """
    Print summary statistics for the gamma sweep experiment.
    
    Args:
        results: Results dictionary from run_gamma_sweep_experiment
        gamma_values: List of gamma values tested
    """
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for gamma in gamma_values:
        print(f"\nγ = {gamma}:")
        data = results[gamma]
        simulation_vars = data['simulation_vars']
        simulation_expectations = data['simulation_expectations']
        
        print("  Final Variance (t=70) - Qubit 1:")
        for method, sim_vars in simulation_vars.items():
            if hasattr(sim_vars, 'shape') and len(sim_vars.shape) == 2:
                final_var = sim_vars[1, -1]  # Qubit 1, final time
            else:
                final_var = sim_vars[-1]
            print(f"    {method:15s}: {final_var:.6f}")
        
        print("\n  Final Approximation Error (t=70) - Qubit 1:")
        if 'exact' in simulation_expectations:
            exact_exp = simulation_expectations['exact']
            if hasattr(exact_exp, 'shape') and len(exact_exp.shape) == 2:
                exact_final = exact_exp[1, -1]  # Qubit 1, final time
            else:
                exact_final = exact_exp[-1]
            
            for method, exp_vals in simulation_expectations.items():
                if method == 'exact':
                    continue
                if hasattr(exp_vals, 'shape') and len(exp_vals.shape) == 2:
                    sim_final = exp_vals[1, -1]  # Qubit 1, final time
                else:
                    sim_final = exp_vals[-1]
                
                error = abs(sim_final - exact_final)
                print(f"    {method:15s}: {error:.6f}")
        
        print("\n  Mean Squared Error (MSE) - Qubit 1:")
        if 'exact' in simulation_expectations:
            exact_exp = simulation_expectations['exact']
            if hasattr(exact_exp, 'shape') and len(exact_exp.shape) == 2:
                exact_vals = exact_exp[1, :]  # Qubit 1, all times
            else:
                exact_vals = exact_exp
            
            for method, exp_vals in simulation_expectations.items():
                if method == 'exact':
                    continue
                if hasattr(exp_vals, 'shape') and len(exp_vals.shape) == 2:
                    sim_vals = exp_vals[1, :]  # Qubit 1, all times
                else:
                    sim_vals = exp_vals
                
                # Ensure same length
                min_len = min(len(exact_vals), len(sim_vals))
                mse = np.mean((exact_vals[:min_len] - sim_vals[:min_len])**2)
                print(f"    {method:15s}: {mse:.6f}")

def main():
    """Main function to run the gamma sweep variance study."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Gamma Sweep Variance Study for Nature Physics')
    parser.add_argument('--load', type=str, help='Load results from pickle file instead of running new simulation')
    parser.add_argument('--save-only', action='store_true', help='Only save results, skip plotting')
    args = parser.parse_args()
    
    print("Gamma Sweep Variance Study for Nature Physics")
    print("=" * 50)
    
    if args.load:
        # Load existing results
        print(f"Loading results from {args.load}")
        results, gamma_values, metadata = load_results_from_pickle(args.load)
    else:
        # Run new experiment
        gamma_values = [0.1, 0.01, 0.001]
        L = 2
        num_layers = 70
        num_traj = 1000
        
        print(f"Parameters:")
        print(f"  Gamma values: {gamma_values}")
        print(f"  Qubits: {L}")
        print(f"  Layers: {num_layers}")
        print(f"  Trajectories: {num_traj}")
        
        # Run experiment
        results = run_gamma_sweep_experiment(gamma_values, L, num_layers, num_traj)
        
        # Save results to pickle file
        pickle_filename = save_results_to_pickle(results, gamma_values)
        print(f"Data saved to: {pickle_filename}")
    
    if not args.save_only:
        # Create plots
        create_publication_plots(results, gamma_values)
        
        # Print summary
        print_summary_statistics(results, gamma_values)
    
    print(f"\nExperiment completed successfully!")
    if not args.load:
        print(f"To reload data later, use:")
        print(f"  python3 gamma_sweep_variance_study.py --load {pickle_filename}")

if __name__ == "__main__":
    main()
