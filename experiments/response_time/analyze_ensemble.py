
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from experiments.response_time.common import run_ising_jump_simulation, calculate_response_time

def analyze_operators():
    L = 10
    T = 10.0
    dt = 0.1
    J = 1.0
    g = 1.0
    jump_time = 4.0
    gamma = 1e-3

    operators = ["x"]
    results = {op: [] for op in operators}
    
    for op in operators:
        print(f"Analyzing operator: {op}")
        for jump_site in range(L):
            print(f"  Jump site: {jump_site}")
            times, baseline_results, jump_results = run_ising_jump_simulation(
                L, T, dt, J, g, jump_time, jump_site, op, gamma
            )
            r_time = calculate_response_time(times, baseline_results, jump_results, jump_time)
            results[op].append(r_time)
            
    # Plotting
    plt.figure(figsize=(10, 6))
    for op in operators:
        plt.plot(range(L), results[op], marker='o', label=f"Operator: {op}")
        
    plt.title(f"Response Time vs. Jump Site for Different Operators\n(Jump at T={jump_time})")
    plt.xlabel("Jumped Site Index")
    plt.ylabel("Response Time (Simulation time units)")
    plt.xticks(range(L))
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = "response_time_operators.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    analyze_operators()
