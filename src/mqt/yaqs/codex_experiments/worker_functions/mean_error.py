import numpy as np

from typing import Dict, Any

def compute_mean_error_vs_exact(yaqs: np.ndarray, exact: np.ndarray) -> float:
    """Mean absolute error across all qubits and layers."""
    return float(np.mean(np.abs(yaqs - exact)))



def print_mean_errors_against_exact(exact: np.ndarray, series_by_label: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute and print mean absolute error of each series vs exact baseline."""
    mean_errors: dict[str, float] = {}
    print("=== Mean absolute error vs exact (lower is better) ===")
    for label, arr in series_by_label.items():
        mae = compute_mean_error_vs_exact(arr, exact)
        mean_errors[label] = mae
        print(f"{label:>12}: {mae:.6e}")
    return mean_errors
