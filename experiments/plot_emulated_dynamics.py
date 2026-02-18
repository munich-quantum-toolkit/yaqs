"""
Plot emulated system dynamics for specific noise strengths.

Loads the trained emulator model and predicts the time evolution of Z observables
for gamma = 10^-3, 10^-2, and 10^-1. Plots the results as heatmaps.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from mqt.yaqs.characterization import emulator

def plot_emulated_dynamics():
    # 1. Configuration
    gammas_to_plot = [1e-3, 1e-2, 1e-1]
    model_path = Path(__file__).parent / "models" / "ising_emulator_mlp.pt"
    data_path = Path(__file__).parent / "data" / "ising_lindblad_data.npz"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Please run train_ising_emulator.py first.")
    
    # 2. visual params
    # We need L and T to reshape the output. Let's load them from the data file metadata
    # or hardcode if we are sure (L=10, T=100 from generation script).
    # Safer to load one sample from data to get shape.
    if data_path.exists():
        data = np.load(data_path)
        # observables shape: (N, L, T)
        sample_obs = data["observables"][0]
        L, T = sample_obs.shape
        times = data["times"]
    else:
        # Fallback if data file missing (though we need it for training usually)
        print("Data file not found, assuming L=10, T=100")
        L = 10
        T = 100
        times = np.linspace(0, 10, T)

    # 3. Load Model
    # We need to know input/output dims to instantiate.
    # Input: 1 (gamma)
    # Output: L*T
    input_dim = 1
    output_dim = L * T
    
    model = emulator.Emulator(input_dim, output_dim, hidden_dim=256, num_layers=4)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 4. Predict and Plot
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True, sharey=True)
    
    # Load comparison data
    all_gammas = data["gammas"]
    all_observables = data["observables"]
    
    print(f"Plotting comparison for target gammas: {gammas_to_plot}")
    
    with torch.no_grad():
        for i, target_gamma in enumerate(gammas_to_plot):
            # A. Find closest true data point
            idx = (np.abs(all_gammas - target_gamma)).argmin()
            true_gamma = all_gammas[idx]
            true_dynamics = all_observables[idx] # Shape (L, T)
            
            # B. Emulate
            gamma_tensor = torch.tensor([[target_gamma]], dtype=torch.float32)
            flat_pred = model(gamma_tensor).numpy().flatten()
            pred_dynamics = flat_pred.reshape(L, T)
            
            # C. Plot Emulated (Left Column)
            ax_emu = axes[i, 0]
            im_emu = ax_emu.imshow(
                pred_dynamics, 
                aspect="auto", 
                origin="lower",
                interpolation="nearest",
                extent=[times[0], times[-1], -0.5, L-0.5],
                cmap="viridis",
                vmin=-1, vmax=1
            )
            ax_emu.set_title(f"Emulated ($\gamma={target_gamma}$)")
            ax_emu.set_ylabel("Site Index")
            
            # D. Plot True (Right Column)
            ax_true = axes[i, 1]
            im_true = ax_true.imshow(
                true_dynamics, 
                aspect="auto", 
                origin="lower",
                interpolation="nearest",
                extent=[times[0], times[-1], -0.5, L-0.5],
                cmap="viridis",
                vmin=-1, vmax=1
            )
            ax_true.set_title(f"True Data ($\gamma \\approx {true_gamma:.5f}$)")
            
            if i == 2:
                ax_emu.set_xlabel("Time ($t$)")
                ax_true.set_xlabel("Time ($t$)")

    # Add colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im_true, cax=cbar_ax, label="Expected Z Magnetization $\langle Z_i(t) \\rangle$")
    
    output_png = Path(__file__).parent / "emulated_vs_true_dynamics.png"
    plt.savefig(output_png)
    print(f"Comparison plot saved to {output_png}")
    # plt.show() # Uncomment to show locally if supported

if __name__ == "__main__":
    plot_emulated_dynamics()
