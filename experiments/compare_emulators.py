"""
Compare emulator performance across Lindblad, MCWF, and TJM datasets and generate heatmaps.
"""

import numpy as np
import torch
from pathlib import Path
import logging
import argparse
import matplotlib.pyplot as plt
from mqt.yaqs.characterization import emulator

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("compare_emulators")

def generate_heatmaps(model, data_path, output_path, method):
    """Generate True vs Predicted heatmaps for a representative gamma."""
    data = np.load(data_path)
    X_raw = data["observables"] # (N, L, T)
    gammas = data["gammas"]     # (N,)
    times = data["times"]
    N, L, T = X_raw.shape
    
    # Select a middle gamma (e.g., around 10^-2)
    target_gamma = 10**-2
    idx = (np.abs(gammas - target_gamma)).argmin()
    actual_gamma = gammas[idx]
    
    # Prepare input
    gamma_tensor = torch.tensor([[actual_gamma]], dtype=torch.float32)
    
    # Predict
    model.eval()
    with torch.no_grad():
        pred_flat = model(gamma_tensor).squeeze().numpy()
        pred_dynamics = pred_flat.reshape(L, T)
        
    true_dynamics = X_raw[idx]
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Common vmin/vmax for comparison
    vmin = min(true_dynamics.min(), pred_dynamics.min())
    vmax = max(true_dynamics.max(), pred_dynamics.max())
    
    im0 = axes[0].imshow(true_dynamics, aspect='auto', origin='lower', 
                         extent=[times[0], times[-1], 0, L-1], vmin=vmin, vmax=vmax)
    axes[0].set_title(f"True Dynamics ({method.upper()})\nGamma = {actual_gamma:.5f}")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Site Index")
    fig.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(pred_dynamics, aspect='auto', origin='lower',
                         extent=[times[0], times[-1], 0, L-1], vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Predicted Dynamics ({method.upper()})\nGamma = {actual_gamma:.5f}")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Site Index")
    fig.colorbar(im1, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    logger.info(f"Heatmap saved to {output_path}")

def run_comparison(epochs: int = 100, batch_size: int = 32):
    data_dir = Path(__file__).parent / "data"
    methods = ["lindblad", "mcwf", "tjm"]
    results_mse = {}

    for method in methods:
        data_path = data_dir / f"ising_{method}_data.npz"
        if not data_path.exists():
            logger.warning(f"Data for {method} not found at {data_path}. Skipping.")
            continue
            
        logger.info(f"=== Training Emulator for {method.upper()} ===")
        
        # Train
        model = emulator.emulate(
            training_data=data_path,
            epochs=epochs,
            batch_size=batch_size,
            lr=1e-3,
            model_class=emulator.Emulator,
            model_kwargs={"hidden_dim": 256, "num_layers": 4}
        )
        
        # Evaluate Final MSE on test set (characterization/emulator internal logs MSE during train, but let's re-verify)
        # Actually emulator.emulate logs Test MSE. We can catch it or just recalculate.
        data = np.load(data_path)
        X_raw = data["observables"]
        gammas = data["gammas"]
        N = X_raw.shape[0]
        
        gamma_tensor = torch.tensor(gammas.reshape(N, 1), dtype=torch.float32)
        true_tensor = torch.tensor(X_raw.reshape(N, -1), dtype=torch.float32)
        
        model.eval()
        with torch.no_grad():
            preds = model(gamma_tensor)
            mse = torch.mean((preds - true_tensor)**2).item()
            
        results_mae = torch.mean(torch.abs(preds - true_tensor)).item()
        results_mse[method] = mse
        logger.info(f"Method: {method}, MSE: {mse:.8f}, MAE: {results_mae:.6f}")
        
        # Generate and save heatmap for this method
        heatmap_path = Path(__file__).parent / f"emulator_heatmap_{method}.png"
        generate_heatmaps(model, data_path, heatmap_path, method)
        
        # Save Model
        models_dir = Path(__file__).parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"ising_emulator_{method}_mlp.pt"
        torch.save(model.state_dict(), model_path)

    # Summary
    logger.info("\n" + "="*40)
    logger.info("      EMULATOR SUMMARY RESULTS")
    logger.info("="*40)
    for method, mse in results_mse.items():
        logger.info(f"{method:10}: MSE = {mse:.8f}")
    logger.info("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    
    run_comparison(epochs=args.epochs, batch_size=args.batch_size)
