"""
Compare characterizer performance across Lindblad, MCWF, and TJM datasets.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import logging
import argparse
import matplotlib.pyplot as plt
from mqt.yaqs.characterization import characterizer

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("compare_methods")

def evaluate_model(model, X_raw, y_raw, model_class):
    """Manually evaluate model on provided raw data."""
    N, L, T = X_raw.shape
    model.eval()
    
    # Reshape input to match model expectations
    if model_class == characterizer.SimpleCNN:
        X_tensor = torch.tensor(X_raw.reshape(N, 1, L, T), dtype=torch.float32)
    elif model_class == characterizer.SimpleRNN:
        X_tensor = torch.tensor(X_raw.transpose(0, 2, 1), dtype=torch.float32)
    else:
        X_tensor = torch.tensor(X_raw.reshape(N, -1), dtype=torch.float32)
        
    y_tensor = torch.tensor(y_raw.reshape(N, 1), dtype=torch.float32)
    
    with torch.no_grad():
        preds = model(X_tensor)
        mae = torch.mean(torch.abs(preds - y_tensor)).item()
        preds_np = preds.numpy().flatten()
    
    return mae, preds_np

def plot_all_results(all_true, all_pred, output_path):
    """Generate a 1x3 grid comparing true vs predicted gammas."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    methods = list(all_true.keys())
    
    for i, method in enumerate(methods):
        ax = axes[i]
        true = all_true[method]
        pred = all_pred[method]
        
        ax.scatter(true, pred, alpha=0.4, s=15, edgecolors='none', color='tab:blue')
        
        # Perfect prediction line
        min_val = min(true.min(), pred.min())
        max_val = max(true.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], color='tab:red', linestyle='--', alpha=0.8)
        
        ax.set_xlabel("True Gamma")
        ax.set_ylabel("Predicted Gamma")
        ax.set_title(f"Method: {method.upper()}")
        
        # Log scales for noise strengths
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    logger.info(f"Plot saved to {output_path}")

def run_comparison(epochs: int = 50, batch_size: int = 32, plot: bool = True):
    data_dir = Path(__file__).parent / "data"
    methods = ["lindblad", "mcwf", "tjm"]
    results_mae = {}
    all_true = {}
    all_pred = {}

    for method in methods:
        data_path = data_dir / f"ising_{method}_data.npz"
        if not data_path.exists():
            logger.warning(f"Data for {method} not found at {data_path}. Skipping.")
            continue
            
        logger.info(f"=== Training Characterizer for {method.upper()} ===")
        
        # Train
        model = characterizer.characterize(
            training_data=data_path,
            epochs=epochs,
            batch_size=batch_size,
            lr=1e-3,
            model_class=characterizer.SimpleCNN,
            model_kwargs={"hidden_channels": 32}
        )
        
        # Evaluate
        data = np.load(data_path)
        X_raw = data["observables"]
        y_raw = data["gammas"]
        
        mae, preds = evaluate_model(model, X_raw, y_raw, characterizer.SimpleCNN)
        results_mae[method] = mae
        all_true[method] = y_raw
        all_pred[method] = preds
        
        logger.info(f"Method: {method}, MAE: {mae:.6f}")
        
        # Save Model
        models_dir = Path(__file__).parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"ising_characterizer_{method}_cnn.pt"
        torch.save(model.state_dict(), model_path)

    if plot and all_true:
        plot_path = Path(__file__).parent / "characterizer_comparison_results.png"
        plot_all_results(all_true, all_pred, plot_path)

    # Summary
    logger.info("\n" + "="*30)
    logger.info("      SUMMARY RESULTS")
    logger.info("="*30)
    for method, mae in results_mae.items():
        logger.info(f"{method:10}: MAE = {mae:.6f}")
    logger.info("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    args = parser.parse_args()
    
    run_comparison(epochs=args.epochs, batch_size=args.batch_size, plot=not args.no_plot)
