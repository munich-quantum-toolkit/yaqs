"""
Train the emulator on the generated Ising data.

This script loads the data from `experiments/data/ising_lindblad_data.npz` and trains
an Emulator (MLP) model to predict the system dynamics (observables) from the noise strength (gamma).
"""

import numpy as np
import torch
import argparse
from pathlib import Path
from mqt.yaqs.characterization import emulator

def train_ising_emulator(epochs: int = 100, batch_size: int = 32):
    # 1. Define Data Path
    data_path = Path(__file__).parent / "data" / "ising_lindblad_data.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}. Please run generate_ising_data.py first.")
    
    # 2. Train Model
    print(f"Starting emulator training for {epochs} epochs using data from {data_path}...")
    
    # The emulator handles data loading and reshaping automatically!
    # Input: gamma (1D)
    # Output: flattened observables (L*T)
    model = emulator.emulate(
        training_data=data_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-3,
        model_class=emulator.Emulator,
        model_kwargs={"hidden_dim": 256, "num_layers": 4}
    )
    
    # 3. Save Model
    output_dir = Path(__file__).parent / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "ising_emulator_mlp.pt"
    
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # 4. Verify Predictions
    data = np.load(data_path)
    X_raw = data["observables"] # Shape (N, L, T)
    gammas = data["gammas"]     # Shape (N,)
    N, L, T = X_raw.shape
    
    model.eval()
    print("\nSample Predictions (Dynamic Reconstruction):")
    indices = np.random.choice(N, min(N, 5), replace=False)
    
    # Prepare input tensor (gammas)
    gamma_tensor = torch.tensor(gammas.reshape(N, 1), dtype=torch.float32)
    
    with torch.no_grad():
        for idx in indices:
            # Input: gamma
            input_sample = gamma_tensor[idx].unsqueeze(0)
            
            # True Output: Flattened observables
            true_dynamics = X_raw[idx].flatten()
            
            # Predicted Output
            pred_dynamics = model(input_sample).squeeze().numpy()
            
            # Calculate MSE for this sample
            mse = np.mean((true_dynamics - pred_dynamics)**2)
            
            print(f"Sample {idx}: Gamma = {gammas[idx]:.5f}, MSE = {mse:.6f}")
            # print(f"  True first 5: {true_dynamics[:5]}")
            # print(f"  Pred first 5: {pred_dynamics[:5]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    
    train_ising_emulator(epochs=args.epochs, batch_size=args.batch_size)
