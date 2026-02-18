"""
Train the characterizer on the generated Ising data.

This script loads the data from `experiments/data/ising_lindblad_data.npz` and trains
a SimpleMLP model to predict the noise strength (gamma) from the time evolution of Z observables.
"""

import numpy as np
import torch
import argparse
from pathlib import Path
from mqt.yaqs.characterization import characterizer

def train_ising_characterizer(epochs: int = 100, batch_size: int = 32):
    # 1. Define Data Path
    data_path = Path(__file__).parent / "data" / "ising_lindblad_data.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}. Please run generate_ising_data.py first.")
    
    # 2. Train Model
    print(f"Starting training for {epochs} epochs using data from {data_path}...")
    
    # The characterizer now handles data loading and reshaping automatically!
    model = characterizer.characterize(
        training_data=data_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-3,
        model_class=characterizer.SimpleCNN,
        model_kwargs={"hidden_channels": 32}
    )
    
    # 3. Save Model
    output_dir = Path(__file__).parent / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "ising_characterizer_cnn.pt"
    
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Optional: Quick evaluation on a few samples
    # We load data here just for the manual check, or we could rely on the training output
    # Let's verify predictions manually
    data = np.load(data_path)
    X_raw = data["observables"]
    gammas = data["gammas"]
    N, L, T = X_raw.shape
    
    model.eval()
    print("\nSample Predictions:")
    indices = np.random.choice(N, min(N, 5), replace=False)
    
    # Need to manually reshape for inference check to match what model expects
    X_tensor = torch.tensor(X_raw.reshape(N, 1, L, T), dtype=torch.float32)
    y_tensor = torch.tensor(gammas.reshape(N, 1), dtype=torch.float32)
    
    with torch.no_grad():
        for idx in indices:
            input_sample = X_tensor[idx].unsqueeze(0)
            true_val = y_tensor[idx].item()
            pred_val = model(input_sample).item()
            print(f"Sample {idx}: True Gamma = {true_val:.5f}, Predicted = {pred_val:.5f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    
    train_ising_characterizer(epochs=args.epochs, batch_size=args.batch_size)
