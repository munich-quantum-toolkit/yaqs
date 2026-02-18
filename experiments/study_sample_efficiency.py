"""
Study sample efficiency of Characterizer (CNN) and Emulator (MLP).

Trains models on subsets of the data [10, 50, 100, 200, 400, 800] and evaluates
performance on a fixed hold-out test set.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from mqt.yaqs.characterization import characterizer, emulator

# Configuration
DATA_PATH = Path(__file__).parent / "data" / "ising_lindblad_data.npz"
TRAIN_SIZES = [10, 50, 100, 200, 400, 800]
EPOCHS_LIST = [10, 50, 100]
TEST_SIZE = 200
BATCH_SIZE = 32

def train_and_evaluate(
    model_func, 
    model_class, 
    model_kwargs,
    X_train, y_train, 
    X_test, y_test,
    epochs
):
    """Train a model on X_train/y_train and evaluate on X_test/y_test."""
    # Create DataLoaders
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model
    input_dim = X_train.shape[1] 
    output_dim = y_train.shape[1]
    
    if model_class == characterizer.SimpleCNN:
        model = model_class(input_dim, output_dim, **model_kwargs)
    elif model_class == emulator.Emulator:
        model = model_class(input_dim, output_dim, **model_kwargs)
    else:
         model = model_class(input_dim, output_dim, **model_kwargs)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    characterizer.train_model(
        model, train_loader, test_loader, epochs=epochs, lr=1e-3, device=device
    )
    
    # Final evaluation
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
             data, target = data.to(device), target.to(device)
             preds = model(data)
             loss = criterion(preds, target)
             total_loss += loss.item()
             
    avg_test_loss = total_loss / len(test_loader)
    return avg_test_loss

def run_study():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
        
    print(f"Loading data from {DATA_PATH}...")
    data = np.load(DATA_PATH)
    
    observables = data["observables"] # Shape (N, L, T)
    gammas = data["gammas"]           # Shape (N,)
    
    N, L, T = observables.shape
    
    # Prepare Data Tensors
    char_X = torch.tensor(observables.reshape(N, 1, L, T), dtype=torch.float32)
    char_y = torch.tensor(gammas.reshape(N, 1), dtype=torch.float32)
    
    emu_X = torch.tensor(gammas.reshape(N, 1), dtype=torch.float32)
    emu_y = torch.tensor(observables.reshape(N, -1), dtype=torch.float32)
    
    test_idx = list(range(N - TEST_SIZE, N))
    train_pool_idx = list(range(0, N - TEST_SIZE))
    
    # Results storage: {epochs: {size: {'char': mse, 'emu': mse}}}
    results = {}
    
    print(f"Starting sample efficiency study...")
    print(f"Test Set Size: {TEST_SIZE}")
    print(f"Training Sizes: {TRAIN_SIZES}")
    print(f"Epoch Settings: {EPOCHS_LIST}")
    
    for epochs in EPOCHS_LIST:
        print(f"\n=== Epochs: {epochs} ===")
        results[epochs] = {}
        for size in TRAIN_SIZES:
            print(f"\n--- Training Set Size: {size} ---")
            subset_idx = train_pool_idx[:size]
            
            # 1. Evaluate Characterizer
            print("Training Characterizer (CNN)...")
            loss_char = train_and_evaluate(
                characterizer.characterize,
                characterizer.SimpleCNN,
                {"hidden_channels": 32},
                char_X[subset_idx], char_y[subset_idx],
                char_X[test_idx], char_y[test_idx],
                epochs
            )
            print(f"  -> Test MSE: {loss_char:.6f}")
            
            # 2. Evaluate Emulator
            print("Training Emulator (MLP)...")
            loss_emu = train_and_evaluate(
                emulator.emulate,
                emulator.Emulator,
                {"hidden_dim": 256, "num_layers": 4},
                emu_X[subset_idx], emu_y[subset_idx],
                emu_X[test_idx], emu_y[test_idx],
                epochs
            )
            print(f"  -> Test MSE: {loss_emu:.6f}")
            
            results[epochs][size] = {'char': loss_char, 'emu': loss_emu}

    # Save results to file
    results_file = Path(__file__).parent / "sample_efficiency_extended_results.txt"
    with open(results_file, "w") as f:
        f.write("Epochs,Train_Size,Characterizer_MSE,Emulator_MSE\n")
        for epochs in EPOCHS_LIST:
            for size in TRAIN_SIZES:
                res = results[epochs][size]
                f.write(f"{epochs},{size},{res['char']:.8f},{res['emu']:.8f}\n")
    print(f"Results saved to {results_file}")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Characterizer
    for epochs in EPOCHS_LIST:
        char_mses = [results[epochs][size]['char'] for size in TRAIN_SIZES]
        ax1.plot(TRAIN_SIZES, char_mses, 'o-', label=f'{epochs} Epochs', linewidth=2)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of Training Samples')
    ax1.set_ylabel('Test MSE (Log Scale)')
    ax1.set_title('Characterizer (CNN) Performance')
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    ax1.legend()
    
    # Plot 2: Emulator
    for epochs in EPOCHS_LIST:
        emu_mses = [results[epochs][size]['emu'] for size in TRAIN_SIZES]
        ax2.plot(TRAIN_SIZES, emu_mses, 's-', label=f'{epochs} Epochs', linewidth=2)

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Number of Training Samples')
    ax2.set_ylabel('Test MSE (Log Scale)')
    ax2.set_title('Emulator (MLP) Performance')
    ax2.grid(True, which="both", ls="-", alpha=0.5)
    ax2.legend()
    
    output_png = Path(__file__).parent / "sample_efficiency_extended_plot.png"
    plt.savefig(output_png)
    print(f"\nStudy complete. Plot saved to {output_png}")

if __name__ == "__main__":
    run_study()
