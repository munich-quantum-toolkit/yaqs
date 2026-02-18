# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Characterization module for MQT YAQS.

This module provides functionality to characterize noise in quantum systems by training
machine learning models on simulated time-evolution data.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.model_selection import train_test_split    
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import Observable
from mqt.yaqs.core.libraries.gate_library import GateLibrary

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mqt.yaqs.core.data_structures.networks import MPO, MPS
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
    from mqt.yaqs.core.libraries.gate_library import GateLibrary

logger = logging.getLogger(__name__)


class SimpleMLP(nn.Module):
    """A simple Multi-Layer Perceptron (MLP) for noise characterization."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, num_layers: int = 3) -> None:
        """Initialize the MLP.

        Args:
            input_dim: Dimension of the input vector (time_steps * observables * sites).
            output_dim: Dimension of the output vector (noise parameters).
            hidden_dim: Dimension of hidden layers. Defaults to 256.
            num_layers: Number of linear layers. must be >= 2. Defaults to 3.
        """
        super().__init__()
        layers: list[nn.Module] = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        # Softplus to ensure positive noise strength predictions
        layers.append(nn.Softplus())
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class SimpleCNN(nn.Module):
    """A simple 2D Convolutional Neural Network (CNN) for noise characterization.

    Assumes input shape (Batch, Channels, Sites, Time).
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_channels: int = 32) -> None:
        """Initialize the CNN.

        Args:
            input_dim: Number of input channels (usually 1 for scalar observables).
            output_dim: Dimension of the output vector (noise parameters).
            hidden_channels: Number of channels in hidden layers. Defaults to 32.
        """
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_dim, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_channels * 2, output_dim),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # x shape: (N, C, H, W) -> (N, hidden*2, 1, 1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # Flatten
        return self.fc_layers(x)


class SimpleRNN(nn.Module):
    """A simple Recurrent Neural Network (RNN) for noise characterization.

    Assumes input shape (Batch, Time, Features).
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 64, num_layers: int = 2) -> None:
        """Initialize the RNN.

        Args:
            input_dim: Number of features per time step (e.g., number of sites).
            output_dim: Dimension of the output vector.
            hidden_size: Hidden state size of LSTM.
            num_layers: Number of LSTM layers.
        """
        super().__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_dim),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # x shape: (Batch, Time, Features)
        # output shape: (Batch, Time, Hidden)
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        last_step_out = lstm_out[:, -1, :]
        
        return self.fc(last_step_out)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str | torch.device = "cpu"
) -> list[float]:
    """Train the PyTorch model.

    Args:
        model: The Neural Network to train.
        train_loader: DataLoader containing training data.
        test_loader: DataLoader containing test data.
        epochs: Number of training epochs.
        lr: Learning rate.
        device: Device to train on (cpu or cuda).

    Returns:
        list[float]: History of loss values per epoch.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    loss_history = []
    
    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * data.size(0)
        
        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch {_ + 1}/{epochs}, Loss: {avg_loss:.6f}")
        loss_history.append(avg_loss)
    
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            preds = model(data)
            loss = criterion(preds, target)

            test_loss += loss.item()

    print(f"Test Loss: {test_loss / len(test_loader):.6f}")
    return loss_history


from pathlib import Path

def characterize(
    training_data: tuple[torch.Tensor, torch.Tensor] | str | Path | None = None,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.001,
    model_class: type[nn.Module] = SimpleMLP,
    model_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    """Characterize the noise of a quantum system using machine learning.

    Orchestrates model initialization and training.

    Args:
        training_data: Training data. Can be:
            - A tuple (X, y) of pre-generated PyTorch tensors.
            - A path (str or Path) to a .npz file containing 'observables' and 'gammas'.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        lr: Learning rate.
        model_class: PyTorch Model class to use (SimpleMLP, SimpleCNN, SimpleRNN).
        model_kwargs: Keyword arguments for model initialization.

    Returns:
        nn.Module: The trained PyTorch model.
    """
    # 1. Prepare Data
    if isinstance(training_data, (str, Path)):
        data_path = Path(training_data)
        if not data_path.exists():
             raise FileNotFoundError(f"Data file not found at {data_path}")
        
        logger.info(f"Loading data from {data_path}...")
        data = np.load(data_path)
        
        # Assume specific keys for now based on user's generation script
        # In a more general version, these keys could be arguments
        if "observables" not in data or "gammas" not in data:
            raise ValueError("Data file must contain 'observables' and 'gammas' keys.")
            
        X_raw = data["observables"] # Shape (N, L, T)
        y_raw = data["gammas"]      # Shape (N,)
        
        N, L, T = X_raw.shape
        
        # Reshape based on model class requirements
        if model_class == SimpleCNN:
            # (N, Channels=1, Height=L, Width=T)
            X_data = X_raw.reshape(N, 1, L, T)
        elif model_class == SimpleRNN:
            # (N, Time=T, Features=L)
            X_data = X_raw.transpose(0, 2, 1)
        else: # SimpleMLP or generic
            # Flatten: (N, Features=L*T)
            X_data = X_raw.reshape(N, -1)
            
        y_data = y_raw.reshape(N, 1)
        
        x_data = torch.tensor(X_data, dtype=torch.float32)
        y_data = torch.tensor(y_data, dtype=torch.float32)
        
    elif isinstance(training_data, tuple):
        x_data, y_data = training_data
    else:
        raise ValueError("training_data must be a tuple of tensors or a file path.")

    # 2. Setup DataLoader
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # 3. Initialize Model
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    if model_kwargs is None:
        model_kwargs = {}

    if model_class is SimpleMLP:
        # Input dim - System size L
        # Output dim - Number of noise parameters
        model = model_class(input_dim, output_dim, **model_kwargs)
    else:
        try:
            # Try initializing with dimensions first (common pattern)
            model = model_class(input_dim, output_dim, **model_kwargs)
        except TypeError:
            # Fallback for models with different signatures
             model = model_class(**model_kwargs)

    # 4. Train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_history = train_model(model, train_loader, test_loader, epochs=epochs, lr=lr, device=device)

    logger.info("Training complete. Final Loss: %f", train_history[-1])
    
    model.to("cpu")
    model.eval()
    
    return model
