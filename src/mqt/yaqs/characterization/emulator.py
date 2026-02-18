# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Emulator module for MQT YAQS.

This module provides functionality to emulate quantum system dynamics by predicting
observables from noise parameters using machine learning models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class Emulator(nn.Module):
    """A simple Multi-Layer Perceptron (MLP) for emulating system dynamics.
    
    Predicts the time evolution of observables (flattened) given noise parameters.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, num_layers: int = 4) -> None:
        """Initialize the Emulator MLP.

        Args:
            input_dim: Dimension of the input vector (noise parameters).
            output_dim: Dimension of the output vector (time_steps * observables * sites).
            hidden_dim: Dimension of hidden layers. Defaults to 256.
            num_layers: Number of linear layers. Defaults to 4.
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
            
        # Output layer (no activation, as observables can be neg/pos)
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


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
        device: Device to train on.

    Returns:
        list[float]: History of loss values per epoch.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    loss_history = []
    
    model.train()
    for dtype in range(epochs):
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
        # Log occasionally
        if (dtype + 1) % 10 == 0:
            logger.info("Epoch %d/%d, Loss: %.6f", dtype + 1, epochs, avg_loss)
            
        loss_history.append(avg_loss)
    
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            preds = model(data)
            loss = criterion(preds, target)

            test_loss += loss.item()

    logger.info("Test MSE: %.6f", test_loss / len(test_loader))
    return loss_history


def emulate(
    training_data: tuple[torch.Tensor, torch.Tensor] | str | Path | None = None,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.001,
    model_class: type[nn.Module] = Emulator,
    model_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    """Emulate the dynamics of a quantum system using machine learning.

    Orchestrates model initialization and training. Inverse of characterization.

    Args:
        training_data: Training data. Can be:
            - A tuple (X, y) of pre-generated PyTorch tensors. X=params, y=observables.
            - A path (str or Path) to a .npz file containing 'observables' and 'gammas'.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        lr: Learning rate.
        model_class: PyTorch Model class to use.
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
        
        if "observables" not in data or "gammas" not in data:
            raise ValueError("Data file must contain 'observables' and 'gammas' keys.")
            
        observables_raw = data["observables"] # Shape (N, L, T)
        gammas_raw = data["gammas"]           # Shape (N,)
        
        N, L, T = observables_raw.shape
        
        # For emulation:
        # Input X: Noise parameters (gammas), shape (N, 1)
        # Output y: System response (observables), flattened shape (N, L*T)
        
        X_data = gammas_raw.reshape(N, 1)
        y_data = observables_raw.reshape(N, -1)
        
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

    try:
        model = model_class(input_dim, output_dim, **model_kwargs)
    except TypeError:
        model = model_class(**model_kwargs)

    # 4. Train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_model(model, train_loader, test_loader, epochs=epochs, lr=lr, device=device)
    
    model.to("cpu")
    model.eval()
    
    return model
