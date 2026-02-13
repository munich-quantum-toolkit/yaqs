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


def generate_training_data(
    hamiltonian: MPO,
    initial_state: MPS,
    expected_gamma: float,
    sim_params: AnalogSimParams,
    num_samples: int,
    num_traj: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate training data for the characterizer.

    Runs simulations with randomized noise models to create a dataset mapping
    observed time evolution to noise parameters.

    Args:
        hamiltonian: The system Hamiltonian (MPO).
        initial_state: The initial state (MPS).
        expected_gamma: The expected magnitude of the noise strength (gamma).
                        Used as the mean for lognormal sampling.
        sim_params: Simulation parameters.
        num_samples: Number of training samples (noise realizations) to generate.
        num_traj: Number of trajectories per simulation. If provided, overrides sim_params.num_traj.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - X_train: Input features (evolution data) of shape (num_samples, feature_dim).
            - y_train: Labels (noise strengths) of shape (num_samples, label_dim).
    """
    if num_traj is not None:
        sim_params.num_traj = num_traj
    
    # We enforce tracking X and Z observables for characterization on all sites
    # Flattened output vector will be constructed from these.
    # Note: Using existing observables in sim_params might be preferred, but 
    # to ensure coverage we might want to check or override.
    # For now, let's assume the user configures sim_params correctly or we append if missing.
    # Actually, simpler to force X and Z on all sites as per the prompt requirement.
    
    # Identify sites from the Hamiltonian or state
    num_sites = initial_state.length
    
    # Setup observables: X and Z for each site
    characterization_observables = []
    for site in range(num_sites):
        characterization_observables.append(Observable(GateLibrary.x(), site))
        characterization_observables.append(Observable(GateLibrary.z(), site))
    
    # Update sim_params with these observables
    sim_params.observables = characterization_observables
    sim_params.get_state = False # Ensure no state return for characterization
    # Trigger re-sorting/initialization internals
    # (We re-create or manually set sorted_observables as AnalogSimParams.__init__ does)
    # Re-initialization of sim_params with new observables is safer but expensive if we copy 
    # large structures. Just re-sorting:
    sortable = [obs for obs in sim_params.observables if obs.gate.name not in {"pvm", "runtime_cost", "max_bond", "total_bond"}]
    unsorted = [obs for obs in sim_params.observables if obs.gate.name in {"pvm", "runtime_cost", "max_bond", "total_bond"}]
    sorted_obs = sorted(
        sortable, 
        key=lambda obs: obs.sites[0] if isinstance(obs.sites, list) else obs.sites
    )
    sim_params.sorted_observables = sorted_obs + unsorted

    # Storage
    evolution_data: list[np.ndarray] = []
    noise_labels: list[np.ndarray] = []

    # Calculate log-normal parameters
    # If X ~ Lognormal(mu, sigma), then E[X] = exp(mu + sigma^2 / 2).
    # We want E[X] = expected_gamma.
    # Let's pick a fixed sigma (variance width) or allow it to be configured? 
    # For simplicity, let's assume sigma=0.5 (moderate spread) and solve for mu.
    sigma = 0.5
    mu = np.log(expected_gamma) - (sigma**2 / 2)

    logger.info("Generating %d training samples with expected gamma %f", num_samples, expected_gamma)

    for _ in tqdm(range(num_samples), desc="Generating Data"):
        # 1. Create a random noise model
        processes = []
        current_labels = []
        
        # We assume independent noise on each site for X, Y, Z (as per prompt implication)
        # prompt: "lognormal distribution for X, Y, Z"
        for site in range(num_sites):
            for axis in ["x", "y", "z"]:
                # Sample strength
                strength = float(np.random.lognormal(mean=mu, sigma=sigma))
                
                processes.append({
                    "name": axis,  # "x", "y", "z" are valid keys in NoiseLibrary/NoiseModel
                    "sites": [site],
                    "strength": strength
                })
                current_labels.append(strength)

        noise_model = NoiseModel(processes=processes)
        
        # 2. Run simulation
        # Note: simulator.run typically modifies initial_state in-place for state evolution 
        # but for analog sim it uses MPO evolution.
        # However, run() normalizes initial_state. To avoid side effects on the passed 
        # initial_state across loops, we should deepcopy it.
        state_copy = copy.deepcopy(initial_state)
        
        # Run simulation
        simulator.run(state_copy, hamiltonian, sim_params, noise_model=noise_model)
        
        # 3. Extract data
        # sim_params.observables now contains results.
        # results structure: sim_params.observables[i].results is array of shape (len(times),)
        
        # We need to flatten all observables into a single vector
        # Order: [Obs1_t0, Obs1_t1, ..., Obs2_t0, ...] or [t0_Obs1, t0_Obs2...]
        # Let's stack them: (Num_Observables, Num_TimeSteps)
        features = []
        for obs in sim_params.sorted_observables:
             if obs.results is not None:
                 features.append(obs.results)
        
        # Stack feature: Shape (2*N, T) -> flatten
        feature_vector = np.concatenate(features, axis=0) # Flattened array
        
        evolution_data.append(feature_vector)
        noise_labels.append(np.array(current_labels))

    # Convert to Tensors
    X_train = torch.tensor(np.array(evolution_data), dtype=torch.float32)
    y_train = torch.tensor(np.array(noise_labels), dtype=torch.float32)
    
    return X_train, y_train


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
        loss_history.append(avg_loss)
    
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            preds = model(data)
            loss = criterion(preds, target)

            test_loss += loss.item()

    print(f"Test Loss: {test_loss / len(test_loader):.6f}")
    return loss_history


def run(
    training_data: tuple[torch.Tensor, torch.Tensor] | None = None,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.001,
    model_class: type[nn.Module] = SimpleMLP,
    model_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    """Characterize the noise of a quantum system using machine learning.

    Orchestrates model initialization and training.

    Args:
        training_data: Optional tuple (X, y) of pre-generated training data.
        num_samples: Number of samples to generate if training_data is None.
        training_num_traj: Override for number of trajectories during data generation.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        lr: Learning rate.
        model_class: PyTorch Model class to use.
        model_kwargs: Keyword arguments for model initialization.

    Returns:
        nn.Module: The trained PyTorch model.
    """
    # 1. Prepare Data
    x_data, y_data = training_data

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
            model = model_class(input_dim, output_dim, **model_kwargs)
        except TypeError:
            model = model_class(**model_kwargs)

    # 4. Train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_history = train_model(model, train_loader, test_loader, epochs=epochs, lr=lr, device=device)

    logger.info("Training complete. Final Loss: %f", train_history[-1])
    
    model.to("cpu")
    model.eval()
    
    return model
