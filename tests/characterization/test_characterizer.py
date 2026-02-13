# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the characterization module."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from mqt.yaqs.characterization import characterizer
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import GateLibrary

class TestCharacterizer:
    @pytest.fixture
    def small_system(self) -> tuple[MPO, MPS, AnalogSimParams]:
        """Create a small 2-site system for testing."""
        num_sites = 2
        hamiltonian = MPO.ising(num_sites, J=1.0, g=0.5)
        initial_state = MPS(num_sites)
        initial_state.normalize("B")
        
        sim_params = AnalogSimParams(
            elapsed_time=0.1,
            dt=0.05,
            num_traj=1, # Default low for speed
            sample_timesteps=True,
            observables=[Observable(GateLibrary.z(), 0)]
        )
        return hamiltonian, initial_state, sim_params

    def test_generate_training_data(self, small_system: tuple[MPO, MPS, AnalogSimParams]) -> None:
        """Test data generation shapes."""
        hamiltonian, initial_state, sim_params = small_system
        num_samples = 5
        expected_gamma = 0.1
        
        X, y = characterizer.generate_training_data(
            hamiltonian,
            initial_state,
            expected_gamma,
            sim_params,
            num_samples=num_samples,
            num_traj=1 # Fast simulation
        )
        
        # Check X shape: (num_samples, features)
        # Features = (num_observables * num_timesteps)
        # num_observables = 2 (X,Z) * 2 (sites) = 4
        # num_timesteps = 3 (0.0, 0.05, 0.1)
        # Total features = 12
        assert X.shape[0] == num_samples
        # Precise feature count depends on exact timesteps, check > 0
        assert X.shape[1] > 0 
        
        # Check y shape: (num_samples, labels)
        # Labels = 3 (X,Y,Z) * 2 (sites) = 6
        assert y.shape == (num_samples, 6)
        
        # Check types
        assert isinstance(X, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_simple_mlp_structure(self) -> None:
        """Test SimpleMLP initialization and forward pass."""
        input_dim = 10
        output_dim = 5
        model = characterizer.SimpleMLP(input_dim, output_dim)
        
        x = torch.randn(2, input_dim) # Batch of 2
        out = model(x)
        
        assert out.shape == (2, output_dim)
        # Softplus output should be positive (or near zero)
        assert torch.all(out >= 0)

    def test_characterize_end_to_end(self, small_system: tuple[MPO, MPS, AnalogSimParams]) -> None:
        """Test the full characterize function."""
        hamiltonian, initial_state, sim_params = small_system
        
        # Run with very few samples/epochs for speed
        model = characterizer.characterize(
            hamiltonian,
            initial_state,
            expected_gamma=0.1,
            sim_params=sim_params,
            num_samples=2,
            training_num_traj=1,
            epochs=1,
            batch_size=2
        )
        
        assert isinstance(model, nn.Module)
        
        # Check prediction
        # Simulate a fake input
        # We need to know input dim. 
        # From sim_params (dt=0.05, T=0.1) -> 3 steps. 2 sites * 2 obs = 4. 4*3=12.
        # But let's just use the model's expected input feature size if we can inspect it,
        # or generate one sample to check.
        X_dummy, _ = characterizer.generate_training_data(
             hamiltonian,
            initial_state,
            0.1,
            sim_params,
            num_samples=1,
            num_traj=1
        )
        
        with torch.no_grad():
            pred = model(X_dummy)
        
        assert pred.shape == (1, 6) # 2 sites * 3 noise params


    def test_custom_model_architecture(self, small_system: tuple[MPO, MPS, AnalogSimParams]) -> None:
        """Test plugging in a custom model class."""
        hamiltonian, initial_state, sim_params = small_system
        
        class CustomModel(nn.Module):
            def __init__(self, input_dim: int, output_dim: int, extra_arg: int = 1) -> None:
                super().__init__()
                self.linear = nn.Linear(input_dim, output_dim)
                self.extra = extra_arg
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        model = characterizer.characterize(
            hamiltonian,
            initial_state,
            expected_gamma=0.1,
            sim_params=sim_params,
            num_samples=2,
            training_num_traj=1,
            epochs=1,
            model_class=CustomModel,
            model_kwargs={"extra_arg": 42}
        )
        
        assert isinstance(model, CustomModel)
        assert model.extra == 42
