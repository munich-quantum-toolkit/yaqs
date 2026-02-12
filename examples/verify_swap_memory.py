"""
Simpler Process Tensor verification using SWAP Hamiltonian.

This test uses a SWAP gate as the Hamiltonian, which should produce
clear memory effects:
- H = X⊗X + Y⊗Y + Z⊗Z (Heisenberg) with coefficient chosen to give SWAP
- At t = π/4, this implements a partial SWAP
- Memory test: prepare |0⟩, evolve, prepare |1⟩, evolve, measure
  - If Markovian: final state independent of first preparation
  - If non-Markovian: final state depends on first preparation (memory!)
"""

import numpy as np
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.tomography.tomography import run_process_tensor_tomography


def create_swap_mpo():
    """Creates a 2-site SWAP Hamiltonian.
    
    H = (X⊗X + Y⊗Y + Z⊗Z) / 2
    
    This gives SWAP gate at t = π/2.
    For partial SWAP, we'll use t = π/4.
    """
    mpo = MPO()
    # Coefficient of 0.5 makes the evolution time more reasonable
    mpo.from_pauli_sum(
        terms=[(0.5, "X0 X1"), (0.5, "Y0 Y1"), (0.5, "Z0 Z1")],
        length=2,
        physical_dimension=2
    )
    return mpo


def verify_swap_memory():
    """Test Process Tensor with SWAP Hamiltonian."""
    print("Creating SWAP Hamiltonian MPO...")
    mpo = create_swap_mpo()
    
    # Time for partial SWAP: π/4 with coefficient 0.5 means t = π/2
    dt = np.pi / 2  # This should give a full SWAP
    
    sim_params = AnalogSimParams(
        elapsed_time=dt,
        dt=0.1,
        num_traj=100,  # Trajectories for averaging
        max_bond_dim=10,
        order=1,
        get_state=True,
    )
    
    # Two steps: 0 -> t1, t1 -> t2
    timesteps = [dt, dt]  # Two evolution periods
    
    print("Running Process Tensor Tomography with SWAP...")
    print(f"Time steps: {timesteps}")
    print(f"Trajectories per config: {sim_params.num_traj}")
    
    pt = run_process_tensor_tomography(
        operator=mpo,
        sim_params=sim_params,
        timesteps=timesteps,
        num_trajectories=sim_params.num_traj
    )
    
    print("\nProcess Tensor Result:")
    print(pt)
    
    # Test for memory: compare outputs for different input sequences
    # Sequence (0, 0): prepare |0⟩ at t=0, |0⟩ at t=1
    # Sequence (1, 0): prepare |1⟩ at t=0, |0⟩ at t=1
    # If there's memory, these should give DIFFERENT final states
    
    print("\nMemory Test:")
    print("If non-Markovian, the first input should affect the final output")
    print("even though we re-prepare the system at t=1.")
    
    # Get the reconstructed process tensor data
    # pt.data has shape (4, 4, 4) for 2 steps, 2-level system
    # Indices: [out, in1, in0]
    
    # Compare: what's the final state if we prepare (in0=0, in1=0) vs (in0=1, in1=0)?
    if hasattr(pt, 'data'):
        # Output distribution for (in0=0, in1=0)
        out_00 = pt.data[:, 0, 0]
        # Output distribution for (in0=1, in1=0)  
        out_10 = pt.data[:, 0, 1]
        
        diff = np.linalg.norm(out_00 - out_10)
        print(f"\nDifference between (0,0) and (1,0) outputs: {diff:.6f}")
        
        if diff > 0.01:
            print("✓ Memory detected! The initial state affects the final output.")
        else:
            print("✗ No memory detected. System appears Markovian.")
    else:
        print("Process tensor data not available for analysis.")


if __name__ == "__main__":
    verify_swap_memory()
