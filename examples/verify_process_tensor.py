
import numpy as np
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.core.libraries.gate_library import GateLibrary
from mqt.yaqs.tomography.tomography import run_process_tensor_tomography

def create_memory_system_mpo():
    """Creates a 2-site Heisenberg MPO (System + Ancilla).
    
    H = X0X1 + Y0Y1 + Z0Z1
    """
    # Use the robust from_pauli_sum constructor
    mpo = MPO()
    mpo.from_pauli_sum(
        terms=[(1.0, "X0 X1"), (1.0, "Y0 Y1"), (1.0, "Z0 Z1")],
        length=2,
        physical_dimension=2
    )
    return mpo


def verify_process_tensor():
    print("Creating Heisenberg System MPO...")
    mpo = create_memory_system_mpo()
    
    # Simulation Parameters
    # We want a SWAP time. For H_Heis, SWAP is at t = pi/4?
    # H = X.X + Y.Y + Z.Z
    # U(t) = exp(-iHt).
    # Eigenvalues of H on triplet are 1, singlet is -3.  (Wait, X.X+...)
    # Let's just pick a time that does SOMETHING. t=0.5
    dt = 0.5
    
    sim_params = AnalogSimParams(
        elapsed_time=dt,
        dt=0.1,
        num_traj=100,  # Trajectories for averaging
        max_bond_dim=10,
        order=1,  # Use first-order Trotterization (simpler, still uses 2-site TDVP)
        get_state=True,
    )
    
    # Two steps: 0 -> t1, t1 -> t2
    timesteps = [0.5, 0.5]
    
    print("Running Process Tensor Tomography (this may take a minute)...")
    pt = run_process_tensor_tomography(mpo, sim_params, timesteps, num_trajectories=100)
    
    print("\nProcess Tensor Result:")
    print(pt)
    
    # Analyze the tensor
    # T shape: [Out(4), In0(4), In1(4)]
    # T_{ijk} corresponds to Rho_out_i given Rho_in0_j and Rho_in1_k
    
    # Check if input 0 affects output (Non-Markovianity) across the gap
    # If Markovian, T approx = Map(In1->Out) * Trace(In0) ?
    # More precisely, if Markovian, the choice of In0 should NOT affect the map from In1->Out.
    # i.e., T[:, fixed, :] should be the same up to scale?
    
    T = pt.tensor
    
    # Slice 1: Input 0 is |0><0| (Index 0 in Pauli basis usually I+Z -> 0)
    # Standard basis order in code: Zeros, Ones, X+, X-, Y+, Y-
    # Zeros is index 0. Ones is index 1.
    
    # In the Frame/Dual basis, index 0 might not correspond directly to state 0 if we use abstract basis.
    # The code uses the DUAL frame.
    # The reconstructed tensor T acts on vectorized density matrices.
    # Predicted rho_vec = T . (rho0_vec \otimes rho1_vec \otimes ...)
    
    # Construct vector for 'zeros' state and 'ones' state
    # We can assume the standard basis vectors are reconstructing correctly.
    # Let's project T onto specific inputs.
    
    # Get basis vectors
    from mqt.yaqs.tomography.tomography import _get_basis_states
    basis = _get_basis_states()
    # Process Tensor acts on flattened density matrices
    rho_0 = basis[0][2].reshape(-1) # zeros
    rho_1 = basis[1][2].reshape(-1) # ones
    
    # Predict Output if (In0=Zeros, In1=Zeros)
    out_00 = np.einsum('ijk, j, k -> i', T, rho_0, rho_0)
    
    # Predict Output if (In0=Ones,  In1=Zeros)
    out_10 = np.einsum('ijk, j, k -> i', T, rho_1, rho_0)
    
    print(f"\nOutput Vec (In0=0, In1=0): {out_00.round(3)}")
    print(f"Output Vec (In0=1, In1=0): {out_10.round(3)}")
    
    diff = np.linalg.norm(out_00 - out_10)
    print(f"Difference (Memory Effect): {diff:.5f}")
    
    if diff > 0.05:
        print("SUCCESS: Non-Markovian memory detected! (Input 0 affected final output)")
    else:
        print("WARNING: Little to no memory detected via this metric (or system is Markovian/Identity).")

if __name__ == "__main__":
    verify_process_tensor()
