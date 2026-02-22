from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.tomography import run

if __name__ == "__main__":
    # 1. Your Hamiltonian (MPO)
    H = MPO.ising(length=10, J=1.0, g=0.5) 
    # 2. Setup your time resolution (dt)
    params = AnalogSimParams(
        dt=0.01,           # Your 'dt' timestep size
        solver="TJM",      # or "MCWF"
        max_bond_dim=32,
    )
    # 3. Define your time segments
    # This tells the code: evolve for t1, stop/intervene, then evolve for t2
    timesteps = [0.5, 0.5]  # t1 = 0.5, t2 = 0.5
    # 4. Run the tomography
    # This will iterate through all basis sequences and return the Process Tensor
    pt = run(H, params, timesteps)
    print(f"Holevo Information: {pt.holevo_information()}")
