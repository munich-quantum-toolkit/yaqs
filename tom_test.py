from mqt.yaqs.tomography import run
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

# 1. Define Operator (e.g., Ising Model)
L = 10
operator = MPO.ising(L, J=1.0, g=1)

# 2. Define Simulation Parameters
params = AnalogSimParams(
    elapsed_time=2,
    dt=0.1,
    order=2,
    max_bond_dim=10,
    get_state=True  # Required by params validation, though tomography overrides observables
)

# 3. Run Tomography
process_tensor = run(operator, params)
# Returns shape (2, 2, 2, 2)
