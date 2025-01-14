from yaqs.general.data_structures.MPO import MPO
from yaqs.general.data_structures.MPS import MPS
from yaqs.general.data_structures.simulation_parameters import PhysicsSimParams
from yaqs.physics.methods.TDVP import single_site_TDVP, two_site_TDVP


def dynamic_TDVP(state: MPS, H: MPO, sim_params):
    """
    Perform a dynamic Time-Dependent Variational Principle (TDVP) evolution of the system state.

    Depending on the current bond dimension of the state, this function either performs a two-site TDVP (2TDVP) or a single-site TDVP (1TDVP) to evolve the state.

    Args:
        state (MPS): The Matrix Product State (MPS) representing the current state of the system.
        H (MPO): The Matrix Product Operator (MPO) representing the Hamiltonian of the system.
    Returns:
        None
    """
    current_max_bond_dim = state.write_max_bond_dim()
    # state.normalize('B')
    if current_max_bond_dim <= sim_params.max_bond_dim:
        # Perform 2TDVP when the current bond dimension is within the allowed limit
        two_site_TDVP(state, H, sim_params)
    else:
        # Perform 1TDVP when the bond dimension exceeds the allowed limit
        single_site_TDVP(state, H, sim_params)
