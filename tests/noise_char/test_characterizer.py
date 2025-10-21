from mqt.yaqs.noise_char.characterizer import Characterizer
from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import X, Y, Z
from mqt.yaqs.noise_char.optimization import LossClass
from mqt.yaqs.noise_char import optimization, propagation
from mqt.yaqs.core.data_structures.networks import MPO, MPS


from pathlib import Path
import numpy as np
import pytest

class Parameters:
    def __init__(self) -> None:
        self.sites = 1
        self.sim_time = 0.6
        self.dt = 0.2
        self.order = 1
        self.threshold = 1e-4
        self.ntraj = 1
        self.max_bond_dim = 4
        self.j = 1
        self.g = 0.5


        self.times = np.arange(0, self.sim_time + self.dt, self.dt)

        self.n_obs = self.sites * 3  # x, y, z for each site
        self.n_jump = self.sites * 2  # lowering and pauli_z for each site
        self.n_t = len(self.times)

        self.gamma_rel = 0.1
        self.gamma_deph = 0.15


        self.d = 2



def create_instances(tmp_path: Path, test: Parameters) -> LossClass:
    """Helper function to create a LossClass instance for testing."""


    h_0 = MPO()
    h_0.init_ising(test.sites, test.j, test.g)


    # Define the initial state
    init_state = MPS(test.sites, state='zeros')


    obs_list = [Observable(X(), site) for site in range(test.sites)]  + [Observable(Y(), site) for site in range(test.sites)] + [Observable(Z(), site) for site in range(test.sites)]


    sim_params = AnalogSimParams(observables=obs_list, elapsed_time=test.sim_time, dt=test.dt, num_traj=test.ntraj, max_bond_dim=test.max_bond_dim, threshold=test.threshold, order=test.order, sample_timesteps=True)


    ref_noise_model =  CompactNoiseModel([{"name": "lowering", "sites": [i for i in range(test.sites)], "strength": test.gamma_rel}]+[{"name": "pauli_z", "sites": [i for i in range(test.sites)], "strength": test.gamma_deph}])



    propagator = propagation.PropagatorWithGradients(
            sim_params=sim_params,
            hamiltonian=h_0,
            compact_noise_model=ref_noise_model,
            init_state=init_state
        )
    
    propagator.set_observable_list(obs_list)
    propagator.run(ref_noise_model)

    return h_0, init_state, obs_list, sim_params, ref_noise_model, propagator


def test_characterizer_init(tmp_path: Path) -> None:
    """Test that Characterizer initializes correctly with given parameters."""
    test = Parameters()


    h_0, init_state, obs_list, sim_params, ref_noise_model, propagator = create_instances(tmp_path, test)


    characterizer = Characterizer(
        sim_params=sim_params,
        hamiltonian=h_0,
        init_guess=ref_noise_model,
        init_state=init_state,
        ref_traj=propagator.obs_traj,
        work_dir=tmp_path,
        print_to_file=False
    )


    assert isinstance(characterizer.init_guess, CompactNoiseModel)
    assert isinstance(characterizer.traj_gradients, propagation.PropagatorWithGradients)
    assert isinstance(characterizer.loss, LossClass)
    assert isinstance(characterizer.init_x, np.ndarray)


    
    characterizer.adam_optimize(max_iterations=1)

    assert isinstance(characterizer.optimal_model, CompactNoiseModel)





