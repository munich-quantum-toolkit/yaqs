# Characterizer example

This document contains the example script `characterizer_example.py` converted to Markdown. It shows how to set up a simple Ising Hamiltonian, generate a reference trajectory with a given noise model, build a LossClass and Characterizer, and run the optimizer.

## Requirements

- numpy
- mqt.yaqs package (propagation, optimization, characterizer, core.data_structures)
- A working environment where PropagatorWithGradients can run (the example uses the package's propagator)

## Usage

Copy the Python code block below into a `.py` file and run it, or execute the cells if using a notebook.

```python
import numpy as np
from pathlib import Path

from mqt.yaqs.noise_char.propagation import PropagatorWithGradients
from mqt.yaqs.noise_char.optimization import LossClass
from mqt.yaqs.noise_char.characterizer import Characterizer

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import (
    AnalogSimParams,
    Observable,
)
from mqt.yaqs.core.libraries.gate_library import X, Y, Z, Create, Destroy


if __name__ == "__main__":
    work_dir = "test/scikit_characterizer/"
    work_dir_path = Path(work_dir)
    work_dir_path.mkdir(parents=True, exist_ok=True)

    # Defining Hamiltonian and observable list
    L = 2
    J = 1
    g = 0.5

    H_0 = MPO()
    H_0.init_ising(L, J, g)

    # Define the initial state
    init_state = MPS(L, state="zeros")

    obs_list = [Observable(X(), site) for site in range(L)]

    noise_operator = "pauli_y"

    # Defining simulation parameters
    T = 3
    dt = 0.1
    N = 1000
    max_bond_dim = 8
    threshold = 1e-6
    order = 1

    sim_params = AnalogSimParams(
        observables=obs_list,
        elapsed_time=T,
        dt=dt,
        num_traj=N,
        max_bond_dim=max_bond_dim,
        threshold=threshold,
        order=order,
        sample_timesteps=True,
    )

    # Defining reference noise model and reference trajectory
    gamma_reference = 0.1

    ref_noise_model = CompactNoiseModel(
        [
            {
                "name": noise_operator,
                "sites": [i for i in range(L)],
                "strength": gamma_reference,
            }
        ]
    )

    # Write reference gammas to file
    np.savetxt(
        work_dir + "gammas.txt", ref_noise_model.strength_list, header="##", fmt="%.6f"
    )

    propagator = PropagatorWithGradients(
        sim_params=sim_params,
        hamiltonian=H_0,
        compact_noise_model=ref_noise_model,
        init_state=init_state,
    )

    propagator.set_observable_list(obs_list)

    print("Computing reference trajectory ... ")
    propagator.run(ref_noise_model)
    ref_traj = propagator.obs_traj
    print("Reference trajectory computed.")

    # Optimizing the model
    gamma_guess = 0.4

    sim_params.num_traj = 300  # Reducing the number of trajectories for the optimization

    guess_noise_model = CompactNoiseModel(
        [
            {
                "name": noise_operator,
                "sites": [i for i in range(L)],
                "strength": gamma_guess,
            }
        ]
    )

    opt_propagator = PropagatorWithGradients(
        sim_params=sim_params,
        hamiltonian=H_0,
        compact_noise_model=guess_noise_model,
        init_state=init_state,
    )

    loss = LossClass(
        ref_traj=ref_traj,
        traj_gradients=opt_propagator,
        working_dir=work_dir,
        print_to_file=True,
    )

    characterizer = Characterizer(
        traj_gradients=opt_propagator, init_guess=guess_noise_model, loss=loss
    )

    print("Optimizing ... ")
    characterizer.adam_optimize(max_iterations=50)
    print("Optimization completed.")
```
