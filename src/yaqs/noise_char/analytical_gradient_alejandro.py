
#%%
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, PhysicsSimParams
from mqt.yaqs.Noise_characterization.optimization import *
from mqt.yaqs.Noise_characterization.propagation import *
from mqt.yaqs import Simulator
from dataclasses import dataclass

import time



#%%



# Generate reference trajectory
sim_params = SimulationParameters()

t, qt_ref_traj, A_kn_exp_vals=qutip_traj(sim_params)


#%%

# Perform gradient descent

initial_params = SimulationParameters()
initial_params.gamma_rel = 0.15
initial_params.gamma_deph = 0.13


loss_history, gr_history, gd_history, dJ_dgr_history, dJ_dgd_history = ADAM_gradient_descent(initial_params, qt_ref_traj, learning_rate=0.1, max_iterations=400)


# Plot 1: Logarithm of the loss history
plt.figure()
plt.plot(np.log(loss_history), label='log(J)')
plt.legend()
plt.grid(True)  # Add grid to the plot
plt.show()  # Display the first plot

# Plot 2: Gamma relaxation and dephasing with a reference line
plt.figure()
plt.plot(gr_history, label='gamma_relaxation')
plt.plot(gd_history, label='gamma_dephasing')
plt.axhline(y=0.1, color='r', linestyle='--', label='gamma_reference')
plt.legend()
plt.grid(True)  # Add grid to the plot
plt.show()  # Display the second plot


# %%

# %%
