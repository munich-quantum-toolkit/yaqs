import copy
import numpy as np
import opt_einsum as oe

from yaqs.general.data_structures.MPS import MPS
from yaqs.general.data_structures.noise_model import NoiseModel
from yaqs.general.operations.operations import scalar_product




def calculate_stochastic_factor(state: MPS) -> float:
    """
    Calculate the stochastic factor for the given state.

    This factor is used to determine if a quantum jump will occur during the stochastic evolution.

    Args:
        state (MPS): The Matrix Product State (MPS) representing the current state of the system.
                     Must be mixed canonical form at site 0 or B normalized

    Returns:
        float: The calculated stochastic factor.
    """
    return 1 - scalar_product(state, state, 0)


def create_probability_distribution(state: MPS, noise_model: NoiseModel, dt: float) -> dict:
    """
    Create a probability distribution for potential quantum jumps in the system.

    This function calculates the probability of each possible jump occurring, based on the noise model and time step.

    Args:
        state (MPS): The Matrix Product State (MPS) representing the current state of the system.
                     Must be in mixed canonical form at site 0 (not normalized).
        noise_model (NoiseModel): The noise model containing jump operators and their corresponding strengths.
        dt (float): The time step for the evolution.

    Returns:
        dict: A dictionary containing jump operators, their strengths, corresponding sites, and the calculated probabilities.
    """
    # Ordered as [Jump 0 Site 0, Jump 1 Site 0, Jump 0 Site 1, Jump 1 Site 1...]
    jump_dict = {'jumps': [], 'strengths': [], 'sites': [], 'probabilities': []}
    dp_m_list = []

    # Dissipative sweep should always result in a mixed canonical form at site L
    for site, _ in enumerate(state.tensors):
        # state.set_canonical_form(site)
        if site != 0 and site != state.length:
            state.shift_orthogonality_center_right(site-1)

        for j, jump_operator in enumerate(noise_model.jump_operators):
            jumped_state = copy.deepcopy(state)
            jumped_state.tensors[site] = oe.contract('ab, bcd->acd', jump_operator, state.tensors[site])

            dp_m = dt * noise_model.strengths[j] * scalar_product(jumped_state, jumped_state, site)
            dp_m_list.append(dp_m.real)
            jump_dict['jumps'].append(jump_operator)
            jump_dict['strengths'].append(noise_model.strengths[j])
            jump_dict['sites'].append(site)
    
    # Normalize the probabilities
    jump_dict['probabilities'] = (dp_m_list / np.sum(dp_m_list)).astype(float)
    return jump_dict


def stochastic_process(state: MPS, noise_model: NoiseModel, dt: float) -> MPS:
    """
    Perform a stochastic process on the given state.

    This function determines whether a quantum jump occurs and applies the corresponding jump operator if necessary.

    Args:
        previous_state (MPS): The previous Matrix Product State (MPS) before the current evolution step.
                              Must be site canonical at site 0 (following dissipative sweep).
        state (MPS): The current Matrix Product State (MPS).
        noise_model (NoiseModel): The noise model containing jump operators and their corresponding strengths.
        dt (float): The time step for the evolution.

    Returns:
        MPS: The updated state after performing the stochastic process.
    """
    dp = calculate_stochastic_factor(state)
    if np.random.rand() >= dp:
        # No jump
        # Replaces normalization since state should be in
        # mixed canonical form at site 0 from TDVP
        state.shift_orthogonality_center_left(0)
        # state.normalize('B')
        return state
    else:
        # Jump
        jump_dict = create_probability_distribution(state, noise_model, dt)
        choices = list(range(len(jump_dict['probabilities'])))
        choice = np.random.choice(choices, p=jump_dict['probabilities'])
        jump_operator = jump_dict['jumps'][choice]
        state.tensors[jump_dict['sites'][choice]] = oe.contract('ab, bcd->acd', jump_operator, state.tensors[jump_dict['sites'][choice]])
        state.normalize('B')
        return state
