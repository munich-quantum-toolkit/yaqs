import numpy as np
import matplotlib.pyplot as plt
import qutip as qt


from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, PhysicsSimParams
from mqt.yaqs import simulator

# Build Hamiltonian
def qt_build_XXZ_operator(delta, L):
    """
    Constructs the XXZ Hamiltonian for a chain of L spins according to PhysRevLett.107.137201.
    """
    sp = qt.sigmap()
    sm = qt.sigmam()
    sz = qt.sigmaz()
    I = qt.qeye(2)
    
    H = 0
    for j in range(L - 1):
        sp_sm = qt.tensor([sp if n == j else sm if n == j + 1 else I for n in range(L)])
        sm_sp = qt.tensor([sm if n == j else sp if n == j + 1 else I for n in range(L)])
        sz_sz = qt.tensor([sz if n == j or n == j + 1 else I for n in range(L)])
        H += 2 * (sp_sm + sm_sp) + delta * sz_sz
    return H

def qt_build_lindblad_operators(L, epsilon):
    """
    Constructs the Lindblad operators for the XXZ chain.
    L: Number of spins
    epsilon: Coupling strength
    """
    sp = qt.sigmap()
    sm = qt.sigmam()
    I = qt.qeye(2)

    c_ops = []
    
    # L1 = sqrt(ε) * σ_1^+
    c_ops.append(np.sqrt(epsilon) * qt.tensor([sp] + [I] * (L - 1)))

    # L2 = sqrt(ε) * σ_n^-
    c_ops.append(np.sqrt(epsilon) * qt.tensor([I] * (L - 1) + [sm]))

    return c_ops


# analytical steady state for the XXZ chain, epsilon >> 
def steady_state(L):
    """
    Computes the exact steady state for the XXZ chain.
    """
    sz_exact = [np.cos(np.pi * (j-1) / (L - 1)) for j in range(1,L+1)]
    return np.array(sz_exact)


if __name__ == "__main__":

    # Define parameters
    L = 8
    J_x = 1
    J_y = 1
    J_z = 1
    g = 0
    delta = 1
    epsilon = 40  # coupling strength (ε)

    T= 4
    dt = 0.1
    t = np.arange(0, T + dt, dt)



    # Qutip setup
    H_qt = qt_build_XXZ_operator(delta, L)
    c_ops = []
    c_ops = qt_build_lindblad_operators(L, epsilon)


    # psi0 = qt.rand_ket(2**L) # random initial state

    zero = qt.basis(2, 0)  # single-qubit |0⟩
    one = qt.basis(2, 1) # single-qubit |1⟩


    # psi0 = qt.tensor([zero] * L)  # all zero state

    # wall state: half |1⟩ and half |0⟩
    half = L // 2
    state_list = [one]*half + [zero]*(L - half)
    psi0 = qt.tensor(state_list)

    psi0.dims = [[2]*L, [1]]  # Matches the L-qubit tensor structure

    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()



    sz_list = [qt.tensor([sz if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]

    # Lindblad solution
    result_lindblad = qt.mesolve(H_qt, psi0, t, c_ops, sz_list, progress_bar=True)


    # analytical steady state
    steadystate_exact = steady_state(L)


    # TJM setup
    H_0 = MPO()
    H_0.init_heisenberg(L, J_x, J_y, J_z, g)

    # Define the initial state
    state = MPS(L, state='wall')

    # Define the simulation parameters
    sample_timesteps = True
    N = 100
    max_bond_dim = 16
    threshold = 1e-6
    order = 2
    measurements = [Observable(Z(), site) for site in range(L)]
    gammas = [0.1,0.2]

    noise_model = NoiseModel([['relaxation', 'excitation'] for _ in range(L)], [[gammas[0], gammas[1]] for _ in range(L)])
    sim_params = PhysicsSimParams(measurements, T, dt, N, max_bond_dim, threshold, order, sample_timesteps=sample_timesteps)

    ########## TJM Example #################

    simulator.run(state, H_0, sim_params, noise_model)


    # qubit ordering is reversed in qutip, so we need to reverse the expectation values
    reversed_lindblad_expect = result_lindblad.expect[::-1]

    # plot the results
    plt.figure(figsize=(10, 6))
    plt.title('TJM Simulation Results')
    plt.xlabel('Time')
    plt.ylabel('Observable Expectation Values')
    for i, observable in enumerate(sim_params.observables):
        plt.plot(t, observable.results, label=f'⟨{observable.gate.name} {observable.site}⟩ TJM', color='orange')
        plt.axhline(y=steadystate_exact[i], linestyle='--', color='gray', label=f'⟨Z_{i}⟩ (exact steady state)' if i == 0 else None)
        plt.plot(t, result_lindblad.expect[i], label=f'⟨Z_{i}⟩ QT', color='blue')
        # difference = reversed_lindblad_expect[i] - observable.results
        # plt.plot(t, difference, label=f'Difference ⟨Z_{i}⟩ - ⟨Z_{i}⟩_exact', linestyle='--', color='red')
    plt.legend()
    plt.grid()
    plt.show()


