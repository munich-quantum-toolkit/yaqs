import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import qiskit.circuit
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import LineLattice, SquareLattice, BoundaryCondition
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.quantum_info import Operator

from mqt.yaqs.core.libraries.circuit_library import create_1D_Fermi_Hubbard_circuit, create_2D_Fermi_Hubbard_circuit

# FH model parameters
t = -1.0  # kinetic hopping
v = 0.5   # chemical potential
u = 2.0   # onsite interaction
(x, y) = (2,2) # lattice dimensions
dt = 0.1
timesteps = 1

# Define the circuit
num_sites = x*y
num_qubits = 2*num_sites
#num_trotter_steps = 10

dt_list = [10**(-x) for x in range(4)]
trotter_steps_list = [10**x for x in range(4)]

for dt in dt_list:
    error_list = []
    for num_trotter_steps in trotter_steps_list:
        circuit = qiskit.circuit.QuantumCircuit(num_qubits)

        model = {'name': '2D_Fermi_Hubbard', 'Lx': x, 'Ly': y, 'mu': v, 'u': u, 't': t, 'num_trotter_steps': num_trotter_steps}
        circuit = create_2D_Fermi_Hubbard_circuit(model, dt=dt, timesteps=timesteps)
        U_yaqs = Operator(circuit).to_matrix()

        # Qiskit implementation
        square_lattice = SquareLattice(rows=x, cols=y, boundary_condition=BoundaryCondition.OPEN)
        #line_lattice = LineLattice(num_nodes=x, boundary_condition=BoundaryCondition.OPEN)

        fh_hamiltonian = FermiHubbardModel(
            square_lattice.uniform_parameters(
                uniform_interaction=t,
                uniform_onsite_potential=v,
            ),
            onsite_interaction=u,
        )

        mapper = JordanWignerMapper()
        qubit_jw_op = mapper.map(fh_hamiltonian.second_q_op())
        H_qiskit = qubit_jw_op.to_matrix()
        U_qiskit = sp.linalg.expm(-1j*dt*timesteps*H_qiskit)

        # Calculate error
        error = np.linalg.norm(U_yaqs - U_qiskit, 2)
        error_list.append(error)
        print("Trotter error: " + str(error))

    print(error_list)
    label = "$\Delta t = $" + str(dt)
    plt.loglog(trotter_steps_list, error_list, linestyle='-', marker='s', label=label)

#plt.rcParams['text.usetex'] = True
plt.xlabel("number of Trotter steps $n$")
plt.ylabel("error")
plt.xscale("log", base=10)
plt.yscale("log",base=10)
plt.title("Convergence of the Trotter splitting error")
plt.legend()
plt.show()
