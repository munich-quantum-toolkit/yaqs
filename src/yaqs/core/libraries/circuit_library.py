from qiskit.circuit import QuantumCircuit, QuantumRegister


def create_Ising_circuit(model, dt, timesteps):
    'H = J ZZ + g X'

    assert model['name'] == 'Ising'

    # Angle on X rotation
    alpha = -2*dt*model['g']
    # Angle on ZZ rotation
    beta = -2*dt*model['J']

    circ = QuantumCircuit(model['L'])
    for _ in range(timesteps):
        for site in range(model['L']):
            circ.rx(theta=alpha, qubit=site)

        for site in range(model['L'] // 2):
            circ.cx(control_qubit=2*site, target_qubit=2*site+1)
            circ.rz(phi=beta, qubit=2*site+1)
            circ.cx(control_qubit=2*site, target_qubit=2*site+1)

        for site in range(1, model['L'] // 2):
            circ.cx(control_qubit=2*site-1, target_qubit=2*site)
            circ.rz(phi=beta, qubit=2*site)
            circ.cx(control_qubit=2*site-1, target_qubit=2*site)

        if model['L'] % 2 != 0 and model['L'] != 1:
            circ.cx(control_qubit=model['L']-2, target_qubit=model['L']-1)
            circ.rz(phi=beta, qubit=model['L']-1)
            circ.cx(control_qubit=model['L']-2, target_qubit=model['L']-1)

    return circ


def create_Heisenberg_circuit(model, dt, timesteps):
    'H = Jx XX + Jy YY + Jz ZZ + h Z'

    assert model['name'] == 'Heisenberg'

    theta_xx = -2*dt*model['Jx']
    theta_yy = -2*dt*model['Jy']
    theta_zz = -2*dt*model['Jz']
    theta_z = -2*dt*model['h']

    circ = QuantumCircuit(model['L'])
    for _ in range(timesteps):
        # Z application
        for site in range(model['L']):
            circ.rz(phi=theta_z, qubit=site)

        # ZZ application
        for site in range(model['L'] // 2):
            circ.rzz(theta=theta_zz, qubit1=2*site, qubit2=2*site+1)

        for site in range(1, model['L'] // 2):
            circ.rzz(theta=theta_zz, qubit1=2*site-1, qubit2=2*site)

        if model['L'] % 2 != 0 and model['L'] != 1:
            circ.rzz(theta=theta_zz, qubit1=model['L']-2, qubit2=model['L']-1)

        # if model['boundary'] == 'periodic' and model['L'] != 2:
        #     circ.rzz(theta=theta_zz, qubit1=0, qubit2=model['L']-1)

        # XX application
        for site in range(model['L'] // 2):
            circ.rxx(theta=theta_xx, qubit1=2*site, qubit2=2*site+1)

        for site in range(1, model['L'] // 2):
            circ.rxx(theta=theta_xx, qubit1=2*site-1, qubit2=2*site)

        if model['L'] % 2 != 0 and model['L'] != 1:
            circ.rxx(theta=theta_xx, qubit1=model['L']-2, qubit2=model['L']-1)

        # if model['boundary'] == 'periodic' and model['L'] != 2:
        #     circ.rxx(theta=theta_zz, qubit1=0, qubit2=model['L']-1)

        # YY application
        for site in range(model['L'] // 2):
            circ.ryy(theta=theta_yy, qubit1=2*site, qubit2=2*site+1)

        for site in range(1, model['L'] // 2):
            circ.ryy(theta=theta_yy, qubit1=2*site-1, qubit2=2*site)

        if model['L'] % 2 != 0 and model['L'] != 1:
            circ.ryy(theta=theta_yy, qubit1=model['L']-2, qubit2=model['L']-1)

        # if model['boundary'] == 'periodic' and model['L'] != 2:
        #     circ.ryy(theta=theta_zz, qubit1=0, qubit2=model['L']-1)

    # print(circ.draw())
    return circ


def create_1D_Fermi_Hubbard_circuit(model, dt, timesteps):
    'H = -1/2 mu (I-Z) + 1/4 u (I-Z) (I-Z) - 1/2 t (XX + YY)'

    assert model['name'] == '1D_Fermi_Hubbard'

    mu = model['mu']
    u = model['u']
    t = model['t']
    n = model['num_trotter_steps']

    spin_up = QuantumRegister(model['L'], '↑')
    spin_down = QuantumRegister(model['L'], '↓')
    circ = QuantumCircuit(spin_up, spin_down)

    def H_1():
        """Add the time evolution of the chemical potential term"""
        theta = mu*dt/(2*n)
        for j in range(model['L']):
            circ.p(theta=theta, qubit=spin_up[j])
            circ.p(theta=theta, qubit=spin_down[j])

    def H_2():
        """Add the time evolution of the onsite interaction term"""
        theta = -u*dt/(2*n)
        for j in range(model['L']):
            circ.cp(theta=theta, control_qubit=spin_up[j], target_qubit=spin_down[j])

    def H_3():
        """Add the time evolution of the kinetic hopping term"""
        theta = -dt*t/n
        for j in range(model['L']-1):
            if j % 2 == 0:
                circ.rxx(theta=theta, qubit1=spin_up[j+1], qubit2=spin_up[j])
                circ.ryy(theta=theta, qubit1=spin_up[j+1], qubit2=spin_up[j])
                circ.rxx(theta=theta, qubit1=spin_down[j+1], qubit2=spin_down[j])
                circ.ryy(theta=theta, qubit1=spin_down[j+1], qubit2=spin_down[j])
        for j in range(model['L']-1):
            if j % 2 != 0:
                circ.rxx(theta=theta, qubit1=spin_up[j+1], qubit2=spin_up[j])
                circ.ryy(theta=theta, qubit1=spin_up[j+1], qubit2=spin_up[j])
                circ.rxx(theta=theta, qubit1=spin_down[j+1], qubit2=spin_down[j])
                circ.ryy(theta=theta, qubit1=spin_down[j+1], qubit2=spin_down[j])

    for _ in range(n*timesteps):
        H_1()
        H_2()
        H_3()
        H_2()
        H_1()
    
    return circ


def lookup_qiskit_ordering(particle, spin) -> int:
    """
    Looks up the Qiskit mapping from a 2D lattice to a 1D qubit-line
    
    Parameters:
        particle (int): The index of the particle in the physical lattice.
        spin (int): 0 for spin up, 1 for spin down.
    
    Returns:
        int: The index in the 1D qubit-line.
    """
    if spin == '↑':
        spin = 0
    elif spin == '↓':
        spin = 1

    if spin not in (0, 1):
        raise ValueError("spin must be 0 or 1")
    
    return 2*particle + spin


def add_long_range_interaction(circ, i, j, outer_op, alpha):
    """
    Add a long range interaction that is decomposed into two-qubit gates.
    outer_op=X: X_i ⊗ Z_{i+1} ⊗ ... ⊗ Z_{j-1} ⊗ X_j
    outer_op=Y: Y_i ⊗ Z_{i+1} ⊗ ... ⊗ Y_{j-1} ⊗ X_j
    """
    import numpy as np
    
    if i >= j:
        raise Exception("Assumption i < j violated.")
    
    phi = 1*alpha
    circ.rz(phi=phi, qubit=j)

    for k in range(i, j):
        # prepend the CNOT gate
        aux_circ = QuantumCircuit(circ.num_qubits)
        aux_circ.cx(control_qubit=k, target_qubit=j)
        circ.compose(aux_circ, front=True, inplace=True)
        # append the CNOT gate
        circ.cx(control_qubit=k, target_qubit=j)
    if outer_op == 'x' or outer_op == 'X':
        theta = np.pi/2
        # prepend the Ry gates
        aux_circ = QuantumCircuit(circ.num_qubits)
        aux_circ.ry(theta=theta, qubit=i)
        aux_circ.ry(theta=theta, qubit=j)
        circ.compose(aux_circ, front=True, inplace=True)
        # append the same Ry gates with negative phase
        circ.ry(theta=-theta, qubit=i)
        circ.ry(theta=-theta, qubit=j)
    elif outer_op == 'y' or outer_op == 'Y':
        theta = np.pi/2
        # prepend the Rx gates
        aux_circ = QuantumCircuit(circ.num_qubits)
        aux_circ.rx(theta=theta, qubit=i)
        aux_circ.rx(theta=theta, qubit=j)
        circ.compose(aux_circ, front=True, inplace=True)
        # append the same Rx gates with negative phase
        circ.rx(theta=-theta, qubit=i)
        circ.rx(theta=-theta, qubit=j)
    else:
        raise Exception("Only Pauli X or Y matrices are supported as outer operator.")
    

def add_hopping_term(circ, i, j, alpha):
    """
    Adds a hopping operator of the form
    exp(-i*(X_i ⊗ Z_{i+1} ⊗ ... ⊗ Z_{j-1} ⊗ X_j + Y_i ⊗ Z_{i+1} ⊗ ... ⊗ Z_{j-1} ⊗ Y_j))
    to the circuit.
    """
    XX = QuantumCircuit(circ.num_qubits)
    YY = QuantumCircuit(circ.num_qubits)
    add_long_range_interaction(XX, i, j, 'X', alpha)
    add_long_range_interaction(YY, i, j, 'Y', alpha)
    circ.compose(XX, inplace=True)
    circ.compose(YY, inplace=True)


def create_2D_Fermi_Hubbard_circuit(model, dt, timesteps):
    'H = -1/2 mu (I-Z) + 1/4 u (I-Z) (I-Z) - 1/2 t (XX + YY)'

    assert model['name'] == '2D_Fermi_Hubbard'

    mu = model['mu']
    u = model['u']
    t = model['t']
    n = model['num_trotter_steps']
    L = model['Lx'] * model['Ly']
    N = 2*L

    circ = QuantumCircuit(N)

    def H_1():
        """Add the time evolution of the chemical potential term"""
        theta = -mu*dt/(2*n)
        for j in range(L):
            q_up = lookup_qiskit_ordering(j, '↑')
            q_down = lookup_qiskit_ordering(j, '↓')
            circ.p(theta=theta, qubit=q_up)
            circ.p(theta=theta, qubit=q_down)

    def H_2():
        """Add the time evolution of the onsite interaction term"""
        theta = -u*dt/(2*n)
        for j in range(L):
            q_up = lookup_qiskit_ordering(j, '↑')
            q_down = lookup_qiskit_ordering(j, '↓')
            circ.cp(theta=theta, control_qubit=q_up, target_qubit=q_down)

    def H_3():
        """Add the time evolution of the kinetic hopping term"""
        alpha = t*dt/n

        def horizontal_odd():
            for y in range(model['Ly']):
                for x in range(model['Lx']-1):
                    if x % 2 == 0:
                        p1 = y*model['Lx'] + x
                        p2 = p1 + 1
                        #print("particle (" + str(p1) + ", " + str(p2) + ")")
                        q1_up = lookup_qiskit_ordering(p1, '↑')
                        q2_up = lookup_qiskit_ordering(p2, '↑')
                        q1_down = lookup_qiskit_ordering(p1, '↓')
                        q2_down = lookup_qiskit_ordering(p2, '↓')
                        add_hopping_term(circ, q1_up, q2_up, alpha)
                        add_hopping_term(circ, q1_down, q2_down, alpha)
                        #print("qubit ↑ (" + str(q1_up) + ", " + str(q2_up) + ")")
                        #print("qubit ↓ (" + str(q1_down) + ", " + str(q2_down) + ")")
                        #print("--")

        def horizontal_even():
            for y in range(model['Ly']):
                for x in range(model['Lx']-1):
                    if x % 2 != 0:
                        p1 = y*model['Lx'] + x
                        p2 = p1 + 1
                        #print("particle (" + str(p1) + ", " + str(p2) + ")")
                        q1_up = lookup_qiskit_ordering(p1, '↑')
                        q2_up = lookup_qiskit_ordering(p2, '↑')
                        q1_down = lookup_qiskit_ordering(p1, '↓')
                        q2_down = lookup_qiskit_ordering(p2, '↓')
                        add_hopping_term(circ, q1_up, q2_up, alpha)
                        add_hopping_term(circ, q1_down, q2_down, alpha)
                        #print("qubit ↑ (" + str(q1_up) + ", " + str(q2_up) + ")")
                        #print("qubit ↓ (" + str(q1_down) + ", " + str(q2_down) + ")")
                        #print("--")

        def vertical_odd():
            for y in range(model['Ly']-1):
                if y % 2 == 0:
                    for x in range(model['Lx']):
                        p1 = y*model['Lx'] + x
                        p2 = p1 + model['Lx']
                        #print("particle (" + str(p1) + ", " + str(p2) + ")")
                        q1_up = lookup_qiskit_ordering(p1, '↑')
                        q2_up = lookup_qiskit_ordering(p2, '↑')
                        q1_down = lookup_qiskit_ordering(p1, '↓')
                        q2_down = lookup_qiskit_ordering(p2, '↓')
                        add_hopping_term(circ, q1_up, q2_up, alpha)
                        add_hopping_term(circ, q1_down, q2_down, alpha)
                        #print("qubit ↑ (" + str(q1_up) + ", " + str(q2_up) + ")")
                        #print("qubit ↓ (" + str(q1_down) + ", " + str(q2_down) + ")")
                        #print("--")  

        def vertical_even():
            for y in range(model['Ly']-1):
                if y % 2 != 0:
                    for x in range(model['Lx']):
                        p1 = y*model['Lx'] + x
                        p2 = p1 + model['Lx']
                        #print("particle (" + str(p1) + ", " + str(p2) + ")")
                        q1_up = lookup_qiskit_ordering(p1, '↑')
                        q2_up = lookup_qiskit_ordering(p2, '↑')
                        q1_down = lookup_qiskit_ordering(p1, '↓')
                        q2_down = lookup_qiskit_ordering(p2, '↓')
                        add_hopping_term(circ, q1_up, q2_up, alpha)
                        add_hopping_term(circ, q1_down, q2_down, alpha)
                        #print("qubit ↑ (" + str(q1_up) + ", " + str(q2_up) + ")")
                        #print("qubit ↓ (" + str(q1_down) + ", " + str(q2_down) + ")")
                        #print("--")
        
        #for _ in range(n):
        horizontal_odd()
        horizontal_even()
        vertical_odd()
        vertical_even()

    for _ in range(timesteps):
        for _ in range(n):
            H_1()
            H_2()
            H_3()
            H_2()
            H_1()
    
    return circ
    