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

    for _ in range(n):
        H_1()
        H_2()
        H_3()
        H_2()
        H_1()
    
    return circ
    