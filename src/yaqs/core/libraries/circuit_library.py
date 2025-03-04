from qiskit.circuit import QuantumCircuit


def create_Ising_circuit(model, dt, timesteps, order=1):
    'H = J ZZ + g X'

    assert model['name'] == 'Ising'
    L = model['L']
    circ = QuantumCircuit(L)

    for _ in range(timesteps):
        if order == 1:
            # First-order Trotterization: full X rotation then ZZ interactions.
            for site in range(L):
                circ.rx(theta=-2*dt*model['g'], qubit=site)
            # Even bonds:
            for site in range(L//2):
                circ.rzz(-2*dt*model['J'], qubit1=2*site, qubit2=2*site+1)
            # Odd bonds:
            for site in range(1, L//2):
                circ.rzz(-2*dt*model['J'], qubit1=2*site-1, qubit2=2*site)
            if L % 2 != 0 and L != 1:
                circ.rzz(-2*dt*model['J'], qubit1=L-2, qubit2=L-1)

        elif order == 2:
            # Second-order Trotterization: half-step X, then full ZZ, then half-step X.
            half_alpha = -2*dt/2*model['g']
            # First half-step for X rotations:
            for site in range(L):
                circ.rx(theta=half_alpha, qubit=site)
            # Full-step ZZ interactions (same as before):
            for site in range(L//2):
                circ.rzz(-2*dt*model['J'], qubit1=2*site, qubit2=2*site+1)
            for site in range(1, L//2):
                circ.rzz(-2*dt*model['J'], qubit1=2*site - 1, qubit2=2*site)
            if L % 2 != 0 and L != 1:
                circ.rzz(-2*dt*model['J'], qubit1=L-2, qubit2=L-1)
            # Second half-step for X rotations:
            for site in range(L):
                circ.rx(theta=half_alpha, qubit=site)

        else:
            raise ValueError("Unsupported Trotterization order. Choose 1 or 2.")

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
