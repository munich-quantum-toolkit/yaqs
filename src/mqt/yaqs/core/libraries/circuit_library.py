# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from qiskit.circuit import QuantumCircuit


def create_Ising_circuit(model, dt, timesteps):
    "H = J ZZ + g X"

    assert model["name"] == "Ising"

    # Angle on X rotation
    alpha = -2 * dt * model["g"]
    # Angle on ZZ rotation
    beta = -2 * dt * model["J"]

    circ = QuantumCircuit(model["L"])
    for _ in range(timesteps):
        for site in range(model["L"]):
            circ.rx(theta=alpha, qubit=site)

        for site in range(model["L"] // 2):
            circ.rzz(beta, qubit1=2 * site, qubit2=2 * site + 1)
            # circ.cx(control_qubit=2*site, target_qubit=2*site+1)
            # circ.rz(phi=beta, qubit=2*site+1)
            # circ.cx(control_qubit=2*site, target_qubit=2*site+1)

        for site in range(1, model["L"] // 2):
            circ.rzz(beta, qubit1=2 * site - 1, qubit2=2 * site)
            # circ.cx(control_qubit=2*site-1, target_qubit=2*site)
            # circ.rz(phi=beta, qubit=2*site)
            # circ.cx(control_qubit=2*site-1, target_qubit=2*site)

        if model["L"] % 2 != 0 and model["L"] != 1:
            circ.rzz(beta, qubit1=model["L"] - 2, qubit2=model["L"] - 1)
            # circ.cx(control_qubit=model['L']-2, target_qubit=model['L']-1)
            # circ.rz(phi=beta, qubit=model['L']-1)
            # circ.cx(control_qubit=model['L']-2, target_qubit=model['L']-1)

    return circ


def create_Heisenberg_circuit(model, dt, timesteps):
    "H = Jx XX + Jy YY + Jz ZZ + h Z"

    assert model["name"] == "Heisenberg"

    theta_xx = -2 * dt * model["Jx"]
    theta_yy = -2 * dt * model["Jy"]
    theta_zz = -2 * dt * model["Jz"]
    theta_z = -2 * dt * model["h"]

    circ = QuantumCircuit(model["L"])
    for _ in range(timesteps):
        # Z application
        for site in range(model["L"]):
            circ.rz(phi=theta_z, qubit=site)

        # ZZ application
        for site in range(model["L"] // 2):
            circ.rzz(theta=theta_zz, qubit1=2 * site, qubit2=2 * site + 1)

        for site in range(1, model["L"] // 2):
            circ.rzz(theta=theta_zz, qubit1=2 * site - 1, qubit2=2 * site)

        if model["L"] % 2 != 0 and model["L"] != 1:
            circ.rzz(theta=theta_zz, qubit1=model["L"] - 2, qubit2=model["L"] - 1)

        # if model['boundary'] == 'periodic' and model['L'] != 2:
        #     circ.rzz(theta=theta_zz, qubit1=0, qubit2=model['L']-1)

        # XX application
        for site in range(model["L"] // 2):
            circ.rxx(theta=theta_xx, qubit1=2 * site, qubit2=2 * site + 1)

        for site in range(1, model["L"] // 2):
            circ.rxx(theta=theta_xx, qubit1=2 * site - 1, qubit2=2 * site)

        if model["L"] % 2 != 0 and model["L"] != 1:
            circ.rxx(theta=theta_xx, qubit1=model["L"] - 2, qubit2=model["L"] - 1)

        # if model['boundary'] == 'periodic' and model['L'] != 2:
        #     circ.rxx(theta=theta_zz, qubit1=0, qubit2=model['L']-1)

        # YY application
        for site in range(model["L"] // 2):
            circ.ryy(theta=theta_yy, qubit1=2 * site, qubit2=2 * site + 1)

        for site in range(1, model["L"] // 2):
            circ.ryy(theta=theta_yy, qubit1=2 * site - 1, qubit2=2 * site)

        if model["L"] % 2 != 0 and model["L"] != 1:
            circ.ryy(theta=theta_yy, qubit1=model["L"] - 2, qubit2=model["L"] - 1)

        # if model['boundary'] == 'periodic' and model['L'] != 2:
        #     circ.ryy(theta=theta_zz, qubit1=0, qubit2=model['L']-1)

    # print(circ.draw())
    return circ
