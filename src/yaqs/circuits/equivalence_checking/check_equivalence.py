from qiskit.converters import dag_to_circuit, circuit_to_dag
import numpy as np
import opt_einsum as oe

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from yaqs.general.data_structures.MPO import MPO


def decompose_theta(theta, threshold):
    dims = theta.shape
    theta = np.transpose(theta, (0, 3, 2, 1, 4, 5))
    theta_matrix = np.reshape(theta, (dims[0]*dims[1]*dims[2], dims[3]*dims[4]*dims[5]))

    # SVD with truncation
    U, S_list, V = np.linalg.svd(theta_matrix, full_matrices=False)
    S_list = S_list[S_list > threshold]
    U = U[:, 0:len(S_list)]
    V = V[0:len(S_list), :]

    # Create site tensors
    U = np.reshape(U, (dims[0], dims[1], dims[2], len(S_list)))
    U = np.transpose(U, (0, 2, 1, 3))
    M = np.diag(S_list) @ V
    M = np.reshape(M, (len(S_list), dims[3], dims[4], dims[5]))
    M = np.transpose(M, (1, 0, 2, 3))

    return U, M


def update_MPO(MPO, dag1, dag2, qubits, threshold):
    n = qubits[0]

    # print("UPDATE", n, n+1)
    # TODO: Update to new indices
    theta = oe.contract('ijkl, mlno->imjkno', MPO.tensors[n], MPO.tensors[n+1])
    theta = apply_causal_cone(theta, dag1, qubits)
    theta = apply_causal_cone(theta, dag2, qubits, conjugate=True)
    MPO.tensors[n], MPO.tensors[n+1] = decompose_theta(theta, threshold)


def apply_layer(MPO, circuit1_dag, circuit2_dag, first_iterator, second_iterator, threshold):
    for n in first_iterator:
        # print(n, n+1)
        update_MPO(MPO, circuit1_dag, circuit2_dag, [n, n+1], threshold)

    for n in second_iterator:
        # print(n, n+1)
        update_MPO(MPO, circuit1_dag, circuit2_dag, [n, n+1], threshold)


def check_longest_gate(DAG_circuit):
    largest_distance = 1
    for layer in DAG_circuit.layers():
        first_layer = layer
        break

    if 'first_layer' in locals():
        layer_circuit = dag_to_circuit(first_layer['graph'])

        for gate in layer_circuit.data:
            if gate.operation.num_qubits > 1:
                distance = np.abs(gate.qubits[0]._index - gate.qubits[-1]._index)+1
                if distance > largest_distance:
                    largest_distance = distance

    return largest_distance


def select_starting_point(N, DAG_circuit):
    assert N > 0

    for layer in DAG_circuit.layers():
        first_layer = layer
        break

    first_iterator = range(0, N-1, 2)
    second_iterator = range(1, N-1, 2)
    odd = False
    if 'first_layer' in locals():
        layer_circuit = dag_to_circuit(first_layer['graph'])
        for i, gate in enumerate(layer_circuit.data):
            if gate.operation.num_qubits == 2:
                if i % 2 != 0:
                    odd = True
                break

        if odd:
            first_iterator = range(1, N-1, 2)
            second_iterator = range(0, N-1, 2)

    return first_iterator, second_iterator


def iterate(mpo: MPO, dag1, dag2, threshold):
    # Ensures we start at a two-qubit gate (Adds performance boost)
    # Loop while nodes are removed from DAG
    N = mpo.length
    if dag1.op_nodes():
        first_iterator, second_iterator = select_starting_point(N, dag1)
    else:
        first_iterator, second_iterator = select_starting_point(N, dag2)

    while dag1.op_nodes() or dag2.op_nodes():
        # TODO: Generalize to dag1 or dag2
        largest_distance1  = check_longest_gate(dag1)
        largest_distance2  = check_longest_gate(dag2)

        if largest_distance1 in [1, 2] and largest_distance2 in [1, 2]:
            apply_layer(mpo, dag1, dag2, first_iterator, second_iterator, threshold)
        else:
            if largest_distance1 >= largest_distance2:
                conjugate = False
            else:
                conjugate = True
            apply_long_range_layer(mpo, dag1, dag2, conjugate, threshold)
    return mpo


def run(circuit1, circuit2, threshold: float=1e-13, fidelity: float=1-1e-13):
    # Initialization
    assert circuit1.num_qubits == circuit2.num_qubits
    N = circuit1.num_qubits
    mpo = MPO()
    mpo.init_identity(circuit1.num_qubits)

    circuit1_dag = circuit_to_dag(circuit1)
    circuit2_dag = circuit_to_dag(circuit2)
    MPO = iterate(MPO, circuit1_dag, circuit2_dag, threshold)

    # return check_if_identity(MPO, fidelity)