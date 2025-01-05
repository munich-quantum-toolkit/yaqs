import numpy as np
import opt_einsum as oe
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.dagcircuit.dagnode import DAGOpNode
import time

from yaqs.general.data_structures.MPO import MPO
from yaqs.general.libraries.tensor_library import TensorLibrary

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from qiskit.circuit.quantumcircuit import QuantumCircuit
    from qiskit._accelerate.circuit import DAGCircuit
    from yaqs.general.data_structures.MPO import MPO
    from yaqs.general.libraries.tensor_library import TensorLibrary



def decompose_theta(theta: np.ndarray, threshold: float):
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
    M = np.diag(S_list) @ V
    M = np.reshape(M, (len(S_list), dims[3], dims[4], dims[5]))
    M = np.transpose(M, (1, 2, 0, 3))

    return U, M


def apply_gate(gate: 'TensorLibrary', theta: np.ndarray, site0: int, site1: int, conjugate: bool=False):
    assert (gate.interaction == 1 or gate.interaction == 2)

    if gate.interaction == 1:
        assert gate.sites[0] in [site0, site1]
    elif gate.interaction == 2:
        assert gate.sites[0] in [site0, site1] and gate.sites[1] in [site0, site1]

    # Application from nearest-neighbor gates
    if theta.ndim == 6:
        if conjugate:
            theta = np.transpose(theta, (3, 4, 2, 0, 1, 5))

        if gate.name == "I":
            theta = theta
        elif gate.interaction == 1:
            if gate.sites[0] == site0:
                if conjugate:
                    theta = oe.contract('ij, jklmno->iklmno', np.conj(gate.tensor), theta)
                else:
                    theta = oe.contract('ij, jklmno->iklmno', gate.tensor, theta)
            elif gate.sites[0] == site1:
                if conjugate:
                    theta = oe.contract('ij, kjlmno->kilmno', np.conj(gate.tensor), theta)
                else:
                    theta = oe.contract('ij, kjlmno->kilmno', gate.tensor, theta)
        elif gate.interaction == 2:
            if conjugate:
                theta = oe.contract('ijkl, klmnop->ijmnop', np.conj(gate.tensor), theta)
            else:
                theta = oe.contract('ijkl, klmnop->ijmnop', gate.tensor, theta)

        if conjugate:
            theta = np.transpose(theta, (3, 4, 2, 0, 1, 5))

    # Application from long-range gates
    elif theta.ndim == 8:
        if conjugate:
            theta = np.transpose(theta, (4, 5, 3, 2, 0, 1, 6, 7))
        if gate.name == "I":
            theta = theta
        elif gate.interaction == 1:
            if gate.sites[0] == site0:
                if conjugate:
                    theta = oe.contract('ab, bcdefghi->acdefghi', np.conj(gate.tensor), theta)
                else:
                    theta = oe.contract('ab, bcdefghi->acdefghi', gate.tensor, theta)
            elif gate.sites[0] == site1:
                if conjugate:
                    theta = oe.contract('ab, cbdefghi->cadefghi', np.conj(gate.tensor), theta)
                else:
                    theta = oe.contract('ab, cbdefghi->cadefghi', gate.tensor, theta)

        elif gate.interaction == 2:
            if conjugate:
                theta = oe.contract('abcd, cdefghij->abefghij', np.conj(gate.tensor), theta)
            else:
                theta = oe.contract('abcd, cdefghij->abefghij', gate.tensor, theta)

        if conjugate:
            theta = np.transpose(theta, (4, 5, 3, 2, 0, 1, 6, 7))

    return theta


def convert_dag_to_tensor_algorithm(dag: 'DAGCircuit'):
    algorithm = []

    # Check if single node
    if type(dag) == DAGOpNode:
        gate = dag
        name = gate.op.name

        attr = getattr(TensorLibrary, name)
        gate_object = attr()
        if gate.op.params:
            gate_object.set_params(gate.op.params)

        sites = [gate.qargs[0]._index]
        if len(gate.qargs) == 2:
            sites.append(gate.qargs[1]._index)
        if len(gate.qargs) == 3:
            sites.append(gate.qargs[1]._index)
            sites.append(gate.qargs[2]._index)

        gate_object.set_sites(*sites)

        algorithm.append(gate_object)
    else:
        for gate in dag.op_nodes():
            name = gate.op.name

            if name == 'measure' or name == 'barrier':
                continue
            attr = getattr(TensorLibrary, name)
            gate_object = attr()
            if gate.op.params:
                gate_object.set_params(gate.op.params)

            # TODO: Cleanup ifs
            sites = [gate.qargs[0]._index]
            if len(gate.qargs) == 2:
                sites.append(gate.qargs[1]._index)
            if len(gate.qargs) == 3:
                sites.append(gate.qargs[1]._index)
                sites.append(gate.qargs[2]._index)

            gate_object.set_sites(*sites)

            algorithm.append(gate_object)

    return algorithm


def get_causal_cone(dag: 'DAGCircuit', qubits: list[int]):
    new_dag = dag.copy_empty_like()
    layers = list(dag.multigraph_layers())
    qubits_to_check = set()
    for qubit in range(min(qubits), max(qubits)+1):
        qubits_to_check.add(dag.qubits[qubit])

    for layer in layers:
        for node in layer:
            if isinstance(node, DAGOpNode):
                qubit_set = set(node.qargs)

                # Gate is entirely within cone
                if qubit_set <= qubits_to_check:
                    if node.op.name == 'measure' or node.op.name == 'barrier':
                        dag.remove_op_node(node)
                        continue
                    new_dag.apply_operation_back(node.op, node.qargs)
                    dag.remove_op_node(node)
                else:
                    if node.op.name == 'measure' or node.op.name == 'barrier':
                        dag.remove_op_node(node)
                        continue
                    # If not a subset, but there is overlap
                    for item in qubit_set & qubits_to_check:
                        qubits_to_check.remove(item)

        # Cone has ended for all
        if len(qubits_to_check) == 0:
            break

    return new_dag


def apply_causal_cone(theta: np.ndarray, dag: 'DAGCircuit', qubits: list[int], conjugate: bool=False):
    n = qubits[0]
    if dag.op_nodes():
        causal_circuit = get_causal_cone(dag, [n, n+1])
        tensor_circuit = convert_dag_to_tensor_algorithm(causal_circuit)
        for gate in tensor_circuit:
            theta = apply_gate(gate, theta, n, n+1, conjugate)
    return theta


def update_MPO(MPO: 'MPO', dag1: 'DAGCircuit', dag2: 'DAGCircuit', qubits: list[int], threshold: float):
    n = qubits[0]

    # TODO: Change order to remove transpose in decompose_theta?
    theta = oe.contract('abcd, efdg->aecbfg', MPO.tensors[n], MPO.tensors[n+1])
    # theta = oe.contract('ijkl, mlno->imjkno', MPO.tensors[n], MPO.tensors[n+1])
    theta = apply_causal_cone(theta, dag1, qubits)
    theta = apply_causal_cone(theta, dag2, qubits, conjugate=True)
    MPO.tensors[n], MPO.tensors[n+1] = decompose_theta(theta, threshold)


def apply_layer(MPO: 'MPO', circuit1_dag: 'DAGCircuit', circuit2_dag: 'DAGCircuit', first_iterator: range, second_iterator: range, threshold: float):
    for n in first_iterator:
        # print(n, n+1)
        update_MPO(MPO, circuit1_dag, circuit2_dag, [n, n+1], threshold)

    for n in second_iterator:
        # print(n, n+1)
        update_MPO(MPO, circuit1_dag, circuit2_dag, [n, n+1], threshold)


def apply_long_range_layer(mpo: 'MPO', dag1: 'DAGCircuit', dag2: 'DAGCircuit', conjugate: bool, threshold: float):
    # TODO: MPO on both sides
    # Identify gate and its position
    if not conjugate:
        dag_to_search = dag1
    else:
        dag_to_search = dag2

    for layer in dag_to_search.layers():
        first_layer = layer
        break

    if 'first_layer' in locals():
        layer_circuit = dag_to_circuit(first_layer['graph'])
        for gate in layer_circuit.data:
            if gate.operation.num_qubits > 1:
                distance = np.abs(gate.qubits[0]._index - gate.qubits[-1]._index)+1
                if distance > 2:
                    # Save location
                    location = min(gate.qubits[0]._index, gate.qubits[-1]._index)
                    # Create gate MPO
                    if not conjugate:
                        for node in dag1.op_nodes():
                            # Guarantees the correct node is removed
                            if node.name == gate.operation.name and len(node.qargs) >= 2 and node.qargs[0]._index == gate.qubits[0]._index and node.qargs[1]._index == gate.qubits[1]._index:
                                gate_MPO = convert_dag_to_tensor_algorithm(node)[0].mpo
                                # Remove from dag
                                dag1.remove_op_node(node)
                                break
                    else:
                        for node in dag2.op_nodes():
                            # Guarantees the correct node is removed
                            if node.name == gate.operation.name and len(node.qargs) >= 2 and node.qargs[0]._index == gate.qubits[0]._index and node.qargs[1]._index == gate.qubits[1]._index:
                                gate_MPO = convert_dag_to_tensor_algorithm(node)[0].mpo
                                gate_MPO.rotate(conjugate=True)
                                # Remove from dag
                                dag2.remove_op_node(node) 
                                break
                    break

    assert 'gate_MPO' in locals()

    assert gate_MPO.length <= mpo.length
    # MPO_2 must be the larger MPO
    if gate_MPO.length == mpo.length:
        sites = range(0, mpo.length)
    else:
        sites = range(location, location+distance)

    # TODO: Only contract first site with gate, then just SVD across chain to second gate tensor
    for site_gate_MPO, overall_site in enumerate(sites):
        # Even sites of gate MPO
        if site_gate_MPO != len(sites)-1 and site_gate_MPO % 2 == 0:
            # TODO: Could be sped up without putting all tensors together
            # TODO: Remove all transposes from new index order
            # sigma i, sigma i+1, upper bond gate, upper bond MPO, sigma' i, sigma' i+1, lower bond gate, lower bond MPO
            if not conjugate:
                tensor1 = np.transpose(gate_MPO.tensors[site_gate_MPO], (0, 2, 1, 3))
                tensor2 = np.transpose(gate_MPO.tensors[site_gate_MPO+1], (0, 2, 1, 3))
                tensor3 = np.transpose(mpo.tensors[overall_site], (0, 2, 1, 3))
                tensor4 = np.transpose(mpo.tensors[overall_site+1], (0, 2, 1, 3))
                theta = oe.contract('abcd,edfg,chij,fjkl->aebhikgl', tensor1, tensor2, tensor3, tensor4)
            else:
                mpo.rotate()
                tensor1 = np.transpose(gate_MPO.tensors[site_gate_MPO], (0, 2, 1, 3))
                tensor2 = np.transpose(gate_MPO.tensors[site_gate_MPO+1], (0, 2, 1, 3))
                tensor3 = np.transpose(mpo.tensors[overall_site], (0, 2, 1, 3))
                tensor4 = np.transpose(mpo.tensors[overall_site+1], (0, 2, 1, 3))
                theta = oe.contract('abcd,edfg,chij,fjkl->ikhbaelg', tensor1, tensor2, tensor3, tensor4)
                mpo.rotate()

            # Apply causal cone on each side
            theta = apply_causal_cone(theta, dag1, [overall_site, overall_site+1], conjugate=False)
            theta = apply_causal_cone(theta, dag2, [overall_site, overall_site+1], conjugate=True)

            dims = theta.shape
            theta = np.reshape(theta, (dims[0], dims[1], dims[2]*dims[3], dims[4], dims[5], dims[6]*dims[7]))
            mpo.tensors[overall_site], mpo.tensors[overall_site+1] = decompose_theta(theta, threshold)

            # Used to track tensors already applied
            gate_MPO.tensors[site_gate_MPO] = None
            gate_MPO.tensors[site_gate_MPO+1] = None

        # Odd length gate MPO, single hanging tensor
        if site_gate_MPO == len(sites)-1 and any(type(tensor) == np.ndarray for tensor in gate_MPO.tensors):
            if not conjugate:
                tensor1 = np.transpose(gate_MPO.tensors[site_gate_MPO], (0, 2, 1, 3))
                tensor2 = np.transpose(mpo.tensors[overall_site], (0, 2, 1, 3))
                theta = oe.contract('abcd,cefg->abefdg', tensor1, tensor2)
            else:
                mpo.rotate()
                tensor1 = np.transpose(gate_MPO.tensors[site_gate_MPO], (0, 2, 1, 3))
                tensor2 = np.transpose(mpo.tensors[overall_site], (0, 2, 1, 3))
                theta = oe.contract('abcd,cefg->febagd', tensor1, tensor2)
                mpo.rotate()

            dims = theta.shape
            theta = np.reshape(theta, (dims[0], dims[1]*dims[2], dims[3], dims[4]*dims[5]))

            tensor1 = np.transpose(mpo.tensors[overall_site-1], (0, 2, 1, 3))
            theta = oe.contract('abcd, edfg->aebcfg', tensor1, theta)
            
            theta = apply_causal_cone(theta, dag1, [overall_site-1, overall_site], conjugate=False)
            theta = apply_causal_cone(theta, dag2, [overall_site-1, overall_site], conjugate=True)

            mpo.tensors[overall_site-1], mpo.tensors[overall_site] = decompose_theta(theta, threshold)
            gate_MPO.tensors[site_gate_MPO] = None

    assert not any(type(tensor) == np.ndarray for tensor in gate_MPO.tensors)


def check_longest_gate(DAG_circuit: 'DAGCircuit'):
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


def select_starting_point(N: int, DAG_circuit: 'DAGCircuit'):
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


def iterate(mpo: 'MPO', dag1: 'DAGCircuit', dag2: 'DAGCircuit', threshold: float):
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


def run(circuit1: 'QuantumCircuit', circuit2: 'QuantumCircuit', threshold: float=1e-13, fidelity: float=1-1e-13):
    assert circuit1.num_qubits == circuit2.num_qubits
    start_time = time.time()
    mpo = MPO()
    mpo.init_identity(circuit1.num_qubits)

    circuit1_dag = circuit_to_dag(circuit1)
    circuit2_dag = circuit_to_dag(circuit2)
    iterate(mpo, circuit1_dag, circuit2_dag, threshold)

    return {'equivalent': mpo.check_if_identity(fidelity), 'elapsed_time': time.time()-start_time}
