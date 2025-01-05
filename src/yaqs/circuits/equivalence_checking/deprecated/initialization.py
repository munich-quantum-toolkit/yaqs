import copy
import numpy as np
from qiskit.converters import circuit_to_dag, dag_to_circuit

from src.tensor_library import TensorLibrary


def initialize_identity_MPO(sites):
    MPO = []

    # Create identiy matrix
    M = np.eye(2)

    # Expand dimensions
    # Convention (sigma, chi_l, sigma', chi_l+1)
    M = np.expand_dims(M, (1, 3))
    for _ in range(sites):
        MPO.append(M)

    return MPO


def generate_gateset(unique_gates):
    gateset = []
    gate = {'name': "I", 'tensor': [], 'interaction': 1}
    gateset.append(gate)
    for gate_number in range(unique_gates):
        # Select n-qubit gate
        interaction = np.random.choice([1, 2])

        # Create random unitary
        random_matrix = np.random.rand(2**interaction, 2**interaction)
        unitary_matrix, _, _ = np.linalg.svd(random_matrix)

        # Tensorize if needed
        if interaction == 1:
            tensor = unitary_matrix
        else:
            # Decomposes unitary by splitting left and right sides of matrix
            tensor = np.reshape(unitary_matrix, (2, 2, 2, 2))

            # Reshape into clockwise index order
            tensor = np.transpose(tensor, (0, 2, 1, 3))

        gate = {'name': gate_number, 'tensor': tensor, 'interaction': interaction}
        gateset.append(gate)

    return gateset


def generate_algorithm(sites, gateset, layers):
    algorithm = []

    for _ in range(layers):
        layer = []
        for site in range(sites):
            gate = np.random.choice(gateset)

            # No 2-qubit gate can be applied at last site
            if gate['interaction'] == 2 and site == sites-1:
                continue
            else:
                if gate['interaction'] == 1:
                    operation = {'gate': gate, 'sites': [site]}
                else:
                    operation = {'gate': gate, 'sites': [site, site+1]}
                    site += 1

                layer.append(operation)
        algorithm.append(layer)

    return algorithm


def convert_circuit_to_tensor_algorithm(circuit):
    algorithm = []
    dag = circuit_to_dag(circuit)

    for layer in dag.layers():
        parsed_layer = []
        # Hit barrier at end of circuit
        if not layer:
            break
        layer_circuit = dag_to_circuit(layer['graph'])
        if layer_circuit.data[0].operation.name == 'barrier':
            break

        for gate in layer_circuit.data:
            name = gate.operation.name
            if name == 'measure':
                continue
            attr = getattr(TensorLibrary, name)
            gate_object = attr()
            if gate.operation.params:
                angle = gate.operation.params[0]
                gate_object.set_theta(angle)

            sites = [gate.qubits[0]._index]
            if gate_object.interaction == 2:
                sites.append(gate.qubits[1]._index)
            elif gate_object.interaction == 3:
                sites.append(gate.qubits[1]._index)
                sites.append(gate.qubits[2]._index)

            if gate_object.interaction == 2 and np.abs(sites[0] - sites[1]) > 1:
                gate_object.create_MPO(sites[0], sites[1])

            if (gate.operation.name == 'cx' or gate.operation.name == 'cz') and sites[1] < sites[0]:
                    gate_object.tensor = np.transpose(gate_object.tensor, (1, 0, 3, 2))

            operation = {'gate': gate_object, 'sites': sites}
            parsed_layer.append(operation)
        algorithm.append(parsed_layer)
    return algorithm


def convert_qasm_to_tensor_algorithm(filename, num_qubits):
    parse = False
    algorithm = []
    layer = []
    last_location = -1
    for line in open(filename,'r'):
        if line.find('barrier') != -1:
            algorithm.append(layer)
            parse = False
        if parse:
            information = line.split()
            # Check if there is a rotation angle
            if len(information[0]) > 4:
                name = information[0][0:information[0].find('(')]
                angle = information[0][information[0].find('(')+1:information[0].find(')')]
                if angle.find('pi') > -1:
                    angle = angle.replace('pi', str(np.pi))
                    angle = eval(angle)
                else:
                    angle = float(angle)

                if angle == 0:
                    continue
                # Set angle
                attr = getattr(TensorLibrary, name)
                gate_object = attr()
                gate_object.set_theta(angle)
                gate = {'name': name, 'tensor': gate_object.tensor, 'interaction': gate_object.interaction}
            else:
                attr = getattr(TensorLibrary, information[0])
                gate_object = attr()
                gate = {'name': information[0], 'tensor': gate_object.tensor, 'interaction': gate_object.interaction}

            qubit0 = int(information[1][information[1].index('[')+1:information[1].index(']')])
            if gate_object.interaction == 1:
                operation = {'gate': gate, 'sites': [qubit0]}
                location = qubit0
            elif gate_object.interaction == 2:
                information[1] = information[1][information[1].index(']')+2:-1]
                qubit1 = int(information[1][information[1].index('[')+1:information[1].index(']')])
                if (gate['name'] == 'cx' or gate['name'] == 'cz') and qubit1 < qubit0:
                    gate['tensor'] = np.transpose(gate['tensor'], (2, 3, 1, 0))
                operation = {'gate': gate, 'sites': [qubit0, qubit1]}
                location = max(qubit0, qubit1)

            if last_location < location:
                layer.append(operation)
                last_location = location
            else:
                algorithm.append(layer)
                layer = []
                layer.append(operation)
                last_location = location

        if line.find('creg') != -1:
            parse = True

    return algorithm


# TODO: Fix for long-range
def convert_layer_to_MPO(layer):
    MPO = []

    for i, operation in enumerate(layer):
        # Determine placement of MPO
        # Assumes that the layer is already constructed
        # in increasing qubit order
        if i == 0:
            start_qubit = min(operation['sites'])
        if i == len(layer)-1:
            end_qubit = max(operation['sites'])

        # Maintains that the layer is properly constructed
        if i != 0:
            assert not(set(operation['sites']) & set(previous_operation['sites']))

        # Adds an intermediate identity, should be removed eventually or replaced with None
        if i != 0:
            while min(operation['sites']) != max(previous_operation['sites'])+1:
                attr = getattr(TensorLibrary, 'id')
                gate_object = attr()
                for tensor in gate_object.MPO:
                    MPO.append(tensor)

                previous_operation = {'gate': gate_object, 'sites': [max(previous_operation['sites'])+1]}

        previous_operation = copy.deepcopy(operation)

        gate = operation['gate'] # getattr(TensorLibrary, operation['gate']['name'])

        # MPO is already hardcoded for single-qubit gates
        # Having the function here stops us from having to store
        # the MPO if not needed
        if gate.interaction == 2:
            gate.create_MPO(*operation['sites'])

        for tensor in gate.MPO:
            MPO.append(tensor)


    # Checksafe to make sure a valid MPO is created
    last_tensor = None
    for tensor in MPO:
        if tensor is not None:
            if last_tensor is not None:
                # Previous right bond dimension is equivalent to
                # next left bond dimension
                assert last_tensor.shape[3] == tensor.shape[1]
            last_tensor = tensor

    MPO_layer = {'MPO': MPO, 'sites': sorted(list(set([start_qubit, end_qubit])))}
    return MPO_layer


def convert_circuit_to_MPO_algorithm(circuit):
    # algorithm = convert_circuit_to_tensor_algorithm(circuit)
    MPO_algorithm = []

    dag = circuit_to_dag(circuit)
    for layer in dag.layers():
        parsed_layer = []
        # Hit barrier at end of circuit
        if not layer:
            break
        layer_circuit = dag_to_circuit(layer['graph'])
        if layer_circuit.data[0].operation.name == 'barrier':
            break

        for gate in layer_circuit.data:
            name = gate.operation.name
            attr = getattr(TensorLibrary, name)
            gate_object = attr()
            if gate.operation.params:
                angle = gate.operation.params[0]
                gate_object.set_theta(angle)

            sites = [gate.qubits[0].index]
            if gate_object.interaction == 2:
                sites.append(gate.qubits[1].index)
            elif gate_object.interaction == 3:
                sites.append(gate.qubits[1].index)
                sites.append(gate.qubits[2].index)

            operation = {'gate': gate_object, 'sites': sites}
            parsed_layer.append(operation)

        MPO = convert_layer_to_MPO(parsed_layer)
        MPO_algorithm.append(MPO)

    return MPO_algorithm