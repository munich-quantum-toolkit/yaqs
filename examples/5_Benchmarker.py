from yaqs.circuits import benchmarker
from yaqs.core.libraries.circuit_library import create_Ising_circuit


num_qubits = 10
model = {'name': 'Ising', 'L': num_qubits, 'J': 1, 'g': 0.5}
demo_circuit = create_Ising_circuit(model, dt=0.1, timesteps=10)

# Run the benchmark on the demo circuit.
benchmarker.run(demo_circuit, style='dots')
