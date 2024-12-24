from qiskit import QuantumCircuit, transpile, Aer, execute

class QuantumCircuitSimulator:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.qc = QuantumCircuit(num_qubits)

    def apply_gate(self, gate, qubit):
        if gate == 'H':
            self.qc.h(qubit)
        elif gate == 'X':
            self.qc.x(qubit)
        elif gate == 'CX':
            self.qc.cx(0, 1)

    def simulate(self):
        simulator = Aer.get_backend('aer_simulator')
        transpiled = transpile(self.qc, simulator)
        result = execute(transpiled, simulator).result()
        return result.get_counts()
