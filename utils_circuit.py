from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from openfermion.ops import QubitOperator
from qiskit.quantum_info import SparsePauliOp

def count_cx_gates(circuit: QuantumCircuit):
    return sum(1 for instr, qargs, cargs in circuit.data if instr.name == 'cx')

def show_state(qc, tol=1e-5):
    """
    Show and return quantum state prepared by the circuit. Note that qiskit qubit ordering is reversed (msb/little endian).

    """

    state = Statevector.from_instruction(qc)
    state_dict = {}

    for i, amplitude in enumerate(state.data):
        basis = format(i, f'0{state.num_qubits}b')

        if abs(amplitude) >=tol:
            print(f"|{basis}⟩: {amplitude}")
            state_dict[basis] = amplitude
    
    return state_dict

def qubit_op_to_sparse_pauli_op(qubit_op: QubitOperator, n_qubits) -> SparsePauliOp:
    """
    Convert an OpenFermion QubitOperator to Qiskit's SparsePauliOp.
    """
    pauli_strings = []
    coeffs = []

    for term, coeff in qubit_op.terms.items():
        # Initialize with identity
        pauli_label = ['I'] * n_qubits

        for qubit_idx, pauli_char in term:
            pauli_label[qubit_idx] = pauli_char

        # Qiskit uses strings from most-significant to least-significant qubit
        # So we reverse to match OpenFermion's convention
        pauli_str = ''.join(pauli_label[::-1])
        pauli_strings.append(pauli_str)
        coeffs.append(coeff)

    print(pauli_strings, coeffs)
    return SparsePauliOp(pauli_strings, coeffs)