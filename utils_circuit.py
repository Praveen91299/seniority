from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from openfermion.ops import QubitOperator
from qiskit.quantum_info import SparsePauliOp, Operator
import numpy as np

def count_cx_gates(circuit: QuantumCircuit):
    return sum(1 for instr, qargs, cargs in circuit.data if instr.name == 'cx')

def show_state(qc: QuantumCircuit, tol=1e-5, reverse=True, silent=True):
    """
    Show and return quantum state prepared by the circuit. 
    
    Note that qiskit qubit ordering is reversed (msb/little endian).
    Set reverse = True gives correct ordering


    """

    if reverse:
        qc = qc.reverse_bits()
    state = Statevector.from_instruction(qc)
    state_dict = {}

    for i, amplitude in enumerate(state.data):
        basis = format(i, f'0{state.num_qubits}b')
        
        if abs(amplitude) >=tol:
            if not silent: print(f"|{basis}⟩: {amplitude}")
            state_dict[basis] = amplitude
    
    return state_dict

def show_operator_matrix(qc, tol=1e-5, reverse=True):
    """
    Show and return operator representing circuit
    set reverse = True for correct ordering of qubits

    """
    if reverse:
        qc = qc.reverse_bits()
    qc_mat = np.array(Operator(qc).to_matrix())

    qc_mat[np.abs(qc_mat) < tol] = 0
    return qc_mat

def check_equivalent_operators(op_1, op_2, tol=1e-5):
    """
    Check if the two operators are equivalent upto a phase

    """
    if np.shape(op_1) != np.shape(op_2):
        return False
    
    n = np.shape(op_1)[0]

    u1_u2 = op_1.conjugate().T @ op_2

    if abs(abs(u1_u2[0, 0]) - 1) < tol:
        if np.sum(np.abs(np.diag(u1_u2) - u1_u2[0, 0])) < tol and np.sum(np.abs(u1_u2 - np.diag(np.diag(u1_u2)))) < tol:
            return True
        else:
            return False
    else:
        return False



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