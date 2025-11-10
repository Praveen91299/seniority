from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from openfermion.ops import QubitOperator
from qiskit.quantum_info import SparsePauliOp, Operator
import numpy as np
from seniority.src.measurement.utils_partitioning import sorted_insertion_decomposition, convert_QubitOperator_to_BinaryHamiltonian
import tequila as tq
from qiskit_aer import AerSimulator
from qiskit import transpile

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

def get_decomp_circuits_estimators(H: QubitOperator, n_qubits, methodtag='fc'):
    """
    Returns fragments, measurement circuits, and diagonal fragments

    """

    decomp = sorted_insertion_decomposition(H, methodtag)
    z_ops = []
    meas_circuits = []

    for op in decomp:
        BH = convert_QubitOperator_to_BinaryHamiltonian(op)
        z_op, circ = BH.get_qubit_wise()
        circ.n_qubits = n_qubits

        z_ops.append(z_op.qubit_operator)
        
        
        qiskit_circ = QuantumCircuit.from_qasm_str(tq.export_open_qasm(circ))
        qiskit_circ.remove_final_measurements()
        meas_circuits.append(qiskit_circ)
        
    return decomp, meas_circuits, z_ops

def estimate_diag_from_measurements(Z_op: QubitOperator, measurements: dict) -> complex:
    """
    Estimate Z_op QubitOperator using measurement statistics {bit string: count} given in measurements

    """
    def eval_term_bs(term, bs):
        assert len(term) <= len(bs), "Insufficient bits in measurement."
        pos = []
        for op in term:
            assert op[1] == 'Z', "Operator not Pauli Z or identity!"
            pos.append(op[0])
        
        return 1 - 2 * (sum([int(bs[p]) for p in pos]) % 2)  
    
    M = sum(measurements.values())
    expectation:complex = 0
    for bs, count in measurements.items():
        
        p = count / M
        for term, value in Z_op.terms.items():
            expectation += value * eval_term_bs(term, bs) * p
    
    return expectation

def simulate_qiskit_circuit(qc: QuantumCircuit, shots: int, noise=False, reverse=True) -> dict:
    """
    TODO add noise
    
    """
    simulator = AerSimulator()
    
    for i in range(qc.num_qubits):
        qc.measure(i, i)

    # Transpile for simulator
    compiled_circuit = transpile(qc, simulator)
    result = simulator.run(compiled_circuit, shots=shots).result()

    counts = result.get_counts()
    
    if reverse:
        counts_rev = {}
        for k, v in counts.items():
            counts_rev[k[::-1]] = v
        return counts_rev
    return counts

def qubit_index(i, qubit_list: list):
    """
    Get position of i in qubit_list, returns -1 if not present

    """
    if i not in qubit_list:
        return -1
    
    return qubit_list.index(i)