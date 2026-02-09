from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from openfermion.ops import QubitOperator
from qiskit.quantum_info import SparsePauliOp, Operator
import numpy as np
from ..measurement_new.utils_m4_partitioning import sorted_insertion_decomposition, convert_QubitOperator_to_BinaryHamiltonian
import tequila as tq
from qiskit_aer import AerSimulator
from qiskit import transpile
from scipy.sparse.linalg import norm
from scipy.sparse import csr_matrix
from copy import deepcopy

def untaper_circuit(circuit: QuantumCircuit, seniorities: list[int]):
    """
    Returns circuit for untapered state.

    seniorities: list[int] - orbital seniorities 0, 1
    
    """
    assert len(seniorities) == circuit.num_qubits

    qc = QuantumCircuit(2*circuit.num_qubits)

    qubits = qc.qubits
    register_1 = qubits[::2]
    register_2 = qubits[1::2]

    for i, q in enumerate(register_2):
        if seniorities[i] == 1:
            qc.x(q)
    
    qc = qc.compose(circuit, register_1)
    _ = [qc.cx(r1, r2) for r1, r2 in zip(register_1, register_2)]

    return qc

def count_cx_gates(circuit: QuantumCircuit):
    return sum(1 for instr, qargs, cargs in circuit.data if instr.name == 'cx')

def show_state(qc: QuantumCircuit, tol=1e-5, reverse=True, silent=True):
    """
    Show and return quantum state prepared by the circuit. 
    
    Note that qiskit qubit ordering is reversed (msb/little endian).
    Set reverse = True gives correct ordering


    """
    circuit = deepcopy(qc)
    circuit.remove_final_measurements()

    if reverse:
        circuit = circuit.reverse_bits()
    state = Statevector.from_instruction(circuit)
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

def measure_all_existing(qc):
    """Measure all qubits into existing clbits in order."""
    if qc.num_clbits == 0:
        qc.measure_all()
        return qc
    else:
        if qc.num_clbits < qc.num_qubits:
            raise ValueError("Not enough classical bits to store all measurements.")
        for q in range(qc.num_qubits):
            qc.measure(q, q)
        return qc

def simulate_qiskit_circuit(qc: QuantumCircuit, shots: int, noise_model=None, reverse=True, add_measurements=False) -> dict:
    """

    qc: QuantumCircuit
    
    add_measurements: add measurements to existing or new clbits. Raises error if incorrect number of clbits provided.
    
    """
    if noise_model is None:
        simulator = AerSimulator()
    else:
        simulator = AerSimulator(noise_model=noise_model)

    if add_measurements:
        qc = measure_all_existing(qc)

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

def int_from_bitstring(bs):
    return sum([int(ai)<<(len(bs) - 1 - i) for i, ai in enumerate(bs)])

def get_sparse_state(dict_state):
    """
    
    {bitstring: coeff} to sparse row matrix (csr matrix)

    """
    n_qubits = len(list(dict_state.keys())[0])

    cols = [int_from_bitstring(bs) for bs in dict_state.keys()]
    vals = list(dict_state.values())

    sparse_state = csr_matrix((vals, (np.zeros_like(cols), cols)), shape=(1, 1<<n_qubits))

    return sparse_state

def verify_circuit_state(circuit, sparse_target_state, truncate_bitstrings = None, tol=1e-5):
    """
    verify circuit with sparse state, truncate bit strings if necessary
    
    """

    state = show_state(circuit)
    bit_strings, coeffs = list(state.keys()), list(state.values())

    if truncate_bitstrings is not None:
        for i in range(len(bit_strings)):
            bs_new = ''
            for j in truncate_bitstrings:
                bs_new += bit_strings[i][j]
            bit_strings[i] = bs_new 
    
    sparse_state = get_sparse_state({k: v for k, v in zip(bit_strings, coeffs)})

    return 1 - abs((sparse_target_state.conjugate() @ sparse_state.T)[(0, 0)]) <= tol