# this module contains functions used for Quantum Circuit calculations in the Quantum Subspace Expansion project

from openfermion import QubitOperator
from utils_fc import obtain_polynomial_representation_of_fc_hamiltonian
from qiskit.quantum_info import SparsePauliOp
import numpy as np
from qiskit.circuit import Parameter
from qiskit.circuit.library import RYGate
from math import log

# misc functions

def print_state(psi, threshold=1e-12):
    """
    gives a sort-of nice print out of the statevector psi
    """
    Nqubits = int(log(len(psi), 2))
    for i in range(len(psi)):
        if np.abs(psi[i]) > 1e-12:
            bin_string = bin(i)[2:]
            bin_string = '0' * (Nqubits - len(bin_string)) + bin_string
            print(bin_string, np.round(psi[i], 6))

    return None

# functions for measuring QWC Hamiltonians and processing measurement results

def convert_to_pauli_string(P, N):
    """
    P is a QubitOperator with a single non-zero term, or a term-tuple of a QubitOperator

    if P is like QubitOperator('X0 Y2 Z3'), and N is 10, the output of this is 'IIIIIIZYIX'

    Note that Openfermion uses big endian and Qiskit uses little endian
    """
    if isinstance(P, QubitOperator):
        P = list(P.terms.keys())[0]
    nontriv_indices = [t[0] for t in P]
    nontriv_paulis  = [t[1] for t in P]

    pauli_string = ''
    tally        = 0
    for i in range(N):
        if i in nontriv_indices:
            pauli_string += nontriv_paulis[tally]
            tally        += 1
        else:
            pauli_string += 'I'
    
    return pauli_string[::-1]

def convert_to_QubitOperator(Pstr):
    """
    Pstr is a Pauli string like 'IIIIIIZYIX'

    output is Openfermion QubitOperator 1.0 [X0 Y2 Z3]
    """
    Pstr       = Pstr[::-1]
    term_tuple = []

    for i, letter in enumerate(Pstr):
        if letter != 'I':
            term_tuple.append((letter, i))
    term_tuple = tuple(term_tuple)

    return QubitOperator(term_tuple)

def get_pauli_support(P):
    """
    P can be openfermion QubitOperator or a string like 'IIIIIIZYIX'
    
    returns qubit indices that P acts non-trivially on

    note that the string format is little endian
    """
    if isinstance(P, QubitOperator):
        return [t[0] for t in list(P.terms.keys())[0]]
    elif isinstance(P, str):
        N = len(P)
        return [(N - i - 1) for i in range(len(P)) if P[i] != 'I'][::-1]

def append_measurement_circuit(qc, P, N, include_measurements=True):
    """
    qc is a qiskit QuantumCircuit
    P is either
        1. an Openfermion QubitOperator with a single term
        2. a string like "XXYZIIYZI" which describes a Pauli operator
           the string is little endian
    """
    if isinstance(P, QubitOperator):
        P = convert_to_pauli_string(P, N)
    
    for i, letter in enumerate(P[::-1]):
        if letter == 'X':
            qc.h(i)
        elif letter == 'Y':
            qc.sdg(i)
            qc.h(i)
            
    if include_measurements:
        qc.measure_all()

    return None

def get_qwc_signature_pauli(H, N):
    """
    H is a QubitOperator which is assumed to be QWC
    N is the number of qubits

    return a string like 'XIIXZYII' such that diagonalizing the Pauli 
    associated with the string necessarily also diagonalizes H
    """
    term_tuple_list = list(H.terms.keys())
    P_string_list   = [convert_to_pauli_string(term_tuple, N) for term_tuple in term_tuple_list]

    QWC_signature_P = ''
    for i in range(N):
        success = False
        for P in P_string_list:
            if P[i] != 'I':
                success          = True
                QWC_signature_P += P[i]
                break
        if not success:
            QWC_signature_P += 'I'

    return QWC_signature_P

def get_irrelevant_qubits(Psig, Nqubits):
    """
    returns list of qubits that Psig acts as identity on
    """
    return [i for i in list(range(Nqubits)) if i not in get_pauli_support(Psig)]

def get_qwc_generators_from_signature_pauli(Psig):
    """
    Psig is a string like 'XIIXZYII'

    return is a list of Openfermion QubitOperator. In this case the output would be 
    [QubitOperator('Y2'), QubitOperator('Z3'), QubitOperator('X4'), QubitOperator('X7')]
    """
    generators = []
    for i, letter in enumerate(Psig[::-1]):
        if letter != 'I':
            generators.append(QubitOperator(f'{letter}{i}'))
    return generators

def process_qwc_hamiltonian_for_measurements(H, N):
    """
    H is a QubitOperator which is assumed to be a QWC Hamiltonian
    N is the number of qubits

    return is four things:
        1. The QWC signature Pauli (string in little endian) needed to get the measurement circuit
        2. Qubit-wise commuting single-qubit Pauli generators for H
        3. The polynomial representation of the QWC Hamiltonian needed to get the eigenvalues from the measurement results
        4. The "irrelevant qubits" (in little endian) which H, and therefore by implication, Psig, act trivially on. 
           These qubits do not need to be measured
    """
    Psig              = get_qwc_signature_pauli(H, N)
    generators        = get_qwc_generators_from_signature_pauli(Psig)
    polynomial        = obtain_polynomial_representation_of_fc_hamiltonian(H, N, generators)
    irrelevant_qubits = get_irrelevant_qubits(Psig, N)

    return Psig, generators, polynomial, irrelevant_qubits

def convert_QubitOperator_to_SparsePauliOp(H, N):
    """
    H is an Openfermion QubitOperator
    N is the number of qubits

    return is Qiskit SparsePauliOp representation of H
    """
    Hqk = SparsePauliOp(['I' * N], [0.0])
    for term, coef in H.terms.items():
        term_string = convert_to_pauli_string(term, N)
        Hqk += SparsePauliOp([term_string], [coef])
    return Hqk

def remove_irrelevant_entries(bin_string, irrelevant_qubits):
    """
    bin_string is something like '100101' (note that the qubits are ordered from right to left)
    irrelevant_qubits is something like [1,3]

    corresponding output would be '1011'

    the purpose of this function is to remove measurement results for qubits which do not effect the energy of the QWC Hamiltonian
    in particular, these are the qubits for which the Hamiltonian acts as identity on for all Pauli terms
    """
    small_bin_string = ''

    for i, num in enumerate(bin_string[::-1]):
        if not i in irrelevant_qubits:
            small_bin_string += num

    return small_bin_string[::-1]

def binary_string_to_parity_list_big_endian(bin_string, irrelevant_qubits=None):
    '''
    bin_string is something like '100101'

    corresponding output would be [-1, 1, -1, 1, 1, -1] 

    the purpose of this function is as a subroutine to go from Qiskit measurement results to Hamiltonian energies,
    which uses Openfermion Hamiltonians and thus requires swapping qubit order
    '''
    if not irrelevant_qubits is None:
        bin_string = remove_irrelevant_entries(bin_string, irrelevant_qubits)

    parity_list = []

    for num in bin_string:
        if num == '0':
            parity_list.append(1)
        elif num == '1':
            parity_list.append(-1)
        else:
            print('input is not a binary string --> returning NoneType')
            return None
        
    return parity_list[::-1]


# functions for preparing initial states

def append_hf_prepare_circuit(qc, Ne):
    """
    qc : qiskit QuantumCircuit
    Ne : number of electrons

    appends circuit for preparing HF state |1>^{\otimes Ne} \otimes |0>^{\otimes Nqubits - Ne}
    """
    for i in range(Ne):
        qc.x(i)
    
    return None

def append_Tia_prepare_circuit(qc, i, a):
    """
    appends circuit implementing Tia to qc
    """

    i_up   = 2 * i
    i_down = 2 * i + 1

    a_up   = 2 * a
    a_down = 2 * a + 1

    qc.h(i_down)
    qc.x(a_down)
    qc.cx(i_down, a_up)
    qc.cx(i_down, i_up)
    qc.cx(a_up, a_down)

    return None

def append_Tiajb_prepare_circuit(qc, i, a, j, b):
    """
    appends circuit for implementing Tia Tjb to qc
    this will only work if i < j
    """
    assert i < j
    assert a != b
    
    if b > a:
        qc.z(i)

    append_Tia_prepare_circuit(qc, i, a)
    append_Tia_prepare_circuit(qc, j, b)

    return None

def append_pair_excitation_unitary_suboptimal(qc, i, a, theta):
    """
    currently the optimal version is bugged so you can use this instead

    appends a circuit implementing exp[theta T_{iiaa}]
    the circuit is based on Fig 5 of [citation] which is sub-optimal
    """
    # the map from the papers notation to our notation
    # (i,j,k,l) -> (a_down, a_up, i_down, i_up)

    multi_RYGate = RYGate(2 * theta).control(3, ctrl_state='010')

    i_up   = 2 * i
    i_down = 2 * i + 1

    a_up   = 2 * a
    a_down = 2 * a + 1
    
    qc.cx(i_down, i_up)
    qc.cx(a_down, a_up)
    
    qc.cx(a_down, i_down)

    qc.append(multi_RYGate, [i_up, i_down, a_up, a_down])

    qc.cx(a_down, i_down)

    qc.cx(a_down, a_up)
    qc.cx(i_down, i_up)

    return None

def append_pair_excitation_unitary(qc, i, a, theta):
    """
    This does not work yet. (i.e., it is not equal to exp[phi Tiiaa] for any phi)
    
    Two bugs currently: it (1) is not real-orthogonal, and (2) does not conserve the number of electrons

    append a circuit implementing exp[theta T_{iiaa}]
    the circuit is based on Fig 7 of [citation] which is currently the best at minimizing CNOT count (13 CNOTs, 11 depth)
    theta can either be a number, or a qiskit.circuit.Parameter instant
    """
    # the map from the papers notation to our notation
    # (i,j,k,l) -> (a_down, a_up, i_down, i_up)

    i_up   = 2 * i
    i_down = 2 * i + 1

    a_up   = 2 * a
    a_down = 2 * a + 1
    
    qc.cx(i_down, i_up)
    qc.cx(a_down, a_up)

    qc.x(i_up)
    qc.x(a_up)

    qc.cx(a_down, i_down)
    
    qc.ry(theta / 4, a_down)
    qc.h(a_up)
    qc.cx(a_down, a_up)

    qc.ry(-theta / 4, a_down)
    qc.h(i_up)
    qc.cx(a_down, i_up)

    qc.ry(theta / 4, a_down)
    qc.cx(a_down, a_up)

    qc.ry(-theta / 4, a_down)
    qc.h(i_down)
    qc.cx(a_down, i_down)

    qc.ry(theta / 4, a_down)
    qc.cx(a_down, a_up)

    qc.ry(-theta / 4, a_down)
    qc.cx(a_down, i_up)

    qc.ry(theta / 4, a_down)
    qc.h(i_up)
    qc.cx(a_down, a_up)

    qc.ry(-theta / 4, a_down)
    qc.h(a_up)
    
    qc.rz(-np.pi/2, i_down)
    qc.cx(a_down, i_down)

    qc.rz(np.pi/2, a_down)
    qc.rz(-np.pi/2, i_down)

    qc.x(a_up)
    qc.ry(-np.pi/2, i_down)
    qc.x(i_up)

    qc.cx(i_down, i_up)
    qc.cx(a_down, a_up)

    return None

