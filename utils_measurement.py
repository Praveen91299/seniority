# this module contains functions used for measuring Hamiltonian fragments

from openfermion import QubitOperator
from utils_fc import obtain_polynomial_representation_of_fc_hamiltonian
from qiskit.quantum_info import SparsePauliOp
import numpy as np
from qiskit.circuit.library import RYGate
from qiskit.primitives import StatevectorSampler
from math import floor 

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

    appends the circuit for measuring P to the end of qc

    note: this circuit by default will put measurement gates on all the qubits for simplicity.
          maybe for hardware-error reasons it makes more sense to only measure the necessary qubits?
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

def estimate_qwc_hamiltonian_expectation_value(psi_qc, H, Nqubits, Nshots):
    """
    inputs
        1. psi_qc  : quantum circuit for preparing psi
        2. H       : Openfermion-QubitOperator Hamiltonian which is assumed to be QWC
        3. Nqubits : number of qubits
        4. Nshots  : number of measurements

    output
        an estimate of <psi|H|psi> obtained by measuring H a total of Nshots times
    """
    Psig, _, poly, irr_qubits = process_qwc_hamiltonian_for_measurements(H, Nqubits)
    qc_for_measurement        = psi_qc.copy()
    append_measurement_circuit(qc_for_measurement, Psig, Nqubits)

    sampler             = StatevectorSampler()
    measurement_results = sampler.run([qc_for_measurement], shots=Nshots) 
    counts              = measurement_results.result()[0].data.meas.get_counts()

    ev_est = 0
    for bin_string, count in counts.items():
        v       = binary_string_to_parity_list_big_endian(bin_string, irr_qubits)
        ev_est += poly(v) * (count / Nshots)

    return ev_est

def optimal_measurement_allocation(variances, Nshots_total):
    """
    inputs
        1. variances    : list of variances, or estimates thereof, of QWC fragments
        2. Nshots_total : the total number of shots

    output
        list of shot allocations for all the fragments obtained via Neyman allocation. The total number of shots may decrease by an amount that is <= len(variances)

    todo: dump remove measurements back in artificially so that the total number of measurements is Nshots_total
    """
    return [floor(Nshots_total * variances[k] / sum(variances)) for k in range(len(variances))]

