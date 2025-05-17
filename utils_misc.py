# functions for debugging and other miscellany

import numpy as np
from math import log
from openfermion import QubitOperator

def duplicate_pauli_term_qubit_wise(term):
    """
    XYIZ -> XXYYZZ
    doesn't work with identities
    """
    newterm = []
    for op in term:
        newterm.append((2*op[0], op[1]))
        newterm.append((2*op[0]+1, op[1]))
    return newterm

def duplicate_hamiltonian_qubit_wise(H):
    """
    same as duplicate_pauli_term_qubit_wise but for a Hamiltonian with multiple Pauli terms
    """
    newH = QubitOperator()
    for term, coef in H.terms.items():
        newH += coef * QubitOperator(duplicate_pauli_term_qubit_wise(term))
    return newH

def duplicate_statevector_qubit_wise(psi):
    """
    psi is a statevector on N qubits

    return is a statevector on 2N qubits obtained by taking binary string (i.e., 0110) of psi-entries and changing them to duplicated versions (i.e., 00111100)
    """
    Nqubits = int(log(len(psi), 2))
    newpsi  = np.zeros(2 ** (2 * Nqubits))

    for i, val in enumerate(psi):
        bit_i                    = bin(i)[2:]
        newbit_i                 = ''.join([s * 2 for s in bit_i])
        newpsi[int(newbit_i, 2)] = val

    return newpsi

