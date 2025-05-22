# this module contains functions for manipulating fully-commuting Hamiltonians
# this includes: finding the diagonalizing Clifford unitary, Hamiltonian abelian group generators
# and polynomial expansion in the generators

# this code was not designed with QWC Hamiltonians in mind. Although it works for QWC Hamiltonians, it may produce
# sub-optimal multi-qubit generators and multi-qubit diagonalization unitaries

# there are functions in utils.py which will correct for this when working with QWC Hamiltonians

import numpy as np
from openfermion import QubitOperator as Q
from numpy.random import uniform
from utils_basic import (
    random_pauli_term,
    is_commuting, 
    apply_unitary_product, 
    clifford
)
from utils_la import (
    obtain_basis_z2, 
    obtain_expansion_coefficients_over_specified_basis_z2
)

NUMERIC_TYPES = [int, 
                 float, 
                 complex, 
                 np.int8, 
                 np.int16, 
                 np.int32, 
                 np.int64, 
                 np.float16, 
                 np.float32, 
                 np.float64, 
                 np.complex64]

def decimal_to_binary_list(n, length=None):
    """
    converts a decimal like 10 to a binary list like [1, 0, 1, 0]

    you can append zeros if desired by setting the length parameter

    so length = 7 would return [0, 0, 0, 1, 0, 1, 0] instead
    """
    bin_list = [int(z) for z in bin(n)[2:]]

    if (length is None) or (len(bin_list) == length):
        return bin_list
    
    elif len(bin_list) > length:
        print('returned list is longer than specified length')
        return bin_list
    
    else:
        while len(bin_list) < length:
            bin_list = [0] + bin_list
        return bin_list
    
def decimal_to_binary_string(n, length=None):
    """
    converts a decimal like 10 to a binary string like '1010'

    you can append zeros if desired by setting the length parameter

    so length = 7 would return '0001010' instead
    """
    bin_list   = decimal_to_binary_list(n, length=length)
    bin_string = ''
    for z in bin_list:
        bin_string += str(z)
    return bin_string
    
def decimal_to_parity_list(n, length=None):
    """
    takes the output of decimal_to_binary_list and transforms 0 -> 1, 1 -> -1
    """
    return 1 - 2 * np.array(decimal_to_binary_list(n, length))
    
def is_fc_hamiltonian(H):
    """
    checks if H is a fully commuting Hamitonian
    """
    for t, _ in H.terms.items():
        for s, _ in H.terms.items():
            if (not is_commuting(t, s)):
                return False
    return True

def fc_inclusion_criterion(H, A):
    """
    H is a QubitOperator which is assumed to be fully-commuting
    A is a QubitOperator with one term

    checks if A can be included in H while preserving the fully commuting criterion
    """
    for t, _ in H.terms.items():
        if (not is_commuting(t, A)):
            return False
    return True

def random_fc_hamiltonian(Nqubits, Nterms):
    """
    returns a random fully commuting Hamiltonian on Nqubits with Nterms
    """
    assert Nterms <= 2 ** Nqubits

    H = Q().zero()

    while len(H.terms) < Nterms:

        _, A = random_pauli_term(Nqubits)
        if fc_inclusion_criterion(H, A):
            H += uniform(-2, 2) * A

    assert is_fc_hamiltonian(H)
    return H

def inverse_phase(x):
    """
    x is \pm 1 or \pm i

    returns the phase y such that x * y = 1
    """
    if x == 1:
        return 1
    elif x == -1:
        return -1
    elif x == 1j:
        return -1j
    elif x == -1j:
        return 1j
    else:
        print('inverse of supplied phase not implemented')
        return None
    
def instantiate_one(onetype):
    """
    returns the identity element associated to the algebra of the input 
    """
    if onetype in NUMERIC_TYPES:
        return 1
    elif onetype == Q:
        return Q("")
    else:
        print("1 for supplied type not implemented")
        return None
    
def instantiate_zero(zerotype):
    """
    returns the zero element associated to the algebra of the input
    """
    if zerotype in NUMERIC_TYPES:
        return 0
    elif zerotype == Q:
        return Q()
    else:
        print("0 for supplied type not implemented")
        return None
    
def fS(P, N):
    """
    converts Pauli operator on N qubits to binary vector in 2N dimensions

    Note the deletion of the coefficient of P. This implies that this map is a 
    4-to-1 homomorphism of the Pauli group onto GF(2)^2N
    """

    P.compress()
    Pterm = list(P.terms.keys())[0]

    v = np.zeros(2 * N)
    for op in Pterm:

        if op[1] == 'Z':
            v[op[0]]     = 1

        elif op[1] == 'X':
            v[N + op[0]] = 1

        elif op[1] == 'Y':
            v[op[0]]     = 1
            v[N + op[0]] = 1
    
    return v

def fO(v):
    """
    Converts binary vector in 2N dimensions to Pauli operator on N qubits

    From the 4-to-1 homomorphism perspective, this map sends the binary vectors 
    back to Pauli operators with coefficient 1.
    """
    
    assert len(v) % 2 == 0

    N = len(v) // 2
    P = []

    for i in range(N):

        if v[i] == 1 and v[N + i] == 0:
            P.append( (i, 'Z') )

        elif v[i] == 0 and v[N + i] == 1:
            P.append( (i, 'X') )

        elif v[i] == 1 and v[N + i] == 1:
            P.append( (i, 'Y') )
    
    return Q(tuple(P))

def BinSym_J(N):
    """
    returns the metric for GF(2)^2N
    """
    zero = np.zeros([N,N])
    one  = np.identity(N)

    return np.block([
        [zero, one],
        [one, zero]
    ])

def BinSym_inner_product(v, w):
    """
    computes the symplectic inner product of v, w in GF(2)^2N
    """
    N = len(v) // 2

    assert (len(v) == 2 * N) and (len(w) == 2 * N)

    return (v @ BinSym_J(N) @ w) % 2

def BinSym_is_orthogonal(v, w):
    """
    check if v and w are orthogonal in the symplectic space GF(2)^2N
    """
    return BinSym_inner_product(v, w) == 0

def BinSym_is_linearly_independent(v_list, w):
    """
    check if w is linearly independent to the vectors in v_list manually

    this function will take exponential time to run in general, and has been replaced
    by functions in utils_la.py
    """

    N = len(w) // 2
    size = len(v_list)

    for n in range(2 ** size):
        coefs = decimal_to_binary_list(n, size)

        w_attempt = np.zeros(2 * N)
        for k in range(size):
            w_attempt += coefs[k] * v_list[k]
        w_attempt = w_attempt % 2

        if np.allclose(w, w_attempt):
            return False
        
    return True

def BinSym_obtain_basis(vectors):
    """
    vectors is a list of GF(2)^2N elements

    returns a list of GF(2)^2N which form a basis for span{vectors}
    
    this function will take exponential time to run in general, and has been replaced
    by functions in utils_la.py
    """
    
    N = len(vectors[0]) // 2
    if np.allclose(vectors[0], np.zeros(2 * N)):
        basis = [vectors[1]]
    else:
        basis = [vectors[0]]

    for w in vectors[1:]:
        if BinSym_is_linearly_independent(basis, w):
            basis.append(w)
    
    return basis

def BinSym_obtain_expansion_coefficients(basis, w):
    """
    returns coefficients for the expansion of w as a linear combination of the vectors
    in the basis, assuming such a linear combination exists

    this function will take exponential time to run in general, and has been replaced
    by functions in utils_la.py
    """

    N = len(w) // 2
    size = len(basis)

    for n in range(2 ** size):
        coefs = decimal_to_binary_list(n, size)

        w_attempt = np.zeros(2 * N)
        for k in range(size):
            w_attempt += coefs[k] * basis[k]
        w_attempt = w_attempt % 2

        if np.allclose(w, w_attempt):
            return coefs
        
    print("supplied vector not in span of basis")
    return None

def obtain_generators(op_list, N):
    """
    given some set of Pauli operators, returns G = {C1,...,CK} such that all P in the set can
    be expressed as products up to phases of elements of G
    """
    binary_vectors = [fS(op, N) for op in op_list]
    basis_vectors  = obtain_basis_z2(binary_vectors)
    generators     = [fO(b) for b in basis_vectors]

    return generators

def obtain_product(C, powers):
    """
    C is a list of one-term QubitOperators
    powers is a list of 0s and 1s

    returns the product (C[0]^powers[0]) * ... * (C[-1]^powers[-1])
    """
    one_type = type(C[0])
    product  = instantiate_one(one_type)

    for k in range(len(C)):
        if powers[k] == 1:
            product *= C[k]

    return product

def obtain_factorization_data(op_list, generators, N):
    """
    returns phase and generator-powers to express all P in op_list as products, up to a phase,
    of the generators
    """
    binary_vectors = [fS(op, N) for op in op_list] 
    basis          = [fS(op, N) for op in generators]
    powers         = obtain_expansion_coefficients_over_specified_basis_z2(basis, binary_vectors)

    ops_from_generators = [obtain_product(generators, pows) for pows in powers]
    phase_from_products = [list(op.terms.values())[0] for op in ops_from_generators]
    phase_correction    = [inverse_phase(x) for x in phase_from_products]

    return powers, phase_correction

def obtain_polynomial_representation_of_fc_hamiltonian(H, N, generators=None):
    """
    H          : QubitOperator representation of a fully-commuting Hamiltonian
    N          : number of qubits
    generators : a list of QubitOperators such that each term in H is a product of the
                 generators

                 if generators is None, then a generators list is calculated and returned as
                 an output of this function

                 if H is QWC, then you should not rely on this function to calculate the generators.
                 Instead, you should pre-calculate the single-qubit generators using other functions 
                 and input them to this function

    return     : let K denote the number of generators. The output is a function p which takes as input, a list of
                 K elements, which can either be QubitOperators or numbers, such that p(generators) == H,
                 and such that p(parity_list) is an eigenvalue of H for all length K parity lists 
    """
    
    if H == Q().zero():

        return_generators = False
        if generators is None:
            return_generators = True
            generators        = []

        def p(C):
            if C == []:
                return Q().zero()
            else:
                zero_type = type(C[0])
                return instantiate_zero(zero_type)
            
        if return_generators:
            return p, generators
        return p

    c = []
    A = []

    for t, s in H.terms.items():
        c.append(s)
        A.append(Q(t))

    return_generators = False
    if generators is None:
        return_generators = True
        generators        = obtain_generators(A, N)

    powers, phase_corrections = obtain_factorization_data(A, generators, N)

    def p(C):

        zero_type = type(C[0])
        value     = instantiate_zero(zero_type)

        Nterms = len(A)
        for k in range(Nterms):
            value += c[k] * phase_corrections[k] * obtain_product(C, powers[k])

        return value

    if return_generators:
        return p, generators
    return p

def obtain_simultaneous_eigenspace_projector_function(C):
    """
    given a list C of mutually commuting independent Pauli operators C, return function Proj such that
    Proj([v1,...,vK]) projects onto simultaneous eigenspace defined by {Ck = vk : 1 <= k <= K}
    """

    K = len(C)

    def Proj(v):
        P = Q("")
        for k in range(K):
            P *= (1 + v[k]*C[k]) / 2
        return P
    
    return Proj

def z_proj(v):
    """
    given simultaneous z-eigenvalues zk = v[k], return projector |v><v|
    """
    P = Q("")
    for k in range(len(v)):
        P *= (1 + v[k] * Q(f'Z{k}')) / 2
    return P

def evaluate_fc_polynomial_through_spectral_decomposition(p, C):
    """
    evaluate \sum_v p(v) P_v(C), which should equal p(C)
    """
    operator = Q()
    K        = len(C)
    Proj     = obtain_simultaneous_eigenspace_projector_function(C)

    for n in range(2 ** K):
        z         = decimal_to_binary_list(n, K)
        v         = 1 - 2 * np.array(z)
        operator += p(v) * Proj(v)

    return operator

def single_qubit_anticommuting(P, start_idx):
    """
    sub-routine for obtaining a Clifford transformation which diagonalizes a fully-commuting Hamiltonian

    This function returns a Pauli operator which
        1. anti-commutes with P
        2. acts on only one qubit, which is one of {start_idx, ..., Nqubits-1}
    """
    P.compress()
    Pterm = list(P.terms.keys())[0]

    for op in Pterm:

        if op[0] >= start_idx:

            if op[1] in ['X', 'Y']:
                return op[0], Q(f'Z{op[0]}')
            
            elif op[1] in ['Z']:
                return op[0], Q(f'X{op[0]}')
            
    print(f'no single qubit anticommuting on qubits {start_idx},...,N')
    return None

def obtain_pairing_to_right_qubit(op, wrong_idx, right_idx):
    """
    sub-routine for obtaining a Clifford transformation which diagonalizes a fully-commuting Hamiltonian
    """
    assert wrong_idx != right_idx

    if op == Q(f'X{wrong_idx}'):
        return clifford(
            op, Q(f'X{right_idx} Z{wrong_idx}')
        ), Q(f'X{right_idx} Z{wrong_idx}')
    
    elif op == Q(f'Z{wrong_idx}'):
        return clifford(
            op, Q(f'X{right_idx} X{wrong_idx}')
        ), Q(f'X{right_idx} X{wrong_idx}')
    
    else:
        print("something went wrong!")
        return None

def map_Ci_to_Zi(C, i):
    """
    returns Clifford unitary that maps an operator Ci to the operator Q(f'Z{i}')

    note: for simplicity, in the case where C could have a phase in its coefficient, I always do the
          mapping of C to a single qubit anticommuting, since the output Pauli of such a mapping 
          has no phase

          thus, I don't include special cases where C is one of Q(f'Z{i}'), Q(f'X{i}'), or Q(f'Y{i}')
          like I did in the map_pauli_to_z0 function for solving AC hamiltonians
    """
    U_list = []

    active_index, ac_op = single_qubit_anticommuting(C, i)
    U_list.append( clifford(C, ac_op) )

    if (active_index == i) and (ac_op == Q(f'Z{i}')):
        return U_list
    
    elif (active_index == i) and (ac_op == Q(f'X{i}')):
        U_list.append( clifford(ac_op, Q(f'Z{i}')) )
        return U_list
    
    else:
        pairing_U, pairing_op = obtain_pairing_to_right_qubit(ac_op, active_index, i)
        finishing_U           = clifford(pairing_op, Q(f'Z{i}')) 
        U_list               += [pairing_U, finishing_U]
        return U_list
    
def solve_multiple_independent_commuting_generators(C_list, start_idx=0):
    """
    finds Clifford unitary that maps C_list to [Q(f'Z{start_idx}'), ..., Q(f'Z{start_idx + len(C_list)}')]
    """
    global_U_list = []

    for i, C in enumerate(C_list):
        C = apply_unitary_product(C, global_U_list)
        global_U_list += map_Ci_to_Zi(C, i + start_idx)

    return global_U_list

def solve_fc_hamiltonian(H, N, C_list=None, start_idx=0):
    """
    returns:
        1. polynomial representation of H
        2. generators of H (assuming they are not pre-calculated)
        3. Clifford unitary mapping H to Ising form
    """

    if C_list is None:
        return_C_list = True
        p, C_list = obtain_polynomial_representation_of_fc_hamiltonian(H, N, generators=None)

    else:
        return_C_list = False
        p = obtain_polynomial_representation_of_fc_hamiltonian(H, N, generators=C_list)

    U_list = solve_multiple_independent_commuting_generators(C_list, start_idx=start_idx)

    if return_C_list:
        return p, C_list, U_list
    return p, U_list

def random_clifford(N):
    """
    returns a random Clifford unitary on N qubits

    method is to generate a random fc Hamiltonian on N qubits and return its solving unitary
    """
    if N < 4:
        Hfc = random_fc_hamiltonian(N, N+1)
        return solve_fc_hamiltonian(Hfc, N)[-1]
    
    else:
        Hfc = random_fc_hamiltonian(N, 3*N)
        return solve_fc_hamiltonian(Hfc, N)[-1]