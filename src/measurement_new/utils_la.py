# this module contains linear algebra routines over \mathbb{R} and \mathbb{Z}_2, used when working with FC and QWC Hamiltonians

import numpy as np
from scipy.linalg import expm, schur
import scipy.optimize as op
from numpy.random import uniform

#
#    misc functions
#

def swap_rows(A, i, j):
    """
    swap rows i and j of matrix A
    """
    Ai = A[i].copy()
    Aj = A[j].copy()

    A[i] = Aj
    A[j] = Ai

    return None

def swap_columns(X, i, j):
    """
    swap columns i and j of matrix X
    """
    Xi = X[:,i].copy()
    Xj = X[:,j].copy()

    X[:,i] = Xj
    X[:,j] = Xi

    return None

#
#    Gauss-Jordan elimination and solving linear systems functions
#

def find_pivot_row(A, col, pivot_indices):
    """
    find first non-zero element of A[:][col] that is not in pivot_indices (called the pivot element of the column)
    return -1 if no pivot in specified column 
    """
    Nrow, Ncol    = A.shape
    pivotted_rows = [x[0] for x in pivot_indices]

    for row in range(Nrow):
        if (not row in pivotted_rows) and np.abs(A[row][col]) > 1e-8:
            return row
    return -1

def gauss_jordan(Amatrix):
    """
    return rref and pivot columns of A by performing Gauss-Jordan elimination over \mathbb{C}
    """
    A          = Amatrix.copy()
    Nrow, Ncol = A.shape

    pivot_indices = []
    for col in range(Ncol):
        pivot_row = find_pivot_row(A, col, pivot_indices)

        if pivot_row != -1:
            A[pivot_row] = A[pivot_row] * (1 / A[pivot_row][col])

            for row in range(Nrow):
                if row != pivot_row:
                    A[row] = A[row] - A[row][col] * A[pivot_row]

            if pivot_row != len(pivot_indices):
                swap_rows(A, pivot_row, len(pivot_indices))
            pivot_indices.append((len(pivot_indices), col))

    return A, tuple( [x[1] for x in pivot_indices] )

def gauss_jordan_z2(Amatrix):
    """
    obtain rref and pivot columns of A via Gauss-Jordan elimination over \mathbb{Z}_2
    this function raises an AssertionError if any of the elements of A are not 0.0 or 1.0
    """
    A = Amatrix.copy()
    Nrow, Ncol = A.shape

    for row in range(Nrow):
        for col in range(Ncol):
            assert A[row][col] in {0.0, 1.0}

    pivot_indices = []
    for col in range(Ncol):
        pivot_row = find_pivot_row(A, col, pivot_indices)
        if pivot_row != -1:

            for row in range(Nrow):
                if row != pivot_row and np.abs(A[row][col]) > 1e-8:
                    A[row] = (A[row] + A[pivot_row]) % 2

            if pivot_row != len(pivot_indices):
                swap_rows(A, pivot_row, len(pivot_indices))
            pivot_indices.append((len(pivot_indices), col))

    return A, tuple( [x[1] for x in pivot_indices] )

def obtain_basis_z2(v_list):
    """
    given list of vectors over \mathbb{Z}_2 field, obtain "left-justified" basis obtained from computing rref
    """
    A            = np.array(v_list, dtype=np.float64).T
    rref, pivots = gauss_jordan_z2(A)

    return [ A[:,i] for i in pivots ]

def obtain_expansion_coefficients_z2(v_list):
    """
    given list of vectors over \mathbb{Z}_2 field, return expansion coefficients of all vectors
    as linear combination over "left-justified" basis obtained from computing rref
    """
    Ncol         = len(v_list)
    A            = np.array(v_list, dtype=np.float64).T
    rref, pivots = gauss_jordan_z2(A)

    return [ rref[:,i][:len(pivots)] for i in range(Ncol) ]

def obtain_expansion_coefficients_over_specified_basis_z2(basis, v_list):
    """
    returns expansion coefficients of all vectors in v_list as linear combination of basis elements
    this function will NOT return an error if there is a vector in v_list that is not in span{basis},
    so this must be confirmed beforehand
    """
    full_v_list          = basis + v_list
    full_expansion_coefs = obtain_expansion_coefficients_z2(full_v_list)
    
    return full_expansion_coefs[len(basis):]

#
#    normal matrices functions
#

def construct_antisymmetric(angles, N):
    """
    angles is a list of real numbers of size N * (N - 1) // 2
    N is dimension of output matrix

    return is anti-symmetric matrix in which the angles are put in the matrix elements in the
    obvious way
    """
    X = np.zeros([N,N])

    tally = 0
    for i in range(N):
        for j in range(i+1, N):
            X[i,j] += angles[tally]
            X[j,i] -= angles[tally]
            tally  += 1
    assert np.allclose(X, -X.T)
    return X

def construct_orthogonal(angles, N):
    """
    angles is a list of real numbers of size N * (N - 1) // 2
    N is dimension of output matrix

    return is special-orthogonal matrix in which the angles are put in the matrix elements of the 
    anti-symmetric generator in the obvious way
    """
    return expm(construct_antisymmetric(angles, N))

def obtain_orthogonal_generator(O):
    """
    O is a special orthogonal matrix

    return is X such that C(X) = || e^X - O || is "small". Small is defined as: obtained via
    non-linear optimization of C(X) 
    """
    
    assert np.allclose(np.linalg.det(O), 1)

    N = O.shape[0]
    def cost(x):
        Oattempt = construct_orthogonal(x, N)
        diff     = O - Oattempt
        return np.sum(diff * diff)
    
    x0 = uniform(-np.pi/2, np.pi/2, N * (N - 1) // 2)

    op_result = op.minimize(cost, x0, method='BFGS')
    return construct_antisymmetric(op_result.x, N)

def SO_real_schur(X):
    """
    compute real Schur decomposition of antisymmetric matrix X such that basis transformation is special orthogonal
    """
    Lam, O = schur(X, output='real')

    if np.allclose(np.linalg.det(O), 1):
        return Lam, O.T
    
    N = O.shape[0]
    for i in range(N - 1):
        if Lam[i,i+1] != 0 and Lam[i+1,i] != 0:
            swap_columns(O, i, i + 1)
            Lam[i,i+1] = -Lam[i,i+1]
            Lam[i+1,i] = -Lam[i+1,i]
            break
    return Lam, O.T