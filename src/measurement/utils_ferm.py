import numpy as np
import scipy as sp
import itertools, sys
from openfermion import FermionOperator, hermitian_conjugated, normal_ordered

def get_one_body_terms(H):
    '''
    Return the one body terms in H
    '''
    one_body = FermionOperator.zero()
    for fw, val in H.terms.items():
        if len(fw) <= 2:
            one_body += FermionOperator(fw, val)
    return one_body

def get_hf(n_spinorb, nelec):
    '''
    Return the hf wfs given number of electrons and orbitals
    '''
    hf = np.zeros((n_spinorb, 1))
    hf[:nelec] = 1
    return np.flip(hf)

def get_creation_op(on):
    '''
    Get the corresponding creation operators based on ON vec 
    e.g. [0, 0, 1] = |001> -> a^+_0 
    '''
    on_flip = np.flip(on)
    op = FermionOperator.identity()
    for i in range(len(on_flip)):
        if on_flip[i] == 1:
            op = FermionOperator(term=(i, 1)) * op
    return op

def get_full_onvec(on):
    '''
    Return the onvec in full 2^n space as basis vector
    '''
    on = np.flip(on)
    n = len(on)
    idx = 0
    for i in range(len(on)):
        if on[i] == 1:
           idx += 2 ** i 
    vec = np.zeros((2**n, 1))
    vec[idx, 0] = 1
    return vec

def fermionic_action_tz(fw,on):
    """
    TZ doesn't flip the on array
    """
    phase = 1
    cur_on = on.copy()
    for fs in reversed(fw):
        if float(fs[1]) == cur_on[fs[0]]:
            cur_on = cur_on*0.0
            return on, 0
        else:
            cur_on[fs[0]] = float(fs[1])
            phase *= (-1)**(sum(cur_on[:fs[0]]))

    return cur_on, phase

def braket_tz(onl, onr, op):
    """
    Obtain <l|Op|r> using fermionic_action_tz
    """
    e = 0.0
    for fw, val in op.terms.items():
        cur_onr, phase = fermionic_action_tz(fw,onr)
       #print(e,fw,val,onl,onr,cur_onr)
        if all(onl == cur_onr):
            e += val * phase

    return e

def op_action_tz(op,on_list,onidx_list,coefs):
    """
    Acting an operator on a state, expanded in the on_list with 
    the coefficients in coefs
    """

    debug = False

    ndim = len(on_list)
    if debug: print(f'\nDimension of input vectors = {ndim}')

    on_list_post_act = on_list.copy()
    phased_val_list = np.zeros([ndim])
    onidx_list_post_act = onidx_list.copy()

    for fw, val in op.terms.items():
        for i, onvec in enumerate(on_list):
            onvec_prime, phase = fermionic_action_tz(fw, onvec)
            phased_val = val * phase
            idx_prime = get_on_idx(onvec_prime)
            if abs(phased_val) > 1.0e-8:
                if idx_prime in onidx_list_post_act:
                    ipos_prime = onidx_list_post_act.index(idx_prime)
                    phased_val_list[ipos_prime] += phased_val * coefs[i]
                else:
                    on_list_post_act.append(onvec_prime)
                    onidx_list_post_act.append(idx_prime)
                    phased_val_list = np.append(phased_val_list, phased_val * coefs[i])

    if debug:
        print(f'\nResult of op_action_tz, dimension = {len(on_list_post_act)}')
        for i in range(len(on_list_post_act)):
            print("".join(f"{phased_val_list[i]:12.6f} {onidx_list_post_act[i]:6d} {on_list_post_act[i]}"))

    return on_list_post_act, onidx_list_post_act, phased_val_list

def op_action_tz_CSF(op,CSF_input):
    """
    Acting an operator on a CSF and returning the resultant CSF
    """

    on_list = CSF_input[0]
    onidx_list = CSF_input[1]
    coefs = CSF_input[2]

    return op_action_tz_remove_0coef(op,on_list,onidx_list,coefs)

def op_action_tz_remove_0coef(op,on_list,onidx_list,coefs):
    """
    Acting an operator on a state, expanded in the on_list with 
    the coefficients in coefs
    """

    debug = False

    ndim = len(on_list)
    if debug: print(f'\nDimension of input vectors = {ndim}')

    on_list_post_act = on_list.copy()
   #phased_val_list = np.zeros([ndim])
    phased_val_list = [0.0]*ndim
    onidx_list_post_act = onidx_list.copy()

    for fw, val in op.terms.items():
        for i, onvec in enumerate(on_list):
            onvec_prime, phase = fermionic_action_tz(fw, onvec)
            phased_val = val * phase
            idx_prime = get_on_idx(onvec_prime)
            if abs(phased_val) > 1.0e-8:
                if idx_prime in onidx_list_post_act:
                    ipos_prime = onidx_list_post_act.index(idx_prime)
                    phased_val_list[ipos_prime] += phased_val * coefs[i]
                else:
                    on_list_post_act.append(onvec_prime)
                    onidx_list_post_act.append(idx_prime)
                   #phased_val_list = np.append(phased_val_list, phased_val * coefs[i])
                    phased_val_list.append(phased_val * coefs[i])

    if debug:
        print(f'\nResult of op_action_tz, dimension = {len(on_list_post_act)}')
        for i in range(len(on_list_post_act)):
            print("".join(f"{phased_val_list[i]:12.6f} {onidx_list_post_act[i]:6d} {on_list_post_act[i]}"))

   #Remove SD with zero coefficients
    ndim_post = len(on_list_post_act)
    for i in range(ndim_post-1,-1,-1):
        if np.isclose(phased_val_list[i],0.0):
            del on_list_post_act[i]
            del onidx_list_post_act[i]
            del phased_val_list[i]
           #phased_val_list.remove(phased_val_list[i])

    phased_val_list = np.array(phased_val_list)

    return on_list_post_act, onidx_list_post_act, phased_val_list


def fermionic_action(fw, on):
    '''
    Acting fermionic word to on vector . 
    Return new on with phase.
    Example: ((1, 1), (0, 0)) |00> = 0 
    '''
    on = np.flip(on)
    phase = 1 
    print(f'flipped on: {on}')
    for fs in reversed(fw):
       #print(fs, fs[1],on[fs[0]])
        if fs[1] == on[fs[0]]:
            return on, 0
        else:
            on[fs[0]] = fs[1]
            phase *= (-1)**(sum(on[fs[0]+1:]))
    return np.flip(on), phase

def braket(onl, onr, H):
    '''
    Obtain the value of <l|H|r>
    '''
   #def fermionic_action(fw, on):
   #    '''
   #    Acting fermionic word to on vector . 
   #    Return new on with phase.
   #    Example: ((1, 1), (0, 0)) |00> = 0 
   #    '''
   #    on = np.flip(on)
   #    phase = 1 
   #    for fs in reversed(fw):
   #        if fs[1] == on[fs[0]]:
   #            return on, 0
   #        else:
   #            on[fs[0]] = fs[1]
   #            phase *= (-1)**(sum(on[fs[0]+1:]))
   #    return np.flip(on), phase
    e = 0
    for fw, val in H.terms.items():
        cur_onr, phase = fermionic_action(fw, np.copy(onr))
        if phase != 0 and all(onl == cur_onr):
            e += val * phase
    return e

def deprecated_braket(onl, onr, H):
    '''
    Obtain the value of <l|H|r>
    '''
    lop = hermitian_conjugated(get_creation_op(onl))
    rop = get_creation_op(onr)
    op = lop * H * rop
    op = normal_ordered(op)
    return op.constant

def get_on_vec(idx, orb):
    '''
    Get the on vector form based on index and orb 
    e.g. |000> then |001> then |010> 
    '''
    on = np.zeros(orb)
    for i in range(orb):
        occ = idx % 2
        idx = idx // 2
        on[i] = occ
    return np.flip(on)
    
def get_on_idx(on):
    '''
    Get the index based on on vec 
    e.g. |000> -> 0. |001> -> 1. then |010> -> 2. 
    '''
    idx = 0
    on = np.flip(on)
    for i in range(len(on)):
        if on[i] == 1:
            idx += 2**i
    return idx

def get_ferm_basis(norb, nelec):
    '''
    Obtain the basis given number of orbitals and electrons
    '''
    choice = list(itertools.combinations([i for i in range(norb)], nelec))
    basis = []

    for indices in choice:
        curbasis = np.zeros((norb, 1))
        for idx in indices:
            curbasis[idx] = 1
        basis.append(curbasis)
    return basis

def get_fermionic_matrix(H : FermionOperator, n=None, nelec=None):
    '''
    Obtain the matrix form of fermionic operators 
    '''
    if n is None:
        n = get_spin_orbitals(H)

    if nelec is None:
        basis = []
        size = 2**n
        for i in range(size):
            basis.append(get_on_vec(i, n))
        matrix = np.zeros((size, size), np.complex128)
    else:
        basis = get_ferm_basis(n, nelec)
        size = len(basis)
        matrix = np.zeros((size, size), np.complex128)

    for i, ion in enumerate(basis):
        for j, jon in enumerate(basis):
            matrix[i, j] = braket(ion, jon, H)
    return matrix

def get_ferm_op_one(obt, spin_orb):
    '''
    Return the corresponding fermionic operators based on one body tensor
    '''
    n = obt.shape[0]
    op = FermionOperator.zero()
    for i in range(n):
        for j in range(n):
            if not spin_orb:
                for a in range(2):
                    op += FermionOperator(
                        term = (
                            (2*i+a, 1), (2*j+a, 0)
                        ), coefficient=obt[i, j]
                    )
            else:
                op += FermionOperator(
                    term = (
                        (i, 1), (j, 0)
                    ), coefficient=obt[i, j]
                )
    return op 

def get_ferm_op_two(tbt, spin_orb):
    '''
    Return the corresponding fermionic operators based on tbt (two body tensor)
    This tensor can index over spin-orbtals or orbitals
    '''
    n = tbt.shape[0]
    op = FermionOperator.zero()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n): 
                    if not spin_orb:
                        for a in range(2):
                            for b in range(2):
                                op += FermionOperator(
                                    term = (
                                        (2*i+a, 1), (2*j+a, 0),
                                        (2*k+b, 1), (2*l+b, 0)
                                    ), coefficient=tbt[i, j, k, l]
                                )
                    else:
                        op += FermionOperator(
                            term=(
                                (i, 1), (j, 0),
                                (k, 1), (l, 0)
                            ), coefficient=tbt[i, j, k, l]
                        )
    return op

def get_ferm_op(tsr, spin_orb=False):
    '''
    Return the corresponding fermionic operators based on the tensor
    This tensor can index over spin-orbtals or orbitals
    '''
    if len(tsr.shape) == 4:
        return get_ferm_op_two(tsr, spin_orb)
    elif len(tsr.shape) == 2:
        return get_ferm_op_one(tsr, spin_orb)

def get_spin_orbitals(H : FermionOperator):
    '''
    Obtain the number of spin orbitals of H
    '''
    n = -1 
    for term, val in H.terms.items():
        if len(term) == 4:
            n = max([
                n, term[0][0], term[1][0],
                term[2][0], term[3][0]
            ])
    n += 1 
    return n

def fci2hf(fci, n=None, tiny=1e-6):
    '''
    Convect fci solutions to pairs of HF. [(hf_i, coeff_i), ...]
    '''
    if n is None:
        n = int(np.log2(len(fci)))
    hf_pairs = []
    
    norm = 0
    for idx, val in enumerate(fci):
        if abs(val) > tiny:
            norm += np.abs(val) ** 2
            hf_pairs.append([get_on_vec(idx, n), val])
    
    scale = norm ** (1/2)
    for i in range(len(hf_pairs)):
        hf_pairs[i][1] = hf_pairs[i][1] / scale
    return hf_pairs

def hfp_braket(hfp, Hf):
    '''
    Given Hermitian Operator Hf and hf pair [(hf_i, coeff_i), ...]
    Obtain <Hf> 
    '''
    e = 0
    nhf = len(hfp)
    for i in range(nhf):
        for j in range(i, nhf):
            cur_val = np.conj(hfp[i][1]) * hfp[j][1] * braket(hfp[i][0], hfp[j][0], Hf)
            if abs(np.imag(cur_val)) > 1e-8:
                print("i: {}. j: {}".format(i, j))
                print("ival: {}. jval: {}".format(hfp[i][1], hfp[j][1]))
            if i == j:
                e += cur_val
            else:
                e += 2 * np.real(cur_val)
    return e

def get_two_body_tensor(H : FermionOperator, n = None):
    '''
    Obtain the 4-rank tensor that represents two body interaction in H. 
    In physics ordering a^ a^ a a 
    '''
    # number of spin orbitals 
    if n is None:
        n = get_spin_orbitals(H)

    tbt = np.zeros((n, n, n, n))
    for term, val in H.terms.items():
        if len(term) == 4:
            tbt[
                term[0][0], term[1][0],
                term[2][0], term[3][0]
            ] = val
    return tbt 

def get_chemist_tbt(H : FermionOperator, n = None, spin_orb=False):
    '''
    Obtain the 4-rank tensor that represents two body interaction in H. 
    In chemist ordering a^ a a^ a. 
    In addition, simplify tensor assuming symmetry between alpha/beta coefficients
    '''
    # getting N^4 phy_tbt and then (N/2)^4 chem_tbt 
    phy_tbt = get_two_body_tensor(H, n)
    chem_tbt = np.transpose(phy_tbt, [0, 3, 1, 2])

    if spin_orb:
        return chem_tbt

    # Spin-orbital to orbital 
    n_orb = phy_tbt.shape[0]
    n_orb = n_orb // 2
    alpha_indices = list(range(0, n_orb * 2, 2))
    beta_indices = list(range(1, n_orb * 2, 2))
    chem_tbt = chem_tbt[
        np.ix_(alpha_indices, alpha_indices,
                    beta_indices, beta_indices)]

    return chem_tbt

def separate_diagonal_tbt(chem_tbt):
    '''
    Separate the terms representing n_p n_q terms 
    '''
    n = chem_tbt.shape[0]
    diag_tbt = np.zeros((n, n, n, n))
    ndia_tbt = chem_tbt.copy()

    for i in range(n):
        for j in range(n):
            diag_tbt[i, i, j, j] = ndia_tbt[i, i, j, j]
            ndia_tbt[i, i, j, j] = 0
    return diag_tbt, ndia_tbt

def separate_number_op(op:FermionOperator):
    '''
    Removing number operator from op 
    '''
    diag = FermionOperator.zero()
    ndia = FermionOperator.zero()
    for term, val in op.terms.items():
        cr = []
        an = [] 
        for fw in term:
            if fw[1] == 1:
                cr.append(fw[0])
            else:
                an.append(fw[0])
        for idx in cr:
            if idx in an:
                an.remove(idx)
        curterm = FermionOperator(term=term, coefficient=val)
        if len(an) == 0:
            diag += curterm
        else:
            ndia += curterm
    return diag, ndia

def get_openfermion_hf(n_qubits, n_electrons):
    """Compute the ground hartree fock state in openfermion's format |psi><psi| 

    Args:
        n_qubits: Number of qubits (spin_orbitals).
        n_electrons: Number of electrons in hartree fock. 

    Returns:
        wfs (sparse_matrix): Density that represents the Hartree Fock state. 
    """
    # Construct ON vector
    occupation_vec = np.zeros(n_qubits)
    occupation_vec[-n_electrons:] = 1

    # Identify corresponding index in exponential basis
    idx = 0
    for i in range(len(occupation_vec)):
        if occupation_vec[i] == 1:
            idx += 2 ** i
    idx_tuple = (idx,)

    # Construct sparse matrix
    dim = 2**n_qubits
    return sp.sparse.csr_matrix(((1,), (idx_tuple, idx_tuple)), shape=(dim, dim))

def get_S_plus(n_spinorb):
    """
    S^+ operator in 2nd quantization
    """

    n_spatial = n_spinorb // 2
    S_plus = FermionOperator()
    for i in range(n_spatial):
        ia, ib = 2*i, 2*i+1
        term = ((ia,1),(ib,0))
        coef = 1.0
        S_plus += FermionOperator(term,coef)
        

    return S_plus

def get_S_minus(n_spinorb):
    """
    S^- operator in 2nd quantization
    """

    n_spatial = n_spinorb // 2
    S_minus = FermionOperator()
    for i in range(n_spatial):
        ia, ib = 2*i, 2*i+1
        term = ((ib,1),(ia,0))
        coef = 1.0
        S_minus += FermionOperator(term,coef)


    return S_minus

def get_S_z(n_spinorb):
    """
    S_z operator in 2nd quantization
    """

    n_spatial = n_spinorb // 2
    S_z = FermionOperator()
    for i in range(n_spatial):
        ia, ib = 2*i, 2*i+1
        term1 = ((ia,1),(ia,0))
        term2 = ((ib,1),(ib,0))
        coef = 0.5
        S_z += FermionOperator(term1,coef) - FermionOperator(term2,coef)

    return S_z

def get_S_squared(n_spinorb):
    """
    S^2 operator in 2nd quantization
    """

    S_plus  = get_S_plus(n_spinorb)
    S_minus = get_S_minus(n_spinorb)
    S_z     = get_S_z(n_spinorb)

    I_op = FermionOperator('')
    S_squared = S_plus*S_minus + S_z*(S_z - I_op)

    return S_squared

def get_S_sq_plus_minus_z(n_spinorb):
    """
    Return S^2, S^+, S^-, and S_z operator
    """

    S_plus  = get_S_plus(n_spinorb)
    S_minus = get_S_minus(n_spinorb)
    S_z     = get_S_z(n_spinorb)
    S_square = get_S_squared(n_spinorb)

    return S_square, S_plus, S_minus, S_z

def judge_eigen_on_list(op,on_list,idx_list,coef,expected_eigval=None):
    """
    Judge whether the input state with a list of on vectors, a list of corresponding
    integer indices, and a vector of corresponding coefficients is an eigenstate
    of the fermion operator op. If yes, return an eigenvalue
    """

    leigen = False
    eigval = 0.0 #default eigenvalue 0 will be returned even if the state is not an eigenstate
    on_list_post_op, idx_list_post_op, coef_post_op = op_action_tz_remove_0coef(op,on_list,idx_list,coef)
   #Special case of 0 eigenvalue
    if len(on_list_post_op) == 0:
        leigen = True
        if expected_eigval != None and eigval != expected_eigval:
            print(f'eigenvalue is inconsistent with expectation: {eigval,expected_eigval}')
            print('Bombing out')
            sys.exit()
        return leigen, eigval
   #If the operator generates basis not in the same space as the input state, the input is not an eigenstate
    if idx_list_post_op != idx_list: return leigen, eigval

    leigen = True
    eigval = coef_post_op[0] / coef[0]
    for ii in range(len(coef_post_op)):
        coef_ratio = coef_post_op[ii] / coef[ii]
        if not np.isclose(coef_ratio,eigval): leigen = False

    if leigen and expected_eigval != None and not np.isclose(eigval,expected_eigval):
        print(f'eigenvalue is inconsistent with expectation: {eigval,expected_eigval}')
        print('Bombing out')
        sys.exit()
        

    if not leigen: eigval = 0.0
    if not leigen and expected_eigval != None:
        print(f'Not an eigenstate and cannot satisfy eigenvalue = {expected_eigval}')
        print('Bombing out')
        sys.exit()
    return leigen, eigval

#def generate_CASCI_space(Norb_total, Nel_total,Nactorb,Nactel,S_by2,debug=False):
#    """
#    Generate all CSFs in a CAS space with total spin S. The integral SX2 is read-in
#    as S_by2. S_by2 also gives the minimum number of singly occupied orbitals
#    """
#
#    from itertools import combinations
#    import copy
#
#    if debug: print('\nIn generate_CASCI_space')
#
#    N_SOMO_min = S_by2
#    if (Nactel - S_by2) % 2 != 0:
#        print(f'The number of active electrons {Nactel} is inconsistent with SX2 {S_by2}')
#        print('Bombing out!')
#        sys.exit()
#
#    N_SOMO_max = 2*Nactorb - Nactel
#
#    Ninactorb = (Nel_total - Nactel) // 2
#    if debug: 
#        print(f'N_SOMO_min {N_SOMO_min}, N_SOMO_max {N_SOMO_max}')
#        print(f'# of inactive orbitals: {Ninactorb}')
#
#    list_actmo = []
#    for orb in range(Ninactorb,Ninactorb+Nactorb):
#        list_actmo.append(orb)
#
#    for N_SOMO in range(N_SOMO_min, N_SOMO_max+1,2):
#
#        N_alpha = (N_SOMO + S_by2) // 2
#        N_beta  = (N_SOMO + S_by2) // 2
#
#        l_opensh = False
#        if N_SOMO != 0: 
#            l_opensh = True
#            geneological_SD_CSF(N_alpha, N_beta,debug)
#
#        Nel_pair = Nactel - N_SOMO
#        if Nel_pair % 2 != 0:
#             print(f'Nactel, {Nactel}, N_SOMO, {N_SOMO}, Nel_pair, {Nel_pair} not even')
#             print(f'Bombing out')
#             sys.exit()
#        Ndmo = Nel_pair // 2
#        list_SOMO_comb = list(combinations(list_actmo,N_SOMO))
#       #print(list_SOMO)
#        for SOMO_comb in list_SOMO_comb:
#            list_left_orb = copy.deepcopy(list_actmo)
#            for SOMO in SOMO_comb:
#                list_left_orb.remove(SOMO)
#           #print(f'SOMOs: {SOMO_comb}, left over orbitals: {list_left_orb}')
#            list_DMO_comb = list(combinations(list_left_orb,Ndmo))
#            for DMO_comb in list_DMO_comb:
#                list_VMO = copy.deepcopy(list_left_orb)
#                for MO in DMO_comb:
#                    list_VMO.remove(MO)
#
#               #print(f'SOMOs: {SOMO_comb}, DMOs: {DMO_comb}, VMOs: {list_VMO}')
#
#def geneological_SD_CSF(N_alpha, N_beta,debug=False):
#    """
#    Geneological coupling between SDs and CSFs with specific numbers of alpha and
#    beta electrons
#    """
#
#    if debug: print(f'\nIn geneological_SD_CSF, N_alpha = {N_alpha}, N_beta = {N_beta}')
#
#    S_by2 = N_alpha - N_beta
#    M_by2 = S_by2
#    S_sq_eigval = float(S_by2)*0.5*np.sqrt(float(S_by2)*0.5+1.0)
#
#    N_open = N_alpha + N_beta
#    list_SOMO = []
#    for i in range(N_open):
#        list_SOMO.append(i)
#    
#    from itertools import combinations
#
#    comb_alpha_orb = list(combinations(list_SOMO,N_alpha))
#    if debug: print(comb_alpha_orb)
#    
#    list_pvec = []
#    list_tvec = []
#    for item in comb_alpha_orb:
#        pvec = np.full(N_open,-1)
#        for orb in item:
#            pvec[orb] = 1
#    
#       #print(pvec)
#        list_pvec.append(pvec)
#        l_tvec = True
#        for orb in range(N_open):
#            if np.sum(pvec[:orb]) < 0:
#                l_tvec = False
#                break
#    
#        if l_tvec: list_tvec.append(pvec)
#    
#    
#    if debug:
#        print('\nlist_pvec:')
#        print(list_pvec)
#        print('\nlist_tvec:')
#        print(list_tvec)
#
#    list_Pvec = []
#    for item in list_pvec:
#       #print(item)
#        Pvec = np.full(N_open,0)
#        for i in range(len(item)):
#           #print(i,np.sum(item[:i+1]))
#            Pvec[i] = np.sum(item[:i+1])
#       #print(Pvec)
#        list_Pvec.append(Pvec)
#    
#    list_Tvec = []
#    for item in list_tvec:
#       #print(item)
#        Tvec = np.full(N_open,0)
#        for i in range(len(item)):
#           #print(i,np.sum(item[:i+1]))
#            Tvec[i] = np.sum(item[:i+1])
#       #print(Tvec)
#        list_Tvec.append(Tvec)
#    
#    if debug:
#        print('\nlist_Pvec:')
#        print(list_Pvec)
#        print('\nlist_Tvec:')
#        print(list_Tvec)
#
#    list_CSF = []
#    
#    #loop over all CSFs
#    for iCSF in range(len(list_Tvec)):
#        Tvec = list_Tvec[iCSF]
#        tvec = list_tvec[iCSF]
#        list_Pvec_include = []
#        list_pvec_include = []
#        coef_p_t = []
#        onlist = []
#        onidx_list = []
#       #Kick out Pvec and pvec if |P_N| > T_N 
#        for iSD in range(len(list_Pvec)):
#            Pvec = list_Pvec[iSD]
#            pvec = list_pvec[iSD]
#            l_include = True
#            CoupCoef = 1.0
#            for orb in range(N_open):
#                if abs(Pvec[orb]) > Tvec[orb]:
#                    l_include = False
#                    break
#                Tn_by2, Pn_by2, tn_by2, pn_by2 = Tvec[orb], Pvec[orb], tvec[orb], pvec[orb]
#                CoupCoef *= SD_CSF_CoupCoef(Tn_by2, Pn_by2, tn_by2, pn_by2)
#            if not l_include: continue
#            list_Pvec_include.append(Pvec)
#            list_pvec_include.append(pvec)
#            coef_p_t.append(CoupCoef)
#            onvec = np.zeros([2*N_open])
#            for orb in range(N_open):
#                if pvec[orb] == 1:
#                    onvec[2*orb]   = 1.0
#                else:
#                    onvec[2*orb+1] = 1.0
#            onlist.append(onvec)
#            onidx_list.append(get_on_idx(onvec))
#    
#        coef_p_t = np.array(coef_p_t)
#        list_CSF.append([onlist,onidx_list,coef_p_t])
#
#    if debug: check_spin_adapted_CSF(list_CSF,N_alpha, N_beta)
#
#def check_spin_adapted_CSF(list_CSF,N_alpha, N_beta):
#    """
#    Check whether the list of CSFs generated by geneological coupling of SDs
#    are eigenstates of S^2 and Sz with appropriate eigenvalue. Also check
#    their orthonormality
#    """
#
#    #Check orthonormality
#    for iCSF, CSF_bra in enumerate(list_CSF):
#        for jCSF in range(iCSF,len(list_CSF)):
#            CSF_ket = list_CSF[jCSF]
#            Selm = overlap_LCSD(CSF_bra[0],CSF_bra[1],CSF_bra[2],CSF_ket[0],CSF_ket[1],CSF_ket[2])
#            if iCSF == jCSF and not np.isclose(Selm,1.0):
#                print(f'CSF {iCSF} not normalized: {Selm}. Bombing out!')
#                sys.exit()
#            if iCSF != jCSF and not np.isclose(Selm,0.0):
#                print(f'CSFs {iCSF, jCSF} not orthogonal: {Selm}. Bombing out!')
#                sys.exit()
#
#def SD_CSF_CoupCoef(S_by2,M_by2,tN_by2,sigma_by2):
#    """
#    The Clebsch-Gordan coefficients in Eq. 2.6.5 and 2.6.6 of the Helgaker book.
#    All inputs are integers and are the variables in the equations multiplied by 2.
#    """
#
#    if sigma_by2 != 1 and sigma_by2 != -1:
#        print(f'sigma_by2 {sigma_by2}, is neither 1 or -1')
#        print('Bombing out')
#        sys.exit()
#    
#    S = float(S_by2) / 2.0
#    M = float(M_by2) / 2.0
#    sigma = float(sigma_by2) / 2.0
#
#    if tN_by2 == 1:
#        CoupCoef = S + 2.0*sigma*M
#        CoupCoef /= 2.0*S
#        CoupCoef = np.sqrt(CoupCoef)
#    elif tN_by2 == -1:
#        CoupCoef = S + 1.0 - 2.0*sigma*M
#        CoupCoef /= 2.0*(S+1.0)
#        CoupCoef = np.sqrt(CoupCoef)
#        CoupCoef *= -2.0*sigma
#    else:
#        print(f'Unrecognized tN_by2 {tN_by2}, which shall only be +1 and -1.')
#        print('Bombing out!')
#        sys.exit()
#    
#    return CoupCoef
