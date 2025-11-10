import numpy as np
import scipy, time, sys, copy
from openfermion import FermionOperator, hermitian_conjugated, normal_ordered, get_ground_state
from seniority.src.measurement.utils_ferm import (
    get_on_idx, op_action_tz_remove_0coef,
    braket_tz, get_S_squared, get_S_z, judge_eigen_on_list
)
import itertools
from itertools import combinations
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
# from pylanczos import PyLanczos
from joblib import Parallel, delayed

def get_kappa_isig_asigprim(i_sig,a_sig,debug=False):
    """
    i_sig is the spin orbital with spatial orbital i and spinor sig
    a_sig is the spin orbital with spatial orbital a and spinor sig
    The two spinors may not be idenical
    """
    if debug: print(f'in get_kappa_isig_asig')
    if not isinstance(i_sig,str):
        print(f'{i_sig} is not a string. Bombing out')
        sys.exit()
    if not isinstance(a_sig,str):
        print(f'{a_sig} is not a string. Bombing out')
        sys.exit()

    if debug: print(i_sig,a_sig)
    i_spatial = int(i_sig[:-1])
    i_spinor = i_sig[-1]
    a_spatial = int(a_sig[:-1])
    a_spinor = a_sig[-1]
    if debug: print(i_spatial,i_spinor,a_spatial,a_spinor)
    if i_spinor == 'a':
        i_spinorb = i_spatial*2
    elif i_spinor == 'b':
        i_spinorb = i_spatial*2+1
    else:
        print(f'Unrecognized spinor for {i_sig}')

    if a_spinor == 'a':
        a_spinorb = a_spatial*2
    elif a_spinor == 'b':
        a_spinorb = a_spatial*2+1
    else:
        print(f'Unrecognized spinor for {a_sig}')

    if debug: print(i_spinorb,a_spinorb)

    kappa_isig_asigprim = get_kappa_ia(i_spinorb,a_spinorb)
    return kappa_isig_asigprim

def get_kappa_ia(i,a):
    """
    Read in two indices of spin orbital and return
    kappa_ia = a_a^+ a_i - a_i^+ a_a
    """

    term = ((a,1),(i,0))
    coef = 1.0
    kappa_ia = FermionOperator(term,coef)
    
    kappa_ia -= hermitian_conjugated(kappa_ia)

    return kappa_ia

def get_kappa_ijab(i,j,a,b):
    """
    Read in four indices of spin orbitals i, j, a, b, and return
    kappa_ijab = a_a^+ a_b^+ a_j a_i - h.c.
    """

    term = ((a,1),(b,1),(j,0),(i,0))
    coef = 1.0
    kappa_ijab  = FermionOperator(term,coef)
    kappa_ijab -= hermitian_conjugated(kappa_ijab)

    return kappa_ijab

def get_kappa_ijab_spaspn(isig,jsig,asig,bsig):
    """
    Read in four indices of spatial-spin orbitals ia, jb, aa, bb, etc. and return
    the corresponding kappa_ijab = a_aa^+ a_bb^+ a_jb a_ia - h.c. etc.
    """


    i_spinorb = ia_to_2i_ib_to_2iplus1(isig)
    j_spinorb = ia_to_2i_ib_to_2iplus1(jsig)
    a_spinorb = ia_to_2i_ib_to_2iplus1(asig)
    b_spinorb = ia_to_2i_ib_to_2iplus1(bsig)


    kappa_ijab = get_kappa_ijab(i_spinorb,j_spinorb,a_spinorb,b_spinorb)

    return kappa_ijab

def ia_to_2i_ib_to_2iplus1(i_sig):
    """
    Convert ia to 2i and ib to 2i+1, i.e., convert spatial-spin orbital index to
    spin orbital index
    """

    i_spatial = int(i_sig[:-1])
    i_spinor = i_sig[-1]
    if i_spinor == 'a':
        i_spinorb = 2*i_spatial
    elif i_spinor == 'b':
        i_spinorb = 2*i_spatial+1
    else:
        print(f'Unrecognized spinor for {i_sig}')

    return i_spinorb

def get_Tia_00(i_spatial,a_spatial):
    """
    Read in two spatial orbitals indexing integers i and a and return
    T_ia_00 = kappa_ialpha_aalpha + kappa_ibeta_abeta
    """

    ia = str(i_spatial)+'a'
    aa = str(a_spatial)+'a'
    ib = str(i_spatial)+'b'
    ab = str(a_spatial)+'b'

    Tia_00  = get_kappa_isig_asigprim(ia,aa) + get_kappa_isig_asigprim(ib,ab)
    Tia_00 *= np.sqrt(0.5)
    return Tia_00

def get_Tia_1m(i_spatial,a_spatial):
    """
    Read in two spatial orbitals indexing integers i and a and return
    T_ia_1p1 = kappa_ibeta_aalpha
    T_ia_1_0 = 1/sqrt(2) ( kappa_ibeta_abeta - kappa_ialpha_aalpha )
    T_ia_1m1 = -kappa_ialpha_abeta
    """

    ia = str(i_spatial)+'a'
    aa = str(a_spatial)+'a'
    ib = str(i_spatial)+'b'
    ab = str(a_spatial)+'b'

    Tia_1p1 = get_kappa_isig_asigprim(ib,aa)
    Tia_1_0 = get_kappa_isig_asigprim(ib,ab) - get_kappa_isig_asigprim(ia,aa)
    Tia_1_0 *= np.sqrt(0.5)
    Tia_1m1 = -get_kappa_isig_asigprim(ia,ab)

    return Tia_1p1, Tia_1_0, Tia_1m1

def get_Tiiaa_00(i_spatial,a_spatial):
    """
    Read in two spatial orbitals indexing integers i and a and return
    Tiiaa_00 = kappa_ialpha_ibeta_aalpha_abeta
    """

    ia = str(i_spatial)+'a'
    aa = str(a_spatial)+'a'
    ib = str(i_spatial)+'b'
    ab = str(a_spatial)+'b'

    Tiiaa_00 = get_kappa_ijab_spaspn(ia,ib,aa,ab)

    return Tiiaa_00

def get_E_ia(i_spinorb,a_spinorb):
    """
    Read in spin orbital indices i and a, and return the excitaiton
    operator E_i^a = a_a^+ a_i
    """

    term = ((a_spinorb,1),(i_spinorb,0))
    coef = 1.0

    E_ia = FermionOperator(term,coef)

    return E_ia

def get_pair_excite(i_spatial,a_spatial):
    """
    Read in two spatial orbitals indexing integers i and a and return
    E_ia^aa * E_ib^ab
    """

    ia = 2*i_spatial
    aa = 2*a_spatial
    ib = ia + 1
    ab = aa + 1
    E_ia_aa = get_E_ia(ia,aa)
    E_ib_ab = get_E_ia(ib,ab)

   #print(E_ia_aa,E_ib_ab,E_ia_aa*E_ib_ab)

    return E_ia_aa*E_ib_ab

def print_matrix(matrix,n_per_group=6):

    nrow = matrix.shape[0]
    ncolumn = matrix.shape[1]

    ngroup = ncolumn // n_per_group
    nleft = ncolumn - ngroup*n_per_group

    for igroup in range(ngroup):
        imin = igroup*n_per_group
        imax = imin + n_per_group
        for row in matrix[:,imin:imax]:
            print(" ".join(f"{num:12.6f}" for num in row))

        print("")

    if nleft > 0:
        if ngroup == 0:
            imin = 0
        else:
            imin = imax
        imax = ncolumn
        for row in matrix[:,imin:imax]:
            print(" ".join(f"{num:12.6f}" for num in row))

        print("")

def print_eigen_solution(eigen_values,eigen_vectors):

    nsolut = len(eigen_values)
    n_per_group = 5
    ngroup = nsolut // n_per_group
    nleft = nsolut - ngroup*n_per_group
   #print(nsolut,ngroup,nleft)
    for igroup in range(ngroup):
        imin = igroup*n_per_group
        imax = imin + n_per_group
        print(" ".join(f"{num:12.6f}" for num in eigen_values[imin:imax]))
        print("")

        for row in eigen_vectors[:,imin:imax]:
            print(" ".join(f"{num:12.6f}" for num in row))

        print("")

    if nleft > 0:
        if ngroup == 0:
            imin = 0
        else:
            imin = imax
        imax = nsolut
        print(" ".join(f"{num:12.6f}" for num in eigen_values[imin:imax]))
        print("")

        for row in eigen_vectors[:,imin:imax]:
            print(" ".join(f"{num:12.6f}" for num in row))

        print("")

def make_and_apply_U(list_ampld,list_genmat,list_mp2_Ecorr,Hmat,vec_before_U,debug=False):
    """
    Read in rotational amplitudes, generator matrices, Hamiltonian matrix, and an inivital vector
    in a space, construct the U matrix, act the rotational U matrix on the
    vector, and calculate the average energy of the resultant state
    """

    assert len(list_ampld) == len(list_genmat)

    if debug: print(f'\nStep Ecorr.        Step MP2 Ecorr.        Accum. Step Ecorr.       Accum. Ecorr.')
    n_basis = Hmat.shape[0]
    Umat = np.eye(n_basis)
    E_input_state = vec_before_U.transpose()@Hmat@vec_before_U
    sum_Ecorr_separate_generator = 0.0
    for igen,genmat in enumerate(list_genmat):
        norm_genmat = np.linalg.norm(genmat)
        if np.isclose(norm_genmat,0.0):
            if debug:
                print(f'Skipping the 0 generator for igen {igen}')
            continue
       #print(f'igen = {igen}, {list_ampld[igen]}')
       #print(f'igen = {igen}, genmat:')
       #print(genmat)
       #tstart = time.perf_counter()
       #exp_genmat_quick = quick_expon_genmat(list_ampld[igen],genmat,False)
       #tend   = time.perf_counter()
       #print(f'Time to get exp_genmat_quick: {tend - tstart}')
       #print(f'exp_genmat_quick:')
       #print_matrix(exp_genmat_quick)
       #tstart = time.perf_counter()
        exp_genmat = scipy.linalg.expm(list_ampld[igen]*genmat)
       #tend   = time.perf_counter()
       #print(f'Time to get exp_genmat: {tend - tstart}')
       #assert np.isclose(np.linalg.norm(exp_genmat_quick - exp_genmat),0.0)
       #print(f'exp_genmat:')
       #print_matrix(exp_genmat)
       #print(f'Norm of deviation matrix: {np.linalg.norm(exp_genmat_quick - exp_genmat)}')
        Umat @= exp_genmat
        if debug:
            E_step = vec_before_U.transpose()@exp_genmat.transpose()@Hmat@exp_genmat@vec_before_U
            sum_Ecorr_separate_generator += E_step - E_input_state
            E_update = vec_before_U.transpose()@Umat.transpose()@Hmat@Umat@vec_before_U
            E_mp2_corr_1_gen = np.sum(np.array(list_mp2_Ecorr[igen]))
           #print(f'E_corr induced by the {igen}-th generator itself: {E_step - E_input_state}')
           #print(f'Sum of independent correlation energies by individual generator: {sum_Ecorr_separate_generator}')
            if debug:
                print(E_step - E_input_state,E_mp2_corr_1_gen,sum_Ecorr_separate_generator,E_update-E_input_state)

       #print(f'Umat, igen = {igen}')
       #print(Umat)

   #print(f'Umat in E_func_of_rotamp:')
   #print(Umat)
    vec_after_U = Umat@vec_before_U
    E_avrg = vec_after_U.transpose()@Hmat@vec_after_U

    return E_avrg, Umat, vec_after_U

def quick_expon_genmat(theta,genmat,debug=False):
    """
    Read in a generator matrix and returns the exponential of the generator matrix
    This code is fast when genmat is a sparse antihermitian matrix.
    It HOWEVER turns out not as fast as the built-in expm code. OK, this code is
    not used momentarily.
    """

    if debug: print('In quick_expon_genmat')

    ndim = genmat.shape[0]

   #expmat = np.zeros([ndim,ndim])
    expmat = np.eye(ndim)

   #assert genmat is a real antihermitian matrix
    assert np.isclose(np.sum((genmat.transpose() + genmat)**2),0.0)
    rows, cols = np.nonzero(genmat)
    if debug: print(rows,cols,len(rows),len(cols))

    row_col_considered = []
    for irow, row in enumerate(rows):
        col = cols[irow]
        if row in row_col_considered or col in row_col_considered:
            if debug: print(f'Row {row} or Col {col} of nonzero item {irow} has been considered')
            continue
        row_considering = []
        item_match = np.where(rows == row)[0]
       #print(f'item_match for row = {row},{item_match}')
        row_considering.append(row)
        for item in item_match:
           #print(f'item = {item}')
            rol_prm = cols[item]
            if rol_prm not in row_considering: row_considering.append(rol_prm)

        if debug: print(f'row_considering after 1st sweeping: {row_considering}')

        nrow_old = len(row_considering)
        nrow_new = -1
        nsweep = 1
        lsweep = True
        while lsweep:
            nsweep += 1
            for row_prm in row_considering:
                item_match = np.where(rows == row_prm)[0]
                for item in item_match:
                    rol_dprm = cols[item]
                    if rol_dprm not in row_considering: row_considering.append(rol_dprm)
            nrow_new = len(row_considering)
            if nrow_new == nrow_old:
                lsweep = False
            else:
                nrow_old = nrow_new

        if debug: print(f'row_considering after {nsweep} sweepings: {row_considering}')
        row_col_considered.extend(row_considering)
            
       #nsubdim = len(row_considering) 
       #subgenmat = genmat[row_considering[:, None],row_considering]
        subgenmat = genmat[np.ix_(row_considering,row_considering)]
       #print('sub genmat:')
       #print_matrix(subgenmat)

        subexpmat = scipy.linalg.expm(theta*subgenmat)
        if debug:
            print('sub expmat:')
            print_matrix(subexpmat)

        expmat[np.ix_(row_considering,row_considering)]=subexpmat

        if debug: 
            print('Updated full expmat')
            print_matrix(expmat)

    return expmat

def make_iapair_genmat_fast(list_iapair,list_basis,list_iapair_st_pairs,debug=False):
    """
    Make list of generator matrices for mp2 ia pairs
    """
    if debug: print('\nIn make_iapair_genmat_fast')

    n_basis = len(list_basis)

    if debug:
        print('list_iapair in make_iapair_genmat_fast')
        for item in list_iapair:
            print(item)

    list_genmat = []
    for ipairs, pairs in enumerate(list_iapair):
       #if len(pairs) > 2:
       #    print(pairs)
       #    print(f'Now the fast code only supports axial symmetr with degeneracy <= 2. Bombing out!')
       #    sys.exit()
        genmat = csr_matrix((n_basis,n_basis))
        for pair in pairs:
            lfound = False
            for item in list_iapair_st_pairs:
                if item[0] == pair:
                    lfound = True
                    states_coupled = item[1:]
                    break
           #if not lfound:
           #    print(f'{pair} not found in list_iapair_st_pairs:')
           #    for item in list_iapair_st_pairs:
           #        print(item)
           #    print('Bombing out!')
           #    sys.exit()
           #lfound = False is very normal. The mp2 ia pair may involve SOMO in the states.
           #So, we simply continue to the next mp2 pair
            if not lfound: continue
            if debug:
                print(f'The following state pairs: {states_coupled}')
                print(f'are coupled by pair excitations {pair}')
            for states_pair in states_coupled:
                state_high = states_pair[0]
                state_low  = states_pair[1]
                if state_high <= state_low:
                    print(f'state_high <= state_low, {states_pair}. Bombing out!')
                    sys.exit()
                genmat[state_high,state_low ] =  1.0
                genmat[state_low ,state_high] = -1.0

            if debug:
                print(f'genmat for exxcitation pair {pair}')
                print(genmat)

       #genmat_sq = genmat@genmat
       #eigval,eigvec = np.linalg.eigh(genmat_sq.toarray())
       #dnorm_fact = -min(eigval)
       #if dnorm_fact > 1.0:
       #    dnorm_fact = np.sqrt(dnorm_fact)
       #    genmat_normalized = genmat / dnorm_fact
       #else:
       #    genmat_normalized = genmat

       #genmat_sq = genmat_normalized@genmat_normalized
       #eigval,eigvec = np.linalg.eigh(genmat_sq.toarray())
       #print(f'ipairs: {ipairs}, pairs: {pairs}')
       #print(f'eigenvalues of genmat_normalized^2: {eigval}')
       #if len(pairs) == 1:
       #    genmat_cube = genmat@genmat@genmat
       #    assert np.isclose(scipy.sparse.linalg.norm(genmat_cube + genmat),0.0)
       #if len(pairs) == 2 and (pairs[0][0] == pairs[1][0] or pairs[0][1] == pairs[1][1]):
       #    genmat_normalized = genmat / np.sqrt(2.0)
       #    genmat_cube = genmat_normalized@genmat_normalized@genmat_normalized
       #    if not np.isclose(scipy.sparse.linalg.norm(genmat_cube + genmat_normalized),0.0):
       #        print(f'{pairs} does not pass the genmat^3 = genmat test')
       #        print('genmat:')
       #        print_matrix(genmat_normalized.toarray())
       #        print('genmat^3:')
       #        print_matrix(genmat_cube.toarray())
       #        sys.exit()
        

        list_genmat.append(genmat)

    if debug:
        list_genmat_check = make_iapair_genmat(list_iapair,list_basis)
        for ii,genmat_check in enumerate(list_genmat_check):
            genmat_sparse = list_genmat[ii]
            genmat_check_sparse = csr_matrix(genmat_check)
            norm_diff = scipy.sparse.linalg.norm(genmat_sparse - genmat_check_sparse)
            if not np.isclose(norm_diff,0.0):
                print(f'Not identical genmats for {list_iapair[ii]}')
                print('genmat_sparse:')
                print(genmat_sparse)
                print('genmat_check_sparse:')
                print(genmat_check_sparse)
                print('Bombing out!')
                sys.exit()

    return list_genmat

def make_Umat_decomp_genmat(list_genmat,list_ampld,debug=False):
    """
    Construct U matrix using decomposed genmats
    """

    if debug: print('\nIn make_Umat_decomp_genmat')

    if len(list_genmat) != len(list_ampld):
        print('Inconsistent dimensios in list_genmat and list_ampld: {len(list_genmat),len(list_ampld)}')
        sys.exit()
    for ii, decomp_genmat in enumerate(list_genmat):
        [list_unique_eigval,list_prj_genmat,eigvec,list_start_end_ind] = decomp_genmat
        theta = list_ampld[ii]
        Umat_1theta = make_analytical_U_decomp_genmat(theta,list_unique_eigval,list_prj_genmat,eigvec,list_start_end_ind)
        if ii == 0:
            Umat = Umat_1theta
        else:
            Umat = Umat_1theta@Umat

   #if debug:
   #   #print(f'Umat obtained in make_Umat_decomp_genmat')
   #   #print_matrix(Umat.toarray())
   #    det_Umat = sparse_det(Umat)
   #    if not np.isclose(det_Umat,1.0):
   #        print('Determinant of U is not 1: {det_Umat}')
   #        sys.exit()
        

    return Umat

def make_Umat_decomp_genmat_joblib(list_genmat,list_ampld,nparal,debug=False):
    """
    A joblib parallel version of make_Umat_decomp_genmat
    """

    n_theta = len(list_ampld)
    def make_Umat_for_one_decomp_genmat(igenmat,list_genmat,list_ampld):
        [list_unique_eigval,list_prj_genmat,eigvec,list_start_end_ind] = list_genmat[igenmat]
        theta = list_ampld[igenmat]
        Umat_1theta = make_analytical_U_decomp_genmat(theta,list_unique_eigval,list_prj_genmat,eigvec,list_start_end_ind)
        return Umat_1theta

    list_Umat_1theta = Parallel(n_jobs=nparal)(delayed(make_Umat_for_one_decomp_genmat)(ii,list_genmat,list_ampld) for ii in range(n_theta))
    Umat = list_Umat_1theta[0]
    for Umat_1theta in list_Umat_1theta[1:]:
        Umat = Umat_1theta@Umat

    return Umat

def sparse_det(A):
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")

    if A.shape[0] <= 500:
        return scipy.linalg.det(A.toarray())

    lu = scipy.sparse.linalg.splu(A)
    det_A = np.prod(lu.diags()) * (-1)**lu.perm_r.size

    return det_A

def make_analytical_U_decomp_genmat(theta,list_unique_eigval,list_prj_genmat,eigvec,list_start_end_ind):
    """
    Make exp(theta*genmat) using decomposed genmat
    """

    ndim = eigvec.shape[0]
    exp_genmat_anl = csr_matrix((ndim,ndim))
    for ival, unival in enumerate(list_unique_eigval):
        [istart, iend] = list_start_end_ind[ival]
        sub_eigvec = eigvec[:,istart:iend+1]
        sub_eigvec = csr_matrix(sub_eigvec)
        prj_genmat = list_prj_genmat[ival]
        if np.isclose(unival,0.0):
            assert np.isclose(scipy.sparse.linalg.norm(prj_genmat),0.0)
            exp_genmat_anl += sub_eigvec @ sub_eigvec.transpose()
        else:
            ndim_sub = prj_genmat.shape[0]
            sub_expgenmat = scipy.sparse.identity(ndim_sub,format="csr")
            sub_expgenmat += prj_genmat*np.sin(np.sqrt(-unival)*theta)
            sub_expgenmat -= prj_genmat@prj_genmat*(np.cos(np.sqrt(-unival)*theta)-1.0)
            sub_expgenmat_numrc = scipy.linalg.expm(np.sqrt(-unival)*theta*prj_genmat.toarray())
            exp_genmat_anl += sub_eigvec@sub_expgenmat@sub_eigvec.transpose()

    return exp_genmat_anl

def make_and_apply_U_matrix(list_genmat,list_ampld,l_feedin_inivec=False,vec_feed_in=None,debug=False):
    """
    Given a list of generator matrices and the corresponding amplitudes,
    make the exponential matrix, U = prod_I exp (theta_I gen_I)
    """
    if debug: print('\nIn make_U_matrix')

    if debug:
        print(f'U amplitudes:')
        print(list_ampld)

    ndim = list_genmat[0].shape[0]
    expmat = np.eye(ndim)
    for ii, ampld in enumerate(list_ampld):
        genmat = list_genmat[ii].toarray()
        norm_genmat = scipy.sparse.linalg.norm(list_genmat[ii])
        if np.isclose(norm_genmat,0.0): continue
        if debug:
           #det_genmat = scipy.linalg.det(genmat)
           #if not np.isclose(det_genmat,0.0):
           #    print(f'genmat does not have a determiannt 1, {det_genmat}')
           #    print_matrix(genmat)
           #    print('Bombing out')
           #    sys.exit()
            norm_to_be_0 = np.linalg.norm(genmat + genmat.transpose())
            if not np.isclose(norm_to_be_0,0.0):
                print(f'genmat {ii} is not antisymmetric:')
                print_matrix(genmat)
                print('genmat + genmat.transpose')
                print_matrix(genmat + genmat.transpose())
                print('Bombing out!')
                sys.exit()
       #tic = time.perf_counter()
        expmat_one_theta = scipy.linalg.expm(ampld*genmat)
       #toc = time.perf_counter()
       #print(f'Time for numerical expmat calculation: {toc - tic}')
       #See whether we can use analytical formula for exp(G)
       #genmat_sparse = list_genmat[ii]
       #tic = time.perf_counter()
       #l_mat_power_equals_scaled_mat, scal_fac = mat_powers_equals_scaled_mat_2nd(genmat_sparse,3)
       #if l_mat_power_equals_scaled_mat:
       #    if scal_fac > 0.0:
       #        print('G^3 = s*G but with s positive: {scal_fac}. Bombing out!')
       #        sys.exit()
       #    else:
       #        genmat_normalized = np.sqrt(-1.0/scal_fac)*genmat_sparse
       #        if not mat_powers_equals_scaled_mat(genmat_normalized,3,-1.0):
       #            print(f'genmat_normalized does not pass mat_powers_equals_scaled_mat')
       #            sys.exit()
       #        scaled_theta = ampld*np.sqrt(-scal_fac)
       #        expmat_one_theta_analytic = np.sin(scaled_theta)*genmat_normalized -\
       #                                    (np.cos(scaled_theta) - 1.0)*genmat_normalized@genmat_normalized
       #        expmat_one_theta_analytic = scipy.sparse.identity(ndim) + expmat_one_theta_analytic
       #       #if not np.isclose(scipy.sparse.linalg.norm(expmat_one_theta_analytic - csr_matrix(expmat_one_theta)),0.0):
       #       #    print('Analytical expm test failed')
       #       #    print(scipy.sparse.linalg.norm(expmat_one_theta_analytic - csr_matrix(expmat_one_theta)))
       #       #    print(expmat_one_theta_analytic - csr_matrix(expmat_one_theta))
       #       #    sys.exit()
       #toc = time.perf_counter()
       #print(f'Time for analytical expmat calculation: {toc - tic}')
        det_expmat_one_theta = scipy.linalg.det(expmat_one_theta)
       #print(f'det_expmat_one_theta: {det_expmat_one_theta}, iampld: {ii}')
        if not np.isclose(det_expmat_one_theta,1.0):
            print(f'Exponentiation of one theta*genmat is not 1')
            print(f'theta = {ampld}')
            print(f'genmat dimension: {ndim}')
            for i in range(ndim):
                for j in range(i,ndim):
                    if not np.isclose(genmat[i,j],-genmat[j,i]):
                        print('The genmat is not an antisymmetric matrix')
                        print(f'{i,j,genmat[i,j],genmat[j,i]}')
            sys.exit()
        expmat = expmat_one_theta@expmat
        det_expmat = scipy.linalg.det(expmat)
       #print(f'Accummulated det_expmat: {det_expmat}, iampld: {ii}')
       #expmat = scipy.linalg.expm(ampld*genmat)@expmat
       #if debug:
       #    print(f'genmat {ii}')
       #    print_matrix(genmat)
       #    print('\nUpdated U')
       #    print_matrix(expmat)
    
   #det_expmat = scipy.linalg.det(expmat)
   #if not np.isclose(det_expmat,1.0):
   #    print(f'expmat does not have a determinant 1, {det_expmat}')
   #    print(f'Shape of expmt: {expmat.shape}')
   #   #print_matrix(expmat)
   #    print('Bombing out')
   #    sys.exit()
   #   #print('Continue without bombing out. The resultant vector will be normalized')
   #The determinant test above is turned off because determinant may not be accurately
   #claculated for large dimension square matrix
    zero_matrix = expmat@expmat.transpose() - np.eye(ndim)
    resid_norm = np.linalg.norm(zero_matrix)
    if not np.isclose(resid_norm,0.0):
        print(f'expmat is not orthogonal. Residule norm: {resid_norm}. Bombing out!')
        sys.exit()

    if not l_feedin_inivec:
        vec_x_by_U = np.zeros([ndim])
        vec_x_by_U[0] = 1.0
    else:
        vec_x_by_U = vec_feed_in

    U_x_vec = expmat@vec_x_by_U
    if debug:
        print(f'U mat in make_and_apply_u_matrix')
        print_matrix(expmat)

   #Normalize U_x_vec in case expmat's determinant is far away from 1
    U_x_vec = U_x_vec / np.linalg.norm(U_x_vec)

    if debug:
        print(f'U_x_vec: {U_x_vec}')

    return expmat,U_x_vec
        
def mat_powers_equals_scaled_mat(mat_sparse,npow,scal_fac):       
    """
    See whether the n-th power a sparse matrix is equal to scal_fac*matrix
    """

    ndim = mat_sparse.shape[0]
    mat_npow = scipy.sparse.identity(ndim)
    for ii in range(npow):
        mat_npow = mat_npow@mat_sparse
    
    return np.isclose(scipy.sparse.linalg.norm(mat_npow - scal_fac*mat_sparse),0.0)

def mat_powers_equals_scaled_mat_2nd(mat_sparse,npow):
    """
    See whether n-th power of a sparse square matrix is equal to scal_fac*matrix and
    returns the scaling factor
    """

   #print(f'nonzero at: {mat_sparse.nonzero()}')
    nonzero_places = mat_sparse.nonzero()
    if len(nonzero_places[0]) == 0:
        print(f'All zero matrix should have been screened off before entering mat_powers_equals_scaled_mat_2nd')
        print('Bombing out!')
        sys.exit()
    nonzero_rows = nonzero_places[0]
    nonzero_cols = nonzero_places[1]
    row_set = list(set(nonzero_rows))
    col_set = list(set(nonzero_cols))
   #print(row_set,col_set)
    assert len(row_set) == len(col_set)
    ndim = len(row_set)
   #print(ndim)
    mat_reduced = csr_matrix((ndim,ndim))
   #print(mat_reduced.shape)
    for ii in range(len(nonzero_rows)):
        irow = row_set.index(nonzero_rows[ii])
        icol = col_set.index(nonzero_cols[ii])
        mat_reduced[irow,icol] = mat_sparse[nonzero_rows[ii],nonzero_cols[ii]]
   #print(mat_reduced)
   #print(mat_sparse)
    mat_npow = scipy.sparse.identity(ndim)
    for ii in range(npow):
        mat_npow = mat_npow@mat_reduced

    if np.isclose(scipy.linalg.det(mat_reduced.toarray()),0.0):
       #print(f'Singular mat_sparse detected. Bombing out!')
       #print(mat_reduced)
       #sys.exit()
        return False, None

    mat_inv = scipy.sparse.linalg.inv(mat_reduced)
    test_mat = mat_inv@mat_npow
    scal_fac = test_mat[0,0]
    if np.isclose(scipy.sparse.linalg.norm(test_mat - scal_fac*scipy.sparse.identity(ndim)),0.0):
        return True, scal_fac
    else:
        return False, None

def make_iapair_genmat(list_iapair,list_basis):
    """
    Create the generator matrices for Tiiaa_00 operators
    for a multi-electronic basis set
    """

    list_genmat = []
    n_basis = len(list_basis)
   #print(f'n_basis = {n_basis}')
    for iampld in range(len(list_iapair)):
       #print(iampld,list_iapair[iampld])
        ex_op = FermionOperator()
        for ia_pair in list_iapair[iampld]:
            i = ia_pair[0]
            a = ia_pair[1]
            Tiiaa_00 = get_Tiiaa_00(i,a)
            ex_op += Tiiaa_00

        generator_matrix = np.zeros([n_basis,n_basis])
        for ibasis,basis_bra in enumerate(list_basis):
            SDs_bra   = basis_bra[0]
            coefs_bra = basis_bra[2]
            for jbasis in range(ibasis+1,n_basis):
                basis_ket = list_basis[jbasis]
                SDs_ket   = basis_ket[0]
                coefs_ket = basis_ket[2]

                dsum = 0.0
                for iSD_bra in range(len(SDs_bra)):
                    onl = SDs_bra[iSD_bra]
                    coefl = coefs_bra[iSD_bra]
                    for jSD_ket in range(len(SDs_ket)):
                        onr = SDs_ket[jSD_ket]
                        coefr = coefs_ket[jSD_ket]
                        dsum += coefl*coefr*braket_tz(onl,onr,ex_op)

                generator_matrix[ibasis,jbasis] =  dsum
                generator_matrix[jbasis,ibasis] = -dsum

       #print(f'\ngenmat {iampld}:')
       #print_matrix(generator_matrix)
       #norm_genmat = np.linalg.norm(generator_matrix)
       #print(f'norm of genmat {iampld}: {norm_genmat}')

        list_genmat.append(generator_matrix)

    return list_genmat

def make_pair_ex_space_fast(list_ref_SD,list_ref_idx,list_ref_coef,mp2_ia_pair,list_orb_exclude = [],debug=False):
    """
    list_orb_exclude contains the explicitly specified orbitals that are not included in dmo-to-vmo excitations.
    """

    if debug: print('\nIn make_pair_ex_space_fast')

    n_spatialmo = len(list_ref_SD[0]) // 2
    on_spatialmo = list_ref_SD[0][0:2*n_spatialmo:2] + list_ref_SD[0][1:2*n_spatialmo:2]
    if debug: print(on_spatialmo)
    for on in list_ref_SD:
        if not np.allclose(on_spatialmo, on[0:2*n_spatialmo:2] + on[1:2*n_spatialmo:2]):
            print(f'Inconsistent occupancies of spatial orbitals in a state')
            print(on_spatialmo)
            print(on)
            print('bombing out')
            sys.exit()

    list_somo = list(np.where(on_spatialmo == 1.0)[0])
    if debug: print(f'SOMO whose occupancies remain in excitations: {list_somo}')


    pair_included = []
   #exclude excitations that involve SOMO
    for ia_pair in mp2_ia_pair:
        l_include = True
        nocc_pair = 0.0
        for orb in ia_pair:
            if orb in list_somo: l_include = False
            if orb in list_orb_exclude: l_include = False
            nocc_pair += nocc_spatial_orb_LCSD(list_ref_SD,list_ref_coef,orb)

        if not np.isclose(nocc_pair,2.0): l_include = False
        

        if l_include: pair_included.append(ia_pair)



    sorted_pair = sorted(pair_included, key=lambda x: x[0])[::-1]
    if debug:
        print(f'ia_pair included in make_pair_ex_space_fast: {sorted_pair}')

    
    list_dmo = []
    list_vmo = []
    for ia_pair in sorted_pair:
        dmo, vmo = ia_pair[0], ia_pair[1]
        if dmo not in list_dmo: list_dmo.append(dmo)
        if vmo not in list_vmo: list_vmo.append(vmo)

    max_nex = min(len(list_dmo),len(list_vmo))
    if debug:
        print(f'List of doubly occupied orbitals included: {list_dmo}')
        print(f'List of virtual         orbitals included: {list_vmo}')
        print(f'maximum pair excitation foldness: {max_nex}')

   #Group the ia pairs based on their occupied orbitals
    list_grouped_pairs_by_dmo = []
    for dmo in list_dmo:
        list_pair_1_dmo = []
        for pair in sorted_pair:
            if pair[0] == dmo:
                if debug: print(f'Adding {pair} to list')
                list_pair_1_dmo.append(pair)
        if debug: print(list_pair_1_dmo)
        list_grouped_pairs_by_dmo.append(list_pair_1_dmo)

    if debug: print(list_grouped_pairs_by_dmo)
    

    list_collected_ex = []
    for nex in range(1,max_nex+1):
        list_ex_one_ex_level = []
        comb_occ = list(combinations(list_dmo,nex))
        comb_vir = list(combinations(list_vmo,nex))
        if debug:
            print(f'Combinations of occupied spatial orbitals:')
            print(comb_occ)
            print(f'Combinations of virtual  spatial orbitals:')
            print(comb_vir)
        for occmo_list in comb_occ:
            if debug: print(f'occmo_list: {occmo_list}')
            list_dmo_group = []
            for dmo in [*occmo_list]:
                idmo_group = np.where(np.array(list_dmo) == dmo)[0][0]
                list_dmo_group.append(idmo_group)
            if debug: print(f'list_dmo_group: {list_dmo_group}')
            list_tmp = []
            for ii, dmo_group in enumerate(list_dmo_group):
                list_tmp.append(list_grouped_pairs_by_dmo[dmo_group])
             
            if debug: print(f'list_tmp: {list_tmp}')
            list_tmp = list(itertools.product(*list_tmp))
            if debug: print(f'list_tmp: {list_tmp}')
            l_remove_list = []
            for item in list_tmp:
               #If there this set consists of the same set of dmos and vmos as a previous set, remove this set
               #E.g., ([3, 6], [2, 5]) vs ([3, 5], [2, 6])
                list_dmo_tmp = []
                list_vmo_tmp = []
                for pair in item:
                    list_dmo_tmp.append(pair[0])
                    list_vmo_tmp.append(pair[1])
                if debug:
                    print(f'list_dmo_tmp: {list_dmo_tmp}')
                    print(f'list_vmo_tmp: {list_vmo_tmp}')
                l_remove = False
                idx_item = list_tmp.index(item)
                for jtem in list_tmp[:idx_item]:
                    list_dmo_jtem = []
                    list_vmo_jtem = []
                    for pair in jtem:
                        list_dmo_jtem.append(pair[0])
                        list_vmo_jtem.append(pair[1])
                    
                    lsame_dmo, _, _ =two_lists_with_same_contents(list_dmo_jtem,list_dmo_tmp)
                    lsame_vmo, _, _ =two_lists_with_same_contents(list_vmo_jtem,list_vmo_tmp)
                    if lsame_dmo and lsame_vmo: 
                        l_remove = True
                        break
                if l_remove:
                    if debug:
                        print('This combination will be removed for containing the same dmos and vmos as a previous set')
                    l_remove_list.append(l_remove)
                    continue
                for dmo in list_dmo_tmp:
                    if len(np.where(np.array(list_dmo_tmp) == dmo)[0]) > 1:
                        l_remove = True
                        break
                for vmo in list_vmo_tmp:
                    if len(np.where(np.array(list_vmo_tmp) == vmo)[0]) > 1: 
                        l_remove = True
                        break
                if l_remove:
                    if debug: print('This combination will be removed')
                l_remove_list.append(l_remove)
            if debug: print(f'l_remove_list: {l_remove_list}')
            for ii in range(len(l_remove_list)-1,-1,-1):
                if l_remove_list[ii]: del list_tmp[ii]
            if debug: print(f'list_tmp after removals: {list_tmp}')
            list_tmp = np.array(list_tmp).tolist()
            if debug: print(f'list_tmp after conversion to list of list: {list_tmp}')
            list_ex_one_ex_level.extend(list_tmp)
        list_collected_ex.append(list_ex_one_ex_level)

    ndim_pair_ex = 0
    for nex in range(1,max_nex+1):
        if debug: 
            print(f'Combinations of {nex} dmo to {nex} vmo mp2 excitations:')
            print(list_collected_ex[nex-1])
        ndim_pair_ex += len(list_collected_ex[nex-1])

    if debug: print(f'\nTotal # of multiple pair excitations: {ndim_pair_ex}')
    ndim_ex_space = ndim_pair_ex + 1 # +1 to include the reference state as the 0th element in list_ex_space

    list_ex_space = [[list_ref_SD,list_ref_idx,list_ref_coef]]
    for nex in range(1,max_nex+1):
        list_one_ex_level = list_collected_ex[nex-1]
        for pair_comb in list_one_ex_level:
            l_skip = False
            if debug: print(f'pair_comb: {pair_comb}')
            list_ex_SD = copy.deepcopy(list_ref_SD)
            list_ex_coef = copy.deepcopy(list_ref_coef)
           #print(f'list_ex_SD: {list_ex_SD}')
           #print(f'list_ref_SD: {list_ref_SD}')
            for pair in pair_comb:
                dmo, vmo = pair[0],pair[1]
                dmoa, dmob,vmoa, vmob = 2*dmo, 2*dmo+1, 2*vmo, 2*vmo+1
                for ii, onvec in enumerate(list_ex_SD):
                   #judge whether dmo is doubly occupied and vmo is empty, or
                   #whether dmo is empty and vmo is doubly occupied. All the other situations are not good! 
                    l_ndmo2_nvmo0,  l_ndmo0_nvmo2 = False, False
                    if onvec[dmoa] == 1.0 and onvec[dmob] == 1.0 and onvec[vmoa] == 0.0 and onvec[vmob] == 0.0:
                        l_ndmo2_nvmo0 = True
                    if onvec[dmoa] == 0.0 and onvec[dmob] == 0.0 and onvec[vmoa] == 1.0 and onvec[vmob] == 1.0:
                        l_ndmo0_nvmo2 = True
                    if l_ndmo2_nvmo0 and l_ndmo0_nvmo2:
                        print(f'l_ndmo2_nvmo0 and l_ndmo0_nvmo2 shall not be both True')
                        print(onvec,dmo,vmo)
                        print('Bombing out')
                        sys.exit()
                    elif not l_ndmo2_nvmo0 and not l_ndmo0_nvmo2:
                        print(f'\nAll ia pair combinations at this level: {list_one_ex_level}')
                        print(f'This pair combination {pair_comb}')
                        print(f'MO {dmo} is not doubly occupied or empty, or MO {vmo} is not doubly occupied or empty')
                        print(onvec)
                        print(f'list_ref_SD:')
                        print(list_ref_SD)
                        print(f'{dmo} and/or {vmo} shall be in list_orb_exclude:?')
                        print(list_orb_exclude)
                       #print('Bombing out!')
                       #sys.exit()
                        print('Skipping this pair combination')
                        l_skip = True
                    elif l_ndmo2_nvmo0:
                        onvec[dmoa],onvec[dmob],onvec[vmoa],onvec[vmob] = 0.0,0.0,1.0,1.0
                    elif l_ndmo0_nvmo2:
                        onvec[vmoa],onvec[vmob],onvec[dmoa],onvec[dmob] = 0.0,0.0,1.0,1.0
                        list_ex_coef *= -1.0

                    if l_skip: break
                if l_skip: break
            list_ex_idx = []
            for onvec in list_ex_SD:
                list_ex_idx.append(get_on_idx(onvec))
            list_ex_space.append([list_ex_SD,list_ex_idx,list_ex_coef])
           #print(f'list_ex_SD: {list_ex_SD}')

   #Check whether the same space is generated by explicit excitations
    if debug:
        icount = 0
        for nex in range(1,max_nex+1):
            list_one_ex_level = list_collected_ex[nex-1]
            for pair_comb in list_one_ex_level:
               #icount += 1
                ex_op = FermionOperator('')
                for pair in pair_comb:
                    dmo, vmo = pair[0],pair[1]
                    E_pair_ia = get_pair_excite(dmo,vmo)
                    D_pair_ia = get_pair_excite(vmo,dmo)
                   #ex_op *= E_pair_ia
                    ex_op *= (E_pair_ia - D_pair_ia)

                on_list_post_ex, idx_list_post_ex,coef_list_post_ex = \
                    op_action_tz_remove_0coef(ex_op,list_ref_SD,list_ref_idx,list_ref_coef)
                if len(on_list_post_ex) == 0: continue
                icount += 1
                list_SD_tmp = list_ex_space[icount][0]
                list_idx_tmp = list_ex_space[icount][1]
                list_coef_tmp = list_ex_space[icount][2]
                lbomb = False
                if len(list_SD_tmp) != len(on_list_post_ex): lbomb = True
                for ii, onvec in enumerate(list_SD_tmp):
                    if not np.isclose(np.linalg.norm(onvec - on_list_post_ex[ii]),0.0): lbomb = True
                    if list_idx_tmp[ii] != idx_list_post_ex[ii]: lbomb = True
                if not np.isclose(np.linalg.norm(list_coef_tmp - coef_list_post_ex),0.0): lbomb = True
                if lbomb:
                    print('\nInconsistent list of SDs from manual excitations and ex_op')
                    print(f'pair combination: {pair_comb}')
                    print(list_SD_tmp)
                    print(on_list_post_ex)
                    print('Bombing out')
                    sys.exit()

    return list_ex_space, list_collected_ex
            

def make_pair_ex_space(list_ref_SD,list_ref_idx,list_ref_coef,hdmo,lvmo,nmo,debug=False):
    """
    Read in a reference state and return list of states generated by all possible
    combinations of pair excitations from doubly occupied orbitals to virtual
    orbitals. The read-in state consists of a list of SDs and a list of coefficients.
    The read-in list of integer indices are to be used for the action function only.
    They are meaningless at this moment.
    hdmo means the highest doubly occupied orbitals and lvmo means
    the lowest virtual occupied orbitals
    """

    if debug: print('in make_pair_ex_space')

    list_dmo = []
    list_vmo = []
    for i in range(0,hdmo+1):
        list_dmo.append(i)
    for a in range(lvmo,nmo):
        list_vmo.append(a)

    
    list_ex_space = [[list_ref_SD,list_ref_idx,list_ref_coef]]
    
    for nex in range(1,min(len(list_dmo),len(list_vmo))+1):
        if debug: print(f'pair excitations involving {nex} occupied spatial orbitals')
        comb_occ = list(combinations(list_dmo,nex))
        comb_vir = list(combinations(list_vmo,nex))
    
        if debug:
            print(f'Combinations of occupied spatial orbitals:')
            print(comb_occ)
            print(f'Combinations of virtual  spatial orbitals:')
            print(comb_vir)
    
        for occmo_list in comb_occ:
            for virmo_list in comb_vir:
               #print('occmo_list',[*occmo_list])
               #print('virmo_list',[*virmo_list])
    
                ex_op = FermionOperator('')
                counter = -1
                for ja,i in enumerate([*occmo_list]):
                    a = [*virmo_list][ja]
                    counter += 1
                    E_pair_ia = get_pair_excite(i,a)
                    ex_op *= E_pair_ia
                   #print(E_pair_ia)
                   #if counter == 0:
                   #    ex_op  = E_pair_ia
                   #   #print(f'ex_op 1st, {ex_op}')
                   #else:
                   #    ex_op *= E_pair_ia
                       #print(f'ex_op, {ex_op}')
    
                   #ex_op = normal_ordered(ex_op)
    
               #print(f'ex_op: {ex_op}')
                on_list_post_ex, idx_list_post_ex,coef_list_post_ex = op_action_tz_remove_0coef(ex_op,list_ref_SD,list_ref_idx,list_ref_coef)
                if len(on_list_post_ex) == 0:
                    if debug:
                        print('The excitaiton operator creates a vacuum state')
                        print('occmo_list',[*occmo_list])
                        print('virmo_list',[*virmo_list])
                else:
                    list_ex_space.append([on_list_post_ex,idx_list_post_ex,coef_list_post_ex])

    if debug: 
        print(f'List of multiple pair excitation space:')
        for i,item in enumerate(list_ex_space):
            print(item)

    return list_ex_space

def prepare_mp2_amplitudes(homo,orbene,tbt,debug=False):
    """
    Prepare mp2 amplitudes, correlation energies, etc. The read in tbt is for spin orbitals
    and tbt[p,q,r,s]p^+ q^+ r s gives the 2-el term in Hamiltonian. I.e., spin orbitals in
    1st and 4th indices are for one electron, and those in 2nd and 3rd are for the other electron.
    """

    ia_pair_list = []
    mp2_ampld_list = []
    mp2_Ecorr_list = []
    lumo = homo+1
    nmo = len(orbene)

    for i in range(homo+1):
        for a in range(lumo,nmo):
            denominator = orbene[i] - orbene[a]
            numerator = tbt[2*a,2*a,2*i,2*i]
            Ecorr_1pair = (2.0*numerator)**2/(2.0*denominator) #The 2.0 multiplication is to compensate the 1/2 scaling for the 2-el integral.
            if debug: 
                print(i,a,orbene[i],orbene[a],denominator,numerator,numerator/denominator,Ecorr_1pair)
            ia_pair_list.append([i,a])
            mp2_ampld_list.append(numerator/denominator)
            mp2_Ecorr_list.append(Ecorr_1pair)

    return mp2_ampld_list,mp2_Ecorr_list,ia_pair_list

def prepare_mp2_amplitudes_actmo(hole_upbound,part_lowbound,orbene,tbt,debug=False):
    """
    Prepare mp2 amplitudes, correlation energies, etc. The read in tbt is for spin orbitals
    and tbt[p,q,r,s]p^+ q^+ r s gives the 2-el term in Hamiltonian. I.e., spin orbitals in
    1st and 4th indices are for one electron, and those in 2nd and 3rd are for the other electron.
    """

    ia_pair_list = []
    mp2_ampld_list = []
    mp2_Ecorr_list = []
    nmo = len(orbene)

    for i in range(hole_upbound+1):
        for a in range(part_lowbound,nmo):
            if i >= a: continue
            denominator = orbene[i] - orbene[a]
            numerator = tbt[2*a,2*a,2*i,2*i]
            if debug:
                print(i,a,orbene[i],orbene[a],denominator,numerator,numerator/denominator,Ecorr_1pair)
            if np.isclose(denominator,0.0):
               #mp2_ampld_list.append(0.25*np.pi)
               #mp2_Ecorr_list.append(-1.0) # -1.0 is a trivial value added to avoide singularity
                continue
            ampld = numerator/denominator
            ia_pair_list.append([i,a])
            Ecorr_1pair = (2.0*numerator)**2/(2.0*denominator) #The 2.0 multiplication is to compensate the 1/2 scaling for the 2-el integral.
           #if abs(ampld) > 0.25*np.pi:
           #    ampld = np.sign(ampld)*0.25*np.pi
           #    Ecorr_1pair = -1.0
            mp2_ampld_list.append(ampld)
            mp2_Ecorr_list.append(Ecorr_1pair)

    return mp2_ampld_list,mp2_Ecorr_list,ia_pair_list

def sort_mp2_amplitudes(mp2_ampld_list,mp2_Ecorr_list,ia_pair_list,debug=False):
    """
    Sort the mp2 amplitudes and correspondingly reorder the list of mp2 correlation 
    energies and i a orbital pairs
    """

   #zipped_list = zip(mp2_ampld_list,ia_pair_list,mp2_Ecorr_list)
   #if debug:  print(list(zipped_list))
    sorted_ampld_list, sorted_ia_pair_list,sorted_Ecorr_list  = zip(*sorted(zip(mp2_ampld_list,ia_pair_list,mp2_Ecorr_list)))
   #sorted_ampld_list, sorted_ia_pair_list,sorted_Ecorr_list = zip(*zipped_list)
    sorted_ampld_list = list(sorted_ampld_list)
    sorted_ia_pair_list = list(sorted_ia_pair_list)
    sorted_Ecorr_list = list(sorted_Ecorr_list)
    if debug:
        print('Sorted mp2 amplitudes')
        print(sorted_ampld_list)
        print('Correspondingly reordered orbital pairs')
        print(sorted_ia_pair_list)
        print('Correspondingly reordered mp2 correlation energies')
        print(sorted_Ecorr_list)

    return sorted_ampld_list, sorted_ia_pair_list, sorted_Ecorr_list

def remove_mp2_amplitudes_old(sorted_ampld_list,sorted_ia_pair_list,sorted_Ecorr_list,small_thrsh,list_act_orb=[]):
    """
    Remove the mp2 excitations whose amplitudes are smaller than the small_thrsh. The read-in
    list of amplitudes must have been sorted
    """

    n_ampld = len(sorted_ampld_list)
    list_removed_pair = []
    for i in range(n_ampld-1,-1,-1):
        [i_orb, a_orb] = sorted_ia_pair_list[i]
        if i_orb in list_act_orb and a_orb in list_act_orb: continue
        if abs(sorted_ampld_list[i]) < small_thrsh:
            list_removed_pair.append(sorted_ia_pair_list[i])
            sorted_ampld_list.pop()
            sorted_ia_pair_list.pop()
            sorted_Ecorr_list.pop()

    print(f'\nRemoved MP2 ia pairs:')
    print(list_removed_pair)

def remove_mp2_amplitudes(sorted_ampld_list,sorted_ia_pair_list,sorted_Ecorr_list,small_thrsh,list_act_orb=[],l_remove_actorb_pair=False):
    """             
    Remove the mp2 excitations whose amplitudes are smaller than the small_thrsh. When both i and a
    orbitals are in list_act_orb, the excitation is not removed, no matter how small the amplitude is.
    But if l_remove_actorb_pair = True, then the active orbita pairs will be removed.
    """         
                    
    n_ampld = len(sorted_ampld_list)
    list_l_remove = [False] * n_ampld
    list_removed_pair = []
    for i in range(n_ampld-1,-1,-1):
        [i_orb, a_orb] = sorted_ia_pair_list[i]
       #if i_orb in list_act_orb and a_orb in list_act_orb and not l_remove_actorb_pair: continue
        if abs(sorted_ampld_list[i]) < small_thrsh or (i_orb in list_act_orb and a_orb in list_act_orb and l_remove_actorb_pair):
            list_l_remove[i] = True
            list_removed_pair.append(sorted_ia_pair_list[i])
           #sorted_ampld_list.pop()
           #sorted_ia_pair_list.pop()
           #sorted_Ecorr_list.pop()

    print(f'\nRemoved MP2 ia pairs:')
    print(list_removed_pair)

    for i in range(n_ampld-1,-1,-1):
        if list_l_remove[i]:
            del sorted_ampld_list[i]
            del sorted_ia_pair_list[i]
            del sorted_Ecorr_list[i]

def group_mp2_amplitudes(sorted_ampld_list,sorted_ia_pair_list,sorted_Ecorr_list):
    """
    Group the sorted mp2 amplitudes. If some of them are degenerate, they are grouped together
    as one item. The orbital pairs and correation energies are also grouped correspondingly.
    
    """

    group_mp2_ampld = []
    group_mp2_iapair = []
    group_mp2_Ecorr = []
    pair_counted = []

    for i in range(len(sorted_ia_pair_list)):
        if i in pair_counted: continue
        group_mp2_ampld.append(sorted_ampld_list[i])
        pair_list_of_1_ampld = []
        Ecorr_list_of_1_ampld = []
        pair_list_of_1_ampld.append(sorted_ia_pair_list[i])
        Ecorr_list_of_1_ampld.append(sorted_Ecorr_list[i])
        pair_counted.append(i)
        for j in range(i+1,len(sorted_ia_pair_list)):
            if np.isclose(sorted_ampld_list[j],sorted_ampld_list[i]):
                pair_list_of_1_ampld.append(sorted_ia_pair_list[j])
                Ecorr_list_of_1_ampld.append(sorted_Ecorr_list[j])
                pair_counted.append(j)
            else:
                break

        group_mp2_iapair.append(pair_list_of_1_ampld)
        group_mp2_Ecorr.append(Ecorr_list_of_1_ampld)

    return group_mp2_ampld,group_mp2_iapair,group_mp2_Ecorr

def make_H_matrix_in_pair_ex_space(Hop,list_pair_ex_space,debug=False):
    """
    Read in a real hermitian operator and a list of real states in a multiple pair excitaiton space 
    and construct the real symmetric matrix in that space
    """

    ndim = len(list_pair_ex_space)
    if debug: print(f'ndim: {ndim}')

    Hmat = np.zeros([ndim,ndim])
    for ibas in range(ndim):
        basis_bra = list_pair_ex_space[ibas]
       #print(basis_bra)
        SDs_bra = basis_bra[0]
        coefs_bra = basis_bra[2] #basis_bra[1] contains the list of integer indices, not useful here
        for jbas in range(ibas,ndim):
            basis_ket = list_pair_ex_space[jbas]
            SDs_ket   = basis_ket[0]
            coefs_ket = basis_ket[2]

            dsum = 0.0
            for iSD_bra in range(len(SDs_bra)):
                onl = SDs_bra[iSD_bra]
                coefl = coefs_bra[iSD_bra]
                for jSD_ket in range(len(SDs_ket)):
                    onr = SDs_ket[jSD_ket]
                    coefr = coefs_ket[jSD_ket]
                    dsum += coefl*coefr*braket_tz(onl,onr,Hop)
    
           #print(ibasis,jbasis,dsum)
            Hmat[ibas,jbas]=dsum
            Hmat[jbas,ibas]=dsum

    if debug:
        print(f'Constructed matrix')
        print_matrix(Hmat)
    
   #Also construct the sparse matrix
    Hmat_sparse = csr_matrix(Hmat)

    return Hmat, Hmat_sparse

def make_H_matrix_in_pair_ex_2spaces(Hop,list_pair_ex_space1,list_pair_ex_space2,debug=False):
    """
    Read in a operator and two lists of real states in two multiple pair excitation spaces
    and construct the real a-symmetric matrix between the two spaces. In space1 are bra states and in
    space2 are ket states.
    """

    ndim1 = len(list_pair_ex_space1)
    ndim2 = len(list_pair_ex_space2)
    if debug: print(f'bra dimension: {ndim1}, ket dimension: {ndim2}')

   #if debug:
   #    for term in Hop:
   #        print(term)

    Hmat = np.zeros([ndim1,ndim2])
    for ibas in range(ndim1):
        basis_bra = list_pair_ex_space1[ibas]
        SDs_bra = basis_bra[0]
        coefs_bra = basis_bra[2] #basis_bra[1] contains the list of integer indices, not useful here
        for jbas in range(ndim2):
           #print(f'ibas,jbas {ibas,jbas}')
            basis_ket = list_pair_ex_space2[jbas]
            SDs_ket   = basis_ket[0]
            coefs_ket = basis_ket[2]

            dsum = 0.0
            for iSD_bra in range(len(SDs_bra)):
                onl = SDs_bra[iSD_bra]
                coefl = coefs_bra[iSD_bra]
                for jSD_ket in range(len(SDs_ket)):
                    onr = SDs_ket[jSD_ket]
                    coefr = coefs_ket[jSD_ket]
                    if ibas == 2 and jbas == 1:
                       #print(coefl,onl,coefr,onr,braket_tz(onl,onr,Hop))
                        for term in Hop:
                            term_wise_element = braket_tz(onl,onr,term)
                            if abs(term_wise_element) > 1.e-5 and debug:
                                print(term,onl,onr,term_wise_element,coefl,coefr)
                    dsum += coefl*coefr*braket_tz(onl,onr,Hop)
                   #print(f'dsum = {dsum}')

            Hmat[ibas,jbas] = dsum

    if debug:
        print(f'Constructed matrix in make_H_matrix_in_pair_ex_2spaces')
        print_matrix(Hmat)

    return Hmat

def make_short_H_ferm_op(const,obt_phys,tbt_phys):
    """
    Read in a hermitian fermionic operator and change it from sum p q r s
    to sum p>q, r>s for the 2-body terms
    """

    # print(f'In clean_H_ferm_op')

    # print(f'const: {const}')

    N = obt_phys.shape[0]
    # print(f'# of spin orbitals: {N}')

    H1 = FermionOperator()
    H2 = FermionOperator()
    for p in range(N):
        for q in range(p,N):
            if not np.isclose(obt_phys[p,q],0.0):
                coef = obt_phys[p,q]
                term = ((p,1), (q,0))
                H1 += FermionOperator(term,coef)
                if p != q:
                    H1 += hermitian_conjugated(FermionOperator(term,coef))

    # print(f'H1:')
    # print(H1)

    for p in range(N):
        for q in range(p):
            for r in range(N):
                for s in range(r+1,N):
                    term = ((p,1), (q,1), (r,0), (s,0))
                    coef_coul = tbt_phys[p,q,r,s]
                    coef_exch = tbt_phys[p,q,s,r]
                    if np.isclose(coef_coul,0.0) and np.isclose(coef_exch,0.0):
                        continue
                    else:
                        H2 += FermionOperator(term,2.0*(coef_coul - coef_exch))
                    

    
    H_short = FermionOperator((),const)
    H_short += H1 + H2

    # print(f'H_short:')
    # print(H_short)

    # print(dir(H_short))
    # print(f' # of terms in H_short {len(H_short.terms)}')
    # print(H_short.actions,H_short.action_strings,H_short.action_before_index)

    return(H_short)

def diff_spin(sigma_in):
    """
    Return a if sigma = b, and return b if sigma = a
    """

    if sigma_in == 'a':
        sigma_out = 'b'
    elif sigma_in == 'b':
        sigma_out = 'a'
    else:
        print('Unrecognized read in spin functions: {sigma_in}, neither a nor b')
        sys.exit()

    return sigma_out

def make_op_for_diagonal_U_space(const,obt_phys,tbt_phys,list_CIS_pair=[],debug=False):
    """
    Construct operators that couple states in the diagonal U_CSF space
    """
    
    if debug: print(f'\n MO pairs in CIS list: {list_CIS_pair}')

    list_SOMO = []
    for mo_pair in list_CIS_pair:
        if len(mo_pair) != 2:
            print(f'{mo_pair} does not contain two spatial mos. Bombing out!')
        if mo_pair[1] < mo_pair[0]:
            print(f'Each pair of MOs should have the smaller index as element 0 and the larger index as element 1 ')
        for mo in mo_pair:
            if mo in list_SOMO:
                print(f'duplicated mo {mo} in CIS mo pair list. Bombing out!')
                sys.exit()
            list_SOMO.append(mo)

    if debug: print(f'\n MOs in CIS list: {list_SOMO}')

    n_spinmo = obt_phys.shape[0]
    n_spatialmo = n_spinmo // 2

    tbt_unscaled = tbt_phys*2.0

    one_el_term = FermionOperator()
    diff_spin_colum_term = FermionOperator()
    same_spin_colex_term = FermionOperator()
    for p in range(n_spatialmo):
        for p_spinor in ['a','b']:
            psig = str(p)+p_spinor
            ipsig = ia_to_2i_ib_to_2iplus1(psig)
            coef = obt_phys[ipsig,ipsig]
           #if not np.isclose(coef,0.0):
            term = ((ipsig,1),(ipsig,0))
            one_el_term += FermionOperator(term,coef)

            if p_spinor == 'a':
                q_range = p+1
            else:
                q_range = p

            for q in range(q_range):
                q_spinor = diff_spin(p_spinor)
                qsig_prm = str(q) + q_spinor
                iqsig_prm = ia_to_2i_ib_to_2iplus1(qsig_prm)
                coef = tbt_unscaled[ipsig,iqsig_prm,iqsig_prm,ipsig]
               #if not np.isclose(coef,0.0):
                term = ((ipsig,1),(iqsig_prm,1),(iqsig_prm,0),(ipsig,0))
                if p == q and p in list_SOMO: continue
                diff_spin_colum_term += FermionOperator(term,coef)

                q_spinor = p_spinor
                qsig = str(q) + q_spinor
                iqsig = ia_to_2i_ib_to_2iplus1(qsig)
                coef = tbt_unscaled[ipsig,iqsig,iqsig,ipsig] - tbt_unscaled[ipsig,iqsig,ipsig,iqsig]
                term = ((ipsig,1),(iqsig,1),(iqsig,0),(ipsig,0))
               #if not np.isclose(coef,0.0):
               #if p in list_exclude and q in list_exclude: continue
                if [q,p] in list_CIS_pair: continue
                same_spin_colex_term += FermionOperator(term,coef)
    if debug:       
        print(f'one_el_terms:')
        print(one_el_term)
        print(f'Coulomb terms between orbitals with different spins:')
        print(diff_spin_colum_term)
        print(f'Coulomb and Exchange terms between orbitals with same spin:')
        print(same_spin_colex_term)

    Type1_term = const+ one_el_term + diff_spin_colum_term + same_spin_colex_term
    if debug:
        print(f'Type1_term:')
        print(Type1_term)

    Type2_term = FermionOperator()
    for p in range(n_spatialmo):
        for q in range(n_spatialmo):
            if p != q:
                ipa = 2*p
                ipb = ipa+1
                iqa = 2*q
                iqb = iqa+1
                term = ((ipa,1),(ipb,1),(iqb,0),(iqa,0))
                coef = tbt_unscaled[ipa,ipb,iqb,iqa]
               #if not np.isclose(coef,0.0):
                if p in list_SOMO or q in list_SOMO: continue
                Type2_term += FermionOperator(term,coef)
    
    if debug:
        print(f'Type2_term:')
        print(Type2_term)

    Type3_term = FermionOperator()
   #for p in list_exclude:
   #    for q in list_exclude:
   #        if p > q:
   #            ipa = 2*p
   #            ipb = ipa+1
   #            iqa = 2*q
   #            iqb = iqa+1
   #            term = ((ipa,1),(iqb,1),(ipb,0),(iqa,0))
   #            coef = tbt_unscaled[ipa,iqb,ipb,iqa]
   #            one_term = FermionOperator(term,coef)
   #            Type3_term += one_term + hermitian_conjugated(one_term)
    for mo_pair in list_CIS_pair:
        i=mo_pair[0]
        a=mo_pair[1]
        iia = 2*i
        iib = iia+1
        iaa = 2*a
        iab = iaa+1
       #The two terms should be hermitian to each other
        term1 = ((iia,1),(iab,1),(iib,0),(iaa,0))
        coef1 = tbt_unscaled[iia,iab,iib,iaa]
        term2 = ((iib,1),(iaa,1),(iia,0),(iab,0))
        coef2 = tbt_unscaled[iib,iaa,iia,iab]
        Type3_term += FermionOperator(term1,coef1) + FermionOperator(term2,coef2)
      

    if debug:           
        print(f'Type3_term:')
        print(Type3_term)

    nterm = len(Type1_term.terms) + len(Type2_term.terms) + len(Type3_term.terms)
    if debug: 
        print(f'# of terms in make_UCSF1_terms: {nterm} ')
        print(f'# of 3 types of terms: {len(Type1_term.terms), len(Type2_term.terms), len(Type3_term.terms)}')

    return Type1_term, Type2_term, Type3_term

def make_op_for_diagonal_U_space_Stt(Enuc,obt_phys,tbt_phys,list_CIS_pair,debug=False):
    """
    Make the operator terms that contribute nonzero to the diagonal matrix elements of
    <CSF_iajb^tt|U^+ H U|CSF_iajb^tt>
    """

    if debug: print('In make_op_for_diagonal_U_space_Stt')

    lbomb = False
    if len(list_CIS_pair) != 3: lbomb = True
    if list_CIS_pair[-1] != 'tt': lbomb = True

    list_SOMO = []
    for mo_pair in list_CIS_pair[:-1]:
        if len(mo_pair) != 2:
            print(f'{mo_pair} does not contain two spatial mos. Bombing out!')
        if mo_pair[1] < mo_pair[0]:
            print(f'Each pair of MOs should have the smaller index as element 0 and the larger index as element 1 ')
        for mo in mo_pair:
            if mo in list_SOMO:
                print(f'duplicated mo {mo} in CIS mo pair list. Bombing out!')
                sys.exit()
            list_SOMO.append(mo)

    if lbomb:
        print(f'Inappropriate list_CIS_pair: {list_CIS_pair}. Bombing out')
        sys.exit()

    if debug: print(f'Singly occupied orbitals: {list_SOMO}')

    n_spinmo = obt_phys.shape[0]
    n_spatialmo = n_spinmo // 2
        
    tbt_unscaled = tbt_phys*2.0

    one_el_term = FermionOperator()
    diff_spin_colum_term = FermionOperator()
    same_spin_colex_term = FermionOperator()
    Type2_term = FermionOperator()

    for p in range(n_spatialmo):
        pa = 2*p
        coef = obt_phys[pa,pa]
        term = ((pa,1),(pa,0))
        one_el_term += FermionOperator(term,coef)

    one_el_term += op_spin_flip(one_el_term)

    for p in range(n_spatialmo):
        pa, pb = 2*p, 2*p+1
        if p not in list_SOMO:
            term = ((pa,1),(pb,1),(pb,0),(pa,0))
           #tbt instead of tbt_unscaled is used here because later in op_spin_flip this term will be doubled
           #coef = tbt_phys[pa,pb,pb,pa]
            coef = tbt_unscaled[pa,pb,pb,pa]
            diff_spin_colum_term += FermionOperator(term,coef)
        for q in range(p+1,n_spatialmo):
            qa, qb = 2*q, 2*q+1
            term = ((pa,1),(qa,1),(qa,0),(pa,0))
            coef = tbt_unscaled[pa,qa,qa,pa] - tbt_unscaled[pa,qa,pa,qa]
            same_spin_colex_term += FermionOperator(term,coef)
            term  = ((pa,1),(qb,1),(qb,0),(pa,0))
            term2 = ((pb,1),(qa,1),(qa,0),(pb,0))
            coef = tbt_unscaled[pa,qb,qb,pa]
            diff_spin_colum_term += FermionOperator(term,coef) + FermionOperator(term2,coef)
            if p not in list_SOMO and q not in list_SOMO:
                term = ((pa,1),(pb,1),(qb,0),(qa,0))
                coef = tbt_unscaled[pa,pb,qb,qa]
                Type2_term += FermionOperator(term,coef)

    same_spin_colex_term += op_spin_flip(same_spin_colex_term)
   #diff_spin_colum_term has been explicitly spin-adapted (term2 above)
   #diff_spin_colum_term += op_spin_flip(diff_spin_colum_term)

   #Type1_term are for each individual SD
    Type1_term = Enuc + one_el_term + same_spin_colex_term + diff_spin_colum_term

    if debug:
        print('\nType1_term:')
        print(Type1_term)

   #Type2_term are for correlation between SDs with the same occupancies in the four SOMOs
    Type2_term += hermitian_conjugated(Type2_term)

    if debug:
        print('\nType2_term:')
        print(Type2_term)

   #[[i,a],[j,b]] = list_CIS_pair[:-1]

    Type3_term = FermionOperator()
    for ip in range(len(list_SOMO)):
        p = list_SOMO[ip]
        pa, pb = 2*p, 2*p+1
        for iq in range(ip+1,len(list_SOMO)):
            q = list_SOMO[iq]
            qa, qb = 2*q, 2*q+1
            term = ((pa,1),(qb,1),(pb,0),(qa,0))
            coef = tbt_unscaled[pa,qb,pb,qa]
            Type3_term += FermionOperator(term,coef)

   #Type3_term are for the correlation between SDs with different occupancies in the four SOMOs but identical
   #pair occupations. This is the only difference from iajb_Sss
    Type3_term += op_spin_flip(Type3_term) #spin flipping also gives hermitian conjugate here

    if debug:
        print('\nType3_term:')
        print(Type3_term)

    nterm = len(Type1_term.terms) + len(Type2_term.terms) + len(Type3_term.terms)
    if debug: 
        print(f'# of terms in make_op_for_diagonal_U_space_Stt: {nterm} ')
        print(f'# of 3 types of terms: {len(Type1_term.terms), len(Type2_term.terms), len(Type3_term.terms)}')
    
            
    return Type1_term, Type2_term, Type3_term
            

def make_op_for_offdiag_UHF_UCSFia(const,obt_phys,tbt_phys,homo,list_CIS_pair=[],debug=False):
    """
    Construct terms that contribute to the bras in UHF space and kets in UCSFia space
    """

    if debug: print('in make_op_for_offdiag_UHF_UCSFia')

    n_spinmo = obt_phys.shape[0]
    n_spatialmo = n_spinmo // 2

    tbt_unscaled = tbt_phys*2.0

    i = list_CIS_pair[0][0]
    a = list_CIS_pair[0][1]
    if i >= a:
        print(f'i >= a detected {i,a}. Bombing out!')
        sys.exit()

    list_rs = [i,a]
    list_SOMO = list_rs
    list_same_spin_rs = []
    list_diff_spin_rs = [i]
    list_same_spin_sr = []
    list_diff_spin_sr = [a]
    list_2elcorr_SOMO = []
    H_offdiag_U_1modiff = make_op_for_offdiag_U_1modiff(obt_phys,tbt_phys,list_rs,list_SOMO,list_same_spin_rs,list_diff_spin_rs,list_same_spin_sr,list_diff_spin_sr,list_2elcorr_SOMO,debug)

    return H_offdiag_U_1modiff

def make_op_for_offdiag_UHF_UCSFiajb(tbt_phys,list_CIS_pair,debug=False):
    """ 
    make operators that contribute to off-diagonal matrix elements between bras in U|CSF1> space
    and kets in U|CSF_iajb> space.
    list_CIS_pair = [[i,a],[j,b]]
    """

    if debug: print('\nIn make_op_for_offdiag_UHF_UCSFiajb\n')

    lbomb = False
    if len(list_CIS_pair) != 2: lbomb = True

    i, j, a, b = list_CIS_pair[0][0], list_CIS_pair[1][0], list_CIS_pair[0][1], list_CIS_pair[1][1]

    if i == j or i == a or j == b or a == b: lbomb = True
    if lbomb:
        print(f'\nInappropriate list_CIS_pair in make_op_for_offdiag_UHF_UCSFiajb: {list_CIS_pair}')
        print('Bombing out!')
        sys.exit()

    list_pqrs = [i,j,a,b]
    if debug: print(f'list_pqrs in make_op_for_offdiag_UHF_UCSFiajb_new:{list_pqrs}')

    H_offdiag_U_2modiff = make_op_for_offdiag_U_2modiff(tbt_phys,list_pqrs,debug)

    return H_offdiag_U_2modiff

def op_spin_flip(op,debug=False):
    """
    Read in an operator and flip all spin operators
    """

    if debug: print('in op_spin_flip')

    op_spinflip = FermionOperator()
    for component in op:
        if debug:
            print(f'component: {component}')
            print(f'component.terms: {component.terms}')

        for key,value in component.terms.items(): #There is only one item in component operator
            if debug: print(f'key,value:{key,value}')
            term_spinflip = ()
            for aadag  in key:
                ispin = aadag[0]
                iaction = aadag[1]
                if ispin % 2:
                    ispin_flip = ispin - 1
                else:
                    ispin_flip = ispin + 1
                aadag_spinflip = (ispin_flip,iaction)
               #print(aadag_spinflip)
               #term_spinflip.append(aadag_spinflip)
                term_spinflip += ((aadag_spinflip),)
               #print(term_spinflip)

            if debug: print(f'term_spinflip: {term_spinflip}')

        component_spinflip = FermionOperator(term_spinflip,value)
        if debug: print(f'Spin flipped component op: {component_spinflip}')
        op_spinflip += component_spinflip

    if debug:
        print(f'spin flipped operator: op_spin_flip')
        print(op_spinflip)

    return op_spinflip

def make_op_for_offdiag_UCSFia_UCSFib(obt_phys,tbt_phys,list_CIS_pair,debug=False):
    """
    make operators that contribute to off-diagonal matrix elements between bras in U|CSFia> space
    and kets in U|CSF_ib> space
    """

    if debug: print('\nIn make_op_for_offdiag_UCSFia_UCSFib')

    if len(list_CIS_pair) != 2 or list_CIS_pair[0][0] != list_CIS_pair[1][0] or list_CIS_pair[0][1] == list_CIS_pair[1][1]:
        print(f'Read in i,a and i,b pairs are not appropriate, {list_CIS_pair}, Bombing out!')
        sys.exit()

    i = list_CIS_pair[0][0]
    a = list_CIS_pair[0][1]
    b = list_CIS_pair[1][1]

    list_SOMO = [i,a,b]
    list_rs = [a,b]
    list_same_spin_rs = []
    list_diff_spin_rs = [i]
    list_same_spin_sr = [i]
    list_diff_spin_sr = [a,b]
    list_2elcorr_SOMO = [i]
    H_offdiag_U_1modiff = make_op_for_offdiag_U_1modiff(obt_phys,tbt_phys,list_rs,list_SOMO,list_same_spin_rs,list_diff_spin_rs,list_same_spin_sr,list_diff_spin_sr,list_2elcorr_SOMO,debug)

    return H_offdiag_U_1modiff

def make_op_for_offdiag_UCSFia_UCSFja(obt_phys,tbt_phys,list_CIS_pair,debug=False):
    """
    make operators that contribute to off-diagonal matrix elements between bras in U|CSFia> space
    and kets in U|CSF_ja> space
    """

    if debug: print('\nIn make_op_for_offdiag_UCSFia_UCSFja')

    if len(list_CIS_pair) != 2 or list_CIS_pair[0][1] != list_CIS_pair[1][1] or list_CIS_pair[0][0] == list_CIS_pair[1][0]:
        print(f'Inappropriate list_CIS_pair: {list_CIS_pair}. Bombing out!')
        sys.exit()

    i = list_CIS_pair[0][0]
    j = list_CIS_pair[1][0]
    a = list_CIS_pair[0][1]

    list_SOMO = [i,j,a]
    list_rs = [j,i]
    list_same_spin_rs = [a]
    list_diff_spin_rs = [i,j]
    list_same_spin_sr = []
    list_diff_spin_sr = [a]
    list_2elcorr_SOMO = [a]

    H_offdiag_U_1modiff = make_op_for_offdiag_U_1modiff(obt_phys,tbt_phys,list_rs,list_SOMO,list_same_spin_rs,list_diff_spin_rs,list_same_spin_sr,list_diff_spin_sr,list_2elcorr_SOMO,debug)

    return H_offdiag_U_1modiff

def make_op_for_offdiag_UCSFia_UCSFiajb(obt_phys,tbt_phys,list_CIS_pair,debug=False):
    """
    make operators that contribute to off-diagonal matrix elements between bras in U|CSFia> space
    and kets in U|CSF_iajb> space
    list_CIS_pair = [[i,a],[j,b]]
    """

    if debug: print('\nmake_op_for_offdiag_UCSFia_UCSFiajb')
    lbomb = False
    if len(list_CIS_pair) != 2: lbomb = True

    list_SOMO = []
    for pair in list_CIS_pair:
        for mo in pair:
            if mo in list_SOMO:
                print('Duplicated mo indices in list_CIS_pair. Bombing out!')
                lbomb = True
            else:
                list_SOMO.append(mo)

    if lbomb: sys.exit()

    i = list_CIS_pair[0][0]
    a = list_CIS_pair[0][1]
    j = list_CIS_pair[1][0]
    b = list_CIS_pair[1][1]

#Redefine list_SOMO since i and a do provide both same spin and diff spin terms in hermitian effective 1-el op
   #list_SOMO = [j,b]
    list_SOMO = [i,j,a,b]
    list_rs = [j,b]
    list_same_spin_rs = [i,a]
    list_diff_spin_rs = [i,j,a]
    list_same_spin_sr = [i,a]
    list_diff_spin_sr = [i,a,b]
    list_2elcorr_SOMO = []

    H_offdiag_U_1modiff = make_op_for_offdiag_U_1modiff(obt_phys,tbt_phys,list_rs,list_SOMO,list_same_spin_rs,list_diff_spin_rs,list_same_spin_sr,list_diff_spin_sr,list_2elcorr_SOMO,debug)


    return H_offdiag_U_1modiff

def make_op_for_offdiag_UCSFiajb_UCSFicjb(obt_phys,tbt_phys,list_CIS_pair,debug=False):
    """
    make operators that contribute to off-diagonal matrix elements between bras in U|CSFiajb> space
    and kets in U|CSF_icjb> space
    list_CIS_pair = [[i,a],[j,b],[i,c],[j,b]]
    """

    if debug: print('\nIn make_op_for_offdiag_UCSFiajb_UCSFicjb')

    lbomb = False
    if len(list_CIS_pair) != 4: lbomb = True
    if list_CIS_pair[3] != list_CIS_pair[1]: lbomb = True
    if list_CIS_pair[0][0] != list_CIS_pair[2][0] or list_CIS_pair[0][1] == list_CIS_pair[2][1]: lbomb = True
    if lbomb:
        print(f'\nInappropriate list_CIS_pair in make_op_for_offdiag_UCSFiajb_UCSFicjb: {list_CIS_pair}. Bombing out!')
        sys.exit()

    i, j    = list_CIS_pair[0][0], list_CIS_pair[1][0]
    a, b, c = list_CIS_pair[0][1], list_CIS_pair[1][1], list_CIS_pair[2][1]

    list_SOMO = [i,j,a,b,c]
    list_rs = [a,c]
    list_same_spin_rs = [j,b]
    list_diff_spin_rs = [i,j,b]
    list_same_spin_sr = [i,j,b]
    list_diff_spin_sr = [j,a,b,c]
    list_2elcorr_SOMO = [i]

    H_offdiag_U_1modiff = make_op_for_offdiag_U_1modiff(obt_phys,tbt_phys,list_rs,list_SOMO,list_same_spin_rs,list_diff_spin_rs,list_same_spin_sr,list_diff_spin_sr,list_2elcorr_SOMO,debug)

    return H_offdiag_U_1modiff

def make_op_for_offdiag_UCSFiajb_UCSFiakb(obt_phys,tbt_phys,list_CIS_pair,debug=False):
    """
    make operators that contribute to off-diagonal matrix elements between bras in U|CSFiajb> space
    and kets in U|CSF_iakb> space
    list_CIS_pair = [[i,a],[j,b],[i,a],[k,b]]
    """

    if debug: print('\nIn make_op_for_offdiag_UCSFiajb_UCSFiakb')

    lbomb = False
    if len(list_CIS_pair) != 4: lbomb = True
    if list_CIS_pair[0] != list_CIS_pair[2]: lbomb = True
    if list_CIS_pair[1][1] != list_CIS_pair[3][1] or list_CIS_pair[1][0] == list_CIS_pair[3][0]: lbomb = True
    if lbomb:
        print(f'\nInappropriate list_CIS_pair in make_op_for_offdiag_UCSFiajb_UCSFiakb: {list_CIS_pair}. Bombing out!')
        sys.exit()

    if debug: print(f'list_CIS_pair: {list_CIS_pair}')

    i, j, k = list_CIS_pair[0][0], list_CIS_pair[1][0], list_CIS_pair[3][0]
    a, b    = list_CIS_pair[0][1], list_CIS_pair[1][1]

    list_SOMO = [i,j,k,a,b]
    list_rs = [k,j]
    list_same_spin_rs = [i,a,b]
    list_diff_spin_rs = [i,j,k,a]
    list_same_spin_sr = [i,a]
    list_diff_spin_sr = [i,a,b]
    list_2elcorr_SOMO = [b]

    H_offdiag_U_1modiff = make_op_for_offdiag_U_1modiff(obt_phys,tbt_phys,list_rs,list_SOMO,list_same_spin_rs,list_diff_spin_rs,list_same_spin_sr,list_diff_spin_sr,list_2elcorr_SOMO,debug)
        
    return H_offdiag_U_1modiff

def make_op_for_offdiag_U_1modiff(obt_phys,tbt_phys,list_rs,list_SOMO,list_same_spin_rs,list_diff_spin_rs,list_same_spin_sr,list_diff_spin_sr,list_2elcorr_SOMO,debug=False):
    """
    Read in the info to generate operator that have nonzero contributions to the matrix elements between
    the bra and ket spaces generated by pair excitations out of reference states differ by one spatial orbital, e.g.,
    <CSF_1|U^+ H U|CSF_ia>, <CSF_ia|U^+ H U|CSF_ib>, <CSF_ia|U^+ H U|CSF_ja>, etc.
    list_rs contains the two spatial orbital indices that reflect the difference between the bra and ket reference states.
    For <CSF_1|U^+ H U|CSF_ia>, list_rs = [i,a], as <CSF_1|a_isig^+ a_asig|CSF_ia> != 0.
    For <CSF_ia|U^+ H U|CSF_ib>, list_rs = [a,b], as <CSF_ia|a_asig^+ a_bsig|CSF_ib> != 0.
    """

    if debug: print('\nIn make_op_for_offdiag_U_1modiff')

    if len(list_rs) != 2 or list_rs[0] == list_rs[1]:
        print(f'Inappropriate list_rs {list_rs}, which should contain two different spatial mo indices.')
        print('Bombing out!')
        sys.exit()

    n_spinmo = obt_phys.shape[0]
    n_spatialmo = n_spinmo // 2

    tbt_unscaled = tbt_phys*2.0
    r = list_rs[0]
    s = list_rs[1]

    ra = 2*r
    rb = ra+1
    sa = 2*s
    sb = sa+1

   #The hermitian part
    term = ((ra,1),(sa,0))
    coef = obt_phys[ra,sa]
    Term_1el = FermionOperator(term,coef)

    Term_1el += op_spin_flip(Term_1el)
    Term_1el += hermitian_conjugated(Term_1el)

    Term_eff1e_diff_spin = FermionOperator()
    Term_eff1e_same_spin = FermionOperator()
    Term_rpsp_2el_corr = FermionOperator()
    for p in range(n_spatialmo):
        if p in list_SOMO: continue
        pa = 2*p
        pb = pa+1

        term = ((ra,1),(pb,1),(pb,0),(sa,0))
        coef = tbt_unscaled[ra,pb,pb,sa]
        Term_eff1e_diff_spin += FermionOperator(term,coef)
        term = ((ra,1),(pa,1),(pa,0),(sa,0))
        coef -= tbt_unscaled[ra,pa,sa,pa]
        Term_eff1e_same_spin += FermionOperator(term,coef)
        term = ((ra,1),(sb,1),(pb,0),(pa,0))
        coef = tbt_unscaled[ra,sb,pb,pa]
        Term_rpsp_2el_corr += FermionOperator(term,coef)

    Term_2elcorr_SOMO = FermionOperator()
    for p in list_2elcorr_SOMO:
        pa = 2*p
        pb = pa+1

        term = ((ra,1),(pb,1),(sb,0),(pa,0))
        coef = tbt_unscaled[ra,pb,sb,pa]
        Term_2elcorr_SOMO += FermionOperator(term,coef)


    Term_eff1e_diff_spin += op_spin_flip(Term_eff1e_diff_spin)
    Term_eff1e_diff_spin += hermitian_conjugated(Term_eff1e_diff_spin)
    Term_eff1e_same_spin += op_spin_flip(Term_eff1e_same_spin)
    Term_eff1e_same_spin += hermitian_conjugated(Term_eff1e_same_spin)
    Term_rpsp_2el_corr += op_spin_flip(Term_rpsp_2el_corr)
    Term_rpsp_2el_corr += hermitian_conjugated(Term_rpsp_2el_corr)
    Term_2elcorr_SOMO += op_spin_flip(Term_2elcorr_SOMO)
    Term_2elcorr_SOMO += hermitian_conjugated(Term_2elcorr_SOMO)

    if debug:
        print('\nTerm_1el:')
        print(Term_1el)
        print('\nTerm_rpsp_2el_corr:')
        print(Term_rpsp_2el_corr)
        print('\nTerm_2elcorr_SOMO:')
        print(Term_2elcorr_SOMO)

   #Now the non-Hermitian part
    Term_eff1e_Ers_same_spin = FermionOperator()
    for p in list_same_spin_rs:
        pa = 2*p
        pb = pa+1
        term = ((ra,1),(pa,1),(pa,0),(sa,0))
        coef = tbt_unscaled[ra,pa,pa,sa] - tbt_unscaled[ra,pa,sa,pa]
        Term_eff1e_Ers_same_spin += FermionOperator(term,coef)

    Term_eff1e_Ers_same_spin += op_spin_flip(Term_eff1e_Ers_same_spin)

    Term_eff1e_Esr_same_spin = FermionOperator()
    for p in list_same_spin_sr:
        pa = 2*p
        pb = pa+1
        term = ((sa,1),(pa,1),(pa,0),(ra,0))
        coef = tbt_unscaled[sa,pa,pa,ra] - tbt_unscaled[sa,pa,ra,pa]
        Term_eff1e_Esr_same_spin += FermionOperator(term,coef)

    Term_eff1e_Esr_same_spin += op_spin_flip(Term_eff1e_Esr_same_spin)

    Term_eff1e_same_spin += Term_eff1e_Ers_same_spin + Term_eff1e_Esr_same_spin

    if debug:
        print('\nTerm_eff1e_same_spin:')
        print(Term_eff1e_same_spin)

    Term_eff1e_Ers_diff_spin = FermionOperator()
    for p in list_diff_spin_rs:
        pa = 2*p
        pb = pa+1
        term = ((ra,1),(pb,1),(pb,0),(sa,0))
        coef = tbt_unscaled[ra,pb,pb,sa]
        Term_eff1e_Ers_diff_spin += FermionOperator(term,coef)

    Term_eff1e_Ers_diff_spin += op_spin_flip(Term_eff1e_Ers_diff_spin)

    Term_eff1e_Esr_diff_spin = FermionOperator()
    for p in list_diff_spin_sr:
        pa = 2*p
        pb = pa+1
        term = ((sa,1),(pb,1),(pb,0),(ra,0))
        coef = tbt_unscaled[sa,pb,pb,ra]
        Term_eff1e_Esr_diff_spin += FermionOperator(term,coef)

    Term_eff1e_Esr_diff_spin += op_spin_flip(Term_eff1e_Esr_diff_spin)

    Term_eff1e_diff_spin += Term_eff1e_Ers_diff_spin + Term_eff1e_Esr_diff_spin

    if debug:
        print('\nTerm_eff1e_diff_spin:')
        print(Term_eff1e_diff_spin)

    H_offdiag_U_1modiff = Term_1el + Term_rpsp_2el_corr + Term_2elcorr_SOMO + Term_eff1e_same_spin + Term_eff1e_diff_spin

    return H_offdiag_U_1modiff

def make_op_for_offdiag_U_2modiff(tbt_phys,list_pqrs,debug=False):
    """
    Read in the info to generate operator that have nonzero contributions to the matrix elements between
    the bra and ket spaces generated by pair excitations out of reference states differ by two spatial orbital, e.g.,
    <CSF_1|U^+ H U|CSF_iajb>, <CSF_ia|U^+ H U|CSF_jb>, <CSF_iajb|U^+ H U|CSF_iakc>,  etc.
    list_pqrs = [p,q,r,s]. <CSF_bra|psig^ rsig qsig'^ ssig'|CSF_ket> != 0. sig and sig' for spin
    For <CSF_1|U^+ H U|CSF_iajb>, list_pqrs = [i,j,a,b]
    For <CSF_ia|U^+ H U|CSF_jb>, list_pqrs = [a,j,i,b]
    """

    if debug: print('\nIn make_op_for_offdiag_U_2modiff')
    p, q, r, s = list_pqrs[0],list_pqrs[1],list_pqrs[2],list_pqrs[3]
    pa, pb, qa, qb = 2*p, 2*p+1, 2*q, 2*q+1
    ra, rb, sa, sb = 2*r, 2*r+1, 2*s, 2*s+1
    if debug: print(f'p q r s in make_op_for_offdiag_U_2modiff: {p,q,r,s}')

    tbt_unscaled = tbt_phys*2.0

    H_offdiag_U_2modiff = FermionOperator()

    term = ((pa,1),(qa,1),(sa,0),(ra,0))
    coef = tbt_unscaled[pa,qa,sa,ra] - tbt_unscaled[pa,qa,ra,sa]
    H_offdiag_U_2modiff += FermionOperator(term,coef)

    term = ((pa,1),(qb,1),(sb,0),(ra,0))
    coef = tbt_unscaled[pa,qb,sb,ra]
    H_offdiag_U_2modiff += FermionOperator(term,coef)

    term = ((pa,1),(sa,1),(qa,0),(ra,0))
    coef = tbt_unscaled[pa,sa,qa,ra] - tbt_unscaled[pa,sa,ra,qa]
    H_offdiag_U_2modiff += FermionOperator(term,coef)
    
    term = ((pa,1),(sb,1),(qb,0),(ra,0))
    coef = tbt_unscaled[pa,sb,qb,ra]
    H_offdiag_U_2modiff += FermionOperator(term,coef)

    term = ((pa,1),(rb,1),(qb,0),(sa,0))
    coef = tbt_unscaled[pa,rb,qb,sa]
    H_offdiag_U_2modiff += FermionOperator(term,coef)

    term = ((pa,1),(rb,1),(sb,0),(qa,0))
    coef = tbt_unscaled[pa,rb,sb,qa]
    H_offdiag_U_2modiff += FermionOperator(term,coef)

    H_offdiag_U_2modiff += op_spin_flip(H_offdiag_U_2modiff)
    H_offdiag_U_2modiff += hermitian_conjugated(H_offdiag_U_2modiff)

    if debug:
        print(f'\nH_offdiag_U_2modiff:\n')
        print(H_offdiag_U_2modiff)

    return H_offdiag_U_2modiff
    
def make_op_for_offdiag_UCSFia_UCSFjb(tbt_phys,list_CIS_pair,debug=False):
    """ 
    make operators that contribute to off-diagonal matrix elements between bras in U|CSFia> space
    and kets in U|CSF_jb> space
    """ 

    if debug: print('\nIn make_op_for_offdiag_UCSFia_UCSFjb_new')

    lbomb = False
    if len(list_CIS_pair)!= 2: lbomb = True
    for pair in list_CIS_pair:
        if len(pair) != 2: lbomb = True

    i,a,j,b = list_CIS_pair[0][0],list_CIS_pair[0][1],list_CIS_pair[1][0],list_CIS_pair[1][1]

    if i == j or a == b or i == a or j == b: lbomb = True

    if lbomb:
        print('\nInappropriate list_CIS_pair: {list_CIS_pair}. Bombing out!')
        sys.exit()

    list_pqrs = [a,j,i,b]

    H_offdiag_U_2modiff = make_op_for_offdiag_U_2modiff(tbt_phys,list_pqrs,debug)
    return H_offdiag_U_2modiff

def make_op_for_offdiag_UCSFia_UCSFibjc(tbt_phys,list_CIS_pair,debug=False):
    """
    make operators that contribute to off-diagonal matrix elements between bras in U|CSFia> space
    and kets in U|CSF_ibjc> space.
    list_CIS_pair = [[i,a],[i,b],[j,c]]
    """

    if debug: print('\nIn make_op_for_offdiag_UCSFia_UCSFibjc')
    
    lbomb = False
    if len(list_CIS_pair) != 3: lbomb = True

    if list_CIS_pair[0][0] != list_CIS_pair[1][0]: lbomb = True
    i = list_CIS_pair[0][0]
    a = list_CIS_pair[0][1]
    b = list_CIS_pair[1][1]
    j = list_CIS_pair[2][0]
    c = list_CIS_pair[2][1]
        
    if i == j or a == b or a == c or b == c: lbomb = True

    if lbomb:
        print('\nInappropriate list_CIS_pair in make_op_for_offdiag_UCSFia_UCSFibjc. Bombing out!')
        print(list_CIS_pair)
        sys.exit()

    list_pqrs = [a,j,b,c]

    H_offdiag_U_2modiff = make_op_for_offdiag_U_2modiff(tbt_phys,list_pqrs,debug)
    return H_offdiag_U_2modiff

def make_op_for_offdiag_UCSFia_UCSFjakb(tbt_phys,list_CIS_pair,debug=False):
    """
    make operators that contribute to off-diagonal matrix elements between bras in U|CSFia> space
    and kets in U|CSF_jakb> space.
    list_CIS_pair = [[i,a],[j,a],[k,b]]
    """

    if debug: print('\nIn make_op_for_offdiag_UCSFia_UCSFjakb')

    lbomb = False
    if len(list_CIS_pair) != 3: lbomb = True
    
    if list_CIS_pair[0][1] != list_CIS_pair[1][1]: lbomb = True
    i = list_CIS_pair[0][0]
    a = list_CIS_pair[0][1]
    j = list_CIS_pair[1][0]
    k = list_CIS_pair[2][0]
    b = list_CIS_pair[2][1]
    
    if i == j or i == k or j == k or a == b: lbomb = True

    if lbomb:
        print('\nInappropriate list_CIS_pair in make_op_for_offdiag_UCSFia_UCSFjakb. Bombing out!')
        print(list_CIS_pair)
        sys.exit()

    list_pqrs = [j,k,i,b]

    H_offdiag_U_2modiff = make_op_for_offdiag_U_2modiff(tbt_phys,list_pqrs,debug)
    return H_offdiag_U_2modiff

def make_op_for_offdiag_UCSFiajb_UCSFiakc(tbt_phys,list_CIS_pair,debug=False):
    """
    make operators that contribute to off-diagonal matrix elements between bras in U|CSFiajb> space
    and kets in U|CSF_iakc> space.
    list_CIS_pair = [[i,a],[j,b],[i,a],[k,c]]
    """

    if debug: print('\nIn make_op_for_offdiag_UCSFia_UCSFjakb')

    lbomb = False
    if len(list_CIS_pair) != 4: lbomb = True

    if list_CIS_pair[0] != list_CIS_pair[2]: lbomb = True

    i,a,j,b, = list_CIS_pair[0][0],list_CIS_pair[0][1],list_CIS_pair[1][0],list_CIS_pair[1][1]
    k,c      = list_CIS_pair[3][0],list_CIS_pair[3][1]

    if i == j or i == k or j == k or i == a or j == b or k == c or a == b or a == c or b == c: lbomb = True

    if lbomb:
        print(f'\nInappropriate list_CIS_pair in make_op_for_offdiag_UCSFiajb_UCSFiakc: {list_CIS_pair}')
        print('Bombing out')
        sys.exit()
    
    list_pqrs = [b,k,j,c]

    H_offdiag_U_2modiff = make_op_for_offdiag_U_2modiff(tbt_phys,list_pqrs,debug)
    return H_offdiag_U_2modiff

def make_op_for_offdiag_UCSFiajb_UCSFkalb(tbt_phys,list_CIS_pair,debug=False):
    """
    make operators that contribute to off-diagonal matrix elements between bras in U|CSFiajb> space
    and kets in U|CSF_kalb> space.
    list_CIS_pair = [[i,a],[j,b],[k,a],[l,c]]
    """

    if debug: print('\nIn make_op_for_offdiag_UCSFia_UCSFkalb')

    lbomb = False
    if len(list_CIS_pair) != 4: lbomb = True

    if list_CIS_pair[0][1] != list_CIS_pair[2][1] or list_CIS_pair[1][1] != list_CIS_pair[3][1]: lbomb = True

    i,a,j,b = list_CIS_pair[0][0], list_CIS_pair[0][1], list_CIS_pair[1][0], list_CIS_pair[1][1]
    k,l     = list_CIS_pair[2][0], list_CIS_pair[3][0]

    list_SOMO = [i,j,k,l,a,b]
    for orb in list_SOMO:
        if len(np.where(list_SOMO == orb)[0]) > 1: lbomb = True

    if lbomb:
        print(f'\nInappropriate list_CIS_pair in make_op_for_offdiag_UCSFiajb_UCSFkalb: {list_CIS_pair}')
        print('Bombing out!')
        sys.exit()

    list_pqrs = [k,l,i,j]

    H_offdiag_U_2modiff = make_op_for_offdiag_U_2modiff(tbt_phys,list_pqrs,debug)
    return H_offdiag_U_2modiff

def make_op_for_offdiag_UCSFiajb_UCSFicjd(tbt_phys,list_CIS_pair,debug=False):
    """
    make operators that contribute to off-diagonal matrix elements between bras in U|CSFiajb> space
    and kets in U|CSF_icjd> space.
    list_CIS_pair = [[i,a],[j,b],[i,c],[j,d]]
    """

    if debug: print('\nIn make_op_for_offdiag_UCSFia_UCSFicjd')

    lbomb = False
    if len(list_CIS_pair) != 4: lbomb = True

    if list_CIS_pair[0][0] != list_CIS_pair[2][0] or list_CIS_pair[1][0] != list_CIS_pair[3][0]:
        lbomb = False

    i,a,j,b = list_CIS_pair[0][0], list_CIS_pair[0][1], list_CIS_pair[1][0], list_CIS_pair[1][1]
    c,d     = list_CIS_pair[2][1], list_CIS_pair[3][1]

    list_SOMO = [i,j,a,b,c,d]
    for orb in list_SOMO:
        if len(np.where(list_SOMO == orb)[0]) > 1: lbomb = True

    if lbomb:
        print(f'\nInappropriate list_CIS_pair in make_op_for_offdiag_UCSFiajb_UCSFicjd: {list_CIS_pair}')
        print('Bombing out!')
        sys.exit()

    list_pqrs = [a,b,c,d]

    H_offdiag_U_2modiff = make_op_for_offdiag_U_2modiff(tbt_phys,list_pqrs,debug)
    return H_offdiag_U_2modiff

def triplet_pair_singlet_ex(on_list,idx_list,coef_vec,list_Thp_pair,debug=False):
    """
    Given an input state with on_list, idx_list, and coef_vec, generate
    a singlet-coupled triplet pair excited state. list_Thp_pair = [[i,a],[j,b]] is a list
    of two triplet hole particle orbital pairs.
    """

    if debug: print('\nIn triplet_pair_singlet_ex')

    lbomb = False
    i,a,j,b = list_Thp_pair[0][0],list_Thp_pair[0][1],list_Thp_pair[1][0],list_Thp_pair[1][1]
    list_SOMO = [i,a,j,b]
    for orb in list_SOMO:
        if len(np.where(list_SOMO == orb)[0]) > 1: lbomb = True

    if lbomb:
        print(f'\nInappropriate list_Thp_pair in triplet_pair_singlet_ex: {list_Thp_pair}')
        print(f'Bombing out!')
        sys.exit()
    

    T_ia_1p1, T_ia_1_0, T_ia_1m1 = get_Tia_1m(i,a)
    T_jb_1p1, T_jb_1_0, T_jb_1m1 = get_Tia_1m(j,b)
    on_list_Tia1p1_hf, idx_list_Tia1p1_hf,coef_Tia1p1_hf = op_action_tz_remove_0coef(T_ia_1p1,on_list,idx_list,coef_vec)
    on_list_Tia1_0_hf, idx_list_Tia1_0_hf,coef_Tia1_0_hf = op_action_tz_remove_0coef(T_ia_1_0,on_list,idx_list,coef_vec)
    on_list_Tia1m1_hf, idx_list_Tia1m1_hf,coef_Tia1m1_hf = op_action_tz_remove_0coef(T_ia_1m1,on_list,idx_list,coef_vec)
    on_list_Tjb1m1Tia1p1_hf, idx_list_Tjb1m1Tia1p1_hf,coef_Tjb1m1Tia1p1_hf = \
        op_action_tz_remove_0coef(T_jb_1m1,on_list_Tia1p1_hf,idx_list_Tia1p1_hf,coef_Tia1p1_hf)
    on_list_Tjb1_0Tia1_0_hf, idx_list_Tjb1_0Tia1_0_hf,coef_Tjb1_0Tia1_0_hf = \
        op_action_tz_remove_0coef(T_jb_1_0,on_list_Tia1_0_hf,idx_list_Tia1_0_hf,coef_Tia1_0_hf)
    on_list_Tjb1p1Tia1m1_hf, idx_list_Tjb1p1Tia1m1_hf,coef_Tjb1p1Tia1m1_hf = \
        op_action_tz_remove_0coef(T_jb_1p1,on_list_Tia1m1_hf,idx_list_Tia1m1_hf,coef_Tia1m1_hf)

    coef_Tjb1m1Tia1p1_hf *= -np.sqrt(1.0/3.0)
    coef_Tjb1_0Tia1_0_hf *=  np.sqrt(1.0/3.0)
    coef_Tjb1p1Tia1m1_hf *= -np.sqrt(1.0/3.0)

    on_list_Tiajbtt_hf  = on_list_Tjb1m1Tia1p1_hf + on_list_Tjb1p1Tia1m1_hf + on_list_Tjb1_0Tia1_0_hf
    coef_Tiajbtt_hf     = np.concatenate((coef_Tjb1m1Tia1p1_hf, coef_Tjb1p1Tia1m1_hf, coef_Tjb1_0Tia1_0_hf))
    idx_list_Tiajbtt_hf = idx_list_Tjb1m1Tia1p1_hf + idx_list_Tjb1p1Tia1m1_hf + idx_list_Tjb1_0Tia1_0_hf

   #Judge whether there is any duplicate on vector? 
    for idx in idx_list_Tiajbtt_hf:
        if len(np.where(idx_list_Tiajbtt_hf == idx)[0]) > 1:
            print(f'Duplicate on vector detected')
            print(on_list_Tiajbtt_hf[np.where(idx_list_Tiajbtt_hf == idx)])
            print('Bombing out!')
            sys.exit()

   #Check normalization
    coef_norm = np.linalg.norm(coef_Tiajbtt_hf)
    if not np.isclose(coef_norm,1.0):
        print(f'Non-normalized state generated by triplet_pair_singlet_ex')
        print(f'Norm = {coef_norm}')
        print(coef_Tiajbtt_hf)
        sys.exit()

    if debug:
        print(f'State generated by triplet_pair_singlet_ex:')
        for ii, item in enumerate(on_list_Tiajbtt_hf):
            print(coef_Tiajbtt_hf[ii],item)

    return on_list_Tiajbtt_hf, idx_list_Tiajbtt_hf, coef_Tiajbtt_hf

def energy_SD(Enuc,obt_phys,tbt_phys,onvec,lrdm=False,debug=False):
    """
    Calculate energy of a Slater determinant with on vec
    """

    if debug: 
        print('\nIn energy_SD')
        print(f'ON: {onvec}')

    if lrdm:
   #    n_spinorb = obt_phys.shape[0]
   #    rdm1 = np.zeros([n_spinorb,n_spinorb])
   #    rdm2 = np.zeros([n_spinorb,n_spinorb,n_spinorb,n_spinorb])
   #    rdm1 = csr_matrix((n_spinorb,n_spinorb))
   #    rdm2 = csr_matrix((n_spinorb,n_spinorb,n_spinorb,n_spinorb))
        list_rdm1 = []
        list_rdm2 = []

    spinorb_occupied = np.where(onvec != 0.0)[0]
    if debug: print(f'In energy_SD, spinorb_occupied: {spinorb_occupied}')
    E1e = 0.0
    E2e = 0.0
    for ii,iorb in enumerate(spinorb_occupied):
        E1e += obt_phys[iorb,iorb]
       #if lrdm: rdm1[iorb,iorb] += 1.0
        if lrdm: list_rdm1.append([iorb,iorb,1.0])
        for jj in range(ii+1,len(spinorb_occupied)):
            jorb = spinorb_occupied[jj]
            E2e += tbt_phys[iorb,jorb,jorb,iorb]*2.0
            if lrdm:
           #    rdm2[iorb,jorb,iorb,jorb] += 1.0
           #    rdm2[jorb,iorb,jorb,iorb] += 1.0
                list_rdm2.append([iorb,jorb,iorb,jorb,1.0])
                list_rdm2.append([jorb,iorb,jorb,iorb,1.0])
            if (iorb + jorb) % 2 == 0:
                E2e -= tbt_phys[iorb,jorb,iorb,jorb]*2.0
                if lrdm:
               #    rdm2[iorb,jorb,jorb,iorb] -= 1.0
               #    rdm2[jorb,iorb,iorb,jorb] -= 1.0
                    list_rdm2.append([iorb,jorb,jorb,iorb,-1.0])
                    list_rdm2.append([jorb,iorb,iorb,jorb,-1.0])

    E_SD = E1e + E2e + Enuc
    if debug:
        print(f'1e energy of SD: {E1e}')
        print(f'2e energy of SD: {E2e}')
        print(f'total energy of SD: {E_SD}')
       #H_short = make_short_H_ferm_op(Enuc,obt_phys,tbt_phys)
       #E_SD_check = braket_tz(onvec,onvec,H_short)
       #assert np.isclose(E_SD,E_SD_check)
       #if lrdm:
       #    E1e_from_rdm1 = elm_from_rdm1(rdm1,obt_phys)
       #    E2e_from_rdm2 = elm_from_rdm2(rdm2,tbt_phys)
       #    print(f'E1e from rdm1: {E1e_from_rdm1}')
       #    print(f'E2e from rdm2: {E2e_from_rdm2}')
       #    E_from_rdm = E1e_from_rdm1+E2e_from_rdm2 + Enuc
       #    if not np.isclose(E_SD,E_from_rdm):
       #        print(f'E_SD: {E_SD} vs. E_from_rdm: {E_from_rdm}. Bombing out!')
       #        sys.exit()
        if lrdm:
           #E1e_from_rdm1 = 0.0
           #for [i,j,rdmelm] in list_rdm1:
           #    E1e_from_rdm1 += rdmelm*obt_phys[j,i]
            E1e_from_rdm1 = elm_from_list_rdm1(list_rdm1,obt_phys)

            assert np.isclose(E1e_from_rdm1,E1e)

           #E2e_from_rdm2 = 0.0
           #for [i,j,k,l,rdmelm] in list_rdm2:
           #    E2e_from_rdm2 += rdmelm*tbt_phys[k,l,j,i]
            E2e_from_rdm2 = elm_from_list_rdm2(list_rdm2,tbt_phys)

            assert np.isclose(E2e_from_rdm2,E2e)
           #if not np.isclose(E2e_from_rdm2,E2e):
           #    print(f'E2e_from_rdm2: {E2e_from_rdm2} vs E2e: {E2e}. Bombing out!')
           #    sys.exit()
            
       
    if lrdm:
        return E_SD, list_rdm1, list_rdm2
    else:
        return E_SD

def elm_from_rdm1(rdm1,obt_phys):
    """
    Calculate matrix elements given 1-el reduced density matrix and 1-el integral
    """

    return np.trace(rdm1.transpose()@obt_phys)

def elm_from_list_rdm1(list_rdm1,obt_phys):
    """
    Calculate matrix elements given a list of 1-el reduced density matrix elements and 1-el integral
    """

    E1e_from_rdm1 = 0.0
    for [i,j,rdmelm] in list_rdm1:
        E1e_from_rdm1 += rdmelm*obt_phys[j,i]

    return E1e_from_rdm1

def elm_from_rdm2(rdm2,tbt_phys):
    """
    Calculate matrix elements given 2-el reduced density matrix and 2-el integral in
    physicist notation. Formula: rdm2[i,j,k,l]*tbt_phys[k,l,j,i]
    """

    return np.einsum('ijkl,klji',rdm2,tbt_phys,optimize=True)

def elm_from_list_rdm2(list_rdm2,tbt_phys):
    """
    Calculate matrix elements given a list of 2-el reduced density matrix elements and 2-el integral
    in physicist notation. Formula: rdm2[i,j,k,l]*tbt_phys[k,l,j,i]
    """

    E2e_from_rdm2 = 0.0
    for [i,j,k,l,rdmelm] in list_rdm2:
        E2e_from_rdm2 += rdmelm*tbt_phys[k,l,j,i]

    return E2e_from_rdm2

def elm_from_list_rdm(list_rdm1,list_rdm2,const_per_electron,obt_phys,tbt_phys):
    """
    Return matrix elements from 1- and 2-el reduced density matrices
    """
    elm1e = elm_from_list_rdm1(list_rdm1,obt_phys)
    elm2e = elm_from_list_rdm2(list_rdm2,tbt_phys)
    dsum = 0.0
    for item in list_rdm1:
        if item[0] == item[1]: dsum += item[-1]

    elm_const = dsum * const_per_electron

    elm = elm_const + elm1e + elm2e
    return elm

def from_rdm_to_oper(list_rdm1,list_rdm2):
    """
    Read in rdm1 and rdm2 and return the 2nd quantized operators that give the rdms
    """

    op = FermionOperator()
    for item in list_rdm1:
        [q,p,elm] = item
        print(p,q,elm,item)
        term = ((int(p),1),(int(q),0))
        print(f'term: {term}, elm: {elm}')
        print(type(p),type(q),type(1),type(0))
        op += FermionOperator(term,elm)

    for item in list_rdm2:
        print(item)
        [s,r,p,q,elm] = item
       #if max(q,s) == max(p,r) or min(q,s) == min(p,r): 
        if p in [r, s] or q in [r, s]: continue #Essentially 1-el oper multiplied by occ oper
        if p < q or s < r: continue # To remove the duplication for swapping el-1 and -2.
        term = ((int(p),1),(int(q),1),(int(r),0),(int(s),0))
        print(term,elm)
        op += FermionOperator(term,elm)

    op = normal_ordered(op)

    return op

def Helm_between_SDs(Enuc,obt_phys,tbt_phys,onl,onr,lrdm,debug=False):
    """
    Calculate off-diagonal matrix elements between two SDs
    """

    if debug: 
        print('\nIn Helm_between_SDs')
        print(f'on bra: {onl}')
        print(f'on ket: {onr}')

    Helm = 0.0
    discoin = np.where(onl != onr)[0]
    if debug: print(discoin)
    nel_l = len(np.where(onl == 1.0)[0])
    nel_r = len(np.where(onr == 1.0)[0])
    if nel_r != nel_l:
        print(f'\nInconsistent numbers of electrons in bra and ket')
        print('Bombing out')
        sys.exit()

    if len(discoin) > 4: 
        if lrdm: 
            return Helm, [], []
        else:
            return Helm

    if len(discoin) == 0: 
       #tic = time.perf_counter()
        if lrdm: 
            Helm, list_rdm1, list_rdm2 = energy_SD(Enuc,obt_phys,tbt_phys,onr,lrdm,debug)
        else:
            Helm = energy_SD(Enuc,obt_phys,tbt_phys,onr,lrdm,debug)
       #toc = time.perf_counter()
       #print(f'time for energy_SD: {toc-tic}')
        if lrdm:
            return Helm, list_rdm1, list_rdm2
        else:
            return Helm

    if len(discoin) == 2:
        if lrdm:
            list_rdm1 = []
            list_rdm2 = []
        spin_occ_common = np.where((onl == onr) & (onl == 1.0) )[0]
        if debug: print(f'common occupied spin orbitals: {spin_occ_common}')
        iorb,jorb = discoin[0],discoin[1]
       #Still continue even if spin is not conserved
       #if (iorb + jorb) % 2 == 1: 
       #    if lrdm:
       #        return Helm, list_rdm1,list_rdm2 #matrix elements of different spins are zero
       #    else:
       #        return Helm
        if debug: print(onl[iorb+1:jorb])
       #If there are odd commonly occupied orbitals between disjoit orbitals, flip phase
        iphase = len(np.where(onl[iorb+1:jorb] == 1.0)[0])
        phase = (-1.0)**iphase
        
        Helm = obt_phys[iorb,jorb]
        if debug: print(Helm)
        for porb in spin_occ_common:
            Helm += 2.0*(tbt_phys[iorb,porb,porb,jorb]-tbt_phys[iorb,porb,jorb,porb])
            if debug: print(Helm)

        Helm *= phase
        if lrdm: 
       #Up to here, iorb < jorb and this does not matter for real-valued Hamiltonian.
       #But in constructing rdm, we need to clarify which of the two is an occupied
       #spin orbital in the bra (onl) and which is occupied in the ket (onr)
           #print(onl,onr)
           #print(iorb,jorb)
            if onl[iorb] > onr[iorb]:
                iiorb, jjorb = iorb, jorb
            else:
                iiorb, jjorb = jorb, iorb
           #print(iiorb,jjorb)
       
            list_rdm1.append([jjorb,iiorb,1.0*phase])
            for porb in spin_occ_common:
                list_rdm2.append([jjorb,porb,iiorb,porb,1.0*phase])
                list_rdm2.append([porb,jjorb,porb,iiorb,1.0*phase])
                list_rdm2.append([jjorb,porb,porb,iiorb,-1.0*phase])
                list_rdm2.append([porb,jjorb,iiorb,porb,-1.0*phase])

        if debug:
          #H_short = make_short_H_ferm_op(Enuc,obt_phys,tbt_phys)
          #Helm_check = braket_tz(onl,onr,H_short)
          #print(Helm,Helm_check)
          #assert np.isclose(Helm,Helm_check)
           if lrdm:
               Helm_1e_from_rdm1 = elm_from_list_rdm1(list_rdm1,obt_phys)
               Helm_2e_from_rdm2 = elm_from_list_rdm2(list_rdm2,tbt_phys)
               Helm_from_rdm = Helm_1e_from_rdm1 + Helm_2e_from_rdm2
               print('Testing rdm for UCSF_ia')
               assert np.isclose(Helm_from_rdm,Helm)

        if lrdm:
            return Helm, list_rdm1, list_rdm2
        else:
            return Helm

    if len(discoin) == 4:
        if lrdm:
            list_rdm1 = []
            list_rdm2 = []
        occ_l = [] #array of spin orbitals that are only occupied in bra
        occ_r = [] #array of spin orbitals that are only occupied in ket
        for ii,orb in enumerate(discoin):
            if onl[orb] == 1.0:
                if debug: print(orb)
                occ_l.append(orb)
            else:
                occ_r.append(orb)

        occ_l = np.array(occ_l)
        occ_r = np.array(occ_r)
        if debug:
            print(f'occ_l: {occ_l}, {occ_l % 2}')
            print(f'occ_r: {occ_r}, {occ_r % 2}')
        if np.sum(occ_l % 2) != np.sum(occ_r % 2):
            if debug: print(f'The 2-spin-orb disjointedness is not spin-conserved')
            Helm = 0.0

        p,q = occ_l[0],occ_l[1]
        r,s = occ_r[0],occ_r[1]
        if np.sum(occ_l % 2) == 0 or np.sum(occ_l % 2) == 2: # <aa|aa> or <bb|bb> combination
           #iphase = len(np.where(onl[p+1:r] == 1.0)[0]) + len(np.where(onl[q+1:s] == 1.0)[0])
           #phase = (-1.0)**iphase
            Helm = 2.0*(tbt_phys[p,q,s,r] - tbt_phys[p,q,r,s])
            if lrdm:
                list_rdm2.append([r,s,p,q,1.0])
                list_rdm2.append([s,r,q,p,1.0])
                list_rdm2.append([r,s,q,p,-1.0])
                list_rdm2.append([s,r,p,q,-1.0])
           #Helm *= phase
        if np.sum(occ_l % 2) == 1: #<ab|ab> combination
            if (occ_l % 2)[0] != (occ_r % 2)[0]: #swap a set of orbitals to align from <ab|ba> to <ab|ab>
                q,p = occ_l[0],occ_l[1]
           #iphase = len(np.where(onl[p+1:r] == 1.0)[0]) + len(np.where(onl[q+1:s] == 1.0)[0])
           #phase = (-1.0)**iphase    
           #print(f'phase = {phase}')
            Helm = 2.0*tbt_phys[p,q,s,r]
            if lrdm:
                list_rdm2.append([r,s,p,q,1.0])
                list_rdm2.append([s,r,q,p,1.0])
           #Helm *= phase

       #
        if debug: print(f'p,q,r,s={p,q,r,s}')
        pr_min, pr_max = min(p,r), max(p,r)
        iphase = len(np.where(onl[pr_min+1:pr_max] == 1.0)[0])
        if debug: print(f'iphase = {iphase}')
        onl_tmp = copy.deepcopy(onl)
        onl_tmp[p] = 0.0
        onl_tmp[r] = 1.0
        qs_min, qs_max = min(q,s), max(q,s)
        iphase += len(np.where(onl_tmp[qs_min+1:qs_max] == 1.0)[0])
        if debug: print(f'iphase = {iphase}')
        onl_tmp[q] = 0.0
        onl_tmp[s] = 1.0
        assert np.allclose(onl_tmp,onr)
        phase = (-1.0)**iphase
        if debug: print(f'phase = {phase}')
        Helm *= phase
        if lrdm:
            for item in list_rdm2:
                item[-1] *= phase

        if debug:
       #   H_short = make_short_H_ferm_op(Enuc,obt_phys,tbt_phys)
       #   Helm_check = braket_tz(onl,onr,H_short)
       #   print(Helm,Helm_check)
       #   assert np.isclose(Helm,Helm_check)
           if lrdm:
               Helm_2e_from_rdm2 = elm_from_list_rdm2(list_rdm2,tbt_phys)
               assert np.isclose(Helm_2e_from_rdm2,Helm)

        if lrdm:
            return Helm, list_rdm1, list_rdm2
        else:
            return Helm

def Helm_between_LCSDs(Enuc,obt_phys,tbt_phys,list_onl,coefs_l,list_onr,coefs_r,lrdm=False,debug=False):
    """
    Calculate Hamiltonian matrix elements between the read-in bra and ket states (l and r)
    """
    if debug: print('\nIn Helm_between_LCSDs')

    Helm = 0.0
    if lrdm:
        list_rdm1_between_LCSDs = []
        list_rdm2_between_LCSDs = []
        list_rdm1_index = []
        list_rdm2_index = []
    for ii, onl in enumerate(list_onl):
        coef_ii = coefs_l[ii]
        for jj, onr in enumerate(list_onr):
            coef_jj = coefs_r[jj]
            if lrdm:
                Helm_SDs, list_rdm1, list_rdm2 = Helm_between_SDs(Enuc,obt_phys,tbt_phys,onl,onr,lrdm,False)
                elm_from_rdm1 = elm_from_list_rdm1(list_rdm1,obt_phys)
                elm_from_rdm2 = elm_from_list_rdm2(list_rdm2,tbt_phys)
                elm_from_rdm = elm_from_rdm1 + elm_from_rdm2
                if np.isclose(np.sum(abs(onl - onr)),0.0): elm_from_rdm += Enuc
                assert np.isclose(elm_from_rdm, Helm_SDs)
            else:
                Helm_SDs = Helm_between_SDs(Enuc,obt_phys,tbt_phys,onl,onr,lrdm,False)

            Helm += Helm_SDs*coef_ii*coef_jj
            if lrdm:
                for item in list_rdm1:
                    item[-1] *= coef_ii*coef_jj
                    if item[0:-1] in list_rdm1_index:
                        pair_index = list_rdm1_index.index(item[0:-1])
                        list_rdm1_between_LCSDs[pair_index][-1] += item[-1]
                    else:
                        list_rdm1_between_LCSDs.append(item)
                        list_rdm1_index.append(item[0:-1])
                for item in list_rdm2:
                    item[-1] *= coef_ii*coef_jj
                    if item[0:-1] in list_rdm2_index:
                        quadruple_index = list_rdm2_index.index(item[0:-1])
                        list_rdm2_between_LCSDs[quadruple_index][-1] += item[-1]
                    else:
                        list_rdm2_between_LCSDs.append(item)
                        list_rdm2_index.append(item[0:-1])


    if lrdm:
        return Helm, list_rdm1_between_LCSDs, list_rdm2_between_LCSDs
    else:
        return Helm

def Helm_between_CSFs(Enuc,obt_phys,tbt_phys,CSFbra,CSFket,lrdm=False):
    """
    Calculate Hamiltonian matrix elements between read in bra and ket CSFs
    """

    list_onl = CSFbra[0]
    coefs_l = CSFbra[2]
    list_onr = CSFket[0]
    coefs_r = CSFket[2]

    n_spinorb = obt_phys.shape[0]

    if lrdm:
        Helm, list_rdm1, list_rdm2 = Helm_between_LCSDs(Enuc,obt_phys,tbt_phys,list_onl,coefs_l,list_onr,coefs_r,lrdm)
        return Helm, list_rdm1, list_rdm2
    else:
        return Helm_between_LCSDs(Enuc,obt_phys,tbt_phys,list_onl,coefs_l,list_onr,coefs_r)

def make_Hmat_diagonal_space(Enuc,obt_phys,tbt_phys,list_pair_ex_space,list_pair_ex_comb,l_makeHmat=False,debug=False):
    """
    Read in a multiple pair excitation space and calculate Hamiltonian matrix elements
    """

    if debug: print('\nIn make_Hmat_diagonal_space')

    list_start_end = [[0,0]]
    ndim_space = len(list_pair_ex_space)
    ndim_pair_ex_comb = len(list_pair_ex_comb)
    list_pair_ex_comb_1col = []
    level_ex = 0
    for pair_ex_level in range(len(list_pair_ex_comb)):
        level_ex += 1
        nstate_one_ex_level = len(list_pair_ex_comb[pair_ex_level])
        start_state = list_start_end[-1][1]+1
        end_state   = list_start_end[-1][1]+nstate_one_ex_level
        list_start_end.append([start_state,end_state])
        list_pair_ex_comb_1col += list_pair_ex_comb[pair_ex_level]
        if debug: print(f'{nstate_one_ex_level} at the excitation levels of {level_ex} pair(s) of dmo and vmo')

    if debug:
        print(f'Dimension of space: {ndim_space}')
        print(f'Foldness of pair excitations: {ndim_pair_ex_comb}')
        print(f'Start and End states of each excitation foldness: {list_start_end}')
        print('One column of combination of pair excitations:')
        for ipairs, pairs in enumerate(list_pair_ex_comb_1col):
            print(ipairs+1,pairs)

    if len(list_start_end) == 1: return []
   #For each ia_pair, find the state pairs that are connected by it
    if debug:
        print(f'\nAssociating ia_pairs to state pairs\n')
        print(list_start_end[1][0],list_start_end[1][1])
    list_iapair_st_pairs = []
    list_ia_pairs = []
    for state in range(list_start_end[1][0],list_start_end[1][1]+1):
        istate = state -1
        ia_pair = list_pair_ex_comb_1col[istate][0]
        list_ia_pairs.append(ia_pair)
        if debug: print(state,istate,list_pair_ex_comb_1col[istate],ia_pair)
        list_iapair_st_pairs.append([ia_pair,[state,0]])

    if debug: 
        print(f'list_iapair_st_pairs after considering pairexcitaiton level 1:')
        for item in list_iapair_st_pairs:
            print(item)
        print(f'list_ia_pairs: {list_ia_pairs}')

    for level_ex in range(2,ndim_pair_ex_comb+1):
        for state_low in range(list_start_end[level_ex-1][0],list_start_end[level_ex-1][1]+1):
            for state_up in range(list_start_end[level_ex][0],list_start_end[level_ex][1]+1):
                ii_pairs = list_pair_ex_comb_1col[state_low-1]
                jj_pairs = list_pair_ex_comb_1col[state_up-1]
                diff_mo = compare_mp2_ex_pairs(ii_pairs,jj_pairs,False)
                diff_mo = sorted(diff_mo)
                if debug:
                    print(f'Examining pairs of states {state_low,state_up} for level_ex {level_ex}, {ii_pairs,jj_pairs}')
                    print(f'diff_mo: {diff_mo}')
               #if len(diff_mo) != 2:
               #    print(f'The two states in adjacent groups do not differ by one ia pair only.')
               #else:
                if len(diff_mo) == 2:
                    if debug: print(f'States {state_low,state_up} differ by one ia pair')
                    if diff_mo in list_ia_pairs:
                        idx_ia_pair = list_ia_pairs.index(diff_mo)
                       #print(list_ia_pairs)
                       #print(f'And the different pair {diff_mo} is in the mp2 list, item {idx_ia_pair}')
                        list_iapair_st_pairs[idx_ia_pair].append([state_up,state_low])
                    diff_mo_reverse = diff_mo
                    diff_mo_reverse.reverse()
                    if diff_mo[0] == diff_mo[1]:
                        print(f'Two elements in diff_mo are identical: {diff_mo}. Bombing out!')
                        sys.exit()
                    if diff_mo_reverse in list_ia_pairs:
                        idx_ia_pair = list_ia_pairs.index(diff_mo_reverse)
                       #print(list_ia_pairs)
                       #print(f'And the different pair {diff_mo} is in the mp2 list, item {idx_ia_pair}')
                        list_iapair_st_pairs[idx_ia_pair].append([state_up,state_low])
                   #else:
                       #print(f'But the different pair is not in the mp2 list: {diff_mo}')

    if debug:
        print('\nSummary of groupping of state pairs to ia pairs')
        for item in list_iapair_st_pairs:
            print(item)

    if debug:
        for i_ia_pair,ia_pair in enumerate(list_ia_pairs):
            i,a = ia_pair[0],ia_pair[1]
            Tiiaa_00 = get_Tiiaa_00(i,a)
            list_tmp_state_pairs = [[i,a]]
            for jstate in range(ndim_space):
                state_j = list_pair_ex_space[jstate]
                onlist_j = state_j[0]
                coef_j   = state_j[2]
                for istate in range(jstate+1,ndim_space):
                    state_i = list_pair_ex_space[istate]
                    onlist_i = state_i[0]
                    coef_i   = state_i[2]
                    Tiiaa_00_elm = 0.0
                    for iSD, SDi in enumerate(onlist_i):
                        coef_SDi = coef_i[iSD]
                        for jSD, SDj in enumerate(onlist_j):
                            coef_SDj = coef_j[jSD]
                            Tiiaa_00_elm += coef_SDi*coef_SDj*braket_tz(SDi,SDj,Tiiaa_00)

                    if np.isclose(Tiiaa_00_elm,1.0): list_tmp_state_pairs.append([istate,jstate])
                    if np.isclose(Tiiaa_00_elm,-1.0):
                        print(f'Tiiaa_00_elm = -1 detected for {istate,jstate}')
                        print('This should not happen. Something wrong with the order of states.')

           #print(f'list for debugging: {list_tmp_state_pairs}')
            lpass, list_inA_notinB, list_inB_notinA = two_lists_with_same_contents(list_tmp_state_pairs,list_iapair_st_pairs[i_ia_pair])
            if not lpass:
                print(f'Different lists of couped state pairs by ia_pair: {ia_pair}')
                print(f'Stored list: {list_iapair_st_pairs[i_ia_pair]}')
                print(f'debug  list: {list_tmp_state_pairs}')
                print(list_inA_notinB)
                print(list_inB_notinA)
                print('Bombing out')
                sys.exit()

    if not l_makeHmat:
        return list_iapair_st_pairs


    H_sparse = csr_matrix((ndim_space,ndim_space))
    H_sparse_quick = csr_matrix((ndim_space,ndim_space))

    if debug:
        print('\nStates in the mp2 ex space:\n')
        for ii, item in enumerate(list_pair_ex_space):
            print(f'State {ii}: {item}')
    
    for ii, state in enumerate(list_pair_ex_space):
        print(f'\n ii = {ii}, {state}')
        onlist = state[0]
        coef   = state[2]
        E_state = Helm_between_LCSDs(Enuc,obt_phys,tbt_phys,onlist,coef,onlist,coef,False,False)
        H_sparse[ii,ii] = E_state
        H_sparse_quick[ii,ii] = E_state
        if ii == 0: continue
       #Check ii belongs to which group 
        for igroup,group in enumerate(list_start_end):
            if ii >= group[0] and ii <= group[1]:
               #print(f'{ii} belongs to group {igroup}')
               break
        jgroup = igroup - 1
       #The following chunk still involves Helm_between_LCSDs and can be slow
        jj_start = list_start_end[jgroup][0]
        jj_end   = ii
        for jj in range(jj_start,jj_end):
            state_jj = list_pair_ex_space[jj]
            onlist_jj = state_jj[0]
            coef_jj   = state_jj[2]
           #print(f'Calculating States {jj,ii}, jj from {jj_start} to {jj_end}')
           #print(f'onlist_jj: {onlist_jj}')
           #print(f'coef_jj: {coef_jj}')
           #print(f'onlist: {onlist}')
           #print(f'coef: {coef}')
            Helm = Helm_between_LCSDs(Enuc,obt_phys,tbt_phys,onlist,coef,onlist_jj,coef_jj,False,False)
           #print(f'Calculated H{jj,ii}: {Helm}')
            if not np.isclose(Helm,0.0):
                H_sparse[jj,ii] = Helm
                H_sparse[ii,jj] = Helm

       #The following chunk only examines discoincidence of excitation pairs
        jj_start = list_start_end[jgroup][0]
       #jj_end   = list_start_end[jgroup][1]+1
        jj_end   = ii
        ii_pos   = ii - 1
        ii_pairs = list_pair_ex_comb_1col[ii_pos]
        if igroup == 1:
            dmo = ii_pairs[0][0]
            vmo = ii_pairs[0][1]
            dmoa,dmob,vmoa,vmob = dmo*2,dmo*2+1,vmo*2,vmo*2+1
            Helm_quick = 2.0*tbt_phys[dmoa,dmob,vmob,vmoa]
            if not np.isclose(Helm_quick,0.0):
                H_sparse_quick[ii,0] = Helm_quick
                H_sparse_quick[0,ii] = Helm_quick
            print(f'Helm_quick for {jj,ii}: {Helm_quick}')
            jj_start = list_start_end[igroup][0]
        for jj in range(jj_start,jj_end):
            jj_pos = jj-1
            jj_pairs = list_pair_ex_comb_1col[jj_pos]
            diff_mo = compare_mp2_ex_pairs(ii_pairs,jj_pairs,False)
               #for ipair, ii_pair in enumerate(ii_pairs):
               #    if ii_pair not in jj_pairs:
               #        n_ii_pair_not_in_jj_pairs += 1
               #        pair_new = ii_pair
               #if n_ii_pair_not_in_jj_pairs > 1: 
               #    print(f'{jj,ii}, {jj_pairs}, {ii_pairs}, differ by {n_ii_pair_not_in_jj_pairs} pairs. Skipping!')
               #    continue
               #print(f'{jj,ii}, {jj_pairs}, {ii_pairs}, differ by one pair: {pair_new}')
            if len(diff_mo) != 2:
                if debug: print(f'Skipping states {jj,ii} for > 1 pair of different MOs: {diff_mo}')
                continue
            dmo = diff_mo[0]
            vmo = diff_mo[1]
            dmoa,dmob,vmoa,vmob = dmo*2,dmo*2+1,vmo*2,vmo*2+1
            Helm_quick = 2.0*tbt_phys[dmoa,dmob,vmob,vmoa]
            if not np.isclose(Helm_quick,0.0):
                H_sparse_quick[ii,jj] = Helm_quick
                H_sparse_quick[jj,ii] = Helm_quick
                print(f'Helm_quick for {jj,ii}: {Helm_quick}')


    print('\nH_sparse:\n')
    print(H_sparse)

    if debug:
        H_sparse_check = csr_matrix((ndim_space,ndim_space))
        for ii in range(ndim_space):
            state_ii = list_pair_ex_space[ii]
            onlist_ii = state_ii[0]
            coef_ii   = state_ii[2]
            E_state = Helm_between_LCSDs(Enuc,obt_phys,tbt_phys,onlist_ii,coef_ii,onlist_ii,coef_ii,False,False)
            H_sparse_check[ii,ii] = E_state
            for jj in range(ii):
                state_jj = list_pair_ex_space[jj]
                onlist_jj = state_jj[0]
                coef_jj   = state_jj[2]
                Helm = Helm_between_LCSDs(Enuc,obt_phys,tbt_phys,onlist_ii,coef_ii,onlist_jj,coef_jj,False,False)
                if not np.isclose(Helm,0.0):
                    H_sparse_check[ii,jj] = Helm
                    H_sparse_check[jj,ii] = Helm

       #print('\nH_sparse_check:\n')
       #print(H_sparse_check)
        norm_diff = scipy.sparse.linalg.norm(H_sparse - H_sparse_check)
        assert np.isclose(norm_diff,0.0)
        print('\nChecking H_sparse vs H_sparse_quick')
       #print('\nH_sparse_quick:\n')
        print(H_sparse_quick)
        norm_diff = scipy.sparse.linalg.norm(H_sparse - H_sparse_quick)
       #print('\nH_sparse - H_sparse_quick:\n')
       #print(H_sparse - H_sparse_quick)
        assert np.isclose(norm_diff,0.0)

    return H_sparse_quick, list_iapair_st_pairs

def obt_phys_spatial_to_spin(obt_phys_spatial):
    """
    Convert the obt of spatial orbitals to obt of spin orbitals
    """

    n_spatial = obt_phys_spatial.shape[0]
    n_spin = 2*n_spatial
    obt_phys_spin = np.zeros([n_spin,n_spin])
    obt_phys_spin[0:n_spin:2,0:n_spin:2] = obt_phys_spatial
    obt_phys_spin[1:n_spin:2,1:n_spin:2] = obt_phys_spatial

    return obt_phys_spin

def obt_phys_spin_to_spatial(obt_phys_spin):
    """ 
    Convert the obt of spin orbitals to obt of spatial orbitals
    """

    n_spin = obt_phys_spin.shape[0]
    n_spatial = n_spin // 2
    obt_phys_spatial = np.zeros([n_spatial,n_spatial])
    obt_phys_spatial = obt_phys_spin[0:n_spin:2,0:n_spin:2]

    return obt_phys_spatial

def tbt_phys_spatial_to_spin(tbt_phys_spatial):
    """
    Convert the tbt in physicist notation of spatial orbitals 
    to tbt of spin orbitals in physicist notation
    """
 
    n_spatial = tbt_phys_spatial.shape[0]
    n_spin = 2*n_spatial
    tbt_phys_spin = np.zeros([n_spin,n_spin,n_spin,n_spin])
    tbt_phys_spin[0:n_spin:2,0:n_spin:2,0:n_spin:2,0:n_spin:2] = tbt_phys_spatial
    tbt_phys_spin[1:n_spin:2,1:n_spin:2,1:n_spin:2,1:n_spin:2] = tbt_phys_spatial
    tbt_phys_spin[0:n_spin:2,1:n_spin:2,1:n_spin:2,0:n_spin:2] = tbt_phys_spatial
    tbt_phys_spin[1:n_spin:2,0:n_spin:2,0:n_spin:2,1:n_spin:2] = tbt_phys_spatial

    return tbt_phys_spin

def tbt_phys_spin_to_spatial(tbt_phys_spin):
    """
    Convert the tbt in physicist notation of spin orbitals 
    to tbt of spatial orbitals in physicist notation
    """

    n_spin = tbt_phys_spin.shape[0]
    n_spatial = n_spin // 2

    tbt_phys_spatial = np.zeros([n_spatial,n_spatial,n_spatial,n_spatial])
    tbt_phys_spatial = tbt_phys_spin[0:n_spin:2,0:n_spin:2,0:n_spin:2,0:n_spin:2]

    return tbt_phys_spatial

def compare_mp2_ex_pairs(ii_pairs,jj_pairs,debug=False):
    """
    Read in two lists of lists like [[dmo1,vmo1],[dmo2,vmo2],...]
    Compare them and return the discoincidence of dmo and vmo
    """

    if debug:
        print('\nIn compare_mp2_ex_pairs')
        print(f'ii_pairs: {ii_pairs}, jj_pairs: {jj_pairs}')
    npair_ii = len(ii_pairs)
    npair_jj = len(jj_pairs)
    ii_exmo = []
    for pair in ii_pairs:
        ii_exmo += pair
   #ii_exmo = sorted(ii_exmo)
   #norb = len(ii_exmo)
   #ii_exdmo = ii_exmo[:norb//2]
   #ii_exvmo = ii_exmo[norb//2:]
    ii_exdmo = []
    ii_exvmo = []
    for pair in ii_pairs:
        ii_exdmo.append(pair[0])
        ii_exvmo.append(pair[1])
    jj_exmo = []
    for pair in jj_pairs:
        jj_exmo += pair
            
   #jj_exmo = sorted(jj_exmo)
   #norb = len(jj_exmo)
   #jj_exdmo = jj_exmo[:norb//2]
   #jj_exvmo = jj_exmo[norb//2:]
    jj_exdmo = []
    jj_exvmo = []
    for pair in jj_pairs:
        jj_exdmo.append(pair[0])
        jj_exvmo.append(pair[1])
    if debug:
        print(f'ii_exmo: {ii_exmo}, ii_exdmo: {ii_exdmo}, ii_exvmo: {ii_exvmo}')
        print(f'jj_exmo: {jj_exmo}, jj_exdmo: {jj_exdmo}, jj_exvmo: {jj_exvmo}')
    dmo_in_ii_not_in_jj = list(filter(lambda x: x not in jj_exdmo, ii_exdmo))
    dmo_in_jj_not_in_ii = list(filter(lambda x: x not in ii_exdmo, jj_exdmo))
    vmo_in_ii_not_in_jj = list(filter(lambda x: x not in jj_exvmo, ii_exvmo))
    vmo_in_jj_not_in_ii = list(filter(lambda x: x not in ii_exvmo, jj_exvmo))
    if debug: print(dmo_in_ii_not_in_jj,dmo_in_jj_not_in_ii,vmo_in_ii_not_in_jj,vmo_in_jj_not_in_ii)
   #if npair_ii == npair_jj:
   #    if len(dmo_in_ii_not_in_jj) != len(dmo_in_jj_not_in_ii) or \
   #       len(vmo_in_ii_not_in_jj) != len(vmo_in_jj_not_in_ii):
   #       print(f'Strange! Dimensions not matched')
   #       print(dmo_in_ii_not_in_jj,dmo_in_jj_not_in_ii,vmo_in_ii_not_in_jj,vmo_in_jj_not_in_ii)
   #       print('Bombing out')
   #       sys.exit()
    if abs(len(dmo_in_ii_not_in_jj) - len(dmo_in_jj_not_in_ii)) != abs(npair_ii - npair_jj) or\
       abs(len(vmo_in_ii_not_in_jj) - len(vmo_in_jj_not_in_ii)) != abs(npair_ii - npair_jj):
           print(f'Strange! Dimensions not matched')
           print(dmo_in_ii_not_in_jj,dmo_in_jj_not_in_ii,vmo_in_ii_not_in_jj,vmo_in_jj_not_in_ii)
           print('Bombing out')
           sys.exit()

    diff_pair = vmo_in_ii_not_in_jj + vmo_in_jj_not_in_ii + dmo_in_ii_not_in_jj + dmo_in_jj_not_in_ii
    if debug:
        print(f'diff_pair: {diff_pair}')
   #if len(diff_pair) != 2 and debug:
   #    print(f'Difference more than two doubly occupied orbitals {diff_pair}. Matrix element = 0')

    return diff_pair
    
def two_lists_with_same_contents(listA,listB,debug=False):
    """
    Judge whether two lists have the same contents, regardless of their orders
    """

    lsame = False
    elm_in_A_not_in_B = list(filter(lambda x: x not in listB, listA))
    elm_in_B_not_in_A = list(filter(lambda x: x not in listA, listB))
    if len(listA) != len(listB):
        if debug:
            print('The two lists do not even have the same length')
            print(f'listA: {listA}')
            print(f'listB: {listB}')
        return lsame, elm_in_A_not_in_B, elm_in_B_not_in_A

    if len(elm_in_A_not_in_B) != 0 or len(elm_in_B_not_in_A) != 0:
        if debug:
            print('The two lists do not contain the same elements')
            print(f'Those in A but not in B: {elm_in_A_not_in_B}')
            print(f'Those in B but not in A: {elm_in_B_not_in_A}')
        return lsame,elm_in_A_not_in_B,elm_in_B_not_in_A

    lsame = True
    return lsame, [], []

def overlap_LCSD(onlist_l,idx_list_l,coefs_l,onlist_r,idx_list_r,coefs_r,debug=False):
    """
    Calculate overlap between two states as linear combinations of SDs
    Quick calculation is done using the indices lists. The on vector lists are used for
    debugging only
    """

    if debug: print('\nIn overlap_LCSD')

    ovrlap = 0.0
    for ii, idx_l in enumerate(idx_list_l):
        coefl = coefs_l[ii]
        for jj, idx_r in enumerate(idx_list_r):
            coefr = coefs_r[jj]
            if idx_l == idx_r: ovrlap += coefl*coefr

    return ovrlap

def overlap_CSFs(CSFbra,CSFket,debug=False):
    """     
    Calculate overlap between two CSFs
    """ 

    [onlist_l, idx_list_l, coefs_l] = CSFbra
    [onlist_r, idx_list_r, coefs_r] = CSFket

    return overlap_LCSD(onlist_l,idx_list_l,coefs_l,onlist_r,idx_list_r,coefs_r,debug)


def make_UCSF_state(list_ex_states,coefs_ex_states,debug=False):

    if debug: print('\nIn make_UCSF_state')

    idx_list_output = []
    onlist_output = []
    coef_output = []
    for ii, state in enumerate(list_ex_states):
        coef_state = coefs_ex_states[ii]
        onlist = state[0]
        idx_list = state[1]
        coef_vec = state[2]
        onlist_output += onlist
        idx_list_output += idx_list
        coef_output += (coef_vec*coef_state).tolist()

    coef_output = np.array(coef_output)
    if debug:
        for ii in range(len(coef_output)):
            print(onlist_output[ii],idx_list_output[ii],coef_output[ii])
       #Check normalization
        norm_state = np.linalg.norm(coef_output)
        assert np.isclose(norm_state,1.0)
        norm_state_2nd = overlap_LCSD(onlist_output,idx_list_output,coef_output,onlist_output,idx_list_output,coef_output)
        assert np.isclose(norm_state_2nd,1.0)

    return [onlist_output,idx_list_output,coef_output]

def prepare_UCSFia(list_bound,ref_onlist,ref_idxlist,ref_coefvec,mp2_ia_pairs,group_mp2_ia_pairs,group_mp2_ampld,debug=False):
    """
    Prepare U|CSFia> states, given the read-in information of a HF reference
    list_bound = [i_low,i_up,a_low,a_up], the lower and upper bounds of hole and particle orbitals
    """

    if debug: print('\nIn prepare_UCSFia')

    UCSF_ia_basis = []
    UCSF_ia_CIS_list = []
    UCSF_ia_vec_list = []
    UCSF_ia_list_ex_space = []
    UCSF_ia_list_list_genmat = []
    [i_low, i_up, a_low, a_up] = list_bound
    for i in range(i_up,i_low-1,-1):
        for a in range(a_low,a_up+1):
            T_ia_00 = get_Tia_00(i,a)
            on_list_Tia_hf, idx_list_Tia_hf,coef_Tia_hf = op_action_tz_remove_0coef(T_ia_00,ref_onlist,ref_idxlist,ref_coefvec)
            list_ia_pair_ex_space, list_ia_pair_ex_comb = make_pair_ex_space_fast(on_list_Tia_hf,idx_list_Tia_hf,coef_Tia_hf,mp2_ia_pairs)
            list_iapair_st_pairs = make_Hmat_diagonal_space(None,None,None,list_ia_pair_ex_space,list_ia_pair_ex_comb,False,False)
            list_genmat = make_iapair_genmat_fast(group_mp2_ia_pairs,list_ia_pair_ex_space,list_iapair_st_pairs,False)
            if len(list_genmat) == 0:
                ndim_space = len(list_ia_pair_ex_space)
                Umat_ia = np.eye(ndim_space)
                U_ia_vec = np.zeros([ndim_space])
                U_ia_vec[0] = 1.0
            else:
                Umat_ia, U_ia_vec = make_and_apply_U_matrix(list_genmat,group_mp2_ampld)
            Uia_state = make_UCSF_state(list_ia_pair_ex_space,U_ia_vec)
            UCSF_ia_basis.append(Uia_state)
            UCSF_ia_CIS_list.append([[i,a]])
            UCSF_ia_vec_list.append(U_ia_vec)
            UCSF_ia_list_ex_space.append(list_ia_pair_ex_space)
            UCSF_ia_list_list_genmat.append(list_genmat)

    return UCSF_ia_basis, UCSF_ia_CIS_list, UCSF_ia_vec_list, UCSF_ia_list_ex_space, UCSF_ia_list_list_genmat

def prepare_UCSFiajb_Sss(list_bound,ref_onlist,ref_idxlist,ref_coefvec,mp2_ia_pairs,group_mp2_ia_pairs,group_mp2_ampld,debug=False):
    """ 
    Prepare U|CSFiajb_Sss> states, given the read-in information of a HF reference
    list_bound = [i_low,i_up,a_low,a_up], the lower and upper bounds of hole and particle orbitals
    """ 
            
    if debug: print('\nIn prepare_UCSFia')
            
    UCSF_iajb_basis = []
    UCSF_iajb_CIS_list = []
    UCSF_iajb_vec_list = []
    UCSF_iajb_list_pair_ex = []
    UCSF_iajb_list_Umat = []
    UCSF_iajb_list_list_genmat = []
    [i_low, i_up, a_low, a_up] = list_bound
    for i in range(i_up,i_low-1,-1):
        for a in range(a_low,a_up+1):
            T_ia_00 = get_Tia_00(i,a)
            on_list_Tia_hf, idx_list_Tia_hf,coef_Tia_hf = op_action_tz_remove_0coef(T_ia_00,ref_onlist,ref_idxlist,ref_coefvec)
            for j in range(i-1,i_low-1,-1):
                for b in range(a+1,a_up+1):
                    T_jb_00 = get_Tia_00(j,b)
                    on_list_Tiajb_hf, idx_list_Tiajb_hf,coef_Tiajb_hf = op_action_tz_remove_0coef(T_jb_00,on_list_Tia_hf,idx_list_Tia_hf,coef_Tia_hf)
                    list_iajb_pair_ex_space, list_iajb_pair_ex_comb = make_pair_ex_space_fast(on_list_Tiajb_hf,idx_list_Tiajb_hf,coef_Tiajb_hf,mp2_ia_pairs)             
                    list_iapair_st_pairs = make_Hmat_diagonal_space(None,None,None,list_iajb_pair_ex_space,list_iajb_pair_ex_comb,False,False)
                    list_genmat = make_iapair_genmat_fast(group_mp2_ia_pairs,list_iajb_pair_ex_space,list_iapair_st_pairs,False)
                    if len(list_genmat) == 0:
                        ndim_space = len(list_iajb_pair_ex_space)
                        Umat_iajb = np.eye(ndim_space)
                        U_iajb_vec = np.zeros([ndim_space])
                        U_iajb_vec[0] = 1.0
                    else:
                        Umat_iajb, U_iajb_vec = make_and_apply_U_matrix(list_genmat,group_mp2_ampld)
                    Uiajb_state = make_UCSF_state(list_iajb_pair_ex_space,U_iajb_vec)
                    UCSF_iajb_basis.append(Uiajb_state)
                    UCSF_iajb_CIS_list.append([[i,a],[j,b]])
                    UCSF_iajb_vec_list.append(U_iajb_vec)
                    UCSF_iajb_list_pair_ex.append(list_iajb_pair_ex_space)
                    UCSF_iajb_list_Umat.append(Umat_iajb)
                    UCSF_iajb_list_list_genmat.append(list_genmat)

    return UCSF_iajb_basis, UCSF_iajb_CIS_list, UCSF_iajb_vec_list, UCSF_iajb_list_pair_ex, UCSF_iajb_list_Umat, UCSF_iajb_list_list_genmat

def prepare_UCSFiajb_Stt(list_bound,ref_onlist,ref_idxlist,ref_coefvec,mp2_ia_pairs,group_mp2_ia_pairs,group_mp2_ampld,debug=False):
    """ 
    Prepare U|CSFiajb_Stt> states, given the read-in information of a HF reference
    list_bound = [i_low,i_up,a_low,a_up], the lower and upper bounds of hole and particle orbitals
    """ 
            
    if debug: print('\nIn prepare_UCSFia')

    UCSF_iajb_basis = []
    UCSF_iajb_CIS_list = []
    UCSF_iajb_vec_list = []
    UCSF_iajb_list_pair_ex = []
    Umat_iajb_list = []
    UCSF_iajb_list_list_genmat = []
    [i_low, i_up, a_low, a_up] = list_bound
    for i in range(i_up,i_low-1,-1):
        for a in range(a_low,a_up+1):
            for j in range(i-1,i_low-1,-1):
                for b in range(a+1,a_up+1):
                    list_Thp_pair = [[i,a],[j,b]]
                    on_list_Tiajb_hf, idx_list_Tiajb_hf,coef_Tiajb_hf = triplet_pair_singlet_ex(ref_onlist,ref_idxlist,ref_coefvec,list_Thp_pair)
                    list_iajb_pair_ex_space, list_iajb_pair_ex_comb = make_pair_ex_space_fast(on_list_Tiajb_hf,idx_list_Tiajb_hf,coef_Tiajb_hf,mp2_ia_pairs)
                    list_iapair_st_pairs = make_Hmat_diagonal_space(None,None,None,list_iajb_pair_ex_space,list_iajb_pair_ex_comb,False,False)
                    list_genmat = make_iapair_genmat_fast(group_mp2_ia_pairs,list_iajb_pair_ex_space,list_iapair_st_pairs,False)
                    if len(list_genmat) == 0:
                        ndim_space = len(list_iajb_pair_ex_space)
                        Umat_iajb = np.eye(ndim_space)
                        U_iajb_vec = np.zeros([ndim_space])
                        U_iajb_vec[0] = 1.0
                    else:
                        Umat_iajb, U_iajb_vec = make_and_apply_U_matrix(list_genmat,group_mp2_ampld)
                    Uiajb_state = make_UCSF_state(list_iajb_pair_ex_space,U_iajb_vec)
                    UCSF_iajb_basis.append(Uiajb_state)
                    UCSF_iajb_CIS_list.append([[i,a],[j,b],'tt'])
                    UCSF_iajb_vec_list.append(U_iajb_vec)
                    UCSF_iajb_list_pair_ex.append(list_iajb_pair_ex_space)
                    Umat_iajb_list.append(Umat_iajb)
                    UCSF_iajb_list_list_genmat.append(list_genmat)

    return UCSF_iajb_basis, UCSF_iajb_CIS_list, UCSF_iajb_vec_list, UCSF_iajb_list_pair_ex, Umat_iajb_list, UCSF_iajb_list_list_genmat

def prepare_UCSF_generic(list_CSF,sorted_mp2_ia_pairs,group_mp2_ia_pairs,group_mp2_ampld,debug=False):
    """
    Prepare U|CSF> for a read-in set of CSFs.
    """

    if debug: print('\nIn prepare_UCSF_generic')

    UCSF_basis = []
   #UCSF_SOMO_list = []
    Uvec_list = []
    UCSF_list_list_pair_ex_space = []
    Umat_list = []
    UCSF_list_list_genmat = []

    for ii, CSF in enumerate(list_CSF):
       #if ii % 10 == 0: print(f'Making UCSF for State {ii} of {len(list_CSF)} states')
        [onlist, onidx_list, coefvec] = CSF
        list_pair_ex_space, list_pair_ex_comb = make_pair_ex_space_fast(onlist,onidx_list,coefvec,sorted_mp2_ia_pairs)
        list_iapair_st_pairs = make_Hmat_diagonal_space(None,None,None,list_pair_ex_space,list_pair_ex_comb,False,False)
        list_genmat = make_iapair_genmat_fast(group_mp2_ia_pairs,list_pair_ex_space,list_iapair_st_pairs,False)
        if len(list_genmat) == 0:
            ndim_space = len(list_pair_ex_space)
            Umat = np.eye(ndim_space)
            Uvec = np.zeros([ndim_space])
            Uvec[0] = 1.0
        else:
            Umat, Uvec = make_and_apply_U_matrix(list_genmat,group_mp2_ampld)

        Ustate = make_UCSF_state(list_pair_ex_space,Uvec)
        UCSF_basis.append(Ustate)
        Uvec_list.append(Uvec)
        UCSF_list_list_pair_ex_space.append(list_pair_ex_space)
        Umat_list.append(Umat)
        UCSF_list_list_genmat.append(list_genmat)

    return UCSF_basis, Uvec_list, UCSF_list_list_pair_ex_space, Umat_list, UCSF_list_list_genmat

def select_ia_pairs_for_CSF_E0(list_CSF,list_ex_space,group_ia_pairs,Enuc,obt,tbt,Ethrsh=1.0e-3,list_list_ia_included=[],debug=False):
    """
    Select ia pairs for each CSF for their individual lowerings of the E0 of the CSF space
    """

    if debug: print('\nIn select_ia_pairs_for_CSF_E0')

    Hmat_ex = construct_Hmat_CSFs(list_ex_space,Enuc,obt,tbt)

    Hmat_ex_sparse = csr_matrix(Hmat_ex)
    E0_ex, _ = get_ground_state(Hmat_ex_sparse)
    print(f'\nE0 of original ex space: {E0_ex}')
    nCSF = len(list_CSF)
    if len(list_list_ia_included) == 0:
        for iCSF in range(nCSF):
            list_list_ia_included.append([])
    list_list_genmat = []
    list_list_ex_space = []
    for iCSF, CSF in enumerate(list_CSF):
        print(f'Working on CSF{iCSF}')
        list_ia_one_CSF = []
        list_improve = []
        for ia_pair in group_ia_pairs:
            if ia_pair in list_list_ia_included[iCSF]: continue
            UCSF, Uvec, list_pair_ex_space, Umat, _ = \
              prepare_UCSF_for_one_CSF(CSF,[ia_pair],[0.0],False)
            list_UCSF_space = list_ex_space + list_pair_ex_space[1:]
           #Hmat_UCSF = construct_Hmat_CSFs(list_UCSF_space,Enuc,obt,tbt)
            ndim_UCSF = len(list_ex_space) + len(list_pair_ex_space[1:])
            Hmat_UCSF = np.zeros([ndim_UCSF,ndim_UCSF])
            Hmat_UCSF[0:len(list_ex_space),0:len(list_ex_space)] = Hmat_ex
            list_pair_ex_space_cut = list_pair_ex_space[1:]
            for ii, CSF_new in enumerate(list_pair_ex_space_cut):
                for iCSF_orig,CSF_orig in enumerate(list_ex_space):
                    Helm = Helm_between_CSFs(Enuc,obt,tbt,CSF_orig,CSF_new)
                    Hmat_UCSF[iCSF_orig,len(list_ex_space)+ ii] = Helm
                    Hmat_UCSF[len(list_ex_space)+ ii,iCSF_orig] = Helm
                for jj in range(ii,len(list_pair_ex_space_cut)):
                    CSF_new_j = list_pair_ex_space_cut[jj]
                    Helm = Helm_between_CSFs(Enuc,obt,tbt,CSF_new,CSF_new_j)
                    Hmat_UCSF[len(list_ex_space)+ ii,len(list_ex_space)+ jj] = Helm
                    Hmat_UCSF[len(list_ex_space)+ jj,len(list_ex_space)+ ii] = Helm
            Hmat_UCSF_sparse = csr_matrix(Hmat_UCSF)
            E0_UCSF, psi_GS_UCSF = get_ground_state(Hmat_UCSF_sparse)
            lowering = E0_UCSF-E0_ex
            if lowering > 1e-8:
                print(f'Strange! Positive lowering in select_ia_pairs_for_CSF_E0: {lowering}. Bombing out!')
                sys.exit()
            if abs(lowering) > Ethrsh:
                list_ia_one_CSF.append(ia_pair)
                list_improve.append(lowering)
                print(f'CSF{iCSF},{ia_pair},E lowering: {lowering}')
#sorted_ampld_list, sorted_ia_pair_list,sorted_Ecorr_list  = zip(*sorted(zip(mp2_ampld_list,ia_pair_list,mp2_Ecorr_list)))

        if len(list_improve) != 0:
            sorted_list_improve, sorted_list_ia_one_CSF = zip(*sorted(zip(list_improve,list_ia_one_CSF)))
            print(f'sorted_list_improve: {sorted_list_improve}')
            sorted_list_ia_one_CSF = list(sorted_list_ia_one_CSF)
            list_list_ia_included[iCSF] += sorted_list_ia_one_CSF
            print(f'\nia_pairs included for CSF{iCSF}')
            print(list_list_ia_included[iCSF])

           #Update the list_ex_space with the new ex CSFs of the newly included ia_pairs
           #Very time consuming. 
            list_theta = [0.0] * len(list_list_ia_included[iCSF])
            _, _, list_pair_ex_space, _, list_genmat, = \
              prepare_UCSF_for_one_CSF(CSF,list_list_ia_included[iCSF],list_theta,False)
            print(f'Done preparing list_ex_space for CSF{iCSF}')
           #list_ex_space += list_pair_ex_space[1:]
           #Hmat_ex = construct_Hmat_CSFs(list_ex_space,Enuc,obt,tbt)
           #Hmat_ex_sparse = csr_matrix(Hmat_ex)
           #E0_ex_new, _ = get_ground_state(Hmat_ex_sparse)
           #print(f'\nE0 of ex space with the expansion after handling CSF{iCSF}: {E0_ex_new}, lowering: {E0_ex_new - E0_ex}')
           #print(f'Dimension of updated ex space: {len(list_ex_space)}')
           #E0_ex = E0_ex_new #Set the reference for improvement for the next iCSF
            list_list_genmat.append(list_genmat)
            list_list_ex_space.append(list_pair_ex_space)
        else:
            list_list_genmat.append([])
            list_list_ex_space.append([list_CSF[iCSF]])

       #print(f'\nlist_list_ex_space[{iCSF}]: {list_list_ex_space[iCSF]}')


    return list_list_ex_space, list_list_ia_included, list_list_genmat

def select_ia_pairs_for_CSF_E0_grad(list_CSF,group_ia_pairs,Enuc,obt,tbt,Gthrsh=1.0e-4,debug=False):
    """
    Select ia pairs for each CSF for their individual gradient of E0
    """

    if debug: print('\nIn select_ia_pairs_for_CSF_E0_grad')

    Hmat_CSF = construct_Hmat_CSFs(list_CSF,Enuc,obt,tbt)

    Hmat_CSF_sparse = csr_matrix(Hmat_CSF)
    E0, psi_GS = get_ground_state(Hmat_CSF)
    print(f'\nE0 of original CSF space: {E0}')
    nCSF = len(list_CSF)
    list_list_ia_included = []
    list_list_genmat = []
    list_list_ex_space = []
    print(group_ia_pairs)
    for iCSF, CSF in enumerate(list_CSF):
        print(f'Working on CSF{iCSF}')
        list_ia_one_CSF = []
        list_grad = []
        for ia_pair in group_ia_pairs:
            T_op = FermionOperator()
            for ipair, pair in enumerate(ia_pair):
                T_op += get_Tiiaa_00(pair[1],pair[0])
            T_op = normal_ordered(T_op)
            T_op.compress()
           #print(ia_pair)
           #print(T_op)
            onlist, onidx_list, on_coefs = op_action_tz_remove_0coef(T_op,CSF[0],CSF[1],CSF[2])
            GCSF = [onlist,onidx_list,on_coefs]
            dE0_dtheta = 0.0
            for jCSF, CSFj in enumerate(list_CSF):
                HGelm = Helm_between_CSFs(Enuc,obt,tbt,CSFj,GCSF)
                dE0_dtheta += psi_GS[iCSF]*psi_GS[jCSF]*HGelm

            dE0_dtheta *= 2.0

            if abs(dE0_dtheta) >= Gthrsh:
                list_ia_one_CSF.append(ia_pair)
                list_grad.append(abs(dE0_dtheta))
               #print(f'CSF{iCSF}, {ia_pair}, {dE0_dtheta}')

        if len(list_grad) != 0:
            sorted_list_grad, sorted_list_ia_one_CSF = zip(*sorted(zip(list_grad,list_ia_one_CSF),reverse=True))
            print(f'sorted_list_grad: {sorted_list_grad}')
            sorted_list_ia_one_CSF = list(sorted_list_ia_one_CSF)
            list_list_ia_included.append(sorted_list_ia_one_CSF)
            print(f'\nia_pairs included for CSF{iCSF}')
            print(list_list_ia_included[iCSF])

            list_theta = [0.0] * len(list_list_ia_included[iCSF])
            _, _, list_pair_ex_space, _, list_genmat, = \
              prepare_UCSF_for_one_CSF(CSF,list_list_ia_included[iCSF],list_theta,False)
            print(f'Done preparing list_ex_space for CSF{iCSF}')
            list_list_genmat.append(list_genmat)
            list_list_ex_space.append(list_pair_ex_space)
        else:
            list_list_genmat.append([])
            list_list_ex_space.append([list_CSF[iCSF]])



#   return list_list_ex_space, list_list_ia_included, list_list_genmat

def select_ia_pairs_for_CSF_E0_with_sym_CSF(list_CSF,list_SOMO_DMO,grouped_ia_pairs,Enuc,obt,tbt,l_include_ia_in_CAS,actmo_start,actmo_end,l_axial_sym=False,list_sym_sign=[],Ethrsh=1.0e-3,debug=False):
    """
    Select ia pairs for each CSF for their individual lowerings of the E0 of the CSF space. The CSFs may be paired
    with their symmetry partners
    """

    if debug: print('\nIn select_ia_pairs_for_CSF_E0_with_sym_CSF')

    Hmat_CSF = construct_Hmat_CSFs(list_CSF,Enuc,obt,tbt)
    E0_CSF, psi_GS_CSF = get_ground_state(Hmat_CSF)
    if debug: print(f'E0 of original CSF space: {E0_CSF}')
    nCSF = len(list_CSF)
    list_list_genmat = []
    list_list_ia_pair = []
    list_list_ex_space = []
    list_sym_CSF_vec = []
    n_sym_CSF = 0
    for iCSF in range(nCSF):
        if iCSF > 0 and l_axial_sym and abs(list_sym_sign[iCSF-1]) == 1: continue #This CSF has been considered as sym partner
        list_ia_one_CSF = []
        list_improve = []
        n_sym_CSF += 1
        if l_axial_sym and list_sym_sign[iCSF] != 0:
            print(f'\nSelecting ia pairs for degenerate CSF{iCSF} and CSF{iCSF+1}')
            CSF1 = list_CSF[iCSF]
            CSF2 = list_CSF[iCSF+1]
            for ia_pair in grouped_ia_pairs:
                UCSF1, Uvec1, list_pair_ex_space1, Umat1, _ = \
                  prepare_UCSF_for_one_CSF(CSF1,[ia_pair],[0.0],False)
                UCSF2, Uvec2, list_pair_ex_space2, Umat2, _ = \
                  prepare_UCSF_for_one_CSF(CSF2,[ia_pair],[0.0],False)
                if len(list_pair_ex_space1) != len(list_pair_ex_space2):
                    print(f'Strange, the two list_ex_spaces of sym pairs do not have the same dimension')
                    print(len(list_pair_ex_space1),len(list_pair_ex_space2),ia_pair)
                    print('Bombing out!')
                    sys.exit()
                list_pair_ex_combine, list_corspnd, _ = combine_two_list_ex_spaces(list_pair_ex_space1,list_pair_ex_space2,list_sym_sign[iCSF],False)
               #if len(list_pair_ex_combine) != 2*len(list_pair_ex_space1):
               #    print(f'ia_pair: {ia_pair}')
               #    print('list_pair_ex_space1:')
               #    print(list_pair_ex_space1)
               #    print('list_pair_ex_space2:')
               #    print(list_pair_ex_space2)
               #    print('list_pair_ex_combine:')
               #    print(list_pair_ex_combine)
               #    print(f'list_corspnd: {list_corspnd}')
               #    input("Press Enter to continue...")
               #   #sys.exit()
                if len(list_pair_ex_space1) == 1: continue #No ex CSFs. Jump to next ia_pair
                list_extra_CSF = list_pair_ex_space1[1:] + list_pair_ex_combine[len(list_pair_ex_space1)+1:]
                if len(list_extra_CSF) == 0:
                    print(f'No extra CSFs in excitation. It should not have got here. Bombing out!')
                    sys.exit()
                ndim_with_extra_CSF = nCSF + len(list_extra_CSF)
                Hmat_CSF_extra = np.zeros([ndim_with_extra_CSF,ndim_with_extra_CSF])
                Hmat_CSF_extra[0:nCSF,0:nCSF] = Hmat_CSF
                Hmat_extra = construct_Hmat_CSFs(list_extra_CSF,Enuc,obt,tbt)
                Hmat_CSF_extra[nCSF:,nCSF:] = Hmat_extra
                for iextra,CSFextra in enumerate(list_extra_CSF):
                    for jCSF, CSForig in enumerate(list_CSF):
                        Helm = Helm_between_CSFs(Enuc,obt,tbt,CSForig,CSFextra)
                        Hmat_CSF_extra[nCSF+iextra,jCSF] = Helm
                        Hmat_CSF_extra[jCSF,nCSF+iextra] = Helm
                E0_extra, _ = get_ground_state(Hmat_CSF_extra)
                improve = E0_extra - E0_CSF
                if improve > 1.0e-8:
                    print(f'Strange! E0 is larger with the extra CSFs by {improve}. Bombing out!')
                    sys.exit()
                l_ia_in_AS = True
                for pair in ia_pair:
                    if pair[0] < actmo_start or pair[1] > actmo_end: l_ia_in_AS = False
                if abs(improve) > Ethrsh or (l_include_ia_in_CAS and l_ia_in_AS):
                    list_improve.append(improve)
                    list_ia_one_CSF.append(ia_pair) 
                    if l_include_ia_in_CAS and l_ia_in_AS: 
                        print(f'{ia_pair} included because being in active space')
        else:
            print(f'\nSelecting ia pairs for CSF{iCSF}')
            CSF = list_CSF[iCSF]
            for ia_pair in grouped_ia_pairs:
                l_ia_in_AS = True
                for pair in ia_pair:
                    if pair[0] < actmo_start or pair[1] > actmo_end: l_ia_in_AS = False
                UCSF, Uvec, list_pair_ex_space, Umat, _ = \
                  prepare_UCSF_for_one_CSF(CSF,[ia_pair],[0.0],False)
                if len(list_pair_ex_space) == 1: continue #No ex CSFs. Jump to next ia_pair
                list_extra_CSF = list_pair_ex_space[1:]
                ndim_with_extra_CSF = nCSF + len(list_extra_CSF)
                Hmat_CSF_extra = np.zeros([ndim_with_extra_CSF,ndim_with_extra_CSF])
                Hmat_CSF_extra[0:nCSF,0:nCSF] = Hmat_CSF
                Hmat_extra = construct_Hmat_CSFs(list_extra_CSF,Enuc,obt,tbt)
                Hmat_CSF_extra[nCSF:,nCSF:] = Hmat_extra
                for iextra,CSFextra in enumerate(list_extra_CSF):
                    for jCSF, CSForig in enumerate(list_CSF):
                        Helm = Helm_between_CSFs(Enuc,obt,tbt,CSForig,CSFextra)
                        Hmat_CSF_extra[nCSF+iextra,jCSF] = Helm
                        Hmat_CSF_extra[jCSF,nCSF+iextra] = Helm
                E0_extra, _ = get_ground_state(Hmat_CSF_extra)
                improve = E0_extra - E0_CSF
                if improve > 1.0e-8:
                    print(f'Strange! E0 is larger with the extra CSFs by {improve}. Bombing out!')
                    sys.exit()
                if abs(improve) > Ethrsh or (l_include_ia_in_CAS and l_ia_in_AS):
                    list_improve.append(improve)
                    list_ia_one_CSF.append(ia_pair)
                    if l_include_ia_in_CAS and l_ia_in_AS:
                        print(f'{ia_pair} included because being in active space')
                

        if len(list_ia_one_CSF) != 0: 
            sorted_list_improve, sorted_list_ia_one_CSF = zip(*sorted(zip(list_improve,list_ia_one_CSF)))
            sorted_list_ia_one_CSF = list(sorted_list_ia_one_CSF)
            print('ia_pairs included:')
            for ii, ia_pairs in enumerate(sorted_list_ia_one_CSF):
                print(ia_pairs ,sorted_list_improve[ii])
        else:
            sorted_list_ia_one_CSF = []

        list_zero_theta = [0.0] * len(sorted_list_ia_one_CSF)
        if l_axial_sym and list_sym_sign[iCSF] != 0:
            print(f'Making list_genmat for degenerate CSF{iCSF} and CSF{iCSF+1}')
            CSF1 = list_CSF[iCSF]
            CSF2 = list_CSF[iCSF+1]
            SOMO_DMO1 = list_SOMO_DMO[iCSF]
            SOMO_DMO2 = list_SOMO_DMO[iCSF+1]
            list_regroupped_ia, list_genmat, list_pair_ex_combine, sym_CSF_vec = make_list_genmat_for_sym_CSF(CSF1,CSF2,SOMO_DMO1,SOMO_DMO2,list_sym_sign[iCSF],sorted_list_ia_one_CSF,list_zero_theta,False)
           #sys.exit()
           #UCSF1, Uvec1, list_pair_ex_space1, Umat1, list_genmat1 = \
           #  prepare_UCSF_for_one_CSF(CSF1,sorted_list_ia_one_CSF,list_zero_theta,False)
           #UCSF2, Uvec2, list_pair_ex_space2, Umat2, list_genmat2 = \
           #  prepare_UCSF_for_one_CSF(CSF2,sorted_list_ia_one_CSF,list_zero_theta,False)
           #if len(list_pair_ex_space1) != len(list_pair_ex_space2):
           #    print(f'Strange, the two list_ex_spaces of sym pairs do not have the same dimension')
           #    print(len(list_pair_ex_space1),len(list_pair_ex_space2),ia_pair)
           #    print('Bombing out!')
           #    sys.exit()
           #list_pair_ex_combine, list_corspnd, sym_CSF_vec = combine_two_list_ex_spaces(list_pair_ex_space1,list_pair_ex_space2,list_sym_sign[iCSF],False)
           #list_genmat = combine_list_genmat(list_genmat1,list_genmat2,list_corspnd)
           #check_combined_list_genmat(CSF1,CSF2,list_sym_sign[iCSF],list_pair_ex_combine,list_genmat,sym_CSF_vec,sorted_list_ia_one_CSF)
            list_list_ex_space.append(list_pair_ex_combine)
            list_list_ia_pair.append(list_regroupped_ia)
        else:
            print(f'Making list_genmat for CSF{iCSF}')
            CSF = list_CSF[iCSF]
           #If [[6,8],[5,7]] in list_ia_one_CSF, [[6,7],[5,8]] shall also be included
           #suppl_list_ia_one_CSF = supplement_pipi_pair_ex(sorted_list_ia_one_CSF,grouped_ia_pairs,False)
           #sorted_list_ia_one_CSF = copy.deepcopy(suppl_list_ia_one_CSF)
            UCSF, Uvec, list_pair_ex_space, Umat, list_genmat = \
              prepare_UCSF_for_one_CSF(CSF,sorted_list_ia_one_CSF,list_zero_theta,False)
            list_list_ex_space.append(list_pair_ex_space)
            list_list_ia_pair.append(sorted_list_ia_one_CSF)
            sym_CSF_vec = np.zeros(len(list_pair_ex_space))
            sym_CSF_vec[0] = 1.0

        list_list_genmat.append(list_genmat)
        list_sym_CSF_vec.append(sym_CSF_vec)
       #print('list_genmat:')
       #for genmat in list_genmat:
       #    print(genmat)
        print(f'Updated length: {len(list_list_genmat)}')

    assert len(list_list_ex_space) == len(list_list_ia_pair)
    assert len(list_list_ia_pair) == len(list_list_genmat)
    assert len(list_list_genmat) == len(list_sym_CSF_vec)
    print(f'Number of sym-adapted CSF basis: {len(list_list_ex_space)}')


    return list_list_ex_space, list_list_ia_pair, list_list_genmat, list_sym_CSF_vec

def supplement_pipi_pair_ex(list_selected_ia,grouped_ia_pairs,debug=False):
    """
    if 5,6 is a pi set, 7, 8 is a pi set, (5->7,6->8) is selected, but (5->8,6->7) is not selected,
    supplemented (5->8,6->7). This is because the (5,6) empty and (7,8) fully occupied can come
    from both routes.
    """
    if debug: print('\nIn supplement_pipi_pair_ex')

    list_pair_to_add = []
    for ia_pair in list_selected_ia:
        if len(ia_pair) == 2:
            [[dmo1,vmo1],[dmo2,vmo2]] = ia_pair
            if dmo1 == dmo2 or vmo1 == vmo2: continue
            if [[dmo1,vmo2],[dmo2,vmo1]] in list_selected_ia or [[dmo2,vmo1],[dmo1,vmo2]] in list_selected_ia: continue
            if [[dmo1,vmo2],[dmo2,vmo1]] in grouped_ia_pairs:
                list_pair_to_add.append([[dmo1,vmo2],[dmo2,vmo1]])
            elif [[dmo2,vmo1],[dmo1,vmo2]] in grouped_ia_pairs:
                list_pair_to_add.append([[dmo2,vmo1],[dmo1,vmo2]])
            else:
                print(f'Strange! {ia_pair} in but {[[dmo2,vmo1],[dmo1,vmo2]]} \
                  or {[[dmo1,vmo2],[dmo2,vmo1]]} not in total list of pairs')
                sys.exit()

    if len(list_pair_to_add) != 0:
        print(f'The following pairs are to be added')
        print(list_pair_to_add)
        supp_list_selected_ia = list_selected_ia + list_pair_to_add
        print(f'Supplemented list of selected pairs: {list_selected_ia}')
        input("Press Enter to continue...")
    else:
        supp_list_selected_ia = list_selected_ia

    return supp_list_selected_ia

def make_list_genmat_for_sym_CSF(CSF1,CSF2,SOMO_DMO1,SOMO_DMO2,isym_sign,grouped_ia_pairs,list_theta,debug=False):
    """
    Prepare and combine list_genmats for symmetrized CSF of two sym partners CSF1 and 2.
    """

    if debug: print('\nIn make_list_genmat_for_sym_CSF')

    ungroupped_ia_pair = []
    for pairs in grouped_ia_pairs:
        for pair in pairs:
            ungroupped_ia_pair.append(pair)

   #print(f'ungroupped_ia_pair: {ungroupped_ia_pair}')
    [onlist,onidx_list,coefvec] = CSF1
    list_pair_ex_space1, list_pair_ex_comb1 = make_pair_ex_space_fast(onlist,onidx_list,coefvec,ungroupped_ia_pair)
   #print('list_pair_ex_comb1:')
   #print(list_pair_ex_comb1)
   #print('list_pair_ex_space1:')
   #print(list_pair_ex_space1)
    [onlist,onidx_list,coefvec] = CSF2
    list_pair_ex_space2, list_pair_ex_comb2 = make_pair_ex_space_fast(onlist,onidx_list,coefvec,ungroupped_ia_pair)
   #print('list_pair_ex_space2:')
   #print(list_pair_ex_space2)
   #print('list_pair_ex_comb2:')
   #print(list_pair_ex_comb2)

    list_pair_ex_combine, list_corspnd, sym_CSF_vec = combine_two_list_ex_spaces(list_pair_ex_space1,list_pair_ex_space2,isym_sign,False)
    ndim = max(list_corspnd) + 1

   #print(f'list_corspnd: {list_corspnd}')
   #print(f'sym_CSF_vec: {sym_CSF_vec}')

    list_ungroupped_ia_pair = []
    for ia_pair in ungroupped_ia_pair:
        list_ungroupped_ia_pair.append([ia_pair])

    list_iapair_st_pairs1 = make_Hmat_diagonal_space(None,None,None,list_pair_ex_space1,list_pair_ex_comb1,False,False)
    list_iapair_st_pairs2 = make_Hmat_diagonal_space(None,None,None,list_pair_ex_space2,list_pair_ex_comb2,False,False)
    list_genmat1 = make_iapair_genmat_fast(list_ungroupped_ia_pair,list_pair_ex_space1,list_iapair_st_pairs1,False)
    list_genmat2 = make_iapair_genmat_fast(list_ungroupped_ia_pair,list_pair_ex_space2,list_iapair_st_pairs2,False)
   #print('list_iapair_st_pairs1:')
   #print(list_iapair_st_pairs1)
   #print('list_iapair_st_pairs2:')
   #print(list_iapair_st_pairs2)
   #print('list_genmat1:')
   #for genmat1 in list_genmat1:
   #    print(genmat1)
   #print('list_genmat2:')
   #for genmat2 in list_genmat2:
   #    print(genmat2)

    list_regroupped_ia_pairs = []

    list_genmat = []
    for ia_pair in grouped_ia_pairs:
        print(f'ia_pair in group: {ia_pair}, {len(ia_pair)}')
        if len(ia_pair) == 1:
           #print(ungroupped_ia_pair.index(ia_pair[0]),ia_pair)
            [pair] = ia_pair
            ind_ungroupped = ungroupped_ia_pair.index(pair)
            genmat1 = list_genmat1[ind_ungroupped]
            genmat2 = list_genmat2[ind_ungroupped]
            genmat = combine_genmats(genmat1,genmat2,list_corspnd,ndim)
            list_regroupped_ia_pairs.append(ia_pair)
            list_genmat.append(genmat)
        elif len(ia_pair) == 2:
            [[dmo1,vmo1],[dmo2,vmo2]] = ia_pair
           #print(dmo1,vmo1,dmo2,vmo2)
           #print(ungroupped_ia_pair.index([dmo1,vmo1]))
           #print(ungroupped_ia_pair.index([dmo2,vmo2]))
            genmat1_1 = list_genmat1[ungroupped_ia_pair.index([dmo1,vmo1])]
            genmat1_2 = list_genmat1[ungroupped_ia_pair.index([dmo2,vmo2])]
            genmat2_1 = list_genmat2[ungroupped_ia_pair.index([dmo1,vmo1])]
            genmat2_2 = list_genmat2[ungroupped_ia_pair.index([dmo2,vmo2])]
            norm1_1 = scipy.sparse.linalg.norm(genmat1_1)
            norm1_2 = scipy.sparse.linalg.norm(genmat1_2)
            norm2_1 = scipy.sparse.linalg.norm(genmat2_1)
            norm2_2 = scipy.sparse.linalg.norm(genmat2_2)
           #print(ia_pair)
           #print('genmat1_1:')
           #print(genmat1_1)
           #print('genmat1_2:')
           #print(genmat1_2)
           #print('genmat2_1:')
           #print(genmat2_1)
           #print('genmat2_2:')
           #print(genmat2_2)
            assert norm1_1 == norm2_2 and norm1_2 == norm2_1
            if np.isclose(norm1_1,0.0) and np.isclose(norm1_2,0.0) and np.isclose(norm2_1,0.0) and np.isclose(norm2_2,0.0):
               #print('Strange: both norm1_1 and norm1_2 = 0. If so, the ia_pair shall not be included')
                print('Strange: all four norms = 0. If so, the ia_pair shall not be included')
                print(ia_pair,SOMO_DMO1,SOMO_DMO2)
                print('Bombing out!')
                sys.exit()
            genmat_11_22 = combine_genmats(genmat1_1,genmat2_2,list_corspnd,ndim)
            genmat_12_21 = combine_genmats(genmat1_2,genmat2_1,list_corspnd,ndim)

            if not np.isclose(scipy.sparse.linalg.norm(genmat_11_22),0.0):
                list_regroupped_ia_pairs.append(ia_pair)
                list_genmat.append(genmat_11_22)
            if not np.isclose(scipy.sparse.linalg.norm(genmat_12_21),0.0):
                list_regroupped_ia_pairs.append(ia_pair)
                list_genmat.append(genmat_12_21)
                 
                
        elif len(ia_pair) == 4:
            print(f'Not yet coded for group pairs length = 4. Bombing out!')
            sys.exit()
        else:
            print(f'Length of pairs shall be within 1, 2, 4 only. {ia_pair}. Bombing out!')
            sys.exit()

    if debug:
        print(f'regroupped ia pairs: {list_regroupped_ia_pairs}')
        print('combined list_genmat:')
        for genmat in list_genmat:
            print(genmat)

    return list_regroupped_ia_pairs, list_genmat, list_pair_ex_combine, sym_CSF_vec

def combine_list_genmat(list_genmat1,list_genmat2,list_corspnd):
    """
    Combine the genmats of the two sym-paired CSFs.
    """

    ndim = max(list_corspnd) + 1
   #print(f'ndim in combine_list_genmat: {ndim}')
    assert len(list_genmat1) == len(list_genmat2)
    list_genmat = []
    for ii in range(len(list_genmat1)):
        genmat = combine_genmats(list_genmat1[ii],list_genmat2[ii],list_corspnd,ndim)
        list_genmat.append(genmat)

    return list_genmat

def combine_genmats(genmat1,genmat2,list_corspnd,ndim):
    """
    Combine two genmats
    """

    genmat = csr_matrix((ndim,ndim))
    ndim1 = genmat1.shape[0]
    assert ndim1 < ndim
   #print(genmat.shape,genmat1.shape)
    genmat[0:ndim1,0:ndim1] = genmat1

    row, col = genmat2.nonzero()
   #print(row)
   #print(col)
    for ii in range(len(row)):
       irow = row[ii]
       icol = col[ii]
       irow_match = list_corspnd[irow]
       icol_match = list_corspnd[icol]
       genmat[irow_match,icol_match] = genmat2[irow,icol]

   #Makre sure genmat is antisymmetric
    assert np.isclose(scipy.sparse.linalg.norm(genmat + genmat.transpose()),0.0)

    return genmat
        
def combine_two_list_ex_spaces(list_ex_space1,list_ex_space2,isign,debug=False):
    """
    Combine two list_ex_spaces to one and also output the indices correlation matrix
    """

    if debug: print('\nIn combine_two_list_ex_spaces')

    list_ex_space = copy.deepcopy(list_ex_space1)
    list_corspnd = []
    for iCSF2, CSF in enumerate(list_ex_space2):
        lfound = False
        for ii, CSF1 in enumerate(list_ex_space1):
            Selm = overlap_CSFs(CSF,CSF1)
            if not np.isclose(Selm,0.0):
                if not np.isclose(abs(Selm),1.0):
                    print('Strange! Fractional overlap is detected. Bombing out!')
                    print('CSF:')
                    print(CSF)
                    print('CSF1:')
                    print(CSF1)
                    sys.exit()
                if np.isclose(Selm,-1.0):
                    print('Strange! -1 overlap detected. Bombing out!')
                    sys.exit()
                lfound = True
                if iCSF2 == 0 and lfound:
                    print(f'The 0th CSF in list_ex_space2 is found in list_ex_space2. This is unreasonable. Bombing out!')
                    sys.exit()
                list_corspnd.append(ii)
                break
        if not lfound:
            list_ex_space.append(CSF)
            list_corspnd.append(len(list_ex_space)-1)

    sym_CSF_vec = np.zeros(len(list_ex_space))

    sym_CSF_vec[0] = np.sqrt(0.5)
    sym_CSF_vec[len(list_ex_space1)] = np.sqrt(0.5)
    if isign < 0: sym_CSF_vec[len(list_ex_space1)] = -np.sqrt(0.5)

    return list_ex_space,list_corspnd, sym_CSF_vec

def check_combined_list_genmat(CSF1,CSF2,isign,list_pair_ex_combine,list_genmat,sym_CSF_vec,list_ia_pair):
    """
    Check the correctness of the combined list_genmat from individual list_genmat
    """
    print('\nIn check_combined_list_genmat')        

    coefs = np.array([np.sqrt(0.5),np.sqrt(0.5)])
    if isign < 0: coefs[1] = -coefs[1]

    comb_CSF = LC_CSFs([CSF1,CSF2],coefs,l_check_norm=True)

    print(len(list_genmat),len(list_ia_pair))
    for ipair, ia_pair in enumerate(list_ia_pair):
        T_op = FermionOperator()
        for [ii,aa] in ia_pair:
            T_op += get_Tiiaa_00(ii,aa)

        T_op = normal_ordered(T_op)
        T_op.compress()
        onlist, onidx_list, on_coefs = op_action_tz_remove_0coef(T_op,comb_CSF[0],comb_CSF[1],comb_CSF[2])
        T_op_CSF = [onlist, onidx_list, on_coefs]
        T_op_vec = list_genmat[ipair]@sym_CSF_vec
        T_op_vec_CSF = LC_CSFs(list_pair_ex_combine,T_op_vec)
        norm_T_op_CSF = np.linalg.norm(T_op_CSF[2])
        norm_T_op_vec_CSF = np.linalg.norm(T_op_vec_CSF[2])
        if np.isclose(norm_T_op_CSF,0.0) and np.isclose(norm_T_op_vec_CSF,0.0):
            print('Good! Both result in null CSF, as expected. Test passed.')
            continue
        assert np.isclose(norm_T_op_CSF,norm_T_op_vec_CSF)
        Selm = overlap_CSFs(T_op_CSF,T_op_vec_CSF)
        Selm /= norm_T_op_CSF*norm_T_op_vec_CSF
        print(f'Selm: {Selm}')
        if not np.isclose(Selm,1.0):
            print(f'ia_pair: {ia_pair}')
            print('CSF1:')
            print(CSF1)
            print('CSF2:')
            print(CSF2)
            print('comb_CSF:')
            print(comb_CSF)
            print('T_op_CSF')
            print(T_op_CSF)
            print('sym_CSF_vec:')
            print(sym_CSF_vec)
            print('T_op_vec:')
            print(T_op_vec)
            print('genmat:')
            print(list_genmat[ipair])
            print('T_op_vec:')
            print(T_op_vec)
            sys.exit()
        
def pick_pairex_within_CAS(list_list_pair_ex_space,list_sym_CSF_vec,actmo_start,actmo_end,l_axial_sym,Enuc,obt,tbt,debug=False):
    """
    Within each pair_ex_space, pick out the states that are obtained from the 0th CSF
    by pair excitation within CAS
    """

    if debug: print('\nIn pick_pairex_within_CAS')

    
    list_list_pairex_CSF_vec = []
    list_list_pairex_CSF = []
    n_total_CSF_in_CAS = 0
    for irefCSF, list_pair_ex_space in enumerate(list_list_pair_ex_space):
        sym_CSF_vec = list_sym_CSF_vec[irefCSF]
        ind_ref_CSF = np.where(sym_CSF_vec != 0.0)[0]
        ind_ex_CSF = np.where(sym_CSF_vec == 0.0)[0]
        list_set_dmo_ref = []
        for ii in ind_ref_CSF:
            list_set_dmo_ref.append(set(dmo_in_SD(list_pair_ex_space[ii][0][0])))

       #print(ii,ind_ref_CSF,ind_ex_CSF)
        print(f'\n{ii,list_set_dmo_ref}')
        ind_ex_from_ref = []
        for iex in ind_ex_CSF:
            set_dmo_exCSF = set(dmo_in_SD(list_pair_ex_space[iex][0][0]))
            for jref, set_dmo_ref in enumerate(list_set_dmo_ref):
                dmo_diff = set_dmo_ref^set_dmo_exCSF
               #print(iex,jref,set_dmo_ref,set_dmo_exCSF,dmo_diff)
                l_within_CAS = True
                for mo in dmo_diff:
                    if mo > actmo_end or mo < actmo_start:
                        l_within_CAS = False
                        break
                if not l_within_CAS: continue
                if iex in ind_ex_from_ref: continue
               #print('This CSF is connected to ref CSF(s) by pair excitation within CAS')
                ind_ex_from_ref.append(iex)

        list_pairex_CSF_vec = []
        ndim = len(list_pair_ex_space)
        vec0 = np.zeros(ndim)
        if len(ind_ex_from_ref) != 0:
            print('\nThe following CSFs are obtained from ref CSF(s) by pair excitations within CAS')        
            list_exCSF_E = []
            for iex in ind_ex_from_ref:
                E_CSF = Helm_between_CSFs(Enuc,obt,tbt,list_pair_ex_space[iex],list_pair_ex_space[iex])
                list_exCSF_E.append(E_CSF)
                print(iex,dmo_in_SD(list_pair_ex_space[iex][0][0]),E_CSF)

#zip(*sorted(zip(list_improve,list_ia_one_CSF)))

            if l_axial_sym:
                list_refCSFs = []
                coefs_ref = []
                for iref in ind_ref_CSF:
                    list_refCSFs.append(list_pair_ex_space[iref])
                    coefs_ref.append(sym_CSF_vec[iref])
    
                coefs_ref = np.array(coefs_ref)
                sym_ref_CSF = LC_CSFs(list_refCSFs,coefs_ref)
               #sorted_ex_CSF_E, sorted_ind_ex = zip(*sorted(zip(list_exCSF_E,ind_ex_from_ref)))
               #sorted_ex_CSF_E = list(sorted_ex_CSF_E)
               #sorted_ind_ex = list(sorted_ind_ex)
               #print('\nE-sorted exCSFs:')
               #for iex_ind in range(len(sorted_ind_ex)):
               #    print(sorted_ind_ex[iex_ind],sorted_ex_CSF_E[iex_ind])
                list_ref_ex_CSFs = [sym_ref_CSF]
                for iex in ind_ex_from_ref:
                    list_ref_ex_CSFs.append(list_pair_ex_space[iex])

                Hmat = construct_Hmat_CSFs(list_ref_ex_CSFs,Enuc,obt,tbt)
                print('\nHamiltonian matrix of ref CSF and those obtained from pair excitations\n')
                print_matrix(Hmat)
               #Hmat_sparse = csr_matrix(Hmat)
               #_, psi_GS = get_ground_state(Hmat_sparse)
               #print(f'\nGround state eigenvector: {psi_GS}')
                degen_thrsh = 1.0e-7
                list_pairing = []
                for iex_bas in range(1,Hmat.shape[0]-1):
                    iex = iex_bas - 1
                    l_paired = False
                    for item in list_pairing:
                        if iex in item:
                            l_paired = True
                            break
                    if l_paired: continue
                    l_pairing = False
                    for jex_bas in range(iex_bas+1,Hmat.shape[0]):
                        jex = jex_bas - 1
                        if abs(Hmat[iex_bas,iex_bas] - Hmat[jex_bas,jex_bas]) < degen_thrsh:
                            list_pairing.append([iex,jex])
                            l_pairing = True
                            break
                    if not l_pairing: list_pairing.append([iex])
                iex = len(ind_ex_from_ref)-1
                l_paired = False
                for item in list_pairing:
                    if iex in item: l_paired = True
                if not l_paired: list_pairing.append([iex])
                print('\nPairing of the exCSFs based on their degeneracies:')    
                print(list_pairing)

                for item in list_pairing:
                    if len(item) == 1:
                        iex_ind = ind_ex_from_ref[item[0]]
                        ex_CSF_vec = copy.deepcopy(vec0)
                        ex_CSF_vec[iex_ind] = 1.0
                       #print(f'Adding 1.0 to {iex_ind}')
                    elif len(item) == 2:
                        [iex,jex] = item
                        iex_bas,jex_bas = iex+1, jex+1
                       #isign = round(np.sign(psi_GS[iex_bas]) * np.sign(psi_GS[jex_bas]))
                        assert np.isclose(Hmat[0,iex_bas],Hmat[0,jex_bas])
                        isign = round(np.sign(Hmat[0,iex_bas]) * np.sign(Hmat[0,jex_bas]))
                        iex_ind = ind_ex_from_ref[iex]
                        jex_ind = ind_ex_from_ref[jex]
                        ex_CSF_vec = copy.deepcopy(vec0)
                        ex_CSF_vec[iex_ind], ex_CSF_vec[jex_ind] = np.sqrt(0.5), np.sqrt(0.5)
                        if isign < 0: ex_CSF_vec[jex_ind] *= -1.0
                       #print(f'Adding sqrt(0.5) to {iex_ind,jex_ind}, {ex_CSF_vec[iex_ind],ex_CSF_vec[jex_ind]}')
                    else:
                        print(f'So far, only double degeneracy is supported. {item}. Bombing out!')
                        sys.exit()

                    list_pairex_CSF_vec.append(ex_CSF_vec)

                print('\nCollected ex_CSF_vec:')
                for item in list_pairex_CSF_vec: print(csr_matrix(item))
                        

            else:
                for iex in ind_ex_from_ref:
                    ex_CSF_vec = copy.deepcopy(vec0)
                    ex_CSF_vec[iex] = 1.0
                    list_pairex_CSF_vec.append(ex_CSF_vec)
                
                print('\nCollected ex_CSF_vec:')
                for item in list_pairex_CSF_vec: print(csr_matrix(item))

        list_pairex_CSF_vec = [sym_CSF_vec] + list_pairex_CSF_vec
        list_list_pairex_CSF_vec.append(list_pairex_CSF_vec)
       #Get the true pairex CSFs for future use, not just the vectors
        list_pairex_CSF = []
        for CSFvec in list_pairex_CSF_vec:
            assert len(CSFvec) == len(list_pair_ex_space)
            list_pairex_CSF.append(LC_CSFs(list_pair_ex_space,CSFvec))

        assert len(list_pairex_CSF) == len(list_pairex_CSF_vec)
        n_total_CSF_in_CAS += len(list_pairex_CSF)

        list_list_pairex_CSF.append(list_pairex_CSF)
        print(f'Dimension reduction for CSF Group {irefCSF} from {len(list_pair_ex_space)} to {len(list_pairex_CSF)}')

    assert len(list_list_pairex_CSF_vec) == len(list_list_pairex_CSF)
    print(f'\nTotal number of CSFs in CAS leftover {n_total_CSF_in_CAS}')
            
    return list_list_pairex_CSF_vec, list_list_pairex_CSF

def remove_pairex_in_CAS(actmo_start,actmo_end,list_list_ia,list_list_genmat,list_list_theta,l_use_decomp_genmat=False,list_list_decomp_genmat=[]):
    """
    Remove the ia pair between active orbitals
    """

    print('\nIn remove_pairex_in_CAS')

    assert len(list_list_ia) == len(list_list_genmat)
    assert len(list_list_genmat) == len(list_list_theta)
    if l_use_decomp_genmat: assert len(list_list_theta) == len(list_list_decomp_genmat)

    list_list_ia_noCAS = []
    list_list_genmat_noCAS = []
    list_list_theta_noCAS = []
    list_list_decomp_genmat_noCAS = []
    for iref_CSF in range(len(list_list_ia)):
        print(f'Removing inCAS Ex for CSF group {iref_CSF}')
        list_ia_noCAS = copy.deepcopy(list_list_ia[iref_CSF])
        list_genmat_noCAS = copy.deepcopy(list_list_genmat[iref_CSF])
        list_theta_noCAS = copy.deepcopy(list_list_theta[iref_CSF])
        if l_use_decomp_genmat: list_decomp_genmat_noCAS = copy.deepcopy(list_list_decomp_genmat[iref_CSF])

        list_l_remove = [False] * len(list_ia_noCAS)
        for ipairs in range(len(list_ia_noCAS)):
            pairs = list_ia_noCAS[ipairs]
            for pair in pairs:
                if pair[0] >= actmo_start and pair[0] <= actmo_end and \
                   pair[1] >= actmo_start and pair[1] <= actmo_end:
                    list_l_remove[ipairs] = True
                    break
            if list_l_remove[ipairs]: print(f'To remove {pairs}')

        for  ipairs in range(len(list_ia_noCAS)-1,-1,-1):
            if list_l_remove[ipairs]:
                del list_ia_noCAS[ipairs], list_genmat_noCAS[ipairs], list_theta_noCAS[ipairs]
                if l_use_decomp_genmat: del list_decomp_genmat_noCAS[ipairs]
                        
        print(f'Resulting external pair ex for CSF group {iref_CSF}: {list_ia_noCAS}')
        assert len(list_ia_noCAS) == len(list_genmat_noCAS)
        assert len(list_ia_noCAS) == len(list_theta_noCAS)
        if l_use_decomp_genmat: assert len(list_ia_noCAS) == len(list_decomp_genmat_noCAS)
        list_list_ia_noCAS.append(list_ia_noCAS)
        list_list_genmat_noCAS.append(list_genmat_noCAS)
        list_list_theta_noCAS.append(list_theta_noCAS)
        if l_use_decomp_genmat: list_list_decomp_genmat_noCAS.append(list_decomp_genmat_noCAS)


    return list_list_ia_noCAS, list_list_genmat_noCAS, list_list_theta_noCAS, list_list_decomp_genmat_noCAS

def LC_CSFs(list_CSF,coefs,l_check_norm=False):
    """
    Linearly combine a list of CSFs using the input coefficients and return the resultant CSF
    """

   #print(coefs)

    assert len(coefs) == len(list_CSF)

    onlist_res = []
    idx_list_res = []
    coefs_res = []
    for ii in range(len(coefs)):
        CSF = copy.deepcopy(list_CSF[ii])
        CSF[2] *= coefs[ii]
        onlist_res += CSF[0]
        idx_list_res += CSF[1]
        coefs_res += CSF[2].tolist()

    coefs_res = np.array(coefs_res)
    res_CSF = [onlist_res,idx_list_res,coefs_res]
   #Check normality
    if l_check_norm:
        Selm = overlap_CSFs(res_CSF,res_CSF)
        assert np.isclose(Selm,1.0)

    return res_CSF

def groupping_list_list_genmat(list_list_genmat,debug=False):
    """
    Decompose each genmat to normalized components, C, which satisfies C^3 = -C
    """

    import random
    if debug: print('\nIn groupping_list_list_genmat')

    list_list_decomp_genmat = []
    for list_genmat in list_list_genmat:
        list_decomp_genmat = []
        for genmat in list_genmat:
            ndim = genmat.shape[0]
            genmat_sq = genmat@genmat
            eigval, eigvec = np.linalg.eigh(genmat_sq.toarray())
            uniq_eigval = [eigval[0]]
            for ival in range(1,len(eigval)):
                if not np.isclose(eigval[ival],eigval[ival-1]):
                    uniq_eigval.append(eigval[ival])
           #print(f'unique eigvalues of genmat^2: {uniq_eigval}')
            list_eigvec = []
            list_start_end_ind = []
            list_prj_genmat = []
            list_unique_eigval = []
            for val in uniq_eigval:
                collist = np.where(abs(eigval - val) < 1.0e-6)[0]
               #print(f'collist for eigval {val}: {collist}')
                istart, iend = collist[0],collist[-1]
               #prjmat = eigvec[:,istart:iend+1]@eigvec[:,istart:iend+1].transpose()
               #prjmat = csr_matrix(prjmat)
                prj_genmat = eigvec[:,istart:iend+1].transpose()@genmat@eigvec[:,istart:iend+1]
                list_start_end_ind.append([istart,iend])
                list_eigvec.append(copy.deepcopy(eigvec[:,istart:iend+1]))
                list_unique_eigval.append(val)
                if not np.isclose(val,0.0): prj_genmat = prj_genmat / np.sqrt(-val)
                prj_genmat = csr_matrix(prj_genmat)
                list_prj_genmat.append(prj_genmat)
                prj_genmat_sq = prj_genmat@prj_genmat
               #print(f'prj_genmat_sq')
               #print_matrix(prj_genmat_sq.toarray())
                eigval_prj, _ = np.linalg.eigh(prj_genmat_sq.toarray())
                uniq_prjeigval = list(set(eigval_prj))
                if debug: print(f'unique eigvalues of prj_genmat^2: {uniq_prjeigval}')

            if debug:
                theta = random.uniform(0,1)
                tic = time.perf_counter()
                exp_genmat = scipy.linalg.expm(theta*genmat.toarray())
                toc = time.perf_counter()
                time_for_num_exp = toc - tic
               #print('\nNumerical exp(theta*G)')
               #print_matrix(exp_genmat)
                exp_genmat = csr_matrix(exp_genmat)
    
                tic = time.perf_counter()
                exp_genmat_anl = csr_matrix((ndim,ndim))
                for ival, unival in enumerate(list_unique_eigval):
                    [istart, iend] = list_start_end_ind[ival]
                    sub_eigvec = eigvec[:,istart:iend+1]
                    sub_eigvec = csr_matrix(sub_eigvec)
                    prj_genmat = list_prj_genmat[ival]
                    if np.isclose(unival,0.0):
                        assert np.isclose(scipy.sparse.linalg.norm(prj_genmat),0.0)
                        exp_genmat_anl += sub_eigvec @ sub_eigvec.transpose()
                    else:
                        ndim_sub = prj_genmat.shape[0]
                        sub_expgenmat = scipy.sparse.identity(ndim_sub,format="csr")
                        sub_expgenmat += prj_genmat*np.sin(np.sqrt(-unival)*theta)
                        sub_expgenmat -= prj_genmat@prj_genmat*(np.cos(np.sqrt(-unival)*theta)-1.0)
                        sub_expgenmat_numrc = scipy.linalg.expm(np.sqrt(-unival)*theta*prj_genmat.toarray())
                        exp_genmat_anl += sub_eigvec@sub_expgenmat@sub_eigvec.transpose()
                        
               #print('\nAnalytical exp(theta*G)')
               #print_matrix(exp_genmat_anl.toarray())
                toc = time.perf_counter()
                time_for_anl_exp = toc - tic
                assert np.isclose(scipy.sparse.linalg.norm(exp_genmat_anl - exp_genmat),0.0)
                print(f'Time for numerical  expm: {time_for_num_exp}')
                print(f'Time for analytical expm: {time_for_anl_exp}')

            list_decomp_genmat.append([list_unique_eigval,list_prj_genmat,eigvec,list_start_end_ind])

        list_list_decomp_genmat.append(list_decomp_genmat)

    return list_list_decomp_genmat

def select_ia_pairs_for_CSF_E0_with_pairex_in_actmo(list_CSF,class_somo_ind,group_iapair,actmo_start,actmo_end,Enuc,obt,tbt,Ethrsh=1.e-3,debug=False):
    """
    Select ia pairs based on their possibility to decrease E0. The same class of CSFs
    have to share the same set of ia pairs.
    """

    import random
    if debug: print('\nIn select_ia_pairs_for_CSF_E0_with_pairex_in_actmo')
    nelec = len(np.where(list_CSF[0][0][0] == 1.0)[0])
    homo = nelec // 2 - 1
    print(f'homo: {homo}')
    list_pair_ex_in_actmo = []
    for ii in range(actmo_start,homo+1):
        for aa in range(homo+1,actmo_end+1):
            list_pair_ex_in_actmo.append([[ii,aa]])

    print(f'\nSelecting ia pair based on Energy lowering Thrshold {Ethrsh}\n')
    list_theta_0 = [0.0]*len(list_pair_ex_in_actmo)

    list_set_dmo = []
    for iCSF, CSF in enumerate(list_CSF):
        list_dmo = dmo_in_SD(CSF[0][0])
        set_dmo = set(list_dmo)
        print(f'DMOs of CSF{iCSF}: {set_dmo}')
        list_set_dmo.append(set_dmo)
     

    print(f'list_pair_ex_in_actmo: {list_pair_ex_in_actmo}')

    Hmat_CSF = construct_Hmat_CSFs(list_CSF,Enuc,obt,tbt)
   
    Hmat_CSF_sparse = csr_matrix(Hmat_CSF)
    E0_CSF, psi_GS_CSF = get_ground_state(Hmat_CSF)
    print(f'\nE0 of original CSF space: {E0_CSF}')

    list_list_ia_included = []
    list_list_ex_space = []
    list_list_CSF_ind_ex_space = []
    list_list_genmat = []
    for iCSF_class in range(len(class_somo_ind)):
        print(f'\nChecking ia pairs for Class {iCSF_class}')
        list_ia_one_class = []
        list_improve = []
        iCSF_start = class_somo_ind[iCSF_class][0]
        iCSF_end   = class_somo_ind[iCSF_class][1]
        list_ex_space = copy.deepcopy(list_CSF[iCSF_start:iCSF_end+1])
        for iCSF in range(iCSF_start,iCSF_end+1):
            print(f'Checking ia pairs for CSF{iCSF}')
            CSF = list_CSF[iCSF]
            l_adding_ia = False
            for ia_pair in group_iapair:
                UCSF,Uvec,list_pair_ex_space,Umat, _ = prepare_UCSF_for_one_CSF(CSF,[ia_pair],[0.0],False)
                if len(list_pair_ex_space) == 1: continue
                list_UCSF_space = list_CSF + list_pair_ex_space[1:]
               #if debug: print(ia_pair,len(list_UCSF_space),len(list_CSF))
                Hmat_UCSF = construct_Hmat_CSFs(list_UCSF_space,Enuc,obt,tbt)
                Hmat_UCSF_sparse = csr_matrix(Hmat_UCSF)
                E0_UCSF, psi_GS_UCSF = get_ground_state(Hmat_UCSF_sparse)
                lowering = E0_UCSF-E0_CSF
                if lowering > 1.0e-8: print(f'Strange, positive lowering: {E0_UCSF} vs {E0_CSF}')
                if abs(lowering) > Ethrsh:
                   if ia_pair not in list_ia_one_class:
                       list_ia_one_class.append(ia_pair) 
                       list_improve.append(lowering)
                       l_adding_ia = True
                       l_exCSF_included = False
                       for item in list_pair_ex_space[1:]:
                           for CSFex in list_ex_space:
                               Selm = overlap_CSFs(item,CSFex)
                               if not np.isclose(Selm,0.0):
                                  #print('Strange! an EX CSF has been included before')
                                  #print(item)
                                   l_exCSF_included = True
                           if not l_exCSF_included: list_ex_space.append(item)
                   else:
                       ind_ia_pair = list_ia_one_class.index(ia_pair)
                       if lowering < list_improve[ind_ia_pair]:
                           print(f'Replacing list_improve: {ia_pair}, {ind_ia_pair},{list_improve[ind_ia_pair]},{lowering}')
                           list_improve[ind_ia_pair] = lowering

            if debug and l_adding_ia: 
                print(f'Updated list_ia_one_class for Class: {iCSF_class}:')
                for ii in range(len(list_ia_one_class)):
                    print(list_ia_one_class[ii],list_improve[ii])

        if len(list_improve) == 0:
            list_list_ia_included.append(list_ia_one_class) #An empty list is appended
            list_list_ex_space.append(list_ex_space) #list_ex_space = the original CSFs and is appended
            list_list_CSF_ind_ex_space.append(list(range(len(list_ex_space)))) #Append a list of [0,1,2,...] as the order is the same
            list_list_genmat.append([])
        else:
            list_improve, list_ia_one_class = zip(*sorted(zip(list_improve,list_ia_one_class)))
            list_ia_one_class = list(list_ia_one_class)
            print(f'\nMaking ex CSF space of a whole class {iCSF_class}')
            list_random_theta = []
            for ii in range(len(list_ia_one_class)):
                list_random_theta.append(random.uniform(0,1))

            print(f'a list of randome thetas: {list_random_theta}')

            list_all_ia_one_class = list_pair_ex_in_actmo + list_ia_one_class
            list_all_theta_one_class = list_theta_0 + list_random_theta
            CSF = list_CSF[iCSF_start]
            UCSF, Uvec, list_pair_ex_space, Umat, list_genmat = prepare_UCSF_for_one_CSF(CSF,list_all_ia_one_class,list_all_theta_one_class,False)
           #list_genmat = list_genmat[len(list_pair_ex_in_actmo):]
            list_genmat = construct_list_genmat_from_occ(list_pair_ex_space,list_ia_one_class,debug=True)
            if iCSF_class == 3:
                for item in list_pair_ex_space:
                    print(f'list dmo: {dmo_in_SD(item[0][0])}')
                for igenmat,genmat in enumerate(list_genmat):
                    print(f'genmat {igenmat}:')
                    print_matrix(genmat.toarray())
            for iCSF in range(iCSF_start+1,iCSF_end+1):
                if list_set_dmo[iCSF] == list_set_dmo[iCSF_start]:
                    CSF = list_CSF[iCSF]
                    UCSF, Uvec, list_pair_ex_space_more, Umat, list_genmat_more = prepare_UCSF_for_one_CSF(CSF,list_all_ia_one_class,list_all_theta_one_class,False)
                    list_genmat_more = construct_list_genmat_from_occ(list_pair_ex_space_more,list_ia_one_class,debug=True)
                    if iCSF_class == 3:
                        for igenmat,genmat_more in enumerate(list_genmat_more):
                            print(f'genmat_more {igenmat}:')
                            print_matrix(genmat_more.toarray())
                    list_pair_ex_space += list_pair_ex_space_more
                   #Expand the genmats to accommodate the space expansion
                    [nrow_orig,ncol_orig] = list_genmat[0].shape
                    [nrow_new,ncol_new] = list_genmat_more[0].shape
                    print(nrow_orig,ncol_orig,nrow_new,ncol_new)
                    for ii in range(len(list_genmat)):
                        list_genmat[ii] = scipy.sparse.hstack([list_genmat[ii],csr_matrix((nrow_orig,ncol_new))])
                        list_genmat_more[ii] = scipy.sparse.hstack([csr_matrix((nrow_new,ncol_orig)),list_genmat_more[ii]])
                        list_genmat[ii] = scipy.sparse.vstack([list_genmat[ii],list_genmat_more[ii]])
                        print(f'Expanded genmat {ii}:')
                        print(list_genmat[ii])
                    
                    
            nCSF_in_ex_space = len(list_pair_ex_space)
            print(f'# of all CSFs in the ex space of Class {iCSF_class}: {nCSF_in_ex_space}')
            list_CSF_ind_in_ex_space = []
            for iCSF, CSF in enumerate(list_CSF[iCSF_start:iCSF_end+1]):
                l_CSF_found = False
                for iCSFex, CSFex  in enumerate(list_pair_ex_space):
                    Selm = overlap_CSFs(CSF, CSFex)
                    if np.isclose(abs(Selm),1.0):
                        l_CSF_found = True
                        list_CSF_ind_in_ex_space.append(iCSFex)
                        break

                if not l_CSF_found:
                    print(f'CSF{iCSF+iCSF_start} not found in EX space. Bombing out!')
                    sys.exit()

            print(f'list of indices of CSF in the exCSF space: {list_CSF_ind_in_ex_space}')

            list_list_ia_included.append(list_ia_one_class)
            list_list_ex_space.append(list_pair_ex_space)
            list_list_CSF_ind_ex_space.append(list_CSF_ind_in_ex_space)
            list_list_genmat.append(list_genmat)

                

        print(f'\nIncluded ia pairs for Class: {iCSF_class}:')
        for ii in range(len(list_ia_one_class)):
            print(list_ia_one_class[ii],list_improve[ii])
        

    return list_list_ex_space, list_list_ia_included, list_list_genmat, list_list_CSF_ind_ex_space

def select_ia_pairs_for_one_CSF(CSF,group_mp2_ia_pairs,group_mp2_ampld,Enuc,obt,tbt,Ethrsh=1.0e-3,debug=False):
    """                 
    select ia pairs that give U|CSF> for a read-in CSF
    """
    if debug: print('\nIn select_ia_pairs_for_one_CSF')

    list_ia_included = []
    list_improve = []
    list_ia_ampld_included = []
   #Ethrsh = 1.0e-5
    for ia in range(len(group_mp2_ia_pairs)):
        UCSF, Uvec, list_pair_ex_space, Umat, list_genmat =\
            prepare_UCSF_for_one_CSF(CSF,group_mp2_ia_pairs[ia:ia+1],[0.0],False)
        n_mp2_ampld_opt = 1
        E_CSF = Helm_between_CSFs(Enuc,obt,tbt,CSF,CSF)
        list_Ustate_opt, list_mp2_ampld_opt, improve, E_UCSF_opt = opt_U_for_GS_of_Hmat([list_pair_ex_space],[list_genmat],[[0.0]],Enuc,obt,tbt,[n_mp2_ampld_opt])
       #Improve is re-defined as the energy lowering compared to the unperturbed CSF
       #improve = E_UCSF_opt - E_CSF
        assert np.isclose(improve, E_UCSF_opt - E_CSF)
        if abs(improve) > Ethrsh:
            list_ia_included.append(group_mp2_ia_pairs[ia])
            list_improve.append(improve)
            list_ia_ampld_included.append(list_mp2_ampld_opt[0][0])

    if len(list_ia_included) == 0:
        print('No ia pairs are found. Bombing out!')
        sys.exit()
    list_improve, list_ia_included, list_ia_ampld_included = zip(*sorted(zip(list_improve,list_ia_included,list_ia_ampld_included)))
    list_ia_included = list(list_ia_included)
    list_ia_ampld_included = list(list_ia_ampld_included)
    if debug:
        for ii in range(len(list_improve)):
            print(list_improve[ii],list_ia_included[ii],list_ia_ampld_included[ii])

    UCSF, Uvec, list_pair_ex_space, Umat, list_genmat =\
      prepare_UCSF_for_one_CSF(CSF,list_ia_included,list_ia_ampld_included,False)

    return list_pair_ex_space, list_ia_included, list_ia_ampld_included, list_genmat

def opt_flexible_U_for_one_CSF(CSF,group_mp2_ia_pairs,group_mp2_ampld,Enuc,obt,tbt,Ethrsh=1.0e-3,debug=False):
    """                 
    optimize U|CSF> for a read-in CSF
    """
    if debug: print('\nIn opt_flexible_U_for_one_CSF')

    list_ia_included = []
    list_improve = []
    list_ia_ampld_included = []
   #Ethrsh = 1.0e-5
    for ia in range(len(group_mp2_ia_pairs)):
        UCSF, Uvec, list_pair_ex_space, Umat, list_genmat =\
            prepare_UCSF_for_one_CSF(CSF,group_mp2_ia_pairs[ia:ia+1],group_mp2_ampld[ia:ia+1],False)
        n_mp2_ampld_opt = 1
       #list_Ustate_opt, list_mp2_ampld_opt, improve, _ = opt_U_for_GS_of_Hmat([list_pair_ex_space],[list_genmat],[group_mp2_ampld[ia:ia+1]],Enuc,obt,tbt,[n_mp2_ampld_opt])
        E_CSF = Helm_between_CSFs(Enuc,obt,tbt,CSF,CSF)
        list_Ustate_opt, list_mp2_ampld_opt, improve, E_UCSF_opt = opt_U_for_GS_of_Hmat([list_pair_ex_space],[list_genmat],[group_mp2_ampld[ia:ia+1]],Enuc,obt,tbt,[n_mp2_ampld_opt])
       #Improve is re-defined as the energy lowering compared to the unperturbed CSF
        improve = E_UCSF_opt - E_CSF
        list_ia_included.append(group_mp2_ia_pairs[ia])
        list_improve.append(improve)
        list_ia_ampld_included.append(list_mp2_ampld_opt[0][0])
       #print(f'list_mp2_ampld_opt: {list_mp2_ampld_opt}')
       #print(f'Energy lowring of {improve} for optimizing theta for {group_mp2_ia_pairs[ia:ia+1]}')


    list_l_remove = [True] * len(group_mp2_ia_pairs)
    l_next_loop = True
    scaled_Ethrsh = Ethrsh*10.0
    nround = 0
    print(f'list_improve: {list_improve}')
    while(l_next_loop):
        nround += 1
        print(f'nround: {nround}')
        scaled_Ethrsh *= 0.1
        for ia in range(len(group_mp2_ia_pairs)):
            if abs(list_improve[ia]) > scaled_Ethrsh:
                list_l_remove[ia] = False
        n_false = list_l_remove.count(False)        
        if n_false != 0: l_next_loop = False
        if nround == 10:
            print(f'Too many rounds')
            sys.exit()
        
    print(list_l_remove)
    for ii in range(len(group_mp2_ia_pairs)-1,-1,-1):
        if list_l_remove[ii]:
            del list_ia_included[ii]
            del list_improve[ii]
            del list_ia_ampld_included[ii]
    


#           list_ia_included.append(group_mp2_ia_pairs[ia])
#           list_improve.append(improve)
#           list_ia_ampld_included.append(list_mp2_ampld_opt[0][0])

   #print(f'list_ia_ampld_included: {list_ia_ampld_included}')
   #print(f'Included ia pairs before sorting')
   #for ii in range(len(list_ia_included)):
   #    print(list_ia_included[ii],list_improve[ii],list_ia_ampld_included[ii])
    list_improve, list_ia_included, list_ia_ampld_included = zip(*sorted(zip(list_improve,list_ia_included,list_ia_ampld_included)))
    list_ia_included = list(list_ia_included)
    list_ia_ampld_included = list(list_ia_ampld_included)
    print(f'The following {len(list_ia_included)} ia pairs pass the first round of screening:')
    for ii in range(len(list_ia_included)):
        print(list_ia_included[ii],list_improve[ii],list_ia_ampld_included[ii])

    UCSF, Uvec, list_pair_ex_space, Umat, list_genmat =\
        prepare_UCSF_for_one_CSF(CSF,list_ia_included,list_ia_ampld_included,False)

    n_mp2_ampld_opt=len(list_ia_ampld_included)
    list_Ustate_opt, list_mp2_ampld_opt, improve, _ = opt_U_for_GS_of_Hmat([list_pair_ex_space],[list_genmat],[list_ia_ampld_included],Enuc,obt,tbt,[n_mp2_ampld_opt])
    print(f'Improve in simultaneous opt of all included pairs: {improve}')
    E_CSF = Helm_between_CSFs(Enuc,obt,tbt,CSF,CSF)
    E_UCSF = Helm_between_CSFs(Enuc,obt,tbt,list_Ustate_opt[0],list_Ustate_opt[0])
    print(f'E_CSF: {E_CSF}, Energy of the opt state: {E_UCSF}, corr. E: {E_UCSF - E_CSF}')
    if debug:
        print(f'\nThe first round of simultaneous opt give the following thetas')
        for ii in range(len(list_mp2_ampld_opt[0])):
            print(list_ia_included[ii],list_mp2_ampld_opt[0][ii])

   #list_ia_train contains the whole train of ia rotation, including duplication of ia pairs.
   #list_ia_included does not contain the duplication. It only includes those in the first round opt.
    list_ia_train = copy.deepcopy(list_ia_included)
   #print(f'list_ia_train: {list_ia_train}')
   #2nd round, the last one is not included because it is trivial to have two identical generator back to back
    n_ia_pair_included = len(list_mp2_ampld_opt[0])
    n_mp2_ampld_opt = n_ia_pair_included + 1
    print(n_ia_pair_included,n_mp2_ampld_opt)

    l_next_round = True
    list_genmat_save = copy.deepcopy(list_genmat)
    list_theta_save = list_mp2_ampld_opt[0]
    list_ia_ind = []
    n_mp2_ampld_opt = len(list_theta_save)
    Ustate_opt = list_Ustate_opt[0]
    n_round = 0

    while(l_next_round): #l_next_round is never turned to False. The while loop is terminated by a "break" statement.
        list_include = []
        list_improve = []
        for ia in range(n_ia_pair_included):
            print(ia)
            list_genmat_2nd = copy.deepcopy(list_genmat_save) + [list_genmat[ia]]
            list_ia_ampld = copy.deepcopy(list_theta_save) + [0.0]
            list_Ustate_opt, list_mp2_ampld_opt_2nd, improve, _ = opt_U_for_GS_of_Hmat([list_pair_ex_space],[list_genmat_2nd],[list_ia_ampld],Enuc,obt,tbt,[n_mp2_ampld_opt+1],False)
            if abs(improve) >  Ethrsh:
                list_include.append([ia,list_mp2_ampld_opt_2nd[0][-1]])
                list_improve.append(improve)
    
        if len(list_include) == 0:
            print(f'No more ia rotation will be added. Opt converged after {n_round} rounds!')
            break

        n_round += 1
        if len(list_improve) != 0: list_improve, list_include = zip(*sorted(zip(list_improve,list_include)))
        list_genmat_2nd = copy.deepcopy(list_genmat_save)
        list_ia_ampld = copy.deepcopy(list_theta_save)
        n_mp2_ampld_opt += len(list_improve) 
        print(f'n_mp2_ampld_opt = {n_mp2_ampld_opt}')
        for ii in range(len(list_include)):
            list_genmat_2nd += [list_genmat[list_include[ii][0]]]
           #list_ia_ampld = list_mp2_ampld_opt[0] + [list_include[ii][1]]
            list_ia_ampld += [0.0]
            list_ia_train.append(list_ia_included[list_include[ii][0]])
            print(list_include[ii],list_improve[ii],list_ia_train[-1])
        print(f'More rounds are needed')
        print(list_ia_ampld)
        list_Ustate_opt, list_mp2_ampld_opt_2nd, improve, _ = opt_U_for_GS_of_Hmat([list_pair_ex_space],[list_genmat_2nd],[list_ia_ampld],Enuc,obt,tbt,[n_mp2_ampld_opt],True)
        print(f'Improve in opt all thetas: {improve}')
        E_CSF = Helm_between_CSFs(Enuc,obt,tbt,CSF,CSF)
        E_UCSF = Helm_between_CSFs(Enuc,obt,tbt,list_Ustate_opt[0],list_Ustate_opt[0])
        print(f'Energy of the opt state: {E_UCSF}, corr. E: {E_UCSF - E_CSF}')
        print(list_mp2_ampld_opt_2nd[0])
        list_genmat_save = list_genmat_2nd
        list_theta_save = list_mp2_ampld_opt_2nd[0]
        Ustate_opt = list_Ustate_opt[0]
        
       #sys.exit()

    assert len(list_theta_save) == len(list_ia_train)
    assert len(list_theta_save) == len(list_genmat_save)
    if debug:
        print('\nThe following train of ia rotation applies to create UCSF:')
        for ii in range(len(list_ia_train)):
            print(list_theta_save[ii],list_ia_train[ii])

    return Ustate_opt, list_pair_ex_space, list_genmat_save, list_theta_save, list_ia_train

def prepare_UCSF_for_one_CSF(CSF,group_mp2_ia_pairs,group_mp2_ampld,debug=False):
    """
    Prepare U|CSF> one one CSF
    """

    if debug: print('\nIn prepare_UCSF_for_one_CSF')

    ungroupped_ia_pair = []
    for pairs in group_mp2_ia_pairs:
        for pair in pairs:
            ungroupped_ia_pair.append(pair)


    [UCSF], [Uvec], [list_pair_ex_space], [Umat], [list_genmat] = prepare_UCSF_generic([CSF],ungroupped_ia_pair,group_mp2_ia_pairs,group_mp2_ampld,debug)
    return UCSF, Uvec, list_pair_ex_space, Umat, list_genmat
    

#def prepare_UCSFiiaa(list_bound,list_hf_pair_ex,Umat,debug=False):
#    """     
#    Prepare U|CSFiiaa> states, given the read-in dmo-to-vmo excited space of a HF reference
#    list_bound = [i_low,i_up,a_low,a_up], the lower and upper bounds of hole and particle orbitals
#    """
#
#    if debug: print('\nIn prepare_UCSFiiaa')
#
#    ndim = len(list_hf_pair_ex)
#    [i_low, i_up, a_low, a_up] = list_bound
#    Uiiaa_basis = []
#    Uiiaa_CIS_list = []
#    for i in range(i_up,i_low-1,-1):
#        for a in range(a_low,a_up+1):
#            lfound = False
#            for istate, state in enumerate(list_hf_pair_ex):
#                onlist = state[0]
#                coefs  = state[2]
#                occ_i = nocc_spatial_orb_LCSD(onlist,coefs,i)
#                occ_a = nocc_spatial_orb_LCSD(onlist,coefs,a)
#                if np.isclose(occ_i,0.0) and np.isclose(occ_a,2.0):
#                    lfound = True
#                    break
#           #It is normal not to find the doubly-excited states. Such an excitation may correspond to small mp2 amplitude
#           #and got screend off in the very beginning.
#           #if not lfound:
#           #    print(f'The {i,a} pair excited state is not found in the list of pair excited states from HF reference')
#           #    print('Bombing out')
#           #    sys.exit()
#            if not lfound: continue
#            if debug: print(i,a,istate,state)
#            Uiiaa_vec = np.zeros([ndim])
#            Uiiaa_vec[istate] = 1.0
#            Uiiaa_vec = Umat@Uiiaa_vec
#            Uiiaa_state = make_UCSF_state(list_hf_pair_ex,Uiiaa_vec)
#            Uiiaa_basis.append(Uiiaa_state)
#            Uiiaa_CIS_list.append([[i,a],[i,a]])
#
#            
#    return Uiiaa_basis, Uiiaa_CIS_list

def prepare_UCSF_pairs_in_subspace(list_bound,list_hf_pair_ex,Umat,n_pair_desired=0,debug=False):
    """
    Prepare U|CSF_I> states with I of all possible pair combination excitations within a space
    of a HF reference.
    list_bound = [i_low,i_up,a_low,a_up]. Orbitals lower than i_low should all have full occupations, and
    those higher than a_up should all have zero occupations
    """

    if debug:
        print('\nIn prepare_UCSF_pairs_in_subspace')
        print('States in hf pair ex space:')
        for item in list_hf_pair_ex:
            print(item)
    

    lbomb = False
    if len(list_bound) != 4: lbomb = True
    if debug: print(f'list_bound: {list_bound}')
    for i, orb in enumerate(list_bound[:-2]):
        if orb > list_bound[i+1]: lbomb = True

    if lbomb:
        print(f'\nInappropriate list_bound: {list_bound}')
        print('Bombing out')
        sys.exit()

    ndim = len(list_hf_pair_ex)
    [i_low, i_up, a_low, a_up] = list_bound


    n_spinorb = len(list_hf_pair_ex[0][0][0])
    n_spatialorb = n_spinorb // 2
   #print(list_hf_pair_ex[0][0][0])
   #nel = np.sum(list_hf_pair_ex[0][0][0])
   #hdmo = nel // 2 - 1
    if debug: print(f'n_spatialorb: {n_spatialorb}')

    Upairs_basis = []
    Upairs_CIS_list = []
    Upairs_vec_list = []

    for istate, state in enumerate(list_hf_pair_ex):
        if istate == 0: continue #Skip the first reference state
        onlist = state[0]
        coefs  = state[2]
        l_exclude = False
        for p in range(i_low):
            if not np.isclose(nocc_spatial_orb_LCSD(onlist,coefs,p),2.0): l_exclude = True
        for p in range(a_up+1,n_spatialorb):
            if not np.isclose(nocc_spatial_orb_LCSD(onlist,coefs,p),0.0): l_exclude = True
        if not l_exclude:
            i_list = []
            a_list = []
            for p in range(i_low,a_up+1):
                occ_p = nocc_spatial_orb_LCSD(onlist,coefs,p)
                if np.isclose(occ_p,0.0) and p <= i_up: i_list.append(p)
                if np.isclose(occ_p,2.0) and p >= a_low: a_list.append(p)

            if len(i_list) != len(a_list):
                print(f'\nInconsistent # of hole pairs and particle pairs in subspace {i_low,a_up}')
                print(f'# of hole pairs: {len(i_list)} in {i_list}')
                print(f'# of part pairs: {len(a_list)} in {a_list}')
                print('Bombing out!')
                sys.exit()

            n_pair_ex = len(i_list)
            if n_pair_desired != 0 and n_pair_ex != n_pair_desired: l_exclude = True
            

        if debug:
            if l_exclude: 
               print(f'Excluded: {state}')
            else:
               print(f'Included: {state}')

        if not l_exclude:
            Upairs_vec = np.zeros([ndim])
            Upairs_vec[istate] = 1.0
            Upairs_vec = Umat@Upairs_vec
            Upairs_state = make_UCSF_state(list_hf_pair_ex,Upairs_vec)
            Upairs_basis.append(Upairs_state)
            
           #i_list = []
           #a_list = []
           #for p in range(i_low,a_up+1):
           #    occ_p = nocc_spatial_orb_LCSD(onlist,coefs,p)
           #    if np.isclose(occ_p,0.0) and p <= i_up: i_list.append(p)
           #    if np.isclose(occ_p,2.0) and p >= a_low: a_list.append(p)

           #if len(i_list) != len(a_list):
           #    print(f'\nInconsistent # of hole pairs and particle pairs in subspace {i_low,a_up}')
           #    print(f'# of hole pairs: {len(i_list)} in {i_list}')
           #    print(f'# of part pairs: {len(a_list)} in {a_list}')
           #    print('Bombing out!')
           #    sys.exit()

            pair_list = []
            for ii in range(len(i_list)):
                pair_list.append([i_list[ii],a_list[ii]])
                pair_list.append([i_list[ii],a_list[ii]])
            
            Upairs_CIS_list.append(pair_list)
            Upairs_vec_list.append(Upairs_vec)

    if debug:
        for ii, state in enumerate(Upairs_basis):
            print(Upairs_CIS_list[ii])
        
    return Upairs_basis,Upairs_CIS_list, Upairs_vec_list
            

def nocc_spatial_orb_SD(onvec,p_spatialmo):
    """
    Return occupation number of a spatial orbital in a SD with ON vector onvec
    """

    pa, pb = p_spatialmo*2, p_spatialmo*2+1

    np = onvec[pa] + onvec[pb]

    return np

def nocc_spatial_orb_LCSD(onlist,coefvec,p_spatialmo):
    """
    Return average occupation number of a spatial orbital in a linear combination of SDs
    """

    np = 0.0
    for ii, on in enumerate(onlist):
        coef = coefvec[ii]
        np += coef*coef*nocc_spatial_orb_SD(on,p_spatialmo)

    return np

def get_SOMO_in_CSF(CSF):
    """
    Return a list of spatial orbitals whose occupancies are 1
    """

    onlist = CSF[0]
    coefvec = CSF[2]
    n_spinmo = len(onlist[0])
    n_spatialmo = n_spinmo // 2
    list_somo = []
    for imo in range(n_spatialmo):
        occ_imo = nocc_spatial_orb_LCSD(onlist,coefvec,imo)
        if np.isclose(occ_imo,1.0): list_somo.append(imo)

    return list_somo

def dmo_in_SD(onvec):
    """
    Return a list of doubly occupied orbitals of an ON vector
    """

    list_dmo = []
    n_spatial = len(onvec) // 2
    for ii in range(n_spatial):
        occ = onvec[2*ii] + onvec[2*ii+1]
        if np.isclose(occ,2.0): list_dmo.append(ii)

    return list_dmo

def calc_Hmat_diagonal_block(list_pair_ex_space,Enuc,obt_phys,tbt_phys,debug=False):
    """
    Given a set of state generated by multi-pair excitations of the reference (0th state), calculate their
    Hamiltonian matrix elements
    """

    if debug: print('\nIn Hmat_diagonal_block')

    ndim = len(list_pair_ex_space)
    Hmat = np.zeros([ndim,ndim])

    for ii, state in enumerate(list_pair_ex_space):
        onlist_i = state[0]
        coefs_i  = state[2]
        for jj in range(ii,ndim):
            onlist_j = list_pair_ex_space[jj][0]
            coefs_j  = list_pair_ex_space[jj][2]
            Helm = Helm_between_LCSDs(Enuc,obt_phys,tbt_phys,onlist_i,coefs_i,onlist_j,coefs_j)
            Hmat[ii,jj] = Helm
            Hmat[jj,ii] = Helm

    return Hmat

def read_fcidump(filename,debug=False):
    """
    Read Hamiltonian from fcidump file. It is taken from Josh Cantin
    """

    import pyscf.tools.fcidump

    fcidump_dic = pyscf.tools.fcidump.read(filename, molpro_orbsym=True, verbose=True)  
    num_orbitals = int(fcidump_dic['NORB'])
    tbt_full = pyscf.ao2mo.restore("s1",fcidump_dic['H2'],num_orbitals)
    tbt_phys_spatial = np.transpose(tbt_full, [0,3,1,2])
   #tbt_phys_spatial = tbt_full
    obt_spatial = fcidump_dic['H1']
    Ecore = fcidump_dic['ECORE']
    if debug:
        print(f'Dimensions of obt_spatial: {obt_spatial.shape}')
        print(f'Dimensions of tbt_phys_spatial: {tbt_phys_spatial.shape}')

    return Ecore, obt_spatial, tbt_phys_spatial
   #stupid line

def explicit_calc_matelm(state_bra,state_ket,op):
    """
    Calculate matrix element explicitly using braket_tz
    """

    onlist_ket = state_ket[0]
    coefs_ket = state_ket[2]
    onlist_bra = state_bra[0]
    coefs_bra = state_bra[2]
    
    elm = 0.0
    for ibra, onl in enumerate(onlist_bra):
        coef_l = coefs_bra[ibra]
        for jket, onr in enumerate(onlist_ket):
            coef_r = coefs_ket[jket]
            elm += coef_l*coef_r*braket_tz(onl,onr,op)

    return elm

def SD_CSF_CoupCoef(S_by2,M_by2,tN_by2,sigma_by2):
    """
    The Clebsch-Gordan coefficients in Eq. 2.6.5 and 2.6.6 of the Helgaker book.
    All inputs are integers and are the variables in the equations multiplied by 2.
    """

    if sigma_by2 != 1 and sigma_by2 != -1:
        print(f'sigma_by2 {sigma_by2}, is neither 1 or -1')
        print('Bombing out')
        sys.exit()

    S = float(S_by2) / 2.0
    M = float(M_by2) / 2.0
    sigma = float(sigma_by2) / 2.0

    if tN_by2 == 1:
        CoupCoef = S + 2.0*sigma*M
        CoupCoef /= 2.0*S
        CoupCoef = np.sqrt(CoupCoef)
    elif tN_by2 == -1:
        CoupCoef = S + 1.0 - 2.0*sigma*M
        CoupCoef /= 2.0*(S+1.0)
        CoupCoef = np.sqrt(CoupCoef)
        CoupCoef *= -2.0*sigma
    else:
        print(f'Unrecognized tN_by2 {tN_by2}, which shall only be +1 and -1.')
        print('Bombing out!')
        sys.exit()

    return CoupCoef

    
def generate_CASCI_space(Norb_total, Nel_total,Nactorb,Nactel,S_by2,list_seniority=[],l_pair_ex=True,debug=False):
    """
    Generate all CSFs in a CAS space with total spin S. The integral SX2 is read-in
    as S_by2. S_by2 also gives the minimum number of singly occupied orbitals
    """

    from itertools import combinations
    import copy

    if debug: print('\nIn generate_CASCI_space')

    N_SOMO_min = S_by2
    if (Nactel - S_by2) % 2 != 0:
        print(f'The number of active electrons {Nactel} is inconsistent with SX2 {S_by2}')
        print('Bombing out!')
        sys.exit()

    N_SOMO_max = 2*Nactorb - Nactel

    Ninactorb = (Nel_total - Nactel) // 2
    if debug: 
        print(f'N_SOMO_min {N_SOMO_min}, N_SOMO_max {N_SOMO_max}')
        print(f'# of inactive orbitals: {Ninactorb}')

    onvec_core = np.zeros([2*Norb_total])
    onvec_core[0:2*Ninactorb] = 1.0
    if debug: print(f'onvec_core: {onvec_core}')

    list_actmo = []
    for orb in range(Ninactorb,Ninactorb+Nactorb):
        list_actmo.append(orb)

    array_of_1_one = np.ones(1)
    list_CSF = []
    list_SOMO_DMO = []
    if len(list_seniority) != 0:
        lbomb = False
        if min(list_seniority) < N_SOMO_min: lbomb = True
        if max(list_seniority) > Nactel:
            lbomb = True
            print('seniority > # of active electrons')
        for seniority in list_seniority:
            if (seniority + N_SOMO_min) % 2 != 0: lbomb = True

        if lbomb:
            print(f'Incompatible list_seniority: {list_seniority} and N_SOMO_min: {N_SOMO_min}')
            print('Bombing out!')
            sys.exit()

    else:
        for N_SOMO in range(N_SOMO_min, N_SOMO_max+1,2):
            list_seniority.append(N_SOMO)
   #for N_SOMO in range(N_SOMO_min, N_SOMO_max+1,2):
    for N_SOMO in list_seniority:

        N_alpha = (N_SOMO + S_by2) // 2
        N_beta  = (N_SOMO - S_by2) // 2

        l_opensh = False
        if N_SOMO != 0: 
            l_opensh = True
            list_CSF_prototype = geneological_SD_CSF(N_alpha, N_beta,False)
            if debug: 
                print(f'\n# of SOMOs: {N_SOMO}')
                print(f'Dimension of list_CSF_prototype {len(list_CSF_prototype)}')
                print('Compare it with Fig 2.1 of Helgaker book')

        Nel_pair = Nactel - N_SOMO
        if Nel_pair % 2 != 0:
             print(f'Nactel, {Nactel}, N_SOMO, {N_SOMO}, Nel_pair, {Nel_pair} not even')
             print(f'Bombing out')
             sys.exit()
        Ndmo = Nel_pair // 2
        list_SOMO_comb = list(combinations(list_actmo,N_SOMO))
       #print(list_SOMO)
        for SOMO_comb in list_SOMO_comb:

            list_CSF_SOMO = []
            if l_opensh:
                for CSF_proto in list_CSF_prototype:
               #    print(CSF_proto)
                    onlist = []
                    for iSD, SD in enumerate(CSF_proto[0]):
                        onvec = copy.deepcopy(onvec_core)
                        for iSOMO, SOMO in enumerate(SOMO_comb):
                            onvec[2*SOMO:2*SOMO+2] = SD[2*iSOMO:2*iSOMO+2]

                       #if debug:
                       #    print(SOMO_comb,SD)
                       #    print(onvec)
                        onlist.append(onvec)

                    list_CSF_SOMO.append(onlist)


                assert len(list_CSF_prototype) == len(list_CSF_SOMO)
               #print(list_CSF_SOMO)
               
            list_left_orb = copy.deepcopy(list_actmo)
            for SOMO in SOMO_comb:
                list_left_orb.remove(SOMO)
           #print(f'SOMOs: {SOMO_comb}, left over orbitals: {list_left_orb}')
            list_DMO_comb = list(combinations(list_left_orb,Ndmo))
            dmo_sum_1st = np.sum(list_DMO_comb[0])
            for DMO_comb in list_DMO_comb:
                dmo_sum = np.sum(DMO_comb)
                if dmo_sum < dmo_sum_1st:
                    print('The 0th DOM_comb does not give the smallest summation. Watching out!!!')
                    print(list_DMO_comb[0],DMO_comb)
                    sys.exit()
                list_VMO = copy.deepcopy(list_left_orb)
                for MO in DMO_comb:
                    list_VMO.remove(MO)

                if debug: print(f'SOMOs: {SOMO_comb}, DMOs: {DMO_comb}, VMOs: {list_VMO}')

                if l_opensh:
                    for iCSF_SOMO, CSF_SOMO in enumerate(list_CSF_SOMO):
                        onlist = []
                        onidx_list = []
                       #print(CSF_SOMO)
                        for on in CSF_SOMO:
                            onvec = copy.deepcopy(on)
                           #print(f'onvec: {onvec}')
                            for dmo in DMO_comb:
                                onvec[2*dmo:2*dmo+2] = 1.0
                           #print(onvec)
                            onlist.append(onvec)
                            onidx_list.append(get_on_idx(onvec))
                        list_CSF.append([onlist,onidx_list,list_CSF_prototype[iCSF_SOMO][2]])
                        list_SOMO_DMO.append([SOMO_comb,DMO_comb])
                else:
                    onvec = copy.deepcopy(onvec_core)
                    for dmo in DMO_comb:
                        onvec[2*dmo:2*dmo+2] = 1.0

                    onidx = get_on_idx(onvec)
                    list_CSF.append([[onvec],[onidx],array_of_1_one])
                    list_SOMO_DMO.append([SOMO_comb,DMO_comb])

               #Only the first in the combinations of dmos is considered if no pair excitaitons are considered
                if not l_pair_ex: break

   #print(list_CSF[0])
   #print(list_CSF[-1])
    n_CSF = len(list_CSF)
    print(f'Total # of CSF states: {n_CSF}')

   #Check whether the CSFs are all orthonormal and have desired spin eigenvalues
    if debug: check_spin_adapted_CSF(list_CSF,S_by2)

    return list_CSF, list_SOMO_DMO


def geneological_SD_CSF(N_alpha, N_beta,debug=False):
    """
    Geneological coupling between SDs and CSFs with specific numbers of alpha and
    beta electrons
    """

    if debug: print(f'\nIn geneological_SD_CSF, N_alpha = {N_alpha}, N_beta = {N_beta}')


    N_open = N_alpha + N_beta
    list_SOMO = []
    for i in range(N_open):
        list_SOMO.append(i)
    
    from itertools import combinations

    comb_alpha_orb = list(combinations(list_SOMO,N_alpha))
    if debug: print(comb_alpha_orb)
    
    list_pvec = []
    list_tvec = []
    for item in comb_alpha_orb:
        pvec = np.full(N_open,-1)
        for orb in item:
            pvec[orb] = 1
    
       #print(pvec)
        list_pvec.append(pvec)
        l_tvec = True
        for orb in range(N_open):
            if np.sum(pvec[:orb]) < 0:
                l_tvec = False
                break
    
        if l_tvec: list_tvec.append(pvec)
    
    
    if debug:
        print('\nlist_pvec:')
        print(list_pvec)
        print('\nlist_tvec:')
        print(list_tvec)

    list_Pvec = []
    for item in list_pvec:
       #print(item)
        Pvec = np.full(N_open,0)
        for i in range(len(item)):
           #print(i,np.sum(item[:i+1]))
            Pvec[i] = np.sum(item[:i+1])
       #print(Pvec)
        list_Pvec.append(Pvec)
    
    list_Tvec = []
    for item in list_tvec:
       #print(item)
        Tvec = np.full(N_open,0)
        for i in range(len(item)):
           #print(i,np.sum(item[:i+1]))
            Tvec[i] = np.sum(item[:i+1])
       #print(Tvec)
        list_Tvec.append(Tvec)
    
    if debug:
        print('\nlist_Pvec:')
        print(list_Pvec)
        print('\nlist_Tvec:')
        print(list_Tvec)

    list_CSF = []
    
    #loop over all CSFs
    for iCSF in range(len(list_Tvec)):
        Tvec = list_Tvec[iCSF]
        tvec = list_tvec[iCSF]
        list_Pvec_include = []
        list_pvec_include = []
        coef_p_t = []
        onlist = []
        onidx_list = []
       #Kick out Pvec and pvec if |P_N| > T_N 
        for iSD in range(len(list_Pvec)):
            Pvec = list_Pvec[iSD]
            pvec = list_pvec[iSD]
            l_include = True
            CoupCoef = 1.0
            for orb in range(N_open):
                if abs(Pvec[orb]) > Tvec[orb]:
                    l_include = False
                    break
                Tn_by2, Pn_by2, tn_by2, pn_by2 = Tvec[orb], Pvec[orb], tvec[orb], pvec[orb]
                CoupCoef *= SD_CSF_CoupCoef(Tn_by2, Pn_by2, tn_by2, pn_by2)
            if not l_include: continue
            list_Pvec_include.append(Pvec)
            list_pvec_include.append(pvec)
            coef_p_t.append(CoupCoef)
            onvec = np.zeros([2*N_open])
            for orb in range(N_open):
                if pvec[orb] == 1:
                    onvec[2*orb]   = 1.0
                else:
                    onvec[2*orb+1] = 1.0
            onlist.append(onvec)
            onidx_list.append(get_on_idx(onvec))
    
        coef_p_t = np.array(coef_p_t)
        list_CSF.append([onlist,onidx_list,coef_p_t])

    if debug: check_spin_adapted_CSF(list_CSF,N_alpha-N_beta)

    if debug:
        print('\nlist_CSF in geneological_SD_CSF')
        print(list_CSF)

    return list_CSF

def check_spin_adapted_CSF(list_CSF,S_by2):
    """
    Check whether the list of CSFs generated by geneological coupling of SDs
    are eigenstates of S^2 and Sz with appropriate eigenvalue. Also check
    their orthonormality
    """

    n_spinorb = len(list_CSF[0][0][0])
   #n_open = N_alpha + N_beta
   #n_spinorb = 2*n_open

   #S_by2 = N_alpha - N_beta
    M_eigval = float(S_by2)*0.5
    S_sq_eigval = float(S_by2)*0.5*(float(S_by2)*0.5+1.0)

    Ssq_op = get_S_squared(n_spinorb)
    Sz_op  = get_S_z(n_spinorb)

    #Check orthonormality
    for iCSF, CSF_bra in enumerate(list_CSF):
        for jCSF in range(iCSF,len(list_CSF)):
            CSF_ket = list_CSF[jCSF]
            Selm = overlap_LCSD(CSF_bra[0],CSF_bra[1],CSF_bra[2],CSF_ket[0],CSF_ket[1],CSF_ket[2])
            if iCSF == jCSF and not np.isclose(Selm,1.0):
                print(f'CSF {iCSF} not normalized: {Selm}. Bombing out!')
                sys.exit()
            if iCSF != jCSF and not np.isclose(Selm,0.0):
                print(f'CSFs {iCSF, jCSF} not orthogonal: {Selm}. Bombing out!')
                sys.exit()

        lSsqeig, Ssqeig = judge_eigen_on_list(Ssq_op,CSF_bra[0],CSF_bra[1],CSF_bra[2],S_sq_eigval)
        lSzeig, Szeig = judge_eigen_on_list(Sz_op,CSF_bra[0],CSF_bra[1],CSF_bra[2],M_eigval)
        if not lSsqeig:
            print(f'CSF {iCSF} is not an eigenstate of S^2. Bombing out')
            for iSD in range(len(CSF_bra[0])):
                print(CSF_bra[2][iSD],CSF_bra[0][iSD])
            sys.exit()
        if not lSzeig:
            print(f'CSF {iCSF} is not an eigenstate of Sz. Bombing out')
            for iSD in range(len(CSF_bra[0])):
                print(CSF_bra[2][iSD],CSF_bra[0][iSD])
            sys.exit()

def create_missing_axial_sym_CSFs(list_CSF,list_SOMO_DMO,list_degmo):
    """
    Create the missing symmetry partners of CSFs
    """

    list_extra_CSF = []
    list_insert_index = []
    list_extra_somo_dmo = []
    nCSF = len(list_CSF)
    for iCSF in range(nCSF):
        tuple_dmo  = list_SOMO_DMO[iCSF][1]
        tuple_somo = list_SOMO_DMO[iCSF][0]
        list_dmo_ex = []
        for [degmo1, degmo2] in list_degmo:
            if degmo1 in tuple_somo or degmo2 in tuple_somo: continue
            if degmo1 in tuple_dmo and degmo2 not in tuple_dmo:
                list_dmo_ex.append([degmo1, degmo2])
            elif degmo1 not in tuple_dmo and degmo2 in tuple_dmo:
                list_dmo_ex.append([degmo2, degmo1])

        if len(list_dmo_ex) != 0:
            print(f'CSF Basis {iCSF}, SOMO: {list_SOMO_DMO[iCSF][0]}, DMO: {list_SOMO_DMO[iCSF][1]}')
            print(f'sym pair to be created, pair excitation: {list_dmo_ex}')
            list_insert_index.append(iCSF+1)
            CSF_sym_partner = copy.deepcopy(list_CSF[iCSF])
            [tuple_somo,tuple_dmo] = copy.deepcopy(list_SOMO_DMO[iCSF])
            list_dmo = list(tuple_dmo)
            for [dmo, vmo] in list_dmo_ex:
                list_dmo[list_dmo.index(dmo)] = vmo
            tuple_dmo = tuple(list_dmo)
            list_extra_somo_dmo.append([tuple_somo,tuple_dmo])
            print(f'CSF sym partner, SOMO: {tuple_somo}, DMO: {tuple_dmo}')
            for ion, onvec in enumerate(CSF_sym_partner[0]):
                for [dmo, vmo] in list_dmo_ex:
                    onvec[2*dmo:2*dmo+2] = 0.0
                    onvec[2*vmo:2*vmo+2] = 1.0
                CSF_sym_partner[1][ion] = get_on_idx(onvec)
            list_extra_CSF.append(CSF_sym_partner)

    for iCSF_extra in range(len(list_extra_CSF)-1,-1,-1):
        print(iCSF_extra)
        list_CSF.insert(list_insert_index[iCSF_extra],list_extra_CSF[iCSF_extra])
        list_SOMO_DMO.insert(list_insert_index[iCSF_extra],list_extra_somo_dmo[iCSF_extra])

   #print(f'List of SOMO and DMO after symmetry partner insertions.')
   #for ii, SOMO_DMO in enumerate(list_SOMO_DMO):
   #    print(f'CSF Basis {ii}, SOMO: {SOMO_DMO[0]}, DMO: {SOMO_DMO[1]}')
            
def reorder_list_CSF_for_sym(list_CSF,list_SOMO_DMO,Enuc,obt,tbt,debug=False):   
    """
    Reorder list_CSF so that symmetry partners are adjacent
    """

    list_E_CSF = []
    list_doubly_degen_ind = []
    for CSF in list_CSF:
        E_CSF = Helm_between_CSFs(Enuc,obt,tbt,CSF,CSF)
        list_E_CSF.append(E_CSF)
    new_order = []
    degen_thrsh = 1e-8
    for iCSF in range(len(list_CSF)):
        if iCSF in new_order: continue
        new_order.append(iCSF)
        list_sym_partner = []
        for jCSF in range(iCSF+1,len(list_CSF)):
           #if np.isclose(list_E_CSF[iCSF],list_E_CSF[jCSF]): #np.isclose misjudge non-degen states as deg states
            if abs(list_E_CSF[iCSF]-list_E_CSF[jCSF]) < degen_thrsh:
                list_sym_partner.append(jCSF)

        if len(list_sym_partner) > 1:
            print(f'Strange. More than 1 partner is found for CSF{iCSF}: {list_sym_partner}')
            print('Bombing out!')
            sys.exit()
        elif len(list_sym_partner) == 1:
            new_order.append(list_sym_partner[0])

    print('new order of CSF with sym partners adjacent:')
    print(new_order)

    list_CSF = [list_CSF[ii] for ii in new_order]
    list_SOMO_DMO = [list_SOMO_DMO[ii] for ii in new_order]
    list_E_CSF = [list_E_CSF[ii] for ii in new_order]

    print(f'SOMOs and DMOs of the sym-reordered CSFs:')
    for ii, SOMO_DMO in enumerate(list_SOMO_DMO):
        E_CSF = Helm_between_CSFs(Enuc,obt,tbt,list_CSF[ii],list_CSF[ii])
        print(f'CSF Basis {ii}, SOMO: {SOMO_DMO[0]}, DMO: {SOMO_DMO[1]}, E: {E_CSF}')

    Hmat_CSF = construct_Hmat_CSFs(list_CSF,Enuc,obt,tbt)
    print('\nHamiltonian matrix of symmetry-ordered CSFs:')
    print_matrix(Hmat_CSF)
    Hmat_CSF_sparse = csr_matrix(Hmat_CSF)
    E_GS_CSF, psi_GS_CSF = get_ground_state(Hmat_CSF_sparse)
    print(f'\nE0 of symmetry-reordered CSFs: {E_GS_CSF}')
    print(f'psi_GS: {psi_GS_CSF}')

    nCSF = len(list_CSF)
    list_sym_sign = [0]*nCSF
    for iCSF in range(len(psi_GS_CSF)-1):
       #if np.isclose(list_E_CSF[iCSF],list_E_CSF[iCSF+1]):
        if abs(list_E_CSF[iCSF]-list_E_CSF[iCSF+1]) < degen_thrsh:
            list_sym_sign[iCSF] = round(np.sign(psi_GS_CSF[iCSF]) * np.sign(psi_GS_CSF[iCSF+1]))

    if debug:
        print(f'\nlist_sym_sign: {list_sym_sign}')

    list_sym_CSF = []
    for iCSF in range(len(list_sym_sign)):
        if list_sym_sign[iCSF] == 1 or list_sym_sign[iCSF] == -1:
            lcvec = np.array([np.sqrt(0.5),list_sym_sign[iCSF]*np.sqrt(0.5)])
            comb_CSF = LC_CSFs([list_CSF[iCSF],list_CSF[iCSF+1]],lcvec,True)
            list_sym_CSF.append(comb_CSF)
        else:
            if iCSF == 0:
                list_sym_CSF.append(list_CSF[iCSF])
            elif list_sym_sign[iCSF-1] == 1 or list_sym_sign[iCSF-1] == -1: 
                continue
            else:
                list_sym_CSF.append(list_CSF[iCSF])

    n_symCSF = len(list_sym_CSF)
    Smat_CSF_symCSF = csr_matrix((nCSF,n_symCSF))
    for iCSF in range(nCSF):
        CSF = list_CSF[iCSF]
        for jsymCSF in range(n_symCSF):
            symCSF = list_sym_CSF[jsymCSF]
            Selm = overlap_CSFs(CSF,symCSF)
            if not np.isclose(Selm,0.0): Smat_CSF_symCSF[iCSF,jsymCSF] = Selm
            
    print('\nOverlap matrix between CSFs and sym-adapted CSFs:')
    print(Smat_CSF_symCSF)

    return list_CSF, list_SOMO_DMO, list_sym_sign, list_sym_CSF

def opt_U_overlap(list_list_ia,list_list_theta,list_list_genmat,list_start_end,psi_GS_full,conv=1.0e-4,list_inivec=[],nparal=1,l_use_decomp_genmat=False,ldisp=False,debug=False):
    """
    Opt U for each CSF to maximize overlap with the ground state of the full pair ex space
    When l_use_decomp_genmat = True, list_list_genmat contains decomposed genmats, which give analytical U matrix
    """

    if debug: print('\nIn opt_U_overlap')

    n_CSF = len(list_list_ia)

    list_list_theta_opt = []
    list_Uvec_opt = []
    list_Umat_opt = []
    l_create_inivec = False
    if len(list_inivec) == 0: l_create_inivec = True
    for iCSF in range(n_CSF):
        print(list_start_end[iCSF][0],list_start_end[iCSF][1])
        target_vec = psi_GS_full[list_start_end[iCSF][0]:list_start_end[iCSF][1]]
        print(f'target vector of UCSF{iCSF}')
        print(target_vec)
        list_genmat = list_list_genmat[iCSF]
        list_theta = list_list_theta[iCSF]
        if l_create_inivec:
            inivec = np.zeros(len(target_vec))
            inivec[0] = 1.0
        else:
            inivec = list_inivec[iCSF]

        assert len(inivec) == len(target_vec)
        if len(list_genmat) != 0:
            list_theta_opt, opt_fun, Umat_opt, Uvec_opt = opt_U_overlap_one_UCSF(list_genmat,list_theta,target_vec,conv,inivec,nparal,l_use_decomp_genmat,ldisp,debug)
           #if 1.0+opt_fun > small:
           #    list_genmat_2nd = list_genmat + [list_genmat[0]]
           #    list_theta_2nd = copy.deepcopy(list_theta_opt)
           #    list_theta_2nd = np.append(list_theta_2nd,[0.0])
           #    print(f'Optimizing overlap using extra length of U train')
           #    print(list_theta_2nd)
           #    list_theta_opt, opt_fun, _, Uvec_opt = opt_U_overlap_one_UCSF(list_genmat_2nd,list_theta_2nd,target_vec,ldisp,debug)
           #    sys.exit()
        else:
            list_theta_opt = []
           #Uvec_opt = np.array([1.0])
           #Use normalized inivec as Uvec_opt for the case of no ia
            Uvec_opt = copy.deepcopy(inivec / np.linalg.norm(inivec))
            Umat_opt = np.eye(len(inivec))
        list_list_theta_opt.append(list_theta_opt)
        list_Uvec_opt.append(Uvec_opt)
        list_Umat_opt.append(Umat_opt)

    return list_list_theta_opt, list_Uvec_opt, list_Umat_opt

def opt_U_overlap_with_noEX_in_CAS(list_list_ia,list_list_theta,list_list_genmat,list_UCSF_subspace_start_end,psi_GS,Uopt_thrsh,list_list_pairex_CSF_vec,l_use_decomp_genmat,conv=1.0e-5,nparal=1,ldisp=False,debug=False):
    """
    Fit U theta parameters for U that contain only excitations out of CAS.
    """

    if debug: print('\nIn opt_U_overlap_with_noEX_in_CAS')

    n_refCSF = len(list_list_ia)
    ndim_full = list_UCSF_subspace_start_end[-1][-1]
    list_list_theta_opt = []
    list_list_Uvec_full = []
    list_list_vec_full = []
    n_Uvec_total = 0
    for iref in range(n_refCSF):
        [istart,iend] = list_UCSF_subspace_start_end[iref]
       #ref_CSF_vec = list_ref_CSF_vec[iref]
        ref_CSF_vec = list_list_pairex_CSF_vec[iref][0]
        assert len(ref_CSF_vec) == iend - istart 
        list_theta = list_list_theta[iref]
        if len(list_theta) == 0:
            list_list_theta_opt.append([])
            list_Uvec_full = []
            for CSFvec in list_list_pairex_CSF_vec[iref]:
                Uvec_full = np.zeros(ndim_full)
                Uvec_full[istart:iend] = CSFvec
                n_Uvec_total += 1
                list_Uvec_full.append(Uvec_full)

            list_list_Uvec_full.append(list_Uvec_full)
            list_list_vec_full.append(list_Uvec_full)
            continue

        list_vec_full = []
        for CSFvec in list_list_pairex_CSF_vec[iref]:
            vec_full = np.zeros(ndim_full)
            vec_full[istart:iend] = CSFvec
            list_vec_full.append(vec_full)
        list_list_vec_full.append(list_vec_full)

        list_genmat = list_list_genmat[iref]
        target_vec = psi_GS[istart:iend]
        target_norm = np.linalg.norm(target_vec)
        print(f'\nNorm of the target vector for iref {iref}: {target_norm}')
        norm_target_vec = copy.deepcopy(target_vec) / target_norm
       #list_inivec = [list_ref_CSF_vec[iref]] + list_list_pairex_CSF_vec[iref]
        list_inivec = list_list_pairex_CSF_vec[iref]
        x0 = np.array(copy.deepcopy(list_theta))

        def cost(x):
            return func_project_psi_into_U_space(x,list_genmat,list_inivec,norm_target_vec,l_use_decomp_genmat)

        options = {
        'maxiter' : 10000,
        'disp'    : ldisp
        }

        tic = time.perf_counter()
        test_fun = cost(x0)
        toc = time.perf_counter()
        print(f'test_fun = {test_fun}, time for evaluation: {toc - tic}')

        tic = time.perf_counter()
       #sol = minimize(cost, x0, method='BFGS',options=options)
        arguments = (list_genmat,list_inivec,norm_target_vec,nparal,l_use_decomp_genmat)
        sol = minimize(grad_project_psi_into_U_space, x0, args=arguments, method='BFGS',options=options,jac=True)
        toc = time.perf_counter()
        print(f'Time for minimization: {toc - tic}')

        opt_fun1 = sol.fun
        theta_opt1 = sol.x
 
        opt_fun = opt_fun1
        theta_opt = theta_opt1
        nround = 1
        max_round = 3
        while (1.0 + opt_fun > conv) and nround < max_round:
            nround += 1
            print(f'\nRound {nround} optimization')
            list_genmat_xN = list_genmat*nround
            x0 = np.random.uniform(low=-0.5, high=-0.5, size = (nround*len(list_theta),))
            def cost_xn(x):
                return func_project_psi_into_U_space(x,list_genmat_xN,list_inivec,norm_target_vec,l_use_decomp_genmat)

            tic = time.perf_counter()
            test_fun = cost_xn(x0)
            toc = time.perf_counter()
            print(f'test_fun = {test_fun}, time for evaluation: {toc - tic}')

            tic = time.perf_counter()
           #sol_xn = minimize(cost_xn, x0, method='BFGS',options=options)
            arguments = (list_genmat_xN,list_inivec,norm_target_vec,nparal,l_use_decomp_genmat)
            sol_xn = minimize(grad_project_psi_into_U_space, x0, args=arguments, method='BFGS',options=options,jac=True)
            toc = time.perf_counter()
            print(f'Time for minimization sol_xn: {toc - tic}')
            opt_fun = sol_xn.fun
            theta_opt = sol_xn.x

        if 1.0 + opt_fun > conv:
            print(f'After {nround} of extension of theta list, convergence is not reached')
            print(f'Still moving on')
        else:
            print(f'Convergece is reached with {nround} train of U')



        inivec = list_inivec[0]
        if nround == 1:
            list_genmat_final = list_genmat
        else:
            list_genmat_final = list_genmat_xN
        if abs(opt_fun) < abs(opt_fun1):
            print(f'Theta list of the first round opt is taken')
            theta_opt = theta_opt1
            list_genmat_final = list_genmat

        list_list_theta_opt.append(theta_opt)
        if l_use_decomp_genmat:
            Umat = make_Umat_decomp_genmat(list_genmat_final,theta_opt)
        else:
            Umat, _ = make_and_apply_U_matrix(list_genmat_final,theta_opt,True,inivec,False)

        list_Uvec = []
        for vec in list_inivec:
            Uvec = Umat@vec
            Uvec_full = np.zeros(ndim_full)
            Uvec_full[istart:iend] = Uvec
            n_Uvec_total += 1
            list_Uvec.append(Uvec_full)

        list_list_Uvec_full.append(list_Uvec)

    Umat_total = np.zeros([ndim_full,n_Uvec_total])
    Vmat_total = np.zeros([ndim_full,n_Uvec_total])
    icol = -1
    for list_Uvec_full in list_list_Uvec_full:
        for Uvec_full in list_Uvec_full:
            icol += 1
            Umat_total[:,icol] = Uvec_full

    icol = -1
    for list_vec_full in list_list_vec_full:
        for vec_full in list_vec_full:
            icol += 1
            Vmat_total[:,icol] = vec_full
            
    return list_list_theta_opt, list_list_Uvec_full, Umat_total, Vmat_total


def opt_U_overlap_with_pair_ex_CSF(psi_GS,list_list_genmat,list_list_theta,list_list_CSF_ind_ex_space,list_UCSF_subspace_start_end,debug=False):
    """
    Fit U theta parameters for CSF that contain both CSFs from explicit pair excitations in active space and
    U pair excitation involving orbitals not in active space.
    """

    if debug: print('\nIn opt_U_overlap_with_pair_ex_CSF')

    nCSF_class = len(list_list_genmat)
    if len(list_list_theta) == 0:
        if debug: print('Initializing list_list_theta for fitting U')
        for iCSF_class in range(nCSF_class):
            list_list_theta.append([0.0]*len(list_list_genmat[iCSF_class]))
            if debug: print(list_list_theta[iCSF_class])

    list_opt_thetas = copy.deepcopy(list_list_theta)
    ndim_full = list_UCSF_subspace_start_end[-1][-1]
    list_list_Uvec = []
    zerovec_ndim_full = np.zeros(ndim_full)
    if debug: print(f'ndim_full: {ndim_full}')
    for iCSF_class in range(nCSF_class):
        [istart, iend] = list_UCSF_subspace_start_end[iCSF_class]
        list_Uvec = []
        if len(list_list_theta[iCSF_class]) == 0: 
            for jj in range(istart,iend):
                uvec = copy.deepcopy(zerovec_ndim_full)
                uvec[jj] = 1.0
                list_Uvec.append(uvec)

            list_list_Uvec.append(list_Uvec)
            continue
        psi_target = psi_GS[istart:iend]
        psi_target = psi_target / np.linalg.norm(psi_target)
        print(istart,iend)
        list_genmat = list_list_genmat[iCSF_class]
       #print(len(list_genmat))
       #print(list_genmat)
        ndim = list_genmat[0].shape[0]
        assert iend - istart == ndim
        list_inivec = []
        for iCSF in range(len(list_list_CSF_ind_ex_space[iCSF_class])):
            print(iCSF,list_list_CSF_ind_ex_space[iCSF_class][iCSF])
            inivec = np.zeros(ndim)
            ind = list_list_CSF_ind_ex_space[iCSF_class][iCSF]
            inivec[ind] = 1.0
            list_inivec.append(inivec)
            for jCSF in range(iCSF):
                assert np.isclose(np.dot(list_inivec[jCSF],inivec),0.0)
                    

        x0 = np.array(list_list_theta[iCSF_class])
        tic = time.perf_counter()
        test_fun = func_project_psi_into_U_space(x0,list_genmat,list_inivec,psi_target,l_use_decomp_genmat)
        toc = time.perf_counter()
        print(f'Time to evaluate function: {toc - tic}')
        print(f'test_fun = {test_fun}')
        def cost(x):
            return func_project_psi_into_U_space(x,list_genmat,list_inivec,psi_target,l_use_decomp_genmat)

        options = {
        'maxiter' : 10000,
        'disp'    : True
        }        
            
        sol = minimize(cost, x0, method='BFGS',options=options)

        Umat, Uvec = make_and_apply_U_matrix(list_genmat,sol.x)
        for inivec in list_inivec:
            Uvec = copy.deepcopy(zerovec_ndim_full)
            Uvec[istart:iend] = Umat@inivec
            list_Uvec.append(Uvec)
        

        list_opt_thetas[iCSF_class] = list(sol.x)
        list_list_Uvec.append(list_Uvec)

   #Check orthonormality between Uvec
    
        
    return list_opt_thetas, list_list_Uvec

def grad_project_psi_into_U_space(x,list_genmat,list_inivec,psi_target,nparal=1,l_use_decomp_genmat=False):

    l_paral = False
    if nparal > 1: l_paral = True

    list_grad_comp = []
    if l_paral:
       #print(f'Parallel calculation of gradient using {nparal} processes')
        list_grad_comp = Parallel(n_jobs=nparal)(delayed(grad_comp_prj_psi_to_U_space)(ii,x,list_genmat,list_inivec,psi_target,l_use_decomp_genmat) for ii in range(len(x)+1))
    else:
        for icomp in range(len(x)+1):
            grad_comp = grad_comp_prj_psi_to_U_space(icomp,x,list_genmat,list_inivec,psi_target,l_use_decomp_genmat)
            list_grad_comp.append(grad_comp)

    funval = list_grad_comp[0]
    grad = np.array(list_grad_comp[1:])

    return funval,grad

def grad_comp_prj_psi_to_U_space(icomp,x,list_genmat,list_inivec,psi_target,l_use_decomp_genmat=False):

    eps = 1.0e-4
    if icomp == 0:
        prjtion = func_project_psi_into_U_space(x,list_genmat,list_inivec,psi_target,l_use_decomp_genmat)
        return prjtion
    else:
        x_disp = x.copy()
        x_disp[icomp-1] = x[icomp-1] + eps
        prj_plus  = func_project_psi_into_U_space(x_disp,list_genmat,list_inivec,psi_target,l_use_decomp_genmat)
        x_disp[icomp-1] = x[icomp-1] - eps
        prj_minus = func_project_psi_into_U_space(x_disp,list_genmat,list_inivec,psi_target,l_use_decomp_genmat)
        grad_comp = (prj_plus - prj_minus) / (2.0*eps)
        return grad_comp


def func_project_psi_into_U_space(x,list_genmat,list_inivec,psi_target,l_use_decomp_genmat=False):
    """
    Projection of psi_target onto a set of {U|vec>} states as function of the theta angles
    in U
    """

    inivec = list_inivec[0]
    if l_use_decomp_genmat:
         Umat = make_Umat_decomp_genmat(list_genmat,x)
         Uvec = Umat@inivec
    else:
        Umat, Uvec = make_and_apply_U_matrix(list_genmat,x,True,inivec,False)
   #for inivec in list_inivec:

    assert np.isclose(np.linalg.norm(Uvec),1.0)
    proj_val = (np.dot(Uvec,psi_target))**2.0
    for inivec in list_inivec[1:]:
        Uvec = Umat@inivec
        proj_val += (np.dot(Uvec,psi_target))**2.0

    proj_val *= -1.0 #To convert maximization to minimization
    if abs(proj_val) > 1.0 + 1.0e-8: #Give it some small room
        print(f'proj_val {proj_val} has magnitude > 1. Bombing out!')
        sys.exit()

    return proj_val
        

def opt_U_overlap_one_UCSF(list_genmat,list_theta,target_vec,conv,inivec,nparal=1,l_use_decomp_genmat=False,ldisp=False,debug=False):
    """
    Opt U for one CSF to maximize overlap with a target vector
    When l_use_decomp_genmat = True, list_genmat contains decomposed genmat that give analytical Umat
    """

   #ndim = list_genmat[0].shape[0]
   #assert ndim == len(target_vec)
    ndim = len(target_vec)
   #print(f'length of list_genmat: {len(list_genmat)}')
   #print(f'length of list_theta: {len(list_theta)}')
   #print(list_theta)
    assert len(list_genmat) == len(list_theta)

   #normalize the target_vec
    target_norm = np.linalg.norm(target_vec)
    norm_target = target_vec / target_norm
    print(f'target norm: {target_norm}')
    print(f'norm_target: {norm_target}')
   #inivec provded in input
   #inivec = np.zeros(ndim)
   #inivec[0] = 1.0

    def cost(x):
        return eval_ref_UCSF_ovlp(x, list_genmat,norm_target,inivec,l_use_decomp_genmat)

    options = {
        'maxiter' : 10000,
        'disp'    : ldisp
    }

    x0 = np.array(copy.deepcopy(list_theta))
    if np.isclose(np.sum(x0*x0),0.0):
        x0 =  np.random.uniform(low=-0.5, high=-0.5, size = (len(list_theta),))
    tic = time.perf_counter()
    ovlp_initial = cost(x0)
    toc = time.perf_counter()
    if debug: 
        print(f'Time to evaluate function: {toc-tic}')
        print(f'\ncost(x0) = {ovlp_initial}')


   #sol = minimize(cost, x0, method='BFGS',options=options)
    arguments = (list_genmat,norm_target,inivec,nparal,l_use_decomp_genmat)
    sol = minimize(grad_ref_UCSF_ovlp, x0, args=arguments, method='BFGS',options=options,jac=True)

    if l_use_decomp_genmat:
        Umat_opt = make_Umat_decomp_genmat(list_genmat,sol.x)
        Uvec_opt = Umat_opt@inivec
    else:
        Umat_opt, Uvec_opt = make_and_apply_U_matrix(list_genmat,sol.x,True,inivec)
    theta_opt = sol.x
    fun_opt = sol.fun
    fun_opt_1st = fun_opt
    theta_opt_1st = theta_opt
    Umat_opt_1st = Umat_opt
    Uvec_opt_1st = Uvec_opt
    print(f'Uvec_opt_1st: {Uvec_opt_1st}')
    print(f'theta_opt after 1st round fitting: {theta_opt}')
   #print(f'\nOverlap between opt Uvec and normalized target vec: {np.dot(Uvec_opt,norm_target)}')

    small = conv

    nround = 1
    max_nround = 10
    while target_norm*(1.0+fun_opt) > small and nround <= max_nround:
        nround += 1
       #if nround > max_nround:
       #    print(f'No convergence up to {max_nround} trains. Bombing out!')
       #    sys.exit()
        print(f'\nDo another round of U opt with simultaneous {nround} trains of U')
        x0 = np.random.uniform(low=-0.5, high=-0.5, size = (nround*len(list_theta),))
       #Randomization of the initial parameters is the key for the success of using two trains.
       #The following all zero initialization for two trains won't work. It gives the same result as one train.
       #This is because the two initial trains with all zero amplitudes commute. The gradients of the two sets
       #of parameters are identical.
       #x0 = np.array([0.0] * 2*len(list_theta)) #This won't work. Randomization of the initial parameters is the key.
        list_genmat_xn = list_genmat*nround
        def cost_Uxn(x):
            return eval_ref_UCSF_ovlp(x, list_genmat_xn,norm_target,inivec,l_use_decomp_genmat)
        tic = time.perf_counter()
        print(f'test_fun {nround} round: {cost_Uxn(x0)}')
        toc = time.perf_counter()
        print(f'Time to evaluate function: {toc - tic}')
       #sol_Uxn = minimize(cost_Uxn,x0,method='BFGS',options=options)
        arguments = (list_genmat_xn,norm_target,inivec,nparal,l_use_decomp_genmat)
        sol_Uxn = minimize(grad_ref_UCSF_ovlp,x0,args=arguments, method='BFGS',options=options,jac=True)
        print(f'Improvement with {nround} trains: {sol_Uxn.fun - fun_opt}')
        if l_use_decomp_genmat:
            Umat_opt = make_Umat_decomp_genmat(list_genmat_xn,sol_Uxn.x)
            Uvec_opt = Umat_opt@inivec
        else:
            Umat_opt, Uvec_opt = make_and_apply_U_matrix(list_genmat_xn,sol_Uxn.x,True,inivec)
        theta_opt = sol_Uxn.x
        print(f'theta_opt: {theta_opt}')
        Uvec_opt_print = Uvec_opt
        if np.dot(Uvec_opt,norm_target) < 0.0: Uvec_opt_print = - Uvec_opt
        print(f'Uvec_opt_print: {Uvec_opt_print}')
        print(f'Uvec_opt dot norm_target: {np.dot(Uvec_opt_print,norm_target)}')
        res_vec = norm_target - Uvec_opt_print
        print(f'residual vector: {res_vec}')
        print(f'Normalized res. vec: {res_vec / np.linalg.norm(res_vec)}')
        fun_opt = sol_Uxn.fun

    if nround > max_nround:
        print(f'Moving on although the U fitting does not converge.')
    else:
        print(f'Iterations on adding U trains converged: with {1.0+fun_opt} vs. threshold {small}')

    if abs(fun_opt - fun_opt_1st) < small:
        theta_opt = theta_opt_1st
        fun_opt = fun_opt_1st
        Umat_opt = Umat_opt_1st
        Uvec_opt = Uvec_opt_1st
        

    return theta_opt, fun_opt, Umat_opt, Uvec_opt
    

def opt_one_UCSF(UCSF_state,list_doci_ex_space,list_genmat,group_mp2_ampld,U_vec,Enuc,obt,tbt,debug=False):
    """
    Given a UCSF state generated by make_UCSF_state, use the symmetry-adapted basis in this state
    to construct a Hamiltonian matrix and get the ground state within this space.
    """

    if debug: print('\nIn opt_one_UCSF')

   #[onlist,onidx_list,coefvec] = UCSF_state
   #basis_list = [list_doci_ex_space[0]]
   #for ibas, basis in enumerate(list_doci_ex_space[1:]):
   #    Selm = overlap_LCSD(UCSF_state[0],UCSF_state[1],UCSF_state[2],basis[0],basis[1],basis[2])
   #    print(ibas,Selm,U_vec[ibas+1])

    ndim = len(list_doci_ex_space)

    Hmat = np.zeros([ndim,ndim])
    for ibra, bas_bra in enumerate(list_doci_ex_space):
        for jket in range(ibra,len(list_doci_ex_space)):
            bas_ket = list_doci_ex_space[jket]
            Helm = Helm_between_LCSDs(Enuc,obt,tbt,bas_bra[0],bas_bra[2],bas_ket[0],bas_ket[2])
            Hmat[ibra,jket] = Helm
            Hmat[jket,ibra] = Helm

    E_GS, psi_GS = get_ground_state(Hmat)
    E_UCSF = U_vec.transpose()@Hmat@U_vec
    Selm = np.dot(psi_GS,U_vec)
    if debug:
        print(f'E_GS in opt_one_UCSF: {E_GS}, vs E_UCSF: {E_UCSF}')
        print(f'Overlap between U state and psi_GS: {Selm}')

   #eigval,eigvec = np.linalg.eigh(Hmat)
   #print('\nEigensolutions of Hmat:')
   #print_eigen_solution(eigval,eigvec)

    inivec = np.zeros(ndim)
    inivec[0] = 1.0

    def cost(x):
        return eval_ref_UCSF_ovlp(x, list_genmat,psi_GS,inivec)

    x0 = copy.deepcopy(group_mp2_ampld)

    options = {
        'maxiter' : 10000,
        'disp'    : True
    }

    sol = minimize(cost, x0, method='BFGS',options=options)

    Umat_opt, Uvec_opt = make_and_apply_U_matrix(list_genmat,sol.x)

    Selm = np.dot(psi_GS,Uvec_opt)
    E_optUCSF = Uvec_opt.transpose()@Hmat@Uvec_opt
    if debug:
        print(f'Overlap between U_opt state and psi_GS: {Selm}')
        print(f'Energy of U_opt state and psi_GS: {E_optUCSF}')

    small = 0.95
    if abs(Selm) < small:
        print(f'Overlap between U_opt state and psi_GS < {small}')
        print('Bombing out!')
        sys.exit()

    Ustate_opt = make_UCSF_state(list_doci_ex_space,Uvec_opt)

    x_list = sol.x.tolist()

    return Ustate_opt, x_list

def compare_num_anl_Umat(x,list_genmat,list_decomp_genmat,nparal,debug=False):
    """
    Compare Umat created by numerical exponentiating genmat and analytical formula
    using decompoised genmat
    """

    tic = time.perf_counter()
    Umat_anl = make_Umat_decomp_genmat(list_decomp_genmat,x)
    toc = time.perf_counter()
    time_Umat_anl = toc - tic
    tic = time.perf_counter()
    Umat_paral = make_Umat_decomp_genmat_joblib(list_decomp_genmat,x,nparal)
    toc = time.perf_counter()
    time_Umat_paral = toc - tic
    print(f'Ratio of paral and sequantial anal. Umat calculation: {time_Umat_paral / time_Umat_anl}')
    tic = time.perf_counter()
    Umat_num, _ = make_and_apply_U_matrix(list_genmat,x)
    toc = time.perf_counter()
    time_Umat_num = toc - tic
    print(f'Time for preparing anal. and num. Umat, and their ratio: {time_Umat_anl,time_Umat_num,time_Umat_anl/time_Umat_num}')
    Umat_num = csr_matrix(Umat_num)

    assert np.isclose(scipy.sparse.linalg.norm(Umat_anl - Umat_num),0.0)
    assert np.isclose(scipy.sparse.linalg.norm(Umat_anl - Umat_paral),0.0)

def grad_comp_ref_UCSF_ovlp(icomp,x,list_genmat,refvec,inivec,l_use_decomp_genmat,eps=1.0e-4):
    """
    Numerical gradient of the eval_ref_UCSF_ovlp function. Component 0 gives the function value.
    Components 1 to len(x) gives the gradient components.
    """

    if icomp == 0:
        grad_comp = eval_ref_UCSF_ovlp(x,list_genmat,refvec,inivec,l_use_decomp_genmat)
    else:
        x_disp = x.copy()
        x_disp[icomp-1] = x[icomp-1] + eps
        func_plus = eval_ref_UCSF_ovlp(x_disp,list_genmat,refvec,inivec,l_use_decomp_genmat)
        x_disp[icomp-1] = x[icomp-1] - eps
        func_minus = eval_ref_UCSF_ovlp(x_disp,list_genmat,refvec,inivec,l_use_decomp_genmat)
        grad_comp = (func_plus - func_minus) / (2.0 * eps)

    return grad_comp
    
def grad_ref_UCSF_ovlp(x,list_genmat,refvec,inivec,nparal,l_use_decomp_genmat):
    
    l_paral = False
    if nparal != 1: l_paral = True
    list_grad_comp = []
    if l_paral:
       #from joblib import Parallel, delayed
        list_grad_comp = Parallel(n_jobs=nparal)(delayed(grad_comp_ref_UCSF_ovlp)(ii,x,list_genmat,refvec,inivec,l_use_decomp_genmat) for ii in range(len(x)+1))
    else:
        for icomp in range(len(x)+1):
            grad_comp = grad_comp_ref_UCSF_ovlp(icomp,x,list_genmat,refvec,inivec,l_use_decomp_genmat)
            list_grad_comp.append(grad_comp)

    funval = list_grad_comp[0]
    grad = np.array(list_grad_comp[1:])

    return funval,grad

def eval_ref_UCSF_ovlp(x,list_genmat,refvec,inivec,l_use_decomp_genmat=False):
    """
    Evaluate <ref|U|ini> as a function of the rotational angles in U
    When l_use_decomp_genmat = True, list_genmat contains decomposed genmat that gives analytical Umat
    """

    if l_use_decomp_genmat:
        Umat = make_Umat_decomp_genmat(list_genmat,x)
    else:
        Umat, Uvec = make_and_apply_U_matrix(list_genmat,x)

    func = refvec.transpose()@Umat@inivec

    func = -abs(func) #The overlap is always taken to be negative so that minimization of func gives maximization of overlap

    return func

def opt_U_for_GS_of_Hmat(list_list_ex_space,list_list_genmat,list_group_mp2_ampld,Enuc,obt,tbt,list_n_mp2_ampld_opt,debug=False):
    """
    Optimize U parameters to get lowest ground state energy
    """

    if debug: print('\nIn opt_U_for_GS_of_Hmat')

    nUCSF = len(list_list_ex_space)
    list_start_end = []
    if debug: print(f'# of states {nUCSF}')
    istart = 0
    x0 = []
    x_frozen = []
   #ndim_x = nUCSF*n_mp2_ampld_opt
   #ndim_x = nUCSF*len(list_group_mp2_ampld[0][0:n_mp2_ampld_opt])
   #print(f'\nlist_list_ex_space:')
   #print(list_list_ex_space)
    for iUCSF in range(nUCSF):
        ndim = len(list_list_ex_space[iUCSF])
        if debug: print(f'# of states in UCSF {iUCSF}: {ndim}')
        list_start_end.append([istart,istart+ndim])
        istart += ndim
       #print(list_group_mp2_ampld[iUCSF],n_mp2_ampld_opt)
        x0 += list_group_mp2_ampld[iUCSF][0:list_n_mp2_ampld_opt[iUCSF]]
        x_frozen += list_group_mp2_ampld[iUCSF][list_n_mp2_ampld_opt[iUCSF]:]

    if debug:
        print(f'\nlist_n_mp2_ampld_opt: {list_n_mp2_ampld_opt}')
        print(f'\nx0: {x0}')
    if debug: print(f'list_start_end: {list_start_end}')
    assert len(x_frozen) == 0 #The present setting disable freezing amplitudes.
    x0 = np.array(x0)
   #assert len(x0) == ndim_x
   #if debug: print(f'Total # of U parameters to optimize: {ndim_x}')

    ndim_all_ex = istart
    if debug: print(f'Dimension of the super Hamiltonian matrix: {ndim_all_ex}')
    if debug: print(f'\nMaking Hamiltonian super matrix for dimension {ndim_all_ex}')
    Hmat_all_ex = np.zeros([ndim_all_ex,ndim_all_ex])
   #Construct the hamiltonian matrix of the primitive basis states in list_list_ex_space
   #print(list_list_ex_space)
    for iUCSF in range(nUCSF):
        istart = list_start_end[iUCSF][0]
        for jUCSF in range(iUCSF,nUCSF):
            jstart = list_start_end[jUCSF][0]
            for ii, bas_i in enumerate(list_list_ex_space[iUCSF]):
                onlist_i = bas_i[0]
                coefs_i  = bas_i[2]
                i_ex_bas = istart + ii
               #if i_ex_bas % 10 == 0 and debug: print(f'i_ex_bas = {i_ex_bas}')
                for jj, bas_j in enumerate(list_list_ex_space[jUCSF]):
                    onlist_j = bas_j[0]
                    coefs_j  = bas_j[2]
                    j_ex_bas = jstart + jj
                    if j_ex_bas < i_ex_bas: continue
                    Helm = Helm_between_LCSDs(Enuc,obt,tbt,onlist_i,coefs_i,onlist_j,coefs_j)
                    Hmat_all_ex[i_ex_bas,j_ex_bas] = Helm
                    Hmat_all_ex[j_ex_bas,i_ex_bas] = Helm

   #EGS_func_of_U(list_list_ex_space,list_list_genmat,x0,Hmat_all_ex,list_start_end,False)
    def cost(x):
        return EGS_func_of_U(list_list_ex_space,list_list_genmat,x,x_frozen,Hmat_all_ex,list_start_end,list_n_mp2_ampld_opt,False)
    def cost_true(x):
        return EGS_func_of_U(list_list_ex_space,list_list_genmat,x,x_frozen,Hmat_all_ex,list_start_end,list_n_mp2_ampld_opt,True)
    E_GS_before_opt = cost(x0)
    if debug: print(f'E_GS before optimization: {E_GS_before_opt}')

    options = { 
        'maxiter' : 10000,
        'disp'    : False
    }               
                    
    sol = minimize(cost, x0, method='BFGS',options=options)

    nx_each_chunk = len(list_group_mp2_ampld[0])
    if debug: print(f'Optimized E_GS: {sol.fun}')

    list_Ustate_opt = []
    list_mp2_ampld_opt = []
    istart = 0
    for iUCSF in range(nUCSF):
        list_genmat = list_list_genmat[iUCSF]
        iend = istart + list_n_mp2_ampld_opt[iUCSF]
        theta_for_iUCSF = sol.x[istart:iend]
        Umat, Uvec = make_and_apply_U_matrix(list_genmat,theta_for_iUCSF)
        Ustate_opt = make_UCSF_state(list_list_ex_space[iUCSF],Uvec)
        list_Ustate_opt.append(Ustate_opt)
        list_mp2_ampld_opt.append(theta_for_iUCSF.tolist())
        istart = iend

    improve = sol.fun - E_GS_before_opt

    return list_Ustate_opt, list_mp2_ampld_opt, improve, sol.fun

def opt_multiple_U_for_GS_of_Hmat(list_list_ex_space,list_list_genmat,list_group_mp2_ampld,Enuc,obt,tbt,n_mp2_ampld_opt,list_n_U=None,debug=False):
    """
    Optimize U parameters to get lowest ground state energy
    """

    if debug: print('\nIn opt_U_for_GS_of_Hmat')


    nUCSF = len(list_list_ex_space)
    n_mp2_ampld_total = len(list_group_mp2_ampld[0][0])
    if list_n_U == None: list_n_U = [1] * nUCSF
    list_start_end = []
    if debug: print(f'# of states {nUCSF}')
    istart = 0
    x0_multi = []
    x_frozen_multi = []
    ndim_x = nUCSF*n_mp2_ampld_opt
   #ndim_x = nUCSF*len(list_group_mp2_ampld[0][0:n_mp2_ampld_opt])
    for iUCSF in range(nUCSF):
       #print(iUCSF,list_group_mp2_ampld[iUCSF],list_n_U[iUCSF])
        assert len(list_group_mp2_ampld[iUCSF]) == list_n_U[iUCSF]
        ndim = len(list_list_ex_space[iUCSF])
        if debug: print(f'# of states in UCSF {iUCSF}: {ndim}')
        list_start_end.append([istart,istart+ndim])
        istart += ndim
       #print(list_group_mp2_ampld[iUCSF])
        for group_mp2_ampld in list_group_mp2_ampld[iUCSF]:
            x0_multi += group_mp2_ampld[0:n_mp2_ampld_opt]
            x_frozen_multi += group_mp2_ampld[n_mp2_ampld_opt:]


    if debug: 
        print(f'length: {len(x0_multi),len(x_frozen_multi)}')
        print(f'list_start_end: {list_start_end}')

    x0_multi = np.array(x0_multi)
    print(f'Total # of multi U parameters to optimize: {len(x0_multi)}')

    ndim_all_ex = istart
    if debug: print(f'Dimension of the super Hamiltonian matrix: {ndim_all_ex}')
    if debug: print(f'\nMaking Hamiltonian super matrix for dimension {ndim_all_ex}')
    Hmat_all_ex = np.zeros([ndim_all_ex,ndim_all_ex])
   #Construct the hamiltonian matrix of the primitive basis states in list_list_ex_space
    for iUCSF in range(nUCSF):
        istart = list_start_end[iUCSF][0]
        for jUCSF in range(iUCSF,nUCSF):
            jstart = list_start_end[jUCSF][0]
            for ii, bas_i in enumerate(list_list_ex_space[iUCSF]):
                onlist_i = bas_i[0]
                coefs_i  = bas_i[2]
                i_ex_bas = istart + ii
               #if i_ex_bas % 10 == 0 and debug: print(f'i_ex_bas = {i_ex_bas}')
                for jj, bas_j in enumerate(list_list_ex_space[jUCSF]):
                    onlist_j = bas_j[0]
                    coefs_j  = bas_j[2]
                    j_ex_bas = jstart + jj
                    if j_ex_bas < i_ex_bas: continue
                    Helm = Helm_between_LCSDs(Enuc,obt,tbt,onlist_i,coefs_i,onlist_j,coefs_j)
                    Hmat_all_ex[i_ex_bas,j_ex_bas] = Helm
                    Hmat_all_ex[j_ex_bas,i_ex_bas] = Helm

   #EGS_func = EGS_func_of_U_multiset(list_list_ex_space,list_list_genmat,x0_multi,x_frozen_multi,list_n_U,Hmat_all_ex,list_start_end,False)
   #print(f'EGS_func = {EGS_func}')
    def cost(x):
        return EGS_func_of_U_multiset(list_list_ex_space,list_list_genmat,x,x_frozen_multi,list_n_U,Hmat_all_ex,list_start_end,False)
    print(f'E_GS before optimization: {cost(x0_multi)}')

    options = { 
        'maxiter' : 10000,
        'disp'    : True
    }               
                    
    sol = minimize(cost, x0_multi, method='BFGS',options=options)

   #nx_each_chunk = len(list_group_mp2_ampld[0][0])
    nchunk = np.sum(np.array(list_n_U))
    nx_each_chunk = len(x0_multi) // nchunk
    nx_frozen_each_chunk = len(x_frozen_multi) // nchunk

    list_Ustate_opt = []
    list_mp2_ampld_opt = []
    ichunk = -1
    for iUCSF in range(nUCSF):
        list_genmat = list_list_genmat[iUCSF]
        list_opt_theta = []
        for iset in range(list_n_U[iUCSF]):
            ichunk += 1
            theta_list = sol.x[ichunk*nx_each_chunk:(ichunk+1)*nx_each_chunk].tolist() +\
                         x_frozen_multi[ichunk*nx_frozen_each_chunk:(ichunk+1)*nx_frozen_each_chunk]
            list_opt_theta.append(theta_list)
            if debug:
                print(f'iUCSF: {iUCSF}, iset: {iset}')
                print(f'theta_list: {theta_list}')

            if iset == 0:
                Umat, Uvec = make_and_apply_U_matrix(list_genmat,theta_list)
            else:
                Umat, Uvec = make_and_apply_U_matrix(list_genmat,theta_list,True,Uvec)
            
        Ustate_opt = make_UCSF_state(list_list_ex_space[iUCSF],Uvec)
        list_Ustate_opt.append(Ustate_opt)
        list_mp2_ampld_opt.append(list_opt_theta)

        if debug: print(f'list_mp2_ampld_opt: {list_mp2_ampld_opt}')
    return list_Ustate_opt, list_mp2_ampld_opt, sol.fun

def EGS_func_of_U(list_list_ex_space,list_list_genmat,x,x_frozen,Hmat_all_ex,list_start_end,list_n_mp2_ampld_opt,debug=False):

    if debug: print('\nIn EGS_func_of_U')
    nUCSF = len(list_list_ex_space)
    ndim_all_ex = Hmat_all_ex.shape[0]
   #nx_each_chunk = len(x) // nUCSF
   #nx_fronzen_each_chunk = len(x_frozen) // nUCSF
    O_rowEX_colUCSF = np.zeros([ndim_all_ex,nUCSF])
   #if debug: print(nx_each_chunk)
    istart = 0
    for iUCSF in range(nUCSF):
        list_genmat = list_list_genmat[iUCSF]
        iend = istart + list_n_mp2_ampld_opt[iUCSF]
       #x_all = x[iUCSF*nx_each_chunk:(iUCSF+1)*nx_each_chunk].tolist() \
       #      + x_frozen[iUCSF*nx_fronzen_each_chunk:(iUCSF+1)*nx_fronzen_each_chunk]
        x_all = x[istart:iend]
        if debug: print(f'UCSF {iUCSF}, x_all: {x_all}')
       #Umat, Uvec = make_and_apply_U_matrix(list_genmat,x[iUCSF*nx_each_chunk:(iUCSF+1)*nx_each_chunk])
       #if debug and iUCSF == 1:
       #    print(f'genmats of iUCSF = 1 in EGS_func_of_U')
       #    for ii, item in enumerate(list_genmat):
       #        print(f'genmat {ii}')
       #        print(item)
        Umat, Uvec = make_and_apply_U_matrix(list_genmat,x_all)
       #if debug:
       #    print(f'Uvec for UCSF{iUCSF}: {Uvec}')
       #    if iUCSF == 1:
       #        print(f'Umat of iUCSF 1:')
       #        print_matrix(Umat)
        [chunk_start, chunk_end] = list_start_end[iUCSF]
        O_rowEX_colUCSF[chunk_start:chunk_end,iUCSF] = Uvec
        istart = iend

    Hmat_UCSF = O_rowEX_colUCSF.transpose()@Hmat_all_ex@O_rowEX_colUCSF
    if debug:
        print(f'\nHmat_UCSF in EGS_func_of_U')
        print_matrix(Hmat_UCSF)
    E_GS, psi_GS = get_ground_state(Hmat_UCSF)
    if debug: print(f'E_GS of Hmat_UCSF: {E_GS}')


    return E_GS

def EGS_func_of_U_multiset(list_list_ex_space,list_list_genmat,x,x_frozen,list_n_U,Hmat_all_ex,list_start_end,debug=False):

    if debug: print('\nIn EGS_func_of_U')
    nUCSF = len(list_list_ex_space)
    ndim_all_ex = Hmat_all_ex.shape[0]
    n_chunk = np.sum(np.array(list_n_U))
    nx_each_chunk = len(x) // n_chunk
    nx_fronzen_each_chunk = len(x_frozen) // n_chunk
   #print(n_chunk,nx_each_chunk,nx_fronzen_each_chunk)
    O_rowEX_colUCSF = np.zeros([ndim_all_ex,nUCSF])
    if debug: print(nx_each_chunk)
    ichunk = -1
    for iUCSF in range(nUCSF):
        list_genmat = list_list_genmat[iUCSF]
        for iset in range(list_n_U[iUCSF]):
            ichunk += 1
            x_all = x[ichunk*nx_each_chunk:(ichunk+1)*nx_each_chunk].tolist() \
              + x_frozen[ichunk*nx_fronzen_each_chunk:(ichunk+1)*nx_fronzen_each_chunk]
       #Umat, Uvec = make_and_apply_U_matrix(list_genmat,x[iUCSF*nx_each_chunk:(iUCSF+1)*nx_each_chunk])
            if iset == 0: 
                Umat, Uvec = make_and_apply_U_matrix(list_genmat,x_all)
            else:
                Umat, Uvec = make_and_apply_U_matrix(list_genmat,x_all,True,Uvec)

        [chunk_start, chunk_end] = list_start_end[iUCSF]
        O_rowEX_colUCSF[chunk_start:chunk_end,iUCSF] = Uvec

    Hmat_UCSF = O_rowEX_colUCSF.transpose()@Hmat_all_ex@O_rowEX_colUCSF
    E_GS, psi_GS = get_ground_state(Hmat_UCSF)
    if debug: print(f'E_GS of Hmat_UCSF: {E_GS}')


    return E_GS

def opt_orbitals_for_GS_of_Hmat(list_list_ex_space,list_list_genmat,list_group_mp2_ampld,Enuc,nelec,obt_spatial,tbt_phys_spatial,list_orb_rot,x0=[],debug=False):
    """
    Optimize U parameters to get lowest ground state energy
    """

    if debug: print('\nIn opt_orbitals_for_GS_of_Hmat')


    nUCSF = len(list_list_ex_space)
    n_mp2_ampld_total = len(list_group_mp2_ampld[0][0])
    list_start_end = []
    if debug: print(f'# of states {nUCSF}')
    istart = 0
    list_n_U = []
    for iUCSF in range(nUCSF):
        list_n_U.append(len(list_group_mp2_ampld[iUCSF]))
        if debug: print(iUCSF,list_group_mp2_ampld[iUCSF],list_n_U[iUCSF])
        ndim = len(list_list_ex_space[iUCSF])
        if debug: print(f'# of states in UCSF {iUCSF}: {ndim}')
        list_start_end.append([istart,istart+ndim])
        istart += ndim


    if debug: 
        print(f'list_start_end: {list_start_end}')

    if len(x0) == 0: x0 = np.zeros(nparam)
    obt, tbt = orthogonal_transform_obt_tbt(x0,list_orb_rot,obt_spatial,tbt_phys_spatial)


    ndim_all_ex = istart
    if debug: print(f'Dimension of the super Hamiltonian matrix: {ndim_all_ex}')
    if debug: print(f'\nMaking Hamiltonian super matrix for dimension {ndim_all_ex}')
    Hmat_all_ex = np.zeros([ndim_all_ex,ndim_all_ex])
    O_rowEX_colUCSF = np.zeros([ndim_all_ex,nUCSF])
   #Construct the hamiltonian matrix of the primitive basis states in list_list_ex_space
   #Construct the reduced density matrices for each pair of bra and ket
    list_list_rdm1 = []
    list_list_rdm2 = []
    ilist_rdm = -1
    for iUCSF in range(nUCSF):
       #Also construct U rotational vectors
        list_genmat = list_list_genmat[iUCSF]
        for iset in range(list_n_U[iUCSF]):
            group_mp2_ampld = list_group_mp2_ampld[iUCSF][iset]
            if iset == 0:
                Umat, Uvec = make_and_apply_U_matrix(list_genmat,group_mp2_ampld)
            else:
                Umat, Uvec = make_and_apply_U_matrix(list_genmat,group_mp2_ampld,True,Uvec)

        [chunk_start, chunk_end] = list_start_end[iUCSF]
        O_rowEX_colUCSF[chunk_start:chunk_end,iUCSF] = Uvec

        istart = list_start_end[iUCSF][0]
        for jUCSF in range(iUCSF,nUCSF):
            jstart = list_start_end[jUCSF][0]
            for ii, bas_i in enumerate(list_list_ex_space[iUCSF]):
                onlist_i = bas_i[0]
                coefs_i  = bas_i[2]
                i_ex_bas = istart + ii
               #if i_ex_bas % 10 == 0 and debug: print(f'i_ex_bas = {i_ex_bas}')
                for jj, bas_j in enumerate(list_list_ex_space[jUCSF]):
                    onlist_j = bas_j[0]
                    coefs_j  = bas_j[2]
                    j_ex_bas = jstart + jj
                    if j_ex_bas < i_ex_bas: continue
                    Helm, list_rdm1, list_rdm2 = Helm_between_LCSDs(Enuc,obt,tbt,onlist_i,coefs_i,onlist_j,coefs_j,lrdm=True)
                    Hmat_all_ex[i_ex_bas,j_ex_bas] = Helm
                    Hmat_all_ex[j_ex_bas,i_ex_bas] = Helm
                    Helm_from_rdm = elm_from_list_rdm(list_rdm1,list_rdm2,Enuc/nelec,obt,tbt)
                    assert np.isclose(Helm,Helm_from_rdm)
                    ilist_rdm += 1
                   #print(f'In storing list_list_rdm: {iUCSF,istart,ii,jUCSF,jstart,jj,i_ex_bas,j_ex_bas,ilist_rdm}')
                    list_list_rdm1.append(list_rdm1)
                    list_list_rdm2.append(list_rdm2)

    if debug:
        Hmat_UCSF = O_rowEX_colUCSF.transpose()@Hmat_all_ex@O_rowEX_colUCSF
        print(f'Hmat of UCSF basis:')
        print_matrix(Hmat_UCSF)

    if debug:
        ilist_rdm = -1
        for iUCSF in range(nUCSF):
            istart = list_start_end[iUCSF][0]
            for jUCSF in range(iUCSF,nUCSF):
                jstart = list_start_end[jUCSF][0]
                for ii, bas_i in enumerate(list_list_ex_space[iUCSF]):
                    i_ex_bas = istart + ii
                    for jj, bas_j in enumerate(list_list_ex_space[jUCSF]):
                        j_ex_bas = jstart + jj
                        if j_ex_bas < i_ex_bas: continue
                        ilist_rdm += 1
                        list_rdm1 = list_list_rdm1[ilist_rdm]
                        list_rdm2 = list_list_rdm2[ilist_rdm]
                        Helm_from_rdm = elm_from_list_rdm(list_rdm1,list_rdm2,Enuc/nelec,obt,tbt)
                        assert np.isclose(Helm_from_rdm,Hmat_all_ex[i_ex_bas,j_ex_bas])


    nparam = len(list_orb_rot)

    if debug: print(f'\n# of orbital rotation angles to optimize: {nparam}')


    def cost(x):
        return EGS_func_of_orbrot(list_list_rdm1,list_list_rdm2,ndim_all_ex,x,list_orb_rot,Enuc,nelec,obt_spatial,tbt_phys_spatial,O_rowEX_colUCSF,list_start_end,list_list_ex_space)
    tic = time.perf_counter()
    func_ini = costval = cost(x0)
    toc = time.perf_counter()
    print(f'Time to one function evaluation: {toc - tic}')
    print(f'test cost function: {func_ini}')

    options = {
        'maxiter' : 10000,
        'disp'    : True
    }                
                    
    sol = minimize(cost, x0, method='BFGS',options=options)

    if debug: print(f'opt rotational angles: {sol.x}')
    print(f'Energy lowering due to orbital rotaiton: {sol.fun - func_ini}')

    obt_opt_spin, tbt_opt_spin = orthogonal_transform_obt_tbt(sol.x,list_orb_rot,obt_spatial,tbt_phys_spatial)

   #Create the spatial orbital integrals of the transformed orbitals
   #n_spatialmo = obt_phys_spatial.shape[0]

   #kappa_mat = np.zeros([n_spatialmo,n_spatialmo])
   #for ix, pair in enumerate(list_orb_rot):
   #    [iorb,jorb] = pair
   #    kappa_mat[iorb,jorb] = sol.x[ix]
   #    kappa_mat[jorb,iorb] = -sol.x[ix]

   #Omat = scipy.linalg.expm(kappa_mat)

   #obt_phys_spatial_trans = np.einsum('pq,pa,qb->ab',obt_phys_spatial,Omat,Omat,optimize=True)
   #tbt_phys_spatial_trans = np.einsum('pqrs,pa,qb,rc,sd->abcd',tbt_phys_spatial,Omat,Omat,Omat,Omat,optimize=True)

   #obt_phys_spin_trans = obt_phys_spatial_to_spin(obt_phys_spatial_trans)
   #tbt_phys_spin_trans = tbt_phys_spatial_to_spin(tbt_phys_spatial_trans)

    return sol.x, sol.fun, obt_opt_spin, tbt_opt_spin

def opt_orbitals_for_GS_of_Hmat_1setU_each_UCSF(list_list_ex_space,list_list_genmat,list_group_mp2_ampld,Enuc,nelec,obt_spatial,tbt_phys_spatial,list_orb_rot,x0=[],debug=False):
    """
    Optimize U parameters to get lowest ground state energy
    """

    if debug: print('\nIn opt_orbitals_for_GS_of_Hmat')


    nUCSF = len(list_list_ex_space)
    list_start_end = []
    if debug: print(f'# of states {nUCSF}')
    istart = 0
    for iUCSF in range(nUCSF):
        ndim = len(list_list_ex_space[iUCSF])
        if debug: print(f'# of states in UCSF {iUCSF}: {ndim}')
        list_start_end.append([istart,istart+ndim])
        istart += ndim


    if debug: 
        print(f'list_start_end: {list_start_end}')

    if len(x0) == 0: x0 = np.zeros(nparam)
    obt, tbt = orthogonal_transform_obt_tbt(x0,list_orb_rot,obt_spatial,tbt_phys_spatial)


    ndim_all_ex = istart
    if debug: print(f'Dimension of the super Hamiltonian matrix: {ndim_all_ex}')
    if debug: print(f'\nMaking Hamiltonian super matrix for dimension {ndim_all_ex}')
    Hmat_all_ex = np.zeros([ndim_all_ex,ndim_all_ex])
    O_rowEX_colUCSF = np.zeros([ndim_all_ex,nUCSF])
   #Construct the hamiltonian matrix of the primitive basis states in list_list_ex_space
   #Construct the reduced density matrices for each pair of bra and ket
    list_list_rdm1 = []
    list_list_rdm2 = []
    ilist_rdm = -1
    for iUCSF in range(nUCSF):
       #Also construct U rotational vectors
        list_genmat = list_list_genmat[iUCSF]
        group_mp2_ampld = list_group_mp2_ampld[iUCSF]
        Umat, Uvec = make_and_apply_U_matrix(list_genmat,group_mp2_ampld)

        [chunk_start, chunk_end] = list_start_end[iUCSF]
        O_rowEX_colUCSF[chunk_start:chunk_end,iUCSF] = Uvec

        istart = list_start_end[iUCSF][0]
        for jUCSF in range(iUCSF,nUCSF):
            jstart = list_start_end[jUCSF][0]
            for ii, bas_i in enumerate(list_list_ex_space[iUCSF]):
                onlist_i = bas_i[0]
                coefs_i  = bas_i[2]
                i_ex_bas = istart + ii
               #if i_ex_bas % 10 == 0 and debug: print(f'i_ex_bas = {i_ex_bas}')
                for jj, bas_j in enumerate(list_list_ex_space[jUCSF]):
                    onlist_j = bas_j[0]
                    coefs_j  = bas_j[2]
                    j_ex_bas = jstart + jj
                    if j_ex_bas < i_ex_bas: continue
                    Helm, list_rdm1, list_rdm2 = Helm_between_LCSDs(Enuc,obt,tbt,onlist_i,coefs_i,onlist_j,coefs_j,lrdm=True)
                    Hmat_all_ex[i_ex_bas,j_ex_bas] = Helm
                    Hmat_all_ex[j_ex_bas,i_ex_bas] = Helm
                    Helm_from_rdm = elm_from_list_rdm(list_rdm1,list_rdm2,Enuc/nelec,obt,tbt)
                    assert np.isclose(Helm,Helm_from_rdm)
                    ilist_rdm += 1
                   #print(f'In storing list_list_rdm: {iUCSF,istart,ii,jUCSF,jstart,jj,i_ex_bas,j_ex_bas,ilist_rdm}')
                    list_list_rdm1.append(list_rdm1)
                    list_list_rdm2.append(list_rdm2)

    if debug:
        Hmat_UCSF = O_rowEX_colUCSF.transpose()@Hmat_all_ex@O_rowEX_colUCSF
        print(f'Hmat of UCSF basis:')
        print_matrix(Hmat_UCSF)

    if debug:
        ilist_rdm = -1
        for iUCSF in range(nUCSF):
            istart = list_start_end[iUCSF][0]
            for jUCSF in range(iUCSF,nUCSF):
                jstart = list_start_end[jUCSF][0]
                for ii, bas_i in enumerate(list_list_ex_space[iUCSF]):
                    i_ex_bas = istart + ii
                    for jj, bas_j in enumerate(list_list_ex_space[jUCSF]):
                        j_ex_bas = jstart + jj
                        if j_ex_bas < i_ex_bas: continue
                        ilist_rdm += 1
                        list_rdm1 = list_list_rdm1[ilist_rdm]
                        list_rdm2 = list_list_rdm2[ilist_rdm]
                        Helm_from_rdm = elm_from_list_rdm(list_rdm1,list_rdm2,Enuc/nelec,obt,tbt)
                        assert np.isclose(Helm_from_rdm,Hmat_all_ex[i_ex_bas,j_ex_bas])


    nparam = len(list_orb_rot)

    if debug: print(f'\n# of orbital rotation angles to optimize: {nparam}')


    def cost(x):
        return EGS_func_of_orbrot(list_list_rdm1,list_list_rdm2,ndim_all_ex,x,list_orb_rot,Enuc,nelec,obt_spatial,tbt_phys_spatial,O_rowEX_colUCSF,list_start_end,list_list_ex_space)
    tic = time.perf_counter()
    func_ini = costval = cost(x0)
    toc = time.perf_counter()
    print(f'Time to one function evaluation: {toc - tic}')
    print(f'test cost function: {func_ini}')

    options = {
        'maxiter' : 10000,
        'disp'    : True
    }                
                    
    sol = minimize(cost, x0, method='BFGS',options=options)

    if debug: print(f'opt rotational angles: {sol.x}')
    print(f'Energy lowering due to orbital rotaiton: {sol.fun - func_ini}')

    obt_opt_spin, tbt_opt_spin = orthogonal_transform_obt_tbt(sol.x,list_orb_rot,obt_spatial,tbt_phys_spatial)

   #Create the spatial orbital integrals of the transformed orbitals
   #n_spatialmo = obt_phys_spatial.shape[0]

   #kappa_mat = np.zeros([n_spatialmo,n_spatialmo])
   #for ix, pair in enumerate(list_orb_rot):
   #    [iorb,jorb] = pair
   #    kappa_mat[iorb,jorb] = sol.x[ix]
   #    kappa_mat[jorb,iorb] = -sol.x[ix]

   #Omat = scipy.linalg.expm(kappa_mat)

   #obt_phys_spatial_trans = np.einsum('pq,pa,qb->ab',obt_phys_spatial,Omat,Omat,optimize=True)
   #tbt_phys_spatial_trans = np.einsum('pqrs,pa,qb,rc,sd->abcd',tbt_phys_spatial,Omat,Omat,Omat,Omat,optimize=True)

   #obt_phys_spin_trans = obt_phys_spatial_to_spin(obt_phys_spatial_trans)
   #tbt_phys_spin_trans = tbt_phys_spatial_to_spin(tbt_phys_spatial_trans)

    return sol.x, sol.fun, obt_opt_spin, tbt_opt_spin

def opt_orbtials_for_GS_of_CSF_space(list_CSF,psi_coefs,Enuc,obt_spatial,tbt_spatial,list_orb_rot,list_list_rdm1,list_list_rdm2,l_axial_sym,list_degmo,x0=[],nparal=1,debug=False):
    """
    Optimize orbitals through rotation of specific orbital pairs for the ground state of list_CSF
    """

    if debug: print('\nIn opt_orbtials_for_GS_of_CSF_space')

    if len(x0) == 0: 
        x0 = np.array([0.0]*len(list_orb_rot))
        obt = obt_phys_spatial_to_spin(obt_spatial)
        tbt = tbt_phys_spatial_to_spin(tbt_spatial)
    else:
        assert len(list_orb_rot) == len(x0)
        obt, tbt = orthogonal_transform_obt_tbt(x0,list_orb_rot,obt_spatial,tbt_spatial)

   #Construct the Hamiltonian matrix of the whole ex space and the reduced density matrices
   #Hmat_CSF, list_list_rdm1, list_list_rdm2 = construct_Hmat_CSFs(list_CSF,Enuc,obt,tbt,lrdm=True)
    if debug: Hmat_CSF = construct_Hmat_CSFs(list_CSF,Enuc,obt,tbt)

    nCSF = len(list_CSF)
    nelec = len(np.where(list_CSF[0][0][0] == 1.0)[0])
    print(f'nelec: {nelec}')
    if debug:
        ilist_rdm = -1
        for iCSF in range(nCSF):
            for jCSF in range(iCSF,nCSF):
                ilist_rdm += 1
                list_rdm1 = list_list_rdm1[ilist_rdm]
                list_rdm2 = list_list_rdm2[ilist_rdm]
                Helm_from_rdm = elm_from_list_rdm(list_rdm1,list_rdm2,Enuc/nelec,obt,tbt)
                if not np.isclose(Helm_from_rdm,Hmat_CSF[iCSF,jCSF]):
                    print('rdm test failed')
                    print(Helm_from_rdm,Hmat_CSF[iCSF,jCSF])
                    print(iCSF,jCSF)
                    print(list_CSF[iCSF])
                    print(list_CSF[jCSF])
                    print('rdm1')
                    print(list_rdm1)
                    print('rdm2')
                    print(list_rdm2)
                    sys.exit()

    if debug: print('rdm test passed')

    def cost(x):
       #if debug:
       #    return EGS_func_of_orbrot_with_CI_coefs(x,list_orb_rot,list_list_rdm1,list_list_rdm2,psi_coefs,Enuc,nelec,obt_spatial,tbt_spatial,list_CSF)
       #else:
            return EGS_func_of_orbrot_with_CI_coefs(x,list_orb_rot,list_list_rdm1,list_list_rdm2,psi_coefs,Enuc,nelec,obt_spatial,tbt_spatial)
    tic = time.perf_counter()
    E_before_min = cost(x0)
    toc = time.perf_counter()
    print(f'time for function evaluation: {toc - tic}')
    if debug: print(f'Function value before minimization: {E_before_min}')

    options = {
        'maxiter' : 10000,
        'disp'    : True
    }                
                    
   #sol = minimize(cost, x0, method='BFGS',options=options)
    arguments = (list_orb_rot,list_list_rdm1,list_list_rdm2,psi_coefs,Enuc,nelec,obt_spatial,tbt_spatial,nparal)
    sol = minimize(grad_EGS_of_orbrot,x0, args=arguments, method='BFGS',options=options,jac=True)

    x_orbrot_opt = sol.x
    if l_axial_sym: symmetrize_xorbrot(list_orb_rot,x_orbrot_opt,list_degmo)

    obt_opt, tbt_opt = orthogonal_transform_obt_tbt(sol.x,list_orb_rot,obt_spatial,tbt_spatial)

    print(f'\nEnergy lowering of orbital rotation in this round: {sol.fun - E_before_min}')

    return sol.x, sol.fun, obt_opt, tbt_opt

def grad_EGS_of_orbrot(x_orbrot,list_orb_rot,list_list_rdm1,list_list_rdm2,psi_coefs,Enuc,nelec,obt_spatial,tbt_spatial,nparal=1):

    n_param = len(x_orbrot)
    list_grad_comp = []
    l_paral = False
    if nparal > 1: l_paral = True
    if l_paral:
        list_grad_comp = Parallel(n_jobs=nparal)(delayed(grad_comp_EGS_of_orbrot)(ii,x_orbrot,list_orb_rot,list_list_rdm1,list_list_rdm2,psi_coefs,Enuc,nelec,obt_spatial,tbt_spatial) for ii in range(n_param+1))
    else:
        for icomp in range(n_param+1): #icomp = 0 returns the function value, icomp = 1 to n_param returns the gradient components
            component = grad_comp_EGS_of_orbrot(icomp,x_orbrot,list_orb_rot,list_list_rdm1,list_list_rdm2,psi_coefs,Enuc,nelec,obt_spatial,tbt_spatial)
            list_grad_comp.append(component)

    funval = list_grad_comp[0]
    fungrd = np.array(list_grad_comp[1:])

    return funval, fungrd

def grad_comp_EGS_of_orbrot(icomp,x_orbrot,list_orb_rot,list_list_rdm1,list_list_rdm2,psi_coefs,Enuc,nelec,obt_spatial,tbt_spatial,eps=1.0e-4):

    if icomp == 0:
        EGS = EGS_func_of_orbrot_with_CI_coefs(x_orbrot,list_orb_rot,list_list_rdm1,list_list_rdm2,psi_coefs,Enuc,nelec,obt_spatial,tbt_spatial)
        return EGS
    else:
        x_disp = x_orbrot.copy()
        x_disp[icomp-1] = x_orbrot[icomp-1]+eps
        EGS_plus = EGS_func_of_orbrot_with_CI_coefs(x_disp,list_orb_rot,list_list_rdm1,list_list_rdm2,psi_coefs,Enuc,nelec,obt_spatial,tbt_spatial)
        x_disp[icomp-1] = x_orbrot[icomp-1]-eps
        EGS_minus = EGS_func_of_orbrot_with_CI_coefs(x_disp,list_orb_rot,list_list_rdm1,list_list_rdm2,psi_coefs,Enuc,nelec,obt_spatial,tbt_spatial)
        grad_comp = (EGS_plus - EGS_minus) / (2.0 * eps)
        return grad_comp

def EGS_func_of_orbrot_with_CI_coefs(x_orbrot,list_orb_rot,list_list_rdm1,list_list_rdm2,psi_coefs,Enuc,nelec,obt_spatial,tbt_spatial,list_CSF=[]):
    """
    Given 1rdm, 2rdm, and CSF coefficients, return the energy of a state as function of orbital rotational angles
    """

    obt_trans_spin, tbt_trans_spin = orthogonal_transform_obt_tbt(x_orbrot,list_orb_rot,obt_spatial,tbt_spatial)

    nCSF = len(psi_coefs)
    E_state = 0.0
    ilist_rdm = -1
    for iCSF in range(nCSF):
        for jCSF in range(iCSF,nCSF):
            ilist_rdm += 1
            list_rdm1 = list_list_rdm1[ilist_rdm]
            list_rdm2 = list_list_rdm2[ilist_rdm]
            Helm = elm_from_list_rdm(list_rdm1,list_rdm2,Enuc/nelec,obt_trans_spin,tbt_trans_spin)
            E_state += Helm*psi_coefs[iCSF]*psi_coefs[jCSF]
            if iCSF != jCSF: E_state += Helm*psi_coefs[iCSF]*psi_coefs[jCSF]


    return E_state

def EGS_func_of_orbrot(list_list_rdm1,list_list_rdm2,ndim_all_ex,x_orbrot,list_orb_rot,Enuc,nelec,obt_phys_spatial,tbt_phys_spatial,O_rowEX_colUCSF,list_start_end,list_list_ex_space):

    obt_trans_spin, tbt_trans_spin = orthogonal_transform_obt_tbt(x_orbrot,list_orb_rot,obt_phys_spatial,tbt_phys_spatial)


    Hmat_all_ex=np.zeros([ndim_all_ex,ndim_all_ex])
    ilist_rdm = -1
   #for i_ex_bas in range(ndim_all_ex):
   #        for j_ex_bas in range(i_ex_bas,ndim_all_ex):
   #            ilist_rdm += 1
   #            list_rdm1 = list_list_rdm1[ilist_rdm]
   #            list_rdm2 = list_list_rdm2[ilist_rdm]
   #            Helm_from_rdm = elm_from_list_rdm(list_rdm1,list_rdm2,Enuc/nelec,obt_trans_spin,tbt_trans_spin)
   #            Hmat_all_ex[i_ex_bas,j_ex_bas] = Helm_from_rdm
   #            Hmat_all_ex[j_ex_bas,i_ex_bas] = Helm_from_rdm

    nUCSF = O_rowEX_colUCSF.shape[1]
    for iUCSF in range(nUCSF):
        istart = list_start_end[iUCSF][0]
        for jUCSF in range(iUCSF,nUCSF):
            jstart = list_start_end[jUCSF][0]
            for ii, bas_i in enumerate(list_list_ex_space[iUCSF]):
                i_ex_bas = istart + ii
                for jj, bas_j in enumerate(list_list_ex_space[jUCSF]):
                    j_ex_bas = jstart + jj
                    if j_ex_bas < i_ex_bas: continue
                    ilist_rdm += 1
                    list_rdm1 = list_list_rdm1[ilist_rdm]
                    list_rdm2 = list_list_rdm2[ilist_rdm]
                    Helm_from_rdm = elm_from_list_rdm(list_rdm1,list_rdm2,Enuc/nelec,obt_trans_spin,tbt_trans_spin)
                    Hmat_all_ex[i_ex_bas,j_ex_bas] = Helm_from_rdm
                    Hmat_all_ex[j_ex_bas,i_ex_bas] = Helm_from_rdm
               

    Hmat_UCSF = O_rowEX_colUCSF.transpose()@Hmat_all_ex@O_rowEX_colUCSF
   #print(f'Hmat_UCSF in EGS_func_of_orbrot')
   #print_matrix(Hmat_UCSF)
    E_GS, psi_GS = get_ground_state(Hmat_UCSF)

   #print(f'E_GS in EGS_func_of_orbrot: {E_GS}')
    return E_GS

def orthogonal_transform_obt_tbt(x_orbrot,list_orb_rot,obt_spatial,tbt_phys_spatial):
    """
    Given the orbital rotational angles and orbital pairs, transform the obt and tbt of spatial orbitals
    """

    assert len(x_orbrot) == len(list_orb_rot)

    n_spatialmo = obt_spatial.shape[0]
    kappa_mat = np.zeros([n_spatialmo,n_spatialmo])

    for ix, pair in enumerate(list_orb_rot):
        [iorb,jorb] = pair
        kappa_mat[iorb,jorb] = x_orbrot[ix]
        kappa_mat[jorb,iorb] = -x_orbrot[ix]


    Omat = scipy.linalg.expm(kappa_mat)
   #print('\nkappa_mat')
   #print_matrix(kappa_mat)
   #print('\nOmat:')
   #print_matrix(Omat)

    obt_phys_spatial_trans = np.einsum('pq,pa,qb->ab',obt_spatial,Omat,Omat,optimize=True)
    tbt_phys_spatial_trans = np.einsum('pqrs,pa,qb,rc,sd->abcd',tbt_phys_spatial,Omat,Omat,Omat,Omat,optimize=True)

    obt_phys_spin_trans = obt_phys_spatial_to_spin(obt_phys_spatial_trans)
    tbt_phys_spin_trans = tbt_phys_spatial_to_spin(tbt_phys_spatial_trans)

    return obt_phys_spin_trans, tbt_phys_spin_trans

def construct_Hmat_CSFs(list_CSF,Enuc,obt,tbt,lrdm=False):
    list_list_rdm1 = []
    list_list_rdm2 = []
    ndim_CSF = len(list_CSF)
    Hmat_CSF = np.zeros([ndim_CSF,ndim_CSF])
    for iCSF in range(ndim_CSF):
        CSFi = list_CSF[iCSF]
        for jCSF in range(iCSF,ndim_CSF):
            CSFj = list_CSF[jCSF]
            if lrdm:
                Helm, list_rdm1, list_rdm2 = Helm_between_CSFs(Enuc,obt,tbt,CSFi,CSFj,lrdm)
                list_list_rdm1.append(list_rdm1)
                list_list_rdm2.append(list_rdm2)
            else:
                Helm = Helm_between_CSFs(Enuc,obt,tbt,CSFi,CSFj)
            Selm = overlap_CSFs(CSFi,CSFj)
            l_orthonormal = False
            if iCSF == jCSF and np.isclose(Selm,1.0): l_orthonormal = True
            if iCSF != jCSF and np.isclose(Selm,0.0): l_orthonormal = True
            if not l_orthonormal:
                print(f'Non-orthonormal CSFs {iCSF,jCSF,Selm}. Bombing out!')
                print('CSFi')
                print(CSFi)
                print('CSFj')
                print(CSFj)
                sys.exit()
            Hmat_CSF[iCSF,jCSF] = Helm
            Hmat_CSF[jCSF,iCSF] = Helm

    if lrdm:
        return Hmat_CSF, list_list_rdm1, list_list_rdm2
    else:
        return Hmat_CSF

def check_orthonormal_CSFs(list_CSF):
    ndim_CSF = len(list_CSF)
    for iCSF in range(ndim_CSF):
        CSFi = list_CSF[iCSF]
        for jCSF in range(iCSF,ndim_CSF):
            CSFj = list_CSF[jCSF]
            Selm = overlap_CSFs(CSFi,CSFj)
            l_orthonormal = False
            if iCSF == jCSF and np.isclose(Selm,1.0): l_orthonormal = True
            if iCSF != jCSF and np.isclose(Selm,0.0): l_orthonormal = True
            if not l_orthonormal:
                print(f'Non-orthonormal CSFs {iCSF,jCSF,Selm}. Bombing out!')
                print('CSFi:')
                print(CSFi)
                print('CSFj:')
                print(CSFj)
                sys.exit()

def op_to_create_hp(onvec,debug=False):
    """
    Read in a ON vector and return the excitation operator that makes this ON from
    a reference with all lowest-indices spin orbitals being occupied
    """

    occ_spinorb = np.where(onvec == 1.0)[0]
    unocc_spinorb = np.where(onvec == 0.0)[0]
    nelec = len(occ_spinorb)
    hoso = nelec - 1
    if debug: 
        print(occ_spinorb,nelec,hoso)
        print(unocc_spinorb)

   #print(np.where(occ_spinorb > hoso))
    part_spinorb = occ_spinorb[np.where(occ_spinorb > hoso)]
    hole_spinorb = unocc_spinorb[np.where(unocc_spinorb <= hoso)]
   #part_spinorb = np.where(occ_spinorb > hoso)[0]
   #hole_spinorb = np.where(unocc_spinorb <= hoso)[0]
    if debug: print(part_spinorb,hole_spinorb)

    assert len(part_spinorb) == len(hole_spinorb)
    op = FermionOperator.identity()
    cur_on = onvec.copy()
    phase = 1.0
    for ihp, hole in enumerate(hole_spinorb):
        part = part_spinorb[ihp]
       #print(np.where(occ_spinorb > hole and occ_spinorb <= hoso))
       #occ_spinorb[occ_spinorb > hole and occ_spinorb <= hoso]
       #nflip = len(occ_spinorb[np.where((occ_spinorb > hole) & (occ_spinorb <= hoso))])
        nflip = sum(cur_on[:part])
        cur_on[part] = 0.0
        nflip += sum(cur_on[:hole])
        cur_on[hole] = 1.0
        phase *= (-1.0)**nflip
        if debug: print(part,hole,nflip,phase)
        term = ((int(part),1),(int(hole),0))
        op *= FermionOperator(term,1.0)

    op *= phase
    op = normal_ordered(op)
    if debug:
        print('\nExcitation operator that creates the read in ON vec:')
        print(op)

    return op

def screen_CSFs_chain_Helm_with_CSF0(list_CSF,Enuc,obt,tbt,small=1e-4,debug=False):
    """
    Select CSFs based on chain of matrix elements with CSF0.
    Solve the eigenvalue problem of the Hamiltonian matrix of list_CSF
    basis set. Select the eigenstate with the lowest eigenvalue and with
    non-zero amplitude of CSF0. All CSFs that have nonzero amplitudes
    in this state are selected.
    """

    if debug: print('\nIn screen_CSFs_chain_Helm_with_CSF0')

    Hmat = construct_Hmat_CSFs(list_CSF,Enuc,obt,tbt)
   #eigval,eigvec = np.linalg.eigh(Hmat)
    nstate = 10
    Hmat_sparse = csr_matrix(Hmat)
    eigval,eigvec = scipy.sparse.linalg.eigsh(Hmat_sparse,k=nstate,which='SA')
    if debug:
        print('\nEigensolution all CSFs')
        print_eigen_solution(eigval[:nstate],eigvec[:,:nstate])
  
    ndim = len(list_CSF)
    iCSF_interest = 0
    for istate in range(ndim):
        if abs(eigvec[iCSF_interest,istate]) > small:
            break

    if debug:
        print(f'The lowest state with nonzero amplitude of CSF{iCSF_interest} is State {istate}')

    list_l_remove_CSF = [False] * ndim
    n_to_remove = 0
    for iCSF in range(ndim):
        if abs(eigvec[iCSF,istate]) < small: 
           #print(f'To remove {iCSF} with {eigvec[iCSF,istate]} amplitude')
            list_l_remove_CSF[iCSF] = True
            n_to_remove += 1

    print(f'# of CSFs to be removed: {n_to_remove}')
    return list_l_remove_CSF

def construct_list_genmat_from_occ(list_ex_space,list_ia_pairs,debug=False):
    """
    Construct list of generating matrices of Tiiaa^00 based on occupancies.
    Assuming that all CSFs in list_ex_space only differ in dmo
    """

    ndim = len(list_ex_space)
    list_set_dmo = []
    for CSF in list_ex_space:
        list_dmo = dmo_in_SD(CSF[0][0])
        set_dmo = set(list_dmo)
        list_set_dmo.append(set_dmo)

    list_ia_accumul = []
    list_ia_associated_CSFs = []
    for iCSF in range(ndim):
        set_dmo_i = list_set_dmo[iCSF]
        for jCSF in range(iCSF+1,ndim):
            set_dmo_j = list_set_dmo[jCSF]
            set_dmo_diff = set_dmo_i^set_dmo_j
            if len(set_dmo_diff) == 0:
                print(f'Impossible identical dmo sets for CSFs{iCSF} and {jCSF}')
                print(set_dmo_i,set_dmo_j)
                print('Bombing out!')
                sys.exit()
            if len(set_dmo_diff) % 2 == 1:
                print(f'Impossible odd difference between dmo sets for CSFs{iCSF} and {jCSF}')
                print(set_dmo_i,set_dmo_j)
                print('Bombing out!')
                sys.exit()
            if len(set_dmo_diff) > 2: continue
            if list(set_dmo_diff) in list_ia_accumul:
                ia_ind = list_ia_accumul.index(list(set_dmo_diff))
                list_ia_associated_CSFs[ia_ind].append([iCSF,jCSF])
            else:
                list_ia_accumul.append(list(set_dmo_diff))
                list_ia_associated_CSFs.append([[iCSF,jCSF]])

    if debug:
        print('\nlist_set_dmo:')
        for ii,item in enumerate(list_set_dmo):
            print(ii,item)
        for ii, item in enumerate(list_ia_accumul):
            print(f'difference dmo pair: {item}')
            print(list_ia_associated_CSFs[ii])

    
    list_genmat = []
    for ia_pairs in list_ia_pairs:
        genmat = csr_matrix((ndim,ndim))
        print(f'ia_pairs: {ia_pairs}')
        for pair in ia_pairs:
            pair_reversed = copy.deepcopy(pair)
            pair_reversed.reverse()
           #print(f'pair and reversed: {pair,pair_reversed}')
            if pair not in list_ia_accumul and pair_reversed not in list_ia_accumul:
                print(f'Strange! {pair} or {pair_reversed} is not in list_ia_accumul: {list_ia_accumul}')
                print('Bombing out!')
                sys.exit()
                
            elif pair in list_ia_accumul:
                pair_ind = list_ia_accumul.index(pair)
                for CSFpair in list_ia_associated_CSFs[pair_ind]:
                    genmat[CSFpair[1],CSFpair[0]] =  1.0
                    genmat[CSFpair[0],CSFpair[1]] = -1.0
            elif pair_reversed in list_ia_accumul:
                pair_ind = list_ia_accumul.index(pair_reversed)
                for CSFpair in list_ia_associated_CSFs[pair_ind]:
                    genmat[CSFpair[1],CSFpair[0]] =  1.0
                    genmat[CSFpair[0],CSFpair[1]] = -1.0
        if debug:
            print('generating matrix:')
            print(genmat)
        list_genmat.append(genmat)

    return list_genmat

def symmetrize_xorbrot(list_orb_rot,x_orbrot,list_degmo,debug=False):
    """
    Rotations of degenerate orbital pairs should have the same rotational angle.
    This function is to perform such a symmetrization.
    """

    same_xrot_thrsh = 1.0e-4
    assert len(list_orb_rot) == len(x_orbrot)
    for ipair in range(len(list_orb_rot)):
        [iorb1,iorb2] = list_orb_rot[ipair]
        x_rot_i = x_orbrot[ipair]
        for jpair in range(ipair+1,len(list_orb_rot)):
            [jorb1,jorb2] = list_orb_rot[jpair]
            x_rot_j = x_orbrot[jpair]
            if ([iorb1,jorb1] in list_degmo or [jorb1,iorb1] in list_degmo) and \
              ([iorb2,jorb2] in list_degmo or [jorb2,iorb2] in list_degmo):
                print(f'Rotations to be symmetrized: {[iorb1,iorb2],[jorb1,jorb2]}')
                if abs(x_rot_i - x_rot_j) > same_xrot_thrsh:
                    print(f'Warning! Original read-in rotational angles differ > {same_xrot_thrsh}')
                x_rot_sym = 0.5*(x_rot_i + x_rot_j)
                x_orbrot[ipair] = x_rot_sym
                x_orbrot[jpair] = x_rot_sym
