import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from seniority.src.circuits.circuits_pair_excitation import PairedExcitationRotation, SymmetricPairedExcitationRotation
from seniority.src.circuits.utils_circuit import show_state
from scipy import sparse as sp
from openfermion import s_squared_operator, get_sparse_operator
import pickle


def get_hf_circuit(n_orb, ne):
    assert ne <= n_orb*2, "Provided number of electrons {} incompatible with {} orbitals.".format(ne, n_orb)

    qc = QuantumCircuit(2*n_orb)
    for i in range(ne):
        qc.x(i)
    return qc

def get_tapered_hf_circuit(n_orb, ne):
    assert ne <= n_orb*2, "Provided number of electrons {} incompatible with {} orbitals.".format(ne, n_orb)

    qc = QuantumCircuit(n_orb)
    for i in range(ne//2):
        qc.x(i)
    if ne%2:
        qc.x(ne//2)
    return qc

def get_tapered_Tia_circuit(i, a, n_orb):
    qc = QuantumCircuit(n_orb)

    qc.x(a)
    qc.h(a)
    qc.cx(control_qubit=a, target_qubit=i)

    return qc

def get_tapered_Stt_circuit(i, j, a, b, n_orb):
    qc = QuantumCircuit(n_orb)

    # takes care of hf state
    qc.x(i)
    qc.x(j)

    qc.h(i)
    qc.ry(theta=(-2)*np.arctan2(1, np.sqrt(2)), qubit=j)
    qc.x(a)
    qc.x(j)
    qc.cx(j, a)
    qc.cx(i, j)
    qc.ch(a, b)
    qc.cx(b, a)
    qc.cx(i, a)
    qc.cx(i, b)
    qc.x(i)

    return qc
 
def get_tapered_Sss_circuit(i, j, a, b, n_orb):
    qc = QuantumCircuit(n_orb)

    qc.append(get_tapered_Tia_circuit(i, a, n_orb).to_gate(), qc.qubits)
    qc.append(get_tapered_Tia_circuit(j, b, n_orb).to_gate(), qc.qubits)

    return qc

def append_tapered_Tia_circuit(qc, i, a):
    """
    Append circuit to prepare 0.707*[|01> - |10>]

    """

    qc.x(i)
    qc.x(a)
    qc.h(a)
    qc.cx(control_qubit=a, target_qubit=i)

    return qc

def append_tapered_Sss_circuit(qc, i, j, a, b):

    append_tapered_Tia_circuit(qc, i, a)
    append_tapered_Tia_circuit(qc, j, b)

def append_tapered_Stt_circuit(qc, i, j, a, b):

    # takes care of hf state
    qc.h(i)
    qc.ry(theta=(-2)*np.arctan2(1, np.sqrt(2)), qubit=j)
    qc.x(a)
    qc.x(j)
    qc.cx(j, a)
    qc.cx(i, j)
    qc.ch(a, b)
    qc.cx(b, a)
    qc.cx(i, a)
    qc.cx(i, b)
    qc.x(i)

def append_cup_csf(qc, i, j, k, a, b, c):
    """
    Circuit for tapered csf of [1/2, 1/2, -1/2, 1/2, -1/2, -1/2]
    
    """
    qc.ry(np.atan2(2*np.sqrt(2), 1), j)
    append_tapered_Stt_circuit(qc, k, a, b, c)
    qc.ch(j, i)
    qc.x(j)
    qc.ccx(j, k, i)
    qc.x(j)
    qc.cx(j, k, ctrl_state=0)
    qc.cx(i, j)
    qc.z(k)

###
"""
Gate efficient controlled using initial state information of 0s - makes it sufficient to control only single qubit gates.

"""
def append_ctrl_init0_tapered_Stt(qc, i, j, a, b, ctrl, ctrl_state=1):
    
    qc.ch(control_qubit=ctrl, target_qubit=i, ctrl_state=ctrl_state)#
    qc.cry(theta=(-2)*np.arctan2(1, np.sqrt(2)), target_qubit=j, control_qubit=ctrl, ctrl_state=ctrl_state)#
    qc.cx(ctrl, a, ctrl_state=ctrl_state)#
    qc.cx(ctrl, j, ctrl_state=ctrl_state)#
    qc.cx(j, a)
    qc.cx(i, j)
    qc.ch(a, b)
    qc.cx(b, a)
    qc.cx(i, a)
    qc.cx(i, b)
    qc.cx(ctrl, i, ctrl_state=ctrl_state)#

def append_ctrl_init0_tapered_Tia(qc, i, a, ctrl, ctrl_state=1):
    qc.cx(ctrl, i, ctrl_state=ctrl_state)#
    qc.cx(ctrl, a, ctrl_state=ctrl_state)#
    qc.ch(ctrl, a, ctrl_state=ctrl_state)#
    qc.cx(control_qubit=a, target_qubit=i)

def append_ctrl_init0_tapered_Sss(qc, i, j, a, b, ctrl, ctrl_state=1):
    append_ctrl_init0_tapered_Tia(qc, i, a, ctrl, ctrl_state=ctrl_state)
    append_ctrl_init0_tapered_Tia(qc, j, b, ctrl, ctrl_state=ctrl_state)

###

def CG(S, M, t, sigma):
    """
    Returns Clebsch Gordon coefficient for
    S - total spin
    M - projected spin
    t - change in total spin, t_n - t_{n-1}, provides information about the Genealogical path.
    sigma - added electron projected/total spin
    
    """
    assert abs(M) <= abs(S)
    
    if t == (1/2):
        assert S != 0, "S cannot be 0 if t is +ve"
            
        return np.sqrt((S + 2*sigma*M)/(2*S))
    if t == (-1/2):
        return -2*sigma*np.sqrt((S + 1 - 2*sigma*M)/(2*(S+1)))

def validate_t_vec(t_vec, N):
    """
    Check if t vector is valid and of sufficient length

    """

    if N == 0:
        assert len(t_vec) == 0
        return
    assert len(t_vec) >= N #ensures the number of spins N+1 does not exceed available information
    assert t_vec[0] == 0.5, 'Invalid first total spin S1: {}'.format(t_vec[0])

    for i in range(len(t_vec)):
        Si = np.sum(t_vec[:i+1])

        assert abs(t_vec[i]) == 0.5, 'Invalid change in total spin, Si - Si-1: '.format(t_vec[i])
        assert Si >= 0, 'Total spin Si: {} should be positive'.format(Si)
        assert 0.5*np.floor(Si/0.5) == Si, 'Total spin Si: {} should be a multiple of 1/2'.format(Si)
    return

def build_recursive_CSF_circuit(t_vec, N, S, M):
    """
    Build recursive circuit to obtain CSF state on an empty state

    S : total spin
    M : projected spin 
    t_vec : Genealogy 
    N: current number of spins, recursive step number. N=1 is base case 
    
    """
    #checks
    assert S >= 0, "Total spin S: {} is not positive".format(S)
    assert S >= abs(M), "Projected spin: {} is larger than total spin S: {}".format(M, S)
    validate_t_vec(t_vec, N)   

    qc = QuantumCircuit(2*N)

    tN = t_vec[N-1]
    alpha = 1/2
    beta = -1/2

    ### edge cases
    if N == 1:
        # single spin
        if M == 1/2:
            qc.x(0)
        elif M == -1/2:
            qc.x(1)
        return qc
    
    if S == 0: assert tN == -1/2, "Total spin change tN: {}, but needs to be negative to reach S: {}".format(tN, S)

    coeff_alpha = CG(S, M, tN, alpha)
    coeff_beta = CG(S, M, tN, beta)
    qubits_N_1 = qc.qubits[:-2]
    qubit_alpha = qc.qubits[-2]
    qubit_beta = qc.qubits[-1]

    theta = 2*np.arctan2(coeff_alpha, coeff_beta)

    qc.ry(theta, qubit=qubit_alpha)
    qc.cx(qubit_alpha, qubit_beta)
    qc.x(qubit_beta)

    if coeff_alpha != 0:
        Ua = build_recursive_CSF_circuit(t_vec, N-1, S - tN, M - alpha)
        qc.append(Ua.to_gate().control(num_ctrl_qubits=1),  [qubit_alpha]+qubits_N_1)
    
    if coeff_beta != 0:
        Ub = build_recursive_CSF_circuit(t_vec, N-1, S - tN, M - beta)
        qc.append(Ub.to_gate().control(num_ctrl_qubits=1, ctrl_state=0),  [qubit_alpha]+qubits_N_1)
        
    return qc


def build_tapered_recursive_CSF_circuit(t_vec, N, S, M):
    """
    Build recursive circuit to obtain tapered CSF state on an empty 0 state

    S : total spin
    M : projected spin 
    t_vec : Genealogy 
    N: current number of spins, recursive step number. N=1 is base case 
    
    """
    #checks
    assert S >= 0, "Total spin S: {} is not positive".format(S)
    assert S >= abs(M), "Projected spin: {} is larger than total spin S: {}".format(M, S)
    validate_t_vec(t_vec, N)   

    qc = QuantumCircuit(N)

    tN = t_vec[N-1]
    alpha = 1/2
    beta = -1/2

    ### edge cases
    if N == 1:
        # single spin
        if M == 1/2:
            qc.x(0)
        return qc
    
    if S == 0: assert tN == -1/2, "Total spin change tN: {}, but needs to be negative to reach S: {}".format(tN, S)

    coeff_alpha = CG(S, M, tN, alpha)
    coeff_beta = CG(S, M, tN, beta)
    qubits_N_1 = qc.qubits[:-1]
    qubit_N = qc.qubits[-1]

    theta = 2*np.arctan2(coeff_alpha, coeff_beta)

    qc.ry(theta, qubit=qubit_N)

    if coeff_alpha != 0:
        Ua = build_tapered_recursive_CSF_circuit(t_vec, N-1, S - tN, M - alpha)
        qc.append(Ua.to_gate().control(num_ctrl_qubits=1),  [qubit_N]+qubits_N_1)
    
    if coeff_beta != 0:
        Ub = build_tapered_recursive_CSF_circuit(t_vec, N-1, S - tN, M - beta)
        qc.append(Ub.to_gate().control(num_ctrl_qubits=1, ctrl_state=0),  [qubit_N]+qubits_N_1)
        
    return qc

def append_x_occ(qc, occ, target_qubits):
    assert len(qc.qubits) >= len(occ), "Insufficient number of qubits."

    for idx, i in enumerate(occ):
        if i == 1:
            qc.x(target_qubits[idx])

def append_cx_occ(qc, occ, target_qubits, ctrl_qubit, ctrl_state=1):
    assert len(qc.qubits) >= len(occ), "Insufficient number of qubits."

    for idx, i in enumerate(occ):
        if i == 1:
            qc.cx(ctrl_qubit, target_qubits[idx], ctrl_state=ctrl_state)

def get_state_from_csf_data(csf_state, orbitals):
    spin_orbs = []
    for o in orbitals:
        spin_orbs.append(2*o)
        spin_orbs.append(2*o + 1)
    
    rows = []

    get_int = lambda sd: np.sum([val * (2**(len(sd) - 1 - i)) for i, val in enumerate(sd)])

    for SD in csf_state[0]:
        trunc_SD = SD[spin_orbs]
        rows.append(get_int(trunc_SD))
    
    arr = sp.csr_matrix((csf_state[2], (rows, [0]*len(rows))), shape=(2**len(spin_orbs), 1), dtype=complex)

    return arr

def get_t_vec(state, n_orb):
    """
    Get t vector from state by taking difference of successive S values

    """
    get_s = lambda ss: -0.5 + 0.5*np.sqrt(1 + 4*ss)

    s_list = [get_s((state.T @ (get_sparse_operator(s_squared_operator(i), 2*n_orb)) @ state).toarray()[0, 0]) for i in range(n_orb+1)]
    t_vec = np.array([s_list[i+1] - s_list[i] for i in range(n_orb)])
    t_vec = np.array(np.rint(2*t_vec), int)
    t_vec = t_vec/2
    return list(t_vec)

def determine_t_vec(csf_state, orbitals):
    """
    Determine the genealogy vector from the csf state [SD vectors, int of SD, coeff], over specified orbitals

    csf_state: list[list[array], list[int], array] - list of SDs and coefficient array for the CSF
    orbitals: list[int] - singly occupied orbitals involved in singlet creation (SOMOs)
    
    """

    arr = get_state_from_csf_data(csf_state=csf_state, orbitals=orbitals)
    return get_t_vec(arr, len(orbitals))

from typing import Dict
def get_tapered_state_from_UCSF(UCSF):
    """
    Returns state as dictionary of {bit_string: coeff}

    """
    def make_bin_str(arr):

        return ''.join(map(str, np.array(arr, int)))
    
    strings = [make_bin_str(state[::2]) for state in UCSF[0]]
    state: Dict[str, complex] = {}
    for i, string in enumerate(strings):
        state[string] = UCSF[2][i]
    
    return state

def compare_states(s1, s2, tol=1e-5):
    """
    compare UCSF states elementwise, upto tol

    """
    
    for k, v in zip(s1.keys(), s1.values()):
        if abs(v) >= tol:
            if k not in s2.keys():
                return False
            
            diff = abs(v - s2[k])
            if diff >= tol:
                return False

    return True

def find_sen_orb(state, target_seniority=2):
    """
    Find doubly occupied orbitals from 2n_o qubit state, provided as a set of SD and coefficients, etc
    
    """

    SDs = state[0]
    SD = SDs[0]

    orb_sen = SD[::2] + SD[1::2]
    doub_occ = np.array(orb_sen == target_seniority, int)

    orbitals = [i for i in range(len(doub_occ)) if doub_occ[i]]
    
    return doub_occ, orbitals


def fill_doubly_occ(ne, n_orb, somo):
    """
    Fill upto ne electrons in n_orb orbitals, avoiding somo

    """
    #fill up from orb 0
    excess_elec = ne - len(somo)
    to_add = np.ceil(excess_elec/2)

    d_occ = np.array([0]*n_orb)

    if excess_elec == 0:
        return d_occ

    added = 0
    for i in range(n_orb):

        if i not in somo:
            d_occ[i] = 1
            added += 1
        
        if added == to_add:
            return d_occ
    
    raise ValueError('Insufficient orbitals {} for {} electrons and {} singly occupied orbitals'.format(n_orb, ne, len(somo)))

def get_csfs_from_dump(input_file, verify_states = False, verbose=True):

    with open(input_file,'rb') as f:
        list_CSF,list_list_ia_CSF,list_list_theta_CSF,list_sym_CSF_vec,list_UCSF_tz,list_UCSF_smik,\
        list_list_SOMO_UCSF_smik,psi_GS_UCSF_smik,list_orb_rot,x_orbrot,Enuc,obt_spatial,tbt_spatial = pickle.load(f)
    
    n_orb=len(tbt_spatial)
    if verbose: print(n_orb)
    ne = sum(list_CSF[0][0][0])
    csfs = []

    for iCSF in range(len(list_CSF)):
        list_ia = list_list_ia_CSF[iCSF]
        orbitals = list_list_SOMO_UCSF_smik[iCSF]

        if len(list_ia) == 0:
            
            excitations = []
        else:
            list_theta = list_list_theta_CSF[iCSF]
            n_U_train = len(list_theta) // len(list_ia)

            excitations = []

            for i_train in range(n_U_train):

                for ipair in range(len(list_ia)):

                    exc = list_ia[ipair]
                    theta = list_theta[ipair+i_train*len(list_ia)]
                    
                    if len(exc) == 1:
                        excitations.append(PairedExcitationRotation(exc, theta, n_orb)) ### note the -ve
                    elif len(exc) == 2:
                        if len(set.intersection(set(exc[0]), set(exc[1]))) == 1:
                            excitations.append(SymmetricPairedExcitationRotation(exc, theta, n_orb))
                        else:
                            excitations.append(PairedExcitationRotation([exc[0]], theta, n_orb))
                            excitations.append(PairedExcitationRotation([exc[1]], theta, n_orb))
        
        csf_state = list_CSF[iCSF]
        t_vec = determine_t_vec(list_CSF[iCSF], orbitals)
        doub_occ, domo = find_sen_orb(csf_state, 2)
        sing_occ, somo = find_sen_orb(csf_state, 1)

        csf = CSF(t_vec, orbitals=somo, n_orb=n_orb, ne = ne, excitations=excitations, doub_occ=doub_occ)
        csfs.append(csf)
    
    if verify_states:
        states = [get_tapered_state_from_UCSF(UCSF) for UCSF in list_UCSF_smik]
        states_from_circuit = [show_state(csf.get_tapered_full_circuit()) for csf in csfs]
        assert len(states) == len(states_from_circuit)

        checks = [compare_states(s1, s2) for s1, s2 in zip(states, states_from_circuit)]
        if all(checks):
            if verbose: print("GET CSFS FROM DUMP: states prepared and verified from dump file.")
        else:
            if verbose: print(checks)
            raise Exception("GET CSFS FROM DUMP: Prepared CSF states do not match.")

    return csfs

def get_Uext_csfs_from_dump(input_file, verify_states=False, use_opt_amplitudes=True, verbose =True):
    """
    Import, construct, and verify CSF class objects for Uext formalism

    input_file (str)
    
    """

    with open(input_file,'rb') as f:
        list_list_refCSF,list_list_Uext_mp2_CSF,list_list_Uext_mp2_ampld,list_list_Uext_opt_ampld,list_orb_rot,x_orbrot,Enuc,obt_spatial,tbt_spatial = pickle.load(f)
    
    n_orb = len(tbt_spatial)
    if verbose: print(f"Importing CSFs from {input_file}\nOrbitals: {n_orb}")
    ne = int(sum(list_list_refCSF[0][0][0][0]))
    csfs = []

    if use_opt_amplitudes:
        if verbose: print("Using optimized excitation amplitudes...")
        list_list_amplitudes = list_list_Uext_opt_ampld
    else:
        if verbose: print("Using MP2 excitation amplitudes...")
        list_list_amplitudes = list_list_Uext_mp2_ampld
    
    for i_refcsf, list_refCSF in enumerate(list_list_refCSF):
        #particular csf group
        list_excitations = list_list_amplitudes[i_refcsf]

        if len(list_excitations) == 0:
            excitations = []
        else:
            excitations = []
            for excitation in list_excitations:

                exc = excitation[0]
                theta = excitation[1]

                if len(exc) == 1:
                    excitations.append(PairedExcitationRotation(exc, theta, n_orb))
                elif len(exc) == 2:
                    if len(set.intersection(set(exc[0]), set(exc[1]))) == 1:
                        excitations.append(SymmetricPairedExcitationRotation(exc, theta, n_orb))
                    else:
                        excitations.append(PairedExcitationRotation([exc[0]], theta, n_orb))
                        excitations.append(PairedExcitationRotation([exc[1]], theta, n_orb))
        
        for j, state in enumerate(list_refCSF):
            doub_occ, domo = find_sen_orb(state, 2)
            sing_occ, somo = find_sen_orb(state, 1)
            t_vec = determine_t_vec(state, somo)

            csf = CSF(t_vec, orbitals=somo, n_orb=n_orb, ne = ne, excitations=excitations, doub_occ=doub_occ)
            csfs.append(csf)

    ###verify
    if verify_states:

        if use_opt_amplitudes:
            if verbose: print("WARNING: State verification currently not available. CSF objects not verified.")
        else:
            states = []
            for i, list_mp2_UCSF in enumerate(list_list_Uext_mp2_CSF):
                for j, mp2_UCSF in enumerate(list_mp2_UCSF):
                    states.append(get_tapered_state_from_UCSF(mp2_UCSF))
            
            states_from_circuit = [show_state(csf.get_tapered_full_circuit()) for csf in csfs]
            assert len(states) == len(states_from_circuit)

            checks = [compare_states(s1, s2) for s1, s2 in zip(states, states_from_circuit)]
            if all(checks):
                if verbose: print("GET CSFS FROM DUMP: states prepared and verified from dump file.")
            else:
                if verbose: print(checks)
                raise Exception("GET CSFS FROM DUMP: Prepared CSF states do not match.")
    
    return csfs

from seniority.src.circuits.circuits_pair_excitation import append_tapered_exc_rot
from seniority.src.circuits.utils_circuit import qubit_index

class CSF:
    """
    Class to store CSF object, their corresponding excitation rotations, and retrieve normal and tapered circuits

    t_vec : list[float] - genealogy path, consisting of +/- 1/2 of length N=len(orbitals) (Note S, M = 0 always (siglet))
    orbitals : list[int] - orbitals involved in singlet state (unpaired orbitals)
    n_orb : Total number of spatial orbitals
    ne : number of electrons
    excitations: list[PairedExcitationRotation] - 


    
    """
    def __init__(self, t_vec, orbitals, n_orb, ne, excitations : list[PairedExcitationRotation] = [], doub_occ = None):
        self.n_orb = n_orb
        self.t_vec = t_vec
        self.ne = ne
        self.t_vec = t_vec

        assert self.get_num_targ_orb() == len(orbitals), 'Incorrect number of orbitals in {} for genealogy vector: {}'.format(orbitals, self.t_vec)
        self.orbitals = orbitals
        self.excitations = excitations

        if doub_occ is not None:
            self._doub_occ = doub_occ
        else:
            self._doub_occ = fill_doubly_occ(ne = self.ne, n_orb = self.n_orb, somo = self.orbitals) # lowest energy orbitals that are not SOMO
    
    def get_excitations(self) ->list[PairedExcitationRotation] :
        return self.excitations
    
    def get_num_targ_orb(self):
        return len(self.t_vec)
    
    def get_tapered_csf_circuit(self, add_double_occ = True)->QuantumCircuit:
        qc = QuantumCircuit(self.n_orb) # 1 for control

        if add_double_occ:
            occ = self.get_doubly_occ_orbitals()
            append_x_occ(qc, occ, qc.qubits)
        
        if self.t_vec == []:
            return qc
        elif self.t_vec == [1/2, -1/2]:
            i, a = self.orbitals

            append_tapered_Tia_circuit(qc, i, a)
        elif self.t_vec == [1/2, 1/2, -1/2, -1/2]:
            i, j, a, b = self.orbitals

            append_tapered_Stt_circuit(qc, i, j, a, b)
        elif self.t_vec == [1/2, -1/2, 1/2, -1/2]:
            i, a, j, b = self.orbitals
            assert i < j and a < b, 'i: {} j: {} a: {} b: {}'.format(i, j, a, b)

            append_tapered_Sss_circuit(qc, i, j, a, b)
        else:
            ### general case
            # redo hf

            N = self.get_num_targ_orb()
            S = 0
            M = 0
            qc.append(build_tapered_recursive_CSF_circuit(self.t_vec, N, S, M).to_gate(), self.orbitals)
        
        return qc
    
    def get_doubly_occ_orbitals(self):
        """
        Return array with 1 at doubly occupied spatial orbitals

        """
        if self._doub_occ is not None:
            # if initialized with doubly occ
            return self._doub_occ
        else:
            return fill_doubly_occ(self.ne, self.n_orb, self.orbitals)

    def get_tapered_full_circuit(self):
        qc = self.get_tapered_csf_circuit(True)
        for exc in self.excitations:
            exc.append_tapered_circuit(qc, qc.qubits)
        
        return qc

    def append_ctrl_tapered_init0_csf_circuit(self, qc, target_qubits, ctrl_qubit, ctrl_state=1, add_double_occ = True):
        if add_double_occ:
            occ = self.get_doubly_occ_orbitals()
            append_cx_occ(qc, occ, target_qubits, ctrl_qubit=ctrl_qubit, ctrl_state=ctrl_state)
        
        if self.t_vec == []:
            return qc
        elif self.t_vec == [1/2, -1/2]:
            i, a = self.orbitals
            iq, aq = target_qubits[i], target_qubits[a]

            append_ctrl_init0_tapered_Tia(qc, iq, aq, ctrl_qubit, ctrl_state)
        elif self.t_vec == [1/2, 1/2, -1/2, -1/2]:
            i, j, a, b = self.orbitals
            iq, jq, aq, bq = target_qubits[i], target_qubits[j], target_qubits[a], target_qubits[b]

            append_ctrl_init0_tapered_Stt(qc, iq, jq, aq, bq, ctrl_qubit, ctrl_state)
        elif self.t_vec == [1/2, -1/2, 1/2, -1/2]:
            i, a, j, b = self.orbitals
            iq, jq, aq, bq = target_qubits[i], target_qubits[j], target_qubits[a], target_qubits[b]

            assert i < j and a < b, 'i: {} j: {} a: {} b: {}'.format(i, j, a, b)

            append_ctrl_init0_tapered_Sss(qc, iq, jq, aq, bq, ctrl_qubit, ctrl_state)
        else:
            ### general case
            # redo hf
            raise "Unknown T vector, no control gate defined"
            #TODO recursive controlled with only single qubit gates controlled.

            # N = self.get_num_targ_orb()
            # S = 0
            # M = 0
            # qc.append(build_tapered_recursive_CSF_circuit(self.t_vec, N, S, M).to_gate(), self.orbitals)

    def append_ctrl_tapered_init0_full_circuit(self, qc, target_qubits, ctrl_qubit, ctrl_state=1):
        self.append_ctrl_tapered_init0_csf_circuit(qc, target_qubits=target_qubits, ctrl_qubit=ctrl_qubit, ctrl_state=ctrl_state, add_double_occ=True)

        for exc in self.excitations:
            exc.append_tapered_circuit(qc, target_qubits)
    
    def append_ctrl_tapered_init0_traced_csf_circuit(self, qc, target_qubits, quantum_indices, ctrl_qubit, ctrl_state=1, add_double_occ=True):
        if add_double_occ:
            occ = self.get_doubly_occ_orbitals()

            #only in the 
            for idx, i in enumerate(occ):
                if i == 1 and idx in quantum_indices:
                    iq = qubit_index(idx, quantum_indices)
                    qc.cx(ctrl_qubit, target_qubits[iq], ctrl_state=ctrl_state)
        
        if self.t_vec == []:
            return qc
        
        assert all([i in quantum_indices for i in self.orbitals]) or all([i not in quantum_indices for i in self.orbitals]), "Quantum indices do not contain all or none of somos"

        if all([i in quantum_indices for i in self.orbitals]):
            #add controlled CSF circuit
            
            if self.t_vec == [1/2, -1/2]:
                i, a = self.orbitals
                iq, aq = target_qubits[qubit_index(i, quantum_indices)], target_qubits[qubit_index(a, quantum_indices)]

                append_ctrl_init0_tapered_Tia(qc, iq, aq, ctrl_qubit, ctrl_state)
            elif self.t_vec == [1/2, 1/2, -1/2, -1/2]:
                i, j, a, b = self.orbitals
                iq = target_qubits[qubit_index(i, quantum_indices)]
                jq = target_qubits[qubit_index(j, quantum_indices)]
                aq = target_qubits[qubit_index(a, quantum_indices)]
                bq = target_qubits[qubit_index(b, quantum_indices)]

                append_ctrl_init0_tapered_Stt(qc, iq, jq, aq, bq, ctrl_qubit, ctrl_state)
            elif self.t_vec == [1/2, -1/2, 1/2, -1/2]:
                i, a, j, b = self.orbitals
                iq = target_qubits[qubit_index(i, quantum_indices)]
                jq = target_qubits[qubit_index(j, quantum_indices)]
                aq = target_qubits[qubit_index(a, quantum_indices)]
                bq = target_qubits[qubit_index(b, quantum_indices)]
                
                assert i < j and a < b, 'i: {} j: {} a: {} b: {}'.format(i, j, a, b)

                append_ctrl_init0_tapered_Sss(qc, iq, jq, aq, bq, ctrl_qubit, ctrl_state)
    
    def append_ctrl_tapered_init0_traced_full_circuit(self, qc, target_qubits, ctrl_qubit, quantum_indices,  ctrl_state=1):
        """
        append ctrl tapered circuits on subset of qubits given by quantum_qubits
        
        """
        self.append_ctrl_tapered_init0_traced_csf_circuit(qc, target_qubits=target_qubits, quantum_indices=quantum_indices, ctrl_qubit=ctrl_qubit, ctrl_state=ctrl_state, add_double_occ=True)

        for exc in self.excitations:
            i, a = self.get_indices()
            iq, aq = qubit_index(i, quantum_indices), qubit_index(a, quantum_indices) #position in the list of quantum qubits
            append_tapered_exc_rot(qc, target_qubits[iq], target_qubits[aq], self.get_theta())

    def get_parity_string(self):

        parity = np.zeros(self.n_orb)

        for i in self.orbitals:
            parity[i] = 1

        return parity