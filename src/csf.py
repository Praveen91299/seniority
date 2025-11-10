"""
CSF state construction related scripts

"""

from seniority.circuits_csf import validate_t_vec, CG
from seniority.src.state import SDState
import numpy as np

def prepare_CSF_SDState(t_vec, N, S, M):
    """
    Build recursive circuit to obtain tapered CSF state on an empty 0 state

    t_vec : Genealogy 
    N: current number of spatial orbitals, recursive step number. N=1 is base case 
    S : total spin
    M : projected spin 
    
    """
    #checks
    assert S >= 0, "Total spin S: {} is not positive".format(S)
    assert S >= abs(M), "Projected spin: {} is larger than total spin S: {}".format(M, S)
    validate_t_vec(t_vec, N)

    if N == 0:
        return SDState({(): 1.0}, 0)

    state = {}

    tN = t_vec[N-1]
    alpha = 1/2
    beta = -1/2

    ### edge cases # single spin
    if N == 1:
        if M == 1/2:
            #10
            state = {(1, 0): 1.0}
        else:
            #01
            state = {(0, 1): 1.0}
        return SDState(state, 2)
    
    if S == 0: assert tN == -1/2, "Total spin change tN: {}, but needs to be negative to reach S: {}".format(tN, S)

    coeff_alpha = CG(S, M, tN, alpha)
    coeff_beta = CG(S, M, tN, beta)
    SDState_alpha = SDState({(1, 0): 1.0}, 2)
    SDState_beta = SDState({(0, 1): 1.0}, 2)

    SDState_new = SDState({}, 2*N)
    if coeff_alpha != 0:
        SDState_alpha_rec = prepare_CSF_SDState(t_vec, N - 1, S - tN, M - alpha)
        SDState_new += coeff_alpha * SDState_alpha.tensor_prod(SDState_alpha_rec)
    if coeff_beta != 0:
        SDState_beta_rec = prepare_CSF_SDState(t_vec, N - 1, S - tN, M - beta)
        SDState_new += coeff_beta * SDState_beta.tensor_prod(SDState_beta_rec)
    return SDState_new

from itertools import combinations

def get_t_vec(N, S):
    """
    Get all genealogical paths towards N spins of total spin S

    """
    #check validity of S, N
    assert S >=0 and N >= 0, "Unphysical total spin and number of spins."
    assert 2*S - int(2*S) == 0, "Total spin S not a multiple of 1/2."
    assert 2*S <= N, "Low N: {} for S: {}".format(N, S)
    assert (2*S - N) % 2 == 0, "N and S of wrong parity."
    

    def validate_t_vec(t_vec, N):
        """
        Check if t vector is valid and of sufficient length

        """
        if len(t_vec) == N == 0:
            return 1
        if not len(t_vec) >= N:
            return 0
        
        if not t_vec[0] == 0.5:
            return 0

        for i in range(len(t_vec)):
            Si = np.sum(t_vec[:i+1])

            if not abs(t_vec[i]) == 0.5:
                return 0
            if not Si >= 0:
                return 0
            if not 0.5*np.floor(Si/0.5) == Si:
                return 0
        return 1
    t_vecs = []
    
    idx = list(range(N))
    combs_up = combinations(idx, int(S + N/2))

    for comb in combs_up:
        t_vec = []
        for i in idx:
            if i in comb:
                t_vec.append(1/2)
            else:
                t_vec.append(-1/2)

        #check physicality
        if validate_t_vec(t_vec, N):
            t_vecs.append(t_vec)
    return t_vecs

def prepare_CSFs(n_orbs, n_elec, active_orbs, seniorities, S, M):
    """
    Returns all CSFs with excitations in active_orbs

    n_orbs: Total number of spatial orbitals
    active_orbs: list of active spatial orbitals
    seniorities: list of total seniority
    S: Total spin 0, 1/2, 1, ...
    M: projected spin Sz, M <= S

    """
    def spatial_to_spin_orb_pair(spatial_orbs):
        spin_orbs = []
        for orb in spatial_orbs:
            spin_orbs.append(2*orb)
            spin_orbs.append(2*orb+1)
        
        return spin_orbs
    
    n_spin_orbs = 2 * n_orbs
    CSF_list = []

    #building all CSFs (unexcited)
    for sen in seniorities:
        t_vecs = get_t_vec(sen, S)
        n_orbs_paired = (n_elec - sen)//2
        n_orb_empty = n_orbs - n_orbs_paired - sen
        unpaired_orbs_list = combinations(active_orbs, sen)

        #fill paired
        hf_SDState = SDState(state = {tuple([1]*(2*n_orbs_paired) + [0]*(2*n_orb_empty)): 1.0}, n_modes = n_spin_orbs - 2*sen)

        for t_vec in t_vecs:
            SD_t_vec = prepare_CSF_SDState(t_vec, sen, S, M) # TODO should we iterate over all M = -S to S??? unclear at the moment

            unpaired_orbs_list = combinations(active_orbs, sen)
            for unpaired_orbs in list(unpaired_orbs_list):
                unpaired_spin_orbs = spatial_to_spin_orb_pair(unpaired_orbs)

                CSF_list.append(hf_SDState.tensor_prod(SD_t_vec, modes_other=unpaired_spin_orbs))
    return CSF_list