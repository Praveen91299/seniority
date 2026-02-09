"""
Code developed by Toby Zeng, 2025

"""

import sys, scipy, os
sys.path.append('../')
from scipy.sparse import csr_matrix
import numpy as np
import pickle, copy
from openfermion import (
    FermionOperator, hermitian_conjugated, normal_ordered,get_ground_state
)
from itertools import combinations
from src.util_CSF_and_UCSF import *
from src.ferm_utils import (
    get_on_idx,
    op_action_tz,
    op_action_tz_remove_0coef,
    braket_tz
)

def run_qsense(filnam_pyscf_phys_spatial, mp2_ampld_thrsh=0.0, nparal=1, actmo_start=2, actmo_end=9, rdist=1, Ethrsh_select_ia=0.0, Uopt_thrsh=1e-5, text_initial_orb_rot=False, text_opt_orb=True, internal_mo_start=2, internal_mo_end=9, list_seniority=[0, 2, 4], dumpfile_prefix=None):
    """
    Run VO and PT methods of QSENSE

    Saves data into dump files

    Inputs:

    
    """
    
    #Parameters section
    l_remove_CSF_with_0_Hmat_with_CSF0 = False
    l_remove_UCSF_with_0_Hmat_with_UCSF0 = False
    l_remove_CSF_with_small_amplitudes = False
    l_remove_CSF_chain_0_Hmat_with_CSF0 = True
    l_opt_orb = True
    l_initial_orb_rot = False
    l_select_ia_based_on_CSF_E0 = True
    l_pairex_within_actmo = False
    l_use_decomp_genmat = True
    l_include_ia_in_CAS = True
    l_no_gen_groupping = True
    l_no_sym = True
    nparal = 1

    l_no_gen_groupping = l_no_sym

        
    #nparal = int(sys.argv[3])
    if nparal <= 0:
        print(f'Non positive number of processes detected, nparal: {nparal}. Bombing out!')
        sys.exit()
    #Threshold for removing some mp2 amplitudes
    #mp2_ampld_thrsh = float(sys.argv[2])
    #list_seniority = [0,2,4]
    #list_seniority = [0,4]
    #list_seniority = []
    
    #actmo_start  = int(sys.argv[4])
    #actmo_end    = int(sys.argv[5])
    S_by2 = 0 #manually set
    #rdist = float(sys.argv[6])

    small_orbrot_angle = 0.01
    orbopt_conv_thrsh = 1.0e-4
    # Ethrsh_select_ia = float(sys.argv[7])
    # Uopt_thrsh = float(sys.argv[8])
    # text_initial_orb_rot = sys.argv[9]
    if text_initial_orb_rot == 'True': l_initial_orb_rot = True
    # text_opt_orb = sys.argv[10]
    if text_opt_orb == 'False': l_opt_orb = False

    #actmos are for creating all possible SOMOs.
    #internal mos are for defining pair excited CSFs to be linearly combined, instead of Uext perturbed.
    # internal_mo_start = actmo_start
    # internal_mo_end = actmo_end
    # internal_mo_start = int(sys.argv[11])
    # internal_mo_end   = int(sys.argv[12])

    #list_mo_exclud = [18,19,22,23]
    list_mo_exclud = []


    #list_orb_rot = []
    #for ii, iorb in enumerate(list_sigma_mo):
    #    for jorb in list_sigma_mo[ii+1:]:
    #        list_orb_rot.append([iorb,jorb])

    #for ii, iorb in enumerate(list_pi_mo):
    #    for jorb in list_pi_mo[ii+1:]:
    #        list_orb_rot.append([iorb,jorb])

    #list_orb_rot = [ [2, 4], [3, 9], [12, 16], [12, 18], [12, 27], [13, 16], [13, 17], [13, 18], [13, 27], [16, 17], [16, 18]]
    #list_orb_rot += [ [17, 18], [17, 27], [18, 27], [5, 6], [7, 8], [10, 19], [11, 20], [14, 15], [14, 19]]
    #list_orb_rot += [ [14, 20], [14, 25], [14, 26], [15, 19], [15, 20], [15, 25], [15, 26], [19, 20], [19, 25], [19, 26], [20, 25], [20, 26], [25, 26]]


    #Parameters section done

    #moltag = sys.argv[1]
    # filnam_pyscf_phys_spatial = sys.argv[1]

    #Print out the key parameters

    if dumpfile_prefix is None:
        dumpfile_prefix = filnam_pyscf_phys_spatial
    
    print(f'\nHamiltonian read from {filnam_pyscf_phys_spatial}')
    print(f'MP2 amplitude threshold to pre-screen pair excitations: {mp2_ampld_thrsh}')
    if l_select_ia_based_on_CSF_E0:
        print(f'ia pairs are selected based on their capabilities in lowering E0')
    print(f'Active orbitals from and including {actmo_start} to {actmo_end}')
    if l_opt_orb:
        print(f'Orbital optimization will be carried out with the convergence threshold of {orbopt_conv_thrsh}')
    else:
        print('No orbital optimization will be carried out')
    if l_initial_orb_rot:
        print(f'Initial orbital rotation is to be carried out')
    print(f'Energh threshold for selecting ia pairs in E0 lowering scheme: {Ethrsh_select_ia}')
    print(f'Convergence threshold for U optimization including pair excitation within CAS: {Uopt_thrsh}')
    if nparal > 1:
        print(f'The calculation will be done in parallel with {nparal} processes.')
    if len(list_seniority) == 0:
        print(f'There is no restriction on seniority selection')
    else:
        print(f'Only CSFs of the following seniority are selected: {list_seniority}')

    with open(filnam_pyscf_phys_spatial,'rb') as f:
        Enuc, obt_spatial, tbt_spatial, orbene, nelec, nactmo, nactel = pickle.load(f)
    if l_use_decomp_genmat:
        print('Analytical U matrices with decomposed generating matrices will be used')
    else:
        print('Numerical U matrices from scipy.linalg.expm will be used')
    if l_remove_CSF_chain_0_Hmat_with_CSF0:
        print('Original CSFs that do not appear in the same eigenstates with CSF0 will be removed')
        print('They do not have the same IRREPs with CSF0')
    print('\nThe following orbitals will be excluded from reference CSF construction')
    print(list_mo_exclud)
    print('\nThe following orbitals will be excluded from internal excitations')
    print(list_mo_exclud)

    print("")

    n_spatialmo = obt_spatial.shape[0]
    n_spinmo = 2*n_spatialmo

    #If read-in orbital energies contain doubly degeneracy, axial symmetry is detected.
    #Symmetry-adapted CSFs will be generated.
    list_degmo = []
    l_axial_sym = False
    for iorb in range(n_spatialmo-1):
        if np.isclose(orbene[iorb],orbene[iorb+1]):
            l_axial_sym = True
            list_degmo.append([iorb,iorb+1])

    if l_axial_sym:
        print('Axial symmetry detected, with the following pairs of doubly degenerate orbitals:')
        for degmo_pair in list_degmo:
            print(degmo_pair)

    if l_no_sym:
        print(f'\nAlthough axial symmetri is detected, no symmetry consideration is applied, given the hard put-in l_no_sym: {l_no_sym}')
        l_axial_sym = False

    list_orb_rot_filnam = 'list_orb_rot_' + str(rdist)

    list_orb_rot = []
    if os.path.isfile(list_orb_rot_filnam):
        f = open(list_orb_rot_filnam, 'r')
        file_data = f.readlines()
        line_list_orb_rot_start = -1
        line_list_orb_rot_end   = -1
        line_ini_orb_rot_start = -1
        line_ini_orb_rot_end   = -1
        for iline in range(len(file_data)):
        #if 'list_orb_rot_start' in file_data[iline] : line_list_orb_rot_start = iline + 1
            if file_data[iline].startswith("list_orb_rot_start"): line_list_orb_rot_start = iline + 1
            if file_data[iline].startswith("list_orb_rot_end"): line_list_orb_rot_end = iline - 1
            if file_data[iline].startswith("list_initial_rotation_start"): line_ini_orb_rot_start = iline + 1
            if file_data[iline].startswith("list_initial_rotation_end"): line_ini_orb_rot_end = iline - 1
        print(f'line_list_orb_rot_start: {line_list_orb_rot_start}')
        print(f'line_list_orb_rot_end  : {line_list_orb_rot_end}')
        if line_list_orb_rot_start >= line_list_orb_rot_end or line_list_orb_rot_start <= 0 or line_list_orb_rot_end <= 0:
            print(f'\n All orbital pair rotations are allowed')
            for imo in range(n_spatialmo):
                for jmo in range(imo+1,n_spatialmo):
                    list_orb_rot.append([imo,jmo])
        else:
            n_orb_rot_list = line_list_orb_rot_end - line_list_orb_rot_start + 1
            print(f'# of orb rot list detected: {n_orb_rot_list}')
            for iline in range(line_list_orb_rot_start,line_list_orb_rot_end+1):
                list_orb_text = file_data[iline].split('[')[1].split(']')[0].split(',')
                list_orb_ind = [int(i) for i in list_orb_text]
            #print(list_orb_ind) 
                for iimo, imo in enumerate(list_orb_ind):
                    for jmo in list_orb_ind[iimo+1:]:
                        list_orb_rot.append([imo,jmo])
        if l_initial_orb_rot:
            print(f'\nInitialize orbital rotation')
            if line_ini_orb_rot_start>= line_ini_orb_rot_end or line_ini_orb_rot_start <= 0 or line_ini_orb_rot_end <=0:
                print(f'Check {list_orb_rot_filnam}. The initial orbital rotations cannot be read. Bombing out!')
                sys.exit()
            list_initial_rotation = []
            x_orbrot_0 = np.zeros(len(list_orb_rot))
            for iline in range(line_ini_orb_rot_start,line_ini_orb_rot_end+1):
                orb1, orb2, x_rot = int(file_data[iline].split()[0]),int(file_data[iline].split()[1]),float(file_data[iline].split()[2])
                list_initial_rotation.append([[orb1,orb2],x_rot])
        #Symmetrize the rotations for degenerate pairs
        #same_xrot_thrsh = 1.0e-4
        #for iitem, item in enumerate(list_initial_rotation):
        #    [[iorb1,iorb2],x_rot_i] = item
        #    for jtem in list_initial_rotation[iitem+1:]:
        #        [[jorb1,jorb2],x_rot_j] = jtem
        #        if ([iorb1,jorb1] in list_degmo or [jorb1,iorb1] in list_degmo) and \
        #          ([iorb2,jorb2] in list_degmo or [jorb2,iorb2] in list_degmo):
        #            print(f'Rotations to be symmetrized: {item,jtem}')
        #            if abs(x_rot_i - x_rot_j) > same_xrot_thrsh:
        #                print(f'Original read-in rotational angles differ > {same_xrot_thrsh}')
        #                print(f'Please check. Bombing out!')
        #                sys.exit()
        #            x_rot_sym = 0.5*(x_rot_i + x_rot_j)
        #            item[1] = x_rot_sym
        #            jtem[1] = x_rot_sym
        #print(f'\nlist_initial_rotation: {list_initial_rotation}')
            for item in list_initial_rotation:
                if item[0] in list_orb_rot:
                #print(f'Found item[0]')
                    x_ind = list_orb_rot.index(item[0])
                    x_orbrot_0[x_ind] = item[1]
            
            if l_axial_sym: symmetrize_xorbrot(list_orb_rot,x_orbrot_0,list_degmo)
            for ii, orbpair in enumerate(list_orb_rot):
                if not np.isclose(x_orbrot_0[ii],0.0):
                    print(orbpair,x_orbrot_0[ii])
        f.close()
    else:
        print(f'\n All orbital pair rotations are allowed')
        list_orb_ind = []
        for imo in range(n_spatialmo):
            list_orb_ind.append(imo)
        for iimo, imo in enumerate(list_orb_ind):
            for jmo in list_orb_ind[iimo+1:]:
                list_orb_rot.append([imo,jmo])


    #print(f'\nList of orbitals that can be mixed:')
    #for item in list_orb_rot: print(item)
                




    if l_initial_orb_rot:
        obt, tbt = orthogonal_transform_obt_tbt(x_orbrot_0,list_orb_rot,obt_spatial,tbt_spatial)
    else:
        obt = obt_phys_spatial_to_spin(obt_spatial)
        tbt = tbt_phys_spatial_to_spin(tbt_spatial)




    list_act_orb = []
    for iorb in range(actmo_start,actmo_end+1):
        list_act_orb.append(iorb)


    homo = nelec // 2 - 1
    lumo = nelec //2

    #Parameters section
    nactorb = actmo_end - actmo_start + 1 #Here, nactorb is allowed to be different from the read-in nactmo
    nelact = nelec - (actmo_start)*2 #nelact is allowed to be different from the read-in nactel
    #Parameters section done

    mp2_ampld_list, mp2_Ecorr_list, ia_pair_list = prepare_mp2_amplitudes_actmo(actmo_end,actmo_start,orbene,tbt,l_no_sym=l_no_sym,debug=True)
    print(f'List of pairs of occupied and unoccupied spatial orbitals:')
    print(ia_pair_list)
    print(f'List of MP2 amplitudes:')
    print(mp2_ampld_list)
    print(f'List of MP2 E corr.:')
    print(mp2_Ecorr_list)


    sorted_mp2_ampld, sorted_ia_pair, sorted_mp2_Ecorr = sort_mp2_amplitudes(mp2_ampld_list,mp2_Ecorr_list,ia_pair_list,False)


    print(sorted_mp2_ampld)
    print(sorted_ia_pair)
    print(sorted_mp2_Ecorr)

    remove_mp2_amplitudes(sorted_mp2_ampld,sorted_ia_pair,sorted_mp2_Ecorr,mp2_ampld_thrsh,list_mo_exclud=list_mo_exclud)
    print('\nAfter removal of small amplitudes:')
    print(sorted_mp2_ampld)
    print(sorted_ia_pair)
    print(sorted_mp2_Ecorr)

    group_mp2_ampld, group_mp2_iapair, group_mp2_Ecorr = group_mp2_amplitudes(sorted_mp2_ampld,sorted_ia_pair,sorted_mp2_Ecorr,l_no_groupping=l_no_gen_groupping)

    print(f'Grouped MP2 amplitudes, orbital pairs, and Ecorr.')
    for iampld,ampld in enumerate(group_mp2_ampld):
        print(ampld,group_mp2_iapair[iampld],group_mp2_Ecorr[iampld])


    list_CSF, list_SOMO_DMO = generate_CASCI_space(n_spatialmo,nelec,nactorb,nelact,S_by2,list_seniority,list_mo_exclud=list_mo_exclud,l_pair_ex=l_pairex_within_actmo,debug=False)

    if l_axial_sym and not l_pairex_within_actmo:
        create_missing_axial_sym_CSFs(list_CSF,list_SOMO_DMO,list_degmo)

    #print(len(list_CSF),len(list_SOMO_DMO))
    print(f'SOMO and DMO of original CSF before screening')
    for ii, SOMO_DMO in enumerate(list_SOMO_DMO):
        print(f'CSF Basis {ii}, SOMO: {SOMO_DMO[0]}, DMO: {SOMO_DMO[1]}')

    if l_axial_sym and not l_pairex_within_actmo:
        print(f'Number of CSFs generated by generate_CASCI_space and with sym partners: {len(list_CSF)}')
    else:
        print(f'Number of CSFs generated by generate_CASCI_space: {len(list_CSF)}')


    if l_remove_CSF_with_0_Hmat_with_CSF0:
        #Screen off some CSFs that have different symmetries with the first CSF
        list_l_remove_CSF = [False] * len(list_CSF)
        list_onl = list_CSF[0][0]
        coefs_l  = list_CSF[0][2]
        for iCSF in range(1,len(list_CSF)):
            if iCSF % 10 == 0: print(f'Looping iCSF for removing CSFs, iCSF = {iCSF}')
            list_onr = list_CSF[iCSF][0]
            coefs_r  = list_CSF[iCSF][2]
            Helm = Helm_between_LCSDs(Enuc,obt,tbt,list_onl,coefs_l,list_onr,coefs_r)
            if np.isclose(Helm,0.0): list_l_remove_CSF[iCSF] = True
        
        for iCSF in range(len(list_CSF)-1,-1,-1):
            if list_l_remove_CSF[iCSF]: 
                del list_CSF[iCSF]
                del list_SOMO_DMO[iCSF]
        
        print(f'Number of CSFs after removing those with 0.0 H element with 0th CSF: {len(list_CSF)}')


    if l_remove_CSF_with_small_amplitudes:
    #Hmat_CSF = construct_Hmat_CSFs(list_CSF,Enuc,obt,tbt)
        Hmat_CSF = construct_Hmat_CSFs_paral_triu(list_CSF,Enuc,obt,tbt,nparal)
        Hmat_CSF_sparse = csr_matrix(Hmat_CSF)
        E_GS_CSF, psi_GS_CSF = get_ground_state(Hmat_CSF_sparse)
        list_l_remove_CSF = [False] * len(list_CSF)
        small = 0.02
        for iCSF, ampld in enumerate(psi_GS_CSF):
            if abs(ampld) < small: list_l_remove_CSF[iCSF] = True

        for iCSF in range(len(list_CSF)-1,-1,-1):
            if list_l_remove_CSF[iCSF]: 
                del list_CSF[iCSF]
                del list_SOMO_DMO[iCSF]

    if l_remove_CSF_chain_0_Hmat_with_CSF0:
        print(f'To select CSF based on chain Helm with CSF0')
        list_l_remove_CSF = screen_CSFs_chain_Helm_with_CSF0(list_CSF,Enuc,obt,tbt,nparal,small=1e-5,debug=True)
        print(f'Done selecing CSF based on chain Helm with CSF0')
        for iCSF in range(len(list_CSF)-1,-1,-1):
            if list_l_remove_CSF[iCSF]: 
                del list_CSF[iCSF]
                del list_SOMO_DMO[iCSF]

    print(f'SOMOs and DMOs of the CSFs:')
    for ii, SOMO_DMO in enumerate(list_SOMO_DMO):
        E_CSF = Helm_between_CSFs(Enuc,obt,tbt,list_CSF[ii],list_CSF[ii])
        print(f'CSF Basis {ii}, SOMO: {SOMO_DMO[0]}, DMO: {SOMO_DMO[1]}, E: {E_CSF}')


    if l_axial_sym:
        list_CSF, list_SOMO_DMO, list_sym_sign, list_sym_CSF = reorder_list_CSF_for_sym(list_CSF,list_SOMO_DMO,Enuc,obt,tbt,nparal,debug=True)
        print(f'SOMOs and DMOs of the sym-reordered CSFs:')
        for ii, SOMO_DMO in enumerate(list_SOMO_DMO):
            E_CSF = Helm_between_CSFs(Enuc,obt,tbt,list_CSF[ii],list_CSF[ii])
            print(f'CSF Basis {ii}, SOMO: {SOMO_DMO[0]}, DMO: {SOMO_DMO[1]}, E: {E_CSF}')
        print(f'# of sym-adapted CSFs: {len(list_sym_CSF)}')
    else:
        list_sym_sign = [] #Define trivial sym sign list

    if l_pairex_within_actmo:
        print('Stop the code for including pair excitation within active space')
        sys.exit()


    list_UCSF = []
    list_list_pair_ex_space = []
    list_list_genmat = []
    list_list_theta = []
    list_list_ia = []

    if l_select_ia_based_on_CSF_E0:
        print(f'Selecting ia pairs based on E0 of CSF space')
        list_ex_space = copy.deepcopy(list_CSF)
        list_list_pair_ex_space, list_list_ia, list_list_genmat, list_sym_CSF_vec = \
        select_ia_pairs_for_CSF_E0_with_sym_CSF(list_CSF,list_SOMO_DMO,group_mp2_iapair,Enuc,obt,tbt,l_include_ia_in_CAS,actmo_start,actmo_end,l_axial_sym,list_sym_sign,Ethrsh=Ethrsh_select_ia,list_mo_exclud=list_mo_exclud,nparal=nparal,debug=True)
        print('\nlist_sym_CSF_vec:')
        for ivec, vec in enumerate(list_sym_CSF_vec):
            print(ivec,vec)
        for iCSF_group in range(len(list_list_ia)):
        #print(len(list_list_ia[iCSF]),len(list_list_genmat[iCSF]))
            list_list_theta.append([0.0]*len(list_list_ia[iCSF_group]))
    else:
        print(f'\nOptimizing flexible U for individual CSF')
        for iCSF, CSF in enumerate(list_CSF):
            print(f'Seeking U space for CSF {iCSF}')
            list_pair_ex_space, list_ia_included, list_ia_ampld, list_genmat = \
            select_ia_pairs_for_one_CSF(CSF,group_mp2_iapair,group_mp2_ampld,Enuc,obt,tbt,Ethrsh=Ethrsh_select_ia,debug=False)
            list_list_pair_ex_space.append(list_pair_ex_space)
            list_list_ia.append(list_ia_included)
            list_list_theta.append(list_ia_ampld)
            list_list_genmat.append(list_genmat)
            ndim = len(list_pair_ex_space)
            print(f'\nDimension of the pair ex space: {ndim}')

    print(f'\nDone selecting ia pairs')

    list_list_pairex_CSF_vec, list_list_pairex_CSF = pick_pairex_within_CAS(list_list_pair_ex_space,list_sym_CSF_vec,internal_mo_start,internal_mo_end,l_axial_sym,Enuc,obt,tbt,nparal,list_mo_exclud=list_mo_exclud)


    #convert list_list_genmat to list_list_group_genmat. Genmat within each grouop has the same eigenvalue
    if l_use_decomp_genmat: 
        list_list_decomp_genmat = groupping_list_list_genmat(list_list_genmat,False)
        for ii in range(len(list_list_decomp_genmat)):
            list_genmat = list_list_genmat[ii]
            list_decomp_genmat = list_list_decomp_genmat[ii]
            assert len(list_genmat) == len(list_decomp_genmat)
            n_theta = len(list_genmat)
            if n_theta == 0: continue
            x_random = np.random.uniform(low=-0.5, high=-0.5, size = n_theta)
            if nparal != 1:
                compare_num_anl_Umat(x_random,list_genmat,list_decomp_genmat,nparal)
    else:
        list_list_decomp_genmat = []

    list_list_ia_noCAS, list_list_genmat_noCAS, list_list_theta_noCAS, list_list_decomp_genmat_noCAS = \
    remove_pairex_in_CAS(actmo_start,actmo_end,list_list_ia,list_list_genmat,list_list_theta,l_use_decomp_genmat,list_list_decomp_genmat)

    print(f'\nlist_list_ia_noCAS:')
    for iref,item in enumerate(list_list_ia_noCAS):
        print(f'Group: {iref}, {item}')
    print(f'\nlist_list_theta_noCAS:')
    for iref,item in enumerate(list_list_theta_noCAS):
        print(f'Group: {iref}, {item}')

    list_list_ia_internal, list_list_ia_external = separate_ia_pairs_internal_external(list_list_ia,internal_mo_start,internal_mo_end,list_mo_exclud=list_mo_exclud,debug=True)


    list_all_CSFs_ex_space = []
    list_UCSF_subspace_start_end = []
    istart = 0
    for iCSF_group in range(len(list_list_pair_ex_space)):
        iend = istart + len(list_list_pair_ex_space[iCSF_group])
        list_all_CSFs_ex_space += list_list_pair_ex_space[iCSF_group]
        list_UCSF_subspace_start_end.append([istart,iend])
        print([istart,iend])
        istart = iend

    #Hmat_full_space, list_list_rdm1, list_list_rdm2 =  construct_Hmat_CSFs(list_all_CSFs_ex_space,Enuc,obt,tbt,lrdm=True)
    Hmat_full_space, list_list_rdm1, list_list_rdm2 =  construct_Hmat_CSFs_paral_triu(list_all_CSFs_ex_space,Enuc,obt,tbt,nparal=nparal,lrdm=True)
    print(f'\nDimension of the accumulated excited CSF space: {len(list_all_CSFs_ex_space)}')
    #print('\nHamiltonian matrix of the whole ex space of kept CSFs')
    #print_matrix(Hmat_full_space)
    Hmat_full_space_sparse = csr_matrix(Hmat_full_space)
    E_GS_full, psi_GS_full = get_ground_state(Hmat_full_space_sparse)
    print(f'\nE0 of the full ex space of all CSFs: {E_GS_full}')
    print(f'\nGround state:')
    small = 0.02
    for ii in range(len(psi_GS_full)):
        if abs(psi_GS_full[ii]) > small: print(ii,psi_GS_full[ii])


    #Enter the macrocycle to optimize orbitals

    if l_opt_orb:
        print('\nOrbital optimization starts')
        if not l_initial_orb_rot:
            x_orbrot_0 = np.array([0.0]*len(list_orb_rot))


        improve = 100.0
        E_previous = E_GS_full
        nround = -1
        while(abs(improve) > orbopt_conv_thrsh):
            nround += 1
            x_orbrot, E_orbopt, obt_opt, tbt_opt = opt_orbtials_for_GS_of_CSF_space(list_all_CSFs_ex_space,psi_GS_full,Enuc,obt_spatial,tbt_spatial,list_orb_rot,list_list_rdm1,list_list_rdm2,l_axial_sym,list_degmo,x_orbrot_0,nparal,debug=False)
        #Hmat_full_space = construct_Hmat_CSFs(list_all_CSFs_ex_space,Enuc,obt_opt,tbt_opt)
            Hmat_full_space = construct_Hmat_CSFs_paral_triu(list_all_CSFs_ex_space,Enuc,obt_opt,tbt_opt,nparal,lrdm=False)
            Hmat_full_space_sparse = csr_matrix(Hmat_full_space)
            E_GS_full, psi_GS_full = get_ground_state(Hmat_full_space_sparse)
            improve = E_GS_full - E_previous
            print(f'Energy lowering in round {nround}: {improve}')
            obt = obt_opt
            tbt = tbt_opt
            x_orbrot_0 = x_orbrot
            E_previous = E_GS_full

        print(f'Orbital optimization converged')
        print(f'resultant orbital rotational angles')
        for ii in range(len(list_orb_rot)):
            if abs(x_orbrot[ii]) > small_orbrot_angle:
                print(list_orb_rot[ii],x_orbrot[ii])
    else:
        x_orbrot = [0.0]*len(list_orb_rot)

    print(f'\nE0 of the full EX space before U optimization: {E_GS_full}')

    #Get the symmetry-adapted basis vectors for all CSFs in the excited space
    if l_axial_sym:
        list_ECSF_all = []
        ndim_full = len(list_all_CSFs_ex_space)
        for iCSF in range(ndim_full):
            list_ECSF_all.append(Hmat_full_space[iCSF,iCSF])

        list_sym_CSF_vec_all_space = []
        CSF_already_considered = []
        degen_thrsh = 1.0e-12
        for iCSF in range(ndim_full):
            if iCSF in CSF_already_considered: continue
            CSF_already_considered.append(iCSF)
            list_degen_pair = []
            for jCSF in range(iCSF+1,ndim_full):
                if abs(list_ECSF_all[iCSF] - list_ECSF_all[jCSF]) < degen_thrsh and np.isclose(abs(psi_GS_full[iCSF]),abs(psi_GS_full[jCSF])):
                    list_degen_pair.append(jCSF)
            sym_CSF_vec = np.zeros(ndim_full)
            if len(list_degen_pair) == 0:
                sym_CSF_vec[iCSF] = 1.0
                list_sym_CSF_vec_all_space.append(sym_CSF_vec)
        #elif len(list_degen_pair) > 1:
        #    print(f'More than one CSF is found dgenerate with CSF{iCSF} with {list_ECSF_all[iCSF], psi_GS_full[iCSF]}')
        #    print(list_all_CSFs_ex_space[iCSF])
        #    for jCSF in list_degen_pair:
        #        print(f'CSF{jCSF}, {list_ECSF_all[jCSF], psi_GS_full[jCSF]}')
        #        print(list_all_CSFs_ex_space[jCSF])
        #    print('Bombing out')
        #    sys.exit()
            else:
                sym_CSF_vec[iCSF] = psi_GS_full[iCSF]
                for jCSF in list_degen_pair:
                    sym_CSF_vec[jCSF] = psi_GS_full[jCSF]
                    CSF_already_considered.append(jCSF)
                norm_vec = np.linalg.norm(sym_CSF_vec)
                sym_CSF_vec = sym_CSF_vec / norm_vec
                list_sym_CSF_vec_all_space.append(sym_CSF_vec)
            
        n_sym_CSF_vec_total = len(list_sym_CSF_vec_all_space)
        print(f'\n#Total number of sym-adapted CSFs in the whole ex space: {n_sym_CSF_vec_total}')
        Umat_sym_adapted = np.zeros([ndim_full,n_sym_CSF_vec_total])
        for icol in range(n_sym_CSF_vec_total):
            Umat_sym_adapted[:,icol] = list_sym_CSF_vec_all_space[icol]

        mat_should_be_1 = Umat_sym_adapted.transpose()@Umat_sym_adapted
        resid_mat = mat_should_be_1 - np.eye(n_sym_CSF_vec_total)
        assert np.isclose(np.linalg.norm(resid_mat),0.0)
    #print('\nIndices of non-zero elements in resid_mat:')
    #print(np.where(abs(resid_mat) > 1.0e-6 ))
        Hmat_CSF_sym_adapted = Umat_sym_adapted.transpose()@Hmat_full_space@Umat_sym_adapted
        Hmat_CSF_sym_sparse = csr_matrix(Hmat_CSF_sym_adapted)
        E_GS_sym_adapted, psi_GS_sym_adapted = get_ground_state(Hmat_CSF_sym_sparse)
        if abs(E_GS_sym_adapted - E_GS_full) > degen_thrsh*100:
            print(f'Too large difference between E_GS using sym-adapted CSFs or not: {E_GS_sym_adapted,E_GS_full}')
            print('Bombing out!')
            sys.exit()

    #Get the external rotational amplitudes one by one as the initial guess for later
    get_ext_ampld_1by1(list_CSF,list_SOMO_DMO,list_sym_sign,list_list_ia_internal,list_list_ia_external,l_axial_sym)

    #sys.exit()

    if l_use_decomp_genmat:
        list_list_genmat_to_use = list_list_decomp_genmat_noCAS
    else:
        list_list_genmat_to_use = list_list_genmat_noCAS

    #Get the mp2 amplitudes for the external ia pairs
    list_list_mp2_ampld_for_Uext = []
    for iref in range(len(list_list_ia_noCAS)):
        print(f'Ref {iref}:')
        list_mp2_ampld_for_Uext = []
        for ia_pairs in list_list_ia_noCAS[iref]:
            for pair in ia_pairs:
                ampld_taken = sorted_mp2_ampld[sorted_ia_pair.index(pair)]

            print(f'pairs: {ia_pairs}, mp2 ampld: {ampld_taken}')
            list_mp2_ampld_for_Uext.append(ampld_taken)
        list_list_mp2_ampld_for_Uext.append(list_mp2_ampld_for_Uext)

    Umat_full_Uext_mp2, Vmat_full_make_Uvec_full, list_list_refCSF, list_list_Uext_mp2_CSF = make_Uvec_full(list_list_ia_noCAS,list_list_mp2_ampld_for_Uext,l_use_decomp_genmat,list_list_genmat_to_use,list_UCSF_subspace_start_end,list_list_pairex_CSF_vec,list_list_pair_ex_space,debug=True)

    Hmat_Uext_mp2 = Umat_full_Uext_mp2.transpose()@Hmat_full_space@Umat_full_Uext_mp2
    E0_Uext_mp2, psi_Uext_mp2 = get_ground_state(csr_matrix(Hmat_Uext_mp2))
    print(f'\nE0_Uext_mp2: {E0_Uext_mp2}')
    print(f'psi_Uext_mp2: {psi_Uext_mp2}')


    #Confirm E0_Uext_mp2 from the actual Uext mp2 CSFs
    #list_all_Uext_mp2_CSF = []
    #for item in list_list_Uext_mp2_CSF:
    #    list_all_Uext_mp2_CSF += item
    #
    #Hmat_Uext_mp2_CSF = construct_Hmat_CSFs(list_all_Uext_mp2_CSF,Enuc,obt,tbt)
    #E0_Uext_mp2_CSF, psi_Uext_mp2_CSF = get_ground_state(csr_matrix(Hmat_Uext_mp2_CSF))
    #print(f'\nE0_Uext_mp2_CSF: {E0_Uext_mp2_CSF}')
    #print(f'psi_Uext_mp2_CSF: {psi_Uext_mp2_CSF}')


    list_list_theta_opt_noCAS, list_list_Uvec_noCAS, Umat_total_noCAS, Vmat_total_inCAS = opt_U_overlap_with_noEX_in_CAS(list_list_ia_noCAS,list_list_theta_noCAS,list_list_genmat_to_use,list_UCSF_subspace_start_end,psi_GS_full,Uopt_thrsh,list_list_pairex_CSF_vec,l_use_decomp_genmat,conv=1.0e-5,nparal=nparal,ldisp=True,debug=True)

    print(f'\nlist_list_theta_opt_noCAS')
    for ii, item in enumerate(list_list_theta_opt_noCAS):
        print(ii,item)

    np.isclose(np.sum(abs(Vmat_full_make_Uvec_full - Vmat_total_inCAS)),0.0)
    print(f'The Vmat_full from make_Uvec_full and opt_U_overlap_with_noEX_in_CAS are consistent')


    Hmat_inCAS = Vmat_total_inCAS.transpose()@Hmat_full_space@Vmat_total_inCAS
    E0_inCAS, psi_GS_inCAS = get_ground_state(csr_matrix(Hmat_inCAS))
    print(f'\nE0 of in CAS basis: {E0_inCAS}')
    print(f'psi_GS_inCAS: {psi_GS_inCAS}')

    print(Umat_total_noCAS.shape,Hmat_full_space.shape)
    Hmat_UnoCAS = Umat_total_noCAS.transpose()@Hmat_full_space@Umat_total_noCAS
    print('\nHamiltonian matrix of the U states without CAS pair excitation\n')
    print_matrix(Hmat_UnoCAS)

    E0_UnoCAS, psi_GS_UnoCAS = get_ground_state(csr_matrix(Hmat_UnoCAS))
    print(f'E0 of UnoCAS basis: {E0_UnoCAS}, diff. from E0 of full space by {E0_UnoCAS - E_GS_full}')
    print(f'psi_GS: {psi_GS_UnoCAS}')

    #Calculate overlap between the ground state of full ex space and the ground state of the reduced space
    #without pair excitations within CAS
    print(f'\nOverlap between ground state of full ex space and noCASex space {psi_GS_full.transpose()@Umat_total_noCAS@psi_GS_UnoCAS}')
    #Overlap = 0.0
    #n_Uvec_total = Umat_total_noCAS.shape[1]
    #for ivec in range(n_Uvec_total):
    #    np.dot(Umat_total_noCAS[:,ivec],)


    print(f'Optimized thetas for Uext excitations:')
    list_list_Uext_mp2_ampld = []
    list_list_Uext_opt_ampld = []
    for irefCSF in range(len(list_list_theta_opt_noCAS)):
        list_Uext_mp2_ampld = []
        list_Uext_opt_ampld = []
        list_theta_opt_noCAS = list_list_theta_opt_noCAS[irefCSF]
        list_ia_noCAS = list_list_ia_noCAS[irefCSF]
        if len(list_theta_opt_noCAS) == 0: 
            list_list_Uext_mp2_ampld.append(list_Uext_mp2_ampld)
            list_list_Uext_opt_ampld.append(list_Uext_opt_ampld)
            continue
        n_train_U = len(list_theta_opt_noCAS) // len(list_ia_noCAS)
        print(f'There are {n_train_U} trains of U')
        for i_train in range(n_train_U):
            print(f'\nTrain {i_train+1:}, iCSF_group: {irefCSF}')
            for ipairs in range(len(list_ia_noCAS)):
                pairs = list_ia_noCAS[ipairs]
                theta_opt = list_theta_opt_noCAS[i_train*len(list_ia_noCAS)+ipairs]
                mp2_idx = group_mp2_iapair.index(pairs)
                print(f'pairs: {pairs}, Opt ampld: {theta_opt}, MP2 ampld: {group_mp2_ampld[mp2_idx]}')
                if i_train == 0:
                    list_Uext_mp2_ampld.append([pairs,group_mp2_ampld[mp2_idx]])
                list_Uext_opt_ampld.append([pairs,theta_opt])

        list_list_Uext_mp2_ampld.append(list_Uext_mp2_ampld)
        list_list_Uext_opt_ampld.append(list_Uext_opt_ampld)
        print(f'Updated {irefCSF}-th Uext amplds. {len(list_list_Uext_mp2_ampld),len(list_list_Uext_opt_ampld)}')

    #assert len(list_list_Uext_mp2_ampld) == len(list_list_refCSF)
    #assert len(list_list_Uext_opt_ampld) == len(list_list_refCSF)
    if len(list_list_Uext_mp2_ampld) != len(list_list_refCSF):
        print(f'length of list_list_Uext_mp2_ampld: {len(list_list_Uext_mp2_ampld)}')
        print(f'length of list_list_refCSF: {len(list_list_refCSF)}')
        print('Inconsistent. Bombing out!')
        sys.exit()
    if len(list_list_Uext_opt_ampld) != len(list_list_refCSF):
        print(f'length of list_list_Uext_opt_ampld: {len(list_list_Uext_opt_ampld)}')
        print(f'length of list_list_refCSF: {len(list_list_refCSF)}')
        print('Inconsistent. Bombing out!')
        sys.exit()

    for iref in range(len(list_list_Uext_mp2_ampld)):
        list_refCSF = list_list_refCSF[iref]
        list_Uext_mp2_ampld = list_list_Uext_mp2_ampld[iref]
        list_Uext_opt_ampld = list_list_Uext_opt_ampld[iref]
        print(f'MP2 amplitudes for external excitations for reference group {iref}:')
        for item in list_Uext_mp2_ampld:
            print(item)
        print(f'Optimized amplitudes for external excitations for reference group {iref}:')
        for item in list_Uext_opt_ampld:
            print(item)
        print(f'The following CSFs have the same SOMOs and same singlet coupling pathway. They share the external amplitudes above')
        for item in list_refCSF:
            print(item)

    ### Save PT files
    save_filename = dumpfile_prefix + '_Uext_CSF.dump'
    with open(save_filename, 'wb') as f:
        pickle.dump([list_list_refCSF,list_list_Uext_mp2_CSF,list_list_Uext_mp2_ampld,list_list_Uext_opt_ampld,\
        list_orb_rot,x_orbrot,Enuc,obt_spatial,tbt_spatial],f)


    #list_list_theta_opt, list_Uvec_opt = opt_U_overlap(list_list_ia,list_list_theta,list_list_genmat,list_UCSF_subspace_start_end,psi_GS_full,Uopt_thrsh,list_sym_CSF_vec,l_use_decomp_genmat=False,ldisp=True,debug=True)
    if l_use_decomp_genmat:
        list_list_theta_opt, list_Uvec_opt, list_Umat_opt = opt_U_overlap(list_list_ia,list_list_theta,list_list_decomp_genmat,list_UCSF_subspace_start_end,psi_GS_full,Uopt_thrsh,list_sym_CSF_vec,nparal,l_use_decomp_genmat,ldisp=True,debug=True)
    else:
        list_list_theta_opt, list_Uvec_opt, list_Umat_opt = opt_U_overlap(list_list_ia,list_list_theta,list_list_genmat,list_UCSF_subspace_start_end,psi_GS_full,Uopt_thrsh,list_sym_CSF_vec,nparal,l_use_decomp_genmat,ldisp=True,debug=True)

    #Ensure consistency among list_Umat_opt, list_Uvec_opt, and list_sym_CSF_vec
    for isymCSF in range(len(list_sym_CSF_vec)):
        temp_norm = np.linalg.norm(list_Uvec_opt[isymCSF]-list_Umat_opt[isymCSF]@list_sym_CSF_vec[isymCSF])
        assert np.isclose(temp_norm,0.0)

    list_UCSF = []
    list_CSF_ref = []
    for iCSF_group in range(len(list_list_pair_ex_space)):
        list_theta = list_list_theta_opt[iCSF_group]
        list_pair_ex_space = list_list_pair_ex_space[iCSF_group]
        Uvec = list_Uvec_opt[iCSF_group]
        [istart,iend] = list_UCSF_subspace_start_end[iCSF_group]
        refvec = copy.deepcopy(psi_GS_full[istart:iend])
        refvec = refvec / np.linalg.norm(refvec)
        UCSF = make_UCSF_state(list_pair_ex_space,Uvec)
        list_UCSF.append(UCSF)
        refCSF = make_UCSF_state(list_pair_ex_space,refvec)
        list_CSF_ref.append(refCSF)
        n_unique_ia_pair = len(list_list_genmat[iCSF_group])
        if n_unique_ia_pair == 0: 
            print(f'\nNo ia pairs for CSF group {iCSF_group}')
            continue
        print(f'\nia pairs of UCSF{iCSF_group}:')
        n_U_trains = len(list_theta) // n_unique_ia_pair
        print(f'# of U trains: {n_U_trains}')
        for i_train in range(n_U_trains):
            for ipair in range(n_unique_ia_pair):
                mp2_idx = group_mp2_iapair.index(list_list_ia[iCSF_group][ipair])
                print(f'pairs: {list_list_ia[iCSF_group][ipair]},Opt ampld: {list_theta[ipair+i_train*n_unique_ia_pair]},\
                MP2 ampld: {group_mp2_ampld[mp2_idx]}')
            print()

    if l_axial_sym: assert len(list_UCSF) == len(list_sym_CSF)

    #Prepare CSFs and UCSFs for Smik and Praveen
    if l_axial_sym: 
        list_UCSF_sym_components = []
        list_list_theta_opt_CSF_comp = []
        list_list_ia_CSF_comp = []
        for iCSF_group in range(len(list_list_pair_ex_space)):
            list_pair_ex_space = list_list_pair_ex_space[iCSF_group]
            ndim_iCSF_group = len(list_pair_ex_space)
            sym_CSF_vec = list_sym_CSF_vec[iCSF_group]
            list_non0_ind = np.where(sym_CSF_vec != 0.0)[0]
            if len(list_non0_ind) == 1:
                list_UCSF_sym_components.append(list_UCSF[iCSF_group])
                list_list_ia_CSF_comp.append(list_list_ia[iCSF_group])
                list_list_theta_opt_CSF_comp.append(list_list_theta_opt[iCSF_group])
            else:
                for ind in list_non0_ind:
                    comp_CSF_vec = np.zeros(ndim_iCSF_group)
                    comp_CSF_vec[ind] = 1.0
                    UCSF_comp_vec = list_Umat_opt[iCSF_group]@comp_CSF_vec
                    UCSF_comp = make_UCSF_state(list_pair_ex_space,UCSF_comp_vec)
                    list_UCSF_sym_components.append(UCSF_comp)
                list_list_ia_CSF_comp.append(list_list_ia[iCSF_group])
                list_list_ia_CSF_comp.append(list_list_ia[iCSF_group])
                list_list_theta_opt_CSF_comp.append(list_list_theta_opt[iCSF_group])
                list_list_theta_opt_CSF_comp.append(list_list_theta_opt[iCSF_group])
                
    else:
        list_UCSF_sym_components = list_UCSF
        list_list_theta_opt_CSF_comp = list_list_theta_opt
        list_list_ia_CSF_comp = list_list_ia

    #Hmat_refCSF = construct_Hmat_CSFs(list_CSF_ref,Enuc,obt,tbt)
    Hmat_refCSF = construct_Hmat_CSFs_paral_triu(list_CSF_ref,Enuc,obt,tbt,nparal)
    print('\nHmat of ref. CSF')
    print_matrix(Hmat_refCSF)
    Hmat_refCSF_sparse = csr_matrix(Hmat_refCSF)
    E_GS_refCSF, psi_GS_refCSF = get_ground_state(Hmat_refCSF_sparse)
    #print(f'\nE0 of ref CSF space: {E_GS_refCSF}')
    #print('\nGround state:')
    #print(psi_GS_refCSF)
    assert np.isclose(E_GS_refCSF,E_GS_full)


    #Hmat_UCSF = construct_Hmat_CSFs(list_UCSF,Enuc,obt,tbt)
    Hmat_UCSF = construct_Hmat_CSFs_paral_triu(list_UCSF,Enuc,obt,tbt,nparal)
    print('\nHmat of UCSF')
    print_matrix(Hmat_UCSF)
    Hmat_UCSF_sparse = csr_matrix(Hmat_UCSF)
    E_GS_UCSF, psi_GS_UCSF = get_ground_state(Hmat_UCSF_sparse)
    print(f'\nE0 of UCSF space of all CSFs: {E_GS_UCSF}')
    print(f'Error from U opt: {E_GS_UCSF - E_GS_full}')
    print('\nGround state:')
    print(psi_GS_UCSF)

    #Ensure the sym components of UCSFs give the same ground state
    #Hmat_UCSF_symcomp = construct_Hmat_CSFs(list_UCSF_sym_components,Enuc,obt,tbt)
    Hmat_UCSF_symcomp = construct_Hmat_CSFs_paral_triu(list_UCSF_sym_components,Enuc,obt,tbt,nparal)
    Hmat_UCSF_symcomp_sparse = csr_matrix(Hmat_UCSF_symcomp)
    E_GS_UCSF_symcomp, psi_GS_UCSF_symcomp = get_ground_state(Hmat_UCSF_symcomp)
    assert np.isclose(E_GS_UCSF_symcomp,E_GS_UCSF)

    list_list_somo_UCSF_symcomp = []
    for UCSF_symcomp in list_UCSF_sym_components:
        list_list_somo_UCSF_symcomp.append(get_SOMO_in_CSF(UCSF_symcomp))
    #print(list_list_somo_UCSF_symcomp[-1])
        


    save_filename = dumpfile_prefix + '_CSF_UCSF_GS_optU_optorb_flexibleU_matchstate_E0select_ia_' + str(rdist) + '.dump'
    if l_axial_sym:
        with open(save_filename, 'wb') as f:
            pickle.dump([list_CSF,list_sym_CSF_vec,list_sym_CSF,list_UCSF,list_list_ia,list_list_theta_opt,list_list_genmat,\
            list_orb_rot,x_orbrot,Enuc,obt_spatial,tbt_spatial],f)
    else:
        with open(save_filename, 'wb') as f:
            pickle.dump([list_CSF,list_UCSF,list_list_ia,list_list_theta_opt,list_list_genmat,\
            list_orb_rot,x_orbrot,Enuc,obt_spatial,tbt_spatial],f)

    # save VO files
    save_filename = dumpfile_prefix + '_UCSF_sym_comp.dump'
    with open(save_filename, 'wb') as f:
        pickle.dump([list_CSF,list_list_ia_CSF_comp,list_list_theta_opt_CSF_comp,list_sym_CSF_vec,list_UCSF,list_UCSF_sym_components,\
        list_list_somo_UCSF_symcomp,psi_GS_UCSF_symcomp,list_orb_rot,x_orbrot,Enuc,obt_spatial,tbt_spatial],f)

