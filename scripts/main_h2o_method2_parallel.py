import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform
from joblib import Parallel, delayed

from openfermion import (
    jordan_wigner,
    FermionOperator,
    QubitOperator,
    get_sparse_operator,
    normal_ordered,
    get_ground_state,
    commutator,
    anticommutator
)

from utils_CSF_and_UCSF import (
    orthogonal_transform_obt_tbt,
    obt_phys_spatial_to_spin,
    tbt_phys_spatial_to_spin,
    make_short_H_ferm_op
)
from utils_basic import (
    copy_hamiltonian
)
from utils_hamiltonian_ferm import (
    process_fermionic_hamiltonian_to_remove_irrelevant_terms
)
from utils_hamiltonian_qubit import (
    process_qubit_hamiltonian_to_remove_irrelevant_terms,
    process_qubit_hamiltonian_to_project_onto_symmetric_subspace,
    XorY_augment,
    sampling_cost
)
from utils_partitioning import (
    sorted_insertion_decomposition,
    augment_decomp_with_pauli_x_plus_i_pauli_y
)
from utils_states import (
    somos_to_seniority_config,
    convert_TZ_format_to_sparse_format,
    convert_dense_format_to_sparse_format,
    create_composite_state,
    expectation, 
    variance_of_general_operator,
    variance_of_decomp,
    compress_state,
    tz_state_seniority_config
)

from math import radians, sin, cos
import pyscf as ps
from pyscf import fci






bond_length     = sys.argv[1]
filename        = f'h2o_data/Uext_CSF_for_Praveen_Smik_{bond_length}.dump'
output_filename = f'results/h2o_method2_parallel_{bond_length}'

with open(filename, 'rb') as f:
    (
    list_list_refCSF,
    list_list_Uext_mp2_CSF,
    list_list_Uext_mp2_ampld,
    list_list_Uext_opt_ampld,
    list_orb_rot,
    x_orbrot,
    Enuc,
    obt_spatial,
    tbt_spatial
    ) = pickle.load(f)

#Rotate orbitals 
if len(list_orb_rot) != 0:
    obt, tbt = orthogonal_transform_obt_tbt(x_orbrot,list_orb_rot,obt_spatial,tbt_spatial)
else:
    obt = obt_phys_spatial_to_spin(obt_spatial)
    tbt = tbt_phys_spatial_to_spin(tbt_spatial)

Nqubits = obt.shape[0]
Norb    = Nqubits // 2
dim     = 2 ** Nqubits

tz_states       = []
quantum_indices = []
tally           = 0
for i, ucsf_list in enumerate(list_list_Uext_mp2_CSF):
    for ucsf in ucsf_list:
        tz_states.append(ucsf)
        
        if not list_list_Uext_mp2_ampld[i] == []:
            quantum_indices.append(tally)

        tally += 1

statevectors = [convert_TZ_format_to_sparse_format(dim, tz_state) for tz_state in tz_states]
configs      = [tz_state_seniority_config(ucsf) for ucsf in tz_states]
Nstates      = len(tz_states)

Hferm = make_short_H_ferm_op(Enuc, obt, tbt)
Hqub  = jordan_wigner(Hferm)

Hsub       = np.zeros([Nstates, Nstates], dtype=np.complex128)
sig_matrix = np.zeros([Nstates, Nstates], dtype=np.complex128)

for i in range(Nstates):

    with open(output_filename, 'a') as f:
        print(i, i, file=f)

    ket        = statevectors[i]
    ket_t      = convert_dense_format_to_sparse_format(compress_state(ket.toarray()[0]))
    ket_config = configs[i]

    Htapered                = process_qubit_hamiltonian_to_project_onto_symmetric_subspace(Hqub, Nqubits, ket_config, ket_config)
    after_tapering_constant = Htapered.constant
    Htapered               -= after_tapering_constant
    Htapered.compress()
    Htapered_sparse         = get_sparse_operator(Htapered + after_tapering_constant, Nqubits // 2)
    Hsub[i,i]               = (ket_t @ Htapered_sparse @ ket_t.T)[0,0]

    if i in quantum_indices:
        decomp          = sorted_insertion_decomposition(Htapered, methodtag='fc')
        sig_matrix[i,i] = np.sqrt(variance_of_decomp(decomp, ket_t, Nqubits // 2, general=True))

def evaluate_off_diagonal(i, j, Hsub, statevectors, configs, Hqub, Nqubits, quantum_indices, methodtag='fc'):
    with open(output_filename, 'a') as f:
        print(i, j, file=f)

    ij_shift = 0.5 * (Hsub[i,i] + Hsub[j,j])

    bra        = statevectors[i]
    bra_t      = convert_dense_format_to_sparse_format(compress_state(bra.toarray()[0]))
    bra_config = configs[i]

    ket        = statevectors[j]
    ket_t      = convert_dense_format_to_sparse_format(compress_state(ket.toarray()[0]))
    ket_config = configs[j]

    comp_t     = create_composite_state(bra_t, ket_t, Nqubits // 2)

    Htapered        = process_qubit_hamiltonian_to_project_onto_symmetric_subspace(Hqub - ij_shift, Nqubits, bra_config, ket_config)
    Htapered_sparse = get_sparse_operator(Htapered, Nqubits // 2)
    matrix_element  = (bra_t @ Htapered_sparse @ ket_t.T)[0,0]

    if (i in quantum_indices) or (j in quantum_indices):
        Htapered_aug     = XorY_augment(Htapered, Nqubits // 2)
        decomp           = sorted_insertion_decomposition(Htapered_aug, methodtag='fc')
        element_variance = np.sqrt(variance_of_decomp(decomp, comp_t, Nqubits // 2 + 1, general=True))
    else:
        element_variance = 0.0

    return matrix_element, element_variance

ij_pairs = []
for i in range(Nstates):
    for j in range(Nstates):
        if i > j:
            ij_pairs.append((i,j))
            
results = Parallel(n_jobs=-1)(delayed(evaluate_off_diagonal)(i,j,Hsub,statevectors,configs,Hqub,Nqubits,quantum_indices,methodtag='fc') for i,j in ij_pairs)

for k, (i,j) in enumerate(ij_pairs):
    matrix_element, element_variance = results[k]

    Hsub[i,j] = matrix_element
    Hsub[j,i] = matrix_element

    sig_matrix[i,j] = element_variance
    sig_matrix[j,i] = element_variance


# for i in range(Nstates):
#     for j in range(Nstates):
#         if i > j:
#             with open(output_filename, 'a') as f:
#                 print(i, j, file=f)

#             ij_shift = 0.5 * (Hsub[i,i] + Hsub[j,j])

#             bra        = statevectors[i]
#             bra_t      = convert_dense_format_to_sparse_format(compress_state(bra.toarray()[0]))
#             bra_config = configs[i]

#             ket        = statevectors[j]
#             ket_t      = convert_dense_format_to_sparse_format(compress_state(ket.toarray()[0]))
#             ket_config = configs[j]

#             comp_t     = create_composite_state(bra_t, ket_t, Nqubits // 2)
            
#             Htapered        = process_qubit_hamiltonian_to_project_onto_symmetric_subspace(Hqub - ij_shift, Nqubits, bra_config, ket_config)
#             Htapered_sparse = get_sparse_operator(Htapered, Nqubits // 2)
#             matrix_element  = (bra_t @ Htapered_sparse @ ket_t.T)[0,0]
#             Hsub[i,j]       = matrix_element
#             Hsub[j,i]       = matrix_element

#             if (i in quantum_indices) or (j in quantum_indices):
#                 Htapered_aug     = XorY_augment(Htapered, Nqubits // 2)
#                 decomp           = sorted_insertion_decomposition(Htapered_aug, methodtag='fc')
#                 element_variance = np.sqrt(variance_of_decomp(decomp, comp_t, Nqubits // 2 + 1, general=True))
#                 sig_matrix[i,j]  = element_variance
#                 sig_matrix[j,i]  = element_variance
#             else:
#                 sig_matrix[i,j] = 0.0
#                 sig_matrix[j,i] = 0.0

vals, vecs = np.linalg.eigh(Hsub)
Egs        = vals[0]
c          = vecs[:,0]
cost       = sampling_cost(c, sig_matrix)


with open(output_filename, 'a') as f:
    print(f'''
        
    Final Results: 
        Method              : {2}
        System              : {'H2O'}
        Bond Length         : {bond_length} 
        Ground State Energy : {Egs}
        Cost                : {cost} 
    ''', file=f)