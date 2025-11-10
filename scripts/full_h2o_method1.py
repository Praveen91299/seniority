import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform

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

from seniority.src.measurement.utils_CSF_and_UCSF import (
    orthogonal_transform_obt_tbt,
    obt_phys_spatial_to_spin,
    tbt_phys_spatial_to_spin,
    make_short_H_ferm_op
)
from seniority.src.measurement.utils_basic import (
    copy_hamiltonian
)
from seniority.src.measurement.utils_hamiltonian_ferm import (
    process_fermionic_hamiltonian_to_remove_irrelevant_terms
)
from seniority.src.measurement.utils_hamiltonian_qubit import (
    process_qubit_hamiltonian_to_remove_irrelevant_terms,
    process_qubit_hamiltonian_to_project_onto_symmetric_subspace,
    XorY_augment,
    sampling_cost
)
from seniority.src.measurement.utils_partitioning import (
    sorted_insertion_decomposition,
    augment_decomp_with_pauli_x_plus_i_pauli_y
)
from seniority.src.measurement.utils_states import (
    somos_to_seniority_config,
    convert_TZ_format_to_sparse_format,
    convert_dense_format_to_sparse_format,
    create_composite_state,
    expectation, 
    matrix_element,
    variance_of_general_operator,
    variance_of_decomp,
    compress_state
)

from math import radians, sin, cos
import pyscf as ps
from pyscf import fci

bond_length     = sys.argv[1]
filename        = f'h2o_data/UCSF_sym_comp_for_Praveen_Smik_{bond_length}.dump'
output_filename = f'results/h2o_method1_{bond_length}'

with open(filename, 'rb') as f:
    (
    list_CSF,
    list_list_ia_CSF,
    list_list_theta_CSF,
    list_sym_CSF_vec,
    list_UCSF_tz,
    tz_states,
    somos_list,
    psi_GS_UCSF_smik,
    list_orb_rot,
    x_orbrot,
    Enuc,
    obt_spatial,
    tbt_spatial
    ) = pickle.load(f)

### import into CSF objects


#Rotate orbitals 
if len(list_orb_rot) != 0:
    obt, tbt = orthogonal_transform_obt_tbt(x_orbrot,list_orb_rot,obt_spatial,tbt_spatial)
else:
    obt = obt_phys_spatial_to_spin(obt_spatial)
    tbt = tbt_phys_spatial_to_spin(tbt_spatial)

Nqubits = obt.shape[0]
Norb    = Nqubits // 2
dim     = 2 ** Nqubits
Nstates = len(tz_states)

statevectors = [convert_TZ_format_to_sparse_format(dim, tz_state) for tz_state in tz_states]
configs      = [somos_to_seniority_config(somo, Norb) for somo in somos_list]

Hferm = make_short_H_ferm_op(Enuc, obt, tbt)
Hqub  = jordan_wigner(Hferm)

quantum_indices = [i for i in range(Nstates) if isinstance(list_list_theta_CSF[i], np.ndarray)]




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

for i in range(Nstates):
    for j in range(Nstates):
        if i > j:
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
            me  = matrix_element(Htapered_sparse, bra_t, ket_t)
            Hsub[i,j]       = me
            Hsub[j,i]       = me

            if (i in quantum_indices) or (j in quantum_indices):
                Htapered_aug     = XorY_augment(Htapered, Nqubits // 2)
                decomp           = sorted_insertion_decomposition(Htapered_aug, methodtag='fc')
                element_variance = np.sqrt(variance_of_decomp(decomp, comp_t, Nqubits // 2 + 1, general=True))
                sig_matrix[i,j]  = element_variance
                sig_matrix[j,i]  = element_variance

vals, vecs = np.linalg.eigh(Hsub)
Egs        = vals[0]
c          = vecs[:,0]
cost       = sampling_cost(c, sig_matrix)


with open(output_filename, 'a') as f:
    print(f'''
        
    Final Results: 
        Method              : {1}
        System              : {'H2O'}
        Bond Length         : {bond_length} 
        Ground State Energy : {Egs}
        Cost                : {cost} 
    ''', file=f)