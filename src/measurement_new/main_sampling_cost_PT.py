from utils_basic import (
    copy_hamiltonian
)
from utils_ferm import (
    orthogonal_transform_obt_tbt,
    obt_phys_spatial_to_spin,
    tbt_phys_spatial_to_spin,
    make_short_H_ferm_op
)
from utils_states import (
    convert_TZ_format_to_sparse_format,
    convert_dense_format_to_sparse_format,
    tz_state_seniority_config,
    compress_state,
    decompress_state,
    create_composite_state
)
from utils_m1_seniority import (
    project_out_seniority_symmetries
)
from utils_m2_factorize import (
    expand_tensor_product,
    expand_tensor_product_for_incomplete_qubit_set,
    get_indices_mapping_2_wvn,
    factorize_state,
    evaluate_fully_classical_factors
)
from utils_m3_swap import (
    XorY_augment
)
from utils_m4_partitioning import (
    sorted_insertion_decomposition
)
from utils_results import (
    variance_of_decomp,
    sampling_cost
)
from openfermion import (
    get_sparse_operator,
    jordan_wigner
)

import numpy as np
import pickle
import sys



# load Q-SENSE basis states

molecule        = sys.argv[1]
bond_length     = float(sys.argv[2])
filename        = f'{molecule}_data/Uext_CSF_for_Praveen_Smik_{bond_length}.dump'
output_filename = f'main_outputs/{molecule}_{bond_length}_PT_serial'

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

# rotate orbitals and obtain Hamiltonian operator

if len(list_orb_rot) != 0:
    obt, tbt = orthogonal_transform_obt_tbt(x_orbrot,list_orb_rot,obt_spatial,tbt_spatial)
else:
    obt = obt_phys_spatial_to_spin(obt_spatial)
    tbt = tbt_phys_spatial_to_spin(tbt_spatial)

Hfer    = make_short_H_ferm_op(Enuc, obt, tbt)
Hqub    = jordan_wigner(Hfer)

Nqubits = obt.shape[0]
Norb    = Nqubits // 2
dim     = 2 ** Nqubits

# obtain relevant information about Q-SENSE states (UCSFs, CSFs, W information) in a linear list

UCSF_tz_states = []
CSF_tz_states  = []
W_amplitudes   = []

for i, ucsf_list in enumerate(list_list_Uext_mp2_CSF):
    for j, ucsf in enumerate(ucsf_list):
        UCSF_tz_states.append(ucsf)
        CSF_tz_states.append(list_list_refCSF[i][j])
        W_amplitudes.append(list_list_Uext_mp2_ampld[i])

# process information so that we can taper and factorize the Q-SENSE states

Nstates          = len(UCSF_tz_states)
configs          = [tz_state_seniority_config(tz_state) for tz_state in UCSF_tz_states]
UCSF_information = [get_indices_mapping_2_wvn(CSF_tz_states[i], W_amplitudes[i], Norb) for i in range(Nstates)]

SW_list          = [tuple([k for k, v in UCSF_information[i][0].items() if v == 'W']) for i in range(Nstates)]
SV_list          = [tuple([k for k, v in UCSF_information[i][0].items() if v == 'V']) for i in range(Nstates)]
SN_list          = [tuple([k for k, v in UCSF_information[i][0].items() if v == 'N']) for i in range(Nstates)]
state_type_list  = [UCSF_information[i][1] for i in range(Nstates)]

# taper and factorize the Q-SENSE basis states

statevectors                    = [convert_TZ_format_to_sparse_format(dim, tz_state) for tz_state in UCSF_tz_states]
tapered_statevectors            = [convert_dense_format_to_sparse_format(compress_state(psi.toarray()[0])) for psi in statevectors]
factorized_tapered_statevectors = [factorize_state(tapered_statevectors[i], SW_list[i], SV_list[i], SN_list[i], state_type_list[i]) 
                                   for i in range(Nstates)]


Hsub       = np.zeros([Nstates, Nstates], dtype=np.complex128)
sig_matrix = np.zeros([Nstates, Nstates], dtype=np.complex128)

for i in range(Nstates):
    with open(output_filename, 'a') as f:
        print(f'{i, i}', file=f)

    ket_f      = factorized_tapered_statevectors[i]
    ket_labels = UCSF_information[i][0]
    ket_config = configs[i]

    Htapered        = project_out_seniority_symmetries(Hqub, Nqubits, ket_config, ket_config)
    HQ, ketQ, _, NQ = evaluate_fully_classical_factors(ket_f, ket_f, ket_labels, ket_labels, Htapered)

    if NQ == 0:
        Hsub[i,i] = HQ.constant

    else:
        HQsparse   = get_sparse_operator(HQ)
        ketQ       = convert_dense_format_to_sparse_format(ketQ)
        Hsub[i,i] = (ketQ @ HQsparse @ ketQ.T)[0,0]

        HQ              -= HQ.constant
        HQ.compress()
        decomp = sorted_insertion_decomposition(HQ, 'fc')
        var_metric       = variance_of_decomp(decomp, ketQ, NQ, general=True)
        sig_matrix[i,i] = np.sqrt(var_metric)

for i in range(Nstates):
    for j in range(Nstates):
        if i > j:
            with open(output_filename, 'a') as f:
                print(f'{i, j}', file=f)

            ij_shift           = 0.5 * (Hsub[i,i] + Hsub[j,j])

            bra_f              = factorized_tapered_statevectors[i]
            bra_labels         = UCSF_information[i][0]
            bra_config         = configs[i]

            ket_f              = factorized_tapered_statevectors[j]
            ket_labels         = UCSF_information[j][0]
            ket_config         = configs[j]

            Htapered           = project_out_seniority_symmetries(Hqub - ij_shift, Nqubits, bra_config, ket_config)
            HQ, braQ, ketQ, NQ = evaluate_fully_classical_factors(bra_f, ket_f, bra_labels, ket_labels, Htapered)

            if NQ == 0:
                Hsub[i,j] = HQ.constant
                Hsub[j,i] = HQ.constant

            else:
                HQsparse         = get_sparse_operator(HQ, NQ)
                braQ             = convert_dense_format_to_sparse_format(braQ)
                ketQ             = convert_dense_format_to_sparse_format(ketQ)
                Hsub[i,j]       = (braQ @ HQsparse @ ketQ.T)[0,0]
                Hsub[j,i]       = (braQ @ HQsparse @ ketQ.T)[0,0]

                comp             = create_composite_state(braQ, ketQ, NQ)
                HQ_aug           = XorY_augment(HQ, NQ)
                decomp           = sorted_insertion_decomposition(HQ_aug, 'fc')
                var_metric       = variance_of_decomp(decomp, comp, NQ + 1, general=True)
                sig_matrix[i,j] = np.sqrt(var_metric)
                sig_matrix[j,i] = np.sqrt(var_metric)

vals, vecs = np.linalg.eigh(Hsub)
Egs        = vals[0]
c          = vecs[:,0]
cost       = sampling_cost(c, sig_matrix)

with open(output_filename, 'a') as f:
    print(f'''
        Final Results:
            Method              : {'PT'}
            Molecule            : {molecule}
            Bond Length         : {bond_length}
            Ground State Energy : {Egs}
            Sampling Cost       : {cost}
    ''', file=f)