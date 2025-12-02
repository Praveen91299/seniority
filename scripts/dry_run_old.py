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
    frag_SD_of_decomp,
    compress_state
)

from seniority.src.circuits.circuits_csf import get_csfs_from_dump

from math import radians, sin, cos
import pyscf as ps
from pyscf import fci

bond_length     = 1.0
filename        = f'seniority/data/h2o_data/UCSF_sym_comp_for_Praveen_Smik_{bond_length}.dump'
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
CSFs = get_csfs_from_dump(filename, True, True)

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
Hsubest    = np.zeros([Nstates, Nstates], dtype=np.complex128)
sig_matrix = np.zeros([Nstates, Nstates], dtype=np.complex128)
print(len(CSFs))
print(Nstates)

from seniority.src.circuits.utils_circuit import show_state, get_decomp_circuits_estimators, estimate_diag_from_measurements, simulate_qiskit_circuit
from qiskit import QuantumCircuit
from seniority.src.circuits.circuits_swap import parallel_swap, build_ext_swap_circuit

### for experiments
fragment_circuits = {}
fragment_z_ops = {}
estimate_const = {}
fragment_SDs = {}

for i in range(Nstates):

    # with open(output_filename, 'a') as f:
    #     print(i, i, file=f)
    print(i, i)

    ket        = statevectors[i]
    ket_t      = convert_dense_format_to_sparse_format(compress_state(ket.toarray()[0]))
    ket_config = configs[i]

    Htapered                = process_qubit_hamiltonian_to_project_onto_symmetric_subspace(Hqub, Nqubits, ket_config, ket_config)
    after_tapering_constant = Htapered.constant
    estimate_const[(i, i)]  = after_tapering_constant  
    Htapered               -= after_tapering_constant
    Htapered.compress()
    Htapered_sparse         = get_sparse_operator(Htapered + after_tapering_constant, Nqubits // 2)
    Hsub[i,i]               = matrix_element(Htapered_sparse, ket_t, ket_t)

    if i in quantum_indices:
        #decomp          = sorted_insertion_decomposition(Htapered, methodtag='fc')
        decomp, meas_circuits, z_ops = get_decomp_circuits_estimators(Htapered, Norb, methodtag='fc')
        sig_matrix[i,i] = np.sqrt(variance_of_decomp(decomp, ket_t, Nqubits // 2, general=True))

        #circuit sim
        csf_circ = CSFs[i].get_tapered_full_circuit()
        circuits = [QuantumCircuit.compose(csf_circ, m) for m in meas_circuits]

        fragment_circuits[(i, i)] = circuits
        fragment_z_ops[(i, i)] = z_ops
        fragment_SDs[(i, i)] = frag_SD_of_decomp(decomp, ket_t, Nqubits//2, general=True)

for i in range(Nstates):
    for j in range(Nstates):
        if i > j:
            
            print(i, j)

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
                decomp, meas_circuits, z_ops = get_decomp_circuits_estimators(Htapered_aug, 2 * Norb + 1, methodtag='fc')
                element_variance = np.sqrt(variance_of_decomp(decomp, comp_t, Nqubits // 2 + 1, general=True))
                sig_matrix[i,j]  = element_variance
                sig_matrix[j,i]  = element_variance

                uv = (i, j)
                fragment_SDs[uv] = frag_SD_of_decomp(decomp, comp_t, Nqubits // 2 + 1, general=True)
                #circuit and meas
                circuit = parallel_swap(CSFs[i], CSFs[j], control_qubit_pos=Norb)
                circuits = [QuantumCircuit.compose(circuit, meas) for meas in meas_circuits]
                fragment_circuits[uv] = circuits
                fragment_z_ops[uv] = z_ops

vals, vecs = np.linalg.eigh(Hsub)
Egs        = vals[0]
c          = vecs[:,0]
cost       = sampling_cost(c, sig_matrix)

print(f'''
        
#     Final Results: 
#         Method              : {1}
#         System              : {'H2O'}
#         Bond Length         : {bond_length} 
#         Ground State Energy : {Egs}
#         Cost                : {cost} 
#     ''')

print("Running quantum experiments...")

def get_shot_allocation(sig_frag_mat: dict, c, N):
    """
    Returns measurment allocations for QSENSE

    sigma_frag_mat: dict[tup: list[float] ] - dictionary with entries consisting of list of fragment standard deviations
    c: solution state coefficient
    N: total shots

    dict[tup: list[float] ] returns dictionary of measurement shots for some matrix elements

    """

    n = len(c)
    c_abs = np.abs(c)
    sig_matrix = {}#np.zeros((n, n)) # SD of matrix entries
    Muv = {}
    frag_Muv = {}

    M = 0
    for uv in sig_frag_mat:
        sig_matrix[uv] = sum(sig_frag_mat[uv])

        u, v = uv
        if u == v:
            #diagonal
            Muv[uv] = c_abs[u]**2 * sig_matrix[uv]
        else:
            #off diagonal, extra factor of two
            Muv[uv] = 2 * c_abs[u] * c_abs[v] * sig_matrix[uv]
        
        M += Muv[uv]
    #fuv = Muv / np.sum(Muv)

    F = {}
    for uv in sig_frag_mat:
        u, v = uv
        F[uv] = np.ceil(np.abs(N * (Muv[uv] / M) * (np.array(sig_frag_mat[uv]) / sig_matrix[u, v]))) # sets a minimum of 1 shot
        
    
    return F

N =  int(1e6 * cost)
Nuv = get_shot_allocation(fragment_SDs, c, N)
print("Allocated {} shots, beginning quantum expts...".format(N))
reps = 100
quant_ests = []


### run circuits with shots
for rep in range(reps):
    print("Expt repetition: {}".format(rep))
    #diagonal
    for i in range(Nstates):
        uv = (i, i)
        if i in quantum_indices:
            #run expt
            #meas alloc and noise TODO
            print("Running {} diagonal expt with total {} shots".format(i, np.sum(Nuv[uv])))
            counts = [simulate_qiskit_circuit(circ, n) for circ, n in zip(fragment_circuits[uv], Nuv[uv])]
            estimates = [estimate_diag_from_measurements(z, count_dict) for z, count_dict in zip(fragment_z_ops[uv], counts)]
            mat_est = sum(estimates) + estimate_const[uv]
            Hsubest[i, i] = mat_est
            #print("Estimate: {}\nTrue: {}".format(Hsubest[i, i], Hsub[i, i]))
        else:
            Hsubest[i, i] = Hsub[i, i]

    #off-diagonal
    for i in range(Nstates):
        for j in range(Nstates):
            if i > j: 
                uv = (i, j)
                if (i in quantum_indices) or (j in quantum_indices):
                    #run expt

                    print("Running ({}, {}) off diagonal expt with total {} shots".format(i, j, np.sum(Nuv[uv])))
                    counts = [simulate_qiskit_circuit(circ, n) for circ, n in zip(fragment_circuits[uv], Nuv[uv])] 
                    estimates = [estimate_diag_from_measurements(z, count_dict) for z, count_dict in zip(fragment_z_ops[uv], counts)]

                    mat_est = sum(estimates)
                    Hsubest[i, j] = mat_est
                    Hsubest[j, i] = mat_est
                    #print("Estimate: {}\nTrue: {}".format(Hsubest[i, j], Hsub[i, j]))

                else:
                    Hsubest[i, j] = Hsub[i, j]
                    Hsubest[j, i] = Hsub[j, i]

    vals, vecs = np.linalg.eigh(Hsubest)
    quant_Egs        = vals[0]
    quant_ests.append(quant_Egs)

print("Completed {} repetitions...".format(reps))

# with open(output_filename, 'a') as f:
#     print(f'''
        
#     Final Results: 
#         Method              : {1}
#         System              : {'H2O'}
#         Bond Length         : {bond_length} 
#         Ground State Energy : {Egs}
#         Cost                : {cost} 
#     ''', file=f)

N =  int(3 * 1e6 * cost)
Nuv = get_shot_allocation(fragment_SDs, c, N)
print("Allocated {} shots, beginning quantum expts...".format(N))
reps = 10
quant_ests = []

### run circuits with shots
for rep in range(reps):
    print("Expt repetition: {}".format(rep))
    #diagonal
    for i in range(Nstates):
        uv = (i, i)
        if i in quantum_indices:
            #run expt
            #meas alloc and noise TODO
            #print("Running {} diagonal expt with total {} shots".format(i, np.sum(Nuv[uv])))
            counts = [simulate_qiskit_circuit(circ, n) for circ, n in zip(fragment_circuits[uv], Nuv[uv])]
            estimates = [estimate_diag_from_measurements(z, count_dict) for z, count_dict in zip(fragment_z_ops[uv], counts)]
            mat_est = sum(estimates) + estimate_const[uv]
            Hsubest[i, i] = mat_est
            #print("Estimate: {}\nTrue: {}".format(Hsubest[i, i], Hsub[i, i]))
        else:
            Hsubest[i, i] = Hsub[i, i]

    #off-diagonal
    for i in range(Nstates):
        for j in range(Nstates):
            if i > j: 
                uv = (i, j)
                if (i in quantum_indices) or (j in quantum_indices):
                    #run expt

                    #print("Running ({}, {}) off diagonal expt with total {} shots".format(i, j, np.sum(Nuv[uv])))
                    counts = [simulate_qiskit_circuit(circ, n) for circ, n in zip(fragment_circuits[uv], Nuv[uv])] 
                    estimates = [estimate_diag_from_measurements(z, count_dict) for z, count_dict in zip(fragment_z_ops[uv], counts)]

                    mat_est = sum(estimates)
                    Hsubest[i, j] = mat_est
                    Hsubest[j, i] = mat_est
                    #print("Estimate: {}\nTrue: {}".format(Hsubest[i, j], Hsub[i, j]))

                else:
                    Hsubest[i, j] = Hsub[i, j]
                    Hsubest[j, i] = Hsub[j, i]

    vals, vecs = np.linalg.eigh(Hsubest)
    quant_Egs        = vals[0]
    quant_ests.append(quant_Egs)

print("Completed {} repetitions...".format(reps))
mean = np.mean(quant_ests)
bias = mean - Egs
std = np.var(quant_ests)**(1/2)
rmse = np.sqrt(np.mean((np.array(quant_ests) - Egs)**2))
print("""mean energy estimate: {}\n
        std: {}      
        bias: {}
        RMSE: {}
      """.format(mean, std, bias, rmse))