### 18/11/2025 VO create fragments and job, job_desc

from seniority.src.measurement_new.utils_basic import (
    copy_hamiltonian
)
from seniority.src.measurement_new.utils_ferm import (
    orthogonal_transform_obt_tbt,
    obt_phys_spatial_to_spin,
    tbt_phys_spatial_to_spin,
    make_short_H_ferm_op
)
from seniority.src.measurement_new.utils_states import (
    convert_TZ_format_to_sparse_format,
    convert_dense_format_to_sparse_format,
    tz_state_seniority_config,
    compress_state,
    decompress_state,
    create_composite_state
)
from seniority.src.measurement_new.utils_m1_seniority import (
    project_out_seniority_symmetries
)
from seniority.src.measurement_new.utils_m2_factorize import (
    expand_tensor_product,
    expand_tensor_product_for_incomplete_qubit_set,
    get_indices_mapping_2_wvn_vo,
    factorize_state,
    evaluate_fully_classical_factors,
    obtain_coarse_dicts,
    QC_assignment_from_qubit_labels
)
from seniority.src.measurement_new.utils_m3_swap import (
    XorY_augment
)
from seniority.src.measurement_new.utils_m4_partitioning import (
    sorted_insertion_decomposition
)
from seniority.src.measurement_new.utils_results import (
    variance_of_decomp,
    sampling_cost
)
from openfermion import (
    get_sparse_operator,
    jordan_wigner,
    count_qubits
)

import numpy as np
import pickle
import sys

from seniority.src.circuits.circuits_csf import get_csfs_from_dump
from seniority.src.circuits.utils_circuit import get_decomp_circuits_estimators, show_state, verify_circuit_state
from qiskit import QuantumCircuit, transpile
from seniority.src.circuits.circuits_swap import get_parallelswap_subcircuit

from seniority.src.expt import get_shot_allocation
from seniority.src.measurement_new.utils_states import frag_SD_of_decomp

# load Q-SENSE basis states

molecule        = 'h2o'
bond_length     = '1.0'
filename        = f'seniority/data/{molecule}_data/UCSF_sym_comp_for_Praveen_Smik_{bond_length}.dump'

with open(filename, 'rb') as f:
    (
    CSF_tz_states,
    W_amplitudes,
    list_list_theta_CSF,
    list_sym_CSF_vec,
    list_UCSF_tz,
    UCSF_tz_states,
    somos_list,
    psi_GS_UCSF_smik,
    list_orb_rot,
    x_orbrot,
    Enuc,
    obt_spatial,
    tbt_spatial
    ) = pickle.load(f)

CSFs = get_csfs_from_dump(filename, verify_states=True, verbose=True)

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

# process information so that we can taper and factorize the Q-SENSE states

Nstates          = len(UCSF_tz_states)
configs          = [tz_state_seniority_config(tz_state) for tz_state in UCSF_tz_states]
UCSF_information = [get_indices_mapping_2_wvn_vo(CSF_tz_states[i], W_amplitudes[i], Norb) for i in range(Nstates)]

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

def get_quantum_qubits(bra_f, ket_f, bra_labels, ket_labels):
    quantum_qubits = []

    join_partition, coarse_dict_bra, coarse_dict_ket = obtain_coarse_dicts(bra_f, ket_f)
    QC_assignment_dict                               = QC_assignment_from_qubit_labels(bra_labels, ket_labels, join_partition)

    for k, v in QC_assignment_dict.items():
        if v == 'Q':
            for idx in k:
                quantum_qubits.append(idx)
    return quantum_qubits

circuit_cx_counts = []
circuit_depth = []

### job
quantum_indices = [] # list of tuples indicating quantum calculated entries
jobs = {} #i: circuit, shots 
circuits = {} #only circuits
jobs_desc = {} #
sig_frag_mat = {}
frag_zops = {}
H_classical = np.zeros([Nstates, Nstates], dtype=np.complex128) # save only classical parts
tol=1e-5

for i in range(Nstates):
    
    print(f'{i, i}')
    uv = (i, i)

    ket_f      = factorized_tapered_statevectors[i]
    ket_labels = UCSF_information[i][0]
    ket_config = configs[i]

    Htapered        = project_out_seniority_symmetries(Hqub, Nqubits, ket_config, ket_config)
    HQ, ketQ, _, NQ = evaluate_fully_classical_factors(ket_f, ket_f, ket_labels, ket_labels, Htapered)    

    if NQ == 0 or np.sum(np.abs(list(HQ.terms.values()))) <=tol:
        Hsub[i,i] = HQ.constant
        H_classical[i,i] = HQ.constant

    else:
        quantum_indices.append(uv)
        quantum_qubits = get_quantum_qubits(ket_f, ket_f, ket_labels, ket_labels)

        HQsparse   = get_sparse_operator(HQ)
        ketQ       = convert_dense_format_to_sparse_format(ketQ)
        Hsub[i,i] = (ketQ @ HQsparse @ ketQ.T)[0,0]
        H_classical[i, i] = HQ.constant

        HQ              -= HQ.constant
        HQ.compress()
        decomp = sorted_insertion_decomposition(HQ, 'fc')
        var_metric       = variance_of_decomp(decomp, ketQ, NQ, general=True)
        sig_matrix[i,i] = np.sqrt(var_metric)

        ### build circuits!
        decomp, meas_circuits, z_ops = get_decomp_circuits_estimators(HQ, NQ, methodtag='fc')
        csf_circ = CSFs[i].get_tapered_full_circuit(quantum_qubits)

        assert verify_circuit_state(csf_circ, ketQ)
        csf_circ_t = transpile(csf_circ, basis_gates=['u3', 'cx'], optimization_level=3)
        csf_circ_t_frags = [QuantumCircuit.compose(csf_circ_t, c) for c in meas_circuits]
        
        circuits[uv] = csf_circ_t_frags #make estimators for jobs?
        sig_frag_mat[uv] = frag_SD_of_decomp(decomp, ketQ, NQ, general=True)
        frag_zops[uv] = z_ops

        #jobs_desc[]

        #circuits = [QuantumCircuit.compose(csf_circ, m) for m in meas_circuits]

        #verify state with ketQ

        

        # fragment_circuits[(i, i)] = circuits
        # fragment_z_ops[(i, i)] = z_ops
        # fragment_SDs[(i, i)] = frag_SD_of_decomp(decomp, ket_t, Nqubits//2, general=True)

for i in range(Nstates):
    for j in range(Nstates):
        if i > j:
            
            print(f'{i, j}')
            uv = (i, j)

            ij_shift           = 0.5 * (Hsub[i,i] + Hsub[j,j])

            bra_f              = factorized_tapered_statevectors[i]
            bra_labels         = UCSF_information[i][0]
            bra_config         = configs[i]

            ket_f              = factorized_tapered_statevectors[j]
            ket_labels         = UCSF_information[j][0]
            ket_config         = configs[j]

            Htapered           = project_out_seniority_symmetries(Hqub - ij_shift, Nqubits, bra_config, ket_config)
            HQ, braQ, ketQ, NQ = evaluate_fully_classical_factors(bra_f, ket_f, bra_labels, ket_labels, Htapered)

            if NQ == 0 or np.sum(np.abs(list(HQ.terms.values()))) <=tol:
                Hsub[i,j] = HQ.constant
                Hsub[j,i] = HQ.constant

                #off-diagonal used twice
                circuit_cx_counts.append(0)
                circuit_depth.append(0)
                circuit_cx_counts.append(0)
                circuit_depth.append(0)

            else:
                quantum_indices.append(uv)
                quantum_qubits = get_quantum_qubits(bra_f, ket_f, bra_labels, ket_labels)

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

                ### build circuits!
                decomp, meas_circuits, z_ops = get_decomp_circuits_estimators(HQ_aug, NQ + 1, methodtag='fc')
                csf_circ = get_parallelswap_subcircuit(CSFs[i], CSFs[j], quantum_qubits=quantum_qubits, control_qubit_pos=NQ)
                assert verify_circuit_state(csf_circ, comp, truncate_bitstrings=list(range(NQ+1)))
                csf_circ_t = transpile(csf_circ, basis_gates=['u3', 'cx'], optimization_level=3)

                csf_circ_t_frags = [QuantumCircuit.compose(csf_circ_t, c) for c in meas_circuits]
                
                circuits[uv] = csf_circ_t_frags #make estimators for jobs?
                sig_frag_mat[uv] = frag_SD_of_decomp(decomp, comp, NQ + 1, general=True)
                frag_zops[uv] = z_ops
                

vals, vecs = np.linalg.eigh(Hsub)
Egs        = vals[0]
c          = vecs[:,0]
cost       = sampling_cost(c, sig_matrix)


print(f'''
    Estimate Results:
        Method              : {'VO'}
        Molecule            : {molecule}
        Bond Length         : {bond_length}
        Ground State Energy : {Egs}
        Sampling Cost       : {cost}
''')

print('''
Beginning quantum experiments...
      ''')

N = int(cost * 1e6)
shot_alloc = get_shot_allocation(sig_frag_mat, c, N)

#run stuff

print("Allocated {} shots, beginning quantum expts...".format(N))
reps = 10
quant_ests = []

from seniority.src.circuits.utils_circuit import simulate_qiskit_circuit, estimate_diag_from_measurements
from seniority.src.expt import get_willow_noise_model

noise_model = get_willow_noise_model('decoherence')

### run circuits with shots
quant_ests = []
reps=1
for rep in range(reps):
    print("Expt repetition: {}".format(rep))
    H_sampled = np.zeros([Nstates, Nstates], dtype=np.complex128)

    #diagonal
    for i in range(Nstates):
        uv = (i, i)
        if uv in quantum_indices:
            #run expt
            #meas alloc and noise TODO
            counts = [simulate_qiskit_circuit(circ, n, noise_model=noise_model, add_measurements=True) for circ, n in zip(circuits[uv], shot_alloc[uv])]
            estimates = [estimate_diag_from_measurements(z, count_dict) for z, count_dict in zip(frag_zops[uv], counts)]
            mat_est = sum(estimates) + H_classical[i, i]
            H_sampled[i, i] = mat_est
            print("Estimate: {}\nTrue: {}".format(H_sampled[i, i], Hsub[i, i]))
        else:
            H_sampled[i, i] = Hsub[i, i]

    #off-diagonal
    for i in range(Nstates):
        for j in range(Nstates):
            if i > j: 
                uv = (i, j)
                if uv in quantum_indices:
                    #run expt

                    print("\nRunning ({}, {}) off diagonal expt with total {} shots".format(i, j, np.sum(shot_alloc[uv])))
                    counts = [simulate_qiskit_circuit(circ, n, noise_model=noise_model, add_measurements=True) for circ, n in zip(circuits[uv], shot_alloc[uv])] 
                    estimates = [estimate_diag_from_measurements(z, count_dict) for z, count_dict in zip(frag_zops[uv], counts)]

                    mat_est = sum(estimates)
                    H_sampled[i, j] = mat_est
                    H_sampled[j, i] = mat_est
                    print("Estimate: {}\nTrue: {}".format(H_sampled[i, j], Hsub[i, j]))

                else:
                    H_sampled[i, j] = Hsub[i, j]
                    H_sampled[j, i] = Hsub[j, i]

    vals, vecs = np.linalg.eigh(H_sampled)
    quant_Egs        = vals[0]
    print("Energy estimate: {}".format(quant_Egs))
    
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