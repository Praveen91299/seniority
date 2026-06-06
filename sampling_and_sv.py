### script that does sampling currently
# SV can be added

### 18/11/2025 VO create fragments and job, job_desc

from src.measurement_new.utils_ferm import (
    orthogonal_transform_obt_tbt,
    obt_phys_spatial_to_spin,
    tbt_phys_spatial_to_spin,
    make_short_H_ferm_op
)
from src.measurement_new.utils_states import (
    convert_TZ_format_to_sparse_format,
    convert_dense_format_to_sparse_format,
    tz_state_seniority_config,
    compress_state,
    create_composite_state
)
from src.measurement_new.utils_m1_seniority import (
    project_out_seniority_symmetries
)
from src.measurement_new.utils_m2_factorize import (
    get_indices_mapping_2_wvn_vo,
    factorize_state,
    evaluate_fully_classical_factors,
    obtain_coarse_dicts,
    QC_assignment_from_qubit_labels
)
from src.measurement_new.utils_m3_swap import (
    XorY_augment
)
from src.measurement_new.utils_m4_partitioning import (
    sorted_insertion_decomposition
)
from src.measurement_new.utils_results import (
    variance_of_decomp,
    sampling_cost
)
from openfermion import (
    get_sparse_operator,
    jordan_wigner
)

import numpy as np
import pickle

from src.circuits.circuits_csf import get_csfs_from_dump
from src.circuits.utils_circuit import get_decomp_circuits_estimators, verify_circuit_state
from qiskit import QuantumCircuit, transpile
from src.circuits.circuits_swap import get_parallelswap_subcircuit

from src.expt import get_shot_allocation
from src.measurement_new.utils_states import frag_SD_of_decomp

from src.mitigation import determine_tapered_parity
from openfermion import QubitOperator

from src.circuits.utils_circuit import simulate_qiskit_circuit, estimate_diag_from_measurements
from src.utils_simulation import resample_counts

def get_parity_op(csf, quantum_qubits):
    return QubitOperator(''.join(['Z{} '.format(i) for i in range(len(quantum_qubits))]), (-1)**determine_tapered_parity(csf, quantum_qubits))

# load Q-SENSE basis states

molecule        = 'h2o'
bond_length     = '1.0'
filename        = f'data/{molecule}_data/UCSF_sym_comp_for_Praveen_Smik_{bond_length}.dump'

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

##mitigation options TODO

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

        ### build circuits!
        decomp, meas_circuits, z_ops = get_decomp_circuits_estimators(HQ, NQ, methodtag='fc')
        csf_circ = CSFs[i].get_tapered_full_circuit(quantum_qubits)

        #assert verify_circuit_state(csf_circ, ketQ)
        #csf_circ_t = transpile(csf_circ, basis_gates=['u3', 'cx'], optimization_level=3)

        csf_circ_frags = [QuantumCircuit.compose(csf_circ, c) for c in meas_circuits]
        
        circuits[uv] = csf_circ_frags #make estimators for jobs?
        sig_frag_mat[uv] = frag_SD_of_decomp(decomp, ketQ, NQ, general=True)
        sig_matrix[i, i] = np.sum(sig_frag_mat[uv])
        frag_zops[uv] = z_ops

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

                ### build circuits!
                decomp, meas_circuits, z_ops = get_decomp_circuits_estimators(HQ_aug, NQ + 1, methodtag='fc')
                csf_circ = get_parallelswap_subcircuit(CSFs[i], CSFs[j], quantum_qubits=quantum_qubits, control_qubit_pos=NQ)
                #assert verify_circuit_state(csf_circ, comp, truncate_bitstrings=list(range(NQ+1)))
                #csf_circ_t = transpile(csf_circ, basis_gates=['u3', 'cx'], optimization_level=3)
                
                csf_circ_frags = [QuantumCircuit.compose(csf_circ, c) for c in meas_circuits]
                
                circuits[uv] = csf_circ_frags #make estimators for jobs?
                sig_frag_mat[uv] = frag_SD_of_decomp(decomp, comp, NQ + 1, general=True)
                sig_matrix[i, j] = np.sum(sig_frag_mat[uv])
                sig_matrix[j, i] = np.sum(sig_frag_mat[uv])
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

# compile circuits and observables
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2, Session
from src.expt import get_precision_for_shots, openfermion_to_sparse_pauli_op, run_estimator, run_pub, calculate_matrix_std, get_precision_for_shots_std

##### fast noise free simulation
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeQuebec
backend = AerSimulator()#FakeQuebec()
estimator = AerEstimator()#.from_backend(backend)


##### for fast estimator with fake noise model
# from qiskit_aer.primitives import EstimatorV2 as AerEstimator
# from qiskit_ibm_runtime.fake_provider import FakeQuebec
# backend = FakeQuebec()
# estimator = AerEstimator().from_backend(backend)

##### for (slow) estimator with noise model from real device
# from qiskit_aer import AerSimulator
# backend=AerSimulator.from_backend(real_backend)
# print(f"We are using {backend.name}")
# estimator = EstimatorV2(mode=backend)

##### for real device, enable or disable mitigation methods accordingly - NOTE shots will automatically increase for mitigation methods
# service = QiskitRuntimeService()
# real_backend = service.least_busy(
#     simulator=False, operational=True, min_num_qubits=21
# )
# backend = real_backend
# estimator = EstimatorV2(mode=real_backend)
# estimator.options.environment.job_tags = ["qsense_quebec_estimator"]
# estimator.options.resilience_level = 0
# estimator.options.twirling.enable_gates = False
# estimator.options.twirling.enable_measure = False
# estimator.options.dynamical_decoupling.enable = True
# estimator.options.dynamical_decoupling.sequence_type = "XY4"

# Create pass manager for transpilation
pm = generate_preset_pass_manager(optimization_level=3,
                                    backend=backend,
                                    seed_transpiler=0)

circuits_transpiled = {}
fragment_zops_transpiled = {}
PUBs = {} # combine circuits, observables and precision required (shots)
for uv in circuits.keys():
    transpiled_circuits = pm.run(circuits[uv])
    circuits_transpiled[uv] = transpiled_circuits

    observable_transpiled = [openfermion_to_sparse_pauli_op(observable, circuit.num_qubits).apply_layout(transpiled_circuit.layout) for observable, circuit, transpiled_circuit in zip(frag_zops[uv], circuits[uv], transpiled_circuits)]
    fragment_zops_transpiled[uv] = observable_transpiled

    precisions = [get_precision_for_shots_std(sig, shots) for sig, shots in zip(sig_frag_mat[uv], shot_alloc[uv])]

    PUBs[uv] = [(circuit, observable, None, precision) for circuit, observable, precision in zip(transpiled_circuits, observable_transpiled, precisions)]


estimator_matrix = np.zeros([Nstates, Nstates], dtype=np.complex128)
std_dev_dict = {} # uv : dev
for i in range(Nstates):
    #diagonal
    j = i
    uv = (i, j)
    if uv in quantum_indices:
        #run expt
        print(len(PUBs[uv]))
        estimates, sds, results = run_pub(PUBs[uv], estimator)
        estimator_matrix[i, j] = np.sum(estimates) + H_classical[i, j]
        std_dev_dict[uv] = np.sqrt(np.sum([s**2 for s in sds])) #sqrt of variance
        print("Estimator estimate: {} +- {}".format(estimator_matrix[i, j], std_dev_dict[uv]))
    else:
        estimator_matrix[i, i] = Hsub[i, i]
    
    #off-diagonal
    for j in range(i):
        uv = (i, j)
        print(uv)
        if uv in quantum_indices:
            #run expt
            print(len(PUBs[uv]))
            estimates, sds, results = run_pub(PUBs[uv], estimator)
            estimator_matrix[i, j] = np.sum(estimates) + H_classical[i, j]
            estimator_matrix[j, i] = np.sum(estimates) + H_classical[j, i]
            std_dev_dict[uv] = np.sqrt(np.sum([s**2 for s in sds])) #sqrt of variance
            print("Estimator estimate: {} +- {}".format(estimator_matrix[i, j], std_dev_dict[uv]))

        else:
            estimator_matrix[i, j] = Hsub[i, j]
            estimator_matrix[j, i] = Hsub[j, i]

vals, vecs = np.linalg.eigh(estimator_matrix)
c          = vecs[:,0]
std = calculate_matrix_std(c, std_dev_dict)
mean = vals[0]
bias = mean - Egs
rmse = np.sqrt(bias **2 + std ** 2)
print("""mean energy estimate from samples: {}\n
        std: {}      
        bias: {}
        RMSE: {}
      """.format(mean, std, bias, rmse))
