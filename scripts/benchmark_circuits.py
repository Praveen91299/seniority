###
# Benchmark CNOT counts for deepest circuit, import csf data from dump files
# MOVE THIS SCRIPT OUT OF THE seniority DIRECTORY BEFORE RUNNING!!
###

from seniority.src.circuits.circuits_csf import CSF, get_state_from_csf_data, determine_t_vec, compare_states, get_tapered_state_from_UCSF, get_Uext_csfs_from_dump, get_csfs_from_dump
from seniority.src.circuits.circuits_pair_excitation import PairedExcitationRotation, SymmetricPairedExcitationRotation
from seniority.src.circuits.utils_circuit import show_state

h2o = 'H2O_r1.0_actmo1_6_orbopt_E0iath0_Uth1e-5_Praveen_Smik.dump'
n2 = 'CSF_UCSF_GS_optU_optorb_flexibleU_matchstate_E0select_ia_1.1_Praveen_Smik.dump'
n2_uext = 'Uext_CSF_for_Praveen_Smik_1.1.dump'

### choose file here!!!
csfs = get_csfs_from_dump(input_file=n2, verify_states=True)
#csfs = get_Uext_csfs_from_dump(input_file=n2_uext, use_opt_amplitudes=True, verify_states=False)

### benchmark circuits
### finding largest circuit

from qiskit import transpile
from seniority.circuits_swap import build_ext_swap_circuit
from seniority.src.circuits.utils_circuit import count_cx_gates

max_cnot = 0
max_csf_pair = ()
for i, c1 in enumerate(csfs):
    for j, c2 in enumerate(csfs):
        if i is not j:
            ext, common = build_ext_swap_circuit(c1, c2)
            # ext_t = transpile(ext, basis_gates=['u3', 'cx'], optimization_level=2)
            # count = count_cx_gates(ext_t)
        else:
            ext = c1.get_tapered_full_circuit()

        ext_t = transpile(ext, basis_gates=['u3', 'cx'], optimization_level=3)
        count = count_cx_gates(ext_t)

        if count > max_cnot:
            max_cnot = count
            max_csf_pair = (i, j)

csf0 = csfs[max_csf_pair[0]]
csf1 = csfs[max_csf_pair[1]]
ext, common_exc = build_ext_swap_circuit(csf0, csf1, True)
print("Out of {} CSFs, pair {} was found with largest circuits".format(len(csfs), max_csf_pair))
print("Genealogy vectors: ")
print(csf0.t_vec, " with {} excitations".format(len(csf0.excitations)))
print(csf1.t_vec, " with {} excitations".format(len(csf1.excitations)))
ext_t = transpile(ext, basis_gates=['u3', 'cx'], optimization_level=3)

### actual device specifications and mapping

from qiskit_ibm_runtime.fake_provider import *
from qiskit.visualization import plot_coupling_map
import rustworkx

### select backend
backend = FakeMarrakesh()
properties = backend.properties()

print("IBMQ Backend: ", backend.name)
print("Number of qubits: ", backend.num_qubits)
#rustworkx.visualization.mpl_draw(backend.coupling_map.graph)
print("Basis gates:", backend.configuration().basis_gates)

min_count = 10000
min_count_tqc = None
n_trials = 20 ### number of transpilation attempts

for i in range(n_trials):
    qc = transpile(ext, basis_gates=['u3', 'cx'])
    tqc = transpile(ext,
                    backend=backend,
                    optimization_level=3,
                    routing_method='sabre')
    
    count = tqc.num_nonlocal_gates()
    
    if count < min_count:
        min_count = count
        min_count_tqc = tqc

print("Initial gate count: ", count_cx_gates(qc))
print("Initial circuit depth: ", qc.depth())

print("Transpiled gate count: ", min_count_tqc.num_nonlocal_gates())
print("Transpiled circuit depth: ", min_count_tqc.depth())

# save_file = "./CircuitFiles/n2_Uext_noarch.qasm"
# save_file_transpiled = "./CircuitFiles/n2_Uext_transpiled.qasm"
# from qiskit import qasm2
# print("Saving untranspiled circuit at: {}".format(save_file))
# print("Saving transpiled circuit at: {}".format(save_file_transpiled))

# qasm2.dump(ext, save_file)
# qasm2.dump(ext_t, save_file_transpiled)

from collections import Counter

gate_counts = Counter()
cx_qubits = []

tqc = min_count_tqc

fidelity = 1.0

# Single-qubit gates
for instr, qargs, _ in tqc.data:
    if instr.num_qubits == 1:
        fidelity *= (1 - 1e-4)

print(fidelity)

for instr, qargs, _ in tqc.data:
    if instr.num_qubits == 2:
        fidelity *= (1 - 1e-3)

print(fidelity)

# Readout errors
for qubit in range(tqc.num_qubits):
    param = properties.readout_error(qubit)
    fidelity *= (1 - param)

# Final result
print(f"Estimated implementation fidelity: {fidelity:.6f}")