import numpy as np
from seniority.circuits_csf import CSF, get_tapered_hf_circuit
from seniority.circuits_pair_excitation import PairedExcitationRotation
from qiskit import QuantumCircuit

def lcs_with_indices(a: list[PairedExcitationRotation], b: list[PairedExcitationRotation]):
    def check_exc_equality(exc_a: PairedExcitationRotation, exc_b: PairedExcitationRotation):
        """
        Return True if orbital indices are the same

        """
        return exc_a.excitations == exc_b.excitations
    
    m, n = len(a), len(b)
    
    # Each cell stores: (lcs_so_far, indices_in_a, indices_in_b)
    dp = [[(0, [], []) for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if check_exc_equality(a[i], b[j]):
                length, ia, ib = dp[i][j]
                dp[i+1][j+1] = (length + 1, ia + [i], ib + [j])
            else:
                left = dp[i][j+1]
                up = dp[i+1][j]
                dp[i+1][j+1] = max(left, up, key=lambda x: x[0])

    length, indices_a, indices_b = dp[m][n]
    sequence = [a[i] for i in indices_a]
    return sequence, indices_a, indices_b

def build_ext_swap_circuit(CSF0: CSF, CSF1: CSF, verbose=False):
    """
    Build parameterized tapered circuit for extended swap test, with different CSFs and Unitaries to estimate
        Re[<CSF1| H |CSF0>]
    
    CSF0, CSF1: CSF - CSF class objects, controlled on 0, 1 respectively 

    Cost reduction with identifying longest ordered subsequence of shared unitaries
    
    TODO qwc measurement circuit missing!!

    """
    assert CSF0.n_orb == CSF1.n_orb, "Incompatible CSFs"
    assert CSF0.ne == CSF1.ne, "Incompatible number of electrons"

    n_orb = CSF0.n_orb

    qc = QuantumCircuit(n_orb+1)
    control_qubit = qc.qubits[0]
    target_qubits = qc.qubits[1:]
    qc.h(0)

    ## CSF state prep
    #hf
    occ0 = CSF0.get_doubly_occ_orbitals()
    occ1 = CSF1.get_doubly_occ_orbitals()

    assert len(occ0) == len(occ1)

    occ0_only = []
    occ1_only = []
    common = []

    for idx, i in enumerate(occ0):
        if occ0[idx] == occ1[idx] and occ0[idx] == 1:
            common.append(target_qubits[idx])
        elif occ1[idx]:
            occ1_only.append(target_qubits[idx])
        elif occ0[idx]:
            occ0_only.append(target_qubits[idx])

    if len(common) > 0: qc.x(common)

    if len(occ0_only) > 0: qc.cx(control_qubit, occ0_only, ctrl_state=0)
    if len(occ1_only) > 0: qc.cx(control_qubit, occ1_only, ctrl_state=1)

    csf0 = CSF0.get_tapered_csf_circuit(False).to_gate()
    csf1 = CSF1.get_tapered_csf_circuit(False).to_gate()

    qc.append(csf0.control(1, ctrl_state=0), [control_qubit] + target_qubits)
    qc.append(csf1.control(1, ctrl_state=1), [control_qubit] + target_qubits)
    
    ## make parameters
    # modify to add name of csf as identifier
    exc_list0, exc_list1 = CSF0.get_excitations(), CSF1.get_excitations()

    # find LCS of elements
    exc_common_list, idx0, idx1 = lcs_with_indices(exc_list0, exc_list1)

    assert len(idx0) == len(idx1)

    if verbose:
        print("{} common pair excitations found.".format(len(idx0)))

    last0 = 0
    last1 = 0

    for i0, i1 in zip(idx0, idx1):
        #implement all before
        exc_sep_0 = exc_list0[last0: i0]
        exc_sep_1 = exc_list1[last1: i1]

        #0
        if len(exc_sep_0) > 0:
            qc.x(control_qubit)
            for j0, exc in enumerate(exc_sep_0):
                exc.append_controlled_tapered_circuit(qc, control_qubit, target_qubits)

            qc.x(control_qubit)

        #1
        if len(exc_sep_1) > 0:
            for j1, exc in enumerate(exc_sep_1):
                exc.append_controlled_tapered_circuit(qc, control_qubit, target_qubits)

        #common
        exc = exc_list0[i0]
        exc_other = exc_list1[i1]
        assert exc.get_excitations() == exc_other.get_excitations()

        u = exc.append_combined_controlled_tapered_circuit(exc_other, qc, control_qubit, target_qubits, 0)

        last0 = i0 + 1
        last1 = i1 + 1

    #any remaining rotations
    if last0 < len(exc_list0):
        exc_sep_0 = exc_list0[last0:]

        qc.x(control_qubit)
        for i0, exc in enumerate(exc_sep_0):
            exc.append_controlled_tapered_circuit(qc, control_qubit, target_qubits)
        qc.x(control_qubit)
        
    if last1 < len(exc_list1):
        exc_sep_1 = exc_list1[last1:]

        for i1, exc in enumerate(exc_sep_1):
            exc.append_controlled_tapered_circuit(qc, control_qubit, target_qubits)


    ## TODO Add measurement circuit

    return qc, exc_common_list #circuit and parameter map