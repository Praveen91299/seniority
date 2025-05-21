from qiskit import QuantumCircuit
import numpy as np
from csf import CSF

def lcs_with_indices(a: list[tuple], b: list[tuple]):
    m, n = len(a), len(b)
    
    # Each cell stores: (lcs_so_far, indices_in_a, indices_in_b)
    dp = [[(0, [], []) for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                length, ia, ib = dp[i][j]
                dp[i+1][j+1] = (length + 1, ia + [i], ib + [j])
            else:
                left = dp[i][j+1]
                up = dp[i+1][j]
                dp[i+1][j+1] = max(left, up, key=lambda x: x[0])

    length, indices_a, indices_b = dp[m][n]
    sequence = [a[i] for i in indices_a]
    return sequence, indices_a, indices_b

def get_hf_circuit(n_orb, ne):
    assert ne <= n_orb*2, "Provided number of electrons {} incompatible with {} orbitals.".format(ne, n_orb)

    qc = QuantumCircuit(2*n_orb)
    for i in range(ne):
        qc.x(i)
    return qc

def get_tapered_hf_circuit(n_orb, ne):
    assert ne <= n_orb*2, "Provided number of electrons {} incompatible with {} orbitals.".format(ne, n_orb)

    qc = QuantumCircuit(n_orb)
    for i in range(ne//2):
        qc.x(i)
    if ne%2:
        qc.x(ne//2)
    return qc


def get_tapered_Tia_circuit(i, a, n_orb):
    qc = QuantumCircuit(n_orb)

    qc.x(a)
    qc.h(a)
    qc.cx(control_qubit=a, target_qubit=i)

    return qc

def get_tapered_Stt_circuit(i, j, a, b, n_orb):
    qc = QuantumCircuit(n_orb)

    # takes care of hf state
    qc.x(i)
    qc.x(j)

    qc.h(i)
    qc.ry(theta=(-2)*np.arctan2(1, np.sqrt(2)), qubit=j)
    qc.x(a)
    qc.x(j)
    qc.cx(j, a)
    qc.cx(i, j)
    qc.ch(a, b)
    qc.cx(b, a)
    qc.cx(i, a)
    qc.cx(i, b)
    qc.x(i)

    return qc
 
def get_tapered_Sss_circuit(i, j, a, b, n_orb):
    qc = QuantumCircuit(n_orb)

    qc.append(get_tapered_Tia_circuit(i, a, n_orb).to_gate(), qc.qubits)
    qc.append(get_tapered_Tia_circuit(j, b, n_orb).to_gate(), qc.qubits)

    return qc


def get_tapered_ctrl_exc_rot(i, a, n_orb, theta):
    """
    Returns tapered rotation of theta, controlled on qubit 0

    """
    qc = QuantumCircuit(n_orb+1)

    c, i, a = 0, i+1, a+1

    qc.cx(i, a)

    qc.ry(theta/4, i)
    qc.cx(a, i)
    qc.ry(-theta/4, i)
    qc.cx(c, i)
    qc.ry(theta/4, i)
    qc.cx(a, i)
    qc.ry(-theta/4, i)
    qc.cx(c, i)

    qc.cx(i, a)
    
    return qc

def get_tapered_ctrl_exc_rot_comb(i, a, n_orb, theta0, theta1):
    """
    Returns tapered combined rotation for theta0 and theta1 
    conditioned on the control on being |0> and |1> respectively, where qubit 0 is control

    """

    qc = QuantumCircuit(n_orb+1)
    c, i, a = 0, i+1, a+1
    delta = theta1 - theta0
    sigma = theta1 + theta0

    qc.cx(i, a)

    qc.cx(c, i)
    qc.ry(-sigma/4, i)
    qc.cx(a, i)
    qc.ry(delta/4, i)
    qc.cx(c, i)
    qc.ry(-delta/4, i)
    qc.cx(a, i)
    qc.ry(sigma/4, i)
    
    qc.cx(i, a)

    return qc

from qiskit.circuit import Parameter


def build_ext_swap_circuit(CSF0: CSF, CSF1: CSF):
    """
    Build parameterized tapered circuit for extended swap test, with different CSFs and Unitaries

    CSF1, CSF2 : tuples to specify type and qubit positions

    qwc measurement circuit missing!!

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
    qc.append(get_tapered_hf_circuit(CSF0.n_orb, CSF0.ne).to_gate(), target_qubits) # initial HF state

    csf0 = CSF0.get_tapered_circuit(False).to_gate()
    csf1 = CSF1.get_tapered_circuit(False).to_gate()

    qc.append(csf0.control(1), [control_qubit] + target_qubits)
    qc.x(control_qubit)
    qc.append(csf1.control(1), [control_qubit] + target_qubits)

    ## Add U
    # discard Us that act on odd parity orbitals
    
    
    ## make parameters
    # modify to add name of csf as identifier
    exc_list0, exc_list1 = CSF0.get_excitations(), CSF1.get_excitations()
    theta0, theta1 = CSF0.get_thetas(), CSF1.get_thetas()

    # find LCS of elements
    exc_common_list, idx0, idx1 = lcs_with_indices(exc_list0, exc_list1)

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
                u = get_tapered_ctrl_exc_rot(exc[0], exc[1], n_orb, theta0[last0 + j0])
                qc.append(u.to_gate(), [control_qubit] + target_qubits)
            qc.x(control_qubit)

        #1
        if len(exc_sep_1) > 0:
            for j1, exc in enumerate(exc_sep_1):
                u = get_tapered_ctrl_exc_rot(exc[0], exc[1], n_orb, theta1[last1 + j1])
                qc.append(u.to_gate(), [control_qubit] + target_qubits)

        #common
        exc = exc_list0[i0]
        assert exc == exc_list1[i1]

        u = get_tapered_ctrl_exc_rot_comb(exc[0], exc[1], n_orb, theta0[i0], theta1[i1])
        qc.append(u.to_gate(), [control_qubit] + target_qubits)

        last0 = i0 + 1
        last1 = i1 + 1

    #any remaining rotations
    if last0 < len(exc_list0):
        exc_sep_0 = exc_list0[last0:]

        qc.x(control_qubit)
        for i0, exc in enumerate(exc_sep_0):
            u = get_tapered_ctrl_exc_rot(exc[0], exc[1], n_orb, theta0[last0 + i0])
            qc.append(u.to_gate(), [control_qubit] + target_qubits)
        qc.x(control_qubit)
        
    if last1 < len(exc_list1):
        exc_sep_1 = exc_list1[last1:]

        for i1, exc in enumerate(exc_sep_1):
            u = get_tapered_ctrl_exc_rot(exc[0], exc[1], n_orb, theta1[last1 + i1])
            qc.append(u.to_gate(), [control_qubit] + target_qubits)


    ## TODO Add measurement circuit

    return qc #circuit and parameter map