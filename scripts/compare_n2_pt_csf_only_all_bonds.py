"""Compare N2 PT and CSF-only subspaces at every saved bond length.

The 20-spin-orbital Hamiltonian is evaluated directly between the sparse
Slater-determinant expansions.  This is algebraically equivalent to building
the full Jordan--Wigner sparse matrix but avoids allocating a 2^20 operator.
"""

from _bootstrap import REPO_ROOT

import csv
import pickle
import re
from pathlib import Path

import numpy as np

from src.measurement_new.utils_ferm import (
    make_short_H_ferm_op,
    obt_phys_spatial_to_spin,
    orthogonal_transform_obt_tbt,
    tbt_phys_spatial_to_spin,
)
from src.saved_run_io import save_json


DATA_DIR = REPO_ROOT / "data/n2_data"
OUTPUT_DIR = REPO_ROOT / "saved/runtime_runs/n2_pt_csf_only_all_bonds"
ORTHONORMALITY_TOLERANCE = 1e-10


def _bond_length(path: Path) -> float:
    match = re.fullmatch(
        r"Uext_CSF_for_Praveen_Smik_([0-9.]+)\.dump", path.name
    )
    if match is None:
        raise ValueError(f"Unexpected PT dump name: {path.name}.")
    return float(match.group(1))


def _flatten_states(groups):
    return [state for group in groups for state in group]


def _coefficient_matrix(states, determinant_positions):
    coefficients = np.zeros(
        (len(states), len(determinant_positions)), dtype=complex
    )
    for state_index, state in enumerate(states):
        for determinant, coefficient in zip(state[1], state[2]):
            coefficients[
                state_index, determinant_positions[int(determinant)]
            ] = coefficient
    return coefficients


def _apply_fermion_term(
    determinant: int, term: tuple, number_spin_orbitals: int
):
    """Apply an OpenFermion ladder-operator term to one determinant.

    TZ determinant integers use mode 0 as the most-significant bit.  Operators
    act from right to left, and the Jordan--Wigner sign is the parity of
    occupied modes with index below the operated mode.
    """
    sign = 1
    for mode, action in reversed(term):
        bit = 1 << (number_spin_orbitals - 1 - mode)
        occupied = bool(determinant & bit)
        lower_mode_occupations = (
            determinant >> (number_spin_orbitals - mode)
        ).bit_count()
        if lower_mode_occupations % 2:
            sign = -sign
        if action == 1:
            if occupied:
                return None, 0
            determinant |= bit
        else:
            if not occupied:
                return None, 0
            determinant &= ~bit
    return determinant, sign


def _determinant_hamiltonian(
    determinants, fermion_hamiltonian, number_spin_orbitals
):
    positions = {determinant: i for i, determinant in enumerate(determinants)}
    matrix = np.zeros((len(determinants), len(determinants)), dtype=complex)
    terms = list(fermion_hamiltonian.terms.items())
    for column, determinant in enumerate(determinants):
        for term, coefficient in terms:
            output, sign = _apply_fermion_term(
                determinant, term, number_spin_orbitals
            )
            row = positions.get(output)
            if row is not None:
                matrix[row, column] += coefficient * sign
    return matrix


def analyze_dump(path: Path) -> tuple[dict, dict]:
    with path.open("rb") as file:
        (
            reference_groups,
            rotated_groups,
            mp2_amplitudes,
            _optimized_amplitudes,
            orbital_rotations,
            orbital_rotation_parameters,
            nuclear_repulsion,
            one_body_spatial,
            two_body_spatial,
        ) = pickle.load(file)

    if len(orbital_rotations):
        one_body, two_body = orthogonal_transform_obt_tbt(
            orbital_rotation_parameters,
            orbital_rotations,
            one_body_spatial,
            two_body_spatial,
        )
    else:
        one_body = obt_phys_spatial_to_spin(one_body_spatial)
        two_body = tbt_phys_spatial_to_spin(two_body_spatial)
    fermion_hamiltonian = make_short_H_ferm_op(
        nuclear_repulsion, one_body, two_body
    )
    number_spin_orbitals = one_body.shape[0]

    reference_states = _flatten_states(reference_groups)
    rotated_states = _flatten_states(rotated_groups)
    if len(reference_states) != len(rotated_states):
        raise ValueError(
            f"Reference/rotated state counts differ in {path}: "
            f"{len(reference_states)} versus {len(rotated_states)}."
        )
    determinants = sorted(
        {
            int(index)
            for state in reference_states + rotated_states
            for index in state[1]
        }
    )
    determinant_positions = {
        determinant: index for index, determinant in enumerate(determinants)
    }
    determinant_hamiltonian = _determinant_hamiltonian(
        determinants, fermion_hamiltonian, number_spin_orbitals
    )
    reference_coefficients = _coefficient_matrix(
        reference_states, determinant_positions
    )
    rotated_coefficients = _coefficient_matrix(
        rotated_states, determinant_positions
    )

    reference_overlap = (
        reference_coefficients.conj() @ reference_coefficients.T
    )
    rotated_overlap = rotated_coefficients.conj() @ rotated_coefficients.T
    reference_matrix = (
        reference_coefficients.conj()
        @ determinant_hamiltonian
        @ reference_coefficients.T
    )
    rotated_matrix = (
        rotated_coefficients.conj()
        @ determinant_hamiltonian
        @ rotated_coefficients.T
    )
    reference_matrix = (reference_matrix + reference_matrix.conj().T) / 2
    rotated_matrix = (rotated_matrix + rotated_matrix.conj().T) / 2

    number_states = len(reference_states)
    identity = np.eye(number_states)
    reference_overlap_error = float(
        np.max(np.abs(reference_overlap - identity))
    )
    rotated_overlap_error = float(
        np.max(np.abs(rotated_overlap - identity))
    )
    reference_rank = int(
        np.linalg.matrix_rank(reference_overlap, tol=ORTHONORMALITY_TOLERANCE)
    )
    rotated_rank = int(
        np.linalg.matrix_rank(rotated_overlap, tol=ORTHONORMALITY_TOLERANCE)
    )
    if (
        reference_overlap_error > ORTHONORMALITY_TOLERANCE
        or rotated_overlap_error > ORTHONORMALITY_TOLERANCE
        or reference_rank != number_states
        or rotated_rank != number_states
    ):
        raise ValueError(
            f"Basis validation failed for {path}: reference error/rank="
            f"{reference_overlap_error}/{reference_rank}, rotated="
            f"{rotated_overlap_error}/{rotated_rank}."
        )

    reference_eigenvalues, reference_eigenvectors = np.linalg.eigh(
        reference_matrix
    )
    rotated_eigenvalues, rotated_eigenvectors = np.linalg.eigh(rotated_matrix)
    reference_ground_state = (
        reference_eigenvectors[:, 0] @ reference_coefficients
    )
    rotated_ground_state = (
        rotated_eigenvectors[:, 0] @ rotated_coefficients
    )
    fidelity = float(
        abs(np.vdot(reference_ground_state, rotated_ground_state)) ** 2
    )

    angles = [
        float(amplitude_record[1])
        for group in mp2_amplitudes
        for amplitude_record in group
    ]
    csf_energy = float(reference_eigenvalues[0])
    pt_energy = float(rotated_eigenvalues[0])
    record = {
        "bond_length_angstrom": _bond_length(path),
        "dump_file": path,
        "number_states": number_states,
        "number_spin_orbitals": number_spin_orbitals,
        "number_union_determinants": len(determinants),
        "csf_only_energy_hartree": csf_energy,
        "rotated_pt_energy_hartree": pt_energy,
        "rotated_minus_csf_hartree": pt_energy - csf_energy,
        "csf_minus_rotated_hartree": csf_energy - pt_energy,
        "physical_ground_state_fidelity": fidelity,
        "csf_overlap_max_error": reference_overlap_error,
        "rotated_overlap_max_error": rotated_overlap_error,
        "csf_overlap_rank": reference_rank,
        "rotated_overlap_rank": rotated_rank,
        "distinct_mp2_angles": sorted(set(angles), key=abs, reverse=True),
        "maximum_absolute_mp2_angle": (
            max(map(abs, angles)) if angles else 0.0
        ),
    }
    arrays = {
        "determinants": np.asarray(determinants, dtype=np.int64),
        "csf_overlap_matrix": reference_overlap,
        "rotated_pt_overlap_matrix": rotated_overlap,
        "csf_only_matrix": reference_matrix,
        "rotated_pt_matrix": rotated_matrix,
        "csf_only_eigenvalues": reference_eigenvalues,
        "rotated_pt_eigenvalues": rotated_eigenvalues,
        "csf_only_eigenvectors": reference_eigenvectors,
        "rotated_pt_eigenvectors": rotated_eigenvectors,
    }
    return record, arrays


def main() -> None:
    paths = sorted(
        DATA_DIR.glob("Uext_CSF_for_Praveen_Smik_*.dump"),
        key=_bond_length,
    )
    if not paths:
        raise FileNotFoundError(f"No N2 PT dumps found under {DATA_DIR}.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for index, path in enumerate(paths, start=1):
        bond_length = _bond_length(path)
        print(
            f"[{index}/{len(paths)}] Analyzing R={bond_length:g} Angstrom.",
            flush=True,
        )
        record, arrays = analyze_dump(path)
        records.append(record)
        np.savez(
            OUTPUT_DIR / f"bond_{bond_length:g}_matrices.npz", **arrays
        )

    save_json(OUTPUT_DIR / "summary.json.gz", records)
    with (OUTPUT_DIR / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.savez(
        OUTPUT_DIR / "energy_summary.npz",
        bond_length_angstrom=np.asarray(
            [record["bond_length_angstrom"] for record in records]
        ),
        csf_only_energy_hartree=np.asarray(
            [record["csf_only_energy_hartree"] for record in records]
        ),
        rotated_pt_energy_hartree=np.asarray(
            [record["rotated_pt_energy_hartree"] for record in records]
        ),
        rotated_minus_csf_hartree=np.asarray(
            [record["rotated_minus_csf_hartree"] for record in records]
        ),
        physical_ground_state_fidelity=np.asarray(
            [record["physical_ground_state_fidelity"] for record in records]
        ),
        maximum_absolute_mp2_angle=np.asarray(
            [record["maximum_absolute_mp2_angle"] for record in records]
        ),
    )
    print(f"Saved comparisons for {len(records)} bond lengths to {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
