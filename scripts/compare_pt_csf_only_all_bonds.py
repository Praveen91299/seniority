"""Compare H2O PT and CSF-only subspaces at every saved bond length.

For each trusted ``Uext_CSF`` dump, the full spin-orbital Hamiltonian is
constructed and projected independently into (1) the reference-CSF basis and
(2) the saved MP2-pair-rotated PT basis.  Both bases are checked for
orthonormality and full rank before ordinary Hermitian diagonalization.
"""

from _bootstrap import REPO_ROOT

import csv
import pickle
import re
from pathlib import Path

import numpy as np
from openfermion import get_sparse_operator, jordan_wigner
from scipy import sparse

from src.measurement_new.utils_ferm import (
    make_short_H_ferm_op,
    obt_phys_spatial_to_spin,
    orthogonal_transform_obt_tbt,
    tbt_phys_spatial_to_spin,
)
from src.measurement_new.utils_states import (
    convert_TZ_format_to_sparse_format,
)
from src.saved_run_io import save_json


DATA_DIR = REPO_ROOT / "data/h2o_data"
OUTPUT_DIR = REPO_ROOT / "saved/runtime_runs/pt_csf_only_all_bonds"
ORTHONORMALITY_TOLERANCE = 1e-10


def _bond_length(path: Path) -> float:
    match = re.fullmatch(
        r"Uext_CSF_for_Praveen_Smik_([0-9.]+)\.dump", path.name
    )
    if match is None:
        raise ValueError(f"Unexpected PT dump name: {path.name}.")
    return float(match.group(1))


def _state_matrix(groups, dimension: int) -> sparse.csr_matrix:
    return sparse.vstack(
        [
            convert_TZ_format_to_sparse_format(dimension, state).astype(
                complex
            )
            for group in groups
            for state in group
        ]
    ).tocsr()


def _project_hamiltonian(states, hamiltonian):
    overlap = (states @ states.getH()).toarray()
    matrix = (states @ hamiltonian @ states.getH()).toarray()
    matrix = (matrix + matrix.conj().T) / 2
    return overlap, matrix


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

    number_spin_orbitals = one_body.shape[0]
    dimension = 1 << number_spin_orbitals
    hamiltonian = get_sparse_operator(
        jordan_wigner(
            make_short_H_ferm_op(
                nuclear_repulsion, one_body, two_body
            )
        ),
        n_qubits=number_spin_orbitals,
    )
    reference_states = _state_matrix(reference_groups, dimension)
    rotated_states = _state_matrix(rotated_groups, dimension)
    if reference_states.shape != rotated_states.shape:
        raise ValueError(
            f"Reference/rotated basis shapes differ in {path}: "
            f"{reference_states.shape} versus {rotated_states.shape}."
        )

    reference_overlap, reference_matrix = _project_hamiltonian(
        reference_states, hamiltonian
    )
    rotated_overlap, rotated_matrix = _project_hamiltonian(
        rotated_states, hamiltonian
    )
    number_states = reference_states.shape[0]
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
    reference_ground_state = np.asarray(
        reference_eigenvectors[:, 0] @ reference_states
    ).ravel()
    rotated_ground_state = np.asarray(
        rotated_eigenvectors[:, 0] @ rotated_states
    ).ravel()
    fidelity = float(
        abs(np.vdot(reference_ground_state, rotated_ground_state)) ** 2
    )

    angles = [
        float(amplitude_record[1])
        for group in mp2_amplitudes
        for amplitude_record in group
    ]
    distinct_angles = sorted(set(angles), key=abs, reverse=True)
    csf_energy = float(reference_eigenvalues[0])
    pt_energy = float(rotated_eigenvalues[0])
    record = {
        "bond_length_angstrom": _bond_length(path),
        "dump_file": path,
        "number_states": number_states,
        "csf_only_energy_hartree": csf_energy,
        "rotated_pt_energy_hartree": pt_energy,
        "rotated_minus_csf_hartree": pt_energy - csf_energy,
        "csf_minus_rotated_hartree": csf_energy - pt_energy,
        "physical_ground_state_fidelity": fidelity,
        "csf_overlap_max_error": reference_overlap_error,
        "rotated_overlap_max_error": rotated_overlap_error,
        "csf_overlap_rank": reference_rank,
        "rotated_overlap_rank": rotated_rank,
        "distinct_mp2_angles": distinct_angles,
        "maximum_absolute_mp2_angle": (
            max(map(abs, angles)) if angles else 0.0
        ),
    }
    arrays = {
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
        raise FileNotFoundError(f"No H2O PT dumps found under {DATA_DIR}.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    energy_rows = []
    for index, path in enumerate(paths, start=1):
        bond_length = _bond_length(path)
        print(
            f"[{index}/{len(paths)}] Analyzing R={bond_length:g} Angstrom.",
            flush=True,
        )
        record, arrays = analyze_dump(path)
        records.append(record)
        energy_rows.append(
            [
                record["bond_length_angstrom"],
                record["csf_only_energy_hartree"],
                record["rotated_pt_energy_hartree"],
                record["rotated_minus_csf_hartree"],
                record["physical_ground_state_fidelity"],
                record["maximum_absolute_mp2_angle"],
            ]
        )
        np.savez(
            OUTPUT_DIR / f"bond_{bond_length:g}_matrices.npz", **arrays
        )

    save_json(OUTPUT_DIR / "summary.json.gz", records)
    fields = list(records[0])
    with (OUTPUT_DIR / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    energy_rows = np.asarray(energy_rows, dtype=float)
    np.savez(
        OUTPUT_DIR / "energy_summary.npz",
        bond_length_angstrom=energy_rows[:, 0],
        csf_only_energy_hartree=energy_rows[:, 1],
        rotated_pt_energy_hartree=energy_rows[:, 2],
        rotated_minus_csf_hartree=energy_rows[:, 3],
        physical_ground_state_fidelity=energy_rows[:, 4],
        maximum_absolute_mp2_angle=energy_rows[:, 5],
    )
    print(f"Saved comparisons for {len(records)} bond lengths to {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
