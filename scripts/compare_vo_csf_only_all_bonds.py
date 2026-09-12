"""Compare H2O VO and CSF-only subspaces at every saved geometry.

The two projections use the same orbital-rotated electronic Hamiltonian.  The
CSF-only calculation uses ``CSF_tz_states`` from each VO dump, while the VO
calculation uses the corresponding optimized ``UCSF_tz_states``.  Thus the
comparison removes only the state-level pair-rotation unitaries.
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
OUTPUT_DIR = REPO_ROOT / "saved/runtime_runs/vo_csf_only_all_bonds"
ORTHONORMALITY_TOLERANCE = 1e-10


def _bond_length(path: Path) -> float:
    match = re.fullmatch(
        r"UCSF_sym_comp_for_Praveen_Smik_([0-9.]+)\.dump", path.name
    )
    if match is None:
        raise ValueError(f"Unexpected VO dump name: {path.name}.")
    return float(match.group(1))


def _state_matrix(states, dimension: int) -> sparse.csr_matrix:
    """Return a matrix whose rows are the saved many-body states."""
    return sparse.vstack(
        [
            convert_TZ_format_to_sparse_format(dimension, state).astype(
                complex
            )
            for state in states
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
            csf_states,
            _w_amplitudes,
            rotation_angles_by_state,
            _symmetry_resolved_csf_vectors,
            _unitaries,
            rotated_vo_states,
            _active_orbitals,
            _saved_vo_ground_coefficients,
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
    csf_state_matrix = _state_matrix(csf_states, dimension)
    vo_state_matrix = _state_matrix(rotated_vo_states, dimension)
    if csf_state_matrix.shape != vo_state_matrix.shape:
        raise ValueError(
            f"CSF/VO basis shapes differ in {path}: "
            f"{csf_state_matrix.shape} versus {vo_state_matrix.shape}."
        )

    csf_overlap, csf_matrix = _project_hamiltonian(
        csf_state_matrix, hamiltonian
    )
    vo_overlap, vo_matrix = _project_hamiltonian(
        vo_state_matrix, hamiltonian
    )
    number_states = csf_state_matrix.shape[0]
    identity = np.eye(number_states)
    csf_overlap_error = float(np.max(np.abs(csf_overlap - identity)))
    vo_overlap_error = float(np.max(np.abs(vo_overlap - identity)))
    csf_rank = int(
        np.linalg.matrix_rank(csf_overlap, tol=ORTHONORMALITY_TOLERANCE)
    )
    vo_rank = int(
        np.linalg.matrix_rank(vo_overlap, tol=ORTHONORMALITY_TOLERANCE)
    )
    if (
        csf_overlap_error > ORTHONORMALITY_TOLERANCE
        or vo_overlap_error > ORTHONORMALITY_TOLERANCE
        or csf_rank != number_states
        or vo_rank != number_states
    ):
        raise ValueError(
            f"Basis validation failed for {path}: CSF error/rank="
            f"{csf_overlap_error}/{csf_rank}, VO="
            f"{vo_overlap_error}/{vo_rank}."
        )

    csf_eigenvalues, csf_eigenvectors = np.linalg.eigh(csf_matrix)
    vo_eigenvalues, vo_eigenvectors = np.linalg.eigh(vo_matrix)
    csf_ground_state = np.asarray(
        csf_eigenvectors[:, 0] @ csf_state_matrix
    ).ravel()
    vo_ground_state = np.asarray(
        vo_eigenvectors[:, 0] @ vo_state_matrix
    ).ravel()
    fidelity = float(abs(np.vdot(csf_ground_state, vo_ground_state)) ** 2)

    angles = (
        np.concatenate(
            [
                np.asarray(values, dtype=float).ravel()
                for values in rotation_angles_by_state
            ]
        )
        if rotation_angles_by_state
        else np.empty(0)
    )
    csf_energy = float(csf_eigenvalues[0])
    vo_energy = float(vo_eigenvalues[0])
    record = {
        "bond_length_angstrom": _bond_length(path),
        "dump_file": path,
        "number_states": number_states,
        "csf_only_energy_hartree": csf_energy,
        "rotated_vo_energy_hartree": vo_energy,
        "rotated_minus_csf_hartree": vo_energy - csf_energy,
        "csf_minus_rotated_hartree": csf_energy - vo_energy,
        "physical_ground_state_fidelity": fidelity,
        "csf_overlap_max_error": csf_overlap_error,
        "rotated_vo_overlap_max_error": vo_overlap_error,
        "csf_overlap_rank": csf_rank,
        "rotated_vo_overlap_rank": vo_rank,
        "number_rotation_parameters": int(angles.size),
        "maximum_absolute_rotation_angle": (
            float(np.max(np.abs(angles))) if angles.size else 0.0
        ),
        "rms_rotation_angle": (
            float(np.sqrt(np.mean(angles**2))) if angles.size else 0.0
        ),
    }
    arrays = {
        "csf_overlap_matrix": csf_overlap,
        "rotated_vo_overlap_matrix": vo_overlap,
        "csf_only_matrix": csf_matrix,
        "rotated_vo_matrix": vo_matrix,
        "csf_only_eigenvalues": csf_eigenvalues,
        "rotated_vo_eigenvalues": vo_eigenvalues,
        "csf_only_eigenvectors": csf_eigenvectors,
        "rotated_vo_eigenvectors": vo_eigenvectors,
    }
    return record, arrays


def main() -> None:
    paths = [
        path
        for path in DATA_DIR.glob(
            "UCSF_sym_comp_for_Praveen_Smik_*.dump"
        )
        if re.fullmatch(
            r"UCSF_sym_comp_for_Praveen_Smik_[0-9.]+\.dump", path.name
        )
    ]
    paths.sort(key=_bond_length)
    if not paths:
        raise FileNotFoundError(f"No H2O VO dumps found under {DATA_DIR}.")
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
                record["rotated_vo_energy_hartree"],
                record["rotated_minus_csf_hartree"],
                record["physical_ground_state_fidelity"],
                record["maximum_absolute_rotation_angle"],
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
        rotated_vo_energy_hartree=energy_rows[:, 2],
        rotated_minus_csf_hartree=energy_rows[:, 3],
        physical_ground_state_fidelity=energy_rows[:, 4],
        maximum_absolute_rotation_angle=energy_rows[:, 5],
    )
    print(f"Saved comparisons for {len(records)} geometries to {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
