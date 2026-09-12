"""Split one Q-SENSE basis state into fixed-orbital sectors and refit each circuit.

The output is a self-contained pickle dictionary.  The two records under
``sectors`` contain optimized six-qubit active circuits, seven-qubit circuits
with orbital 0 fixed, angles, targets, overlaps, and reconstruction weights.
"""

from _bootstrap import REPO_ROOT

import argparse
import pickle
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from src.circuits.circuits_csf import get_csfs_from_dump
from src.circuits.circuits_pair_excitation import append_tapered_exc_rot


def _generator_support(rotation):
    return sorted({orbital for pair in rotation.get_excitations() for orbital in pair})


def _restricted_parameterized_circuit(reference, rotations, fixed_orbital):
    """Return an active-only parameterized circuit and its parameter vector."""
    active_orbitals = [
        orbital for orbital in range(len(reference)) if orbital != fixed_orbital
    ]
    active_position = {
        orbital: position for position, orbital in enumerate(active_orbitals)
    }
    parameters = ParameterVector("theta", len(rotations))
    circuit = QuantumCircuit(len(active_orbitals))

    for orbital in active_orbitals:
        if reference[orbital]:
            circuit.x(active_position[orbital])

    for rotation, parameter in zip(rotations, parameters):
        indices = rotation.get_indices()
        if len(indices) != 2:
            raise NotImplementedError(
                "This minimal script currently supports the simple paired "
                "rotations used by state 0."
            )
        i, a = indices
        append_tapered_exc_rot(
            circuit,
            active_position[i],
            active_position[a],
            parameter,
        )
    return circuit, parameters, active_orbitals


def _bind(circuit, parameters, angles):
    return circuit.assign_parameters(dict(zip(parameters, angles)))


def _circuit_metrics(circuit):
    counts = {str(name): int(count) for name, count in circuit.count_ops().items()}
    return {
        "num_qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "size": circuit.size(),
        "count_ops": counts,
        "two_qubit_gates": sum(
            counts.get(name, 0) for name in ("cx", "cz", "ecr")
        ),
    }


def _embed_active_circuit(active_circuit, num_orbitals, fixed_orbital, fixed_value):
    full = QuantumCircuit(num_orbitals)
    if fixed_value:
        full.x(fixed_orbital)
    active_orbitals = [
        orbital for orbital in range(num_orbitals) if orbital != fixed_orbital
    ]
    full.compose(active_circuit, qubits=active_orbitals, inplace=True)
    return full


def _sector_target(full_target, num_orbitals, fixed_orbital, fixed_value):
    indices = np.arange(2**num_orbitals)
    mask = ((indices >> fixed_orbital) & 1) == fixed_value
    projected = np.where(mask, full_target, 0.0)
    probability = float(np.vdot(projected, projected).real)
    if probability == 0:
        raise ValueError(f"The target has no support in sector {fixed_value}.")

    normalized_full = projected / np.sqrt(probability)
    active_orbitals = [
        orbital for orbital in range(num_orbitals) if orbital != fixed_orbital
    ]
    normalized_active = np.empty(2 ** len(active_orbitals), dtype=complex)
    for active_index in range(normalized_active.size):
        full_index = 0
        for active_position, orbital in enumerate(active_orbitals):
            full_index |= ((active_index >> active_position) & 1) << orbital
        full_index |= fixed_value << fixed_orbital
        normalized_active[active_index] = normalized_full[full_index]
    return probability, normalized_full, normalized_active


def _reference_from_largest_amplitude(
    normalized_full_target,
    num_orbitals,
):
    index = int(np.argmax(np.abs(normalized_full_target) ** 2))
    return [(index >> orbital) & 1 for orbital in range(num_orbitals)]


def _optimize_sector(
    full_target,
    rotations,
    warm_angles,
    num_orbitals,
    fixed_orbital,
    fixed_value,
    maxiter,
):
    probability, target_full, target_active = _sector_target(
        full_target,
        num_orbitals,
        fixed_orbital,
        fixed_value,
    )
    reference = _reference_from_largest_amplitude(target_full, num_orbitals)
    parameterized, parameters, active_orbitals = _restricted_parameterized_circuit(
        reference,
        rotations,
        fixed_orbital,
    )

    evaluations = 0

    def objective(angles):
        nonlocal evaluations
        evaluations += 1
        state = np.asarray(
            Statevector.from_instruction(
                _bind(parameterized, parameters, angles)
            ).data
        )
        return 1.0 - abs(np.vdot(target_active, state)) ** 2

    initial_infidelity = float(objective(warm_angles))
    result = minimize(
        objective,
        warm_angles,
        method="L-BFGS-B",
        options={
            "ftol": 1e-13,
            "gtol": 1e-9,
            "maxiter": maxiter,
            "maxfun": 30_000,
        },
    )

    optimized_active = _bind(parameterized, parameters, result.x)
    prepared_active = np.asarray(
        Statevector.from_instruction(optimized_active).data
    )

    # Fidelity optimization leaves a global phase undetermined.  Align it so
    # the two sector circuits can reconstruct the original state coherently.
    overlap_before_phase = np.vdot(target_active, prepared_active)
    phase_correction = -float(np.angle(overlap_before_phase))
    optimized_active.global_phase += phase_correction
    prepared_active = np.asarray(
        Statevector.from_instruction(optimized_active).data
    )
    overlap = np.vdot(target_active, prepared_active)
    optimized_full = _embed_active_circuit(
        optimized_active,
        num_orbitals,
        fixed_orbital,
        fixed_value,
    )
    prepared_full = np.asarray(
        Statevector.from_instruction(optimized_full).data
    )

    return {
        "fixed_value": fixed_value,
        "sector_probability": probability,
        "reconstruction_weight": np.sqrt(probability),
        "reference_occupation": reference,
        "active_orbitals": active_orbitals,
        "warm_start_angles": np.asarray(warm_angles),
        "optimized_angles": np.asarray(result.x),
        "target_full_statevector": target_full,
        "target_active_statevector": target_active,
        "prepared_full_statevector": prepared_full,
        "prepared_active_statevector": prepared_active,
        "active_circuit": optimized_active,
        "fixed_sector_circuit": optimized_full,
        "active_circuit_metrics": _circuit_metrics(optimized_active),
        "fixed_sector_circuit_metrics": _circuit_metrics(optimized_full),
        "initial_infidelity": initial_infidelity,
        "optimized_infidelity": float(1.0 - abs(overlap) ** 2),
        "overlap": overlap,
        "phase_correction": phase_correction,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(result.nit),
        "objective_evaluations": evaluations,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dump",
        type=Path,
        default=Path(
            "data/h2o_data/UCSF_sym_comp_for_Praveen_Smik_1.0.dump"
        ),
    )
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--fixed-orbital", type=int, default=0)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with args.input_dump.open("rb") as file:
        source_dump_payload = pickle.load(file)
    csfs = get_csfs_from_dump(
        args.input_dump,
        verify_states=True,
        verbose=False,
    )
    source_csf = csfs[args.state_index]
    source_circuit = source_csf.get_tapered_full_circuit()
    target = np.asarray(Statevector.from_instruction(source_circuit).data)
    num_orbitals = source_circuit.num_qubits

    removed = []
    retained = []
    for index, rotation in enumerate(source_csf.get_excitations()):
        record = {
            "original_index": index,
            "excitations": rotation.get_excitations(),
            "support": _generator_support(rotation),
            "angle": float(rotation.get_theta()),
        }
        if args.fixed_orbital in record["support"]:
            removed.append(record)
        else:
            retained.append((rotation, record))

    rotations = [rotation for rotation, _ in retained]
    retained_records = [record for _, record in retained]
    warm_angles = np.array(
        [rotation.get_theta() for rotation in rotations],
        dtype=float,
    )

    sectors = {}
    for fixed_value in (0, 1):
        print(
            f"Optimizing state {args.state_index}, orbital "
            f"{args.fixed_orbital}={fixed_value}...",
            flush=True,
        )
        sectors[fixed_value] = _optimize_sector(
            target,
            rotations,
            warm_angles,
            num_orbitals,
            args.fixed_orbital,
            fixed_value,
            args.maxiter,
        )
        sector = sectors[fixed_value]
        print(
            f"  probability={sector['sector_probability']:.12g}, "
            f"initial infidelity={sector['initial_infidelity']:.3e}, "
            f"optimized infidelity={sector['optimized_infidelity']:.3e}",
            flush=True,
        )

    reconstructed = sum(
        sector["reconstruction_weight"] * sector["prepared_full_statevector"]
        for sector in sectors.values()
    )
    reconstruction_overlap = np.vdot(target, reconstructed)
    reconstruction_fidelity = float(abs(reconstruction_overlap) ** 2)
    reconstruction_error = float(np.linalg.norm(reconstructed - target))

    payload = {
        "format": "qsense_fixed_orbital_split_v1",
        "source_dump_file": str(args.input_dump),
        "source_dump_payload": source_dump_payload,
        "state_index": args.state_index,
        "fixed_orbital": args.fixed_orbital,
        "source_circuit": source_circuit,
        "source_circuit_metrics": _circuit_metrics(source_circuit),
        "source_statevector": target,
        "removed_generators": removed,
        "retained_generators": retained_records,
        "sectors": sectors,
        "reconstructed_statevector": reconstructed,
        "reconstruction_overlap": reconstruction_overlap,
        "reconstruction_fidelity": reconstruction_fidelity,
        "reconstruction_l2_error": reconstruction_error,
    }

    output = args.output or args.input_dump.with_name(
        f"{args.input_dump.stem}_state{args.state_index}"
        f"_orbital{args.fixed_orbital}_split.dump"
    )
    with output.open("wb") as file:
        pickle.dump(payload, file)

    summary = output.with_suffix(".txt")
    lines = [
        "Q-SENSE fixed-orbital state split",
        f"source: {args.input_dump}",
        f"state_index: {args.state_index}",
        f"fixed_orbital: {args.fixed_orbital}",
        f"source_rotations: {len(source_csf.get_excitations())}",
        f"retained_rotations: {len(retained_records)}",
        f"removed_rotations: {len(removed)}",
    ]
    for fixed_value, sector in sectors.items():
        lines.extend(
            [
                "",
                f"sector {fixed_value}:",
                f"  probability: {sector['sector_probability']:.16g}",
                f"  reconstruction_weight: "
                f"{sector['reconstruction_weight']:.16g}",
                f"  reference_occupation: "
                f"{sector['reference_occupation']}",
                f"  initial_infidelity: "
                f"{sector['initial_infidelity']:.6e}",
                f"  optimized_infidelity: "
                f"{sector['optimized_infidelity']:.6e}",
                f"  optimizer_success: {sector['optimizer_success']}",
                f"  optimizer_iterations: "
                f"{sector['optimizer_iterations']}",
                f"  objective_evaluations: "
                f"{sector['objective_evaluations']}",
                f"  active_circuit_metrics: "
                f"{sector['active_circuit_metrics']}",
            ]
        )
    lines.extend(
        [
            "",
            f"reconstruction_fidelity: {reconstruction_fidelity:.16g}",
            f"reconstruction_l2_error: {reconstruction_error:.6e}",
            f"pickle_output: {output}",
        ]
    )
    summary.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-4:]))


if __name__ == "__main__":
    main()
