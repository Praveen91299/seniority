"""Test an exact-parity control-variate shift on one saved noisy fragment."""

from _bootstrap import REPO_ROOT

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
from qiskit import ClassicalRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeQuebec


def _observable_support(*observables):
    support = np.zeros(observables[0].num_qubits, dtype=bool)
    for observable in observables:
        if np.any(observable.paulis.x):
            raise ValueError("This diagnostic expects diagonal I/Z observables.")
        support |= np.any(observable.paulis.z, axis=0)
    return np.flatnonzero(support).tolist()


def _observable_values(observable, measured_qubits):
    z_support = np.asarray(observable.paulis.z[:, measured_qubits], dtype=np.int8)
    coefficients = np.real_if_close(np.asarray(observable.coeffs)).astype(float)
    values = np.empty(2 ** len(measured_qubits))
    for packed_bits in range(values.size):
        bits = np.array(
            [(packed_bits >> index) & 1 for index in range(len(measured_qubits))],
            dtype=np.int8,
        )
        signs = 1 - 2 * ((z_support @ bits) % 2)
        values[packed_bits] = coefficients @ signs
    return values


def _count_arrays(counts, value_arrays):
    frequencies = []
    selected_values = [[] for _ in value_arrays]
    for bitstring, frequency in counts.items():
        # The diagnostic register is added last and printed left-most by Qiskit.
        packed_bits = int(bitstring.split()[0], 2)
        frequencies.append(int(frequency))
        for output, values in zip(selected_values, value_arrays):
            output.append(values[packed_bits])
    return np.asarray(frequencies), [np.asarray(values) for values in selected_values]


def _summary(samples, ideal):
    return {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples, ddof=1)),
        "bias": float(np.mean(samples) - ideal),
        "rmse": float(np.sqrt(np.mean((samples - ideal) ** 2))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--column", type=int, default=0)
    parser.add_argument("--fragment-index", type=int, default=0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument(
        "--shift-coefficient",
        type=float,
        help="lambda in O_shifted = O + lambda*S; defaults to minus the largest coefficient.",
    )
    args = parser.parse_args()

    uv = (args.row, args.column)
    with (args.run_dir / "fragment_artifacts.pkl").open("rb") as file:
        fragments = pickle.load(file)
    with (args.run_dir / "experiment_inputs.pkl").open("rb") as file:
        experiment = pickle.load(file)
    fragment = next(
        item
        for item in fragments
        if tuple(item["uv"]) == uv
        and int(item["fragment_index"]) == args.fragment_index
    )
    with (args.run_dir / f"bootstrap_noiseless_entry_{uv[0]}_{uv[1]}.pkl").open(
        "rb"
    ) as file:
        noiseless = pickle.load(file)
    ideal_fragment = next(
        item["exact_mean"]
        for item in noiseless["fragments"]
        if int(item["fragment_index"]) == args.fragment_index
    )

    circuit = fragment["circuit"]
    observable = fragment["observable"]
    input_qubits = int(circuit.layout._input_qubit_count)
    logical_symmetry = SparsePauliOp.from_list([("Z" * input_qubits, 1.0)])
    symmetry = logical_symmetry.apply_layout(circuit.layout)
    symmetry_eigenvalue = -1.0

    largest_index = int(np.argmax(np.abs(observable.coeffs)))
    largest_coefficient = float(np.real(observable.coeffs[largest_index]))
    shift_coefficient = (
        args.shift_coefficient
        if args.shift_coefficient is not None
        else -largest_coefficient
    )
    shifted_observable = observable + shift_coefficient * symmetry

    measured_qubits = _observable_support(observable, symmetry)
    observable_values = _observable_values(observable, measured_qubits)
    symmetry_values = _observable_values(symmetry, measured_qubits)
    shifted_values = observable_values + shift_coefficient * symmetry_values

    measured_circuit = circuit.copy()
    register = ClassicalRegister(len(measured_qubits), "symmetry_shift")
    measured_circuit.add_register(register)
    measured_circuit.measure(measured_qubits, register)
    shots = int(fragment["allocated_shots"])
    simulator = AerSimulator.from_backend(
        FakeQuebec(), method="statevector", enable_truncation=True
    )
    print(
        f"Running noisy fragment {args.fragment_index} for entry {uv}: "
        f"shots={shots}, measured_qubits={len(measured_qubits)}, "
        f"lambda={shift_coefficient:.12g}",
        flush=True,
    )
    started = time.perf_counter()
    result = simulator.run(
        measured_circuit,
        shots=shots,
        seed_simulator=args.seed,
    ).result()
    elapsed = time.perf_counter() - started
    counts = result.get_counts(0)
    frequencies, (observed_o, observed_s, observed_shifted) = _count_arrays(
        counts, (observable_values, symmetry_values, shifted_values)
    )
    probabilities = frequencies / frequencies.sum()

    raw_estimate = float(probabilities @ observed_o)
    symmetry_estimate = float(probabilities @ observed_s)
    shifted_raw_estimate = float(probabilities @ observed_shifted)
    recovered_estimate = shifted_raw_estimate - (
        shift_coefficient * symmetry_eigenvalue
    )

    rng = np.random.default_rng(args.seed)
    bootstrap_counts = rng.multinomial(
        shots, probabilities, size=args.bootstrap_repetitions
    )
    bootstrap_probabilities = bootstrap_counts / shots
    raw_samples = bootstrap_probabilities @ observed_o
    symmetry_samples = bootstrap_probabilities @ observed_s
    recovered_samples = (
        bootstrap_probabilities @ observed_shifted
        - shift_coefficient * symmetry_eigenvalue
    )

    symmetry_variance = float(np.var(symmetry_samples, ddof=1))
    covariance = float(np.cov(raw_samples, symmetry_samples, ddof=1)[0, 1])
    variance_optimal_lambda = (
        -covariance / symmetry_variance if symmetry_variance > 0 else 0.0
    )
    variance_optimal_samples = raw_samples + variance_optimal_lambda * (
        symmetry_samples - symmetry_eigenvalue
    )
    symmetry_displacement = symmetry_estimate - symmetry_eigenvalue
    bias_optimal_lambda = (
        -(raw_estimate - ideal_fragment) / symmetry_displacement
        if symmetry_displacement != 0
        else 0.0
    )
    ideal_entry = float(np.real(experiment["ideal_matrix"][uv]))
    classical_contribution = float(np.real(experiment["classical_matrix"][uv]))
    other_noisy_fragments = float(
        sum(
            item["noisy_fragment_estimate"]
            for item in fragments
            if tuple(item["uv"]) == uv
            and int(item["fragment_index"]) != args.fragment_index
        )
    )
    fixed_entry_contribution = classical_contribution + other_noisy_fragments
    full_raw_samples = fixed_entry_contribution + raw_samples
    full_shifted_samples = fixed_entry_contribution + recovered_samples

    payload = {
        "uv": uv,
        "fragment_index": args.fragment_index,
        "shots": shots,
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "counts": counts,
        "measured_qubits": measured_qubits,
        "original_observable": observable,
        "symmetry_observable": symmetry,
        "shifted_observable": shifted_observable,
        "ideal_fragment_estimate": float(ideal_fragment),
        "largest_observable_coefficient": largest_coefficient,
        "symmetry_eigenvalue": symmetry_eigenvalue,
        "shift_coefficient": float(shift_coefficient),
        "raw_estimate": raw_estimate,
        "symmetry_estimate": symmetry_estimate,
        "shifted_raw_estimate": shifted_raw_estimate,
        "recovered_estimate": recovered_estimate,
        "raw_bootstrap": _summary(raw_samples, ideal_fragment),
        "shifted_bootstrap": _summary(recovered_samples, ideal_fragment),
        "variance_optimal_lambda": variance_optimal_lambda,
        "variance_optimal_bootstrap": _summary(
            variance_optimal_samples, ideal_fragment
        ),
        "diagnostic_bias_optimal_lambda": bias_optimal_lambda,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "ideal_entry_estimate": ideal_entry,
        "classical_contribution": classical_contribution,
        "other_noisy_fragment_contribution": other_noisy_fragments,
        "full_raw_bootstrap": _summary(full_raw_samples, ideal_entry),
        "full_shifted_bootstrap": _summary(full_shifted_samples, ideal_entry),
    }
    stem = f"symmetry_shift_entry_{uv[0]}_{uv[1]}_fragment_{args.fragment_index}"
    with (args.run_dir / f"{stem}.pkl").open("wb") as file:
        pickle.dump(payload, file)

    lines = [
        "Exact-parity symmetry-shift diagnostic",
        f"entry: {uv}",
        f"fragment: {args.fragment_index}",
        f"shots: {shots}",
        f"ideal fragment estimate: {ideal_fragment:.12g}",
        f"raw noisy estimate: {raw_estimate:.12g}",
        f"sampled symmetry <S>: {symmetry_estimate:.12g}",
        f"symmetry eigenvalue: {symmetry_eigenvalue:.12g}",
        f"lambda: {shift_coefficient:.12g}",
        f"shifted raw estimate: {shifted_raw_estimate:.12g}",
        f"recovered estimate: {recovered_estimate:.12g}",
        f"raw bootstrap: {payload['raw_bootstrap']}",
        f"shifted bootstrap: {payload['shifted_bootstrap']}",
        f"variance-optimal lambda: {variance_optimal_lambda:.12g}",
        f"variance-optimal bootstrap: {payload['variance_optimal_bootstrap']}",
        f"diagnostic bias-optimal lambda: {bias_optimal_lambda:.12g}",
        f"full entry raw bootstrap: {payload['full_raw_bootstrap']}",
        f"full entry shifted bootstrap: {payload['full_shifted_bootstrap']}",
    ]
    (args.run_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Saved {stem}.pkl and {stem}.txt under {args.run_dir}")


if __name__ == "__main__":
    main()
