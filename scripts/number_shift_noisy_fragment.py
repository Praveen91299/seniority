"""Apply a fixed-particle-number observable shift to saved noisy counts."""

from _bootstrap import REPO_ROOT

import argparse
import pickle
from pathlib import Path

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from symmetry_shift_noisy_fragment import (
    _count_arrays,
    _observable_values,
    _summary,
)


def _logical_pair_number(num_qubits):
    identity = "I" * num_qubits
    terms = [(identity, num_qubits / 2)]
    for qubit in range(num_qubits):
        label = ["I"] * num_qubits
        # Pauli labels are printed most-significant qubit first.
        label[num_qubits - 1 - qubit] = "Z"
        terms.append(("".join(label), -0.5))
    return SparsePauliOp.from_list(terms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--column", type=int, default=0)
    parser.add_argument("--fragment-index", type=int, default=0)
    parser.add_argument("--exact-pair-number", type=float, default=5.0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=161803)
    parser.add_argument(
        "--output-tag",
        help="Optional suffix used to preserve results from multiple lambda choices.",
    )
    parser.add_argument(
        "--shift-coefficient",
        type=float,
        help=(
            "lambda in O_shifted=O+lambda*N; defaults to twice the largest "
            "coefficient divided by the exact pair number."
        ),
    )
    args = parser.parse_args()

    uv = (args.row, args.column)
    parity_path = args.run_dir / (
        f"symmetry_shift_entry_{uv[0]}_{uv[1]}_fragment_{args.fragment_index}.pkl"
    )
    with parity_path.open("rb") as file:
        saved_counts = pickle.load(file)
    with (args.run_dir / "experiment_inputs.pkl").open("rb") as file:
        experiment = pickle.load(file)
    with (args.run_dir / "fragment_artifacts.pkl").open("rb") as file:
        fragments = pickle.load(file)

    fragment = next(
        item
        for item in fragments
        if tuple(item["uv"]) == uv
        and int(item["fragment_index"]) == args.fragment_index
    )
    observable = fragment["observable"]
    circuit = fragment["circuit"]
    input_qubits = int(circuit.layout._input_qubit_count)
    logical_number = _logical_pair_number(input_qubits)
    number_observable = logical_number.apply_layout(circuit.layout)

    largest_index = int(np.argmax(np.abs(observable.coeffs)))
    largest_coefficient = float(np.real(observable.coeffs[largest_index]))
    shift_coefficient = (
        args.shift_coefficient
        if args.shift_coefficient is not None
        else 2 * largest_coefficient / args.exact_pair_number
    )
    shifted_observable = observable + shift_coefficient * number_observable

    measured_qubits = saved_counts["measured_qubits"]
    observable_values = _observable_values(observable, measured_qubits)
    number_values = _observable_values(number_observable, measured_qubits)
    shifted_values = observable_values + shift_coefficient * number_values
    frequencies, (observed_o, observed_n, observed_shifted) = _count_arrays(
        saved_counts["counts"],
        (observable_values, number_values, shifted_values),
    )
    shots = int(frequencies.sum())
    probabilities = frequencies / shots

    raw_estimate = float(probabilities @ observed_o)
    number_estimate = float(probabilities @ observed_n)
    shifted_raw_estimate = float(probabilities @ observed_shifted)
    recovered_estimate = shifted_raw_estimate - (
        shift_coefficient * args.exact_pair_number
    )

    rng = np.random.default_rng(args.seed)
    bootstrap_counts = rng.multinomial(
        shots, probabilities, size=args.bootstrap_repetitions
    )
    bootstrap_probabilities = bootstrap_counts / shots
    raw_samples = bootstrap_probabilities @ observed_o
    number_samples = bootstrap_probabilities @ observed_n
    recovered_samples = (
        bootstrap_probabilities @ observed_shifted
        - shift_coefficient * args.exact_pair_number
    )

    ideal_fragment = float(saved_counts["ideal_fragment_estimate"])
    number_variance = float(np.var(number_samples, ddof=1))
    covariance = float(np.cov(raw_samples, number_samples, ddof=1)[0, 1])
    variance_optimal_lambda = (
        -covariance / number_variance if number_variance > 0 else 0.0
    )
    variance_optimal_samples = raw_samples + variance_optimal_lambda * (
        number_samples - args.exact_pair_number
    )
    number_displacement = number_estimate - args.exact_pair_number
    bias_optimal_lambda = (
        -(raw_estimate - ideal_fragment) / number_displacement
        if number_displacement != 0
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
        "source_counts_file": parity_path.name,
        "counts": saved_counts["counts"],
        "measured_qubits": measured_qubits,
        "original_observable": observable,
        "number_observable": number_observable,
        "shifted_observable": shifted_observable,
        "exact_pair_number": args.exact_pair_number,
        "noisy_pair_number": number_estimate,
        "shift_coefficient": shift_coefficient,
        "largest_observable_coefficient": largest_coefficient,
        "ideal_fragment_estimate": ideal_fragment,
        "raw_estimate": raw_estimate,
        "shifted_raw_estimate": shifted_raw_estimate,
        "recovered_estimate": recovered_estimate,
        "raw_bootstrap": _summary(raw_samples, ideal_fragment),
        "shifted_bootstrap": _summary(recovered_samples, ideal_fragment),
        "variance_optimal_lambda": variance_optimal_lambda,
        "variance_optimal_bootstrap": _summary(
            variance_optimal_samples, ideal_fragment
        ),
        "diagnostic_bias_optimal_lambda": bias_optimal_lambda,
        "ideal_entry_estimate": ideal_entry,
        "classical_contribution": classical_contribution,
        "other_noisy_fragment_contribution": other_noisy_fragments,
        "full_raw_bootstrap": _summary(full_raw_samples, ideal_entry),
        "full_shifted_bootstrap": _summary(full_shifted_samples, ideal_entry),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "seed": args.seed,
    }
    stem = f"number_shift_entry_{uv[0]}_{uv[1]}_fragment_{args.fragment_index}"
    if args.output_tag:
        stem += f"_{args.output_tag}"
    with (args.run_dir / f"{stem}.pkl").open("wb") as file:
        pickle.dump(payload, file)

    lines = [
        "Exact-pair-number observable-shift diagnostic",
        f"entry: {uv}",
        f"fragment: {args.fragment_index}",
        f"shots: {shots}",
        f"ideal pair number: {args.exact_pair_number:.12g}",
        f"noisy pair number: {number_estimate:.12g}",
        f"ideal fragment estimate: {ideal_fragment:.12g}",
        f"raw noisy estimate: {raw_estimate:.12g}",
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
