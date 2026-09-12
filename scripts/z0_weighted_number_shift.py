"""Test the exact control variate (N_pair - 5) Z_0 on saved noisy counts."""

from _bootstrap import REPO_ROOT

import argparse
import pickle
from pathlib import Path

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from number_shift_noisy_fragment import _observable_values, _count_arrays, _summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--column", type=int, default=0)
    parser.add_argument("--fragment-index", type=int, default=0)
    parser.add_argument("--exact-pair-number", type=float, default=5.0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=141421)
    args = parser.parse_args()

    uv = (args.row, args.column)
    source_path = args.run_dir / (
        f"number_shift_entry_{uv[0]}_{uv[1]}_fragment_{args.fragment_index}.pkl"
    )
    with source_path.open("rb") as file:
        source = pickle.load(file)

    observable = source["original_observable"]
    number_observable = source["number_observable"]
    measured_qubits = source["measured_qubits"]
    counts = source["counts"]
    largest_index = int(np.argmax(np.abs(observable.coeffs)))
    largest_coefficient = float(np.real(observable.coeffs[largest_index]))
    z0_observable = SparsePauliOp(
        observable.paulis[largest_index : largest_index + 1], [1.0]
    )
    identity = SparsePauliOp.from_list([("I" * observable.num_qubits, 1.0)])
    centered_number = number_observable - args.exact_pair_number * identity
    control_observable = centered_number.compose(z0_observable).simplify()

    observable_values = _observable_values(observable, measured_qubits)
    control_values = _observable_values(control_observable, measured_qubits)
    frequencies, (observed_o, observed_c) = _count_arrays(
        counts, (observable_values, control_values)
    )
    shots = int(frequencies.sum())
    probabilities = frequencies / shots
    raw_estimate = float(probabilities @ observed_o)
    control_estimate = float(probabilities @ observed_c)

    rng = np.random.default_rng(args.seed)
    bootstrap_counts = rng.multinomial(
        shots, probabilities, size=args.bootstrap_repetitions
    )
    bootstrap_probabilities = bootstrap_counts / shots
    raw_samples = bootstrap_probabilities @ observed_o
    control_samples = bootstrap_probabilities @ observed_c
    control_variance = float(np.var(control_samples, ddof=1))
    covariance = float(np.cov(raw_samples, control_samples, ddof=1)[0, 1])
    variance_optimal_lambda = (
        -covariance / control_variance if control_variance > 0 else 0.0
    )

    ideal_fragment = float(source["ideal_fragment_estimate"])
    raw_bias = raw_estimate - ideal_fragment
    bias_optimal_lambda = (
        -raw_bias / control_estimate if control_estimate != 0 else 0.0
    )
    # For seven tapered qubits and N_pair=5,
    # (N-5)Z0 = -(3 Z0 + I + sum_{j != 0} Z0 Zj)/2.
    coefficient_cancelling_lambda = 2 * largest_coefficient / 3
    candidate_lambdas = {
        "raw": 0.0,
        "two_a_over_five": 2 * largest_coefficient / args.exact_pair_number,
        "cancel_z0_coefficient": coefficient_cancelling_lambda,
        "a": largest_coefficient,
        "two_a": 2 * largest_coefficient,
        "variance_optimal": variance_optimal_lambda,
        "diagnostic_bias_optimal": bias_optimal_lambda,
    }

    fixed_entry_contribution = (
        source["classical_contribution"]
        + source["other_noisy_fragment_contribution"]
    )
    ideal_entry = float(source["ideal_entry_estimate"])
    results = {}
    for name, coefficient in candidate_lambdas.items():
        samples = raw_samples + coefficient * control_samples
        results[name] = {
            "lambda": float(coefficient),
            "fragment": _summary(samples, ideal_fragment),
            "full_entry": _summary(
                fixed_entry_contribution + samples, ideal_entry
            ),
        }

    payload = {
        "uv": uv,
        "fragment_index": args.fragment_index,
        "shots": shots,
        "source_counts_file": source_path.name,
        "counts": counts,
        "measured_qubits": measured_qubits,
        "original_observable": observable,
        "number_observable": number_observable,
        "z0_observable": z0_observable,
        "control_observable": control_observable,
        "exact_control_expectation": 0.0,
        "noisy_control_expectation": control_estimate,
        "largest_observable_coefficient": largest_coefficient,
        "ideal_fragment_estimate": ideal_fragment,
        "ideal_entry_estimate": ideal_entry,
        "results": results,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "seed": args.seed,
    }
    stem = f"z0_weighted_number_shift_entry_{uv[0]}_{uv[1]}_fragment_{args.fragment_index}"
    with (args.run_dir / f"{stem}.pkl").open("wb") as file:
        pickle.dump(payload, file)

    lines = [
        "Z0-weighted exact-number control-variate diagnostic",
        f"entry: {uv}",
        f"fragment: {args.fragment_index}",
        f"shots: {shots}",
        f"noisy <(N-5)Z0>: {control_estimate:.12g}",
        f"ideal fragment estimate: {ideal_fragment:.12g}",
    ]
    for name, result in results.items():
        lines.append(
            f"{name}: lambda={result['lambda']:.12g}; "
            f"fragment={result['fragment']}; full_entry={result['full_entry']}"
        )
    (args.run_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Saved {stem}.pkl and {stem}.txt under {args.run_dir}")


if __name__ == "__main__":
    main()
