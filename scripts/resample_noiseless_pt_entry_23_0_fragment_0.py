"""Check finite-shot convergence of PT (23, 0), fragment 0 without noise."""

from _bootstrap import REPO_ROOT

import argparse
import pickle

import numpy as np
from qiskit_aer import AerSimulator

from src.capturing_backend_estimator import CapturingBackendEstimatorV2
from src.saved_run_io import save_json


ENTRY = (23, 0)
FRAGMENT_POSITION = 0
CACHE = REPO_ROOT / (
    "saved/runtime_cache/"
    "sampling_and_sv_h2o_1.0_pt_mp2_qwc_fake_quebec.pkl"
)
OUTPUT = REPO_ROOT / (
    "saved/runtime_runs/pt_h2o_1.0/"
    "pt_readout_linear_zne_20260813T134200Z/"
    "resample_noiseless_entry_23_0_fragment_0.json.gz"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shots", type=int, nargs="+", default=[100, 500, 2000, 10000]
    )
    parser.add_argument("--replicas", type=int, default=32)
    parser.add_argument("--seed", type=int, default=230_013)
    args = parser.parse_args()

    with CACHE.open("rb") as file:
        cache = pickle.load(file)
    circuit, observable, parameter_values, _ = cache["PUBs"][ENTRY][
        FRAGMENT_POSITION
    ]
    ideal_fragment = float(
        np.real(cache["ideal_frag_estimates"][ENTRY][FRAGMENT_POSITION])
    )

    seed_sequences = np.random.SeedSequence(args.seed).spawn(
        len(args.shots) * args.replicas
    )
    seeds = iter(
        int(sequence.generate_state(1, dtype=np.uint32)[0])
        for sequence in seed_sequences
    )
    records = []
    for shots in args.shots:
        print(f"Starting noiseless {shots}-shot replicas.", flush=True)
        for replica in range(args.replicas):
            seed = next(seeds)
            estimator = CapturingBackendEstimatorV2(
                backend=AerSimulator(
                    method="statevector", enable_truncation=True
                ),
                options={"abelian_grouping": True, "seed_simulator": seed},
            )
            result = estimator.run(
                [
                    (
                        circuit,
                        observable,
                        parameter_values,
                        (1.0 + 1e-12) / np.sqrt(shots),
                    )
                ]
            ).result()[0]
            estimate = float(np.asarray(result.data.evs).item())
            # If all other fragments are exact, the entry error equals this
            # fragment's sampling error.
            records.append(
                {
                    "shots": shots,
                    "replica": replica,
                    "seed": seed,
                    "fragment_estimate": estimate,
                    "entry_bias": estimate - ideal_fragment,
                }
            )
        print(f"Completed noiseless {shots}-shot replicas.", flush=True)

    summaries = []
    for shots in args.shots:
        biases = np.asarray(
            [r["entry_bias"] for r in records if r["shots"] == shots]
        )
        summaries.append(
            {
                "shots": shots,
                "mean_bias": float(np.mean(biases)),
                "standard_error_of_mean_bias": float(
                    np.std(biases, ddof=1) / np.sqrt(args.replicas)
                ),
                "single_run_std": float(np.std(biases, ddof=1)),
                "rmse": float(np.sqrt(np.mean(biases**2))),
                "rmse_times_sqrt_shots": float(
                    np.sqrt(np.mean(biases**2)) * np.sqrt(shots)
                ),
            }
        )

    save_json(
        OUTPUT,
        {
            "entry": ENTRY,
            "fragment_position": FRAGMENT_POSITION,
            "ideal_fragment_estimate": ideal_fragment,
            "noise_model": None,
            "other_fragments": "held at ideal estimates",
            "replicas_per_shot_count": args.replicas,
            "summaries": summaries,
            "replicas": records,
        },
    )
    print("\nNoiseless convergence")
    for item in summaries:
        print(
            f"  shots={item['shots']}: mean bias="
            f"{item['mean_bias']:+.6g} +/- "
            f"{item['standard_error_of_mean_bias']:.3g}, "
            f"single-run std={item['single_run_std']:.6g}, "
            f"RMSE={item['rmse']:.6g}"
        )


if __name__ == "__main__":
    main()
