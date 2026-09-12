"""Probabilistic local-folding ZNE for a selected split-basis diagonal entry."""

from __future__ import annotations

from _bootstrap import REPO_ROOT

import argparse
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qiskit import qpy
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeQuebec

from src.capturing_backend_estimator import (
    CapturingBackendEstimatorV2,
    RAW_SIMULATOR_COUNTS,
)
from src.mitigation import (
    linear_zne,
    probabilistic_local_fold_circuit,
    run_local_readout_calibration,
    variance_weighted_linear_zne,
)
from src.saved_run_io import load_json, save_json


DEFAULT_RUN_DIR = Path(
    "saved/runtime_runs/"
    "20260724T024719Z_h2o_1.0_"
    "state0_orbital0_split_v1_qwc_fake_quebec"
)
DEFAULT_CACHE = Path(
    "saved/runtime_cache/"
    "sampling_and_sv_h2o_1.0_"
    "state0_orbital0_split_v1_qwc_fake_quebec.pkl"
)
DEFAULT_PILOT_RESULTS = DEFAULT_RUN_DIR / (
    "probabilistic_zne_entry_1_1_20260725T200028Z/results.json.gz"
)


def _support_qubits(pubs):
    support = set()
    for _, observable, _, _ in pubs:
        active = np.any(
            np.logical_or(observable.paulis.z, observable.paulis.x),
            axis=0,
        )
        support.update(map(int, np.flatnonzero(active)))
    return sorted(support)


def _split_shots(total, parts):
    quotient, remainder = divmod(int(total), int(parts))
    return [
        quotient + (index < remainder)
        for index in range(parts)
    ]


def _integer_allocation(total, weights):
    """Allocate an integer total by largest remainder."""
    weights = np.asarray(weights, dtype=float)
    if np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("Allocation weights must be nonnegative and nonzero.")
    exact = int(total) * weights / np.sum(weights)
    allocation = np.floor(exact).astype(int)
    remainder = int(total) - int(np.sum(allocation))
    order = np.argsort(-(exact - allocation), kind="stable")
    allocation[order[:remainder]] += 1
    return allocation


def _noisy_gate_count(circuit):
    counts = circuit.count_ops()
    return int(sum(counts.get(name, 0) for name in ("sx", "x", "ecr")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--entry", type=int, default=1)
    parser.add_argument(
        "--noise-factors",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0],
    )
    parser.add_argument("--randomizations", type=int, default=8)
    parser.add_argument("--fold-seed", type=int, default=104_729)
    parser.add_argument("--calibration-shots", type=int, default=20_000)
    parser.add_argument("--calibration-seed", type=int, default=8_675_309)
    parser.add_argument("--simulator-seed", type=int, default=16_180)
    parser.add_argument(
        "--pilot-results",
        type=Path,
        default=None,
        help=(
            "Previous results.json.gz used to allocate a fixed total ZNE "
            "budget in proportion to |c_i| times the per-shot standard "
            "deviation. If omitted, every factor receives the cached budget."
        ),
    )
    parser.add_argument(
        "--total-budget-multiplier",
        type=float,
        default=None,
        help=(
            "Total ZNE shots as a multiple of the cached single-factor "
            "budget. Defaults to the number of noise factors when pilot "
            "allocation is enabled."
        ),
    )
    parser.add_argument(
        "--per-factor-budget-multiplier",
        type=float,
        default=1.0,
        help=(
            "Scale the cached allocation assigned to every factor when no "
            "pilot allocation is requested."
        ),
    )
    args = parser.parse_args()

    entry = (args.entry, args.entry)
    factors = tuple(map(float, args.noise_factors))
    if factors[0] != 1.0 or any(
        right <= left for left, right in zip(factors, factors[1:])
    ):
        raise ValueError("Noise factors must increase strictly from 1.")
    if any(factor > 3 for factor in factors):
        raise ValueError("Probabilistic factors must not exceed 3.")

    # Trusted local compilation cache; all newly generated results are portable.
    with args.cache.open("rb") as file:
        cache = pickle.load(file)
    source_pubs = cache["PUBs"][entry]
    source_metadata = sorted(
        (
            item
            for item in cache["pub_metadata"]
            if tuple(item["uv"]) == entry
        ),
        key=lambda item: item["fragment_index"],
    )
    if len(source_pubs) != len(source_metadata):
        raise ValueError(f"PUB/metadata mismatch for entry {entry}.")

    cached_fragment_allocation = np.asarray(
        cache["shot_alloc"][entry],
        dtype=int,
    )
    cached_factor_budget = int(np.sum(cached_fragment_allocation))
    pilot_allocation = None
    if args.pilot_results is not None:
        pilot = load_json(args.pilot_results)
        pilot_factors = np.asarray(pilot["noise_factors"], dtype=float)
        if not np.array_equal(pilot_factors, np.asarray(factors)):
            raise ValueError(
                "Pilot and requested noise factors differ: "
                f"{pilot_factors.tolist()} versus {list(factors)}."
            )
        pilot_stds = np.asarray(
            [item["std"] for item in pilot["factor_summaries"]],
            dtype=float,
        )
        pilot_coefficients = np.asarray(
            pilot["linear_coefficients"],
            dtype=float,
        )
        allocation_weights = np.abs(pilot_coefficients) * pilot_stds
        multiplier = (
            len(factors)
            if args.total_budget_multiplier is None
            else args.total_budget_multiplier
        )
        total_budget = int(round(multiplier * cached_factor_budget))
        factor_budgets = _integer_allocation(total_budget, allocation_weights)
        pilot_allocation = {
            "pilot_results": args.pilot_results,
            "pilot_stds": pilot_stds,
            "pilot_linear_coefficients": pilot_coefficients,
            "allocation_weights": allocation_weights,
            "total_budget": total_budget,
            "factor_budgets": factor_budgets,
        }
    else:
        factor_budgets = np.full(
            len(factors),
            int(round(args.per_factor_budget_multiplier * cached_factor_budget)),
            dtype=int,
        )

    fragment_allocations_by_factor = [
        _integer_allocation(budget, cached_fragment_allocation)
        for budget in factor_budgets
    ]
    print(
        "Shot budgets by factor: "
        + ", ".join(
            f"lambda={factor:g}: {budget}"
            for factor, budget in zip(factors, factor_budgets)
        )
        + f" (total={int(np.sum(factor_budgets))})",
        flush=True,
    )

    backend = FakeQuebec()
    folded_pubs = []
    manifest = []
    original_noisy_counts = [
        _noisy_gate_count(pub[0]) for pub in source_pubs
    ]
    for factor_index, factor in enumerate(factors):
        repetitions = 1 if factor == 1 else args.randomizations
        shots_by_fragment = [
            _split_shots(
                fragment_allocations_by_factor[factor_index][fragment],
                repetitions,
            )
            for fragment in range(len(source_pubs))
        ]
        for randomization in range(repetitions):
            for fragment, (source_pub, source_item) in enumerate(
                zip(source_pubs, source_metadata)
            ):
                circuit, observable, parameter_values, _ = source_pub
                shots = shots_by_fragment[fragment][randomization]
                if shots <= 0:
                    raise ValueError(
                        "The allocation is too small for the requested "
                        "number of randomizations."
                    )
                fold_seed = (
                    args.fold_seed
                    + 100_000 * factor_index
                    + 1_000 * randomization
                    + fragment
                )
                folded = probabilistic_local_fold_circuit(
                    circuit,
                    factor,
                    seed=fold_seed,
                    backend=backend,
                )
                pub_index = len(folded_pubs)
                folded.metadata = {
                    **(folded.metadata or {}),
                    "qsense_pub_index": pub_index,
                    "qsense_uv": entry,
                    "qsense_fragment_index": fragment,
                }
                folded_pubs.append(
                    (
                        folded,
                        observable,
                        parameter_values,
                        (1.0 + 1e-12) / np.sqrt(shots),
                    )
                )
                manifest.append(
                    {
                        "pub_index": pub_index,
                        "entry": entry,
                        "noise_factor": factor,
                        "randomization": randomization,
                        "num_randomizations": repetitions,
                        "fragment_index": fragment,
                        "shots": shots,
                        "fold_seed": fold_seed,
                        "folded_gate_count": folded.metadata.get(
                            "zne_folded_gate_count", 0
                        ),
                        "eligible_gate_count": folded.metadata.get(
                            "zne_eligible_gate_count", 0
                        ),
                        "realized_noisy_gate_factor": (
                            _noisy_gate_count(folded)
                            / original_noisy_counts[fragment]
                        ),
                        "original_depth": circuit.depth(),
                        "folded_depth": folded.depth(),
                    }
                )

    for factor in factors:
        factor_records = [
            item for item in manifest if item["noise_factor"] == factor
        ]
        total_by_fragment = [
            sum(
                item["shots"]
                for item in factor_records
                if item["fragment_index"] == fragment
            )
            for fragment in range(len(source_pubs))
        ]
        factor_index = factors.index(factor)
        expected = list(
            map(int, fragment_allocations_by_factor[factor_index])
        )
        if total_by_fragment != expected:
            raise ValueError(
                f"Shot allocation changed at factor {factor}: "
                f"{total_by_fragment} versus {expected}."
            )
        realized = [
            item["realized_noisy_gate_factor"] for item in factor_records
        ]
        print(
            f"factor={factor:g}: circuits={len(factor_records)}, "
            f"shots={sum(total_by_fragment)}, realized noisy-gate factor "
            f"{min(realized):.4f}-{max(realized):.4f}",
            flush=True,
        )

    support = _support_qubits(folded_pubs)
    print(f"Physical readout support: {support}", flush=True)
    mitigator, calibration = run_local_readout_calibration(
        backend,
        support,
        shots=args.calibration_shots,
        seed_simulator=args.calibration_seed,
        simulator_method="statevector",
    )
    print("Readout calibration complete.", flush=True)

    simulator = AerSimulator.from_backend(
        backend,
        method="statevector",
        enable_truncation=True,
    )
    estimator = CapturingBackendEstimatorV2(
        backend=simulator,
        readout_mitigator=mitigator,
        options={
            "abelian_grouping": True,
            "seed_simulator": args.simulator_seed,
        },
    )
    RAW_SIMULATOR_COUNTS.reset()
    primitive_result = estimator.run(folded_pubs).result()
    print("Probabilistically folded simulations complete.", flush=True)

    histograms_by_pub = {}
    for histogram in RAW_SIMULATOR_COUNTS.snapshot():
        histograms_by_pub.setdefault(
            histogram["pub_index"], []
        ).append(histogram)
    fragment_results = []
    estimates = {}
    variances = {}
    for item, pub_result in zip(manifest, primitive_result):
        key = (
            item["noise_factor"],
            item["randomization"],
            item["fragment_index"],
        )
        estimate = np.asarray(pub_result.data.evs).item()
        std = float(np.asarray(pub_result.data.stds).item())
        estimates[key] = estimate
        variances[key] = std**2
        histograms = histograms_by_pub.get(item["pub_index"], [])
        if not histograms:
            raise ValueError(f"No histograms captured for PUB {item['pub_index']}.")
        if any(
            sum(histogram["counts"].values()) != item["shots"]
            for histogram in histograms
        ):
            raise ValueError(f"Histogram shot mismatch for PUB {item['pub_index']}.")
        fragment_results.append(
            {
                **item,
                "mitigated_fragment_estimate": estimate,
                "mitigated_fragment_std": std,
                "raw_measurement_histograms": histograms,
            }
        )

    classical = complex(cache["H_classical"][entry])
    factor_summaries = []
    factor_estimates = []
    factor_variances = []
    for factor in factors:
        repetitions = 1 if factor == 1 else args.randomizations
        replica_estimates = []
        replica_variances = []
        for randomization in range(repetitions):
            replica_estimates.append(
                classical
                + sum(
                    estimates[(factor, randomization, fragment)]
                    for fragment in range(len(source_pubs))
                )
            )
            replica_variances.append(
                sum(
                    variances[(factor, randomization, fragment)]
                    for fragment in range(len(source_pubs))
                )
            )
        replica_estimates = np.asarray(replica_estimates)
        replica_variances = np.asarray(replica_variances)
        factor_estimate = np.mean(replica_estimates)
        shot_variance_of_mean = float(
            np.sum(replica_variances) / repetitions**2
        )
        if repetitions > 1:
            observed_replica_variance = float(
                np.var(replica_estimates, ddof=1)
            )
            pattern_variance = max(
                observed_replica_variance
                - float(np.mean(replica_variances)),
                0.0,
            )
            pattern_variance_of_mean = pattern_variance / repetitions
        else:
            observed_replica_variance = 0.0
            pattern_variance_of_mean = 0.0
        factor_variance = (
            shot_variance_of_mean + pattern_variance_of_mean
        )
        factor_estimates.append(factor_estimate)
        factor_variances.append(factor_variance)
        factor_summaries.append(
            {
                "noise_factor": factor,
                "estimate": factor_estimate,
                "std": np.sqrt(factor_variance),
                "shot_variance_of_mean": shot_variance_of_mean,
                "pattern_variance_of_mean": pattern_variance_of_mean,
                "observed_replica_variance": observed_replica_variance,
                "replica_estimates": replica_estimates,
                "replica_stds": np.sqrt(replica_variances),
            }
        )

    factor_estimates = np.asarray(factor_estimates)
    factor_variances = np.asarray(factor_variances)
    unweighted_estimate, unweighted_variance, unweighted_coefficients = linear_zne(
        factors,
        factor_estimates,
        factor_variances,
    )
    zne_estimate, zne_variance, coefficients = (
        variance_weighted_linear_zne(
            factors,
            factor_estimates,
            factor_variances,
        )
    )
    ideal = complex(cache["Hsub"][entry])
    bias = zne_estimate - ideal
    std = float(np.sqrt(max(zne_variance, 0.0)))
    rmse = float(np.sqrt(abs(bias) ** 2 + zne_variance))
    scale_one_bias = factor_estimates[0] - ideal
    scale_one_std = float(np.sqrt(factor_variances[0]))
    scale_one_rmse = float(
        np.sqrt(abs(scale_one_bias) ** 2 + factor_variances[0])
    )
    unweighted_bias = unweighted_estimate - ideal
    unweighted_std = float(np.sqrt(max(unweighted_variance, 0.0)))
    unweighted_rmse = float(
        np.sqrt(abs(unweighted_bias) ** 2 + unweighted_variance)
    )

    output_dir = args.run_dir / (
        f"probabilistic_zne_entry_{entry[0]}_{entry[1]}_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    output_dir.mkdir(parents=True)
    results = {
        "entry": entry,
        "noise_factors": factors,
        "randomizations_per_nonunit_factor": args.randomizations,
        "cached_shot_allocation_per_factor": cache["shot_alloc"][entry],
        "factor_shot_budgets": factor_budgets,
        "fragment_allocations_by_factor": fragment_allocations_by_factor,
        "pilot_allocation": pilot_allocation,
        "factor_summaries": factor_summaries,
        "linear_coefficients": unweighted_coefficients,
        "variance_weighted_linear_coefficients": coefficients,
        "ideal_estimate": ideal,
        "zne_estimate": zne_estimate,
        "zne_bias": bias,
        "zne_std": std,
        "zne_rmse": rmse,
        "unweighted_zne_estimate": unweighted_estimate,
        "unweighted_zne_bias": unweighted_bias,
        "unweighted_zne_std": unweighted_std,
        "unweighted_zne_rmse": unweighted_rmse,
        "scale_one_estimate": factor_estimates[0],
        "scale_one_bias": scale_one_bias,
        "scale_one_std": scale_one_std,
        "scale_one_rmse": scale_one_rmse,
        "physical_readout_qubits": support,
        "fold_seed": args.fold_seed,
        "simulator_seed": args.simulator_seed,
        "calibration_seed": args.calibration_seed,
        "calibration_shots": 2 * args.calibration_shots,
    }
    save_json(output_dir / "results.json.gz", results)
    save_json(output_dir / "calibration.json.gz", calibration)
    save_json(output_dir / "fragment_results.json.gz", fragment_results)
    with (output_dir / "folded_circuits.qpy").open("wb") as file:
        qpy.dump([pub[0] for pub in folded_pubs], file)
    np.savez(
        output_dir / "zne_arrays.npz",
        noise_factors=np.asarray(factors),
        estimates=factor_estimates,
        variances=factor_variances,
        coefficients=coefficients,
        unweighted_coefficients=unweighted_coefficients,
        factor_shot_budgets=factor_budgets,
    )

    print("\nFactor estimates")
    for summary in factor_summaries:
        print(
            f"  lambda={summary['noise_factor']:g}: "
            f"{summary['estimate']} +- {summary['std']} "
            f"(pattern variance of mean="
            f"{summary['pattern_variance_of_mean']})"
        )
    print(f"Ideal entry: {ideal}")
    print(
        f"lambda=1 bias={scale_one_bias}, std={scale_one_std}, "
        f"RMSE={scale_one_rmse}"
    )
    print(
        f"Unweighted ZNE estimate={unweighted_estimate}, "
        f"bias={unweighted_bias}, std={unweighted_std}, "
        f"variance={unweighted_variance}, RMSE={unweighted_rmse}"
    )
    print(
        f"Variance-weighted ZNE estimate={zne_estimate}, bias={bias}, "
        f"std={std}, variance={zne_variance}, RMSE={rmse}"
    )
    print(f"Saved portable results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
