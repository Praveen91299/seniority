"""Compare weighted linear, quadratic, and offset-exponential ZNE models."""

from __future__ import annotations

from _bootstrap import REPO_ROOT

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2, norm

from src.saved_run_io import load_json, save_json


def _weighted_polynomial(x, y, variances, degree):
    design = np.vander(x, degree + 1, increasing=True)
    weights = 1.0 / variances
    covariance = np.linalg.inv(
        design.T @ (weights[:, None] * design)
    )
    mapping = covariance @ (design.T * weights)
    parameters = mapping @ y
    return parameters, covariance, mapping, design @ parameters


def _exponential_model(x, intercept, initial_slope, decay):
    """Offset exponential parameterized by its zero-noise intercept."""
    if abs(decay) < 1e-10:
        scaled_x = x
    else:
        scaled_x = -np.expm1(-decay * x) / decay
    return intercept + initial_slope * scaled_x


def _weighted_exponential(x, y, stds, initial=None):
    if initial is None:
        linear, _, _, _ = _weighted_polynomial(
            x, y, stds**2, degree=1
        )
        initial = (linear[0], linear[1], 0.25)
    parameters, covariance = curve_fit(
        _exponential_model,
        x,
        y,
        p0=initial,
        sigma=stds,
        absolute_sigma=True,
        bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, 5.0]),
        maxfev=50_000,
    )
    return (
        parameters,
        covariance,
        _exponential_model(x, *parameters),
    )


def _quality(y, fitted, variances, parameter_count):
    statistic = float(np.sum((y - fitted) ** 2 / variances))
    dof = int(y.size - parameter_count)
    return {
        "chi_square": statistic,
        "degrees_of_freedom": dof,
        "reduced_chi_square": statistic / dof,
        "p_value": float(chi2.sf(statistic, dof)),
        "aic_without_common_constant": statistic + 2 * parameter_count,
        "bic_without_common_constant": (
            statistic + parameter_count * np.log(y.size)
        ),
        "residuals": y - fitted,
        "standardized_residuals": (y - fitted) / np.sqrt(variances),
    }


def _pool_results(paths):
    """Inverse-variance pool independently calibrated factor estimates."""
    records = {}
    ideal = None
    for path in paths:
        source = load_json(path)
        source_ideal = complex(source["ideal_estimate"]).real
        if ideal is None:
            ideal = source_ideal
        elif not np.isclose(ideal, source_ideal):
            raise ValueError("Source runs have different ideal estimates.")
        for summary in source["factor_summaries"]:
            factor = float(summary["noise_factor"])
            records.setdefault(factor, []).append(
                (
                    complex(summary["estimate"]).real,
                    float(summary["std"]) ** 2,
                    path,
                )
            )
    factors = np.asarray(sorted(records))
    estimates = []
    variances = []
    components = {}
    for factor in factors:
        items = records[factor]
        inverse_variances = np.asarray([1.0 / item[1] for item in items])
        estimates.append(
            np.sum(
                inverse_variances
                * np.asarray([item[0] for item in items])
            )
            / np.sum(inverse_variances)
        )
        variances.append(1.0 / np.sum(inverse_variances))
        components[str(factor)] = [
            {
                "estimate": estimate,
                "variance": variance,
                "source": source,
            }
            for estimate, variance, source in items
        ]
    return (
        factors,
        np.asarray(estimates),
        np.asarray(variances),
        ideal,
        components,
    )


def _prediction_variance(model, x_value, parameters, covariance):
    gradient = np.empty(parameters.size)
    for index in range(parameters.size):
        step = 1e-5 * max(1.0, abs(parameters[index]))
        upper = parameters.copy()
        lower = parameters.copy()
        upper[index] += step
        lower[index] -= step
        gradient[index] = (
            model(np.asarray([x_value]), *upper)[0]
            - model(np.asarray([x_value]), *lower)[0]
        ) / (2 * step)
    return float(gradient @ covariance @ gradient)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2_718_281)
    parser.add_argument("--holdout-factor", type=float)
    args = parser.parse_args()

    (
        factors,
        estimates,
        variances,
        ideal,
        pooling_components,
    ) = _pool_results(args.results)
    stds = np.sqrt(variances)

    holdout = None
    if args.holdout_factor is not None:
        matches = np.flatnonzero(
            np.isclose(factors, args.holdout_factor)
        )
        if matches.size != 1:
            raise ValueError(
                f"Holdout factor {args.holdout_factor} is unavailable."
            )
        holdout_index = int(matches[0])
        training = np.arange(factors.size) != holdout_index
        train_x = factors[training]
        train_y = estimates[training]
        train_v = variances[training]
        train_s = np.sqrt(train_v)
        holdout = {
            "noise_factor": factors[holdout_index],
            "measured_estimate": estimates[holdout_index],
            "measured_variance": variances[holdout_index],
            "models": {},
        }
        for name, degree in (("linear", 1), ("quadratic", 2)):
            parameters, covariance, _, _ = _weighted_polynomial(
                train_x, train_y, train_v, degree
            )
            row = np.asarray(
                [factors[holdout_index] ** power for power in range(degree + 1)]
            )
            prediction = float(row @ parameters)
            prediction_variance = float(row @ covariance @ row)
            residual_variance = (
                prediction_variance + variances[holdout_index]
            )
            holdout["models"][name] = {
                "prediction": prediction,
                "prediction_variance": prediction_variance,
                "residual": estimates[holdout_index] - prediction,
                "standardized_residual": (
                    estimates[holdout_index] - prediction
                ) / np.sqrt(residual_variance),
            }
        parameters, covariance, _ = _weighted_exponential(
            train_x, train_y, train_s
        )
        prediction = float(
            _exponential_model(
                np.asarray([factors[holdout_index]]), *parameters
            )[0]
        )
        prediction_variance = _prediction_variance(
            _exponential_model,
            factors[holdout_index],
            parameters,
            covariance,
        )
        residual_variance = prediction_variance + variances[holdout_index]
        holdout["models"]["offset_exponential"] = {
            "prediction": prediction,
            "prediction_variance": prediction_variance,
            "residual": estimates[holdout_index] - prediction,
            "standardized_residual": (
                estimates[holdout_index] - prediction
            ) / np.sqrt(residual_variance),
        }

    linear = _weighted_polynomial(factors, estimates, variances, 1)
    quadratic = _weighted_polynomial(factors, estimates, variances, 2)
    exponential = _weighted_exponential(factors, estimates, stds)

    rng = np.random.default_rng(args.seed)
    resampled = rng.normal(
        loc=estimates,
        scale=stds,
        size=(args.bootstrap_samples, estimates.size),
    )
    samples = {
        "linear": resampled @ linear[2][0],
        "quadratic": resampled @ quadratic[2][0],
        "linear_parameters": resampled @ linear[2].T,
        "quadratic_parameters": resampled @ quadratic[2].T,
    }
    exponential_samples = []
    failed_exponential_fits = 0
    for sample in resampled:
        try:
            fit, _, _ = _weighted_exponential(
                factors,
                sample,
                stds,
                initial=exponential[0],
            )
            exponential_samples.append(fit[0])
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            failed_exponential_fits += 1
    samples["offset_exponential"] = np.asarray(exponential_samples)

    fits = {}
    definitions = {
        "linear": (linear[0], linear[3], 2),
        "quadratic": (quadratic[0], quadratic[3], 3),
        "offset_exponential": (exponential[0], exponential[2], 3),
    }
    for name, (parameters, fitted, parameter_count) in definitions.items():
        intercept = float(parameters[0])
        bootstrap = samples[name]
        bootstrap_variance = float(np.var(bootstrap, ddof=1))
        bias = intercept - ideal
        fits[name] = {
            "parameters": parameters,
            "zero_noise_estimate": intercept,
            "observed_bias": bias,
            "bootstrap_variance": bootstrap_variance,
            "bootstrap_std": np.sqrt(bootstrap_variance),
            "bootstrap_95_percent_interval": np.quantile(
                bootstrap, [0.025, 0.975]
            ),
            "rmse_from_observed_bias": np.sqrt(
                bias**2 + bootstrap_variance
            ),
            "fit_quality": _quality(
                estimates,
                fitted,
                variances,
                parameter_count,
            ),
            "fitted_values": fitted,
        }
    curvature_samples = samples["quadratic_parameters"][:, 2]
    curvature_std = float(np.std(curvature_samples, ddof=1))
    fits["quadratic"]["curvature"] = {
        "estimate": float(quadratic[0][2]),
        "bootstrap_std": curvature_std,
        "bootstrap_95_percent_interval": np.quantile(
            curvature_samples, [0.025, 0.975]
        ),
        "two_sided_normal_p_value": float(
            2 * norm.sf(abs(quadratic[0][2]) / curvature_std)
        ),
    }

    output_dir = args.results[-1].parent
    payload = {
        "source_results": args.results,
        "pooling_components": pooling_components,
        "noise_factors": factors,
        "factor_estimates": estimates,
        "factor_variances": variances,
        "ideal_estimate": ideal,
        "bootstrap_samples_requested": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "failed_exponential_fits": failed_exponential_fits,
        "holdout_test": holdout,
        "fits": fits,
    }
    save_json(output_dir / "zne_model_comparison.json.gz", payload)
    np.savez(
        output_dir / "zne_model_bootstrap_samples.npz",
        **samples,
    )

    for name, fit in fits.items():
        quality = fit["fit_quality"]
        print(
            f"{name}: estimate={fit['zero_noise_estimate']:.9f}, "
            f"bias={fit['observed_bias']:+.9f}, "
            f"variance={fit['bootstrap_variance']:.9f}, "
            f"std={fit['bootstrap_std']:.9f}, "
            f"RMSE={fit['rmse_from_observed_bias']:.9f}, "
            f"chi2/dof={quality['chi_square']:.4f}/"
            f"{quality['degrees_of_freedom']}, "
            f"p={quality['p_value']:.4f}"
        )
    print(f"Failed exponential bootstrap fits: {failed_exponential_fits}")
    print(f"Saved model comparison to {output_dir}")


if __name__ == "__main__":
    main()
