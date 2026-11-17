import numpy as np
from typing import Dict


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    """
    Compute mean, median, and two-sided percentile bootstrap confidence intervals.
    Automatically filters out NaN values (e.g. from HD95 on empty-empty pairs).
    """
    clean_vals = np.array(values, dtype=float)
    clean_vals = clean_vals[~np.isnan(clean_vals)]

    if len(clean_vals) == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_samples": 0,
        }

    rng = np.random.RandomState(seed)
    n = len(clean_vals)
    boot_means = np.empty(n_resamples)
    boot_medians = np.empty(n_resamples)

    alpha = (1.0 - confidence_level) / 2.0
    low_idx = int(round(alpha * n_resamples))
    high_idx = int(round((1.0 - alpha) * n_resamples))

    for b in range(n_resamples):
        resample = rng.choice(clean_vals, size=n, replace=True)
        boot_means[b] = np.mean(resample)
        boot_medians[b] = np.median(resample)

    boot_means.sort()
    boot_medians.sort()

    return {
        "mean": float(np.mean(clean_vals)),
        "median": float(np.median(clean_vals)),
        "ci_lower": float(boot_means[low_idx]),
        "ci_upper": float(boot_means[min(high_idx, n_resamples - 1)]),
        "median_ci_lower": float(boot_medians[low_idx]),
        "median_ci_upper": float(boot_medians[min(high_idx, n_resamples - 1)]),
        "n_samples": n,
    }


def paired_bootstrap_ci(
    vals_a: np.ndarray,
    vals_b: np.ndarray,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    """
    Compute bootstrap CI on paired per-image differences (vals_a - vals_b).
    """
    assert len(vals_a) == len(vals_b), "Input arrays must have identical length for paired analysis"
    diffs = np.array(vals_a, dtype=float) - np.array(vals_b, dtype=float)
    return bootstrap_ci(diffs, n_resamples=n_resamples, confidence_level=confidence_level, seed=seed)
