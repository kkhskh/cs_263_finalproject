#!/usr/bin/env python3
"""
Simulated batching noise experiment.

Reads clean VRAM fingerprints, applies analytical noise from iteration-level
batching (vLLM/Orca style), and tests whether the attacker can recover the
true (intercept, slope) fingerprint from noisy observations alone.

No GPU required — pure CPU simulation.

Usage:
    python -m experiments.simulated_noisy                         # full sweep
    python -m experiments.simulated_noisy --validate-dists        # check distributions
    python -m experiments.simulated_noisy --lambdas 2.0 --n-obs 100 --prompt-dists sharegpt --n-trials 100
    python -m experiments.simulated_noisy --model Llama-3.2-1B --lambdas 1.0 2.0
"""

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Duplicated from attacker.gpu_fingerprint to avoid pulling in heavy deps
# (torch, requests, transformers) that aren't needed for pure-CPU simulation.
MARGINAL_PROMPT_LENGTHS = [2, 1000, 1000, 2000]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Truncated Zipf prompt-length distributions. Wang et al. (BurstGPT) report
# that ChatGPT and Llama-2-13b-chat request lengths both follow a Zipf
# distribution and use theta=1.1 in their public benchmark suite. We sweep
# theta to cover the range of mean prompt lengths observed across published
# traces (LMSYS ~70, ShareGPT ~200, Splitwise conversation ~1020).
#
# At truncation N=ZIPF_MAX=4096, the means are:
#   theta=1.5 -> mean ~49,   median 2   (LMSYS-like)
#   theta=1.1 -> mean ~318,  median 19  (BurstGPT default)
#   theta=0.7 -> mean ~1015, median 507 (Splitwise-like)
PROMPT_DISTS = {
    # name: (theta, source / motivation)
    "zipf_low":  (1.5, "high theta -> short prompts, mean~50 (LMSYS-like)"),
    "zipf_mid":  (1.1, "BurstGPT (Wang et al. 2025) §5 demo eval default"),
    "zipf_high": (0.7, "low theta -> long prompts, mean~1000 (Splitwise-like)"),
}

ZIPF_MAX = 4096
PROMPT_LENGTH_CLIP = (1, 8192)

DEFAULT_LAMBDAS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 100.0]
DEFAULT_N_OBS = [10, 50, 100, 500, 1000]
DEFAULT_PROMPT_DISTS = list(PROMPT_DISTS.keys())
DEFAULT_N_TRIALS = 1000
DEFAULT_N_BOOTSTRAP = 10_000

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FINGERPRINTS_DIR = os.path.join(RESULTS_DIR, "fingerprints")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "simulated_noisy")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CleanFit:
    model: str
    quant: str
    intercept_mb: float
    slope_mb_per_token: float
    r_squared: float
    prompt_lengths: np.ndarray
    peak_vram_mb: np.ndarray


# ---------------------------------------------------------------------------
# 1. Load clean results and fit linear model
# ---------------------------------------------------------------------------

def _fit_constrained_linear(X: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Fit y = a*x + b constrained through first point. Returns (slope, intercept, r2)."""
    x0, y0 = X[0], y[0]
    X_shift = X - x0
    y_shift = y - y0
    denom = np.sum(X_shift ** 2)
    if denom == 0:
        return 0.0, y0, 1.0
    a = np.sum(X_shift * y_shift) / denom
    b = y0 - a * x0
    y_pred = a * X + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return a, b, r2


def load_clean_results(results_dir: str = FINGERPRINTS_DIR) -> Dict[Tuple[str, str], CleanFit]:
    """Load fingerprint CSVs, average across seeds, fit constrained linear model."""
    pattern = os.path.join(results_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"ERROR: No CSV files found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    file_re = re.compile(r"^(.+?)\[(.+?)\]\[seed=(\d+)\]\.csv$")
    records = []
    for fpath in files:
        m = file_re.match(os.path.basename(fpath))
        if not m:
            continue
        model, quant, seed = m.group(1), m.group(2), int(m.group(3))
        df = pd.read_csv(fpath)
        for _, row in df.iterrows():
            records.append({
                "model": model, "quant": quant, "seed": seed,
                "prompt_length": row["prompt_length"],
                "peak_vram_mb": row["peak_vram_mb"],
            })

    all_df = pd.DataFrame(records)
    fits = {}

    for (model, quant), sub in all_df.groupby(["model", "quant"]):
        avg = sub.groupby("prompt_length", as_index=False)["peak_vram_mb"].mean()
        X = avg["prompt_length"].values.astype(float)
        y = avg["peak_vram_mb"].values.astype(float)
        a, b, r2 = _fit_constrained_linear(X, y)
        fits[(model, quant)] = CleanFit(
            model=model, quant=quant,
            intercept_mb=b, slope_mb_per_token=a, r_squared=r2,
            prompt_lengths=X, peak_vram_mb=y,
        )

    print(f"Loaded {len(fits)} model/quant combos from {len(files)} CSVs")
    return fits


# ---------------------------------------------------------------------------
# 2. Prompt length distributions
# ---------------------------------------------------------------------------

def _make_zipf_draw(theta: float, x_max: int = ZIPF_MAX) -> Callable:
    """Truncated Zipf draw on {1, ..., x_max} with probability p(k) ~ k^(-theta)."""
    x = np.arange(1, x_max + 1, dtype=np.float64)
    pmf = x ** (-theta)
    pmf /= pmf.sum()
    cdf = np.cumsum(pmf)
    def draw(rng: np.random.Generator, size):
        u = rng.random(size=size)
        return (np.searchsorted(cdf, u) + 1).astype(int)
    return draw


def make_prompt_length_distribution(name: str) -> Callable:
    if name not in PROMPT_DISTS:
        raise ValueError(f"Unknown prompt distribution: {name}. Known: {list(PROMPT_DISTS)}")
    theta, _ = PROMPT_DISTS[name]
    return _make_zipf_draw(theta=float(theta))


def validate_distributions(n_samples: int = 100_000):
    rng = np.random.default_rng(42)
    print(f"\n=== Prompt-length Distribution Validation (100k draws, Zipf, N={ZIPF_MAX}) ===\n")
    print(f"{'Name':<12} {'theta':>6} {'Emp mean':>12} {'Emp median':>12} {'Note':<60}")
    print("-" * 105)
    for name, (theta, source) in PROMPT_DISTS.items():
        draw = make_prompt_length_distribution(name)
        samples = draw(rng, n_samples)
        print(f"{name:<12} {theta:>6.2f} {np.mean(samples):>12.1f} "
              f"{np.median(samples):>12.1f} {source:<60}")
    print()


# ---------------------------------------------------------------------------
# 3. Noise simulation (vectorized, chunked for memory safety)
# ---------------------------------------------------------------------------

def _simulate_batch_noise_chunk(
    slope: float, lambda_val: float, prompt_dist_fn: Callable,
    rng: np.random.Generator, n_obs: int, n_trials: int,
) -> np.ndarray:
    B = rng.poisson(lambda_val, size=(n_trials, n_obs))
    max_b = int(B.max())
    if max_b == 0:
        return np.zeros((n_trials, n_obs))
    all_lengths = prompt_dist_fn(rng, size=(n_trials, n_obs, max_b)).astype(np.float32)
    mask = np.arange(max_b)[None, None, :] < B[:, :, None]
    return slope * (all_lengths * mask).sum(axis=2)


_MAX_ELEMENTS = 50_000_000


def simulate_batch_noise(
    slope: float, lambda_val: float, prompt_dist_fn: Callable,
    rng: np.random.Generator, n_obs: int, n_trials: int = 1,
) -> np.ndarray:
    """Returns noise array of shape (n_trials, n_obs) in MB."""
    if lambda_val == 0:
        return np.zeros((n_trials, n_obs))
    estimated_max_b = max(int(lambda_val + 4 * lambda_val**0.5), 10)
    chunk_size = max(1, _MAX_ELEMENTS // (n_obs * estimated_max_b))
    if chunk_size >= n_trials:
        return _simulate_batch_noise_chunk(slope, lambda_val, prompt_dist_fn, rng, n_obs, n_trials)
    chunks = []
    remaining = n_trials
    while remaining > 0:
        c = min(chunk_size, remaining)
        chunks.append(_simulate_batch_noise_chunk(slope, lambda_val, prompt_dist_fn, rng, n_obs, c))
        remaining -= c
    return np.concatenate(chunks, axis=0)


# ---------------------------------------------------------------------------
# 4. Bias correction — observation-only, no true params leaked
# ---------------------------------------------------------------------------

def correct_bias_distribution_aware(
    observations_per_step: Dict[float, np.ndarray],
    assumed_dist_name: str,
    rng: np.random.Generator,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
) -> Tuple[float, float]:
    """
    Distribution-aware bias correction using only the attacker's observations.

    1. Fit slope from noisy mins at each prompt-length step
    2. Estimate lambda from observation variance
    3. Bootstrap bias using estimated slope + estimated lambda + assumed distribution

    Returns (corrected_intercept, estimated_lambda).
    """
    # Step 1: get noisy mins and fit slope
    prompt_lengths = sorted(observations_per_step.keys())
    noisy_mins = np.array([observations_per_step[pl].min() for pl in prompt_lengths])
    pl_arr = np.array(prompt_lengths, dtype=float)
    est_slope, est_intercept, _ = _fit_constrained_linear(pl_arr, noisy_mins)

    if est_slope <= 0 or len(prompt_lengths) < 2:
        return est_intercept, 0.0

    # Step 2: estimate lambda from observation variance at the first step
    # Var(noise) = slope^2 * lambda * E[L^2]
    # We estimate E[L^2] from the assumed distribution
    dist_fn = make_prompt_length_distribution(assumed_dist_name)
    sample_lengths = dist_fn(rng, 100_000).astype(float)
    e_l2 = np.mean(sample_lengths ** 2)

    # Use variance of observations at first prompt-length step
    first_obs = observations_per_step[prompt_lengths[0]]
    obs_var = np.var(first_obs)
    est_lambda = obs_var / (est_slope ** 2 * e_l2) if (est_slope ** 2 * e_l2) > 0 else 0.0
    est_lambda = max(est_lambda, 0.0)

    if est_lambda < 0.01:
        return est_intercept, 0.0

    # Step 3: bootstrap bias using estimated params
    n_obs = len(first_obs)
    noise_sim = simulate_batch_noise(est_slope, est_lambda, dist_fn, rng, n_obs, n_bootstrap)
    block_mins = noise_sim.min(axis=1)
    bootstrap_bias = float(np.mean(block_mins))

    corrected = est_intercept - bootstrap_bias
    return corrected, est_lambda


def correct_bias_distribution_free(
    observations: np.ndarray,
) -> float:
    """
    Distribution-free bias correction using order statistics.

    Uses the gap between the smallest observations to estimate bias.
    For a sample of size N from F with min m, the expected bias is approximately
    the mean gap between order statistics near the minimum.

    Returns corrected_intercept.
    """
    sorted_obs = np.sort(observations)
    n = len(sorted_obs)
    raw_min = sorted_obs[0]

    if n < 3:
        return raw_min

    # Use spacing between lowest order statistics
    # E[X_(1)] ≈ true_min + mean_spacing / 1
    # mean_spacing ≈ (X_(k) - X_(1)) / (k-1) for small k
    k = max(2, min(int(n * 0.05), 20))  # use bottom 5% or at most 20 points
    spacings = sorted_obs[1:k] - sorted_obs[0]
    if len(spacings) == 0:
        return raw_min

    # The expected min for N iid draws with local spacing s is approximately s/N
    # (exponential approximation near the minimum)
    avg_spacing = np.mean(spacings)
    bias_estimate = avg_spacing * (k - 1) / n

    return raw_min - bias_estimate


# ---------------------------------------------------------------------------
# 5. Full trial simulation — recovers (intercept, slope) from noisy obs
# ---------------------------------------------------------------------------

def simulate_full_trial(
    clean_fit: CleanFit,
    lambda_val: float,
    dist_fn: Callable,
    dist_name: str,
    n_obs: int,
    n_trials: int,
    rng: np.random.Generator,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
) -> pd.DataFrame:
    """
    Simulate full fingerprint recovery under noise.

    For each trial:
    1. Generate noisy observations at each prompt-length step
    2. Take min at each step → fit (intercept, slope)
    3. Apply bias corrections (distribution-aware and distribution-free)
    4. Compute MAPE for intercept and slope

    The distribution-aware bootstrap is computed ONCE using pilot estimates
    (median slope/lambda across trials), then applied to all trials.
    """
    prompt_lengths = clean_fit.prompt_lengths
    clean_vram = clean_fit.peak_vram_mb
    true_intercept = clean_fit.intercept_mb
    true_slope = clean_fit.slope_mb_per_token
    n_steps = len(prompt_lengths)
    pl_arr = prompt_lengths.astype(float)

    # Generate noise for all steps: (n_steps, n_trials, n_obs)
    all_noise = np.zeros((n_steps, n_trials, n_obs))
    for s in range(n_steps):
        all_noise[s] = simulate_batch_noise(
            true_slope, lambda_val, dist_fn, rng, n_obs, n_trials
        )

    # Observed VRAM: clean + noise — shape (n_steps, n_trials, n_obs)
    all_observed = clean_vram[:, None, None] + all_noise

    # Min at each step per trial — shape (n_steps, n_trials)
    step_mins = all_observed.min(axis=2)

    # --- Fit slope/intercept from minima (constrained through first-step min) ---
    x0, y0s = pl_arr[0], step_mins[0, :]  # first-step mins per trial
    X_shift = pl_arr - x0  # (n_steps,)
    Y_shift = step_mins - y0s[None, :]  # (n_steps, n_trials)
    denom = np.sum(X_shift ** 2)
    rec_slopes = np.sum(X_shift[:, None] * Y_shift, axis=0) / denom  # (n_trials,)
    rec_intercepts = y0s - rec_slopes * x0  # (n_trials,)

    # --- Distribution-free correction (vectorized) ---
    first_step_sorted = np.sort(all_observed[0], axis=1)  # (n_trials, n_obs)
    n = n_obs
    k = max(2, min(int(n * 0.05), 20))
    raw_mins = first_step_sorted[:, 0]
    if k > 1 and n >= 3:
        spacings = first_step_sorted[:, 1:k] - first_step_sorted[:, 0:1]  # (n_trials, k-1)
        avg_spacing = spacings.mean(axis=1)
        bias_free = avg_spacing * (k - 1) / n
        corrected_free = raw_mins - bias_free
    else:
        corrected_free = raw_mins

    # --- Distribution-aware correction (computed ONCE from pilot estimates) ---
    # Use median slope and estimated lambda across all trials as pilot
    est_slope_pilot = float(np.median(rec_slopes))
    if est_slope_pilot > 0 and lambda_val > 0:
        # Estimate lambda from observation variance (median across trials)
        obs_vars = np.var(all_observed[0], axis=1)  # (n_trials,)
        median_var = float(np.median(obs_vars))

        dist_fn_assumed = make_prompt_length_distribution(dist_name)
        sample_lengths = dist_fn_assumed(rng, 100_000).astype(float)
        e_l2 = np.mean(sample_lengths ** 2)

        est_lambda_pilot = median_var / (est_slope_pilot ** 2 * e_l2) if (est_slope_pilot ** 2 * e_l2) > 0 else 0.0
        est_lambda_pilot = max(est_lambda_pilot, 0.0)

        if est_lambda_pilot > 0.01:
            noise_sim = simulate_batch_noise(est_slope_pilot, est_lambda_pilot, dist_fn_assumed, rng, n_obs, n_bootstrap)
            bootstrap_bias = float(np.mean(noise_sim.min(axis=1)))
        else:
            bootstrap_bias = 0.0
            est_lambda_pilot = 0.0
    else:
        bootstrap_bias = 0.0
        est_lambda_pilot = 0.0

    corrected_aware = rec_intercepts - bootstrap_bias

    # --- Compute MAPEs ---
    int_mape_raw = np.abs(rec_intercepts - true_intercept) / abs(true_intercept) * 100 if true_intercept != 0 else np.zeros(n_trials)
    int_mape_free = np.abs(corrected_free - true_intercept) / abs(true_intercept) * 100 if true_intercept != 0 else np.zeros(n_trials)
    int_mape_aware = np.abs(corrected_aware - true_intercept) / abs(true_intercept) * 100 if true_intercept != 0 else np.zeros(n_trials)
    slope_mape = np.abs(rec_slopes - true_slope) / abs(true_slope) * 100 if true_slope != 0 else np.zeros(n_trials)

    # Build DataFrame
    df = pd.DataFrame({
        "trial": np.arange(n_trials),
        "recovered_intercept_mb": rec_intercepts,
        "recovered_slope": rec_slopes,
        "corrected_intercept_free_mb": corrected_free,
        "corrected_intercept_aware_mb": corrected_aware,
        "estimated_lambda": est_lambda_pilot,
        "true_intercept_mb": true_intercept,
        "true_slope": true_slope,
        "intercept_mape_raw": int_mape_raw,
        "intercept_mape_free": int_mape_free,
        "intercept_mape_aware": int_mape_aware,
        "slope_mape": slope_mape,
    })

    return df


# ---------------------------------------------------------------------------
# 6. Phase 2: per-step noisy observations (for fingerprint shape plots)
# ---------------------------------------------------------------------------

def simulate_phase2(
    clean_fit: CleanFit, lambda_val: float, dist_fn: Callable,
    n_obs: int, n_trials: int, rng: np.random.Generator,
) -> pd.DataFrame:
    records = []
    for pl, clean_vram in zip(clean_fit.prompt_lengths, clean_fit.peak_vram_mb):
        noise = simulate_batch_noise(
            clean_fit.slope_mb_per_token, lambda_val, dist_fn, rng, n_obs, n_trials
        )
        observed = clean_vram + noise
        mins = observed.min(axis=1)
        peaks = observed.max(axis=1)
        for t in range(n_trials):
            records.append({
                "trial": t, "prompt_length": int(pl),
                "noisy_min_mb": mins[t], "noisy_peak_mb": peaks[t],
                "clean_vram_mb": clean_vram,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 7. Main sweep
# ---------------------------------------------------------------------------

def _append_csv(path: str, df: pd.DataFrame, columns: List[str]):
    write_header = not os.path.exists(path)
    df[columns].to_csv(path, mode="a", header=write_header, index=False)


def run_sweep(
    fits: Dict[Tuple[str, str], CleanFit],
    lambdas: List[float],
    n_obs_list: List[int],
    prompt_dist_names: List[str],
    n_trials: int,
    n_bootstrap: int,
    output_dir: str,
    model_filter: Optional[str] = None,
):
    os.makedirs(output_dir, exist_ok=True)

    # Output paths
    trial_path = os.path.join(output_dir, "trial_summary.csv")
    p2_path = os.path.join(output_dir, "phase2_summary.csv")

    # Column orderings
    col_order_trial = [
        "model", "quant", "lambda", "prompt_dist", "n_obs", "trial",
        "recovered_intercept_mb", "recovered_slope",
        "corrected_intercept_free_mb", "corrected_intercept_aware_mb",
        "estimated_lambda",
        "true_intercept_mb", "true_slope",
        "intercept_mape_raw", "intercept_mape_free", "intercept_mape_aware",
        "slope_mape",
    ]
    col_order_p2 = [
        "model", "quant", "lambda", "prompt_dist", "n_obs", "trial",
        "prompt_length", "noisy_min_mb", "noisy_peak_mb", "clean_vram_mb",
    ]

    # Resume support: load already-completed configs
    done_configs = set()
    if os.path.exists(trial_path):
        existing = pd.read_csv(trial_path, usecols=["model", "quant", "lambda", "prompt_dist", "n_obs"])
        for _, row in existing.drop_duplicates().iterrows():
            done_configs.add((row["model"], row["quant"], row["lambda"], row["prompt_dist"], int(row["n_obs"])))
        print(f"Resuming: {len(done_configs)} configs already completed")

    # Write clean_fits.csv
    clean_rows = []
    for (model, quant), fit in sorted(fits.items()):
        clean_rows.append({
            "model": model, "quant": quant,
            "intercept_mb": fit.intercept_mb,
            "slope_mb_per_token": fit.slope_mb_per_token,
            "r_squared": fit.r_squared,
        })
    pd.DataFrame(clean_rows).to_csv(os.path.join(output_dir, "clean_fits.csv"), index=False)
    print(f"Wrote clean_fits.csv ({len(clean_rows)} models)")

    # Filter models
    target_fits = fits
    if model_filter:
        target_fits = {k: v for k, v in fits.items() if model_filter in k[0]}
        if not target_fits:
            print(f"ERROR: No models matching '{model_filter}'", file=sys.stderr)
            sys.exit(1)
        print(f"Filtered to {len(target_fits)} models matching '{model_filter}'")

    total_configs = len(target_fits) * len(lambdas) * len(prompt_dist_names) * len(n_obs_list)
    config_idx = 0
    trial_total = 0
    p2_total = 0

    rng = np.random.default_rng(2024)

    for (model, quant), fit in sorted(target_fits.items()):
        for dist_name in prompt_dist_names:
            dist_fn = make_prompt_length_distribution(dist_name)

            for lam in lambdas:
                for n_obs in n_obs_list:
                    config_idx += 1
                    config_key = (model, quant, lam, dist_name, n_obs)
                    if config_key in done_configs:
                        continue
                    if config_idx % 20 == 0 or config_idx == 1:
                        print(f"  [{config_idx}/{total_configs}] {model}[{quant}] "
                              f"λ={lam} dist={dist_name} N={n_obs}")

                    # Full trial simulation
                    trial_df = simulate_full_trial(
                        fit, lam, dist_fn, dist_name, n_obs, n_trials, rng, n_bootstrap,
                    )
                    trial_df["model"] = model
                    trial_df["quant"] = quant
                    trial_df["lambda"] = lam
                    trial_df["prompt_dist"] = dist_name
                    trial_df["n_obs"] = n_obs
                    _append_csv(trial_path, trial_df, col_order_trial)
                    trial_total += len(trial_df)

                    # Phase 2: per-step observations (cap at 100 trials to save disk)
                    p2_trials = min(n_trials, 100)
                    p2 = simulate_phase2(fit, lam, dist_fn, n_obs, p2_trials, rng)
                    p2["model"] = model
                    p2["quant"] = quant
                    p2["lambda"] = lam
                    p2["prompt_dist"] = dist_name
                    p2["n_obs"] = n_obs
                    _append_csv(p2_path, p2, col_order_p2)
                    p2_total += len(p2)

    print(f"Wrote trial_summary.csv ({trial_total} rows)")
    print(f"Wrote phase2_summary.csv ({p2_total} rows)")
    print(f"\nAll outputs in: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulated batching noise experiment for VRAM fingerprinting"
    )
    parser.add_argument("--validate-dists", action="store_true")
    parser.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    parser.add_argument("--n-obs", nargs="+", type=int, default=DEFAULT_N_OBS)
    parser.add_argument("--prompt-dists", nargs="+", default=DEFAULT_PROMPT_DISTS)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=FINGERPRINTS_DIR)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.validate_dists:
        validate_distributions()
        return

    print("=== Simulated Batching Noise Experiment ===\n")
    fits = load_clean_results(args.results_dir)
    print()

    run_sweep(
        fits=fits,
        lambdas=args.lambdas,
        n_obs_list=args.n_obs,
        prompt_dist_names=args.prompt_dists,
        n_trials=args.n_trials,
        n_bootstrap=args.n_bootstrap,
        output_dir=args.output_dir,
        model_filter=args.model,
    )


if __name__ == "__main__":
    main()
