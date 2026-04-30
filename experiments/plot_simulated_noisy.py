#!/usr/bin/env python3
"""
Plot results from the simulated batching noise experiment.

Usage:
    python -m experiments.plot_simulated_noisy
    python -m experiments.plot_simulated_noisy --input-dir results/simulated_noisy --output-dir figures/simulated_noisy
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "serif"

INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "simulated_noisy")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures", "simulated_noisy")


def load_data(input_dir):
    clean = pd.read_csv(os.path.join(input_dir, "clean_fits.csv"))
    trials = pd.read_csv(os.path.join(input_dir, "trial_summary.csv"))
    p2 = pd.read_csv(os.path.join(input_dir, "phase2_summary.csv"))
    return clean, trials, p2


# ---------------------------------------------------------------------------
# Helper: MAPE heatmap
# ---------------------------------------------------------------------------

def _plot_mape_heatmap(pivot, ax, title, fmt=".2f", vmax=None):
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(int), fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"λ={v:g}" for v in pivot.index], fontsize=10)
    ax.set_xlabel("N (observations)", fontsize=14)
    ax.set_ylabel("Mean batch size (λ)", fontsize=14)
    ax.set_title(title, fontsize=16)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "white" if val > (vmax or pivot.values.max()) * 0.5 else "black"
            ax.text(j, i, f"{val:{fmt}}%", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)


# ---------------------------------------------------------------------------
# Plot 1: Intercept MAPE heatmap
# ---------------------------------------------------------------------------

def plot_intercept_mape_heatmap(trials, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    for ax, col, title in [
        (axes[0], "intercept_mape_raw", "Raw Min"),
        (axes[1], "intercept_mape_free", "Distribution-Free Correction"),
        (axes[2], "intercept_mape_aware", "Distribution-Aware Correction"),
    ]:
        pivot = trials.groupby(["lambda", "n_obs"])[col].mean().unstack("n_obs")
        vmax = trials.groupby(["lambda", "n_obs"])["intercept_mape_raw"].mean().max()
        _plot_mape_heatmap(pivot, ax, f"{title} — Intercept MAPE (%)", vmax=vmax)
    fig.suptitle("Intercept Recovery MAPE (averaged over all models & distributions)", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "intercept_mape_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved intercept_mape_heatmap.png")


# ---------------------------------------------------------------------------
# Plot 2: Slope MAPE heatmap
# ---------------------------------------------------------------------------

def plot_slope_mape_heatmap(trials, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot = trials.groupby(["lambda", "n_obs"])["slope_mape"].mean().unstack("n_obs")
    _plot_mape_heatmap(pivot, ax, "Slope MAPE (%)")
    fig.suptitle("Slope Recovery MAPE (averaged over all models & distributions)", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "slope_mape_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved slope_mape_heatmap.png")


# ---------------------------------------------------------------------------
# Plot 3: Intercept MAPE vs N (line plot)
# ---------------------------------------------------------------------------

def _mape_palette(n):
    """Muted academic palette for line plots — good contrast, colorblind-safe."""
    return sns.color_palette("colorblind", n_colors=n)


def plot_intercept_mape_vs_n(trials, output_dir):
    agg = trials.groupby(["lambda", "n_obs"]).agg(
        median=("intercept_mape_aware", "median"),
        q25=("intercept_mape_aware", lambda x: x.quantile(0.25)),
        q75=("intercept_mape_aware", lambda x: x.quantile(0.75)),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(13, 6))
    palette = _mape_palette(agg["lambda"].nunique())
    for i, (lam, g) in enumerate(agg.groupby("lambda")):
        g = g.sort_values("n_obs")
        ax.plot(g["n_obs"], g["median"], marker="o", linewidth=2, markersize=6,
                label=f"λ={lam:g}", color=palette[i])
        ax.fill_between(g["n_obs"], g["q25"], g["q75"], alpha=0.15, color=palette[i])

    ax.set_xlabel("N (observations per trial)", fontsize=22)
    ax.set_ylabel("Median intercept MAPE (%)", fontsize=22)
    ax.set_title("Intercept Recovery MAPE vs Sample Size (distribution-aware)", fontsize=26)
    ax.tick_params(axis="both", labelsize=18)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=18, title="Mean batch size", title_fontsize=20,
              ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "intercept_mape_vs_n.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved intercept_mape_vs_n.png")


# ---------------------------------------------------------------------------
# Plot 4: Slope MAPE vs N (line plot)
# ---------------------------------------------------------------------------

def plot_slope_mape_vs_n(trials, output_dir):
    agg = trials.groupby(["lambda", "n_obs"]).agg(
        median=("slope_mape", "median"),
        q25=("slope_mape", lambda x: x.quantile(0.25)),
        q75=("slope_mape", lambda x: x.quantile(0.75)),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = _mape_palette(agg["lambda"].nunique())
    for i, (lam, g) in enumerate(agg.groupby("lambda")):
        g = g.sort_values("n_obs")
        ax.plot(g["n_obs"], g["median"], marker="o", linewidth=2, markersize=6,
                label=f"λ={lam:g}", color=palette[i])
        ax.fill_between(g["n_obs"], g["q25"], g["q75"], alpha=0.15, color=palette[i])

    ax.set_xlabel("N (observations per trial)", fontsize=14)
    ax.set_ylabel("Median slope MAPE (%)", fontsize=14)
    ax.set_title("Slope Recovery MAPE vs Sample Size", fontsize=16)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10, title="Mean batch size", title_fontsize=12, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "slope_mape_vs_n.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved slope_mape_vs_n.png")


# ---------------------------------------------------------------------------
# Plot 5: Intercept MAPE by prompt distribution
# ---------------------------------------------------------------------------

_ZIPF_LABELS = {"zipf_low": "Zipf(1.5)", "zipf_mid": "Zipf(1.1)", "zipf_high": "Zipf(0.7)"}


def plot_intercept_mape_by_dist(trials, output_dir):
    agg = trials.groupby(["lambda", "prompt_dist"])["intercept_mape_aware"].mean().reset_index()
    agg["prompt_dist"] = agg["prompt_dist"].map(_ZIPF_LABELS).fillna(agg["prompt_dist"])
    fig, ax = plt.subplots(figsize=(12, 6))
    dist_palette = {"Zipf(1.5)": "#4c72b0", "Zipf(1.1)": "#dd8452", "Zipf(0.7)": "#55a868"}
    sns.barplot(data=agg, x="lambda", y="intercept_mape_aware", hue="prompt_dist",
                hue_order=["Zipf(1.5)", "Zipf(1.1)", "Zipf(0.7)"], ax=ax, palette=dist_palette)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f%%", fontsize=11, padding=3, rotation=45)
    ax.set_xlabel("Mean batch size (λ)", fontsize=22)
    ax.set_ylabel("Mean intercept MAPE (%)", fontsize=22)
    ax.set_title("Intercept MAPE by Prompt Distribution (distribution-aware)", fontsize=26)
    ax.tick_params(axis="both", labelsize=18)
    ax.set_yscale("symlog", linthresh=0.01)
    cur_top = ax.get_ylim()[1]
    ax.set_ylim(top=cur_top * 30)
    ax.legend(fontsize=18, title="Prompt distribution", title_fontsize=20)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "intercept_mape_by_dist.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved intercept_mape_by_dist.png")


# ---------------------------------------------------------------------------
# Plot 6: Slope MAPE by prompt distribution
# ---------------------------------------------------------------------------

def plot_slope_mape_by_dist(trials, output_dir):
    agg = trials.groupby(["lambda", "prompt_dist"])["slope_mape"].mean().reset_index()
    agg["prompt_dist"] = agg["prompt_dist"].map(_ZIPF_LABELS).fillna(agg["prompt_dist"])
    fig, ax = plt.subplots(figsize=(12, 6))
    dist_palette = {"Zipf(1.5)": "#4c72b0", "Zipf(1.1)": "#dd8452", "Zipf(0.7)": "#55a868"}
    sns.barplot(data=agg, x="lambda", y="slope_mape", hue="prompt_dist",
                hue_order=["Zipf(1.5)", "Zipf(1.1)", "Zipf(0.7)"], ax=ax, palette=dist_palette)
    ax.set_xlabel("Mean batch size (λ)", fontsize=14)
    ax.set_ylabel("Mean slope MAPE (%)", fontsize=14)
    ax.set_title("Slope MAPE by Prompt Distribution", fontsize=16)
    ax.legend(fontsize=11, title="Prompt distribution", title_fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "slope_mape_by_dist.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved slope_mape_by_dist.png")


# ---------------------------------------------------------------------------
# Plot 7: Bias correction comparison
# ---------------------------------------------------------------------------

def plot_bias_correction_comparison(trials, output_dir):
    """Compare raw, dist-free, dist-aware for intercept, plus single slope bar."""
    import matplotlib as mpl
    from matplotlib.patches import Patch

    agg = trials.groupby(["lambda", "n_obs"]).agg(
        int_raw=("intercept_mape_raw", "mean"),
        int_free=("intercept_mape_free", "mean"),
        int_aware=("intercept_mape_aware", "mean"),
        slope_raw=("slope_mape", "mean"),
    ).reset_index()

    # Pick N=100 as representative
    sub = agg[agg["n_obs"] == 100].copy()
    if sub.empty:
        sub = agg[agg["n_obs"] == agg["n_obs"].median()].copy()

    n_lam = len(sub)
    x = np.arange(n_lam)
    n_val = int(sub["n_obs"].iloc[0])

    # Colors for the three intercept correction methods
    c_raw = "#c44e52"    # red
    c_free = "#dd8452"   # orange
    c_aware = "#4c72b0"  # blue
    c_slope = "#999999"  # gray for single slope bar

    # Layout: 3 intercept bars + gap + 1 slope bar (with white hatch)
    w = 0.13
    gap = 0.08
    group_width = 3 * w + gap + w
    start = -group_width / 2 + w / 2

    old_hatch_color = mpl.rcParams.get("hatch.color", "black")
    old_hatch_lw = mpl.rcParams.get("hatch.linewidth", 1.0)
    mpl.rcParams["hatch.color"] = "white"
    mpl.rcParams["hatch.linewidth"] = 2.0

    fig, ax = plt.subplots(figsize=(12, 6))

    # Intercept bars (solid)
    bars_raw   = ax.bar(x + start,         sub["int_raw"].values,   w, color=c_raw,   edgecolor="black", linewidth=0.4)
    bars_free  = ax.bar(x + start + w,     sub["int_free"].values,  w, color=c_free,  edgecolor="black", linewidth=0.4)
    bars_aware = ax.bar(x + start + 2 * w, sub["int_aware"].values, w, color=c_aware, edgecolor="black", linewidth=0.4)

    # Single slope bar (hatched) — no correction applies to slope
    slope_pos = start + 3 * w + gap
    bars_slope = ax.bar(x + slope_pos, sub["slope_raw"].values, w, color=c_slope, edgecolor="black", linewidth=0.4, hatch="//")

    # Skip labels for the three smallest λ (already near 0) on raw/aware/slope,
    # but always label the distribution-free bar. Shift labels half a bar width
    # to the right so they read above the bar's right edge.
    skip_lams = {0.5, 1.0, 2.0}
    always_label = {id(bars_free)}
    lam_values = sub["lambda"].values
    for bars in (bars_raw, bars_free, bars_aware, bars_slope):
        for bar, lam in zip(bars, lam_values):
            if lam in skip_lams and id(bars) not in always_label:
                continue
            xpos = bar.get_x() + bar.get_width()
            ypos = bar.get_height()
            ax.annotate(f"{ypos:.0f}%", xy=(xpos, ypos),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([f"λ={v:g}" for v in sub["lambda"]], fontsize=16)
    ax.tick_params(axis="x", pad=18)

    for xi in x:
        int_center = xi + start + w
        sl_center = xi + slope_pos
        ax.annotate("intercept", xy=(int_center, 0),
                    xycoords=("data", "data"),
                    xytext=(int_center, -0.04), textcoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=7, fontstyle="italic",
                    annotation_clip=False)
        ax.annotate("slope", xy=(sl_center, 0),
                    xycoords=("data", "data"),
                    xytext=(sl_center, -0.04), textcoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=7, fontstyle="italic",
                    annotation_clip=False)

    ax.set_xlabel("Mean batch size (λ)", fontsize=22, labelpad=22)
    ax.set_ylabel("Mean MAPE (%)", fontsize=22)
    ax.set_yscale("symlog", linthresh=0.01)
    cur_top_bc = ax.get_ylim()[1]
    ax.set_ylim(top=cur_top_bc * 5)
    ax.set_title(f"Bias Correction Comparison (N={n_val})", fontsize=26)
    ax.tick_params(axis="y", labelsize=18)

    legend_elements = [
        Patch(facecolor=c_raw, edgecolor="black", linewidth=0.4, label="No correction"),
        Patch(facecolor=c_free, edgecolor="black", linewidth=0.4, label="Distribution-free"),
        Patch(facecolor=c_aware, edgecolor="black", linewidth=0.4, label="Distribution-aware"),
        Patch(facecolor=c_slope, edgecolor="black", linewidth=0.4, hatch="//", label="Slope (no correction needed)"),
    ]
    ax.legend(handles=legend_elements, fontsize=16, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bias_correction_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    mpl.rcParams["hatch.color"] = old_hatch_color
    mpl.rcParams["hatch.linewidth"] = old_hatch_lw

    print("  Saved bias_correction_comparison.png")


# ---------------------------------------------------------------------------
# Plot 8: Phase 2 — fingerprint with uncertainty bands
# ---------------------------------------------------------------------------

def plot_phase2_example(p2, clean, output_dir):
    # Pick a model that exists in both clean and p2 data, preferring high R²
    available_models = set(zip(p2["model"], p2["quant"]))
    models = clean.sort_values("r_squared", ascending=False)
    pick = None
    for _, row in models.iterrows():
        if (row["model"], row["quant"]) in available_models:
            pick = row
            break
    if pick is None:
        print("  Skipping phase2_example.png (no matching models)")
        return
    model, quant = pick["model"], pick["quant"]

    sub = p2[(p2["model"] == model) & (p2["quant"] == quant)].copy()
    n_vals = sorted(sub["n_obs"].unique())
    if not n_vals:
        return
    n_mid = n_vals[len(n_vals) // 2]
    sub = sub[sub["n_obs"] == n_mid]
    dist = sorted(sub["prompt_dist"].unique())[0]
    sub = sub[sub["prompt_dist"] == dist]

    target_lambdas = [0.5, 2.0, 5.0, 10.0, 50.0, 100.0]
    all_lambdas = sorted(sub["lambda"].unique())
    lambdas = [l for l in target_lambdas if l in all_lambdas]
    if not lambdas:
        lambdas = all_lambdas[:5]

    fig, axes = plt.subplots(1, len(lambdas), figsize=(4 * len(lambdas), 5), sharey=True)
    if len(lambdas) == 1:
        axes = [axes]

    for ax, lam in zip(axes, lambdas):
        lam_sub = sub[sub["lambda"] == lam]
        agg = lam_sub.groupby("prompt_length").agg(
            clean=("clean_vram_mb", "first"),
            noisy_min_med=("noisy_min_mb", "median"),
            noisy_min_q10=("noisy_min_mb", lambda x: x.quantile(0.10)),
            noisy_min_q90=("noisy_min_mb", lambda x: x.quantile(0.90)),
            noisy_peak_med=("noisy_peak_mb", "median"),
            noisy_peak_q10=("noisy_peak_mb", lambda x: x.quantile(0.10)),
            noisy_peak_q90=("noisy_peak_mb", lambda x: x.quantile(0.90)),
        ).reset_index()

        ax.plot(agg["prompt_length"], agg["clean"], "o-", color="C0",
                linewidth=2.5, markersize=7, label="Clean (ground truth)", zorder=5)
        ax.plot(agg["prompt_length"], agg["noisy_min_med"], "s--", color="C1",
                linewidth=2, markersize=5, label="Noisy Min (median)")
        ax.fill_between(agg["prompt_length"], agg["noisy_min_q10"], agg["noisy_min_q90"],
                        alpha=0.2, color="C1", label="Noisy Min (10-90th pct)")
        ax.plot(agg["prompt_length"], agg["noisy_peak_med"], "^--", color="C3",
                linewidth=2, markersize=5, label="Noisy Peak (median)")
        ax.fill_between(agg["prompt_length"], agg["noisy_peak_q10"], agg["noisy_peak_q90"],
                        alpha=0.15, color="C3", label="Noisy Peak (10-90th pct)")

        ax.set_title(f"λ={lam:.0f}", fontsize=14)
        ax.set_xlabel("Prompt length (tokens)", fontsize=12)
        ax.tick_params(labelsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("VRAM (MB)", fontsize=14)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Fingerprint Recovery: {model} [{quant}] (N={n_mid}, dist={dist})",
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "phase2_example.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved phase2_example.png")


# ---------------------------------------------------------------------------
# Plot 9: 2D 1-NN accuracy heatmap
# ---------------------------------------------------------------------------

def _compute_nn_accuracy_2d(trials, clean):
    """1-NN in normalized (intercept, slope) space."""
    gallery_labels = []
    gallery_intercepts = []
    gallery_slopes = []
    for _, row in clean.iterrows():
        gallery_labels.append(f"{row['model']}[{row['quant']}]")
        gallery_intercepts.append(row["intercept_mb"])
        gallery_slopes.append(row["slope_mb_per_token"])
    gallery_intercepts = np.array(gallery_intercepts)
    gallery_slopes = np.array(gallery_slopes)

    # Normalize by std so both dimensions contribute equally
    int_std = gallery_intercepts.std() if gallery_intercepts.std() > 0 else 1.0
    slope_std = gallery_slopes.std() if gallery_slopes.std() > 0 else 1.0

    records = []
    for (lam, dist, n_obs), group in trials.groupby(["lambda", "prompt_dist", "n_obs"]):
        for method, int_col, slope_col in [
            ("raw", "recovered_intercept_mb", "recovered_slope"),
            ("dist_aware", "corrected_intercept_aware_mb", "recovered_slope"),
        ]:
            est_int = group[int_col].values
            est_slope = group[slope_col].values
            true_labels = (group["model"] + "[" + group["quant"] + "]").values

            # 2D distance (normalized)
            d_int = (est_int[:, None] - gallery_intercepts[None, :]) / int_std
            d_slope = (est_slope[:, None] - gallery_slopes[None, :]) / slope_std
            dists = np.sqrt(d_int**2 + d_slope**2)
            predicted_idx = dists.argmin(axis=1)
            predicted_labels = np.array(gallery_labels)[predicted_idx]
            accuracy = (predicted_labels == true_labels).mean() * 100

            records.append({
                "lambda": lam, "prompt_dist": dist, "n_obs": n_obs,
                "method": method, "accuracy": accuracy,
            })

    return pd.DataFrame(records)


def plot_nn_accuracy_heatmap(trials, clean, output_dir):
    acc_df = _compute_nn_accuracy_2d(trials, clean)

    fig, axes = plt.subplots(1, 2, figsize=(22, 8))
    for ax, method, title in [
        (axes[0], "raw", "Raw (intercept + slope) — 1-NN Accuracy (%)"),
        (axes[1], "dist_aware", "Distribution-Aware — 1-NN Accuracy (%)"),
    ]:
        sub = acc_df[acc_df["method"] == method]
        pivot = sub.groupby(["lambda", "n_obs"])["accuracy"].mean().unstack("n_obs")
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns.astype(int), fontsize=18)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"λ={v:g}" for v in pivot.index], fontsize=18)
        ax.set_xlabel("N (observations)", fontsize=22)
        ax.set_ylabel("Mean batch size (λ)", fontsize=22)
        ax.set_title(title, fontsize=24)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                color = "white" if val < 50 else "black"
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=15, color=color, fontweight="bold")
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=16)

    fig.suptitle("2D 1-NN Model Identification Accuracy (intercept + slope)",
                 fontsize=28, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "nn_accuracy_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved nn_accuracy_heatmap.png")


# ---------------------------------------------------------------------------
# Plot 10: 1-NN accuracy vs N (line plot)
# ---------------------------------------------------------------------------

def plot_nn_accuracy_lines(trials, clean, output_dir):
    acc_df = _compute_nn_accuracy_2d(trials, clean)
    sub = acc_df[acc_df["method"] == "dist_aware"]
    agg = sub.groupby(["lambda", "n_obs"])["accuracy"].agg(["mean", "min", "max"]).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("tab10", n_colors=agg["lambda"].nunique())
    for i, (lam, g) in enumerate(agg.groupby("lambda")):
        g = g.sort_values("n_obs")
        ax.plot(g["n_obs"], g["mean"], marker="o", linewidth=2, markersize=6,
                label=f"λ={lam:g}", color=palette[i])
        ax.fill_between(g["n_obs"], g["min"], g["max"], alpha=0.15, color=palette[i])

    ax.axhline(y=100 / len(clean), color="gray", linestyle=":", linewidth=1.5,
               label=f"Random guess ({100/len(clean):.1f}%)")
    ax.set_xlabel("N (observations per trial)", fontsize=14)
    ax.set_ylabel("1-NN Accuracy (%)", fontsize=14)
    ax.set_title("Model Identification Accuracy — 2D (intercept + slope)", fontsize=16)
    ax.set_xscale("log")
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=10, title="Mean batch size", title_fontsize=12, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "nn_accuracy_lines.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved nn_accuracy_lines.png")


# ---------------------------------------------------------------------------
# Plot 11: Confusion matrix
# ---------------------------------------------------------------------------

def plot_nn_confusion(trials, clean, output_dir):
    gallery_labels = []
    gallery_intercepts = []
    gallery_slopes = []
    for _, row in clean.iterrows():
        label = f"{row['model']}\n{row['quant']}"
        gallery_labels.append(label)
        gallery_intercepts.append(row["intercept_mb"])
        gallery_slopes.append(row["slope_mb_per_token"])
    gallery_intercepts = np.array(gallery_intercepts)
    gallery_slopes = np.array(gallery_slopes)
    int_std = gallery_intercepts.std() if gallery_intercepts.std() > 0 else 1.0
    slope_std = gallery_slopes.std() if gallery_slopes.std() > 0 else 1.0

    target_lam = 10.0
    available_lams = sorted(trials["lambda"].unique())
    if target_lam not in available_lams:
        target_lam = available_lams[-1]

    sub = trials[(trials["lambda"] == target_lam) & (trials["n_obs"] == 100)].copy()
    if sub.empty:
        return

    est_int = sub["corrected_intercept_aware_mb"].values
    est_slope = sub["recovered_slope"].values
    true_labels = (sub["model"] + "\n" + sub["quant"]).values

    d_int = (est_int[:, None] - gallery_intercepts[None, :]) / int_std
    d_slope = (est_slope[:, None] - gallery_slopes[None, :]) / slope_std
    dists = np.sqrt(d_int**2 + d_slope**2)
    predicted_idx = dists.argmin(axis=1)
    predicted_labels = np.array(gallery_labels)[predicted_idx]

    n_classes = len(gallery_labels)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    label_to_idx = {l: i for i, l in enumerate(gallery_labels)}
    for true, pred in zip(true_labels, predicted_labels):
        if true in label_to_idx and pred in label_to_idx:
            cm[label_to_idx[true], label_to_idx[pred]] += 1

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_pct = cm / row_sums * 100

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(gallery_labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(gallery_labels, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("True", fontsize=14)
    ax.set_title(f"2D 1-NN Confusion Matrix (λ={target_lam:.0f}, N=100, distribution-aware)", fontsize=16)
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm_pct[i, j]
            if val > 0.5:
                color = "white" if val > 50 else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=8, color=color)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Classification rate (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "nn_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved nn_confusion_matrix.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot simulated noisy experiment results")
    parser.add_argument("--input-dir", default=INPUT_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    clean, trials, p2 = load_data(args.input_dir)
    print(f"  trials: {len(trials)} rows, phase2: {len(p2)} rows\n")

    # Filter out λ=50 from all plots
    trials = trials[trials["lambda"] != 50.0]
    p2 = p2[p2["lambda"] != 50.0]

    print("Generating plots...")
    plot_intercept_mape_heatmap(trials, args.output_dir)
    plot_slope_mape_heatmap(trials, args.output_dir)
    plot_intercept_mape_vs_n(trials, args.output_dir)
    plot_intercept_mape_by_dist(trials, args.output_dir)
    plot_slope_mape_by_dist(trials, args.output_dir)
    plot_bias_correction_comparison(trials, args.output_dir)
    plot_phase2_example(p2, clean, args.output_dir)

    print("\nComputing 2D 1-NN classification...")
    plot_nn_accuracy_heatmap(trials, clean, args.output_dir)
    plot_nn_accuracy_lines(trials, clean, args.output_dir)
    plot_nn_confusion(trials, clean, args.output_dir)

    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
