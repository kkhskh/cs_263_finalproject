"""Reproduce the two top-level plots from prompt_and_probe_plots.ipynb.

Reads fingerprint CSVs from results/fingerprints/, fits the constrained
linear model per family, and saves two figures:

  1. Mistral case study: v0.1 vs Instruct-v0.2 (same intercept, different slope).
  2. Fingerprint scatter: intercept (base VRAM) vs slope (per-token KV growth)
     across all 11 model/quant families.

Run:
    python -m experiments.plot_clean_fingerprints \
        --fingerprints-dir results/fingerprints \
        --out-dir figures
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


def load_fingerprints(base: Path) -> pd.DataFrame:
    rows = []
    for fpath in base.glob("*.csv"):
        model_id = fpath.stem
        df = pd.read_csv(fpath)
        df["model_id"] = model_id
        df["seed"] = (
            int(model_id.split("seed=")[-1].rstrip("]"))
            if "seed=" in model_id else -1
        )
        df["family"] = re.sub(r"\[seed=\d+\]", "", model_id)
        rows.append(df)
    if not rows:
        raise SystemExit(f"No fingerprint CSVs in {base}")
    return pd.concat(rows, ignore_index=True)


def fit_constrained_linear(X: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    x0, y0 = X[0], y[0]
    X_shift = X - x0
    y_shift = y - y0
    a = np.sum(X_shift * y_shift) / np.sum(X_shift ** 2)
    b = y0 - a * x0
    y_pred = a * X + b
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    return float(a), float(b), float(r2), float(mape)


def fit_per_family(df: pd.DataFrame, reducer: str = "mean") -> dict[str, dict[str, float]]:
    """Fit constrained linear model per family. Reducer aggregates seeds at each
    prompt length before fitting: 'mean' (default) or 'min'."""
    if reducer not in ("mean", "min"):
        raise ValueError(f"reducer must be 'mean' or 'min', got {reducer!r}")
    fits = {}
    for family, sub in df.groupby("family"):
        agg = sub.groupby("prompt_length", as_index=False)["peak_vram_mb"].agg(reducer)
        X = agg["prompt_length"].values
        y = agg["peak_vram_mb"].values
        a, b, r2, mape = fit_constrained_linear(X, y)
        fits[family] = dict(a=a, b=b, r2=r2, mape=mape)
    return fits


def plot_mistral_case_study(df: pd.DataFrame, fits: dict, out_path: Path):
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "serif"

    mistral_df = df[df["model_id"].str.contains("Mistral", case=False)].copy()

    families = {"v0.1": "Mistral-7B-v0.1", "v0.2": "Mistral-7B-Instruct-v0.2"}
    palette = sns.color_palette("tab10", n_colors=2)
    family_colors = {"v0.1": palette[0], "v0.2": palette[1]}

    fig, ax = plt.subplots(figsize=(20, 12))
    legend_scatter, legend_lines = {}, {}

    for fam_key, fam_base in families.items():
        color = family_colors[fam_key]
        fam_runs = mistral_df[mistral_df["model_id"].str.contains(fam_base, regex=False)]
        for _, grp in fam_runs.groupby("model_id"):
            ax.scatter(grp["prompt_length"], grp["peak_vram_mb"],
                       s=35, alpha=0.5, edgecolor="black", linewidth=0.3, color=color)

        matching = [v for k, v in fits.items() if fam_base in k]
        a_mean = np.mean([v["a"] for v in matching])
        b_mean = np.mean([v["b"] for v in matching])
        xs = np.linspace(fam_runs["prompt_length"].min(),
                         fam_runs["prompt_length"].max(), 300)
        ax.plot(xs, a_mean * xs + b_mean, color=color, linewidth=4)

        legend_scatter[fam_key] = Line2D([], [], marker='o', linestyle='None',
                                         markersize=10, color=color, markeredgecolor="black")
        legend_lines[fam_key] = Line2D([], [], color=color, linewidth=4)

    handles = [legend_scatter["v0.1"], legend_lines["v0.1"],
               legend_scatter["v0.2"], legend_lines["v0.2"]]
    labels = ["Mistral-7B-v0.1 (data)", "Mistral-7B-v0.1 (linear model)",
              "Mistral-7B-v0.2 (data)", "Mistral-7B-v0.2 (linear model)"]
    ax.legend(handles=handles, labels=labels, fontsize=30,
              title="Models", title_fontsize=32, frameon=True)
    ax.set_xlabel("Prompt Length (tokens)", fontsize=38)
    ax.set_ylabel("Peak VRAM (MB)", fontsize=38)
    ax.set_title("VRAM Scaling vs Prompt Length: Mistral v0.1 vs v0.2", fontsize=44)
    ax.tick_params(axis="both", labelsize=30)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_r2_bars(fits: dict, out_path: Path):
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "serif"

    items = sorted(fits.items(), key=lambda kv: kv[1]["r2"])
    families = [f for f, _ in items]
    r2_vals = [v["r2"] for _, v in items]

    fig, ax = plt.subplots(figsize=(16, 10))
    palette = sns.color_palette("viridis", n_colors=len(families))
    bars = ax.barh(families, r2_vals, color=palette, edgecolor="black")

    for bar, r2 in zip(bars, r2_vals):
        ax.text(r2 + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{r2:.3f}", va="center", fontsize=18)

    ax.set_xlim(0.0, 1.05)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(r"$R^2$ of constrained linear fit", fontsize=28)
    ax.set_title("Linear fit quality per model/quant family", fontsize=32)
    ax.tick_params(axis="x", labelsize=22)
    ax.tick_params(axis="y", labelsize=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_fingerprint_scatter(fits: dict, out_path: Path):
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "serif"

    families = sorted(fits.keys())
    intercepts = [fits[f]["b"] for f in families]
    slopes = [fits[f]["a"] for f in families]

    fig, ax = plt.subplots(figsize=(18, 12))
    sns.scatterplot(x=intercepts, y=slopes, s=250, color="C0",
                    edgecolor="black", alpha=0.8, ax=ax)

    # Per-family label placement to avoid overlap.
    # value: (dx_pts, dy_pts, valign)
    label_offsets = {
        "Llama-3.2-1B[q-8bit]":              (-50, -28, "top"),
        "Llama-3.1-8B[q-8bit]":              (-50, -28, "top"),
        "Qwen2-7B-Instruct[q-8bit]":         (-50, -28, "top"),
        "Mistral-7B-Instruct-v0.2[q-8bit]":  (-50, -28, "top"),
        "Llama-3.2-3B[q-8bit]":              (-50, -28, "top"),
        "gemma-2b[fp16]":                    ( 60,  16, "bottom"),
        "gemma-7b[q-8bit]":                  (-50,  32, "bottom"),
    }
    default_offset = (-50, 16, "bottom")
    for family, b, a in zip(families, intercepts, slopes):
        dx, dy, valign = label_offsets.get(family, default_offset)
        ax.annotate(family, xy=(b, a), xytext=(dx, dy),
                    textcoords="offset points", ha="center",
                    va=valign, fontsize=26)

    ax.set_xlim(left=0, right=10000)
    ax.set_xlabel("Base VRAM Allocated [MB]", fontsize=30)
    ax.set_ylabel("Marginal VRAM Allocated [MB/Token]", fontsize=30)
    ax.set_title("LLM VRAM Fingerprint", fontsize=36)
    ax.tick_params(axis="both", labelsize=24)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fingerprints-dir", default="results/fingerprints")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--max-prompt-length", type=int, default=3000,
                        help="Drop rows beyond this cumulative length (default 3000)")
    parser.add_argument("--reducer", choices=("mean", "min"), default="min",
                        help="Aggregator across seeds at each prompt length (default: min)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_fingerprints(Path(args.fingerprints_dir))
    df = df[df["prompt_length"] <= args.max_prompt_length]

    fits = fit_per_family(df, reducer=args.reducer)
    print(f"\nFitted {len(fits)} families ({args.reducer}-reduced across seeds)")
    for family, vals in sorted(fits.items()):
        print(f"  {family:<35}  a={vals['a']:.6f}  b={vals['b']:.2f}  R²={vals['r2']:.4f}")

    plot_mistral_case_study(df, fits, out_dir / "linear_model_mistral.png")
    plot_fingerprint_scatter(fits, out_dir / "fingerprint_scatter.png")
    plot_r2_bars(fits, out_dir / "r2_fits.png")


if __name__ == "__main__":
    main()
