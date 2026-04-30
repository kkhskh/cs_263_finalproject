# Prompt & Probe

Prompt & Probe is a GPU side-channel attack that fingerprints a co-located large language model by measuring how its peak GPU VRAM grows with prompt length. The victim runs an inference server in one container; the attacker, in a separate container sharing the same GPU, sends prompts at increasing lengths and reads device-wide memory via `cudaMemGetInfo()`. The recovered (intercept, slope) pair forms a model-specific fingerprint.

This repository contains:

1. **The clean GPU experiment** (`experiments/prompt_and_probe.py`) — sweeps 11 model/quant combinations on a real GPU and records peak VRAM at four cumulative prompt lengths.
2. **A simulated-noise extension** (`experiments/simulated_noisy.py`) — Monte Carlo simulation that adds analytical batching noise on top of the clean fingerprints and tests whether the attacker can still recover them under realistic co-resident traffic.

## Threat model

Victim and attacker run as **two separate Docker containers** sharing the same GPU through the NVIDIA Container Toolkit. They share no filesystem, no Python process, and no network namespace beyond the bridge that lets the attacker reach the victim's HTTP endpoint. The host PyTorch venv is bind-mounted read-only so the images stay small.

Compose file: `infra/docker-compose.yml`. Containers: `pp-victim` and `pp-attacker`.

## 1. Clean GPU fingerprinting experiment

### Prerequisites

- Docker with the NVIDIA Container Toolkit.
- A host-side PyTorch venv at `/opt/pytorch` (or update the bind-mount path in `infra/docker-compose.yml`).
- A HuggingFace token at `~/.cache/huggingface/token` for gated Llama / Mistral models.
- A GPU with ≥16 GB VRAM (we used a single NVIDIA T4).

### Running the sweep

```bash
# One-time: build the two images
docker compose -f infra/docker-compose.yml build

# Full sweep: 11 models × 5 seeds, sequential prompt lengths [2, 1000, 2000, 3000]
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
python -m experiments.prompt_and_probe --full --seeds 5 --skip-build
```

The orchestrator starts a fresh `pp-victim` container per (model, quant, seed), waits for the model to load, runs the attacker against it, and tears down. Each attacker run records four observations of peak VRAM at cumulative prompt lengths `[2, 1000, 2000, 3000]` tokens and writes a single CSV to `results/fingerprints/{model}[{quant}][seed={seed}].csv`. Per-step VRAM time-series traces go to `results/timeseries/`.

Useful flags:

- `--seeds N` — repetitions per (model, quant); default 1.
- `--full` — full 11-config matrix; otherwise just two small models for smoke-testing.
- `--cumulative-lengths "1000,2000"` + `--append` — add extra prompt-length steps to existing CSVs without rerunning the full sweep.

### Plotting the clean fingerprints

```bash
python -m experiments.plot_clean_fingerprints
```

Reads every CSV in `results/fingerprints/`, reduces across seeds with the **min** statistic (default; `--reducer mean` available), fits the constrained linear model `VRAM(L) = a·L + b` per (model, quant), and writes three figures to `figures/`:

- `linear_model_mistral.png` — Mistral v0.1 vs Instruct-v0.2 case study showing identical intercepts but distinct slopes (a clean architectural-fingerprint demonstration).
- `fingerprint_scatter.png` — (intercept, slope) scatter across all 11 model/quant combinations.
- `r2_fits.png` — per-family R² of the constrained linear fit, sorted.

### Key files

- `experiments/prompt_and_probe.py` — orchestrator that drives `docker compose` for each config.
- `experiments/plot_clean_fingerprints.py` — plotting script for the three clean-fingerprint figures.
- `attacker/run_attack.py` — attacker entry point (probes VRAM during the HTTP request).
- `attacker/gpu_fingerprint.py` — core VRAM-probe loop and prompt construction.
- `victim_service/pp_server.py` — victim inference server (FastAPI + Transformers).
- `victim_service/hf_model_backend.py` — HuggingFace model loading (fp16 / bitsandbytes 8-bit).
- `infra/docker-compose.yml` — victim + attacker container pair.

## 2. Robustness to batching noise (simulation)

Once the clean fingerprints are collected, a separate Monte Carlo experiment evaluates whether the attacker can still recover them under realistic continuous-batching noise. The attacker's observation at prompt-length step `t` is modeled as

```
V_obs(t)  =  A + B · t           (clean fingerprint)
          +  B · Σᵢ Lᵢ           (additive noise from co-resident traffic)
```

where the number of co-resident requests `B ~ Poisson(λ)` and each `Lᵢ` is drawn from a truncated Zipf distribution (the prompt-length form fitted by Wang et al. 2025 / BurstGPT). **No GPU is required** — all noise is generated analytically on top of the previously-measured clean fingerprints.

### Running the simulation

```bash
python -m experiments.simulated_noisy
```

Reads `results/fingerprints/` for the clean (intercept, slope) per family, sweeps over:

- `λ ∈ {0.5, 1, 2, 5, 10, 20, 100}` (mean co-resident batch size)
- `Zipf θ ∈ {1.5, 1.1, 0.7}` (BurstGPT default 1.1, plus low/high sensitivity)
- `N ∈ {10, 50, 100, 500, 1000}` (attacker observations per probe length)

For each configuration, runs 100 independent trials with 200 Monte Carlo bootstrap draws for the distribution-aware bias correction. Outputs go to `results/simulated_noisy/`:

- `clean_fits.csv` — clean (intercept, slope) per family.
- `trial_summary.csv` — one row per trial: recovered intercept/slope, raw MAPE, distribution-aware corrected MAPE, distribution-free corrected MAPE.
- `phase2_summary.csv` — per-step noisy VRAM observations.

Useful flags: `--n-trials`, `--n-bootstrap`, `--lambdas`, `--prompt-dists`, `--n-obs`, `--validate-dists`.

### Plotting the simulation results

```bash
python -m experiments.plot_simulated_noisy
```

Writes a set of figures to `figures/simulated_noisy/`. The four headline figures for the paper:

- `bias_correction_comparison.png` — raw vs. distribution-free vs. distribution-aware bias corrections at fixed N.
- `intercept_mape_vs_n.png` — median intercept MAPE vs. observation count, faceted by λ.
- `intercept_mape_by_dist.png` — mean intercept MAPE per Zipf parameter at each λ.
- `nn_accuracy_heatmap.png` — 2D 1-NN model identification accuracy vs. (λ, N), raw vs. distribution-aware.

### Key files

- `experiments/simulated_noisy.py` — CPU-only Monte Carlo simulation.
- `experiments/plot_simulated_noisy.py` — plotting script for all simulation figures.
