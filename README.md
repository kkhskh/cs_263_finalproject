# LLM Side-Channel Attack Research Framework (Final)

A comprehensive framework for investigating microarchitectural side-channel attacks against multi-tenant LLM inference services.

## Features

- **Model Fingerprinting**: Identify models via timing analysis (100% accuracy demonstrated)
- **Covert Channels**: Leak sensitive query information through timing
- **FLUSH+RELOAD**: Cache-based side-channel attack implementation
- **Timing Obfuscation**: Mitigation strategies (random delay, bucketing, constant-time)
- **gVisor Support**: Test stronger isolation
- **Statistical Analysis**: Comprehensive statistics with confidence intervals

## Project Structure

```
llm_sidechannel_final/
├── victim_service/          # Target LLM service
│   ├── server.py           # FastAPI with obfuscation support
│   ├── pp_server.py        # Prompt and probe victim server
│   ├── model_backend.py    # Models + TimingObfuscator
│   ├── hf_model_backend.py # HuggingFace model backend
│   ├── covert_channel.py   # Covert channel implementation
│   └── Dockerfile
├── attacker/                # Attack implementations
│   ├── flush_reload.c      # Complete FLUSH+RELOAD with all modes
│   ├── gpu_fingerprint.py  # GPU fingerprinting via prompt and probe
│   ├── Makefile            # Release/debug/profile builds
│   └── Dockerfile
├── experiments/             # Experiment automation
│   ├── traffic_gen.py      # Traffic generator
│   ├── analyze_stats.py    # Statistical analysis
│   ├── run_experiment.py   # Full experiment orchestration
│   └── prompt_and_probe.py # Prompt and probe orchestrator
├── infra/                   # Infrastructure
│   ├── docker-compose.yml  # All service variants
│   ├── docker-compose.gvisor.yml
│   └── setup_gvisor.sh
├── mitigations/             # Mitigation evaluation
│   └── evaluate_mitigation.py
├── prompt_and_probe_plots.ipynb  # Jupyter notebook for visualizing results
└── README.md
```

## Quick Start

### 1. Build Images

```bash
cd victim_service && docker build -t llm-victim .
cd ../attacker && docker build -t llm-attacker .
```

### 2. Run Baseline Fingerprinting

```bash
# Start victim
docker run -d --rm --name victim -p 8000:8000 \
    -e MODEL_NAME=fake_a -e USE_REAL_MODELS=0 llm-victim

# Run experiment
cd experiments
python3 traffic_gen.py --mode fingerprint --model-tag fake_a --n 100

# Repeat for other models, then analyze
python3 analyze_stats.py --files traffic_*.csv --plot results.png
```

### 3. Test Timing Obfuscation (Mitigation)

```bash
# With random delay (50ms max)
docker run -d --rm --name victim -p 8000:8000 \
    -e MODEL_NAME=fake_a \
    -e OBFUSCATION_STRATEGY=random \
    -e OBFUSCATION_PARAM=50 \
    llm-victim

# With bucket rounding (100ms buckets)
docker run -d --rm --name victim -p 8000:8000 \
    -e MODEL_NAME=fake_a \
    -e OBFUSCATION_STRATEGY=bucket \
    -e OBFUSCATION_PARAM=100 \
    llm-victim
```

### 4. Test gVisor Isolation

```bash
# Install gVisor (as root)
sudo ./infra/setup_gvisor.sh

# Run with gVisor
docker run -d --rm --runtime=runsc --name victim -p 8000:8000 \
    -e MODEL_NAME=fake_a llm-victim
```

### 5. Run Full Experiment Matrix

```bash
cd experiments

# Run with 3 repetitions, multiple models, multiple obfuscations
python3 run_experiment.py \
    --models fake_a fake_b distilgpt2 \
    --requests 100 \
    --repetitions 3 \
    --obfuscations none random_50 bucket_100 \
    --output-dir ./full_results \
    --plot full_results.png


## Attacker CLI

bash
./flush_reload [iterations] [threshold] [mode] [target_lib] [offset]

Modes:
  0 = CSV output (iter,cycles,hit)
  1 = Statistics only
  2 = Calibration (find threshold)
  3 = Realtime monitoring

Examples:
  ./flush_reload 10000 0 2           # Calibrate
  ./flush_reload 100000 150 0        # CSV output
  ./flush_reload 100000 0 1          # Stats with 


### Victim Service

| Variable | Default | Description |
|----------|---------|-------------|
| MODEL_NAME | fake_a | Model to serve |
| USE_REAL_MODELS | 0 | Enable HuggingFace models |
| COVERT_ENABLED | 0 | Enable covert channel |
| OBFUSCATION_STRATEGY | none | none/random/bucket/constant |
| OBFUSCATION_PARAM | 0 | Strategy parameter (ms) |

### Attacker

| Variable | Default | Description |
|----------|---------|-------------|
| ITERATIONS | 100000 | Probe iterations |
| THRESHOLD | 0 | Hit/miss threshold (0=auto) |
| MODE | 0 | Output mode |
| PIN_CPU | "" | CPU core to pin |

## Results Summary

| Model | Mean Latency | Std Dev | Distinguishable |
|-------|-------------|---------|-----------------|
| fake_a | 49.4ms | 0.14ms | yes |
| fake_b | 59.7ms | 0.09ms | yes |
| distilgpt2 | 564.5ms | 210.8ms | yes |
| opt-125m | 1003.4ms | 16.9ms | yes |
| gpt2-medium | 2652.7ms | 55.5ms | yes |
```


## Prompt and Probe

The Prompt & Probe attack identifies a co-located LLM by measuring how its peak GPU VRAM grows with prompt length. The victim runs an inference server in one container; the attacker, in a separate container sharing the same GPU, sends prompts at increasing lengths and reads device-wide memory via `cudaMemGetInfo()`. The recovered (intercept, slope) pair forms a model-specific fingerprint.

### Threat-model setup

Victim and attacker run as **two separate Docker containers** sharing the GPU through the NVIDIA Container Toolkit. They share no filesystem, no Python process, and no network namespace beyond the bridge that lets the attacker reach the victim's HTTP endpoint. Both bind-mount a read-only PyTorch venv from the host so the images stay small.

Compose file: `infra/docker-compose.yml`. Containers: `pp-victim` and `pp-attacker`.

### Running the experiment

Prerequisites: Docker with the NVIDIA runtime, an HF access token in `~/.cache/huggingface/token` (gated models), and a host-side PyTorch venv at `/opt/pytorch` (or update the bind-mount path in `infra/docker-compose.yml`).

```bash
# Build the two images (one-time)
docker compose -f infra/docker-compose.yml build

# Run the full sweep: 11 models × 5 seeds, sequential prompt lengths [2, 1000, 2000, 3000]
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
python -m experiments.prompt_and_probe --full --seeds 5 --skip-build
```

The orchestrator starts a fresh `pp-victim` container per (model, quant, seed), waits for the model to load, runs the attacker against it, and tears down. Each attacker run records four observations of peak VRAM at cumulative prompt lengths `[2, 1000, 2000, 3000]` tokens and writes a single CSV to `results/fingerprints/{model}[{quant}][seed={seed}].csv`. Per-step time-series traces go to `results/timeseries/`.

Useful flags:
- `--seeds N` — repetitions per (model, quant); default 1.
- `--full` — full 11-config matrix; otherwise just two small models for smoke-testing.
- `--cumulative-lengths "1000,2000"` + `--append` — add extra prompt-length steps to existing CSVs without rerunning the full sweep.

### Plotting the results

```bash
python -m experiments.plot_clean_fingerprints
```

Reads every CSV in `results/fingerprints/`, fits the constrained linear model `VRAM(L) = a·L + b` per (model, quant) by reducing across seeds with the **min** statistic (default; `--reducer mean` available), and writes three figures to `figures/`:

- `linear_model_mistral.png` — Mistral v0.1 vs Instruct-v0.2 case study showing identical intercepts but distinct slopes.
- `fingerprint_scatter.png` — (intercept, slope) scatter across all 11 model/quant combinations.
- `r2_fits.png` — per-family R² of the linear fit, sorted.

### Key files

- `experiments/prompt_and_probe.py` — orchestrator (drives `docker compose` for each config).
- `experiments/plot_clean_fingerprints.py` — plotting script.
- `attacker/run_attack.py` — attacker entry point (probes VRAM during HTTP request).
- `attacker/gpu_fingerprint.py` — core VRAM-probe loop and prompt construction.
- `victim_service/pp_server.py` — victim inference server (FastAPI + Transformers).
- `victim_service/hf_model_backend.py` — HuggingFace model loading (fp16 / bitsandbytes 8-bit).
- `infra/docker-compose.yml` — victim + attacker container pair.

## Robustness to batching noise (simulation)

Once the clean fingerprints are collected, a separate Monte Carlo experiment evaluates whether the attacker can recover them under realistic continuous-batching noise. Co-resident traffic is modeled as `B ~ Poisson(λ)` requests, each contributing `slope · L` to the noise where `L ~ Zipf(θ)` truncated at the operator's context window. **No GPU is required** — all noise is generated analytically.

### Running the simulation

```bash
python -m experiments.simulated_noisy
```

Reads `results/fingerprints/` for the clean (intercept, slope) per family, sweeps over:

- `λ ∈ {0.5, 1, 2, 5, 10, 20, 100}` (mean co-resident batch size)
- `Zipf θ ∈ {1.5, 1.1, 0.7}` (BurstGPT default 1.1, plus low/high sensitivity)
- `N ∈ {10, 50, 100, 500, 1000}` (attacker observations per probe)

For each configuration, runs 1000 independent trials with 200 bootstrap draws for the distribution-aware bias correction. Outputs go to `results/simulated_noisy/`:

- `clean_fits.csv` — clean (intercept, slope) per family.
- `trial_summary.csv` — one row per trial: recovered intercept/slope, raw MAPE, distribution-aware corrected MAPE, distribution-free corrected MAPE.
- `phase2_summary.csv` — per-step noisy VRAM observations for plotting.

Useful flags: `--n-trials`, `--n-bootstrap`, `--lambdas`, `--prompt-dists`, `--n-obs`, `--validate-dists`.

### Plotting the simulation results

```bash
python -m experiments.plot_simulated_noisy
```

Produces a set of figures in `figures/simulated_noisy/`. The four most important ones for the paper:

- `bias_correction_comparison.png` — raw vs. distribution-free vs. distribution-aware corrections at fixed N.
- `intercept_mape_vs_n.png` — median intercept MAPE vs. observation count, faceted by λ.
- `intercept_mape_by_dist.png` — mean intercept MAPE per Zipf parameter at each λ.
- `nn_accuracy_heatmap.png` — 2D 1-NN model identification accuracy vs. (λ, N), raw vs. distribution-aware.

### Key files

- `experiments/simulated_noisy.py` — Monte Carlo simulation (CPU-only).
- `experiments/plot_simulated_noisy.py` — plotting script for all simulation figures.

