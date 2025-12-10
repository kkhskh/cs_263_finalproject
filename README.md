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
│   ├── model_backend.py    # Models + TimingObfuscator
│   ├── covert_channel.py   # Covert channel implementation
│   └── Dockerfile
├── attacker/                # Attack implementations
│   ├── flush_reload.c      # Complete FLUSH+RELOAD with all modes
│   ├── Makefile            # Release/debug/profile builds
│   └── Dockerfile
├── experiments/             # Experiment automation
│   ├── traffic_gen.py      # Traffic generator
│   ├── analyze_stats.py    # Statistical analysis
│   └── run_experiment.py   # Full experiment orchestration
├── infra/                   # Infrastructure
│   ├── docker-compose.yml  # All service variants
│   ├── docker-compose.gvisor.yml
│   └── setup_gvisor.sh
├── mitigations/             # Mitigation evaluation
│   └── evaluate_mitigation.py
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

