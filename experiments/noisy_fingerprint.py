"""
Noisy Fingerprinting Experiment
================================
Two-phase experiment that tests whether a min-based estimator can recover
a model's VRAM fingerprint (intercept + slope) in the presence of
background traffic that creates KV-cache noise.

Phase 1 (passive): Observe VRAM for 30s with only background traffic.
         min(samples) estimates the model weight footprint (intercept).

Phase 2 (active):  Send known-length prompts while background traffic
         continues.  Record min and peak VRAM per prompt length.

Usage:
    python -m experiments.noisy_fingerprint --rate 2.0 --seeds 1
    python -m experiments.noisy_fingerprint --rates 0.5 1.0 2.0 5.0 --seeds 3
"""

import argparse
import os
import random
import statistics
import threading
import time

import requests
import torch

from attacker.gpu_fingerprint import MARGINAL_PROMPT_LENGTHS, make_prompt
from experiments.prompt_and_probe import (
    HF_MODELS,
    VICTIM_PORT,
    VICTIM_URL,
    start_server,
    stop_server,
    wait_for_server_ready,
)

RESULTS_DIR = "results/noisy_fingerprints"
PHASE1_DURATION_S = 30
PROBE_INTERVAL_S = 0.001  # 1 ms


# ---------------------------------------------------------------------------
# Background traffic generator
# ---------------------------------------------------------------------------

class BackgroundTraffic:
    """Sends random prompts at a Poisson rate in a background thread."""

    def __init__(self, rate_hz: float, model_name: str):
        self.rate_hz = rate_hz
        self.model_name = model_name
        self._stop = threading.Event()
        self._thread = None
        self._request_count = 0

    def start(self):
        self._stop.clear()
        self._request_count = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)

    @property
    def request_count(self):
        return self._request_count

    def _run(self):
        while not self._stop.is_set():
            # Poisson inter-arrival
            delay = random.expovariate(self.rate_hz)
            if self._stop.wait(timeout=delay):
                break

            # Random prompt length uniform [100, 2000] tokens
            length = random.randint(100, 2000)
            prompt = make_prompt(length, self.model_name, seed=random.randint(0, 2**31))

            try:
                requests.post(VICTIM_URL, json={"prompt": prompt}, timeout=120)
            except Exception:
                pass
            self._request_count += 1


# ---------------------------------------------------------------------------
# Phase 1: Passive VRAM observation
# ---------------------------------------------------------------------------

def phase1_passive_probe(duration_s: float = PHASE1_DURATION_S):
    """Probe VRAM for *duration_s* seconds without sending any requests.

    Returns (timeseries, summary) where
        timeseries = list of (timestamp_ms, used_mb)
        summary = dict with n_samples, min, median, mean, std
    """
    samples = []
    start = time.time()

    while time.time() - start < duration_s:
        free, total = torch.cuda.mem_get_info()
        used_mb = (total - free) / (1024 ** 2)
        ts_ms = (time.time() - start) * 1000.0
        samples.append((ts_ms, used_mb))
        time.sleep(PROBE_INTERVAL_S)

    used_values = [s[1] for s in samples]
    summary = {
        "n_samples": len(used_values),
        "min_vram_mb": min(used_values),
        "median_vram_mb": statistics.median(used_values),
        "mean_vram_mb": statistics.mean(used_values),
        "std_vram_mb": statistics.pstdev(used_values),
    }

    return samples, summary


# ---------------------------------------------------------------------------
# Phase 2: Active probing (estimate slope)
# ---------------------------------------------------------------------------

def phase2_active_probe(model_name: str, seed: int):
    """Send prompts at each length in MARGINAL_PROMPT_LENGTHS and record
    min / peak VRAM during each request.

    VRAM accumulates across prompt lengths (no cache clearing).

    Returns list of (cumulative_prompt_length, min_vram_mb, peak_vram_mb).
    """
    results = []
    cumulative = 0

    for L in MARGINAL_PROMPT_LENGTHS:
        cumulative += L
        prompt = make_prompt(L, model_name, seed=seed)

        # Probe VRAM while the request is in flight
        min_used = float("inf")
        peak_used = 0.0
        done = False

        def send_request():
            nonlocal done
            try:
                requests.post(VICTIM_URL, json={"prompt": prompt}, timeout=300)
            except Exception as e:
                print(f"[phase2] Request error at L={L}: {e}")
            finally:
                done = True

        t = threading.Thread(target=send_request)
        t.start()

        while not done:
            free, total = torch.cuda.mem_get_info()
            used = (total - free) / (1024 ** 2)
            min_used = min(min_used, used)
            peak_used = max(peak_used, used)
            time.sleep(PROBE_INTERVAL_S)

        t.join()

        results.append((cumulative, min_used, peak_used))
        print(f"  [phase2] cumL={cumulative}  min={min_used:.2f}  peak={peak_used:.2f}")
        time.sleep(1)

    return results


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _make_tag(model_name: str, quant_mode: str, rate: float, seed: int) -> str:
    safe = model_name.split("/")[-1]
    return f"{safe}[{quant_mode}][rate={rate}][seed={seed}]"


def save_phase1(tag: str, timeseries, summary):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ts_path = os.path.join(RESULTS_DIR, f"{tag}_phase1.csv")
    with open(ts_path, "w") as f:
        f.write("timestamp_ms,used_mb\n")
        for ts_ms, used_mb in timeseries:
            f.write(f"{ts_ms:.2f},{used_mb:.2f}\n")

    sum_path = os.path.join(RESULTS_DIR, f"{tag}_phase1_summary.csv")
    with open(sum_path, "w") as f:
        f.write(",".join(summary.keys()) + "\n")
        f.write(",".join(f"{v:.2f}" if isinstance(v, float) else str(v)
                         for v in summary.values()) + "\n")

    print(f"  Phase 1 saved: {ts_path}  (min={summary['min_vram_mb']:.2f} MB)")


def save_phase2(tag: str, rows):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    path = os.path.join(RESULTS_DIR, f"{tag}_phase2.csv")
    with open(path, "w") as f:
        f.write("cumulative_prompt_length,min_vram_mb,peak_vram_mb\n")
        for cum, mn, pk in rows:
            f.write(f"{cum},{mn:.2f},{pk:.2f}\n")

    print(f"  Phase 2 saved: {path}")


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(model_name: str, quant_mode: str, rate: float, seed: int):
    """Run one (model, quant, rate, seed) trial."""
    tag = _make_tag(model_name, quant_mode, rate, seed)
    print(f"\n--- Trial: {tag} ---")

    # Start background traffic
    traffic = BackgroundTraffic(rate_hz=rate, model_name=model_name)
    traffic.start()
    print(f"  Background traffic started at {rate} req/s, waiting 5s to establish...")
    time.sleep(5)

    # Phase 1: passive
    print(f"  Phase 1: passive observation for {PHASE1_DURATION_S}s ...")
    timeseries, summary = phase1_passive_probe(PHASE1_DURATION_S)
    save_phase1(tag, timeseries, summary)

    # Phase 2: active probing
    print("  Phase 2: active probing ...")
    phase2_rows = phase2_active_probe(model_name, seed)
    save_phase2(tag, phase2_rows)

    # Stop background traffic
    traffic.stop()
    print(f"  Background traffic stopped (sent {traffic.request_count} requests)")


def main():
    parser = argparse.ArgumentParser(
        description="Noisy fingerprinting experiment (two-phase)"
    )
    parser.add_argument(
        "--rate", type=float, default=None,
        help="Single background traffic rate in req/s (e.g. 2.0)",
    )
    parser.add_argument(
        "--rates", type=float, nargs="+", default=None,
        help="Multiple background traffic rates to sweep (e.g. 0.5 1.0 2.0 5.0)",
    )
    parser.add_argument(
        "--seeds", type=int, default=3,
        help="Number of seeds/repetitions per configuration (default: 3)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Run only this model (e.g. meta-llama/Llama-3.2-1B)",
    )
    args = parser.parse_args()

    # Determine rates to use
    if args.rates is not None:
        rates = args.rates
    elif args.rate is not None:
        rates = [args.rate]
    else:
        rates = [2.0]

    seeds = list(range(args.seeds))

    models = HF_MODELS
    if args.model:
        if args.model not in HF_MODELS:
            print(f"Unknown model: {args.model}")
            print(f"Available: {list(HF_MODELS.keys())}")
            return
        models = {args.model: HF_MODELS[args.model]}

    for model_name, quant_modes in models.items():
        for quant_mode in quant_modes:
            proc = start_server(model_name, quant_mode)
            wait_for_server_ready()

            for rate in rates:
                for seed in seeds:
                    run_experiment(model_name, quant_mode, rate, seed)

            stop_server(proc)

    print("\n=== All noisy fingerprinting experiments complete! ===")


if __name__ == "__main__":
    main()
