#!/usr/bin/env python3
"""Containerized GPU prompt-and-probe attack.

Runs against a victim service over the docker network, writes the fingerprint
CSV and the per-step VRAM time series to /output (bind-mounted from host).
"""
import argparse
import os
import time

import requests

import gpu_fingerprint
from gpu_fingerprint import MARGINAL_PROMPT_LENGTHS, make_prompt, prompt_and_probe


def wait_for_victim(url: str, timeout: int = 1800):
    print(f"[attacker] waiting for victim at {url}", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.post(url, json={"prompt": "warmup"}, timeout=10)
            if r.status_code == 200:
                print(f"[attacker] victim ready after {time.time() - start:.1f}s", flush=True)
                return
        except requests.RequestException:
            pass
        time.sleep(3)
    raise RuntimeError(f"victim at {url} not ready after {timeout}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--quant", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--victim-url", default=os.getenv("VICTIM_URL", "http://victim:8000/generate"))
    p.add_argument("--output-dir", default="/output")
    p.add_argument("--cumulative-lengths", default=os.getenv("CUMULATIVE_LENGTHS", ""),
                   help="comma-separated cumulative lengths; if set, overrides the default sweep")
    p.add_argument("--append", action="store_true",
                   default=bool(os.getenv("ATTACKER_APPEND")),
                   help="append rows to existing CSV instead of overwriting")
    args = p.parse_args()

    gpu_fingerprint.VICTIM_URL = args.victim_url

    safe = args.model.split("/")[-1]
    fp_dir = os.path.join(args.output_dir, "fingerprints")
    ts_dir = os.path.join(args.output_dir, "timeseries", f"{safe}[{args.quant}][seed={args.seed}]")
    os.makedirs(fp_dir, exist_ok=True)
    os.makedirs(ts_dir, exist_ok=True)
    csv_path = os.path.join(fp_dir, f"{safe}[{args.quant}][seed={args.seed}].csv")

    wait_for_victim(args.victim_url)

    if args.cumulative_lengths:
        cum_lengths = [int(x) for x in args.cumulative_lengths.split(",")]
    else:
        cum_lengths = []
        c = 0
        for L in MARGINAL_PROMPT_LENGTHS:
            c += L
            cum_lengths.append(c)

    mode = "a" if args.append else "w"
    write_header = not (args.append and os.path.exists(csv_path))
    print(f"[attacker] {'appending to' if args.append else 'writing'} {csv_path}", flush=True)
    with open(csv_path, mode) as f:
        if write_header:
            f.write("prompt_length,peak_vram_mb\n")
        for cumulative in cum_lengths:
            prompt = make_prompt(cumulative, args.model, seed=args.seed)
            peak = prompt_and_probe(prompt, cumulative, out_dir=ts_dir)
            f.write(f"{cumulative},{peak:.2f}\n")
            print(f"[attacker] L={cumulative} peak_vram={peak:.2f}MB", flush=True)
            time.sleep(1)
    print("[attacker] done", flush=True)


if __name__ == "__main__":
    main()
