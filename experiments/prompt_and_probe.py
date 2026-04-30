"""Container-based orchestrator for the GPU Prompt & Probe attack.

Spins up victim and attacker as **separate Docker containers** via
infra/docker-compose.yml, one (model, quant, seed) configuration at a time.
The attacker writes its fingerprint CSV into ../results/fingerprints/ via a
bind mount.
"""
import argparse
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COMPOSE_FILE = os.path.join(REPO_ROOT, "infra", "docker-compose.yml")

# Default minimal model set — small fp16 models for quick re-runs.
DEFAULT_MODELS = {
    "meta-llama/Llama-3.2-1B": ["fp16"],
    "meta-llama/Llama-3.2-3B": ["fp16"],
}

# Full set kept here for reference (matches the paper).
HF_MODELS_FULL = {
    "meta-llama/Llama-3.1-8B": ["q-8bit"],
    "meta-llama/Llama-3.2-1B": ["fp16", "q-8bit"],
    "meta-llama/Llama-3.2-3B": ["fp16", "q-8bit"],
    "google/gemma-2b": ["fp16", "q-8bit"],
    "google/gemma-7b": ["q-8bit"],
    "mistralai/Mistral-7B-v0.1": ["q-8bit"],
    "mistralai/Mistral-7B-Instruct-v0.2": ["q-8bit"],
    "Qwen/Qwen2-7B-Instruct": ["q-8bit"],
}


def compose(*args, env=None, check=True, capture=False):
    cmd = ["docker", "compose", "-f", COMPOSE_FILE, *args]
    print(f"[orch] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(
        cmd,
        env={**os.environ, **(env or {})},
        check=check,
        cwd=REPO_ROOT,
        text=True,
        capture_output=capture,
    )


def build_images():
    compose("build")


def teardown():
    # Keep named volumes (HF cache) across runs — only remove containers.
    compose("down", "--remove-orphans", check=False)


def hf_token() -> str:
    tok = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return ""


def run_one(model: str, quant: str, seed: int, token: str,
            cumulative_lengths: str = "", append: bool = False):
    safe = model.split("/")[-1]
    out_csv = os.path.join(REPO_ROOT, "results", "fingerprints",
                           f"{safe}[{quant}][seed={seed}].csv")
    print(f"\n=== {model} [{quant}] seed={seed} ===", flush=True)

    env = {
        "MODEL_NAME": model,
        "QUANT_MODE": quant,
        "SEED": str(seed),
        "HF_TOKEN": token,
        "CUMULATIVE_LENGTHS": cumulative_lengths,
        "ATTACKER_APPEND": "1" if append else "",
    }

    # Fresh state each run — prevents cross-config GPU memory carryover.
    teardown()

    compose("up", "-d", "victim", env=env)
    try:
        # Run attacker (foreground) and stream logs
        compose("run", "--rm", "--no-deps", "attacker", env=env)
        if not os.path.exists(out_csv):
            print(f"[orch] WARNING: expected output {out_csv} not found", file=sys.stderr)
    finally:
        teardown()


def main():
    parser = argparse.ArgumentParser(description="Containerized Prompt & Probe orchestrator")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of seeds per model (default: 1 for quick re-run)")
    parser.add_argument("--full", action="store_true",
                        help="Use the full model matrix from the paper instead of the minimal set")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--cumulative-lengths", default="",
                        help="comma-separated cumulative lengths to run (overrides default sweep)")
    parser.add_argument("--append", action="store_true",
                        help="append to existing CSV instead of overwriting")
    args = parser.parse_args()

    models = HF_MODELS_FULL if args.full else DEFAULT_MODELS
    seeds = list(range(args.seeds))
    token = hf_token()
    if not token:
        print("[orch] WARNING: no HF token found; gated models will fail", file=sys.stderr)

    if not args.skip_build:
        build_images()

    for model, quants in models.items():
        for quant in quants:
            for seed in seeds:
                run_one(model, quant, seed, token,
                        cumulative_lengths=args.cumulative_lengths,
                        append=args.append)

    print("\n=== All configurations complete ===")


if __name__ == "__main__":
    main()
