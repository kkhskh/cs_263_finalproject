import os
import subprocess
import time
import signal
import requests
import sys

from attacker.gpu_fingerprint import make_prompt, prompt_and_probe, PROMPT_LENGTHS

VICTIM_PORT = 8000
VICTIM_URL = f"http://127.0.0.1:{VICTIM_PORT}/generate"
N_REPEATS = 10
SEEDS = list(range(N_REPEATS))


HF_MODELS = { # mapping from models to quantization
    # meta
    "meta-llama/Llama-3.1-8B": ["q-8bit"],
    "meta-llama/Llama-3.2-1B": ["fp16", "q-8bit"],
    "meta-llama/Llama-3.2-3B": ["fp16", "q-8bit"],
    # google
    "google/gemma-2b": ["fp16", "q-8bit"],
    "google/gemma-7b": ["q-8bit"],
    # mistral
    "mistralai/Mistral-7B-v0.1": ["q-8bit"],
    "mistralai/Mistral-7B-Instruct-v0.2": ["q-8bit"],
    # alibaba
    "Qwen/Qwen2-7B-Instruct": ["q-8bit"],
}

MODELS = HF_MODELS

def wait_for_server_ready(timeout=600):
    """
    Poll the /generate endpoint until the server responds.
    This avoids relying on log output ordering.
    """
    print("[orchestrator] Waiting for server to become ready...")
    start = time.time()

    while True:
        if time.time() - start > timeout:
            raise RuntimeError("Server failed to start within timeout")

        try:
            r = requests.post(VICTIM_URL, json={"prompt": "warmup"})
            if r.status_code == 200:
                print("[orchestrator] Server is ready.")
                return
        except Exception:
            pass

        time.sleep(5)


def start_server(model_name: str, quant_mode: str) -> subprocess.Popen:
    """
    Start the victim server with MODEL_NAME and QUANT_MODE set.
    """
    env = os.environ.copy()
    env["MODEL_NAME"] = model_name
    env["QUANT_MODE"] = quant_mode

    print(f"\n=== Starting server for model: {model_name} ===")
    print(f"[orchestrator] Using victim_service directory: victim_service")

    proc = subprocess.Popen(
        [
            "uvicorn",
            "server:app",
            "--host", "0.0.0.0",
            "--port", str(VICTIM_PORT),
        ],
        cwd="victim_service",
        env=env,
        preexec_fn=os.setsid,   # mayeb delete this
    )
    return proc


def stop_server(proc):
    """Kill the victim server."""
    print("Stopping server...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass

def run_attack_for_model(model_name: str, quant_mode: str, seed: int=42):
    """Run fingerprinting on a given model."""
    safe_name = model_name.split("/")[-1] # replace so not interpreted as path
    out_csv = f"fingerprints/{safe_name}[{quant_mode}][seed={seed}].csv"
    print(f"Writing fingerprint data to {out_csv}")

    os.makedirs("fingerprints", exist_ok=True)

    with open(out_csv, "w") as f:
        f.write("prompt_length,peak_vram_mb\n")

        for L in PROMPT_LENGTHS:
            prompt = make_prompt(L, model_name, seed=seed)
            peak = prompt_and_probe(prompt, L, out_dir=f"timeseries_{safe_name}[{quant_mode}][seed={seed}]")
            f.write(f"{L},{peak:.2f}\n")
            time.sleep(1)

def main():
    for model_name, quant_modes in MODELS.items():
        for quant_mode in quant_modes:
            for seed in SEEDS:
                proc = start_server(model_name, quant_mode)
                wait_for_server_ready()
                run_attack_for_model(model_name, quant_mode, seed=seed)
                stop_server(proc)

    print("\n=== All models tested! ===")

if __name__ == "__main__":
    main()
