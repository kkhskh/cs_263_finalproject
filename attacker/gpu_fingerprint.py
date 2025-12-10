import os
import time
import requests
import torch
import csv
import string
import random
import threading

from transformers import AutoTokenizer

VICTIM_URL = "http://127.0.0.1:8000/generate"
MODEL_NAME = os.getenv("MODEL_NAME", "gpt2")
OUT_CSV = f"{MODEL_NAME}_fingerprint.csv"


# different prompt lengths to test
PROMPT_LENGTHS = [2, 1000, 2000, 3000, 4000]


def make_prompt(length: int, model_name: str, seed: int=42) -> str:
    """
    Generate a random prompt where each character maps to exactly 1 token
    under byte-level BPE tokenizers (GPT-2, LLaMA, Mistral, Gemma, etc.).
    """
    rng = random.Random(seed) 

    alphabet = ["0.", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = "".join(rng.choice(alphabet) for _ in range(length // 2))

    print(
        f"Prompt length for tokenizer {model_name}:",
        len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
    )
    return prompt


def prompt_and_probe(prompt: str, L: int, out_dir="timeseries") -> float:
    """
    Send prompt to victim and measure GPU VRAM usage while it processes.
    Logs a complete time series to: timeseries/run_L=<len>.csv
    Returns peak VRAM above baseline (in MB).
    """

    os.makedirs(out_dir, exist_ok=True)

    # prepare timeseries output file
    ts_path = os.path.join(out_dir, f"run_L={L}.csv")
    ts_file = open(ts_path, "w")
    ts_file.write("timestamp_ms,used_mb\n")

    # baseline measurement
    free0, total0 = torch.cuda.mem_get_info()
    baseline_used = (total0 - free0) / (1024 ** 2)

    peak_used = baseline_used
    done = False
    start_time = time.time()

    # start request in background thread
    def send_request():
        nonlocal done
        try:
            response = requests.post(
                VICTIM_URL,
                json={"prompt": prompt}
            )
        except Exception as e:
            print("Request error:", e)
        finally:
            done = True

    t = threading.Thread(target=send_request)
    t.start()

    # probe loop
    while not done:
        free, total = torch.cuda.mem_get_info()
        used = (total - free) / (1024 ** 2)

        peak_used = max(peak_used, used)

        timestamp_ms = (time.time() - start_time) * 1000.0
        ts_file.write(f"{timestamp_ms:.2f},{used:.2f}\n")

        time.sleep(0.001)  # 1 ms sampling

    t.join()
    ts_file.close()

    print(f"[L={L}] Timeseries saved to: {ts_path}")

    # return peak
    return peak_used


def main():
    print("Running GPU fingerprinting attack...")
    print(f"Saving results to: fingerprints/{OUT_CSV}")

    with open(f"fingerprints/{OUT_CSV}", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_length", "peak_vram_mb"])

        for L in PROMPT_LENGTHS:
            print(f"\n=== Prompt length {L} ===")
            prompt = make_prompt(L, "gpt2")
            peak = prompt_and_probe(prompt, L)
            print(f"Peak VRAM: {peak:.2f} MB")
            writer.writerow([L, peak])

    print("\nDone!")


if __name__ == "__main__":
    main()
