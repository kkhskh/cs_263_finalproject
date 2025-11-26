# experiments/traffic_gen.py
"""
Send a bunch of prompts to the victim /generate endpoint.

Usage:
  python traffic_gen.py --model-tag fake_a --n 100 --url http://localhost:8000/generate
"""

import argparse
import random
import string
import time
import requests


SENSITIVE = [
    "Tell me about crypto wallets.",
    "How do I reset my password safely?",
    "What is the best way to store a secret API key?",
]

NON_SENSITIVE = [
    "What is your favorite movie?",
    "Explain the rules of soccer.",
    "Describe the process of photosynthesis.",
]


def rand_noise(length: int = 16) -> str:
    chars = string.ascii_letters + string.digits
    return "".join(random.choice(chars) for _ in range(length))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/generate")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--model-tag", default="unknown")
    parser.add_argument("--sleep-ms", type=float, default=50.0)
    args = parser.parse_args()

    for i in range(args.n):
        if random.random() < 0.5:
            prompt = random.choice(SENSITIVE) + " " + rand_noise()
        else:
            prompt = random.choice(NON_SENSITIVE) + " " + rand_noise()

        payload = {
            "prompt": prompt,
            "tag": f"{args.model_tag}_req{i}",
        }

        t0 = time.time()
        r = requests.post(args.url, json=payload, timeout=10)
        dt = (time.time() - t0) * 1000.0

        print(f"[{i}] status={r.status_code} local_elapsed={dt:.1f}ms")

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)


if __name__ == "__main__":
    main()
