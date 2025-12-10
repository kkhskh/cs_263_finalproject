<<<<<<< HEAD
#!/usr/bin/env python3
"""
Traffic Generator for LLM Side-Channel Experiments
"""

import os
import csv
import time
import random
import argparse
import requests
from datetime import datetime
from typing import List, Dict, Optional

NON_SENSITIVE_PROMPTS = [
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "Tell me about the history of jazz music.",
    "What are the benefits of exercise?",
    "How do airplanes fly?",
    "Describe the water cycle.",
    "What is machine learning?",
    "Tell me about ancient Rome.",
    "How does the internet work?",
    "What causes earthquakes?",
]

SENSITIVE_PROMPTS = [
    "How do I secure my crypto wallet?",
    "What is the best password manager?",
    "Tell me about secret government programs.",
    "How do hackers attack systems?",
    "What are the best encryption keys?",
    "How to protect PII data?",
    "How does credit card fraud work?",
    "What is SSN identity theft?",
    "Tell me about medical record privacy.",
    "How do bank security systems work?",
]

FINGERPRINT_PROMPTS = [
    "The future of artificial intelligence is",
    "Once upon a time in a land far away",
    "The quick brown fox jumps over",
    "In the beginning, there was",
    "To be or not to be, that is",
]


def send_request(url: str, prompt: str, tag: Optional[str] = None,
                 max_new_tokens: int = 20, timeout: float = 60.0) -> Dict:
    payload = {"prompt": prompt, "max_new_tokens": max_new_tokens, "tag": tag}
    result = {
        "success": False, "status_code": 0, "local_elapsed_ms": 0,
        "server_elapsed_ms": 0, "actual_elapsed_ms": 0, "covert_triggered": False,
        "model_name": "", "obfuscation_strategy": "", "prompt": prompt,
        "tag": tag, "timestamp": time.time()
    }
    
    try:
        start = time.perf_counter()
        response = requests.post(url, json=payload, timeout=timeout,
                                headers={"Content-Type": "application/json"})
        elapsed = time.perf_counter() - start
        
        result["local_elapsed_ms"] = elapsed * 1000
        result["status_code"] = response.status_code
        
        if response.status_code == 200:
            data = response.json()
            result["success"] = True
            result["server_elapsed_ms"] = data.get("elapsed_ms", 0)
            result["actual_elapsed_ms"] = data.get("actual_elapsed_ms", 0)
            result["covert_triggered"] = data.get("covert_triggered", False)
            result["model_name"] = data.get("model_name", "")
            result["obfuscation_strategy"] = data.get("obfuscation_strategy", "")
    except Exception as e:
        result["error"] = str(e)
    
    return result


def run_fingerprint_mode(url: str, n: int, model_tag: str = "") -> List[Dict]:
    results = []
    print(f"Starting fingerprint experiment: {n} requests")
    print(f"Model tag: {model_tag}")
    print("-" * 60)
    
    for i in range(n):
        prompt = FINGERPRINT_PROMPTS[i % len(FINGERPRINT_PROMPTS)]
        result = send_request(url, prompt, tag=f"{model_tag}_fp{i}")
        result["request_id"] = i
        results.append(result)
        
        status = "OK  " if result["success"] else "FAIL"
        print(f"[{i:4d}] {status} local={result['local_elapsed_ms']:.1f}ms "
              f"server={result['server_elapsed_ms']:.1f}ms "
              f"actual={result['actual_elapsed_ms']:.1f}ms")
    
    return results


def run_mixed_mode(url: str, n: int, sensitive_ratio: float = 0.5,
                   model_tag: str = "") -> List[Dict]:
    results = []
    print(f"Starting mixed experiment: {n} requests, {sensitive_ratio:.0%} sensitive")
    print("-" * 60)
    
    for i in range(n):
        is_sensitive = random.random() < sensitive_ratio
        prompt = random.choice(SENSITIVE_PROMPTS if is_sensitive else NON_SENSITIVE_PROMPTS)
        result = send_request(url, prompt, tag=f"{model_tag}_{i}")
        result["is_sensitive"] = is_sensitive
        result["request_id"] = i
        results.append(result)
        
        status = "OK  " if result["success"] else "FAIL"
        sens = "SENS" if is_sensitive else "    "
        covert = "COVERT" if result["covert_triggered"] else "      "
        print(f"[{i:4d}] {status} {sens} {covert} server={result['server_elapsed_ms']:.1f}ms")
    
    return results


def save_results(results: List[Dict], filename: str):
    if not results:
        return
    fieldnames = ["success", "status_code", "local_elapsed_ms", "server_elapsed_ms",
                  "actual_elapsed_ms", "covert_triggered", "model_name", 
                  "obfuscation_strategy", "request_id", "tag", "is_sensitive", "timestamp"]
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} results to {filename}")


def print_summary(results: List[Dict]):
    if not results:
        return
    successful = [r for r in results if r["success"]]
    server_times = [r["server_elapsed_ms"] for r in successful if r["server_elapsed_ms"] > 0]
    actual_times = [r["actual_elapsed_ms"] for r in successful if r.get("actual_elapsed_ms", 0) > 0]
    
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Total: {len(results)}, Success: {len(successful)}")
    
    if server_times:
        print(f"Server timing: min={min(server_times):.1f}, max={max(server_times):.1f}, "
              f"mean={sum(server_times)/len(server_times):.1f}ms")
    if actual_times:
        print(f"Actual timing: min={min(actual_times):.1f}, max={max(actual_times):.1f}, "
              f"mean={sum(actual_times)/len(actual_times):.1f}ms")
    
    covert = sum(1 for r in successful if r.get("covert_triggered"))
    print(f"Covert triggered: {covert} ({100*covert/len(results):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Traffic generator")
    parser.add_argument("--url", default="http://localhost:8000/generate")
    parser.add_argument("--mode", choices=["mixed", "fingerprint"], default="fingerprint")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--model-tag", default="")
    parser.add_argument("--sensitive-ratio", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    
    if args.mode == "fingerprint":
        results = run_fingerprint_mode(args.url, args.n, args.model_tag)
    else:
        results = run_mixed_mode(args.url, args.n, args.sensitive_ratio, args.model_tag)
    
    print_summary(results)
    
    output = args.output or f"traffic_{args.model_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    save_results(results, output)
=======
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
>>>>>>> 8213a2da0756912468d3a900e918f4e3f949446b


if __name__ == "__main__":
    main()
