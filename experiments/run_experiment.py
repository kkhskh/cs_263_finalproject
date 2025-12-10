#!/usr/bin/env python3
"""
Comprehensive Experiment Runner

Runs fingerprinting experiments across:
  - Multiple models (fake_a, fake_b, distilgpt2, opt-125m, gpt2-medium)
  - Multiple isolation levels (Docker, gVisor, Firecracker)
  - Multiple mitigation strategies (none, random, bucket, constant)
  - Multiple repetitions for statistical significance
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime
from typing import List, Optional

# Configuration
MODELS = {
    "fake": ["fake_a", "fake_b", "fake_c"],
    "real": ["distilgpt2", "gpt2", "opt-125m", "gpt2-medium"],
    "all": ["fake_a", "fake_b", "fake_c", "distilgpt2", "gpt2", "opt-125m", "gpt2-medium"],
}

OBFUSCATION_STRATEGIES = {
    "none": {"strategy": "none", "param": 0},
    "random_50": {"strategy": "random", "param": 50},
    "random_100": {"strategy": "random", "param": 100},
    "bucket_100": {"strategy": "bucket", "param": 100},
    "bucket_500": {"strategy": "bucket", "param": 500},
    "constant_1000": {"strategy": "constant", "param": 1000},
    "constant_3000": {"strategy": "constant", "param": 3000},
}

VICTIM_IMAGE = "llm-victim"
VICTIM_PORT = 8000

# Startup times
STARTUP_TIMES = {
    "fake_a": 5, "fake_b": 5, "fake_c": 5,
    "distilgpt2": 60, "gpt2": 90, "opt-125m": 90, "gpt2-medium": 120,
}


def run_cmd(cmd: List[str], capture: bool = False) -> Optional[str]:
    try:
        if capture:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        else:
            subprocess.run(cmd, check=True)
            return None
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd)}")
        return None


def stop_container(name: str):
    run_cmd(["docker", "stop", name], capture=True)
    run_cmd(["docker", "rm", "-f", name], capture=True)


def start_victim(model: str, runtime: str = "docker", 
                 obfuscation: str = "none", obfuscation_param: float = 0) -> bool:
    """Start victim container with specified configuration."""
    container_name = "victim_exp"
    stop_container(container_name)
    
    use_real = "1" if model in MODELS["real"] else "0"
    
    cmd = ["docker", "run", "-d", "--rm", "--name", container_name,
           "-p", f"{VICTIM_PORT}:{VICTIM_PORT}",
           "-e", f"MODEL_NAME={model}",
           "-e", f"USE_REAL_MODELS={use_real}",
           "-e", "COVERT_ENABLED=0",
           "-e", f"OBFUSCATION_STRATEGY={obfuscation}",
           "-e", f"OBFUSCATION_PARAM={obfuscation_param}"]
    
    # Add runtime if not default docker
    if runtime == "gvisor":
        cmd.extend(["--runtime=runsc"])
    elif runtime == "firecracker":
        # Firecracker requires different setup
        print("Firecracker not implemented in this script")
        return False
    
    cmd.append(VICTIM_IMAGE)
    
    print(f"Starting: {' '.join(cmd)}")
    result = run_cmd(cmd, capture=True)
    return result is not None


def wait_for_service(timeout: int = 120) -> bool:
    """Wait for victim service to be ready."""
    import urllib.request
    import urllib.error
    
    url = f"http://localhost:{VICTIM_PORT}/health"
    start = time.time()
    
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return True
        except:
            pass
        time.sleep(2)
        print(".", end="", flush=True)
    
    print(" timeout!")
    return False


def run_traffic(model_tag: str, n_requests: int, output_dir: str) -> Optional[str]:
    """Run traffic generator."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"exp_{model_tag}_{timestamp}.csv")
    
    cmd = ["python3", "traffic_gen.py",
           "--mode", "fingerprint",
           "--model-tag", model_tag,
           "--n", str(n_requests),
           "--output", output_file]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        return output_file
    except:
        return None


def run_single_experiment(model: str, n_requests: int, output_dir: str,
                         runtime: str = "docker", obfuscation: str = "none") -> Optional[str]:
    """Run experiment for single configuration."""
    obf_config = OBFUSCATION_STRATEGIES.get(obfuscation, {"strategy": "none", "param": 0})
    
    tag = f"{model}_{runtime}_{obfuscation}"
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {tag}")
    print(f"{'='*60}")
    
    # Start container
    if not start_victim(model, runtime, obf_config["strategy"], obf_config["param"]):
        return None
    
    # Wait for service
    startup_time = STARTUP_TIMES.get(model, 60)
    print(f"Waiting for service (up to {startup_time}s)...", end="", flush=True)
    
    if not wait_for_service(startup_time):
        stop_container("victim_exp")
        return None
    
    print(" ready!")
    time.sleep(2)  # Extra settle time
    
    # Run traffic
    output = run_traffic(tag, n_requests, output_dir)
    
    # Stop container
    stop_container("victim_exp")
    
    return output


def run_full_experiment(models: List[str], n_requests: int, n_repetitions: int,
                       output_dir: str, runtime: str, obfuscations: List[str]):
    """Run full experiment matrix."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print("FULL EXPERIMENT")
    print(f"Models: {models}")
    print(f"Requests per run: {n_requests}")
    print(f"Repetitions: {n_repetitions}")
    print(f"Runtime: {runtime}")
    print(f"Obfuscations: {obfuscations}")
    print(f"Output: {output_dir}")
    print(f"{'#'*60}")
    
    results = []
    
    for rep in range(n_repetitions):
        print(f"\n*** REPETITION {rep + 1}/{n_repetitions} ***")
        
        for model in models:
            for obf in obfuscations:
                output = run_single_experiment(
                    model, n_requests, output_dir, runtime, obf)
                if output:
                    results.append(output)
                else:
                    print(f"WARNING: Experiment failed for {model}/{obf}")
    
    return results


def run_analysis(result_files: List[str], output_plot: str):
    """Run statistical analysis."""
    if not result_files:
        return
    
    print(f"\n{'='*60}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*60}")
    
    cmd = ["python3", "analyze_stats.py", "--files"] + result_files
    if output_plot:
        cmd.extend(["--plot", output_plot])
    
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive experiments")
    parser.add_argument("--models", "-m", nargs="+", default=["fake_a", "fake_b"])
    parser.add_argument("--requests", "-n", type=int, default=50)
    parser.add_argument("--repetitions", "-r", type=int, default=3,
                       help="Number of repetitions for statistical significance")
    parser.add_argument("--output-dir", "-o", default="./experiment_results")
    parser.add_argument("--runtime", choices=["docker", "gvisor"], default="docker")
    parser.add_argument("--obfuscations", nargs="+", default=["none"],
                       choices=list(OBFUSCATION_STRATEGIES.keys()))
    parser.add_argument("--plot", "-p", default="results.png")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--list-obfuscations", action="store_true")
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        for category, models in MODELS.items():
            print(f"  {category}: {', '.join(models)}")
        return
    
    if args.list_obfuscations:
        print("Available obfuscation strategies:")
        for name, config in OBFUSCATION_STRATEGIES.items():
            print(f"  {name}: {config}")
        return
    
    # Expand model groups
    models = []
    for m in args.models:
        if m in MODELS:
            models.extend(MODELS[m])
        else:
            models.append(m)
    
    # Change to experiments directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run experiments
    results = run_full_experiment(
        models, args.requests, args.repetitions,
        args.output_dir, args.runtime, args.obfuscations)
    
    # Run analysis
    if args.plot:
        plot_path = os.path.join(args.output_dir, args.plot)
    else:
        plot_path = None
    
    run_analysis(results, plot_path)
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"Results: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
