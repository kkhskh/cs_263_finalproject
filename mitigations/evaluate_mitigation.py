#!/usr/bin/env python3
"""
Mitigation Effectiveness Evaluation

Evaluates how well timing obfuscation defeats fingerprinting attacks:
  - Compares baseline (no obfuscation) vs mitigated timing distributions
  - Measures entropy increase
  - Computes classification accuracy degradation
"""

import os
import csv
import math
import argparse
import statistics
from typing import List, Dict, Tuple


def load_timings(filename: str) -> List[float]:
    """Load server timings from CSV."""
    timings = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row.get('server_elapsed_ms', 0) or 0)
            if t > 0:
                timings.append(t)
    return timings


def compute_entropy(values: List[float], n_bins: int = 50) -> float:
    """Compute entropy of timing distribution."""
    if not values:
        return 0.0
    
    min_val, max_val = min(values), max(values)
    if max_val == min_val:
        return 0.0
    
    bin_width = (max_val - min_val) / n_bins
    bins = [0] * n_bins
    
    for v in values:
        idx = min(int((v - min_val) / bin_width), n_bins - 1)
        bins[idx] += 1
    
    total = len(values)
    entropy = 0.0
    for count in bins:
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    
    return entropy


def compute_overlap_ratio(dist1: List[float], dist2: List[float]) -> float:
    """Compute overlap ratio between two distributions."""
    if not dist1 or not dist2:
        return 0.0
    
    min1, max1 = min(dist1), max(dist1)
    min2, max2 = min(dist2), max(dist2)
    
    overlap_start = max(min1, min2)
    overlap_end = min(max1, max2)
    
    if overlap_start >= overlap_end:
        return 0.0
    
    overlap = overlap_end - overlap_start
    total_range = max(max1, max2) - min(min1, min2)
    
    return overlap / total_range if total_range > 0 else 0.0


def estimate_classification_accuracy(distributions: Dict[str, List[float]]) -> float:
    """
    Estimate classification accuracy using simple threshold classifier.
    Lower accuracy = better mitigation.
    """
    if len(distributions) < 2:
        return 1.0
    
    # Sort models by mean timing
    models = sorted(distributions.keys(), 
                   key=lambda m: statistics.mean(distributions[m]))
    
    correct = 0
    total = 0
    
    for i in range(len(models) - 1):
        m1, m2 = models[i], models[i + 1]
        d1, d2 = distributions[m1], distributions[m2]
        
        # Find optimal threshold between two distributions
        threshold = (statistics.mean(d1) + statistics.mean(d2)) / 2
        
        # Count correctly classified samples
        correct += sum(1 for v in d1 if v < threshold)
        correct += sum(1 for v in d2 if v >= threshold)
        total += len(d1) + len(d2)
    
    return correct / total if total > 0 else 1.0


def evaluate_mitigation(baseline_files: Dict[str, str], 
                        mitigated_files: Dict[str, str]) -> Dict:
    """
    Evaluate mitigation effectiveness.
    
    Args:
        baseline_files: {model_name: csv_path} for baseline (no obfuscation)
        mitigated_files: {model_name: csv_path} for mitigated
    
    Returns:
        Evaluation metrics
    """
    results = {
        'baseline': {},
        'mitigated': {},
        'comparison': {}
    }
    
    # Load and analyze baseline
    baseline_dist = {}
    for model, path in baseline_files.items():
        timings = load_timings(path)
        baseline_dist[model] = timings
        results['baseline'][model] = {
            'n': len(timings),
            'mean': statistics.mean(timings) if timings else 0,
            'stdev': statistics.stdev(timings) if len(timings) > 1 else 0,
            'entropy': compute_entropy(timings),
        }
    
    # Load and analyze mitigated
    mitigated_dist = {}
    for model, path in mitigated_files.items():
        timings = load_timings(path)
        mitigated_dist[model] = timings
        results['mitigated'][model] = {
            'n': len(timings),
            'mean': statistics.mean(timings) if timings else 0,
            'stdev': statistics.stdev(timings) if len(timings) > 1 else 0,
            'entropy': compute_entropy(timings),
        }
    
    # Compare
    results['comparison']['baseline_accuracy'] = estimate_classification_accuracy(baseline_dist)
    results['comparison']['mitigated_accuracy'] = estimate_classification_accuracy(mitigated_dist)
    results['comparison']['accuracy_reduction'] = (
        results['comparison']['baseline_accuracy'] - 
        results['comparison']['mitigated_accuracy']
    )
    
    # Entropy increase
    baseline_entropies = [results['baseline'][m]['entropy'] for m in baseline_files]
    mitigated_entropies = [results['mitigated'][m]['entropy'] for m in mitigated_files]
    
    results['comparison']['mean_baseline_entropy'] = statistics.mean(baseline_entropies) if baseline_entropies else 0
    results['comparison']['mean_mitigated_entropy'] = statistics.mean(mitigated_entropies) if mitigated_entropies else 0
    results['comparison']['entropy_increase'] = (
        results['comparison']['mean_mitigated_entropy'] - 
        results['comparison']['mean_baseline_entropy']
    )
    
    return results


def print_evaluation(results: Dict):
    """Print evaluation results."""
    print("\n" + "=" * 70)
    print("MITIGATION EFFECTIVENESS EVALUATION")
    print("=" * 70)
    
    print("\n--- Baseline (No Obfuscation) ---")
    for model, stats in results['baseline'].items():
        print(f"  {model}: mean={stats['mean']:.1f}ms, std={stats['stdev']:.1f}ms, "
              f"entropy={stats['entropy']:.2f} bits")
    
    print("\n--- Mitigated (With Obfuscation) ---")
    for model, stats in results['mitigated'].items():
        print(f"  {model}: mean={stats['mean']:.1f}ms, std={stats['stdev']:.1f}ms, "
              f"entropy={stats['entropy']:.2f} bits")
    
    print("\n--- Comparison ---")
    c = results['comparison']
    print(f"  Baseline classification accuracy: {c['baseline_accuracy']*100:.1f}%")
    print(f"  Mitigated classification accuracy: {c['mitigated_accuracy']*100:.1f}%")
    print(f"  Accuracy reduction: {c['accuracy_reduction']*100:.1f} percentage points")
    print(f"  Entropy increase: {c['entropy_increase']:.2f} bits")
    
    if c['accuracy_reduction'] > 0.3:
        print("\n  ✓ MITIGATION EFFECTIVE: >30% accuracy reduction")
    elif c['accuracy_reduction'] > 0.1:
        print("\n  ~ MITIGATION PARTIAL: 10-30% accuracy reduction")
    else:
        print("\n  ✗ MITIGATION INEFFECTIVE: <10% accuracy reduction")


def main():
    parser = argparse.ArgumentParser(description="Evaluate mitigation effectiveness")
    parser.add_argument("--baseline", "-b", nargs="+", required=True,
                       help="Baseline CSV files (format: model:path)")
    parser.add_argument("--mitigated", "-m", nargs="+", required=True,
                       help="Mitigated CSV files (format: model:path)")
    args = parser.parse_args()
    
    # Parse file arguments
    baseline_files = {}
    for arg in args.baseline:
        if ':' in arg:
            model, path = arg.split(':', 1)
        else:
            model = os.path.basename(arg).split('_')[0]
            path = arg
        baseline_files[model] = path
    
    mitigated_files = {}
    for arg in args.mitigated:
        if ':' in arg:
            model, path = arg.split(':', 1)
        else:
            model = os.path.basename(arg).split('_')[0]
            path = arg
        mitigated_files[model] = path
    
    # Evaluate
    results = evaluate_mitigation(baseline_files, mitigated_files)
    print_evaluation(results)


if __name__ == "__main__":
    main()
