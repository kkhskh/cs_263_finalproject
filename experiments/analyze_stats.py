#!/usr/bin/env python3
"""
Statistical Analysis for LLM Side-Channel Experiments

Features:
  - Comprehensive statistics (mean, median, std, percentiles)
  - Cohen's d effect size for all pairwise comparisons
  - Welch's t-test for statistical significance
  - Confidence intervals
  - Fingerprinting accuracy estimation
  - Publication-quality plots
"""

import os
import csv
import argparse
import statistics
from typing import List, Dict, Tuple
from collections import defaultdict
import math

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/numpy not available")


def load_csv(filename: str) -> List[Dict]:
    data = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = {
                'success': row.get('success', 'True') == 'True',
                'server_ms': float(row.get('server_elapsed_ms', 0) or 0),
                'actual_ms': float(row.get('actual_elapsed_ms', 0) or 0),
                'local_ms': float(row.get('local_elapsed_ms', 0) or 0),
                'model_name': row.get('model_name', ''),
                'obfuscation': row.get('obfuscation_strategy', 'none'),
            }
            data.append(record)
    return data


def compute_stats(values: List[float]) -> Dict:
    if not values:
        return {}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    return {
        'n': n,
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'stdev': statistics.stdev(values) if n > 1 else 0,
        'stderr': statistics.stdev(values) / math.sqrt(n) if n > 1 else 0,
        'min': min(values),
        'max': max(values),
        'p25': sorted_vals[int(n * 0.25)],
        'p50': sorted_vals[int(n * 0.50)],
        'p75': sorted_vals[int(n * 0.75)],
        'p90': sorted_vals[int(n * 0.90)],
        'p95': sorted_vals[int(n * 0.95)],
        'p99': sorted_vals[min(int(n * 0.99), n - 1)],
        'ci95_low': statistics.mean(values) - 1.96 * statistics.stdev(values) / math.sqrt(n) if n > 1 else 0,
        'ci95_high': statistics.mean(values) + 1.96 * statistics.stdev(values) / math.sqrt(n) if n > 1 else 0,
    }


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float('inf')
    
    mean1, mean2 = statistics.mean(group1), statistics.mean(group2)
    var1, var2 = statistics.variance(group1), statistics.variance(group2)
    
    # Pooled standard deviation
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return float('inf')
    
    return abs(mean1 - mean2) / pooled_std


def welch_t_test(group1: List[float], group2: List[float]) -> Tuple[float, float]:
    """
    Welch's t-test for unequal variances.
    Returns (t_statistic, degrees_of_freedom).
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return (float('inf'), 0)
    
    mean1, mean2 = statistics.mean(group1), statistics.mean(group2)
    var1, var2 = statistics.variance(group1), statistics.variance(group2)
    
    se1, se2 = var1 / n1, var2 / n2
    se_diff = math.sqrt(se1 + se2)
    
    if se_diff == 0:
        return (float('inf'), 0)
    
    t_stat = (mean1 - mean2) / se_diff
    
    # Welch-Satterthwaite degrees of freedom
    df = (se1 + se2) ** 2 / (se1 ** 2 / (n1 - 1) + se2 ** 2 / (n2 - 1))
    
    return (abs(t_stat), df)


def check_distributions_overlap(group1: List[float], group2: List[float]) -> bool:
    """Check if two distributions have overlapping ranges."""
    return not (max(group1) < min(group2) or max(group2) < min(group1))


def analyze_file(filename: str) -> Dict:
    data = load_csv(filename)
    successful = [r for r in data if r['success'] and r['server_ms'] > 0]
    
    model_name = successful[0]['model_name'] if successful else os.path.basename(filename)
    obfuscation = successful[0]['obfuscation'] if successful else 'none'
    
    server_times = [r['server_ms'] for r in successful]
    actual_times = [r['actual_ms'] for r in successful if r['actual_ms'] > 0]
    
    return {
        'filename': filename,
        'model_name': model_name,
        'obfuscation': obfuscation,
        'n_total': len(data),
        'n_successful': len(successful),
        'server_stats': compute_stats(server_times),
        'actual_stats': compute_stats(actual_times) if actual_times else None,
        'server_times': server_times,
        'actual_times': actual_times,
    }


def pairwise_analysis(results: List[Dict]) -> List[Dict]:
    """Compute all pairwise statistical comparisons."""
    comparisons = []
    
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            r1, r2 = results[i], results[j]
            times1 = r1['server_times']
            times2 = r2['server_times']
            
            if not times1 or not times2:
                continue
            
            d = cohens_d(times1, times2)
            t_stat, df = welch_t_test(times1, times2)
            overlap = check_distributions_overlap(times1, times2)
            
            # Classification is trivial if no overlap or Cohen's d > 2
            distinguishable = not overlap or d > 2.0
            
            comparisons.append({
                'model1': r1['model_name'],
                'model2': r2['model_name'],
                'mean_diff_ms': abs(r1['server_stats']['mean'] - r2['server_stats']['mean']),
                'cohens_d': d,
                't_statistic': t_stat,
                'df': df,
                'overlap': overlap,
                'distinguishable': distinguishable,
            })
    
    return comparisons


def print_analysis(results: List[Dict]):
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS REPORT")
    print("=" * 70)
    
    for r in results:
        print(f"\n--- {r['model_name']} ({r['filename']}) ---")
        print(f"Obfuscation: {r['obfuscation']}")
        print(f"Samples: {r['n_successful']}/{r['n_total']}")
        
        s = r['server_stats']
        if s:
            print(f"\nServer Timing (ms):")
            print(f"  Mean: {s['mean']:.2f} ± {s['stdev']:.2f} (std)")
            print(f"  95% CI: [{s['ci95_low']:.2f}, {s['ci95_high']:.2f}]")
            print(f"  Median: {s['median']:.2f}")
            print(f"  Range: [{s['min']:.2f}, {s['max']:.2f}]")
            print(f"  Percentiles: P50={s['p50']:.2f}, P90={s['p90']:.2f}, P99={s['p99']:.2f}")
        
        if r['actual_stats']:
            a = r['actual_stats']
            print(f"\nActual Timing (before obfuscation):")
            print(f"  Mean: {a['mean']:.2f} ± {a['stdev']:.2f}")


def print_pairwise_analysis(comparisons: List[Dict]):
    print("\n" + "=" * 70)
    print("PAIRWISE STATISTICAL COMPARISONS")
    print("=" * 70)
    
    distinguishable_count = 0
    total_count = len(comparisons)
    
    for c in comparisons:
        status = "✓ DISTINGUISHABLE" if c['distinguishable'] else "✗ OVERLAPPING"
        print(f"\n{c['model1']} vs {c['model2']}:")
        print(f"  Mean difference: {c['mean_diff_ms']:.2f} ms")
        print(f"  Cohen's d: {c['cohens_d']:.2f} {'(large)' if c['cohens_d'] > 0.8 else '(small)'}")
        print(f"  Welch's t: {c['t_statistic']:.2f} (df={c['df']:.1f})")
        print(f"  Distributions overlap: {'Yes' if c['overlap'] else 'No'}")
        print(f"  Status: {status}")
        
        if c['distinguishable']:
            distinguishable_count += 1
    
    accuracy = 100 * distinguishable_count / total_count if total_count > 0 else 0
    print(f"\n" + "=" * 70)
    print(f"OVERALL CLASSIFICATION POTENTIAL: {accuracy:.1f}%")
    print(f"({distinguishable_count}/{total_count} model pairs distinguishable)")
    print("=" * 70)


def plot_results(results: List[Dict], comparisons: List[Dict], output: str):
    if not HAS_PLOTTING:
        print("Plotting not available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Bar chart with confidence intervals
    models = [r['model_name'] for r in results]
    means = [r['server_stats']['mean'] for r in results]
    ci_low = [r['server_stats']['mean'] - r['server_stats']['ci95_low'] for r in results]
    ci_high = [r['server_stats']['ci95_high'] - r['server_stats']['mean'] for r in results]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    bars = axes[0, 0].bar(models, means, yerr=[ci_low, ci_high], capsize=5, 
                          color=colors, edgecolor='black')
    axes[0, 0].set_ylabel('Response Time (ms)')
    axes[0, 0].set_title('Mean Response Time with 95% CI')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    for bar, mean in zip(bars, means):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                        f'{mean:.0f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Box plots
    data_for_box = [r['server_times'] for r in results]
    bp = axes[0, 1].boxplot(data_for_box, labels=models, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    axes[0, 1].set_ylabel('Response Time (ms)')
    axes[0, 1].set_title('Response Time Distribution')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Cohen's d heatmap
    n_models = len(models)
    d_matrix = np.zeros((n_models, n_models))
    for c in comparisons:
        i1 = models.index(c['model1'])
        i2 = models.index(c['model2'])
        d_matrix[i1, i2] = c['cohens_d']
        d_matrix[i2, i1] = c['cohens_d']
    
    im = axes[1, 0].imshow(d_matrix, cmap='YlOrRd', aspect='auto')
    axes[1, 0].set_xticks(range(n_models))
    axes[1, 0].set_yticks(range(n_models))
    axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 0].set_yticklabels(models)
    axes[1, 0].set_title("Cohen's d Effect Size Matrix")
    plt.colorbar(im, ax=axes[1, 0])
    
    # Plot 4: Scatter of all samples
    for i, (r, color) in enumerate(zip(results, colors)):
        y = [i] * len(r['server_times'])
        axes[1, 1].scatter(r['server_times'], y, alpha=0.5, c=[color], s=20, label=r['model_name'])
    
    axes[1, 1].set_yticks(range(len(models)))
    axes[1, 1].set_yticklabels(models)
    axes[1, 1].set_xlabel('Response Time (ms)')
    axes[1, 1].set_title('All Samples by Model')
    
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {output}")


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis")
    parser.add_argument("--files", "-f", nargs="+", required=True)
    parser.add_argument("--plot", "-p", default=None)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()
    
    results = []
    for f in args.files:
        if os.path.exists(f):
            print(f"Analyzing {f}...")
            results.append(analyze_file(f))
    
    if not results:
        print("No valid files")
        return
    
    print_analysis(results)
    
    if len(results) >= 2:
        comparisons = pairwise_analysis(results)
        print_pairwise_analysis(comparisons)
        
        if args.plot:
            plot_results(results, comparisons, args.plot)
    
    if args.output:
        with open(args.output, 'w') as f:
            for r in results:
                s = r['server_stats']
                f.write(f"{r['model_name']},{s['mean']:.2f},{s['stdev']:.2f},{s['n']}\n")
        print(f"Saved summary to {args.output}")


if __name__ == "__main__":
    main()
