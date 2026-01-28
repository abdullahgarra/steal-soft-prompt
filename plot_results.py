#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_results.py

Generate plots from experimental results.
"""

import os
import csv
import argparse
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def load_csv_results(csv_path: str):
    """Load results from CSV."""
    results = []
    if not os.path.exists(csv_path):
        return results
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed = {}
            for key, val in row.items():
                try:
                    processed[key] = float(val)
                except (ValueError, TypeError):
                    processed[key] = val
            results.append(processed)
    return results


def aggregate_by_budget(results):
    """Group by budget and compute statistics."""
    grouped = defaultdict(lambda: defaultdict(list))
    
    for row in results:
        budget = int(row['budget'])
        for key, val in row.items():
            if key not in ['task', 'seed', 'budget', 'oracle', 'dp']:
                if isinstance(val, (int, float)) and not np.isnan(val):
                    grouped[budget][key].append(val)
    
    summary = {}
    for budget, metrics in grouped.items():
        summary[budget] = {}
        for metric, values in metrics.items():
            if len(values) > 0:
                summary[budget][f"{metric}_mean"] = np.mean(values)
                summary[budget][f"{metric}_std"] = np.std(values)
    
    return summary


def plot_metric_comparison(task, metric_name, output_dir):
    """Plot a metric across budgets for different configurations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{task.upper()} - {metric_name}', fontsize=16, fontweight='bold')
    
    configs = [
        ('nodp', 'probs', 'No DP, Soft Oracle', axes[0, 0]),
        ('nodp', 'hard', 'No DP, Hard Oracle', axes[0, 1]),
        ('dp', 'probs', 'DP, Soft Oracle', axes[1, 0]),
        ('dp', 'hard', 'DP, Hard Oracle', axes[1, 1]),
    ]
    
    for dp_mode, oracle, title, ax in configs:
        csv_path = os.path.join(output_dir, task, f"results_{dp_mode}_{oracle}.csv")
        
        if not os.path.exists(csv_path):
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue
        
        results = load_csv_results(csv_path)
        if not results:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue
        
        summary = aggregate_by_budget(results)
        
        budgets = sorted(summary.keys())
        
        # Plot victim and stolen metrics
        for model_type, color, marker in [('victim', 'blue', 'o'), ('stolen', 'red', 's')]:
            mean_key = f"{model_type}_{metric_name}_mean"
            std_key = f"{model_type}_{metric_name}_std"
            
            means = [summary[b].get(mean_key, np.nan) for b in budgets]
            stds = [summary[b].get(std_key, 0) for b in budgets]
            
            if not all(np.isnan(means)):
                ax.errorbar(budgets, means, yerr=stds, 
                           label=model_type.capitalize(), 
                           marker=marker, color=color, linewidth=2, 
                           capsize=5, markersize=8)
        
        ax.set_xlabel('Query Budget', fontsize=11)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, task, f"plot_{metric_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {output_path}")


def plot_pair_metrics(task, output_dir):
    """Plot pair metrics (agreement, KL divergence)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{task.upper()} - Stealing Effectiveness', fontsize=16, fontweight='bold')
    
    metrics = [
        ('pair_agreement', 'Agreement Rate', axes[0, 0]),
        ('pair_avg_kl_victim_to_stolen', 'KL Divergence (V||S)', axes[0, 1]),
        ('pair_prompt_cos_sim', 'Prompt Cosine Similarity', axes[1, 0]),
        ('pair_prompt_l2_dist', 'Prompt L2 Distance', axes[1, 1]),
    ]
    
    for metric_name, ylabel, ax in metrics:
        for dp_mode, color, linestyle in [('nodp', 'blue', '-'), ('dp', 'red', '--')]:
            for oracle, marker in [('probs', 'o'), ('hard', 's')]:
                csv_path = os.path.join(output_dir, task, f"results_{dp_mode}_{oracle}.csv")
                
                if not os.path.exists(csv_path):
                    continue
                
                results = load_csv_results(csv_path)
                if not results:
                    continue
                
                summary = aggregate_by_budget(results)
                budgets = sorted(summary.keys())
                
                mean_key = f"{metric_name}_mean"
                std_key = f"{metric_name}_std"
                
                means = [summary[b].get(mean_key, np.nan) for b in budgets]
                stds = [summary[b].get(std_key, 0) for b in budgets]
                
                if not all(np.isnan(means)):
                    label = f"{'DP' if dp_mode == 'dp' else 'No DP'}, {oracle}"
                    ax.errorbar(budgets, means, yerr=stds, 
                               label=label, marker=marker, color=color,
                               linestyle=linestyle, linewidth=2, 
                               capsize=4, markersize=6)
        
        ax.set_xlabel('Query Budget', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, task, f"plot_pair_metrics.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, 
                       choices=['sst2', 'qqp', 'mnli', 'qnli'])
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    
    task_dir = os.path.join(args.results_dir, args.task)
    
    if not os.path.exists(task_dir):
        print(f"Error: {task_dir} not found")
        return
    
    print(f"\nGenerating plots for {args.task.upper()}...")
    
    # Plot individual model metrics
    for metric in ['acc', 'f1']:
        plot_metric_comparison(args.task, metric, args.results_dir)
    
    # Plot pair metrics
    plot_pair_metrics(args.task, args.results_dir)
    
    print(f"\n✓ All plots saved to {task_dir}/\n")


if __name__ == "__main__":
    main()
