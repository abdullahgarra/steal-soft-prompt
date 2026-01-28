#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_results.py

Aggregate results across seeds and generate summary statistics.
"""

import os
import csv
import argparse
from collections import defaultdict
from typing import Dict, List
import numpy as np


def load_csv_results(csv_path: str) -> List[Dict]:
    """Load results from CSV file."""
    results = []
    if not os.path.exists(csv_path):
        return results
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric strings to floats
            processed_row = {}
            for key, val in row.items():
                try:
                    processed_row[key] = float(val)
                except (ValueError, TypeError):
                    processed_row[key] = val
            results.append(processed_row)
    return results


def aggregate_by_budget(results: List[Dict]) -> Dict:
    """Aggregate results grouped by budget, averaging over seeds."""
    grouped = defaultdict(lambda: defaultdict(list))
    
    for row in results:
        budget = int(row['budget'])
        for key, val in row.items():
            if key not in ['task', 'seed', 'budget', 'oracle', 'dp']:
                if isinstance(val, (int, float)) and not np.isnan(val):
                    grouped[budget][key].append(val)
    
    # Compute mean and std for each budget
    summary = {}
    for budget, metrics in grouped.items():
        summary[budget] = {}
        for metric, values in metrics.items():
            if len(values) > 0:
                summary[budget][f"{metric}_mean"] = np.mean(values)
                summary[budget][f"{metric}_std"] = np.std(values)
                summary[budget][f"{metric}_n"] = len(values)
    
    return summary


def print_summary_table(summary: Dict, title: str):
    """Print formatted summary table."""
    if not summary:
        print(f"\nNo data for {title}")
        return
    
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    budgets = sorted(summary.keys())
    
    # Key metrics to display
    key_metrics = [
        'victim_acc',
        'stolen_acc',
        'pair_agreement',
        'pair_avg_kl_victim_to_stolen',
        'victim_f1',
        'stolen_f1',
    ]
    
    for metric in key_metrics:
        metric_mean = f"{metric}_mean"
        metric_std = f"{metric}_std"
        
        # Check if this metric exists in any budget
        if not any(metric_mean in summary[b] for b in budgets):
            continue
        
        print(f"\n{metric.upper()}:")
        print(f"  {'Budget':<10} {'Mean':<12} {'Std':<12} {'N':<5}")
        print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*5}")
        
        for budget in budgets:
            if metric_mean in summary[budget]:
                mean_val = summary[budget][metric_mean]
                std_val = summary[budget].get(metric_std, 0.0)
                n = int(summary[budget].get(f"{metric}_n", 0))
                print(f"  {budget:<10} {mean_val:<12.4f} {std_val:<12.4f} {n:<5}")
    
    print(f"{'='*80}\n")


def save_summary_csv(summary: Dict, output_path: str):
    """Save aggregated summary to CSV."""
    if not summary:
        return
    
    # Flatten summary structure
    rows = []
    for budget, metrics in sorted(summary.items()):
        row = {'budget': budget}
        row.update(metrics)
        rows.append(row)
    
    if not rows:
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        fieldnames = ['budget'] + sorted([k for k in rows[0].keys() if k != 'budget'])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ Summary saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=['sst2', 'qqp', 'mnli', 'qnli'])
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    
    task_dir = os.path.join(args.results_dir, args.task)
    
    if not os.path.exists(task_dir):
        print(f"Error: Results directory not found: {task_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"ANALYZING RESULTS FOR: {args.task.upper()}")
    print(f"{'='*80}")
    
    # Process each combination
    for dp_mode in ['nodp', 'dp']:
        for oracle in ['probs', 'hard']:
            csv_path = os.path.join(task_dir, f"results_{dp_mode}_{oracle}.csv")
            
            if not os.path.exists(csv_path):
                print(f"\nWarning: {csv_path} not found, skipping...")
                continue
            
            results = load_csv_results(csv_path)
            
            if not results:
                print(f"\nWarning: No results in {csv_path}")
                continue
            
            summary = aggregate_by_budget(results)
            
            title = f"{args.task.upper()} | DP={dp_mode.upper()} | Oracle={oracle.upper()}"
            print_summary_table(summary, title)
            
            # Save summary CSV
            summary_path = os.path.join(task_dir, f"summary_{dp_mode}_{oracle}.csv")
            save_summary_csv(summary, summary_path)
    
    print(f"\n{'='*80}")
    print(f"Analysis complete for {args.task.upper()}")
    print(f"Check {task_dir}/ for summary CSVs")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
