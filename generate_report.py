#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_report.py

Generate comprehensive analysis report for all tasks.
"""

import os
import subprocess
import argparse


def run_analysis(task, results_dir):
    """Run analysis for a single task."""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {task.upper()}")
    print(f"{'='*80}")
    
    # Run analyze_results.py
    cmd = ["python", "analyze_results.py", "--task", task, "--results_dir", results_dir]
    subprocess.run(cmd)
    
    # Generate plots if matplotlib is available
    try:
        import matplotlib
        cmd = ["python", "plot_results.py", "--task", task, "--results_dir", results_dir]
        subprocess.run(cmd)
    except ImportError:
        print("\nNote: matplotlib not available, skipping plots")


def generate_summary_table(results_dir):
    """Generate a cross-task comparison table."""
    import csv
    from collections import defaultdict
    import numpy as np
    
    print(f"\n{'='*80}")
    print("CROSS-TASK SUMMARY (Budget=1000, averaged over seeds)")
    print(f"{'='*80}\n")
    
    tasks = ['sst2', 'qqp', 'mnli', 'qnli']
    budget = 1000
    
    # Table headers
    print(f"{'Task':<10} {'Config':<20} {'Victim Acc':<12} {'Stolen Acc':<12} {'Agreement':<12} {'KL Div':<12}")
    print(f"{'-'*10} {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    
    for task in tasks:
        for dp_mode in ['nodp', 'dp']:
            for oracle in ['probs', 'hard']:
                csv_path = os.path.join(results_dir, task, f"results_{dp_mode}_{oracle}.csv")
                
                if not os.path.exists(csv_path):
                    continue
                
                # Load and filter for budget=1000
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = [r for r in reader if int(float(r['budget'])) == budget]
                
                if not rows:
                    continue
                
                # Compute means
                victim_acc = np.mean([float(r['victim_acc']) for r in rows])
                stolen_acc = np.mean([float(r['stolen_acc']) for r in rows])
                agreement = np.mean([float(r['pair_agreement']) for r in rows])
                kl_div = np.mean([float(r['pair_avg_kl_victim_to_stolen']) for r in rows])
                
                config = f"{'DP' if dp_mode == 'dp' else 'No-DP'}/{oracle}"
                
                print(f"{task:<10} {config:<20} {victim_acc:>11.4f} {stolen_acc:>11.4f} {agreement:>11.4f} {kl_div:>11.6f}")
        
        print()  # Blank line between tasks
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--skip_plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()
    
    tasks = ['sst2', 'qqp', 'mnli', 'qnli']
    
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS ANALYSIS")
    print("="*80)
    
    # Run analysis for each task
    for task in tasks:
        task_dir = os.path.join(args.results_dir, task)
        if os.path.exists(task_dir):
            run_analysis(task, args.results_dir)
        else:
            print(f"\nWarning: No results found for {task}")
    
    # Generate cross-task summary
    try:
        generate_summary_table(args.results_dir)
    except Exception as e:
        print(f"\nWarning: Could not generate summary table: {e}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults organized in: {args.results_dir}/")
    print("Each task directory contains:")
    print("  - results_*.csv: Raw experimental data")
    print("  - summary_*.csv: Aggregated statistics")
    print("  - plot_*.png: Visualizations (if matplotlib available)")
    print("\n")


if __name__ == "__main__":
    main()
