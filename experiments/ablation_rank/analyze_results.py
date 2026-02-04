#!/usr/bin/env python3
"""
Analyze UMTAM Ablation Results
Creates plots and summary table
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = str(Path(__file__).parent / "results")

def load_results():
    """Load all rank results."""
    results = {}
    output_path = Path(OUTPUT_DIR)
    
    for results_file in sorted(output_path.glob('results_rank_*.json')):
        # Extract rank from filename
        rank = int(results_file.stem.split('_')[-1])
        
        with open(results_file) as f:
            results[rank] = json.load(f)
            print(f"Loaded rank {rank}: {len(results[rank]['train_losses'])} steps")
    
    return results

def create_plots(results):
    """Create MoFaSGD-style comparison plots."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Val loss vs steps
    for rank in sorted(results.keys()):
        data = results[rank]
        ax1.plot(data['eval_steps'], data['val_losses'],
                label=f'UMTAM (r={rank})', linewidth=2, marker='o', markersize=4)
    
    ax1.set_xlabel('Steps', fontsize=12)
    ax1.set_ylabel('Validation Loss', fontsize=12)
    ax1.set_title('UMTAM: Validation Loss vs Training Steps', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Val loss vs time
    for rank in sorted(results.keys()):
        data = results[rank]
        # Compute cumulative times for eval steps
        eval_times = []
        for eval_step in data['eval_steps']:
            if eval_step <= len(data['step_times']):
                cumulative_time = sum(data['step_times'][:eval_step])
                eval_times.append(cumulative_time)
        
        # Only plot if we have matching eval times
        if len(eval_times) == len(data['val_losses']):
            ax2.plot(eval_times, data['val_losses'],
                    label=f'UMTAM (r={rank})', linewidth=2, marker='o', markersize=4)
    
    ax2.set_xlabel('Wall-clock Time (seconds)', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('UMTAM: Validation Loss vs Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = Path(OUTPUT_DIR) / 'ablation_plots.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plots to: {plot_path}")
    plt.close()

def create_summary_table(results):
    """Create MoFaSGD-style summary table."""
    
    print("\n" + "="*80)
    print("UMTAM Ablation Study - Summary Table")
    print("="*80)
    print(f"{'Rank':<8} {'Val Loss':<12} {'Runtime(s)':<12} {'Throughput':<15} {'Memory(MB)':<12}")
    print("-"*80)
    
    summary = []
    for rank in sorted(results.keys()):
        data = results[rank]
        row = {
            'Rank': rank,
            'Val Loss': data.get('final_val_loss', 0),
            'Runtime (s)': data.get('runtime_seconds', 0),
            'Throughput': data.get('throughput', 0),
            'Memory (MB)': data.get('memory_mb', 0)
        }
        summary.append(row)
        
        print(f"{row['Rank']:<8} "
              f"{row['Val Loss']:<12.4f} "
              f"{row['Runtime (s)']:<12.1f} "
              f"{row['Throughput']:<15.1f} "
              f"{row['Memory (MB)']:<12.1f}")
    
    print("="*80 + "\n")
    
    # Save to file
    table_path = Path(OUTPUT_DIR) / 'summary_table.txt'
    with open(table_path, 'w') as f:
        f.write("UMTAM Ablation Study - Summary Table\n")
        f.write("="*80 + "\n")
        f.write(f"{'Rank':<8} {'Val Loss':<12} {'Runtime(s)':<12} {'Throughput':<15} {'Memory(MB)':<12}\n")
        f.write("-"*80 + "\n")
        for row in summary:
            f.write(f"{row['Rank']:<8} "
                   f"{row['Val Loss']:<12.4f} "
                   f"{row['Runtime (s)']:<12.1f} "
                   f"{row['Throughput']:<15.1f} "
                   f"{row['Memory (MB)']:<12.1f}\n")
    
    print(f"Saved summary to: {table_path}")

def main():
    print("Loading ablation results...")
    results = load_results()
    
    if not results:
        print("No results found! Run training scripts first.")
        print(f"Looking in: {OUTPUT_DIR}")
        return
    
    print(f"\nFound results for ranks: {sorted(results.keys())}")
    
    print("\nCreating plots...")
    create_plots(results)
    
    print("\nCreating summary table...")
    create_summary_table(results)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == '__main__':
    main()
