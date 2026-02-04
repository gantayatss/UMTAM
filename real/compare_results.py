"""
Compare Training Results: UMTAM vs Adam
========================================

Analyzes and compares training logs from UMTAM and Adam experiments.

Usage:
    python compare_results.py --umtam_log ./outputs_umtam/training_log.jsonl --adam_log ./outputs_adam/training_log.jsonl
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_log(log_path):
    """Load training log from JSONL file."""
    data = []
    with open(log_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def smooth_curve(values, window=10):
    """Apply moving average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode='valid')
    return smoothed


def plot_comparison(umtam_data, adam_data, output_dir):
    """Create comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    umtam_steps = [d['step'] for d in umtam_data]
    umtam_loss = [d['loss'] for d in umtam_data]
    umtam_ppl = [d['perplexity'] for d in umtam_data if d['perplexity'] != float('inf')]
    
    adam_steps = [d['step'] for d in adam_data]
    adam_loss = [d['loss'] for d in adam_data]
    adam_ppl = [d['perplexity'] for d in adam_data if d['perplexity'] != float('inf')]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Loss comparison
    ax = axes[0, 0]
    ax.plot(umtam_steps, umtam_loss, label='UMTAM', alpha=0.3, color='blue')
    ax.plot(umtam_steps, smooth_curve(umtam_loss), label='UMTAM (smoothed)', color='blue', linewidth=2)
    ax.plot(adam_steps, adam_loss, label='Adam', alpha=0.3, color='red')
    ax.plot(adam_steps, smooth_curve(adam_loss), label='Adam (smoothed)', color='red', linewidth=2)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Perplexity comparison
    ax = axes[0, 1]
    umtam_ppl_steps = umtam_steps[:len(umtam_ppl)]
    adam_ppl_steps = adam_steps[:len(adam_ppl)]
    ax.plot(umtam_ppl_steps, umtam_ppl, label='UMTAM', alpha=0.3, color='blue')
    ax.plot(umtam_ppl_steps, smooth_curve(umtam_ppl), label='UMTAM (smoothed)', color='blue', linewidth=2)
    ax.plot(adam_ppl_steps, adam_ppl, label='Adam', alpha=0.3, color='red')
    ax.plot(adam_ppl_steps, smooth_curve(adam_ppl), label='Adam (smoothed)', color='red', linewidth=2)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Perplexity')
    ax.set_title('Perplexity Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, min(100, max(max(umtam_ppl[:100]), max(adam_ppl[:100]))))
    
    # Plot 3: Training speed
    ax = axes[1, 0]
    umtam_speed = [d.get('steps_per_sec', 0) for d in umtam_data]
    adam_speed = [d.get('steps_per_sec', 0) for d in adam_data]
    
    if umtam_speed and adam_speed:
        ax.plot(umtam_steps, umtam_speed, label='UMTAM', color='blue', alpha=0.6)
        ax.plot(adam_steps, adam_speed, label='Adam', color='red', alpha=0.6)
        ax.axhline(np.mean(umtam_speed), color='blue', linestyle='--', label=f'UMTAM avg: {np.mean(umtam_speed):.2f}')
        ax.axhline(np.mean(adam_speed), color='red', linestyle='--', label=f'Adam avg: {np.mean(adam_speed):.2f}')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Steps/Second')
        ax.set_title('Training Speed Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Memory usage
    ax = axes[1, 1]
    umtam_mem = [d.get('optimizer_memory_mb', 0) for d in umtam_data]
    adam_mem = [d.get('optimizer_memory_mb', 0) for d in adam_data]
    
    if umtam_mem and adam_mem:
        umtam_avg = np.mean(umtam_mem)
        adam_avg = np.mean(adam_mem)
        
        bars = ax.bar(['UMTAM', 'Adam'], [umtam_avg, adam_avg], color=['blue', 'red'], alpha=0.7)
        ax.set_ylabel('Optimizer Memory (MB)')
        ax.set_title('Memory Usage Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f} MB',
                   ha='center', va='bottom')
        
        # Add savings annotation
        savings = ((adam_avg - umtam_avg) / adam_avg) * 100
        ax.text(0.5, 0.95, f'Memory savings: {savings:.1f}%',
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_dir / 'training_comparison.png'}")
    plt.close()


def print_summary(umtam_data, adam_data):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    # Final metrics
    umtam_final_loss = umtam_data[-1]['loss']
    adam_final_loss = adam_data[-1]['loss']
    
    umtam_best_loss = min(d['loss'] for d in umtam_data)
    adam_best_loss = min(d['loss'] for d in adam_data)
    
    print(f"\nFinal Loss:")
    print(f"  UMTAM: {umtam_final_loss:.4f}")
    print(f"  Adam:  {adam_final_loss:.4f}")
    print(f"  Difference: {(umtam_final_loss - adam_final_loss):.4f}")
    
    print(f"\nBest Loss:")
    print(f"  UMTAM: {umtam_best_loss:.4f}")
    print(f"  Adam:  {adam_best_loss:.4f}")
    print(f"  Difference: {(umtam_best_loss - adam_best_loss):.4f}")
    
    # Perplexity
    umtam_final_ppl = umtam_data[-1].get('perplexity', 0)
    adam_final_ppl = adam_data[-1].get('perplexity', 0)
    
    if umtam_final_ppl != float('inf') and adam_final_ppl != float('inf'):
        print(f"\nFinal Perplexity:")
        print(f"  UMTAM: {umtam_final_ppl:.2f}")
        print(f"  Adam:  {adam_final_ppl:.2f}")
    
    # Speed
    umtam_speed = [d.get('steps_per_sec', 0) for d in umtam_data]
    adam_speed = [d.get('steps_per_sec', 0) for d in adam_data]
    
    if umtam_speed and adam_speed:
        print(f"\nAverage Speed (steps/sec):")
        print(f"  UMTAM: {np.mean(umtam_speed):.2f}")
        print(f"  Adam:  {np.mean(adam_speed):.2f}")
        print(f"  Ratio: {np.mean(umtam_speed) / np.mean(adam_speed):.2f}x")
    
    # Memory
    umtam_mem = [d.get('optimizer_memory_mb', 0) for d in umtam_data]
    adam_mem = [d.get('optimizer_memory_mb', 0) for d in adam_data]
    
    if umtam_mem and adam_mem:
        umtam_avg_mem = np.mean(umtam_mem)
        adam_avg_mem = np.mean(adam_mem)
        savings = ((adam_avg_mem - umtam_avg_mem) / adam_avg_mem) * 100
        
        print(f"\nOptimizer Memory (MB):")
        print(f"  UMTAM: {umtam_avg_mem:.1f}")
        print(f"  Adam:  {adam_avg_mem:.1f}")
        print(f"  Savings: {savings:.1f}%")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Compare UMTAM and Adam training results')
    parser.add_argument('--umtam_log', type=str, required=True,
                       help='Path to UMTAM training log')
    parser.add_argument('--adam_log', type=str, required=True,
                       help='Path to Adam training log')
    parser.add_argument('--output_dir', type=str, default='./comparison',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Load logs
    print(f"Loading UMTAM log from {args.umtam_log}...")
    umtam_data = load_log(args.umtam_log)
    
    print(f"Loading Adam log from {args.adam_log}...")
    adam_data = load_log(args.adam_log)
    
    print(f"\nLoaded {len(umtam_data)} UMTAM steps and {len(adam_data)} Adam steps")
    
    # Print summary
    print_summary(umtam_data, adam_data)
    
    # Create plots
    print(f"\nCreating comparison plots...")
    plot_comparison(umtam_data, adam_data, args.output_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
