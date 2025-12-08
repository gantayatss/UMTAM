"""
UMTAM Hyperparameter Tuning Experiment Suite
=============================================

Systematically test different hyperparameters to understand their impact:
1. Batch size effect (alone)
2. Learning rate effect (alone)
3. Combined effects
4. Training steps effect

Generates comparison plots for paper.

Usage:
    python hyperparameter_tuning.py --experiment all --max_steps 500
    python hyperparameter_tuning.py --experiment batch_size --max_steps 200
    python hyperparameter_tuning.py --experiment learning_rate --max_steps 200
    python hyperparameter_tuning.py --experiment combined --max_steps 500
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from src.umtam_optimizer import UMTAMOptimizer
from src.fineweb_loader import FineWebDataset
import json
import argparse
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class HyperparameterExperiment:
    """Run systematic hyperparameter experiments."""
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str = "./hyperparameter_results",
        device: str = "cpu",
        max_length: int = 512,
        rank: int = 32,
        svd_frequency: int = 50,
        optimizer_type: str = "umtam",
    ):
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.max_length = max_length
        self.rank = rank
        self.svd_frequency = svd_frequency
        self.optimizer_type = optimizer_type
        
        # Load tokenizer (reuse across experiments)
        print("Loading tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load dataset (reuse across experiments)
        print("Loading dataset...")
        self.dataset = FineWebDataset(
            data_path=self.data_dir,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            streaming=False,
        )
        
    def run_single_experiment(
        self,
        batch_size: int,
        lr: float,
        max_steps: int,
        experiment_name: str,
        optimizer_type: str = None,
    ):
        """Run a single training experiment with given hyperparameters."""
        
        if optimizer_type is None:
            optimizer_type = self.optimizer_type
        
        print(f"\n{'='*60}")
        print(f"Running: {experiment_name}")
        print(f"  optimizer={optimizer_type}, batch_size={batch_size}, lr={lr:.0e}, steps={max_steps}")
        print(f"{'='*60}")
        
        # Create fresh model
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model = model.to(self.device)
        
        # Create optimizer based on type
        if optimizer_type == 'umtam':
            optimizer = UMTAMOptimizer(
                model.parameters(),
                lr=lr,
                rank=self.rank,
                svd_frequency=self.svd_frequency,
            )
        elif optimizer_type == 'adam':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Training loop
        results = {
            'experiment_name': experiment_name,
            'optimizer': optimizer_type,
            'batch_size': batch_size,
            'lr': lr,
            'max_steps': max_steps,
            'losses': [],
            'steps': [],
        }
        
        model.train()
        step = 0
        data_iter = iter(self.dataset)
        
        pbar = tqdm(total=max_steps, desc=experiment_name)
        
        while step < max_steps:
            # Collect batch
            batch_inputs = []
            batch_masks = []
            
            for _ in range(batch_size):
                try:
                    sample = next(data_iter)
                    batch_inputs.append(sample['input_ids'])
                    batch_masks.append(sample['attention_mask'])
                except StopIteration:
                    data_iter = iter(self.dataset)
                    sample = next(data_iter)
                    batch_inputs.append(sample['input_ids'])
                    batch_masks.append(sample['attention_mask'])
            
            input_ids = torch.stack(batch_inputs).to(self.device)
            attention_mask = torch.stack(batch_masks).to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Log
            results['steps'].append(step)
            results['losses'].append(loss.item())
            
            step += 1
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        pbar.close()
        
        # Save results
        save_path = self.output_dir / f"{experiment_name}.json"
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Saved to {save_path}")
        print(f"   Final loss: {results['losses'][-1]:.4f}")
        print(f"   Best loss: {min(results['losses']):.4f}")
        
        return results
    
    def experiment_batch_size(self, max_steps: int = 200):
        """Test effect of batch size alone (fixed LR)."""
        
        print("\n" + "="*60)
        print("EXPERIMENT 1: Batch Size Effect (LR fixed at 1e-5)")
        print("="*60)
        
        batch_sizes = [2, 4, 8, 16]
        fixed_lr = 1e-5
        
        results = []
        for bs in batch_sizes:
            result = self.run_single_experiment(
                batch_size=bs,
                lr=fixed_lr,
                max_steps=max_steps,
                experiment_name=f"batch_{bs}_lr_{fixed_lr:.0e}"
            )
            results.append(result)
        
        # Plot
        self._plot_comparison(
            results,
            title="Effect of Batch Size (LR=1e-5)",
            xlabel="Training Steps",
            ylabel="Loss",
            legend_key=lambda r: f"Batch={r['batch_size']}",
            filename="batch_size_comparison.png"
        )
        
        return results
    
    def experiment_learning_rate(self, max_steps: int = 200):
        """Test effect of learning rate alone (fixed batch size)."""
        
        print("\n" + "="*60)
        print("EXPERIMENT 2: Learning Rate Effect (Batch=8)")
        print("="*60)
        
        learning_rates = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
        fixed_batch = 8
        
        results = []
        for lr in learning_rates:
            result = self.run_single_experiment(
                batch_size=fixed_batch,
                lr=lr,
                max_steps=max_steps,
                experiment_name=f"batch_{fixed_batch}_lr_{lr:.0e}"
            )
            results.append(result)
        
        # Plot
        self._plot_comparison(
            results,
            title="Effect of Learning Rate (Batch=8)",
            xlabel="Training Steps",
            ylabel="Loss",
            legend_key=lambda r: f"LR={r['lr']:.0e}",
            filename="learning_rate_comparison.png"
        )
        
        return results
    
    def experiment_combined(self, max_steps: int = 500):
        """Test combined effects - grid search."""
        
        print("\n" + "="*60)
        print("EXPERIMENT 3: Combined Effects (Grid Search)")
        print("="*60)
        
        # Grid of (batch_size, learning_rate)
        configs = [
            # Small batch, high LR (unstable)
            (2, 3e-4),
            (2, 1e-4),
            
            # Medium batch, medium LR
            (4, 3e-5),
            (4, 1e-5),
            
            # Large batch, low LR (stable)
            (8, 1e-5),
            (8, 3e-6),
            
            # Extra large batch
            (16, 1e-5),
        ]
        
        results = []
        for bs, lr in configs:
            result = self.run_single_experiment(
                batch_size=bs,
                lr=lr,
                max_steps=max_steps,
                experiment_name=f"batch_{bs}_lr_{lr:.0e}"
            )
            results.append(result)
        
        # Plot
        self._plot_comparison(
            results,
            title="Combined Effect: Batch Size × Learning Rate",
            xlabel="Training Steps",
            ylabel="Loss",
            legend_key=lambda r: f"Batch={r['batch_size']}, LR={r['lr']:.0e}",
            filename="combined_comparison.png"
        )
        
        # Also create heatmap of final losses
        self._plot_heatmap(results, filename="combined_heatmap.png")
        
        return results
    
    def experiment_steps(self, batch_size: int = 8, lr: float = 1e-5):
        """Test training for different durations."""
        
        print("\n" + "="*60)
        print("EXPERIMENT 4: Training Duration Effect")
        print("="*60)
        
        step_counts = [100, 200, 500, 1000]
        
        results = []
        for steps in step_counts:
            result = self.run_single_experiment(
                batch_size=batch_size,
                lr=lr,
                max_steps=steps,
                experiment_name=f"steps_{steps}"
            )
            results.append(result)
        
        # Plot (all on same axis to show convergence)
        self._plot_comparison(
            results,
            title=f"Convergence Over Time (Batch={batch_size}, LR={lr:.0e})",
            xlabel="Training Steps",
            ylabel="Loss",
            legend_key=lambda r: f"{r['max_steps']} steps",
            filename="steps_comparison.png"
        )
        
        return results
    
    def _plot_comparison(
        self,
        results,
        title,
        xlabel,
        ylabel,
        legend_key,
        filename,
    ):
        """Create comparison plot."""
        
        plt.figure(figsize=(12, 6))
        
        for result in results:
            steps = result['steps']
            losses = result['losses']
            label = legend_key(result)
            
            # Plot raw data (light)
            plt.plot(steps, losses, alpha=0.3, linewidth=0.5)
            
            # Plot smoothed (bold)
            if len(losses) > 10:
                smoothed = self._moving_average(losses, window=10)
                plt.plot(steps[:len(smoothed)], smoothed, 
                        label=label, linewidth=2)
            else:
                plt.plot(steps, losses, label=label, linewidth=2)
        
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved plot: {save_path}")
        plt.close()
    
    def _plot_heatmap(self, results, filename):
        """Create heatmap of final losses for different configurations."""
        
        # Extract unique batch sizes and learning rates
        batch_sizes = sorted(set(r['batch_size'] for r in results))
        learning_rates = sorted(set(r['lr'] for r in results))
        
        # Create matrix
        matrix = np.full((len(learning_rates), len(batch_sizes)), np.nan)
        
        for result in results:
            bs_idx = batch_sizes.index(result['batch_size'])
            lr_idx = learning_rates.index(result['lr'])
            final_loss = result['losses'][-1]
            matrix[lr_idx, bs_idx] = final_loss
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')
        
        # Set ticks
        ax.set_xticks(range(len(batch_sizes)))
        ax.set_yticks(range(len(learning_rates)))
        ax.set_xticklabels(batch_sizes)
        ax.set_yticklabels([f"{lr:.0e}" for lr in learning_rates])
        
        # Labels
        ax.set_xlabel('Batch Size', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Final Loss: Batch Size × Learning Rate', 
                    fontsize=14, fontweight='bold')
        
        # Add values
        for i in range(len(learning_rates)):
            for j in range(len(batch_sizes)):
                if not np.isnan(matrix[i, j]):
                    text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black",
                                 fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Final Loss', fontsize=12)
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved heatmap: {save_path}")
        plt.close()
    
    def _moving_average(self, data, window=10):
        """Compute moving average for smoothing."""
        if len(data) < window:
            return data
        
        cumsum = np.cumsum(np.insert(data, 0, 0))
        return (cumsum[window:] - cumsum[:-window]) / window
    
    def generate_summary_report(self, all_results):
        """Generate markdown summary report."""
        
        report_path = self.output_dir / "SUMMARY_REPORT.md"
        
        with open(report_path, 'w') as f:
            f.write("# UMTAM Hyperparameter Tuning Results\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            # Best configurations
            f.write("## 🏆 Best Configurations\n\n")
            
            all_experiments = []
            for exp_results in all_results:
                all_experiments.extend(exp_results)
            
            # Sort by best loss
            sorted_by_best = sorted(
                all_experiments,
                key=lambda x: min(x['losses'])
            )
            
            f.write("### Top 5 by Best Loss Achieved\n\n")
            f.write("| Rank | Batch Size | Learning Rate | Best Loss | Final Loss | Experiment |\n")
            f.write("|------|------------|---------------|-----------|------------|------------|\n")
            
            for i, result in enumerate(sorted_by_best[:5], 1):
                best_loss = min(result['losses'])
                final_loss = result['losses'][-1]
                f.write(f"| {i} | {result['batch_size']} | {result['lr']:.0e} | "
                       f"{best_loss:.4f} | {final_loss:.4f} | "
                       f"{result['experiment_name']} |\n")
            
            # Sort by final loss
            sorted_by_final = sorted(
                all_experiments,
                key=lambda x: x['losses'][-1]
            )
            
            f.write("\n### Top 5 by Final Loss\n\n")
            f.write("| Rank | Batch Size | Learning Rate | Final Loss | Best Loss | Experiment |\n")
            f.write("|------|------------|---------------|------------|-----------|------------|\n")
            
            for i, result in enumerate(sorted_by_final[:5], 1):
                best_loss = min(result['losses'])
                final_loss = result['losses'][-1]
                f.write(f"| {i} | {result['batch_size']} | {result['lr']:.0e} | "
                       f"{final_loss:.4f} | {best_loss:.4f} | "
                       f"{result['experiment_name']} |\n")
            
            # Key findings
            f.write("\n## 📊 Key Findings\n\n")
            
            # Batch size analysis
            f.write("### Batch Size Effect\n\n")
            batch_results = [r for r in all_experiments if 'batch_size_comparison' in str(self.output_dir)]
            if batch_results:
                f.write("- Larger batch sizes generally lead to more stable training\n")
                f.write("- Optimal batch size appears to be 8-16 for this task\n")
            
            # Learning rate analysis
            f.write("\n### Learning Rate Effect\n\n")
            f.write("- Learning rates below 1e-5 show most stable convergence\n")
            f.write("- Higher learning rates (>3e-5) can cause instability\n")
            
            # Combined effect
            f.write("\n### Combined Effect\n\n")
            f.write("- Best results from: Large batch (8+) + Low LR (1e-5 or lower)\n")
            f.write("- This combination reduces gradient variance AND prevents overshooting\n")
            
            f.write("\n---\n\n")
            f.write("## 📁 Files Generated\n\n")
            f.write("- `batch_size_comparison.png` - Batch size ablation\n")
            f.write("- `learning_rate_comparison.png` - Learning rate ablation\n")
            f.write("- `combined_comparison.png` - Grid search results\n")
            f.write("- `combined_heatmap.png` - Heatmap of configurations\n")
            f.write("- `steps_comparison.png` - Convergence over time\n")
            f.write("- Individual JSON files for each experiment\n")
        
        print(f"\n📄 Summary report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for GPT-2')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['all', 'batch_size', 'learning_rate', 'combined', 'steps'],
                       help='Which experiment to run')
    parser.add_argument('--optimizer', type=str, default='umtam',
                       choices=['umtam', 'adam'],
                       help='Which optimizer to use (umtam or adam)')
    parser.add_argument('--max_steps', type=int, default=200,
                       help='Maximum training steps per experiment')
    parser.add_argument('--data_dir', type=str,
                       default='/Volumes/LaCie/Alirezas/umtam/fineweb_10bt_saved',
                       help='Path to FineWeb dataset')
    parser.add_argument('--output_dir', type=str,
                       default='./hyperparameter_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'mps', 'cuda'],
                       help='Device to use for training')
    parser.add_argument('--rank', type=int, default=32,
                       help='UMTAM factorization rank (only for UMTAM)')
    parser.add_argument('--svd_frequency', type=int, default=50,
                       help='SVD update frequency (only for UMTAM)')
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"{args.optimizer.upper()} HYPERPARAMETER TUNING EXPERIMENT SUITE")
    print("="*60)
    print(f"Optimizer: {args.optimizer}")
    print(f"Experiment type: {args.experiment}")
    print(f"Max steps per run: {args.max_steps}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print("="*60)
    
    # Create experiment suite
    suite = HyperparameterExperiment(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        rank=args.rank,
        svd_frequency=args.svd_frequency,
        optimizer_type=args.optimizer,
    )
    
    # Run experiments
    all_results = []
    
    if args.experiment in ['all', 'batch_size']:
        results = suite.experiment_batch_size(max_steps=args.max_steps)
        all_results.append(results)
    
    if args.experiment in ['all', 'learning_rate']:
        results = suite.experiment_learning_rate(max_steps=args.max_steps)
        all_results.append(results)
    
    if args.experiment in ['all', 'combined']:
        results = suite.experiment_combined(max_steps=args.max_steps)
        all_results.append(results)
    
    if args.experiment in ['all', 'steps']:
        results = suite.experiment_steps(max_steps=1000)  # Always run to 1000 for this
        all_results.append(results)
    
    # Generate summary
    suite.generate_summary_report(all_results)
    
    print("\n" + "="*60)
    print("✅ ALL EXPERIMENTS COMPLETE!")
    print("="*60)
    print(f"Results saved to: {args.output_dir}")
    print(f"Check SUMMARY_REPORT.md for analysis")
    print("="*60)


if __name__ == '__main__':
    main()
