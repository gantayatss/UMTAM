"""
Quick Hyperparameter Tuning (Fast Version)
===========================================

Faster version with fewer configurations for quick testing.
Now supports both UMTAM and Adam optimizers for comparison.

Usage:
    # UMTAM only (original)
    python quick_tuning.py --max_steps 100
    
    # Both UMTAM and Adam
    python quick_tuning.py --max_steps 100 --compare_optimizers
    
    # Adam only
    python quick_tuning.py --max_steps 100 --optimizer adam
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from src.umtam_optimizer import UMTAMOptimizer
from src.fineweb_loader import FineWebDataset
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse


def quick_experiment(
    data_dir,
    batch_size,
    lr,
    max_steps,
    experiment_name,
    device='cpu',
    optimizer_type='umtam',
    rank=32,
    svd_frequency=50,
):
    """Run a quick training experiment."""
    
    print(f"\nRunning: {experiment_name}")
    print(f"  Optimizer={optimizer_type}, Batch={batch_size}, LR={lr:.0e}, Steps={max_steps}")
    
    # Load
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model = model.to(device)
    
    dataset = FineWebDataset(
        data_path=data_dir,
        tokenizer=tokenizer,
        max_length=512,
        streaming=False,
    )
    
    # Create optimizer based on type
    if optimizer_type == 'umtam':
        optimizer = UMTAMOptimizer(
            model.parameters(),
            lr=lr,
            rank=rank,
            svd_frequency=svd_frequency,
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
    
    # Train
    losses = []
    steps = []
    
    model.train()
    data_iter = iter(dataset)
    
    for step in tqdm(range(max_steps), desc=experiment_name):
        # Get batch
        batch_inputs = []
        batch_masks = []
        
        for _ in range(batch_size):
            try:
                sample = next(data_iter)
            except StopIteration:
                data_iter = iter(dataset)
                sample = next(data_iter)
            
            batch_inputs.append(sample['input_ids'])
            batch_masks.append(sample['attention_mask'])
        
        input_ids = torch.stack(batch_inputs).to(device)
        attention_mask = torch.stack(batch_masks).to(device)
        
        # Train step
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        steps.append(step)
    
    return {
        'name': experiment_name,
        'optimizer': optimizer_type,
        'batch_size': batch_size,
        'lr': lr,
        'steps': steps,
        'losses': losses,
        'final_loss': losses[-1],
        'best_loss': min(losses),
        'avg_final_10': np.mean(losses[-10:]) if len(losses) >= 10 else losses[-1],
        'variance': np.std(losses[-10:]) if len(losses) >= 10 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--data_dir', type=str,
                       default='/Volumes/LaCie/Alirezas/umtam/fineweb_10bt_saved')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--optimizer', type=str, default='umtam',
                       choices=['umtam', 'adam', 'both'],
                       help='Which optimizer to test (umtam, adam, or both)')
    parser.add_argument('--compare_optimizers', action='store_true',
                       help='Compare UMTAM and Adam side-by-side')
    args = parser.parse_args()
    
    # If compare_optimizers flag is set, override optimizer choice
    if args.compare_optimizers:
        args.optimizer = 'both'
    
    print("="*60)
    print("QUICK HYPERPARAMETER TUNING")
    if args.optimizer == 'both':
        print("Comparing: UMTAM vs Adam")
    else:
        print(f"Testing: {args.optimizer.upper()}")
    print("="*60)
    
    # Configurations to test
    configs = [
        # Baseline (unstable)
        {'batch_size': 2, 'lr': 3e-4, 'name': 'Small batch + High LR (unstable)'},
        
        # Only increase batch
        {'batch_size': 8, 'lr': 3e-4, 'name': 'Large batch + High LR'},
        
        # Only decrease LR
        {'batch_size': 2, 'lr': 1e-5, 'name': 'Small batch + Low LR'},
        
        # Both (stable)
        {'batch_size': 8, 'lr': 1e-5, 'name': 'Large batch + Low LR (stable)'},
    ]
    
    results = []
    
    # Determine which optimizers to test
    if args.optimizer == 'both':
        optimizers_to_test = ['umtam', 'adam']
    else:
        optimizers_to_test = [args.optimizer]
    
    for config in configs:
        for opt_type in optimizers_to_test:
            exp_name = f"{opt_type.upper()}: {config['name']}" if args.optimizer == 'both' else config['name']
            result = quick_experiment(
                data_dir=args.data_dir,
                batch_size=config['batch_size'],
                lr=config['lr'],
                max_steps=args.max_steps,
                experiment_name=exp_name,
                device=args.device,
                optimizer_type=opt_type,
            )
            results.append(result)
    
    # Plot comparison
    if args.optimizer == 'both':
        # Comparison mode - 2x3 grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('UMTAM vs Adam: Hyperparameter Comparison', fontsize=16, fontweight='bold')
        
        # Separate results by optimizer
        umtam_results = [r for r in results if r['optimizer'] == 'umtam']
        adam_results = [r for r in results if r['optimizer'] == 'adam']
        
        # Plot 1: All experiments
        ax = axes[0, 0]
        for result in results:
            linestyle = '-' if result['optimizer'] == 'umtam' else '--'
            color = '#1f77b4' if result['optimizer'] == 'umtam' else '#ff7f0e'
            ax.plot(result['steps'], result['losses'], 
                   label=result['name'], linewidth=2, alpha=0.7, 
                   linestyle=linestyle, color=color)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss')
        ax.set_title('All Configurations')
        ax.legend(fontsize=6, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: UMTAM only (smoothed)
        ax = axes[0, 1]
        for result in umtam_results:
            if len(result['losses']) > 10:
                smoothed = np.convolve(result['losses'], np.ones(10)/10, mode='valid')
                label = result['name'].replace('UMTAM: ', '')
                ax.plot(result['steps'][:len(smoothed)], smoothed,
                       label=label, linewidth=2)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss (smoothed)')
        ax.set_title('UMTAM Only')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Adam only (smoothed)
        ax = axes[0, 2]
        for result in adam_results:
            if len(result['losses']) > 10:
                smoothed = np.convolve(result['losses'], np.ones(10)/10, mode='valid')
                label = result['name'].replace('Adam: ', '')
                ax.plot(result['steps'][:len(smoothed)], smoothed,
                       label=label, linewidth=2)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss (smoothed)')
        ax.set_title('Adam Only')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Final loss comparison
        ax = axes[1, 0]
        names = [r['name'].split(': ')[-1][:20] for r in results]
        final_losses = [r['final_loss'] for r in results]
        colors = ['#1f77b4' if r['optimizer'] == 'umtam' else '#ff7f0e' for r in results]
        
        ax.bar(range(len(names)), final_losses, color=colors, alpha=0.7)
        ax.set_ylabel('Final Loss')
        ax.set_title('Final Loss Comparison')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=6)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#1f77b4', label='UMTAM'),
                          Patch(facecolor='#ff7f0e', label='Adam')]
        ax.legend(handles=legend_elements)
        
        # Plot 5: Best loss comparison
        ax = axes[1, 1]
        best_losses = [r['best_loss'] for r in results]
        
        ax.bar(range(len(names)), best_losses, color=colors, alpha=0.7)
        ax.set_ylabel('Best Loss')
        ax.set_title('Best Loss Achieved')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=6)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 6: Variance comparison
        ax = axes[1, 2]
        variances = [r['variance'] for r in results]
        
        ax.bar(range(len(names)), variances, color=colors, alpha=0.7)
        ax.set_ylabel('Variance (last 10 steps)')
        ax.set_title('Training Stability')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=6)
        ax.grid(True, alpha=0.3, axis='y')
        
    else:
        # Single optimizer mode - original 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{args.optimizer.upper()} Hyperparameter Effect Comparison', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: All experiments
        ax = axes[0, 0]
        for result in results:
            ax.plot(result['steps'], result['losses'], 
                   label=result['name'], linewidth=2, alpha=0.7)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss')
        ax.set_title('All Configurations')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Batch size effect
        ax = axes[0, 1]
        batch_results = [results[0], results[1]]  # Same LR, different batch
        for result in batch_results:
            # Smoothed
            if len(result['losses']) > 10:
                smoothed = np.convolve(result['losses'], np.ones(10)/10, mode='valid')
                ax.plot(result['steps'][:len(smoothed)], smoothed,
                       label=f"Batch={result['batch_size']}", linewidth=2)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss (smoothed)')
        ax.set_title('Batch Size Effect (LR fixed at 3e-4)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Learning rate effect
        ax = axes[1, 0]
        lr_results = [results[0], results[2]]  # Same batch, different LR
        for result in lr_results:
            # Smoothed
            if len(result['losses']) > 10:
                smoothed = np.convolve(result['losses'], np.ones(10)/10, mode='valid')
                ax.plot(result['steps'][:len(smoothed)], smoothed,
                       label=f"LR={result['lr']:.0e}", linewidth=2)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss (smoothed)')
        ax.set_title('Learning Rate Effect (Batch fixed at 2)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Final loss comparison
        ax = axes[1, 1]
        names = [r['name'].split('(')[0].strip() for r in results]
        final_losses = [r['final_loss'] for r in results]
        best_losses = [r['best_loss'] for r in results]
        
        x = np.arange(len(names))
        width = 0.35
        
        ax.bar(x - width/2, final_losses, width, label='Final Loss', alpha=0.8)
        ax.bar(x + width/2, best_losses, width, label='Best Loss', alpha=0.8)
        
        ax.set_ylabel('Loss')
        ax.set_title('Final & Best Loss Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('quick_tuning_results.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Saved: quick_tuning_results.png")
    
    # Print summary
    print("\n" + "="*100)
    print("RESULTS SUMMARY")
    print("="*100)
    
    if args.optimizer == 'both':
        # Comparison summary
        print(f"{'Optimizer':<10} {'Config':<35} {'Final Loss':<12} {'Best Loss':<12} {'Variance':<10}")
        print("-"*100)
        for result in results:
            config_name = result['name'].split(': ')[-1][:30]
            print(f"{result['optimizer'].upper():<10} {config_name:<35} "
                  f"{result['final_loss']:<12.4f} {result['best_loss']:<12.4f} "
                  f"{result['variance']:<10.4f}")
        
        print("\n" + "="*100)
        # Find best for each optimizer
        umtam_best = min([r for r in results if r['optimizer'] == 'umtam'], 
                        key=lambda x: x['final_loss'])
        adam_best = min([r for r in results if r['optimizer'] == 'adam'], 
                       key=lambda x: x['final_loss'])
        
        print("🏆 BEST CONFIGURATIONS:")
        print(f"  UMTAM: B={umtam_best['batch_size']}, LR={umtam_best['lr']:.0e} → Loss={umtam_best['final_loss']:.4f}")
        print(f"  Adam:  B={adam_best['batch_size']}, LR={adam_best['lr']:.0e} → Loss={adam_best['final_loss']:.4f}")
        
        gap = ((umtam_best['final_loss'] - adam_best['final_loss']) / adam_best['final_loss'] * 100)
        print(f"\n  Performance Gap: {gap:+.1f}% ({'UMTAM better' if gap < 0 else 'Adam better'})")
        print("="*100)
    else:
        # Single optimizer summary
        for result in results:
            print(f"\n{result['name']}")
            print(f"  Batch={result['batch_size']}, LR={result['lr']:.0e}")
            print(f"  Final Loss: {result['final_loss']:.4f}")
            print(f"  Best Loss:  {result['best_loss']:.4f}")
            print(f"  Variance:   {result['variance']:.4f}")
    
    # Save JSON
    with open('quick_tuning_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved: quick_tuning_results.json")
    
    print("\n" + "="*60)
    print("✅ QUICK TUNING COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
