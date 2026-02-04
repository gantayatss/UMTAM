"""
Train GPT-2 with UMTAM Optimizer on FineWeb Dataset
====================================================

This script trains a GPT-2 Small model using the UMTAM optimizer
on the FineWeb-10BT dataset.

Usage:
    python train_gpt2_umtam.py --config config.yaml
"""

import sys
from pathlib import Path
# Add parent directory to path to import from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from src.umtam_optimizer import UMTAMOptimizer
from src.fineweb_loader import FineWebDataset
import argparse
import time
import json
import os
from pathlib import Path
import math
from typing import Dict, Optional
from tqdm import tqdm


class Trainer:
    """Training manager for GPT-2 with UMTAM."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: UMTAMOptimizer,
        train_dataset,
        config: Dict,
        device: str = 'cpu',
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.config = config
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Create output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.log_file = self.output_dir / 'training_log.jsonl'
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Perform a single training step."""
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        loss = outputs.loss
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Optimizer step
        self.optimizer.step()
        
        return loss.item()
    
    def log_metrics(self, metrics: Dict, step: int):
        """Log metrics to file."""
        log_entry = {
            'step': step,
            'epoch': self.epoch,
            **metrics,
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def save_checkpoint(self, step: int, loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'step': step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config,
        }
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_step_{step}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint to {best_path}")
    
    def train(self):
        """Main training loop."""
        config = self.config
        
        # Training parameters
        max_steps = config['max_steps']
        batch_size = config['batch_size']
        gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        log_interval = config.get('log_interval', 10)
        save_interval = config.get('save_interval', 1000)
        eval_interval = config.get('eval_interval', 500)
        
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}")
        print(f"Max steps: {max_steps}")
        print(f"Batch size: {batch_size}")
        print(f"Gradient accumulation: {gradient_accumulation_steps}")
        print(f"Device: {self.device}")
        print(f"Output dir: {self.output_dir}")
        print(f"{'='*60}\n")
        
        # Training loop
        step = 0
        running_loss = 0.0
        start_time = time.time()
        
        # Iterate through dataset
        data_iter = iter(self.train_dataset)
        
        # Accumulate batches
        accumulated_batches = []
        
        pbar = tqdm(total=max_steps, desc="Training")
        
        while step < max_steps:
            try:
                # Collect batches for gradient accumulation
                for _ in range(gradient_accumulation_steps):
                    sample = next(data_iter)
                    
                    # Create batch (in this case, batch_size=1 per sample)
                    # You can modify this to collect multiple samples
                    batch = {
                        'input_ids': sample['input_ids'].unsqueeze(0),
                        'attention_mask': sample['attention_mask'].unsqueeze(0),
                    }
                    accumulated_batches.append(batch)
                
                # Combine accumulated batches
                combined_batch = {
                    'input_ids': torch.cat([b['input_ids'] for b in accumulated_batches], dim=0),
                    'attention_mask': torch.cat([b['attention_mask'] for b in accumulated_batches], dim=0),
                }
                accumulated_batches = []
                
                # Training step
                loss = self.train_step(combined_batch)
                running_loss += loss
                step += 1
                self.global_step = step
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({'loss': f'{loss:.4f}'})
                
                # Logging
                if step % log_interval == 0:
                    avg_loss = running_loss / log_interval
                    elapsed = time.time() - start_time
                    steps_per_sec = log_interval / elapsed
                    
                    # Get memory usage
                    mem_stats = self.optimizer.get_memory_usage()
                    
                    metrics = {
                        'loss': avg_loss,
                        'perplexity': math.exp(avg_loss) if avg_loss < 10 else float('inf'),
                        'steps_per_sec': steps_per_sec,
                        'optimizer_memory_mb': mem_stats['state_memory_mb'],
                        'total_memory_mb': mem_stats['total_memory_mb'],
                    }
                    
                    self.log_metrics(metrics, step)
                    
                    print(f"\nStep {step}/{max_steps} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"PPL: {metrics['perplexity']:.2f} | "
                          f"Speed: {steps_per_sec:.2f} steps/s | "
                          f"Opt Mem: {mem_stats['state_memory_mb']:.1f} MB")
                    
                    running_loss = 0.0
                    start_time = time.time()
                
                # Checkpointing
                if step % save_interval == 0:
                    is_best = loss < self.best_loss
                    if is_best:
                        self.best_loss = loss
                    self.save_checkpoint(step, loss, is_best)
                
            except StopIteration:
                print("Dataset exhausted, restarting...")
                data_iter = iter(self.train_dataset)
                self.epoch += 1
                
            except KeyboardInterrupt:
                print("\n\nTraining interrupted by user!")
                print("Saving checkpoint...")
                self.save_checkpoint(step, loss)
                break
        
        pbar.close()
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Total steps: {step}")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Train GPT-2 with UMTAM')
    parser.add_argument('--data_dir', type=str, 
                       default='/Volumes/LaCie/Alirezas/umtam/fineweb_10bt_saved',
                       help='Path to FineWeb dataset cache')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--model_name', type=str, default='gpt2',
                       help='Model name or path (gpt2, gpt2-medium, etc.)')
    
    # Training hyperparameters
    parser.add_argument('--max_steps', type=int, default=10000,
                       help='Maximum training steps')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size (effective batch size with accumulation)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--max_length', type=int, default=1024,
                       help='Maximum sequence length')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    
    # UMTAM hyperparameters
    parser.add_argument('--rank', type=int, default=32,
                       help='Rank for momentum factorization')
    parser.add_argument('--beta1', type=float, default=0.9,
                       help='First moment decay')
    parser.add_argument('--beta2', type=float, default=0.999,
                       help='Second moment decay')
    parser.add_argument('--gamma', type=float, default=0.9,
                       help='Error feedback decay')
    parser.add_argument('--eps', type=float, default=1e-8,
                       help='Regularization constant')
    parser.add_argument('--svd_frequency', type=int, default=1,
                       help='SVD computation frequency')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cpu, mps, cuda)')
    
    # Logging
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=1000,
                       help='Checkpoint saving interval')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"Loading model: {args.model_name}...")
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {total_params:,} parameters ({total_params/1e6:.1f}M)")
    
    # Create dataset
    print("Creating dataset...")
    train_dataset = FineWebDataset(
        data_path=args.data_dir,  # Now works with saved datasets!
        tokenizer=tokenizer,
        max_length=args.max_length,
        streaming=False,  # Use saved data (faster!)
    )
    
    # Create optimizer
    print("Creating UMTAM optimizer...")
    optimizer = UMTAMOptimizer(
        model.parameters(),
        lr=args.lr,
        rank=args.rank,
        beta1=args.beta1,
        beta2=args.beta2,
        gamma=args.gamma,
        eps=args.eps,
        svd_frequency=args.svd_frequency,
    )
    
    # Print optimizer memory usage
    mem_stats = optimizer.get_memory_usage()
    print(f"\nOptimizer Memory Statistics:")
    print(f"  Parameters: {mem_stats['total_params']:,} ({mem_stats['param_memory_mb']:.1f} MB)")
    print(f"  Optimizer state: {mem_stats['state_memory_mb']:.1f} MB")
    print(f"  Total: {mem_stats['total_memory_mb']:.1f} MB")
    print(f"  State/Param ratio: {mem_stats['state_to_param_ratio']:.2f}x")
    
    # Create config dict
    config = {
        'model_name': args.model_name,
        'max_steps': args.max_steps,
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'max_length': args.max_length,
        'lr': args.lr,
        'rank': args.rank,
        'beta1': args.beta1,
        'beta2': args.beta2,
        'gamma': args.gamma,
        'eps': args.eps,
        'svd_frequency': args.svd_frequency,
        'output_dir': args.output_dir,
        'log_interval': args.log_interval,
        'save_interval': args.save_interval,
        'device': device,
    }
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        config=config,
        device=device,
    )
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
