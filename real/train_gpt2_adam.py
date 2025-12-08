"""
Train GPT-2 with Standard Adam (Baseline)
==========================================

This script trains GPT-2 with standard Adam optimizer for comparison with UMTAM.

Usage:
    python train_gpt2_adam.py --output_dir ./outputs_adam
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
import argparse
import time
import json
import os
from pathlib import Path
import math
from tqdm import tqdm


class FineWebDataset:
    """Streaming dataset for FineWeb with tokenization."""
    
    def __init__(self, cache_dir, tokenizer, max_length=1024, streaming=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"Loading FineWeb from {cache_dir}...")
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-10BT",
            split="train",
            cache_dir=cache_dir,
            streaming=streaming,
        )
        self.streaming = streaming
        
    def __iter__(self):
        for sample in self.dataset:
            text = sample['text']
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
            )
            yield {
                'input_ids': tokens['input_ids'].squeeze(0),
                'attention_mask': tokens['attention_mask'].squeeze(0),
            }


def get_optimizer_memory(optimizer, model):
    """Calculate optimizer memory usage."""
    total_params = sum(p.numel() for p in model.parameters())
    # Adam stores: first moment + second moment = 2x params
    state_memory_mb = (total_params * 2 * 4) / (1024 ** 2)
    param_memory_mb = (total_params * 4) / (1024 ** 2)
    
    return {
        'total_params': total_params,
        'param_memory_mb': param_memory_mb,
        'state_memory_mb': state_memory_mb,
        'total_memory_mb': param_memory_mb + state_memory_mb,
    }


def main():
    parser = argparse.ArgumentParser(description='Train GPT-2 with Adam (Baseline)')
    parser.add_argument('--data_dir', type=str, 
                       default='/Volumes/LaCie/Alirezas/umtam/fineweb_10bt_saved',
                       help='Path to FineWeb dataset cache')
    parser.add_argument('--output_dir', type=str, default='./outputs_adam',
                       help='Output directory')
    parser.add_argument('--model_name', type=str, default='gpt2',
                       help='Model name')
    parser.add_argument('--max_steps', type=int, default=10000,
                       help='Maximum training steps')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Gradient accumulation')
    parser.add_argument('--max_length', type=int, default=1024,
                       help='Maximum sequence length')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=1000,
                       help='Save interval')
    
    args = parser.parse_args()
    
    # Device
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
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'training_log.jsonl'
    
    # Load model and tokenizer
    print("Loading tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {total_params:,} parameters ({total_params/1e6:.1f}M)")
    
    # Create optimizer
    print("Creating Adam optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    
    # Memory stats
    mem_stats = get_optimizer_memory(optimizer, model)
    print(f"\nOptimizer Memory Statistics:")
    print(f"  Parameters: {mem_stats['total_params']:,} ({mem_stats['param_memory_mb']:.1f} MB)")
    print(f"  Optimizer state: {mem_stats['state_memory_mb']:.1f} MB")
    print(f"  Total: {mem_stats['total_memory_mb']:.1f} MB")
    
    # Create dataset
    print("Creating dataset...")
    train_dataset = FineWebDataset(
        cache_dir=args.data_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
        streaming=True,
    )
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting Training (Adam Baseline)")
    print(f"{'='*60}")
    print(f"Max steps: {args.max_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    step = 0
    running_loss = 0.0
    start_time = time.time()
    best_loss = float('inf')
    
    data_iter = iter(train_dataset)
    accumulated_batches = []
    
    pbar = tqdm(total=args.max_steps, desc="Training")
    
    while step < args.max_steps:
        try:
            model.train()
            
            # Collect batches
            for _ in range(args.gradient_accumulation_steps):
                sample = next(data_iter)
                batch = {
                    'input_ids': sample['input_ids'].unsqueeze(0),
                    'attention_mask': sample['attention_mask'].unsqueeze(0),
                }
                accumulated_batches.append(batch)
            
            # Combine
            combined_batch = {
                'input_ids': torch.cat([b['input_ids'] for b in accumulated_batches], dim=0),
                'attention_mask': torch.cat([b['attention_mask'] for b in accumulated_batches], dim=0),
            }
            accumulated_batches = []
            
            # Move to device
            input_ids = combined_batch['input_ids'].to(device)
            attention_mask = combined_batch['attention_mask'].to(device)
            
            # Forward
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_val = loss.item()
            running_loss += loss_val
            step += 1
            
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss_val:.4f}'})
            
            # Logging
            if step % args.log_interval == 0:
                avg_loss = running_loss / args.log_interval
                elapsed = time.time() - start_time
                steps_per_sec = args.log_interval / elapsed
                
                metrics = {
                    'step': step,
                    'loss': avg_loss,
                    'perplexity': math.exp(avg_loss) if avg_loss < 10 else float('inf'),
                    'steps_per_sec': steps_per_sec,
                    'optimizer_memory_mb': mem_stats['state_memory_mb'],
                    'total_memory_mb': mem_stats['total_memory_mb'],
                }
                
                with open(log_file, 'a') as f:
                    f.write(json.dumps(metrics) + '\n')
                
                print(f"\nStep {step}/{args.max_steps} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"PPL: {metrics['perplexity']:.2f} | "
                      f"Speed: {steps_per_sec:.2f} steps/s")
                
                running_loss = 0.0
                start_time = time.time()
            
            # Checkpointing
            if step % args.save_interval == 0:
                checkpoint = {
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_val,
                }
                checkpoint_path = output_dir / f'checkpoint_step_{step}.pt'
                torch.save(checkpoint, checkpoint_path)
                
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_path = output_dir / 'checkpoint_best.pt'
                    torch.save(checkpoint, best_path)
                    print(f"Saved best checkpoint (loss={loss_val:.4f})")
                    
        except StopIteration:
            print("Dataset exhausted, restarting...")
            data_iter = iter(train_dataset)
            
        except KeyboardInterrupt:
            print("\nTraining interrupted!")
            break
    
    pbar.close()
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
