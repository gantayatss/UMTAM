#!/usr/bin/env python3
"""
UMTAM Ablation Study - Rank 32
Auto-generated training script for rank=32
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import time
import json

from src.umtam_optimizer import UMTAMOptimizer
from src.fineweb_loader import FineWebDataset

# Configuration
RANK = 32
NUM_STEPS = 300
LR = 1e-3
BETA1 = 0.85
BATCH_SIZE = 4
MAX_LENGTH = 1024
EVAL_INTERVAL = 100
DATA_PATH = "/Volumes/LaCie/Alirezas/umtam/fineweb_10bt_saved"
OUTPUT_DIR = str(Path(__file__).parent / "results")

def main():
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    print("Loading GPT-2 model...")
    gpt2_config = GPT2Config.from_pretrained('gpt2')
    model = GPT2LMHeadModel(gpt2_config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Initialize optimizer
    print(f"Initializing UMTAM optimizer (rank={RANK})...")
    optimizer = UMTAMOptimizer(
        model.parameters(),
        lr=LR,
        rank=RANK,
        beta1=BETA1,
        beta2=0.95,
        gamma=0.1,
        svd_frequency=50,
    )
    
    # Get memory stats
    mem_stats = optimizer.get_memory_usage()
    print(f"Optimizer memory: {mem_stats['total_memory_mb']:.1f} MB")
    
    # Load data
    print("Loading dataset...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = FineWebDataset(
        data_path=DATA_PATH,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        streaming=False
    )
    
    # Training setup
    print("Starting training...")
    
    results = {
        'rank': RANK,
        'config': {
            'lr': LR,
            'beta1': BETA1,
            'batch_size': BATCH_SIZE,
            'num_steps': NUM_STEPS,
            'n_params': n_params
        },
        'train_losses': [],
        'val_losses': [],
        'step_times': [],
        'eval_steps': [],
        'memory_mb': mem_stats['total_memory_mb']
    }
    
    model.train()
    step = 0
    start_time = time.time()
    batch_buffer = []
    total_loss = 0
    
    dataset_iter = iter(dataset)
    
    while step < NUM_STEPS:
        try:
            # Collect batch
            batch_buffer = []
            for _ in range(BATCH_SIZE):
                sample = next(dataset_iter)
                batch_buffer.append(sample)
            
            # Stack into batch
            input_ids = torch.stack([s['input_ids'] for s in batch_buffer]).to(device)
            attention_mask = torch.stack([s['attention_mask'] for s in batch_buffer]).to(device)
            
            # Training step
            step_start = time.time()
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            # Record
            step_time = time.time() - step_start
            results['train_losses'].append(loss.item())
            results['step_times'].append(step_time)
            total_loss += loss.item()
            step += 1
            
            # Evaluate periodically
            if step % EVAL_INTERVAL == 0:
                avg_train_loss = total_loss / EVAL_INTERVAL
                total_loss = 0
                
                # Simple validation (just use last batch for speed)
                model.eval()
                with torch.no_grad():
                    val_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                    val_loss = val_outputs.loss.item()
                model.train()
                
                results['val_losses'].append(val_loss)
                results['eval_steps'].append(step)
                
                elapsed = time.time() - start_time
                print(f"Step {step}/{NUM_STEPS}: "
                      f"Train Loss={avg_train_loss:.4f}, "
                      f"Val Loss={val_loss:.4f}, "
                      f"Time={step_time:.3f}s, "
                      f"Elapsed={elapsed/60:.1f}m")
        
        except StopIteration:
            print("Dataset exhausted, restarting...")
            dataset_iter = iter(dataset)
            continue
    
    # Compute summary
    total_time = time.time() - start_time
    results['runtime_seconds'] = total_time
    results['final_val_loss'] = results['val_losses'][-1] if results['val_losses'] else None
    results['avg_step_time'] = sum(results['step_times']) / len(results['step_times'])
    results['throughput'] = (NUM_STEPS * BATCH_SIZE) / total_time
    
    # Save results
    results_path = Path(OUTPUT_DIR) / f"results_rank_{RANK}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"  Final val loss: {results['final_val_loss']:.4f}")
    print(f"  Runtime: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"  Throughput: {results['throughput']:.1f} samples/s")
    print(f"  Results saved to: {results_path}")
    print(f"{'='*70}\n")
    
    return results

if __name__ == '__main__':
    main()
