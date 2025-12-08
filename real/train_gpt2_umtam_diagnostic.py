"""
Train GPT-2 with UMTAM and Diagnostics
======================================

Enhanced training script that validates theoretical properties during training:
- Tracks stable rank of momentum
- Monitors compression error
- Analyzes saliency concentration

Usage:
    python train_gpt2_umtam_diagnostic.py --diagnostics --diagnostic_interval 100
"""

import sys
from pathlib import Path
# Add parent directory to path to import from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from src.umtam_optimizer import UMTAMOptimizer
from src.umtam_diagnostics import UMTAMDiagnostics
import argparse
import time
import json
import math
from tqdm import tqdm
from datasets import load_dataset


class FineWebDataset:
    """Streaming dataset for FineWeb."""
    
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


def main():
    parser = argparse.ArgumentParser(description='Train GPT-2 with UMTAM (with diagnostics)')
    
    # Data
    parser.add_argument('--data_dir', type=str,
                       default='/Volumes/LaCie/Alirezas/umtam/fineweb_10bt_saved')
    parser.add_argument('--output_dir', type=str, default='./outputs_diagnostic')
    parser.add_argument('--model_name', type=str, default='gpt2')
    
    # Training
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=3e-4)
    
    # UMTAM
    parser.add_argument('--rank', type=int, default=32)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--svd_frequency', type=int, default=10)
    
    # Diagnostics
    parser.add_argument('--diagnostics', action='store_true',
                       help='Enable diagnostic tracking')
    parser.add_argument('--diagnostic_interval', type=int, default=50,
                       help='Diagnostic logging frequency')
    
    # Device
    parser.add_argument('--device', type=str, default='auto')
    
    # Logging
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=500)
    
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
    print("Loading model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters ({total_params/1e6:.1f}M)")
    
    # Create dataset
    print("Creating dataset...")
    train_dataset = FineWebDataset(
        cache_dir=args.data_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
        streaming=True,
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
    
    # Create diagnostics tracker
    diagnostics = None
    if args.diagnostics:
        print(f"✅ Diagnostics enabled (interval={args.diagnostic_interval})")
        diagnostics = UMTAMDiagnostics(log_interval=args.diagnostic_interval)
    
    # Print memory stats
    mem_stats = optimizer.get_memory_usage()
    print(f"\nMemory Statistics:")
    print(f"  Parameters: {mem_stats['param_memory_mb']:.1f} MB")
    print(f"  Optimizer state: {mem_stats['state_memory_mb']:.1f} MB")
    print(f"  Total: {mem_stats['total_memory_mb']:.1f} MB")
    print(f"  Ratio: {mem_stats['state_to_param_ratio']:.2f}x")
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Training with Diagnostics")
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
            
            # Update diagnostics
            if diagnostics:
                diagnostics.update(optimizer, step)
            
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss_val:.4f}'})
            
            # Logging
            if step % args.log_interval == 0:
                avg_loss = running_loss / args.log_interval
                elapsed = time.time() - start_time
                steps_per_sec = args.log_interval / elapsed
                
                mem_stats = optimizer.get_memory_usage()
                
                metrics = {
                    'step': step,
                    'loss': avg_loss,
                    'perplexity': math.exp(avg_loss) if avg_loss < 10 else float('inf'),
                    'steps_per_sec': steps_per_sec,
                    'optimizer_memory_mb': mem_stats['state_memory_mb'],
                }
                
                # Add diagnostic metrics
                if diagnostics and diagnostics.stable_ranks:
                    metrics['stable_rank'] = diagnostics.stable_ranks[-1]
                if diagnostics and diagnostics.compression_errors:
                    metrics['compression_error'] = diagnostics.compression_errors[-1]
                
                with open(log_file, 'a') as f:
                    f.write(json.dumps(metrics) + '\n')
                
                diag_str = ""
                if diagnostics and diagnostics.stable_ranks:
                    diag_str = f" | Rank: {diagnostics.stable_ranks[-1]:.1f}"
                
                print(f"\nStep {step}/{args.max_steps} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"PPL: {metrics['perplexity']:.2f} | "
                      f"Speed: {steps_per_sec:.2f} s/s{diag_str}")
                
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
                
                # Save checkpoint
                checkpoint_path = output_dir / f'checkpoint_step_{step}.pt'
                torch.save(checkpoint, checkpoint_path)
                
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_path = output_dir / 'checkpoint_best.pt'
                    torch.save(checkpoint, best_path)
                    print(f"✅ Saved best checkpoint (loss={loss_val:.4f})")
                
                # Save diagnostic plots
                if diagnostics:
                    diagnostics.plot_diagnostics(output_dir / 'diagnostics')
                    diagnostics.print_summary()
                    
                    # Validate theoretical properties
                    validation = diagnostics.validate_theoretical_properties()
                    print("\nTheoretical Properties Validation:")
                    for prop, passes in validation.items():
                        status = "✅" if passes else "⚠️"
                        print(f"  {status} {prop}")
                    
        except StopIteration:
            print("Dataset exhausted, restarting...")
            data_iter = iter(train_dataset)
            
        except KeyboardInterrupt:
            print("\nTraining interrupted!")
            break
    
    pbar.close()
    
    # Final diagnostics
    if diagnostics:
        print(f"\n{'='*60}")
        print("Final Diagnostics")
        print(f"{'='*60}")
        
        diagnostics.print_summary()
        diagnostics.plot_diagnostics(output_dir / 'diagnostics')
        
        # Validate
        validation = diagnostics.validate_theoretical_properties()
        print("\nTheoretical Properties Validation:")
        all_pass = all(validation.values())
        for prop, passes in validation.items():
            status = "✅ PASS" if passes else "⚠️ WARN"
            print(f"  {status} {prop}")
        
        if all_pass:
            print("\n✅ All theoretical properties validated!")
        else:
            print("\n⚠️  Some properties need attention (see above)")
    
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
