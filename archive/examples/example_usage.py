"""
Simple Example: How to Use UMTAM Optimizer
===========================================

This shows the minimal code needed to use UMTAM with any PyTorch model.
"""

import sys
from pathlib import Path
# Add parent directory to path to import from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from src.umtam_optimizer import UMTAMOptimizer


# Example 1: Basic Usage
# ----------------------
def basic_example():
    """Minimal example of using UMTAM."""
    print("="*60)
    print("Example 1: Basic Usage")
    print("="*60)
    
    # Create any PyTorch model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    # Create UMTAM optimizer (just like Adam!)
    optimizer = UMTAMOptimizer(
        model.parameters(),
        lr=0.001,
        rank=16,  # Key UMTAM parameter
    )
    
    # Training loop - exactly like normal PyTorch
    for step in range(10):
        # Your data
        x = torch.randn(32, 100)
        y = torch.randn(32, 10)
        
        # Forward
        y_pred = model(x)
        loss = nn.MSELoss()(y_pred, y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Update (UMTAM does its magic here!)
        optimizer.step()
        
        print(f"Step {step}: Loss = {loss.item():.4f}")
    
    print("✓ Done!\n")


# Example 2: With GPT-2 (like your use case)
# ------------------------------------------
def gpt2_example():
    """Example with GPT-2."""
    print("="*60)
    print("Example 2: GPT-2 Training")
    print("="*60)
    
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    # Load model
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create UMTAM optimizer
    optimizer = UMTAMOptimizer(
        model.parameters(),
        lr=3e-4,      # Standard GPT-2 learning rate
        rank=32,      # Rank for factorization
        beta1=0.9,    # Momentum decay
        beta2=0.999,  # Second moment decay
        gamma=0.9,    # Error feedback
    )
    
    # Check memory usage
    mem_stats = optimizer.get_memory_usage()
    print(f"\nMemory Usage:")
    print(f"  Model params: {mem_stats['param_memory_mb']:.1f} MB")
    print(f"  Optimizer state: {mem_stats['state_memory_mb']:.1f} MB")
    print(f"  Total: {mem_stats['total_memory_mb']:.1f} MB")
    print(f"  Ratio: {mem_stats['state_to_param_ratio']:.2f}x")
    
    # Training loop
    text = "Hello, this is a test."
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    
    for step in range(3):
        # Forward
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Step {step}: Loss = {loss.item():.4f}")
    
    print("✓ Done!\n")


# Example 3: Retrieving Saliency Scores (for later merging)
# ---------------------------------------------------------
def saliency_example():
    """Example of tracking and retrieving saliency scores."""
    print("="*60)
    print("Example 3: Saliency Tracking")
    print("="*60)
    
    # Create model
    model = nn.Linear(10, 5)
    
    # Create optimizer
    optimizer = UMTAMOptimizer(
        model.parameters(),
        lr=0.01,
        rank=4,
        alpha=0.99,  # Saliency decay factor
    )
    
    # Train for a few steps
    for step in range(20):
        x = torch.randn(50, 10)
        y = torch.randn(50, 5)
        
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        optimizer.step()
    
    # Get saliency scores (for merging later!)
    saliency_scores = optimizer.get_saliency_scores()
    
    print(f"\nSaliency scores for {len(saliency_scores)} parameter groups")
    for param_id, scores in saliency_scores.items():
        print(f"  Param {param_id}:")
        print(f"    Shape: {scores.shape}")
        print(f"    Mean: {scores.mean():.6f}")
        print(f"    Max: {scores.max():.6f}")
        print(f"    Top 10% threshold: {scores.quantile(0.9):.6f}")
    
    print("\n✓ These scores will be used in Algorithm 2 (merging phase)!\n")


# Example 4: Customizing Hyperparameters
# --------------------------------------
def custom_hyperparameters():
    """Example showing all available hyperparameters."""
    print("="*60)
    print("Example 4: All Hyperparameters")
    print("="*60)
    
    model = nn.Linear(100, 50)
    
    optimizer = UMTAMOptimizer(
        model.parameters(),
        
        # Learning
        lr=0.001,                    # Learning rate
        
        # UMTAM-specific
        rank=32,                     # Factorization rank (key parameter!)
        beta1=0.9,                   # First moment decay
        beta2=0.999,                 # Second moment decay
        gamma=0.9,                   # Error feedback decay
        alpha=0.99,                  # Saliency tracking decay
        
        # Numerical stability
        eps=1e-8,                    # Regularization constant
        grad_clip=1.0,               # Gradient clipping threshold
        
        # Performance tuning
        svd_frequency=1,             # How often to compute SVD (higher = faster)
        
        # Adaptive rank (experimental)
        adaptive_rank=False,         # Enable adaptive rank adjustment
        min_rank=8,                  # Minimum rank
        max_rank=128,                # Maximum rank
        
        # Standard
        weight_decay=0.0,            # L2 regularization
    )
    
    print("Optimizer created with custom hyperparameters!")
    print("\nKey parameters to tune:")
    print("  - rank: 16, 32, 64, 128 (lower = more memory savings)")
    print("  - lr: 1e-4, 3e-4, 5e-4, 1e-3 (standard range)")
    print("  - svd_frequency: 1, 10, 20 (higher = faster but less accurate)")
    print("  - beta1: 0.85, 0.9, 0.95 (momentum decay)")
    print("✓ Done!\n")


# Example 5: Comparing with Adam
# ------------------------------
def compare_with_adam():
    """Side-by-side comparison with Adam."""
    print("="*60)
    print("Example 5: UMTAM vs Adam Comparison")
    print("="*60)
    
    # Create two identical models
    model_umtam = nn.Linear(1000, 500)
    model_adam = nn.Linear(1000, 500)
    
    # Copy weights to ensure identical start
    model_adam.load_state_dict(model_umtam.state_dict())
    
    # Create optimizers
    optimizer_umtam = UMTAMOptimizer(model_umtam.parameters(), lr=0.001, rank=32)
    optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=0.001)
    
    # Compare memory
    mem_umtam = optimizer_umtam.get_memory_usage()
    
    total_params = sum(p.numel() for p in model_adam.parameters())
    mem_adam = (total_params * 3 * 4) / (1024 ** 2)  # 3x for Adam
    
    print("\nMemory Comparison:")
    print(f"  Model params: {total_params:,}")
    print(f"  Adam memory: {mem_adam:.1f} MB")
    print(f"  UMTAM memory: {mem_umtam['total_memory_mb']:.1f} MB")
    print(f"  Savings: {(mem_adam - mem_umtam['total_memory_mb']) / mem_adam * 100:.1f}%")
    
    # Train both for a few steps
    print("\nTraining comparison:")
    x = torch.randn(100, 1000)
    y = torch.randn(100, 500)
    
    for step in range(10):
        # UMTAM
        optimizer_umtam.zero_grad()
        loss_umtam = nn.MSELoss()(model_umtam(x), y)
        loss_umtam.backward()
        optimizer_umtam.step()
        
        # Adam
        optimizer_adam.zero_grad()
        loss_adam = nn.MSELoss()(model_adam(x), y)
        loss_adam.backward()
        optimizer_adam.step()
        
        if step % 3 == 0:
            print(f"  Step {step}: UMTAM loss={loss_umtam.item():.4f}, "
                  f"Adam loss={loss_adam.item():.4f}")
    
    print("\n✓ Both should converge similarly!\n")


if __name__ == '__main__':
    # Run all examples
    basic_example()
    gpt2_example()
    saliency_example()
    custom_hyperparameters()
    compare_with_adam()
    
    print("="*60)
    print("All examples completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Try UMTAM on your own models")
    print("2. Tune the rank parameter for your use case")
    print("3. Compare memory and performance with Adam")
    print("4. Use saliency scores for model merging (Option B)")
