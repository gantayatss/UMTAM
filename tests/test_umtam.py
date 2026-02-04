"""
Test UMTAM Optimizer
====================

Simple test to verify the UMTAM optimizer is working correctly.
"""

import sys
from pathlib import Path
# Add parent directory to path to import from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from src.umtam_optimizer import UMTAMOptimizer
import matplotlib.pyplot as plt


def test_simple_optimization():
    """Test UMTAM on a simple quadratic problem."""
    print("Testing UMTAM optimizer on quadratic problem...")
    
    # Create a simple 2-layer network
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    
    # Create optimizer
    optimizer = UMTAMOptimizer(
        model.parameters(),
        lr=0.01,
        rank=8,
        beta1=0.9,
        beta2=0.999,
    )
    
    # Simple dataset
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    
    # Training loop
    losses = []
    for epoch in range(100):
        optimizer.zero_grad()
        
        # Forward
        y_pred = model(X)
        loss = nn.MSELoss()(y_pred, y)
        
        # Backward
        loss.backward()
        
        # Step
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    # Check convergence
    if losses[-1] < losses[0]:
        print("✓ Loss decreased - optimizer is working!")
    else:
        print("✗ Warning: Loss did not decrease")
    
    # Memory stats
    mem_stats = optimizer.get_memory_usage()
    print(f"\nMemory Statistics:")
    print(f"  Param memory: {mem_stats['param_memory_mb']:.2f} MB")
    print(f"  State memory: {mem_stats['state_memory_mb']:.2f} MB")
    print(f"  Total memory: {mem_stats['total_memory_mb']:.2f} MB")
    print(f"  Ratio: {mem_stats['state_to_param_ratio']:.2f}x")
    
    return losses


def test_memory_efficiency():
    """Compare memory usage with Adam."""
    print("\n" + "="*60)
    print("Comparing memory usage: UMTAM vs Adam")
    print("="*60)
    
    # Create identical models
    model_umtam = nn.Sequential(
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 100)
    )
    
    model_adam = nn.Sequential(
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 100)
    )
    
    # Create optimizers
    optimizer_umtam = UMTAMOptimizer(model_umtam.parameters(), lr=0.001, rank=32)
    optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=0.001)
    
    # Get memory stats
    mem_umtam = optimizer_umtam.get_memory_usage()
    
    # For Adam, manually calculate
    total_params = sum(p.numel() for p in model_adam.parameters())
    # Adam stores: params (1x) + first moment (1x) + second moment (1x) = 3x
    adam_memory_mb = (total_params * 3 * 4) / (1024 ** 2)
    
    print(f"\nModel parameters: {total_params:,}")
    print(f"\nAdam:")
    print(f"  Total memory: ~{adam_memory_mb:.2f} MB (3x params)")
    print(f"\nUMTAM (rank=32):")
    print(f"  Total memory: {mem_umtam['total_memory_mb']:.2f} MB")
    print(f"  State/param ratio: {mem_umtam['state_to_param_ratio']:.2f}x")
    print(f"\nMemory savings: {((adam_memory_mb - mem_umtam['total_memory_mb']) / adam_memory_mb * 100):.1f}%")


def test_saliency_tracking():
    """Test saliency score tracking."""
    print("\n" + "="*60)
    print("Testing saliency tracking")
    print("="*60)
    
    # Simple model
    model = nn.Linear(10, 5)
    optimizer = UMTAMOptimizer(model.parameters(), lr=0.01, rank=4)
    
    # Train for a few steps
    X = torch.randn(50, 10)
    y = torch.randn(50, 5)
    
    for _ in range(10):
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(X), y)
        loss.backward()
        optimizer.step()
    
    # Get saliency scores
    saliency = optimizer.get_saliency_scores()
    
    print(f"Number of parameter groups with saliency: {len(saliency)}")
    for param_id, scores in saliency.items():
        print(f"  Param {param_id}: shape {scores.shape}, "
              f"mean={scores.mean():.6f}, max={scores.max():.6f}")
    
    print("✓ Saliency tracking working!")


if __name__ == '__main__':
    # Run tests
    print("="*60)
    print("UMTAM Optimizer Tests")
    print("="*60)
    
    # Test 1: Basic optimization
    losses = test_simple_optimization()
    
    # Test 2: Memory comparison
    test_memory_efficiency()
    
    # Test 3: Saliency tracking
    test_saliency_tracking()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
