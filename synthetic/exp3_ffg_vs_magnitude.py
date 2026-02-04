"""
Experiment 1.3: FFG vs Magnitude Pruning (Synthetic)

Compares Fast Fisher Grafting (curvature-weighted) pruning against
simple magnitude pruning on synthetic task vector recovery.

Expected result: FFG outperforms magnitude by 15-30% at low densities
Runtime: ~1 minute on Mac
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def generate_synthetic_task(d, rank_true, noise_level=0.1):
    """
    Generate synthetic fine-tuning scenario:
    - True update is low-rank (rank_true)
    - Noisy updates added during training
    
    Returns:
        delta_w_true: True low-rank update
        delta_w_noisy: Noisy observed update
        H: Diagonal curvature (importance weights)
    """
    # True low-rank update
    U = torch.randn(d, rank_true)
    U, _ = torch.linalg.qr(U)
    coeffs = torch.randn(rank_true)
    delta_w_true = U @ coeffs
    
    # Curvature: true signal has high curvature
    H = torch.ones(d) * 0.1
    true_support = (delta_w_true.abs() > delta_w_true.abs().quantile(0.7))
    H[true_support] = 10.0  # High importance for true signal
    
    # Add noise
    noise = noise_level * torch.randn(d)
    delta_w_noisy = delta_w_true + noise
    
    return delta_w_true, delta_w_noisy, H

def ffg_pruning(delta_w, H, density):
    """Fast Fisher Grafting: prune by (delta_w)² * H"""
    saliency = (delta_w ** 2) * H
    k = int(density * len(delta_w))
    threshold = torch.topk(saliency, k).values[-1]
    mask = saliency >= threshold
    return delta_w * mask.float(), mask

def magnitude_pruning(delta_w, density):
    """Standard magnitude pruning: prune by |delta_w|"""
    saliency = delta_w.abs()
    k = int(density * len(delta_w))
    threshold = torch.topk(saliency, k).values[-1]
    mask = saliency >= threshold
    return delta_w * mask.float(), mask

def compute_recovery_score(delta_w_pruned, delta_w_true):
    """
    Measure how well pruned vector recovers true signal
    Using correlation (higher is better)
    """
    numerator = (delta_w_pruned * delta_w_true).sum()
    denom1 = torch.norm(delta_w_pruned)
    denom2 = torch.norm(delta_w_true)
    
    if denom1 < 1e-10 or denom2 < 1e-10:
        return 0.0
    
    correlation = numerator / (denom1 * denom2)
    return correlation.item()

def run_experiment():
    print("=" * 60)
    print("Experiment 1.3: FFG vs Magnitude Pruning")
    print("=" * 60)
    
    # Parameters
    d = 1000
    rank_true = 10
    noise_level = 0.2
    densities = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
    num_trials = 10
    
    print(f"\nConfiguration:")
    print(f"  Dimension (d): {d}")
    print(f"  True rank: {rank_true}")
    print(f"  Noise level: {noise_level}")
    print(f"  Densities tested: {densities}")
    print(f"  Trials per density: {num_trials}")
    
    # Storage
    results_ffg = {density: [] for density in densities}
    results_mag = {density: [] for density in densities}
    
    print("\nRunning trials...")
    for trial in range(num_trials):
        if trial % 2 == 0:
            print(f"  Trial {trial+1}/{num_trials}")
        
        # Generate task
        delta_w_true, delta_w_noisy, H = generate_synthetic_task(d, rank_true, noise_level)
        
        for density in densities:
            # FFG pruning
            delta_ffg, _ = ffg_pruning(delta_w_noisy, H, density)
            score_ffg = compute_recovery_score(delta_ffg, delta_w_true)
            results_ffg[density].append(score_ffg)
            
            # Magnitude pruning
            delta_mag, _ = magnitude_pruning(delta_w_noisy, density)
            score_mag = compute_recovery_score(delta_mag, delta_w_true)
            results_mag[density].append(score_mag)
    
    # Compute statistics
    avg_ffg = [np.mean(results_ffg[d]) for d in densities]
    std_ffg = [np.std(results_ffg[d]) for d in densities]
    avg_mag = [np.mean(results_mag[d]) for d in densities]
    std_mag = [np.std(results_mag[d]) for d in densities]
    
    # Print results table
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"{'Density':<10} {'FFG':<15} {'Magnitude':<15} {'Improvement':<15}")
    print("-" * 60)
    for i, density in enumerate(densities):
        improvement = ((avg_ffg[i] - avg_mag[i]) / avg_mag[i] * 100) if avg_mag[i] > 0 else 0
        print(f"{density:<10.0%} {avg_ffg[i]:<15.3f} {avg_mag[i]:<15.3f} {improvement:>+14.1f}%")
    print("=" * 60)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.array(densities) * 100  # Convert to percentage
    
    ax.plot(x, avg_ffg, 'o-', linewidth=2, markersize=8, label='FFG (Curvature-weighted)', color='steelblue')
    ax.fill_between(x, 
                     np.array(avg_ffg) - np.array(std_ffg), 
                     np.array(avg_ffg) + np.array(std_ffg), 
                     alpha=0.2, color='steelblue')
    
    ax.plot(x, avg_mag, 's-', linewidth=2, markersize=8, label='Magnitude Pruning', color='orangered')
    ax.fill_between(x, 
                     np.array(avg_mag) - np.array(std_mag), 
                     np.array(avg_mag) + np.array(std_mag), 
                     alpha=0.2, color='orangered')
    
    ax.set_xlabel('Density (%)', fontsize=12)
    ax.set_ylabel('Recovery Score (Correlation with True Signal)', fontsize=12)
    ax.set_title('FFG vs Magnitude Pruning: Signal Recovery Quality', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    
    # Save
    output_dir = Path('../results')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'exp3_ffg_vs_magnitude.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_dir / 'exp3_ffg_vs_magnitude.png'}")
    
    plt.show()
    
    # Validation
    low_density_improvement = ((avg_ffg[0] - avg_mag[0]) / avg_mag[0] * 100)
    print("\n" + "=" * 60)
    if low_density_improvement > 15:
        print("✅ PASS: FFG significantly outperforms magnitude pruning")
        print(f"   (At 5% density: +{low_density_improvement:.1f}% improvement)")
    else:
        print("⚠️  WARN: Improvement lower than expected")
    print("=" * 60)
    
    return {
        'densities': densities,
        'avg_ffg': avg_ffg,
        'avg_mag': avg_mag,
        'improvement': low_density_improvement
    }

if __name__ == "__main__":
    results = run_experiment()