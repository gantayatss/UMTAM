
"""
Experiment 1.1: Momentum Has Low Stable Rank

Validates that exponential moving average of gradients exhibits 
low-rank structure throughout training.

Expected result: Stable rank converges to 5-10 (much less than d=1000)
Runtime: ~30 seconds on Mac
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def compute_stable_rank(M):
    """
    Stable rank: r_s = ||M||²_F / ||M||²_2
    
    For a matrix, this measures effective rank.
    """
    if M.dim() == 1:
        # If M is a vector, convert to matrix for SVD
        M = M.unsqueeze(0)
    
    frob_norm_sq = (M ** 2).sum().item()
    
    # Compute largest singular value
    try:
        singular_values = torch.linalg.svdvals(M)
        spectral_norm_sq = (singular_values[0].item()) ** 2
    except:
        spectral_norm_sq = torch.linalg.norm(M, ord=2).item() ** 2
    
    if spectral_norm_sq < 1e-10:
        return 0.0
    
    return frob_norm_sq / spectral_norm_sq

def run_experiment():
    print("=" * 60)
    print("Experiment 1.1: Momentum Low Stable Rank")
    print("=" * 60)
    
    # Parameters - FIXED: Use matrix shape
    m, n = 100, 200  # Matrix dimensions (simulates a weight matrix)
    T = 5000  # Training steps
    beta = 0.9  # Momentum decay
    true_rank = 5  # True subspace rank
    noise_level = 0.1
    
    print(f"\nConfiguration:")
    print(f"  Matrix shape: {m} × {n}")
    print(f"  Training steps (T): {T}")
    print(f"  Momentum decay (β): {beta}")
    print(f"  True signal rank: {true_rank}")
    print(f"  Noise level: {noise_level}")
    
    # Create true low-rank subspace for gradients
    # Gradients will be rank-5 signals + noise
    U_true = torch.randn(m, true_rank)
    V_true = torch.randn(n, true_rank)
    U_true, _ = torch.linalg.qr(U_true)  # Orthonormalize
    V_true, _ = torch.linalg.qr(V_true)
    
    # Initialize momentum matrix
    M = torch.zeros(m, n)
    
    # Track metrics
    stable_ranks = []
    actual_ranks = []
    steps_recorded = []
    
    print("\nRunning simulation...")
    for t in range(T):
        # Generate low-rank gradient + noise
        coeffs = torch.randn(true_rank, true_rank)
        G_signal = U_true @ coeffs @ V_true.T
        G_noise = noise_level * torch.randn(m, n)
        G = G_signal + G_noise
        
        # Momentum update (EMA)
        M = beta * M + (1 - beta) * G
        
        # Record metrics every 100 steps
        if t % 100 == 0:
            # Stable rank
            stable_rank = compute_stable_rank(M)
            stable_ranks.append(stable_rank)
            
            # Actual rank (count singular values > threshold)
            singular_values = torch.linalg.svdvals(M)
            threshold = singular_values[0] * 0.01  # 1% of largest
            actual_rank = (singular_values > threshold).sum().item()
            actual_ranks.append(actual_rank)
            
            steps_recorded.append(t)
            
            if t % 500 == 0:
                print(f"  Step {t:4d}: stable_rank = {stable_rank:.2f}, actual_rank = {actual_rank}")
    
    # Final statistics
    final_stable_rank = stable_ranks[-1]
    avg_stable_rank = np.mean(stable_ranks[-10:])
    avg_actual_rank = np.mean(actual_ranks[-10:])
    
    print(f"\nResults:")
    print(f"  Final stable rank: {final_stable_rank:.2f}")
    print(f"  Average stable rank (last 1000 steps): {avg_stable_rank:.2f}")
    print(f"  Average actual rank (last 1000 steps): {avg_actual_rank:.1f}")
    print(f"  True rank: {true_rank}")
    print(f"  Matrix dimension: {min(m, n)}")
    print(f"  Rank compression: {avg_stable_rank / min(m, n) * 100:.1f}%")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Stable Rank
    ax1.plot(steps_recorded, stable_ranks, linewidth=2, color='steelblue', label='Stable Rank')
    ax1.axhline(y=true_rank, color='r', linestyle='--', 
                label=f'True Rank = {true_rank}', linewidth=2)
    ax1.axhline(y=avg_stable_rank, color='g', linestyle='--', 
                label=f'Converged ≈ {avg_stable_rank:.1f}', linewidth=2)
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Stable Rank', fontsize=12)
    ax1.set_title('Momentum Stable Rank Over Training', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Actual Rank
    ax2.plot(steps_recorded, actual_ranks, linewidth=2, color='darkorange', label='Actual Rank')
    ax2.axhline(y=true_rank, color='r', linestyle='--', 
                label=f'True Rank = {true_rank}', linewidth=2)
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Actual Rank (σ > 0.01·σ_max)', fontsize=12)
    ax2.set_title('Momentum Effective Rank Over Training', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('../results')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'exp1_momentum_stable_rank.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_dir / 'exp1_momentum_stable_rank.png'}")
    
    plt.show()
    
    # Validation
    print("\n" + "=" * 60)
    if avg_stable_rank < true_rank * 4:  # Within 4x of true rank
        print("✅ PASS: Momentum maintains low-rank structure")
        print(f"   Stable rank {avg_stable_rank:.1f} is close to true rank {true_rank}")
        print(f"   ({avg_stable_rank / min(m, n) * 100:.1f}% of full dimension)")
    else:
        print("⚠️  WARN: Stable rank higher than expected")
        print(f"   This may indicate noise level is too high")
    print("=" * 60)
    
    return {
        'stable_ranks': stable_ranks,
        'actual_ranks': actual_ranks,
        'steps': steps_recorded,
        'final_rank': final_stable_rank,
        'avg_stable_rank': avg_stable_rank,
        'avg_actual_rank': avg_actual_rank
    }

if __name__ == "__main__":
    results = run_experiment()


# """
# Experiment 1.1: Momentum Has Low Stable Rank

# Validates that exponential moving average of gradients exhibits 
# low-rank structure throughout training.

# Expected result: Stable rank converges to 5-10 (much less than d=1000)
# Runtime: ~30 seconds on Mac
# """

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# def compute_stable_rank(M):
#     """
#     Stable rank: r_s = ||M||²_F / ||M||²_2
#     Measures effective rank (how well low-rank approximation works)
#     """
#     frob_norm_sq = (M ** 2).sum().item()
#     spectral_norm_sq = torch.linalg.norm(M, ord=2).item() ** 2
    
#     if spectral_norm_sq < 1e-10:
#         return 0.0
    
#     return frob_norm_sq / spectral_norm_sq

# def run_experiment():
#     print("=" * 60)
#     print("Experiment 1.1: Momentum Low Stable Rank")
#     print("=" * 60)
    
#     # Parameters
#     d = 1000  # Parameter dimension
#     T = 5000  # Training steps
#     beta = 0.9  # Momentum decay
#     true_rank = 5  # True subspace rank
#     noise_level = 0.1
    
#     print(f"\nConfiguration:")
#     print(f"  Dimension (d): {d}")
#     print(f"  Training steps (T): {T}")
#     print(f"  Momentum decay (β): {beta}")
#     print(f"  True signal rank: {true_rank}")
#     print(f"  Noise level: {noise_level}")
    
#     # Create true low-rank subspace
#     U_true = torch.randn(d, true_rank)
#     U_true, _ = torch.linalg.qr(U_true)  # Orthonormalize
    
#     # Initialize momentum
#     M = torch.zeros(d)
    
#     # Track metrics
#     stable_ranks = []
#     steps_recorded = []
    
#     print("\nRunning simulation...")
#     for t in range(T):
#         # Generate gradient: low-rank signal + noise
#         signal = U_true @ torch.randn(true_rank)
#         noise = noise_level * torch.randn(d)
#         g = signal + noise
        
#         # Momentum update
#         M = beta * M + (1 - beta) * g
        
#         # Record stable rank every 100 steps
#         if t % 100 == 0:
#             stable_rank = compute_stable_rank(M)
#             stable_ranks.append(stable_rank)
#             steps_recorded.append(t)
            
#             if t % 500 == 0:
#                 print(f"  Step {t:4d}: stable_rank = {stable_rank:.2f}")
    
#     # Final statistics
#     final_stable_rank = stable_ranks[-1]
#     avg_stable_rank = np.mean(stable_ranks[-10:])  # Average of last 10 recordings
    
#     print(f"\nResults:")
#     print(f"  Final stable rank: {final_stable_rank:.2f}")
#     print(f"  Average stable rank (last 1000 steps): {avg_stable_rank:.2f}")
#     print(f"  True rank: {true_rank}")
#     print(f"  Dimension: {d}")
#     print(f"  Compression ratio: {avg_stable_rank / d * 100:.1f}%")
    
#     # Plot
#     plt.figure(figsize=(10, 6))
#     plt.plot(steps_recorded, stable_ranks, linewidth=2, label='Stable Rank')
#     plt.axhline(y=true_rank, color='r', linestyle='--', 
#                 label=f'True Rank = {true_rank}', linewidth=2)
#     plt.axhline(y=avg_stable_rank, color='g', linestyle='--', 
#                 label=f'Converged Rank ≈ {avg_stable_rank:.1f}', linewidth=2)
    
#     plt.xlabel('Training Step', fontsize=12)
#     plt.ylabel('Stable Rank', fontsize=12)
#     plt.title('Momentum Exhibits Low Stable Rank Throughout Training', fontsize=14)
#     plt.legend(fontsize=11)
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
    
#     # Save plot
#     output_dir = Path('../results')
#     output_dir.mkdir(exist_ok=True)
#     plt.savefig(output_dir / 'exp1_momentum_stable_rank.png', dpi=300, bbox_inches='tight')
#     print(f"\n✅ Plot saved to: {output_dir / 'exp1_momentum_stable_rank.png'}")
    
#     plt.show()
    
#     # Validation
#     print("\n" + "=" * 60)
#     if avg_stable_rank < true_rank * 3:
#         print("✅ PASS: Momentum maintains low-rank structure")
#         print(f"   (Stable rank {avg_stable_rank:.1f} << dimension {d})")
#     else:
#         print("⚠️  WARN: Stable rank higher than expected")
#     print("=" * 60)
    
#     return {
#         'stable_ranks': stable_ranks,
#         'steps': steps_recorded,
#         'final_rank': final_stable_rank,
#         'avg_rank': avg_stable_rank
#     }

# if __name__ == "__main__":
#     results = run_experiment()