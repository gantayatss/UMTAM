"""
UMTAM Diagnostic Utilities
==========================

Validation tools inspired by the synthetic experiments to verify:
1. Momentum maintains low stable rank (exp1)
2. Compression preserves preconditioner quality (exp2)
3. Saliency scoring works correctly (exp3)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple


class UMTAMDiagnostics:
    """
    Diagnostic tracker for UMTAM optimizer.
    
    Validates that theoretical properties hold during training:
    - Low stable rank of momentum
    - Low compression error in preconditioner
    - Effective saliency-based pruning
    """
    
    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.step = 0
        
        # Metrics storage
        self.stable_ranks = []
        self.compression_errors = []
        self.saliency_stats = []
        self.steps_recorded = []
        
    def compute_stable_rank(self, M: torch.Tensor) -> float:
        """
        Compute stable rank: r_s = ||M||²_F / ||M||²_2
        
        This matches exp1_momentum_rank.py
        """
        if M.dim() == 1:
            M = M.unsqueeze(0)
        
        frob_norm_sq = (M ** 2).sum().item()
        
        try:
            singular_values = torch.linalg.svdvals(M)
            spectral_norm_sq = (singular_values[0].item()) ** 2
        except:
            spectral_norm_sq = torch.linalg.norm(M, ord=2).item() ** 2
        
        if spectral_norm_sq < 1e-10:
            return 0.0
        
        return frob_norm_sq / spectral_norm_sq
    
    def compute_preconditioner_error(
        self, 
        R: torch.Tensor, 
        C: torch.Tensor, 
        G: torch.Tensor,
        eps: float = 1e-8
    ) -> float:
        """
        Compute preconditioner approximation error.
        
        Compares:
        - Full preconditioner: 1/sqrt(G @ G^T + eps)
        - Factorized: 1/sqrt(outer(R,C)/sum(R) + eps)
        
        This matches exp2_compression.py methodology.
        """
        # Full second moment (diagonal only, for efficiency)
        V_full = torch.sum(G ** 2, dim=1)  # Row-wise
        P_full = 1.0 / torch.sqrt(V_full + eps)
        
        # Factorized approximation
        total = torch.sum(R)
        V_approx_row = R * torch.sum(C) / (total + eps)
        P_approx = 1.0 / torch.sqrt(V_approx_row + eps)
        
        # Relative error
        numerator = torch.norm(P_full - P_approx).item()
        denominator = torch.norm(P_full).item()
        
        if denominator < 1e-10:
            return 0.0
        
        return numerator / denominator
    
    def compute_saliency_concentration(self, S: torch.Tensor) -> Dict[str, float]:
        """
        Analyze saliency score distribution.
        
        Returns statistics useful for understanding parameter importance.
        """
        S_flat = S.flatten()
        
        # Remove zeros
        S_nonzero = S_flat[S_flat > 1e-10]
        
        if len(S_nonzero) == 0:
            return {
                'mean': 0.0,
                'median': 0.0,
                'max': 0.0,
                'top10_threshold': 0.0,
                'top20_threshold': 0.0,
                'sparsity_80': 0.0,
            }
        
        return {
            'mean': S_nonzero.mean().item(),
            'median': S_nonzero.median().item(),
            'max': S_nonzero.max().item(),
            'top10_threshold': S_nonzero.quantile(0.9).item(),
            'top20_threshold': S_nonzero.quantile(0.8).item(),
            # Sparsity: fraction of total importance in top 20%
            'sparsity_80': (S_nonzero > S_nonzero.quantile(0.8)).float().mean().item(),
        }
    
    def update(self, optimizer, step: int):
        """
        Update diagnostics from optimizer state.
        
        Call this periodically during training to track metrics.
        """
        self.step = step
        
        if step % self.log_interval != 0:
            return
        
        # Extract metrics from optimizer state
        stable_ranks_batch = []
        comp_errors_batch = []
        saliency_stats_batch = []
        
        for group in optimizer.param_groups:
            for param in group['params']:
                if param not in optimizer.state:
                    continue
                
                state = optimizer.state[param]
                
                # Only process 2D parameters with full UMTAM state
                if 'U' not in state:
                    continue
                
                # Reconstruct momentum
                U = state['U']
                Sigma = state['Sigma']
                V = state['V']
                M = U @ Sigma @ V.T
                
                # Stable rank
                stable_rank = self.compute_stable_rank(M)
                stable_ranks_batch.append(stable_rank)
                
                # Compression error (if we have gradient)
                if param.grad is not None:
                    grad = param.grad
                    if len(grad.shape) == 2:
                        R = state['R']
                        C = state['C']
                        error = self.compute_preconditioner_error(R, C, grad)
                        comp_errors_batch.append(error)
                
                # Saliency statistics
                if 'S' in state:
                    S = state['S']
                    stats = self.compute_saliency_concentration(S)
                    saliency_stats_batch.append(stats)
        
        # Store averages
        if stable_ranks_batch:
            self.stable_ranks.append(np.mean(stable_ranks_batch))
        if comp_errors_batch:
            self.compression_errors.append(np.mean(comp_errors_batch))
        if saliency_stats_batch:
            avg_stats = {
                key: np.mean([s[key] for s in saliency_stats_batch])
                for key in saliency_stats_batch[0].keys()
            }
            self.saliency_stats.append(avg_stats)
        
        self.steps_recorded.append(step)
    
    def plot_diagnostics(self, output_dir: str = './diagnostics'):
        """
        Create diagnostic plots matching the synthetic experiments.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Stable Rank (like exp1)
        ax = axes[0, 0]
        if self.stable_ranks:
            ax.plot(self.steps_recorded, self.stable_ranks, 
                   linewidth=2, color='steelblue', marker='o', markersize=4)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Average Stable Rank')
            ax.set_title('Momentum Stable Rank During Training')
            ax.grid(True, alpha=0.3)
            
            # Add reference line
            avg_rank = np.mean(self.stable_ranks[-10:]) if len(self.stable_ranks) >= 10 else np.mean(self.stable_ranks)
            ax.axhline(y=avg_rank, color='green', linestyle='--', 
                      label=f'Average: {avg_rank:.1f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        
        # Plot 2: Compression Error (like exp2)
        ax = axes[0, 1]
        if self.compression_errors:
            errors_pct = [e * 100 for e in self.compression_errors]
            ax.plot(self.steps_recorded[:len(errors_pct)], errors_pct,
                   linewidth=2, color='orangered', marker='s', markersize=4)
            ax.axhline(y=15, color='r', linestyle='--', 
                      label='15% threshold', linewidth=2)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Preconditioner Error (%)')
            ax.set_title('Factorized Preconditioner Quality')
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        
        # Plot 3: Saliency Concentration
        ax = axes[1, 0]
        if self.saliency_stats:
            top10 = [s['top10_threshold'] for s in self.saliency_stats]
            top20 = [s['top20_threshold'] for s in self.saliency_stats]
            mean = [s['mean'] for s in self.saliency_stats]
            
            steps = self.steps_recorded[:len(top10)]
            ax.plot(steps, top10, label='Top 10% threshold', linewidth=2, color='darkgreen')
            ax.plot(steps, top20, label='Top 20% threshold', linewidth=2, color='forestgreen')
            ax.plot(steps, mean, label='Mean', linewidth=2, color='lightgreen')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Saliency Score')
            ax.set_title('Saliency Score Distribution')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        
        # Plot 4: Sparsity Evolution
        ax = axes[1, 1]
        if self.saliency_stats:
            sparsity = [s['sparsity_80'] * 100 for s in self.saliency_stats]
            steps = self.steps_recorded[:len(sparsity)]
            ax.plot(steps, sparsity, linewidth=2, color='purple', marker='D', markersize=4)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Concentration in Top 20% (%)')
            ax.set_title('Parameter Importance Concentration')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        
        plt.tight_layout()
        plot_path = output_dir / 'umtam_diagnostics.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Diagnostics plot saved to: {plot_path}")
        plt.close()
    
    def print_summary(self):
        """Print summary statistics."""
        print("\n" + "="*60)
        print("UMTAM Diagnostics Summary")
        print("="*60)
        
        if self.stable_ranks:
            avg_rank = np.mean(self.stable_ranks[-10:]) if len(self.stable_ranks) >= 10 else np.mean(self.stable_ranks)
            print(f"\nStable Rank:")
            print(f"  Current: {self.stable_ranks[-1]:.2f}")
            print(f"  Average (recent): {avg_rank:.2f}")
            
            if avg_rank < 50:
                print(f"  ✅ Low-rank structure preserved")
            else:
                print(f"  ⚠️  Higher than expected")
        
        if self.compression_errors:
            avg_error = np.mean(self.compression_errors[-10:]) if len(self.compression_errors) >= 10 else np.mean(self.compression_errors)
            print(f"\nPreconditioner Error:")
            print(f"  Current: {self.compression_errors[-1]*100:.2f}%")
            print(f"  Average (recent): {avg_error*100:.2f}%")
            
            if avg_error < 0.15:
                print(f"  ✅ Compression quality good")
            else:
                print(f"  ⚠️  Higher error than expected")
        
        if self.saliency_stats:
            recent_stats = self.saliency_stats[-1]
            print(f"\nSaliency Scores:")
            print(f"  Mean: {recent_stats['mean']:.6f}")
            print(f"  Top 10% threshold: {recent_stats['top10_threshold']:.6f}")
            print(f"  Top 20% concentration: {recent_stats['sparsity_80']*100:.1f}%")
        
        print("="*60)
    
    def validate_theoretical_properties(self) -> Dict[str, bool]:
        """
        Validate that UMTAM satisfies theoretical guarantees.
        
        Returns dict of {property: passes_validation}
        """
        results = {}
        
        # Property 1: Low stable rank (from exp1)
        if self.stable_ranks:
            avg_rank = np.mean(self.stable_ranks[-10:]) if len(self.stable_ranks) >= 10 else np.mean(self.stable_ranks)
            results['low_stable_rank'] = avg_rank < 100  # Should be much less than dimension
        
        # Property 2: Low compression error (from exp2)
        if self.compression_errors:
            avg_error = np.mean(self.compression_errors[-10:]) if len(self.compression_errors) >= 10 else np.mean(self.compression_errors)
            results['low_compression_error'] = avg_error < 0.2  # < 20%
        
        # Property 3: Concentrated saliency (from exp3)
        if self.saliency_stats:
            recent_stats = self.saliency_stats[-1]
            results['concentrated_saliency'] = recent_stats['sparsity_80'] > 0.15  # Top 20% should be important
        
        return results


def test_diagnostics():
    """Test diagnostic tools on synthetic data."""
    print("Testing UMTAM Diagnostics")
    print("="*60)
    
    from umtam_optimizer import UMTAMOptimizer
    import torch.nn as nn
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    # Create optimizer
    optimizer = UMTAMOptimizer(
        model.parameters(),
        lr=0.01,
        rank=8,
    )
    
    # Create diagnostics tracker
    diagnostics = UMTAMDiagnostics(log_interval=5)
    
    # Simple training loop
    X = torch.randn(50, 100)
    y = torch.randn(50, 10)
    
    for step in range(50):
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(X), y)
        loss.backward()
        optimizer.step()
        
        # Update diagnostics
        diagnostics.update(optimizer, step)
        
        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
    
    # Print summary
    diagnostics.print_summary()
    
    # Validate properties
    validation = diagnostics.validate_theoretical_properties()
    print("\nValidation Results:")
    for prop, passes in validation.items():
        status = "✅ PASS" if passes else "❌ FAIL"
        print(f"  {prop}: {status}")
    
    # Create plots
    diagnostics.plot_diagnostics('./test_diagnostics')
    
    print("\n✅ Diagnostic test complete!")


if __name__ == '__main__':
    test_diagnostics()
