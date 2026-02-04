"""
UMTAM Optimizer Implementation
===============================
Unified Momentum-Trajectory Aware Training and Merging (UMTAM)

Based on: "Bridging Training and Merging Through Momentum-Aware Optimization"

This optimizer implements Algorithm 1 from the paper:
- Dual momentum factorization with low-rank SVD
- Factorized second-order statistics (Adafactor-style)
- Error feedback for compression
- Adaptive preconditioning
- Task saliency tracking
"""

import torch
import torch.optim as optim
from typing import Optional, Callable, Tuple
import math


class UMTAMOptimizer(optim.Optimizer):
    """
    UMTAM: Unified Momentum-Trajectory Aware Training and Merging Optimizer
    
    Implements memory-efficient optimization through:
    1. Low-rank momentum factorization (U, Σ, V)
    2. Factorized second-order statistics (R, C)
    3. Error feedback mechanism
    4. Curvature-aware adaptive preconditioning
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        rank: Rank for momentum factorization (default: 32)
        beta1: First moment decay rate (default: 0.9)
        beta2: Second moment decay rate (default: 0.999)
        gamma: Error feedback decay (default: 0.9)
        eps: Regularization constant (default: 1e-8)
        alpha: Saliency tracking decay (default: 0.99)
        grad_clip: Gradient clipping threshold (default: 1.0)
        svd_frequency: How often to recompute SVD (default: 1)
        adaptive_rank: Whether to adaptively adjust rank (default: False)
        min_rank: Minimum rank for adaptive adjustment (default: 8)
        max_rank: Maximum rank for adaptive adjustment (default: 128)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        rank: int = 32,
        beta1: float = 0.9,
        beta2: float = 0.999,
        gamma: float = 0.9,
        eps: float = 1e-8,
        alpha: float = 0.99,
        grad_clip: float = 1.0,
        svd_frequency: int = 1,
        adaptive_rank: bool = False,
        min_rank: int = 8,
        max_rank: int = 128,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if rank <= 0:
            raise ValueError(f"Invalid rank: {rank}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"Invalid gamma: {gamma}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
            
        defaults = dict(
            lr=lr,
            rank=rank,
            beta1=beta1,
            beta2=beta2,
            gamma=gamma,
            eps=eps,
            alpha=alpha,
            grad_clip=grad_clip,
            svd_frequency=svd_frequency,
            adaptive_rank=adaptive_rank,
            min_rank=min_rank,
            max_rank=max_rank,
            weight_decay=weight_decay,
        )
        super(UMTAMOptimizer, self).__init__(params, defaults)
        
    def _init_group_state(self, group, param):
        """Initialize optimizer state for a parameter."""
        state = self.state[param]
        
        # Skip initialization if already done
        if len(state) > 0:
            return
            
        # Get parameter shape
        shape = param.shape
        device = param.device
        dtype = param.dtype
        
        # Initialize step counter
        state['step'] = 0
        
        # Store initial parameters for saliency tracking
        state['w0'] = param.data.clone().detach()
        
        # Handle matrix parameters (2D) with full UMTAM
        if len(shape) == 2:
            m, n = shape
            rank = min(group['rank'], min(m, n))
            
            # Momentum factors: U ∈ R^{m×r}, Σ ∈ R^{r×r}, V ∈ R^{n×r}
            state['U'] = torch.randn(m, rank, device=device, dtype=dtype) / math.sqrt(m * rank)
            state['Sigma'] = torch.eye(rank, device=device, dtype=dtype) * group['eps']
            state['V'] = torch.randn(n, rank, device=device, dtype=dtype) / math.sqrt(n * rank)
            
            # Error accumulator E ∈ R^{m×n}
            state['E'] = torch.zeros_like(param.data)
            
            # Factorized second moments: R ∈ R^m, C ∈ R^n
            state['R'] = torch.ones(m, device=device, dtype=dtype) * group['eps']
            state['C'] = torch.ones(n, device=device, dtype=dtype) * group['eps']
            
            # Task saliency scores S ∈ R^{m×n}
            state['S'] = torch.zeros_like(param.data)
            
            # Current rank
            state['current_rank'] = rank
            
        # Handle vector/bias parameters (1D) with standard Adam-like update
        elif len(shape) == 1:
            # First moment
            state['m'] = torch.zeros_like(param.data)
            # Second moment
            state['v'] = torch.zeros_like(param.data)
            
        else:
            # For higher-dimensional tensors, reshape to 2D
            # Reshape to (first_dim, rest)
            state['original_shape'] = shape
            state['reshaped'] = True
            
            m = shape[0]
            n = math.prod(shape[1:])
            rank = min(group['rank'], min(m, n))
            
            # Momentum factors
            state['U'] = torch.randn(m, rank, device=device, dtype=dtype) / math.sqrt(m * rank)
            state['Sigma'] = torch.eye(rank, device=device, dtype=dtype) * group['eps']
            state['V'] = torch.randn(n, rank, device=device, dtype=dtype) / math.sqrt(n * rank)
            
            # Error accumulator
            state['E'] = torch.zeros(m, n, device=device, dtype=dtype)
            
            # Factorized second moments
            state['R'] = torch.ones(m, device=device, dtype=dtype) * group['eps']
            state['C'] = torch.ones(n, device=device, dtype=dtype) * group['eps']
            
            # Task saliency
            state['S'] = torch.zeros(m, n, device=device, dtype=dtype)
            
            state['current_rank'] = rank
    
    def _truncated_svd(self, M: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute truncated SVD: M ≈ U @ Sigma @ V^T
        
        Args:
            M: Input matrix (m × n)
            rank: Target rank
            
        Returns:
            U (m × r), Sigma (r × r diagonal), V (n × r)
        """
        # Use PyTorch's SVD
        # For large matrices, this could be replaced with randomized SVD for speed
        try:
            U_full, S_full, Vh_full = torch.linalg.svd(M, full_matrices=False)
            
            # Truncate to rank
            actual_rank = min(rank, len(S_full))
            U = U_full[:, :actual_rank]
            S = torch.diag(S_full[:actual_rank])
            V = Vh_full[:actual_rank, :].T  # Transpose to get V from V^T
            
            return U, S, V
            
        except RuntimeError as e:
            # Fallback: if SVD fails, use previous values with small perturbation
            print(f"SVD failed: {e}. Using fallback.")
            m, n = M.shape
            device = M.device
            dtype = M.dtype
            U = torch.randn(m, rank, device=device, dtype=dtype) / math.sqrt(m * rank)
            S = torch.eye(rank, device=device, dtype=dtype) * 0.01
            V = torch.randn(n, rank, device=device, dtype=dtype) / math.sqrt(n * rank)
            return U, S, V
    
    def _compute_stable_rank(self, M: torch.Tensor) -> float:
        """Compute stable rank: ||M||_F^2 / ||M||_2^2"""
        frob_norm_sq = torch.sum(M ** 2)
        spec_norm_sq = torch.linalg.matrix_norm(M, ord=2) ** 2
        if spec_norm_sq < 1e-10:
            return 1.0
        return (frob_norm_sq / spec_norm_sq).item()
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Perform a single optimization step.
        
        Args:
            closure: Optional closure to recompute the loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                
                # Initialize state if needed
                self._init_group_state(group, param)
                
                state = self.state[param]
                grad = param.grad
                
                # Increment step counter
                state['step'] += 1
                step = state['step']
                
                # Get hyperparameters
                lr = group['lr']
                beta1 = group['beta1']
                beta2 = group['beta2']
                gamma = group['gamma']
                eps = group['eps']
                alpha = group['alpha']
                grad_clip = group['grad_clip']
                svd_freq = group['svd_frequency']
                weight_decay = group['weight_decay']
                
                # Apply weight decay
                if weight_decay > 0:
                    param.data.mul_(1 - lr * weight_decay)
                
                # Handle 1D parameters (bias) with standard Adam
                if 'v' in state:
                    # Adam-like update for bias terms
                    m = state['m']
                    v = state['v']
                    
                    # Gradient clipping
                    grad_norm = torch.norm(grad)
                    if grad_norm > grad_clip:
                        grad = grad * (grad_clip / grad_norm)
                    
                    # Update moments
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
                    # Bias correction
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    
                    # Update parameters
                    step_size = lr / bias_correction1
                    denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    param.data.addcdiv_(m, denom, value=-step_size)
                    
                    continue
                
                # Handle 2D parameters with UMTAM
                # Reshape if needed
                if 'reshaped' in state and state['reshaped']:
                    original_shape = state['original_shape']
                    grad_2d = grad.reshape(original_shape[0], -1)
                    param_2d = param.data.reshape(original_shape[0], -1)
                else:
                    grad_2d = grad
                    param_2d = param.data
                
                # Gradient clipping
                grad_norm = torch.norm(grad_2d, p='fro')
                if grad_norm > grad_clip:
                    grad_2d = grad_2d * (grad_clip / grad_norm)
                
                # Get state variables
                U = state['U']
                Sigma = state['Sigma']
                V = state['V']
                E = state['E']
                R = state['R']
                C = state['C']
                S = state['S']
                w0 = state['w0']
                
                if 'reshaped' in state and state['reshaped']:
                    w0 = w0.reshape(original_shape[0], -1)
                
                # 1. Momentum update with error feedback
                # M̃_t = β₁ · U_t · Σ_t · V_t^T + (1-β₁) · G_t + γ · E_{t-1}
                M_reconstructed = U @ Sigma @ V.T
                M_tilde = beta1 * M_reconstructed + (1 - beta1) * grad_2d + gamma * E
                
                # 2. Truncated SVD (only every svd_freq steps to save computation)
                if step % svd_freq == 0:
                    current_rank = state['current_rank']
                    U_new, Sigma_new, V_new = self._truncated_svd(M_tilde, current_rank)
                    
                    # Update state
                    state['U'] = U_new
                    state['Sigma'] = Sigma_new
                    state['V'] = V_new
                    U, Sigma, V = U_new, Sigma_new, V_new
                    
                    # 3. Compression error with error feedback
                    # E_t = M̃_t - U_{t+1} · Σ_{t+1} · V_{t+1}^T
                    M_compressed = U @ Sigma @ V.T
                    E_new = M_tilde - M_compressed
                    state['E'] = E_new
                else:
                    # Use existing factorization
                    M_compressed = U @ Sigma @ V.T
                
                # 4. Update factorized second moments
                # R_t = β₂ · R_{t-1} + (1-β₂) · diag(G_t · G_t^T)
                # C_t = β₂ · C_{t-1} + (1-β₂) · diag(G_t^T · G_t)
                R_new = beta2 * R + (1 - beta2) * torch.sum(grad_2d ** 2, dim=1)
                C_new = beta2 * C + (1 - beta2) * torch.sum(grad_2d ** 2, dim=0)
                state['R'] = R_new
                state['C'] = C_new
                
                # 5. Construct factorized preconditioner
                # Ŝ_t = (R_t · C_t^T) / (1_m^T · R_t)
                # This is an outer product approximation
                S_hat = torch.outer(R_new, C_new) / (torch.sum(R_new) + eps)
                
                # 6. Adaptive regularization
                # ε_t = ε · max(1, ||G_t||_F / ||W_t||_F)
                param_norm = torch.norm(param_2d, p='fro')
                if param_norm > eps:
                    eps_adaptive = eps * max(1.0, grad_norm / param_norm)
                else:
                    eps_adaptive = eps
                
                # 7. Compute preconditioner P_t = (Ŝ_t + ε_t · I)^{-1/2}
                # Element-wise inverse square root
                P = 1.0 / torch.sqrt(S_hat + eps_adaptive)
                
                # 8. Parameter update
                # W_{t+1} = W_t - η_t · P_t ⊙ M_{t+1}
                update = lr * P * M_compressed
                param_2d.sub_(update)
                
                # 9. Update task saliency scores
                # S_t^{(i,j)} = α · S_{t-1}^{(i,j)} + (1-α) · (W_t^{(i,j)} - W_0^{(i,j)})² · √(R_t^{(i)} · C_t^{(j)})
                param_deviation = (param_2d - w0) ** 2
                curvature_weight = torch.sqrt(torch.outer(R_new, C_new))
                S_new = alpha * S + (1 - alpha) * param_deviation * curvature_weight
                state['S'] = S_new
                
                # Reshape back if needed
                if 'reshaped' in state and state['reshaped']:
                    param.data = param_2d.reshape(original_shape)
        
        return loss
    
    def get_saliency_scores(self):
        """
        Retrieve task saliency scores for all parameters.
        Useful for model merging later.
        
        Returns:
            Dictionary mapping parameter names to saliency scores
        """
        saliency_dict = {}
        for group in self.param_groups:
            for param in group['params']:
                if param in self.state and 'S' in self.state[param]:
                    state = self.state[param]
                    S = state['S']
                    
                    # Reshape if needed
                    if 'reshaped' in state and state['reshaped']:
                        S = S.reshape(state['original_shape'])
                    
                    # Use parameter's id as key (you can customize this)
                    saliency_dict[id(param)] = S.clone().detach()
        
        return saliency_dict
    
    def get_memory_usage(self):
        """
        Calculate memory usage of optimizer state.
        
        Returns:
            Dictionary with memory statistics
        """
        total_params = 0
        total_state_memory = 0
        
        for group in self.param_groups:
            for param in group['params']:
                if param not in self.state:
                    continue
                    
                state = self.state[param]
                param_elements = param.numel()
                total_params += param_elements
                
                # Count state memory
                if 'U' in state:
                    total_state_memory += state['U'].numel()
                    total_state_memory += state['Sigma'].numel()
                    total_state_memory += state['V'].numel()
                    total_state_memory += state['E'].numel()
                    total_state_memory += state['R'].numel()
                    total_state_memory += state['C'].numel()
                    total_state_memory += state['S'].numel()
                    total_state_memory += state['w0'].numel()
                elif 'v' in state:
                    total_state_memory += state['m'].numel()
                    total_state_memory += state['v'].numel()
        
        # Bytes (assuming float32)
        param_memory_mb = (total_params * 4) / (1024 ** 2)
        state_memory_mb = (total_state_memory * 4) / (1024 ** 2)
        
        return {
            'total_params': total_params,
            'param_memory_mb': param_memory_mb,
            'state_memory_mb': state_memory_mb,
            'total_memory_mb': param_memory_mb + state_memory_mb,
            'state_to_param_ratio': state_memory_mb / param_memory_mb if param_memory_mb > 0 else 0,
        }
