"""
UMTAM Optimizer Implementation - STABILIZED VERSION
====================================================
Fixed version with numerical stability improvements

Key fixes:
1. Preconditioner clamping to prevent explosion
2. Update norm clipping
3. Better error handling
4. Safer initialization
"""

import torch
import torch.optim as optim
from typing import Optional, Callable, Tuple
import math


class UMTAMOptimizerStable(optim.Optimizer):
    """
    UMTAM: Unified Momentum-Trajectory Aware Training and Merging Optimizer
    STABILIZED VERSION with numerical safeguards
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-4, reduced from 1e-3 for stability)
        rank: Rank for momentum factorization (default: 32)
        beta1: First moment decay rate (default: 0.9)
        beta2: Second moment decay rate (default: 0.999)
        gamma: Error feedback decay (default: 0.9)
        eps: Regularization constant (default: 1e-8)
        alpha: Saliency tracking decay (default: 0.99)
        grad_clip: Gradient clipping threshold (default: 1.0)
        svd_frequency: How often to recompute SVD (default: 10, reduced from 50)
        max_precond: Maximum preconditioner value (default: 10.0)
        max_update_norm: Maximum update norm (default: 1.0)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-4,  # Reduced from 1e-3
        rank: int = 32,
        beta1: float = 0.9,
        beta2: float = 0.999,
        gamma: float = 0.9,
        eps: float = 1e-8,
        alpha: float = 0.99,
        grad_clip: float = 1.0,
        svd_frequency: int = 10,  # Reduced from 50
        max_precond: float = 10.0,  # NEW: Limit preconditioner
        max_update_norm: float = 1.0,  # NEW: Limit update size
        min_stat: float = 1e-4,  # NEW: Minimum statistic value
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
            max_precond=max_precond,
            max_update_norm=max_update_norm,
            min_stat=min_stat,
            weight_decay=weight_decay,
        )
        super(UMTAMOptimizerStable, self).__init__(params, defaults)
        
    def _init_group_state(self, group, param):
        """Initialize optimizer state for a parameter."""
        state = self.state[param]
        
        if len(state) > 0:
            return
            
        shape = param.shape
        device = param.device
        dtype = param.dtype
        
        state['step'] = 0
        state['w0'] = param.data.clone().detach()
        
        if len(shape) == 2:
            m, n = shape
            rank = min(group['rank'], min(m, n))
            
            # Initialize with small values for stability
            state['U'] = torch.randn(m, rank, device=device, dtype=dtype) * 0.01
            state['Sigma'] = torch.eye(rank, device=device, dtype=dtype) * 0.01
            state['V'] = torch.randn(n, rank, device=device, dtype=dtype) * 0.01
            state['E'] = torch.zeros_like(param.data)
            
            # Initialize with safe minimum values
            min_stat = group.get('min_stat', 1e-4)
            state['R'] = torch.ones(m, device=device, dtype=dtype) * min_stat
            state['C'] = torch.ones(n, device=device, dtype=dtype) * min_stat
            state['S'] = torch.zeros_like(param.data)
            state['current_rank'] = rank
            
        elif len(shape) == 1:
            state['m'] = torch.zeros_like(param.data)
            state['v'] = torch.zeros_like(param.data)
            
        else:
            # Higher-dimensional tensors
            state['original_shape'] = shape
            state['reshaped'] = True
            
            m = shape[0]
            n = math.prod(shape[1:])
            rank = min(group['rank'], min(m, n))
            
            state['U'] = torch.randn(m, rank, device=device, dtype=dtype) * 0.01
            state['Sigma'] = torch.eye(rank, device=device, dtype=dtype) * 0.01
            state['V'] = torch.randn(n, rank, device=device, dtype=dtype) * 0.01
            state['E'] = torch.zeros(m, n, device=device, dtype=dtype)
            
            min_stat = group.get('min_stat', 1e-4)
            state['R'] = torch.ones(m, device=device, dtype=dtype) * min_stat
            state['C'] = torch.ones(n, device=device, dtype=dtype) * min_stat
            state['S'] = torch.zeros(m, n, device=device, dtype=dtype)
            state['current_rank'] = rank
    
    def _truncated_svd(self, M: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute truncated SVD with error handling."""
        try:
            U_full, S_full, Vh_full = torch.linalg.svd(M, full_matrices=False)
            actual_rank = min(rank, len(S_full))
            U = U_full[:, :actual_rank]
            S = torch.diag(S_full[:actual_rank])
            V = Vh_full[:actual_rank, :].T
            return U, S, V
        except RuntimeError as e:
            print(f"SVD failed: {e}. Using previous values.")
            # Return current values (will be handled by caller)
            raise
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform optimization step with stability safeguards."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                
                self._init_group_state(group, param)
                state = self.state[param]
                grad = param.grad
                
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
                max_precond = group['max_precond']
                max_update_norm = group['max_update_norm']
                min_stat = group['min_stat']
                
                if weight_decay > 0:
                    param.data.mul_(1 - lr * weight_decay)
                
                # Handle 1D parameters (bias)
                if 'v' in state:
                    m = state['m']
                    v = state['v']
                    
                    grad_norm = torch.norm(grad)
                    if grad_norm > grad_clip:
                        grad = grad * (grad_clip / grad_norm)
                    
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    
                    step_size = lr / bias_correction1
                    denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    param.data.addcdiv_(m, denom, value=-step_size)
                    continue
                
                # Handle 2D parameters
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
                
                # Get state
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
                
                # 1. Momentum update
                M_reconstructed = U @ Sigma @ V.T
                M_tilde = beta1 * M_reconstructed + (1 - beta1) * grad_2d + gamma * E
                
                # 2. SVD (with error handling)
                if step % svd_freq == 0:
                    try:
                        current_rank = state['current_rank']
                        U_new, Sigma_new, V_new = self._truncated_svd(M_tilde, current_rank)
                        state['U'] = U_new
                        state['Sigma'] = Sigma_new
                        state['V'] = V_new
                        U, Sigma, V = U_new, Sigma_new, V_new
                        
                        M_compressed = U @ Sigma @ V.T
                        E_new = M_tilde - M_compressed
                        state['E'] = E_new
                    except RuntimeError:
                        # If SVD fails, use current values
                        M_compressed = U @ Sigma @ V.T
                else:
                    M_compressed = U @ Sigma @ V.T
                
                # 3. Update second moments with CLAMPING
                R_new = beta2 * R + (1 - beta2) * torch.sum(grad_2d ** 2, dim=1)
                C_new = beta2 * C + (1 - beta2) * torch.sum(grad_2d ** 2, dim=0)
                
                # STABILITY FIX: Clamp to minimum values
                R_new = torch.clamp(R_new, min=min_stat)
                C_new = torch.clamp(C_new, min=min_stat)
                
                state['R'] = R_new
                state['C'] = C_new
                
                # 4. Construct preconditioner with STABILITY
                S_hat = torch.outer(R_new, C_new) / (torch.sum(R_new) + eps)
                
                # 5. Adaptive regularization
                param_norm = torch.norm(param_2d, p='fro')
                if param_norm > eps:
                    eps_adaptive = eps * max(1.0, grad_norm / param_norm)
                else:
                    eps_adaptive = eps
                
                # 6. Compute preconditioner with CLAMPING
                P = 1.0 / torch.sqrt(S_hat + eps_adaptive)
                # STABILITY FIX: Limit maximum preconditioner value
                P = torch.clamp(P, max=max_precond)
                
                # 7. Parameter update with NORM CLIPPING
                update = lr * P * M_compressed
                
                # STABILITY FIX: Clip update norm
                update_norm = torch.norm(update, p='fro')
                if update_norm > max_update_norm:
                    update = update * (max_update_norm / update_norm)
                
                param_2d.sub_(update)
                
                # 8. Update saliency
                param_deviation = (param_2d - w0) ** 2
                curvature_weight = torch.sqrt(torch.outer(R_new, C_new))
                S_new = alpha * S + (1 - alpha) * param_deviation * curvature_weight
                state['S'] = S_new
                
                # Reshape back
                if 'reshaped' in state and state['reshaped']:
                    param.data = param_2d.reshape(original_shape)
        
        return loss
    
    def get_memory_usage(self):
        """Calculate memory usage."""
        total_params = 0
        total_state_memory = 0
        
        for group in self.param_groups:
            for param in group['params']:
                if param not in self.state:
                    # If state not initialized, initialize it
                    self._init_group_state(group, param)
                    
                state = self.state[param]
                param_elements = param.numel()
                total_params += param_elements
                
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
        
        param_memory_mb = (total_params * 4) / (1024 ** 2)
        state_memory_mb = (total_state_memory * 4) / (1024 ** 2)
        
        return {
            'total_params': total_params,
            'param_memory_mb': param_memory_mb,
            'state_memory_mb': state_memory_mb,
            'total_memory_mb': param_memory_mb + state_memory_mb,
            'state_to_param_ratio': state_memory_mb / param_memory_mb if param_memory_mb > 0 else 0,
        }
    
    def get_saliency_scores(self):
        """Retrieve task saliency scores."""
        saliency_dict = {}
        for group in self.param_groups:
            for param in group['params']:
                if param in self.state and 'S' in self.state[param]:
                    state = self.state[param]
                    S = state['S']
                    if 'reshaped' in state and state['reshaped']:
                        S = S.reshape(state['original_shape'])
                    saliency_dict[id(param)] = S.clone().detach()
        return saliency_dict