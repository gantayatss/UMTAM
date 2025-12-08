"""
UMTAM Pruning Experiment - Standalone Script
============================================

Run this script locally instead of using the Colab notebook.

Usage:
    python umtam_pruning_experiment.py --quick  # Fast test (2 tasks)
    python umtam_pruning_experiment.py --full   # Full experiment (4 tasks)
    
Requirements:
    pip install transformers datasets torch evaluate scikit-learn matplotlib seaborn tqdm
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import evaluate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional
import json
import argparse
import copy
import warnings
import math
warnings.filterwarnings('ignore')


# ==================== UMTAM OPTIMIZER ====================

class UMTAMOptimizer(optim.Optimizer):
    """
    UMTAM: Unified Momentum-Trajectory Aware Training and Merging Optimizer
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        rank: int = 32,
        beta1: float = 0.9,
        beta2: float = 0.999,
        gamma: float = 0.9,
        eps: float = 1e-8,
        alpha: float = 0.99,
        grad_clip: float = 1.0,
        svd_frequency: int = 10,
        max_precond: float = 10.0,
        max_update_norm: float = 1.0,
        min_stat: float = 1e-4,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr, rank=rank, beta1=beta1, beta2=beta2, gamma=gamma,
            eps=eps, alpha=alpha, grad_clip=grad_clip,
            svd_frequency=svd_frequency, max_precond=max_precond,
            max_update_norm=max_update_norm, min_stat=min_stat,
            weight_decay=weight_decay,
        )
        super(UMTAMOptimizer, self).__init__(params, defaults)
        
    def _init_group_state(self, group, param):
        """Initialize optimizer state."""
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
            
            state['U'] = torch.randn(m, rank, device=device, dtype=dtype) * 0.01
            state['Sigma'] = torch.eye(rank, device=device, dtype=dtype) * 0.01
            state['V'] = torch.randn(n, rank, device=device, dtype=dtype) * 0.01
            state['E'] = torch.zeros_like(param.data)
            
            min_stat = group.get('min_stat', 1e-4)
            state['R'] = torch.ones(m, device=device, dtype=dtype) * min_stat
            state['C'] = torch.ones(n, device=device, dtype=dtype) * min_stat
            state['S'] = torch.zeros_like(param.data)
            state['current_rank'] = rank
            
        elif len(shape) == 1:
            state['m'] = torch.zeros_like(param.data)
            state['v'] = torch.zeros_like(param.data)
    
    def _truncated_svd(self, M: torch.Tensor, rank: int):
        """Compute truncated SVD."""
        try:
            U_full, S_full, Vh_full = torch.linalg.svd(M, full_matrices=False)
            actual_rank = min(rank, len(S_full))
            U = U_full[:, :actual_rank]
            S = torch.diag(S_full[:actual_rank])
            V = Vh_full[:actual_rank, :].T
            return U, S, V
        except RuntimeError:
            raise
    
    @torch.no_grad()
    def step(self, closure: Optional = None):
        """Optimization step."""
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
                
                # 1D parameters (bias)
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
                
                # 2D parameters
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
                
                # Momentum update with error feedback
                M_reconstructed = U @ Sigma @ V.T
                M_tilde = beta1 * M_reconstructed + (1 - beta1) * grad_2d + gamma * E
                
                # SVD
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
                        M_compressed = U @ Sigma @ V.T
                else:
                    M_compressed = U @ Sigma @ V.T
                
                # Update second moments
                R_new = beta2 * R + (1 - beta2) * torch.sum(grad_2d ** 2, dim=1)
                C_new = beta2 * C + (1 - beta2) * torch.sum(grad_2d ** 2, dim=0)
                R_new = torch.clamp(R_new, min=min_stat)
                C_new = torch.clamp(C_new, min=min_stat)
                state['R'] = R_new
                state['C'] = C_new
                
                # Construct preconditioner
                S_hat = torch.outer(R_new, C_new) / (torch.sum(R_new) + eps)
                
                # Adaptive regularization
                param_norm = torch.norm(param_2d, p='fro')
                eps_adaptive = eps * max(1.0, grad_norm / param_norm) if param_norm > eps else eps
                
                # Compute preconditioner
                P = 1.0 / torch.sqrt(S_hat + eps_adaptive)
                P = torch.clamp(P, max=max_precond)
                
                # Parameter update
                update = lr * P * M_compressed
                update_norm = torch.norm(update, p='fro')
                if update_norm > max_update_norm:
                    update = update * (max_update_norm / update_norm)
                param_2d.sub_(update)
                
                # Update saliency
                param_deviation = (param_2d - w0) ** 2
                curvature_weight = torch.sqrt(torch.outer(R_new, C_new))
                S_new = alpha * S + (1 - alpha) * param_deviation * curvature_weight
                state['S'] = S_new
        
        return loss
    
    def get_saliency_scores(self):
        """Get saliency scores."""
        saliency_dict = {}
        for group in self.param_groups:
            for param in group['params']:
                if param in self.state and 'S' in self.state[param]:
                    saliency_dict[id(param)] = self.state[param]['S'].clone().detach()
        return saliency_dict
    
    def get_initial_weights(self):
        """Get initial weights."""
        w0_dict = {}
        for group in self.param_groups:
            for param in group['params']:
                if param in self.state and 'w0' in self.state[param]:
                    w0_dict[id(param)] = self.state[param]['w0'].clone().detach()
        return w0_dict


# ==================== EXPERIMENT FUNCTIONS ====================

def train_model(model, optimizer, train_dataset, num_epochs, batch_size, task_name, device):
    """Train model."""
    model.train()
    model.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    total_steps = len(train_loader) * num_epochs
    pbar = tqdm(total=total_steps, desc=f"Training {task_name}")
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        print(f"\\nEpoch {epoch+1}/{num_epochs} - Loss: {epoch_loss/len(train_loader):.4f}")
    
    pbar.close()
    return model


def evaluate_model(model, eval_dataset, batch_size, task_name, metric_name, device):
    """Evaluate model."""
    model.eval()
    model.to(device)
    
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
    metric = evaluate.load('glue', task_name)
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"Evaluating {task_name}", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            metric.add_batch(predictions=predictions.cpu(), references=labels.cpu())
    
    results = metric.compute()
    return results[metric_name]


def apply_pruning_mask(model, base_model, saliency_scores, w0_dict, density, use_curvature=True):
    """Apply pruning mask."""
    pruned_model = copy.deepcopy(base_model)
    
    all_scores = []
    param_info = []
    
    for name, param in model.named_parameters():
        param_id = id(param)
        
        if param_id in saliency_scores and len(param.shape) == 2:
            base_param = dict(base_model.named_parameters())[name]
            w0 = w0_dict.get(param_id, base_param.data)
            task_vector = param.data - w0
            
            if use_curvature:
                scores = saliency_scores[param_id]
            else:
                scores = torch.abs(task_vector)
            
            scores_flat = scores.flatten().cpu()
            all_scores.append(scores_flat)
            param_info.append({
                'name': name,
                'param': param,
                'task_vector': task_vector,
                'w0': w0,
            })
    
    all_scores_flat = torch.cat(all_scores)
    k = int(len(all_scores_flat) * (density / 100.0))
    threshold = torch.topk(all_scores_flat, k).values[-1] if k > 0 else float('inf')
    
    for info in param_info:
        name = info['name']
        task_vector = info['task_vector']
        w0 = info['w0']
        
        if use_curvature:
            param_scores = saliency_scores[id(info['param'])]
        else:
            param_scores = torch.abs(task_vector)
        
        mask = (param_scores >= threshold).float()
        pruned_param = dict(pruned_model.named_parameters())[name]
        pruned_param.data = w0 + mask * task_vector
    
    return pruned_model


def run_experiment(args):
    """Main experiment."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Task configs
    task_configs = {
        'sst2': {'subset': 'sst2', 'num_labels': 2, 'metric': 'accuracy'},
        'mrpc': {'subset': 'mrpc', 'num_labels': 2, 'metric': 'f1'},
        'cola': {'subset': 'cola', 'num_labels': 2, 'metric': 'matthews_correlation'},
        'qqp': {'subset': 'qqp', 'num_labels': 2, 'metric': 'f1'},
    }
    
    tasks = ['sst2', 'mrpc'] if args.quick else ['sst2', 'mrpc', 'cola', 'qqp']
    densities = [1, 10, 40, 90] if args.quick else [1, 5, 10, 20, 40, 60, 90]
    
    all_results = {}
    
    for task_name in tasks:
        print(f"\\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")
        
        # Load tokenizer and dataset
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        dataset = load_dataset('glue', task_configs[task_name]['subset'])
        
        # Tokenize
        def tokenize(examples):
            return tokenizer(examples['sentence'], padding='max_length', 
                           truncation=True, max_length=args.max_length)
        
        tokenized = dataset.map(tokenize, batched=True)
        
        # Prepare datasets
        train_dataset = tokenized['train'].select(range(min(args.max_train_samples, len(tokenized['train']))))
        eval_dataset = tokenized['validation'].select(range(min(args.max_eval_samples, len(tokenized['validation']))))
        
        if 'label' in train_dataset.features:
            train_dataset = train_dataset.rename_column('label', 'labels')
            eval_dataset = eval_dataset.rename_column('label', 'labels')
        
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        # Models
        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=task_configs[task_name]['num_labels']
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=task_configs[task_name]['num_labels']
        )
        
        # Optimizer
        optimizer = UMTAMOptimizer(model.parameters(), lr=args.lr, rank=args.rank)
        
        # Train
        model = train_model(model, optimizer, train_dataset, args.num_epochs, 
                          args.batch_size, task_name, device)
        
        # Get saliency
        saliency_scores = optimizer.get_saliency_scores()
        w0_dict = optimizer.get_initial_weights()
        
        # Test pruning
        results = {'umtam': [], 'magnitude': [], 'densities': densities}
        
        for density in densities:
            print(f"\\nDensity: {density}%")
            
            # UMTAM
            pruned = apply_pruning_mask(model, base_model, saliency_scores, w0_dict, density, True)
            perf = evaluate_model(pruned, eval_dataset, args.batch_size, 
                                task_name, task_configs[task_name]['metric'], device)
            results['umtam'].append(perf)
            print(f"  UMTAM: {perf:.4f}")
            
            # Magnitude
            pruned = apply_pruning_mask(model, base_model, saliency_scores, w0_dict, density, False)
            perf = evaluate_model(pruned, eval_dataset, args.batch_size, 
                                task_name, task_configs[task_name]['metric'], device)
            results['magnitude'].append(perf)
            print(f"  Magnitude: {perf:.4f}")
        
        all_results[task_name] = results
    
    # Save
    with open('results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\\n✓ Experiment complete! Results saved to results.json")
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--full', action='store_true', help='Full experiment')
    parser.add_argument('--model_name', default='bert-base-uncased')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--rank', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--max_train_samples', type=int, default=1000)
    parser.add_argument('--max_eval_samples', type=int, default=500)
    
    args = parser.parse_args()
    
    if args.quick:
        args.max_train_samples = 500
        args.max_eval_samples = 200
        args.num_epochs = 2
    
    run_experiment(args)
