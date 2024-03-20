import torch
from torch.optim.optimizer import Optimizer
import math

class Fusedbun(Optimizer):
    def __init__(self, params, lr=1e-3, eps=1e-8, beta_decay=0.8, Lambda=0.01, momentum_beta=0.9, centralize=True, use_rms=True):
        """
        Args:
            params (iterable): An iterable of parameters.
            lr (float, optional): The learning rate (default: 1e-3).
            eps (float, optional): A small value for numerical stability (default: 1e-8).
            beta_decay (float, optional): A parameter controlling the rate of decay for the moving average of squared gradients (default: 0.8).
            Lambda (float, optional): default: 0.01
            momentum_beta (float, optional): Coefficient for the moving average of gradients (momentum) (default: 0.9).
            centralize (bool, optional): Boolean indicating whether to centralize gradients to have zero mean (default: True).
            use_rms (bool, optional): Boolean indicating whether to use RMSprop-like denominator normalization (default: True).
        """
        defaults = dict(lr=lr, eps=eps, beta_decay=beta_decay, Lambda=Lambda, momentum_beta=momentum_beta, centralize=centralize, use_rms=use_rms)
        super(Fusedbun, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        Returns:
            loss (Tensor, optional): The loss tensor, if the closure is provided.
        """
        loss = None
        if closure is not None:
            # compute the loss if a closure is provided
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            beta_decay = group['beta_decay']
            Lambda = group['Lambda']
            momentum_beta = group['momentum_beta']
            centralize = group['centralize']
            use_rms = group['use_rms']
            
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if centralize and len(p.shape) > 1:
                    # centralize gradients for non-scalar parameters (from adalite)
                    grad = grad.sub(grad.mean(dim=tuple(range(1, len(grad.shape))), keepdim=True))
                    
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if momentum_beta > 0:
                        state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1

                if momentum_beta > 0:
                    momentum_buffer = state['momentum_buffer']
                    momentum_buffer.mul_(momentum_beta).add_(grad)
                    grad = momentum_buffer

                #sparse update mechanism
                if p.dim() > 1:
                    mask = grad.abs() > eps
                    grad = grad * mask
                    exp_avg_sq = torch.where(mask, exp_avg_sq*beta_decay + (1-beta_decay)*grad.pow(2), exp_avg_sq)
                else:
                    exp_avg_sq.mul_(beta_decay).addcmul_(grad, grad, value=1-beta_decay)

                denom = exp_avg_sq.sqrt().add_(eps)
                grad_normalized = grad / denom if use_rms else grad

                if Lambda != 0:
                    p.data.mul_(1 - lr * Lambda)

                p.data.add_(grad_normalized, alpha=-lr)
        return loss
