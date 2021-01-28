
import math
import torch
from torch.optim.optimizer import Optimizer, required

class PNM(Optimizer):
    r"""Implements Positive-Negative Momentum (PNM).
    It has be proposed in 
    `Positive-Negative Momentum: Manipulating Stochastic Gradient Noise to Improve 
    Generalization`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        betas (Tuple[float, float], optional): inertia coefficients used for computing
            pn momentum(default: (0.9, 1.))
        weight_decay (float, optional): weight decay (default: 0)
        decoupled (bool, optional): decoupled weight decay or L2 regularization (default: True)
    """

    def __init__(self, params, lr=required, betas=(0.9, 1.), weight_decay=0, decoupled=True):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not  0. <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, decoupled=decoupled)
        super(PNM, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PNM, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('decoupled', True)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad

                # Perform decoupled weight decay or L2 Regularization
                if group['decoupled']:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                else:
                    d_p.add_(p.data, alpha=group['weight_decay'])

                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    
                state['step'] += 1

                param_state = self.state[p]
                if state['step'] == 1:
                    pos_momentum = param_state['pos_momentum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    neg_momentum = param_state['neg_momentum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                elif state['step'] % 2 == 1:
                    pos_momentum = param_state['pos_momentum']
                    neg_momentum = param_state['neg_momentum']
                else:
                    neg_momentum = param_state['pos_momentum']
                    pos_momentum = param_state['neg_momentum']
                    
                pos_momentum.mul_(beta1**2).add_(d_p, alpha=1-beta1**2)
                noise_norm = math.sqrt((1+beta2) ** 2 + beta2 ** 2)
                delta_p = pos_momentum.mul(1+beta2).add(neg_momentum, alpha=-beta2).mul(1/noise_norm)
                
                p.add_(delta_p, alpha=-group['lr'])
        return loss
