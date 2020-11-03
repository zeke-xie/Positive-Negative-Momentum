
import torch
from torch.optim.optimizer import Optimizer, required

class SGDPNM(Optimizer):
    r"""Implements Stochastic Gradient Descent with Positive-Negative Momentum (SGDPNM).
    It has be proposed in 
    `A Gradient Noise Amplification Method Improves Learning for Deep Networks`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        betas (Tuple[float, float], optional): inertia coefficients used for computing
            momentum (default: (0.9, -1.))
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=required, betas=(0.9, -1.), weight_decay=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not  0. <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not betas[1] < 0.:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(SGDPNM, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDPNM, self).__setstate__(state)

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
            weight_decay = group['weight_decay']
            

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad

                # Perform decoupled weight decay
                p.mul_(1 - group['lr'] * group['weight_decay'])
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    
                state['step'] += 1
                #bias_correction1 = 1 - beta1 ** ((state['step'] + 1) / 2)

                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach() * (1 - beta1)
                    neg_momentum = param_state['neg_momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                elif state['step'] % 2 == 1:
                    buf = param_state['momentum_buffer']
                    neg_momentum = param_state['neg_momentum_buffer']
                    buf.mul_(beta1).add_(d_p, alpha=1-beta1)
                    neg_momentum.mul_(beta1)
                else:
                    neg_momentum = param_state['momentum_buffer']
                    buf = param_state['neg_momentum_buffer']
                    buf.mul_(beta1).add_(d_p, alpha=1-beta1)
                    neg_momentum.mul_(beta1)

                delta_p = buf.mul(1-beta2/2.).add(neg_momentum, alpha=beta2/2.)
                p.add_(delta_p, alpha=-group['lr'])
        return loss