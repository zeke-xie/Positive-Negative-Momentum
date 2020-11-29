# Positive-Negative-Momentum
Positive-Negative Momentum Optimizers.


# Usage

#You may use it as a standard PyTorch optimizer.

```python
from pnm_optim import *

PNM_optimizer = PNM(net.parameters(), lr=lr, betas=(0.9, 1.), weight_decay=weight_decay)
AdaPNM_optimizer = AdaPNM(net.parameters(), lr=lr, betas=(0.9, 0.999, 1.), eps=1e-08, weight_decay=weight_decay, amsgrad=True)
```
