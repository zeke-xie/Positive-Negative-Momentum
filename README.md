# Positive-Negative-Momentum
The official PyTorch Implementations of Positive-Negative Momentum Optimizers.

The algortihms are proposed in our paper: 
[Positive-Negative Momentum: Manipulating Stochastic Gradient Noise to Improve Generalization](https://arxiv.org/abs/2103.17182), which is accepted by ICML 2021. We fixed several notation typos in the updated arxiv version.


# Why Positive-Negative Momentum?

It is well-known that stochastic gradient noise matters a lot to generalization. The Positive-Negative Momentum (PNM) approach, which is a powerful alternative to conventional Momentum in classic optimizers, can manipulate stochastic gradient noise by adjusting the extrahyperparameter.


# The environment is as bellow:

Python 3.7.3 

PyTorch >= 1.4.0


# Usage

#You may use it as a standard PyTorch optimizer.

```python
from pnm_optim import *

PNM_optimizer = PNM(net.parameters(), lr=lr, betas=(0.9, 1.), weight_decay=weight_decay)
AdaPNM_optimizer = AdaPNM(net.parameters(), lr=lr, betas=(0.9, 0.999, 1.), eps=1e-08, weight_decay=weight_decay)
```


# Test performance

PNM versus conventional Momentum. We report the mean and the standard deviations (as the subscripts) of the optimal test errors computed over three runs of each experiment. The proposed PNM-based methods show significantly better generalization than conventional momentum-based methods. Particularly, as the theoretical analysis indicates, Stochastic PNM indeed consistently outperforms the conventional baseline, SGD.

| Dataset   | Model       | PNM                    | AdaPNM                     | SGD M                | Adam                 | AMSGrad              | AdamW                | AdaBound             | Padam                | Yogi                 | RAdam                |
|:----------|:------------|:-------------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| CIFAR-10  | ResNet18    | **4.48**<sub>0.09</sub>  | 4.94<sub>0.05</sub>  | 5.01<sub>0.03</sub>  | 6.53<sub>0.03</sub>  | 6.16<sub>0.18</sub>  | 5.08<sub>0.07</sub>  | 5.65<sub>0.08</sub>  | 5.12<sub>0.04</sub>  | 5.87<sub>0.12</sub>  | 6.01<sub>0.10</sub>  |
|           | VGG16       | 6.26<sub>0.05</sub>  | **5.99**<sub>0.11</sub>  | 6.42<sub>0.02</sub>  | 7.31<sub>0.25</sub>  | 7.14<sub>0.14</sub>  | 6.48<sub>0.13</sub>  | 6.76<sub>0.12</sub>  | 6.15<sub>0.06</sub>  | 6.90<sub>0.22</sub>  | 6.56<sub>0.04</sub>  |
| CIFAR-100 | ResNet34    | 20.59<sub>0.29</sub> | **20.41**<sub>0.18</sub> | 21.52<sub>0.37</sub> | 27.16<sub>0.55</sub> | 25.53<sub>0.19</sub> | 22.99<sub>0.40</sub> | 22.87<sub>0.13</sub> | 22.72<sub>0.10</sub> | 23.57<sub>0.12</sub> | 24.41<sub>0.40</sub> |
|           | DenseNet121 | **19.76**<sub>0.28</sub> | 20.68<sub>0.11</sub> | 19.81<sub>0.33</sub> | 25.11<sub>0.15</sub> | 24.43<sub>0.09</sub> | 21.55<sub>0.14</sub> | 22.69<sub>0.15</sub> | 21.10<sub>0.23</sub> | 22.15<sub>0.36</sub> | 22.27<sub>0.22</sub> |
|           | GoogLeNet   | 20.38<sub>0.31</sub> | **20.26**<sub>0.21</sub> | 21.21<sub>0.29</sub> | 26.12<sub>0.33</sub> | 25.53<sub>0.17</sub> | 21.29<sub>0.17</sub> | 23.18<sub>0.31</sub> | 21.82<sub>0.17</sub> | 24.24<sub>0.16</sub> | 22.23<sub>0.15</sub> |

# Citing

If you use Positive-Negative Momentum in your work, please cite

```
@InProceedings{xie2021positive,
  title = 	 {Positive-Negative Momentum: Manipulating Stochastic Gradient Noise to Improve Generalization},
  author =       {Xie, Zeke and Yuan, Li and Zhu, Zhanxing and Sugiyama, Masashi},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {11448--11458},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
}
```
