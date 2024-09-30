from Taiyi.quantity.singlestep.input_mean import InputMean
from Taiyi.quantity.singlestep.input_std import InputStd
from Taiyi.quantity.singlestep.input_norm import InputSndNorm

import torch
from torch import nn as nn

l = nn.Linear(2, 3)
cov = nn.Conv2d(1, 2, 3, 1, 1)
x = torch.randn((4, 2))
print(len(x.shape))
x_c = torch.randn((4, 1, 3, 3))
quantity_l = InputStd(l)
quantity_c = InputSndNorm(cov)
for hook in quantity_l.forward_extensions():
    l.register_forward_hook(hook)

for i in range(3):
    y = l(x)
    quantity_l.track(i)

print(quantity_l.get_output()[0])
print(x.shape)
print(x.mean())
print(quantity_c.get_output()[0])
print(x_c.mean())