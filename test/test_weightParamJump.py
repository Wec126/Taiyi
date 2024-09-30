from Taiyi.quantity.multistep.weight_param_jump import WeightParamJump


import torch
from torch import nn as nn

l = nn.LayerNorm(3)
x = torch.randn((4, 9, 3))
quantity_l = WeightParamJump(l)

for hook in quantity_l.forward_extensions():
    l.register_forward_hook(hook)

for j in range(1):
    y = l(x)
    quantity_l.track(j)
print(quantity_l.get_output()[0])