from Taiyi.quantity.singlestep.input_cov_condition20 import InputCovCondition20
from Taiyi.quantity.singlestep.input_cov_condition50 import InputCovCondition50
from Taiyi.quantity.singlestep.input_cov_condition80 import InputCovCondition80
from Taiyi.quantity.singlestep.input_cov_condition import InputCovCondition
from Taiyi.quantity.singlestep.input_cov_max_eig import InputCovMaxEig
from Taiyi.quantity.singlestep.input_cov_stable_rank import InputCovStableRank


import torch
from torch import nn as nn

l = nn.Linear(2, 3)
cov = nn.Conv2d(2, 2, 3, 1, 1)
x = torch.randn((4, 2))
x_c = torch.randn((4, 2, 3, 3))
quantity_l = InputCovCondition20(l)
quantity_c = InputCovCondition(cov)

for hook in quantity_l.forward_extensions():
    l.register_forward_hook(hook)
for hook in quantity_c.forward_extensions():
    cov.register_forward_hook(hook)

for j in range(1):
    y = l(x)
    y_c = cov(x_c)
    quantity_l.track(j)
    quantity_c.track(j)
print(quantity_l.get_output()[0])
print(quantity_c.get_output()[0])