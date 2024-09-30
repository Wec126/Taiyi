from Taiyi.quantity.singlestep.rankme import RankMe

import torch
from torch import nn as nn

l = nn.Linear(2, 3).cuda()
cov = nn.Conv2d(2, 2, 3, 1, 1).cuda()
x = torch.randn((4, 8, 2)).cuda()
x_c = torch.randn((4, 2, 3, 3)).cuda()
quantity_l = RankMe(l)
quantity_c = RankMe(cov)

for hook in quantity_l.forward_extensions():
    l.register_forward_hook(hook)
for hook in quantity_c.forward_extensions():
    cov.register_forward_hook(hook)

for j in range(6):
    y = l(x)
    y_c = cov(x_c)
    quantity_l.track(j)
    quantity_c.track(j)
print(quantity_l.get_output()[0])
print(quantity_c.get_output()[0])