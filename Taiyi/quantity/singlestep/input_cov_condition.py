import math

from Taiyi.quantity.singlestep.base_class import SingleStepQuantity
from Taiyi.extensions import ForwardInputEigOfCovExtension
from Taiyi.quantity.utils.calculation import *
import torch
import numpy as np
import pdb
# 主要用于计算输入的协方差矩阵的特征值和对应的条件数
class InputCovCondition(SingleStepQuantity):
    def _compute(self, global_step):
        eig_values, step = getattr(self._module, 'eig_values', (None, None))
        if eig_values is None or step is None or step != global_step:
            data = self._module.input_eig_data
            cov = cal_cov_matrix(data) # 计算协方差矩阵
            eig_values = cal_eig(cov) # 计算协方差矩阵的特征值
            eig_values, _ = torch.sort(eig_values, descending=True)
            setattr(self._module, 'eig_values', (eig_values, global_step)) # 最后可以在module的eig_values中获得对应的值
        eps = 1e-7
        condition = eig_values[0] / (torch.abs(eig_values[-1]) + eps) # 即特征值的最大值与绝对值最小值之比,计算条件数
        # print(eig_values)
        return condition

    def forward_extensions(self):
        extensions = [ForwardInputEigOfCovExtension()]
        return extensions