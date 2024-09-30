import torch
import numpy as np
class InputCovarianceEigenvaluesHook:
    def __init__(self):
        self.eigenvalues = []

    def hook(self, module, input, output):
        # 提取输入数据
        input_data = input[0]

        # 计算输入数据的协方差矩阵
        batch_size = input_data.size(0)
        input_data_flattened = input_data.view(batch_size, -1)
        covariance_matrix = torch.cov(input_data_flattened.T)

        # 提取协方差矩阵的特征值
        eigenvalues = np.linalg.eigvalsh(covariance_matrix)

        # 保存特征值
        self.eigenvalues.append(eigenvalues)

# 创建模型
model = torch.nn.Linear(10, 5)

# 创建输入协方差矩阵特征值钩子
hook = InputCovarianceEigenvaluesHook()

# 注册前向传播钩子
handle = model.register_forward_hook(hook.hook)

# 准备输入数据
input_tensor = torch.randn(2, 10)

# 执行前向传播
output_tensor = model(input_tensor)

# 打印特征值
print("Eigenvalues of input covariance matrix:")
for eig in hook.eigenvalues:
    print(eig)

# 移除钩子
handle.remove()
