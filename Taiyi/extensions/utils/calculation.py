import torch
import torch.linalg as linalg # 主要包含了一些线性代数的计算


def cal_cov_matrix(input):
    if input.dim() == 2:
        return torch.cov(input.T) # 计算输入的协方差矩阵

    if input.dim() == 3:
        # 将第0维和第2维交换之后变为连续之后使用view重新改变其大小
        input = input.transpose(0, 2).contiguous().view(input.shape[2], -1)

        return torch.cov(input)


def cal_eig(input):
    eigvals = linalg.eigvals(input) # 用于计算输入的特征值
    return eigvals


if __name__ == '__main__':
    # test cal_cov_matrix
    x = torch.randn((5, 10, 6))
    print(x.T.shape)
    y = cal_cov_matrix(x)
    print(y.shape)

    # test cal_eig
    # x = torch.randn((10, 10))
    # y = cal_eig(x)
    # print(y)
    # print(y.size())