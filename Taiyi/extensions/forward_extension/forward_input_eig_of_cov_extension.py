from Taiyi.extensions.extension import Extension
# from Tiayiutils.calculation import cal_cov_matrix, cal_eig
import numpy as np

class ForwardInputEigOfCovExtension(Extension):
    """
    获取当前module的输入的协方差矩阵的特征值，并将结果保存到module.input_eig字段中
    """
    """
    warning:
        当输入是(1,10[*])的时候自由度小于0会报错
    """

    def __init__(self):
        super(ForwardInputEigOfCovExtension, self).__init__()
        self._name = 'input_eig_data' # 最后获得的结果将保存在'input_eig_data'下面

    def _default(self, module, input, output):
        data = input[0]
        return data

    def _Linear(self, module, input, output):
        data = input[0]
        return data

    def _Conv2d(self, module, input, output):
        data = input[0]
        b, c, w, h = data.shape
        assert (c > 1), "channel must > 1"
        #将输入变为 c* (b*w*h)的形状
        data = data.transpose(0, 1).contiguous().view(-1, c)
        return data
if __name__ == '__main__':

    from torch.nn import Conv2d
    import torch

    l = Conv2d(in_channels=2, out_channels=3, stride=1, padding=2, kernel_size=3)
    forward_input_extension = ForwardInputEigOfCovExtension()
    x = torch.randn((2, 2,  3, 3))
    l.register_forward_hook(forward_input_extension)
    y = l(x)

