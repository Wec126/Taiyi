from Taiyi.quantity.utils.calculation import *

x = torch.randn((5, 10, 6))
y = x.transpose(0, 2).contiguous().view(x.shape[2], -1)
# y = cal_cov_matrix(x)
print(y.shape)
# z = x.transpose(0, 2).contiguous().view(x.shape[2], -1)
# u = z.T - torch.mean(z.T, axis=0)
# U, S, V = torch.svd(u)
# print(S.sort()[0]**2)
# # test cal_eig
# # x = torch.randn((10, 10))
# y = cal_eig(y)
# print(y)
# # print(y.size())
print(cal_eig_not_sym(y))