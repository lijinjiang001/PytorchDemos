import torch
import numpy as np

# 二维数组
a_matrix = torch.randn(2, 3)
print(a_matrix)
print(a_matrix.size())
print(a_matrix.shape)
print(a_matrix.size(0))
print(a_matrix.size(1))
print(a_matrix.shape[1])

# 三维数组
print("\n" + "="*100)
b_matrix = torch.rand(1, 2, 3)
print(b_matrix)
print(b_matrix.size())
print(list(b_matrix.shape))

# 四维数组
print("\n" + "="*100)
c_matrix = torch.randn(2, 3, 28, 28)
print(c_matrix.size())
print(c_matrix.size(1))
print(c_matrix.shape[3])
# 输出2*3*28*28
print(c_matrix.numel())
print(2*3*28*28)
print(c_matrix.dim())

print("\n" + "="*100)
d_matrix = torch.randn(2, 3)
print(d_matrix)
e_matrix = torch.rand_like(d_matrix)
print(e_matrix)
# 指定随机生成数组的范围，注意是[1, 10)
f_matrix = torch.randint(1, 10, [3, 2])
print(f_matrix)