import torch
import numpy as np

a = torch.full([2, 2], 3)
b = torch.ones(2, 2)

print("torch.mm只适用与二维矩阵相乘:")
print(torch.mm(a, b))

print("\n多维矩阵相乘:")
print(torch.matmul(a, b))
print(a @ b)

print("多维矩阵相乘，只是进行后两维运算:")
aa = torch.rand(3, 4, 28, 32)
bb = torch.rand(3, 4, 32, 64)
print(torch.matmul(aa, bb).shape)