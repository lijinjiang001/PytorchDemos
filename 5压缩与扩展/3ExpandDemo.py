import torch
import numpy as np

print("分别生成a矩阵4*32*14*14, b矩阵1*32*1*1: ")
a = torch.rand(4, 32, 14, 14)
b = torch.rand(1, 32, 1, 1)
print(a.shape)
print(b.shape)

print("将b扩展成与a一样的shape: \n注意原始b的shape并不改变")
print(b.expand(4, 32, 14, 14).shape)
print(b.shape)

print("\n注意：只能从１扩展到m, -1表示当前的维度保持不变：　")
print(b.expand(-1, -1, 14, 14).size())