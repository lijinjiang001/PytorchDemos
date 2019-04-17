import torch
import numpy as np

print("生成b矩阵为1 * 32 * 1 * 1: ")
b = torch.rand(1, 32, 1, 1)
print(b.shape)

print("\n将所有维度为１的都压缩：")
print(b.squeeze().shape)

print("\n压缩0维度：")
print(b.squeeze(0).shape)
print("压缩最后一个维度：")
print(b.squeeze(-1).shape)

print("\n对于非１的维度无法压缩:")
print(b.squeeze(1).shape)