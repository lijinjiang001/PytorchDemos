import torch
import numpy as np

print("b的shape: ")
b = torch.randn(1, 32, 1, 1)
print(b.size())

print("\n使用repeat将会复制多遍： \n若某一维度保持不变,　则填入１")
print(b.repeat(4, 32, 14, 14).shape)
print(b.repeat(4, 1, 14, 14).size())