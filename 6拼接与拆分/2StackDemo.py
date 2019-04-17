import torch
import numpy as np

print("使用torch.stack要求两个tensor的shape必须相同： ")
a = torch.rand(4, 32, 8)
b = torch.rand(4, 32, 8)
print(torch.stack([a, b], dim=1).shape)
print(torch.stack([a, b], dim=0).size())

print("\n取出指定的数据： ")
c = torch.stack([a, b], dim=1)
print(c[:, 0, :, :].shape)
print(c[:, 0, ...].shape)
