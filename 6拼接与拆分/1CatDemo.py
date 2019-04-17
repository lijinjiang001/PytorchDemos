import torch
import numpy as np

a = torch.rand(4, 32, 8)
b = torch.rand(5, 32, 8)
print("按照指定维度拼接： ")
print(torch.cat([a, b], dim=0).shape)

c = torch.rand(4, 28, 8)
print(torch.cat([a, c], dim=1).shape)