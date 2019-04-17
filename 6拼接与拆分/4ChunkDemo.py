import torch
import numpy as np

print("chunk是按照数量进行拆分： ")
a = torch.rand(2, 32, 8)
aa, bb = a.chunk(2, dim=0)
print(aa.size())
print(bb.size())

print("在第二维度进行拆分：")
cc, dd = a.chunk(2, dim=1)
print(cc.shape)
print(dd.shape)