import torch
import numpy as np

print("split是按照长度进行拆分： ")
a = torch.rand(2, 32, 8)
aa, bb = a.split([1, 1], dim=0)
print(aa.shape)
print(bb.shape)

print("在第二维度上进行拆分: ")
cc, dd = a.split([2, 30], dim=1)
print(cc.shape)
print(dd.shape)