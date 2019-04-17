import torch
import numpy as np

print("若两个矩阵的shape不同，则会默认将较小的维度broadcasting到较大的维度： ")
x = torch.empty(5, 3, 4, 1)
y = torch.empty(3, 4, 1)
print((x+y).shape)

print("\n注意能够broadcasting的前提是其余维度相同, 只能在最前面补维度：")
a = torch.zeros(5, 2, 4, 1)
b = torch.zeros(5, 2, 4)
# print((a+b).size())　　# 会出错

print("\n可以某一维度为１，则会自动扩展到相同维度")
c = torch.zeros(2, 1, 1)
print((a+c).size())