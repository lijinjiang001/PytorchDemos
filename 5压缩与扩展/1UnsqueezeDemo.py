import torch
import numpy as np

print("生成一个大小为４＊１＊２８＊２８的矩阵a: ")
a = torch.rand(4, 1, 28, 28)
print(a.shape)

print("\n将a在维度0处添加一个维度：")
print(a.unsqueeze(0).shape)
print("注意a的shape并没有改变")
print(a.shape)

print("\n在ａ的最后一个维度处添加一个维度：")
print(a.unsqueeze(-1).shape)

print("注意在.unsqueeze()括号中正负的不同之处：\n整数表示在当前要插入维度的前面插入，负数则表示在当前插入维度的后面插入")
print(a.size())
print(a.unsqueeze(2).shape)
print(a.unsqueeze(-2).shape)

print("\n向量的维度扩展：")
b = torch.tensor([1.2, 2.4])
print(b)
print(b.size())
print("在列维度上扩展：")
c = b.unsqueeze(1)
print(c)
print(c.size())
print("在行维度上扩展：")
d = b.unsqueeze(0)
print(d)
print(d.size())