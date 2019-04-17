import torch
import numpy as np

grad = torch.rand(2, 3)*15
print(grad)

print("查看最大值:")
print(grad.max())
print("查看最小值:")
print(grad.min())
print("查看均值:")
print(grad.median())

print("比10小的均置为10:")
print(grad.clamp(10))

print("设置值在5和10之间:")
print(grad.clamp(5, 10))