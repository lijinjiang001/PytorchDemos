import torch
import numpy as np

a = torch.tensor(3.14)
print(a)
print("向下取整:")
print(a.floor())

print("\n向上取整:")
print(a.ceil())

print("\n取整数部分:")
print(a.trunc())

print("\n取小数部分:")
print(a.frac())

print("\n四舍五入:")
print(a.round())
