import torch
import numpy as np

a = torch.rand(3, 4)
b = torch.rand(4)
print("输出a: ")
print(a)
print("输出b: ")
print(b)
print("执行a+b操作默认会进行broadcasting： ")
print(a + b)

print(torch.add(a, b))

print("\n判断每一位是否相等: ")
print(torch.eq(a-b, torch.sub(a, b)))
print("判断乘全部是否相等: ")
print(torch.all(torch.eq(a*b, torch.mul(a, b))))
print("判断除是否全部相等: ")
print(torch.all(torch.eq(a/b, torch.div(a, b))))
