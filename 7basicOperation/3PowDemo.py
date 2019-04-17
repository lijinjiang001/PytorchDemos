import torch
import numpy as np

print("输出a的平方:")
a = torch.full([2, 2], 3)
print(a.pow(2))
print(a**2)

print("开方:")
aa = a.pow(2)
print(aa.sqrt())

print("\n开方再取倒数:")
print(aa.rsqrt())

print("\ne的指数幂:")
c = torch.exp(torch.ones(2, 2))
print(c)

print("\n对数:")
print(torch.log(c))