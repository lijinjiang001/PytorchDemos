import torch
import numpy as np

a = torch.randn(2, 3)
print("a = torch.randn(2, 3): ")
print(a)

print("\n" + "="*100)
print("a.type(): " + a.type())

print("\n" + "="*100)
print("type(a): ")
print(type(a))

# 判断是否为torch.FloatTEnsor的实例
print("\n" + "="*100)
print("isinstance(a, torch.FloatTensor): ")
print(isinstance(a, torch.FloatTensor))
print("isinstance(a, torch.DoubleTensor):")
print(isinstance(a, torch.DoubleTensor))


# 同一个tensor在cpu和gpu中是不同的
print("\n" + "="*100)
print("isinstance(a, torch.cuda.FloatTensor): ")
print(isinstance(a, torch.cuda.FloatTensor))
data = a.cuda()
print("isinstance(data, torch.cuda.FloatTensor): ")
print(isinstance(data, torch.cuda.FloatTensor))

#　使用pytorch生成标量
print("\n" + "="*100)
a = torch.tensor(1.)
print("a: ")
print(a)
print(a.type())
print(a.shape)
b = torch.tensor(1)
print("b: ")
print(b)
print(b.type())
print(b.size())

#　生成向量
print("\n" + "="*100)
a_vec = torch.tensor([1.1])
print(a_vec)
print(a_vec.size)
print(a.shape)
# 生成向量的另一种方式
b_vec = torch.FloatTensor(1)
print(b_vec)
print(torch.FloatTensor(2))
#　生成一个值为１，长度指定的向量
print("torch.ones(5): ")
print(torch.ones(5))
data = np.ones(5)
print("np.ones(5): ")
print(data)
#　将numpy转换成tensor
data1 = torch.from_numpy(data)
print(data1)
