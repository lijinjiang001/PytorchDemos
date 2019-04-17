import torch
import numpy as np

print("\n生成一个向量：")
a_vec = np.array([2, 2,3])
print(a_vec)
print(torch.from_numpy(a_vec))

print("\n生成全１向量:")
one_vec = np.ones(2)
print(one_vec)

print("\n生成全１矩阵:")
one_matrix = np.ones([2, 3])
print(torch.from_numpy(one_matrix))

print("\n生成tensor的两种方法:")
a_tensor = torch.tensor([2, 2,3])
print(a_tensor)

print("\n" + "="*20)
print(torch.Tensor([2, 3.5]))
print("\n随机生成一个2*3的矩阵：")
print(torch.Tensor(2, 3))

print(torch.FloatTensor([3., 4.2]))
print(torch.IntTensor(3, 5))

print("\n生成随机初始化的矩阵:")
a = torch.empty(2)
print(a)
print(a.shape)

print("\n对tensor的类型进行改变:")
data = torch.FloatTensor(2, 3)
print(data)
torch.set_default_tensor_type(torch.DoubleTensor)
print(torch.Tensor([2.1, 2]).type())
torch.set_default_tensor_type(torch.FloatTensor)

print("\n生成指定初始值的矩阵:")
result = torch.full([2, 3], 7)
print(result)

print("\n生成一个初始值为5的标量:")
print(torch.full([], 5))

print("\n生成一个初始值为8的向量:")
print(torch.full([1], 8))

print("\n生成一个递增数列, 注意：左闭右开:")
print(torch.arange(0, 10))

print("\n在指定区间线性生成step个数:")
print(torch.linspace(2, 20, 5))

print("\n" + "指定递增数列的步长:")
print(torch.arange(1, 10, 2))

print("\n生成10的指数幂:")
print(torch.logspace(0, -1, 5))

print("\n生成对角阵:")
print(torch.eye(3))
print(torch.eye(3, 3))
print(torch.eye(3, 4))

print("\n随机打散:")
print(torch.randperm(10))

print("\n将两个矩阵进行相同的行变换:")
a_data = torch.rand(2, 3)
print(a_data)
b_data = torch.rand(2, 2)
print(b_data)
idx = torch.randperm(2)
print(idx)
print(a_data[idx])
print(b_data[idx])