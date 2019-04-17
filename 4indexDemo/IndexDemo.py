import torch
import numpy as np

print("假设有4张图片,每张图片有3个通道,大小为28*28")
a_data = torch.rand(4, 3, 28, 28)
print("取第一张图片：")
print(a_data[0].shape)
print("取第一张图片的第一个通道：")
print(a_data[0, 0].size())
print("取第一张图片的第一个通道中的第二行第四列的值：")
print(a_data[0, 0, 2, 4])

print("\n取前两张图片,每张图片取两个通道：")
print(a_data[:2, :2, :, :].shape)

print("\n取前两张图片,每张图片取第二个通道后的所有：")
print(a_data[:2, 1:, :, :].size())

print("\n取所有图片,每张图片取最后一个通道:")
print(a_data[:, -1, :, :].shape)

print("\n取所有图片的所有通道，其余值值按步长为2进行取值：")
print(a_data[:, :, 0:28:2, 0:28:2].size())
print(a_data[:, :, ::2, ::2].shape)


print("\n取第一张和第三张图片:")
index_tensor = torch.LongTensor([0, 2])
print(index_tensor)
print(torch.index_select(a_data, 0, index_tensor).shape)

index_tensor1 = torch.LongTensor([1, 2])
print(torch.index_select(a_data, 1, index_tensor1).shape)

print("\n用'...'表示所有:")
print(a_data[...].size())

print("\n取第１张图片的所有:")
print(a_data[0, ...].shape)

print("\n取所有图片的第一通道:")
print(a_data[:, 1, ...].size())

print("\n取所有的b,c,h的前两个w：")
print(a_data[..., :2].size())


print("\n从矩阵中选出大于0.5的位置：")
x = torch.randn(3, 4)
print(x)
mask = x.ge(0.5)
print(mask)
print("将大于0.5的值输出,注意输出的值被打平成了向量:")
print(torch.masked_select(x, mask))
print(torch.masked_select(x, mask).shape)

print("\n将矩阵打平并取出指定位置的元素:")
a_matrix = torch.tensor([[2, 3, 4],
                         [5, 6, 7]])
print(a_matrix)
print(torch.take(a_matrix, torch.tensor([0, 2, 5])))
