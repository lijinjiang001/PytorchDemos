import torch
import torch.nn.functional as F

# 建立一个单层的感知机
x = torch.randn(1, 10)
w = torch.randn(1, 10, requires_grad=True)

# 计算输出
ouput = torch.sigmoid(x@w.t())
print("ouput.shape: ")
print(ouput.shape)

# 计算均方差损失函数
loss = F.mse_loss(torch.ones(1, 1), ouput)
print("\nloss.shape: ")
print(loss.shape)

# 反向传播
loss.backward()

# 输出梯度
print(w.grad)