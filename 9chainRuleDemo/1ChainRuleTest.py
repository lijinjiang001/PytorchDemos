import torch
from torch import autograd

x = torch.tensor(1.)
w1 = torch.tensor(2., requires_grad=True)
b1 = torch.tensor(1.)
w2 = torch.tensor(2., requires_grad=True)
b2 = torch.tensor(1.)

print("y1=x*w1+b1")
y1 = x * w1 + b1
print(y1.shape)
y2 = y1*w2 + b2

print()
# 计算y2对y1的梯度
dy2_dy1 = autograd.grad(y2, [y1], retain_graph=True)
print(dy2_dy1)
dy2_dy1 = dy2_dy1[0]

# 计算y1对w1的梯度
dy1_dw1 = autograd.grad(y1, [w1], retain_graph=True)[0]

# 计算y2对w1的梯度
dy2_dw1 = autograd.grad(y2, [w1], retain_graph=True)[0]

# 根据链式法则计算出的y2对w1的梯度
print("\n根据链式法则计算出的y2对w1的梯度: ")
print(dy2_dy1 * dy1_dw1)


# 直接使用autograd计算的y2对w1的梯度
print("\n直接使用autograd计算的y2对w1的梯度")
print(dy2_dw1)
