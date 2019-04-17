import torch
import torch.nn.functional as F

x = torch.randn(1, 10)
w = torch.randn(2, 10, requires_grad = True)

output = torch.sigmoid(x@w.t())
print(output)
print(output.size())

loss = F.mse_loss(torch.ones(1, 2), output)
print()
print(loss)

loss.backward()

gradsResult = w.grad
print()
print(gradsResult.shape)
print(gradsResult)