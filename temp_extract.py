import torch

# Test 1: does matrix_exp support backward through A?
A = torch.randn(3, 3, requires_grad=True)
expA = torch.linalg.matrix_exp(A)
loss = expA.sum()
loss.backward()
print("Test 1 - grad through A:", A.grad is not None)

# Test 2: does matrix_exp support backward through a scalar parameter that enters A?
y = torch.tensor(0.3, requires_grad=True)
A2 = torch.stack([
    torch.stack([torch.zeros(1).squeeze(), torch.ones(1).squeeze()]),
    torch.stack([-y, torch.tensor(-1.0)])
])
ts = 62.5e-6
Ad = torch.linalg.matrix_exp(A2 * ts)
loss2 = Ad.sum()
loss2.backward()
print("Test 2 - grad through scalar y:", y.grad is not None, "| y.grad =", y.grad.item())
