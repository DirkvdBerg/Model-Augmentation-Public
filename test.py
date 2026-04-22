import torch

assert torch.cuda.is_available(), "PyTorch does not see an NVIDIA GPU"

def f(x, y):
    return torch.sin(x) + torch.cos(y)

g = torch.compile(f)

x = torch.randn(1024, device="cuda")
y = torch.randn(1024, device="cuda")
z = g(x, y)
torch.cuda.synchronize()

print("torch.compile CUDA works")
print(z[:5])
