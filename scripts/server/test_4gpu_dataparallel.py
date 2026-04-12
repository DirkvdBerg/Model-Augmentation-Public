"""
Smoke test for single-node multi-GPU execution on the server.

Checks:
1. torch / CUDA are available in the activated environment
2. at least 4 GPUs are visible to the job
3. torch.nn.DataParallel can split one batch across 4 GPUs
4. forward + backward complete successfully

Run this through Slurm with the matching job script in scripts/server/.
"""

from __future__ import annotations

import os
import socket

import torch
import torch.nn as nn


class EchoNet(nn.Module):
    """Tiny model that prints which replica/device processed each chunk."""

    def __init__(self, width: int = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )

    def forward(self, x):
        print(
            f"replica_device={x.device} chunk_shape={tuple(x.shape)}",
            flush=True,
        )
        return self.net(x)


def main():
    print(f"hostname={socket.gethostname()}", flush=True)
    print(f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    print(f"torch_cuda={torch.version.cuda}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"device_count={torch.cuda.device_count()}", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this job.")

    if torch.cuda.device_count() < 4:
        raise RuntimeError(
            f"Expected at least 4 visible GPUs, got {torch.cuda.device_count()}."
        )

    for i in range(torch.cuda.device_count()):
        print(f"gpu_{i}={torch.cuda.get_device_name(i)}", flush=True)

    device_ids = [0, 1, 2, 3]
    model = EchoNet().cuda(device_ids[0])
    model = nn.DataParallel(model, device_ids=device_ids)

    # Batch dimension > number of GPUs so DataParallel actually scatters work.
    x = torch.randn(64, 4096, device=f"cuda:{device_ids[0]}")
    target = torch.zeros_like(x)

    out = model(x)
    loss = (out - target).pow(2).mean()
    print(f"loss={loss.item():.6e}", flush=True)
    loss.backward()
    print("backward_ok=True", flush=True)
    print("test_status=PASS", flush=True)


if __name__ == "__main__":
    main()
