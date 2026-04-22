"""Minimal torch.compile smoke test for local and Slurm runs."""

from __future__ import annotations

import argparse
import shutil
import sys
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal torch.compile smoke test.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Target device. 'auto' uses CUDA when available, otherwise CPU.",
    )
    parser.add_argument(
        "--numel",
        type=int,
        default=1024 * 1024,
        help="Number of elements per input tensor.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64", "float16", "bfloat16"),
        default="float32",
        help="Tensor dtype.",
    )
    parser.add_argument(
        "--backend",
        default="inductor",
        help="torch.compile backend to use.",
    )
    parser.add_argument(
        "--mode",
        default="default",
        help="torch.compile mode to use.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Extra compiled runs before timing steady-state iterations.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=3,
        help="Number of timed steady-state iterations.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available in this environment.")
    return requested


def maybe_sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def summarize_environment(device: str) -> None:
    print("torch version:", torch.__version__)
    print("has torch.compile:", hasattr(torch, "compile"))
    print("cuda available:", torch.cuda.is_available())
    print("torch CUDA version:", torch.version.cuda)
    print("gcc:", shutil.which("gcc"))
    print("g++:", shutil.which("g++"))
    print("ninja:", shutil.which("ninja"))
    try:
        from torch.utils._triton import has_triton

        print("has_triton():", has_triton())
    except Exception as exc:  # pragma: no cover - diagnostics only
        print("has_triton() check failed:", repr(exc))
    if device == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
        print("device count:", torch.cuda.device_count())


def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sin(x) + torch.cos(y) + 0.1 * (x * y)


def main() -> int:
    args = parse_args()
    if not hasattr(torch, "compile"):
        print("torch.compile is not available in this PyTorch build.", file=sys.stderr)
        return 2

    device = resolve_device(args.device)
    dtype = getattr(torch, args.dtype)

    summarize_environment(device)
    print("selected device:", device)
    print("selected dtype:", args.dtype)
    print("numel:", args.numel)
    print("backend:", args.backend)
    print("mode:", args.mode)

    torch.manual_seed(0)
    x = torch.randn(args.numel, device=device, dtype=dtype)
    y = torch.randn(args.numel, device=device, dtype=dtype)

    eager_out = kernel(x, y)
    maybe_sync(device)

    compiled_kernel = torch.compile(kernel, backend=args.backend, mode=args.mode)

    start = time.perf_counter()
    compiled_out = compiled_kernel(x, y)
    maybe_sync(device)
    first_run_s = time.perf_counter() - start

    for _ in range(args.warmup):
        compiled_out = compiled_kernel(x, y)
        maybe_sync(device)

    steady_times = []
    for _ in range(args.iters):
        start = time.perf_counter()
        compiled_out = compiled_kernel(x, y)
        maybe_sync(device)
        steady_times.append(time.perf_counter() - start)

    torch.testing.assert_close(eager_out, compiled_out)

    print("first compiled run (s):", round(first_run_s, 6))
    if steady_times:
        avg_s = sum(steady_times) / len(steady_times)
        print("steady avg (s):", round(avg_s, 6))
    print("torch.compile smoke test passed on", device)
    print("output[:5]:", compiled_out[:5])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
