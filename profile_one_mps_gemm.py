from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def sync() -> None:
    torch.mps.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--bench-iters", type=int, default=100)
    parser.add_argument("--capture-repeats", type=int, default=5)
    parser.add_argument("--open-xcode", action="store_true")
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")

    if not torch.mps.profiler.is_metal_capture_enabled():
        raise RuntimeError(
            "Metal capture is disabled. "
            "Run with MTL_CAPTURE_ENABLED=1."
        )

    torch.set_grad_enabled(False)

    device = torch.device("mps")
    dtype = torch.bfloat16

    # 与 nn.Linear 一致：
    # input [1,M,K], weight [N,K]
    # 实际 GEMM：[M,K] × [K,N]
    x = torch.randn(
        (1, args.m, args.k),
        device=device,
        dtype=torch.float32,
    ).to(dtype)

    weight = torch.randn(
        (args.n, args.k),
        device=device,
        dtype=torch.float32,
    ).to(dtype)

    sync()

    print("=" * 80)
    print("label:", args.label)
    print(
        f"GEMM [{args.m},{args.k}] "
        f"x [{args.k},{args.n}] "
        f"-> [{args.m},{args.n}]"
    )
    print("input:", tuple(x.shape))
    print("weight:", tuple(weight.shape))
    print("dtype:", dtype)
    print("=" * 80, flush=True)

    # 建图、编译和缓存不进入测量。
    for _ in range(args.warmups):
        y = F.linear(x, weight)

    sync()

    start = time.perf_counter()

    for _ in range(args.bench_iters):
        y = F.linear(x, weight)

    sync()
    elapsed = time.perf_counter() - start

    mean_s = elapsed / args.bench_iters
    flops = 2.0 * args.m * args.k * args.n
    effective_tflops = flops / mean_s / 1e12

    print(
        f"BENCH mean_ms={mean_s * 1000:.6f} "
        f"effective_TFLOPS={effective_tflops:.3f}",
        flush=True,
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    capture_stem = (
        f"{args.label}_m{args.m}_k{args.k}_n{args.n}_"
        f"{timestamp}"
    )

    print(
        f"CAPTURE_BEGIN repeats={args.capture_repeats}",
        flush=True,
    )

    sync()

    with torch.mps.profiler.metal_capture(capture_stem):
        for _ in range(args.capture_repeats):
            y = F.linear(x, weight)

        sync()

    print("CAPTURE_END", flush=True)

    matches = sorted(
        Path(".").glob(f"*-{capture_stem}.gputrace")
    )

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one gputrace, found: {matches}"
        )

    trace_dir = Path("./traces")
    trace_dir.mkdir(parents=True, exist_ok=True)

    destination = trace_dir / matches[0].name
    matches[0].replace(destination)
    destination = destination.resolve()

    print(f"GPU_TRACE_SAVED {destination}", flush=True)

    subprocess.run(
        ["du", "-sh", str(destination)],
        check=False,
    )

    if args.open_xcode:
        print("OPENING_XCODE", flush=True)
        subprocess.run(
            ["open", "-a", "Xcode", str(destination)],
            check=True,
        )


if __name__ == "__main__":
    main()
