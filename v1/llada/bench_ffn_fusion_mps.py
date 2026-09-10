import torch
import time

device = "mps"

torch.manual_seed(0)

# 根据 LLaDA-8B FFN shape
B = 1
T = 32
H = 4096
FFN = 11008

dtype = torch.bfloat16

x = torch.randn(
    B, T, H,
    device=device,
    dtype=dtype
)

gate = torch.nn.Linear(
    H,
    FFN,
    bias=False,
    device=device,
    dtype=dtype
)

up = torch.nn.Linear(
    H,
    FFN,
    bias=False,
    device=device,
    dtype=dtype
)

fused = torch.nn.Linear(
    H,
    FFN * 2,
    bias=False,
    device=device,
    dtype=dtype
)

with torch.no_grad():
    fused.weight.copy_(
        torch.cat(
            [
                gate.weight,
                up.weight
            ],
            dim=0
        )
    )


def bench(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()

    torch.mps.synchronize()

    start = time.perf_counter()

    for _ in range(iters):
        fn()

    torch.mps.synchronize()

    end = time.perf_counter()

    return (end-start)*1000/iters


def original():
    g = gate(x)
    u = up(x)
    return g, u


def fused_run():
    y = fused(x)
    g, u = y.chunk(2, dim=-1)
    return g, u


t1 = bench(original)
t2 = bench(fused_run)

print("Original gate+up ms:", t1)
print("Fused gate_up ms:", t2)
print("Speedup:", t1/t2)

