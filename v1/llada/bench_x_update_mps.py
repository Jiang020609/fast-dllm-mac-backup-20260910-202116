import torch
import time

device = "mps"

B = 1
L = 1331
block = 32

x0 = torch.randint(
    0,
    100,
    (B, L),
    device=device,
    dtype=torch.long,
)

new_block = torch.randint(
    0,
    100,
    (B, block),
    device=device,
    dtype=torch.long,
)


def bench(fn, warmup=50, iters=500):
    for _ in range(warmup):
        fn()

    torch.mps.synchronize()

    t0 = time.perf_counter()

    for _ in range(iters):
        fn()

    torch.mps.synchronize()

    return (time.perf_counter() - t0) * 1000 / iters


def cat_update():
    x = torch.cat(
        (
            x0[:, :100],
            new_block,
            x0[:, 132:],
        ),
        dim=1,
    )
    return x


def inplace_update():
    x = x0.clone()
    x[:, 100:132] = new_block
    return x


t1 = bench(cat_update)
t2 = bench(inplace_update)

print("torch.cat ms:", t1)
print("inplace update ms:", t2)
print("speedup:", t1/t2)
