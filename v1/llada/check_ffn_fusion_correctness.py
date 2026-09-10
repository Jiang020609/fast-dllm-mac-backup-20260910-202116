import torch

torch.manual_seed(0)

device="mps"
dtype=torch.bfloat16

B,T,H,FFN=1,32,4096,11008

x=torch.randn(
    B,T,H,
    device=device,
    dtype=dtype
)

gate=torch.nn.Linear(
    H,FFN,
    bias=False,
    device=device,
    dtype=dtype
)

up=torch.nn.Linear(
    H,FFN,
    bias=False,
    device=device,
    dtype=dtype
)

fused=torch.nn.Linear(
    H,FFN*2,
    bias=False,
    device=device,
    dtype=dtype
)

with torch.no_grad():
    fused.weight.copy_(
        torch.cat(
            [gate.weight,up.weight],
            dim=0
        )
    )


with torch.no_grad():

    g1=gate(x)
    u1=up(x)

    y=fused(x)
    g2,u2=y.chunk(2,dim=-1)


print(
    "gate max error:",
    (g1-g2).abs().max().item()
)

print(
    "up max error:",
    (u1-u2).abs().max().item()
)
