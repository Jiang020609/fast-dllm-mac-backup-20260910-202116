import torch
from transformers import AutoConfig
from model.modeling_llada import LLaDAModelLM

torch.manual_seed(0)

model_path = "/Users/z33/models/LLaDA-8B-Instruct"

config = AutoConfig.from_pretrained(model_path)

model = LLaDAModelLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    config=config,
)

model = model.to("mps").eval()

block = model.model.transformer.blocks[0]

x = torch.randn(
    1,
    32,
    config.hidden_size,
    device="mps",
    dtype=torch.bfloat16,
)

with torch.no_grad():
    old_ff = block.ff_proj(x)
    old_up = block.up_proj(x)

    new_ff, new_up = block.ff_proj(x), block.up_proj(x)

    if block._ff_up_weight_cache is None:
        block._ff_up_weight_cache = torch.cat(
            [
                block.ff_proj.weight,
                block.up_proj.weight,
            ],
            dim=0,
        )

    new_ff, new_up = torch.nn.functional.linear(
        x,
        block._ff_up_weight_cache,
        block._ff_up_bias_cache,
    ).chunk(2, dim=-1)


print("ff max error:", (old_ff-new_ff).abs().max().item())
print("up max error:", (old_up-new_up).abs().max().item())
