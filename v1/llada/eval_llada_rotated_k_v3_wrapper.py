from __future__ import annotations

import gc
import runpy
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel

from model import modeling_llada as modeling


if not torch.backends.mps.is_available():
    raise RuntimeError("MPS is unavailable")


# ============================================================
# 加载已经验证过的 Rotated-K v2 实现，但不启动其评测。
# ============================================================

V2_WRAPPER = (
    Path.home()
    / "Fast-dLLM"
    / "v1"
    / "llada"
    / "eval_llada_rotated_k_v2_wrapper.py"
)

source = V2_WRAPPER.read_text(encoding="utf-8")
marker = "\nEVAL_PATH = ("

if marker not in source:
    raise RuntimeError(
        "无法在 v2 wrapper 中找到 EVAL_PATH 分隔符"
    )

namespace: dict[str, Any] = {
    "__name__": "rotated_k_v2_implementation",
}

exec(
    compile(
        source.split(marker, 1)[0],
        str(V2_WRAPPER),
        "exec",
    ),
    namespace,
)

# 上面的代码已经将 attention 替换成 Rotated-K v2。
rotated_k_v2_attention = namespace[
    "rotated_k_v2_attention"
]

modeling.LLaDABlock.attention = (
    rotated_k_v2_attention
)


# ============================================================
# Gate/Up fused projection
# ============================================================

class FusedGateUp(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()

        self.weight = nn.Parameter(
            weight,
            requires_grad=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return F.linear(
            x,
            self.weight,
            bias=None,
        )


def fuse_gate_up_modules(
    model: nn.Module,
) -> int:
    """
    Convert each pair:

        ff_proj: [12288, 4096]
        up_proj: [12288, 4096]

    into:

        fused_gate_up: [24576, 4096]

    Original modules are removed after loading, so persistent model memory
    does not retain all three copies.
    """

    fused_count = 0

    for module in list(model.modules()):
        if module.__class__.__name__ != "LLaDALlamaBlock":
            continue

        if hasattr(module, "fused_gate_up"):
            continue

        if not hasattr(module, "ff_proj"):
            continue

        if not hasattr(module, "up_proj"):
            continue

        ff_proj = module.ff_proj
        up_proj = module.up_proj

        if ff_proj.bias is not None:
            raise RuntimeError(
                "ff_proj bias is unsupported"
            )

        if up_proj.bias is not None:
            raise RuntimeError(
                "up_proj bias is unsupported"
            )

        if ff_proj.weight.shape != up_proj.weight.shape:
            raise RuntimeError(
                "Gate/Up weight shapes differ: "
                f"{ff_proj.weight.shape} vs "
                f"{up_proj.weight.shape}"
            )

        fused_weight = torch.cat(
            (
                ff_proj.weight.detach(),
                up_proj.weight.detach(),
            ),
            dim=0,
        ).contiguous()

        module.fused_gate_up = FusedGateUp(
            fused_weight
        )

        # 模型已经完成 checkpoint 加载，可以安全删除原模块。
        delattr(module, "ff_proj")
        delattr(module, "up_proj")

        fused_count += 1

        del ff_proj
        del up_proj
        del fused_weight

    gc.collect()

    return fused_count


# ============================================================
# 在 checkpoint 加载完成后自动完成 fusion。
# ============================================================

ORIGINAL_FROM_PRETRAINED = (
    PreTrainedModel.from_pretrained.__func__
)


def patched_from_pretrained(
    cls,
    *args,
    **kwargs,
):
    model = ORIGINAL_FROM_PRETRAINED(
        cls,
        *args,
        **kwargs,
    )

    count = fuse_gate_up_modules(model)

    if count:
        print("=" * 80)
        print(
            f"Fused Gate/Up projection enabled: "
            f"{count} transformer blocks"
        )
        print("=" * 80)

    return model


PreTrainedModel.from_pretrained = classmethod(
    patched_from_pretrained
)


# ============================================================
# 使用 fused Gate/Up 的 LLaDA block forward。
# ============================================================

def fused_block_forward(
    self,
    x: torch.Tensor,
    attention_bias: torch.Tensor | None = None,
    layer_past=None,
    use_cache: bool = False,
    replace_position: torch.Tensor | None = None,
):
    if self._activation_checkpoint_fn is not None:
        raise RuntimeError(
            "Gate/Up v3 wrapper only supports inference"
        )

    if not hasattr(self, "fused_gate_up"):
        raise RuntimeError(
            "Gate/Up fusion was not applied to this block"
        )

    # Attention pre-norm and projections.
    x_normed = self.attn_norm(x)

    q = self.q_proj(x_normed)
    k = self.k_proj(x_normed)
    v = self.v_proj(x_normed)

    att, cache = self.attention(
        q,
        k,
        v,
        attention_bias,
        layer_past=layer_past,
        use_cache=use_cache,
        replace_position=replace_position,
    )

    hidden = x + self.dropout(att)

    # Fused SwiGLU input projection.
    mlp_input = self.ff_norm(hidden)

    gate_up = self.fused_gate_up(
        mlp_input
    )

    gate, up = gate_up.chunk(
        2,
        dim=-1,
    )

    activated = self.act(gate) * up

    mlp_output = self.ff_out(
        activated
    )

    output = hidden + self.dropout(
        mlp_output
    )

    return output, cache


modeling.LLaDALlamaBlock.forward = (
    fused_block_forward
)


print("=" * 80)
print("Experimental Rotated-K Cache v3 enabled")
print("v2 position metadata reuse + fused Gate/Up projection")
print("=" * 80)


EVAL_PATH = (
    Path.home()
    / "Fast-dLLM"
    / "v1"
    / "llada"
    / "eval_llada.py"
)

sys.argv[0] = str(EVAL_PATH)

runpy.run_path(
    str(EVAL_PATH),
    run_name="__main__",
)
