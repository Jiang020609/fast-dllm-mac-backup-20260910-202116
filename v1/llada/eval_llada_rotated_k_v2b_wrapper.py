from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

import generate as generate_module
from generate_v2b import generate_with_dual_cache_v2b
from model import modeling_llada as modeling
from rotated_k_position_context import (
    BlockBounds,
    get_position_state,
)


if not torch.backends.mps.is_available():
    raise RuntimeError("MPS is unavailable")


def select_rope_rows(
    table: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """
    table:        [1, 1, table_length, head_dim]
    position_ids: [batch, query_length]
    output:       [batch, 1, query_length, head_dim]
    """
    base = table[0, 0]

    if position_ids.shape[0] == 1:
        return (
            base.index_select(
                0,
                position_ids[0],
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

    rows = [
        base.index_select(
            0,
            position_ids[batch_index],
        )
        for batch_index in range(
            position_ids.shape[0]
        )
    ]

    return torch.stack(
        rows,
        dim=0,
    ).unsqueeze(1)


def build_position_context(
    self,
    q: torch.Tensor,
    layer_past: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ],
    bounds: BlockBounds | None,
) -> dict:
    batch_size = q.shape[0]
    query_length = q.shape[-2]
    device = q.device

    if bounds is not None:
        if layer_past is None:
            raise RuntimeError(
                "Explicit replacement bounds require layer_past"
            )

        key_length = int(
            layer_past[0].shape[-2]
        )

        if bounds.length != query_length:
            raise RuntimeError(
                "Block/query length mismatch: "
                f"bounds={bounds.start}:{bounds.end}, "
                f"query_length={query_length}"
            )

        if bounds.end > key_length:
            raise RuntimeError(
                "Replacement bounds exceed KV cache: "
                f"end={bounds.end}, key_length={key_length}"
            )

        positions = torch.arange(
            bounds.start,
            bounds.end,
            device=device,
            dtype=torch.long,
        )

    elif layer_past is None:
        key_length = query_length

        positions = torch.arange(
            0,
            query_length,
            device=device,
            dtype=torch.long,
        )

    else:
        past_length = int(
            layer_past[0].shape[-2]
        )
        key_length = (
            past_length + query_length
        )

        positions = torch.arange(
            past_length,
            key_length,
            device=device,
            dtype=torch.long,
        )

    position_ids = positions.unsqueeze(0).expand(
        batch_size,
        -1,
    )

    rope_table_length = max(
        int(self.config.max_sequence_length),
        key_length,
    )

    work_dtype = (
        torch.float32
        if self.config.rope_full_precision
        else q.dtype
    )

    pos_sin, pos_cos = (
        self.rotary_emb.get_rotary_embedding(
            rope_table_length,
            device,
        )
    )

    pos_sin = pos_sin.to(
        dtype=work_dtype
    )
    pos_cos = pos_cos.to(
        dtype=work_dtype
    )

    selected_sin = select_rope_rows(
        pos_sin,
        position_ids,
    )
    selected_cos = select_rope_rows(
        pos_cos,
        position_ids,
    )

    return {
        "query_length": query_length,
        "key_length": key_length,
        "bounds": bounds,
        # Q 和 new K 的绝对位置相同，直接复用切片。
        "sin": selected_sin,
        "cos": selected_cos,
    }


def apply_cached_rope(
    self,
    tensor: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
) -> torch.Tensor:
    original_dtype = tensor.dtype

    work = (
        tensor.float()
        if self.config.rope_full_precision
        else tensor
    )

    with torch.autocast(
        device_type=tensor.device.type,
        enabled=False,
    ):
        rotated = (
            self.rotary_emb
            .apply_rotary_pos_emb(
                sin,
                cos,
                work,
            )
        )

    return rotated.to(original_dtype)


def rotated_k_v2b_attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    attention_bias: Optional[torch.Tensor] = None,
    layer_past: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None,
    use_cache: bool = False,
    replace_position: Optional[
        torch.Tensor
    ] = None,
    position_state=None,
):
    if replace_position is not None:
        raise RuntimeError(
            "v2b does not accept replace_position; "
            "use explicit block bounds"
        )

    state = (
        position_state
        if position_state is not None
        else get_position_state()
    )

    batch_size, query_length, hidden_size = (
        q.size()
    )
    dtype = k.dtype

    if (
        self.q_norm is not None
        and self.k_norm is not None
    ):
        q = self.q_norm(q).to(dtype=dtype)
        k = self.k_norm(k).to(dtype=dtype)

    head_dim = (
        hidden_size
        // self.config.n_heads
    )

    q = q.view(
        batch_size,
        query_length,
        self.config.n_heads,
        head_dim,
    ).transpose(1, 2)

    k = k.view(
        batch_size,
        query_length,
        self.config.effective_n_kv_heads,
        head_dim,
    ).transpose(1, 2)

    v = v.view(
        batch_size,
        query_length,
        self.config.effective_n_kv_heads,
        head_dim,
    ).transpose(1, 2)

    if self.layer_id == 0:
        state.position_context = (
            build_position_context(
                self,
                q,
                layer_past,
                state.bounds,
            )
        )

    context = state.position_context

    if context is None:
        raise RuntimeError(
            "Position context was not initialized"
        )

    if (
        context["query_length"]
        != query_length
    ):
        raise RuntimeError(
            "Query length changed across layers"
        )

    q_rotated = apply_cached_rope(
        self,
        q,
        context["sin"],
        context["cos"],
    )

    new_k_rotated = apply_cached_rope(
        self,
        k,
        context["sin"],
        context["cos"],
    )

    bounds = context["bounds"]

    if layer_past is None:
        full_k_rotated = new_k_rotated
        full_v = v

    elif bounds is None:
        past_k_rotated, past_value = (
            layer_past
        )

        full_k_rotated = torch.cat(
            (
                past_k_rotated,
                new_k_rotated,
            ),
            dim=-2,
        )

        full_v = torch.cat(
            (
                past_value,
                v,
            ),
            dim=-2,
        )

    else:
        past_k_rotated, past_value = (
            layer_past
        )

        full_k_rotated = past_k_rotated
        full_v = past_value

        # 连续区间直接切片，支持整个 batch；
        # 不再使用 nonzero、高级索引或 Python batch loop。
        full_k_rotated[
            :,
            :,
            bounds.start:bounds.end,
            :,
        ] = new_k_rotated

        full_v[
            :,
            :,
            bounds.start:bounds.end,
            :,
        ] = v

    present = (
        (full_k_rotated, full_v)
        if use_cache
        else None
    )

    final_key_length = int(
        full_k_rotated.shape[-2]
    )

    if attention_bias is not None:
        attention_bias = (
            self._cast_attn_bias(
                attention_bias[
                    :,
                    :,
                    final_key_length
                    - query_length:
                    final_key_length,
                    :final_key_length,
                ],
                dtype,
            )
        )

    attention = (
        self._scaled_dot_product_attention(
            q_rotated,
            full_k_rotated,
            full_v,
            attn_mask=None,
            dropout_p=(
                0.0
                if not self.training
                else self.config.attention_dropout
            ),
            is_causal=False,
        )
    )

    attention = (
        attention.transpose(1, 2)
        .contiguous()
        .view(
            batch_size,
            query_length,
            hidden_size,
        )
    )

    return self.attn_out(attention), present


modeling.LLaDABlock.attention = (
    rotated_k_v2b_attention
)

# eval_llada.py 使用：
# from generate import generate_with_dual_cache
# 因此在 runpy 前替换 generate 模块属性即可。
generate_module.generate_with_dual_cache = (
    generate_with_dual_cache_v2b
)


print("=" * 80)
print("Rotated-K v2b enabled")
print("Explicit block bounds + request-local ContextVar")
print("No replace_position/nonzero/global POSITION_CONTEXT")
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
