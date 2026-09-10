from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import torch

from model import modeling_llada as modeling


if not torch.backends.mps.is_available():
    raise RuntimeError("MPS is unavailable")


POSITION_CONTEXT: dict[str, Any] | None = None


def select_rope_rows(
    table: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """
    table: [1, 1, table_len, head_dim]
    position_ids: [B, T]
    output: [B, 1, T, head_dim]
    """
    base = table[0, 0]

    if position_ids.shape[0] == 1:
        return (
            base.index_select(0, position_ids[0])
            .unsqueeze(0)
            .unsqueeze(0)
        )

    rows = [
        base.index_select(0, position_ids[b])
        for b in range(position_ids.shape[0])
    ]

    return torch.stack(rows, dim=0).unsqueeze(1)


def build_position_context(
    self,
    q: torch.Tensor,
    layer_past: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ],
    replace_position: Optional[torch.Tensor],
) -> dict[str, Any]:
    batch_size = q.shape[0]
    query_len = q.shape[-2]
    device = q.device

    if layer_past is None:
        key_length = query_len

        positions = torch.arange(
            query_len,
            device=device,
            dtype=torch.long,
        )

        q_positions = positions.unsqueeze(0).expand(
            batch_size,
            -1,
        )

        new_k_positions = q_positions
        same_positions = True

    elif replace_position is None:
        past_length = int(layer_past[0].shape[-2])
        key_length = past_length + query_len

        positions = torch.arange(
            past_length,
            key_length,
            device=device,
            dtype=torch.long,
        )

        q_positions = positions.unsqueeze(0).expand(
            batch_size,
            -1,
        )

        new_k_positions = q_positions
        same_positions = True

    else:
        key_length = int(layer_past[0].shape[-2])
        replacement_positions = []

        for batch_index in range(batch_size):
            indices = (
                replace_position[batch_index]
                .nonzero(as_tuple=True)[0]
            )

            if indices.numel() != query_len:
                raise RuntimeError(
                    "Replacement count mismatch: "
                    f"expected={query_len}, "
                    f"actual={indices.numel()}"
                )

            replacement_positions.append(indices)

        new_k_positions = torch.stack(
            replacement_positions,
            dim=0,
        )

        nonzero = replace_position.nonzero(
            as_tuple=True
        )

        if nonzero[1].numel() > 0:
            block_end_index = nonzero[1].max() + 1
        else:
            block_end_index = key_length

        q_range = torch.arange(
            block_end_index - query_len,
            block_end_index,
            device=device,
            dtype=torch.long,
        )

        q_positions = q_range.unsqueeze(0).expand(
            batch_size,
            -1,
        )

        same_positions = False

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

    pos_sin = pos_sin.to(dtype=work_dtype)
    pos_cos = pos_cos.to(dtype=work_dtype)

    q_sin = select_rope_rows(
        pos_sin,
        q_positions,
    )
    q_cos = select_rope_rows(
        pos_cos,
        q_positions,
    )

    if same_positions:
        k_sin = q_sin
        k_cos = q_cos
    else:
        k_sin = select_rope_rows(
            pos_sin,
            new_k_positions,
        )
        k_cos = select_rope_rows(
            pos_cos,
            new_k_positions,
        )

    return {
        "query_len": query_len,
        "key_length": key_length,
        "new_k_positions": new_k_positions,
        "q_sin": q_sin,
        "q_cos": q_cos,
        "k_sin": k_sin,
        "k_cos": k_cos,
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
        rotated = self.rotary_emb.apply_rotary_pos_emb(
            sin,
            cos,
            work,
        )

    return rotated.to(original_dtype)


def rotated_k_v2_attention(
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
    replace_position: Optional[torch.Tensor] = None,
):
    global POSITION_CONTEXT

    batch_size, query_len, hidden_size = q.size()
    dtype = k.dtype

    if self.q_norm is not None and self.k_norm is not None:
        q = self.q_norm(q).to(dtype=dtype)
        k = self.k_norm(k).to(dtype=dtype)

    head_dim = hidden_size // self.config.n_heads

    q = q.view(
        batch_size,
        query_len,
        self.config.n_heads,
        head_dim,
    ).transpose(1, 2)

    k = k.view(
        batch_size,
        query_len,
        self.config.effective_n_kv_heads,
        head_dim,
    ).transpose(1, 2)

    v = v.view(
        batch_size,
        query_len,
        self.config.effective_n_kv_heads,
        head_dim,
    ).transpose(1, 2)

    # 第 0 层只构建一次，后续 31 层直接复用。
    if self.layer_id == 0:
        POSITION_CONTEXT = build_position_context(
            self,
            q,
            layer_past,
            replace_position,
        )

    if POSITION_CONTEXT is None:
        raise RuntimeError(
            "Position context was not initialized"
        )

    if POSITION_CONTEXT["query_len"] != query_len:
        raise RuntimeError(
            "Query length changed inside one model forward"
        )

    q_rotated = apply_cached_rope(
        self,
        q,
        POSITION_CONTEXT["q_sin"],
        POSITION_CONTEXT["q_cos"],
    )

    new_k_rotated = apply_cached_rope(
        self,
        k,
        POSITION_CONTEXT["k_sin"],
        POSITION_CONTEXT["k_cos"],
    )

    if layer_past is None:
        full_k_rotated = new_k_rotated
        full_v = v

    elif replace_position is None:
        past_k_rotated, past_value = layer_past

        full_k_rotated = torch.cat(
            (past_k_rotated, new_k_rotated),
            dim=-2,
        )

        full_v = torch.cat(
            (past_value, v),
            dim=-2,
        )

    else:
        past_k_rotated, past_value = layer_past
        full_k_rotated = past_k_rotated
        full_v = past_value

        positions = POSITION_CONTEXT[
            "new_k_positions"
        ]

        for batch_index in range(batch_size):
            indices = positions[batch_index]

            full_k_rotated[
                batch_index,
                :,
                indices,
                :,
            ] = new_k_rotated[batch_index]

            full_v[
                batch_index,
                :,
                indices,
                :,
            ] = v[batch_index]

    present = (
        (full_k_rotated, full_v)
        if use_cache
        else None
    )

    final_key_length = int(
        full_k_rotated.shape[-2]
    )

    if attention_bias is not None:
        attention_bias = self._cast_attn_bias(
            attention_bias[
                :,
                :,
                final_key_length - query_len:
                final_key_length,
                :final_key_length,
            ],
            dtype,
        )

    attention = self._scaled_dot_product_attention(
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

    attention = (
        attention.transpose(1, 2)
        .contiguous()
        .view(
            batch_size,
            query_len,
            hidden_size,
        )
    )

    return self.attn_out(attention), present


modeling.LLaDABlock.attention = (
    rotated_k_v2_attention
)


print("=" * 80)
print("Experimental Rotated-K Cache v2 enabled")
print(
    "Position metadata and RoPE slices "
    "are reused across 32 layers"
)
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
