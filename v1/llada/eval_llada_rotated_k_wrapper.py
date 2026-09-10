from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

from model import modeling_llada as modeling


if not torch.backends.mps.is_available():
    raise RuntimeError("MPS is unavailable")


def apply_rope_at_positions(
    self,
    tensor: torch.Tensor,
    position_ids: torch.Tensor,
    table_length: int,
) -> torch.Tensor:
    """
    tensor:
        [B, H, T, D]

    position_ids:
        [B, T]

    Apply exactly the same split-half RoPE as the original implementation,
    but only at the requested absolute positions.
    """

    if position_ids.ndim != 2:
        raise ValueError(
            f"position_ids must be [B, T], got {position_ids.shape}"
        )

    batch_size, _, seq_len, _ = tensor.shape

    if position_ids.shape != (batch_size, seq_len):
        raise ValueError(
            "position_ids shape mismatch: "
            f"tensor={tensor.shape}, positions={position_ids.shape}"
        )

    original_dtype = tensor.dtype

    if self.config.rope_full_precision:
        work = tensor.float()
    else:
        work = tensor

    with torch.autocast(
        device_type=tensor.device.type,
        enabled=False,
    ):
        pos_sin, pos_cos = (
            self.rotary_emb.get_rotary_embedding(
                table_length,
                tensor.device,
            )
        )

        pos_sin = pos_sin.type_as(work)
        pos_cos = pos_cos.type_as(work)

        # Use index_select rather than complicated advanced indexing
        # for better MPS compatibility.
        selected_sin = []
        selected_cos = []

        sin_table = pos_sin[0, 0]
        cos_table = pos_cos[0, 0]

        for batch_index in range(batch_size):
            indices = position_ids[batch_index]

            selected_sin.append(
                sin_table.index_select(0, indices)
            )
            selected_cos.append(
                cos_table.index_select(0, indices)
            )

        selected_sin = torch.stack(
            selected_sin,
            dim=0,
        ).unsqueeze(1)

        selected_cos = torch.stack(
            selected_cos,
            dim=0,
        ).unsqueeze(1)

        rotated = self.rotary_emb.apply_rotary_pos_emb(
            selected_sin,
            selected_cos,
            work,
        )

    return rotated.to(original_dtype)


def rotated_k_attention(
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
    """
    Experimental attention implementation.

    Cache format:
        K cache: already RoPE-rotated BF16 K
        V cache: unchanged BF16 V
    """

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

    # ========================================================
    # Work out absolute RoPE positions
    # ========================================================

    if layer_past is None:
        past_length = 0
        key_length = query_len

        positions = torch.arange(
            0,
            query_len,
            device=q.device,
            dtype=torch.long,
        )

        q_positions = positions.unsqueeze(0).expand(
            batch_size,
            -1,
        )

        new_k_positions = q_positions

    elif replace_position is None:
        # Prefix/append-cache path.
        past_key, _ = layer_past
        past_length = int(past_key.shape[-2])
        key_length = past_length + query_len

        positions = torch.arange(
            past_length,
            key_length,
            device=q.device,
            dtype=torch.long,
        )

        q_positions = positions.unsqueeze(0).expand(
            batch_size,
            -1,
        )

        new_k_positions = q_positions

    else:
        # DualCache replacement path.
        past_key, _ = layer_past
        past_length = int(past_key.shape[-2])
        key_length = past_length

        replacement_positions = []

        for batch_index in range(batch_size):
            indices = (
                replace_position[batch_index]
                .nonzero(as_tuple=True)[0]
            )

            if indices.numel() != query_len:
                raise RuntimeError(
                    "Rotated-K Cache expected exactly "
                    f"{query_len} replacement positions, but batch "
                    f"{batch_index} has {indices.numel()}"
                )

            replacement_positions.append(indices)

        new_k_positions = torch.stack(
            replacement_positions,
            dim=0,
        )

        # Preserve the original q-position behavior:
        # q uses the final query_len positions ending at
        # the maximum replacement position.
        if replace_position.any():
            block_end_index = (
                replace_position
                .nonzero(as_tuple=True)[1]
                .max()
                + 1
            )
        else:
            block_end_index = key_length

        q_position_range = torch.arange(
            block_end_index - query_len,
            block_end_index,
            device=q.device,
            dtype=torch.long,
        )

        q_positions = q_position_range.unsqueeze(0).expand(
            batch_size,
            -1,
        )

    rope_table_length = max(
        int(self.config.max_sequence_length),
        int(key_length),
    )

    # Only rotate the current Q and current K.
    q_rotated = apply_rope_at_positions(
        self,
        q,
        q_positions,
        rope_table_length,
    )

    new_k_rotated = apply_rope_at_positions(
        self,
        k,
        new_k_positions,
        rope_table_length,
    )

    # ========================================================
    # Construct/update the rotated-K cache
    # ========================================================

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

        # Match the original in-place replacement semantics.
        full_k_rotated = past_k_rotated
        full_v = past_value

        for batch_index in range(batch_size):
            indices = new_k_positions[batch_index]

            full_k_rotated[
                batch_index,
                :,
                indices,
                :,
            ] = new_k_rotated[
                batch_index,
                :,
                :,
                :,
            ]

            full_v[
                batch_index,
                :,
                indices,
                :,
            ] = v[
                batch_index,
                :,
                :,
                :,
            ]

    present = (
        (full_k_rotated, full_v)
        if use_cache
        else None
    )

    final_key_length = int(full_k_rotated.shape[-2])

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


# LLaDALlamaBlock inherits attention from LLaDABlock.
modeling.LLaDABlock.attention = rotated_k_attention

print("=" * 80)
print("Experimental Rotated-K Cache enabled")
print("K cache stores RoPE-rotated keys")
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
