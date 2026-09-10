from __future__ import annotations

import atexit
import json
import os
import runpy
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch

from model import modeling_llada as modeling


if not torch.backends.mps.is_available():
    raise RuntimeError("MPS is unavailable")


TARGET_INDICES = {
    int(value.strip())
    for value in os.environ.get(
        "LLADA_PROFILE_FORWARD_INDICES",
        "0,8,32",
    ).split(",")
    if value.strip()
}

NUM_LAYERS = int(
    os.environ.get("LLADA_NUM_LAYERS", "32")
)

OUTPUT_JSON = Path(
    os.environ.get(
        "LLADA_COMPONENT_PROFILE_JSON",
        str(
            Path.home()
            / "Fast-dLLM"
            / "results"
            / "llada_component_profile.json"
        ),
    )
)

ORIGINAL_FORWARD = modeling.LLaDALlamaBlock.forward

STATE: dict[str, Any] = {
    "forward_index": -1,
    "records": {},
}


def sync() -> None:
    torch.mps.synchronize()


def measure(
    record: dict[str, Any],
    layer_id: int,
    name: str,
    function: Callable[[], Any],
) -> Any:
    sync()
    started = time.perf_counter()

    result = function()

    sync()
    elapsed = time.perf_counter() - started

    record["phases"][name] += elapsed
    record["layers"][str(layer_id)][name] += elapsed

    return result


def profiled_forward(
    self,
    x: torch.Tensor,
    attention_bias: torch.Tensor | None = None,
    layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
    use_cache: bool = False,
    replace_position: torch.Tensor | None = None,
):
    # layer 0 marks the start of one complete 32-layer model forward.
    if self.layer_id == 0:
        STATE["forward_index"] += 1

        index = STATE["forward_index"]

        if index in TARGET_INDICES:
            past_len = (
                int(layer_past[0].shape[-2])
                if layer_past is not None
                else 0
            )

            STATE["records"][index] = {
                "forward_index": index,
                "input_shape": list(x.shape),
                "input_dtype": str(x.dtype),
                "past_length_at_layer0": past_len,
                "replace_count": (
                    int(replace_position.sum().item())
                    if replace_position is not None
                    else 0
                ),
                "phases": defaultdict(float),
                "layers": {
                    str(layer): defaultdict(float)
                    for layer in range(NUM_LAYERS)
                },
            }

    index = STATE["forward_index"]

    if index not in TARGET_INDICES:
        return ORIGINAL_FORWARD(
            self,
            x,
            attention_bias=attention_bias,
            layer_past=layer_past,
            use_cache=use_cache,
            replace_position=replace_position,
        )

    record = STATE["records"][index]
    layer_id = self.layer_id

    # --------------------------------------------------------
    # Attention input
    # --------------------------------------------------------

    x_normed = measure(
        record,
        layer_id,
        "attn_norm",
        lambda: self.attn_norm(x),
    )

    def qkv_projections():
        return (
            self.q_proj(x_normed),
            self.k_proj(x_normed),
            self.v_proj(x_normed),
        )

    q, k, v = measure(
        record,
        layer_id,
        "qkv_projections",
        qkv_projections,
    )

    batch_size, query_len, hidden_size = q.shape
    head_dim = hidden_size // self.config.n_heads

    def reshape_heads():
        q_heads = q.view(
            batch_size,
            query_len,
            self.config.n_heads,
            head_dim,
        ).transpose(1, 2)

        k_heads = k.view(
            batch_size,
            query_len,
            self.config.effective_n_kv_heads,
            head_dim,
        ).transpose(1, 2)

        v_heads = v.view(
            batch_size,
            query_len,
            self.config.effective_n_kv_heads,
            head_dim,
        ).transpose(1, 2)

        return q_heads, k_heads, v_heads

    q, k, v = measure(
        record,
        layer_id,
        "head_reshape",
        reshape_heads,
    )

    # --------------------------------------------------------
    # DualCache / Prefix Cache update
    # --------------------------------------------------------

    def update_cache():
        updated_k = k
        updated_v = v

        if layer_past is None:
            return updated_k, updated_v

        past_key, past_value = layer_past

        if replace_position is None:
            updated_k = torch.cat(
                (past_key, updated_k),
                dim=-2,
            )
            updated_v = torch.cat(
                (past_value, updated_v),
                dim=-2,
            )

            return updated_k, updated_v

        replacement_batch = replace_position.shape[0]

        for batch_index in range(replacement_batch):
            replacement_indices = (
                replace_position[batch_index]
                .nonzero(as_tuple=True)[0]
            )

            count = len(replacement_indices)

            if count == 0:
                continue

            past_key[
                batch_index,
                :,
                replacement_indices,
            ] = updated_k[
                batch_index,
                :,
                :count,
            ]

            past_value[
                batch_index,
                :,
                replacement_indices,
            ] = updated_v[
                batch_index,
                :,
                :count,
            ]

        return past_key, past_value

    k, v = measure(
        record,
        layer_id,
        "cache_update",
        update_cache,
    )

    # Cache contains unrotated K/V, matching the source.
    present = (k, v) if use_cache else None

    # --------------------------------------------------------
    # RoPE
    # --------------------------------------------------------

    def apply_rope():
        if not self.config.rope:
            return q, k

        if replace_position is None:
            return self.rotary_emb(q, k)

        if replace_position.any():
            block_end_index = (
                replace_position
                .nonzero(as_tuple=True)[1]
                .max()
                + 1
            )
        else:
            block_end_index = k.shape[-2]

        return self.rotary_emb(
            q,
            k,
            block_end_index,
        )

    q_rotated, k_rotated = measure(
        record,
        layer_id,
        "rope",
        apply_rope,
    )

    # --------------------------------------------------------
    # SDPA
    # --------------------------------------------------------

    attention = measure(
        record,
        layer_id,
        "sdpa",
        lambda: self._scaled_dot_product_attention(
            q_rotated,
            k_rotated,
            v,
            attn_mask=None,
            dropout_p=(
                0.0
                if not self.training
                else self.config.attention_dropout
            ),
            is_causal=False,
        ),
    )

    def attention_output():
        merged = (
            attention.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                query_len,
                hidden_size,
            )
        )

        projected = self.attn_out(merged)

        return x + self.dropout(projected)

    hidden = measure(
        record,
        layer_id,
        "attn_out_residual",
        attention_output,
    )

    # --------------------------------------------------------
    # SwiGLU MLP
    # --------------------------------------------------------

    mlp_input = measure(
        record,
        layer_id,
        "ff_norm",
        lambda: self.ff_norm(hidden),
    )

    gate, up = measure(
        record,
        layer_id,
        "gate_up_projections",
        lambda: (
            self.ff_proj(mlp_input),
            self.up_proj(mlp_input),
        ),
    )

    activated = measure(
        record,
        layer_id,
        "silu_multiply",
        lambda: self.act(gate) * up,
    )

    output = measure(
        record,
        layer_id,
        "ff_out_residual",
        lambda: hidden
        + self.dropout(self.ff_out(activated)),
    )

    return output, present


modeling.LLaDALlamaBlock.forward = profiled_forward


PHASE_ORDER = [
    "attn_norm",
    "qkv_projections",
    "head_reshape",
    "cache_update",
    "rope",
    "sdpa",
    "attn_out_residual",
    "ff_norm",
    "gate_up_projections",
    "silu_multiply",
    "ff_out_residual",
]


def report() -> None:
    records = STATE["records"]

    serializable = {}

    print()
    print("=" * 100)
    print("LLaDA component profile")
    print("=" * 100)
    print(
        "注意：每个 phase 前后都进行了 MPS synchronize；"
        "结果适合比较占比和排序，不等同于无扰动端到端延迟。"
    )

    for index in sorted(records):
        record = records[index]

        phases = {
            key: float(value)
            for key, value in record["phases"].items()
        }

        total = sum(phases.values())

        serializable[str(index)] = {
            "forward_index": index,
            "input_shape": record["input_shape"],
            "input_dtype": record["input_dtype"],
            "past_length_at_layer0": (
                record["past_length_at_layer0"]
            ),
            "replace_count": record["replace_count"],
            "profiled_phase_total_seconds": total,
            "phases_seconds": phases,
            "layers_seconds": {
                layer: {
                    key: float(value)
                    for key, value in layer_values.items()
                }
                for layer, layer_values
                in record["layers"].items()
            },
        }

        print()
        print("-" * 100)
        print(
            f"Forward index={index}, "
            f"x={record['input_shape']}, "
            f"past_len={record['past_length_at_layer0']}, "
            f"replace_count={record['replace_count']}"
        )
        print("-" * 100)

        print(
            f"{'Phase':<26}"
            f"{'Total ms':>14}"
            f"{'Per layer ms':>16}"
            f"{'Share':>12}"
        )

        for phase in sorted(
            PHASE_ORDER,
            key=lambda name: phases.get(name, 0.0),
            reverse=True,
        ):
            seconds = phases.get(phase, 0.0)
            milliseconds = seconds * 1000
            per_layer = milliseconds / NUM_LAYERS
            share = seconds / total * 100 if total else 0.0

            print(
                f"{phase:<26}"
                f"{milliseconds:>14.3f}"
                f"{per_layer:>16.3f}"
                f"{share:>11.2f}%"
            )

        groups = {
            "Attention projections": (
                phases.get("attn_norm", 0.0)
                + phases.get("qkv_projections", 0.0)
            ),
            "Cache + RoPE": (
                phases.get("head_reshape", 0.0)
                + phases.get("cache_update", 0.0)
                + phases.get("rope", 0.0)
            ),
            "SDPA": phases.get("sdpa", 0.0),
            "Attention output": phases.get(
                "attn_out_residual",
                0.0,
            ),
            "MLP": (
                phases.get("ff_norm", 0.0)
                + phases.get("gate_up_projections", 0.0)
                + phases.get("silu_multiply", 0.0)
                + phases.get("ff_out_residual", 0.0)
            ),
        }

        print()
        print("Grouped:")

        for name, seconds in sorted(
            groups.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            share = seconds / total * 100 if total else 0.0

            print(
                f"  {name:<24}"
                f"{seconds * 1000:>12.3f} ms"
                f"{share:>10.2f}%"
            )

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_JSON.write_text(
        json.dumps(
            serializable,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("Profile JSON:")
    print(OUTPUT_JSON)


atexit.register(report)


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
