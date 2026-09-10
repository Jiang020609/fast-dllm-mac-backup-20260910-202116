from __future__ import annotations

import atexit
import json
import os
import runpy
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torch

from model import modeling_llada as modeling


if not torch.backends.mps.is_available():
    raise RuntimeError("MPS is unavailable")


# ============================================================
# 复用已经通过 smoke test 和 5 轮 A/B 的 Rotated-K 实现。
# 只执行 wrapper 中 EVAL_PATH 之前的定义，不启动里面的评测。
# ============================================================

ROTATED_WRAPPER = (
    Path.home()
    / "Fast-dLLM"
    / "v1"
    / "llada"
    / "eval_llada_rotated_k_wrapper.py"
)

source = ROTATED_WRAPPER.read_text(encoding="utf-8")
marker = "\nEVAL_PATH = ("

if marker not in source:
    raise RuntimeError(
        "无法从 Rotated-K wrapper 中找到 EVAL_PATH 分隔符"
    )

implementation_source = source.split(marker, 1)[0]

namespace: dict[str, Any] = {
    "__name__": "rotated_k_implementation",
}

exec(
    compile(
        implementation_source,
        str(ROTATED_WRAPPER),
        "exec",
    ),
    namespace,
)

apply_rope_at_positions = namespace[
    "apply_rope_at_positions"
]
rotated_k_attention = namespace[
    "rotated_k_attention"
]

# 非目标 forward 继续走已经验证过的 Rotated-K attention。
modeling.LLaDABlock.attention = rotated_k_attention

ORIGINAL_FORWARD = modeling.LLaDALlamaBlock.forward


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
            / "rotated_k_component_profile.json"
        ),
    )
)

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
    start = time.perf_counter()

    result = function()

    sync()
    elapsed = time.perf_counter() - start

    record["phases"][name] += elapsed
    record["layers"][str(layer_id)][name] += elapsed

    return result


def profiled_forward(
    self,
    x: torch.Tensor,
    attention_bias: Optional[torch.Tensor] = None,
    layer_past: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None,
    use_cache: bool = False,
    replace_position: Optional[torch.Tensor] = None,
):
    # 第 0 层代表一次完整 32 层 forward 的开始。
    if self.layer_id == 0:
        STATE["forward_index"] += 1
        index = STATE["forward_index"]

        if index in TARGET_INDICES:
            STATE["records"][index] = {
                "forward_index": index,
                "input_shape": list(x.shape),
                "input_dtype": str(x.dtype),
                "past_length_at_layer0": (
                    int(layer_past[0].shape[-2])
                    if layer_past is not None
                    else 0
                ),
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

    # 未选中的 forward 走真实 Rotated-K 实现，不插同步。
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

    # ========================================================
    # Attention input projection
    # ========================================================

    x_normed = measure(
        record,
        layer_id,
        "attn_norm",
        lambda: self.attn_norm(x),
    )

    q, new_k, new_v = measure(
        record,
        layer_id,
        "qkv_projections",
        lambda: (
            self.q_proj(x_normed),
            self.k_proj(x_normed),
            self.v_proj(x_normed),
        ),
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

        k_heads = new_k.view(
            batch_size,
            query_len,
            self.config.effective_n_kv_heads,
            head_dim,
        ).transpose(1, 2)

        v_heads = new_v.view(
            batch_size,
            query_len,
            self.config.effective_n_kv_heads,
            head_dim,
        ).transpose(1, 2)

        return q_heads, k_heads, v_heads

    q, new_k, new_v = measure(
        record,
        layer_id,
        "head_reshape",
        reshape_heads,
    )

    # ========================================================
    # 构建 Q 和新 K 的绝对位置
    # ========================================================

    def build_positions():
        if layer_past is None:
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

            return (
                q_positions,
                q_positions,
                key_length,
            )

        past_key, _ = layer_past
        past_length = int(past_key.shape[-2])

        if replace_position is None:
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

            return (
                q_positions,
                q_positions,
                key_length,
            )

        key_length = past_length
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

        if replace_position.any():
            block_end_index = (
                replace_position
                .nonzero(as_tuple=True)[1]
                .max()
                + 1
            )
        else:
            block_end_index = key_length

        q_range = torch.arange(
            block_end_index - query_len,
            block_end_index,
            device=q.device,
            dtype=torch.long,
        )

        q_positions = q_range.unsqueeze(0).expand(
            batch_size,
            -1,
        )

        return (
            q_positions,
            new_k_positions,
            key_length,
        )

    q_positions, new_k_positions, key_length = measure(
        record,
        layer_id,
        "position_build",
        build_positions,
    )

    rope_table_length = max(
        int(self.config.max_sequence_length),
        int(key_length),
    )

    # Rotated-K 的核心：只旋转当前 Q 和当前新 K。
    q_rotated = measure(
        record,
        layer_id,
        "q_rope",
        lambda: apply_rope_at_positions(
            self,
            q,
            q_positions,
            rope_table_length,
        ),
    )

    new_k_rotated = measure(
        record,
        layer_id,
        "new_k_rope",
        lambda: apply_rope_at_positions(
            self,
            new_k,
            new_k_positions,
            rope_table_length,
        ),
    )

    # ========================================================
    # Rotated-K Cache update
    # ========================================================

    def update_cache():
        if layer_past is None:
            return new_k_rotated, new_v

        past_k_rotated, past_value = layer_past

        if replace_position is None:
            full_k = torch.cat(
                (past_k_rotated, new_k_rotated),
                dim=-2,
            )

            full_v = torch.cat(
                (past_value, new_v),
                dim=-2,
            )

            return full_k, full_v

        full_k = past_k_rotated
        full_v = past_value

        for batch_index in range(batch_size):
            indices = new_k_positions[batch_index]

            full_k[
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
            ] = new_v[
                batch_index,
                :,
                :,
                :,
            ]

        return full_k, full_v

    full_k_rotated, full_v = measure(
        record,
        layer_id,
        "cache_update",
        update_cache,
    )

    present = (
        (full_k_rotated, full_v)
        if use_cache
        else None
    )

    # ========================================================
    # SDPA
    # ========================================================

    attention = measure(
        record,
        layer_id,
        "sdpa",
        lambda: self._scaled_dot_product_attention(
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

    # ========================================================
    # SwiGLU MLP
    # ========================================================

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
    "position_build",
    "q_rope",
    "new_k_rope",
    "cache_update",
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
    print("Rotated-K component profile")
    print("=" * 100)
    print(
        "各阶段前后包含 MPS synchronize；"
        "用于比较占比和排序，不代表无扰动绝对延迟。"
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
            "past_length_at_layer0": (
                record["past_length_at_layer0"]
            ),
            "replace_count": record["replace_count"],
            "profiled_phase_total_seconds": total,
            "phases_seconds": phases,
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
            total_ms = seconds * 1000
            per_layer_ms = total_ms / NUM_LAYERS
            share = seconds / total * 100 if total else 0.0

            print(
                f"{phase:<26}"
                f"{total_ms:>14.3f}"
                f"{per_layer_ms:>16.3f}"
                f"{share:>11.2f}%"
            )

        groups = {
            "MLP": (
                phases.get("ff_norm", 0.0)
                + phases.get("gate_up_projections", 0.0)
                + phases.get("silu_multiply", 0.0)
                + phases.get("ff_out_residual", 0.0)
            ),
            "Attention projections": (
                phases.get("attn_norm", 0.0)
                + phases.get("qkv_projections", 0.0)
            ),
            "New Q/K RoPE": (
                phases.get("q_rope", 0.0)
                + phases.get("new_k_rope", 0.0)
            ),
            "Cache update": phases.get(
                "cache_update",
                0.0,
            ),
            "SDPA": phases.get("sdpa", 0.0),
            "Attention output": phases.get(
                "attn_out_residual",
                0.0,
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
