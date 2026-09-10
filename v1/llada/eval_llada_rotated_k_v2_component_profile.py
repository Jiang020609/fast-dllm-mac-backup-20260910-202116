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
# 加载已经通过 A/B 的 Rotated-K v2 实现，但不启动其评测。
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

implementation_source = source.split(marker, 1)[0]

namespace: dict[str, Any] = {
    "__name__": "rotated_k_v2_implementation",
}

exec(
    compile(
        implementation_source,
        str(V2_WRAPPER),
        "exec",
    ),
    namespace,
)

build_position_context = namespace["build_position_context"]
apply_cached_rope = namespace["apply_cached_rope"]
rotated_k_v2_attention = namespace["rotated_k_v2_attention"]

# 非目标 forward 使用真实 v2 attention。
modeling.LLaDABlock.attention = rotated_k_v2_attention

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
            / "rotated_k_v2_component_profile.json"
        ),
    )
)

STATE: dict[str, Any] = {
    "forward_index": -1,
    "records": {},
}

PROFILE_POSITION_CONTEXT: dict[str, Any] | None = None


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
    attention_bias: Optional[torch.Tensor] = None,
    layer_past: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None,
    use_cache: bool = False,
    replace_position: Optional[torch.Tensor] = None,
):
    global PROFILE_POSITION_CONTEXT

    if self.layer_id == 0:
        STATE["forward_index"] += 1
        PROFILE_POSITION_CONTEXT = None

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
    # Attention projections
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
    # 只在第 0 层构建一次 position metadata 和 RoPE slices。
    # ========================================================

    if self.layer_id == 0:
        PROFILE_POSITION_CONTEXT = measure(
            record,
            layer_id,
            "position_context_build",
            lambda: build_position_context(
                self,
                q,
                layer_past,
                replace_position,
            ),
        )

    if PROFILE_POSITION_CONTEXT is None:
        raise RuntimeError(
            "Position context was not initialized"
        )

    if PROFILE_POSITION_CONTEXT["query_len"] != query_len:
        raise RuntimeError(
            "Query length changed inside one model forward"
        )

    # ========================================================
    # 只旋转当前 Q 和当前新 K。
    # ========================================================

    q_rotated = measure(
        record,
        layer_id,
        "q_rope",
        lambda: apply_cached_rope(
            self,
            q,
            PROFILE_POSITION_CONTEXT["q_sin"],
            PROFILE_POSITION_CONTEXT["q_cos"],
        ),
    )

    new_k_rotated = measure(
        record,
        layer_id,
        "new_k_rope",
        lambda: apply_cached_rope(
            self,
            new_k,
            PROFILE_POSITION_CONTEXT["k_sin"],
            PROFILE_POSITION_CONTEXT["k_cos"],
        ),
    )

    # ========================================================
    # Rotated-K / V Cache update
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

        positions = PROFILE_POSITION_CONTEXT[
            "new_k_positions"
        ]

        for batch_index in range(batch_size):
            indices = positions[batch_index]

            full_k[
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
            ] = new_v[batch_index]

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
    "position_context_build",
    "attn_norm",
    "qkv_projections",
    "head_reshape",
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
    print("Rotated-K v2 component profile")
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
            f"{'Phase':<28}"
            f"{'Total ms':>14}"
            f"{'Share':>12}"
        )

        for phase in sorted(
            PHASE_ORDER,
            key=lambda name: phases.get(name, 0.0),
            reverse=True,
        ):
            seconds = phases.get(phase, 0.0)
            total_ms = seconds * 1000
            share = seconds / total * 100 if total else 0.0

            print(
                f"{phase:<28}"
                f"{total_ms:>14.3f}"
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
            "Position metadata": phases.get(
                "position_context_build",
                0.0,
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
